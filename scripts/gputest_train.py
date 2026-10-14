#!/usr/bin/env python3
"""Train a sleap-nn model in 1-hour gputest hops.

Resumes from last.ckpt / interrupt.ckpt in a stable run directory, saves on
SIGUSR1 (Slurm preemption warning), and exits 75 when the hop should be
requeued. Exit 0 means training actually finished (max epochs or early stop).

Does not run sleap-nn's post-training eval (that step has OOMed on 8K frames).
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from omegaconf import OmegaConf

# Exit code the Slurm wrapper treats as "save was OK, submit the next hop".
EXIT_REQUEUE = 75

_INTERRUPTED = False
_TRAINER = None
_RUN_DIR: Optional[Path] = None


def _disable_lightning_scontrol_requeue() -> None:
    """Stop Lightning from calling ``scontrol requeue`` (the Slurm wrapper does that)."""
    from lightning.fabric.plugins.environments.slurm import SLURMEnvironment

    orig_init = SLURMEnvironment.__init__

    def _init(self, auto_requeue: bool = True, requeue_signal=None):  # noqa: ARG001
        orig_init(self, auto_requeue=False, requeue_signal=requeue_signal)

    SLURMEnvironment.__init__ = _init  # type: ignore[method-assign]


def _newest_ckpt(run_dir: Path) -> Optional[Path]:
    candidates = list(run_dir.glob("last*.ckpt")) + [
        run_dir / "interrupt.ckpt",
        run_dir / "best.ckpt",
    ]
    existing = [p for p in candidates if p.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _hide_best_ckpt(run_dir: Path) -> Optional[Path]:
    """sleap-nn suffixes run_name whenever best.ckpt exists; hide it during setup."""
    best = run_dir / "best.ckpt"
    hidden = run_dir / ".best.ckpt.hidden"
    if best.is_file():
        if hidden.exists():
            hidden.unlink()
        best.rename(hidden)
        logger.info(f"Temporarily hid {best} so sleap-nn reuses this run directory")
        return hidden
    return None


def _restore_best_ckpt(hidden: Optional[Path], run_dir: Path) -> None:
    if hidden is None or not hidden.exists():
        return
    best = run_dir / "best.ckpt"
    if best.exists():
        hidden.unlink()
        return
    hidden.rename(best)
    logger.info(f"Restored {best}")


def _install_stop_handler() -> None:
    def _handle(signum, _frame) -> None:
        global _INTERRUPTED
        _INTERRUPTED = True
        name = signal.Signals(signum).name
        logger.warning(
            f"Received {name}; saving checkpoint and stopping after this batch"
        )
        trainer = _TRAINER
        if trainer is None or _RUN_DIR is None:
            return
        ckpt = _RUN_DIR / "last.ckpt"
        try:
            trainer.save_checkpoint(ckpt.as_posix())
            logger.info(f"Saved preemption checkpoint: {ckpt}")
        except Exception:
            logger.exception("Failed to save preemption checkpoint")
        trainer.should_stop = True

    signal.signal(signal.SIGUSR1, _handle)
    signal.signal(signal.SIGTERM, _handle)


def _patch_trainer_for_hop(run_dir: Path) -> None:
    """Stash the Lightning Trainer, add mid-epoch saves, handle preemption."""
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

    orig_init = L.Trainer.__init__

    class _StopIfInterrupted(Callback):
        def on_fit_start(self, trainer, _pl_module) -> None:
            if _INTERRUPTED:
                trainer.should_stop = True

        def on_train_batch_start(self, trainer, pl_module, _batch, _batch_idx) -> None:
            _ensure_epoch_timers(pl_module)
            if _INTERRUPTED:
                trainer.should_stop = True

    def _init(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        for cb in callbacks:
            if isinstance(cb, EarlyStopping):
                # Mid-epoch resume finishes the train epoch before validation, so
                # val/loss is missing and strict=True crashes. Check after val instead.
                cb.strict = False
                cb._check_on_train_epoch_end = False
        callbacks.append(_StopIfInterrupted())
        callbacks.append(
            ModelCheckpoint(
                dirpath=run_dir.as_posix(),
                filename="interrupt",
                every_n_train_steps=10,
                save_top_k=1,
                monitor=None,
                enable_version_counter=False,
            )
        )
        kwargs["callbacks"] = callbacks
        orig_init(self, *args, **kwargs)
        global _TRAINER
        _TRAINER = self

    L.Trainer.__init__ = _init  # type: ignore[method-assign]


def _ensure_epoch_timers(module) -> None:
    """sleap-nn sets these in on_train_epoch_start, which is skipped on mid-epoch resume."""
    now = time.time()
    if not hasattr(module, "train_start_time"):
        module.train_start_time = now
    if not hasattr(module, "val_start_time"):
        module.val_start_time = now
    if not hasattr(module, "_epoch_loss_sum"):
        module._epoch_loss_sum = 0.0
    if not hasattr(module, "_epoch_loss_count"):
        module._epoch_loss_count = 0
    if not hasattr(module, "_epoch_sample_count"):
        module._epoch_sample_count = 0


def _patch_epoch_timers() -> None:
    from sleap_nn.training.lightning_modules import LightningModel

    orig_train_end = LightningModel.on_train_epoch_end
    orig_val_end = LightningModel.on_validation_epoch_end

    def on_train_epoch_end(self):
        _ensure_epoch_timers(self)
        return orig_train_end(self)

    def on_validation_epoch_end(self):
        _ensure_epoch_timers(self)
        return orig_val_end(self)

    LightningModel.on_train_epoch_end = on_train_epoch_end  # type: ignore[method-assign]
    LightningModel.on_validation_epoch_end = on_validation_epoch_end  # type: ignore[method-assign]


def _keep_csv_history() -> None:
    """Do not wipe training_log.csv at the start of each hop."""
    from sleap_nn.training.callbacks import CSVLoggerCallback

    orig = CSVLoggerCallback._init_file

    def _init_file(self):
        if Path(self.filepath).is_file():
            self.initialized = True
            return
        orig(self)

    CSVLoggerCallback._init_file = _init_file  # type: ignore[method-assign]


def _ignore_stop_signals() -> None:
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="sleap-nn YAML")
    parser.add_argument(
        "--max-hops",
        type=int,
        default=80,
        help="Safety cap on gputest hops (also enforced in the Slurm wrapper).",
    )
    args, rest = parser.parse_known_args(argv)
    return args, rest


def main(argv: Optional[list[str]] = None) -> int:
    global _RUN_DIR

    args, overrides = _parse_args(argv or sys.argv[1:])
    cfg = OmegaConf.load(args.config)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    ckpt_dir = Path(cfg.trainer_config.ckpt_dir)
    run_name = str(cfg.trainer_config.run_name)
    run_dir = ckpt_dir / run_name
    _RUN_DIR = run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    complete_marker = run_dir / "TRAINING_COMPLETE"
    if complete_marker.exists():
        logger.info(f"Already complete: {complete_marker}")
        return 0

    hops_file = run_dir / ".gputest_hops"
    hop = int(hops_file.read_text().strip()) if hops_file.exists() else 0
    if hop > args.max_hops:
        logger.error(f"Reached max gputest hops ({args.max_hops}); not starting")
        return 1

    resume = _newest_ckpt(run_dir)
    if resume is not None:
        cfg.trainer_config.resume_ckpt_path = resume.as_posix()
        logger.info(f"Resuming from {resume}")
    else:
        cfg.trainer_config.resume_ckpt_path = None
        logger.info("No checkpoint found; starting a new run")

    cfg.trainer_config.model_ckpt.save_last = True
    cfg.trainer_config.save_ckpt = True
    if cfg.trainer_config.seed is None:
        cfg.trainer_config.seed = 42

    _disable_lightning_scontrol_requeue()
    _patch_trainer_for_hop(run_dir)
    _install_stop_handler()

    hidden = _hide_best_ckpt(run_dir) if resume is not None else None
    from sleap_nn.training.model_trainer import ModelTrainer

    _patch_epoch_timers()
    _keep_csv_history()
    try:
        model_trainer = ModelTrainer.get_model_trainer_from_config(cfg)
    finally:
        _restore_best_ckpt(hidden, run_dir)

    actual_run = (
        Path(model_trainer.config.trainer_config.ckpt_dir)
        / model_trainer.config.trainer_config.run_name
    )
    if actual_run.resolve() != run_dir.resolve():
        logger.error(
            "sleap-nn changed run_name from "
            f"{run_dir.name!r} to {actual_run.name!r}. "
            "Refusing to train so checkpoints stay in one directory."
        )
        return 1

    logger.info(f"gputest hop starting in {run_dir}")
    if _INTERRUPTED:
        logger.warning("Preemption signal during setup; requeue")
        return EXIT_REQUEUE
    try:
        model_trainer.train()
    except Exception:
        logger.exception("Training raised")
        return 1
    finally:
        _ignore_stop_signals()

    if _INTERRUPTED:
        logger.warning("Hop ended on preemption signal; requeue")
        return EXIT_REQUEUE

    pl_trainer = model_trainer.trainer
    epoch = int(pl_trainer.current_epoch) if pl_trainer is not None else -1
    complete_marker.write_text(f"epoch={epoch}\n")
    logger.info(f"Training finished at epoch {epoch}; wrote {complete_marker}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
