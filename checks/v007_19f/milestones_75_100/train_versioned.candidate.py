#!/usr/bin/env python3
"""Versioned SLEAP training worker. Exit 75 alone authorizes a Slurm requeue.

Runs from fresh weights in a new run directory; subsequent hops resume only that
run. Signal handlers request a stop; checkpoint I/O occurs at safe batch hooks,
not from an asynchronous signal handler. Full-frame post-training eval is off.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import sys
import traceback

from omegaconf import OmegaConf

from pipeline_common import (ROOT, EXIT_REQUEUE, atomic_json, dataset_manifest,
    exclusive_lock, fingerprint, hop_progress, hop_start, load_run, model_dir,
    pin_manifest, project_path, read_json, sha256, versions)

STOP_REQUESTED = False


class HopStop(Exception):
    """Clean interruption after a safe checkpoint (or before training started)."""


def request_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received {signal.Signals(signum).name}; requesting safe batch-boundary stop", flush=True)


def implementation_hashes() -> dict:
    files = ["scripts/train_versioned.py", "scripts/pipeline_common.py",
             "scripts/gputest_train.py", "scripts/train_versioned_chain.sh",
             "scripts/milestone_checkpoints.py"]
    return {name: sha256(Path(__file__).resolve() if name == "scripts/train_versioned.py" else ROOT / name) for name in files}


def checkpoint_summary(path: Path) -> dict:
    import torch
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not checkpoint.get("state_dict") or not checkpoint.get("optimizer_states"):
        raise RuntimeError(f"Not a full training checkpoint: {path}")
    step = int(checkpoint["global_step"])
    epoch = int(checkpoint["epoch"])
    if step < 0 or epoch < 0:
        raise RuntimeError(f"Invalid checkpoint progress: {path}")
    return {"global_step": step, "epoch": epoch}


def select_checkpoint(directory: Path):
    paths = [directory / name for name in ("interrupt.ckpt", "interrupt.previous.ckpt", "best.ckpt")]
    paths += sorted(directory.glob("last*.ckpt"))
    valid = []
    for path in paths:
        if not path.is_file():
            continue
        try:
            summary = checkpoint_summary(path)
            valid.append((summary["global_step"], summary["epoch"], path.stat().st_mtime_ns, path, summary))
        except Exception as error:
            print(f"Checkpoint rejected: {path}: {error}", flush=True)
    if not valid:
        if any(path.is_file() for path in paths):
            raise RuntimeError("Existing checkpoints are all unreadable; refusing silent fresh restart")
        return None, {"global_step": 0, "epoch": 0}
    selected = max(valid, key=lambda item: item[:3])
    return selected[3], selected[4]


def save_interrupt(trainer, directory: Path) -> None:
    target = directory / "interrupt.ckpt"
    temp = directory / ".interrupt.tmp.ckpt"
    trainer.save_checkpoint(str(temp))
    summary = checkpoint_summary(temp)
    if summary["global_step"] != int(trainer.global_step):
        raise RuntimeError("Checkpoint did not capture the expected training step")
    # Keep a second independently named, previously validated checkpoint. No old
    # checkpoint is removed until its replacement is already readable.
    if target.exists():
        previous_temp = directory / ".interrupt.previous.tmp.ckpt"
        previous_temp.unlink(missing_ok=True)
        os.link(target, previous_temp)
        previous_temp.replace(directory / "interrupt.previous.ckpt")
    temp.replace(target)
    atomic_json(directory / "checkpoint_progress.json", summary)
    print(f"Checkpoint saved: epoch={summary['epoch']} global_step={summary['global_step']}", flush=True)


def patch_lightning(directory: Path, every_steps: int, smoke: bool, stop_after_steps, milestone_callback=None):
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
    from gputest_train import _ensure_epoch_timers

    original_init = L.Trainer.__init__

    class SafeHop(Callback):
        def on_fit_start(self, trainer, module):
            if STOP_REQUESTED:
                raise HopStop("Stop requested during setup")

        def on_train_batch_start(self, trainer, module, batch, batch_idx):
            _ensure_epoch_timers(module)

        def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
            import torch
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if isinstance(loss, torch.Tensor) and not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Non-finite training loss; not requeuing")
            if stop_after_steps is not None and trainer.global_step >= stop_after_steps:
                # Exercise the same handler as Slurm, only in the isolated smoke run.
                signal.raise_signal(signal.SIGUSR1)
            if STOP_REQUESTED or trainer.global_step % every_steps == 0:
                save_interrupt(trainer, directory)
            if STOP_REQUESTED:
                raise HopStop("Saved checkpoint at training batch boundary")

        def on_validation_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx=0):
            if STOP_REQUESTED:
                if trainer.global_step > 0:
                    save_interrupt(trainer, directory)
                raise HopStop("Stop requested during validation")

        def on_validation_end(self, trainer, module):
            value = trainer.callback_metrics.get("val/loss")
            if value is not None and not math.isfinite(float(value)):
                raise RuntimeError("Non-finite validation loss; not requeuing")

    def init(self, *args, **kwargs):
        callbacks = list(kwargs.get("callbacks") or [])
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                # Always evaluate after validation, including after a mid-epoch hop.
                callback._check_on_train_epoch_end = False
                callback.strict = True
            if isinstance(callback, ModelCheckpoint):
                callback.enable_version_counter = False
        callbacks.append(SafeHop())
        if milestone_callback is not None:
            callbacks.append(milestone_callback)
        kwargs["callbacks"] = callbacks
        if smoke:
            kwargs["limit_val_batches"] = 2
            kwargs["num_sanity_val_steps"] = 0
        original_init(self, *args, **kwargs)
    L.Trainer.__init__ = init


def run(args) -> int:
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_cfg = load_run(args.config)
    dataset = dataset_manifest(run_cfg)
    video_info = dataset["manifest"]["video"]
    stat = project_path(video_info["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (video_info["bytes"], video_info["mtime_ns"]):
        raise RuntimeError("Source video changed after dataset preparation")
    config_path = project_path(dataset["manifest"]["resolved_configs"][args.role])
    config = OmegaConf.load(config_path)
    training = run_cfg["training"]
    actual_code_hashes = implementation_hashes()
    code_hashes = actual_code_hashes
    checkpoint_extension = None
    if not args.smoke:
        preflight = read_json(ROOT / "checks" / run_cfg["run_id"] / "PREFLIGHT_COMPLETE.json")
        if (preflight["run_config_sha256"] != sha256(project_path(args.config))
                or preflight["dataset_sha256"] != fingerprint(dataset)):
            raise RuntimeError("Preflight does not match this configuration/dataset")
        if preflight["implementation"] != actual_code_hashes:
            from milestone_checkpoints import load_extension
            checkpoint_extension = load_extension(preflight, actual_code_hashes, run_cfg["run_id"])
            # manifest.json remains the immutable ORIGINAL training baseline.
            # Actual runtime hashes and the tested extension are recorded below;
            # no unapproved code difference is accepted by load_extension.
            code_hashes = preflight["implementation"]
    directory = model_dir(run_cfg, args.role)
    if args.smoke:
        directory = ROOT / "checks" / run_cfg["run_id"] / "training_smoke" / args.role
        config.trainer_config.ckpt_dir = str(directory.parent)
        config.trainer_config.run_name = directory.name
        config.trainer_config.max_epochs = 2
        config.trainer_config.train_steps_per_epoch = 2
        config.trainer_config.min_train_steps_per_epoch = 2
    elif args.stop_after_steps is not None:
        raise ValueError("--stop-after-steps is restricted to smoke checks")

    with exclusive_lock(directory):
        manifest = {"run_id": run_cfg["run_id"], "role": args.role, "smoke": args.smoke,
            "dataset_sha256": fingerprint(dataset), "config_sha256": sha256(config_path),
            "config": OmegaConf.to_container(config, resolve=True), "versions": versions(),
            "implementation": code_hashes, "training_policy": training}
        manifest_hash = pin_manifest(directory, manifest)
        if checkpoint_extension is not None and args.role == "centroid":
            atomic_json(directory / "checkpoint_extension_active.json", {
                "baseline_manifest_sha256": manifest_hash,
                "actual_implementation": actual_code_hashes,
                "extension": checkpoint_extension,
                "activation_job_id": os.environ.get("SLURM_JOB_ID"),
            })
        complete = directory / "TRAINING_COMPLETE.json"
        if complete.exists():
            marker = read_json(complete)
            if marker["manifest_sha256"] != manifest_hash or sha256(directory / "best.ckpt") != marker["best_sha256"]:
                raise RuntimeError("Stale/invalid training completion marker")
            checkpoint_summary(directory / "best.ckpt")
            print(f"Already complete and validated: {directory}", flush=True)
            return 0
        hop = hop_start(directory, training["max_hops"])
        resume, initial = select_checkpoint(directory)
        print(f"Hop {hop}: role={args.role} directory={directory} resume={resume} progress={initial}", flush=True)
        config.trainer_config.resume_ckpt_path = None if resume is None else str(resume)

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable inside the GPU allocation")
        print(f"GPU={torch.cuda.get_device_name(0)} torch={torch.__version__} CUDA={torch.version.cuda}", flush=True)
        from gputest_train import (_disable_lightning_scontrol_requeue, _patch_epoch_timers, _keep_csv_history)
        _disable_lightning_scontrol_requeue()
        _patch_epoch_timers()
        _keep_csv_history()
        milestone_callback = None
        if checkpoint_extension is not None and args.role == "centroid":
            from milestone_checkpoints import make_milestone_callback
            job_id = os.environ.get("SLURM_JOB_ID")
            log_path = None if job_id is None else ROOT / "logs" / run_cfg["run_id"] / f"centroid_{job_id}.out"
            milestone_callback = make_milestone_callback(directory, log_path)
        patch_lightning(directory, training["checkpoint_every_steps"], args.smoke, args.stop_after_steps, milestone_callback)
        from sleap_nn.training.model_trainer import ModelTrainer

        class StableRunTrainer(ModelTrainer):
            def _setup_ckpt_path(self):
                # Explicit new run directory + manifest + lock replace the old
                # temporary best.ckpt hiding workaround. Never rename best.ckpt.
                actual = Path(self.config.trainer_config.ckpt_dir) / self.config.trainer_config.run_name
                if actual.resolve() != directory.resolve():
                    raise RuntimeError("Training run path changed")
                if self.config.data_config.data_pipeline_fw != "torch_dataset_cache_img_memory":
                    raise RuntimeError("This run is configured for in-memory image caching")

        worker = StableRunTrainer.get_model_trainer_from_config(config)
        interrupted = False
        try:
            if STOP_REQUESTED:
                raise HopStop("Stop requested during model setup")
            worker.train()
        except HopStop as error:
            interrupted = True
            print(str(error), flush=True)
        # All other exceptions propagate as fatal errors, even after a stop signal.
        checkpoint, progress = select_checkpoint(directory)
        hop_progress(directory, progress["global_step"], training["max_no_progress_hops"])
        if interrupted:
            return EXIT_REQUEUE
        trainer = worker.trainer
        from lightning.pytorch.callbacks import EarlyStopping
        early_stopped = any(isinstance(cb, EarlyStopping) and cb.stopped_epoch > 0 for cb in trainer.callbacks)
        reached_limit = trainer.current_epoch >= config.trainer_config.max_epochs
        if not (reached_limit or early_stopped):
            if STOP_REQUESTED:
                return EXIT_REQUEUE
            raise RuntimeError("Training returned without reaching the epoch limit or early stopping")
        best = directory / "best.ckpt"
        best_summary = checkpoint_summary(best)
        atomic_json(complete, {"manifest_sha256": manifest_hash, "best_sha256": sha256(best),
            "epoch": int(trainer.current_epoch), "global_step": int(trainer.global_step),
            "best_progress": best_summary, "reason": "max_epochs" if reached_limit else "early_stopping"})
        print(f"Training complete: {complete}", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("centroid", "centered"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
