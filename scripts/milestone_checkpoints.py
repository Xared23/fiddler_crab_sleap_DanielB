"""Additive, immutable epoch-end snapshots. No optimizer/training policy changes.

The original run/preflight manifests remain baseline records. An explicit,
validated extension receipt binds those exact baseline hashes to the actual
worker/helper hashes; it does NOT turn off code/configuration integrity checks.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import time

from pipeline_common import ROOT, atomic_json, read_json, sha256

LOG = logging.getLogger(__name__)
COMPLETED_EPOCHS = (75, 100)


def validate_extension(receipt: dict, preflight: dict, actual: dict, run_id: str) -> None:
    expected_keys = set(preflight["implementation"]) | {"scripts/milestone_checkpoints.py"}
    if (receipt.get("schema_version") != 1
            or receipt.get("run_id") != run_id
            or run_id != "v007_19f"
            or receipt.get("roles") != ["centroid"]
            or receipt.get("completed_epochs") != list(COMPLETED_EPOCHS)
            or receipt.get("baseline_implementation") != preflight["implementation"]
            or receipt.get("implementation") != actual
            or set(actual) != expected_keys
            or receipt.get("run_config_sha256") != preflight["run_config_sha256"]
            or receipt.get("dataset_sha256") != preflight["dataset_sha256"]
            or receipt.get("tests_passed") is not True):
        raise RuntimeError("Checkpoint extension receipt does not match this exact baseline/code/data/run")
    # Only the worker and new snapshot module may change. The original training,
    # dataset, epoch-timer and Slurm-chain helpers must remain byte-identical.
    for name, digest in preflight["implementation"].items():
        if name != "scripts/train_versioned.py" and actual.get(name) != digest:
            raise RuntimeError(f"Unapproved implementation change: {name}")


def load_extension(preflight: dict, actual: dict, run_id: str) -> dict:
    path = ROOT / "checks" / run_id / "milestones_75_100" / "APPROVED.json"
    receipt = read_json(path)
    validate_extension(receipt, preflight, actual, run_id)
    report = ROOT / "checks" / run_id / "milestones_75_100" / "tests.json"
    if sha256(report) != receipt["test_report_sha256"] or read_json(report).get("passed") is not True:
        raise RuntimeError("Checkpoint extension test evidence changed or did not pass")
    return receipt


def timing_summary(log_path: Path | None, fit_started: float) -> dict:
    previous_loop_seconds = 0.0
    first_setup = None
    if log_path is not None and log_path.is_file():
        with log_path.open(errors="replace") as stream:
            for line in stream:
                match = re.search(r"Finished training loop\. \[([0-9.]+) min\]", line)
                if match:
                    previous_loop_seconds += 60 * float(match[1])
                if first_setup is None:
                    match = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) \| Creating train-val split", line)
                    if match:
                        first_setup = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").timestamp()
    current_fit_seconds = time.monotonic() - fit_started
    return {
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "current_hop_fit_seconds": current_fit_seconds,
        "previous_completed_hop_loop_seconds_approx": previous_loop_seconds if log_path else None,
        "cumulative_fit_seconds_approx": previous_loop_seconds + current_fit_seconds if log_path else None,
        "wall_seconds_since_first_logged_setup": None if first_setup is None else time.time() - first_setup,
        "definition": "Fit time includes validation/checkpoint overhead, excludes most startup and queue waits. Prior hop durations are rounded in the original logs. Wall time includes intervening queue waits; neither field is GPU kernel time.",
    }


def save_snapshot(trainer, run_dir: Path, completed_epoch: int, timing: dict) -> Path:
    """Publish an inference-ready model directory only after full validation."""
    import torch
    parent = run_dir / "milestones"
    parent.mkdir(exist_ok=True)
    destination = parent / f"epoch_{completed_epoch:04d}"
    if destination.exists():
        record = read_json(destination / "snapshot.json")
        if (record["completed_epochs"] != completed_epoch
                or sha256(destination / "best.ckpt") != record["checkpoint_sha256"]
                or sha256(destination / "training_config.yaml") != record["training_config_sha256"]):
            raise RuntimeError(f"Existing milestone failed integrity checks: {destination}")
        return destination
    temporary = parent / f".epoch_{completed_epoch:04d}.tmp"
    if temporary.exists():
        # A previous interrupted unpublished attempt; never touch a final snapshot.
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        checkpoint = temporary / "best.ckpt"
        trainer.save_checkpoint(str(checkpoint))
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (int(saved["epoch"]) != completed_epoch - 1
                or int(saved["global_step"]) != int(trainer.global_step)
                or not saved.get("state_dict") or not saved.get("optimizer_states")):
            raise RuntimeError("Milestone checkpoint epoch/step/state validation failed")
        del saved
        config = temporary / "training_config.yaml"
        shutil.copy2(run_dir / "training_config.yaml", config)
        for name in ("manifest.json", "checkpoint_extension_active.json", "training_log.csv"):
            source = run_dir / name
            if source.is_file():
                shutil.copy2(source, temporary / name)
        val = trainer.callback_metrics.get("val/loss")
        val = None if val is None else float(val)
        if val is not None and not math.isfinite(val):
            val = None
        record = {
            "schema_version": 1,
            "kind": "epoch_end_weights_NOT_best_so_far",
            "completed_epochs": completed_epoch,
            "checkpoint_epoch_zero_based": int(trainer.current_epoch),
            "global_step": int(trainer.global_step),
            "validation_loss": val,
            "learning_rates": [float(opt.param_groups[0]["lr"]) for opt in trainer.optimizers],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "checkpoint_sha256": sha256(checkpoint),
            "training_config_sha256": sha256(config),
            "timing": timing,
            "note": "best.ckpt is the SLEAP inference filename, but contains this epoch's weights, not the validation-best weights from an earlier epoch. CSV copied inside the callback may not yet include this epoch; validation_loss above is captured directly.",
        }
        atomic_json(temporary / "snapshot.json", record)
        temporary.replace(destination)
        print(f"MILESTONE_SAVED completed_epochs={completed_epoch} step={trainer.global_step} path={destination}", flush=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination


def make_milestone_callback(run_dir: Path, log_path: Path | None = None, targets=COMPLETED_EPOCHS):
    from lightning.pytorch.callbacks import Callback

    class MilestoneSnapshots(Callback):
        def __init__(self):
            self.fit_started = time.monotonic()

        def on_fit_start(self, trainer, module):
            self.fit_started = time.monotonic()
            print(f"Milestone snapshots enabled after completed epochs {list(targets)}", flush=True)

        def on_train_epoch_end(self, trainer, module):
            completed = int(trainer.current_epoch) + 1
            if completed not in targets:
                return
            try:
                save_snapshot(trainer, run_dir, completed, timing_summary(log_path, self.fit_started))
            except Exception as error:
                # Auxiliary artifact failures must not kill otherwise healthy
                # production training. Make the failure explicit, never silently
                # substitute a later epoch under the requested epoch's name.
                LOG.exception("MILESTONE_SAVE_FAILED completed_epochs=%s", completed)
                try:
                    atomic_json(run_dir / "milestones" / f"epoch_{completed:04d}.FAILED.json", {
                        "completed_epochs": completed, "error": repr(error),
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "training_continues": True,
                    })
                except Exception:
                    LOG.exception("Could not write milestone failure report")
    return MilestoneSnapshots()
