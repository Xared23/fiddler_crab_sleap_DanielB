#!/usr/bin/env python3
"""Finite Slurm-only extension build/test or verification after a reviewed edit.

No watcher, polling loop, submission, cancellation, requeue, or training restart.
This script never modifies the production worker. The tested patch is applied
with the project edit tool during a confirmed safe active-hop window.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from pipeline_common import (ROOT, atomic_json, dataset_manifest, fingerprint,
    load_run, read_json, sha256)
from milestone_checkpoints import validate_extension

CHECKS = ROOT / "checks" / "v007_19f" / "milestones_75_100"
WORKER = ROOT / "scripts" / "train_versioned.py"
CANDIDATE = CHECKS / "train_versioned.candidate.py"
RUN_DIR = ROOT / "models" / "v007_19f" / "centroid"


def baseline():
    preflight = read_json(ROOT / "checks/v007_19f/PREFLIGHT_COMPLETE.json")
    run = load_run("configs/v007_19f/run.yaml")
    if (sha256(ROOT / "configs/v007_19f/run.yaml") != preflight["run_config_sha256"]
            or fingerprint(dataset_manifest(run)) != preflight["dataset_sha256"]):
        raise RuntimeError("Training data/configuration no longer matches the original preflight")
    for name, digest in preflight["implementation"].items():
        if name != "scripts/train_versioned.py" and sha256(ROOT / name) != digest:
            raise RuntimeError(f"Original helper changed: {name}")
    return preflight


def actual_hashes(worker: Path):
    names = list(read_json(ROOT / "checks/v007_19f/PREFLIGHT_COMPLETE.json")["implementation"])
    names.append("scripts/milestone_checkpoints.py")
    return {name: sha256(worker if name == "scripts/train_versioned.py" else ROOT / name) for name in names}


def build_and_test():
    preflight = baseline()
    if sha256(WORKER) != preflight["implementation"]["scripts/train_versioned.py"]:
        raise RuntimeError("Worker differs from baseline; refusing to build an unreviewed patch")
    original = WORKER.read_text()
    replacements = read_json(CHECKS / "worker_patch.json")
    for replacement in replacements:
        if original.count(replacement["old"]) != 1:
            raise RuntimeError("Patch context is missing or ambiguous")
    candidate = original
    for replacement in replacements:
        candidate = candidate.replace(replacement["old"], replacement["new"], 1)
    compile(candidate, str(CANDIDATE), "exec")
    backup = CHECKS / "train_versioned.original.py"
    if backup.exists() and sha256(backup) != sha256(WORKER):
        raise RuntimeError("Baseline source backup mismatch")
    if not backup.exists():
        shutil.copy2(WORKER, backup)
    CANDIDATE.write_text(candidate)
    implementation = actual_hashes(CANDIDATE)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests",
                             "-p", "test_milestone_checkpoints.py", "-v"],
                            cwd=ROOT, env=env, capture_output=True, text=True)
    (CHECKS / "tests.stdout.txt").write_text(result.stdout)
    (CHECKS / "tests.stderr.txt").write_text(result.stderr)
    print(result.stdout, flush=True)
    print(result.stderr, file=sys.stderr, flush=True)
    if actual_hashes(CANDIDATE) != implementation:
        raise RuntimeError("Code changed during checkpoint extension tests")
    report = {"passed": result.returncode == 0, "exit_code": result.returncode,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "implementation": implementation,
              "test_source_sha256": sha256(ROOT / "tests/test_milestone_checkpoints.py"),
              "checks": ["75/100 completed-epoch indexing", "immutable and atomic snapshot outputs",
                         "RNG preservation", "snapshot failure leaves training running",
                         "integrity receipt rejects drift", "real Lightning CPU save hooks",
                         "identical toy-model optimization with and without snapshots"]}
    atomic_json(CHECKS / "tests.json", report)
    if result.returncode != 0:
        raise RuntimeError("Extension tests failed; production worker remains unchanged")
    receipt = {"schema_version": 1, "run_id": "v007_19f", "roles": ["centroid"],
               "completed_epochs": [75, 100], "baseline_implementation": preflight["implementation"],
               "implementation": implementation, "run_config_sha256": preflight["run_config_sha256"],
               "dataset_sha256": preflight["dataset_sha256"], "tests_passed": True,
               "test_report_sha256": sha256(CHECKS / "tests.json"),
               "approved_scope": "User-requested additional epoch-end checkpoint retention only; no training hyperparameter or scheduler changes."}
    validate_extension(receipt, preflight, implementation, "v007_19f")
    atomic_json(CHECKS / "APPROVED.json", receipt)
    print("TESTED_AND_READY: production worker not modified by the test job.", flush=True)


def verify_install():
    preflight = baseline()
    receipt = read_json(CHECKS / "APPROVED.json")
    validate_extension(receipt, preflight, actual_hashes(WORKER), "v007_19f")
    if sha256(CHECKS / "tests.json") != receipt["test_report_sha256"]:
        raise RuntimeError("Extension test evidence changed")
    # Also exercise the exact production receipt loader and its evidence checks.
    from milestone_checkpoints import load_extension
    load_extension(preflight, actual_hashes(WORKER), "v007_19f")
    atomic_json(CHECKS / "INSTALLED.json", {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "worker_sha256": sha256(WORKER),
        "observed_hop_state": read_json(RUN_DIR / "hop_state.json"),
        "observed_progress": read_json(RUN_DIR / "checkpoint_progress.json"),
        "activation": "next natural Slurm requeue after the reviewed edit",
        "receipt_sha256": sha256(CHECKS / "APPROVED.json"),
        "training_settings_unchanged": True,
    })
    print("INSTALLED_AND_VERIFIED: production worker exactly matches the tested candidate.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("test", "verify-install"))
    args = parser.parse_args()
    build_and_test() if args.action == "test" else verify_install()
