#!/usr/bin/env python3
"""Finite Slurm-only checkpoint preservation, resume checks, or notes relocation.

Never submits, cancels, requeues, trains, predicts, or edits pinned training code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from omegaconf import OmegaConf
from pipeline_common import (ROOT, atomic_json, dataset_manifest, exclusive_lock,
    fingerprint, load_run, project_path, read_json, sha256, versions)
from train_versioned import implementation_hashes, select_checkpoint
from milestone_checkpoints import load_extension

RUN = ROOT / "models/v007_19f"
PAUSE = RUN / "pauses/step_00021212"
OLD_JOB = "15256499"


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if hasattr(value, "numel"):
        return value.item() if value.numel() == 1 else {"tensor_shape": list(value.shape)}
    return value


def inspect_checkpoint(path):
    import torch
    saved = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "optimizer_states", "lr_schedulers", "callbacks", "loops"):
        if not saved.get(key):
            raise RuntimeError(f"Missing full training state {key}: {path}")
    if not any("EarlyStopping" in key for key in saved["callbacks"]):
        raise RuntimeError(f"Missing early-stopping callback state: {path}")
    steps = [float(state["step"]) for opt in saved["optimizer_states"]
             for state in opt["state"].values() if "step" in state]
    report = {
        "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        "epoch_zero_based": int(saved["epoch"]), "global_step": int(saved["global_step"]),
        "optimizer_parameter_states": sum(len(opt["state"]) for opt in saved["optimizer_states"]),
        "optimizer_step_range": [min(steps), max(steps)] if steps else None,
        "optimizer_groups": [{k: json_safe(v) for k, v in group.items() if k != "params"}
                             for opt in saved["optimizer_states"] for group in opt["param_groups"]],
        "lr_schedulers": json_safe(saved["lr_schedulers"]),
        "callbacks": json_safe(saved["callbacks"]),
        "loop_state": json_safe(saved["loops"]),
    }
    if report["global_step"] < 1 or not report["optimizer_parameter_states"]:
        raise RuntimeError(f"Checkpoint has no learned optimizer state: {path}")
    return report


def check_training_baseline():
    cfg = load_run("configs/v007_19f/run.yaml")
    data = dataset_manifest(cfg)
    preflight = read_json(ROOT / "checks/v007_19f/PREFLIGHT_COMPLETE.json")
    if (sha256(ROOT / "configs/v007_19f/run.yaml") != preflight["run_config_sha256"]
            or fingerprint(data) != preflight["dataset_sha256"]):
        raise RuntimeError("Run configuration or data changed")
    extension = load_extension(preflight, implementation_hashes(), cfg["run_id"])
    video = data["manifest"]["video"]
    stat = project_path(video["path"]).stat()
    if (stat.st_size, stat.st_mtime_ns) != (video["bytes"], video["mtime_ns"]):
        raise RuntimeError("Training video changed")
    for role in ("centroid", "centered"):
        config_path = project_path(data["manifest"]["resolved_configs"][role])
        expected = {"run_id": cfg["run_id"], "role": role, "smoke": False,
            "dataset_sha256": fingerprint(data), "config_sha256": sha256(config_path),
            "config": OmegaConf.to_container(OmegaConf.load(config_path), resolve=True),
            "versions": versions(), "implementation": preflight["implementation"],
            "training_policy": cfg["training"]}
        if read_json(RUN / role / "manifest.json") != expected:
            raise RuntimeError(f"Pinned {role} run manifest no longer matches")
    return {"implementation": implementation_hashes(), "versions": versions(),
            "run_config_sha256": preflight["run_config_sha256"],
            "dataset_sha256": preflight["dataset_sha256"], "extension": extension}


def verify_frozen():
    receipt = read_json(PAUSE / "PAUSE_COMPLETE.json")
    for item in receipt["frozen_files"]:
        path = PAUSE / item["relative_path"]
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen artifact changed: {path}")
    return receipt


def verify_resume():
    baseline = check_training_baseline()
    receipt = verify_frozen()
    with exclusive_lock(RUN / "centroid"):
        if (RUN / "centroid/TRAINING_COMPLETE.json").exists():
            raise RuntimeError("Centroid training is already complete; no resume needed")
        selected, progress = select_checkpoint(RUN / "centroid")
        if selected is None or progress["global_step"] < receipt["resume_progress"]["global_step"]:
            raise RuntimeError("Latest canonical checkpoint regressed behind the preserved pause")
        state = inspect_checkpoint(selected)
        hop = read_json(RUN / "centroid/hop_state.json")
        policy = load_run("configs/v007_19f/run.yaml")["training"]
        if hop["hops"] >= policy["max_hops"] or hop["no_progress_hops"] >= policy["max_no_progress_hops"]:
            raise RuntimeError("Existing hop safety limit reached; review instead of resetting counters")
    report = {"passed": True, "checked_utc": datetime.now(timezone.utc).isoformat(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "selected_checkpoint": str(selected.relative_to(ROOT)), "progress": progress,
              "optimizer_step_range": state["optimizer_step_range"], "hop_state": hop,
              "actual_implementation": baseline["implementation"],
              "note": "Verification only; no optimizer step, training restart, or GPU test performed."}
    atomic_json(ROOT / "checks/v007_19f/pause_15256499/RESUME_CHECK.json", report)
    print(json.dumps(report, indent=2), flush=True)


def preserve():
    baseline = check_training_baseline()
    with exclusive_lock(RUN / "centroid"), exclusive_lock(RUN / "centered"):
        if PAUSE.exists():
            raise RuntimeError("Pause bundle already exists; use verify-resume, never overwrite it")
        selected, progress = select_checkpoint(RUN / "centroid")
        if selected is None or progress != {"global_step": 21212, "epoch": 104}:
            raise RuntimeError(f"Unexpected pause position: {selected}, {progress}; review before snapshot")
        milestones = {}
        for completed in (75, 100):
            directory = RUN / "centroid/milestones" / f"epoch_{completed:04d}"
            marker = read_json(directory / "snapshot.json")
            state = inspect_checkpoint(directory / "best.ckpt")
            if (state["sha256"] != marker["checkpoint_sha256"]
                    or state["epoch_zero_based"] != completed - 1
                    or marker["completed_epochs"] != completed
                    or state["global_step"] != marker["global_step"]
                    or sha256(directory / "training_config.yaml") != marker["training_config_sha256"]):
                raise RuntimeError(f"Milestone validation failed: {completed}")
            milestones[str(completed)] = {"checkpoint": state, "snapshot": marker}
        centered_marker = read_json(RUN / "centered/TRAINING_COMPLETE.json")
        if (sha256(RUN / "centered/best.ckpt") != centered_marker["best_sha256"]
                or fingerprint(read_json(RUN / "centered/manifest.json")) != centered_marker["manifest_sha256"]):
            raise RuntimeError("Completed pose model failed validation")
        temporary = PAUSE.with_name(f".{PAUSE.name}.staging_{os.environ.get('SLURM_JOB_ID', 'unknown')}")
        temporary.mkdir(parents=True, exist_ok=False)
        models = {}
        sources = {"centroid_latest": selected, "centroid_best": RUN / "centroid/best.ckpt",
                   "centered_best": RUN / "centered/best.ckpt"}
        for name, source in sources.items():
            state = inspect_checkpoint(source)
            directory = temporary / name
            directory.mkdir()
            checkpoint = directory / "best.ckpt"
            shutil.copy2(source, checkpoint)
            if sha256(checkpoint) != state["sha256"]:
                raise RuntimeError(f"Checkpoint copy changed: {source}")
            # Independently deserialize the COPY, not just the original file.
            copied = inspect_checkpoint(checkpoint)
            if (copied["epoch_zero_based"], copied["global_step"]) != (state["epoch_zero_based"], state["global_step"]):
                raise RuntimeError("Checkpoint copy progress mismatch")
            for filename in ("training_config.yaml", "initial_config.yaml", "manifest.json",
                             "checkpoint_extension_active.json", "training_log.csv", "TRAINING_COMPLETE.json"):
                item = source.parent / filename
                if item.is_file():
                    shutil.copy2(item, directory / filename)
            atomic_json(directory / "snapshot.json", {
                "kind": "latest_partial_epoch_weights" if name == "centroid_latest" else "validation_best_weights",
                "source_checkpoint": state, "inference_filename": "best.ckpt",
                "note": "Full optimizer/scheduler/callback state preserved; best.ckpt in centroid_latest is NOT historical validation-best weights."})
            models[name] = state
        frozen_files = [{"relative_path": str(path.relative_to(temporary)), "sha256": sha256(path)}
                        for path in sorted(temporary.rglob("*")) if path.is_file()]
        receipt = {"schema_version": 1, "saved_utc": datetime.now(timezone.utc).isoformat(),
            "verification_job_id": os.environ.get("SLURM_JOB_ID"), "stopped_training_job_id": OLD_JOB,
            "stop_context": "User requested pause; Slurm cancellation was confirmed while pending between allocations.",
            "resume_checkpoint": str(selected.relative_to(ROOT)), "resume_progress": progress,
            "hop_state": read_json(RUN / "centroid/hop_state.json"),
            "models": models, "milestones": milestones, "baseline": baseline,
            "frozen_files": frozen_files,
            "resume": "Submit jobs/resume_v007_centroid.sh only when the user wants training to continue.",
            "not_training_complete": True}
        atomic_json(temporary / "PAUSE_COMPLETE.json", receipt)
        temporary.rename(PAUSE)
        atomic_json(RUN / "TRAINING_PAUSED.json", {
            "stopped_training_job_id": OLD_JOB, "pause_receipt": str((PAUSE / "PAUSE_COMPLETE.json").relative_to(ROOT)),
            "pause_receipt_sha256": sha256(PAUSE / "PAUSE_COMPLETE.json"), "resume_progress": progress,
            "note": "Historical pause receipt, not a scheduler lock or live job-status file. No completion was declared."})
    # Syntax-check the future batch script without running/submitting it.
    script = ROOT / "jobs/resume_v007_centroid.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    if "#SBATCH --dependency" in script.read_text():
        raise RuntimeError("Resume script must not retain the expired launch dependency")
    verify_resume()
    print(f"PRESERVED_AND_VERIFIED {PAUSE}", flush=True)


def organize_notes():
    mapping = {"README.md": "docs/PROJECT.md",
               "logs/legacy_dataset/README.md": "docs/datasets/legacy_native_tracklets_v001/JOBS.md"}
    for name in ("README.md", "ASSESSMENT.md", "SCHEMA_PROPOSAL.md"):
        mapping[f"datasets/legacy_native_tracklets_v001/{name}"] = f"docs/datasets/legacy_native_tracklets_v001/{name}"
    with exclusive_lock(ROOT / "docs", ".organization.lock"):
        outside = []
        for base, directories, files in os.walk(ROOT):
            directories[:] = [d for d in directories if not d.startswith(".") and d not in ("__pycache__", "node_modules")]
            for name in files:
                path = Path(base) / name
                if path.suffix.lower() == ".md" and not path.is_symlink() and not path.is_relative_to(ROOT / "docs"):
                    outside.append(str(path.relative_to(ROOT)))
        if set(outside) != set(mapping):
            raise RuntimeError(f"Unexpected project Markdown outside docs; inspect before moving: {outside}")
        records = []
        for old, new in mapping.items():
            source, target = ROOT / old, ROOT / new
            if not source.is_file() or target.exists():
                raise RuntimeError(f"Cannot safely relocate {old} -> {new}")
            target.parent.mkdir(parents=True, exist_ok=True)
            records.append({"old": old, "new": new, "sha256": sha256(source)})
        for record in records:
            source, target = ROOT / record["old"], ROOT / record["new"]
            if sha256(source) != record["sha256"]:
                raise RuntimeError("Document changed during relocation")
            source.rename(target)
            if sha256(target) != record["sha256"]:
                raise RuntimeError("Relocated document checksum mismatch")
        broken = []
        for doc in (ROOT / "docs").rglob("*.md"):
            for link in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", doc.read_text()):
                link = link.split("#", 1)[0]
                if not link or re.match(r"[A-Za-z]+:", link):
                    continue
                if not (doc.parent / link).exists():
                    broken.append({"document": str(doc.relative_to(ROOT)), "link": link})
        atomic_json(ROOT / "docs/organization.json", {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "moved": records,
            "broken_local_links": broken, "scope": "Project notes only; hidden tool/agent configuration and dependencies excluded."})
        if broken:
            raise RuntimeError(f"Broken documentation links: {broken}")
        print("NOTES_ORGANIZED: all canonical project Markdown is under docs; local links validated.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preserve", "verify-resume", "organize-notes"))
    args = parser.parse_args()
    {"preserve": preserve, "verify-resume": verify_resume, "organize-notes": organize_notes}[args.action]()
