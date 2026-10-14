#!/usr/bin/env python3
"""Back up/relink v007, freeze 17/2 frame split and materialize training YAMLs.

Slurm only. Relinking the source SLP is explicitly authorized by the user; the
collaborator's original is backed up and verified before any replacement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
import sleap_io as sio

from pipeline_common import (ROOT, atomic_json, dataset_dir, dataset_manifest,
    exclusive_lock, fingerprint, load_run, pin_manifest, project_path,
    save_slp_atomic, sha256, versions)


def annotation_signature(labels) -> dict:
    """Video-path-independent content check; no deduplication or label edits."""
    return {"frames": [{"index": int(lf.frame_idx), "instances": [
        {"points": inst.points.tolist(), "track": None if inst.track is None else inst.track.name,
         "nodes": list(inst.skeleton.node_names)} for inst in lf.instances]
    } for lf in labels], "suggestion_indices": [int(s.frame_idx) for s in labels.suggestions]}


def signature(labels) -> str:
    import hashlib
    # Points contain NaNs and numpy scalars; serialize consistently rather than ==.
    text = json.dumps(annotation_signature(labels), sort_keys=True, default=lambda x: x.tolist())
    return hashlib.sha256(text.encode()).hexdigest()


def split_indices(indices: list[int], count: int, seed: int):
    indices = sorted(indices)
    if len(set(indices)) != len(indices) or not 0 < count < len(indices):
        raise ValueError("Invalid frame inventory or validation size")
    val = sorted(int(i) for i in np.random.default_rng(seed).choice(indices, count, replace=False))
    return [i for i in indices if i not in val], val


def validate_split(labels, video: Path, indices: list[int], instances: int) -> None:
    if len(labels.videos) != 1 or Path(labels.videos[0].filename) != video:
        raise RuntimeError("Split contains incorrect video reference")
    if [int(lf.frame_idx) for lf in labels] != indices:
        raise RuntimeError("Split frame inventory mismatch")
    if sum(len(lf.user_instances) for lf in labels) != instances or any(lf.predicted_instances for lf in labels):
        raise RuntimeError("Split instance inventory mismatch")


def prepare(config: Path) -> None:
    run = load_run(config)
    cfg = run["dataset"]
    directory = dataset_dir(run)
    with exclusive_lock(directory):
        if (directory / "DATASET_COMPLETE.json").exists():
            dataset_manifest(run)
            print("Dataset already prepared and checksums validated.")
            return
        source = project_path(cfg["source"])
        video_path = project_path(cfg["video"])
        backup = directory / "source" / "danlabels.v007.collaborator.slp"
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            if sha256(source) != cfg["collaborator_sha256"]:
                raise RuntimeError("Source differs from the inventoried collaborator file; refusing remap")
            temp = backup.with_suffix(".tmp.slp")
            shutil.copy2(source, temp)
            if sha256(temp) != cfg["collaborator_sha256"]:
                raise RuntimeError("Backup verification failed")
            temp.replace(backup)
        if sha256(backup) != cfg["collaborator_sha256"]:
            raise RuntimeError("Original backup changed")
        original = sio.load_slp(backup, open_videos=False)
        labels = sio.load_slp(source, open_videos=False)
        if signature(labels) != signature(original):
            raise RuntimeError("Source annotations changed; refusing to overwrite")
        if len(labels.videos) != 1 or len(labels.skeletons) != 1:
            raise RuntimeError("Expected one source video and one skeleton")
        if list(labels.skeletons[0].node_names) != ["Shell", "Shoulder", "Elbow", "Claw", "Arm"]:
            raise RuntimeError("Unexpected node order")
        frames = sorted([lf for lf in labels if lf.user_instances], key=lambda lf: lf.frame_idx)
        if len(frames) != cfg["expected_user_frames"]:
            raise RuntimeError(f"Expected 19 human-labeled frames, found {len(frames)}")
        recorded_shape = original.videos[0].backend_metadata.get("shape")
        labels.videos[0].replace_filename(video_path)
        actual_shape = list(labels.videos[0].shape)
        if recorded_shape is not None and actual_shape != recorded_shape:
            raise RuntimeError(f"Local video shape {actual_shape} != original {recorded_shape}")
        # Verify every labeled source image, not just the first frame.
        for lf in frames:
            image = lf.image
            if tuple(image.shape) != tuple(actual_shape[1:]):
                raise RuntimeError(f"Unexpected image shape for frame {lf.frame_idx}")
        original_signature = signature(original)
        def validate_remap(check):
            if signature(check) != original_signature:
                raise RuntimeError("Relinking would change annotations")
            if Path(check.videos[0].filename) != video_path:
                raise RuntimeError("Relink read-back failed")
        if Path(sio.load_slp(source, open_videos=False).videos[0].filename) != video_path:
            save_slp_atomic(labels, source, validate_remap)
        else:
            validate_remap(sio.load_slp(source, open_videos=False))

        train_idx, val_idx = split_indices([int(lf.frame_idx) for lf in frames], cfg["validation_frames"], cfg["split_seed"])
        outputs, split_details = [], {}
        for name, indices in (("train", train_idx), ("val", val_idx)):
            selected = [lf for lf in frames if lf.frame_idx in indices]
            for lf in selected:
                lf.instances = list(lf.user_instances)
            subset = sio.Labels(labeled_frames=selected, videos=labels.videos, skeletons=labels.skeletons)
            count = sum(len(lf.user_instances) for lf in subset)
            path = directory / f"{name}.slp"
            save_slp_atomic(subset, path, lambda check, ix=indices, n=count: validate_split(check, video_path, ix, n))
            outputs.append({"path": str(path), "sha256": sha256(path)})
            split_details[name] = {"frames": indices, "n_frames": len(indices), "n_instances": count}

        resolved_configs = {}
        baselines = {}
        config_dir = ROOT / "configs" / run["run_id"]
        for role, baseline in run["training"]["base_configs"].items():
            base = project_path(baseline)
            baselines[role] = {"path": str(base), "sha256": sha256(base)}
            resolved = OmegaConf.load(base)
            resolved.data_config.train_labels_path = [str(directory / "train.slp")]
            resolved.data_config.val_labels_path = [str(directory / "val.slp")]
            resolved.data_config.use_same_data_for_val = False
            resolved.data_config.validation_fraction = cfg["validation_fraction"]
            tc = resolved.trainer_config
            tc.max_epochs = run["training"]["max_epochs"]
            tc.early_stopping.patience = run["training"]["early_stopping_patience"]
            tc.seed = run["training"]["seed"]
            tc.ckpt_dir = str(ROOT / "models" / run["run_id"])
            tc.run_name = role
            tc.resume_ckpt_path = None
            tc.trainer_accelerator = "gpu"
            tc.enable_progress_bar = False
            tc.model_ckpt.save_last = True
            tc.zmq.controller_port = None
            tc.zmq.publish_port = None
            tc.wandb.name = f"{run['run_id']}_{role}"
            resolved.description = "v007 only: 17 train / 2 validation frames; no regional treatment"
            path = config_dir / f"{role}.yaml"
            temporary = path.with_suffix(".tmp.yaml")
            OmegaConf.save(resolved, temporary)
            temporary.replace(path)
            outputs.append({"path": str(path), "sha256": sha256(path)})
            resolved_configs[role] = str(path)

        manifest = {"run_id": run["run_id"], "dataset_config": cfg,
            "sources": [{"path": str(source), "sha256": sha256(source)}],
            "original_backup": {"path": str(backup), "sha256": sha256(backup)},
            "video": {"path": str(video_path), "shape": actual_shape,
                      "bytes": video_path.stat().st_size, "mtime_ns": video_path.stat().st_mtime_ns},
            "splits": split_details, "baseline_configs": baselines,
            "resolved_configs": resolved_configs, "versions": versions(),
            "regional_handling": "none"}
        pin_manifest(directory, manifest)
        atomic_json(directory / "DATASET_COMPLETE.json", {"manifest_sha256": fingerprint(manifest), "outputs": outputs})
        dataset_manifest(run)
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.config)
