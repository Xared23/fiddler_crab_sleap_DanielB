#!/usr/bin/env python3
"""Read-only label inventory. Trust the collaborator's non-overlap assertion.

Does not merge, deduplicate, remap, modify labels, or open external video paths.
Run only in Slurm. JSON output records enough provenance for later preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import sleap_io as sio

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inventory(path: Path) -> dict:
    labels = sio.load_slp(path, open_videos=False)
    user_frames = [lf for lf in labels if lf.user_instances]
    videos = []
    for i, video in enumerate(labels.videos):
        frames = [lf for lf in user_frames if lf.video is video]
        coordinates = [inst.numpy() for lf in frames for inst in lf.user_instances]
        widths, heights = [], []
        for xy in coordinates:
            good = xy[np.isfinite(xy).all(axis=1)]
            if len(good):
                widths.append(float(np.ptp(good[:, 0])))
                heights.append(float(np.ptp(good[:, 1])))
        source = getattr(video, "source_video", None)
        videos.append({
            "index": i,
            "filename": video.filename,
            "backend_metadata": video.backend_metadata,
            "source_video": None if source is None else {
                "filename": source.filename,
                "backend_metadata": source.backend_metadata,
            },
            "user_frame_indices": [int(lf.frame_idx) for lf in frames],
            "user_frames": len(frames),
            "user_instances": sum(len(lf.user_instances) for lf in frames),
            "pose_width_quantiles_px": None if not widths else np.quantile(widths, [0, .5, .95, 1]).tolist(),
            "pose_height_quantiles_px": None if not heights else np.quantile(heights, [0, .5, .95, 1]).tolist(),
        })
    return {
        "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path),
        "stored_frames": len(labels), "user_labeled_frames": len(user_frames),
        "user_instances": sum(len(lf.user_instances) for lf in user_frames),
        "predicted_instances": sum(len(lf.predicted_instances) for lf in labels),
        "skeletons": [{
            "nodes": [node.name for node in skel.nodes],
            "edges": [[edge.source.name, edge.destination.name] for edge in skel.edges],
        } for skel in labels.skeletons],
        "videos": videos,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(ROOT)
    sources = [inventory(ROOT / name) for name in ("danlabels.v004.slp", "danlabels.v007.slp")]
    total = sum(source["user_labeled_frames"] for source in sources)
    holdout = max(1, int(np.ceil(total * .1))) if total > 1 else 0
    report = {
        "overlap_policy": "Sources assumed non-overlapping per collaborator/user; no overlap audit performed.",
        "sources": sources,
        "total_user_labeled_frames": total,
        "proposed_validation_frames": holdout,
        "proposed_training_frames": total - holdout,
        "same_skeletons": sources[0]["skeletons"] == sources[1]["skeletons"],
        "versions": {name: importlib.metadata.version(name) for name in ("sleap-io", "sleap-nn", "torch", "lightning")},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(report, indent=2, default=str) + "\n")
    tmp.replace(output)
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
