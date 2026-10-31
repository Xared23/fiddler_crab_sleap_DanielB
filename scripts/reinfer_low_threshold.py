#!/usr/bin/env python3

"""
Bounded re-inference of one video range at a low centroid threshold (v003, WP-B).

Runs the frozen v007 centroid/centered pair named by a baseline prediction
config over frames ``start .. stop - 1`` of one video and writes, under
``--out``:

- ``<video stem>.frames_<start:06d>_<stop-1:06d>.predictions.slp``: every
  predicted instance, untracked, ``PredictedInstance.score`` = centroid score,
  frame indices as in the video file (``start .. stop - 1``). Because
  ``prediction_finalize.derive_tracks`` requires ``frame_idx == index``
  (frames ``0 .. n - 1``), it consumes this SLP unchanged only when
  ``start == 0``; for ``start > 0`` cut the range into its own clip first (as
  the demo was) or renumber the frames;
- ``raw_candidates.npz``: every centroid slot with keypoints, landmark scores,
  centroid positions and scores, ``frame`` and ``slot``;
- ``RUN_MANIFEST.json``: arguments, baseline config, thresholds, sha256 of the
  model files, package versions and output hashes.

``--dry-run`` validates the arguments, resolves the model directories from the
baseline config and reports which checkpoint files are missing, printing the
plan as JSON; it exits 2 when anything is missing and 0 otherwise, and never
imports torch or sleap_nn. The real run needs a GPU Slurm allocation
(``jobs/reinfer_low_threshold.sh``); it reuses ``grace_prediction_tests``'s
``unpack_output`` and ``predict_streaming`` call pattern.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Sequence

# ----- local modules -----
from pipeline_common import ROOT, atomic_json, read_json, sha256

# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_BASELINE = 'configs/prediction/v007_best_body_first_v001.json'
ROLES = ('centroid', 'centered')
REQUIRED_MODEL_FILES = ('best.ckpt', 'training_config.yaml')
OPTIONAL_MODEL_FILES = ('snapshot.json',)
UNPACK_MIN_CENTROID = 0.05   # grace_prediction_tests.unpack_output asserts scores >= 0.05
EXIT_MISSING = 2

# ============================================================
# ARGUMENTS AND PLAN
# ============================================================


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser."""
    parser = argparse.ArgumentParser(
        description='Re-infer frames start..stop-1 of one video at a low centroid threshold.',
        epilog='The output SLP keeps the video frame numbers (start..stop-1). '
               'prediction_finalize.derive_tracks requires start == 0 (frame_idx == index), '
               'so for start > 0 cut the range into its own clip (as the demo was) or '
               'renumber the frames first. Example: --video crab_vids/Crab_Cam_2_2.mp4 '
               '--start 0 --stop 300 --out predictions/reinfer_low_threshold/cam2_2_000000')
    parser.add_argument('--video', type=Path, required=True, help='video file (mp4)')
    parser.add_argument('--start', type=int, required=True, help='first frame (inclusive)')
    parser.add_argument('--stop', type=int, required=True, help='last frame + 1 (exclusive)')
    parser.add_argument('--out', type=Path, required=True, help='output directory')
    parser.add_argument('--centroid-threshold', type=float, default=0.05,
                        help='centroid peak threshold (default 0.05)')
    parser.add_argument('--keypoint-threshold', type=float, default=0.2,
                        help='centered-instance keypoint threshold (default 0.2)')
    parser.add_argument('--baseline-config', type=Path, default=Path(DEFAULT_BASELINE),
                        help=f'prediction config naming the model pair '
                             f'(default {DEFAULT_BASELINE})')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate and print the plan as JSON; no torch, no inference')
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Exit via ``parser.error`` (status 2) on invalid ranges or thresholds."""
    if not 0 <= args.start < args.stop:
        parser.error(f"need stop > start >= 0, got start={args.start} stop={args.stop}")
    for name in ('centroid_threshold', 'keypoint_threshold'):
        value = getattr(args, name)
        if not 0 < value < 1:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1), got {value}")


def root_path(value: Path) -> Path:
    """Resolve a path relative to the repository root (absolute paths unchanged)."""
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def output_name(video: Path, start: int, stop: int) -> str:
    """``<stem>.frames_<start:06d>_<stop-1:06d>.predictions.slp``."""
    return f"{Path(video).stem}.frames_{start:06d}_{stop - 1:06d}.predictions.slp"


def make_plan(args: argparse.Namespace) -> dict:
    """Resolve every input and output path and list what is missing.

    Parameters
    ----------
    args : argparse.Namespace
        Validated arguments.

    Returns
    -------
    dict
        ``video``, ``frames``, ``thresholds``, ``baseline_config``, ``models``
        (per role: directory, files, expected sha256), ``outputs``, ``missing``
        (paths that must exist and do not), ``problems`` (other blockers) and
        ``ok``.
    """
    missing: list[str] = []
    problems: list[str] = []
    video = Path(args.video).resolve()
    if not video.is_file():
        missing.append(str(video))
    baseline_path = root_path(args.baseline_config)
    models: dict[str, dict] = {}
    baseline: dict = {}
    if not baseline_path.is_file():
        missing.append(str(baseline_path))
    else:
        baseline = read_json(baseline_path)
        for role in ROLES:
            if role not in baseline:
                problems.append(f"baseline config has no {role!r} model directory")
                continue
            directory = root_path(baseline[role])
            files = {name: str(directory / name) for name in REQUIRED_MODEL_FILES}
            models[role] = {'directory': str(directory), 'files': files,
                            'expected_checkpoint_sha256':
                                baseline.get('checkpoint_sha256', {}).get(role)}
            missing.extend(path for path in files.values() if not Path(path).is_file())
    if args.centroid_threshold < UNPACK_MIN_CENTROID:
        problems.append(f"centroid threshold {args.centroid_threshold} < {UNPACK_MIN_CENTROID}: "
                        "grace_prediction_tests.unpack_output asserts centroid scores >= 0.05")
    out = Path(args.out).resolve()
    outputs = {'slp': str(out / output_name(video, args.start, args.stop)),
               'raw_candidates': str(out / 'raw_candidates.npz'),
               'manifest': str(out / 'RUN_MANIFEST.json')}
    existing = [path for path in outputs.values() if Path(path).exists()]
    if existing:
        problems.append(f"outputs already exist: {existing}")
    return {
        'video': str(video), 'frames': {'start': args.start, 'stop_exclusive': args.stop,
                                        'count': args.stop - args.start},
        'thresholds': {'centroid': args.centroid_threshold, 'keypoint': args.keypoint_threshold},
        'baseline_config': str(baseline_path), 'models': models, 'outputs': outputs,
        'missing': missing, 'problems': problems, 'ok': not missing and not problems,
    }


# ============================================================
# INFERENCE (GPU; imports torch and sleap_nn lazily)
# ============================================================


def model_hashes(plan: dict) -> dict[str, str]:
    """sha256 of every required and optional model file that exists."""
    hashes: dict[str, str] = {}
    for model in plan['models'].values():
        directory = Path(model['directory'])
        for name in REQUIRED_MODEL_FILES + OPTIONAL_MODEL_FILES:
            if (directory / name).is_file():
                hashes[str(directory / name)] = sha256(directory / name)
    return hashes


def infer(args: argparse.Namespace, plan: dict) -> dict:
    """Run the predictor over the range and write the three outputs.

    Parameters
    ----------
    args : argparse.Namespace
        Validated arguments.
    plan : dict
        Output of :func:`make_plan` with ``ok`` True.

    Returns
    -------
    dict
        The run manifest written to ``RUN_MANIFEST.json``.
    """
    import numpy as np
    import sleap_io as sio
    import torch
    from sleap_nn.inference.predictor import Predictor

    from grace_prediction_tests import unpack_output
    from pipeline_common import validate_labels, versions
    from threshold_prediction_tests import NODES, save_final

    started = time.monotonic()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Execution requires a Slurm allocation')
    if not torch.cuda.is_available():
        raise RuntimeError('GPU allocation required; refusing to run on CPU')
    baseline = read_json(Path(plan['baseline_config']))
    hashes = model_hashes(plan)
    for role, model in plan['models'].items():
        expected = model['expected_checkpoint_sha256']
        actual = hashes[model['files']['best.ckpt']]
        if expected is not None and actual != expected:
            raise RuntimeError(f"{role} checkpoint sha256 {actual} != baseline {expected}")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    video_path = Path(plan['video'])
    start, stop = args.start, args.stop
    centroid_t, keypoint_t = args.centroid_threshold, args.keypoint_threshold

    predictor = Predictor.from_model_paths(
        [plan['models'][role]['directory'] for role in ROLES], device='cuda', batch_size=1,
        peak_threshold=keypoint_t, max_instances=None, tracker_config=None, filter_config=None)
    layer = predictor._scoped_postprocess_layer(peak_threshold=keypoint_t,
                                                centroid_threshold=centroid_t,
                                                keypoint_threshold=keypoint_t)
    assert layer.centroid_layer.postprocess_config.peak_threshold == centroid_t
    assert layer.centered_instance_layer.postprocess_config.peak_threshold == keypoint_t
    assert layer.centroid_layer.max_instances is None and not layer.centroid_nms
    skeleton = predictor.skeleton
    assert list(skeleton.node_names) == NODES, skeleton.node_names
    video = sio.load_video(str(video_path))
    n_frames, height, width = video.shape[:3]
    assert (height, width) == (baseline['height'], baseline['width']), video.shape
    if stop > n_frames:
        raise ValueError(f"stop {stop} > {n_frames} frames in {video_path}")

    frames, raw_arrays = [], []
    print(f"INFERENCE_BEGIN frames {start}-{stop - 1} centroid {centroid_t} "
          f"keypoint {keypoint_t}", flush=True)
    stream = predictor.predict_streaming(
        str(video_path), frames=list(range(start, stop)), peak_threshold=keypoint_t,
        centroid_threshold=centroid_t, keypoint_threshold=keypoint_t, max_instances=None,
        return_confmaps=False, return_crops=False)
    with torch.inference_mode():
        for index, output in enumerate(stream):
            frame_idx = start + index
            assert frame_idx < stop
            instances, _, _, arrays = unpack_output(output, skeleton, frame_idx)
            frames.append(sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances))
            raw_arrays.append(arrays)
            if (index + 1) % 100 == 0:
                print('INFERENCE_PROGRESS', index + 1, '/', stop - start, flush=True)
    assert len(frames) == stop - start, (len(frames), stop - start)
    labels = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton])
    labels.provenance = dict(low_threshold_reinference=dict(
        centroid_threshold=centroid_t, keypoint_threshold=keypoint_t,
        instance_score='centroid score', tracked=False,
        baseline_config=str(plan['baseline_config']), frames=[start, stop]))
    validate_labels(labels, video_path, start, stop, NODES)
    target = Path(plan['outputs']['slp'])
    save_final(labels, target, video_path, start, stop)

    arrays = {key: np.concatenate([a[key] for a in raw_arrays], axis=0) for key in raw_arrays[0]}
    arrays['selected_frames'] = np.arange(start, stop)
    arrays['node_names'] = np.array(NODES)
    npz = Path(plan['outputs']['raw_candidates'])
    np.savez_compressed(npz, **arrays)
    with np.load(npz, allow_pickle=False) as check:
        for key, value in arrays.items():
            np.testing.assert_array_equal(check[key], value)

    stat = video_path.stat()
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'job_id': os.environ.get('SLURM_JOB_ID'), 'elapsed_s': time.monotonic() - started,
        'arguments': {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        'plan': plan, 'baseline_config': baseline,
        'thresholds': {'centroid': centroid_t, 'keypoint': keypoint_t, 'peak': keypoint_t},
        'model_sha256': hashes, 'versions': versions(),
        'video': {'path': str(video_path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                  'shape': list(video.shape)},
        'frame_indices': 'video-file frame indices start..stop-1 (not re-based to 0)',
        'instance_score': 'centroid score (PredictedInstance.score)',
        'outputs': {Path(p).name: {'sha256': sha256(Path(p)), 'bytes': Path(p).stat().st_size}
                    for p in (target, npz)},
        'counts': {'frames': len(frames), 'instances': int(sum(len(lf.instances) for lf in frames)),
                   'centroid_slots': int(len(arrays['centroid_scores']))},
    }
    atomic_json(Path(plan['outputs']['manifest']), manifest)
    print('COMPLETE', out, flush=True)
    return manifest


# ============================================================
# ENTRY POINT
# ============================================================


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, validate, plan; then print the plan (``--dry-run``) or run inference.

    Returns
    -------
    int
        0 on success; 2 when an input is missing or the plan has a problem.
    """
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    validate_args(parser, args)
    plan = make_plan(args)
    if args.dry_run or not plan['ok']:
        print(json.dumps(dict(plan, dry_run=bool(args.dry_run)), indent=2, sort_keys=True),
              flush=True)
        return 0 if plan['ok'] else EXIT_MISSING
    infer(args, plan)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
