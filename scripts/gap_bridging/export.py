#!/usr/bin/env python3

"""
Export interpolated (bridged) frames as weak labels for centroid fine-tuning (v003, WP-B).

Every ``source == 'interpolated'`` row of a gap-bridging output becomes one
candidate: the interpolated pose, the frame index local to the video file the
pose refers to, and a ``crop_size`` square crop window centered on the Shell
and clipped to stay inside the image. These rows are fine-tuning candidates,
not labels: they are interpolations between real detections and have not been
reviewed by a person.

``main(argv)`` is the ``bridge_gaps.py export`` subcommand::

    export --bridged DIR --out DIR [--frame-offset N] [--max-gap-frames N]
           [--crop-size N] [--confidences high medium] [--force]

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ----- local modules -----
from gap_bridging import tables
from gap_bridging.types import NODES
from pipeline_common import atomic_json, fingerprint, sha256

# ============================================================
# CONSTANTS
# ============================================================

WIDTH, HEIGHT = 7680, 4320                    # native frame size of the crab videos
CONFIDENCES = ('high', 'medium', 'low')
LABEL_STATUS = 'weak_interpolated'
PARQUET_NAME = 'weak_labels.parquet'
README_NAME = 'README.md'
COMPLETE_NAME = 'WEAK_LABELS_COMPLETE.json'
NODE_COLUMNS = tuple(f"{node}_{axis}" for node in NODES for axis in ('x', 'y'))
WEAK_SCHEMA = pa.schema(
    [('video_id', pa.string()), ('frame', pa.int64()), ('local_frame', pa.int64()),
     ('trajectory_id', pa.string()), ('bridge_id', pa.string()), ('source', pa.string()),
     ('gap_frames', pa.int64()), ('fill_confidence', pa.string())]
    + [(name, pa.float64()) for name in NODE_COLUMNS]
    + [('crop_x0', pa.int64()), ('crop_y0', pa.int64()), ('crop_size', pa.int64()),
       ('label_status', pa.string())])

# ============================================================
# EXPORT
# ============================================================


def _frame_offsets(obs: pa.Table) -> dict[str, int]:
    """Per-video ``frame - demo_frame`` over rows with a ``demo_frame``.

    Raises
    ------
    ValueError
        If the offset is not constant within a video.
    """
    rows = obs.select(['video_id', 'frame', 'demo_frame']).filter(pc.is_valid(obs['demo_frame']))
    diff = pc.subtract(rows['frame'], rows['demo_frame'].cast(pa.int64()))
    stats = pa.table({'video_id': rows['video_id'], 'd': diff}).group_by('video_id').aggregate(
        [('d', 'min'), ('d', 'max')]).to_pydict()
    offsets: dict[str, int] = {}
    for video, lo, hi in zip(stats['video_id'], stats['d_min'], stats['d_max']):
        if lo != hi:
            raise ValueError(f"frame - demo_frame is not constant in video {video!r}")
        offsets[video] = int(lo)
    return offsets


def export_weak_labels(bridged_obs: pa.Table, *, frame_offset: int | None, crop_size: int = 256,
                       max_gap_frames: int | None,
                       confidences: Sequence[str] | None = ('high', 'medium'),
                       width: int = WIDTH, height: int = HEIGHT) -> pa.Table:
    """Select interpolated rows of a bridged observation table as weak labels.

    Parameters
    ----------
    bridged_obs : pa.Table
        ``observations.parquet`` of a gap-bridging run.
    frame_offset : int or None
        ``local_frame = frame - frame_offset``. None derives the offset per video
        from ``frame - demo_frame`` when a ``demo_frame`` column exists (the demo
        clip; interpolated rows have a null ``demo_frame``), else uses 0.
    crop_size : int
        Side of the square crop window in pixels (``1 <= crop_size <= min(width, height)``).
    max_gap_frames : int or None
        Keep rows whose bridge has ``gap_frames <= max_gap_frames``; None keeps all.
    confidences : sequence of str or None
        Keep rows whose ``fill_confidence`` is listed; None keeps all.
    width, height : int
        Image size used to clip the crop windows.

    Returns
    -------
    pa.Table
        :data:`WEAK_SCHEMA`, sorted by (video_id, frame, trajectory_id).
        ``crop_x0 = clip(round(shell_x) - crop_size // 2, 0, width - crop_size)``
        (same for y), so ``[crop_x0, crop_x0 + crop_size)`` lies inside the frame.

    Raises
    ------
    ValueError
        On invalid parameters, a missing column, a non-finite Shell position, an
        inconsistent ``demo_frame`` offset or a negative ``local_frame``.
    """
    if not 1 <= crop_size <= min(width, height):
        raise ValueError(f"crop_size must be in [1, {min(width, height)}], got {crop_size}")
    if max_gap_frames is not None and max_gap_frames < 1:
        raise ValueError(f"max_gap_frames must be >= 1, got {max_gap_frames}")
    if confidences is not None and set(confidences) - set(CONFIDENCES):
        raise ValueError(f"unknown confidences {sorted(set(confidences) - set(CONFIDENCES))}")
    needed = {'video_id', 'frame', 'trajectory_id', 'bridge_id', 'source', 'gap_frames',
              'fill_confidence', *NODE_COLUMNS}
    missing = needed - set(bridged_obs.schema.names)
    if missing:
        raise ValueError(f"bridged observations lack columns {sorted(missing)}")

    keep = pc.fill_null(pc.equal(bridged_obs['source'], 'interpolated'), False)
    if max_gap_frames is not None:
        keep = pc.and_(keep, pc.fill_null(pc.less_equal(bridged_obs['gap_frames'],
                                                        max_gap_frames), False))
    if confidences is not None:
        allowed = pa.array(list(confidences), pa.string())
        keep = pc.and_(keep, pc.fill_null(pc.is_in(bridged_obs['fill_confidence'],
                                                   value_set=allowed), False))
    rows = bridged_obs.filter(keep).sort_by(
        [('video_id', 'ascending'), ('frame', 'ascending'), ('trajectory_id', 'ascending')])

    frame = rows['frame'].to_numpy().astype(np.int64)
    if frame_offset is not None:
        local = frame - int(frame_offset)
    elif 'demo_frame' in bridged_obs.schema.names:
        offsets = _frame_offsets(bridged_obs)
        videos = rows['video_id'].to_pylist()
        unknown = set(videos) - set(offsets)
        if unknown:
            raise ValueError(f"no demo_frame offset for videos {sorted(unknown)}")
        local = frame - np.asarray([offsets[v] for v in videos], dtype=np.int64)
    else:
        local = frame.copy()
    if local.size and local.min() < 0:
        raise ValueError(f"negative local_frame (min {local.min()}); check frame_offset")

    shell = {axis: rows[f'shell_{axis}'].to_numpy(zero_copy_only=False).astype(float)
             for axis in ('x', 'y')}                     # nulls become NaN here only
    if not (np.isfinite(shell['x']).all() and np.isfinite(shell['y']).all()):
        raise ValueError("an interpolated row has no finite Shell position")
    half = crop_size // 2
    crop_x0 = np.clip(np.round(shell['x']).astype(np.int64) - half, 0, width - crop_size)
    crop_y0 = np.clip(np.round(shell['y']).astype(np.int64) - half, 0, height - crop_size)
    n = rows.num_rows
    columns = {
        'video_id': rows['video_id'], 'frame': pa.array(frame, pa.int64()),
        'local_frame': pa.array(local, pa.int64()), 'trajectory_id': rows['trajectory_id'],
        'bridge_id': rows['bridge_id'], 'source': rows['source'], 'gap_frames': rows['gap_frames'],
        'fill_confidence': rows['fill_confidence'],
        **{name: rows[name].cast(pa.float64()) for name in NODE_COLUMNS},   # nulls kept
        'crop_x0': pa.array(crop_x0, pa.int64()), 'crop_y0': pa.array(crop_y0, pa.int64()),
        'crop_size': pa.array(np.full(n, crop_size, np.int64), pa.int64()),
        'label_status': pa.array([LABEL_STATUS] * n, pa.string()),
    }
    return pa.table({name: columns[name] for name in WEAK_SCHEMA.names}).cast(WEAK_SCHEMA)


# ============================================================
# WRITING
# ============================================================


def _counts(table: pa.Table) -> dict:
    """Row counts overall, by fill confidence and by video."""
    def value_counts(column: str) -> dict[str, int]:
        pairs = pc.value_counts(table[column]).to_pylist()
        return {str(v['values']): int(v['counts']) for v in pairs}
    return {'rows': table.num_rows, 'by_fill_confidence': value_counts('fill_confidence'),
            'by_video': value_counts('video_id'),
            'bridges': len(set(table['bridge_id'].to_pylist())),
            'trajectories': len(set(table['trajectory_id'].to_pylist()))}


def _readme(provenance: dict, counts: dict) -> str:
    """README text: purpose, schema, provenance and counts."""
    schema_lines = '\n'.join(f"| `{f.name}` | {f.type} |" for f in WEAK_SCHEMA)
    params = provenance['parameters']
    return f"""# Weak labels from gap bridging

**These rows are fine-tuning candidates, not labels.** Each row is a pose
interpolated by gap bridging between two real detections of the same
trajectory; no person has reviewed it and no detector produced it. Use them
only as weak supervision (e.g. candidate centroid positions for a fine-tuning
round), after visual review of a sample, and never as ground truth or as an
evaluation set.

## Provenance

- Source run: `{provenance['source_dir']}`
- Source marker `{provenance['source_marker_name']}` sha256
  ({provenance['source_marker_hash_kind']}):
  `{provenance['source_marker_sha256']}`
- Source `observations.parquet` sha256: `{provenance['source_observations_sha256']}`
- Source interpolated rows: {provenance['source_interpolated_rows']}
- Exported rows: {counts['rows']} from {counts['bridges']} bridges in
  {counts['trajectories']} trajectories; by fill confidence:
  {json.dumps(counts['by_fill_confidence'], sort_keys=True)}
- Parameters: frame_offset={params['frame_offset']}, crop_size={params['crop_size']},
  max_gap_frames={params['max_gap_frames']}, confidences={params['confidences']},
  image {params['width']} x {params['height']}

## Conventions

- `frame` is the original video frame; `local_frame` indexes the video file the
  poses refer to (`demo_frame` offset for the demo clip, else `frame - frame_offset`).
- Coordinates are native-image pixels (x right, y down), as in the source table.
  Node coordinates are carried unchanged (null or NaN where the source pose has none).
- The crop window is `[crop_x0, crop_x0 + crop_size) x [crop_y0, crop_y0 + crop_size)`,
  centered on the rounded Shell position and shifted to stay inside the image.
- `label_status` is always `{LABEL_STATUS}`.

## Schema (`{PARQUET_NAME}`)

| Column | Type |
|---|---|
{schema_lines}
"""


def write_weak_labels(table: pa.Table, out_dir: Path, source_marker: dict, force: bool, *,
                      source_marker_path: Path | None = None,
                      parameters: dict | None = None) -> dict:
    """Write ``weak_labels.parquet``, ``README.md`` and ``WEAK_LABELS_COMPLETE.json``.

    Parameters
    ----------
    table : pa.Table
        Output of :func:`export_weak_labels`.
    out_dir : Path
        Output directory (created if needed).
    source_marker : dict
        Content of the source run's ``GAP_BRIDGING_COMPLETE.json``.
    force : bool
        Overwrite existing outputs.
    source_marker_path : Path, optional
        The marker file; its sha256 is recorded. Without it the sha256 of the
        canonical JSON of ``source_marker`` (sorted keys) is recorded instead.
    parameters : dict, optional
        Export parameters recorded in the README and the marker.

    Returns
    -------
    dict
        The completion marker: provenance, parameters, counts and ``outputs``
        (sha256 and rows/bytes of the parquet and the README).

    Raises
    ------
    FileExistsError
        If an output exists and ``force`` is False.
    """
    if not table.schema.equals(WEAK_SCHEMA, check_metadata=False):
        raise ValueError("table does not have WEAK_SCHEMA")
    out_dir = Path(out_dir)
    names = (PARQUET_NAME, README_NAME, COMPLETE_NAME)
    existing = [n for n in names if (out_dir / n).exists()]
    if existing and not force:
        raise FileExistsError(f"outputs already exist in {out_dir}: {existing}; use --force")
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / COMPLETE_NAME).exists():
        (out_dir / COMPLETE_NAME).unlink()

    if source_marker_path is not None:
        marker_sha, kind = sha256(Path(source_marker_path)), 'file'
    else:
        marker_sha, kind = fingerprint(source_marker), 'canonical JSON, sorted keys'
    outputs_in = source_marker.get('outputs', {})
    provenance = {
        'source_dir': str(Path(source_marker_path).parent) if source_marker_path else None,
        'source_marker_name': tables.COMPLETE_NAME, 'source_marker_sha256': marker_sha,
        'source_marker_hash_kind': kind,
        'source_observations_sha256': outputs_in.get('observations.parquet', {}).get('sha256'),
        'source_interpolated_rows': source_marker.get('counts', {}).get('n_interpolated'),
        'source_config': source_marker.get('config'),
        'parameters': dict(parameters or {}),
    }
    for key in ('frame_offset', 'crop_size', 'max_gap_frames', 'confidences', 'width', 'height'):
        provenance['parameters'].setdefault(key, None)
    counts = _counts(table)

    temp = out_dir / f".{PARQUET_NAME}.tmp"
    try:
        pq.write_table(table, temp, compression='zstd')
        os.replace(temp, out_dir / PARQUET_NAME)
    finally:
        if temp.exists():
            temp.unlink()
    readme_temp = out_dir / f".{README_NAME}.tmp"
    readme_temp.write_text(_readme(provenance, counts))
    os.replace(readme_temp, out_dir / README_NAME)

    marker = {
        'label_status': LABEL_STATUS,
        'statement': 'fine-tuning candidates, not labels',
        'provenance': provenance, 'counts': counts,
        'outputs': {PARQUET_NAME: {'sha256': sha256(out_dir / PARQUET_NAME),
                                   'rows': table.num_rows},
                    README_NAME: {'sha256': sha256(out_dir / README_NAME),
                                  'bytes': (out_dir / README_NAME).stat().st_size}},
    }
    atomic_json(out_dir / COMPLETE_NAME, marker)
    return marker


# ============================================================
# COMMAND LINE
# ============================================================


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the ``export`` arguments to ``parser`` (for a subcommand parser)."""
    parser.add_argument('--bridged', type=Path, required=True,
                        help='gap-bridging output directory (with GAP_BRIDGING_COMPLETE.json)')
    parser.add_argument('--out', type=Path, required=True, help='output directory')
    parser.add_argument('--frame-offset', type=int, default=None,
                        help='local_frame = frame - N (default: demo_frame offset, else 0)')
    parser.add_argument('--max-gap-frames', type=int, default=None,
                        help='keep bridges with at most N missing frames (default: all)')
    parser.add_argument('--crop-size', type=int, default=256, help='crop side in pixels')
    parser.add_argument('--confidences', nargs='+', choices=CONFIDENCES,
                        default=['high', 'medium'], help='fill confidences to keep')
    parser.add_argument('--width', type=int, default=WIDTH, help='image width in pixels')
    parser.add_argument('--height', type=int, default=HEIGHT, help='image height in pixels')
    parser.add_argument('--force', action='store_true', help='overwrite existing outputs')
    return parser


def run(args: argparse.Namespace) -> dict:
    """Execute an export from parsed arguments; returns the completion marker.

    Raises
    ------
    FileNotFoundError
        If the source marker or observations are missing.
    ValueError
        If ``observations.parquet`` does not match the hash in the source marker.
    """
    bridged = Path(args.bridged)
    marker_path = bridged / tables.COMPLETE_NAME
    obs_path = bridged / 'observations.parquet'
    if not marker_path.is_file():
        raise FileNotFoundError(f"{marker_path} missing: not a complete gap-bridging run")
    source_marker = json.loads(marker_path.read_text())
    expected = source_marker.get('outputs', {}).get('observations.parquet', {}).get('sha256')
    if expected != sha256(obs_path):
        raise ValueError(f"{obs_path} does not match the sha256 in {marker_path}")
    table = export_weak_labels(pq.read_table(obs_path), frame_offset=args.frame_offset,
                               crop_size=args.crop_size, max_gap_frames=args.max_gap_frames,
                               confidences=tuple(args.confidences), width=args.width,
                               height=args.height)
    parameters = {'frame_offset': args.frame_offset, 'crop_size': args.crop_size,
                  'max_gap_frames': args.max_gap_frames, 'confidences': list(args.confidences),
                  'width': args.width, 'height': args.height}
    return write_weak_labels(table, args.out, source_marker, args.force,
                             source_marker_path=marker_path, parameters=parameters)


def main(argv: Sequence[str] | None = None) -> int:
    """``bridge_gaps.py export`` entry point; ``argv`` excludes the subcommand name."""
    parser = add_arguments(argparse.ArgumentParser(
        prog='bridge_gaps.py export',
        description='Export interpolated frames as weak labels (fine-tuning candidates, '
                    'not labels).'))
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    marker = run(args)
    print(f"[export] {marker['counts']['rows']} weak-label rows -> {Path(args.out) / PARQUET_NAME}",
          flush=True)
    return 0
