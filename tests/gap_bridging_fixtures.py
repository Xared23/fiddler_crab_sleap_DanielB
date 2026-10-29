"""Synthetic observation tables for the gap-bridging test suites.

A ``Crab`` is a ground-truth trajectory; ``make_observations`` renders crabs
into the published observation schema (core columns of
scripts/prediction_finalize.py OBS_SCHEMA plus centroid/time columns), splits
each crab into one tracklet per contiguous observed run (exactly what the
adjacent-frame tracker produces when frames are missing) and can plant
rejected detections at chosen frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

NODES = ('shell', 'shoulder', 'elbow', 'claw', 'arm')
# Body-frame offsets (px) of each node for a heading of 0 rad (shell at origin,
# shoulder ahead along +x). Realistic scale for the 8K footage.
DEFAULT_OFFSETS = np.array([[0.0, 0.0], [22.0, 0.0], [34.0, 6.0], [40.0, -14.0], [18.0, -10.0]])
FPS = 30000 / 1001

CORE_FIELDS = [
    ('video_id', pa.string()), ('frame', pa.int64()), ('instance_index', pa.int32()),
    ('raw_instance_id', pa.int64()), ('detection_id', pa.string()),
    ('tracklet_id', pa.string()), ('tracklet_local_id', pa.int64()),
    ('accepted_for_tracking', pa.bool_()), ('rejection_reason', pa.string()),
    ('incoming_reason', pa.string()), ('instance_score', pa.float64()),
    ('body_x', pa.float64()), ('body_y', pa.float64()), ('body_score', pa.float64()),
    ('centroid_source', pa.string()), ('water', pa.bool_()),
] + [(f'{node}_{suffix}', kind) for node in NODES
     for suffix, kind in [('x', pa.float64()), ('y', pa.float64()), ('score', pa.float64()), ('visible', pa.bool_())]]
EXTRA_FIELDS = [('centroid_x', pa.float64()), ('centroid_y', pa.float64()), ('time_s', pa.float64())]
SCHEMA = pa.schema(CORE_FIELDS + EXTRA_FIELDS)


@dataclass
class Crab:
    """Ground truth: positions per frame; heading from the Shell->Shoulder vector."""
    crab_id: int
    frames: np.ndarray          # (n,) int64, contiguous
    positions: np.ndarray       # (n,2) shell positions
    headings: np.ndarray        # (n,) radians (NaN = shoulder missing on that frame)
    offsets: np.ndarray = field(default_factory=lambda: DEFAULT_OFFSETS.copy())
    video_id: str = 'synthetic'


def straight_crab(crab_id, start_frame, n_frames, origin, velocity, heading=None, video_id='synthetic'):
    frames = np.arange(start_frame, start_frame + n_frames, dtype=np.int64)
    positions = np.asarray(origin, float)[None, :] + np.arange(n_frames)[:, None] * np.asarray(velocity, float)[None, :]
    if heading is None:
        heading = math.atan2(velocity[1], velocity[0]) if np.hypot(*velocity) > 0 else 0.0
    return Crab(crab_id, frames, positions, np.full(n_frames, float(heading)), video_id=video_id)


def turning_crab(crab_id, start_frame, n_frames, center, radius, omega, phase=0.0, video_id='synthetic'):
    """Uniform circular motion; heading tangent to the circle."""
    frames = np.arange(start_frame, start_frame + n_frames, dtype=np.int64)
    angle = phase + omega * np.arange(n_frames)
    positions = np.asarray(center, float)[None, :] + radius * np.stack([np.cos(angle), np.sin(angle)], axis=1)
    headings = angle + (math.pi / 2 if omega >= 0 else -math.pi / 2)
    return Crab(crab_id, frames, positions, headings, video_id=video_id)


def node_positions(position, heading, offsets=DEFAULT_OFFSETS):
    """Rigid placement of all nodes for one pose; NaN heading keeps only the shell."""
    out = np.full((len(NODES), 2), np.nan)
    out[0] = position
    if np.isfinite(heading):
        c, s = math.cos(heading), math.sin(heading)
        rotation = np.array([[c, -s], [s, c]])
        out[:] = position[None, :] + offsets @ rotation.T
    return out


def make_observations(crabs, drop_frames=None, reject_frames=None, scores=0.9, seed=0):
    """Render crabs to an observation table.

    drop_frames:   {crab_id: iterable of frames} that are absent entirely (missed detections).
    reject_frames: {crab_id: {frame: rejection_reason}} present as unassigned rows; a
                   reason of 'low_body_score' gives body_score 0.3, 'missing_position'
                   gives NaN shell coordinates, other reasons keep the true pose.
    Tracklets: one per contiguous accepted run per crab, numbered in order of
    (first frame, crab_id) from 1. incoming_reason is 'linked' except at tracklet
    starts ('video_start' or 'no_candidate'). Detections within a frame are
    ordered by crab_id. Rows are sorted by (frame, instance_index).
    """
    drop_frames = {k: set(v) for k, v in (drop_frames or {}).items()}
    reject_frames = {k: dict(v) for k, v in (reject_frames or {}).items()}
    rng = np.random.default_rng(seed)
    records = []
    for crab in crabs:
        for k, frame in enumerate(crab.frames):
            frame = int(frame)
            if frame in drop_frames.get(crab.crab_id, ()):
                continue
            reason = reject_frames.get(crab.crab_id, {}).get(frame)
            nodes = node_positions(crab.positions[k], crab.headings[k], crab.offsets)
            body_score = scores if reason != 'low_body_score' else 0.3
            if reason == 'missing_position':
                nodes[0] = np.nan
            records.append(dict(crab_id=crab.crab_id, video_id=crab.video_id, frame=frame,
                                nodes=nodes, body_score=body_score, reason=reason))
    records.sort(key=lambda r: (r['video_id'], r['frame'], r['crab_id']))
    # Tracklet segmentation per crab over accepted rows.
    runs: dict[tuple[str, int], list[list[dict]]] = {}
    for r in records:
        if r['reason'] is not None:
            continue
        key = (r['video_id'], r['crab_id'])
        segments = runs.setdefault(key, [])
        if segments and segments[-1][-1]['frame'] + 1 == r['frame']:
            segments[-1].append(r)
        else:
            segments.append([r])
    all_segments = sorted((seg for segs in runs.values() for seg in segs),
                          key=lambda seg: (seg[0]['video_id'], seg[0]['frame'], seg[0]['crab_id']))
    for local_id, seg in enumerate(all_segments, 1):
        for i, r in enumerate(seg):
            r['local_id'] = local_id
            r['incoming'] = 'linked' if i else ('video_start' if r['frame'] == min(x['frame'] for x in records) else 'no_candidate')
    columns = {field_.name: [] for field_ in SCHEMA}
    instance_index = {}
    for row_number, r in enumerate(records):
        key = (r['video_id'], r['frame'])
        j = instance_index.get(key, 0)
        instance_index[key] = j + 1
        accepted = r['reason'] is None
        nodes = r['nodes']
        shell = nodes[0]
        columns['video_id'].append(r['video_id'])
        columns['frame'].append(r['frame'])
        columns['instance_index'].append(j)
        columns['raw_instance_id'].append(row_number)
        columns['detection_id'].append(f"{r['video_id']}/f{r['frame']:06d}/i{j:04d}")
        columns['tracklet_id'].append(f"synthetic/{r['video_id']}/t{r['local_id']:09d}" if accepted else None)
        columns['tracklet_local_id'].append(r['local_id'] if accepted else None)
        columns['accepted_for_tracking'].append(accepted)
        columns['rejection_reason'].append(r['reason'])
        columns['incoming_reason'].append(r['incoming'] if accepted else 'rejected')
        columns['instance_score'].append(scores)
        columns['body_x'].append(None if np.isnan(shell[0]) else float(shell[0]))
        columns['body_y'].append(None if np.isnan(shell[1]) else float(shell[1]))
        columns['body_score'].append(r['body_score'])
        columns['centroid_source'].append('shell_proxy')
        columns['water'].append(False)
        for n, node in enumerate(NODES):
            xy = nodes[n]
            known = bool(np.isfinite(xy).all())
            columns[f'{node}_x'].append(float(xy[0]) if known else None)
            columns[f'{node}_y'].append(float(xy[1]) if known else None)
            columns[f'{node}_score'].append((r['body_score'] if n == 0 else scores) if known else None)
            columns[f'{node}_visible'].append(known)
        columns['centroid_x'].append(columns['body_x'][-1])
        columns['centroid_y'].append(columns['body_y'][-1])
        columns['time_s'].append(r['frame'] / FPS)
    del rng
    return pa.Table.from_pydict(columns, schema=SCHEMA)


def tracklet_count(table):
    return len(set(v for v in table['tracklet_id'].to_pylist() if v is not None))
