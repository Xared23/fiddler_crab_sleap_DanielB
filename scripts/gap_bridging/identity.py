#!/usr/bin/env python3

"""
Spurious-identity filter for bridged trajectories (v003, WP-B).

A trajectory is flagged ``'spurious'`` when it has fewer than ``min_frames``
real frames (rows with ``source`` in :data:`REAL_SOURCES`) AND the median
``instance_score`` over those real rows is below ``min_score``; every other
trajectory is an ``'identity'``. The flag is computed after linking, on the
final trajectories, so a one-frame tracklet bridged into a chain is an
identity. Nothing is deleted and linking is never affected: the status is an
annotation (``trajectory_status``) and an alternative way of counting.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

from typing import Mapping

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# ============================================================
# CONSTANTS
# ============================================================

REAL_SOURCES = ('observed', 'recovered', 'recovered_tail')
STATUSES = ('identity', 'spurious')
STATUS_COLUMN = 'trajectory_status'

# ============================================================
# CLASSIFICATION
# ============================================================


def classify_trajectories(observations: pa.Table, *, min_frames: int,
                          min_score: float) -> dict[str, str]:
    """Classify every trajectory of an assembled observation table.

    Parameters
    ----------
    observations : pa.Table
        Bridged observations with ``trajectory_id``, ``source`` and
        ``instance_score``; rows with a null ``trajectory_id`` are ignored.
    min_frames : int
        A trajectory with at least this many real frames is an identity.
    min_score : float
        A trajectory whose median real-row ``instance_score`` is at least this
        is an identity.

    Returns
    -------
    dict
        ``trajectory_id -> 'identity' | 'spurious'`` for every trajectory with
        at least one row. ``'spurious'`` iff real frames < ``min_frames`` and
        median score < ``min_score``. A trajectory without real rows, or whose
        real rows all have a null/NaN score, has no score evidence and is
        ``'spurious'`` when it is also short.

    Raises
    ------
    ValueError
        If ``min_frames < 1`` or a required column is missing.
    """
    if min_frames < 1:
        raise ValueError(f"min_frames must be >= 1, got {min_frames}")
    missing = {'trajectory_id', 'source', 'instance_score'} - set(observations.schema.names)
    if missing:
        raise ValueError(f"observations lack columns {sorted(missing)}")
    assigned = observations.select(['trajectory_id', 'source', 'instance_score']).filter(
        pc.is_valid(observations['trajectory_id']))
    trajectories = sorted(set(assigned['trajectory_id'].to_pylist()))
    is_real = pc.fill_null(pc.is_in(assigned['source'],
                                    value_set=pa.array(REAL_SOURCES, pa.string())), False)
    real = assigned.filter(is_real)
    scores: dict[str, list[float]] = {t: [] for t in trajectories}
    n_real: dict[str, int] = dict.fromkeys(trajectories, 0)
    for traj, score in zip(real['trajectory_id'].to_pylist(), real['instance_score'].to_pylist()):
        n_real[traj] += 1
        if score is not None and np.isfinite(score):
            scores[traj].append(float(score))
    status: dict[str, str] = {}
    for traj in trajectories:
        median = float(np.median(scores[traj])) if scores[traj] else float('nan')
        low_score = not median >= min_score            # NaN (no evidence) counts as low
        status[traj] = 'spurious' if n_real[traj] < min_frames and low_score else 'identity'
    return status


def status_column(observations: pa.Table, status: Mapping[str, str]) -> pa.Array:
    """The ``trajectory_status`` of every row (null where ``trajectory_id`` is null).

    Parameters
    ----------
    observations : pa.Table
        Table with a ``trajectory_id`` column (observations or trajectories).
    status : mapping
        Output of :func:`classify_trajectories`.

    Returns
    -------
    pa.Array
        String array aligned with the rows of ``observations``.

    Raises
    ------
    ValueError
        If a non-null ``trajectory_id`` has no status or a status is unknown.
    """
    bad = set(status.values()) - set(STATUSES)
    if bad:
        raise ValueError(f"unknown statuses: {sorted(bad)}")
    keys = sorted(status)
    idx = pc.index_in(observations['trajectory_id'], value_set=pa.array(keys, pa.string()))
    uncovered = pc.sum(pc.and_(pc.is_valid(observations['trajectory_id']),
                               pc.is_null(idx))).as_py() or 0
    if uncovered:
        raise ValueError(f"{uncovered} rows have a trajectory_id without a status")
    column = pc.take(pa.array([status[k] for k in keys], pa.string()), idx)
    return column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column


# ============================================================
# SUMMARIES
# ============================================================


def _video_first_frames(observations: pa.Table) -> dict[str, int]:
    """First frame of every video over its non-interpolated rows."""
    table = observations.select([n for n in ('video_id', 'frame', 'source')
                                 if n in observations.schema.names])
    if 'source' in table.schema.names:
        keep = pc.fill_null(pc.not_equal(table['source'], 'interpolated'), True)
        table = table.filter(keep)
    first = table.group_by('video_id').aggregate([('frame', 'min')]).to_pydict()
    return {v: int(f) for v, f in zip(first['video_id'], first['frame_min']) if v is not None}


def identity_starts(trajectory_table: pa.Table, observations: pa.Table,
                    status: Mapping[str, str] | None) -> dict[str, int]:
    """Count trajectories that start after their video's first frame.

    Parameters
    ----------
    trajectory_table : pa.Table
        ``trajectory_id``, ``video_id`` and ``start_frame`` per trajectory.
    observations : pa.Table
        Bridged observations (``video_id``, ``frame``, optionally ``source``);
        the first frame of a video is its smallest non-interpolated frame.
    status : mapping or None
        Output of :func:`classify_trajectories`; None counts every trajectory
        as an identity.

    Returns
    -------
    dict
        ``{'all': n, 'identity': m}`` summed over videos: ``n`` trajectories
        with ``start_frame`` > the video's first frame, ``m`` of them identities.

    Raises
    ------
    ValueError
        If a trajectory's video has no observation or it has no status.
    """
    t = trajectory_table.select(['trajectory_id', 'video_id', 'start_frame']).to_pydict()
    first = _video_first_frames(observations)
    counts = {'all': 0, 'identity': 0}
    for traj, video, start in zip(t['trajectory_id'], t['video_id'], t['start_frame']):
        if video not in first:
            raise ValueError(f"video {video!r} of {traj!r} has no observations")
        if status is not None and traj not in status:
            raise ValueError(f"trajectory {traj!r} has no status")
        if start > first[video]:
            counts['all'] += 1
            counts['identity'] += int(status is None or status[traj] == 'identity')
    return counts


def filtered_lengths(trajectory_table: pa.Table,
                     status: Mapping[str, str] | None) -> dict[str, float | int | None]:
    """Length statistics (frames, ``end - start + 1``) of the identity trajectories.

    Parameters
    ----------
    trajectory_table : pa.Table
        ``trajectory_id``, ``start_frame`` and ``end_frame`` per trajectory.
    status : mapping or None
        Output of :func:`classify_trajectories`; None keeps every trajectory.

    Returns
    -------
    dict
        ``n`` kept trajectories, ``mean`` and ``median`` length (None when
        ``n == 0``) and ``units_len1`` (kept trajectories of length 1).
    """
    t = trajectory_table.select(['trajectory_id', 'start_frame', 'end_frame']).to_pydict()
    lengths = [int(e) - int(s) + 1 for traj, s, e in
               zip(t['trajectory_id'], t['start_frame'], t['end_frame'])
               if status is None or status[traj] == 'identity']
    if not lengths:
        return {'n': 0, 'mean': None, 'median': None, 'units_len1': 0}
    arr = np.asarray(lengths, dtype=float)
    return {'n': len(lengths), 'mean': float(arr.mean()), 'median': float(np.median(arr)),
            'units_len1': int((arr == 1).sum())}
