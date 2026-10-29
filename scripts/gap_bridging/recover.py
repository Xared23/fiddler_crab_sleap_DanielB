"""Adopt previously rejected real detections into bridged gaps.

For every missing frame of every bridge, the candidates are rows of that video
and frame with ``accepted_for_tracking == False``, a ``rejection_reason`` in
``cfg.recoverable_reasons`` and a finite body position, not already adopted by
an earlier bridge. The candidate nearest to the bridge's interpolated shell
position ``p(f)`` is adopted when its distance is ``<= cfg.recovery_radius_px``
(ties broken by ``detection_id``). Bridges are processed in
(video_id, first gap frame, trajectory_id) order and frames in increasing
order. The interpolated path is not refit after a recovery.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from gap_bridging.config import BridgeConfig
from gap_bridging.types import Bridge

RecoveryKey = tuple[str, int, str]   # (video_id, frame, trajectory_id)
Candidate = tuple[str, float, float]  # (detection_id, body_x, body_y)


# --------------------------------------------------------------------------- #
# Candidate index
# --------------------------------------------------------------------------- #
def candidate_index(obs: pa.Table, reasons: Iterable[str]) -> dict[tuple[str, int], list[Candidate]]:
    """Index recoverable rejected rows by (video_id, frame).

    Parameters
    ----------
    obs : pa.Table
        Observation table with ``video_id``, ``frame``, ``detection_id``,
        ``accepted_for_tracking``, ``rejection_reason``, ``body_x``, ``body_y``.
    reasons : iterable of str
        Recoverable rejection reasons.

    Returns
    -------
    dict
        ``(video_id, frame) -> [(detection_id, x, y), ...]`` sorted by
        detection_id. Rows with null ``accepted_for_tracking`` are excluded.
    """
    reasons = list(reasons)
    columns = ['video_id', 'frame', 'detection_id', 'body_x', 'body_y']
    if not reasons or obs.num_rows == 0:
        return {}
    mask = pc.and_(
        pc.equal(obs['accepted_for_tracking'], False),
        pc.is_in(obs['rejection_reason'], value_set=pa.array(reasons, pa.string())))
    mask = pc.and_(mask, pc.and_(pc.is_finite(obs['body_x']), pc.is_finite(obs['body_y'])))
    rows = obs.select(columns).filter(pc.fill_null(mask, False)).to_pydict()
    index: dict[tuple[str, int], list[Candidate]] = {}
    for video, frame, det, x, y in zip(*(rows[c] for c in columns)):
        if det is None:
            raise ValueError(f"recoverable row at {video}/{frame} has a null detection_id")
        index.setdefault((video, int(frame)), []).append((det, float(x), float(y)))
    for candidates in index.values():
        candidates.sort(key=lambda c: c[0])
    return index


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #
def bridge_order(bridges: Iterable[Bridge]) -> list[Bridge]:
    """Sort bridges by (video_id, first gap frame, trajectory_id, bridge_id)."""
    def key(bridge: Bridge) -> tuple[str, int, str, str]:
        first = int(bridge.frames[0]) if len(bridge.frames) else int(bridge.a.frame) + 1
        return (bridge.video_id, first, bridge.trajectory_id, bridge.bridge_id)
    return sorted(bridges, key=key)


def recover_detections(obs: pa.Table, bridges: list[Bridge], cfg: BridgeConfig) -> dict[RecoveryKey, str]:
    """Choose, per bridged frame, a rejected real detection to adopt.

    Parameters
    ----------
    obs : pa.Table
        Input observation table (published schema).
    bridges : list of Bridge
        Filled bridges carrying the interpolated shell path.
    cfg : BridgeConfig
        Uses ``recoverable_reasons`` and ``recovery_radius_px``.

    Returns
    -------
    dict
        ``(video_id, frame, trajectory_id) -> detection_id``; each detection is
        adopted at most once.
    """
    index = candidate_index(obs, cfg.recoverable_reasons)
    radius = float(cfg.recovery_radius_px)
    used: set[str] = set()
    out: dict[RecoveryKey, str] = {}
    for bridge in bridge_order(bridges):
        positions = np.asarray(bridge.positions, dtype=float).reshape(len(bridge.frames), 2)
        for frame, (px, py) in zip(bridge.frames, positions):
            frame = int(frame)
            key = (bridge.video_id, frame, bridge.trajectory_id)
            if key in out:
                raise ValueError(f"two bridges cover {key}")
            if not (math.isfinite(px) and math.isfinite(py)):
                continue
            best: tuple[float, str] | None = None
            for det, x, y in index.get((bridge.video_id, frame), ()):
                if det in used:
                    continue
                d = math.hypot(x - px, y - py)
                if d <= radius and (best is None or (d, det) < best):
                    best = (d, det)
            if best is not None:
                used.add(best[1])
                out[key] = best[1]
    return out


def bridge_of_recovered(bridges: Iterable[Bridge], recovered: dict[RecoveryKey, str]) -> dict[RecoveryKey, str]:
    """Map each recovery key to the ``bridge_id`` of the bridge that covers it.

    Raises
    ------
    KeyError
        If a recovery key is not covered by any bridge.
    """
    cover: dict[RecoveryKey, str] = {}
    for bridge in bridges:
        for frame in bridge.frames:
            cover[(bridge.video_id, int(frame), bridge.trajectory_id)] = bridge.bridge_id
    return {key: cover[key] for key in recovered}
