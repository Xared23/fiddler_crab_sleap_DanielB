#!/usr/bin/env python3

"""
Endpoint motion state and gap filling for tracklet bridging (WP-A).

``endpoints`` summarises every tracklet's START and END (shell position,
clipped velocity, heading and body-frame node offsets) from the accepted
rows of an observation table. ``fill_bridge`` synthesises the missing
frames between a linked END and START: cubic Hermite on the shell path,
shortest-arc interpolation of the heading and rigid placement of the
remaining nodes. ``synthetic_rows`` renders a bridge as observation rows
with the caller's schema. Units: pixels, px/frame, int64 frames.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import math

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# ----- local modules -----
from gap_bridging.config import BridgeConfig
from gap_bridging.types import NODES, SHELL, SHOULDER, Bridge, Endpoint

# ============================================================
# CONSTANTS
# ============================================================

N_NODES = len(NODES)
# Shell->Shoulder lengths (px) accepted as a heading measurement.
MIN_SHOULDER_LENGTH_PX = 10.0
MAX_SHOULDER_LENGTH_PX = 65.0
# Columns ``endpoints`` reads; everything else is dropped before filtering.
ENDPOINT_COLUMNS = (('video_id', 'frame', 'tracklet_id', 'accepted_for_tracking', 'body_x', 'body_y')
                    + tuple(f'{node}_{axis}' for node in NODES for axis in ('x', 'y', 'visible')))

# ============================================================
# COLUMN ACCESS
# ============================================================


def _float_column(table: pa.Table, name: str) -> np.ndarray:
    """Column as float64 with NaN for nulls; all NaN when the column is absent."""
    if name not in table.column_names:
        return np.full(table.num_rows, np.nan)
    column = pc.fill_null(pc.cast(table[name], pa.float64()), math.nan)
    return np.asarray(column.to_numpy(), dtype=np.float64)


def _bool_column(table: pa.Table, name: str, default: bool) -> np.ndarray:
    """Column as bool with nulls False; ``default`` everywhere when absent."""
    if name not in table.column_names:
        return np.full(table.num_rows, default, dtype=bool)
    column = pc.fill_null(pc.cast(table[name], pa.bool_()), False)
    return np.asarray(column.to_numpy(zero_copy_only=False), dtype=bool)


def _node_arrays(table: pa.Table) -> tuple[np.ndarray, np.ndarray]:
    """Node coordinates (n,5,2) and validity (n,5): visible and finite.

    The shell is the tracked position: it falls back to ``body_x/body_y``
    where the shell coordinates are null and its validity only needs
    finite coordinates.
    """
    n = table.num_rows
    xy = np.empty((n, N_NODES, 2))
    valid = np.empty((n, N_NODES), dtype=bool)
    for j, node in enumerate(NODES):
        xy[:, j, 0] = _float_column(table, f'{node}_x')
        xy[:, j, 1] = _float_column(table, f'{node}_y')
        valid[:, j] = _bool_column(table, f'{node}_visible', True)
    body = np.stack([_float_column(table, 'body_x'), _float_column(table, 'body_y')], axis=1)
    missing = ~np.isfinite(xy[:, SHELL]).all(axis=1)
    xy[missing, SHELL] = body[missing]
    finite = np.isfinite(xy).all(axis=2)
    valid &= finite
    valid[:, SHELL] = finite[:, SHELL]
    return xy, valid


# ============================================================
# ENDPOINTS
# ============================================================


def _sorted_accepted(obs: pa.Table) -> tuple[pa.Table, np.ndarray, list[str]]:
    """Accepted rows sorted by (tracklet, frame), integer tracklet codes, id strings."""
    for name in ('video_id', 'frame', 'tracklet_id', 'accepted_for_tracking'):
        if name not in obs.column_names:
            raise ValueError(f"observation table lacks required column {name!r}")
    obs = obs.select([name for name in ENDPOINT_COLUMNS if name in obs.column_names])
    mask = pc.and_(pc.fill_null(obs['accepted_for_tracking'], False), pc.is_valid(obs['tracklet_id']))
    accepted = obs.filter(mask)
    encoded = pc.dictionary_encode(accepted['tracklet_id'].combine_chunks())
    codes = np.asarray(encoded.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
    frames = np.asarray(accepted['frame'].to_numpy(), dtype=np.int64)
    order = np.lexsort((frames, codes))
    names = encoded.dictionary.to_pylist()
    return accepted.take(pa.array(order)), codes[order], names


def _check_tracklets(codes: np.ndarray, frames: np.ndarray, videos: np.ndarray,
                     names: list[str]) -> None:
    """Raise unless every tracklet has contiguous frames and a single video."""
    same = codes[1:] == codes[:-1]
    broken = same & (np.diff(frames) != 1)
    if broken.any():
        i = int(np.flatnonzero(broken)[0])
        raise ValueError(f"tracklet {names[codes[i]]!r} has non-contiguous frames "
                         f"{int(frames[i])} -> {int(frames[i + 1])}")
    mixed = same & (videos[1:] != videos[:-1])
    if mixed.any():
        raise ValueError(f"tracklet {names[codes[int(np.flatnonzero(mixed)[0])]]!r} spans several videos")


def _row_pose(xy: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row heading (NaN if invalid) and body-frame offsets (n,5,2), NaN where unknown."""
    vector = xy[:, SHOULDER] - xy[:, SHELL]
    length = np.hypot(vector[:, 0], vector[:, 1])
    ok = valid[:, SHOULDER] & valid[:, SHELL]
    ok &= (length >= MIN_SHOULDER_LENGTH_PX) & (length <= MAX_SHOULDER_LENGTH_PX)
    heading = np.where(ok, np.arctan2(vector[:, 1], vector[:, 0]), np.nan)
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    d = xy - xy[:, SHELL:SHELL + 1]
    # R(-h) @ d = (c*dx + s*dy, -s*dx + c*dy)
    offsets = np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], axis=2)
    offsets[~(valid & ok[:, None])] = np.nan
    return heading, offsets


def _window_state(first: np.ndarray, step: int, k: np.ndarray, window: int,
                  heading: np.ndarray, offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heading (nearest valid frame) and mean offsets over each endpoint's window.

    Parameters
    ----------
    first : np.ndarray
        (m,) row index of each endpoint frame.
    step : int
        +1 walks into the tracklet from its start, -1 from its end.
    k : np.ndarray
        (m,) window depth; rows ``first + step*j`` for ``0 <= j <= k``.
    window : int
        Maximum depth (``velocity_window``).
    heading, offsets : np.ndarray
        Per-row heading (n,) and body-frame offsets (n,5,2).

    Returns
    -------
    tuple of np.ndarray
        Endpoint headings (m,) and offsets (m,5,2) with NaN rows unknown and
        the shell row zero.
    """
    m = len(first)
    out_heading = np.full(m, np.nan)
    total = np.zeros((m, N_NODES, 2))
    count = np.zeros((m, N_NODES, 1))
    for j in range(window + 1):
        inside = j <= k
        rows = np.where(inside, first + step * j, first)
        h = np.where(inside, heading[rows], np.nan)
        take = np.isnan(out_heading) & np.isfinite(h)
        out_heading[take] = h[take]
        o = offsets[rows]
        known = inside[:, None, None] & np.isfinite(o).all(axis=2, keepdims=True)
        total += np.where(known, o, 0.0)
        count += known
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    mean[:, SHELL] = 0.0
    return out_heading, mean


def _clip_speed(vel: np.ndarray, vmax: float) -> np.ndarray:
    """Scale each (m,2) velocity row down to speed ``vmax`` when faster."""
    speed = np.hypot(vel[:, 0], vel[:, 1])
    with np.errstate(invalid='ignore', divide='ignore'):
        scale = np.where(speed > vmax, vmax / np.where(speed > 0, speed, 1.0), 1.0)
    return vel * scale[:, None]


def endpoints(obs: pa.Table, cfg: BridgeConfig) -> dict[str, tuple[Endpoint, Endpoint]]:
    """Start and end motion state of every tracklet.

    Parameters
    ----------
    obs : pa.Table
        Observation table (``prediction_finalize.OBS_SCHEMA`` columns; extra
        columns ignored). Only rows with ``accepted_for_tracking`` true and a
        non-null ``tracklet_id`` are used.
    cfg : BridgeConfig
        Uses ``velocity_window`` and ``max_velocity_px_per_frame``.

    Returns
    -------
    dict
        ``tracklet_id -> (start, end)`` endpoints.

    Raises
    ------
    ValueError
        If a tracklet's frames are not contiguous or it spans several videos.
    """
    if cfg is None or cfg.velocity_window < 1:
        raise ValueError("a BridgeConfig with velocity_window >= 1 is required")
    accepted, codes, names = _sorted_accepted(obs)
    if accepted.num_rows == 0:
        return {}
    frames = np.asarray(accepted['frame'].to_numpy(), dtype=np.int64)
    videos = np.asarray(accepted['video_id'].to_numpy(zero_copy_only=False), dtype=object)
    _check_tracklets(codes, frames, videos, names)
    xy, valid = _node_arrays(accepted)
    heading, offsets = _row_pose(xy, valid)
    pos = xy[:, SHELL]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    ends = np.r_[starts[1:], len(codes)] - 1
    n = ends - starts + 1
    k = np.minimum(cfg.velocity_window, n - 1)
    kk = np.maximum(k, 1)[:, None]
    known = n >= 2
    vel_start = np.where(known[:, None], (pos[starts + k] - pos[starts]) / kk, 0.0)
    vel_end = np.where(known[:, None], (pos[ends] - pos[ends - k]) / kk, 0.0)
    vel_start = _clip_speed(vel_start, cfg.max_velocity_px_per_frame)
    vel_end = _clip_speed(vel_end, cfg.max_velocity_px_per_frame)
    h_start, o_start = _window_state(starts, 1, k, cfg.velocity_window, heading, offsets)
    h_end, o_end = _window_state(ends, -1, k, cfg.velocity_window, heading, offsets)
    result: dict[str, tuple[Endpoint, Endpoint]] = {}
    for i in range(len(starts)):
        tid, video = names[codes[starts[i]]], str(videos[starts[i]])
        common = dict(tracklet_id=tid, video_id=video, n_frames=int(n[i]), vel_known=bool(known[i]))
        result[tid] = (
            Endpoint(frame=int(frames[starts[i]]), pos=pos[starts[i]].copy(), vel=vel_start[i],
                     heading=float(h_start[i]), offsets=o_start[i], **common),
            Endpoint(frame=int(frames[ends[i]]), pos=pos[ends[i]].copy(), vel=vel_end[i],
                     heading=float(h_end[i]), offsets=o_end[i], **common))
    assert len(result) == len(names), "every accepted tracklet must yield endpoints"
    return result


# ============================================================
# INTERPOLATION PRIMITIVES
# ============================================================


def hermite(pa: np.ndarray, va: np.ndarray, pb: np.ndarray, vb: np.ndarray,
            dt: float, t: float) -> np.ndarray:
    """Cubic Hermite position between two endpoints.

    Parameters
    ----------
    pa, va : np.ndarray
        (2,) position and velocity (px/frame) at the earlier endpoint.
    pb, vb : np.ndarray
        (2,) position and velocity at the later endpoint.
    dt : float
        Frames between the endpoints (tangents are scaled by ``dt``).
    t : float
        Normalised time in [0, 1].

    Returns
    -------
    np.ndarray
        (2,) position ``h00*pa + h10*dt*va + h01*pb + h11*dt*vb``.
    """
    if not 0.0 <= t <= 1.0 or dt <= 0:
        raise ValueError(f"need 0 <= t <= 1 and dt > 0, got t={t}, dt={dt}")
    t2, t3 = t * t, t * t * t
    h00, h10 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t
    h01, h11 = -2 * t3 + 3 * t2, t3 - t2
    pa, va, pb, vb = (np.asarray(x, dtype=np.float64) for x in (pa, va, pb, vb))
    return h00 * pa + h10 * dt * va + h01 * pb + h11 * dt * vb


def wrap_angle(x: float) -> float:
    """Wrap an angle into (-pi, pi]."""
    return math.pi - (math.pi - x) % (2 * math.pi)


def interp_angle(ha: float, hb: float, t: float) -> float:
    """Shortest-arc heading ``ha + t*wrap(hb - ha)``.

    With one heading NaN the other is returned unchanged; with both NaN the
    result is NaN. The result is not wrapped (``t=0`` returns ``ha`` exactly).
    """
    a_known, b_known = math.isfinite(ha), math.isfinite(hb)
    if a_known and b_known:
        return ha + t * wrap_angle(hb - ha)
    if a_known:
        return float(ha)
    return float(hb) if b_known else math.nan


def place_nodes(pos: np.ndarray, heading: float, offsets: np.ndarray) -> np.ndarray:
    """Node positions (5,2) ``pos + R(heading) @ offset``; shell is ``pos``.

    Nodes with a NaN offset, or all non-shell nodes when the heading is NaN,
    are NaN.
    """
    pos = np.asarray(pos, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.float64)
    if pos.shape != (2,) or offsets.shape != (N_NODES, 2):
        raise ValueError(f"expected pos (2,) and offsets ({N_NODES},2), got {pos.shape}, {offsets.shape}")
    out = np.full((N_NODES, 2), np.nan)
    if math.isfinite(heading):
        c, s = math.cos(heading), math.sin(heading)
        rotation = np.array([[c, -s], [s, c]])
        out[:] = pos[None, :] + offsets @ rotation.T
        out[~np.isfinite(offsets).all(axis=1)] = np.nan
    out[SHELL] = pos
    return out


def _merge_offsets(oa: np.ndarray, ob: np.ndarray) -> np.ndarray:
    """Row-wise mean of two (5,2) offset tables; one side when the other is NaN."""
    a_ok = np.isfinite(oa).all(axis=1, keepdims=True)
    b_ok = np.isfinite(ob).all(axis=1, keepdims=True)
    both = np.where(a_ok & b_ok, 0.5 * (np.where(a_ok, oa, 0.0) + np.where(b_ok, ob, 0.0)), np.nan)
    merged = np.where(a_ok & ~b_ok, oa, np.where(b_ok & ~a_ok, ob, both))
    merged[SHELL] = 0.0
    return merged


# ============================================================
# FILL
# ============================================================


def fill_bridge(a: Endpoint, b: Endpoint, trajectory_id: str, bridge_id: str,
                cfg: BridgeConfig) -> Bridge:
    """Synthesise the missing frames between END ``a`` and START ``b``.

    Parameters
    ----------
    a, b : Endpoint
        Linked END (frame e) and START (frame s) of the same video, s - e >= 2.
    trajectory_id, bridge_id : str
        Identifiers carried on the bridge.
    cfg : BridgeConfig
        Accepted for interface symmetry; no parameter is currently used.

    Returns
    -------
    Bridge
        Frames e+1..s-1 with Hermite shell path (linear when either endpoint
        lacks a velocity), interpolated headings and rigidly placed nodes.
    """
    if a.video_id != b.video_id:
        raise ValueError(f"cannot bridge across videos {a.video_id!r} and {b.video_id!r}")
    dt = int(b.frame) - int(a.frame)
    if dt < 2:
        raise ValueError(f"bridge needs at least one missing frame, got dt={dt}")
    del cfg
    frames = np.arange(a.frame + 1, b.frame, dtype=np.int64)
    va, vb = np.asarray(a.vel, float), np.asarray(b.vel, float)
    if not (a.vel_known and b.vel_known):
        va = vb = (np.asarray(b.pos, float) - np.asarray(a.pos, float)) / dt
    offsets = _merge_offsets(np.asarray(a.offsets, float), np.asarray(b.offsets, float))
    k = len(frames)
    positions, headings = np.empty((k, 2)), np.empty(k)
    nodes = np.empty((k, N_NODES, 2))
    for i, frame in enumerate(frames):
        t = (int(frame) - a.frame) / dt
        positions[i] = hermite(a.pos, va, b.pos, vb, dt, t)
        headings[i] = interp_angle(a.heading, b.heading, t)
        nodes[i] = place_nodes(positions[i], headings[i], offsets)
    return Bridge(video_id=a.video_id, a=a, b=b, trajectory_id=trajectory_id, bridge_id=bridge_id,
                  frames=frames, positions=positions, headings=headings, nodes=nodes)


def synthetic_rows(bridge: Bridge, template_schema: pa.Schema, *,
                   fps: float | None = None) -> pa.Table:
    """One observation row per bridged frame, in ``template_schema``.

    Every column is null except ``video_id``, ``frame``, ``body_x/body_y``
    and ``centroid_x/centroid_y`` (= shell, the published centroid proxy),
    node ``*_x/*_y`` (null where unknown), node ``*_visible``
    (True where the coordinates are known, else False),
    ``centroid_source='interpolated'``, ``accepted_for_tracking=False``,
    ``incoming_reason='bridged'``, ``instance_index=-1``,
    ``raw_instance_id=-1``, ``detection_id`` and, when ``fps`` is given,
    ``time_s = frame / fps``. Node scores stay null. Columns absent from the
    template are skipped.

    Parameters
    ----------
    bridge : Bridge
        Filled bridge.
    template_schema : pa.Schema
        Schema of the observation table the rows will be appended to.
    fps : float, optional
        Frame rate for ``time_s``; None leaves ``time_s`` null.

    Returns
    -------
    pa.Table
        ``len(bridge.frames)`` rows with exactly ``template_schema``.
    """
    if fps is not None and not fps > 0:
        raise ValueError(f"fps must be positive, got {fps}")
    k = len(bridge.frames)
    if bridge.nodes.shape != (k, N_NODES, 2):
        raise ValueError(f"bridge.nodes has shape {bridge.nodes.shape}, expected ({k},{N_NODES},2)")
    frames = [int(f) for f in bridge.frames]

    def coords(values: np.ndarray) -> list:
        return [float(v) if math.isfinite(v) else None for v in values]

    values: dict[str, list] = {
        'video_id': [bridge.video_id] * k, 'frame': frames,
        'body_x': coords(bridge.positions[:, 0]), 'body_y': coords(bridge.positions[:, 1]),
        'centroid_x': coords(bridge.positions[:, 0]), 'centroid_y': coords(bridge.positions[:, 1]),
        'centroid_source': ['interpolated'] * k, 'accepted_for_tracking': [False] * k,
        'incoming_reason': ['bridged'] * k, 'instance_index': [-1] * k, 'raw_instance_id': [-1] * k,
        'detection_id': [f"{bridge.video_id}/f{f:06d}/{bridge.bridge_id}" for f in frames],
    }
    for j, node in enumerate(NODES):
        values[f'{node}_x'] = coords(bridge.nodes[:, j, 0])
        values[f'{node}_y'] = coords(bridge.nodes[:, j, 1])
        values[f'{node}_visible'] = [bool(np.isfinite(bridge.nodes[i, j]).all()) for i in range(k)]
    if fps is not None:
        values['time_s'] = [f / fps for f in frames]
    columns = [pa.array(values[f.name], type=f.type) if f.name in values else pa.nulls(k, type=f.type)
               for f in template_schema]
    table = pa.Table.from_arrays(columns, schema=template_schema)
    assert table.num_rows == k and table.schema.equals(template_schema)
    return table
