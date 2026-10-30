#!/usr/bin/env python3

"""
Tail recovery: extend trajectories into rejected detections at their ends (v003).

A continuation-only relaxation of the Shell-score gate. Trajectories are
processed in ``(video_id, start frame, trajectory_id)`` order; for each one
the backward tail (before its START) is grown first, then the forward tail
(after its END).

Backward tail: anchor ``p = start.pos`` at frame ``start.frame`` with fixed
velocity ``v = start.vel`` (zeros unless ``vel_known``). For
``j = 1 .. max_frames`` the probed frame is ``f = start.frame - j`` and the
prediction is ``p - v * min(dt, horizon)``, with ``dt`` the number of frames
between ``f`` and the anchor's frame. Candidates are rows of
:func:`recover.candidate_index` at ``(video_id, f)`` whose ``detection_id`` is
not in ``used``, minus any candidate within ``exclusion_px`` (inclusive) of a
position occupied at ``f`` by another trajectory. The nearest candidate within
``radius_px`` (inclusive; ties by ``detection_id``) is adopted, the anchor
moves to its body position and frame, its id is added to ``used`` and its
position to the occupancy index. The tail stops at the first frame without
an adoption, after ``max_frames`` frames or below the video's first frame
(minimum ``frame`` of that video in ``obs``). The forward tail mirrors this
from the END (``p + v * min(dt, horizon)``, stopping above the video's last
frame).

Because a tail stops at its first miss, the anchor is always the previous
frame (``dt == 1``), so ``horizon`` (>= 1) never shortens the extrapolation;
it is honoured for completeness (``dt = j`` is *not* used: the anchor has
already moved to the adopted position, and extrapolating from it by ``j``
frames would overshoot by ``(j - 1) * |v|``).

Occupancy at frame ``f`` (the exclusion set) is built once from

* accepted rows of ``obs`` with a finite body position, owned by the
  trajectory of their ``tracklet_id`` (an accepted row whose tracklet is in
  no trajectory, or has a null ``tracklet_id``, counts as foreign to every
  trajectory);
* rows whose ``detection_id`` is in ``used`` on entry (the bridge recoveries
  of :func:`recover.recover_detections`), counted as foreign to every
  trajectory. This is safe because tails probe only frames outside their own
  trajectory's ``[start, end]`` span, which holds all of its own recoveries;
* adopted tail rows, added as tails grow and owned by their trajectory.

Interpolated frames of other trajectories are not rows of ``obs`` and their
positions are not known here, so they exclude nothing: a candidate at a frame
strictly inside another trajectory's span, near that trajectory's
interpolated (unrecovered) path, can still be adopted.

Output rows are labelled by the integrator: ``source='recovered_tail'``,
``bridge_id`` from :func:`tail_bridge_of`, ``gap_frames`` and
``fill_confidence`` from :func:`tail_info` (or :func:`tail_lengths` mapped
through :func:`geometry.fill_confidence`).

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

# ----- numerics -----
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# ----- local modules -----
from gap_bridging.config import BridgeConfig
from gap_bridging.geometry import fill_confidence
from gap_bridging.recover import Candidate, candidate_index
from gap_bridging.types import Endpoint

# ============================================================
# CONSTANTS
# ============================================================

TailKey = Tuple[str, int, str]   # (video_id, frame, trajectory_id)
Occupant = Tuple[Optional[str], float, float]   # (owner trajectory or None = foreign, x, y)
BACK, FWD = 'tail_back', 'tail_fwd'

# ============================================================
# IDENTIFIERS
# ============================================================


def tail_bridge_ids(trajectory_id: str) -> Tuple[str, str]:
    """Bridge ids ``(f"{tid}/tail_back", f"{tid}/tail_fwd")`` of a trajectory's two tails."""
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError(f"trajectory_id must be a non-empty string, got {trajectory_id!r}")
    return f"{trajectory_id}/{BACK}", f"{trajectory_id}/{FWD}"


def _span(members: Sequence[str],
          endpoints: Mapping[str, Tuple[Endpoint, Endpoint]]) -> Tuple[Endpoint, Endpoint]:
    """START endpoint of the first member and END endpoint of the last member."""
    if not members:
        raise ValueError("a trajectory has at least one member tracklet")
    return endpoints[members[0]][0], endpoints[members[-1]][1]


# ============================================================
# INDEXES
# ============================================================


def _video_bounds(obs: pa.Table) -> Dict[str, Tuple[int, int]]:
    """``video_id -> (first frame, last frame)`` over all rows of ``obs``."""
    grouped = obs.select(['video_id', 'frame']).filter(pc.is_valid(obs['video_id']))
    grouped = grouped.group_by('video_id').aggregate([('frame', 'min'), ('frame', 'max')])
    rows = grouped.to_pydict()
    return {v: (int(lo), int(hi))
            for v, lo, hi in zip(rows['video_id'], rows['frame_min'], rows['frame_max'])}


def _occupancy(obs: pa.Table, trajectories: Mapping[str, Sequence[str]], used: Set[str],
               keys: Set[Tuple[str, int]]) -> Dict[Tuple[str, int], List[Occupant]]:
    """Occupied body positions at the ``(video_id, frame)`` keys that hold candidates.

    Parameters
    ----------
    obs : pa.Table
        Observation table.
    trajectories : mapping
        ``trajectory_id -> member tracklet ids``.
    used : set of str
        Detection ids already assigned (bridge recoveries); foreign to every trajectory.
    keys : set of (str, int)
        Only these ``(video_id, frame)`` keys are indexed (exclusion matters only where
        a candidate exists).

    Returns
    -------
    dict
        ``(video_id, frame) -> [(owner, x, y), ...]``; ``owner`` None means foreign.
    """
    owner_of = {m: traj for traj, members in trajectories.items() for m in members}
    if not keys:
        return {}
    frames = pa.array(sorted({f for _, f in keys}), pa.int64())
    finite = pc.and_(pc.is_finite(obs['body_x']), pc.is_finite(obs['body_y']))
    accepted = pc.equal(obs['accepted_for_tracking'], True)
    taken = pc.is_in(obs['detection_id'], value_set=pa.array(sorted(used), pa.string()))
    mask = pc.and_(pc.and_(finite, pc.is_in(obs['frame'], value_set=frames)),
                   pc.or_(pc.fill_null(accepted, False), pc.fill_null(taken, False)))
    columns = ['video_id', 'frame', 'tracklet_id', 'accepted_for_tracking', 'body_x', 'body_y']
    rows = obs.select(columns).filter(pc.fill_null(mask, False)).to_pydict()
    index: Dict[Tuple[str, int], List[Occupant]] = {}
    for video, frame, tracklet, acc, x, y in zip(*(rows[c] for c in columns)):
        key = (video, int(frame))
        if key not in keys:
            continue
        owner = owner_of.get(tracklet) if acc else None
        index.setdefault(key, []).append((owner, float(x), float(y)))
    return index


# ============================================================
# TAIL GROWTH
# ============================================================


def _choose(candidates: Sequence[Candidate], occupied: Sequence[Occupant], trajectory_id: str,
            used: Set[str], pred: np.ndarray, radius_px: float,
            exclusion_px: float) -> Optional[Candidate]:
    """Nearest free, non-excluded candidate within ``radius_px`` of ``pred`` (tie: id)."""
    assert pred.shape == (2,), "prediction is one 2-D point"
    best: Optional[Tuple[float, str, Candidate]] = None
    for cand in candidates:
        det, x, y = cand
        if det in used:
            continue
        d = math.hypot(x - pred[0], y - pred[1])
        if d > radius_px or (best is not None and (d, det) >= best[:2]):
            continue
        if any(owner != trajectory_id and math.hypot(x - ox, y - oy) <= exclusion_px
               for owner, ox, oy in occupied):
            continue
        best = (d, det, cand)
    return None if best is None else best[2]


def _grow(anchor: Endpoint, step: int, limit: int, trajectory_id: str,
          index: Mapping[Tuple[str, int], List[Candidate]],
          occupancy: Dict[Tuple[str, int], List[Occupant]], used: Set[str],
          params: Tuple[float, int, float, int], out: Dict[TailKey, str]) -> None:
    """Grow one tail from ``anchor`` in direction ``step`` (-1 backward, +1 forward).

    Parameters
    ----------
    anchor : Endpoint
        START (backward) or END (forward) endpoint of the trajectory.
    step : int
        -1 or +1.
    limit : int
        First (backward) or last (forward) frame of the video; never crossed.
    trajectory_id : str
        Owner of the tail.
    index, occupancy : dict
        Candidate index and occupancy index (updated in place).
    used : set of str
        Detection ids already taken (updated in place).
    params : tuple
        ``(radius_px, max_frames, exclusion_px, horizon)``.
    out : dict
        Result, updated in place with ``(video_id, frame, trajectory_id) -> detection_id``.
    """
    assert step in (-1, 1), "step is -1 or +1"
    radius_px, max_frames, exclusion_px, horizon = params
    video = anchor.video_id
    vel = np.asarray(anchor.vel, float) if anchor.vel_known else np.zeros(2)
    p, p_frame = np.asarray(anchor.pos, float).reshape(2), int(anchor.frame)
    if not np.all(np.isfinite(p)):
        return
    if not np.all(np.isfinite(vel)):
        vel = np.zeros(2)
    for j in range(1, max_frames + 1):
        f = int(anchor.frame) + step * j
        if (step < 0 and f < limit) or (step > 0 and f > limit):
            return
        pred = p + step * vel * min(abs(f - p_frame), horizon)
        chosen = _choose(index.get((video, f), ()), occupancy.get((video, f), ()),
                         trajectory_id, used, pred, radius_px, exclusion_px)
        if chosen is None:
            return
        det, x, y = chosen
        key = (video, f, trajectory_id)
        assert key not in out, f"tail key {key} adopted twice"
        out[key] = det
        used.add(det)
        occupancy.setdefault((video, f), []).append((trajectory_id, x, y))
        p, p_frame = np.array([x, y]), f


def recover_tails(obs: pa.Table, trajectories: Mapping[str, Sequence[str]],
                  endpoints: Mapping[str, Tuple[Endpoint, Endpoint]], used: Set[str], *,
                  radius_px: float, max_frames: int, reasons: Sequence[str],
                  exclusion_px: float, horizon: int) -> Dict[TailKey, str]:
    """Adopt rejected detections that continue each trajectory beyond its ends.

    Parameters
    ----------
    obs : pa.Table
        Input observation table (published schema).
    trajectories : mapping
        :func:`chains.build_trajectories` output, ``trajectory_id -> ordered members``.
    endpoints : mapping
        ``tracklet_id -> (start, end)`` from :func:`geometry.endpoints`.
    used : set of str
        Detection ids that may not be adopted (bridge recoveries). Updated in place
        with every adopted id; its rows on entry also count as occupied by another
        trajectory for the exclusion test.
    radius_px : float
        Adoption radius around the prediction (``cfg.tail_radius_px``).
    max_frames : int
        Maximum length of each tail (``cfg.tail_max_frames``).
    reasons : sequence of str
        Rejection reasons eligible for adoption (``cfg.tail_reasons``).
    exclusion_px : float
        Candidates within this distance of another trajectory's row at the same
        frame are never adopted (``cfg.tail_exclusion_px``).
    horizon : int
        Velocity extrapolation cap in frames (``cfg.prediction_horizon_frames``).

    Returns
    -------
    dict
        ``(video_id, frame, trajectory_id) -> detection_id`` in adoption order; each
        detection at most once. Empty when ``max_frames == 0`` or ``radius_px == 0``.
    """
    if radius_px < 0 or max_frames < 0 or exclusion_px < 0 or horizon < 1:
        raise ValueError("radius_px, max_frames, exclusion_px must be >= 0 and horizon >= 1")
    if used is None:
        raise ValueError("used must be a set (pass set() when nothing is taken)")
    out: Dict[TailKey, str] = {}
    if max_frames == 0 or radius_px == 0:
        return out
    index = candidate_index(obs, reasons)
    if not index:
        return out
    occupancy = _occupancy(obs, trajectories, used, set(index))
    bounds = _video_bounds(obs)
    spans = {t: _span(m, endpoints) for t, m in trajectories.items()}
    order = sorted(spans, key=lambda t: (spans[t][0].video_id, spans[t][0].frame, t))
    params = (float(radius_px), int(max_frames), float(exclusion_px), int(horizon))
    for traj in order:
        start, end = spans[traj]
        first, last = bounds[start.video_id]
        _grow(start, -1, first, traj, index, occupancy, used, params, out)
        _grow(end, +1, last, traj, index, occupancy, used, params, out)
    assert len(set(out.values())) == len(out), "a detection was adopted twice"
    return out


# ============================================================
# LABELS FOR THE INTEGRATOR
# ============================================================


def tail_bridge_of(tails: Mapping[TailKey, str], trajectories: Mapping[str, Sequence[str]],
                   endpoints: Mapping[str, Tuple[Endpoint, Endpoint]]) -> Dict[TailKey, str]:
    """Map each tail key to its bridge id (``.../tail_back`` or ``.../tail_fwd``).

    Raises
    ------
    ValueError
        If a key's frame lies inside its trajectory's ``[start, end]`` span.
    """
    out: Dict[TailKey, str] = {}
    for key in tails:
        video, frame, traj = key
        start, end = _span(trajectories[traj], endpoints)
        back, fwd = tail_bridge_ids(traj)
        if video != start.video_id or start.frame <= frame <= end.frame:
            raise ValueError(f"tail key {key} is not outside its trajectory's span")
        out[key] = back if frame < start.frame else fwd
    return out


def tail_lengths(tails: Mapping[TailKey, str], trajectories: Mapping[str, Sequence[str]],
                 endpoints: Mapping[str, Tuple[Endpoint, Endpoint]]) -> Dict[str, int]:
    """``bridge_id -> tail length`` (frames) for every non-empty tail of ``tails``.

    Parameters
    ----------
    tails : mapping
        :func:`recover_tails` output.
    trajectories, endpoints : mapping
        The inputs given to :func:`recover_tails`.

    Returns
    -------
    dict
        Tail bridge id -> number of adopted frames (>= 1); the ``gap_frames`` of its rows.
    """
    lengths: Dict[str, int] = {}
    for bridge_id in tail_bridge_of(tails, trajectories, endpoints).values():
        lengths[bridge_id] = lengths.get(bridge_id, 0) + 1
    assert sum(lengths.values()) == len(tails), "every tail key is counted once"
    return lengths


def tail_info(tails: Mapping[TailKey, str], trajectories: Mapping[str, Sequence[str]],
              endpoints: Mapping[str, Tuple[Endpoint, Endpoint]],
              cfg: BridgeConfig) -> Dict[str, Tuple[int, str]]:
    """``bridge_id -> (tail length, geometry.fill_confidence(length, cfg))``.

    Parameters
    ----------
    tails : mapping
        :func:`recover_tails` output.
    trajectories, endpoints : mapping
        The inputs given to :func:`recover_tails`.
    cfg : BridgeConfig
        Uses ``fill_confidence_gaps``.

    Returns
    -------
    dict
        Tail bridge id -> ``(gap_frames, fill_confidence)`` for its rows.
    """
    if cfg is None:
        raise ValueError("a BridgeConfig is required")
    return {b: (n, fill_confidence(n, cfg))
            for b, n in tail_lengths(tails, trajectories, endpoints).items()}
