#!/usr/bin/env python3

"""
Gated one-to-one linking of tracklet END endpoints to tracklet START endpoints.

For one video, every (END a, START b) pair separated by ``1 <= gap <= max_gap``
missing frames is a candidate; under the sqrt model, pairs whose body distance
exceeds the largest radius any gate could reach (``_prune_radius``) are dropped
before any per-pair work and only counted. Candidates pass a displacement gate and a
prediction-error gate, receive a cost, and are assigned one-to-one with a
Hungarian solve over an ``(m+n) x (m+n)`` matrix that mirrors
``body_first_tracklets.association`` (non-candidate cells ``1e9``, dummy
diagonals ``unmatched_cost``, dummy-dummy block 0). An assigned pair whose
runner-up competitor is within ``minimum_cost_margin`` is vetoed; competitors
that are chain-reachable (a plausible successor of b, or predecessor of a,
under the motion gates) do not count as runner-ups. Every
candidate produces exactly one ``LinkDecision`` audit row.

Two gate models (``cfg.gate_model``) set the gates and the cost scale. ``linear``
is the v001 behaviour: gates ``12 dt`` and ``18 dt`` px, cost in px per frame.
``sqrt`` uses the radius ``R = cfg.gate_radius(dt) + max(|v_a|, |v_b|) min(dt, h)``
for both gates and the cost, extrapolates velocities over at most
``h = prediction_horizon_frames``, and divides by ``R`` so the cost is
dimensionless, with a ``(1 + gap_penalty)^(gap-1)`` gap penalty. The thresholds
``max_assignment_cost``, ``unmatched_cost`` and ``minimum_cost_margin`` come
from ``cfg.cost_thresholds()`` (the ``*_sqrt`` fields under sqrt). Under sqrt,
linking is staged by gap (``cfg.link_stages``, shortest first) so a short-gap
link is fixed before longer gaps compete for its endpoints; an endpoint whose
assigned pair is margin-vetoed at one stage is not linkable at later stages.

Appearance cues (sqrt only, ``cfg.appearance_cues``): the whole-tracklet
carapace size and claw side on each Endpoint add
``size_weight * size_term + claw_side_weight * side_term`` to the cost
(``_cue_terms``), and a confident claw-side mismatch refuses the pair with
gate ``claw_side`` (after the motion gates, before ``max_cost``); such a pair
is neither assignable nor chain-reachable. The linear model ignores the cues.

The assignment is solved per connected component of the assignable-pair graph;
non-candidate cells are never chosen at the optimum, so this equals the global
``(m+n) x (m+n)`` solve while keeping memory proportional to component size.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# ----- numerics -----
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ----- local modules -----
from gap_bridging.config import BridgeConfig
from gap_bridging.types import SHOULDER, Endpoint, LinkDecision

# ============================================================
# CONSTANTS
# ============================================================

BIG = 1e9
GATE_DISPLACEMENT = 'displacement'
GATE_PREDICTION = 'prediction_error'
GATE_MAX_COST = 'max_cost'
GATE_CLAW_SIDE = 'claw_side'


# ============================================================
# ENUMERATION
# ============================================================

def _sort_key(endpoint: Endpoint) -> Tuple[int, str]:
    """Deterministic ordering key: (frame, tracklet_id)."""
    return int(endpoint.frame), str(endpoint.tracklet_id)


def _check_inputs(ends: List[Endpoint], starts: List[Endpoint]) -> None:
    """Raise ValueError on mixed videos or duplicate tracklet ids per side."""
    videos = {e.video_id for e in ends} | {s.video_id for s in starts}
    if len(videos) > 1:
        raise ValueError(f"link_tracklets is per video; got {sorted(map(str, videos))}")
    for side, items in (('ends', ends), ('starts', starts)):
        ids = [e.tracklet_id for e in items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate tracklet_id among {side}")


def _prune_radius(ends: List[Endpoint], starts: List[Endpoint], cfg: BridgeConfig) -> float:
    """
    Body distance beyond which no pair can pass the displacement gate.

    Sqrt: ``R_max = r(max_gap + 1) + v * min(max_gap + 1, h)`` with
    ``v = max(max_velocity_px_per_frame, fastest known endpoint speed)``
    (pipeline endpoints are clipped to the cap, so ``v`` is the cap), which
    bounds the extended radius ``R`` of every enumerable pair because ``r`` and
    ``min(dt, h)`` are non-decreasing in ``dt``. Linear: ``inf``; the v001 audit
    table keeps a row for every pair in the time window.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Endpoints of one video.
    cfg : BridgeConfig
        Gate model, ``max_gap``, radius parameters, speed cap and horizon.

    Returns
    -------
    float
        ``R_max`` in px (``inf`` under linear).
    """
    if cfg.gate_model == 'linear':
        return math.inf
    assert cfg.gate_model == 'sqrt', f"unknown gate_model {cfg.gate_model!r}"
    speeds = [float(np.linalg.norm(np.asarray(p.vel, dtype=float)))
              for p in list(ends) + list(starts) if p.vel_known]
    v_max = max([float(cfg.max_velocity_px_per_frame)] + [v for v in speeds if math.isfinite(v)])
    dt = int(cfg.max_gap) + 1
    r_max = float(cfg.gate_radius(dt)) + v_max * min(dt, int(cfg.prediction_horizon_frames))
    assert r_max >= 0.0, "prefilter radius must be non-negative"
    return r_max


def _enumerate_pairs(ends: List[Endpoint], starts: List[Endpoint], max_gap: int,
                     r_max: float = math.inf) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Enumerate candidate (end, start) index pairs in deterministic order.

    Works per START frame: the ENDs ``2 .. max_gap + 1`` frames earlier form a
    contiguous slice of the sorted ends, and their distances to that frame's
    STARTs are one vectorised block. Pairs farther apart than ``r_max`` are
    counted but not returned.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Already sorted by (frame, tracklet_id).
    max_gap : int
        Largest number of missing frames, ``>= 1``.
    r_max : float
        Keep only pairs with body distance ``<= r_max`` (``inf`` keeps all).

    Returns
    -------
    (I, J, n_pruned) : tuple
        Int64 indices into ``ends`` and ``starts`` (ends in order, then start
        frame ascending, then start tracklet_id) and the number of
        time-window pairs dropped for distance.
    """
    assert max_gap >= 1, "max_gap must be >= 1 for enumeration"
    assert not math.isnan(r_max) and r_max >= 0.0, "r_max must be a non-negative number"
    end_frames = np.array([int(e.frame) for e in ends], dtype=np.int64)
    start_frames = np.array([int(s.frame) for s in starts], dtype=np.int64)
    end_pos = np.array([np.asarray(e.pos, dtype=float) for e in ends]).reshape(-1, 2)
    start_pos = np.array([np.asarray(s.pos, dtype=float) for s in starts]).reshape(-1, 2)
    end_ids = np.array([e.tracklet_id for e in ends], dtype=object)
    start_ids = np.array([s.tracklet_id for s in starts], dtype=object)
    frames, first = np.unique(start_frames, return_index=True)
    bounds = np.append(first, len(starts))
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    n_window = 0
    for f, lo, hi in zip(frames, bounds[:-1], bounds[1:]):
        a = int(np.searchsorted(end_frames, f - 1 - max_gap, side='left'))
        b = int(np.searchsorted(end_frames, f - 2, side='right'))
        if a >= b:
            continue
        keep = end_ids[a:b, None] != start_ids[None, lo:hi]
        n_window += int(keep.sum())
        if math.isfinite(r_max):
            dist = np.linalg.norm(start_pos[None, lo:hi] - end_pos[a:b, None], axis=2)
            keep &= dist <= r_max
        r, c = np.nonzero(keep)
        rows.append(r.astype(np.int64) + a)
        cols.append(c.astype(np.int64) + int(lo))
    I = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    J = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    order = np.lexsort((J, I))
    assert len(I) == len(J) <= n_window
    return I[order], J[order], n_window - len(I)


# ============================================================
# COSTS AND GATES
# ============================================================

def _shoulder_vectors(points: List[Endpoint]) -> np.ndarray:
    """Image-frame shoulder-shell vectors ``R(heading) @ offsets[1]``; NaN if unknown."""
    out = np.full((len(points), 2), np.nan)
    for k, p in enumerate(points):
        h = float(p.heading)
        off = np.asarray(p.offsets, dtype=float)[SHOULDER]
        if math.isfinite(h) and np.all(np.isfinite(off)):
            c, s = math.cos(h), math.sin(h)
            out[k] = (c * off[0] - s * off[1], s * off[0] + c * off[1])
    return out


def _stack(points: List[Endpoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack positions (n,2), velocities (n,2) and vel_known (n,) of endpoints."""
    pos = np.array([np.asarray(p.pos, dtype=float) for p in points]).reshape(-1, 2)
    vel = np.array([np.asarray(p.vel, dtype=float) for p in points]).reshape(-1, 2)
    known = np.array([bool(p.vel_known) for p in points], dtype=bool)
    assert pos.shape == vel.shape == (len(points), 2)
    return pos, vel, known


def _pair_geometry(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
                   cols: np.ndarray, horizon: Optional[int],
                   cap: float) -> Tuple[np.ndarray, ...]:
    """
    Frame separation, displacement, prediction error and shoulder change per pair.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Sorted endpoints.
    rows, cols : np.ndarray
        Candidate indices from ``_enumerate_pairs``.
    horizon : int or None
        Frames over which endpoint velocities are extrapolated, capped at
        ``dt``; ``None`` extrapolates over the full ``dt`` (linear model).
    cap : float
        Shoulder-vector change cap in px.

    Returns
    -------
    (dt, dist, err, shoulder) : tuple of np.ndarray
        ``dt`` int64 frames between END and START; ``dist`` ``|p_b - p_a|``;
        ``err`` mean forward/backward prediction error over the known
        velocities (``dist`` when neither is known); ``shoulder``
        ``min(|u_a - u_b|, cap)`` in px, 0 when either vector is unknown.
    """
    pa, va, ka = (x[rows] for x in _stack(ends))
    pb, vb, kb = (x[cols] for x in _stack(starts))
    dt = (np.array([s.frame for s in starts], dtype=np.int64)[cols]
          - np.array([e.frame for e in ends], dtype=np.int64)[rows])
    assert np.all(dt >= 2), "candidate pairs must have gap >= 1"
    steps = dt if horizon is None else np.minimum(dt, int(horizon))
    t = steps.astype(float)[:, None]
    dist = np.linalg.norm(pb - pa, axis=1)
    forward = np.linalg.norm(pb - (pa + va * t), axis=1)
    backward = np.linalg.norm(pa - (pb - vb * t), axis=1)
    n_known = ka.astype(float) + kb.astype(float)
    summed = np.where(ka, forward, 0.0) + np.where(kb, backward, 0.0)
    err = np.where(n_known > 0, summed / np.maximum(n_known, 1.0), dist)
    ua = _shoulder_vectors(ends)[rows]
    ub = _shoulder_vectors(starts)[cols]
    du = np.linalg.norm(ua - ub, axis=1)
    shoulder = np.where(np.isfinite(du), np.minimum(np.nan_to_num(du, nan=0.0),
                                                    cap), 0.0)
    return dt, dist, err, shoulder


def _gate_labels(cost: np.ndarray, disp_ok: np.ndarray, err_ok: np.ndarray,
                 max_cost: float, side_veto: Optional[np.ndarray] = None) -> np.ndarray:
    """First failing gate per pair: displacement, prediction error, claw side, max cost."""
    gate = np.full(len(cost), '', dtype=object)
    gate[~(np.isfinite(cost) & (cost <= max_cost))] = GATE_MAX_COST
    if side_veto is not None:
        gate[np.asarray(side_veto, dtype=bool)] = GATE_CLAW_SIDE
    gate[~err_ok] = GATE_PREDICTION
    gate[~disp_ok] = GATE_DISPLACEMENT
    return gate


def _pair_costs_linear(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
                       cols: np.ndarray, cfg: BridgeConfig) -> Tuple[np.ndarray, ...]:
    """
    Legacy (v001) gates and per-frame-equivalent cost, unchanged.

    Gates ``dist <= max_body_displacement_px * dt`` and
    ``err <= max_prediction_error_px * dt`` (velocities extrapolated over the
    full ``dt``); cost ``(1-w) dist/dt + w err/dt + w_s min(|du|, cap)`` in px
    per frame; ``max_cost`` gate at ``cfg.max_assignment_cost``.

    Returns
    -------
    (dt, cost, gate) : tuple of np.ndarray
        As for ``_pair_costs``.
    """
    dt, dist, err, shoulder = _pair_geometry(ends, starts, rows, cols, None,
                                             cfg.shoulder_vector_cost_cap_px)
    w = cfg.motion_weight
    cost = (1 - w) * dist / dt + w * err / dt + cfg.shoulder_vector_weight * shoulder
    disp_ok = dist <= cfg.max_body_displacement_px * dt
    err_ok = err <= cfg.max_prediction_error_px * dt
    return dt, cost, _gate_labels(cost, disp_ok, err_ok, cfg.max_assignment_cost)


def _sqrt_extended_radius(cfg: BridgeConfig, dt: np.ndarray, speed: np.ndarray) -> np.ndarray:
    """
    Extended search radius of the sqrt model, ``R = r(dt) + speed * min(dt, h)``.

    The Brownian term ``r(dt) = cfg.gate_radius(dt)`` is extended by directed
    motion over the capped horizon ``h = cfg.prediction_horizon_frames``
    (u-track "directed + Brownian"), so a crab walking into a gap still passes.

    Parameters
    ----------
    cfg : BridgeConfig
        Radius parameters and prediction horizon.
    dt : np.ndarray
        Frames between END and START per pair.
    speed : np.ndarray
        ``max(|v_a|, |v_b|)`` in px/frame per pair (an endpoint whose velocity
        is unknown contributes 0), which keeps ``R`` time-reversal symmetric.

    Returns
    -------
    np.ndarray
        Radius ``R`` in px per pair.
    """
    dt = np.asarray(dt)
    steps = np.minimum(dt, int(cfg.prediction_horizon_frames)).astype(float)
    out = np.asarray(cfg.gate_radius(dt), dtype=float) + np.asarray(speed, dtype=float) * steps
    assert out.shape == dt.shape, "radius must be per pair"
    return out


def _cue_terms(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
               cols: np.ndarray, cfg: BridgeConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Appearance cue terms of each candidate pair (END ``a = ends[rows]``, START ``b = starts[cols]``).

    With carapace sizes ``L`` over ``n`` frames and mean claw sides ``s`` over
    ``n_s`` frames (``Endpoint.size_px/size_n/claw_side/claw_side_n``):

    * ``size_term = min(size_cost_cap, |L_a - L_b| / sqrt(size_scale_px^2
      + (size_noise_px^2/n_a + size_noise_px^2/n_b)))``, 0 unless both ``n > 0``;
    * ``side_term = max(0, -s_a s_b)``, 0 unless both ``n_s > 0``;
    * ``side_veto`` iff both ``|s| >= claw_side_confidence``, both
      ``n_s >= claw_side_min_frames`` and ``s_a s_b < 0``.

    All three are symmetric under exchanging the cue values of a and b.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Sorted endpoints.
    rows, cols : np.ndarray
        Candidate indices from ``_enumerate_pairs``.
    cfg : BridgeConfig
        ``size_*`` and ``claw_side_*`` parameters.

    Returns
    -------
    (size_term, side_term, side_veto) : tuple of np.ndarray
        Float terms in ``[0, size_cost_cap]`` and ``[0, 1]`` and a bool veto mask.
    """
    def cues(points: List[Endpoint], idx: np.ndarray) -> Tuple[np.ndarray, ...]:
        size = np.array([float(p.size_px) for p in points], dtype=float).reshape(-1)[idx]
        size_n = np.array([int(p.size_n) for p in points], dtype=np.int64).reshape(-1)[idx]
        side = np.array([float(p.claw_side) for p in points], dtype=float).reshape(-1)[idx]
        side_n = np.array([int(p.claw_side_n) for p in points], dtype=np.int64).reshape(-1)[idx]
        return size, size_n, side, side_n

    la, na, sa, nsa = cues(ends, rows)
    lb, nb, sb, nsb = cues(starts, cols)
    sized = (na > 0) & (nb > 0) & np.isfinite(la) & np.isfinite(lb)
    noise2 = float(cfg.size_noise_px) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        spread = np.sqrt(float(cfg.size_scale_px) ** 2 + (noise2 / na + noise2 / nb))
        z = np.abs(la - lb) / spread
    size_term = np.where(sized, np.minimum(float(cfg.size_cost_cap), np.nan_to_num(z, nan=0.0)), 0.0)
    sided = (nsa > 0) & (nsb > 0)
    product = sa * sb
    side_term = np.where(sided, np.maximum(0.0, -product), 0.0)
    conf, need = float(cfg.claw_side_confidence), int(cfg.claw_side_min_frames)
    side_veto = ((np.abs(sa) >= conf) & (np.abs(sb) >= conf) & (nsa >= need) & (nsb >= need)
                 & (product < 0))
    assert size_term.shape == side_term.shape == side_veto.shape == rows.shape
    return size_term, side_term, side_veto


def _cue_columns(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
                 cols: np.ndarray, cfg: BridgeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """``(size_term, side_term)`` for the audit rows; NaN under linear or with cues off."""
    if cfg.gate_model != 'sqrt' or not cfg.appearance_cues:
        return np.full(len(rows), np.nan), np.full(len(rows), np.nan)
    size_term, side_term, _ = _cue_terms(ends, starts, rows, cols, cfg)
    return size_term, side_term


def _pair_costs_sqrt(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
                     cols: np.ndarray, cfg: BridgeConfig) -> Tuple[np.ndarray, ...]:
    """
    Speed-extended sub-linear radius, capped horizon, dimensionless cost.

    With ``h = cfg.prediction_horizon_frames`` and the extended radius
    ``R = cfg.gate_radius(dt) + max(|v_a|, |v_b|) min(dt, h)``: gates
    ``dist <= R`` and ``err <= prediction_error_ratio * R`` (velocities
    extrapolated over ``min(dt, h)`` frames); cost
    ``[(1-w) dist + w err] / R * (1 + gap_penalty)^(dt-2) + w_s min(|du|, cap)/cap``,
    so a 1-frame gap (``dt = 2``) carries no penalty; with
    ``cfg.appearance_cues`` the cost gains
    ``size_weight * size_term + claw_side_weight * side_term`` and a
    ``side_veto`` pair fails the ``claw_side`` gate (``_cue_terms``);
    ``max_cost`` gate at ``cfg.cost_thresholds()['max_assignment_cost']``.

    Returns
    -------
    (dt, cost, gate) : tuple of np.ndarray
        As for ``_pair_costs``.
    """
    cap = cfg.shoulder_vector_cost_cap_px
    dt, dist, err, shoulder = _pair_geometry(ends, starts, rows, cols,
                                             cfg.prediction_horizon_frames, cap)
    _, va, ka = (x[rows] for x in _stack(ends))
    _, vb, kb = (x[cols] for x in _stack(starts))
    speed = np.maximum(np.where(ka, np.linalg.norm(va, axis=1), 0.0),
                       np.where(kb, np.linalg.norm(vb, axis=1), 0.0))
    radius = _sqrt_extended_radius(cfg, dt, speed)
    penalty = (1.0 + cfg.gap_penalty) ** (dt.astype(float) - 2.0)
    w = cfg.motion_weight
    with np.errstate(divide='ignore', invalid='ignore'):
        motion = ((1 - w) * dist + w * err) / radius * penalty
        shape = shoulder / cap if cap > 0 else np.zeros_like(shoulder)
    cost = motion + cfg.shoulder_vector_weight * shape
    side_veto = None
    if cfg.appearance_cues:
        size_term, side_term, side_veto = _cue_terms(ends, starts, rows, cols, cfg)
        cost = cost + cfg.size_weight * size_term + cfg.claw_side_weight * side_term
    disp_ok = dist <= radius
    err_ok = err <= cfg.prediction_error_ratio * radius
    max_cost = cfg.cost_thresholds()['max_assignment_cost']
    return dt, cost, _gate_labels(cost, disp_ok, err_ok, max_cost, side_veto)


def _pair_costs(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
                cols: np.ndarray, cfg: BridgeConfig) -> Tuple[np.ndarray, ...]:
    """
    Compute dt, cost and the first failing gate for each candidate pair.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Sorted endpoints.
    rows, cols : np.ndarray
        Candidate indices from ``_enumerate_pairs``.
    cfg : BridgeConfig
        Gate model, gate thresholds and cost weights.

    Returns
    -------
    (dt, cost, gate) : tuple of np.ndarray
        ``dt`` int64; ``cost`` float64 (finite whenever both motion gates pass);
        ``gate`` object array holding '' (passed all gates), 'displacement',
        'prediction_error', 'claw_side' (sqrt with cues only) or 'max_cost'.
        A 'claw_side' pair keeps its finite cost here; its audit row reports
        NaN. ``cost`` is px per frame under the
        linear model and a fraction of the extended radius under the sqrt model.
    """
    if cfg.gate_model == 'linear':
        return _pair_costs_linear(ends, starts, rows, cols, cfg)
    assert cfg.gate_model == 'sqrt', f"unknown gate_model {cfg.gate_model!r}"
    return _pair_costs_sqrt(ends, starts, rows, cols, cfg)


# ============================================================
# ASSIGNMENT
# ============================================================

def _solve_component(rows: np.ndarray, cols: np.ndarray, cost: np.ndarray,
                     unmatched_cost: float) -> List[int]:
    """
    Hungarian solve on one component's ``(m+n) x (m+n)`` matrix.

    Parameters
    ----------
    rows, cols : np.ndarray
        Global end/start indices of the component's assignable pairs.
    cost : np.ndarray
        Costs of those pairs.
    unmatched_cost : float
        Dummy-diagonal cost of leaving an end or a start unmatched.

    Returns
    -------
    list of int
        Positions (into ``rows``) of the pairs assigned to each other.
    """
    end_ids, r = np.unique(rows, return_inverse=True)
    start_ids, c = np.unique(cols, return_inverse=True)
    m, n = len(end_ids), len(start_ids)
    assert m >= 1 and n >= 1
    matrix = np.full((m + n, m + n), BIG, dtype=float)
    matrix[r, c] = cost
    matrix[np.arange(m), n + np.arange(m)] = unmatched_cost
    matrix[m + np.arange(n), np.arange(n)] = unmatched_cost
    matrix[m:, n:] = 0.0
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(r, c))}
    chosen: List[int] = []
    for i, j in zip(*linear_sum_assignment(matrix)):
        if i < m and j < n and (int(i), int(j)) in lookup:
            chosen.append(lookup[(int(i), int(j))])
    return chosen


def _assign(rows: np.ndarray, cols: np.ndarray, cost: np.ndarray, assignable: np.ndarray,
            m: int, unmatched_cost: float) -> np.ndarray:
    """Boolean mask of pairs chosen by the one-to-one assignment (per component)."""
    assigned = np.zeros(len(rows), dtype=bool)
    idx = np.flatnonzero(assignable)
    if idx.size == 0:
        return assigned
    n_nodes = m + int(cols.max()) + 1
    graph = coo_matrix((np.ones(idx.size), (rows[idx], m + cols[idx])),
                       shape=(n_nodes, n_nodes))
    _, labels = connected_components(graph, directed=False)
    groups: Dict[int, List[int]] = defaultdict(list)
    for k in idx:
        groups[int(labels[rows[k]])].append(int(k))
    for members in groups.values():
        sub = np.asarray(members, dtype=np.int64)
        for local in _solve_component(rows[sub], cols[sub], cost[sub], unmatched_cost):
            assigned[sub[local]] = True
    assert len(set(rows[assigned])) == len(set(cols[assigned])) == int(assigned.sum())
    return assigned


# ============================================================
# DECISIONS
# ============================================================

def _runner_up(k: int, rows: np.ndarray, cols: np.ndarray, gated_cost: np.ndarray,
               by_row: Dict[int, List[int]], by_col: Dict[int, List[int]],
               chain: Tuple[np.ndarray, np.ndarray, Set[Tuple[str, str]]]) -> Tuple[float, int]:
    """
    Runner-up cost for assigned pair k over non-chain-reachable competitors.

    Parameters
    ----------
    k : int
        Index of the assigned pair (a, b).
    rows, cols, gated_cost : np.ndarray
        Pair indices and costs (``inf`` where a motion gate failed).
    by_row, by_col : dict
        Pair indices grouped by end index and by start index.
    chain : tuple
        (end tracklet ids, start tracklet ids, set of motion-gated
        (end_id, start_id) pairs). A row competitor (a, c) is ignored when
        (b, c) is motion-gated; a column competitor (d, b) is ignored when
        (d, a) is motion-gated.

    Returns
    -------
    (runner_up, n_ignored) : tuple of (float, int)
        Minimum competitor cost (NaN if none) and number of motion-gated
        competitors ignored as chain-reachable.
    """
    end_ids, start_ids, gated = chain
    a, b = end_ids[rows[k]], start_ids[cols[k]]
    kept: List[float] = []
    ignored = 0
    for q in by_row[int(rows[k])] + by_col[int(cols[k])]:
        if q == k or not math.isfinite(float(gated_cost[q])):
            continue
        if rows[q] == rows[k]:
            reachable = (b, start_ids[cols[q]]) in gated
        else:
            reachable = (end_ids[rows[q]], a) in gated
        if reachable:
            ignored += 1
        else:
            kept.append(float(gated_cost[q]))
    assert ignored + len(kept) <= len(by_row[int(rows[k])]) + len(by_col[int(cols[k])])
    return (min(kept) if kept else math.nan), ignored


def _fmt(x: float) -> str:
    """Format a cost for reason strings."""
    return f"{x:.4f}"


def _motion_gated(rows: np.ndarray, cols: np.ndarray, gate: np.ndarray,
                  ids: Tuple[np.ndarray, np.ndarray]) -> Set[Tuple[str, str]]:
    """(end_id, start_id) pairs passing the motion gates and the claw-side veto (``''`` or ``max_cost``)."""
    motion_ok = (gate == '') | (gate == GATE_MAX_COST)
    end_ids, start_ids = ids
    return {(end_ids[r], start_ids[c]) for r, c in zip(rows[motion_ok], cols[motion_ok])}


def _decide(rows: np.ndarray, cols: np.ndarray, cost: np.ndarray, gate: np.ndarray,
            assigned: np.ndarray, ids: Tuple[np.ndarray, np.ndarray],
            margin: float, gated: Set[Tuple[str, str]]) -> List[Tuple[str, float, float, str, int]]:
    """
    Label each candidate pair.

    Parameters
    ----------
    rows, cols, cost, gate : np.ndarray
        Per-pair indices, costs and first failing gate.
    assigned : np.ndarray
        Mask of Hungarian-assigned pairs.
    ids : tuple of np.ndarray
        Tracklet ids of the sorted ends and of the sorted starts; used to find
        chain-reachable competitors, which do not count toward the runner-up.
    margin : float
        ``minimum_cost_margin`` on the gate model's cost scale.
    gated : set of (str, str)
        Motion-gated (end_id, start_id) pairs over all gaps up to ``max_gap``
        (``_motion_gated``); defines chain reachability.

    Returns
    -------
    list of (decision, cost, runner_up_cost, reason, n_chain_ignored)
        One entry per pair, in pair order; ``n_chain_ignored`` counts the
        chain-reachable competitors excluded from the runner-up of an
        assigned pair (0 for unassigned pairs).
    """
    motion_ok = (gate == '') | (gate == GATE_MAX_COST)
    gated_cost = np.where(motion_ok, cost, np.inf)
    end_ids, start_ids = ids
    by_row: Dict[int, List[int]] = defaultdict(list)
    by_col: Dict[int, List[int]] = defaultdict(list)
    win_row: Dict[int, float] = {}
    win_col: Dict[int, float] = {}
    for k in range(len(rows)):
        by_row[int(rows[k])].append(k)
        by_col[int(cols[k])].append(k)
        if assigned[k]:
            win_row[int(rows[k])] = win_col[int(cols[k])] = float(cost[k])
    out: List[Tuple[str, float, float, str, int]] = []
    for k in range(len(rows)):
        c = float(cost[k])
        if assigned[k]:
            ru, n_chain = _runner_up(k, rows, cols, gated_cost, by_row, by_col,
                                     (end_ids, start_ids, gated))
            note = f"; {n_chain} chain competitors ignored" if n_chain else ""
            if math.isfinite(ru) and ru - c < margin:
                out.append(('refused_margin', c, ru, f"cost {_fmt(c)} runner-up {_fmt(ru)} "
                            f"margin {_fmt(ru - c)} < {margin:g}{note}", n_chain))
            elif math.isfinite(ru):
                out.append(('linked', c, ru, f"cost {_fmt(c)} margin {_fmt(ru - c)}{note}", n_chain))
            else:
                out.append(('linked', c, math.nan, f"cost {_fmt(c)} no competitor{note}", n_chain))
        elif gate[k] == '':
            wins = [x for x in (win_row.get(int(rows[k])), win_col.get(int(cols[k])))
                    if x is not None]
            ru = min(wins) if wins else math.nan
            note = f"assignment chose cost {_fmt(ru)}" if wins else "not assigned"
            out.append(('refused_assignment', c, ru, note, 0))
        elif gate[k] == GATE_MAX_COST:
            out.append(('refused_gate', c, math.nan, GATE_MAX_COST, 0))
        else:
            out.append(('refused_gate', math.nan, math.nan, str(gate[k]), 0))
    assert len(out) == len(rows)
    return out


# ============================================================
# PUBLIC API
# ============================================================

def _stages(cfg: BridgeConfig) -> Tuple[int, ...]:
    """Upper gap bound of each linking stage: ``(max_gap,)`` for linear, staged for sqrt."""
    if cfg.gate_model == 'linear':
        return (int(cfg.max_gap),)
    stages = sorted({int(s) for s in cfg.link_stages if s < cfg.max_gap} | {int(cfg.max_gap)})
    assert stages[-1] == cfg.max_gap and stages[0] >= 1
    return tuple(stages)


def _excluded_label(k: int, cost: np.ndarray, gate: np.ndarray, rows: np.ndarray,
                    cols: np.ndarray, done: Tuple[Dict[int, Tuple[int, float]], ...],
                    vetoed: Tuple[Dict[int, Tuple[int, float]], ...]
                    ) -> Tuple[str, float, float, str, int]:
    """
    Label of a pair whose END or START was linked or margin-vetoed at an earlier stage.

    Parameters
    ----------
    k : int
        Pair index.
    cost, gate, rows, cols : np.ndarray
        Per-pair costs, first failing gates and end/start indices.
    done, vetoed : tuple of two dict
        ``(by end index, by start index) -> (stage, cost)`` of the earlier
        links and of the earlier ``refused_margin`` pairs.

    Returns
    -------
    tuple
        ``refused_gate`` when a gate fails; otherwise ``refused_assignment``
        with reason ``endpoint linked at stage s`` (that link's cost as the
        runner-up), or, when no endpoint was linked, ``endpoint vetoed at stage
        s`` (the vetoed pair's cost as the runner-up).
    """
    c = float(cost[k])
    if gate[k] == GATE_MAX_COST:
        return 'refused_gate', c, math.nan, GATE_MAX_COST, 0
    if gate[k] != '':
        return 'refused_gate', math.nan, math.nan, str(gate[k]), 0
    for label, (by_end, by_start) in (('linked', done), ('vetoed', vetoed)):
        earlier = [x for x in (by_end.get(int(rows[k])), by_start.get(int(cols[k])))
                   if x is not None]
        if earlier:
            stage = min(s for s, _ in earlier)
            ru = min(v for _, v in earlier)
            return 'refused_assignment', c, ru, f"endpoint {label} at stage {stage}", 0
    raise AssertionError("excluded pair must touch an endpoint linked or vetoed earlier")


def link_tracklets(ends: List[Endpoint], starts: List[Endpoint], cfg: BridgeConfig,
                   stats: Optional[Dict[str, int]] = None
                   ) -> Tuple[List[Tuple[str, str]], List[LinkDecision]]:
    """
    Link tracklet ENDs to STARTs of one video across 1..max_gap missing frames.

    Linking runs in stages of increasing gap (``_stages``): stage k assigns the
    pairs with ``stage_{k-1} < gap <= stage_k`` whose END and START are both
    still free, fixes its links, and removes the matched endpoints before
    stage k+1. An assigned pair refused by the margin veto removes its END and
    START too: the veto means "ambiguous, leave unlinked", so neither endpoint
    is retried at a later stage. Gates and costs are per pair and
    stage-independent; runner-ups are taken over the stage's own candidates,
    and chain reachability over all motion-gated pairs up to ``max_gap``
    (claw-side-vetoed pairs excluded). The linear model has a single stage.

    Parameters
    ----------
    ends : list of Endpoint
        One END endpoint per tracklet (all of the same video).
    starts : list of Endpoint
        One START endpoint per tracklet (same video as ``ends``).
    cfg : BridgeConfig
        Gate model, gates, cost weights and ``link_stages``. The unmatched cost
        and margin are taken from ``cfg.cost_thresholds()`` (the px-scale
        values under the linear model, the ``*_sqrt`` values under sqrt).
    stats : dict, optional
        Replaced in place by ``{'enumerated': n, 'pruned_by_radius': m}``:
        ``n`` pairs received an audit row, ``m`` pairs inside the time window
        were dropped for body distance ``> R_max`` (``_prune_radius``; always
        0 under linear).

    Returns
    -------
    links : list of (tracklet_a, tracklet_b)
        One-to-one accepted links, in enumeration order.
    decisions : list of LinkDecision
        One row per enumerated pair (ends sorted by (frame, tracklet_id), then
        starts by (frame, tracklet_id)); pairs beyond ``R_max`` have no row.
        ``stage`` is the 1-based stage whose gap range holds the pair. A
        gate-passing pair whose END or START was linked at an earlier stage is
        ``refused_assignment`` with reason ``endpoint linked at stage s`` and
        that link's cost as runner-up; if instead an endpoint was margin-vetoed
        at an earlier stage, the reason is ``endpoint vetoed at stage s`` and
        the runner-up is the vetoed pair's cost. A pair refused by the
        claw-side veto is ``refused_gate`` with reason ``claw_side`` and NaN
        cost. ``size_term``/``side_term`` hold the cue terms added to the
        cost (NaN under linear or with ``appearance_cues`` off).

    Raises
    ------
    ValueError
        If the endpoints span more than one video or a side repeats a tracklet_id.
    """
    if ends is None or starts is None or cfg is None:
        raise ValueError("ends, starts and cfg are required")
    _check_inputs(ends, starts)
    if stats is not None:
        stats.clear()
        stats.update(enumerated=0, pruned_by_radius=0)
    if cfg.max_gap == 0 or not ends or not starts:
        return [], []
    ends = sorted(ends, key=_sort_key)
    starts = sorted(starts, key=_sort_key)
    rows, cols, n_pruned = _enumerate_pairs(ends, starts, cfg.max_gap,
                                            _prune_radius(ends, starts, cfg))
    if stats is not None:
        stats.update(enumerated=int(rows.size), pruned_by_radius=int(n_pruned))
    if rows.size == 0:
        return [], []
    scale = cfg.cost_thresholds()
    dt, cost, gate = _pair_costs(ends, starts, rows, cols, cfg)
    size_term, side_term = _cue_columns(ends, starts, rows, cols, cfg)
    ids = (np.array([e.tracklet_id for e in ends], dtype=object),
           np.array([b.tracklet_id for b in starts], dtype=object))
    gated = _motion_gated(rows, cols, gate, ids)
    stages = _stages(cfg)
    stage_of = np.searchsorted(np.asarray(stages), dt - 1, side='left') + 1
    assert stage_of.min() >= 1 and stage_of.max() <= len(stages)
    labels: List[Tuple[str, float, float, str, int]] = [None] * len(rows)  # type: ignore
    done_end: Dict[int, Tuple[int, float]] = {}
    done_start: Dict[int, Tuple[int, float]] = {}
    vetoed_end: Dict[int, Tuple[int, float]] = {}
    vetoed_start: Dict[int, Tuple[int, float]] = {}
    for stage in range(1, len(stages) + 1):
        in_stage = stage_of == stage
        free = np.array([int(r) not in done_end and int(c) not in done_start
                         and int(r) not in vetoed_end and int(c) not in vetoed_start
                         for r, c in zip(rows, cols)], dtype=bool)
        for k in np.flatnonzero(in_stage & ~free):
            labels[k] = _excluded_label(int(k), cost, gate, rows, cols, (done_end, done_start),
                                        (vetoed_end, vetoed_start))
        idx = np.flatnonzero(in_stage & free)
        if idx.size == 0:
            continue
        sub_rows, sub_cols, sub_cost, sub_gate = rows[idx], cols[idx], cost[idx], gate[idx]
        assigned = _assign(sub_rows, sub_cols, sub_cost, sub_gate == '', len(ends),
                           scale['unmatched_cost'])
        sub = _decide(sub_rows, sub_cols, sub_cost, sub_gate, assigned, ids,
                      scale['minimum_cost_margin'], gated)
        for k, label in zip(idx, sub):
            labels[k] = label
            if label[0] == 'linked':
                done_end[int(rows[k])] = done_start[int(cols[k])] = (stage, float(cost[k]))
            elif label[0] == 'refused_margin':
                vetoed_end[int(rows[k])] = vetoed_start[int(cols[k])] = (stage, float(cost[k]))
    assert all(label is not None for label in labels)
    links: List[Tuple[str, str]] = []
    decisions: List[LinkDecision] = []
    for k, (decision, c, ru, reason, n_chain) in enumerate(labels):
        a, b = ends[rows[k]].tracklet_id, starts[cols[k]].tracklet_id
        decisions.append(LinkDecision(a, b, int(dt[k]) - 1, c, ru, decision, reason, int(n_chain),
                                      int(stage_of[k]), float(size_term[k]), float(side_term[k])))
        if decision == 'linked':
            links.append((a, b))
    assert len({a for a, _ in links}) == len({b for _, b in links}) == len(links)
    return links, decisions
