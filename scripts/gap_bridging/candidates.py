#!/usr/bin/env python3

"""
Gated one-to-one linking of tracklet END endpoints to tracklet START endpoints.

For one video, every (END a, START b) pair separated by ``1 <= gap <= max_gap``
missing frames is a candidate. Candidates pass a displacement gate and a
prediction-error gate, receive a per-frame-equivalent cost, and are assigned
one-to-one with a Hungarian solve over an ``(m+n) x (m+n)`` matrix that mirrors
``body_first_tracklets.association`` (non-candidate cells ``1e9``, dummy
diagonals ``unmatched_cost``, dummy-dummy block 0). An assigned pair whose
runner-up competitor is within ``minimum_cost_margin`` is vetoed. Every
candidate produces exactly one ``LinkDecision`` audit row.

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
from typing import Dict, List, Tuple

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


def _enumerate_pairs(ends: List[Endpoint], starts: List[Endpoint],
                     max_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enumerate candidate (end, start) index pairs in deterministic order.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Already sorted by (frame, tracklet_id).
    max_gap : int
        Largest number of missing frames, ``>= 1``.

    Returns
    -------
    (I, J) : tuple of int64 arrays
        Indices into ``ends`` and ``starts``; ends in order, then start frame
        ascending, then start tracklet_id.
    """
    assert max_gap >= 1, "max_gap must be >= 1 for enumeration"
    by_frame: Dict[int, List[int]] = defaultdict(list)
    for j, start in enumerate(starts):
        by_frame[int(start.frame)].append(j)
    rows: List[int] = []
    cols: List[int] = []
    for i, end in enumerate(ends):
        e = int(end.frame)
        for s in range(e + 2, e + 2 + max_gap):
            for j in by_frame.get(s, ()):
                if starts[j].tracklet_id != end.tracklet_id:
                    rows.append(i)
                    cols.append(j)
    assert len(rows) == len(cols)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


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
        Gate thresholds and cost weights.

    Returns
    -------
    (dt, cost, gate) : tuple of np.ndarray
        ``dt`` int64; ``cost`` float64 (finite whenever both motion gates pass);
        ``gate`` object array holding '' (passed all gates), 'displacement',
        'prediction_error' or 'max_cost'.
    """
    pa, va, ka = (x[rows] for x in _stack(ends))
    pb, vb, kb = (x[cols] for x in _stack(starts))
    dt = (np.array([s.frame for s in starts], dtype=np.int64)[cols]
          - np.array([e.frame for e in ends], dtype=np.int64)[rows])
    assert np.all(dt >= 2), "candidate pairs must have gap >= 1"
    t = dt.astype(float)[:, None]
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
                                                    cfg.shoulder_vector_cost_cap_px), 0.0)
    w = cfg.motion_weight
    cost = (1 - w) * dist / dt + w * err / dt + cfg.shoulder_vector_weight * shoulder
    disp_ok = dist <= cfg.max_body_displacement_px * dt
    err_ok = err <= cfg.max_prediction_error_px * dt
    gate = np.full(len(rows), '', dtype=object)
    gate[~(np.isfinite(cost) & (cost <= cfg.max_assignment_cost))] = GATE_MAX_COST
    gate[~err_ok] = GATE_PREDICTION
    gate[~disp_ok] = GATE_DISPLACEMENT
    return dt, cost, gate


# ============================================================
# ASSIGNMENT
# ============================================================

def _solve_component(rows: np.ndarray, cols: np.ndarray, cost: np.ndarray,
                     cfg: BridgeConfig) -> List[int]:
    """
    Hungarian solve on one component's ``(m+n) x (m+n)`` matrix.

    Parameters
    ----------
    rows, cols : np.ndarray
        Global end/start indices of the component's assignable pairs.
    cost : np.ndarray
        Costs of those pairs.
    cfg : BridgeConfig
        Supplies ``unmatched_cost``.

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
    matrix[np.arange(m), n + np.arange(m)] = cfg.unmatched_cost
    matrix[m + np.arange(n), np.arange(n)] = cfg.unmatched_cost
    matrix[m:, n:] = 0.0
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(r, c))}
    chosen: List[int] = []
    for i, j in zip(*linear_sum_assignment(matrix)):
        if i < m and j < n and (int(i), int(j)) in lookup:
            chosen.append(lookup[(int(i), int(j))])
    return chosen


def _assign(rows: np.ndarray, cols: np.ndarray, cost: np.ndarray, assignable: np.ndarray,
            m: int, cfg: BridgeConfig) -> np.ndarray:
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
        for local in _solve_component(rows[sub], cols[sub], cost[sub], cfg):
            assigned[sub[local]] = True
    assert len(set(rows[assigned])) == len(set(cols[assigned])) == int(assigned.sum())
    return assigned


# ============================================================
# DECISIONS
# ============================================================

def _runner_up(k: int, rows: np.ndarray, cols: np.ndarray, gated_cost: np.ndarray,
               by_row: Dict[int, List[int]], by_col: Dict[int, List[int]]) -> float:
    """Minimum gated cost over the other pairs sharing pair k's row or column."""
    others = [q for q in by_row[int(rows[k])] + by_col[int(cols[k])] if q != k]
    values = gated_cost[others] if others else np.empty(0)
    values = values[np.isfinite(values)]
    return float(values.min()) if values.size else math.nan


def _fmt(x: float) -> str:
    """Format a cost for reason strings."""
    return f"{x:.4f}"


def _decide(rows: np.ndarray, cols: np.ndarray, cost: np.ndarray, gate: np.ndarray,
            assigned: np.ndarray,
            cfg: BridgeConfig) -> List[Tuple[str, float, float, str]]:
    """
    Label each candidate pair.

    Parameters
    ----------
    rows, cols, cost, gate : np.ndarray
        Per-pair indices, costs and first failing gate.
    assigned : np.ndarray
        Mask of Hungarian-assigned pairs.
    cfg : BridgeConfig
        Supplies ``minimum_cost_margin``.

    Returns
    -------
    list of (decision, cost, runner_up_cost, reason)
        One entry per pair, in pair order.
    """
    motion_ok = (gate == '') | (gate == GATE_MAX_COST)
    gated_cost = np.where(motion_ok, cost, np.inf)
    by_row: Dict[int, List[int]] = defaultdict(list)
    by_col: Dict[int, List[int]] = defaultdict(list)
    win_row: Dict[int, float] = {}
    win_col: Dict[int, float] = {}
    for k in range(len(rows)):
        by_row[int(rows[k])].append(k)
        by_col[int(cols[k])].append(k)
        if assigned[k]:
            win_row[int(rows[k])] = win_col[int(cols[k])] = float(cost[k])
    out: List[Tuple[str, float, float, str]] = []
    for k in range(len(rows)):
        c = float(cost[k])
        if assigned[k]:
            ru = _runner_up(k, rows, cols, gated_cost, by_row, by_col)
            if math.isfinite(ru) and ru - c < cfg.minimum_cost_margin:
                out.append(('refused_margin', c, ru, f"cost {_fmt(c)} runner-up {_fmt(ru)} "
                            f"margin {_fmt(ru - c)} < {cfg.minimum_cost_margin:g}"))
            elif math.isfinite(ru):
                out.append(('linked', c, ru, f"cost {_fmt(c)} margin {_fmt(ru - c)}"))
            else:
                out.append(('linked', c, math.nan, f"cost {_fmt(c)} no competitor"))
        elif gate[k] == '':
            wins = [x for x in (win_row.get(int(rows[k])), win_col.get(int(cols[k])))
                    if x is not None]
            ru = min(wins) if wins else math.nan
            note = f"assignment chose cost {_fmt(ru)}" if wins else "not assigned"
            out.append(('refused_assignment', c, ru, note))
        elif gate[k] == GATE_MAX_COST:
            out.append(('refused_gate', c, math.nan, GATE_MAX_COST))
        else:
            out.append(('refused_gate', math.nan, math.nan, str(gate[k])))
    assert len(out) == len(rows)
    return out


# ============================================================
# PUBLIC API
# ============================================================

def link_tracklets(ends: List[Endpoint], starts: List[Endpoint],
                   cfg: BridgeConfig) -> Tuple[List[Tuple[str, str]], List[LinkDecision]]:
    """
    Link tracklet ENDs to STARTs of one video across 1..max_gap missing frames.

    Parameters
    ----------
    ends : list of Endpoint
        One END endpoint per tracklet (all of the same video).
    starts : list of Endpoint
        One START endpoint per tracklet (same video as ``ends``).
    cfg : BridgeConfig
        Gates, cost weights, unmatched cost and margin.

    Returns
    -------
    links : list of (tracklet_a, tracklet_b)
        One-to-one accepted links, in enumeration order.
    decisions : list of LinkDecision
        One row per enumerated pair (ends sorted by (frame, tracklet_id), then
        starts by (frame, tracklet_id)).

    Raises
    ------
    ValueError
        If the endpoints span more than one video or a side repeats a tracklet_id.
    """
    if ends is None or starts is None or cfg is None:
        raise ValueError("ends, starts and cfg are required")
    _check_inputs(ends, starts)
    if cfg.max_gap == 0 or not ends or not starts:
        return [], []
    ends = sorted(ends, key=_sort_key)
    starts = sorted(starts, key=_sort_key)
    rows, cols = _enumerate_pairs(ends, starts, cfg.max_gap)
    if rows.size == 0:
        return [], []
    dt, cost, gate = _pair_costs(ends, starts, rows, cols, cfg)
    assigned = _assign(rows, cols, cost, gate == '', len(ends), cfg)
    labels = _decide(rows, cols, cost, gate, assigned, cfg)
    links: List[Tuple[str, str]] = []
    decisions: List[LinkDecision] = []
    for k, (decision, c, ru, reason) in enumerate(labels):
        a, b = ends[rows[k]].tracklet_id, starts[cols[k]].tracklet_id
        decisions.append(LinkDecision(a, b, int(dt[k]) - 1, c, ru, decision, reason))
        if decision == 'linked':
            links.append((a, b))
    assert len({a for a, _ in links}) == len({b for _, b in links}) == len(links)
    return links, decisions
