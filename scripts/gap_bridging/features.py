#!/usr/bin/env python3

"""
Per-pair feature export for the learned link scorer (v004).

One row per enumerated candidate pair, in the enumeration order of
:func:`gap_bridging.candidates.link_tracklets` (ENDs sorted by (frame,
tracklet_id), then STARTs likewise), with the same R_max prefilter. The
features are the quantities the hand cost is built from, kept separate and
unweighted, plus tracklet-level context (lengths, cue sample sizes, rival
counts). A rival of a pair is another pair from the same END (or into the
same START) that passes both motion gates and is not claw-side vetoed (first
failing gate ``''`` or ``'max_cost'``: the pairs the assignment can see).
Rival counts use the pair's own ``dt`` as the window, so every value is
identical at every horizon that enumerates the pair; this is what lets the
trainer deduplicate pairs across horizons.

Feature values (and the gate set) depend on the config keys of
:data:`FEATURE_CONFIG_KEYS`; :func:`feature_config` extracts them so that a
scorer can be tied to the configuration its training features came from.

Features are always computed under the sqrt gate model with the hand cost
(``cost_model='hand'``), whatever ``cfg.cost_model`` says, so a table written
by a learned-cost run carries the same columns as a hand-cost run. Notes on
individual columns:

* ``shoulder`` is the dimensionless ``min(|du|, cap) / cap`` that the sqrt
  cost multiplies by ``shoulder_vector_weight`` (``candidates._pair_geometry``
  returns the capped value in px); like the cost it is 0, not NaN, when either
  shoulder vector is unknown.
* ``size_term`` and ``side_term`` are the hand cue terms of
  ``candidates._cue_terms_full`` (the side term carries the claw-frame
  confidence weight and is 0 on an axis flip), but NaN where a side has no
  size (claw-side) frames, where the hand cost uses 0. They are computed even
  when ``cfg.appearance_cues`` is off; ``hand_cost`` then omits them.
* ``hand_cost`` is the sqrt cost of every pair, finite also where a gate
  failed; ``links.parquet`` reports NaN for gate-refused pairs (including
  ``claw_side``), this column keeps the value.
* Ratios over a zero radius (only with ``radius_min_px = 0`` and a zero
  fitted radius) are NaN rather than inf.

Feature subsets (v005). A scorer may be trained on an ordered subset of
:data:`FEATURE_NAMES` (:data:`FEATURE_SETS`, :func:`resolve_feature_set`);
its manifest records the names and inference selects those columns by name
(:func:`check_feature_subset`, :func:`select_features`). ``no_length`` drops
the tracklet-length features of :data:`LENGTH_FEATURES` (the v004 scorer
learned tracklet length, D-009); ``no_length_no_speed`` also drops
:data:`SPEED_FEATURES`. Neither blinds the model to length, so the ablation
is a control, not a fix: ``speed_*`` are 0 when the velocity is unknown
(1-frame tracklets); ``radius`` is built from ``max(speed_a, speed_b)``, so
it (and the ratios over it) still sees unknown velocities; ``abs_side_*`` is
a mean of signs over ``claw_side_n`` frames and so takes only a few discrete
values when ``n <= 3`` (1 for one frame, 0 or 1 for two, 1/3 or 1 for
three); ``size_term``'s noise term ``size_noise_px^2 / n`` and
``side_term``'s confidence factor ``min(1, n / claw_side_min_frames)``
depend on the frame counts, and ``hand_cost`` carries all of these.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

# ----- numerics -----
import numpy as np
import pyarrow as pa

# ----- local modules -----
from gap_bridging import candidates, geometry
from gap_bridging.config import BridgeConfig
from gap_bridging.types import Endpoint

# ============================================================
# CONSTANTS
# ============================================================

#: Feature columns in matrix order. All float64; NaN where unknown.
FEATURE_NAMES: Tuple[str, ...] = (
    'gap',                 # dt - 1 missing frames
    'dist',                # |pos_b - pos_a| px
    'err',                 # prediction error (both-sided, _pair_geometry) px
    'radius',              # extended radius R = r(dt) + speed * min(dt, h) px
    'dist_over_radius',    # dist / R
    'err_over_radius',     # err / R
    'speed_a', 'speed_b',  # |vel| px/frame (0 when unknown)
    'vel_known_a', 'vel_known_b',   # 1.0 / 0.0
    'shoulder',            # min(|du|, cap) / cap, shoulder-vector term in [0, 1]
    'heading_change',      # |wrap(h_b - h_a)| rad; NaN if a heading is NaN
    'size_term',           # hand size term; NaN if a side has size_n == 0
    'size_diff',           # |L_a - L_b| px; NaN likewise
    'side_product',        # claw_side_a * claw_side_b; NaN if a side has claw_side_n == 0
    'side_term',           # max(0, -side_product); NaN likewise
    'abs_side_a', 'abs_side_b',     # |claw_side|; NaN when unknown
    'size_n_a', 'size_n_b',
    'claw_side_n_a', 'claw_side_n_b',
    'n_frames_a', 'n_frames_b',
    'n_rival_starts',      # other gate-passing pairs (gate '' or 'max_cost') from the same END, dt' <= dt
    'n_rival_ends',        # other gate-passing pairs (gate '' or 'max_cost') into the same START, dt' <= dt
    'axis_flip',           # 1.0 if _axis_flip (Shoulder axis reversed, claw fixed), else 0.0
    'hand_cost',           # the sqrt hand cost (finite when both motion gates pass)
)

#: Monotonic constraint per feature for the scorer: -1 means P(true link) is
#: non-increasing in the feature; 0 means unconstrained.
MONOTONIC: dict = {name: -1 for name in ('gap', 'dist', 'err', 'dist_over_radius',
                                          'err_over_radius', 'shoulder', 'size_term',
                                          'side_term', 'hand_cost')}

#: Tracklet-length features (frame counts and what is 1:1 with ``n >= 2``),
#: dropped by the ``no_length`` ablation (D-009).
LENGTH_FEATURES: Tuple[str, ...] = ('n_frames_a', 'n_frames_b', 'size_n_a', 'size_n_b',
                                    'claw_side_n_a', 'claw_side_n_b', 'vel_known_a', 'vel_known_b')
#: Endpoint speeds (0 when the velocity is unknown), also dropped by ``no_length_no_speed``.
SPEED_FEATURES: Tuple[str, ...] = ('speed_a', 'speed_b')

#: Named feature subsets, each in :data:`FEATURE_NAMES` order.
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    'all': FEATURE_NAMES,
    'no_length': tuple(n for n in FEATURE_NAMES if n not in LENGTH_FEATURES),
    'no_length_no_speed': tuple(n for n in FEATURE_NAMES
                                if n not in LENGTH_FEATURES + SPEED_FEATURES),
}

#: Key columns identifying the pair (prefix of :data:`PAIR_FEATURES_SCHEMA`).
KEY_NAMES: Tuple[str, ...] = ('video_id', 'tracklet_a', 'tracklet_b', 'gap_frames', 'gate')

PAIR_FEATURES_SCHEMA = pa.schema(
    [('video_id', pa.string()), ('tracklet_a', pa.string()), ('tracklet_b', pa.string()),
     ('gap_frames', pa.int64()), ('gate', pa.string())]
    + [(name, pa.float64()) for name in FEATURE_NAMES])

PAIR_FEATURES_NAME = 'pair_features.parquet'

#: ``BridgeConfig`` keys that change a feature value or the gate column
#: (motion gates, radius, prediction, cue terms and the hand cost). A scorer
#: records them (``ScorerManifest.feature_config``) and refuses a config
#: that differs. ``max_gap`` is absent: features do not depend on the horizon.
FEATURE_CONFIG_KEYS: Tuple[str, ...] = (
    'velocity_window', 'max_velocity_px_per_frame', 'motion_weight', 'shoulder_vector_weight',
    'shoulder_vector_cost_cap_px', 'gate_model', 'radius_r0_px', 'radius_k_px', 'radius_safety',
    'radius_min_px', 'prediction_error_ratio', 'prediction_horizon_frames', 'gap_penalty',
    'appearance_cues', 'size_scale_px', 'size_noise_px', 'size_cost_cap', 'size_weight',
    'claw_side_weight', 'claw_side_confidence', 'claw_side_min_frames')


# ============================================================
# FEATURE SUBSETS
# ============================================================


def check_feature_subset(names: Sequence[str]) -> Tuple[int, ...]:
    """
    Column indices of an ordered subset of :data:`FEATURE_NAMES`.

    Parameters
    ----------
    names : sequence of str
        Feature names in the order a scorer expects them.

    Returns
    -------
    tuple of int
        Strictly increasing indices into :data:`FEATURE_NAMES`;
        ``FEATURE_NAMES`` itself gives ``range(28)``.

    Raises
    ------
    ValueError
        If ``names`` is empty, holds an unknown or repeated name, or is not in
        :data:`FEATURE_NAMES` order (a permuted list would feed the model the
        wrong columns).
    """
    if isinstance(names, str):
        raise ValueError(f"feature names must be a sequence of names, got the string {names!r}")
    names = tuple(names)
    if not names:
        raise ValueError("feature names must not be empty")
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown feature names {unknown}; known: {list(FEATURE_NAMES)}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"repeated feature names {repeated}")
    idx = tuple(FEATURE_NAMES.index(n) for n in names)
    if any(b <= a for a, b in zip(idx, idx[1:])):
        expected = [n for n in FEATURE_NAMES if n in names]
        raise ValueError(f"feature names {list(names)} are not in FEATURE_NAMES order; "
                         f"expected {expected}")
    assert len(idx) == len(names)
    return idx


def resolve_feature_set(spec: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """
    The feature names of a subset specification, in :data:`FEATURE_NAMES` order.

    Parameters
    ----------
    spec : str or sequence of str
        A :data:`FEATURE_SETS` key, a comma-separated list of feature names
        (any order; blanks around names ignored), or a sequence of names.

    Returns
    -------
    tuple of str
        The names, reordered as in :data:`FEATURE_NAMES`.

    Raises
    ------
    ValueError
        If the spec is empty or names an unknown or repeated feature.
    """
    if isinstance(spec, str):
        if spec in FEATURE_SETS:
            return FEATURE_SETS[spec]
        names = tuple(part.strip() for part in spec.split(','))
        if any(not n for n in names):
            raise ValueError(f"empty feature name in {spec!r}; give one of "
                             f"{sorted(FEATURE_SETS)} or a comma-separated list of names")
    else:
        names = tuple(spec)
    if not names:
        raise ValueError(f"empty feature set; give one of {sorted(FEATURE_SETS)} or names")
    unknown = [n for n in names if n not in FEATURE_NAMES]
    if unknown:
        raise ValueError(f"unknown feature names {unknown}; feature sets are "
                         f"{sorted(FEATURE_SETS)}, names are {list(FEATURE_NAMES)}")
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"repeated feature names {repeated}")
    out = tuple(n for n in FEATURE_NAMES if n in names)
    assert len(out) == len(names)
    return out


def monotonic_for(names: Sequence[str]) -> Tuple[int, ...]:
    """Monotonic constraint per name (:data:`MONOTONIC`, 0 when unconstrained)."""
    return tuple(int(MONOTONIC.get(n, 0)) for n in names)


def select_features(X: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """
    The columns ``names`` of a full feature matrix.

    Parameters
    ----------
    X : np.ndarray
        ``(n, len(FEATURE_NAMES))`` in :data:`FEATURE_NAMES` order.
    names : sequence of str
        An ordered subset (:func:`check_feature_subset`).

    Returns
    -------
    np.ndarray
        ``(n, len(names))`` float64 copy.

    Raises
    ------
    ValueError
        If ``X`` is not ``(n, 28)`` or ``names`` is not an ordered subset.
    """
    idx = check_feature_subset(names)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected X of shape (n, {len(FEATURE_NAMES)}), got {X.shape}")
    out = X[:, list(idx)].copy()
    assert out.shape == (X.shape[0], len(idx))
    return out


# ============================================================
# HELPERS
# ============================================================


def feature_config(cfg: Any) -> dict:
    """
    The feature-relevant subset of a configuration.

    Parameters
    ----------
    cfg : BridgeConfig or mapping
        A config, or its ``to_dict()`` (e.g. the ``config`` of a
        ``holdout_<h>.json``).

    Returns
    -------
    dict
        ``{key: value}`` for every key of :data:`FEATURE_CONFIG_KEYS`, in
        that order; tuples as lists (as ``BridgeConfig.to_dict``).

    Raises
    ------
    ValueError
        If a mapping lacks one of the keys.
    """
    values = dict(cfg) if isinstance(cfg, Mapping) else cfg.to_dict()
    missing = [key for key in FEATURE_CONFIG_KEYS if key not in values]
    if missing:
        raise ValueError(f"config lacks feature keys {missing}")
    out = {key: (list(values[key]) if isinstance(values[key], tuple) else values[key])
           for key in FEATURE_CONFIG_KEYS}
    assert len(out) == len(FEATURE_CONFIG_KEYS)
    return out


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """``num / den`` elementwise; NaN where the quotient is not finite."""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.asarray(num, dtype=float) / np.asarray(den, dtype=float)
    return np.where(np.isfinite(out), out, np.nan)


def _side_cues(points: List[Endpoint], idx: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Per-pair ``(size_px, size_n, claw_side, claw_side_n, n_frames)`` of one side."""
    size = np.array([float(p.size_px) for p in points], dtype=float).reshape(-1)[idx]
    size_n = np.array([int(p.size_n) for p in points], dtype=float).reshape(-1)[idx]
    side = np.array([float(p.claw_side) for p in points], dtype=float).reshape(-1)[idx]
    side_n = np.array([int(p.claw_side_n) for p in points], dtype=float).reshape(-1)[idx]
    n_frames = np.array([int(p.n_frames) for p in points], dtype=float).reshape(-1)[idx]
    return size, size_n, side, side_n, n_frames


def _rival_counts(group: np.ndarray, dt: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """
    Number of other gate-passing pairs sharing a group with ``dt' <= dt``.

    Parameters
    ----------
    group : np.ndarray
        ``(n,)`` int64 END (or START) index of each pair.
    dt : np.ndarray
        ``(n,)`` int64 frame separation, ``>= 2``.
    ok : np.ndarray
        ``(n,)`` bool, the pair passes both motion gates and is not claw-side
        vetoed (first failing gate ``''`` or ``'max_cost'``); only such pairs
        count as rivals (a vetoed pair can never take the endpoint).

    Returns
    -------
    np.ndarray
        ``(n,)`` float64 count over pairs ``q != k`` with ``group[q] ==
        group[k]``, ``ok[q]`` and ``dt[q] <= dt[k]``; ``O(n log n)`` through
        one sorted key ``group * span + dt``.
    """
    assert group.shape == dt.shape == ok.shape
    if group.size == 0:
        return np.zeros(0, dtype=float)
    span = int(dt.max()) + 1
    key = group.astype(np.int64) * span + dt.astype(np.int64)
    gated = np.sort(key[ok])
    upto = np.searchsorted(gated, key, side='right')
    first = np.searchsorted(gated, group.astype(np.int64) * span, side='left')
    count = upto - first - ok.astype(np.int64)
    assert np.all(count >= 0), "rival count must be non-negative"
    return count.astype(float)


# ============================================================
# FEATURES
# ============================================================


def pair_features(ends: List[Endpoint], starts: List[Endpoint], rows: np.ndarray,
                  cols: np.ndarray, cfg: BridgeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Feature matrix and first failing motion gate for candidate pairs.

    Parameters
    ----------
    ends, starts : list of Endpoint
        Sorted as in ``link_tracklets`` (``candidates._sort_key``).
    rows, cols : np.ndarray
        Pair indices into ``ends`` and ``starts`` (``candidates._enumerate_pairs``).
    cfg : BridgeConfig
        Gate and cue parameters; evaluated with ``cost_model='hand'``.

    Returns
    -------
    X : np.ndarray
        ``(n, len(FEATURE_NAMES))`` float64 in :data:`FEATURE_NAMES` order;
        NaN where a value is unknown, never inf.
    gate : np.ndarray
        ``(n,)`` object: first failing gate per pair under the hand cost (''
        when none fails; 'displacement', 'prediction_error', 'claw_side' or
        'max_cost').

    Raises
    ------
    ValueError
        If ``cfg.gate_model`` is not ``'sqrt'`` or ``rows``/``cols`` differ in
        length.
    """
    if cfg.gate_model != 'sqrt':
        raise ValueError(f"pair features need gate_model 'sqrt', got {cfg.gate_model!r}")
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if rows.shape != cols.shape or rows.ndim != 1:
        raise ValueError("rows and cols must be 1-d arrays of equal length")
    n = rows.size
    out = np.full((n, len(FEATURE_NAMES)), np.nan)
    if n == 0:
        return out, np.full(0, '', dtype=object)
    cfg_hand = replace(cfg, cost_model='hand')
    cap = float(cfg_hand.shoulder_vector_cost_cap_px)
    dt, dist, err, shoulder_px = candidates._pair_geometry(
        ends, starts, rows, cols, cfg_hand.prediction_horizon_frames, cap)
    _, va, ka = (x[rows] for x in candidates._stack(ends))
    _, vb, kb = (x[cols] for x in candidates._stack(starts))
    speed_a = np.where(ka, np.linalg.norm(va, axis=1), 0.0)
    speed_b = np.where(kb, np.linalg.norm(vb, axis=1), 0.0)
    radius = candidates._sqrt_extended_radius(cfg_hand, dt, np.maximum(speed_a, speed_b))
    dt_cost, hand_cost, gate = candidates._pair_costs_sqrt(ends, starts, rows, cols, cfg_hand)
    assert np.array_equal(dt, dt_cost), "geometry and cost disagree on dt"

    heading_a = np.array([float(p.heading) for p in ends], dtype=float).reshape(-1)[rows]
    heading_b = np.array([float(p.heading) for p in starts], dtype=float).reshape(-1)[cols]
    heading_change = np.abs(geometry.wrap_angle(heading_b - heading_a))

    size_a, size_n_a, side_a, side_n_a, n_frames_a = _side_cues(ends, rows)
    size_b, size_n_b, side_b, side_n_b, n_frames_b = _side_cues(starts, cols)
    size_term, side_term, _, _ = candidates._cue_terms_full(ends, starts, rows, cols, cfg_hand)
    sized = (size_n_a > 0) & (size_n_b > 0) & np.isfinite(size_a) & np.isfinite(size_b)
    known_a, known_b = side_n_a > 0, side_n_b > 0
    sided = known_a & known_b

    ok = (gate == '') | (gate == candidates.GATE_MAX_COST)
    values = {
        'gap': (dt - 1).astype(float),
        'dist': dist,
        'err': err,
        'radius': radius,
        'dist_over_radius': _ratio(dist, radius),
        'err_over_radius': _ratio(err, radius),
        'speed_a': speed_a,
        'speed_b': speed_b,
        'vel_known_a': ka.astype(float),
        'vel_known_b': kb.astype(float),
        'shoulder': (shoulder_px / cap) if cap > 0 else np.zeros(n),
        'heading_change': heading_change,
        'size_term': np.where(sized, size_term, np.nan),
        'size_diff': np.where(sized, np.abs(size_a - size_b), np.nan),
        'side_product': np.where(sided, side_a * side_b, np.nan),
        'side_term': np.where(sided, side_term, np.nan),
        'abs_side_a': np.where(known_a, np.abs(side_a), np.nan),
        'abs_side_b': np.where(known_b, np.abs(side_b), np.nan),
        'size_n_a': size_n_a,
        'size_n_b': size_n_b,
        'claw_side_n_a': side_n_a,
        'claw_side_n_b': side_n_b,
        'n_frames_a': n_frames_a,
        'n_frames_b': n_frames_b,
        'n_rival_starts': _rival_counts(rows, dt, ok),
        'n_rival_ends': _rival_counts(cols, dt, ok),
        'axis_flip': candidates._axis_flip(ends, starts, rows, cols).astype(float),
        'hand_cost': np.where(np.isfinite(hand_cost), hand_cost, np.nan),
    }
    assert set(values) == set(FEATURE_NAMES), "every feature must be computed"
    for j, name in enumerate(FEATURE_NAMES):
        column = np.asarray(values[name], dtype=float)
        assert column.shape == (n,), f"feature {name} must be per pair"
        out[:, j] = column
    assert out.shape == (n, len(FEATURE_NAMES))
    assert not np.isinf(out).any(), "features must not be infinite"
    return out, gate


def pair_feature_table(ends: List[Endpoint], starts: List[Endpoint], cfg: BridgeConfig,
                       video_id: str) -> pa.Table:
    """
    Feature rows of every candidate pair of one video, in ``link_tracklets`` order.

    Parameters
    ----------
    ends, starts : list of Endpoint
        One END and one START per tracklet of the video, in any order.
    cfg : BridgeConfig
        ``max_gap``, gate and cue parameters (sqrt gate model).
    video_id : str
        Written to the ``video_id`` column.

    Returns
    -------
    pa.Table
        :data:`PAIR_FEATURES_SCHEMA`; one row per pair enumerated by
        ``link_tracklets`` (same R_max prefilter, same order), so row k
        matches decision k. ``tracklet_a`` is the END's id, ``tracklet_b``
        the START's, ``gap_frames = dt - 1``. Empty when ``max_gap == 0``
        or no pair is enumerated.

    Raises
    ------
    ValueError
        As ``link_tracklets`` (mixed videos, duplicate ids per side), or if
        the gate model is not sqrt.
    """
    if ends is None or starts is None or cfg is None:
        raise ValueError("ends, starts and cfg are required")
    candidates._check_inputs(ends, starts)
    if cfg.gate_model != 'sqrt':
        raise ValueError(f"pair features need gate_model 'sqrt', got {cfg.gate_model!r}")
    if cfg.max_gap == 0 or not ends or not starts:
        return PAIR_FEATURES_SCHEMA.empty_table()
    ends = sorted(ends, key=candidates._sort_key)
    starts = sorted(starts, key=candidates._sort_key)
    rows, cols, _ = candidates._enumerate_pairs(ends, starts, cfg.max_gap,
                                                candidates._prune_radius(ends, starts, cfg))
    if rows.size == 0:
        return PAIR_FEATURES_SCHEMA.empty_table()
    X, gate = pair_features(ends, starts, rows, cols, cfg)
    gap = X[:, FEATURE_NAMES.index('gap')].astype(np.int64)
    assert np.all((gap >= 1) & (gap <= cfg.max_gap)), "enumerated gaps must lie in 1..max_gap"
    columns = {
        'video_id': pa.array([str(video_id)] * rows.size, type=pa.string()),
        'tracklet_a': pa.array([str(ends[r].tracklet_id) for r in rows], type=pa.string()),
        'tracklet_b': pa.array([str(starts[c].tracklet_id) for c in cols], type=pa.string()),
        'gap_frames': pa.array(gap, type=pa.int64()),
        'gate': pa.array([str(g) for g in gate], type=pa.string()),
    }
    for j, name in enumerate(FEATURE_NAMES):
        columns[name] = pa.array(X[:, j], type=pa.float64())
    table = pa.Table.from_pydict(columns, schema=PAIR_FEATURES_SCHEMA)
    assert table.num_rows == rows.size
    return table


def feature_matrix(table: pa.Table, names: Sequence[str] = FEATURE_NAMES) -> np.ndarray:
    """
    ``(n, len(names))`` float64 matrix of the named columns of a features table.

    Parameters
    ----------
    table : pa.Table
        A :data:`PAIR_FEATURES_SCHEMA` table (or any table with the columns).
    names : sequence of str
        Columns in matrix order (default :data:`FEATURE_NAMES`).

    Returns
    -------
    np.ndarray
        Float64 matrix; nulls become NaN.

    Raises
    ------
    ValueError
        If a named column is missing.
    """
    names = list(names)
    missing = [name for name in names if name not in table.schema.names]
    if missing:
        raise ValueError(f"features table has no column(s) {missing}")
    out = np.full((table.num_rows, len(names)), np.nan)
    for j, name in enumerate(names):
        column = table[name].cast(pa.float64()).fill_null(math.nan)
        out[:, j] = column.to_numpy() if table.num_rows else np.zeros(0)
    assert out.shape == (table.num_rows, len(names))
    return out
