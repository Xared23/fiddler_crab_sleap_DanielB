#!/usr/bin/env python3

"""
Tests for the v004 assignment options and the learned cost in gap_bridging.candidates.

Covers ``assignment='global'`` (one stage over gaps 1..max_gap, equal to a
staged run with a single stage), the adaptive dummy diagonal
(``unmatched_cost_mode='adaptive'``: per component, ``adaptive_factor`` times
the largest finite cost), the learned cost model (``-log p`` from a scorer,
with the motion gates and the claw-side veto unchanged; v005: a scorer on an
ordered feature subset gets exactly those columns), and the refactor of
the sqrt gate terms (``_sqrt_gate_terms``), which must leave
``_pair_costs_sqrt`` and the linear path numerically unchanged.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import List, Optional, Tuple
import unittest
from unittest import mock

# ----- numerics -----
from hypothesis import example, given, settings, strategies as st
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

# ----- local modules -----
from gap_bridging import candidates, features, geometry  # noqa: E402
from gap_bridging.candidates import link_tracklets  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.types import DECISIONS, SHOULDER, Endpoint  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
LINEAR = BridgeConfig(gate_model='linear', max_gap=5)
SQRT = BridgeConfig(gate_model='sqrt', max_gap=5, radius_r0_px=0.0, radius_k_px=3.73, radius_safety=1.25,
                    radius_min_px=6.0, prediction_error_ratio=1.5, prediction_horizon_frames=3,
                    gap_penalty=0.06, max_assignment_cost_sqrt=1.0, unmatched_cost_sqrt=0.6,
                    minimum_cost_margin_sqrt=0.25)    # pinned so example numbers survive refits
LEARNED = dict(cost_model='learned', learned_min_probability=math.exp(-1.0),
               learned_unmatched_probability=math.exp(-0.6), learned_margin=0.25)
HAND_COST = features.FEATURE_NAMES.index('hand_cost')
GATE_REASONS = {'displacement', 'prediction_error', 'max_cost', 'claw_side'}
EXCLUDED = 'endpoint linked at stage '
VETOED = 'endpoint vetoed at stage '
COMBOS = [(a, u) for a in ('staged', 'global') for u in ('fixed', 'adaptive')]

# ============================================================
# HELPERS
# ============================================================


def ep(tid: str, frame: int, pos: Tuple[float, float], vel: Tuple[float, float] = (0., 0.),
       vel_known: bool = True, heading: float = math.nan,
       shoulder: Optional[Tuple[float, float]] = None, video: str = 'v',
       claw: Optional[Tuple[float, float]] = None) -> Endpoint:
    """Build an Endpoint; ``shoulder`` and ``claw`` are body-frame node offsets."""
    offsets = np.full((5, 2), np.nan)
    offsets[0] = 0.0
    if shoulder is not None:
        offsets[1] = shoulder
    if claw is not None:
        offsets[candidates.CLAW] = claw
    return Endpoint(tid, video, int(frame), 5 if vel_known else 1,
                    np.asarray(pos, dtype=float), np.asarray(vel, dtype=float),
                    bool(vel_known), float(heading), offsets)


def same_decision(x, y) -> bool:
    """Field-by-field equality of two LinkDecisions, NaN equal to NaN, floats exact."""
    def eq(u, v):
        if isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v):
            return True
        return type(u) is type(v) and u == v
    return all(eq(getattr(x, f), getattr(y, f)) for f in x.__dataclass_fields__)


def by_pair(decisions) -> dict:
    """Index decisions by (tracklet_a, tracklet_b)."""
    return {(d.tracklet_a, d.tracklet_b): d for d in decisions}


def chain_free_runner_up(decisions, a: str, b: str) -> float:
    """Independent runner-up for (a, b) over non-chain-reachable same-stage competitors
    (copied from test_gap_bridging_candidates)."""
    gated = {(d.tracklet_a, d.tracklet_b) for d in decisions if math.isfinite(d.cost)}
    stage = {(d.tracklet_a, d.tracklet_b): d.stage for d in decisions}[(a, b)]
    taken = ('linked', 'refused_margin')
    end_at = {d.tracklet_a: d.stage for d in decisions if d.decision in taken}
    start_at = {d.tracklet_b: d.stage for d in decisions if d.decision in taken}
    values = [d.cost for d in decisions
              if math.isfinite(d.cost) and d.stage == stage
              and end_at.get(d.tracklet_a, stage) >= stage
              and start_at.get(d.tracklet_b, stage) >= stage
              and ((d.tracklet_a == a and d.tracklet_b != b and (b, d.tracklet_b) not in gated)
                   or (d.tracklet_b == b and d.tracklet_a != a and (d.tracklet_a, a) not in gated))]
    return min(values) if values else math.nan


def sorted_pairs(ends: List[Endpoint], starts: List[Endpoint], cfg: BridgeConfig):
    """Sorted endpoints and the enumerated (rows, cols) as ``link_tracklets`` builds them."""
    ends = sorted(ends, key=candidates._sort_key)
    starts = sorted(starts, key=candidates._sort_key)
    rows, cols, _ = candidates._enumerate_pairs(ends, starts, cfg.max_gap,
                                                candidates._prune_radius(ends, starts, cfg))
    return ends, starts, rows, cols


def legacy_pair_costs_sqrt(ends, starts, rows, cols, cfg):
    """Verbatim copy of ``candidates._pair_costs_sqrt`` before the v004 refactor (76bac9f)."""
    cap = cfg.shoulder_vector_cost_cap_px
    dt, dist, err, shoulder = candidates._pair_geometry(ends, starts, rows, cols,
                                                        cfg.prediction_horizon_frames, cap)
    _, va, ka = (x[rows] for x in candidates._stack(ends))
    _, vb, kb = (x[cols] for x in candidates._stack(starts))
    speed = np.maximum(np.where(ka, np.linalg.norm(va, axis=1), 0.0),
                       np.where(kb, np.linalg.norm(vb, axis=1), 0.0))
    radius = candidates._sqrt_extended_radius(cfg, dt, speed)
    penalty = (1.0 + cfg.gap_penalty) ** (dt.astype(float) - 2.0)
    w = cfg.motion_weight
    with np.errstate(divide='ignore', invalid='ignore'):
        motion = ((1 - w) * dist + w * err) / radius * penalty
        shape = shoulder / cap if cap > 0 else np.zeros_like(shoulder)
    cost = motion + cfg.shoulder_vector_weight * shape
    side_veto = None
    if cfg.appearance_cues:
        size_term, side_term, side_veto = candidates._cue_terms(ends, starts, rows, cols, cfg)
        cost = cost + cfg.size_weight * size_term + cfg.claw_side_weight * side_term
    disp_ok = dist <= radius
    err_ok = err <= cfg.prediction_error_ratio * radius
    max_cost = cfg.cost_thresholds()['max_assignment_cost']
    return dt, cost, candidates._gate_labels(cost, disp_ok, err_ok, max_cost, side_veto)


# ----- frozen v003 linear cost path (f6840ff: scripts/gap_bridging/candidates.py) -----
def v003_shoulder_vectors(points: List[Endpoint]) -> np.ndarray:
    """Image-frame shoulder-shell vectors ``R(heading) @ offsets[1]``; NaN if unknown."""
    out = np.full((len(points), 2), np.nan)
    for k, p in enumerate(points):
        h = float(p.heading)
        off = np.asarray(p.offsets, dtype=float)[SHOULDER]
        if math.isfinite(h) and np.all(np.isfinite(off)):
            c, s = math.cos(h), math.sin(h)
            out[k] = (c * off[0] - s * off[1], s * off[0] + c * off[1])
    return out


def v003_stack(points: List[Endpoint]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack positions (n,2), velocities (n,2) and vel_known (n,) of endpoints."""
    pos = np.array([np.asarray(p.pos, dtype=float) for p in points]).reshape(-1, 2)
    vel = np.array([np.asarray(p.vel, dtype=float) for p in points]).reshape(-1, 2)
    known = np.array([bool(p.vel_known) for p in points], dtype=bool)
    assert pos.shape == vel.shape == (len(points), 2)
    return pos, vel, known


def v003_pair_geometry(ends, starts, rows, cols, horizon, cap):
    """Frame separation, displacement, prediction error and shoulder change per pair."""
    pa, va, ka = (x[rows] for x in v003_stack(ends))
    pb, vb, kb = (x[cols] for x in v003_stack(starts))
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
    ua = v003_shoulder_vectors(ends)[rows]
    ub = v003_shoulder_vectors(starts)[cols]
    du = np.linalg.norm(ua - ub, axis=1)
    shoulder = np.where(np.isfinite(du), np.minimum(np.nan_to_num(du, nan=0.0),
                                                    cap), 0.0)
    return dt, dist, err, shoulder


def v003_gate_labels(cost, disp_ok, err_ok, max_cost, side_veto=None):
    """First failing gate per pair: displacement, prediction error, claw side, max cost."""
    gate = np.full(len(cost), '', dtype=object)
    gate[~(np.isfinite(cost) & (cost <= max_cost))] = 'max_cost'
    if side_veto is not None:
        gate[np.asarray(side_veto, dtype=bool)] = 'claw_side'
    gate[~err_ok] = 'prediction_error'
    gate[~disp_ok] = 'displacement'
    return gate


def v003_pair_costs_linear(ends, starts, rows, cols, cfg):
    """Verbatim body of the v003 ``_pair_costs_linear`` with its helpers frozen above."""
    dt, dist, err, shoulder = v003_pair_geometry(ends, starts, rows, cols, None,
                                                 cfg.shoulder_vector_cost_cap_px)
    w = cfg.motion_weight
    cost = (1 - w) * dist / dt + w * err / dt + cfg.shoulder_vector_weight * shoulder
    disp_ok = dist <= cfg.max_body_displacement_px * dt
    err_ok = err <= cfg.max_prediction_error_px * dt
    return dt, cost, v003_gate_labels(cost, disp_ok, err_ok, cfg.max_assignment_cost)


def hand_pair_features(ends, starts, rows, cols, cfg):
    """Stand-in for ``features.pair_features``: only the hand_cost column is filled."""
    dt, cost, gate = candidates._pair_costs_sqrt(ends, starts, rows, cols,
                                                 replace(cfg, cost_model='hand'))
    X = np.zeros((len(rows), len(features.FEATURE_NAMES)))
    X[:, HAND_COST] = cost
    return X, gate


def zero_pair_features(ends, starts, rows, cols, cfg):
    """Stand-in for ``features.pair_features`` returning an all-zero matrix."""
    return np.zeros((len(rows), len(features.FEATURE_NAMES))), np.full(len(rows), '', dtype=object)


class HandScorer:
    """Scorer whose ``-log p`` is the hand_cost feature."""
    feature_names = features.FEATURE_NAMES
    threshold = math.exp(-1.0)

    def cost(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float)[:, HAND_COST].copy()


class ConstantScorer:
    """Scorer returning the same probability ``p`` for every pair."""
    feature_names = features.FEATURE_NAMES

    def __init__(self, p: float):
        self.p = float(p)
        self.threshold = self.p

    def cost(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), -math.log(self.p))


class ExactScorer:
    """Scorer returning ``-np.log(p)`` for every pair, as ``scorer.Scorer.cost`` does."""
    feature_names = features.FEATURE_NAMES

    def __init__(self, p: float):
        self.p = float(p)
        self.threshold = self.p

    def cost(self, X: np.ndarray) -> np.ndarray:
        return -np.log(np.full(len(X), self.p))


@st.composite
def scenes(draw, max_tracklets: int = 10, cues: bool = True):
    """
    Draw (ends, starts) for one video (as in test_gap_bridging_candidates).

    With ``cues``, each tracklet may also carry a carapace size, a claw side
    and a claw offset (identical on its START and END), so the claw-side veto,
    the cue cost terms and the axis-flip suppression are exercised.
    """
    n = draw(st.integers(0, max_tracklets))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    ends: List[Endpoint] = []
    starts: List[Endpoint] = []
    for k in range(n):
        first = draw(st.integers(0, 24))
        length = draw(st.integers(1, 8))
        known = length >= 2
        extra = {}
        if cues and draw(st.booleans()):
            extra = dict(size_px=float(rng.uniform(20, 30)), size_n=int(rng.integers(1, 20)),
                         claw_side=float(rng.choice([-1.0, -0.95, 0.3, 0.95, 1.0])),
                         claw_side_n=int(rng.integers(1, 12)))
        for frame, out in ((first, starts), (first + length - 1, ends)):
            heading = float(rng.uniform(-math.pi, math.pi)) if draw(st.booleans()) else math.nan
            shoulder = (tuple(rng.uniform(10, 40, 2)) if draw(st.booleans()) else None)
            claw = (tuple(rng.uniform(-30, 30, 2)) if cues and draw(st.booleans()) else None)
            vel = tuple(rng.uniform(-6, 6, 2)) if known else (0., 0.)
            point = ep(f"t{k:03d}", frame, tuple(rng.uniform(0, 60, 2)), vel, known,
                       heading, shoulder, claw=claw)
            out.append(replace(point, **extra))
    return ends, starts


def veto_scene(spacing: float = 1.5, c_gap: int = 9, c_dist: float = 3.0):
    """D-001 scene: A ends at f10; B, B2 start at f12 at (0, +-spacing); C starts c_gap frames on."""
    a = ep('A', 10, (0., 0.), vel_known=False)
    b, b2 = ep('B', 12, (0., spacing), vel_known=False), ep('B2', 12, (0., -spacing),
                                                            vel_known=False)
    b_end, b2_end = ep('B', 200, (0., spacing)), ep('B2', 200, (0., -spacing))
    c = ep('C', 11 + c_gap, (c_dist, 0.), vel_known=False)
    return [a, b_end, b2_end], [b, b2, c]


def lone_pair(dist: float = 1.0) -> Tuple[List[Endpoint], List[Endpoint]]:
    """One END and one START, gap 1, ``dist`` px apart, velocities unknown."""
    return [ep('A', 10, (0., 0.), vel_known=False)], [ep('B', 12, (dist, 0.), vel_known=False)]


class Recorder:
    """Wrap ``candidates._solve_component`` and record ``(cost, unmatched_cost)`` per call."""

    def __init__(self):
        self.calls: List[Tuple[np.ndarray, float]] = []
        self.original = candidates._solve_component

    def __call__(self, rows, cols, cost, unmatched_cost):
        self.calls.append((np.array(cost, dtype=float), float(unmatched_cost)))
        return self.original(rows, cols, cost, unmatched_cost)


# ============================================================
# STAGES AND GLOBAL ASSIGNMENT
# ============================================================

class StagesTests(unittest.TestCase):

    def test_global_is_single_stage(self):
        for base in (SQRT, LINEAR):
            for max_gap in (1, 5, 10, 30, 60):
                cfg = replace(base, max_gap=max_gap, assignment='global')
                self.assertEqual(candidates._stages(cfg), (max_gap,))

    def test_staged_unchanged(self):
        for max_gap, expected in ((1, (1,)), (5, (5,)), (10, (5, 10)), (30, (5, 15, 30)),
                                  (60, (5, 15, 30, 45, 60))):
            self.assertEqual(candidates._stages(replace(SQRT, max_gap=max_gap)), expected)
        self.assertEqual(candidates._stages(replace(SQRT, max_gap=7, link_stages=(2, 7, 9))), (2, 7))
        self.assertEqual(candidates._stages(replace(LINEAR, max_gap=30)), (30,))

    def test_regression_D001_staged_still_holds(self):
        """D-001 under staged: A's stage-1 veto holds at stage 2."""
        ends, starts = veto_scene()
        links, decisions = link_tracklets(ends, starts, replace(SQRT, max_gap=15))
        self.assertEqual(links, [])
        d = by_pair(decisions)
        self.assertEqual((d[('A', 'C')].stage, d[('A', 'C')].decision, d[('A', 'C')].reason),
                         (2, 'refused_assignment', 'endpoint vetoed at stage 1'))

    def test_D001_scene_under_global(self):
        """Global: one stage, no 'at stage' reason, and the excluded-label path is never reached."""
        ends, starts = veto_scene()
        cfg = replace(SQRT, max_gap=15, assignment='global')
        with mock.patch.object(candidates, '_excluded_label',
                               side_effect=AssertionError('reached')) as hook:
            links, decisions = link_tracklets(ends, starts, cfg)
        hook.assert_not_called()
        self.assertEqual({d.stage for d in decisions}, {1})
        self.assertFalse(any('at stage' in d.reason for d in decisions))
        # Single global solve: A's veto and (A, C) compete in the same stage.
        self.assertNotIn('A', {a for a, _ in links})
        staged1, staged_dec = link_tracklets(ends, starts, replace(SQRT, max_gap=15, link_stages=(15,)))
        self.assertEqual(links, staged1)
        self.assertTrue(all(same_decision(x, y) for x, y in zip(decisions, staged_dec)))


class GlobalAssignmentPropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=120)
    @given(scenes(), st.integers(1, 30), st.sampled_from(['sqrt', 'linear']),
           st.sampled_from(['fixed', 'adaptive']))
    def test_global_equals_single_stage(self, scene, max_gap, model, mode):
        ends, starts = scene
        base = replace(SQRT if model == 'sqrt' else LINEAR, max_gap=max_gap, unmatched_cost_mode=mode)
        with mock.patch.object(candidates, '_excluded_label',
                               side_effect=AssertionError('reached')) as hook:
            links_g, dec_g = link_tracklets(ends, starts, replace(base, assignment='global'))
        hook.assert_not_called()
        for stages in ((max_gap,), (60,)):
            links_s, dec_s = link_tracklets(ends, starts, replace(base, link_stages=stages))
            self.assertEqual(links_g, links_s)
            self.assertEqual(len(dec_g), len(dec_s))
            for x, y in zip(dec_g, dec_s):
                self.assertTrue(same_decision(x, y), (x, y))
        self.assertTrue(all(d.stage == 1 and 'at stage' not in d.reason for d in dec_g))


# ============================================================
# ADAPTIVE DUMMY
# ============================================================

class AdaptiveUnmatchedTests(unittest.TestCase):

    def test_adaptive_unmatched_value(self):
        cost = np.array([0.2, np.inf, 0.5, np.nan, 0.1])
        self.assertEqual(candidates._adaptive_unmatched(cost, 1.05), 1.05 * 0.5)
        self.assertEqual(candidates._adaptive_unmatched(np.array([0.3]), 1.0), 0.3)
        with self.assertRaises(AssertionError):
            candidates._adaptive_unmatched(np.array([np.nan, np.inf]), 1.05)

    def test_lone_pair_linked_under_adaptive_beyond_fixed_dummy(self):
        # Fixed dummy lowered to 0.3: a lone pair costing 0.75 passes every gate (max cost 1.0)
        # but stays unlinked (0.75 > 2 * 0.3); the adaptive dummy 1.05 * 0.75 links it.
        cfg = replace(SQRT, unmatched_cost_sqrt=0.3, motion_weight=0.0)
        r = float(cfg.gate_radius(2))
        ends, starts = lone_pair(0.75 * r)
        _, fixed = link_tracklets(ends, starts, cfg)
        self.assertGreater(fixed[0].cost, 0.6)
        self.assertLess(fixed[0].cost, 1.0)
        self.assertEqual((fixed[0].decision, fixed[0].reason), ('refused_assignment', 'not assigned'))
        links, adaptive = link_tracklets(ends, starts, replace(cfg, unmatched_cost_mode='adaptive'))
        self.assertEqual(links, [('A', 'B')])
        self.assertEqual(adaptive[0].reason, f"cost {fixed[0].cost:.4f} no competitor")

    @settings(deadline=None, max_examples=80)
    @given(st.floats(0.0, 5.9), st.sampled_from([1.0, 1.05, 2.0]))
    def test_lone_gated_pair_always_linked_under_adaptive(self, dist, factor):
        cfg = replace(SQRT, unmatched_cost_mode='adaptive', adaptive_factor=factor)
        ends, starts = lone_pair(dist)
        links, decisions = link_tracklets(ends, starts, cfg)
        if decisions[0].decision != 'refused_gate':
            self.assertEqual(links, [('A', 'B')])

    @settings(deadline=None, max_examples=150)
    @given(scenes(), st.integers(1, 30), st.sampled_from(COMBOS), st.sampled_from([1.0, 1.05, 1.5]))
    def test_component_dummy_bounds_component_costs(self, scene, max_gap, combo, factor):
        ends, starts = scene
        assignment, mode = combo
        cfg = replace(SQRT, max_gap=max_gap, assignment=assignment, unmatched_cost_mode=mode,
                      adaptive_factor=factor)
        recorder = Recorder()
        with mock.patch.object(candidates, '_solve_component', side_effect=recorder):
            link_tracklets(ends, starts, cfg)
        for cost, dummy in recorder.calls:
            self.assertTrue(np.all(np.isfinite(cost)))
            if mode == 'adaptive':
                self.assertEqual(dummy, candidates._adaptive_unmatched(cost, factor))
                self.assertEqual(dummy, factor * float(cost.max()))
                self.assertTrue(np.all(cost <= dummy))
            else:
                self.assertEqual(dummy, cfg.cost_thresholds()['unmatched_cost'])

    @settings(deadline=None, max_examples=200)
    @given(scenes(), st.integers(1, 30), st.sampled_from(COMBOS),
           st.sampled_from([(5, 15, 30, 45), (1, 2, 3), (2, 8), (30,)]))
    def test_audit_and_link_invariants(self, scene, max_gap, combo, link_stages):
        """Adapted from test_gap_bridging_candidates under {staged, global} x {fixed, adaptive}."""
        ends, starts = scene
        assignment, mode = combo
        cfg = replace(SQRT, max_gap=max_gap, link_stages=link_stages, assignment=assignment,
                      unmatched_cost_mode=mode)
        scale = cfg.cost_thresholds()
        stages = candidates._stages(cfg)
        links, decisions = link_tracklets(ends, starts, cfg)
        if assignment == 'global':
            self.assertEqual(stages, (max_gap,))
        linked_at, vetoed_at = {}, {}
        for d in decisions:
            if d.decision == 'linked':
                linked_at[('end', d.tracklet_a)] = linked_at[('start', d.tracklet_b)] = d
            elif d.decision == 'refused_margin':
                self.assertNotIn(('end', d.tracklet_a), vetoed_at)
                self.assertNotIn(('start', d.tracklet_b), vetoed_at)
                vetoed_at[('end', d.tracklet_a)] = vetoed_at[('start', d.tracklet_b)] = d
        self.assertFalse(set(linked_at) & set(vetoed_at))
        pairs = [(d.tracklet_a, d.tracklet_b) for d in decisions]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(len({a for a, _ in links}), len(links))
        self.assertEqual(len({b for _, b in links}), len(links))
        self.assertEqual(links, [p for p, d in zip(pairs, decisions) if d.decision == 'linked'])
        for d in decisions:
            self.assertIn(d.decision, DECISIONS)
            self.assertTrue(1 <= d.gap_frames <= max_gap)
            lo = stages[d.stage - 2] if d.stage >= 2 else 0
            self.assertTrue(lo < d.gap_frames <= stages[d.stage - 1])
            if assignment == 'global':
                self.assertNotIn('at stage', d.reason)
            if d.reason.startswith(EXCLUDED) or d.reason.startswith(VETOED):
                self.assertEqual(d.decision, 'refused_assignment')
                table = linked_at if d.reason.startswith(EXCLUDED) else vetoed_at
                earlier = [x for x in (table.get(('end', d.tracklet_a)), table.get(('start', d.tracklet_b)))
                           if x is not None and x.stage < d.stage]
                self.assertTrue(earlier)
                self.assertEqual(d.runner_up_cost, min(x.cost for x in earlier))
            if d.decision in ('linked', 'refused_margin'):
                expected = chain_free_runner_up(decisions, d.tracklet_a, d.tracklet_b)
                np.testing.assert_allclose(d.runner_up_cost, expected, rtol=1e-12)
            if d.decision == 'linked':
                self.assertTrue(math.isnan(expected)
                                or expected - d.cost >= scale['minimum_cost_margin'])
                self.assertLessEqual(d.cost, scale['max_assignment_cost'])
                if mode == 'fixed':
                    self.assertLess(d.cost, scale['unmatched_cost'] * 2)
            elif d.decision == 'refused_margin':
                self.assertLess(expected - d.cost, scale['minimum_cost_margin'])
            elif d.decision == 'refused_gate':
                self.assertIn(d.reason.split(';')[0], GATE_REASONS)
                self.assertEqual(math.isnan(d.cost), not d.reason.startswith('max_cost'))
            else:
                self.assertTrue(math.isfinite(d.cost))
                self.assertLessEqual(d.cost, scale['max_assignment_cost'])


# ============================================================
# LEARNED COST
# ============================================================

class LearnedCostTests(unittest.TestCase):

    def learned(self, **kw) -> BridgeConfig:
        return replace(SQRT, **{**LEARNED, **kw})

    def test_scorer_required(self):
        ends, starts, rows, cols = sorted_pairs(*lone_pair(), SQRT)
        with self.assertRaises(ValueError):
            candidates._pair_costs(ends, starts, rows, cols, self.learned())

    def test_wrong_feature_names_rejected(self):
        ends, starts, rows, cols = sorted_pairs(*lone_pair(), SQRT)
        scorer = ConstantScorer(0.9)
        scorer.feature_names = tuple(reversed(features.FEATURE_NAMES))
        with mock.patch.object(features, 'pair_features', side_effect=zero_pair_features):
            with self.assertRaises(ValueError):
                candidates._pair_costs(ends, starts, rows, cols, self.learned(), scorer=scorer)
            with self.assertRaises(ValueError):
                link_tracklets(*lone_pair(), self.learned(), scorer=scorer)

    def test_max_cost_gate_iff_probability_below_minimum(self):
        p = 0.4
        ends, starts = lone_pair()
        with mock.patch.object(features, 'pair_features', side_effect=zero_pair_features):
            for p_min in (0.1, 0.3, 0.39, 0.41, 0.5, 0.9):
                links, decisions = link_tracklets(ends, starts, self.learned(learned_min_probability=p_min),
                                                  scorer=ConstantScorer(p))
                d = decisions[0]
                self.assertAlmostEqual(d.cost, -math.log(p))
                if p < p_min:
                    self.assertEqual((d.decision, d.reason), ('refused_gate', 'max_cost'))
                    self.assertEqual(links, [])
                else:
                    self.assertEqual(d.decision, 'linked')

    def test_regression_probability_at_minimum_is_not_refused(self):
        # math.log and np.log differ by one ulp at this t; the gate limit must use the
        # scorer's np.log so that p == learned_min_probability passes.
        t = 0.9571529830729761
        self.assertNotEqual(-math.log(t), float(-np.log(t)))
        cfg = self.learned(learned_min_probability=t)
        ends, starts = lone_pair()
        e, s, rows, cols = sorted_pairs(ends, starts, cfg)
        with mock.patch.object(features, 'pair_features', side_effect=zero_pair_features):
            _, cost, gate = candidates._pair_costs(e, s, rows, cols, cfg, scorer=ExactScorer(t))
            links, decisions = link_tracklets(ends, starts, cfg, scorer=ExactScorer(t))
        self.assertEqual(cost[0], -np.log(t))
        self.assertEqual(list(gate), [''])
        self.assertNotEqual(decisions[0].decision, 'refused_gate')
        self.assertNotEqual(decisions[0].reason, 'max_cost')
        self.assertEqual(links, [('A', 'B')])

    @settings(deadline=None, max_examples=200)
    @example(t=0.9571529830729761)
    @given(st.floats(0.5, 1.0, exclude_max=True))
    def test_property_max_cost_gate_iff_below_minimum(self, t):
        cfg = self.learned(learned_min_probability=t)
        e, s, rows, cols = sorted_pairs(*lone_pair(), cfg)
        below = float(np.nextafter(t, 0.0))
        with mock.patch.object(features, 'pair_features', side_effect=zero_pair_features):
            at = candidates._pair_costs(e, s, rows, cols, cfg, scorer=ExactScorer(t))[2]
            under = candidates._pair_costs(e, s, rows, cols, cfg, scorer=ExactScorer(below))[2]
        self.assertEqual(list(at), [''])
        self.assertEqual(list(under), [candidates.GATE_MAX_COST])

    def test_motion_gates_unchanged_whatever_the_scorer(self):
        far = [ep('A', 10, (0., 0.), vel_known=False)], [ep('B', 12, (20., 0.), vel_known=False)]
        _, hand = link_tracklets(*far, SQRT)
        self.assertEqual(hand[0].reason, 'displacement')
        with mock.patch.object(features, 'pair_features', side_effect=zero_pair_features):
            for p in (0.01, 0.5, 1.0):
                links, decisions = link_tracklets(*far, self.learned(), scorer=ConstantScorer(p))
                self.assertEqual(links, [])
                self.assertEqual((decisions[0].decision, decisions[0].reason),
                                 ('refused_gate', 'displacement'))
                self.assertTrue(math.isnan(decisions[0].cost))

    def test_claw_side_veto_unchanged_whatever_the_scorer(self):
        ends, starts = lone_pair()
        ends = [replace(ends[0], claw_side=1.0, claw_side_n=10)]
        starts = [replace(starts[0], claw_side=-1.0, claw_side_n=10)]
        _, hand = link_tracklets(ends, starts, SQRT)
        self.assertEqual(hand[0].reason, 'claw_side')
        with mock.patch.object(features, 'pair_features', side_effect=zero_pair_features):
            links, decisions = link_tracklets(ends, starts, self.learned(), scorer=ConstantScorer(0.999))
        self.assertEqual(links, [])
        self.assertEqual((decisions[0].decision, decisions[0].reason), ('refused_gate', 'claw_side'))
        self.assertEqual(decisions[0].side_term, hand[0].side_term)

    def test_pair_costs_learned_gate_and_nan(self):
        # Motion-failed pairs carry NaN cost; motion-passing pairs carry the scorer's -log p.
        ends = [ep('A', 10, (0., 0.), vel_known=False)]
        starts = [ep('B', 12, (1., 0.), vel_known=False), ep('C', 12, (20., 0.), vel_known=False)]
        ends, starts, rows, cols = sorted_pairs(ends, starts, SQRT)
        with mock.patch.object(features, 'pair_features', side_effect=zero_pair_features):
            dt, cost, gate = candidates._pair_costs(ends, starts, rows, cols, self.learned(),
                                                    scorer=ConstantScorer(0.8))
        self.assertEqual(list(gate), ['', 'displacement'])
        self.assertAlmostEqual(cost[0], -math.log(0.8))
        self.assertTrue(math.isnan(cost[1]))
        self.assertEqual(list(dt), [2, 2])

    @settings(deadline=None, max_examples=150)
    @example(scene=veto_scene(), max_gap=15, combo=('staged', 'fixed'))
    @given(scenes(), st.integers(1, 30), st.sampled_from(COMBOS))
    def test_hand_scorer_reproduces_hand_links(self, scene, max_gap, combo):
        ends, starts = scene
        assignment, mode = combo
        hand_cfg = replace(SQRT, max_gap=max_gap, assignment=assignment, unmatched_cost_mode=mode)
        links_h, dec_h = link_tracklets(ends, starts, hand_cfg)
        with mock.patch.object(features, 'pair_features', side_effect=hand_pair_features):
            links_l, dec_l = link_tracklets(ends, starts, replace(hand_cfg, **LEARNED),
                                            scorer=HandScorer())
        self.assertEqual(links_h, links_l)
        self.assertEqual(len(dec_h), len(dec_l))
        for x, y in zip(dec_h, dec_l):
            self.assertTrue(same_decision(x, y), (x, y))

    def test_unresolved_scorer_is_resolved_lazily(self):
        resolved = HandScorer()
        fake = mock.MagicMock()
        fake.resolve_scorer.return_value = resolved
        with mock.patch.dict(sys.modules, {'gap_bridging.scorer': fake}), \
                mock.patch('gap_bridging.scorer', fake, create=True), \
                mock.patch.object(features, 'pair_features', side_effect=hand_pair_features):
            links, _ = link_tracklets(*lone_pair(), self.learned())
        fake.resolve_scorer.assert_called_once()
        self.assertEqual(links, [('A', 'B')])

    @unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
    def test_demo_hand_scorer_reproduces_default_links(self):
        import pyarrow.parquet as pq
        cfg = BridgeConfig()
        eps = geometry.endpoints(pq.read_table(DEMO), cfg)
        ends = [e for _, e in eps.values()]
        starts = [s for s, _ in eps.values()]
        by_video = {}
        for e, s in zip(ends, starts):
            by_video.setdefault(e.video_id, ([], []))
            by_video[e.video_id][0].append(e)
            by_video[e.video_id][1].append(s)
        n_links = 0
        for video, (v_ends, v_starts) in sorted(by_video.items()):
            links_h, dec_h = link_tracklets(v_ends, v_starts, cfg)
            with mock.patch.object(features, 'pair_features', side_effect=hand_pair_features):
                links_l, dec_l = link_tracklets(v_ends, v_starts, replace(cfg, **LEARNED),
                                                scorer=HandScorer())
            self.assertEqual(links_h, links_l)
            for name in DECISIONS:
                self.assertEqual(sum(d.decision == name for d in dec_h),
                                 sum(d.decision == name for d in dec_l), name)
            self.assertTrue(all(same_decision(x, y) for x, y in zip(dec_h, dec_l)))
            n_links += len(links_h)
        self.assertGreater(n_links, 0)


# ============================================================
# LEARNED COST ON A FEATURE SUBSET (v005)
# ============================================================

class SubsetScorer:
    """Scorer on the ``no_length`` columns; ``-log p`` is hand_cost; records every X it gets."""
    feature_names = features.FEATURE_SETS['no_length']
    threshold = math.exp(-1.0)

    def __init__(self):
        self.seen: List[np.ndarray] = []

    def cost(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == len(self.feature_names), X.shape
        self.seen.append(X.copy())
        return X[:, self.feature_names.index('hand_cost')].copy()


class LearnedSubsetTests(unittest.TestCase):

    def learned(self, **kw) -> BridgeConfig:
        return replace(SQRT, **{**LEARNED, **kw})

    def scene(self):
        ends = [ep('A', 10, (0., 0.), (0.5, 0.), heading=0.2, shoulder=(22., 0.)),
                ep('D', 11, (5., 5.), vel_known=False)]
        starts = [ep('B', 12, (1., 0.2), (0.4, 0.), heading=0.3),
                  ep('C', 13, (2., -1.), vel_known=False), ep('E', 12, (40., 0.), vel_known=False)]
        return [replace(e, size_px=25.0, size_n=3) for e in ends], starts

    def assert_columns(self, ends, starts, cfg):
        e, s, rows, cols = sorted_pairs(ends, starts, cfg)
        recorder = SubsetScorer()
        _, cost, gate = candidates._pair_costs(e, s, rows, cols, cfg, scorer=recorder)
        X_full, _ = features.pair_features(e, s, rows, cols, replace(cfg, cost_model='hand'))
        motion_ok = ~np.isin(gate.astype(str), ['displacement', 'prediction_error'])
        idx = list(features.check_feature_subset(SubsetScorer.feature_names))
        if not motion_ok.any():
            self.assertEqual(recorder.seen, [])
            return 0
        self.assertEqual(len(recorder.seen), 1)
        X = recorder.seen[0]
        self.assertEqual(X.shape, (int(motion_ok.sum()), 20))
        np.testing.assert_array_equal(X, X_full[motion_ok][:, idx])
        np.testing.assert_array_equal(cost[motion_ok], X_full[motion_ok, HAND_COST])
        return int(motion_ok.sum())

    def test_subset_scorer_receives_selected_columns(self):
        ends, starts = self.scene()
        self.assertGreater(self.assert_columns(ends, starts, self.learned()), 1)

    @settings(deadline=None, max_examples=60)
    @given(scenes(), st.integers(1, 20))
    def test_property_subset_columns(self, scene, max_gap):
        self.assert_columns(*scene, self.learned(max_gap=max_gap))

    @settings(deadline=None, max_examples=60)
    @given(scenes(), st.integers(1, 20), st.sampled_from(COMBOS))
    def test_property_subset_hand_scorer_reproduces_hand_links(self, scene, max_gap, combo):
        ends, starts = scene
        assignment, mode = combo
        hand_cfg = replace(SQRT, max_gap=max_gap, assignment=assignment, unmatched_cost_mode=mode)
        links_h, dec_h = link_tracklets(ends, starts, hand_cfg)
        links_l, dec_l = link_tracklets(ends, starts, replace(hand_cfg, **LEARNED),
                                        scorer=SubsetScorer())
        self.assertEqual(links_h, links_l)
        self.assertEqual(len(dec_h), len(dec_l))
        for x, y in zip(dec_h, dec_l):
            self.assertTrue(same_decision(x, y), (x, y))

    def test_permuted_or_unknown_names_rejected(self):
        ends, starts = self.scene()
        e, s, rows, cols = sorted_pairs(ends, starts, SQRT)
        names = list(SubsetScorer.feature_names)
        swapped = names[:]
        swapped[0], swapped[1] = swapped[1], swapped[0]
        for bad in (tuple(swapped), tuple(reversed(names)), ('hand_cost', 'gap'), ('bogus',), ()):
            scorer = SubsetScorer()
            scorer.feature_names = bad
            with self.assertRaises(ValueError, msg=repr(bad)):
                candidates._pair_costs(e, s, rows, cols, self.learned(), scorer=scorer)
            with self.assertRaises(ValueError):
                link_tracklets(ends, starts, self.learned(), scorer=scorer)
            self.assertEqual(scorer.seen, [])

    def test_full_name_scorers_unchanged(self):
        # The v004 fakes (all 28 names) still get the full matrix.
        seen = []

        class FullRecorder(HandScorer):
            def cost(self, X):
                seen.append(np.asarray(X, dtype=float).copy())
                return super().cost(X)

        ends, starts = self.scene()
        e, s, rows, cols = sorted_pairs(ends, starts, SQRT)
        candidates._pair_costs(e, s, rows, cols, self.learned(), scorer=FullRecorder())
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].shape[1], len(features.FEATURE_NAMES))
        X_full, gate = features.pair_features(e, s, rows, cols, SQRT)
        motion_ok = ~np.isin(gate.astype(str), ['displacement', 'prediction_error'])
        np.testing.assert_array_equal(seen[0], X_full[motion_ok])


# ============================================================
# REFACTOR: SQRT AND LINEAR NUMERICS UNCHANGED
# ============================================================

class RefactorTests(unittest.TestCase):

    @settings(deadline=None, max_examples=150)
    @given(scenes(), st.integers(1, 30), st.booleans(), st.sampled_from([0.0, 12.0]))
    def test_pair_costs_sqrt_unchanged(self, scene, max_gap, cues, cap):
        cfg = replace(SQRT, max_gap=max_gap, appearance_cues=cues, shoulder_vector_cost_cap_px=cap)
        ends, starts, rows, cols = sorted_pairs(*scene, cfg)
        if rows.size == 0:
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            old = legacy_pair_costs_sqrt(ends, starts, rows, cols, cfg)
        new = candidates._pair_costs_sqrt(ends, starts, rows, cols, cfg)
        np.testing.assert_array_equal(old[0], new[0])
        np.testing.assert_array_equal(old[1], new[1])     # NaN-equal, bitwise on finite values
        self.assertEqual(list(old[2]), list(new[2]))
        via = candidates._pair_costs(ends, starts, rows, cols, cfg)
        np.testing.assert_array_equal(via[1], new[1])
        terms = candidates._sqrt_gate_terms(ends, starts, rows, cols, cfg)
        self.assertEqual(len(terms), 8)
        dt, dist, err, shoulder, radius, speed, disp_ok, err_ok = terms
        np.testing.assert_array_equal(dt, new[0])
        np.testing.assert_array_equal(disp_ok, dist <= radius)
        np.testing.assert_array_equal(err_ok, err <= cfg.prediction_error_ratio * radius)

    @settings(deadline=None, max_examples=100)
    @given(scenes(), st.integers(1, 30))
    def test_linear_pair_costs_match_frozen_v003(self, scene, max_gap):
        # Guard independent of the current module: the v003 body is frozen in this file.
        cfg = replace(LINEAR, max_gap=max_gap)
        ends, starts, rows, cols = sorted_pairs(*scene, cfg)
        if rows.size == 0:
            return
        new = candidates._pair_costs(ends, starts, rows, cols, cfg)
        with np.errstate(divide='ignore', invalid='ignore'):
            old = v003_pair_costs_linear(ends, starts, rows, cols, cfg)
        np.testing.assert_array_equal(old[0], new[0])
        self.assertEqual(new[0].dtype, np.int64)
        np.testing.assert_array_equal(old[1], new[1])     # bitwise on finite values, NaN-equal
        self.assertEqual(list(old[2]), list(new[2]))


if __name__ == '__main__':
    unittest.main()
