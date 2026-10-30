#!/usr/bin/env python3

"""
Tests for gap_bridging.candidates.link_tracklets (candidate linking).

Example tests pin the gates, the cost formula, the margin veto and the
per-video contract under the legacy ``linear`` gate model, and the sub-linear
radius, gap penalty and capped prediction horizon of the ``sqrt`` model; the
linear model is checked field by field against the pre-change cost code.
Hypothesis property tests, run under both gate models, check one-to-one links,
the one-audit-row-per-candidate invariant, the margin rule, invariance under
relabeling and permutation, and time-reversal symmetry. Under sqrt, pairs
beyond the prefilter radius ``R_max`` get no audit row; the remaining rows and
the links equal those of an unpruned run.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Set, Tuple
from unittest import mock

# ----- numerics -----
import numpy as np

# ----- testing -----
from hypothesis import example, given, settings, strategies as st

# ----- local modules -----
ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
sys.path.insert(0, str(ROOT / 'scripts'))
from gap_bridging import candidates, geometry  # noqa: E402
from gap_bridging.candidates import link_tracklets  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.types import DECISIONS, Endpoint  # noqa: E402

# ============================================================
# HELPERS
# ============================================================

CFG = BridgeConfig(gate_model='linear')      # legacy v001 behaviour; example tests pin its numbers
SQRT = BridgeConfig(gate_model='sqrt', radius_r0_px=0.0, radius_k_px=3.73, radius_safety=1.25,
                    radius_min_px=6.0, prediction_error_ratio=1.5, prediction_horizon_frames=3,
                    gap_penalty=0.06, max_assignment_cost_sqrt=1.0, unmatched_cost_sqrt=0.6,
                    minimum_cost_margin_sqrt=0.25)    # pinned so example numbers survive refits
MODELS = {'linear': CFG, 'sqrt': SQRT}
GATE_REASONS = {'displacement', 'prediction_error', 'max_cost'}


def ep(tid: str, frame: int, pos: Tuple[float, float], vel: Tuple[float, float] = (0., 0.),
       vel_known: bool = True, heading: float = math.nan,
       shoulder: Optional[Tuple[float, float]] = None, video: str = 'v') -> Endpoint:
    """Build an Endpoint; ``shoulder`` is the body-frame shoulder offset."""
    offsets = np.full((5, 2), np.nan)
    offsets[0] = 0.0
    if shoulder is not None:
        offsets[1] = shoulder
    return Endpoint(tid, video, int(frame), 5 if vel_known else 1,
                    np.asarray(pos, dtype=float), np.asarray(vel, dtype=float),
                    bool(vel_known), float(heading), offsets)


def enumerated(ends: List[Endpoint], starts: List[Endpoint], max_gap: int) -> Set[Tuple[str, str]]:
    """Brute-force candidate set: same video, 1 <= gap <= max_gap, a != b."""
    return {(a.tracklet_id, b.tracklet_id) for a in ends for b in starts
            if 1 <= b.frame - a.frame - 1 <= max_gap and a.tracklet_id != b.tracklet_id}


def prefilter_radius(ends: List[Endpoint], starts: List[Endpoint], cfg: BridgeConfig) -> float:
    """
    Independent R_max: ``inf`` under linear; under sqrt
    ``r(max_gap+1) + v * min(max_gap+1, h)`` with ``v`` the larger of the speed
    cap and the fastest known endpoint speed.
    """
    if cfg.gate_model == 'linear':
        return math.inf
    speeds = [float(np.hypot(*p.vel)) for p in ends + starts if p.vel_known]
    v = max([cfg.max_velocity_px_per_frame] + speeds)
    dt = cfg.max_gap + 1
    return float(cfg.gate_radius(dt)) + v * min(dt, cfg.prediction_horizon_frames)


def audited(ends: List[Endpoint], starts: List[Endpoint],
            cfg: BridgeConfig) -> Set[Tuple[str, str]]:
    """Brute-force audited pairs: ``enumerated`` pairs whose body distance is <= R_max."""
    r_max = prefilter_radius(ends, starts, cfg)
    pos = {('e', a.tracklet_id): a.pos for a in ends}
    pos.update({('s', b.tracklet_id): b.pos for b in starts})
    return {(a, b) for a, b in enumerated(ends, starts, cfg.max_gap)
            if float(np.hypot(*(pos[('s', b)] - pos[('e', a)]))) <= r_max}


def by_pair(decisions) -> dict:
    """Index decisions by (tracklet_a, tracklet_b)."""
    return {(d.tracklet_a, d.tracklet_b): d for d in decisions}


EXCLUDED = 'endpoint linked at stage '
VETOED = 'endpoint vetoed at stage '


def chain_free_runner_up(decisions, a: str, b: str) -> float:
    """
    Independent runner-up for (a, b) over non-chain-reachable same-stage competitors.

    Motion-gated pairs are those with a finite cost (over all stages). Competitors
    are motion-gated pairs of the same stage as (a, b) whose END and START were
    not linked or margin-vetoed at an earlier stage. A row competitor (a, c) is
    dropped when (b, c) is motion-gated; a column competitor (d, b) is dropped when
    (d, a) is motion-gated.
    """
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


def radius(cfg: BridgeConfig, dt: int) -> float:
    """Gate radius r(dt) in px as a Python float."""
    return float(cfg.gate_radius(dt))


def reversed_scene(ends: List[Endpoint],
                   starts: List[Endpoint]) -> Tuple[List[Endpoint], List[Endpoint]]:
    """Time-reverse a scene: negate frames and velocities, swap ENDs and STARTs."""
    rev_ends = [replace(s, frame=-s.frame, vel=-np.asarray(s.vel)) for s in starts]
    rev_starts = [replace(e, frame=-e.frame, vel=-np.asarray(e.vel)) for e in ends]
    return rev_ends, rev_starts


def legacy_pair_costs(ends, starts, rows, cols, cfg):
    """Verbatim copy of the pre-gate-model ``candidates._pair_costs`` (commit 38d43c0)."""
    pa, va, ka = (x[rows] for x in candidates._stack(ends))
    pb, vb, kb = (x[cols] for x in candidates._stack(starts))
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
    ua = candidates._shoulder_vectors(ends)[rows]
    ub = candidates._shoulder_vectors(starts)[cols]
    du = np.linalg.norm(ua - ub, axis=1)
    shoulder = np.where(np.isfinite(du), np.minimum(np.nan_to_num(du, nan=0.0),
                                                    cfg.shoulder_vector_cost_cap_px), 0.0)
    w = cfg.motion_weight
    cost = (1 - w) * dist / dt + w * err / dt + cfg.shoulder_vector_weight * shoulder
    disp_ok = dist <= cfg.max_body_displacement_px * dt
    err_ok = err <= cfg.max_prediction_error_px * dt
    gate = np.full(len(rows), '', dtype=object)
    gate[~(np.isfinite(cost) & (cost <= cfg.max_assignment_cost))] = 'max_cost'
    gate[~err_ok] = 'prediction_error'
    gate[~disp_ok] = 'displacement'
    return dt, cost, gate


def same_decision(x, y) -> bool:
    """Field-by-field equality of two LinkDecisions, NaN equal to NaN, floats exact."""
    def eq(u, v):
        if isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v):
            return True
        return type(u) is type(v) and u == v
    return all(eq(getattr(x, f), getattr(y, f)) for f in x.__dataclass_fields__)


@st.composite
def scenes(draw, max_tracklets: int = 10, gap_free: bool = False):
    """
    Draw (ends, starts) for one video.

    Structure (counts, frames, lengths, which headings/shoulders are known)
    comes from strategies; continuous coordinates come from a NumPy generator
    seeded by a drawn integer so exact cost ties (which make the optimal
    assignment non-unique) have probability zero and failures still replay.
    """
    n = draw(st.integers(0, max_tracklets))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    ends: List[Endpoint] = []
    starts: List[Endpoint] = []
    for k in range(n):
        first = draw(st.integers(0, 1)) if gap_free else draw(st.integers(0, 24))
        length = draw(st.integers(1, 8))
        known = length >= 2
        for frame, out in ((first, starts), (first + length - 1, ends)):
            heading = float(rng.uniform(-math.pi, math.pi)) if draw(st.booleans()) else math.nan
            shoulder = (tuple(rng.uniform(10, 40, 2)) if draw(st.booleans()) else None)
            vel = tuple(rng.uniform(-6, 6, 2)) if known else (0., 0.)
            out.append(ep(f"t{k:03d}", frame, tuple(rng.uniform(0, 60, 2)), vel, known,
                          heading, shoulder))
    return ends, starts


# ============================================================
# EXAMPLE TESTS
# ============================================================

class CandidateExampleTests(unittest.TestCase):

    def straight_pair(self, dt: int):
        a = ep('a', 10, (100., 100.), (2., 0.))
        b = ep('b', 10 + dt, (100. + 2. * dt, 100.), (2., 0.))
        return link_tracklets([a], [b], CFG)

    def test_straight_line_links_gap_1_to_5(self):
        for gap in range(1, 6):
            links, decisions = self.straight_pair(gap + 1)
            self.assertEqual(links, [('a', 'b')])
            self.assertEqual(len(decisions), 1)
            d = decisions[0]
            self.assertEqual((d.decision, d.gap_frames), ('linked', gap))
            self.assertAlmostEqual(d.cost, 0.7 * 2.0)  # (1-w)*dist/dt, err = 0
            self.assertTrue(math.isnan(d.runner_up_cost))

    def test_straight_line_gap_0_and_6_not_enumerated(self):
        for gap in (0, 6):
            self.assertEqual(self.straight_pair(gap + 1), ([], []))

    def test_max_gap_zero_disables(self):
        a, b = ep('a', 0, (0., 0.)), ep('b', 2, (1., 0.))
        self.assertEqual(link_tracklets([a], [b], replace(CFG, max_gap=0)), ([], []))

    def test_same_tracklet_not_enumerated(self):
        a, b = ep('a', 0, (0., 0.)), ep('a', 2, (1., 0.))
        self.assertEqual(link_tracklets([a], [b], CFG), ([], []))

    def test_displacement_gate(self):
        a = ep('a', 0, (0., 0.), (13., 0.))
        b = ep('b', 2, (26.5, 0.), (13., 0.))  # dist 26.5 > 12*2, err tiny
        links, decisions = link_tracklets([a], [b], CFG)
        self.assertEqual(links, [])
        self.assertEqual((decisions[0].decision, decisions[0].reason),
                         ('refused_gate', 'displacement'))
        self.assertTrue(math.isnan(decisions[0].cost))

    def test_prediction_error_gate(self):
        a = ep('a', 0, (0., 0.), (-10., 0.))
        b = ep('b', 2, (20., 0.), (-10., 0.))  # dist 20 <= 24, err 40 > 36
        links, decisions = link_tracklets([a], [b], CFG)
        self.assertEqual(links, [])
        self.assertEqual((decisions[0].decision, decisions[0].reason),
                         ('refused_gate', 'prediction_error'))
        self.assertTrue(math.isnan(decisions[0].cost))

    def test_max_cost_gate_keeps_finite_cost(self):
        a = ep('a', 0, (0., 0.), vel_known=False)
        b = ep('b', 2, (26., 0.), vel_known=False)  # err = dist; cost 13 > 12
        cfg = replace(CFG, max_body_displacement_px=20.0)
        links, decisions = link_tracklets([a], [b], cfg)
        self.assertEqual(links, [])
        self.assertEqual((decisions[0].decision, decisions[0].reason), ('refused_gate', 'max_cost'))
        self.assertAlmostEqual(decisions[0].cost, 13.0)

    def test_err_uses_only_known_velocities(self):
        a = ep('a', 0, (0., 0.), (5., 0.))
        b = ep('b', 2, (4., 0.), vel_known=False)  # forward error only: |4 - 10| = 6
        _, decisions = link_tracklets([a], [b], CFG)
        self.assertAlmostEqual(decisions[0].cost, 0.7 * 4 / 2 + 0.3 * 6 / 2)

    def test_equidistant_starts_refused_margin_with_both_costs(self):
        a = ep('a', 10, (100., 100.), vel_known=False)
        b1 = ep('b1', 12, (103., 100.), vel_known=False)
        b2 = ep('b2', 12, (97., 100.), vel_known=False)
        links, decisions = link_tracklets([a], [b1, b2], CFG)
        self.assertEqual(links, [])
        labels = sorted(d.decision for d in decisions)
        self.assertEqual(labels, ['refused_assignment', 'refused_margin'])
        for d in decisions:
            self.assertAlmostEqual(d.cost, 1.5)
            self.assertAlmostEqual(d.runner_up_cost, 1.5)
        margin = next(d for d in decisions if d.decision == 'refused_margin')
        self.assertEqual(margin.reason.count('1.5000'), 2)

    def test_two_by_two_ambiguity_gives_two_refused_margin_rows(self):
        ends = [ep('a1', 10, (0., 0.), vel_known=False), ep('a2', 10, (0., 6.), vel_known=False)]
        starts = [ep('b1', 12, (1., 0.), vel_known=False), ep('b2', 12, (1., 6.), vel_known=False)]
        links, decisions = link_tracklets(ends, starts, CFG)
        self.assertEqual(links, [])
        margins = [d for d in decisions if d.decision == 'refused_margin']
        self.assertEqual({(d.tracklet_a, d.tracklet_b) for d in margins},
                         {('a1', 'b1'), ('a2', 'b2')})
        for d in margins:
            self.assertAlmostEqual(d.cost, 0.5)
            self.assertAlmostEqual(d.runner_up_cost, math.sqrt(37) / 2)
            self.assertIn(f"{d.cost:.4f}", d.reason)
            self.assertIn(f"{d.runner_up_cost:.4f}", d.reason)
        others = [d for d in decisions if d.decision == 'refused_assignment']
        self.assertEqual(len(others), 2)
        for d in others:
            self.assertAlmostEqual(d.runner_up_cost, 0.5)

    def test_clear_winner_links_despite_far_competitor(self):
        a = ep('a', 10, (0., 0.), vel_known=False)
        near = ep('near', 12, (1., 0.), vel_known=False)
        far = ep('far', 12, (12., 0.), vel_known=False)
        links, decisions = link_tracklets([a], [near, far], CFG)
        self.assertEqual(links, [('a', 'near')])
        d = by_pair(decisions)
        self.assertAlmostEqual(d[('a', 'near')].runner_up_cost, 6.0)
        self.assertEqual(d[('a', 'far')].decision, 'refused_assignment')
        self.assertAlmostEqual(d[('a', 'far')].runner_up_cost, 0.5)

    def test_missing_shoulder_links_on_body_only(self):
        a = ep('a', 0, (0., 0.), (1., 0.), heading=0.0, shoulder=(20., 0.))
        b = ep('b', 3, (3., 0.), (1., 0.), heading=math.nan, shoulder=(20., 0.))
        links, decisions = link_tracklets([a], [b], CFG)
        self.assertEqual(links, [('a', 'b')])
        self.assertAlmostEqual(decisions[0].cost, 0.7 * 1.0)
        b_rot = replace(b, heading=math.pi / 2)  # u_a=(20,0), u_b=(0,20): |du|=28.3 -> cap 12
        _, rotated = link_tracklets([a], [b_rot], CFG)
        self.assertAlmostEqual(rotated[0].cost, 0.7 + 0.15 * 12.0)
        b_same = replace(b, heading=0.0, offsets=np.array(b.offsets))
        b_same.offsets[1] = (23., 4.)  # |du| = 5 < cap
        _, close = link_tracklets([a], [b_same], CFG)
        self.assertAlmostEqual(close[0].cost, 0.7 + 0.15 * 5.0)

    def test_chain_successor_not_counted_as_competitor(self):
        # A -> B -> C is one crab; (A, C) is B's successor, not A's alternative.
        a = ep('A', 10, (0., 0.), (1., 0.))
        b_start, b_end = ep('B', 12, (2., 0.), (1., 0.)), ep('B', 13, (3., 0.), (1., 0.))
        c = ep('C', 15, (5., 0.), (1., 0.))
        links, decisions = link_tracklets([a, b_end], [b_start, c], CFG)
        self.assertEqual(links, [('A', 'B'), ('B', 'C')])
        d = by_pair(decisions)
        self.assertAlmostEqual(d[('A', 'C')].cost, d[('A', 'B')].cost)  # would veto pre-fix
        for key in (('A', 'B'), ('B', 'C')):
            self.assertTrue(math.isnan(d[key].runner_up_cost))
            self.assertIn('1 chain competitors ignored', d[key].reason)
        self.assertEqual(d[('A', 'B')].decision, 'linked')
        self.assertEqual(d[('A', 'B')].n_chain_ignored, 1)
        self.assertEqual(d[('A', 'C')].decision, 'refused_assignment')
        self.assertEqual(d[('A', 'C')].n_chain_ignored, 0)

    def test_chain_successor_only_row_exclusion(self):
        # (B, C) passes the motion gates but fails max_cost: C still counts as B's successor.
        cfg = replace(CFG, max_body_displacement_px=20.0)
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = ep('B', 12, (4., 0.), vel_known=False)
        b_end = ep('B', 13, (2., 0.), vel_known=False)
        c = ep('C', 15, (28., 0.), vel_known=False)
        links, decisions = link_tracklets([a, b_end], [b_start, c], cfg)
        d = by_pair(decisions)
        self.assertEqual((d[('B', 'C')].decision, d[('B', 'C')].reason),
                         ('refused_gate', 'max_cost'))
        self.assertLess(d[('A', 'C')].cost - d[('A', 'B')].cost, cfg.minimum_cost_margin)
        self.assertEqual(links, [('A', 'B')])
        self.assertTrue(math.isnan(d[('A', 'B')].runner_up_cost))
        self.assertIn('1 chain competitors ignored', d[('A', 'B')].reason)

    def test_chain_predecessor_only_column_exclusion(self):
        # Mirror: (D, A) is motion-gated (fails only max_cost), so (D, B) is not B's rival.
        cfg = replace(CFG, max_body_displacement_px=20.0)
        dd = ep('D', 10, (0., 0.), vel_known=False)
        a_start = ep('A', 12, (26., 0.), vel_known=False)
        a_end = ep('A', 13, (24., 0.), vel_known=False)
        b = ep('B', 15, (28., 0.), vel_known=False)
        links, decisions = link_tracklets([dd, a_end], [a_start, b], cfg)
        d = by_pair(decisions)
        self.assertEqual((d[('D', 'A')].decision, d[('D', 'A')].reason),
                         ('refused_gate', 'max_cost'))
        self.assertLess(d[('D', 'B')].cost - d[('A', 'B')].cost, cfg.minimum_cost_margin)
        self.assertEqual(links, [('A', 'B')])
        self.assertTrue(math.isnan(d[('A', 'B')].runner_up_cost))
        self.assertIn('1 chain competitors ignored', d[('A', 'B')].reason)

    def test_unenumerated_successor_pair_still_vetoes(self):
        # B is still present when C starts, so (B, C) is not enumerated and C stays a rival.
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = ep('B', 12, (2., 0.), vel_known=False)
        b_end = ep('B', 20, (2., 0.), vel_known=False)
        c = ep('C', 15, (0., 10.), vel_known=False)
        links, decisions = link_tracklets([a, b_end], [b_start, c], CFG)
        d = by_pair(decisions)
        self.assertNotIn(('B', 'C'), d)
        self.assertEqual(links, [])
        self.assertEqual(d[('A', 'B')].decision, 'refused_margin')
        self.assertAlmostEqual(d[('A', 'B')].runner_up_cost, 2.0)
        self.assertNotIn('chain', d[('A', 'B')].reason)

    def test_genuine_second_crab_still_vetoes(self):
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = ep('B', 12, (2., 0.), vel_known=False)
        b_end = ep('B', 20, (2., 0.), vel_known=False)
        c_start = ep('C', 12, (0., 3.), vel_known=False)
        c_end = ep('C', 20, (0., 3.), vel_known=False)
        links, decisions = link_tracklets([a, b_end, c_end], [b_start, c_start], CFG)
        d = by_pair(decisions)
        self.assertEqual(links, [])
        self.assertEqual(d[('A', 'B')].decision, 'refused_margin')
        self.assertAlmostEqual(d[('A', 'B')].runner_up_cost, 1.5)
        self.assertIn('1.0000', d[('A', 'B')].reason)
        self.assertIn('1.5000', d[('A', 'B')].reason)
        self.assertNotIn('chain', d[('A', 'B')].reason)

    def test_cross_video_never_links(self):
        a = ep('a', 0, (0., 0.), video='v1')
        b = ep('b', 2, (1., 0.), video='v2')
        with self.assertRaises(ValueError):
            link_tracklets([a], [b], CFG)
        self.assertEqual(link_tracklets([a], [replace(b, video_id='v1')], CFG)[0], [('a', 'b')])

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            link_tracklets([ep('a', 0, (0., 0.)), ep('a', 1, (0., 0.))], [], CFG)

    def test_decision_order_is_sorted_enumeration(self):
        ends = [ep('z', 5, (0., 0.)), ep('c', 3, (0., 0.)), ep('a', 5, (0., 0.))]
        starts = [ep('y', 7, (1., 0.)), ep('x', 6, (1., 0.)), ep('w', 7, (1., 0.))]
        _, decisions = link_tracklets(ends, starts, CFG)
        order = [(d.tracklet_a, d.tracklet_b) for d in decisions]
        self.assertEqual(order, [('c', 'x'), ('c', 'w'), ('c', 'y'),
                                 ('a', 'w'), ('a', 'y'), ('z', 'w'), ('z', 'y')])

    def test_large_input_enumerates_all_candidates(self):
        rng = np.random.default_rng(0)
        n = 3000
        first = rng.integers(0, 3000, n)
        last = first + rng.integers(0, 5, n)
        pos = rng.uniform(0, 2000, (n, 2))
        ends = [ep(f"t{k}", last[k], tuple(pos[k])) for k in range(n)]
        starts = [ep(f"t{k}", first[k], tuple(pos[k] + 3.0)) for k in range(n)]
        links, decisions = link_tracklets(ends, starts, CFG)
        gap = first[None, :] - last[:, None] - 1
        ok = (gap >= 1) & (gap <= CFG.max_gap)
        np.fill_diagonal(ok, False)
        self.assertEqual(len(decisions), int(ok.sum()))
        self.assertEqual(len({a for a, _ in links}), len(links))
        self.assertEqual(len({b for _, b in links}), len(links))


class LinearEquivalenceTests(unittest.TestCase):
    """``gate_model='linear'`` reproduces the pre-change code path exactly."""

    def assert_matches_legacy(self, ends, starts, cfg):
        links, decisions = link_tracklets(ends, starts, cfg)
        with mock.patch.object(candidates, '_pair_costs', legacy_pair_costs):
            ref_links, ref_decisions = link_tracklets(ends, starts, cfg)
        self.assertEqual(links, ref_links)
        self.assertEqual(len(decisions), len(ref_decisions))
        for d, ref in zip(decisions, ref_decisions):
            self.assertTrue(same_decision(d, ref), (d, ref))

    def test_fixed_scene_matches_legacy(self):
        ends = [ep('A', 10, (0., 0.), (1., 0.), heading=0.3, shoulder=(20., 3.)),
                ep('B', 13, (3., 0.), (1., 0.)), ep('D', 10, (0., 5.), vel_known=False),
                ep('E', 11, (40., 40.), (-2., 1.))]
        starts = [ep('B', 12, (2., 0.), (1., 0.), heading=0.1, shoulder=(19., 2.)),
                  ep('C', 15, (5., 0.), (1., 0.)), ep('F', 13, (1., 5.), vel_known=False),
                  ep('G', 14, (35., 43.), (-2., 1.)), ep('H', 16, (90., 0.))]
        for margin in (4.0, 2.5, 0.0):
            self.assert_matches_legacy(ends, starts, replace(CFG, minimum_cost_margin=margin))
        _, decisions = link_tracklets(ends, starts, CFG)
        self.assertEqual({d.decision for d in decisions},
                         set(DECISIONS))   # every label is exercised

    def test_configured_thresholds_apply_under_linear(self):
        a = ep('a', 10, (100., 100.), vel_known=False)
        b1 = ep('b1', 12, (103., 100.), vel_known=False)
        b2 = ep('b2', 12, (96., 100.), vel_known=False)   # costs 1.5 vs 2.0
        _, decisions = link_tracklets([a], [b1, b2], replace(CFG, minimum_cost_margin=0.75))
        margin = by_pair(decisions)[('a', 'b1')]
        self.assertEqual(margin.decision, 'refused_margin')
        self.assertTrue(margin.reason.endswith('< 0.75'))
        links, _ = link_tracklets([a], [b1, b2], replace(CFG, minimum_cost_margin=0.25))
        self.assertEqual(links, [('a', 'b1')])

    @settings(deadline=None, max_examples=150)
    @given(scenes(), st.integers(1, 30), st.sampled_from([0.0, 1.0, 4.0]))
    def test_random_scenes_match_legacy(self, scene, max_gap, margin):
        ends, starts = scene
        self.assert_matches_legacy(ends, starts,
                                   replace(CFG, max_gap=max_gap, minimum_cost_margin=margin))


class SqrtGateModelTests(unittest.TestCase):
    """Speed-extended sub-linear radius, dimensionless cost, gap penalty, capped horizon."""

    def penalty(self, dt: int) -> float:
        """Gap penalty (1 + p)^(gap - 1) = (1 + p)^(dt - 2) of the pinned SQRT config."""
        return 1.06 ** (dt - 2)

    def extended(self, cfg: BridgeConfig, dt: int, speed: float) -> float:
        """Extended radius R(dt, speed) as computed by the module."""
        return float(candidates._sqrt_extended_radius(cfg, np.array([dt]), np.array([speed]))[0])

    @settings(deadline=None, max_examples=200)
    @given(st.integers(1, 400), st.integers(0, 400), st.floats(0, 20), st.floats(0, 10),
           st.floats(0.1, 3), st.floats(0, 20), st.floats(0, 6), st.integers(1, 40))
    def test_gate_radius_monotone_and_linear_is_12_dt(self, dt, extra, r0, k, safety, floor,
                                                      speed, horizon):
        cfg = replace(SQRT, radius_r0_px=r0, radius_k_px=k, radius_safety=safety,
                      radius_min_px=floor, prediction_horizon_frames=horizon)
        self.assertLessEqual(radius(cfg, dt), radius(cfg, dt + extra))
        self.assertGreaterEqual(radius(cfg, dt), floor)
        ext = [self.extended(cfg, t, speed) for t in (dt, dt + extra)]
        self.assertLessEqual(ext[0], ext[1])                    # monotone in dt at fixed speed
        np.testing.assert_allclose(ext[0], radius(cfg, dt) + speed * min(dt, horizon), rtol=1e-12)
        self.assertEqual(radius(CFG, dt), 12.0 * dt)
        self.assertEqual(radius(CFG, dt), CFG.max_body_displacement_px * dt)

    def test_displacement_gate_is_gate_radius_without_velocity(self):
        cfg = replace(SQRT, max_gap=30)
        for dt in (2, 4, 6, 11, 31):
            r = radius(cfg, dt)
            self.assertLess(r, 12.0 * dt)
            a = ep('a', 10, (0., 0.), vel_known=False)
            inside = ep('b', 10 + dt, (r * (1 - 1e-9), 0.), vel_known=False)
            outside = ep('b', 10 + dt, (r * (1 + 1e-6), 0.), vel_known=False)
            _, d_in = link_tracklets([a], [inside], cfg)
            _, d_out = link_tracklets([a], [outside], cfg)
            self.assertNotEqual(d_in[0].reason, 'displacement')
            self.assertEqual((d_out[0].decision, d_out[0].reason), ('refused_gate', 'displacement'))
            self.assertTrue(math.isnan(d_out[0].cost))

    def test_walking_crab_links_across_six_frames(self):
        # 3 px/frame into a dt = 6 gap, reappearing 18 px on: R = r(6) + 3 * min(6, 3) = 20.4.
        r6 = radius(SQRT, 6)
        self.assertAlmostEqual(r6, 1.25 * 3.73 * math.sqrt(6))
        big_r = r6 + 9.
        walker = ep('a', 10, (0., 0.), (3., 0.))
        b = ep('b', 16, (18., 0.), (3., 0.))
        links, moving = link_tracklets([walker], [b], SQRT)  # err = |18 - 9| = 9
        expected = (0.7 * 18. + 0.3 * 9.) / big_r * self.penalty(6)
        self.assertLess(expected, 1.0)
        self.assertEqual(links, [('a', 'b')])
        self.assertAlmostEqual(moving[0].cost, expected)
        # Stationary (or velocity-unknown) endpoints on both sides: R = r(6) < 18.
        for still_a, still_b in ((replace(walker, vel=np.zeros(2)), replace(b, vel=np.zeros(2))),
                                 (replace(walker, vel_known=False), replace(b, vel_known=False))):
            _, d = link_tracklets([still_a], [still_b], SQRT)
            self.assertEqual((d[0].decision, d[0].reason), ('refused_gate', 'displacement'))

    def test_extended_radius_is_symmetric_in_end_and_start_speed(self):
        walker = ep('a', 10, (0., 0.), (3., 0.))
        b = ep('b', 16, (18., 0.), (3., 0.))
        moving_end = link_tracklets([walker], [replace(b, vel=np.zeros(2))], SQRT)[1][0]
        moving_start = link_tracklets([replace(walker, vel=np.zeros(2))], [b], SQRT)[1][0]
        big_r = radius(SQRT, 6) + 9.
        err = (9. + 18.) / 2                            # forward 9, backward 18
        expected = (0.7 * 18. + 0.3 * err) / big_r * self.penalty(6)
        for d in (moving_end, moving_start):
            self.assertNotEqual(d.reason, 'displacement')
            self.assertAlmostEqual(d.cost, expected)
        unknown_end = link_tracklets([replace(walker, vel_known=False)], [b], SQRT)[1][0]
        self.assertNotEqual(unknown_end.reason, 'displacement')   # start speed still counts

    def test_cost_formula(self):
        cfg = replace(SQRT, max_gap=30)
        dt = 8
        a = ep('a', 0, (0., 0.), (1., 0.), heading=0.0, shoulder=(20., 0.))
        b = ep('b', dt, (4., 3.), vel_known=False, heading=0.0, shoulder=(25., 0.))
        _, decisions = link_tracklets([a], [b], cfg)
        big_r = radius(cfg, dt) + 1. * 3                 # speed max(1, 0), horizon min(8, 3)
        err = math.hypot(4. - 3., 3.)                    # forward only
        expected = (0.7 * 5. + 0.3 * err) / big_r * self.penalty(dt) + 0.15 * 5. / 12.
        self.assertAlmostEqual(decisions[0].cost, expected, places=12)
        self.assertEqual(decisions[0].decision, 'linked')
        self.assertEqual(decisions[0].reason, f"cost {expected:.4f} no competitor")

    def test_one_frame_gap_has_no_penalty(self):
        a = ep('a', 10, (0., 0.), vel_known=False)
        b = ep('b', 12, (3., 0.), vel_known=False)
        self.assertAlmostEqual(link_tracklets([a], [b], SQRT)[1][0].cost, 3. / radius(SQRT, 2))
        self.assertAlmostEqual(link_tracklets([a], [b], replace(SQRT, gap_penalty=0.5))[1][0].cost,
                               3. / radius(SQRT, 2))

    def test_cost_thresholds(self):
        scale = SQRT.cost_thresholds()
        self.assertEqual((scale['max_assignment_cost'], scale['unmatched_cost'],
                          scale['minimum_cost_margin']), (1.0, 0.6, 0.25))
        a = ep('a', 10, (0., 0.), vel_known=False)
        r = radius(SQRT, 2)
        b1 = ep('b1', 12, (0.30 * r, 0.), vel_known=False)
        b2 = ep('b2', 12, (-0.50 * r, 0.), vel_known=False)   # margin 0.2 < 0.25
        links, decisions = link_tracklets([a], [b1, b2], SQRT)
        self.assertEqual(links, [])
        d = by_pair(decisions)[('a', 'b1')]
        self.assertEqual(d.decision, 'refused_margin')
        self.assertTrue(d.reason.endswith('< 0.25'))
        b2_far = replace(b2, pos=np.array([-0.56 * r, 0.]))    # margin 0.26 >= 0.25
        self.assertEqual(link_tracklets([a], [b1, b2_far], SQRT)[0], [('a', 'b1')])
        # A pair inside both motion gates but above cost 1.0 fails max_cost and keeps its cost.
        far = ep('far', 13, (0.99 * radius(SQRT, 3), 0.), vel_known=False)
        _, rows = link_tracklets([a], [far], SQRT)
        self.assertEqual((rows[0].decision, rows[0].reason), ('refused_gate', 'max_cost'))
        self.assertAlmostEqual(rows[0].cost, 0.99 * self.penalty(3))

    def test_sqrt_thresholds_are_configurable(self):
        a = ep('a', 10, (0., 0.), vel_known=False)
        r = radius(SQRT, 2)
        b1 = ep('b1', 12, (0.30 * r, 0.), vel_known=False)
        b2 = ep('b2', 12, (-0.50 * r, 0.), vel_known=False)   # margin 0.2
        loose = replace(SQRT, minimum_cost_margin_sqrt=0.15)
        links, decisions = link_tracklets([a], [b1, b2], loose)
        self.assertEqual(links, [('a', 'b1')])
        self.assertEqual(by_pair(decisions)[('a', 'b1')].reason, 'cost 0.3000 margin 0.2000')
        strict = replace(SQRT, minimum_cost_margin_sqrt=0.21)
        self.assertTrue(by_pair(link_tracklets([a], [b1, b2], strict)[1])[('a', 'b1')]
                        .reason.endswith('< 0.21'))
        # The px-scale (linear) thresholds do not affect the sqrt model.
        self.assertEqual(link_tracklets([a], [b1, b2], replace(loose, minimum_cost_margin=100.))[0],
                         [('a', 'b1')])
        far = ep('far', 13, (0.99 * radius(SQRT, 3), 0.), vel_known=False)  # cost 1.049
        self.assertEqual(link_tracklets([a], [far], replace(SQRT, max_assignment_cost_sqrt=1.1))[0],
                         [('a', 'far')])
        self.assertEqual(link_tracklets([a], [b1], replace(SQRT, unmatched_cost_sqrt=0.1))[0], [])

    def test_gap_penalty_prefers_short_gap_at_equal_pixel_distance(self):
        # (1.06)^29 = 5.4 outweighs r(31)/r(2) = 3.94, so the 1-frame gap is cheaper.
        cfg = replace(SQRT, max_gap=30)
        a = ep('a', 10, (0., 0.), vel_known=False)
        short = ep('short', 12, (6., 0.), vel_known=False)     # gap 1
        long_ = ep('long', 41, (0., 6.), vel_known=False)      # gap 30
        links, decisions = link_tracklets([a], [short, long_], cfg)
        self.assertEqual(links, [('a', 'short')])
        d = by_pair(decisions)
        self.assertAlmostEqual(d[('a', 'short')].cost, 6. / radius(cfg, 2))
        self.assertAlmostEqual(d[('a', 'long')].cost, 6. / radius(cfg, 31) * 1.06 ** 29)
        self.assertGreater(1.06 ** 29, radius(cfg, 31) / radius(cfg, 2))
        self.assertEqual(d[('a', 'long')].decision, 'refused_gate')   # 1.25 > max cost 1.0
        self.assertEqual(d[('a', 'long')].reason, 'max_cost')

    def test_gap_penalty_at_equal_normalised_distance(self):
        cfg = replace(SQRT, max_gap=30)
        a = ep('a', 10, (0., 0.), vel_known=False)
        short = ep('short', 12, (0.15 * radius(cfg, 2), 0.), vel_known=False)
        long_ = ep('long', 41, (0., 0.15 * radius(cfg, 31)), vel_known=False)
        links, decisions = link_tracklets([a], [short, long_], cfg)
        self.assertEqual(links, [('a', 'short')])
        d = by_pair(decisions)
        self.assertAlmostEqual(d[('a', 'short')].cost, 0.15)
        self.assertAlmostEqual(d[('a', 'long')].cost, 0.15 * 1.06 ** 29)
        self.assertEqual(d[('a', 'long')].decision, 'refused_assignment')

    def test_capped_prediction_horizon(self):
        cfg = replace(SQRT, max_gap=30)
        dt = 10
        a = ep('a', 10, (0., 0.), (6., 0.))
        # (i) A 6 px/frame crab that stops: reappears 10 px on after dt = 10.
        stop = ep('b', 10 + dt, (10., 0.), (6., 0.))
        links, d = link_tracklets([a], [stop], cfg)          # err = |10 - 6 * 3| = 8
        big_r = radius(cfg, dt) + 6. * 3
        self.assertEqual(links, [('a', 'b')])
        self.assertAlmostEqual(d[0].cost, (0.7 * 10. + 0.3 * 8.) / big_r * self.penalty(dt))
        uncapped = replace(cfg, prediction_horizon_frames=dt)  # err = |10 - 60| = 50
        _, d = link_tracklets([a], [stop], uncapped)
        self.assertAlmostEqual(d[0].cost, (0.7 * 10. + 0.3 * 50.) / (radius(cfg, dt) + 60.)
                               * self.penalty(dt))
        # (ii) The cap bounds the directed extension: a crab that keeps walking 60 px is
        # outside R = r(10) + 18 with h = 3, but inside both gates with h = 10.
        walk = ep('b', 10 + dt, (60., 0.), (6., 0.))
        stats: dict = {}
        self.assertEqual(link_tracklets([a], [walk], cfg, stats=stats), ([], []))  # beyond R_max
        self.assertEqual(stats, {'enumerated': 0, 'pruned_by_radius': 1})
        # 45 px is still outside R(10) = 32.7 but inside R_max(60) = r(61) + 18 = 54.4: a row.
        nearer = replace(walk, pos=np.array([45., 0.]))
        self.assertEqual(link_tracklets([a], [nearer], replace(cfg, max_gap=60))[1][0].reason,
                         'displacement')
        self.assertNotIn(link_tracklets([a], [walk], uncapped)[1][0].reason,
                         ('displacement', 'prediction_error'))
        # Linear extrapolates over the full dt (err 50) and its gates are 120 and 180 px.
        links, d = link_tracklets([a], [stop], replace(CFG, max_gap=30))
        self.assertEqual(links, [('a', 'b')])
        self.assertAlmostEqual(d[0].cost, 0.7 * 10. / dt + 0.3 * 50. / dt)

    def test_horizon_longer_than_gap_changes_nothing(self):
        a = ep('a', 10, (0., 0.), (2., 0.))
        b = ep('b', 12, (4.5, 0.), (2., 0.))
        cfg = replace(SQRT, prediction_horizon_frames=3)
        c1 = link_tracklets([a], [b], cfg)[1][0].cost
        c2 = link_tracklets([a], [b], replace(cfg, prediction_horizon_frames=50))[1][0].cost
        self.assertEqual(c1, c2)


class ChainAwareBothModelsTests(unittest.TestCase):
    """Chain-reachable competitors are ignored under either gate model."""

    def test_chain_successor_not_counted_as_competitor(self):
        for name, cfg in MODELS.items():
            with self.subTest(model=name):
                a = ep('A', 10, (0., 0.), (1., 0.))
                b_start, b_end = ep('B', 12, (2., 0.), (1., 0.)), ep('B', 13, (3., 0.), (1., 0.))
                c = ep('C', 15, (5., 0.), (1., 0.))
                links, decisions = link_tracklets([a, b_end], [b_start, c], cfg)
                self.assertEqual(links, [('A', 'B'), ('B', 'C')])
                d = by_pair(decisions)
                margin = cfg.cost_thresholds()['minimum_cost_margin']
                self.assertLess(d[('A', 'C')].cost - d[('A', 'B')].cost, margin)  # would veto
                for key in (('A', 'B'), ('B', 'C')):
                    self.assertTrue(math.isnan(d[key].runner_up_cost))
                    self.assertIn('1 chain competitors ignored', d[key].reason)
                    self.assertEqual(d[key].n_chain_ignored, 1)
                self.assertEqual(d[('A', 'C')].decision, 'refused_assignment')
                self.assertEqual(d[('A', 'C')].n_chain_ignored, 0)

    def test_genuine_second_crab_still_vetoes(self):
        for name, cfg in MODELS.items():
            with self.subTest(model=name):
                a = ep('A', 10, (0., 0.), vel_known=False)
                b_start = ep('B', 12, (2., 0.), vel_known=False)
                b_end = ep('B', 20, (2., 0.), vel_known=False)
                c_start = ep('C', 12, (0., 3.), vel_known=False)
                c_end = ep('C', 20, (0., 3.), vel_known=False)
                links, decisions = link_tracklets([a, b_end, c_end], [b_start, c_start], cfg)
                d = by_pair(decisions)
                self.assertEqual(links, [])
                self.assertEqual(d[('A', 'B')].decision, 'refused_margin')
                self.assertEqual(d[('A', 'B')].runner_up_cost, d[('A', 'C')].cost)
                self.assertEqual(d[('A', 'B')].n_chain_ignored, 0)
                self.assertNotIn('chain', d[('A', 'B')].reason)

    def sqrt_row_scene(self):
        """(B, C) passes the motion gates but fails max_cost under SQRT."""
        r3 = radius(SQRT, 3)
        c_x = 5.5
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = ep('B', 12, (3., 0.), vel_known=False)
        b_end = ep('B', 13, (c_x - 0.98 * r3, 0.), vel_known=False)
        c = ep('C', 16, (c_x, 0.), vel_known=False)
        return [a, b_end], [b_start, c]

    def test_sqrt_chain_successor_only_row_exclusion(self):
        ends, starts = self.sqrt_row_scene()
        links, decisions = link_tracklets(ends, starts, SQRT)
        d = by_pair(decisions)
        self.assertEqual((d[('B', 'C')].decision, d[('B', 'C')].reason),
                         ('refused_gate', 'max_cost'))
        self.assertAlmostEqual(d[('B', 'C')].cost, 0.98 * 1.06)
        self.assertLess(d[('A', 'C')].cost - d[('A', 'B')].cost, 0.25)
        self.assertEqual(links, [('A', 'B')])
        self.assertTrue(math.isnan(d[('A', 'B')].runner_up_cost))
        self.assertIn('1 chain competitors ignored', d[('A', 'B')].reason)

    def test_sqrt_chain_predecessor_only_column_exclusion(self):
        # Time-reversed row scene: (C, B) motion-gated, so (C, A) is not A's rival for B.
        ends, starts = reversed_scene(*self.sqrt_row_scene())
        links, decisions = link_tracklets(ends, starts, SQRT)
        d = by_pair(decisions)
        self.assertEqual((d[('C', 'B')].decision, d[('C', 'B')].reason),
                         ('refused_gate', 'max_cost'))
        self.assertLess(d[('C', 'A')].cost - d[('B', 'A')].cost, 0.25)
        self.assertEqual(links, [('B', 'A')])
        self.assertTrue(math.isnan(d[('B', 'A')].runner_up_cost))
        self.assertIn('1 chain competitors ignored', d[('B', 'A')].reason)

    def test_sqrt_unenumerated_successor_pair_still_vetoes(self):
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = ep('B', 12, (2., 0.), vel_known=False)
        b_end = ep('B', 20, (2., 0.), vel_known=False)
        c = ep('C', 15, (0., 4.), vel_known=False)
        links, decisions = link_tracklets([a, b_end], [b_start, c], SQRT)
        d = by_pair(decisions)
        self.assertNotIn(('B', 'C'), d)
        self.assertEqual(links, [])
        self.assertEqual(d[('A', 'B')].decision, 'refused_margin')
        self.assertAlmostEqual(d[('A', 'B')].runner_up_cost, 4. / radius(SQRT, 5) * 1.06 ** 3)
        self.assertNotIn('chain', d[('A', 'B')].reason)


class StagedLinkingTests(unittest.TestCase):
    """Sqrt linking runs stage by stage, shortest gaps first."""

    def abc_scene(self):
        """A -> B (gap 2) -> C (gap 15) at about the same spot; A -> C has gap 25."""
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = ep('B', 13, (0.3, 0.), vel_known=False)
        b_end = ep('B', 20, (0.3, 0.2), vel_known=False)
        c = ep('C', 36, (0.1, 0.4), vel_known=False)
        return [a, b_end], [b_start, c]

    def test_stages_from_config(self):
        for max_gap, expected in ((1, (1,)), (5, (5,)), (10, (5, 10)), (30, (5, 15, 30)),
                                  (60, (5, 15, 30, 45, 60))):
            self.assertEqual(candidates._stages(replace(SQRT, max_gap=max_gap)), expected)
        self.assertEqual(candidates._stages(replace(SQRT, max_gap=7, link_stages=(2, 7, 9))),
                         (2, 7))
        self.assertEqual(candidates._stages(replace(CFG, max_gap=30)), (30,))
        with self.assertRaises(ValueError):
            replace(SQRT, link_stages=(5, 5))

    def test_short_gap_chain_survives_longer_horizon(self):
        ends, starts = self.abc_scene()
        cfg = replace(SQRT, max_gap=30)
        links, decisions = link_tracklets(ends, starts, cfg)
        self.assertEqual(sorted(links), [('A', 'B'), ('B', 'C')])
        d = by_pair(decisions)
        self.assertEqual([d[p].stage for p in (('A', 'B'), ('B', 'C'), ('A', 'C'))], [1, 2, 3])
        self.assertEqual(d[('A', 'C')].decision, 'refused_assignment')
        self.assertEqual(d[('A', 'C')].reason, 'endpoint linked at stage 1')
        self.assertEqual(d[('A', 'C')].runner_up_cost,                # both ends linked earlier
                         min(d[('A', 'B')].cost, d[('B', 'C')].cost))

    def test_short_link_not_lost_to_cheaper_long_gap(self):
        # The demo defect: A ends; B starts 2 frames later and runs on; a one-frame blip C,
        # 15 frames after A and during B, is cheaper for A than B. One global solve picks
        # (A, C), the margin vetoes it (runner-up (A, B)), and A stays unlinked.
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start, b_end = ep('B', 13, (0.15, -3.6), vel_known=False), ep('B', 150, (30., -20.))
        c_start, c_end = ep('C', 26, (1.6, -2.2), vel_known=False), ep('C', 26, (1.6, -2.2))
        ends, starts = [a, b_end, c_end], [b_start, c_start]
        cfg = replace(SQRT, max_gap=30)
        single, single_dec = link_tracklets(ends, starts, replace(cfg, link_stages=(30,)))
        self.assertEqual(single, [])
        s = by_pair(single_dec)
        self.assertLess(s[('A', 'C')].cost, s[('A', 'B')].cost)
        self.assertEqual(s[('A', 'C')].decision, 'refused_margin')
        links, decisions = link_tracklets(ends, starts, cfg)
        self.assertEqual(links, [('A', 'B')])
        d = by_pair(decisions)
        self.assertEqual((d[('A', 'B')].stage, d[('A', 'C')].stage), (1, 2))
        self.assertEqual(d[('A', 'C')].reason, 'endpoint linked at stage 1')

    def test_long_gap_alone_links_at_stage_2(self):
        a = ep('A', 10, (0., 0.), vel_known=False)
        b = ep('B', 21, (1., 0.), vel_known=False)                    # gap 10
        links, decisions = link_tracklets([a], [b], replace(SQRT, max_gap=30))
        self.assertEqual(links, [('A', 'B')])
        self.assertEqual((decisions[0].stage, decisions[0].gap_frames), (2, 10))

    def test_chain_reachability_spans_stages(self):
        # Stage 2 holds (A, B) and (A, C), within the margin. C is B's successor through the
        # stage-1 pair (B, C), which fails only max_cost (so C stays free); chain reachability
        # is built over all gaps, so (A, C) is ignored and A -> B links.
        r4 = radius(SQRT, 4)
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = ep('B', 18, (2., 0.), vel_known=False)             # (A, B) gap 7
        b_end = ep('B', 19, (3. - 0.98 * r4, 0.), vel_known=False)
        c = ep('C', 23, (3., 0.), vel_known=False)                   # (B, C) gap 3, (A, C) gap 12
        links, decisions = link_tracklets([a, b_end], [b_start, c], replace(SQRT, max_gap=30))
        d = by_pair(decisions)
        self.assertEqual((d[('B', 'C')].stage, d[('B', 'C')].reason), (1, 'max_cost'))
        self.assertEqual((d[('A', 'B')].stage, d[('A', 'C')].stage), (2, 2))
        self.assertLess(d[('A', 'C')].cost - d[('A', 'B')].cost, 0.25)
        self.assertEqual(links, [('A', 'B')])
        self.assertTrue(math.isnan(d[('A', 'B')].runner_up_cost))
        self.assertEqual(d[('A', 'B')].n_chain_ignored, 1)

    def test_linear_is_single_stage(self):
        ends, starts = self.abc_scene()
        _, decisions = link_tracklets(ends, starts, replace(CFG, max_gap=30))
        self.assertEqual({d.stage for d in decisions}, {1})

    def veto_scene(self, spacing: float = 1.5, c_gap: int = 9, c_dist: float = 3.0):
        """A ends at f10; B, B2 start at f12 at (0, +-spacing); C starts c_gap frames on."""
        a = ep('A', 10, (0., 0.), vel_known=False)
        b, b2 = ep('B', 12, (0., spacing), vel_known=False), ep('B2', 12, (0., -spacing),
                                                                vel_known=False)
        b_end, b2_end = ep('B', 200, (0., spacing)), ep('B2', 200, (0., -spacing))
        c = ep('C', 11 + c_gap, (c_dist, 0.), vel_known=False)
        return [a, b_end, b2_end], [b, b2, c]

    def test_regression_D001_vetoed_endpoint_not_relinked(self):
        """D-001: A's stage-1 veto (two equal gap-1 successors) holds at stage 2."""
        ends, starts = self.veto_scene()
        links5, dec5 = link_tracklets(ends, starts, replace(SQRT, max_gap=5))
        self.assertEqual(links5, [])
        self.assertEqual([d.decision for d in dec5 if d.decision == 'refused_margin'],
                         ['refused_margin'])
        links, decisions = link_tracklets(ends, starts, replace(SQRT, max_gap=15))
        self.assertEqual(links, [])                                    # was [('A', 'C')]
        d = by_pair(decisions)
        vetoed = next(x for x in decisions if x.decision == 'refused_margin')
        self.assertEqual((vetoed.tracklet_a, vetoed.stage), ('A', 1))
        self.assertEqual(vetoed.cost, vetoed.runner_up_cost)            # margin 0
        self.assertEqual((d[('A', 'C')].stage, d[('A', 'C')].gap_frames), (2, 9))
        self.assertEqual(d[('A', 'C')].decision, 'refused_assignment')
        self.assertEqual(d[('A', 'C')].reason, 'endpoint vetoed at stage 1')
        self.assertEqual(d[('A', 'C')].runner_up_cost, vetoed.cost)
        self.assertLessEqual(d[('A', 'C')].cost, 1.0)                   # passes every gate
        # Edge: a gate-failing later pair keeps its plain refused_gate label.
        far = replace(starts[2], pos=np.array([0.99 * radius(SQRT, 10), 0.]))
        _, far_dec = link_tracklets(ends, starts[:2] + [far], replace(SQRT, max_gap=15))
        far_row = by_pair(far_dec)[('A', 'C')]
        self.assertEqual((far_row.decision, far_row.reason), ('refused_gate', 'max_cost'))
        # Linear has one stage, so the scene is unaffected there.
        self.assertEqual(link_tracklets(ends, starts, replace(CFG, max_gap=15))[0], [])

    @settings(deadline=None, max_examples=100)
    @example(spacing=1.5, c_gap=9, c_dist=3.0, max_gap=15)
    @given(st.floats(0.0, 4.0), st.integers(6, 30), st.floats(0.0, 8.0), st.integers(6, 30))
    def test_vetoed_endpoint_never_linked(self, spacing, c_gap, c_dist, max_gap):
        """Symmetric gap-1 successors veto A at stage 1; no later candidate links A."""
        ends, starts = self.veto_scene(spacing, c_gap, c_dist)
        links, decisions = link_tracklets(ends, starts, replace(SQRT, max_gap=max_gap))
        self.assertNotIn('A', {a for a, _ in links})
        for d in decisions:
            if d.tracklet_a == 'A' and d.stage >= 2:
                self.assertIn(d.decision, ('refused_assignment', 'refused_gate'))
                if d.decision == 'refused_assignment':
                    self.assertEqual(d.reason, 'endpoint vetoed at stage 1')


    @settings(deadline=None, max_examples=150)
    @given(scenes(), st.integers(5, 30))
    def test_first_stage_links_do_not_depend_on_max_gap(self, scene, max_gap):
        ends, starts = scene
        short, _ = link_tracklets(ends, starts, replace(SQRT, max_gap=5))
        links, decisions = link_tracklets(ends, starts, replace(SQRT, max_gap=max_gap))
        gap = {(d.tracklet_a, d.tracklet_b): d.gap_frames for d in decisions}
        self.assertTrue(set(short) <= {p for p in links if gap[p] <= 5})

    @unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
    def test_demo_short_links_kept_at_max_gap_30(self):
        import pyarrow.parquet as pq
        cfg = BridgeConfig()
        eps = geometry.endpoints(pq.read_table(DEMO), cfg)
        ends = [e for _, e in eps.values()]
        starts = [s for s, _ in eps.values()]
        links5, _ = link_tracklets(ends, starts, replace(cfg, max_gap=5))
        links30, dec30 = link_tracklets(ends, starts, replace(cfg, max_gap=30))
        gap = {(d.tracklet_a, d.tracklet_b): d.gap_frames for d in dec30}
        self.assertEqual(len(set(links5) - {p for p in links30 if gap[p] <= 5}), 0)  # measured 0
        self.assertGreater(len(links30), len(links5))


class RadiusPrefilterTests(unittest.TestCase):
    """Under sqrt, pairs beyond R_max are not enumerated; nothing the gates keep is lost."""

    def test_regression_D004_far_pairs_not_enumerated(self):
        """D-004: a pair beyond R_max has no audit row and is counted as pruned."""
        cfg = replace(SQRT, max_gap=5)
        r_max = float(cfg.gate_radius(6)) + 6.0 * 3                     # 29.42 px
        a = ep('a', 10, (0., 0.), vel_known=False)
        at = ep('at', 12, (r_max, 0.), vel_known=False)                  # on the bound: kept
        beyond = ep('beyond', 12, (0., r_max * (1 + 1e-9)), vel_known=False)
        stats: dict = {'stale': 1}
        _, decisions = link_tracklets([a], [at, beyond], cfg, stats=stats)
        self.assertEqual([(d.tracklet_b, d.decision, d.reason) for d in decisions],
                         [('at', 'refused_gate', 'displacement')])
        self.assertEqual(stats, {'enumerated': 1, 'pruned_by_radius': 1})
        # Linear never prunes: both pairs keep their v001 refused_gate rows.
        _, lin = link_tracklets([a], [at, beyond], replace(CFG, max_gap=5), stats=stats)
        self.assertEqual(len(lin), 2)
        self.assertEqual(stats, {'enumerated': 2, 'pruned_by_radius': 0})

    def test_fast_endpoint_widens_bound(self):
        # An endpoint faster than the speed cap still links: R_max uses the larger speed.
        cfg = replace(SQRT, max_gap=5)
        a = ep('a', 10, (0., 0.), (12., 0.))
        b = ep('b', 16, (40., 0.), (12., 0.))          # R = r(6) + 36 = 47.4; capped R_max 29.4
        self.assertEqual(link_tracklets([a], [b], cfg)[0], [('a', 'b')])

    def test_empty_inputs_fill_stats(self):
        stats: dict = {}
        self.assertEqual(link_tracklets([], [], SQRT, stats=stats), ([], []))
        self.assertEqual(stats, {'enumerated': 0, 'pruned_by_radius': 0})

    @settings(deadline=None, max_examples=150)
    @given(scenes(), st.integers(1, 30), st.sampled_from(sorted(MODELS)),
           st.sampled_from([(5, 15, 30, 45), (1, 2, 3), (30,)]))
    def test_prefilter_matches_unpruned_run(self, scene, max_gap, model, link_stages):
        ends, starts = scene
        cfg = replace(MODELS[model], max_gap=max_gap, link_stages=link_stages)
        stats: dict = {}
        links, decisions = link_tracklets(ends, starts, cfg, stats=stats)
        with mock.patch.object(candidates, '_prune_radius', lambda *args: math.inf):
            ref_links, ref_decisions = link_tracklets(ends, starts, cfg)
        self.assertEqual(links, ref_links)
        ref = by_pair(ref_decisions)
        kept = audited(ends, starts, cfg)
        self.assertEqual(set(by_pair(decisions)), kept)
        self.assertEqual(stats, {'enumerated': len(decisions),
                                 'pruned_by_radius': len(ref_decisions) - len(decisions)})
        for d in decisions:
            self.assertTrue(same_decision(d, ref[(d.tracklet_a, d.tracklet_b)]), d)
        for key in set(ref) - kept:
            self.assertEqual((ref[key].decision, ref[key].reason), ('refused_gate', 'displacement'))
        if model == 'linear':
            self.assertEqual(stats['pruned_by_radius'], 0)


# ============================================================
# PROPERTY TESTS
# ============================================================

class CandidatePropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=60)
    @given(scenes(gap_free=True), st.integers(1, 30), st.sampled_from(sorted(MODELS)))
    def test_gap_free_input_gives_nothing(self, scene, max_gap, model):
        ends, starts = scene
        cfg = replace(MODELS[model], max_gap=max_gap)
        self.assertEqual(link_tracklets(ends, starts, cfg), ([], []))

    @settings(deadline=None, max_examples=250)
    @given(scenes(), st.integers(1, 30), st.sampled_from(sorted(MODELS)),
           st.sampled_from([(5, 15, 30, 45), (1, 2, 3), (2, 8), (30,)]))
    def test_audit_and_link_invariants(self, scene, max_gap, model, link_stages):
        ends, starts = scene
        cfg = replace(MODELS[model], max_gap=max_gap, link_stages=link_stages)
        scale = cfg.cost_thresholds()
        stages = candidates._stages(cfg)
        links, decisions = link_tracklets(ends, starts, cfg)
        linked_at, vetoed_at = {}, {}
        for d in decisions:
            if d.decision == 'linked':
                linked_at[('end', d.tracklet_a)] = linked_at[('start', d.tracklet_b)] = d
            elif d.decision == 'refused_margin':
                self.assertNotIn(('end', d.tracklet_a), vetoed_at)     # vetoed at most once
                self.assertNotIn(('start', d.tracklet_b), vetoed_at)
                vetoed_at[('end', d.tracklet_a)] = vetoed_at[('start', d.tracklet_b)] = d
        # A margin-vetoed endpoint is never linked at any stage.
        self.assertFalse(set(linked_at) & set(vetoed_at))
        pairs = [(d.tracklet_a, d.tracklet_b) for d in decisions]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(set(pairs), audited(ends, starts, cfg))
        self.assertEqual(len({a for a, _ in links}), len(links))
        self.assertEqual(len({b for _, b in links}), len(links))
        self.assertEqual(links, [p for p, d in zip(pairs, decisions) if d.decision == 'linked'])
        for d in decisions:
            self.assertIn(d.decision, DECISIONS)
            self.assertTrue(1 <= d.gap_frames <= max_gap)
            lo = stages[d.stage - 2] if d.stage >= 2 else 0
            self.assertTrue(lo < d.gap_frames <= stages[d.stage - 1])   # stage holds its gap
            if d.reason.startswith(EXCLUDED):
                # An endpoint was linked at an earlier stage; that link is the runner-up.
                self.assertEqual(d.decision, 'refused_assignment')
                earlier = [x for x in (linked_at.get(('end', d.tracklet_a)),
                                       linked_at.get(('start', d.tracklet_b)))
                           if x is not None and x.stage < d.stage]
                self.assertTrue(earlier)
                self.assertEqual(int(d.reason[len(EXCLUDED):]), min(x.stage for x in earlier))
                self.assertEqual(d.runner_up_cost, min(x.cost for x in earlier))
            elif d.reason.startswith(VETOED):
                # No endpoint linked earlier, but one was margin-vetoed at an earlier stage.
                self.assertEqual(d.decision, 'refused_assignment')
                earlier = [x for x in (vetoed_at.get(('end', d.tracklet_a)),
                                       vetoed_at.get(('start', d.tracklet_b)))
                           if x is not None and x.stage < d.stage]
                self.assertTrue(earlier)
                self.assertEqual(int(d.reason[len(VETOED):]), min(x.stage for x in earlier))
                self.assertEqual(d.runner_up_cost, min(x.cost for x in earlier))
            elif d.decision == 'refused_assignment':
                # Not excluded: neither endpoint was linked or vetoed at an earlier stage.
                for key in (('end', d.tracklet_a), ('start', d.tracklet_b)):
                    self.assertFalse(key in linked_at and linked_at[key].stage < d.stage)
                    self.assertFalse(key in vetoed_at and vetoed_at[key].stage < d.stage)
            if d.decision in ('linked', 'refused_margin'):
                expected = chain_free_runner_up(decisions, d.tracklet_a, d.tracklet_b)
                np.testing.assert_allclose(d.runner_up_cost, expected, rtol=1e-12)
            if d.decision == 'linked':
                self.assertTrue(math.isnan(expected)
                                or expected - d.cost >= scale['minimum_cost_margin'])
                self.assertLessEqual(d.cost, scale['max_assignment_cost'])
                self.assertLess(d.cost, scale['unmatched_cost'] * 2)
            elif d.decision == 'refused_margin':
                self.assertLess(expected - d.cost, scale['minimum_cost_margin'])
            elif d.decision == 'refused_gate':
                self.assertIn(d.reason, GATE_REASONS)
                self.assertEqual(math.isnan(d.cost), d.reason != 'max_cost')
                if d.reason == 'max_cost':
                    self.assertGreater(d.cost, scale['max_assignment_cost'])
            else:
                self.assertTrue(math.isfinite(d.cost))
                self.assertLessEqual(d.cost, scale['max_assignment_cost'])

    @settings(deadline=None, max_examples=100)
    @given(scenes(), st.data(), st.sampled_from(sorted(MODELS)))
    def test_invariant_under_relabeling_and_permutation(self, scene, data, model):
        ends, starts = scene
        cfg = replace(MODELS[model], max_gap=30)
        links, decisions = link_tracklets(ends, starts, cfg)
        ids = sorted({e.tracklet_id for e in ends})
        new = data.draw(st.permutations([f"r{k:03d}" for k in range(len(ids))]))
        mapping = dict(zip(ids, new))
        ends2 = data.draw(st.permutations(
            [replace(e, tracklet_id=mapping[e.tracklet_id]) for e in ends]))
        starts2 = data.draw(st.permutations(
            [replace(s, tracklet_id=mapping[s.tracklet_id]) for s in starts]))
        links2, decisions2 = link_tracklets(list(ends2), list(starts2), cfg)
        self.assertEqual({(mapping[a], mapping[b]) for a, b in links}, set(links2))
        labels = {(mapping[d.tracklet_a], mapping[d.tracklet_b]): d.decision for d in decisions}
        self.assertEqual(labels, {(d.tracklet_a, d.tracklet_b): d.decision for d in decisions2})

    @settings(deadline=None, max_examples=100)
    @given(scenes(), st.sampled_from(sorted(MODELS)))
    def test_time_reversal_mirrors_links(self, scene, model):
        ends, starts = scene
        cfg = replace(MODELS[model], max_gap=30)
        links, decisions = link_tracklets(ends, starts, cfg)
        rev_links, rev_decisions = link_tracklets(*reversed_scene(ends, starts), cfg)
        self.assertEqual({(b, a) for a, b in links}, set(rev_links))
        forward = by_pair(decisions)
        backward = {(b, a): d for (a, b), d in by_pair(rev_decisions).items()}
        self.assertEqual(set(forward), set(backward))
        for key, d in forward.items():
            self.assertEqual(d.decision, backward[key].decision)
            np.testing.assert_allclose(d.cost, backward[key].cost, rtol=1e-9, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
