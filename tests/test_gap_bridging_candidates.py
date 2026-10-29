#!/usr/bin/env python3

"""
Tests for gap_bridging.candidates.link_tracklets (WP-B candidate linking).

Example tests pin the gates, the cost formula, the margin veto and the
per-video contract; Hypothesis property tests check one-to-one links, the
one-audit-row-per-candidate invariant, the margin rule, invariance under
relabeling and permutation, and time-reversal symmetry.

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

# ----- numerics -----
import numpy as np

# ----- testing -----
from hypothesis import given, settings, strategies as st

# ----- local modules -----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from gap_bridging.candidates import link_tracklets  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.types import DECISIONS, Endpoint  # noqa: E402

# ============================================================
# HELPERS
# ============================================================

CFG = BridgeConfig()
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


def by_pair(decisions) -> dict:
    """Index decisions by (tracklet_a, tracklet_b)."""
    return {(d.tracklet_a, d.tracklet_b): d for d in decisions}


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


# ============================================================
# PROPERTY TESTS
# ============================================================

class CandidatePropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=60)
    @given(scenes(gap_free=True), st.integers(1, 5))
    def test_gap_free_input_gives_nothing(self, scene, max_gap):
        ends, starts = scene
        self.assertEqual(link_tracklets(ends, starts, replace(CFG, max_gap=max_gap)), ([], []))

    @settings(deadline=None, max_examples=150)
    @given(scenes(), st.integers(1, 5))
    def test_audit_and_link_invariants(self, scene, max_gap):
        ends, starts = scene
        cfg = replace(CFG, max_gap=max_gap)
        links, decisions = link_tracklets(ends, starts, cfg)
        pairs = [(d.tracklet_a, d.tracklet_b) for d in decisions]
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(set(pairs), enumerated(ends, starts, max_gap))
        self.assertEqual(len({a for a, _ in links}), len(links))
        self.assertEqual(len({b for _, b in links}), len(links))
        self.assertEqual(links, [p for p, d in zip(pairs, decisions) if d.decision == 'linked'])
        for d in decisions:
            self.assertIn(d.decision, DECISIONS)
            self.assertTrue(1 <= d.gap_frames <= max_gap)
            if d.decision == 'linked':
                self.assertTrue(math.isnan(d.runner_up_cost)
                                or d.runner_up_cost - d.cost >= cfg.minimum_cost_margin)
                self.assertLessEqual(d.cost, cfg.max_assignment_cost)
            elif d.decision == 'refused_margin':
                self.assertLess(d.runner_up_cost - d.cost, cfg.minimum_cost_margin)
            elif d.decision == 'refused_gate':
                self.assertIn(d.reason, GATE_REASONS)
                self.assertEqual(math.isnan(d.cost), d.reason != 'max_cost')
            else:
                self.assertTrue(math.isfinite(d.cost))

    @settings(deadline=None, max_examples=100)
    @given(scenes(), st.data())
    def test_invariant_under_relabeling_and_permutation(self, scene, data):
        ends, starts = scene
        links, decisions = link_tracklets(ends, starts, CFG)
        ids = sorted({e.tracklet_id for e in ends})
        new = data.draw(st.permutations([f"r{k:03d}" for k in range(len(ids))]))
        mapping = dict(zip(ids, new))
        ends2 = data.draw(st.permutations(
            [replace(e, tracklet_id=mapping[e.tracklet_id]) for e in ends]))
        starts2 = data.draw(st.permutations(
            [replace(s, tracklet_id=mapping[s.tracklet_id]) for s in starts]))
        links2, decisions2 = link_tracklets(list(ends2), list(starts2), CFG)
        self.assertEqual({(mapping[a], mapping[b]) for a, b in links}, set(links2))
        labels = {(mapping[d.tracklet_a], mapping[d.tracklet_b]): d.decision for d in decisions}
        self.assertEqual(labels, {(d.tracklet_a, d.tracklet_b): d.decision for d in decisions2})

    @settings(deadline=None, max_examples=100)
    @given(scenes())
    def test_time_reversal_mirrors_links(self, scene):
        ends, starts = scene
        links, decisions = link_tracklets(ends, starts, CFG)
        rev_ends = [replace(s, frame=-s.frame, vel=-np.asarray(s.vel)) for s in starts]
        rev_starts = [replace(e, frame=-e.frame, vel=-np.asarray(e.vel)) for e in ends]
        rev_links, rev_decisions = link_tracklets(rev_ends, rev_starts, CFG)
        self.assertEqual({(b, a) for a, b in links}, set(rev_links))
        forward = by_pair(decisions)
        backward = {(b, a): d for (a, b), d in by_pair(rev_decisions).items()}
        self.assertEqual(set(forward), set(backward))
        for key, d in forward.items():
            self.assertEqual(d.decision, backward[key].decision)
            np.testing.assert_allclose(d.cost, backward[key].cost, rtol=1e-9, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
