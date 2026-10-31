#!/usr/bin/env python3

"""
Tests for the v003 appearance cues of gap bridging (WP-C).

``geometry.tracklet_cues`` measures, per tracklet, the median Shell-Shoulder
length (carapace size) and the mean sign of the claw relative to the body
axis (claw side); ``geometry.endpoints`` copies both onto START and END.
``candidates._cue_terms`` turns a pair's cues into a size term, a side term
and a claw-side veto; under the sqrt model the terms enter the cost and the
veto refuses the pair (``refused_gate 'claw_side'``), while the linear model
ignores the cues. Example tests pin the sign convention, the medians of short
tracklets and the veto; Hypothesis tests check symmetry, the zero terms of
unknown cues, the veto conditions, that a vetoed pair is never linked nor
chain-reachable, and that ``appearance_cues=False`` equals a run without cue
values. ``AxisFlipTests`` pin the axis-flip rule (a reversed Shell->Shoulder
axis with the claw in place is a Shoulder flip: no veto, side term 0, reason
suffix ``axis_flip``). A demo test compares cues on and off at ``max_gap = 15``.

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
from typing import List, Tuple

# ----- numerics -----
import numpy as np
import pyarrow.parquet as pq

# ----- testing -----
from hypothesis import assume, given, settings, strategies as st

# ----- local modules -----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from gap_bridging import candidates  # noqa: E402
from gap_bridging.candidates import link_tracklets  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.geometry import (MAX_SHOULDER_LENGTH_PX, MIN_SHOULDER_LENGTH_PX,  # noqa: E402
                                   endpoints, tracklet_cues)
from gap_bridging.types import Endpoint  # noqa: E402
from gap_bridging_fixtures import (DEFAULT_OFFSETS, Crab, make_observations,  # noqa: E402
                                   node_positions, straight_crab)
from test_gap_bridging_candidates import (CFG as LINEAR, SQRT, by_pair,  # noqa: E402
                                          chain_free_runner_up, ep, radius, reversed_scene,
                                          same_decision, scenes)

# ============================================================
# HELPERS
# ============================================================

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
CUES = replace(SQRT, appearance_cues=True, size_scale_px=3.0, size_noise_px=1.0, size_cost_cap=3.0,
               size_weight=0.1, claw_side_weight=0.15, claw_side_confidence=0.9,
               claw_side_min_frames=5)        # pinned so example numbers survive config changes
NO_CUES = replace(CUES, appearance_cues=False)
MIRRORED = DEFAULT_OFFSETS * np.array([1.0, -1.0])      # claw at body-frame (40, +14)
MOTION_GATES = ('displacement', 'prediction_error')


def with_cues(p: Endpoint, size: float = math.nan, size_n: int = 0, side: float = 0.0,
              side_n: int = 0) -> Endpoint:
    """Copy of ``p`` carrying the given appearance cues."""
    return replace(p, size_px=float(size), size_n=int(size_n), claw_side=float(side),
                   claw_side_n=int(side_n))


def cleared(points: List[Endpoint]) -> List[Endpoint]:
    """Endpoints with the cue fields reset to their unknown defaults."""
    return [with_cues(p) for p in points]


def pose_rows(lengths: List[float], claws: List[Tuple[float, float]],
              claw_valid: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
    """
    One tracklet's (n,5,2) nodes and (n,5) validity, heading 0, shell at the origin.

    Row i has its shoulder at ``(lengths[i], 0)`` and its claw at ``claws[i]``
    (invalid when ``claw_valid[i]`` is False); elbow and arm are invalid.
    """
    n = len(lengths)
    xy = np.zeros((n, 5, 2))
    valid = np.zeros((n, 5), dtype=bool)
    valid[:, 0] = valid[:, 1] = True
    xy[:, 1, 0] = lengths
    xy[:, 3] = np.asarray(claws, dtype=float).reshape(n, 2)
    valid[:, 3] = claw_valid
    return xy, valid


def expected_terms(a: Endpoint, b: Endpoint, cfg: BridgeConfig) -> Tuple[float, float, bool]:
    """Independent scalar evaluation of the size term, side term and veto of (a, b)."""
    size = 0.0
    if a.size_n > 0 and b.size_n > 0:
        spread = math.sqrt(cfg.size_scale_px ** 2 + cfg.size_noise_px ** 2 / a.size_n
                           + cfg.size_noise_px ** 2 / b.size_n)
        size = min(cfg.size_cost_cap, abs(a.size_px - b.size_px) / spread)
    k = cfg.claw_side_min_frames
    weight = min(1.0, a.claw_side_n / k) * min(1.0, b.claw_side_n / k)
    side = (max(0.0, -a.claw_side * b.claw_side) * weight
            if a.claw_side_n > 0 and b.claw_side_n > 0 else 0.0)
    veto = (abs(a.claw_side) >= cfg.claw_side_confidence and abs(b.claw_side) >= cfg.claw_side_confidence
            and a.claw_side_n >= cfg.claw_side_min_frames and b.claw_side_n >= cfg.claw_side_min_frames
            and a.claw_side * b.claw_side < 0)
    return size, side, veto


sides = st.sampled_from([-1.0, -0.95, -0.9, -0.89, -0.5, 0.0, 0.5, 0.89, 0.9, 0.95, 1.0])


@st.composite
def cue_values(draw) -> dict:
    """One tracklet's cues: size known with n >= 1 or unknown; side known with n >= 1 or 0."""
    size_n = draw(st.integers(0, 10))
    size = draw(st.floats(10.0, 40.0)) if size_n else math.nan
    side_n = draw(st.integers(0, 10))
    side = draw(sides) if side_n else 0.0
    return dict(size=size, size_n=size_n, side=side, side_n=side_n)


@st.composite
def cue_scenes(draw, max_tracklets: int = 10) -> Tuple[List[Endpoint], List[Endpoint]]:
    """``scenes`` with whole-tracklet cues (identical on START and END) per tracklet."""
    ends, starts = draw(scenes(max_tracklets=max_tracklets))
    cues = {e.tracklet_id: draw(cue_values()) for e in ends}
    return ([with_cues(e, **cues[e.tracklet_id]) for e in ends],
            [with_cues(s, **cues[s.tracklet_id]) for s in starts])


def posed(p: Endpoint, heading: float, claw: Tuple[float, float]) -> Endpoint:
    """Copy of ``p`` with ``heading`` and the body-frame claw offset ``claw``."""
    offsets = np.array(p.offsets, dtype=float)
    offsets[candidates.CLAW] = claw
    return replace(p, heading=float(heading), offsets=offsets)


def body_claw(heading: float, image_claw: Tuple[float, float]) -> Tuple[float, float]:
    """Body-frame offset whose image-frame direction at ``heading`` is ``image_claw``."""
    c, s = math.cos(heading), math.sin(heading)
    x, y = image_claw
    return (c * x + s * y, -s * x + c * y)


def pair_full(a: Endpoint, b: Endpoint, cfg: BridgeConfig) -> Tuple[float, float, bool, bool]:
    """``candidates._cue_terms_full`` for the single pair (a, b)."""
    size, side, veto, supp = candidates._cue_terms_full([a], [b], np.array([0]), np.array([0]),
                                                        cfg)
    return float(size[0]), float(side[0]), bool(veto[0]), bool(supp[0])


def pair_terms(a: Endpoint, b: Endpoint, cfg: BridgeConfig) -> Tuple[float, float, bool]:
    """``candidates._cue_terms`` for the single pair (a, b)."""
    size, side, veto = candidates._cue_terms([a], [b], np.array([0]), np.array([0]), cfg)
    return float(size[0]), float(side[0]), bool(veto[0])


# ============================================================
# GEOMETRY: SIGN CONVENTION, SIZES AND MEANS
# ============================================================

class TrackletCueGeometryTests(unittest.TestCase):

    def cues_of(self, crab: Crab) -> Endpoint:
        start, end = next(iter(endpoints(make_observations([crab]), CUES).values()))
        for name in ('size_px', 'size_n', 'claw_side', 'claw_side_n'):
            self.assertEqual(getattr(start, name), getattr(end, name), name)   # whole-tracklet
        return start

    def test_sign_convention_default_offsets(self):
        # Heading 0: u = (22, 0), w = (40, -14), u_x w_y - u_y w_x = 22 * (-14) < 0.
        crab = straight_crab(1, 0, 6, origin=(500.0, 500.0), velocity=(0.0, 0.0), heading=0.0)
        start = self.cues_of(crab)
        self.assertEqual((start.claw_side, start.claw_side_n), (-1.0, 6))
        self.assertEqual((start.size_px, start.size_n), (22.0, 6))

    def test_mirrored_claw_gives_plus_one(self):
        crab = straight_crab(1, 0, 6, origin=(500.0, 500.0), velocity=(0.0, 0.0), heading=0.0)
        crab.offsets = MIRRORED.copy()
        start = self.cues_of(crab)
        self.assertEqual((start.claw_side, start.claw_side_n), (1.0, 6))

    @settings(deadline=None, max_examples=60)
    @given(st.floats(-math.pi, math.pi), st.booleans(),
           st.tuples(st.floats(-1e3, 1e3), st.floats(-1e3, 1e3)))
    def test_side_and_size_rotation_invariant(self, heading, mirrored, origin):
        n = 4
        offsets = MIRRORED if mirrored else DEFAULT_OFFSETS
        xy = np.stack([node_positions(np.asarray(origin, float), heading, offsets)] * n)
        size, size_n, side, side_n = tracklet_cues(xy, np.ones((n, 5), dtype=bool),
                                                   np.array([0]), np.array([n - 1]))
        self.assertEqual(side[0], 1.0 if mirrored else -1.0)
        self.assertEqual((size_n[0], side_n[0]), (n, n))
        self.assertAlmostEqual(size[0], 22.0, places=9)

    def test_one_frame_tracklet(self):
        xy, valid = pose_rows([20.0], [(30.0, 5.0)], [True])
        size, size_n, side, side_n = tracklet_cues(xy, valid, np.array([0]), np.array([0]))
        self.assertEqual((size[0], size_n[0], side[0], side_n[0]), (20.0, 1, 1.0, 1))

    def test_two_frame_tracklet_median_is_midpoint(self):
        xy, valid = pose_rows([20.0, 30.0], [(30.0, 5.0), (30.0, -5.0)], [True, True])
        size, size_n, side, side_n = tracklet_cues(xy, valid, np.array([0]), np.array([1]))
        self.assertEqual((size[0], size_n[0], side[0], side_n[0]), (25.0, 2, 0.0, 2))

    def test_three_frame_tracklet_median_and_mean(self):
        xy, valid = pose_rows([20.0, 40.0, 30.0], [(30.0, 5.0), (30.0, -5.0), (30.0, 5.0)],
                              [True, True, True])
        size, size_n, side, side_n = tracklet_cues(xy, valid, np.array([0]), np.array([2]))
        self.assertEqual((size[0], size_n[0], side_n[0]), (30.0, 3, 3))
        self.assertAlmostEqual(side[0], 1.0 / 3.0)

    def test_excluded_frames(self):
        # Row 0 too short, row 4 too long: no size, no side. Row 2: claw invalid (size only).
        # Row 3: claw on the body axis (sign 0, size only).
        lengths = [MIN_SHOULDER_LENGTH_PX - 0.5, 24.0, 26.0, 28.0, MAX_SHOULDER_LENGTH_PX + 0.5]
        claws = [(30.0, 5.0), (30.0, -5.0), (30.0, 5.0), (30.0, 0.0), (30.0, 5.0)]
        xy, valid = pose_rows(lengths, claws, [True, True, False, True, True])
        size, size_n, side, side_n = tracklet_cues(xy, valid, np.array([0]), np.array([4]))
        self.assertEqual((size[0], size_n[0], side[0], side_n[0]), (26.0, 3, -1.0, 1))

    def test_bounds_inclusive_and_missing_shoulder(self):
        lengths = [MIN_SHOULDER_LENGTH_PX, MAX_SHOULDER_LENGTH_PX, 30.0]
        xy, valid = pose_rows(lengths, [(30.0, 5.0)] * 3, [True] * 3)
        valid[2, 1] = False                         # shoulder invisible on row 2
        size, size_n, side, side_n = tracklet_cues(xy, valid, np.array([0]), np.array([2]))
        self.assertEqual((size[0], size_n[0], side[0], side_n[0]),
                         (0.5 * (MIN_SHOULDER_LENGTH_PX + MAX_SHOULDER_LENGTH_PX), 2, 1.0, 2))

    def test_unknown_cues_and_several_tracklets(self):
        xy1, valid1 = pose_rows([5.0, 5.0], [(30.0, 5.0)] * 2, [True] * 2)   # all too short
        xy2, valid2 = pose_rows([20.0, 22.0, 21.0], [(30.0, -5.0)] * 3, [True] * 3)
        xy3, valid3 = pose_rows([18.0], [(30.0, 5.0)], [False])
        xy = np.concatenate([xy1, xy2, xy3])
        valid = np.concatenate([valid1, valid2, valid3])
        size, size_n, side, side_n = tracklet_cues(xy, valid, np.array([0, 2, 5]), np.array([1, 4, 5]))
        self.assertTrue(math.isnan(size[0]))
        np.testing.assert_array_equal(size[1:], [21.0, 18.0])
        np.testing.assert_array_equal(size_n, [0, 3, 1])
        np.testing.assert_array_equal(side, [0.0, -1.0, 0.0])
        np.testing.assert_array_equal(side_n, [0, 3, 0])

    @settings(deadline=None, max_examples=80)
    @given(st.lists(st.lists(st.tuples(st.floats(0.0, 80.0), st.floats(-20.0, 20.0), st.booleans()),
                             min_size=1, max_size=6), min_size=1, max_size=5))
    def test_matches_per_tracklet_reference(self, tracklets):
        parts = [pose_rows([r[0] for r in t], [(30.0, r[1]) for r in t], [r[2] for r in t])
                 for t in tracklets]
        xy = np.concatenate([p[0] for p in parts])
        valid = np.concatenate([p[1] for p in parts])
        counts = np.array([len(t) for t in tracklets])
        ends = np.cumsum(counts) - 1
        size, size_n, side, side_n = tracklet_cues(xy, valid, ends - counts + 1, ends)
        for i, t in enumerate(tracklets):
            ok = [r for r in t if MIN_SHOULDER_LENGTH_PX <= r[0] <= MAX_SHOULDER_LENGTH_PX]
            signs = [math.copysign(1.0, r[1]) for r in ok if r[2] and r[0] * r[1] != 0]
            self.assertEqual(size_n[i], len(ok))
            self.assertEqual(side_n[i], len(signs))
            if ok:
                self.assertEqual(size[i], float(np.median([r[0] for r in ok])))
            else:
                self.assertTrue(math.isnan(size[i]))
            self.assertAlmostEqual(side[i], float(np.mean(signs)) if signs else 0.0)

    def test_segments_must_tile_rows(self):
        xy, valid = pose_rows([20.0, 21.0], [(30.0, 5.0)] * 2, [True] * 2)
        with self.assertRaises(ValueError):
            tracklet_cues(xy, valid, np.array([0]), np.array([0]))


# ============================================================
# CUE TERMS
# ============================================================

class CueTermPropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=300)
    @given(cue_values(), cue_values())
    def test_terms_match_formula_and_are_symmetric(self, x, y):
        a = with_cues(ep('A', 10, (0., 0.)), **x)
        b = with_cues(ep('B', 14, (1., 0.)), **y)
        size, side, veto = pair_terms(a, b, CUES)
        swapped = pair_terms(with_cues(a, **y), with_cues(b, **x), CUES)
        self.assertEqual((size, side, veto), swapped)
        exp_size, exp_side, exp_veto = expected_terms(a, b, CUES)
        self.assertAlmostEqual(size, exp_size, places=12)
        self.assertAlmostEqual(side, exp_side, places=12)
        self.assertEqual(veto, exp_veto)
        self.assertTrue(0.0 <= size <= CUES.size_cost_cap)
        self.assertTrue(0.0 <= side <= 1.0)
        unweighted = max(0.0, -x['side'] * y['side'])
        self.assertLessEqual(side, unweighted)
        if x['size_n'] == 0 or y['size_n'] == 0:
            self.assertEqual(size, 0.0)
        if x['side_n'] == 0 or y['side_n'] == 0:
            self.assertEqual((side, veto), (0.0, False))
        if min(x['side_n'], y['side_n']) >= CUES.claw_side_min_frames:
            self.assertEqual(side, unweighted)       # full weight once both sides are confident

    def test_example_values(self):
        a = with_cues(ep('A', 10, (0., 0.)), size=20.0, size_n=4, side=1.0, side_n=10)
        b = with_cues(ep('B', 14, (1., 0.)), size=24.0, size_n=1, side=-0.95, side_n=5)
        size, side, veto = pair_terms(a, b, CUES)
        self.assertAlmostEqual(size, 4.0 / math.sqrt(9.0 + 0.25 + 1.0))
        self.assertEqual((side, veto), (0.95, True))
        self.assertEqual(pair_terms(a, replace(b, claw_side_n=4), CUES)[2], False)     # too few frames
        self.assertEqual(pair_terms(a, replace(b, claw_side=-0.89), CUES)[2], False)   # not confident
        self.assertEqual(pair_terms(a, replace(b, claw_side=0.95), CUES), (size, 0.0, False))
        one = pair_terms(a, replace(b, claw_side=-1.0, claw_side_n=1), CUES)    # 1 frame: weight 1/5
        self.assertEqual((one[1], one[2]), (0.2, False))
        two = pair_terms(replace(a, claw_side_n=2), replace(b, claw_side=-1.0, claw_side_n=3), CUES)
        self.assertAlmostEqual(two[1], 0.4 * 0.6)
        far = with_cues(b, size=60.0, size_n=50, side=-0.95, side_n=5)
        self.assertEqual(pair_terms(a, far, CUES)[0], CUES.size_cost_cap)


# ============================================================
# LINKING
# ============================================================

class CueLinkingTests(unittest.TestCase):

    def test_confident_mismatch_is_refused_and_never_linked(self):
        a = with_cues(ep('A', 10, (0., 0.), (1., 0.)), 22.0, 10, 1.0, 10)
        b = with_cues(ep('B', 12, (2., 0.), (1., 0.)), 22.0, 10, -1.0, 10)
        links, decisions = link_tracklets([a], [b], CUES)
        d = decisions[0]
        self.assertEqual(links, [])
        self.assertEqual((d.decision, d.reason), ('refused_gate', 'claw_side'))
        self.assertTrue(math.isnan(d.cost) and math.isnan(d.runner_up_cost))
        self.assertEqual((d.size_term, d.side_term), (0.0, 1.0))
        # Without cues, under linear, or below the confidence the pair links.
        for cfg, pair in ((NO_CUES, (a, b)), (LINEAR, (a, b)),
                          (CUES, (a, replace(b, claw_side=-0.8)))):
            self.assertEqual(link_tracklets([pair[0]], [pair[1]], cfg)[0], [('A', 'B')])

    def test_motion_gates_take_priority_over_claw_side(self):
        a = with_cues(ep('A', 10, (0., 0.), vel_known=False), 22.0, 10, 1.0, 10)
        b = with_cues(ep('B', 12, (radius(CUES, 2) + 1.0, 0.), vel_known=False), 22.0, 10, -1.0, 10)
        d = link_tracklets([a], [b], CUES)[1][0]
        self.assertEqual((d.decision, d.reason), ('refused_gate', 'displacement'))

    def test_claw_side_takes_priority_over_max_cost(self):
        r = radius(CUES, 2)
        a = with_cues(ep('A', 10, (0., 0.), vel_known=False), 22.0, 10, 1.0, 10)
        # Motion cost 0.99 plus a size term of 8 / sqrt(9.2) * 0.1 exceeds max cost 1.
        b = with_cues(ep('B', 12, (0.99 * r, 0.), vel_known=False), 30.0, 10, -1.0, 10)
        d_on = link_tracklets([a], [b], CUES)[1][0]
        d_off = link_tracklets([a], [replace(b, claw_side=1.0)], CUES)[1][0]
        self.assertEqual((d_off.decision, d_off.reason), ('refused_gate', 'max_cost'))
        self.assertEqual((d_on.decision, d_on.reason), ('refused_gate', 'claw_side'))

    def test_cost_gains_weighted_terms(self):
        a = with_cues(ep('A', 10, (0., 0.), (1., 0.)), 20.0, 4, 0.5, 3)
        b = with_cues(ep('B', 12, (2.5, 0.), (1., 0.)), 23.0, 2, -0.5, 3)
        d_on = link_tracklets([a], [b], CUES)[1][0]
        d_off = link_tracklets([a], [b], NO_CUES)[1][0]
        size, side, _ = expected_terms(a, b, CUES)
        self.assertEqual(d_on.decision, 'linked')
        self.assertAlmostEqual(d_on.size_term, size)
        self.assertAlmostEqual(d_on.side_term, side)
        self.assertAlmostEqual(side, 0.25 * 0.6 * 0.6)        # 3 claw frames each: weight (3/5)^2
        self.assertAlmostEqual(d_on.cost, d_off.cost + 0.1 * size + 0.15 * side, places=12)
        self.assertTrue(math.isnan(d_off.size_term) and math.isnan(d_off.side_term))

    def test_size_term_breaks_a_tie(self):
        # Two starts equidistant from A: without cues the margin veto refuses; the size cue decides.
        a = with_cues(ep('A', 10, (0., 0.), vel_known=False), 20.0, 30, 0.0, 0)
        b = with_cues(ep('B', 12, (0., 2.), vel_known=False), 20.0, 30, 0.0, 0)
        c = with_cues(ep('C', 12, (0., -2.), vel_known=False), 32.0, 30, 0.0, 0)
        off = by_pair(link_tracklets([a], [b, c], NO_CUES)[1])
        self.assertEqual(off[('A', 'B')].decision, 'refused_margin')
        links, decisions = link_tracklets([a], [b, c], CUES)
        self.assertEqual(links, [('A', 'B')])
        self.assertAlmostEqual(by_pair(decisions)[('A', 'C')].size_term, 3.0)   # capped

    def test_vetoed_successor_is_not_chain_reachable(self):
        # A -> B and A -> C are near-ties; (B, C) passes the motion gates. Without the veto
        # (A, C) is a chain competitor and is ignored; with (B, C) vetoed it counts.
        r3 = radius(CUES, 3)
        c_x = 5.5
        a = ep('A', 10, (0., 0.), vel_known=False)
        b_start = with_cues(ep('B', 12, (3., 0.), vel_known=False), side=1.0, side_n=10)
        b_end = with_cues(ep('B', 13, (c_x - 0.5 * r3, 0.), vel_known=False), side=1.0, side_n=10)
        c = with_cues(ep('C', 16, (c_x, 0.), vel_known=False), side=-1.0, side_n=10)
        same = replace(c, claw_side=1.0)
        links, decisions = link_tracklets([a, b_end], [b_start, same], CUES)
        d = by_pair(decisions)
        self.assertEqual(links, [('A', 'B'), ('B', 'C')])
        self.assertEqual(d[('A', 'B')].n_chain_ignored, 1)
        links, decisions = link_tracklets([a, b_end], [b_start, c], CUES)
        d = by_pair(decisions)
        self.assertEqual((d[('B', 'C')].decision, d[('B', 'C')].reason), ('refused_gate', 'claw_side'))
        self.assertEqual(links, [])
        self.assertEqual(d[('A', 'B')].decision, 'refused_margin')
        self.assertEqual(d[('A', 'B')].runner_up_cost, d[('A', 'C')].cost)
        self.assertEqual(d[('A', 'B')].n_chain_ignored, 0)


class AxisFlipTests(unittest.TestCase):
    """A reversed body axis with the claw in place is a Shoulder flip, not a claw side."""

    CLAW_IMAGE = (40.0, -14.0)

    def pair(self, heading_b: float, claw_b_image: Tuple[float, float], side_b: float
             ) -> Tuple[Endpoint, Endpoint]:
        a = posed(with_cues(ep('A', 10, (0., 0.), (1., 0.)), 22.0, 10, -1.0, 10), 0.0,
                  body_claw(0.0, self.CLAW_IMAGE))
        b = posed(with_cues(ep('B', 12, (2., 0.), (1., 0.)), 22.0, 10, side_b, 10), heading_b,
                  body_claw(heading_b, claw_b_image))
        return a, b

    def test_axis_reversed_claw_aligned_is_not_vetoed(self):
        # u_a = (1, 0), u_b = (-1, 0); claws both at image (40, -14): the side sign
        # flips (-1 -> +1) only because the axis did.
        a, b = self.pair(math.pi, self.CLAW_IMAGE, side_b=1.0)
        self.assertTrue(bool(candidates._axis_flip([a], [b], np.array([0]), np.array([0]))[0]))
        size, side, veto, supp = pair_full(a, b, CUES)
        self.assertEqual((side, veto, supp), (0.0, False, True))
        links, decisions = link_tracklets([a], [b], CUES)
        d = decisions[0]
        self.assertEqual(links, [('A', 'B')])
        self.assertEqual(d.decision, 'linked')
        self.assertTrue(d.reason.endswith('; axis_flip'), d.reason)
        self.assertEqual(d.side_term, 0.0)
        # Same cost as with agreeing sides: the side term is dropped entirely.
        same = link_tracklets([a], [replace(b, claw_side=-1.0)], CUES)[1][0]
        self.assertEqual(d.cost, same.cost)
        self.assertFalse(same.reason.endswith('axis_flip'))      # no veto to lift

    def test_genuine_opposite_claw_is_vetoed(self):
        # Axes aligned, claws mirrored across the axis: a different claw side.
        a, b = self.pair(0.0, (40.0, 14.0), side_b=1.0)
        self.assertFalse(bool(candidates._axis_flip([a], [b], np.array([0]), np.array([0]))[0]))
        self.assertEqual(pair_full(a, b, CUES)[1:], (1.0, True, False))
        links, decisions = link_tracklets([a], [b], CUES)
        self.assertEqual(links, [])
        self.assertEqual((decisions[0].decision, decisions[0].reason), ('refused_gate', 'claw_side'))

    def test_true_turn_keeps_the_ordinary_rule(self):
        # Axis and claw both reversed (a 180-degree turn): no flip; the whole-tracklet
        # sides decide as before.
        a, b = self.pair(math.pi, (-40.0, 14.0), side_b=-1.0)
        self.assertFalse(bool(candidates._axis_flip([a], [b], np.array([0]), np.array([0]))[0]))
        self.assertEqual(pair_full(a, b, CUES)[1:], (0.0, False, False))
        opposite = replace(b, claw_side=1.0)
        self.assertEqual(pair_full(a, opposite, CUES)[1:], (1.0, True, False))
        self.assertEqual(link_tracklets([a], [opposite], CUES)[1][0].reason, 'claw_side')

    def test_unknown_vectors_never_flip(self):
        a, b = self.pair(math.pi, self.CLAW_IMAGE, side_b=1.0)
        for x, y in ((replace(a, heading=math.nan), b),
                     (a, posed(b, math.pi, (math.nan, math.nan))),
                     (posed(a, 0.0, (0.0, 0.0)), b)):
            self.assertEqual(pair_full(x, y, CUES)[2:], (True, False))

    @settings(deadline=None, max_examples=300)
    @given(st.floats(-math.pi, math.pi), st.floats(-math.pi, math.pi),
           st.floats(-math.pi, math.pi), st.floats(-math.pi, math.pi),
           cue_values(), cue_values())
    def test_suppression_iff_axis_reversed_and_claw_kept(self, ha, hb, ca, cb, x, y):
        claw_a = (30.0 * math.cos(ca), 30.0 * math.sin(ca))     # body-frame claw offsets
        claw_b = (30.0 * math.cos(cb), 30.0 * math.sin(cb))
        a = posed(with_cues(ep('A', 10, (0., 0.)), **x), ha, claw_a)
        b = posed(with_cues(ep('B', 14, (1., 0.)), **y), hb, claw_b)
        dot_u = math.cos(ha - hb)
        dot_c = math.cos((ha + ca) - (hb + cb))
        assume(abs(dot_u) > 1e-9 and abs(dot_c) > 1e-9)       # off the decision boundary
        flip = dot_u < 0 and dot_c > 0
        self.assertEqual(bool(candidates._axis_flip([a], [b], np.array([0]), np.array([0]))[0]),
                         flip)
        size, side, veto, supp = pair_full(a, b, CUES)
        exp_size, exp_side, exp_veto = expected_terms(a, b, CUES)
        self.assertAlmostEqual(size, exp_size, places=12)
        self.assertEqual(supp, exp_veto and flip)
        self.assertEqual(veto, exp_veto and not flip)
        self.assertAlmostEqual(side, 0.0 if flip else exp_side, places=12)
        self.assertEqual(pair_full(b, a, CUES), (size, side, veto, supp))     # symmetric


class CueLinkingPropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=200)
    @given(cue_scenes(), st.integers(1, 30),
           st.sampled_from([(5, 15, 30, 45), (1, 2, 3), (30,)]))
    def test_veto_and_terms_in_decisions(self, scene, max_gap, link_stages):
        ends, starts = scene
        cfg = replace(CUES, max_gap=max_gap, link_stages=link_stages)
        links, decisions = link_tracklets(ends, starts, cfg)
        off = by_pair(link_tracklets(ends, starts, replace(cfg, appearance_cues=False))[1])
        end_of = {e.tracklet_id: e for e in ends}
        start_of = {s.tracklet_id: s for s in starts}
        self.assertEqual(set(by_pair(decisions)), set(off))
        linked = set(links)
        for d in decisions:
            a, b = end_of[d.tracklet_a], start_of[d.tracklet_b]
            size, side, veto = expected_terms(a, b, cfg)
            self.assertAlmostEqual(d.size_term, size, places=12)
            self.assertAlmostEqual(d.side_term, side, places=12)
            motion_ok = off[(d.tracklet_a, d.tracklet_b)].reason not in MOTION_GATES
            self.assertEqual(d.reason == 'claw_side', veto and motion_ok, d)
            if veto:
                self.assertNotIn((d.tracklet_a, d.tracklet_b), linked)
                self.assertEqual(d.decision, 'refused_gate')
                self.assertTrue(math.isnan(d.cost))
            if d.decision in ('linked', 'refused_margin'):
                # Runner-ups ignore vetoed pairs as chain links (independent re-derivation).
                np.testing.assert_allclose(d.runner_up_cost,
                                           chain_free_runner_up(decisions, d.tracklet_a, d.tracklet_b),
                                           rtol=1e-12)

    @settings(deadline=None, max_examples=150)
    @given(cue_scenes(), st.integers(1, 30))
    def test_cues_off_equals_unknown_cues(self, scene, max_gap):
        ends, starts = scene
        cfg_off = replace(CUES, max_gap=max_gap, appearance_cues=False)
        links, decisions = link_tracklets(ends, starts, cfg_off)
        ref_links, ref = link_tracklets(cleared(ends), cleared(starts), cfg_off)
        self.assertEqual(links, ref_links)
        self.assertEqual(len(decisions), len(ref))
        for x, y in zip(decisions, ref):
            self.assertTrue(same_decision(x, y), (x, y))
            self.assertTrue(math.isnan(x.size_term) and math.isnan(x.side_term))
        # Cues on with every cue unknown: same decisions, terms 0 instead of NaN.
        on_links, on = link_tracklets(cleared(ends), cleared(starts), replace(cfg_off, appearance_cues=True))
        self.assertEqual(on_links, links)
        for x, y in zip(decisions, on):
            self.assertTrue(same_decision(replace(x, size_term=0.0, side_term=0.0), y), (x, y))

    @settings(deadline=None, max_examples=100)
    @given(cue_scenes(), st.integers(1, 30))
    def test_linear_ignores_cues(self, scene, max_gap):
        ends, starts = scene
        cfg = replace(LINEAR, max_gap=max_gap, appearance_cues=True)
        links, decisions = link_tracklets(ends, starts, cfg)
        ref_links, ref = link_tracklets(cleared(ends), cleared(starts), cfg)
        self.assertEqual(links, ref_links)
        for x, y in zip(decisions, ref):
            self.assertTrue(same_decision(x, y), (x, y))
            self.assertTrue(math.isnan(x.size_term) and math.isnan(x.side_term))

    @settings(deadline=None, max_examples=100)
    @given(cue_scenes())
    def test_time_reversal_with_cues(self, scene):
        ends, starts = scene
        cfg = replace(CUES, max_gap=30)
        links, decisions = link_tracklets(ends, starts, cfg)
        rev_links, rev_decisions = link_tracklets(*reversed_scene(ends, starts), cfg)
        self.assertEqual({(b, a) for a, b in links}, set(rev_links))
        backward = {(b, a): d for (a, b), d in by_pair(rev_decisions).items()}
        for key, d in by_pair(decisions).items():
            self.assertEqual((d.decision, d.side_term), (backward[key].decision, backward[key].side_term))
            np.testing.assert_allclose(d.size_term, backward[key].size_term, rtol=1e-12)


# ============================================================
# DEMO
# ============================================================

@unittest.skipUnless(DEMO.exists(), 'demo observations not present')
class DemoCueTests(unittest.TestCase):
    """Cues on vs off on the demo clip at max_gap 15 (one video)."""

    @classmethod
    def setUpClass(cls):
        cfg = replace(BridgeConfig(), max_gap=15)
        pairs = list(endpoints(pq.read_table(DEMO), cfg).values())
        cls.starts = [s for s, _ in pairs]
        cls.ends = [e for _, e in pairs]
        cls.cfg = cfg

    def run_model(self, model: str, cues: bool):
        return link_tracklets(self.ends, self.starts,
                              replace(self.cfg, gate_model=model, appearance_cues=cues))

    def test_cues_change_few_sqrt_decisions(self):
        links_on, on = self.run_model('sqrt', True)
        links_off, off = self.run_model('sqrt', False)
        self.assertEqual([(d.tracklet_a, d.tracklet_b) for d in on],
                         [(d.tracklet_a, d.tracklet_b) for d in off])
        changed = [(x, y) for x, y in zip(on, off) if x.decision != y.decision]
        vetoed = [d for d in on if d.reason == 'claw_side']
        lifted = [d for d in on if d.reason.endswith('; axis_flip')]
        # Every confident opposite-side pair on the demo is a Shoulder flip (axis
        # reversed, claw in place): the axis-flip rule lifts all of them.
        self.assertEqual(len(vetoed), 0, vetoed)
        self.assertTrue(1 <= len(lifted) <= 10, len(lifted))
        self.assertTrue(all(d.side_term == 0.0 for d in lifted))
        self.assertLessEqual(len(changed), 15)
        self.assertLessEqual(len(set(links_on) ^ set(links_off)), 6)
        self.assertTrue(all(math.isfinite(d.size_term) and math.isfinite(d.side_term) for d in on))

    def test_linear_decisions_unchanged(self):
        links_on, on = self.run_model('linear', True)
        links_off, off = self.run_model('linear', False)
        self.assertEqual(links_on, links_off)
        self.assertEqual(len(on), len(off))
        self.assertTrue(all(same_decision(x, y) for x, y in zip(on, off)))


if __name__ == '__main__':
    unittest.main()
