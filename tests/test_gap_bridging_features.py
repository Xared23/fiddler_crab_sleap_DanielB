#!/usr/bin/env python3

"""
Tests for gap_bridging.features (per-pair feature export for the learned scorer).

Example tests pin each column against the private candidates helpers it is
built from (``_pair_geometry``, ``_sqrt_extended_radius``, ``_cue_terms_full``,
``_pair_costs_sqrt``), the NaN conventions of the cue columns, the heading wrap
and hand-computed rival counts on a 3-END/3-START scene; the table's row order
and hand cost are checked against ``link_tracklets``. Hypothesis property tests
on random scenes check time-reversal symmetry, horizon invariance of every
feature (which dedupe across horizons relies on), the gate/ratio equivalences
and the absence of inf. A guarded demo test compares the table with the
``links.parquet`` of ``pipeline.run``.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Optional, Tuple
import unittest

# ----- numerics -----
from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

# ----- local modules -----
from gap_bridging import candidates, features, geometry  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.features import (FEATURE_NAMES, PAIR_FEATURES_SCHEMA,  # noqa: E402
                                   feature_matrix, pair_feature_table, pair_features)
from gap_bridging.types import Endpoint  # noqa: E402
from gap_bridging_fixtures import (DEFAULT_OFFSETS, make_observations,  # noqa: E402
                                   straight_crab)

# ============================================================
# CONSTANTS
# ============================================================

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
# Gate-model numbers pinned (as in test_gap_bridging_candidates) so the examples survive refits.
SQRT = BridgeConfig(gate_model='sqrt', max_gap=5, radius_r0_px=0.0, radius_k_px=3.73, radius_safety=1.25,
                    radius_min_px=6.0, prediction_error_ratio=1.5, prediction_horizon_frames=3,
                    gap_penalty=0.06, max_assignment_cost_sqrt=1.0, unmatched_cost_sqrt=0.6,
                    minimum_cost_margin_sqrt=0.25)
COL = {name: j for j, name in enumerate(FEATURE_NAMES)}
MIRRORED = DEFAULT_OFFSETS * np.array([1.0, -1.0])     # claw on the other side
# Features of END a / START b that trade places under time reversal.
SWAPS = (('speed_a', 'speed_b'), ('vel_known_a', 'vel_known_b'), ('abs_side_a', 'abs_side_b'),
         ('size_n_a', 'size_n_b'), ('claw_side_n_a', 'claw_side_n_b'),
         ('n_frames_a', 'n_frames_b'), ('n_rival_starts', 'n_rival_ends'))

# ============================================================
# HELPERS
# ============================================================


def ep(tid: str, frame: int, pos: Tuple[float, float], vel: Tuple[float, float] = (0., 0.),
       vel_known: bool = True, heading: float = math.nan,
       shoulder: Optional[Tuple[float, float]] = None, video: str = 'v',
       size: Tuple[float, int] = (math.nan, 0), side: Tuple[float, int] = (0.0, 0),
       n_frames: Optional[int] = None) -> Endpoint:
    """Build an Endpoint; ``shoulder`` is the body-frame shoulder offset, ``size``
    and ``side`` the (value, n) cue pairs."""
    offsets = np.full((5, 2), np.nan)
    offsets[0] = 0.0
    if shoulder is not None:
        offsets[1] = shoulder
    n = n_frames if n_frames is not None else (5 if vel_known else 1)
    return Endpoint(tid, video, int(frame), int(n), np.asarray(pos, dtype=float),
                    np.asarray(vel, dtype=float), bool(vel_known), float(heading), offsets,
                    size_px=float(size[0]), size_n=int(size[1]),
                    claw_side=float(side[0]), claw_side_n=int(side[1]))


def enumerate_sorted(ends: List[Endpoint], starts: List[Endpoint],
                     cfg: BridgeConfig) -> Tuple[List[Endpoint], List[Endpoint], np.ndarray, np.ndarray]:
    """Sort and enumerate exactly as ``link_tracklets``."""
    ends = sorted(ends, key=candidates._sort_key)
    starts = sorted(starts, key=candidates._sort_key)
    rows, cols, _ = candidates._enumerate_pairs(ends, starts, cfg.max_gap,
                                                candidates._prune_radius(ends, starts, cfg))
    return ends, starts, rows, cols


def keyed(table: pa.Table) -> Dict[Tuple[str, str], np.ndarray]:
    """Feature vectors keyed by (tracklet_a, tracklet_b)."""
    X = feature_matrix(table)
    return {(a, b): X[k] for k, (a, b) in enumerate(zip(table['tracklet_a'].to_pylist(),
                                                         table['tracklet_b'].to_pylist()))}


def assert_same(test: unittest.TestCase, x: np.ndarray, y: np.ndarray, msg: str = '',
                rtol: float = 0.0, atol: float = 0.0) -> None:
    """Equality with NaN == NaN (exact unless a tolerance is given)."""
    test.assertEqual(np.isnan(x).tolist(), np.isnan(y).tolist(), msg)
    known = ~np.isnan(x)
    np.testing.assert_allclose(x[known], y[known], rtol=rtol, atol=atol, err_msg=msg)


def reversed_scene(ends: List[Endpoint], starts: List[Endpoint],
                   last: int = 1000) -> Tuple[List[Endpoint], List[Endpoint]]:
    """Time-reverse a scene: frames f -> last - f, negate velocities, swap ENDs and STARTs."""
    rev_ends = [replace(s, frame=last - s.frame, vel=-np.asarray(s.vel)) for s in starts]
    rev_starts = [replace(e, frame=last - e.frame, vel=-np.asarray(e.vel)) for e in ends]
    return rev_ends, rev_starts


@st.composite
def ep_scenes(draw, max_tracklets: int = 10):
    """
    Draw (ends, starts) for one video from random endpoints (as the candidates
    tests' ``scenes``) with random appearance cues, some unknown.

    Continuous values come from a NumPy generator seeded by a drawn integer so
    failures replay and exact ties have probability zero.
    """
    n = draw(st.integers(0, max_tracklets))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    ends: List[Endpoint] = []
    starts: List[Endpoint] = []
    for k in range(n):
        first = draw(st.integers(0, 24))
        length = draw(st.integers(1, 8))
        known = length >= 2
        size = (float(rng.uniform(15, 30)), int(rng.integers(1, 20))) if draw(st.booleans()) \
            else (math.nan, 0)
        side = (float(rng.uniform(-1, 1)), int(rng.integers(1, 12))) if draw(st.booleans()) \
            else (0.0, 0)
        for frame, out in ((first, starts), (first + length - 1, ends)):
            heading = float(rng.uniform(-math.pi, math.pi)) if draw(st.booleans()) else math.nan
            shoulder = (tuple(rng.uniform(10, 40, 2)) if draw(st.booleans()) else None)
            vel = tuple(rng.uniform(-6, 6, 2)) if known else (0., 0.)
            out.append(ep(f"t{k:03d}", frame, tuple(rng.uniform(0, 60, 2)), vel, known,
                          heading, shoulder, size=size, side=side, n_frames=length))
    return ends, starts


@st.composite
def crab_scenes(draw):
    """
    Draw (ends, starts) of 2-8 straight crabs in a 120-px box with random missed
    runs, rendered by ``make_observations`` and reduced by ``geometry.endpoints``.
    Every other crab may carry its claw mirrored (opposite claw side).
    """
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    n_crabs = draw(st.integers(2, 8))
    crabs, drops = [], {}
    for c in range(1, n_crabs + 1):
        n_frames = draw(st.integers(8, 40))
        crab = straight_crab(c, draw(st.integers(0, 10)), n_frames, tuple(rng.uniform(0, 120, 2)),
                             tuple(rng.uniform(-2.5, 2.5, 2)))
        if draw(st.booleans()):
            crab.offsets = MIRRORED.copy()
        crabs.append(crab)
        runs = draw(st.lists(st.tuples(st.integers(1, n_frames - 2), st.integers(1, 12)),
                             max_size=3))
        drops[c] = {int(crab.frames[0]) + f + d for f, g in runs for d in range(g)
                    if f + d < n_frames - 1}
    obs = make_observations(crabs, drop_frames=drops)
    eps = geometry.endpoints(obs, BridgeConfig())
    return [e for _, e in eps.values()], [s for s, _ in eps.values()]


def scene_strategy():
    """Either kind of random scene."""
    return st.one_of(ep_scenes(), crab_scenes())


# ============================================================
# EXAMPLE TESTS
# ============================================================

class FeatureColumnTests(unittest.TestCase):
    """Each column against the candidates helper it is built from."""

    def setUp(self):
        size, side = (20.0, 10), (-1.0, 10)
        self.ends = [
            ep('a1', 10, (0.0, 0.0), (0.5, 0.0), heading=3.0, shoulder=(22.0, 0.0), size=size,
               side=(1.0, 10)),
            ep('a2', 10, (3.0, 1.0), (0.0, 0.0), vel_known=False, size=(24.0, 3)),
            ep('a3', 11, (1.0, -2.0), (-0.4, 0.3), heading=0.3, shoulder=(20.0, 2.0)),
        ]
        self.starts = [
            ep('b1', 13, (1.5, 0.2), (0.6, 0.1), heading=-3.0, shoulder=(22.0, 1.0), size=(21.0, 4),
               side=side),
            ep('b2', 14, (4.0, 0.0), (0.0, 0.0), vel_known=False, size=(30.0, 0), side=(0.5, 2)),
            ep('b3', 15, (-3.0, 3.0), (0.2, -0.2), heading=1.0),
        ]
        self.cfg = SQRT
        self.ends_s, self.starts_s, self.rows, self.cols = enumerate_sorted(self.ends, self.starts,
                                                                            self.cfg)
        self.X, self.gate = pair_features(self.ends_s, self.starts_s, self.rows, self.cols, self.cfg)

    def test_shape_and_geometry(self):
        self.assertEqual(self.X.shape, (len(self.rows), len(FEATURE_NAMES)))
        self.assertGreater(len(self.rows), 0)
        dt, dist, err, shoulder = candidates._pair_geometry(
            self.ends_s, self.starts_s, self.rows, self.cols, self.cfg.prediction_horizon_frames,
            self.cfg.shoulder_vector_cost_cap_px)
        np.testing.assert_array_equal(self.X[:, COL['gap']], dt - 1)
        np.testing.assert_array_equal(self.X[:, COL['dist']], dist)
        np.testing.assert_array_equal(self.X[:, COL['err']], err)
        # shoulder is exported dimensionless (capped px / cap), as the sqrt cost uses it.
        np.testing.assert_allclose(self.X[:, COL['shoulder']],
                                   shoulder / self.cfg.shoulder_vector_cost_cap_px, rtol=1e-15)
        self.assertTrue(np.all((self.X[:, COL['shoulder']] >= 0) & (self.X[:, COL['shoulder']] <= 1)))
        speeds = np.array([[np.hypot(*e.vel) if e.vel_known else 0.0 for e in self.ends_s][r]
                           for r in self.rows])
        speeds_b = np.array([[np.hypot(*s.vel) if s.vel_known else 0.0 for s in self.starts_s][c]
                             for c in self.cols])
        np.testing.assert_allclose(self.X[:, COL['speed_a']], speeds, rtol=1e-15)
        np.testing.assert_allclose(self.X[:, COL['speed_b']], speeds_b, rtol=1e-15)
        radius = candidates._sqrt_extended_radius(self.cfg, dt, np.maximum(speeds, speeds_b))
        np.testing.assert_allclose(self.X[:, COL['radius']], radius, rtol=1e-15)
        np.testing.assert_allclose(self.X[:, COL['dist_over_radius']], dist / radius, rtol=1e-15)
        np.testing.assert_allclose(self.X[:, COL['err_over_radius']], err / radius, rtol=1e-15)
        known = {e.tracklet_id: float(e.vel_known) for e in self.ends}
        self.assertEqual(self.X[:, COL['vel_known_a']].tolist(),
                         [known[self.ends_s[r].tracklet_id] for r in self.rows])

    def test_cue_columns(self):
        size_term, side_term, _, _ = candidates._cue_terms_full(self.ends_s, self.starts_s,
                                                                self.rows, self.cols, self.cfg)
        for k, (r, c) in enumerate(zip(self.rows, self.cols)):
            a, b = self.ends_s[r], self.starts_s[c]
            x = self.X[k]
            sized = a.size_n > 0 and b.size_n > 0
            self.assertEqual(math.isnan(x[COL['size_term']]), not sized, (a.tracklet_id, b.tracklet_id))
            self.assertEqual(math.isnan(x[COL['size_diff']]), not sized)
            if sized:
                self.assertEqual(x[COL['size_term']], size_term[k])
                self.assertAlmostEqual(x[COL['size_diff']], abs(a.size_px - b.size_px), places=12)
            sided = a.claw_side_n > 0 and b.claw_side_n > 0
            self.assertEqual(math.isnan(x[COL['side_product']]), not sided)
            self.assertEqual(math.isnan(x[COL['side_term']]), not sided)
            if sided:
                self.assertEqual(x[COL['side_product']], a.claw_side * b.claw_side)
                self.assertEqual(x[COL['side_term']], side_term[k])
            self.assertEqual(math.isnan(x[COL['abs_side_a']]), a.claw_side_n == 0)
            self.assertEqual(math.isnan(x[COL['abs_side_b']]), b.claw_side_n == 0)
            for name, value in (('size_n_a', a.size_n), ('size_n_b', b.size_n),
                                ('claw_side_n_a', a.claw_side_n), ('claw_side_n_b', b.claw_side_n),
                                ('n_frames_a', a.n_frames), ('n_frames_b', b.n_frames)):
                self.assertEqual(x[COL[name]], float(value), name)
        # a1 (side +1) -> b1 (side -1): a negative product, both confident.
        k = [(self.ends_s[r].tracklet_id, self.starts_s[c].tracklet_id)
             for r, c in zip(self.rows, self.cols)].index(('a1', 'b1'))
        self.assertEqual(self.X[k, COL['side_product']], -1.0)

    def test_heading_change_wraps(self):
        pairs = [(self.ends_s[r].tracklet_id, self.starts_s[c].tracklet_id)
                 for r, c in zip(self.rows, self.cols)]
        k = pairs.index(('a1', 'b1'))            # headings 3.0 and -3.0
        self.assertAlmostEqual(self.X[k, COL['heading_change']], 2 * math.pi - 6.0, places=12)
        self.assertTrue(math.isnan(self.X[pairs.index(('a2', 'b1')), COL['heading_change']]))
        self.assertAlmostEqual(self.X[pairs.index(('a3', 'b3')), COL['heading_change']], 0.7,
                               places=12)

    def test_hand_cost_and_gate(self):
        _, cost, gate = candidates._pair_costs_sqrt(self.ends_s, self.starts_s, self.rows,
                                                    self.cols, self.cfg)
        np.testing.assert_array_equal(self.X[:, COL['hand_cost']], cost)
        self.assertEqual(self.gate.tolist(), gate.tolist())
        flip = candidates._axis_flip(self.ends_s, self.starts_s, self.rows, self.cols)
        np.testing.assert_array_equal(self.X[:, COL['axis_flip']], flip.astype(float))

    def test_learned_cost_model_gives_hand_features(self):
        learned = replace(self.cfg, cost_model='learned', scorer_path='unused.joblib')
        X, gate = pair_features(self.ends_s, self.starts_s, self.rows, self.cols, learned)
        assert_same(self, X, self.X)
        self.assertEqual(gate.tolist(), self.gate.tolist())

    def test_linear_gate_model_refused(self):
        with self.assertRaises(ValueError):
            pair_features(self.ends_s, self.starts_s, self.rows, self.cols,
                          replace(self.cfg, gate_model='linear'))
        with self.assertRaises(ValueError):
            pair_feature_table(self.ends, self.starts, replace(self.cfg, gate_model='linear'), 'v')

    def test_empty_inputs(self):
        X, gate = pair_features(self.ends_s, self.starts_s, np.zeros(0, np.int64),
                                np.zeros(0, np.int64), self.cfg)
        self.assertEqual(X.shape, (0, len(FEATURE_NAMES)))
        self.assertEqual(gate.shape, (0,))
        for args in ((self.ends, self.starts, replace(self.cfg, max_gap=0)),
                     ([], self.starts, self.cfg), (self.ends, [], self.cfg)):
            table = pair_feature_table(*args, 'v')
            self.assertEqual(table.num_rows, 0)
            self.assertTrue(table.schema.equals(PAIR_FEATURES_SCHEMA))
        self.assertEqual(feature_matrix(PAIR_FEATURES_SCHEMA.empty_table()).shape,
                         (0, len(FEATURE_NAMES)))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            pair_feature_table(self.ends, [replace(self.starts[0], video_id='w')] + self.starts[1:],
                               self.cfg, 'v')
        with self.assertRaises(ValueError):
            pair_feature_table(self.ends + [self.ends[0]], self.starts, self.cfg, 'v')
        with self.assertRaises(ValueError):
            feature_matrix(pa.table({'gap': [1.0]}))

    def test_feature_matrix_nulls_and_subset(self):
        table = pa.table({'gap': pa.array([1.0, None]), 'dist': pa.array([None, 2.5])})
        X = feature_matrix(table, ('dist', 'gap'))
        self.assertTrue(math.isnan(X[0, 0]) and math.isnan(X[1, 1]))
        self.assertEqual((X[1, 0], X[0, 1]), (2.5, 1.0))


class RivalCountTests(unittest.TestCase):
    """
    Three ENDs at frame 10, three STARTs at frames 12, 13 and 15 (dt 2, 3, 5).

    e3 (claw side -1) -> s1 (claw side +1) is claw-side vetoed; every pair into
    s3 (20 px away) fails the displacement gate. By hand, with rivals counted
    over motion-gated pairs with dt' <= dt:
    """

    EXPECTED = {   # (a, b): (gate, n_rival_starts, n_rival_ends)
        ('e1', 's1'): ('', 0, 1), ('e1', 's2'): ('', 1, 2), ('e1', 's3'): ('displacement', 2, 0),
        ('e2', 's1'): ('', 0, 1), ('e2', 's2'): ('', 1, 2), ('e2', 's3'): ('displacement', 2, 0),
        ('e3', 's1'): ('claw_side', 0, 2), ('e3', 's2'): ('', 0, 2),
        ('e3', 's3'): ('displacement', 1, 0),
    }

    def scene(self) -> Tuple[List[Endpoint], List[Endpoint]]:
        ends = [ep('e1', 10, (0.0, 0.0)), ep('e2', 10, (2.0, 0.0)),
                ep('e3', 10, (0.0, 2.0), side=(-1.0, 10))]
        starts = [ep('s1', 12, (1.0, 0.0), side=(1.0, 10)), ep('s2', 13, (0.0, 1.0)),
                  ep('s3', 15, (20.0, 0.0))]
        return ends, starts

    def test_hand_counts(self):
        table = pair_feature_table(*self.scene(), SQRT, 'v')
        got = {(a, b): (g, rs, re) for a, b, g, rs, re in zip(
            table['tracklet_a'].to_pylist(), table['tracklet_b'].to_pylist(),
            table['gate'].to_pylist(), table['n_rival_starts'].to_pylist(),
            table['n_rival_ends'].to_pylist())}
        self.assertEqual(got, self.EXPECTED)

    def test_order_and_cost_match_link_tracklets(self):
        ends, starts = self.scene()
        _, decisions = candidates.link_tracklets(ends, starts, SQRT)
        table = pair_feature_table(ends, starts, SQRT, 'v')
        self.assertEqual(list(zip(table['tracklet_a'].to_pylist(), table['tracklet_b'].to_pylist(),
                                  table['gap_frames'].to_pylist())),
                         [(d.tracklet_a, d.tracklet_b, d.gap_frames) for d in decisions])
        for d, cost, gate in zip(decisions, table['hand_cost'].to_pylist(),
                                 table['gate'].to_pylist()):
            if math.isfinite(d.cost):
                self.assertEqual(cost, d.cost)
            else:
                self.assertEqual(d.decision, 'refused_gate')
                self.assertEqual(d.reason.split(';')[0], gate)
                self.assertTrue(math.isfinite(cost))   # features keep the gated pair's cost


# ============================================================
# PROPERTY TESTS
# ============================================================

class FeaturePropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=120)
    @given(scene_strategy(), st.integers(1, 30))
    def test_property_table_matches_link_tracklets(self, scene, max_gap):
        ends, starts = scene
        cfg = replace(SQRT, max_gap=max_gap)
        table = pair_feature_table(ends, starts, cfg, 'v')
        _, decisions = candidates.link_tracklets(ends, starts, cfg)
        self.assertTrue(table.schema.equals(PAIR_FEATURES_SCHEMA))
        self.assertEqual(list(zip(table['tracklet_a'].to_pylist(), table['tracklet_b'].to_pylist(),
                                  table['gap_frames'].to_pylist())),
                         [(d.tracklet_a, d.tracklet_b, d.gap_frames) for d in decisions])
        X = feature_matrix(table)
        self.assertFalse(np.isinf(X).any())
        np.testing.assert_array_equal(X[:, COL['gap']], np.asarray(table['gap_frames'].to_pylist(),
                                                                   dtype=float))
        gate = np.asarray(table['gate'].to_pylist(), dtype=object)
        for d, cost, g in zip(decisions, X[:, COL['hand_cost']], gate):
            if math.isfinite(d.cost):
                self.assertEqual(cost, d.cost)
            if d.decision == 'refused_gate':
                self.assertEqual(d.reason.split(';')[0], g)
        # Gate equivalences: displacement iff dist/R > 1; among the rest,
        # prediction_error iff err/R > prediction_error_ratio.
        disp = gate == 'displacement'
        self.assertEqual((X[:, COL['dist_over_radius']] <= 1.0).tolist(), (~disp).tolist())
        rest = ~disp
        self.assertEqual((X[rest, COL['err_over_radius']] <= cfg.prediction_error_ratio).tolist(),
                         (gate[rest] != 'prediction_error').tolist())

    @settings(deadline=None, max_examples=120)
    @given(scene_strategy())
    def test_property_time_reversal(self, scene):
        ends, starts = scene
        cfg = replace(SQRT, max_gap=30)
        forward = pair_feature_table(ends, starts, cfg, 'v')
        backward = pair_feature_table(*reversed_scene(ends, starts), cfg, 'v')
        fx, bx = keyed(forward), {(b, a): x for (a, b), x in keyed(backward).items()}
        self.assertEqual(set(fx), set(bx))
        fgate = dict(zip(zip(forward['tracklet_a'].to_pylist(), forward['tracklet_b'].to_pylist()),
                         forward['gate'].to_pylist()))
        bgate = dict(zip(zip(backward['tracklet_b'].to_pylist(), backward['tracklet_a'].to_pylist()),
                         backward['gate'].to_pylist()))
        self.assertEqual(fgate, bgate)
        perm = list(range(len(FEATURE_NAMES)))
        for a, b in SWAPS:
            perm[COL[a]], perm[COL[b]] = COL[b], COL[a]
        for key, x in fx.items():
            y = bx[key][perm]
            assert_same(self, x, y, msg=str(key), rtol=1e-9, atol=1e-9)

    @settings(deadline=None, max_examples=80)
    @given(scene_strategy(), st.lists(st.integers(1, 60), min_size=1, max_size=4))
    def test_property_features_independent_of_horizon(self, scene, horizons):
        ends, starts = scene
        reference = keyed(pair_feature_table(ends, starts, replace(SQRT, max_gap=60), 'v'))
        for max_gap in horizons:
            table = pair_feature_table(ends, starts, replace(SQRT, max_gap=max_gap), 'v')
            got = keyed(table)
            # Every pair of a smaller horizon is enumerated at 60 with identical features.
            self.assertLessEqual(set(got), set(reference))
            for key, x in got.items():
                assert_same(self, x, reference[key], msg=f"{key} at max_gap {max_gap}")
            # ... and the smaller horizon keeps exactly those within its gap and R_max prefilter.
            cfg = replace(SQRT, max_gap=max_gap)
            r_max = candidates._prune_radius(ends, starts, cfg)
            self.assertEqual(set(got), {key for key, x in reference.items()
                                        if x[COL['gap']] <= max_gap and x[COL['dist']] <= r_max})


# ============================================================
# DEMO
# ============================================================

@unittest.skipUnless(DEMO.exists(), "demo observations not present")
class DemoFeatureTests(unittest.TestCase):
    """Demo clip at the default horizon (30): rows and costs equal links.parquet."""

    @classmethod
    def setUpClass(cls):
        from gap_bridging import pipeline
        cls.cfg = BridgeConfig()
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.run(DEMO, Path(tmp), cls.cfg)
            cls.links = pq.read_table(Path(tmp) / 'links.parquet')
        obs = pq.read_table(DEMO)
        eps = geometry.endpoints(obs, cls.cfg)
        videos = sorted({e.video_id for _, e in eps.values()})
        assert len(videos) == 1, "the demo is one video"
        cls.table = pair_feature_table([e for _, e in eps.values()], [s for s, _ in eps.values()],
                                       cls.cfg, videos[0])

    def test_rows_match_links(self):
        self.assertEqual(self.cfg.max_gap, 30)
        self.assertEqual(self.table.num_rows, self.links.num_rows)
        for name in ('tracklet_a', 'tracklet_b', 'gap_frames', 'video_id'):
            self.assertEqual(self.table[name].to_pylist(), self.links[name].to_pylist(), name)

    def test_hand_cost_matches_links_cost(self):
        cost = np.asarray(self.links['cost'].to_pylist(), dtype=float)
        hand = np.asarray(self.table['hand_cost'].to_pylist(), dtype=float)
        finite = np.isfinite(cost)
        np.testing.assert_array_equal(hand[finite], cost[finite])
        # links reports NaN for gate-refused pairs; claw-side refusals keep a finite hand cost.
        reason = np.asarray([r.split(';')[0] for r in self.links['reason'].to_pylist()],
                            dtype=object)   # a max_cost reason may carry '; axis_flip'
        claw = reason == 'claw_side'
        self.assertTrue(np.all(np.isfinite(hand[claw])))
        gate = np.asarray(self.table['gate'].to_pylist(), dtype=object)
        refused = np.asarray(self.links['decision'].to_pylist(), dtype=object) == 'refused_gate'
        self.assertEqual(gate[refused].tolist(), reason[refused].tolist())
        X = feature_matrix(self.table)
        self.assertFalse(np.isinf(X).any())


if __name__ == '__main__':
    unittest.main()
