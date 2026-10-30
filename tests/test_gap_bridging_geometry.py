#!/usr/bin/env python3

"""
Tests for scripts/gap_bridging/geometry.py (WP-A): endpoints, Hermite and
angle interpolation, rigid node placement, bridge filling and synthetic rows.

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from pathlib import Path
import sys
import unittest

# ----- numerics -----
from hypothesis import example, given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# ----- local modules -----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.geometry import (damped_tangents, endpoints, fill_bridge,  # noqa: E402
                                   fill_confidence, hermite, interp_angle, place_nodes,
                                   synthetic_rows)
from gap_bridging.types import Endpoint  # noqa: E402
from gap_bridging_fixtures import (DEFAULT_OFFSETS, FPS, NODES, SCHEMA, make_observations,  # noqa: E402
                                   node_positions, straight_crab, turning_crab)

# ============================================================
# HELPERS
# ============================================================

CFG = BridgeConfig(gate_model='sqrt')     # damped fill; the linear model fills undamped
LINEAR = BridgeConfig(gate_model='linear')
DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'

coord = st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False, allow_subnormal=False)
speed = st.floats(-50, 50, allow_nan=False, allow_infinity=False, allow_subnormal=False)
unit = st.floats(0.0, 1.0, allow_nan=False)
angle = st.floats(-math.pi, math.pi, allow_nan=False)
vec = st.tuples(coord, coord).map(np.array)
vel = st.tuples(speed, speed).map(np.array)
dts = st.integers(2, 6)


def angdiff(x: float, y: float) -> float:
    """Absolute angular distance in [0, pi]."""
    return abs(math.atan2(math.sin(x - y), math.cos(x - y)))


def motion_endpoint(frame: int, pos, vel, known: bool = True) -> Endpoint:
    """Bare endpoint with a position and velocity; heading and offsets unknown."""
    return Endpoint('t', 'v', int(frame), 3 if known else 1, np.asarray(pos, float),
                    np.asarray(vel, float), known, math.nan, np.full((5, 2), np.nan))


def chord_deviation(bridge, a: Endpoint, b: Endpoint) -> np.ndarray:
    """Distance of each filled shell position from the uniform-time chord point."""
    dt = b.frame - a.frame
    t = (bridge.frames - a.frame)[:, None] / dt
    return np.linalg.norm(bridge.positions - (a.pos + t * (b.pos - a.pos)), axis=1)


def damped_bound(a: Endpoint, b: Endpoint, horizon: int) -> float:
    """(4/27) * dt * lam * (|va - c| + |vb - c|) with lam = min(1, horizon / dt)."""
    dt = b.frame - a.frame
    c = (b.pos - a.pos) / dt
    lam = min(1.0, horizon / dt)
    return (4 / 27) * dt * lam * (np.linalg.norm(a.vel - c) + np.linalg.norm(b.vel - c))


# ============================================================
# ENDPOINT TESTS
# ============================================================


class EndpointTests(unittest.TestCase):

    def test_constant_velocity_gives_exact_velocity(self):
        crab = straight_crab(1, 10, 12, origin=(100.0, 200.0), velocity=(2.0, -1.0))
        result = endpoints(make_observations([crab]), CFG)
        self.assertEqual(len(result), 1)
        start, end = next(iter(result.values()))
        self.assertEqual((start.frame, end.frame), (10, 21))
        self.assertEqual((start.n_frames, end.n_frames), (12, 12))
        self.assertTrue(start.vel_known and end.vel_known)
        np.testing.assert_allclose(start.vel, [2.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(end.vel, [2.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(start.pos, [100.0, 200.0])
        np.testing.assert_allclose(end.pos, [122.0, 189.0])
        self.assertEqual(start.video_id, 'synthetic')

    def test_single_frame_has_zero_unknown_velocity(self):
        crab = straight_crab(1, 5, 1, origin=(50.0, 60.0), velocity=(3.0, 0.0))
        start, end = next(iter(endpoints(make_observations([crab]), CFG).values()))
        self.assertFalse(start.vel_known or end.vel_known)
        np.testing.assert_array_equal(start.vel, [0.0, 0.0])
        np.testing.assert_array_equal(end.vel, [0.0, 0.0])
        self.assertEqual(start.frame, end.frame)
        self.assertAlmostEqual(start.heading, 0.0)

    def test_two_frames_use_one_step(self):
        crab = straight_crab(1, 0, 2, origin=(0.0, 0.0), velocity=(1.5, 2.0))
        start, end = next(iter(endpoints(make_observations([crab]), CFG).values()))
        self.assertTrue(start.vel_known)
        np.testing.assert_allclose(start.vel, [1.5, 2.0])
        np.testing.assert_allclose(end.vel, [1.5, 2.0])

    def test_speed_is_clipped(self):
        crab = straight_crab(1, 0, 8, origin=(0.0, 0.0), velocity=(10.0, 0.0))
        start, end = next(iter(endpoints(make_observations([crab]), CFG).values()))
        np.testing.assert_allclose(end.vel, [6.0, 0.0])
        np.testing.assert_allclose(start.vel, [6.0, 0.0])
        slow = BridgeConfig(max_velocity_px_per_frame=3.0)
        diagonal = straight_crab(2, 0, 8, origin=(0.0, 0.0), velocity=(4.0, 4.0))
        _, end = next(iter(endpoints(make_observations([diagonal]), slow).values()))
        self.assertAlmostEqual(float(np.hypot(*end.vel)), 3.0)
        self.assertAlmostEqual(float(end.vel[0]), float(end.vel[1]))

    def test_heading_nan_when_shoulder_missing(self):
        crab = straight_crab(1, 0, 6, origin=(0.0, 0.0), velocity=(1.0, 0.0), heading=float('nan'))
        start, end = next(iter(endpoints(make_observations([crab]), CFG).values()))
        for point in (start, end):
            self.assertTrue(math.isnan(point.heading))
            np.testing.assert_array_equal(point.offsets[0], [0.0, 0.0])
            self.assertTrue(np.isnan(point.offsets[1:]).all())

    def test_heading_from_nearest_valid_frame_in_window(self):
        crab = straight_crab(1, 0, 10, origin=(0.0, 0.0), velocity=(1.0, 0.0), heading=0.5)
        crab.headings[-1] = np.nan          # end frame: shoulder missing
        crab.headings[-2] = 0.7             # nearest valid frame
        crab.headings[0:4] = np.nan         # start window (frames 0..3) has none
        start, end = next(iter(endpoints(make_observations([crab]), CFG).values()))
        self.assertAlmostEqual(end.heading, 0.7)
        self.assertTrue(math.isnan(start.heading))
        np.testing.assert_allclose(end.offsets, DEFAULT_OFFSETS, atol=1e-9)

    def test_shoulder_length_outside_range_is_not_a_heading(self):
        crab = straight_crab(1, 0, 5, origin=(0.0, 0.0), velocity=(1.0, 0.0), heading=0.3)
        crab.offsets = DEFAULT_OFFSETS.copy()
        crab.offsets[1] = [5.0, 0.0]        # shoulder only 5 px from the shell
        start, end = next(iter(endpoints(make_observations([crab]), CFG).values()))
        self.assertTrue(math.isnan(start.heading) and math.isnan(end.heading))
        crab.offsets[1] = [70.0, 0.0]
        start, _ = next(iter(endpoints(make_observations([crab]), CFG).values()))
        self.assertTrue(math.isnan(start.heading))

    def test_offsets_invariant_under_rotation_by_90_degrees(self):
        base = turning_crab(1, 0, 9, center=(400.0, 300.0), radius=80.0, omega=0.05, phase=0.2)
        quarter = np.array([[0.0, -1.0], [1.0, 0.0]])
        rotated = turning_crab(1, 0, 9, center=(400.0, 300.0), radius=80.0, omega=0.05, phase=0.2)
        rotated.positions = base.positions @ quarter.T
        rotated.headings = base.headings + math.pi / 2
        s0, e0 = next(iter(endpoints(make_observations([base]), CFG).values()))
        s1, e1 = next(iter(endpoints(make_observations([rotated]), CFG).values()))
        np.testing.assert_allclose(s1.offsets, s0.offsets, atol=1e-9)
        np.testing.assert_allclose(e1.offsets, e0.offsets, atol=1e-9)
        np.testing.assert_allclose(e0.offsets, DEFAULT_OFFSETS, atol=1e-9)
        self.assertAlmostEqual(angdiff(e1.heading, e0.heading + math.pi / 2), 0.0, places=9)
        np.testing.assert_allclose(e1.vel, quarter @ e0.vel, atol=1e-9)

    def test_invisible_node_gives_nan_offset_row(self):
        crab = straight_crab(1, 0, 5, origin=(0.0, 0.0), velocity=(1.0, 0.0), heading=0.0)
        table = make_observations([crab])
        table = table.set_column(table.schema.get_field_index('claw_visible'), 'claw_visible',
                                 pa.array([False] * table.num_rows))
        start, _ = next(iter(endpoints(table, CFG).values()))
        self.assertTrue(np.isnan(start.offsets[NODES.index('claw')]).all())
        np.testing.assert_allclose(start.offsets[NODES.index('elbow')], DEFAULT_OFFSETS[2], atol=1e-9)

    def test_rejected_rows_and_split_tracklets(self):
        crab = straight_crab(1, 0, 12, origin=(0.0, 0.0), velocity=(1.0, 1.0))
        other = straight_crab(2, 0, 12, origin=(500.0, 0.0), velocity=(-1.0, 0.0))
        table = make_observations([crab, other], drop_frames={1: [5, 6]},
                                  reject_frames={2: {11: 'low_body_score'}})
        result = endpoints(table, CFG)
        self.assertEqual(len(result), 3)
        spans = sorted((s.frame, e.frame, s.n_frames) for s, e in result.values())
        self.assertEqual(spans, [(0, 4, 5), (0, 10, 11), (7, 11, 5)])

    def test_non_contiguous_tracklet_raises(self):
        crab = straight_crab(1, 0, 10, origin=(0.0, 0.0), velocity=(1.0, 0.0))
        table = make_observations([crab], drop_frames={1: [4]})
        index = table.schema.get_field_index('tracklet_id')
        merged = pa.array(['same' if v is not None else None for v in table['tracklet_id'].to_pylist()])
        with self.assertRaises(ValueError):
            endpoints(table.set_column(index, 'tracklet_id', merged), CFG)

    def test_empty_table_gives_no_endpoints(self):
        self.assertEqual(endpoints(SCHEMA.empty_table(), CFG), {})

    @unittest.skipUnless(DEMO.exists(), 'demo observations absent')
    def test_demo_clip_endpoints(self):
        table = pq.read_table(DEMO)
        result = endpoints(table, CFG)
        accepted = table.filter(pc.fill_null(table['accepted_for_tracking'], False))
        self.assertEqual(len(result), len(set(accepted['tracklet_id'].to_pylist())))
        self.assertEqual(sum(s.n_frames for s, _ in result.values()), accepted.num_rows)
        speeds = [np.hypot(*p.vel) for pair in result.values() for p in pair]
        self.assertLessEqual(max(speeds), CFG.max_velocity_px_per_frame + 1e-9)
        self.assertTrue(all(np.isfinite(p.pos).all() for pair in result.values() for p in pair))


# ============================================================
# FILL TESTS
# ============================================================


class FillTests(unittest.TestCase):

    def bridge_for(self, crab, drop):
        table = make_observations([crab], drop_frames={crab.crab_id: drop})
        pairs = sorted(endpoints(table, CFG).values(), key=lambda p: p[0].frame)
        self.assertEqual(len(pairs), 2)
        return fill_bridge(pairs[0][1], pairs[1][0], 'synthetic/trajectory0000001',
                           'synthetic/trajectory0000001/b001', CFG)

    def test_straight_crab_bridge_is_exact(self):
        crab = straight_crab(1, 0, 20, origin=(100.0, 100.0), velocity=(2.0, 1.0))
        bridge = self.bridge_for(crab, [8, 9, 10])
        np.testing.assert_array_equal(bridge.frames, [8, 9, 10])
        np.testing.assert_allclose(bridge.positions, crab.positions[8:11], atol=1e-9)
        expected = np.stack([node_positions(crab.positions[f], crab.headings[f]) for f in (8, 9, 10)])
        np.testing.assert_allclose(bridge.nodes, expected, atol=1e-9)
        np.testing.assert_array_equal(bridge.nodes[:, 0], bridge.positions)
        self.assertEqual(bridge.bridge_id, 'synthetic/trajectory0000001/b001')

    def test_turning_crab_headings_interpolate(self):
        crab = turning_crab(1, 0, 20, center=(500.0, 500.0), radius=60.0, omega=0.08)
        bridge = self.bridge_for(crab, [9, 10])
        for i, frame in enumerate((9, 10)):
            self.assertLess(angdiff(bridge.headings[i], crab.headings[frame]), 1e-9)
            self.assertLess(np.hypot(*(bridge.positions[i] - crab.positions[frame])), 0.5)

    def test_linear_when_velocity_unknown(self):
        a_crab = straight_crab(1, 0, 1, origin=(0.0, 0.0), velocity=(5.0, 5.0))
        b_crab = straight_crab(2, 4, 3, origin=(12.0, -8.0), velocity=(5.0, 5.0))
        result = endpoints(make_observations([a_crab, b_crab]), CFG)
        (_, a), (b, _) = sorted(result.values(), key=lambda p: p[0].frame)
        bridge = fill_bridge(a, b, 't', 't/b001', CFG)
        expected = np.array([[3.0, -2.0], [6.0, -4.0], [9.0, -6.0]])
        np.testing.assert_allclose(bridge.positions, expected, atol=1e-12)

    def test_one_sided_heading_and_offsets(self):
        a_crab = straight_crab(1, 0, 5, origin=(0.0, 0.0), velocity=(1.0, 0.0), heading=float('nan'))
        b_crab = straight_crab(2, 7, 5, origin=(7.0, 0.0), velocity=(1.0, 0.0), heading=0.4)
        result = endpoints(make_observations([a_crab, b_crab]), CFG)
        (_, a), (b, _) = sorted(result.values(), key=lambda p: p[0].frame)
        bridge = fill_bridge(a, b, 't', 't/b001', CFG)
        np.testing.assert_allclose(bridge.headings, [0.4, 0.4])
        expected = node_positions(bridge.positions[0], 0.4)
        np.testing.assert_allclose(bridge.nodes[0], expected, atol=1e-9)

    def test_fill_confidence_stored_on_bridge(self):
        for gap, expected in ((1, 'high'), (5, 'high'), (6, 'medium'), (15, 'medium'), (16, 'low')):
            a = motion_endpoint(10, (0.0, 0.0), (1.0, 0.0))
            b = motion_endpoint(11 + gap, (5.0, 0.0), (1.0, 0.0))
            with self.subTest(gap=gap):
                self.assertEqual(fill_bridge(a, b, 't', 't/b001', CFG).fill_confidence, expected)

    def test_invalid_bridges_raise(self):
        crab = straight_crab(1, 0, 5, origin=(0.0, 0.0), velocity=(1.0, 0.0))
        start, end = next(iter(endpoints(make_observations([crab]), CFG).values()))
        with self.assertRaises(ValueError):
            fill_bridge(end, start, 't', 't/b001', CFG)


class DampedFillTests(unittest.TestCase):

    def test_unchanged_within_horizon(self):
        horizon = CFG.prediction_horizon_frames
        a = motion_endpoint(20, (100.0, 50.0), (3.0, -2.0))
        for dt in range(2, horizon + 1):
            b = motion_endpoint(20 + dt, (104.0, 49.0), (-1.5, 4.0))
            bridge = fill_bridge(a, b, 't', 't/b001', CFG)
            undamped = np.stack([hermite(a.pos, a.vel, b.pos, b.vel, dt, (f - a.frame) / dt)
                                 for f in bridge.frames])
            with self.subTest(dt=dt):
                np.testing.assert_array_equal(bridge.positions, undamped)

    def test_long_gap_bound_and_damping(self):
        dt = 60
        a = motion_endpoint(0, (100.0, 100.0), (4.0, 3.0))
        b = motion_endpoint(dt, (110.0, 95.0), (-3.0, 5.0))
        bridge = fill_bridge(a, b, 't', 't/b001', CFG)
        deviation = chord_deviation(bridge, a, b)
        self.assertLessEqual(deviation.max(), damped_bound(a, b, CFG.prediction_horizon_frames) + 1e-9)
        undamped = np.stack([hermite(a.pos, a.vel, b.pos, b.vel, dt, (f - a.frame) / dt)
                             for f in bridge.frames])
        undamped_dev = np.linalg.norm(undamped - (a.pos + ((bridge.frames - a.frame)[:, None] / dt)
                                                  * (b.pos - a.pos)), axis=1)
        self.assertLess(deviation.max(), 0.1 * undamped_dev.max())
        self.assertEqual(bridge.fill_confidence, 'low')

    def test_tends_to_chord(self):
        a_vel, b_vel = np.array([5.0, -2.0]), np.array([-4.0, 6.0])
        horizon = CFG.prediction_horizon_frames
        for dt in (4, 10, 30, 100, 1000):
            a = motion_endpoint(0, (0.0, 0.0), a_vel)
            b = motion_endpoint(dt, (20.0, 10.0), b_vel)
            c = (b.pos - a.pos) / dt
            # Damped: bounded by a dt-independent constant, so the fill hugs the chord as dt grows.
            limit = (4 / 27) * horizon * (np.linalg.norm(a_vel - c) + np.linalg.norm(b_vel - c))
            bridge = fill_bridge(a, b, 't', 't/b001', CFG)
            deviation = chord_deviation(bridge, a, b).max()
            t = (bridge.frames - a.frame) / dt
            undamped = np.stack([hermite(a.pos, a_vel, b.pos, b_vel, dt, x) for x in t])
            undamped_dev = np.linalg.norm(undamped - (a.pos + t[:, None] * (b.pos - a.pos)), axis=1).max()
            with self.subTest(dt=dt):
                self.assertLessEqual(deviation, limit + 1e-9)
                self.assertGreater(undamped_dev, dt / 30 * limit)  # undamped grows linearly in dt

    def test_constant_velocity_exact_for_any_gap(self):
        for dt in (2, 7, 45):
            a = motion_endpoint(5, (10.0, 20.0), (1.5, -0.5))
            b = motion_endpoint(5 + dt, (10.0 + 1.5 * dt, 20.0 - 0.5 * dt), (1.5, -0.5))
            bridge = fill_bridge(a, b, 't', 't/b001', CFG)
            expected = a.pos + (bridge.frames - a.frame)[:, None] * a.vel
            np.testing.assert_allclose(bridge.positions, expected, rtol=0, atol=1e-9)

    def test_linear_fallback_not_damped(self):
        a = motion_endpoint(0, (0.0, 0.0), (9.0, 9.0), known=False)
        b = motion_endpoint(30, (30.0, -60.0), (-9.0, 4.0))
        bridge = fill_bridge(a, b, 't', 't/b001', CFG)
        expected = (bridge.frames[:, None] / 30) * np.array([30.0, -60.0])
        np.testing.assert_allclose(bridge.positions, expected, rtol=0, atol=1e-9)

    def test_regression_D002_linear_fill_not_damped(self):
        """D-002: under gate_model='linear' the fill is the undamped (v001) Hermite."""
        for dt in (2, 3, 4, 60):
            a = motion_endpoint(0, (100.0, 100.0), (4.0, 3.0))
            b = motion_endpoint(dt, (110.0, 95.0), (-3.0, 5.0))
            bridge = fill_bridge(a, b, 't', 't/b001', LINEAR)
            undamped = np.stack([hermite(a.pos, a.vel, b.pos, b.vel, dt, (f - a.frame) / dt)
                                 for f in bridge.frames])
            with self.subTest(dt=dt):
                np.testing.assert_array_equal(bridge.positions, undamped)
                if dt > CFG.prediction_horizon_frames:   # sqrt damps the same bridge
                    damped = fill_bridge(a, b, 't', 't/b001', CFG).positions
                    self.assertGreater(np.abs(damped - undamped).max(), 1e-3)

    def test_damped_tangents_validation(self):
        with self.assertRaises(ValueError):
            damped_tangents(np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2), 0, 3)
        with self.assertRaises(ValueError):
            damped_tangents(np.zeros(2), np.zeros(2), np.ones(2), np.zeros(2), 5, 0)


class FillConfidenceTests(unittest.TestCase):

    def test_default_boundaries(self):
        self.assertEqual(CFG.fill_confidence_gaps, (5, 15))
        cases = {1: 'high', 5: 'high', 6: 'medium', 15: 'medium', 16: 'low', 59: 'low'}
        self.assertEqual({g: fill_confidence(g, CFG) for g in cases}, cases)

    def test_custom_thresholds_and_invalid_gap(self):
        cfg = BridgeConfig(fill_confidence_gaps=(2, 2))
        self.assertEqual([fill_confidence(g, cfg) for g in (1, 2, 3)], ['high', 'high', 'low'])
        with self.assertRaises(ValueError):
            fill_confidence(0, CFG)

    @given(st.integers(1, 500), st.integers(1, 100), st.integers(0, 100))
    def test_monotone_in_gap(self, gap, high, extra):
        cfg = BridgeConfig(fill_confidence_gaps=(high, high + extra))
        rank = {'high': 0, 'medium': 1, 'low': 2}
        self.assertLessEqual(rank[fill_confidence(gap, cfg)], rank[fill_confidence(gap + 1, cfg)])


class SyntheticRowTests(unittest.TestCase):

    def make_bridge(self, heading=0.3):
        crab = straight_crab(1, 0, 12, origin=(10.0, 20.0), velocity=(1.0, 2.0), heading=heading)
        table = make_observations([crab], drop_frames={1: [5, 6, 7]})
        pairs = sorted(endpoints(table, CFG).values(), key=lambda p: p[0].frame)
        return fill_bridge(pairs[0][1], pairs[1][0], 'synthetic/trajectory0000001',
                           'synthetic/trajectory0000001/b001', CFG)

    def test_rows_match_template_schema(self):
        bridge = self.make_bridge()
        rows = synthetic_rows(bridge, SCHEMA, fps=FPS)
        self.assertTrue(rows.schema.equals(SCHEMA))
        self.assertEqual(rows.num_rows, 3)
        self.assertEqual(rows['frame'].to_pylist(), [5, 6, 7])
        self.assertEqual(rows['detection_id'].to_pylist()[0],
                         'synthetic/f000005/synthetic/trajectory0000001/b001')
        for node in NODES:
            self.assertEqual(rows[f'{node}_score'].null_count, 3)
            self.assertEqual(rows[f'{node}_visible'].to_pylist(), [True] * 3)
        self.assertEqual(rows['body_score'].null_count, 3)
        self.assertEqual(rows['tracklet_id'].null_count, 3)
        self.assertEqual(rows['accepted_for_tracking'].to_pylist(), [False] * 3)
        self.assertEqual(set(rows['centroid_source'].to_pylist()), {'interpolated'})
        self.assertEqual(set(rows['incoming_reason'].to_pylist()), {'bridged'})
        self.assertEqual(set(rows['instance_index'].to_pylist()), {-1})
        self.assertEqual(set(rows['raw_instance_id'].to_pylist()), {-1})
        np.testing.assert_allclose(rows['body_x'].to_numpy(), rows['shell_x'].to_numpy())
        np.testing.assert_allclose(rows['body_x'].to_numpy(), [15.0, 16.0, 17.0], atol=1e-9)
        np.testing.assert_allclose(rows['time_s'].to_numpy(), np.array([5, 6, 7]) / FPS)

    def test_unknown_nodes_are_null_and_invisible(self):
        rows = synthetic_rows(self.make_bridge(heading=float('nan')), SCHEMA)
        self.assertEqual(rows['time_s'].null_count, 3)
        self.assertEqual(rows['shell_visible'].to_pylist(), [True] * 3)
        for node in NODES[1:]:
            self.assertEqual(rows[f'{node}_x'].null_count, 3)
            self.assertEqual(rows[f'{node}_visible'].to_pylist(), [False] * 3)
            self.assertEqual(rows[f'{node}_score'].null_count, 3)

    def test_centroid_equals_body_when_template_has_it(self):
        bridge = self.make_bridge()
        rows = synthetic_rows(bridge, SCHEMA)
        self.assertEqual(rows['centroid_x'].null_count, 0)
        self.assertEqual(rows['centroid_x'].to_pylist(), rows['body_x'].to_pylist())
        self.assertEqual(rows['centroid_y'].to_pylist(), rows['body_y'].to_pylist())
        bare = pa.schema([f for f in SCHEMA if f.name not in ('centroid_x', 'centroid_y')])
        plain = synthetic_rows(bridge, bare)
        self.assertTrue(plain.schema.equals(bare))
        self.assertNotIn('centroid_x', plain.column_names)
        self.assertTrue(plain.equals(rows.drop_columns(['centroid_x', 'centroid_y'])))

    @unittest.skipUnless(DEMO.exists(), 'demo observations absent')
    def test_demo_schema_round_trip(self):
        schema = pq.read_schema(DEMO)
        rows = synthetic_rows(self.make_bridge(), schema, fps=FPS)
        self.assertTrue(rows.schema.equals(schema))
        self.assertEqual(rows['dataset_id'].null_count, 3)
        self.assertEqual(rows['shell_y'].null_count, 0)


# ============================================================
# PROPERTY TESTS
# ============================================================


class HermiteProperties(unittest.TestCase):

    @settings(deadline=None)
    @given(vec, vel, vec, vel, dts)
    def test_endpoints_exact(self, pa_, va, pb, vb, dt):
        np.testing.assert_array_equal(hermite(pa_, va, pb, vb, dt, 0.0), pa_)
        np.testing.assert_array_equal(hermite(pa_, va, pb, vb, dt, 1.0), pb)

    @settings(deadline=None)
    @given(vec, vec, dts, unit)
    def test_linear_reproduces_chord(self, pa_, pb, dt, t):
        c = (pb - pa_) / dt
        np.testing.assert_allclose(hermite(pa_, c, pb, c, dt, t), pa_ + t * (pb - pa_), rtol=0, atol=1e-9)

    @settings(deadline=None)
    @given(vec, vel, vec, vel, dts, unit)
    def test_overshoot_bound(self, pa_, va, pb, vb, dt, t):
        c = (pb - pa_) / dt
        deviation = np.linalg.norm(hermite(pa_, va, pb, vb, dt, t) - (pa_ + t * (pb - pa_)))
        bound = (4 / 27) * dt * (np.linalg.norm(va - c) + np.linalg.norm(vb - c))
        self.assertLessEqual(deviation, bound + 1e-9)

    @settings(deadline=None)
    @given(vec, vel, vec, vel, dts, unit)
    def test_reversal_symmetry(self, pa_, va, pb, vb, dt, t):
        forward = hermite(pa_, va, pb, vb, dt, t)
        backward = hermite(pb, -vb, pa_, -va, dt, 1.0 - t)
        np.testing.assert_allclose(backward, forward, rtol=0, atol=1e-9)


class DampedFillProperties(unittest.TestCase):

    @settings(deadline=None, max_examples=200)
    @given(vec, vel, vec, vel, st.integers(2, 400), st.integers(1, 30))
    def test_damped_bound_any_gap(self, pa_, va, pb, vb, dt, horizon):
        cfg = BridgeConfig(gate_model='sqrt', prediction_horizon_frames=horizon)
        a, b = motion_endpoint(100, pa_, va), motion_endpoint(100 + dt, pb, vb)
        bridge = fill_bridge(a, b, 't', 't/b001', cfg)
        self.assertEqual(len(bridge.frames), dt - 1)
        scale = 1.0 + float(np.abs(np.concatenate([pa_, pb])).max())
        self.assertLessEqual(chord_deviation(bridge, a, b).max(),
                             damped_bound(a, b, horizon) + 1e-9 * scale)
        if dt <= horizon:
            undamped = np.stack([hermite(pa_, va, pb, vb, dt, (f - a.frame) / dt) for f in bridge.frames])
            np.testing.assert_array_equal(bridge.positions, undamped)

    @settings(deadline=None, max_examples=100)
    @example(np.array([100.0, 100.0]), np.array([4.0, 3.0]), np.array([110.0, 95.0]),
             np.array([-3.0, 5.0]), 60, 3)
    @given(vec, vel, vec, vel, st.integers(2, 400), st.integers(1, 30))
    def test_linear_model_fill_is_undamped(self, pa_, va, pb, vb, dt, horizon):
        cfg = BridgeConfig(gate_model='linear', prediction_horizon_frames=horizon)
        a, b = motion_endpoint(100, pa_, va), motion_endpoint(100 + dt, pb, vb)
        bridge = fill_bridge(a, b, 't', 't/b001', cfg)
        undamped = np.stack([hermite(pa_, va, pb, vb, dt, (f - a.frame) / dt)
                             for f in bridge.frames])
        np.testing.assert_array_equal(bridge.positions, undamped)

    @settings(deadline=None)
    @given(vec, vel, vec, vel, st.integers(1, 400), st.integers(1, 30))
    def test_damped_tangents_shrink_towards_chord(self, pa_, va, pb, vb, dt, horizon):
        da, db = damped_tangents(pa_, va, pb, vb, dt, horizon)
        c = (pb - pa_) / dt
        lam = min(1.0, horizon / dt)
        tol = 1e-9 * (1.0 + float(np.abs(np.concatenate([va, vb, c])).max()))
        self.assertLessEqual(np.linalg.norm(da - c), lam * np.linalg.norm(va - c) + tol)
        self.assertLessEqual(np.linalg.norm(db - c), lam * np.linalg.norm(vb - c) + tol)


class AngleProperties(unittest.TestCase):

    @given(angle, angle)
    def test_endpoints_exact(self, ha, hb):
        self.assertEqual(interp_angle(ha, hb, 0.0), ha)
        self.assertLess(angdiff(interp_angle(ha, hb, 1.0), hb), 1e-12)

    @given(angle, angle, unit)
    def test_shortest_arc(self, ha, hb, t):
        delta = interp_angle(ha, hb, 1.0) - ha
        self.assertLessEqual(abs(delta), math.pi + 1e-12)
        self.assertGreater(delta, -math.pi - 1e-12)
        self.assertLess(abs(interp_angle(ha, hb, t) - ha), abs(delta) + 1e-12)

    @given(angle, angle, unit, st.integers(-5, 5), st.integers(-5, 5))
    def test_invariant_to_full_turns(self, ha, hb, t, ka, kb):
        base = interp_angle(ha, hb, t)
        shifted = interp_angle(ha + 2 * math.pi * ka, hb + 2 * math.pi * kb, t)
        self.assertLess(angdiff(shifted, base), 1e-9)

    @given(angle, angle, unit)
    def test_reversal_symmetry(self, ha, hb, t):
        if abs(angdiff(ha, hb) - math.pi) < 1e-9:
            return  # antipodal headings: both arcs are shortest; direction is a convention
        self.assertLess(angdiff(interp_angle(hb, ha, 1.0 - t), interp_angle(ha, hb, t)), 1e-9)

    @given(angle, st.sampled_from([0.0, 0.25, 1.0]))
    def test_missing_headings(self, h, t):
        self.assertEqual(interp_angle(h, math.nan, t), h)
        self.assertEqual(interp_angle(math.nan, h, t), h)
        self.assertTrue(math.isnan(interp_angle(math.nan, math.nan, t)))


offset_rows = st.lists(st.tuples(st.floats(-80, 80), st.floats(-80, 80)), min_size=4, max_size=4)


class PlaceNodesProperties(unittest.TestCase):

    @settings(deadline=None)
    @given(vec, angle, offset_rows)
    def test_preserves_inter_node_distances(self, pos, heading, rows):
        offsets = np.vstack([[0.0, 0.0], np.array(rows)])
        nodes = place_nodes(pos, heading, offsets)
        np.testing.assert_array_equal(nodes[0], pos)
        placed = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=2)
        reference = np.linalg.norm(offsets[:, None] - offsets[None, :], axis=2)
        np.testing.assert_allclose(placed, reference, rtol=1e-9, atol=1e-7)

    @given(vec, angle, st.integers(1, 4))
    def test_nan_offset_or_heading(self, pos, heading, row):
        offsets = DEFAULT_OFFSETS.copy()
        offsets[row] = np.nan
        nodes = place_nodes(pos, heading, offsets)
        self.assertTrue(np.isnan(nodes[row]).all())
        self.assertEqual(int(np.isnan(nodes).any(axis=1).sum()), 1)
        nodes = place_nodes(pos, math.nan, DEFAULT_OFFSETS)
        np.testing.assert_array_equal(nodes[0], pos)
        self.assertTrue(np.isnan(nodes[1:]).all())


if __name__ == '__main__':
    unittest.main()
