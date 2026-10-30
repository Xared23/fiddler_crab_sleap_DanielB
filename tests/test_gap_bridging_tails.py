#!/usr/bin/env python3

"""
Tests for gap-bridging tail recovery (WP-A of v003): adoption of rejected rows
just before a trajectory's START and just after its END.

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
from typing import Dict, List, Optional, Set, Tuple
import unittest

# ----- numerics -----
from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

# ----- local modules -----
from gap_bridging import chains, geometry, tails  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging_fixtures import Crab, make_observations, straight_crab  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

CFG = BridgeConfig()
LOW = 'low_body_score'
REASONS = (LOW,)
KW = dict(radius_px=8.0, max_frames=15, reasons=REASONS, exclusion_px=5.0, horizon=3)

# ============================================================
# HELPERS
# ============================================================


def scene(crabs: List[Crab], drop=None, reject=None):
    """Observations, trajectories (one per tracklet, no links) and endpoints."""
    obs = make_observations(crabs, drop_frames=drop, reject_frames=reject)
    ends = geometry.endpoints(obs, CFG)
    trajectories = chains.build_trajectories(sorted(ends), [], ends)
    return obs, trajectories, ends


def run(obs, trajectories, ends, used: Optional[Set[str]] = None, **overrides):
    """recover_tails with the default keywords, ``overrides`` applied."""
    kw = {**KW, **overrides}
    return tails.recover_tails(obs, trajectories, ends, set() if used is None else used, **kw)


def rows_by_detection(obs: pa.Table) -> Dict[str, dict]:
    """``detection_id -> row`` (as a dict) for every row of ``obs``."""
    return {r['detection_id']: r for r in obs.to_pylist()}


def det_at(obs: pa.Table, frame: int, reason: Optional[str]) -> List[str]:
    """Detection ids at ``frame`` with ``rejection_reason == reason`` (None: accepted)."""
    rows = obs.filter(pc.equal(obs['frame'], frame)).to_pylist()
    return [r['detection_id'] for r in rows if r['rejection_reason'] == reason]


def only_trajectory(trajectories) -> str:
    """The id of the single trajectory of a one-crab scene."""
    assert len(trajectories) == 1
    return next(iter(trajectories))


def span(trajectories, ends, traj: str) -> Tuple[int, int]:
    """Observed ``(start, end)`` frames of a trajectory."""
    members = trajectories[traj]
    return ends[members[0]][0].frame, ends[members[-1]][1].frame


# ============================================================
# EXAMPLES
# ============================================================


class TailExampleTests(unittest.TestCase):
    """Hand-built scenes with planted rejected rows at trajectory ends."""

    def test_adopts_one_frame_before_start_and_after_end(self):
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (1.5, -0.5))
        obs, trajs, ends = scene([crab], reject={1: {10: LOW, 30: LOW}})
        traj = only_trajectory(trajs)
        self.assertEqual(span(trajs, ends, traj), (11, 29))
        used: Set[str] = set()
        out = run(obs, trajs, ends, used)
        expected = {('synthetic', 10, traj): det_at(obs, 10, LOW)[0],
                    ('synthetic', 30, traj): det_at(obs, 30, LOW)[0]}
        self.assertEqual(out, expected)
        self.assertEqual(used, set(expected.values()))
        back, fwd = tails.tail_bridge_ids(traj)
        self.assertEqual((back, fwd), (f"{traj}/tail_back", f"{traj}/tail_fwd"))
        self.assertEqual(tails.tail_bridge_of(out, trajs, ends),
                         {('synthetic', 10, traj): back, ('synthetic', 30, traj): fwd})
        self.assertEqual(tails.tail_lengths(out, trajs, ends), {back: 1, fwd: 1})
        self.assertEqual(tails.tail_info(out, trajs, ends, CFG),
                         {back: (1, 'high'), fwd: (1, 'high')})

    def test_stops_at_first_miss(self):
        # Planted at start-1 and start-3, start-2 absent: only start-1 adopted.
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (1.0, 0.0))
        obs, trajs, ends = scene([crab], drop={1: {11}}, reject={1: {10: LOW, 12: LOW}})
        traj = only_trajectory(trajs)
        self.assertEqual(span(trajs, ends, traj)[0], 13)
        out = run(obs, trajs, ends)
        self.assertEqual(list(out), [('synthetic', 12, traj)])

    def test_multi_frame_tail_follows_motion_and_respects_max_frames(self):
        crab = straight_crab(1, 10, 30, (500.0, 500.0), (5.0, 0.0))
        reject = {1: {f: LOW for f in (10, 11, 12, 13, 14, 35, 36, 37, 38, 39)}}
        obs, trajs, ends = scene([crab], reject=reject)
        traj = only_trajectory(trajs)
        out = run(obs, trajs, ends)
        self.assertEqual(sorted(k[1] for k in out), [10, 11, 12, 13, 14, 35, 36, 37, 38, 39])
        back, fwd = tails.tail_bridge_ids(traj)
        self.assertEqual(tails.tail_lengths(out, trajs, ends), {back: 5, fwd: 5})
        capped = run(obs, trajs, ends, max_frames=3)
        self.assertEqual(sorted(k[1] for k in capped), [12, 13, 14, 35, 36, 37])

    def test_disabled_by_zero_frames_or_radius(self):
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (1.0, 0.0))
        obs, trajs, ends = scene([crab], reject={1: {10: LOW, 30: LOW}})
        self.assertEqual(run(obs, trajs, ends, max_frames=0), {})
        self.assertEqual(run(obs, trajs, ends, radius_px=0.0), {})
        self.assertEqual(run(obs, trajs, ends, reasons=()), {})

    def test_missing_position_and_other_reasons_ignored(self):
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (1.0, 0.0))
        obs, trajs, ends = scene([crab], reject={1: {10: 'missing_position', 30: 'invisible'}})
        self.assertEqual(run(obs, trajs, ends), {})
        # 'missing_position' has no body position: never adoptable even when listed.
        out = run(obs, trajs, ends, reasons=(LOW, 'missing_position'))
        self.assertEqual(out, {})
        out = run(obs, trajs, ends, reasons=(LOW, 'invisible'))
        self.assertEqual([k[1] for k in out], [30])

    def test_exclusion_near_other_trajectory(self):
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (0.0, 0.0))
        for offset, adopted in ((3.0, False), (5.0, False), (6.5, True)):
            other = straight_crab(2, 5, 30, (500.0 + offset, 500.0), (0.0, 0.0))
            obs, trajs, ends = scene([crab, other], reject={1: {10: LOW}})
            out = run(obs, trajs, ends)
            self.assertEqual(bool(out), adopted, offset)

    def test_used_never_adopted_and_next_candidate_taken(self):
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (0.0, 0.0))
        near = straight_crab(2, 10, 1, (502.0, 500.0), (0.0, 0.0))
        far = straight_crab(3, 10, 1, (507.0, 500.0), (0.0, 0.0))
        obs, trajs, ends = scene([crab, near, far], reject={1: {10: LOW}, 2: {10: LOW},
                                                            3: {10: LOW}})
        traj = only_trajectory(trajs)
        rows = rows_by_detection(obs)
        planted = sorted(det_at(obs, 10, LOW), key=lambda d: abs(rows[d]['body_x'] - 500.0))
        out = run(obs, trajs, ends)
        self.assertEqual(out, {('synthetic', 10, traj): planted[0]})
        # A used row is never adopted and counts as occupied by another trajectory:
        # the row 2 px from it is excluded, the row 7 px from it is adopted.
        out = run(obs, trajs, ends, used={planted[0]})
        self.assertEqual(out, {('synthetic', 10, traj): planted[2]})
        out = run(obs, trajs, ends, used={planted[1]})
        self.assertEqual(out, {})

    def test_detection_adopted_once_between_two_trajectories(self):
        # Crab 1 ends at 19, crab 2 starts at 21; one planted row at 20 between them.
        a = straight_crab(1, 10, 10, (500.0, 500.0), (0.0, 0.0))
        b = straight_crab(2, 21, 10, (502.0, 500.0), (0.0, 0.0))
        planted = straight_crab(3, 20, 1, (501.0, 500.0), (0.0, 0.0))
        obs, trajs, ends = scene([a, b, planted], reject={3: {20: LOW}})
        out = run(obs, trajs, ends)
        self.assertEqual(len(out), 1)
        first = min(trajs, key=lambda t: span(trajs, ends, t)[0])
        self.assertEqual(list(out), [('synthetic', 20, first)])

    def test_tail_bridge_of_rejects_key_inside_span(self):
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (1.0, 0.0))
        obs, trajs, ends = scene([crab])
        traj = only_trajectory(trajs)
        with self.assertRaises(ValueError):
            tails.tail_bridge_of({('synthetic', 15, traj): 'x'}, trajs, ends)

    def test_invalid_arguments(self):
        crab = straight_crab(1, 10, 21, (500.0, 500.0), (1.0, 0.0))
        obs, trajs, ends = scene([crab])
        with self.assertRaises(ValueError):
            run(obs, trajs, ends, horizon=0)
        with self.assertRaises(ValueError):
            run(obs, trajs, ends, radius_px=-1.0)
        with self.assertRaises(ValueError):
            tails.tail_bridge_ids('')


# ============================================================
# PROPERTIES
# ============================================================


@st.composite
def scenes(draw):
    """Random straight crabs, some close together, with rejected rows at random frames."""
    n = draw(st.integers(1, 4))
    crabs, reject = [], {}
    for c in range(1, n + 1):
        start = draw(st.integers(0, 10))
        length = draw(st.integers(2, 20))
        origin = (400.0 + draw(st.floats(0.0, 60.0)), 400.0 + draw(st.floats(0.0, 20.0)))
        velocity = (draw(st.floats(-3.0, 3.0)), draw(st.floats(-3.0, 3.0)))
        crabs.append(straight_crab(c, start, length, origin, velocity))
        frames = draw(st.sets(st.integers(start, start + length - 1), max_size=length - 1))
        reasons = st.sampled_from([LOW, LOW, 'missing_position', 'invisible'])
        reject[c] = {f: draw(reasons) for f in frames}
    return crabs, reject


def anchored_ok(key, tails_out, positions, trajs, ends, radius, horizon) -> bool:
    """Adopted row lies within ``radius`` of the prediction from its anchor frame."""
    video, frame, traj = key
    members = trajs[traj]
    start, end = ends[members[0]][0], ends[members[-1]][1]
    anchor_ep, step = (start, -1) if frame < start.frame else (end, +1)
    anchor_frame = frame - step
    if anchor_frame == anchor_ep.frame:
        anchor = np.asarray(anchor_ep.pos, float)
    else:
        anchor = np.asarray(positions[tails_out[(video, anchor_frame, traj)]], float)
    vel = np.asarray(anchor_ep.vel, float) if anchor_ep.vel_known else np.zeros(2)
    pred = anchor + step * vel * min(1, horizon)
    x, y = positions[tails_out[key]]
    return math.hypot(x - pred[0], y - pred[1]) <= radius + 1e-9


class TailPropertyTests(unittest.TestCase):
    """Invariants of recover_tails on random scenes."""

    @settings(deadline=None, max_examples=120)
    @given(scenes(), st.integers(0, 6), st.floats(0.5, 10.0), st.floats(0.0, 8.0),
           st.integers(1, 4), st.data())
    def test_tail_invariants(self, drawn, max_frames, radius, exclusion, horizon, data):
        crabs, reject = drawn
        obs, trajs, ends = scene(crabs, reject=reject)
        rows = rows_by_detection(obs)
        rejected = sorted(d for d, r in rows.items() if r['rejection_reason'] is not None)
        used0 = set(data.draw(st.lists(st.sampled_from(rejected), unique=True), 'used')
                    if rejected else [])
        kw = dict(max_frames=max_frames, radius_px=radius, exclusion_px=exclusion,
                  horizon=horizon)
        used = set(used0)
        out = run(obs, trajs, ends, used, **kw)
        positions = {d: (r['body_x'], r['body_y']) for d, r in rows.items()}
        # Determinism, `used` bookkeeping, at most once per detection.
        self.assertEqual(out, run(obs, trajs, ends, set(used0), **kw))
        self.assertEqual(len(set(out.values())), len(out))
        self.assertTrue(used0.isdisjoint(out.values()))
        self.assertEqual(used, used0 | set(out.values()))
        owner = {}
        for d, r in rows.items():
            if r['accepted_for_tracking']:
                owner[d] = chains.tracklet_to_trajectory(trajs)[r['tracklet_id']]
        for key, det in out.items():
            video, frame, traj = key
            first, last = span(trajs, ends, traj)
            self.assertFalse(first <= frame <= last, key)
            self.assertIn(rows[det]['rejection_reason'], REASONS)
            self.assertEqual(rows[det]['frame'], frame)
            self.assertTrue(anchored_ok(key, out, positions, trajs, ends, radius, horizon), key)
            # Contiguous from the endpoint and no longer than max_frames.
            self.assertLessEqual(max(first - frame, frame - last), max_frames)
            # Exclusion: far from every other trajectory's accepted, used or tail row.
            occupied = [(owner[d], positions[d]) for d in owner if rows[d]['frame'] == frame]
            occupied += [(None, positions[d]) for d in used0
                         if rows[d]['frame'] == frame and rows[d]['body_x'] is not None]
            occupied += [(k[2], positions[d]) for k, d in out.items()
                         if k[1] == frame and k != key]
            for other, (x, y) in occupied:
                if other != traj:
                    px, py = positions[det]
                    self.assertGreater(math.hypot(px - x, py - y), exclusion, key)
        lengths = tails.tail_lengths(out, trajs, ends)
        self.assertEqual(sum(lengths.values()), len(out))
        self.assertTrue(all(1 <= n <= max(max_frames, 1) for n in lengths.values()))


# ============================================================
# __main__
# ============================================================

if __name__ == "__main__":
    unittest.main()
