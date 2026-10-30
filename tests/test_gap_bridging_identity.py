"""Tests for gap-bridging v003 WP-B: the spurious-identity filter."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from hypothesis import given, settings, strategies as st
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from gap_bridging import identity, pipeline, tables  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from pipeline_common import sha256  # noqa: E402

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
CFG = BridgeConfig()
SCHEMA = pa.schema([('video_id', pa.string()), ('frame', pa.int64()),
                    ('trajectory_id', pa.string()), ('source', pa.string()),
                    ('instance_score', pa.float64())])
Row = tuple  # (video_id, frame, trajectory_id, source, instance_score)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def obs_table(rows: list[Row]) -> pa.Table:
    return pa.Table.from_pylist([dict(zip(SCHEMA.names, r)) for r in rows], schema=SCHEMA)


def trajectory(traj: str, sources: list[str], scores: list[float | None], start: int = 0,
               video: str = 'v') -> list[Row]:
    return [(video, start + i, traj, s, sc) for i, (s, sc) in enumerate(zip(sources, scores))]


def classify(rows: list[Row], min_frames: int = 2, min_score: float = 0.6) -> dict[str, str]:
    return identity.classify_trajectories(obs_table(rows), min_frames=min_frames,
                                          min_score=min_score)


def traj_table(spans: list[tuple[str, str, int, int]]) -> pa.Table:
    return pa.Table.from_pylist([dict(trajectory_id=t, video_id=v, start_frame=s, end_frame=e)
                                 for t, v, s, e in spans])


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
class ClassifyTests(unittest.TestCase):
    def test_min_frames_boundary(self):
        rows = (trajectory('one', ['observed'], [0.1])
                + trajectory('two', ['observed'] * 2, [0.1, 0.1], start=10))
        self.assertEqual(classify(rows), {'one': 'spurious', 'two': 'identity'})
        self.assertEqual(classify(rows, min_frames=3), {'one': 'spurious', 'two': 'spurious'})
        self.assertEqual(classify(rows, min_frames=1), {'one': 'identity', 'two': 'identity'})

    def test_min_score_boundary(self):
        for score, expected in [(0.59, 'spurious'), (0.6, 'identity'), (0.61, 'identity')]:
            with self.subTest(score=score):
                self.assertEqual(classify(trajectory('t', ['observed'], [score])), {'t': expected})

    def test_median_over_real_rows_only(self):
        # Interpolated rows never count as frames, and their (null) scores are ignored.
        rows = trajectory('t', ['observed', 'interpolated', 'interpolated'], [0.3, None, None])
        self.assertEqual(classify(rows), {'t': 'spurious'})
        rows = trajectory('t', ['observed', 'interpolated', 'interpolated'], [0.9, None, None])
        self.assertEqual(classify(rows), {'t': 'identity'})
        # Median, not mean: 0.1, 0.65, 0.7 has median 0.65 >= 0.6.
        rows = trajectory('t', ['observed', 'recovered', 'observed'], [0.1, 0.65, 0.7])
        self.assertEqual(classify(rows, min_frames=4), {'t': 'identity'})
        rows = trajectory('t', ['observed', 'recovered', 'observed'], [0.1, 0.2, 0.9])
        self.assertEqual(classify(rows, min_frames=4), {'t': 'spurious'})

    def test_recovered_tail_counts_as_real(self):
        rows = trajectory('t', ['observed', 'recovered_tail'], [0.2, 0.3])
        self.assertEqual(classify(rows), {'t': 'identity'})
        rows = trajectory('t', ['observed', 'recovered'], [0.2, 0.3])
        self.assertEqual(classify(rows), {'t': 'identity'})

    def test_interpolated_only_is_spurious(self):
        rows = trajectory('t', ['interpolated'] * 5, [None] * 5)
        self.assertEqual(classify(rows), {'t': 'spurious'})
        self.assertEqual(classify(rows, min_score=0.0), {'t': 'spurious'})

    def test_unassigned_rows_ignored(self):
        rows = trajectory('t', ['observed'] * 3, [0.9] * 3) + [('v', 7, None, 'observed', 0.01)]
        self.assertEqual(classify(rows), {'t': 'identity'})

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            classify(trajectory('t', ['observed'], [0.9]), min_frames=0)
        with self.assertRaises(ValueError):
            identity.classify_trajectories(obs_table([]).drop_columns(['instance_score']),
                                           min_frames=2, min_score=0.6)


# Strategy: a list of trajectories, each a list of (source, score) rows.
SOURCE = st.sampled_from(['observed', 'recovered', 'recovered_tail', 'interpolated'])
SCORE = st.one_of(st.none(), st.floats(0.0, 1.0))
TRAJ_ROWS = st.lists(st.tuples(SOURCE, SCORE), min_size=1, max_size=5)


def build(trajs: list[list[tuple[str, float | None]]], prefix: str = 't') -> list[Row]:
    rows: list[Row] = []
    for k, items in enumerate(trajs):
        rows += trajectory(f"{prefix}{k}", [s for s, _ in items], [sc for _, sc in items],
                           start=10 * k)
    return rows


class ClassifyPropertyTests(unittest.TestCase):
    @settings(max_examples=150, deadline=None)
    @given(trajs=st.lists(TRAJ_ROWS, min_size=1, max_size=6),
           others=st.lists(TRAJ_ROWS, max_size=4),
           seed=st.randoms(use_true_random=False), min_frames=st.integers(1, 4),
           min_score=st.floats(0.0, 1.0))
    def test_depends_only_on_own_rows(self, trajs, others, seed, min_frames, min_score):
        rows = build(trajs)
        base = classify(rows, min_frames, min_score)
        self.assertEqual(base, classify(rows, min_frames, min_score), "deterministic")
        shuffled = list(rows) + build(others, prefix='other')
        seed.shuffle(shuffled)
        mixed = classify(shuffled, min_frames, min_score)
        self.assertEqual({k: mixed[k] for k in base}, base)
        for k, items in enumerate(trajs):
            alone = classify(build([items]), min_frames, min_score)['t0']
            self.assertEqual(base[f"t{k}"], alone)

    @settings(max_examples=150, deadline=None)
    @given(items=TRAJ_ROWS, min_frames=st.integers(1, 4), min_score=st.floats(0.0, 1.0))
    def test_definition(self, items, min_frames, min_score):
        real = [sc for s, sc in items if s != 'interpolated']
        scores = sorted(sc for sc in real if sc is not None)
        median = None
        if scores:
            n = len(scores)
            median = scores[n // 2] if n % 2 else (scores[n // 2 - 1] + scores[n // 2]) / 2
        spurious = len(real) < min_frames and (median is None or median < min_score)
        got = classify(build([items]), min_frames, min_score)['t0']
        self.assertEqual(got, 'spurious' if spurious else 'identity')


# --------------------------------------------------------------------------- #
# Status column and summaries
# --------------------------------------------------------------------------- #
class SummaryTests(unittest.TestCase):
    def test_status_column(self):
        table = obs_table(trajectory('a', ['observed'], [0.1])
                          + [('v', 3, None, 'observed', 0.2)]
                          + trajectory('b', ['observed'] * 2, [0.9, 0.9], start=5))
        column = identity.status_column(table, {'a': 'spurious', 'b': 'identity'})
        self.assertEqual(column.type, pa.string())
        self.assertEqual(column.to_pylist(), ['spurious', None, 'identity', 'identity'])
        with self.assertRaises(ValueError):
            identity.status_column(table, {'a': 'spurious'})
        with self.assertRaises(ValueError):
            identity.status_column(table, {'a': 'spurious', 'b': 'maybe'})

    def test_identity_starts_arithmetic(self):
        # Video v starts at frame 100, video w at frame 0 (interpolated rows never define it).
        obs = obs_table([('v', 100, None, 'observed', 0.5), ('v', 150, 'v1', 'observed', 0.5),
                         ('w', 0, 'w0', 'observed', 0.5), ('w', 0, None, 'interpolated', None)])
        traj = traj_table([('v0', 'v', 100, 120), ('v1', 'v', 101, 101), ('v2', 'v', 140, 160),
                           ('w0', 'w', 0, 3), ('w1', 'w', 5, 5), ('w2', 'w', 9, 30)])
        status = {'v0': 'identity', 'v1': 'spurious', 'v2': 'identity',
                  'w0': 'spurious', 'w1': 'spurious', 'w2': 'identity'}
        self.assertEqual(identity.identity_starts(traj, obs, status), {'all': 4, 'identity': 2})
        self.assertEqual(identity.identity_starts(traj, obs, None), {'all': 4, 'identity': 4})
        with self.assertRaises(ValueError):
            identity.identity_starts(traj, obs, {'v0': 'identity'})

    def test_filtered_lengths(self):
        traj = traj_table([('a', 'v', 0, 0), ('b', 'v', 0, 9), ('c', 'v', 3, 3), ('d', 'v', 5, 7)])
        status = {'a': 'spurious', 'b': 'identity', 'c': 'identity', 'd': 'identity'}
        self.assertEqual(identity.filtered_lengths(traj, status),
                         {'n': 3, 'mean': 14 / 3, 'median': 3.0, 'units_len1': 1})
        self.assertEqual(identity.filtered_lengths(traj, None),
                         {'n': 4, 'mean': 15 / 4, 'median': 2.0, 'units_len1': 2})
        none = {k: 'spurious' for k in status}
        self.assertEqual(identity.filtered_lengths(traj, none),
                         {'n': 0, 'mean': None, 'median': None, 'units_len1': 0})


# --------------------------------------------------------------------------- #
# Demo run
# --------------------------------------------------------------------------- #
class DemoIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / 'run'
        pipeline.run(DEMO, cls.out, BridgeConfig(max_gap=15))
        cls.obs = pq.read_table(cls.out / 'observations.parquet')
        cls.traj = pq.read_table(cls.out / 'trajectories.parquet')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_classification_never_changes_outputs(self):
        before = {n: sha256(self.out / n) for n in tables.OUTPUT_NAMES}
        status = identity.classify_trajectories(self.obs, min_frames=CFG.min_identity_frames,
                                                min_score=CFG.min_identity_score)
        identity.status_column(self.obs, status)
        identity.status_column(self.traj, status)
        self.assertEqual(before, {n: sha256(self.out / n) for n in tables.OUTPUT_NAMES})
        self.assertEqual(set(status), set(self.traj['trajectory_id'].to_pylist()))

    def test_demo_counts(self):
        status = identity.classify_trajectories(self.obs, min_frames=CFG.min_identity_frames,
                                                min_score=CFG.min_identity_score)
        n_spurious = sum(v == 'spurious' for v in status.values())
        lengths = identity.filtered_lengths(self.traj, None)
        filtered = identity.filtered_lengths(self.traj, status)
        starts = identity.identity_starts(self.traj, self.obs, status)
        # Only short units can be spurious; the filtered set loses exactly those.
        self.assertEqual(filtered['n'], lengths['n'] - n_spurious)
        self.assertLess(filtered['units_len1'], lengths['units_len1'])
        self.assertGreater(n_spurious, 0)
        self.assertLessEqual(starts['identity'], starts['all'])
        self.assertEqual(starts['all'], identity.identity_starts(self.traj, self.obs, None)['all'])
        t = self.traj.to_pydict()
        for k, traj in enumerate(t['trajectory_id']):
            real = t['n_observed'][k] + t['n_recovered'][k] + t['n_recovered_tail'][k]
            if status[traj] == 'spurious':
                self.assertLess(real, CFG.min_identity_frames)

if __name__ == '__main__':
    unittest.main()
