#!/usr/bin/env python3

"""
Tests for the gap-bridging hold-out evaluation (planting, scoring, end to end).

Sam
10/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import csv
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Tuple
import unittest

# ----- numerics -----
from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

# ----- local modules -----
from gap_bridging import holdout, tables  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging_fixtures import Crab, make_observations, straight_crab  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
N_FRAMES = 60
GAPS = (1, 3, 5)

# ============================================================
# HELPERS
# ============================================================


def far_crabs(n: int, n_frames: int = N_FRAMES) -> List[Crab]:
    """``n`` slow straight crabs on a 600-px grid (no cross-crab candidate passes a gate)."""
    crabs = []
    for c in range(1, n + 1):
        angle = 0.7 * c
        velocity = (0.5 * math.cos(angle), 0.5 * math.sin(angle))
        origin = (300.0 + 600.0 * ((c - 1) % 5), 300.0 + 600.0 * ((c - 1) // 5))
        crabs.append(straight_crab(c, 0, n_frames, origin, velocity))
    return crabs


def write(table: pa.Table, directory: Path, name: str = 'observations.parquet') -> Path:
    """Write ``table`` to ``directory / name`` and return the path."""
    path = directory / name
    pq.write_table(table, path)
    return path


def links_table(rows: List[Tuple[str, str, str]]) -> pa.Table:
    """A :data:`tables.LINKS_SCHEMA` table from ``(tracklet_a, tracklet_b, decision)``."""
    columns: Dict[str, list] = {name: [] for name in tables.LINKS_SCHEMA.names}
    for a, b, decision in rows:
        for name, value in (('video_id', 'v'), ('tracklet_a', a), ('tracklet_b', b),
                            ('gap_frames', 1), ('cost', 0.1), ('runner_up_cost', math.nan),
                            ('decision', decision), ('reason', ''), ('n_chain_ignored', 0), ('stage', 1)):
            columns[name].append(value)
    return pa.Table.from_pydict(columns, schema=tables.LINKS_SCHEMA)


def truth_table(rows: List[Tuple[str, str, int]]) -> pa.Table:
    """A truth table from ``(left_id, right_id, gap)``."""
    return pa.Table.from_pydict({
        'left_id': [r[0] for r in rows], 'right_id': [r[1] for r in rows],
        'original_id': [r[0] for r in rows], 'gap_frames': [r[2] for r in rows],
        'deleted_first_frame': [10] * len(rows),
        'deleted_last_frame': [9 + r[2] for r in rows]}, schema=holdout.TRUTH_SCHEMA)


def score_dict(scores: pa.Table) -> Dict[int, dict]:
    """Scores rows keyed by ``gap_frames``."""
    return {row['gap_frames']: row for row in scores.to_pylist()}


BASE = make_observations(far_crabs(12))

# ============================================================
# PLANTING
# ============================================================


class PlantTests(unittest.TestCase):
    """plant_gaps on 12 synthetic 60-frame tracklets."""

    @classmethod
    def setUpClass(cls):
        cls.planted, cls.truth = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7)

    def test_truth_shape_and_frames(self):
        self.assertEqual(self.truth.schema, holdout.TRUTH_SCHEMA)
        self.assertEqual(self.truth.num_rows, 6)
        self.assertEqual(sorted(self.truth['gap_frames'].to_pylist()), [1, 1, 3, 3, 5, 5])
        for row in self.truth.to_pylist():
            self.assertEqual(row['left_id'], row['original_id'])
            self.assertEqual(row['deleted_last_frame'] - row['deleted_first_frame'] + 1,
                             row['gap_frames'])
            # margin 3 on a 0..59 tracklet
            self.assertGreaterEqual(row['deleted_first_frame'], 3)
            self.assertLessEqual(row['deleted_last_frame'], N_FRAMES - 1 - 3)

    def test_rows_removed_and_right_half_relabelled(self):
        self.assertEqual(BASE.num_rows - self.planted.num_rows,
                         sum(self.truth['gap_frames'].to_pylist()))
        local_max = pc.max(BASE['tracklet_local_id']).as_py()
        for n, row in enumerate(self.truth.to_pylist(), 1):
            self.assertEqual(row['right_id'], f"{row['original_id']}/holdout{n:03d}")
            before = BASE.filter(pc.equal(BASE['tracklet_id'], row['original_id']))
            left = self.planted.filter(pc.equal(self.planted['tracklet_id'], row['left_id']))
            right = self.planted.filter(pc.equal(self.planted['tracklet_id'], row['right_id']))
            self.assertEqual(before.num_rows - left.num_rows - right.num_rows,
                             row['gap_frames'])
            self.assertEqual(pc.max(left['frame']).as_py(), row['deleted_first_frame'] - 1)
            self.assertEqual(pc.min(right['frame']).as_py(), row['deleted_last_frame'] + 1)
            self.assertEqual(set(right['tracklet_local_id'].to_pylist()), {local_max + n})
            reasons = right['incoming_reason'].to_pylist()
            self.assertEqual(reasons[0], 'no_candidate')
            self.assertEqual(set(reasons[1:]), {'linked'})
            # Left half: identical to the original rows before the gap.
            original_left = before.filter(pc.less(before['frame'], row['deleted_first_frame']))
            self.assertTrue(left.equals(original_left))
            # Right half: identical apart from the three relabelled columns.
            original_right = before.filter(pc.greater(before['frame'],
                                                      row['deleted_last_frame']))
            other = [c for c in BASE.schema.names
                     if c not in ('tracklet_id', 'tracklet_local_id', 'incoming_reason')]
            self.assertTrue(right.select(other).equals(original_right.select(other)))

    def test_other_rows_unchanged(self):
        touched = pa.array(self.truth['original_id'].to_pylist()
                           + self.truth['right_id'].to_pylist())
        mask_base = pc.invert(pc.fill_null(pc.is_in(BASE['tracklet_id'], touched), False))
        mask_planted = pc.invert(pc.fill_null(pc.is_in(self.planted['tracklet_id'], touched),
                                              False))
        self.assertTrue(BASE.filter(mask_base).equals(self.planted.filter(mask_planted)))
        self.assertTrue(self.planted.schema.equals(BASE.schema, check_metadata=True))

    def test_deterministic_in_seed(self):
        again, truth = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7)
        self.assertTrue(again.equals(self.planted))
        self.assertTrue(truth.equals(self.truth))

    def test_too_few_tracklets_raises(self):
        with self.assertRaises(ValueError):
            holdout.plant_gaps(BASE, GAPS, per_gap=5)
        with self.assertRaises(ValueError):
            holdout.plant_gaps(BASE, (1,), per_gap=1, min_tracklet_len=N_FRAMES + 1)
        with self.assertRaises(ValueError):    # g + 2 * margin > tracklet length
            holdout.plant_gaps(BASE, (N_FRAMES - 5,), per_gap=1)

    def test_tracklet_with_real_gap_is_not_eligible(self):
        # Crab 1 misses frame 30; merge its two runs into one tracklet id with a hole.
        table = make_observations(far_crabs(2), drop_frames={1: [30]})
        ids = table['tracklet_id'].to_pylist()
        first, second = ids[0], ids[table['frame'].to_pylist().index(31)]
        holed = table.set_column(table.schema.get_field_index('tracklet_id'), 'tracklet_id',
                                 pa.array([first if v == second else v for v in ids]))
        contiguous = ids[1]
        for seed in range(5):
            _, truth = holdout.plant_gaps(holed, (2,), per_gap=1, min_tracklet_len=20,
                                          seed=seed)
            self.assertEqual(truth['original_id'].to_pylist(), [contiguous])
        with self.assertRaises(ValueError):
            holdout.plant_gaps(holed, (2,), per_gap=2, min_tracklet_len=20)

    @settings(deadline=None, max_examples=40)
    @given(per_gap=st.integers(1, 4),
           gaps=st.lists(st.integers(1, 20), min_size=1, max_size=3),
           seed=st.integers(0, 2**31))
    def test_property_counts_and_no_reuse(self, per_gap, gaps, seed):
        if per_gap * len(gaps) > 12:
            with self.assertRaises(ValueError):
                holdout.plant_gaps(BASE, gaps, per_gap=per_gap, seed=seed)
            return
        planted, truth = holdout.plant_gaps(BASE, gaps, per_gap=per_gap, seed=seed)
        self.assertEqual(truth.num_rows, per_gap * len(gaps))
        originals = truth['original_id'].to_pylist()
        self.assertEqual(len(set(originals)), len(originals))
        self.assertEqual(BASE.num_rows - planted.num_rows, per_gap * sum(gaps))
        self.assertEqual(sorted(truth['gap_frames'].to_pylist()),
                         sorted(g for g in gaps for _ in range(per_gap)))
        n_ids = len(set(planted['tracklet_id'].drop_null().to_pylist()))
        self.assertEqual(n_ids, 12 + truth.num_rows)


# ============================================================
# SCORING
# ============================================================


class ScoreTests(unittest.TestCase):
    """score_links arithmetic on hand-built tables."""

    TRUTH = truth_table([('A', 'A/h1', 1), ('B', 'B/h2', 1), ('C', 'C/h3', 3),
                         ('D', 'D/h4', 3), ('E', 'E/h5', 5)])

    def test_recovered_false_missed(self):
        links = links_table([
            ('A', 'A/h1', 'linked'),           # recovered
            ('B', 'X', 'linked'),              # left linked elsewhere -> false
            ('Y', 'C/h3', 'linked'),           # right linked from elsewhere -> false
            ('D', 'D/h4', 'refused_margin'),   # not linked -> missed
            ('U', 'V', 'linked'),              # unplanted, ignored
        ])
        scores = holdout.score_links(links, self.TRUTH)
        self.assertEqual(scores.schema, holdout.SCORES_SCHEMA)
        self.assertEqual(scores['gap_frames'].to_pylist(), [1, 3, 5, -1])
        by_gap = score_dict(scores)
        self.assertEqual([by_gap[1][k] for k in ('planted', 'recovered', 'false_link', 'missed')],
                         [2, 1, 1, 0])
        self.assertEqual([by_gap[3][k] for k in ('planted', 'recovered', 'false_link', 'missed')],
                         [2, 0, 1, 1])
        self.assertEqual([by_gap[5][k] for k in ('planted', 'recovered', 'false_link', 'missed')],
                         [1, 0, 0, 1])
        self.assertEqual(by_gap[1]['recall'], 0.5)
        self.assertEqual(by_gap[1]['precision'], 0.5)
        self.assertEqual(by_gap[3]['recall'], 0.0)
        self.assertEqual(by_gap[3]['precision'], 0.0)
        self.assertTrue(math.isnan(by_gap[5]['precision']))
        self.assertEqual(by_gap[5]['recall'], 0.0)
        total = by_gap[-1]
        self.assertEqual([total[k] for k in ('planted', 'recovered', 'false_link', 'missed')],
                         [5, 1, 2, 2])
        self.assertAlmostEqual(total['recall'], 0.2)
        self.assertAlmostEqual(total['precision'], 1 / 3)

    def test_nothing_linked(self):
        scores = holdout.score_links(links_table([]), self.TRUTH)
        for row in scores.to_pylist():
            self.assertEqual(row['recovered'] + row['false_link'], 0)
            self.assertEqual(row['missed'], row['planted'])
            self.assertTrue(math.isnan(row['precision']))
            self.assertEqual(row['recall'], 0.0)

    @settings(deadline=None, max_examples=60)
    @given(outcomes=st.lists(st.sampled_from(['recovered', 'false_left', 'false_right',
                                              'missed']), min_size=1, max_size=12),
           gaps=st.lists(st.integers(1, 4), min_size=12, max_size=12))
    def test_property_counts_partition(self, outcomes, gaps):
        truth_rows, link_rows = [], []
        for i, outcome in enumerate(outcomes):
            left, right = f'T{i}', f'T{i}/h'
            truth_rows.append((left, right, gaps[i]))
            if outcome == 'recovered':
                link_rows.append((left, right, 'linked'))
            elif outcome == 'false_left':
                link_rows.append((left, f'Z{i}', 'linked'))
            elif outcome == 'false_right':
                link_rows.append((f'Z{i}', right, 'linked'))
        total = score_dict(holdout.score_links(links_table(link_rows),
                                               truth_table(truth_rows)))[-1]
        self.assertEqual(total['planted'], len(outcomes))
        self.assertEqual(total['recovered'], outcomes.count('recovered'))
        self.assertEqual(total['false_link'],
                         outcomes.count('false_left') + outcomes.count('false_right'))
        self.assertEqual(total['missed'], outcomes.count('missed'))


# ============================================================
# END TO END
# ============================================================


class EndToEndTests(unittest.TestCase):
    """run_holdout on far-apart straight crabs: 6 planted, 2 left intact."""

    def run_case(self, tmp: Path, max_gap: int) -> Tuple[dict, Path]:
        path = write(make_observations(far_crabs(8)), tmp)
        out = tmp / f'holdout{max_gap}'
        summary = holdout.run_holdout(path, out, BridgeConfig(max_gap=max_gap), GAPS, per_gap=2,
                                      seed=3)
        return summary, out

    def test_max_gap_5_recovers_everything(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, out = self.run_case(Path(tmp), 5)
            for name in ('planted_observations.parquet', 'holdout_truth.parquet',
                         'holdout_scores.csv', 'holdout_5.json', 'bridged/links.parquet'):
                self.assertTrue((out / name).is_file(), name)
            self.assertEqual(json.loads((out / 'holdout_5.json').read_text()), summary)
            with open(out / 'holdout_scores.csv') as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 4)
            with self.assertRaises(FileExistsError):
                holdout.run_holdout(out / 'planted_observations.parquet', out,
                                    BridgeConfig(max_gap=5), GAPS, per_gap=2, seed=3)
        self.assertEqual(summary['overall'], {'precision': 1.0, 'recall': 1.0})
        self.assertEqual([r['planted'] for r in summary['scores']], [2, 2, 2, 6])
        self.assertEqual([r['recovered'] for r in summary['scores']], [2, 2, 2, 6])
        self.assertEqual(summary['n_unplanted_links'], 0)
        self.assertEqual(summary['config'], BridgeConfig(max_gap=5).to_dict())
        self.assertEqual((summary['seed'], summary['gap_lengths'], summary['per_gap']),
                         (3, [1, 3, 5], 2))

    def test_max_gap_2_misses_long_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, _ = self.run_case(Path(tmp), 2)
        by_gap = {r['gap_frames']: r for r in summary['scores']}
        self.assertEqual(by_gap[1]['recall'], 1.0)
        self.assertEqual(by_gap[3]['recall'], 0.0)
        self.assertEqual(by_gap[5]['recall'], 0.0)
        self.assertEqual(by_gap[3]['missed'] + by_gap[5]['missed'], 4)
        self.assertEqual(summary['overall']['precision'], 1.0)


class DecoyTests(unittest.TestCase):
    """A decoy tracklet 2 px from a planted right-half start competes for the left end.

    Every crab moves at 1 px/frame; the gap-1 plant leaves the left end at
    p_e and the right half starting at p_e + 2 v. A stationary decoy starts on
    the right half's first frame at p_e (2 px from the right-half start, on the
    left end), so it beats the true start on displacement and loses on
    prediction error: the two costs are within the margin.
    """

    def test_decoy_spoils_the_plant(self):
        crabs = [straight_crab(1, 0, N_FRAMES, (300.0, 300.0), (1.0, 0.0))]
        crabs += [straight_crab(c, 0, N_FRAMES, (300.0 + 600.0 * c, 900.0), (0.0, 1.0))
                  for c in (2, 3)]
        base = make_observations(crabs)
        _, truth = holdout.plant_gaps(base, (1, 3), per_gap=1, seed=11)
        row = [r for r in truth.to_pylist() if r['gap_frames'] == 1][0]
        crab = next(c for c in crabs
                    if row['original_id'].endswith(f"/t{c.crab_id:09d}"))
        end_pos = crab.positions[row['deleted_first_frame'] - 1]
        start = row['deleted_last_frame'] + 1
        decoy = Crab(9, np.arange(start, start + 10, dtype=np.int64), np.tile(end_pos, (10, 1)),
                     np.full(10, crab.headings[0]))
        self.assertAlmostEqual(float(np.hypot(*(crab.positions[start] - end_pos))), 2.0)
        with_decoy = make_observations(crabs + [decoy])
        with tempfile.TemporaryDirectory() as tmp:
            path = write(with_decoy, Path(tmp))
            summary = holdout.run_holdout(path, Path(tmp) / 'out', BridgeConfig(max_gap=5),
                                          (1, 3), per_gap=1, seed=11)
            truth_again = pq.read_table(Path(tmp) / 'out/holdout_truth.parquet')
            links = pq.read_table(Path(tmp) / 'out/bridged/links.parquet')
        self.assertTrue(truth_again.equals(truth))   # the short decoy is not eligible
        gap1 = {r['gap_frames']: r for r in summary['scores']}[1]
        self.assertEqual(gap1['recovered'], 0)
        from_left = links.filter(pc.equal(links['tracklet_a'], row['left_id'])).to_pylist()
        decisions = {r['decision'] for r in from_left}
        # Either the decoy is linked (a false link) or the margin veto refuses the
        # left end's assigned pair, leaving the plant missed. Observed (sqrt gate,
        # default margin): the assignment picks the decoy, the veto refuses it and
        # the true pair is refused_assignment.
        wrong_link = gap1['false_link'] == 1 and summary['overall']['precision'] < 1
        vetoed = (gap1['missed'] == 1 and 'refused_margin' in decisions
                  and 'linked' not in decisions)
        self.assertTrue(wrong_link or vetoed, (gap1, from_left))


# ============================================================
# DEMO
# ============================================================


@unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
class DemoSmokeTests(unittest.TestCase):
    """Plant {1, 5, 10, 30} x 5 on the demo clip and bridge with max_gap 30."""

    def test_demo_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = holdout.run_holdout(DEMO, Path(tmp), BridgeConfig(max_gap=30),
                                          (1, 5, 10, 30), per_gap=5, seed=0)
            planted = pq.read_table(Path(tmp) / 'planted_observations.parquet')
            self.assertTrue((Path(tmp) / 'holdout_30.json').is_file())
        self.assertEqual(len(summary['scores']), 5)
        self.assertEqual([r['gap_frames'] for r in summary['scores']], [1, 5, 10, 30, -1])
        for row in summary['scores']:
            for key in ('planted', 'recovered', 'false_link', 'missed'):
                self.assertIsInstance(row[key], int)
                self.assertGreaterEqual(row[key], 0)
            self.assertEqual(row['recovered'] + row['false_link'] + row['missed'],
                             row['planted'])
        self.assertEqual(summary['scores'][-1]['planted'], 20)
        self.assertEqual(pq.read_table(DEMO).num_rows - planted.num_rows, 5 * (1 + 5 + 10 + 30))
        self.assertGreaterEqual(summary['n_unplanted_links'], 0)


if __name__ == '__main__':
    unittest.main()
