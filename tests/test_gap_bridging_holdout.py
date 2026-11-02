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
from unittest import mock

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
from gap_bridging import geometry, holdout, tables  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging_fixtures import (DEFAULT_OFFSETS, Crab, make_observations,  # noqa: E402
                                   straight_crab)

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
                            ('decision', decision), ('reason', ''), ('n_chain_ignored', 0), ('stage', 1), ('size_term', math.nan), ('side_term', math.nan)):
            columns[name].append(value)
    return pa.Table.from_pydict(columns, schema=tables.LINKS_SCHEMA)


def truth_table(rows: List[tuple]) -> pa.Table:
    """A truth table from ``(left_id, right_id, gap[, role, middle_id, partner_id])``."""
    rows = [tuple(r) + ('plant', None, None)[len(r) - 3:] for r in rows]
    return pa.Table.from_pydict({
        'left_id': [r[0] for r in rows], 'right_id': [r[1] for r in rows],
        'original_id': [r[0] for r in rows], 'gap_frames': [r[2] for r in rows],
        'deleted_first_frame': [10] * len(rows),
        'deleted_last_frame': [9 + r[2] for r in rows], 'role': [r[3] for r in rows],
        'middle_id': [r[4] for r in rows], 'partner_id': [r[5] for r in rows],
        'profile': ['standard'] * len(rows), 'category': [None] * len(rows),
        'left_frames': [10] * len(rows), 'right_frames': [N_FRAMES - 10 - r[2] for r in rows]},
        schema=holdout.TRUTH_SCHEMA)


MIRRORED = DEFAULT_OFFSETS * np.array([1.0, -1.0])     # claw on the other side: claw_side +1


def cue_crabs(n: int, n_frames: int = N_FRAMES, speed: float = 0.5) -> List[Crab]:
    """``far_crabs`` with the claw mirrored on every other crab (confident opposite sides)."""
    crabs = far_crabs(n, n_frames)
    for crab in crabs:
        crab.positions = crab.positions[:1] + (crab.positions - crab.positions[:1]) * (speed / 0.5)
        if crab.crab_id % 2 == 0:
            crab.offsets = MIRRORED.copy()
    return crabs


def tracklet_rows(table: pa.Table, tracklet: str) -> pa.Table:
    return table.filter(pc.fill_null(pc.equal(table['tracklet_id'], tracklet), False))


def score_dict(scores: pa.Table) -> Dict[int, dict]:
    """Scores rows keyed by ``gap_frames``."""
    return {row['gap_frames']: row for row in scores.to_pylist()}


def assert_decoy_geometry(test: unittest.TestCase, base: pa.Table, planted: pa.Table,
                          truth: pa.Table, offset_px: float = 3.0) -> int:
    """Check every decoy pair of ``truth``; return the number of pairs.

    Target and donor share the run; the donor's right half (untrimmed) is
    translated next to the target's right-half start; the target's left half
    is the original rows just before the run (trimmed at its start under the
    ``'short'`` profile, where only its first row's ``incoming_reason``
    changes); the target's left half and the donor's right half differ in
    appearance as measured on the planted table.
    """
    rows = truth.to_pylist()
    cfg = BridgeConfig()
    coords = holdout.coordinate_columns(base.schema)
    test.assertIn('body_x', coords)
    test.assertIn('claw_y', coords)
    test.assertIn('centroid_x', coords)
    ends = geometry.endpoints(planted, cfg)
    n_pairs = 0
    for k, row in enumerate(rows):
        if row['role'] != 'decoy_target':
            continue
        n_pairs += 1
        donor = rows[k + 1]
        test.assertEqual(donor['role'], 'decoy_donor')
        test.assertEqual(row['partner_id'], donor['right_id'])
        test.assertEqual(donor['partner_id'], row['left_id'])
        test.assertEqual((row['deleted_first_frame'], row['gap_frames']),
                         (donor['deleted_first_frame'], donor['gap_frames']))
        start = row['deleted_last_frame'] + 1
        target_right = tracklet_rows(planted, row['right_id'])
        donor_right = tracklet_rows(planted, donor['right_id'])
        test.assertEqual(pc.min(target_right['frame']).as_py(), start)
        test.assertEqual(pc.min(donor_right['frame']).as_py(), start)
        t0, d0 = target_right.to_pylist()[0], donor_right.to_pylist()[0]
        test.assertAlmostEqual(d0['body_x'] - t0['body_x'], offset_px, delta=1e-9)
        test.assertAlmostEqual(d0['body_y'] - t0['body_y'], 0.0, delta=1e-9)
        # Every coordinate column of the donor's whole right half moved by the same vector.
        donor_base = tracklet_rows(base, donor['original_id'])
        original = donor_base.filter(pc.greater_equal(donor_base['frame'], start))
        test.assertEqual(donor_right.num_rows, original.num_rows)       # donors are never trimmed
        test.assertEqual(donor['left_frames'] + donor['gap_frames'] + donor['right_frames'],
                         donor_base.num_rows)
        test.assertIsNone(donor['category'])
        moved = donor_right.to_pydict()
        dx = moved['body_x'][0] - original['body_x'][0].as_py()
        dy = moved['body_y'][0] - original['body_y'][0].as_py()
        for name in coords:
            shift = dx if name.endswith('_x') else dy
            for new, old in zip(moved[name], original[name].to_pylist()):
                test.assertAlmostEqual(new, old + shift, delta=1e-9, msg=name)
        # The target's left half: the original rows just before the run.
        target_left = tracklet_rows(planted, row['left_id'])
        target_base = tracklet_rows(base, row['original_id'])
        expected = target_base.filter(pc.and_(
            pc.less(target_base['frame'], row['deleted_first_frame']),
            pc.greater_equal(target_base['frame'],
                             row['deleted_first_frame'] - row['left_frames'])))
        if expected.num_rows == pc.sum(pc.less(target_base['frame'],
                                               row['deleted_first_frame'])).as_py():
            test.assertTrue(target_left.equals(expected))
        else:
            other = [c for c in base.schema.names if c != 'incoming_reason']
            test.assertTrue(target_left.select(other).equals(expected.select(other)))
            test.assertEqual(target_left['incoming_reason'][0].as_py(), 'no_candidate')
        # The pair differs in appearance, measured on the (trimmed) planted halves.
        left_cue = ends[row['left_id']][1]
        right_cue = ends[donor['right_id']][0]
        test.assertTrue(holdout.cues_differ(
            (left_cue.size_px, left_cue.size_n, left_cue.claw_side, left_cue.claw_side_n),
            (right_cue.size_px, right_cue.size_n, right_cue.claw_side, right_cue.claw_side_n),
            cfg))
    return n_pairs


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
        self.assertEqual(set(self.truth['role'].to_pylist()), {'plant'})
        self.assertEqual(self.truth['middle_id'].null_count, 6)
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


class DecoyFlickerPlantTests(unittest.TestCase):
    """plant_gaps with decoys and flicker singletons on 16 synthetic tracklets."""

    @classmethod
    def setUpClass(cls):
        cls.base = make_observations(cue_crabs(16))
        cls.planted, cls.truth = holdout.plant_gaps(cls.base, (1, 3), per_gap=2, seed=5,
                                                    decoys=2, flicker=1)

    def test_roles_and_counts(self):
        roles = self.truth['role'].to_pylist()
        self.assertEqual(roles, ['plant', 'plant', 'decoy_target', 'decoy_donor', 'decoy_target',
                                 'decoy_donor', 'flicker'] * 2)
        originals = self.truth['original_id'].to_pylist()
        self.assertEqual(len(set(originals)), len(originals))
        deleted = sum(g * (2 if r == 'flicker' else 1)
                      for g, r in zip(self.truth['gap_frames'].to_pylist(), roles))
        self.assertEqual(self.base.num_rows - self.planted.num_rows, deleted)
        new = [t for t in set(self.planted['tracklet_id'].drop_null().to_pylist())
               if '/holdout' in t]
        self.assertEqual(len(new), 7 * 2 + 2)       # one per split, two per flicker

    def test_plants_unchanged_by_extras(self):
        """With decoys and flicker the plants of each gap are drawn first, as without."""
        _, plain = holdout.plant_gaps(self.base, (1,), per_gap=2, seed=5)
        first = self.truth.filter(pc.equal(self.truth['gap_frames'], 1)).slice(0, 2)
        self.assertEqual(first['original_id'].to_pylist(), plain['original_id'].to_pylist())
        self.assertEqual(first['deleted_first_frame'].to_pylist(),
                         plain['deleted_first_frame'].to_pylist())

    def test_decoy_geometry(self):
        self.assertEqual(assert_decoy_geometry(self, self.base, self.planted, self.truth), 4)

    def test_flicker_parts(self):
        for row in self.truth.filter(pc.equal(self.truth['role'], 'flicker')).to_pylist():
            g = row['gap_frames']
            self.assertEqual(row['deleted_last_frame'] - row['deleted_first_frame'] + 1, 2 * g + 1)
            middle = tracklet_rows(self.planted, row['middle_id'])
            self.assertEqual(middle['frame'].to_pylist(), [row['deleted_first_frame'] + g])
            self.assertEqual(middle['incoming_reason'].to_pylist(), ['no_candidate'])
            left = tracklet_rows(self.planted, row['left_id'])
            right = tracklet_rows(self.planted, row['right_id'])
            self.assertEqual(pc.max(left['frame']).as_py(), row['deleted_first_frame'] - 1)
            self.assertEqual(pc.min(right['frame']).as_py(), row['deleted_last_frame'] + 1)

    def test_cues_differ(self):
        cfg = BridgeConfig()
        self.assertTrue(holdout.cues_differ((20.0, 3, 0.0, 0), (26.0, 1, 0.0, 0), cfg))
        self.assertFalse(holdout.cues_differ((20.0, 3, 0.0, 0), (25.9, 1, 0.0, 0), cfg))
        self.assertFalse(holdout.cues_differ((20.0, 0, 0.0, 0), (30.0, 3, 0.0, 0), cfg))
        self.assertTrue(holdout.cues_differ((20.0, 9, 1.0, 9), (20.0, 9, -0.9, 5), cfg))
        self.assertFalse(holdout.cues_differ((20.0, 9, 1.0, 9), (20.0, 9, -0.9, 4), cfg))
        self.assertFalse(holdout.cues_differ((20.0, 9, 1.0, 9), (20.0, 9, 1.0, 9), cfg))

    def test_short_pool_plants_fewer_extras(self):
        _, truth = holdout.plant_gaps(make_observations(far_crabs(4)), (1,), per_gap=2, seed=0,
                                      decoys=3, flicker=3)
        roles = truth['role'].to_pylist()
        self.assertEqual(roles.count('plant'), 2)
        self.assertEqual(roles.count('decoy_target'), 0)    # all crabs share their cues
        self.assertEqual(roles.count('flicker'), 2)         # only two tracklets left


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

    def test_decoys_and_flicker(self):
        truth = truth_table([
            ('A', 'A/h1', 2, 'decoy_target', None, 'B/h2'),
            ('B', 'B/h2', 2, 'decoy_donor', None, 'A'),
            ('C', 'C/h3', 2, 'decoy_target', None, 'D/h4'),
            ('D', 'D/h4', 2, 'decoy_donor', None, 'C'),
            ('E', 'E/h6', 2, 'flicker', 'E/h5', None),
            ('F', 'F/h8', 2, 'flicker', 'F/h7', None),
            ('G', 'G/h9', 2, 'plant'),
        ])
        links = links_table([('A', 'B/h2', 'linked'),       # target linked to the decoy
                             ('C', 'C/h3', 'linked'),       # target recovered
                             ('E', 'E/h5', 'linked'), ('E/h5', 'E/h6', 'linked'),
                             ('F', 'F/h8', 'linked'),       # flicker middle left alone
                             ('G', 'G/h9', 'linked')])
        row = score_dict(holdout.score_links(links, truth, {'F/h7': 'spurious',
                                                            'E/h5': 'identity'}))[2]
        self.assertEqual([row[k] for k in ('planted', 'recovered', 'false_link', 'missed')],
                         [3, 2, 1, 0])
        self.assertAlmostEqual(row['precision'], 2 / 3)
        self.assertEqual((row['decoy_planted'], row['decoy_recovered'], row['decoy_false_link']),
                         (2, 1, 1))
        self.assertEqual((row['flicker_planted'], row['flicker_recovered'],
                          row['flicker_spurious']), (2, 1, 1))
        self.assertEqual(score_dict(holdout.score_links(links, truth))[2]['flicker_spurious'], 0)

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


class DecoyPropertyTests(unittest.TestCase):
    """Decoys whose claw side is confidently opposite are never linked with cues on,
    and cues never lower precision."""

    @settings(deadline=None, max_examples=8)
    @given(seed=st.integers(0, 2**16), speed=st.sampled_from([0.0, 0.3, 0.8]))
    def test_mismatched_decoy_never_linked(self, seed, speed):
        table = make_observations(cue_crabs(16, speed=speed))
        scores = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = write(table, Path(tmp))
            for cues in (True, False):
                cfg = BridgeConfig(max_gap=5, appearance_cues=cues)
                summary = holdout.run_holdout(path, Path(tmp) / f'cues_{cues}', cfg, (1, 3, 5),
                                              per_gap=1, seed=seed, decoys=1)
                scores[cues] = summary['scores'][-1]
                self.assertEqual(summary['decoys'], 1)
        on, off = scores[True], scores[False]
        self.assertEqual(on['decoy_planted'], 3)
        self.assertEqual(on['decoy_false_link'], 0)
        precision = {k: v['precision'] for k, v in scores.items()}
        if precision[False] is not None:
            self.assertIsNotNone(precision[True])
            self.assertGreaterEqual(precision[True], precision[False])


def shift_tracklet(table: pa.Table, tracklet: str, dx: float) -> pa.Table:
    """``table`` with every coordinate column of ``tracklet``'s rows moved by ``dx`` along x."""
    mask = pc.fill_null(pc.equal(table['tracklet_id'], tracklet), False).to_numpy(
        zero_copy_only=False)
    for name in holdout.coordinate_columns(table.schema):
        if not name.endswith('_x'):
            continue
        values = np.asarray(table[name].to_numpy(zero_copy_only=False), dtype=float)
        values = np.where(mask, values + dx, values)
        index = table.schema.get_field_index(name)
        table = table.set_column(index, table.schema.field(index),
                                 pa.array(values, table.schema.field(index).type))
    return table


class DecoyVetoExampleTests(unittest.TestCase):
    """Motion favours the decoy: the target's right half is moved 6 px off its predicted
    start and the donor's right half sits on it (decoy_offset_px = 0). Without cues the
    decoy is linked (a false link); with cues the claw-side veto refuses it."""

    def test_veto_turns_false_link_into_recovery_or_miss(self):
        table = make_observations(cue_crabs(16, speed=0.3))
        planted, truth = holdout.plant_gaps(table, (2,), per_gap=1, seed=4, decoys=1,
                                            decoy_offset_px=0.0)
        target = truth.filter(pc.equal(truth['role'], 'decoy_target')).to_pylist()[0]
        planted = shift_tracklet(planted, target['right_id'], 6.0)
        outcome = {}
        with tempfile.TemporaryDirectory() as tmp:
            path = write(planted, Path(tmp))
            for cues in (True, False):
                out = Path(tmp) / f'cues_{cues}'
                from gap_bridging import pipeline
                pipeline.run(path, out, BridgeConfig(max_gap=5, appearance_cues=cues))
                links = pq.read_table(out / 'links.parquet')
                outcome[cues] = score_dict(holdout.score_links(links, truth))[2]
                pair = links.filter(pc.and_(pc.equal(links['tracklet_a'], target['left_id']),
                                            pc.equal(links['tracklet_b'],
                                                     target['partner_id']))).to_pylist()
                self.assertEqual(len(pair), 1)
                if cues:
                    self.assertEqual((pair[0]['decision'], pair[0]['reason']),
                                     ('refused_gate', 'claw_side'))
        self.assertEqual(outcome[False]['decoy_false_link'], 1)
        self.assertEqual(outcome[True]['decoy_false_link'], 0)
        self.assertGreater(outcome[True]['precision'], outcome[False]['precision'])


class FlickerPropertyTests(unittest.TestCase):
    """A flicker singleton bridged on both sides is never flagged spurious, even with low
    scores (it would be spurious if left alone)."""

    @settings(deadline=None, max_examples=6)
    @given(seed=st.integers(0, 2**16), max_gap=st.sampled_from([3, 5]))
    def test_flicker_never_spurious_within_horizon(self, seed, max_gap):
        table = make_observations(far_crabs(8, 80), scores=0.3)
        gaps = tuple(g for g in (1, 2, 3, 5) if g <= max_gap)
        with tempfile.TemporaryDirectory() as tmp:
            summary = holdout.run_holdout(write(table, Path(tmp)), Path(tmp) / 'out',
                                          BridgeConfig(max_gap=max_gap), gaps, per_gap=1,
                                          seed=seed, flicker=1)
        total = summary['scores'][-1]
        self.assertEqual(total['flicker_planted'], len(gaps))
        self.assertEqual(total['flicker_spurious'], 0)
        self.assertEqual(total['flicker_recovered'], len(gaps))

    def test_flicker_beyond_horizon_is_spurious(self):
        table = make_observations(far_crabs(4, 80), scores=0.3)
        with tempfile.TemporaryDirectory() as tmp:
            summary = holdout.run_holdout(write(table, Path(tmp)), Path(tmp) / 'out',
                                          BridgeConfig(max_gap=2), (3,), per_gap=1, seed=0,
                                          flicker=1)
        total = summary['scores'][-1]
        self.assertEqual((total['flicker_planted'], total['flicker_recovered'],
                          total['flicker_spurious']), (1, 0, 1))


# ============================================================
# POOLING AND THE HORIZON RULE
# ============================================================


def by_gap_row(max_gap: int, gap: int, planted: int, recovered: int, false_link: int,
               decoy_planted: int = 0, decoy_recovered: int = 0,
               decoy_false_link: int = 0) -> dict:
    """One hand-built ``holdout_by_gap.csv`` row (no flicker singletons)."""
    return {'max_gap': max_gap, 'gap': gap, 'planted': planted, 'recovered': recovered,
            'false_link': false_link, 'missed': planted - recovered - false_link,
            'decoy_planted': decoy_planted, 'decoy_recovered': decoy_recovered,
            'decoy_false_link': decoy_false_link, 'flicker_planted': 0,
            'flicker_recovered': 0, 'flicker_spurious': 0}


def horizon_sweep(false_at_38_under_45: int = 1) -> List[dict]:
    """Horizons 30 and 45 scored on the same planted gaps 5, 30, 38 and 60.

    Under 30 the plants of 38 and 60 frames are beyond the horizon: missed.
    Under 45 the 38-frame plants are bridged, one of them (by default) to the
    wrong crab; the 60-frame plants stay missed.
    """
    return [by_gap_row(30, 5, 8, 8, 0), by_gap_row(30, 30, 8, 6, 0),
            by_gap_row(30, 38, 8, 0, 0), by_gap_row(30, 60, 8, 0, 0),
            by_gap_row(45, 5, 8, 8, 0), by_gap_row(45, 30, 8, 6, 0),
            by_gap_row(45, 38, 8, 8 - false_at_38_under_45 - 1, false_at_38_under_45),
            by_gap_row(45, 60, 8, 0, 0)]


class PoolTests(unittest.TestCase):
    """``holdout.pool_sweeps`` sums counts and recomputes the ratios from the sums."""

    def test_pooling_arithmetic(self):
        a = [by_gap_row(5, 1, 8, 8, 0, 4, 4, 0), by_gap_row(5, 10, 8, 0, 0, 4, 0, 0)]
        b = [by_gap_row(5, 1, 8, 7, 1, 4, 3, 1), by_gap_row(5, 10, 8, 0, 0, 2, 0, 0)]
        by_gap, by_horizon = holdout.pool_sweeps([a, b])
        g1, g10 = by_gap
        self.assertEqual((g1['planted'], g1['recovered'], g1['false_link'], g1['missed']),
                         (16, 15, 1, 0))
        self.assertEqual((g1['decoy_planted'], g1['decoy_recovered'], g1['decoy_false_link']),
                         (8, 7, 1))
        self.assertEqual(g1['links'], 16)
        self.assertEqual(g1['precision'], 15 / 16)       # of the sums, not the mean of 1 and 7/8
        self.assertEqual(g1['recall'], 15 / 16)
        self.assertTrue(math.isnan(g10['precision']) and math.isnan(g10['precision_lower']))
        self.assertEqual((g10['recall'], g10['links'], g10['decoy_planted']), (0.0, 0, 6))
        h, = by_horizon
        self.assertEqual((h['max_gap'], h['planted'], h['recovered'], h['false_link'],
                          h['missed'], h['links'], h['n_gaps']), (5, 32, 15, 1, 16, 16, 2))
        self.assertEqual((h['precision'], h['recall']), (15 / 16, 15 / 32))

    def test_decoy_false_link_lowers_precision(self):
        # A decoy target linked to its donor is a false link of the target.
        by_gap, _ = holdout.pool_sweeps([[by_gap_row(15, 3, 12, 11, 1, 4, 3, 1)]])
        self.assertEqual(by_gap[0]['precision'], 11 / 12)
        with self.assertRaises(ValueError):            # more decoy false links than false links
            holdout.pool_sweeps([[by_gap_row(15, 3, 12, 12, 0, 4, 3, 1)]])

    def test_exact_lower_bound(self):
        self.assertAlmostEqual(holdout.precision_lower_bound(34, 34), 0.05 ** (1 / 34), places=12)
        self.assertAlmostEqual(holdout.precision_lower_bound(60, 60), 0.05 ** (1 / 60), places=12)
        self.assertAlmostEqual(holdout.precision_lower_bound(8, 9), 0.57086, places=5)
        self.assertEqual(holdout.precision_lower_bound(0, 5), 0.0)
        self.assertTrue(math.isnan(holdout.precision_lower_bound(0, 0)))
        with self.assertRaises(ValueError):
            holdout.precision_lower_bound(3, 2)
        by_gap, _ = holdout.pool_sweeps([[by_gap_row(45, 45, 60, 34, 0)]])
        self.assertAlmostEqual(by_gap[0]['precision_lower'], 0.05 ** (1 / 34), places=12)

    def test_invalid_inputs(self):
        row = by_gap_row(5, 1, 8, 8, 0)
        for sweeps in ([], [[row], [row, by_gap_row(5, 2, 8, 8, 0)]], [[row, row]],
                       [[dict(row, missed=1)]]):
            with self.assertRaises(ValueError):
                holdout.pool_sweeps(sweeps)

    @settings(deadline=None, max_examples=60)
    @given(st.lists(st.lists(st.tuples(st.integers(0, 6), st.integers(0, 6), st.integers(0, 6)),
                             min_size=3, max_size=3), min_size=1, max_size=5))
    def test_pooled_counts_are_sums(self, sweeps):
        tables = [[by_gap_row(10, gap, r + f + m, r, f) for gap, (r, f, m) in zip((1, 5, 10), s)]
                  for s in sweeps]
        by_gap, by_horizon = holdout.pool_sweeps(tables)
        pooled_rev, _ = holdout.pool_sweeps(tables[::-1])
        self.assertEqual([r['planted'] for r in by_gap], [r['planted'] for r in pooled_rev])
        for k, row in enumerate(by_gap):
            for name in holdout.POOL_COUNTS:
                self.assertEqual(row[name], sum(t[k][name] for t in tables))
            if row['links']:
                self.assertEqual(row['precision'], row['recovered'] / row['links'])
        self.assertEqual(by_horizon[0]['planted'], sum(r['planted'] for r in by_gap))


class RuleTests(unittest.TestCase):
    """``holdout.apply_rule``: highest horizon with precision >= the floor at every planted gap."""

    def test_false_link_at_38_under_45_fails_45_and_30_passes(self):
        chosen, table = holdout.apply_rule(*holdout.pool_sweeps([horizon_sweep()]))
        self.assertEqual(chosen, 30)
        h30, h45 = table
        self.assertEqual((h30['max_gap'], h30['rule_pass'], h30['failing_gaps']), (30, True, ''))
        self.assertEqual((h45['max_gap'], h45['rule_pass'], h45['failing_gaps']),
                         (45, False, '38:0.857'))
        self.assertEqual(h45['min_gap_precision'], 6 / 7)
        # Plants beyond the horizon count as misses: same planted total at both horizons.
        self.assertEqual((h30['planted'], h30['missed'], h45['planted']), (32, 18, 32))

    def test_without_the_false_link_45_is_chosen(self):
        chosen, table = holdout.apply_rule(*holdout.pool_sweeps([horizon_sweep(0)]))
        self.assertEqual(chosen, 45)
        self.assertTrue(all(r['rule_pass'] for r in table))

    def test_false_link_from_a_plant_beyond_the_horizon(self):
        rows = horizon_sweep(0)
        rows[-1] = by_gap_row(45, 60, 8, 0, 1)         # 60-frame plant linked to another crab
        chosen, table = holdout.apply_rule(*holdout.pool_sweeps([rows]))
        self.assertEqual((chosen, table[1]['failing_gaps']), (30, '60:0.000'))

    def test_false_link_pooled_over_seeds(self):
        clean = horizon_sweep(0)
        chosen, _ = holdout.apply_rule(*holdout.pool_sweeps([clean, clean, horizon_sweep(1)]))
        self.assertEqual(chosen, 30)

    def test_rule_precision_floor(self):
        pooled = holdout.pool_sweeps([horizon_sweep()])
        self.assertEqual(holdout.apply_rule(*pooled, rule_precision=0.85)[0], 45)
        self.assertEqual(holdout.apply_rule(*pooled, rule_precision=0.0)[0], 45)
        with self.assertRaises(ValueError):
            holdout.apply_rule(*pooled, rule_precision=1.5)

    def test_no_passing_horizon(self):
        rows = [by_gap_row(5, 1, 8, 7, 1)]
        self.assertEqual(holdout.apply_rule(*holdout.pool_sweeps([rows]))[0], None)

    def test_capped_sweep_is_refused(self):
        # A v003 sweep capped at the horizon (gap 38 not planted under 30) cannot be ruled on.
        rows = [r for r in horizon_sweep() if not (r['max_gap'] == 30 and r['gap'] > 30)]
        with self.assertRaises(ValueError):
            holdout.apply_rule(*holdout.pool_sweeps([rows]))

    def test_read_by_gap_round_trip(self):
        rows = horizon_sweep()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'holdout_by_gap.csv'
            with open(path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) + ['recall'])
                writer.writeheader()
                writer.writerows([dict(r, recall=0.5) for r in rows])
            self.assertEqual(holdout.read_by_gap(path), rows)


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


# --------------------------------------------------------------------------- #
# v004: feature export from the hold-out
# --------------------------------------------------------------------------- #
class HoldoutFeatureExportTests(unittest.TestCase):
    """``run_holdout(export_features=True)`` writes the training features and keeps the config."""

    def test_export_features(self):
        from gap_bridging import features, labels
        crabs = [straight_crab(c, 0, N_FRAMES, (200.0 + 500.0 * c, 300.0 + 400.0 * (c % 2)),
                               (0.5, 0.2)) for c in range(1, 9)]
        table = make_observations(crabs)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.parquet'
            pq.write_table(table, path)
            out = Path(tmp) / 'out'
            cfg = BridgeConfig(max_gap=5)
            summary = holdout.run_holdout(path, out, cfg, GAPS, per_gap=2, seed=3,
                                          export_features=True)
            feats = pq.read_table(out / 'bridged' / features.PAIR_FEATURES_NAME)
            links = pq.read_table(out / 'bridged' / 'links.parquet')
            self.assertEqual(feats.num_rows, links.num_rows)
            self.assertEqual(summary['config'], cfg.to_dict())
            self.assertEqual(summary['outputs']['pair_features'], 'bridged/pair_features.parquet')
            truth = pq.read_table(out / 'holdout_truth.parquet')
            labelled = labels.label_pairs(feats, truth)
            positives = labelled.filter(pc.equal(labelled['label'], 1))
            self.assertEqual(positives.num_rows, truth.num_rows)   # every plant is enumerated
            self.assertTrue(set(labelled['fold'].to_pylist()) <= set(labels.FOLDS))
            # The caller's cfg is recorded in holdout_<h>.json; the bridged marker records the
            # configuration actually run, with the export switched on.
            marker = json.loads((out / 'bridged' / tables.COMPLETE_NAME).read_text())
            self.assertFalse(summary['config']['export_pair_features'])
            self.assertTrue(marker['config']['export_pair_features'])
            self.assertIn(features.PAIR_FEATURES_NAME, marker['outputs'])

    def test_regression_linear_export_is_skipped(self):
        # features.pair_feature_table is sqrt-only: under the linear model the export
        # request is ignored instead of failing after the linking.
        from gap_bridging import features
        crabs = [straight_crab(c, 0, N_FRAMES, (200.0 + 500.0 * c, 300.0 + 400.0 * (c % 2)),
                               (0.5, 0.2)) for c in range(1, 5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'observations.parquet'
            pq.write_table(make_observations(crabs), path)
            out = Path(tmp) / 'out'
            cfg = BridgeConfig(max_gap=5, gate_model='linear')
            summary = holdout.run_holdout(path, out, cfg, GAPS, per_gap=1, seed=3,
                                          export_features=True)
            self.assertFalse((out / 'bridged' / features.PAIR_FEATURES_NAME).exists())
            self.assertNotIn('pair_features', summary['outputs'])
            marker = json.loads((out / 'bridged' / tables.COMPLETE_NAME).read_text())
            self.assertFalse(marker['config']['export_pair_features'])
            self.assertNotIn(features.PAIR_FEATURES_NAME, marker['outputs'])


# --------------------------------------------------------------------------- #
# v005: planting profiles (DEFECTS.md D-009)
# --------------------------------------------------------------------------- #
DEMO_GAPS = (1, 2, 3, 5, 10, 15, 20, 30, 45, 60)
GOLDEN = ROOT / 'logs/gap_sweep_v004/current'
SHORT_MAX, LONG_MIN = 3, 10


def check_halves(test: unittest.TestCase, base: pa.Table, planted: pa.Table,
                 truth: pa.Table) -> int:
    """Check the kept halves of every truth row; return the expected number of deleted rows.

    Per row: ``left_id`` / ``right_id`` hold exactly ``left_frames`` /
    ``right_frames`` accepted, consecutive frames adjacent to the run; no
    accepted row of the original (under any of its ids) remains outside the
    halves (and the flicker middle); a left half trimmed at its start begins
    with ``incoming_reason == 'no_candidate'``; categories meet their length
    constraints.
    """
    deleted = 0
    for row in truth.to_pylist():
        before = tracklet_rows(base, row['original_id'])
        before = before.filter(pc.equal(before['accepted_for_tracking'], True))
        frames = before['frame'].to_pylist()
        n, first = len(frames), min(frames)
        left = tracklet_rows(planted, row['left_id'])
        right = tracklet_rows(planted, row['right_id'])
        nl, nr = row['left_frames'], row['right_frames']
        dff, dlf = row['deleted_first_frame'], row['deleted_last_frame']
        test.assertEqual(left['frame'].to_pylist(), list(range(dff - nl, dff)))
        test.assertEqual(right['frame'].to_pylist(), list(range(dlf + 1, dlf + 1 + nr)))
        test.assertGreaterEqual(min(nl, nr), 1)
        parts = {row['left_id'], row['right_id']} | ({row['middle_id']} - {None})
        family = planted.filter(pc.fill_null(pc.or_(
            pc.equal(planted['tracklet_id'], row['original_id']),
            pc.starts_with(planted['tracklet_id'], row['original_id'] + '/holdout')), False))
        kept = set(range(dff - nl, dff)) | set(range(dlf + 1, dlf + 1 + nr))
        if row['role'] == 'flicker':
            kept.add(dff + row['gap_frames'])
        family_ids = set(family['tracklet_id'].to_pylist())
        test.assertEqual(family_ids, parts)
        test.assertEqual(sorted(family['frame'].to_pylist()), sorted(kept))
        reasons = left['incoming_reason'].to_pylist()
        if dff - nl > first:
            test.assertEqual(reasons[0], 'no_candidate')
        else:
            test.assertEqual(reasons[0], before['incoming_reason'][0].as_py())
        test.assertEqual(right['incoming_reason'][0].as_py(), 'no_candidate')
        category = row['category']
        lo, hi = min(nl, nr), max(nl, nr)
        if row['profile'] == 'standard' or row['role'] in ('flicker', 'decoy_donor'):
            test.assertIsNone(category)
        elif row['role'] == 'plant':
            test.assertIn(category, holdout.CATEGORIES)
        if category is None:      # untrimmed: everything from the tracklet start / to its end
            test.assertEqual(n - nl - nr, dlf - dff + 1)
        elif category == 'both_short':
            test.assertLessEqual(hi, SHORT_MAX)
        elif category == 'one_short':
            test.assertTrue(lo <= SHORT_MAX and hi >= LONG_MIN, (nl, nr))
        elif category == 'short_mid':
            test.assertTrue(lo <= SHORT_MAX and 4 <= hi <= 9, (nl, nr))
        elif category == 'mid':
            test.assertTrue(4 <= lo <= 9 and hi >= 4, (nl, nr))
        else:
            test.assertEqual(category, 'both_long')
            test.assertGreaterEqual(lo, LONG_MIN)
        deleted += n - nl - nr - (1 if row['role'] == 'flicker' else 0)
    return deleted


def size_crabs(n: int, n_frames: int = N_FRAMES) -> List[Crab]:
    """``far_crabs`` with every other crab 1.4 times larger (sizes 22 and 30.8 px differ)."""
    crabs = far_crabs(n, n_frames)
    for crab in crabs:
        if crab.crab_id % 2 == 0:
            crab.offsets = DEFAULT_OFFSETS * 1.4
    return crabs


def length_crabs(lengths: List[int]) -> List[Crab]:
    """Far-apart straight crabs with the given numbers of frames."""
    crabs = []
    for c, n_frames in enumerate(lengths, 1):
        origin = (300.0 + 600.0 * ((c - 1) % 5), 300.0 + 600.0 * ((c - 1) // 5))
        crabs.append(straight_crab(c, 0, n_frames, origin, (0.3, 0.2)))
    return crabs


class HalfProfileTests(unittest.TestCase):
    """``HalfProfile`` validation, names and serialisation; ``draw_half_lengths``."""

    def test_named_profiles(self):
        self.assertIs(holdout.HalfProfile.from_name('standard'), holdout.STANDARD_PROFILE)
        self.assertIs(holdout.HalfProfile.from_name('short'), holdout.SHORT_PROFILE)
        self.assertEqual(holdout.STANDARD_PROFILE.min_tracklet_len, 40)
        self.assertEqual(holdout.SHORT_PROFILE.min_tracklet_len, 10)
        self.assertEqual(holdout.PROFILES, ('standard', 'short'))
        self.assertEqual(holdout.CATEGORIES,
                         ('both_short', 'one_short', 'short_mid', 'mid', 'both_long'))
        self.assertEqual(holdout.CATEGORY_PROBABILITIES, (0.40, 0.18, 0.21, 0.13, 0.08))
        with self.assertRaises(ValueError):
            holdout.HalfProfile.from_name('long')
        with self.assertRaises(ValueError):
            holdout.plant_gaps(BASE, GAPS, per_gap=1, profile='long')
        with self.assertRaises(TypeError):
            holdout.plant_gaps(BASE, GAPS, per_gap=1, profile=3)

    def test_invalid_profiles_raise(self):
        bad = [dict(name='other'), dict(min_tracklet_len=0),
               dict(probabilities=(0.5, 0.5, 0.0, 0.0, 0.1)),
               dict(probabilities=(0.5, 0.5, 0.0, 0.0)),        # the v005 four categories
               dict(probabilities=(1.2, -0.2, 0.0, 0.0, 0.0)),
               dict(short_weights=(0.5, 0.5, 0.5)), dict(short_weights=(0.5, 0.5)),
               dict(short_lengths=(0, 1, 2)), dict(short_lengths=(2, 1, 3)),
               dict(short_lengths=(1, 1, 3)), dict(mid_lengths=()),
               dict(mid_lengths=(3, 4, 5)),          # overlaps the short lengths
               dict(long_min=3), dict(mid_min=0)]
        for change in bad:
            kwargs = {'name': 'short', 'min_tracklet_len': 10, **change}
            with self.assertRaises(ValueError, msg=str(change)):
                holdout.HalfProfile(**kwargs)
        ok = holdout.HalfProfile('short', 10, probabilities=[1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(ok.probabilities, (1.0, 0.0, 0.0, 0.0, 0.0))     # stored as a tuple
        hash(ok)

    def test_to_dict_json(self):
        for profile in (holdout.STANDARD_PROFILE, holdout.SHORT_PROFILE):
            d = json.loads(json.dumps(profile.to_dict()))
            self.assertEqual(holdout.HalfProfile(**{k: tuple(v) if isinstance(v, list) else v
                                                    for k, v in d.items()}), profile)
        self.assertEqual(holdout.SHORT_PROFILE.to_dict()['probabilities'],
                         list(holdout.CATEGORY_PROBABILITIES))

    @settings(deadline=None, max_examples=40)
    @given(seed=st.integers(0, 2**31))
    def test_draw_half_lengths_sets(self, seed):
        rng = np.random.default_rng(seed)
        p = holdout.SHORT_PROFILE
        for _ in range(20):
            category, kl, kr = holdout.draw_half_lengths(rng, p)
            self.assertIn(category, holdout.CATEGORIES)
            if category == 'both_short':
                self.assertIn(kl, p.short_lengths)
                self.assertIn(kr, p.short_lengths)
            elif category == 'one_short':
                self.assertEqual((kl is None) + (kr is None), 1)
                self.assertIn(kl if kr is None else kr, p.short_lengths)
            elif category == 'short_mid':            # both sides fixed: one short, one mid
                self.assertTrue((kl in p.short_lengths and kr in p.mid_lengths)
                                or (kr in p.short_lengths and kl in p.mid_lengths), (kl, kr))
            elif category == 'mid':
                self.assertEqual((kl is None) + (kr is None), 1)
                self.assertIn(kl if kr is None else kr, p.mid_lengths)
            else:
                self.assertEqual((kl, kr), (None, None))

    def test_draw_frequencies(self):
        rng = np.random.default_rng(0)
        draws = [holdout.draw_half_lengths(rng, holdout.SHORT_PROFILE) for _ in range(20000)]
        for category, prob in zip(holdout.CATEGORIES, holdout.CATEGORY_PROBABILITIES):
            self.assertAlmostEqual(sum(d[0] == category for d in draws) / len(draws), prob,
                                   delta=0.015)
        shorts = [v for c, kl, kr in draws if c in ('both_short', 'one_short', 'short_mid')
                  for v in (kl, kr) if v is not None and v in holdout.SHORT_LENGTHS]
        for length, weight in zip(holdout.SHORT_LENGTHS, holdout.SHORT_LENGTH_WEIGHTS):
            self.assertAlmostEqual(shorts.count(length) / len(shorts), weight, delta=0.015)
        sides = [kl is None for c, kl, kr in draws if c in ('one_short', 'mid')]
        self.assertAlmostEqual(np.mean(sides), 0.5, delta=0.02)
        short_left = [kl in holdout.SHORT_LENGTHS for c, kl, kr in draws if c == 'short_mid']
        self.assertAlmostEqual(np.mean(short_left), 0.5, delta=0.03)
        mids = [max(kl, kr) for c, kl, kr in draws if c == 'short_mid']
        for length in holdout.MID_LENGTHS:
            self.assertAlmostEqual(mids.count(length) / len(mids), 1 / 6, delta=0.03)

    @settings(deadline=None, max_examples=40)
    @given(n=st.integers(10, 80), gap=st.integers(1, 40), seed=st.integers(0, 2**31),
           kl=st.sampled_from([None, 1, 2, 3, 4, 7, 9]),
           kr=st.sampled_from([None, 1, 2, 3, 4, 7, 9]))
    def test_plant_window(self, n, gap, kl, kr, seed):
        p = holdout.SHORT_PROFILE
        need_l = kl if kl is not None else (p.mid_min if kr in p.mid_lengths else p.long_min)
        need_r = kr if kr is not None else (p.mid_min if kl in p.mid_lengths else p.long_min)
        rng = np.random.default_rng(seed)
        if gap + need_l + need_r > n:
            with self.assertRaises(ValueError):
                holdout._plant_window(100, n, gap, kl, kr, p, rng)
            return
        first_del, tb, ta = holdout._plant_window(100, n, gap, kl, kr, p, rng)
        left = first_del - 100 - tb
        right = 100 + n - 1 - (first_del + gap - 1) - ta
        self.assertGreaterEqual(min(tb, ta), 0)
        self.assertEqual(left, kl if kl is not None else first_del - 100)
        self.assertEqual(right, kr if kr is not None else n - (first_del - 100) - gap)
        self.assertGreaterEqual(left, need_l)
        self.assertGreaterEqual(right, need_r)
        if kl is None:
            self.assertEqual(tb, 0)
        if kr is None:
            self.assertEqual(ta, 0)


class StandardProfileTests(unittest.TestCase):
    """The default profile is v004's draw; the v005 truth columns describe it."""

    def test_explicit_standard_equals_default(self):
        planted, truth = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7)
        for profile in ('standard', holdout.STANDARD_PROFILE):
            again, truth_again = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7,
                                                    profile=profile)
            self.assertTrue(again.equals(planted))
            self.assertTrue(truth_again.equals(truth))
        self.assertEqual(truth.schema, holdout.TRUTH_SCHEMA)
        self.assertEqual(truth.schema.names[:9], list(holdout.TRUTH_V004_NAMES))
        self.assertEqual(truth.schema.names[9:],
                         ['profile', 'category', 'left_frames', 'right_frames'])
        self.assertEqual(set(truth['profile'].to_pylist()), {'standard'})
        self.assertEqual(truth['category'].null_count, truth.num_rows)
        for row in truth.to_pylist():          # every tracklet of BASE spans 0 .. N_FRAMES - 1
            self.assertEqual(row['left_frames'], row['deleted_first_frame'] - 0)
            self.assertEqual(row['right_frames'], N_FRAMES - 1 - row['deleted_last_frame'])
        self.assertEqual(check_halves(self, BASE, planted, truth),
                         BASE.num_rows - planted.num_rows)

    def test_standard_ignores_profile_fields(self):
        # Under 'standard' the min_tracklet_len argument rules, not the profile's.
        custom = holdout.HalfProfile('standard', min_tracklet_len=5, long_min=2, mid_min=1)
        a = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=3, profile=custom)
        b = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=3)
        self.assertTrue(a[0].equals(b[0]) and a[1].equals(b[1]))
        with self.assertRaises(ValueError):
            holdout.plant_gaps(BASE, (1,), per_gap=1, min_tracklet_len=N_FRAMES + 1,
                               profile=custom)

    def test_decoy_flicker_frames(self):
        base = make_observations(cue_crabs(16))
        planted, truth = holdout.plant_gaps(base, (1, 3), per_gap=2, seed=5, decoys=2,
                                            flicker=1)
        self.assertEqual(check_halves(self, base, planted, truth),
                         base.num_rows - planted.num_rows)
        for row in truth.filter(pc.equal(truth['role'], 'flicker')).to_pylist():
            self.assertEqual(row['left_frames'] + 2 * row['gap_frames'] + 1
                             + row['right_frames'], N_FRAMES)


class ShortProfileTests(unittest.TestCase):
    """``profile='short'`` on 12 synthetic 60-frame tracklets."""

    @classmethod
    def setUpClass(cls):
        cls.planted, cls.truth = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7,
                                                    profile='short')

    def test_regression_D009_short_profile_trims_halves(self):
        # D-009: plants used to keep >= 3 frames per side of tracklets >= 40 frames, so a
        # learned scorer never saw a 1-frame positive. The short profile trims the halves.
        truth, planted = self.truth, self.planted
        self.assertEqual(truth.schema, holdout.TRUTH_SCHEMA)
        self.assertEqual(truth.num_rows, 6)
        self.assertEqual(set(truth['profile'].to_pylist()), {'short'})
        self.assertTrue(set(truth['category'].to_pylist()) <= set(holdout.CATEGORIES))
        expected = check_halves(self, BASE, planted, truth)
        self.assertEqual(BASE.num_rows - planted.num_rows, expected)
        for row in truth.to_pylist():
            self.assertEqual(row['deleted_last_frame'] - row['deleted_first_frame'] + 1,
                             row['gap_frames'])
            for side in ('left', 'right'):
                n_side = tracklet_rows(planted, row[f'{side}_id'])
                n_side = pc.sum(n_side['accepted_for_tracking']).as_py()
                self.assertEqual(n_side, row[f'{side}_frames'])
        # This seed exercises a left half trimmed at its start and a short side.
        rows = truth.to_pylist()
        self.assertTrue(any(r['deleted_first_frame'] - r['left_frames'] > 0 for r in rows))
        self.assertTrue(any(min(r['left_frames'], r['right_frames']) <= SHORT_MAX
                            for r in rows))
        # Untouched tracklets stay byte-identical.
        touched = pa.array(truth['original_id'].to_pylist() + truth['right_id'].to_pylist())
        keep_base = pc.invert(pc.fill_null(pc.is_in(BASE['tracklet_id'], touched), False))
        keep_planted = pc.invert(pc.fill_null(pc.is_in(planted['tracklet_id'], touched), False))
        self.assertTrue(BASE.filter(keep_base).equals(planted.filter(keep_planted)))

    def test_deterministic_and_differs_from_standard(self):
        again, truth = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7, profile='short')
        self.assertTrue(again.equals(self.planted))
        self.assertTrue(truth.equals(self.truth))
        same_obj = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7,
                                      profile=holdout.SHORT_PROFILE)
        self.assertTrue(same_obj[1].equals(self.truth))
        standard = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=7)[1]
        self.assertFalse(standard.select(list(holdout.TRUTH_V004_NAMES)).equals(
            self.truth.select(list(holdout.TRUTH_V004_NAMES))))
        other_seed = holdout.plant_gaps(BASE, GAPS, per_gap=2, seed=8, profile='short')[1]
        self.assertFalse(other_seed.equals(self.truth))

    def test_shorter_gaps_independent_of_longer(self):
        for kwargs in (dict(), dict(decoys=1, flicker=1)):
            base = make_observations(size_crabs(20)) if kwargs else BASE
            _, short = holdout.plant_gaps(base, (1, 3), per_gap=2, seed=11, profile='short',
                                          **kwargs)
            _, longer = holdout.plant_gaps(base, (1, 3, 5), per_gap=2, seed=11,
                                           profile='short', **kwargs)
            head = longer.filter(pc.less_equal(longer['gap_frames'], 3))
            self.assertTrue(head.equals(short), kwargs)

    def test_eligibility_by_length(self):
        lengths = [9, 9, 9, 10, 11, 12, 13, 14, 15]
        base = make_observations(length_crabs(lengths))
        ids = base['tracklet_id'].unique().to_pylist()
        n_of = {t: tracklet_rows(base, t).num_rows for t in ids}
        # Both sides 1..3 frames: every 10..15-frame tracklet is plantable for gap 1.
        both_short = holdout.HalfProfile('short', 10, probabilities=(1.0, 0.0, 0.0, 0.0, 0.0))
        for seed in range(10):
            planted, truth = holdout.plant_gaps(base, (1,), per_gap=6, seed=seed,
                                                profile=both_short)
            self.assertEqual(sorted(n_of[t] for t in truth['original_id'].to_pylist()),
                             [10, 11, 12, 13, 14, 15])
            self.assertEqual(check_halves(self, base, planted, truth),
                             base.num_rows - planted.num_rows)
        with self.assertRaises(ValueError):          # the 9-frame tracklets never qualify
            holdout.plant_gaps(base, (1,), per_gap=7, seed=0, profile=both_short)
        # The default short profile: some 10..15-frame plants, never a 9-frame one; a draw
        # that no tracklet can hold (here: a long side >= 10 frames) raises.
        planted_lengths, raised = set(), 0
        for seed in range(30):
            try:
                planted, truth = holdout.plant_gaps(base, (1,), per_gap=1, seed=seed,
                                                    profile='short')
            except ValueError:
                raised += 1
                continue
            planted_lengths |= {n_of[t] for t in truth['original_id'].to_pylist()}
            check_halves(self, base, planted, truth)
        self.assertTrue(planted_lengths and planted_lengths <= set(range(10, 16)))
        self.assertGreater(raised, 0)

    def test_pool_exhaustion_raises(self):
        with self.assertRaises(ValueError):
            holdout.plant_gaps(BASE, GAPS, per_gap=5, profile='short')     # 15 plants, 12 crabs
        with self.assertRaises(ValueError):
            holdout.plant_gaps(BASE, (N_FRAMES - 1,), per_gap=1, profile='short')

    def test_decoy_target_trimmed_donor_not(self):
        base = make_observations(size_crabs(20))
        planted, truth = holdout.plant_gaps(base, (1, 3), per_gap=2, seed=5, decoys=2,
                                            flicker=1, profile='short')
        roles = truth['role'].to_pylist()
        self.assertGreaterEqual(roles.count('decoy_target'), 2)
        self.assertEqual(roles.count('decoy_target'), roles.count('decoy_donor'))
        self.assertEqual(check_halves(self, base, planted, truth),
                         base.num_rows - planted.num_rows)
        self.assertEqual(assert_decoy_geometry(self, base, planted, truth),
                         roles.count('decoy_target'))
        targets = truth.filter(pc.equal(truth['role'], 'decoy_target')).to_pylist()
        self.assertTrue(all(r['category'] in holdout.CATEGORIES for r in targets))
        self.assertTrue(any(r['left_frames'] + r['gap_frames'] + r['right_frames'] < N_FRAMES
                            for r in targets))
        for row in truth.filter(pc.equal(truth['role'], 'flicker')).to_pylist():
            self.assertIsNone(row['category'])
            self.assertEqual(row['left_frames'] + 2 * row['gap_frames'] + 1
                             + row['right_frames'], N_FRAMES)

    def test_untrimmed_decoys(self):
        base = make_observations(size_crabs(20))
        profile = holdout.HalfProfile('short', 10, trim_decoys=False)
        planted, truth = holdout.plant_gaps(base, (1, 3), per_gap=2, seed=5, decoys=2,
                                            profile=profile)
        targets = truth.filter(pc.equal(truth['role'], 'decoy_target')).to_pylist()
        self.assertGreaterEqual(len(targets), 2)
        for row in targets:
            self.assertIsNone(row['category'])
            self.assertEqual(row['left_frames'] + row['gap_frames'] + row['right_frames'],
                             N_FRAMES)
            self.assertGreaterEqual(min(row['left_frames'], row['right_frames']), 3)
        assert_decoy_geometry(self, base, planted, truth)
        self.assertEqual(check_halves(self, base, planted, truth),
                         base.num_rows - planted.num_rows)

    @settings(deadline=None, max_examples=40)
    @given(per_gap=st.integers(1, 4),
           gaps=st.lists(st.integers(1, 20), min_size=1, max_size=3),
           seed=st.integers(0, 2**31))
    def test_property_categories_and_no_reuse(self, per_gap, gaps, seed):
        if per_gap * len(gaps) > 12:
            with self.assertRaises(ValueError):
                holdout.plant_gaps(BASE, gaps, per_gap=per_gap, seed=seed, profile='short')
            return
        planted, truth = holdout.plant_gaps(BASE, gaps, per_gap=per_gap, seed=seed,
                                            profile='short')
        self.assertEqual(truth.num_rows, per_gap * len(gaps))
        originals = truth['original_id'].to_pylist()
        self.assertEqual(len(set(originals)), len(originals))
        self.assertEqual(truth['gap_frames'].to_pylist(),
                         [g for g in gaps for _ in range(per_gap)])
        self.assertTrue(set(truth['category'].to_pylist()) <= set(holdout.CATEGORIES))
        self.assertEqual(check_halves(self, BASE, planted, truth),
                         BASE.num_rows - planted.num_rows)


def sized_length_crabs(lengths: List[int]) -> List[Crab]:
    """``length_crabs`` with every other crab 1.4 times larger (decoy cues differ)."""
    crabs = length_crabs(lengths)
    for crab in crabs:
        if crab.crab_id % 2 == 0:
            crab.offsets = DEFAULT_OFFSETS * 1.4
    return crabs


class ShortDecoyRetryTests(unittest.TestCase):
    """Under ``trim_decoys`` a decoy category no tracklet can hold does not end the gap's
    decoys; another attempt draws again (at most ``DECOY_ATTEMPTS * decoys`` attempts)."""

    # Gap 1 on 20-frame tracklets: both_short needs 10 frames, both_long 21 (never fits).
    PROFILE = holdout.HalfProfile('short', 10, probabilities=(0.7, 0.0, 0.0, 0.0, 0.3))

    def plant(self, base: pa.Table, seed: int, profile: holdout.HalfProfile, decoys: int = 3):
        """``plant_gaps`` at gap 1, also returning the drawn categories in order."""
        drawn: List[str] = []
        real = holdout.draw_half_lengths

        def record(rng, prof):
            out = real(rng, prof)
            drawn.append(out[0])
            return out

        with mock.patch.object(holdout, 'draw_half_lengths', side_effect=record):
            planted, truth = holdout.plant_gaps(base, (1,), per_gap=1, seed=seed, decoys=decoys,
                                                profile=profile)
        return planted, truth, drawn

    def test_regression_unfit_category_does_not_end_decoys(self):
        base = make_observations(sized_length_crabs([20] * 14))
        planted, truth, drawn = self.plant(base, 0, self.PROFILE)
        # The first decoy attempt draws both_long, which no tracklet can hold: the old loop
        # stopped there with no decoy at all.
        self.assertEqual(drawn[:2], ['both_short', 'both_long'])
        targets = truth.filter(pc.equal(truth['role'], 'decoy_target'))
        self.assertEqual(targets.num_rows, 3)
        self.assertEqual(set(targets['category'].to_pylist()), {'both_short'})
        self.assertEqual(assert_decoy_geometry(self, base, planted, truth), 3)
        self.assertEqual(check_halves(self, base, planted, truth),
                         base.num_rows - planted.num_rows)
        n_checked = 0
        for seed in range(1, 12):
            try:
                planted, truth, drawn = self.plant(base, seed, self.PROFILE)
            except ValueError:            # the plant itself drew both_long: plants are mandatory
                continue
            n_checked += 1
            targets = truth.filter(pc.equal(truth['role'], 'decoy_target'))
            self.assertEqual(targets.num_rows, 3, seed)
            self.assertEqual(set(targets['category'].to_pylist()), {'both_short'})
            self.assertEqual(drawn.count('both_short'), 4, (seed, drawn))      # 1 plant + 3
            self.assertEqual(planted, self.plant(base, seed, self.PROFILE)[0])  # deterministic
        self.assertGreater(n_checked, 3)

    def test_attempts_are_bounded(self):
        # One 25-frame tracklet holds the both_long plant; no decoy target can ever fit.
        base = make_observations(sized_length_crabs([25] + [20] * 9))
        both_long = holdout.HalfProfile('short', 10, probabilities=(0.0, 0.0, 0.0, 0.0, 1.0))
        _, truth, drawn = self.plant(base, 0, both_long, decoys=2)
        self.assertEqual(truth['role'].to_pylist(), ['plant'])
        self.assertEqual(len(drawn), 1 + holdout.DECOY_ATTEMPTS * 2)
        # Without trim_decoys an empty target pool still ends the gap's decoys at once.
        untrimmed = holdout.HalfProfile('short', 22, probabilities=(0.0, 0.0, 0.0, 0.0, 1.0),
                                        trim_decoys=False)
        _, truth, drawn = self.plant(base, 0, untrimmed, decoys=2)
        self.assertEqual(truth['role'].to_pylist(), ['plant'])
        self.assertEqual(len(drawn), 1)


class ShortDecoyPropertyTests(unittest.TestCase):
    """Short-profile decoys (targets trimmed per category, donors untrimmed) satisfy the
    plant invariants and the decoy geometry for any seed."""

    BASE = make_observations(size_crabs(20))

    @settings(deadline=None, max_examples=25)
    @given(seed=st.integers(0, 2**31))
    def test_property_decoys_keep_plant_invariants(self, seed):
        planted, truth = holdout.plant_gaps(self.BASE, (1, 3), per_gap=2, seed=seed, decoys=2,
                                            flicker=1, profile='short')
        roles = truth['role'].to_pylist()
        self.assertEqual(roles.count('plant'), 4)
        self.assertEqual(roles.count('decoy_target'), roles.count('decoy_donor'))
        self.assertGreaterEqual(roles.count('decoy_target'), 1)
        originals = truth['original_id'].to_pylist()
        self.assertEqual(len(set(originals)), len(originals))
        self.assertEqual(check_halves(self, self.BASE, planted, truth),
                         self.BASE.num_rows - planted.num_rows)
        self.assertEqual(assert_decoy_geometry(self, self.BASE, planted, truth),
                         roles.count('decoy_target'))
        for row in truth.to_pylist():
            if row['role'] == 'decoy_target':
                self.assertIn(row['category'], holdout.CATEGORIES)
            elif row['role'] == 'decoy_donor':
                self.assertIsNone(row['category'])


class RunHoldoutMinLengthTests(unittest.TestCase):
    """``holdout_<h>.json`` records the effective ``min_tracklet_len``."""

    def test_regression_json_records_effective_min_tracklet_len(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(BASE, Path(tmp))
            for profile, expected in (('standard', 30), ('short', 10)):
                summary = holdout.run_holdout(path, Path(tmp) / profile, BridgeConfig(max_gap=5),
                                              (1, 3), per_gap=1, min_tracklet_len=30, seed=0,
                                              profile=profile)
                self.assertEqual(summary['min_tracklet_len'], expected, profile)
                self.assertEqual(summary['profile'],
                                 holdout.HalfProfile.from_name(profile).to_dict())
                recorded = json.loads((Path(tmp) / profile / 'holdout_5.json').read_text())
                self.assertEqual(recorded['min_tracklet_len'], expected)
            summary = holdout.run_holdout(path, Path(tmp) / 'default', BridgeConfig(max_gap=5),
                                          (1,), per_gap=1, seed=0)
            self.assertEqual(summary['min_tracklet_len'], 40)


class LinkMixTests(unittest.TestCase):
    """``link_side_lengths`` and ``link_mix`` on hand-built tables."""

    def test_side_lengths(self):
        rows = []
        for tracklet, n_frames in (('A', 1), ('B', 2), ('C', 12), ('D', 5)):
            rows += [{'tracklet_id': tracklet, 'accepted_for_tracking': True}] * n_frames
        rows += [{'tracklet_id': 'C', 'accepted_for_tracking': False},
                 {'tracklet_id': None, 'accepted_for_tracking': True}]
        obs = pa.Table.from_pylist(rows)
        links = links_table([('A', 'B', 'linked'), ('C', 'D', 'refused_gate'),
                             ('D', 'C', 'linked'), ('B', 'Z', 'linked')])
        n_a, n_b = holdout.link_side_lengths(obs, links)
        self.assertEqual(n_a.tolist(), [1, 5, 2])
        self.assertEqual(n_b.tolist(), [2, 12, 0])
        self.assertEqual(n_a.dtype, np.int64)

    def test_mix_exact(self):
        pairs = [(1, 1), (3, 2), (2, 2), (1, 10), (40, 3), (4, 9), (5, 30), (2, 6), (12, 10),
                 (0, 5)]
        n_a, n_b = zip(*pairs)
        mix = holdout.link_mix(np.array(n_a), np.array(n_b))
        self.assertEqual(mix['n'], 10)
        self.assertEqual(tuple(mix[k] for k in holdout.CATEGORIES + ('other',)),
                         (0.3, 0.2, 0.1, 0.2, 0.1, 0.1))     # short_mid: (2, 6)
        self.assertAlmostEqual(sum(mix[k] for k in holdout.CATEGORIES + ('other',)), 1.0)
        # Sides of 1..3 frames: 1 x3, 2 x4, 3 x2.
        self.assertEqual(mix['short_weights'], (3 / 9, 4 / 9, 2 / 9))
        wide = holdout.link_mix(np.array(n_a), np.array(n_b), short_max=4, long_min=6)
        self.assertEqual(tuple(wide[k] for k in holdout.CATEGORIES + ('other',)),
                         (0.3, 0.4, 0.0, 0.1, 0.1, 0.1))
        self.assertEqual(set(mix), set(holdout.CATEGORIES) | {'other', 'short_weights', 'n'})
        self.assertEqual(len(wide['short_weights']), 4)

    def test_mix_empty_and_invalid(self):
        mix = holdout.link_mix(np.array([], dtype=int), np.array([], dtype=int))
        self.assertEqual(mix['n'], 0)
        self.assertTrue(math.isnan(mix['mid']) and math.isnan(mix['short_weights'][0]))
        with self.assertRaises(ValueError):
            holdout.link_mix(np.array([1, 2]), np.array([1]))
        with self.assertRaises(ValueError):
            holdout.link_mix(np.array([1]), np.array([1]), short_max=5, long_min=5)
        with self.assertRaises(ValueError):
            holdout.link_mix(np.array([-1]), np.array([1]))

    @settings(deadline=None, max_examples=40)
    @given(st.lists(st.tuples(st.integers(0, 40), st.integers(0, 40)), min_size=1,
                    max_size=50))
    def test_mix_partitions(self, pairs):
        n_a, n_b = (np.array(v) for v in zip(*pairs))
        mix = holdout.link_mix(n_a, n_b)
        self.assertAlmostEqual(sum(mix[k] for k in holdout.CATEGORIES + ('other',)), 1.0)
        for a, b in zip(n_a, n_b):          # each link in exactly one bin
            one = holdout.link_mix(np.array([a]), np.array([b]))
            self.assertEqual(sorted(one[k] for k in holdout.CATEGORIES + ('other',)),
                             [0.0] * 5 + [1.0])
        self.assertEqual(mix, holdout.link_mix(n_b, n_a))          # symmetric in the sides


@unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
class DemoProfileTests(unittest.TestCase):
    """v005 profiles on the demo clip."""

    @classmethod
    def setUpClass(cls):
        cls.obs = pq.read_table(DEMO)

    def plant(self, seed: int, profile: str = 'standard'):
        return holdout.plant_gaps(self.obs, DEMO_GAPS, 8, seed=seed, decoys=4, flicker=2,
                                  cue_config=BridgeConfig(max_gap=30), profile=profile)

    def test_standard_matches_v004_golden(self):
        for seed in (0, 1):
            folder = GOLDEN / f'seed_{seed}' / 'holdout_30'
            if not (folder / 'planted_observations.parquet').is_file():
                self.skipTest(f'v004 golden artefacts absent: {folder}')
            planted, truth = self.plant(seed)
            self.assertTrue(planted.equals(pq.read_table(folder / 'planted_observations.parquet')))
            golden = pq.read_table(folder / 'holdout_truth.parquet')
            self.assertTrue(truth.select(list(holdout.TRUTH_V004_NAMES)).equals(golden))
            recorded = json.loads((folder / 'holdout_30.json').read_text())
            self.assertEqual(recorded['input']['planted_rows'], planted.num_rows)
            self.assertEqual(check_halves(self, self.obs, planted, truth),
                             self.obs.num_rows - planted.num_rows)

    def test_short_profile_plants(self):
        counts = {c: 0 for c in holdout.CATEGORIES}
        n_plants = 0
        for seed in range(5):
            planted, truth = self.plant(seed, 'short')
            self.assertEqual(check_halves(self, self.obs, planted, truth),
                             self.obs.num_rows - planted.num_rows)
            rows = truth.to_pylist()
            for gap in DEMO_GAPS:     # the pool suffices for every role at every gap
                roles = [r['role'] for r in rows if r['gap_frames'] == gap]
                self.assertEqual((roles.count('plant'), roles.count('decoy_target'),
                                  roles.count('flicker')), (8, 4, 2), (seed, gap))
            plants = [r for r in rows if r['role'] == 'plant']
            self.assertEqual(len(plants), 80)
            self.assertTrue(any(min(r['left_frames'], r['right_frames']) == 1 for r in plants))
            for r in plants:
                counts[r['category']] += 1
            n_plants += len(plants)
        # Pooled over 400 plants (one seed of 80 can stray by > 0.12: seed 3 has 42 both_short).
        for category, prob in zip(holdout.CATEGORIES, holdout.CATEGORY_PROBABILITIES):
            self.assertGreater(counts[category], 0)
            self.assertAlmostEqual(counts[category] / n_plants, prob, delta=0.06, msg=category)

    def test_natural_link_mix(self):
        from gap_bridging import pipeline
        with tempfile.TemporaryDirectory() as tmp:
            pipeline.run(DEMO, Path(tmp), BridgeConfig(max_gap=30))
            links = pq.read_table(Path(tmp) / 'links.parquet')
        mix = holdout.link_mix(*holdout.link_side_lengths(self.obs, links))
        self.assertGreater(mix['n'], 500)
        for category, prob in zip(holdout.CATEGORIES, holdout.CATEGORY_PROBABILITIES):
            self.assertAlmostEqual(mix[category], prob, delta=0.05, msg=category)
        self.assertEqual(mix['other'], 0.0)
        for got, want in zip(mix['short_weights'], holdout.SHORT_LENGTH_WEIGHTS):
            self.assertAlmostEqual(got, want, delta=0.05)

    def test_run_holdout_short_with_features(self):
        from gap_bridging import features
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            summary = holdout.run_holdout(DEMO, out, BridgeConfig(max_gap=30), (1, 5, 30),
                                          per_gap=4, seed=0, decoys=1, flicker=1,
                                          export_features=True, profile='short')
            recorded = json.loads((out / 'holdout_30.json').read_text())
            feats = pq.read_table(out / 'bridged' / features.PAIR_FEATURES_NAME)
            links = pq.read_table(out / 'bridged' / 'links.parquet')
            truth = pq.read_table(out / 'holdout_truth.parquet')
        self.assertEqual(recorded, summary)
        self.assertEqual(summary['profile']['name'], 'short')
        self.assertEqual(summary['profile'], holdout.SHORT_PROFILE.to_dict())
        self.assertEqual(feats.num_rows, links.num_rows)
        self.assertEqual(set(truth['profile'].to_pylist()), {'short'})
        self.assertEqual(summary['scores'][-1]['planted'],
                         truth.filter(pc.is_in(truth['role'], pa.array(
                             ['plant', 'decoy_target']))).num_rows)
