#!/usr/bin/env python3

"""
Tests for gap_bridging.labels (hold-out labels and folds for the learned scorer).

A synthetic hold-out (``holdout.plant_gaps`` with plants, decoys and flicker
singletons on far-apart crabs, plus crabs with real missed runs that are never
planted) is reduced to endpoints and pair features; the labels are checked
against the truth table: the positive set, the decoy hard negative, the
unlabelled flicker and donor (left, right) pairs, the role/original columns
and fold constancy. Hypothesis property tests cover ``fold_of`` (deterministic,
salt-sensitive, calibrated), ``truth_folds``, label/truth consistency over
planting seeds and ``dedupe_pairs`` across horizons.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
from dataclasses import replace
from pathlib import Path
import re
import sys
from typing import Dict, List, Tuple
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
from gap_bridging import geometry, holdout, labels  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging.features import FEATURE_NAMES, PAIR_FEATURES_SCHEMA, pair_feature_table  # noqa: E402
from gap_bridging.labels import (LABELLED_SCHEMA, dedupe_pairs, fold_of,  # noqa: E402
                                 label_pairs, truth_folds)
from gap_bridging_fixtures import DEFAULT_OFFSETS, make_observations, straight_crab  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

N_FRAMES = 60
MIRRORED = DEFAULT_OFFSETS * np.array([1.0, -1.0])     # claw on the other side
CFG = BridgeConfig(max_gap=10)

# ============================================================
# HELPERS
# ============================================================


def base_observations() -> pa.Table:
    """
    16 slow straight crabs on a 600-px grid (claw mirrored on every other one),
    plus two crabs placed 15 px from crab 1 with a real 2-frame miss each:
    their 28- and 30-frame parts are too short to plant (< 40 frames), so the
    pairs among them have no planted side.
    """
    crabs = []
    for c in range(1, 17):
        angle = 0.7 * c
        velocity = (0.5 * np.cos(angle), 0.5 * np.sin(angle))
        origin = (300.0 + 600.0 * ((c - 1) % 5), 300.0 + 600.0 * ((c - 1) // 5))
        crab = straight_crab(c, 0, N_FRAMES, origin, velocity)
        if c % 2 == 0:
            crab.offsets = MIRRORED.copy()
        crabs.append(crab)
    crabs.append(straight_crab(17, 0, N_FRAMES, (3000.0, 3000.0), (0.3, 0.0)))
    crabs.append(straight_crab(18, 0, N_FRAMES, (3015.0, 3000.0), (0.3, 0.0)))
    return make_observations(crabs, drop_frames={17: {28, 29}, 18: {28, 29}})


def features_of(planted: pa.Table, cfg: BridgeConfig = CFG) -> pa.Table:
    """Pair features of a planted observation table (one video)."""
    eps = geometry.endpoints(planted, cfg)
    return pair_feature_table([e for _, e in eps.values()], [s for s, _ in eps.values()], cfg,
                              'synthetic')


def by_pair(table: pa.Table) -> Dict[Tuple[str, str], dict]:
    """Rows keyed by (tracklet_a, tracklet_b)."""
    return {(r['tracklet_a'], r['tracklet_b']): r for r in table.to_pylist()}


def expected_positives(truth: pa.Table) -> set:
    """True pairs from the truth table, computed independently of labels.py."""
    out = set()
    for row in truth.to_pylist():
        if row['role'] in ('plant', 'decoy_target'):
            out.add((row['left_id'], row['right_id']))
        elif row['role'] == 'flicker':
            out.update({(row['left_id'], row['middle_id']), (row['middle_id'], row['right_id'])})
    return out


BASE = base_observations()
PLANTED, TRUTH = holdout.plant_gaps(BASE, (1, 3), per_gap=2, seed=5, decoys=2, flicker=1)
FEATURES = features_of(PLANTED)

# ============================================================
# EXAMPLE TESTS
# ============================================================

class FoldTests(unittest.TestCase):

    def test_fold_of_matches_definition(self):
        import hashlib
        for oid in ('a', 'v/t000000001', ''):
            u = int.from_bytes(hashlib.sha256(f"s\x00{oid}".encode()).digest()[:8], 'big') / 2**64
            self.assertEqual(fold_of(oid, 's', 0.4), 'test' if u < 0.4 else 'train')

    def test_fraction_bounds(self):
        self.assertEqual(fold_of('x', test_fraction=0.0), 'train')
        self.assertEqual(fold_of('x', test_fraction=1.0), 'test')
        for bad in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                fold_of('x', test_fraction=bad)

    def test_truth_folds_column(self):
        out = truth_folds(TRUTH, 'salt', 0.5)
        self.assertEqual(out.schema.names, TRUTH.schema.names + ['fold'])
        self.assertEqual(out['fold'].to_pylist(),
                         [fold_of(o, 'salt', 0.5) for o in TRUTH['original_id'].to_pylist()])
        # A truth table without role is all plants (as score_links).
        plain = truth_folds(TRUTH.drop_columns(['role', 'middle_id', 'partner_id']), 'salt', 0.5)
        self.assertEqual(plain['fold'].to_pylist(), out['fold'].to_pylist())

    def test_roles_match_holdout(self):
        self.assertEqual(labels._ROLES, holdout.ROLES)


class LabelExampleTests(unittest.TestCase):
    """Plants, decoys and flicker singletons at gaps 1 and 3; everything in one fold."""

    @classmethod
    def setUpClass(cls):
        cls.out = label_pairs(FEATURES, TRUTH, salt='s', test_fraction=0.0)
        cls.rows = by_pair(cls.out)
        cls.truth = TRUTH.to_pylist()

    def test_schema_and_order(self):
        self.assertTrue(self.out.schema.equals(LABELLED_SCHEMA))
        self.assertEqual(self.out.select(PAIR_FEATURES_SCHEMA.names).to_pylist(),
                         FEATURES.to_pylist())
        roles = [r['role'] for r in self.truth]
        self.assertEqual(sorted(set(roles)), sorted(holdout.ROLES))

    def test_positive_set(self):
        positives = {key for key, r in self.rows.items() if r['label'] == 1}
        self.assertEqual(positives, expected_positives(TRUTH))

    def test_decoys(self):
        for row in self.truth:
            if row['role'] == 'decoy_target':
                hard = self.rows[(row['left_id'], row['partner_id'])]
                self.assertEqual(hard['label'], 0)
                self.assertEqual((hard['role_a'], hard['role_b']), ('decoy_target', 'decoy_donor'))
                self.assertEqual(hard['fold'], 'train')
            if row['role'] == 'decoy_donor':
                fake = self.rows.get((row['left_id'], row['right_id']))
                if fake is not None:
                    self.assertEqual(fake['label'], -1)
                    self.assertEqual(fake['fold'], 'unlabelled')
                # Every END into the donor's moved right half is a marked negative.
                into = [r for (a, b), r in self.rows.items()
                        if b == row['right_id'] and a != row['left_id']]
                self.assertTrue(into)
                for r in into:
                    self.assertEqual((r['label'], r['role_b'], r['original_b']),
                                     (0, 'decoy_donor', row['original_id']))

    def test_flicker(self):
        flickers = [r for r in self.truth if r['role'] == 'flicker']
        self.assertTrue(flickers)
        for row in flickers:
            outer = self.rows[(row['left_id'], row['right_id'])]
            self.assertEqual((outer['label'], outer['fold']), (-1, 'unlabelled'))
            for key in ((row['left_id'], row['middle_id']), (row['middle_id'], row['right_id'])):
                r = self.rows[key]
                self.assertEqual((r['label'], r['role_a'], r['role_b']), (1, 'flicker', 'flicker'))
                self.assertEqual((r['original_a'], r['original_b']),
                                 (row['original_id'], row['original_id']))

    def test_unplanted_pairs_unlabelled(self):
        planted = {r[k] for r in self.truth for k in ('left_id', 'right_id', 'middle_id')
                   if r[k] is not None}
        free = [r for (a, b), r in self.rows.items() if a not in planted and b not in planted]
        self.assertTrue(free, "the scene must contain a pair with no planted side")
        for r in free:
            self.assertEqual((r['label'], r['fold'], r['role_a'], r['role_b']),
                             (-1, 'unlabelled', None, None))

    def test_role_and_original_columns(self):
        end_side = {r['left_id']: r for r in self.truth}
        start_side = {r['right_id']: r for r in self.truth}
        half_original = {}
        for r in self.truth:
            if r['role'] == 'flicker':
                end_side[r['middle_id']] = start_side[r['middle_id']] = r
            for k in ('left_id', 'right_id', 'middle_id'):
                if r[k] is not None:
                    half_original[r[k]] = r['original_id']
        for (a, b), r in self.rows.items():
            ta, tb = end_side.get(a), start_side.get(b)
            self.assertEqual(r['role_a'], ta['role'] if ta else None)
            self.assertEqual(r['role_b'], tb['role'] if tb else None)
            # Every side has an original: its truth row's for any half, else itself.
            self.assertEqual(r['original_a'], half_original.get(a, a))
            self.assertEqual(r['original_b'], half_original.get(b, b))
            if r['label'] != -1:
                self.assertTrue(ta or tb)

    def test_split_folds_unlabel_cross_fold_pairs(self):
        targets = [(r, d) for r, d in zip(self.truth, self.truth[1:])
                   if r['role'] == 'decoy_target' and d['role'] == 'decoy_donor']
        self.assertTrue(targets)
        target, donor = targets[0]
        salt = next(f"s{k}" for k in range(1000)
                    if fold_of(target['original_id'], f"s{k}", 0.5)
                    != fold_of(donor['original_id'], f"s{k}", 0.5))
        rows = by_pair(label_pairs(FEATURES, TRUTH, salt=salt, test_fraction=0.5))
        hard = rows[(target['left_id'], target['partner_id'])]
        self.assertEqual((hard['label'], hard['fold']), (-1, 'unlabelled'))
        true = rows[(target['left_id'], target['right_id'])]
        self.assertEqual((true['label'], true['fold']),
                         (1, fold_of(target['original_id'], salt, 0.5)))

    def test_fold_constancy(self):
        out = label_pairs(FEATURES, TRUTH, salt='x', test_fraction=0.5)
        folds_by_original: Dict[str, set] = {}
        for r in out.to_pylist():
            if r['label'] == -1:
                self.assertEqual(r['fold'], 'unlabelled')
                continue
            for original in (r['original_a'], r['original_b']):
                if original is not None:
                    self.assertEqual(r['fold'], fold_of(original, 'x', 0.5))
                    folds_by_original.setdefault(original, set()).add(r['fold'])
        self.assertTrue(all(len(f) == 1 for f in folds_by_original.values()))

    def test_invalid_truth(self):
        bad = TRUTH.set_column(TRUTH.schema.get_field_index('role'), 'role',
                               pa.array(['bogus'] * TRUTH.num_rows, type=pa.string()))
        with self.assertRaises(ValueError):
            label_pairs(FEATURES, bad)
        twice = pa.concat_tables([TRUTH, TRUTH.slice(0, 1)])
        with self.assertRaises(ValueError):
            label_pairs(FEATURES, twice)
        with self.assertRaises(ValueError):
            label_pairs(FEATURES.drop_columns(['gate']), TRUTH)

    def test_truth_without_role_is_all_plants(self):
        plants = TRUTH.filter(pc.equal(TRUTH['role'], 'plant'))
        out = by_pair(label_pairs(FEATURES, plants.drop_columns(['role', 'middle_id', 'partner_id'])))
        positives = {key for key, r in out.items() if r['label'] == 1}
        self.assertEqual(positives, expected_positives(plants))
        self.assertTrue(all(out[key]['role_a'] == 'plant' for key in positives))


class DedupeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.horizons = (2, 4, 10)
        cls.tables = []
        for h in cls.horizons:
            table = label_pairs(features_of(PLANTED, replace(CFG, max_gap=h)), TRUTH)
            marker = pa.array([f"h{h}"] * table.num_rows, type=pa.string())
            cls.tables.append(table.set_column(0, 'video_id', marker))   # identifies the source

    def test_one_row_per_pair_from_largest_horizon(self):
        out = dedupe_pairs(self.tables[::-1][1:] + self.tables[-1:],
                           list(self.horizons[::-1][1:]) + [self.horizons[-1]])
        self.assertTrue(out.schema.equals(LABELLED_SCHEMA))
        keys = list(zip(out['tracklet_a'].to_pylist(), out['tracklet_b'].to_pylist()))
        self.assertEqual(len(keys), len(set(keys)))
        present = {h: set(zip(t['tracklet_a'].to_pylist(), t['tracklet_b'].to_pylist()))
                   for h, t in zip(self.horizons, self.tables)}
        self.assertEqual(set(keys), set().union(*present.values()))
        for key, source in zip(keys, out['video_id'].to_pylist()):
            self.assertEqual(source, f"h{max(h for h in self.horizons if key in present[h])}")

    def test_perturbed_feature_raises(self):
        small = self.tables[0]
        k = small.schema.get_field_index('dist')
        values = small['dist'].to_pylist()
        values[0] += 1e-9
        perturbed = small.set_column(k, 'dist', pa.array(values, type=pa.float64()))
        a, b = small['tracklet_a'][0].as_py(), small['tracklet_b'][0].as_py()
        with self.assertRaisesRegex(ValueError, re.escape(a)):
            dedupe_pairs([perturbed, self.tables[-1]], [self.horizons[0], self.horizons[-1]])
        self.assertIn((a, b), set(zip(self.tables[-1]['tracklet_a'].to_pylist(),
                                      self.tables[-1]['tracklet_b'].to_pylist())))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            dedupe_pairs(self.tables, self.horizons[:2])
        with self.assertRaises(ValueError):
            dedupe_pairs(self.tables[:2], [3, 3])
        with self.assertRaises(ValueError):
            dedupe_pairs([self.tables[0], FEATURES], [1, 2])
        with self.assertRaises(ValueError):
            dedupe_pairs([pa.concat_tables([self.tables[0], self.tables[0].slice(0, 1)])], [1])

    def test_feature_tables_dedupe_too(self):
        tables = [features_of(PLANTED, replace(CFG, max_gap=h)) for h in self.horizons]
        out = dedupe_pairs(tables, self.horizons)
        self.assertTrue(out.schema.equals(PAIR_FEATURES_SCHEMA))
        self.assertEqual(out.num_rows, tables[-1].num_rows)


def key_table(pairs: List[Tuple[str, str]]) -> pa.Table:
    """A PAIR_FEATURES_SCHEMA table with the given (END, START) keys, gate '' and null features."""
    rows = [dict({'video_id': 'v', 'tracklet_a': a, 'tracklet_b': b, 'gap_frames': 1, 'gate': ''},
                 **{name: None for name in FEATURE_NAMES}) for a, b in pairs]
    return pa.Table.from_pylist(rows, schema=PAIR_FEATURES_SCHEMA)


def truth_table(rows: List[dict]) -> pa.Table:
    """A TRUTH_SCHEMA table from ``{left_id, right_id, original_id[, role, middle_id]}`` dicts."""
    defaults = {'gap_frames': 1, 'deleted_first_frame': 10, 'deleted_last_frame': 10,
                'role': 'plant', 'middle_id': None, 'partner_id': None,
                # v005 columns (ignored by the labels; present so the schema matches)
                'profile': 'standard', 'category': None, 'left_frames': 9, 'right_frames': 9}
    full = [{k: v for k, v in dict(defaults, **r).items() if k in holdout.TRUTH_SCHEMA.names}
            for r in rows]
    return pa.Table.from_pylist(full, schema=holdout.TRUTH_SCHEMA)


class OriginalFoldTests(unittest.TestCase):
    """Regression (review finding 5): the unplanted side's original also decides the fold."""

    FRACTION = 0.5

    @classmethod
    def setUpClass(cls):
        want = {'t0': 'test', 't1': 'train', 't3': 'test', 't4': 'train'}
        cls.salt = next(f"s{k}" for k in range(10000)
                        if all(fold_of(o, f"s{k}", cls.FRACTION) == f for o, f in want.items()))
        cls.truth = truth_table([
            {'left_id': 't0', 'right_id': 't0/holdout001', 'original_id': 't0'},
            {'left_id': 't1', 'right_id': 't1/holdout002', 'original_id': 't1'}])
        pairs = [('t0/holdout001', 't1/holdout002'),   # END of t0's right half -> t1's right half
                 ('t1', 't3'),                         # t1's planted END -> unplanted t3 (test)
                 ('t1', 't4'),                         # t1's planted END -> unplanted t4 (train)
                 ('t4', 't1/holdout002'),              # unplanted END (train) -> planted START
                 ('t1', 't1/holdout002'),              # the true pair
                 ('t0', 't1/holdout002'),              # both planted, folds differ
                 ('t3', 't0/holdout001')]              # unplanted END (test) -> planted START
        cls.rows = by_pair(label_pairs(key_table(pairs), cls.truth, cls.salt, cls.FRACTION))

    def check(self, key, label, fold, originals, roles):
        r = self.rows[key]
        self.assertEqual((r['label'], r['fold']), (label, fold), key)
        self.assertEqual((r['original_a'], r['original_b']), originals, key)
        self.assertEqual((r['role_a'], r['role_b']), roles, key)

    def test_end_of_right_half_maps_to_its_original(self):
        self.check(('t0/holdout001', 't1/holdout002'), -1, 'unlabelled', ('t0', 't1'),
                   (None, 'plant'))

    def test_unplanted_side_in_other_fold_is_unlabelled(self):
        self.check(('t1', 't3'), -1, 'unlabelled', ('t1', 't3'), ('plant', None))

    def test_same_fold_pairs_keep_labels(self):
        self.check(('t1', 't4'), 0, 'train', ('t1', 't4'), ('plant', None))
        self.check(('t4', 't1/holdout002'), 0, 'train', ('t4', 't1'), (None, 'plant'))
        self.check(('t1', 't1/holdout002'), 1, 'train', ('t1', 't1'), ('plant', 'plant'))
        self.check(('t3', 't0/holdout001'), 0, 'test', ('t3', 't0'), (None, 'plant'))
        self.check(('t0', 't1/holdout002'), -1, 'unlabelled', ('t0', 't1'), ('plant', 'plant'))

    def test_half_of_two_originals_rejected(self):
        bad = truth_table([{'left_id': 'a', 'right_id': 'x', 'original_id': 'a'},
                           {'left_id': 'x', 'right_id': 'y', 'original_id': 'b'}])
        with self.assertRaises(ValueError):
            label_pairs(key_table([('a', 'x')]), bad)


# ============================================================
# PROPERTY TESTS
# ============================================================

def random_ids(seed: int, n: int) -> List[str]:
    """``n`` distinct tracklet-like ids from a seeded generator."""
    rng = np.random.default_rng(seed)
    return [f"run{rng.integers(1 << 30)}/cam/t{k:09d}" for k in range(n)]


class FoldPropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=30)
    @given(st.integers(0, 2**32 - 1), st.text(min_size=1, max_size=8), st.text(min_size=1, max_size=8))
    def test_property_fold_of_deterministic_and_salt_sensitive(self, seed, salt, other):
        ids = random_ids(seed, 200)
        first = [fold_of(i, salt, 0.4) for i in ids]
        self.assertEqual(first, [fold_of(i, salt, 0.4) for i in ids])
        if salt != other:
            self.assertNotEqual(first, [fold_of(i, other, 0.4) for i in ids])

    @settings(deadline=None, max_examples=10)
    @given(st.integers(0, 2**32 - 1), st.text(max_size=8))
    def test_property_fold_fraction(self, seed, salt):
        ids = random_ids(seed, 5000)
        folds = [fold_of(i, salt, 0.4) for i in ids]
        self.assertLess(abs(folds.count('test') / len(ids) - 0.4), 0.05)
        self.assertEqual({fold_of(i, salt, 0.0) for i in ids}, {'train'})
        self.assertEqual({fold_of(i, salt, 1.0) for i in ids}, {'test'})

    @settings(deadline=None, max_examples=30)
    @given(st.lists(st.integers(0, 9), min_size=1, max_size=40), st.text(max_size=8),
           st.floats(0.0, 1.0))
    def test_property_truth_folds_constant_per_original(self, originals, salt, fraction):
        n = len(originals)
        truth = truth_table([{'left_id': f"l{k}", 'right_id': f"r{k}", 'original_id': f"o{o}"}
                             for k, o in enumerate(originals)])
        out = truth_folds(truth, salt, fraction)
        fold = {}
        for o, f in zip(out['original_id'].to_pylist(), out['fold'].to_pylist()):
            self.assertEqual(fold.setdefault(o, f), f)
            self.assertIn(f, ('train', 'test'))


class LabelPropertyTests(unittest.TestCase):

    @settings(deadline=None, max_examples=60)
    @given(st.lists(st.sampled_from(['none', 'plant', 'flicker']), min_size=2, max_size=8),
           st.lists(st.tuples(st.integers(0, 100), st.integers(0, 100)), min_size=1, max_size=40),
           st.text(max_size=6), st.floats(0.0, 1.0))
    def test_property_labelled_rows_share_the_originals_fold(self, kinds, picks, salt, fraction):
        truth_rows, ends, starts, original_of = [], [], [], {}
        for k, kind in enumerate(kinds):
            o = f"t{k}"
            if kind == 'none':
                ends.append(o)
                starts.append(o)
                original_of[o] = o
                continue
            row = {'left_id': o, 'right_id': f"{o}/holdout001", 'original_id': o, 'role': kind}
            if kind == 'flicker':
                row['middle_id'] = f"{o}/holdout002"
            truth_rows.append(row)
            halves = [o, f"{o}/holdout001"] + ([row['middle_id']] if kind == 'flicker' else [])
            ends.extend(halves)
            starts.extend(halves)
            original_of.update({h: o for h in halves})
        pairs = sorted({(ends[i % len(ends)], starts[j % len(starts)]) for i, j in picks})
        truth = truth_table(truth_rows) if truth_rows else truth_table(
            [{'left_id': 'zz', 'right_id': 'zz/holdout001', 'original_id': 'zz'}])
        out = label_pairs(key_table(pairs), truth, salt, fraction)
        for r in out.to_pylist():
            a, b = r['tracklet_a'], r['tracklet_b']
            self.assertEqual(r['original_a'], original_of.get(a, a))
            self.assertEqual(r['original_b'], original_of.get(b, b))
            if r['label'] == -1:
                self.assertEqual(r['fold'], 'unlabelled')
                continue
            self.assertTrue(r['role_a'] is not None or r['role_b'] is not None)
            self.assertEqual(fold_of(r['original_a'], salt, fraction), r['fold'])
            self.assertEqual(fold_of(r['original_b'], salt, fraction), r['fold'])

    @settings(deadline=None, max_examples=8)
    @given(st.integers(0, 1000), st.sampled_from([(1,), (2, 4), (1, 3, 6)]),
           st.integers(0, 2), st.integers(0, 1), st.floats(0.0, 1.0))
    def test_property_labels_consistent_with_truth(self, seed, gaps, decoys, flicker, fraction):
        planted, truth = holdout.plant_gaps(BASE, gaps, per_gap=1, seed=seed, decoys=decoys,
                                            flicker=flicker)
        out = label_pairs(features_of(planted), truth, 'p', fraction)
        rows = by_pair(out)
        positives = expected_positives(truth)
        self.assertEqual({k for k, r in rows.items() if r['label'] == 1}, positives & set(rows))
        # Every planted pair within the horizon is enumerated (slow crabs, short gaps).
        self.assertLessEqual(positives, set(rows))
        for key, r in rows.items():
            self.assertIn(r['label'], (-1, 0, 1))
            self.assertEqual(r['fold'] == 'unlabelled', r['label'] == -1)
            if r['label'] == 1:
                self.assertEqual(r['original_a'], r['original_b'])
                self.assertEqual(r['fold'], fold_of(r['original_a'], 'p', fraction))
            if r['label'] == 0:
                self.assertTrue(r['role_a'] is not None or r['role_b'] is not None)
                self.assertNotIn(key, positives)


if __name__ == '__main__':
    unittest.main()
