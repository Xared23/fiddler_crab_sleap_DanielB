#!/usr/bin/env python3

"""
Tests for training the learned link scorer (data collection, fit, threshold, command).

``labels.label_pairs``, ``labels.dedupe_pairs`` and ``features.feature_matrix``
are replaced by simple deterministic fakes, so that these tests exercise the
trainer alone.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import csv
import json
from pathlib import Path
import sys
import tempfile
from typing import Dict, List, Sequence, Tuple
import unittest
from unittest import mock

# ----- numerics -----
from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

# ----- local modules -----
from gap_bridging import features, holdout, labels, scorer, train  # noqa: E402
from gap_bridging.config import load_config  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

NAMES = features.FEATURE_NAMES
F = len(NAMES)
I_DOR = NAMES.index('dist_over_radius')
I_HAND = NAMES.index('hand_cost')
CONSTRAINED = tuple(n for n in NAMES if features.MONOTONIC.get(n, 0) == -1)
UNCONSTRAINED = tuple(i for i, n in enumerate(NAMES) if features.MONOTONIC.get(n, 0) == 0)
FAST = train.TrainParams(max_iter=40, min_samples_leaf=10)
HORIZONS = (5, 10)
PLANT_GAPS = (2, 4, 8)

# ============================================================
# FAKES
# ============================================================


def fake_feature_matrix(table: pa.Table, names: Sequence[str] = NAMES) -> np.ndarray:
    """Named float columns, nulls as NaN."""
    cols = [np.asarray(table[n].to_numpy(zero_copy_only=False), dtype=float) for n in names]
    return np.column_stack(cols) if cols else np.zeros((table.num_rows, 0))


def fake_fold(original: str) -> str:
    """'test' for originals o000, o001, o005, o006, ...; else 'train'."""
    return 'test' if int(original[1:]) % 5 in (0, 1) else 'train'


def fake_label_pairs(pairs: pa.Table, truth: pa.Table, salt: str = '',
                     test_fraction: float = 0.4) -> pa.Table:
    """Positive when (a, b) is a truth row's (left, right); negative when it touches one."""
    rows = truth.to_pylist()
    by_left = {r['left_id']: r for r in rows}
    by_right = {r['right_id']: r for r in rows}
    out: Dict[str, list] = {n: [] for n in labels.LABELLED_SCHEMA.names}
    for pair in pairs.to_pylist():
        a, b = pair['tracklet_a'], pair['tracklet_b']
        ra, rb = by_left.get(a), by_right.get(b)
        if ra is not None and ra['right_id'] == b:
            label = labels.POSITIVE
        elif ra is not None or rb is not None:
            label = labels.NEGATIVE
        else:
            label = labels.UNLABELLED
        original = (ra or rb or {}).get('original_id')
        for name in pairs.schema.names:
            out[name].append(pair[name])
        out['label'].append(label)
        out['fold'].append('unlabelled' if original is None else fake_fold(original))
        out['role_a'].append(ra['role'] if ra else None)
        out['role_b'].append(rb['role'] if rb else None)
        out['original_a'].append(ra['original_id'] if ra else None)
        out['original_b'].append(rb['original_id'] if rb else None)
    return pa.Table.from_pydict(out, schema=labels.LABELLED_SCHEMA)


def fake_dedupe_pairs(tables: Sequence[pa.Table], horizons: Sequence[int]) -> pa.Table:
    """Last row per (tracklet_a, tracklet_b) in ascending horizon order."""
    kept: Dict[Tuple[str, str], dict] = {}
    for _, table in sorted(zip(horizons, tables), key=lambda t: t[0]):
        for row in table.to_pylist():
            kept[(row['tracklet_a'], row['tracklet_b'])] = row
    return pa.Table.from_pylist(list(kept.values()), schema=tables[0].schema)


def patched():
    """Context manager stack replacing the three unimplemented functions."""
    stack = mock.patch.multiple(labels, label_pairs=fake_label_pairs,
                                dedupe_pairs=fake_dedupe_pairs)
    return stack, mock.patch.object(features, 'feature_matrix', fake_feature_matrix)


class Patched(unittest.TestCase):
    """Base class: fakes active for every test."""

    def setUp(self):
        for p in patched():
            p.start()
            self.addCleanup(p.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)


# ============================================================
# HELPERS
# ============================================================


def separable(n: int = 1000, seed: int = 0, nan_fraction: float = 0.1):
    """``(X, y, groups)``: label = dist_over_radius + hand_cost < 1 with a margin;
    NaN in unconstrained columns; groups of 5 rows."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, F))
    s = X[:, I_DOR] + X[:, I_HAND]
    keep = np.abs(s - 1.0) > 0.1
    X = X[keep]
    y = (X[:, I_DOR] + X[:, I_HAND] < 1.0).astype(int)
    cols = np.asarray(UNCONSTRAINED)
    mask = rng.uniform(size=(X.shape[0], cols.size)) < nan_fraction
    sub = X[:, cols]
    sub[mask] = np.nan
    X[:, cols] = sub
    groups = np.asarray([f'g{i // 5}' for i in range(X.shape[0])], dtype=object)
    return X, y, groups


def pair_row(rng: np.random.Generator, a: str, b: str, gap: int, positive: bool,
             gate: str) -> dict:
    """One PAIR_FEATURES_SCHEMA row; positives have small distance and hand cost."""
    row = {'video_id': 'v', 'tracklet_a': a, 'tracklet_b': b, 'gap_frames': gap, 'gate': gate}
    values = rng.uniform(0.0, 1.0, F)
    values[rng.uniform(size=F) < 0.1] = np.nan
    values[I_DOR] = rng.uniform(0.0, 0.4) if positive else rng.uniform(0.6, 1.5)
    values[I_HAND] = rng.uniform(0.0, 0.5) if positive else rng.uniform(0.7, 2.0)
    for name, v in zip(NAMES, values):
        row[name] = None if np.isnan(v) else float(v)
    row['gap'] = float(gap)
    return row


def make_sweep(root: Path, seed: int = 0, n_plants: int = 40, n_decoys: int = 4,
               horizons: Sequence[int] = HORIZONS, write_features: bool = True) -> Path:
    """A fake sweep directory: ``holdout_<h>/`` with truth, pair features and the JSON.

    Plants ``o000..``: left ``oNNN``, right ``oNNN_r``, gap ``PLANT_GAPS[i % 3]``.
    Per plant: the true pair, two negatives to other plants' right halves (one
    gated out by 'displacement'), one 'max_cost' negative. Decoys: donor right
    halves ``dNNN_r`` linked from a plant's left half (negatives). Plus
    unlabelled pairs between untouched tracklets. Every pair is enumerated at
    every horizon >= its gap with identical features.
    """
    rng = np.random.default_rng(seed)
    truth: List[dict] = []
    pairs: List[dict] = []
    for i in range(n_plants):
        o = f'o{i:03d}'
        truth.append({'left_id': o, 'right_id': f'{o}_r', 'original_id': o,
                      'gap_frames': PLANT_GAPS[i % 3], 'deleted_first_frame': 100,
                      'deleted_last_frame': 100 + PLANT_GAPS[i % 3] - 1, 'role': 'plant',
                      'middle_id': None, 'partner_id': None})
    for i in range(n_plants):
        o, g = f'o{i:03d}', PLANT_GAPS[i % 3]
        other = f'o{(i + 1) % n_plants:03d}_r'
        third = f'o{(i + 7) % n_plants:03d}_r'
        pairs.append(pair_row(rng, o, f'{o}_r', g, True, ''))
        pairs.append(pair_row(rng, o, other, g, False, ''))
        pairs.append(pair_row(rng, o, third, g, False, 'displacement'))
        pairs.append(pair_row(rng, f'u{i:03d}', other, g, False, 'max_cost'))
    for k in range(n_decoys):
        d = f'd{k:03d}'
        truth.append({'left_id': d, 'right_id': f'{d}_r', 'original_id': f'o{900 + k}',
                      'gap_frames': 2, 'deleted_first_frame': 50, 'deleted_last_frame': 51,
                      'role': 'decoy_donor', 'middle_id': None, 'partner_id': f'o{k:03d}'})
        pairs.append(pair_row(rng, f'o{k:03d}', f'{d}_r', 2, False, ''))
    for k in range(5):
        pairs.append(pair_row(rng, f'x{k}', f'y{k}', 2, False, ''))
    truth_table = pa.Table.from_pylist(truth, schema=holdout.TRUTH_SCHEMA)
    sweep = root / f'sweep_{seed}'
    for h in horizons:
        run = sweep / f'holdout_{h}'
        (run / holdout.BRIDGED_DIR).mkdir(parents=True)
        pq.write_table(truth_table, run / holdout.TRUTH_NAME)
        if write_features:
            rows = [r for r in pairs if r['gap_frames'] <= h]
            pq.write_table(pa.Table.from_pylist(rows, schema=features.PAIR_FEATURES_SCHEMA),
                           run / holdout.BRIDGED_DIR / features.PAIR_FEATURES_NAME)
        (run / f'holdout_{h}.json').write_text('{}\n')
    (sweep / 'holdout_99').mkdir()          # incomplete: no JSON
    (sweep / 'max_gap_5').mkdir()
    return sweep


# ============================================================
# TESTS: MODEL
# ============================================================


class FitScorerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.groups = separable()
        cls.hgb, cls.cal, cls.p_oof = train.fit_scorer(cls.X, cls.y, cls.groups,
                                                       train.TrainParams(max_iter=60))
        cls.scorer = scorer.LinkScorer(cls.hgb, cls.cal, mock.Mock(
            feature_names=NAMES, threshold=0.5))

    def test_oof_auc(self):
        self.assertEqual(self.p_oof.shape, self.y.shape)
        self.assertTrue(np.all((self.p_oof >= 0) & (self.p_oof <= 1)))
        self.assertGreater(train.metrics(self.p_oof, self.y, 0.5)['roc_auc'], 0.99)

    def test_monotonic_constraints_set(self):
        cst = list(self.hgb.monotonic_cst)
        self.assertEqual(cst, [features.MONOTONIC.get(n, 0) for n in NAMES])
        self.assertEqual(sum(c == -1 for c in cst), len(CONSTRAINED))

    @settings(deadline=None, max_examples=40)
    @given(st.integers(0, 10**6), st.sampled_from(CONSTRAINED), st.booleans(),
           st.lists(st.floats(-1.0, 4.0), min_size=2, max_size=15))
    def test_property_non_increasing_in_constrained(self, row, name, nan_rest, grid):
        base = self.X[row % self.X.shape[0]].copy()
        if nan_rest:
            base[np.asarray(UNCONSTRAINED)] = np.nan
        values = np.sort(np.asarray(grid))
        X = np.tile(base, (values.size, 1))
        X[:, NAMES.index(name)] = values
        p = self.scorer.predict_proba(X)
        self.assertTrue(np.all(np.diff(p) <= 1e-12), (name, values, p))

    def test_sigmoid_calibration(self):
        _, cal, p = train.fit_scorer(self.X, self.y, self.groups,
                                     train.TrainParams(max_iter=20, calibration='sigmoid'))
        self.assertIsInstance(cal, scorer.SigmoidCalibrator)
        self.assertGreater(train.metrics(p, self.y, 0.5)['roc_auc'], 0.99)

    def test_split_fallback(self):
        y = np.array([1, 1, 0, 0, 0, 0, 1, 0])
        g = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'c', 'd'])
        self.assertEqual(train.effective_splits(y, g, 5), 2)
        with self.assertRaises(ValueError):
            train.effective_splits(np.zeros(4), np.arange(4), 5)
        with self.assertRaises(ValueError):
            train.effective_splits(np.array([1, 0, 0]), np.array(['a', 'b', 'c']), 5)

    def test_deterministic(self):
        params = train.TrainParams(max_iter=20, seed=3)
        a = train.fit_scorer(self.X, self.y, self.groups, params)
        b = train.fit_scorer(self.X, self.y, self.groups, params)
        np.testing.assert_array_equal(a[2], b[2])
        np.testing.assert_array_equal(a[0].predict_proba(self.X), b[0].predict_proba(self.X))


# ============================================================
# TESTS: THRESHOLD AND METRICS
# ============================================================


class ThresholdTests(unittest.TestCase):

    @settings(deadline=None, max_examples=80)
    @given(st.lists(st.tuples(st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.7, 0.9, 1.0]),
                              st.integers(0, 1)), min_size=1, max_size=40),
           st.sampled_from([0.5, 0.8, 0.95, 1.0]))
    def test_property_threshold_meets_target_and_is_minimal(self, data, target):
        p = np.array([d[0] for d in data])
        y = np.array([d[1] for d in data])
        t = train.choose_threshold(p, y, target)

        def precision(u):
            sel = p >= u
            return y[sel].mean() if sel.any() else np.nan

        meets = [u for u in np.unique(p) if precision(u) >= target]
        if not (y == 0).any():
            self.assertEqual(t, p[y == 1].min())
        elif meets:
            self.assertEqual(t, min(meets))
            self.assertGreaterEqual(precision(t), target)
        else:
            self.assertGreater(t, p[y == 0].max())
            self.assertEqual(t, np.nextafter(p[y == 0].max(), np.inf))

    def test_fallback_all_negatives_above_positives(self):
        p = np.array([0.1, 0.2, 0.8, 0.9])
        y = np.array([1, 1, 0, 0])
        self.assertEqual(train.choose_threshold(p, y, 1.0), np.nextafter(0.9, np.inf))

    def test_no_negatives(self):
        self.assertEqual(train.choose_threshold(np.array([0.3, 0.6]), np.array([1, 1]), 1.0), 0.3)

    def test_recall_maximal_on_separable(self):
        p = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
        y = np.array([0, 0, 0, 1, 1])
        self.assertEqual(train.choose_threshold(p, y, 1.0), 0.6)
        self.assertEqual(train.choose_threshold(p, y, 0.6), 0.3)

    def test_metrics_perfect(self):
        p = np.array([0.0, 0.1, 0.9, 1.0])
        y = np.array([0, 0, 1, 1])
        m = train.metrics(p, y, 0.5)
        self.assertEqual((m['n'], m['n_pos']), (4, 2))
        self.assertEqual((m['precision'], m['recall'], m['f1'], m['roc_auc']), (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(m['brier'], 0.005)
        self.assertLess(m['log_loss'], 0.06)

    def test_metrics_one_class_and_empty(self):
        m = train.metrics(np.array([0.2, 0.3]), np.array([0, 0]), 0.5)
        self.assertTrue(np.isnan(m['roc_auc']) and np.isnan(m['precision']))
        self.assertTrue(np.isnan(m['recall']))
        e = train.metrics(np.zeros(0), np.zeros(0), 0.5)
        self.assertEqual(e['n'], 0)
        self.assertTrue(np.isnan(e['brier']))

    def test_per_gap_metrics(self):
        table = pa.table({'gap_frames': pa.array([2, 2, 2, 5, 5], pa.int64()),
                          'label': pa.array([1, 0, 1, 0, 0], pa.int8())})
        p = np.array([0.9, 0.6, 0.3, 0.2, 0.7])
        rows = train.per_gap_metrics(table, p, 0.5)
        self.assertEqual([r['gap_frames'] for r in rows], [2, 5])
        r2, r5 = rows
        self.assertEqual((r2['n'], r2['n_pos'], r2['precision'], r2['recall']), (3, 2, 0.5, 0.5))
        self.assertAlmostEqual(r2['mean_p_pos'], 0.6)
        self.assertAlmostEqual(r2['mean_p_neg'], 0.6)
        self.assertEqual((r5['n'], r5['n_pos'], r5['precision']), (2, 0, 0.0))
        self.assertTrue(np.isnan(r5['recall']) and np.isnan(r5['mean_p_pos']))
        self.assertAlmostEqual(r5['mean_p_neg'], 0.45)


# ============================================================
# TESTS: DATA
# ============================================================


class DataTests(Patched):

    def test_holdout_dirs_sorted(self):
        sweep = self.tmp / 's'
        for h, done in ((45, True), (5, True), (10, True), (7, False)):
            (sweep / f'holdout_{h}').mkdir(parents=True)
            if done:
                (sweep / f'holdout_{h}' / f'holdout_{h}.json').write_text('{}')
        (sweep / 'max_gap_5').mkdir()
        (sweep / 'holdout_x').mkdir()
        found = train.holdout_dirs(sweep)
        self.assertEqual([h for h, _ in found], [5, 10, 45])
        self.assertEqual([d.name for _, d in found], ['holdout_5', 'holdout_10', 'holdout_45'])
        self.assertEqual(train.holdout_dirs(self.tmp / 'missing'), [])

    def test_collect(self):
        sweep = make_sweep(self.tmp, n_plants=40)
        table = train.collect_training_rows([sweep], 'salt', 0.4)
        keys = list(zip(table['tracklet_a'].to_pylist(), table['tracklet_b'].to_pylist()))
        self.assertEqual(len(keys), len(set(keys)), "deduplicated across horizons")
        self.assertIn(('o000', 'o000_r'), keys)                      # gap 2, both horizons
        self.assertIn(('o002', 'o002_r'), keys)                      # gap 8, horizon 10 only
        self.assertTrue(set(table['gate'].to_pylist()) <= {'', 'max_cost'})
        self.assertIn('max_cost', table['gate'].to_pylist())
        self.assertNotIn(labels.UNLABELLED, table['label'].to_pylist())
        self.assertFalse(any(a.startswith('x') for a in table['tracklet_a'].to_pylist()))
        self.assertEqual(set(table['sweep'].to_pylist()), {str(sweep)})
        # 40 positives + 40 '' negatives + 40 max_cost negatives + 4 donor negatives
        self.assertEqual(table.num_rows, 124)
        self.assertEqual(sum(table['label'].to_pylist()), 40)
        roles = list(zip(table['label'].to_pylist(), table['role_b'].to_pylist()))
        self.assertEqual(roles.count((0, 'decoy_donor')), 4)
        dropped = train.collect_training_rows([sweep], 'salt', 0.4, drop_donor_negatives=True)
        self.assertEqual(dropped.num_rows, 120)
        self.assertNotIn('decoy_donor', dropped['role_b'].to_pylist())

    def test_collect_two_sweeps(self):
        a, b = make_sweep(self.tmp, seed=0), make_sweep(self.tmp, seed=1)
        table = train.collect_training_rows([a, b], 'salt', 0.4)
        self.assertEqual(set(table['sweep'].to_pylist()), {str(a), str(b)})
        self.assertEqual(table.num_rows, 2 * 124)

    def test_missing_features(self):
        sweep = make_sweep(self.tmp, write_features=False)
        with self.assertRaises(FileNotFoundError) as ctx:
            train.collect_training_rows([sweep], 'salt', 0.4)
        self.assertIn(features.PAIR_FEATURES_NAME, str(ctx.exception))
        self.assertIn('bridge_gaps.py sweep', str(ctx.exception))
        with self.assertRaises(FileNotFoundError):
            train.collect_training_rows([self.tmp / 'empty'], 'salt', 0.4)

    def test_split_folds(self):
        table = train.collect_training_rows([make_sweep(self.tmp)], 'salt', 0.4)
        folds = train.split_folds(table)
        self.assertEqual(set(folds), {'train', 'test'})
        n = 0
        for name, (X, y, groups) in folds.items():
            self.assertEqual(X.shape, (y.size, F))
            n += y.size
            self.assertTrue(all(fake_fold(g) == name for g in groups))
        self.assertEqual(n, table.num_rows)
        u = table.filter(pc.equal(table['tracklet_a'], 'u001'))
        self.assertEqual(train.split_folds(u)['train'][2].tolist()
                         + train.split_folds(u)['test'][2].tolist(), ['o002'])

    def test_table_sha256_chunking_invariant(self):
        table = train.collect_training_rows([make_sweep(self.tmp)], 'salt', 0.4)
        chunked = pa.concat_tables([table.slice(0, 10), table.slice(10)])
        self.assertEqual(train.table_sha256(table), train.table_sha256(chunked))
        self.assertNotEqual(train.table_sha256(table), train.table_sha256(table.slice(1)))


# ============================================================
# TESTS: COMMAND
# ============================================================


class TrainCommandTests(Patched):

    def setUp(self):
        super().setUp()
        self.sweeps = [make_sweep(self.tmp, seed=0, n_plants=60),
                       make_sweep(self.tmp, seed=1, n_plants=60)]

    def test_end_to_end(self):
        out = self.tmp / 'models' / 'scorer.joblib'
        overlay = self.tmp / 'cfg' / 'overlay.yaml'
        overlay.parent.mkdir()
        overlay.write_text('# old comment\ncost_model: hand\nlearned_margin: 0.1\n')
        m = train.train_command(self.sweeps, out, FAST, 'salt', 0.4, overlay, force=False)
        self.assertTrue(out.is_file() and scorer.manifest_path(out).is_file())
        self.assertGreater(m.n_test, 0)
        self.assertGreater(m.n_train, 0)
        self.assertEqual(m.n_train + m.n_test, 2 * 184)
        self.assertEqual(m.sweeps, tuple(str(s) for s in self.sweeps))
        self.assertEqual(len(m.training_data_sha256), 64)
        self.assertTrue(0 < m.threshold <= 1)
        self.assertGreaterEqual(m.oof_metrics['precision'], 1.0)
        self.assertIn('roc_auc', m.test_metrics)
        self.assertIn('out-of-fold', m.hyperparameters['note'])
        loaded = scorer.load_scorer(out)
        self.assertEqual(loaded.manifest.to_json(), m.to_json())
        self.assertEqual(loaded.threshold, m.threshold)
        with open(train.metrics_csv_path(out)) as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(out.with_name('scorer_metrics.csv'), train.metrics_csv_path(out))
        self.assertEqual({r['fold'] for r in rows}, {'test'})
        self.assertEqual(sorted(int(r['gap_frames']) for r in rows), sorted(PLANT_GAPS))
        text = overlay.read_text()
        self.assertTrue(text.startswith('# Written by `bridge_gaps.py train`'))
        self.assertNotIn('old comment', text)
        cfg = load_config(overlay)
        self.assertEqual(cfg.cost_model, 'learned')
        self.assertEqual(cfg.learned_min_probability, m.threshold)
        self.assertEqual(cfg.learned_margin, 0.1)
        self.assertEqual(Path(cfg.scorer_path).resolve(), out.resolve())
        self.assertEqual(yaml.safe_load(text)['scorer_path'], '../models/scorer.joblib')
        self.assertIs(scorer.resolve_scorer(cfg), scorer.resolve_scorer(cfg))
        with self.assertRaises(FileExistsError):
            train.train_command(self.sweeps, out, FAST, 'salt', 0.4, None, force=False)
        train.train_command(self.sweeps, out, FAST, 'salt', 0.4, None, force=True)

    def test_no_test_fold(self):
        out = self.tmp / 'oof.joblib'
        with mock.patch.object(sys.modules[__name__], 'fake_fold', lambda original: 'train'):
            m = train.train_command(self.sweeps[:1], out, FAST, 'salt', 0.0, None, False)
        self.assertEqual((m.n_test, m.test_metrics), (0, {}))
        with open(train.metrics_csv_path(out)) as fh:
            self.assertEqual({r['fold'] for r in csv.DictReader(fh)}, {'oof'})
        json.loads(scorer.manifest_path(out).read_text())

    def test_deterministic(self):
        a = train.train_command(self.sweeps, self.tmp / 'a.joblib', FAST, 'salt', 0.4, None, False)
        b = train.train_command(self.sweeps, self.tmp / 'b.joblib', FAST, 'salt', 0.4, None, False)
        self.assertEqual(a.training_data_sha256, b.training_data_sha256)
        self.assertEqual(a.threshold, b.threshold)
        self.assertEqual(a.oof_metrics, b.oof_metrics)
        X, _, _ = separable(200, seed=4)
        np.testing.assert_array_equal(scorer.load_scorer(self.tmp / 'a.joblib').predict_proba(X),
                                      scorer.load_scorer(self.tmp / 'b.joblib').predict_proba(X))


if __name__ == '__main__':
    unittest.main()
