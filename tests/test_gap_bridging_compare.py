#!/usr/bin/env python3

"""
Tests for the fold-wise variant comparison of gap bridging (compare.py, v004).

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Dict, List
import unittest
from unittest import mock

# ----- numerics -----
from hypothesis import given, settings, strategies as st
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

# ----- local modules -----
import bridge_gaps  # noqa: E402
from gap_bridging import compare, holdout, labels  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging_fixtures import DEFAULT_OFFSETS, make_observations, straight_crab  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

HORIZONS = (5, 10)
GAPS = (1, 3, 12)
SEEDS = (1, 2)
COUNTS = holdout.POOL_COUNTS
LOGS = ROOT / 'logs/gap_sweep'

# ============================================================
# HELPERS
# ============================================================


def fake_fold(original_id: str) -> str:
    """Deterministic stand-in for labels.fold_of: parity of the second SHA-256 byte
    (chosen so both folds hold plants of both synthetic seeds)."""
    return 'test' if hashlib.sha256(original_id.encode()).digest()[1] % 2 else 'train'


def fake_truth_folds(truth: pa.Table, salt: str = labels.DEFAULT_SALT,
                     test_fraction: float = labels.DEFAULT_TEST_FRACTION) -> pa.Table:
    """Truth plus ``fold`` from :func:`fake_fold` of ``original_id``."""
    folds = [fake_fold(o) for o in truth['original_id'].to_pylist()]
    return truth.append_column('fold', pa.array(folds, pa.string()))


def cue_scene(n: int = 16, n_frames: int = 80):
    """Far-apart slow crabs, claw mirrored on every other one (decoys find donors)."""
    crabs = []
    for c in range(1, n + 1):
        velocity = (0.5 * math.cos(0.7 * c), 0.5 * math.sin(0.7 * c))
        origin = (300.0 + 600.0 * ((c - 1) % 5), 300.0 + 600.0 * ((c - 1) // 5))
        crab = straight_crab(c, 0, n_frames, origin, velocity)
        if c % 2 == 0:
            crab.offsets = DEFAULT_OFFSETS * [1.0, -1.0]
        crabs.append(crab)
    return make_observations(crabs)


def build_sweep(directory: Path, observations: Path, seed: int) -> List[dict]:
    """``holdout_<h>/`` for every horizon (as ``bridge_gaps.sweep --holdout`` makes them).

    Returns the ``holdout_by_gap.csv`` rows the sweep would have written.
    """
    rows = []
    for horizon in HORIZONS:
        summary = holdout.run_holdout(observations, directory / f'holdout_{horizon}',
                                      BridgeConfig(max_gap=horizon), GAPS, per_gap=2,
                                      seed=seed, decoys=1, flicker=1)
        for score in summary['scores'][:-1]:
            rows.append({'max_gap': horizon, 'gap': score['gap_frames'],
                         **{k: score[k] for k in COUNTS},
                         'recall': compare._by_gap_row(horizon, score)['recall'],
                         'precision': compare._by_gap_row(horizon, score)['precision']})
    return rows


def by_gap(rows: List[dict]) -> Dict[tuple, dict]:
    return {(r['max_gap'], r['gap']): r for r in rows}


def row(gap: int, precision: float, recall: float, max_gap: int = 30) -> dict:
    """A pooled by-gap row with only what promotion_check reads."""
    return {'max_gap': max_gap, 'gap': gap, 'precision': precision, 'recall': recall}


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or a == b


# ============================================================
# FOLD SCORES
# ============================================================


class FoldScoreTests(unittest.TestCase):
    """fold_scores / sweep_fold_rows on two synthetic seeds with decoys and flicker."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        path = root / 'observations.parquet'
        pq.write_table(cue_scene(), path)
        cls.sweeps, cls.expected = [], []
        for seed in SEEDS:
            cls.sweeps.append(root / f'seed_{seed}')
            cls.expected.append(build_sweep(cls.sweeps[-1], path, seed))
        # Same plants under other directories (a second "variant").
        cls.copies = [root / f'copy_{seed}' for seed in SEEDS]
        for sweep, copy in zip(cls.sweeps, cls.copies):
            shutil.copytree(sweep, copy)
        cls.patch = mock.patch.object(labels, 'truth_folds', fake_truth_folds)
        cls.patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.patch.stop()
        cls.tmp.cleanup()

    def test_scene_has_both_folds_and_every_role(self):
        truth = pq.read_table(self.sweeps[0] / 'holdout_5' / holdout.TRUTH_NAME)
        folds = {fake_fold(o) for o in truth['original_id'].to_pylist()}
        self.assertEqual(folds, {'train', 'test'})
        self.assertEqual(set(truth['role'].to_pylist()), set(holdout.ROLES))

    def test_all_fold_reproduces_holdout_by_gap(self):
        for sweep, expected in zip(self.sweeps, self.expected):
            got = compare.sweep_fold_rows(sweep, 'all')
            self.assertEqual([(r['max_gap'], r['gap']) for r in got],
                             [(h, g) for h in HORIZONS for g in GAPS])
            for g, e in zip(got, expected):
                self.assertEqual({k: g[k] for k in COUNTS}, {k: e[k] for k in COUNTS})
                self.assertTrue(same(g['recall'], e['recall']))
                self.assertTrue(same(g['precision'], e['precision']))

    def test_columns_match_holdout_by_gap_csv(self):
        self.assertEqual(compare.BY_GAP_COLUMNS, bridge_gaps.BY_GAP_COLUMNS)
        self.assertEqual(compare.POOLED_GAP_COLUMNS, bridge_gaps.POOLED_GAP_COLUMNS)
        self.assertEqual(compare.POOLED_HORIZON_COLUMNS, bridge_gaps.POOLED_HORIZON_COLUMNS)
        for fold in compare.FOLD_CHOICES:
            for r in compare.fold_scores(self.sweeps[0] / 'holdout_10', fold):
                self.assertEqual(tuple(r), bridge_gaps.BY_GAP_COLUMNS)
                self.assertEqual(r['max_gap'], 10)

    def test_train_plus_test_is_all(self):
        for sweep in self.sweeps:
            folds = {f: by_gap(compare.sweep_fold_rows(sweep, f)) for f in compare.FOLD_CHOICES}
            self.assertEqual(set(folds['train']), set(folds['all']))
            self.assertEqual(set(folds['test']), set(folds['all']))
            for key, total in folds['all'].items():
                for name in COUNTS:
                    self.assertEqual(folds['train'][key][name] + folds['test'][key][name],
                                     total[name], (key, name))
            self.assertGreater(sum(r['planted'] for r in folds['test'].values()), 0)
            self.assertGreater(sum(r['planted'] for r in folds['train'].values()), 0)

    def test_fold_counts_only_its_plants(self):
        truth = pq.read_table(self.sweeps[0] / 'holdout_5' / holdout.TRUTH_NAME).to_pylist()
        for fold in ('train', 'test'):
            scored = [r for r in truth if r['role'] in ('plant', 'decoy_target')
                      and fake_fold(r['original_id']) == fold]
            rows = compare.fold_scores(self.sweeps[0] / 'holdout_5', fold)
            self.assertEqual(sum(r['planted'] for r in rows), len(scored))
            flick = [r for r in truth if r['role'] == 'flicker'
                     and fake_fold(r['original_id']) == fold]
            self.assertEqual(sum(r['flicker_planted'] for r in rows), len(flick))

    def test_keep_fold_keeps_own_fold_and_donors_do_not_score(self):
        directory = self.sweeps[0] / 'holdout_5'
        truth = fake_truth_folds(pq.read_table(directory / holdout.TRUTH_NAME))
        for fold in ('train', 'test'):
            kept = compare._keep_fold(truth, fold).to_pylist()
            self.assertEqual(kept, [r for r in truth.to_pylist() if r['fold'] == fold])
        links = pq.read_table(directory / holdout.BRIDGED_DIR / 'links.parquet')
        plain = truth.drop_columns(['fold'])
        no_donors = plain.filter(pa.array([r != 'decoy_donor'
                                           for r in plain['role'].to_pylist()]))
        self.assertLess(no_donors.num_rows, plain.num_rows)
        self.assertEqual(compare._json_safe(holdout.score_links(links, plain).to_pylist()),
                         compare._json_safe(holdout.score_links(links, no_donors).to_pylist()))

    def test_incomplete_holdout_dir_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            sweep = Path(tmp) / 'sweep'
            shutil.copytree(self.sweeps[0], sweep)
            (sweep / 'holdout_10' / 'holdout_10.json').unlink()
            self.assertEqual([h for h, _ in compare.holdout_dirs(sweep)], [5])
            self.assertEqual({r['max_gap'] for r in compare.sweep_fold_rows(sweep, 'all')}, {5})
            with self.assertRaisesRegex(ValueError, r"'b' sweep 0 .*horizons \[5\]"):
                compare.compare_variants({'a': self.sweeps[:1], 'b': [sweep]}, fold='all')

    def test_unknown_fold_and_empty_sweep(self):
        with self.assertRaises(ValueError):
            compare.fold_scores(self.sweeps[0] / 'holdout_5', 'validation')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                compare.sweep_fold_rows(Path(tmp), 'all')

    def test_identical_variants(self):
        result = compare.compare_variants({'a': self.sweeps, 'b': list(self.sweeps)},
                                          fold='test', through_gap=10)
        a, b = result['variants']['a'], result['variants']['b']
        self.assertEqual(result['baseline'], 'a')
        self.assertEqual(json.dumps(compare._json_safe(a['by_gap'])),
                         json.dumps(compare._json_safe(b['by_gap'])))
        self.assertEqual(json.dumps(compare._json_safe(a['by_horizon'])),
                         json.dumps(compare._json_safe(b['by_horizon'])))
        self.assertEqual(a['chosen'], b['chosen'])
        self.assertTrue(b['promotion']['recall_ok'])
        self.assertEqual(b['promotion'], a['promotion'])
        self.assertEqual(b['promotion']['ok'], b['promotion']['precision_ok'])
        expected_h = a['chosen'] if a['chosen'] is not None else compare.FALLBACK_HORIZON
        self.assertEqual(result['horizon'], expected_h)
        planted = sum(r['planted'] for r in a['by_gap'])
        self.assertEqual(planted, sum(r['planted'] for s in self.sweeps
                                      for r in compare.sweep_fold_rows(s, 'test')))

    def test_same_plants_required(self):
        self.assertEqual(compare.check_same_plants({'a': self.sweeps, 'b': self.copies}),
                         list(HORIZONS))
        with self.assertRaisesRegex(ValueError, 'different numbers of sweep directories'):
            compare.compare_variants({'a': self.sweeps, 'b': self.copies[:1]}, fold='all')
        with self.assertRaisesRegex(ValueError, r"'b' sweep 0 .*horizon 5"):
            compare.compare_variants({'a': self.sweeps, 'b': self.sweeps[::-1]}, fold='all')
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp) / 'seed_2'
            shutil.copytree(self.sweeps[1], changed)
            path = changed / 'holdout_10' / holdout.TRUTH_NAME
            truth = pq.read_table(path).to_pylist()
            truth[0]['deleted_last_frame'] += 1
            pq.write_table(pa.Table.from_pylist(truth, schema=holdout.TRUTH_SCHEMA), path)
            with self.assertRaisesRegex(ValueError, r"'b' sweep 1 .*horizon 10.*same plants"):
                compare.compare_variants({'a': self.sweeps, 'b': [self.copies[0], changed]},
                                         fold='test')
            # Row order and schema metadata alone do not count as a difference.
            truth[0]['deleted_last_frame'] -= 1
            shuffled = pa.Table.from_pylist(truth[::-1], schema=holdout.TRUTH_SCHEMA)
            pq.write_table(shuffled.replace_schema_metadata({'x': 'y'}), path)
            compare.check_same_plants({'a': self.sweeps, 'b': [self.copies[0], changed]})

    def test_markdown_rows_and_variants(self):
        result = compare.compare_variants({'cur': self.sweeps, 'new': self.copies},
                                          fold='all', through_gap=10)
        text = compare.markdown_tables(result)
        for horizon in HORIZONS:
            section = text.split(f"#### `max_gap` {horizon}\n")[1].split('####')[0]
            body = [line for line in section.splitlines() if line.startswith('| ')][2:]
            self.assertEqual([line.split('|')[1].strip() for line in body], [str(g) for g in GAPS])
            self.assertIn('cur precision', section)
            self.assertIn('new recall', section)
        self.assertIn('#### Per horizon', text)
        verdicts = text.split('#### Verdicts')[1]
        self.assertIn('| cur |', verdicts)
        self.assertIn('| new |', verdicts)

    def test_write_comparison_and_force(self):
        result = compare.compare_variants({'a': self.sweeps}, fold='all', through_gap=10)
        result['command'] = 'bridge_gaps.py compare --variant a=x'
        with tempfile.TemporaryDirectory() as tmp:
            out, docs = Path(tmp) / 'cmp', Path(tmp) / 'docs/table.md'
            compare.write_comparison(result, out, docs, force=False)
            for name in (compare.COMPARISON_GAP_NAME, compare.COMPARISON_HORIZON_NAME,
                         compare.PROMOTION_NAME):
                self.assertTrue((out / name).is_file(), name)
            with open(out / compare.COMPARISON_GAP_NAME, newline='') as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames), compare.COMPARISON_GAP_COLUMNS)
                self.assertEqual(len(list(reader)), len(HORIZONS) * len(GAPS))
            with open(out / compare.COMPARISON_HORIZON_NAME, newline='') as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames[0], 'variant')
                self.assertEqual(len(list(reader)), len(HORIZONS))
            verdicts = json.loads((out / compare.PROMOTION_NAME).read_text())
            self.assertEqual(verdicts['variants']['a']['chosen'],
                             result['variants']['a']['chosen'])
            self.assertIn('bridge_gaps.py compare --variant a=x', docs.read_text())
            before = docs.read_text()
            docs.write_text('hand edits\n')
            with self.assertRaises(FileExistsError):
                compare.write_comparison(result, Path(tmp) / 'other', docs, force=False)
            self.assertEqual(docs.read_text(), 'hand edits\n')
            self.assertFalse((Path(tmp) / 'other').exists())
            compare.write_comparison(result, out, docs, force=True)
            self.assertEqual(docs.read_text(), before)

    def test_main_parses_variants(self):
        parser = argparse.ArgumentParser()
        compare.add_arguments(parser)
        args = parser.parse_args(['--variant', 'a=d1,d2', '--variant', 'b=d3', '--out', 'o'])
        self.assertEqual(args.variant, [('a', [Path('d1'), Path('d2')]), ('b', [Path('d3')])])
        self.assertEqual((args.fold, args.through_gap, args.rule_precision, args.force),
                         ('test', 30, 1.0, False))
        self.assertEqual((args.salt, args.test_fraction),
                         (labels.DEFAULT_SALT, labels.DEFAULT_TEST_FRACTION))
        with mock.patch.object(compare, 'compare_variants', side_effect=RuntimeError('x')) \
                as called:
            with self.assertRaises(RuntimeError):
                compare.main(args)
        self.assertEqual(called.call_args.args[0], {'a': [Path('d1'), Path('d2')],
                                                    'b': [Path('d3')]})
        for bad in ('a', '=d1', 'a='):
            with self.assertRaises(SystemExit):
                parser.parse_args(['--variant', bad, '--out', 'o'])

    def test_parse_variant(self):
        self.assertEqual(compare.parse_variant(' a =d1,,d2'), ('a', [Path('d1'), Path('d2')]))
        for bad in ('a', 'a/b/c', '=d1', ' =d1', 'a=', 'a=,', 'a= , '):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                compare.parse_variant(bad)

    def test_main_end_to_end(self):
        parser = argparse.ArgumentParser()
        compare.add_arguments(parser)
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp) / 'cmp.md'
            args = parser.parse_args(
                ['--variant', f"cur={self.sweeps[0]},{self.sweeps[1]}",
                 '--variant', f"new={self.copies[0]},{self.copies[1]}",
                 '--out', str(Path(tmp) / 'out'), '--docs', str(docs),
                 '--fold', 'train', '--through-gap', '10'])
            result = compare.main(args)
            self.assertTrue(docs.is_file())
            self.assertIn('Generated by:', docs.read_text())
        self.assertEqual(result['fold'], 'train')
        self.assertEqual(json.dumps(compare._json_safe(result['variants']['cur']['by_gap'])),
                         json.dumps(compare._json_safe(result['variants']['new']['by_gap'])))


# ============================================================
# PROMOTION
# ============================================================


class PromotionTests(unittest.TestCase):
    """promotion_check truth table on handcrafted pooled rows."""

    CURRENT = [row(1, 1.0, 1.0), row(5, 1.0, 0.9), row(30, 1.0, 0.5), row(45, 1.0, 0.4),
               row(1, 1.0, 1.0, max_gap=15)]

    def test_identical_passes(self):
        check = compare.promotion_check(self.CURRENT, self.CURRENT, 30)
        self.assertTrue(check['ok'] and check['precision_ok'] and check['recall_ok'])
        self.assertEqual(check['failing'], [])
        self.assertEqual(check['gaps'], [1, 5, 30])

    def test_precision_below_one_fails(self):
        cand = [dict(r) for r in self.CURRENT]
        cand[1]['precision'] = 0.95
        check = compare.promotion_check(self.CURRENT, cand, 30)
        self.assertEqual((check['ok'], check['precision_ok'], check['recall_ok']),
                         (False, False, True))
        self.assertEqual(check['failing'], ['5:precision'])

    def test_recall_drop_fails(self):
        cand = [dict(r) for r in self.CURRENT]
        cand[2]['recall'] = 0.45
        check = compare.promotion_check(self.CURRENT, cand, 30)
        self.assertEqual((check['ok'], check['precision_ok'], check['recall_ok']),
                         (False, True, False))
        self.assertEqual(check['failing'], ['30:recall'])

    def test_nan_precision_passes(self):
        cand = [dict(r) for r in self.CURRENT]
        cand[0]['precision'] = math.nan
        cand[0]['recall'] = math.nan
        self.assertTrue(compare.promotion_check(self.CURRENT, cand, 30)['ok'])

    def test_gaps_beyond_through_gap_and_other_horizons_ignored(self):
        cand = [dict(r) for r in self.CURRENT]
        cand[3].update(precision=0.5, recall=0.0)       # gap 45 > through_gap 30
        cand[4].update(precision=0.0, recall=0.0)       # another horizon
        self.assertTrue(compare.promotion_check(self.CURRENT, cand, 30, through_gap=30)['ok'])
        check = compare.promotion_check(self.CURRENT, cand, 30, through_gap=45)
        self.assertEqual(check['failing'], ['45:precision', '45:recall'])

    def test_missing_gap_or_horizon_fails(self):
        check = compare.promotion_check(self.CURRENT, self.CURRENT[1:], 30)
        self.assertEqual(check['failing'], ['1:missing'])
        self.assertFalse(check['ok'])
        self.assertFalse(compare.promotion_check(self.CURRENT, self.CURRENT, 60)['ok'])

    @settings(deadline=None, max_examples=60)
    @given(cur=st.lists(st.tuples(st.sampled_from([math.nan, 0.5, 0.9, 1.0]),
                                  st.floats(0, 1)), min_size=1, max_size=6),
           drops=st.lists(st.booleans(), min_size=6, max_size=6),
           bad=st.lists(st.booleans(), min_size=6, max_size=6))
    def test_property_verdict_matches_definition(self, cur, drops, bad):
        gaps = [1, 2, 3, 5, 10, 20][:len(cur)]
        current = [row(g, 1.0, r) for g, (_, r) in zip(gaps, cur)]
        cand = [row(g, 0.75 if b else p, r - 0.1 if d else r)
                for g, (p, r), d, b in zip(gaps, cur, drops, bad)]
        check = compare.promotion_check(current, cand, 30)
        precision_ok = all(math.isnan(c['precision']) or c['precision'] == 1.0 for c in cand)
        recall_ok = not any(drops[:len(gaps)])
        self.assertEqual(check['precision_ok'], precision_ok)
        self.assertEqual(check['recall_ok'], recall_ok)
        self.assertEqual(check['ok'], precision_ok and recall_ok)
        self.assertEqual(len(check['failing']),
                         sum(not (math.isnan(c['precision']) or c['precision'] == 1.0)
                             for c in cand) + sum(drops[:len(gaps)]))


# ============================================================
# REAL SWEEPS
# ============================================================


@unittest.skipUnless((LOGS / bridge_gaps.BY_GAP_NAME).is_file()
                     and (LOGS / 'holdout_5' / holdout.TRUTH_NAME).is_file(),
                     'logs/gap_sweep not present')
class RealSweepTests(unittest.TestCase):
    """Fold 'all' re-scoring reproduces the committed seed-0 sweep's holdout_by_gap.csv."""

    def test_all_fold_matches_csv(self):
        expected = holdout.read_by_gap(LOGS / bridge_gaps.BY_GAP_NAME)
        got = compare.sweep_fold_rows(LOGS, 'all')
        self.assertEqual([{k: r[k] for k in ('max_gap', 'gap') + COUNTS} for r in got],
                         expected)


if __name__ == '__main__':
    unittest.main()
