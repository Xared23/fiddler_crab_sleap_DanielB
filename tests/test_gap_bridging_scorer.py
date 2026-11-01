#!/usr/bin/env python3

"""
Tests for the learned link scorer: manifest, save/load, version guard, cache.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import contextlib
from dataclasses import replace
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

# ----- numerics -----
from hypothesis import given, settings, strategies as st
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

# ----- local modules -----
from gap_bridging import features, scorer  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

F = len(features.FEATURE_NAMES)
I_DOR = features.FEATURE_NAMES.index('dist_over_radius')

# ============================================================
# HELPERS
# ============================================================


def synthetic(n: int = 300, seed: int = 0):
    """``(X, y)``: 28 features, label = dist_over_radius < 0.5."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(n, F))
    y = (X[:, I_DOR] < 0.5).astype(int)
    return X, y


def manifest(threshold: float = 0.4, **overrides) -> scorer.ScorerManifest:
    """A valid manifest for the running scikit-learn."""
    import sklearn

    values = dict(
        sklearn_version=sklearn.__version__, feature_names=tuple(features.FEATURE_NAMES),
        monotonic_cst=tuple(features.MONOTONIC.get(n, 0) for n in features.FEATURE_NAMES),
        hyperparameters={'max_depth': 2, 'seed': 0}, calibration='isotonic',
        threshold=threshold, training_data_sha256='0' * 64, n_train=300, n_test=0,
        n_positive_train=150, n_positive_test=0,
        oof_metrics={'precision': 1.0, 'n': 300}, test_metrics={}, salt='s',
        test_fraction=0.4, sweeps=('logs/a', 'logs/b'), created='2026-11-01T00:00:00+00:00',
        git_commit='abc')
    values.update(overrides)
    return scorer.ScorerManifest(**values)


def fitted_scorer(seed: int = 0) -> scorer.LinkScorer:
    """A small fitted classifier + isotonic calibrator on :func:`synthetic`."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    X, y = synthetic(seed=seed)
    hgb = HistGradientBoostingClassifier(max_iter=10, max_depth=2, random_state=0).fit(X, y)
    cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(
        hgb.predict_proba(X)[:, 1], y)
    return scorer.LinkScorer(hgb, cal, manifest())


def fitted_subset_scorer(names, seed: int = 0) -> scorer.LinkScorer:
    """As :func:`fitted_scorer`, on the columns ``names`` (an ordered subset)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.isotonic import IsotonicRegression

    X_full, y = synthetic(seed=seed)
    X = features.select_features(X_full, names)
    hgb = HistGradientBoostingClassifier(max_iter=10, max_depth=2, random_state=0).fit(X, y)
    cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip').fit(
        hgb.predict_proba(X)[:, 1], y)
    return scorer.LinkScorer(hgb, cal, manifest(
        feature_names=tuple(names), monotonic_cst=features.monotonic_for(names),
        feature_set='no_length'))


# ============================================================
# TESTS
# ============================================================


class ManifestTests(unittest.TestCase):

    def test_json_round_trip(self):
        m = manifest()
        d = m.to_json()
        self.assertIsInstance(d['feature_names'], list)
        self.assertIsInstance(d['sweeps'], list)
        back = scorer.ScorerManifest.from_json(json.loads(json.dumps(d)))
        self.assertEqual(back, m)
        self.assertIsInstance(back.feature_names, tuple)
        self.assertIsInstance(back.monotonic_cst, tuple)

    def test_nan_metric_is_null_in_json_and_nan_back(self):
        m = manifest(test_metrics={'roc_auc': math.nan, 'n': 3})
        d = m.to_json()
        self.assertIsNone(d['test_metrics']['roc_auc'])
        json.dumps(d, allow_nan=False)
        back = scorer.ScorerManifest.from_json(d)
        self.assertTrue(math.isnan(back.test_metrics['roc_auc']))

    def test_feature_config_round_trip_and_old_manifest(self):
        fc = features.feature_config(BridgeConfig())
        m = manifest(feature_config=fc)
        back = scorer.ScorerManifest.from_json(json.loads(json.dumps(m.to_json())))
        self.assertEqual(back.feature_config, fc)
        old = manifest(feature_config=fc).to_json()
        del old['feature_config']                    # a manifest written before the field
        self.assertEqual(scorer.ScorerManifest.from_json(old).feature_config, {})

    def test_feature_set_round_trip_and_old_manifest(self):
        self.assertEqual(manifest().feature_set, 'all')
        names = features.FEATURE_SETS['no_length']
        m = manifest(feature_names=names, monotonic_cst=features.monotonic_for(names),
                     feature_set='no_length')
        d = json.loads(json.dumps(m.to_json()))
        self.assertEqual(d['feature_set'], 'no_length')
        back = scorer.ScorerManifest.from_json(d)
        self.assertEqual(back, m)
        self.assertEqual(back.feature_names, names)
        del d['feature_set']                            # a manifest written before v005
        self.assertEqual(scorer.ScorerManifest.from_json(d).feature_set, 'all')

    def test_unknown_key_rejected(self):
        d = manifest().to_json()
        d['bogus'] = 1
        with self.assertRaises(ValueError):
            scorer.ScorerManifest.from_json(d)


class ScorerTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.scorer = fitted_scorer()

    def test_protocol_properties(self):
        s = self.scorer
        self.assertEqual(s.threshold, 0.4)
        self.assertEqual(s.feature_names, features.FEATURE_NAMES)
        self.assertIs(s.manifest.threshold, 0.4)

    def test_predict_and_cost(self):
        X, y = synthetic(50, seed=3)
        X[::3, 0] = np.nan
        p = self.scorer.predict_proba(X)
        self.assertEqual(p.shape, (50,))
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        c = self.scorer.cost(X)
        self.assertTrue(np.all(np.isfinite(c)))
        np.testing.assert_allclose(c, -np.log(np.clip(p, 1e-12, 1.0)))
        self.assertGreater(p[y == 1].mean(), p[y == 0].mean())

    def test_wrong_width_rejected(self):
        with self.assertRaises(ValueError):
            self.scorer.predict_proba(np.zeros((2, F - 1)))
        with self.assertRaises(ValueError):
            self.scorer.cost(np.zeros(F))

    def test_empty_input(self):
        self.assertEqual(self.scorer.cost(np.zeros((0, F))).shape, (0,))

    @settings(deadline=None, max_examples=30)
    @given(st.lists(st.floats(0.0, 1.0), min_size=1, max_size=20))
    def test_property_cost_finite_nonnegative(self, values):
        X = np.tile(np.asarray(values)[:, None], (1, F))
        c = self.scorer.cost(X)
        self.assertTrue(np.all(np.isfinite(c)) and np.all(c >= 0))


class SigmoidCalibratorTests(unittest.TestCase):

    def test_fit_monotone_and_picklable(self):
        import pickle

        rng = np.random.default_rng(1)
        p = rng.uniform(0, 1, 400)
        y = (rng.uniform(0, 1, 400) < p).astype(int)
        cal = scorer.SigmoidCalibrator.fit(p, y)
        self.assertGreater(cal.coef, 0)
        grid = np.linspace(0, 1, 101)
        q = cal.predict(grid)
        self.assertTrue(np.all(np.diff(q) >= 0))
        self.assertTrue(np.all((q >= 0) & (q <= 1)))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(cal)).predict(grid), q)

    def test_anti_correlated_rejected(self):
        p = np.linspace(0, 1, 100)
        y = (p < 0.5).astype(int)
        with self.assertRaises(ValueError):
            scorer.SigmoidCalibrator.fit(p, y)


class PersistenceTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.path = self.tmp / 'model.joblib'
        self.scorer = fitted_scorer()
        scorer.save_scorer(self.scorer, self.path)

    def tearDown(self):
        self._tmp.cleanup()

    def _edit_manifest(self, **changes):
        meta = scorer.manifest_path(self.path)
        d = json.loads(meta.read_text())
        d.update(changes)
        meta.write_text(json.dumps(d))

    def test_manifest_path(self):
        self.assertEqual(scorer.manifest_path(Path('a/b.joblib')), Path('a/b.json'))

    def test_round_trip(self):
        loaded = scorer.load_scorer(self.path)
        self.assertEqual(loaded.manifest, self.scorer.manifest)
        X, _ = synthetic(40, seed=5)
        np.testing.assert_array_equal(loaded.predict_proba(X), self.scorer.predict_proba(X))
        text = scorer.manifest_path(self.path).read_text()
        self.assertEqual(json.loads(text), self.scorer.manifest.to_json())
        self.assertIn('\n  "', text)        # indent=2

    def test_save_refuses_overwrite_without_force(self):
        with self.assertRaises(FileExistsError):
            scorer.save_scorer(self.scorer, self.path)
        scorer.save_scorer(self.scorer, self.path, force=True)

    def test_missing_model(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            scorer.load_scorer(self.tmp / 'nope.joblib')
        self.assertIn('nope.joblib', str(ctx.exception))
        self.assertIn('bridge_gaps.py train', str(ctx.exception))

    def test_missing_manifest(self):
        scorer.manifest_path(self.path).unlink()
        with self.assertRaises(FileNotFoundError) as ctx:
            scorer.load_scorer(self.path)
        self.assertIn('bridge_gaps.py train', str(ctx.exception))

    def test_version_mismatch(self):
        import sklearn

        self._edit_manifest(sklearn_version='0.0.0')
        with self.assertRaises(scorer.ScorerVersionError) as ctx:
            scorer.load_scorer(self.path)
        message = str(ctx.exception)
        self.assertIn('0.0.0', message)
        self.assertIn(sklearn.__version__, message)
        self.assertIn('retrain', message)

    def test_wrong_feature_names(self):
        self._edit_manifest(feature_names=list(reversed(features.FEATURE_NAMES)))
        with self.assertRaises(ValueError) as ctx:
            scorer.load_scorer(self.path)
        self.assertNotIsInstance(ctx.exception, scorer.ScorerVersionError)
        self.assertIn('ordered subset', str(ctx.exception))

    def test_cached_scorer(self):
        scorer.cached_scorer.cache_clear()
        mtime = self.path.stat().st_mtime_ns
        a = scorer.cached_scorer(str(self.path), mtime)
        b = scorer.cached_scorer(str(self.path), mtime)
        self.assertIs(a, b)
        scorer.save_scorer(self.scorer, self.path, force=True)
        os.utime(self.path, ns=(mtime + 10**9, mtime + 10**9))
        c = scorer.cached_scorer(str(self.path), self.path.stat().st_mtime_ns)
        self.assertIsNot(a, c)

    def test_resolve_scorer(self):
        with self.assertRaises(ValueError):
            scorer.resolve_scorer(BridgeConfig())
        with self.assertRaises(ValueError):
            scorer.resolve_scorer(BridgeConfig(cost_model='learned'))
        with self.assertRaises(FileNotFoundError) as ctx:
            scorer.resolve_scorer(BridgeConfig(cost_model='learned',
                                               scorer_path=str(self.tmp / 'gone.joblib')))
        self.assertIn('bridge_gaps.py train', str(ctx.exception))
        cfg = BridgeConfig(cost_model='learned', scorer_path=str(self.path))
        s = scorer.resolve_scorer(cfg)
        self.assertIsInstance(s, scorer.LinkScorer)
        self.assertIs(scorer.resolve_scorer(replace(cfg)), s)


class FeatureConfigTests(unittest.TestCase):
    """Review finding 6: a scorer refuses a config that computes different features."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.base = BridgeConfig(cost_model='learned')
        fitted = fitted_scorer()
        self.path = self.tmp / 'model.joblib'
        tied = scorer.LinkScorer(fitted.hgb, fitted.calibrator, manifest(
            feature_config=features.feature_config(self.base)))
        scorer.save_scorer(tied, self.path)
        self.old_path = self.tmp / 'old.joblib'
        scorer.save_scorer(fitted, self.old_path)            # manifest without feature_config

    def cfg(self, path, **changes):
        return replace(self.base, scorer_path=str(path), **changes)

    def test_matching_config_accepted(self):
        # max_gap, the learned thresholds and the assignment are not feature keys.
        cfg = self.cfg(self.path, max_gap=60, learned_min_probability=0.9, assignment='global')
        self.assertIsInstance(scorer.resolve_scorer(cfg), scorer.LinkScorer)
        scorer.load_scorer(self.path, cfg)

    def test_mismatch_lists_keys(self):
        for changes in ({'gap_penalty': 0.2}, {'radius_k_px': 4.0, 'appearance_cues': False},
                        {'prediction_horizon_frames': 5}, {'shoulder_vector_cost_cap_px': 6.0}):
            cfg = self.cfg(self.path, **changes)
            with self.assertRaises(ValueError) as ctx:
                scorer.resolve_scorer(cfg)
            for key in changes:
                self.assertIn(key, str(ctx.exception))
            with self.assertRaises(ValueError):
                scorer.load_scorer(self.path, cfg)
        scorer.load_scorer(self.path)                         # no cfg: no check

    def test_old_manifest_warns_once(self):
        scorer._warn_no_feature_config.cache_clear()
        cfg = self.cfg(self.old_path, gap_penalty=0.2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            first = scorer.resolve_scorer(cfg)
            scorer.resolve_scorer(cfg)
        self.assertIsInstance(first, scorer.LinkScorer)
        self.assertEqual(buf.getvalue().count('WARNING'), 1)
        self.assertIn('feature_config', buf.getvalue())


class FeatureSubsetTests(unittest.TestCase):
    """v005: a scorer trained on an ordered subset of FEATURE_NAMES (e.g. no_length)."""

    NAMES = features.FEATURE_SETS['no_length']

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.scorer = fitted_subset_scorer(self.NAMES)
        self.path = self.tmp / 'subset.joblib'
        scorer.save_scorer(self.scorer, self.path)

    def test_column_indices(self):
        idx = self.scorer.column_indices
        self.assertEqual(idx, features.check_feature_subset(self.NAMES))
        self.assertEqual(tuple(features.FEATURE_NAMES[i] for i in idx), self.NAMES)
        self.assertEqual(fitted_scorer().column_indices, tuple(range(F)))

    def test_save_load_and_width(self):
        loaded = scorer.load_scorer(self.path)
        self.assertEqual(loaded.feature_names, self.NAMES)
        self.assertEqual(loaded.manifest.feature_set, 'no_length')
        self.assertEqual(loaded.manifest.monotonic_cst, features.monotonic_for(self.NAMES))
        self.assertEqual(loaded.hgb.n_features_in_, 20)
        X_full, _ = synthetic(40, seed=6)
        X = features.select_features(X_full, self.NAMES)
        self.assertEqual(X.shape, (40, 20))
        p = loaded.predict_proba(X)
        self.assertEqual(p.shape, (40,))
        np.testing.assert_array_equal(p, self.scorer.predict_proba(X))
        self.assertTrue(np.all(np.isfinite(loaded.cost(X))))
        with self.assertRaises(ValueError):
            loaded.predict_proba(X_full)                    # (n, 28) refused
        with self.assertRaises(ValueError):
            loaded.cost(X_full)

    def test_permuted_subset_refused_on_load(self):
        meta = scorer.manifest_path(self.path)
        d = json.loads(meta.read_text())
        names = list(self.NAMES)
        names[0], names[1] = names[1], names[0]
        d['feature_names'] = names
        meta.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'ordered subset') as ctx:
            scorer.load_scorer(self.path)
        self.assertIn('retrain', str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, scorer.ScorerVersionError)

    def test_unknown_name_refused_on_load(self):
        meta = scorer.manifest_path(self.path)
        d = json.loads(meta.read_text())
        d['feature_names'] = ['bogus'] + list(self.NAMES[1:])
        meta.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'bogus'):
            scorer.load_scorer(self.path)

    def test_manifest_disagreeing_with_model_width_fails(self):
        meta = scorer.manifest_path(self.path)
        d = json.loads(meta.read_text())
        d['feature_names'] = list(self.NAMES[:-1])
        d['monotonic_cst'] = list(features.monotonic_for(self.NAMES[:-1]))
        meta.write_text(json.dumps(d))
        with self.assertRaises(AssertionError):
            scorer.load_scorer(self.path)


class CommittedManifestTests(unittest.TestCase):
    """The committed v001 scorer manifest (written before feature_set) keeps loading."""

    PATH = ROOT / 'configs/tracking/link_scorer_v001.json'

    @unittest.skipUnless(PATH.is_file(), 'committed scorer manifest not present')
    def test_committed_manifest_parses_as_all(self):
        with open(self.PATH) as fh:
            d = json.load(fh)
        self.assertNotIn('feature_set', d)
        m = scorer.ScorerManifest.from_json(d)
        self.assertEqual(m.feature_names, features.FEATURE_NAMES)
        self.assertEqual(m.feature_set, 'all')
        self.assertTrue(m.feature_config)
        self.assertEqual(features.check_feature_subset(m.feature_names), tuple(range(F)))
        self.assertEqual(m.monotonic_cst, features.monotonic_for(m.feature_names))

    @unittest.skipUnless(PATH.with_suffix('.joblib').is_file(), 'committed scorer model not present')
    def test_committed_scorer_loads(self):
        import sklearn

        with open(self.PATH) as fh:
            version = json.load(fh)['sklearn_version']
        if version != sklearn.__version__:
            self.skipTest(f'manifest is for scikit-learn {version}, {sklearn.__version__} installed')
        loaded = scorer.load_scorer(self.PATH.with_suffix('.joblib'))
        self.assertEqual(loaded.feature_names, features.FEATURE_NAMES)
        self.assertEqual(loaded.column_indices, tuple(range(F)))
        self.assertEqual(loaded.manifest.feature_set, 'all')


class LazyImportTests(unittest.TestCase):

    def test_import_does_not_load_sklearn(self):
        code = ("import sys; sys.path.insert(0, 'scripts'); import gap_bridging.scorer; "
                "import gap_bridging.train; "
                "assert 'sklearn' not in sys.modules and 'joblib' not in sys.modules")
        result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True,
                                text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
