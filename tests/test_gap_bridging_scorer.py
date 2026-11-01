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
from dataclasses import replace
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
