#!/usr/bin/env python3

"""
Tests for the pipeline-level threshold of the learned link scorer (threshold.py, v006).

Unit tests of the candidate grid and of the prefix-closed choice rule on
hand-built pooled curves, Hypothesis properties of both, and scene tests that
plant gaps in a small synthetic observation table, write a hold-out run
directory and run the real linker with a fake scorer whose probability
decreases with ``dist_over_radius``. A demo-gated smoke test runs one curve
point on a real horizon-30 hold-out run with the v005 S1 scorer.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import dataclasses
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from typing import Dict, List, Sequence, Tuple
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
from gap_bridging import features, holdout, labels, pipeline, threshold  # noqa: E402
from gap_bridging.config import BridgeConfig  # noqa: E402
from gap_bridging_fixtures import Crab, make_observations, straight_crab  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

GAPS = (1, 3, 5)
PER_GAP = 2
N_CRABS = 8
N_FRAMES = 60
HORIZON = 5
K_FAKE = 40.0            # fake scorer: p = 1 / (1 + exp(K (dist_over_radius - C)))
C_FAKE = 0.18            # gap 1 pairs: p ~ 0.88, gap 3: ~ 0.45, gap 5: ~ 0.11
GRID = (0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95)
COUNT_KEYS = ('planted', 'recovered', 'false_link', 'missed')
LEARNED_FIELDS = ('cost_model', 'scorer_path', 'learned_min_probability',
                  'learned_unmatched_probability')
DEMO_RUN = ROOT / 'logs/gap_sweep_v005/current_short/seed_0/holdout_30'
DEMO_SCORER = ROOT / 'configs/tracking/link_scorer_v005_s1.joblib'

# ============================================================
# HELPERS
# ============================================================


class FakeScorer:
    """Scorer stand-in: P(link) falls off logistically in ``dist_over_radius``."""

    def __init__(self, k: float = K_FAKE, c: float = C_FAKE) -> None:
        self.k, self.c = float(k), float(c)
        self.feature_names = tuple(features.FEATURE_NAMES)
        self.threshold = 0.5
        self._col = self.feature_names.index('dist_over_radius')

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability per row of the full feature matrix."""
        d = np.asarray(X, dtype=float)[:, self._col]
        return 1.0 / (1.0 + np.exp(self.k * (d - self.c)))

    def cost(self, X: np.ndarray) -> np.ndarray:
        """``-log p`` per row, clipped like ``LinkScorer.cost``."""
        return -np.log(np.clip(self.predict_proba(X), 1e-12, 1.0))


def far_crabs(n: int = N_CRABS, n_frames: int = N_FRAMES) -> List[Crab]:
    """``n`` slow straight crabs on a 600-px grid (no cross-crab candidate passes a gate)."""
    crabs = []
    for c in range(1, n + 1):
        angle = 0.7 * c
        velocity = (0.5 * math.cos(angle), 0.5 * math.sin(angle))
        origin = (300.0 + 600.0 * ((c - 1) % 5), 300.0 + 600.0 * ((c - 1) // 5))
        crabs.append(straight_crab(c, 0, n_frames, origin, velocity))
    return crabs


def write_run(run_dir: Path, planted: pa.Table, truth: pa.Table,
              cfg: BridgeConfig) -> Path:
    """Write a hold-out run directory as ``holdout.run_holdout`` lays it out."""
    run_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(planted, run_dir / holdout.PLANTED_NAME)
    pq.write_table(truth, run_dir / holdout.TRUTH_NAME)
    (run_dir / f'holdout_{cfg.max_gap}.json').write_text(
        json.dumps({'config': cfg.to_dict()}, indent=2))
    return run_dir


def poison_test_fold(truth: pa.Table) -> pa.Table:
    """Replace ``right_id`` of every test-fold truth row by an id no tracklet has."""
    folds = [labels.fold_of(o) for o in truth['original_id'].to_pylist()]
    rights = [f'wrong/{r}' if f == 'test' else r
              for r, f in zip(truth['right_id'].to_pylist(), folds)]
    i = truth.schema.get_field_index('right_id')
    return truth.set_column(i, 'right_id', pa.array(rights, pa.string()))


def pooled_table(spec: Dict[float, Dict[int, Tuple[int, int, int]]]) -> pa.Table:
    """POOLED_SCHEMA rows from ``{t: {gap: (planted, recovered, false_link)}}``."""
    rows = []
    for t in sorted(spec):
        for gap in sorted(spec[t]):
            planted, recovered, false_link = spec[t][gap]
            rows.append({'threshold': float(t), 'gap_frames': gap, 'planted': planted,
                         'recovered': recovered, 'false_link': false_link,
                         'missed': planted - recovered - false_link,
                         'precision': (recovered / (recovered + false_link)
                                       if recovered + false_link else math.nan),
                         'recall': recovered / planted if planted else math.nan})
    return pa.Table.from_pylist(rows, schema=threshold.POOLED_SCHEMA)


def rows_at(pooled: pa.Table, t: float) -> List[dict]:
    """Rows of ``pooled`` at threshold ``t``."""
    return pooled.filter(pc.equal(pooled['threshold'], t)).to_pylist()


def same_rows(a: Sequence[dict], b: Sequence[dict]) -> bool:
    """Row lists equal, NaN equal to NaN."""
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra.keys() != rb.keys():
            return False
        for key in ra:
            x, y = ra[key], rb[key]
            if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
                continue
            if x != y:
                return False
    return True


def counts_only(rows: Sequence[dict]) -> List[dict]:
    """The per-gap count columns of score or curve rows."""
    return [{'gap_frames': int(r['gap_frames']), **{k: int(r[k]) for k in COUNT_KEYS}}
            for r in rows]


# ----- hypothesis strategies -----

@st.composite
def pooled_curves(draw) -> Tuple[pa.Table, int]:
    """A random pooled curve over 1-6 thresholds and gaps 1-4, plus ``through_gap``."""
    ts = draw(st.lists(st.floats(0.01, 0.99, allow_nan=False), min_size=1, max_size=6,
                       unique=True))
    through_gap = draw(st.integers(1, 3))
    spec: Dict[float, Dict[int, Tuple[int, int, int]]] = {}
    for t in ts:
        spec[t] = {}
        for gap in (1, 2, 3, 4):
            planted = draw(st.integers(1, 4))
            recovered = draw(st.sampled_from([0, planted, planted, planted]))
            false_link = draw(st.sampled_from([0, 0, 0, planted - recovered]))
            spec[t][gap] = (planted, recovered, min(false_link, planted - recovered))
    return pooled_table(spec), through_gap


# ============================================================
# CANDIDATE THRESHOLDS
# ============================================================


class TestCandidateThresholds(unittest.TestCase):
    def test_sorted_unique_inside_unit_interval(self) -> None:
        p = np.array([0.7, 0.2, 0.7, 0.0, 1.0, 0.4, 0.9])
        y = np.array([1, 0, 0, 0, 1, 1, -1])
        np.testing.assert_array_equal(threshold.candidate_thresholds(p, y), [0.2, 0.4, 0.7])

    def test_cap_keeps_extremes(self) -> None:
        p = np.linspace(0.01, 0.99, 50)
        y = np.ones(50, dtype=int)
        out = threshold.candidate_thresholds(p, y, max_candidates=7)
        self.assertLessEqual(out.size, 7)
        self.assertEqual(out[0], p[0])
        self.assertEqual(out[-1], p[-1])
        self.assertTrue(np.all(np.diff(out) > 0))
        self.assertEqual(threshold.candidate_thresholds(p, y, max_candidates=50).size, 50)

    def test_errors(self) -> None:
        with self.assertRaises(ValueError):
            threshold.candidate_thresholds(np.array([0.0, 1.0]), np.array([0, 1]))
        with self.assertRaises(ValueError):
            threshold.candidate_thresholds(np.array([0.5]), np.array([2]))
        with self.assertRaises(ValueError):
            threshold.candidate_thresholds(np.array([0.3, 0.5]), np.array([0, 1]),
                                           max_candidates=1)
        with self.assertRaises(ValueError):
            threshold.candidate_thresholds(np.array([0.3, 0.5]), np.array([0, 1, 1]))

    @settings(max_examples=200, deadline=None)
    @given(st.lists(st.tuples(st.sampled_from([0.0, 1.0, 0.25, 0.5])
                              | st.floats(0.0, 1.0, allow_nan=False),
                              st.integers(-1, 2)), min_size=1, max_size=80),
           st.integers(2, 12))
    def test_properties(self, pairs: List[Tuple[float, int]], cap: int) -> None:
        p = np.array([a for a, _ in pairs], dtype=float)
        y = np.array([b for _, b in pairs], dtype=np.int64)
        eligible = np.unique(p[np.isin(y, (0, 1)) & (p > 0) & (p < 1)])
        if eligible.size == 0:
            with self.assertRaises(ValueError):
                threshold.candidate_thresholds(p, y, cap)
            return
        out = threshold.candidate_thresholds(p, y, cap)
        self.assertTrue(set(out.tolist()) <= set(eligible.tolist()))
        self.assertTrue(np.all(np.diff(out) > 0))
        self.assertLessEqual(out.size, cap)
        self.assertEqual(out.size, min(cap, eligible.size))
        self.assertEqual(out[0], eligible[0])
        self.assertEqual(out[-1], eligible[-1])


# ============================================================
# CHOICE RULE
# ============================================================


class TestChoiceRule(unittest.TestCase):
    def test_qualifies(self) -> None:
        good = {1: (3, 3, 0), 2: (3, 1, 0)}
        rows = rows_at(pooled_table({0.5: good}), 0.5)
        self.assertTrue(threshold.qualifies(rows, 2))
        rows = rows_at(pooled_table({0.5: {**good, 2: (3, 0, 0)}}), 0.5)
        self.assertFalse(threshold.qualifies(rows, 2))      # recovered 0, false 0
        rows = rows_at(pooled_table({0.5: {**good, 2: (3, 2, 1)}}), 0.5)
        self.assertFalse(threshold.qualifies(rows, 2))
        self.assertTrue(threshold.qualifies(rows, 1))       # gap 2 beyond through_gap
        self.assertFalse(threshold.qualifies(rows, 0))      # no row up to through_gap
        self.assertFalse(threshold.qualifies([], 30))

    def test_smallest_qualifying_wins(self) -> None:
        pooled = pooled_table({0.2: {1: (4, 4, 0), 2: (4, 3, 1)},
                               0.3: {1: (4, 4, 0), 2: (4, 3, 0)},
                               0.6: {1: (4, 3, 0), 2: (4, 2, 0)},
                               0.9: {1: (4, 1, 0), 2: (4, 1, 0)}})
        self.assertEqual(threshold.choose_pipeline_threshold(pooled, 2), 0.3)

    def test_dip_is_prefix_closed_from_top(self) -> None:
        pooled = pooled_table({0.3: {1: (4, 4, 0)}, 0.4: {1: (4, 3, 1)},
                               0.5: {1: (4, 3, 0)}, 0.7: {1: (4, 2, 0)}})
        self.assertEqual(threshold.choose_pipeline_threshold(pooled, 1), 0.5)

    def test_vanishing_recall_above_the_choice_does_not_block(self) -> None:
        """The highest candidates recover nothing at some gap (precision undefined) but make
        no false link: they are clean, not qualifying, and the choice is the smallest
        qualifying candidate below them."""
        pooled = pooled_table({0.2: {1: (4, 4, 0), 2: (4, 3, 1)},
                               0.3: {1: (4, 4, 0), 2: (4, 3, 0)},
                               0.9: {1: (4, 0, 0), 2: (4, 1, 0)},
                               0.99: {1: (4, 0, 0), 2: (4, 0, 0)}})
        self.assertEqual(threshold.choose_pipeline_threshold(pooled, 2), 0.3)
        self.assertTrue(threshold.clean(rows_at(pooled, 0.99), 2))
        self.assertFalse(threshold.qualifies(rows_at(pooled, 0.99), 2))
        self.assertFalse(threshold.clean(rows_at(pooled, 0.2), 2))
        # a false link above a qualifying candidate still blocks it
        pooled = pooled_table({0.3: {1: (4, 4, 0)}, 0.9: {1: (4, 1, 1)}, 0.99: {1: (4, 0, 0)}})
        with self.assertRaisesRegex(ValueError, r'false-linking candidate 0\.9'):
            threshold.choose_pipeline_threshold(pooled, 1)

    def test_gaps_beyond_through_gap_ignored(self) -> None:
        pooled = pooled_table({0.2: {1: (4, 4, 0), 45: (4, 0, 4)},
                               0.8: {1: (4, 2, 0), 45: (4, 0, 2)}})
        self.assertEqual(threshold.choose_pipeline_threshold(pooled, 30), 0.2)

    def test_no_qualifying_top_names_failing_gap(self) -> None:
        pooled = pooled_table({0.2: {1: (4, 4, 0), 3: (4, 2, 2), 5: (4, 1, 1)},
                               0.8: {1: (4, 2, 0), 3: (4, 3, 1), 5: (4, 0, 0)}})
        with self.assertRaisesRegex(ValueError, r'gap_frames 3\b'):
            threshold.choose_pipeline_threshold(pooled, 5)
        pooled = pooled_table({0.8: {1: (4, 2, 0), 3: (4, 0, 0)}})
        with self.assertRaisesRegex(ValueError, r'gap_frames 3\b'):
            threshold.choose_pipeline_threshold(pooled, 30)
        pooled = pooled_table({0.2: {1: (4, 4, 0), 3: (4, 2, 2)}, 0.8: {1: (4, 2, 0), 3: (4, 0, 0)}})
        with self.assertRaisesRegex(ValueError, r'false-linking candidate 0\.2.*gap_frames 3\b'):
            threshold.choose_pipeline_threshold(pooled, 30)

    def test_summary(self) -> None:
        pooled = pooled_table({0.2: {1: (4, 4, 0), 3: (4, 2, 1), 45: (4, 0, 4)},
                               0.5: {1: (4, 3, 0), 3: (4, 2, 0), 45: (4, 0, 1)},
                               0.9: {1: (4, 1, 0), 3: (4, 1, 0), 45: (4, 0, 0)}})
        chosen = threshold.choose_pipeline_threshold(pooled, 30)
        out = threshold.summary(pooled, chosen, 30, pair_threshold=0.6)
        self.assertEqual(out, {'n_candidates': 3, 'chosen': 0.5, 'through_gap': 30,
                               'at_chosen': {'planted': 8, 'recovered': 5, 'false_link': 0,
                                             'missed': 3},
                               'at_pair_threshold': {'planted': 8, 'recovered': 2,
                                                     'false_link': 0, 'missed': 6}})
        json.dumps(out)
        self.assertIsNone(threshold.summary(pooled, chosen, 30)['at_pair_threshold'])
        self.assertIsNone(threshold.summary(pooled, chosen, 30, 0.95)['at_pair_threshold'])
        self.assertEqual(threshold.summary(pooled, chosen, 30, 0.5)['at_pair_threshold'],
                         out['at_chosen'])
        self.assertIsInstance(out['chosen'], float)
        self.assertIsInstance(out['at_chosen']['planted'], int)

    @settings(max_examples=200, deadline=None)
    @given(pooled_curves(), st.floats(0.0, 1.0, exclude_min=True, exclude_max=True))
    def test_choice_properties(self, curve: Tuple[pa.Table, int], extra: float) -> None:
        pooled, through_gap = curve
        candidates = sorted(set(pooled['threshold'].to_pylist()))
        ok = {t: threshold.qualifies(rows_at(pooled, t), through_gap) for t in candidates}
        cl = {t: threshold.clean(rows_at(pooled, t), through_gap) for t in candidates}
        # reference: the smallest qualifying candidate above the largest unclean one
        unclean = [t for t in candidates if not cl[t]]
        barrier = unclean[-1] if unclean else -1.0
        expected = [t for t in candidates if t > barrier and ok[t]]
        if not expected:
            with self.assertRaises(ValueError):
                threshold.choose_pipeline_threshold(pooled, through_gap)
            return
        chosen = threshold.choose_pipeline_threshold(pooled, through_gap)
        self.assertEqual(chosen, expected[0])
        self.assertIn(chosen, candidates)
        self.assertTrue(ok[chosen])
        self.assertTrue(all(cl[t] for t in candidates if t >= chosen))
        # removing every candidate below the barrier (the barrier itself kept) changes nothing
        below_barrier = [t for t in candidates if t < barrier]
        if below_barrier:
            kept = pooled.filter(pc.invert(pc.is_in(pooled['threshold'],
                                                    value_set=pa.array(below_barrier))))
            self.assertEqual(threshold.choose_pipeline_threshold(kept, through_gap), chosen)
        # removing clean non-qualifying candidates between the barrier and the choice too
        idle = [t for t in candidates if barrier < t < chosen and not ok[t]]
        if idle:
            kept = pooled.filter(pc.invert(pc.is_in(pooled['threshold'], value_set=pa.array(idle))))
            self.assertEqual(threshold.choose_pipeline_threshold(kept, through_gap), chosen)
        # adding a false-linking candidate below the choice changes nothing
        if extra < chosen and extra not in candidates:
            failing = pooled_table({extra: {1: (2, 1, 1), 2: (2, 1, 1), 3: (2, 1, 1)}})
            grown = pa.concat_tables([failing, pooled])
            self.assertEqual(threshold.choose_pipeline_threshold(grown, through_gap), chosen)
        # adding a clean, non-recovering candidate above everything changes nothing
        top = pooled_table({0.999999: {1: (2, 0, 0), 2: (2, 0, 0), 3: (2, 0, 0)}})
        if 0.999999 not in candidates:
            grown = pa.concat_tables([pooled, top])
            self.assertEqual(threshold.choose_pipeline_threshold(grown, through_gap), chosen)


# ============================================================
# SCENES
# ============================================================


class TestScenes(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        obs = make_observations(far_crabs())
        cls.planted, cls.truth = holdout.plant_gaps(obs, gap_lengths=GAPS, per_gap=PER_GAP,
                                                    min_tracklet_len=10, seed=0,
                                                    profile='standard')
        cls.cfg = BridgeConfig(max_gap=HORIZON)
        cls.run_dir = write_run(tmp / f'holdout_{HORIZON}', cls.planted, cls.truth, cls.cfg)
        cls.poisoned_dir = write_run(tmp / 'poisoned' / f'holdout_{HORIZON}', cls.planted,
                                     poison_test_fold(cls.truth), cls.cfg)
        cls.scene = threshold.load_scene(cls.run_dir)
        cls.scorer = FakeScorer()
        cls.tmp = tmp

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_load_scene(self) -> None:
        scene = self.scene
        self.assertEqual(scene.run_dir, self.run_dir)
        self.assertEqual(scene.cfg, self.cfg)
        self.assertIsNone(scene.cfg.scorer_path)
        self.assertEqual(scene.videos, ['synthetic'])
        self.assertEqual(set(scene.endpoints), set(pc.unique(
            self.planted['tracklet_id'].drop_null()).to_pylist()))
        self.assertEqual(scene.n_truth, self.truth.num_rows)
        self.assertEqual(scene.truth_train.schema.names, list(holdout.TRUTH_SCHEMA.names))
        test_ids = {o for o in self.truth['original_id'].to_pylist()
                    if labels.fold_of(o) == 'test'}
        train_ids = set(self.truth['original_id'].to_pylist()) - test_ids
        self.assertTrue(test_ids and train_ids, "the scene must have both folds")
        self.assertEqual(set(scene.truth_train['original_id'].to_pylist()), train_ids)

    def test_load_scene_forces_scorer_path_none(self) -> None:
        cfg = dataclasses.replace(self.cfg, scorer_path='/nowhere/model.joblib')
        run = write_run(self.tmp / 'with_scorer' / f'holdout_{HORIZON}', self.planted,
                        self.truth, cfg)
        self.assertIsNone(threshold.load_scene(run).cfg.scorer_path)

    def test_load_scene_refusals(self) -> None:
        linear = BridgeConfig(max_gap=HORIZON, gate_model='linear')
        run = write_run(self.tmp / 'linear' / f'holdout_{HORIZON}', self.planted, self.truth,
                        linear)
        with self.assertRaisesRegex(ValueError, 'linear'):
            threshold.load_scene(run)
        with self.assertRaises(FileNotFoundError):
            threshold.load_scene(self.tmp / 'missing')
        empty = self.tmp / 'empty'
        empty.mkdir(exist_ok=True)
        with self.assertRaisesRegex(FileNotFoundError, 'empty'):
            threshold.load_scene(empty)

    def test_scene_config(self) -> None:
        cfg = threshold.scene_config(self.cfg, 0.3)
        self.assertEqual((cfg.cost_model, cfg.scorer_path, cfg.learned_min_probability,
                          cfg.learned_unmatched_probability), ('learned', None, 0.3, 0.3))
        for bad in (0.0, 1.0, -0.2, 1.5, math.nan):
            with self.assertRaises(ValueError):
                threshold.scene_config(self.cfg, bad)

    def test_curve_point_equals_direct_call(self) -> None:
        for t in (0.3, 0.6):
            cfg = dataclasses.replace(self.cfg, cost_model='learned', scorer_path=None,
                                      learned_min_probability=t,
                                      learned_unmatched_probability=t)
            _, links = pipeline.link_videos(self.scene.endpoints, self.scene.videos, cfg,
                                            scorer=self.scorer)
            scores = holdout.score_links(links, self.scene.truth_train, None).to_pylist()
            direct = counts_only([r for r in scores if r['gap_frames'] >= 1])
            got = threshold.curve_point(self.scene, self.scorer, t)
            self.assertEqual(got, direct)
            self.assertEqual([r['gap_frames'] for r in got], sorted(set(
                self.scene.truth_train['gap_frames'].to_pylist())))

    def test_curve_point_config_tied_to_threshold(self) -> None:
        seen: List[BridgeConfig] = []
        real = pipeline.link_videos

        def spy(endpoints, videos, cfg, stats=None, scorer=None):
            seen.append(cfg)
            self.assertIs(scorer, self.scorer)
            return real(endpoints, videos, cfg, stats=stats, scorer=scorer)

        with mock.patch.object(pipeline, 'link_videos', side_effect=spy):
            threshold.curve_point(self.scene, self.scorer, 0.37)
        self.assertEqual(len(seen), 1)
        cfg = seen[0]
        self.assertEqual(cfg.cost_model, 'learned')
        self.assertIsNone(cfg.scorer_path)
        self.assertEqual(cfg.learned_min_probability, 0.37)
        self.assertEqual(cfg.learned_unmatched_probability, 0.37)
        got, base = cfg.to_dict(), self.scene.cfg.to_dict()
        for key in LEARNED_FIELDS:
            got.pop(key)
            base.pop(key)
        self.assertEqual(got, base)

    def test_train_fold_only(self) -> None:
        poisoned = threshold.load_scene(self.poisoned_dir)
        self.assertEqual(poisoned.truth_train.to_pylist(), self.scene.truth_train.to_pylist())
        clean = threshold.pipeline_curve([self.scene], self.scorer, GRID)
        dirty = threshold.pipeline_curve([poisoned], self.scorer, GRID)
        self.assertEqual(clean.drop_columns(['run']).to_pylist(),
                         dirty.drop_columns(['run']).to_pylist())
        test_ids = {o for o in self.truth['original_id'].to_pylist()
                    if labels.fold_of(o) == 'test'}
        self.assertFalse(test_ids & set(poisoned.truth_train['original_id'].to_pylist()))
        # the poison would show if test rows were scored: their right halves never link
        full = holdout.score_links(pipeline.link_videos(
            poisoned.endpoints, poisoned.videos, threshold.scene_config(poisoned.cfg, 0.05),
            scorer=self.scorer)[1], poison_test_fold(self.truth), None).to_pylist()
        self.assertGreater(full[-1]['false_link'], 0)

    def test_curve_point_drops_gaps_without_plants(self) -> None:
        """Integration item 3: a train-fold gap holding only a decoy donor (its target in
        the other fold) has ``planted == 0`` in ``score_links`` and is dropped, so it can
        never fail the rule for lack of evidence."""
        donor_gap = max(GAPS) + 10
        extra = {'left_id': 'z_l', 'right_id': 'z_r', 'original_id': 'z', 'gap_frames': donor_gap,
                 'deleted_first_frame': 1, 'deleted_last_frame': donor_gap, 'role': 'decoy_donor',
                 'middle_id': None, 'partner_id': 'absent', 'profile': 'standard',
                 'category': None, 'left_frames': 1, 'right_frames': 1}
        rows = self.scene.truth_train.to_pylist() + [extra]
        truth = pa.Table.from_pylist(rows, schema=self.scene.truth_train.schema)
        scene = dataclasses.replace(self.scene, truth_train=truth)
        direct = holdout.score_links(pipeline.link_videos(
            scene.endpoints, scene.videos, threshold.scene_config(scene.cfg, 0.5),
            scorer=self.scorer)[1], truth, None).to_pylist()
        self.assertIn(donor_gap, [r['gap_frames'] for r in direct if r['planted'] == 0])
        point = threshold.curve_point(scene, self.scorer, 0.5)
        self.assertNotIn(donor_gap, [r['gap_frames'] for r in point])
        self.assertTrue(all(r['planted'] > 0 for r in point))
        self.assertEqual(point, threshold.curve_point(self.scene, self.scorer, 0.5))

    def test_crossfit_scenes_partition_and_scorer_list(self) -> None:
        """v006 review finding 1: fold scenes partition the train-fold truth, keep endpoints
        and config, name the fold in run_dir, and pipeline_curve pairs each with its scorer;
        the pooled curve of the parts equals the whole scene's when every part uses the same
        scorer."""
        originals = sorted(set(self.scene.truth_train['original_id'].to_pylist()))
        fold_of = {o: i % 2 for i, o in enumerate(originals)}
        parts = threshold.crossfit_scenes(self.scene, fold_of, 2)
        self.assertEqual(len(parts), 2)
        self.assertEqual(sum(p.truth_train.num_rows for p in parts),
                         self.scene.truth_train.num_rows)
        for k, part in enumerate(parts):
            self.assertEqual(str(part.run_dir), f'{self.scene.run_dir}#fold{k}')
            self.assertIs(part.endpoints, self.scene.endpoints)
            self.assertEqual(part.cfg, self.scene.cfg)
            self.assertTrue(all(fold_of[o] == k for o in part.truth_train['original_id'].to_pylist()))
        whole = threshold.pool_curve(threshold.pipeline_curve([self.scene], self.scorer, GRID))
        split = threshold.pool_curve(threshold.pipeline_curve(parts, [self.scorer] * 2, GRID))
        self.assertEqual(counts_only(whole.to_pylist()), counts_only(split.to_pylist()))
        # unknown originals go to fold 0; an empty fold is skipped; bad inputs refused
        only = threshold.crossfit_scenes(self.scene, {}, 3)
        self.assertEqual([str(p.run_dir)[-5:] for p in only], ['fold0'])
        with self.assertRaises(ValueError):
            threshold.crossfit_scenes(self.scene, {originals[0]: 5}, 2)
        with self.assertRaises(ValueError):
            threshold.crossfit_scenes(self.scene, fold_of, 0)
        with self.assertRaises(ValueError):
            threshold.pipeline_curve(parts, [self.scorer], GRID)   # one scorer for two scenes
        # a different scorer per fold changes only that fold's rows
        strict = FakeScorer(c=0.0)                       # links nothing
        mixed = threshold.pipeline_curve(parts, [self.scorer, strict], GRID)
        same = threshold.pipeline_curve(parts, [self.scorer, self.scorer], GRID)

        def fold(t: pa.Table, k: int) -> List[dict]:
            return [r for r in t.to_pylist() if r['run'].endswith(f'fold{k}')]
        self.assertEqual(fold(mixed, 0), fold(same, 0))
        self.assertNotEqual(fold(mixed, 1), fold(same, 1))
        self.assertTrue(all(r['recovered'] == 0 for r in fold(mixed, 1)))

    def test_scene_config_refuses_adaptive_dummy(self) -> None:
        cfg = dataclasses.replace(self.scene.cfg, unmatched_cost_mode='adaptive')
        with self.assertRaisesRegex(ValueError, 'adaptive'):
            threshold.scene_config(cfg, 0.5)

    def test_pipeline_curve_rows(self) -> None:
        scenes = [self.scene, threshold.load_scene(self.poisoned_dir)]
        messages: List[str] = []
        curve = threshold.pipeline_curve(scenes, self.scorer, GRID, progress=messages.append)
        self.assertTrue(curve.schema.equals(threshold.CURVE_SCHEMA))
        n_gaps = len(set(self.scene.truth_train['gap_frames'].to_pylist()))
        self.assertEqual(curve.num_rows, len(GRID) * len(scenes) * n_gaps)
        self.assertEqual(len(messages), len(GRID))
        rows = curve.to_pylist()
        k = 0
        for t in GRID:
            for scene in scenes:
                expected = threshold.curve_point(scene, self.scorer, t)
                for row in expected:
                    self.assertEqual(rows[k], {'threshold': t, 'run': str(scene.run_dir), **row})
                    k += 1
        self.assertEqual(k, curve.num_rows)
        with self.assertRaises(ValueError):
            threshold.pipeline_curve([], self.scorer, GRID)
        with self.assertRaises(ValueError):
            threshold.pipeline_curve(scenes, self.scorer, [])

    def test_pool_curve_sums(self) -> None:
        scenes = [self.scene, threshold.load_scene(self.poisoned_dir)]
        curve = threshold.pipeline_curve(scenes, self.scorer, GRID)
        pooled = threshold.pool_curve(curve)
        self.assertTrue(pooled.schema.equals(threshold.POOLED_SCHEMA))
        keys = [(r['threshold'], r['gap_frames']) for r in pooled.to_pylist()]
        self.assertEqual(keys, sorted(set((r['threshold'], r['gap_frames'])
                                          for r in curve.to_pylist())))
        for row in pooled.to_pylist():
            part = [r for r in curve.to_pylist()
                    if (r['threshold'], r['gap_frames']) == (row['threshold'], row['gap_frames'])]
            for key in COUNT_KEYS:
                self.assertEqual(row[key], sum(r[key] for r in part))
        for key in COUNT_KEYS:
            self.assertEqual(sum(pooled[key].to_pylist()), sum(curve[key].to_pylist()))

    def test_pool_curve_ratios(self) -> None:
        curve = pa.Table.from_pylist([
            {'threshold': 0.4, 'run': 'a', 'gap_frames': 1, 'planted': 2, 'recovered': 1,
             'false_link': 0, 'missed': 1},
            {'threshold': 0.4, 'run': 'b', 'gap_frames': 1, 'planted': 2, 'recovered': 0,
             'false_link': 1, 'missed': 1},
            {'threshold': 0.4, 'run': 'a', 'gap_frames': 2, 'planted': 2, 'recovered': 0,
             'false_link': 0, 'missed': 2},
            {'threshold': 0.4, 'run': 'b', 'gap_frames': 3, 'planted': 0, 'recovered': 0,
             'false_link': 0, 'missed': 0},
        ], schema=threshold.CURVE_SCHEMA)
        rows = threshold.pool_curve(curve).to_pylist()
        self.assertEqual([r['gap_frames'] for r in rows], [1, 2, 3])
        self.assertEqual((rows[0]['planted'], rows[0]['recovered'], rows[0]['false_link']),
                         (4, 1, 1))
        self.assertEqual((rows[0]['precision'], rows[0]['recall']), (0.5, 0.25))
        self.assertTrue(math.isnan(rows[1]['precision']))
        self.assertEqual(rows[1]['recall'], 0.0)
        self.assertTrue(math.isnan(rows[2]['precision']) and math.isnan(rows[2]['recall']))

    def test_curve_paths(self) -> None:
        model = Path('/x/models/link_scorer_x.joblib')
        self.assertEqual(threshold.curve_paths(model),
                         (Path('/x/models/link_scorer_x_threshold_curve.csv'),
                          Path('/x/models/link_scorer_x_threshold_curve_runs.csv')))

    def test_write_read_round_trip(self) -> None:
        curve = threshold.pipeline_curve([self.scene], self.scorer, GRID)
        pooled = threshold.pool_curve(curve)
        model = self.tmp / 'models' / 'link_scorer_x.joblib'
        model.parent.mkdir(exist_ok=True)
        pooled_path, runs_path = threshold.write_curve(curve, pooled, model)
        self.assertEqual(pooled_path, model.with_name('link_scorer_x' + threshold.CURVE_SUFFIX))
        self.assertEqual(runs_path,
                         model.with_name('link_scorer_x' + threshold.CURVE_RUNS_SUFFIX))
        self.assertEqual((pooled_path, runs_path), threshold.curve_paths(model))
        back_pooled, back_runs = threshold.read_curve(pooled_path), threshold.read_curve(runs_path)
        self.assertTrue(back_pooled.schema.equals(threshold.POOLED_SCHEMA))
        self.assertTrue(back_runs.schema.equals(threshold.CURVE_SCHEMA))
        self.assertTrue(same_rows(back_pooled.to_pylist(), pooled.to_pylist()))
        self.assertTrue(same_rows(back_runs.to_pylist(), curve.to_pylist()))
        header = pooled_path.read_text().splitlines()[0]
        self.assertEqual(header, ','.join(threshold.POOLED_SCHEMA.names))

    def test_recovered_monotone_at_gap_one(self) -> None:
        # Crabs are 600 px apart, so every planted END has exactly one gated
        # candidate START (its own right half) and vice versa: each assignment
        # component is a single pair, linked iff p >= t. Lowering t can only
        # admit that pair, never displace it, so recovered is non-decreasing
        # as t decreases (and false links stay at 0).
        recovered = []
        for t in sorted(GRID, reverse=True):
            rows = {r['gap_frames']: r for r in threshold.curve_point(self.scene, self.scorer, t)}
            recovered.append(rows[1]['recovered'])
            self.assertTrue(all(r['false_link'] == 0 for r in rows.values()))
        self.assertTrue(all(b >= a for a, b in zip(recovered, recovered[1:])))
        self.assertLess(recovered[0], recovered[-1], "the grid must span the gap-1 probability")


# ============================================================
# DEMO SMOKE
# ============================================================


@unittest.skipUnless(DEMO_RUN.is_dir() and DEMO_SCORER.is_file(),
                     'v005 short hold-out run or S1 scorer not present')
class TestDemoSmoke(unittest.TestCase):
    def test_curve_point_on_real_run(self) -> None:
        from gap_bridging import scorer as scorer_mod
        scene = threshold.load_scene(DEMO_RUN)
        model = scorer_mod.load_scorer(DEMO_SCORER, scene.cfg)
        started = time.perf_counter()
        rows = threshold.curve_point(scene, model, 0.5)
        elapsed = time.perf_counter() - started
        roles = scene.truth_train['role'].to_pylist()
        n_plants = sum(r in ('plant', 'decoy_target') for r in roles)
        self.assertEqual(sum(r['planted'] for r in rows), n_plants)
        self.assertGreater(n_plants, 0)
        self.assertLess(elapsed, 5.0)


# ============================================================
# __main__
# ============================================================

if __name__ == '__main__':
    unittest.main()
