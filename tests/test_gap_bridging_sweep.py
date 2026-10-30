"""Tests for gap-bridging WP-D: radius fit, horizon sweep, hold-out CSVs and figures."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

from hypothesis import given, settings, strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

import bridge_gaps  # noqa: E402
from gap_bridging import figures, fit, pipeline  # noqa: E402
from gap_bridging.config import BridgeConfig, load_config  # noqa: E402
from gap_bridging_fixtures import make_observations, straight_crab  # noqa: E402

DEMO_DIR = ROOT / 'demo/crab_cam2_2_middle_300'
DEMO = DEMO_DIR / 'observations.parquet'
VIDEO = DEMO_DIR / 'Crab_Cam_2_2.frames_014250_014549.mp4'
YAML = ROOT / 'configs/tracking/gap_bridging_v001.yaml'


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def track_table(tracks: list[np.ndarray], frames: list[np.ndarray] | None = None) -> pa.Table:
    """Minimal observation table: one accepted tracklet per (n,2) position array."""
    columns = {'tracklet_id': [], 'frame': [], 'accepted_for_tracking': [], 'body_x': [],
               'body_y': []}
    for i, xy in enumerate(tracks):
        f = np.arange(len(xy)) if frames is None else frames[i]
        columns['tracklet_id'] += [f't{i:04d}'] * len(xy)
        columns['frame'] += [int(v) for v in f]
        columns['accepted_for_tracking'] += [True] * len(xy)
        columns['body_x'] += xy[:, 0].tolist()
        columns['body_y'] += xy[:, 1].tolist()
    return pa.table(columns)


def read_csv(path: Path) -> list[dict]:
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def sweep_scene() -> pa.Table:
    """Six slow, far-apart straight crabs over 60 frames; crab 1 misses frames 20-23."""
    crabs = [straight_crab(c, 0, 60, (300.0 + 600.0 * c, 400.0 + 300.0 * (c % 2)),
                           (0.4 * (-1) ** c, 0.3)) for c in range(1, 7)]
    return make_observations(crabs, drop_frames={1: [20, 21, 22, 23]})


def nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
class FitTests(unittest.TestCase):
    """Displacement quantiles and the r0 + k sqrt(dt) fit."""

    @unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
    def test_demo_fit_reproduces_committed_yaml(self):
        quantiles = fit.displacement_quantiles(pq.read_table(DEMO))
        result = fit.fit_radius(quantiles)
        cfg = load_config(YAML)
        self.assertEqual(quantiles.num_rows, 60)
        self.assertEqual(round(result['radius_r0_px'], 2), cfg.radius_r0_px)
        self.assertEqual(round(result['radius_k_px'], 2), cfg.radius_k_px)
        self.assertEqual(round(result['radius_k_px'], 2), BridgeConfig().radius_k_px)

    @unittest.skipUnless(DEMO.exists(), 'demo parquet not present')
    def test_fit_command_writes_loadable_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            overlay = Path(tmp) / 'radius.yaml'
            code = bridge_gaps.main(['fit', '--observations', str(DEMO), '--write-yaml',
                                     str(overlay)])
            self.assertEqual(code, 0)
            cfg = load_config(overlay)
        self.assertEqual((cfg.radius_r0_px, cfg.radius_k_px), (0.0, 3.73))

    def test_diffusive_crabs_recover_k(self):
        # 2D Gaussian steps with per-axis sigma: |X(dt)| is Rayleigh with scale
        # sigma sqrt(dt), so its 99th percentile is sigma sqrt(-2 ln 0.01) sqrt(dt).
        sigma, rng = 1.5, np.random.default_rng(3)
        tracks = [np.cumsum(rng.normal(0.0, sigma, (400, 2)), axis=0) for _ in range(60)]
        result = fit.fit_radius(fit.displacement_quantiles(track_table(tracks), max_dt=30))
        k_true = sigma * math.sqrt(-2.0 * math.log(0.01))
        self.assertLess(abs(result['radius_k_px'] - k_true) / k_true, 0.10)
        self.assertLess(result['radius_r0_px'], 0.1 * k_true)

    def test_broken_tracklets_are_ignored(self):
        still = np.zeros((50, 2))
        jumpy = np.c_[np.arange(50.0) * 100.0, np.zeros(50)]
        frames = [np.arange(50), np.r_[np.arange(25), np.arange(26, 51)]]
        quantiles = fit.displacement_quantiles(track_table([still, jumpy], frames), max_dt=5,
                                               min_samples=10)
        self.assertEqual(quantiles['quantile'].to_pylist(), [0.0] * 5)
        self.assertEqual(quantiles['n'].to_pylist(), [50 - dt for dt in range(1, 6)])

    @settings(deadline=None, max_examples=25)
    @given(st.floats(0.0, 10.0), st.floats(0.0, 10.0))
    def test_exact_curve_is_recovered(self, r0, k):
        dt = np.arange(1, 31)
        quantiles = pa.table({'dt': dt, 'quantile': r0 + k * np.sqrt(dt), 'n': [500] * 30},
                             schema=fit.QUANTILE_SCHEMA)
        result = fit.fit_radius(quantiles)
        self.assertAlmostEqual(result['radius_r0_px'], r0, delta=1e-6)
        self.assertAlmostEqual(result['radius_k_px'], k, delta=1e-6)
        self.assertLess(result['rmse'], 1e-6)


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #
class SweepTests(unittest.TestCase):
    """CLI sweep with hold-out on a synthetic scene at two horizons."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = Path(cls._tmp.name)
        cls.path = tmp / 'observations.parquet'
        pq.write_table(sweep_scene(), cls.path)
        cls.out = tmp / 'sweep'
        cls.args = ['sweep', '--observations', str(cls.path), '--out', str(cls.out),
                    '--max-gaps', '5,2', '--holdout', '--holdout-gaps', '1,2,3',
                    '--per-gap', '1']
        cls.code = bridge_gaps.main(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_directories(self):
        self.assertEqual(self.code, 0)
        for horizon in (2, 5):
            run = self.out / f'max_gap_{horizon}'
            self.assertTrue(nonempty(run / 'GAP_BRIDGING_COMPLETE.json'))
            holdout = self.out / f'holdout_{horizon}'
            self.assertTrue(nonempty(holdout / f'holdout_{horizon}.json'))
            self.assertTrue(nonempty(holdout / 'bridged' / 'links.parquet'))

    def test_summary_rows(self):
        rows = read_csv(self.out / bridge_gaps.SUMMARY_NAME)
        self.assertEqual([int(r['max_gap']) for r in rows], [2, 5])
        self.assertEqual(list(rows[0]), list(bridge_gaps.SUMMARY_COLUMNS))
        short, full = rows
        # The 4-frame gap of crab 1 is bridged only at horizon 5.
        self.assertEqual((int(short['links']), int(full['links'])), (0, 1))
        self.assertEqual((int(short['trajectories']), int(full['trajectories'])), (7, 6))
        self.assertEqual(int(full['interpolated']), 4)
        self.assertEqual(int(full['interpolated_high']), 4)
        self.assertEqual((int(short['holdout_planted']), int(full['holdout_planted'])), (2, 3))
        for row in rows:
            self.assertEqual(float(row['holdout_precision']), 1.0)
            self.assertEqual(float(row['holdout_recall']), 1.0)
            self.assertEqual(int(row['decision_linked']), int(row['links']))

    def test_holdout_by_gap(self):
        rows = read_csv(self.out / bridge_gaps.BY_GAP_NAME)
        self.assertEqual([(int(r['max_gap']), int(r['gap'])) for r in rows],
                         [(2, 1), (2, 2), (5, 1), (5, 2), (5, 3)])
        for row in rows:
            self.assertEqual((int(row['planted']), int(row['recovered'])), (1, 1))

    def test_figures(self):
        fig_dir = self.out / bridge_gaps.FIGURES_DIR
        for name in ('sweep', 'holdout', 'radius_fit'):
            self.assertTrue(nonempty(fig_dir / f'{name}.pdf'), name)
            self.assertTrue(nonempty(fig_dir / f'{name}.png'), name)
        self.assertEqual(list(fig_dir.glob('bridges_*.png')), [])   # no video for the scene

    def test_refuses_overwrite(self):
        self.assertEqual(bridge_gaps.main(self.args), 1)


class CliCompatibilityTests(unittest.TestCase):
    """The v001 invocation without a subcommand still means ``run``."""

    def test_default_subcommand(self):
        args = bridge_gaps.parse_args(['--observations', 'a.parquet', '--out', 'b', '--max-gap', '3'])
        self.assertEqual((args.command, args.max_gap, args.force), ('run', 3, False))

    def test_explicit_subcommands(self):
        self.assertEqual(bridge_gaps.parse_args(['fit', '--observations', 'a']).max_dt, 60)
        args = bridge_gaps.parse_args(['sweep', '--observations', 'a', '--out', 'b'])
        self.assertEqual(args.max_gaps, [5, 10, 15, 30, 45, 60])
        self.assertEqual(args.holdout_gaps, [1, 2, 3, 5, 10, 15, 20, 30, 45, 60])
        self.assertEqual((args.per_gap, args.seed, args.holdout), (8, 0, False))


# --------------------------------------------------------------------------- #
# Rendered bridges
# --------------------------------------------------------------------------- #
@unittest.skipUnless(DEMO.exists() and VIDEO.exists(), 'demo parquet or video not present')
class BridgeFigureTests(unittest.TestCase):
    """fig_bridges renders long demo bridges from the video."""

    def test_demo_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pipeline.run(DEMO, tmp / 'run', BridgeConfig(max_gap=15))
            pages = figures.fig_bridges(tmp / 'run' / 'observations.parquet', VIDEO,
                                        tmp / 'figures' / 'bridges', n=2, min_gap=10)
            self.assertGreaterEqual(len(pages), 1)
            for page in pages:
                self.assertEqual(page.suffix, '.png')
                self.assertTrue(nonempty(page))
            none = figures.fig_bridges(tmp / 'run' / 'observations.parquet', VIDEO,
                                       tmp / 'figures' / 'none', min_gap=16)
            self.assertEqual(none, [])


if __name__ == '__main__':
    unittest.main()
