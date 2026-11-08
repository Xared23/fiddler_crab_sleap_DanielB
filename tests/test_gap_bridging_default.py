#!/usr/bin/env python3

"""
The v007 pipeline default: learned S1 at p >= 0.5 (configs/tracking/gap_bridging_v007.yaml).

Pins what the default is (CLI fallback, YAML contents, model files), that the hand cost
stays reachable and stays the library default, and the default's result on the 300-frame
demo (golden link and interpolation digests).

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# ----- numerics and tables -----
import pyarrow.parquet as pq
import yaml

# ----- local modules -----
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))

import bridge_gaps  # noqa: E402
from gap_bridging import scorer  # noqa: E402
from gap_bridging.config import BridgeConfig, load_config  # noqa: E402
from test_gap_bridging_pipeline import interpolated_digest, links_digest  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

DEMO = ROOT / 'demo/crab_cam2_2_middle_300/observations.parquet'
DEFAULT = ROOT / 'configs/tracking/gap_bridging_v007.yaml'
HAND = ROOT / 'configs/tracking/gap_bridging_v001.yaml'
LEARNED_KEYS = {'cost_model', 'scorer_path', 'learned_min_probability',
                'learned_unmatched_probability', 'learned_margin'}
# The default on the 300-frame demo (2026-11-08, scikit-learn 1.9.1).
GOLDEN_COUNTS = {'links': 822, 'trajectories': 359}
GOLDEN_LINKS_SHA256 = '58dbb66ed14fd18997aaaa4390b5d5685e3a1cf5990f9bf8895f1fb95396fa7b'
GOLDEN_INTERPOLATED_SHA256 = 'c2e2f0c3eee99bcaa10643ac9506a199d11bf2e34aff1b06a562c47f0c30ff3e'

# ============================================================
# TESTS
# ============================================================


class DefaultConfigTests(unittest.TestCase):
    """What `run`/`sweep` use without --config, and what stays the hand cost."""

    def test_cli_falls_back_to_v007(self):
        self.assertEqual(bridge_gaps.DEFAULT_CONFIG, DEFAULT)
        self.assertEqual(bridge_gaps.HAND_CONFIG, HAND)
        self.assertEqual(bridge_gaps.config_path(None), DEFAULT)
        self.assertEqual(bridge_gaps.config_path(HAND), HAND)

    def test_default_is_learned_s1_at_half(self):
        cfg = load_config(DEFAULT)
        self.assertEqual(cfg.cost_model, 'learned')
        self.assertEqual((cfg.learned_min_probability, cfg.learned_unmatched_probability,
                          cfg.learned_margin), (0.5, 0.5, 0.0))
        self.assertEqual(Path(cfg.scorer_path), ROOT / 'configs/tracking/link_scorer_v007.joblib')
        manifest = json.loads(Path(cfg.scorer_path).with_suffix('.json').read_text())
        self.assertEqual((manifest['feature_set'], manifest['training_profile']), ('all', 'short'))
        self.assertIsNotNone(scorer.load_scorer(Path(cfg.scorer_path), cfg))

    def test_default_keeps_every_hand_key(self):
        """Same keys as the v003 hand cost, each once; only cost_model and scorer_path differ."""
        default, hand = yaml.safe_load(DEFAULT.read_text()), yaml.safe_load(HAND.read_text())
        self.assertEqual(set(default), set(hand))
        self.assertEqual({k for k in hand if default[k] != hand[k]}, {'cost_model', 'scorer_path'})
        self.assertTrue(LEARNED_KEYS <= set(default))
        self.assertEqual(default['max_gap'], 30)
        keys = [line.split(':', 1)[0] for line in DEFAULT.read_text().splitlines()
                if line and not line.startswith((' ', '#', '-'))]
        self.assertEqual(len(keys), len(set(keys)), 'duplicate top-level keys')

    def test_library_default_stays_hand(self):
        self.assertEqual(BridgeConfig().cost_model, 'hand')
        self.assertEqual(load_config(None), BridgeConfig())
        self.assertEqual(load_config(HAND), BridgeConfig())

    def test_missing_default_is_a_clear_error(self):
        saved = bridge_gaps.DEFAULT_CONFIG
        try:
            bridge_gaps.DEFAULT_CONFIG = ROOT / 'configs/tracking/absent.yaml'
            with self.assertRaisesRegex(FileNotFoundError, 'gap_bridging_v001.yaml'):
                bridge_gaps.config_path(None)
        finally:
            bridge_gaps.DEFAULT_CONFIG = saved


@unittest.skipUnless(DEMO.is_file(), 'demo clip not available')
class DefaultDemoGoldenTests(unittest.TestCase):
    """`bridge_gaps.py run` without --config on the 300-frame demo (about 4 s)."""

    def test_default_demo_golden(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'run'
            done = subprocess.run([sys.executable, str(ROOT / 'scripts/bridge_gaps.py'),
                                   '--observations', str(DEMO), '--out', str(out)],
                                  capture_output=True, text=True, check=False)
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            marker = json.loads((out / 'GAP_BRIDGING_COMPLETE.json').read_text())
            links = pq.read_table(out / 'links.parquet')
            obs = pq.read_table(out / 'observations.parquet')
        self.assertEqual(marker['config']['cost_model'], 'learned')
        self.assertEqual({k: marker['counts'][k] for k in GOLDEN_COUNTS}, GOLDEN_COUNTS)
        self.assertEqual(links_digest(links), GOLDEN_LINKS_SHA256)
        self.assertEqual(interpolated_digest(obs), GOLDEN_INTERPOLATED_SHA256)


if __name__ == '__main__':
    unittest.main()
