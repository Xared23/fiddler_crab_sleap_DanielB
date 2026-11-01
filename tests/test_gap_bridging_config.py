#!/usr/bin/env python3

"""
Tests for the gap-bridging configuration: YAML/default agreement, v004 fields,
threshold conversion and validation.

Sam
11/2026
"""

# ============================================================
# IMPORTS
# ============================================================

# ----- standard library -----
import math
from pathlib import Path
import sys
import tempfile
import unittest

# ----- numerics -----
from hypothesis import given, settings, strategies as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

# ----- local modules -----
from gap_bridging.config import BridgeConfig, load_config  # noqa: E402

# ============================================================
# CONSTANTS
# ============================================================

CONFIGS = ROOT / 'configs' / 'tracking'
DEFAULT_YAML = CONFIGS / 'gap_bridging_v001.yaml'
OVERLAYS = ('gap_bridging_v004_global.yaml', 'gap_bridging_v004_learned.yaml',
            'gap_bridging_v004_learned_global.yaml')


# ============================================================
# TESTS
# ============================================================
class DefaultYamlTests(unittest.TestCase):
    """The committed default YAML mirrors ``BridgeConfig()`` key for key."""

    def test_default_yaml_equals_dataclass_defaults(self):
        self.assertEqual(load_config(DEFAULT_YAML), BridgeConfig())

    def test_default_yaml_lists_every_field(self):
        import yaml
        keys = set(yaml.safe_load(DEFAULT_YAML.read_text()))
        self.assertEqual(keys, set(BridgeConfig().to_dict()))

    def test_to_dict_round_trips(self):
        cfg = BridgeConfig()
        self.assertEqual(load_config(None, **cfg.to_dict()), cfg)


class OverlayTests(unittest.TestCase):
    """The v004 overlays load on top of the defaults."""

    def test_global_overlay(self):
        cfg = load_config(CONFIGS / 'gap_bridging_v004_global.yaml')
        self.assertEqual((cfg.assignment, cfg.unmatched_cost_mode, cfg.cost_model),
                         ('global', 'adaptive', 'hand'))
        self.assertEqual(cfg.max_gap, BridgeConfig().max_gap)

    def test_learned_overlays_resolve_scorer_path_relative_to_yaml(self):
        for name in OVERLAYS[1:]:
            cfg = load_config(CONFIGS / name)
            self.assertEqual(cfg.cost_model, 'learned')
            self.assertEqual(Path(cfg.scorer_path), (CONFIGS / 'link_scorer_v001.joblib').resolve())
            self.assertTrue(Path(cfg.scorer_path).is_absolute())

    def test_learned_and_learned_global_share_learned_parameters(self):
        a = load_config(CONFIGS / OVERLAYS[1])
        b = load_config(CONFIGS / OVERLAYS[2])
        for key in ('learned_min_probability', 'learned_unmatched_probability', 'learned_margin',
                    'scorer_path'):
            self.assertEqual(getattr(a, key), getattr(b, key), key)
        self.assertEqual((b.assignment, b.unmatched_cost_mode), ('global', 'adaptive'))

    def test_absolute_scorer_path_is_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / 'c.yaml'
            yaml_path.write_text(f"cost_model: learned\nscorer_path: {tmp}/m.joblib\n")
            self.assertEqual(load_config(yaml_path).scorer_path, f"{tmp}/m.joblib")

    def test_override_scorer_path_becomes_str(self):
        cfg = load_config(None, cost_model='learned', scorer_path=Path('/x/y.joblib'))
        self.assertEqual(cfg.scorer_path, '/x/y.joblib')


class ThresholdTests(unittest.TestCase):
    """``cost_thresholds`` per cost model."""

    def test_hand_sqrt_thresholds(self):
        cfg = BridgeConfig()
        self.assertEqual(cfg.cost_thresholds(), {'max_assignment_cost': 1.0, 'unmatched_cost': 0.6,
                                                 'minimum_cost_margin': 0.25})

    def test_hand_linear_thresholds(self):
        cfg = BridgeConfig(gate_model='linear')
        self.assertEqual(cfg.cost_thresholds(), {'max_assignment_cost': 12.0, 'unmatched_cost': 7.0,
                                                 'minimum_cost_margin': 4.0})

    @settings(deadline=None, max_examples=50)
    @given(p_min=st.floats(1e-6, 1.0), p_un=st.floats(1e-6, 1.0), margin=st.floats(0.0, 5.0))
    def test_property_learned_thresholds_are_minus_log(self, p_min, p_un, margin):
        cfg = BridgeConfig(cost_model='learned', learned_min_probability=p_min,
                           learned_unmatched_probability=p_un, learned_margin=margin)
        t = cfg.cost_thresholds()
        self.assertAlmostEqual(t['max_assignment_cost'], -math.log(p_min))
        self.assertAlmostEqual(t['unmatched_cost'], -math.log(p_un))
        self.assertEqual(t['minimum_cost_margin'], margin)
        self.assertGreaterEqual(t['max_assignment_cost'], 0.0)


class ValidationTests(unittest.TestCase):
    """Invalid v004 values are refused at construction."""

    def test_invalid_values_raise(self):
        bad = (dict(cost_model='x'), dict(cost_model='learned', gate_model='linear'),
               dict(assignment='z'), dict(unmatched_cost_mode='y'), dict(adaptive_factor=0.99),
               dict(learned_min_probability=0.0), dict(learned_min_probability=1.5),
               dict(learned_unmatched_probability=0.0), dict(learned_margin=-0.1))
        for kwargs in bad:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    BridgeConfig(**kwargs)

    def test_learned_requires_sqrt_even_via_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'c.yaml'
            path.write_text("cost_model: learned\ngate_model: linear\n")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_defaults_preserve_v003_behaviour(self):
        cfg = BridgeConfig()
        self.assertEqual((cfg.cost_model, cfg.assignment, cfg.unmatched_cost_mode,
                          cfg.export_pair_features, cfg.scorer_path),
                         ('hand', 'staged', 'fixed', False, None))


if __name__ == '__main__':
    unittest.main()
