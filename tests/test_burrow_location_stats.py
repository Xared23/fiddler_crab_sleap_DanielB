"""Known-answer CPU controls for burrow count/spacing summaries."""
import os
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from burrow_location_stats import distance_summary, spacing, frame_summary, valid_points
from female_regional_validation import right_bank_geometry


class BurrowTests(unittest.TestCase):
    def test_three_four_five_triangle_nearest_vs_all_pairs(self):
        result = spacing([[0, 0], [3, 0], [3, 4]])
        self.assertEqual(result['nearest_neighbor']['n'], 3)
        self.assertAlmostEqual(result['nearest_neighbor']['mean_px'], 10/3)
        self.assertAlmostEqual(result['all_unordered_pairs']['mean_px'], 4)
        self.assertEqual(result['all_unordered_pairs']['n'], 3)

    def test_square_and_rotation_invariance(self):
        xy = np.array([[0, 0], [0, 10], [10, 0], [10, 10]], dtype=float)
        result = spacing(xy)
        self.assertAlmostEqual(result['nearest_neighbor']['mean_px'], 10)
        self.assertAlmostEqual(result['all_unordered_pairs']['mean_px'], (40+20*np.sqrt(2))/6)
        rotation = np.array([[0., -1.], [1., 0.]])
        rotated = spacing(xy @ rotation)
        self.assertAlmostEqual(rotated['nearest_neighbor']['mean_px'], 10)
        self.assertAlmostEqual(rotated['all_unordered_pairs']['mean_px'], result['all_unordered_pairs']['mean_px'])

    def test_exact_duplicates_audited_not_zero_spacing(self):
        result = spacing([[0, 0], [0, 0], [3, 4]])
        self.assertEqual(result['annotation_points'], 3)
        self.assertEqual(result['distinct_coordinate_points'], 2)
        self.assertEqual(result['exact_duplicate_points_excluded_from_spacing'], 1)
        self.assertAlmostEqual(result['nearest_neighbor']['mean_px'], 5)

    def test_empty_singleton_and_invalid_points(self):
        for xy in ([], [[1, 2]]):
            self.assertIsNone(spacing(xy)['nearest_neighbor']['mean_px'])
            self.assertIsNone(spacing(xy)['all_unordered_pairs']['mean_px'])
        with self.assertRaises(ValueError):
            spacing([[np.nan, 1], [2, 3]])
        self.assertEqual(valid_points([[0, 0], [-1, 0], [10, 1], [2, np.nan]], 10, 10).tolist(),
                         [True, False, False, False])

    def test_same_right_roi_for_counts_and_neighbor_candidates(self):
        geometry = right_bank_geometry([[2, 1], [6, 1], [6, 5], [2, 5]], 10, 10)
        result = frame_summary([[7, 1], [7, 5], [5.9, 1]], geometry)
        self.assertEqual(result['annotated_instances'], 3)
        self.assertEqual(result['right_of_puddle_instances'], 2)
        self.assertEqual(result['right_region_spacing']['nearest_neighbor']['mean_px'], 4)
        self.assertAlmostEqual(result['right_region_anchors_nearest_neighbor_anywhere_in_frame']['mean_px'], 2.55)

    def test_repeated_frames_are_independent_not_pooled(self):
        geometry = right_bank_geometry([[2, 1], [6, 1], [6, 5], [2, 5]], 10, 10)
        for _ in range(2):
            result = frame_summary([[7, 1], [7, 6]], geometry)
            self.assertEqual(result['full_frame_spacing']['nearest_neighbor']['mean_px'], 5)
            self.assertEqual(result['annotated_instances'], 2)

    def test_distance_summary_quantiles(self):
        summary = distance_summary([1, 2, 3, 4, 5])
        self.assertEqual(summary['mean_px'], 3)
        self.assertEqual(summary['median_px'], 3)
        self.assertAlmostEqual(summary['p10_px'], 1.4)
        self.assertAlmostEqual(summary['p90_px'], 4.6)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    unittest.main()
