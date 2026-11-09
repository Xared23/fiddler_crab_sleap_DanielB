"""Known-answer right-shoreline conditioning controls; Slurm CPU only."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from female_regional_validation import right_bank_geometry, region_membership, summarize
from pipeline_common import read_json


class RegionTests(unittest.TestCase):
    def rectangle(self):
        return right_bank_geometry([[2, 1], [6, 1], [6, 5], [2, 5]], 10, 10)

    def test_vertical_top_and_bottom_extensions_and_strict_boundary(self):
        g = self.rectangle()
        points = [[7, 0], [7, 3], [7, 9], [6, 0], [6, 3], [6, 9], [5.9, 0], [5.9, 9]]
        self.assertEqual(region_membership(points, g).tolist(), [True, True, True, False, False, False, False, False])
        self.assertEqual(g['top_tip_xy'], [6., 1.])
        self.assertEqual(g['bottom_tip_xy'], [6., 5.])

    def test_water_left_bank_and_invalid_points_excluded(self):
        points = [[3, 3], [1, 3], [np.nan, 3], [7, np.inf], [-1, 3], [10, 3], [7, 10], [7, -1]]
        self.assertFalse(region_membership(points, self.rectangle()).any())
        self.assertEqual(region_membership([], self.rectangle()).shape, (0,))

    def test_follows_actual_indented_bank_not_rowwise_max_x(self):
        g = right_bank_geometry([[2, 1], [6, 1], [6, 3], [4, 2], [4, 5], [2, 5]], 10, 10)
        # This land pocket is right of the actual bent bank, despite another
        # part of the water being further right at the same y.
        self.assertEqual(region_membership([[4.5, 2.5], [5.5, 2.5], [7, 2.5]], g).tolist(), [True, False, True])
        self.assertEqual(g['roi'].intersection(g['water']).area, 0)

    def test_polygon_orientation_and_closed_vertex_independent(self):
        points = [[2, 1], [6, 1], [6, 5], [2, 5]]
        a = right_bank_geometry(points, 10, 10)
        b = right_bank_geometry(list(reversed(points)), 10, 10)
        c = right_bank_geometry(points+[points[0]], 10, 10)
        self.assertTrue(a['roi'].equals(b['roi']))
        self.assertTrue(a['roi'].equals(c['roi']))

    def test_invalid_geometry_refused(self):
        for polygon in ([[1, 1], [2, 2], [3, 3]], [[1, 1], [np.nan, 2], [3, 3]],
                        [[1, 1], [2, 2], [10, 3]], [[1, 1], [4, 4], [1, 4], [4, 1]]):
            with self.assertRaises(ValueError):
                right_bank_geometry(polygon, 10, 10)

    def test_filter_both_sources_before_one_to_one_matching(self):
        truths = {0: np.array([[7., 1.], [7., 3.], [3., 2.]])}
        predictions = {0: np.array([[7.1, 1.], [3.1, 2.], [9., 9.]])}
        r = summarize(truths, predictions, self.rectangle(), [.5], True)
        m = r['metrics_by_matching_radius_px']['0.5']
        self.assertEqual((r['human_instances'], r['predicted_instances']), (2, 2))
        self.assertEqual((m['true_positives'], m['false_negatives'], m['false_positives_relative_to_labels']), (1, 1, 1))
        self.assertAlmostEqual(m['precision'], .5)
        self.assertAlmostEqual(m['recall'], .5)
        full = summarize(truths, predictions, self.rectangle(), [.5], False)
        self.assertEqual(full['metrics_by_matching_radius_px']['0.5']['true_positives'], 2)

    def test_real_accepted_puddle_tips_and_no_water_overlap(self):
        mask = read_json(ROOT / 'datasets/v007_best_body_first_dataframe_v001/config/water_regions.json')
        g = right_bank_geometry(mask['videos']['Crab_Cam_2_2']['polygon_xy'], 7680, 4320)
        self.assertEqual(g['top_tip_xy'], [3290., 0.])
        self.assertEqual(g['bottom_tip_xy'], [2590., 3170.])
        self.assertLessEqual(g['roi'].intersection(g['water']).area, 1e-7)
        self.assertEqual(region_membership([[2591, 4000], [2589, 4000], [2590, 4000], [3291, 0], [3290, 0]], g).tolist(),
                         [True, False, False, True, False])


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    unittest.main()
