"""Small synthetic checks of audit primitives; no inference or original-file writes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('legacy_audit', ROOT / 'scripts/legacy_dataset/audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

INSTANCE_DTYPE = np.dtype([
    ('instance_id', 'i8'), ('instance_type', 'u1'), ('frame_id', 'u8'),
    ('skeleton', 'u4'), ('track', 'i4'), ('from_predicted', 'i8'),
    ('score', 'f4'), ('point_id_start', 'u8'), ('point_id_end', 'u8'),
    ('tracking_score', 'f4')])
FRAME_DTYPE = np.dtype([
    ('frame_id', 'u8'), ('video', 'u4'), ('frame_idx', 'u8'),
    ('instance_id_start', 'u8'), ('instance_id_end', 'u8')])
POINT_DTYPE = np.dtype([
    ('x', 'f8'), ('y', 'f8'), ('visible', '?'), ('complete', '?'), ('score', 'f8')])


def observations(count):
    inst = np.zeros(count, dtype=INSTANCE_DTYPE)
    inst['instance_type'] = 1
    inst['instance_id'] = np.arange(count)
    inst['point_id_start'] = np.arange(count) * 5
    inst['point_id_end'] = (np.arange(count) + 1) * 5
    inst['track'] = -1
    inst['from_predicted'] = -1
    inst['score'] = .6
    inst['tracking_score'] = np.nan
    pts = np.zeros((count, 5), dtype=POINT_DTYPE)
    pts['x'] = 100
    pts['y'] = 200
    pts['x'][:, 1] = 110
    pts['score'] = .8
    pts['visible'] = True
    return inst, pts


class AuditTests(unittest.TestCase):
    def test_empty_and_valid_payload_and_bad_range(self):
        (ROOT / 'checks/legacy_dataset').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'checks/legacy_dataset') as td:
            path = Path(td) / 'synthetic.h5'
            inst, pts = observations(1)
            with h5py.File(path, 'w') as f:
                f.create_dataset('instances', data=inst)
                f.create_dataset('pred_points', data=pts.ravel())
            with h5py.File(path, 'r') as f:
                fr = np.array([(0, 0, 39, 0, 1)], dtype=FRAME_DTYPE)[0]
                got, points, digest = audit.frame_payload(f, fr)
                self.assertEqual(len(got), 1)
                self.assertEqual(points.shape, (1, 5))
                self.assertEqual(len(digest), 64)
                fr['instance_id_end'] = 0
                self.assertEqual(len(audit.frame_payload(f, fr)[0]), 0)
                fr['instance_id_end'] = 2
                with self.assertRaises(AssertionError):
                    audit.frame_payload(f, fr)

    def test_missing_outside_and_low_score_are_separate(self):
        measure = audit.Measurements(80, 1000, 500)
        inst, pts = observations(4)
        pts['x'][1, 0] = np.nan
        pts['y'][2, 1] = 500
        pts['score'][3, 0] = .1
        measure(39, inst, pts, 'synthetic')
        summary = measure.summary()['counts']
        self.assertEqual(summary['instances'], 4)
        self.assertEqual(summary['Shell_missing_xy'], 1)
        self.assertEqual(summary['Shoulder_out_of_bounds_xy'], 1)
        self.assertEqual(summary['Shell_score_below_0.2_or_missing'], 1)
        self.assertEqual(summary['trusted_both_finite_inbounds_score_ge_0.2'], 1)

    def test_adjacency_diagnostic_does_not_bridge_missing_or_empty_frames(self):
        measure = audit.Measurements(80, 1000, 500)
        inst, pts = observations(1)
        measure(38, inst, pts, 'a')
        measure(40, inst, pts, 'b')
        key = 'adjacent_nearest_shell_distance_px_diagnostic_not_identity'
        self.assertNotIn(key, measure.parts)
        empty_inst, empty_pts = observations(0)
        measure(41, empty_inst, empty_pts, 'b')
        measure(42, inst, pts, 'b')
        self.assertNotIn(key, measure.parts)
        measure(43, inst, pts, 'b')
        self.assertEqual(len(measure.parts[key]), 1)

    def test_chunk_boundary_is_adjacent(self):
        measure = audit.Measurements(80, 1000, 500)
        inst, pts = observations(1)
        measure(39, inst, pts, 'a')
        moved = pts.copy()
        moved['x'] += 3
        moved['y'] += 4
        measure(40, inst, moved, 'b')
        key = 'adjacent_nearest_shell_distance_px_diagnostic_not_identity'
        self.assertEqual(measure.parts[key][0][0], 5)

    def test_quantiles_do_not_turn_missing_into_zero(self):
        result = audit.quantiles([np.nan, np.inf, 2, 4])
        self.assertEqual(result['finite'], 2)
        self.assertEqual(result['quantiles_0_01_05_50_95_99_100'][3], 3)
        self.assertIsNone(audit.quantiles([np.nan])['quantiles_0_01_05_50_95_99_100'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
