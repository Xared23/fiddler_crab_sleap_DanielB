"""Finite dataframe conversion/reconciliation regression tests (Slurm only)."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import v007_dataframe as export
from prediction_finalize import OBS_SCHEMA as SOURCE_SCHEMA, TRACK_SCHEMA


class DataframeTests(unittest.TestCase):
    def setUp(self):
        self.video, self.run_id = 'video', 'run'
        self.cfg = {'dataset_id': 'dataset', 'prediction_run_id': 'run', 'prediction_root': 'predictions/run'}
        self.run = {'chunk_frames': 1, 'fps_denominator': 1001, 'fps_numerator': 30000}
        self.frames = np.zeros(2, dtype=[('instance_id_start', 'i8'), ('instance_id_end', 'i8')])
        self.frames['instance_id_start'] = [0, 2]
        self.frames['instance_id_end'] = [2, 3]
        self.instances = np.zeros(3, dtype=[('instance_id', 'i8'), ('frame_id', 'i8'), ('point_id_start', 'i8'), ('point_id_end', 'i8'),
                                                ('track', 'i8'), ('score', 'f8'), ('tracking_score', 'f8')])
        self.instances['instance_id'] = [0, 1, 2]
        self.instances['frame_id'] = [0, 0, 1]
        self.instances['point_id_start'] = [0, 5, 10]
        self.instances['point_id_end'] = [5, 10, 15]
        self.instances['track'] = -1
        self.instances['score'] = [.9, .1, 1.2]
        self.instances['tracking_score'] = [0, np.nan, 0]
        self.points = np.zeros((3, 5), dtype=[('x', 'f8'), ('y', 'f8'), ('score', 'f8'), ('visible', '?')])
        self.points['x'][:] = [[10, 30, 40, 50, 60], [np.nan] * 5, [11, 31, 41, 51, 61]]
        self.points['y'][:] = [[10] * 5, [np.nan] * 5, [10] * 5]
        self.points['score'][:] = [[1.2] * 5, [np.nan] * 5, [1.1] * 5]
        self.points['visible'][:] = [[True] * 5, [False] * 5, [True] * 5]
        self.tracked = self.instances.copy()
        self.tracked['track'] = [0, -1, 0]
        self.water = [True, None, True]
        rows = []
        for k, (f, i, tid) in enumerate([(0, 0, 1), (0, 1, None), (1, 0, 1)]):
            row = {'video_id': 'video', 'frame': f, 'instance_index': i, 'raw_instance_id': k,
                   'detection_id': f'video/f{f:06d}/i{i:04d}', 'tracklet_id': 'run/video/t000000001' if tid else None,
                   'tracklet_local_id': tid, 'accepted_for_tracking': tid is not None,
                   'rejection_reason': None if tid else 'invalid_body', 'incoming_reason': 'linked' if k == 2 else 'video_start',
                   'instance_score': self.instances['score'][k], 'centroid_source': 'shell_proxy', 'water': self.water[k]}
            for field in ['x', 'y', 'score']:
                row[f'body_{field}'] = float(self.points[field][k, 0])
            for n, name in enumerate(export.NODES):
                for field in ['x', 'y', 'score', 'visible']:
                    row[f'{name}_{field}'] = self.points[field][k, n].item()
            rows.append(row)
        self.table = pa.Table.from_pylist(rows, schema=export.BASE_SCHEMA)
        self.tracks = pa.Table.from_pylist([{'tracklet_id': 'run/video/t000000001', 'tracklet_local_id': 1, 'video_id': 'video',
                                            'start_frame': 0, 'end_frame': 1, 'length': 2, 'start_reason': 'video_start',
                                            'termination_reason': 'video_end'}], schema=TRACK_SCHEMA)

    def normalize(self):
        return export.normalize(self.table, self.cfg, self.run, 'video', ['chunk0.slp', 'chunk1.slp'], [0, 2], self.instances['tracking_score'])

    def verify(self, table=None):
        export.verify_measurements(self.table if table is None else table, self.instances, self.points, self.frames,
                                   self.water, 'video', 'run', 0, self.tracked)

    def test_source_schema_is_exact(self):
        self.assertEqual(export.BASE_SCHEMA, SOURCE_SCHEMA)

    def test_all_measurements_and_optional_ids_reconcile(self):
        self.verify()
        ledger = export.Ledger(2, 1)
        ledger.update(self.table)
        ledger.check_tracks(self.tracks, 'video', 'run')
        self.assertEqual(ledger.summary()['observations'], 3)
        self.assertEqual(ledger.summary()['unassigned_observations'], 1)

    def test_namespace_and_both_source_locations(self):
        table = self.normalize()
        self.assertEqual(table['detection_id'][0].as_py(), 'dataset/video/f000000/i0000')
        self.assertEqual(table['prediction_detection_id'], self.table['detection_id'])
        self.assertEqual(table['tracklet_id'], self.table['tracklet_id'])
        self.assertEqual(table['source_chunk_instance_row'].to_pylist(), [0, 1, 0])
        self.assertEqual(table['source_instance_row'].to_pylist(), [0, 1, 2])
        self.assertEqual(table['source_chunk_path'].to_pylist(), ['chunk0.slp', 'chunk0.slp', 'chunk1.slp'])
        self.assertEqual(table['centroid_x'], table['body_x'])
        self.assertEqual(table['time_s'][2].as_py(), 1001 / 30000)

    def test_missing_is_null_without_dropping_rows(self):
        table = self.normalize()
        self.assertEqual(table.num_rows, 3)
        for name in ['body_x', 'centroid_x', 'shoulder_x', 'water', 'tracklet_id', 'source_tracking_score']:
            self.assertIsNone(table[name][1].as_py())
        self.assertEqual(table['shell_score'][0].as_py(), 1.2)  # no confidence clipping

    def test_measurement_corruption_fails(self):
        table = self.table.set_column(self.table.schema.get_field_index('elbow_x'), 'elbow_x', pa.array([99., np.nan, 41.]))
        with self.assertRaises(AssertionError):
            self.verify(table)

    def test_frame_or_source_key_corruption_fails(self):
        table = self.table.set_column(self.table.schema.get_field_index('instance_index'), 'instance_index', pa.array([0, 0, 0], type=pa.int32()))
        with self.assertRaises(AssertionError):
            self.verify(table)

    def test_tracked_slp_measurement_corruption_fails(self):
        self.tracked['score'][0] += .01
        with self.assertRaises(AssertionError):
            self.verify()

    def test_frame_split_across_batches_and_chunk_boundary(self):
        ledger = export.Ledger(2, 1)
        for row in range(3):
            ledger.update(self.table.slice(row, 1), chunk_frames=1)
        ledger.check_tracks(self.tracks, 'video', 'run')
        self.assertEqual(ledger.cross_chunk, 1)

    def test_gap_rejected(self):
        table = self.table.set_column(1, 'frame', pa.array([0, 0, 2], type=pa.int64()))
        with self.assertRaisesRegex(AssertionError, 'Gap'):
            export.Ledger(3, 1).update(table)

    def test_duplicate_track_frame_rejected(self):
        rows = self.table.to_pylist()
        rows[1].update(tracklet_local_id=1, accepted_for_tracking=True)
        with self.assertRaisesRegex(AssertionError, 'Repeated'):
            export.Ledger(2, 1).update(pa.Table.from_pylist(rows, schema=export.BASE_SCHEMA))

    def test_video_isolation_checked(self):
        ledger = export.Ledger(2, 1)
        ledger.update(self.table)
        with self.assertRaises(AssertionError):
            ledger.check_tracks(self.tracks, 'other_video', 'run')

    def test_roi_boundary_and_unknown(self):
        pts = np.array([[10, 10], [0, 10], [30, 10], [np.nan, 10], [-1, 10]])
        polygon = [[0, 0], [20, 0], [20, 20], [0, 20]]
        self.assertEqual(export.water_membership(pts, 100, 100, polygon), [True, True, False, None, None])
        self.assertEqual(export.water_membership(pts, 100, 100, None), [None] * 5)
        self.assertEqual(export.water_membership(pts, 100, 100, polygon, confirmed=False), [None] * 5)

    def test_parquet_roundtrip_and_pandas_nullable_filter(self):
        parent = ROOT / 'checks/v007_dataframe/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as directory:
            path = Path(directory) / 'part.parquet'
            table = self.normalize()
            pq.write_table(table, path)
            self.assertTrue(pq.read_table(path).equals(table))
            df = pd.read_parquet(path, dtype_backend='numpy_nullable')
            self.assertEqual(str(df['water'].dtype), 'boolean')
            self.assertEqual(str(df['centroid_x'].dtype), 'Float64')
            self.assertEqual(str(df['tracklet_local_id'].dtype), 'Int64')
            self.assertEqual(len(df.loc[df.accepted_for_tracking]), 2)
            self.assertEqual(len(df.loc[df.water.eq(False).fillna(False)]), 0)
            self.assertEqual(len(pd.read_parquet(path, filters=[('frame', '==', 1)])), 1)

    def test_empty_output_schema(self):
        empty = self.table.slice(0, 0)
        table = export.normalize(empty, self.cfg, self.run, 'video', ['chunk0.slp'], [0], [])
        self.assertEqual(table.schema, export.OBS_SCHEMA)
        self.assertEqual(table.num_rows, 0)


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DataframeTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'success': result.wasSuccessful(),
              'failures': [[str(test), text] for test, text in result.failures],
              'errors': [[str(test), text] for test, text in result.errors],
              'code_sha256': export.sha(ROOT / 'scripts/v007_dataframe.py'),
              'test_sha256': export.sha(Path(__file__)), 'job_id': os.environ['SLURM_JOB_ID']}
    export.dump(ROOT / 'checks/v007_dataframe' / f"tests_{os.environ['SLURM_JOB_ID']}.json", report)
    sys.exit(0 if result.wasSuccessful() else 1)
