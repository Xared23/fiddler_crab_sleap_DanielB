"""Finite synthetic acceptance tests. Run only through the CPU Slurm entry point."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap')
sys.path.insert(0, str(ROOT / 'scripts/legacy_dataset'))
from tracklets import Frame, Tracker, water_membership
from export import (Parts, SingleFile, OBS_SCHEMA, TRACK_SCHEMA, table_from_columns,
                    rows_for_frame, validate_saved_video)

CONFIG_PATH = ROOT / 'datasets/legacy_native_tracklets_v001/config/tracklets.yaml'
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())


def frame(index, xs=(100,), ys=None, scores=None, shoulder=True):
    xs = np.asarray(xs, dtype=float)
    if ys is None:
        ys = np.full(len(xs), 100.)
    xy = np.column_stack([xs, ys])
    sh = xy + [20., 0.]
    if not shoulder:
        sh[:] = np.nan
    return Frame(index, xy, sh, np.full(len(xs), .9) if scores is None else np.asarray(scores, float),
                 np.full(len(xs), .25 if shoulder else 0.), np.ones(len(xs), bool), np.full(len(xs), shoulder, bool))


def tracker(video='test_video'):
    return Tracker(video, 1000, 500, dict(CONFIG))


class TrackletTests(unittest.TestCase):
    def test_adjacent_and_chunk_boundary_continuity(self):
        tr = tracker()
        a = tr.process(frame(39))
        b = tr.process(frame(40, (103,)))
        self.assertEqual(a['ids'][0], b['ids'][0])
        self.assertEqual(b['incoming_reason'][0], 'linked')
        ended = tr.finish()
        self.assertEqual((ended[0]['start_frame'], ended[0]['end_frame'], ended[0]['length']), (39, 40, 2))

    def test_missing_span_breaks_id(self):
        tr = tracker()
        a = tr.process(frame(0))
        b = tr.process(frame(2))
        self.assertNotEqual(a['ids'][0], b['ids'][0])
        self.assertEqual(b['ended'][0]['termination_reason'], 'missing_span')

    def test_explicit_missing_record_breaks_id(self):
        tr = tracker()
        a = tr.process(frame(0))
        missing = tr.process(Frame.empty(1, observed=False))
        b = tr.process(frame(2))
        self.assertEqual(missing['ended'][0]['termination_reason'], 'missing_span')
        self.assertNotEqual(a['ids'][0], b['ids'][0])
        self.assertEqual(b['incoming_reason'][0], 'missing_span')

    def test_empty_frame_breaks_id(self):
        tr = tracker()
        a = tr.process(frame(0))
        empty = tr.process(Frame.empty(1))
        b = tr.process(frame(2))
        self.assertEqual(empty['ended'][0]['termination_reason'], 'empty_frame')
        self.assertNotEqual(a['ids'][0], b['ids'][0])

    def test_near_equivalent_competitors_are_not_assigned(self):
        tr = tracker()
        tr.process(frame(0, (100, 120)))
        result = tr.process(frame(1, (110,)))
        self.assertEqual(result['incoming_reason'][0], 'ambiguous_competitors')
        self.assertEqual(len(result['ended']), 2)
        self.assertTrue(all(x['termination_reason'] == 'ambiguous_competitors' for x in result['ended']))

    def test_guard_blocks_competitor_just_outside_motion_gate(self):
        tr = tracker()
        tr.process(frame(0, (100, 128)))
        result = tr.process(frame(1, (110,)))
        self.assertEqual(result['incoming_reason'][0], 'ambiguous_competitors')

    def test_low_score_competitor_still_blocks_link(self):
        tr = tracker()
        tr.process(frame(0, (100, 118), scores=(.9, .1)))
        result = tr.process(frame(1, (110,)))
        self.assertEqual(result['incoming_reason'][0], 'ambiguous_competitors')

    def test_one_to_two_collision_does_not_choose_winner(self):
        tr = tracker()
        tr.process(frame(0, (100,)))
        result = tr.process(frame(1, (94, 106)))
        self.assertEqual(list(result['incoming_reason']), ['ambiguous_competitors'] * 2)
        self.assertEqual(len(set(result['ids'])), 2)

    def test_duplicate_candidates_reject_both(self):
        result = tracker().process(frame(0, (100, 103)))
        self.assertEqual(list(result['validity']['reasons']), ['duplicate_candidate'] * 2)
        self.assertEqual(list(result['ids']), [-1, -1])

    def test_missing_outside_and_low_score_body_rejection(self):
        f = frame(0, (np.nan, -1, 100, 300), scores=(.9, .9, .1, .9))
        f.body_visible[3] = False
        result = tracker().process(f)
        self.assertEqual(list(result['validity']['reasons']), ['missing_position', 'outside_image', 'low_body_score', 'invisible'])

    def test_shoulder_missing_retains_singletons_no_bridging(self):
        tr = tracker()
        a = tr.process(frame(0))
        b = tr.process(frame(1, shoulder=False))
        c = tr.process(frame(2))
        self.assertTrue(b['validity']['accepted'][0])
        self.assertFalse(b['validity']['linkable'][0])
        self.assertEqual(len({a['ids'][0], b['ids'][0], c['ids'][0]}), 3)
        self.assertEqual(c['ended'][0]['length'], 1)

    def test_outside_shoulder_retains_body_but_no_link(self):
        f = frame(0)
        f.shoulder[0, 0] = 1001
        result = tracker().process(f)
        self.assertTrue(result['validity']['accepted'][0])
        self.assertFalse(result['validity']['linkable'][0])

    def test_motion_gate_breaks(self):
        tr = tracker()
        tr.process(frame(0))
        result = tr.process(frame(1, (113,)))
        self.assertEqual(result['incoming_reason'][0], 'body_displacement_gate')

    def test_shoulder_vector_gate_breaks(self):
        tr = tracker()
        tr.process(frame(0))
        f = frame(1)
        f.shoulder[0] = [100, 120]
        result = tr.process(f)
        self.assertEqual(result['incoming_reason'][0], 'shoulder_displacement_or_vector_gate')

    def test_video_scoped_ids(self):
        a, b = tracker('a'), tracker('b')
        ra, rb = a.process(frame(0)), b.process(frame(0))
        self.assertNotEqual(a.name(ra['ids'][0]), b.name(rb['ids'][0]))

    def test_determinism_and_only_adjacent_active_state(self):
        def run():
            tr = tracker()
            output, ended = [], []
            for f in [frame(0, (100, 300)), frame(1, (101, 303)), Frame.empty(2), frame(3, (100, 300))]:
                r = tr.process(f)
                output.append(r['ids'].tolist())
                ended.extend(r['ended'])
            ended.extend(tr.finish())
            return output, ended
        self.assertEqual(run(), run())

    def test_nonchronological_input_fails(self):
        tr = tracker()
        tr.process(frame(1))
        with self.assertRaises(AssertionError):
            tr.process(frame(1))

    def test_roi_inside_outside_boundary_and_missing(self):
        xy = np.array([[15, 15], [30, 15], [10, 15], [np.nan, 15], [-1, 15]], float)
        roi = [[10, 10], [20, 10], [20, 20], [10, 20]]
        self.assertEqual(water_membership(xy, 100, 100, roi), [True, False, True, None, None])
        self.assertEqual(water_membership(xy, 100, 100, roi, confirmed=False), [None] * 5)
        self.assertEqual(water_membership(xy, 100, 100, None), [None] * 5)
        self.assertEqual(water_membership(xy, 100, 100, roi, uncertainty_px=1), [True, False, None, None, None])

    def test_static_roi_per_observation_no_tracking_effect(self):
        roi = [[101, 80], [200, 80], [200, 130], [101, 130]]
        tr = tracker()
        a, b = frame(0, (100,)), frame(1, (102,))
        ra, rb = tr.process(a), tr.process(b)
        self.assertEqual(ra['ids'][0], rb['ids'][0])
        self.assertEqual(water_membership(a.body, 1000, 500, roi), [False])
        self.assertEqual(water_membership(b.body, 1000, 500, roi), [True])

    def test_end_to_end_small_export_and_full_readback(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'checks/legacy_dataset') as td:
            work = Path(td)
            (work / 'tables').mkdir()
            tr = tracker()
            obs = Parts(work / 'tables/observations/test_video', OBS_SCHEMA)
            rej = Parts(work / 'tables/rejected_observations/test_video', OBS_SCHEMA)
            tw = SingleFile(work / 'tables/tracklets.parquet', TRACK_SCHEMA)
            source = [frame(0, (100, 300)), frame(1, (101, 303)),
                      frame(2, (102, 304), shoulder=False), frame(3, (100, 102)),
                      Frame.empty(4), frame(5, (100, 300), scores=(.9, .1))]
            previous = None
            offset = 0
            for f in source:
                n = len(f)
                inst = np.zeros(n, dtype=[('track', 'i4'), ('score', 'f4'), ('tracking_score', 'f4')])
                inst['track'] = -1
                inst['score'] = .7
                inst['tracking_score'] = np.nan
                points = np.zeros((n, 5), dtype=[('x', 'f8'), ('y', 'f8'), ('score', 'f8'), ('visible', '?')])
                points['x'][:] = np.nan
                points['y'][:] = np.nan
                points['x'][:, 0], points['y'][:, 0] = f.body[:, 0], f.body[:, 1]
                points['x'][:, 1], points['y'][:, 1] = f.shoulder[:, 0], f.shoulder[:, 1]
                points['score'][:, 0], points['score'][:, 1] = f.body_score, f.shoulder_score
                points['visible'][:, 0], points['visible'][:, 1] = f.body_visible, f.shoulder_visible
                result = tr.process(f)
                tw.add(result['ended'])
                fr = {'frame_idx': f.index, 'instance_id_start': offset, 'instance_id_end': offset + n}
                a, r, previous = rows_for_frame('test_video', 'synthetic.slp', fr, inst, points, tr, result,
                                               {'polygon_xy': [[0, 0], [500, 0], [500, 400], [0, 400]]}, 30., previous)
                obs.add(a)
                rej.add(r)
                offset += n
            tw.add(tr.finish())
            tw.close()
            obs.close()
            rej.close()
            summary = validate_saved_video(work, 'test_video', tr.next_id - 1, np.array([len(f) for f in source]), CONFIG)
            self.assertEqual(summary['source_observations'], 10)
            self.assertEqual(summary['accepted_observations'], 7)
            self.assertEqual(summary['rejected_observations'], 3)
            self.assertEqual(summary['tracklets'], 5)
            self.assertEqual(summary['singletons'], 3)

    def test_parquet_roundtrip_nullable_schema_and_empty_table(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'checks/legacy_dataset') as td:
            cols = {}
            for f in OBS_SCHEMA:
                if pa.types.is_string(f.type):
                    cols[f.name] = ['value', None]
                elif pa.types.is_integer(f.type):
                    cols[f.name] = [1, None]
                elif pa.types.is_floating(f.type):
                    cols[f.name] = [1.2, np.nan]
                else:
                    cols[f.name] = [True, None]
            table = table_from_columns(cols, OBS_SCHEMA)
            parts = Parts(Path(td) / 'parts', OBS_SCHEMA)
            parts.add(table)
            parts.close()
            df = pd.read_parquet(parts.parts[0], dtype_backend='numpy_nullable')
            self.assertEqual(str(df.water.dtype), 'boolean')
            self.assertEqual(str(df.body_score.dtype), 'Float64')
            self.assertTrue(pd.isna(df.water.iloc[1]))
            empty = Parts(Path(td) / 'empty', OBS_SCHEMA)
            empty.close()
            self.assertEqual(pq.read_table(empty.parts[0]).num_rows, 0)


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TrackletTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = {'job_id': os.environ['SLURM_JOB_ID'], 'tests_run': result.testsRun,
               'success': result.wasSuccessful(), 'failures': len(result.failures), 'errors': len(result.errors),
               'tracklets_code_sha256': sha(ROOT / 'scripts/legacy_dataset/tracklets.py'),
               'export_code_sha256': sha(ROOT / 'scripts/legacy_dataset/export.py'),
               'test_code_sha256': sha(Path(__file__)), 'config_sha256': sha(CONFIG_PATH)}
    path = ROOT / 'checks/legacy_dataset' / f'tracklets_tests_{os.environ["SLURM_JOB_ID"]}.json'
    with path.open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    sys.exit(0 if result.wasSuccessful() else 1)
