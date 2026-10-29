"""Finite CPU integration tests: actual SLP/Parquet serialization and raw invariance."""
import copy
from pathlib import Path
import os
import sys
import tempfile
import unittest

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
import sleap_io as sio
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from prediction_finalize import inspect_chunk, merge_raw, derive_tracks
from pipeline_common import sha256

CONFIG = yaml.safe_load((ROOT/'configs/tracking/body_first_v001.yaml').read_text())


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.directory = Path(self.temp.name)
        self.video_path = self.directory/'synthetic.mp4'
        writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30., (320, 240))
        self.assertTrue(writer.isOpened())
        for _ in range(44):
            writer.write(np.zeros((240, 320, 3), np.uint8))
        writer.release()
        self.video = sio.load_video(self.video_path)
        self.skeleton = sio.Skeleton(nodes=['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm'])
        self.chunks = []
        for start, stop in [(0, 40), (40, 44)]:
            frames = []
            for index in range(start, stop):
                instances = []
                if index != 42:  # explicitly empty, not unprocessed
                    for x, score in [(40.+index*.1, .9), (150.+index*.1, .9), (250., .1)]:
                        points = np.full((5, 2), np.nan)
                        points[0] = [x, 100.]
                        if index % 2 == 0:
                            points[1] = [x, 120.]
                        instances.append(sio.PredictedInstance.from_numpy(points, skeleton=self.skeleton,
                            point_scores=np.array([score, .5, 0., 0., 0.]), score=.8))
                    if index % 2:
                        instances.reverse()
                frames.append(sio.LabeledFrame(video=self.video, frame_idx=index, instances=instances))
            labels = sio.Labels(labeled_frames=frames, videos=[self.video], skeletons=[self.skeleton])
            # Real inference carries the dedicated provenance_json dataset.
            labels.provenance = {'test_fixture': True, 'source_range': [start, stop]}
            path = self.directory/f'chunk_{start}.slp'
            labels.save(str(path), embed=False, format='slp')
            self.chunks.append((path, start, stop))
        self.raw = self.directory/'raw.slp'

    def tearDown(self):
        self.temp.cleanup()

    def merge(self):
        return merge_raw(self.chunks, self.raw, 44, self.video_path)

    def test_actual_slp_merge_and_all_detection_preservation(self):
        info = self.merge()
        self.assertEqual(info['instances'], 129)
        labels = sio.load_slp(self.raw, open_videos=False)
        self.assertEqual(len(labels), 44)
        self.assertEqual(len(labels[42].instances), 0)
        self.assertEqual(len(labels.tracks), 0)
        self.assertEqual(labels.provenance['raw_merge']['frames'], 44)
        self.assertEqual(labels.provenance['first_chunk_provenance']['source_range'], [0, 40])
        self.assertTrue(all(i.track is None for lf in labels for i in lf))

    def test_tracking_chunk_boundary_gap_and_raw_bytes(self):
        self.merge()
        before = sha256(self.raw)
        output = self.directory/'tracklets'
        info = derive_tracks(self.raw, output, 'test_video', copy.deepcopy(CONFIG), 320, 240)
        self.assertEqual(sha256(self.raw), before)
        self.assertEqual(info['observations'], 129)
        self.assertEqual(info['unassigned_observations'], 43)
        self.assertEqual(info['tracklets'], 4)
        labels = sio.load_slp(output/'test_video.tracked.slp', open_videos=False)
        # Same physical body despite reversed source order, across 39 -> 40.
        name39 = min(labels[39].instances, key=lambda i: i.numpy()[0, 0]).track.name
        name40 = min(labels[40].instances, key=lambda i: i.numpy()[0, 0]).track.name
        name43 = min(labels[43].instances, key=lambda i: i.numpy()[0, 0]).track.name
        self.assertEqual(name39, name40)
        self.assertNotEqual(name40, name43)
        rows = pq.read_table(output/'observations.parquet').to_pylist()
        self.assertEqual(len(rows), 129)
        self.assertEqual(sum(r['tracklet_id'] is None for r in rows), 43)
        self.assertEqual(len({r['detection_id'] for r in rows}), 129)
        self.assertTrue(all(r['centroid_source'] == 'shell_proxy' for r in rows))
        self.assertTrue(all(r['water'] is None for r in rows))

    def test_missing_or_repeated_coverage_refused(self):
        with self.assertRaises(AssertionError):
            merge_raw([self.chunks[1]], self.raw, 44, self.video_path)
        self.assertFalse(self.raw.exists())

    def test_chunk_bad_frame_range_refused(self):
        with self.assertRaises(AssertionError):
            inspect_chunk(self.chunks[0][0], 1, 41, self.video_path)

    def test_existing_outputs_not_overwritten(self):
        self.merge()
        before = sha256(self.raw)
        with self.assertRaises(FileExistsError):
            self.merge()
        self.assertEqual(sha256(self.raw), before)
        output = self.directory/'tracklets'
        output.mkdir()
        with self.assertRaises(FileExistsError):
            derive_tracks(self.raw, output, 'test', CONFIG, 320, 240)
        self.assertEqual(sha256(self.raw), before)

    def test_empty_first_chunk_and_all_unassigned_retained(self):
        # Exercise empty SLP arrays, schema persistence, and a zero-track catalog.
        empty = sio.Labels(labeled_frames=[sio.LabeledFrame(self.video, i, []) for i in range(2)],
                           videos=[self.video], skeletons=[self.skeleton])
        path = self.directory/'empty.slp'
        empty.save(str(path), embed=False, format='slp')
        result = merge_raw([(path, 0, 2)], self.raw, 2, self.video_path)
        self.assertEqual(result['instances'], 0)
        info = derive_tracks(self.raw, self.directory/'tracklets', 'empty', CONFIG, 320, 240)
        self.assertEqual(info['tracklets'], 0)
        self.assertEqual(info['observations'], 0)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run only through a finite project Slurm job')
    unittest.main()
