from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sleap_io as sio
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import augment_crab_demo as a
from v007_dataframe import OBS_SCHEMA


def config():
    cfg = yaml.safe_load((ROOT / 'configs/tracking/body_first_v001.yaml').read_text())
    cfg.pop('pilot', None)
    cfg.update(dataset_id=a.FEMALE_TRACK_SCOPE, body_node='FBody', shoulder_node=None,
               shoulder_vector_weight=0.0)
    return cfg


def video(path, count):
    return sio.Video(filename=str(path), backend_metadata={'shape': [count, 4320, 7680, 3],
                     'fps': a.FPS, 'grayscale': False}, open_backend=False)


def female_source(path, positions):
    v = video(path.with_suffix('.mp4'), len(positions))
    sk = sio.Skeleton(['FBody'])
    frames = []
    for k, xs in enumerate(positions):
        instances = [sio.PredictedInstance.from_numpy(np.array([[x, 100.]]), sk,
                     point_scores=np.array([.9 if x != 200 else .1]), score=.8, tracking_score=0.) for x in xs]
        frames.append(sio.LabeledFrame(v, k, instances))
    sio.save_slp(sio.Labels(frames, videos=[v], skeletons=[sk]), path)


class AugmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='demo_augment_test_', dir=ROOT / 'checks')
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_slice_preserves_all_point_bytes_and_indices(self):
        source, target = self.directory / 'source.slp', self.directory / 'subset.slp'
        female_source(source, [[20], [21, 80], [], [23]])
        f, i, p, offset = a.slice_female(source, target, self.directory / 'chunk.mp4', 1, 3)
        np.testing.assert_array_equal(f['frame_idx'], [0, 1, 2])
        np.testing.assert_array_equal(i['frame_id'], [0, 0, 2])
        self.assertEqual(offset, 1)
        with h5py.File(source) as src, h5py.File(target) as dst:
            self.assertEqual(src['pred_points'][1:].tobytes(), dst['pred_points'][:].tobytes())
            self.assertEqual(json.loads(dst['videos_json'][0])['backend']['filename'], 'chunk.mp4')

    def test_slice_empty_span(self):
        source = self.directory / 'source.slp'
        female_source(source, [[20], [], []])
        _, i, p, _ = a.slice_female(source, self.directory / 'empty.slp', self.directory / 'chunk.mp4', 1, 2)
        self.assertEqual(len(i), 0)
        self.assertEqual(len(p), 0)

    def test_slice_refuses_overwrite(self):
        source, target = self.directory / 'source.slp', self.directory / 'subset.slp'
        female_source(source, [[20]])
        a.slice_female(source, target, self.directory / 'chunk.mp4', 0, 1)
        with self.assertRaises(FileExistsError):
            a.slice_female(source, target, self.directory / 'chunk.mp4', 0, 1)

    def test_tracking_crosses_chunks_and_keeps_rejected_detections(self):
        source = self.directory / 'source.slp'
        female_source(source, [[20, 80, 200], [81, 21, 200], [22, 82, 200], [23, 83, 200]])
        tracker = a.Tracker(a.VIDEO_ID, 7680, 4320, config())
        diagnostics = []
        for start in [0, 2]:
            raw, tracked = self.directory / f'raw{start}.slp', self.directory / f'tracked{start}.slp'
            a.slice_female(source, raw, self.directory / 'movie.mp4', start, 2)
            records, _ = a.track_female(raw, tracked, start, tracker)
            diagnostics.extend(records)
        self.assertEqual(diagnostics[0]['tracklet_id'], diagnostics[4]['tracklet_id'])
        self.assertEqual(diagnostics[0]['tracklet_id'], diagnostics[6]['tracklet_id'])
        self.assertEqual(diagnostics[6]['incoming_reason'], 'linked')
        self.assertIsNone(diagnostics[2]['tracklet_id'])
        self.assertEqual(len(diagnostics), 12)
        self.assertEqual(sorted(t['length'] for t in tracker.finish()), [4, 4])

    def test_no_bridging_empty_frames(self):
        source, raw = self.directory / 'source.slp', self.directory / 'raw.slp'
        female_source(source, [[20], [], [20]])
        a.slice_female(source, raw, self.directory / 'movie.mp4', 0, 3)
        diagnostics, ended = a.track_female(raw, self.directory / 'tracked.slp', 0,
                                          a.Tracker(a.VIDEO_ID, 7680, 4320, config()))
        self.assertNotEqual(diagnostics[0]['tracklet_id'], diagnostics[1]['tracklet_id'])
        self.assertEqual(ended[0]['termination_reason'], 'empty_frame')

    def test_display_union_has_no_invented_anatomy(self):
        source = sio.PredictedInstance.from_numpy(np.array([[12.5, 23.5]]), sio.Skeleton(['FBody']),
                    point_scores=np.array([.8]), score=.9, tracking_score=0.)
        sk = sio.Skeleton(['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm', 'FBody', 'Burrow'])
        displayed = a.display_instance(source, sk)
        a.assert_group([source], [displayed])
        self.assertTrue(np.isnan(displayed.points['xy'][:5]).all())
        self.assertFalse(displayed.points['visible'][:5].any())
        np.testing.assert_array_equal(displayed.points['xy'][5], [12.5, 23.5])

    def test_invisible_coordinates_and_flags_preserved(self):
        sk = sio.Skeleton(['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm'])
        source = sio.PredictedInstance.from_numpy(np.arange(10).reshape(5, 2), sk,
                    point_scores=np.linspace(.1, .9, 5), score=.9)
        source.points['visible'][2] = False
        source.points['complete'][1] = True
        displayed = a.display_instance(source, sio.Skeleton(sk.node_names + ['FBody', 'Burrow']))
        a.assert_group([source], [displayed])
        self.assertEqual(displayed.points['xy'][2, 0], 4.)
        self.assertFalse(displayed.points['visible'][2])

    def make_fixture(self):
        old = self.directory / 'old'
        old.mkdir()
        (old / 'DEMO_COMPLETE.json').write_text('{}')
        (old / 'README.md').write_text('Original male fixture\n')
        (old / 'frame_map.csv').write_text('source_frame\n1\n2\n3\n4\n')
        source = self.directory / 'female.slp'
        female_source(source, [[19], [20], [21], [22], [23], [24]])
        burrow_path = self.directory / 'BurrowLabels3.slp'
        bv = video(self.directory / 'clip_cam2_8k.mp4', 1000)
        bs = sio.Skeleton(['Burrow'])
        bfs = [sio.LabeledFrame(bv, frame, [sio.Instance.from_numpy(np.array([[x, 50.]]), bs)
                    for x in xs]) for frame, xs in [(563, [30., 90.]), (948, [31., 91.])]]
        sio.save_slp(sio.Labels(bfs, videos=[bv], skeletons=[bs]), burrow_path)
        chunks, obs, frames = [], [], []
        for number, start in enumerate([1, 3], 1):
            cid = f'chunk_{number:03d}'
            d = old / cid
            d.mkdir()
            movie = d / 'movie.mp4'
            writer = cv2.VideoWriter(str(movie), cv2.VideoWriter_fourcc(*'mp4v'), a.FPS, (64, 48))
            assert writer.isOpened()
            for k in range(2):
                writer.write(np.full((48, 64, 3), k * 50, np.uint8))
            writer.release()
            v = video(movie, 2)
            sk = sio.Skeleton(['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm'], edges=[('Shell', 'Shoulder')])
            track = sio.Track(name='male_original/t000000001')
            lfs, rows, frs = [], [], []
            for local in range(2):
                inst = sio.PredictedInstance.from_numpy(np.arange(10).reshape(5, 2) + local, sk,
                        point_scores=np.full(5, .9), score=.8, track=track, tracking_score=0.)
                lfs.append(sio.LabeledFrame(v, local, [inst]))
                row = dict(video_id=a.VIDEO_ID, frame=start+local, instance_index=0, raw_instance_id=start+local,
                    detection_id=f'male/f{start+local}', tracklet_id=track.name, instance_score=.8,
                    tracklet_local_id=1, body_x=float(local), body_y=float(local+1), body_score=.9,
                    accepted_for_tracking=True, centroid_source='shell_proxy')
                for j, node in enumerate(['shell', 'shoulder', 'elbow', 'claw', 'arm']):
                    row.update({f'{node}_x': float(2*j+local), f'{node}_y': float(2*j+local+1),
                                f'{node}_score': .9, f'{node}_visible': True})
                row.update(demo_frame=start+local-1, demo_time_s=(start+local-1)/a.FPS,
                    chunk_frame=local, chunk_time_s=local/a.FPS, chunk_id=cid,
                    chunk_instance_row=local, demo_instance_row=start+local-1)
                rows.append(row)
                frs.append(dict(frame=start+local, demo_frame=start+local-1, chunk_frame=local,
                                source_count=1, assigned_count=1, chunk_id=cid))
            for suffix in ['predictions', 'tracked']:
                sio.save_slp(sio.Labels(lfs, videos=[v], skeletons=[sk], tracks=[track]), d / f'movie.{suffix}.slp')
            extra = pa.schema([('demo_frame', pa.int64()), ('demo_time_s', pa.float64()),
                    ('chunk_frame', pa.int64()), ('chunk_time_s', pa.float64()), ('chunk_id', pa.string()),
                    ('chunk_instance_row', pa.int64()), ('demo_instance_row', pa.int64())])
            table = pa.Table.from_pylist(rows, schema=pa.schema(list(OBS_SCHEMA) + list(extra)))
            ft = pa.Table.from_pylist(frs)
            pq.write_table(table, d / 'observations.parquet')
            pq.write_table(ft, d / 'frames.parquet')
            obs.append(table)
            frames.append(ft)
            chunks.append(dict(chunk_id=cid, source_start=start, source_stop_exclusive=start+2,
                frames=2, demo_start=start-1, video=f'{cid}/movie.mp4', raw_slp=f'{cid}/movie.predictions.slp',
                tracked_slp=f'{cid}/movie.tracked.slp'))
        manifest = dict(source_frame_start=1, source_frame_stop_exclusive=5, frames=4, chunks=chunks)
        (old / 'chunks.json').write_text(json.dumps(manifest))
        pq.write_table(pa.concat_tables(obs), old / 'observations.parquet')
        pq.write_table(pa.concat_tables(frames), old / 'frames.parquet')
        return old, source, burrow_path, manifest

    def test_full_synthetic_augmentation_preserves_males_and_copies_burrows(self):
        old, source, burrows, chunks = self.make_fixture()
        before = {p.relative_to(old): a.sha256(p) for p in old.rglob('*') if p.is_file()}
        burrow_hash = a.sha256(burrows)
        stage = self.directory / 'new'
        original_check = a.check_portable_slp
        with patch.object(a, 'check_portable_slp', side_effect=lambda p, m, n: original_check(p, m, n, (48, 64))):
            result = a.augment(stage, old, chunks, source, burrows, config(), [[0, 0], [10, 0], [10, 10], [0, 10]])
        self.assertEqual(result['male_observations'], 4)
        self.assertEqual(result['female_observations'], 4)
        self.assertEqual(result['female_tracklets'], 1)
        self.assertEqual(result['female_links_across_chunk_boundaries'], 1)
        self.assertEqual(result['static_burrows'], 2)
        self.assertTrue(burrows.exists())
        self.assertEqual(a.sha256(stage / 'burrows' / burrows.name), burrow_hash)
        self.assertEqual(before, {p.relative_to(old): a.sha256(p) for p in old.rglob('*') if p.is_file()})
        table = pq.read_table(stage / 'observations.parquet')
        male = table.filter(pc.equal(table['entity_type'], 'male'))
        original = pq.read_table(old / 'observations.parquet')
        self.assertTrue(male.select(original.column_names).equals(original))
        self.assertEqual(set(table['sex'].to_pylist()), {'male', 'female'})
        females = table.filter(pc.equal(table['entity_type'], 'female'))
        self.assertEqual(females['shell_x'].null_count, len(females))
        self.assertEqual(females['source_chunk_path'].null_count, len(females))
        np.testing.assert_array_equal(females['frame'].to_numpy(), [1, 2, 3, 4])
        np.testing.assert_array_equal(table['observation_row'].to_numpy(), np.arange(8))
        statics = pq.read_table(stage / 'burrows.parquet')
        np.testing.assert_array_equal(statics['burrow_x'].to_numpy(), [30., 90.])
        self.assertEqual(set(statics['source_label_frame'].to_pylist()), {563})
        for chunk in json.loads((stage / 'chunks.json').read_text())['chunks']:
            labels = sio.load_slp(stage / chunk['combined_slp'], open_videos=False)
            for lf in labels:
                self.assertEqual(len(lf.instances), 4)
                self.assertEqual(sum(isinstance(i, sio.PredictedInstance) for i in lf), 2)
                self.assertEqual(sum(not isinstance(i, sio.PredictedInstance) for i in lf), 2)
            for row in table.filter(pc.equal(table['chunk_id'], chunk['chunk_id'])).to_pylist():
                lf = labels[row['chunk_frame']]
                inst = lf.instances[row['combined_instance_index']]
                self.assertEqual(inst.track.name, row['tracklet_id'])
                self.assertEqual(inst.points['xy'][0 if row['sex'] == 'male' else 5, 0], row['body_x'])

    def test_burrow_maps_are_not_merged_or_averaged(self):
        _, _, source, _ = self.make_fixture()
        first, _ = a.burrow_reference(source, 563)
        second, _ = a.burrow_reference(source, 948)
        self.assertEqual(len(first), 2)
        np.testing.assert_array_equal([i.points['xy'][0, 0] for i in first], [30., 90.])
        np.testing.assert_array_equal([i.points['xy'][0, 0] for i in second], [31., 91.])

    def test_absent_burrow_reference_fails(self):
        _, _, source, _ = self.make_fixture()
        with self.assertRaises(StopIteration):
            a.burrow_reference(source, 123)

    def test_stage_overwrite_refused(self):
        stage = self.directory / 'new'
        stage.mkdir()
        with self.assertRaises(FileExistsError):
            a.augment(stage, self.directory, {}, self.directory/'x', self.directory/'y', {}, [])

    def test_parameters_are_not_tuned_on_demo(self):
        original = yaml.safe_load((ROOT/'configs/tracking/body_first_v001.yaml').read_text())
        actual = config()
        for key in ('body_score_min', 'duplicate_shell_radius_px', 'max_body_displacement_px',
                    'max_prediction_error_px', 'motion_weight', 'minimum_alternative_margin',
                    'no_gap_bridging', 'no_reidentification'):
            self.assertEqual(actual[key], original[key])
        self.assertEqual(actual['shoulder_vector_weight'], 0.)
        self.assertIsNone(actual['shoulder_node'])


if __name__ == '__main__':
    unittest.main()
