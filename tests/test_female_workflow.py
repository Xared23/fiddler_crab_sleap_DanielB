"""Female-specific controls. Run only in the finite Slurm preflight."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
import sleap_io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from pipeline_common import dataset_manifest, load_run, sha256
from prepare_v007 import signature, split_indices
from female_prediction_io import inspect_chunk, merge_raw
from female_workflow import matched_distances, metrics, chunk_receipt, code_digest


class FemaleDataTests(unittest.TestCase):
    def test_fixed_whole_frame_holdout(self):
        indices = list(range(16))
        a, b = split_indices(indices, 2, 42)
        self.assertEqual((len(a), len(b)), (14, 2))
        self.assertEqual((a, b), split_indices(indices, 2, 42))
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a+b), set(indices))

    def test_actual_female_split_and_unchanged_upload(self):
        run = load_run('configs/female_v002/run.yaml')
        ds = dataset_manifest(run)['manifest']
        self.assertEqual(sha256(ROOT / run['dataset']['source']), run['dataset']['collaborator_sha256'])
        original = sio.load_slp(ROOT / run['dataset']['source'], open_videos=False)
        original_frames = {int(lf.frame_idx): signature(sio.Labels([lf])) for lf in original}
        seen = set()
        n = 0
        for split, count in [('train', 14), ('val', 2)]:
            labels = sio.load_slp(ROOT / 'data/labels/female_v002' / f'{split}.slp', open_videos=False)
            self.assertEqual(len(labels), count)
            self.assertEqual(labels.skeletons[0].node_names, ['FBody'])
            self.assertEqual(Path(labels.videos[0].filename), ROOT / 'clip_cam2_8k.mp4')
            for lf in labels:
                self.assertNotIn(int(lf.frame_idx), seen)
                seen.add(int(lf.frame_idx))
                self.assertEqual(signature(sio.Labels([lf])), original_frames[int(lf.frame_idx)])
                n += len(lf.user_instances)
        self.assertEqual(n, 1369)
        self.assertEqual(seen, set(original_frames))
        self.assertEqual(ds['splits']['val']['n_frames'], 2)

    def test_female_model_heads_and_inherited_training_logic(self):
        run = load_run('configs/female_v002/run.yaml')
        for role in ('centroid', 'centered'):
            base = OmegaConf.load(ROOT / run['training']['base_configs'][role])
            cfg = OmegaConf.load(ROOT / 'configs/female_v002' / f'{role}.yaml')
            self.assertEqual(cfg.data_config.skeletons[0].nodes[0].name, 'FBody')
            self.assertEqual(len(cfg.data_config.skeletons[0].nodes), 1)
            self.assertEqual(cfg.data_config.preprocessing, base.data_config.preprocessing)
            self.assertEqual(cfg.data_config.augmentation_config, base.data_config.augmentation_config)
            self.assertEqual(cfg.model_config.backbone_config, base.model_config.backbone_config)
            self.assertEqual(cfg.trainer_config.lr_scheduler, base.trainer_config.lr_scheduler)
            self.assertEqual(cfg.trainer_config.max_epochs, 200)
            self.assertEqual(cfg.trainer_config.early_stopping.patience, 40)
            self.assertEqual(cfg.trainer_config.train_steps_per_epoch, 200)
            self.assertFalse(cfg.trainer_config.eval.enabled)
            self.assertFalse(cfg.data_config.use_same_data_for_val)
            self.assertIsNone(cfg.trainer_config.resume_ckpt_path)
            name = 'centroid' if role == 'centroid' else 'centered_instance'
            head = cfg.model_config.head_configs[name].confmaps
            self.assertEqual(head.anchor_part, 'FBody')
            if role == 'centered':
                self.assertEqual(list(head.part_names), ['FBody'])

    def test_matching_maximum_cardinality_one_to_one(self):
        d = matched_distances([[0, 0], [2, 0]], [[1, 0], [-1, 0]], 1.1)
        np.testing.assert_allclose(d, [1, 1])
        self.assertEqual(len(matched_distances([[0, 0], [0, 0]], [[0, 0]], 1)), 1)
        self.assertEqual(len(matched_distances([[0, 0]], [[100, 0]], 20)), 0)

    def test_matching_empty_boundaries_and_metrics(self):
        self.assertEqual(len(matched_distances([], [[0, 0]], 1)), 0)
        self.assertEqual(len(matched_distances([[0, 0]], [], 1)), 0)
        np.testing.assert_allclose(matched_distances([[0, 0]], [[3, 4]], 5), [5])
        m = metrics(4, 5, [1, 2, 3])
        self.assertEqual(m['false_negatives'], 1)
        self.assertEqual(m['false_positives_relative_to_labels'], 2)
        self.assertAlmostEqual(m['precision'], .6)
        self.assertAlmostEqual(m['recall'], .75)
        self.assertAlmostEqual(m['f1'], 2/3)
        self.assertIsNone(metrics(0, 0, [])['f1'])
        with self.assertRaises(AssertionError):
            matched_distances([[np.nan, 0]], [[0, 0]], 1)


class FemaleIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.directory = Path(self.temp.name)
        self.video_path = self.directory / 'synthetic.mp4'
        writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*'mp4v'), 30., (64, 64))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.zeros((64, 64, 3), np.uint8))
        writer.release()
        self.video = sio.load_video(str(self.video_path))
        self.skeleton = sio.Skeleton(['FBody'])
        self.chunks = []
        for start, stop in [(0, 2), (2, 4)]:
            frames = []
            for frame in range(start, stop):
                instances = [] if frame in (0, 2) else [sio.PredictedInstance.from_numpy(
                    np.array([[10.+frame, 20.]]), skeleton=self.skeleton,
                    point_scores=np.array([.9]), score=.8)]
                frames.append(sio.LabeledFrame(self.video, frame, instances))
            labels = sio.Labels(frames, videos=[self.video], skeletons=[self.skeleton])
            path = self.directory / f'chunk_{start}.slp'
            labels.save(str(path), embed=False, format='slp')
            self.chunks.append((path, start, stop))
        self.target = self.directory / 'merged.slp'

    def tearDown(self):
        self.temp.cleanup()

    def test_real_one_node_slp_merge_empty_frames_and_measurement_bytes(self):
        result = merge_raw(self.chunks, self.target, 4, self.video_path, ['FBody'])
        self.assertEqual(result['instances'], 2)
        labels = sio.load_slp(self.target, open_videos=False)
        self.assertEqual([len(lf.instances) for lf in labels], [0, 1, 0, 1])
        self.assertEqual([lf.frame_idx for lf in labels], [0, 1, 2, 3])
        self.assertFalse(labels.tracks)
        self.assertEqual(labels.skeletons[0].node_names, ['FBody'])
        with h5py.File(self.target, 'r') as merged:
            for j, (path, _, _) in enumerate(self.chunks):
                with h5py.File(path, 'r') as chunk:
                    self.assertEqual(merged['pred_points'][j:j+1].tobytes(), chunk['pred_points'][:].tobytes())

    def test_all_empty_chunk_skeleton_survives(self):
        labels = sio.Labels([sio.LabeledFrame(self.video, i, []) for i in range(4)],
                            videos=[self.video], skeletons=[self.skeleton])
        path = self.directory / 'empty.slp'
        labels.save(str(path), embed=False, format='slp')
        result = merge_raw([(path, 0, 4)], self.target, 4, self.video_path, ['FBody'])
        self.assertEqual(result['instances'], 0)
        self.assertEqual(sio.load_slp(self.target, open_videos=False).skeletons[0].node_names, ['FBody'])

    def test_mismatched_skeleton_or_coverage_refused(self):
        path, start, stop = self.chunks[0]
        with self.assertRaises(RuntimeError):
            inspect_chunk(path, start, stop, self.video_path, ['Shell'])
        with self.assertRaises(RuntimeError):
            inspect_chunk(path, 1, 3, self.video_path, ['FBody'])
        with self.assertRaises(RuntimeError):
            merge_raw(self.chunks[1:], self.target, 4, self.video_path, ['FBody'])
        self.assertFalse(self.target.exists())

    def test_corrupted_point_offsets_refused(self):
        path, start, stop = self.chunks[0]
        with h5py.File(path, 'r+') as f:
            a = f['instances'][:]
            a['point_id_end'] += 1
            f['instances'][:] = a
        with self.assertRaises((AssertionError, ValueError)):
            inspect_chunk(path, start, stop, self.video_path, ['FBody'])

    def test_existing_final_not_overwritten(self):
        merge_raw(self.chunks, self.target, 4, self.video_path, ['FBody'])
        before = sha256(self.target)
        with self.assertRaises(FileExistsError):
            merge_raw(self.chunks, self.target, 4, self.video_path, ['FBody'])
        self.assertEqual(sha256(self.target), before)

    def test_receipt_rejects_wrong_model_signature(self):
        from pipeline_common import atomic_json
        path, start, stop = self.chunks[0]
        atomic_json(path.with_suffix('.json'), {'signature': 'a', 'start': start,
                    'stop': stop, 'sha256': sha256(path)})
        with self.assertRaises(AssertionError):
            chunk_receipt(path, 'b', start, stop, self.video_path, ['FBody'])

    def test_only_slurm_dependency_metadata_excluded_from_hash(self):
        import female_workflow
        path = self.directory / 'jobs/check.sh'
        path.parent.mkdir()
        path.write_text('#!/bin/bash\n#SBATCH --dependency=afterok:1\necho safe\n')
        with patch.object(female_workflow, 'ROOT', self.directory):
            first = code_digest('jobs/check.sh')
            path.write_text('#!/bin/bash\n#SBATCH --dependency=afterok:2\necho safe\n')
            self.assertEqual(code_digest('jobs/check.sh'), first)
            path.write_text('#!/bin/bash\n#SBATCH --dependency=afterok:2\necho changed\n')
            self.assertNotEqual(code_digest('jobs/check.sh'), first)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    unittest.main()
