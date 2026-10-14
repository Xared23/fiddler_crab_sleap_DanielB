"""Finite regression checks; execute only through the Slurm preflight script."""
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import sleap_io as sio
import torch
from omegaconf import OmegaConf
from pipeline_common import (atomic_json, dataset_manifest, exclusive_lock, fingerprint,
    hop_progress, hop_start, load_run, pin_manifest, read_json, save_slp_atomic, sha256)
from prepare_v007 import signature, split_indices
from train_versioned import select_checkpoint


class WorkflowChecks(unittest.TestCase):
    def setUp(self):
        base = ROOT / "checks" / "v007_19f" / "unit_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_split_is_fixed_disjoint_and_exact(self):
        train, val = split_indices(list(range(19)), 2, 42)
        self.assertEqual((len(train), len(val)), (17, 2))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(train + val, split_indices(list(range(19)), 2, 42)[0] + val)
        self.assertEqual(set(train + val), set(range(19)))

    def test_manifest_rejects_changed_inputs(self):
        pin_manifest(self.directory, {"version": 1})
        with self.assertRaises(RuntimeError):
            pin_manifest(self.directory, {"version": 2})

    def test_lock_rejects_second_worker(self):
        with exclusive_lock(self.directory):
            with self.assertRaises(RuntimeError):
                with exclusive_lock(self.directory):
                    pass

    def test_hop_cap_and_no_progress(self):
        hop_start(self.directory, 4)
        hop_progress(self.directory, 10, 2)
        hop_start(self.directory, 4)
        hop_progress(self.directory, 10, 2)
        hop_start(self.directory, 4)
        with self.assertRaises(RuntimeError):
            hop_progress(self.directory, 10, 2)
        hop_start(self.directory, 4)
        with self.assertRaises(RuntimeError):
            hop_start(self.directory, 4)

    def test_atomic_slp_keeps_previous_on_validation_failure(self):
        target = self.directory / "result.slp"
        skeleton = sio.Skeleton(["Shell"])
        video = sio.Video(filename=str(ROOT / "clip_cam2_8k.mp4"), open_backend=False)
        instance = sio.Instance.from_numpy(np.array([[1., 2.]]), skeleton=skeleton)
        labels = sio.Labels([sio.LabeledFrame(video=video, frame_idx=0, instances=[instance])])
        save_slp_atomic(labels, target, lambda _: None)
        before = sha256(target)
        def fail(_):
            raise RuntimeError("simulated read-back failure")
        with self.assertRaises(RuntimeError):
            save_slp_atomic(labels, target, fail)
        self.assertEqual(before, sha256(target))
        self.assertFalse(target.with_suffix(".tmp.slp").exists())

    def test_failed_save_keeps_previous(self):
        target = self.directory / "result.slp"
        target.write_bytes(b"previous valid output")
        class FailingLabels:
            def save(self, filename, **kwargs):
                Path(filename).write_bytes(b"partial output")
                raise OSError("simulated filesystem failure")
        with self.assertRaises(OSError):
            save_slp_atomic(FailingLabels(), target, lambda _: None)
        self.assertEqual(target.read_bytes(), b"previous valid output")

    def test_checkpoint_prefers_progress_over_mtime_and_rejects_corruption(self):
        for name, step in (("last.ckpt", 20), ("best.ckpt", 10)):
            torch.save({"state_dict": {"w": torch.tensor([1.])}, "optimizer_states": [{}],
                        "global_step": step, "epoch": 1}, self.directory / name)
        os.utime(self.directory / "best.ckpt", (2000000000, 2000000000))
        (self.directory / "interrupt.ckpt").write_bytes(b"corrupt")
        selected, progress = select_checkpoint(self.directory)
        self.assertEqual(selected.name, "last.ckpt")
        self.assertEqual(progress["global_step"], 20)
        (self.directory / "last.ckpt").unlink()
        (self.directory / "best.ckpt").unlink()
        with self.assertRaises(RuntimeError):
            select_checkpoint(self.directory)

    def test_actual_prepared_dataset_and_no_regional_changes(self):
        run = load_run("configs/v007_19f/run.yaml")
        bundle = dataset_manifest(run)
        manifest = bundle["manifest"]
        self.assertEqual(manifest["splits"]["train"]["n_frames"], 17)
        self.assertEqual(manifest["splits"]["val"]["n_frames"], 2)
        self.assertFalse(set(manifest["splits"]["train"]["frames"]) & set(manifest["splits"]["val"]["frames"]))
        original = sio.load_slp(manifest["original_backup"]["path"], open_videos=False)
        remapped = sio.load_slp(ROOT / "danlabels.v007.slp", open_videos=False)
        self.assertEqual(signature(original), signature(remapped))
        self.assertEqual(Path(remapped.videos[0].filename), ROOT / "clip_cam2_8k.mp4")
        self.assertEqual(sum(len(lf.user_instances) for lf in remapped), 2573)
        for role in ("centroid", "centered"):
            baseline = OmegaConf.load(ROOT / run["training"]["base_configs"][role])
            config = OmegaConf.load(manifest["resolved_configs"][role])
            self.assertEqual(config.model_config, baseline.model_config)
            self.assertEqual(config.data_config.preprocessing, baseline.data_config.preprocessing)
            self.assertEqual(config.data_config.augmentation_config, baseline.data_config.augmentation_config)
            self.assertFalse(config.data_config.use_same_data_for_val)
            self.assertEqual(config.trainer_config.max_epochs, 200)
            self.assertIsNone(config.trainer_config.resume_ckpt_path)
            self.assertEqual(len(config.data_config.train_labels_path), 1)
            self.assertEqual(len(config.data_config.val_labels_path), 1)


if __name__ == "__main__":
    unittest.main()
