"""Finite CPU Slurm tests for the additive 75/100 epoch snapshot extension."""
import copy
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
from pipeline_common import read_json, sha256
from milestone_checkpoints import make_milestone_callback, save_snapshot, validate_extension

CHECKS = ROOT / "checks" / "v007_19f" / "milestones_75_100"


class FakeTrainer:
    current_epoch = 74
    global_step = 15030
    callback_metrics = {"val/loss": torch.tensor(0.00004)}
    optimizers = [type("Optimizer", (), {"param_groups": [{"lr": .0001}]})()]
    def save_checkpoint(self, path):
        torch.save({"state_dict": {"w": torch.tensor([float(self.current_epoch)])},
                    "optimizer_states": [{"state": {}}], "epoch": self.current_epoch,
                    "global_step": self.global_step}, path)


class MilestoneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=CHECKS)
        self.directory = Path(self.temp.name)
        (self.directory / "training_config.yaml").write_text("model_config: {}\n")
        self.trainer = FakeTrainer()

    def tearDown(self):
        self.temp.cleanup()

    def test_targets_mean_completed_epochs_not_zero_based_indices(self):
        callback = make_milestone_callback(self.directory)
        callback.on_fit_start(self.trainer, None)
        self.trainer.current_epoch = 73
        callback.on_train_epoch_end(self.trainer, None)
        self.assertFalse((self.directory / "milestones").exists())
        for index, completed in ((74, 75), (99, 100)):
            self.trainer.current_epoch = index
            self.trainer.global_step = 200 * completed + 30
            callback.on_train_epoch_end(self.trainer, None)
            path = self.directory / "milestones" / f"epoch_{completed:04d}"
            saved = torch.load(path / "best.ckpt", weights_only=False)
            self.assertEqual(saved["epoch"], index)
            self.assertEqual(saved["global_step"], self.trainer.global_step)
            self.assertEqual(read_json(path / "snapshot.json")["completed_epochs"], completed)
            self.assertTrue((path / "training_config.yaml").is_file())

    def test_snapshot_is_immutable_and_does_not_consume_rng(self):
        random.seed(12)
        np.random.seed(12)
        torch.manual_seed(12)
        before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        path = save_snapshot(self.trainer, self.directory, 75, {})
        original_hash = sha256(path / "best.ckpt")
        self.trainer.global_step += 9
        save_snapshot(self.trainer, self.directory, 75, {})
        self.assertEqual(original_hash, sha256(path / "best.ckpt"))
        self.assertEqual(before[0], random.getstate())
        np.testing.assert_array_equal(before[1][1], np.random.get_state()[1])
        torch.testing.assert_close(before[2], torch.get_rng_state())

    def test_partial_save_is_not_published_and_training_callback_does_not_raise(self):
        callback = make_milestone_callback(self.directory)
        callback.on_fit_start(self.trainer, None)
        def fail(path):
            Path(path).write_bytes(b"partial")
            raise OSError("simulated write failure")
        with patch.object(self.trainer, "save_checkpoint", side_effect=fail):
            callback.on_train_epoch_end(self.trainer, None)
        parent = self.directory / "milestones"
        self.assertFalse((parent / "epoch_0075").exists())
        self.assertFalse((parent / ".epoch_0075.tmp").exists())
        self.assertTrue((parent / "epoch_0075.FAILED.json").is_file())

    def test_receipt_rejects_drift(self):
        preflight = read_json(ROOT / "checks/v007_19f/PREFLIGHT_COMPLETE.json")
        spec = importlib.util.spec_from_file_location("milestone_candidate", CHECKS / "train_versioned.candidate.py")
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        actual = candidate.implementation_hashes()
        receipt = {"schema_version": 1, "run_id": "v007_19f", "roles": ["centroid"],
                   "completed_epochs": [75, 100], "baseline_implementation": preflight["implementation"],
                   "implementation": actual, "tests_passed": True,
                   "run_config_sha256": preflight["run_config_sha256"],
                   "dataset_sha256": preflight["dataset_sha256"]}
        validate_extension(receipt, preflight, actual, "v007_19f")
        for key, value in (("completed_epochs", [76, 100]), ("dataset_sha256", "changed"), ("tests_passed", False)):
            bad = copy.deepcopy(receipt)
            bad[key] = value
            with self.assertRaises(RuntimeError):
                validate_extension(bad, preflight, actual, "v007_19f")
        bad_actual = dict(actual, **{"scripts/pipeline_common.py": "changed"})
        bad_receipt = dict(receipt, implementation=bad_actual)
        with self.assertRaises(RuntimeError):
            validate_extension(bad_receipt, preflight, bad_actual, "v007_19f")

    def test_real_lightning_epoch_end_save_and_unchanged_optimization(self):
        import lightning.pytorch as L
        from torch.utils.data import DataLoader, TensorDataset
        spec = importlib.util.spec_from_file_location("milestone_candidate_real", CHECKS / "train_versioned.candidate.py")
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        class Toy(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.5))
            def training_step(self, batch, batch_idx):
                return ((self.weight * batch[0] - 1) ** 2).mean()
            def validation_step(self, batch, batch_idx):
                self.log("val/loss", ((self.weight * batch[0] - 1) ** 2).mean())
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=.1)
        data = DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=1)
        kwargs = dict(accelerator="cpu", devices=1, max_epochs=2, logger=False,
                      enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False, limit_train_batches=2,
                      limit_val_batches=1, num_sanity_val_steps=0)
        baseline = Toy()
        L.Trainer(**kwargs).fit(baseline, data, data)
        original_init = L.Trainer.__init__
        try:
            callback = make_milestone_callback(self.directory, targets=(1, 2))
            candidate.patch_lightning(self.directory, 10, False, None, callback)
            model = Toy()
            trainer = L.Trainer(**kwargs)
            trainer.fit(model, data, data)
        finally:
            L.Trainer.__init__ = original_init
        torch.testing.assert_close(model.weight, baseline.weight, rtol=0, atol=0)
        for completed in (1, 2):
            saved = torch.load(self.directory / "milestones" / f"epoch_{completed:04d}" / "best.ckpt", weights_only=False)
            self.assertEqual(saved["epoch"], completed - 1)
            self.assertEqual(saved["global_step"], completed * 2)
        torch.testing.assert_close(saved["state_dict"]["weight"], model.weight)


if __name__ == "__main__":
    unittest.main()
