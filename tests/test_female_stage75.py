"""Finite CPU regression checks for the user-requested exact-75 epoch stop."""
from __future__ import annotations
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
from pipeline_common import atomic_json, fingerprint, read_json, sha256
import female_stage75 as stage


class FakeTrainer:
    current_epoch = 75
    global_step = 15025
    callback_metrics = {'val/loss': torch.tensor(.0001)}
    fit_loop = type('Fit', (), {'epoch_progress': type('Epoch', (), {
        'total': type('Total', (), {'completed': 75})()})()})()
    def save_checkpoint(self, path):
        torch.save({'epoch': self.current_epoch, 'global_step': self.global_step,
            'state_dict': {'w': torch.tensor([75.])},
            'optimizer_states': [{'state': {0: {'step': torch.tensor(float(self.global_step)),
                'exp_avg': torch.tensor([.1]), 'exp_avg_sq': torch.tensor([.2])}}}],
            'lr_schedulers': [{'last_epoch': 75}],
            'callbacks': {'EarlyStopping': {'wait_count': 42}},
            'loops': {'fit_loop': {'epoch_progress': {'total': {'completed': self.current_epoch}}}}}, path)


class Stage75Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'])
        self.directory = Path(self.temp.name)
        (self.directory / 'training_config.yaml').write_text('trainer_config:\n  max_epochs: 75\n')
        atomic_json(self.directory / 'manifest.json', {'baseline': 'unchanged'})
        self.fake = FakeTrainer()

    def tearDown(self):
        self.temp.cleanup()

    def test_exact75_full_state_and_baseline_best_is_untouched(self):
        best = self.directory / 'best.ckpt'
        best.write_bytes(b'preserved historical best')
        before = sha256(best)
        target = stage.save_stage(self.fake, self.directory)
        progress = stage.full_checkpoint(target / 'best.ckpt', 75)
        self.assertEqual(progress['completed_epochs'], 75)
        self.assertEqual(progress['epoch_counter'], 75)
        self.assertEqual(progress['global_step'], 15025)
        self.assertEqual(sha256(best), before)
        self.assertEqual(stage.read_stage(self.directory)[0], target)
        first = sha256(target / 'best.ckpt')
        stage.save_stage(self.fake, self.directory)
        self.assertEqual(first, sha256(target / 'best.ckpt'))

    def test_wrong_epoch_or_incomplete_checkpoint_refused(self):
        self.fake.current_epoch = 74
        with self.assertRaises(AssertionError):
            stage.save_stage(self.fake, self.directory)
        self.fake.current_epoch = 75
        target = stage.save_stage(self.fake, self.directory)
        saved = torch.load(target / 'best.ckpt', weights_only=False)
        del saved['lr_schedulers']
        torch.save(saved, target / 'best.ckpt')
        with self.assertRaises(RuntimeError):
            stage.full_checkpoint(target / 'best.ckpt', 75)
        with self.assertRaises(AssertionError):
            stage.read_stage(self.directory)

    def test_failed_checkpoint_is_fatal_and_never_published(self):
        def fail(path):
            Path(path).write_bytes(b'partial')
            raise OSError('simulated save failure')
        with patch.object(self.fake, 'save_checkpoint', side_effect=fail):
            with self.assertRaises(OSError):
                stage.stage_callback(self.directory).on_train_end(self.fake, None)
        self.assertFalse((self.directory / 'stages/epoch_0075').exists())
        self.assertFalse(list((self.directory / 'stages').iterdir()))

    def test_extension_receipt_rejects_unrelated_drift(self):
        preflight = read_json(ROOT / 'checks/female_v002/PREFLIGHT_COMPLETE.json')
        actual = read_json(stage.CHECKS / 'CANDIDATE_SIGNATURE.json')
        receipt = {'schema_version': 1, 'run_id': 'female_v002', 'tests_passed': True,
            'baseline_preflight': preflight, 'active_signature': actual,
            'policy': stage.load_policy(), 'additions': {p: sha256(ROOT / p) for p in stage.ADDITIONS}}
        stage.validate_extension(receipt, preflight, actual)
        for key in ('dataset_sha256', 'run_config_sha256', 'implementation', 'versions'):
            bad = copy.deepcopy(actual)
            bad[key] = 'changed'
            with self.assertRaises(AssertionError):
                stage.validate_extension(dict(receipt, active_signature=bad), preflight, bad)
        bad = copy.deepcopy(actual)
        bad['female_implementation']['scripts/female_prediction_io.py'] = 'changed'
        with self.assertRaises(AssertionError):
            stage.validate_extension(dict(receipt, active_signature=bad), preflight, bad)
        with self.assertRaises(AssertionError):
            stage.validate_extension(dict(receipt, policy=dict(stage.load_policy(), completed_epochs=74)), preflight, actual)

    def candidate_workflow(self):
        spec = importlib.util.spec_from_file_location('female_workflow_candidate',
            stage.CHECKS / 'candidates/female_workflow.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_downstream_selects_exact_epoch75_not_earlier_best(self):
        workflow = self.candidate_workflow()
        models = self.directory / 'models/female_v002'
        for role in ('centroid', 'centered'):
            directory = models / role
            directory.mkdir(parents=True)
            manifest = {'dataset_sha256': fingerprint({}), 'implementation': {}}
            atomic_json(directory / 'manifest.json', manifest)
            (directory / 'training_config.yaml').write_text('model_config: {}\n')
            trainer = FakeTrainer()
            trainer.current_epoch = 27
            trainer.global_step = 5600
            trainer.save_checkpoint(directory / 'best.ckpt')
            marker = {'manifest_sha256': fingerprint(manifest),
                      'best_sha256': sha256(directory / 'best.ckpt')}
            if role == 'centroid':
                target = stage.save_stage(self.fake, directory)
                marker.update({'reason': 'user_requested_75_completed_epoch_stage',
                    'stage_checkpoint': str(target.relative_to(self.directory)),
                    'stage_marker_sha256': sha256(target / 'STAGE_COMPLETE.json')})
            atomic_json(directory / 'TRAINING_COMPLETE.json', marker)
        with patch.object(workflow, 'ROOT', self.directory), \
             patch.object(workflow, 'model_dir', side_effect=lambda run, role: models / role), \
             patch.object(workflow, 'dataset_manifest', return_value={}), \
             patch.object(workflow, 'input_signature', return_value={'implementation': {}}):
            selected = workflow.freeze_models({'run_id': 'female_v002'})
        self.assertEqual(selected['centroid']['selection_kind'], 'exact_epoch75_weights')
        a = self.directory / selected['centroid']['directory'] / 'best.ckpt'
        self.assertEqual(sha256(a), sha256(models / 'centroid/stages/epoch_0075/best.ckpt'))
        self.assertNotEqual(sha256(a), sha256(models / 'centroid/best.ckpt'))
        b = self.directory / selected['centered']['directory'] / 'best.ckpt'
        self.assertEqual(sha256(b), sha256(models / 'centered/best.ckpt'))

    def test_real_lightning_exact_stop_full_resume_and_unchanged_optimization(self):
        import lightning.pytorch as L
        from lightning.pytorch.callbacks import EarlyStopping
        from torch.utils.data import DataLoader, TensorDataset
        class Toy(L.LightningModule):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.5))
            def training_step(self, batch, batch_idx):
                return ((self.weight*batch[0]-1)**2).mean()
            def validation_step(self, batch, batch_idx):
                self.log('val/loss', torch.tensor(1.))  # intentionally early-stop plateau
            def configure_optimizers(self):
                opt = torch.optim.Adam(self.parameters(), lr=.01)
                return {'optimizer': opt, 'lr_scheduler': torch.optim.lr_scheduler.StepLR(opt, 1, gamma=.99)}
        data = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
        kwargs = dict(accelerator='cpu', devices=1, logger=False,
            enable_checkpointing=False, enable_progress_bar=False,
            enable_model_summary=False, num_sanity_val_steps=0)
        def trainer(epochs, extra=()):
            early = EarlyStopping('val/loss', patience=2, check_on_train_epoch_end=False)
            return L.Trainer(**kwargs, min_epochs=epochs, max_epochs=epochs,
                             callbacks=[early, *extra])
        reference75 = Toy()
        trainer(75).fit(reference75, data, data)
        model = Toy()
        worker = trainer(75, [stage.stage_callback(self.directory)])
        worker.fit(model, data, data)
        self.assertEqual(worker.current_epoch, 75)
        self.assertEqual(worker.global_step, 150)
        torch.testing.assert_close(model.weight, reference75.weight, rtol=0, atol=0)
        path, marker = stage.read_stage(self.directory)
        state = torch.load(path / 'best.ckpt', weights_only=False)
        self.assertEqual(marker['progress']['completed_epochs'], 75)
        self.assertTrue(any(v.get('wait_count', 0) > 2 for v in state['callbacks'].values()))
        reference76 = Toy()
        reference_trainer76 = trainer(76)
        reference_trainer76.fit(reference76, data, data)
        resumed = Toy()
        resume_trainer = trainer(76)
        resume_trainer.fit(resumed, data, data, ckpt_path=str(path / 'best.ckpt'))
        self.assertEqual(resume_trainer.current_epoch, 76)
        self.assertEqual(resume_trainer.global_step, 152)
        torch.testing.assert_close(resumed.weight, reference76.weight, rtol=0, atol=0)
        self.assertEqual(resume_trainer.lr_scheduler_configs[0].scheduler.state_dict(),
                         reference_trainer76.lr_scheduler_configs[0].scheduler.state_dict())
        self.assertAlmostEqual(resume_trainer.optimizers[0].param_groups[0]['lr'],
                               .01 * .99**76)
        for key, value in resume_trainer.optimizers[0].state[resumed.weight].items():
            torch.testing.assert_close(value, reference_trainer76.optimizers[0].state[reference76.weight][key], rtol=0, atol=0)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    unittest.main()
