#!/usr/bin/env python3
"""Authorized centroid-only 75-completed-epoch stage; baseline files stay pinned.

Uses the original safe training/checkpoint helpers and original run manifest.
The effective 75/75 min/max epoch override is explicit in a separately tested
extension receipt. Saving at on_train_end captures the completed loop, scheduler,
optimizer and callback state; the snapshot is NOT historical validation-best.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import signal
import sys
import traceback

from omegaconf import OmegaConf
from pipeline_common import (ROOT, atomic_json, dataset_manifest, exclusive_lock,
    fingerprint, hop_progress, hop_start, load_run, model_dir, pin_manifest,
    project_path, read_json, sha256, versions)
import train_versioned as base

RUN_CONFIG = ROOT / 'configs/female_v002/run.yaml'
POLICY = ROOT / 'configs/female_v002/stage75.json'
CHECKS = ROOT / 'checks/female_v002/stage75'
ALLOWED_CHANGES = {'scripts/female_workflow.py', 'scripts/female_train_chain.sh'}
ADDITIONS = ['scripts/female_stage75.py', 'configs/female_v002/stage75.json',
             'tests/test_female_stage75.py', 'scripts/install_female_stage75.py']


def load_policy():
    policy = read_json(POLICY)
    assert policy['schema_version'] == 1 and policy['run_id'] == 'female_v002'
    assert policy['role'] == 'centroid' and policy['completed_epochs'] == policy['minimum_epochs'] == 75
    assert policy['original_epoch_ceiling'] == 200 and policy['preserve_full_training_state'] is True
    assert policy['downstream_centroid_selection'] == 'exact_completed_epoch_75_NOT_validation_best'
    assert policy['continue_automatically'] is False and policy['centered_training_changed'] is False
    return policy


def validate_extension(receipt, preflight, actual):
    """Only two named female entry points may differ from the original preflight."""
    assert receipt['schema_version'] == 1 and receipt['run_id'] == 'female_v002'
    assert receipt['tests_passed'] is True and receipt['baseline_preflight'] == preflight
    assert receipt['active_signature'] == actual
    for key in ('run_config_sha256', 'dataset_sha256', 'implementation', 'versions'):
        assert actual[key] == preflight[key], f'Unapproved baseline change: {key}'
    old, new = preflight['female_implementation'], actual['female_implementation']
    assert set(old) == set(new)
    assert {p for p in old if old[p] != new[p]} == ALLOWED_CHANGES
    assert receipt['policy'] == load_policy()
    assert set(receipt['additions']) == set(ADDITIONS)
    for name, digest in receipt['additions'].items():
        assert sha256(ROOT / name) == digest, f'Extension code/policy changed: {name}'


def verify_extension(preflight, actual):
    receipt = read_json(CHECKS / 'APPROVED.json')
    validate_extension(receipt, preflight, actual)
    tests = CHECKS / 'TESTS.json'
    assert sha256(tests) == receipt['test_report_sha256'] and read_json(tests)['passed'] is True
    return receipt


def full_checkpoint(path, completed=None):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('state_dict', 'optimizer_states', 'lr_schedulers', 'callbacks', 'loops'):
        if not state.get(key):
            raise RuntimeError(f'Missing full training state {key}: {path}')
    assert any('EarlyStopping' in key for key in state['callbacks'])
    count = int(state['loops']['fit_loop']['epoch_progress']['total']['completed'])
    if completed is not None:
        assert count == completed and int(state['epoch']) == completed
    steps = [float(s['step']) for opt in state['optimizer_states'] for s in opt['state'].values() if 'step' in s]
    assert steps and min(steps) == max(steps) == int(state['global_step'])
    return {'epoch_counter': int(state['epoch']), 'completed_epochs': count,
            'global_step': int(state['global_step']), 'optimizer_step_range': [min(steps), max(steps)]}


def read_stage(directory, completed=75):
    destination = directory / 'stages' / f'epoch_{completed:04d}'
    marker = read_json(destination / 'STAGE_COMPLETE.json')
    assert marker['completed_epochs'] == completed
    assert marker['checkpoint_sha256'] == sha256(destination / 'best.ckpt')
    assert marker['training_config_sha256'] == sha256(destination / 'training_config.yaml')
    assert marker['progress'] == full_checkpoint(destination / 'best.ckpt', completed)
    assert marker['manifest_sha256'] == sha256(directory / 'manifest.json')
    return destination, marker


def save_stage(trainer, directory, completed=75):
    """Atomic full-state snapshot after ALL target epochs and scheduler updates."""
    import tempfile
    parent = directory / 'stages'
    parent.mkdir(exist_ok=True)
    destination = parent / f'epoch_{completed:04d}'
    if destination.exists():
        return read_stage(directory, completed)[0]
    assert int(trainer.current_epoch) == completed
    assert int(trainer.fit_loop.epoch_progress.total.completed) == completed
    temp = Path(tempfile.mkdtemp(prefix=f'.epoch_{completed:04d}.', dir=parent))
    try:
        path = temp / 'best.ckpt'
        trainer.save_checkpoint(str(path))
        progress = full_checkpoint(path, completed)
        assert progress['global_step'] == int(trainer.global_step)
        for name in ('training_config.yaml', 'manifest.json', 'training_log.csv', 'stage75_active.json'):
            source = directory / name
            if source.is_file():
                shutil.copy2(source, temp / name)
        marker = {'schema_version': 1, 'kind': 'exact_completed_epoch_weights_NOT_validation_best',
            'completed_epochs': completed, 'progress': progress,
            'checkpoint_sha256': sha256(path), 'training_config_sha256': sha256(temp / 'training_config.yaml'),
            'manifest_sha256': sha256(directory / 'manifest.json'),
            'validation_loss': float(trainer.callback_metrics['val/loss']),
            'job_id': os.environ.get('SLURM_JOB_ID'),
            'note': 'Saved at on_train_end after loop completed, so checkpoint epoch counter is 75, not the male on_train_epoch_end index 74. Both represent 75 completed epochs; complete loop/optimizer/scheduler/callback state is retained.'}
        atomic_json(temp / 'STAGE_COMPLETE.json', marker)
        temp.rename(destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)
    print(f'FEMALE_STAGE_SAVED completed_epochs={completed} step={trainer.global_step} path={destination}', flush=True)
    return destination


def stage_callback(directory, target=75):
    from lightning.pytorch.callbacks import Callback
    class StageCheckpoint(Callback):
        def on_train_end(self, trainer, module):
            # A failure is fatal: never release downstream jobs without the
            # exact, readable, fully resumable requested model.
            save_stage(trainer, directory, target)
    return StageCheckpoint()


def publish_completion(directory, manifest_hash, policy):
    destination, marker = read_stage(directory, policy['completed_epochs'])
    best = directory / 'best.ckpt'
    best_progress = base.checkpoint_summary(best)
    complete = {'manifest_sha256': manifest_hash, 'best_sha256': sha256(best),
        'epoch': 75, 'global_step': marker['progress']['global_step'], 'best_progress': best_progress,
        'reason': 'user_requested_75_completed_epoch_stage',
        'not_original_200_epoch_ceiling': True,
        'stage_checkpoint': str(destination.relative_to(ROOT)),
        'stage_marker_sha256': sha256(destination / 'STAGE_COMPLETE.json'),
        'downstream_uses': 'exact_epoch75_snapshot_NOT_canonical_validation_best',
        'continuation': 'Only on a new explicit user request; all full-state checkpoint data preserved.'}
    atomic_json(directory / 'TRAINING_COMPLETE.json', complete)
    print('FEMALE_75_EPOCH_STAGE_COMPLETE', complete, flush=True)


def train():
    from female_workflow import verify_preflight
    policy = load_policy()
    run = load_run(RUN_CONFIG)
    verify_preflight(run)
    dataset = dataset_manifest(run)
    info = dataset['manifest']['video']
    st = project_path(info['path']).stat()
    assert (st.st_size, st.st_mtime_ns) == (info['bytes'], info['mtime_ns'])
    config_path = project_path(dataset['manifest']['resolved_configs']['centroid'])
    config = OmegaConf.load(config_path)
    directory = model_dir(run, 'centroid')
    training = run['training']
    signal.signal(signal.SIGUSR1, base.request_stop)
    signal.signal(signal.SIGTERM, base.request_stop)
    with exclusive_lock(directory):
        # Validate the ORIGINAL manifest; do not rewrite historical scientific
        # inputs or pretend that the old GPU preflight tested this new policy.
        expected = {'run_id': run['run_id'], 'role': 'centroid', 'smoke': False,
            'dataset_sha256': fingerprint(dataset), 'config_sha256': sha256(config_path),
            'config': OmegaConf.to_container(config, resolve=True), 'versions': versions(),
            'implementation': base.implementation_hashes(), 'training_policy': training}
        manifest_hash = pin_manifest(directory, expected)
        active = {'policy': policy, 'approval_sha256': sha256(CHECKS / 'APPROVED.json'),
                  'baseline_manifest_sha256': manifest_hash, 'worker_sha256': sha256(Path(__file__)),
                  'activation_job_id': os.environ.get('SLURM_JOB_ID')}
        atomic_json(directory / 'stage75_active.json', active)
        if (directory / 'stages/epoch_0075/STAGE_COMPLETE.json').exists():
            publish_completion(directory, manifest_hash, policy)
            return 0
        if (directory / 'TRAINING_COMPLETE.json').exists():
            raise RuntimeError('Unexpected earlier completion; do not silently resume/relabel it')
        hop = hop_start(directory, training['max_hops'])
        resume, progress = base.select_checkpoint(directory)
        if resume is None or progress['global_step'] <= 0 or progress['epoch'] >= 75:
            raise RuntimeError(f'Expected existing in-progress weights below epoch 75: {resume} {progress}')
        print(f'Hop {hop}: female centroid75 resume={resume} progress={progress} effective_epochs=75', flush=True)
        config.trainer_config.max_epochs = 75  # runtime extension; baseline YAML remains 200
        config.trainer_config.resume_ckpt_path = str(resume)
        import torch
        assert torch.cuda.is_available(), 'CUDA required'
        from gputest_train import _disable_lightning_scontrol_requeue, _patch_epoch_timers, _keep_csv_history
        _disable_lightning_scontrol_requeue()
        _patch_epoch_timers()
        _keep_csv_history()
        base.patch_lightning(directory, training['checkpoint_every_steps'], False, None,
                             stage_callback(directory))
        # Honor the user's EXACT target even if early stopping would otherwise
        # fire earlier. Lightning min_epochs defers stopping but still records
        # the original patience/wait/scheduler state in checkpoints.
        import lightning.pytorch as L
        original_init = L.Trainer.__init__
        def init(self, *args, **kwargs):
            kwargs['min_epochs'] = 75
            kwargs['max_epochs'] = 75
            original_init(self, *args, **kwargs)
        L.Trainer.__init__ = init
        from sleap_nn.training.model_trainer import ModelTrainer
        class StableStageTrainer(ModelTrainer):
            def _setup_ckpt_path(self):
                actual = Path(self.config.trainer_config.ckpt_dir) / self.config.trainer_config.run_name
                assert actual.resolve() == directory.resolve()
                assert self.config.data_config.data_pipeline_fw == 'torch_dataset_cache_img_memory'
        worker = StableStageTrainer.get_model_trainer_from_config(config)
        try:
            if base.STOP_REQUESTED:
                raise base.HopStop('Stop requested during setup')
            worker.train()
        except base.HopStop as error:
            print(str(error), flush=True)
            _, saved = base.select_checkpoint(directory)
            hop_progress(directory, saved['global_step'], training['max_no_progress_hops'])
            return 75
        assert worker.trainer.current_epoch == 75
        _, marker = read_stage(directory)
        hop_progress(directory, marker['progress']['global_step'], training['max_no_progress_hops'])
        publish_completion(directory, manifest_hash, policy)
        return 0


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    try:
        sys.exit(train())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
