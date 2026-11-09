#!/usr/bin/env python3
"""Prepare female_v002 without changing uploaded SLPs or any male run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
from omegaconf import OmegaConf
import sleap_io as sio
from pipeline_common import (ROOT, atomic_json, dataset_dir, dataset_manifest,
    exclusive_lock, fingerprint, load_run, pin_manifest, project_path,
    save_slp_atomic, sha256, versions)
from prepare_v007 import signature, split_indices, validate_split


def prepare(config):
    run = load_run(config)
    cfg = run['dataset']
    directory = dataset_dir(run)
    with exclusive_lock(directory):
        if (directory / 'DATASET_COMPLETE.json').exists():
            dataset_manifest(run)
            print('Existing female dataset validated; no overwrite')
            return
        source, video_path = project_path(cfg['source']), project_path(cfg['video'])
        if sha256(source) != cfg['collaborator_sha256']:
            raise RuntimeError('Uploaded labels differ from audited source')
        backup = directory / 'source' / source.name
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(source, backup)
        if sha256(backup) != cfg['collaborator_sha256']:
            raise RuntimeError('Backup integrity failure')
        original = sio.load_slp(source, open_videos=False)
        labels = sio.load_slp(backup, open_videos=False)
        if len(labels.videos) != 1 or len(labels.skeletons) != 1:
            raise RuntimeError('Expected one source video and skeleton')
        if labels.skeletons[0].node_names != cfg['nodes'] or cfg['anchor_part'] not in cfg['nodes']:
            raise RuntimeError('Female skeleton/anchor mismatch')
        if any(lf.predicted_instances for lf in labels):
            raise RuntimeError('Unexpected predictions in human labels')
        recorded_shape = labels.videos[0].backend_metadata['shape']
        if Path(labels.videos[0].filename).name != video_path.name:
            raise RuntimeError('Video basename does not match uploaded reference')
        original_signature = signature(original)
        labels.videos[0].replace_filename(video_path)
        actual_shape = list(labels.videos[0].shape)
        if actual_shape != recorded_shape:
            raise RuntimeError(f'Video shape differs: {actual_shape} != {recorded_shape}')
        frames = sorted((lf for lf in labels if lf.user_instances), key=lambda lf: lf.frame_idx)
        if len(frames) != cfg['expected_user_frames']:
            raise RuntimeError('Unexpected human frame count')
        if sum(len(lf.user_instances) for lf in frames) != cfg['expected_user_instances']:
            raise RuntimeError('Unexpected human instance count')
        for lf in frames:
            if not 0 <= lf.frame_idx < actual_shape[0] or list(lf.image.shape) != actual_shape[1:]:
                raise RuntimeError(f'Cannot decode source frame {lf.frame_idx}')
            for inst in lf.user_instances:
                xy = inst.numpy()
                if not np.isfinite(xy).all() or not ((xy >= 0).all() and (xy[:, 0] < actual_shape[2]).all() and (xy[:, 1] < actual_shape[1]).all()):
                    raise RuntimeError(f'Missing/out-of-image FBody in frame {lf.frame_idx}')
        if signature(labels) != original_signature:
            raise RuntimeError('Video relink changed annotations')
        train, val = split_indices([int(lf.frame_idx) for lf in frames], cfg['validation_frames'], cfg['split_seed'])
        outputs, splits = [], {}
        for name, indices in [('train', train), ('val', val)]:
            subset = sio.Labels(labeled_frames=[lf for lf in frames if lf.frame_idx in indices],
                                videos=labels.videos, skeletons=labels.skeletons)
            count = sum(len(lf.user_instances) for lf in subset)
            path = directory / f'{name}.slp'
            expected = signature(subset)
            def validate(check, ix=indices, n=count, sig=expected):
                validate_split(check, video_path, ix, n)
                if signature(check) != sig:
                    raise RuntimeError('Split annotations changed during save')
            save_slp_atomic(subset, path, validate)
            outputs.append({'path': str(path), 'sha256': sha256(path)})
            splits[name] = {'frames': indices, 'n_frames': len(indices), 'n_instances': count}
        resolved_configs, baselines = {}, {}
        for role, baseline in run['training']['base_configs'].items():
            base = project_path(baseline)
            resolved = OmegaConf.load(base)
            baselines[role] = {'path': str(base), 'sha256': sha256(base)}
            dc, tc = resolved.data_config, resolved.trainer_config
            dc.train_labels_path = [str(directory / 'train.slp')]
            dc.val_labels_path = [str(directory / 'val.slp')]
            dc.use_same_data_for_val = False
            dc.validation_fraction = len(val) / len(frames)
            dc.skeletons = [{'nodes': [{'name': n} for n in cfg['nodes']],
                             'edges': [], 'symmetries': [], 'name': None}]
            head = resolved.model_config.head_configs['centroid' if role == 'centroid' else 'centered_instance']
            head.confmaps.anchor_part = cfg['anchor_part']
            if role == 'centered':
                head.confmaps.part_names = cfg['nodes']
            # Parameter counts are measured again by SLEAP, not inherited as evidence.
            resolved.model_config.total_params = None
            tc.max_epochs = run['training']['max_epochs']
            tc.early_stopping.patience = run['training']['early_stopping_patience']
            tc.seed = run['training']['seed']
            tc.ckpt_dir = str(ROOT / 'models' / run['run_id'])
            tc.run_name = role
            tc.resume_ckpt_path = None
            tc.trainer_accelerator = 'gpu'
            tc.model_ckpt.save_last = True
            tc.eval.enabled = False  # explicit separate bounded held-out evaluation
            tc.wandb.name = f"{run['run_id']}_{role}"
            resolved.description = f'Female FBody only: {len(train)} train / {len(val)} validation frames; no regional handling'
            path = ROOT / 'configs' / run['run_id'] / f'{role}.yaml'
            temp = path.with_suffix('.tmp.yaml')
            OmegaConf.save(resolved, temp)
            temp.replace(path)
            outputs.append({'path': str(path), 'sha256': sha256(path)})
            resolved_configs[role] = str(path)
        if sha256(source) != cfg['collaborator_sha256']:
            raise RuntimeError('Uploaded source changed during preparation')
        manifest = {'run_id': run['run_id'], 'dataset_config': cfg,
            'sources': [{'path': str(source), 'sha256': sha256(source)}],
            'original_backup': {'path': str(backup), 'sha256': sha256(backup)},
            'video': {'path': str(video_path), 'shape': actual_shape,
                      'bytes': video_path.stat().st_size, 'mtime_ns': video_path.stat().st_mtime_ns},
            'splits': splits, 'baseline_configs': baselines, 'resolved_configs': resolved_configs,
            'versions': versions(), 'regional_handling': 'none',
            'preparation_code': {name: sha256(ROOT / name) for name in ['scripts/prepare_female.py', 'scripts/prepare_v007.py']},
            'mapping_evidence': 'same clip basename and full shape; every labeled frame decoded; original external path never opened'}
        pin_manifest(directory, manifest)
        atomic_json(directory / 'DATASET_COMPLETE.json', {'manifest_sha256': fingerprint(manifest), 'outputs': outputs})
        dataset_manifest(run)
        print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/female_v002/run.yaml'))
    prepare(parser.parse_args().config)
