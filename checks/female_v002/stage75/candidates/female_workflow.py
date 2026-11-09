#!/usr/bin/env python3
"""Finite female preflight receipts, held-out evaluation, chunked inference/merge.

Training itself uses the unchanged latest male train_versioned.py worker.
Only explicit exit 75 requests same-ID Slurm requeue during video inference.
"""
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path
import shutil
import signal
import sys
import time

import numpy as np
import sleap_io as sio
from pipeline_common import (ROOT, atomic_json, chunk_ranges, dataset_manifest,
    exclusive_lock, fingerprint, hop_progress, hop_start, load_run, model_dir,
    pin_manifest, project_path, read_json, sha256, versions)
from female_prediction_io import inspect_chunk, merge_raw

CONFIG = ROOT / 'configs/female_v002/run.yaml'
CODE = ['scripts/female_workflow.py', 'scripts/female_prediction_io.py',
        'scripts/prepare_female.py', 'scripts/female_train_chain.sh',
        'tests/test_female_workflow.py', 'jobs/preflight_female.sh',
        'jobs/train_female_centroid.sh', 'jobs/train_female_centered.sh',
        'jobs/validate_female.sh', 'jobs/predict_female.sh', 'jobs/finalize_female.sh']


def code_digest(name):
    # Submission dependencies are scheduling metadata filled after predecessor IDs
    # are known. Every other byte (including resources and executable code) is pinned.
    if name.startswith('jobs/'):
        import hashlib
        text = ''.join(line for line in (ROOT / name).read_text().splitlines(keepends=True)
                       if not line.startswith('#SBATCH --dependency='))
        return hashlib.sha256(text.encode()).hexdigest()
    return sha256(ROOT / name)


def input_signature(run):
    from train_versioned import implementation_hashes
    return {'run_config_sha256': sha256(CONFIG),
            'dataset_sha256': fingerprint(dataset_manifest(run)),
            'implementation': implementation_hashes(),
            'female_implementation': {p: code_digest(p) for p in CODE},
            'versions': versions()}


def preflight_receipt():
    from train_versioned import checkpoint_summary
    run = load_run(CONFIG)
    receipt = input_signature(run)
    for role in ('centroid', 'centered'):
        directory = ROOT / 'checks' / run['run_id'] / 'training_smoke' / role
        marker = read_json(directory / 'TRAINING_COMPLETE.json')
        assert marker['global_step'] >= 4 and marker['epoch'] == 2, marker
        assert sha256(directory / 'best.ckpt') == marker['best_sha256']
        checkpoint_summary(directory / 'best.ckpt')
    # Exercise the actual two-stage inference API/one-node serialization before
    # launching long training. This is a single-frame software smoke check, not
    # an accuracy experiment or model selection on additional data.
    from gputest_predict import predict_chunk
    ds = dataset_manifest(run)['manifest']
    frame = ds['splits']['val']['frames'][0]
    smoke = ROOT / 'checks' / run['run_id'] / 'training_smoke'
    path = smoke / 'api_smoke.slp'
    if path.exists():
        raise FileExistsError(path)
    labels = predict_chunk(project_path(ds['video']['path']), path, frame, frame+1,
                           [smoke / 'centroid', smoke / 'centered'])
    del labels
    inspect_chunk(path, frame, frame+1, project_path(ds['video']['path']), run['dataset']['nodes'])
    receipt.update({'passed': True, 'job_id': os.environ['SLURM_JOB_ID'],
        'checks': ['female unit tests', 'existing training regression tests',
                   'GPU forward/backward both FBody models', 'signal/save/resume both roles',
                   'held-out validation in smoke training', 'best checkpoint readback',
                   'one-frame two-stage inference/SLP API smoke']})
    path = ROOT / 'checks' / run['run_id'] / 'PREFLIGHT_COMPLETE.json'
    if path.exists():
        raise FileExistsError(path)
    atomic_json(path, receipt)
    print('FEMALE_PREFLIGHT_COMPLETE', flush=True)


def verify_preflight(run):
    receipt = read_json(ROOT / 'checks' / run['run_id'] / 'PREFLIGHT_COMPLETE.json')
    assert receipt['passed'] is True
    actual = input_signature(run)
    if not all(receipt[k] == v for k, v in actual.items()):
        from female_stage75 import verify_extension
        verify_extension(receipt, actual)
    return actual


def check_video_pin(info):
    path = project_path(info['path'])
    st = path.stat()
    assert (st.st_size, st.st_mtime_ns) == (info['bytes'], info['mtime_ns']), 'Source video changed'
    return path


def freeze_models(run):
    """Select completed training's validation-best pair, never latest/old male weights."""
    from train_versioned import checkpoint_summary
    directory = ROOT / 'models' / run['run_id'] / 'selected'
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for role in ('centroid', 'centered'):
        source = model_dir(run, role)
        completion_path = source / 'TRAINING_COMPLETE.json'
        marker = read_json(completion_path)
        manifest = read_json(source / 'manifest.json')
        assert marker['manifest_sha256'] == fingerprint(manifest)
        assert manifest['dataset_sha256'] == fingerprint(dataset_manifest(run))
        assert manifest['implementation'] == input_signature(run)['implementation']
        assert sha256(source / 'best.ckpt') == marker['best_sha256']
        checkpoint_summary(source / 'best.ckpt')
        if role == 'centroid':
            from female_stage75 import read_stage
            source, stage = read_stage(source)
            assert marker['reason'] == 'user_requested_75_completed_epoch_stage'
            assert marker['stage_checkpoint'] == str(source.relative_to(ROOT))
            assert marker['stage_marker_sha256'] == sha256(source / 'STAGE_COMPLETE.json')
        dest = directory / role
        dest.mkdir(exist_ok=True)
        files = {}
        for name in ('best.ckpt', 'training_config.yaml'):
            a, b = source / name, dest / name
            digest = sha256(a)
            if not b.exists():
                tmp = b.with_suffix(b.suffix + '.tmp')
                shutil.copy2(a, tmp)
                assert sha256(tmp) == digest
                tmp.replace(b)
            assert sha256(b) == digest, 'Frozen model changed'
            files[name] = digest
        result[role] = {'directory': str(dest.relative_to(ROOT)), 'files': files,
                        'training_completion_sha256': sha256(completion_path),
                        'source_model_directory': str(source.relative_to(ROOT)),
                        'selection_kind': 'exact_epoch75_weights' if role == 'centroid' else 'validation_best_weights',
                        'best_progress': checkpoint_summary(dest / 'best.ckpt')}
    pin_manifest(directory, result)
    return result


def matched_distances(truth, predictions, radius):
    """Maximum-cardinality one-to-one radius matching, then minimize pixel error."""
    from scipy.optimize import linear_sum_assignment
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    predictions = np.asarray(predictions, dtype=float).reshape(-1, 2)
    assert np.isfinite(truth).all() and np.isfinite(predictions).all() and radius > 0
    m, n = len(truth), len(predictions)
    if not m or not n:
        return np.empty(0, float)
    distance = np.linalg.norm(truth[:, None, :] - predictions[None, :, :], axis=-1)
    penalty = (max(m, n)+1)*(radius+1)
    cost = np.full((m, n+m), penalty)
    cost[:, :n] = np.where(distance <= radius, distance, 2*penalty)
    rows, cols = linear_sum_assignment(cost)
    keep = cols < n
    result = distance[rows[keep], cols[keep]]
    assert (result <= radius).all()
    return result


def metrics(ntruth, npred, distances):
    tp = len(distances)
    precision = tp / npred if npred else None
    recall = tp / ntruth if ntruth else None
    return {'true_positives': tp, 'false_positives_relative_to_labels': npred-tp,
            'false_negatives': ntruth-tp, 'precision': precision, 'recall': recall,
            'f1': 2*tp/(ntruth+npred) if ntruth+npred else None,
            'matched_error_px': None if not tp else {
                'mean': float(np.mean(distances)), 'median': float(np.median(distances)),
                'p90': float(np.quantile(distances, .9)), 'max': float(np.max(distances))}}


def evaluate():
    run = load_run(CONFIG)
    signature = verify_preflight(run)
    ds = dataset_manifest(run)['manifest']
    video = check_video_pin(ds['video'])
    directory = ROOT / 'checks' / run['run_id'] / 'validation'
    with exclusive_lock(directory):
        if (directory / 'VALIDATION_COMPLETE.json').exists():
            raise FileExistsError('Validation already published; no automatic rerun')
        selected = freeze_models(run)
        model_paths = [project_path(selected[r]['directory']) for r in ('centroid', 'centered')]
        indices = ds['splits']['val']['frames']
        target = directory / 'heldout.predictions.slp'
        temp = target.with_suffix('.tmp.slp')
        if target.exists() or temp.exists():
            raise FileExistsError('Prior validation output retained for review')
        from sleap_nn.inference.run import predict
        labels = predict(str(video), model_paths=[str(p) for p in model_paths], frames=indices,
            output_path=str(temp), output_format='slp', device='cuda', batch_size=1,
            peak_threshold=run['prediction']['peak_threshold'], max_instances=None,
            tracker_config=None, filter_config=None, clean_empty_frames=False, embed='false')
        del labels
        labels = sio.load_slp(temp, open_videos=False)
        assert [int(lf.frame_idx) for lf in labels] == indices
        assert len(labels.skeletons) == 1 and labels.skeletons[0].node_names == run['dataset']['nodes']
        assert Path(labels.videos[0].filename).resolve() == video
        assert not labels.tracks and not any(lf.user_instances for lf in labels)
        truth = sio.load_slp(ROOT / 'data/labels' / run['run_id'] / 'val.slp', open_videos=False)
        truths = {int(lf.frame_idx): np.array([i.numpy()[0] for i in lf.user_instances]).reshape(-1, 2) for lf in truth}
        per_frame, all_distances = [], {r: [] for r in run['validation']['matching_radii_px']}
        nt = npred = 0
        for lf in labels:
            xy = np.array([i.numpy()[0] for i in lf.instances]).reshape(-1, 2)
            # Missing FBody instances cannot match, but remain in prediction counts.
            finite = xy[np.isfinite(xy).all(axis=1)]
            gt = truths[int(lf.frame_idx)]
            frame = {'frame': int(lf.frame_idx), 'human_instances': len(gt),
                     'predicted_instances': len(xy), 'finite_FBody_predictions': len(finite), 'metrics': {}}
            for radius in all_distances:
                d = matched_distances(gt, finite, radius)
                frame['metrics'][str(radius)] = metrics(len(gt), len(xy), d)
                all_distances[radius].extend(d.tolist())
            nt, npred = nt+len(gt), npred+len(xy)
            per_frame.append(frame)
        temp.rename(target)
        report = {'run_id': run['run_id'], 'frames': indices, 'n_frames': len(indices),
            'human_instances': nt, 'predicted_instances': npred, 'per_frame': per_frame,
            'metrics_by_matching_radius_px': {str(r): metrics(nt, npred, d) for r, d in all_distances.items()},
            'primary_matching_radius_px': run['validation']['primary_matching_radius_px'],
            'limitations': ['Only two held-out frames; same clip as training, possible temporal correlation.',
                'Validation frames also select best checkpoints/early stopping; not an independent final test set.',
                'Precision is relative to provided labels; unlabeled animals can be counted as false positives.',
                'FBody localization/detection only; no limb pose or identity accuracy measured.'],
            'selected_models': selected, 'input_signature': signature,
            'prediction_sha256': sha256(target), 'job_id': os.environ['SLURM_JOB_ID']}
        assert signature == verify_preflight(run)
        atomic_json(directory / 'VALIDATION_COMPLETE.json', report)
        (directory / 'report.md').write_text('# Female held-out validation\n\n' +
            f'Frames: {indices}; {nt} human instances; {npred} predictions.\n\n' +
            '| Radius (px) | Precision | Recall | F1 | Median error (px) |\n|---|---|---|---|---|\n' +
            ''.join(f"| {r} | {m['precision']} | {m['recall']} | {m['f1']} | {None if m['matched_error_px'] is None else m['matched_error_px']['median']} |\n"
                    for r, m in report['metrics_by_matching_radius_px'].items()) +
            '\n' + '\n'.join('- '+s for s in report['limitations']) + '\n')
        print('FEMALE_VALIDATION_COMPLETE', report['metrics_by_matching_radius_px'], flush=True)


def prediction_inputs(run):
    signature = verify_preflight(run)
    report_path = ROOT / 'checks' / run['run_id'] / 'validation/VALIDATION_COMPLETE.json'
    report = read_json(report_path)
    assert report['input_signature'] == signature
    selected = report['selected_models']
    for item in selected.values():
        for name, digest in item['files'].items():
            assert sha256(project_path(item['directory']) / name) == digest, 'Selected model changed'
    videos = {}
    for key, value in run['videos'].items():
        path = project_path(value)
        v = sio.load_video(str(path))
        videos[key] = {'path': str(path), 'shape': list(v.shape),
            'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        assert list(v.shape)[1:3] == [4320, 7680]
    return {'signature': signature, 'selected_models': selected, 'videos': videos,
            'validation_sha256': sha256(report_path)}


def chunk_receipt(path, signature, start, stop, video, nodes):
    receipt_path = path.with_suffix('.json')
    if not receipt_path.exists():
        return None
    receipt = read_json(receipt_path)
    assert receipt['signature'] == signature and (receipt['start'], receipt['stop']) == (start, stop)
    assert sha256(path) == receipt['sha256']
    inspect_chunk(path, start, stop, video, nodes)
    return receipt


def predict_video(key):
    run = load_run(CONFIG)
    assert key in run['videos']
    inputs = prediction_inputs(run)
    signature = fingerprint(inputs)
    info = inputs['videos'][key]
    video = check_video_pin(info)
    n = info['shape'][0]
    policy, nodes = run['prediction'], run['dataset']['nodes']
    directory = ROOT / 'predictions' / run['run_id'] / key
    raw = directory / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    stop_requested = False
    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with exclusive_lock(directory):
        pin_manifest(directory, inputs)
        marker = raw / 'CHUNKS_COMPLETE.json'
        if marker.exists():
            assert read_json(marker)['signature'] == signature
            return 0
        state_path = directory / 'inference_progress.json'
        start = read_json(state_path)['next_frame'] if state_path.exists() else 0
        if state_path.exists():
            assert read_json(state_path)['signature'] == signature
        hop_start(directory, policy['max_hops'])
        state = read_json(directory / 'hop_state.json')
        if state['no_progress_hops'] > policy['max_no_progress_hops']:
            raise RuntimeError('No-progress cap reached')
        started = time.monotonic()
        from gputest_predict import predict_chunk
        while start < n:
            if stop_requested or time.monotonic()-started >= policy['hop_work_seconds']:
                hop_progress(directory, start, policy['max_no_progress_hops'])
                return 75
            stop = min(start+policy['chunk_frames'], n)
            target = raw / f'{key}.frames_{start:06d}_{stop-1:06d}.slp'
            receipt = chunk_receipt(target, signature, start, stop, video, nodes)
            if receipt is None:
                if target.exists():
                    raise RuntimeError(f'Unreceipted final chunk retained: {target}')
                temp = target.with_suffix('.tmp.slp')
                temp.unlink(missing_ok=True)
                models = [project_path(inputs['selected_models'][r]['directory']) for r in ('centroid', 'centered')]
                print(f'FEMALE_INFERENCE {key} frames={start}:{stop}', flush=True)
                labels = predict_chunk(video, temp, start, stop, models,
                    policy['batch_size'], policy['peak_threshold'], policy['max_instances'])
                del labels
                f, i, p = inspect_chunk(temp, start, stop, video, nodes)
                receipt = {'signature': signature, 'start': start, 'stop': stop,
                    'instances': len(i), 'sha256': sha256(temp), 'job_id': os.environ['SLURM_JOB_ID']}
                del f, i, p
                temp.rename(target)
                atomic_json(target.with_suffix('.json'), receipt)
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            start = stop
            atomic_json(state_path, {'signature': signature, 'next_frame': start})
            hop_progress(directory, start, policy['max_no_progress_hops'])
            print(f'FEMALE_CHUNK_COMPLETE {key} next_frame={start}/{n}', flush=True)
        receipts = []
        for lo, hi in chunk_ranges(n, policy['chunk_frames']):
            path = raw / f'{key}.frames_{lo:06d}_{hi-1:06d}.slp'
            receipt = read_json(path.with_suffix('.json'))
            assert receipt['signature'] == signature and (receipt['start'], receipt['stop']) == (lo, hi)
            receipts.append(dict(receipt, filename=path.name))
        atomic_json(marker, {'signature': signature, 'frames': n, 'chunks': receipts})
        return 0


def finalize(key):
    run = load_run(CONFIG)
    inputs = prediction_inputs(run)
    signature = fingerprint(inputs)
    directory = ROOT / 'predictions' / run['run_id'] / key
    raw = directory / 'raw'
    with exclusive_lock(directory):
        assert read_json(directory / 'manifest.json') == inputs
        marker = read_json(raw / 'CHUNKS_COMPLETE.json')
        assert marker['signature'] == signature
        info = inputs['videos'][key]
        video = check_video_pin(info)
        ranges = chunk_ranges(info['shape'][0], run['prediction']['chunk_frames'])
        assert [(i['start'], i['stop']) for i in marker['chunks']] == ranges
        chunks = []
        for item in marker['chunks']:
            path = raw / item['filename']
            assert path.parent == raw and path.name == f"{key}.frames_{item['start']:06d}_{item['stop']-1:06d}.slp"
            assert sha256(path) == item['sha256']
            chunks.append((path, item['start'], item['stop']))
        target = raw / f'{key}.female.predictions.slp'
        summary = merge_raw(chunks, target, info['shape'][0], video, run['dataset']['nodes'])
        assert inputs == prediction_inputs(run)
        atomic_json(raw / 'RAW_COMPLETE.json', dict(summary, signature=signature))
        atomic_json(directory / 'PREDICT_COMPLETE.json', {'signature': signature,
            'raw': summary, 'tracking_enabled': False, 'job_id': os.environ['SLURM_JOB_ID']})
        print('FEMALE_PREDICTION_COMPLETE', target, summary, flush=True)


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['preflight-receipt', 'verify-training', 'evaluate', 'predict', 'finalize'])
    parser.add_argument('--video-key', choices=['Crab_Cam_2_2', 'Crab_Cam_2_3'])
    args = parser.parse_args()
    if args.action == 'preflight-receipt':
        preflight_receipt()
    elif args.action == 'verify-training':
        verify_preflight(load_run(CONFIG))
    elif args.action == 'evaluate':
        evaluate()
    else:
        if args.video_key is None:
            parser.error('--video-key required')
        if args.action == 'predict':
            return predict_video(args.video_key)
        finalize(args.video_key)
    return 0


if __name__ == '__main__':
    sys.exit(main())
