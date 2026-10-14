"""Versioned safety/finalization for the existing gputest_predict entry point."""
from __future__ import annotations
import gc
import importlib.metadata
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import yaml
from pipeline_common import ROOT, atomic_json, exclusive_lock, fingerprint, project_path, read_json, sha256, versions
from prediction_finalize import inspect_chunk, merge_raw, derive_tracks

CODE = ['scripts/gputest_predict.py', 'scripts/prediction_run.py', 'scripts/prediction_finalize.py',
        'scripts/body_first_tracklets.py', 'scripts/legacy_dataset/tracklets.py',
        'scripts/pipeline_common.py', 'tests/test_prediction_run.py', 'tests/test_body_first_tracklets.py']


def load_config(path):
    path = project_path(path)
    cfg = read_json(path)
    assert re.fullmatch(r'[A-Za-z0-9_-]+', cfg['run_id'])
    output = project_path(cfg['output_root'])
    assert output == ROOT/'predictions'/cfg['run_id']
    assert cfg['chunk_frames'] > 0 and cfg['batch_size'] == 1
    assert cfg['frames_per_video'] > 0 and 0 < cfg['peak_threshold'] < 1
    assert 0 < cfg['max_no_progress_hops'] <= cfg['max_hops']
    assert cfg['raw_tracking_enabled'] is False and cfg['regional_inference_or_tracking_filtering'] is False
    for role in ('centroid', 'centered'):
        project_path(cfg[role])
    return path, cfg


def inventory(config_path, cfg):
    files = CODE + [str(config_path.relative_to(ROOT)), cfg['tracker_config'], cfg['water_regions']]
    files += [f'{cfg[role]}/{name}' for role in ('centroid', 'centered')
              for name in ('best.ckpt', 'training_config.yaml', 'snapshot.json')]
    hashes = {name: sha256(project_path(name)) for name in files}
    for role in ('centroid', 'centered'):
        assert hashes[f'{cfg[role]}/best.ckpt'] == cfg['checkpoint_sha256'][role]
    videos = {}
    for key, value in cfg['videos'].items():
        assert Path(value).stem == key
        p = project_path(value)
        st = p.stat()
        videos[key] = {'path': value, 'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns}
    return {'config': cfg, 'hashes': hashes, 'videos': videos,
            'versions': dict(versions(), **{p: importlib.metadata.version(p) for p in ('scipy', 'h5py', 'pandas', 'pyarrow')})}


def check_dir(cfg):
    return ROOT/'checks'/cfg['run_id']


def verify(config_path):
    path, cfg = load_config(config_path)
    receipt = read_json(check_dir(cfg)/'PREFLIGHT_COMPLETE.json')
    assert receipt['passed'] is True
    manifest = inventory(path, cfg)
    assert manifest == receipt['manifest'], 'Prediction inputs/code/environment changed: use a new run ID'
    return cfg, fingerprint(manifest)


def selected_video(cfg, video):
    path = project_path(video)
    keys = [key for key, value in cfg['videos'].items() if project_path(value) == path]
    assert len(keys) == 1, 'Video not authorized in run configuration'
    return keys[0], path


def preflight(config_path):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    path, cfg = load_config(config_path)
    directory = check_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(directory):
        if (directory/'PREFLIGHT_COMPLETE.json').exists():
            raise FileExistsError('Preflight already published; no overwrite')
        evidence = directory/f'attempt_{os.environ["SLURM_JOB_ID"]}'
        evidence.mkdir(exist_ok=False)
        before = inventory(path, cfg)
        atomic_json(evidence/'INPUTS.json', before)
        output = project_path(cfg['output_root'])
        if output.exists():
            raise FileExistsError(f'Refusing pre-existing prediction output root: {output}')
        for test in ('tests/test_body_first_tracklets.py', 'tests/test_prediction_run.py'):
            result = subprocess.run([sys.executable, test, '-v'], cwd=ROOT, text=True, capture_output=True)
            (evidence/(Path(test).stem+'.stdout.txt')).write_text(result.stdout)
            (evidence/(Path(test).stem+'.stderr.txt')).write_text(result.stderr)
            print(result.stdout, result.stderr, flush=True)
            if result.returncode:
                raise RuntimeError(f'Tests failed: {test}')
        for script in ('predict_dan_native.sh', 'jobs/preflight_v007_prediction.sh', 'jobs/finalize_v007_prediction.sh'):
            subprocess.run(['bash', '-n', str(ROOT/script)], check=True)
        from sleap_nn.inference.run import predict
        import inspect
        for keyword in ('clean_empty_frames', 'tracker_config', 'filter_config', 'output_format'):
            assert keyword in inspect.signature(predict).parameters
        # Audit checks source identity; only a finite selected old chunk is decoded
        # by the file reader, not new-model inference or an extra video experiment.
        legacy = ROOT/'datasets/legacy_native_tracklets_v001'
        manifest = read_json(legacy/'manifest.json')
        assert sha256(legacy/'manifest.json') == read_json(legacy/'DATASET_COMPLETE.json')['manifest_sha256']
        pins = {p['path']: p for p in manifest['source_pins'] + manifest['input_pins']}
        import cv2
        for key, value in cfg['videos'].items():
            pinned = pins[value]
            actual = before['videos'][key]
            assert (actual['bytes'], actual['mtime_ns']) == (pinned['bytes'], pinned['mtime_ns'])
            capture = cv2.VideoCapture(str(project_path(value)))
            assert capture.isOpened()
            assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == cfg['frames_per_video']
            assert round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == cfg['width']
            assert round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == cfg['height']
            capture.release()
            first = f'predictions/{key}/{key}.frames_000000_000039.slp'
            assert sha256(ROOT/first) == pins[first]['sha256']
            inspect_chunk(ROOT/first, 0, 40, ROOT/value)
        pilot = read_json(ROOT/'checks/body_first_v001/pilot_15525082/PILOT_REPORT.json')
        assert pilot['synthetic_tests_passed'] and pilot['numerical_gate_passed']
        for name in ('scripts/body_first_tracklets.py', cfg['tracker_config']):
            assert before['hashes'][name] == pilot['implementation'][name]
        # Config copies must also match the independently frozen pause receipt.
        pause = project_path(cfg['centroid']).parent
        for item in read_json(pause/'PAUSE_COMPLETE.json')['frozen_files']:
            relative = item['relative_path']
            if relative.startswith(('centroid_best/', 'centered_best/')):
                assert sha256(pause/relative) == item['sha256']
        assert before == inventory(path, cfg), 'Inputs changed during preflight'
        output.mkdir(exist_ok=False)
        atomic_json(output/'RUN_MANIFEST.json', before)
        atomic_json(directory/'PREFLIGHT_COMPLETE.json', {
            'passed': True, 'manifest': before, 'job_id': os.environ['SLURM_JOB_ID'],
            'identity_accuracy': 'experimental_user_authorized_not_ground_truth_validated',
            'new_model_gpu_loading': 'checked_by_each_production_job_first_chunk_not_yet_run'})
        print('PREFLIGHT_PASSED', cfg['run_id'], flush=True)


def read_chunk_receipt(path, signature, start, stop, video):
    receipt_path = path.with_suffix('.json')
    if not receipt_path.exists():
        return None
    receipt = read_json(receipt_path)
    assert receipt['signature'] == signature and receipt['start'] == start and receipt['stop'] == stop
    assert sha256(path) == receipt['sha256'], f'Published chunk changed: {path}'
    inspect_chunk(path, start, stop, video)
    return receipt


def run_hop(config_path, video):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    cfg, signature = verify(config_path)
    key, video = selected_video(cfg, video)
    directory = project_path(cfg['output_root'])/key
    raw = directory/'raw'
    raw.mkdir(parents=True, exist_ok=True)
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with exclusive_lock(directory):
        complete = raw/'CHUNKS_COMPLETE.json'
        if complete.exists():
            assert read_json(complete)['signature'] == signature
            return 0
        state_path = directory/'hop_state.json'
        state = read_json(state_path) if state_path.exists() else {'hops': 0, 'no_progress_hops': 0, 'next_frame': 0, 'signature': signature}
        assert state['signature'] == signature
        if state['hops'] >= cfg['max_hops'] or state['no_progress_hops'] >= cfg['max_no_progress_hops']:
            raise RuntimeError('Finite hop/no-progress bound reached; no automatic retries')
        state['hops'] += 1
        state['no_progress_hops'] += 1  # durable before inference, including hard kills
        atomic_json(state_path, state)
        started = time.monotonic()
        from gputest_predict import predict_chunk
        n = cfg['frames_per_video']
        start = state['next_frame']
        while start < n:
            if stop_requested or time.monotonic()-started >= cfg['hop_work_seconds']:
                return 75
            stop = min(start+cfg['chunk_frames'], n)
            path = raw/f'{key}.frames_{start:06d}_{stop-1:06d}.slp'
            receipt = read_chunk_receipt(path, signature, start, stop, video)
            if receipt is None:
                # An orphan final (crash between rename and receipt) is never
                # trusted/replaced automatically. Temporary files are safe to retry.
                if path.exists():
                    raise RuntimeError(f'Unreceipted final chunk retained for inspection: {path}')
                tmp = path.with_suffix('.tmp.slp')
                if tmp.exists():
                    tmp.unlink()
                print(f'INFERENCE {key} frames={start}:{stop} hop={state["hops"]}', flush=True)
                labels = predict_chunk(video, tmp, start, stop,
                    [project_path(cfg['centroid']), project_path(cfg['centered'])],
                    cfg['batch_size'], cfg['peak_threshold'], cfg['max_instances'])
                del labels
                frames, instances, points = inspect_chunk(tmp, start, stop, video)
                receipt = {'signature': signature, 'start': start, 'stop': stop,
                           'instances': len(instances), 'sha256': sha256(tmp),
                           'job_id': os.environ['SLURM_JOB_ID'], 'tracking_enabled': False}
                del frames, instances, points
                tmp.rename(path)
                atomic_json(path.with_suffix('.json'), receipt)
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            start = stop
            state['next_frame'] = start
            state['no_progress_hops'] = 0
            atomic_json(state_path, state)
            print(f'CHUNK_COMPLETE {key} next_frame={start}/{n}', flush=True)
        chunks = []
        for first in range(0, n, cfg['chunk_frames']):
            end = min(first+cfg['chunk_frames'], n)
            path = raw/f'{key}.frames_{first:06d}_{end-1:06d}.slp'
            receipt = read_json(path.with_suffix('.json'))
            assert receipt['signature'] == signature and (receipt['start'], receipt['stop']) == (first, end)
            chunks.append(dict(receipt, filename=path.name))
        atomic_json(complete, {'signature': signature, 'frames': n, 'chunks': chunks})
        return 0


def finalize(config_path, video):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm required')
    cfg, signature = verify(config_path)
    key, video = selected_video(cfg, video)
    directory = project_path(cfg['output_root'])/key
    raw = directory/'raw'
    with exclusive_lock(directory):
        chunk_receipt = read_json(raw/'CHUNKS_COMPLETE.json')
        assert chunk_receipt['signature'] == signature
        chunks = []
        for item in chunk_receipt['chunks']:
            path = raw/item['filename']
            assert path.parent == raw and sha256(path) == item['sha256']
            chunks.append((path, item['start'], item['stop']))
        raw_final = raw/f'{key}.predictions.slp'
        if (raw/'RAW_COMPLETE.json').exists():
            raw_summary = read_json(raw/'RAW_COMPLETE.json')
            assert raw_summary['signature'] == signature and sha256(raw_final) == raw_summary['sha256']
        else:
            raw_summary = merge_raw(chunks, raw_final, cfg['frames_per_video'], video)
            atomic_json(raw/'RAW_COMPLETE.json', dict(raw_summary, signature=signature, raw_tracking_enabled=False))
        print('RAW_PREDICTIONS_PUBLISHED', raw_final, flush=True)
        tracker_config = yaml.safe_load(project_path(cfg['tracker_config']).read_text())
        # Unique run scope in names; parameters/association code remain pilot-identical.
        tracker_config['dataset_id'] = cfg['run_id']
        rois = read_json(project_path(cfg['water_regions']))['videos']
        derived = derive_tracks(raw_final, directory/'tracklets', key, tracker_config,
                                cfg['width'], cfg['height'], rois[key]['polygon_xy'])
        atomic_json(directory/'PREDICT_COMPLETE.json', {
            'signature': signature, 'raw': raw_summary, 'tracklets': derived,
            'identities_optional': True, 'identity_accuracy': 'not_ground_truth_validated',
            'job_id': os.environ['SLURM_JOB_ID']})
        print('RAW_AND_TRACKLETS_COMPLETE', key, flush=True)
    return 0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight'])
    parser.add_argument('--run-config', required=True)
    arguments = parser.parse_args()
    preflight(arguments.run_config)
