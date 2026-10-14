#!/usr/bin/env python3
"""One fixed, bounded CPU pilot. Produces evidence, never grants production approval."""
from __future__ import annotations

from collections import Counter, defaultdict
import importlib.metadata as md
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import yaml

from pipeline_common import ROOT, atomic_json, read_json, sha256
from body_first_tracklets import Frame, Tracker
from legacy_dataset.tracklets import Tracker as BaselineTracker, water_membership

LEGACY = ROOT / 'datasets/legacy_native_tracklets_v001'
CONFIG_PATH = ROOT / 'configs/tracking/body_first_v001.yaml'


def load_frames(video, start, stop, pins, checked):
    for a in range(start // 40 * 40, stop, 40):
        relative = f'predictions/{video}/{video}.frames_{a:06d}_{a + 39:06d}.slp'
        path = ROOT / relative
        expected = pins[relative]
        if sha256(path) != expected['sha256']:
            raise RuntimeError(f'Source changed: {relative}')
        checked[relative] = expected
        with h5py.File(path, 'r') as f:
            for fr in f['frames'][:]:
                index = int(fr['frame_idx'])
                if not start <= index < stop:
                    continue
                instances = f['instances'][int(fr['instance_id_start']):int(fr['instance_id_end'])]
                n = len(instances)
                assert (instances['instance_type'] == 1).all()
                if n:
                    lo, hi = int(instances['point_id_start'][0]), int(instances['point_id_end'][-1])
                    assert hi - lo == 5 * n
                    np.testing.assert_array_equal(instances['point_id_start'], np.arange(lo, hi, 5))
                    points = f['pred_points'][lo:hi].reshape(n, 5)
                else:
                    points = np.empty((0, 5), dtype=f['pred_points'].dtype)
                xy = np.stack([points['x'], points['y']], axis=-1)
                yield Frame(index, xy[:, 0], xy[:, 1], points['score'][:, 0], points['score'][:, 1],
                            points['visible'][:, 0], points['visible'][:, 1]), relative


def metrics(tracks):
    lengths = np.array([x['length'] for x in tracks], dtype=np.int64)
    observations = int(lengths.sum())
    return {'observations': observations, 'tracklets': len(tracks),
            'singletons': int((lengths == 1).sum()),
            'singleton_observation_fraction': float((lengths == 1).sum() / max(1, observations)),
            'coverage_at_least_30_frames': float(lengths[lengths >= 30].sum() / max(1, observations)),
            'coverage_at_least_150_frames': float(lengths[lengths >= 150].sum() / max(1, observations)),
            'length_quantiles_50_95_99_max': np.quantile(lengths, [.5, .95, .99, 1.]).tolist() if len(lengths) else [],
            'termination_reasons': dict(Counter(x['termination_reason'] for x in tracks))}


def select_reviews(captures, stop, config):
    # Deliberately review newly enabled links, water, and large movement / near
    # competitors. These are diagnostic cases, not a random accuracy sample.
    candidates = defaultdict(list)
    horizon = config['pilot']['consecutive_review_frames']
    for index, capture in captures.items():
        if index + horizon > stop:
            continue
        result, base, frame = capture['new'], capture['baseline'], capture['frame']
        for j, detail in result['link_details'].items():
            entry = (index, int(result['ids'][j]), j)
            if base['predecessor_index'][j] < 0:
                candidates['new_link_body_only_or_previously_vetoed'].append((not detail['shoulder_used'], detail['body_step'], entry))
            if capture['water'][j] is True:
                candidates['water'].append((detail['body_step'], 0, entry))
            margin = detail['margin']
            candidates['motion_or_close_competitor'].append((detail['body_step'] + (0 if margin is None else 8 / max(2, margin)), 0, entry))
    chosen, used = [], set()
    for kind in ('new_link_body_only_or_previously_vetoed', 'water', 'motion_or_close_competitor'):
        for _, _, entry in sorted(candidates[kind], reverse=True):
            if entry[1] not in used:
                chosen.append({'category': kind, 'start': entry[0], 'tracklet_local_id': entry[1], 'source_index': entry[2]})
                used.add(entry[1])
                break
    return chosen[:config['pilot']['max_review_examples_per_clip']]


def render_reviews(video, clip_id, captures, reviews, config, destination):
    import cv2
    from PIL import Image, ImageDraw
    horizon = config['pilot']['consecutive_review_frames']
    cap = cv2.VideoCapture(str(ROOT / 'crab_vids' / f'{video}.mp4'))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot decode {video}')
    results = []
    for number, entry in enumerate(reviews):
        start, identity = entry['start'], entry['tracklet_local_id']
        center = captures[start]['frame'].body[entry['source_index']]
        left = max(0, min(7680 - 256, int(round(center[0])) - 128))
        top = max(0, min(4320 - 256, int(round(center[1])) - 128))
        # 18 consecutive full-resolution 256px crops, displayed at 1.25x.
        tile_width, tile_height, columns = 320, 364, 6
        canvas = Image.new('RGB', (tile_width * columns, tile_height * ((horizon + columns - 1) // columns) + 54), '#141414')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f'{video} | {clip_id} | {entry["category"]} | focal NEW ID {identity}', fill='white')
        draw.text((8, 23), 'Magenta = focal body; cyan = other accepted bodies; yellow = Shoulder. Labels: new ID / old ID. Cross = rejected.', fill='white')
        raw_canvas = Image.new('RGB', canvas.size, '#141414')
        ImageDraw.Draw(raw_canvas).text((8, 8), f'{video} {clip_id}: unannotated fixed crop, target initially at crop center; consecutive frames', fill='white')
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        evidence = []
        for k in range(horizon):
            index = start + k
            ok, image = cap.read()
            if not ok or image.shape[:2] != (4320, 7680):
                raise RuntimeError(f'Decode failed or changed dimensions: {video}:{index}')
            crop = Image.fromarray(cv2.cvtColor(image[top:top + 256, left:left + 256], cv2.COLOR_BGR2RGB)).resize((320, 320))
            tile = Image.new('RGB', (tile_width, tile_height), '#202020')
            tile.paste(crop, (0, 44))
            raw_tile = tile.copy()
            ImageDraw.Draw(raw_tile).text((5, 4), f'frame {index}', fill='white')
            raw_canvas.paste(raw_tile, ((k % columns) * tile_width, 54 + (k // columns) * tile_height))
            dr = ImageDraw.Draw(tile)
            c = captures[index]
            f, new, old = c['frame'], c['new'], c['baseline']
            focal = np.flatnonzero(new['ids'] == identity)
            dr.text((5, 4), f'frame {index} | focal {"present" if len(focal) else "ended / absent"}', fill='white')
            if len(focal):
                j = int(focal[0])
                detail = new['link_details'].get(j)
                dr.text((5, 21), f'old ID {old["ids"][j]} | shoulder {bool(new["validity"]["shoulder_valid"][j])}', fill='white')
                evidence.append({'frame': index, 'source_instance_index': j, 'body_xy': f.body[j].tolist(),
                                 'old_id': int(old['ids'][j]), 'new_id': identity,
                                 'incoming': str(new['incoming_reason'][j]), 'link_details': detail})
            else:
                evidence.append({'frame': index, 'new_id': identity, 'focal_absent': True})
            for j, xy in enumerate(f.body):
                if not np.isfinite(xy).all():
                    continue
                x, y = (xy - [left, top]) * 1.25
                if not (0 <= x < 320 and 0 <= y < 320):
                    continue
                y += 44
                tid = int(new['ids'][j])
                color = 'magenta' if tid == identity else ('cyan' if tid > 0 else 'red')
                dr.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
                dr.text((x + 4, y - 10), f'{tid}/{old["ids"][j]}', fill=color)
                if np.isfinite(f.shoulder[j]).all():
                    sx, sy = (f.shoulder[j] - [left, top]) * 1.25
                    dr.line((x, y, sx, sy + 44), fill='yellow')
            canvas.paste(tile, ((k % columns) * tile_width, 54 + (k // columns) * tile_height))
        name = f'{clip_id}.review_{number:02d}.png'
        canvas.save(destination / name)
        raw_name = name.replace('.png', '.raw.png')
        raw_canvas.save(destination / raw_name)
        results.append(dict(entry, file=name, raw_file=raw_name, raw_sha256=sha256(destination / raw_name), crop_xyxy=[left, top, left + 256, top + 256],
                            frames=evidence, sha256=sha256(destination / name)))
    cap.release()
    return results


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Run only as a finite Slurm job')
    work = ROOT / 'checks/body_first_v001' / f'pilot_{os.environ["SLURM_JOB_ID"]}'
    work.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    source_files = [CONFIG_PATH, ROOT / 'scripts/body_first_tracklets.py', Path(__file__),
                    ROOT / 'tests/test_body_first_tracklets.py', ROOT / 'scripts/legacy_dataset/tracklets.py',
                    LEGACY / 'config/tracklets.yaml', LEGACY / 'config/water_regions_effective_v002.json']
    implementation = {str(p.relative_to(ROOT)): sha256(p) for p in source_files}
    tests = subprocess.run([sys.executable, 'tests/test_body_first_tracklets.py', '-v'], cwd=ROOT,
                           capture_output=True, text=True, check=False)
    (work / 'tests.stdout.txt').write_text(tests.stdout)
    (work / 'tests.stderr.txt').write_text(tests.stderr)
    print(tests.stderr, flush=True)
    if tests.returncode:
        raise RuntimeError('Synthetic tests failed; no real-data pilot or production allowed')
    legacy_manifest = read_json(LEGACY / 'manifest.json')
    completion = read_json(LEGACY / 'DATASET_COMPLETE.json')
    if sha256(LEGACY / 'manifest.json') != completion['manifest_sha256']:
        raise RuntimeError('Published legacy manifest changed')
    pins = {p['path']: p for p in legacy_manifest['source_pins'] + legacy_manifest['input_pins']}
    for relative in ('scripts/legacy_dataset/tracklets.py', 'datasets/legacy_native_tracklets_v001/config/tracklets.yaml'):
        if sha256(ROOT / relative) != pins[relative]['sha256']:
            raise RuntimeError(f'Baseline code/config changed: {relative}')
    baseline_config = yaml.safe_load((LEGACY / 'config/tracklets.yaml').read_text())
    rois = read_json(LEGACY / 'config/water_regions_effective_v002.json')['videos']
    checked, report, aggregate, all_reviews = {}, {}, defaultdict(lambda: defaultdict(list)), []
    for clip in config['pilot']['clips']:
        video, start, stop = clip['video'], clip['start'], clip['stop']
        clip_id = f'{video}_{start:06d}_{stop - 1:06d}'
        video_pin = pins[f'crab_vids/{video}.mp4']
        stat = (ROOT / video_pin['path']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (video_pin['bytes'], video_pin['mtime_ns']):
            raise RuntimeError('Video changed since the published audit')
        checked[video_pin['path']] = dict(video_pin, checked_now='size_and_mtime; SHA256 inherited from published audit')
        trackers = {'baseline': BaselineTracker(clip_id, 7680, 4320, baseline_config),
                    'new': Tracker(clip_id, 7680, 4320, config)}
        tracks, captures, rows, history = defaultdict(list), {}, [], {'baseline': {}, 'new': {}}
        count = 0
        for frame, source in load_frames(video, start, stop, pins, checked):
            assert frame.index == start + count
            count += 1
            results = {name: t.process(frame) for name, t in trackers.items()}
            np.testing.assert_array_equal(results['baseline']['validity']['accepted'], results['new']['validity']['accepted'])
            water = water_membership(frame.body, 7680, 4320, rois[video]['polygon_xy'])
            captures[frame.index] = {'frame': frame, 'baseline': results['baseline'], 'new': results['new'], 'water': water}
            for method, result in results.items():
                tracks[method].extend(result['ended'])
                ids = result['ids']
                assert len(np.unique(ids[ids > 0])) == (ids > 0).sum()
                for j in np.flatnonzero(ids > 0):
                    identity = int(ids[j])
                    if identity in history[method]:
                        assert history[method][identity] + 1 == frame.index
                    history[method][identity] = frame.index
                    rows.append({'method': method, 'frame': frame.index, 'instance_index': int(j),
                                 'tracklet_id': trackers[method].name(identity), 'body_x': frame.body[j, 0],
                                 'body_y': frame.body[j, 1], 'water': water[j],
                                 'incoming': str(result['incoming_reason'][j])})
        assert count == stop - start
        for method, tracker in trackers.items():
            tracks[method].extend(tracker.finish())
            assert sum(t['length'] for t in tracks[method]) == sum(r['method'] == method for r in rows)
            aggregate[video][method].extend(tracks[method])
        table = pd.DataFrame(rows)
        target = work / f'{clip_id}.observations.parquet'
        table.to_parquet(target, index=False)
        pd.testing.assert_frame_equal(table, pd.read_parquet(target))
        selected = select_reviews(captures, stop, config)
        reviews = render_reviews(video, clip_id, captures, selected, config, work)
        all_reviews.extend([dict(r, video=video, clip_id=clip_id) for r in reviews])
        report[clip_id] = {'selection': clip, 'metrics': {method: metrics(ts) for method, ts in tracks.items()},
                           'review_files': [r['file'] for r in reviews], 'observations_sha256': sha256(target)}
        print(json.dumps({clip_id: report[clip_id]['metrics']}), flush=True)
    totals = {video: {method: metrics(ts) for method, ts in methods.items()} for video, methods in aggregate.items()}
    gates = {}
    for video, methods in totals.items():
        old, new = methods['baseline'], methods['new']
        gates[video] = {
            'same_accepted_observations': old['observations'] == new['observations'],
            'singleton_reduction_pass': new['singleton_observation_fraction'] <= old['singleton_observation_fraction'] * (1 - config['pilot']['minimum_singleton_observation_reduction_fraction']),
            '30_frame_coverage_gain_pass': new['coverage_at_least_30_frames'] - old['coverage_at_least_30_frames'] >= config['pilot']['minimum_30_frame_observation_coverage_gain']}
    if implementation != {str(p.relative_to(ROOT)): sha256(p) for p in source_files}:
        raise RuntimeError('Implementation/configuration changed during pilot')
    passed = all(all(checks.values()) for checks in gates.values())
    atomic_json(work / 'review_examples.json', all_reviews)
    atomic_json(work / 'PILOT_REPORT.json', {
        'job_id': os.environ['SLURM_JOB_ID'], 'config': config, 'implementation': implementation,
        'synthetic_tests_passed': True, 'numerical_gate_passed': passed,
        'production_approved': False, 'visual_review_required': True,
        'clips': report, 'aggregate': totals, 'numerical_checks': gates,
        'checked_sources': checked, 'versions': {p: md.version(p) for p in ('numpy', 'scipy', 'pandas', 'pyarrow', 'h5py')},
        'limitations': ['Selected short clips; no exhaustive identity ground truth.',
                        'Clips truncate tracks equally for both methods.',
                        'No inference or changes to existing data. Numerical gains alone do not approve production.']})
    print(f'PILOT_FINISHED numerical_gate_passed={passed}; visual review still required. {work}', flush=True)


if __name__ == '__main__':
    main()
