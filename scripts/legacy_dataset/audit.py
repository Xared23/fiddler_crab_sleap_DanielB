#!/usr/bin/env python3
"""Finite, read-only original-prediction audit. No inference and no video relinking.

All execution must be in the dedicated Slurm allocation. Outputs are published
atomically into a new audit directory; existing successful output is never replaced.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import h5py
import numpy as np
import sleap_io as sio
from sleap_io.io.slp import read_skeletons, read_metadata, read_provenance

ROOT = Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap')
DEST = ROOT / 'datasets/legacy_native_tracklets_v001'
VIDEOS = ['Crab_Cam_2_2', 'Crab_Cam_2_3']
NODES = ['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm']


def native(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, bytes):
        return x.decode()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=native, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    return {'path': str(path.relative_to(ROOT)), 'bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': h.hexdigest()}


def quantiles(values):
    a = np.asarray(values, dtype=float).ravel()
    good = a[np.isfinite(a)]
    return {'count': len(a), 'finite': len(good),
            'quantiles_0_01_05_50_95_99_100': np.quantile(good, [0, .01, .05, .5, .95, .99, 1]).tolist() if len(good) else None}


def json_dataset(f, name):
    return [json.loads(x) for x in f[name][:]] if name in f else []


def frame_payload(f, frame):
    """Read a frame, validating indexing and retaining the serialized raw fields."""
    a, b = int(frame['instance_id_start']), int(frame['instance_id_end'])
    assert 0 <= a <= b <= len(f['instances'])
    inst = f['instances'][a:b]
    assert np.all(inst['instance_id'] == np.arange(a, b))
    assert np.all(inst['frame_id'] == frame['frame_id'])
    assert np.all(inst['instance_type'] == 1), 'Non-predicted instances require separate handling'
    assert np.all(inst['skeleton'] == 0)
    if b > a:
        starts, ends = inst['point_id_start'], inst['point_id_end']
        assert np.all(ends - starts == len(NODES))
        assert np.all(starts[1:] == ends[:-1])
        p0, p1 = int(starts[0]), int(ends[-1])
        assert 0 <= p0 <= p1 <= len(f['pred_points'])
        points = f['pred_points'][p0:p1].reshape(-1, len(NODES))
    else:
        points = np.empty((0, len(NODES)), dtype=f['pred_points'].dtype)
    # Hash canonical field bytes, independent of file-global instance/point IDs.
    h = hashlib.sha256()
    for name in ['x', 'y', 'score', 'visible', 'complete']:
        h.update(np.ascontiguousarray(points[name]).tobytes())
    for name in ['score', 'track', 'tracking_score', 'skeleton', 'instance_type']:
        if name in inst.dtype.names:
            h.update(np.ascontiguousarray(inst[name]).tobytes())
    return inst, points, h.hexdigest()


def inspect_file(path, video, width, height, accumulate=None):
    record = sha(path)
    issues = []
    record['issues'] = issues
    try:
        with h5py.File(path, 'r') as f:
            record['hdf5_datasets'] = {}
            def visit(name, item):
                if isinstance(item, h5py.Dataset):
                    record['hdf5_datasets'][name] = {'shape': list(item.shape), 'dtype': str(item.dtype)}
            f.visititems(visit)
            record['root_attrs'] = dict(f.attrs)
            videos = json_dataset(f, 'videos_json')
            record['videos_json'] = videos
            record['tracks_json'] = json_dataset(f, 'tracks_json')
            record['metadata'] = read_metadata(str(path))
            record['provenance'] = read_provenance(str(path), record['metadata'])
            record['skeleton_nodes'] = [s.node_names for s in read_skeletons(str(path))]
            assert record['skeleton_nodes'] == [NODES], 'Unexpected skeleton'
            assert len(videos) == 1, 'Expected one video'
            backend = videos[0].get('backend', videos[0])
            reference = backend.get('filename', videos[0].get('filename'))
            record['reference_matches_local_absolute_path'] = reference == str(video)
            if reference != str(video):
                issues.append('Video reference not exact local raw-video path; not opened or relinked')
            frames = f['frames'][:]
            indices = frames['frame_idx'].astype(int)
            record['frame_records'] = len(frames)
            record['frame_indices'] = indices.tolist()
            record['strictly_increasing_frames'] = bool(np.all(np.diff(indices) > 0))
            record['instance_count'] = len(f['instances'])
            record['detector_centroid_datasets'] = [k for k in record['hdf5_datasets'] if 'centroid' in k.lower()]
            record['crop_transform_datasets'] = [k for k in record['hdf5_datasets'] if any(s in k.lower() for s in ['crop', 'bbox', 'transform'])]
            assert np.all(frames['video'] == 0), 'Invalid video indices'
            assert np.all(frames['frame_id'] == np.arange(len(frames))), 'Frame IDs not sequential'
            assert record['strictly_increasing_frames'], 'Duplicate or unordered frames'
            if len(frames):
                assert frames[0]['instance_id_start'] == 0
                assert frames[-1]['instance_id_end'] == len(f['instances'])
                assert np.all(frames['instance_id_start'][1:] == frames['instance_id_end'][:-1])
            digests, counts, track_ids = {}, {}, collections.Counter()
            for frame in frames:
                idx = int(frame['frame_idx'])
                inst, pts, digest = frame_payload(f, frame)
                digests[idx] = digest
                counts[idx] = len(inst)
                track_ids.update(map(int, inst['track']))
                if accumulate is not None:
                    accumulate(idx, inst, pts, path.name)
            record['instance_track_counts'] = dict(track_ids)
            record['empty_frame_records'] = [i for i, n in counts.items() if n == 0]
            if '.frames_' in path.name:
                labels = sio.load_slp(str(path), open_videos=False)
                assert [lf.frame_idx for lf in labels] == indices.tolist()
                assert sum(len(lf.instances) for lf in labels) == len(f['instances'])
                record['sleap_io_readback_no_video_open'] = True
            record['readable_and_structurally_valid'] = True
            return record, digests, counts
    except Exception as exc:
        issues.append(f'{type(exc).__name__}: {exc}')
        record['readable_and_structurally_valid'] = False
        return record, {}, {}


def video_metadata(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    info = {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'container_frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'container_fps': float(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count_method': 'OpenCV container metadata plus bounded final-frame decode, not exhaustive decode',
            'time_s_policy': 'frame / container_fps nominal conversion only; no per-frame timestamp verification'}
    cap.release()
    # Optional ffprobe metadata only (never a scan/decode of all frames).
    try:
        proc = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                               '-show_entries', 'stream=width,height,nb_frames,r_frame_rate,avg_frame_rate,time_base,duration,start_time',
                               '-of', 'json', str(path)], capture_output=True, text=True, timeout=30, check=True)
        info['ffprobe'] = json.loads(proc.stdout)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        info['ffprobe_unavailable'] = str(exc)
    return info


class Measurements:
    def __init__(self, nframes, width, height):
        self.nframes, self.width, self.height = nframes, width, height
        self.counts = collections.Counter()
        self.parts = collections.defaultdict(list)
        self.frames = {}
        self.prev = None
        self.max_count = (-1, 0)
        self.closest = (float('inf'), 0, None)
        self.duplicates = []

    def __call__(self, idx, inst, pts, source):
        self.counts['instances'] += len(inst)
        xy = np.stack([pts['x'], pts['y']], axis=-1)
        finite = np.isfinite(xy).all(-1)
        inside = finite & (xy[..., 0] >= 0) & (xy[..., 0] < self.width) & (xy[..., 1] >= 0) & (xy[..., 1] < self.height)
        for j, name in enumerate(NODES):
            self.counts[f'{name}_missing_xy'] += int((~finite[:, j]).sum())
            self.counts[f'{name}_out_of_bounds_xy'] += int((finite[:, j] & ~inside[:, j]).sum())
            self.counts[f'{name}_score_below_0.2_or_missing'] += int((~np.isfinite(pts['score'][:, j]) | (pts['score'][:, j] < .2)).sum())
            self.parts[f'{name}_score'].append(pts['score'][:, j].copy())
        self.parts['instance_score'].append(inst['score'].copy())
        if 'tracking_score' in inst.dtype.names:
            self.parts['tracking_score'].append(inst['tracking_score'].copy())
        if len(inst):
            self.parts['shell_shoulder_distance_px'].append(np.linalg.norm(xy[:, 0] - xy[:, 1], axis=1))
        good = inside[:, 0] & inside[:, 1] & np.isfinite(pts['score'][:, :2]).all(1) & (pts['score'][:, :2] >= .2).all(1)
        self.counts['trusted_both_finite_inbounds_score_ge_0.2'] += int(good.sum())
        shell = xy[inside[:, 0], 0]
        dmin = None
        center = None
        if len(shell) > 1:
            d = np.linalg.norm(shell[:, None] - shell[None, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(d.argmin(), d.shape)
            dmin = float(d[i, j])
            center = ((shell[i] + shell[j]) / 2).tolist()
            npairs = int(np.count_nonzero(d < 5) // 2)
            self.counts['shell_pairs_under_5px'] += npairs
            if npairs and len(self.duplicates) < 30:
                self.duplicates.append({'frame': idx, 'pairs_under_5px': npairs, 'closest_pair_px': dmin})
            if dmin < self.closest[0] and idx < self.nframes - 1:
                self.closest = (dmin, idx, center)
        if len(inst) > self.max_count[0] and idx < self.nframes - 1:
            self.max_count = (len(inst), idx)
        if self.prev is not None and self.prev[0] == idx - 1 and len(shell) and len(self.prev[1]):
            d = np.linalg.norm(self.prev[1][:, None] - shell[None, :], axis=-1)
            self.parts['adjacent_nearest_shell_distance_px_diagnostic_not_identity'].append(d.min(1))
        self.prev = (idx, shell)
        self.frames[idx] = {'frame': idx, 'source': source, 'count': len(inst),
                            'trusted_count': int(good.sum()), 'closest_shell_pair_px': dmin,
                            'closest_pair_center': center}

    def summary(self):
        return {'counts': dict(self.counts),
                'distributions': {k: quantiles(np.concatenate(v)) for k, v in self.parts.items() if v},
                'duplicate_candidate_examples': self.duplicates,
                'densest_frame': self.max_count[1],
                'closest_pair_frame': self.closest[1], 'closest_pair_center': self.closest[2]}


def previews(video, info, measurement, work):
    """At most 12 distinct frames/video, with a bounded close-pair crop."""
    import cv2
    from PIL import Image, ImageDraw
    n = info['container_frame_count']
    selected = sorted(set([0, 1, 39, 40, n // 2, n // 2 + 1, n - 2, n - 1,
                           measurement.max_count[1], measurement.max_count[1] + 1,
                           measurement.closest[1], measurement.closest[1] + 1]))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_N_THREADS, 1) if hasattr(cv2, 'CAP_PROP_N_THREADS') else None
    evidence = []
    tiles = []
    for idx in selected:
        if not 0 <= idx < n:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            evidence.append({'frame': idx, 'decoded': False})
            continue
        assert bgr.shape[:2] == (info['height'], info['width'])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        points = np.empty((0, 5, 2))
        entry = measurement.frames.get(idx)
        if entry:
            path = ROOT / 'predictions' / video.stem / entry['source']
            with h5py.File(path, 'r') as f:
                fr = f['frames'][:]
                _, p, _ = frame_payload(f, fr[fr['frame_idx'] == idx][0])
                points = np.stack([p['x'], p['y']], axis=-1)
        w = 1600
        h = round(image.height * w / image.width)
        preview = image.resize((w, h))
        canvas = Image.new('RGB', (w, h + 64), '#202020')
        canvas.paste(preview, (0, 64))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f'{video.stem} frame {idx} | {image.width} x {image.height} original pixels', fill='white')
        draw.text((10, 27), 'Shell=cyan Shoulder=yellow. Water ROI UNCONFIRMED; no water/land classification.', fill='white')
        s = w / image.width
        for k, nodes in enumerate(points):
            for j, color in [(0, 'cyan'), (1, 'yellow')]:
                if np.isfinite(nodes[j]).all():
                    x, y = nodes[j] * s
                    draw.ellipse((x - 2, y + 62, x + 2, y + 66), outline=color, width=1)
            if np.isfinite(nodes[:2]).all():
                a, b = nodes[0] * s, nodes[1] * s
                draw.line((a[0], a[1] + 64, b[0], b[1] + 64), fill='cyan')
        for x in range(0, image.width, 1000):
            draw.text((int(x * s) + 2, 48), f'x={x}', fill='white')
        for y in range(0, image.height, 1000):
            draw.text((2, int(y * s) + 66), f'y={y}', fill='white')
        name = f'{video.stem}.frame_{idx:06d}.png'
        canvas.save(work / name)
        tiles.append(canvas.resize((800, round(canvas.height / 2))))
        entry_out = {'frame': idx, 'decoded': True, 'opencv_pos_frames_after_read': cap.get(cv2.CAP_PROP_POS_FRAMES),
                     'opencv_pos_msec_after_read': cap.get(cv2.CAP_PROP_POS_MSEC), 'preview': name}
        if idx in (measurement.closest[1], measurement.closest[1] + 1) and measurement.closest[2] is not None:
            cx, cy = measurement.closest[2]
            left = max(0, min(image.width - 640, round(cx) - 320))
            top = max(0, min(image.height - 640, round(cy) - 320))
            crop = image.crop((left, top, left + 640, top + 640))
            dr = ImageDraw.Draw(crop)
            for k, nodes in enumerate(points):
                for j, color in [(0, 'cyan'), (1, 'yellow')]:
                    if np.isfinite(nodes[j]).all():
                        x, y = nodes[j] - [left, top]
                        dr.ellipse((x - 3, y - 3, x + 3, y + 3), outline=color, width=2)
                        if j == 0 and 0 <= x < 640 and 0 <= y < 640:
                            dr.text((x + 4, y), str(k), fill='cyan')
            cropname = f'{video.stem}.frame_{idx:06d}.close_pair.png'
            crop.save(work / cropname)
            entry_out['close_pair_crop'] = {'file': cropname, 'original_bounds_xyxy': [left, top, left + 640, top + 640]}
        evidence.append(entry_out)
    cap.release()
    if tiles:
        sheet = Image.new('RGB', (1600, tiles[0].height * ((len(tiles) + 1) // 2)), 'black')
        for k, tile in enumerate(tiles):
            sheet.paste(tile, ((k % 2) * 800, (k // 2) * tile.height))
        sheet.save(work / f'{video.stem}.contact_sheet.png')
    return {'selection_rule': 'early/middle/late pairs; 39/40 chunk boundary; max-count pair; globally closest finite inbounds Shell pair. No claim of known physical identity or water semantics.',
            'frames': evidence}


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run via Slurm only'
    DEST.mkdir(parents=True, exist_ok=True)
    final = DEST / 'audit'
    if final.exists():
        raise RuntimeError(f'Refusing to overwrite {final}')
    lock = DEST / '.audit.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, os.environ['SLURM_JOB_ID'].encode())
    os.close(fd)
    work = DEST / f'.audit_job_{os.environ["SLURM_JOB_ID"]}'
    work.mkdir(exist_ok=False)
    try:
        versions = {'python': sys.version}
        for package in ['sleap-io', 'sleap-nn', 'numpy', 'h5py', 'pandas', 'pyarrow', 'fastparquet', 'opencv-python', 'av', 'Pillow', 'torch']:
            try:
                versions[package] = md.version(package)
            except md.PackageNotFoundError:
                versions[package] = None
        report = {'audit_version': 1, 'job_id': os.environ['SLURM_JOB_ID'],
                  'created_unix_s': time.time(), 'versions': versions,
                  'parquet_engine_available': any(importlib.util.find_spec(k) is not None for k in ['pyarrow', 'fastparquet']),
                  'files': [], 'videos': {},
                  'limitations': ['No inference; no writes to sources; external video paths never opened.',
                                  'Existing library source is supporting evidence, not a pin of the historical inference runtime.',
                                  'Missing frame records are unknown, not zero detections.',
                                  'Quantiles and nearest-neighbor distances are diagnostics, not validated identities.']}
        for p in [Path(__file__), ROOT / 'jobs/audit_legacy_dataset.sh', ROOT / 'scripts/gputest_predict.py',
                  ROOT / 'predict_dan_native.sh', ROOT / 'merge_dan_native.sh',
                  ROOT / '.venv/lib/python3.13/site-packages/sleap_nn/inference/predictors.py',
                  ROOT / '.venv/lib/python3.13/site-packages/sleap_io/io/slp.py']:
            report['files'].append(sha(p))
        for model in ['dan_centroid_native', 'dan_centered_native']:
            for name in ['best.ckpt', 'initial_config.yaml', 'training_config.yaml', 'TRAINING_COMPLETE']:
                p = ROOT / 'models' / model / name
                if p.exists():
                    report['files'].append(sha(p))
        for video_id in VIDEOS:
            print(f'Auditing {video_id}', flush=True)
            video = ROOT / 'crab_vids' / f'{video_id}.mp4'
            info = video_metadata(video)
            info['source'] = sha(video)
            n = info['container_frame_count']
            width, height = info['width'], info['height']
            measurement = Measurements(n, width, height)
            directory = ROOT / 'predictions' / video_id
            markers = {}
            for name in ['PREDICT_COMPLETE', '.predict_opts', '.video', '.next_frame']:
                p = directory / name
                if p.exists():
                    markers[name] = {'contents': p.read_text(), **sha(p)}
            inventory, digests, counts = [], {}, {}
            declared = np.zeros(n, dtype=np.int32)
            duplicate_frames = []
            for number, path in enumerate(sorted(directory.glob('*.frames_*.slp'))):
                match = re.fullmatch(re.escape(video_id) + r'\.frames_(\d+)_(\d+)\.slp', path.name)
                if match is None:
                    raise ValueError(f'Unexpected chunk name {path}')
                start, end = map(int, match.groups())
                rec, part_digest, part_count = inspect_file(path, video, width, height, measurement)
                rec['declared_range_inclusive'] = [start, end]
                rec['missing_declared_frames'] = sorted(set(range(start, end + 1)) - set(part_digest))
                rec['frames_outside_declared_range'] = sorted(k for k in part_digest if k < start or k > end)
                if 0 <= start <= end < n:
                    declared[start:end + 1] += 1
                else:
                    rec['issues'].append('Declared chunk range outside video')
                for idx in part_digest:
                    if idx in digests:
                        duplicate_frames.append(idx)
                digests.update(part_digest)
                counts.update(part_count)
                inventory.append(rec)
                if number % 100 == 0:
                    print(f'  {number + 1} chunks checked', flush=True)
            merged = directory / f'{video_id}.predictions.slp'
            if merged.exists():
                mrec, mdig, mcount = inspect_file(merged, video, width, height)
            else:
                mrec, mdig, mcount = {'path': str(merged.relative_to(ROOT)), 'readable_and_structurally_valid': False, 'issues': ['Missing']}, {}, {}
            mismatches = sorted(i for i in set(digests) & set(mdig) if digests[i] != mdig[i])
            unknown = sorted(set(range(n)) - set(digests))
            bad = [r['path'] for r in inventory if not r['readable_and_structurally_valid'] or r['issues'] or r['frames_outside_declared_range']]
            exact = not mismatches and set(mdig) == set(digests) and mrec['readable_and_structurally_valid']
            disposition = ('usable_as_is_for_posthoc_tracking' if exact else 'repairable_from_chunks') if not bad and not duplicate_frames and not unknown else 'partially_usable_with_explicit_exclusions_or_further_inspection'
            result = {'video': info, 'markers': markers, 'chunk_count': len(inventory),
                      'chunks': inventory, 'merged': mrec,
                      'coverage': {'declared_missing_frames': np.flatnonzero(declared == 0).tolist(),
                                   'declared_overlapping_frames': np.flatnonzero(declared > 1).tolist(),
                                   'unknown_missing_frame_records': unknown, 'duplicate_chunk_frame_records': duplicate_frames,
                                   'out_of_video_frame_records': sorted(i for i in digests if not 0 <= i < n),
                                   'chunk_frame_records': len(digests), 'chunk_instances': sum(counts.values()),
                                   'explicit_empty_frames': sorted(i for i, count in counts.items() if count == 0),
                                   'merged_exact_frame_payload_match': exact, 'merged_mismatch_frames': mismatches,
                                   'merged_only_frames': sorted(set(mdig) - set(digests)),
                                   'chunks_only_frames': sorted(set(digests) - set(mdig)), 'problem_chunk_files': bad},
                      'measurements': measurement.summary(), 'disposition': disposition}
            result['preview_evidence'] = previews(video, info, measurement, work)
            with (work / f'{video_id}.frames.csv').open('w') as fp:
                writer = csv.DictWriter(fp, fieldnames=['frame', 'status', 'source', 'count', 'trusted_count', 'closest_shell_pair_px'])
                writer.writeheader()
                for idx in range(n):
                    row = dict(measurement.frames.get(idx, {'frame': idx}))
                    row.pop('closest_pair_center', None)
                    row['status'] = 'recorded_empty' if idx in counts and counts[idx] == 0 else ('recorded_detections' if idx in counts else 'unknown_missing_record')
                    writer.writerow(row)
            report['videos'][video_id] = result
            dump(work / 'inventory.json', report)
            print(f'{video_id}: {disposition}; frames={len(digests)} observations={sum(counts.values())}', flush=True)
        dump(work / 'summary.json', {**{k: v for k, v in report.items() if k not in ['files', 'videos']},
                                    'videos': {k: {key: val for key, val in v.items() if key not in ['chunks', 'merged', 'markers']} for k, v in report['videos'].items()}})
        dump(work / 'AUDIT_COMPLETE.json', {'job_id': report['job_id'], 'inventory': sha(work / 'inventory.json'),
                                          'status': 'structural_audit_complete_visual_review_and_user_decisions_pending'})
        work.rename(final)
        print(f'Published {final}', flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
