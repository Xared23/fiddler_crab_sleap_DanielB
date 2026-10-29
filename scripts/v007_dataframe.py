#!/usr/bin/env python3
"""Validate existing v007 outputs and organize a new dataframe without retracking.

All execution is finite Slurm work. Source files and the legacy dataset remain
read-only. Build stages; publish requires a matching bounded-review receipt.
"""
from __future__ import annotations
import collections
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sleap_io as sio

from legacy_dataset.tracklets import water_membership

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/datasets/v007_best_body_first_dataframe_v001.json'
NODES = ['shell', 'shoulder', 'elbow', 'claw', 'arm']
BASE_SCHEMA = pa.schema([
    ('video_id', pa.string()), ('frame', pa.int64()), ('instance_index', pa.int32()),
    ('raw_instance_id', pa.int64()), ('detection_id', pa.string()), ('tracklet_id', pa.string()),
    ('tracklet_local_id', pa.int64()), ('accepted_for_tracking', pa.bool_()),
    ('rejection_reason', pa.string()), ('incoming_reason', pa.string()), ('instance_score', pa.float64()),
    ('body_x', pa.float64()), ('body_y', pa.float64()), ('body_score', pa.float64()),
    ('centroid_source', pa.string()), ('water', pa.bool_())
] + [(f'{node}_{suffix}', kind) for node in NODES for suffix, kind in
     [('x', pa.float64()), ('y', pa.float64()), ('score', pa.float64()), ('visible', pa.bool_())]])
EXTRA_SCHEMA = pa.schema([
    ('prediction_detection_id', pa.string()), ('dataset_id', pa.string()), ('prediction_run_id', pa.string()),
    ('tracking_algorithm', pa.string()), ('identity_status', pa.string()),
    ('centroid_x', pa.float64()), ('centroid_y', pa.float64()), ('time_s', pa.float64()),
    ('source_path', pa.string()), ('source_instance_index', pa.int64()), ('source_instance_row', pa.int64()),
    ('source_chunk_path', pa.string()), ('source_chunk_instance_row', pa.int64()),
    ('source_observations_path', pa.string()), ('source_tracking_score', pa.float64())])
OBS_SCHEMA = pa.schema(list(BASE_SCHEMA) + list(EXTRA_SCHEMA))


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as fp:
        for block in iter(lambda: fp.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(path, expected=None):
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
    if expected is not None:
        assert digest == expected, f'Hash mismatch: {path}'
    return {'path': str(path.relative_to(ROOT)), 'bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': digest}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as fp:
        json.dump(obj, fp, indent=2, allow_nan=False)
        fp.write('\n')
    assert read_json(path) == obj


def normal_float(array):
    """NaN is missing, but do not silently erase infinities or out-of-image values."""
    return pc.if_else(pc.is_nan(array), pa.scalar(None, array.type), array)


def normalize(table, cfg, run, video, chunk_paths, chunk_offsets, raw_tracking_score):
    assert table.schema == BASE_SCHEMA
    original_names = table['detection_id']
    arrays = [normal_float(table[f.name]) if pa.types.is_floating(f.type) else table[f.name] for f in BASE_SCHEMA]
    normalized = pa.Table.from_arrays(arrays, schema=BASE_SCHEMA)
    n = table.num_rows
    frame = table['frame'].to_numpy()
    raw_id = table['raw_instance_id'].to_numpy()
    chunk_idx = frame // run['chunk_frames']
    normalized = normalized.set_column(BASE_SCHEMA.get_field_index('detection_id'), 'detection_id',
                                       pa.array([cfg['dataset_id'] + '/' + x for x in original_names.to_pylist()], type=pa.string()))
    extras = {
        'prediction_detection_id': original_names,
        'dataset_id': [cfg['dataset_id']] * n,
        'prediction_run_id': [cfg['prediction_run_id']] * n,
        'tracking_algorithm': ['body_first_v001'] * n,
        'identity_status': ['experimental_not_ground_truth_validated'] * n,
        'centroid_x': normalized['body_x'], 'centroid_y': normalized['body_y'],
        'time_s': frame * (run['fps_denominator'] / run['fps_numerator']),
        'source_path': [f"{cfg['prediction_root']}/{video}/raw/{video}.predictions.slp"] * n,
        'source_instance_index': table['instance_index'].cast(pa.int64()),
        'source_instance_row': table['raw_instance_id'],
        'source_chunk_path': [chunk_paths[int(k)] for k in chunk_idx],
        'source_chunk_instance_row': raw_id - np.asarray(chunk_offsets)[chunk_idx],
        'source_observations_path': [f"{cfg['prediction_root']}/{video}/tracklets/observations.parquet"] * n,
        'source_tracking_score': normal_float(pa.array(raw_tracking_score, type=pa.float64()))}
    for field in EXTRA_SCHEMA:
        normalized = normalized.append_column(field, pa.array(extras[field.name], type=field.type))
    assert normalized.schema == OBS_SCHEMA
    return normalized


def same_numeric(a, b):
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def verify_measurements(table, instances, points, frames, expected_water, video, run_id, offset, tracked_instances):
    """Independently reconcile every saved row with the raw and tracked SLPs."""
    n = table.num_rows
    assert table.schema == BASE_SCHEMA
    ids = table['raw_instance_id'].to_numpy()
    same_numeric(ids, np.arange(offset, offset + n))
    same_numeric(instances['instance_id'], ids)
    assert np.all(instances['track'] == -1)
    same_numeric(instances['point_id_start'], ids * 5)
    same_numeric(instances['point_id_end'], (ids + 1) * 5)
    frame = table['frame'].to_numpy()
    assert np.all((frame >= 0) & (frame < len(frames)))
    same_numeric(frame, instances['frame_id'])
    same_numeric(table['instance_index'].to_numpy(), ids - frames['instance_id_start'][frame])
    assert table['video_id'].to_pylist() == [video] * n
    names = [f'{video}/f{int(f):06d}/i{int(i):04d}' for f, i in zip(frame, table['instance_index'].to_numpy())]
    assert table['detection_id'].to_pylist() == names
    for k, node in enumerate(NODES):
        for field in ['x', 'y', 'score', 'visible']:
            same_numeric(table[f'{node}_{field}'].to_numpy(), points[field][:, k])
    for field in ['x', 'y', 'score']:
        same_numeric(table[f'body_{field}'].to_numpy(), points[field][:, 0])
    same_numeric(table['instance_score'].to_numpy(), instances['score'])
    assert table['centroid_source'].to_pylist() == ['shell_proxy'] * n
    assert table['water'].to_pylist() == expected_water
    local = table['tracklet_local_id'].fill_null(0).to_numpy()
    assigned = table['accepted_for_tracking'].to_numpy()
    same_numeric(assigned, local > 0)
    assert table['tracklet_id'].to_pylist() == [f'{run_id}/{video}/t{int(i):09d}' if i > 0 else None for i in local]
    assert all((reason is None) == bool(accepted) for reason, accepted in zip(table['rejection_reason'].to_pylist(), assigned))
    same_numeric(tracked_instances['track'], np.where(assigned, local - 1, -1))
    restored = tracked_instances.copy()
    restored['track'] = instances['track']
    assert restored.tobytes() == instances.tobytes(), 'Tracked SLP changed raw measurements'


class Ledger:
    def __init__(self, nframes, ntracks):
        self.offset = 0
        self.counts = np.zeros(ntracks + 1, dtype=np.int64)
        self.first = np.full(ntracks + 1, -1, dtype=np.int64)
        self.last = self.first.copy()
        self.frames = np.zeros(nframes, dtype=np.int64)
        self.assigned_frames = self.frames.copy()
        self.water = collections.Counter()
        self.rejected = collections.Counter()
        self.incoming = collections.Counter()
        self.cross_chunk = 0

    def update(self, table, chunk_frames=40):
        raw_id = table['raw_instance_id'].to_numpy()
        same_numeric(raw_id, np.arange(self.offset, self.offset + len(raw_id)))
        self.offset += len(raw_id)
        frame = table['frame'].to_numpy()
        assert np.all(np.diff(frame) >= 0)
        local = table['tracklet_local_id'].fill_null(0).to_numpy()
        assigned = table['accepted_for_tracking'].to_numpy()
        same_numeric(local > 0, assigned)
        assert np.all((local >= 0) & (local < len(self.counts)))
        np.add.at(self.frames, frame, 1)
        np.add.at(self.assigned_frames, frame[assigned], 1)
        reasons = np.asarray(table['incoming_reason'].to_pylist(), object)
        if len(frame):
            starts = np.r_[0, np.flatnonzero(np.diff(frame)) + 1]
            ends = np.r_[starts[1:], len(frame)]
            for lo, hi in zip(starts, ends):
                mask = assigned[lo:hi]
                ids = local[lo:hi][mask]
                f = int(frame[lo])
                assert len(np.unique(ids)) == len(ids), 'Repeated tracklet within frame'
                old = self.counts[ids] > 0
                assert np.all(self.last[ids[old]] == f - 1), 'Gap or repeated frame in tracklet'
                same_numeric(old, reasons[lo:hi][mask] == 'linked')
                self.first[ids[~old]] = f
                self.last[ids] = f
                self.counts[ids] += 1
                if f > 0 and f % chunk_frames == 0:
                    self.cross_chunk += int(old.sum())
        self.incoming.update(table['incoming_reason'].to_pylist())
        self.rejected.update(x for x in table['rejection_reason'].to_pylist() if x is not None)
        self.water.update('unknown' if x is None else ('water' if x else 'land') for x in table['water'].to_pylist())

    def check_tracks(self, table, video, run_id):
        seen = np.zeros(len(self.counts), bool)
        ids = table['tracklet_local_id'].to_numpy()
        assert np.all((ids > 0) & (ids < len(self.counts))) and len(np.unique(ids)) == len(ids)
        seen[ids] = True
        assert seen[1:].all() and (self.counts[1:] > 0).all()
        assert table['video_id'].to_pylist() == [video] * len(ids)
        assert table['tracklet_id'].to_pylist() == [f'{run_id}/{video}/t{int(i):09d}' for i in ids]
        same_numeric(table['length'].to_numpy(), self.counts[ids])
        same_numeric(table['start_frame'].to_numpy(), self.first[ids])
        same_numeric(table['end_frame'].to_numpy(), self.last[ids])
        same_numeric(self.last[ids] - self.first[ids] + 1, self.counts[ids])

    def summary(self):
        length = self.counts[1:]
        return {'observations': self.offset, 'assigned_observations': int(length.sum()),
                'unassigned_observations': self.offset - int(length.sum()), 'tracklets': len(length),
                'singleton_tracklets': int(np.count_nonzero(length == 1)),
                'length_quantiles_0_25_50_75_95_99_100': np.quantile(length, [0, .25, .5, .75, .95, .99, 1]).tolist(),
                'observations_in_tracks_ge_30_frames': int(length[length >= 30].sum()),
                'observations_in_tracks_ge_150_frames': int(length[length >= 150].sum()),
                'chunk_boundary_links': self.cross_chunk, 'water_all_observations': dict(self.water),
                'rejection_reasons': dict(self.rejected), 'incoming_reasons': dict(self.incoming)}


def preview(video, path, captures, polygon, directory):
    """Six static frames/video only. Crop pairs are aligned, not full overlay movies."""
    import cv2
    from PIL import Image, ImageDraw
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened()
    records = []
    for start in [39, 14400, 28798]:
        a, b = captures[start], captures[start + 1]
        body = b[['body_x', 'body_y']].to_numpy(dtype=float, na_value=np.nan)
        eligible = b['accepted_for_tracking'].fillna(False).to_numpy(bool)
        water = b['water'].fillna(False).to_numpy(bool)
        indices = np.flatnonzero(eligible & water & np.isfinite(body).all(1))
        if not len(indices):
            indices = np.flatnonzero(eligible & np.isfinite(body).all(1))
        center = body[indices[0]] if len(indices) else np.array([3840, 2160])
        left, top = max(0, min(6880, round(center[0]) - 400)), max(0, min(3520, round(center[1]) - 400))
        canvas = Image.new('RGB', (1600, 840), '#202020')
        for side, df in enumerate([a, b]):
            f = start + side
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, image = cap.read()
            assert ok and image.shape[:2] == (4320, 7680)
            crop = Image.fromarray(cv2.cvtColor(image[top:top + 800, left:left + 800], cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(crop)
            vertices = [(x - left, y - top) for x, y in polygon]
            draw.line(vertices + [vertices[0]], fill='orange', width=2)
            for row in df.itertuples():
                if pd.isna(row.body_x) or pd.isna(row.body_y):
                    continue
                x, y = row.body_x - left, row.body_y - top
                if not (0 <= x < 800 and 0 <= y < 800):
                    continue
                assigned = bool(row.accepted_for_tracking)
                color = 'cyan' if assigned else 'red'
                label = f't{int(row.tracklet_local_id)}' if assigned else f'i{row.instance_index}:unassigned'
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                draw.text((x + 6, y - 10), label, fill=color)
                if not pd.isna(row.shoulder_x) and not pd.isna(row.shoulder_y) and np.isfinite([row.shoulder_x, row.shoulder_y]).all():
                    sx, sy = row.shoulder_x - left, row.shoulder_y - top
                    draw.line((x, y, sx, sy), fill=color)
                    draw.rectangle((sx - 2, sy - 2, sx + 2, sy + 2), outline='yellow')
            canvas.paste(crop, (800 * side, 40))
            ImageDraw.Draw(canvas).text((800 * side + 5, 5), f'{video} frame {f}: EXISTING body-first IDs (experimental)', fill='white')
            ImageDraw.Draw(canvas).text((800 * side + 5, 22), f'Original crop x={left}, y={top}; orange = SAME fixed water ROI', fill='orange')
        filename = f'{video}.frames_{start:06d}_{start + 1:06d}.png'
        canvas.save(directory / filename)
        records.append({'video_id': video, 'frames': [start, start + 1], 'crop_xyxy': [left, top, left + 800, top + 800],
                        'selection': 'first assigned finite water point in second frame, else first assigned finite body',
                        'file': filename, 'sha256': sha(directory / filename)})
    cap.release()
    return records


def build():
    job = os.environ['SLURM_JOB_ID']
    cfg = read_json(CONFIG)
    target = ROOT / cfg['destination']
    assert not target.exists(), 'Refuse to overwrite existing dataset'
    stage = target.with_name('.' + target.name + '.build_' + job)
    lock = target.with_name('.' + target.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        stage.mkdir(exist_ok=False)
        for name in ['tables/observations', 'tables/unassigned_observations', 'config', 'audit', 'qc']:
            (stage / name).mkdir(parents=True, exist_ok=True)
        pins = [pin(CONFIG), pin(Path(__file__)), pin(ROOT / 'tests/test_v007_dataframe.py'), pin(ROOT / 'jobs/v007_dataframe.sh')]
        prediction = ROOT / cfg['prediction_root']
        source_manifest = read_json(prediction / 'RUN_MANIFEST.json')
        pins.append(pin(prediction / 'RUN_MANIFEST.json'))
        run = source_manifest['config']
        assert run['run_id'] == cfg['prediction_run_id'] and run['nodes'] == [x.capitalize() for x in NODES]
        assert (run['width'], run['height'], run['frames_per_video']) == (7680, 4320, 28800)
        for filename, digest in source_manifest['hashes'].items():
            pins.append(pin(ROOT / filename, digest))
        roi = read_json(ROOT / run['water_regions'])
        assert roi['boundary_included'] and roi['uncertainty_band_px'] == 0
        dump(stage / 'config/export.json', cfg)
        dump(stage / 'config/prediction_run.json', run)
        dump(stage / 'config/water_regions.json', roi)
        dump(stage / 'audit/prediction_run_manifest.json', source_manifest)
        testpath = ROOT / 'checks/v007_dataframe' / f'tests_{job}.json'
        tests = read_json(testpath)
        assert tests['success'] and tests['code_sha256'] == sha(Path(__file__))
        assert tests['test_sha256'] == sha(ROOT / 'tests/test_v007_dataframe.py')
        dump(stage / 'audit/tests.json', tests)
        stats, previews, combined_tracks, frame_tables = {}, [], [], []
        signature = hashlib.sha256(json.dumps(source_manifest, sort_keys=True).encode()).hexdigest()
        for video in cfg['videos']:
            print('VERIFY_AND_EXPORT', video, flush=True)
            src = prediction / video
            complete = read_json(src / 'PREDICT_COMPLETE.json')
            raw_receipt = read_json(src / 'raw/RAW_COMPLETE.json')
            chunk_receipt = read_json(src / 'raw/CHUNKS_COMPLETE.json')
            tracking = read_json(src / 'tracklets/TRACKLETS_COMPLETE.json')
            assert complete['signature'] == raw_receipt['signature'] == chunk_receipt['signature'] == signature
            assert tracking == complete['tracklets']
            assert complete['raw'] == {k: raw_receipt[k] for k in ['frames', 'instances', 'sha256']}
            assert raw_receipt['frames'] == chunk_receipt['frames'] == run['frames_per_video']
            assert complete['identities_optional'] and not raw_receipt['raw_tracking_enabled']
            raw_path = src / 'raw' / f'{video}.predictions.slp'
            tracked_path = src / 'tracklets' / f'{video}.tracked.slp'
            obs_path, track_path = src / 'tracklets/observations.parquet', src / 'tracklets/tracklets.parquet'
            pins.append(pin(raw_path, raw_receipt['sha256']))
            for filename, digest in tracking['files'].items():
                pins.append(pin(src / 'tracklets' / filename, digest))
            for filename in ['PREDICT_COMPLETE.json', 'raw/RAW_COMPLETE.json', 'raw/CHUNKS_COMPLETE.json', 'tracklets/TRACKLETS_COMPLETE.json']:
                pins.append(pin(src / filename))
            video_path = ROOT / run['videos'][video]
            recorded_video = source_manifest['videos'][video]
            assert video_path.stat().st_size == recorded_video['bytes'] and video_path.stat().st_mtime_ns == recorded_video['mtime_ns']
            pins.append(pin(video_path))
            labels = sio.load_slp(str(raw_path), open_videos=False, lazy=True)
            assert len(labels) == run['frames_per_video'] and not labels.tracks
            assert labels.skeletons[0].node_names == run['nodes']
            assert Path(labels.videos[0].filename) == video_path
            for f in [0, 39, 40, 14400, 28799]:
                assert labels[f].frame_idx == f
            del labels
            chunk_paths, offsets = [], []
            with h5py.File(raw_path, 'r') as raw, h5py.File(tracked_path, 'r') as tracked:
                frames = raw['frames'][:]
                same_numeric(frames['frame_idx'], np.arange(run['frames_per_video']))
                same_numeric(frames['frame_id'], np.arange(run['frames_per_video']))
                assert np.all(frames['video'] == 0)
                assert frames['instance_id_start'][0] == 0
                same_numeric(frames['instance_id_start'][1:], frames['instance_id_end'][:-1])
                assert frames['instance_id_end'][-1] == len(raw['instances']) == raw_receipt['instances'] == tracking['observations']
                assert len(raw['pred_points']) == len(tracked['pred_points']) == 5 * len(raw['instances'])
                assert len(tracked['instances']) == len(raw['instances'])
                assert raw['frames'][:].tobytes() == tracked['frames'][:].tobytes()
                assert len(raw['points']) == len(tracked['points']) == 0
                catalog = [json.loads(x) for x in tracked['tracks_json'][:]]
                assert len(catalog) == tracking['tracklets']
                for i, item in enumerate(catalog, 1):
                    assert item[1] == f"{run['run_id']}/{video}/t{i:09d}"
                assert len(chunk_receipt['chunks']) == 720
                end = off = 0
                for chunk in chunk_receipt['chunks']:
                    start, stop = chunk['start'], chunk['stop']
                    assert start == end and stop == start + run['chunk_frames']
                    assert chunk['signature'] == signature and chunk['tracking_enabled'] is False
                    path = src / 'raw' / chunk['filename']
                    assert path.parent == src / 'raw'
                    pins.append(pin(path, chunk['sha256']))
                    pins.append(pin(path.with_suffix('.json')))
                    saved = read_json(path.with_suffix('.json'))
                    assert saved == {k: v for k, v in chunk.items() if k != 'filename'}
                    chunk_paths.append(str(path.relative_to(ROOT)))
                    offsets.append(off)
                    with h5py.File(path, 'r') as part:
                        f, ins, pts = part['frames'][:], part['instances'][:], part['pred_points'][:]
                        assert len(ins) == chunk['instances'] and np.all(ins['track'] == -1)
                        same_numeric(f['frame_idx'], np.arange(start, stop))
                        f['frame_id'] += start
                        f['instance_id_start'] += off
                        f['instance_id_end'] += off
                        ins['frame_id'] += start
                        ins['instance_id'] += off
                        ins['point_id_start'] += 5 * off
                        ins['point_id_end'] += 5 * off
                        assert raw['frames'][start:stop].tobytes() == f.tobytes()
                        assert raw['instances'][off:off + len(ins)].tobytes() == ins.tobytes()
                        assert raw['pred_points'][off * 5:(off + len(ins)) * 5].tobytes() == pts.tobytes()
                    end, off = stop, off + len(ins)
                assert end == run['frames_per_video'] and off == len(raw['instances'])
                tracks = pq.read_table(track_path)
                ledger = Ledger(len(frames), tracking['tracklets'])
                outdir = stage / 'tables/observations' / video
                rejectdir = stage / 'tables/unassigned_observations' / video
                outdir.mkdir()
                rejectdir.mkdir()
                captures = collections.defaultdict(list)
                part_count = 0
                for batch in pq.ParquetFile(obs_path).iter_batches(batch_size=cfg['batch_rows']):
                    table = pa.Table.from_batches([batch])
                    lo, hi = ledger.offset, ledger.offset + table.num_rows
                    instances = raw['instances'][lo:hi]
                    pts = raw['pred_points'][lo * 5:hi * 5].reshape(-1, 5)
                    ti = tracked['instances'][lo:hi]
                    assert tracked['pred_points'][lo * 5:hi * 5].tobytes() == pts.ravel().tobytes()
                    expected_water = water_membership(np.column_stack([pts['x'][:, 0], pts['y'][:, 0]]), run['width'], run['height'], roi['videos'][video]['polygon_xy'])
                    verify_measurements(table, instances, pts, frames, expected_water, video, run['run_id'], lo, ti)
                    output = normalize(table, cfg, run, video, chunk_paths, offsets, instances['tracking_score'])
                    partpath = outdir / f'part-{part_count:05d}.parquet'
                    pq.write_table(output, partpath, compression='zstd')
                    back = pq.read_table(partpath)
                    assert back.equals(output)
                    ledger.update(back, run['chunk_frames'])
                    unassigned = back.filter(pc.invert(back['accepted_for_tracking']))
                    if unassigned.num_rows:
                        rejectpath = rejectdir / partpath.name
                        pq.write_table(unassigned, rejectpath, compression='zstd')
                        assert pq.read_table(rejectpath).equals(unassigned)
                    fidx = back['frame'].to_numpy()
                    for f in cfg['preview_frames']:
                        if np.any(fidx == f):
                            captures[f].append(back.filter(pa.array(fidx == f)))
                    part_count += 1
                ledger.check_tracks(tracks, video, run['run_id'])
                same_numeric(ledger.first[1:], np.array([x[0] for x in catalog]))
                same_numeric(ledger.frames, frames['instance_id_end'] - frames['instance_id_start'])
                assert ledger.offset == tracking['observations']
                assert int(ledger.counts.sum()) == tracking['accepted_for_tracking']
                if not list(rejectdir.iterdir()):
                    pq.write_table(pa.Table.from_pylist([], schema=OBS_SCHEMA), rejectdir / 'part-00000.parquet')
                stats[video] = ledger.summary()
                stats[video]['source_finalizer_job'] = complete['job_id']
                stats[video]['checks'] = 'all raw measurements, source counts, IDs, consecutive frames, ROI and Parquet readback passed'
                combined_tracks.append(tracks)
                frame_tables.append(pa.table({'video_id': [video] * len(frames), 'frame': np.arange(len(frames)),
                                              'source_count': ledger.frames, 'assigned_count': ledger.assigned_frames,
                                              'unassigned_count': ledger.frames - ledger.assigned_frames,
                                              'status': ['recorded_detections' if x else 'recorded_empty' for x in ledger.frames]}))
                for name in ['observations', 'unassigned_observations']:
                    df = pd.read_parquet(stage / 'tables' / name / video, columns=['frame', 'centroid_x', 'water', 'tracklet_local_id'],
                                         filters=[('frame', '<', 80)], dtype_backend='numpy_nullable')
                    assert str(df['water'].dtype) == 'boolean' and str(df['centroid_x'].dtype) == 'Float64'
                    assert str(df['tracklet_local_id'].dtype) == 'Int64'
                captures = {f: pa.concat_tables(items).to_pandas(types_mapper=pd.ArrowDtype) for f, items in captures.items()}
                assert set(captures) == set(cfg['preview_frames'])
            previews.extend(preview(video, video_path, captures, roi['videos'][video]['polygon_xy'], stage / 'qc'))
            print(json.dumps(stats[video], indent=2), flush=True)
        for name, tables in [('tracklets', combined_tracks), ('frames', frame_tables)]:
            table = pa.concat_tables(tables)
            path = stage / 'tables' / (name + '.parquet')
            pq.write_table(table, path, compression='zstd')
            assert pq.read_table(path).equals(table)
        for record in pins:
            stat = (ROOT / record['path']).stat()
            assert (stat.st_size, stat.st_mtime_ns) == (record['bytes'], record['mtime_ns']), record['path']
        dump(stage / 'qc/summary.json', stats)
        dump(stage / 'qc/previews.json', previews)
        artifacts = [{'path': str(p.relative_to(stage)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(stage.rglob('*')) if p.is_file()]
        manifest = {'dataset_id': cfg['dataset_id'], 'prediction_run_id': cfg['prediction_run_id'], 'build_job_id': job,
                    'source_run_signature': signature, 'source_pins': pins, 'artifacts': artifacts, 'videos': stats,
                    'schema': str(OBS_SCHEMA), 'python': sys.version,
                    'versions': {p: metadata.version(p) for p in ['pandas', 'pyarrow', 'numpy', 'h5py', 'sleap-io', 'shapely']},
                    'tracking_recomputed': False, 'source_ids_preserved': True, 'inference_rerun': False,
                    'row_policy': cfg['row_policy'], 'centroid_source': 'shell_proxy',
                    'water_status': 'accepted_fixed_polygon_shell_point_boundary_included',
                    'unknown_frames': 0, 'identity_accuracy': 'experimental_not_ground_truth_validated',
                    'selection_note': 'Unassigned table is a subset/view-copy of observations, not additional detections. Unlike legacy canonical observations, the primary table retains all source detections.'}
        dump(stage / 'manifest.json', manifest)
        dump(stage / 'BUILD_VALIDATED.json', {'build_job_id': job, 'manifest_sha256': sha(stage / 'manifest.json'), 'dataset_complete': False})
        print('VALIDATED_STAGE', stage, flush=True)
    finally:
        lock.unlink()


def publish(build_id):
    assert build_id.isdigit()
    cfg = read_json(CONFIG)
    target = ROOT / cfg['destination']
    stage = target.with_name('.' + target.name + '.build_' + build_id)
    assert not target.exists()
    lock = target.with_name('.' + target.name + '.lock')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        validated = read_json(stage / 'BUILD_VALIDATED.json')
        assert validated['build_job_id'] == build_id and sha(stage / 'manifest.json') == validated['manifest_sha256']
        manifest = read_json(stage / 'manifest.json')
        review = read_json(stage / 'qc/review.json')
        assert review['manifest_sha256'] == validated['manifest_sha256']
        assert review['disposition'] == 'accept_with_documented_limits'
        assert sorted(review['reviewed_previews']) == sorted(x['file'] for x in read_json(stage / 'qc/previews.json'))
        for record in manifest['source_pins']:
            path = ROOT / record['path']
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (record['bytes'], record['mtime_ns']), path
            if path.suffix in ['.py', '.sh', '.json', '.yaml']:
                assert sha(path) == record['sha256']
        for record in manifest['artifacts']:
            path = stage / record['path']
            assert sha(path) == record['sha256']
        marker = {'dataset_id': cfg['dataset_id'], 'status': 'validated_complete', 'build_job_id': build_id,
                  'publication_job_id': os.environ['SLURM_JOB_ID'], 'manifest_sha256': sha(stage / 'manifest.json'),
                  'review_sha256': sha(stage / 'qc/review.json'), 'centroid_source': 'shell_proxy',
                  'water_status': manifest['water_status'], 'identity_accuracy': manifest['identity_accuracy'],
                  'counts': {key: sum(v[key] for v in manifest['videos'].values()) for key in
                             ['observations', 'assigned_observations', 'unassigned_observations', 'tracklets']}}
        dump(stage / 'DATASET_COMPLETE.json', marker)
        stage.rename(target)
        print(json.dumps(marker, indent=2))
    finally:
        lock.unlink()


if __name__ == '__main__':
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    if sys.argv[1:] == ['build']:
        build()
    elif len(sys.argv) == 3 and sys.argv[1] == 'publish':
        publish(sys.argv[2])
    else:
        raise SystemExit('Usage: v007_dataframe.py build | publish BUILD_JOB_ID')
