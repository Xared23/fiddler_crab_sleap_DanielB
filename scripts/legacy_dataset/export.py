#!/usr/bin/env python3
"""Finite streaming conversion and validation; publication requires QC review.

Original SLPs/videos/models are only opened read-only. No inference imports.
Run through jobs/build_legacy_dataset.sh; no automatic retry or requeue.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from tracklets import Frame, Tracker, water_membership

ROOT = Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap')
DEST = ROOT / 'datasets/legacy_native_tracklets_v001'
DATASET_ID = DEST.name


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    before = path.stat()
    digest = checksum(path)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return {'path': str(path.relative_to(ROOT)), 'bytes': after.st_size,
            'mtime_ns': after.st_mtime_ns, 'sha256': digest}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def atomic_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    assert not path.exists() and not tmp.exists()
    dump(tmp, obj)
    assert json.loads(tmp.read_text()) == obj
    tmp.replace(path)


TEXT = ['tracklet_id', 'video_id', 'detection_id', 'centroid_source', 'source_path',
        'incoming_reason', 'rejection_reason', 'predecessor_detection_id']
INTEGER = ['tracklet_local_id', 'frame', 'source_instance_index', 'source_instance_row', 'source_track_index']
FLOAT = ['time_s', 'centroid_x', 'centroid_y', 'body_x', 'body_y', 'body_score',
         'shoulder_x', 'shoulder_y', 'shoulder_score', 'instance_score',
         'source_tracking_score', 'body_displacement_px', 'shoulder_displacement_px']
BOOL = ['water', 'body_visible', 'shoulder_visible', 'shoulder_valid', 'association_eligible', 'duplicate_candidate']
OBS_SCHEMA = pa.schema([(k, pa.string()) for k in TEXT] + [(k, pa.int64()) for k in INTEGER]
                       + [(k, pa.float64()) for k in FLOAT] + [(k, pa.bool_()) for k in BOOL])
TRACK_SCHEMA = pa.schema([(k, pa.string()) for k in ['tracklet_id', 'video_id', 'start_reason', 'termination_reason']]
                        + [(k, pa.int64()) for k in ['tracklet_local_id', 'start_frame', 'end_frame', 'length']])
FRAME_SCHEMA = pa.schema([('video_id', pa.string()), ('frame', pa.int64()), ('status', pa.string())]
                        + [(k, pa.int64()) for k in ['source_count', 'accepted_count', 'rejected_count', 'linked_count']])


def table_from_columns(columns, schema):
    arrays = []
    for field in schema:
        value = columns[field.name]
        if pa.types.is_floating(field.type):
            value = np.asarray(value, float)
            arrays.append(pa.array(value, mask=~np.isfinite(value), type=field.type))
        else:
            arrays.append(pa.array(value, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


class Parts:
    """~100k-row Parquet parts with exact whole-table read-back on each flush."""
    def __init__(self, directory, schema):
        self.directory, self.schema = directory, schema
        directory.mkdir(parents=True, exist_ok=False)
        self.parts = []
        self.pending = []
        self.n = self.total = 0

    def add(self, table):
        assert table.schema == self.schema
        if table.num_rows:
            self.pending.append(table)
            self.n += table.num_rows
            self.total += table.num_rows
        if self.n >= 100000:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        table = pa.concat_tables(self.pending)
        path = self.directory / f'part-{len(self.parts):05d}.parquet'
        pq.write_table(table, path, compression='zstd', row_group_size=100000)
        readback = pq.read_table(path)
        assert table.equals(readback), path
        self.parts.append(path)
        self.pending, self.n = [], 0

    def close(self):
        self.flush()
        if not self.parts:
            path = self.directory / 'part-00000.parquet'
            table = pa.Table.from_pylist([], schema=self.schema)
            pq.write_table(table, path, compression='zstd')
            assert pq.read_table(path).equals(table)
            self.parts.append(path)


class SingleFile:
    def __init__(self, path, schema):
        self.path, self.schema = path, schema
        self.writer = pq.ParquetWriter(path, schema, compression='zstd')
        self.pending = []
        self.total = 0

    def add(self, rows):
        self.pending.extend(rows)
        self.total += len(rows)
        if len(self.pending) >= 100000:
            self.flush()

    def flush(self):
        if self.pending:
            self.writer.write_table(pa.Table.from_pylist(self.pending, schema=self.schema))
            self.pending = []

    def close(self):
        self.flush()
        self.writer.close()
        assert pq.read_metadata(self.path).num_rows == self.total
        assert pq.read_schema(self.path) == self.schema


def detection_name(video, frame, index):
    return f'{DATASET_ID}/{video}/f{frame:06d}/i{index:04d}'


def rows_for_frame(video, source, fr, instances, points, tracker, result, roi, fps, previous_capture):
    n, frame = len(instances), int(fr['frame_idx'])
    ids, valid = result['ids'], result['validity']
    xy = np.stack([points['x'], points['y']], axis=-1)
    pred = result['predecessor_index']
    predecessor_names = [None] * n
    bd, sd = np.full(n, np.nan), np.full(n, np.nan)
    if previous_capture is not None:
        for j in np.flatnonzero(pred >= 0):
            i = int(pred[j])
            predecessor_names[j] = previous_capture['names'][i]
            bd[j] = np.linalg.norm(xy[j, 0] - previous_capture['xy'][i, 0])
            sd[j] = np.linalg.norm(xy[j, 1] - previous_capture['xy'][i, 1])
    names = [detection_name(video, frame, j) for j in range(n)]
    water = water_membership(xy[:, 0], tracker.width, tracker.height, roi['polygon_xy'], confirmed=True)
    # Invalid Shell scores/visibility do not get meaningful water classifications.
    # Duplicate candidates may have a valid location, but remain rejected rows.
    for j in range(n):
        if valid['reasons'][j] in ['missing_position', 'outside_image', 'invisible', 'low_body_score']:
            water[j] = None
    cols = {'tracklet_id': [tracker.name(int(i)) if i > 0 else None for i in ids],
            'tracklet_local_id': [int(i) if i > 0 else None for i in ids],
            'video_id': [video] * n, 'detection_id': names,
            'frame': [frame] * n, 'source_instance_index': list(range(n)),
            'source_instance_row': list(range(int(fr['instance_id_start']), int(fr['instance_id_end']))),
            'source_track_index': instances['track'], 'source_path': [source] * n,
            'centroid_source': ['shell_proxy'] * n, 'centroid_x': xy[:, 0, 0], 'centroid_y': xy[:, 0, 1],
            'body_x': xy[:, 0, 0], 'body_y': xy[:, 0, 1], 'body_score': points['score'][:, 0],
            'shoulder_x': xy[:, 1, 0], 'shoulder_y': xy[:, 1, 1], 'shoulder_score': points['score'][:, 1],
            'time_s': np.full(n, frame / fps), 'instance_score': instances['score'],
            'source_tracking_score': instances['tracking_score'],
            'incoming_reason': [str(result['incoming_reason'][j]) if ids[j] > 0 else 'rejected' for j in range(n)],
            'rejection_reason': [str(r) if r else None for r in valid['reasons']],
            'predecessor_detection_id': predecessor_names,
            'body_displacement_px': bd, 'shoulder_displacement_px': sd,
            'water': water, 'body_visible': points['visible'][:, 0], 'shoulder_visible': points['visible'][:, 1],
            'shoulder_valid': valid['shoulder_valid'], 'association_eligible': valid['linkable'],
            'duplicate_candidate': valid['duplicate']}
    table = table_from_columns(cols, OBS_SCHEMA)
    accepted = table.filter(pa.array(valid['accepted']))
    rejected = table.filter(pa.array(~valid['accepted']))
    assert accepted.num_rows + rejected.num_rows == n
    assert sorted(accepted['source_instance_index'].to_pylist() + rejected['source_instance_index'].to_pylist()) == list(range(n))
    capture = {'frame': frame, 'names': names, 'xy': xy.copy(), 'ids': ids.copy(),
               'accepted': valid['accepted'].copy(), 'water': water,
               'incoming': list(map(str, result['incoming_reason'])), 'reasons': list(map(str, valid['reasons'])),
               'body_displacement': bd, 'source': source}
    return accepted, rejected, capture


def validate_saved_video(work, video, ntracks, source_counts, config):
    """Read all saved observations/rejections and check keys, counts and continuity.

Memory is O(number of source observations + tracklets), using numeric arrays,
not millions of Python instances or full-video pose objects.
"""
    base = np.r_[0, np.cumsum(source_counts)]
    seen = np.zeros(int(base[-1]), dtype=np.uint8)
    count = np.zeros(ntracks + 1, dtype=np.int64)
    first = np.full(ntracks + 1, -1, dtype=np.int64)
    last = first.copy()
    per_frame = {'observations': np.zeros(len(source_counts), dtype=np.int64),
                 'rejected_observations': np.zeros(len(source_counts), dtype=np.int64)}
    water_counts = collections.Counter()
    reasons = collections.Counter()
    incoming = collections.Counter()
    prefix = f'{DATASET_ID}/{video}/t'
    for kind in per_frame:
        for path in sorted((work / 'tables' / kind / video).glob('*.parquet')):
            assert pq.read_schema(path) == OBS_SCHEMA
            for batch in pq.ParquetFile(path).iter_batches(batch_size=100000):
                df = batch.to_pandas(types_mapper=pd.ArrowDtype)
                frame = df['frame'].to_numpy(dtype=np.int64)
                inst = df['source_instance_index'].to_numpy(dtype=np.int64)
                assert (df['video_id'] == video).all()
                assert np.all((frame >= 0) & (frame < len(source_counts)))
                assert np.all((inst >= 0) & (inst < source_counts[frame]))
                assert np.all(np.diff(frame) >= 0)
                ordinal = base[frame] + inst
                np.add.at(seen, ordinal, 1)
                np.add.at(per_frame[kind], frame, 1)
                assert df['detection_id'].tolist() == [detection_name(video, int(f), int(i)) for f, i in zip(frame, inst)]
                assert (df['centroid_source'] == 'shell_proxy').all()
                assert df['centroid_x'].equals(df['body_x']) and df['centroid_y'].equals(df['body_y'])
                if kind == 'observations':
                    assert df['rejection_reason'].isna().all()
                    assert not df['tracklet_id'].isna().any()
                    assert df['water'].dtype == pd.ArrowDtype(pa.bool_())
                    assert df['body_x'].notna().all() and df['body_y'].notna().all()
                    assert (df['body_score'] >= config['body_score_min']).all()
                    local = df['tracklet_local_id'].to_numpy(dtype=np.int64)
                    assert np.all((local > 0) & (local <= ntracks))
                    assert df['tracklet_id'].tolist() == [prefix + f'{i:09d}' for i in local]
                    starts = np.r_[0, np.flatnonzero(np.diff(frame)) + 1]
                    ends = np.r_[starts[1:], len(frame)]
                    for a, b in zip(starts, ends):
                        ids = local[a:b]
                        f = frame[a]
                        assert len(np.unique(ids)) == len(ids), '(tracklet_id, frame) duplicate'
                        old = count[ids] > 0
                        assert np.all(last[ids[old]] == f - 1), 'Gap bridged or duplicate tracklet frame'
                        is_linked = np.asarray(df['incoming_reason'].iloc[a:b] == 'linked', dtype=bool)
                        assert np.array_equal(old, is_linked), 'Incoming-link/start mismatch'
                        first[ids[~old]] = f
                        last[ids] = f
                        count[ids] += 1
                    assert np.all(df.loc[df['incoming_reason'] == 'linked', 'body_displacement_px'].to_numpy(float) <= config['max_body_displacement_px'])
                    assert np.all(df.loc[df['incoming_reason'] == 'linked', 'shoulder_displacement_px'].to_numpy(float) <= config['max_shoulder_displacement_px'])
                    assert df.loc[df['incoming_reason'] == 'linked', 'association_eligible'].all()
                    assert df.loc[df['incoming_reason'] == 'linked', 'predecessor_detection_id'].notna().all()
                    for value, n in df['water'].value_counts(dropna=False).items():
                        water_counts['unknown' if pd.isna(value) else ('water' if value else 'land')] += int(n)
                    incoming.update(df['incoming_reason'].tolist())
                else:
                    assert df['tracklet_id'].isna().all() and df['tracklet_local_id'].isna().all()
                    assert df['rejection_reason'].notna().all()
                    reasons.update(df['rejection_reason'].tolist())
    assert np.all(seen == 1), 'Source accounting: drop or duplication'
    assert np.array_equal(per_frame['observations'] + per_frame['rejected_observations'], source_counts)
    assert np.all(count[1:] > 0), 'ID gaps or missing tracklet observations'
    assert np.array_equal(last[1:] - first[1:] + 1, count[1:])
    track_seen = np.zeros(ntracks + 1, bool)
    terminations = collections.Counter()
    for batch in pq.ParquetFile(work / 'tables/tracklets.parquet').iter_batches(batch_size=100000):
        df = batch.to_pandas()
        df = df.loc[df['video_id'] == video]
        if df.empty:
            continue
        ids = df['tracklet_local_id'].to_numpy(dtype=np.int64)
        assert len(np.unique(ids)) == len(ids) and not track_seen[ids].any()
        assert df['tracklet_id'].tolist() == [prefix + f'{i:09d}' for i in ids]
        assert np.array_equal(df['length'].to_numpy(), count[ids])
        assert np.array_equal(df['start_frame'].to_numpy(), first[ids])
        assert np.array_equal(df['end_frame'].to_numpy(), last[ids])
        track_seen[ids] = True
        terminations.update(df['termination_reason'].tolist())
    assert track_seen[1:].all()
    # Explicit pandas nullable dtype loading, not just Arrow metadata inspection.
    example = sorted((work / 'tables/observations' / video).glob('*.parquet'))[0]
    pdf = pd.read_parquet(example, dtype_backend='numpy_nullable')
    assert str(pdf['water'].dtype) == 'boolean'
    assert str(pdf['body_score'].dtype) == 'Float64'
    assert str(pdf['frame'].dtype) == 'Int64'
    lengths = count[1:]
    return {'source_observations': int(base[-1]), 'accepted_observations': int(per_frame['observations'].sum()),
            'rejected_observations': int(per_frame['rejected_observations'].sum()),
            'tracklets': ntracks, 'singletons': int(np.count_nonzero(lengths == 1)),
            'length_quantiles_0_25_50_75_95_99_100': np.quantile(lengths, [0, .25, .5, .75, .95, .99, 1]).tolist(),
            'length_histogram': {str(int(k)): int(v) for k, v in zip(*np.unique(lengths, return_counts=True))},
            'incoming_reasons': dict(incoming), 'termination_reasons': dict(terminations),
            'rejection_reasons': dict(reasons), 'water_counts': dict(water_counts),
            'saved_count_key_continuity_and_dtype_checks': 'passed'}


def render_review(video, video_info, captures, audit_result, destination, roi):
    """Six explicitly selected adjacent pairs/video, bounded crop previews only."""
    import cv2
    from PIL import Image, ImageDraw
    starts = [0, 39, 14400, audit_result['measurements']['densest_frame'],
              audit_result['measurements']['closest_pair_frame'], video_info['container_frame_count'] - 2]
    starts = sorted(set(starts))
    cap = cv2.VideoCapture(str(ROOT / video_info['source']['path']))
    assert cap.isOpened()
    records = []
    palette = ['#00ff80', '#00cfff', '#ff80ff', '#ffff00', '#ffb050', '#b0a0ff']
    for start in starts:
        a, b = captures[start], captures[start + 1]
        if start == audit_result['measurements']['closest_pair_frame']:
            center = audit_result['measurements']['closest_pair_center']
            selection = 'audit_global_closest_shell_pair'
        else:
            candidates = np.flatnonzero(b['accepted'] & np.array([x == 'linked' for x in b['incoming']]))
            water_candidates = [i for i in candidates if b['water'][i] is True]
            if start in [39, 14400] and water_candidates:
                selected = water_candidates[0]
                selection = 'first_linked_water_detection'
            elif len(candidates):
                selected = candidates[int(np.argmax(b['body_displacement'][candidates]))]
                selection = 'largest_accepted_body_step_in_selected_pair'
            else:
                valid = np.flatnonzero(np.isfinite(b['xy'][:, 0]).all(1))
                selected = int(valid[0]) if len(valid) else None
                selection = 'first_finite_body_no_accepted_links'
            center = b['xy'][selected, 0].tolist() if selected is not None else [3840, 2160]
        left = max(0, min(video_info['width'] - 800, round(center[0]) - 400))
        top = max(0, min(video_info['height'] - 800, round(center[1]) - 400))
        sheet = Image.new('RGB', (1600, 860), '#202020')
        rows = []
        for side, capture in enumerate([a, b]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, capture['frame'])
            ok, bgr = cap.read()
            assert ok, f'Decode failed {video} {capture["frame"]}'
            image = Image.fromarray(cv2.cvtColor(bgr[top:top + 800, left:left + 800], cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            polygon = [(x - left, y - top) for x, y in roi['polygon_xy']]
            draw.line(polygon + [polygon[0]], fill='orange', width=2)
            for j, (body, shoulder) in enumerate(capture['xy'][:, :2]):
                if not np.isfinite(body).all():
                    continue
                x, y = body - [left, top]
                if not (0 <= x < 800 and 0 <= y < 800):
                    continue
                local_id = int(capture['ids'][j])
                accepted = bool(capture['accepted'][j])
                color = palette[local_id % len(palette)] if accepted else '#ff3030'
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
                label = f't{local_id}' if accepted else f'i{j}:REJECT'
                draw.text((x + 6, y - 12), label, fill=color)
                if np.isfinite(shoulder).all():
                    sx, sy = shoulder - [left, top]
                    draw.line((x, y, sx, sy), fill=color, width=1)
                    draw.rectangle((sx - 2, sy - 2, sx + 2, sy + 2), outline='white')
                rows.append({'frame': capture['frame'], 'source_instance_index': j,
                             'detection_id': capture['names'][j],
                             'tracklet_id': f'{DATASET_ID}/{video}/t{local_id:09d}' if accepted else None,
                             'incoming': capture['incoming'][j], 'rejection': capture['reasons'][j],
                             'water': capture['water'][j]})
            sheet.paste(image, (side * 800, 60))
            label = ImageDraw.Draw(sheet)
            label.text((side * 800 + 8, 6), f'{video} frame {capture["frame"]} | original crop x={left} y={top}', fill='white')
            label.text((side * 800 + 8, 24), 't# = NEW local tracklet ID; red = rejected; orange = FIXED water boundary', fill='white')
            label.text((side * 800 + 8, 42), selection, fill='white')
        name = f'{video}.tracks_{start:06d}_{start + 1:06d}.png'
        sheet.save(destination / name)
        records.append({'video_id': video, 'frames': [start, start + 1], 'selection': selection,
                        'original_crop_xyxy': [left, top, left + 800, top + 800],
                        'preview': name, 'preview_sha256': checksum(destination / name), 'observations_shown': rows})
    cap.release()
    return records


def build():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    job = os.environ['SLURM_JOB_ID']
    assert not (DEST / 'DATASET_COMPLETE.json').exists()
    assert not (DEST / 'tables').exists(), 'Refuse overwrite of previous tables'
    lock = DEST / '.conversion.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, job.encode())
    os.close(fd)
    work = DEST / f'.build_job_{job}'
    work.mkdir(exist_ok=False)
    (work / 'tables').mkdir()
    (work / 'qc').mkdir()
    try:
        config = yaml.safe_load((DEST / 'config/tracklets.yaml').read_text())
        decisions = json.loads((DEST / 'config/user_decisions_002.json').read_text())
        assert decisions['centroid_proxy_approved'] is True
        assert decisions['centroid_source'] == 'shell_proxy'
        assert config['dataset_id'] == DATASET_ID
        assert config['water_uncertain_boundary_band_px'] == decisions['uncertain_boundary_band_px'] == 0
        proposal_path = DEST / 'config/water_regions.json'
        accepted_roi_receipt = json.loads((DEST / 'config/user_decisions_001.json').read_text())
        assert checksum(proposal_path) == accepted_roi_receipt['water_polygon']['geometry_file_sha256']
        roi_proposal = json.loads(proposal_path.read_text())
        effective_rois = {'version': 'legacy_water_accepted_v002', 'temporal_geometry': 'fixed_in_all_frames',
                          'centroid_source': 'shell_proxy', 'boundary_included': True, 'uncertainty_band_px': 0,
                          'acceptance_receipts': ['config/user_decisions_001.json', 'config/user_decisions_002.json'],
                          'videos': {video: {**entry, 'confirmation_status': 'accepted_for_initial_use',
                                            'reference_point_source': 'shell_proxy', 'uncertain_boundary_band_px': 0}
                                     for video, entry in roi_proposal['videos'].items()}}
        dump(work / 'water_regions_effective.json', effective_rois)
        inventory_path = DEST / 'audit/inventory.json'
        marker = json.loads((DEST / 'audit/AUDIT_COMPLETE.json').read_text())
        assert checksum(inventory_path) == marker['inventory']['sha256']
        audit = json.loads(inventory_path.read_text())
        code = [ROOT / 'scripts/legacy_dataset' / name for name in ['tracklets.py', 'export.py', 'publish.py']]
        code += [ROOT / 'tests/legacy_dataset/test_tracklets.py', ROOT / 'jobs/build_legacy_dataset.sh', ROOT / 'jobs/publish_legacy_dataset.sh']
        config_paths = [DEST / 'config' / name for name in ['tracklets.yaml', 'water_regions.json', 'user_decisions_001.json', 'user_decisions_002.json']]
        input_pins = [pin(p) for p in code + config_paths + [inventory_path]]
        test_receipt = ROOT / 'checks/legacy_dataset' / f'tracklets_tests_{job}.json'
        tests = json.loads(test_receipt.read_text())
        assert tests['success'] and tests['failures'] == 0 and tests['errors'] == 0
        assert tests['tracklets_code_sha256'] == checksum(code[0])
        assert tests['export_code_sha256'] == checksum(code[1])
        assert tests['config_sha256'] == checksum(config_paths[0])
        source_pins = []
        # Reverify large raw videos and old-model artifacts against the audit once.
        for rec in audit['files']:
            if rec['path'].startswith('models/'):
                actual = pin(ROOT / rec['path'])
                assert actual['sha256'] == rec['sha256'], rec['path']
                source_pins.append(actual)
        track_writer = SingleFile(work / 'tables/tracklets.parquet', TRACK_SCHEMA)
        frame_writer = SingleFile(work / 'tables/frames.parquet', FRAME_SCHEMA)
        state = {}
        for video, entry in sorted(audit['videos'].items()):
            print(f'Building {video}', flush=True)
            assert entry['disposition'] == 'usable_as_is_for_posthoc_tracking'
            info = entry['video']
            actual = pin(ROOT / info['source']['path'])
            assert actual['sha256'] == info['source']['sha256']
            source_pins.append(actual)
            width, height = info['width'], info['height']
            tracker = Tracker(video, width, height, config)
            obs = Parts(work / 'tables/observations' / video, OBS_SCHEMA)
            rej = Parts(work / 'tables/rejected_observations' / video, OBS_SCHEMA)
            source_counts = np.zeros(info['container_frame_count'], dtype=np.int64)
            selected_frames = {x['frame'] for x in entry['preview_evidence']['frames']}
            captures = {}
            previous_capture = None
            expected_frame = 0
            linked_across_chunks = 0
            for c, rec in enumerate(entry['chunks']):
                actual = pin(ROOT / rec['path'])
                assert actual['sha256'] == rec['sha256'], rec['path']
                source_pins.append(actual)
                with h5py.File(ROOT / rec['path'], 'r') as f:
                    frames, instances, points = f['frames'][:], f['instances'][:], f['pred_points'][:]
                chunk_accepted, chunk_rejected, frame_rows = [], [], []
                for fr in frames:
                    index = int(fr['frame_idx'])
                    assert index == expected_frame, 'Coverage differs from audited complete inputs'
                    expected_frame += 1
                    start, end = int(fr['instance_id_start']), int(fr['instance_id_end'])
                    inst = instances[start:end]
                    n = len(inst)
                    source_counts[index] = n
                    if n:
                        p0, p1 = int(inst['point_id_start'][0]), int(inst['point_id_end'][-1])
                        pts = points[p0:p1].reshape(n, 5)
                    else:
                        pts = np.empty((0, 5), dtype=points.dtype)
                    frame = Frame(index, np.stack([pts['x'][:, 0], pts['y'][:, 0]], axis=1),
                                  np.stack([pts['x'][:, 1], pts['y'][:, 1]], axis=1),
                                  pts['score'][:, 0], pts['score'][:, 1], pts['visible'][:, 0], pts['visible'][:, 1])
                    result = tracker.process(frame)
                    track_writer.add(result['ended'])
                    accepted, rejected, capture = rows_for_frame(video, rec['path'], fr, inst, pts, tracker, result,
                                                                  effective_rois['videos'][video], info['container_fps'], previous_capture)
                    previous_capture = capture
                    chunk_accepted.append(accepted)
                    chunk_rejected.append(rejected)
                    if index in selected_frames:
                        captures[index] = capture
                    links = int(np.count_nonzero(result['predecessor_index'] >= 0))
                    if index > 0 and index % 40 == 0:
                        linked_across_chunks += links
                    frame_rows.append({'video_id': video, 'frame': index,
                                       'status': 'recorded_detections' if n else 'recorded_empty',
                                       'source_count': n, 'accepted_count': accepted.num_rows,
                                       'rejected_count': rejected.num_rows, 'linked_count': links})
                obs.add(pa.concat_tables(chunk_accepted))
                rej.add(pa.concat_tables(chunk_rejected))
                frame_writer.add(frame_rows)
                if c % 100 == 0:
                    print(f'  {c + 1} chunks; accepted={obs.total}; tracklets_started={tracker.next_id - 1}', flush=True)
            assert expected_frame == info['container_frame_count']
            assert int(source_counts.sum()) == entry['coverage']['chunk_instances']
            track_writer.add(tracker.finish())
            obs.close()
            rej.close()
            state[video] = {'ntracks': tracker.next_id - 1, 'source_counts': source_counts,
                            'chunk_boundary_links': linked_across_chunks,
                            'previews': render_review(video, info, captures, entry, work / 'qc', effective_rois['videos'][video])}
            print(f'{video}: streamed {obs.total} accepted, {rej.total} rejected', flush=True)
        track_writer.close()
        frame_writer.close()
        summary = {}
        review_examples = []
        for video, item in state.items():
            print(f'Read-back validation {video}', flush=True)
            summary[video] = validate_saved_video(work, video, item['ntracks'], item['source_counts'], config)
            summary[video]['chunk_boundary_links'] = item['chunk_boundary_links']
            review_examples.extend(item['previews'])
        saved_frames = pd.read_parquet(work / 'tables/frames.parquet')
        assert not saved_frames.duplicated(['video_id', 'frame']).any()
        for video, item in state.items():
            f = saved_frames.loc[saved_frames.video_id == video].sort_values('frame')
            assert np.array_equal(f['frame'].to_numpy(), np.arange(len(item['source_counts'])))
            assert np.array_equal(f['source_count'].to_numpy(), item['source_counts'])
            assert (f['accepted_count'] + f['rejected_count'] == f['source_count']).all()
            assert int(f['accepted_count'].sum()) == summary[video]['accepted_observations']
        # Detect source/code mutation since verification; do not combine revisions.
        for rec in input_pins + source_pins:
            stat = (ROOT / rec['path']).stat()
            assert stat.st_size == rec['bytes'] and stat.st_mtime_ns == rec['mtime_ns'], rec['path']
        dump(work / 'qc/summary.json', summary)
        dump(work / 'qc/review_examples.json', review_examples)
        dump(work / 'qc/tests.json', tests)
        versions = {name: metadata.version(name) for name in ['numpy', 'pandas', 'pyarrow', 'h5py', 'scipy', 'shapely', 'PyYAML', 'sleap-io', 'sleap-nn']}
        artifacts = [{'path': str(p.relative_to(work)), 'bytes': p.stat().st_size, 'sha256': checksum(p)}
                     for p in sorted(work.rglob('*')) if p.is_file()]
        manifest = {'dataset_id': DATASET_ID, 'algorithm_version': config['algorithm_version'],
                    'build_job_id': job, 'build_time_unix_s': time.time(), 'python': sys.version, 'versions': versions,
                    'input_pins': input_pins, 'source_pins': source_pins, 'artifacts': artifacts,
                    'input_audit_job': audit['job_id'], 'observation_schema': str(OBS_SCHEMA),
                    'raw_points_retention': 'All five original nodes/scores/visibility flags remain in SHA256-pinned source SLPs, linked by source_path/frame/source_instance_index/source_instance_row.',
                    'body': 'Centered-instance Shell landmark, not outline/size', 'centroid_source': 'shell_proxy',
                    'water_status': 'accepted_fixed_polygon_shell_point_membership_boundary_included_no_uncertainty_band',
                    'coordinate_system': 'Original-resolution pixels, x right and y down; no export rescaling',
                    'time_s': 'Nominal frame/container_fps; both audited FPS=30000/1001, not a full timestamp scan',
                    'input_completeness': 'Both videos have all audited frame records; unknown spans are never interpreted as empty',
                    'summaries': summary, 'derived_tracked_slp': False,
                    'limitations': ['Conservative tracklets are not persistent animal identities or a guarantee of zero switches.',
                                    'Missing Shoulder prevents linking but retains reliable body singleton observations.',
                                    'Rejected close duplicates are not a ground-truth duplicate classification.',
                                    'Scores are uncalibrated; only Shell and Shoulder drive validity/association.',
                                    'Water membership is not any-part overlap, segmentation, or a tracking exclusion.',
                                    'Existing model hashes pin present artifacts, not historical checkpoint identity.']}
        dump(work / 'manifest.json', manifest)
        atomic_json(work / 'BUILD_VALIDATED.json', {'build_job_id': job, 'manifest_sha256': checksum(work / 'manifest.json'),
                                                  'status': 'tests_and_readback_passed_waiting_bounded_visual_review',
                                                  'dataset_complete': False})
        print(f'VALIDATED STAGING: {work}', flush=True)
        print(json.dumps({v: {k: value for k, value in s.items() if k != 'length_histogram'} for v, s in summary.items()}, indent=2))
    finally:
        lock.unlink()


if __name__ == '__main__':
    build()
