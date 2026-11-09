#!/usr/bin/env python3
"""Add unchanged female detections, experimental IDs, and static burrows to demo.

Slurm-only entry point. No inference, video processing, source-label edits,
full-video retracking, Git operations, or cross-model identity inference.
"""
from __future__ import annotations
import gc
import json
import os
from pathlib import Path
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sleap_io as sio
import yaml

from body_first_tracklets import Frame, Tracker
from legacy_dataset.tracklets import water_membership
from pipeline_common import ROOT, atomic_json, sha256

DEMO = ROOT / 'demo/crab_cam2_2_middle_2000'
FEMALE_ROOT = ROOT / 'predictions/female_v002/Crab_Cam_2_2'
FEMALE = FEMALE_ROOT / 'raw/Crab_Cam_2_2.female.predictions.slp'
BURROWS = ROOT / 'data/labels/burrows/BurrowLabels3.slp'
BURROW_HASH = '1c8aa0bbda851459f470e49c2d1cc73cda93b4eb06e5bae083f5a848ef7f7903'
VIDEO_ID = 'Crab_Cam_2_2'
FPS = 30000 / 1001
FEMALE_TRACK_SCOPE = 'female_demo_body_first_v001'


def write_table(table, path):
    pq.write_table(table, path, compression='zstd')
    assert pq.read_table(path).equals(table), path


def slice_female(source, target, movie, start, count):
    """Copy native HDF5 bytes; modify only frame/instance/point index offsets."""
    with h5py.File(source, 'r') as src:
        f = src['frames'][start:start + count].copy()
        np.testing.assert_array_equal(f['frame_idx'], np.arange(start, start + count))
        np.testing.assert_array_equal(f['frame_id'], np.arange(start, start + count))
        lo, hi = int(f[0]['instance_id_start']), int(f[-1]['instance_id_end'])
        i = src['instances'][lo:hi].copy()
        p0 = int(i[0]['point_id_start']) if len(i) else 0
        p = src['pred_points'][p0:p0 + len(i)].copy()
        assert len(src['points']) == len(src['tracks_json']) == 0
        assert np.all(i['track'] == -1) and np.all(i['skeleton'] == 0)
        assert np.all(i['instance_type'] == 1) and np.all(i['from_predicted'] == -1)
        np.testing.assert_array_equal(i['point_id_start'], np.arange(len(i)) + p0)
        np.testing.assert_array_equal(i['point_id_end'], np.arange(len(i)) + p0 + 1)
        f['frame_id'] -= start
        f['frame_idx'] -= start
        f['instance_id_start'] -= lo
        f['instance_id_end'] -= lo
        i['instance_id'] -= lo
        i['frame_id'] -= start
        i['point_id_start'] -= p0
        i['point_id_end'] -= p0
        with h5py.File(target, 'x') as out:
            for key, value in src.attrs.items():
                out.attrs[key] = value
            for key in ('metadata', 'points'):
                src.copy(key, out)
            for key, array in [('frames', f), ('instances', i), ('pred_points', p)]:
                out.create_dataset(key, data=array, compression='gzip', shuffle=True)
            dt = h5py.string_dtype('utf-8')
            videos = [json.loads(v) for v in src['videos_json'][:]]
            assert len(videos) == 1
            videos[0]['filename'] = movie.name
            videos[0]['backend'].update(filename=movie.name, shape=[count, 4320, 7680, 3], fps=FPS)
            out.create_dataset('videos_json', data=[json.dumps(v) for v in videos], dtype=dt)
            for key in ('tracks_json', 'sessions_json', 'suggestions_json'):
                out.create_dataset(key, data=[], dtype=dt)
            out.create_dataset('provenance_json', data=np.bytes_(json.dumps({
                'source_slp': str(source), 'source_start': start, 'source_stop_exclusive': start + count,
                'coordinates_scores_flags_unchanged': True, 'tracking_rerun': False})))
    with h5py.File(target, 'r') as out:
        assert out['pred_points'][:].tobytes() == p.tobytes()
    labels = sio.load_slp(target, open_videos=False)
    assert labels.skeletons[0].node_names == ['FBody']
    assert [lf.frame_idx for lf in labels] == list(range(count))
    return f, i, p, lo


def track_female(raw, tracked, start, tracker):
    """Same body-first logic as male; FBody only, no invented Shoulder."""
    shutil.copy2(raw, tracked)
    diagnostics = []
    assigned_names = []
    ended = []
    with h5py.File(tracked, 'r+') as out:
        f, i, p = (out[key][:] for key in ('frames', 'instances', 'pred_points'))
        for row in f:
            local = int(row['frame_idx'])
            lo, hi = int(row['instance_id_start']), int(row['instance_id_end'])
            xy = np.column_stack([p['x'][lo:hi], p['y'][lo:hi]])
            frame = Frame(start + local, xy, np.full_like(xy, np.nan), p['score'][lo:hi],
                          np.zeros(hi-lo), p['visible'][lo:hi], np.zeros(hi-lo, bool))
            result = tracker.process(frame)
            ended.extend(result['ended'])
            for j in range(hi-lo):
                local_id = int(result['ids'][j])
                name = tracker.name(local_id) if local_id > 0 else None
                assigned_names.append(name)
                diagnostics.append({'tracklet_id': name, 'tracklet_local_id': local_id if local_id > 0 else None,
                    'accepted_for_tracking': bool(result['validity']['accepted'][j]),
                    'rejection_reason': str(result['validity']['reasons'][j]) or None,
                    'incoming_reason': str(result['incoming_reason'][j]),
                    'predecessor_instance_index': int(result['predecessor_index'][j]) if result['predecessor_index'][j] >= 0 else None})
        names = sorted({name for name in assigned_names if name is not None})
        lookup = {name: k for k, name in enumerate(names)}
        i['track'] = [lookup.get(name, -1) for name in assigned_names]
        out['instances'][:] = i
        del out['tracks_json']
        # The SLP spawned_on field is a source-frame index, as for retained male IDs.
        starts = {}
        for row, name in zip(i, assigned_names):
            if name is not None:
                starts.setdefault(name, start + int(row['frame_id']))
        out.create_dataset('tracks_json', data=[json.dumps([starts[name], name]) for name in names],
                           dtype=h5py.string_dtype('utf-8'))
        del out['provenance_json']
        out.create_dataset('provenance_json', data=np.bytes_(json.dumps({
            'source_raw_demo_slp': raw.name, 'algorithm': 'body_first_v001_FBody_only',
            'tracking_scope': FEMALE_TRACK_SCOPE, 'parameters': tracker.config,
            'no_gap_bridging': True, 'experimental_not_identity_validated': True})))
    with h5py.File(raw, 'r') as a, h5py.File(tracked, 'r') as b:
        assert a['pred_points'][:].tobytes() == b['pred_points'][:].tobytes()
        aa, bb = a['instances'][:], b['instances'][:]
        for field in aa.dtype.names:
            if field != 'track':
                assert aa[field].tobytes() == bb[field].tobytes()
    return diagnostics, ended


def burrow_reference(source, reference_frame):
    labels = sio.load_slp(source, open_videos=False)
    assert len(labels.videos) == len(labels.skeletons) == 1
    assert labels.skeletons[0].node_names == ['Burrow']
    lf = next(lf for lf in labels if lf.frame_idx == reference_frame)
    instances = lf.user_instances
    assert instances and not lf.predicted_instances
    xy = np.array([i.points['xy'][0] for i in instances])
    assert np.isfinite(xy).all() and len(np.unique(xy, axis=0)) == len(xy)
    assert ((xy >= 0) & (xy < [7680, 4320])).all()
    return instances, labels.videos[0].filename


def display_instance(source, skeleton, track=None):
    """Display-only union, with every non-applicable node NaN and invisible."""
    predicted = isinstance(source, sio.PredictedInstance)
    cls = sio.PredictedInstance if predicted else sio.Instance
    kwargs = {'score': source.score} if predicted else {}
    result = cls.from_numpy(np.full((len(skeleton.nodes), 2), np.nan), skeleton=skeleton,
                            track=source.track if track is None else track,
                            tracking_score=source.tracking_score, **kwargs)
    for src_index, name in enumerate(source.skeleton.node_names):
        dst_index = skeleton.node_names.index(name)
        for field in ('xy', 'visible', 'complete') + (('score',) if predicted else ()):
            result.points[field][dst_index] = source.points[field][src_index]
    return result


def assert_group(sources, actuals, track_names=None):
    """Vectorized per-frame validation, including invisible raw coordinates."""
    assert len(sources) == len(actuals)
    if not sources:
        return
    predicted = isinstance(sources[0], sio.PredictedInstance)
    assert all(isinstance(i, sio.PredictedInstance) == predicted for i in sources + actuals)
    names = sources[0].skeleton.node_names
    dst_names = actuals[0].skeleton.node_names
    selected = [dst_names.index(name) for name in names]
    for field in ('xy', 'visible', 'complete') + (('score',) if predicted else ()):
        expected = np.asarray([i.points[field] for i in sources])
        actual = np.asarray([i.points[field][selected] for i in actuals])
        np.testing.assert_equal(expected, actual)
    absent = [k for k, name in enumerate(dst_names) if name not in names]
    assert all(np.isnan(i.points['xy'][absent]).all() and not i.points['visible'][absent].any() for i in actuals)
    if predicted:
        for field in ('score', 'tracking_score'):
            expected = np.asarray([np.nan if getattr(i, field) is None else getattr(i, field) for i in sources])
            actual = np.asarray([np.nan if getattr(i, field) is None else getattr(i, field) for i in actuals])
            np.testing.assert_equal(expected, actual)
    expected_names = track_names if track_names is not None else [i.track.name if i.track else None for i in sources]
    assert [i.track.name if i.track else None for i in actuals] == expected_names


def combine_chunk(male_path, female_path, burrow_instances, reference_frame, combined_path, burrow_path):
    male, female = (sio.load_slp(p, open_videos=False) for p in (male_path, female_path))
    assert len(male) == len(female) and len(male.videos) == len(female.videos) == 1
    assert male.skeletons[0].node_names == ['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm']
    assert female.skeletons[0].node_names == ['FBody']
    skeleton = sio.Skeleton(nodes=male.skeletons[0].node_names + ['FBody', 'Burrow'],
                            edges=male.skeletons[0].edge_names, name='demo_display_only_union')
    video = male.videos[0]
    female.videos[0] = video
    burrow_tracks = [sio.Track(name=f'burrow_static/f{reference_frame:06d}/b{k:04d}')
                     for k in range(len(burrow_instances))]
    all_names = [t.name for t in male.tracks + female.tracks + burrow_tracks]
    assert len(all_names) == len(set(all_names)), 'Entity track namespaces collide'
    combined_frames, burrow_frames = [], []
    mapping = {'male': [], 'female': []}
    absolute_row = 0
    for local, (m, f) in enumerate(zip(male, female)):
        assert m.frame_idx == f.frame_idx == local
        merged = []
        for kind, lf in [('male', m), ('female', f)]:
            for source in lf.instances:
                mapping[kind].append((len(merged), absolute_row + len(merged)))
                merged.append(display_instance(source, skeleton))
        statics = [display_instance(source, skeleton, track)
                   for source, track in zip(burrow_instances, burrow_tracks)]
        merged.extend(statics)
        combined_frames.append(sio.LabeledFrame(video, local, merged))
        native_statics = [sio.Instance(points=source.points.copy(), skeleton=source.skeleton, track=track)
                          for source, track in zip(burrow_instances, burrow_tracks)]
        burrow_frames.append(sio.LabeledFrame(video, local, native_statics))
        absolute_row += len(merged)
    provenance = {'display_only': True, 'not_for_training': True,
        'male_and_female_predictions_unchanged': True, 'burrows': 'static reference positions, not per-frame annotations',
        'burrow_reference_frame': reference_frame, 'unused_union_nodes': 'NaN and invisible; not inferred landmarks'}
    sio.save_slp(sio.Labels(combined_frames, videos=[video], skeletons=[skeleton],
                           tracks=male.tracks + female.tracks + burrow_tracks, provenance=provenance), combined_path)
    sio.save_slp(sio.Labels(burrow_frames, videos=[video], skeletons=[burrow_instances[0].skeleton],
                           tracks=burrow_tracks, provenance=provenance), burrow_path)
    # The current sleap-io writer emits absolute media paths even when the
    # media is adjacent. Relink metadata explicitly, without touching poses.
    for path in (combined_path, burrow_path):
        with h5py.File(path, 'r+') as out:
            videos = [json.loads(v) for v in out['videos_json'][:]]
            assert len(videos) == 1
            basename = Path(video.filename).name
            videos[0]['filename'] = basename
            videos[0]['backend']['filename'] = basename
            del out['videos_json']
            out.create_dataset('videos_json', data=[json.dumps(v) for v in videos],
                               dtype=h5py.string_dtype('utf-8'))
    # Read back EVERY instance/point/score/flag/ID, not just a preview sample.
    restored = sio.load_slp(combined_path, open_videos=False)
    static_restored = sio.load_slp(burrow_path, open_videos=False)
    assert len(restored.skeletons) == 1
    assert restored.skeletons[0].edge_names == male.skeletons[0].edge_names
    for local, (m, f, out, bs) in enumerate(zip(male, female, restored, static_restored)):
        assert out.frame_idx == bs.frame_idx == local
        moving = m.instances + f.instances
        assert len(out.instances) == len(moving) + len(burrow_instances)
        assert len(bs.instances) == len(burrow_instances)
        assert_group(m.instances, out.instances[:len(m.instances)])
        assert_group(f.instances, out.instances[len(m.instances):len(moving)])
        names = [track.name for track in burrow_tracks]
        assert_group(burrow_instances, out.instances[len(moving):], names)
        assert_group(burrow_instances, bs.instances, names)
        assigned = [i.track.name for i in out.instances if i.track]
        assert len(assigned) == len(set(assigned))
    return mapping


def add_columns(table, values):
    for name, value in values.items():
        assert name not in table.column_names, name
        table = table.append_column(name, pa.array(value))
    return table


def female_table(old, chunk, f, instances, points, source_offset, diagnostics, polygon, selection_start):
    """Populate shared columns honestly; male-only anatomy/provenance stays null."""
    rows = []
    k = 0
    for frame_row in f:
        local = int(frame_row['frame_idx'])
        frame = chunk['source_start'] + local
        lo, hi = int(frame_row['instance_id_start']), int(frame_row['instance_id_end'])
        water = water_membership(np.column_stack([points['x'][lo:hi], points['y'][lo:hi]]),
                                 7680, 4320, polygon)
        for j, row_id in enumerate(range(lo, hi)):
            p, inst = points[row_id], instances[row_id]
            source_id = source_offset + row_id
            identity = f'{VIDEO_ID}/f{frame:06d}/i{j:04d}'
            rows.append(dict(diagnostics[k], video_id=VIDEO_ID, frame=frame, instance_index=j,
                raw_instance_id=source_id, detection_id=f'female_v002_demo/{identity}',
                prediction_detection_id=identity, dataset_id='female_v002_demo', prediction_run_id='female_v002',
                tracking_algorithm='body_first_v001_FBody_only', identity_status='experimental_not_ground_truth_validated',
                instance_score=float(inst['score']), source_tracking_score=float(inst['tracking_score']),
                body_x=float(p['x']), body_y=float(p['y']), body_score=float(p['score']),
                centroid_x=float(p['x']), centroid_y=float(p['y']), centroid_source='FBody_proxy',
                water=water[j], time_s=frame / FPS,
                source_path=str(FEMALE.relative_to(ROOT)), source_instance_row=source_id, source_instance_index=j,
                source_chunk_path=None, source_chunk_instance_row=None, source_observations_path=None,
                demo_frame=frame-selection_start, demo_time_s=(frame-selection_start)/FPS,
                chunk_frame=local, chunk_time_s=local/FPS, chunk_id=chunk['chunk_id'],
                chunk_instance_row=row_id, demo_instance_row=None,
                fbody_x=float(p['x']), fbody_y=float(p['y']), fbody_score=float(p['score']),
                fbody_visible=bool(p['visible'])))
            k += 1
    assert k == len(points) == len(diagnostics)
    extra = pa.schema([('fbody_x', pa.float64()), ('fbody_y', pa.float64()), ('fbody_score', pa.float64()),
                       ('fbody_visible', pa.bool_()), ('predecessor_instance_index', pa.int64())])
    table = pa.Table.from_pylist(rows, schema=pa.schema(list(old.schema) + list(extra)))
    for field in table.schema:
        if pa.types.is_floating(field.type):
            a = table[field.name]
            table = table.set_column(table.schema.get_field_index(field.name), field.name,
                                     pc.if_else(pc.is_nan(a), pa.scalar(None, field.type), a))
    assert all(table[f'{node}_x'].null_count == len(table) for node in ('shell', 'shoulder', 'elbow', 'claw', 'arm'))
    np.testing.assert_equal(table['fbody_x'].to_numpy(), points['x'])
    np.testing.assert_equal(table['fbody_y'].to_numpy(), points['y'])
    np.testing.assert_equal(table['fbody_score'].to_numpy(), points['score'])
    return table


def layer_columns(table, kind, chunk, slp_path, combined_path, mapping):
    n = len(table)
    # Existing demo_instance_row retains its original male-only meaning; new
    # observation_row is the authoritative row in the augmented root table.
    return add_columns(table, {'entity_type': [kind] * n, 'sex': [kind] * n,
        'observation_kind': ['model_prediction'] * n,
        'tracking_scope': [('original_full_video' if kind == 'male' else 'demo_2000_frames_only')] * n,
        'layer_slp': [slp_path] * n, 'combined_slp': [combined_path] * n,
        'combined_instance_index': pa.array([v[0] for v in mapping], type=pa.int64()),
        'combined_chunk_instance_row': pa.array([v[1] for v in mapping], type=pa.int64())})


def check_portable_slp(path, movie, count, dimensions_hw=(4320, 7680)):
    labels = sio.load_slp(path, open_videos=True, lazy=True)
    assert Path(labels.videos[0].filename).resolve() == movie.resolve()
    assert labels.videos[0][0].shape[:2] == dimensions_hw
    assert labels.videos[0][count-1].shape[:2] == dimensions_hw
    labels.videos[0].close()


def augment(stage, old_demo, chunks, female_source, burrow_source, config, polygon, reference_frame=563):
    """Build only in a new stage; original demo/source inputs stay read-only."""
    stage.mkdir(exist_ok=False)
    (stage / 'provenance').mkdir()
    for name in ('DEMO_COMPLETE.json', 'README.md', 'chunks.json'):
        shutil.copy2(old_demo / name, stage / 'provenance' / ('MALE_' + name))
    os.link(old_demo / 'frame_map.csv', stage / 'frame_map.csv')
    (stage / 'burrows').mkdir()
    shutil.copy2(burrow_source, stage / 'burrows' / burrow_source.name)
    assert sha256(burrow_source) == sha256(stage / 'burrows' / burrow_source.name)
    burrow_instances, burrow_video = burrow_reference(burrow_source, reference_frame)
    assert Path(burrow_video).name == 'clip_cam2_8k.mp4'
    xy = np.array([i.points['xy'][0] for i in burrow_instances])
    statics = pa.Table.from_pylist([dict(burrow_id=f'burrow_static/f{reference_frame:06d}/b{k:04d}',
        entity_type='burrow', observation_kind='static_reference_assumed', video_id=VIDEO_ID,
        burrow_x=float(point[0]), burrow_y=float(point[1]), source_label_frame=reference_frame,
        source_instance_index=k, source_label_video=Path(burrow_video).name,
        source_path=f'burrows/{burrow_source.name}', source_sha256=sha256(burrow_source),
        valid_source_frame_start=chunks['source_frame_start'],
        valid_source_frame_stop_exclusive=chunks['source_frame_stop_exclusive'],
        geometry_transfer='assumed same fixed camera geometry; not independently registered') for k, point in enumerate(xy)])
    write_table(statics, stage / 'burrows.parquet')
    tracker = Tracker(VIDEO_ID, 7680, 4320, config)
    tables, frame_tables, track_rows, chunk_rows = [], [], [], []
    observation_count = 0
    cross_chunk_links = 0
    female_count = assigned_count = 0
    for chunk_number, chunk in enumerate(chunks['chunks']):
        cid = chunk['chunk_id']
        directory = stage / cid
        directory.mkdir()
        for key in ('video', 'raw_slp', 'tracked_slp'):
            os.link(old_demo / chunk[key], stage / chunk[key])
        movie = stage / chunk['video']
        stem = movie.stem
        paths = {key: f'{cid}/{stem}.{suffix}.slp' for key, suffix in [
            ('female_raw_slp', 'female.predictions'), ('female_tracked_slp', 'female.tracked'),
            ('combined_slp', 'combined.tracked'), ('burrow_slp', 'burrows')]}
        f, i, p, offset = slice_female(female_source, stage / paths['female_raw_slp'], movie,
                                      chunk['source_start'], chunk['frames'])
        diagnostics, ended = track_female(stage / paths['female_raw_slp'], stage / paths['female_tracked_slp'],
                                           chunk['source_start'], tracker)
        track_rows.extend(ended)
        if chunk_number:
            first_count = int(f[0]['instance_id_end']) - int(f[0]['instance_id_start'])
            cross_chunk_links += sum(d['incoming_reason'] == 'linked' for d in diagnostics[:first_count])
        mapping = combine_chunk(stage / chunk['tracked_slp'], stage / paths['female_tracked_slp'],
            burrow_instances, reference_frame, stage / paths['combined_slp'], stage / paths['burrow_slp'])
        old = pq.read_table(old_demo / cid / 'observations.parquet')
        female = female_table(old, chunk, f, i, p, offset, diagnostics, polygon, chunks['source_frame_start'])
        male = layer_columns(old, 'male', chunk, chunk['tracked_slp'], paths['combined_slp'], mapping['male'])
        female = layer_columns(female, 'female', chunk, paths['female_tracked_slp'], paths['combined_slp'], mapping['female'])
        table = pa.concat_tables([male, female], promote_options='default')
        # Stable frame order; existing male columns/row order preserved when filtering sex.
        table = table.take(pc.sort_indices(table, sort_keys=[('chunk_frame', 'ascending'), ('entity_type', 'descending'), ('instance_index', 'ascending')]))
        table = table.append_column('observation_row', pa.array(np.arange(len(table)) + observation_count))
        assert table.filter(pc.equal(table['entity_type'], 'male')).select(old.column_names).equals(old)
        assert len(set(table['detection_id'].to_pylist())) == len(table)
        frame_table = pq.read_table(old_demo / cid / 'frames.parquet')
        female_counts = f['instance_id_end'].astype(np.int64) - f['instance_id_start'].astype(np.int64)
        assigned = np.bincount(i['frame_id'][np.array([d['accepted_for_tracking'] for d in diagnostics], bool)].astype(int), minlength=chunk['frames'])
        frame_table = add_columns(frame_table, {'male_source_count': frame_table['source_count'],
            'female_source_count': female_counts, 'female_assigned_count': assigned,
            'all_crab_source_count': frame_table['source_count'].to_numpy() + female_counts,
            'static_burrow_count': [len(burrow_instances)] * len(frame_table),
            'combined_slp': [paths['combined_slp']] * len(frame_table)})
        write_table(table, directory / 'observations.parquet')
        write_table(frame_table, directory / 'frames.parquet')
        # Independent relocated loading: only basename movie references are allowed.
        for path in paths.values():
            with h5py.File(stage / path, 'r') as file:
                videos = [json.loads(v) for v in file['videos_json'][:]]
                assert len(videos) == 1
                assert Path(videos[0]['backend']['filename']).name == movie.name
                assert not Path(videos[0]['backend']['filename']).is_absolute()
        for key in ('combined_slp', 'female_tracked_slp', 'burrow_slp'):
            check_portable_slp(stage / paths[key], movie, chunk['frames'])
        tables.append(table)
        frame_tables.append(frame_table)
        observation_count += len(table)
        female_count += len(female)
        assigned_count += int(assigned.sum())
        chunk_rows.append(dict(chunk, **paths, male_observations=len(old), female_observations=len(female),
                               female_assigned_observations=int(assigned.sum()), static_burrows=len(burrow_instances)))
        print('CHUNK_AUGMENTED', cid, 'male', len(old), 'female', len(female), 'burrows/frame', len(burrow_instances), flush=True)
        gc.collect()
    track_rows.extend(tracker.finish())
    write_table(pa.Table.from_pylist(track_rows), stage / 'female_tracklets.parquet')
    table = pa.concat_tables(tables)
    frames = pa.concat_tables(frame_tables)
    np.testing.assert_array_equal(frames['demo_frame'].to_numpy(), np.arange(chunks['frames']))
    original_male = pq.read_table(old_demo / 'observations.parquet')
    assert table.filter(pc.equal(table['entity_type'], 'male')).select(original_male.column_names).equals(original_male)
    original_frames = pq.read_table(old_demo / 'frames.parquet')
    assert frames.select(original_frames.column_names).equals(original_frames)
    assert len(set(table['detection_id'].to_pylist())) == len(table)
    # IDs are never duplicated within a frame; all female tracklets are adjacent-frame only.
    assigned = table.filter(pc.and_(pc.equal(table['entity_type'], 'female'), pc.is_valid(table['tracklet_id'])))
    ids = assigned['tracklet_id'].to_pylist()
    times = assigned['frame'].to_numpy()
    by_id = {}
    for name, frame in zip(ids, times):
        by_id.setdefault(name, []).append(int(frame))
    assert len(by_id) == len(track_rows)
    for track in track_rows:
        indices = sorted(by_id[track['tracklet_id']])
        assert indices == list(range(track['start_frame'], track['end_frame'] + 1))
        assert len(indices) == track['length']
    write_table(table, stage / 'observations.parquet')
    write_table(frames, stage / 'frames.parquet')
    atomic_json(stage / 'chunks.json', dict(chunks, chunks=chunk_rows,
        crab_observations='male + female; static burrows are a separate relational table',
        burrows='burrows.parquet', female_tracklets='female_tracklets.parquet'))
    return dict(male_observations=len(original_male), female_observations=female_count,
        female_assigned_observations=assigned_count, female_tracklets=len(track_rows),
        female_links_across_chunk_boundaries=cross_chunk_links, observations=len(table),
        static_burrows=len(burrow_instances), burrow_reference_frame=reference_frame,
        repeated_static_burrow_gui_instances=len(burrow_instances) * chunks['frames'])


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    assert not (DEMO / 'AUGMENTATION_COMPLETE.json').exists(), 'Already augmented; no automatic replacement'
    job = os.environ['SLURM_JOB_ID']
    checks = ROOT / f'checks/demo_female_burrows_{job}'
    checks.mkdir(exist_ok=False)
    original = json.loads((DEMO / 'DEMO_COMPLETE.json').read_text())
    chunks = json.loads((DEMO / 'chunks.json').read_text())
    assert original['status'] == 'validated_complete' and original['frames'] == chunks['frames'] == 2000
    assert (chunks['source_frame_start'], chunks['source_frame_stop_exclusive']) == (13410, 15410)
    # Validate every current payload, including the videos and all original male files.
    old_hashes = {}
    for rel, pin in original['artifacts'].items():
        assert sha256(DEMO / rel) == pin['sha256'], rel
        old_hashes[rel] = pin['sha256']
    female_receipt = json.loads((FEMALE_ROOT / 'PREDICT_COMPLETE.json').read_text())
    raw_receipt = json.loads((FEMALE_ROOT / 'raw/RAW_COMPLETE.json').read_text())
    assert female_receipt['tracking_enabled'] is False
    assert female_receipt['raw']['frames'] == raw_receipt['frames'] == 28800
    assert female_receipt['raw']['nodes'] == ['FBody']
    assert sha256(FEMALE) == female_receipt['raw']['sha256'] == raw_receipt['sha256']
    assert sha256(BURROWS) == BURROW_HASH
    protected = [FEMALE, BURROWS, FEMALE_ROOT / 'PREDICT_COMPLETE.json', FEMALE_ROOT / 'raw/RAW_COMPLETE.json',
        ROOT / 'configs/tracking/body_first_v001.yaml',
        ROOT / 'datasets/v007_best_body_first_dataframe_v001/config/water_regions.json']
    inputs = {str(p.relative_to(ROOT)): sha256(p) for p in protected}
    config = yaml.safe_load(protected[4].read_text())
    config.pop('pilot', None)
    config.update(dataset_id=FEMALE_TRACK_SCOPE, body_node='FBody', shoulder_node=None,
                  shoulder_vector_weight=0.0)
    mask = json.loads(protected[5].read_text())
    polygon = mask['videos'][VIDEO_ID]['polygon_xy']
    stage = DEMO.parent / f'.build_female_burrows_{job}'
    counts = augment(stage, DEMO, chunks, FEMALE, BURROWS, config, polygon)
    (stage / 'README.md').write_text((ROOT / 'docs/demo/README_combined.template.md').read_text())
    for rel in old_hashes:
        if rel.endswith(('.mp4', '.slp')) or rel == 'frame_map.csv':
            assert sha256(stage / rel) == old_hashes[rel], rel
    assert inputs == {str(p.relative_to(ROOT)): sha256(p) for p in protected}, 'Source changed'
    # Originals retained intact. This is an archive of the old demo, NOT a move
    # of the uploaded burrow file, which was explicitly copied above.
    archive = ROOT / f'archive/demo/crab_cam2_2_middle_2000_male_only_{job}'
    archive.parent.mkdir(parents=True, exist_ok=True)
    assert not archive.exists()
    code = ['scripts/augment_crab_demo.py', 'tests/test_augment_crab_demo.py',
            'jobs/augment_crab_demo.sh', 'docs/demo/README_combined.template.md']
    receipt = dict(status='validated_complete', job_id=job, frames=2000,
        source_frame_start=13410, source_frame_stop_exclusive=15410, width=7680, height=4320,
        fps_numerator=30000, fps_denominator=1001, chunk_count=10, **counts,
        source_inputs=inputs, female_tracking_parameters=config,
        male_original_receipt='provenance/MALE_DEMO_COMPLETE.json',
        previous_demo_archive=str(archive.relative_to(ROOT)),
        code_sha256={rel: sha256(ROOT / rel) for rel in code},
        tests=json.loads((ROOT / f'checks/demo_augmentation_tests_{job}.json').read_text()),
        validation=dict(all_original_male_pose_score_flag_and_id_values_preserved=True,
            original_male_slp_and_video_files_byte_identical=True, frame_map_unchanged=True,
            female_raw_and_tracked_point_bytes_identical=True, all_combined_instance_values_read_back=True,
            non_applicable_landmarks_nan_invisible=True, female_ids_unique_per_frame=True,
            female_no_gaps_or_interpolation=True, source_burrows_copied_not_moved=True,
            portable_basename_video_links=True, no_inference_or_video_reencoding=True,
            desktop_gui_tested=False),
        limitations=['Model class is not independently verified biological sex; no cross-model deduplication or identity matching.',
            'Female IDs are demo-only experimental body-first tracklets, not validated animals or original full-video IDs.',
            'Burrow map uses frame 563, not a union/average with frame 948. Constant positions and transfer from clip_cam2_8k are assumed, not independently registered.',
            'Combined seven-node skeleton is display-only, never training data. Non-applicable landmarks remain missing.',
            'Desktop SLEAP GUI not exercised; SLP data readback and relocated video loading tested.',
            'Local demo update only; no Git commit or GitHub push requested or performed.'])
    assert receipt['tests']['passed'] is True
    atomic_json(stage / 'AUGMENTATION_COMPLETE.json', receipt)
    artifacts = {str(p.relative_to(stage)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                 for p in stage.rglob('*') if p.is_file()}
    assert all(pin['bytes'] < 100 * 1024**2 for pin in artifacts.values()), 'Payload exceeds 100 MiB'
    atomic_json(stage / 'DEMO_COMPLETE.json', dict(receipt, artifacts=artifacts))
    # Check the still-live original immediately before the two-rename publication.
    for rel, digest in old_hashes.items():
        assert sha256(DEMO / rel) == digest
    DEMO.rename(archive)
    try:
        stage.rename(DEMO)
    except BaseException:
        archive.rename(DEMO)
        raise
    atomic_json(checks / 'PUBLISHED.json', dict(status='validated_complete', demo=str(DEMO.relative_to(ROOT)),
        archive=str(archive.relative_to(ROOT)), receipt_sha256=sha256(DEMO / 'DEMO_COMPLETE.json'), **counts))
    print('PUBLISHED', json.dumps(counts, indent=2), flush=True)


if __name__ == '__main__':
    main()
