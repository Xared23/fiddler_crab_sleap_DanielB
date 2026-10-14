"""Bounded-memory SLP finalization; raw poses and experimental IDs stay separate.

Restricted to the pinned, single-video, five-node, untracked prediction format.
Unexpected schemas fail closed. No raw observation is deleted by tracking.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sleap_io as sio

from body_first_tracklets import Frame, Tracker
from legacy_dataset.tracklets import water_membership
from pipeline_common import sha256, atomic_json

NODES = ['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm']
TABLES = ('frames', 'instances', 'pred_points')
ALLOWED = set(TABLES) | {'points', 'metadata', 'videos_json', 'tracks_json',
                         'suggestions_json', 'sessions_json', 'provenance_json'}


def inspect_chunk(path, start, stop, video=None):
    """Read back every point/index plus SLEAP metadata, including empty frames."""
    labels = sio.load_slp(path, open_videos=False)
    assert len(labels.videos) == 1
    assert len(labels.skeletons) == 1 and labels.skeletons[0].node_names == NODES
    if video is not None:
        assert Path(labels.videos[0].filename).resolve() == Path(video).resolve()
    assert [lf.frame_idx for lf in labels] == list(range(start, stop))
    assert not labels.tracks
    for lf in labels:
        assert all(isinstance(i, sio.PredictedInstance) and i.track is None for i in lf)
    with h5py.File(path, 'r') as f:
        assert set(f.keys()) <= ALLOWED, list(f.keys())
        assert len(f['points']) == 0 and len(f['tracks_json']) == 0
        frames, instances, points = (f[k][:] for k in TABLES)
    np.testing.assert_array_equal(frames['frame_idx'], np.arange(start, stop))
    np.testing.assert_array_equal(frames['frame_id'], np.arange(stop-start))
    assert np.all(frames['video'] == 0)
    n = len(instances)
    np.testing.assert_array_equal(instances['instance_id'], np.arange(n))
    assert np.all(instances['instance_type'] == 1) and np.all(instances['skeleton'] == 0)
    assert np.all(instances['track'] == -1) and np.all(instances['from_predicted'] == -1)
    np.testing.assert_array_equal(instances['point_id_start'], np.arange(n)*5)
    np.testing.assert_array_equal(instances['point_id_end'], (np.arange(n)+1)*5)
    assert len(points) == 5*n
    end = 0
    for row in frames:
        lo, hi = int(row['instance_id_start']), int(row['instance_id_end'])
        assert lo == end and lo <= hi <= n
        assert np.all(instances['frame_id'][lo:hi] == row['frame_id'])
        end = hi
    assert end == n
    return frames, instances, points


def append_table(file, key, array):
    ds = file[key]
    old = len(ds)
    ds.resize((old + len(array),))
    ds[old:] = array


def readback_slp(path, total_frames, expected_tracks=None):
    """All frame metadata read; representative frames materialized by SLEAP."""
    labels = sio.load_slp(path, open_videos=False, lazy=True)
    assert len(labels) == total_frames
    assert labels.skeletons[0].node_names == NODES
    if expected_tracks is not None:
        assert len(labels.tracks) == expected_tracks
    for index in sorted(set([0, min(39, total_frames-1), min(40, total_frames-1), total_frames-1])):
        assert labels[index].frame_idx == index
    return len(labels.tracks)


def merge_raw(chunks, target, total_frames, video):
    """Append native SLP tables with remapped offsets, never all Python instances.

    The raw merged file contains exactly the same point/score bytes as the chunks.
    Validation runs before atomic publication. Existing finals are never replaced.
    """
    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    tmp = target.with_suffix('.tmp.slp')
    if tmp.exists():
        raise FileExistsError(f'Incomplete prior finalization retained: {tmp}')
    next_frame = count_instances = count_points = 0
    with h5py.File(tmp, 'x') as out:
        for path, start, stop in chunks:
            assert start == next_frame
            frames, instances, points = inspect_chunk(path, start, stop, video)
            with h5py.File(path, 'r') as source:
                if start == 0:
                    for key in source:
                        if key not in TABLES:
                            source.copy(key, out)
                    for key, value in source.attrs.items():
                        out.attrs[key] = value
                    for key, data in zip(TABLES, (frames, instances, points)):
                        out.create_dataset(key, shape=(0,), maxshape=(None,), dtype=data.dtype, chunks=True)
                else:
                    np.testing.assert_array_equal(source['videos_json'][:], out['videos_json'][:])
            frames['frame_id'] += next_frame
            frames['instance_id_start'] += count_instances
            frames['instance_id_end'] += count_instances
            instances['instance_id'] += count_instances
            instances['frame_id'] += next_frame
            instances['point_id_start'] += count_points
            instances['point_id_end'] += count_points
            for key, data in zip(TABLES, (frames, instances, points)):
                append_table(out, key, data)
            next_frame = stop
            count_instances += len(instances)
            count_points += len(points)
        assert next_frame == total_frames
        first_provenance = (json.loads(out['provenance_json'][()])
                            if 'provenance_json' in out else
                            json.loads(out['metadata'].attrs['json']).get('provenance', {}))
        provenance = {'first_chunk_provenance': first_provenance,
                      'raw_merge': {'frames': total_frames,
                                    'chunks': [str(p) for p, _, _ in chunks],
                                    'tracking_enabled': False}}
        if 'provenance_json' in out:
            del out['provenance_json']
        out.create_dataset('provenance_json', data=np.bytes_(json.dumps(provenance)))
        out.flush()
    # Full byte-for-byte chunk/table comparison; includes NaN representations.
    ioff = poff = 0
    with h5py.File(tmp, 'r') as merged:
        for path, start, stop in chunks:
            f, i, p = inspect_chunk(path, start, stop, video)
            f['frame_id'] += start
            f['instance_id_start'] += ioff
            f['instance_id_end'] += ioff
            i['instance_id'] += ioff
            i['frame_id'] += start
            i['point_id_start'] += poff
            i['point_id_end'] += poff
            assert merged['frames'][start:stop].tobytes() == f.tobytes()
            assert merged['instances'][ioff:ioff+len(i)].tobytes() == i.tobytes()
            assert merged['pred_points'][poff:poff+len(p)].tobytes() == p.tobytes()
            ioff += len(i)
            poff += len(p)
    readback_slp(tmp, total_frames, 0)
    tmp.rename(target)
    return {'frames': total_frames, 'instances': count_instances, 'sha256': sha256(target)}


OBS_SCHEMA = pa.schema([
    ('video_id', pa.string()), ('frame', pa.int64()), ('instance_index', pa.int32()),
    ('raw_instance_id', pa.int64()), ('detection_id', pa.string()),
    ('tracklet_id', pa.string()), ('tracklet_local_id', pa.int64()),
    ('accepted_for_tracking', pa.bool_()), ('rejection_reason', pa.string()),
    ('incoming_reason', pa.string()), ('instance_score', pa.float64()),
    ('body_x', pa.float64()), ('body_y', pa.float64()), ('body_score', pa.float64()),
    ('centroid_source', pa.string()), ('water', pa.bool_()),
] + [(f'{node.lower()}_{suffix}', kind) for node in NODES
     for suffix, kind in [('x', pa.float64()), ('y', pa.float64()), ('score', pa.float64()), ('visible', pa.bool_())]])
TRACK_SCHEMA = pa.schema([
    ('tracklet_id', pa.string()), ('tracklet_local_id', pa.int64()), ('video_id', pa.string()),
    ('start_frame', pa.int64()), ('end_frame', pa.int64()), ('length', pa.int64()),
    ('start_reason', pa.string()), ('termination_reason', pa.string())])


def derive_tracks(raw_path, output_directory, video_id, config, width, height, polygon=None):
    """All detections retained. Rejected bodies have null IDs, not missing rows."""
    directory = Path(output_directory)
    directory.mkdir(exist_ok=False)
    target = directory / f'{video_id}.tracked.slp'
    tmp = target.with_suffix('.tmp.slp')
    shutil.copyfile(raw_path, tmp)
    tracker = Tracker(video_id, width, height, config)
    obs_path = directory / 'observations.tmp.parquet'
    tracks_path = directory / 'tracklets.tmp.parquet'
    count = accepted = track_count = 0
    obs_buffer, track_buffer = [], []
    with h5py.File(raw_path, 'r') as raw, h5py.File(tmp, 'r+') as tracked, \
         pq.ParquetWriter(obs_path, OBS_SCHEMA, compression='zstd') as obs_writer, \
         pq.ParquetWriter(tracks_path, TRACK_SCHEMA, compression='zstd') as track_writer:
        del tracked['tracks_json']
        catalog = tracked.create_dataset('tracks_json', shape=(0,), maxshape=(None,), dtype=h5py.string_dtype('utf-8'))
        frames = raw['frames'][:]
        for frame_index, row in enumerate(frames):
            assert int(row['frame_idx']) == frame_index
            lo, hi = int(row['instance_id_start']), int(row['instance_id_end'])
            instances = raw['instances'][lo:hi]
            if len(instances):
                points = raw['pred_points'][int(instances['point_id_start'][0]):int(instances['point_id_end'][-1])].reshape(-1, 5)
            else:
                points = np.empty((0, 5), dtype=raw['pred_points'].dtype)
            xy = np.stack([points['x'], points['y']], axis=-1)
            frame = Frame(frame_index, xy[:, 0], xy[:, 1], points['score'][:, 0], points['score'][:, 1],
                          points['visible'][:, 0], points['visible'][:, 1])
            result = tracker.process(frame)
            ids = result['ids']
            assert len(set(ids[ids > 0])) == int((ids > 0).sum())
            track_buffer.extend(result['ended'])
            water = water_membership(frame.body, width, height, polygon) if polygon is not None else [None]*len(frame)
            for j, local_id in enumerate(ids):
                assigned = int(local_id) > 0
                record = {'video_id': video_id, 'frame': frame_index, 'instance_index': j,
                          'raw_instance_id': lo+j, 'detection_id': f'{video_id}/f{frame_index:06d}/i{j:04d}',
                          'tracklet_id': tracker.name(int(local_id)) if assigned else None,
                          'tracklet_local_id': int(local_id) if assigned else None,
                          'accepted_for_tracking': assigned, 'rejection_reason': str(result['validity']['reasons'][j]) or None,
                          'incoming_reason': str(result['incoming_reason'][j]), 'instance_score': float(instances['score'][j]),
                          'body_x': float(frame.body[j, 0]), 'body_y': float(frame.body[j, 1]),
                          'body_score': float(frame.body_score[j]), 'centroid_source': 'shell_proxy', 'water': water[j]}
                for k, node in enumerate(NODES):
                    for field in ('x', 'y', 'score', 'visible'):
                        value = points[field][j, k]
                        record[f'{node.lower()}_{field}'] = bool(value) if field == 'visible' else float(value)
                obs_buffer.append(record)
            # Only the track-reference field changes. Scores and every raw node survive.
            instances['track'] = np.where(ids > 0, ids-1, -1)
            tracked['instances'][lo:hi] = instances
            new_count = tracker.next_id-1
            if new_count > track_count:
                catalog.resize((new_count,))
                catalog[track_count:new_count] = [json.dumps([tracker.active[i][0], tracker.name(i)]) for i in range(track_count+1, new_count+1)]
                track_count = new_count
            count += len(ids)
            accepted += int((ids > 0).sum())
            if (frame_index+1) % 40 == 0:
                obs_writer.write_table(pa.Table.from_pylist(obs_buffer, schema=OBS_SCHEMA))
                track_writer.write_table(pa.Table.from_pylist(track_buffer, schema=TRACK_SCHEMA))
                obs_buffer.clear()
                track_buffer.clear()
        track_buffer.extend(tracker.finish())
        obs_writer.write_table(pa.Table.from_pylist(obs_buffer, schema=OBS_SCHEMA))
        track_writer.write_table(pa.Table.from_pylist(track_buffer, schema=TRACK_SCHEMA))
        tracked.attrs['identity_status'] = 'experimental_not_ground_truth_validated'
        tracked.attrs['raw_predictions_sha256'] = sha256(raw_path)
        tracked.flush()
    # Verify all raw measurement bytes are unaltered and IDs stay within bounds.
    with h5py.File(raw_path, 'r') as raw, h5py.File(tmp, 'r') as tracked:
        for key in ('frames', 'pred_points', 'points'):
            assert raw[key].shape == tracked[key].shape
            for start in range(0, len(raw[key]), 100000):
                assert raw[key][start:start+100000].tobytes() == tracked[key][start:start+100000].tobytes()
        for start in range(0, len(raw['instances']), 100000):
            a, b = raw['instances'][start:start+100000], tracked['instances'][start:start+100000]
            assert np.all((b['track'] >= -1) & (b['track'] < track_count)) if track_count else np.all(b['track'] == -1)
            b['track'] = a['track']
            assert a.tobytes() == b.tobytes()
        expected_ids = tracked['instances']['track'][:]
    # Full Parquet read-back: row ordering, assignment correspondence, lifespan counts.
    offset = 0
    lengths = np.zeros(track_count, np.int64)
    last_frames = np.full(track_count, -1, np.int64)
    for batch in pq.ParquetFile(obs_path).iter_batches(batch_size=10000):
        rows = batch.to_pylist()
        for row in rows:
            tid = row['tracklet_local_id']
            assert row['raw_instance_id'] == offset
            assert expected_ids[offset] == (-1 if tid is None else tid-1)
            if tid is not None:
                k = tid-1
                assert last_frames[k] == -1 or last_frames[k]+1 == row['frame']
                last_frames[k] = row['frame']
                lengths[k] += 1
            offset += 1
    assert offset == count and int(lengths.sum()) == accepted
    seen = set()
    for batch in pq.ParquetFile(tracks_path).iter_batches():
        for row in batch.to_pylist():
            k = row['tracklet_local_id']-1
            assert k not in seen and row['length'] == lengths[k]
            assert row['end_frame']-row['start_frame']+1 == row['length']
            seen.add(k)
    assert len(seen) == track_count
    readback_slp(tmp, len(frames), track_count)
    tmp.rename(target)
    obs_final, tracks_final = directory/'observations.parquet', directory/'tracklets.parquet'
    obs_path.rename(obs_final)
    tracks_path.rename(tracks_final)
    summary = {'observations': count, 'accepted_for_tracking': accepted,
               'unassigned_observations': count-accepted, 'tracklets': track_count,
               'raw_measurements_unchanged': True, 'parquet_readback_passed': True,
               'scientific_identity_accuracy': 'not_ground_truth_validated',
               'files': {p.name: sha256(p) for p in (target, obs_final, tracks_final)}}
    atomic_json(directory/'TRACKLETS_COMPLETE.json', summary)
    return summary
