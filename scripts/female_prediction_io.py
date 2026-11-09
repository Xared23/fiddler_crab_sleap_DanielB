"""One-node raw SLP validation/streaming merge, adapted from the male finalizer.

Kept separate because the pinned male finalizer assumes five landmarks. No
tracking, filtering, rescaling, or modifications to existing male code/data.
"""
from __future__ import annotations
import json
from pathlib import Path
import h5py
import numpy as np
import sleap_io as sio
from pipeline_common import sha256

TABLES = ('frames', 'instances', 'pred_points')
ALLOWED = set(TABLES) | {'points', 'metadata', 'videos_json', 'tracks_json',
                         'suggestions_json', 'sessions_json', 'provenance_json'}


def inspect_chunk(path, start, stop, video, nodes):
    labels = sio.load_slp(path, open_videos=False)
    if len(labels.videos) != 1 or Path(labels.videos[0].filename).resolve() != Path(video).resolve():
        raise RuntimeError('Incorrect prediction video')
    if len(labels.skeletons) != 1 or labels.skeletons[0].node_names != nodes:
        raise RuntimeError('Incorrect prediction skeleton')
    if [int(lf.frame_idx) for lf in labels] != list(range(start, stop)):
        raise RuntimeError('Missing/duplicate/unordered prediction frames')
    if labels.tracks or any(not isinstance(i, sio.PredictedInstance) or i.track is not None for lf in labels for i in lf):
        raise RuntimeError('Raw predictions contain labels or identities')
    with h5py.File(path, 'r') as source:
        if not set(source) <= ALLOWED or len(source['points']) or len(source['tracks_json']):
            raise RuntimeError('Unsupported SLP schema')
        frames, instances, points = (source[key][:] for key in TABLES)
    np.testing.assert_array_equal(frames['frame_idx'], np.arange(start, stop))
    np.testing.assert_array_equal(frames['frame_id'], np.arange(stop-start))
    assert np.all(frames['video'] == 0)
    n, k = len(instances), len(nodes)
    np.testing.assert_array_equal(instances['instance_id'], np.arange(n))
    assert np.all(instances['instance_type'] == 1) and np.all(instances['skeleton'] == 0)
    assert np.all(instances['track'] == -1) and np.all(instances['from_predicted'] == -1)
    np.testing.assert_array_equal(instances['point_id_start'], np.arange(n)*k)
    np.testing.assert_array_equal(instances['point_id_end'], (np.arange(n)+1)*k)
    assert len(points) == n*k
    end = 0
    for row in frames:
        lo, hi = int(row['instance_id_start']), int(row['instance_id_end'])
        assert lo == end and lo <= hi <= n
        assert np.all(instances['frame_id'][lo:hi] == row['frame_id'])
        end = hi
    assert end == n
    return frames, instances, points


def offset_tables(frames, instances, frame_offset, instance_offset, point_offset):
    frames['frame_id'] += frame_offset
    frames['instance_id_start'] += instance_offset
    frames['instance_id_end'] += instance_offset
    instances['instance_id'] += instance_offset
    instances['frame_id'] += frame_offset
    instances['point_id_start'] += point_offset
    instances['point_id_end'] += point_offset


def merge_raw(chunks, target, total_frames, video, nodes):
    target = Path(target)
    tmp = target.with_suffix('.tmp.slp')
    if target.exists() or tmp.exists():
        raise FileExistsError(f'Existing output retained: {target} / {tmp}')
    next_frame = ni = npnt = 0
    with h5py.File(tmp, 'x') as out:
        for path, start, stop in chunks:
            if start != next_frame:
                raise RuntimeError('Chunk ranges are missing, overlapping, or unordered')
            f, i, p = inspect_chunk(path, start, stop, video, nodes)
            with h5py.File(path, 'r') as source:
                if start == 0:
                    for key in source:
                        if key not in TABLES:
                            source.copy(key, out)
                    for key, value in source.attrs.items():
                        out.attrs[key] = value
                    for key, data in zip(TABLES, (f, i, p)):
                        out.create_dataset(key, shape=(0,), maxshape=(None,), dtype=data.dtype, chunks=True)
                else:
                    np.testing.assert_array_equal(source['videos_json'][:], out['videos_json'][:])
                    a = json.loads(source['metadata'].attrs['json'])
                    b = json.loads(out['metadata'].attrs['json'])
                    assert a['skeletons'] == b['skeletons'] and a['nodes'] == b['nodes']
            offset_tables(f, i, next_frame, ni, npnt)
            for key, data in zip(TABLES, (f, i, p)):
                ds = out[key]
                old = len(ds)
                ds.resize((old+len(data),))
                ds[old:] = data
            next_frame, ni, npnt = stop, ni+len(i), npnt+len(p)
        if next_frame != total_frames:
            raise RuntimeError('Incomplete video coverage')
        provenance = {'raw_merge': {'frames': total_frames, 'chunks': [str(p) for p, _, _ in chunks],
                                    'tracking_enabled': False, 'nodes': nodes}}
        if 'provenance_json' in out:
            del out['provenance_json']
        out.create_dataset('provenance_json', data=np.bytes_(json.dumps(provenance)))
        out.flush()
    # Every coordinate/score byte and all index offsets are checked, chunk by chunk.
    ioff = poff = 0
    with h5py.File(tmp, 'r') as merged:
        for path, start, stop in chunks:
            f, i, p = inspect_chunk(path, start, stop, video, nodes)
            offset_tables(f, i, start, ioff, poff)
            assert merged['frames'][start:stop].tobytes() == f.tobytes()
            assert merged['instances'][ioff:ioff+len(i)].tobytes() == i.tobytes()
            assert merged['pred_points'][poff:poff+len(p)].tobytes() == p.tobytes()
            ioff, poff = ioff+len(i), poff+len(p)
    labels = sio.load_slp(tmp, open_videos=False, lazy=True)
    assert len(labels) == total_frames and not labels.tracks
    assert labels.skeletons[0].node_names == nodes
    for idx in sorted({0, min(39, total_frames-1), min(40, total_frames-1), total_frames-1}):
        assert labels[idx].frame_idx == idx
    tmp.rename(target)
    return {'frames': total_frames, 'instances': ni, 'sha256': sha256(target), 'nodes': nodes}
