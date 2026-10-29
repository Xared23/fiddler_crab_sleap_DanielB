"""Prepare the user's 300-frame, portable v007 collaboration demo (Slurm only).

No inference/tracking, Git commands or modifications to source data. Native
spatial coordinates; clip SLP indices are local, dataframe source columns stay
unchanged and gain explicit demo-index columns. Existing output is never replaced.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import h5py
import imageio_ffmpeg
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sleap_io as sio

from pipeline_common import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
VIDEO_ID = 'Crab_Cam_2_2'
START, COUNT, TOTAL = 14250, 300, 28800
STOP = START + COUNT
FPS = 30000 / 1001
WIDTH, HEIGHT = 7680, 4320
NAME = 'Crab_Cam_2_2.frames_014250_014549'
FINAL = ROOT / 'demo/crab_cam2_2_middle_300'
DATASET = ROOT / 'datasets/v007_best_body_first_dataframe_v001'
PREDICTIONS = ROOT / 'predictions/v007_best_body_first_v001' / VIDEO_ID
VIDEO = ROOT / 'crab_vids/Crab_Cam_2_2.mp4'
NODES = ['shell','shoulder','elbow','claw','arm']


def normalize_tables(frames, instances, points, start, count):
    """Index transformation only; preserve all pose/score bytes and row order."""
    frames, instances, points = frames.copy(), instances.copy(), points.copy()
    np.testing.assert_array_equal(frames['frame_idx'], np.arange(start, start+count))
    np.testing.assert_array_equal(frames['frame_id'], np.arange(start, start+count))
    assert np.all(frames['video'] == 0)
    lo, hi = int(frames[0]['instance_id_start']), int(frames[-1]['instance_id_end'])
    assert len(instances) == hi-lo
    assert np.all(frames['instance_id_end'] >= frames['instance_id_start'])
    np.testing.assert_array_equal(frames['instance_id_start'][1:], frames['instance_id_end'][:-1])
    np.testing.assert_array_equal(instances['instance_id'], np.arange(lo, hi))
    expected_frames = np.repeat(np.arange(start, start+count), (frames['instance_id_end']-frames['instance_id_start']).astype(np.int64))
    np.testing.assert_array_equal(instances['frame_id'], expected_frames)
    assert np.all(instances['instance_type'] == 1) and np.all(instances['from_predicted'] == -1)
    p0 = int(instances[0]['point_id_start']) if len(instances) else 0
    assert len(points) == len(instances)*5
    np.testing.assert_array_equal(instances['point_id_start'], np.arange(len(instances))*5+p0)
    np.testing.assert_array_equal(instances['point_id_end'], (np.arange(len(instances))+1)*5+p0)
    frames['frame_id'] -= start
    frames['frame_idx'] -= start
    frames['instance_id_start'] -= lo
    frames['instance_id_end'] -= lo
    instances['instance_id'] -= lo
    instances['frame_id'] -= start
    instances['point_id_start'] -= p0
    instances['point_id_end'] -= p0
    return frames, instances, points


def slice_slp(source, target, clip_name, start=START, count=COUNT):
    with h5py.File(source, 'r') as src:
        assert set(src.keys()) == {'frames','instances','metadata','points','pred_points',
                                  'provenance_json','sessions_json','suggestions_json','tracks_json','videos_json'}
        assert len(src['points']) == 0
        f = src['frames'][start:start+count]
        lo, hi = int(f[0]['instance_id_start']), int(f[-1]['instance_id_end'])
        i = src['instances'][lo:hi]
        p = src['pred_points'][int(i[0]['point_id_start']):int(i[-1]['point_id_end'])] if len(i) else src['pred_points'][:0]
        f, i, p = normalize_tables(f, i, p, start, count)
        used = sorted(set(int(v) for v in i['track'] if v >= 0))
        track_map = {old: new for new, old in enumerate(used)}
        original_tracks = i['track'].copy()
        for old, new in track_map.items():
            i['track'][original_tracks == old] = new
        catalog = src['tracks_json'][used] if used else []
        videos = [json.loads(v) for v in src['videos_json'][:]]
        assert len(videos) == 1
        videos[0]['filename'] = clip_name
        videos[0]['backend'].update(filename=clip_name, shape=[count, HEIGHT, WIDTH, 3], fps=FPS)
        with h5py.File(target, 'x') as out:
            for key in ['metadata','points']:
                src.copy(key, out)
            for key, value in src.attrs.items():
                out.attrs[key] = value
            for key, table in [('frames', f), ('instances',i), ('pred_points',p)]:
                out.create_dataset(key, data=table, compression='gzip', shuffle=True)
            dt = h5py.string_dtype('utf-8')
            out.create_dataset('videos_json', data=[json.dumps(v) for v in videos], dtype=dt)
            out.create_dataset('tracks_json', data=[v.decode() if isinstance(v, bytes) else v for v in catalog], dtype=dt)
            # No unrelated source-video suggestions or sessions are part of this clip.
            for key in ['sessions_json','suggestions_json']:
                out.create_dataset(key, data=[], dtype=dt)
            provenance = {'demo_source_slp': str(source.relative_to(ROOT)), 'source_frame_start': start,
                          'source_frame_stop_exclusive': start+count, 'demo_frame_mapping': 'source_frame = demo_frame + source_frame_start',
                          'measurements_unchanged': True, 'tracking_rerun': False,
                          'track_names': 'original full-video IDs; only the internal catalog indices are compacted'}
            out.create_dataset('provenance_json', data=np.bytes_(json.dumps(provenance)))
    with h5py.File(target, 'r') as out:
        assert out['frames'][:].tobytes() == f.tobytes()
        assert out['instances'][:].tobytes() == i.tobytes()
        assert out['pred_points'][:].tobytes() == p.tobytes()
        names = [json.loads(v)[1] for v in out['tracks_json'][:]]
        assert len(set(names)) == len(names)
    return {'source_instance_start': lo, 'source_instance_stop': hi, 'observations': len(i),
            'tracks': len(used), 'frames': f, 'instances': i, 'points': p, 'track_names': names}


def add_demo_indices(table):
    assert 'demo_frame' not in table.column_names
    frame = table['frame'].to_numpy()
    assert np.all((frame >= START) & (frame < STOP))
    return table.append_column('demo_frame', pa.array(frame-START, type=pa.int64())).append_column(
        'demo_time_s', pa.array((frame-START)/FPS, type=pa.float64()))


def check_observations(table, slp):
    assert table.num_rows == slp['observations']
    np.testing.assert_array_equal(table['source_instance_row'].to_numpy(),
                                  np.arange(slp['source_instance_start'], slp['source_instance_stop']))
    np.testing.assert_array_equal(table['raw_instance_id'].to_numpy(), table['source_instance_row'].to_numpy())
    np.testing.assert_array_equal(table['frame'].to_numpy(), slp['instances']['frame_id']+START)
    np.testing.assert_array_equal(table['demo_frame'].to_numpy(), slp['instances']['frame_id'])
    points = slp['points'].reshape(-1, 5)
    for n, node in enumerate(NODES):
        for field in ['x','y','score','visible']:
            np.testing.assert_equal(table[f'{node}_{field}'].to_numpy(), points[field][:, n])
    np.testing.assert_equal(table['instance_score'].to_numpy(), slp['instances']['score'].astype(float))
    np.testing.assert_equal(table['source_tracking_score'].to_numpy(), slp['instances']['tracking_score'].astype(float))
    actual = [None if idx < 0 else slp['track_names'][idx] for idx in slp['instances']['track']]
    assert table['tracklet_id'].to_pylist() == actual
    assert len(set(table['detection_id'].to_pylist())) == table.num_rows


def make_video(stage):
    """Sequentially decode exactly the selected source indices; no resize/overlay."""
    clip = stage / f'{NAME}.mp4'
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, '-hide_banner', '-loglevel','warning', '-nostdin', '-n',
               '-f','rawvideo','-pix_fmt','bgr24','-s',f'{WIDTH}x{HEIGHT}', '-r','30000/1001', '-i','pipe:0',
               '-an','-c:v','libx264','-preset','veryfast','-crf','18','-maxrate','60M','-bufsize','60M',
               '-pix_fmt','yuv420p','-threads','4','-frames:v',str(COUNT),'-movflags','+faststart',str(clip)]
    cap = cv2.VideoCapture(str(VIDEO), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == TOTAL
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (WIDTH,HEIGHT)
    assert abs(cap.get(cv2.CAP_PROP_FPS)-FPS) < 1e-8
    assert cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == START
    records, references = [], {}
    log = stage / 'encoding.log'
    with log.open('xb') as err:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
        try:
            for local in range(COUNT):
                assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == START+local
                ok, frame = cap.read()
                assert ok and frame.shape == (HEIGHT, WIDTH, 3), local
                assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == START+local+1
                records.append({'demo_frame': local, 'source_frame': START+local,
                                'source_time_s': (START+local)/FPS, 'demo_time_s': local/FPS,
                                'source_decoded_bgr_sha256': hashlib.sha256(frame.tobytes()).hexdigest()})
                if local in [0, COUNT//2, COUNT-1]:
                    references[local] = frame[::8, ::8].copy()
                process.stdin.write(frame.tobytes())
                if local % 50 == 0:
                    print('VIDEO_FRAME', local, flush=True)
            process.stdin.close()
            assert process.wait(timeout=300) == 0, log.read_text()
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            cap.release()
    assert clip.stat().st_size < 100*1024**2, 'Clip exceeds ordinary GitHub per-file limit; do not stage'
    with (stage/'frame_map.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    cap = cv2.VideoCapture(str(clip), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == COUNT
    assert abs(cap.get(cv2.CAP_PROP_FPS)-FPS) < 1e-8
    decoded, mae = 0, {}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (HEIGHT, WIDTH, 3)
        if decoded in references:
            error = float(np.abs(frame[::8,::8].astype(np.float32)-references[decoded].astype(np.float32)).mean())
            assert error < 10, (decoded, error)
            mae[str(decoded)] = error
        decoded += 1
    cap.release()
    assert decoded == COUNT
    return {'command': command, 'decoded_frames': decoded, 'sampled_bgr_mean_absolute_errors': mae,
            'encoding': 'H.264 CRF 18, VBV maxrate/bufsize 60 Mbit; native 7680x4320; lossy video only, no coordinate changes'}


def build():
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    assert not FINAL.exists(), f'Refusing to replace {FINAL}'
    complete = json.loads((DATASET/'DATASET_COMPLETE.json').read_text())
    assert complete['status'] == 'validated_complete'
    assert sha256(DATASET/'manifest.json') == complete['manifest_sha256']
    manifest = json.loads((DATASET/'manifest.json').read_text())
    assert manifest['prediction_run_id'] == 'v007_best_body_first_v001'
    pins = {x['path']: x for x in manifest['source_pins']}
    sources = [VIDEO, PREDICTIONS/'raw'/f'{VIDEO_ID}.predictions.slp',
               PREDICTIONS/'tracklets'/f'{VIDEO_ID}.tracked.slp']
    source_state = {}
    for path in sources:
        pin = pins[str(path.relative_to(ROOT))]
        assert path.stat().st_size == pin['bytes'] and path.stat().st_mtime_ns == pin['mtime_ns']
        # Rehash SLPs. The ~11.45 GB video uses its published size/mtime pin;
        # selected decoded frames get fresh hashes in frame_map.csv.
        if path.suffix == '.slp':
            assert sha256(path) == pin['sha256']
        source_state[str(path.relative_to(ROOT))] = {'bytes': pin['bytes'], 'mtime_ns': pin['mtime_ns'],
                                                    'published_sha256': pin['sha256'], 'rehashed_now': path.suffix == '.slp'}
    artifacts = {x['path']: x for x in manifest['artifacts']}
    # Rehash each source table file actually used in the bounded extraction.
    obs_dir = DATASET / 'tables/observations' / VIDEO_ID
    selected = []
    for path in sorted(obs_dir.glob('*.parquet')):
        part = pq.read_table(path, filters=[('frame','>=',START),('frame','<',STOP)])
        if part.num_rows:
            rel = str(path.relative_to(DATASET))
            assert sha256(path) == artifacts[rel]['sha256'], rel
            selected.append((path, part))
    assert selected
    observations = pa.concat_tables([part for _, part in selected])
    observations = observations.take(pc.sort_indices(observations, sort_keys=[('frame','ascending'),('instance_index','ascending')]))
    assert set(observations['video_id'].to_pylist()) == {VIDEO_ID}
    frames_path = DATASET / 'tables/frames.parquet'
    assert sha256(frames_path) == artifacts['tables/frames.parquet']['sha256']
    frames = pq.read_table(frames_path, filters=[('video_id','==',VIDEO_ID),('frame','>=',START),('frame','<',STOP)])
    frames = frames.take(pc.sort_indices(frames, sort_keys=[('frame','ascending')]))
    np.testing.assert_array_equal(frames['frame'].to_numpy(), np.arange(START, STOP))
    FINAL.parent.mkdir(parents=True, exist_ok=True)
    stage = FINAL.parent / f'.build_crab_demo_{os.environ["SLURM_JOB_ID"]}'
    stage.mkdir(exist_ok=False)
    video_info = make_video(stage)
    raw = slice_slp(sources[1], stage/f'{NAME}.predictions.slp', f'{NAME}.mp4')
    tracked = slice_slp(sources[2], stage/f'{NAME}.tracked.slp', f'{NAME}.mp4')
    assert raw['points'].tobytes() == tracked['points'].tobytes()
    for field in raw['instances'].dtype.names:
        if field != 'track':
            assert raw['instances'][field].tobytes() == tracked['instances'][field].tobytes()
    out_obs = add_demo_indices(observations).append_column('demo_instance_row', pa.array(np.arange(observations.num_rows)))
    out_frames = add_demo_indices(frames)
    check_observations(out_obs, tracked)
    np.testing.assert_array_equal(frames['source_count'].to_numpy(), raw['frames']['instance_id_end']-raw['frames']['instance_id_start'])
    assigned = np.bincount(tracked['instances']['frame_id'][tracked['instances']['track'] >= 0].astype(int), minlength=COUNT)
    np.testing.assert_array_equal(frames['assigned_count'].to_numpy(), assigned)
    for name, table in [('observations.parquet',out_obs),('frames.parquet',out_frames)]:
        pq.write_table(table, stage/name, compression='zstd')
        assert pq.read_table(stage/name).equals(table)
    # Confirm original dataframe columns are copied literally, not recomputed.
    assert pq.read_table(stage/'observations.parquet', columns=observations.column_names).equals(observations)
    df = pd.read_parquet(stage/'observations.parquet', dtype_backend='numpy_nullable')
    assert len(df) == raw['observations'] and df['demo_frame'].between(0,COUNT-1).all()
    # Relocate independent copies of the SLPs plus a clip hardlink, then open
    # from a different cwd: no absolute Della video path is needed for playback.
    portability = ROOT/'logs/demo_preparation'/f'portability_{os.environ["SLURM_JOB_ID"]}'
    portability.mkdir(exist_ok=False)
    os.link(stage/f'{NAME}.mp4', portability/f'{NAME}.mp4')
    for suffix in ['predictions','tracked']:
        path = portability/f'{NAME}.{suffix}.slp'
        shutil.copy2(stage/path.name, path)
        old_cwd = Path.cwd()
        try:
            os.chdir(ROOT)
            labels = sio.load_slp(path, open_videos=True)
            assert len(labels) == COUNT
            assert [lf.frame_idx for lf in labels] == list(range(COUNT))
            assert labels.skeletons[0].node_names == ['Shell','Shoulder','Elbow','Claw','Arm']
            assert sum(len(lf.instances) for lf in labels) == raw['observations']
            assert Path(labels.videos[0].filename).resolve() == (portability/f'{NAME}.mp4').resolve()
            assert labels.videos[0][0].shape[:2] == (HEIGHT,WIDTH)
            assert len(labels.tracks) == (0 if suffix == 'predictions' else tracked['tracks'])
            labels.videos[0].close()
        finally:
            os.chdir(old_cwd)
    (stage/'README.md').write_text((ROOT/'docs/demo/README.template.md').read_text())
    source_files = ['scripts/prepare_crab_demo.py','scripts/archive_v004.py','tests/test_crab_demo.py','jobs/prepare_crab_demo.sh']
    receipt = {'status':'validated_complete','job_id':os.environ['SLURM_JOB_ID'], 'source_video_id':VIDEO_ID,
               'source_frame_start':START,'source_frame_stop_exclusive':STOP,'demo_frame_start':0,'frames':COUNT,
               'width':WIDTH,'height':HEIGHT,'fps_numerator':30000,'fps_denominator':1001,
               'observations':raw['observations'],'assigned_observations':int(assigned.sum()),
               'unassigned_observations':int(raw['observations']-assigned.sum()),'experimental_tracklets_present':tracked['tracks'],
               'source_dataset_manifest_sha256':complete['manifest_sha256'], 'source_inputs':source_state,
               'source_table_parts':[{'path':str(p.relative_to(ROOT)),'sha256':artifacts[str(p.relative_to(DATASET))]['sha256']} for p,_ in selected],
               'video':video_info,'versions':{p:importlib.metadata.version(p) for p in ['sleap-io','numpy','h5py','pandas','pyarrow','opencv-python','imageio-ffmpeg']},
               'code_sha256':{p:sha256(ROOT/p) for p in source_files},
               'validation':{'all_300_video_frames_decoded':True,'raw_and_tracked_points_identical_to_source':True,
                             'dataframe_source_columns_copied_exactly':True,'dataframe_matches_every_slp_detection_and_track':True,
                             'portable_slp_video_loading_from_relocated_copy':True,'no_git_commands_executed':True,
                             'neural_inference_or_tracking_rerun':False},
               'limitations':['Video is lossy re-encoded, without resizing or spatial transforms.',
                              'Track IDs are experimental; a clip can truncate full-video tracklets.',
                              'Original source-reference dataframe columns are provenance, not files bundled in the demo.',
                              'Full source video hash is inherited; fresh verification uses size/mtime plus selected decoded-frame hashes.']}
    for rel, pin in source_state.items():
        path = ROOT/rel
        assert path.stat().st_size == pin['bytes'] and path.stat().st_mtime_ns == pin['mtime_ns']
    # User branch/commit operations are independent of data validity. This
    # script never invokes Git; concurrent legitimate Git changes do not block publication.
    receipt['artifacts'] = {p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in stage.iterdir() if p.is_file()}
    assert all(p['bytes'] < 100*1024**2 for p in receipt['artifacts'].values())
    atomic_json(stage/'DEMO_COMPLETE.json', receipt)
    stage.rename(FINAL)
    print('DEMO_COMPLETE', json.dumps({k:receipt[k] for k in ['frames','observations','assigned_observations','unassigned_observations','experimental_tracklets_present','validation']}, indent=2), flush=True)


if __name__ == '__main__':
    build()
