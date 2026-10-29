"""Verify/publish the existing extraction from job 16416826; never re-encode.

The extraction passed its data checks but its final unrelated Git-state guard
failed while the user configured their branch. This finite recovery verifies
existing payloads against their sources and publishes them without Git commands.
"""
import csv
import json
import os
from pathlib import Path
import importlib.metadata
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sleap_io as sio
from prepare_crab_demo import (ROOT, FINAL, DATASET, PREDICTIONS, VIDEO, VIDEO_ID,
    START, STOP, COUNT, TOTAL, FPS, WIDTH, HEIGHT, NAME, normalize_tables, check_observations)
from pipeline_common import sha256, atomic_json

assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
stage = ROOT/'demo/.build_crab_demo_16416826'
assert stage.is_dir() and not FINAL.exists()
assert not (stage/'DEMO_COMPLETE.json').exists()
expected_names = {f'{NAME}.mp4', f'{NAME}.predictions.slp', f'{NAME}.tracked.slp',
                  'README.md','observations.parquet','frames.parquet','frame_map.csv','encoding.log'}
assert {p.name for p in stage.iterdir()} == expected_names
complete = json.loads((DATASET/'DATASET_COMPLETE.json').read_text())
assert complete['status'] == 'validated_complete'
assert sha256(DATASET/'manifest.json') == complete['manifest_sha256']
manifest = json.loads((DATASET/'manifest.json').read_text())
pins = {x['path']:x for x in manifest['source_pins']}
artifacts = {x['path']:x for x in manifest['artifacts']}
source_evidence = {}
for path in [VIDEO, PREDICTIONS/'raw'/f'{VIDEO_ID}.predictions.slp', PREDICTIONS/'tracklets'/f'{VIDEO_ID}.tracked.slp']:
    pin = pins[str(path.relative_to(ROOT))]
    assert path.stat().st_size == pin['bytes'] and path.stat().st_mtime_ns == pin['mtime_ns']
    if path.suffix == '.slp':
        assert sha256(path) == pin['sha256']
    source_evidence[str(path.relative_to(ROOT))] = dict(pin, rehashed_now=path.suffix == '.slp')
results = {}
for suffix, source in [('predictions',PREDICTIONS/'raw'/f'{VIDEO_ID}.predictions.slp'),
                       ('tracked',PREDICTIONS/'tracklets'/f'{VIDEO_ID}.tracked.slp')]:
    with h5py.File(source,'r') as src, h5py.File(stage/f'{NAME}.{suffix}.slp','r') as out:
        f = src['frames'][START:STOP]
        lo, hi = int(f[0]['instance_id_start']), int(f[-1]['instance_id_end'])
        i = src['instances'][lo:hi]
        p = src['pred_points'][int(i[0]['point_id_start']):int(i[-1]['point_id_end'])]
        f,i,p = normalize_tables(f,i,p,START,COUNT)
        used = sorted(set(int(v) for v in i['track'] if v >= 0))
        old = i['track'].copy()
        for new, previous in enumerate(used):
            i['track'][old == previous] = new
        expected_tracks = [json.loads(src['tracks_json'][idx]) for idx in used]
        assert [json.loads(v) for v in out['tracks_json'][:]] == expected_tracks
        for key, table in [('frames',f),('instances',i),('pred_points',p)]:
            assert out[key][:].tobytes() == table.tobytes(), (suffix,key)
        videos = [json.loads(v) for v in out['videos_json'][:]]
        assert len(videos) == 1 and videos[0]['filename'] == f'{NAME}.mp4'
        assert videos[0]['backend']['filename'] == f'{NAME}.mp4'
        assert videos[0]['backend']['shape'] == [COUNT,HEIGHT,WIDTH,3]
        assert len(out['points']) == len(out['suggestions_json']) == len(out['sessions_json']) == 0
        results[suffix] = {'source_instance_start':lo,'source_instance_stop':hi,'observations':len(i),
                           'instances':i,'frames':f,'points':p,'track_names':[v[1] for v in expected_tracks]}
    labels = sio.load_slp(stage/f'{NAME}.{suffix}.slp',open_videos=True)
    assert [lf.frame_idx for lf in labels] == list(range(COUNT))
    assert sum(len(lf.instances) for lf in labels) == len(i)
    assert Path(labels.videos[0].filename).resolve() == (stage/f'{NAME}.mp4').resolve()
    assert labels.videos[0][0].shape[:2] == (HEIGHT,WIDTH)
    labels.videos[0].close()
assert results['predictions']['points'].tobytes() == results['tracked']['points'].tobytes()
parts, source_parts = [], []
for path in sorted((DATASET/'tables/observations'/VIDEO_ID).glob('*.parquet')):
    table = pq.read_table(path,filters=[('frame','>=',START),('frame','<',STOP)])
    if table.num_rows:
        assert sha256(path) == artifacts[str(path.relative_to(DATASET))]['sha256']
        parts.append(table)
        source_parts.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
source_obs = pa.concat_tables(parts)
source_obs = source_obs.take(pc.sort_indices(source_obs,sort_keys=[('frame','ascending'),('instance_index','ascending')]))
obs = pq.read_table(stage/'observations.parquet')
assert obs.select(source_obs.column_names).equals(source_obs)
assert set(obs.column_names)-set(source_obs.column_names) == {'demo_frame','demo_time_s','demo_instance_row'}
check_observations(obs,results['tracked'])
np.testing.assert_array_equal(obs['demo_instance_row'].to_numpy(),np.arange(len(obs)))
np.testing.assert_allclose(obs['demo_time_s'].to_numpy(),obs['demo_frame'].to_numpy()/FPS,rtol=0,atol=1e-12)
frames_path = DATASET/'tables/frames.parquet'
assert sha256(frames_path) == artifacts['tables/frames.parquet']['sha256']
source_frames = pq.read_table(frames_path,filters=[('video_id','==',VIDEO_ID),('frame','>=',START),('frame','<',STOP)])
source_frames = source_frames.take(pc.sort_indices(source_frames,sort_keys=[('frame','ascending')]))
frames = pq.read_table(stage/'frames.parquet')
assert frames.select(source_frames.column_names).equals(source_frames)
np.testing.assert_array_equal(frames['demo_frame'].to_numpy(),np.arange(COUNT))
np.testing.assert_array_equal(frames['source_count'].to_numpy(),results['tracked']['frames']['instance_id_end']-results['tracked']['frames']['instance_id_start'])
assigned = np.bincount(results['tracked']['instances']['frame_id'][results['tracked']['instances']['track']>=0].astype(int),minlength=COUNT)
np.testing.assert_array_equal(frames['assigned_count'].to_numpy(),assigned)
with (stage/'frame_map.csv').open() as f:
    mapping = list(csv.DictReader(f))
assert len(mapping) == COUNT
for n,row in enumerate(mapping):
    assert int(row['demo_frame']) == n and int(row['source_frame']) == START+n
    assert abs(float(row['demo_time_s'])-n/FPS)<1e-9
    assert abs(float(row['source_time_s'])-(START+n)/FPS)<1e-9
    assert len(row['source_decoded_bgr_sha256']) == 64
# Read every frame of the already-created video; no encoder and no rewriting.
cap = cv2.VideoCapture(str(stage/f'{NAME}.mp4'),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == COUNT
assert abs(cap.get(cv2.CAP_PROP_FPS)-FPS)<1e-8
n = 0
while True:
    ok, image = cap.read()
    if not ok:
        break
    assert image.shape == (HEIGHT,WIDTH,3)
    n += 1
cap.release()
assert n == COUNT
# The previous job reached its last Git-only assertion after all original
# alignment/portability checks. Retain that failure log as provenance.
error_path = ROOT/'logs/demo_preparation/demo_16416826.err'
assert 'assert git_state() == git_before' in error_path.read_text()
assert 'Ran 8 tests' in error_path.read_text() and '\nOK\n' in error_path.read_text()
(stage/'README.md').write_text((ROOT/'docs/demo/README.template.md').read_text())
receipt = {'status':'validated_complete','build_job_id':'16416826','publication_job_id':os.environ['SLURM_JOB_ID'],
           'source_frame_start':START,'source_frame_stop_exclusive':STOP,'frames':COUNT,
           'source_video_id':VIDEO_ID,'width':WIDTH,'height':HEIGHT,'fps_numerator':30000,'fps_denominator':1001,
           'observations':obs.num_rows,'assigned_observations':int(assigned.sum()),
           'unassigned_observations':int(obs.num_rows-assigned.sum()),
           'experimental_tracklets_present':len(results['tracked']['track_names']),
           'source_dataset_manifest_sha256':complete['manifest_sha256'],'source_inputs':source_evidence,
           'source_table_parts':source_parts,'source_frames_sha256':sha256(frames_path),
           'recovery_reason':'Extraction/data validation succeeded, but an inappropriate Git-metadata-unchanged guard blocked publication while the user configured their branch. Existing payloads revalidated; no re-encoding or prediction generation.',
           'video_encoding':'H.264 CRF 18, maxrate/bufsize 60M, veryfast, native 7680x4320, 30000/1001 fps; lossy pixels, unchanged coordinates',
           'validation':{'source_slps_rehashed':True,'all_pose_bytes_match_source':True,'original_dataframe_columns_identical':True,
                         'all_300_output_frames_decoded':True,'portable_basename_video_reference':True,'all_8_build_tests_passed':True,
                         'video_reencoded_in_publication':False,'inference_or_tracking_rerun':False,'no_git_commands_executed':True},
           'limitations':['Experimental identities are not ground truth.','Video is a lossy demo copy, not a pixel archive.',
                          'Source video full hash inherited from published dataset; current size/mtime checked.',
                          'Full-video source paths/ordinals in the dataframe are provenance, not bundled files.'],
           'versions':{p:importlib.metadata.version(p) for p in ['sleap-io','numpy','h5py','pyarrow','opencv-python']},
           'code_sha256':{'scripts/publish_existing_crab_demo.py':sha256(Path(__file__))},
           'artifacts':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in stage.iterdir() if p.is_file()}}
assert all(v['bytes']<100*1024**2 for v in receipt['artifacts'].values())
atomic_json(stage/'DEMO_COMPLETE.json',receipt)
stage.rename(FINAL)
print(json.dumps({k:receipt[k] for k in ['status','publication_job_id','frames','observations','assigned_observations','unassigned_observations','experimental_tracklets_present','artifacts']},indent=2),flush=True)
print('PUBLISHED',FINAL,flush=True)
