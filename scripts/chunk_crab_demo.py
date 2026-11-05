"""Lossless, independently playable video chunks and aligned v007 demo data.

Production video operation is stream copy only. The source has one non-reordered
packet per frame. Every packet hash AND every decoded BGR frame are verified.
"""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sleap_io as sio

import prepare_crab_demo as base
from pipeline_common import ROOT, atomic_json, sha256

COUNT = 2000
DESIRED_START = 13400
FINAL = ROOT/'demo/crab_cam2_2_middle_2000'
LIMIT = 100*1024**2
PACKET_BUDGET = 90*1024**2
TEMPLATE = ROOT/'docs/demo/README_2000.template.md'
SOURCES = ['scripts/chunk_crab_demo.py','scripts/prepare_crab_demo.py',
           'scripts/replace_crab_demo_2000.py','tests/test_crab_demo.py',
           'tests/test_crab_demo_chunks.py','jobs/replace_crab_demo_2000.sh',
           'docs/demo/README_2000.template.md']


def probe(path):
    command = [shutil.which('ffprobe') or '/usr/bin/ffprobe', '-v','error',
        '-select_streams','v:0','-show_packets','-show_streams','-show_data_hash','sha256',
        '-show_entries','stream=codec_name,codec_tag_string,profile,width,height,pix_fmt,color_range,color_space,color_transfer,color_primaries,r_frame_rate,avg_frame_rate,time_base,nb_frames,extradata_hash:packet=pts,dts,duration,size,flags,data_hash',
        '-of','json',str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
    return json.loads(result.stdout)


def validate_packet_sequence(packets, step=1001):
    for i,p in enumerate(packets):
        assert int(p['pts']) == int(p['dts']) == i*step, 'Reordered/nonuniform source requires a different cutter'
        assert int(p['duration']) == step
        assert int(p['size']) > 0 and p['data_hash'].startswith('SHA256:')


def select_start(packets, desired=DESIRED_START, count=COUNT):
    keys = [i for i,p in enumerate(packets) if 'K' in p['flags'] and i+count <= len(packets)]
    if not keys: raise ValueError('No suitable keyframe start')
    return min(keys, key=lambda i:(abs(i-desired), i))


def plan_chunks(packets, start, count, budget=PACKET_BUDGET):
    stop = start+count
    if count <= 0 or start < 0 or stop > len(packets) or 'K' not in packets[start]['flags']:
        raise ValueError('Invalid selection/keyframe')
    ends = [i for i in range(start+1,stop) if 'K' in packets[i]['flags']] + [stop]
    cumulative = np.r_[0,np.cumsum([int(p['size']) for p in packets],dtype=np.int64)]
    result=[]; cursor=start
    while cursor < stop:
        candidates=[end for end in ends if end > cursor and cumulative[end]-cumulative[cursor] <= budget]
        if not candidates: raise ValueError('A single GOP exceeds the chunk packet budget; no re-encoding allowed')
        end=max(candidates)
        result.append((cursor,end)); cursor=end
    assert result[0][0]==start and result[-1][1]==stop
    assert sum(b-a for a,b in result)==count
    return result


def copy_video(source, target, start, count):
    """Copy existing compressed packets; NEVER invoke a video encoder/filter."""
    if Path(target).exists(): raise FileExistsError(target)
    command = [imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-nostdin','-n',
        '-ss',f'{start*1001/30000:.9f}','-i',str(source),'-map','0:v:0',
        '-c:v','copy','-frames:v',str(count),'-an','-sn','-dn',
        '-avoid_negative_ts','make_zero','-movflags','+faststart',str(target)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=180)
    if result.returncode: raise RuntimeError(result.stderr)
    return command


def validate_copy(source_info, clip_info, start, stop):
    original=source_info['packets'][start:stop]; copied=clip_info['packets']
    assert len(original)==len(copied)==stop-start
    validate_packet_sequence(copied)
    for a,b in zip(original,copied):
        assert a['data_hash']==b['data_hash'], 'Video packet was changed'
        assert int(a['size'])==int(b['size'])
    a,b=source_info['streams'][0],clip_info['streams'][0]
    for key in ['codec_name','profile','width','height','pix_fmt','color_range','color_space',
                'color_transfer','color_primaries','r_frame_rate','time_base','extradata_hash']:
        assert a.get(key)==b.get(key), (key,a.get(key),b.get(key))


def add_indices(table, selection_start, chunk_start, chunk_id):
    frame=table['frame'].to_numpy()
    result=table
    for name,values in [('demo_frame',frame-selection_start),('demo_time_s',(frame-selection_start)/base.FPS),
                        ('chunk_frame',frame-chunk_start),('chunk_time_s',(frame-chunk_start)/base.FPS),
                        ('chunk_id',np.full(len(frame),chunk_id))]:
        assert name not in result.column_names
        result=result.append_column(name,pa.array(values))
    return result


def source_tables(start,stop):
    complete=json.loads((base.DATASET/'DATASET_COMPLETE.json').read_text())
    assert complete['status']=='validated_complete'
    assert sha256(base.DATASET/'manifest.json')==complete['manifest_sha256']
    manifest=json.loads((base.DATASET/'manifest.json').read_text())
    assert manifest['prediction_run_id']=='v007_best_body_first_v001'
    pins={x['path']:x for x in manifest['source_pins']}
    source_state={}
    for path in [base.VIDEO,base.PREDICTIONS/'raw'/f'{base.VIDEO_ID}.predictions.slp',
                 base.PREDICTIONS/'tracklets'/f'{base.VIDEO_ID}.tracked.slp']:
        pin=pins[str(path.relative_to(ROOT))]
        assert (path.stat().st_size,path.stat().st_mtime_ns)==(pin['bytes'],pin['mtime_ns'])
        if path.suffix=='.slp': assert sha256(path)==pin['sha256']
        source_state[str(path.relative_to(ROOT))]=dict(pin,rehashed_now=path.suffix=='.slp')
    artifacts={x['path']:x for x in manifest['artifacts']}
    parts=[]; table_pins={}
    for path in sorted((base.DATASET/'tables/observations'/base.VIDEO_ID).glob('*.parquet')):
        table=pq.read_table(path,filters=[('frame','>=',start),('frame','<',stop)])
        if table.num_rows:
            digest=sha256(path); assert digest==artifacts[str(path.relative_to(base.DATASET))]['sha256']
            parts.append(table); table_pins[str(path.relative_to(ROOT))]=digest
    observations=pa.concat_tables(parts)
    observations=observations.take(pc.sort_indices(observations,sort_keys=[('frame','ascending'),('instance_index','ascending')]))
    path=base.DATASET/'tables/frames.parquet'
    digest=sha256(path); assert digest==artifacts['tables/frames.parquet']['sha256']
    table_pins[str(path.relative_to(ROOT))]=digest
    frames=pq.read_table(path,filters=[('video_id','==',base.VIDEO_ID),('frame','>=',start),('frame','<',stop)])
    frames=frames.take(pc.sort_indices(frames,sort_keys=[('frame','ascending')]))
    np.testing.assert_array_equal(frames['frame'].to_numpy(),np.arange(start,stop))
    assert set(observations['video_id'].to_pylist())=={base.VIDEO_ID}
    return observations,frames,source_state,table_pins,complete['manifest_sha256']


def build(logdir):
    assert os.environ.get('SLURM_JOB_ID'), 'Slurm only'
    assert not FINAL.exists(), 'Refusing to overwrite a published bundle'
    source_info=probe(base.VIDEO)
    packets=source_info['packets']; validate_packet_sequence(packets)
    assert len(packets)==base.TOTAL
    stream=source_info['streams'][0]
    assert (stream['width'],stream['height'],stream['r_frame_rate'],stream['time_base'])==(7680,4320,'30000/1001','1/30000')
    start=select_start(packets); stop=start+COUNT
    plan=plan_chunks(packets,start,COUNT)
    atomic_json(logdir/'CHUNK_PLAN.json',dict(source_start=start,source_stop_exclusive=stop,
        desired_start=DESIRED_START,frames=COUNT,codec=stream,intervals=plan,packet_budget=PACKET_BUDGET))
    print('PLAN',start,stop,'chunks',len(plan),flush=True)
    observations,frames,source_pins,table_pins,manifest_hash=source_tables(start,stop)
    stage=FINAL.parent/f'.build_crab_chunks_{os.environ["SLURM_JOB_ID"]}'
    stage.mkdir(exist_ok=False)
    source_cap=cv2.VideoCapture(str(base.VIDEO),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
    assert source_cap.isOpened() and source_cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    maps=[]; chunk_rows=[]; out_obs=[]; out_frames=[]; total_instances=0
    for number,(a,b) in enumerate(plan,1):
        chunk_id=f'chunk_{number:03d}'; directory=stage/chunk_id; directory.mkdir()
        name=f'{base.VIDEO_ID}.frames_{a:06d}_{b-1:06d}'
        movie=directory/f'{name}.mp4'
        command=copy_video(base.VIDEO,movie,a,b-a)
        assert movie.stat().st_size < LIMIT
        clip_info=probe(movie); validate_copy(source_info,clip_info,a,b)
        cap=cv2.VideoCapture(str(movie),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
        assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==b-a
        assert abs(cap.get(cv2.CAP_PROP_FPS)-base.FPS)<1e-8
        for local,original_frame in enumerate(range(a,b)):
            assert round(source_cap.get(cv2.CAP_PROP_POS_FRAMES))==original_frame
            ok,source_image=source_cap.read(); good,copy_image=cap.read()
            assert ok and good and source_image.shape==copy_image.shape==(base.HEIGHT,base.WIDTH,3)
            assert np.array_equal(source_image,copy_image), f'Decoded pixels differ: frame {original_frame}'
            maps.append(dict(chunk_id=chunk_id,chunk_frame=local,demo_frame=original_frame-start,
                source_frame=original_frame,chunk_time_s=local/base.FPS,demo_time_s=(original_frame-start)/base.FPS,
                source_time_s=original_frame/base.FPS,decoded_bgr_sha256=hashlib.sha256(source_image.tobytes()).hexdigest(),
                source_packet_sha256=packets[original_frame]['data_hash']))
        assert not cap.read()[0]; cap.release()
        raw=base.slice_slp(base.PREDICTIONS/'raw'/f'{base.VIDEO_ID}.predictions.slp',directory/f'{name}.predictions.slp',movie.name,a,b-a)
        tracked=base.slice_slp(base.PREDICTIONS/'tracklets'/f'{base.VIDEO_ID}.tracked.slp',directory/f'{name}.tracked.slp',movie.name,a,b-a)
        assert raw['points'].tobytes()==tracked['points'].tobytes()
        for field in raw['instances'].dtype.names:
            if field!='track': assert raw['instances'][field].tobytes()==tracked['instances'][field].tobytes()
        subset=observations.filter(pc.and_(pc.greater_equal(observations['frame'],a),pc.less(observations['frame'],b)))
        subframes=frames.filter(pc.and_(pc.greater_equal(frames['frame'],a),pc.less(frames['frame'],b)))
        table=add_indices(subset,start,a,chunk_id)
        table=table.append_column('chunk_instance_row',pa.array(np.arange(len(table))))
        table=table.append_column('demo_instance_row',pa.array(total_instances+np.arange(len(table))))
        frame_table=add_indices(subframes,start,a,chunk_id)
        base.check_observations(table,tracked,start=a,frame_column='chunk_frame')
        np.testing.assert_array_equal(subframes['source_count'].to_numpy(),raw['frames']['instance_id_end']-raw['frames']['instance_id_start'])
        assigned=np.bincount(tracked['instances']['frame_id'][tracked['instances']['track']>=0].astype(int),minlength=b-a)
        np.testing.assert_array_equal(subframes['assigned_count'].to_numpy(),assigned)
        for filename,t in [('observations.parquet',table),('frames.parquet',frame_table)]:
            pq.write_table(t,directory/filename,compression='zstd')
            assert pq.read_table(directory/filename).equals(t)
        assert table.select(subset.column_names).equals(subset)
        out_obs.append(table); out_frames.append(frame_table)
        total_instances+=len(table)
        # Independent path/cwd check, without duplicating source videos or using absolute links.
        portable=logdir/'portability'/chunk_id; portable.mkdir(parents=True)
        os.link(movie,portable/movie.name)
        for suffix,expected_tracks in [('predictions',0),('tracked',tracked['tracks'])]:
            path=directory/f'{name}.{suffix}.slp'; shutil.copy2(path,portable/path.name)
            labels=sio.load_slp(portable/path.name,open_videos=True)
            assert [lf.frame_idx for lf in labels]==list(range(b-a))
            assert sum(len(lf.instances) for lf in labels)==len(table)
            assert len(labels.tracks)==expected_tracks
            assert Path(labels.videos[0].filename).resolve()==(portable/movie.name).resolve()
            assert labels.videos[0][0].shape[:2]==(base.HEIGHT,base.WIDTH)
            labels.videos[0].close()
            del labels
        chunk_rows.append(dict(chunk_id=chunk_id,source_start=a,source_stop_exclusive=b,frames=b-a,
            demo_start=a-start,observations=len(table),assigned_observations=int(assigned.sum()),
            video=f'{chunk_id}/{movie.name}',raw_slp=f'{chunk_id}/{name}.predictions.slp',
            tracked_slp=f'{chunk_id}/{name}.tracked.slp',video_bytes=movie.stat().st_size,stream_copy_command=command))
        print('CHUNK_VERIFIED',chunk_id,a,b,'frames',b-a,'bytes',movie.stat().st_size,flush=True)
    source_cap.release()
    combined=pa.concat_tables(out_obs); combined_frames=pa.concat_tables(out_frames)
    assert combined.select(observations.column_names).equals(observations)
    assert combined_frames.select(frames.column_names).equals(frames)
    np.testing.assert_array_equal(combined_frames['demo_frame'].to_numpy(),np.arange(COUNT))
    assert len(maps)==COUNT and [m['source_frame'] for m in maps]==list(range(start,stop))
    for name,table in [('observations.parquet',combined),('frames.parquet',combined_frames)]:
        pq.write_table(table,stage/name,compression='zstd'); assert pq.read_table(stage/name).equals(table)
    with (stage/'frame_map.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(maps[0])); writer.writeheader(); writer.writerows(maps)
    atomic_json(stage/'chunks.json',dict(source_frame_start=start,source_frame_stop_exclusive=stop,
        frames=COUNT,chunks=chunk_rows))
    readme=TEMPLATE.read_text().replace('@START@',str(start)).replace('@LAST@',str(stop-1)).replace('@CHUNKS@',str(len(plan)))
    (stage/'README.md').write_text(readme)
    for rel,pin in source_pins.items():
        path=ROOT/rel; assert (path.stat().st_size,path.stat().st_mtime_ns)==(pin['bytes'],pin['mtime_ns'])
    artifacts={str(p.relative_to(stage)):dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in stage.rglob('*') if p.is_file()}
    assert all(x['bytes']<LIMIT for x in artifacts.values())
    receipt=dict(status='validated_complete',job_id=os.environ['SLURM_JOB_ID'],frames=COUNT,
        source_frame_start=start,source_frame_stop_exclusive=stop,width=base.WIDTH,height=base.HEIGHT,
        fps_numerator=30000,fps_denominator=1001,chunk_count=len(plan),observations=total_instances,
        source_dataset_manifest_sha256=manifest_hash,source_inputs=source_pins,source_table_sha256=table_pins,
        code_sha256={p:sha256(ROOT/p) for p in SOURCES},artifacts=artifacts,
        validation=dict(no_video_reencoding=True,all_encoded_packets_identical=True,
            all_2000_decoded_frames_pixel_identical=True,no_missing_or_duplicate_frames=True,
            all_pose_score_bytes_unchanged=True,dataframe_source_columns_identical=True,
            all_chunks_below_100_MiB=True,portable_slp_video_loading=True,inference_or_tracking_rerun=False),
        limitations=['Original video full hash inherited; current size/mtime checked. Selected encoded packets and decoded pixels verified.',
                     'Experimental track IDs are not validated unique-animal identities.',
                     'Local bundle completion is not proof of a GitHub push.'])
    atomic_json(stage/'DEMO_COMPLETE.json',receipt)
    stage.rename(FINAL)
    print('DEMO_COMPLETE',FINAL,'frames',COUNT,'observations',total_instances,flush=True)
    return receipt
