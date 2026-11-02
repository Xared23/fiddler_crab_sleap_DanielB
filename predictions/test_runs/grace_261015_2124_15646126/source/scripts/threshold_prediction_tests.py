"""Bounded threshold comparison: two 200-frame inferences, no saved frame chunks.

Reuses the frozen v007 model pair, SLEAP Predictor API and existing body-first
association. One final all-instance SLP per condition; IDs are optional metadata.
"""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import sleap_io as sio
import yaml

from body_first_tracklets import Frame, Tracker
from pipeline_common import ROOT, atomic_json, project_path, read_json, sha256, validate_labels
from legacy_dataset.tracklets import in_image

CONFIG = ROOT/'configs/prediction/threshold_tests_200_v001.json'
NODES = ['Shell', 'Shoulder', 'Elbow', 'Claw', 'Arm']


def frame_arrays(lf):
    """Raw measurements, including invisible/missing nodes; no score clipping."""
    n = len(lf.instances)
    xy = np.array([i.points['xy'] for i in lf], dtype=float).reshape(n, 5, 2)
    score = np.array([i.points['score'] for i in lf], dtype=float).reshape(n, 5)
    visible = np.array([i.points['visible'] for i in lf], dtype=bool).reshape(n, 5)
    return xy, score, visible


def measurement_snapshot(labels):
    result = []
    for lf in labels:
        xy, scores, visible = frame_arrays(lf)
        complete = np.array([i.points['complete'] for i in lf], dtype=bool).reshape(len(lf), 5)
        # SLP stores instance/tracking scores as float32; points/node scores are float64.
        instance_scores = np.array([i.score for i in lf], dtype=np.float32)
        tracking_scores = np.array([i.tracking_score for i in lf], dtype=np.float32)
        result.append((lf.frame_idx, xy.copy(), scores.copy(), visible.copy(), complete, instance_scores, tracking_scores))
    return result


def assert_same_measurements(before, after):
    assert len(before) == len(after)
    for a,b in zip(before,after):
        assert a[0] == b[0]
        for x,y in zip(a[1:],b[1:]):
            np.testing.assert_array_equal(x,y)


def attach_existing_tracker(labels, tracker_config, scope, video_id, width, height):
    """Assign optional IDs without removing any predicted instance or changing points."""
    before = measurement_snapshot(labels)
    cfg = copy.deepcopy(tracker_config)
    cfg['dataset_id'] = scope
    tracker = Tracker(video_id, width, height, cfg)
    catalog, records, ended = {}, [], []
    prev = {}
    for lf in labels:
        assert all(i.track is None for i in lf), 'Expected untracked input'
        xy,scores,visible = frame_arrays(lf)
        f = Frame(lf.frame_idx, xy[:,0], xy[:,1], scores[:,0], scores[:,1], visible[:,0], visible[:,1])
        result = tracker.process(f)
        assigned = result['ids'] > 0
        assert len(np.unique(result['ids'][assigned])) == int(assigned.sum())
        for instance, tid in zip(lf.instances,result['ids']):
            if tid > 0:
                tid = int(tid)
                if tid not in catalog:
                    catalog[tid] = sio.Track(name=tracker.name(tid))
                instance.track = catalog[tid]
                if tid in prev:
                    previous_frame, previous_xy = prev[tid]
                    assert previous_frame+1 == lf.frame_idx
                    assert np.linalg.norm(instance.points['xy'][0]-previous_xy) <= cfg['max_body_displacement_px']+1e-6
                prev[tid] = (lf.frame_idx,instance.points['xy'][0].copy())
        ended.extend(result['ended'])
        records.append({'frame':lf.frame_idx,'time_s':lf.frame_idx*1001/30000,
                        'raw':len(lf),'finite_in_image':int(in_image(xy[:,0],width,height).sum()),
                        'assigned':int(assigned.sum()),'unassigned':int((~assigned).sum()),
                        'reasons':dict(zip(*np.unique(result['validity']['reasons'][~assigned],return_counts=True)))})
    ended.extend(tracker.finish())
    labels.tracks = list(catalog.values())
    assert_same_measurements(before,measurement_snapshot(labels))
    assert sum(row['length'] for row in ended) == sum(r['assigned'] for r in records)
    return records, ended


def save_final(labels, target, video, start, stop):
    """Atomic whole-selection save, all frames and measurements validated on readback."""
    target = Path(target)
    if target.exists() or target.with_suffix('.tmp.slp').exists():
        raise FileExistsError(target)
    validate_labels(labels, video, start, stop, NODES)
    before = measurement_snapshot(labels)
    before_names = [[i.track.name if i.track else None for i in lf] for lf in labels]
    tmp = target.with_suffix('.tmp.slp')
    labels.save(str(tmp), embed=False, format='slp')
    check = sio.load_slp(tmp, open_videos=False)
    validate_labels(check,video,start,stop,NODES)
    assert_same_measurements(before,measurement_snapshot(check))
    assert before_names == [[i.track.name if i.track else None for i in lf] for lf in check]
    assert len(check.tracks) == len(labels.tracks)
    tmp.rename(target)
    return check


def spatial_matches(base_xy, new_xy, radius=5.):
    """One-to-one gated assignment; unmatched positions are NOT proven new animals."""
    base_valid = np.flatnonzero(np.isfinite(base_xy).all(axis=1))
    new_valid = np.flatnonzero(np.isfinite(new_xy).all(axis=1))
    if not len(base_valid) or not len(new_valid):
        return [], base_valid.tolist(), new_valid.tolist()
    distances = cdist(base_xy[base_valid],new_xy[new_valid])
    penalty = (max(distances.shape)+1)*(radius+1)
    rows,cols = linear_sum_assignment(np.where(distances<=radius,distances,penalty))
    matches = [(int(base_valid[i]),int(new_valid[j]),float(distances[i,j]))
               for i,j in zip(rows,cols) if distances[i,j]<=radius]
    used_base={i for i,j,d in matches}; used_new={j for i,j,d in matches}
    return matches,[int(i) for i in base_valid if i not in used_base],[int(j) for j in new_valid if j not in used_new]


def inference(predictor, video, threshold, start, stop):
    if threshold not in (.1,.05):
        raise ValueError('Only the two requested new thresholds are permitted')
    return predictor.predict(str(video), frames=list(range(start,stop)), make_labels=True,
                             clean_empty_frames=False, peak_threshold=.2,
                             centroid_threshold=threshold, keypoint_threshold=.2,
                             max_instances=None, return_confmaps=False, return_crops=False)


def verify_inputs(cfg):
    old = read_json(project_path(cfg['baseline_manifest']))
    baseline = read_json(project_path(cfg['baseline_config']))
    assert old['config'] == baseline
    assert cfg['centroid_thresholds'] == [.1,.05] and cfg['keypoint_threshold'] == .2
    assert (cfg['start_frame'],cfg['stop_frame_exclusive']) == (0,200)
    assert cfg['max_instances'] is None and cfg['persist_frame_chunks'] is False
    assert cfg['video_id'] == 'Crab_Cam_2_2' and baseline['videos'][cfg['video_id']] == cfg['video']
    assert project_path(cfg['output_root']) == ROOT/'predictions'/'test runs'
    pins = {}
    names=[cfg['baseline_config'],baseline['tracker_config'],'scripts/body_first_tracklets.py',
           'scripts/legacy_dataset/tracklets.py']
    names += [f"{baseline[role]}/{filename}" for role in ('centroid','centered')
              for filename in ('best.ckpt','training_config.yaml','snapshot.json')]
    for name in names:
        digest=sha256(project_path(name)); assert digest==old['hashes'][name],name
        pins[name]=digest
    versions={name:importlib.metadata.version(name) for name in old['versions']}
    assert versions==old['versions'], 'Environment versions differ from the existing baseline'
    video=project_path(cfg['video']); stat=video.stat(); expected=old['videos'][cfg['video_id']]
    assert (stat.st_size,stat.st_mtime_ns)==(expected['bytes'],expected['mtime_ns'])
    dataset=project_path(cfg['dataset']); complete=read_json(dataset/'DATASET_COMPLETE.json')
    assert complete['status']=='validated_complete'
    assert sha256(dataset/'manifest.json')==complete['manifest_sha256']
    manifest=read_json(dataset/'manifest.json')
    source_pin=next(p for p in manifest['source_pins'] if p['path']==cfg['baseline_raw_slp'])
    pins[cfg['baseline_raw_slp']]=sha256(project_path(cfg['baseline_raw_slp']))
    assert pins[cfg['baseline_raw_slp']]==source_pin['sha256']
    return baseline,pins,versions,{'path':cfg['video'],'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                 'verification':'size/mtime checked against baseline; full video not rehashed'}


def run_tests(directory):
    sys.path.insert(0,str(ROOT/'tests'))
    import test_body_first_tracklets
    import test_threshold_prediction_tests
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(test_body_first_tracklets),
                             unittest.defaultTestLoader.loadTestsFromModule(test_threshold_prediction_tests)])
    with (directory/'tests.txt').open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    atomic_json(directory/'TESTS.json',{'passed':result.wasSuccessful(),'tests_run':result.testsRun,
                                       'failures':len(result.failures),'errors':len(result.errors)})
    if not result.wasSuccessful():
        raise RuntimeError('Tests failed; no real inference started. See tests.txt')
    print('TESTS_PASSED',result.testsRun,flush=True)


def compare(conditions, records, directory, cfg):
    base=conditions[.2]
    comparisons=[]; unmatched={}
    for threshold in (.1,.05):
        unmatched[threshold]={}
        for lf0,lf in zip(base,conditions[threshold]):
            assert lf0.frame_idx==lf.frame_idx
            xy0,_,_=frame_arrays(lf0); xy,_,_=frame_arrays(lf)
            # Out-of-image positions are excluded, as well as missing positions.
            p0=xy0[:,0].copy(); p=xy[:,0].copy()
            p0[~in_image(p0,7680,4320)]=np.nan; p[~in_image(p,7680,4320)]=np.nan
            matches,missing,added=spatial_matches(p0,p,cfg['baseline_matching_radius_px'])
            unmatched[threshold][lf.frame_idx]=added
            comparisons.append({'centroid_threshold':threshold,'frame':lf.frame_idx,
                'matched_baseline_positions':len(matches),'baseline_positions_unmatched':len(missing),
                'new_positions_unmatched':len(added),'mean_match_distance_px':np.mean([v[2] for v in matches]) if matches else np.nan})
    comp=pd.DataFrame(comparisons); comp.to_csv(directory/'baseline_spatial_comparison.csv',index=False)
    flat=[]
    for threshold,rr in records.items():
        flat.extend(dict(centroid_threshold=threshold,**{k:v for k,v in r.items() if k!='reasons'}) for r in rr)
    counts=pd.DataFrame(flat); counts.to_csv(directory/'counts_per_frame.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True,layout='constrained')
    for threshold,color in [(.2,'#333333'),(.1,'#1675ac'),(.05,'#c85322')]:
        z=counts[counts.centroid_threshold==threshold]
        label=f'centroid {threshold:g}; landmarks 0.2'+(' (existing)' if threshold==.2 else '')
        axes[0].plot(z.time_s,z.raw,color=color,label=label)
        axes[1].plot(z.time_s,z.assigned,color=color,label=label)
        if threshold!=.2:
            q=comp[comp.centroid_threshold==threshold]
            axes[2].plot(q.frame*1001/30000,q.new_positions_unmatched,color=color,label=label)
    axes[0].set_ylabel('All pose detections / frame')
    axes[1].set_ylabel('Detections assigned an ID / frame')
    axes[2].set_ylabel('Unmatched in-image Shell positions')
    axes[2].set_xlabel('Nominal video time (s)')
    for ax in axes:
        ax.set_ylim(bottom=0); ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('200-frame detection-threshold comparison; no fixed count or retraining')
    fig.savefig(directory/'threshold_comparison.png',dpi=160)
    fig.savefig(directory/'threshold_comparison.pdf')
    plt.close(fig)
    return counts,comp,unmatched


def previews(conditions, unmatched, directory, cfg):
    import cv2
    import matplotlib.pyplot as plt
    capture=cv2.VideoCapture(str(project_path(cfg['video'])))
    assert capture.isOpened()
    assert round(capture.get(cv2.CAP_PROP_FRAME_COUNT))==28800
    assert (round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))==(7680,4320)
    evidence=[]
    for frame in cfg['review_frames']:
        assert capture.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,bgr=capture.read(); assert ok
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        lf=conditions[.05][frame]; xy,score,_=frame_arrays(lf)
        added=unmatched[.05][frame]
        good=[j for j in added if lf.instances[j].track is not None]
        pool=good or added
        if pool:
            chosen=max(pool,key=lambda j:float(np.nan_to_num(score[j,0],nan=-1)))
            center=xy[chosen,0]
        else:
            chosen=None; center=np.array([4000,2200])
        x0=int(np.clip(center[0]-600,0,7680-1200)); y0=int(np.clip(center[1]-450,0,4320-900))
        fig,axs=plt.subplots(2,2,figsize=(14,10.5),layout='constrained')
        for ax,threshold in zip(axs.flat,[None,.2,.1,.05]):
            ax.imshow(rgb[y0:y0+900,x0:x0+1200],extent=(x0,x0+1200,y0+900,y0))
            ax.set_xlim(x0,x0+1200); ax.set_ylim(y0+900,y0)
            if threshold is not None:
                current=conditions[threshold][frame]
                extra=set(unmatched.get(threshold,{}).get(frame,[]))
                for j,instance in enumerate(current):
                    p=instance.points['xy']; vis=instance.points['visible'] & np.isfinite(p).all(axis=1)
                    color='lime' if instance.track is not None else 'orange'
                    for a,b in [(0,1),(0,4),(1,2),(2,3)]:
                        if vis[a] and vis[b]: ax.plot(p[[a,b],0],p[[a,b],1],color=color,lw=.8)
                    ax.scatter(p[vis,0],p[vis,1],s=7,color=color)
                    if vis[0] and j in extra:
                        ax.scatter(*p[0],s=65,facecolors='none',edgecolors='magenta',marker='s',linewidths=1)
                ax.set_title(f'centroid {threshold:g}, landmarks 0.2 | whole-frame N={len(current)}')
            else:
                ax.set_title('Original image; no overlays')
            ax.set_xlabel('original x (px)'); ax.set_ylabel('original y (px)')
        fig.suptitle(f"{cfg['video_id']} frame {frame}: selected added-candidate review\nGreen: ID assigned; orange: unassigned; magenta box: >5 px unmatched to baseline")
        filename=f'review_frame_{frame:06d}.png'
        fig.savefig(directory/filename,dpi=140); plt.close(fig)
        evidence.append({'frame':frame,'crop_xywh':[x0,y0,1200,900],'selected_detection_index_c050':chosen,
                         'selection':'highest Shell score among unmatched c050 assigned detections; fallback any unmatched then fixed center',
                         'file':filename,'limitation':'Deliberately selected candidate, not random precision/recall validation'})
    capture.release()
    atomic_json(directory/'preview_selection.json',evidence)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run only in Slurm'
    started=time.monotonic(); cfg=read_json(CONFIG)
    stamp=datetime.now(timezone.utc).strftime('%y%m%d_%H%M')
    run_id=f"threshold_{stamp}_{os.environ['SLURM_JOB_ID']}"
    directory=project_path(cfg['output_root'])/run_id
    directory.mkdir(parents=True,exist_ok=False)
    os.environ['TMPDIR']=str(directory/'test_tmp'); Path(os.environ['TMPDIR']).mkdir()
    print('RUN_DIRECTORY',directory,flush=True)
    run_tests(directory)
    baseline,pins,versions,video_pin=verify_inputs(cfg)
    code_paths=[Path(__file__),ROOT/'tests/test_threshold_prediction_tests.py',ROOT/'tests/test_body_first_tracklets.py',
                ROOT/'jobs/predict_threshold_tests.sh',CONFIG]
    snap=directory/'source'; snap.mkdir()
    for path in code_paths:
        shutil.copyfile(path,snap/path.name); pins[str(path.relative_to(ROOT))]=sha256(path)
    atomic_json(directory/'RUN_MANIFEST.json',{'run_id':run_id,'timestamp_utc':stamp,'config':cfg,
                'baseline_model_config':baseline,'verified_inputs_sha256':pins,'versions':versions,
                'video':video_pin,'job_id':os.environ['SLURM_JOB_ID'],
                'baseline_source':'existing raw v007 predictions; no baseline inference',
                'persisted_prediction_chunks':False,'prediction_files':'one all-instance, optionally tracked final SLP per condition'})
    video=project_path(cfg['video']); start,stop=cfg['start_frame'],cfg['stop_frame_exclusive']
    original=sio.load_slp(project_path(cfg['baseline_raw_slp']),open_videos=False,lazy=True)
    subset=sio.Labels(labeled_frames=[original[k] for k in range(start,stop)],
                      videos=list(original.videos),skeletons=list(original.skeletons))
    validate_labels(subset,video,start,stop,NODES)
    tracker_cfg=yaml.safe_load(project_path(baseline['tracker_config']).read_text())
    conditions={}; records={}; summary={}
    def publish(labels, threshold, origin):
        tag=f'c{round(threshold*1000):03d}'
        before=sum(len(lf) for lf in labels)
        rr,tracks=attach_existing_tracker(labels,tracker_cfg,f'{run_id}/{tag}',cfg['video_id'],7680,4320)
        labels.provenance=dict(labels.provenance or {}, threshold_test={'run_id':run_id,
            'centroid_threshold':threshold,'keypoint_threshold':.2,'source':origin,
            'frame_start':start,'frame_stop_exclusive':stop,'max_instances':None,
            'tracking':'unchanged_body_first_v001_no_gaps_no_count_cap',
            'all_detections_retained':True,'identity_status':'experimental_not_ground_truth_validated'})
        path=directory/f"{cfg['video_id']}_{stamp}_{tag}.predictions.slp"
        conditions[threshold]=save_final(labels,path,video,start,stop)
        assert before==sum(len(lf) for lf in conditions[threshold])
        records[threshold]=rr
        reasons={}
        for r in rr:
            for k,v in r['reasons'].items(): reasons[str(k)]=reasons.get(str(k),0)+int(v)
        summary[tag]={'centroid_threshold':threshold,'keypoint_threshold':.2,'source':origin,
            'frames':len(rr),'raw_detections':before,'mean_raw_per_frame':before/len(rr),
            'mean_in_image_per_frame':sum(r['finite_in_image'] for r in rr)/len(rr),
            'assigned_observations':sum(r['assigned'] for r in rr),
            'mean_assigned_per_frame':sum(r['assigned'] for r in rr)/len(rr),
            'tracklets':len(tracks),'unassigned_reasons':reasons,
            'filename':path.name,'sha256':sha256(path),'measurement_readback_passed':True}
        atomic_json(directory/f'{tag}_COMPLETE.json',summary[tag])
        print('CONDITION_COMPLETE',tag,summary[tag],flush=True)
    publish(subset,.2,'existing baseline excerpt; identities reapplied with unchanged algorithm')
    # Baseline count agreement ensures the in-memory association agrees with the existing export.
    frame_reference=pd.read_parquet(project_path(cfg['dataset'])/'tables/frames.parquet')
    ref=frame_reference[(frame_reference.video_id==cfg['video_id'])&(frame_reference.frame<stop)].sort_values('frame')
    np.testing.assert_array_equal(ref.source_count,[r['raw'] for r in records[.2]])
    np.testing.assert_array_equal(ref.assigned_count,[r['assigned'] for r in records[.2]])
    del original,subset; gc.collect()
    from sleap_nn.inference.predictor import Predictor
    import torch
    assert torch.cuda.is_available(), 'GPU allocation required'
    predictor=Predictor.from_model_paths([str(project_path(baseline[role])) for role in ('centroid','centered')],
                device='cuda',batch_size=1,peak_threshold=.2,max_instances=None,
                tracker_config=None,filter_config=None)
    effective=[]
    for threshold in cfg['centroid_thresholds']:
        layer=predictor._scoped_postprocess_layer(peak_threshold=.2,centroid_threshold=threshold,keypoint_threshold=.2)
        assert layer.centroid_layer.postprocess_config.peak_threshold==threshold
        assert layer.centered_instance_layer.postprocess_config.peak_threshold==.2
        assert layer.centroid_layer.max_instances is None
        assert layer.centroid_layer.postprocess_config.max_instances is None
        effective.append({'centroid_threshold':threshold,'keypoint_threshold':.2,'max_instances':None})
        print('INFERENCE_BEGIN',threshold,'frames',start,stop,flush=True)
        labels=inference(predictor,video,threshold,start,stop)
        publish(labels,threshold,'new inference')
        del labels,layer; gc.collect(); torch.cuda.empty_cache()
    atomic_json(directory/'effective_thresholds.json',effective)
    del predictor; gc.collect(); torch.cuda.empty_cache()
    counts,comparison,unmatched=compare(conditions,records,directory,cfg)
    for tag in ['c100','c050']:
        threshold=summary[tag]['centroid_threshold']; q=comparison[comparison.centroid_threshold==threshold]
        summary[tag]['mean_unmatched_new_in_image_per_frame']=float(q.new_positions_unmatched.mean())
        summary[tag]['mean_unmatched_baseline_per_frame']=float(q.baseline_positions_unmatched.mean())
    atomic_json(directory/'summary.json',summary)
    previews(conditions,unmatched,directory,cfg)
    lines=['# 200-frame centroid threshold comparison','',f'Run: `{run_id}`; video {cfg["video_id"]}, frames 0-199.',
           'Same frozen model pair; no training. Landmark threshold remains 0.2. No detection/track count cap.',
           'One final SLP per condition, with all detections and optional body-first IDs; no frame chunks.',
           '', '| Centroid threshold | Mean raw/frame | Mean assigned/frame | Tracklets |', '|---|---:|---:|---:|']
    for tag in ['c200','c100','c050']:
        s=summary[tag]; lines.append(f"| {s['centroid_threshold']} | {s['mean_raw_per_frame']:.2f} | {s['mean_assigned_per_frame']:.2f} | {s['tracklets']} |")
    lines.extend(['','Threshold 0.2 is extracted from existing predictions, not rerun. Raw counts are not validated animal counts.',
                  'Tracking retains the existing Shell score >=0.5 and duplicate/ambiguity rules; added poses may remain unassigned.',
                  'Tracking cannot create a missed detection or maintain identity through an unobserved burrow/occlusion interval.',
                  'Spatial matching is one-to-one within 5 px; unmatched detections are candidates, not proven recovered males.',
                  'Review PNGs deliberately focus on high-score added candidates and do not estimate precision or recall.',
                  'All SLPs reference the original full video and original frame numbers. Do not relink them to a resized/cropped clip.',
                  'The primary review plots are threshold_comparison.png and review_frame_*.png.'])
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n')
    assert not list(directory.glob('*.frames_*.slp'))
    assert len(list(directory.glob('*.predictions.slp')))==3
    assert not list(directory.glob('*.tmp.slp'))
    artifacts=[{'path':str(p.relative_to(directory)),'bytes':p.stat().st_size,'sha256':sha256(p)}
               for p in sorted(directory.rglob('*')) if p.is_file()]
    atomic_json(directory/'COMPLETE.json',{'status':'complete','job_id':os.environ['SLURM_JOB_ID'],
                'elapsed_s':time.monotonic()-started,'new_prediction_frames':400,'baseline_frames_extracted':200,
                'no_frame_chunks_saved':True,'artifacts':artifacts})
    print('COMPLETE',directory,flush=True)


if __name__=='__main__':
    main()
