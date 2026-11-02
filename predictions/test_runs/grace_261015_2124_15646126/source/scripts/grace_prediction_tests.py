"""Two bounded 1000-frame hysteresis/grace tests, one shared frozen-model pass."""
from __future__ import annotations
from datetime import datetime, timezone
import gc
import importlib.metadata
import os
from pathlib import Path
import shutil
import sys
import time
import unittest

import numpy as np
import pandas as pd
import sleap_io as sio
import yaml

from body_first_tracklets import Frame
from grace_tracklets import GraceTracker
from pipeline_common import ROOT, atomic_json, project_path, read_json, sha256, validate_labels
from threshold_prediction_tests import (NODES, frame_arrays, save_final,
    attach_existing_tracker, measurement_snapshot, assert_same_measurements)

CONFIG = ROOT/'configs/prediction/grace_tests_1000_v001.json'


def clone_instance(instance, track=None):
    return sio.PredictedInstance(points=instance.points.copy(), skeleton=instance.skeleton,
        score=instance.score, tracking_score=instance.tracking_score, track=track)


def unpack_output(output, skeleton, expected_frame):
    """Explicitly verify centroid confidence; never substitute a landmark score."""
    out = output.slim()
    assert out.batch_size == 1
    assert out.frame_indices.tolist() == [expected_frame]
    data = out.numpy()
    xy = data['pred_keypoints'][0]
    node_scores = data['pred_peak_values'][0]
    centroids = data['pred_centroids'][0]
    scores = data['pred_centroid_values'][0]
    np.testing.assert_array_equal(data['instance_scores'][0], scores)
    assert xy.shape == (len(scores), 5, 2) and node_scores.shape == (len(scores), 5)
    assert np.isfinite(scores).all() and (scores >= .05).all()
    assert np.isfinite(centroids).all()
    slots = np.flatnonzero(~np.isnan(xy).all(axis=(1, 2)))
    instances = out.to_instances(skeleton)
    assert len(instances) == len(slots)
    for inst, slot in zip(instances, slots):
        np.testing.assert_array_equal(inst.points['xy'], xy[slot])
        np.testing.assert_array_equal(inst.points['score'], node_scores[slot])
        assert inst.score == float(scores[slot])
    raw = dict(xy=xy, landmark_scores=node_scores, centroid_xy=centroids,
               centroid_scores=scores, instance_scores=data['instance_scores'][0],
               frame=np.full(len(scores), expected_frame, dtype=np.int64),
               slot=np.arange(len(scores), dtype=np.int64))
    return instances, scores[slots].copy(), slots, raw


def infer(predictor, video):
    return predictor.predict_streaming(str(video), frames=list(range(1000)),
        peak_threshold=.2, centroid_threshold=.05, keypoint_threshold=.2,
        max_instances=None, return_confmaps=False, return_crops=False)


def apply_condition(raw, centroid_scores, slot_indices, cfg, tracker_cfg, scope, birth):
    if not (len(raw) == len(centroid_scores) == len(slot_indices)):
        raise ValueError('Frame/score/slot coverage differs')
    tracker = GraceTracker(tracker_cfg, birth, cfg['continuation_threshold'], cfg['grace_frames'])
    catalog, frames, counts, audit, gaps, tracks = {}, [], [], [], [], []
    previous = {}
    for lf, scores, slots in zip(raw, centroid_scores, slot_indices):
        assert len(lf) == len(scores) == len(slots)
        xy, node_scores, visible = frame_arrays(lf)
        f = Frame(lf.frame_idx, xy[:, 0], xy[:, 1], node_scores[:, 0], node_scores[:, 1], visible[:, 0], visible[:, 1])
        result = tracker.process(f, scores)
        instances = []
        for j, (inst, tid, keep) in enumerate(zip(lf, result['ids'], result['keep'])):
            tid = int(tid)
            if tid > 0:
                if tid not in catalog:
                    assert scores[j] >= birth
                    catalog[tid] = sio.Track(name=f'{scope}/t{tid:09d}')
                else:
                    prior_frame, prior_xy = previous[tid]
                    dt = lf.frame_idx-prior_frame
                    assert 1 <= dt <= cfg['grace_frames']+1
                    assert result['previous_frame'][j] == prior_frame
                    assert np.linalg.norm(xy[j, 0]-prior_xy) <= tracker_cfg['max_body_displacement_px']*dt+1e-6
                    if dt > 1:
                        gaps.append(dict(track_id=tid, previous_frame=prior_frame, frame=lf.frame_idx,
                            first_missing_frame=prior_frame+1, last_missing_frame=lf.frame_idx-1,
                            missed_frames=dt-1))
                previous[tid] = (lf.frame_idx, xy[j, 0].copy())
                assert keep and scores[j] >= cfg['continuation_threshold']
            if keep:
                instances.append(clone_instance(inst, catalog.get(tid)))
            audit.append(dict(frame=lf.frame_idx, raw_pose_index=j, raw_slot=int(slots[j]),
                centroid_score=float(scores[j]), shell_score=float(node_scores[j, 0]),
                retained=bool(keep), track_id=tid, reason=str(result['reason'][j]),
                body_rejection=str(result['validity']['reasons'][j]),
                previous_frame=int(result['previous_frame'][j]), link_cost=float(result['link_cost'][j])))
        frames.append(sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx, instances=instances))
        assigned = result['ids'] > 0
        counts.append(dict(frame=lf.frame_idx, time_s=lf.frame_idx*1001/30000,
            raw_candidate_poses=len(lf), static_threshold_poses=int((scores >= birth).sum()),
            retained=len(instances), assigned=int(assigned.sum()),
            low_score_continuations=int((assigned & (scores < birth)).sum()),
            gap_links=int((result['reason'] == 'gap_link').sum()),
            pending_unobserved_tracks=result['pending_tracks']))
        tracks.extend(result['ended'])
    tracks.extend(tracker.finish())
    assert sum(t['observations'] for t in tracks) == sum(c['assigned'] for c in counts)
    assert sum(t['missing_frames'] for t in tracks) == sum(g['missed_frames'] for g in gaps)
    labels = sio.Labels(labeled_frames=frames, videos=list(raw.videos),
                        skeletons=list(raw.skeletons), tracks=list(catalog.values()))
    return labels, pd.DataFrame(counts), pd.DataFrame(audit), pd.DataFrame(tracks, columns=[
        'track_id','start_frame','end_frame','observations','span_frames','missing_frames','termination']), pd.DataFrame(
        gaps, columns=['track_id','previous_frame','frame','first_missing_frame','last_missing_frame','missed_frames'])


def control(raw, scores, threshold, tracker_cfg, scope):
    frames = [sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx,
        instances=[clone_instance(i) for i, s in zip(lf, ss) if s >= threshold])
        for lf, ss in zip(raw, scores)]
    labels = sio.Labels(labeled_frames=frames, videos=list(raw.videos), skeletons=list(raw.skeletons))
    records, tracks = attach_existing_tracker(labels, tracker_cfg, scope, 'Crab_Cam_2_2', 7680, 4320)
    return pd.DataFrame([{k:v for k,v in row.items() if k != 'reasons'} for row in records]), pd.DataFrame(tracks)


def summarize(counts, tracks, control_counts, control_tracks):
    delta = np.diff(counts.retained.to_numpy())
    control_delta = np.diff(control_counts.raw.to_numpy())
    return dict(frames=len(counts), mean_retained=float(counts.retained.mean()),
        mean_assigned=float(counts.assigned.mean()), track_ids=len(tracks),
        low_score_continuations=int(counts.low_score_continuations.sum()), gap_links=int(counts.gap_links.sum()),
        mean_abs_count_change=float(np.abs(delta).mean()), variance_abs_count_change=float(np.abs(delta).var()),
        control_mean_raw=float(control_counts.raw.mean()), control_mean_assigned=float(control_counts.assigned.mean()),
        control_track_ids=len(control_tracks), control_mean_abs_count_change=float(np.abs(control_delta).mean()),
        observation_median_per_id=float(tracks.observations.median()) if len(tracks) else None,
        control_observation_median_per_id=float(control_tracks.length.median()) if len(control_tracks) else None,
        identities_accuracy_validated=False)


def verify_inputs(cfg):
    assert cfg['birth_thresholds'] == [.2, .1] and cfg['continuation_threshold'] == .05
    assert cfg['keypoint_threshold'] == .2 and cfg['grace_frames'] == 5
    assert (cfg['start_frame'], cfg['stop_frame_exclusive']) == (0, 1000)
    assert cfg['video_id'] == 'Crab_Cam_2_2' and cfg['batch_size'] == 1
    assert cfg['max_instances'] is None and not cfg['persist_frame_chunks']
    assert project_path(cfg['output_root']) == ROOT/'predictions/test runs'
    baseline = read_json(project_path(cfg['baseline_config']))
    old = read_json(project_path(cfg['baseline_manifest']))
    assert old['config'] == baseline
    assert cfg['video'] == baseline['videos'][cfg['video_id']]
    names = [cfg['baseline_config'], baseline['tracker_config'], 'scripts/body_first_tracklets.py',
             'scripts/legacy_dataset/tracklets.py']
    names += [f'{baseline[role]}/{file}' for role in ('centroid','centered')
              for file in ('best.ckpt','training_config.yaml','snapshot.json')]
    pins = {name: sha256(project_path(name)) for name in names}
    for name, digest in pins.items():
        assert digest == old['hashes'][name], name
    versions = {name: importlib.metadata.version(name) for name in old['versions']}
    assert versions == old['versions']
    stat = project_path(cfg['video']).stat()
    video_pin = old['videos'][cfg['video_id']]
    assert (stat.st_size, stat.st_mtime_ns) == (video_pin['bytes'], video_pin['mtime_ns'])
    return baseline, pins, versions, video_pin


def run_tests(directory):
    sys.path.insert(0, str(ROOT/'tests'))
    modules = ['test_body_first_tracklets', 'test_threshold_prediction_tests', 'test_grace_prediction_tests']
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    with (directory/'tests.txt').open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    atomic_json(directory/'TESTS.json', dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors)))
    if not result.wasSuccessful():
        raise RuntimeError('Test gate failed; real inference was not started. See tests.txt.')
    print('TESTS_PASSED', result.testsRun, flush=True)


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Execution requires Slurm'
    started = time.monotonic()
    cfg = read_json(CONFIG)
    stamp = datetime.now(timezone.utc).strftime('%y%m%d_%H%M')
    run_id = f'grace_{stamp}_{os.environ["SLURM_JOB_ID"]}'
    directory = project_path(cfg['output_root'])/run_id
    directory.mkdir(parents=True, exist_ok=False)
    os.environ['TMPDIR'] = str(directory/'test_tmp')
    Path(os.environ['TMPDIR']).mkdir()
    print('RUN_DIRECTORY', directory, flush=True)
    source = directory/'source'; source.mkdir()
    sources = ['scripts/grace_prediction_tests.py', 'scripts/grace_tracklets.py',
        'scripts/threshold_prediction_tests.py', 'scripts/body_first_tracklets.py',
        'scripts/legacy_dataset/tracklets.py', 'scripts/pipeline_common.py',
        'tests/test_grace_prediction_tests.py', 'tests/test_threshold_prediction_tests.py',
        'tests/test_body_first_tracklets.py', 'jobs/predict_grace_tests.sh', str(CONFIG.relative_to(ROOT))]
    for name in sources:
        dest = source/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, dest)
    run_tests(directory)
    baseline, pins, versions, video_pin = verify_inputs(cfg)
    pins.update({name: sha256(ROOT/name) for name in sources})
    tracker_cfg = yaml.safe_load(project_path(baseline['tracker_config']).read_text())
    atomic_json(directory/'RUN_MANIFEST.json', dict(run_id=run_id, config=cfg, baseline_model_config=baseline,
        verified_inputs_sha256=pins, versions=versions, video_stat_pin=video_pin,
        video_verification='size/mtime against production manifest; not a full video hash',
        original_tracker_config=tracker_cfg,
        effective_tracker_overrides=dict(no_gap_bridging=False, grace_frames=5,
            reidentification_after_expiry=False, ambiguous_existing_candidate_birth=False),
        job_id=os.environ['SLURM_JOB_ID'],
        inference_passes=1, inference_frames=1000, independent_tracking_conditions=2))
    import torch
    from sleap_nn.inference.predictor import Predictor
    assert torch.cuda.is_available(), 'GPU allocation required'
    predictor = Predictor.from_model_paths([str(project_path(baseline[r])) for r in ('centroid','centered')],
        device='cuda', batch_size=1, peak_threshold=.2, max_instances=None, tracker_config=None, filter_config=None)
    layer = predictor._scoped_postprocess_layer(peak_threshold=.2, centroid_threshold=.05, keypoint_threshold=.2)
    assert layer.centroid_layer.postprocess_config.peak_threshold == .05
    assert layer.centered_instance_layer.postprocess_config.peak_threshold == .2
    assert layer.centroid_layer.max_instances is None and layer.centroid_layer.postprocess_config.max_instances is None
    assert not layer.centroid_nms
    skeleton = predictor.skeleton
    assert skeleton.node_names == NODES
    video_path = project_path(cfg['video'])
    video = sio.load_video(video_path)
    assert video.shape[:3] == (28800, 4320, 7680), video.shape
    frames, scores, slots, raw_arrays = [], [], [], []
    print('INFERENCE_BEGIN: shared 0.05 candidates, frames 0-999', flush=True)
    with torch.inference_mode():
        for index, output in enumerate(infer(predictor, video_path)):
            assert index < 1000
            instances, sc, sl, arrays = unpack_output(output, skeleton, index)
            frames.append(sio.LabeledFrame(video=video, frame_idx=index, instances=instances))
            scores.append(sc); slots.append(sl); raw_arrays.append(arrays)
            if (index+1) % 100 == 0:
                print('INFERENCE_PROGRESS', index+1, '/1000', flush=True)
    assert len(frames) == 1000
    raw = sio.Labels(labeled_frames=frames, videos=[video], skeletons=[skeleton])
    validate_labels(raw, video_path, 0, 1000, NODES)
    arrays = {key: np.concatenate([a[key] for a in raw_arrays], axis=0) for key in raw_arrays[0]}
    arrays['selected_frames'] = np.arange(1000)
    arrays['node_names'] = np.array(NODES)
    np.savez_compressed(directory/'raw_candidates.npz', **arrays)
    with np.load(directory/'raw_candidates.npz', allow_pickle=False) as check:
        for key, value in arrays.items():
            np.testing.assert_array_equal(check[key], value)
    del raw_arrays, arrays, output, predictor, layer
    gc.collect(); torch.cuda.empty_cache()
    snapshot = measurement_snapshot(raw)
    summary, all_counts = {}, []
    for birth in cfg['birth_thresholds']:
        tag = f'b{round(birth*1000):03d}_c050_g5'
        labels, counts, audit, tracks, gaps = apply_condition(raw, scores, slots, cfg, tracker_cfg, f'{run_id}/{tag}', birth)
        labels.provenance = dict(grace_test=dict(run_id=run_id, birth_centroid_threshold=birth,
            continuation_centroid_threshold=.05, keypoint_threshold=.2, grace_frames=5,
            missing_positions_interpolated=False, identity_status='experimental_unvalidated',
            raw_candidates='raw_candidates.npz', selection_policy=cfg['final_slp_policy']))
        target = directory/f'{cfg["video_id"]}_{stamp}_{tag}.predictions.slp'
        check = save_final(labels, target, video_path, 0, 1000)
        assert_same_measurements(snapshot, measurement_snapshot(raw))
        assert all(i.track is None for lf in raw for i in lf)
        counts.to_csv(directory/f'{tag}_counts.csv', index=False)
        audit.to_parquet(directory/f'{tag}_assignments.parquet', index=False)
        tracks.to_csv(directory/f'{tag}_tracks.csv', index=False)
        gaps.to_csv(directory/f'{tag}_gaps.csv', index=False)
        control_counts, control_tracks = control(raw, scores, birth, tracker_cfg, f'{run_id}/{tag}/control')
        np.testing.assert_array_equal(control_counts.raw, counts.static_threshold_poses)
        assert (counts.retained >= control_counts.raw).all()
        assert (counts.retained <= counts.raw_candidate_poses).all()
        control_counts.to_csv(directory/f'{tag}_control_counts.csv', index=False)
        control_tracks.to_csv(directory/f'{tag}_control_tracks.csv', index=False)
        summary[tag] = dict(summarize(counts, tracks, control_counts, control_tracks),
            birth_threshold=birth, filename=target.name, sha256=sha256(target),
            measurement_roundtrip_passed=True)
        atomic_json(directory/f'{tag}_COMPLETE.json', summary[tag])
        counts['control_raw'] = control_counts.raw
        counts['control_assigned'] = control_counts.assigned
        all_counts.append((birth, counts))
        print('CONDITION_COMPLETE', tag, summary[tag], flush=True)
        del labels, check, audit; gc.collect()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, layout='constrained')
    for col, (birth, counts) in enumerate(all_counts):
        for row, (metric, base) in enumerate([('retained','control_raw'), ('assigned','control_assigned')]):
            ax = axes[row, col]
            ax.plot(counts.time_s, counts[base], color='0.5', lw=.8, label='Static threshold; adjacent-only IDs')
            ax.plot(counts.time_s, counts[metric], color='#147aa3', lw=.8, label='0.05 continuation; 5 missed frames')
            ax.set_ylabel('Observed poses' if row == 0 else 'Observed poses with IDs')
            ax.set_ylim(bottom=0); ax.grid(alpha=.2)
            ax.set_title(f'New-track centroid threshold {birth:g}')
            if row == 0: ax.legend(fontsize=8)
            else: ax.set_xlabel('Nominal video time (s)')
    fig.suptitle('Cam 2_2 frames 0-999: missing positions are not counted or filled')
    fig.savefig(directory/'comparison.png', dpi=160)
    plt.close(fig)
    atomic_json(directory/'summary.json', summary)
    lines = ['# 1000-frame continuation/grace tests', '', f'Run `{run_id}`; original Cam 2_2 frames 0-999.',
        'One shared inference pass at centroid 0.05; two independent trackers. Frozen models; no training.',
        'Landmarks 0.2; Shell identity gate 0.5; variable counts; no interpolation or count cap.',
        'SLPs keep high-threshold poses (including unassigned) plus successful low-threshold continuations.',
        'All raw centroid slots and pose measurements are retained in raw_candidates.npz; no chunk SLPs.',
        'Gaps CSVs explicitly list missing frames. Pending identities are memory, not observations.', '',
        '| Birth threshold | Mean observed poses | Mean assigned | Track IDs | Low-score continuations | Gap links |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary.values():
        lines.append(f'| {s["birth_threshold"]} | {s["mean_retained"]:.3f} | {s["mean_assigned"]:.3f} | {s["track_ids"]} | {s["low_score_continuations"]} | {s["gap_links"]} |')
    lines += ['', 'Controls use the same candidate pool, filtered at each birth threshold, with the unchanged adjacent-frame tracker.',
        'Control counts are not a new independent detector run. Count fluctuations and longer tracks do not establish identity accuracy.',
        'No visual ground truth has been collected; false links, false positives and missed males still require review.',
        'Relink SLPs only to the matching full-resolution original MP4, not a resized or temporally cropped video.']
    (directory/'comparison.md').write_text('\n'.join(lines)+'\n')
    # Reverify immutable inputs and executed sources before declaring success.
    verify_inputs(cfg)
    for name, digest in pins.items():
        assert sha256(ROOT/name) == digest, f'Input/source changed during run: {name}'
    assert len(list(directory.glob('*.predictions.slp'))) == 2
    assert not list(directory.glob('*.tmp.slp')) and not list(directory.glob('*.frames_*.slp'))
    artifacts = [dict(path=str(p.relative_to(directory)), bytes=p.stat().st_size, sha256=sha256(p))
                 for p in sorted(directory.rglob('*')) if p.is_file()]
    atomic_json(directory/'COMPLETE.json', dict(status='complete', job_id=os.environ['SLURM_JOB_ID'],
        elapsed_s=time.monotonic()-started, inference_frames=1000, conditions=2,
        no_frame_chunk_slps=True, artifacts=artifacts))
    print('COMPLETE', directory, flush=True)


if __name__ == '__main__':
    main()
