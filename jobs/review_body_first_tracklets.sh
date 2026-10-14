#!/bin/bash
# Evidence-only rerender: include the predecessor of each selected new link.
# Replays the SAME six clips and frozen method, checks IDs against saved tables.
#SBATCH --job-name=crab_tracking_review
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/tracking/review_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/tracking/review_%j.err
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/.venv/bin/python - <<'PY'
import os, sys
from pathlib import Path
ROOT = Path('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap')
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import pandas as pd
import yaml
from pipeline_common import read_json, sha256, atomic_json
from validate_body_first_tracklets import load_frames, render_reviews
from body_first_tracklets import Tracker
from legacy_dataset.tracklets import Tracker as BaselineTracker, water_membership

pilot = ROOT / 'checks/body_first_v001/pilot_15525082'
report = read_json(pilot / 'PILOT_REPORT.json')
assert report['synthetic_tests_passed'] and report['numerical_gate_passed']
for path, expected in report['implementation'].items():
    assert sha256(ROOT / path) == expected, path
config = report['config']
legacy = ROOT / 'datasets/legacy_native_tracklets_v001'
manifest = read_json(legacy / 'manifest.json')
pins = {p['path']: p for p in manifest['source_pins'] + manifest['input_pins']}
baseline_config = yaml.safe_load((legacy / 'config/tracklets.yaml').read_text())
rois = read_json(legacy / 'config/water_regions_effective_v002.json')['videos']
reviews = read_json(pilot / 'review_examples.json')
work = ROOT / 'checks/body_first_v001' / f'predecessor_review_{os.environ["SLURM_JOB_ID"]}'
work.mkdir(parents=True, exist_ok=False)
all_reviews, checked = [], {}
for clip in config['pilot']['clips']:
    video, start, stop = clip['video'], clip['start'], clip['stop']
    clip_id = f'{video}_{start:06d}_{stop-1:06d}'
    trackers = {'baseline': BaselineTracker(clip_id, 7680, 4320, baseline_config),
                'new': Tracker(clip_id, 7680, 4320, config)}
    captures = {}
    expected = pd.read_parquet(pilot / f'{clip_id}.observations.parquet')
    groups = {key: group for key, group in expected.groupby(['method', 'frame'], sort=False)}
    for frame, source in load_frames(video, start, stop, pins, checked):
        result = {method: t.process(frame) for method, t in trackers.items()}
        for method, value in result.items():
            indices = np.flatnonzero(value['ids'] > 0)
            saved = groups.get((method, frame.index))
            if saved is None:
                assert len(indices) == 0
            else:
                np.testing.assert_array_equal(saved['instance_index'].to_numpy(), indices)
                assert saved['tracklet_id'].tolist() == [trackers[method].name(int(value['ids'][j])) for j in indices]
        captures[frame.index] = dict(frame=frame, baseline=result['baseline'], new=result['new'],
            water=water_membership(frame.body, 7680, 4320, rois[video]['polygon_xy']))
    corrected = []
    for original in reviews:
        if original['clip_id'] != clip_id:
            continue
        t, j = original['start'], original['source_index']
        predecessor = int(captures[t]['new']['predecessor_index'][j])
        assert predecessor >= 0 and t > start
        identity = original['tracklet_local_id']
        assert int(captures[t-1]['new']['ids'][predecessor]) == identity
        corrected.append(dict(category=original['category'], start=t-1,
                              tracklet_local_id=identity, source_index=predecessor))
    rendered = render_reviews(video, clip_id, captures, corrected, config, work)
    all_reviews.extend([dict(item, video=video, clip_id=clip_id) for item in rendered])
    print(f'Reviewed input/replayed exact identities and rendered {clip_id}', flush=True)
atomic_json(work / 'review_examples.json', all_reviews)
atomic_json(work / 'RENDER_COMPLETE.json', {
    'job_id': os.environ['SLURM_JOB_ID'], 'pilot_report_sha256': sha256(pilot / 'PILOT_REPORT.json'),
    'review_examples_sha256': sha256(work / 'review_examples.json'),
    'exact_replay_passed': True, 'examples': len(all_reviews),
    'reason': 'Original montage started at the newly accepted link, omitting its predecessor. Corrected views include that predecessor. No method, parameters, clips, or predictions changed.',
    'production_approved': False, 'visual_review_required': True})
print(work, flush=True)
PY
