#!/bin/bash
# Finite read-only verification of the tested project-tool edits. No training,
# source edits, checkpoint writes, submission, requeue, cancellation or signals.
#SBATCH --job-name=female_stage75_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_stage75_verify_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_stage75_verify_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash -n "$ROOT/scripts/female_train_chain.sh"
"$ROOT/.venv/bin/python" - <<'PY'
import os
import sys
sys.path.insert(0, 'scripts')
from pipeline_common import ROOT, atomic_json, fingerprint, load_run, read_json, sha256
from female_workflow import verify_preflight
from female_stage75 import CHECKS, ALLOWED_CHANGES, full_checkpoint
from train_versioned import select_checkpoint
run = load_run('configs/female_v002/run.yaml')
expected = read_json(CHECKS / 'CANDIDATE_SIGNATURE.json')
assert verify_preflight(run) == expected
for name in ALLOWED_CHANGES:
    assert sha256(ROOT / name) == sha256(CHECKS / 'candidates' / name.rsplit('/', 1)[1])
    if name.endswith('.py'):
        compile((ROOT / name).read_text(), str(ROOT / name), 'exec')
directory = ROOT / 'models/female_v002/centroid'
checkpoint, progress = select_checkpoint(directory)
assert checkpoint is not None and progress['epoch'] < 75
state = full_checkpoint(checkpoint)
centered = ROOT / 'models/female_v002/centered'
marker = read_json(centered / 'TRAINING_COMPLETE.json')
assert marker['best_sha256'] == sha256(centered / 'best.ckpt')
assert marker['manifest_sha256'] == fingerprint(read_json(centered / 'manifest.json'))
full_checkpoint(centered / 'best.ckpt')
atomic_json(CHECKS / 'INSTALLED.json', {
    'passed': True, 'verification_job_id': os.environ['SLURM_JOB_ID'],
    'activation': 'next_natural_centroid_allocation_same_job_id',
    'centroid_job_id': '16998958',
    'installation': 'exact tested patch applied with project edit tools while centroid was PENDING between allocations; no cancellation or forced restart',
    'approval_sha256': sha256(CHECKS / 'APPROVED.json'),
    'active_signature': expected, 'selected_resume_checkpoint': str(checkpoint.relative_to(ROOT)),
    'observed_progress': progress, 'full_state_progress': state,
    'original_manifest_sha256': sha256(directory / 'manifest.json'),
    'centered_completion_unchanged': marker,
    'downstream_job_ids_unchanged': ['16998986', '16999003', '16999004', '16999056', '16999177'],
    'stage75_checkpoint_created_yet': (directory / 'stages/epoch_0075/STAGE_COMPLETE.json').exists(),
})
print('FEMALE_STAGE75_INSTALLED_AND_VERIFIED; no production restart or new training job submitted')
PY
