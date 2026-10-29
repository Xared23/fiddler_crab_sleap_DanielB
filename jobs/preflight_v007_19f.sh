#!/bin/bash
# Finite training/restart checks; never requeues and never launches production.
#SBATCH --job-name=v007_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/preflight_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/preflight_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=expandable_segments:True
PY="$ROOT/.venv/bin/python"
for script in jobs/train_v007_centroid.sh jobs/train_v007_centered.sh scripts/train_versioned_chain.sh; do
  bash -n "$script"
done
"$PY" -m unittest discover -s tests -p 'test_training_workflow.py' -v
for role in centroid centered; do
  set +e
  "$PY" scripts/train_versioned.py --config configs/v007_19f/run.yaml --role "$role" --smoke --stop-after-steps 1
  rc=$?
  set -e
  if [[ "$rc" -ne 75 ]]; then
    echo "Expected a clean interrupted smoke hop (75), got $rc for $role" >&2
    exit 1
  fi
  "$PY" scripts/train_versioned.py --config configs/v007_19f/run.yaml --role "$role" --smoke
done
"$PY" - <<'PY'
import sys
sys.path.insert(0, 'scripts')
from pipeline_common import ROOT, atomic_json, dataset_manifest, fingerprint, load_run, read_json, sha256, versions
from train_versioned import checkpoint_summary, implementation_hashes
run = load_run('configs/v007_19f/run.yaml')
for role in ('centroid', 'centered'):
    path = ROOT / 'checks' / run['run_id'] / 'training_smoke' / role
    marker = read_json(path / 'TRAINING_COMPLETE.json')
    assert marker['global_step'] >= 4, marker
    assert marker['epoch'] == 2, marker
    checkpoint_summary(path / 'best.ckpt')
atomic_json(ROOT / 'checks' / run['run_id'] / 'PREFLIGHT_COMPLETE.json', {
    'implementation': implementation_hashes(),
    'run_config_sha256': sha256(ROOT / 'configs' / run['run_id'] / 'run.yaml'),
    'dataset_sha256': fingerprint(dataset_manifest(run)),
    'versions': versions(),
    'checks': ['regressions', 'GPU forward/backward both models', 'signal/save/resume both models', 'held-out validation', 'best checkpoint read-back'],
})
print('PREFLIGHT_COMPLETE: both model workers passed interrupted/resumed GPU training.')
PY
