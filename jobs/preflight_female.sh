#!/bin/bash
#SBATCH --job-name=female_preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:30:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_preflight_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_preflight_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR="$ROOT/checks/female_v002/tmp_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
PY="$ROOT/.venv/bin/python"
for script in jobs/*female*.sh scripts/female_train_chain.sh; do bash -n "$script"; done
"$PY" -m unittest discover -s tests -p test_training_workflow.py -v
"$PY" -m unittest discover -s tests -p test_female_workflow.py -v
for role in centroid centered; do
    set +e
    "$PY" scripts/train_versioned.py --config configs/female_v002/run.yaml \
        --role "$role" --smoke --stop-after-steps 1
    rc=$?
    set -e
    if [[ "$rc" -ne 75 ]]; then
        echo "Expected clean smoke interruption exit 75 for $role, got $rc" >&2
        exit 1
    fi
    "$PY" scripts/train_versioned.py --config configs/female_v002/run.yaml --role "$role" --smoke
done
"$PY" scripts/female_workflow.py preflight-receipt
