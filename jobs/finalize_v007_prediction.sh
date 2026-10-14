#!/bin/bash
# Finite CPU finalization; submit once per video with afterok dependency in this
# script. The complete untracked SLP is published BEFORE optional tracking begins.
#SBATCH --job-name=v007_prediction_finish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/finish_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/finish_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "$#" -ne 2 || "$1" != --video ]]; then
  echo 'Usage: sbatch jobs/finalize_v007_prediction.sh --video <authorized video>' >&2
  exit 2
fi
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/gputest_predict.py" \
  --run-config "${ROOT}/configs/prediction/v007_best_body_first_v001.json" \
  --video "$2" --merge-only
