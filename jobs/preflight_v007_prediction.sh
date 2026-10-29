#!/bin/bash
#SBATCH --job-name=v007_prediction_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/check_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/check_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=${ROOT}/checks/v007_best_body_first_v001/tmp
mkdir -p "${TMPDIR}"
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/prediction_run.py" preflight \
  --run-config "${ROOT}/configs/prediction/v007_best_body_first_v001.json"
