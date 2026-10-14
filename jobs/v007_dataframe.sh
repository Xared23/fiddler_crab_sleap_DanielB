#!/bin/bash
# Finite CPU-only tests/export or separately gated publication. Never inference/retracking.
#SBATCH --job-name=v007_dataframe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_dataframe/job_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_dataframe/job_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ "${1:-}" == build && $# -eq 1 ]]; then
    "$ROOT/.venv/bin/python" tests/test_v007_dataframe.py
    "$ROOT/.venv/bin/python" scripts/v007_dataframe.py build
elif [[ "${1:-}" == publish && $# -eq 2 && "$2" =~ ^[0-9]+$ ]]; then
    "$ROOT/.venv/bin/python" scripts/v007_dataframe.py publish "$2"
else
    echo 'Usage: jobs/v007_dataframe.sh build | publish BUILD_JOB_ID' >&2
    exit 2
fi
