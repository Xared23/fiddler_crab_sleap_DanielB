#!/bin/bash
# Fixed six-clip pilot + synthetic tests; no inference or production launch.
#SBATCH --job-name=crab_tracking_pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/tracking/pilot_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/tracking/pilot_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.venv/bin/python" scripts/validate_body_first_tracklets.py
