#!/bin/bash
# Finite one-shot publication, only after reviewed build. Never submits other jobs.
#SBATCH --job-name=legacy_tracklets_publish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/legacy_dataset/publish_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/legacy_dataset/publish_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.venv/bin/python" scripts/legacy_dataset/publish.py "${1:?Pass validated build job ID}"
