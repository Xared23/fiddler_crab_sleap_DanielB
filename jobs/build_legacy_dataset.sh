#!/bin/bash
# Finite tests + streaming legacy tracklets/Parquet build. Never runs inference.
#SBATCH --job-name=legacy_tracklets_build
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/legacy_dataset/build_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/legacy_dataset/build_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$ROOT/checks/legacy_dataset"
"$ROOT/.venv/bin/python" tests/legacy_dataset/test_tracklets.py
"$ROOT/.venv/bin/python" scripts/legacy_dataset/export.py
