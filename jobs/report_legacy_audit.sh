#!/bin/bash
# Finite metadata summarization and static ROI proposal previews only.
#SBATCH --job-name=legacy_audit_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/legacy_dataset/report_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/legacy_dataset/report_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.venv/bin/python" scripts/legacy_dataset/report.py
