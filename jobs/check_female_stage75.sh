#!/bin/bash
# Finite CPU test or narrowly scoped atomic installation. Never submits,
# cancels, requeues or signals a production job; no training or inference.
#SBATCH --job-name=female_stage75_check
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_stage75_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_stage75_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$ROOT/checks/female_v002/stage75/tmp_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
"$ROOT/.venv/bin/python" "$ROOT/scripts/install_female_stage75.py" "${1:?Pass test or install}"
