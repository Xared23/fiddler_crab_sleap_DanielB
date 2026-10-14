#!/bin/bash
# Finite CPU preservation/verification or project-notes relocation only.
#SBATCH --job-name=crab_v007_maintenance
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/maintenance_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/maintenance_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
"$ROOT/.venv/bin/python" scripts/maintenance_v007.py "${1:?Pass preserve, verify-resume, or organize-notes}"
