#!/bin/bash
#SBATCH --job-name=crab_demo_2000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/demo_preparation/replace_2000_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/demo_preparation/replace_2000_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/.venv/bin/python \
  /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/scripts/replace_crab_demo_2000.py
