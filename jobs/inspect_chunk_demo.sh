#!/bin/bash
#SBATCH --job-name=crab_chunk_metadata
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:03:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/demo_preparation/chunk_inspect_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/demo_preparation/chunk_inspect_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/.venv/bin/python /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/checks/demo_preparation/inspect_chunk_sources.py
