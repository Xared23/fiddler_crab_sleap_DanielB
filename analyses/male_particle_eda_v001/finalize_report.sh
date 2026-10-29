#!/bin/bash
#SBATCH --job-name=male_crab_report
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/analyses/male_particle_eda_v001/job_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/analyses/male_particle_eda_v001/job_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
BASE="$ROOT/analyses/male_particle_eda_v001"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$BASE/mpl_cache_${SLURM_JOB_ID}"
"$ROOT/.venv/bin/python" "$BASE/test_analysis.py"
"$ROOT/.venv/bin/python" "$BASE/finalize_report.py"
