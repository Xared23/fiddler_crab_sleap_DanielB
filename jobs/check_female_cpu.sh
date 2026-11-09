#!/bin/bash
# Finite software checks; no training, inference, submission or requeue.
#SBATCH --job-name=female_cpu_checks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_cpu_checks_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_cpu_checks_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$ROOT/checks/female_v002/cpu_tmp_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
for script in jobs/*female*.sh scripts/female_train_chain.sh; do bash -n "$script"; done
"$ROOT/.venv/bin/python" -m unittest discover -s tests -p test_training_workflow.py -v
"$ROOT/.venv/bin/python" -m unittest discover -s tests -p test_female_workflow.py -v
