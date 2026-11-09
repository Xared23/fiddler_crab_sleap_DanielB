#!/bin/bash
#SBATCH --job-name=female_finalize
#SBATCH --dependency=afterok:16999004
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_finalize_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_finalize_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
if [[ $# -ne 1 || ( "$1" != Crab_Cam_2_2 && "$1" != Crab_Cam_2_3 ) ]]; then
    echo 'Usage: sbatch jobs/finalize_female.sh Crab_Cam_2_{2,3}' >&2
    exit 2
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.venv/bin/python" "$ROOT/scripts/female_workflow.py" finalize --video-key "$1"
