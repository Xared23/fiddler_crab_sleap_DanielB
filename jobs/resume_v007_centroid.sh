#!/bin/bash
# Explicit future resume of the SAME v007_19f run. Do not submit while paused.
# No dependency on the historical preflight job ID; matching saved preflight and
# tested extension receipts are still required by the verifier and worker.
#SBATCH --job-name=v007_centroid
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:58:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/centroid_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/centroid_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=db8682@princeton.edu
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
# Verify the pause bundle and full resume state only on the explicit new
# submission. The existing worker repeats its own integrity checks on EVERY hop.
if [[ "${SLURM_RESTART_COUNT:-0}" -eq 0 ]]; then
    "$ROOT/.venv/bin/python" scripts/maintenance_v007.py verify-resume
fi
ROLE=centroid
SCRIPT_PATH="$ROOT/jobs/resume_v007_centroid.sh"
source "$ROOT/scripts/train_versioned_chain.sh"
