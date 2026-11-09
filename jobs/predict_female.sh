#!/bin/bash
#SBATCH --job-name=female_predict
#SBATCH --dependency=afterok:16998986
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:20:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_predict_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_predict_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
if [[ $# -ne 1 || ( "$1" != Crab_Cam_2_2 && "$1" != Crab_Cam_2_3 ) ]]; then
    echo 'Usage: sbatch jobs/predict_female.sh Crab_Cam_2_{2,3}' >&2
    exit 2
fi
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.venv/bin/python" "$ROOT/scripts/female_workflow.py" predict --video-key "$1" &
PID=$!
trap 'kill -USR1 "$PID" 2>/dev/null || true' USR1
trap 'kill -TERM "$PID" 2>/dev/null || true' TERM
set +e
while true; do
    wait "$PID"
    RC=$?
    if ! kill -0 "$PID" 2>/dev/null; then break; fi
done
set -e
trap - USR1 TERM
if [[ "$RC" -eq 75 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$RC"
