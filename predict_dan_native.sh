#!/bin/bash
# Existing native prediction entry point, now with explicit versioned best models.
# Raw 40-frame SLPs only here; finite dependent CPU jobs merge and add optional IDs.
# Usage: sbatch predict_dan_native.sh --video crab_vids/Crab_Cam_2_2.mp4
#SBATCH --job-name=v007_best_predict
# Historical launch used: sbatch --dependency=afterok:15527306 (job finished; pass on the command line if needed).
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
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=db8682@princeton.edu
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CONFIG=${ROOT}/configs/prediction/v007_best_body_first_v001.json
if [[ "$#" -ne 2 || "$1" != --video ]]; then
  echo 'Usage: sbatch predict_dan_native.sh --video <authorized video>' >&2
  exit 2
fi
"${ROOT}/.venv/bin/python" "${ROOT}/scripts/gputest_predict.py" \
  --run-config "${CONFIG}" --video "$2" &
PID=$!
trap 'kill -USR1 "${PID}" 2>/dev/null || true' USR1
trap 'kill -TERM "${PID}" 2>/dev/null || true' TERM
set +e
while true; do
  wait "${PID}"
  RC=$?
  if ! kill -0 "${PID}" 2>/dev/null; then break; fi
done
set -e
trap - USR1 TERM
# Only the worker's explicit durable-checkpoint/time-budget exit may requeue.
# Fatal errors/signals do not become retries; there is no fallback sbatch chain.
if [[ "${RC}" -eq 75 ]]; then
  scontrol requeue "${SLURM_JOB_ID}" || exit 1
  exit 0
fi
exit "${RC}"
