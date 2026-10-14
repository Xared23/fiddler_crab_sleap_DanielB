#!/bin/bash
# CPU merge of already-predicted chunk .slp files. No GPU, no gputest loop.
#
# Submit (put --video AFTER the script name):
#   sbatch merge_dan_native.sh --video crab_vids/Crab_Cam_2_3.mp4
#
# Done when predictions/<video_stem>/PREDICT_COMPLETE exists.
#SBATCH --job-name=v004_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/predict_logs/merge_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/predict_logs/merge_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=db8682@princeton.edu

set -euo pipefail
module purge

PROJECT_ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
LOG_DIR="${PROJECT_ROOT}/predict_logs"
mkdir -p "${LOG_DIR}"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/gputest_chain.sh"

if [[ "${1:-}" == --video=* ]]; then
  resolve_project_video "${1#--video=}"
elif [[ "${1:-}" == "--video" && -n "${2:-}" ]]; then
  resolve_project_video "$2"
else
  echo "Usage: sbatch merge_dan_native.sh --video <path-under-fiddler_crab_sleap>" >&2
  exit 1
fi

STEM="$(basename "${VIDEO}")"
STEM="${STEM%.*}"
RUN_DIR="${PROJECT_ROOT}/predictions/${STEM}"
mkdir -p "${RUN_DIR}"

cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh"

"${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/scripts/gputest_predict.py" \
  --video "${VIDEO_REL}" \
  --run-dir "${RUN_DIR}" \
  --merge-only
