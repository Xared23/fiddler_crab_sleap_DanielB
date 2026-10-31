#!/bin/bash
# Bounded low-centroid-threshold re-inference of one video range (gap bridging v003, WP-B).
#
# Submit from the repository root (the job's root is the submit directory; the log
# directory must exist before submission):
#   mkdir -p logs/reinfer_low_threshold
#   VIDEO=crab_vids/Crab_Cam_2_2.mp4 START=0 STOP=300 \
#     OUT=predictions/reinfer_low_threshold/cam2_2_000000 sbatch jobs/reinfer_low_threshold.sh
# The output SLP keeps the video's frame numbers (START .. STOP-1).
# prediction_finalize.derive_tracks requires frame_idx == index (frames 0..n-1), so
# it consumes the output unchanged only when START == 0; for START > 0 cut the
# range into its own clip first (as the demo was) or renumber the frames.
# or pass the Python arguments directly:
#   sbatch jobs/reinfer_low_threshold.sh --video ... --start ... --stop ... --out ...
# Optional env: CENTROID_THRESHOLD (0.05), KEYPOINT_THRESHOLD (0.2), BASELINE_CONFIG.
#SBATCH --job-name=crab_reinfer_low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/reinfer_low_threshold/job_%j.out
#SBATCH --error=logs/reinfer_low_threshold/job_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
SCRIPT="$ROOT/scripts/reinfer_low_threshold.py"
if [[ ! -x "$PYTHON" || ! -f "$SCRIPT" ]]; then
    echo "ERROR: $ROOT is not the repository root (submit from there)" >&2
    exit 1
fi

# Fail fast without a GPU: never fall back to CPU inference.
if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi failed; no GPU visible. Refusing to run on CPU." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl_reinfer_${SLURM_JOB_ID:-local}"

if [[ $# -gt 0 ]]; then
    ARGS=("$@")
else
    ARGS=(--video "${VIDEO:?set VIDEO or pass arguments}"
          --start "${START:?set START or pass arguments}"
          --stop "${STOP:?set STOP or pass arguments}"
          --out "${OUT:?set OUT or pass arguments}"
          --centroid-threshold "${CENTROID_THRESHOLD:-0.05}"
          --keypoint-threshold "${KEYPOINT_THRESHOLD:-0.2}")
    if [[ -n "${BASELINE_CONFIG:-}" ]]; then
        ARGS+=(--baseline-config "$BASELINE_CONFIG")
    fi
fi

# Preflight (no torch): exits 2 and names the missing checkpoints or video.
"$PYTHON" "$SCRIPT" "${ARGS[@]}" --dry-run
"$PYTHON" "$SCRIPT" "${ARGS[@]}"
