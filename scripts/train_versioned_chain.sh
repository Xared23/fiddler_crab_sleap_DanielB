# Source from a role-specific batch script after ROLE and SCRIPT_PATH are set.
# The worker's explicit exit 75 is the ONLY retry request. Fatal/OOM/save errors
# are never overridden by the arrival of a Slurm signal.
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1
export WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

"$ROOT/.venv/bin/python" scripts/train_versioned.py \
  --config configs/v007_19f/run.yaml --role "$ROLE" &
worker_pid=$!
forward_stop() {
  kill -s "$1" "$worker_pid" 2>/dev/null || true
}
trap 'forward_stop USR1' USR1
trap 'forward_stop TERM' TERM
set +e
while true; do
  wait "$worker_pid"
  rc=$?
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    break
  fi
done
set -e
trap - USR1 TERM
printf 'Training worker exit code: %s\n' "$rc"
case "$rc" in
  0) exit 0 ;;
  75)
    # Reuse the same slot and job ID; --open-mode=append preserves all hop logs.
    # If Slurm rejects requeue, fail clearly rather than creating a second chain.
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
    ;;
  *) echo "Fatal worker exit $rc; NOT requeuing." >&2; exit "$rc" ;;
esac
