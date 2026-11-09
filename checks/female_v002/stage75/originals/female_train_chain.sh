# Source from the female role-specific batch entry point. Same safe same-ID
# requeue contract as the latest male pipeline; no replacement sbatch jobs.
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=offline
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
"$ROOT/.venv/bin/python" scripts/female_workflow.py verify-training
"$ROOT/.venv/bin/python" scripts/train_versioned.py \
    --config configs/female_v002/run.yaml --role "$ROLE" &
worker_pid=$!
trap 'kill -USR1 "$worker_pid" 2>/dev/null || true' USR1
trap 'kill -TERM "$worker_pid" 2>/dev/null || true' TERM
set +e
while true; do
    wait "$worker_pid"
    rc=$?
    if ! kill -0 "$worker_pid" 2>/dev/null; then break; fi
done
set -e
trap - USR1 TERM
printf 'Female training worker exit code: %s\n' "$rc"
case "$rc" in
    0) exit 0 ;;
    75) scontrol requeue "$SLURM_JOB_ID"; exit 0 ;;
    *) echo "Fatal worker exit $rc; NOT requeuing." >&2; exit "$rc" ;;
esac
