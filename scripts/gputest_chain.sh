# Shared gputest hop + requeue helpers. Source after RUN_DIR / CONFIG / RUN_NAME
# / LOG_DIR are set (predict may set RUN_DIR after sourcing). Expects to be
# sourced from a Slurm batch script.

PROJECT_ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
MAX_HOPS="${MAX_HOPS:-80}"
EXIT_REQUEUE=75
SCRIPT_PATH="${SCRIPT_PATH:-$0}"
GOT_PREEMPT=0
COMPLETE_MARKER="${COMPLETE_MARKER:-TRAINING_COMPLETE}"
REQUEUE_NEEDS_CKPT="${REQUEUE_NEEDS_CKPT:-1}"
# 1 = always sbatch a new job so each hop gets a new %j log file.
PREFER_NEW_JOB="${PREFER_NEW_JOB:-0}"
if [[ -z "${SBATCH_EXTRA_ARGS+x}" ]]; then
  SBATCH_EXTRA_ARGS=()
fi

if [[ -n "${LOG_DIR:-}" && -n "${RUN_DIR:-}" ]]; then
  mkdir -p "${LOG_DIR}" "${RUN_DIR}"
fi

# Resolve --video against PROJECT_ROOT so relative paths attach to
# fiddler_crab_sleap (including nested folders). Rejects anything outside it.
resolve_project_video() {
  local raw="$1"
  local root_real root_name candidate
  root_real="$(realpath "${PROJECT_ROOT}")"
  root_name="$(basename "${root_real}")"
  raw="${raw#./}"
  if [[ "${raw}" == "${root_name}/"* ]]; then
    raw="${raw#"${root_name}"/}"
  fi
  if [[ "${raw}" == /* ]]; then
    candidate="${raw}"
  else
    candidate="${root_real}/${raw}"
  fi
  if [[ ! -f "${candidate}" ]]; then
    echo "Video not found: ${candidate}" >&2
    echo "Pass --video as a path under ${root_real} (relative paths start there)." >&2
    exit 1
  fi
  candidate="$(realpath "${candidate}")"
  case "${candidate}" in
    "${root_real}"/*) ;;
    *)
      echo "Video must be under ${root_real} (got ${candidate})" >&2
      exit 1
      ;;
  esac
  VIDEO="${candidate}"
  VIDEO_REL="${candidate#"${root_real}"/}"
}

increment_hop() {
  mkdir -p "${LOG_DIR}" "${RUN_DIR}"
  local hops=0
  if [[ -f "${RUN_DIR}/.gputest_hops" ]]; then
    hops="$(<"${RUN_DIR}/.gputest_hops")"
  fi
  hops=$((hops + 1))
  printf '%s\n' "${hops}" >"${RUN_DIR}/.gputest_hops"
  echo "gputest hop ${hops}/${MAX_HOPS}  run_dir=${RUN_DIR}"
  if (( hops > MAX_HOPS )); then
    echo "Reached MAX_HOPS=${MAX_HOPS}; not starting another hop." >&2
    exit 1
  fi
}

# wait(1) returns 128+N when the shell itself is interrupted by a trapped
# signal (Slurm USR1), even if the child is still running. Keep waiting
# until the trainer PID is actually gone.
wait_for_trainer() {
  local pid="$1"
  TRAINER_RC=0
  while true; do
    wait "${pid}"
    TRAINER_RC=$?
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  done
}

submit_next_hop() {
  trap - USR1 TERM
  # Prefer reusing this job slot (gpu-test allows only 2 jobs/user), unless
  # this workflow wants a new job ID so Slurm opens a new log file.
  if [[ "${PREFER_NEW_JOB}" -ne 1 && -n "${SLURM_JOB_ID:-}" ]] && scontrol requeue "${SLURM_JOB_ID}"; then
    echo "scontrol requeue ${SLURM_JOB_ID} succeeded"
    exit 0
  fi
  echo "Submitting next hop: ${SCRIPT_PATH}" "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}"
  # Do not export this job's GPU env into the next allocation.
  if [[ "${PREFER_NEW_JOB}" -eq 1 ]]; then
    sbatch --export=NONE \
      --output="${LOG_DIR}/%j.out" \
      --error="${LOG_DIR}/%j.err" \
      "${SCRIPT_PATH}" "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}"
  else
    sbatch --export=NONE "${SCRIPT_PATH}" "${SBATCH_EXTRA_ARGS[@]+"${SBATCH_EXTRA_ARGS[@]}"}"
  fi
}

requeue_if_needed() {
  local rc="$1"
  if [[ -f "${RUN_DIR}/${COMPLETE_MARKER}" ]]; then
    echo "Complete (${RUN_DIR}/${COMPLETE_MARKER}). Not requeuing."
    return 0
  fi
  # Time-limit hops sometimes save a ckpt and then exit 0 after Lightning
  # stops the loop. Still requeue if we forwarded USR1/TERM this hop.
  if [[ "${GOT_PREEMPT}" -eq 1 ]]; then
    echo "Hop was preempted (exit ${rc}); requeuing."
    submit_next_hop
    return 0
  fi
  if [[ "${rc}" -eq 0 ]]; then
    echo "Worker exited 0 without ${COMPLETE_MARKER}; not requeuing."
    return 0
  fi
  # 75 = worker asked for another hop; 143/137 = SIGTERM/SIGKILL;
  # 138 = SIGUSR1 if the process died before the Python handler ran.
  if [[ "${rc}" -ne "${EXIT_REQUEUE}" && "${rc}" -ne 143 && "${rc}" -ne 137 && "${rc}" -ne 138 ]]; then
    echo "Worker failed with exit ${rc}; not requeuing." >&2
    return "${rc}"
  fi
  if [[ "${REQUEUE_NEEDS_CKPT}" -eq 1 ]]; then
    if [[ ! -f "${RUN_DIR}/last.ckpt" && ! -f "${RUN_DIR}/interrupt.ckpt" && ! -f "${RUN_DIR}/best.ckpt" ]]; then
      if [[ "${rc}" -eq 137 ]]; then
        echo "Killed with no checkpoint in ${RUN_DIR} (often OOM); not requeuing." >&2
        return 1
      fi
      echo "No checkpoint in ${RUN_DIR}; requeuing to retry setup." >&2
    fi
  fi
  submit_next_hop
}

run_python_hop() {
  increment_hop
  cd "${PROJECT_ROOT}"
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/env.sh"
  export WANDB_MODE=offline

  # Call venv python directly so USR1/TERM hit the worker, not a uv wrapper.
  "${PROJECT_ROOT}/.venv/bin/python" "$@" &
  PY_PID=$!
  forward_stop() {
    local sig="$1"
    GOT_PREEMPT=1
    echo "${sig} forwarded to pid ${PY_PID} and children" >&2
    kill -s "${sig}" "${PY_PID}" 2>/dev/null || true
    pkill -s "${sig}" -P "${PY_PID}" 2>/dev/null || true
  }
  trap 'forward_stop USR1' USR1
  trap 'forward_stop TERM' TERM
  set +e
  wait_for_trainer "${PY_PID}"
  rc="${TRAINER_RC}"
  set -e
  trap - USR1 TERM
  echo "Worker exit code: ${rc}"
  requeue_if_needed "${rc}"
}

run_gputest_hop() {
  run_python_hop \
    "${PROJECT_ROOT}/scripts/gputest_train.py" \
    --config "${CONFIG}" \
    --max-hops "${MAX_HOPS}" \
    "trainer_config.ckpt_dir=${PROJECT_ROOT}/models" \
    "trainer_config.run_name=${RUN_NAME}" \
    "trainer_config.model_ckpt.save_last=true" \
    "trainer_config.seed=42"
}

run_gputest_predict_hop() {
  local py_args=(
    --video "${VIDEO}"
    --run-dir "${RUN_DIR}"
    --chunk-frames "${CHUNK_FRAMES:-40}"
    --max-hops "${MAX_HOPS}"
    --peak-threshold "${PEAK_THRESHOLD:-0.2}"
  )
  if [[ -n "${MAX_INSTANCES:-}" ]]; then
    py_args+=(--max-instances "${MAX_INSTANCES}")
  fi
  run_python_hop "${PROJECT_ROOT}/scripts/gputest_predict.py" "${py_args[@]}"
}
