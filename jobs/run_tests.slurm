#!/bin/bash
# Full test suite on a compute node. The login node kills a single pytest
# process after roughly 25 CPU-minutes, which the full suite exceeds, so run
# it here (mkdir -p logs/tests && sbatch jobs/run_tests.slurm; Slurm opens the
# output file before the script runs, so the directory must exist at submit
# time) or split it in two on the login node (docs/README.md, "Tests and lint").
#SBATCH --job-name=crab_tests
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=logs/tests/slurm_%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(dirname "$(readlink -f "$0")")")}"
[ -x .venv/bin/python ] || { echo "no .venv/bin/python here; submit from the project root (uv sync --extra dev)" >&2; exit 1; }
mkdir -p logs/tests
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
# Lint and tests both run; the job fails if either does.
lint=0
if .venv/bin/python -c 'import flake8' 2>/dev/null; then
  .venv/bin/python -m flake8 --count . || lint=$?
else
  echo 'flake8 not installed in .venv (uv sync --extra dev); lint skipped' >&2
fi
.venv/bin/python -m pytest -q -p no:cacheprovider tests
exit "$lint"
