#!/bin/bash
# One finite read-only annotation/spacing analysis. No training or inference.
#SBATCH --job-name=burrow_location_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/burrow_stats_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/burrow_stats_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.venv/bin/python" - <<'PY'
import os
import subprocess
import sys
sys.path.insert(0, 'scripts')
from pipeline_common import ROOT, atomic_json, sha256
result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
    '-p', 'test_burrow_location_stats.py', '-v'], capture_output=True, text=True)
print(result.stdout, result.stderr, flush=True)
base = ROOT / f'checks/burrow_location_tests_{os.environ["SLURM_JOB_ID"]}'
base.with_suffix('.stdout.txt').write_text(result.stdout)
base.with_suffix('.stderr.txt').write_text(result.stderr)
atomic_json(base.with_suffix('.json'), {'passed': result.returncode == 0,
    'exit_code': result.returncode, 'job_id': os.environ['SLURM_JOB_ID'],
    'test_source_sha256': sha256(ROOT / 'tests/test_burrow_location_stats.py'),
    'analysis_source_sha256': sha256(ROOT / 'scripts/burrow_location_stats.py')})
if result.returncode:
    raise SystemExit(result.returncode)
PY
"$ROOT/.venv/bin/python" "$ROOT/scripts/burrow_location_stats.py"
