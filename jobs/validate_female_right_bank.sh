#!/bin/bash
# One finite CPU-only regional reanalysis of saved validation points.
# No neural inference, training, production edits, retries or requeue.
#SBATCH --job-name=female_right_bank_validation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_right_bank_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/female_right_bank_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR="$ROOT/checks/female_v002/right_bank_tmp_$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
"$ROOT/.venv/bin/python" - <<'PY'
import os
import subprocess
import sys
sys.path.insert(0, 'scripts')
from pipeline_common import ROOT, atomic_json, sha256
result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
    '-p', 'test_female_regional_validation.py', '-v'], capture_output=True, text=True)
print(result.stdout, result.stderr, flush=True)
base = ROOT / f'checks/female_v002/regional_tests_{os.environ["SLURM_JOB_ID"]}'
base.with_suffix('.stdout.txt').write_text(result.stdout)
base.with_suffix('.stderr.txt').write_text(result.stderr)
atomic_json(base.with_suffix('.json'), {'passed': result.returncode == 0,
    'exit_code': result.returncode, 'job_id': os.environ['SLURM_JOB_ID'],
    'test_source_sha256': sha256(ROOT / 'tests/test_female_regional_validation.py'),
    'analysis_source_sha256': sha256(ROOT / 'scripts/female_regional_validation.py')})
if result.returncode:
    raise SystemExit(result.returncode)
PY
"$ROOT/.venv/bin/python" "$ROOT/scripts/female_regional_validation.py"
