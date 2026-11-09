#!/bin/bash
# One finite local demo build, gated by tests; no Git operations or inference.
#SBATCH --job-name=demo_female_burrows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/demo_female_burrows_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/demo_female_burrows_%j.err
set -euo pipefail
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.venv/bin/python" - <<'PY'
import os
import subprocess
import sys
sys.path.insert(0, 'scripts')
from pipeline_common import ROOT, atomic_json, sha256
base = ROOT / f'checks/demo_augmentation_tests_{os.environ["SLURM_JOB_ID"]}'
results = []
for pattern in ('test_augment_crab_demo.py', 'test_body_first_tracklets.py'):
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'],
                            capture_output=True, text=True)
    print(result.stdout, result.stderr, flush=True)
    base.with_name(base.name + '_' + pattern + '.txt').write_text(result.stdout + result.stderr)
    results.append({'pattern': pattern, 'exit_code': result.returncode})
    if result.returncode:
        break
atomic_json(base.with_suffix('.json'), {'passed': all(r['exit_code'] == 0 for r in results),
    'job_id': os.environ['SLURM_JOB_ID'], 'results': results,
    'test_source_sha256': sha256(ROOT / 'tests/test_augment_crab_demo.py'),
    'analysis_source_sha256': sha256(ROOT / 'scripts/augment_crab_demo.py')})
if not all(r['exit_code'] == 0 for r in results):
    raise SystemExit(1)
PY
"$ROOT/.venv/bin/python" "$ROOT/scripts/augment_crab_demo.py"
