#!/bin/bash
#SBATCH --job-name=crab_label_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_combined/audit_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_combined/audit_%j.err
set -euo pipefail
# Use the existing venv directly; this job does not require environment modules.
ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
cd "$ROOT"
export PYTHONUNBUFFERED=1
"$ROOT/.venv/bin/python" scripts/audit_labels.py \
  --output data/labels/v007_combined/audit.json
