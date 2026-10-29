#!/bin/bash
# Production: v007-only 17/2 split, fresh pose model, 58-minute gpu-test hops.
#SBATCH --job-name=v007_centered
# Launch gate for this production submission; only successful preflight releases it.
# Historical launch used: sbatch --dependency=afterok:15256367 (job finished; pass on the command line if needed).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:58:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/centered_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/v007_19f/centered_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=db8682@princeton.edu
set -euo pipefail
ROLE=centered
SCRIPT_PATH=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/jobs/train_v007_centered.sh
source /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/scripts/train_versioned_chain.sh
