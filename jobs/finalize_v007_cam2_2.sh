#!/bin/bash
# Fixed downstream job for the submitted Cam 2_2 GPU run.
#SBATCH --job-name=v007_finish_cam2_2
#SBATCH --dependency=afterok:15527343
#SBATCH --kill-on-invalid-dep=yes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/finish_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/logs/prediction_v007/finish_%j.err
set -euo pipefail
exec bash /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/jobs/finalize_v007_prediction.sh \
  --video crab_vids/Crab_Cam_2_2.mp4
