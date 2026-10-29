#!/bin/bash
# 1-hour Della gpu-test hop for the centroid model.
# Lands in the gpu-test QOS (time < 61 min, typically ~2 jobs/user, short queue).
# Saves last.ckpt, then requeues this job until training finishes.
#
# Submit once:  sbatch train_centroid_dan_v004.sh
# Done when models/dan_centroid_native/TRAINING_COMPLETE exists.
#SBATCH --job-name=v004_centroid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
<<<<<<< HEAD
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/SHAEVITZ/kr3541/crab/centroid_v004_logs/sleap_%j.out
#SBATCH --error=/scratch/gpfs/SHAEVITZ/kr3541/crab/centroid_v004_logs/sleap_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=kr3541@princeton.edu

mkdir -p /scratch/gpfs/SHAEVITZ/kr3541/crab/centroid_v004_logs
=======
#SBATCH --time=00:58:00
#SBATCH --signal=B:USR1@90
#SBATCH --requeue
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/centroid_v004_logs/sleap_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/centroid_v004_logs/sleap_%j.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=db8682@princeton.edu
>>>>>>> my-code

set -euo pipefail
module purge

PROJECT_ROOT=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
CONFIG="${PROJECT_ROOT}/della_centroid_config_dan_v004.yaml"
RUN_NAME=dan_centroid_native
RUN_DIR="${PROJECT_ROOT}/models/${RUN_NAME}"
LOG_DIR="${PROJECT_ROOT}/centroid_v004_logs"
SCRIPT_PATH="${PROJECT_ROOT}/train_centroid_dan_v004.sh"

# shellcheck disable=SC1091
source "${PROJECT_ROOT}/scripts/gputest_chain.sh"
run_gputest_hop
