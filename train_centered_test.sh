#!/bin/bash
#SBATCH --job-name=v004_cent_inst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/centered_v004_logs/sleap_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/centered_v004_logs/sleap_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=db8682@princeton.edu

set -euo pipefail
module purge

mkdir -p /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/centered_v004_logs

cd /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
# shellcheck disable=SC1091
source env.sh

export WANDB_MODE=offline

uv run --extra train sleap-nn train \
    --config /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/della_centered_instance_config_dan_v004.yaml \
    "data_config.train_labels_path=[/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/danlabels.v004.slp]" \
    "trainer_config.ckpt_dir=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/models" \
    "trainer_config.run_name=dan_centered_instance_v004"


