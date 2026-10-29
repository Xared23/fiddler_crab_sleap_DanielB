#!/bin/bash
#SBATCH --job-name=v004_cent_inst
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/remap_v004_logs/sleap_%j.out
#SBATCH --error=/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/remap_v004_logs/sleap_%j.err
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=db8682@princeton.edu

set -euo pipefail
module purge

cd /scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap
# shellcheck disable=SC1091
source env.sh

uv run python - <<'EOF'
import sleap_io as sio
labels = sio.load_slp('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/danlabels.v004.slp')
labels.videos[0].replace_filename('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/clip_cam2_8k.mp4')
labels.save('/scratch/gpfs/WBIALEK/db8682/fiddler_crab_sleap/danlabels.v004.slp')
print("Done - video remapped to clip_cam2_8k.mp4")
EOF
