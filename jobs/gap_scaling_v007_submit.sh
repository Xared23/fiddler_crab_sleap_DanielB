#!/bin/bash
# Submit the gap-bridging scaling study (v007) as a dependent Slurm chain.
# Run from the repository root after:
#   .venv/bin/python scripts/gap_scaling.py windows --source demo/crab_cam2_2_middle_2000/observations.parquet \
#       --out logs/gap_scaling_v007/windows --sizes 300,600,1000,2000
#   .venv/bin/python scripts/gap_scaling.py overlay --scorer configs/tracking/link_scorer_v005_s1.joblib \
#       --probability 0.5 --out logs/gap_scaling_v007/configs/s1p50_frozen.yaml
# Then: bash jobs/gap_scaling_v007_submit.sh   (job ids are written to logs/gap_scaling_v007/slurm/jobs.txt)
set -euo pipefail
R=logs/gap_scaling_v007
for W in 300 600 1000 2000; do [ -f $R/windows/w$W/window.json ] || { echo "missing window w$W" >&2; exit 2; }; done
[ -f $R/configs/s1p50_frozen.yaml ] || { echo "missing frozen p50 overlay" >&2; exit 2; }
mkdir -p $R/slurm
J=jobs/gap_scaling_v007.slurm
O="--parsable --output=$R/slurm/%x_%A_%a.out"
CUR=$(sbatch $O --job-name=gs7_current --array=0-19 $J sweep_current)
FRZ=$(sbatch $O --job-name=gs7_frozen --array=0-39 $J sweep_frozen)
TRN=$(sbatch $O --job-name=gs7_train --array=0-3 --mem=16G --time=12:00:00 \
      --dependency=afterok:$CUR $J train)
RT=$(sbatch $O --job-name=gs7_rt --array=0-39 --dependency=afterok:$TRN $J sweep_rt)
MEA=$(sbatch $O --job-name=gs7_measure --cpus-per-task=4 --mem=16G --time=02:00:00 $J measure)
FIN=$(sbatch $O --job-name=gs7_finalize --mem=32G --time=04:00:00 \
      --dependency=afterok:$CUR:$FRZ:$RT $J finalize)
printf "current %s\nfrozen %s\ntrain %s\nretrained %s\nmeasure %s\nfinalize %s\n" \
    "$CUR" "$FRZ" "$TRN" "$RT" "$MEA" "$FIN" | tee $R/slurm/jobs.txt
