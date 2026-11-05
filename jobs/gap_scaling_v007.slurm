#!/bin/bash
# Gap-bridging scaling study (v007): one stage per submission, selected by $1.
# Submit through jobs/gap_scaling_v007_submit.sh from the repository root; the
# stages, array sizes and dependencies are set there. Windows and overlays
# must exist (scripts/gap_scaling.py windows / overlay) before submission.
#
#   sweep_current  array 0-19: window x seed, hand cost (v003)
#   sweep_frozen   array 0-39: window x seed x {s1_frozen, s1p50_frozen}
#   train          array 0-3:  retrain S1 per window (pipeline rule) + p50 overlay
#   sweep_rt       array 0-39: window x seed x {s1_rt, s1p50_rt}
#   measure        single:     wall time, stage timings and peak RSS, all windows
#   finalize       single:     compare, natural links, audit video and audits
#SBATCH --job-name=gap_scaling_v007
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail

STAGE=${1:?usage: sbatch [opts] jobs/gap_scaling_v007.slurm <stage>}
ROOT=${SLURM_SUBMIT_DIR:?submit from the repository root}
cd "$ROOT"
[ -x .venv/bin/python ] && [ -f scripts/bridge_gaps.py ] || { echo "not the repository root: $ROOT" >&2; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
PY=.venv/bin/python
R=logs/gap_scaling_v007
SIZES=(300 600 1000 2000)
HORIZONS=5,10,15,30,45,60
I=${SLURM_ARRAY_TASK_ID:-0}

field() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"; }

sweep() {  # sweep <size> <seed> <variant> <config or ''>
    local W=$1 K=$2 V=$3 C=$4 J=$R/windows/w$1/window.json
    local OUT=$R/sweeps/w$W/$V/seed_$K
    mkdir -p "$(dirname "$OUT")"
    $PY scripts/bridge_gaps.py sweep --observations $R/windows/w$W/observations.parquet \
        --max-gaps $HORIZONS --holdout --profile short --seed "$K" --no-figures --force \
        --per-gap "$(field $J per_gap)" --decoys "$(field $J decoys)" --flicker "$(field $J flicker)" \
        ${C:+--config "$C"} --out "$OUT" > "$OUT.log" 2>&1
    echo "w$W $V seed $K exit 0"
}

case "$STAGE" in
sweep_current)
    sweep "${SIZES[$((I / 5))]}" $((I % 5)) current '' ;;
sweep_frozen)
    V=(s1_frozen s1p50_frozen); C=(configs/tracking/gap_bridging_v006_s1.yaml $R/configs/s1p50_frozen.yaml)
    sweep "${SIZES[$((I / 10))]}" $((I % 5)) "${V[$((I / 5 % 2))]}" "${C[$((I / 5 % 2))]}" ;;
train)
    W=${SIZES[$I]}; M=$R/models/w$W; mkdir -p "$M"
    SW=$(for k in 0 1 2 3 4; do printf "%s " $R/sweeps/w$W/current/seed_$k; done)
    $PY scripts/bridge_gaps.py train --sweeps $SW --features all --threshold-rule pipeline \
        --out "$M/link_scorer_s1.joblib" --write-config "$M/gap_bridging_s1.yaml" --force \
        > "$M/train.log" 2>&1
    $PY scripts/gap_scaling.py overlay --scorer "$M/link_scorer_s1.joblib" --probability 0.5 \
        --out "$M/gap_bridging_s1p50.yaml"
    echo "trained w$W" ;;
sweep_rt)
    W=${SIZES[$((I / 10))]}; V=(s1_rt s1p50_rt); C=(gap_bridging_s1.yaml gap_bridging_s1p50.yaml)
    sweep "$W" $((I % 5)) "${V[$((I / 5 % 2))]}" "$R/models/w$W/${C[$((I / 5 % 2))]}" ;;
measure)
    mkdir -p $R/measure
    hostname > $R/measure/host.txt
    for W in "${SIZES[@]}"; do
        for S in current s1 s1p50; do
            case $S in current) C='';; s1) C=configs/tracking/gap_bridging_v006_s1.yaml;;
                       s1p50) C=$R/configs/s1p50_frozen.yaml;; esac
            $PY scripts/gap_scaling.py measure --observations $R/windows/w$W/observations.parquet \
                ${C:+--config "$C"} --label "w$W:$S" --repeats 3 --out $R/measure/w${W}_$S.csv
        done
    done ;;
finalize)
    for W in "${SIZES[@]}"; do
        VARS=()
        for V in current s1_frozen s1p50_frozen s1_rt s1p50_rt; do
            VARS+=(--variant "$V=$(for k in 0 1 2 3 4; do printf "%s," $R/sweeps/w$W/$V/seed_$k; done)")
        done
        mkdir -p $R/compare/w$W
        for F in test all; do
            $PY scripts/bridge_gaps.py compare "${VARS[@]}" --fold $F --through-gap 30 \
                --demo-horizon 30 --out $R/compare/w$W/$F --force > $R/compare/w${W}_$F.log 2>&1
        done
    done
    $PY scripts/gap_scaling.py natural --root $R --sizes 300,600,1000,2000
    VID=$R/video/crab_cam2_2_middle_2000.mp4
    [ -f "${VID%.mp4}.json" ] || $PY scripts/gap_scaling.py concat-video \
        --bundle demo/crab_cam2_2_middle_2000 --out "$VID"
    for V in s1_frozen s1p50_frozen s1_rt; do
        $PY scripts/bridge_gaps.py audit --current $R/sweeps/w2000/current/seed_0/max_gap_30 \
            --candidate $R/sweeps/w2000/$V/seed_0/max_gap_30 \
            --observations $R/windows/w2000/observations.parquet --video "$VID" \
            --out $R/audit/w2000_$V --n 20 --force
    done ;;
*) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
