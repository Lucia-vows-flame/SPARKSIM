#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python="$repo_root/../.conda-envs/mnsim-py312/bin/python"
dataset="inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"

if [[ ! -x "$python" || ! -d "$dataset" ]]; then
    echo "Training environment or V7 dataset is unavailable" >&2
    exit 1
fi

launch() {
    local model="$1" gpu="$2" run_dir="$3" epochs="$4" batch="$5" milestone1="$6" milestone2="$7" seed="$8"
    mkdir -p "$run_dir"
    if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" ]]; then
        echo "Existing launch artifacts in $run_dir; refusing to overwrite" >&2
        exit 1
    fi

    setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" MNSIM_DVS_NUM_WORKERS=8 OMP_NUM_THREADS=4 \
        "$python" -u scripts/train_snn_v7.py \
        --model "$model" \
        --dataset-root "$dataset" \
        --sim-config SimConfig.ini \
        --output-dir "$run_dir" \
        --device cuda:0 \
        --epochs "$epochs" \
        --train-batch "$batch" \
        --test-batch "$batch" \
        --learning-rate 0.001 \
        --momentum 0.9 \
        --weight-decay 0.0001 \
        --milestones "$milestone1" "$milestone2" \
        --gamma 0.1 \
        --gradient-clip-value 1.0 \
        --lif-threshold 0.5 \
        --lif-tau 2.0 \
        --seed "$seed" \
        > "$run_dir/train.log" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$run_dir/train.pid"
    printf '%s physical GPU %s PID %s log %s/train.log\n' "$model" "$gpu" "$pid" "$run_dir"
}

launch resnet34_snn_v7_phase6 1 \
    runs/resnet34_snn_v7_phase6_membrane_20261004_b32 200 32 100 160 20261015
launch resnet50_snn_v7_phase6 3 \
    runs/resnet50_snn_v7_phase6_membrane_20261004_b16 240 16 120 190 20261016
