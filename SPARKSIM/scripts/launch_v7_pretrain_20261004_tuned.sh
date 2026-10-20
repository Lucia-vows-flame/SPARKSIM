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
    local model="$1" gpu="$2" epochs="$3" batch="$4" lr="$5" milestone1="$6" milestone2="$7"
    local run_dir="runs/${model}_tuned_20261004_software_pretrain"
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
        --learning-rate "$lr" \
        --momentum 0.9 \
        --weight-decay 0.0001 \
        --milestones "$milestone1" "$milestone2" \
        --gamma 0.1 \
        --seed 20261004 \
        > "$run_dir/train.log" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$run_dir/train.pid"
    printf '%s physical GPU %s PID %s log %s/train.log\n' "$model" "$gpu" "$pid" "$run_dir"
}

launch vgg16_snn_v7_phase6 0 120 32 0.005 70 100
launch alexnet_snn_v7_phase6 0 100 64 0.01 50 80
launch lenet_snn_v7_phase6 0 100 64 0.01 50 80
