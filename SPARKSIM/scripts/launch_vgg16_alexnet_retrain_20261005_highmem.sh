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
    local model="$1" gpu="$2" epochs="$3" batch="$4" lr="$5"
    local milestone1="$6" milestone2="$7" seed="$8" run_dir="$9"
    mkdir -p "$run_dir"
    if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" || \
          -e "$run_dir/${model}_software_pretrain_latest.pth" ]]; then
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
        --gradient-clip-value 1.0 \
        --lif-threshold 0.01 \
        --lif-tau 50.5016666556 \
        --seed "$seed" \
        > "$run_dir/train.log" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$run_dir/train.pid"
    printf '%s physical GPU %s batch %s PID %s log %s/train.log\n' \
        "$model" "$gpu" "$batch" "$pid" "$run_dir"
}

# Larger batches use the free capacity on GPU0/GPU1 while retaining the
# validated V7 LIF parameters and clipped SGD schedule.
launch vgg16_snn_v7_phase6 0 240 64 0.005 120 190 20261024 \
    runs/vgg16_snn_v7_phase6_retrain_20261005_b64_highmem
launch alexnet_snn_v7_phase6 1 240 256 0.005 120 190 20261025 \
    runs/alexnet_snn_v7_phase6_retrain_20261005_b256_highmem
