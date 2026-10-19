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
        --seed "$seed" \
        > "$run_dir/train.log" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$run_dir/train.pid"
    printf '%s physical GPU %s batch %s PID %s log %s/train.log\n' \
        "$model" "$gpu" "$batch" "$pid" "$run_dir"
}

if [[ "${1:-}" == "vgg16_long" ]]; then
    launch vgg16_snn_v7_phase6 0 300 56 0.01 80 220 20261003 \
        runs/vgg16_snn_v7_phase6_retrain_20261007_b56_lr01_e300_seed20261003
    exit 0
fi

if [[ "${1:-}" == "vgg16_long_gpu2" ]]; then
    launch vgg16_snn_v7_phase6 2 300 56 0.01 80 220 20261003 \
        runs/vgg16_snn_v7_phase6_retrain_20261007_b56_lr01_e300_seed20261003_gpu2
    exit 0
fi

# These match the configurations that previously reached 79.05% (VGG16) and
# 88.98% (AlexNet). Keep optimizer updates per epoch comparable to those runs.
launch vgg16_snn_v7_phase6 0 180 56 0.01 80 130 20261003 \
    runs/vgg16_snn_v7_phase6_retrain_20261006_b56_lr01_seed20261003
launch alexnet_snn_v7_phase6 2 180 64 0.01 90 150 20261004 \
    runs/alexnet_snn_v7_phase6_retrain_20261006_b64_lr01_seed20261004
