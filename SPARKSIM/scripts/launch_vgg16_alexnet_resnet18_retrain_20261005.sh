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
    local milestone1="$6" milestone2="$7" seed="$8" tau="$9" threshold="${10}"
    local run_dir="${11}"
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
        --lif-threshold "$threshold" \
        --lif-tau "$tau" \
        --seed "$seed" \
        > "$run_dir/train.log" 2>&1 < /dev/null &
    local pid=$!
    printf '%s\n' "$pid" > "$run_dir/train.pid"
    printf '%s physical GPU %s PID %s log %s/train.log\n' "$model" "$gpu" "$pid" "$run_dir"
}

# VGG11/VGG13 already converge with the shared V7 LIF setting. Keep that
# setting for VGG16 and address its instability with a smaller batch, lower
# learning rate, gradient clipping, and a longer decay schedule.
launch vgg16_snn_v7_phase6 0 240 32 0.005 120 190 20261021 50.5016666556 0.01 \
    runs/vgg16_snn_v7_phase6_retrain_20261005_b32_lr005_clip1

# AlexNet's previous 0.01 run reached 88.98%; retain the validated LIF
# setting and use a lower learning rate with a longer schedule for refinement.
launch alexnet_snn_v7_phase6 1 240 64 0.005 120 190 20261022 50.5016666556 0.01 \
    runs/alexnet_snn_v7_phase6_retrain_20261005_b64_lr005_clip1

# ResNet18 follows the validated deep-residual settings used by ResNet34/50.
launch resnet18_snn_v7_phase6 2 200 32 0.001 100 160 20261023 2.0 0.5 \
    runs/resnet18_snn_v7_phase6_retrain_20261005_b32_membrane_mean
