#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python="$repo_root/../.conda-envs/mnsim-py312/bin/python"
dataset="inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"
model="resnet50_snn_v7_phase6"
run_dir="runs/${model}_tuned_20261004_b8_seed20261007"

if [[ ! -x "$python" || ! -d "$dataset" ]]; then
    echo "Training environment or V7 dataset is unavailable" >&2
    exit 1
fi
mkdir -p "$run_dir"
if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" ]]; then
    echo "Existing launch artifacts in $run_dir; refusing to overwrite" >&2
    exit 1
fi

setsid nohup env CUDA_VISIBLE_DEVICES=3 MNSIM_DVS_NUM_WORKERS=8 OMP_NUM_THREADS=4 \
    "$python" -u scripts/train_snn_v7.py \
    --model "$model" \
    --dataset-root "$dataset" \
    --sim-config SimConfig.ini \
    --output-dir "$run_dir" \
    --device cuda:0 \
    --epochs 120 \
    --train-batch 8 \
    --test-batch 8 \
    --learning-rate 0.001 \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --milestones 60 95 \
    --gamma 0.1 \
    --seed 20261007 \
    > "$run_dir/train.log" 2>&1 < /dev/null &

pid=$!
printf '%s\n' "$pid" > "$run_dir/train.pid"
printf '%s physical GPU 3 PID %s log %s/train.log\n' "$model" "$pid" "$run_dir"
