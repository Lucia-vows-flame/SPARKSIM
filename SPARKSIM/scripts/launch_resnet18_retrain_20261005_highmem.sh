#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python="$repo_root/../.conda-envs/mnsim-py312/bin/python"
dataset="inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"
run_dir="runs/resnet18_snn_v7_phase6_retrain_20261005_b96_membrane_mean"

if [[ ! -x "$python" || ! -d "$dataset" ]]; then
    echo "Training environment or V7 dataset is unavailable" >&2
    exit 1
fi
mkdir -p "$run_dir"
if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" || \
      -e "$run_dir/resnet18_snn_v7_phase6_software_pretrain_latest.pth" ]]; then
    echo "Existing launch artifacts in $run_dir; refusing to overwrite" >&2
    exit 1
fi

setsid nohup env CUDA_VISIBLE_DEVICES=2 MNSIM_DVS_NUM_WORKERS=8 OMP_NUM_THREADS=4 \
    "$python" -u scripts/train_snn_v7.py \
    --model resnet18_snn_v7_phase6 \
    --dataset-root "$dataset" \
    --sim-config SimConfig.ini \
    --output-dir "$run_dir" \
    --device cuda:0 \
    --epochs 200 \
    --train-batch 96 \
    --test-batch 96 \
    --learning-rate 0.001 \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --milestones 100 160 \
    --gamma 0.1 \
    --gradient-clip-value 1.0 \
    --lif-threshold 0.5 \
    --lif-tau 2.0 \
    --seed 20261027 \
    > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/train.pid"
printf 'resnet18_snn_v7_phase6 physical GPU 2 batch 96 PID %s log %s/train.log\n' "$pid" "$run_dir"
