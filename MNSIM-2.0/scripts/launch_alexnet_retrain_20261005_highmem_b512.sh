#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python="$repo_root/../.conda-envs/mnsim-py312/bin/python"
dataset="inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"
run_dir="runs/alexnet_snn_v7_phase6_retrain_20261005_b512_highmem"

mkdir -p "$run_dir"
if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" || \
      -e "$run_dir/alexnet_snn_v7_phase6_software_pretrain_latest.pth" ]]; then
    echo "Existing launch artifacts in $run_dir; refusing to overwrite" >&2
    exit 1
fi

setsid nohup env CUDA_VISIBLE_DEVICES=1 MNSIM_DVS_NUM_WORKERS=8 OMP_NUM_THREADS=4 \
    "$python" -u scripts/train_snn_v7.py \
    --model alexnet_snn_v7_phase6 \
    --dataset-root "$dataset" \
    --sim-config SimConfig.ini \
    --output-dir "$run_dir" \
    --device cuda:0 \
    --epochs 240 \
    --train-batch 512 \
    --test-batch 512 \
    --learning-rate 0.005 \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --milestones 120 190 \
    --gamma 0.1 \
    --gradient-clip-value 1.0 \
    --lif-threshold 0.01 \
    --lif-tau 50.5016666556 \
    --seed 20261026 \
    > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/train.pid"
printf 'alexnet_snn_v7_phase6 physical GPU 1 batch 512 PID %s log %s/train.log\n' "$pid" "$run_dir"
