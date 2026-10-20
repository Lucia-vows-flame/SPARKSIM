#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python="$repo_root/../.conda-envs/mnsim-py312/bin/python"
dataset="inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"
run_dir="runs/vgg16_snn_v7_phase6_retrain_20261007_b56_lr01_e300_seed20261003"

if [[ ! -x "$python" || ! -d "$dataset" ]]; then
    echo "Training environment or V7 dataset is unavailable" >&2
    exit 1
fi
mkdir -p "$run_dir"
if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" || \
      -e "$run_dir/vgg16_snn_v7_phase6_software_pretrain_latest.pth" ]]; then
    echo "Existing launch artifacts in $run_dir; refusing to overwrite" >&2
    exit 1
fi

setsid nohup env CUDA_VISIBLE_DEVICES=0 MNSIM_DVS_NUM_WORKERS=8 OMP_NUM_THREADS=4 \
    "$python" -u scripts/train_snn_v7.py \
    --model vgg16_snn_v7_phase6 \
    --dataset-root "$dataset" \
    --sim-config SimConfig.ini \
    --output-dir "$run_dir" \
    --device cuda:0 \
    --epochs 300 \
    --train-batch 56 \
    --test-batch 56 \
    --learning-rate 0.01 \
    --momentum 0.9 \
    --weight-decay 0.0001 \
    --milestones 80 220 \
    --gamma 0.1 \
    --seed 20261003 \
    > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/train.pid"
printf 'vgg16_snn_v7_phase6 long retrain physical GPU 0 batch 56 epochs 300 PID %s log %s/train.log\n' "$pid" "$run_dir"
