#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python="$repo_root/../.conda-envs/mnsim-py312/bin/python"
dataset="inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"
checkpoint="inputs/weights/pretrain/vgg8_snn_v7_phase6_software_pretrain_best.pth"
run_dir="runs/vgg8_snn_v7_phase6_qat_finetune_20261007_b64"

if [[ ! -x "$python" || ! -d "$dataset" || ! -f "$checkpoint" ]]; then
    echo "Training environment, V7 dataset, or VGG8 checkpoint is unavailable" >&2
    exit 1
fi
mkdir -p "$run_dir"
if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" || \
      -e "$run_dir/vgg8_snn_v7_phase6_software_qat_finetune_latest.pth" ]]; then
    echo "Existing launch artifacts in $run_dir; refusing to overwrite" >&2
    exit 1
fi

setsid nohup env CUDA_VISIBLE_DEVICES=2 MNSIM_DVS_NUM_WORKERS=8 OMP_NUM_THREADS=4 \
    "$python" -u scripts/finetune_snn_v7.py \
    --model vgg8_snn_v7_phase6 \
    --checkpoint "$checkpoint" \
    --dataset-root "$dataset" \
    --sim-config SimConfig.ini \
    --output-dir "$run_dir" \
    --device cuda:0 \
    --epochs 20 \
    --learning-rate 0.001 \
    --train-batch 64 \
    --test-batch 64 \
    --seed 20261030 \
    > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/train.pid"
printf 'vgg8_snn_v7_phase6 FIX_TRAIN finetune physical GPU 2 batch 64 PID %s log %s/train.log\n' "$pid" "$run_dir"
