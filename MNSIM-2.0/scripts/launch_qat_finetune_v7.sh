#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${MNSIM_PYTHON:-$repo_root/../.conda-envs/mnsim-py312/bin/python}"
dataset="${MNSIM_V7_DATASET:-inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6}"

usage() {
    cat >&2 <<'EOF'
usage: launch_qat_finetune_v7.sh MODEL GPU [BATCH] [EPOCHS] [LR] [SEED] [RUN_DIR]

MODEL is one of the *snn_v7_phase6 model names. GPU is the physical CUDA index.
The command starts a resumable background job and writes its PID/log under runs/.
Set MNSIM_QAT_RUN_DIR to override the output directory.
EOF
    exit 2
}

[[ $# -ge 2 && $# -le 7 ]] || usage
model="$1"
gpu="$2"
batch="${3:-32}"
epochs="${4:-20}"
learning_rate="${5:-0.001}"
seed="${6:-20261007}"

case "$model" in
    vgg11_snn_v7_phase6|vgg13_snn_v7_phase6|vgg16_snn_v7_phase6|vgg8_snn_v7_phase6|\
    alexnet_snn_v7_phase6|lenet_snn_v7_phase6|resnet18_snn_v7_phase6|resnet34_snn_v7_phase6|resnet50_snn_v7_phase6) ;;
    *) echo "unsupported model: $model" >&2; exit 2 ;;
esac

checkpoint="${MNSIM_QAT_CHECKPOINT:-inputs/weights/pretrain/${model}_software_pretrain_best.pth}"
run_dir="${7:-${MNSIM_QAT_RUN_DIR:-runs/${model}_qat_finetune_20261007_b${batch}_lr${learning_rate}_seed${seed}}}"
latest="$run_dir/${model}_software_qat_finetune_latest.pth"

[[ -x "$python_bin" ]] || { echo "Python environment unavailable: $python_bin" >&2; exit 1; }
[[ -d "$dataset" ]] || { echo "V7 dataset unavailable: $dataset" >&2; exit 1; }
[[ -f "$checkpoint" ]] || { echo "pretrain checkpoint unavailable: $checkpoint" >&2; exit 1; }

if ! "$python_bin" - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("CUDA is unavailable; refusing to start QAT on CPU")
PY
then
    echo "CUDA driver/device is unavailable; QAT job was not started" >&2
    exit 1
fi

mkdir -p "$run_dir"
if [[ -e "$run_dir/train.log" || -e "$run_dir/train.pid" ]]; then
    if [[ -f "$latest" ]]; then
        resume_flag=(--resume)
    else
        echo "Existing launch artifacts in $run_dir without a resumable checkpoint" >&2
        exit 1
    fi
else
    resume_flag=()
fi

setsid nohup env CUDA_VISIBLE_DEVICES="$gpu" MNSIM_DVS_NUM_WORKERS="${MNSIM_DVS_NUM_WORKERS:-8}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "$python_bin" -u scripts/finetune_snn_v7.py \
    --model "$model" \
    --checkpoint "$checkpoint" \
    --dataset-root "$dataset" \
    --sim-config SimConfig.ini \
    --output-dir "$run_dir" \
    --device cuda:0 \
    --epochs "$epochs" \
    --learning-rate "$learning_rate" \
    --train-batch "$batch" \
    --test-batch "$batch" \
    --milestones "$((epochs * 50 / 100))" "$((epochs * 75 / 100))" \
    --gamma 0.1 \
    --seed "$seed" \
    "${resume_flag[@]}" \
    > "$run_dir/train.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$run_dir/train.pid"
printf '%s QAT physical GPU %s batch %s epochs %s PID %s log %s/train.log\n' \
    "$model" "$gpu" "$batch" "$epochs" "$pid" "$run_dir"
