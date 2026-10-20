#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
launcher="$repo_root/scripts/launch_qat_finetune_v7.sh"
queue_dir="runs/qat_queue_20261007"
mkdir -p "$queue_dir"
queue_log="$queue_dir/queue.log"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" >> "$queue_log"
}

wait_for_pid() {
    local pid="$1"
    while kill -0 "$pid" 2>/dev/null; do
        sleep 10
    done
}

worker() {
    local gpu="$1"
    shift
    while (($#)); do
        local model="$1" batch="$2" seed="$3" run_dir
        shift 3
        run_dir="runs/${model}_qat_finetune_20261007_b${batch}_lr0.001_seed${seed}"
        log "waiting for ${model} on GPU ${gpu}"
        while ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; do
            sleep 10
        done
        local output pid
        output="$(bash "$launcher" "$model" "$gpu" "$batch" 20 0.001 "$seed" "$run_dir" 2>&1)" || {
            log "launch failed for ${model}: ${output}"
            return 1
        }
        log "$output"
        pid="$(printf '%s\n' "$output" | sed -n 's/.* PID \([0-9][0-9]*\) log .*/\1/p')"
        if [[ -z "$pid" ]]; then
            log "could not read PID for ${model}"
            return 1
        fi
        wait_for_pid "$pid"
        if ! rg -q 'final test accuracy:' "$run_dir/train.log"; then
            log "${model} stopped before completion; leaving checkpoint for resume"
            return 1
        fi
        log "${model} completed"
    done
}

log "queue started; waiting for active VGG11/VGG13 QAT jobs"
wait_for_pid 3812420
wait_for_pid 3813472
log "initial VGG11/VGG13 QAT jobs finished"

worker 0 \
    vgg16_snn_v7_phase6 32 20261016 \
    resnet18_snn_v7_phase6 16 20261018 \
    resnet50_snn_v7_phase6 8 20261020 &
gpu0_worker=$!
worker 3 \
    lenet_snn_v7_phase6 128 20261017 \
    alexnet_snn_v7_phase6 64 20261019 \
    resnet34_snn_v7_phase6 16 20261021 &
gpu3_worker=$!

wait "$gpu0_worker"
wait "$gpu3_worker"
log "all queued QAT runs finished"
