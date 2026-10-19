#!/usr/bin/env bash
set -euo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${MNSIM_PYTHON:-python3}"
model="${1:-vgg8}"
stage="${2:-accuracy}"
device="${MNSIM_DEVICE:-0}"
dataset_root="${bundle_root}/inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"
weights_root="${bundle_root}/inputs/weights"
sim_config="${bundle_root}/SimConfig.ini"

case "${model}" in
    vgg8|standard)
        network="vgg8_snn_v7_phase6"
        checkpoint="${weights_root}/vgg8_snn_v7_phase6_software_pretrain_best.pth"
        ;;
    *)
        printf 'usage: %s {vgg8} {accuracy|hardware|full}\n' "$0" >&2
        exit 2
        ;;
esac

case "${stage}" in
    accuracy|hardware|full) ;;
    *)
        printf 'usage: %s {vgg8} {accuracy|hardware|full}\n' "$0" >&2
        exit 2
        ;;
esac

if [[ "${python_bin}" == */* ]]; then
    [[ -x "${python_bin}" ]] || {
        printf 'Python executable not found: %s\n' "${python_bin}" >&2
        exit 2
    }
elif ! command -v "${python_bin}" >/dev/null 2>&1; then
    printf 'Python executable not found on PATH: %s\n' "${python_bin}" >&2
    exit 2
fi
[[ -f "${checkpoint}" ]] || { printf 'checkpoint not found: %s\n' "${checkpoint}" >&2; exit 2; }
[[ -f "${sim_config}" ]] || { printf 'SimConfig not found: %s\n' "${sim_config}" >&2; exit 2; }
[[ -d "${dataset_root}/pretrain" && -d "${dataset_root}/test" ]] || {
    printf 'portable DVS pretrain/test splits are missing under %s\n' "${dataset_root}" >&2
    exit 2
}

output_dir="${bundle_root}/runs/${model}_${stage}"
common_args=(
    --network "${network}"
    --weights "${checkpoint}"
    --num-classes 8
    --sim-config "${sim_config}"
    --output-dir "${output_dir}"
    --overwrite
    --device "${device}"
    --schedule-policy seq
)

export MNSIM_DVS_BINNED_ROOT="${dataset_root}"
export MNSIM_DVS_TRAIN_SPLIT=pretrain
export MNSIM_DVS_TEST_SPLIT=test
export MNSIM_DVS_TRAIN_BATCH="${MNSIM_DVS_TRAIN_BATCH:-1}"
export MNSIM_DVS_TEST_BATCH="${MNSIM_DVS_TEST_BATCH:-1}"

case "${stage}" in
    accuracy)
        common_args+=(--dataset-module MNSIM.Interface.dvs_binned_npz)
        ;;
    hardware)
        common_args+=(--skip-accuracy --run-hardware --show-rounds)
        ;;
    full)
        common_args+=(--dataset-module MNSIM.Interface.dvs_binned_npz --run-hardware)
        ;;
esac

if [[ -n "${MNSIM_MAX_SAMPLES:-}" ]]; then
    common_args+=(--max-samples "${MNSIM_MAX_SAMPLES}")
fi

cd "${bundle_root}"
MNSIM_PYTHON="${python_bin}" exec bash scripts/run_round_orchestrated_sim.sh "${common_args[@]}"
