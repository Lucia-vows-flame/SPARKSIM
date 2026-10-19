#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${MNSIM_PYTHON:-python3}"

# Real standard_vgg8_snn checkpoint: 7 Conv + 1 FC + 8 Digital-LIF, T=30.
checkpoint="${STANDARD_VGG8_SNN_CHECKPOINT:-${repo_root}/inputs/weights/pretrain/vgg8_snn_v7_phase6_software_pretrain_best.pth}"

# This is the same sequential physical-round baseline as the CPU launcher.
# Accuracy/QAT evaluation is disabled; only hardware performance is evaluated.
sim_config="${STANDARD_VGG8_SNN_SIM_CONFIG:-${repo_root}/SimConfig.ini}"
output_dir="${STANDARD_VGG8_SNN_OUTPUT_DIR:-${repo_root}/results/standard_vgg8_snn_t30_g8_seq_gpu}"
gpu_index="${STANDARD_VGG8_SNN_GPU:-0}"

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

if [[ "${python_bin}" == */* ]]; then
    [[ -x "${python_bin}" ]] || fail "Python executable not found: ${python_bin}"
else
    command -v "${python_bin}" >/dev/null 2>&1 || fail "Python executable not found: ${python_bin}"
fi
[[ -f "${checkpoint}" ]] || fail "checkpoint not found: ${checkpoint}"
[[ -f "${sim_config}" ]] || fail "SimConfig not found: ${sim_config}"
[[ "${gpu_index}" =~ ^[0-9]+$ ]] || fail "GPU index must be a non-negative integer: ${gpu_index}"

# Fail fast when CUDA is unavailable or the requested GPU index is invalid.
"${python_bin}" -c '
import sys
import torch

index = int(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the selected Python environment")
count = torch.cuda.device_count()
if index >= count:
    raise SystemExit(f"GPU index {index} is invalid; visible CUDA device count is {count}")
print(f"CUDA device {index}: {torch.cuda.get_device_name(index)}")
' "${gpu_index}"

printf 'Running standard_vgg8_snn G8 sequential baseline on GPU %s\n' "${gpu_index}"
printf 'Checkpoint: %s\n' "${checkpoint}"
printf 'SimConfig:  %s\n' "${sim_config}"
printf 'Output:     %s\n\n' "${output_dir}"

# The wrapped CLI prints physical rounds, latency, energy, TOPS, and TOPS/W.
MNSIM_PYTHON="${python_bin}" exec bash "${script_dir}/run_round_orchestrated_sim.sh" \
    --network standard_vgg8_snn \
    --weights "${checkpoint}" \
    --num-classes 8 \
    --sim-config "${sim_config}" \
    --output-dir "${output_dir}" \
    --run-hardware \
    --skip-accuracy \
    --overwrite \
    --device "${gpu_index}" \
    --show-rounds
