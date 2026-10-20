#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

python_bin="${MNSIM_PYTHON:-python3}"

# Real standard_vgg8_snn checkpoint: 7 Conv + 1 FC + 8 Digital-LIF, T=30.
checkpoint="${STANDARD_VGG8_SNN_CHECKPOINT:-${repo_root}/inputs/weights/pretrain/vgg8_snn_v7_phase6_software_pretrain_best.pth}"

# This run is the sequential physical-round baseline. Accuracy/QAT evaluation
# is intentionally disabled; only hardware performance is evaluated.
sim_config="${STANDARD_VGG8_SNN_SIM_CONFIG:-${repo_root}/SimConfig.ini}"
output_dir="${STANDARD_VGG8_SNN_OUTPUT_DIR:-${repo_root}/results/standard_vgg8_snn_t30_g8_seq_cpu}"

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

printf 'Running standard_vgg8_snn G8 sequential baseline on CPU\n'
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
    --device cpu \
    --show-rounds
