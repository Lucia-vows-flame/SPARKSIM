#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

# User options. Edit these defaults for a persistent local preset, or set the
# matching VGG8_G8_* environment variable for one invocation.
# Original MNSIM-2.0 ANN VGG8 checkpoint available on this host. Change this
# only when evaluating a different compatible VGG8 checkpoint.
# Checkpoints are intentionally supplied by the caller; the old master checkout
# is archived and must not remain an active runtime dependency.
DEFAULT_CHECKPOINT="${repo_root}/checkpoints/cifar10_vgg8_params.pth"
DEFAULT_NUM_CLASSES=10
DEFAULT_RUN_ACCURACY=0
DEFAULT_DATASET_MODULE="MNSIM.Interface.cifar10"
DEFAULT_OUTPUT_DIR="${repo_root}/results/vgg8_g8"
DEFAULT_DEVICE=0
DEFAULT_SHOW_ROUNDS=1
DEFAULT_OVERWRITE=1
# This wrapper requests hardware accounting and defaults to the calibrated ASU
# transition model. Stanford-PKU can also be selected by model id.
DEFAULT_DEVICE_PROFILE="rram_stanford_asu_256"
DEFAULT_PYTHON_BIN="python3"

# G8 is fixed at 128x128 xbars, 6x6 tiles, 4x4 PEs per tile, and two MNSIM
# groups per PE: 6 * 6 * 4 * 4 * 2 = 1152 group slots. The physical-array
# count additionally includes Xbar_Polarity from the generated SimConfig.
G8_XBAR_SIZE=128
G8_TILE_NUM=6
G8_PE_NUM=4
G8_GROUP_NUM=2

checkpoint="${VGG8_G8_CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
num_classes="${VGG8_G8_NUM_CLASSES:-${DEFAULT_NUM_CLASSES}}"
run_accuracy="${VGG8_G8_RUN_ACCURACY:-${DEFAULT_RUN_ACCURACY}}"
dataset_module="${VGG8_G8_DATASET_MODULE:-${DEFAULT_DATASET_MODULE}}"
output_dir="${VGG8_G8_OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
device="${VGG8_G8_DEVICE:-${DEFAULT_DEVICE}}"
show_rounds="${VGG8_G8_SHOW_ROUNDS:-${DEFAULT_SHOW_ROUNDS}}"
overwrite="${VGG8_G8_OVERWRITE:-${DEFAULT_OVERWRITE}}"
base_sim_config="${VGG8_G8_BASE_SIM_CONFIG:-}"
device_profile="${VGG8_G8_DEVICE_PROFILE-${DEFAULT_DEVICE_PROFILE}}"
python_bin="${VGG8_G8_PYTHON:-${MNSIM_PYTHON:-${DEFAULT_PYTHON_BIN}}}"
config_python_bin="${VGG8_G8_CONFIG_PYTHON:-${DEFAULT_PYTHON_BIN}}"

fail() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

require_flag() {
    local name="$1"
    local value="$2"
    case "${value}" in
        0|1) ;;
        *) fail "${name} must be 0 or 1, got ${value}" ;;
    esac
}

[[ -n "${checkpoint}" ]] || fail "set VGG8_G8_CHECKPOINT"
[[ -f "${checkpoint}" ]] || fail "checkpoint does not exist: ${checkpoint}"
command -v "${python_bin}" >/dev/null || fail "Python executable not found: ${python_bin}"
command -v "${config_python_bin}" >/dev/null || fail "Config Python executable not found: ${config_python_bin}"

require_flag "RUN_ACCURACY" "${run_accuracy}"
require_flag "SHOW_ROUNDS" "${show_rounds}"
require_flag "OVERWRITE" "${overwrite}"

if [[ "${run_accuracy}" == "1" && -z "${dataset_module}" ]]; then
    fail "set DATASET_MODULE or VGG8_G8_DATASET_MODULE when RUN_ACCURACY=1"
fi

# Resolve user paths before the wrapped launcher changes to the repository root.
checkpoint="$(cd -- "$(dirname -- "${checkpoint}")" && pwd)/$(basename -- "${checkpoint}")"
if [[ -n "${base_sim_config}" ]]; then
    [[ -f "${base_sim_config}" ]] || fail "base SimConfig does not exist: ${base_sim_config}"
    base_sim_config="$(cd -- "$(dirname -- "${base_sim_config}")" && pwd)/$(basename -- "${base_sim_config}")"
fi
mkdir -p -- "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd)"
g8_config="${output_dir}/SimConfig.g8.ini"

if [[ -n "${base_sim_config}" ]]; then
    source_sim_config="${base_sim_config}"
else
    source_sim_config="${repo_root}/SimConfig.ini"
fi

case "${device_profile}" in
    asu|stanford_asu|rram_stanford_asu_256) device_profile="rram_stanford_asu_256" ;;
    pku|stanford_pku_write|rram_memtorch_stanford_pku_write256) device_profile="rram_memtorch_stanford_pku_write256" ;;
    *) fail "DEVICE_PROFILE must select Stanford/ASU or Stanford-PKU, got ${device_profile}" ;;
esac

cd "${repo_root}"
"${config_python_bin}" -c '
import runpy
import sys
sys.argv = ["apply_final_rram_model.py", *sys.argv[1:]]
runpy.run_path("scripts/apply_final_rram_model.py", run_name="__main__")
' \
    --base-sim-config "${source_sim_config}" \
    --output "${g8_config}" \
    --device-model "${device_profile}"

sed -i \
    -e "s/^Xbar_Size[[:space:]]*=.*/Xbar_Size = ${G8_XBAR_SIZE},${G8_XBAR_SIZE}/" \
    -e "s/^Subarray_Size[[:space:]]*=.*/Subarray_Size = ${G8_XBAR_SIZE}/" \
    -e "s/^Group_Num[[:space:]]*=.*/Group_Num = ${G8_GROUP_NUM}/" \
    -e "s/^PE_Num[[:space:]]*=.*/PE_Num = ${G8_PE_NUM},${G8_PE_NUM}/" \
    -e "s/^Tile_Num[[:space:]]*=.*/Tile_Num = ${G8_TILE_NUM},${G8_TILE_NUM}/" \
    "${g8_config}"

g8_polarity="$(awk -F= '/^Xbar_Polarity[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2; exit}' "${g8_config}")"
[[ "${g8_polarity}" =~ ^[12]$ ]] || fail "generated SimConfig has invalid Xbar_Polarity: ${g8_polarity:-missing}"
g8_group_slots=$((G8_TILE_NUM * G8_TILE_NUM * G8_PE_NUM * G8_PE_NUM * G8_GROUP_NUM))
g8_physical_xbars=$((g8_group_slots * g8_polarity))

args=(
    --network vgg8
    --weights "${checkpoint}"
    --num-classes "${num_classes}"
    --sim-config "${g8_config}"
    --output-dir "${output_dir}"
    --run-hardware
    --device "${device}"
)

if [[ "${run_accuracy}" == "1" ]]; then
    args+=(--dataset-module "${dataset_module}")
else
    args+=(--skip-accuracy)
fi

if [[ "${show_rounds}" == "1" ]]; then
    args+=(--show-rounds)
fi

if [[ "${overwrite}" == "1" ]]; then
    args+=(--overwrite)
fi

printf 'Running VGG8 on G8: %d physical xbar arrays (%d MNSIM group slots, %dx%d xbar, %dx%d tiles, %dx%d PEs, groups=%d, polarity=%d)\n' \
    "${g8_physical_xbars}" "${g8_group_slots}" \
    "${G8_XBAR_SIZE}" "${G8_XBAR_SIZE}" \
    "${G8_TILE_NUM}" "${G8_TILE_NUM}" \
    "${G8_PE_NUM}" "${G8_PE_NUM}" "${G8_GROUP_NUM}" "${g8_polarity}"
printf 'Device model: %s\n' "${device_profile}"
printf 'Output directory: %s\n' "${output_dir}"

# Delegate all Python launching to the supported runpy-based wrapper.
MNSIM_PYTHON="${python_bin}" exec bash "${script_dir}/run_round_orchestrated_sim.sh" "${args[@]}"
