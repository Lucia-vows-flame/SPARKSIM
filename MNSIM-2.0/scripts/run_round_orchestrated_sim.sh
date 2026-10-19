#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${MNSIM_PYTHON:-python3}"
cuda_preflight="${MNSIM_CUDA_PREFLIGHT:-1}"

device=""
expect_device_value=0
for argument in "$@"; do
    if (( expect_device_value )); then
        device="${argument}"
        expect_device_value=0
        continue
    fi
    case "${argument}" in
        --device)
            expect_device_value=1
            ;;
        --device=*)
            device="${argument#--device=}"
            ;;
    esac
done

case "${device}" in
    cuda|cuda:[0-9]*|[0-9]*)
        if [[ "${cuda_preflight}" != "0" ]]; then
            gpu_index="${device}"
            if [[ "${gpu_index}" == "cuda" ]]; then
                gpu_index=0
            elif [[ "${gpu_index}" == cuda:* ]]; then
                gpu_index="${gpu_index#cuda:}"
            fi
            if ! nvidia-smi --query-gpu=index --format=csv,noheader >/dev/null 2>&1; then
                "${python_bin}" -c '
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA preflight failed: nvidia-smi failed and Torch reports CUDA unavailable")
index = int(sys.argv[1])
if index >= torch.cuda.device_count():
    raise SystemExit(f"CUDA preflight failed: GPU index {index} is out of range")
            ' "${gpu_index}"
            fi
        fi
        ;;
esac

cd "${repo_root}"
exec "${python_bin}" -c '
import os
import subprocess
import runpy
import sys

def _requested_gpu(arguments):
    value = None
    expect_value = False
    for argument in arguments:
        if expect_value:
            value = argument
            expect_value = False
        elif argument == "--device":
            expect_value = True
        elif argument.startswith("--device="):
            value = argument.split("=", 1)[1]
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "cuda":
        return 0
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    if text.isdigit():
        return int(text)
    return None

gpu_index = _requested_gpu(sys.argv[1:])
if gpu_index is not None and os.environ.get("MNSIM_CUDA_PREFLIGHT", "1") != "0":
    subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        check=False,
        stdout=subprocess.DEVNULL,
    )

sys.argv = ["round_orchestrated_cli.py", *sys.argv[1:]]
runpy.run_path("MNSIM/Interface/round_orchestrated_cli.py", run_name="__main__")
' "$@"
