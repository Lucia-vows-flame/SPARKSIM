#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${MNSIM_PYTHON:-python3}"

cd "${repo_root}"
exec "${python_bin}" -c '
import runpy
import sys
sys.argv = ["run_pim_numerical_audit.py", *sys.argv[1:]]
runpy.run_path("scripts/run_pim_numerical_audit.py", run_name="__main__")
' "$@"
