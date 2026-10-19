# Portable MNSIM-2.0 Bundle

This directory is a self-contained copy of the active
`task-20260707-round-orchestrated-mnsim` project plus the runtime inputs used by
the two final v7 SNN models.

Included:

- active MNSIM source, tests, configs, scripts, docs, and device artifacts;
- `third_party/DRAMSim3` source checkout and configuration files;
- both software-pretrain v7 checkpoints under `inputs/weights/`;
- the complete four-split DVS dataset under `inputs/datasets/`;
- `run_snn_v7.sh`, which resolves every project input relative to this bundle.
- `PORTABLE_PYTHON_ENV.txt` and `PORTABLE_RUNTIME_INFO.txt` for reproducing the
  source Python environment on the destination server.

Excluded deliberately:

- `archive/`: legacy architecture and historical worktrees;
- `results/`: experiment outputs and interrupted runs;
- `.git/`, `.venv/`, Python caches, and host-specific Codex metadata.

The current `SimConfig.ini` keeps DDR disabled. DRAMSim3 is included for later
optional use, but a new server must build its binary before enabling DDR.

## Requirements

Install Python dependencies in the target environment, including PyTorch,
SpikingJelly, NumPy, SciPy, and pytest as needed. Set `MNSIM_PYTHON` to that
environment's Python executable; otherwise `python3` is used.

## Run the v7 SNN flows

From this directory:

```bash
MNSIM_PYTHON=/path/to/python ./run_snn_v7.sh standard accuracy
MNSIM_PYTHON=/path/to/python ./run_snn_v7.sh modified accuracy
MNSIM_PYTHON=/path/to/python MNSIM_DEVICE=0 ./run_snn_v7.sh standard hardware
MNSIM_PYTHON=/path/to/python MNSIM_DEVICE=0 ./run_snn_v7.sh modified full
```

The hardware policy is fixed to `seq`; no `flash-ptr` policy is selected by
the launcher. Accuracy uses the software-pretrain checkpoint, physical
`pretrain` calibration data, and physical `test` data. Set
`MNSIM_MAX_SAMPLES` to a positive integer for a smoke run.

The launcher writes outputs below `runs/` inside this directory.
