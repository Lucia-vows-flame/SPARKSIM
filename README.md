# MNSIM-2.0 SNN-CIM Simulator

This repository contains the MNSIM-2.0 simulator and its SNN extensions for
processing-in-memory (PIM) hardware modeling. The simulator code is the primary
project; training outputs, experiment ledgers, and logs are kept out of the
source release.

## Repository layout

- `MNSIM-2.0/MNSIM/`: simulator implementation, including round-aware execution,
  SNN temporal behavior, residual and multi-branch support, and hardware models.
- `MNSIM-2.0/configs/` and `device_models/`: example configurations and
  checksum-validated device profiles with source provenance.
- `MNSIM-2.0/scripts/`: training, scheduling, simulation, and validation entry
  points.
- `MNSIM-2.0/tests/`: regression and contract tests.

## Environment

Use Python 3.12. Install PyTorch matching the CUDA or CPU runtime available on
your system, then install the project dependencies required by the workflow you
intend to run. Training, accuracy evaluation, and hardware-only analysis have
different optional dependencies. Datasets and trained checkpoints are not
included; provide their paths explicitly.

## Run tests

Install PyTorch and the dependencies used by the selected workflow. The test
suite requires PyTorch, SpikingJelly, NumPy, SciPy, and pytest. From
`MNSIM-2.0/`, run:

```bash
python -m pytest -q
```

For an accuracy or hardware run, provide a compatible checkpoint and, for
accuracy, the corresponding dataset path through the documented CLI or wrapper
environment variables. Neither is distributed with this repository.

See [`MNSIM-2.0/README.md`](MNSIM-2.0/README.md) for simulator commands,
configuration details, scheduling formats, and SNN workflows.

## Device-model provenance

Device profiles and their source revisions, checksums, and license notices are
documented under `MNSIM-2.0/device_models/`. A 256-state lookup-table
discretization represents the simulator's numerical interface; it does not by
itself establish measured multilevel storage capability or programming
reachability.

## Publication status

No repository-wide license has been selected. See
[`PUBLICATION_CHECKLIST.md`](PUBLICATION_CHECKLIST.md) before redistributing
this repository or describing it as open source.
