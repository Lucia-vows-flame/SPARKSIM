# SPARKSIM Source Release

This publication copy contains simulator source, tests, configurations, device
model tables, scripts, and documentation. Datasets, trained checkpoints, run
outputs, local Python environments, archives, and external DRAMSim3 sources
are not bundled.

Use the [repository README](../README.md) for the supported installation and
usage tutorial. Run commands from this directory, which contains
`SimConfig.ini`, and supply dataset/checkpoint paths explicitly.

`run_snn_v7.sh` is an optional local-layout convenience launcher for VGG8. It
expects inputs below `inputs/`, checks for dataset splits even in its hardware
mode, and does not replace the general CLI documented in the main tutorial.
