# Bundled RRAM Models

Only the Stanford/ASU and Stanford-PKU device families are retained. Their
canonical JSON files are checksum-locked in
`MNSIM/Interface/final_rram_contract.py`.

| Device | Model ID | Runtime role |
| --- | --- | --- |
| Stanford/ASU | `rram_stanford_asu_256` | 256-state conductance table and calibrated write transition matrix |
| Stanford-PKU | `rram_memtorch_stanford_pku_256` | 256-state nominal read/accuracy table |
| Stanford-PKU | `rram_memtorch_stanford_pku_write256` | 256-state nominal table with compact-model write transitions |

The ASU model is generated from the delivered Stanford/ASU model package and
its sequential 256-state readback data. It uses the established RESET/SET
protocol and computes transition costs offline from the compact model; MNSIM
loads the resulting table and does not run HSPICE.

Stanford-PKU is derived from the pinned MemTorch Stanford-PKU model. The
nominal read profile discretizes the continuous conductance range into 256
codes. The write profile adds ideal compact-model state transitions. Its
programming reachability is a model prediction, not measured 256-level device
evidence. See `README_stanford_pku_256.md` for source, conversion, and limits.

Use `configs/SimConfig_rram_stanford_asu_256.ini` for ASU runs,
`configs/SimConfig_rram_memtorch_stanford_pku_256.ini` for PKU read/accuracy
runs, and `configs/SimConfig_rram_memtorch_stanford_pku_write256.ini` when
hardware write accounting is required. The nominal-only PKU profile is
rejected by hardware write workflows.
