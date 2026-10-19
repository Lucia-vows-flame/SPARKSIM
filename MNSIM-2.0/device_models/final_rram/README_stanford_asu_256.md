# Stanford/ASU 256-level readback model

`rram_stanford_asu_256` is the MNSIM numerical adapter for the
`stanford_asu_rram_mnsim` delivery package. It loads the 256 HSPICE readback
levels as the `Device_Resistance` table used by MNSIM accuracy and conductance
paths.

This integration intentionally does not run HSPICE. The Verilog-A/CML files
remain provenance artifacts; the current MNSIM path is the discrete software
model defined by `rram_stanford_asu_256.json` and its level table.

The imported model now includes a complete 256x256 source-state to target-state
transition table. Increasing-code transitions use the measured sequential SET
pulses. Decreasing-code transitions use the delivered 100 ns RESET protocol
followed by SET to the target. Energy is integrated from the compact model's
nonlinear I-V equation over the gap trajectory, with each interval calibrated
to its measured pulse width. The table is consumed by the scheduler and the
Phase 2 physical write ledger.

These are deterministic compact-model predictions, not measured array-level
energy or latency. Device and cycle variation, read noise, retention, and
peripheral programming energy remain outside this table.

Use `configs/SimConfig_rram_stanford_asu_256.ini` for MNSIM software runs.
