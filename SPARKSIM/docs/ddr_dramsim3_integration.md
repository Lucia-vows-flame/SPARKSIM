# Phase 2 DDR Boundary Accounting

Round-orchestrated Phase 2 accounts for external DDR traffic at the
`BoundaryStateManager` commit boundary. The executor and report layers only
aggregate the result returned by that commit.

The checkout is pinned locally to DRAMSim3 commit
`29817593b3389f1337235d63cac515024ab8fd6e`; the external source tree is
ignored by the project repository.

The default `SimConfig.ini` keeps this backend disabled. To enable the cloned
DRAMSim3 backend, change the `[DDR]` section to:

```ini
[DDR]
DDR_Enable = 1
DDR_Backend = dramsim3
DRAMSim3_Binary = third_party/DRAMSim3/dramsim3main.out
DRAMSim3_Config = third_party/DRAMSim3/configs/DDR4_8Gb_x8_3200.ini
DDR_TransactionBytes = 64
DDR_DefaultBitwidth = 8
DRAMSim3_CyclesPerTransaction = 128
DRAMSim3_TimeoutSeconds = 120
```

Build the official checkout before enabling it:

```bash
cd third_party/DRAMSim3
make -j2
```

The backend converts boundary read/write events into aligned DRAMSim3 trace
transactions. DRAMSim3's pJ and cycle results are converted to joules and
seconds in the `ddr_transfer` result attached to each boundary commit. DDR
states are excluded from the on-chip SRAM static-power ledger.
