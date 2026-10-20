# SNN-FlashSeF Active-Region Retention

`snn_flashsef_temporal` extends the existing FlashSeF resident-weight
schedule with an exact fragment write mask. A fragment occupies only its
valid input/output rectangle inside a physical xbar. Cells outside that
rectangle are padding and are never observed by the lowered MNSIM operation.

At a weighted-round boundary the policy overwrites the valid rectangle and
retains the conductance state of inactive cells. A later fragment that needs a
larger rectangle overwrites that newly active region before its first read.
This uses the resident-slot state machine already present in the Phase 2
physical write ledger and removes only writes to cells that the current
fragment cannot read.

The policy keeps the existing SNN contract: one physical write boundary per
weighted round, resident weights reused for all `T` timesteps, unchanged
timestep order and LIF updates, and no bit-hop, routed-spike, NoC, or
activity-profile term. The default `flashsef_ptr` and `seq` paths retain the
historical padded-cell behavior; the optimization is opted into by the new
schedule policy artifact.

Phase 2 replay is authoritative for energy and latency. Every weighted-round
write record includes `inactive_cell_policy` so the retained-cell behavior is
auditable.

## Current VGG8 boundary

On the current two-round VGG8 plan, the policy is a valid energy optimization
but it does not reduce the end-to-end critical path: FlashSeF replay measures
`5.456382 mJ / 22.235890 ms`, while the new replay measures
`5.455320 mJ / 22.235890 ms`. The read path is identical and the write
latency is unchanged because the active fragment rectangles occupy the same
OU critical path. A claim of simultaneous energy and latency improvement
therefore requires a network with partial fragments or an additional
PE/group critical-path balancing stage.

The next stronger extension is **Temporal Critical-Path Balanced FlashSeF**:
enumerate legal PE/group slot permutations, keep only assignments whose exact
Phase 2 write energy is no greater than FlashSeF, then minimize the maximum
OU-to-PE write latency. Its candidate oracle must use the exact fragment
shapes and resident slots; compact tile-level costs are insufficient evidence.
