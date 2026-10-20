# SNN-Aware Weight Scheduling

`snn_flashsef_ptr` is the compatibility name for the SNN temporal resident
state policy. It keeps the RoundPlan, round order, timestep order, LIF updates,
partial-result barriers, and Phase 2 execution unchanged. Its SNN awareness is
the fixed temporal execution contract: weights remain resident across all `T`
timesteps and are reassigned only at weighted round boundaries.

The policy uses the active SimConfig's native source-to-target write primitive:

- per-cell programming energy from the device transition table;
- per-cell pulse latency, aggregated by OU and PE as Phase 2 does;
- resident state for physical tiles and idle fragment slots;
- canonical sequential assignment as a non-regression candidate.

Candidates are accepted only when both native write energy and native write
latency are no worse than the sequential assignment, subject to the configured
tolerance. Among accepted candidates, the policy minimizes native write energy,
then native write latency, then changed cells for deterministic tie-breaking.

The policy does not use routed-spike traffic, NoC distance, bit-hop counts, or
calibration activity profiles. Those quantities are outside the current MNSIM
energy and latency accounting paths. The deprecated
`--snn-activity-profile` argument remains accepted for command-line
compatibility but is ignored by this policy.

## Invocation

```sh
python -m MNSIM.Interface.round_orchestrated_cli \
  --network <snn-network> --weights <checkpoint> --num-classes <n> \
  --sim-config SimConfig.ini --output-dir <output> --run-hardware \
  --schedule-only --schedule-plan-out <schedule.json> \
  --schedule-policy snn_flashsef_ptr
```

The schedule artifact records the native transition source, the sequential
counterfactual, the selected energy and latency, and the fixed round order.
Phase 2 replay remains authoritative for end-to-end energy and latency.
