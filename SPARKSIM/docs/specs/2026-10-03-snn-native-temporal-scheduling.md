# SNN-Native Temporal Weight Scheduling

## Purpose

`snn_native_temporal` is the native-cost implementation behind the public
compatibility policy name `snn_flashsef_ptr`. It contains no routed-spike,
NoC, bit-hop, or activity-distance term. The policy uses the SNN model contract
to keep the weighted resident state fixed across all `T` timesteps, then
optimizes legal round placement against the physical write transition
primitive already used by Phase 2.

The policy is an artifact-generation policy. It does not alter convolution,
LIF equations, reset behavior, timestep order, partial-result barriers, or
branch merge semantics.

## Native cost contract

The scheduler loads the final RRAM model from `SimConfig.ini` and derives the
same source-code to target-code tables as
`MNSIM.Scheduling.phase2.physical_write_ledger`:

- per-cell programming energy;
- per-cell pulse latency;
- reset and set pulse counts;
- erased code-zero initial state.

For the native SNN path, every logical tile also carries an internal
fragment-to-[PE, group, polarity] payload. The scheduler pads each active
fragment matrix to a physical xbar, applies the exact per-cell table, and
retains resident state for slots absent from the current round. This is the
same active-slot and erased-initial-state contract used by the Phase 2 ledger;
the transition table is compared element-for-element before a schedule is used
for experiments.

The scheduler's latency objective applies the native OU maximum and PE maximum
inside each tile. Phase 2 remains authoritative for cross-tile round
concurrency and end-to-end latency reporting. The schedule artifact records the
native fragment-slot provenance, while the large internal payload is
intentionally omitted from the public artifact because Phase 2 replays the
frozen placement and checkpoint codes.

## Placement objective

For every fixed round, the policy evaluates:

1. canonical sequential placement;
2. minimum native write-energy assignment;
3. minimum native pulse-latency assignment.

The sequential placement is always a legal candidate. A candidate is selected
only when it is no worse than sequential on both native energy and native
latency, subject to the configured tolerance. This prevents the scheduler from
trading a small write-energy reduction for a latency regression without an
explicit experiment setting.

The artifact records both the selected native metrics and the sequential
counterfactual for every round. `changed_cells` remains an audit field and is
not the native objective.

## Legal round-order search

Round order is searched only inside an explicitly declared contiguous weighted
block. A round may list `predecessor_round_ids`; candidate permutations that
violate those dependencies are rejected. Zero-xbar structural rounds and
unmarked weighted rounds remain fixed. The selected order is written into the
artifact, copied into the frozen `RoundPlan`, reindexed, rehashed, and replayed
by Phase 2 without replanning.

The current VGG8 plan has a dependency chain, so its order remains `R0, R1`.
Networks with independent residual or multi-branch weighted rounds can expose a
larger legal block and therefore provide the main evaluation target.

## Invocation

```sh
python -m MNSIM.Interface.round_orchestrated_cli \
  --network <snn-network> --weights <checkpoint> --num-classes <n> \
  --sim-config SimConfig.ini --output-dir <output> --run-hardware \
  --schedule-only --schedule-plan-out <schedule.json> \
  --schedule-policy snn_native_temporal
```

Replay uses the existing frozen artifact path:

```sh
python -m MNSIM.Interface.round_orchestrated_cli \
  --network <snn-network> --weights <checkpoint> --num-classes <n> \
  --sim-config SimConfig.ini --output-dir <output> --run-hardware \
  --infer-with-schedule --schedule-plan-in <schedule.json> \
  --schedule-policy snn_native_temporal
```

## Claim boundary

This version can support a claim about SNN-constrained resident-aware weight
scheduling and native RRAM write-cost reduction. It cannot support a claim
about routed spike traffic or NoC energy. End-to-end gains must be measured by
Phase 2 replay, and all functional outputs, LIF update counts, and branch
completion barriers must match the sequential artifact.
