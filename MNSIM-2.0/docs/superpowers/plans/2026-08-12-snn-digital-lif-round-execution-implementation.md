# SNN Digital-LIF Round Execution Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with TDD. Use the
> repository Python through `python -m pytest`; do not run repository Python
> files as bare scripts.

**Goal:** Extend the current round-orchestrated ANN simulator into the confirmed
feed-forward SNN v1 runtime: each weighted physical round programs once and
executes all `T` timesteps, completed ADC-derived accumulator codes enter a
stateful digital LIF without requantization, and reports include LIF compute and
on-chip state costs while continuing to exclude cross-round DDR transfer cost.

**Architecture:** Preserve the existing
`ModelIR -> RoundPlan -> phase2.execute_round_plan` path. Add accumulator-domain
metadata and SNN validation to Phase 1, timestep identity to the existing
boundary and round-local state managers, and a new `digital_lif.py` module for
the integer reference operation, state store, Tile ownership, Buffer traffic,
and composed hardware cost. Keep execution round-major and reuse the existing
ordered segments. ANN remains the `T = 1`, no-LIF regression case.

**Tech Stack:** Python standard library, existing MNSIM hardware primitives
(`Adder`, `ShiftReg`, `Reg`, `JointModule`, `Buffer`), the existing
Scheduling Phase 1/2 modules, and `pytest`.

---

## Source Spec

Implement exactly:

`docs/superpowers/specs/2026-08-12-snn-digital-lif-round-execution-design.md`

Do not implement analog LIF/Vmem, HSPICE LIF LUTs, per-timestep weight
programming, DDR transfer cost, state preemption/spill, general recurrent
graphs, activity-aware gating, accuracy, or QAT execution.

## Working Directory

Run all commands from:

```bash
cd /home/xzh/workspace/cim_project/MNSIM-2.0/.worktrees/task-20260707-round-orchestrated-mnsim/round_orchestrated_simulator/MNSIM-2.0
```

Use:

```bash
GIT=/home/xzh/micromamba/envs/workspace/bin/git
PYTHON=/home/xzh/micromamba/envs/workspace/bin/python
```

All tests use `$PYTHON -m pytest ...`. Do not run repository Python files as
bare scripts. Do not update `exports/` until canonical code and tests pass.

## Planned Files

Create:

- `MNSIM/Scheduling/digital_domain.py`
- `MNSIM/Scheduling/phase2/digital_lif.py`
- `tests/test_round_snn_accumulator_domain_characterization.py`
- `tests/test_round_snn_digital_domain.py`
- `tests/test_round_snn_phase1.py`
- `tests/test_round_snn_boundary.py`
- `tests/test_round_snn_digital_lif.py`
- `tests/test_round_snn_lif_cost.py`
- `tests/test_round_snn_executor.py`
- `tests/test_round_snn_report.py`
- `tests/test_round_snn_integration.py`

Modify:

- `MNSIM/Scheduling/round_types.py`
- `MNSIM/Scheduling/non_weighted_layers.py`
- `MNSIM/Scheduling/phase1/mnsim_frontend.py`
- `MNSIM/Scheduling/phase1/round_planner.py`
- `MNSIM/Scheduling/phase2/types.py`
- `MNSIM/Scheduling/phase2/boundary.py`
- `MNSIM/Scheduling/phase2/round_local_state.py`
- `MNSIM/Scheduling/phase2/lowerer.py`
- `MNSIM/Scheduling/phase2/executor.py`
- `MNSIM/Scheduling/phase2/report.py`
- `MNSIM/Scheduling/phase2/__init__.py`
- `MNSIM/Interface/round_orchestrated_runner.py`
- `MNSIM/Interface/round_orchestrated_cli.py`
- `SimConfig.ini`

Do not modify `physical_write_ledger.py` except for a strictly necessary
compatibility fix. Its rule remains one programming event per weighted round.

## Public Contracts

Preserve:

```python
build_round_plan(model_ir, sim_config) -> RoundPlan
execute_round_plan(...) -> Phase2Report
run_round_orchestrated_simulation(request) -> RoundOrchestratedRunResult
```

Add JSON-friendly forms of:

```python
AccumulatorDomain
CompiledLifConstants
InvocationKey
DigitalLifRequest
DigitalLifResult
LifStateRecord
LifStateStore
DigitalLifCostModel
compile_lif_constants(...)
execute_digital_lif_update(...)
```

---

### Task 0: Characterize The Existing Completed-Accumulator Domain

**Files:**
- Create: `tests/test_round_snn_accumulator_domain_characterization.py`
- Inspect first: `MNSIM/Interface/quantize.py`
- Inspect first: `MNSIM/Accuracy_Model/pim_numerical_audit.py`
- Inspect first: `MNSIM/Scheduling/phase2/mnsim_static_eval.py`
- Inspect first: `MNSIM/Scheduling/phase2/mnsim_fragment_adapter.py`

- [ ] **Step 1: Write characterization tests around the current numerical path**

Build tiny signed Conv/FC cases with known ADC codes and weight planes. Record:

```text
ADC output code width and code scale
bit-slice/weight-plane shifts
row/subarray partial additions
PE/Tile partial additions
final completed accumulator code and sufficient signed width
```

The reference identity must be:

```text
physical output = completed_accumulator_code * S_acc
```

Do not invoke an activation output quantizer after the completed sum.

- [ ] **Step 2: Prove which existing metadata is authoritative**

Use the current ADC and bit-slice calculation, not
`quantize.last_activation_scale`, a QAT output scale, or a post-activation
quantizer. If the performance-only path does not expose enough information,
add a pure metadata derivation helper based on layer/hardware configuration;
do not run accuracy inference merely to obtain the descriptor.

- [ ] **Step 3: Freeze width/scale formulas with boundary cases**

Test positive/negative extrema, multiple weight planes, multiple row partials,
and split physical fragments. The derived width must be sufficient for exact
addition and identical for unsplit versus split-then-completed execution.

- [ ] **Step 4: Run the characterization suite**

```bash
$PYTHON -m pytest tests/test_round_snn_accumulator_domain_characterization.py tests/test_pim_numerical_audit.py -v
```

Do not begin Task 1 until this test establishes a reproducible source for
`B_acc` and `S_acc` without post-ADC requantization.

---

### Task 1: Accumulator Domain And Compiled Constants

**Files:**
- Create: `MNSIM/Scheduling/digital_domain.py`
- Create: `tests/test_round_snn_digital_domain.py`
- Modify: `MNSIM/Scheduling/round_types.py`

- [ ] **Step 1: Write failing domain tests**

Cover deterministic serialization; exact scale identity; bit-exact compatible
handoff; rejection of different id/width/scale; nearest-even compilation of
threshold/reset/initial state; `initial_state_code == reset_code`; compile
error reporting; and signed-range failure.

Represent scale as exact numerator/denominator fields, not an unqualified
binary float. Identical metadata must produce the same `domain_id`.

- [ ] **Step 2: Run the failing test**

```bash
$PYTHON -m pytest tests/test_round_snn_digital_domain.py -v
```

Expected: import failure for `MNSIM.Scheduling.digital_domain`.

- [ ] **Step 3: Implement minimal types and compiler**

Use frozen dataclasses:

```python
AccumulatorDomain(
    signed_bit_width,
    scale_numerator,
    scale_denominator,
    domain_id,
)
CompiledLifConstants(
    threshold_code,
    reset_code,
    initial_state_code,
    source_values,
    compile_errors,
    domain,
    rounding_mode,
)
```

Use exact rational/decimal arithmetic. Apply nearest-even only at compile time
and verify signed-width representability.

Construct each descriptor from the characterized completed-accumulator
metadata in Task 0. Do not substitute activation/QAT scale metadata.

- [ ] **Step 4: Add backward-compatible domain fields**

Add defaulted domain metadata to `LocalGraphNode` or a dedicated JSON-friendly
field reachable from it. Existing ANN plans without it must deserialize.

- [ ] **Step 5: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_digital_domain.py tests/test_round_phase1_types.py -v
$GIT add MNSIM/Scheduling/digital_domain.py MNSIM/Scheduling/round_types.py tests/test_round_snn_digital_domain.py
$GIT commit -m "feat: add accumulator domain contract"
```

---

### Task 2: Phase 1 SNN Validation And Metadata

**Files:**
- Create: `tests/test_round_snn_phase1.py`
- Modify: `MNSIM/Scheduling/non_weighted_layers.py`
- Modify: `MNSIM/Scheduling/phase1/mnsim_frontend.py`
- Modify: `MNSIM/Scheduling/phase1/round_planner.py`

- [ ] **Step 1: Write failing Phase 1 tests**

Cover:

```text
valid T>1 ordered recurrent LIF
unknown/duplicate/non-LIF recurrent ids rejected
LIF has exactly one completed input domain
weighted -> LIF domain inheritance
same-domain digital merge accepted
zero-input, independent multi-input, and mismatched merge domains rejected
tau/v_th/v_reset/reset metadata preserved
LIF uses zero xbars
logical output-neuron region needed to derive Tile ownership is preserved
ANN T=1 serialization remains compatible
```

- [ ] **Step 2: Run the failing test**

```bash
$PYTHON -m pytest tests/test_round_snn_phase1.py -v
```

- [ ] **Step 3: Preserve frontend metadata**

Preserve functional attributes, neuron shape/count, compiled domain/constants,
and the logical output-neuron region. Constants must be compiled before Phase
2. Do not invent `N_k` in the frontend: final Tile ownership is unavailable
until Phase 2 placement is prepared.

- [ ] **Step 4: Add planner validation**

Extend sequence/model validation with stable errors. Preserve existing
full-layer pack and single-layer split rules; do not add an SNN planner. The
plan must retain enough producer-region identity for Phase 2 to derive `N_k`
from the real placement.

- [ ] **Step 5: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_phase1.py tests/test_mnsim_non_weighted_layer_registry.py tests/test_round_phase1_types.py tests/test_round_phase1_mnsim_frontend.py tests/test_round_phase1_planner.py -v
$GIT add MNSIM/Scheduling/non_weighted_layers.py MNSIM/Scheduling/phase1/mnsim_frontend.py MNSIM/Scheduling/phase1/round_planner.py tests/test_round_snn_phase1.py
$GIT commit -m "feat: validate digital lif graphs in phase1"
```

---

### Task 3: Sequence-Aware Boundary And Local State

**Files:**
- Create: `tests/test_round_snn_boundary.py`
- Modify: `MNSIM/Scheduling/phase2/types.py`
- Modify: `MNSIM/Scheduling/phase2/boundary.py`
- Modify: `MNSIM/Scheduling/phase2/round_local_state.py`

- [ ] **Step 1: Write failing boundary tests**

Cover:

```text
InvocationKey = (sample_id, node_id, timestep)
partial identity includes sample and timestep
same fragment id at different timesteps is legal
same-timestep duplicate fragment fails
each timestep remains partial until its full fragment coverage
physical [T,...] batching retains logical timestep audit
spike_sequence keeps T order and one-bit logical width
same-round handoff emits no DDR event
second sample cannot observe first sample state
```

- [ ] **Step 2: Run the failing test**

```bash
$PYTHON -m pytest tests/test_round_snn_boundary.py -v
```

- [ ] **Step 3: Extend defaulted Phase 2 records**

Add `sample_id`, timestep/range, sequence length, bit width, domain, and
payload kind without breaking current ANN constructors.

- [ ] **Step 4: Sequence-key completion and duplicate checks**

Key logical completion by sample and timestep. Physical batching remains an
implementation detail. Continue reporting DDR bytes/events with zero transfer
latency and energy.

- [ ] **Step 5: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_boundary.py tests/test_round_phase2_types.py tests/test_round_phase2_boundary.py tests/test_round_phase2_round_local_state.py -v
$GIT add MNSIM/Scheduling/phase2/types.py MNSIM/Scheduling/phase2/boundary.py MNSIM/Scheduling/phase2/round_local_state.py tests/test_round_snn_boundary.py
$GIT commit -m "feat: add timestep-aware boundary state"
```

---

### Task 4: Integer Digital LIF And State Store

**Files:**
- Create: `MNSIM/Scheduling/phase2/digital_lif.py`
- Create: `tests/test_round_snn_digital_lif.py`
- Modify: `MNSIM/Scheduling/phase2/__init__.py`

- [ ] **Step 1: Write failing integer tests**

Test exact v1 arithmetic:

```text
delta = I - V
p = delta * 24130
q = arithmetic floor shift(p, 15)
nearest-even by remainder/tie/parity
u = V + scaled
spike = u >= threshold
hard reset
checked B-bit writeback
```

Include positive/negative non-ties and half-way ties, zero input, threshold
equality, signed extrema, exact writeback, and overflow. Prove there is no
input requantization, clipping, wraparound, or saturation.

- [ ] **Step 2: Run the failing test**

```bash
$PYTHON -m pytest tests/test_round_snn_digital_lif.py -v
```

- [ ] **Step 3: Implement pure integer functions**

Keep arithmetic independent from hardware cost classes so differential tests
are cheap and deterministic.

- [ ] **Step 4: Implement `LifStateStore`**

Key by `(sample_id, lif_node_id)`; initialize from reset; require consecutive
timesteps; reject duplicates/skips/domain mismatch; prove exactly `T`
invocations; omit terminal Buffer write in v1.

- [ ] **Step 5: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_digital_lif.py -v
$GIT add MNSIM/Scheduling/phase2/digital_lif.py MNSIM/Scheduling/phase2/__init__.py tests/test_round_snn_digital_lif.py
$GIT commit -m "feat: implement integer digital lif state"
```

---

### Task 5: Digital LIF Hardware Cost

**Files:**
- Create: `tests/test_round_snn_lif_cost.py`
- Modify: `MNSIM/Scheduling/phase2/digital_lif.py`
- Modify: `SimConfig.ini`

- [ ] **Step 1: Write failing cost/config tests**

Cover required `[Digital LIF]` keys, beta code/CSD, full operation inventory,
`7.329 ns` reference latency, `N*T` energy scaling, lane/width area scaling,
8/16/32-bit monotonic cost, legacy <=16-bit slicing, and unsupported profile
failure.

- [ ] **Step 2: Add the formal SimConfig section**

Add exact spec keys without changing existing ANN peripheral values.

- [ ] **Step 3: Compose the per-lane hardware**

```text
1 W_delta Adder: I-V
4 W_prod Adders: CSD tree
1 W_prod ShiftReg: arithmetic shift
1 W_prod JointModule: rounding-decision proxy
1 W_delta Adder: conditional increment
1 W_state Adder: state addition
1 W_state JointModule: compare proxy
1 W_state JointModule: reset mux proxy
3 B-bit registers + 1 spike bit
```

Use explicit per-bit models when configured. Slice legacy lookup primitives
into 16-bit-or-smaller instances and report slice counts.

- [ ] **Step 4: Add ownership and Buffer formulas**

After `prepare_round_placement`, derive actual `N_k` by intersecting each
producer output-neuron region with its prepared Tile/fragment placement. Reject
missing, overlapping, or incomplete ownership coverage. Then apply:

```text
batches_per_step = max_k ceil(N_k / 64)
state_bytes[k] = ceil(N_k * B / 8)
Buffer ceilings apply separately per Tile and timestep
Tile energy sums; parallel Tile latency takes max
read, compute, write do not overlap
```

Use existing `Buffer`; never charge excluded DDR transfer cost.

- [ ] **Step 5: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_lif_cost.py tests/test_custom_buffer_energy.py tests/test_peripheral_literature_profile.py -v
$GIT add MNSIM/Scheduling/phase2/digital_lif.py SimConfig.ini tests/test_round_snn_lif_cost.py
$GIT commit -m "feat: model digital lif hardware cost"
```

---

### Task 6: Lowering And Round-Major Execution

**Files:**
- Create: `tests/test_round_snn_executor.py`
- Modify: `MNSIM/Scheduling/phase2/lowerer.py`
- Modify: `MNSIM/Scheduling/phase2/executor.py`

- [ ] **Step 1: Write failing executor tests**

Prove ANN call order is unchanged; each weighted round programs once; weighted
read/non-weighted/LIF work runs `T` times; zero-xbar LIF runs `T` times
without programming; same-round chains execute per timestep; split rounds
store all timestep partials without early LIF; closing rounds execute LIF
`t=0..T-1`; exactly-once holds; and no cross-timestep overlap is reported.

- [ ] **Step 2: Run the failing test**

```bash
$PYTHON -m pytest tests/test_round_snn_executor.py -v
```

- [ ] **Step 3: Lower LIF to real work**

Return `DigitalLifRequest`, not the current zero-cost hook. Leave ordinary
hooks and structural non-weighted handling unchanged.

- [ ] **Step 4: Refactor the executor loop**

Keep execution-item/round as the outer loop. Prepare and program once, then run
canonical segments for each timestep, with sample/timestep state identity, and
finalize batched boundaries after the item. Do not multiply write cost by `T`.

- [ ] **Step 5: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_executor.py tests/test_round_phase2_lowerer.py tests/test_round_phase2_executor.py tests/test_ann_round_physical_write_ledger.py -v
$GIT add MNSIM/Scheduling/phase2/lowerer.py MNSIM/Scheduling/phase2/executor.py tests/test_round_snn_executor.py
$GIT commit -m "feat: execute snn timesteps in physical rounds"
```

---

### Task 7: Metrics And Reports

**Files:**
- Create: `tests/test_round_snn_report.py`
- Modify: `MNSIM/Scheduling/phase2/types.py`
- Modify: `MNSIM/Scheduling/phase2/report.py`
- Modify: `MNSIM/Scheduling/phase2/executor.py`

- [ ] **Step 1: Write failing report tests**

Require sequence/sample identity, LIF invocation count, domain metadata,
compiled constants/errors, `N_k`, batches, proxy/slice costs, LIF compute and
state ledgers, sequence boundary bytes, DDR exclusion, programming outside
timestep totals, and dense-equivalent SNN operation count.

Verify:

```text
round energy = program + weighted read + ordinary non-weighted
               + digital LIF compute + on-chip LIF state
round latency = program + serialized sequence execution
```

- [ ] **Step 2: Run the failing test**

```bash
$PYTHON -m pytest tests/test_round_snn_report.py -v
```

- [ ] **Step 3: Extend aggregation compatibly**

Keep ANN keys. Add SNN fields. LIF operations contribute latency/energy but not
the TOPS numerator; report `lif_update_count` separately.

- [ ] **Step 4: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_report.py tests/test_round_phase2_report.py tests/test_round_orchestrated_metrics.py -v
$GIT add MNSIM/Scheduling/phase2/types.py MNSIM/Scheduling/phase2/report.py MNSIM/Scheduling/phase2/executor.py tests/test_round_snn_report.py
$GIT commit -m "feat: report digital lif performance costs"
```

---

### Task 8: Runner, CLI, And Artifacts

**Files:**
- Create: `tests/test_round_snn_integration.py`
- Modify: `MNSIM/Interface/round_orchestrated_runner.py`
- Modify: `MNSIM/Interface/round_orchestrated_cli.py`
- Modify: `tests/test_round_orchestrated_runner.py`
- Modify: `tests/test_round_orchestrated_cli.py`

- [ ] **Step 1: Write failing integration tests**

Verify sequence metadata reaches both phases; SNN audit fields are written to
`round_plan.json` and `phase2_report.json`; terminal output still prints
latency, energy, TOPS, TOPS/W and rounds plus SNN `T` and update count; ANN
output remains compatible; invalid SNN exits nonzero with a stable code.

- [ ] **Step 2: Run the failing test**

```bash
$PYTHON -m pytest tests/test_round_snn_integration.py -v
```

- [ ] **Step 3: Thread SNN through the existing runner**

Do not add a second CLI. Add only backward-compatible optional request metadata
when frontend derivation is insufficient. Continue using the supported wrapper
and runpy-safe launcher paths.

- [ ] **Step 4: Verify and commit**

```bash
$PYTHON -m pytest tests/test_round_snn_integration.py tests/test_round_orchestrated_runner.py tests/test_round_orchestrated_cli.py tests/test_round_orchestrated_wrapper.py -v
$GIT add MNSIM/Interface/round_orchestrated_runner.py MNSIM/Interface/round_orchestrated_cli.py tests/test_round_snn_integration.py tests/test_round_orchestrated_runner.py tests/test_round_orchestrated_cli.py
$GIT commit -m "feat: expose snn round simulation in runner"
```

---

### Task 9: Regression And Full-T Acceptance

- [ ] **Step 1: Run the focused SNN suite**

```bash
$PYTHON -m pytest tests/test_round_snn_accumulator_domain_characterization.py tests/test_round_snn_digital_domain.py tests/test_round_snn_phase1.py tests/test_round_snn_boundary.py tests/test_round_snn_digital_lif.py tests/test_round_snn_lif_cost.py tests/test_round_snn_executor.py tests/test_round_snn_report.py tests/test_round_snn_integration.py -v
```

- [ ] **Step 2: Run the complete round scheduler suite**

```bash
$PYTHON -m pytest tests/test_round_phase1_types.py tests/test_round_phase1_config.py tests/test_round_phase1_resource_prepass.py tests/test_round_phase1_planner.py tests/test_round_phase1_mnsim_frontend.py tests/test_round_phase2_types.py tests/test_round_phase2_boundary.py tests/test_round_phase2_round_local_state.py tests/test_round_phase2_lowerer.py tests/test_round_phase2_executor.py tests/test_round_phase2_report.py tests/test_ann_round_physical_write_ledger.py tests/test_round_orchestrated_runner.py tests/test_round_orchestrated_cli.py -q
```

- [ ] **Step 3: Run formal G8 ANN regression**

Use the existing supported wrapper. Confirm `T=1`, two rounds, identical
xbar/programming counts and performance within floating tolerance, and no LIF
state/cost ledger.

- [ ] **Step 4: Run single-sample SNN full-T acceptance**

Start with a deterministic small feed-forward LIF network, then the formal SNN
network/checkpoint when available. Confirm one programming event per weighted
round, `T` reads, `T` LIF invocations, timestep-correct split closure,
`N*(T-1)*B` state reads/writes, DDR bytes scaling with `T` but zero DDR
cost, and exact metric reconciliation.

- [ ] **Step 5: Run full repository checks**

```bash
$PYTHON -m pytest -q
$GIT diff --check
$GIT status --short
```

- [ ] **Step 6: Commit acceptance artifacts**

```bash
$GIT add tests docs/superpowers/specs/2026-08-12-snn-digital-lif-round-execution-design.md docs/superpowers/plans/2026-08-12-snn-digital-lif-round-execution-implementation.md
$GIT commit -m "test: validate full-t digital lif execution"
```

---

## Completion Checklist

- [ ] ANN/CNN `T=1` remains behaviorally and numerically compatible.
- [ ] SNN uses the existing static `RoundPlan`; no SNN-only planner exists.
- [ ] Each weighted round programs once and reads `T` times.
- [ ] Each `(sample, lif_node, timestep)` executes exactly once.
- [ ] ADC-derived completed accumulator codes enter LIF bit-exactly.
- [ ] `B_acc/S_acc` come from the ADC/bit-slice accumulator path, not QAT or
      post-activation quantization metadata.
- [ ] Constant compilation is separate from the signal path and auditable.
- [ ] Signed nearest-even matches the integer reference model.
- [ ] State writeback uses checked narrowing with overflow-as-error.
- [ ] Tile ownership uses actual `N_k`, without free rebalancing.
- [ ] Buffer transactions use per-Tile/per-timestep ceiling semantics.
- [ ] LIF compute and on-chip state costs are included and reconciled.
- [ ] DDR payload is materialized while transfer cost remains excluded.
- [ ] Reports preserve ANN fields and expose SNN audit data.
- [ ] Focused, scheduler-wide, G8 ANN, and SNN full-T validations pass.
