# Unified Weight Scheduling Design

## 1. Scope

The round-orchestrated simulator needs one weight scheduling framework. A
policy such as sequential scheduling or FlashSeF-PTR is selected inside that
framework; it is not a separate execution architecture. Device models are
also selected through the same generic interface. The scheduler must not
contain model-specific branches for RRAMMilo or any future device.

The design preserves the current MNSIM boundary:

```text
ModelIR -> base RoundPlan -> unified weight scheduler -> ScheduleArtifact
         -> artifact-only Phase 2 executor -> phase2_report.json
```

The existing layer graph, resource prepass, round construction, TCG lowering,
boundary handling, LIF execution, static accounting, and report aggregation
remain the simulator core. The new framework supplies a frozen physical weight
schedule to that core.

## 2. Two-stage contract

The workflow has two explicit modes.

### Stage 1: generate

Stage 1 builds the normal base `RoundPlan`, extracts logical weighted-round
payloads, selects one scheduling policy, and runs the unified scheduler. The
result is a self-contained `ScheduleArtifact` containing every physical
assignment and resident-state transition needed by execution.

```python
artifact = generate_weight_schedule(
    base_round_plan=plan,
    weight_codes_by_layer=weight_codes,
    sim_config_path=sim_config_path,
    policy_id="seq",
    device_model_id="rram_milo_2021",
)
```

Stage 1 is the only stage allowed to invoke a scheduling policy.

### Stage 2: execute

Stage 2 loads the artifact, validates its identity and invariants, restores the
frozen placement snapshots, and invokes the existing Phase 2 execution stack.
It must not build a new policy, call a planner, recalculate an assignment, or
silently fall back to another policy.

```python
report = execute_weight_schedule(
    schedule_artifact_path=artifact_path,
    sim_config_path=sim_config_path,
    model_inputs=model_inputs,
    static_backend=static_backend,
)
```

An artifact validation failure is terminal for that run.

## 3. Unified scheduler model

The scheduler owns the common lifecycle; policies only make assignment
decisions.

```text
WeightScheduleProblem
  - base RoundPlan
  - logical weighted rounds and tile payloads
  - physical resources and constraints
  - device model and transition-cost engine
  - initial resident state

UnifiedWeightScheduler
  - iterate rounds in canonical order
  - call selected policy
  - validate the returned decision
  - evaluate state transitions
  - commit resident state
  - append an artifact round record
```

Suggested interfaces:

```python
class SchedulingPolicy(Protocol):
    policy_id: str

    def choose_assignment(
        self,
        context: RoundSchedulingContext,
        resident_state: ResidentState,
        transition_costs: TransitionCostView,
        constraints: SchedulingConstraints,
    ) -> AssignmentDecision:
        ...


class UnifiedWeightScheduler:
    def generate(
        self,
        problem: WeightScheduleProblem,
        policy: SchedulingPolicy,
    ) -> ScheduleArtifact:
        ...
```

Every policy returns the same `AssignmentDecision` shape:

```python
@dataclass(frozen=True)
class AssignmentDecision:
    round_id: str
    logical_to_physical_tile: dict[str, int]
    logical_tile_slots: dict[str, dict[str, int]]
    target_states: dict[int, tuple[int, ...]]
    policy_metadata: dict[str, Any]
```

The scheduler, rather than the policy, checks that each logical tile is
assigned exactly once, physical tiles are not duplicated, local slots remain
valid, code values are in range, and the target state matches the round
payload.

## 4. Device and transition-cost contracts

The scheduler consumes a generic discrete-state device model. It does not know
the model name or its provenance.

```python
class DeviceModel(Protocol):
    model_id: str
    device_bits: int
    device_level: int

    def validate_code(self, code: int) -> None:
        ...

    def initial_code(self) -> int:
        ...

    def cell_transition_cost(
        self,
        source_code: int,
        target_code: int,
    ) -> DeviceTransitionCost:
        ...
```

The generic cost engine calls that interface for every cell and performs all
tile, xbar, and round aggregation:

```python
class TransitionCostEngine:
    def matrix(self, source_state, target_state) -> TransitionCostView:
        ...

    def aggregate(self, source_state, target_state) -> AggregatedTransitionCost:
        ...
```

The result format is shared by all policies and all devices:

```python
@dataclass(frozen=True)
class DeviceTransitionCost:
    source_code: int
    target_code: int
    changed: bool
    objective_cost: float
    objective_unit: str
    latency_s: float | None
    energy_j: float | None
    exact: bool
    modeled_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
```

```python
@dataclass(frozen=True)
class AggregatedTransitionCost:
    changed_cells: int
    objective_cost: float
    objective_unit: str
    latency_s: float | None
    energy_j: float | None
    exact: bool
    provenance: dict[str, Any]
```

The objective must be explicit. For example, `changed_cell_count` is a valid
state-transition objective when a model does not provide measured programming
parameters. It must not be reported as write energy or write latency.

The current 7-bit CrossSim RRAMMilo profile is registered as an ordinary
`DeviceModel` with `device_level=128` and `device_bits=7`. Its nominal states
and programming-error metadata are available to the model. Since the profile
does not provide reliable programming voltage or pulse timing, its default
capability is:

```text
objective: changed_cell_count
latency: unavailable
energy: unavailable
exact: false
```

No RRAMMilo-specific branch belongs in the scheduler or cost engine. A future
device with measured pulse parameters implements the same interface and can
provide exact latency and energy without changing either policy.

## 5. Policy semantics

`seq` and `FlashSeF-PTR` are policy implementations only.

`seq` returns the canonical sequential assignment for the current round. It
still uses the shared resident-state manager and shared transition-cost engine.

`FlashSeF-PTR` returns an assignment selected from the shared cost view. It
preserves physical tile residency, keeps local slots stable, and may use a
Hungarian solver to minimize the configured objective. It does not own state
commit, artifact serialization, or validation.

Adding a policy therefore requires only a new `SchedulingPolicy` and registry
entry. It does not require another executor or another artifact format.

## 6. Schedule artifact

All policies produce the same artifact schema. Policy-specific facts live under
`policy_metadata`; the execution contract is policy-independent.

```json
{
  "schema_version": 1,
  "scheduler_id": "unified_weight_scheduler",
  "policy_id": "flashsef_ptr",
  "path_id": "sched-flashsef",
  "base_round_plan_sha256": "...",
  "model_ir_sha256": "...",
  "weights_sha256": "...",
  "sim_config_sha256": "...",
  "device_model": {
    "model_id": "rram_milo_2021",
    "device_bits": 7,
    "device_level": 128
  },
  "transition_cost": {
    "objective": "changed_cell_count",
    "objective_unit": "cell_transition",
    "exact": false,
    "latency_modeled": false,
    "energy_modeled": false
  },
  "initial_resident_state": {},
  "rounds": [],
  "final_resident_state": {},
  "execution_mode": "artifact_only",
  "policy_metadata": {}
}
```

Each round record contains the common fields `round_id`, logical tile payloads,
assignment, placement snapshot, resident state before and after, and aggregated
transition cost. The executor uses only these records and the base-plan
identity; it does not use policy metadata to make new decisions.

The placement snapshot contains the existing `RoundPlacementPlan` contract:
fragment specifications, physical group slots, output ownership, mapping
layer indices, capacity evidence, and the TCG layer-tile view needed by the
original MNSIM cost models. Stage 2 restores this data as a `FrozenTCG` view.
It may initialize hardware cost objects from the SimConfig, but it does not
invoke the original TCG constructor or run a mapping pass. A missing snapshot
for any weighted round is an execution error rather than a trigger to re-plan.

## 7. Configuration and integration boundary

The configuration selects one scheduler and one policy:

```ini
[Weight Scheduling]
Scheduler = unified_weight_scheduler
Policy = seq
Schedule_Mode = generate_and_execute
Schedule_Plan_In =
Schedule_Plan_Out =
Allow_Replan_During_Execution = 0
Transition_Cost_Objective = changed_cell_count
```

Canonical path IDs remain `sched-seq` and `sched-flashsef`. They identify the
selected policy, not separate execution frameworks.

The root runner now exposes two explicit scheduled modes:

- `--schedule-only --schedule-plan-out PATH` builds the normal Phase 1 plan,
  generates one policy artifact, and stops before Phase 2 inference;
- `--infer-with-schedule --schedule-plan-in PATH` restores the serialized
  `RoundPlan` and placement snapshots, validates model/checkpoint/SimConfig
  hashes, and invokes Phase 2 with `schedule_artifact`.

Stage 1 uses the normal Phase 1 planner. Stage 2 does not call `plan_builder`,
a scheduling policy, or a TCG mapping pass. Artifact schema v2 stores the
complete `RoundPlan` snapshot and a placement snapshot for every weighted
round, so a missing snapshot is a hard execution error rather than a trigger
to re-plan.

## 8. Validation and test plan

Before execution, the artifact validator must check:

- scheduler and policy identifiers;
- base plan, model, weights, and SimConfig hashes;
- device model identity, level count, and bit width;
- contiguous round IDs and complete round coverage;
- legal assignment and placement snapshot;
- resident-state chaining between adjacent rounds;
- cost objective and capability declaration;
- `execution_mode=artifact_only` and replan disabled.

Required tests include deterministic generation, policy output conformance,
duplicate-tile rejection, resident-state chaining, modified-artifact
rejection, planner-not-called during execution, 7-bit/128-level code bounds,
and regression coverage for the existing MNSIM Phase 1 and Phase 2 tests.

## 9. Implementation order

1. Add the common data contracts and deterministic serialization.
2. Add the generic device registry and transition-cost engine.
3. Add the unified scheduler and artifact validator.
4. Implement `seq` as the first policy.
5. Connect artifact-only execution to the existing runner.
6. Implement `FlashSeF-PTR` against the same policy interface.
7. Register the current RRAMMilo model through the generic device contract.
8. Add command-line and SimConfig integration after contract tests pass.

At no point should the implementation fork MNSIM's original mapping or
execution logic for a particular policy or device model.
