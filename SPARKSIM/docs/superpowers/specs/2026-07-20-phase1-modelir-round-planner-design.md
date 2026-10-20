# Phase 1 ModelIR Round Planner Design

Date: 2026-07-20

## 1. Scope

This spec defines Phase 1 of the new round-orchestrated simulator.

Phase 1 starts from an independent `ModelIR` and a MNSIM-2.0 `SimConfig.ini`
view, computes per-weighted-layer resource footprints, and emits an ordered
`RoundPlan.execution_items[]`.

```text
ModelIR + SimConfig.ini
  -> ResourcePrepass
  -> LayerResourceDescriptor[]
  -> RoundPlanner
  -> RoundPlan.execution_items[]
```

Phase 1 does not execute the model and does not call MNSIM TCG or Model_*.

## 2. Non-goals

Phase 1 v1 does not implement:

```text
Phase 2 lowering
MNSIM TCG invocation
Model_latency / Model_area / Model_power / Model_energy
BoundaryStateManager runtime commit
DDR / boundary-store latency or energy
scheduled physical write / remap / persistent placement
PyTorch / ONNX frontend import
direct integration with existing scheduled FragmentSpec / BundleSpec / LayerExecutionPlan
SNN backend
```

Existing MNSIM-2.0 code remains the static-evaluation authority for later
phases. Phase 1 must not duplicate TCG placement logic.

## 3. First Implementation Entry

The first implementation is an independent `ModelIR` core. It does not start
from the existing scheduled path.

Public entry:

```text
build_round_plan(model_ir, sim_config) -> RoundPlan
```

Inputs:

```text
model_ir:
  Static graph produced by a future FrontendNormalizer.
  Tests construct synthetic ModelIR directly.

sim_config:
  SimConfig.ini path or equivalent parsed config object.

weights:
  Phase 1 does not read raw weight values.
  ModelIR only carries weight_ref_id for future Phase 2 lookup.
```

`FrontendNormalizer` is outside Phase 1 v1. Phase 1 only validates and consumes
an already constructed `ModelIR`.

## 4. Code Layout

Phase 1 files should be added under:

```text
MNSIM/Interface/round_phase1_types.py
MNSIM/Interface/round_phase1_config.py
MNSIM/Interface/round_phase1_resource_prepass.py
MNSIM/Interface/round_phase1_planner.py
```

Responsibilities:

```text
round_phase1_types.py:
  dataclass public schema and to_dict/from_dict helpers

round_phase1_config.py:
  SimConfig.ini parsing and HardwareProjection

round_phase1_resource_prepass.py:
  source-aligned formula ResourcePrepass v1

round_phase1_planner.py:
  stable topological ready-set RoundPlanner
```

## 5. Public Schema Rules

Public Phase 1 objects use standard-library `dataclasses`.

Rules:

```text
1. Public fields must be JSON-friendly:
   str / int / float / bool / list / dict / None.
2. Enum-like values are string literals, not serialized Enum objects.
3. RoundPlan must support deterministic to_dict/from_dict.
4. RoundPlan must not contain raw weight tensors, numpy arrays, torch tensors,
   MNSIM NetStruct objects, or TCG mapping objects.
5. Planner internals may be mutable; emitted public dataclasses are the
   contract consumed by later phases.
```

String literal sets:

```text
item_kind:
  weighted_physical_round
  zero_xbar_round

round_kind:
  full_layer_pack
  single_layer_split
  single_layer_continuation

zero_xbar_kind:
  hook_only
  structural_mnsim
  mixed_non_weighted

node_role:
  weighted
  hook
  structural_non_weighted
```

Supported layer types in Phase 1 v1:

```text
weighted:
  conv
  fc
  fc_like_projection

non-weighted:
  relu
  bn
  view
  reshape
  flatten
  dropout
  pooling
  concat
  element_sum
  element_multiply
  backend_hook
```

Unknown layer types make the plan invalid.

## 6. ModelIR

Minimum schema:

```text
ModelIR
  model_id
  backend_kind
  sequence_spec
  layers[]

SequenceSpec
  sequence_length
  ordered_steps
  cross_step_dependency
  recurrent_state_ids[]

LayerIR
  layer_id
  layer_type
  is_weighted
  stable_topological_index
  input_tensor_ids[]
  output_tensor_ids[]
  input_layer_ids[]
  output_consumer_ids[]
  weight_ref_id
  shape
  backend_attrs
```

`stable_topological_index` is required. The planner does not infer framework
order by itself.

Required shape fields:

```text
conv:
  Inputchannel
  Outputchannel
  Kernelsize

fc:
  Infeature
  Outfeature

fc_like_projection:
  Infeature
  Outfeature
```

Non-weighted nodes need enough tensor metadata for dependency validation, but
Phase 1 does not compute xbar footprint for them.

## 7. ResourcePrepass

ResourcePrepass v1 is a source-aligned formula prepass. It only computes
weighted footprint and opaque lookup metadata.

Output per weighted layer:

```text
LayerResourceDescriptor
  layer_id
  layer_type
  input_layer_ids
  output_consumer_ids
  required_xbar_count
  prepass_ref
  completion_contract
```

`required_xbar_count` is `M_i`.

The public `prepass_ref` is an opaque handle. RoundPlanner stores it but never
opens it.

## 8. Hardware Projection

ResourcePrepass derives:

```text
xbar_size = Crossbar level Xbar_Size
xbar_nums = Tile_Num[0] * Tile_Num[1] * PE_Num[0] * PE_Num[1] * Group_Num
```

These fields do not enter `xbar_nums`:

```text
Device_Level
Weight_precision
Xbar_Polarity
```

They affect `M_i` through bit split and polarity split.

## 9. Resource Formula

FC and RNN-like projection:

```text
row_split_count = ceil(Infeature / xbar_row)
output_subset_count = ceil(Outfeature / xbar_col)
```

Conv:

```text
input_channels_per_xbar = floor(xbar_row / (Kernelsize * Kernelsize))
row_split_count = ceil(Inputchannel / input_channels_per_xbar)
output_subset_count = ceil(Outputchannel / xbar_col)
```

Precision and polarity:

```text
if Xbar_Polarity == 2:
  weight_bit_split_part = ceil((Weight_precision - 1) / log2(Device_Level))
else:
  weight_bit_split_part = ceil(Weight_precision / log2(Device_Level))

polarity_count = Xbar_Polarity
```

Final footprint:

```text
M_i = row_split_count
    * output_subset_count
    * weight_bit_split_part
    * polarity_count
```

Conv uses the GEMM footprint for counting xbar blocks, but later lowering still
uses MNSIM native `conv`. Phase 1 does not force Conv into FC.

## 10. RoundPlan

Main output:

```text
RoundPlan
  plan_id
  model_id
  backend_kind
  hardware_budget
    xbar_size
    xbar_nums
  execution_items[]
  layer_resource_descriptors[]
  model_graph_ref
  feasible
  invalid_reason
```

Canonical execution object:

```text
ExecutionItem = WeightedPhysicalRound | ZeroXbarRound
```

`RoundPlan.rounds[]` is not a primary execution contract. If kept for
compatibility, it must be derived as:

```text
rounds[] = filter(execution_items, item_kind == weighted_physical_round)
```

## 11. WeightedPhysicalRound

Schema:

```text
WeightedPhysicalRound
  item_kind = weighted_physical_round
  item_id
  item_index
  round_kind
  selected_entries[]
  local_graph_nodes[]
  external_inputs[]
  same_item_edges[]
  external_outputs[]
  partial_output_contract[]
  resource_proof
  backend_runtime
```

`round_kind`:

```text
full_layer_pack
single_layer_split
single_layer_continuation
```

Each selected entry:

```text
layer_id
layer_type
xbar_range_start
xbar_range_end
used_xbars
layer_complete_in_round
prepass_ref
```

Constraints:

```text
1. selected_entries[] only contains weighted layers/projections.
2. local_graph_nodes[] only contains weighted nodes.
3. full_layer_pack contains only complete weighted layers.
4. single-layer split/continuation contains only one original weighted layer.
5. full + partial is invalid.
6. partial + any other weighted layer is invalid.
7. multiple partial layers in one weighted round are invalid.
```

## 12. ZeroXbarRound

Schema:

```text
ZeroXbarRound
  item_kind = zero_xbar_round
  item_id
  item_index
  zero_xbar_kind
  selected_entries[] = []
  local_graph_nodes[]
  external_inputs[]
  same_item_edges[]
  external_outputs[]
  partial_output_contract[] = []
  resource_proof.used_xbars = 0
```

All non-weighted layers/hooks go into `ZeroXbarRound`.

`zero_xbar_kind`:

```text
hook_only:
  all nodes are hook-like

structural_mnsim:
  all nodes are structural non-weighted

mixed_non_weighted:
  hook-like and structural nodes both appear
```

Phase 1 only classifies. It does not promise that Phase 2 already supports
structural-only MNSIM evaluation.

## 13. Planning Algorithm

The planner uses explicit DAG readiness and stable topological greedy packing.

Loop:

```text
while unfinished nodes exist:
  ready_non_weighted = non-weighted nodes whose inputs are complete
  ready_weighted = weighted nodes whose inputs are complete

  if ready_non_weighted is not empty:
    emit maximal ZeroXbarRound closure
    continue

  if earliest stable-topological ready weighted node has M_i > xbar_nums:
    emit one single-layer split/continuation weighted item for that node
    continue

  start full_layer_pack WeightedPhysicalRound
  while exists ready weighted node with used + M_i <= xbar_nums:
    choose fit node with smallest stable_topological_index
    add it complete
  emit WeightedPhysicalRound
```

If the earliest ready weighted node does not fit, but a later ready weighted
node fits, v1 may skip the earliest node and choose the fit node with the
smallest `stable_topological_index`. This is deterministic and avoids wasted
capacity in DAG branch cases.

The planner does not search for optimal bin packing.

## 14. Split and Completion

If:

```text
M_i > xbar_nums
```

the planner emits continuation weighted items for that same layer:

```text
W0: xbar_range = [0, xbar_nums), incomplete
W1: xbar_range = [xbar_nums, 2*xbar_nums), maybe complete
...
Wk: xbar_range = [k*xbar_nums, M_i), complete
```

Partial state is never visible to downstream nodes:

```text
partial_sum -> internal only
output_subset -> internal only
complete output -> downstream visible
```

If the tail continuation completes the layer, it still ends the
`WeightedPhysicalRound`; downstream non-weighted work starts from a later
`ZeroXbarRound`.

## 15. ZeroXbar Closure

ZeroXbarRound v1 uses maximal ready non-weighted closure:

```text
start ZeroXbarRound
while exists ready non-weighted node:
  choose smallest stable_topological_index
  add node
  mark its outputs complete inside this item
  update ready non_weighted set
emit ZeroXbarRound
```

Rules:

```text
1. ReLU -> BN -> view can become one ZeroXbarRound.
2. Independent ready non-weighted nodes can be grouped deterministically.
3. The closure cannot cross a weighted node.
4. ZeroXbarRound cannot consume partial_sum or output_subset.
5. Unknown non-weighted type is invalid.
```

## 16. RNN-like Handling

RNN-like is represented as:

```text
FC-like weighted projection graph
+ recurrent SequenceSpec
+ backend state hooks
```

A vanilla RNN projection uses:

```text
Infeature = input_size + hidden_size
Outfeature = hidden_size
```

LSTM fused projection usually uses:

```text
Infeature = input_size + hidden_size
Outfeature = 4 * hidden_size
```

State update hooks are non-weighted and go into `ZeroXbarRound`:

```text
W0: RNN_projection complete
Z0: RNN_state_update hook
```

For split projection:

```text
W0: RNN_projection partial
W1: RNN_projection continuation complete
Z0: RNN_state_update hook
```

Phase 1 does not unroll sequence steps into duplicate weight layers.

## 17. Error Handling

Phase 1 returns an invalid plan for user input, model description, or hardware
projection errors:

```text
RoundPlan.feasible = false
RoundPlan.invalid_reason = "<ERROR_CODE>: <detail>"
RoundPlan.execution_items = []
```

v1 records one deterministic `invalid_reason`: the first error encountered in
validation order.

Error codes:

```text
INVALID_SIM_CONFIG
INVALID_HARDWARE_PROJECTION
UNSUPPORTED_LAYER_TYPE
INVALID_LAYER_SHAPE
INVALID_MODEL_DAG
INVALID_SEQUENCE_SPEC
INVALID_RESOURCE_FOOTPRINT
INFEASIBLE_LAYER_ATOMIC_SCOPE
INTERNAL_PLANNER_INVARIANT
```

Invalid conditions include:

```text
xbar_nums <= 0
xbar_row <= 0 or xbar_col <= 0
Device_Level <= 1
Weight_precision <= 0
unsupported Xbar_Polarity
Conv: Kernelsize * Kernelsize > xbar_row
M_i <= 0
missing required SimConfig field
missing required layer shape field
unknown weighted layer type
unknown non-weighted layer type
cycle in ModelIR DAG
duplicate layer_id
duplicate stable_topological_index
dependency contradicts stable topological order
```

`INTERNAL_PLANNER_INVARIANT` indicates an implementation bug, such as emitting
a full+partial weighted item. Development tests may assert; public entry should
convert it into an invalid plan.

## 18. Test Matrix

Tests use synthetic `ModelIR` and config dictionaries. They do not call Phase 2,
TCG, or Model_*.

Required tests:

```text
1. hardware projection
   2x2 Tile, 2x2 PE, Group_Num=1 -> xbar_nums=16
   Device_Level / Weight_precision / Xbar_Polarity do not change xbar_nums

2. ResourcePrepass formulas
   FC M_i
   Conv M_i with kernel-preserving input-channel split
   RNN fc_like_projection M_i
   precision / polarity split changes M_i

3. linear CNN
   Conv -> ReLU -> Conv -> ReLU -> FC
   expected W/Z/W/Z/W

4. full-layer pack
   two ready weighted layers fit in one WeightedPhysicalRound

5. no full+partial
   complete layer fits, next layer does not fit completely;
   planner ends current item and splits next layer later

6. single-layer split / continuation
   M_i > xbar_nums
   expected W partial, W continuation, complete only at tail

7. ZeroXbarRound maximal closure
   ReLU -> BN -> view becomes one ZeroXbarRound
   independent ready non-weighted nodes group deterministically

8. DAG branch / merge
   Conv0 -> ConvA
         -> ConvB
   ConvA + ConvB -> element_sum -> ConvC
   expected W Conv0, W ConvA+ConvB, Z element_sum, W ConvC

9. RNN-like full projection
   RNN_projection complete -> ZeroXbarRound state_update

10. RNN-like split projection
    W projection partial, W projection continuation, Z state_update

11. invalid config and invalid model
    missing SimConfig field
    xbar_nums <= 0
    Device_Level <= 1
    Weight_precision <= 0
    Conv Kernelsize*Kernelsize > xbar_row
    unsupported weighted layer type
    unsupported non-weighted layer type
    cyclic ModelIR DAG

12. deterministic serialization
    RoundPlan.to_dict() output stable
    RoundPlan.from_dict(to_dict(plan)) preserves public fields
```

## 19. Acceptance Criteria

Phase 1 is complete when:

```text
1. build_round_plan(model_ir, sim_config) emits deterministic RoundPlan objects.
2. ResourcePrepass computes expected M_i for conv / fc / fc_like_projection.
3. RoundPlanner emits only legal WeightedPhysicalRound / ZeroXbarRound items.
4. No non-weighted node appears inside WeightedPhysicalRound.local_graph_nodes.
5. No full+partial weighted item can be emitted.
6. Invalid inputs return feasible=false with stable invalid_reason.
7. Synthetic ModelIR tests cover CNN, DAG, RNN-like, split, zero-xbar closure,
   resource formula, and invalid cases.
8. Phase 1 tests do not invoke TCG or Model_*.
```

## 20. Implementation Plan

Implementation is tracked in:

`docs/superpowers/plans/2026-07-21-phase1-modelir-round-planner-implementation.md`
