# Phase 2 Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 2 of the round-orchestrated simulator: lower fixed Phase 1 `ExecutionItem`s into item-local MNSIM static work or hook work, commit cross-item state through `BoundaryStateManager`, and emit normalized reports with TOPS/W.

**Architecture:** Add Phase 2 core modules under `MNSIM/Scheduling/phase2`. Keep MNSIM static evaluation behind `run_mnsim_static_eval(...)`, keep all cross-item partial and visibility semantics in `BoundaryStateManager`, and keep final metric aggregation in `ReportAggregator`. Tests use small synthetic `RoundPlan` and fake static-eval backends first, then a minimal original-MNSIM smoke test.

**Tech Stack:** Python standard library (`dataclasses`, `math`, `collections`, `typing`), existing MNSIM-2.0 static classes (`TCG`, `Model_latency`, `Model_area`, `Model_inference_power`, `Model_energy`), tests with `pytest`; run tests via `/home/xzh/micromamba/envs/workspace/bin/python -m pytest`.

---

## Source Specs

Implement the confirmed Phase 2 contracts in:

- `knowledge_base/phase2_execution_spec.md`
- `knowledge_base/phase2_netstruct_lowering.md`
- `knowledge_base/mnsim_static_interface.md`
- `knowledge_base/boundary_state_manager.md`
- `knowledge_base/phase2_report_aggregator.md`

Do not implement sequence executor, SNN runtime, physical write/remap ledger, DDR/boundary-store latency or energy modeling, output-subset pipeline, or static `concat` execution.

## Working Directory

Run all commands from:

```bash
cd /home/xzh/workspace/cim_project/MNSIM-2.0/.worktrees/task-20260707-round-orchestrated-mnsim/round_orchestrated_simulator/MNSIM-2.0
```

Use the project git binary:

```bash
/home/xzh/micromamba/envs/workspace/bin/git
```

Do not run repository Python files as bare scripts. Use:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest ...
```

## File Structure

Create:

- `MNSIM/Scheduling/phase2/types.py`
  Public Phase 2 dataclasses, string constants, `Phase2Invalid`, serialization helpers.

- `MNSIM/Scheduling/phase2/materializer.py`
  Convert `SelectedEntry.prepass_ref + xbar_range` into concrete Conv/FC spatial scopes, with precision-group alignment checks.

- `MNSIM/Scheduling/phase2/netstruct_builder.py`
  Build item-local MNSIM-compatible `NetStruct` from local layer infos. v1 uses MNSIM-compatible dictionaries and `tile_array=None`; original MNSIM `TCG` and Model_* consume `NetStruct[layer][0][0]`.

- `MNSIM/Scheduling/phase2/mnsim_static_eval.py`
  `run_mnsim_static_eval(...)` wrapper with validation, injectable test backend, original MNSIM backend, context filtering, and latency offset normalization.

- `MNSIM/Scheduling/phase2/boundary.py`
  `BoundaryStateManager`, state table, input view construction, commit state machine, audit events.

- `MNSIM/Scheduling/phase2/lowerer.py`
  `RoundLowerer` for full-layer-pack, split/continuation synthetic conv/fc, standalone structural context frame, and hook-only bindings.

- `MNSIM/Scheduling/phase2/report.py`
  `ReportAggregator` and `Phase2Report` construction.

- `MNSIM/Scheduling/phase2/executor.py`
  End-to-end Phase 2 item loop.

Modify:

- `MNSIM/Scheduling/phase2/__init__.py`
  Export the Phase 2 public entry points.

- `MNSIM/Scheduling/phase1/resource_prepass.py`
  Add materialization metadata into new `PrepassRef.shape_signature` values without changing RoundPlanner's public responsibility.

Tests:

- `tests/test_round_phase2_types.py`
- `tests/test_round_phase2_materializer.py`
- `tests/test_round_phase2_static_eval.py`
- `tests/test_round_phase2_boundary.py`
- `tests/test_round_phase2_lowerer.py`
- `tests/test_round_phase2_report.py`
- `tests/test_round_phase2_executor.py`

## Public Names To Preserve

```python
BoundaryStateManager
RoundLowerer
ReportAggregator
run_mnsim_static_eval
execute_round_plan
Phase2Invalid
StaticEvalRequest
StaticEvalFrame
StaticEvalResult
BoundaryInputView
BoundaryCommitRequest
BoundaryCommitResult
HookResult
Phase2Report
```

Use string literal fields, not Python `Enum` objects.

## Data Units

Normalize metric units at the Phase 2 wrapper boundary:

```text
MNSIM latency ns -> StaticEvalResult seconds
MNSIM energy nJ  -> StaticEvalResult joules
MNSIM power W    -> StaticEvalResult watts
MNSIM area um2   -> StaticEvalResult um2
```

`ReportAggregator` consumes only normalized SI latency/energy/power values and `area_um2`.

---

### Task 1: Phase 2 Public Types

**Files:**
- Create: `tests/test_round_phase2_types.py`
- Create: `MNSIM/Scheduling/phase2/types.py`
- Modify: `MNSIM/Scheduling/phase2/__init__.py`

- [ ] **Step 1: Write the failing type and serialization tests**

Create `tests/test_round_phase2_types.py`:

```python
from MNSIM.Scheduling.phase2.types import (
    BOUNDARY_STATUS_COMPLETE_OUTPUT,
    BOUNDARY_STATUS_HOOK_COMPLETE,
    CONTRIBUTION_COMPLETE_OUTPUT,
    CONTRIBUTION_HOOK_OUTPUT,
    BoundaryBinding,
    BoundaryCommitRequest,
    BoundaryCommitResult,
    BoundaryInputView,
    BoundaryStateRecord,
    HookResult,
    MetricTotals,
    StaticEvalFrame,
    StaticEvalRequest,
    StaticEvalResult,
    StaticLayerResult,
)


def test_static_eval_result_serializes_reportable_and_context_layers():
    binding = BoundaryBinding(
        local_layer_id=1,
        hook_node_id=None,
        origin_layer_id="Pool1",
        origin_layer_type="pooling",
        report_role="reportable",
        tensor_role="activation",
        shape={"Outputchannel": 8, "Outputsize": [16, 16]},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="structural_output",
        source_state_ids=["state:conv1"],
        downstream_consumer_ids=["Conv2"],
    )
    result = StaticEvalResult(
        item_id="Z0",
        item_kind="zero_xbar_structural",
        status="ok",
        layer_results=[
            StaticLayerResult(
                local_layer_id=0,
                origin_layer_id="ctx:0",
                layer_type="conv",
                report_role="context_only",
                latency_s=0.0,
                begin_time_s=0.0,
                finish_time_s=0.0,
                area_um2=10.0,
                power_w=1.0,
                energy_j=1e-9,
                tcg_mapping={"tilenum": 1},
            ),
            StaticLayerResult(
                local_layer_id=1,
                origin_layer_id="Pool1",
                layer_type="pooling",
                report_role="reportable",
                latency_s=2e-9,
                begin_time_s=0.0,
                finish_time_s=2e-9,
                area_um2=3.0,
                power_w=0.5,
                energy_j=1e-9,
                tcg_mapping={"tilenum": 1},
            ),
        ],
        item_totals=MetricTotals(latency_s=2e-9, energy_j=1e-9, power_w=0.5, area_um2=3.0),
        mapping_summary={"layer_count": 2},
        raw_mnsim_summary={"raw_latency_total_s": 7e-9},
        boundary_bindings=[binding],
        warnings=["context layer filtered"],
    )

    data = result.to_dict()

    assert data["layer_results"][0]["report_role"] == "context_only"
    assert data["item_totals"]["latency_s"] == 2e-9
    assert data["boundary_bindings"][0]["origin_layer_id"] == "Pool1"


def test_boundary_commit_request_supports_hook_only_binding():
    hook_binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"kind": "same_as_input"},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind=CONTRIBUTION_HOOK_OUTPUT,
        source_state_ids=["state:conv1"],
        downstream_consumer_ids=["Conv2"],
    )
    request = BoundaryCommitRequest(
        item_id="R0",
        item_kind="weighted_physical_round",
        execution_index=0,
        input_view_id="view:R0",
        source_kind="static_eval",
        static_eval_result=None,
        hook_result=HookResult(
            item_id="R0",
            item_kind="weighted_physical_round",
            hook_bindings=[hook_binding],
            warnings=[],
        ),
        boundary_bindings=[hook_binding],
    )

    assert request.to_dict()["boundary_bindings"][0]["local_layer_id"] is None
    assert request.boundary_bindings[0].hook_node_id == "Relu1"


def test_boundary_records_and_commit_result_schema():
    state = BoundaryStateRecord(
        state_id="state:relu1",
        producer_item_id="R0",
        producer_layer_id="Relu1",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        status=BOUNDARY_STATUS_HOOK_COMPLETE,
        storage_backend="boundary_store",
        allowed_consumer_item_ids=["R1"],
        lifetime_policy="release_after_consumers",
        source_state_ids=["state:conv1"],
    )
    view = BoundaryInputView(
        input_view_id="view:R1",
        item_id="R1",
        readable_state_ids=["state:relu1"],
        state_bindings={"L002_conv": ["state:relu1"]},
        warnings=[],
    )
    commit = BoundaryCommitResult(
        item_id="R1",
        item_kind="weighted_physical_round",
        status="ok",
        created_states=[state],
        updated_states=[],
        released_states=[],
        events=[],
        completion_updates=[
            {
                "origin_layer_id": "Conv2",
                "completion_status": BOUNDARY_STATUS_COMPLETE_OUTPUT,
                "state_id": "state:conv2",
                "input_range_covered": {"kind": "full"},
                "output_range_covered": {"kind": "full"},
                "downstream_visible": True,
            }
        ],
        visibility_updates=[],
        output_state_bindings=[],
        warnings=[],
    )

    assert view.input_view_id == "view:R1"
    assert commit.created_states[0].status == BOUNDARY_STATUS_HOOK_COMPLETE
    assert CONTRIBUTION_COMPLETE_OUTPUT == "complete_output"
```

- [ ] **Step 2: Run the type tests and confirm they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_types.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing Phase 2 dataclass names.

- [ ] **Step 3: Implement `types.py`**

Create `MNSIM/Scheduling/phase2/types.py` with string constants and dataclasses matching the test names. Reuse `SerializableDataclass` from `MNSIM.Scheduling.round_types`.

Required constants:

```python
BOUNDARY_STATUS_MODEL_INPUT = "model_input"
BOUNDARY_STATUS_PARTIAL_SUM = "partial_sum"
BOUNDARY_STATUS_OUTPUT_SUBSET = "output_subset"
BOUNDARY_STATUS_COMPLETE_OUTPUT = "complete_output"
BOUNDARY_STATUS_HOOK_COMPLETE = "hook_complete"
BOUNDARY_STATUS_RELEASED = "released"

CONTRIBUTION_PARTIAL = "partial_contribution"
CONTRIBUTION_COMPLETE_OUTPUT = "complete_output"
CONTRIBUTION_STRUCTURAL_OUTPUT = "structural_output"
CONTRIBUTION_HOOK_OUTPUT = "hook_output"

REPORT_ROLE_REPORTABLE = "reportable"
REPORT_ROLE_CONTEXT_ONLY = "context_only"
STATIC_STATUS_OK = "ok"
SOURCE_KIND_STATIC_EVAL = "static_eval"
SOURCE_KIND_HOOK = "hook"
```

Required dataclasses:

```python
MetricTotals
BoundaryBinding
StaticEvalFrame
StaticEvalRequest
StaticLayerResult
StaticEvalResult
HookResult
BoundaryStateRecord
BoundaryAccessEvent
BoundaryInputView
BoundaryCommitRequest
BoundaryCommitResult
LoweredItem
Phase2Report
```

`Phase2Invalid` must expose `.code`, `.detail`, and `.reason`.

- [ ] **Step 4: Export public names**

Modify `MNSIM/Scheduling/phase2/__init__.py` to export the dataclasses and constants that the tests import.

- [ ] **Step 5: Run the type tests and confirm they pass**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_types.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Scheduling/phase2/__init__.py MNSIM/Scheduling/phase2/types.py tests/test_round_phase2_types.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase2 public schema"
```

---

### Task 2: Prepass Materializer And Precision Alignment Guard

**Files:**
- Create: `tests/test_round_phase2_materializer.py`
- Create: `MNSIM/Scheduling/phase2/materializer.py`
- Modify: `MNSIM/Scheduling/phase1/resource_prepass.py`

- [ ] **Step 1: Write failing materializer tests**

Create `tests/test_round_phase2_materializer.py`:

```python
from MNSIM.Scheduling.phase2.materializer import materialize_selected_entry_scope
from MNSIM.Scheduling.phase2.types import Phase2Invalid
from MNSIM.Scheduling.round_types import PrepassRef, SelectedEntry


def _prepass_ref(layer_type, shape):
    return PrepassRef(
        ref_id=f"m0:L0:{layer_type}:mnsim20_formula_v1",
        model_id="m0",
        layer_id="L0",
        layer_type=layer_type,
        resource_version="mnsim20_formula_v1",
        sim_config_fingerprint="xbar=128x128;device=2;weight=8;polarity=2",
        layer_config_fingerprint="layer:test",
        weight_ref_id="w0",
        shape_signature=shape,
    )


def test_materializes_conv_range_to_channel_scope():
    entry = SelectedEntry(
        layer_id="L0",
        layer_type="conv",
        xbar_range_start=0,
        xbar_range_end=14,
        used_xbars=14,
        layer_complete_in_round=False,
        prepass_ref=_prepass_ref(
            "conv",
            {
                "Inputchannel": 32,
                "Outputchannel": 256,
                "Kernelsize": 3,
                "Inputsize": [32, 32],
                "Outputsize": [30, 30],
                "Inputbit": 8,
                "Weightbit": 8,
                "outputbit": 8,
            },
        ),
    )

    scopes = materialize_selected_entry_scope(entry, xbar_size=(128, 128))

    assert len(scopes) == 1
    assert scopes[0]["layer_type"] == "conv"
    assert scopes[0]["input_range"] == [0, 14]
    assert scopes[0]["output_range"] == [0, 128]
    assert scopes[0]["kernel_scope"] == "full"


def test_materializes_fc_range_to_feature_scope():
    entry = SelectedEntry(
        layer_id="L0",
        layer_type="fc",
        xbar_range_start=14,
        xbar_range_end=28,
        used_xbars=14,
        layer_complete_in_round=False,
        prepass_ref=_prepass_ref(
            "fc",
            {
                "Infeature": 256,
                "Outfeature": 256,
                "Inputbit": 8,
                "Weightbit": 8,
                "outputbit": 8,
            },
        ),
    )

    scopes = materialize_selected_entry_scope(entry, xbar_size=(128, 128))

    assert scopes == [
        {
            "layer_id": "L0",
            "layer_type": "fc",
            "input_range": [0, 128],
            "output_range": [128, 256],
            "full_input_range": [0, 256],
            "full_output_range": [0, 256],
        }
    ]


def test_rejects_precision_temporal_split_range():
    entry = SelectedEntry(
        layer_id="L0",
        layer_type="fc",
        xbar_range_start=1,
        xbar_range_end=14,
        used_xbars=13,
        layer_complete_in_round=False,
        prepass_ref=_prepass_ref(
            "fc",
            {
                "Infeature": 128,
                "Outfeature": 128,
                "Inputbit": 8,
                "Weightbit": 8,
                "outputbit": 8,
            },
        ),
    )

    try:
        materialize_selected_entry_scope(entry, xbar_size=(128, 128))
    except Phase2Invalid as exc:
        assert exc.code == "PRECISION_TEMPORAL_SPLIT_UNSUPPORTED"
    else:
        raise AssertionError("expected precision split rejection")
```

- [ ] **Step 2: Run materializer tests and confirm they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_materializer.py -v
```

Expected: FAIL with missing `MNSIM.Scheduling.phase2.materializer`.

- [ ] **Step 3: Implement materializer**

Create `MNSIM/Scheduling/phase2/materializer.py`.

Required behavior:

```text
precision_group_size =
  weight_bit_split_part * polarity_count

selected xbar range must satisfy:
  xbar_range_start % precision_group_size == 0
  xbar_range_end % precision_group_size == 0

spatial index range =
  [xbar_range_start / precision_group_size,
   xbar_range_end / precision_group_size)
```

For Conv:

```text
input_channels_per_xbar = floor(xbar_row / (Kernelsize * Kernelsize))
row_block_count = ceil(Inputchannel / input_channels_per_xbar)
output_block_count = ceil(Outputchannel / xbar_col)
spatial block order = row_block major, then output block
```

For FC:

```text
row_block_count = ceil(Infeature / xbar_row)
output_block_count = ceil(Outfeature / xbar_col)
spatial block order = row block major, then output block
```

Return one scope per spatial block. Merge adjacent scopes only when both input range and output range are contiguous under the same layer type.

- [ ] **Step 4: Add Phase 1 metadata for new plans**

Modify `MNSIM/Scheduling/phase1/resource_prepass.py` so `_prepass_ref(...).shape_signature` includes these keys for weighted layers:

```python
"_phase1_materialization_order": "row_major_input_then_output",
"_phase1_precision_group_size": _weight_bit_split_part(hw) * hw.xbar_polarity,
```

This does not let RoundPlanner inspect MNSIM internals. It only records the materializer contract in the opaque `prepass_ref` that Phase 2 consumes.

- [ ] **Step 5: Run materializer and Phase 1 resource tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_materializer.py tests/test_round_phase1_resource_prepass.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Scheduling/phase2/materializer.py MNSIM/Scheduling/phase1/resource_prepass.py tests/test_round_phase2_materializer.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: materialize phase2 selected scopes"
```

---

### Task 3: Static Evaluation Wrapper

**Files:**
- Create: `tests/test_round_phase2_static_eval.py`
- Create: `MNSIM/Scheduling/phase2/mnsim_static_eval.py`

- [ ] **Step 1: Write failing wrapper tests with a fake backend**

Create `tests/test_round_phase2_static_eval.py`:

```python
from MNSIM.Scheduling.phase2.mnsim_static_eval import run_mnsim_static_eval
from MNSIM.Scheduling.phase2.types import (
    BoundaryBinding,
    MetricTotals,
    StaticEvalFrame,
    StaticEvalRequest,
)


class FakeBackend:
    def run(self, request):
        return {
            "layers": [
                {
                    "local_layer_id": 0,
                    "layer_type": "conv",
                    "begin_time_s": 0.0,
                    "finish_time_s": 5e-9,
                    "area_um2": 10.0,
                    "power_w": 2.0,
                    "energy_j": 10e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
                {
                    "local_layer_id": 1,
                    "layer_type": "pooling",
                    "begin_time_s": 5e-9,
                    "finish_time_s": 7e-9,
                    "area_um2": 3.0,
                    "power_w": 1.0,
                    "energy_j": 2e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
            ],
            "raw_totals": {
                "raw_latency_total_s": 7e-9,
                "raw_area_total_um2": 13.0,
                "raw_power_total_w": 3.0,
                "raw_energy_total_j": 12e-9,
            },
            "mapping_summary": {"layer_count": 2},
        }


def _binding(local_layer_id, origin_layer_id, contribution_kind="complete_output"):
    return BoundaryBinding(
        local_layer_id=local_layer_id,
        hook_node_id=None,
        origin_layer_id=origin_layer_id,
        origin_layer_type="pooling" if origin_layer_id == "Pool1" else "conv",
        report_role="reportable",
        tensor_role="activation",
        shape={"kind": "full"},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind=contribution_kind,
        source_state_ids=[],
        downstream_consumer_ids=[],
    )


def test_filters_context_layers_and_normalizes_latency_offset():
    request = StaticEvalRequest(
        item_id="Z0",
        item_kind="zero_xbar_structural",
        netstruct=[[({"type": "conv"}, None)], [({"type": "pooling"}, None)]],
        sim_config_path="SimConfig.ini",
        multiple=[1, 1],
        latency_api="pipe",
        latency_fill_mode=0,
        frame=StaticEvalFrame(
            context_layer_ids=[0],
            reportable_layer_ids=[1],
            latency_offset_source_ids=[0],
            boundary_bindings=[_binding(1, "Pool1", "structural_output")],
        ),
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "ok"
    assert len(result.layer_results) == 2
    assert result.layer_results[0].report_role == "context_only"
    assert result.layer_results[1].begin_time_s == 0.0
    assert result.layer_results[1].finish_time_s == 2e-9
    assert result.item_totals == MetricTotals(
        latency_s=2e-9,
        energy_j=2e-9,
        power_w=1.0,
        area_um2=3.0,
    )
    assert result.raw_mnsim_summary["raw_latency_total_s"] == 7e-9


def test_rejects_multiple_length_mismatch():
    request = StaticEvalRequest(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[[({"type": "conv"}, None)]],
        sim_config_path="SimConfig.ini",
        multiple=[1, 1],
        latency_api="pipe",
        latency_fill_mode=0,
        frame=StaticEvalFrame(
            context_layer_ids=[],
            reportable_layer_ids=[0],
            latency_offset_source_ids=[],
            boundary_bindings=[_binding(0, "Conv1")],
        ),
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.warnings[0].startswith("netstruct_validation")
```

- [ ] **Step 2: Run wrapper tests and confirm they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_static_eval.py -v
```

Expected: FAIL with missing `run_mnsim_static_eval`.

- [ ] **Step 3: Implement `run_mnsim_static_eval` with fake-backend support**

Implement validation:

```text
multiple length == len(netstruct)
context/reportable ids are within range
concat rejected
zero_xbar_structural has at least one reportable layer
standalone structural reportable layer requires context ids
```

Implement normalization:

```text
report_role = context_only if local id in context_layer_ids else reportable
context_ready_time = max(finish_time_s for context_layer_ids)
subtract context_ready_time from reportable begin/finish only
item_totals aggregate reportable layers only
```

When validation fails, return `StaticEvalResult(status="error", item_totals=MetricTotals(0,0,0,0), warnings=[...])`.

- [ ] **Step 4: Add original MNSIM backend**

Inside `mnsim_static_eval.py`, add `_OriginalMnsimBackend.run(request)`:

```text
TCG(request.netstruct, request.sim_config_path, multiple=request.multiple)
Model_latency(..., TCG_mapping=tcg)
calculate_model_latency(...) or calculate_model_latency_nopipe()
Model_area(..., TCG_mapping=tcg)
Model_inference_power(..., TCG_mapping=tcg)
Model_energy(..., model_latency=latency, model_power=power, TCG_mapping=tcg)
```

Extract per-layer metrics from:

```text
latency.begin_time / latency.finish_time / latency.total_* lists
area.arch_area
power.arch_power
energy.arch_energy
tcg.layer_tileinfo
```

Convert ns to seconds and nJ to joules before constructing `StaticEvalResult`.

- [ ] **Step 5: Run wrapper tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_static_eval.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Scheduling/phase2/mnsim_static_eval.py tests/test_round_phase2_static_eval.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase2 static eval wrapper"
```

---

### Task 4: BoundaryStateManager

**Files:**
- Create: `tests/test_round_phase2_boundary.py`
- Create: `MNSIM/Scheduling/phase2/boundary.py`

- [ ] **Step 1: Write failing Boundary tests**

Create `tests/test_round_phase2_boundary.py`:

```python
from MNSIM.Scheduling.phase2.boundary import BoundaryStateManager
from MNSIM.Scheduling.phase2.types import (
    BOUNDARY_STATUS_COMPLETE_OUTPUT,
    BOUNDARY_STATUS_HOOK_COMPLETE,
    BOUNDARY_STATUS_OUTPUT_SUBSET,
    BOUNDARY_STATUS_PARTIAL_SUM,
    BoundaryBinding,
    BoundaryCommitRequest,
    HookResult,
    MetricTotals,
    StaticEvalResult,
)


def _binding(origin, contribution, input_range, output_range, full_input, full_output, source_state_ids=None):
    return BoundaryBinding(
        local_layer_id=0,
        hook_node_id=None,
        origin_layer_id=origin,
        origin_layer_type="conv",
        report_role="reportable",
        tensor_role="activation",
        shape={"C": full_output[1], "H": 1, "W": 1},
        region={"kind": "channel"},
        input_range=input_range,
        output_range=output_range,
        full_input_range=full_input,
        full_output_range=full_output,
        contribution_kind=contribution,
        source_state_ids=source_state_ids or ["model_input"],
        downstream_consumer_ids=["Relu1"],
    )


def _static_result(item_id, binding):
    return StaticEvalResult(
        item_id=item_id,
        item_kind="weighted_partial",
        status="ok",
        layer_results=[],
        item_totals=MetricTotals(1e-9, 2e-9, 2.0, 3.0),
        mapping_summary={},
        raw_mnsim_summary={},
        boundary_bindings=[binding],
        warnings=[],
    )


def test_partial_sum_output_subset_and_complete_closure():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")

    b0 = _binding("Conv1", "partial_contribution", [0, 2], [0, 2], [0, 4], [0, 4])
    r0 = mgr.commit(
        BoundaryCommitRequest(
            item_id="R0",
            item_kind="weighted_physical_round",
            execution_index=0,
            input_view_id="view:R0",
            source_kind="static_eval",
            static_eval_result=_static_result("R0", b0),
            hook_result=None,
            boundary_bindings=[b0],
        )
    )
    assert r0.completion_updates[0]["completion_status"] == BOUNDARY_STATUS_PARTIAL_SUM

    b1 = _binding("Conv1", "partial_contribution", [2, 4], [0, 2], [0, 4], [0, 4])
    r1 = mgr.commit(
        BoundaryCommitRequest(
            item_id="R1",
            item_kind="weighted_physical_round",
            execution_index=1,
            input_view_id="view:R1",
            source_kind="static_eval",
            static_eval_result=_static_result("R1", b1),
            hook_result=None,
            boundary_bindings=[b1],
        )
    )
    assert r1.completion_updates[0]["completion_status"] == BOUNDARY_STATUS_OUTPUT_SUBSET

    b2 = _binding("Conv1", "partial_contribution", [0, 4], [2, 4], [0, 4], [0, 4])
    r2 = mgr.commit(
        BoundaryCommitRequest(
            item_id="R2",
            item_kind="weighted_physical_round",
            execution_index=2,
            input_view_id="view:R2",
            source_kind="static_eval",
            static_eval_result=_static_result("R2", b2),
            hook_result=None,
            boundary_bindings=[b2],
        )
    )
    assert r2.completion_updates[-1]["completion_status"] == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert r2.output_state_bindings[-1]["downstream_visible"] is True


def test_downstream_input_view_rejects_partial_state():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["R1"],
    )

    try:
        mgr.build_input_view_for_state_ids("R1", [state.state_id], allow_partial=False)
    except Exception as exc:
        assert "partial" in str(exc)
    else:
        raise AssertionError("expected partial state rejection")


def test_hook_commit_creates_hook_complete_state():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("state:conv1", shape={"C": 8}, tensor_role="activation")
    binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="hook_output",
        source_state_ids=["state:conv1"],
        downstream_consumer_ids=["Conv2"],
    )
    result = mgr.commit(
        BoundaryCommitRequest(
            item_id="Z0",
            item_kind="zero_xbar_round",
            execution_index=0,
            input_view_id="view:Z0",
            source_kind="hook",
            static_eval_result=None,
            hook_result=HookResult("Z0", "zero_xbar_round", [binding], []),
            boundary_bindings=[binding],
        )
    )

    assert result.created_states[0].status == BOUNDARY_STATUS_HOOK_COMPLETE
```

- [ ] **Step 2: Run Boundary tests and confirm they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_boundary.py -v
```

Expected: FAIL with missing `BoundaryStateManager`.

- [ ] **Step 3: Implement `BoundaryStateManager`**

Implement:

```python
class BoundaryStateManager:
    def seed_model_input(self, state_id, shape, tensor_role="activation") -> BoundaryStateRecord
    def build_input_view(self, item) -> BoundaryInputView
    def build_input_view_for_state_ids(self, item_id, state_ids, allow_partial=False) -> BoundaryInputView
    def commit(self, request: BoundaryCommitRequest) -> BoundaryCommitResult
```

State behavior:

```text
model_input -> downstream readable
partial_sum -> only same original layer continuation readable
output_subset -> internal only in v1
complete_output -> downstream readable
hook_complete -> downstream readable
```

Commit behavior:

```text
partial_contribution:
  merge same origin/output_range contributions
  input range incomplete -> partial_sum
  input range complete -> output_subset
  full output range complete -> complete_output

complete_output / structural_output:
  create complete_output

hook_output:
  create hook_complete

context_only:
  ignore_context audit event, no state
```

- [ ] **Step 4: Run Boundary tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_boundary.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Scheduling/phase2/boundary.py tests/test_round_phase2_boundary.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase2 boundary state manager"
```

---

### Task 5: NetStruct Builder And RoundLowerer

**Files:**
- Create: `tests/test_round_phase2_lowerer.py`
- Create: `MNSIM/Scheduling/phase2/netstruct_builder.py`
- Create: `MNSIM/Scheduling/phase2/lowerer.py`

- [ ] **Step 1: Write failing lowerer tests**

Create `tests/test_round_phase2_lowerer.py`:

```python
from MNSIM.Scheduling.phase2.boundary import BoundaryStateManager
from MNSIM.Scheduling.phase2.lowerer import RoundLowerer
from MNSIM.Scheduling.phase2.types import CONTRIBUTION_HOOK_OUTPUT
from MNSIM.Scheduling.round_types import (
    LocalGraphNode,
    PrepassRef,
    ResourceProof,
    SelectedEntry,
    WeightedLayerSegment,
    WeightedPhysicalRound,
    ZeroXbarRound,
    ZeroXbarSegment,
)


def _prepass(layer_type, shape):
    return PrepassRef(
        ref_id=f"m0:L0:{layer_type}:mnsim20_formula_v1",
        model_id="m0",
        layer_id="L0",
        layer_type=layer_type,
        resource_version="mnsim20_formula_v1",
        sim_config_fingerprint="xbar=128x128;device=2;weight=8;polarity=2",
        layer_config_fingerprint="layer:test",
        weight_ref_id="w0",
        shape_signature=shape,
    )


def _entry(layer_type, shape, end=14, complete=True):
    return SelectedEntry(
        layer_id="L0",
        layer_type=layer_type,
        xbar_range_start=0,
        xbar_range_end=end,
        used_xbars=end,
        layer_complete_in_round=complete,
        prepass_ref=_prepass(layer_type, shape),
    )


def test_lowers_full_layer_pack_conv_and_hook_tail():
    entry = _entry(
        "conv",
        {
            "Inputchannel": 3,
            "Outputchannel": 16,
            "Kernelsize": 3,
            "Inputsize": [32, 32],
            "Outputsize": [30, 30],
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        },
        end=14,
        complete=True,
    )
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[entry],
        local_graph_nodes=[],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 14),
        ordered_segments=[
            WeightedLayerSegment(0, entry, LocalGraphNode("L0", "conv", "weighted")),
            ZeroXbarSegment(1, "hook_only", [LocalGraphNode("Relu1", "relu", "hook")]),
        ],
    )
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", {"C": 3}, "activation")
    view = mgr.build_input_view_for_state_ids("R0", ["model_input"])

    lowered = RoundLowerer(xbar_size=(128, 128)).lower(item, view)

    assert lowered.static_eval_request is not None
    assert lowered.static_eval_request.netstruct[0][0][0]["type"] == "conv"
    assert lowered.boundary_bindings[0].origin_layer_id == "L0"
    assert lowered.boundary_bindings[1].contribution_kind == CONTRIBUTION_HOOK_OUTPUT


def test_lowers_partial_fc_to_synthetic_local_fc():
    entry = _entry(
        "fc",
        {
            "Infeature": 256,
            "Outfeature": 256,
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        },
        end=14,
        complete=False,
    )
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="single_layer_split",
        selected_entries=[entry],
        local_graph_nodes=[],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 14),
        ordered_segments=[
            WeightedLayerSegment(0, entry, LocalGraphNode("L0", "fc", "weighted")),
        ],
    )

    lowered = RoundLowerer(xbar_size=(128, 128)).lower(
        item,
        BoundaryStateManager().build_input_view_for_state_ids("R0", []),
    )

    layer_info = lowered.static_eval_request.netstruct[0][0][0]
    assert layer_info["type"] == "fc"
    assert layer_info["Infeature"] == 128
    assert layer_info["Outfeature"] == 128
    assert lowered.boundary_bindings[0].contribution_kind == "partial_contribution"


def test_standalone_pooling_uses_context_frame():
    item = ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind="structural_mnsim",
        selected_entries=[],
        local_graph_nodes=[LocalGraphNode("Pool1", "pooling", "structural_non_weighted")],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 0),
    )
    mgr = BoundaryStateManager()
    mgr.seed_model_input("state:conv1", {"Outputchannel": 8, "Outputsize": [16, 16], "outputbit": 8}, "activation")
    view = mgr.build_input_view_for_state_ids("Z0", ["state:conv1"])

    lowered = RoundLowerer(xbar_size=(128, 128)).lower(item, view)

    assert lowered.static_eval_request.frame.context_layer_ids == [0]
    assert lowered.static_eval_request.frame.reportable_layer_ids == [1]
    assert lowered.static_eval_request.netstruct[1][0][0]["type"] == "pooling"


def test_concat_fails_fast():
    item = ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind="structural_mnsim",
        selected_entries=[],
        local_graph_nodes=[LocalGraphNode("Concat1", "concat", "structural_non_weighted")],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 0),
    )

    try:
        RoundLowerer(xbar_size=(128, 128)).lower(
            item,
            BoundaryStateManager().build_input_view_for_state_ids("Z0", []),
        )
    except Exception as exc:
        assert "concat" in str(exc)
    else:
        raise AssertionError("expected concat rejection")
```

- [ ] **Step 2: Run lowerer tests and confirm they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_lowerer.py -v
```

Expected: FAIL with missing `RoundLowerer`.

- [ ] **Step 3: Implement `netstruct_builder.py`**

Implement:

```python
def build_netstruct(layer_infos):
    return [[(copy.deepcopy(layer_info), None)] for layer_info in layer_infos]
```

The builder must set:

```text
Layerindex = local layer id
Inputindex = relative local input offsets
Outputindex = relative local output offsets
row_split_num = 1 for synthetic already-sliced weighted layers
weight_bit_split_part = 1 for synthetic local layer dictionaries passed to wrapper tests
```

When a real original MNSIM layer info exists in `LayerIR.backend_attrs["mnsim_layer_info"]`, preserve fields such as `Inputsize`, `Outputsize`, `Stride`, `Padding`, `Inputbit`, `Weightbit`, and `outputbit`.

- [ ] **Step 4: Implement `RoundLowerer`**

Required constructor:

```python
RoundLowerer(xbar_size: tuple[int, int])
```

Required method:

```python
lower(item, input_view) -> LoweredItem
```

Behavior:

```text
WeightedPhysicalRound:
  selected weighted segments -> static_eval_request
  hook segments -> hook_output boundary bindings
  full pack complete weighted -> complete_output binding
  split/continuation weighted -> partial_contribution binding

ZeroXbarRound hook_only:
  no static_eval_request
  hook_bindings and boundary_bindings only

ZeroXbarRound structural_mnsim:
  context stub producer(s) -> real structural layer
  static_eval_request with frame context/reportable ids

ZeroXbarRound mixed_non_weighted:
  lower hook nodes as hook bindings
  lower structural nodes as context-framed static eval

concat:
  raise Phase2Invalid("UNSUPPORTED_STATIC_CONCAT", ...)
```

- [ ] **Step 5: Run lowerer tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_lowerer.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Scheduling/phase2/netstruct_builder.py MNSIM/Scheduling/phase2/lowerer.py tests/test_round_phase2_lowerer.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: lower phase2 items to local work"
```

---

### Task 6: ReportAggregator

**Files:**
- Create: `tests/test_round_phase2_report.py`
- Create: `MNSIM/Scheduling/phase2/report.py`

- [ ] **Step 1: Write failing report tests**

Create `tests/test_round_phase2_report.py`:

```python
from MNSIM.Scheduling.phase2.report import ReportAggregator
from MNSIM.Scheduling.phase2.types import (
    BoundaryCommitResult,
    MetricTotals,
    StaticEvalResult,
    StaticLayerResult,
)


def _commit(item_id, origin_layer_id, status="complete_output"):
    return BoundaryCommitResult(
        item_id=item_id,
        item_kind="weighted_physical_round",
        status="ok",
        created_states=[],
        updated_states=[],
        released_states=[],
        events=[],
        completion_updates=[
            {
                "origin_layer_id": origin_layer_id,
                "completion_status": status,
                "state_id": f"state:{origin_layer_id}",
                "input_range_covered": {"kind": "full"},
                "output_range_covered": {"kind": "full"},
                "downstream_visible": True,
            }
        ],
        visibility_updates=[],
        output_state_bindings=[],
        warnings=[],
    )


def test_aggregates_normalized_metrics_and_tops_per_w():
    agg = ReportAggregator(
        model_name="m0",
        sim_config_path="SimConfig.ini",
        ops_total=2_000_000_000,
        ops_count_policy="mac_as_1_op",
    )
    result = StaticEvalResult(
        item_id="R0",
        item_kind="weighted_full_pack",
        status="ok",
        layer_results=[
            StaticLayerResult(0, "Conv1", "conv", "reportable", 1e-6, 0.0, 1e-6, 100.0, 2.0, 2e-6, {}),
            StaticLayerResult(1, "ctx", "conv", "context_only", 99e-6, 0.0, 99e-6, 999.0, 9.0, 999e-6, {}),
        ],
        item_totals=MetricTotals(latency_s=1e-6, energy_j=2e-6, power_w=2.0, area_um2=100.0),
        mapping_summary={"layer_count": 2},
        raw_mnsim_summary={"raw_energy_total_j": 1001e-6},
        boundary_bindings=[],
        warnings=["context filtered"],
    )

    agg.record_static_result(result, _commit("R0", "Conv1"))
    report = agg.finalize()

    assert report.inference_metrics["latency_s"] == 1e-6
    assert report.inference_metrics["energy_j"] == 2e-6
    assert report.inference_metrics["power_avg_w"] == 2.0
    assert report.inference_metrics["tops"] == 2000.0
    assert report.inference_metrics["tops_per_w"] == 1000.0
    assert report.raw_mnsim_audit[0]["raw_energy_total_j"] == 1001e-6


def test_area_uses_max_mapped_area_fallback_not_sum():
    agg = ReportAggregator("m0", "SimConfig.ini", ops_total=1, ops_count_policy="mac_as_1_op")
    for idx, area in enumerate([10.0, 20.0]):
        agg.record_static_result(
            StaticEvalResult(
                item_id=f"R{idx}",
                item_kind="weighted_full_pack",
                status="ok",
                layer_results=[],
                item_totals=MetricTotals(latency_s=1.0, energy_j=1.0, power_w=1.0, area_um2=area),
                mapping_summary={},
                raw_mnsim_summary={},
                boundary_bindings=[],
                warnings=[],
            ),
            _commit(f"R{idx}", f"Conv{idx}"),
        )

    report = agg.finalize()

    assert report.inference_metrics["area_um2"] == 20.0
    assert "max reportable item mapped area" in report.assumptions[0]
```

- [ ] **Step 2: Run report tests and confirm they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_report.py -v
```

Expected: FAIL with missing `ReportAggregator`.

- [ ] **Step 3: Implement `ReportAggregator`**

Implement:

```python
class ReportAggregator:
    def __init__(self, model_name, sim_config_path, ops_total, ops_count_policy, area_um2=None)
    def record_static_result(self, static_eval_result, boundary_commit_result)
    def record_hook_result(self, hook_result, boundary_commit_result)
    def finalize(self) -> Phase2Report
```

Rules:

```text
latency_s = sum item_totals.latency_s in item order
energy_j = sum item_totals.energy_j
power_avg_w = energy_j / latency_s if latency_s > 0 else 0.0
mapped_active_area_um2 = max item_totals.area_um2
area_um2 = explicit area provider value, otherwise mapped_active_area_um2 with assumption
tops = ops_total / latency_s / 1e12
tops_per_w = ops_total / energy_j / 1e12
context_only layers and raw_mnsim_summary never enter final totals
```

- [ ] **Step 4: Run report tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_report.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Scheduling/phase2/report.py tests/test_round_phase2_report.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: aggregate phase2 reports"
```

---

### Task 7: Phase 2 Executor

**Files:**
- Create: `tests/test_round_phase2_executor.py`
- Create: `MNSIM/Scheduling/phase2/executor.py`
- Modify: `MNSIM/Scheduling/phase2/__init__.py`

- [ ] **Step 1: Write failing executor test**

Create `tests/test_round_phase2_executor.py`:

```python
from MNSIM.Scheduling.phase2.executor import execute_round_plan
from MNSIM.Scheduling.phase2.types import MetricTotals, StaticEvalResult
from MNSIM.Scheduling.round_types import (
    LocalGraphNode,
    PrepassRef,
    ResourceProof,
    RoundPlan,
    SelectedEntry,
    WeightedLayerSegment,
    WeightedPhysicalRound,
)


class FakeStaticBackend:
    def run(self, request):
        return {
            "layers": [
                {
                    "local_layer_id": 0,
                    "layer_type": request.netstruct[0][0][0]["type"],
                    "begin_time_s": 0.0,
                    "finish_time_s": 1e-6,
                    "area_um2": 10.0,
                    "power_w": 2.0,
                    "energy_j": 2e-6,
                    "tcg_mapping": {"tilenum": 1},
                }
            ],
            "raw_totals": {
                "raw_latency_total_s": 1e-6,
                "raw_area_total_um2": 10.0,
                "raw_power_total_w": 2.0,
                "raw_energy_total_j": 2e-6,
            },
            "mapping_summary": {"layer_count": 1},
        }


def test_executes_one_weighted_round_end_to_end_with_fake_backend():
    prepass = PrepassRef(
        ref_id="m0:L0:conv:mnsim20_formula_v1",
        model_id="m0",
        layer_id="L0",
        layer_type="conv",
        resource_version="mnsim20_formula_v1",
        sim_config_fingerprint="xbar=128x128;device=2;weight=8;polarity=2",
        layer_config_fingerprint="layer:test",
        weight_ref_id="w0",
        shape_signature={
            "Inputchannel": 3,
            "Outputchannel": 16,
            "Kernelsize": 3,
            "Inputsize": [32, 32],
            "Outputsize": [30, 30],
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        },
    )
    entry = SelectedEntry("L0", "conv", 0, 14, 14, True, prepass)
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[entry],
        local_graph_nodes=[],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 14),
        ordered_segments=[WeightedLayerSegment(0, entry, LocalGraphNode("L0", "conv", "weighted"))],
    )
    plan = RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 128},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )

    report = execute_round_plan(
        plan,
        sim_config_path="SimConfig.ini",
        ops_total=2_000_000,
        ops_count_policy="mac_as_1_op",
        static_backend=FakeStaticBackend(),
        model_inputs={"model_input": {"shape": {"C": 3}, "tensor_role": "activation"}},
    )

    assert report.feasible is True
    assert report.execution_item_count == 1
    assert report.inference_metrics["latency_s"] == 1e-6
    assert report.inference_metrics["energy_j"] == 2e-6
```

- [ ] **Step 2: Run executor test and confirm it fails**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_executor.py -v
```

Expected: FAIL with missing `execute_round_plan`.

- [ ] **Step 3: Implement executor**

Create `MNSIM/Scheduling/phase2/executor.py`:

```text
validate RoundPlan.feasible
seed model input states
for item in execution_items sorted by item_index:
  input_view = boundary.build_input_view(item)
  lowered = lowerer.lower(item, input_view)
  if lowered.static_eval_request:
    static_result = run_mnsim_static_eval(lowered.static_eval_request, backend=static_backend)
  if no static request and hook bindings exist:
    hook_result = HookResult(...)
  commit_result = boundary.commit(BoundaryCommitRequest(...))
  aggregator.record(...)
return aggregator.finalize()
```

Do not repack, reorder, or change `SelectedEntry`.

- [ ] **Step 4: Export executor**

Modify `MNSIM/Scheduling/phase2/__init__.py` to export:

```python
execute_round_plan
BoundaryStateManager
RoundLowerer
ReportAggregator
run_mnsim_static_eval
```

- [ ] **Step 5: Run executor and prior Phase 2 tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_types.py tests/test_round_phase2_materializer.py tests/test_round_phase2_static_eval.py tests/test_round_phase2_boundary.py tests/test_round_phase2_lowerer.py tests/test_round_phase2_report.py tests/test_round_phase2_executor.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Scheduling/phase2/__init__.py MNSIM/Scheduling/phase2/executor.py tests/test_round_phase2_executor.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: execute phase2 round plans"
```

---

### Task 8: Original MNSIM Static Smoke And Architecture Regression

**Files:**
- Create: `tests/test_round_phase2_mnsim_static_smoke.py`
- Modify: `tests/test_round_scheduling_architecture.py`

- [ ] **Step 1: Write original MNSIM smoke test**

Create `tests/test_round_phase2_mnsim_static_smoke.py`:

```python
import collections

from MNSIM.Scheduling.phase2.mnsim_static_eval import run_mnsim_static_eval
from MNSIM.Scheduling.phase2.types import BoundaryBinding, StaticEvalFrame, StaticEvalRequest


def test_original_mnsim_backend_runs_single_fc_static_eval():
    layer_info = collections.OrderedDict(
        [
            ("type", "fc"),
            ("Infeature", 16),
            ("Outfeature", 8),
            ("Inputbit", 8),
            ("Weightbit", 8),
            ("outputbit", 8),
            ("row_split_num", 1),
            ("weight_bit_split_part", 1),
            ("Inputindex", [-1]),
            ("Outputindex", []),
            ("Layerindex", 0),
        ]
    )
    binding = BoundaryBinding(
        local_layer_id=0,
        hook_node_id=None,
        origin_layer_id="FC0",
        origin_layer_type="fc",
        report_role="reportable",
        tensor_role="activation",
        shape={"Outfeature": 8},
        region={"kind": "full"},
        input_range=[0, 16],
        output_range=[0, 8],
        full_input_range=[0, 16],
        full_output_range=[0, 8],
        contribution_kind="complete_output",
        source_state_ids=["model_input"],
        downstream_consumer_ids=[],
    )
    request = StaticEvalRequest(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[[(layer_info, None)]],
        sim_config_path="SimConfig.ini",
        multiple=[1],
        latency_api="nopipe",
        latency_fill_mode=0,
        frame=StaticEvalFrame([], [0], [], [binding]),
    )

    result = run_mnsim_static_eval(request)

    assert result.status == "ok"
    assert result.item_totals.latency_s >= 0.0
    assert result.item_totals.area_um2 > 0.0
```

- [ ] **Step 2: Add architecture import regression**

Modify `tests/test_round_scheduling_architecture.py` to assert:

```python
def test_phase2_core_lives_under_scheduling_package():
    import MNSIM.Scheduling.phase2 as phase2

    assert phase2.BoundaryStateManager.__module__ == "MNSIM.Scheduling.phase2.boundary"
    assert phase2.RoundLowerer.__module__ == "MNSIM.Scheduling.phase2.lowerer"
    assert phase2.ReportAggregator.__module__ == "MNSIM.Scheduling.phase2.report"
```

- [ ] **Step 3: Run smoke and architecture tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase2_mnsim_static_smoke.py tests/test_round_scheduling_architecture.py -v
```

Expected: PASS. If the original MNSIM smoke fails because `SimConfig.ini` has a hardware field mismatch, keep the unit-test backend path intact and fix the wrapper extraction or test NetStruct field, not the MNSIM Model_* formulas.

- [ ] **Step 4: Run full round simulator test subset**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_types.py tests/test_round_phase1_config.py tests/test_round_phase1_resource_prepass.py tests/test_round_phase1_planner.py tests/test_round_phase1_mnsim_frontend.py tests/test_round_phase1_real_networks.py tests/test_round_phase2_types.py tests/test_round_phase2_materializer.py tests/test_round_phase2_static_eval.py tests/test_round_phase2_boundary.py tests/test_round_phase2_lowerer.py tests/test_round_phase2_report.py tests/test_round_phase2_executor.py tests/test_round_phase2_mnsim_static_smoke.py tests/test_round_scheduling_architecture.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git add tests/test_round_phase2_mnsim_static_smoke.py tests/test_round_scheduling_architecture.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "test: verify phase2 mnsim static integration"
```

---

## Self-Review Checklist For Implementers

Before reporting implementation complete:

- [ ] `RoundPlan.execution_items[]` is consumed as fixed input.
- [ ] No Phase 2 code repacks or reorders rounds.
- [ ] `SelectedEntry.prepass_ref + xbar_range` is the only source for partial materialization.
- [ ] MNSIM receives only legal complete item-local `NetStruct`.
- [ ] `partial_sum` and `output_subset` are never downstream readable.
- [ ] Hook-only work does not call MNSIM static Model_*.
- [ ] Structural standalone work uses context frame or fails fast.
- [ ] `concat` fails fast.
- [ ] Context-only metrics do not enter final totals.
- [ ] Raw MNSIM totals are audit-only.
- [ ] Area is not item-summed as chip area.
- [ ] TOPS/W uses model op-count metadata.
- [ ] DDR/boundary-store read/write latency and energy are not modeled.
- [ ] Existing MNSIM power/energy formulas are not modified.

## Final Verification

Run from the nested simulator directory:

```bash
cd /home/xzh/workspace/cim_project/MNSIM-2.0/.worktrees/task-20260707-round-orchestrated-mnsim/round_orchestrated_simulator/MNSIM-2.0
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_types.py tests/test_round_phase1_config.py tests/test_round_phase1_resource_prepass.py tests/test_round_phase1_planner.py tests/test_round_phase1_mnsim_frontend.py tests/test_round_phase1_real_networks.py tests/test_round_phase2_types.py tests/test_round_phase2_materializer.py tests/test_round_phase2_static_eval.py tests/test_round_phase2_boundary.py tests/test_round_phase2_lowerer.py tests/test_round_phase2_report.py tests/test_round_phase2_executor.py tests/test_round_phase2_mnsim_static_smoke.py tests/test_round_scheduling_architecture.py -v
```

Expected final result:

```text
all selected tests pass
```

Then run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git diff --check
```

Expected final result:

```text
no output
```
