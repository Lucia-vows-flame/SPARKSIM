# Phase 1 ModelIR Round Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1 of the new round-orchestrated simulator: independent `ModelIR` input, resource prepass, deterministic round planning, and JSON-friendly `RoundPlan.execution_items[]`.

**Architecture:** Add a standalone Phase 1 core under `MNSIM/Interface/round_phase1_*.py`. Tests construct synthetic `ModelIR` objects, config dictionaries, `ConfigParser` objects, and temporary `SimConfig.ini` files directly; Phase 1 does not call MNSIM TCG, Model_*, or existing scheduled fragment code. The implementation is TDD-first, with small commits after each component.

**Tech Stack:** Python standard library only for implementation (`dataclasses`, `math`, `configparser`, `pathlib`, `typing`); tests use `pytest`; run tests via `/home/xzh/micromamba/envs/workspace/bin/python -m pytest`.

---

## Source Spec

Implement exactly the Phase 1 scope in:

`docs/superpowers/specs/2026-07-20-phase1-modelir-round-planner-design.md`

Do not implement Phase 2 lowering, TCG calls, Model_* calls, BoundaryStateManager runtime commit, PyTorch/ONNX import, SNN support, or scheduled fragment integration.

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

- `MNSIM/Interface/round_phase1_types.py`
  Public dataclasses, constants, error codes, deterministic `to_dict/from_dict` helpers.

- `MNSIM/Interface/round_phase1_config.py`
  `HardwareProjection` and SimConfig path / `ConfigParser` / config-dict parsing.

- `MNSIM/Interface/round_phase1_resource_prepass.py`
  ResourcePrepass v1 formulas for `conv`, `fc`, and `fc_like_projection`.

- `MNSIM/Interface/round_phase1_planner.py`
  ModelIR validation, `build_round_plan`, stable ready-set planner, split/continuation, zero-xbar closure.

- `tests/test_round_phase1_types.py`
  Dataclass serialization and string-literal schema tests.

- `tests/test_round_phase1_config.py`
  Hardware projection tests.

- `tests/test_round_phase1_resource_prepass.py`
  Formula and invalid footprint tests.

- `tests/test_round_phase1_planner.py`
  Round planning, DAG, zero-xbar, RNN-like, invalid model tests.

Do not modify existing MNSIM TCG, `Model_*`, `network.py`, `quantize.py`, or scheduled runtime files.

## Public Names To Preserve

Use these exact names so later Phase 2 code can depend on them:

```python
build_round_plan(model_ir, sim_config) -> RoundPlan
build_hardware_projection(sim_config) -> HardwareProjection
build_layer_resource_descriptors(model_ir, hardware) -> list[LayerResourceDescriptor]
```

Dataclasses:

```python
SequenceSpec
LayerIR
ModelIR
PrepassRef
CompletionContract
LayerResourceDescriptor
SelectedEntry
LocalGraphNode
BoundaryRef
SameItemEdge
ResourceProof
ExecutionItem
WeightedPhysicalRound
ZeroXbarRound
RoundPlan
HardwareProjection
```

Use string literal fields, not Python Enum objects.

---

### Task 1: Public Dataclasses And Serialization

**Files:**
- Create: `tests/test_round_phase1_types.py`
- Create: `MNSIM/Interface/round_phase1_types.py`

- [ ] **Step 1: Write the failing dataclass serialization tests**

Create `tests/test_round_phase1_types.py` with:

```python
from MNSIM.Interface.round_phase1_types import (
    CompletionContract,
    LayerIR,
    LocalGraphNode,
    ModelIR,
    PrepassRef,
    ResourceProof,
    RoundPlan,
    SelectedEntry,
    SequenceSpec,
    WeightedPhysicalRound,
    ZeroXbarRound,
    round_plan_from_dict,
)


def test_round_plan_serialization_is_deterministic_and_round_trips():
    seq = SequenceSpec(
        sequence_length=1,
        ordered_steps=False,
        cross_step_dependency=False,
        recurrent_state_ids=[],
    )
    model_ir = ModelIR(
        model_id="m0",
        backend_kind="ann",
        sequence_spec=seq,
        layers=[
            LayerIR(
                layer_id="Conv1",
                layer_type="conv",
                is_weighted=True,
                stable_topological_index=0,
                input_tensor_ids=["input"],
                output_tensor_ids=["conv1_raw"],
                input_layer_ids=[],
                output_consumer_ids=["Relu1"],
                weight_ref_id="w_conv1",
                shape={"Inputchannel": 3, "Outputchannel": 8, "Kernelsize": 3},
                backend_attrs={},
            )
        ],
    )
    prepass_ref = PrepassRef(
        ref_id="prepass:Conv1",
        model_id="m0",
        layer_id="Conv1",
        layer_type="conv",
        resource_version="mnsim20_formula_v1",
        sim_config_fingerprint="sim:abc",
        layer_config_fingerprint="layer:abc",
        weight_ref_id="w_conv1",
        shape_signature={"Inputchannel": 3, "Outputchannel": 8, "Kernelsize": 3},
    )
    item = WeightedPhysicalRound(
        item_id="W0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[
            SelectedEntry(
                layer_id="Conv1",
                layer_type="conv",
                xbar_range_start=0,
                xbar_range_end=2,
                used_xbars=2,
                layer_complete_in_round=True,
                prepass_ref=prepass_ref,
            )
        ],
        local_graph_nodes=[
            LocalGraphNode(layer_id="Conv1", layer_type="conv", node_role="weighted")
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=16, used_xbars=2),
        backend_runtime={},
    )
    plan = RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 16},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": model_ir.model_id},
        feasible=True,
        invalid_reason=None,
    )

    first = plan.to_dict()
    second = plan.to_dict()

    assert first == second
    assert first["execution_items"][0]["item_kind"] == "weighted_physical_round"
    assert first["execution_items"][0]["selected_entries"][0]["prepass_ref"]["ref_id"] == "prepass:Conv1"
    assert round_plan_from_dict(first).to_dict() == first


def test_zero_xbar_round_schema_uses_no_selected_entries_and_zero_xbars():
    item = ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind="hook_only",
        selected_entries=[],
        local_graph_nodes=[
            LocalGraphNode(layer_id="Relu1", layer_type="relu", node_role="hook")
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=16, used_xbars=0),
        backend_runtime={},
    )

    assert item.to_dict()["item_kind"] == "zero_xbar_round"
    assert item.to_dict()["selected_entries"] == []
    assert item.to_dict()["resource_proof"]["used_xbars"] == 0


def test_completion_contract_defaults_are_phase1_defaults():
    contract = CompletionContract()

    assert contract.partial_visibility == "internal_only"
    assert contract.downstream_visibility == "complete_only"
    assert contract.same_item_handoff_allowed is True
    assert contract.completion_hook_trigger == "on_complete"
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_types.py -v
```

Expected: fail during import with `ModuleNotFoundError: No module named 'MNSIM.Interface.round_phase1_types'`.

- [ ] **Step 3: Implement `round_phase1_types.py`**

Create `MNSIM/Interface/round_phase1_types.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


ITEM_KIND_WEIGHTED = "weighted_physical_round"
ITEM_KIND_ZERO_XBAR = "zero_xbar_round"

ROUND_KIND_FULL_LAYER_PACK = "full_layer_pack"
ROUND_KIND_SINGLE_LAYER_SPLIT = "single_layer_split"
ROUND_KIND_SINGLE_LAYER_CONTINUATION = "single_layer_continuation"

ZERO_XBAR_KIND_HOOK_ONLY = "hook_only"
ZERO_XBAR_KIND_STRUCTURAL_MNSIM = "structural_mnsim"
ZERO_XBAR_KIND_MIXED_NON_WEIGHTED = "mixed_non_weighted"

NODE_ROLE_WEIGHTED = "weighted"
NODE_ROLE_HOOK = "hook"
NODE_ROLE_STRUCTURAL_NON_WEIGHTED = "structural_non_weighted"

WEIGHTED_LAYER_TYPES = {"conv", "fc", "fc_like_projection"}
HOOK_LAYER_TYPES = {"relu", "bn", "view", "reshape", "flatten", "dropout", "backend_hook"}
STRUCTURAL_NON_WEIGHTED_LAYER_TYPES = {"pooling", "concat", "element_sum", "element_multiply"}
NON_WEIGHTED_LAYER_TYPES = HOOK_LAYER_TYPES | STRUCTURAL_NON_WEIGHTED_LAYER_TYPES
SUPPORTED_LAYER_TYPES = WEIGHTED_LAYER_TYPES | NON_WEIGHTED_LAYER_TYPES

INVALID_SIM_CONFIG = "INVALID_SIM_CONFIG"
INVALID_HARDWARE_PROJECTION = "INVALID_HARDWARE_PROJECTION"
UNSUPPORTED_LAYER_TYPE = "UNSUPPORTED_LAYER_TYPE"
INVALID_LAYER_SHAPE = "INVALID_LAYER_SHAPE"
INVALID_MODEL_DAG = "INVALID_MODEL_DAG"
INVALID_SEQUENCE_SPEC = "INVALID_SEQUENCE_SPEC"
INVALID_RESOURCE_FOOTPRINT = "INVALID_RESOURCE_FOOTPRINT"
INFEASIBLE_LAYER_ATOMIC_SCOPE = "INFEASIBLE_LAYER_ATOMIC_SCOPE"
INTERNAL_PLANNER_INVARIANT = "INTERNAL_PLANNER_INVARIANT"


class Phase1Invalid(Exception):
    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    @property
    def reason(self) -> str:
        return f"{self.code}: {self.detail}"


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _plain(val) for key, val in asdict(value).items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(val) for key, val in value.items()}
    return value


@dataclass
class DictSerializable:
    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


@dataclass
class SequenceSpec(DictSerializable):
    sequence_length: int
    ordered_steps: bool
    cross_step_dependency: bool
    recurrent_state_ids: list[str] = field(default_factory=list)


@dataclass
class LayerIR(DictSerializable):
    layer_id: str
    layer_type: str
    is_weighted: bool
    stable_topological_index: int
    input_tensor_ids: list[str] = field(default_factory=list)
    output_tensor_ids: list[str] = field(default_factory=list)
    input_layer_ids: list[str] = field(default_factory=list)
    output_consumer_ids: list[str] = field(default_factory=list)
    weight_ref_id: str | None = None
    shape: dict[str, Any] = field(default_factory=dict)
    backend_attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelIR(DictSerializable):
    model_id: str
    backend_kind: str
    sequence_spec: SequenceSpec
    layers: list[LayerIR]


@dataclass
class PrepassRef(DictSerializable):
    ref_id: str
    model_id: str
    layer_id: str
    layer_type: str
    resource_version: str
    sim_config_fingerprint: str
    layer_config_fingerprint: str
    weight_ref_id: str | None
    shape_signature: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompletionContract(DictSerializable):
    partial_visibility: str = "internal_only"
    downstream_visibility: str = "complete_only"
    same_item_handoff_allowed: bool = True
    completion_hook_trigger: str = "on_complete"


@dataclass
class LayerResourceDescriptor(DictSerializable):
    layer_id: str
    layer_type: str
    input_layer_ids: list[str]
    output_consumer_ids: list[str]
    required_xbar_count: int
    prepass_ref: PrepassRef
    completion_contract: CompletionContract = field(default_factory=CompletionContract)


@dataclass
class SelectedEntry(DictSerializable):
    layer_id: str
    layer_type: str
    xbar_range_start: int
    xbar_range_end: int
    used_xbars: int
    layer_complete_in_round: bool
    prepass_ref: PrepassRef


@dataclass
class LocalGraphNode(DictSerializable):
    layer_id: str
    layer_type: str
    node_role: str


@dataclass
class BoundaryRef(DictSerializable):
    state_id: str
    tensor_role: str
    producer_item_id: str | None = None
    consumer_item_ids: list[str] = field(default_factory=list)
    partial_or_complete: str = "complete"


@dataclass
class SameItemEdge(DictSerializable):
    producer_node_id: str
    consumer_node_id: str


@dataclass
class ResourceProof(DictSerializable):
    xbar_nums: int
    used_xbars: int


@dataclass
class ExecutionItem(DictSerializable):
    item_id: str
    item_index: int
    item_kind: str
    local_graph_nodes: list[LocalGraphNode]
    external_inputs: list[BoundaryRef]
    same_item_edges: list[SameItemEdge]
    external_outputs: list[BoundaryRef]
    partial_output_contract: list[BoundaryRef]
    resource_proof: ResourceProof
    backend_runtime: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightedPhysicalRound(ExecutionItem):
    round_kind: str = ROUND_KIND_FULL_LAYER_PACK
    selected_entries: list[SelectedEntry] = field(default_factory=list)

    def __init__(
        self,
        item_id: str,
        item_index: int,
        round_kind: str,
        selected_entries: list[SelectedEntry],
        local_graph_nodes: list[LocalGraphNode],
        external_inputs: list[BoundaryRef],
        same_item_edges: list[SameItemEdge],
        external_outputs: list[BoundaryRef],
        partial_output_contract: list[BoundaryRef],
        resource_proof: ResourceProof,
        backend_runtime: dict[str, Any] | None = None,
    ):
        super().__init__(
            item_id=item_id,
            item_index=item_index,
            item_kind=ITEM_KIND_WEIGHTED,
            local_graph_nodes=local_graph_nodes,
            external_inputs=external_inputs,
            same_item_edges=same_item_edges,
            external_outputs=external_outputs,
            partial_output_contract=partial_output_contract,
            resource_proof=resource_proof,
            backend_runtime={} if backend_runtime is None else backend_runtime,
        )
        self.round_kind = round_kind
        self.selected_entries = selected_entries


@dataclass
class ZeroXbarRound(ExecutionItem):
    zero_xbar_kind: str = ZERO_XBAR_KIND_HOOK_ONLY
    selected_entries: list[SelectedEntry] = field(default_factory=list)

    def __init__(
        self,
        item_id: str,
        item_index: int,
        zero_xbar_kind: str,
        selected_entries: list[SelectedEntry],
        local_graph_nodes: list[LocalGraphNode],
        external_inputs: list[BoundaryRef],
        same_item_edges: list[SameItemEdge],
        external_outputs: list[BoundaryRef],
        partial_output_contract: list[BoundaryRef],
        resource_proof: ResourceProof,
        backend_runtime: dict[str, Any] | None = None,
    ):
        super().__init__(
            item_id=item_id,
            item_index=item_index,
            item_kind=ITEM_KIND_ZERO_XBAR,
            local_graph_nodes=local_graph_nodes,
            external_inputs=external_inputs,
            same_item_edges=same_item_edges,
            external_outputs=external_outputs,
            partial_output_contract=partial_output_contract,
            resource_proof=resource_proof,
            backend_runtime={} if backend_runtime is None else backend_runtime,
        )
        self.zero_xbar_kind = zero_xbar_kind
        self.selected_entries = selected_entries


@dataclass
class RoundPlan(DictSerializable):
    plan_id: str
    model_id: str
    backend_kind: str
    hardware_budget: dict[str, Any]
    execution_items: list[ExecutionItem]
    layer_resource_descriptors: list[LayerResourceDescriptor]
    model_graph_ref: dict[str, Any]
    feasible: bool
    invalid_reason: str | None = None


def _prepass_ref_from_dict(data: dict[str, Any]) -> PrepassRef:
    return PrepassRef(**data)


def _selected_entry_from_dict(data: dict[str, Any]) -> SelectedEntry:
    copied = dict(data)
    copied["prepass_ref"] = _prepass_ref_from_dict(copied["prepass_ref"])
    return SelectedEntry(**copied)


def _local_node_from_dict(data: dict[str, Any]) -> LocalGraphNode:
    return LocalGraphNode(**data)


def _boundary_ref_from_dict(data: dict[str, Any]) -> BoundaryRef:
    return BoundaryRef(**data)


def _same_item_edge_from_dict(data: dict[str, Any]) -> SameItemEdge:
    return SameItemEdge(**data)


def _resource_proof_from_dict(data: dict[str, Any]) -> ResourceProof:
    return ResourceProof(**data)


def _descriptor_from_dict(data: dict[str, Any]) -> LayerResourceDescriptor:
    copied = dict(data)
    copied["prepass_ref"] = _prepass_ref_from_dict(copied["prepass_ref"])
    copied["completion_contract"] = CompletionContract(**copied.get("completion_contract", {}))
    return LayerResourceDescriptor(**copied)


def execution_item_from_dict(data: dict[str, Any]) -> ExecutionItem:
    common = {
        "item_id": data["item_id"],
        "item_index": data["item_index"],
        "local_graph_nodes": [_local_node_from_dict(v) for v in data["local_graph_nodes"]],
        "external_inputs": [_boundary_ref_from_dict(v) for v in data["external_inputs"]],
        "same_item_edges": [_same_item_edge_from_dict(v) for v in data["same_item_edges"]],
        "external_outputs": [_boundary_ref_from_dict(v) for v in data["external_outputs"]],
        "partial_output_contract": [_boundary_ref_from_dict(v) for v in data["partial_output_contract"]],
        "resource_proof": _resource_proof_from_dict(data["resource_proof"]),
        "backend_runtime": dict(data.get("backend_runtime", {})),
    }
    if data["item_kind"] == ITEM_KIND_WEIGHTED:
        return WeightedPhysicalRound(
            round_kind=data["round_kind"],
            selected_entries=[_selected_entry_from_dict(v) for v in data["selected_entries"]],
            **common,
        )
    if data["item_kind"] == ITEM_KIND_ZERO_XBAR:
        return ZeroXbarRound(
            zero_xbar_kind=data["zero_xbar_kind"],
            selected_entries=[_selected_entry_from_dict(v) for v in data["selected_entries"]],
            **common,
        )
    raise Phase1Invalid(INTERNAL_PLANNER_INVARIANT, f"unknown item_kind {data['item_kind']}")


def round_plan_from_dict(data: dict[str, Any]) -> RoundPlan:
    return RoundPlan(
        plan_id=data["plan_id"],
        model_id=data["model_id"],
        backend_kind=data["backend_kind"],
        hardware_budget=dict(data["hardware_budget"]),
        execution_items=[execution_item_from_dict(v) for v in data["execution_items"]],
        layer_resource_descriptors=[_descriptor_from_dict(v) for v in data["layer_resource_descriptors"]],
        model_graph_ref=dict(data["model_graph_ref"]),
        feasible=data["feasible"],
        invalid_reason=data.get("invalid_reason"),
    )
```

- [ ] **Step 4: Run the dataclass tests and verify they pass**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_types.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Interface/round_phase1_types.py tests/test_round_phase1_types.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase1 public schema"
```

---

### Task 2: Hardware Projection From SimConfig Path, ConfigParser, Or Dict

**Files:**
- Create: `tests/test_round_phase1_config.py`
- Create: `MNSIM/Interface/round_phase1_config.py`

- [ ] **Step 1: Write failing hardware projection tests**

Create `tests/test_round_phase1_config.py` with:

```python
import configparser

from MNSIM.Interface.round_phase1_config import build_hardware_projection
from MNSIM.Interface.round_phase1_types import Phase1Invalid


def _base_config():
    return {
        "Xbar_Size": [128, 128],
        "Tile_Num": [2, 2],
        "PE_Num": [2, 2],
        "Group_Num": 1,
        "Device_Level": 4,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }


def test_hardware_projection_computes_real_xbar_nums():
    hw = build_hardware_projection(_base_config())

    assert hw.xbar_size == [128, 128]
    assert hw.xbar_row == 128
    assert hw.xbar_col == 128
    assert hw.xbar_nums == 16
    assert hw.device_level == 4
    assert hw.weight_precision == 8
    assert hw.xbar_polarity == 2


def test_hardware_projection_reads_mnsim_simconfig_ini_path(tmp_path):
    config_path = tmp_path / "SimConfig.ini"
    config_path.write_text(
        """
[Device level]
Device_Level = 4

[Crossbar level]
Xbar_Size = 128,128

[Process element level]
Xbar_Polarity = 2
Group_Num = 1

[Tile level]
PE_Num = 2,2

[Architecture level]
Tile_Num = 2,2

[Algorithm Configuration]
Weight_precision = 8
""",
        encoding="utf-8",
    )

    hw = build_hardware_projection(config_path)

    assert hw.xbar_size == [128, 128]
    assert hw.xbar_nums == 16
    assert hw.weight_precision == 8


def test_hardware_projection_reads_configparser_object():
    parser = configparser.ConfigParser()
    parser.read_dict(
        {
            "Device level": {"Device_Level": "4"},
            "Crossbar level": {"Xbar_Size": "128,128"},
            "Process element level": {"Xbar_Polarity": "2", "Group_Num": "1"},
            "Tile level": {"PE_Num": "2,2"},
            "Architecture level": {"Tile_Num": "2,2"},
            "Algorithm Configuration": {"Weight_precision": "8"},
        }
    )

    hw = build_hardware_projection(parser)

    assert hw.xbar_nums == 16
    assert hw.device_level == 4


def test_precision_fields_do_not_change_xbar_nums():
    config = _base_config()
    config["Device_Level"] = 16
    config["Weight_precision"] = 4
    config["Xbar_Polarity"] = 1

    hw = build_hardware_projection(config)

    assert hw.xbar_nums == 16
    assert hw.device_level == 16
    assert hw.weight_precision == 4
    assert hw.xbar_polarity == 1


def test_invalid_hardware_projection_reports_phase1_invalid():
    config = _base_config()
    config["Tile_Num"] = [0, 2]

    try:
        build_hardware_projection(config)
    except Phase1Invalid as exc:
        assert exc.code == "INVALID_HARDWARE_PROJECTION"
        assert "xbar_nums" in exc.detail
    else:
        raise AssertionError("expected Phase1Invalid")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_config.py -v
```

Expected: fail during import with `ModuleNotFoundError: No module named 'MNSIM.Interface.round_phase1_config'`.

- [ ] **Step 3: Implement `round_phase1_config.py`**

Create `MNSIM/Interface/round_phase1_config.py` with:

```python
from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from MNSIM.Interface.round_phase1_types import (
    INVALID_HARDWARE_PROJECTION,
    INVALID_SIM_CONFIG,
    Phase1Invalid,
)


SimConfigInput = dict[str, Any] | str | Path | configparser.ConfigParser

SECTION_KEYS = {
    "Xbar_Size": ("Crossbar level", "Xbar_Size"),
    "Tile_Num": ("Architecture level", "Tile_Num"),
    "PE_Num": ("Tile level", "PE_Num"),
    "Group_Num": ("Process element level", "Group_Num"),
    "Device_Level": ("Device level", "Device_Level"),
    "Weight_precision": ("Algorithm Configuration", "Weight_precision"),
    "Xbar_Polarity": ("Process element level", "Xbar_Polarity"),
}


@dataclass
class HardwareProjection:
    xbar_size: list[int]
    xbar_row: int
    xbar_col: int
    xbar_nums: int
    device_level: int
    weight_precision: int
    xbar_polarity: int
    raw_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "xbar_size": list(self.xbar_size),
            "xbar_row": self.xbar_row,
            "xbar_col": self.xbar_col,
            "xbar_nums": self.xbar_nums,
            "device_level": self.device_level,
            "weight_precision": self.weight_precision,
            "xbar_polarity": self.xbar_polarity,
        }


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise Phase1Invalid(INVALID_SIM_CONFIG, f"missing required SimConfig field {key}")
    return config[key]


def _pair(config: dict[str, Any], key: str) -> list[int]:
    value = _required(config, key)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
        value = [int(part) for part in parts]
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise Phase1Invalid(INVALID_SIM_CONFIG, f"{key} must be a pair")
    return [int(value[0]), int(value[1])]


def _from_config_parser(parser: configparser.ConfigParser) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for public_key, (section, option) in SECTION_KEYS.items():
        if not parser.has_section(section):
            raise Phase1Invalid(INVALID_SIM_CONFIG, f"missing SimConfig section {section}")
        if not parser.has_option(section, option):
            raise Phase1Invalid(INVALID_SIM_CONFIG, f"missing required SimConfig field {section}.{option}")
        flattened[public_key] = parser.get(section, option)
    return flattened


def _read_config_path(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like)
    parser = configparser.ConfigParser()
    loaded = parser.read(path)
    if not loaded:
        raise Phase1Invalid(INVALID_SIM_CONFIG, f"could not read SimConfig path {path}")
    return _from_config_parser(parser)


def _normalize_config(sim_config: SimConfigInput) -> dict[str, Any]:
    if isinstance(sim_config, configparser.ConfigParser):
        return _from_config_parser(sim_config)
    if isinstance(sim_config, (str, Path)):
        return _read_config_path(sim_config)
    if isinstance(sim_config, dict):
        return dict(sim_config)
    raise Phase1Invalid(INVALID_SIM_CONFIG, f"unsupported sim_config input type {type(sim_config).__name__}")


def build_hardware_projection(sim_config: SimConfigInput) -> HardwareProjection:
    config = _normalize_config(sim_config)
    xbar_size = _pair(config, "Xbar_Size")
    tile_num = _pair(config, "Tile_Num")
    pe_num = _pair(config, "PE_Num")
    group_num = int(_required(config, "Group_Num"))
    device_level = int(_required(config, "Device_Level"))
    weight_precision = int(_required(config, "Weight_precision"))
    xbar_polarity = int(_required(config, "Xbar_Polarity"))

    xbar_row, xbar_col = xbar_size
    xbar_nums = tile_num[0] * tile_num[1] * pe_num[0] * pe_num[1] * group_num

    if xbar_row <= 0 or xbar_col <= 0:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, "xbar_size dimensions must be positive")
    if xbar_nums <= 0:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, f"xbar_nums must be positive, got {xbar_nums}")
    if device_level <= 1:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, "Device_Level must be greater than 1")
    if weight_precision <= 0:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, "Weight_precision must be positive")
    if xbar_polarity not in (1, 2):
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, f"unsupported Xbar_Polarity {xbar_polarity}")

    return HardwareProjection(
        xbar_size=xbar_size,
        xbar_row=xbar_row,
        xbar_col=xbar_col,
        xbar_nums=xbar_nums,
        device_level=device_level,
        weight_precision=weight_precision,
        xbar_polarity=xbar_polarity,
        raw_config=dict(config),
    )
```

- [ ] **Step 4: Run hardware projection tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_config.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Interface/round_phase1_config.py tests/test_round_phase1_config.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase1 hardware projection"
```

---

### Task 3: ResourcePrepass Formulas

**Files:**
- Create: `tests/test_round_phase1_resource_prepass.py`
- Create: `MNSIM/Interface/round_phase1_resource_prepass.py`

- [ ] **Step 1: Write failing ResourcePrepass tests**

Create `tests/test_round_phase1_resource_prepass.py` with:

```python
from MNSIM.Interface.round_phase1_config import build_hardware_projection
from MNSIM.Interface.round_phase1_resource_prepass import build_layer_resource_descriptors
from MNSIM.Interface.round_phase1_types import LayerIR, ModelIR, Phase1Invalid, SequenceSpec


def _hw_config(**overrides):
    config = {
        "Xbar_Size": [128, 128],
        "Tile_Num": [2, 2],
        "PE_Num": [2, 2],
        "Group_Num": 1,
        "Device_Level": 4,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }
    config.update(overrides)
    return config


def _model(layers):
    return ModelIR(
        model_id="m0",
        backend_kind="ann",
        sequence_spec=SequenceSpec(1, False, False, []),
        layers=layers,
    )


def _weighted(layer_id, layer_type, shape, index=0):
    return LayerIR(
        layer_id=layer_id,
        layer_type=layer_type,
        is_weighted=True,
        stable_topological_index=index,
        input_tensor_ids=["input"],
        output_tensor_ids=[f"{layer_id}:out"],
        input_layer_ids=[],
        output_consumer_ids=[],
        weight_ref_id=f"w:{layer_id}",
        shape=shape,
        backend_attrs={},
    )


def test_fc_resource_prepass_counts_spatial_precision_and_polarity():
    hw = build_hardware_projection(_hw_config())
    model = _model([_weighted("FC1", "fc", {"Infeature": 192, "Outfeature": 129})])

    [desc] = build_layer_resource_descriptors(model, hw)

    assert desc.layer_id == "FC1"
    assert desc.required_xbar_count == 2 * 2 * 4 * 2
    assert desc.prepass_ref.layer_id == "FC1"
    assert desc.prepass_ref.resource_version == "mnsim20_formula_v1"


def test_conv_resource_prepass_preserves_kernel_per_input_channel():
    hw = build_hardware_projection(_hw_config())
    model = _model([_weighted("Conv1", "conv", {"Inputchannel": 20, "Outputchannel": 130, "Kernelsize": 3})])

    [desc] = build_layer_resource_descriptors(model, hw)

    input_channels_per_xbar = 128 // (3 * 3)
    row_split_count = 2  # ceil(20 / 14)
    output_subset_count = 2  # ceil(130 / 128)
    assert input_channels_per_xbar == 14
    assert desc.required_xbar_count == row_split_count * output_subset_count * 4 * 2


def test_rnn_like_projection_uses_fc_formula():
    hw = build_hardware_projection(_hw_config())
    model = _model([_weighted("RNNProj", "fc_like_projection", {"Infeature": 192, "Outfeature": 128})])

    [desc] = build_layer_resource_descriptors(model, hw)

    assert desc.required_xbar_count == 2 * 1 * 4 * 2


def test_single_polarity_uses_full_weight_precision_bits():
    hw = build_hardware_projection(_hw_config(Xbar_Polarity=1))
    model = _model([_weighted("FC1", "fc", {"Infeature": 128, "Outfeature": 128})])

    [desc] = build_layer_resource_descriptors(model, hw)

    assert desc.required_xbar_count == 1 * 1 * 4 * 1


def test_conv_kernel_larger_than_xbar_row_is_invalid():
    hw = build_hardware_projection(_hw_config(Xbar_Size=[8, 128]))
    model = _model([_weighted("ConvBad", "conv", {"Inputchannel": 1, "Outputchannel": 1, "Kernelsize": 3})])

    try:
        build_layer_resource_descriptors(model, hw)
    except Phase1Invalid as exc:
        assert exc.code == "INVALID_LAYER_SHAPE"
        assert "Kernelsize" in exc.detail
    else:
        raise AssertionError("expected Phase1Invalid")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_resource_prepass.py -v
```

Expected: fail during import with `ModuleNotFoundError: No module named 'MNSIM.Interface.round_phase1_resource_prepass'`.

- [ ] **Step 3: Implement `round_phase1_resource_prepass.py`**

Create `MNSIM/Interface/round_phase1_resource_prepass.py` with these functions and constants:

```python
from __future__ import annotations

import math
from typing import Any

from MNSIM.Interface.round_phase1_config import HardwareProjection
from MNSIM.Interface.round_phase1_types import (
    CompletionContract,
    INVALID_LAYER_SHAPE,
    INVALID_RESOURCE_FOOTPRINT,
    LayerIR,
    LayerResourceDescriptor,
    ModelIR,
    Phase1Invalid,
    PrepassRef,
    UNSUPPORTED_LAYER_TYPE,
    WEIGHTED_LAYER_TYPES,
)

RESOURCE_VERSION = "mnsim20_formula_v1"


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise Phase1Invalid(INVALID_RESOURCE_FOOTPRINT, f"invalid denominator {denominator}")
    return (numerator + denominator - 1) // denominator


def _positive_int(shape: dict[str, Any], key: str, layer: LayerIR) -> int:
    if key not in shape:
        raise Phase1Invalid(INVALID_LAYER_SHAPE, f"{layer.layer_id} missing shape field {key}")
    value = int(shape[key])
    if value <= 0:
        raise Phase1Invalid(INVALID_LAYER_SHAPE, f"{layer.layer_id} shape field {key} must be positive")
    return value


def _weight_bit_split_part(hw: HardwareProjection) -> int:
    bits_per_cell = math.log2(hw.device_level)
    if hw.xbar_polarity == 2:
        return math.ceil((hw.weight_precision - 1) / bits_per_cell)
    return math.ceil(hw.weight_precision / bits_per_cell)


def _required_xbars(layer: LayerIR, hw: HardwareProjection) -> int:
    if layer.layer_type in ("fc", "fc_like_projection"):
        rows = _positive_int(layer.shape, "Infeature", layer)
        cols = _positive_int(layer.shape, "Outfeature", layer)
        row_split_count = _ceil_div(rows, hw.xbar_row)
        output_subset_count = _ceil_div(cols, hw.xbar_col)
    elif layer.layer_type == "conv":
        input_channels = _positive_int(layer.shape, "Inputchannel", layer)
        output_channels = _positive_int(layer.shape, "Outputchannel", layer)
        kernel = _positive_int(layer.shape, "Kernelsize", layer)
        kernel_area = kernel * kernel
        if kernel_area > hw.xbar_row:
            raise Phase1Invalid(
                INVALID_LAYER_SHAPE,
                f"{layer.layer_id} Kernelsize*Kernelsize={kernel_area} exceeds xbar_row={hw.xbar_row}",
            )
        input_channels_per_xbar = hw.xbar_row // kernel_area
        row_split_count = _ceil_div(input_channels, input_channels_per_xbar)
        output_subset_count = _ceil_div(output_channels, hw.xbar_col)
    else:
        raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"unsupported weighted layer type {layer.layer_type}")

    required = row_split_count * output_subset_count * _weight_bit_split_part(hw) * hw.xbar_polarity
    if required <= 0:
        raise Phase1Invalid(INVALID_RESOURCE_FOOTPRINT, f"{layer.layer_id} has non-positive footprint {required}")
    return int(required)


def _prepass_ref(model: ModelIR, layer: LayerIR, hw: HardwareProjection) -> PrepassRef:
    shape_signature = dict(layer.shape)
    sim_fingerprint = (
        f"xbar={hw.xbar_row}x{hw.xbar_col};"
        f"device={hw.device_level};weight={hw.weight_precision};polarity={hw.xbar_polarity}"
    )
    layer_fingerprint = f"{layer.layer_type}:{sorted(shape_signature.items())}"
    return PrepassRef(
        ref_id=f"{model.model_id}:{layer.layer_id}:{RESOURCE_VERSION}",
        model_id=model.model_id,
        layer_id=layer.layer_id,
        layer_type=layer.layer_type,
        resource_version=RESOURCE_VERSION,
        sim_config_fingerprint=sim_fingerprint,
        layer_config_fingerprint=layer_fingerprint,
        weight_ref_id=layer.weight_ref_id,
        shape_signature=shape_signature,
    )


def build_layer_resource_descriptors(
    model_ir: ModelIR,
    hardware: HardwareProjection,
) -> list[LayerResourceDescriptor]:
    descriptors: list[LayerResourceDescriptor] = []
    for layer in sorted(model_ir.layers, key=lambda node: node.stable_topological_index):
        if not layer.is_weighted:
            continue
        if layer.layer_type not in WEIGHTED_LAYER_TYPES:
            raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"unsupported weighted layer type {layer.layer_type}")
        descriptors.append(
            LayerResourceDescriptor(
                layer_id=layer.layer_id,
                layer_type=layer.layer_type,
                input_layer_ids=list(layer.input_layer_ids),
                output_consumer_ids=list(layer.output_consumer_ids),
                required_xbar_count=_required_xbars(layer, hardware),
                prepass_ref=_prepass_ref(model_ir, layer, hardware),
                completion_contract=CompletionContract(),
            )
        )
    return descriptors
```

- [ ] **Step 4: Run ResourcePrepass tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_resource_prepass.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run previous tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_types.py tests/test_round_phase1_config.py tests/test_round_phase1_resource_prepass.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Interface/round_phase1_resource_prepass.py tests/test_round_phase1_resource_prepass.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase1 resource prepass"
```

---

### Task 4: Planner Entry, Model Validation, And Invalid Plan Wrapper

**Files:**
- Create: `tests/test_round_phase1_planner.py`
- Create: `MNSIM/Interface/round_phase1_planner.py`

- [ ] **Step 1: Write failing invalid-plan tests**

Create `tests/test_round_phase1_planner.py` with this initial content:

```python
from MNSIM.Interface.round_phase1_planner import build_round_plan
from MNSIM.Interface.round_phase1_types import LayerIR, ModelIR, SequenceSpec


def _config(**overrides):
    config = {
        "Xbar_Size": [128, 128],
        "Tile_Num": [2, 2],
        "PE_Num": [2, 2],
        "Group_Num": 1,
        "Device_Level": 4,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }
    config.update(overrides)
    return config


def _seq(length=1, ordered=False, cross_step=False, states=None):
    return SequenceSpec(length, ordered, cross_step, [] if states is None else states)


def _layer(
    layer_id,
    layer_type,
    is_weighted,
    index,
    inputs=None,
    consumers=None,
    shape=None,
    weight_ref_id=None,
):
    return LayerIR(
        layer_id=layer_id,
        layer_type=layer_type,
        is_weighted=is_weighted,
        stable_topological_index=index,
        input_tensor_ids=[],
        output_tensor_ids=[f"{layer_id}:out"],
        input_layer_ids=[] if inputs is None else inputs,
        output_consumer_ids=[] if consumers is None else consumers,
        weight_ref_id=weight_ref_id,
        shape={} if shape is None else shape,
        backend_attrs={},
    )


def _model(layers, backend_kind="ann", seq=None):
    return ModelIR(
        model_id="m0",
        backend_kind=backend_kind,
        sequence_spec=_seq() if seq is None else seq,
        layers=layers,
    )


def test_invalid_unknown_layer_type_returns_invalid_plan():
    model = _model([
        _layer("Mystery", "custom_op", False, 0),
    ])

    plan = build_round_plan(model, _config())

    assert plan.feasible is False
    assert plan.execution_items == []
    assert plan.invalid_reason.startswith("UNSUPPORTED_LAYER_TYPE:")


def test_invalid_duplicate_layer_id_returns_invalid_plan():
    model = _model([
        _layer("A", "relu", False, 0),
        _layer("A", "relu", False, 1),
    ])

    plan = build_round_plan(model, _config())

    assert plan.feasible is False
    assert plan.invalid_reason.startswith("INVALID_MODEL_DAG:")


def test_invalid_duplicate_stable_index_returns_invalid_plan():
    model = _model([
        _layer("A", "relu", False, 0),
        _layer("B", "relu", False, 0),
    ])

    plan = build_round_plan(model, _config())

    assert plan.feasible is False
    assert plan.invalid_reason.startswith("INVALID_MODEL_DAG:")


def test_invalid_cycle_returns_invalid_plan():
    model = _model([
        _layer("A", "relu", False, 0, inputs=["B"], consumers=["B"]),
        _layer("B", "relu", False, 1, inputs=["A"], consumers=["A"]),
    ])

    plan = build_round_plan(model, _config())

    assert plan.feasible is False
    assert plan.invalid_reason.startswith("INVALID_MODEL_DAG:")
```

- [ ] **Step 2: Run the tests and verify they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_planner.py -v
```

Expected: fail during import with `ModuleNotFoundError: No module named 'MNSIM.Interface.round_phase1_planner'`.

- [ ] **Step 3: Implement planner shell and validation**

Create `MNSIM/Interface/round_phase1_planner.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from MNSIM.Interface.round_phase1_config import SimConfigInput, build_hardware_projection
from MNSIM.Interface.round_phase1_resource_prepass import build_layer_resource_descriptors
from MNSIM.Interface.round_phase1_types import (
    HOOK_LAYER_TYPES,
    INVALID_MODEL_DAG,
    INVALID_SEQUENCE_SPEC,
    ITEM_KIND_WEIGHTED,
    ITEM_KIND_ZERO_XBAR,
    LocalGraphNode,
    ModelIR,
    NON_WEIGHTED_LAYER_TYPES,
    NODE_ROLE_HOOK,
    NODE_ROLE_STRUCTURAL_NON_WEIGHTED,
    NODE_ROLE_WEIGHTED,
    Phase1Invalid,
    ResourceProof,
    RoundPlan,
    STRUCTURAL_NON_WEIGHTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    UNSUPPORTED_LAYER_TYPE,
    WEIGHTED_LAYER_TYPES,
)


@dataclass
class _PlannerState:
    complete_layers: set[str]
    emitted_layers: set[str]
    split_cursors: dict[str, int]


def _invalid_plan(model_ir: ModelIR, reason: str) -> RoundPlan:
    return RoundPlan(
        plan_id=f"plan:{model_ir.model_id}",
        model_id=model_ir.model_id,
        backend_kind=model_ir.backend_kind,
        hardware_budget={},
        execution_items=[],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": model_ir.model_id},
        feasible=False,
        invalid_reason=reason,
    )


def _validate_sequence_spec(model_ir: ModelIR) -> None:
    seq = model_ir.sequence_spec
    if seq.sequence_length <= 0:
        raise Phase1Invalid(INVALID_SEQUENCE_SPEC, "sequence_length must be positive")
    if seq.cross_step_dependency and not seq.ordered_steps:
        raise Phase1Invalid(INVALID_SEQUENCE_SPEC, "cross_step_dependency requires ordered_steps")


def _validate_model_ir(model_ir: ModelIR) -> None:
    _validate_sequence_spec(model_ir)
    seen_ids: set[str] = set()
    seen_indices: set[int] = set()
    by_id = {layer.layer_id: layer for layer in model_ir.layers}
    for layer in model_ir.layers:
        if layer.layer_id in seen_ids:
            raise Phase1Invalid(INVALID_MODEL_DAG, f"duplicate layer_id {layer.layer_id}")
        seen_ids.add(layer.layer_id)
        if layer.stable_topological_index in seen_indices:
            raise Phase1Invalid(INVALID_MODEL_DAG, f"duplicate stable_topological_index {layer.stable_topological_index}")
        seen_indices.add(layer.stable_topological_index)
        if layer.layer_type not in SUPPORTED_LAYER_TYPES:
            raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"unsupported layer type {layer.layer_type}")
        if layer.is_weighted and layer.layer_type not in WEIGHTED_LAYER_TYPES:
            raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"weighted layer {layer.layer_id} has type {layer.layer_type}")
        if not layer.is_weighted and layer.layer_type not in NON_WEIGHTED_LAYER_TYPES:
            raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"non-weighted layer {layer.layer_id} has type {layer.layer_type}")
        for producer_id in layer.input_layer_ids:
            if producer_id not in by_id:
                raise Phase1Invalid(INVALID_MODEL_DAG, f"{layer.layer_id} references missing producer {producer_id}")
            if by_id[producer_id].stable_topological_index >= layer.stable_topological_index:
                raise Phase1Invalid(INVALID_MODEL_DAG, f"dependency {producer_id}->{layer.layer_id} contradicts stable order")


def _empty_feasible_plan(model_ir: ModelIR, hardware: Any, descriptors: list[Any]) -> RoundPlan:
    return RoundPlan(
        plan_id=f"plan:{model_ir.model_id}",
        model_id=model_ir.model_id,
        backend_kind=model_ir.backend_kind,
        hardware_budget={"xbar_size": list(hardware.xbar_size), "xbar_nums": hardware.xbar_nums},
        execution_items=[],
        layer_resource_descriptors=descriptors,
        model_graph_ref={"model_id": model_ir.model_id},
        feasible=True,
        invalid_reason=None,
    )


def build_round_plan(model_ir: ModelIR, sim_config: SimConfigInput) -> RoundPlan:
    try:
        _validate_model_ir(model_ir)
        hardware = build_hardware_projection(sim_config)
        descriptors = build_layer_resource_descriptors(model_ir, hardware)
        return _empty_feasible_plan(model_ir, hardware, descriptors)
    except Phase1Invalid as exc:
        return _invalid_plan(model_ir, exc.reason)
```

- [ ] **Step 4: Run invalid-plan tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_planner.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Run all current tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_types.py tests/test_round_phase1_config.py tests/test_round_phase1_resource_prepass.py tests/test_round_phase1_planner.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Interface/round_phase1_planner.py tests/test_round_phase1_planner.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase1 planner validation entry"
```

---

### Task 5: WeightedPhysicalRound Full Pack And Split Planning

**Files:**
- Modify: `tests/test_round_phase1_planner.py`
- Modify: `MNSIM/Interface/round_phase1_planner.py`

- [ ] **Step 1: Add failing full-pack and split tests**

Append to `tests/test_round_phase1_planner.py`:

```python
def _fc(layer_id, index, infeature=128, outfeature=128, inputs=None, consumers=None):
    return _layer(
        layer_id,
        "fc",
        True,
        index,
        inputs=[] if inputs is None else inputs,
        consumers=[] if consumers is None else consumers,
        shape={"Infeature": infeature, "Outfeature": outfeature},
        weight_ref_id=f"w:{layer_id}",
    )


def test_full_layer_pack_groups_ready_weighted_layers_that_fit():
    model = _model([
        _fc("FC1", 0, outfeature=64, consumers=["FC2"]),
        _fc("FC2", 1, outfeature=64, inputs=["FC1"]),
    ])
    plan = build_round_plan(model, _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    assert len(plan.execution_items) == 1
    item = plan.execution_items[0]
    assert item.item_kind == "weighted_physical_round"
    assert item.round_kind == "full_layer_pack"
    assert [entry.layer_id for entry in item.selected_entries] == ["FC1", "FC2"]
    assert all(entry.layer_complete_in_round for entry in item.selected_entries)


def test_no_full_plus_partial_when_next_layer_does_not_fit():
    model = _model([
        _fc("Small", 0, infeature=128, outfeature=128, consumers=["Large"]),
        _fc("Large", 1, infeature=128, outfeature=1024, inputs=["Small"]),
    ])
    plan = build_round_plan(model, _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4))

    assert plan.feasible is True
    assert plan.execution_items[0].round_kind == "full_layer_pack"
    assert [entry.layer_id for entry in plan.execution_items[0].selected_entries] == ["Small"]
    assert plan.execution_items[1].round_kind == "single_layer_split"
    assert plan.execution_items[1].selected_entries[0].layer_id == "Large"


def test_single_layer_split_and_continuation_use_abstract_xbar_ranges():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=1024),
    ])
    plan = build_round_plan(model, _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4))

    assert plan.feasible is True
    assert [item.round_kind for item in plan.execution_items] == ["single_layer_split", "single_layer_continuation"]
    assert plan.execution_items[0].selected_entries[0].xbar_range_start == 0
    assert plan.execution_items[0].selected_entries[0].xbar_range_end == 4
    assert plan.execution_items[0].selected_entries[0].layer_complete_in_round is False
    assert plan.execution_items[1].selected_entries[0].xbar_range_start == 4
    assert plan.execution_items[1].selected_entries[0].xbar_range_end == 8
    assert plan.execution_items[1].selected_entries[0].layer_complete_in_round is True
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_planner.py::test_full_layer_pack_groups_ready_weighted_layers_that_fit tests/test_round_phase1_planner.py::test_no_full_plus_partial_when_next_layer_does_not_fit tests/test_round_phase1_planner.py::test_single_layer_split_and_continuation_use_abstract_xbar_ranges -v
```

Expected: fail because `build_round_plan` currently returns an empty feasible plan.

- [ ] **Step 3: Implement weighted planning helpers**

Modify `MNSIM/Interface/round_phase1_planner.py` to add:

```python
from MNSIM.Interface.round_phase1_types import (
    BoundaryRef,
    CompletionContract,
    LayerResourceDescriptor,
    ROUND_KIND_FULL_LAYER_PACK,
    ROUND_KIND_SINGLE_LAYER_CONTINUATION,
    ROUND_KIND_SINGLE_LAYER_SPLIT,
    SameItemEdge,
    SelectedEntry,
    WeightedPhysicalRound,
    ZeroXbarRound,
)


def _descriptor_map(descriptors: list[LayerResourceDescriptor]) -> dict[str, LayerResourceDescriptor]:
    return {desc.layer_id: desc for desc in descriptors}


def _layers_by_id(model_ir: ModelIR) -> dict[str, Any]:
    return {layer.layer_id: layer for layer in model_ir.layers}


def _is_ready(layer: Any, complete_layers: set[str]) -> bool:
    return all(input_id in complete_layers for input_id in layer.input_layer_ids)


def _weighted_item(
    item_index: int,
    round_kind: str,
    selected_entries: list[SelectedEntry],
    hardware_xbars: int,
) -> WeightedPhysicalRound:
    used = sum(entry.used_xbars for entry in selected_entries)
    return WeightedPhysicalRound(
        item_id=f"W{item_index}",
        item_index=item_index,
        round_kind=round_kind,
        selected_entries=selected_entries,
        local_graph_nodes=[
            LocalGraphNode(layer_id=entry.layer_id, layer_type=entry.layer_type, node_role=NODE_ROLE_WEIGHTED)
            for entry in selected_entries
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=hardware_xbars, used_xbars=used),
        backend_runtime={},
    )


def _selected_entry(
    desc: LayerResourceDescriptor,
    start: int,
    end: int,
    complete: bool,
) -> SelectedEntry:
    return SelectedEntry(
        layer_id=desc.layer_id,
        layer_type=desc.layer_type,
        xbar_range_start=start,
        xbar_range_end=end,
        used_xbars=end - start,
        layer_complete_in_round=complete,
        prepass_ref=desc.prepass_ref,
    )
```

Then replace `_empty_feasible_plan` usage in `build_round_plan` with a call to `_plan_execution_items(...)` and add:

```python
def _plan_execution_items(model_ir: ModelIR, hardware: Any, descriptors: list[LayerResourceDescriptor]) -> list[Any]:
    by_id = _layers_by_id(model_ir)
    desc_by_id = _descriptor_map(descriptors)
    unfinished = {layer.layer_id for layer in model_ir.layers}
    complete_layers: set[str] = set()
    emitted_weighted: set[str] = set()
    split_cursors: dict[str, int] = {}
    items: list[Any] = []

    while unfinished:
        ready_non_weighted = sorted(
            [
                by_id[layer_id]
                for layer_id in unfinished
                if not by_id[layer_id].is_weighted and _is_ready(by_id[layer_id], complete_layers)
            ],
            key=lambda layer: layer.stable_topological_index,
        )
        if ready_non_weighted:
            raise Phase1Invalid(INTERNAL_PLANNER_INVARIANT, "ZeroXbarRound planning is added in Task 6")

        ready_weighted = sorted(
            [
                by_id[layer_id]
                for layer_id in unfinished
                if by_id[layer_id].is_weighted and _is_ready(by_id[layer_id], complete_layers)
            ],
            key=lambda layer: layer.stable_topological_index,
        )
        if not ready_weighted:
            raise Phase1Invalid(INVALID_MODEL_DAG, "no ready node; graph may contain a cycle or unsupported dependency")

        first = ready_weighted[0]
        first_desc = desc_by_id[first.layer_id]
        cursor = split_cursors.get(first.layer_id, 0)
        remaining = first_desc.required_xbar_count - cursor
        if first_desc.required_xbar_count > hardware.xbar_nums or cursor > 0:
            take = min(hardware.xbar_nums, remaining)
            start = cursor
            end = cursor + take
            complete = end == first_desc.required_xbar_count
            round_kind = ROUND_KIND_SINGLE_LAYER_CONTINUATION if start > 0 else ROUND_KIND_SINGLE_LAYER_SPLIT
            items.append(_weighted_item(len(items), round_kind, [_selected_entry(first_desc, start, end, complete)], hardware.xbar_nums))
            if complete:
                complete_layers.add(first.layer_id)
                unfinished.remove(first.layer_id)
                split_cursors.pop(first.layer_id, None)
            else:
                split_cursors[first.layer_id] = end
            continue

        selected: list[SelectedEntry] = []
        used = 0
        while True:
            fit_nodes = [
                layer
                for layer in ready_weighted
                if layer.layer_id in unfinished
                and layer.layer_id not in {entry.layer_id for entry in selected}
                and used + desc_by_id[layer.layer_id].required_xbar_count <= hardware.xbar_nums
            ]
            if not fit_nodes:
                break
            layer = min(fit_nodes, key=lambda node: node.stable_topological_index)
            desc = desc_by_id[layer.layer_id]
            selected.append(_selected_entry(desc, 0, desc.required_xbar_count, True))
            used += desc.required_xbar_count
            complete_layers.add(layer.layer_id)
            unfinished.remove(layer.layer_id)
            ready_weighted = sorted(
                [
                    by_id[layer_id]
                    for layer_id in unfinished
                    if by_id[layer_id].is_weighted and _is_ready(by_id[layer_id], complete_layers)
                ],
                key=lambda node: node.stable_topological_index,
            )
        if not selected:
            raise Phase1Invalid(INTERNAL_PLANNER_INVARIANT, "planner failed to select a weighted node")
        items.append(_weighted_item(len(items), ROUND_KIND_FULL_LAYER_PACK, selected, hardware.xbar_nums))

    return items
```

Update `build_round_plan`:

```python
        execution_items = _plan_execution_items(model_ir, hardware, descriptors)
        plan = _empty_feasible_plan(model_ir, hardware, descriptors)
        plan.execution_items = execution_items
        return plan
```

- [ ] **Step 4: Run weighted planner tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_planner.py::test_full_layer_pack_groups_ready_weighted_layers_that_fit tests/test_round_phase1_planner.py::test_no_full_plus_partial_when_next_layer_does_not_fit tests/test_round_phase1_planner.py::test_single_layer_split_and_continuation_use_abstract_xbar_ranges -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run all current tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_types.py tests/test_round_phase1_config.py tests/test_round_phase1_resource_prepass.py tests/test_round_phase1_planner.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 5**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Interface/round_phase1_planner.py tests/test_round_phase1_planner.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add phase1 weighted round planning"
```

---

### Task 6: ZeroXbarRound Closure, DAG Merge, And RNN-like Cases

**Files:**
- Modify: `tests/test_round_phase1_planner.py`
- Modify: `MNSIM/Interface/round_phase1_planner.py`

- [ ] **Step 1: Add failing zero-xbar, DAG, and RNN tests**

Append to `tests/test_round_phase1_planner.py`:

```python
def _non_weighted(layer_id, layer_type, index, inputs=None, consumers=None):
    return _layer(
        layer_id,
        layer_type,
        False,
        index,
        inputs=[] if inputs is None else inputs,
        consumers=[] if consumers is None else consumers,
        shape={},
        weight_ref_id=None,
    )


def test_linear_cnn_inserts_zero_xbar_items_between_weighted_layers():
    layers = [
        _layer("Conv1", "conv", True, 0, consumers=["Relu1"], shape={"Inputchannel": 3, "Outputchannel": 8, "Kernelsize": 3}, weight_ref_id="w:Conv1"),
        _non_weighted("Relu1", "relu", 1, inputs=["Conv1"], consumers=["Conv2"]),
        _layer("Conv2", "conv", True, 2, inputs=["Relu1"], consumers=["Relu2"], shape={"Inputchannel": 8, "Outputchannel": 8, "Kernelsize": 3}, weight_ref_id="w:Conv2"),
        _non_weighted("Relu2", "relu", 3, inputs=["Conv2"], consumers=["FC"]),
        _fc("FC", 4, infeature=128, outfeature=8, inputs=["Relu2"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[4, 4], PE_Num=[2, 2]))

    assert plan.feasible is True
    assert [item.item_kind for item in plan.execution_items] == [
        "weighted_physical_round",
        "zero_xbar_round",
        "weighted_physical_round",
        "zero_xbar_round",
        "weighted_physical_round",
    ]
    assert [node.layer_id for node in plan.execution_items[1].local_graph_nodes] == ["Relu1"]


def test_zero_xbar_maximal_closure_groups_ready_non_weighted_chain():
    layers = [
        _fc("FC1", 0, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["FC1"], consumers=["BN1"]),
        _non_weighted("BN1", "bn", 2, inputs=["Relu1"], consumers=["View1"]),
        _non_weighted("View1", "view", 3, inputs=["BN1"], consumers=["FC2"]),
        _fc("FC2", 4, inputs=["View1"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    assert [node.layer_id for node in plan.execution_items[1].local_graph_nodes] == ["Relu1", "BN1", "View1"]
    assert plan.execution_items[1].zero_xbar_kind == "hook_only"


def test_branch_merge_requires_zero_xbar_merge_item():
    layers = [
        _fc("Conv0", 0, consumers=["ConvA", "ConvB"]),
        _fc("ConvA", 1, inputs=["Conv0"], consumers=["Sum"]),
        _fc("ConvB", 2, inputs=["Conv0"], consumers=["Sum"]),
        _non_weighted("Sum", "element_sum", 3, inputs=["ConvA", "ConvB"], consumers=["ConvC"]),
        _fc("ConvC", 4, inputs=["Sum"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    assert [item.item_kind for item in plan.execution_items] == [
        "weighted_physical_round",
        "weighted_physical_round",
        "zero_xbar_round",
        "weighted_physical_round",
    ]
    assert [entry.layer_id for entry in plan.execution_items[1].selected_entries] == ["ConvA", "ConvB"]
    assert [node.layer_id for node in plan.execution_items[2].local_graph_nodes] == ["Sum"]
    assert plan.execution_items[2].zero_xbar_kind == "structural_mnsim"


def test_rnn_like_projection_is_weighted_then_state_update_zero_xbar():
    seq = _seq(length=3, ordered=True, cross_step=True, states=["h"])
    layers = [
        _layer(
            "RNN_projection",
            "fc_like_projection",
            True,
            0,
            consumers=["RNN_state_update"],
            shape={"Infeature": 192, "Outfeature": 128},
            weight_ref_id="w:RNN",
        ),
        _non_weighted("RNN_state_update", "backend_hook", 1, inputs=["RNN_projection"]),
    ]
    plan = build_round_plan(_model(layers, backend_kind="rnn_like", seq=seq), _config())

    assert plan.feasible is True
    assert [item.item_kind for item in plan.execution_items] == ["weighted_physical_round", "zero_xbar_round"]
    assert plan.execution_items[1].zero_xbar_kind == "hook_only"


def test_rnn_like_split_projection_delays_state_update_until_tail_complete():
    seq = _seq(length=3, ordered=True, cross_step=True, states=["h"])
    layers = [
        _layer(
            "RNN_projection",
            "fc_like_projection",
            True,
            0,
            consumers=["RNN_state_update"],
            shape={"Infeature": 128, "Outfeature": 1024},
            weight_ref_id="w:RNN",
        ),
        _non_weighted("RNN_state_update", "backend_hook", 1, inputs=["RNN_projection"]),
    ]
    plan = build_round_plan(
        _model(layers, backend_kind="rnn_like", seq=seq),
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    assert [item.item_kind for item in plan.execution_items] == [
        "weighted_physical_round",
        "weighted_physical_round",
        "zero_xbar_round",
    ]
    assert [item.round_kind for item in plan.execution_items[:2]] == ["single_layer_split", "single_layer_continuation"]
    assert plan.execution_items[2].local_graph_nodes[0].layer_id == "RNN_state_update"
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_planner.py::test_linear_cnn_inserts_zero_xbar_items_between_weighted_layers tests/test_round_phase1_planner.py::test_zero_xbar_maximal_closure_groups_ready_non_weighted_chain tests/test_round_phase1_planner.py::test_branch_merge_requires_zero_xbar_merge_item tests/test_round_phase1_planner.py::test_rnn_like_projection_is_weighted_then_state_update_zero_xbar tests/test_round_phase1_planner.py::test_rnn_like_split_projection_delays_state_update_until_tail_complete -v
```

Expected: fail because zero-xbar planning still raises `INTERNAL_PLANNER_INVARIANT`.

- [ ] **Step 3: Implement zero-xbar planning helpers**

Modify `MNSIM/Interface/round_phase1_planner.py` to add:

```python
from MNSIM.Interface.round_phase1_types import (
    ZERO_XBAR_KIND_HOOK_ONLY,
    ZERO_XBAR_KIND_MIXED_NON_WEIGHTED,
    ZERO_XBAR_KIND_STRUCTURAL_MNSIM,
)


def _node_role_for_non_weighted(layer: Any) -> str:
    if layer.layer_type in HOOK_LAYER_TYPES:
        return NODE_ROLE_HOOK
    if layer.layer_type in STRUCTURAL_NON_WEIGHTED_LAYER_TYPES:
        return NODE_ROLE_STRUCTURAL_NON_WEIGHTED
    raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"unsupported non-weighted layer type {layer.layer_type}")


def _zero_xbar_kind(nodes: list[Any]) -> str:
    roles = {_node_role_for_non_weighted(node) for node in nodes}
    if roles == {NODE_ROLE_HOOK}:
        return ZERO_XBAR_KIND_HOOK_ONLY
    if roles == {NODE_ROLE_STRUCTURAL_NON_WEIGHTED}:
        return ZERO_XBAR_KIND_STRUCTURAL_MNSIM
    return ZERO_XBAR_KIND_MIXED_NON_WEIGHTED


def _zero_xbar_item(item_index: int, nodes: list[Any], hardware_xbars: int) -> ZeroXbarRound:
    return ZeroXbarRound(
        item_id=f"Z{item_index}",
        item_index=item_index,
        zero_xbar_kind=_zero_xbar_kind(nodes),
        selected_entries=[],
        local_graph_nodes=[
            LocalGraphNode(layer_id=node.layer_id, layer_type=node.layer_type, node_role=_node_role_for_non_weighted(node))
            for node in nodes
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=hardware_xbars, used_xbars=0),
        backend_runtime={},
    )


def _consume_zero_xbar_closure(
    item_index: int,
    by_id: dict[str, Any],
    unfinished: set[str],
    complete_layers: set[str],
    hardware_xbars: int,
) -> ZeroXbarRound:
    nodes: list[Any] = []
    while True:
        ready = sorted(
            [
                by_id[layer_id]
                for layer_id in unfinished
                if not by_id[layer_id].is_weighted and _is_ready(by_id[layer_id], complete_layers)
            ],
            key=lambda layer: layer.stable_topological_index,
        )
        if not ready:
            break
        node = ready[0]
        nodes.append(node)
        complete_layers.add(node.layer_id)
        unfinished.remove(node.layer_id)
    return _zero_xbar_item(item_index, nodes, hardware_xbars)
```

Then replace the zero-xbar branch in `_plan_execution_items`:

```python
        if ready_non_weighted:
            items.append(_consume_zero_xbar_closure(len(items), by_id, unfinished, complete_layers, hardware.xbar_nums))
            continue
```

- [ ] **Step 4: Run zero-xbar, DAG, and RNN tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_planner.py::test_linear_cnn_inserts_zero_xbar_items_between_weighted_layers tests/test_round_phase1_planner.py::test_zero_xbar_maximal_closure_groups_ready_non_weighted_chain tests/test_round_phase1_planner.py::test_branch_merge_requires_zero_xbar_merge_item tests/test_round_phase1_planner.py::test_rnn_like_projection_is_weighted_then_state_update_zero_xbar tests/test_round_phase1_planner.py::test_rnn_like_split_projection_delays_state_update_until_tail_complete -v
```

Expected: all selected tests pass.

- [ ] **Step 5: Run all tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 6**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Interface/round_phase1_planner.py tests/test_round_phase1_planner.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "feat: add zero-xbar and dag phase1 planning"
```

---

### Task 7: Final Acceptance Tests And Invariant Checks

**Files:**
- Modify: `tests/test_round_phase1_planner.py`
- Modify: `MNSIM/Interface/round_phase1_planner.py`

- [ ] **Step 1: Add final invariant tests**

Append to `tests/test_round_phase1_planner.py`:

```python
def test_weighted_rounds_never_contain_non_weighted_local_nodes():
    layers = [
        _fc("FC1", 0, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["FC1"], consumers=["FC2"]),
        _fc("FC2", 2, inputs=["Relu1"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    for item in plan.execution_items:
        if item.item_kind == "weighted_physical_round":
            assert all(node.node_role == "weighted" for node in item.local_graph_nodes)


def test_invalid_conv_shape_is_returned_as_invalid_plan():
    model = _model([
        _layer("ConvBad", "conv", True, 0, shape={"Inputchannel": 1, "Outputchannel": 1, "Kernelsize": 9}, weight_ref_id="w:bad"),
    ])

    plan = build_round_plan(model, _config(Xbar_Size=[8, 128]))

    assert plan.feasible is False
    assert plan.invalid_reason.startswith("INVALID_LAYER_SHAPE:")
    assert plan.execution_items == []


def test_round_plan_to_dict_contains_no_runtime_objects():
    model = _model([
        _fc("FC1", 0, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["FC1"]),
    ])
    plan = build_round_plan(model, _config(Xbar_Polarity=1, Weight_precision=2))

    data = plan.to_dict()

    assert data["feasible"] is True
    assert isinstance(data["execution_items"], list)
    assert data["execution_items"][0]["selected_entries"][0]["prepass_ref"]["weight_ref_id"] == "w:FC1"
    assert "NetStruct" not in repr(data)
    assert "mapping_result" not in repr(data)
```

- [ ] **Step 2: Run final invariant tests**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests/test_round_phase1_planner.py::test_weighted_rounds_never_contain_non_weighted_local_nodes tests/test_round_phase1_planner.py::test_invalid_conv_shape_is_returned_as_invalid_plan tests/test_round_phase1_planner.py::test_round_plan_to_dict_contains_no_runtime_objects -v
```

Expected: all selected tests pass.

- [ ] **Step 3: Run the full Phase 1 suite**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit Task 7**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add MNSIM/Interface/round_phase1_planner.py tests/test_round_phase1_planner.py
/home/xzh/micromamba/envs/workspace/bin/git commit -m "test: cover phase1 planner invariants"
```

---

### Task 8: Final Verification And Documentation Link

**Files:**
- Modify: `docs/superpowers/specs/2026-07-20-phase1-modelir-round-planner-design.md`

- [ ] **Step 1: Add implementation status note to the spec**

Append this section to `docs/superpowers/specs/2026-07-20-phase1-modelir-round-planner-design.md` after the acceptance criteria:

```markdown
## 20. Implementation Plan

Implementation is tracked in:

`docs/superpowers/plans/2026-07-21-phase1-modelir-round-planner-implementation.md`
```

- [ ] **Step 2: Run the full test suite**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 3: Check git status**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git status --short
```

Expected: only the spec file is modified.

- [ ] **Step 4: Commit Task 8**

```bash
/home/xzh/micromamba/envs/workspace/bin/git add docs/superpowers/specs/2026-07-20-phase1-modelir-round-planner-design.md
/home/xzh/micromamba/envs/workspace/bin/git commit -m "docs: link phase1 implementation plan"
```

- [ ] **Step 5: Final status check**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/git status --short
/home/xzh/micromamba/envs/workspace/bin/git log --oneline -5
```

Expected:

```text
# status has no output
<latest commit> docs: link phase1 implementation plan
<previous commit> test: cover phase1 planner invariants
<previous commit> feat: add zero-xbar and dag phase1 planning
<previous commit> feat: add phase1 weighted round planning
<previous commit> feat: add phase1 planner validation entry
```

## Self-Review

Spec coverage:

```text
Independent ModelIR entry: Task 4
Dataclass schema and deterministic serialization: Task 1 and Task 7
HardwareProjection xbar_nums and SimConfig path parsing: Task 2
ResourcePrepass formulas: Task 3
Weighted full-pack and split policy: Task 5
ZeroXbarRound closure: Task 6
DAG branch/merge: Task 6
RNN-like projection and state hook: Task 6
Invalid plan behavior: Task 4 and Task 7
No TCG / Model_* dependency: all tests use synthetic ModelIR and pytest only
```

Placeholder scan:

```text
The plan contains no unresolved implementation placeholders. Every task lists exact files, exact commands, expected outcomes, and concrete code snippets for tests and core implementation.
```

Type consistency:

```text
Public names are consistent across tasks:
build_round_plan
build_hardware_projection
build_layer_resource_descriptors
RoundPlan
WeightedPhysicalRound
ZeroXbarRound
LayerResourceDescriptor
PrepassRef
```
