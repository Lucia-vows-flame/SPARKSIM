"""Public Phase 2 execution schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from MNSIM.Scheduling.round_types import (
    SerializableDataclass,
    WeightedPhysicalRound,
    ZeroXbarRound,
)


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
SOURCE_KIND_ROUND_FINALIZE = "round_finalize"

PHYSICAL_XBAR_UNIT_MISMATCH = "PHYSICAL_XBAR_UNIT_MISMATCH"
PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH = "PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH"
INVALID_PHYSICAL_XBAR_RANGE = "INVALID_PHYSICAL_XBAR_RANGE"
SELECTED_FRAGMENT_COUNT_MISMATCH = "SELECTED_FRAGMENT_COUNT_MISMATCH"
TCG_ACTIVE_XBAR_PROOF_MISMATCH = "TCG_ACTIVE_XBAR_PROOF_MISMATCH"
DUPLICATE_FRAGMENT_CONTRIBUTION = "DUPLICATE_FRAGMENT_CONTRIBUTION"


class Phase2Invalid(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(self.reason)

    @property
    def reason(self) -> str:
        return f"{self.code}: {self.detail}"


def _deep_freeze_json_value(value: Any) -> Any:
    if isinstance(value, _FrozenSerializableMapping):
        return value
    if isinstance(value, Mapping):
        return _FrozenSerializableMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze_json_value(item) for item in value)
    return value


class _FrozenSerializableMapping(Mapping[str, Any]):
    """An immutable mapping backed only by recursively immutable values."""

    __slots__ = ("_items",)

    def __init__(self, values: Mapping[str, Any]):
        object.__setattr__(
            self,
            "_items",
            tuple(
                (key, _deep_freeze_json_value(value))
                for key, value in values.items()
            ),
        )

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise TypeError("native_layer_info is immutable")

    def __getitem__(self, key: str) -> Any:
        for item_key, value in self._items:
            if item_key == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (key for key, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __deepcopy__(self, _memo: dict[int, Any]) -> "_FrozenSerializableMapping":
        return self


@dataclass(frozen=True)
class MnsimGroupFragment(SerializableDataclass):
    fragment_id: str
    layer_id: str
    layer_type: str
    input_range: tuple[int, int]
    output_range: tuple[int, int]
    full_input_range: tuple[int, int]
    full_output_range: tuple[int, int]
    weight_plane_index: int
    polarity_count: int
    physical_xbar_count: int
    input_block_index: int = 0


@dataclass(frozen=True)
class MnsimFragmentLayout(SerializableDataclass):
    layer_id: str
    layer_type: str
    fragments: tuple[MnsimGroupFragment, ...]
    native_layer_info: Mapping[str, Any]
    active_group_count: int
    active_physical_xbar_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "native_layer_info",
            _FrozenSerializableMapping(self.native_layer_info),
        )


@dataclass(frozen=True)
class FragmentInventory(SerializableDataclass):
    layer_id: str
    required_xbar_count: int
    fragments: tuple[MnsimGroupFragment, ...]


@dataclass
class MetricTotals(SerializableDataclass):
    latency_s: float
    energy_j: float
    power_w: float
    area_um2: float


@dataclass
class BoundaryBinding(SerializableDataclass):
    local_layer_id: int | None
    hook_node_id: str | None
    origin_layer_id: str
    origin_layer_type: str
    report_role: str
    tensor_role: str
    shape: dict[str, Any]
    region: dict[str, Any]
    input_range: dict[str, Any]
    output_range: dict[str, Any]
    full_input_range: dict[str, Any]
    full_output_range: dict[str, Any]
    contribution_kind: str
    source_state_ids: list[str]
    downstream_consumer_ids: list[str]
    source_local_state_ids: list[str] = field(default_factory=list)
    fragment_ids: list[str] = field(default_factory=list)
    required_fragment_ids: list[str] = field(default_factory=list)
    sample_id: str = "sample:0"
    timestep: int = 0
    sequence_length: int = 1
    data_bitwidth: int | None = None
    accumulator_domain: dict[str, Any] = field(default_factory=dict)
    payload_kind: str = "activation"
    numerical_input_payload_identity: str | None = None
    state_payload_identity: str | None = None
    spike_payload_identity: str | None = None
    spike_codes: list[int] | None = None


@dataclass
class StaticEvalFrame(SerializableDataclass):
    context_layer_ids: list[int]
    reportable_layer_ids: list[int]
    latency_offset_source_ids: list[int]
    boundary_bindings: list[BoundaryBinding]


@dataclass
class StaticEvalRequest(SerializableDataclass):
    item_id: str
    item_kind: str
    netstruct: list[Any]
    sim_config_path: str
    multiple: list[int]
    latency_api: str
    latency_fill_mode: int
    frame: StaticEvalFrame
    fragment_layout: MnsimFragmentLayout | None = None


@dataclass
class StaticLayerResult(SerializableDataclass):
    local_layer_id: int
    origin_layer_id: str
    layer_type: str
    report_role: str
    latency_s: float
    begin_time_s: float
    finish_time_s: float
    area_um2: float
    power_w: float
    energy_j: float
    tcg_mapping: dict[str, Any]


@dataclass
class StaticEvalResult(SerializableDataclass):
    item_id: str
    item_kind: str
    status: str
    layer_results: list[StaticLayerResult]
    item_totals: MetricTotals
    mapping_summary: dict[str, Any]
    raw_mnsim_summary: dict[str, Any]
    boundary_bindings: list[BoundaryBinding]
    warnings: list[str]


@dataclass
class HookResult(SerializableDataclass):
    item_id: str
    item_kind: str
    hook_bindings: list[BoundaryBinding]
    warnings: list[str]


@dataclass
class BoundaryStateRecord(SerializableDataclass):
    state_id: str
    producer_item_id: str
    producer_layer_id: str
    tensor_role: str
    shape: dict[str, Any]
    region: dict[str, Any]
    status: str
    storage_backend: str
    allowed_consumer_item_ids: list[str]
    lifetime_policy: str
    source_state_ids: list[str]
    sample_id: str = "sample:0"
    timestep: int = 0
    sequence_length: int = 1
    data_bitwidth: int | None = None
    accumulator_domain: dict[str, Any] = field(default_factory=dict)
    payload_kind: str = "activation"
    numerical_input_payload_identity: str | None = None
    state_payload_identity: str | None = None
    spike_payload_identity: str | None = None
    spike_codes: list[int] | None = None


@dataclass
class BoundaryAccessEvent(SerializableDataclass):
    event_kind: str
    item_id: str
    state_id: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class BoundaryInputView(SerializableDataclass):
    input_view_id: str
    item_id: str
    readable_state_ids: list[str]
    state_bindings: dict[str, list[str]]
    warnings: list[str]
    state_records: list[BoundaryStateRecord] = field(default_factory=list)
    events: list[BoundaryAccessEvent] = field(default_factory=list)


@dataclass
class StepInputView(SerializableDataclass):
    item_id: str
    step_id: str
    boundary_input_view_id: str
    boundary_state_ids: list[str]
    boundary_state_records: list[BoundaryStateRecord]
    local_state_ids: list[str]
    local_state_records: list["RoundLocalStateRecord"]


@dataclass
class RoundLocalStateRecord(SerializableDataclass):
    local_state_id: str
    origin_layer_id: str
    origin_layer_type: str
    shape: dict[str, Any]
    contribution_kind: str
    boundary_source_state_ids: list[str]


@dataclass
class BoundaryCommitRequest(SerializableDataclass):
    item_id: str
    item_kind: str
    execution_index: int
    input_view_id: str
    source_kind: str
    static_eval_result: StaticEvalResult | None
    hook_result: HookResult | None
    boundary_bindings: list[BoundaryBinding]
    step_results: list[StepExecutionResult] = field(default_factory=list)
    input_events: list[BoundaryAccessEvent] = field(default_factory=list)
    sample_id: str = "sample:0"
    timestep: int = 0


@dataclass
class StepExecutionResult(SerializableDataclass):
    item_id: str
    step_id: str
    segment_index: int
    step_kind: str
    source_kind: str
    status: str
    origin_layer_ids: list[str]
    metrics: MetricTotals
    layer_results: list[StaticLayerResult]
    mapping_summary: dict[str, Any]
    raw_mnsim_audit: list[dict[str, Any]]
    boundary_bindings: list[BoundaryBinding]
    warnings: list[str]
    sample_id: str = "sample:0"
    timestep: int = 0


@dataclass
class BoundaryCommitResult(SerializableDataclass):
    item_id: str
    item_kind: str
    status: str
    created_states: list[BoundaryStateRecord]
    updated_states: list[BoundaryStateRecord]
    released_states: list[BoundaryStateRecord]
    events: list[BoundaryAccessEvent]
    completion_updates: list[dict[str, Any]]
    visibility_updates: list[dict[str, Any]]
    output_state_bindings: list[dict[str, Any]]
    warnings: list[str]
    ddr_transfer: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoweredItem(SerializableDataclass):
    item_id: str
    item_kind: str
    execution_index: int
    input_view: BoundaryInputView
    static_eval_request: StaticEvalRequest | None
    hook_result: HookResult | None
    boundary_bindings: list[BoundaryBinding]
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreparedRound:
    item: WeightedPhysicalRound | ZeroXbarRound
    boundary_input_view: BoundaryInputView
    placement: "RoundPlacementPlan | None"


@dataclass
class LoweredStep:
    item_id: str
    step_id: str
    segment_index: int
    step_kind: str
    static_eval_requests: list[StaticEvalRequest]
    placement_slices: list[Any]
    hook_result: HookResult | None
    boundary_bindings: list[BoundaryBinding]
    warnings: list[str]
    digital_lif_request: Any | None = None


@dataclass
class Phase2Report(SerializableDataclass):
    model_name: str
    sim_config_path: str
    feasible: bool
    execution_item_count: int
    inference_metrics: dict[str, Any]
    item_results: list[dict[str, Any]]
    step_results: list[dict[str, Any]]
    raw_mnsim_audit: list[dict[str, Any]]
    boundary_events: list[BoundaryAccessEvent]
    warnings: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)


__all__ = [
    "BOUNDARY_STATUS_MODEL_INPUT",
    "BOUNDARY_STATUS_PARTIAL_SUM",
    "BOUNDARY_STATUS_OUTPUT_SUBSET",
    "BOUNDARY_STATUS_COMPLETE_OUTPUT",
    "BOUNDARY_STATUS_HOOK_COMPLETE",
    "BOUNDARY_STATUS_RELEASED",
    "CONTRIBUTION_PARTIAL",
    "CONTRIBUTION_COMPLETE_OUTPUT",
    "CONTRIBUTION_STRUCTURAL_OUTPUT",
    "CONTRIBUTION_HOOK_OUTPUT",
    "REPORT_ROLE_REPORTABLE",
    "REPORT_ROLE_CONTEXT_ONLY",
    "STATIC_STATUS_OK",
    "SOURCE_KIND_STATIC_EVAL",
    "SOURCE_KIND_HOOK",
    "SOURCE_KIND_ROUND_FINALIZE",
    "PHYSICAL_XBAR_UNIT_MISMATCH",
    "PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH",
    "INVALID_PHYSICAL_XBAR_RANGE",
    "SELECTED_FRAGMENT_COUNT_MISMATCH",
    "TCG_ACTIVE_XBAR_PROOF_MISMATCH",
    "DUPLICATE_FRAGMENT_CONTRIBUTION",
    "Phase2Invalid",
    "MnsimGroupFragment",
    "MnsimFragmentLayout",
    "FragmentInventory",
    "MetricTotals",
    "BoundaryBinding",
    "StaticEvalFrame",
    "StaticEvalRequest",
    "StaticLayerResult",
    "StaticEvalResult",
    "HookResult",
    "BoundaryStateRecord",
    "BoundaryAccessEvent",
    "BoundaryInputView",
    "StepInputView",
    "RoundLocalStateRecord",
    "BoundaryCommitRequest",
    "StepExecutionResult",
    "BoundaryCommitResult",
    "LoweredItem",
    "PreparedRound",
    "LoweredStep",
    "Phase2Report",
]
