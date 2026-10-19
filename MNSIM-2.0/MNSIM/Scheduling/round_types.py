"""Public Phase 1 round-plan schema and deterministic serialization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from MNSIM.Scheduling.non_weighted_layers import REGISTERED_PHASE1_HOOK_LAYER_TYPES


ITEM_KIND_WEIGHTED = "weighted_physical_round"
ITEM_KIND_ZERO_XBAR = "zero_xbar_round"

SEGMENT_KIND_WEIGHTED_LAYER = "weighted_layer_segment"
SEGMENT_KIND_ZERO_XBAR = "zero_xbar_segment"

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
BASE_HOOK_LAYER_TYPES = {"relu", "bn", "view", "reshape", "flatten", "dropout", "backend_hook"}
HOOK_LAYER_TYPES = BASE_HOOK_LAYER_TYPES | REGISTERED_PHASE1_HOOK_LAYER_TYPES
STRUCTURAL_NON_WEIGHTED_LAYER_TYPES = {"pooling", "concat", "element_sum", "element_multiply"}
NON_WEIGHTED_LAYER_TYPES = HOOK_LAYER_TYPES | STRUCTURAL_NON_WEIGHTED_LAYER_TYPES
SUPPORTED_LAYER_TYPES = WEIGHTED_LAYER_TYPES | NON_WEIGHTED_LAYER_TYPES

INVALID_SIM_CONFIG = "INVALID_SIM_CONFIG"
INVALID_HARDWARE_PROJECTION = "INVALID_HARDWARE_PROJECTION"
UNSUPPORTED_LAYER_TYPE = "UNSUPPORTED_LAYER_TYPE"
INVALID_LAYER_SHAPE = "INVALID_LAYER_SHAPE"
INVALID_MODEL_DAG = "INVALID_MODEL_DAG"
INVALID_SEQUENCE_SPEC = "INVALID_SEQUENCE_SPEC"
INVALID_LIF_DOMAIN = "INVALID_LIF_DOMAIN"
INVALID_RESOURCE_FOOTPRINT = "INVALID_RESOURCE_FOOTPRINT"
INFEASIBLE_LAYER_ATOMIC_SCOPE = "INFEASIBLE_LAYER_ATOMIC_SCOPE"
INTERNAL_PLANNER_INVARIANT = "INTERNAL_PLANNER_INVARIANT"


class Phase1Invalid(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(self.reason)

    @property
    def reason(self) -> str:
        return f"{self.code}: {self.detail}"


def _to_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            dataclass_field.name: _to_json_value(getattr(value, dataclass_field.name))
            for dataclass_field in fields(value)
            if not (
                dataclass_field.metadata.get("omit_empty", False)
                and not getattr(value, dataclass_field.name)
            )
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            _to_json_key(key): _to_json_value(value[key])
            for key in sorted(value.keys(), key=lambda item: str(item))
        }
    return value


def _to_json_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    return str(key)


class SerializableDataclass:
    def to_dict(self) -> dict[str, Any]:
        return _to_json_value(self)


@dataclass
class SequenceSpec(SerializableDataclass):
    sequence_length: int
    ordered_steps: bool
    cross_step_dependency: bool
    recurrent_state_ids: list[str]


@dataclass
class LayerIR(SerializableDataclass):
    layer_id: str
    layer_type: str
    is_weighted: bool
    stable_topological_index: int
    input_tensor_ids: list[str]
    output_tensor_ids: list[str]
    input_layer_ids: list[str]
    output_consumer_ids: list[str]
    weight_ref_id: str | None
    shape: dict[str, Any]
    backend_attrs: dict[str, Any]


@dataclass
class ModelIR(SerializableDataclass):
    model_id: str
    backend_kind: str
    sequence_spec: SequenceSpec
    layers: list[LayerIR]


@dataclass
class PrepassRef(SerializableDataclass):
    ref_id: str
    model_id: str
    layer_id: str
    layer_type: str
    resource_version: str
    sim_config_fingerprint: str
    layer_config_fingerprint: str
    weight_ref_id: str | None
    shape_signature: dict[str, Any]


@dataclass
class CompletionContract(SerializableDataclass):
    partial_visibility: str = "internal_only"
    downstream_visibility: str = "complete_only"
    same_item_handoff_allowed: bool = True
    completion_hook_trigger: str = "on_complete"


@dataclass
class LayerResourceDescriptor(SerializableDataclass):
    layer_id: str
    layer_type: str
    input_layer_ids: list[str]
    output_consumer_ids: list[str]
    required_xbar_count: int
    prepass_ref: PrepassRef
    completion_contract: CompletionContract = field(default_factory=CompletionContract)


@dataclass
class SelectedEntry(SerializableDataclass):
    layer_id: str
    layer_type: str
    xbar_range_start: int
    xbar_range_end: int
    used_xbars: int
    layer_complete_in_round: bool
    prepass_ref: PrepassRef


@dataclass
class LocalGraphNode(SerializableDataclass):
    layer_id: str
    layer_type: str
    node_role: str
    node_shape: dict[str, Any] = field(default_factory=dict)
    digital_metadata: dict[str, Any] = field(
        default_factory=dict,
        metadata={"omit_empty": True},
    )


@dataclass
class WeightedLayerSegment(SerializableDataclass):
    segment_index: int
    selected_entry: SelectedEntry
    local_graph_node: LocalGraphNode
    segment_kind: str = field(init=False, default=SEGMENT_KIND_WEIGHTED_LAYER)


@dataclass
class ZeroXbarSegment(SerializableDataclass):
    segment_index: int
    zero_xbar_kind: str
    local_graph_nodes: list[LocalGraphNode]
    segment_kind: str = field(init=False, default=SEGMENT_KIND_ZERO_XBAR)


RoundSegment = WeightedLayerSegment | ZeroXbarSegment


@dataclass
class BoundaryRef(SerializableDataclass):
    state_id: str
    tensor_role: str
    producer_item_id: str | None = None
    consumer_item_ids: list[str] = field(default_factory=list)
    consumer_node_ids: list[str] = field(default_factory=list)
    partial_or_complete: str = "complete"


@dataclass
class SameItemEdge(SerializableDataclass):
    producer_node_id: str
    consumer_node_id: str


@dataclass
class ResourceProof(SerializableDataclass):
    xbar_nums: int
    used_xbars: int


@dataclass
class ExecutionItem(SerializableDataclass):
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
class WeightedPhysicalRound(SerializableDataclass):
    item_id: str
    item_index: int
    round_kind: str
    selected_entries: list[SelectedEntry]
    local_graph_nodes: list[LocalGraphNode]
    external_inputs: list[BoundaryRef]
    same_item_edges: list[SameItemEdge]
    external_outputs: list[BoundaryRef]
    partial_output_contract: list[BoundaryRef]
    resource_proof: ResourceProof
    backend_runtime: dict[str, Any] = field(default_factory=dict)
    ordered_segments: list[RoundSegment] = field(default_factory=list)
    item_kind: str = field(init=False, default=ITEM_KIND_WEIGHTED)

    def __post_init__(self) -> None:
        had_ordered_segments = bool(self.ordered_segments)
        self.ordered_segments = canonical_ordered_segments(self)
        if not had_ordered_segments:
            self.local_graph_nodes = _flatten_segment_nodes(self.ordered_segments)


@dataclass
class ZeroXbarRound(SerializableDataclass):
    item_id: str
    item_index: int
    zero_xbar_kind: str
    selected_entries: list[SelectedEntry]
    local_graph_nodes: list[LocalGraphNode]
    external_inputs: list[BoundaryRef]
    same_item_edges: list[SameItemEdge]
    external_outputs: list[BoundaryRef]
    partial_output_contract: list[BoundaryRef]
    resource_proof: ResourceProof
    backend_runtime: dict[str, Any] = field(default_factory=dict)
    ordered_segments: list[RoundSegment] = field(default_factory=list)
    item_kind: str = field(init=False, default=ITEM_KIND_ZERO_XBAR)

    def __post_init__(self) -> None:
        had_ordered_segments = bool(self.ordered_segments)
        self.ordered_segments = canonical_ordered_segments(self)
        if not had_ordered_segments:
            self.local_graph_nodes = _flatten_segment_nodes(self.ordered_segments)


@dataclass
class RoundPlan(SerializableDataclass):
    plan_id: str
    model_id: str
    backend_kind: str
    hardware_budget: dict[str, Any]
    execution_items: list[WeightedPhysicalRound | ZeroXbarRound | ExecutionItem]
    layer_resource_descriptors: list[LayerResourceDescriptor]
    model_graph_ref: dict[str, Any]
    feasible: bool
    invalid_reason: str | None = None
    sequence_spec: SequenceSpec = field(
        default_factory=lambda: SequenceSpec(1, True, False, [])
    )


def step_id(item_id: str, segment_index: int) -> str:
    return f"{item_id}.S{int(segment_index)}"


def _flatten_segment_nodes(segments: list[RoundSegment]) -> list[LocalGraphNode]:
    nodes: list[LocalGraphNode] = []
    for segment in segments:
        if isinstance(segment, WeightedLayerSegment):
            nodes.append(segment.local_graph_node)
        elif isinstance(segment, ZeroXbarSegment):
            nodes.extend(segment.local_graph_nodes)
        else:
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"unknown segment object {segment!r}",
            )
    return nodes


def canonical_ordered_segments(
    item: WeightedPhysicalRound | ZeroXbarRound,
) -> list[RoundSegment]:
    segments = list(getattr(item, "ordered_segments", []) or [])
    if segments:
        return sorted(
            segments,
            key=lambda segment: _coerce_segment_index(
                segment.segment_index,
                item.item_id,
            ),
        )

    if isinstance(item, ZeroXbarRound):
        return [
            ZeroXbarSegment(
                segment_index=index,
                zero_xbar_kind=item.zero_xbar_kind,
                local_graph_nodes=[node],
            )
            for index, node in enumerate(item.local_graph_nodes)
        ]

    if isinstance(item, WeightedPhysicalRound):
        return _legacy_weighted_ordered_segments(item)

    raise Phase1Invalid(
        INTERNAL_PLANNER_INVARIANT,
        f"unsupported execution item {item!r}",
    )


def _legacy_weighted_node(
    item: WeightedPhysicalRound,
    entry: SelectedEntry,
) -> LocalGraphNode:
    for node in item.local_graph_nodes:
        if (
            node.layer_id == entry.layer_id
            and node.layer_type == entry.layer_type
            and node.node_role == NODE_ROLE_WEIGHTED
        ):
            return node
    return LocalGraphNode(
        layer_id=entry.layer_id,
        layer_type=entry.layer_type,
        node_role=NODE_ROLE_WEIGHTED,
    )


def _legacy_weighted_ordered_segments(
    item: WeightedPhysicalRound,
) -> list[RoundSegment]:
    if not item.local_graph_nodes:
        return [
            WeightedLayerSegment(
                segment_index=index,
                selected_entry=entry,
                local_graph_node=_legacy_weighted_node(item, entry),
            )
            for index, entry in enumerate(item.selected_entries)
        ]

    unused_entries = list(item.selected_entries)
    segments: list[RoundSegment] = []
    for segment_index, node in enumerate(item.local_graph_nodes):
        if node.node_role == NODE_ROLE_WEIGHTED:
            entry_index = next(
                (
                    index
                    for index, entry in enumerate(unused_entries)
                    if entry.layer_id == node.layer_id
                ),
                None,
            )
            if entry_index is None:
                raise Phase1Invalid(
                    INTERNAL_PLANNER_INVARIANT,
                    f"legacy weighted node {node.layer_id} in {item.item_id} has no unused selected entry",
                )
            segments.append(
                WeightedLayerSegment(
                    segment_index=segment_index,
                    selected_entry=unused_entries.pop(entry_index),
                    local_graph_node=node,
                )
            )
            continue
        segments.append(
            ZeroXbarSegment(
                segment_index=segment_index,
                zero_xbar_kind=_legacy_zero_xbar_kind(node, item.item_id),
                local_graph_nodes=[node],
            )
        )

    if unused_entries:
        raise Phase1Invalid(
            INTERNAL_PLANNER_INVARIANT,
            f"legacy weighted round {item.item_id} did not consume every selected entry",
        )
    return segments


def _legacy_zero_xbar_kind(node: LocalGraphNode, item_id: str) -> str:
    if node.node_role == NODE_ROLE_HOOK and node.layer_type in HOOK_LAYER_TYPES:
        return ZERO_XBAR_KIND_HOOK_ONLY
    if (
        node.node_role == NODE_ROLE_STRUCTURAL_NON_WEIGHTED
        and node.layer_type in STRUCTURAL_NON_WEIGHTED_LAYER_TYPES
    ):
        return ZERO_XBAR_KIND_STRUCTURAL_MNSIM
    raise Phase1Invalid(
        INTERNAL_PLANNER_INVARIANT,
        f"unsupported legacy nonweighted node {node.layer_id} in {item_id}",
    )


def _coerce_segment_index(value: Any, item_id: str) -> int:
    if type(value) is not int:
        raise Phase1Invalid(
            INTERNAL_PLANNER_INVARIANT,
            f"segment_index for {item_id} must be an integer",
        )
    return value


def validate_round_plan(plan: RoundPlan) -> None:
    item_ids: set[str] = set()
    for expected_item_index, item in enumerate(plan.execution_items):
        if item.item_index != expected_item_index:
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"item_index for {item.item_id} must be {expected_item_index}",
            )
        if item.item_id in item_ids:
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"duplicate item_id {item.item_id}",
            )
        item_ids.add(item.item_id)
        segments = canonical_ordered_segments(item)
        item.ordered_segments = segments
        if [
            _coerce_segment_index(segment.segment_index, item.item_id)
            for segment in segments
        ] != list(
            range(len(segments))
        ):
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"segment_index values for {item.item_id} must be contiguous from zero",
            )
        if _flatten_segment_nodes(segments) != item.local_graph_nodes:
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"local_graph_nodes for {item.item_id} do not match ordered segments",
            )
        if isinstance(item, ZeroXbarRound):
            if item.selected_entries:
                raise Phase1Invalid(
                    INTERNAL_PLANNER_INVARIANT,
                    f"zero-xbar round {item.item_id} cannot select weighted entries",
                )
            if int(item.resource_proof.used_xbars) != 0:
                raise Phase1Invalid(
                    INTERNAL_PLANNER_INVARIANT,
                    f"used_xbars proof for zero-xbar round {item.item_id} must be zero",
                )
            segment_roles = set()
            for segment in segments:
                if not isinstance(segment, ZeroXbarSegment):
                    raise Phase1Invalid(
                        INTERNAL_PLANNER_INVARIANT,
                        f"zero-xbar round {item.item_id} may contain only zero_xbar_segment",
                    )
                if segment.segment_kind != SEGMENT_KIND_ZERO_XBAR:
                    raise Phase1Invalid(
                        INTERNAL_PLANNER_INVARIANT,
                        f"zero-xbar round {item.item_id} may contain only zero_xbar_segment",
                    )
                segment_roles.update(
                    _validate_zero_xbar_segment(segment, item.item_id)
                )
            expected_roles = {
                ZERO_XBAR_KIND_HOOK_ONLY: {NODE_ROLE_HOOK},
                ZERO_XBAR_KIND_STRUCTURAL_MNSIM: {NODE_ROLE_STRUCTURAL_NON_WEIGHTED},
                ZERO_XBAR_KIND_MIXED_NON_WEIGHTED: {
                    NODE_ROLE_HOOK,
                    NODE_ROLE_STRUCTURAL_NON_WEIGHTED,
                },
            }
            if item.zero_xbar_kind not in expected_roles or not segment_roles.issubset(
                expected_roles[item.zero_xbar_kind]
            ):
                raise Phase1Invalid(
                    INTERNAL_PLANNER_INVARIANT,
                    f"zero-xbar node roles for {item.item_id} do not match {item.zero_xbar_kind}",
                )
            continue
        if not isinstance(item, WeightedPhysicalRound):
            continue
        weighted_segments = []
        for segment in segments:
            if isinstance(segment, WeightedLayerSegment):
                if segment.segment_kind != SEGMENT_KIND_WEIGHTED_LAYER:
                    raise Phase1Invalid(
                        INTERNAL_PLANNER_INVARIANT,
                        f"unknown segment_kind {segment.segment_kind} in {item.item_id}",
                    )
                if segment.local_graph_node.node_role != NODE_ROLE_WEIGHTED:
                    raise Phase1Invalid(
                        INTERNAL_PLANNER_INVARIANT,
                        f"weighted segment node role for {item.item_id} must be weighted",
                    )
                weighted_segments.append(segment)
            elif isinstance(segment, ZeroXbarSegment):
                if segment.segment_kind != SEGMENT_KIND_ZERO_XBAR:
                    raise Phase1Invalid(
                        INTERNAL_PLANNER_INVARIANT,
                        f"unknown segment_kind {segment.segment_kind} in {item.item_id}",
                    )
                _validate_zero_xbar_segment(segment, item.item_id)
            else:
                raise Phase1Invalid(
                    INTERNAL_PLANNER_INVARIANT,
                    f"unknown segment object {segment!r}",
                )
        if item.round_kind == ROUND_KIND_FULL_LAYER_PACK and any(
            not entry.layer_complete_in_round for entry in item.selected_entries
        ):
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"full_layer_pack {item.item_id} requires every selected entry to be complete",
            )
        if item.round_kind in {
            ROUND_KIND_SINGLE_LAYER_SPLIT,
            ROUND_KIND_SINGLE_LAYER_CONTINUATION,
        } and (len(item.selected_entries) != 1 or len(weighted_segments) != 1):
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"{item.round_kind} {item.item_id} requires exactly one selected entry and one weighted segment",
            )
        selected_used_xbars = sum(
            int(entry.used_xbars) for entry in item.selected_entries
        )
        if int(item.resource_proof.used_xbars) != selected_used_xbars:
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"used_xbars proof for {item.item_id} must equal "
                f"selected entry sum {selected_used_xbars}",
            )
        selected_identities = sorted(
            _selected_entry_identity(entry) for entry in item.selected_entries
        )
        segment_identities = sorted(
            _selected_entry_identity(segment.selected_entry)
            for segment in segments
            if isinstance(segment, WeightedLayerSegment)
        )
        if segment_identities != selected_identities:
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"weighted segment for {item.item_id} is missing a selected entry",
            )


def _validate_zero_xbar_segment(
    segment: ZeroXbarSegment,
    item_id: str,
) -> set[str]:
    if segment.zero_xbar_kind not in {
        ZERO_XBAR_KIND_HOOK_ONLY,
        ZERO_XBAR_KIND_STRUCTURAL_MNSIM,
        ZERO_XBAR_KIND_MIXED_NON_WEIGHTED,
    }:
        raise Phase1Invalid(
            INTERNAL_PLANNER_INVARIANT,
            f"unknown zero_xbar_kind {segment.zero_xbar_kind} in {item_id}",
        )
    roles = set()
    for node in segment.local_graph_nodes:
        if node.node_role == NODE_ROLE_HOOK and node.layer_type in HOOK_LAYER_TYPES:
            roles.add(NODE_ROLE_HOOK)
        elif (
            node.node_role == NODE_ROLE_STRUCTURAL_NON_WEIGHTED
            and node.layer_type in STRUCTURAL_NON_WEIGHTED_LAYER_TYPES
        ):
            roles.add(NODE_ROLE_STRUCTURAL_NON_WEIGHTED)
        else:
            raise Phase1Invalid(
                INTERNAL_PLANNER_INVARIANT,
                f"unsupported zero-xbar node role/type for {item_id}: {node.node_role}/{node.layer_type}",
            )
    expected_roles = {
        ZERO_XBAR_KIND_HOOK_ONLY: {NODE_ROLE_HOOK},
        ZERO_XBAR_KIND_STRUCTURAL_MNSIM: {NODE_ROLE_STRUCTURAL_NON_WEIGHTED},
        ZERO_XBAR_KIND_MIXED_NON_WEIGHTED: {
            NODE_ROLE_HOOK,
            NODE_ROLE_STRUCTURAL_NON_WEIGHTED,
        },
    }[segment.zero_xbar_kind]
    if not roles or not roles.issubset(expected_roles):
        raise Phase1Invalid(
            INTERNAL_PLANNER_INVARIANT,
            f"zero-xbar node roles for {item_id} do not match {segment.zero_xbar_kind}",
        )
    return roles


def _selected_entry_identity(entry: SelectedEntry) -> tuple[str, int, int]:
    return (
        entry.layer_id,
        int(entry.xbar_range_start),
        int(entry.xbar_range_end),
    )


def _prepass_ref_from_dict(data: dict[str, Any]) -> PrepassRef:
    return PrepassRef(
        ref_id=data["ref_id"],
        model_id=data["model_id"],
        layer_id=data["layer_id"],
        layer_type=data["layer_type"],
        resource_version=data["resource_version"],
        sim_config_fingerprint=data["sim_config_fingerprint"],
        layer_config_fingerprint=data["layer_config_fingerprint"],
        weight_ref_id=data.get("weight_ref_id"),
        shape_signature=dict(data["shape_signature"]),
    )


def _selected_entry_from_dict(data: dict[str, Any]) -> SelectedEntry:
    return SelectedEntry(
        layer_id=data["layer_id"],
        layer_type=data["layer_type"],
        xbar_range_start=data["xbar_range_start"],
        xbar_range_end=data["xbar_range_end"],
        used_xbars=data["used_xbars"],
        layer_complete_in_round=data["layer_complete_in_round"],
        prepass_ref=_prepass_ref_from_dict(data["prepass_ref"]),
    )


def _local_graph_node_from_dict(data: dict[str, Any]) -> LocalGraphNode:
    return LocalGraphNode(
        layer_id=data["layer_id"],
        layer_type=data["layer_type"],
        node_role=data["node_role"],
        node_shape=dict(data.get("node_shape", {}) or {}),
        digital_metadata=dict(data.get("digital_metadata", {}) or {}),
    )


def _weighted_layer_segment_from_dict(data: dict[str, Any]) -> WeightedLayerSegment:
    return WeightedLayerSegment(
        segment_index=data["segment_index"],
        selected_entry=_selected_entry_from_dict(data["selected_entry"]),
        local_graph_node=_local_graph_node_from_dict(data["local_graph_node"]),
    )


def _zero_xbar_segment_from_dict(data: dict[str, Any]) -> ZeroXbarSegment:
    return ZeroXbarSegment(
        segment_index=data["segment_index"],
        zero_xbar_kind=data["zero_xbar_kind"],
        local_graph_nodes=[
            _local_graph_node_from_dict(item) for item in data["local_graph_nodes"]
        ],
    )


def _round_segment_from_dict(
    data: dict[str, Any],
) -> WeightedLayerSegment | ZeroXbarSegment:
    segment_kind = data["segment_kind"]
    if segment_kind == SEGMENT_KIND_WEIGHTED_LAYER:
        return _weighted_layer_segment_from_dict(data)
    if segment_kind == SEGMENT_KIND_ZERO_XBAR:
        return _zero_xbar_segment_from_dict(data)
    raise Phase1Invalid(
        INTERNAL_PLANNER_INVARIANT, f"unknown segment_kind {segment_kind}"
    )


def _boundary_ref_from_dict(data: dict[str, Any]) -> BoundaryRef:
    return BoundaryRef(
        state_id=data["state_id"],
        tensor_role=data["tensor_role"],
        producer_item_id=data.get("producer_item_id"),
        consumer_item_ids=list(data.get("consumer_item_ids", [])),
        consumer_node_ids=list(data.get("consumer_node_ids", [])),
        partial_or_complete=data.get("partial_or_complete", "complete"),
    )


def _same_item_edge_from_dict(data: dict[str, Any]) -> SameItemEdge:
    return SameItemEdge(
        producer_node_id=data["producer_node_id"],
        consumer_node_id=data["consumer_node_id"],
    )


def _resource_proof_from_dict(data: dict[str, Any]) -> ResourceProof:
    return ResourceProof(
        xbar_nums=data["xbar_nums"],
        used_xbars=data["used_xbars"],
    )


def _completion_contract_from_dict(data: dict[str, Any]) -> CompletionContract:
    return CompletionContract(
        partial_visibility=data.get("partial_visibility", "internal_only"),
        downstream_visibility=data.get("downstream_visibility", "complete_only"),
        same_item_handoff_allowed=data.get("same_item_handoff_allowed", True),
        completion_hook_trigger=data.get("completion_hook_trigger", "on_complete"),
    )


def _layer_resource_descriptor_from_dict(data: dict[str, Any]) -> LayerResourceDescriptor:
    return LayerResourceDescriptor(
        layer_id=data["layer_id"],
        layer_type=data["layer_type"],
        input_layer_ids=list(data["input_layer_ids"]),
        output_consumer_ids=list(data["output_consumer_ids"]),
        required_xbar_count=data["required_xbar_count"],
        prepass_ref=_prepass_ref_from_dict(data["prepass_ref"]),
        completion_contract=_completion_contract_from_dict(
            data.get("completion_contract", {})
        ),
    )


def execution_item_from_dict(
    data: dict[str, Any],
) -> WeightedPhysicalRound | ZeroXbarRound | ExecutionItem:
    item_kind = data["item_kind"]
    common = {
        "item_id": data["item_id"],
        "item_index": data["item_index"],
        "local_graph_nodes": [
            _local_graph_node_from_dict(item) for item in data["local_graph_nodes"]
        ],
        "external_inputs": [
            _boundary_ref_from_dict(item) for item in data["external_inputs"]
        ],
        "same_item_edges": [
            _same_item_edge_from_dict(item) for item in data["same_item_edges"]
        ],
        "external_outputs": [
            _boundary_ref_from_dict(item) for item in data["external_outputs"]
        ],
        "partial_output_contract": [
            _boundary_ref_from_dict(item) for item in data["partial_output_contract"]
        ],
        "resource_proof": _resource_proof_from_dict(data["resource_proof"]),
        "backend_runtime": dict(data.get("backend_runtime", {})),
    }
    if item_kind == ITEM_KIND_WEIGHTED:
        return WeightedPhysicalRound(
            round_kind=data["round_kind"],
            selected_entries=[
                _selected_entry_from_dict(item) for item in data["selected_entries"]
            ],
            ordered_segments=[
                _round_segment_from_dict(item)
                for item in data.get("ordered_segments", [])
            ],
            **common,
        )
    if item_kind == ITEM_KIND_ZERO_XBAR:
        return ZeroXbarRound(
            zero_xbar_kind=data["zero_xbar_kind"],
            selected_entries=[
                _selected_entry_from_dict(item) for item in data["selected_entries"]
            ],
            ordered_segments=[
                _round_segment_from_dict(item)
                for item in data.get("ordered_segments", [])
            ],
            **common,
        )
    raise Phase1Invalid(INTERNAL_PLANNER_INVARIANT, f"unknown item_kind {item_kind}")


def round_plan_from_dict(data: dict[str, Any]) -> RoundPlan:
    plan = RoundPlan(
        plan_id=data["plan_id"],
        model_id=data["model_id"],
        backend_kind=data["backend_kind"],
        hardware_budget=dict(data["hardware_budget"]),
        execution_items=[
            execution_item_from_dict(item) for item in data["execution_items"]
        ],
        layer_resource_descriptors=[
            _layer_resource_descriptor_from_dict(item)
            for item in data["layer_resource_descriptors"]
        ],
        model_graph_ref=dict(data["model_graph_ref"]),
        feasible=data["feasible"],
        invalid_reason=data.get("invalid_reason"),
        sequence_spec=SequenceSpec(
            **dict(
                data.get(
                    "sequence_spec",
                    {
                        "sequence_length": 1,
                        "ordered_steps": True,
                        "cross_step_dependency": False,
                        "recurrent_state_ids": [],
                    },
                )
            )
        ),
    )
    validate_round_plan(plan)
    return plan


__all__ = [
    "ITEM_KIND_WEIGHTED",
    "ITEM_KIND_ZERO_XBAR",
    "SEGMENT_KIND_WEIGHTED_LAYER",
    "SEGMENT_KIND_ZERO_XBAR",
    "ROUND_KIND_FULL_LAYER_PACK",
    "ROUND_KIND_SINGLE_LAYER_SPLIT",
    "ROUND_KIND_SINGLE_LAYER_CONTINUATION",
    "ZERO_XBAR_KIND_HOOK_ONLY",
    "ZERO_XBAR_KIND_STRUCTURAL_MNSIM",
    "ZERO_XBAR_KIND_MIXED_NON_WEIGHTED",
    "NODE_ROLE_WEIGHTED",
    "NODE_ROLE_HOOK",
    "NODE_ROLE_STRUCTURAL_NON_WEIGHTED",
    "WEIGHTED_LAYER_TYPES",
    "HOOK_LAYER_TYPES",
    "STRUCTURAL_NON_WEIGHTED_LAYER_TYPES",
    "NON_WEIGHTED_LAYER_TYPES",
    "SUPPORTED_LAYER_TYPES",
    "INVALID_SIM_CONFIG",
    "INVALID_HARDWARE_PROJECTION",
    "UNSUPPORTED_LAYER_TYPE",
    "INVALID_LAYER_SHAPE",
    "INVALID_MODEL_DAG",
    "INVALID_SEQUENCE_SPEC",
    "INVALID_LIF_DOMAIN",
    "INVALID_RESOURCE_FOOTPRINT",
    "INFEASIBLE_LAYER_ATOMIC_SCOPE",
    "INTERNAL_PLANNER_INVARIANT",
    "Phase1Invalid",
    "SequenceSpec",
    "LayerIR",
    "ModelIR",
    "PrepassRef",
    "CompletionContract",
    "LayerResourceDescriptor",
    "SelectedEntry",
    "LocalGraphNode",
    "WeightedLayerSegment",
    "ZeroXbarSegment",
    "RoundSegment",
    "BoundaryRef",
    "SameItemEdge",
    "ResourceProof",
    "ExecutionItem",
    "WeightedPhysicalRound",
    "ZeroXbarRound",
    "RoundPlan",
    "step_id",
    "canonical_ordered_segments",
    "validate_round_plan",
    "execution_item_from_dict",
    "round_plan_from_dict",
]
