from __future__ import annotations

from typing import Any

from MNSIM.Scheduling.digital_domain import AccumulatorDomain, DigitalDomainError
from MNSIM.Scheduling.phase1.config import SimConfigInput, build_hardware_projection
from MNSIM.Scheduling.phase1.resource_prepass import build_layer_resource_descriptors
from MNSIM.Scheduling.round_types import (
    HOOK_LAYER_TYPES,
    INTERNAL_PLANNER_INVARIANT,
    INVALID_MODEL_DAG,
    INVALID_LIF_DOMAIN,
    INVALID_SEQUENCE_SPEC,
    BoundaryRef,
    LayerResourceDescriptor,
    LocalGraphNode,
    ModelIR,
    NON_WEIGHTED_LAYER_TYPES,
    NODE_ROLE_HOOK,
    NODE_ROLE_STRUCTURAL_NON_WEIGHTED,
    NODE_ROLE_WEIGHTED,
    Phase1Invalid,
    ROUND_KIND_FULL_LAYER_PACK,
    ROUND_KIND_SINGLE_LAYER_CONTINUATION,
    ROUND_KIND_SINGLE_LAYER_SPLIT,
    ResourceProof,
    RoundPlan,
    SameItemEdge,
    SelectedEntry,
    STRUCTURAL_NON_WEIGHTED_LAYER_TYPES,
    SUPPORTED_LAYER_TYPES,
    UNSUPPORTED_LAYER_TYPE,
    WEIGHTED_LAYER_TYPES,
    WeightedLayerSegment,
    WeightedPhysicalRound,
    ZERO_XBAR_KIND_HOOK_ONLY,
    ZERO_XBAR_KIND_MIXED_NON_WEIGHTED,
    ZERO_XBAR_KIND_STRUCTURAL_MNSIM,
    ZeroXbarSegment,
    ZeroXbarRound,
    validate_round_plan,
)


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
        sequence_spec=model_ir.sequence_spec,
    )


def _validate_sequence_spec(model_ir: ModelIR) -> None:
    seq = model_ir.sequence_spec
    if seq.sequence_length <= 0:
        raise Phase1Invalid(INVALID_SEQUENCE_SPEC, "sequence_length must be positive")
    if seq.cross_step_dependency and not seq.ordered_steps:
        raise Phase1Invalid(INVALID_SEQUENCE_SPEC, "cross_step_dependency requires ordered_steps")
    by_id = {layer.layer_id: layer for layer in model_ir.layers}
    lif_ids = {layer.layer_id for layer in model_ir.layers if layer.layer_type == "lif"}
    if not lif_ids:
        return
    if len(seq.recurrent_state_ids) != len(set(seq.recurrent_state_ids)):
        raise Phase1Invalid(INVALID_SEQUENCE_SPEC, "duplicate recurrent_state_id")
    for state_id in seq.recurrent_state_ids:
        if state_id not in by_id:
            raise Phase1Invalid(
                INVALID_SEQUENCE_SPEC,
                f"unknown recurrent_state_id {state_id}",
            )
        if by_id[state_id].layer_type != "lif":
            raise Phase1Invalid(
                INVALID_SEQUENCE_SPEC,
                f"recurrent_state_id {state_id} must resolve to lif",
            )
    if seq.cross_step_dependency and lif_ids != set(seq.recurrent_state_ids):
        missing = sorted(lif_ids - set(seq.recurrent_state_ids))
        if missing:
            raise Phase1Invalid(
                INVALID_SEQUENCE_SPEC,
                f"lif nodes missing from recurrent_state_ids: {missing}",
            )


def _layer_domain(layer: Any) -> AccumulatorDomain:
    metadata = layer.backend_attrs.get("accumulator_domain")
    if not isinstance(metadata, dict):
        raise Phase1Invalid(
            INVALID_LIF_DOMAIN,
            f"{layer.layer_id} has no accumulator domain metadata",
        )
    try:
        return AccumulatorDomain.from_dict(metadata)
    except (KeyError, DigitalDomainError) as exc:
        raise Phase1Invalid(
            INVALID_LIF_DOMAIN,
            f"{layer.layer_id} has invalid accumulator domain: {exc}",
        ) from exc


def _validate_lif_nodes(model_ir: ModelIR) -> None:
    by_id = {layer.layer_id: layer for layer in model_ir.layers}
    for layer in model_ir.layers:
        if layer.layer_type != "lif":
            continue
        if len(layer.input_layer_ids) != 1:
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"{layer.layer_id} must have exactly one direct input",
            )
        producer = by_id[layer.input_layer_ids[0]]
        owner_id = layer.backend_attrs.get('lif_owner_layer_id')
        if owner_id is not None:
            # Only transparent BN/view and the declared main residual input
            # may bridge LIF placement to a weighted reducer. Never traverse
            # past another LIF or silently choose the shortcut's placement.
            owner = producer
            while owner.layer_type in {'bn', 'view', 'element_sum'}:
                if owner.layer_type == 'element_sum':
                    if (owner.backend_attrs.get('merge_ownership_policy') != 'main_branch_reducer'
                            or not owner.input_layer_ids):
                        raise Phase1Invalid(INVALID_LIF_DOMAIN, f'{owner.layer_id}: invalid merge ownership')
                elif len(owner.input_layer_ids) != 1:
                    raise Phase1Invalid(INVALID_LIF_DOMAIN, f'{owner.layer_id}: invalid owner bridge')
                owner = by_id[owner.input_layer_ids[0]]
            if not owner.is_weighted or owner.layer_id != owner_id:
                raise Phase1Invalid(INVALID_LIF_DOMAIN, f'{layer.layer_id}: invalid weighted owner {owner_id}')
        producer_domain = _layer_domain(producer)
        lif_domain = _layer_domain(layer)
        try:
            producer_domain.require_bit_exact_compatible(lif_domain)
        except DigitalDomainError as exc:
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"{layer.layer_id} accumulator domain mismatch: {exc}",
            ) from exc
        constants = layer.backend_attrs.get("compiled_lif_constants")
        if not isinstance(constants, dict):
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"{layer.layer_id} has no compiled LIF constants",
            )
        constant_domain = constants.get("domain")
        try:
            compiled_domain = AccumulatorDomain.from_dict(constant_domain)
            lif_domain.require_bit_exact_compatible(compiled_domain)
        except (TypeError, KeyError, DigitalDomainError) as exc:
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"{layer.layer_id} compiled constant accumulator domain mismatch: {exc}",
            ) from exc


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
    _validate_lif_nodes(model_ir)


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
        sequence_spec=model_ir.sequence_spec,
    )


def _descriptor_map(descriptors: list[LayerResourceDescriptor]) -> dict[str, LayerResourceDescriptor]:
    return {desc.layer_id: desc for desc in descriptors}


def _layers_by_id(model_ir: ModelIR) -> dict[str, Any]:
    return {layer.layer_id: layer for layer in model_ir.layers}


def _is_ready(layer: Any, complete_layers: set[str]) -> bool:
    return all(input_id in complete_layers for input_id in layer.input_layer_ids)


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


def _digital_metadata(layer: Any) -> dict[str, Any]:
    return {
        key: layer.backend_attrs[key]
        for key in ("accumulator_domain", "compiled_lif_constants", "lif_owner_layer_id",
                    "snn_merge_mode", "merge_ownership_policy", "merge_source_ids",
                    "merge_input_regions", "concat_dim", "time_policy", "merge_mode",
                    "value_domain")
        if key in layer.backend_attrs
    }


def _weighted_layer_segment(
    segment_index: int,
    entry: SelectedEntry,
    layer: Any,
) -> WeightedLayerSegment:
    return WeightedLayerSegment(
        segment_index=segment_index,
        selected_entry=entry,
        local_graph_node=LocalGraphNode(
            layer_id=entry.layer_id,
            layer_type=entry.layer_type,
            node_role=NODE_ROLE_WEIGHTED,
            node_shape=dict(layer.shape),
            digital_metadata=_digital_metadata(layer),
        ),
    )


def _zero_xbar_segment(segment_index: int, nodes: list[Any]) -> ZeroXbarSegment:
    return ZeroXbarSegment(
        segment_index=segment_index,
        zero_xbar_kind=_zero_xbar_kind(nodes),
        local_graph_nodes=[
            LocalGraphNode(
                layer_id=node.layer_id,
                layer_type=node.layer_type,
                node_role=_node_role_for_non_weighted(node),
                node_shape=dict(node.shape),
                digital_metadata=_digital_metadata(node),
            )
            for node in nodes
        ],
    )


def _flatten_segment_nodes(
    ordered_segments: list[WeightedLayerSegment | ZeroXbarSegment],
) -> list[LocalGraphNode]:
    nodes: list[LocalGraphNode] = []
    for segment in ordered_segments:
        if isinstance(segment, WeightedLayerSegment):
            nodes.append(segment.local_graph_node)
        elif isinstance(segment, ZeroXbarSegment):
            nodes.extend(segment.local_graph_nodes)
        else:
            raise Phase1Invalid(INTERNAL_PLANNER_INVARIANT, f"unknown segment object {segment!r}")
    return nodes


def _plan_full_layer_pack(
    item_index: int,
    resource_round_index: int,
    desc_by_id: dict[str, LayerResourceDescriptor],
    by_id: dict[str, Any],
    global_complete: set[str],
    unfinished: set[str],
    hardware_xbars: int,
    leading_zero_nodes: list[Any] | None = None,
    forced_first_weighted: Any | None = None,
) -> tuple[WeightedPhysicalRound, set[str], set[str]]:
    local_complete = set(global_complete)
    local_unfinished = set(unfinished)
    selected: list[SelectedEntry] = []
    selected_ids: set[str] = set()
    ordered_segments: list[WeightedLayerSegment | ZeroXbarSegment] = []
    used = 0

    for node in leading_zero_nodes or []:
        local_complete.add(node.layer_id)
        local_unfinished.discard(node.layer_id)
        ordered_segments.append(_zero_xbar_segment(len(ordered_segments), [node]))

    while True:
        ready_weighted = _ready_weighted_layers(by_id, local_unfinished, local_complete)
        if forced_first_weighted is not None and not selected:
            fit_nodes = [forced_first_weighted] if (
                forced_first_weighted in ready_weighted
                and used + desc_by_id[forced_first_weighted.layer_id].required_xbar_count <= hardware_xbars
            ) else []
        else:
            fit_nodes = [
                layer
                for layer in ready_weighted
                if layer.layer_id not in selected_ids
                and used + desc_by_id[layer.layer_id].required_xbar_count <= hardware_xbars
            ]
        if not fit_nodes:
            break

        layer = fit_nodes[0]
        desc = desc_by_id[layer.layer_id]
        ready_zero_before_weighted = {
            node.layer_id
            for node in _ready_non_weighted_layers(by_id, local_unfinished, local_complete)
        }
        entry = _selected_entry(desc, 0, desc.required_xbar_count, True)
        selected.append(entry)
        selected_ids.add(layer.layer_id)
        ordered_segments.append(
            _weighted_layer_segment(len(ordered_segments), entry, layer)
        )
        used += desc.required_xbar_count
        local_complete.add(layer.layer_id)
        local_unfinished.remove(layer.layer_id)

        zero_nodes = _consume_newly_unlocked_zero_xbar_closure_nodes(
            by_id,
            local_unfinished,
            local_complete,
            ready_zero_before_weighted,
        )
        for node in zero_nodes:
            ordered_segments.append(_zero_xbar_segment(len(ordered_segments), [node]))

    if not selected:
        raise Phase1Invalid(INTERNAL_PLANNER_INVARIANT, "planner failed to select a weighted node")
    item = _weighted_item(
        item_index,
        resource_round_index,
        ROUND_KIND_FULL_LAYER_PACK,
        selected,
        ordered_segments,
        hardware_xbars,
    )
    return item, local_complete, local_unfinished


def _weighted_item(
    item_index: int,
    resource_round_index: int,
    round_kind: str,
    selected_entries: list[SelectedEntry],
    ordered_segments: list[WeightedLayerSegment | ZeroXbarSegment],
    hardware_xbars: int,
) -> WeightedPhysicalRound:
    used = sum(entry.used_xbars for entry in selected_entries)
    return WeightedPhysicalRound(
        item_id=f"R{resource_round_index}",
        item_index=item_index,
        round_kind=round_kind,
        selected_entries=selected_entries,
        local_graph_nodes=_flatten_segment_nodes(ordered_segments),
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=hardware_xbars, used_xbars=used),
        backend_runtime={},
        ordered_segments=ordered_segments,
    )


def _ready_non_weighted_layers(by_id: dict[str, Any], unfinished: set[str], complete_layers: set[str]) -> list[Any]:
    return sorted(
        [
            by_id[layer_id]
            for layer_id in unfinished
            if not by_id[layer_id].is_weighted and _is_ready(by_id[layer_id], complete_layers)
        ],
        key=lambda layer: layer.stable_topological_index,
    )


def _ready_weighted_layers(by_id: dict[str, Any], unfinished: set[str], complete_layers: set[str]) -> list[Any]:
    return sorted(
        [
            by_id[layer_id]
            for layer_id in unfinished
            if by_id[layer_id].is_weighted and _is_ready(by_id[layer_id], complete_layers)
        ],
        key=lambda layer: layer.stable_topological_index,
    )


def _plan_split_or_continuation(
    item_index: int,
    resource_round_index: int,
    desc: LayerResourceDescriptor,
    cursor: int,
    hardware_xbars: int,
    layer: Any,
    leading_zero_nodes: list[Any] | None = None,
) -> tuple[WeightedPhysicalRound, int, bool]:
    remaining = desc.required_xbar_count - cursor
    take = min(hardware_xbars, remaining)
    start = cursor
    end = cursor + take
    complete = end == desc.required_xbar_count
    round_kind = ROUND_KIND_SINGLE_LAYER_CONTINUATION if start > 0 else ROUND_KIND_SINGLE_LAYER_SPLIT
    entry = _selected_entry(desc, start, end, complete)
    ordered_segments: list[WeightedLayerSegment | ZeroXbarSegment] = []
    if cursor == 0:
        for node in leading_zero_nodes or []:
            ordered_segments.append(_zero_xbar_segment(len(ordered_segments), [node]))
    ordered_segments.append(
        _weighted_layer_segment(len(ordered_segments), entry, layer)
    )
    item = _weighted_item(
        item_index,
        resource_round_index,
        round_kind,
        [entry],
        ordered_segments,
        hardware_xbars,
    )
    return item, end, complete


def _find_next_weighted_host(
    by_id: dict[str, Any],
    complete_layers: set[str],
    unfinished: set[str],
) -> tuple[list[Any], Any] | None:
    """Find the first weighted node whose unfinished ancestors are all non-weighted."""
    for candidate in _ready_weighted_candidates(by_id, unfinished):
        required_non_weighted: set[str] = set()
        visiting: set[str] = set()

        def collect_ancestors(layer_id: str) -> bool:
            if layer_id in complete_layers:
                return True
            if layer_id in visiting:
                return False
            node = by_id[layer_id]
            if node.is_weighted:
                return False
            visiting.add(layer_id)
            required_non_weighted.add(layer_id)
            valid = all(collect_ancestors(input_id) for input_id in node.input_layer_ids)
            visiting.remove(layer_id)
            return valid

        if not all(collect_ancestors(input_id) for input_id in candidate.input_layer_ids):
            continue
        closure_nodes = sorted(
            (by_id[layer_id] for layer_id in required_non_weighted),
            key=lambda node: node.stable_topological_index,
        )
        simulated_complete = complete_layers | required_non_weighted
        if _is_ready(candidate, simulated_complete):
            return closure_nodes, candidate
    return None


def _ready_weighted_candidates(by_id: dict[str, Any], unfinished: set[str]) -> list[Any]:
    return sorted(
        [by_id[layer_id] for layer_id in unfinished if by_id[layer_id].is_weighted],
        key=lambda layer: layer.stable_topological_index,
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


def _zero_xbar_item(item_index: int, zero_round_index: int, nodes: list[Any], hardware_xbars: int) -> ZeroXbarRound:
    return ZeroXbarRound(
        item_id=f"Z{zero_round_index}",
        item_index=item_index,
        zero_xbar_kind=_zero_xbar_kind(nodes),
        selected_entries=[],
        local_graph_nodes=[
            LocalGraphNode(
                layer_id=node.layer_id,
                layer_type=node.layer_type,
                node_role=_node_role_for_non_weighted(node),
                node_shape=dict(node.shape),
                digital_metadata=_digital_metadata(node),
            )
            for node in nodes
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=hardware_xbars, used_xbars=0),
        backend_runtime={},
    )


def _append_zero_xbar_item(
    items: list[Any],
    zero_round_index: int,
    nodes: list[Any],
    hardware_xbars: int,
) -> int:
    if not nodes:
        return zero_round_index
    items.append(_zero_xbar_item(len(items), zero_round_index, nodes, hardware_xbars))
    return zero_round_index + 1


def _consume_zero_xbar_closure_nodes(
    by_id: dict[str, Any],
    unfinished: set[str],
    complete_layers: set[str],
) -> list[Any]:
    nodes: list[Any] = []
    while True:
        ready = _ready_non_weighted_layers(by_id, unfinished, complete_layers)
        if not ready:
            break
        node = ready[0]
        nodes.append(node)
        complete_layers.add(node.layer_id)
        unfinished.remove(node.layer_id)
    return nodes


def _consume_newly_unlocked_zero_xbar_closure_nodes(
    by_id: dict[str, Any],
    unfinished: set[str],
    complete_layers: set[str],
    ready_before_completion: set[str],
) -> list[Any]:
    """Consume only the zero-xbar closure made ready by the current completion."""
    nodes: list[Any] = []
    while True:
        ready = [
            node
            for node in _ready_non_weighted_layers(by_id, unfinished, complete_layers)
            if node.layer_id not in ready_before_completion
        ]
        if not ready:
            break
        node = ready[0]
        nodes.append(node)
        complete_layers.add(node.layer_id)
        unfinished.remove(node.layer_id)
    return nodes


def _plan_execution_items(model_ir: ModelIR, hardware: Any, descriptors: list[LayerResourceDescriptor]) -> list[Any]:
    by_id = _layers_by_id(model_ir)
    desc_by_id = _descriptor_map(descriptors)
    unfinished = {layer.layer_id for layer in model_ir.layers}
    complete_layers: set[str] = set()
    split_cursors: dict[str, int] = {}
    items: list[Any] = []
    resource_round_index = 0
    zero_round_index = 0

    while unfinished:
        ready_weighted = _ready_weighted_layers(by_id, unfinished, complete_layers)
        if not ready_weighted:
            ready_non_weighted = _ready_non_weighted_layers(by_id, unfinished, complete_layers)
            if ready_non_weighted:
                zero_nodes = _consume_zero_xbar_closure_nodes(by_id, unfinished, complete_layers)
                items.append(_zero_xbar_item(len(items), zero_round_index, zero_nodes, hardware.xbar_nums))
                zero_round_index += 1
                continue
            raise Phase1Invalid(INVALID_MODEL_DAG, "no ready node; graph may contain a cycle or unsupported dependency")

        first = ready_weighted[0]
        first_desc = desc_by_id[first.layer_id]
        cursor = split_cursors.get(first.layer_id, 0)
        if first_desc.required_xbar_count > hardware.xbar_nums or cursor > 0:
            item, new_cursor, complete = _plan_split_or_continuation(
                len(items),
                resource_round_index,
                first_desc,
                cursor,
                hardware.xbar_nums,
                first,
            )
            items.append(item)
            resource_round_index += 1
            if complete:
                ready_zero_before_final_continuation = {
                    node.layer_id
                    for node in _ready_non_weighted_layers(by_id, unfinished, complete_layers)
                }
                complete_layers.add(first.layer_id)
                unfinished.remove(first.layer_id)
                split_cursors.pop(first.layer_id, None)
                host = _find_next_weighted_host(by_id, complete_layers, unfinished)
                if host is not None:
                    leading_zero_nodes, forced_first_weighted = host
                    forced_desc = desc_by_id[forced_first_weighted.layer_id]
                    for node in leading_zero_nodes:
                        complete_layers.add(node.layer_id)
                        unfinished.remove(node.layer_id)
                    if forced_desc.required_xbar_count <= hardware.xbar_nums:
                        item, local_complete, local_unfinished = _plan_full_layer_pack(
                            len(items),
                            resource_round_index,
                            desc_by_id,
                            by_id,
                            complete_layers,
                            unfinished,
                            hardware.xbar_nums,
                            leading_zero_nodes=leading_zero_nodes,
                            forced_first_weighted=forced_first_weighted,
                        )
                        items.append(item)
                        complete_layers = local_complete
                        unfinished = local_unfinished
                        resource_round_index += 1
                    else:
                        item, new_cursor, _ = _plan_split_or_continuation(
                            len(items),
                            resource_round_index,
                            forced_desc,
                            0,
                            hardware.xbar_nums,
                            forced_first_weighted,
                            leading_zero_nodes=leading_zero_nodes,
                        )
                        items.append(item)
                        resource_round_index += 1
                        split_cursors[forced_first_weighted.layer_id] = new_cursor
                else:
                    zero_nodes = _consume_newly_unlocked_zero_xbar_closure_nodes(
                        by_id,
                        unfinished,
                        complete_layers,
                        ready_zero_before_final_continuation,
                    )
                    zero_round_index = _append_zero_xbar_item(
                        items,
                        zero_round_index,
                        zero_nodes,
                        hardware.xbar_nums,
                    )
            else:
                split_cursors[first.layer_id] = new_cursor
            continue

        item, local_complete, local_unfinished = _plan_full_layer_pack(
            len(items),
            resource_round_index,
            desc_by_id,
            by_id,
            complete_layers,
            unfinished,
            hardware.xbar_nums,
        )
        items.append(item)
        complete_layers = local_complete
        unfinished = local_unfinished
        resource_round_index += 1

    return items


def _populate_boundary_refs(model_ir: ModelIR, items: list[Any]) -> None:
    item_by_id = {item.item_id: item for item in items}
    producer_item_by_layer_id = _item_by_layer_id(items)
    consumer_items_by_layer_id = _items_by_layer_id(items)
    selected_items_by_layer_id = _selected_items_by_layer_id(items)
    for item in items:
        item.external_inputs = []
        item.external_outputs = []
        item.same_item_edges = []
        item.partial_output_contract = []

    for layer in model_ir.layers:
        consumer_items = consumer_items_by_layer_id.get(layer.layer_id, [])
        if not layer.output_consumer_ids:
            state_id = _state_id_for_layer(layer.layer_id)
            for producer_item in consumer_items:
                _add_boundary_ref(
                    producer_item.external_outputs,
                    state_id=state_id,
                    tensor_role="activation",
                    producer_item_id=producer_item.item_id,
                    consumer_item_id=None,
                    consumer_node_id=None,
                    partial_or_complete="complete",
                )
            _add_partial_output_contract_refs(
                selected_items_by_layer_id.get(layer.layer_id, []),
                producer_layer_id=layer.layer_id,
                state_id=state_id,
                consumer_item_id=None,
                consumer_node_id=None,
            )
        if not consumer_items:
            continue
        for consumer_item in consumer_items:
            if not layer.input_layer_ids or 'model_input' in layer.backend_attrs.get('merge_source_ids', []):
                _add_boundary_ref(
                    consumer_item.external_inputs,
                    state_id="model_input",
                    tensor_role="activation",
                    producer_item_id=None,
                    consumer_item_id=consumer_item.item_id,
                    consumer_node_id=layer.layer_id,
                    partial_or_complete="complete",
                )
                if not layer.input_layer_ids:
                    continue

            for producer_layer_id in layer.input_layer_ids:
                producer_item = producer_item_by_layer_id.get(producer_layer_id)
                if producer_item is None:
                    continue
                if producer_item.item_id == consumer_item.item_id:
                    _add_same_item_edge(
                        consumer_item,
                        producer_node_id=producer_layer_id,
                        consumer_node_id=layer.layer_id,
                    )
                    continue

                state_id = _state_id_for_layer(producer_layer_id)
                _add_boundary_ref(
                    consumer_item.external_inputs,
                    state_id=state_id,
                    tensor_role="activation",
                    producer_item_id=producer_item.item_id,
                    consumer_item_id=consumer_item.item_id,
                    consumer_node_id=layer.layer_id,
                    partial_or_complete="complete",
                )
                _add_boundary_ref(
                    producer_item.external_outputs,
                    state_id=state_id,
                    tensor_role="activation",
                    producer_item_id=producer_item.item_id,
                    consumer_item_id=consumer_item.item_id,
                    consumer_node_id=layer.layer_id,
                    partial_or_complete="complete",
                )
                _add_partial_output_contract_refs(
                    selected_items_by_layer_id.get(producer_layer_id, []),
                    producer_layer_id=producer_layer_id,
                    state_id=state_id,
                    consumer_item_id=consumer_item.item_id,
                    consumer_node_id=layer.layer_id,
                )

    for item in items:
        item.external_inputs.sort(key=lambda ref: ref.state_id)
        item.external_outputs.sort(key=lambda ref: ref.state_id)
        item.same_item_edges.sort(key=lambda edge: (edge.producer_node_id, edge.consumer_node_id))
        if item.item_id not in item_by_id:
            raise Phase1Invalid(INTERNAL_PLANNER_INVARIANT, f"missing item {item.item_id}")


def _item_by_layer_id(items: list[Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in items:
        for node in getattr(item, "local_graph_nodes", []) or []:
            result[node.layer_id] = item
    return result


def _items_by_layer_id(items: list[Any]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for item in items:
        seen_layer_ids: set[str] = set()
        for node in getattr(item, "local_graph_nodes", []) or []:
            if node.layer_id in seen_layer_ids:
                continue
            seen_layer_ids.add(node.layer_id)
            result.setdefault(node.layer_id, []).append(item)
    return result


def _selected_items_by_layer_id(items: list[Any]) -> dict[str, list[Any]]:
    result: dict[str, list[Any]] = {}
    for item in items:
        for entry in getattr(item, "selected_entries", []) or []:
            result.setdefault(entry.layer_id, []).append(item)
    return result


def _add_partial_output_contract_refs(
    selected_items: list[Any],
    *,
    producer_layer_id: str,
    state_id: str,
    consumer_item_id: str | None,
    consumer_node_id: str | None,
) -> None:
    if not _selected_items_include_partial_scope(selected_items, producer_layer_id):
        return
    for item in selected_items:
        _add_boundary_ref(
            item.partial_output_contract,
            state_id=state_id,
            tensor_role="activation",
            producer_item_id=item.item_id,
            consumer_item_id=consumer_item_id,
            consumer_node_id=consumer_node_id,
            partial_or_complete="complete",
        )


def _selected_items_include_partial_scope(selected_items: list[Any], layer_id: str) -> bool:
    for item in selected_items:
        round_kind = str(getattr(item, "round_kind", ""))
        for entry in getattr(item, "selected_entries", []) or []:
            if entry.layer_id != layer_id:
                continue
            if (
                not bool(entry.layer_complete_in_round)
                or round_kind in {
                    ROUND_KIND_SINGLE_LAYER_SPLIT,
                    ROUND_KIND_SINGLE_LAYER_CONTINUATION,
                }
            ):
                return True
    return False


def _add_same_item_edge(item: Any, producer_node_id: str, consumer_node_id: str) -> None:
    candidate = SameItemEdge(
        producer_node_id=str(producer_node_id),
        consumer_node_id=str(consumer_node_id),
    )
    for edge in item.same_item_edges:
        if (
            edge.producer_node_id == candidate.producer_node_id
            and edge.consumer_node_id == candidate.consumer_node_id
        ):
            return
    item.same_item_edges.append(candidate)


def _add_boundary_ref(
    refs: list[BoundaryRef],
    *,
    state_id: str,
    tensor_role: str,
    producer_item_id: str | None,
    consumer_item_id: str | None,
    consumer_node_id: str | None,
    partial_or_complete: str,
) -> None:
    for ref in refs:
        if (
            ref.state_id == state_id
            and ref.tensor_role == tensor_role
            and ref.producer_item_id == producer_item_id
            and ref.partial_or_complete == partial_or_complete
        ):
            if consumer_item_id is not None and consumer_item_id not in ref.consumer_item_ids:
                ref.consumer_item_ids.append(consumer_item_id)
                ref.consumer_item_ids.sort()
            if consumer_node_id is not None and consumer_node_id not in ref.consumer_node_ids:
                ref.consumer_node_ids.append(consumer_node_id)
                ref.consumer_node_ids.sort()
            return
    refs.append(
        BoundaryRef(
            state_id=state_id,
            tensor_role=tensor_role,
            producer_item_id=producer_item_id,
            consumer_item_ids=[] if consumer_item_id is None else [consumer_item_id],
            consumer_node_ids=[] if consumer_node_id is None else [consumer_node_id],
            partial_or_complete=partial_or_complete,
        )
    )


def _state_id_for_layer(layer_id: str) -> str:
    return f"state:{layer_id}"


def build_round_plan(model_ir: ModelIR, sim_config: SimConfigInput) -> RoundPlan:
    try:
        _validate_model_ir(model_ir)
        hardware = build_hardware_projection(sim_config)
        descriptors = build_layer_resource_descriptors(model_ir, hardware)
        execution_items = _plan_execution_items(model_ir, hardware, descriptors)
        plan = _empty_feasible_plan(model_ir, hardware, descriptors)
        plan.execution_items = execution_items
        _populate_boundary_refs(model_ir, plan.execution_items)
        validate_round_plan(plan)
        return plan
    except Phase1Invalid as exc:
        return _invalid_plan(model_ir, exc.reason)


__all__ = ["build_round_plan"]
