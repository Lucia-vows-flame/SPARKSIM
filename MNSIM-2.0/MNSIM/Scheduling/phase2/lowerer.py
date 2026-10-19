"""Lower Phase 1 execution items into Phase 2 item-local work."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from MNSIM.Scheduling.phase2.mnsim_fragment_adapter import MnsimFragmentAdapter
from MNSIM.Scheduling.digital_domain import AccumulatorDomain, CompiledLifConstants
from MNSIM.Scheduling.phase2.digital_lif import DigitalLifRequest, InvocationKey
from MNSIM.Scheduling.phase2.netstruct_builder import build_netstruct
from MNSIM.Scheduling.phase2.tcg_adapter import (
    WeightedScope,
    deserialize_round_placement,
    prepare_round_placement,
)
from MNSIM.Scheduling.phase2.types import (
    CONTRIBUTION_COMPLETE_OUTPUT,
    CONTRIBUTION_HOOK_OUTPUT,
    CONTRIBUTION_PARTIAL,
    CONTRIBUTION_STRUCTURAL_OUTPUT,
    REPORT_ROLE_REPORTABLE,
    BoundaryBinding,
    BoundaryInputView,
    HookResult,
    LoweredStep,
    LoweredItem,
    MnsimFragmentLayout,
    Phase2Invalid,
    PreparedRound,
    StaticEvalFrame,
    StaticEvalRequest,
    StepInputView,
)
from MNSIM.Scheduling.round_types import (
    HOOK_LAYER_TYPES,
    ITEM_KIND_WEIGHTED,
    ITEM_KIND_ZERO_XBAR,
    NODE_ROLE_HOOK,
    ROUND_KIND_SINGLE_LAYER_CONTINUATION,
    ROUND_KIND_SINGLE_LAYER_SPLIT,
    LocalGraphNode,
    SelectedEntry,
    WeightedLayerSegment,
    ZeroXbarSegment,
    canonical_ordered_segments,
    step_id,
)


_STATIC_STRUCTURAL_TYPES = {"pooling", "concat", "element_sum", "element_multiply"}
_DEFAULT_SIM_CONFIG_PATH = "SimConfig.ini"


class RoundLowerer:
    def __init__(
        self,
        xbar_size: tuple[int, int],
        sim_config_path: str = _DEFAULT_SIM_CONFIG_PATH,
        required_xbars_by_layer: Mapping[str, int] | None = None,
        latency_api: str = "pipe",
        latency_fill_mode: int = 1,
        placement_snapshots: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        self.xbar_size = (int(xbar_size[0]), int(xbar_size[1]))
        self.sim_config_path = str(sim_config_path)
        if required_xbars_by_layer is None:
            raise TypeError("required_xbars_by_layer is required")
        self.fragment_adapter = MnsimFragmentAdapter(
            self.sim_config_path,
            required_xbars_by_layer,
        )
        self.latency_api = str(latency_api)
        self.latency_fill_mode = int(latency_fill_mode)
        self.placement_snapshots = {
            str(item_id): snapshot
            for item_id, snapshot in (placement_snapshots or {}).items()
        }

    def prepare_round(
        self,
        item: Any,
        boundary_input_view: BoundaryInputView,
    ) -> PreparedRound:
        """Map all weighted residency in a round once before step lowering."""
        if str(getattr(item, "item_kind", "")) == ITEM_KIND_ZERO_XBAR:
            return PreparedRound(
                item=item,
                boundary_input_view=boundary_input_view,
                placement=None,
            )
        if str(getattr(item, "item_kind", "")) != ITEM_KIND_WEIGHTED:
            raise Phase2Invalid(
                "UNSUPPORTED_EXECUTION_ITEM",
                f"unsupported execution item kind {getattr(item, 'item_kind', None)!r}",
            )

        weighted_scopes: list[WeightedScope] = []
        for segment in canonical_ordered_segments(item):
            if not isinstance(segment, WeightedLayerSegment):
                continue
            for scope_index, layout in enumerate(
                self._layouts_for_entry(item, segment.selected_entry)
            ):
                weighted_scopes.append(
                    WeightedScope(
                        step_id=step_id(item.item_id, segment.segment_index),
                        scope_index=scope_index,
                        layer_info=_weighted_layer_info(segment.selected_entry, layout),
                        fragment_layout=layout,
                    )
                )

        frozen_snapshot = self.placement_snapshots.get(str(item.item_id))
        if frozen_snapshot is not None:
            placement = deserialize_round_placement(
                frozen_snapshot,
                sim_config_path=self.sim_config_path,
            )
            if placement.item_id != str(item.item_id):
                raise Phase2Invalid(
                    "PLACEMENT_SNAPSHOT_ITEM_MISMATCH",
                    f"snapshot item {placement.item_id!r} does not match {item.item_id!r}",
                )
        else:
            placement = prepare_round_placement(
                item_id=item.item_id,
                sim_config_path=self.sim_config_path,
                weighted_scopes=weighted_scopes,
                expected_used_xbars=item.resource_proof.used_xbars,
            )
        return PreparedRound(
            item=item,
            boundary_input_view=boundary_input_view,
            placement=placement,
        )

    def _layouts_for_entry(
        self,
        item: Any,
        entry: SelectedEntry,
    ) -> list[MnsimFragmentLayout]:
        selected_layout = self.fragment_adapter.select_entry(entry)
        if not _is_partial_entry(item, entry):
            return [selected_layout]
        return [
            _single_fragment_layout(selected_layout, fragment_index)
            for fragment_index in range(len(selected_layout.fragments))
        ]

    def lower_step(
        self,
        prepared: PreparedRound,
        segment: WeightedLayerSegment | ZeroXbarSegment,
        input_view: StepInputView,
        *,
        sample_id: str = "sample:0",
        timestep: int = 0,
        sequence_length: int = 1,
    ) -> LoweredStep:
        expected_step_id = step_id(prepared.item.item_id, segment.segment_index)
        if not any(segment is owned_segment for owned_segment in prepared.item.ordered_segments):
            raise Phase2Invalid(
                "ROUND_SEGMENT_OWNERSHIP_MISMATCH",
                f"{expected_step_id}: segment is not owned by prepared item {prepared.item.item_id}",
            )
        if str(input_view.boundary_input_view_id) != str(prepared.boundary_input_view.input_view_id):
            raise Phase2Invalid(
                "STEP_INPUT_BOUNDARY_VIEW_MISMATCH",
                (
                    f"{expected_step_id}: expected boundary input view "
                    f"{prepared.boundary_input_view.input_view_id}, got "
                    f"{input_view.boundary_input_view_id}"
                ),
            )
        if str(input_view.item_id) != str(prepared.item.item_id):
            raise Phase2Invalid(
                "STEP_INPUT_ITEM_MISMATCH",
                f"{expected_step_id}: input view belongs to {input_view.item_id}",
            )
        if str(input_view.step_id) != expected_step_id:
            raise Phase2Invalid(
                "STEP_INPUT_ID_MISMATCH",
                f"expected {expected_step_id}, got {input_view.step_id}",
            )
        if isinstance(segment, WeightedLayerSegment):
            return self._lower_weighted_step(
                prepared,
                segment,
                input_view,
                sample_id=sample_id,
                timestep=timestep,
                sequence_length=sequence_length,
            )
        if isinstance(segment, ZeroXbarSegment):
            return self._lower_zero_xbar_step(
                prepared,
                segment,
                input_view,
                sample_id=sample_id,
                timestep=timestep,
                sequence_length=sequence_length,
            )
        raise Phase2Invalid(
            "UNSUPPORTED_ROUND_SEGMENT",
            f"{prepared.item.item_id}: unsupported segment {segment!r}",
        )

    def _lower_weighted_step(
        self,
        prepared: PreparedRound,
        segment: WeightedLayerSegment,
        input_view: StepInputView,
        *,
        sample_id: str,
        timestep: int,
        sequence_length: int,
    ) -> LoweredStep:
        if prepared.placement is None:
            raise Phase2Invalid(
                "MISSING_ROUND_PLACEMENT",
                f"{prepared.item.item_id}: weighted step has no resident placement",
            )
        item = prepared.item
        entry = segment.selected_entry
        current_step_id = step_id(item.item_id, segment.segment_index)
        required_fragment_ids = self._required_fragment_ids_for_entry(entry)
        requests: list[StaticEvalRequest] = []
        placement_slices: list[Any] = []
        bindings: list[BoundaryBinding] = []
        for scope_index, layout in enumerate(self._layouts_for_entry(item, entry)):
            scope = _layout_scope(layout)
            layer_info = _weighted_layer_info(entry, layout)
            layer_info["Layerindex"] = 0
            layer_info["Inputindex"] = [-1]
            layer_info["Outputindex"] = []
            contribution_kind = (
                CONTRIBUTION_PARTIAL
                if _is_partial_entry(item, entry)
                else CONTRIBUTION_COMPLETE_OUTPUT
            )
            binding = _boundary_binding(
                local_layer_id=0,
                hook_node_id=None,
                origin_layer_id=entry.layer_id,
                origin_layer_type=entry.layer_type,
                origin_layer_shape=_weighted_output_shape(entry, scope),
                region={"kind": "materialized_scope", "scope": deepcopy(scope)},
                input_range=scope["input_range"],
                output_range=scope["output_range"],
                full_input_range=scope["full_input_range"],
                full_output_range=scope["full_output_range"],
                contribution_kind=contribution_kind,
                source_state_ids=list(input_view.boundary_state_ids),
                source_local_state_ids=list(input_view.local_state_ids),
                downstream_consumer_ids=_downstream_consumer_ids(item, entry.layer_id),
                fragment_ids=[fragment.fragment_id for fragment in layout.fragments],
                required_fragment_ids=required_fragment_ids,
                sample_id=sample_id,
                timestep=timestep,
                sequence_length=sequence_length,
                data_bitwidth=_node_accumulator_width(segment.local_graph_node),
                accumulator_domain=_node_accumulator_domain_dict(segment.local_graph_node),
                payload_kind=(
                    "partial_sum_sequence"
                    if contribution_kind == CONTRIBUTION_PARTIAL
                    else "activation_sequence"
                ),
            )
            request = self._single_local_request(
                item=item,
                item_kind="weighted_layer_segment",
                layer_info=layer_info,
                binding=binding,
                fragment_layout=layout,
            )
            requests.append(request)
            bindings.append(binding)
            placement_slices.append(
                prepared.placement.slice_for(current_step_id, scope_index, request.netstruct)
            )
        return LoweredStep(
            item_id=item.item_id,
            step_id=current_step_id,
            segment_index=segment.segment_index,
            step_kind=segment.segment_kind,
            static_eval_requests=requests,
            placement_slices=placement_slices,
            hook_result=None,
            boundary_bindings=bindings,
            warnings=[],
        )

    def _required_fragment_ids_for_entry(self, entry: SelectedEntry) -> list[str]:
        return [
            fragment.fragment_id
            for fragment in self.fragment_adapter.inventory_for(entry).fragments
        ]

    def _lower_zero_xbar_step(
        self,
        prepared: PreparedRound,
        segment: ZeroXbarSegment,
        input_view: StepInputView,
        *,
        sample_id: str,
        timestep: int,
        sequence_length: int,
    ) -> LoweredStep:
        item = prepared.item
        current_step_id = step_id(item.item_id, segment.segment_index)
        nodes = list(segment.local_graph_nodes)
        if len(nodes) != 1:
            raise Phase2Invalid(
                "AMBIGUOUS_ZERO_XBAR_SEGMENT",
                f"{current_step_id}: expected one zero-xbar node, got {len(nodes)}",
            )
        node = nodes[0]
        if str(node.layer_type) == "lif":
            binding = _boundary_binding(
                local_layer_id=None,
                hook_node_id=node.layer_id,
                origin_layer_id=node.layer_id,
                origin_layer_type=node.layer_type,
                origin_layer_shape=_node_shape_or_fallback(
                    node, _step_input_shape(input_view)
                ),
                region={"kind": "digital_lif"},
                input_range={"kind": "full"},
                output_range={"kind": "full"},
                full_input_range={"kind": "full"},
                full_output_range={"kind": "full"},
                contribution_kind=CONTRIBUTION_HOOK_OUTPUT,
                source_state_ids=list(input_view.boundary_state_ids),
                source_local_state_ids=list(input_view.local_state_ids),
                downstream_consumer_ids=_downstream_consumer_ids(item, node.layer_id),
                sample_id=sample_id,
                timestep=timestep,
                sequence_length=sequence_length,
                data_bitwidth=_node_accumulator_width(node),
                accumulator_domain=_node_accumulator_domain_dict(node),
                payload_kind="spike_sequence",
            )
            domain = AccumulatorDomain.from_dict(
                _node_accumulator_domain_dict(node)
            )
            constants = CompiledLifConstants.from_dict(
                dict(node.digital_metadata["compiled_lif_constants"])
            )
            neuron_count = _lif_neuron_count(node)
            return LoweredStep(
                item_id=item.item_id,
                step_id=current_step_id,
                segment_index=segment.segment_index,
                step_kind=segment.segment_kind,
                static_eval_requests=[],
                placement_slices=[],
                hook_result=None,
                boundary_bindings=[binding],
                warnings=[],
                digital_lif_request=DigitalLifRequest(
                    invocation=InvocationKey(sample_id, node.layer_id, timestep),
                    sequence_length=sequence_length,
                    input_codes=(),
                    previous_state_codes=(),
                    domain=domain,
                    constants=constants,
                    neuron_count=neuron_count,
                ),
            )
        if _is_hook_node(node):
            binding = _boundary_binding(
                local_layer_id=None,
                hook_node_id=node.layer_id,
                origin_layer_id=node.layer_id,
                origin_layer_type=node.layer_type,
                origin_layer_shape=_node_shape_or_fallback(node, _step_input_shape(input_view)),
                region={"kind": "hook_only"},
                input_range={"kind": "full"},
                output_range={"kind": "full"},
                full_input_range={"kind": "full"},
                full_output_range={"kind": "full"},
                contribution_kind=CONTRIBUTION_HOOK_OUTPUT,
                source_state_ids=list(input_view.boundary_state_ids),
                source_local_state_ids=list(input_view.local_state_ids),
                downstream_consumer_ids=_downstream_consumer_ids(item, node.layer_id),
                sample_id=sample_id,
                timestep=timestep,
                sequence_length=sequence_length,
                data_bitwidth=_node_accumulator_width(node),
                accumulator_domain=_node_accumulator_domain_dict(node),
                payload_kind="activation_sequence",
            )
            return LoweredStep(
                item_id=item.item_id,
                step_id=current_step_id,
                segment_index=segment.segment_index,
                step_kind=segment.segment_kind,
                static_eval_requests=[],
                placement_slices=[],
                hook_result=HookResult(
                    item_id=item.item_id,
                    item_kind=item.item_kind,
                    hook_bindings=[binding],
                    warnings=[],
                ),
                boundary_bindings=[binding],
                warnings=[],
            )
        if str(node.layer_type) not in _STATIC_STRUCTURAL_TYPES:
            raise Phase2Invalid(
                "UNSUPPORTED_LAYER_TYPE",
                f"unsupported zero-xbar layer type {node.layer_type!r}",
            )

        input_count = len(input_view.boundary_state_ids) + len(input_view.local_state_ids)
        input_indices = [-1] if str(node.layer_type) == "pooling" else [-1] * input_count
        layer_info = _structural_layer_info(
            node=node,
            shape=_node_shape_or_fallback(node, _step_input_shape(input_view)),
            local_layer_id=0,
            input_indices=input_indices,
        )
        if str(node.layer_type) == "concat":
            structural_region = {
                "kind": "channel_concat",
                "concat_dim": deepcopy(layer_info.get("concat_dim", 1)),
                "input_regions": deepcopy(layer_info.get("merge_input_regions", [])),
                "ownership_policy": str(
                    layer_info.get(
                        "merge_ownership_policy", "concat_channel_segments"
                    )
                ),
                "time_policy": str(layer_info.get("time_policy", "same_timestep")),
            }
        else:
            structural_region = {"kind": "standalone_structural"}
        binding = _boundary_binding(
            local_layer_id=0,
            hook_node_id=None,
            origin_layer_id=node.layer_id,
            origin_layer_type=node.layer_type,
            origin_layer_shape=_structural_output_shape(layer_info),
            region=structural_region,
            input_range={"kind": "full"},
            output_range={"kind": "full"},
            full_input_range={"kind": "full"},
            full_output_range={"kind": "full"},
            contribution_kind=CONTRIBUTION_STRUCTURAL_OUTPUT,
            source_state_ids=list(input_view.boundary_state_ids),
            source_local_state_ids=list(input_view.local_state_ids),
            downstream_consumer_ids=_downstream_consumer_ids(item, node.layer_id),
            sample_id=sample_id,
            timestep=timestep,
            sequence_length=sequence_length,
            data_bitwidth=_node_accumulator_width(node),
            accumulator_domain=_node_accumulator_domain_dict(node),
            payload_kind="activation_sequence",
        )
        request = self._single_local_request(
            item=item,
            item_kind="zero_xbar_structural",
            layer_info=layer_info,
            binding=binding,
        )
        return LoweredStep(
            item_id=item.item_id,
            step_id=current_step_id,
            segment_index=segment.segment_index,
            step_kind=segment.segment_kind,
            static_eval_requests=[request],
            placement_slices=[],
            hook_result=None,
            boundary_bindings=[binding],
            warnings=[],
        )

    def _single_local_request(
        self,
        *,
        item: Any,
        item_kind: str,
        layer_info: dict[str, Any],
        binding: BoundaryBinding,
        fragment_layout: MnsimFragmentLayout | None = None,
    ) -> StaticEvalRequest:
        return StaticEvalRequest(
            item_id=item.item_id,
            item_kind=item_kind,
            netstruct=build_netstruct([layer_info]),
            sim_config_path=self.sim_config_path,
            multiple=[1],
            latency_api=self.latency_api,
            latency_fill_mode=self.latency_fill_mode,
            frame=StaticEvalFrame(
                context_layer_ids=[],
                reportable_layer_ids=[0],
                latency_offset_source_ids=[],
                boundary_bindings=[binding],
            ),
            fragment_layout=fragment_layout,
        )

    def lower(self, item: Any, input_view: BoundaryInputView) -> LoweredItem:
        raise Phase2Invalid(
            "LEGACY_ITEM_LOWERING_UNSUPPORTED",
            "use prepare_round() and lower_step() for ordered round execution",
        )


def _is_partial_entry(item: Any, entry: SelectedEntry) -> bool:
    round_kind = str(getattr(item, "round_kind", ""))
    return (
        not bool(entry.layer_complete_in_round)
        or round_kind in {ROUND_KIND_SINGLE_LAYER_SPLIT, ROUND_KIND_SINGLE_LAYER_CONTINUATION}
    )


def _single_fragment_layout(
    layout: MnsimFragmentLayout,
    fragment_index: int,
) -> MnsimFragmentLayout:
    try:
        fragment = layout.fragments[fragment_index]
    except IndexError as exc:
        raise Phase2Invalid(
            "EMPTY_SELECTED_FRAGMENT_LAYOUT",
            f"{layout.layer_id}: selected layout has no fragment {fragment_index}",
        ) from exc
    return MnsimFragmentLayout(
        layer_id=layout.layer_id,
        layer_type=layout.layer_type,
        fragments=(fragment,),
        native_layer_info=layout.native_layer_info,
        active_group_count=1,
        active_physical_xbar_count=fragment.physical_xbar_count,
    )


def _layout_scope(layout: MnsimFragmentLayout) -> dict[str, list[int]]:
    if not layout.fragments:
        raise Phase2Invalid(
            "EMPTY_SELECTED_FRAGMENT_LAYOUT",
            f"{layout.layer_id}: selected layout contains no fragments",
        )
    first_fragment = layout.fragments[0]
    if len(layout.fragments) == 1:
        input_range = first_fragment.input_range
        output_range = first_fragment.output_range
    else:
        input_range = first_fragment.full_input_range
        output_range = first_fragment.full_output_range
    return {
        "input_range": list(input_range),
        "output_range": list(output_range),
        "full_input_range": list(first_fragment.full_input_range),
        "full_output_range": list(first_fragment.full_output_range),
    }


def _weighted_layer_info(
    entry: SelectedEntry,
    layout: MnsimFragmentLayout,
) -> dict[str, Any]:
    shape = layout.to_dict()["native_layer_info"]
    scope = _layout_scope(layout)
    layer_type = "fc" if entry.layer_type == "fc_like_projection" else entry.layer_type
    layer_info = deepcopy(shape)
    layer_info["type"] = layer_type
    if layer_type == "conv":
        layer_info["Inputchannel"] = _range_len(scope["input_range"])
        layer_info["Outputchannel"] = _range_len(scope["output_range"])
        layer_info.setdefault("Kernelsize", 1)
        layer_info.setdefault("Inputsize", [1, 1])
        layer_info.setdefault("Outputsize", _output_size_from_input(layer_info))
        layer_info.setdefault("Stride", 1)
        layer_info.setdefault("Padding", 0)
    elif layer_type == "fc":
        layer_info["Infeature"] = _range_len(scope["input_range"])
        layer_info["Outfeature"] = _range_len(scope["output_range"])
        layer_info["Inputsize"] = deepcopy(layer_info.get("Inputsize", [1, 1]))
        layer_info["Outputsize"] = deepcopy(layer_info.get("Outputsize", [1, 1]))
        layer_info["Inputchannel"] = int(layer_info.get("Inputchannel", 1))
        layer_info["Outputchannel"] = _range_len(scope["output_range"])
        layer_info["Kernelsize"] = int(layer_info.get("Kernelsize", 1))
        layer_info["Stride"] = int(layer_info.get("Stride", 1))
        layer_info["Padding"] = int(layer_info.get("Padding", 0))
    else:
        raise Phase2Invalid("UNSUPPORTED_LAYER_TYPE", f"unsupported weighted layer type {entry.layer_type!r}")
    layer_info.setdefault("Inputbit", 8)
    layer_info["Weightbit"] = int(layer_info["Weightbit"])
    layer_info.setdefault("outputbit", 8)
    return layer_info


def _weighted_output_shape(entry: SelectedEntry, scope: dict[str, Any]) -> dict[str, Any]:
    shape = entry.prepass_ref.shape_signature
    if entry.layer_type == "conv":
        return {
            "Outputchannel": _range_len(scope["output_range"]),
            "Outputsize": deepcopy(shape.get("Outputsize", _output_size_from_shape(shape))),
            "outputbit": int(shape.get("outputbit", 8)),
        }
    return {
        "Outfeature": _range_len(scope["output_range"]),
        "outputbit": int(shape.get("outputbit", 8)),
    }


def _structural_layer_info(
    *,
    node: LocalGraphNode,
    shape: dict[str, Any],
    local_layer_id: int,
    input_indices: list[int],
) -> dict[str, Any]:
    layer_type = str(node.layer_type)
    if layer_type not in _STATIC_STRUCTURAL_TYPES:
        raise Phase2Invalid("UNSUPPORTED_LAYER_TYPE", f"unsupported structural type {layer_type!r}")
    input_size = deepcopy(shape.get("Inputsize", shape.get("Outputsize", [1, 1])))
    output_size = deepcopy(shape.get("Outputsize", input_size))
    inputchannel = int(shape.get("Inputchannel", shape.get("Outputchannel", shape.get("C", 1))))
    outputchannel = int(shape.get("Outputchannel", inputchannel))
    outputbit = int(shape.get("outputbit", shape.get("Outputbit", shape.get("Inputbit", 8))))
    layer_info = {
        "type": layer_type,
        "Layerindex": local_layer_id,
        "Inputindex": list(input_indices),
        "Outputindex": [],
        "Inputsize": input_size,
        "Outputsize": output_size,
        "Kernelsize": int(shape.get("Kernelsize", 1 if layer_type != "pooling" else 2)),
        "Stride": int(shape.get("Stride", 1 if layer_type != "pooling" else 2)),
        "Padding": int(shape.get("Padding", 0)),
        "Inputchannel": inputchannel,
        "Outputchannel": outputchannel,
        "Inputbit": int(shape.get("Inputbit", outputbit)),
        "Weightbit": int(shape.get("Weightbit", 8)),
        "outputbit": outputbit,
    }
    if layer_type == "concat":
        # Concat is a zero-crossbar channel-routing step.  Its output channel
        # count is already materialized by NetworkGraph.structure_forward;
        # each input is represented by one same-timestep Inputindex.
        layer_info["concat_dim"] = shape.get("concat_dim", 1)
        layer_info["merge_ownership_policy"] = shape.get(
            "merge_ownership_policy", "concat_channel_segments"
        )
        layer_info["time_policy"] = shape.get("time_policy", "same_timestep")
        layer_info["merge_input_regions"] = deepcopy(shape.get("merge_input_regions", []))
        layer_info["Inputchannel"] = outputchannel

    return layer_info


def _boundary_binding(
    *,
    local_layer_id: int | None,
    hook_node_id: str | None,
    origin_layer_id: str,
    origin_layer_type: str,
    origin_layer_shape: dict[str, Any],
    region: dict[str, Any],
    input_range: Any,
    output_range: Any,
    full_input_range: Any,
    full_output_range: Any,
    contribution_kind: str,
    source_state_ids: list[str],
    downstream_consumer_ids: list[str],
    source_local_state_ids: list[str] | None = None,
    fragment_ids: list[str] | None = None,
    required_fragment_ids: list[str] | None = None,
    sample_id: str = "sample:0",
    timestep: int = 0,
    sequence_length: int = 1,
    data_bitwidth: int | None = None,
    accumulator_domain: dict[str, Any] | None = None,
    payload_kind: str = "activation",
) -> BoundaryBinding:
    return BoundaryBinding(
        local_layer_id=local_layer_id,
        hook_node_id=hook_node_id,
        origin_layer_id=str(origin_layer_id),
        origin_layer_type=str(origin_layer_type),
        report_role=REPORT_ROLE_REPORTABLE,
        tensor_role="activation",
        shape=deepcopy(origin_layer_shape),
        region=deepcopy(region),
        input_range=deepcopy(input_range),
        output_range=deepcopy(output_range),
        full_input_range=deepcopy(full_input_range),
        full_output_range=deepcopy(full_output_range),
        contribution_kind=contribution_kind,
        source_state_ids=list(source_state_ids),
        downstream_consumer_ids=list(downstream_consumer_ids),
        source_local_state_ids=list(source_local_state_ids or []),
        fragment_ids=list(fragment_ids or []),
        required_fragment_ids=list(required_fragment_ids or []),
        sample_id=str(sample_id),
        timestep=int(timestep),
        sequence_length=int(sequence_length),
        data_bitwidth=data_bitwidth,
        accumulator_domain=deepcopy(accumulator_domain or {}),
        payload_kind=str(payload_kind),
    )


def _node_accumulator_domain_dict(node: LocalGraphNode) -> dict[str, Any]:
    value = dict(getattr(node, "digital_metadata", {}) or {}).get(
        "accumulator_domain"
    )
    return deepcopy(dict(value or {}))


def _node_accumulator_width(node: LocalGraphNode) -> int | None:
    value = _node_accumulator_domain_dict(node).get("signed_bit_width")
    return int(value) if value is not None else None


def _lif_neuron_count(node: LocalGraphNode) -> int:
    region = dict(getattr(node, "node_shape", {}) or {}).get(
        "output_neuron_region"
    )
    if isinstance(region, (list, tuple)) and len(region) == 2:
        count = int(region[1]) - int(region[0])
        if count > 0:
            return count
    shape = dict(getattr(node, "node_shape", {}) or {})
    for key in ("Outfeature", "Outputchannel", "C"):
        if key in shape and int(shape[key]) > 0:
            return int(shape[key])
    raise Phase2Invalid(
        "INVALID_LIF_NEURON_REGION",
        f"LIF node {node.layer_id} has no positive output-neuron region",
    )


def _is_hook_node(node: LocalGraphNode) -> bool:
    return str(node.node_role) == NODE_ROLE_HOOK or str(node.layer_type) in HOOK_LAYER_TYPES


def _downstream_consumer_ids(item: Any, origin_layer_id: str | None = None) -> list[str]:
    consumers: list[str] = []
    expected_state_ids = _state_ids_for_origin(origin_layer_id)
    for ref in list(getattr(item, "external_outputs", []) or []) + list(
        getattr(item, "partial_output_contract", []) or []
    ):
        if expected_state_ids and str(getattr(ref, "state_id", "")) not in expected_state_ids:
            continue
        for consumer_id in getattr(ref, "consumer_item_ids", []) or []:
            if consumer_id not in consumers:
                consumers.append(str(consumer_id))
    return consumers


def _state_ids_for_origin(origin_layer_id: str | None) -> set[str]:
    if origin_layer_id is None:
        return set()
    return {
        f"state:{origin_layer_id}",
        f"state:{origin_layer_id}:partial",
    }


def _step_input_shape(input_view: StepInputView) -> dict[str, Any]:
    if input_view.local_state_records:
        return deepcopy(input_view.local_state_records[0].shape)
    if input_view.boundary_state_records:
        return deepcopy(input_view.boundary_state_records[0].shape)
    return _default_activation_shape()


def _default_activation_shape() -> dict[str, Any]:
    return {"Outputchannel": 1, "Outputsize": [1, 1], "outputbit": 8}


def _structural_output_shape(layer_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "Outputchannel": int(layer_info.get("Outputchannel", 1)),
        "Outputsize": deepcopy(layer_info.get("Outputsize", [1, 1])),
        "outputbit": int(layer_info.get("outputbit", 8)),
    }


def _node_shape_or_fallback(node: LocalGraphNode, fallback: dict[str, Any]) -> dict[str, Any]:
    node_shape = getattr(node, "node_shape", {}) or {}
    if not node_shape:
        return deepcopy(fallback)
    merged = deepcopy(fallback)
    merged.update(deepcopy(node_shape))
    return merged


def _range_len(value: Any) -> int:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise Phase2Invalid("INVALID_MATERIALIZED_SCOPE", f"invalid range {value!r}")
    return int(value[1]) - int(value[0])


def _output_size_from_shape(shape: dict[str, Any]) -> list[int]:
    if "Outputsize" in shape:
        return deepcopy(shape["Outputsize"])
    return _output_size_from_input(shape)


def _output_size_from_input(shape: dict[str, Any]) -> list[int]:
    if "Inputsize" not in shape:
        return [1, 1]
    input_size = list(shape["Inputsize"])
    kernel = int(shape.get("Kernelsize", 1))
    stride = int(shape.get("Stride", 1))
    padding = int(shape.get("Padding", 0))
    return [
        max(1, (int(input_size[0]) + 2 * padding - kernel) // stride + 1),
        max(1, (int(input_size[1]) + 2 * padding - kernel) // stride + 1),
    ]


__all__ = ["RoundLowerer"]
