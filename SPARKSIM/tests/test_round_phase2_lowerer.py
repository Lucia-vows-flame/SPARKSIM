import configparser
from dataclasses import replace
import math

import pytest

import MNSIM.Scheduling.phase2.lowerer as lowerer_module
from MNSIM.Scheduling.device_precision import device_bits_from_level
from MNSIM.Scheduling.phase2.boundary import BoundaryStateManager
from MNSIM.Scheduling.phase2.lowerer import RoundLowerer
from MNSIM.Scheduling.phase2.tcg_adapter import serialize_round_placement
from MNSIM.Scheduling.phase2.round_local_state import RoundLocalState
from MNSIM.Scheduling.phase2.types import (
    CONTRIBUTION_COMPLETE_OUTPUT,
    CONTRIBUTION_PARTIAL,
    Phase2Invalid,
    StepExecutionResult,
    StepInputView,
)
from MNSIM.Scheduling.round_types import (
    BoundaryRef,
    LocalGraphNode,
    PrepassRef,
    ResourceProof,
    SameItemEdge,
    SelectedEntry,
    WeightedLayerSegment,
    WeightedPhysicalRound,
    ZeroXbarRound,
    ZeroXbarSegment,
)


def _sim_config_path(tmp_path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("SimConfig.ini", encoding="UTF-8")
    parser.set("Crossbar level", "Xbar_Size", "128,128")
    parser.set("Crossbar level", "Subarray_Size", "128")
    parser.set("Architecture level", "Tile_Num", "2,1")
    parser.set("Tile level", "PE_Num", "2,1")
    parser.set("Process element level", "Group_Num", "4")
    parser.set("Process element level", "Xbar_Polarity", "2")
    parser.set("Process element level", "DAC_Num", "128")
    parser.set("Process element level", "ADC_Num", "128")
    path = tmp_path / "SimConfig.round-lowerer.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return str(path)


def _one_bit_fragment_config_path(tmp_path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("SimConfig.ini", encoding="UTF-8")
    parser.set("Crossbar level", "Xbar_Size", "128,128")
    parser.set("Crossbar level", "Subarray_Size", "128")
    parser.set("Architecture level", "Tile_Num", "2,1")
    parser.set("Tile level", "PE_Num", "2,1")
    parser.set("Process element level", "Group_Num", "4")
    parser.set("Process element level", "Xbar_Polarity", "2")
    parser.set("Process element level", "DAC_Num", "128")
    parser.set("Process element level", "ADC_Num", "128")
    parser.set("Device level", "Device_Level", "2")
    parser.set("Device level", "Device_Resistance", "1000000,1000")
    parser.set("Algorithm Configuration", "Weight_precision", "8")
    path = tmp_path / "SimConfig.one-bit-fragment.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return str(path)


def _prepass(layer_id, layer_type, shape):
    return PrepassRef(
        ref_id=f"m0:{layer_id}:{layer_type}:mnsim20_physical_xbar_v2",
        model_id="m0",
        layer_id=layer_id,
        layer_type=layer_type,
        resource_version="mnsim20_physical_xbar_v2",
        sim_config_fingerprint="xbar=128x128;device=256;weight=8;polarity=2",
        layer_config_fingerprint=f"layer:{layer_id}",
        weight_ref_id=f"weight:{layer_id}",
        shape_signature=shape,
    )


def _conv_entry(layer_id):
    return SelectedEntry(
        layer_id=layer_id,
        layer_type="conv",
        xbar_range_start=0,
        xbar_range_end=2,
        used_xbars=2,
        layer_complete_in_round=True,
        prepass_ref=_prepass(
            layer_id,
            "conv",
            {
                "Inputchannel": 1,
                "Outputchannel": 1,
                "Kernelsize": 1,
                "Inputsize": [4, 4],
                "Outputsize": [4, 4],
                "Stride": 1,
                "Padding": 0,
                "Inputbit": 8,
                "Weightbit": 8,
                "outputbit": 8,
            },
        ),
    )


def _fc_split_entry():
    entry = SelectedEntry(
        layer_id="FC0",
        layer_type="fc",
        xbar_range_start=0,
        xbar_range_end=2,
        used_xbars=2,
        layer_complete_in_round=False,
        prepass_ref=_prepass(
            "FC0",
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
    return entry


def _diagonal_two_block_fc_entry():
    entry = SelectedEntry(
        layer_id="FC0",
        layer_type="fc",
        xbar_range_start=0,
        xbar_range_end=4,
        used_xbars=4,
        layer_complete_in_round=False,
        prepass_ref=_prepass(
            "FC0",
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
    return entry


def _one_bit_fc_entry(*, start, end, layer_complete_in_round):
    prepass = PrepassRef(
        ref_id="m0:FC0:mnsim20_physical_xbar_v2",
        model_id="m0",
        layer_id="FC0",
        layer_type="fc",
        resource_version="mnsim20_physical_xbar_v2",
        sim_config_fingerprint="xbar=128x128;device=2;weight=8;polarity=2",
        layer_config_fingerprint="fc:one-bit-fragment",
        weight_ref_id="weight:FC0",
        shape_signature={
            "Infeature": 128,
            "Outfeature": 128,
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        },
    )
    return SelectedEntry(
        layer_id="FC0",
        layer_type="fc",
        xbar_range_start=start,
        xbar_range_end=end,
        used_xbars=end - start,
        layer_complete_in_round=layer_complete_in_round,
        prepass_ref=prepass,
    )


def _required_xbar_count(entry):
    shape = entry.prepass_ref.shape_signature
    if entry.layer_type == "conv":
        input_count = int(shape["Inputchannel"])
        output_count = int(shape["Outputchannel"])
        input_block_size = 128 // (int(shape["Kernelsize"]) ** 2)
    else:
        input_count = int(shape["Infeature"])
        output_count = int(shape["Outfeature"])
        input_block_size = 128
    fingerprint = dict(
        part.split("=", 1)
        for part in entry.prepass_ref.sim_config_fingerprint.split(";")
    )
    device_level = int(fingerprint["device"])
    polarity = int(fingerprint["polarity"])
    cell_bits = device_bits_from_level(device_level)
    precision_bits = int(shape["Weightbit"]) - 1 if polarity == 2 else int(shape["Weightbit"])
    plane_count = math.ceil(precision_bits / cell_bits)
    return (
        math.ceil(input_count / input_block_size)
        * math.ceil(output_count / 128)
        * plane_count
        * polarity
    )


def _lowerer(tmp_path, *entries, sim_config_path=None):
    return RoundLowerer(
        (128, 128),
        sim_config_path=sim_config_path or _sim_config_path(tmp_path),
        required_xbars_by_layer={
            entry.layer_id: _required_xbar_count(entry)
            for entry in entries
        },
    )


def _input_view(item, state_id="model_input"):
    manager = BoundaryStateManager()
    manager.seed_model_input(state_id, {"Outputchannel": 1, "Outputsize": [4, 4]})
    return manager.build_input_view_for_state_ids(item.item_id, [state_id])


def _round(*, segments, entries, edges=None, round_kind="full_layer_pack"):
    first_segment = segments[0]
    first_node = getattr(first_segment, "local_graph_node", None)
    if first_node is None:
        first_node = first_segment.local_graph_nodes[0]
    return WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind=round_kind,
        selected_entries=entries,
        local_graph_nodes=[],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["R0"],
                consumer_node_ids=[first_node.layer_id],
            )
        ],
        same_item_edges=list(edges or []),
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=sum(entry.used_xbars for entry in entries), used_xbars=sum(entry.used_xbars for entry in entries)),
        ordered_segments=segments,
    )


def _zero_round(segment):
    node = segment.local_graph_nodes[0]
    return ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind=segment.zero_xbar_kind,
        selected_entries=[],
        local_graph_nodes=[],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["Z0"],
                consumer_node_ids=[node.layer_id],
            )
        ],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=0, used_xbars=0),
        ordered_segments=[segment],
    )


def _commit_local_step(local_state, item, segment_index, binding):
    from MNSIM.Scheduling.phase2.types import MetricTotals

    local_state.commit_step(
        StepExecutionResult(
            item_id=item.item_id,
            step_id=f"{item.item_id}.S{segment_index}",
            segment_index=segment_index,
            step_kind="test",
            source_kind="test",
            status="ok",
            origin_layer_ids=[binding.origin_layer_id],
            metrics=MetricTotals(0.0, 0.0, 0.0, 0.0),
            layer_results=[],
            mapping_summary={},
            raw_mnsim_audit=[],
            boundary_bindings=[binding],
            warnings=[],
        )
    )


def test_prepares_weighted_mapping_once_and_lowers_w_hook_w_in_order(tmp_path):
    conv0 = _conv_entry("Conv0")
    conv1 = _conv_entry("Conv1")
    relu = LocalGraphNode("Relu0", "relu", "hook")
    item = _round(
        segments=[
            WeightedLayerSegment(0, conv0, LocalGraphNode("Conv0", "conv", "weighted")),
            ZeroXbarSegment(1, "hook_only", [relu]),
            WeightedLayerSegment(2, conv1, LocalGraphNode("Conv1", "conv", "weighted")),
        ],
        entries=[conv0, conv1],
        edges=[SameItemEdge("Conv0", "Relu0"), SameItemEdge("Relu0", "Conv1")],
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path, conv0, conv1)

    prepared = lowerer.prepare_round(item, boundary_input)
    local_state = RoundLocalState.seed(item, boundary_input)
    first = lowerer.lower_step(prepared, item.ordered_segments[0], local_state.input_view_for(item.ordered_segments[0]))
    _commit_local_step(local_state, item, 0, first.boundary_bindings[0])
    middle = lowerer.lower_step(prepared, item.ordered_segments[1], local_state.input_view_for(item.ordered_segments[1]))
    _commit_local_step(local_state, item, 1, middle.boundary_bindings[0])
    last = lowerer.lower_step(prepared, item.ordered_segments[2], local_state.input_view_for(item.ordered_segments[2]))

    assert prepared.placement.mapping_summary["weighted_step_ids"] == ["R0.S0", "R0.S2"]
    assert prepared.placement.mapping_layer_ids_by_step == {"R0.S0": [0], "R0.S2": [1]}
    assert len(first.static_eval_requests) == len(first.placement_slices) == 1
    assert middle.static_eval_requests == []
    assert middle.placement_slices == []
    assert middle.hook_result is not None
    assert len(last.static_eval_requests) == len(last.placement_slices) == 1


def test_lowerer_restores_frozen_placement_without_replanning(tmp_path, monkeypatch):
    entry = _conv_entry("Conv0")
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("Conv0", "conv", "weighted"))],
        entries=[entry],
    )
    boundary_input = _input_view(item)
    initial = _lowerer(tmp_path, entry)
    prepared = initial.prepare_round(item, boundary_input)
    snapshot = serialize_round_placement(prepared.placement)

    def fail_if_planner_called(**_kwargs):
        raise AssertionError("Stage 2 attempted to re-plan a frozen placement")

    monkeypatch.setattr(lowerer_module, "prepare_round_placement", fail_if_planner_called)
    frozen = _lowerer(tmp_path, entry)
    frozen.placement_snapshots = {"R0": snapshot}
    restored = frozen.prepare_round(item, boundary_input)

    assert restored.placement.fragment_slots_by_id == prepared.placement.fragment_slots_by_id


def test_lowers_pooling_as_standalone_structural_work_outside_weighted_placement(tmp_path):
    conv0 = _conv_entry("Conv0")
    conv1 = _conv_entry("Conv1")
    pool = LocalGraphNode(
        "Pool0",
        "pooling",
        "structural_non_weighted",
        {"Inputchannel": 1, "Outputchannel": 1, "Inputsize": [4, 4], "Outputsize": [2, 2], "Kernelsize": 2, "Stride": 2},
    )
    item = _round(
        segments=[
            WeightedLayerSegment(0, conv0, LocalGraphNode("Conv0", "conv", "weighted")),
            ZeroXbarSegment(1, "structural_mnsim", [pool]),
            WeightedLayerSegment(2, conv1, LocalGraphNode("Conv1", "conv", "weighted")),
        ],
        entries=[conv0, conv1],
        edges=[SameItemEdge("Conv0", "Pool0"), SameItemEdge("Pool0", "Conv1")],
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path, conv0, conv1)
    prepared = lowerer.prepare_round(item, boundary_input)
    step = lowerer.lower_step(
        prepared,
        item.ordered_segments[1],
        StepInputView("R0", "R0.S1", boundary_input.input_view_id, [], [], ["local:R0:Conv0"], []),
    )

    assert prepared.placement.mapping_summary["weighted_step_ids"] == ["R0.S0", "R0.S2"]
    assert len(step.static_eval_requests) == 1
    assert step.static_eval_requests[0].item_kind == "zero_xbar_structural"
    assert step.static_eval_requests[0].netstruct[0][0][0]["type"] == "pooling"
    assert step.placement_slices == []


def test_public_lowerer_constructor_rejects_embedded_structural_policy(tmp_path):
    with pytest.raises(TypeError, match="embedded_structural_policy"):
        RoundLowerer(
            (128, 128),
            sim_config_path=_sim_config_path(tmp_path),
            required_xbars_by_layer={},
            embedded_structural_policy="hook",
        )


def test_public_legacy_lower_guard_explains_the_step_lowering_api(tmp_path):
    with pytest.raises(Phase2Invalid) as exc_info:
        _lowerer(tmp_path).lower(object(), object())

    assert exc_info.value.code == "LEGACY_ITEM_LOWERING_UNSUPPORTED"
    assert exc_info.value.detail == "use prepare_round() and lower_step() for ordered round execution"


def test_rejects_foreign_segment_with_matching_step_index(tmp_path):
    entry = _conv_entry("Conv0")
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("Conv0", "conv", "weighted"))],
        entries=[entry],
    )
    foreign_entry = _conv_entry("Foreign0")
    foreign_item = _round(
        segments=[WeightedLayerSegment(0, foreign_entry, LocalGraphNode("Foreign0", "conv", "weighted"))],
        entries=[foreign_entry],
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path, entry)
    prepared = lowerer.prepare_round(item, boundary_input)

    with pytest.raises(Phase2Invalid) as exc_info:
        lowerer.lower_step(
            prepared,
            foreign_item.ordered_segments[0],
            RoundLocalState.seed(item, boundary_input).input_view_for(item.ordered_segments[0]),
        )

    assert exc_info.value.code == "ROUND_SEGMENT_OWNERSHIP_MISMATCH"
    assert exc_info.value.detail == "R0.S0: segment is not owned by prepared item R0"


def test_rejects_step_view_from_a_different_boundary_view(tmp_path):
    entry = _conv_entry("Conv0")
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("Conv0", "conv", "weighted"))],
        entries=[entry],
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path, entry)
    prepared = lowerer.prepare_round(item, boundary_input)
    valid_view = RoundLocalState.seed(item, boundary_input).input_view_for(item.ordered_segments[0])

    with pytest.raises(Phase2Invalid) as exc_info:
        lowerer.lower_step(
            prepared,
            item.ordered_segments[0],
            replace(valid_view, boundary_input_view_id="view:other"),
        )

    assert exc_info.value.code == "STEP_INPUT_BOUNDARY_VIEW_MISMATCH"
    assert exc_info.value.detail == "R0.S0: expected boundary input view view:R0, got view:other"


def test_lowers_partial_fc_fragments_in_native_input_output_order(tmp_path):
    entry = _diagonal_two_block_fc_entry()
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("FC0", "fc", "weighted"))],
        entries=[entry],
        round_kind="single_layer_split",
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path, entry)

    prepared = lowerer.prepare_round(item, boundary_input)
    step = lowerer.lower_step(
        prepared,
        item.ordered_segments[0],
        RoundLocalState.seed(item, boundary_input).input_view_for(item.ordered_segments[0]),
    )

    assert prepared.placement.mapping_layer_ids_by_step == {"R0.S0": [0, 1]}
    assert prepared.placement.mapping_summary["phase1_used_xbars"] == entry.used_xbars
    assert len(step.static_eval_requests) == len(step.placement_slices) == 2
    assert [request.frame.boundary_bindings[0].input_range for request in step.static_eval_requests] == [
        [0, 128],
        [0, 128],
    ]
    assert [request.frame.boundary_bindings[0].output_range for request in step.static_eval_requests] == [
        [0, 128],
        [128, 256],
    ]
    assert [
        request.fragment_layout.fragments[0].weight_plane_index
        for request in step.static_eval_requests
    ] == [0, 0]
    assert [placement_slice.source_mapping_layer_ids for placement_slice in step.placement_slices] == [[0], [1]]


def test_lowers_partial_fc_step_to_one_synthetic_local_fc_and_partial_binding(tmp_path):
    entry = _fc_split_entry()
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("FC0", "fc", "weighted"))],
        entries=[entry],
        round_kind="single_layer_split",
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path, entry)

    step = lowerer.lower_step(
        lowerer.prepare_round(item, boundary_input),
        item.ordered_segments[0],
        RoundLocalState.seed(item, boundary_input).input_view_for(item.ordered_segments[0]),
    )

    layer = step.static_eval_requests[0].netstruct[0][0][0]
    assert layer["type"] == "fc"
    assert layer["Infeature"] == 128
    assert layer["Outfeature"] == 128
    assert step.boundary_bindings[0].contribution_kind == CONTRIBUTION_PARTIAL
    assert len(step.placement_slices) == 1


def test_lowers_one_bit_partial_fc_continuation_as_one_signed_group_fragment(tmp_path):
    entry = _one_bit_fc_entry(start=0, end=2, layer_complete_in_round=False)
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("FC0", "fc", "weighted"))],
        entries=[entry],
        round_kind="single_layer_continuation",
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(
        tmp_path,
        entry,
        sim_config_path=_one_bit_fragment_config_path(tmp_path),
    )

    step = lowerer.lower_step(
        lowerer.prepare_round(item, boundary_input),
        item.ordered_segments[0],
        RoundLocalState.seed(item, boundary_input).input_view_for(item.ordered_segments[0]),
    )

    assert len(step.static_eval_requests) == 1
    layout = step.static_eval_requests[0].fragment_layout
    assert layout.active_physical_xbar_count == 2
    assert layout.fragments[0].weight_plane_index == 0


def test_lowerer_binds_selected_and_required_fragment_ids_for_partial_fc(tmp_path):
    entry = _one_bit_fc_entry(start=0, end=2, layer_complete_in_round=False)
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("FC0", "fc", "weighted"))],
        entries=[entry],
        round_kind="single_layer_split",
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(
        tmp_path,
        entry,
        sim_config_path=_one_bit_fragment_config_path(tmp_path),
    )

    step = lowerer.lower_step(
        lowerer.prepare_round(item, boundary_input),
        item.ordered_segments[0],
        RoundLocalState.seed(item, boundary_input).input_view_for(item.ordered_segments[0]),
    )

    binding = step.boundary_bindings[0]
    assert binding.fragment_ids == ["FC0|in:0:128|out:0:128|plane:0"]
    assert binding.required_fragment_ids == [
        f"FC0|in:0:128|out:0:128|plane:{plane_index}"
        for plane_index in range(7)
    ]


def test_lowers_full_layer_pack_as_one_native_request_with_full_fragment_inventory(tmp_path):
    entry = _one_bit_fc_entry(start=0, end=14, layer_complete_in_round=True)
    item = _round(
        segments=[WeightedLayerSegment(0, entry, LocalGraphNode("FC0", "fc", "weighted"))],
        entries=[entry],
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(
        tmp_path,
        entry,
        sim_config_path=_one_bit_fragment_config_path(tmp_path),
    )

    step = lowerer.lower_step(
        lowerer.prepare_round(item, boundary_input),
        item.ordered_segments[0],
        RoundLocalState.seed(item, boundary_input).input_view_for(item.ordered_segments[0]),
    )

    assert len(step.static_eval_requests) == 1
    layer = step.static_eval_requests[0].netstruct[0][0][0]
    layout = step.static_eval_requests[0].fragment_layout
    assert layer["type"] == "fc"
    assert layer["Infeature"] == 128
    assert layer["Outfeature"] == 128
    assert layer["Weightbit"] == 8
    assert layout.active_physical_xbar_count == 14
    assert len(layout.fragments) == 7
    assert step.boundary_bindings[0].fragment_ids == [
        f"FC0|in:0:128|out:0:128|plane:{plane_index}"
        for plane_index in range(7)
    ]
    assert step.boundary_bindings[0].required_fragment_ids == [
        f"FC0|in:0:128|out:0:128|plane:{plane_index}"
        for plane_index in range(7)
    ]


def test_step_bindings_keep_boundary_and_local_sources_separate(tmp_path):
    conv = _conv_entry("Conv0")
    relu = LocalGraphNode("Relu0", "relu", "hook")
    item = _round(
        segments=[
            WeightedLayerSegment(0, conv, LocalGraphNode("Conv0", "conv", "weighted")),
            ZeroXbarSegment(1, "hook_only", [relu]),
        ],
        entries=[conv],
        edges=[SameItemEdge("Conv0", "Relu0")],
    )
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path, conv)
    prepared = lowerer.prepare_round(item, boundary_input)
    local_state = RoundLocalState.seed(item, boundary_input)
    weighted = lowerer.lower_step(prepared, item.ordered_segments[0], local_state.input_view_for(item.ordered_segments[0]))
    _commit_local_step(local_state, item, 0, weighted.boundary_bindings[0])
    hook = lowerer.lower_step(prepared, item.ordered_segments[1], local_state.input_view_for(item.ordered_segments[1]))

    assert weighted.boundary_bindings[0].source_state_ids == ["model_input"]
    assert weighted.boundary_bindings[0].source_local_state_ids == []
    assert hook.boundary_bindings[0].source_state_ids == []
    assert hook.boundary_bindings[0].source_local_state_ids == ["local:R0:Conv0"]


@pytest.mark.parametrize("layer_type", ["element_sum", "element_multiply"])
def test_direct_structural_step_preserves_all_input_arity(layer_type, tmp_path):
    node = LocalGraphNode(layer_type, layer_type, "structural_non_weighted")
    item = _zero_round(ZeroXbarSegment(0, "structural_mnsim", [node]))
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path)
    step = lowerer.lower_step(
        lowerer.prepare_round(item, boundary_input),
        item.ordered_segments[0],
        StepInputView("Z0", "Z0.S0", boundary_input.input_view_id, ["model_input"], [], ["local:Z0:other"], []),
    )

    assert len(step.static_eval_requests[0].netstruct[0][0][0]["Inputindex"]) == 2


def test_concat_lowers_as_zero_xbar_channel_routing(tmp_path):
    node = LocalGraphNode(
        "Concat0",
        "concat",
        "structural_non_weighted",
        {
            "Inputchannel": 12,
            "Outputchannel": 12,
            "Inputsize": [4, 4],
            "Outputsize": [4, 4],
            "Inputbit": 1,
            "outputbit": 1,
            "concat_dim": "channel",
            "merge_ownership_policy": "concat_channel_segments",
            "time_policy": "same_timestep",
            "merge_input_regions": [[0, 64], [64, 128], [128, 192]],
        },
    )
    item = _zero_round(ZeroXbarSegment(0, "structural_mnsim", [node]))
    boundary_input = _input_view(item)
    lowerer = _lowerer(tmp_path)

    step = lowerer.lower_step(
        lowerer.prepare_round(item, boundary_input),
        item.ordered_segments[0],
        StepInputView(
            "Z0", "Z0.S0", boundary_input.input_view_id,
            ["branch:a", "branch:b"], [], ["local:Z0:branch-c"], [],
        ),
    )

    layer_info = step.static_eval_requests[0].netstruct[0][0][0]
    assert layer_info["type"] == "concat"
    assert layer_info["Inputindex"] == [-1, -1, -1]
    assert layer_info["merge_ownership_policy"] == "concat_channel_segments"
    assert step.boundary_bindings[0].region["kind"] == "channel_concat"
    assert step.boundary_bindings[0].region["input_regions"] == [
        [0, 64], [64, 128], [128, 192]
    ]
    assert step.boundary_bindings[0].region["time_policy"] == "same_timestep"
    assert step.placement_slices == []
