import pytest

import MNSIM.Scheduling.phase2 as phase2
from MNSIM.Scheduling.phase2.types import (
    CONTRIBUTION_COMPLETE_OUTPUT,
    CONTRIBUTION_HOOK_OUTPUT,
    CONTRIBUTION_PARTIAL,
    STATIC_STATUS_OK,
    BoundaryBinding,
    BoundaryInputView,
    BoundaryStateRecord,
    MetricTotals,
    Phase2Invalid,
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
    ZeroXbarSegment,
)


def _entry(layer_id):
    return SelectedEntry(
        layer_id=layer_id,
        layer_type="conv",
        xbar_range_start=0,
        xbar_range_end=1,
        used_xbars=1,
        layer_complete_in_round=True,
        prepass_ref=PrepassRef(
            ref_id=f"m0:{layer_id}",
            model_id="m0",
            layer_id=layer_id,
            layer_type="conv",
            resource_version="test",
            sim_config_fingerprint="test",
            layer_config_fingerprint="test",
            weight_ref_id=f"weight:{layer_id}",
            shape_signature={"C": 8},
        ),
    )


def _wzw_item(*, external_output=False):
    conv1_entry = _entry("Conv1")
    conv2_entry = _entry("Conv2")
    conv1_node = LocalGraphNode("Conv1", "conv", "weighted", {"C": 8})
    relu1_node = LocalGraphNode("Relu1", "relu", "hook", {"C": 8})
    conv2_node = LocalGraphNode("Conv2", "conv", "weighted", {"C": 8})
    return WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[conv1_entry, conv2_entry],
        local_graph_nodes=[conv1_node, relu1_node, conv2_node],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["R0"],
                consumer_node_ids=["Conv1"],
            )
        ],
        same_item_edges=[
            SameItemEdge("Conv1", "Relu1"),
            SameItemEdge("Relu1", "Conv2"),
        ],
        external_outputs=(
            [
                BoundaryRef(
                    state_id="state:Conv2",
                    tensor_role="activation",
                    producer_item_id="R0",
                )
            ]
            if external_output
            else []
        ),
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=2, used_xbars=2),
        ordered_segments=[
            WeightedLayerSegment(0, conv1_entry, conv1_node),
            ZeroXbarSegment(1, "hook_only", [relu1_node]),
            WeightedLayerSegment(2, conv2_entry, conv2_node),
        ],
    )


def _boundary_view(item_id, state_id):
    record = BoundaryStateRecord(
        state_id=state_id,
        producer_item_id="model_input",
        producer_layer_id="model_input",
        tensor_role="activation",
        shape={"C": 3},
        region={"kind": "full"},
        status="model_input",
        storage_backend="model_input",
        allowed_consumer_item_ids=[],
        lifetime_policy="inference_input",
        source_state_ids=[],
    )
    return BoundaryInputView(
        input_view_id=f"view:{item_id}",
        item_id=item_id,
        readable_state_ids=[state_id],
        state_bindings={"model_input": [state_id]},
        warnings=[],
        state_records=[record],
    )


def _binding(
    origin_layer_id,
    origin_layer_type,
    contribution_kind,
    *,
    source_state_ids=None,
    source_local_state_ids=None,
    fragment_ids=None,
    required_fragment_ids=None,
):
    return BoundaryBinding(
        local_layer_id=None if contribution_kind == CONTRIBUTION_HOOK_OUTPUT else 0,
        hook_node_id=origin_layer_id if contribution_kind == CONTRIBUTION_HOOK_OUTPUT else None,
        origin_layer_id=origin_layer_id,
        origin_layer_type=origin_layer_type,
        report_role="reportable",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind=contribution_kind,
        source_state_ids=list(source_state_ids or []),
        downstream_consumer_ids=[],
        source_local_state_ids=list(source_local_state_ids or []),
        fragment_ids=list(fragment_ids or []),
        required_fragment_ids=list(required_fragment_ids or []),
    )


def _step_result(item_id, segment_index, step_kind, source_kind, binding):
    return phase2.StepExecutionResult(
        item_id=item_id,
        step_id=f"{item_id}.S{segment_index}",
        segment_index=segment_index,
        step_kind=step_kind,
        source_kind=source_kind,
        status=STATIC_STATUS_OK,
        origin_layer_ids=[binding.origin_layer_id],
        metrics=MetricTotals(latency_s=0.0, energy_j=0.0, power_w=0.0, area_um2=0.0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[binding],
        warnings=[],
    )


def test_wzw_item_hands_off_same_round_outputs_locally_only():
    item = _wzw_item()
    local_state = phase2.RoundLocalState.seed(item, _boundary_view("R0", "model_input"))

    conv1 = _binding(
        "Conv1",
        "conv",
        CONTRIBUTION_COMPLETE_OUTPUT,
        source_state_ids=["model_input"],
    )
    local_state.commit_step(
        _step_result("R0", 0, "weighted_layer_segment", "static_eval", conv1)
    )

    relu_input = local_state.input_view_for(item.ordered_segments[1])
    assert relu_input.boundary_state_ids == []
    assert relu_input.local_state_ids == ["local:R0:Conv1"]
    assert relu_input.local_state_records[0].boundary_source_state_ids == ["model_input"]

    relu1 = _binding(
        "Relu1",
        "relu",
        CONTRIBUTION_HOOK_OUTPUT,
        source_local_state_ids=["local:R0:Conv1"],
    )
    local_state.commit_step(
        _step_result("R0", 1, "zero_xbar_segment", "hook", relu1)
    )

    conv2_input = local_state.input_view_for(item.ordered_segments[2])
    assert conv2_input.local_state_ids == ["local:R0:Relu1"]
    assert conv2_input.boundary_state_ids == []
    assert conv2_input.local_state_records[0].boundary_source_state_ids == ["model_input"]
    assert relu1.source_state_ids == ["model_input"]
    assert relu1.to_dict()["source_local_state_ids"] == ["local:R0:Conv1"]
    assert local_state.boundary_bindings_for_finalize() == []


def test_wzw_item_finalizes_only_declared_external_output():
    item = _wzw_item(external_output=True)
    local_state = phase2.RoundLocalState.seed(item, _boundary_view("R0", "model_input"))

    conv1 = _binding(
        "Conv1",
        "conv",
        CONTRIBUTION_COMPLETE_OUTPUT,
        source_state_ids=["model_input"],
    )
    relu1 = _binding(
        "Relu1",
        "relu",
        CONTRIBUTION_HOOK_OUTPUT,
        source_local_state_ids=["local:R0:Conv1"],
    )
    conv2 = _binding(
        "Conv2",
        "conv",
        CONTRIBUTION_COMPLETE_OUTPUT,
        source_local_state_ids=["local:R0:Relu1"],
    )

    local_state.commit_step(
        _step_result("R0", 0, "weighted_layer_segment", "static_eval", conv1)
    )
    local_state.commit_step(
        _step_result("R0", 1, "zero_xbar_segment", "hook", relu1)
    )
    local_state.commit_step(
        _step_result("R0", 2, "weighted_layer_segment", "static_eval", conv2)
    )

    assert conv2.source_state_ids == ["model_input"]
    assert local_state.boundary_bindings_for_finalize() == [conv2]


def test_partial_fragment_bindings_with_same_spatial_scope_all_finalize():
    item = _wzw_item()
    item.partial_output_contract = [
        BoundaryRef(
            state_id="state:Conv1",
            tensor_role="activation",
            producer_item_id="R0",
        )
    ]
    local_state = phase2.RoundLocalState.seed(item, _boundary_view("R0", "model_input"))
    required_fragment_ids = [
        "Conv1|in:0:128|out:0:128|plane:0",
        "Conv1|in:0:128|out:0:128|plane:1",
    ]
    first_plane = _binding(
        "Conv1",
        "conv",
        CONTRIBUTION_PARTIAL,
        source_state_ids=["model_input"],
        fragment_ids=[required_fragment_ids[0]],
        required_fragment_ids=required_fragment_ids,
    )
    second_plane = _binding(
        "Conv1",
        "conv",
        CONTRIBUTION_PARTIAL,
        source_state_ids=["model_input"],
        fragment_ids=[required_fragment_ids[1]],
        required_fragment_ids=required_fragment_ids,
    )

    local_state.commit_step(
        _step_result("R0", 0, "weighted_layer_segment", "static_eval", first_plane)
    )
    local_state.commit_step(
        _step_result("R0", 0, "weighted_layer_segment", "static_eval", second_plane)
    )

    assert local_state.boundary_bindings_for_finalize() == [first_plane, second_plane]


def test_same_round_consumer_rejects_nonready_local_source():
    item = _wzw_item()
    local_state = phase2.RoundLocalState.seed(item, _boundary_view("R0", "model_input"))

    with pytest.raises(Phase2Invalid) as exc_info:
        local_state.input_view_for(item.ordered_segments[1])

    assert exc_info.value.code == "ROUND_LOCAL_SOURCE_NOT_READY"
    assert exc_info.value.detail == "R0: Conv1 is not ready"


def test_commit_rejects_local_id_in_direct_boundary_sources():
    item = _wzw_item()
    local_state = phase2.RoundLocalState.seed(item, _boundary_view("R0", "model_input"))
    binding = _binding(
        "Conv1",
        "conv",
        CONTRIBUTION_COMPLETE_OUTPUT,
        source_state_ids=["local:R0:Conv1"],
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        local_state.commit_step(
            _step_result("R0", 0, "weighted_layer_segment", "static_eval", binding)
        )

    assert exc_info.value.code == "ROUND_LOCAL_INVALID_BOUNDARY_SOURCE"
    assert "local:R0:Conv1" in exc_info.value.detail


def test_input_view_returns_snapshot_of_local_state_records():
    item = _wzw_item()
    local_state = phase2.RoundLocalState.seed(item, _boundary_view("R0", "model_input"))
    local_state.commit_step(
        _step_result(
            "R0",
            0,
            "weighted_layer_segment",
            "static_eval",
            _binding(
                "Conv1",
                "conv",
                CONTRIBUTION_COMPLETE_OUTPUT,
                source_state_ids=["model_input"],
            ),
        )
    )

    view = local_state.input_view_for(item.ordered_segments[1])
    view.local_state_records[0].shape["C"] = 999
    view.local_state_records[0].boundary_source_state_ids.append("corrupt")

    later_view = local_state.input_view_for(item.ordered_segments[1])
    assert later_view.local_state_records[0].shape == {"C": 8}
    assert later_view.local_state_records[0].boundary_source_state_ids == ["model_input"]


def test_input_view_returns_snapshot_of_boundary_state_records():
    item = _wzw_item()
    local_state = phase2.RoundLocalState.seed(item, _boundary_view("R0", "model_input"))

    view = local_state.input_view_for(item.ordered_segments[0])
    view.boundary_state_records[0].shape["C"] = 999
    view.boundary_state_records[0].source_state_ids.append("corrupt")

    later_view = local_state.input_view_for(item.ordered_segments[0])
    assert later_view.boundary_state_records[0].shape == {"C": 3}
    assert later_view.boundary_state_records[0].source_state_ids == []
