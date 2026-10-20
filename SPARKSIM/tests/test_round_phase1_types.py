import pytest

from MNSIM.Scheduling.round_types import (
    CompletionContract,
    LayerIR,
    LocalGraphNode,
    ModelIR,
    Phase1Invalid,
    PrepassRef,
    ResourceProof,
    RoundPlan,
    SelectedEntry,
    SequenceSpec,
    WeightedLayerSegment,
    WeightedPhysicalRound,
    ZeroXbarSegment,
    ZeroXbarRound,
    execution_item_from_dict,
    round_plan_from_dict,
    validate_round_plan,
)


def _plan_with_item(item):
    return RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 16},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )


def _old_weighted_round_dict_without_ordered_segments():
    return {
        "item_kind": "weighted_physical_round",
        "item_id": "W0",
        "item_index": 0,
        "round_kind": "full_layer_pack",
        "selected_entries": [
            {
                "layer_id": "Conv1",
                "layer_type": "conv",
                "xbar_range_start": 0,
                "xbar_range_end": 2,
                "used_xbars": 2,
                "layer_complete_in_round": True,
                "prepass_ref": {
                    "ref_id": "prepass:Conv1",
                    "model_id": "m0",
                    "layer_id": "Conv1",
                    "layer_type": "conv",
                    "resource_version": "mnsim20_formula_v1",
                    "sim_config_fingerprint": "sim:abc",
                    "layer_config_fingerprint": "layer:abc",
                    "weight_ref_id": "w_conv1",
                    "shape_signature": {
                        "Inputchannel": 3,
                        "Outputchannel": 8,
                        "Kernelsize": 3,
                    },
                },
            }
        ],
        "local_graph_nodes": [
            {"layer_id": "Conv1", "layer_type": "conv", "node_role": "weighted"}
        ],
        "external_inputs": [],
        "same_item_edges": [],
        "external_outputs": [],
        "partial_output_contract": [],
        "resource_proof": {"xbar_nums": 16, "used_xbars": 2},
        "backend_runtime": {},
    }


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


def test_local_graph_digital_metadata_survives_round_trip():
    node = LocalGraphNode(
        layer_id="L001_lif",
        layer_type="lif",
        node_role="hook",
        digital_metadata={"signed_bit_width": 9, "domain_id": "acc:test"},
    )
    item = ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind="hook_only",
        selected_entries=[],
        local_graph_nodes=[node],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=16, used_xbars=0),
        backend_runtime={},
    )

    restored = execution_item_from_dict(item.to_dict())

    assert restored.local_graph_nodes[0].digital_metadata == node.digital_metadata
    assert restored.to_dict() == item.to_dict()


def test_zero_xbar_round_serializes_and_deserializes_canonical_per_node_segments():
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

    data = item.to_dict()

    assert data["item_kind"] == "zero_xbar_round"
    assert data["selected_entries"] == []
    assert data["resource_proof"]["used_xbars"] == 0
    assert data["ordered_segments"] == [
        {
            "segment_index": 0,
            "zero_xbar_kind": "hook_only",
            "local_graph_nodes": [
                {
                    "layer_id": "Relu1",
                    "layer_type": "relu",
                    "node_role": "hook",
                    "node_shape": {},
                }
            ],
            "segment_kind": "zero_xbar_segment",
        }
    ]
    restored = execution_item_from_dict(data)
    assert [segment.segment_index for segment in restored.ordered_segments] == [0]
    assert [
        node.layer_id
        for segment in restored.ordered_segments
        for node in segment.local_graph_nodes
    ] == ["Relu1"]
    assert restored.to_dict() == data


def test_zero_xbar_round_deserialization_preserves_serialized_ordered_segments():
    item = ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind="mixed_non_weighted",
        selected_entries=[],
        local_graph_nodes=[
            LocalGraphNode(layer_id="Relu1", layer_type="relu", node_role="hook"),
            LocalGraphNode(
                layer_id="Pool1",
                layer_type="pooling",
                node_role="structural_non_weighted",
            ),
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=16, used_xbars=0),
        backend_runtime={},
        ordered_segments=[
            ZeroXbarSegment(
                segment_index=0,
                zero_xbar_kind="hook_only",
                local_graph_nodes=[
                    LocalGraphNode(layer_id="Relu1", layer_type="relu", node_role="hook")
                ],
            ),
            ZeroXbarSegment(
                segment_index=1,
                zero_xbar_kind="structural_mnsim",
                local_graph_nodes=[
                    LocalGraphNode(
                        layer_id="Pool1",
                        layer_type="pooling",
                        node_role="structural_non_weighted",
                    )
                ],
            ),
        ],
    )
    data = item.to_dict()

    restored = execution_item_from_dict(data)

    assert [segment.zero_xbar_kind for segment in restored.ordered_segments] == [
        "hook_only",
        "structural_mnsim",
    ]
    assert restored.to_dict() == data


def test_full_layer_pack_rejects_incomplete_selected_entry():
    data = _old_weighted_round_dict_without_ordered_segments()
    data["selected_entries"][0]["layer_complete_in_round"] = False

    try:
        validate_round_plan(_plan_with_item(execution_item_from_dict(data)))
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "complete" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


@pytest.mark.parametrize("round_kind", ["single_layer_split", "single_layer_continuation"])
def test_single_layer_round_rejects_multiple_selected_entries(round_kind):
    data = _old_weighted_round_dict_without_ordered_segments()
    data["round_kind"] = round_kind
    data["selected_entries"].append(dict(data["selected_entries"][0], layer_id="Conv2"))
    data["local_graph_nodes"].append(
        {"layer_id": "Conv2", "layer_type": "conv", "node_role": "weighted"}
    )
    data["resource_proof"]["used_xbars"] = 4

    try:
        validate_round_plan(_plan_with_item(execution_item_from_dict(data)))
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "exactly one" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_weighted_round_rejects_unrecognized_segment_kind_and_node_role():
    item = execution_item_from_dict(_old_weighted_round_dict_without_ordered_segments())
    item.ordered_segments[0].segment_kind = "unexpected"
    item.ordered_segments[0].local_graph_node.node_role = "unexpected"

    try:
        validate_round_plan(_plan_with_item(item))
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "segment_kind" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_weighted_round_rejects_unrecognized_node_role():
    item = execution_item_from_dict(_old_weighted_round_dict_without_ordered_segments())
    item.ordered_segments[0].local_graph_node.node_role = "unexpected"

    try:
        validate_round_plan(_plan_with_item(item))
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "node role" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_zero_xbar_round_rejects_invalid_segment_role_and_used_xbar_proof():
    item = ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind="hook_only",
        selected_entries=[],
        local_graph_nodes=[
            LocalGraphNode(layer_id="Relu1", layer_type="relu", node_role="weighted")
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=16, used_xbars=1),
        backend_runtime={},
    )

    try:
        validate_round_plan(_plan_with_item(item))
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "used_xbars" in exc.detail or "role" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_zero_xbar_round_rejects_non_zero_xbar_segment_kind():
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
    item.ordered_segments[0].segment_kind = "weighted_layer_segment"

    try:
        validate_round_plan(_plan_with_item(item))
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "zero_xbar_segment" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_weighted_round_ordered_segments_round_trip_weighted_and_zero_xbar_work():
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
    selected = SelectedEntry(
        layer_id="Conv1",
        layer_type="conv",
        xbar_range_start=0,
        xbar_range_end=2,
        used_xbars=2,
        layer_complete_in_round=True,
        prepass_ref=prepass_ref,
    )
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[selected],
        local_graph_nodes=[
            LocalGraphNode(layer_id="Conv1", layer_type="conv", node_role="weighted"),
            LocalGraphNode(layer_id="BN1", layer_type="bn", node_role="hook"),
            LocalGraphNode(layer_id="Relu1", layer_type="relu", node_role="hook"),
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=16, used_xbars=2),
        backend_runtime={},
        ordered_segments=[
            WeightedLayerSegment(
                segment_index=0,
                selected_entry=selected,
                local_graph_node=LocalGraphNode(
                    layer_id="Conv1", layer_type="conv", node_role="weighted"
                ),
            ),
            ZeroXbarSegment(
                segment_index=1,
                zero_xbar_kind="hook_only",
                local_graph_nodes=[
                    LocalGraphNode(layer_id="BN1", layer_type="bn", node_role="hook"),
                    LocalGraphNode(layer_id="Relu1", layer_type="relu", node_role="hook"),
                ],
            ),
        ],
    )

    data = item.to_dict()

    assert data["item_id"] == "R0"
    assert data["ordered_segments"][0]["segment_kind"] == "weighted_layer_segment"
    assert data["ordered_segments"][0]["selected_entry"]["layer_id"] == "Conv1"
    assert data["ordered_segments"][1]["segment_kind"] == "zero_xbar_segment"
    assert data["ordered_segments"][1]["zero_xbar_kind"] == "hook_only"
    assert [
        node["layer_id"] for node in data["ordered_segments"][1]["local_graph_nodes"]
    ] == ["BN1", "Relu1"]
    assert execution_item_from_dict(data).to_dict() == data


def test_legacy_weighted_round_deserialization_canonicalizes_each_selected_entry():
    old_data = _old_weighted_round_dict_without_ordered_segments()
    old_data["selected_entries"].append(
        {
            "layer_id": "FC1",
            "layer_type": "fc",
            "xbar_range_start": 0,
            "xbar_range_end": 1,
            "used_xbars": 1,
            "layer_complete_in_round": True,
            "prepass_ref": {
                "ref_id": "prepass:FC1",
                "model_id": "m0",
                "layer_id": "FC1",
                "layer_type": "fc",
                "resource_version": "mnsim20_formula_v1",
                "sim_config_fingerprint": "sim:abc",
                "layer_config_fingerprint": "layer:def",
                "weight_ref_id": "w_fc1",
                "shape_signature": {"Infeature": 8, "Outfeature": 4},
            },
        }
    )
    old_data["local_graph_nodes"].append(
        {"layer_id": "FC1", "layer_type": "fc", "node_role": "weighted"}
    )
    old_data["resource_proof"]["used_xbars"] = 3

    data = execution_item_from_dict(old_data).to_dict()

    assert [segment["segment_index"] for segment in data["ordered_segments"]] == [0, 1]
    assert [
        segment["selected_entry"]["layer_id"]
        for segment in data["ordered_segments"]
    ] == ["Conv1", "FC1"]
    assert data["item_kind"] == "weighted_physical_round"
    assert data["round_kind"] == "full_layer_pack"
    assert [entry["layer_id"] for entry in data["selected_entries"]] == ["Conv1", "FC1"]


def test_legacy_weighted_round_deserialization_preserves_weighted_and_nonweighted_node_order():
    old_data = _old_weighted_round_dict_without_ordered_segments()
    old_data["selected_entries"].append(
        {
            "layer_id": "FC1",
            "layer_type": "fc",
            "xbar_range_start": 0,
            "xbar_range_end": 1,
            "used_xbars": 1,
            "layer_complete_in_round": True,
            "prepass_ref": {
                "ref_id": "prepass:FC1",
                "model_id": "m0",
                "layer_id": "FC1",
                "layer_type": "fc",
                "resource_version": "mnsim20_formula_v1",
                "sim_config_fingerprint": "sim:abc",
                "layer_config_fingerprint": "layer:def",
                "weight_ref_id": "w_fc1",
                "shape_signature": {"Infeature": 8, "Outfeature": 4},
            },
        }
    )
    old_data["local_graph_nodes"] = [
        {"layer_id": "Conv1", "layer_type": "conv", "node_role": "weighted"},
        {"layer_id": "Relu1", "layer_type": "relu", "node_role": "hook"},
        {"layer_id": "FC1", "layer_type": "fc", "node_role": "weighted"},
    ]
    old_data["resource_proof"]["used_xbars"] = 3

    restored = execution_item_from_dict(old_data)

    assert [segment.segment_kind for segment in restored.ordered_segments] == [
        "weighted_layer_segment",
        "zero_xbar_segment",
        "weighted_layer_segment",
    ]
    assert [
        node.layer_id
        for segment in restored.ordered_segments
        for node in (
            [segment.local_graph_node]
            if isinstance(segment, WeightedLayerSegment)
            else segment.local_graph_nodes
        )
    ] == ["Conv1", "Relu1", "FC1"]
    assert restored.ordered_segments[1].zero_xbar_kind == "hook_only"


def test_serialized_noninteger_segment_index_raises_phase1_invalid():
    data = execution_item_from_dict(
        _old_weighted_round_dict_without_ordered_segments()
    ).to_dict()
    data["ordered_segments"][0]["segment_index"] = "not-an-integer"

    try:
        execution_item_from_dict(data)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "segment_index" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


@pytest.mark.parametrize("segment_index", [0.5, False])
def test_serialized_non_int_segment_index_raises_phase1_invalid(segment_index):
    data = execution_item_from_dict(
        _old_weighted_round_dict_without_ordered_segments()
    ).to_dict()
    data["ordered_segments"][0]["segment_index"] = segment_index

    try:
        execution_item_from_dict(data)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "segment_index" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_unknown_ordered_segment_kind_is_rejected():
    data = {
        **_old_weighted_round_dict_without_ordered_segments(),
        "ordered_segments": [{"segment_kind": "unexpected", "segment_index": 0}],
    }

    try:
        execution_item_from_dict(data)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "unknown segment_kind unexpected" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_round_plan_validation_rejects_weighted_segment_without_selected_entry():
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
    selected_entry = SelectedEntry(
        layer_id="Conv1",
        layer_type="conv",
        xbar_range_start=0,
        xbar_range_end=2,
        used_xbars=2,
        layer_complete_in_round=True,
        prepass_ref=prepass_ref,
    )
    missing_entry = SelectedEntry(
        layer_id="FC1",
        layer_type="fc",
        xbar_range_start=0,
        xbar_range_end=1,
        used_xbars=1,
        layer_complete_in_round=True,
        prepass_ref=PrepassRef(
            ref_id="prepass:FC1",
            model_id="m0",
            layer_id="FC1",
            layer_type="fc",
            resource_version="mnsim20_formula_v1",
            sim_config_fingerprint="sim:abc",
            layer_config_fingerprint="layer:def",
            weight_ref_id="w_fc1",
            shape_signature={"Infeature": 8, "Outfeature": 4},
        ),
    )
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[selected_entry],
        local_graph_nodes=[
            LocalGraphNode(layer_id="FC1", layer_type="fc", node_role="weighted")
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=16, used_xbars=2),
        backend_runtime={},
        ordered_segments=[
            WeightedLayerSegment(
                segment_index=0,
                selected_entry=missing_entry,
                local_graph_node=LocalGraphNode(
                    layer_id="FC1",
                    layer_type="fc",
                    node_role="weighted",
                ),
            )
        ],
    )
    plan = RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 16},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )

    try:
        validate_round_plan(plan)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "selected entry" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_round_plan_deserialization_rejects_noncontiguous_item_indexes():
    item = _old_weighted_round_dict_without_ordered_segments()
    item["item_index"] = 1
    data = {
        "plan_id": "plan:m0",
        "model_id": "m0",
        "backend_kind": "ann",
        "hardware_budget": {"xbar_size": [128, 128], "xbar_nums": 16},
        "execution_items": [item],
        "layer_resource_descriptors": [],
        "model_graph_ref": {"model_id": "m0"},
        "feasible": True,
        "invalid_reason": None,
    }

    try:
        round_plan_from_dict(data)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "item_index" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_round_plan_validation_rejects_noncontiguous_segment_indexes():
    item = execution_item_from_dict(_old_weighted_round_dict_without_ordered_segments())
    item.ordered_segments[0].segment_index = 1
    plan = RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 16},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )

    try:
        validate_round_plan(plan)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "segment_index" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_round_plan_validation_rejects_local_graph_nodes_outside_segment_order():
    item = execution_item_from_dict(_old_weighted_round_dict_without_ordered_segments())
    item.local_graph_nodes = []
    plan = RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 16},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )

    try:
        validate_round_plan(plan)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "local_graph_nodes" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_round_plan_validation_rejects_selected_entries_in_zero_xbar_round():
    selected_entry = execution_item_from_dict(
        _old_weighted_round_dict_without_ordered_segments()
    ).selected_entries[0]
    item = ZeroXbarRound(
        item_id="Z0",
        item_index=0,
        zero_xbar_kind="hook_only",
        selected_entries=[selected_entry],
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
    plan = RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 16},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )

    try:
        validate_round_plan(plan)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "zero-xbar" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_round_plan_validation_rejects_incorrect_used_xbar_proof():
    item = execution_item_from_dict(_old_weighted_round_dict_without_ordered_segments())
    item.resource_proof.used_xbars = 1
    plan = RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 16},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )

    try:
        validate_round_plan(plan)
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "used_xbars" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")


def test_completion_contract_defaults_are_phase1_defaults():
    contract = CompletionContract()

    assert contract.partial_visibility == "internal_only"
    assert contract.downstream_visibility == "complete_only"
    assert contract.same_item_handoff_allowed is True
    assert contract.completion_hook_trigger == "on_complete"


def test_unknown_execution_item_kind_is_rejected():
    try:
        execution_item_from_dict(
            {
                "item_kind": "unexpected",
                "item_id": "X0",
                "item_index": 0,
                "local_graph_nodes": [],
                "external_inputs": [],
                "same_item_edges": [],
                "external_outputs": [],
                "partial_output_contract": [],
                "resource_proof": {"xbar_nums": 16, "used_xbars": 0},
                "backend_runtime": {},
            }
        )
    except Phase1Invalid as exc:
        assert exc.code == "INTERNAL_PLANNER_INVARIANT"
        assert "unknown item_kind unexpected" in exc.detail
    else:
        raise AssertionError("Phase1Invalid was not raised")
