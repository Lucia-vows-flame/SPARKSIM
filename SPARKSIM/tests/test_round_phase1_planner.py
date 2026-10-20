import MNSIM.Scheduling.phase1.round_planner as round_planner
from MNSIM.Scheduling.phase1.round_planner import build_round_plan
from MNSIM.Scheduling.round_types import LayerIR, ModelIR, SequenceSpec


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


def _segment_layer_ids(item):
    ids = []
    for segment in item.ordered_segments:
        if segment.segment_kind == "weighted_layer_segment":
            ids.append(segment.selected_entry.layer_id)
        elif segment.segment_kind == "zero_xbar_segment":
            ids.append([node.layer_id for node in segment.local_graph_nodes])
        else:
            raise AssertionError(f"unexpected segment kind {segment.segment_kind}")
    return ids


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
        _fc("FC1", 0, outfeature=64, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["FC1"], consumers=["FC2"]),
        _fc("FC2", 2, outfeature=64, inputs=["Relu1"]),
    ])
    plan = build_round_plan(model, _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    assert len(plan.execution_items) == 1
    item = plan.execution_items[0]
    assert item.item_kind == "weighted_physical_round"
    assert item.round_kind == "full_layer_pack"
    assert [entry.layer_id for entry in item.selected_entries] == ["FC1", "FC2"]
    assert all(entry.layer_complete_in_round for entry in item.selected_entries)
    assert _segment_layer_ids(item) == ["FC1", ["Relu1"], "FC2"]
    assert item.resource_proof.used_xbars == sum(entry.used_xbars for entry in item.selected_entries)
    assert [(edge.producer_node_id, edge.consumer_node_id) for edge in item.same_item_edges] == [
        ("FC1", "Relu1"),
        ("Relu1", "FC2"),
    ]
    assert item.external_outputs[0].state_id == "state:FC2"
    assert item.external_outputs[0].consumer_item_ids == []


def test_non_weighted_closure_stays_in_full_pack_when_next_weighted_layer_does_not_fit():
    model = _model([
        _fc("FC1", 0, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["FC1"], consumers=["FC2"]),
        _fc("FC2", 2, inputs=["Relu1"]),
    ])
    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1]),
    )

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0", "R1"]
    assert _segment_layer_ids(plan.execution_items[0]) == ["FC1", ["Relu1"]]
    assert [entry.layer_id for entry in plan.execution_items[1].selected_entries] == ["FC2"]
    assert all(item.item_kind == "weighted_physical_round" for item in plan.execution_items)


def test_terminal_weighted_layer_has_external_output_with_no_consumers():
    plan = build_round_plan(_model([_fc("FC1", 0)]), _config())

    assert plan.feasible is True
    output = plan.execution_items[0].external_outputs
    assert [(ref.state_id, ref.consumer_item_ids, ref.consumer_node_ids) for ref in output] == [
        ("state:FC1", [], []),
    ]


def test_terminal_split_and_continuation_keep_partial_output_contract():
    model = _model([_fc("Large", 0, infeature=128, outfeature=1024)])
    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    for item in plan.execution_items:
        assert [ref.state_id for ref in item.external_outputs] == ["state:Large"]
        assert [ref.state_id for ref in item.partial_output_contract] == ["state:Large"]
        assert item.external_outputs[0].consumer_item_ids == []
        assert item.partial_output_contract[0].consumer_item_ids == []


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


def test_split_uses_all_hardware_xbars_even_when_precision_groups_are_unaligned():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=128 * 83),
    ])
    plan = build_round_plan(
        model,
        _config(
            Device_Level=2,
            Weight_precision=8,
            Xbar_Polarity=2,
            Tile_Num=[6, 6],
            PE_Num=[4, 4],
            Group_Num=1,
        ),
    )

    assert plan.feasible is True
    assert [item.round_kind for item in plan.execution_items] == [
        "single_layer_split",
        "single_layer_continuation",
    ]
    ranges = [
        (entry.xbar_range_start, entry.xbar_range_end)
        for item in plan.execution_items
        for entry in item.selected_entries
    ]
    assert ranges == [(0, 1152), (1152, 1328)]


def test_split_uses_even_physical_ranges_without_full_precision_group_alignment():
    plan = build_round_plan(
        _model([_fc("Large", 0, infeature=128, outfeature=128)]),
        _config(
            Device_Level=2,
            Weight_precision=8,
            Xbar_Polarity=2,
            Tile_Num=[1, 1],
            PE_Num=[1, 1],
            Group_Num=1,
        ),
    )

    assert plan.feasible is True
    assert plan.hardware_budget["xbar_nums"] == 2
    [desc] = plan.layer_resource_descriptors
    assert desc.required_xbar_count == 16
    assert [item.round_kind for item in plan.execution_items] == [
        "single_layer_split",
        *["single_layer_continuation"] * 7,
    ]
    entries = [
        entry
        for item in plan.execution_items
        for entry in item.selected_entries
    ]
    assert [
        (entry.xbar_range_start, entry.xbar_range_end)
        for entry in entries
    ] == [(index, index + 2) for index in range(0, 16, 2)]
    assert all(entry.used_xbars == entry.xbar_range_end - entry.xbar_range_start for entry in entries)
    assert all(entry.xbar_range_start % 2 == 0 and entry.xbar_range_end % 2 == 0 for entry in entries)
    assert entries[0].used_xbars < desc.required_xbar_count


def test_final_continuation_hosts_fitting_downstream_weighted_layer_with_leading_closure():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=1024, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["Large"], consumers=["Next"]),
        _fc("Next", 2, infeature=128, outfeature=128, inputs=["Relu1"]),
    ])

    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0", "R1", "R2"]
    assert [item.round_kind for item in plan.execution_items] == [
        "single_layer_split",
        "single_layer_continuation",
        "full_layer_pack",
    ]
    assert _segment_layer_ids(plan.execution_items[2]) == [["Relu1"], "Next"]
    assert all(item.item_kind == "weighted_physical_round" for item in plan.execution_items)


def test_forced_host_keeps_preexisting_unrelated_zero_in_separate_exceptional_round():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=1024, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["Large"], consumers=["Next"]),
        _fc("Next", 2, infeature=128, outfeature=128, inputs=["Relu1"]),
        _non_weighted("Unrelated", "relu", 3),
    ])

    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0", "R1", "R2", "Z0"]
    assert _segment_layer_ids(plan.execution_items[2]) == [["Relu1"], "Next"]
    assert [node.layer_id for node in plan.execution_items[3].local_graph_nodes] == ["Unrelated"]


def test_final_continuation_emits_terminal_nonweighted_tail_as_exceptional_zero_round():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=1024, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["Large"]),
    ])

    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0", "R1", "Z0"]
    assert [node.layer_id for node in plan.execution_items[-1].local_graph_nodes] == ["Relu1"]


def test_terminal_continuation_tail_keeps_preexisting_unrelated_zero_separate():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=1024, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["Large"]),
        _non_weighted("Unrelated", "relu", 2),
    ])

    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0", "R1", "Z0", "Z1"]
    assert [node.layer_id for node in plan.execution_items[2].local_graph_nodes] == ["Relu1"]
    assert [node.layer_id for node in plan.execution_items[3].local_graph_nodes] == ["Unrelated"]


def test_existing_prefix_zero_round_stays_before_first_weighted_round():
    model = _model([
        _non_weighted("InputView", "view", 0, consumers=["InputRelu"]),
        _non_weighted("InputRelu", "relu", 1, inputs=["InputView"], consumers=["Large"]),
        _fc("Large", 2, infeature=128, outfeature=1024, inputs=["InputRelu"]),
    ])

    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["Z0", "R0", "R1"]
    assert [node.layer_id for node in plan.execution_items[0].local_graph_nodes] == [
        "InputView",
        "InputRelu",
    ]


def test_oversized_forced_downstream_layer_owns_leading_closure_only_in_first_split():
    model = _model([
        _fc("Large1", 0, infeature=128, outfeature=1024, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["Large1"], consumers=["Large2"]),
        _fc("Large2", 2, infeature=128, outfeature=1024, inputs=["Relu1"]),
    ])

    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0", "R1", "R2", "R3"]
    assert _segment_layer_ids(plan.execution_items[2]) == [["Relu1"], "Large2"]
    assert _segment_layer_ids(plan.execution_items[3]) == ["Large2"]
    assert plan.execution_items[2].selected_entries[0].layer_complete_in_round is False


def test_each_continuation_round_has_exactly_one_weighted_origin():
    model = _model([
        _fc("Large1", 0, infeature=128, outfeature=1024, consumers=["Large2"]),
        _fc("Large2", 1, infeature=128, outfeature=1024, inputs=["Large1"]),
    ])

    plan = build_round_plan(
        model,
        _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4),
    )

    assert plan.feasible is True
    for item in plan.execution_items:
        if item.round_kind != "single_layer_continuation":
            continue
        weighted_ids = {
            segment.selected_entry.layer_id
            for segment in item.ordered_segments
            if segment.segment_kind == "weighted_layer_segment"
        }
        assert len(weighted_ids) == 1


def test_split_and_continuation_rounds_all_read_the_original_activation():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=1024),
    ])
    plan = build_round_plan(model, _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4))

    assert plan.feasible is True
    assert [[ref.state_id for ref in item.external_inputs] for item in plan.execution_items] == [
        ["model_input"],
        ["model_input"],
    ]


def test_split_and_continuation_rounds_share_final_downstream_contract():
    model = _model([
        _fc("Large", 0, infeature=128, outfeature=1024, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["Large"]),
    ])
    plan = build_round_plan(model, _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[1, 1], PE_Num=[1, 1], Group_Num=4))
    weighted_items = [item for item in plan.execution_items if item.item_kind == "weighted_physical_round"]

    assert plan.feasible is True
    assert [ref.state_id for ref in weighted_items[0].partial_output_contract] == ["state:Large"]
    assert [ref.state_id for ref in weighted_items[1].partial_output_contract] == ["state:Large"]
    assert [ref.consumer_item_ids for ref in weighted_items[0].partial_output_contract] == [["Z0"]]
    assert [ref.consumer_item_ids for ref in weighted_items[1].partial_output_contract] == [["Z0"]]


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


def test_linear_cnn_keeps_non_weighted_closures_in_weighted_rounds():
    layers = [
        _layer("Conv1", "conv", True, 0, consumers=["Relu1"], shape={"Inputchannel": 3, "Outputchannel": 8, "Kernelsize": 3}, weight_ref_id="w:Conv1"),
        _non_weighted("Relu1", "relu", 1, inputs=["Conv1"], consumers=["Conv2"]),
        _layer("Conv2", "conv", True, 2, inputs=["Relu1"], consumers=["Relu2"], shape={"Inputchannel": 8, "Outputchannel": 8, "Kernelsize": 3}, weight_ref_id="w:Conv2"),
        _non_weighted("Relu2", "relu", 3, inputs=["Conv2"], consumers=["FC"]),
        _fc("FC", 4, infeature=128, outfeature=8, inputs=["Relu2"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2, Tile_Num=[4, 4], PE_Num=[2, 2]))

    assert plan.feasible is True
    assert [item.item_kind for item in plan.execution_items] == ["weighted_physical_round"]
    item = plan.execution_items[0]
    assert [entry.layer_id for entry in item.selected_entries] == ["Conv1", "Conv2", "FC"]
    assert _segment_layer_ids(item) == ["Conv1", ["Relu1"], "Conv2", ["Relu2"], "FC"]


def test_zero_xbar_closure_becomes_independent_round_after_layer_completion():
    layers = [
        _fc("FC1", 0, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["FC1"], consumers=["BN1"]),
        _non_weighted("BN1", "bn", 2, inputs=["Relu1"], consumers=["View1"]),
        _non_weighted("View1", "view", 3, inputs=["BN1"], consumers=["FC2"]),
        _fc("FC2", 4, inputs=["View1"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0"]
    assert _segment_layer_ids(plan.execution_items[0]) == [
        "FC1",
        ["Relu1"],
        ["BN1"],
        ["View1"],
        "FC2",
    ]


def test_branch_merge_zero_xbar_round_splits_before_downstream_weighted_layer():
    layers = [
        _fc("Conv0", 0, consumers=["ConvA", "ConvB"]),
        _fc("ConvA", 1, inputs=["Conv0"], consumers=["Sum"]),
        _fc("ConvB", 2, inputs=["Conv0"], consumers=["Sum"]),
        _non_weighted("Sum", "element_sum", 3, inputs=["ConvA", "ConvB"], consumers=["ConvC"]),
        _fc("ConvC", 4, inputs=["Sum"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0"]
    assert [entry.layer_id for entry in plan.execution_items[0].selected_entries] == ["Conv0", "ConvA", "ConvB", "ConvC"]
    assert _segment_layer_ids(plan.execution_items[0]) == ["Conv0", "ConvA", "ConvB", ["Sum"], "ConvC"]


def test_rnn_like_projection_state_update_is_independent_zero_xbar_round():
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
    assert [item.item_id for item in plan.execution_items] == ["R0"]
    assert _segment_layer_ids(plan.execution_items[0]) == ["RNN_projection", ["RNN_state_update"]]


def test_rnn_like_split_projection_delays_state_update_until_tail_complete():
    seq = _seq(length=3, ordered=True, cross_step=True, states=["h"])
    layers = [
        _layer(
            "RNN_projection",
            "fc_like_projection",
            True,
            0,
            consumers=["RNN_state_update"],
            shape={"Infeature": 128, "Outfeature": 1536},
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
        "weighted_physical_round",
        "zero_xbar_round",
    ]
    assert [item.round_kind for item in plan.execution_items[:3]] == [
        "single_layer_split",
        "single_layer_continuation",
        "single_layer_continuation",
    ]
    assert _segment_layer_ids(plan.execution_items[0]) == ["RNN_projection"]
    assert _segment_layer_ids(plan.execution_items[1]) == ["RNN_projection"]
    assert _segment_layer_ids(plan.execution_items[2]) == ["RNN_projection"]
    assert [node.layer_id for node in plan.execution_items[3].local_graph_nodes] == ["RNN_state_update"]
    assert plan.execution_items[0].selected_entries[0].layer_complete_in_round is False
    assert plan.execution_items[1].selected_entries[0].layer_complete_in_round is False
    assert plan.execution_items[2].selected_entries[0].layer_complete_in_round is True


def test_weighted_round_resource_proof_and_selected_entries_ignore_zero_segments():
    layers = [
        _fc("FC1", 0, consumers=["Relu1"]),
        _non_weighted("Relu1", "relu", 1, inputs=["FC1"], consumers=["FC2"]),
        _fc("FC2", 2, inputs=["Relu1"]),
    ]
    plan = build_round_plan(_model(layers), _config(Xbar_Polarity=1, Weight_precision=2))

    assert plan.feasible is True
    assert [item.item_id for item in plan.execution_items] == ["R0"]
    item = plan.execution_items[0]
    assert [entry.layer_id for entry in item.selected_entries] == ["FC1", "FC2"]
    assert _segment_layer_ids(item) == ["FC1", ["Relu1"], "FC2"]
    assert item.resource_proof.used_xbars == sum(entry.used_xbars for entry in item.selected_entries)
    assert all(node.node_role != "hook" for node in item.local_graph_nodes if node.layer_id != "Relu1")
    assert all(entry.layer_type in {"fc", "conv", "fc_like_projection"} for entry in item.selected_entries)


def test_standalone_zero_xbar_round_is_only_used_when_no_weighted_work_exists():
    layers = [
        _non_weighted("InputView", "view", 0, consumers=["InputRelu"]),
        _non_weighted("InputRelu", "relu", 1, inputs=["InputView"]),
    ]
    plan = build_round_plan(_model(layers), _config())

    assert plan.feasible is True
    assert [item.item_kind for item in plan.execution_items] == ["zero_xbar_round"]
    assert plan.execution_items[0].item_id == "Z0"
    assert [node.layer_id for node in plan.execution_items[0].local_graph_nodes] == ["InputView", "InputRelu"]
    assert plan.execution_items[0].resource_proof.used_xbars == 0


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


def test_build_round_plan_converts_post_boundary_validation_failure_to_invalid_plan(monkeypatch):
    model = _model([_fc("FC1", 0)])
    original_populate = round_planner._populate_boundary_refs

    def populate_and_corrupt(model_ir, items):
        original_populate(model_ir, items)
        items[0].selected_entries[0].layer_complete_in_round = False

    monkeypatch.setattr(round_planner, "_populate_boundary_refs", populate_and_corrupt)

    plan = build_round_plan(model, _config())

    assert plan.feasible is False
    assert plan.invalid_reason.startswith("INTERNAL_PLANNER_INVARIANT:")
