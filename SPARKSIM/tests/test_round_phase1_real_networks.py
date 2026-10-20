from pathlib import Path

import pytest
import torch

from MNSIM.Interface.network import get_net
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_network
from MNSIM.Scheduling.phase1.round_planner import build_round_plan


def _g8_sim_config():
    return {
        "Xbar_Size": [128, 128],
        "Tile_Num": [6, 6],
        "PE_Num": [4, 4],
        "Group_Num": 2,
        "Device_Level": 256,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }


def _mnsim_network(category):
    return get_net(
        hardware_config={
            "xbar_size": 128,
            "input_bit": 9,
            "weight_bit": 8,
            "ADC_quantize_bit": 10,
            "DAC_num": 64,
            "xbar_polarity": 2,
        },
        cate=category,
        num_classes=10,
    )


def _standard_vgg8_snn_network():
    return get_net(
        hardware_config={
            "xbar_size": 128,
            "input_bit": 8,
            "weight_bit": 8,
            "ADC_quantize_bit": 10,
            "DAC_num": 128,
            "xbar_polarity": 2,
        },
        cate="standard_vgg8_snn",
        num_classes=8,
    )


def _weighted_descriptors(plan):
    return {
        descriptor.layer_id: descriptor
        for descriptor in plan.layer_resource_descriptors
    }


def _weighted_items(plan):
    return [
        item
        for item in plan.execution_items
        if item.item_kind == "weighted_physical_round"
    ]


def _zero_xbar_items(plan):
    return [
        item
        for item in plan.execution_items
        if item.item_kind == "zero_xbar_round"
    ]


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


def _assert_final_continuation_closure_is_hosted_by_next_weighted_round(weighted_items):
    for previous, current in zip(weighted_items, weighted_items[1:]):
        if previous.round_kind != "single_layer_continuation":
            continue
        if not previous.selected_entries[0].layer_complete_in_round:
            continue
        assert all(segment.segment_kind == "weighted_layer_segment" for segment in previous.ordered_segments)
        first_weighted_index = next(
            index
            for index, segment in enumerate(current.ordered_segments)
            if segment.segment_kind == "weighted_layer_segment"
        )
        assert all(
            segment.segment_kind == "zero_xbar_segment"
            for segment in current.ordered_segments[:first_weighted_index]
        )


def _small_sim_config():
    return {
        "Xbar_Size": [128, 128],
        "Tile_Num": [1, 1],
        "PE_Num": [1, 1],
        "Group_Num": 1,
        "Device_Level": 256,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }


def test_vgg8_real_mnsim_network_builds_feasible_g8_round_plan():
    model_ir = build_model_ir_from_mnsim_network(_mnsim_network("vgg8"), model_id="real_vgg8")

    assert len(model_ir.layers) == 27
    assert sum(1 for layer in model_ir.layers if layer.is_weighted) == 8
    assert model_ir.layers[0].layer_id == "L000_conv"
    assert model_ir.layers[0].shape["Inputchannel"] == 3
    assert model_ir.layers[-1].layer_id == "L026_fc"
    assert model_ir.layers[-1].shape["Infeature"] == 1024

    plan = build_round_plan(model_ir, _g8_sim_config())

    assert plan.feasible is True
    assert plan.hardware_budget == {"xbar_size": [128, 128], "xbar_nums": 2304}
    assert len(plan.layer_resource_descriptors) == 8
    assert sum(desc.required_xbar_count for desc in plan.layer_resource_descriptors) == 1194
    assert _weighted_descriptors(plan)["L021_conv"].required_xbar_count == 592
    weighted_items = _weighted_items(plan)
    zero_items = _zero_xbar_items(plan)
    assert len(weighted_items) == 1
    assert zero_items == []
    assert [item.item_id for item in weighted_items] == ["R0"]
    assert [item.round_kind for item in weighted_items] == ["full_layer_pack"]
    assert [item.resource_proof.used_xbars for item in weighted_items] == [1194]
    assert [[entry.layer_id for entry in item.selected_entries] for item in weighted_items] == [
        ["L000_conv", "L003_conv", "L007_conv", "L010_conv", "L014_conv", "L017_conv", "L021_conv", "L026_fc"],
    ]
    assert _segment_layer_ids(weighted_items[0])[:21] == [
        "L000_conv", ["L001_bn"], ["L002_relu"],
        "L003_conv", ["L004_bn"], ["L005_relu"], ["L006_pooling"],
        "L007_conv", ["L008_bn"], ["L009_relu"],
        "L010_conv", ["L011_bn"], ["L012_relu"], ["L013_pooling"],
        "L014_conv", ["L015_bn"], ["L016_relu"],
        "L017_conv", ["L018_bn"], ["L019_relu"], ["L020_pooling"],
    ]
    assert _segment_layer_ids(weighted_items[0])[-6:] == [
        "L021_conv", ["L022_bn"], ["L023_relu"], ["L024_pooling"], ["L025_view"], "L026_fc",
    ]
    assert [ref.state_id for ref in weighted_items[0].external_outputs] == ["state:L026_fc"]
    assert [ref.state_id for ref in weighted_items[0].external_inputs] == ["model_input"]


def test_project_vgg8_snn_without_compiled_domains_is_rejected():
    network = _mnsim_network("vgg8_snn")
    layer_types = [layer.layer_config["type"] for layer in network.layer_list]

    assert network.input_params["input_shape"] == (1, 2, 32, 32)
    assert layer_types.count("conv") == 11
    assert layer_types.count("fc") == 1
    assert layer_types.count("lif") == 12
    assert "relu" not in layer_types
    assert "bn" not in layer_types
    assert "pooling" not in layer_types
    for index, layer_type in enumerate(layer_types):
        if layer_type == "conv":
            assert layer_types[index + 1] == "lif"
    assert layer_types[-3:] == ["view", "fc", "lif"]

    model_ir = build_model_ir_from_mnsim_network(network, model_id="project_vgg8_snn", backend_kind="snn")
    plan = build_round_plan(model_ir, _g8_sim_config())

    assert len(model_ir.layers) == 25
    assert sum(1 for layer in model_ir.layers if layer.is_weighted) == 12
    assert plan.feasible is False
    assert plan.invalid_reason == (
        "INVALID_LIF_DOMAIN: L000_conv has no accumulator domain metadata"
    )


def test_standard_vgg8_snn_checkpoint_exposes_t30_and_adc_domains():
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "archive/legacy_master_20260925/wsl_vgg8_std_mod_seq_hw_balanced16_20260814/"
        / "inputs/weights/standard_vgg8_snn_v7_software_pretrain_best.pth"
    )
    if not checkpoint.is_file():
        pytest.skip(f"standard VGG8 SNN pretrain checkpoint is unavailable: {checkpoint}")
    network = _standard_vgg8_snn_network()
    try:
        network.load_change_weights(torch.load(checkpoint, map_location="cpu"))
    except RuntimeError as error:
        pytest.skip(f"checkpoint is incompatible with the local network input contract: {error}")

    layer_types = [layer.layer_config["type"] for layer in network.layer_list]
    assert layer_types.count("conv") == 7
    assert layer_types.count("fc") == 1
    assert layer_types.count("lif") == 8
    assert layer_types[-3:] == ["view", "fc", "lif"]
    assert network.sequence_length == 30

    model_ir = build_model_ir_from_mnsim_network(
        network,
        model_id="standard_vgg8_snn_t30",
        backend_kind="snn",
        sim_config=Path(__file__).resolve().parents[1] / "SimConfig.ini",
    )

    assert model_ir.sequence_spec.sequence_length == 30
    assert len(model_ir.sequence_spec.recurrent_state_ids) == 8
    by_id = {layer.layer_id: layer for layer in model_ir.layers}
    for lif_id in model_ir.sequence_spec.recurrent_state_ids:
        lif = by_id[lif_id]
        producer = by_id[lif.input_layer_ids[0]]
        producer_domain = producer.backend_attrs["accumulator_domain"]
        assert lif.backend_attrs["accumulator_domain"] == producer_domain
        constants = lif.backend_attrs["compiled_lif_constants"]
        assert constants["domain"] == producer_domain
        assert constants["source_values"]["threshold"] == "1/100"
        assert producer_domain["source"] == "fixed_adc_completed_accumulator"

    plan = build_round_plan(model_ir, Path(__file__).resolve().parents[1] / "SimConfig.ini")
    assert plan.feasible is True


def test_resnet18_real_mnsim_network_builds_feasible_g8_round_plan_with_residual_dependencies():
    model_ir = build_model_ir_from_mnsim_network(_mnsim_network("resnet18"), model_id="real_resnet18")

    assert len(model_ir.layers) == 71
    assert sum(1 for layer in model_ir.layers if layer.is_weighted) == 22
    shortcut = next(layer for layer in model_ir.layers if layer.layer_id == "L023_conv")
    merge = next(layer for layer in model_ir.layers if layer.layer_id == "L025_element_sum")
    assert shortcut.input_layer_ids == ["L017_relu"]
    assert set(merge.input_layer_ids) == {"L022_bn", "L024_bn"}

    plan = build_round_plan(model_ir, _g8_sim_config())

    assert plan.feasible is True
    assert plan.hardware_budget == {"xbar_size": [128, 128], "xbar_nums": 2304}
    assert len(plan.layer_resource_descriptors) == 22
    assert sum(desc.required_xbar_count for desc in plan.layer_resource_descriptors) == 1758
    assert max(desc.required_xbar_count for desc in plan.layer_resource_descriptors) == 296
    weighted_items = _weighted_items(plan)
    zero_items = _zero_xbar_items(plan)
    assert len(weighted_items) == 1
    assert zero_items == []
    assert [item.item_id for item in weighted_items] == ["R0"]
    assert [item.resource_proof.used_xbars for item in weighted_items] == [1758]
    assert [[entry.layer_id for entry in item.selected_entries] for item in weighted_items] == [
        ["L000_conv", "L004_conv", "L007_conv", "L011_conv", "L014_conv", "L018_conv", "L021_conv", "L023_conv", "L027_conv", "L030_conv", "L034_conv", "L037_conv", "L039_conv", "L043_conv", "L046_conv", "L050_conv", "L053_conv", "L055_conv", "L059_conv", "L062_conv", "L067_fc", "L070_fc"],
    ]
    assert all(
        layer_id in [node.layer_id for node in item.local_graph_nodes]
        for item in weighted_items
        for layer_id in [entry.layer_id for entry in item.selected_entries]
    )
    assert any(["L064_element_sum"] == segment for segment in _segment_layer_ids(weighted_items[0]))


def test_vgg8_real_mnsim_network_two_xbar_splits_large_layers_without_ordinary_zero_rounds():
    model_ir = build_model_ir_from_mnsim_network(_mnsim_network("vgg8"), model_id="real_vgg8_small")
    plan = build_round_plan(model_ir, _small_sim_config())

    assert plan.feasible is True
    weighted_items = _weighted_items(plan)
    zero_items = _zero_xbar_items(plan)
    assert len(weighted_items) == 597
    assert zero_items == []
    assert [item.item_id for item in weighted_items] == [f"R{index}" for index in range(597)]
    assert all(item.resource_proof.used_xbars == 2 for item in weighted_items)
    assert sum(desc.required_xbar_count for desc in plan.layer_resource_descriptors) == 1194
    assert _weighted_descriptors(plan)["L003_conv"].required_xbar_count == 20
    assert _weighted_descriptors(plan)["L021_conv"].required_xbar_count == 592
    assert _weighted_descriptors(plan)["L026_fc"].required_xbar_count == 16
    _assert_final_continuation_closure_is_hosted_by_next_weighted_round(weighted_items)


def test_resnet18_real_mnsim_network_two_xbar_has_879_resource_rounds_without_ordinary_zero_rounds():
    model_ir = build_model_ir_from_mnsim_network(_mnsim_network("resnet18"), model_id="real_resnet18_small")
    plan = build_round_plan(model_ir, _small_sim_config())

    assert plan.feasible is True
    weighted_items = _weighted_items(plan)
    zero_items = _zero_xbar_items(plan)
    assert len(weighted_items) == 879
    assert zero_items == []
    assert [item.item_id for item in weighted_items] == [f"R{index}" for index in range(879)]
    assert all(item.resource_proof.used_xbars == 2 for item in weighted_items)
    assert sum(desc.required_xbar_count for desc in plan.layer_resource_descriptors) == 1758
    assert _weighted_descriptors(plan)["L004_conv"].required_xbar_count == 10
    assert _weighted_descriptors(plan)["L067_fc"].required_xbar_count == 128
    assert _weighted_descriptors(plan)["L070_fc"].required_xbar_count == 8
    _assert_final_continuation_closure_is_hosted_by_next_weighted_round(weighted_items)
