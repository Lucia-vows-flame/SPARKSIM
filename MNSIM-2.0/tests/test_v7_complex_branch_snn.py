from pathlib import Path
from types import SimpleNamespace

import torch

from MNSIM.Interface.network import NetworkGraph, get_net
from MNSIM.Interface.quantize import ConcatLayer, EleMulLayer
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_network
from MNSIM.Scheduling.phase1.round_planner import build_round_plan
from MNSIM.Scheduling.phase2.executor import execute_round_plan


HW = {
    "xbar_size": 128,
    "input_shape": (1, 6, 32, 32),
    "weight_quantization_bit": 9,
    "DAC_num": 128,
    "input_bit": 1,
    "weight_bit": 9,
    "ADC_quantize_bit": 5,
    "xbar_polarity": 2,
}


def _network():
    return get_net(HW, "inception_snn_v7_phase6", num_classes=8)


class _EchoStaticBackend:
    def run(self, request):
        layers = []
        for local_layer_id, layer_entry in enumerate(request.netstruct):
            layer_type = layer_entry[0][0]["type"]
            begin_time_s = local_layer_id * 1e-9
            layers.append(
                {
                    "local_layer_id": local_layer_id,
                    "layer_type": layer_type,
                    "begin_time_s": begin_time_s,
                    "finish_time_s": begin_time_s + 1e-9,
                    "area_um2": 1.0,
                    "power_w": 1.0,
                    "energy_j": 1e-9,
                    "tcg_mapping": {},
                }
            )
        count = len(layers)
        return {
            "layers": layers,
            "raw_totals": {
                "raw_latency_total_s": count * 1e-9,
                "raw_area_total_um2": float(count),
                "raw_power_total_w": float(count),
                "raw_energy_total_j": count * 1e-9,
            },
            "mapping_summary": {"layer_count": count},
        }


def test_inception_snn_v7_has_aligned_three_branch_concat_nodes():
    net = _network()
    types = [layer.layer_config["type"] for layer in net.layer_list]

    assert types.count("concat") == 3
    assert "pooling" not in types
    assert all(len(inputs) > 1 for kind, inputs in zip(types, net.input_index_list) if kind == "concat")
    for index, kind in enumerate(types):
        if kind != "concat":
            continue
        assert net.layer_list[index].layer_config["time_policy"] == "same_timestep"
        assert types[index + 1] == "conv"
        assert net.layer_list[index].layer_config["merge_mode"] == "channel_concat"


def test_merge_layers_preserve_channel_gate_broadcast_and_sequence_channel_axis():
    gate = torch.rand(2, 3)
    feature = torch.rand(2, 3, 4, 4)
    multiplied = EleMulLayer()([gate, feature, feature])
    assert multiplied.shape == feature.shape

    sequence_gate = torch.rand(5, 2, 3)
    sequence_feature = torch.rand(5, 2, 3, 4, 4)
    assert EleMulLayer()([sequence_gate, sequence_feature]).shape == sequence_feature.shape

    concat = ConcatLayer(dim=1)
    merged = concat([sequence_feature, sequence_feature])
    assert merged.shape == (5, 2, 6, 4, 4)
    layer = SimpleNamespace(layer_config={"type": "concat"}, layer=concat)
    NetworkGraph._validate_merge_inputs(
        layer, [sequence_feature, sequence_feature], sequence=True
    )


def test_multi_branch_sequence_accuracy_qat_and_gradients():
    torch.manual_seed(17)
    net = _network().train()
    sequence = torch.randint(0, 2, (2, 12, 6, 32, 32), dtype=torch.float32)

    output = net.sequence_qat_forward(sequence)
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in net.parameters()
    )

    net.eval()
    with torch.no_grad():
        original = net.sequence_accuracy_forward(sequence, weighted_mode="ORIGINAL")
        pim = net.sequence_accuracy_forward(
            sequence,
            weighted_mode="PIM",
            net_bit_weights=net.get_weights(),
            pim_profile="adc_digital_lif",
        )
    assert original.shape == pim.shape == (2, 8)
    assert torch.isfinite(original).all() and torch.isfinite(pim).all()


def test_concat_structure_records_channel_offsets_and_phase1_dag():
    net = _network()
    for layer in net.layer_list:
        if hasattr(layer, "bit_scale_list"):
            layer.bit_scale_list[1, 1] = 1.0

    sim_config = Path(__file__).resolve().parents[1] / "SimConfig.ini"
    ir = build_model_ir_from_mnsim_network(
        net,
        "inception_snn_v7_phase6",
        backend_kind="snn",
        sim_config=sim_config,
    )
    concat_nodes = [node for node in ir.layers if node.layer_type == "concat"]
    plan = build_round_plan(ir, sim_config)

    assert plan.feasible, plan.invalid_reason
    assert len(concat_nodes) == 3
    assert all(node.shape["Inputbit"] == 1 for node in ir.layers if node.is_weighted)
    concat_steps = [
        node
        for item in plan.execution_items
        for segment in item.ordered_segments
        for node in getattr(segment, "local_graph_nodes", [])
        if node.layer_type == "concat"
    ]
    assert len(concat_steps) == 3
    for node in concat_nodes:
        assert node.backend_attrs["time_policy"] == "same_timestep"
        assert node.backend_attrs["merge_ownership_policy"] == "concat_channel_segments"
        assert node.backend_attrs["value_domain"] == "binary_spike"
        assert node.shape["Inputbit"] == node.shape["outputbit"] == 1
        regions = node.backend_attrs["merge_input_regions"]
        assert len(regions) == len(node.input_layer_ids) == 3
        assert regions[0][0] == 0
        assert all(left[1] == right[0] for left, right in zip(regions, regions[1:]))

    report = execute_round_plan(
        plan,
        sim_config_path=str(sim_config),
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=_EchoStaticBackend(),
        model_inputs={
            "model_input": {
                "shape": {"Inputchannel": 6, "Inputsize": [32, 32], "Inputbit": 1},
                "tensor_role": "activation",
            }
        },
    )
    executed_concat_steps = [
        step
        for step in report.step_results
        if any("concat" in layer_id for layer_id in step["origin_layer_ids"])
    ]
    assert report.feasible
    assert len(executed_concat_steps) == 3 * net.sequence_length
    assert all(step["status"] == "ok" for step in executed_concat_steps)
