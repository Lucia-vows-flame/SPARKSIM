from pathlib import Path

import torch

from MNSIM.Interface.network import get_net
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_network
from MNSIM.Scheduling.phase1.round_planner import build_round_plan


HW = dict(
    xbar_size=128,
    input_shape=(1, 6, 32, 32),
    weight_quantization_bit=9,
    DAC_num=128,
    input_bit=1,
    weight_bit=8,
    ADC_quantize_bit=5,
    xbar_polarity=2,
)


def test_mobilenet_v1_snn_graph_and_sequence_forward():
    net = get_net(HW, "mobilenet_snn_v7_phase6", 8).eval()
    types = [layer.layer_config["type"] for layer in net.layer_list]
    assert types.count("conv") == 27
    assert types.count("fc") == 1
    assert types.count("lif") == 28
    assert types.count("pooling") == 0
    assert sum(layer.layer_config.get("depthwise") == "separable" for layer in net.layer_list) == 13
    assert net.layer_list[-2].layer_config["in_features"] == 1024
    with torch.no_grad():
        output = net.sequence_accuracy_forward(
            torch.zeros(1, 2, 6, 32, 32), weighted_mode="ORIGINAL"
        )
    assert output.shape == (1, 8)
    assert net.sequence_length == 30


def test_mobilenet_v1_snn_qat_forward_has_finite_gradients():
    net = get_net(HW, "mobilenet_snn_v7_phase6", 8).train()
    inputs = torch.randint(0, 2, (1, 2, 6, 32, 32)).float()
    output = net.sequence_qat_forward(inputs)
    output.sum().backward()
    assert output.shape == (1, 8)
    assert torch.isfinite(output).all()
    assert any(parameter.grad is not None for parameter in net.parameters())

    net.eval()
    with torch.no_grad():
        pim = net.sequence_accuracy_forward(
            inputs,
            weighted_mode="PIM",
            net_bit_weights=net.get_weights(),
            pim_profile="adc_digital_lif",
        )
    assert pim.shape == (1, 8)
    assert torch.isfinite(pim).all()


def test_mobilenet_v1_snn_phase1_plan_supports_bn_domain_propagation():
    net = get_net(HW, "mobilenet_snn_v7_phase6", 8)
    for layer in net.layer_list:
        if hasattr(layer, "bit_scale_list"):
            layer.bit_scale_list[1, 1] = 1.0
    sim_config = Path(__file__).resolve().parents[1] / "SimConfig.ini"
    model_ir = build_model_ir_from_mnsim_network(
        net,
        "mobilenet_snn_v7_phase6",
        backend_kind="snn",
        sim_config=sim_config,
    )
    plan = build_round_plan(model_ir, sim_config)
    assert plan.feasible, plan.invalid_reason
    assert all(
        node.backend_attrs.get("accumulator_domain")
        for node in model_ir.layers
        if node.layer_type in {"conv", "bn", "lif"}
    )
