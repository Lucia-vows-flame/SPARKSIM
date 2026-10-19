import pytest
import torch

from MNSIM.Interface.network import get_net


_HARDWARE_CONFIG = {
    "xbar_size": 128,
    "input_shape": (1, 6, 32, 32),
    "weight_quantization_bit": 9,
    "DAC_num": 128,
    "input_bit": 1,
    "weight_bit": 9,
    "ADC_quantize_bit": 5,
    "xbar_polarity": 2,
}


@pytest.mark.parametrize(
    "network",
    [
        "lenet_snn_v7_phase6",
        "alexnet_snn_v7_phase6",
        "vgg8_snn_v7_phase6",
        "vgg11_snn_v7_phase6",
        "vgg13_snn_v7_phase6",
        "vgg16_snn_v7_phase6",
    ],
)
def test_v7_sequential_snn_contract_has_no_pooling_or_branches(network):
    net = get_net(_HARDWARE_CONFIG, cate=network, num_classes=8)
    layer_types = [layer.layer_config["type"] for layer in net.layer_list]

    assert net.input_params["input_shape"] == (1, 6, 32, 32)
    assert "pooling" not in layer_types
    assert layer_types.count("lif") >= 1
    assert all(input_index == [-1] for input_index in net.input_index_list)


def test_v7_vgg8_name_is_the_formal_name_for_the_existing_standard_topology():
    canonical = get_net(_HARDWARE_CONFIG, cate="vgg8_snn_v7_phase6", num_classes=8)
    legacy_alias = get_net(
        _HARDWARE_CONFIG,
        cate="standard_vgg8_snn_v7_phase6",
        num_classes=8,
    )

    assert [layer.layer_config for layer in canonical.layer_list] == [
        layer.layer_config for layer in legacy_alias.layer_list
    ]


def test_v7_sequence_qat_forward_supports_binary_input_and_bptt():
    net = get_net(_HARDWARE_CONFIG, cate="vgg8_snn_v7_phase6", num_classes=8)
    net.train()
    sequence = torch.randint(0, 2, (1, 2, 6, 32, 32), dtype=torch.float32)

    output = net.sequence_qat_forward(sequence)
    assert output.shape == (1, 8)
    assert output.requires_grad
    output.sum().backward()

    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in net.parameters()
    )
    assert all(
        float(layer.last_value.item()) > 0
        for layer in net.layer_list
        if layer.layer_config["type"] == "conv"
    )
