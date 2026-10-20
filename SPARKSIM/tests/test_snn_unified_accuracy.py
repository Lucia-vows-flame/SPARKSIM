import torch

from MNSIM.Accuracy_Model.snn_accuracy_contract import BINARY_SPIKE_DOMAIN
from MNSIM.Interface.interface import TrainTestInterface
from MNSIM.Interface.network import NetworkGraph, get_net
from MNSIM.Interface.quantize import QuantizeLayer


def _hardware_config():
    return {
        "xbar_size": 128,
        "input_bit": 8,
        "weight_bit": 8,
        "ADC_quantize_bit": 5,
        "DAC_num": 128,
        "xbar_polarity": 2,
        "ADC_quantization_mode": "UNIFORM",
        "ADC_adaptive_thresholds": None,
    }


def _tiny_two_stage_snn():
    net = NetworkGraph(
        _hardware_config(),
        [
            {"type": "conv", "in_channels": 1, "out_channels": 1, "kernel_size": 1},
            {"type": "lif", "tau": 50.5016666556, "v_th": 0.01, "v_reset": 0.0},
            {"type": "conv", "in_channels": 1, "out_channels": 1, "kernel_size": 1},
            {"type": "lif", "tau": 50.5016666556, "v_th": 0.01, "v_reset": 0.0},
        ],
        [
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
        ],
        [[-1], [-1], [-1], [-1]],
        {
            "activation_scale": 1.0,
            "activation_bit": 9,
            "input_shape": (1, 1, 1, 1),
        },
    )
    with torch.no_grad():
        for index in (0, 2):
            layer = net.layer_list[index]
            layer.sublayer_list[0].weight.fill_(1.0)
            layer.bit_scale_list[1, 1] = 1.0
            layer.last_value.fill_(1.0)
    net.eval()
    return net


def test_unified_sequence_accuracy_requires_valid_mode_and_pim_weights():
    net = get_net(_hardware_config(), cate="standard_vgg8_snn", num_classes=8)
    sequence = torch.zeros(1, 2, 2, 32, 32)

    try:
        net.sequence_accuracy_forward(sequence, weighted_mode="invalid")
    except ValueError as error:
        assert "weighted mode" in str(error)
    else:
        raise AssertionError("invalid weighted mode must fail")

    try:
        net.sequence_accuracy_forward(sequence, weighted_mode="PIM")
    except ValueError as error:
        assert "net_bit_weights" in str(error)
    else:
        raise AssertionError("PIM mode without mapped weights must fail")


def test_binary_spike_domain_reaches_every_weighted_layer_without_conversion(
    monkeypatch,
):
    net = _tiny_two_stage_snn()
    observed = []
    original = QuantizeLayer.set_weights_forward

    def capture(self, input, bit_weights, adc_action, **kwargs):
        observed.append((input.detach().clone(), kwargs["input_quantization"]))
        return original(self, input, bit_weights, adc_action, **kwargs)

    monkeypatch.setattr(QuantizeLayer, "set_weights_forward", capture)
    sequence = torch.zeros(1, 2, 1, 1, 1)
    sequence[:, 0] = 1
    net.sequence_accuracy_forward(
        sequence,
        weighted_mode="PIM",
        adc_action="SCALE",
        net_bit_weights=net.get_weights(),
        pim_profile="adc_digital_lif",
    )

    assert len(observed) == 2
    assert all(domain is BINARY_SPIKE_DOMAIN for _value, domain in observed)
    assert all(
        set(value.unique().tolist()) <= {0.0, 1.0}
        for value, _domain in observed
    )


def test_accuracy_adapter_preserves_legacy_argument_and_selects_adc_profile():
    net = _tiny_two_stage_snn()
    adapter = object.__new__(TrainTestInterface)
    adapter.net = net
    sequence = torch.zeros(1, 2, 1, 1, 1)
    sequence[:, 0] = 1

    output = adapter._accuracy_forward(
        sequence,
        method="SET_WEIGHTS",
        adc_action="SCALE",
        net_bit_weights=net.get_weights(),
        reference_sim_config="legacy-call-site.ini",
    )

    assert output.shape == (1, 1, 1, 1)
