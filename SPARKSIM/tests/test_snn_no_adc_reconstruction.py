import numpy as np
import torch

import MNSIM.Interface.quantize as quantize_module
from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    BINARY_SPIKE_DOMAIN,
    FRONTEND_LUT_LIF_LUT,
)
from MNSIM.Interface.network import NetworkGraph
from MNSIM.Interface.quantize import QuantizeLayer


def _fc_layer():
    layer = object.__new__(QuantizeLayer)
    torch.nn.Module.__init__(layer)
    layer.hardware_config = {
        "xbar_polarity": 2,
        "weight_bit": 8,
        "input_bit": 1,
        "ADC_quantize_bit": 5,
        "xbar_size": 2,
        "type": 0,
        "ADC_quantization_mode": "READC_LLOYD_MAX",
    }
    layer.quantize_config = {
        "weight_bit": 9,
        "activation_bit": 9,
        "point_shift": 0,
    }
    layer.layer_config = {"type": "fc", "extend_ADC_bitwidth": 0}
    layer.split_input = 2
    layer.sublayer_list = torch.nn.ModuleList(
        [torch.nn.Linear(2, 1, bias=False)]
    )
    layer.last_value = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
    layer.bit_scale_list = torch.nn.Parameter(
        torch.tensor([[9.0, 1.0], [9.0, 0.25], [9.0, 1.0 / 255.0]]),
        requires_grad=False,
    )
    layer.eval()
    return layer


def test_no_adc_reconstruction_matches_the_bit_sliced_weighted_sum(monkeypatch):
    layer = _fc_layer()

    def forbidden_adc(*_args, **_kwargs):
        raise AssertionError("LUT profile must not call an ADC quantizer")

    def forbidden_output_quantizer(*_args, **_kwargs):
        raise AssertionError("LUT profile must not quantize weighted output")

    monkeypatch.setattr(quantize_module, "_adc_quantize", forbidden_adc)
    monkeypatch.setattr(
        layer, "output_activation_quantization", forbidden_output_quantizer
    )
    bit_weights = {
        "split0_weight0_positive": np.array([[2.0, 0.0]]),
        "split0_weight0_negative": np.array([[0.0, 1.0]]),
    }
    spikes = torch.tensor([[1.0, 1.0]])

    actual = layer.set_weights_forward(
        spikes,
        bit_weights,
        adc_action=None,
        numerical_mode="NO_ADC_RECONSTRUCTION",
        input_quantization=BINARY_SPIKE_DOMAIN,
    )

    assert torch.equal(actual, torch.tensor([[0.25]]))


def test_no_adc_reconstruction_rejects_nonbinary_weighted_input():
    layer = _fc_layer()
    with torch.no_grad():
        try:
            layer.set_weights_forward(
                torch.tensor([[1.0, 0.25]]),
                {
                    "split0_weight0_positive": np.array([[1.0, 0.0]]),
                    "split0_weight0_negative": np.array([[0.0, 0.0]]),
                },
                adc_action=None,
                numerical_mode="NO_ADC_RECONSTRUCTION",
                input_quantization=BINARY_SPIKE_DOMAIN,
            )
        except ValueError as error:
            assert "binary spike" in str(error)
        else:
            raise AssertionError("nonbinary LUT-profile input must fail")


def test_lut_profile_routes_weighted_sequence_compute_to_no_adc(monkeypatch):
    layer = _fc_layer()
    calls = []

    def record_weighted_call(
        _layer,
        input_tensor,
        bit_weights,
        adc_action,
        **kwargs,
    ):
        calls.append((bit_weights, adc_action, kwargs))
        return torch.zeros(
            (input_tensor.shape[0], 1),
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )

    def forbidden_output_quantizer(*_args, **_kwargs):
        raise AssertionError("LUT weighted output must not have a quantization state")

    monkeypatch.setattr(QuantizeLayer, "set_weights_forward", record_weighted_call)
    monkeypatch.setattr(
        layer, "output_activation_quantization", forbidden_output_quantizer
    )

    network = object.__new__(NetworkGraph)
    torch.nn.Module.__init__(network)
    network.layer_list = torch.nn.ModuleList([layer])
    network.input_index_list = [[-1]]
    bit_weights = {"sentinel": object()}

    output = network.sequence_accuracy_forward(
        torch.ones((1, 2, 1, 1, 2)),
        weighted_mode="PIM",
        adc_action="SCALE",
        net_bit_weights=[bit_weights],
        pim_profile=FRONTEND_LUT_LIF_LUT,
    )

    assert output.shape == (1, 1)
    assert calls == [
        (
            bit_weights,
            None,
            {
                "input_quantization": BINARY_SPIKE_DOMAIN,
                "numerical_mode": "NO_ADC_RECONSTRUCTION",
            },
        )
    ]
