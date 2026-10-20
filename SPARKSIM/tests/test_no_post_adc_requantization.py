import torch

from MNSIM.Interface.quantize import QuantizeLayer, activation_quantization_state


def test_weighted_layer_does_not_round_accumulated_adc_result_again(monkeypatch):
    layer = object.__new__(QuantizeLayer)
    torch.nn.Module.__init__(layer)
    layer.hardware_config = {
        "xbar_polarity": 1,
        "weight_bit": 2,
        "input_bit": 2,
        "ADC_quantize_bit": 5,
        "xbar_size": 1,
        "type": 0,
        "ADC_quantization_mode": "READC_LLOYD_MAX",
        "ADC_readc_table": {"boundaries": [0.3], "levels": [0.25, 0.75]},
    }
    layer.quantize_config = {"weight_bit": 2, "activation_bit": 8, "point_shift": 0}
    layer.layer_config = {
        "type": "fc", "extend_ADC_bitwidth": 0,
    }
    layer.split_input = 1
    layer.sublayer_list = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False)])
    layer.last_value = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    layer.bit_scale_list = torch.nn.Parameter(
        torch.tensor([[8.0, 1 / 127], [2.0, 1.0], [8.0, 1 / 127]]),
        requires_grad=False,
    )
    layer.eval()
    monkeypatch.setattr(
        layer, "output_activation_quantization",
        lambda _state=None: activation_quantization_state(8, 1.0),
    )
    result = layer.set_weights_forward(
        torch.tensor([[1.0]]), {"split0_weight0": __import__('numpy').array([[1.0]])},
        "SCALE", input_quantization=activation_quantization_state(8, 1.0),
    )
    # A post-ADC activation quantizer would force this onto an n/127 grid.
    assert not torch.allclose(result * 127, torch.round(result * 127))
