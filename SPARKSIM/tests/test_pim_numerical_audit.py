import torch

from MNSIM.Accuracy_Model.pim_numerical_audit import audit_pim_numerics
from MNSIM.Interface.network import NetworkGraph


def _small_network(point_shift):
    layer_config = [{"type": "fc", "in_features": 2, "out_features": 2}]
    quantize_config = [
        {"weight_bit": 3, "activation_bit": 3, "point_shift": point_shift}
    ]
    input_index = [[-1]]
    network = NetworkGraph(
        {
            "xbar_size": 4,
            "input_bit": 2,
            "weight_bit": 2,
            "ADC_quantize_bit": 2,
            "DAC_num": 4,
            "xbar_polarity": 1,
        },
        layer_config,
        quantize_config,
        input_index,
        {"activation_scale": 1.0, "activation_bit": 3, "input_shape": (1, 2)},
    )
    with torch.no_grad():
        network.layer_list[0].sublayer_list[0].weight.copy_(
            torch.tensor([[1.0, 1.0], [1.0, -1.0]])
        )
    return network


def _small_batch_norm_network():
    network = NetworkGraph(
        {
            "xbar_size": 4,
            "input_bit": 2,
            "weight_bit": 2,
            "ADC_quantize_bit": 2,
            "DAC_num": 4,
            "xbar_polarity": 1,
        },
        [
            {"type": "conv", "in_channels": 1, "out_channels": 1, "kernel_size": 1},
            {"type": "bn", "features": 1},
        ],
        [
            {"weight_bit": 3, "activation_bit": 3, "point_shift": -2},
            {"weight_bit": 3, "activation_bit": 3, "point_shift": -2},
        ],
        [[-1], [-1]],
        {"activation_scale": 1.0, "activation_bit": 3, "input_shape": (1, 1, 1, 1)},
    )
    with torch.no_grad():
        network.layer_list[0].sublayer_list[0].weight.fill_(1.0)
    return network


def _calibrate(network, *batches):
    network.calibrate_activation_scales(batches)
    return network


def test_audit_reports_adc_saturation_for_a_constrained_layer():
    input_tensor = torch.tensor([[3.0, 3.0], [3.0, -3.0]])
    result = audit_pim_numerics(
        _calibrate(_small_network(point_shift=3), input_tensor),
        input_tensor,
    )

    assert result["weighted_layer_count"] == 1
    assert result["layers"][0]["network_layer_index"] == 0
    assert result["layers"][0]["adc"]["conversion_count"] > 0
    assert result["layers"][0]["adc"]["upper_saturation_count"] > 0
    assert result["adc_summary"]["saturation_count"] > 0


def test_audit_none_is_exact_bit_slice_baseline_without_adc_clamping():
    input_tensor = torch.tensor([[3.0, 3.0], [3.0, -3.0]])
    result = audit_pim_numerics(
        _calibrate(_small_network(point_shift=-2), input_tensor),
        input_tensor,
        adc_action="NONE",
    )

    assert result["adc_summary"] == {
        "conversion_count": 0,
        "saturation_count": 0,
        "saturation_fraction": 0.0,
    }
    assert result["input_quantization_summary"]["saturation_count"] == 0
    assert result["output"]["metrics"]["relative_l2"] == 0.0


def test_frozen_reference_calibration_excludes_batch_norm_scale_updates():
    input_tensor = torch.tensor([[[[3.0]]], [[[-3.0]]]])
    result = audit_pim_numerics(
        _calibrate(_small_batch_norm_network(), input_tensor),
        input_tensor,
        adc_action="NONE",
    )

    assert result["calibration_mode"] == "frozen_reference"
    assert result["output"]["metrics"]["relative_l2"] == 0.0


def test_audit_reports_input_code_clipping_separately_from_adc_clipping():
    calibration_tensor = torch.tensor([[3.0, 3.0], [3.0, -3.0]])
    result = audit_pim_numerics(
        _calibrate(_small_network(point_shift=-2), calibration_tensor),
        torch.tensor([[10.0, 10.0], [10.0, -10.0]]),
        adc_action="NONE",
    )

    diagnostic = result["layers"][0]["adc"]
    assert diagnostic["conversion_count"] == 0
    assert diagnostic["input_upper_saturation_count"] > 0
    assert diagnostic["input_saturation_fraction"] > 0.0
    assert result["input_quantization_summary"]["saturation_count"] > 0
