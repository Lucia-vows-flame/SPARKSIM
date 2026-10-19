import pytest
import torch

from MNSIM.Accuracy_Model.pim_numerical_audit import audit_pim_numerics
from MNSIM.Interface import quantize
from MNSIM.Interface.network import NetworkGraph


def _residual_network(post_add_bit=3):
    network = NetworkGraph(
        {
            "xbar_size": 4,
            "input_bit": 2,
            "weight_bit": 2,
            "ADC_quantize_bit": 8,
            "DAC_num": 4,
            "xbar_polarity": 1,
        },
        [
            {"type": "fc", "in_features": 2, "out_features": 2},
            {"type": "element_sum", "input_index": [-1, -2]},
            {"type": "fc", "in_features": 2, "out_features": 2},
        ],
        [
            {"weight_bit": 4, "activation_bit": 5, "point_shift": -2},
            {"weight_bit": 4, "activation_bit": post_add_bit, "point_shift": -2},
            {"weight_bit": 4, "activation_bit": 4, "point_shift": -2},
        ],
        [[-1], [-1, -2], [-1]],
        {"activation_scale": 0.25, "activation_bit": 5, "input_shape": (1, 2)},
    )
    with torch.no_grad():
        identity = torch.eye(2)
        network.layer_list[0].sublayer_list[0].weight.copy_(identity)
        network.layer_list[2].sublayer_list[0].weight.copy_(identity)
    return network


def _calibrated_residual_network(post_add_bit=3):
    network = _residual_network(post_add_bit)
    calibration_batch = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    network.calibrate_activation_scales(calibration_batch)
    return network, calibration_batch


def test_residual_add_uses_its_own_configured_quantization_state():
    network, input_tensor = _calibrated_residual_network(post_add_bit=3)

    result = audit_pim_numerics(network, input_tensor, adc_action="NONE")

    post_add = result["layers"][1]
    following_fc = result["layers"][2]
    assert post_add["layer_type"] == "element_sum"
    assert post_add["output_quantization"]["activation_bit"] == 3
    assert following_fc["input_quantization"]["activation_bit"] == 3
    assert following_fc["input_quantization"]["activation_scale"] == pytest.approx(
        post_add["output_quantization"]["activation_scale"]
    )
    assert int(network.layer_list[2].bit_scale_list[0, 0].item()) == 3
    assert result["output"]["metrics"]["relative_l2"] == 0.0


def test_pim_weighted_layer_uses_explicit_post_add_state_not_legacy_global():
    network, input_tensor = _calibrated_residual_network(post_add_bit=3)
    network.eval()
    network(input_tensor, "FIX_TRAIN")
    post_add_layer = network.layer_list[1]
    following_fc = network.layer_list[2]
    post_add_state = post_add_layer.output_activation_quantization()
    post_add_value = network.layer_list[1].forward(
        [
            network.layer_list[0].forward(
                input_tensor,
                "FIX_TRAIN",
                input_quantization=network.input_activation_quantization(),
            ),
            input_tensor,
        ],
        "FIX_TRAIN",
    )
    bit_weights = following_fc.get_bit_weights()

    quantize.last_activation_bit = 2
    quantize.last_activation_scale = 1000.0
    first = following_fc.set_weights_forward(
        post_add_value,
        bit_weights,
        "NONE",
        input_quantization=post_add_state,
    )
    quantize.last_activation_bit = 15
    quantize.last_activation_scale = 1e-6
    second = following_fc.set_weights_forward(
        post_add_value,
        bit_weights,
        "NONE",
        input_quantization=post_add_state,
    )

    torch.testing.assert_close(first, second)


def test_eval_freezes_residual_add_scale_after_explicit_calibration():
    network, input_tensor = _calibrated_residual_network(post_add_bit=4)
    post_add_layer = network.layer_list[1]
    before = post_add_layer.last_value.detach().clone()

    network.eval()
    network(20 * input_tensor, "FIX_TRAIN")

    torch.testing.assert_close(post_add_layer.last_value, before)


def test_eval_requires_calibration_for_new_activation_boundaries():
    network = _residual_network()
    network.eval()

    with pytest.raises(RuntimeError, match="activation scale is not calibrated"):
        network(torch.tensor([[1.0, -1.0]]), "FIX_TRAIN")


def test_legacy_checkpoint_is_marked_for_post_add_calibration():
    source = _residual_network()
    legacy_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.endswith("post_add_quantizer_version")
    }
    target = _residual_network()

    target.load_state_dict(legacy_state)

    post_add_layer = target.layer_list[1]
    assert int(post_add_layer.post_add_quantizer_version.item()) == 0
    with torch.no_grad():
        for layer in target.layer_list:
            if hasattr(layer, "last_value"):
                layer.last_value.fill_(1.0)
    target.eval()
    with pytest.raises(RuntimeError, match="checkpoint predates post-add"):
        target(torch.tensor([[1.0, -1.0]]), "FIX_TRAIN")

    target.calibrate_activation_scales(torch.tensor([[1.0, -1.0]]), reset=True)
    assert int(post_add_layer.post_add_quantizer_version.item()) == 1
