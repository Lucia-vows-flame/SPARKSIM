from pathlib import Path

import pytest
import torch

from MNSIM.Interface.network import NetworkGraph
from round_orchestrated_compat import enrich_legacy_element_sum_markers


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _residual_network():
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
            {"weight_bit": 4, "activation_bit": 5, "point_shift": -2},
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


def test_legacy_checkpoint_with_stored_residual_range_runs_without_calibration():
    source = _residual_network()
    source.calibrate_activation_scales(torch.tensor([[1.0, -1.0]]))
    with torch.no_grad():
        source.layer_list[1].last_value.fill_(2.5)
    legacy_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.endswith("post_add_quantizer_version")
    }

    target = _residual_network()
    target.load_state_dict(
        enrich_legacy_element_sum_markers(target, legacy_state), strict=True
    )
    target.eval()

    assert int(target.layer_list[1].post_add_quantizer_version.item()) == 1
    output = target(torch.tensor([[1.0, -1.0]]), "FIX_TRAIN")

    assert output.shape == (1, 2)
    assert torch.isfinite(output).all()


def test_legacy_checkpoint_without_residual_range_still_requires_calibration():
    source = _residual_network()
    source.calibrate_activation_scales(torch.tensor([[1.0, -1.0]]))
    with torch.no_grad():
        source.layer_list[1].last_value.fill_(-1.0)
    legacy_state = {
        key: value.clone()
        for key, value in source.state_dict().items()
        if not key.endswith("post_add_quantizer_version")
    }

    target = _residual_network()
    target.load_state_dict(
        enrich_legacy_element_sum_markers(target, legacy_state), strict=True
    )
    target.eval()

    with pytest.raises(RuntimeError, match="activation scale is not calibrated"):
        target(torch.tensor([[1.0, -1.0]]), "FIX_TRAIN")


@pytest.mark.parametrize(
    "checkpoint",
    [
        _REPO_ROOT
        / "archive/legacy_master_20260925/MNSIM-2.0-master/model/"
        / "cifar10_resnet18_params.pth",
        _REPO_ROOT
        / "archive/legacy_master_20260925/MNSIM-2.0-master/MNSIM/Weight_update/"
        / "cifar10_resnet18_params.pth",
    ],
)
def test_real_legacy_resnet_checkpoint_is_available_or_skipped(checkpoint):
    if not checkpoint.is_file():
        pytest.skip(f"legacy ResNet18 checkpoint is unavailable: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(state, dict)
