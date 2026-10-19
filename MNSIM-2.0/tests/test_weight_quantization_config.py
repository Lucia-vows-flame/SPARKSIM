from pathlib import Path

import pytest

from MNSIM.Interface.network import get_net
from MNSIM.Interface.snn_hw_config import read_weight_quantization_bit


def _config(path: Path, value: str | None = None) -> Path:
    body = "[Algorithm Configuration]\n"
    if value is not None:
        body += f"Weight_Quantization_Bit = {value}\n"
    path.write_text(body, encoding="utf-8")
    return path


def test_weight_quantization_config_defaults_to_legacy_nine_bits(tmp_path):
    assert read_weight_quantization_bit(str(_config(tmp_path / "SimConfig.ini"))) == 9


def test_weight_quantization_config_reads_custom_width(tmp_path):
    assert read_weight_quantization_bit(
        str(_config(tmp_path / "SimConfig.ini", "7"))
    ) == 7


@pytest.mark.parametrize("value", ["0", "1", "-3"])
def test_weight_quantization_config_rejects_unsigned_or_empty_signed_width(
    tmp_path, value
):
    with pytest.raises(ValueError, match="at least 2"):
        read_weight_quantization_bit(str(_config(tmp_path / "SimConfig.ini", value)))


def test_network_applies_shared_weight_quantization_width_to_weighted_layers():
    net = get_net(
        {
            "xbar_size": 128,
            "input_bit": 8,
            "weight_bit": 8,
            "weight_quantization_bit": 7,
            "ADC_quantize_bit": 5,
            "DAC_num": 128,
            "xbar_polarity": 2,
        },
        cate="standard_vgg8_snn",
        num_classes=8,
    )
    weighted = [
        layer
        for layer in net.layer_list
        if hasattr(layer, "quantize_config") and "weight_bit" in layer.quantize_config
    ]
    assert weighted
    assert {int(layer.quantize_config["weight_bit"]) for layer in weighted} == {7}


def test_network_keeps_legacy_nine_bit_default_without_new_hardware_key():
    net = get_net(
        {
            "xbar_size": 128,
            "input_bit": 8,
            "weight_bit": 8,
            "ADC_quantize_bit": 5,
            "DAC_num": 128,
            "xbar_polarity": 2,
        },
        cate="standard_vgg8_snn",
        num_classes=8,
    )
    weighted = [
        layer
        for layer in net.layer_list
        if hasattr(layer, "quantize_config") and "weight_bit" in layer.quantize_config
    ]
    assert {int(layer.quantize_config["weight_bit"]) for layer in weighted} == {9}
