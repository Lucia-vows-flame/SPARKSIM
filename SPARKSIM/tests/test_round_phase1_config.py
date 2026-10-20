import configparser

import pytest

from MNSIM.Scheduling.phase1.config import build_hardware_projection
from MNSIM.Scheduling.round_types import INVALID_SIM_CONFIG, Phase1Invalid


def _base_config():
    return {
        "Xbar_Size": [128, 128],
        "Tile_Num": [2, 2],
        "PE_Num": [2, 2],
        "Group_Num": 1,
        "Device_Level": 4,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }


def test_hardware_projection_computes_physical_xbar_nums():
    hw = build_hardware_projection(_base_config())

    assert hw.xbar_size == [128, 128]
    assert hw.xbar_row == 128
    assert hw.xbar_col == 128
    assert hw.xbar_nums == 32
    assert hw.device_level == 4
    assert hw.weight_precision == 8
    assert hw.xbar_polarity == 2


@pytest.mark.parametrize(
    ("device_level", "expected_bits"),
    [(2, 1), (4, 2), (8, 3), (16, 4), (256, 8)],
)
def test_hardware_projection_exposes_generic_device_bit_width(
    device_level,
    expected_bits,
):
    config = _base_config()
    config["Device_Level"] = device_level

    hardware = build_hardware_projection(config)

    assert hardware.device_level == device_level
    assert hardware.device_bits == expected_bits
    assert hardware.to_dict()["device_bits"] == expected_bits


@pytest.mark.parametrize("device_level", [1, 3, 6, 12])
def test_hardware_projection_rejects_non_bit_precise_device_levels(device_level):
    config = _base_config()
    config["Device_Level"] = device_level

    with pytest.raises(Phase1Invalid) as exc_info:
        build_hardware_projection(config)

    assert exc_info.value.code == "INVALID_HARDWARE_PROJECTION"
    assert "Device_Level" in exc_info.value.detail


def test_hardware_projection_reads_mnsim_simconfig_ini_path(tmp_path):
    config_path = tmp_path / "SimConfig.ini"
    config_path.write_text(
        """
[Device level]
Device_Level = 4

[Crossbar level]
Xbar_Size = 128,128

[Process element level]
Xbar_Polarity = 2
Group_Num = 1

[Tile level]
PE_Num = 2,2

[Architecture level]
Tile_Num = 2,2

[Algorithm Configuration]
Weight_precision = 8
""",
        encoding="utf-8",
    )

    hw = build_hardware_projection(config_path)

    assert hw.xbar_size == [128, 128]
    assert hw.xbar_nums == 32
    assert hw.weight_precision == 8


def test_hardware_projection_reads_configparser_object():
    parser = configparser.ConfigParser()
    parser.read_dict(
        {
            "Device level": {"Device_Level": "4"},
            "Crossbar level": {"Xbar_Size": "128,128"},
            "Process element level": {"Xbar_Polarity": "2", "Group_Num": "1"},
            "Tile level": {"PE_Num": "2,2"},
            "Architecture level": {"Tile_Num": "2,2"},
            "Algorithm Configuration": {"Weight_precision": "8"},
        }
    )

    hw = build_hardware_projection(parser)

    assert hw.xbar_nums == 32
    assert hw.device_level == 4


def test_device_level_and_weight_precision_do_not_change_physical_capacity_but_polarity_does():
    base_hw = build_hardware_projection(_base_config())
    precision_config = _base_config()
    precision_config["Device_Level"] = 16
    precision_config["Weight_precision"] = 4
    precision_hw = build_hardware_projection(precision_config)
    single_polarity_config = dict(precision_config)
    single_polarity_config["Xbar_Polarity"] = 1
    single_polarity_hw = build_hardware_projection(single_polarity_config)

    assert base_hw.xbar_nums == 32
    assert precision_hw.xbar_nums == 32
    assert single_polarity_hw.xbar_nums == 16
    assert precision_hw.device_level == 16
    assert precision_hw.weight_precision == 4


@pytest.mark.parametrize("group_num", [0, -1])
def test_hardware_projection_requires_positive_group_num(group_num):
    config = _base_config()
    config["Group_Num"] = group_num

    with pytest.raises(Phase1Invalid) as exc_info:
        build_hardware_projection(config)

    assert exc_info.value.code == "INVALID_HARDWARE_PROJECTION"
    assert "Group_Num" in exc_info.value.detail


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("Group_Num", 1.5),
        ("Xbar_Polarity", 2.5),
        ("Group_Num", "not-an-integer"),
        ("Xbar_Polarity", "not-an-integer"),
        ("Group_Num", True),
        ("Xbar_Polarity", True),
    ],
)
def test_hardware_projection_rejects_non_integer_scalar_config_values(field, value):
    config = _base_config()
    config[field] = value

    with pytest.raises(Phase1Invalid) as exc_info:
        build_hardware_projection(config)

    assert exc_info.value.code == INVALID_SIM_CONFIG
    assert field in exc_info.value.detail


def test_hardware_projection_rejects_unsupported_integral_polarity():
    config = _base_config()
    config["Xbar_Polarity"] = 3

    with pytest.raises(Phase1Invalid) as exc_info:
        build_hardware_projection(config)

    assert exc_info.value.code == "INVALID_HARDWARE_PROJECTION"
    assert "unsupported Xbar_Polarity 3" in exc_info.value.detail


def test_invalid_hardware_projection_reports_phase1_invalid():
    config = _base_config()
    config["Tile_Num"] = [0, 2]

    try:
        build_hardware_projection(config)
    except Phase1Invalid as exc:
        assert exc.code == "INVALID_HARDWARE_PROJECTION"
        assert "xbar_nums" in exc.detail
    else:
        raise AssertionError("expected Phase1Invalid")
