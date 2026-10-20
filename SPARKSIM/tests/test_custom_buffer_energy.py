import configparser
from pathlib import Path

import pytest

from MNSIM.Hardware_Model.ADC import ADC
from MNSIM.Hardware_Model.Buffer import buffer


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_custom_buffer_config(tmp_path, **overrides):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(_REPO_ROOT / "SimConfig.ini", encoding="UTF-8")
    for option in (
        "Buffer_AccessEnergyPerBit",
        "Buffer_LeakagePowerPerBit",
        "Buffer_AccessLatency",
        "Buffer_IncludeLeakageEnergy",
    ):
        parser.remove_option("Architecture level", option)
    parser.set("Architecture level", "Buffer_Choice", "0")
    parser.set("Architecture level", "Buffer_ReadEnergy", "2.5")
    parser.set("Architecture level", "Buffer_WriteEnergy", "4.0")
    parser.set("Architecture level", "Buffer_LeakagePower", "0.25")
    parser.set("Architecture level", "Buffer_Cycle", "0.5")
    parser.set("Architecture level", "Buffer_Bitwidth", "64")
    for name, value in overrides.items():
        parser.set("Architecture level", name, str(value))
    path = tmp_path / "custom-buffer.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return path


def _write_legacy_lookup_buffer_config(tmp_path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(_REPO_ROOT / "SimConfig.ini", encoding="UTF-8")
    for option in (
        "Buffer_AccessEnergyPerBit",
        "Buffer_LeakagePowerPerBit",
        "Buffer_AccessLatency",
        "Buffer_IncludeLeakageEnergy",
        "Buffer_ReadEnergy",
        "Buffer_WriteEnergy",
        "Buffer_LeakagePower",
        "Buffer_Cycle",
    ):
        parser.remove_option("Architecture level", option)
    parser.set("Architecture level", "Buffer_Choice", "1")
    parser.set("Architecture level", "Buffer_Technology", "90")
    parser.set("Architecture level", "Buffer_Bitwidth", "64")
    path = tmp_path / "legacy-lookup-buffer.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return path


def test_custom_buffer_energy_model_uses_configured_per_access_energy(tmp_path):
    config_path = _write_custom_buffer_config(tmp_path)
    tested = buffer(str(config_path), buf_level=1, default_buf_size=4)

    tested.calculate_buf_read_power()
    tested.calculate_buf_write_power()
    tested.calculate_buf_read_latency(rdata=16)
    tested.calculate_buf_write_latency(wdata=16)
    tested.calculate_buf_read_energy(rdata=16)
    tested.calculate_buf_write_energy(wdata=16)

    assert tested.custom_energy_model is True
    assert tested.buf_cycle == pytest.approx(0.5)
    assert tested.dynamic_buf_rpower == pytest.approx(5.0)
    assert tested.dynamic_buf_wpower == pytest.approx(8.0)
    assert tested.buf_rpower == pytest.approx(5.25)
    assert tested.buf_wpower == pytest.approx(8.25)
    assert tested.buf_rlatency == pytest.approx(1.0)
    assert tested.buf_wlatency == pytest.approx(1.0)
    assert tested.buf_renergy == pytest.approx(0.00525)
    assert tested.buf_wenergy == pytest.approx(0.00825)


def test_custom_buffer_energy_model_rejects_incomplete_config(tmp_path):
    config_path = _write_custom_buffer_config(tmp_path)
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="UTF-8")
    parser.remove_option("Architecture level", "Buffer_ReadEnergy")
    with config_path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)

    with pytest.raises(ValueError, match="Buffer_ReadEnergy"):
        buffer(str(config_path), buf_level=1, default_buf_size=4)


def test_legacy_sram_buffer_still_uses_lookup_tables(tmp_path):
    tested = buffer(
        str(_write_legacy_lookup_buffer_config(tmp_path)),
        buf_level=1,
        default_buf_size=4,
    )
    tested.calculate_buf_read_power()
    tested.calculate_buf_write_power()

    assert tested.custom_energy_model is False
    assert tested.buf_cycle == pytest.approx(20.0)
    assert tested.dynamic_buf_rpower > 0
    assert tested.dynamic_buf_wpower > 0


def test_g8_600plus_profile_keeps_g8_topology_and_six_bit_adc():
    profile = _REPO_ROOT / "configs" / "hardware_profiles" / "g8_600plus_ann.ini"
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(profile, encoding="UTF-8")

    assert parser.get("Crossbar level", "Xbar_Size") == "128,128"
    assert parser.get("Process element level", "Group_Num") == "2"
    assert parser.get("Tile level", "PE_Num") == "4,4"
    assert parser.get("Architecture level", "Tile_Num") == "6,6"
    assert parser.getint("Process element level", "ADC_Num") == 128
    assert parser.getint("Interface level", "ADC_Precision") == 6

    profile_buffer = buffer(str(profile), buf_level=1, default_buf_size=4)
    profile_buffer.calculate_buf_read_power()
    profile_adc = ADC(str(profile))
    profile_adc.calculate_ADC_power()
    profile_adc.calculate_ADC_latency()
    profile_adc.calculate_ADC_energy()

    assert profile_buffer.custom_energy_model is True
    assert profile_buffer.custom_read_energy_pj == pytest.approx(0.001)
    assert profile_buffer.custom_write_energy_pj == pytest.approx(0.002)
    assert profile_adc.ADC_precision == 6
    assert profile_adc.ADC_power == pytest.approx(1e-6)
    assert profile_adc.ADC_latency == pytest.approx(1.0)
    assert profile_adc.ADC_energy == pytest.approx(1e-6)
