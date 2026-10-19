import configparser
import math
from pathlib import Path

import pytest

from MNSIM.Hardware_Model.ADC import ADC
from MNSIM.Hardware_Model.Adder import adder
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Hardware_Model.DAC import DAC
from MNSIM.Hardware_Model.JointModule import JointModule
from MNSIM.Hardware_Model.Reg import reg
from MNSIM.Hardware_Model.ShiftReg import shiftreg


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PROFILE = _REPO_ROOT / "configs" / "hardware_profiles" / "g8_literature_peripheral_ann.ini"
_G8_REFERENCE = _REPO_ROOT / "configs" / "hardware_profiles" / "g8_600plus_ann.ini"
_FORMAL_SIM_CONFIG = _REPO_ROOT / "SimConfig.ini"


def _config(path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding="UTF-8")
    return parser


def _physical_xbar_count(config):
    tile_x, tile_y = map(int, config.get("Architecture level", "Tile_Num").split(","))
    pe_x, pe_y = map(int, config.get("Tile level", "PE_Num").split(","))
    groups = config.getint("Process element level", "Group_Num")
    polarity = config.getint("Process element level", "Xbar_Polarity")
    return tile_x * tile_y * pe_x * pe_y * groups * polarity


def test_literature_profile_keeps_excel_rram_and_defines_1152_xbars():
    profile = _config(_PROFILE)
    reference = _config(_G8_REFERENCE)

    assert dict(profile.items("Device level")) == dict(reference.items("Device level"))
    for section, option in (
        ("Crossbar level", "Xbar_Size"),
        ("Crossbar level", "Subarray_Size"),
        ("Process element level", "Xbar_Polarity"),
        ("Process element level", "Group_Num"),
        ("Process element level", "DAC_Num"),
        ("Tile level", "PE_Num"),
    ):
        assert profile.get(section, option) == reference.get(section, option)
    assert profile.get("Architecture level", "Tile_Num") == "6,3"
    assert _physical_xbar_count(profile) == 1152
    assert profile.get("Algorithm Configuration", "OU_Size") == "4,4"
    assert profile.getboolean("Static Power", "Static_Power_Enable") is True
    assert (
        profile.getboolean("Architecture level", "Buffer_IncludeLeakageEnergy")
        is False
    )


def test_formal_simconfig_matches_the_literature_g8_profile_with_snn_overlay():
    formal = _config(_FORMAL_SIM_CONFIG)
    profile = _config(_PROFILE)

    assert [
        section
        for section in formal.sections()
        if section not in {"SNN Accuracy", "Neuron", "DDR"}
    ] == profile.sections()
    for section in profile.sections():
        formal_items = dict(formal.items(section))
        profile_items = dict(profile.items(section))
        if section == "Interface level":
            assert formal_items.pop("ADC_Quantization_Mode") == "READC_LLOYD_MAX"
            assert profile_items.pop("ADC_Quantization_Mode") == "AUTO"
        assert formal_items == profile_items
    assert dict(formal.items("DDR")) == {
        "DDR_Enable": "0",
        "DDR_Backend": "disabled",
        "DRAMSim3_Binary": "third_party/DRAMSim3/dramsim3main.out",
        "DRAMSim3_Config": "third_party/DRAMSim3/configs/DDR4_8Gb_x8_3200.ini",
        "DDR_TransactionBytes": "64",
        "DDR_DefaultBitwidth": "8",
        "DRAMSim3_CyclesPerTransaction": "128",
        "DRAMSim3_TimeoutSeconds": "120",
    }
    assert dict(formal.items("SNN Accuracy")) == {
        "SNN_Hardware_Profile": "adc_digital_lif",
        "Hardware_Current_Model": "physical_conductance",
        "LIF_Perturbation_Artifact": "",
        "LIF_Perturbation_Seed": "112",
    }
    assert dict(formal.items("Neuron")) == {
        "LIF_Mode": "digital",
        "Analog_LIF_LUT_Root": "08_lut",
        "Analog_LIF_LUT_Verify_Checksums": "1",
        "Analog_LIF_LUT_Profile": "fast_1h_experimental_7pct",
        "Analog_LIF_Negative_Vmem_Policy": "error",
    }


def test_literature_converter_models_use_explicit_energy_and_latency():
    adc = ADC(str(_PROFILE))
    dac = DAC(str(_PROFILE))
    adc.calculate_ADC_power()
    adc.calculate_ADC_latency()
    adc.calculate_ADC_energy()
    dac.calculate_DAC_power()
    dac.calculate_DAC_latency()
    dac.calculate_DAC_energy()

    assert adc.ADC_uses_explicit_conversion_model is True
    assert adc.ADC_precision == 5
    assert adc.ADC_latency == pytest.approx(0.165)
    assert adc.ADC_energy == pytest.approx(0.00001258)
    assert adc.ADC_power == pytest.approx(0.0000762424242424)
    assert _config(_PROFILE).getint("Process element level", "ADC_Num") == 32
    assert dac.DAC_uses_explicit_conversion_model is True
    assert dac.DAC_precision == 8
    assert dac.DAC_latency == pytest.approx(16.0)
    assert dac.DAC_energy == pytest.approx(0.000027)
    assert dac.DAC_power == pytest.approx(1.6875e-6)


def test_literature_buffer_model_scales_dynamic_energy_by_payload_bits():
    tested = buffer(str(_PROFILE), buf_level=1, default_buf_size=4)
    tested.calculate_buf_read_power()
    tested.calculate_buf_write_power()
    tested.calculate_buf_read_energy(16)
    tested.calculate_buf_write_energy(16)

    expected_leakage_mw = 4 * 1024 * 8 * 1.372e-14 * 1e3
    expected_dynamic_nj = 16 * 8 * 0.000001

    assert tested.custom_energy_model is False
    assert tested.buf_uses_explicit_access_energy_model is True
    assert tested.include_leakage_energy is False
    assert tested.buf_cycle == pytest.approx(1.84)
    assert tested.leakage_power == pytest.approx(expected_leakage_mw)
    assert tested.accounted_leakage_power == 0.0
    assert tested.dynamic_buf_rpower == pytest.approx(128 * 0.000001 / 1.84 * 1e3)
    assert tested.dynamic_buf_wpower == pytest.approx(128 * 0.000001 / 1.84 * 1e3)
    assert tested.buf_renergy == pytest.approx(expected_dynamic_nj)
    assert tested.buf_wenergy == pytest.approx(expected_dynamic_nj)


def test_literature_digital_models_use_per_bit_energy_and_operation_latency():
    add = adder(str(_PROFILE), bitwidth=8)
    merge = JointModule(str(_PROFILE), max_bitwidth=8)
    register = reg(str(_PROFILE), bitwidth=8)
    shifter = shiftreg(str(_PROFILE), max_shiftbase=16)

    add.calculate_adder_energy()
    merge.calculate_jointmodule_energy()
    register.calculate_reg_energy()
    shifter.calculate_shiftreg_energy()

    assert add.adder_uses_explicit_energy_model is True
    assert merge.jointmodule_uses_explicit_energy_model is True
    assert register.reg_uses_explicit_energy_model is True
    assert shifter.shiftreg_uses_explicit_energy_model is True
    assert add.adder_latency == pytest.approx(0.481)
    assert merge.jointmodule_latency == pytest.approx(0.481)
    assert register.reg_latency == pytest.approx(1.0)
    assert shifter.shiftreg_latency == pytest.approx(1.0)
    assert add.adder_energy == pytest.approx(8 * 3.73e-9)
    assert merge.jointmodule_energy == pytest.approx(8 * 3.73e-9)
    assert register.reg_energy == pytest.approx(8 * 1.53e-7)
    assert shifter.shiftreg_energy == pytest.approx(16 * 1.53e-7)
    assert math.isclose(add.adder_power, 8 * 3.73e-9 / 0.481, rel_tol=0.0, abs_tol=1e-20)
