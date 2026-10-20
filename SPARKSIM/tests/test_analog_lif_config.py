import pytest

from MNSIM.Scheduling.phase2.lif_mode import (
    AnalogLifConfig,
    LifModeConfigError,
    load_lif_mode_config,
)


def _write_config(path, body=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_missing_neuron_section_defaults_to_digital_mode(tmp_path):
    config = load_lif_mode_config(_write_config(tmp_path / "SimConfig.ini", "[Device]\nfoo=1\n"))

    assert config == AnalogLifConfig(
        mode="digital",
        lut_root=(tmp_path / "08_lut").resolve(),
        verify_checksums=True,
        profile="fast_1h_experimental_7pct",
        negative_vmem_policy="error",
    )


def test_explicit_analog_lut_configuration_is_normalized_and_resolved(tmp_path):
    config_path = _write_config(
        tmp_path / "nested" / "SimConfig.ini",
        """[Neuron]
LIF_Mode = ANALOG_LUT
Analog_LIF_LUT_Root = artifacts/08_lut
Analog_LIF_LUT_Verify_Checksums = 0
Analog_LIF_LUT_Profile = fast_1h_experimental_7pct
Analog_LIF_Negative_Vmem_Policy = CLAMP_ZERO
""",
    )

    config = load_lif_mode_config(config_path)

    assert config.mode == "analog_lut"
    assert config.lut_root == (config_path.parent / "artifacts" / "08_lut").resolve()
    assert config.verify_checksums is False
    assert config.profile == "fast_1h_experimental_7pct"
    assert config.negative_vmem_policy == "clamp_zero"


def test_default_analog_lut_profile_uses_qualified_seven_percent_bundle(tmp_path):
    config = load_lif_mode_config(_write_config(tmp_path / "SimConfig.ini"))

    assert config.profile == "fast_1h_experimental_7pct"


def test_invalid_mode_fails_closed(tmp_path):
    config_path = _write_config(
        tmp_path / "SimConfig.ini",
        "[Neuron]\nLIF_Mode = analytic\n",
    )

    with pytest.raises(LifModeConfigError, match="LIF_Mode"):
        load_lif_mode_config(config_path)


def test_invalid_negative_vmem_policy_fails_closed(tmp_path):
    config_path = _write_config(
        tmp_path / "SimConfig.ini",
        "[Neuron]\nAnalog_LIF_Negative_Vmem_Policy = wrap\n",
    )

    with pytest.raises(LifModeConfigError, match="Analog_LIF_Negative_Vmem_Policy"):
        load_lif_mode_config(config_path)


def test_invalid_checksum_flag_fails_closed(tmp_path):
    config_path = _write_config(
        tmp_path / "SimConfig.ini",
        "[Neuron]\nAnalog_LIF_LUT_Verify_Checksums = yes\n",
    )

    with pytest.raises(LifModeConfigError, match="Analog_LIF_LUT_Verify_Checksums"):
        load_lif_mode_config(config_path)


def test_empty_required_values_fail_closed(tmp_path):
    config_path = _write_config(
        tmp_path / "SimConfig.ini",
        """[Neuron]
Analog_LIF_LUT_Root =
Analog_LIF_LUT_Profile =
""",
    )

    with pytest.raises(LifModeConfigError):
        load_lif_mode_config(config_path)


def test_legacy_profile_alias_is_not_accepted(tmp_path):
    config_path = _write_config(
        tmp_path / "SimConfig.ini",
        """[Neuron]
Analog_LIF_Profile = fast_1h_experimental_7pct
""",
    )

    with pytest.raises(LifModeConfigError, match="unsupported"):
        load_lif_mode_config(config_path)
