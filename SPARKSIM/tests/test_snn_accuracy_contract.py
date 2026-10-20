import configparser
import hashlib

import numpy as np
import pytest
import torch

from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    ADC_DIGITAL_LIF,
    BINARY_SPIKE_DOMAIN,
    FRONTEND_LUT_LIF_LUT,
    SUPPORTED_SNN_HARDWARE_PROFILES,
    BinarySpikeDomain,
    SNNAccuracyContractError,
    bit_weights_sha256,
    canonical_json_sha256,
    file_sha256,
    parse_snn_accuracy_config,
)


def _parser(profile: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser["SNN Accuracy"] = {
        "SNN_Hardware_Profile": profile,
        "Hardware_Current_Model": "physical_conductance",
        "LIF_Perturbation_Artifact": "",
        "LIF_Perturbation_Seed": "112",
    }
    parser["Interface level"] = {
        "ADC_Profile": "HONG_2025_READC_5BIT",
        "ADC_Quantization_Mode": "READC_LLOYD_MAX",
        "ADC_Precision": "5",
        "ADC_ReADC_Percentile_Low": "0.5",
        "ADC_ReADC_Percentile_High": "99.5",
        "ADC_ReADC_Max_Samples_Per_Layer": "8000",
    }
    return parser


def test_binary_spike_domain_has_exact_metadata_and_preserves_valid_tensor():
    assert SUPPORTED_SNN_HARDWARE_PROFILES == frozenset(
        {ADC_DIGITAL_LIF, FRONTEND_LUT_LIF_LUT}
    )
    assert BINARY_SPIKE_DOMAIN.values == (0, 1)
    assert BINARY_SPIKE_DOMAIN.scale == 1.0
    assert BINARY_SPIKE_DOMAIN.signed is False
    assert BINARY_SPIKE_DOMAIN.input_digit_max == 1
    assert BINARY_SPIKE_DOMAIN.activation_cycle == 1

    value = torch.tensor([0.0, 1.0])
    assert BINARY_SPIKE_DOMAIN.validate(value) is value


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([0.0, 0.49, 1.0]),
        torch.tensor([-1.0, 0.0, 1.0]),
        torch.tensor([0.0, 2.0]),
    ],
)
def test_binary_spike_domain_rejects_nonbinary_values(value):
    with pytest.raises(ValueError, match="binary spike"):
        BINARY_SPIKE_DOMAIN.validate(value)


@pytest.mark.parametrize(
    "value",
    [torch.tensor([float("nan")]), torch.tensor([float("inf")])],
)
def test_binary_spike_domain_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="non-finite"):
        BINARY_SPIKE_DOMAIN.validate(value)


def test_binary_spike_domain_rejects_non_tensors():
    with pytest.raises(TypeError, match="torch.Tensor"):
        BINARY_SPIKE_DOMAIN.validate([0, 1])


@pytest.mark.parametrize(
    "override",
    [
        {"values": (0, 2)},
        {"scale": 0.5},
        {"signed": True},
        {"input_digit_max": 2},
        {"activation_cycle": 2},
    ],
)
def test_binary_spike_domain_rejects_noncanonical_construction(override):
    with pytest.raises(ValueError, match="formal binary spike domain"):
        BinarySpikeDomain(**override)


def test_adc_profile_requires_the_frozen_readc_configuration():
    config = parse_snn_accuracy_config(_parser(ADC_DIGITAL_LIF), source="test.ini")

    assert config.hardware_profile == ADC_DIGITAL_LIF
    assert config.hardware_current_model == "physical_conductance"
    assert config.adc_model == "READC_LLOYD_MAX"
    assert config.adc_precision == 5
    assert config.readc_percentile_low == 0.5
    assert config.readc_percentile_high == 99.5
    assert config.readc_max_samples_per_layer == 8000
    assert config.lif_perturbation_artifact is None
    assert config.lif_perturbation_seed == 112


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("ADC_Profile", "LEGACY"),
        ("ADC_Quantization_Mode", "UNIFORM"),
        ("ADC_Precision", "6"),
        ("ADC_ReADC_Percentile_Low", "1.0"),
        ("ADC_ReADC_Percentile_High", "99.0"),
        ("ADC_ReADC_Max_Samples_Per_Layer", "7999"),
    ],
)
def test_adc_profile_rejects_any_readc_configuration_mismatch(option, value):
    parser = _parser(ADC_DIGITAL_LIF)
    parser.set("Interface level", option, value)

    with pytest.raises(SNNAccuracyContractError, match=option):
        parse_snn_accuracy_config(parser, source="test.ini")


def test_lut_profile_requires_an_artifact_path():
    with pytest.raises(SNNAccuracyContractError, match="LIF_Perturbation_Artifact"):
        parse_snn_accuracy_config(_parser(FRONTEND_LUT_LIF_LUT), source="test.ini")


def test_lut_profile_accepts_explicit_no_perturbation_baseline():
    parser = _parser(FRONTEND_LUT_LIF_LUT)
    parser.set("SNN Accuracy", "LIF_Perturbation_Mode", "none")

    config = parse_snn_accuracy_config(parser, source="test.ini")

    assert config.lif_perturbation_mode == "none"
    assert config.lif_perturbation_artifact is None


def test_lut_profile_resolves_relative_artifact_against_source_parent(tmp_path):
    parser = _parser(FRONTEND_LUT_LIF_LUT)
    parser.set(
        "SNN Accuracy", "LIF_Perturbation_Artifact", "characterization/lif.json"
    )

    config = parse_snn_accuracy_config(parser, source=tmp_path / "configs" / "test.ini")

    assert config.hardware_profile == FRONTEND_LUT_LIF_LUT
    assert config.adc_model is None
    assert config.adc_precision is None
    assert config.readc_percentile_low is None
    assert config.readc_percentile_high is None
    assert config.readc_max_samples_per_layer is None
    assert config.lif_perturbation_artifact == (
        tmp_path / "configs" / "characterization" / "lif.json"
    ).resolve()


def test_unknown_snn_profile_fails_closed():
    with pytest.raises(SNNAccuracyContractError, match="unsupported"):
        parse_snn_accuracy_config(_parser("mixed_adc_lut"), source="test.ini")


def test_adc_profile_rejects_a_lut_perturbation_artifact():
    parser = _parser(ADC_DIGITAL_LIF)
    parser.set("SNN Accuracy", "LIF_Perturbation_Artifact", "lif.json")

    with pytest.raises(SNNAccuracyContractError, match="must not configure"):
        parse_snn_accuracy_config(parser, source="test.ini")


def test_hardware_current_model_must_be_physical_conductance():
    parser = _parser(ADC_DIGITAL_LIF)
    parser.set("SNN Accuracy", "Hardware_Current_Model", "ideal_mac")

    with pytest.raises(SNNAccuracyContractError, match="physical_conductance"):
        parse_snn_accuracy_config(parser, source="test.ini")


def test_missing_or_invalid_required_configuration_fails_with_contract_error():
    parser = _parser(ADC_DIGITAL_LIF)
    parser.remove_option("SNN Accuracy", "SNN_Hardware_Profile")
    with pytest.raises(SNNAccuracyContractError, match="SNN_Hardware_Profile"):
        parse_snn_accuracy_config(parser, source="test.ini")

    parser = _parser(ADC_DIGITAL_LIF)
    parser.set("SNN Accuracy", "LIF_Perturbation_Seed", "not-an-int")
    with pytest.raises(SNNAccuracyContractError, match="LIF_Perturbation_Seed"):
        parse_snn_accuracy_config(parser, source="test.ini")


def test_canonical_json_hash_is_deterministic_and_key_order_independent():
    left = canonical_json_sha256({"b": [2, 3], "a": 1})
    right = canonical_json_sha256({"a": 1, "b": [2, 3]})

    assert left == right
    assert left == hashlib.sha256(b'{"a":1,"b":[2,3]}').hexdigest()
    assert left != canonical_json_sha256({"a": 1, "b": [2, 4]})
    with pytest.raises(ValueError):
        canonical_json_sha256({"value": float("nan")})


def test_file_hash_streams_exact_file_content(tmp_path):
    path = tmp_path / "artifact.bin"
    payload = (b"formal-snn-accuracy\0" * 70000) + b"tail"
    path.write_bytes(payload)

    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()


def test_bit_weights_hash_is_deterministic_and_sensitive_to_structure():
    weights = [
        {
            "weight": np.array([[1, 2]], dtype=np.int8),
            "bias": np.array([3.0], dtype=np.float32),
        },
        None,
    ]
    reordered = [{"bias": weights[0]["bias"], "weight": weights[0]["weight"]}, None]

    digest = bit_weights_sha256(weights)
    assert digest == bit_weights_sha256(reordered)
    assert digest != bit_weights_sha256([weights[0]])
    assert digest != bit_weights_sha256([None, weights[0]])
    assert digest != bit_weights_sha256(
        [
            {
                "weight": np.array([[1, 3]], dtype=np.int8),
                "bias": weights[0]["bias"],
            },
            None,
        ]
    )
    assert digest != bit_weights_sha256(
        [
            {
                "weight": weights[0]["weight"].astype(np.int16),
                "bias": weights[0]["bias"],
            },
            None,
        ]
    )
    assert digest != bit_weights_sha256(
        [
            {
                "weight": weights[0]["weight"].reshape(2, 1),
                "bias": weights[0]["bias"],
            },
            None,
        ]
    )
