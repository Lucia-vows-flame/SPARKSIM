import configparser
from pathlib import Path

import pytest

from MNSIM.Interface.final_rram_contract import (
    FINAL_RRAM_MODEL_IDS,
    FinalRramContractError,
    apply_final_rram_model,
    load_final_rram_model,
    load_snn_accuracy_device_contract,
    normalize_final_rram_model_id,
    require_final_rram_hardware_write_support,
    validate_final_rram_parser,
    validate_final_rram_simconfig,
)
from MNSIM.Scheduling.phase2.physical_write_ledger import load_physical_write_tables


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("model_id", FINAL_RRAM_MODEL_IDS)
def test_canonical_model_can_be_applied_and_validated(model_id):
    parser = configparser.ConfigParser()
    metadata = apply_final_rram_model(parser, model_id, repo_root=REPO_ROOT)

    assert validate_final_rram_parser(parser, repo_root=REPO_ROOT) == metadata
    assert parser.getint("Device level", "Device_Level") == metadata["levels"]
    assert parser.getfloat("Device level", "Read_Voltage") == metadata["read_voltage_v"]
    assert parser.getfloat("SNN RouteA", "V_read") == metadata["read_voltage_v"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("stanford_asu", "rram_stanford_asu_256"),
        ("stanford-asu", "rram_stanford_asu_256"),
        ("stanford_pku", "rram_memtorch_stanford_pku_256"),
        ("stanford-pku", "rram_memtorch_stanford_pku_256"),
        ("stanford_pku_write", "rram_memtorch_stanford_pku_write256"),
    ],
)
def test_only_asu_and_pku_aliases_are_accepted(value, expected):
    assert normalize_final_rram_model_id(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "20ns",
        "30ns",
        "rram_20ns_linear_fit",
        "rram_30ns_linear_fit",
        "milo",
        "data_driven2021",
        "sky130",
        "legacy_1bit",
        "/tmp/custom-device.json",
    ],
)
def test_removed_or_unapproved_device_models_are_rejected(value):
    with pytest.raises(FinalRramContractError, match="unsupported device model"):
        normalize_final_rram_model_id(value)


@pytest.mark.parametrize(
    ("section", "option", "value"),
    [
        ("Device level", "Device_Resistance", "1,2"),
        ("Device level", "Write_Latency", "10"),
        ("Device level", "Device_Variation", "5"),
        ("SNN RouteA", "V_read", "1"),
        ("Final RRAM Model", "Source_SHA256", "tampered"),
    ],
)
def test_tampering_with_any_locked_device_field_is_rejected(section, option, value):
    parser = configparser.ConfigParser()
    apply_final_rram_model(parser, "stanford_asu", repo_root=REPO_ROOT)
    parser.set(section, option, value)

    with pytest.raises(FinalRramContractError, match=option):
        validate_final_rram_parser(parser, repo_root=REPO_ROOT)


def test_root_config_is_locked_to_stanford_asu():
    metadata = validate_final_rram_simconfig(REPO_ROOT / "SimConfig.ini", repo_root=REPO_ROOT)
    assert metadata["model_id"] == "rram_stanford_asu_256"


def test_config_without_final_model_provenance_is_rejected():
    parser = configparser.ConfigParser()
    parser["Device level"] = {"Device_Level": "256"}
    with pytest.raises(FinalRramContractError, match="Final RRAM Model"):
        validate_final_rram_parser(parser, repo_root=REPO_ROOT)


def test_asu_transition_matrix_drives_phase2_write_tables():
    parser = configparser.ConfigParser()
    metadata = apply_final_rram_model(parser, "stanford_asu", repo_root=REPO_ROOT)
    assert metadata["hardware_write_accounting_supported"] is True
    require_final_rram_hardware_write_support(parser, source="test", repo_root=REPO_ROOT)

    config = REPO_ROOT / "device_models/final_rram/configs/SimConfig_rram_stanford_asu_256.ini"
    tables = load_physical_write_tables(str(config))
    assert tables.latency_ns[0, 1] == pytest.approx(409.51)
    assert tables.latency_ns[1, 0] == pytest.approx(100.0)
    assert tables.energy_nj[0, 1] > 0.0
    assert tables.energy_nj[1, 0] > 0.0


def test_pku_write_model_has_256_state_transition_costs():
    model_id = "rram_memtorch_stanford_pku_write256"
    parser = configparser.ConfigParser()
    metadata = apply_final_rram_model(parser, model_id, repo_root=REPO_ROOT)
    assert metadata["levels"] == 256
    assert metadata["hardware_write_accounting_supported"] is True
    require_final_rram_hardware_write_support(parser, source="test", repo_root=REPO_ROOT)
    transition = load_final_rram_model(model_id, repo_root=REPO_ROOT)["transition_cost"]
    assert len(transition["latency_s"]) == 256 * 256
    assert len(transition["energy_j"]) == 256 * 256


def test_loaded_model_mutation_cannot_poison_cached_canonical_content():
    model = load_final_rram_model("stanford_asu", repo_root=REPO_ROOT)
    canonical_read_voltage = model["device"]["read_voltage_v"]
    model["device"]["read_voltage_v"] = 1.25
    reloaded = load_final_rram_model("stanford_asu", repo_root=REPO_ROOT)
    parser = configparser.ConfigParser()
    apply_final_rram_model(parser, "stanford_asu", repo_root=REPO_ROOT)
    assert reloaded["device"]["read_voltage_v"] == canonical_read_voltage
    assert parser.getfloat("Device level", "Read_Voltage") == float(canonical_read_voltage)


@pytest.mark.parametrize("model_id", FINAL_RRAM_MODEL_IDS)
def test_models_expose_fixed_snn_accuracy_device_contract(model_id):
    parser = configparser.ConfigParser()
    apply_final_rram_model(parser, model_id, repo_root=REPO_ROOT)
    contract = load_snn_accuracy_device_contract(parser, source="test", repo_root=REPO_ROOT)
    assert contract.model_id == model_id
    assert contract.device_seed == 112
    assert contract.enable_r_ratio is True
    assert contract.enable_variation is True
    assert contract.enable_saf is False
    assert contract.saf_percent == (0.0, 0.0)


def test_snn_accuracy_device_contract_rejects_tampered_controls():
    parser = configparser.ConfigParser()
    apply_final_rram_model(parser, "stanford_asu", repo_root=REPO_ROOT)
    parser.set("SNN RouteA", "Enable_SAF", "1")
    with pytest.raises(FinalRramContractError, match="Enable_SAF"):
        load_snn_accuracy_device_contract(parser, source="tampered", repo_root=REPO_ROOT)
