"""Source-backed nominal tables, reproducibility, and runtime boundary guards."""

import configparser
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from MNSIM.Hardware_Model.Conductance import ConductanceTable
from MNSIM.Interface.final_rram_contract import (
    FinalRramContractError,
    apply_final_rram_model,
    load_final_rram_model,
    require_final_rram_hardware_write_support,
    validate_final_rram_simconfig,
)
from MNSIM.Scheduling.weight_scheduling.device import device_model_from_dict


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "device_models/final_rram/models"
SOURCES = ROOT / "device_models/final_rram/source/open_models"
spec = importlib.util.spec_from_file_location("open_rram_builder", ROOT / "scripts/build_open_rram_256_models.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
MODEL_IDS = [item[0] for item in builder.MODEL_SPECS]


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_nominal_tables_survive_runtime_loading_and_csv_roundtrip(model_id, tmp_path):
    model = load_final_rram_model(model_id, repo_root=ROOT)
    g = np.asarray(model["levels"]["nominal_conductance_s"])
    r = np.asarray(model["levels"]["nominal_resistance_ohm"])
    assert model["device"]["bits"] == 8
    assert model["levels"]["code"] == list(range(256))
    assert len(np.unique(g)) == 256
    assert np.all(np.isfinite(g)) and np.all(g > 0) and np.all(np.diff(g) > 0)
    np.testing.assert_allclose(g * r, 1, rtol=1e-14)
    np.testing.assert_allclose(g, np.linspace(g[0], g[-1], 256), rtol=1e-13)
    np.testing.assert_allclose(model["levels"]["read_current_a"], 0.1 * g, rtol=1e-14)
    with (MODELS / model["source"]["levels_file"]).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_array_equal([float(row["nominal_conductance_s"]) for row in rows], g)
    assert len(rows) == 256

    cfg = configparser.ConfigParser()
    apply_final_rram_model(cfg, model_id, repo_root=ROOT)
    path = tmp_path / "SimConfig.ini"
    with path.open("w") as stream:
        cfg.write(stream)
    table = ConductanceTable.from_simconfig(str(path))
    assert table.levels == 256
    torch.testing.assert_close(table.map_codes(torch.arange(256), dtype=torch.float64), torch.from_numpy(g), rtol=1e-11, atol=0)
    assert validate_final_rram_simconfig(path, repo_root=ROOT)["model_id"] == model_id


@pytest.mark.parametrize("model_id", MODEL_IDS)
def test_write_accounting_is_unavailable_until_programming_is_validated(model_id):
    model = load_final_rram_model(model_id, repo_root=ROOT)
    cfg = configparser.ConfigParser()
    apply_final_rram_model(cfg, model_id, repo_root=ROOT)
    assert cfg.getfloat("Device level", "Write_Latency") == 0
    with pytest.raises(FinalRramContractError, match="write transition model"):
        require_final_rram_hardware_write_support(cfg, repo_root=ROOT)
    transition = device_model_from_dict(model).cell_transition_cost(0, 255)
    assert transition.energy_j is None and transition.latency_s is None
    assert transition.exact is False
    assert not model["conversion"]["external_hspice_energy_read"]
    assert not model["conversion"]["external_spice_execution"]


def test_pku_source_endpoints_and_nonlinear_readback_are_preserved():
    stanford = load_final_rram_model(MODEL_IDS[0], repo_root=ROOT)
    np.testing.assert_allclose(stanford["levels"]["nominal_conductance_s"][::255], [1 / 218586, 1 / 542], rtol=1e-14)
    assert not math.isclose(stanford["levels"]["nominal_resistance_ohm"][0], 218586)


def test_sources_and_artifacts_regenerate_without_network_or_spice(tmp_path):
    output = tmp_path / "models"
    report = builder.build(ROOT, output)
    for model_id in MODEL_IDS:
        for suffix in [".json", "_levels.csv"]:
            assert (output / (model_id + suffix)).read_bytes() == (MODELS / (model_id + suffix)).read_bytes()
        assert report["models"][model_id]["model_sha256"] == hashlib.sha256((MODELS / (model_id + ".json")).read_bytes()).hexdigest()
    assert not report["external_spice_execution"]
    assert not report["external_hspice_energy_read"]
    assert json.loads((MODELS / "open_rram_256_validation.json").read_text()) == report


def test_modified_pinned_source_is_rejected(tmp_path):
    clone = tmp_path / "sources"
    shutil.copytree(SOURCES, clone)
    path = clone / "Stanford_PKU.py"
    path.write_text(path.read_text().replace("r_on=542", "r_on=999"))
    with pytest.raises(ValueError, match="pinned source checksum mismatch"):
        builder.verify_sources(clone)
