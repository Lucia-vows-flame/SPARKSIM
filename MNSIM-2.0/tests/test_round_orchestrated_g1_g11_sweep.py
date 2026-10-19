import importlib.util
from pathlib import Path

import pytest

from MNSIM.Interface.final_rram_contract import FinalRramContractError


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_round_orchestrated_g1_g11_sweep.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("round_g1_g11_sweep", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fixed_ladder_and_physical_xbar_accounting(tmp_path):
    module = _load_module()
    base_config = Path(__file__).resolve().parents[1] / "SimConfig.ini"

    assert module.RESOURCE_LADDER[0] == ("G1", (2, 3))
    assert module.RESOURCE_LADDER[7] == ("G8", (6, 6))
    assert module.RESOURCE_LADDER[-1] == ("G11", (9, 10))

    g1_config = tmp_path / "G1" / "SimConfig.ini"
    physical_xbars, model_metadata = module._write_group_config(
        base_config=base_config,
        output_path=g1_config,
        tile_num=(2, 3),
        repo_root=Path(__file__).resolve().parents[1],
    )
    generated = module._load_config(g1_config)

    assert generated.get("Architecture level", "Tile_Num") == "2,3"
    assert generated.get("Tile level", "PE_Num") == "4,4"
    assert generated.getint("Process element level", "Group_Num") == 2
    assert generated.getint("Process element level", "Xbar_Polarity") == 2
    assert generated.getboolean("Static Power", "Static_Power_Enable") is True
    assert physical_xbars == 384
    assert model_metadata["model_id"] == "rram_stanford_asu_256"


def test_final_rram_model_is_applied_without_changing_resource_projection(tmp_path):
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]
    base_config = repo_root / "SimConfig.ini"
    generated_path = tmp_path / "G1" / "SimConfig.ini"

    physical_xbars, metadata = module._write_group_config(
        base_config=base_config,
        output_path=generated_path,
        tile_num=(2, 3),
        device_model_id="rram_stanford_asu_256",
        repo_root=repo_root,
    )
    generated = module._load_config(generated_path)

    assert physical_xbars == 384
    assert metadata["model_id"] == "rram_stanford_asu_256"
    assert generated.getint("Device level", "Device_Level") == 256
    assert generated.getfloat("Device level", "Write_Latency") == 0.0
    assert generated.get("Device level", "Device_SAF") == "0,0"
    assert len(generated.get("Device level", "Device_Resistance").split(",")) == 256
    assert generated.get("Final RRAM Model", "Model_ID") == "rram_stanford_asu_256"
    assert generated.get("SNN RouteA", "Device_Seed") == "112"


def test_asu_and_pku_models_resolve_to_bundled_artifacts():
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]

    assert module._resolve_device_model(repo_root, "asu") == "rram_stanford_asu_256"
    assert module._resolve_device_model(repo_root, "pku") == "rram_memtorch_stanford_pku_256"


def test_nominal_only_pku_is_rejected_by_hardware_sweep(tmp_path):
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]

    with pytest.raises(
        FinalRramContractError,
        match="does not provide a complete source-to-target write transition model",
    ):
        module._write_group_config(
            base_config=repo_root / "SimConfig.ini",
            output_path=tmp_path / "G1" / "SimConfig.ini",
            tile_num=(2, 3),
            device_model_id="rram_memtorch_stanford_pku_256",
            repo_root=repo_root,
        )


@pytest.mark.parametrize("value", ["legacy_1bit", "measured_8bit", "/tmp/custom.json"])
def test_nonfinal_device_models_are_rejected(value):
    module = _load_module()

    with pytest.raises(FinalRramContractError, match="unsupported device model"):
        module._resolve_device_model(Path(__file__).resolve().parents[1], value)
