import configparser
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


def test_wrapper_is_executable_and_launches_cli_via_runpy_from_any_cwd(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = repo_root / "scripts" / "run_round_orchestrated_sim.sh"
    recorder = tmp_path / "record_python_argv.py"
    record_path = tmp_path / "argv.json"
    other_cwd = tmp_path / "other-cwd"
    other_cwd.mkdir()
    recorder.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ROUND_ORCHESTRATED_ARGV_RECORD']).write_text(\n"
        "    json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    recorder.chmod(recorder.stat().st_mode | stat.S_IXUSR)

    completed = subprocess.run(
        ["bash", str(wrapper), "--network", "vgg8", "--show-rounds"],
        cwd=other_cwd,
        env={
            **os.environ,
            "MNSIM_PYTHON": str(recorder),
            "MNSIM_CUDA_PREFLIGHT": "0",
            "ROUND_ORCHESTRATED_ARGV_RECORD": str(record_path),
        },
        capture_output=True,
        text=True,
        check=False,
    )

    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    recorded_argv = recorded["argv"]
    assert completed.returncode == 0
    assert os.access(wrapper, os.X_OK)
    assert recorded["cwd"] == str(repo_root)
    assert recorded_argv[0] == "-c"
    assert "runpy.run_path(\"MNSIM/Interface/round_orchestrated_cli.py\"" in recorded_argv[1]
    assert recorded_argv == [
        "-c",
        recorded_argv[1],
        "--network",
        "vgg8",
        "--show-rounds",
    ]


def _write_python_argv_recorder(tmp_path):
    recorder = tmp_path / "record_python_argv.py"
    record_path = tmp_path / "argv.json"
    recorder.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['ROUND_ORCHESTRATED_ARGV_RECORD']).write_text(\n"
        "    json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}), encoding='utf-8'\n"
        ")\n",
        encoding="utf-8",
    )
    recorder.chmod(recorder.stat().st_mode | stat.S_IXUSR)
    return recorder, record_path


def _write_four_level_custom_sim_config(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(repo_root / "SimConfig.ini", encoding="UTF-8")
    parser.set("Device level", "Device_Level", "4")
    parser.set(
        "Device level",
        "Device_Resistance",
        "1000000,100000,10000,1000",
    )
    path = tmp_path / "four-level-custom.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return path


def test_vgg8_g8_launcher_writes_g8_config_and_forwards_hardware_options(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_vgg8_g8.sh"
    assert launcher.exists()

    checkpoint = tmp_path / "vgg8.pth"
    checkpoint.touch()
    recorder, record_path = _write_python_argv_recorder(tmp_path)
    output_dir = tmp_path / "output"
    env = {
        **os.environ,
        "MNSIM_PYTHON": str(recorder),
        "MNSIM_CUDA_PREFLIGHT": "0",
        "ROUND_ORCHESTRATED_ARGV_RECORD": str(record_path),
        "VGG8_G8_CHECKPOINT": str(checkpoint),
        "VGG8_G8_OUTPUT_DIR": str(output_dir),
    }
    env.pop("VGG8_G8_BASE_SIM_CONFIG", None)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Device model: rram_stanford_asu_256" in completed.stdout
    assert "2304 physical xbar arrays" in completed.stdout
    assert "1152 MNSIM group slots" in completed.stdout
    config_path = output_dir / "SimConfig.g8.ini"
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="UTF-8")
    assert parser.get("Crossbar level", "Xbar_Size") == "128,128"
    assert parser.get("Crossbar level", "Subarray_Size") == "128"
    assert parser.get("Process element level", "Group_Num") == "2"
    assert parser.get("Tile level", "PE_Num") == "4,4"
    assert parser.get("Architecture level", "Tile_Num") == "6,6"
    assert not parser.has_option("Algorithm Configuration", "Phase1_Device_Level")

    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    assert recorded["cwd"] == str(repo_root)
    assert recorded["argv"][2:] == [
        "--network",
        "vgg8",
        "--weights",
        str(checkpoint),
        "--num-classes",
        "10",
        "--sim-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-hardware",
        "--device",
        "0",
        "--skip-accuracy",
        "--show-rounds",
        "--overwrite",
    ]


def test_vgg8_g8_launcher_copies_custom_sim_config_and_patches_geometry(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_vgg8_g8.sh"
    custom_config = _write_four_level_custom_sim_config(tmp_path)
    checkpoint = tmp_path / "vgg8.pth"
    checkpoint.touch()
    recorder, record_path = _write_python_argv_recorder(tmp_path)
    output_dir = tmp_path / "output"
    env = {
        **os.environ,
        "MNSIM_PYTHON": str(recorder),
        "MNSIM_CUDA_PREFLIGHT": "0",
        "ROUND_ORCHESTRATED_ARGV_RECORD": str(record_path),
        "VGG8_G8_CHECKPOINT": str(checkpoint),
        "VGG8_G8_OUTPUT_DIR": str(output_dir),
        "VGG8_G8_BASE_SIM_CONFIG": custom_config.name,
    }
    env.pop("VGG8_G8_DEVICE_PROFILE", None)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Device model: rram_stanford_asu_256" in completed.stdout
    config_path = output_dir / "SimConfig.g8.ini"
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="UTF-8")
    assert parser.getint("Device level", "Device_Level") == 256
    assert len(parser.get("Device level", "Device_Resistance").split(",")) == 256
    assert parser.get("Final RRAM Model", "Model_ID") == "rram_stanford_asu_256"
    assert parser.get("Crossbar level", "Xbar_Size") == "128,128"
    assert parser.get("Process element level", "Group_Num") == "2"
    assert parser.get("Tile level", "PE_Num") == "4,4"
    assert parser.get("Architecture level", "Tile_Num") == "6,6"

    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    assert recorded["cwd"] == str(repo_root)
    assert recorded["argv"][recorded["argv"].index("--sim-config") + 1] == str(config_path)


@pytest.mark.parametrize(
    "device_profile",
    ["legacy_1bit", "measured_8bit", "", "/tmp/custom-device.json"],
    ids=["legacy-profile", "measured-profile", "empty-profile", "json-path"],
)
def test_vgg8_g8_launcher_rejects_nonfinal_device_profile(
    tmp_path,
    device_profile,
):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_vgg8_g8.sh"
    custom_config = _write_four_level_custom_sim_config(tmp_path)
    checkpoint = tmp_path / "vgg8.pth"
    checkpoint.touch()
    recorder, record_path = _write_python_argv_recorder(tmp_path)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env={
            **os.environ,
            "MNSIM_PYTHON": str(recorder),
            "MNSIM_CUDA_PREFLIGHT": "0",
            "ROUND_ORCHESTRATED_ARGV_RECORD": str(record_path),
            "VGG8_G8_CHECKPOINT": str(checkpoint),
            "VGG8_G8_OUTPUT_DIR": str(tmp_path / "output"),
            "VGG8_G8_BASE_SIM_CONFIG": str(custom_config),
            "VGG8_G8_DEVICE_PROFILE": device_profile,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "must select Stanford/ASU or Stanford-PKU" in completed.stderr
    assert not record_path.exists()


def test_vgg8_g8_launcher_rejects_removed_device_profiles(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_vgg8_g8.sh"
    checkpoint = tmp_path / "vgg8.pth"
    checkpoint.touch()
    recorder, record_path = _write_python_argv_recorder(tmp_path)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env={
            **os.environ,
            "MNSIM_PYTHON": str(recorder),
            "MNSIM_CUDA_PREFLIGHT": "0",
            "ROUND_ORCHESTRATED_ARGV_RECORD": str(record_path),
            "VGG8_G8_CHECKPOINT": str(checkpoint),
            "VGG8_G8_OUTPUT_DIR": str(tmp_path / "output"),
            "VGG8_G8_DEVICE_PROFILE": "20ns",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "must select Stanford/ASU or Stanford-PKU" in completed.stderr
    assert not record_path.exists()


def test_vgg8_g8_launcher_uses_the_locked_asu_device_by_default(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_vgg8_g8.sh"
    checkpoint = tmp_path / "vgg8.pth"
    checkpoint.touch()
    recorder, _ = _write_python_argv_recorder(tmp_path)
    output_dir = tmp_path / "output"
    env = {
        **os.environ,
        "MNSIM_PYTHON": str(recorder),
        "MNSIM_CUDA_PREFLIGHT": "0",
        "ROUND_ORCHESTRATED_ARGV_RECORD": str(tmp_path / "argv.json"),
        "VGG8_G8_CHECKPOINT": str(checkpoint),
        "VGG8_G8_OUTPUT_DIR": str(output_dir),
    }
    env.pop("VGG8_G8_BASE_SIM_CONFIG", None)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    parser = configparser.ConfigParser()
    parser.read(output_dir / "SimConfig.g8.ini", encoding="UTF-8")
    resistance = [
        float(value.strip())
        for value in parser.get("Device level", "Device_Resistance").split(",")
    ]

    assert parser.getint("Device level", "Device_Level") == 256
    assert parser.getint("Device level", "Device_Resistance_Sort") == 0
    assert parser.getfloat("Device level", "Read_Voltage") == 0.1
    assert parser.getfloat("Device level", "Write_Latency") == 0.0
    assert len(resistance) == 256
    assert resistance[0] > resistance[-1] > 0.0
    assert parser.get("SNN RouteA", "V_read") == "0.1"
    assert parser.get("Final RRAM Model", "Model_ID") == "rram_stanford_asu_256"
    assert not parser.has_option("Algorithm Configuration", "Phase1_Device_Level")


def test_vgg8_g8_launcher_selects_the_locked_pku_write_device_model(
    tmp_path,
    monkeypatch,
):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_vgg8_g8.sh"
    inherited_custom_config = _write_four_level_custom_sim_config(tmp_path)
    monkeypatch.setenv("VGG8_G8_BASE_SIM_CONFIG", str(inherited_custom_config))
    checkpoint = tmp_path / "vgg8.pth"
    checkpoint.touch()
    recorder, _ = _write_python_argv_recorder(tmp_path)
    output_dir = tmp_path / "output"
    env = {
        **os.environ,
        "MNSIM_PYTHON": str(recorder),
        "MNSIM_CUDA_PREFLIGHT": "0",
        "ROUND_ORCHESTRATED_ARGV_RECORD": str(tmp_path / "argv.json"),
        "VGG8_G8_CHECKPOINT": str(checkpoint),
        "VGG8_G8_OUTPUT_DIR": str(output_dir),
        "VGG8_G8_DEVICE_PROFILE": "pku",
    }
    env.pop("VGG8_G8_BASE_SIM_CONFIG", None)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    parser = configparser.ConfigParser()
    parser.read(output_dir / "SimConfig.g8.ini", encoding="UTF-8")

    assert "Device model: rram_memtorch_stanford_pku_write256" in completed.stdout
    assert parser.getint("Device level", "Device_Level") == 256
    assert parser.getint("Device level", "Device_Resistance_Sort") == 0
    assert parser.get("Device level", "Read_Voltage") == "0.1"
    assert parser.getfloat("Device level", "Write_Latency") == 0.0
    assert parser.get("Final RRAM Model", "Model_ID") == "rram_memtorch_stanford_pku_write256"
    assert not parser.has_option("Algorithm Configuration", "Phase1_Device_Level")


def test_vgg8_g8_launcher_enables_accuracy_with_the_configured_dataset(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    launcher = repo_root / "scripts" / "run_vgg8_g8.sh"
    assert launcher.exists()

    checkpoint = tmp_path / "vgg8.pth"
    checkpoint.touch()
    recorder, record_path = _write_python_argv_recorder(tmp_path)
    env = {
        **os.environ,
        "MNSIM_PYTHON": str(recorder),
        "MNSIM_CUDA_PREFLIGHT": "0",
        "ROUND_ORCHESTRATED_ARGV_RECORD": str(record_path),
        "VGG8_G8_CHECKPOINT": str(checkpoint),
        "VGG8_G8_OUTPUT_DIR": str(tmp_path / "output"),
        "VGG8_G8_RUN_ACCURACY": "1",
        "VGG8_G8_DATASET_MODULE": "MNSIM.Interface.cifar10",
        "VGG8_G8_SHOW_ROUNDS": "0",
        "VGG8_G8_OVERWRITE": "0",
    }
    env.pop("VGG8_G8_BASE_SIM_CONFIG", None)

    completed = subprocess.run(
        ["bash", str(launcher)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    recorded = json.loads(record_path.read_text(encoding="utf-8"))
    assert "--dataset-module" in recorded["argv"]
    assert "MNSIM.Interface.cifar10" in recorded["argv"]
    assert "--skip-accuracy" not in recorded["argv"]
    assert "--show-rounds" not in recorded["argv"]
    assert "--overwrite" not in recorded["argv"]
