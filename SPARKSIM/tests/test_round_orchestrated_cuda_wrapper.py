import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_round_orchestrated_sim.sh"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def test_explicit_cuda_wrapper_runs_nvidia_smi_before_python(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    probe_marker = tmp_path / "nvidia-smi.called"
    python_marker = tmp_path / "python.called"

    _write_executable(
        bin_dir / "nvidia-smi",
        f"#!/usr/bin/env bash\nprintf '0\\n' > {probe_marker}\n",
    )
    _write_executable(
        bin_dir / "fake-python",
        f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > {python_marker}\n",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MNSIM_PYTHON"] = str(bin_dir / "fake-python")

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--device", "cuda:0"],
        cwd=SCRIPT.parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert probe_marker.is_file()
    assert python_marker.is_file()


def test_explicit_cuda_wrapper_stops_when_nvidia_smi_fails(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python_marker = tmp_path / "python.called"

    _write_executable(
        bin_dir / "nvidia-smi",
        "#!/usr/bin/env bash\nexit 17\n",
    )
    _write_executable(
        bin_dir / "fake-python",
        f"#!/usr/bin/env bash\n"
        f"case \"$*\" in *torch.cuda*) exit 17;; esac\n"
        f"printf 'called\\n' > {python_marker}\n",
    )

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["MNSIM_PYTHON"] = str(bin_dir / "fake-python")

    completed = subprocess.run(
        ["bash", str(SCRIPT), "--device=cuda:0"],
        cwd=SCRIPT.parents[1],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert not python_marker.exists()
