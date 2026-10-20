from __future__ import annotations

import importlib.util
import os
import subprocess
from collections import Counter
from pathlib import Path


BUNDLE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "exports"
    / "wsl_v7_lut_pim1600_prequantized_20260816"
)


def _load_runner_module():
    path = BUNDLE_ROOT / "tools" / "run_lut_pim.py"
    spec = importlib.util.spec_from_file_location("wsl_lut_pim_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_task(task: str) -> str:
    env = os.environ.copy()
    env["PYTHON_BIN"] = "/bin/echo"
    completed = subprocess.run(
        ["bash", str(BUNDLE_ROOT / "RUN_TASK.sh"), task],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return completed.stdout


def test_legacy_task_keeps_variation_enabled() -> None:
    output = _run_task("standard-20ns")

    assert "--variation on" in output


def test_explicit_task_can_disable_variation() -> None:
    output = _run_task("standard-20ns-var-off")

    assert "--variation off" in output


def test_all_both_runs_each_model_and_device_with_both_states() -> None:
    output = _run_task("all-both")

    assert output.count("--variation on") == 4
    assert output.count("--variation off") == 4


def test_seeded_stratified_subset_is_balanced_and_repeatable() -> None:
    runner = _load_runner_module()
    files = [Path(f"class_{label}_{index}.npz") for label in range(8) for index in range(5)]

    def read_label(path: Path) -> int:
        return int(path.stem.split("_")[1])

    first = runner._select_stratified_files(
        files,
        label_reader=read_label,
        num_classes=8,
        samples_per_class=3,
        seed=112,
    )
    second = runner._select_stratified_files(
        files,
        label_reader=read_label,
        num_classes=8,
        samples_per_class=3,
        seed=112,
    )

    assert first == second
    assert len(first) == 24
    assert Counter(read_label(path) for path in first) == {label: 3 for label in range(8)}


def test_subset_manifest_hash_is_stable() -> None:
    runner = _load_runner_module()
    root = Path("/dataset")
    selected = [(root / "test" / f"{index:06d}.npz", index % 8) for index in range(16)]

    first = runner._build_subset_manifest(root, selected, seed=112, samples_per_class=2)
    second = runner._build_subset_manifest(root, selected, seed=112, samples_per_class=2)

    assert first == second
    assert first["sample_count"] == 16
    assert first["class_counts"] == {str(label): 2 for label in range(8)}
    assert len(first["manifest_sha256"]) == 64


def test_device_variation_rng_uses_fresh_system_entropy(monkeypatch) -> None:
    runner = _load_runner_module()
    expected = object()
    calls = []

    def default_rng(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(runner.np.random, "default_rng", default_rng)

    assert runner._new_device_variation_rng() is expected
    assert calls == [((), {})]
