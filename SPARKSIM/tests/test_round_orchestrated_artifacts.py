import json
from pathlib import Path
from typing import get_type_hints

import pytest

import MNSIM.Interface.round_orchestrated_artifacts as artifact_module
from MNSIM.Scheduling.phase2.types import Phase2Report
from MNSIM.Scheduling.round_types import RoundPlan
from MNSIM.Interface.round_orchestrated_artifacts import (
    RUNNER_OWNED_ARTIFACTS,
    ArtifactStore,
)
from MNSIM.Interface.round_orchestrated_types import (
    ErrorRecord,
    RoundOrchestratedRunRequest,
    RoundOrchestratedRunResult,
    RunnerError,
    StageStatus,
)


def test_nonempty_directory_without_overwrite_is_rejected_with_stable_error_code(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "user-notes.txt").write_text("keep me", encoding="utf-8")

    with pytest.raises(RunnerError) as error:
        ArtifactStore.open(output_dir, overwrite=False)

    assert error.value.code == "INVALID_OUTPUT_DIRECTORY"
    assert str(error.value).startswith("INVALID_OUTPUT_DIRECTORY:")


def test_open_anchors_relative_output_directory_before_cwd_changes(tmp_path, monkeypatch):
    initial_cwd = tmp_path / "initial-cwd"
    later_cwd = tmp_path / "later-cwd"
    initial_cwd.mkdir()
    later_cwd.mkdir()
    monkeypatch.chdir(initial_cwd)

    store = ArtifactStore.open(Path("relative-output"), overwrite=False)
    expected_output_dir = initial_cwd / "relative-output"

    assert store.output_dir == expected_output_dir
    monkeypatch.chdir(later_cwd)
    target = store.write_json("summary.json", {"status": "written"})

    assert target == expected_output_dir / "summary.json"
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "written"}
    assert not (later_cwd / "relative-output").exists()


def test_open_preserves_a_user_supplied_symlink_output_path(tmp_path):
    target_directory = tmp_path / "target"
    target_directory.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(target_directory, target_is_directory=True)

    store = ArtifactStore.open(output_link, overwrite=False)
    written = store.write_json("summary.json", {"status": "written"})

    assert store.output_dir == output_link.absolute()
    assert written == output_link.absolute() / "summary.json"
    assert (target_directory / "summary.json").exists()


def test_output_path_that_is_a_regular_file_is_rejected_with_stable_error_code(tmp_path):
    output_path = tmp_path / "run"
    output_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(RunnerError) as error:
        ArtifactStore.open(output_path, overwrite=False)

    assert error.value.code == "INVALID_OUTPUT_DIRECTORY"


def test_mkdir_failure_is_converted_to_stable_runner_error(tmp_path, monkeypatch):
    def fail_mkdir(self, *args, **kwargs):
        raise OSError("mkdir denied")

    monkeypatch.setattr(artifact_module.Path, "mkdir", fail_mkdir)

    with pytest.raises(RunnerError) as error:
        ArtifactStore.open(tmp_path / "run", overwrite=False)

    assert error.value.code == "INVALID_OUTPUT_DIRECTORY"


def test_overwrite_removes_only_runner_owned_files_and_preserves_user_content(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "summary.json").write_text('{"old": true}\n', encoding="utf-8")
    (output_dir / "round_plan.json").write_text('{"old": true}\n', encoding="utf-8")
    (output_dir / "user-notes.txt").write_text("keep me", encoding="utf-8")
    user_directory = output_dir / "user-data"
    user_directory.mkdir()
    (user_directory / "notes.txt").write_text("also keep me", encoding="utf-8")

    store = ArtifactStore.open(output_dir, overwrite=True)

    assert not (output_dir / "summary.json").exists()
    assert not (output_dir / "round_plan.json").exists()
    assert (output_dir / "user-notes.txt").read_text(encoding="utf-8") == "keep me"
    assert (user_directory / "notes.txt").read_text(encoding="utf-8") == "also keep me"

    summary_path = store.write_json("summary.json", {"status": "new"})

    assert summary_path == output_dir / "summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == {"status": "new"}


def test_overwrite_rejects_runner_owned_path_that_is_not_a_regular_file(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "summary.json").mkdir()

    with pytest.raises(RunnerError) as error:
        ArtifactStore.open(output_dir, overwrite=True)

    assert error.value.code == "INVALID_OUTPUT_DIRECTORY"


def test_overwrite_removes_only_regular_runner_owned_temp_files(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_owned_temp = output_dir / ".summary.json.crashed.tmp"
    stale_owned_temp.write_text("partial runner data", encoding="utf-8")
    unrelated_temp = output_dir / ".user-notes.crashed.tmp"
    unrelated_temp.write_text("keep me", encoding="utf-8")
    similarly_named_file = output_dir / ".summary.json.tmp"
    similarly_named_file.write_text("keep me too", encoding="utf-8")
    similarly_named_directory = output_dir / ".summary.json.directory.tmp"
    similarly_named_directory.mkdir()

    ArtifactStore.open(output_dir, overwrite=True)

    assert not stale_owned_temp.exists()
    assert unrelated_temp.read_text(encoding="utf-8") == "keep me"
    assert similarly_named_file.read_text(encoding="utf-8") == "keep me too"
    assert similarly_named_directory.is_dir()


def test_atomic_writer_leaves_no_temp_file_and_writes_readable_json(tmp_path):
    store = ArtifactStore.open(tmp_path / "run", overwrite=False)

    target = store.write_json("summary.json", {"beta": [1, 2], "alpha": True})

    contents = target.read_bytes()
    text = contents.decode("utf-8")

    assert json.loads(text) == {
        "alpha": True,
        "beta": [1, 2],
    }
    assert text == '{\n  "alpha": true,\n  "beta": [\n    1,\n    2\n  ]\n}\n'
    assert not text.endswith("\n\n")
    assert list(target.parent.glob(".*.tmp")) == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_writer_rejects_nonfinite_floats_and_cleans_its_temporary_file(tmp_path, value):
    store = ArtifactStore.open(tmp_path / "run", overwrite=False)

    with pytest.raises(ValueError, match="not JSON compliant"):
        store.write_json("summary.json", {"value": value})

    assert list(store.output_dir.glob(".*.tmp")) == []


def test_writer_syncs_output_directory_after_atomic_replace(tmp_path, monkeypatch):
    store = ArtifactStore.open(tmp_path / "run", overwrite=False)
    events = []
    real_replace = artifact_module.os.replace

    def record_replace(source, target):
        events.append("replace")
        return real_replace(source, target)

    def record_directory_fsync(output_dir):
        events.append(("directory_fsync", output_dir))

    monkeypatch.setattr(artifact_module.os, "replace", record_replace)
    monkeypatch.setattr(
        artifact_module,
        "_fsync_directory",
        record_directory_fsync,
        raising=False,
    )

    store.write_json("summary.json", {"status": "written"})

    assert events == ["replace", ("directory_fsync", store.output_dir)]


def test_directory_fsync_failure_is_propagated_after_replace(tmp_path, monkeypatch):
    store = ArtifactStore.open(tmp_path / "run", overwrite=False)

    def fail_directory_fsync(output_dir):
        raise OSError("directory fsync failed")

    monkeypatch.setattr(
        artifact_module,
        "_fsync_directory",
        fail_directory_fsync,
        raising=False,
    )

    with pytest.raises(OSError, match="directory fsync failed"):
        store.write_json("summary.json", {"status": "written"})

    assert (store.output_dir / "summary.json").exists()
    assert list(store.output_dir.glob(".*.tmp")) == []


def test_unknown_artifact_is_rejected_with_stable_error_code(tmp_path):
    store = ArtifactStore.open(tmp_path / "run", overwrite=False)

    with pytest.raises(RunnerError) as error:
        store.write_json("user-notes.txt", {"status": "not-owned"})

    assert error.value.code == "INVALID_OUTPUT_DIRECTORY"


@pytest.mark.parametrize(
    ("skip_accuracy", "run_hardware", "expected"),
    [
        (False, False, ("accuracy",)),
        (False, True, ("accuracy", "hardware")),
        (True, False, ()),
        (True, True, ("hardware",)),
    ],
)
def test_request_reports_the_requested_stage_combination(
    tmp_path, skip_accuracy, run_hardware, expected
):
    request = RoundOrchestratedRunRequest(
        network="fake-network",
        weights=tmp_path / "weights.pth",
        num_classes=2,
        sim_config=tmp_path / "SimConfig.ini",
        output_dir=tmp_path / "output",
        skip_accuracy=skip_accuracy,
        run_hardware=run_hardware,
    )

    assert request.requested_stages() == expected
    assert isinstance(request.requested_stages(), tuple)


def test_requested_stages_declares_the_stable_tuple_contract():
    hints = get_type_hints(RoundOrchestratedRunRequest.requested_stages)

    assert hints["return"] == tuple[str, ...]


def test_result_type_hints_resolve_scheduling_contracts_at_runtime():
    hints = get_type_hints(RoundOrchestratedRunResult)

    assert hints["round_plan"] == RoundPlan | None
    assert hints["phase2_report"] == Phase2Report | None


def test_request_and_result_serializations_are_json_safe_and_track_completed_stages(tmp_path):
    request = RoundOrchestratedRunRequest(
        network="fake-network",
        weights=Path(tmp_path / "weights.pth"),
        num_classes=2,
        sim_config=Path(tmp_path / "SimConfig.ini"),
        output_dir=Path(tmp_path / "output"),
        dataset_module="tests.fake_dataset",
        run_hardware=True,
    )
    result = RoundOrchestratedRunResult(
        status="partial_failure",
        exit_code=1,
        request=request,
        stage_statuses={
            "hardware": StageStatus(
                name="hardware",
                status="failed",
                error=ErrorRecord("HARDWARE_FAILED", "fake failure", stage="hardware"),
                summary={"trace_path": tmp_path / "hardware.trace"},
            ),
            "accuracy": StageStatus(
                name="accuracy",
                status="succeeded",
                summary={"report_path": tmp_path / "accuracy.detail"},
            ),
        },
        artifacts={"summary": str(tmp_path / "summary.json")},
        errors=[ErrorRecord("HARDWARE_FAILED", "fake failure", stage="hardware")],
        accuracy_report={"report_path": tmp_path / "accuracy_report.json"},
        hardware_summary={"report_path": tmp_path / "hardware_summary.json"},
    )

    request_data = request.to_dict()
    manifest = result.manifest_dict()
    summary = result.summary_dict()

    assert request_data["weights"] == str(tmp_path / "weights.pth")
    assert request_data["sim_config"] == str(tmp_path / "SimConfig.ini")
    assert request_data["output_dir"] == str(tmp_path / "output")
    assert result.completed_stages() == ["accuracy"]
    assert manifest["schema_version"] == 1
    assert manifest["requested_stages"] == ["accuracy", "hardware"]
    assert isinstance(manifest["requested_stages"], list)
    assert manifest["stage_statuses"]["accuracy"]["summary"]["report_path"] == str(
        tmp_path / "accuracy.detail"
    )
    assert manifest["artifacts"]["summary"] == str(tmp_path / "summary.json")
    assert summary["schema_version"] == 1
    assert summary["network"] == "fake-network"
    assert summary["sim_config"] == str(tmp_path / "SimConfig.ini")
    assert summary["requested_stages"] == ["accuracy", "hardware"]
    assert isinstance(summary["requested_stages"], list)
    assert summary["completed_stages"] == ["accuracy"]
    assert summary["accuracy"]["report_path"] == str(tmp_path / "accuracy_report.json")
    assert summary["hardware"]["report_path"] == str(tmp_path / "hardware_summary.json")
    assert summary["errors"][0] == {
        "code": "HARDWARE_FAILED",
        "detail": "fake failure",
        "stage": "hardware",
    }
    json.dumps(request_data)
    json.dumps(manifest)
    json.dumps(summary)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_result_serialization_rejects_nonfinite_floats(tmp_path, value):
    request = RoundOrchestratedRunRequest(
        network="fake-network",
        weights=tmp_path / "weights.pth",
        num_classes=2,
        sim_config=tmp_path / "SimConfig.ini",
        output_dir=tmp_path / "output",
    )
    result = RoundOrchestratedRunResult(
        status="failed",
        exit_code=1,
        request=request,
        accuracy_report={"value": value},
    )

    with pytest.raises(TypeError, match="non-finite float"):
        result.summary_dict()


def test_manifest_rejects_unknown_stage_summary_values(tmp_path):
    request = RoundOrchestratedRunRequest(
        network="fake-network",
        weights=tmp_path / "weights.pth",
        num_classes=2,
        sim_config=tmp_path / "SimConfig.ini",
        output_dir=tmp_path / "output",
    )
    result = RoundOrchestratedRunResult(
        status="failed",
        exit_code=1,
        request=request,
        stage_statuses={
            "accuracy": StageStatus(
                name="accuracy",
                status="failed",
                summary={"unsupported": object()},
            )
        },
    )

    with pytest.raises(TypeError, match="unsupported JSON value"):
        result.manifest_dict()


@pytest.mark.parametrize("report_field", ["accuracy_report", "hardware_summary"])
def test_summary_rejects_unknown_report_values(tmp_path, report_field):
    request = RoundOrchestratedRunRequest(
        network="fake-network",
        weights=tmp_path / "weights.pth",
        num_classes=2,
        sim_config=tmp_path / "SimConfig.ini",
        output_dir=tmp_path / "output",
    )
    result = RoundOrchestratedRunResult(
        status="failed",
        exit_code=1,
        request=request,
        **{report_field: {"unsupported": object()}},
    )

    with pytest.raises(TypeError, match="unsupported JSON value"):
        result.summary_dict()


def test_owned_artifact_set_is_exact_and_immutable():
    assert RUNNER_OWNED_ARTIFACTS == frozenset(
        {
            "run_manifest.json",
            "summary.json",
            "accuracy_report.json",
            "deployment_weights_9bit.pth",
            "deployment_weights_9bit_manifest.json",
            "readc_calibration.json",
            "round_plan.json",
            "phase2_report.json",
        }
    )


@pytest.mark.parametrize("failure_target", ["json_dump", "replace"])
def test_failed_atomic_write_removes_its_temporary_file(tmp_path, monkeypatch, failure_target):
    store = ArtifactStore.open(tmp_path / "run", overwrite=False)

    if failure_target == "json_dump":
        monkeypatch.setattr(
            artifact_module.json,
            "dump",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("dump failed")),
        )
        expected_error = RuntimeError
    else:
        monkeypatch.setattr(
            artifact_module.os,
            "replace",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replace failed")),
        )
        expected_error = OSError

    with pytest.raises(expected_error):
        store.write_json("summary.json", {"status": "new"})

    assert list(store.output_dir.glob(".*.tmp")) == []


def test_replace_error_is_not_masked_by_temporary_cleanup_error(tmp_path, monkeypatch):
    store = ArtifactStore.open(tmp_path / "run", overwrite=False)
    original_unlink = artifact_module.Path.unlink

    def fail_replace(*args, **kwargs):
        raise OSError("replace failed")

    def fail_temporary_unlink(path, *args, **kwargs):
        if path.name.startswith(".summary.json.") and path.name.endswith(".tmp"):
            raise OSError("cleanup failed")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patches:
        patches.setattr(artifact_module.os, "replace", fail_replace)
        patches.setattr(artifact_module.Path, "unlink", fail_temporary_unlink)

        with pytest.raises(OSError, match="replace failed"):
            store.write_json("summary.json", {"status": "new"})

    temporary_paths = list(store.output_dir.glob(".summary.json.*.tmp"))
    assert temporary_paths
    for temporary_path in temporary_paths:
        temporary_path.unlink()
    assert list(store.output_dir.glob(".summary.json.*.tmp")) == []
