from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from MNSIM.Scheduling.weight_scheduling import load_schedule_artifact
from MNSIM.Scheduling.weight_scheduling import legacy_migration


@dataclass
class FakeHardware:
    device_level: int = 128
    simconfig_path: str = ""
    active_tile_budget: int = 2


@dataclass
class FakeFragment:
    fragment_id: str
    payload: np.ndarray


@dataclass
class FakeLayer:
    fragments: list[FakeFragment]


@dataclass
class FakeRound:
    round_id: str
    fragment_ids: list[str]
    layer_entries: list[dict[str, int]] = field(default_factory=list)
    used_payload_slots: int = 1
    used_tile_count: int = 1


@dataclass
class FakePlan:
    path_name: str
    hardware: FakeHardware
    layers: list[FakeLayer]
    rounds: list[FakeRound]
    feasible: bool = True


def _legacy(path_id: str) -> SimpleNamespace:
    payload = np.arange(12, dtype=np.int16).reshape(3, 4)
    plan = FakePlan(
        path_name=path_id,
        hardware=FakeHardware(),
        layers=[FakeLayer([FakeFragment("F0", payload)])],
        rounds=[FakeRound("R0", ["F0"], [{"raw_layer_index": 0}])],
    )
    return SimpleNamespace(
        artifact_type="mnsim.schedule",
        format_version=1,
        path_id=path_id,
        write_mode="sequential" if path_id == "sched-seq" else "flashsef",
        prepared=True,
        plan=plan,
        schedule_config={"source_config_sha256": "sim-config-hash"},
        hardware_config_snapshot={"active_tile_budget": 2},
        model_metadata={"network": "fake-net", "weights_sha256": "weights-hash"},
        optimizer_metadata={
            "optimizer": "none" if path_id == "sched-seq" else "flashsef"
        },
        physical_write_ledger={
            "enabled": True,
            "round_reports": [{"round_id": "R0", "changed_cells": 7}],
        },
        deployment_plan_by_round_id={
            "R0": {
                "round_id": "R0",
                "optimized_tile_state_by_tile": {
                    0: np.arange(8, dtype=np.uint8).reshape(2, 4)
                },
                "transform_metadata_by_tile": {0: {"permutation": [1, 0]}},
            }
        },
    )


@pytest.mark.parametrize(
    ("path_id", "policy_id"),
    [("sched-seq", "seq"), ("sched-flashsef", "flashsef_ptr")],
)
def test_migrates_legacy_schedule_to_audit_only_schema_v2(
    tmp_path: Path, monkeypatch, path_id: str, policy_id: str
):
    source = tmp_path / "legacy.schedule.pt"
    source.write_bytes(b"trusted legacy pickle fixture")
    destination = tmp_path / "legacy.schedule.schema-v2.json"
    monkeypatch.setattr(
        legacy_migration,
        "_load_trusted_legacy_artifact",
        lambda _path, *, repo_root: _legacy(path_id),
    )

    migrated = legacy_migration.migrate_legacy_schedule(
        source, destination, repo_root=tmp_path
    )
    loaded = load_schedule_artifact(destination)

    assert loaded == migrated
    assert loaded.schema_version == 2
    assert loaded.path_id == path_id
    assert loaded.policy_id == policy_id
    assert loaded.execution_compatible is False
    assert loaded.migration_metadata["source_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert loaded.migration_metadata["execution_status"] == "audit_only"
    assert loaded.round_plan_snapshot["kind"] == "legacy_master_model_execution_plan"
    assert loaded.rounds[0].context_metadata["physical_tile_state_digests"]["0"] == {
        "kind": "ndarray_sha256",
        "dtype": "uint8",
        "shape": [2, 4],
        "element_count": 8,
        "sha256": hashlib.sha256(np.arange(8, dtype=np.uint8).tobytes()).hexdigest(),
    }
    payload = loaded.legacy_schedule_snapshot["plan"]["layers"][0]["fragments"][0][
        "payload"
    ]
    assert payload["kind"] == "ndarray_sha256"
    assert payload["shape"] == [3, 4]
    assert "data" not in payload


def test_missing_legacy_hardware_reports_domain_error(tmp_path: Path, monkeypatch):
    legacy = _legacy("sched-seq")
    legacy.plan.hardware = None
    source = tmp_path / "legacy.schedule.pt"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        legacy_migration,
        "_load_trusted_legacy_artifact",
        lambda _path, *, repo_root: legacy,
    )

    with pytest.raises(
        legacy_migration.LegacyScheduleMigrationError,
        match="Device_Level",
    ):
        legacy_migration.migrate_legacy_schedule(
            source, tmp_path / "out.json", repo_root=tmp_path
        )


def test_rejects_unknown_legacy_path_before_writing(tmp_path: Path, monkeypatch):
    source = tmp_path / "legacy.schedule.pt"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        legacy_migration,
        "_load_trusted_legacy_artifact",
        lambda _path, *, repo_root: _legacy("sched-remap"),
    )
    destination = tmp_path / "out.json"

    with pytest.raises(
        legacy_migration.LegacyScheduleMigrationError,
        match="unsupported legacy path id",
    ):
        legacy_migration.migrate_legacy_schedule(
            source, destination, repo_root=tmp_path
        )
    assert not destination.exists()
