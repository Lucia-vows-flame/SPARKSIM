"""Migrate trusted legacy torch schedule artifacts to schema-v2 JSON audits."""

from __future__ import annotations

import configparser
import dataclasses
import hashlib
import json
import math
import pickle
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from MNSIM.Scheduling.weight_scheduling.artifact import (
    ScheduleArtifact,
    ScheduleRoundRecord,
    save_schedule_artifact,
)
from MNSIM.Scheduling.weight_scheduling.problem import canonical_sha256


class LegacyScheduleMigrationError(ValueError):
    """A legacy schedule cannot be represented as a schema-v2 audit."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        payload = json.dumps(
            _json_snapshot(array.tolist()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest_kind = "ndarray_json_sha256"
    else:
        payload = array.tobytes(order="C")
        digest_kind = "ndarray_sha256"
    return {
        "kind": digest_kind,
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
        "element_count": int(array.size),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _opaque_summary(value: Any) -> dict[str, Any]:
    payload = pickle.dumps(value, protocol=4)
    return {
        "kind": "legacy_python_object_sha256",
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "pickle_protocol": 4,
        "serialized_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _json_snapshot(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"kind": "non_finite_float", "value": repr(value)}
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _array_summary(value)
    if isinstance(value, np.generic):
        return _json_snapshot(value.item())
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_snapshot(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_snapshot(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_snapshot(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_snapshot(item) for item in value), key=repr)
    return _opaque_summary(value)


def _legacy_archive_root(repo_root: Path) -> Path:
    return repo_root / "archive" / "legacy_master_20260925" / "MNSIM-2.0-master"


def _load_trusted_legacy_artifact(path: Path, *, repo_root: Path) -> Any:
    """Load a trusted local pickle using its archived class definitions."""

    archive_root = _legacy_archive_root(repo_root)
    if not archive_root.is_dir():
        raise LegacyScheduleMigrationError(
            f"legacy class archive is unavailable: {archive_root}"
        )
    import importlib

    mnsim_package = importlib.import_module("MNSIM")
    interface_package = importlib.import_module("MNSIM." + "Interface")

    legacy_package = str(archive_root / "MNSIM")
    legacy_interface = str(archive_root / "MNSIM" / "Interface")
    if legacy_package not in mnsim_package.__path__:
        mnsim_package.__path__.append(legacy_package)
    if legacy_interface not in interface_package.__path__:
        interface_package.__path__.append(legacy_interface)
    if str(archive_root) not in sys.path:
        sys.path.append(str(archive_root))

    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise LegacyScheduleMigrationError(
            f"cannot load trusted legacy schedule {path}: {exc}"
        ) from exc


def _policy_id(path_id: str) -> str:
    if path_id == "sched-seq":
        return "seq"
    if path_id == "sched-flashsef":
        return "flashsef_ptr"
    raise LegacyScheduleMigrationError(f"unsupported legacy path id: {path_id!r}")


def _device_identity(legacy: Any) -> tuple[str, int, int]:
    hardware = getattr(getattr(legacy, "plan", None), "hardware", None)
    level = int(getattr(hardware, "device_level", 0) or 0)
    if level < 2 or level & (level - 1):
        raise LegacyScheduleMigrationError(
            f"legacy Device_Level must be a power of two, got {level}"
        )
    bits = level.bit_length() - 1
    config_path = str(getattr(hardware, "simconfig_path", "") or "")
    model_id = ""
    if config_path and Path(config_path).is_file():
        parser = configparser.ConfigParser()
        parser.read(config_path, encoding="UTF-8")
        model_id = parser.get("Final RRAM Model", "Model_ID", fallback="").strip()
    return model_id or f"legacy_device_level_{level}", bits, level


def _legacy_rounds(legacy: Any) -> list[Any]:
    return list(getattr(getattr(legacy, "plan", None), "rounds", []) or [])


def _round_report_by_id(legacy: Any) -> dict[str, dict[str, Any]]:
    ledger = dict(getattr(legacy, "physical_write_ledger", {}) or {})
    return {
        str(item.get("round_id", "")): dict(item)
        for item in list(ledger.get("round_reports", []) or [])
        if isinstance(item, Mapping) and item.get("round_id")
    }


def _round_records(legacy: Any) -> list[ScheduleRoundRecord]:
    reports = _round_report_by_id(legacy)
    deployment = dict(getattr(legacy, "deployment_plan_by_round_id", {}) or {})
    records: list[ScheduleRoundRecord] = []
    for index, legacy_round in enumerate(_legacy_rounds(legacy)):
        round_id = str(getattr(legacy_round, "round_id", f"legacy_R{index}"))
        fragment_ids = [
            str(item) for item in list(getattr(legacy_round, "fragment_ids", []) or [])
        ]
        deployment_round = deployment.get(round_id, {})
        physical_states = dict(
            deployment_round.get("optimized_tile_state_by_tile", {})
            if isinstance(deployment_round, Mapping)
            else {}
        )
        state_digests = {
            str(tile_id): _array_summary(np.asarray(state))
            for tile_id, state in sorted(
                physical_states.items(), key=_tile_sort_key
            )
        }
        report = reports.get(round_id, {})
        records.append(
            ScheduleRoundRecord(
                round_id=round_id,
                round_index=index,
                logical_to_physical_tile={},
                logical_tile_slots={},
                target_states={},
                resident_state_before={},
                resident_state_after={},
                transition_costs={},
                policy_metadata={
                    "migration_status": "legacy_assignment_not_root_compatible",
                    "fragment_count": len(fragment_ids),
                    "used_payload_slots": int(
                        getattr(legacy_round, "used_payload_slots", len(fragment_ids))
                    ),
                    "used_tile_count": int(
                        getattr(legacy_round, "used_tile_count", len(state_digests))
                    ),
                },
                context_metadata={
                    "legacy_fragment_ids": fragment_ids,
                    "legacy_layer_entries": _json_snapshot(
                        getattr(legacy_round, "layer_entries", [])
                    ),
                    "physical_tile_state_digests": state_digests,
                    "write_report": _json_snapshot(report),
                },
                placement_snapshot=None,
            )
        )
    return records


def _plan_snapshot(legacy: Any) -> dict[str, Any]:
    plan = getattr(legacy, "plan", None)
    if plan is None:
        raise LegacyScheduleMigrationError("legacy artifact is missing plan")
    snapshot = _json_snapshot(plan)
    if not isinstance(snapshot, dict):
        raise LegacyScheduleMigrationError("legacy plan snapshot is not an object")
    return snapshot


def _source_sim_config_sha256(legacy: Any) -> str:
    schedule = dict(getattr(legacy, "schedule_config", {}) or {})
    recorded = str(schedule.get("source_config_sha256", "") or "")
    if recorded:
        return recorded
    plan = getattr(legacy, "plan", None)
    hardware = getattr(plan, "hardware", None)
    path = str(getattr(hardware, "simconfig_path", "") or "")
    return sha256_file(path) if path and Path(path).is_file() else ""


def _weights_sha256(legacy: Any) -> str:
    weights = dict(getattr(legacy, "weights_metadata", {}) or {})
    model = dict(getattr(legacy, "model_metadata", {}) or {})
    return str(weights.get("sha256") or model.get("weights_sha256") or "")


def migrate_legacy_schedule(
    source: str | Path,
    destination: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> ScheduleArtifact:
    """Convert one trusted legacy pickle into a deterministic v2 JSON audit.

    The legacy master plan is not the current root RoundPlan and has no current
    RoundPlacementPlan snapshots. The migrated artifact therefore retains all
    portable metadata and array digests while declaring itself audit-only.
    """

    source_path = Path(source).resolve()
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if not source_path.is_file():
        raise LegacyScheduleMigrationError(f"legacy schedule does not exist: {source_path}")
    legacy = _load_trusted_legacy_artifact(source_path, repo_root=root)
    path_id = str(getattr(legacy, "path_id", ""))
    policy_id = _policy_id(path_id)
    plan_snapshot = _plan_snapshot(legacy)
    device_model_id, device_bits, device_level = _device_identity(legacy)
    model_metadata = dict(getattr(legacy, "model_metadata", {}) or {})
    source_sha256 = sha256_file(source_path)
    records = _round_records(legacy)
    legacy_snapshot = {
        "artifact_type": str(getattr(legacy, "artifact_type", "mnsim.schedule")),
        "format_version": int(getattr(legacy, "format_version", 1)),
        "path_id": path_id,
        "write_mode": str(getattr(legacy, "write_mode", "")),
        "prepared": bool(getattr(legacy, "prepared", False)),
        "plan": plan_snapshot,
        "schedule_config": _json_snapshot(
            dict(getattr(legacy, "schedule_config", {}) or {})
        ),
        "hardware_config_snapshot": _json_snapshot(
            dict(getattr(legacy, "hardware_config_snapshot", {}) or {})
        ),
        "topology_snapshot": _json_snapshot(
            dict(getattr(legacy, "topology_snapshot", {}) or {})
        ),
        "model_metadata": _json_snapshot(model_metadata),
        "weights_metadata": _json_snapshot(
            dict(getattr(legacy, "weights_metadata", {}) or {})
        ),
        "dataset_metadata": _json_snapshot(
            dict(getattr(legacy, "dataset_metadata", {}) or {})
        ),
        "optimizer_metadata": _json_snapshot(
            dict(getattr(legacy, "optimizer_metadata", {}) or {})
        ),
        "physical_write_ledger": _json_snapshot(
            dict(getattr(legacy, "physical_write_ledger", {}) or {})
        ),
        "deployment_rounds": _deployment_snapshot(legacy),
        "precomputed_write_ledgers": _json_snapshot(
            dict(getattr(legacy, "precomputed_write_ledgers", {}) or {})
        ),
        "provenance": _json_snapshot(
            dict(getattr(legacy, "provenance", {}) or {})
        ),
    }
    artifact = ScheduleArtifact(
        schema_version=2,
        scheduler_id="unified_weight_scheduler",
        policy_id=policy_id,
        path_id=path_id,
        model_id=str(model_metadata.get("network", "legacy-model")),
        base_round_plan_sha256=canonical_sha256(plan_snapshot),
        model_ir_sha256="",
        weights_sha256=_weights_sha256(legacy),
        sim_config_sha256=_source_sim_config_sha256(legacy),
        device_model_id=device_model_id,
        device_bits=device_bits,
        device_level=device_level,
        initial_resident_state={},
        rounds=records,
        final_resident_state={},
        transition_cost_objective="legacy_physical_write",
        transition_cost_exact=False,
        execution_mode="artifact_only",
        policy_metadata={
            "legacy_optimizer": _json_snapshot(
                dict(getattr(legacy, "optimizer_metadata", {}) or {})
            ),
            "legacy_write_mode": str(getattr(legacy, "write_mode", "")),
        },
        round_plan_snapshot={
            "kind": "legacy_master_model_execution_plan",
            "sha256": canonical_sha256(plan_snapshot),
            "snapshot": plan_snapshot,
        },
        execution_compatible=False,
        migration_metadata={
            "migration_id": "legacy_master_schedule_pt_to_schema_v2_json_v1",
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "source_size_bytes": source_path.stat().st_size,
            "source_trust_requirement": "torch_pickle_must_be_trusted",
            "array_policy": "shape_dtype_sha256_without_inline_payload",
            "execution_status": "audit_only",
            "incompatibility_reason": (
                "legacy ModelExecutionPlan and physical placement are not the current "
                "root RoundPlan/RoundPlacementPlan contracts"
            ),
        },
        legacy_schedule_snapshot=legacy_snapshot,
    )
    save_schedule_artifact(destination, artifact)
    return artifact


def _deployment_snapshot(legacy: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    deployment = dict(getattr(legacy, "deployment_plan_by_round_id", {}) or {})
    for round_id, raw in sorted(deployment.items(), key=lambda pair: str(pair[0])):
        item = dict(raw or {})
        states = dict(item.pop("optimized_tile_state_by_tile", {}) or {})
        transforms = dict(item.pop("transform_metadata_by_tile", {}) or {})
        result[str(round_id)] = {
            **_json_snapshot(item),
            "optimized_tile_state_by_tile": {
                str(tile): _array_summary(np.asarray(state))
                for tile, state in sorted(states.items(), key=_tile_sort_key)
            },
            "transform_metadata_by_tile": {
                str(tile): _opaque_summary(value)
                for tile, value in sorted(
                    transforms.items(), key=_tile_sort_key
                )
            },
        }
    return result


def _tile_sort_key(item: tuple[Any, Any]) -> tuple[int, int | str]:
    try:
        return (0, int(item[0]))
    except (TypeError, ValueError):
        return (1, str(item[0]))


__all__ = [
    "LegacyScheduleMigrationError",
    "migrate_legacy_schedule",
    "sha256_file",
]
