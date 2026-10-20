"""Deterministic schedule artifact serialization and validation."""

from __future__ import annotations

import hashlib
import json
import base64
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from MNSIM.Scheduling.weight_scheduling.transition_cost import AggregatedTransitionCost


class ScheduleArtifactError(ValueError):
    """A schedule artifact is malformed or does not match execution inputs."""


_STATE_COMPRESSION_THRESHOLD = 4096


def _state_to_json(state: list[int] | tuple[int, ...]) -> list[int] | dict[str, Any]:
    """Keep small fixtures readable and compress large tile payloads."""
    values = [int(item) for item in state]
    if len(values) <= _STATE_COMPRESSION_THRESHOLD:
        return values
    if any(item < 0 or item > 255 for item in values):
        raise ScheduleArtifactError("device state code cannot be encoded as a byte")
    packed = zlib.compress(bytes(values), level=1)
    return {
        "encoding": "zlib-base64-u8",
        "length": len(values),
        "data": base64.b64encode(packed).decode("ascii"),
    }


def _state_from_json(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if not isinstance(value, dict) or value.get("encoding") != "zlib-base64-u8":
        raise ScheduleArtifactError("invalid encoded device state")
    try:
        raw = zlib.decompress(base64.b64decode(str(value["data"]), validate=True))
        length = int(value["length"])
    except (KeyError, TypeError, ValueError, zlib.error) as error:
        raise ScheduleArtifactError("invalid compressed device state") from error
    if length != len(raw):
        raise ScheduleArtifactError("compressed device state length mismatch")
    return list(raw)


def _cost_to_dict(cost: AggregatedTransitionCost) -> dict[str, Any]:
    return {
        "changed_cells": int(cost.changed_cells),
        "objective_cost": float(cost.objective_cost),
        "objective_unit": cost.objective_unit,
        "latency_s": cost.latency_s,
        "energy_j": cost.energy_j,
        "exact": bool(cost.exact),
        "provenance": dict(cost.provenance),
    }


def _cost_from_dict(value: dict[str, Any]) -> AggregatedTransitionCost:
    return AggregatedTransitionCost(
        changed_cells=int(value["changed_cells"]),
        objective_cost=float(value["objective_cost"]),
        objective_unit=str(value["objective_unit"]),
        latency_s=None if value.get("latency_s") is None else float(value["latency_s"]),
        energy_j=None if value.get("energy_j") is None else float(value["energy_j"]),
        exact=bool(value["exact"]),
        provenance=dict(value.get("provenance", {})),
    )


@dataclass(frozen=True)
class ScheduleRoundRecord:
    round_id: str
    round_index: int
    logical_to_physical_tile: dict[str, int]
    logical_tile_slots: dict[str, dict[str, int]]
    target_states: dict[str, list[int]]
    resident_state_before: dict[str, list[int]]
    resident_state_after: dict[str, list[int]]
    transition_costs: dict[str, AggregatedTransitionCost]
    policy_metadata: dict[str, Any] = field(default_factory=dict)
    context_metadata: dict[str, Any] = field(default_factory=dict)
    placement_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "round_index": self.round_index,
            "logical_to_physical_tile": dict(sorted(self.logical_to_physical_tile.items())),
            "logical_tile_slots": {
                key: dict(sorted(value.items()))
                for key, value in sorted(self.logical_tile_slots.items())
            },
            "target_states": {
                key: _state_to_json(value) for key, value in sorted(self.target_states.items())
            },
            "resident_state_before": {
                key: _state_to_json(value)
                for key, value in sorted(self.resident_state_before.items())
            },
            "resident_state_after": {
                key: _state_to_json(value)
                for key, value in sorted(self.resident_state_after.items())
            },
            "transition_costs": {
                key: _cost_to_dict(value)
                for key, value in sorted(self.transition_costs.items())
            },
            "policy_metadata": self.policy_metadata,
            "context_metadata": self.context_metadata,
            "placement_snapshot": self.placement_snapshot,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScheduleRoundRecord":
        return cls(
            round_id=str(value["round_id"]),
            round_index=int(value["round_index"]),
            logical_to_physical_tile={
                str(key): int(item) for key, item in value["logical_to_physical_tile"].items()
            },
            logical_tile_slots={
                str(key): {str(slot): int(item) for slot, item in slots.items()}
                for key, slots in value.get("logical_tile_slots", {}).items()
            },
            target_states={
                str(key): _state_from_json(state)
                for key, state in value["target_states"].items()
            },
            resident_state_before={
                str(key): _state_from_json(state)
                for key, state in value["resident_state_before"].items()
            },
            resident_state_after={
                str(key): _state_from_json(state)
                for key, state in value["resident_state_after"].items()
            },
            transition_costs={
                str(key): _cost_from_dict(item)
                for key, item in value.get("transition_costs", {}).items()
            },
            policy_metadata=dict(value.get("policy_metadata", {})),
            context_metadata=dict(value.get("context_metadata", {})),
            placement_snapshot=(
                dict(value["placement_snapshot"])
                if isinstance(value.get("placement_snapshot"), dict)
                else None
            ),
        )


@dataclass(frozen=True)
class ScheduleArtifact:
    schema_version: int
    scheduler_id: str
    policy_id: str
    path_id: str
    model_id: str
    base_round_plan_sha256: str
    model_ir_sha256: str
    weights_sha256: str
    sim_config_sha256: str
    device_model_id: str
    device_bits: int
    device_level: int
    initial_resident_state: dict[str, list[int]]
    rounds: list[ScheduleRoundRecord]
    final_resident_state: dict[str, list[int]]
    transition_cost_objective: str
    transition_cost_exact: bool
    execution_mode: str = "artifact_only"
    policy_metadata: dict[str, Any] = field(default_factory=dict)
    round_plan_snapshot: dict[str, Any] | None = None
    execution_compatible: bool = True
    migration_metadata: dict[str, Any] = field(default_factory=dict)
    legacy_schedule_snapshot: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scheduler_id": self.scheduler_id,
            "policy_id": self.policy_id,
            "path_id": self.path_id,
            "model_id": self.model_id,
            "base_round_plan_sha256": self.base_round_plan_sha256,
            "model_ir_sha256": self.model_ir_sha256,
            "weights_sha256": self.weights_sha256,
            "sim_config_sha256": self.sim_config_sha256,
            "device_model": {
                "model_id": self.device_model_id,
                "device_bits": self.device_bits,
                "device_level": self.device_level,
            },
            "transition_cost": {
                "objective": self.transition_cost_objective,
                "exact": self.transition_cost_exact,
                "latency_modeled": any(
                    cost.latency_s is not None
                    for record in self.rounds
                    for cost in record.transition_costs.values()
                ),
                "energy_modeled": any(
                    cost.energy_j is not None
                    for record in self.rounds
                    for cost in record.transition_costs.values()
                ),
            },
            "initial_resident_state": {
                key: _state_to_json(value)
                for key, value in sorted(self.initial_resident_state.items())
            },
            "rounds": [record.to_dict() for record in self.rounds],
            "final_resident_state": {
                key: _state_to_json(value)
                for key, value in sorted(self.final_resident_state.items())
            },
            "execution_mode": self.execution_mode,
            "policy_metadata": self.policy_metadata,
            "round_plan_snapshot": self.round_plan_snapshot,
            "execution_compatible": self.execution_compatible,
            "migration_metadata": self.migration_metadata,
            "legacy_schedule_snapshot": self.legacy_schedule_snapshot,
        }

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScheduleArtifact":
        schema_version = int(value.get("schema_version", -1))
        if schema_version not in {1, 2}:
            raise ScheduleArtifactError("unsupported schedule artifact schema")
        device = value.get("device_model")
        if not isinstance(device, dict):
            raise ScheduleArtifactError("artifact is missing device_model")
        transition = value.get("transition_cost") or {}
        if value.get("execution_mode") != "artifact_only":
            raise ScheduleArtifactError("schedule artifact must use artifact_only execution")
        return cls(
            schema_version=schema_version,
            scheduler_id=str(value["scheduler_id"]),
            policy_id=str(value["policy_id"]),
            path_id=str(value["path_id"]),
            model_id=str(value.get("model_id", "unknown")),
            base_round_plan_sha256=str(value.get("base_round_plan_sha256", "")),
            model_ir_sha256=str(value.get("model_ir_sha256", "")),
            weights_sha256=str(value.get("weights_sha256", "")),
            sim_config_sha256=str(value.get("sim_config_sha256", "")),
            device_model_id=str(device["model_id"]),
            device_bits=int(device["device_bits"]),
            device_level=int(device["device_level"]),
            initial_resident_state={
                str(key): _state_from_json(state)
                for key, state in value.get("initial_resident_state", {}).items()
            },
            rounds=[ScheduleRoundRecord.from_dict(item) for item in value.get("rounds", [])],
            final_resident_state={
                str(key): _state_from_json(state)
                for key, state in value.get("final_resident_state", {}).items()
            },
            transition_cost_objective=str(transition.get("objective", "changed_cell_count")),
            transition_cost_exact=bool(transition.get("exact", False)),
            execution_mode="artifact_only",
            policy_metadata=dict(value.get("policy_metadata", {})),
            round_plan_snapshot=(
                dict(value["round_plan_snapshot"])
                if isinstance(value.get("round_plan_snapshot"), dict)
                else None
            ),
            execution_compatible=bool(value.get("execution_compatible", True)),
            migration_metadata=dict(value.get("migration_metadata", {})),
            legacy_schedule_snapshot=(
                dict(value["legacy_schedule_snapshot"])
                if isinstance(value.get("legacy_schedule_snapshot"), dict)
                else None
            ),
        )

    def validate(self) -> None:
        if self.scheduler_id != "unified_weight_scheduler":
            raise ScheduleArtifactError("unknown scheduler id")
        expected_path = {
            "seq": "sched-seq",
            "sequential": "sched-seq",
            "flashsef_ptr": "sched-flashsef",
            "flashsef-ptr": "sched-flashsef",
            "snn_native_temporal": "sched-snn_native_temporal",
            "snn_flashsef_temporal": "sched-snn_flashsef_temporal",
            "snn-flashsef-temporal": "sched-snn_flashsef_temporal",
        }.get(self.policy_id, f"sched-{self.policy_id}")
        if self.path_id != expected_path:
            raise ScheduleArtifactError(
                f"policy {self.policy_id!r} does not match path id {self.path_id!r}"
            )
        if self.device_level != 1 << self.device_bits:
            raise ScheduleArtifactError("artifact device level does not match device bits")
        if self.execution_mode != "artifact_only":
            raise ScheduleArtifactError("artifact execution mode must be artifact_only")
        if self.schema_version >= 2 and not isinstance(self.round_plan_snapshot, dict):
            raise ScheduleArtifactError(
                "schema v2 schedule artifact must contain round_plan_snapshot"
            )
        if not self.execution_compatible:
            if not self.migration_metadata:
                raise ScheduleArtifactError(
                    "non-executable artifact must contain migration_metadata"
                )
            if not isinstance(self.legacy_schedule_snapshot, dict):
                raise ScheduleArtifactError(
                    "non-executable artifact must contain legacy_schedule_snapshot"
                )
        for expected_index, record in enumerate(self.rounds):
            if record.round_index != expected_index:
                raise ScheduleArtifactError("artifact round indexes are not contiguous")
            if len(set(record.logical_to_physical_tile.values())) != len(record.logical_to_physical_tile):
                raise ScheduleArtifactError(f"round {record.round_id} has duplicate physical tiles")
            assigned_physical = set(record.logical_to_physical_tile.values())
            if set(record.target_states) != {str(value) for value in assigned_physical}:
                raise ScheduleArtifactError(
                    f"round {record.round_id} target states do not match assignment"
                )
            if set(record.transition_costs) != set(record.logical_to_physical_tile):
                raise ScheduleArtifactError(
                    f"round {record.round_id} transition costs do not match logical tiles"
                )
            for state in record.target_states.values():
                if any(code < 0 or code >= self.device_level for code in state):
                    raise ScheduleArtifactError(f"round {record.round_id} contains an invalid device code")
            if record.round_index > 0:
                previous = self.rounds[record.round_index - 1].resident_state_after
                if record.resident_state_before != previous:
                    raise ScheduleArtifactError(
                        f"round {record.round_id} does not continue the previous resident state"
                    )
        if self.rounds and self.final_resident_state != self.rounds[-1].resident_state_after:
            raise ScheduleArtifactError("final resident state does not match the last round")


def save_schedule_artifact(path: str | Path, artifact: ScheduleArtifact) -> Path:
    artifact.validate()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def load_schedule_artifact(path: str | Path) -> ScheduleArtifact:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScheduleArtifactError(f"cannot load schedule artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ScheduleArtifactError("schedule artifact root must be an object")
    artifact = ScheduleArtifact.from_dict(value)
    artifact.validate()
    return artifact


__all__ = [
    "ScheduleArtifact",
    "ScheduleArtifactError",
    "ScheduleRoundRecord",
    "load_schedule_artifact",
    "save_schedule_artifact",
]
