"""Custom schedule decisions, independent of the internal execution artifact."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Mapping

from MNSIM.Scheduling.weight_scheduling.artifact import ScheduleArtifact
from MNSIM.Scheduling.weight_scheduling.scheduler import (
    AssignmentDecision,
    RoundSchedulingContext,
    WeightScheduleProblem,
    generate_weight_schedule,
)


CUSTOM_SCHEDULE_FORMAT = "mnsim.weight_schedule"
CUSTOM_SCHEDULE_SCHEMA_VERSION = 1
_IDENTITY_FIELDS = {
    "model_id", "base_round_plan_sha256", "model_ir_sha256",
    "weights_sha256", "sim_config_sha256", "device_model_id",
}


class CustomScheduleError(ValueError):
    """A custom schedule cannot be parsed or bound to the simulator inputs."""


def _object(value: Any, name: str, fields: set[str] | None = None) -> dict:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CustomScheduleError(f"{name} must be an object with string keys")
    if fields is not None and set(value) - fields:
        raise CustomScheduleError(f"{name} contains unknown fields: {sorted(set(value) - fields)}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CustomScheduleError(f"{name} must be a non-empty string")
    return value


def _metadata(value: Any, name: str) -> dict:
    _object(value, name)
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise CustomScheduleError(f"{name} must contain finite JSON data") from error
    return value


@dataclass(frozen=True)
class CustomScheduleRound:
    round_id: str
    assignments: dict[str, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.round_id, "round_id")
        _object(self.assignments, "assignments")
        if not self.assignments:
            raise CustomScheduleError(f"round {self.round_id!r} has no assignments")
        for logical_id, physical_id in self.assignments.items():
            _text(logical_id, "logical tile id")
            if type(physical_id) is not int or physical_id < 0:
                raise CustomScheduleError("physical tile ids must be non-negative integers")
        if len(set(self.assignments.values())) != len(self.assignments):
            raise CustomScheduleError(f"round {self.round_id!r} assigns duplicate physical tiles")
        _metadata(self.metadata, "round.metadata")

    def to_dict(self) -> dict[str, Any]:
        value = {"round_id": self.round_id, "assignments": dict(sorted(self.assignments.items()))}
        if self.metadata:
            value["metadata"] = deepcopy(self.metadata)
        return value

    @classmethod
    def from_dict(cls, value: Any) -> CustomScheduleRound:
        _object(value, "round", {"round_id", "assignments", "metadata"})
        return cls(value.get("round_id"), value.get("assignments"), value.get("metadata", {}))


@dataclass(frozen=True)
class CustomSchedule:
    policy_id: str
    rounds: tuple[CustomScheduleRound, ...]
    identity: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CUSTOM_SCHEDULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise CustomScheduleError(f"unsupported custom schedule schema {self.schema_version!r}")
        if not isinstance(self.policy_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.policy_id):
            raise CustomScheduleError("policy_id must use letters, digits, dots, underscores or hyphens")
        if not isinstance(self.rounds, (tuple, list)) or not self.rounds:
            raise CustomScheduleError("rounds must contain at least one round")
        if any(not isinstance(item, CustomScheduleRound) for item in self.rounds):
            raise CustomScheduleError("rounds must contain CustomScheduleRound objects")
        ids = [item.round_id for item in self.rounds]
        if len(set(ids)) != len(ids):
            raise CustomScheduleError("custom schedule round ids must be unique")
        _object(self.identity, "identity", _IDENTITY_FIELDS)
        for key, value in self.identity.items():
            _text(value, f"identity.{key}")
            if key.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise CustomScheduleError(f"identity.{key} must be a lowercase SHA-256 digest")
        _metadata(self.metadata, "metadata")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "format": CUSTOM_SCHEDULE_FORMAT,
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "rounds": [item.to_dict() for item in self.rounds],
        }
        if self.identity:
            value["identity"] = dict(sorted(self.identity.items()))
        if self.metadata:
            value["metadata"] = deepcopy(self.metadata)
        return value

    def canonical_bytes(self) -> bytes:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> CustomSchedule:
        _object(value, "custom schedule", {"format", "schema_version", "policy_id", "rounds", "identity", "metadata"})
        if value.get("format") != CUSTOM_SCHEDULE_FORMAT:
            raise CustomScheduleError(f"unsupported custom schedule format {value.get('format')!r}")
        if not isinstance(value.get("rounds"), list):
            raise CustomScheduleError("rounds must be an array")
        return cls(
            policy_id=value.get("policy_id"),
            rounds=tuple(CustomScheduleRound.from_dict(item) for item in value["rounds"]),
            identity=value.get("identity", {}), metadata=value.get("metadata", {}),
            schema_version=value.get("schema_version"),
        )


def save_custom_schedule(path: str | Path, schedule: CustomSchedule) -> Path:
    CustomSchedule.from_dict(schedule.to_dict())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(schedule.canonical_bytes() + b"\n")
    temporary.replace(target)
    return target


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise CustomScheduleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_custom_schedule(path: str | Path) -> CustomSchedule:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CustomScheduleError(f"cannot load custom schedule {path}: {error}") from error
    return CustomSchedule.from_dict(value)


def custom_schedule_from_artifact(
    artifact: ScheduleArtifact, *, policy_id: str | None = None,
) -> CustomSchedule:
    """Extract a tile-level decision template, not a full artifact replay.

    Internal PE/group permutations are deliberately omitted. The base plan
    digest is omitted too: executable artifacts can hash a reordered plan,
    whereas a custom schedule binds to the original, unscheduled plan.
    """
    artifact.validate()
    if not artifact.execution_compatible:
        raise CustomScheduleError("cannot export an audit-only schedule artifact")
    identity = {
        key: getattr(artifact, key)
        for key in _IDENTITY_FIELDS - {"base_round_plan_sha256"}
        if getattr(artifact, key)
    }
    return CustomSchedule(
        policy_id=artifact.policy_id if policy_id is None else policy_id,
        rounds=tuple(
            CustomScheduleRound(record.round_id, dict(record.logical_to_physical_tile))
            for record in artifact.rounds
        ),
        identity=identity,
    )


def _problem_identity(problem: WeightScheduleProblem) -> dict[str, str]:
    return {
        **{key: str(getattr(problem, key)) for key in _IDENTITY_FIELDS - {"device_model_id"}},
        "device_model_id": str(problem.device_model.model_id),
    }


def _validate_order(problem: WeightScheduleProblem, ordered: tuple[RoundSchedulingContext, ...]) -> None:
    # Absent reorder metadata fixes a round in place. Explicit contiguous
    # blocks allow permutations, but dependencies still constrain each order.
    block_by_id = {}
    block = -1
    previous_key = None
    for context in problem.rounds:
        key = context.metadata.get("reorder_block")
        if key is None or key == "fixed" or key != previous_key:
            block += 1
        block_by_id[context.round_id] = block
        previous_key = None if key == "fixed" else key
    if [block_by_id[c.round_id] for c in ordered] != [block_by_id[c.round_id] for c in problem.rounds]:
        raise CustomScheduleError("round order crosses a fixed execution boundary")
    seen = set()
    for context in ordered:
        predecessors = set(context.metadata.get("predecessor_round_ids", []))
        if not predecessors.issubset(seen):
            raise CustomScheduleError(f"round {context.round_id!r} violates predecessor dependencies: {sorted(predecessors - seen)}")
        seen.add(context.round_id)


class CustomSchedulePolicy:
    """Custom execution decisions adapted to the existing scheduler lifecycle."""

    def __init__(self, schedule: CustomSchedule):
        # Snapshot mutable dictionaries before validating or executing them.
        self.schedule = CustomSchedule.from_dict(schedule.to_dict())
        self.policy_id = self.schedule.policy_id
        self._rounds_by_id = {item.round_id: item for item in self.schedule.rounds}
        self._planned_rounds = ()
        self._digest = self.schedule.sha256()
        self.schedule_metadata = {
            "custom_schedule_schema_version": 1,
            "custom_schedule_sha256": self._digest,
            "custom_metadata": deepcopy(self.schedule.metadata),
            "round_order": [item.round_id for item in self.schedule.rounds],
        }

    def prepare_schedule(self, problem: WeightScheduleProblem, _cost_engine: Any) -> None:
        actual_identity = _problem_identity(problem)
        for key, expected in self.schedule.identity.items():
            if expected != actual_identity[key]:
                raise CustomScheduleError(f"custom schedule identity mismatch for {key}")
        contexts = {context.round_id: context for context in problem.rounds}
        if set(self._rounds_by_id) != set(contexts):
            raise CustomScheduleError(
                "custom schedule round ids do not match problem: "
                f"missing={sorted(set(contexts) - set(self._rounds_by_id))}, "
                f"extra={sorted(set(self._rounds_by_id) - set(contexts))}"
            )
        self._planned_rounds = tuple(contexts[item.round_id] for item in self.schedule.rounds)
        _validate_order(problem, self._planned_rounds)
        # Validate every decision before the scheduler builds cost matrices.
        for context in self._planned_rounds:
            assignments = self._rounds_by_id[context.round_id].assignments
            expected = {tile.tile_id for tile in context.logical_tiles}
            if set(assignments) != expected:
                raise CustomScheduleError(f"round {context.round_id!r} tile ids differ: missing={sorted(expected - set(assignments))}, extra={sorted(set(assignments) - expected)}")
            if not set(assignments.values()).issubset(problem.constraints.physical_tile_ids):
                raise CustomScheduleError(f"round {context.round_id!r} uses unavailable physical tiles")
            limit = problem.constraints.max_active_tiles
            if limit is not None and len(assignments) > limit:
                raise CustomScheduleError(f"round {context.round_id!r} exceeds max_active_tiles")

    def choose_assignment(self, context, resident_state, transition_costs, constraints):
        custom_round = self._rounds_by_id[context.round_id]
        return AssignmentDecision(
            round_id=context.round_id,
            logical_to_physical_tile=dict(custom_round.assignments),
            logical_tile_slots={tile.tile_id: dict(tile.local_slots) for tile in context.logical_tiles},
            target_states={custom_round.assignments[tile.tile_id]: tile.target_state for tile in context.logical_tiles},
            policy_metadata={
                "source": "custom_schedule", "custom_schedule_sha256": self._digest,
                "custom_metadata": deepcopy(custom_round.metadata),
            },
        )


def generate_weight_schedule_from_custom(problem: WeightScheduleProblem, schedule: CustomSchedule):
    """Reconstruct simulator-owned resident states and costs from decisions."""
    artifact = generate_weight_schedule(problem, CustomSchedulePolicy(schedule))
    artifact.validate()
    return artifact


def remap_custom_schedule_placement(snapshot: Mapping[str, Any], assignments: Mapping[str, int]) -> dict[str, Any]:
    """Move whole tiles and recalculate reducer/LIF owners from physical slots."""
    from MNSIM.Scheduling.phase2.tcg_adapter import _output_owner_shards_by_layer

    remapped = deepcopy(dict(snapshot))
    for slot in remapped["fragment_slots_by_id"].values():
        slot["tile_id"] = assignments[f"tile:{int(slot['tile_id'])}"]
    remapped["output_owner_shards_by_layer"] = {
        layer_id: {str(tile_id): count for tile_id, count in shards.items()}
        for layer_id, shards in _output_owner_shards_by_layer(
            item_id=remapped["item_id"],
            fragment_specs_by_id={key: SimpleNamespace(**value) for key, value in remapped["fragment_specs_by_id"].items()},
            fragment_slots_by_id=remapped["fragment_slots_by_id"],
        ).items()
    }
    return remapped


__all__ = [
    "CUSTOM_SCHEDULE_FORMAT", "CUSTOM_SCHEDULE_SCHEMA_VERSION", "CustomSchedule",
    "CustomScheduleError", "CustomSchedulePolicy", "CustomScheduleRound",
    "generate_weight_schedule_from_custom", "load_custom_schedule", "save_custom_schedule",
    "custom_schedule_from_artifact",
]
