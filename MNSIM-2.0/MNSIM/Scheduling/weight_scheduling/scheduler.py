"""Unified round-by-round weight scheduler."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from MNSIM.Scheduling.weight_scheduling.device import DeviceModel
from MNSIM.Scheduling.weight_scheduling.transition_cost import (
    AggregatedTransitionCost,
    TransitionCostEngine,
    TransitionCostView,
)


class SchedulingError(ValueError):
    """A scheduling input or policy decision violates the common contract."""


@dataclass(frozen=True)
class LogicalTile:
    tile_id: str
    target_state: tuple[int, ...]
    local_slots: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.tile_id):
            raise SchedulingError("logical tile id cannot be empty")
        if not self.target_state:
            raise SchedulingError(f"logical tile {self.tile_id} has an empty state")


@dataclass(frozen=True)
class RoundSchedulingContext:
    round_id: str
    round_index: int
    logical_tiles: tuple[LogicalTile, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulingConstraints:
    physical_tile_ids: tuple[int, ...]
    preserve_local_slots: bool = True
    max_active_tiles: int | None = None

    def __post_init__(self) -> None:
        if not self.physical_tile_ids:
            raise SchedulingError("at least one physical tile is required")
        if len(set(self.physical_tile_ids)) != len(self.physical_tile_ids):
            raise SchedulingError("physical tile ids must be unique")
        if self.max_active_tiles is not None and self.max_active_tiles <= 0:
            raise SchedulingError("max_active_tiles must be positive")


@dataclass
class ResidentState:
    by_physical_tile: dict[int, tuple[int, ...]] = field(default_factory=dict)
    # Native physical scheduling keeps a second, slot-aware view of resident
    # state.  The compact tile payload above remains the public artifact view;
    # this map preserves idle fragment slots between weighted rounds.
    physical_by_physical_tile: dict[int, dict[str, dict[str, Any]]] = field(
        default_factory=dict
    )
    last_round_index: int = -1
    assignments_by_round: dict[str, dict[str, int]] = field(default_factory=dict)

    def copy(self) -> "ResidentState":
        return ResidentState(
            dict(self.by_physical_tile),
            {
                int(tile_id): {str(slot): dict(state) for slot, state in slots.items()}
                for tile_id, slots in self.physical_by_physical_tile.items()
            },
            self.last_round_index,
            {key: dict(value) for key, value in self.assignments_by_round.items()},
        )

    def digest(self) -> str:
        payload = {
            "last_round_index": self.last_round_index,
            "by_physical_tile": {
                str(key): list(value)
                for key, value in sorted(self.by_physical_tile.items())
            },
            "physical_by_physical_tile": {
                str(tile_id): {
                    str(slot): {
                        "shape": list(state["shape"]),
                        "values": list(state["values"]),
                    }
                    for slot, state in sorted(slots.items())
                }
                for tile_id, slots in sorted(self.physical_by_physical_tile.items())
            },
            "assignments_by_round": {
                round_id: dict(sorted(assignment.items()))
                for round_id, assignment in sorted(self.assignments_by_round.items())
            },
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class AssignmentDecision:
    round_id: str
    logical_to_physical_tile: dict[str, int]
    logical_tile_slots: dict[str, dict[str, int]]
    target_states: dict[int, tuple[int, ...]]
    policy_metadata: dict[str, Any] = field(default_factory=dict)


class SchedulingPolicy(Protocol):
    policy_id: str

    def choose_assignment(
        self,
        context: RoundSchedulingContext,
        resident_state: ResidentState,
        transition_costs: TransitionCostView,
        constraints: SchedulingConstraints,
    ) -> AssignmentDecision:
        ...


@dataclass(frozen=True)
class WeightScheduleProblem:
    rounds: tuple[RoundSchedulingContext, ...]
    constraints: SchedulingConstraints
    device_model: DeviceModel
    model_id: str = "unknown"
    base_round_plan_sha256: str = ""
    model_ir_sha256: str = ""
    weights_sha256: str = ""
    sim_config_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indexes = [round_context.round_index for round_context in self.rounds]
        if indexes != list(range(len(indexes))):
            raise SchedulingError("round indexes must be contiguous and start at zero")
        ids = [round_context.round_id for round_context in self.rounds]
        if len(ids) != len(set(ids)):
            raise SchedulingError("round ids must be unique")
        for context in self.rounds:
            tile_ids = [tile.tile_id for tile in context.logical_tiles]
            if len(tile_ids) != len(set(tile_ids)):
                raise SchedulingError(f"round {context.round_id} has duplicate logical tiles")
            if len(tile_ids) > len(self.constraints.physical_tile_ids):
                raise SchedulingError(
                    f"round {context.round_id} requires {len(tile_ids)} physical tiles, "
                    f"but only {len(self.constraints.physical_tile_ids)} are available"
                )
            for tile in context.logical_tiles:
                for code in tile.target_state:
                    self.device_model.validate_code(int(code))


def _assignment_costs(
    logical_tiles: tuple[LogicalTile, ...],
    resident_state: ResidentState,
    constraints: SchedulingConstraints,
    cost_engine: TransitionCostEngine,
) -> TransitionCostView:
    return cost_engine.matrix(
        resident_state.by_physical_tile,
        {tile.tile_id: tile.target_state for tile in logical_tiles},
        list(constraints.physical_tile_ids),
        target_tiles=logical_tiles,
        source_physical_state_by_tile=resident_state.physical_by_physical_tile,
    )


class UnifiedWeightScheduler:
    """Run any policy through one state, cost, and artifact lifecycle."""

    def __init__(self, *, cost_engine: TransitionCostEngine):
        self.cost_engine = cost_engine

    def generate(self, problem: WeightScheduleProblem, policy: SchedulingPolicy):
        from MNSIM.Scheduling.weight_scheduling.artifact import (
            ScheduleArtifact,
            ScheduleRoundRecord,
        )

        prepare_schedule = getattr(policy, "prepare_schedule", None)
        if callable(prepare_schedule):
            prepare_schedule(problem, self.cost_engine)

        resident = ResidentState()
        initial_digest = resident.digest()
        records: list[ScheduleRoundRecord] = []
        planned_rounds = tuple(getattr(policy, "_planned_rounds", ()) or problem.rounds)
        if {context.round_id for context in planned_rounds} != {
            context.round_id for context in problem.rounds
        }:
            raise SchedulingError("policy planned rounds do not match the scheduling problem")
        for schedule_index, context in enumerate(planned_rounds):
            costs = _assignment_costs(
                context.logical_tiles, resident, problem.constraints, self.cost_engine
            )
            decision = policy.choose_assignment(
                context, resident.copy(), costs, problem.constraints
            )
            self._validate_decision(context, decision, problem.constraints, problem.device_model)
            before = resident.copy()
            round_costs: dict[str, AggregatedTransitionCost] = {}
            for tile in context.logical_tiles:
                physical_id = decision.logical_to_physical_tile[tile.tile_id]
                adjust_tile = getattr(policy, "adjust_tile_for_assignment", None)
                scheduled_tile = (
                    adjust_tile(context, tile, physical_id, resident)
                    if callable(adjust_tile)
                    else tile
                )
                cost = self.cost_engine.aggregate_tile(
                    resident.by_physical_tile.get(
                        physical_id,
                        tuple(problem.device_model.initial_code() for _ in tile.target_state),
                    ),
                    scheduled_tile,
                    source_physical_state=resident.physical_by_physical_tile.get(
                        physical_id, {}
                    ),
                    physical_tile_id=physical_id,
                )
                round_costs[tile.tile_id] = cost
                resident.by_physical_tile[physical_id] = tuple(tile.target_state)
                resident.physical_by_physical_tile[physical_id] = (
                    self.cost_engine.updated_physical_state(
                        resident.physical_by_physical_tile.get(physical_id, {}),
                        scheduled_tile,
                        physical_tile_id=physical_id,
                    )
                )
            resident.assignments_by_round[context.round_id] = dict(
                decision.logical_to_physical_tile
            )
            resident.last_round_index = schedule_index
            records.append(
                ScheduleRoundRecord(
                    round_id=context.round_id,
                    round_index=schedule_index,
                    logical_to_physical_tile=dict(decision.logical_to_physical_tile),
                    logical_tile_slots={
                        key: dict(value) for key, value in decision.logical_tile_slots.items()
                    },
                    target_states={str(key): list(value) for key, value in decision.target_states.items()},
                    resident_state_before={str(key): list(value) for key, value in before.by_physical_tile.items()},
                    resident_state_after={str(key): list(value) for key, value in resident.by_physical_tile.items()},
                    transition_costs=round_costs,
                    policy_metadata=dict(decision.policy_metadata),
                    context_metadata={
                        key: value
                        for key, value in context.metadata.items()
                        if key != "_native_physical_targets"
                    },
                    placement_snapshot=(
                        dict(context.metadata["placement_snapshot"])
                        if isinstance(context.metadata.get("placement_snapshot"), dict)
                        else None
                    ),
                )
            )
        cost_values = [
            cost
            for record in records
            for cost in record.transition_costs.values()
        ]
        return ScheduleArtifact(
            schema_version=1,
            scheduler_id="unified_weight_scheduler",
            policy_id=str(policy.policy_id),
            path_id=_path_id(str(policy.policy_id)),
            model_id=problem.model_id,
            base_round_plan_sha256=problem.base_round_plan_sha256,
            model_ir_sha256=problem.model_ir_sha256,
            weights_sha256=problem.weights_sha256,
            sim_config_sha256=problem.sim_config_sha256,
            device_model_id=str(problem.device_model.model_id),
            device_bits=int(problem.device_model.device_bits),
            device_level=int(problem.device_model.device_level),
            initial_resident_state={},
            rounds=records,
            final_resident_state={str(key): list(value) for key, value in resident.by_physical_tile.items()},
            transition_cost_objective=(
                records[0].transition_costs[next(iter(records[0].transition_costs))].objective_unit
                if records and records[0].transition_costs
                else "changed_cell_count"
            ),
            transition_cost_exact=bool(cost_values) and all(cost.exact for cost in cost_values),
            execution_mode="artifact_only",
            policy_metadata={
                "initial_state_digest": initial_digest,
                **dict(problem.metadata),
                **dict(getattr(policy, "schedule_metadata", {})),
            },
        )

    @staticmethod
    def _validate_decision(
        context: RoundSchedulingContext,
        decision: AssignmentDecision,
        constraints: SchedulingConstraints,
        device_model: DeviceModel,
    ) -> None:
        if decision.round_id != context.round_id:
            raise SchedulingError("policy returned an assignment for the wrong round")
        expected = {tile.tile_id for tile in context.logical_tiles}
        actual = set(decision.logical_to_physical_tile)
        if actual != expected:
            raise SchedulingError(
                f"assignment logical tile ids differ from round payload: {actual ^ expected}"
            )
        physical = list(decision.logical_to_physical_tile.values())
        if len(physical) != len(set(physical)):
            raise SchedulingError("two logical tiles were assigned to one physical tile")
        available = set(constraints.physical_tile_ids)
        if not set(physical).issubset(available):
            raise SchedulingError("assignment uses an unavailable physical tile")
        for tile in context.logical_tiles:
            assigned = decision.target_states.get(decision.logical_to_physical_tile[tile.tile_id])
            if assigned is None or tuple(assigned) != tuple(tile.target_state):
                raise SchedulingError(f"target state mismatch for logical tile {tile.tile_id}")
            for code in tile.target_state:
                device_model.validate_code(int(code))


def _path_id(policy_id: str) -> str:
    aliases = {
        "seq": "sched-seq",
        "sequential": "sched-seq",
        "flashsef_ptr": "sched-flashsef",
        "flashsef-ptr": "sched-flashsef",
        "snn_native_temporal": "sched-snn_native_temporal",
        "snn_flashsef_temporal": "sched-snn_flashsef_temporal",
        "snn-flashsef-temporal": "sched-snn_flashsef_temporal",
    }
    return aliases.get(policy_id, f"sched-{policy_id}")


def generate_weight_schedule(problem: WeightScheduleProblem, policy: SchedulingPolicy):
    return UnifiedWeightScheduler(cost_engine=TransitionCostEngine(problem.device_model)).generate(problem, policy)


__all__ = [
    "AssignmentDecision",
    "LogicalTile",
    "ResidentState",
    "RoundSchedulingContext",
    "SchedulingConstraints",
    "SchedulingError",
    "UnifiedWeightScheduler",
    "WeightScheduleProblem",
    "generate_weight_schedule",
]
