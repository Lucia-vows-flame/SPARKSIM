"""SNN temporal scheduling against MNSIM-native write costs.

This policy deliberately contains no routed-spike or bit-hop term.  It chooses
resident tile assignments using the physical per-cell write energy and pulse
latency tables derived from the active SimConfig.  The canonical sequential
assignment is always a candidate, so the local decision cannot regress both
native write metrics at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

from MNSIM.Scheduling.weight_scheduling.scheduler import (
    AssignmentDecision,
    LogicalTile,
    ResidentState,
    RoundSchedulingContext,
    SchedulingConstraints,
    SchedulingError,
    WeightScheduleProblem,
)
from MNSIM.Scheduling.weight_scheduling.transition_cost import (
    TransitionCostEngine,
    TransitionCostView,
)


class SNNNativeTemporalPolicy:
    """Resident-aware SNN policy using only native MNSIM write costs.

    The first implementation keeps the canonical execution order.  Its SNN
    contract is temporal: a weighted resident state is selected once and then
    reused by every timestep in the fixed SNN sequence.  Legal round-order
    search is enabled only by a later plan-level dependency contract.
    """

    policy_id = "snn_native_temporal"

    def __init__(self, *, tolerance: float = 0.0):
        if tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")
        self.tolerance = float(tolerance)
        self._planned_assignments: dict[str, dict[str, int]] = {}
        self._planned_metadata: dict[str, dict[str, object]] = {}
        self._planned_rounds: tuple[RoundSchedulingContext, ...] = ()
        self.schedule_metadata: dict[str, object] = {}

    def prepare_schedule(
        self,
        problem: WeightScheduleProblem,
        cost_engine: TransitionCostEngine,
    ) -> None:
        if not problem.rounds:
            raise SchedulingError("native temporal scheduling requires at least one round")
        if str(problem.metadata.get("backend_kind", "")).lower() not in {
            "",
            "snn",
        }:
            raise SchedulingError("snn_native_temporal requires an SNN scheduling problem")

        self._planned_assignments.clear()
        self._planned_metadata.clear()
        resident = ResidentState()
        planned_rounds: list[RoundSchedulingContext] = []
        total_energy = total_latency = baseline_energy = baseline_latency = 0.0
        used_native_cost = False
        order_changed = False

        for block in _reorder_blocks(problem.rounds):
            original_order = tuple(block)
            baseline = _simulate_order(
                original_order,
                resident,
                problem,
                cost_engine,
                policy=self,
                force_sequential=True,
            )
            native_original = _simulate_order(
                original_order,
                resident,
                problem,
                cost_engine,
                policy=self,
            )
            selected = (
                native_original
                if _dominates(native_original, baseline, self.tolerance)
                else baseline
            )
            for candidate_order in _candidate_orders(original_order):
                if tuple(context.round_id for context in candidate_order) == tuple(
                    context.round_id for context in original_order
                ):
                    continue
                candidate = _simulate_order(
                    candidate_order,
                    resident,
                    problem,
                    cost_engine,
                    policy=self,
                )
                if _dominates(candidate, baseline, self.tolerance) and _result_key(candidate) < _result_key(selected):
                    selected = candidate
            order_changed |= tuple(context.round_id for context in selected.contexts) != tuple(
                context.round_id for context in original_order
            )
            planned_rounds.extend(selected.contexts)
            resident = selected.resident
            total_energy += selected.energy
            total_latency += selected.latency
            baseline_energy += baseline.energy
            baseline_latency += baseline.latency
            used_native_cost |= selected.used_native
            for round_id, assignment in selected.assignments.items():
                self._planned_assignments[round_id] = dict(assignment)
                self._planned_metadata[round_id] = dict(selected.metadata[round_id])

        self._planned_rounds = tuple(planned_rounds)

        self.schedule_metadata = {
            "native_objective": "mnsim_physical_write_energy_then_cell_pulse_latency",
            "native_cost_source": "SimConfig Device level write tables",
            "native_cost_exact_energy": True,
            "native_cost_scope": "active_physical_fragment_slots_with_idle_residency",
            "native_cost_latency_scope": "per-tile OU max then PE max; Phase 2 remains authoritative for round/tile concurrency",
            "baseline_policy": "sequential_assignment_is_always_a_candidate",
            "baseline_tolerance": self.tolerance,
            "round_order": [context.round_id for context in self._planned_rounds],
            "round_order_optimization": "topological_block_search",
            "round_order_changed": order_changed,
            "placement_dependent_candidate_selected": used_native_cost,
            "native_write_energy_j_total": total_energy,
            "native_write_latency_tile_round_max_s_total": total_latency,
            "sequential_write_energy_j_total": baseline_energy,
            "sequential_write_latency_tile_round_max_s_total": baseline_latency,
        }

    def choose_assignment(
        self,
        context: RoundSchedulingContext,
        resident_state: ResidentState,
        transition_costs: TransitionCostView,
        constraints: SchedulingConstraints,
    ) -> AssignmentDecision:
        del resident_state
        assignment = self._planned_assignments.get(context.round_id)
        if assignment is None:
            assignment = _minimum_metric_assignment(
                [tile.tile_id for tile in sorted(context.logical_tiles, key=lambda tile: tile.tile_id)],
                sorted(constraints.physical_tile_ids),
                transition_costs,
                "energy_j",
            )
        targets = {
            assignment[tile.tile_id]: tuple(tile.target_state)
            for tile in context.logical_tiles
        }
        return AssignmentDecision(
            round_id=context.round_id,
            logical_to_physical_tile=dict(assignment),
            logical_tile_slots={tile.tile_id: dict(tile.local_slots) for tile in context.logical_tiles},
            target_states=targets,
            policy_metadata=dict(self._planned_metadata.get(context.round_id, {})),
        )

    def assignment_candidates(
        self,
        context: RoundSchedulingContext,
        transition_costs: TransitionCostView,
        physical_ids: list[int],
        *,
        force_sequential: bool = False,
    ) -> list[tuple[str, dict[str, int]]]:
        """Return legal assignment candidates for one round.

        Keeping candidate generation as a hook lets a stricter temporal policy
        reuse the resident-state simulation without duplicating its round-order
        and dependency handling.
        """

        logical_ids = [
            tile.tile_id
            for tile in sorted(context.logical_tiles, key=lambda tile: tile.tile_id)
        ]
        sequential = {
            logical_id: physical_ids[index]
            for index, logical_id in enumerate(logical_ids)
        }
        evaluated = [("sequential", sequential)]
        if not force_sequential:
            evaluated.extend(
                [
                    (
                        "native_energy_hungarian",
                        _minimum_metric_assignment(
                            logical_ids, physical_ids, transition_costs, "energy_j"
                        ),
                    ),
                    (
                        "native_latency_hungarian",
                        _minimum_metric_assignment(
                            logical_ids, physical_ids, transition_costs, "latency_s"
                        ),
                    ),
                ]
            )
        return evaluated

    def adjust_tile_for_assignment(
        self,
        context: RoundSchedulingContext,
        tile: LogicalTile,
        physical_tile_id: int,
        resident_state: ResidentState,
    ) -> LogicalTile:
        """Return the physical target payload for a selected assignment."""

        del context, physical_tile_id, resident_state
        return tile

    def select_assignment_candidate(
        self,
        evaluated: list[tuple[str, dict[str, int], tuple[float, float, int]]],
        logical_ids: list[str],
        *,
        force_sequential: bool = False,
        reference_energy: float | None = None,
    ) -> tuple[str, dict[str, int], tuple[float, float, int]]:
        baseline = next(item for item in evaluated if item[0] == "sequential")
        baseline_metrics = baseline[2]
        candidates = [
            item
            for item in evaluated
            if force_sequential
            or (
                item[2][0] <= baseline_metrics[0] * (1.0 + self.tolerance)
                and item[2][1] <= baseline_metrics[1] * (1.0 + self.tolerance)
            )
        ]
        return min(
            candidates,
            key=lambda item: (
                item[2][0],
                item[2][1],
                item[2][2],
                tuple(item[1][logical_id] for logical_id in logical_ids),
            ),
        )


@dataclass(frozen=True)
class _SimulationResult:
    contexts: tuple[RoundSchedulingContext, ...]
    assignments: dict[str, dict[str, int]]
    metadata: dict[str, dict[str, object]]
    resident: ResidentState
    energy: float
    latency: float
    changed: int
    used_native: bool


def _reorder_blocks(
    contexts: tuple[RoundSchedulingContext, ...],
) -> list[tuple[RoundSchedulingContext, ...]]:
    """Split only explicitly declared contiguous reorderable round blocks."""

    blocks: list[tuple[RoundSchedulingContext, ...]] = []
    current: list[RoundSchedulingContext] = []
    current_key: object = None
    for context in contexts:
        key = context.metadata.get("reorder_block")
        if key is None or key == "fixed":
            if current:
                blocks.append(tuple(current))
                current = []
                current_key = None
            blocks.append((context,))
            continue
        if current and key != current_key:
            blocks.append(tuple(current))
            current = []
        current_key = key
        current.append(context)
    if current:
        blocks.append(tuple(current))
    return blocks


def _candidate_orders(
    contexts: tuple[RoundSchedulingContext, ...],
) -> list[tuple[RoundSchedulingContext, ...]]:
    if len(contexts) <= 1:
        return [contexts]
    by_id = {context.round_id: context for context in contexts}
    external_dependencies = {
        str(value)
        for context in contexts
        for value in context.metadata.get("predecessor_round_ids", [])
        if str(value) not in by_id
    }
    # Dependencies outside the declared block are not movable by this local
    # search; retaining the original order is the only safe option.
    if external_dependencies:
        return [contexts]
    dependencies = {
        context.round_id: {
            str(value)
            for value in context.metadata.get("predecessor_round_ids", [])
            if str(value) in by_id
        }
        for context in contexts
    }

    def legal(order: tuple[RoundSchedulingContext, ...]) -> bool:
        seen: set[str] = set()
        for context in order:
            if not dependencies[context.round_id].issubset(seen):
                return False
            seen.add(context.round_id)
        return True

    if len(contexts) <= 7:
        return [
            tuple(by_id[round_id] for round_id in selected)
            for selected in permutations(by_id)
            if legal(tuple(by_id[round_id] for round_id in selected))
        ]
    return [contexts]


def _simulate_order(
    contexts: tuple[RoundSchedulingContext, ...],
    initial_resident: ResidentState,
    problem: WeightScheduleProblem,
    cost_engine: TransitionCostEngine,
    *,
    policy: SNNNativeTemporalPolicy,
    force_sequential: bool = False,
) -> _SimulationResult:
    resident = initial_resident.copy()
    assignments: dict[str, dict[str, int]] = {}
    metadata: dict[str, dict[str, object]] = {}
    total_energy = total_latency = 0.0
    total_changed = 0
    used_native = False
    physical_ids = sorted(problem.constraints.physical_tile_ids)

    for context in contexts:
        costs = cost_engine.matrix(
            resident.by_physical_tile,
            {tile.tile_id: tile.target_state for tile in context.logical_tiles},
            physical_ids,
            target_tiles=context.logical_tiles,
            source_physical_state_by_tile=resident.physical_by_physical_tile,
        )
        logical_ids = [
            tile.tile_id
            for tile in sorted(context.logical_tiles, key=lambda tile: tile.tile_id)
        ]
        flashsef_reference_energy = _flashsef_reference_energy(
            logical_ids,
            physical_ids,
            context,
            resident,
            cost_engine,
            problem,
        )
        evaluated = policy.assignment_candidates(
            context,
            costs,
            physical_ids,
            force_sequential=force_sequential,
        )
        evaluated_with_metrics = [
            (method, assignment, _assignment_metrics(assignment, costs))
            for method, assignment in evaluated
        ]
        baseline = next(item for item in evaluated_with_metrics if item[0] == "sequential")
        _, _, baseline_metrics = baseline
        method, assignment, metrics = policy.select_assignment_candidate(
            evaluated_with_metrics,
            logical_ids,
            force_sequential=force_sequential,
            reference_energy=flashsef_reference_energy,
        )
        assignments[context.round_id] = dict(assignment)
        metadata[context.round_id] = {
            "assignment_method": method,
            "objective": "native_mnsim_write_energy_then_latency",
            "native_write_energy_j": metrics[0],
            "native_write_latency_cell_sum_s": metrics[1],
            "changed_cells": metrics[2],
            "sequential_write_energy_j": baseline_metrics[0],
            "sequential_write_latency_cell_sum_s": baseline_metrics[1],
            "baseline_preserved": True,
            "flashsef_reference_energy_j": flashsef_reference_energy,
        }
        audit = getattr(policy, "_last_selection_audit", None)
        if audit:
            metadata[context.round_id]["candidate_audit"] = list(audit)
        total_energy += metrics[0]
        total_latency += metrics[1]
        total_changed += metrics[2]
        used_native |= method != "sequential"
        for tile in context.logical_tiles:
            resident.by_physical_tile[assignment[tile.tile_id]] = tuple(tile.target_state)
            physical_id = assignment[tile.tile_id]
            adjusted_tile = policy.adjust_tile_for_assignment(
                context,
                tile,
                physical_id,
                resident,
            )
            resident.physical_by_physical_tile[physical_id] = (
                cost_engine.updated_physical_state(
                    resident.physical_by_physical_tile.get(physical_id, {}),
                    adjusted_tile,
                    physical_tile_id=physical_id,
                )
            )
        resident.assignments_by_round[context.round_id] = dict(assignment)
        resident.last_round_index = context.round_index

    return _SimulationResult(
        contexts=contexts,
        assignments=assignments,
        metadata=metadata,
        resident=resident,
        energy=total_energy,
        latency=total_latency,
        changed=total_changed,
        used_native=used_native,
    )


def _dominates(
    candidate: _SimulationResult,
    baseline: _SimulationResult,
    tolerance: float,
) -> bool:
    return (
        candidate.energy <= baseline.energy * (1.0 + tolerance)
        and candidate.latency <= baseline.latency * (1.0 + tolerance)
    )


def _result_key(result: _SimulationResult) -> tuple[float, float, int, tuple[str, ...]]:
    return (
        result.energy,
        result.latency,
        result.changed,
        tuple(context.round_id for context in result.contexts),
    )


def _minimum_metric_assignment(
    logical_ids: list[str],
    physical_ids: list[int],
    costs: TransitionCostView,
    metric: str,
) -> dict[str, int]:
    if len(logical_ids) > len(physical_ids):
        raise SchedulingError("more logical tiles than physical tiles")
    matrix = []
    for logical_id in logical_ids:
        row = []
        for physical_id in physical_ids:
            value = getattr(costs.cost(logical_id, physical_id), metric)
            if value is None:
                raise SchedulingError(
                    f"native temporal policy requires transition metric {metric}"
                )
            row.append(float(value))
        matrix.append(row)
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _small_exact_assignment(logical_ids, physical_ids, matrix)
    rows, columns = linear_sum_assignment(matrix)
    if len(rows) != len(logical_ids):
        raise SchedulingError("native assignment solver did not assign every logical tile")
    return {
        logical_ids[int(row)]: physical_ids[int(column)]
        for row, column in zip(rows, columns)
    }


def _small_exact_assignment(
    logical_ids: list[str], physical_ids: list[int], matrix: list[list[float]]
) -> dict[str, int]:
    import itertools

    best: tuple[float, tuple[int, ...]] | None = None
    for selected in itertools.permutations(physical_ids, len(logical_ids)):
        score = sum(matrix[index][physical_ids.index(tile)] for index, tile in enumerate(selected))
        candidate = (float(score), tuple(selected))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise SchedulingError("no feasible native assignment")
    return dict(zip(logical_ids, best[1]))


def _assignment_metrics(
    assignment: dict[str, int], costs: TransitionCostView
) -> tuple[float, float, int]:
    selected = [costs.cost(logical_id, physical_id) for logical_id, physical_id in assignment.items()]
    physical_latency = any(
        bool(cost.provenance.get("physical_fragment_slot_exact")) for cost in selected
    )
    latency_values = [float(cost.latency_s or 0.0) for cost in selected]
    return (
        sum(float(cost.energy_j or 0.0) for cost in selected),
        max(latency_values, default=0.0)
        if physical_latency
        else sum(latency_values),
        sum(int(cost.changed_cells) for cost in selected),
    )


def _flashsef_reference_energy(
    logical_ids: list[str],
    physical_ids: list[int],
    context: RoundSchedulingContext,
    resident: ResidentState,
    cost_engine: TransitionCostEngine,
    problem: WeightScheduleProblem,
) -> float | None:
    """Evaluate the incumbent FlashSeF compact transition objective.

    FlashSeF scores compact logical tile payloads.  Native SNN policies score
    exact fragment slots, so the two views must be computed separately before
    comparing a candidate's energy frontier.
    """

    compact_by_logical: dict[str, dict[int, object]] = {}
    for tile in context.logical_tiles:
        row: dict[int, object] = {}
        for physical_id in physical_ids:
            source = resident.by_physical_tile.get(int(physical_id))
            if source is None:
                source = tuple(
                    problem.device_model.initial_code() for _ in tile.target_state
                )
            row[int(physical_id)] = cost_engine.aggregate(source, tile.target_state)
        compact_by_logical[str(tile.tile_id)] = row
    compact_view = TransitionCostView(compact_by_logical)
    try:
        assignment = _minimum_metric_assignment(
            logical_ids,
            physical_ids,
            compact_view,
            "objective_cost",
        )
    except (AttributeError, SchedulingError):
        return None
    values = [
        compact_view.cost(logical_id, physical_id).energy_j
        for logical_id, physical_id in assignment.items()
    ]
    if any(value is None for value in values):
        return None
    return float(sum(float(value) for value in values if value is not None))


__all__ = ["SNNNativeTemporalPolicy"]
