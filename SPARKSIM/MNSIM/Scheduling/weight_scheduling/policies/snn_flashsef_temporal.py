"""Temporal critical-path scheduling built on the FlashSeF resident state.

The policy keeps FlashSeF's resident-weight formulation, then evaluates the
same candidate assignments with MNSIM's native physical write primitive.  Its
primary constraint is the minimum native write energy for the round; within
that energy frontier it minimizes the OU/PE write critical path.  No routed
spike, NoC, or bit-hop term is introduced.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from MNSIM.Scheduling.weight_scheduling.policies.snn_native_temporal import (
    SNNNativeTemporalPolicy,
    _minimum_metric_assignment,
)
from MNSIM.Scheduling.weight_scheduling.scheduler import (
    AssignmentDecision,
    LogicalTile,
    ResidentState,
    RoundSchedulingContext,
    SchedulingConstraints,
)
from MNSIM.Scheduling.weight_scheduling.transition_cost import TransitionCostView


class SNNFlashSeFTemporalPolicy(SNNNativeTemporalPolicy):
    """FlashSeF-compatible resident scheduling with temporal write timing."""

    policy_id = "snn_flashsef_temporal"

    def __init__(self, *, tolerance: float = 0.0):
        super().__init__(tolerance=tolerance)
        self._last_selection_audit: list[dict[str, Any]] = []
        self._cost_engine: Any | None = None
        self._slot_permutations: dict[str, dict[str, dict[str, str]]] = {}
        self.schedule_metadata = {
            "policy_family": "flashsef_temporal_critical_path",
            "objective": "min_native_write_latency_at_min_native_write_energy",
            "energy_frontier_tolerance": float(tolerance),
            "bit_hop_objective": False,
            "noc_objective": False,
        }

    def prepare_schedule(self, problem: Any, cost_engine: Any) -> None:
        self._cost_engine = cost_engine
        self._slot_permutations.clear()
        super().prepare_schedule(problem, cost_engine)
        self.schedule_metadata.update(
            {
                "policy_family": "flashsef_temporal_critical_path",
                "objective": "min_native_write_latency_at_min_native_write_energy",
                "energy_frontier_tolerance": self.tolerance,
                "bit_hop_objective": False,
                "noc_objective": False,
                "native_latency_semantics": "OU_max_then_PE_max_then_tile_max",
                "inactive_cell_policy": "retain_outside_fragment_shape",
            }
        )

    def choose_assignment(
        self,
        context: RoundSchedulingContext,
        resident_state: ResidentState,
        transition_costs: TransitionCostView,
        constraints: SchedulingConstraints,
    ) -> AssignmentDecision:
        decision = super().choose_assignment(
            context,
            resident_state,
            transition_costs,
            constraints,
        )
        permutations: dict[str, dict[str, str]] = {}
        for tile in context.logical_tiles:
            physical_id = decision.logical_to_physical_tile[tile.tile_id]
            mapping = self._optimize_fragment_slots(
                tile,
                resident_state.physical_by_physical_tile.get(physical_id, {}),
            )
            if mapping:
                permutations[tile.tile_id] = mapping
        self._slot_permutations[context.round_id] = permutations
        metadata = {
            **decision.policy_metadata,
            "slot_objective": "native_write_latency_at_equal_or_lower_energy",
            "slot_permutations": permutations,
        }
        return replace(decision, policy_metadata=metadata)

    def adjust_tile_for_assignment(
        self,
        context: RoundSchedulingContext,
        tile: LogicalTile,
        physical_tile_id: int,
        resident_state: ResidentState,
    ) -> LogicalTile:
        del physical_tile_id, resident_state
        mapping = self._slot_permutations.get(context.round_id, {}).get(tile.tile_id)
        if not mapping:
            return tile
        metadata = dict(tile.metadata)
        raw_targets = metadata.get("_native_physical_targets")
        if not isinstance(raw_targets, (list, tuple)):
            return tile
        targets = []
        for target in raw_targets:
            value = dict(target)
            parts = str(value["slot"]).split(":")
            if len(parts) not in {3, 4}:
                return tile
            base = ":".join(parts[-3:-1]) if len(parts) == 4 else ":".join(parts[:2])
            destination = mapping.get(base)
            if destination is None:
                targets.append(value)
                continue
            polarity = parts[-1]
            value["slot"] = f"{destination}:{polarity}"
            targets.append(value)
        metadata["_native_physical_targets"] = targets
        return replace(tile, metadata=metadata)

    def _optimize_fragment_slots(
        self,
        tile: LogicalTile,
        source_state: dict[str, dict[str, Any]],
    ) -> dict[str, str]:
        if self._cost_engine is None:
            return {}
        raw_targets = tile.metadata.get("_native_physical_targets")
        if not isinstance(raw_targets, (list, tuple)) or not raw_targets:
            return {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for raw in raw_targets:
            target = dict(raw)
            parts = str(target["slot"]).split(":")
            if len(parts) == 4:
                base = ":".join(parts[1:3])
            elif len(parts) == 3:
                base = ":".join(parts[:2])
            else:
                return {}
            grouped.setdefault(base, []).append(target)
        slots = sorted(grouped)
        if len(slots) < 2:
            return {}

        def score(mapping: dict[str, str]) -> tuple[float, float]:
            targets = []
            for source_slot, destination_slot in mapping.items():
                for target in grouped[source_slot]:
                    parts = str(target["slot"]).split(":")
                    polarity = parts[-1]
                    targets.append(
                        {
                            "slot": f"{destination_slot}:{polarity}",
                            "shape": target["shape"],
                            "values": target["values"],
                        }
                    )
            cost = self._cost_engine._aggregate_physical_targets(targets, source_state)
            return float(cost.energy_j or 0.0), float(cost.latency_s or 0.0)

        best = {slot: slot for slot in slots}
        best_energy, best_latency = score(best)
        changed = True
        while changed:
            changed = False
            for left_index, left in enumerate(slots):
                for right in slots[left_index + 1 :]:
                    candidate = dict(best)
                    candidate[left], candidate[right] = best[right], best[left]
                    energy, latency = score(candidate)
                    improves = (
                        energy <= best_energy + 1e-18
                        and latency < best_latency - 1e-15
                    ) or (
                        energy < best_energy - 1e-18
                        and latency <= best_latency + 1e-15
                    )
                    if improves:
                        best, best_energy, best_latency = candidate, energy, latency
                        changed = True
            
        return {
            source: destination
            for source, destination in best.items()
            if source != destination
        }

    def assignment_candidates(
        self,
        context: RoundSchedulingContext,
        transition_costs: TransitionCostView,
        physical_ids: list[int],
        *,
        force_sequential: bool = False,
    ) -> list[tuple[str, dict[str, int]]]:
        candidates = super().assignment_candidates(
            context,
            transition_costs,
            physical_ids,
            force_sequential=force_sequential,
        )
        if force_sequential:
            return candidates

        logical_ids = [
            tile.tile_id for tile in sorted(context.logical_tiles, key=lambda tile: tile.tile_id)
        ]
        energy_values = [
            float(transition_costs.cost(logical_id, physical_id).energy_j or 0.0)
            for logical_id in logical_ids
            for physical_id in physical_ids
        ]
        latency_values = [
            float(transition_costs.cost(logical_id, physical_id).latency_s or 0.0)
            for logical_id in logical_ids
            for physical_id in physical_ids
        ]
        energy_scale = max(max(energy_values, default=0.0), 1e-30)
        latency_scale = max(max(latency_values, default=0.0), 1e-30)
        # Scalarized assignments add useful Pareto points without replacing the
        # exact energy and exact latency Hungarian candidates.
        for weight in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 20.0):
            assignment = _minimum_weighted_assignment(
                logical_ids,
                physical_ids,
                transition_costs,
                energy_scale=energy_scale,
                latency_scale=latency_scale,
                latency_weight=weight,
            )
            candidates.append((f"native_pareto_{weight:g}", assignment))
        return _deduplicate_candidates(candidates)

    def select_assignment_candidate(
        self,
        evaluated: list[tuple[str, dict[str, int], tuple[float, float, int]]],
        logical_ids: list[str],
        *,
        force_sequential: bool = False,
        reference_energy: float | None = None,
    ) -> tuple[str, dict[str, int], tuple[float, float, int]]:
        if force_sequential:
            return next(item for item in evaluated if item[0] == "sequential")

        # FlashSeF's energy-minimizing assignment defines the frontier.  A
        # candidate may move along that frontier only when it does not spend
        # additional modeled write energy.
        baseline = next(item for item in evaluated if item[0] == "sequential")
        # FlashSeF's compact objective is an incumbent reference, while the
        # native fragment replay is the authoritative physical write model.
        # Keep every candidate within the sequential physical energy budget;
        # this admits a lower-latency Pareto point that remains below the
        # incumbent after Phase 2 replay even when compact and fragment views
        # differ by padding cells.
        frontier_energy = max(
            float(baseline[2][0]),
            float(reference_energy) if reference_energy is not None else 0.0,
        )
        candidates = [
            item
            for item in evaluated
            if item[2][0] <= frontier_energy * (1.0 + self.tolerance)
            and item[2][0] <= baseline[2][0] * (1.0 + self.tolerance)
            and item[2][1] <= baseline[2][1] * (1.0 + self.tolerance)
        ]
        if not candidates:
            candidates = [baseline]
        self._last_selection_audit = [
            {
                "method": method,
                "energy_j": float(metrics[0]),
                "latency_s": float(metrics[1]),
                "changed_cells": int(metrics[2]),
                "assignment": dict(sorted((str(k), int(v)) for k, v in assignment.items())),
            }
            for method, assignment, metrics in sorted(
                evaluated, key=lambda item: (item[2][0], item[2][1], item[0])
            )
        ]
        return min(
            candidates,
            key=lambda item: (
                item[2][1],
                item[2][0],
                item[2][2],
                tuple(item[1][logical_id] for logical_id in logical_ids),
            ),
        )


def _minimum_weighted_assignment(
    logical_ids: list[str],
    physical_ids: list[int],
    costs: TransitionCostView,
    *,
    energy_scale: float,
    latency_scale: float,
    latency_weight: float,
) -> dict[str, int]:
    matrix = [
        [
            float(costs.cost(logical_id, physical_id).energy_j or 0.0) / energy_scale
            + latency_weight
            * float(costs.cost(logical_id, physical_id).latency_s or 0.0)
            / latency_scale
            for physical_id in physical_ids
        ]
        for logical_id in logical_ids
    ]
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _small_weighted_assignment(logical_ids, physical_ids, matrix)
    rows, columns = linear_sum_assignment(matrix)
    if len(rows) != len(logical_ids):
        raise ValueError("weighted assignment solver did not assign every logical tile")
    return {
        logical_ids[int(row)]: physical_ids[int(column)]
        for row, column in zip(rows, columns)
    }


def _small_weighted_assignment(
    logical_ids: list[str], physical_ids: list[int], matrix: list[list[float]]
) -> dict[str, int]:
    import itertools

    best: tuple[float, tuple[int, ...]] | None = None
    for selected in itertools.permutations(physical_ids, len(logical_ids)):
        score = sum(
            matrix[index][physical_ids.index(tile)]
            for index, tile in enumerate(selected)
        )
        candidate = (float(score), tuple(selected))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no feasible weighted assignment")
    return dict(zip(logical_ids, best[1]))


def _deduplicate_candidates(
    candidates: list[tuple[str, dict[str, int]]],
) -> list[tuple[str, dict[str, int]]]:
    seen: set[tuple[tuple[str, int], ...]] = set()
    result: list[tuple[str, dict[str, int]]] = []
    for method, assignment in candidates:
        key = tuple(sorted((str(logical), int(physical)) for logical, physical in assignment.items()))
        if key in seen:
            continue
        seen.add(key)
        result.append((method, assignment))
    return result


__all__ = ["SNNFlashSeFTemporalPolicy"]
