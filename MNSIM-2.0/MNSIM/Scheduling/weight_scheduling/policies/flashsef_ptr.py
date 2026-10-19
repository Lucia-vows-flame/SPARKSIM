"""FlashSeF-PTR assignment policy using the shared transition cost view."""

from __future__ import annotations

from MNSIM.Scheduling.weight_scheduling.scheduler import (
    AssignmentDecision,
    ResidentState,
    RoundSchedulingContext,
    SchedulingConstraints,
)
from MNSIM.Scheduling.weight_scheduling.transition_cost import TransitionCostView


class FlashSeFPTRPolicy:
    policy_id = "flashsef_ptr"

    def choose_assignment(
        self,
        context: RoundSchedulingContext,
        resident_state: ResidentState,
        transition_costs: TransitionCostView,
        constraints: SchedulingConstraints,
    ) -> AssignmentDecision:
        del resident_state
        logical_tiles = sorted(context.logical_tiles, key=lambda tile: tile.tile_id)
        physical_tiles = sorted(constraints.physical_tile_ids)
        assignment = _minimum_cost_assignment(
            [tile.tile_id for tile in logical_tiles], physical_tiles, transition_costs
        )
        targets = {
            assignment[tile.tile_id]: tuple(tile.target_state)
            for tile in logical_tiles
        }
        return AssignmentDecision(
            round_id=context.round_id,
            logical_to_physical_tile=assignment,
            logical_tile_slots={tile.tile_id: dict(tile.local_slots) for tile in logical_tiles},
            target_states=targets,
            policy_metadata={
                "assignment_method": "hungarian",
                "objective": "transition_cost",
            },
        )


def _minimum_cost_assignment(
    logical_ids: list[str], physical_ids: list[int], costs: TransitionCostView
) -> dict[str, int]:
    if len(logical_ids) > len(physical_ids):
        raise ValueError("more logical tiles than physical tiles")
    # Hungarian is used when SciPy is available; the deterministic exhaustive
    # fallback keeps small unit tests independent of optional packages.
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return _small_exact_assignment(logical_ids, physical_ids, costs)
    matrix = [
        [costs.cost(logical_id, physical_id).objective_cost for physical_id in physical_ids]
        for logical_id in logical_ids
    ]
    rows, columns = linear_sum_assignment(matrix)
    if len(rows) != len(logical_ids):
        raise ValueError("assignment solver did not assign every logical tile")
    return {
        logical_ids[int(row)]: physical_ids[int(column)]
        for row, column in zip(rows, columns)
    }


def _small_exact_assignment(
    logical_ids: list[str], physical_ids: list[int], costs: TransitionCostView
) -> dict[str, int]:
    import itertools

    best: tuple[float, tuple[int, ...]] | None = None
    for selected in itertools.permutations(physical_ids, len(logical_ids)):
        score = sum(
            costs.cost(logical_id, physical_id).objective_cost
            for logical_id, physical_id in zip(logical_ids, selected)
        )
        candidate = (float(score), tuple(selected))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no feasible assignment")
    return dict(zip(logical_ids, best[1]))


__all__ = ["FlashSeFPTRPolicy"]
