"""Canonical sequential assignment policy."""

from __future__ import annotations

from MNSIM.Scheduling.weight_scheduling.scheduler import (
    AssignmentDecision,
    ResidentState,
    RoundSchedulingContext,
    SchedulingConstraints,
)
from MNSIM.Scheduling.weight_scheduling.transition_cost import TransitionCostView


class SequentialPolicy:
    policy_id = "seq"

    def choose_assignment(
        self,
        context: RoundSchedulingContext,
        resident_state: ResidentState,
        transition_costs: TransitionCostView,
        constraints: SchedulingConstraints,
    ) -> AssignmentDecision:
        del resident_state, transition_costs
        logical_tiles = sorted(context.logical_tiles, key=lambda tile: tile.tile_id)
        physical_tiles = sorted(constraints.physical_tile_ids)
        assignment = {
            tile.tile_id: physical_tiles[index]
            for index, tile in enumerate(logical_tiles)
        }
        targets = {
            assignment[tile.tile_id]: tuple(tile.target_state)
            for tile in logical_tiles
        }
        return AssignmentDecision(
            round_id=context.round_id,
            logical_to_physical_tile=assignment,
            logical_tile_slots={tile.tile_id: dict(tile.local_slots) for tile in logical_tiles},
            target_states=targets,
            policy_metadata={"assignment_method": "canonical_sequential"},
        )


__all__ = ["SequentialPolicy"]
