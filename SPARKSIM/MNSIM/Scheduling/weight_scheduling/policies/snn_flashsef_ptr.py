"""SNN temporal resident weight scheduling using native MNSIM write costs.

``snn_flashsef_ptr`` is kept as the public policy name for compatibility with
existing schedule artifacts and command lines. The policy deliberately does
not use routed-spike, NoC, or bit-hop estimates. It schedules the resident
weight state of an SNN against the physical source-to-target write table used
by Phase 2, while preserving the sequential assignment as a non-regression
candidate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from MNSIM.Scheduling.weight_scheduling.policies.snn_native_temporal import (
    SNNNativeTemporalPolicy,
)
from MNSIM.Scheduling.weight_scheduling.scheduler import (
    AssignmentDecision,
    ResidentState,
    RoundSchedulingContext,
    SchedulingConstraints,
    WeightScheduleProblem,
)
from MNSIM.Scheduling.weight_scheduling.transition_cost import (
    TransitionCostEngine,
    TransitionCostView,
)


class SNNFlashSeFPTRPolicy(SNNNativeTemporalPolicy):
    """Compatibility-named SNN policy with native physical objectives.

    The old implementation optimized changed-cell count plus a routed spike
    bit-hop proxy. That proxy is outside MNSIM's modeled energy and latency
    paths. The implementation now uses the same exact native transition
    primitive as :class:`SNNNativeTemporalPolicy`.

    ``tile_coordinates`` and activity-profile provenance are accepted for
    compatibility with existing callers, but are intentionally not used as a
    scheduling objective.
    """

    policy_id = "snn_flashsef_ptr"

    def __init__(
        self,
        tile_coordinates: Mapping[int, tuple[int, int]] | None = None,
        *,
        activity_profile_id: str | None = None,
        activity_profile_sha256: str | None = None,
        tolerance: float = 0.0,
    ) -> None:
        del tile_coordinates
        super().__init__(tolerance=tolerance)
        self.activity_profile_id = activity_profile_id
        self.activity_profile_sha256 = activity_profile_sha256

    def prepare_schedule(
        self,
        problem: WeightScheduleProblem,
        cost_engine: TransitionCostEngine,
    ) -> None:
        super().prepare_schedule(problem, cost_engine)
        self.schedule_metadata = {
            **self.schedule_metadata,
            "snn_policy_family": "temporal_resident_state",
            "snn_objective": "mnsim_physical_write_energy_then_cell_pulse_latency",
            "snn_bit_hop_objective": False,
            "snn_activity_profile_objective": False,
        }
        if self.activity_profile_id or self.activity_profile_sha256:
            self.schedule_metadata["deprecated_activity_profile_ignored"] = True

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
        metadata = {
            **decision.policy_metadata,
            "snn_policy_family": "temporal_resident_state",
            "snn_objective": "mnsim_physical_write_energy_then_cell_pulse_latency",
            "snn_bit_hop_objective": False,
            "snn_activity_profile_objective": False,
        }
        if self.activity_profile_id or self.activity_profile_sha256:
            metadata["deprecated_activity_profile_ignored"] = True
        return replace(decision, policy_metadata=metadata)


__all__ = ["SNNFlashSeFPTRPolicy"]
