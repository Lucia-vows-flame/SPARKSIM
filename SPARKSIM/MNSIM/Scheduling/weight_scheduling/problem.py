"""Adapters for binding a base RoundPlan to the unified scheduler."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from MNSIM.Scheduling.weight_scheduling.device import DeviceModel
from MNSIM.Scheduling.weight_scheduling.scheduler import (
    LogicalTile,
    RoundSchedulingContext,
    SchedulingConstraints,
    WeightScheduleProblem,
)


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible snapshot with stable key ordering."""

    if hasattr(value, "to_dict"):
        value = value.to_dict()
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_problem_from_round_plan(
    base_round_plan: Any,
    *,
    logical_rounds: Sequence[RoundSchedulingContext],
    constraints: SchedulingConstraints,
    device_model: DeviceModel,
    model_ir: Any | None = None,
    weights_snapshot: Any | None = None,
    sim_config_sha256: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> WeightScheduleProblem:
    """Bind scheduler-owned payloads to an immutable Phase-1 plan identity.

    The adapter deliberately does not inspect or mutate MNSIM's planner
    internals.  The caller supplies logical tile payloads extracted from the
    normal round substrate; this keeps the scheduler independent from a
    particular frontend or checkpoint representation.
    """

    return WeightScheduleProblem(
        rounds=tuple(logical_rounds),
        constraints=constraints,
        device_model=device_model,
        model_id=str(getattr(base_round_plan, "model_id", "unknown")),
        base_round_plan_sha256=canonical_sha256(base_round_plan),
        model_ir_sha256=canonical_sha256(model_ir) if model_ir is not None else "",
        weights_sha256=canonical_sha256(weights_snapshot) if weights_snapshot is not None else "",
        sim_config_sha256=str(sim_config_sha256),
        metadata=dict(metadata or {}),
    )


__all__ = ["build_problem_from_round_plan", "canonical_sha256"]
