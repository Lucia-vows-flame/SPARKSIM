"""Artifact-only execution guard for the unified scheduling contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from MNSIM.Scheduling.weight_scheduling.artifact import (
    ScheduleArtifact,
    ScheduleArtifactError,
    load_schedule_artifact,
)


def execute_schedule_artifact(
    artifact_path: str,
    executor: Callable[[ScheduleArtifact], Any],
    *,
    expected: Mapping[str, Any] | None = None,
) -> Any:
    """Validate and pass a frozen artifact to an execution callback.

    The callback receives the loaded artifact only.  No planner or policy is
    available through this entry point, making accidental re-planning explicit
    in the caller's API review.
    """

    artifact = load_schedule_artifact(artifact_path)
    _validate_expected_identity(artifact, expected or {})
    return executor(artifact)


def _validate_expected_identity(
    artifact: ScheduleArtifact, expected: Mapping[str, Any]
) -> None:
    checks = {
        "scheduler_id": artifact.scheduler_id,
        "policy_id": artifact.policy_id,
        "path_id": artifact.path_id,
        "model_id": artifact.model_id,
        "base_round_plan_sha256": artifact.base_round_plan_sha256,
        "model_ir_sha256": artifact.model_ir_sha256,
        "weights_sha256": artifact.weights_sha256,
        "sim_config_sha256": artifact.sim_config_sha256,
        "device_model_id": artifact.device_model_id,
        "device_bits": artifact.device_bits,
        "device_level": artifact.device_level,
    }
    for key, expected_value in expected.items():
        if key not in checks:
            raise ScheduleArtifactError(f"unsupported execution identity field {key!r}")
        if expected_value is not None and checks[key] != expected_value:
            raise ScheduleArtifactError(
                f"schedule artifact identity mismatch for {key}: "
                f"expected {expected_value!r}, got {checks[key]!r}"
            )


__all__ = ["execute_schedule_artifact"]
