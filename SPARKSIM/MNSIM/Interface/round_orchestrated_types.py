"""Public request, result, and status records for round-orchestrated runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from MNSIM.Scheduling.phase2.types import Phase2Report
from MNSIM.Scheduling.round_types import RoundPlan


SCHEMA_VERSION = 1


class RunnerError(Exception):
    """A stable error reported by the round-orchestrated runner."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def _json_safe(value: Any) -> Any:
    """Convert contract values into deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise TypeError("non-finite float is not JSON-safe")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("unsupported JSON mapping key")
        return {
            key: _json_safe(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=repr)

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    if is_dataclass(value):
        return {
            dataclass_field.name: _json_safe(getattr(value, dataclass_field.name))
            for dataclass_field in fields(value)
        }
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


@dataclass(frozen=True)
class ErrorRecord:
    code: str
    detail: str
    stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "code": self.code,
                "detail": self.detail,
                "stage": self.stage,
            }
        )


@dataclass
class StageStatus:
    name: str
    status: str
    error: ErrorRecord | None = None
    summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "name": self.name,
                "status": self.status,
                "error": self.error,
                "summary": self.summary,
            }
        )


@dataclass(frozen=True)
class RoundOrchestratedRunRequest:
    network: str
    weights: Path
    num_classes: int
    sim_config: Path
    output_dir: Path
    dataset_module: str | None = None
    run_hardware: bool = False
    skip_accuracy: bool = False
    run_original: bool = True
    run_pim: bool = True
    overwrite: bool = False
    enable_saf: bool = True
    enable_variation: bool = False
    enable_r_ratio: bool = False
    accuracy_seed: int | None = None
    fixed_quantization_range: bool = False
    device: str = "0"
    disable_inner_pipeline: bool = False
    max_samples: int | None = None
    schedule_only: bool = False
    schedule_plan_out: Path | None = None
    infer_with_schedule: bool = False
    schedule_plan_in: Path | None = None
    custom_schedule_in: Path | None = None
    schedule_policy: str = "seq"
    snn_activity_profile: Path | None = None

    def requested_stages(self) -> tuple[str, ...]:
        if self.skip_accuracy:
            return ("hardware",) if self.run_hardware else ()
        return ("accuracy", "hardware") if self.run_hardware else ("accuracy",)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "network": self.network,
                "weights": self.weights,
                "num_classes": self.num_classes,
                "sim_config": self.sim_config,
                "output_dir": self.output_dir,
                "dataset_module": self.dataset_module,
                "run_hardware": self.run_hardware,
                "skip_accuracy": self.skip_accuracy,
                "run_original": self.run_original,
                "run_pim": self.run_pim,
                "overwrite": self.overwrite,
                "enable_saf": self.enable_saf,
                "enable_variation": self.enable_variation,
                "enable_r_ratio": self.enable_r_ratio,
                "accuracy_seed": self.accuracy_seed,
                "fixed_quantization_range": self.fixed_quantization_range,
                "device": self.device,
                "disable_inner_pipeline": self.disable_inner_pipeline,
                "max_samples": self.max_samples,
                "schedule_only": self.schedule_only,
                "schedule_plan_out": self.schedule_plan_out,
                "infer_with_schedule": self.infer_with_schedule,
                "schedule_plan_in": self.schedule_plan_in,
                "custom_schedule_in": self.custom_schedule_in,
                "schedule_policy": self.schedule_policy,
                "snn_activity_profile": self.snn_activity_profile,
            }
        )


@dataclass
class RoundOrchestratedRunResult:
    status: str
    exit_code: int
    request: RoundOrchestratedRunRequest
    stage_statuses: dict[str, StageStatus] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    errors: list[ErrorRecord] = field(default_factory=list)
    accuracy_report: dict[str, Any] | None = None
    round_plan: RoundPlan | None = None
    phase2_report: Phase2Report | None = None
    hardware_summary: dict[str, Any] | None = None

    def completed_stages(self) -> list[str]:
        return sorted(
            name
            for name, stage_status in self.stage_statuses.items()
            if stage_status.status == "succeeded"
        )

    def manifest_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": SCHEMA_VERSION,
                "status": self.status,
                "exit_code": self.exit_code,
                "request": self.request.to_dict(),
                "requested_stages": list(self.request.requested_stages()),
                "stage_statuses": {
                    name: self.stage_statuses[name].to_dict()
                    for name in sorted(self.stage_statuses)
                },
                "errors": [error.to_dict() for error in self.errors],
                "artifacts": self.artifacts,
            }
        )

    def summary_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "schema_version": SCHEMA_VERSION,
                "status": self.status,
                "exit_code": self.exit_code,
                "network": self.request.network,
                "sim_config": self.request.sim_config,
                "requested_stages": list(self.request.requested_stages()),
                "completed_stages": self.completed_stages(),
                "accuracy": self.accuracy_report,
                "hardware": self.hardware_summary,
                "errors": [error.to_dict() for error in self.errors],
            }
        )


__all__ = [
    "SCHEMA_VERSION",
    "RunnerError",
    "ErrorRecord",
    "StageStatus",
    "RoundOrchestratedRunRequest",
    "RoundOrchestratedRunResult",
]
