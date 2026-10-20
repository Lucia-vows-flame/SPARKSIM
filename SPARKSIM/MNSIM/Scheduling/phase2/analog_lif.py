"""Local LUT-backed analog LIF transition and cost accounting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from MNSIM.Hardware_Model.analog_lif_lut import AnalogLifLutBundle


class AnalogLifError(ValueError):
    """Raised when analog LIF execution violates its local contract."""


@dataclass(frozen=True)
class AnalogLifCost:
    latency_s: float
    energy_j: float
    power_w: float

    def to_dict(self) -> dict[str, float]:
        return {
            "latency_s": self.latency_s,
            "energy_j": self.energy_j,
            "power_w": self.power_w,
        }


@dataclass(frozen=True)
class AnalogLifResult:
    vmem_next: np.ndarray
    spike_codes: tuple[int, ...]
    frontend_iout_to_lif_a: np.ndarray
    raw_vmem_next_v: np.ndarray
    frontend_latency_s: float
    frontend_energy_j: float
    lif_latency_s: float
    lif_energy_j: float
    cost: AnalogLifCost
    audit_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vmem_next": _json_array(self.vmem_next),
            "spike_codes": list(self.spike_codes),
            "frontend_iout_to_lif_a": _json_array(self.frontend_iout_to_lif_a),
            "raw_vmem_next_v": _json_array(self.raw_vmem_next_v),
            "frontend_latency_s": self.frontend_latency_s,
            "frontend_energy_j": self.frontend_energy_j,
            "lif_latency_s": self.lif_latency_s,
            "lif_energy_j": self.lif_energy_j,
            "cost": self.cost.to_dict(),
            "audit_metadata": _json_value(self.audit_metadata),
        }


class AnalogLifBackend:
    """Execute one analog current-driven LIF action through local LUT queries."""

    _POLICIES = {"error", "clamp_zero"}

    def __init__(
        self,
        bundle: AnalogLifLutBundle,
        negative_vmem_policy: str = "error",
    ) -> None:
        if not isinstance(bundle, AnalogLifLutBundle):
            raise AnalogLifError("bundle must be an AnalogLifLutBundle")
        policy = str(negative_vmem_policy)
        if policy not in self._POLICIES:
            raise AnalogLifError(
                f"unsupported negative_vmem_policy {policy!r}; "
                f"expected one of {sorted(self._POLICIES)!r}"
            )
        self.bundle = bundle
        self.negative_vmem_policy = policy

    def execute(self, *, ipos_a: Any, ineg_a: Any, vmem0_v: Any) -> AnalogLifResult:
        ipos = _finite_float_array(ipos_a, "ipos_a")
        ineg = _finite_float_array(ineg_a, "ineg_a")
        vmem0 = _finite_float_array(vmem0_v, "vmem0_v")
        if ipos.size == 0:
            raise AnalogLifError("analog LIF inputs must contain at least one neuron")
        if ipos.shape != ineg.shape or ipos.shape != vmem0.shape:
            raise AnalogLifError(
                "ipos_a, ineg_a, and vmem0_v must have the same shape"
            )

        frontend = self.bundle.query_frontend(ipos, ineg)
        iout = _query_array(frontend.iout_to_lif_a, ipos.shape, "iout_to_lif_a")
        # All closed differential-xbar outputs settle in one synchronous
        # frontend action window; energy is additive, latency is the slowest
        # queried output.
        frontend_latency = _max_query(frontend.settle_latency_s, "settle_latency_s")
        frontend_energy = _sum_query(frontend.energy_vdd_j, "energy_vdd_j")

        lif = self.bundle.query_lif(iout, vmem0)
        raw_vmem_next = _query_array(lif.vmem_next_v, ipos.shape, "vmem_next_v")
        spike_flags = _query_array(lif.spike_flag, ipos.shape, "spike_flag")
        lif_energy = _sum_query(lif.energy_rails_j, "energy_rails_j")
        lif_latency = _positive_float(lif.step_latency_s, "step_latency_s")

        vmem_next, clamp_count, raw_min_clamped = self._apply_negative_policy(
            raw_vmem_next
        )
        total_latency = frontend_latency + lif_latency
        total_energy = frontend_energy + lif_energy
        cost = AnalogLifCost(
            latency_s=total_latency,
            energy_j=total_energy,
            power_w=total_energy / total_latency,
        )
        query_count = int(ipos.size) if ipos.shape else 1
        audit = {
            "lut_schema_id": str(self.bundle.manifest.get("schema_id", "")),
            "lut_profile": self.bundle.profile,
            "lut_root": Path(self.bundle.root).as_posix(),
            "checksum_verified": self.bundle.checksum_verified,
            "interpolation": "bounded_bilinear",
            "query_counts": {"frontend": query_count, "lif": query_count},
            "axis_clamp_count": 0,
            "axis_reject_count": 0,
            "invalid_neighbor_reject_count": 0,
            "negative_vmem_policy": self.negative_vmem_policy,
            "negative_vmem_clamp_count": clamp_count,
            "raw_min_clamped_vmem_next_v": raw_min_clamped,
            "frontend_latency_s": frontend_latency,
            "frontend_energy_j": frontend_energy,
            "lif_latency_s": lif_latency,
            "lif_energy_j": lif_energy,
            "cost_source": "lut_only",
            "power_source": "lut_energy_divided_by_lut_latency",
        }
        return AnalogLifResult(
            vmem_next=vmem_next,
            spike_codes=tuple(int(value) for value in np.ravel(spike_flags)),
            frontend_iout_to_lif_a=iout,
            raw_vmem_next_v=raw_vmem_next,
            frontend_latency_s=frontend_latency,
            frontend_energy_j=frontend_energy,
            lif_latency_s=lif_latency,
            lif_energy_j=lif_energy,
            cost=cost,
            audit_metadata=audit,
        )

    def _apply_negative_policy(
        self, raw_vmem_next: np.ndarray
    ) -> tuple[np.ndarray, int, float | None]:
        negative_mask = raw_vmem_next < 0.0
        clamp_count = int(np.count_nonzero(negative_mask))
        if clamp_count == 0:
            return raw_vmem_next.copy(), 0, None
        raw_min = float(np.min(raw_vmem_next[negative_mask]))
        if self.negative_vmem_policy == "error":
            raise AnalogLifError(
                f"negative Vmem successor {raw_min} V encountered"
            )
        return np.maximum(raw_vmem_next, 0.0), clamp_count, raw_min


def _finite_float_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise AnalogLifError(f"{name} must contain only finite values")
    return array


def _query_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise AnalogLifError(f"{name} returned shape {array.shape}, expected {shape}")
    if np.any(~np.isfinite(array.astype(float))):
        raise AnalogLifError(f"{name} returned non-finite values")
    return array.copy()


def _sum_query(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise AnalogLifError(f"{name} returned non-finite values")
    return float(np.sum(array))


def _max_query(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if np.any(~np.isfinite(array)):
        raise AnalogLifError(f"{name} returned non-finite values")
    return float(np.max(array))


def _positive_float(value: Any, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise AnalogLifError(f"{name} must be finite and strictly positive")
    return scalar


def _json_array(value: np.ndarray) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_array(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


__all__ = [
    "AnalogLifBackend",
    "AnalogLifCost",
    "AnalogLifError",
    "AnalogLifResult",
]
