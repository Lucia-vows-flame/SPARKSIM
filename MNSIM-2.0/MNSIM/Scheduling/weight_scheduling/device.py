"""Generic discrete-state device models for weight scheduling."""

from __future__ import annotations

import configparser
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

class DeviceModelError(ValueError):
    """A device model cannot satisfy the scheduling device contract."""


@dataclass(frozen=True)
class DeviceTransitionCost:
    source_code: int
    target_code: int
    changed: bool
    objective_cost: float
    objective_unit: str
    latency_s: float | None
    energy_j: float | None
    exact: bool
    modeled_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]


class DeviceModel(Protocol):
    model_id: str
    device_bits: int
    device_level: int

    def validate_code(self, code: int) -> None:
        ...

    def initial_code(self) -> int:
        ...

    def cell_transition_cost(self, source_code: int, target_code: int) -> DeviceTransitionCost:
        ...


@dataclass(frozen=True)
class JsonDeviceModel:
    """Device model loaded from the repository's canonical JSON description."""

    model_id: str
    device_bits: int
    device_level: int
    nominal_states: tuple[float, ...]
    initial_code_value: int = 0
    objective_unit: str = "changed_cell_count"
    exact_write_latency: bool = False
    exact_write_energy: bool = False
    transition_latency_s: tuple[float, ...] | None = None
    transition_energy_j: tuple[float, ...] | None = None
    metadata: Mapping[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.device_level < 2 or self.device_bits <= 0:
            raise DeviceModelError("device level and bit width must be positive")
        if self.device_level != 1 << self.device_bits:
            raise DeviceModelError(
                f"device_level {self.device_level} does not match device_bits {self.device_bits}"
            )
        if len(self.nominal_states) != self.device_level:
            raise DeviceModelError("nominal state count does not match device level")
        if not 0 <= self.initial_code_value < self.device_level:
            raise DeviceModelError("initial device code is outside the device range")
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def validate_code(self, code: int) -> None:
        if isinstance(code, bool) or not isinstance(code, int):
            raise DeviceModelError(f"device code must be an integer, got {code!r}")
        if code < 0 or code >= self.device_level:
            raise DeviceModelError(
                f"device code {code} is outside [0, {self.device_level})"
            )

    def initial_code(self) -> int:
        return self.initial_code_value

    def cell_transition_cost(self, source_code: int, target_code: int) -> DeviceTransitionCost:
        self.validate_code(source_code)
        self.validate_code(target_code)
        changed = source_code != target_code
        latency = None
        energy = None
        modeled: list[str] = []
        unavailable: list[str] = []
        if self.transition_latency_s is not None:
            latency = float(self.transition_latency_s[source_code * self.device_level + target_code])
            modeled.append("latency_s")
        else:
            unavailable.append("latency_s")
        if self.transition_energy_j is not None:
            energy = float(self.transition_energy_j[source_code * self.device_level + target_code])
            modeled.append("energy_j")
        else:
            unavailable.append("energy_j")
        return DeviceTransitionCost(
            source_code=source_code,
            target_code=target_code,
            changed=changed,
            objective_cost=float(changed),
            objective_unit=self.objective_unit,
            latency_s=latency,
            energy_j=energy,
            exact=bool(self.exact_write_latency and self.exact_write_energy),
            modeled_fields=tuple(modeled),
            unavailable_fields=tuple(unavailable),
        )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeviceModelError(f"device model field {name!r} must be an object")
    return value


def _levels(model: Mapping[str, Any]) -> tuple[float, ...]:
    levels = _mapping(model.get("levels"), "levels")
    raw = levels.get("nominal_resistance_ohm")
    if raw is None:
        raw = _mapping(levels, "levels").get("nominal_conductance_s")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise DeviceModelError("device model must provide nominal device levels")
    return tuple(float(value) for value in raw)


def _optional_matrix(values: Any, level: int, name: str) -> tuple[float, ...] | None:
    if values is None:
        return None
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise DeviceModelError(f"{name} must be a sequence")
    expected = level * level
    if len(values) != expected:
        raise DeviceModelError(f"{name} must contain {expected} entries")
    return tuple(float(value) for value in values)


def device_model_from_dict(model: Mapping[str, Any]) -> JsonDeviceModel:
    model_id = str(model.get("model_id", "")).strip()
    if not model_id:
        raise DeviceModelError("device model is missing model_id")
    device = _mapping(model.get("device"), "device")
    level = int(device.get("levels"))
    bits = int(device.get("bits", level.bit_length() - 1))
    transition = model.get("transition_cost") or {}
    transition = _mapping(transition, "transition_cost")
    latency = _optional_matrix(transition.get("latency_s"), level, "latency_s")
    energy = _optional_matrix(transition.get("energy_j"), level, "energy_j")
    capabilities = _mapping(model.get("capabilities", {}), "capabilities")
    has_transition_matrix = (
        latency is not None
        and energy is not None
        and len(latency) == level * level
        and len(energy) == level * level
    )
    legacy_pulse_model = (
        device.get("pulse_width_ns") is not None
        and device.get("programming_voltage_abs_v") is not None
        and isinstance(_mapping(model.get("levels"), "levels").get("pulse"), Sequence)
        and len(_mapping(model.get("levels"), "levels").get("pulse")) == level
    )
    hardware_write_supported = bool(
        capabilities.get("hardware_write_accounting", True)
        and (has_transition_matrix or legacy_pulse_model)
    )
    return JsonDeviceModel(
        model_id=model_id,
        device_bits=bits,
        device_level=level,
        nominal_states=_levels(model),
        initial_code_value=int(device.get("initial_code", 0)),
        objective_unit=str(transition.get("objective_unit", "changed_cell_count")),
        exact_write_latency=latency is not None and hardware_write_supported,
        exact_write_energy=energy is not None and hardware_write_supported,
        transition_latency_s=latency,
        transition_energy_j=energy,
        metadata={
            "capabilities": dict(capabilities),
            "hardware_write_accounting_supported": hardware_write_supported,
        },
    )


_DEVICE_REGISTRY: dict[str, DeviceModel] = {}


def register_device_model(model: DeviceModel) -> None:
    model_id = str(model.model_id).strip()
    if not model_id:
        raise DeviceModelError("device model id cannot be empty")
    _DEVICE_REGISTRY[model_id] = model


def load_device_model(model_id: str, *, repo_root: str | Path | None = None) -> DeviceModel:
    canonical = str(model_id).strip()
    if canonical in _DEVICE_REGISTRY:
        return _DEVICE_REGISTRY[canonical]
    root = (
        Path(repo_root).resolve()
        if repo_root is not None
        else Path(__file__).resolve().parents[3]
    )
    path = root / "device_models" / "final_rram" / "models" / f"{canonical}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeviceModelError(f"cannot load device model {canonical!r} from {path}: {exc}") from exc
    model = device_model_from_dict(payload)
    register_device_model(model)
    return model


def load_device_model_from_sim_config(
    sim_config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> DeviceModel:
    parser = configparser.ConfigParser()
    if not parser.read(sim_config_path, encoding="UTF-8"):
        raise DeviceModelError(f"cannot read SimConfig {sim_config_path}")
    try:
        model_id = parser.get("Final RRAM Model", "Model_ID")
    except configparser.Error as exc:
        raise DeviceModelError("SimConfig is missing [Final RRAM Model].Model_ID") from exc
    return load_device_model(model_id, repo_root=repo_root)


def load_native_device_model_from_sim_config(
    sim_config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> DeviceModel:
    """Load the final device model and attach MNSIM's physical write tables.

    Canonical JSON transition matrices are exposed directly. Legacy profiles
    without such matrices retain the historical SimConfig-derived primitive.
    """

    base = load_device_model_from_sim_config(sim_config_path, repo_root=repo_root)
    if not isinstance(base, JsonDeviceModel):
        raise DeviceModelError("native write costs require a JsonDeviceModel")
    if not bool(base.metadata.get("hardware_write_accounting_supported", False)):
        raise DeviceModelError(
            f"{base.model_id} does not provide a complete source/target write transition primitive"
        )
    if base.transition_latency_s is not None and base.transition_energy_j is not None:
        metadata = dict(base.metadata or {})
        metadata.update(
            {
                "native_write_cost_source": "final_rram_json_transition_matrix",
                "native_write_cost_sim_config": str(Path(sim_config_path).resolve()),
                "native_write_cost_semantics": "canonical source-to-target transition matrix",
            }
        )
        return replace(base, metadata=metadata)
    parser = configparser.ConfigParser()
    if not parser.read(sim_config_path, encoding="UTF-8"):
        raise DeviceModelError(f"cannot read SimConfig {sim_config_path}")
    try:
        level = parser.getint("Device level", "Device_Level")
        resistance = _float_vector(parser, "Device level", "Device_Resistance")
        write_voltage = _float_vector(parser, "Device level", "Write_Voltage")
        pulse_latency = _float_vector(parser, "Device level", "pulse_latency")
        set_pulse = _float_vector(parser, "Device level", "SET_pulse")
        reset_pulse = _float_vector(parser, "Device level", "RESET_pulse")
    except (configparser.Error, ValueError) as exc:
        raise DeviceModelError(f"invalid native write-cost fields: {exc}") from exc
    if level != base.device_level:
        raise DeviceModelError(
            f"SimConfig device level {level} does not match model level {base.device_level}"
        )
    if len(resistance) != level or len(set_pulse) != level or len(reset_pulse) != level:
        raise DeviceModelError("native write vectors do not match Device_Level")
    if len(write_voltage) != 2 or len(pulse_latency) != 2:
        raise DeviceModelError("Write_Voltage and pulse_latency must contain two values")

    source = np.arange(level, dtype=np.float64).reshape(level, 1)
    target = np.arange(level, dtype=np.float64).reshape(1, level)
    resistance_array = np.asarray(resistance, dtype=np.float64)
    set_array = np.asarray(set_pulse, dtype=np.float64)
    reset_array = np.asarray(reset_pulse, dtype=np.float64)
    source_le_target = source <= target
    set_count = np.where(
        source_le_target,
        set_array[target.astype(np.intp)] - set_array[source.astype(np.intp)],
        set_array[target.astype(np.intp)],
    )
    reset_count = np.where(source_le_target, 0.0, reset_array[source.astype(np.intp)])
    latency_ns = reset_count * float(pulse_latency[0]) + set_count * float(pulse_latency[1])

    reset_resistance = np.sqrt(resistance_array[0] * resistance_array[source.astype(np.intp)])
    set_resistance = np.where(
        source_le_target,
        np.sqrt(resistance_array[source.astype(np.intp)] * resistance_array[target.astype(np.intp)]),
        np.sqrt(resistance_array[0] * resistance_array[target.astype(np.intp)]),
    )
    reset_power = (float(write_voltage[0]) ** 2) / reset_resistance
    set_power = (float(write_voltage[1]) ** 2) / set_resistance
    energy_nj = reset_power * (reset_count * float(pulse_latency[0]))
    energy_nj += set_power * (set_count * float(pulse_latency[1]))

    metadata = dict(base.metadata or {})
    metadata.update(
        {
            "native_write_cost_source": "MNSIM.Scheduling.phase2.physical_write_ledger",
            "native_write_cost_sim_config": str(Path(sim_config_path).resolve()),
            "native_write_cost_semantics": "per-cell energy and pulse latency",
        }
    )
    return replace(
        base,
        objective_unit="physical_write_energy_j",
        exact_write_latency=True,
        exact_write_energy=True,
        transition_latency_s=tuple(float(value) * 1e-9 for value in latency_ns.reshape(-1)),
        transition_energy_j=tuple(float(value) * 1e-9 for value in energy_nj.reshape(-1)),
        metadata=metadata,
    )


def _float_vector(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> list[float]:
    raw = parser.get(section, option)
    return [float(value.strip()) for value in raw.replace("[", "").replace("]", "").split(",") if value.strip()]


__all__ = [
    "DeviceModel",
    "DeviceModelError",
    "DeviceTransitionCost",
    "JsonDeviceModel",
    "device_model_from_dict",
    "load_device_model",
    "load_device_model_from_sim_config",
    "load_native_device_model_from_sim_config",
    "register_device_model",
]
