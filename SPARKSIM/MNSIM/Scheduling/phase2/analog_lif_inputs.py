"""Strict numerical input and persistent state contract for analog LIF."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from MNSIM.Scheduling.phase2.analog_crossbar import (
    AnalogCrossbarNumericalEngine,
    AnalogCrossbarNumericalError,
)


class AnalogLifPayloadError(ValueError):
    """An analog invocation lacks an honest, supported numerical payload."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class AnalogLifNumericalInput:
    ipos_a: np.ndarray
    ineg_a: np.ndarray
    vmem0_v: np.ndarray
    source: str
    payload_identity: str
    crossbar_audit: dict[str, Any] = field(default_factory=dict)


class AnalogLifPayloadSession:
    """Resolve per-step currents and retain measured membrane successors."""

    def __init__(
        self,
        model_inputs: Any,
        *,
        sim_config_path: str | None = None,
        weight_codes_by_layer: Mapping[str, Any] | None = None,
    ) -> None:
        self._payloads = _collect_payloads(model_inputs)
        self._crossbar = None
        self._weight_codes_by_layer = weight_codes_by_layer
        if sim_config_path is not None:
            try:
                self._crossbar = AnalogCrossbarNumericalEngine.from_sim_config(
                    sim_config_path
                )
            except (AnalogCrossbarNumericalError, OSError, ValueError, KeyError) as error:
                raise AnalogLifPayloadError(
                    "INVALID_ANALOG_CROSSBAR_CONFIGURATION", str(error)
                ) from error
        self._states: dict[tuple[str, str], np.ndarray] = {}
        self._last_timestep: dict[tuple[str, str], int] = {}
        self._neuron_counts: dict[tuple[str, str], int] = {}

    def resolve(
        self,
        *,
        sample_id: str,
        node_id: str,
        timestep: int,
        sequence_length: int,
        neuron_count: int,
    ) -> AnalogLifNumericalInput:
        payload = self._payloads.get(str(node_id))
        if payload is None:
            raise AnalogLifPayloadError(
                "MISSING_ANALOG_LIF_PAYLOAD",
                f"model_inputs metadata has no analog_lif_payloads entry for {node_id}",
            )
        _validate_payload_header(payload, node_id, sequence_length, neuron_count)
        key = (str(sample_id), str(node_id))
        previous_count = self._neuron_counts.get(key)
        if previous_count is not None and previous_count != neuron_count:
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_NEURON_COUNT",
                f"{key} expected neuron count {previous_count}, got {neuron_count}",
            )
        self._neuron_counts[key] = int(neuron_count)
        expected = self._last_timestep.get(key, -1) + 1
        if timestep != expected:
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_INVOCATION_ORDER",
                f"{key} expected timestep {expected}, got {timestep}",
            )
        timesteps = payload["timesteps"]
        step = timesteps[timestep]
        if not isinstance(step, Mapping):
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_PAYLOAD",
                f"{node_id} timestep {timestep} must be a mapping",
            )
        if "initial_vmem_v" in step or "vmem0_v" in step:
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_STATE_RESET",
                "per-timestep membrane initialization is not supported",
            )
        source = str(step.get("source", ""))
        if source != "input_weight":
            raise AnalogLifPayloadError(
                "UNSUPPORTED_ANALOG_NUMERICAL_SOURCE",
                f"{node_id} timestep {timestep} source must be 'input_weight'; "
                "precomputed currents and explicit conductance payloads are not public inputs",
            )
        _reject_forbidden_external_numerical_fields(payload, step, node_id, timestep)
        if self._crossbar is None:
            raise AnalogLifPayloadError(
                "MISSING_ANALOG_CROSSBAR_ENGINE",
                "analog input/weight payloads require sim_config_path",
            )
        inputs = _vector_any_length(step.get("input_values"), "input_values")
        if "read_voltage_v" in step:
            try:
                read_voltage = float(step["read_voltage_v"])
            except (TypeError, ValueError) as error:
                raise AnalogLifPayloadError(
                    "INVALID_ANALOG_LIF_PAYLOAD",
                    "read_voltage_v must be a finite positive scalar",
                ) from error
            if not np.isfinite(read_voltage) or read_voltage <= 0.0:
                raise AnalogLifPayloadError(
                    "INVALID_ANALOG_LIF_PAYLOAD",
                    "read_voltage_v must be a finite positive scalar",
                )
            if not np.isclose(read_voltage, self._crossbar.read_voltage_v):
                raise AnalogLifPayloadError(
                    "INVALID_ANALOG_LIF_PAYLOAD",
                    "payload read_voltage_v does not match SimConfig read voltage",
                )
        codes_by_layer = _weight_codes_for_step(
            payload,
            step,
            self._weight_codes_by_layer,
        )
        try:
            ipos, ineg, crossbar_audit = self._crossbar.compute(
                input_values=inputs,
                weight_codes_by_layer=codes_by_layer,
                output_neuron_count=neuron_count,
            )
        except AnalogCrossbarNumericalError as error:
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_PAYLOAD", str(error)
            ) from error
        if np.any(ipos < 0.0) or np.any(ineg < 0.0):
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_PAYLOAD",
                "closed positive and negative rail currents must be non-negative",
            )
        if key in self._states:
            vmem0 = self._states[key].copy()
        else:
            vmem0 = _vector(payload.get("initial_vmem_v"), "initial_vmem_v", neuron_count)
        identity = _payload_identity(node_id, timestep, step)
        return AnalogLifNumericalInput(
            ipos,
            ineg,
            vmem0,
            source,
            identity,
            crossbar_audit,
        )

    def commit(
        self,
        *,
        sample_id: str,
        node_id: str,
        timestep: int,
        vmem_next_v: Any,
        expected_neuron_count: int | None = None,
    ) -> None:
        key = (str(sample_id), str(node_id))
        expected = self._last_timestep.get(key, -1) + 1
        if timestep != expected:
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_INVOCATION_ORDER",
                f"{key} expected commit timestep {expected}, got {timestep}",
            )
        expected_count = self._neuron_counts.get(key)
        if expected_count is None:
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_SUCCESSOR",
                f"{key} has no resolved neuron-count contract",
            )
        if (
            expected_neuron_count is not None
            and int(expected_neuron_count) != expected_count
        ):
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_NEURON_COUNT",
                f"{key} expected {expected_count} neurons, got {expected_neuron_count}",
            )
        state = np.asarray(vmem_next_v, dtype=float)
        if (
            state.ndim != 1
            or state.size != expected_count
            or np.any(~np.isfinite(state))
        ):
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_SUCCESSOR",
                f"analog backend returned invalid membrane successor; expected {expected_count} values",
            )
        self._states[key] = state.copy()
        self._last_timestep[key] = int(timestep)


def _collect_payloads(model_inputs: Any) -> dict[str, Mapping[str, Any]]:
    if model_inputs is None:
        return {}
    if not isinstance(model_inputs, Mapping):
        raise AnalogLifPayloadError(
            "INVALID_MODEL_INPUTS", "model_inputs must be a mapping"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for state_id, metadata in model_inputs.items():
        if not isinstance(metadata, Mapping):
            continue
        entries = metadata.get("analog_lif_payloads")
        if entries is None:
            continue
        if not isinstance(entries, Mapping):
            raise AnalogLifPayloadError(
                "INVALID_ANALOG_LIF_PAYLOAD",
                f"model input {state_id!r} analog_lif_payloads must be a mapping",
            )
        for node_id, payload in entries.items():
            key = str(node_id)
            if key in result:
                raise AnalogLifPayloadError(
                    "DUPLICATE_ANALOG_LIF_PAYLOAD",
                    f"multiple model input records provide payload for {key}",
                )
            if not isinstance(payload, Mapping):
                raise AnalogLifPayloadError(
                    "INVALID_ANALOG_LIF_PAYLOAD",
                    f"payload for {key} must be a mapping",
                )
            result[key] = payload
    return result


def _validate_payload_header(
    payload: Mapping[str, Any], node_id: str, sequence_length: int, neuron_count: int
) -> None:
    if payload.get("contract_version") != 1:
        raise AnalogLifPayloadError(
            "UNSUPPORTED_ANALOG_PAYLOAD_VERSION",
            f"{node_id} contract_version must be 1",
        )
    if payload.get("electrical_layout") != "closed_differential_outputs":
        raise AnalogLifPayloadError(
            "UNSUPPORTED_ANALOG_ELECTRICAL_LAYOUT",
            (
                f"{node_id} must provide closed_differential_outputs; bit-slice "
                "scaling or merging is not inferred"
            ),
        )
    if payload.get("neuron_count") != neuron_count:
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_NEURON_COUNT",
            f"{node_id} payload neuron_count does not equal request count {neuron_count}",
        )
    timesteps = payload.get("timesteps")
    if not isinstance(timesteps, list) or len(timesteps) != sequence_length:
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_PAYLOAD",
            f"{node_id} must provide exactly {sequence_length} timestep payloads",
        )


def _vector(value: Any, name: str, size: int) -> np.ndarray:
    result = _vector_any_length(value, name)
    if result.size != size:
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_NEURON_COUNT",
            f"{name} has {result.size} values, expected {size}",
        )
    return result


def _vector_any_length(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_PAYLOAD", f"{name} must be a numerical vector"
        ) from error
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)):
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_PAYLOAD",
            f"{name} must be a non-empty finite one-dimensional vector",
        )
    return result.copy()


def _conductance_matrix(
    value: Any, name: str, rows: int, columns: int
) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_PAYLOAD", f"{name} must be a numerical matrix"
        ) from error
    if result.shape != (rows, columns) or np.any(~np.isfinite(result)):
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_PAYLOAD",
            f"{name} shape must be {(rows, columns)}, got {result.shape}",
        )
    if np.any(result < 0.0):
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_PAYLOAD", f"{name} must be non-negative"
        )
    return result


def _weight_codes_for_step(
    payload: Mapping[str, Any],
    step: Mapping[str, Any],
    defaults: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    value = step.get("weight_codes_by_layer")
    if value is None:
        value = payload.get("weight_codes_by_layer")
    if value is None:
        value = defaults
    if not isinstance(value, Mapping) or not value:
        raise AnalogLifPayloadError(
            "MISSING_ANALOG_WEIGHT_DATA",
            "input_weight payload requires weight_codes_by_layer",
        )
    layer_id = step.get("weight_layer_id", payload.get("weight_layer_id"))
    if layer_id is not None:
        layer_id = str(layer_id)
        if layer_id not in value:
            raise AnalogLifPayloadError(
                "MISSING_ANALOG_WEIGHT_DATA",
                f"weight_layer_id {layer_id!r} is absent from weight_codes_by_layer",
            )
        return {layer_id: value[layer_id]}
    if len(value) != 1:
        raise AnalogLifPayloadError(
            "AMBIGUOUS_ANALOG_WEIGHT_DATA",
            "multiple weight layers require weight_layer_id",
        )
    return value


def _reject_forbidden_external_numerical_fields(
    payload: Mapping[str, Any],
    step: Mapping[str, Any],
    node_id: str,
    timestep: int,
) -> None:
    forbidden = {
        "ipos_a",
        "ineg_a",
        "gpos_s",
        "gneg_s",
    }
    present = sorted(
        field
        for field in forbidden
        if field in payload or field in step
    )
    if present:
        raise AnalogLifPayloadError(
            "FORBIDDEN_ANALOG_NUMERICAL_FIELDS",
            f"{node_id} timestep {timestep} cannot provide external numerical fields: "
            + ", ".join(present),
        )


def _payload_identity(node_id: str, timestep: int, step: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            {"node_id": node_id, "timestep": timestep, "payload": step},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AnalogLifPayloadError(
            "INVALID_ANALOG_LIF_PAYLOAD",
            "numerical payload must be finite and JSON-serializable",
        ) from error
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "AnalogLifNumericalInput",
    "AnalogLifPayloadError",
    "AnalogLifPayloadSession",
]
