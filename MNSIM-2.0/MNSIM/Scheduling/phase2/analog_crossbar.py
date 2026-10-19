"""Local numerical current engine for differential analog crossbars.

The engine deliberately models only the closed electrical contract needed by
phase 2: a one-dimensional input code vector drives one two-dimensional
signed weight matrix.  Input codes become row voltages by multiplying the
configured read voltage.  Positive and negative signed codes select separate
rails; no bit-slice or multi-output inference is performed.
"""

from __future__ import annotations

import configparser
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from MNSIM.Hardware_Model.Conductance import ConductanceTable


class AnalogCrossbarNumericalError(ValueError):
    """Raised when the local crossbar numerical contract is violated."""


def compute_differential_currents(
    input_codes: Any,
    read_voltage_v: Any,
    gpos_s: Any,
    gneg_s: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute differential-rail currents in amperes.

    ``input_codes`` is a finite, non-negative row vector.  The row voltage
    vector is ``input_codes * read_voltage_v`` and both conductance matrices
    must have shape ``(len(input_codes), output_count)``.
    """

    codes = _finite_vector(input_codes, "input_codes")
    if np.any(codes < 0.0):
        raise AnalogCrossbarNumericalError("input_codes must be non-negative")
    voltage = codes * _positive_scalar(read_voltage_v, "read_voltage_v")
    gpos = _conductance_matrix(gpos_s, "gpos_s", codes.size)
    gneg = _conductance_matrix(gneg_s, "gneg_s", codes.size)
    if gpos.shape != gneg.shape:
        raise AnalogCrossbarNumericalError(
            f"gpos_s and gneg_s shape mismatch: {gpos.shape} versus {gneg.shape}"
        )
    ipos = np.asarray(voltage @ gpos, dtype=np.float64)
    ineg = np.asarray(voltage @ gneg, dtype=np.float64)
    if np.any(~np.isfinite(ipos)) or np.any(~np.isfinite(ineg)):
        raise AnalogCrossbarNumericalError("computed currents must be finite")
    return ipos, ineg


@dataclass(frozen=True)
class AnalogCrossbarNumericalEngine:
    """Evaluate local signed weight codes as positive/negative rail currents."""

    conductance: ConductanceTable
    read_voltage_v: float
    polarity: int = 2
    xbar_size: tuple[int, int] = (128, 128)

    @classmethod
    def from_sim_config(
        cls,
        sim_config_path: str | Path,
        weight_codes: Any = None,
        polarity: int = 2,
    ) -> "AnalogCrossbarNumericalEngine":
        path = Path(sim_config_path)
        conductance = ConductanceTable.from_simconfig(str(path))
        read_voltage_v = _read_voltage(path)
        polarity_value = _positive_integer(polarity, "polarity")
        configured_polarity = _read_polarity(path)
        if configured_polarity != polarity_value:
            raise AnalogCrossbarNumericalError(
                f"Xbar_Polarity mismatch: SimConfig declares {configured_polarity}, "
                f"request declares {polarity_value}"
            )
        if polarity_value != 2:
            raise AnalogCrossbarNumericalError(
                "only polarity=2 closed differential crossbars are supported"
            )
        if weight_codes is not None:
            configured = _signed_weight_matrix(weight_codes, "weight_codes")
            _validate_code_range(configured, conductance.levels)
        parser = configparser.ConfigParser()
        parser.read(path, encoding="UTF-8")
        raw_size = parser.get("Crossbar level", "Xbar_Size", fallback="128,128")
        try:
            size = tuple(int(part.strip()) for part in raw_size.split(","))
        except ValueError as error:
            raise AnalogCrossbarNumericalError("Xbar_Size must be two positive integers") from error
        if len(size) != 2 or any(value <= 0 for value in size):
            raise AnalogCrossbarNumericalError("Xbar_Size must be two positive integers")
        return cls(conductance, read_voltage_v, polarity_value, size)

    def compute(
        self,
        input_values: Any,
        weight_codes_by_layer: Mapping[str, Any],
        output_neuron_count: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Compute one layer's currents and return an auditable result.

        The mapping must contain exactly one layer and its value must be a
        signed integer-valued matrix with shape
        ``(len(input_values), output_neuron_count)``.
        """

        inputs = _finite_vector(input_values, "input_values")
        if np.any(inputs < 0.0):
            raise AnalogCrossbarNumericalError("input_values must be non-negative")
        output_count = _positive_integer(output_neuron_count, "output_neuron_count")
        if not isinstance(weight_codes_by_layer, Mapping):
            raise AnalogCrossbarNumericalError(
                "weight_codes_by_layer must be a layer-id mapping"
            )
        if len(weight_codes_by_layer) != 1:
            raise AnalogCrossbarNumericalError(
                "weight_codes_by_layer must contain exactly one layer; "
                "multi-layer or multi-output inference is unsupported"
            )
        layer_id, raw_codes = next(iter(weight_codes_by_layer.items()))
        codes = _signed_weight_matrix(raw_codes, f"weight_codes_by_layer[{layer_id!r}]")
        expected_shape = (inputs.size, output_count)
        if codes.shape != expected_shape:
            raise AnalogCrossbarNumericalError(
                f"weight code matrix shape must be {expected_shape}, got {codes.shape}"
            )
        _validate_code_range(codes, self.conductance.levels)

        positive_codes = np.maximum(codes, 0)
        negative_codes = np.maximum(-codes, 0)
        gpos = self._map_codes(positive_codes)
        gneg = self._map_codes(negative_codes)
        ipos, ineg = compute_differential_currents(
            inputs, self.read_voltage_v, gpos, gneg
        )
        row_voltage = inputs * self.read_voltage_v
        load = row_voltage[:, None] ** 2 * (gpos + gneg)
        read_blocks = []
        for start_row in range(0, inputs.size, self.xbar_size[0]):
            end_row = min(start_row + self.xbar_size[0], inputs.size)
            for start_col in range(0, output_count, self.xbar_size[1]):
                end_col = min(start_col + self.xbar_size[1], output_count)
                read_blocks.append({
                    "input_range": [start_row, end_row],
                    "output_range": [start_col, end_col],
                    "row_voltage_v": row_voltage[start_row:end_row].tolist(),
                    "xbar_load_power_w": float(np.sum(load[start_row:end_row, start_col:end_col])),
                })
        audit = {
            "engine": "local_v_at_g",
            "current_units": "A",
            "read_voltage_v": self.read_voltage_v,
            "conductance_profile": "local_device_table",
            "device_levels": self.conductance.levels,
            "polarity": self.polarity,
            "layer_id": str(layer_id),
            "input_shape": [int(inputs.size)],
            "weight_shape": [int(value) for value in codes.shape],
            # Activity evidence only. Existing xbar energy remains owned by
            # original MNSIM; the extra peripheral model uses this electrical
            # load solely to compute supply conversion losses.
            "row_voltage_v": (inputs * self.read_voltage_v).tolist(),
            "xbar_load_power_w": float(np.sum(load)),
            "physical_read_blocks": read_blocks,
            "rail_shapes": {
                "ipos_a": [int(value) for value in ipos.shape],
                "ineg_a": [int(value) for value in ineg.shape],
            },
            "payload_identity": _payload_identity(
                inputs, codes, self.read_voltage_v, str(layer_id)
            ),
        }
        return ipos, ineg, audit

    def _map_codes(self, codes: np.ndarray) -> np.ndarray:
        import torch

        mapped = self.conductance.map_codes(
            torch.as_tensor(codes, dtype=torch.int64), dtype=torch.float64
        )
        return mapped.detach().cpu().numpy()


def _read_voltage(path: Path) -> float:
    parser = configparser.ConfigParser()
    if not parser.read(path, encoding="UTF-8") or "Device level" not in parser:
        raise AnalogCrossbarNumericalError(
            f"SimConfig is missing [Device level]: {path}"
        )
    raw = parser.get("Device level", "Read_Voltage", fallback="")
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) != 1:
        raise AnalogCrossbarNumericalError(
            "Read_Voltage must contain exactly one scalar value"
        )
    return _positive_scalar(values[0], "Read_Voltage")


def _read_polarity(path: Path) -> int:
    parser = configparser.ConfigParser()
    if not parser.read(path, encoding="UTF-8") or "Process element level" not in parser:
        raise AnalogCrossbarNumericalError(
            f"SimConfig is missing [Process element level]: {path}"
        )
    raw = parser.get("Process element level", "Xbar_Polarity", fallback="")
    try:
        polarity = int(raw)
    except (TypeError, ValueError) as error:
        raise AnalogCrossbarNumericalError(
            "Xbar_Polarity must be a positive integer"
        ) from error
    if polarity <= 0:
        raise AnalogCrossbarNumericalError(
            "Xbar_Polarity must be a positive integer"
        )
    return polarity


def _finite_vector(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AnalogCrossbarNumericalError(f"{name} must be a numerical vector") from error
    if result.ndim != 1 or result.size == 0 or np.any(~np.isfinite(result)):
        raise AnalogCrossbarNumericalError(
            f"{name} must be a non-empty finite one-dimensional vector"
        )
    return result.copy()


def _conductance_matrix(value: Any, name: str, rows: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AnalogCrossbarNumericalError(f"{name} must be a numerical matrix") from error
    if result.ndim != 2 or result.shape[0] != rows:
        raise AnalogCrossbarNumericalError(
            f"{name} shape must be ({rows}, output_count), got {result.shape}"
        )
    if result.shape[1] == 0 or np.any(~np.isfinite(result)):
        raise AnalogCrossbarNumericalError(f"{name} must contain finite values")
    if np.any(result < 0.0):
        raise AnalogCrossbarNumericalError(f"{name} must be non-negative")
    return result.copy()


def _signed_weight_matrix(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise AnalogCrossbarNumericalError(f"{name} must be a 2-D numerical matrix") from error
    if result.ndim != 2 or 0 in result.shape:
        raise AnalogCrossbarNumericalError(
            f"{name} must be a non-empty 2-D matrix; bit-slice tensors are unsupported"
        )
    try:
        numeric = result.astype(np.float64)
    except (TypeError, ValueError) as error:
        raise AnalogCrossbarNumericalError(f"{name} must be numerical") from error
    if np.any(~np.isfinite(numeric)):
        raise AnalogCrossbarNumericalError(f"{name} must contain finite values")
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        raise AnalogCrossbarNumericalError(f"{name} must contain integer-valued codes")
    return rounded.astype(np.int64)


def _validate_code_range(codes: np.ndarray, levels: int) -> None:
    if np.any(codes >= levels) or np.any(codes <= -levels):
        raise AnalogCrossbarNumericalError(
            f"weight code magnitude must be below device levels ({levels})"
        )


def _positive_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalogCrossbarNumericalError(f"{name} must be a scalar") from error
    if not np.isfinite(result) or result <= 0.0:
        raise AnalogCrossbarNumericalError(f"{name} must be finite and positive")
    return result


def _positive_integer(value: Any, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise AnalogCrossbarNumericalError(f"{name} must be a positive integer") from error
    if isinstance(value, bool) or result <= 0 or float(value) != result:
        raise AnalogCrossbarNumericalError(f"{name} must be a positive integer")
    return result


def _payload_identity(
    inputs: np.ndarray, codes: np.ndarray, read_voltage_v: float, layer_id: str
) -> str:
    encoded = json.dumps(
        {
            "layer_id": layer_id,
            "input_values": inputs.tolist(),
            "weight_codes": codes.tolist(),
            "read_voltage_v": read_voltage_v,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "AnalogCrossbarNumericalEngine",
    "AnalogCrossbarNumericalError",
    "compute_differential_currents",
]
