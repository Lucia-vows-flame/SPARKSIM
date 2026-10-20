"""Physical weight-programming ledger for round-orchestrated ANN execution.

The scheduler owns a physical round boundary.  At every weighted boundary this
module materializes the real quantized checkpoint codes in the slots selected
by the TCG, compares them with the device state left by the preceding weighted
round, and evaluates the resulting cell transitions.
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

from MNSIM.Scheduling.phase2.types import MnsimGroupFragment, Phase2Invalid


PHYSICAL_WRITE_LEDGER_INVALID = "PHYSICAL_WRITE_LEDGER_INVALID"
PHYSICAL_WRITE_CODES_MISSING = "PHYSICAL_WRITE_CODES_MISSING"


@dataclass(frozen=True)
class PhysicalWriteTables:
    """Cell-transition tables derived directly from one SimConfig."""

    sim_config_path: str
    xbar_shape: tuple[int, int]
    ou_shape: tuple[int, int]
    device_level: int
    latency_ns: np.ndarray
    energy_nj: np.ndarray
    table_digest: str


_TABLE_CACHE: dict[str, PhysicalWriteTables] = {}
_PhysicalSlot = tuple[int, int, int, int]


def load_physical_write_tables(sim_config_path: str) -> PhysicalWriteTables:
    """Build the SNN-compatible physical transition primitive for ANN rounds.

    Legacy profiles use the formula from ``Weight_update.PhysicalWriteCost``.
    Profiles with a canonical JSON transition matrix use that source-to-target
    table directly, while retaining the same xbar and OU aggregation rules.
    """

    parser = configparser.ConfigParser()
    try:
        loaded = parser.read(sim_config_path, encoding="UTF-8")
    except (OSError, configparser.Error) as error:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"could not read SimConfig {sim_config_path}: {error}",
        ) from error
    if not loaded:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"could not read SimConfig {sim_config_path}",
        )

    xbar_shape = _parse_pair(parser, "Crossbar level", "Xbar_Size")
    ou_shape = _parse_pair(parser, "Algorithm Configuration", "OU_Size")
    device_level = _positive_int(
        _config_value(parser, "Device level", "Device_Level"),
        "Device_Level",
    )
    resistance = _float_vector(parser, "Device level", "Device_Resistance")
    write_voltage = _float_vector(parser, "Device level", "Write_Voltage")
    pulse_latency = _float_vector(parser, "Device level", "pulse_latency")
    set_pulse = _float_vector(parser, "Device level", "SET_pulse")
    reset_pulse = _float_vector(parser, "Device level", "RESET_pulse")
    transition_override = _load_transition_override(parser, sim_config_path, device_level)

    if len(resistance) != device_level:
        raise _invalid_config(
            f"Device_Resistance has {len(resistance)} entries, expected {device_level}"
        )
    if len(set_pulse) != device_level or len(reset_pulse) != device_level:
        raise _invalid_config(
            "SET_pulse and RESET_pulse must each have Device_Level entries"
        )
    if len(write_voltage) != 2 or len(pulse_latency) != 2:
        raise _invalid_config(
            "Write_Voltage and pulse_latency must each contain RESET,SET values"
        )
    if any(value <= 0.0 for value in resistance):
        raise _invalid_config("Device_Resistance values must be positive")

    cache_key = repr(
        (
            str(sim_config_path),
            xbar_shape,
            ou_shape,
            device_level,
            resistance,
            write_voltage,
            pulse_latency,
            set_pulse,
            reset_pulse,
            transition_override[2] if transition_override is not None else None,
        )
    )
    cached = _TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    resistance_array = np.asarray(resistance, dtype=np.float64)
    voltage_array = np.asarray(write_voltage, dtype=np.float64)
    pulse_latency_array = np.asarray(pulse_latency, dtype=np.float64)
    set_pulse_array = np.asarray(set_pulse, dtype=np.float64)
    reset_pulse_array = np.asarray(reset_pulse, dtype=np.float64)

    original_idx = np.arange(device_level, dtype=np.intp).reshape(device_level, 1)
    target_idx = np.arange(device_level, dtype=np.intp).reshape(1, device_level)
    original_le_target = original_idx <= target_idx

    set_pulse_count = np.where(
        original_le_target,
        set_pulse_array[target_idx] - set_pulse_array[original_idx],
        set_pulse_array[target_idx],
    )
    reset_pulse_count = np.where(
        original_le_target,
        0.0,
        reset_pulse_array[original_idx],
    )
    reset_latency_ns = reset_pulse_count * float(pulse_latency_array[0])
    set_latency_ns = set_pulse_count * float(pulse_latency_array[1])
    latency_ns = np.asarray(reset_latency_ns + set_latency_ns, dtype=np.float64)

    if transition_override is not None:
        latency_ns, energy_nj, transition_digest = transition_override
        latency_ns = latency_ns.reshape(device_level, device_level)
        energy_nj = energy_nj.reshape(device_level, device_level)
    else:
        reset_resistance = np.sqrt(resistance_array[0] * resistance_array[original_idx])
        set_resistance = np.where(
            original_le_target,
            np.sqrt(resistance_array[original_idx] * resistance_array[target_idx]),
            np.sqrt(resistance_array[0] * resistance_array[target_idx]),
        )
        reset_power_w = (float(voltage_array[0]) ** 2) / reset_resistance
        set_power_w = (float(voltage_array[1]) ** 2) / set_resistance
        energy_nj = np.asarray(
            reset_power_w * reset_latency_ns + set_power_w * set_latency_ns,
            dtype=np.float64,
        )
        transition_digest = ""

    digest = hashlib.blake2b(digest_size=16)
    digest.update(cache_key.encode("utf-8"))
    digest.update(np.ascontiguousarray(latency_ns).view(np.uint8))
    digest.update(np.ascontiguousarray(energy_nj).view(np.uint8))
    digest.update(str(transition_digest).encode("utf-8"))
    tables = PhysicalWriteTables(
        sim_config_path=str(sim_config_path),
        xbar_shape=xbar_shape,
        ou_shape=ou_shape,
        device_level=device_level,
        latency_ns=latency_ns,
        energy_nj=energy_nj,
        table_digest=digest.hexdigest(),
    )
    _TABLE_CACHE[cache_key] = tables
    return tables


def _load_transition_override(
    parser: configparser.ConfigParser,
    sim_config_path: str,
    device_level: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Load a canonical JSON source-to-target table when one is declared."""
    try:
        model_reference = parser.get("Final RRAM Model", "Model_JSON")
    except (configparser.NoSectionError, configparser.NoOptionError):
        return None
    model_path = Path(model_reference)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[3] / model_path
    try:
        payload = json.loads(model_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"cannot read transition model {model_path}: {error}",
        ) from error
    transition = payload.get("transition_cost")
    if not isinstance(transition, dict):
        return None
    expected = int(device_level) * int(device_level)
    try:
        latency_s = np.asarray(transition["latency_s"], dtype=np.float64)
        energy_j = np.asarray(transition["energy_j"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"invalid transition model arrays in {model_path}: {error}",
        ) from error
    if latency_s.size != expected or energy_j.size != expected:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"transition model arrays must contain {expected} entries",
        )
    if np.any(latency_s < 0.0) or np.any(energy_j < 0.0):
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            "transition model latency and energy must be non-negative",
        )
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return latency_s * 1e9, energy_j * 1e9, digest


def score_xbar_physical_transition(
    *,
    tables: PhysicalWriteTables,
    original_xbar: np.ndarray,
    target_xbar: np.ndarray,
) -> tuple[float, float]:
    """Score one complete physical xbar using the shared OU aggregation rule."""

    original = _fit_xbar_matrix(original_xbar, tables.xbar_shape)
    target = _fit_xbar_matrix(target_xbar, tables.xbar_shape)
    _validate_device_codes(original, tables, "original xbar")
    _validate_device_codes(target, tables, "target xbar")
    if np.array_equal(original, target):
        return 0.0, 0.0

    latency_cost = tables.latency_ns[original, target]
    energy_cost = tables.energy_nj[original, target]
    # Padding represents absent cells with the erased code.  Their zero-to-zero
    # transition has zero latency and energy, preserving partially sized OUs.
    padded_latency = _pad_to_ou(latency_cost, tables.ou_shape)
    ou_rows, ou_cols = tables.ou_shape
    latency_by_ou = padded_latency.reshape(
        padded_latency.shape[0] // ou_rows,
        ou_rows,
        padded_latency.shape[1] // ou_cols,
        ou_cols,
    )
    latency_ns = float(latency_by_ou.max(axis=3).sum())
    energy_nj = float(energy_cost.sum())
    return latency_ns * 1e-9, energy_nj * 1e-9


class AnnRoundPhysicalWriteLedger:
    """Carry physical device state across generic ANN weighted rounds."""

    def __init__(
        self,
        *,
        sim_config_path: str,
        weight_codes_by_layer: Mapping[str, Mapping[str, Any]],
        preserve_inactive_cells: bool = False,
    ) -> None:
        if not isinstance(weight_codes_by_layer, Mapping):
            raise Phase2Invalid(
                PHYSICAL_WRITE_CODES_MISSING,
                "weight_codes_by_layer must be a layer-id mapping",
            )
        self.tables = load_physical_write_tables(str(sim_config_path))
        self._weight_codes_by_layer = {
            str(layer_id): codes
            for layer_id, codes in weight_codes_by_layer.items()
        }
        self._preserve_inactive_cells = bool(preserve_inactive_cells)
        self._device_state: dict[_PhysicalSlot, np.ndarray] = {}
        self._weighted_round_count = 0

    def score_round(self, *, item_id: str, placement: Any) -> dict[str, Any]:
        """Program one weighted round and advance the retained device state.

        There is deliberately no first-load path.  The first invocation uses
        the same transition loop as every later invocation, with absent state
        interpreted as the erased code-zero state.
        """

        target_by_slot, active_slots, valid_shapes = self._materialize_round_target(placement)
        zero_xbar = np.zeros(self.tables.xbar_shape, dtype=np.int32)
        previous_slots = set(self._device_state)
        # A round only programs slots assigned by that round's placement.  An
        # unassigned xbar is idle and retains its resident conductance state;
        # it is not implicitly erased merely because another round is active.
        transition_slots = sorted(target_by_slot)

        xbar_costs: dict[_PhysicalSlot, tuple[float, float]] = {}
        for slot in transition_slots:
            original = self._device_state.get(slot, zero_xbar)
            target = target_by_slot[slot]
            if self._preserve_inactive_cells:
                rows, cols = valid_shapes[slot]
                latency_s, energy_j = score_xbar_physical_transition(
                    tables=self.tables,
                    original_xbar=original[:rows, :cols],
                    target_xbar=target[:rows, :cols],
                )
            else:
                latency_s, energy_j = score_xbar_physical_transition(
                    tables=self.tables,
                    original_xbar=original,
                    target_xbar=target,
                )
            if latency_s > 0.0 or energy_j > 0.0:
                xbar_costs[slot] = (latency_s, energy_j)

        latency_s, energy_j = _aggregate_round_cost(xbar_costs)
        for slot, target in target_by_slot.items():
            if np.any(target):
                if self._preserve_inactive_cells:
                    rows, cols = valid_shapes[slot]
                    retained = self._device_state.get(slot, zero_xbar).copy()
                    retained[:rows, :cols] = target[:rows, :cols]
                    self._device_state[slot] = retained
                else:
                    self._device_state[slot] = target.copy()
            else:
                self._device_state.pop(slot, None)
        round_index = self._weighted_round_count
        self._weighted_round_count += 1

        return {
            "item_id": str(item_id),
            "round_index": round_index,
            "latency_s": float(latency_s),
            "energy_j": float(energy_j),
            "target_physical_xbar_count": int(len(active_slots)),
            "target_nonzero_xbar_count": int(
                sum(bool(np.any(target)) for target in target_by_slot.values())
            ),
            "transitioned_physical_xbar_count": int(len(xbar_costs)),
            "retained_idle_slot_count": int(len(previous_slots - active_slots)),
            # Kept as a compatibility audit field.  Idle slots are retained,
            # so the scheduler never performs an implicit clear operation.
            "cleared_former_slot_count": 0,
            "table_digest": self.tables.table_digest,
            "state_model": "erased_initial_then_active_slot_overwrite_idle_retained",
            "inactive_cell_policy": (
                "retain_outside_fragment_shape"
                if self._preserve_inactive_cells
                else "program_padded_cells_to_erased_code"
            ),
        }

    def _materialize_round_target(
        self,
        placement: Any,
    ) -> tuple[
        dict[_PhysicalSlot, np.ndarray],
        set[_PhysicalSlot],
        dict[_PhysicalSlot, tuple[int, int]],
    ]:
        fragment_specs = getattr(placement, "fragment_specs_by_id", None)
        fragment_slots = getattr(placement, "fragment_slots_by_id", None)
        if not isinstance(fragment_specs, Mapping) or not isinstance(fragment_slots, Mapping):
            raise Phase2Invalid(
                PHYSICAL_WRITE_LEDGER_INVALID,
                "weighted round placement does not expose exact fragment-to-slot metadata",
            )
        if set(fragment_specs) != set(fragment_slots):
            raise Phase2Invalid(
                PHYSICAL_WRITE_LEDGER_INVALID,
                "fragment specification and physical slot maps disagree",
            )

        target_by_slot: dict[_PhysicalSlot, np.ndarray] = {}
        active_slots: set[_PhysicalSlot] = set()
        valid_shapes: dict[_PhysicalSlot, tuple[int, int]] = {}
        for fragment_id in sorted(fragment_specs):
            fragment = fragment_specs[fragment_id]
            if not isinstance(fragment, MnsimGroupFragment):
                raise Phase2Invalid(
                    PHYSICAL_WRITE_LEDGER_INVALID,
                    f"fragment {fragment_id!r} has invalid physical metadata",
                )
            group_slot = fragment_slots[fragment_id]
            for polarity_index in range(int(fragment.polarity_count)):
                slot = _physical_slot(group_slot, polarity_index)
                if slot in target_by_slot:
                    raise Phase2Invalid(
                        PHYSICAL_WRITE_LEDGER_INVALID,
                        f"physical slot {slot} receives more than one fragment target",
                    )
                target, valid_shape = _fragment_target_matrix(
                    fragment=fragment,
                    polarity_index=polarity_index,
                    weight_codes_by_layer=self._weight_codes_by_layer,
                    tables=self.tables,
                    return_shape=True,
                )
                target_by_slot[slot] = target
                valid_shapes[slot] = valid_shape
                active_slots.add(slot)
        return target_by_slot, active_slots, valid_shapes


def _fragment_target_matrix(
    *,
    fragment: MnsimGroupFragment,
    polarity_index: int,
    weight_codes_by_layer: Mapping[str, Mapping[str, Any]],
    tables: PhysicalWriteTables,
    return_shape: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[int, int]]:
    try:
        layer_codes = weight_codes_by_layer[str(fragment.layer_id)]
    except KeyError as error:
        raise Phase2Invalid(
            PHYSICAL_WRITE_CODES_MISSING,
            f"missing checkpoint device codes for layer {fragment.layer_id}",
        ) from error
    if not isinstance(layer_codes, Mapping):
        raise Phase2Invalid(
            PHYSICAL_WRITE_CODES_MISSING,
            f"checkpoint device codes for {fragment.layer_id} must be a mapping",
        )

    input_block_index = _input_block_index(fragment, tables.xbar_shape[0])
    plane_index = int(fragment.weight_plane_index)
    if int(fragment.polarity_count) == 2:
        polarity_name = "positive" if polarity_index == 0 else "negative"
        key = f"split{input_block_index}_weight{plane_index}_{polarity_name}"
    elif int(fragment.polarity_count) == 1:
        key = f"split{input_block_index}_weight{plane_index}"
    else:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"fragment {fragment.fragment_id} has unsupported polarity count {fragment.polarity_count}",
        )
    if key not in layer_codes:
        raise Phase2Invalid(
            PHYSICAL_WRITE_CODES_MISSING,
            f"layer {fragment.layer_id} is missing checkpoint code matrix {key}",
        )

    array = _integer_codes(layer_codes[key], label=f"{fragment.layer_id}:{key}")
    if array.ndim < 2:
        raise Phase2Invalid(
            PHYSICAL_WRITE_CODES_MISSING,
            f"{fragment.layer_id}:{key} must have an output and input dimension",
        )
    output_start, output_end = (int(value) for value in fragment.output_range)
    if output_start < 0 or output_end <= output_start or output_end > array.shape[0]:
        raise Phase2Invalid(
            PHYSICAL_WRITE_CODES_MISSING,
            f"{fragment.layer_id}:{key} cannot select output range {fragment.output_range}",
        )

    # MNSIM quantizers expose [output, flattened-input] matrices.  A physical
    # xbar is [input-row, output-column], hence the transpose below.
    output_major = array[output_start:output_end].reshape(output_end - output_start, -1)
    matrix = np.asarray(output_major.T, dtype=np.int32)
    if matrix.shape[0] > tables.xbar_shape[0] or matrix.shape[1] > tables.xbar_shape[1]:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            (
                f"fragment {fragment.fragment_id} target shape {matrix.shape} exceeds "
                f"xbar shape {tables.xbar_shape}"
            ),
        )
    _validate_device_codes(matrix, tables, f"fragment {fragment.fragment_id}")
    target = np.zeros(tables.xbar_shape, dtype=np.int32)
    target[: matrix.shape[0], : matrix.shape[1]] = matrix
    return (target, tuple(int(value) for value in matrix.shape)) if return_shape else target


def _input_block_index(fragment: MnsimGroupFragment, xbar_rows: int) -> int:
    input_start, input_end = (int(value) for value in fragment.input_range)
    if input_start < 0 or input_end <= input_start:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"fragment {fragment.fragment_id} has invalid input range {fragment.input_range}",
        )
    del xbar_rows
    explicit_index = getattr(fragment, "input_block_index", None)
    if isinstance(explicit_index, int) and explicit_index >= 0:
        return int(explicit_index)
    raise Phase2Invalid(
        PHYSICAL_WRITE_LEDGER_INVALID,
        f"fragment {fragment.fragment_id} needs a non-negative input-block index",
    )


def _physical_slot(group_slot: Any, polarity_index: int) -> _PhysicalSlot:
    if not isinstance(group_slot, Mapping):
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"physical group slot must be a mapping, got {type(group_slot).__name__}",
        )
    try:
        return (
            _nonnegative_int(group_slot["tile_id"], "tile_id"),
            _nonnegative_int(group_slot["pe_id"], "pe_id"),
            _nonnegative_int(group_slot["group_id"], "group_id"),
            _nonnegative_int(polarity_index, "polarity_index"),
        )
    except KeyError as error:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"physical group slot is missing {error.args[0]}",
        ) from error


def _aggregate_round_cost(
    xbar_costs: Mapping[_PhysicalSlot, tuple[float, float]],
) -> tuple[float, float]:
    pe_costs: dict[tuple[int, int], list[float]] = {}
    for (tile_id, pe_id, _group_id, _polarity_id), (latency_s, energy_j) in xbar_costs.items():
        aggregate = pe_costs.setdefault((tile_id, pe_id), [0.0, 0.0])
        aggregate[0] = max(aggregate[0], float(latency_s))
        aggregate[1] += float(energy_j)

    tile_costs: dict[int, list[float]] = {}
    for (tile_id, _pe_id), (latency_s, energy_j) in pe_costs.items():
        aggregate = tile_costs.setdefault(tile_id, [0.0, 0.0])
        aggregate[0] = max(aggregate[0], float(latency_s))
        aggregate[1] += float(energy_j)

    return (
        max((cost[0] for cost in tile_costs.values()), default=0.0),
        sum((cost[1] for cost in tile_costs.values()), 0.0),
    )


def _fit_xbar_matrix(matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.int32)
    if source.ndim != 2:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"xbar state must be rank-2, got shape {source.shape}",
        )
    target = np.zeros(shape, dtype=np.int32)
    rows = min(source.shape[0], shape[0])
    cols = min(source.shape[1], shape[1])
    target[:rows, :cols] = source[:rows, :cols]
    return target


def _pad_to_ou(cost: np.ndarray, ou_shape: tuple[int, int]) -> np.ndarray:
    row_pad = (-cost.shape[0]) % ou_shape[0]
    col_pad = (-cost.shape[1]) % ou_shape[1]
    if row_pad == 0 and col_pad == 0:
        return cost
    return np.pad(cost, ((0, row_pad), (0, col_pad)), mode="constant")


def _validate_device_codes(matrix: np.ndarray, tables: PhysicalWriteTables, label: str) -> None:
    if matrix.size == 0:
        return
    low = int(np.min(matrix))
    high = int(np.max(matrix))
    if low < 0 or high >= tables.device_level:
        raise Phase2Invalid(
            PHYSICAL_WRITE_CODES_MISSING,
            f"{label} has codes [{low}, {high}] outside [0, {tables.device_level})",
        )


def _integer_codes(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
            raise Phase2Invalid(
                PHYSICAL_WRITE_CODES_MISSING,
                f"{label} must contain finite integer device codes",
            )
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise Phase2Invalid(
                PHYSICAL_WRITE_CODES_MISSING,
                f"{label} must contain integer device codes",
            )
        array = rounded
    return np.asarray(array, dtype=np.int32)


def _parse_pair(parser: configparser.ConfigParser, section: str, option: str) -> tuple[int, int]:
    values = _config_value(parser, section, option).replace(",", " ").split()
    if len(values) != 2:
        raise _invalid_config(f"{section}.{option} must contain two positive integers")
    return (
        _positive_int(values[0], f"{section}.{option}[0]"),
        _positive_int(values[1], f"{section}.{option}[1]"),
    )


def _float_vector(parser: configparser.ConfigParser, section: str, option: str) -> tuple[float, ...]:
    raw = _config_value(parser, section, option)
    try:
        return tuple(float(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise _invalid_config(f"{section}.{option} must be a comma-separated numeric vector") from error


def _config_value(parser: configparser.ConfigParser, section: str, option: str) -> str:
    try:
        return parser.get(section, option)
    except configparser.Error as error:
        raise _invalid_config(f"missing {section}.{option}") from error


def _positive_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise _invalid_config(f"{label} must be a positive integer") from error
    if result <= 0:
        raise _invalid_config(f"{label} must be a positive integer")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise Phase2Invalid(PHYSICAL_WRITE_LEDGER_INVALID, f"{label} must be a non-negative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise Phase2Invalid(
            PHYSICAL_WRITE_LEDGER_INVALID,
            f"{label} must be a non-negative integer",
        ) from error
    if result < 0:
        raise Phase2Invalid(PHYSICAL_WRITE_LEDGER_INVALID, f"{label} must be a non-negative integer")
    return result


def _invalid_config(detail: str) -> Phase2Invalid:
    return Phase2Invalid(PHYSICAL_WRITE_LEDGER_INVALID, detail)


__all__ = [
    "AnnRoundPhysicalWriteLedger",
    "PHYSICAL_WRITE_CODES_MISSING",
    "PHYSICAL_WRITE_LEDGER_INVALID",
    "PhysicalWriteTables",
    "load_physical_write_tables",
    "score_xbar_physical_transition",
]
