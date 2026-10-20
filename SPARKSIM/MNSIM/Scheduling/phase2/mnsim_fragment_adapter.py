"""Shape-only original-MNSIM group-fragment inventory and range selection."""

from __future__ import annotations

import configparser
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
import re
from typing import Any
from MNSIM.Scheduling.depthwise import is_depthwise, depthwise_block_size

from MNSIM.Scheduling.device_precision import (
    DevicePrecisionError,
    device_bits_from_level,
)
from MNSIM.Scheduling.phase2.types import (
    DUPLICATE_FRAGMENT_CONTRIBUTION,
    INVALID_PHYSICAL_XBAR_RANGE,
    PHYSICAL_XBAR_UNIT_MISMATCH,
    PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH,
    SELECTED_FRAGMENT_COUNT_MISMATCH,
    FragmentInventory,
    MnsimFragmentLayout,
    MnsimGroupFragment,
    Phase2Invalid,
)
from MNSIM.Scheduling.round_types import PrepassRef, SelectedEntry


_INVALID_FRAGMENT_CONFIG = "INVALID_MNSIM_FRAGMENT_CONFIG"
_INVALID_FRAGMENT_REF = "INVALID_MNSIM_FRAGMENT_REF"
_INVALID_FRAGMENT_SHAPE = "INVALID_MNSIM_FRAGMENT_SHAPE"
_PHYSICAL_XBAR_RESOURCE_VERSION = "mnsim20_physical_xbar_v2"
_SUPPORTED_LAYER_TYPES = {"conv", "fc", "fc_like_projection"}
_SIM_CONFIG_FINGERPRINT_PATTERN = re.compile(
    r"xbar=(?P<xbar_row>[1-9]\d*)x(?P<xbar_col>[1-9]\d*);"
    r"device=(?P<device_level>[1-9]\d*);"
    r"weight=(?P<weight_precision>[1-9]\d*);"
    r"polarity=(?P<polarity_count>[1-9]\d*)"
)


@dataclass(frozen=True)
class _FragmentHardware:
    xbar_row: int
    xbar_col: int
    device_level: int
    device_bits: int
    weight_precision: int
    polarity_count: int


class MnsimFragmentAdapter:
    def __init__(
        self,
        sim_config_path: str,
        required_xbars_by_layer: Mapping[str, int],
    ) -> None:
        self._hardware = _read_fragment_hardware(sim_config_path)
        if not isinstance(required_xbars_by_layer, Mapping):
            raise Phase2Invalid(
                PHYSICAL_XBAR_UNIT_MISMATCH,
                "required_xbars_by_layer must be a layer-id mapping",
            )
        self._required_xbars_by_layer: dict[str, int] = {}
        self._inventory_cache: dict[tuple[Any, ...], FragmentInventory] = {}
        for raw_layer_id, raw_required_xbars in required_xbars_by_layer.items():
            layer_id = str(raw_layer_id)
            if layer_id in self._required_xbars_by_layer:
                raise Phase2Invalid(
                    PHYSICAL_XBAR_UNIT_MISMATCH,
                    f"duplicate required-xbar descriptor for layer {layer_id}",
                )
            self._required_xbars_by_layer[layer_id] = _positive_int_value(
                raw_required_xbars,
                PHYSICAL_XBAR_UNIT_MISMATCH,
                f"{layer_id} required physical xbar count",
            )

    def inventory_for(self, entry: SelectedEntry) -> FragmentInventory:
        _validate_entry_ref(entry)
        _validate_prepass_identity(entry.prepass_ref, self._hardware)
        cache_key = _inventory_cache_key(entry)
        inventory = self._inventory_cache.get(cache_key)
        if inventory is None:
            inventory = _build_fragment_inventory(
                entry,
                self._hardware,
                self._required_xbars_by_layer,
            )
            self._inventory_cache[cache_key] = inventory
        return inventory

    def select_entry(self, entry: SelectedEntry) -> MnsimFragmentLayout:
        inventory = self.inventory_for(entry)
        return _select_fragment_layout(entry, inventory, self._hardware)


def _read_fragment_hardware(sim_config_path: str) -> _FragmentHardware:
    parser = configparser.ConfigParser()
    try:
        loaded_paths = parser.read(sim_config_path, encoding="UTF-8")
    except (OSError, configparser.Error) as exc:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_CONFIG,
            f"could not read SimConfig path {sim_config_path}: {exc}",
        ) from exc
    if not loaded_paths:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_CONFIG,
            f"could not read SimConfig path {sim_config_path}",
        )

    xbar_size_value = _config_value(parser, "Crossbar level", "Xbar_Size")
    xbar_size_parts = xbar_size_value.replace(",", " ").split()
    if len(xbar_size_parts) != 2:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_CONFIG,
            f"Xbar_Size must contain row and column dimensions, got {xbar_size_value!r}",
        )
    xbar_row = _positive_int_value(
        xbar_size_parts[0],
        _INVALID_FRAGMENT_CONFIG,
        "Xbar_Size row dimension",
    )
    xbar_col = _positive_int_value(
        xbar_size_parts[1],
        _INVALID_FRAGMENT_CONFIG,
        "Xbar_Size column dimension",
    )
    device_level = _positive_int_value(
        _config_value(parser, "Device level", "Device_Level"),
        _INVALID_FRAGMENT_CONFIG,
        "Device_Level",
    )
    try:
        device_bits = device_bits_from_level(device_level)
    except DevicePrecisionError as exc:
        raise Phase2Invalid(_INVALID_FRAGMENT_CONFIG, str(exc)) from exc
    polarity_count = _positive_int_value(
        _config_value(parser, "Process element level", "Xbar_Polarity"),
        _INVALID_FRAGMENT_CONFIG,
        "Xbar_Polarity",
    )
    if polarity_count not in (1, 2):
        raise Phase2Invalid(
            _INVALID_FRAGMENT_CONFIG,
            f"Xbar_Polarity must be 1 or 2, got {polarity_count}",
        )
    weight_precision = _positive_int_value(
        _config_value(parser, "Algorithm Configuration", "Weight_precision"),
        _INVALID_FRAGMENT_CONFIG,
        "Algorithm Configuration.Weight_precision",
    )
    return _FragmentHardware(
        xbar_row=xbar_row,
        xbar_col=xbar_col,
        device_level=device_level,
        device_bits=device_bits,
        weight_precision=weight_precision,
        polarity_count=polarity_count,
    )


def _config_value(
    parser: configparser.ConfigParser,
    section: str,
    option: str,
) -> str:
    try:
        return parser.get(section, option)
    except configparser.Error as exc:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_CONFIG,
            f"missing or invalid SimConfig field {section}.{option}",
        ) from exc


def _int_value(value: Any, code: str, detail: str) -> int:
    if isinstance(value, bool):
        raise Phase2Invalid(code, f"{detail} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    raise Phase2Invalid(code, f"{detail} must be an integer, got {value!r}")


def _positive_int_value(value: Any, code: str, detail: str) -> int:
    result = _int_value(value, code, detail)
    if result <= 0:
        raise Phase2Invalid(code, f"{detail} must be positive, got {result}")
    return result


def _selected_entry_integral_value(value: Any, code: str, detail: str) -> int:
    """Accept only the concrete integer values frozen into a Phase 1 plan."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise Phase2Invalid(code, f"{detail} must be an integer, got {value!r}")
    return int(value)


def _shape_positive_int(shape: Mapping[str, Any], key: str, ref: PrepassRef) -> int:
    if key not in shape:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_SHAPE,
            f"{ref.layer_id} missing shape field {key}",
        )
    return _positive_int_value(
        shape[key],
        _INVALID_FRAGMENT_SHAPE,
        f"{ref.layer_id} shape field {key}",
    )


def _validate_entry_ref(entry: SelectedEntry) -> Mapping[str, Any]:
    ref = entry.prepass_ref
    if not isinstance(ref, PrepassRef):
        raise Phase2Invalid(
            _INVALID_FRAGMENT_REF,
            f"{entry.layer_id} selected entry has no valid PrepassRef",
        )
    if entry.layer_id != ref.layer_id:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_REF,
            (
                f"selected entry layer_id {entry.layer_id!r} does not match "
                f"PrepassRef layer_id {ref.layer_id!r}"
            ),
        )
    if entry.layer_type != ref.layer_type:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_REF,
            (
                f"selected entry layer_type {entry.layer_type!r} does not match "
                f"PrepassRef layer_type {ref.layer_type!r}"
            ),
        )
    if entry.layer_type not in _SUPPORTED_LAYER_TYPES:
        raise Phase2Invalid(
            _INVALID_FRAGMENT_REF,
            f"{entry.layer_id} has unsupported weighted layer type {entry.layer_type!r}",
        )
    if not isinstance(ref.shape_signature, Mapping):
        raise Phase2Invalid(
            _INVALID_FRAGMENT_REF,
            f"{entry.layer_id} PrepassRef shape_signature must be a mapping",
        )
    return ref.shape_signature


def _inventory_cache_key(entry: SelectedEntry) -> tuple[Any, ...]:
    """Identify immutable layer metadata while excluding the selected range."""
    shape = _validate_entry_ref(entry)
    ref = entry.prepass_ref
    return (
        entry.layer_id,
        entry.layer_type,
        ref.resource_version,
        ref.sim_config_fingerprint,
        ref.layer_config_fingerprint,
        _freeze_cache_value(shape),
    )


def _freeze_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_cache_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _validate_prepass_identity(
    ref: PrepassRef,
    hardware: _FragmentHardware,
) -> None:
    if ref.resource_version != _PHYSICAL_XBAR_RESOURCE_VERSION:
        raise Phase2Invalid(
            PHYSICAL_XBAR_UNIT_MISMATCH,
            (
                f"{ref.layer_id} PrepassRef resource_version "
                f"{ref.resource_version!r} does not match "
                f"{_PHYSICAL_XBAR_RESOURCE_VERSION!r}"
            ),
        )
    if not isinstance(ref.sim_config_fingerprint, str):
        fingerprint_match = None
    else:
        fingerprint_match = _SIM_CONFIG_FINGERPRINT_PATTERN.fullmatch(
            ref.sim_config_fingerprint
        )
    if fingerprint_match is None:
        raise Phase2Invalid(
            PHYSICAL_XBAR_UNIT_MISMATCH,
            (
                f"{ref.layer_id} PrepassRef has missing or malformed canonical "
                f"SimConfig fingerprint {ref.sim_config_fingerprint!r}"
            ),
        )

    fingerprint_hardware = {
        key: int(value)
        for key, value in fingerprint_match.groupdict().items()
    }
    expected_hardware = {
        "xbar_row": hardware.xbar_row,
        "xbar_col": hardware.xbar_col,
        "device_level": hardware.device_level,
        "weight_precision": hardware.weight_precision,
        "polarity_count": hardware.polarity_count,
    }
    mismatches = [
        f"{key}={fingerprint_hardware[key]} (SimConfig {expected_value})"
        for key, expected_value in expected_hardware.items()
        if fingerprint_hardware[key] != expected_value
    ]
    if mismatches:
        raise Phase2Invalid(
            PHYSICAL_XBAR_UNIT_MISMATCH,
            (
                f"{ref.layer_id} PrepassRef SimConfig fingerprint disagrees with "
                f"the real SimConfig: {', '.join(mismatches)}"
            ),
        )


def _weight_plane_count(
    shape: Mapping[str, Any],
    ref: PrepassRef,
    hardware: _FragmentHardware,
) -> int:
    if "Weightbit" in shape:
        native_weightbit = _shape_positive_int(shape, "Weightbit", ref)
        precision_bits = (
            native_weightbit - 1
            if hardware.polarity_count == 2
            else native_weightbit
        )
    else:
        # Phase 1's SimConfig fallback is already the effective magnitude
        # precision. Only a native MNSIM Weightbit carries a sign bit here.
        native_weightbit = None
        precision_bits = hardware.weight_precision

    cell_bits = hardware.device_bits
    plane_count = (precision_bits + cell_bits - 1) // cell_bits
    if plane_count <= 0:
        precision_label = (
            f"Weightbit={native_weightbit}"
            if native_weightbit is not None
            else f"Weight_precision={hardware.weight_precision}"
        )
        raise Phase2Invalid(
            _INVALID_FRAGMENT_SHAPE,
            (
                f"{ref.layer_id} {precision_label} with polarity "
                f"{hardware.polarity_count} produces no weight planes"
            ),
        )
    return plane_count


def _fragment_dimensions(
    entry: SelectedEntry,
    shape: Mapping[str, Any],
    hardware: _FragmentHardware,
) -> tuple[int, int, int, int]:
    ref = entry.prepass_ref
    if entry.layer_type == "conv":
        full_input_count = _shape_positive_int(shape, "Inputchannel", ref)
        full_output_count = _shape_positive_int(shape, "Outputchannel", ref)
        kernel_size = _shape_positive_int(shape, "Kernelsize", ref)
        kernel_area = kernel_size * kernel_size
        if kernel_area > hardware.xbar_row:
            raise Phase2Invalid(
                _INVALID_FRAGMENT_SHAPE,
                (
                    f"{entry.layer_id} Kernelsize*Kernelsize={kernel_area} exceeds "
                    f"xbar row count {hardware.xbar_row}"
                ),
            )
        input_block_size = hardware.xbar_row // kernel_area
        if is_depthwise(shape):
            input_block_size = depthwise_block_size(shape, hardware.xbar_row, hardware.xbar_col)
    else:
        full_input_count = _shape_positive_int(shape, "Infeature", ref)
        full_output_count = _shape_positive_int(shape, "Outfeature", ref)
        input_block_size = hardware.xbar_row
    return (
        full_input_count,
        full_output_count,
        input_block_size,
        hardware.xbar_col,
    )


def _build_fragment_inventory(
    entry: SelectedEntry,
    hardware: _FragmentHardware,
    required_xbars_by_layer: Mapping[str, int],
) -> FragmentInventory:
    shape = _validate_entry_ref(entry)
    _validate_prepass_identity(entry.prepass_ref, hardware)
    if entry.layer_id not in required_xbars_by_layer:
        raise Phase2Invalid(
            PHYSICAL_XBAR_UNIT_MISMATCH,
            f"{entry.layer_id} has no Phase 1 required physical-xbar descriptor",
        )
    required_xbar_count = required_xbars_by_layer[entry.layer_id]
    plane_count = _weight_plane_count(shape, entry.prepass_ref, hardware)
    (
        full_input_count,
        full_output_count,
        input_block_size,
        output_block_size,
    ) = _fragment_dimensions(entry, shape, hardware)

    input_block_count = (
        full_input_count + input_block_size - 1
    ) // input_block_size
    output_block_count = (
        full_output_count + output_block_size - 1
    ) // output_block_size
    depthwise = entry.layer_type == "conv" and is_depthwise(shape)
    if depthwise:
        output_block_count = 1
    physical_xbar_count = (
        input_block_count
        * output_block_count
        * plane_count
        * hardware.polarity_count
    )
    if physical_xbar_count != required_xbar_count:
        raise Phase2Invalid(
            PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH,
            (
                f"{entry.layer_id} MNSIM fragment inventory has {physical_xbar_count} "
                f"physical xbars but Phase 1 required {required_xbar_count}"
            ),
        )

    fragments: list[MnsimGroupFragment] = []
    for input_block_index, input_start in enumerate(
        range(0, full_input_count, input_block_size)
    ):
        input_end = min(input_start + input_block_size, full_input_count)
        output_ranges = [(input_start, input_end)] if depthwise else [
            (start, min(start + output_block_size, full_output_count))
            for start in range(0, full_output_count, output_block_size)]
        for output_start, output_end in output_ranges:
            for plane_index in range(plane_count):
                fragment_id = (
                    f"{entry.layer_id}|in:{input_start}:{input_end}"
                    f"|out:{output_start}:{output_end}|plane:{plane_index}"
                )
                fragments.append(
                    MnsimGroupFragment(
                        fragment_id=fragment_id,
                        layer_id=entry.layer_id,
                        layer_type=entry.layer_type,
                        input_range=(input_start, input_end),
                        output_range=(output_start, output_end),
                        full_input_range=(0, full_input_count),
                        full_output_range=(0, full_output_count),
                        weight_plane_index=plane_index,
                        polarity_count=hardware.polarity_count,
                        physical_xbar_count=hardware.polarity_count,
                        input_block_index=input_block_index,
                    )
                )

    return FragmentInventory(
        layer_id=entry.layer_id,
        required_xbar_count=required_xbar_count,
        fragments=tuple(fragments),
    )


def _physical_interval(entry: SelectedEntry, inventory: FragmentInventory) -> tuple[int, int]:
    start = _selected_entry_integral_value(
        entry.xbar_range_start,
        INVALID_PHYSICAL_XBAR_RANGE,
        f"{entry.layer_id} physical xbar range start",
    )
    end = _selected_entry_integral_value(
        entry.xbar_range_end,
        INVALID_PHYSICAL_XBAR_RANGE,
        f"{entry.layer_id} physical xbar range end",
    )
    if start < 0 or end <= start or end > inventory.required_xbar_count:
        raise Phase2Invalid(
            INVALID_PHYSICAL_XBAR_RANGE,
            (
                f"{entry.layer_id} physical xbar range [{start}, {end}) is not a "
                f"non-empty interval inside [0, {inventory.required_xbar_count})"
            ),
        )
    return start, end


def _public_native_layer_info(
    entry: SelectedEntry,
    hardware: _FragmentHardware,
) -> dict[str, Any]:
    native_layer_info: dict[str, Any] = {}
    for key, value in entry.prepass_ref.shape_signature.items():
        if not isinstance(key, str):
            raise Phase2Invalid(
                _INVALID_FRAGMENT_REF,
                f"{entry.layer_id} has non-string shape field {key!r}",
            )
        if not key.startswith("_"):
            native_layer_info[key] = deepcopy(value)
    fallback_weightbit = (
        hardware.weight_precision + 1
        if hardware.polarity_count == 2
        else hardware.weight_precision
    )
    native_layer_info.setdefault("Weightbit", fallback_weightbit)
    native_layer_info["type"] = entry.layer_type
    return native_layer_info


def _select_fragment_layout(
    entry: SelectedEntry,
    inventory: FragmentInventory,
    hardware: _FragmentHardware,
) -> MnsimFragmentLayout:
    start, end = _physical_interval(entry, inventory)
    if start % hardware.polarity_count or end % hardware.polarity_count:
        raise Phase2Invalid(
            INVALID_PHYSICAL_XBAR_RANGE,
            (
                f"{entry.layer_id} physical xbar range [{start}, {end}) cuts a "
                f"polarity-{hardware.polarity_count} fragment payload"
            ),
        )

    fragment_start = start // hardware.polarity_count
    fragment_end = end // hardware.polarity_count
    fragments = inventory.fragments[fragment_start:fragment_end]
    fragment_ids = [fragment.fragment_id for fragment in fragments]
    if len(fragment_ids) != len(set(fragment_ids)):
        raise Phase2Invalid(
            DUPLICATE_FRAGMENT_CONTRIBUTION,
            f"{entry.layer_id} selected layout contains duplicate fragment ids",
        )

    if any(
        fragment.polarity_count != hardware.polarity_count
        or fragment.physical_xbar_count != hardware.polarity_count
        for fragment in fragments
    ):
        raise Phase2Invalid(
            PHYSICAL_XBAR_UNIT_MISMATCH,
            f"{entry.layer_id} selected fragments do not use complete polarity payloads",
        )

    selected_count = sum(fragment.physical_xbar_count for fragment in fragments)
    interval_count = end - start
    used_xbars = _selected_entry_integral_value(
        entry.used_xbars,
        SELECTED_FRAGMENT_COUNT_MISMATCH,
        f"{entry.layer_id} used_xbars",
    )
    if selected_count != interval_count or used_xbars != interval_count:
        raise Phase2Invalid(
            SELECTED_FRAGMENT_COUNT_MISMATCH,
            (
                f"{entry.layer_id} selected fragment count {selected_count}, "
                f"used_xbars {used_xbars}, and range length {interval_count} disagree"
            ),
        )

    return MnsimFragmentLayout(
        layer_id=entry.layer_id,
        layer_type=entry.layer_type,
        fragments=fragments,
        native_layer_info=_public_native_layer_info(entry, hardware),
        active_group_count=len(fragments),
        active_physical_xbar_count=selected_count,
    )


__all__ = ["MnsimFragmentAdapter"]
