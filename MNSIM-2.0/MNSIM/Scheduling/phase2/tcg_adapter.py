"""Round-resident, weighted-only adapters for original MNSIM TCG placement."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Integral
from collections.abc import Mapping
from typing import Any

import numpy as np

from MNSIM.Scheduling.phase2.netstruct_builder import build_netstruct
from MNSIM.Scheduling.phase2.types import (
    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
    MnsimFragmentLayout,
    Phase2Invalid,
)


_PLACEMENT_INCONSISTENT = "TCG_ROUND_PLACEMENT_INCONSISTENT"
_LEGAL_MAPPING_TYPES = {"conv", "fc"}


def _snapshot_value(value: Any) -> Any:
    """Convert placement metadata to deterministic JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _snapshot_value(value.to_dict())
    if isinstance(value, Mapping):
        return {
            str(key): _snapshot_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_snapshot_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported placement snapshot value {type(value).__name__}")


class FrozenTCG:
    """Read-only TCG-shaped view restored from a Stage-1 placement snapshot."""

    def __init__(self, sim_config_path: str, snapshot: Mapping[str, Any]) -> None:
        from MNSIM.Hardware_Model.Tile import tile

        self.sim_config_path = str(sim_config_path)
        self.tile = tile(self.sim_config_path)
        for name in (
            "layer_tileinfo",
            "max_inbuf_size",
            "max_outbuf_size",
            "global_buf_size",
            "global_adder_num",
            "global_multiplier_num",
            "global_adder_bitwidth",
            "global_multiplier_bitwidth",
            "global_data_size",
            "inLayer_distance",
            "transLayer_distance",
            "aggregate_arg",
            "buffer_capacity_evidence",
            "mnsim_group_capacity",
            "physical_xbar_capacity",
            "used_tile_num",
            "layer_num",
        ):
            if name in snapshot:
                setattr(self, name, deepcopy(snapshot[name]))
        self.net = deepcopy(snapshot.get("net", []))

    def mapping_net(self) -> None:
        """A restored placement must never invoke a mapping pass."""

    def calculate_transfer_distance(self) -> None:
        """Transfer distances are already frozen in the snapshot."""


def serialize_round_placement(placement: "RoundPlacementPlan") -> dict[str, Any]:
    """Serialize only the placement contract required by Phase 2."""
    if not isinstance(placement, RoundPlacementPlan):
        raise TypeError("placement must be a RoundPlacementPlan")
    tcg = placement.tcg
    tcg_snapshot = {
        name: _snapshot_value(getattr(tcg, name))
        for name in (
            "layer_tileinfo",
            "max_inbuf_size",
            "max_outbuf_size",
            "global_buf_size",
            "global_adder_num",
            "global_multiplier_num",
            "global_adder_bitwidth",
            "global_multiplier_bitwidth",
            "global_data_size",
            "inLayer_distance",
            "transLayer_distance",
            "aggregate_arg",
            "buffer_capacity_evidence",
            "mnsim_group_capacity",
            "physical_xbar_capacity",
            "used_tile_num",
            "layer_num",
        )
        if hasattr(tcg, name)
    }
    tcg_snapshot["net"] = _snapshot_value(getattr(tcg, "net", placement.mapping_netstruct))
    return {
        "item_id": str(placement.item_id),
        "sim_config_path": str(placement.sim_config_path),
        "mapping_layer_ids_by_step": {
            str(key): [int(value) for value in values]
            for key, values in placement.mapping_layer_ids_by_step.items()
        },
        "mapping_summary": _snapshot_value(placement.mapping_summary),
        "mapped_area_um2": float(placement.mapped_area_um2),
        "fragment_specs_by_id": {
            str(key): _snapshot_value(value)
            for key, value in placement.fragment_specs_by_id.items()
        },
        "fragment_slots_by_id": {
            str(key): {
                "tile_id": int(value["tile_id"]),
                "pe_id": int(value["pe_id"]),
                "group_id": int(value["group_id"]),
            }
            for key, value in placement.fragment_slots_by_id.items()
        },
        "output_owner_shards_by_layer": _snapshot_value(
            placement.output_owner_shards_by_layer
        ),
        "tcg": tcg_snapshot,
    }


def deserialize_round_placement(
    snapshot: Mapping[str, Any], *, sim_config_path: str | None = None
) -> "RoundPlacementPlan":
    """Restore a placement without constructing or invoking original TCG."""
    if not isinstance(snapshot, Mapping):
        raise Phase2Invalid(_PLACEMENT_INCONSISTENT, "placement snapshot must be a mapping")
    item_id = str(snapshot.get("item_id", ""))
    if not item_id:
        raise Phase2Invalid(_PLACEMENT_INCONSISTENT, "placement snapshot is missing item_id")
    resolved_config = str(sim_config_path or snapshot.get("sim_config_path", ""))
    if not resolved_config:
        raise Phase2Invalid(_PLACEMENT_INCONSISTENT, f"{item_id}: snapshot is missing SimConfig path")
    raw_specs = snapshot.get("fragment_specs_by_id")
    if not isinstance(raw_specs, Mapping):
        raise Phase2Invalid(_PLACEMENT_INCONSISTENT, f"{item_id}: snapshot is missing fragment specs")
    from MNSIM.Scheduling.phase2.types import MnsimGroupFragment

    fragment_specs = {
        str(key): MnsimGroupFragment(
            fragment_id=str(value["fragment_id"]),
            layer_id=str(value["layer_id"]),
            layer_type=str(value["layer_type"]),
            input_range=tuple(int(item) for item in value["input_range"]),
            output_range=tuple(int(item) for item in value["output_range"]),
            full_input_range=tuple(int(item) for item in value["full_input_range"]),
            full_output_range=tuple(int(item) for item in value["full_output_range"]),
            weight_plane_index=int(value["weight_plane_index"]),
            polarity_count=int(value["polarity_count"]),
            physical_xbar_count=int(value["physical_xbar_count"]),
            input_block_index=int(value.get("input_block_index", 0)),
        )
        for key, value in raw_specs.items()
    }
    raw_slots = snapshot.get("fragment_slots_by_id")
    if not isinstance(raw_slots, Mapping):
        raise Phase2Invalid(_PLACEMENT_INCONSISTENT, f"{item_id}: snapshot is missing fragment slots")
    fragment_slots = {
        str(key): {
            "tile_id": int(value["tile_id"]),
            "pe_id": int(value["pe_id"]),
            "group_id": int(value["group_id"]),
        }
        for key, value in raw_slots.items()
    }
    if set(fragment_specs) != set(fragment_slots):
        raise Phase2Invalid(_PLACEMENT_INCONSISTENT, f"{item_id}: fragment specs and slots disagree")
    tcg_snapshot = snapshot.get("tcg")
    if not isinstance(tcg_snapshot, Mapping):
        raise Phase2Invalid(_PLACEMENT_INCONSISTENT, f"{item_id}: snapshot is missing TCG view")
    return RoundPlacementPlan(
        item_id=item_id,
        sim_config_path=resolved_config,
        tcg=FrozenTCG(resolved_config, tcg_snapshot),
        mapping_netstruct=deepcopy(tcg_snapshot.get("net", [])),
        mapping_layer_ids_by_step={
            str(key): [int(item) for item in values]
            for key, values in snapshot.get("mapping_layer_ids_by_step", {}).items()
        },
        mapping_summary=deepcopy(dict(snapshot.get("mapping_summary", {}))),
        mapped_area_um2=float(snapshot.get("mapped_area_um2", 0.0)),
        fragment_specs_by_id=fragment_specs,
        fragment_slots_by_id=fragment_slots,
        output_owner_shards_by_layer={
            str(layer): {int(tile): int(count) for tile, count in shards.items()}
            for layer, shards in snapshot.get("output_owner_shards_by_layer", {}).items()
        },
    )


@dataclass
class WeightedScope:
    step_id: str
    scope_index: int
    layer_info: dict[str, Any]
    fragment_layout: MnsimFragmentLayout

    @property
    def active_physical_xbar_count(self) -> int:
        return self.fragment_layout.active_physical_xbar_count

    @property
    def active_group_count(self) -> int:
        return self.fragment_layout.active_group_count


class ReindexedTCGPlacementSlice:
    """A one-step view of an already resident original-MNSIM placement."""

    def __init__(
        self,
        placement: "RoundPlacementPlan",
        netstruct: list[Any],
        source_mapping_layer_ids: list[int],
        *,
        reindexed_tileinfo: list[dict[str, Any]] | None = None,
    ) -> None:
        self.placement = placement
        self.net = netstruct
        self.layer_num = len(netstruct)
        self.tile = placement.tcg.tile
        self.source_mapping_layer_ids = list(source_mapping_layer_ids)
        if self.layer_num != len(self.source_mapping_layer_ids):
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                (
                    f"{placement.item_id}: slice netstruct has {self.layer_num} layers but "
                    f"{len(self.source_mapping_layer_ids)} source mapping ids"
                ),
            )
        if reindexed_tileinfo is None:
            self.tileinfo = [
                _reindexed_tileinfo(
                    placement.tcg.layer_tileinfo[source_layer_id],
                    local_layer_id,
                )
                for local_layer_id, source_layer_id in enumerate(self.source_mapping_layer_ids)
            ]
        else:
            if len(reindexed_tileinfo) != self.layer_num:
                raise Phase2Invalid(
                    _PLACEMENT_INCONSISTENT,
                    (
                        f"{placement.item_id}: slice has {len(reindexed_tileinfo)} tileinfo rows for "
                        f"{self.layer_num} local layers"
                    ),
                )
            self.tileinfo = deepcopy(reindexed_tileinfo)
        self.layer_tileinfo = self.tileinfo

        self.max_inbuf_size = placement.tcg.max_inbuf_size
        self.max_outbuf_size = placement.tcg.max_outbuf_size
        self.global_buf_size = 0
        self.global_adder_num = 0
        self.global_multiplier_num = 0
        self.global_adder_bitwidth = placement.tcg.global_adder_bitwidth
        self.global_multiplier_bitwidth = placement.tcg.global_multiplier_bitwidth
        self.global_data_size = 0
        self.inLayer_distance = np.zeros((1, self.layer_num))
        self.transLayer_distance = np.zeros((1, self.layer_num))
        self.aggregate_arg = np.zeros((self.layer_num, 2))

    def mapping_net(self) -> None:
        """Do not remap a slice: its placement is inherited from the round."""

    def calculate_transfer_distance(self) -> None:
        """Do not reconstruct transfer distances from a step-local graph."""

    def with_context_prefix(self, prefixed_netstruct: list[Any]) -> "ReindexedTCGPlacementSlice":
        """Prefix one context Conv table without changing resident placement."""
        if self.layer_num != 1 or len(prefixed_netstruct) != 2:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                (
                    f"{self.placement.item_id}: context prefix requires one mapped local layer "
                    f"and a two-layer netstruct, got {self.layer_num} and {len(prefixed_netstruct)}"
                ),
            )

        context_layer_info = _netstruct_layer_info(prefixed_netstruct[0])
        if str(context_layer_info.get("type", "")) != "conv":
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{self.placement.item_id}: context prefix must start with a conv layer",
            )

        # Build a valid standalone table before reconnecting it to the mapped target.
        standalone_context_info = deepcopy(dict(context_layer_info))
        standalone_context_info["Layerindex"] = 0
        standalone_context_info["Inputindex"] = [-1]
        standalone_context_info["Outputindex"] = []
        context_netstruct = build_netstruct([standalone_context_info])

        from MNSIM.Mapping_Model.Tile_connection_graph import TCG

        try:
            context_tcg = TCG(
                context_netstruct,
                self.placement.sim_config_path,
                multiple=[1],
            )
        except AssertionError as exc:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{self.placement.item_id}: original TCG could not build the context conv table",
            ) from exc

        context_tileinfo = _reindexed_tileinfo(context_tcg.layer_tileinfo[0], 0)
        context_tileinfo["Inputindex"] = [-1]
        context_tileinfo["Outputindex"] = [1]
        target_tileinfo = _reindexed_tileinfo(self.layer_tileinfo[0], 1)
        target_tileinfo["Inputindex"] = [-1]
        target_tileinfo["Outputindex"] = []

        return ReindexedTCGPlacementSlice(
            self.placement,
            prefixed_netstruct,
            [-1, *self.source_mapping_layer_ids],
            reindexed_tileinfo=[context_tileinfo, target_tileinfo],
        )


@dataclass
class RoundPlacementPlan:
    item_id: str
    sim_config_path: str
    tcg: Any
    mapping_netstruct: list[Any]
    mapping_layer_ids_by_step: dict[str, list[int]]
    mapping_summary: dict[str, Any]
    mapped_area_um2: float
    fragment_specs_by_id: dict[str, Any]
    fragment_slots_by_id: dict[str, dict[str, int]]
    output_owner_shards_by_layer: dict[str, dict[int, int]]

    def slice_for(
        self,
        step_id: str,
        scope_index: int,
        netstruct: list[Any],
    ) -> ReindexedTCGPlacementSlice:
        if step_id not in self.mapping_layer_ids_by_step:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{self.item_id}: unknown weighted step {step_id!r}",
            )
        if not isinstance(scope_index, Integral) or isinstance(scope_index, bool):
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{self.item_id}: step {step_id!r} has invalid scope index {scope_index!r}",
            )
        scope_index = int(scope_index)
        step_mapping = self.mapping_layer_ids_by_step[step_id]
        if scope_index < 0 or scope_index >= len(step_mapping):
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{self.item_id}: step {step_id!r} has out-of-range scope index {scope_index!r}",
            )
        mapping_layer_id = step_mapping[scope_index]
        return ReindexedTCGPlacementSlice(self, netstruct, [mapping_layer_id])


def prepare_round_placement(
    *,
    item_id: str,
    sim_config_path: str,
    weighted_scopes: list[WeightedScope],
    expected_used_xbars: int,
) -> RoundPlacementPlan:
    return _build_round_placement(
        item_id=item_id,
        sim_config_path=sim_config_path,
        weighted_scopes=weighted_scopes,
        expected_used_xbars=expected_used_xbars,
    )


def _build_round_placement(
    *,
    item_id: str,
    sim_config_path: str,
    weighted_scopes: list[WeightedScope],
    expected_used_xbars: int,
) -> RoundPlacementPlan:
    _validate_active_xbar_count(
        item_id=item_id,
        value=expected_used_xbars,
        detail="expected_used_xbars",
    )

    mapping_layer_infos: list[dict[str, Any]] = []
    mapping_layer_ids_by_step_index: dict[str, dict[int, int]] = {}
    weighted_step_ids: list[str] = []
    active_fragment_layout: dict[int, dict[str, Any]] = {}
    active_fragment_ids: list[str] = []
    fragment_specs_by_id: dict[str, Any] = {}
    for mapping_layer_id, scope in enumerate(weighted_scopes):
        step_id = scope.step_id
        if not isinstance(step_id, str) or not step_id:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{item_id}: weighted scope has invalid step id {step_id!r}",
            )
        if not isinstance(scope.layer_info, Mapping):
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{item_id}: weighted step {step_id!r} has invalid layer info",
            )
        scope_index = scope.scope_index
        if not isinstance(scope_index, Integral) or isinstance(scope_index, bool) or scope_index < 0:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{item_id}: weighted scope {scope.step_id} has invalid scope index {scope_index!r}",
            )
        scope_index = int(scope_index)
        step_mapping = mapping_layer_ids_by_step_index.setdefault(step_id, {})
        if scope_index in step_mapping:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{item_id}: weighted step {step_id} repeats scope index {scope_index}",
            )
        step_mapping[scope_index] = mapping_layer_id
        layer_info = deepcopy(dict(scope.layer_info))
        layer_type = str(layer_info.get("type", ""))
        if layer_type not in _LEGAL_MAPPING_TYPES:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{item_id}: weighted scope {scope.step_id} has illegal mapping type {layer_type!r}",
            )
        layout_payload = _fragment_layout_payload(
            item_id=item_id,
            scope=scope,
            mapping_layer_type=layer_type,
        )
        layer_info["Layerindex"] = mapping_layer_id
        layer_info["Inputindex"] = [-1]
        layer_info["Outputindex"] = []
        mapping_layer_infos.append(layer_info)
        active_fragment_layout[mapping_layer_id] = layout_payload
        active_fragment_ids.extend(layout_payload["fragment_ids"])
        for fragment in scope.fragment_layout.fragments:
            if fragment.fragment_id in fragment_specs_by_id:
                raise Phase2Invalid(
                    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                    f"{item_id}: active fragment id {fragment.fragment_id!r} is duplicated",
                )
            fragment_specs_by_id[fragment.fragment_id] = fragment
        if scope.step_id not in weighted_step_ids:
            weighted_step_ids.append(scope.step_id)

    mapping_layer_ids_by_step: dict[str, list[int]] = {}
    for step_id, mapping_by_scope_index in mapping_layer_ids_by_step_index.items():
        scope_indexes = sorted(mapping_by_scope_index)
        expected_scope_indexes = list(range(len(scope_indexes)))
        if scope_indexes != expected_scope_indexes:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                (
                    f"{item_id}: weighted step {step_id} scope indexes must be contiguous "
                    f"from zero, got {scope_indexes}"
                ),
            )
        mapping_layer_ids_by_step[step_id] = [
            mapping_by_scope_index[index] for index in expected_scope_indexes
        ]

    if len(active_fragment_ids) != len(set(active_fragment_ids)):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            f"{item_id}: active fragment ids must be unique within one weighted round",
        )

    active_physical_xbar_count = sum(
        payload["active_physical_xbar_count"]
        for payload in active_fragment_layout.values()
    )
    active_group_count = sum(
        payload["active_group_count"]
        for payload in active_fragment_layout.values()
    )
    if active_physical_xbar_count != expected_used_xbars:
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item_id}: active physical count {active_physical_xbar_count} "
                f"does not equal Phase 1 used_xbars {expected_used_xbars}"
            ),
        )

    (
        mnsim_group_capacity,
        physical_xbar_capacity,
        xbar_polarity,
    ) = _mnsim_hardware_capacity(sim_config_path)
    if active_group_count > mnsim_group_capacity:
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item_id}: active group count {active_group_count} exceeds "
                f"hardware group capacity {mnsim_group_capacity}"
            ),
        )
    if active_physical_xbar_count > physical_xbar_capacity:
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item_id}: active physical count {active_physical_xbar_count} exceeds "
                f"hardware physical xbar capacity {physical_xbar_capacity}"
            ),
        )
    if any(
        payload["polarity_count"] != xbar_polarity
        for payload in active_fragment_layout.values()
    ):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item_id}: active fragment polarity does not match SimConfig "
                f"Xbar_Polarity {xbar_polarity}"
            ),
        )

    mapping_netstruct = build_netstruct(mapping_layer_infos)
    multiple = [1] * len(mapping_netstruct)

    from MNSIM.Mapping_Model.Tile_connection_graph import TCG

    try:
        tcg = TCG(
            mapping_netstruct,
            sim_config_path,
            multiple=multiple,
            round_residency=True,
            active_fragment_layout=active_fragment_layout,
        )
    except AssertionError as exc:
        raise Phase2Invalid(
            _PLACEMENT_INCONSISTENT,
            f"{item_id}: original TCG could not place the fixed Phase 1 round",
        ) from exc

    tcg.tile.calculate_tile_area(
        SimConfig_path=sim_config_path,
        default_inbuf_size=tcg.max_inbuf_size,
        default_outbuf_size=tcg.max_outbuf_size,
    )
    round_residency_tile_count = math.ceil(
        active_group_count /
        (int(tcg.tile.tile_PE_total_num) * int(tcg.tile.group_num))
    )
    placement_area = tcg.tile.tile_area * round_residency_tile_count
    tcg_allocated_xbar_slots = sum(
        int(info["PEnum"]) * int(tcg.tile.group_num)
        for info in tcg.layer_tileinfo
    )
    fragment_slots_by_id = _fragment_slots_by_id(
        item_id=item_id,
        tcg=tcg,
        expected_fragment_ids=set(fragment_specs_by_id),
    )
    output_owner_shards_by_layer = _output_owner_shards_by_layer(
        item_id=item_id,
        fragment_specs_by_id=fragment_specs_by_id,
        fragment_slots_by_id=fragment_slots_by_id,
    )
    mapping_summary = {
        "item_id": item_id,
        "weighted_step_ids": weighted_step_ids,
        "zero_step_ids": [],
        "mapping_layer_count": len(mapping_netstruct),
        "phase1_used_xbars": int(active_physical_xbar_count),
        "expected_used_xbars": int(expected_used_xbars),
        "active_group_count": int(active_group_count),
        "active_physical_xbar_count": int(active_physical_xbar_count),
        "mnsim_group_capacity": int(mnsim_group_capacity),
        "physical_xbar_capacity": int(physical_xbar_capacity),
        "active_fragment_ids": list(active_fragment_ids),
        "tcg_allocated_xbar_slots": tcg_allocated_xbar_slots,
        "xbar_capacity": physical_xbar_capacity,
        "round_residency_tile_count": round_residency_tile_count,
        "logical_layer_tile_count": int(tcg.used_tile_num),
        "tcg_buffer_capacity_evidence": deepcopy(
            list(getattr(tcg, "buffer_capacity_evidence", []) or [])
        ),
        "tcg_pe_input_rounded_capacity_bits_per_instance": int(
            math.ceil(float(tcg.max_inbuf_size) * 1024 * 8)
        ),
        "tcg_tile_output_rounded_capacity_bits_per_instance": int(
            math.ceil(float(tcg.max_outbuf_size) * 1024 * 8)
        ),
    }
    return RoundPlacementPlan(
        item_id=item_id,
        sim_config_path=sim_config_path,
        tcg=tcg,
        mapping_netstruct=mapping_netstruct,
        mapping_layer_ids_by_step=mapping_layer_ids_by_step,
        mapping_summary=mapping_summary,
        mapped_area_um2=placement_area,
        fragment_specs_by_id=fragment_specs_by_id,
        fragment_slots_by_id=fragment_slots_by_id,
        output_owner_shards_by_layer=output_owner_shards_by_layer,
    )


def _output_owner_shards_by_layer(
    *,
    item_id: str,
    fragment_specs_by_id: dict[str, Any],
    fragment_slots_by_id: dict[str, dict[str, int]],
) -> dict[str, dict[int, int]]:
    """Assign each completed output interval to its final physical reducer Tile."""
    interval_owner: dict[str, dict[tuple[int, int], int]] = {}
    full_ranges: dict[str, tuple[int, int]] = {}
    ordered_fragments = sorted(
        fragment_specs_by_id.items(),
        key=lambda item: (
            int(fragment_slots_by_id[item[0]]["tile_id"]),
            int(fragment_slots_by_id[item[0]]["pe_id"]),
            int(fragment_slots_by_id[item[0]]["group_id"]),
            item[0],
        ),
    )
    for fragment_id, fragment in ordered_fragments:
        layer_id = str(fragment.layer_id)
        interval = tuple(int(value) for value in fragment.output_range)
        full_range = tuple(int(value) for value in fragment.full_output_range)
        if len(interval) != 2 or len(full_range) != 2:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{item_id}: fragment {fragment_id} has invalid output range",
            )
        previous_full = full_ranges.setdefault(layer_id, full_range)
        if previous_full != full_range:
            raise Phase2Invalid(
                _PLACEMENT_INCONSISTENT,
                f"{item_id}: layer {layer_id} has inconsistent full output ranges",
            )
        # Physical slot order is execution order. The last contributor for an
        # output interval is the Tile where the completed accumulator closes.
        interval_owner.setdefault(layer_id, {})[interval] = int(
            fragment_slots_by_id[fragment_id]["tile_id"]
        )

    result: dict[str, dict[int, int]] = {}
    for layer_id, owners in interval_owner.items():
        full_start, full_end = full_ranges[layer_id]
        owner_by_neuron: dict[int, int] = {}
        for (start, end), tile_id in owners.items():
            if start < full_start or end > full_end or end <= start:
                raise Phase2Invalid(
                    _PLACEMENT_INCONSISTENT,
                    f"{item_id}: layer {layer_id} has invalid output interval {(start, end)}",
                )
            for neuron_id in range(start, end):
                existing = owner_by_neuron.get(neuron_id)
                if existing is not None and existing != tile_id:
                    raise Phase2Invalid(
                        _PLACEMENT_INCONSISTENT,
                        f"{item_id}: layer {layer_id} neuron {neuron_id} has overlapping owner intervals",
                    )
                owner_by_neuron[neuron_id] = tile_id
        counts: dict[int, int] = {}
        for tile_id in owner_by_neuron.values():
            counts[tile_id] = counts.get(tile_id, 0) + 1
        result[layer_id] = counts
    return result


def _fragment_layout_payload(
    *,
    item_id: str,
    scope: WeightedScope,
    mapping_layer_type: str,
) -> dict[str, Any]:
    layout = scope.fragment_layout
    if not isinstance(layout, MnsimFragmentLayout):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            f"{item_id}: weighted step {scope.step_id!r} has no valid fragment layout",
        )
    layout_mapping_type = "fc" if layout.layer_type == "fc_like_projection" else layout.layer_type
    if layout_mapping_type != mapping_layer_type:
        raise Phase2Invalid(
            _PLACEMENT_INCONSISTENT,
            (
                f"{item_id}: weighted step {scope.step_id!r} has {layout.layer_type!r} "
                f"fragment layout for mapping type {mapping_layer_type!r}"
            ),
        )
    _validate_active_xbar_count(
        item_id=item_id,
        value=layout.active_group_count,
        detail=f"weighted step {scope.step_id!r} active_group_count",
    )
    _validate_active_xbar_count(
        item_id=item_id,
        value=layout.active_physical_xbar_count,
        detail=f"weighted step {scope.step_id!r} active_physical_xbar_count",
    )
    fragments = tuple(layout.fragments)
    if len(fragments) != int(layout.active_group_count):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item_id}: weighted step {scope.step_id!r} has {len(fragments)} fragments "
                f"for active_group_count {layout.active_group_count}"
            ),
        )

    fragment_ids: list[str] = []
    polarity_counts: set[int] = set()
    fragment_physical_count = 0
    for fragment in fragments:
        if not isinstance(fragment.fragment_id, str) or not fragment.fragment_id:
            raise Phase2Invalid(
                TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                f"{item_id}: weighted step {scope.step_id!r} has an invalid fragment id",
            )
        _validate_active_xbar_count(
            item_id=item_id,
            value=fragment.polarity_count,
            detail=f"weighted step {scope.step_id!r} fragment polarity_count",
        )
        _validate_active_xbar_count(
            item_id=item_id,
            value=fragment.physical_xbar_count,
            detail=f"weighted step {scope.step_id!r} fragment physical_xbar_count",
        )
        if int(fragment.physical_xbar_count) != int(fragment.polarity_count):
            raise Phase2Invalid(
                TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                (
                    f"{item_id}: weighted step {scope.step_id!r} fragment "
                    "does not contain one complete polarity payload"
                ),
            )
        fragment_ids.append(fragment.fragment_id)
        polarity_counts.add(int(fragment.polarity_count))
        fragment_physical_count += int(fragment.physical_xbar_count)

    if len(fragment_ids) != len(set(fragment_ids)):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            f"{item_id}: weighted step {scope.step_id!r} repeats fragment ids",
        )
    if len(polarity_counts) != 1:
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            f"{item_id}: weighted step {scope.step_id!r} mixes fragment polarities",
        )
    if fragment_physical_count != int(layout.active_physical_xbar_count):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item_id}: weighted step {scope.step_id!r} fragment physical count "
                f"{fragment_physical_count} does not match active_physical_xbar_count "
                f"{layout.active_physical_xbar_count}"
            ),
        )
    return {
        "fragment_ids": fragment_ids,
        "active_group_count": int(layout.active_group_count),
        "active_physical_xbar_count": int(layout.active_physical_xbar_count),
        "polarity_count": polarity_counts.pop(),
    }


def _validate_active_xbar_count(*, item_id: str, value: Any, detail: str) -> None:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            f"{item_id}: {detail} must be a positive integral value, got {value!r}",
        )


def _mnsim_hardware_capacity(sim_config_path: str) -> tuple[int, int, int]:
    import configparser

    from MNSIM.Hardware_Model.Tile import tile

    config = configparser.ConfigParser()
    config.read(sim_config_path, encoding="UTF-8")
    tile_num = list(map(int, config.get("Architecture level", "Tile_Num").split(",")))
    if tile_num[0] == 0:
        tile_num = [8, 8]
    hardware_tile = tile(sim_config_path)
    xbar_polarity = int(config.get("Process element level", "Xbar_Polarity"))
    mnsim_group_capacity = (
        int(tile_num[0])
        * int(tile_num[1])
        * int(hardware_tile.tile_PE_total_num)
        * int(hardware_tile.group_num)
    )
    return (
        mnsim_group_capacity,
        mnsim_group_capacity * xbar_polarity,
        xbar_polarity,
    )


def _fragment_slots_by_id(
    *,
    item_id: str,
    tcg: Any,
    expected_fragment_ids: set[str],
) -> dict[str, dict[str, int]]:
    raw_slots_by_layer = getattr(tcg, "active_fragment_group_slots", None)
    if not isinstance(raw_slots_by_layer, Mapping):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            f"{item_id}: TCG does not expose active fragment group slots",
        )

    result: dict[str, dict[str, int]] = {}
    for layer_slots in raw_slots_by_layer.values():
        if not isinstance(layer_slots, list):
            raise Phase2Invalid(
                TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                f"{item_id}: TCG active fragment slots must be lists",
            )
        for raw_slot in layer_slots:
            if not isinstance(raw_slot, Mapping):
                raise Phase2Invalid(
                    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                    f"{item_id}: TCG emitted an invalid physical group slot",
                )
            fragment_id = raw_slot.get("fragment_id")
            if not isinstance(fragment_id, str) or not fragment_id:
                raise Phase2Invalid(
                    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                    f"{item_id}: TCG emitted a physical group slot without fragment_id",
                )
            if fragment_id in result:
                raise Phase2Invalid(
                    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                    f"{item_id}: TCG assigned fragment {fragment_id!r} twice",
                )
            try:
                result[fragment_id] = {
                    "tile_id": int(raw_slot["tile_id"]),
                    "pe_id": int(raw_slot["pe_id"]),
                    "group_id": int(raw_slot["group_id"]),
                }
            except (KeyError, TypeError, ValueError) as error:
                raise Phase2Invalid(
                    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                    f"{item_id}: TCG emitted malformed physical group coordinates",
                ) from error
    if set(result) != expected_fragment_ids:
        missing = sorted(expected_fragment_ids - set(result))
        unexpected = sorted(set(result) - expected_fragment_ids)
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item_id}: TCG physical group slots do not match active fragments; "
                f"missing={missing[:3]}, unexpected={unexpected[:3]}"
            ),
        )
    return result


def _netstruct_layer_info(layer_entry: Any) -> Mapping[str, Any]:
    if isinstance(layer_entry, Mapping):
        return layer_entry
    if isinstance(layer_entry, (list, tuple)) and layer_entry:
        first = layer_entry[0]
        if isinstance(first, Mapping):
            return first
        if isinstance(first, (list, tuple)) and first and isinstance(first[0], Mapping):
            return first[0]
    raise Phase2Invalid(
        _PLACEMENT_INCONSISTENT,
        "context prefix contains an invalid NetStruct layer entry",
    )


def _reindexed_tileinfo(
    source_tileinfo: dict[str, Any],
    local_layer_id: int = 0,
) -> dict[str, Any]:
    tileinfo = deepcopy(dict(source_tileinfo))
    tileinfo["startid"] = int(local_layer_id)
    tileinfo["Inputindex"] = [-1]
    tileinfo["Outputindex"] = []
    tileinfo["is_branchin"] = -1
    tileinfo["is_branchout"] = -1
    return tileinfo


__all__ = [
    "FrozenTCG",
    "ReindexedTCGPlacementSlice",
    "RoundPlacementPlan",
    "WeightedScope",
    "deserialize_round_placement",
    "prepare_round_placement",
    "serialize_round_placement",
]
