"""Device-independent transition cost aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from MNSIM.Scheduling.weight_scheduling.device import (
    DeviceModel,
    DeviceTransitionCost,
    JsonDeviceModel,
)


@dataclass(frozen=True)
class AggregatedTransitionCost:
    changed_cells: int
    objective_cost: float
    objective_unit: str
    latency_s: float | None
    energy_j: float | None
    exact: bool
    provenance: dict[str, Any]


@dataclass(frozen=True)
class TransitionCostView:
    """Candidate assignment costs for one logical tile against physical tiles."""

    by_logical_tile: dict[str, dict[int, AggregatedTransitionCost]]

    def cost(self, logical_tile_id: str, physical_tile_id: int) -> AggregatedTransitionCost:
        return self.by_logical_tile[str(logical_tile_id)][int(physical_tile_id)]


class TransitionCostEngine:
    """Apply one device model to arbitrary cell-state sequences."""

    def __init__(self, device_model: DeviceModel):
        self.device_model = device_model
        self._native_target_cache: dict[int, list[dict[str, Any]] | None] = {}

    def _targets_for_tile(self, tile: Any) -> list[dict[str, Any]] | None:
        key = id(tile)
        if key not in self._native_target_cache:
            self._native_target_cache[key] = _native_physical_targets(tile)
        return self._native_target_cache[key]

    def aggregate(
        self,
        source_state: tuple[int, ...] | list[int],
        target_state: tuple[int, ...] | list[int],
    ) -> AggregatedTransitionCost:
        source = tuple(int(code) for code in source_state)
        target = tuple(int(code) for code in target_state)
        # A scheduled physical tile may carry different numbers of occupied
        # cells in different rounds.  Treat cells absent from the compact
        # payload as erased to the device's initial code before comparing the
        # two fixed-capacity tile states.
        if len(source) != len(target):
            initial = int(self.device_model.initial_code())
            width = max(len(source), len(target))
            source = source + (initial,) * (width - len(source))
            target = target + (initial,) * (width - len(target))
        # Profiles without transition matrices retain the historical fast
        # changed-cell objective. Native profiles use the same per-cell energy
        # and pulse-latency tables as the physical write ledger.
        if isinstance(self.device_model, JsonDeviceModel) and (
            self.device_model.transition_latency_s is None
            or self.device_model.transition_energy_j is None
        ):
            source_array = np.asarray(source, dtype=np.uint16)
            target_array = np.asarray(target, dtype=np.uint16)
            changed = int(np.count_nonzero(source_array != target_array))
            return AggregatedTransitionCost(
                changed_cells=changed,
                objective_cost=float(changed),
                objective_unit=str(self.device_model.objective_unit),
                latency_s=None,
                energy_j=None,
                exact=False,
                provenance={
                    "device_model_id": str(self.device_model.model_id),
                    "device_bits": int(self.device_model.device_bits),
                    "device_level": int(self.device_model.device_level),
                    "cell_count": len(source),
                    "modeled_fields": [],
                    "unavailable_fields": ["energy_j", "latency_s"],
                },
            )
        if isinstance(self.device_model, JsonDeviceModel):
            source_array = np.asarray(source, dtype=np.intp)
            target_array = np.asarray(target, dtype=np.intp)
            changed = int(np.count_nonzero(source_array != target_array))
            latency_table = np.asarray(
                self.device_model.transition_latency_s, dtype=np.float64
            ).reshape(self.device_model.device_level, self.device_model.device_level)
            energy_table = np.asarray(
                self.device_model.transition_energy_j, dtype=np.float64
            ).reshape(self.device_model.device_level, self.device_model.device_level)
            return AggregatedTransitionCost(
                changed_cells=changed,
                objective_cost=float(np.sum(energy_table[source_array, target_array])),
                objective_unit=str(self.device_model.objective_unit),
                latency_s=float(np.sum(latency_table[source_array, target_array])),
                energy_j=float(np.sum(energy_table[source_array, target_array])),
                exact=bool(
                    self.device_model.exact_write_latency
                    and self.device_model.exact_write_energy
                ),
                provenance={
                    "device_model_id": str(self.device_model.model_id),
                    "device_bits": int(self.device_model.device_bits),
                    "device_level": int(self.device_model.device_level),
                    "cell_count": len(source),
                    "modeled_fields": ["energy_j", "latency_s"],
                    "unavailable_fields": [],
                    "objective_semantics": "changed_cell_count_with_native_transition_audit",
                },
            )
        costs = [
            self.device_model.cell_transition_cost(left, right)
            for left, right in zip(source, target)
        ]
        latency_values = [cost.latency_s for cost in costs]
        energy_values = [cost.energy_j for cost in costs]
        latency = sum(latency_values) if all(value is not None for value in latency_values) else None
        energy = sum(energy_values) if all(value is not None for value in energy_values) else None
        exact = bool(costs) and all(cost.exact for cost in costs)
        objective_unit = costs[0].objective_unit if costs else "changed_cell_count"
        return AggregatedTransitionCost(
            changed_cells=sum(cost.changed for cost in costs),
            objective_cost=sum(cost.objective_cost for cost in costs),
            objective_unit=objective_unit,
            latency_s=latency,
            energy_j=energy,
            exact=exact,
            provenance={
                "device_model_id": str(self.device_model.model_id),
                "device_bits": int(self.device_model.device_bits),
                "device_level": int(self.device_model.device_level),
                "cell_count": len(costs),
                "modeled_fields": sorted(
                    {
                        field
                        for cost in costs
                        for field in cost.modeled_fields
                    }
                ),
                "unavailable_fields": sorted(
                    {
                        field
                        for cost in costs
                        for field in cost.unavailable_fields
                    }
                ),
            },
        )

    def aggregate_tile(
        self,
        source_state: tuple[int, ...] | list[int],
        tile: Any,
        *,
        source_physical_state: dict[str, dict[str, Any]] | None = None,
        physical_tile_id: int | None = None,
    ) -> AggregatedTransitionCost:
        """Aggregate one tile, using exact physical fragment slots when present.

        The native SNN path carries an internal slot-aware payload because a
        weighted round may update only a subset of a physical tile's xbars.
        Generic policies continue to use the compact cell-state aggregate.
        """

        targets = self._targets_for_tile(tile)
        if targets is None or not _native_physical_enabled(self.device_model):
            return self.aggregate(source_state, tile.target_state)
        if physical_tile_id is not None:
            targets = _rebase_native_targets(targets, int(physical_tile_id))
        return self._aggregate_physical_targets(
            targets,
            source_physical_state or {},
        )

    def updated_physical_state(
        self,
        source_physical_state: dict[str, dict[str, Any]],
        tile: Any,
        *,
        physical_tile_id: int | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Apply active physical fragment targets while retaining idle slots."""

        targets = self._targets_for_tile(tile)
        if targets is None or not _native_physical_enabled(self.device_model):
            return dict(source_physical_state)
        if physical_tile_id is not None:
            targets = _rebase_native_targets(targets, int(physical_tile_id))
        updated = {
            str(slot): dict(state)
            for slot, state in source_physical_state.items()
        }
        for target in targets:
            updated[target["slot"]] = {
                "shape": target["shape"],
                "values": target["values"],
            }
        return updated

    def matrix(
        self,
        source_state_by_physical_tile: dict[int, tuple[int, ...]],
        target_state_by_logical_tile: dict[str, tuple[int, ...]],
        physical_tile_ids: list[int] | tuple[int, ...],
        *,
        target_tiles: tuple[Any, ...] | list[Any] | None = None,
        source_physical_state_by_tile: dict[int, dict[str, dict[str, Any]]] | None = None,
    ) -> TransitionCostView:
        result: dict[str, dict[int, AggregatedTransitionCost]] = {}
        target_by_id = {
            str(tile.tile_id): tile for tile in (target_tiles or ())
        }
        initial_by_length: dict[int, tuple[int, ...]] = {}
        initial_costs: dict[tuple[int, int], AggregatedTransitionCost] = {}
        for logical_id, target in target_state_by_logical_tile.items():
            result[str(logical_id)] = {}
            for physical_id in physical_tile_ids:
                physical_id = int(physical_id)
                tile = target_by_id.get(str(logical_id))
                if (
                    tile is not None
                    and _native_physical_enabled(self.device_model)
                    and self._targets_for_tile(tile) is not None
                ):
                    cost = self.aggregate_tile(
                        tuple(),
                        tile,
                        source_physical_state=(source_physical_state_by_tile or {}).get(
                            physical_id, {}
                        ),
                        physical_tile_id=physical_id,
                    )
                    result[str(logical_id)][physical_id] = cost
                    continue
                source = source_state_by_physical_tile.get(physical_id)
                if source is None:
                    length = len(target)
                    source = initial_by_length.setdefault(
                        length,
                        tuple(self.device_model.initial_code() for _ in range(length)),
                    )
                    cache_key = (id(target), length)
                    cost = initial_costs.get(cache_key)
                    if cost is None:
                        cost = self.aggregate(source, target)
                        initial_costs[cache_key] = cost
                else:
                    cost = self.aggregate(source, target)
                result[str(logical_id)][physical_id] = cost
        return TransitionCostView(result)

    def _aggregate_physical_targets(
        self,
        targets: list[dict[str, Any]],
        source_by_slot: dict[str, dict[str, Any]],
    ) -> AggregatedTransitionCost:
        """Use the Phase 2 per-cell tables on each active physical xbar."""

        tables = _native_tables(self.device_model)
        changed = 0
        energy = 0.0
        latency_by_pe: dict[int, float] = {}
        cell_count = 0
        for target in targets:
            shape = tuple(int(value) for value in target["shape"])
            target_values = np.asarray(target["values"], dtype=np.intp).reshape(shape)
            source_payload = source_by_slot.get(target["slot"])
            if source_payload is None:
                source_values = np.zeros(shape, dtype=np.intp)
            else:
                source_shape = tuple(int(value) for value in source_payload["shape"])
                source_values = np.asarray(source_payload["values"], dtype=np.intp).reshape(
                    source_shape
                )
            source_xbar = _fit_xbar(source_values, tables.xbar_shape)
            target_xbar = _fit_xbar(target_values, tables.xbar_shape)
            cell_latency = tables.latency_ns[source_xbar, target_xbar]
            cell_energy = tables.energy_nj[source_xbar, target_xbar]
            changed += int(np.count_nonzero(source_xbar != target_xbar))
            energy += float(np.sum(cell_energy)) * 1e-9
            padded_latency = _pad_to_ou(cell_latency, tables.ou_shape)
            ou_rows, ou_cols = tables.ou_shape
            latency_by_xbar = float(
                padded_latency.reshape(
                    padded_latency.shape[0] // ou_rows,
                    ou_rows,
                    padded_latency.shape[1] // ou_cols,
                    ou_cols,
                )
                .max(axis=3)
                .sum()
            ) * 1e-9
            pe_id = int(target["slot"].split(":", 1)[0])
            latency_by_pe[pe_id] = max(latency_by_pe.get(pe_id, 0.0), latency_by_xbar)
            cell_count += int(source_xbar.size)
        latency = max(latency_by_pe.values(), default=0.0)
        return AggregatedTransitionCost(
            changed_cells=changed,
            objective_cost=energy,
            objective_unit="physical_write_energy_j",
            latency_s=latency,
            energy_j=energy,
            exact=True,
            provenance={
                "device_model_id": str(self.device_model.model_id),
                "device_bits": int(self.device_model.device_bits),
                "device_level": int(self.device_model.device_level),
                "cell_count": cell_count,
                "modeled_fields": ["energy_j", "latency_s"],
                "unavailable_fields": [],
                "physical_fragment_slot_exact": True,
                "latency_aggregation": "ou_max_then_pe_max",
                "idle_slot_policy": "retain_resident_state",
            },
        )


__all__ = [
    "AggregatedTransitionCost",
    "DeviceTransitionCost",
    "TransitionCostEngine",
    "TransitionCostView",
]


def _native_physical_targets(tile: Any) -> list[dict[str, Any]] | None:
    metadata = getattr(tile, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    raw = metadata.get("_native_physical_targets")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise ValueError("_native_physical_targets must be a sequence")
    targets: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, dict):
            raise ValueError("native physical target must be a mapping")
        slot = value.get("slot")
        if isinstance(slot, (list, tuple)):
            slot_key = ":".join(str(int(item)) for item in slot)
        else:
            slot_key = str(slot)
        shape = tuple(int(item) for item in value["shape"])
        values = tuple(int(item) for item in value["values"])
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError("native physical target shape must be rank-2")
        if len(values) != shape[0] * shape[1]:
            raise ValueError("native physical target payload shape mismatch")
        targets.append({"slot": slot_key, "shape": shape, "values": values})
    return targets


def _rebase_native_targets(
    targets: list[dict[str, Any]], physical_tile_id: int
) -> list[dict[str, Any]]:
    """Move logical fragment slots onto the candidate physical tile.

    The scheduler keeps fragment metadata in the canonical logical placement.
    A candidate assignment changes only the tile coordinate; PE, group, and
    polarity coordinates remain unchanged.  Phase 2 performs this same
    rebasing when it applies a schedule artifact, so Phase 1 must score the
    candidate against the identical physical slots.
    """

    rebased: list[dict[str, Any]] = []
    for target in targets:
        parts = str(target["slot"]).split(":")
        if len(parts) not in {3, 4}:
            raise ValueError(
                "native physical target slot must be tile:pe:group[:polarity]"
            )
        # Production fragment slots are local PE/group/polarity coordinates;
        # the owning physical tile is already represented by the surrounding
        # ``source_physical_state`` map.  Older fixtures may include a tile
        # prefix, which is the only form that needs rebasing.
        slot = ":".join([str(physical_tile_id), *parts[1:]]) if len(parts) == 4 else target["slot"]
        rebased.append({"slot": slot, "shape": target["shape"], "values": target["values"]})
    return rebased


def _native_tables(device_model: DeviceModel) -> Any:
    sim_config = getattr(device_model, "metadata", {}).get("native_write_cost_sim_config")
    if not sim_config:
        raise ValueError("native physical targets require a SimConfig-backed device model")
    from MNSIM.Scheduling.phase2.physical_write_ledger import load_physical_write_tables

    return load_physical_write_tables(str(sim_config))


def _native_physical_enabled(device_model: DeviceModel) -> bool:
    metadata = getattr(device_model, "metadata", {})
    return bool(
        isinstance(metadata, dict)
        and metadata.get("native_write_cost_source")
        == "MNSIM.Scheduling.phase2.physical_write_ledger"
    )


def _fit_xbar(matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("native physical target matrix must be rank-2")
    result = np.zeros(shape, dtype=np.intp)
    rows = min(int(matrix.shape[0]), int(shape[0]))
    cols = min(int(matrix.shape[1]), int(shape[1]))
    result[:rows, :cols] = matrix[:rows, :cols]
    return result


def _pad_to_ou(cost: np.ndarray, ou_shape: tuple[int, int]) -> np.ndarray:
    row_pad = (-cost.shape[0]) % int(ou_shape[0])
    col_pad = (-cost.shape[1]) % int(ou_shape[1])
    if row_pad == 0 and col_pad == 0:
        return cost
    return np.pad(cost, ((0, row_pad), (0, col_pad)), mode="constant")
