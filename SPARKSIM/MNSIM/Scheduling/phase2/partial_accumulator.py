"""Cost model for scheduled partial-sum reductions outside one MNSIM mapping."""

from __future__ import annotations

import configparser
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from MNSIM.Hardware_Model.Adder import adder
from MNSIM.Hardware_Model.Buffer import buffer
from MNSIM.Scheduling.phase2.types import (
    BOUNDARY_STATUS_PARTIAL_SUM,
    MetricTotals,
    StepExecutionResult,
)


SOURCE_KIND_PARTIAL_ACCUMULATOR = "scheduled_partial_accumulator"


@dataclass(frozen=True)
class PartialAccumulatorCost:
    latency_s: float
    energy_j: float
    addition_count: int
    contribution_count: int
    bit_widths: tuple[int, ...]
    lanes: int
    state_read_count: int
    state_write_count: int
    state_read_bytes: int
    state_write_bytes: int
    state_read_latency_s: float
    state_write_latency_s: float
    state_read_energy_j: float
    state_write_energy_j: float


class ScheduledPartialAccumulatorCostModel:
    """Score additions that become explicit when fragments span scheduler rounds."""

    def __init__(self, sim_config_path: str, plan: Any):
        self.sim_config_path = str(sim_config_path)
        self._spatial_by_layer = _spatial_by_layer(plan)
        self._lanes = _provisioned_accumulator_lanes(self.sim_config_path)
        self._seen_output_intervals: set[tuple[str, str, int, tuple[int, int]]] = set()

    def score_commit(self, item: Any, commit_result: Any) -> StepExecutionResult | None:
        additions_by_timestep: dict[int, list[tuple[int, int]]] = defaultdict(list)
        storage_by_key: dict[
            tuple[str, str, int, tuple[int, int]], dict[str, Any]
        ] = {}
        seen_before_commit = set(self._seen_output_intervals)
        contribution_count = 0
        for update in list(getattr(commit_result, "completion_updates", []) or []):
            selected = list(update.get("selected_fragment_ids", []) or [])
            if not selected:
                continue
            output_range = update.get(
                "contribution_output_range", update.get("output_range_covered")
            )
            if not isinstance(output_range, (list, tuple)) or len(output_range) != 2:
                continue
            output_count = int(output_range[1]) - int(output_range[0])
            if output_count <= 0:
                continue
            layer_id = str(update["origin_layer_id"])
            sample_id = str(update.get("sample_id", "sample:0"))
            timestep = int(update.get("timestep", 0))
            interval = (int(output_range[0]), int(output_range[1]))
            accumulator_key = (layer_id, sample_id, timestep, interval)
            needs_addition = accumulator_key in self._seen_output_intervals
            self._seen_output_intervals.add(accumulator_key)
            bit_width = int(update.get("data_bitwidth") or 0)
            if bit_width <= 0:
                continue
            neuron_count = output_count * self._spatial_by_layer.get(layer_id, 1)
            storage_by_key[accumulator_key] = {
                "timestep": timestep,
                "byte_count": math.ceil(neuron_count * bit_width / 8),
                "output_incomplete": (
                    str(update.get("completion_status", ""))
                    == BOUNDARY_STATUS_PARTIAL_SUM
                ),
            }
            if needs_addition:
                additions_by_timestep[timestep].append(
                    (neuron_count, bit_width)
                )
                contribution_count += 1

        if not additions_by_timestep and not any(
            key in seen_before_commit or spec["output_incomplete"]
            for key, spec in storage_by_key.items()
        ):
            return None

        arithmetic_energy_j = 0.0
        arithmetic_latency_s = 0.0
        addition_count = 0
        bit_widths: set[int] = set()
        for additions in additions_by_timestep.values():
            timestep_latency_ns = 0.0
            for neuron_count, bit_width in additions:
                primitive = adder(self.sim_config_path, bitwidth=bit_width)
                primitive.calculate_adder_energy()
                arithmetic_energy_j += neuron_count * primitive.adder_energy * 1e-9
                timestep_latency_ns = max(
                    timestep_latency_ns,
                    math.ceil(neuron_count / self._lanes) * primitive.adder_latency,
                )
                addition_count += neuron_count
                bit_widths.add(bit_width)
            arithmetic_latency_s += timestep_latency_ns * 1e-9

        state_read_count = 0
        state_write_count = 0
        state_read_bytes = 0
        state_write_bytes = 0
        state_read_energy_j = 0.0
        state_write_energy_j = 0.0
        read_latency_by_timestep: dict[int, float] = defaultdict(float)
        write_latency_by_timestep: dict[int, float] = defaultdict(float)
        for accumulator_key, spec in storage_by_key.items():
            timestep = int(spec["timestep"])
            byte_count = int(spec["byte_count"])
            if accumulator_key in seen_before_commit:
                access_latency_s, access_energy_j = self._buffer_access_cost(
                    byte_count, read=True
                )
                state_read_count += 1
                state_read_bytes += byte_count
                state_read_energy_j += access_energy_j
                read_latency_by_timestep[timestep] = max(
                    read_latency_by_timestep[timestep], access_latency_s
                )
            if bool(spec["output_incomplete"]):
                access_latency_s, access_energy_j = self._buffer_access_cost(
                    byte_count, read=False
                )
                state_write_count += 1
                state_write_bytes += byte_count
                state_write_energy_j += access_energy_j
                write_latency_by_timestep[timestep] = max(
                    write_latency_by_timestep[timestep], access_latency_s
                )

        state_read_latency_s = sum(read_latency_by_timestep.values())
        state_write_latency_s = sum(write_latency_by_timestep.values())
        latency_s = (
            arithmetic_latency_s + state_read_latency_s + state_write_latency_s
        )
        energy_j = (
            arithmetic_energy_j + state_read_energy_j + state_write_energy_j
        )

        cost = PartialAccumulatorCost(
            latency_s=latency_s,
            energy_j=energy_j,
            addition_count=addition_count,
            contribution_count=contribution_count,
            bit_widths=tuple(sorted(bit_widths)),
            lanes=self._lanes,
            state_read_count=state_read_count,
            state_write_count=state_write_count,
            state_read_bytes=state_read_bytes,
            state_write_bytes=state_write_bytes,
            state_read_latency_s=state_read_latency_s,
            state_write_latency_s=state_write_latency_s,
            state_read_energy_j=state_read_energy_j,
            state_write_energy_j=state_write_energy_j,
        )
        return StepExecutionResult(
            item_id=str(item.item_id),
            step_id=f"{item.item_id}.partial_accumulator",
            segment_index=len(list(getattr(item, "ordered_segments", []) or [])),
            step_kind="scheduled_partial_accumulator",
            source_kind=SOURCE_KIND_PARTIAL_ACCUMULATOR,
            status="ok",
            origin_layer_ids=sorted(
                {
                    str(update["origin_layer_id"])
                    for update in commit_result.completion_updates
                    if update.get("selected_fragment_ids")
                }
            ),
            metrics=MetricTotals(
                latency_s=cost.latency_s,
                energy_j=cost.energy_j,
                power_w=cost.energy_j / cost.latency_s if cost.latency_s > 0 else 0.0,
                area_um2=0.0,
            ),
            layer_results=[],
            mapping_summary={
                "scheduled_partial_accumulator": {
                    "addition_count": cost.addition_count,
                    "contribution_count": cost.contribution_count,
                    "bit_widths": list(cost.bit_widths),
                    "lanes": cost.lanes,
                    "primitive": "MNSIM.Hardware_Model.Adder+Buffer",
                    "arithmetic_latency_s": arithmetic_latency_s,
                    "arithmetic_energy_j": arithmetic_energy_j,
                    "state_read_count": cost.state_read_count,
                    "state_write_count": cost.state_write_count,
                    "state_read_bytes": cost.state_read_bytes,
                    "state_write_bytes": cost.state_write_bytes,
                    "state_read_latency_s": cost.state_read_latency_s,
                    "state_write_latency_s": cost.state_write_latency_s,
                    "state_read_energy_j": cost.state_read_energy_j,
                    "state_write_energy_j": cost.state_write_energy_j,
                }
            },
            raw_mnsim_audit=[],
            boundary_bindings=[],
            warnings=[],
        )

    def _buffer_access_cost(
        self, byte_count: int, *, read: bool
    ) -> tuple[float, float]:
        state_buffer = buffer(
            self.sim_config_path,
            buf_level=2,
            default_buf_size=max(1, math.ceil(byte_count / 1024)),
        )
        if read:
            state_buffer.calculate_buf_read_latency(rdata=byte_count)
            state_buffer.calculate_buf_read_energy(rdata=byte_count)
            return state_buffer.buf_rlatency * 1e-9, state_buffer.buf_renergy * 1e-9
        state_buffer.calculate_buf_write_latency(wdata=byte_count)
        state_buffer.calculate_buf_write_energy(wdata=byte_count)
        return state_buffer.buf_wlatency * 1e-9, state_buffer.buf_wenergy * 1e-9


def _spatial_by_layer(plan: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for descriptor in list(getattr(plan, "layer_resource_descriptors", []) or []):
        shape = dict(getattr(descriptor.prepass_ref, "shape_signature", {}) or {})
        output_size = shape.get("Outputsize")
        spatial = 1
        if isinstance(output_size, (list, tuple)) and len(output_size) == 2:
            spatial = int(output_size[0]) * int(output_size[1])
        result[str(descriptor.layer_id)] = max(1, spatial)
    return result


def _provisioned_accumulator_lanes(sim_config_path: str) -> int:
    parser = configparser.ConfigParser()
    if not parser.read(sim_config_path, encoding="UTF-8"):
        raise ValueError(f"cannot read SimConfig {sim_config_path}")
    adc_num = parser.getint("Process element level", "ADC_Num")
    pe_rows, pe_cols = (
        int(value.strip())
        for value in parser.get("Tile level", "PE_Num").split(",")
    )
    tile_rows, tile_cols = (
        int(value.strip())
        for value in parser.get("Architecture level", "Tile_Num").split(",")
    )
    return max(1, adc_num * pe_rows * pe_cols * tile_rows * tile_cols)


__all__ = [
    "SOURCE_KIND_PARTIAL_ACCUMULATOR",
    "ScheduledPartialAccumulatorCostModel",
]
