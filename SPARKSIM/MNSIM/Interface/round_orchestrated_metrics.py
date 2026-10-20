"""Deterministic ModelIR operation accounting for round-orchestrated reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from MNSIM.Interface.round_orchestrated_types import RunnerError
from MNSIM.Scheduling.round_types import LayerIR, ModelIR


INVALID_MAC_ESTIMATE = "INVALID_MAC_ESTIMATE"
_MISSING = object()


@dataclass(frozen=True)
class ModelOperationCounts:
    macs_total: int
    ops_total: int
    ops_count_policy: str = "mac_as_2_ops"


def count_model_operations(model_ir: ModelIR) -> ModelOperationCounts:
    """Count MACs from weighted ModelIR layer shapes without reading weights."""

    macs_total = 0
    for layer in model_ir.layers:
        if not layer.is_weighted:
            continue
        if layer.layer_type == "conv":
            macs_total += _conv_macs(layer)
        elif layer.layer_type == "fc":
            macs_total += _fc_macs(layer)
        else:
            raise RunnerError(
                INVALID_MAC_ESTIMATE,
                f"unsupported weighted layer type {layer.layer_type!r} for layer {layer.layer_id}",
            )

    sequence_length = int(getattr(model_ir.sequence_spec, "sequence_length", 1))
    macs_total *= sequence_length
    policy = "dense_equivalent_mac_as_2_ops" if sequence_length > 1 else "mac_as_2_ops"
    return ModelOperationCounts(
        macs_total=macs_total,
        ops_total=2 * macs_total,
        ops_count_policy=policy,
    )


def derive_hardware_rates(
    phase2_metrics: Mapping[str, Any], counts: ModelOperationCounts
) -> dict[str, Any]:
    """Add MAC-derived rates while preserving the original Phase 2 metrics."""

    latency_s = float(phase2_metrics["latency_s"])
    compute_latency_s = float(phase2_metrics.get("read_latency_s", latency_s))
    energy_j = float(phase2_metrics["energy_j"])
    workload_top = counts.ops_total / 1e12
    modeled_efficiency = counts.ops_total / energy_j / 1e12 if energy_j > 0 else 0.0
    accounting_complete = bool(phase2_metrics.get("energy_accounting_complete", True))
    compute_throughput = (
        counts.ops_total / compute_latency_s / 1e12
        if compute_latency_s > 0
        else 0.0
    )
    end_to_end_throughput = (
        counts.ops_total / latency_s / 1e12 if latency_s > 0 else 0.0
    )
    return {
        **dict(phase2_metrics),
        "macs_total": counts.macs_total,
        "ops_total": counts.ops_total,
        "ops_count_policy": counts.ops_count_policy,
        "network_workload_top_per_inference": workload_top,
        "compute_tmacs_per_s": (
            counts.macs_total / compute_latency_s / 1e12
            if compute_latency_s > 0
            else 0.0
        ),
        "compute_throughput_tops": compute_throughput,
        "compute_tops": compute_throughput,
        "tmacs_per_s": counts.macs_total / latency_s / 1e12 if latency_s > 0 else 0.0,
        "end_to_end_throughput_tops": end_to_end_throughput,
        "tops": end_to_end_throughput,
        "modeled_energy_efficiency_tops_per_w": modeled_efficiency,
        "tops_per_w": modeled_efficiency if accounting_complete else None,
        "energy_accounting_complete": accounting_complete,
    }


def _conv_macs(layer: LayerIR) -> int:
    shape = _shape_mapping(layer)
    input_channels = _positive_shape_int(shape, "Inputchannel", layer)
    output_channels = _positive_shape_int(shape, "Outputchannel", layer)
    kernel_height, kernel_width = _kernel_dimensions(shape, layer)
    output_height, output_width = _pair_dimensions(shape, "Outputsize", layer)
    effective_input_channels = _effective_input_channels(shape, input_channels, layer)
    return (
        output_height
        * output_width
        * output_channels
        * effective_input_channels
        * kernel_height
        * kernel_width
    )


def _fc_macs(layer: LayerIR) -> int:
    shape = _shape_mapping(layer)
    return (
        _positive_shape_int(shape, "Infeature", layer)
        * _positive_shape_int(shape, "Outfeature", layer)
    )


def _shape_mapping(layer: LayerIR) -> Mapping[str, Any]:
    if not isinstance(layer.shape, Mapping):
        raise RunnerError(
            INVALID_MAC_ESTIMATE,
            f"layer {layer.layer_id} shape must be a mapping",
        )
    return layer.shape


def _positive_shape_int(shape: Mapping[str, Any], key: str, layer: LayerIR) -> int:
    if key not in shape:
        raise RunnerError(
            INVALID_MAC_ESTIMATE,
            f"layer {layer.layer_id} missing shape field {key}",
        )
    return _positive_int(shape[key], f"layer {layer.layer_id} shape field {key}")


def _kernel_dimensions(shape: Mapping[str, Any], layer: LayerIR) -> tuple[int, int]:
    if "Kernelsize" not in shape:
        raise RunnerError(
            INVALID_MAC_ESTIMATE,
            f"layer {layer.layer_id} missing shape field Kernelsize",
        )

    kernel = shape["Kernelsize"]
    if _is_positive_int(kernel):
        scalar = int(kernel)
        return scalar, scalar
    return _two_positive_ints(kernel, "Kernelsize", layer)


def _pair_dimensions(
    shape: Mapping[str, Any], key: str, layer: LayerIR
) -> tuple[int, int]:
    if key not in shape:
        raise RunnerError(
            INVALID_MAC_ESTIMATE,
            f"layer {layer.layer_id} missing shape field {key}",
        )
    return _two_positive_ints(shape[key], key, layer)


def _two_positive_ints(value: Any, key: str, layer: LayerIR) -> tuple[int, int]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RunnerError(
            INVALID_MAC_ESTIMATE,
            f"layer {layer.layer_id} shape field {key} must be a scalar or exactly two integers",
        )
    if len(value) != 2:
        raise RunnerError(
            INVALID_MAC_ESTIMATE,
            f"layer {layer.layer_id} shape field {key} must contain exactly two integers",
        )
    return (
        _positive_int(value[0], f"layer {layer.layer_id} shape field {key}[0]"),
        _positive_int(value[1], f"layer {layer.layer_id} shape field {key}[1]"),
    )


def _effective_input_channels(
    shape: Mapping[str, Any], input_channels: int, layer: LayerIR
) -> int:
    depthwise = shape.get("Depthwise", _MISSING)
    if depthwise is _MISSING:
        return input_channels
    if isinstance(depthwise, str) and depthwise in ("normal", "point"):
        return input_channels
    if isinstance(depthwise, str) and depthwise == "separable":
        return 1
    raise RunnerError(
        INVALID_MAC_ESTIMATE,
        f"unsupported Depthwise value {depthwise!r} for layer {layer.layer_id}",
    )


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value > 0


def _positive_int(value: Any, label: str) -> int:
    if not _is_positive_int(value):
        raise RunnerError(INVALID_MAC_ESTIMATE, f"{label} must be a positive integer")
    return int(value)


__all__ = [
    "ModelOperationCounts",
    "count_model_operations",
    "derive_hardware_rates",
]
