from __future__ import annotations

from typing import Any
from MNSIM.Scheduling.depthwise import is_depthwise, depthwise_block_size

from MNSIM.Scheduling.phase1.config import HardwareProjection
from MNSIM.Scheduling.round_types import (
    CompletionContract,
    INVALID_LAYER_SHAPE,
    INVALID_RESOURCE_FOOTPRINT,
    LayerIR,
    LayerResourceDescriptor,
    ModelIR,
    Phase1Invalid,
    PrepassRef,
    UNSUPPORTED_LAYER_TYPE,
    WEIGHTED_LAYER_TYPES,
)

RESOURCE_VERSION = "mnsim20_physical_xbar_v2"


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise Phase1Invalid(INVALID_RESOURCE_FOOTPRINT, f"invalid denominator {denominator}")
    return (numerator + denominator - 1) // denominator


def _positive_int(shape: dict[str, Any], key: str, layer: LayerIR) -> int:
    if key not in shape:
        raise Phase1Invalid(INVALID_LAYER_SHAPE, f"{layer.layer_id} missing shape field {key}")
    value = int(shape[key])
    if value <= 0:
        raise Phase1Invalid(INVALID_LAYER_SHAPE, f"{layer.layer_id} shape field {key} must be positive")
    return value


def _weight_bit_split_part(layer: LayerIR, hw: HardwareProjection) -> int:
    if "Weightbit" in layer.shape:
        native_weightbit = _positive_int(layer.shape, "Weightbit", layer)
        precision_bits = (
            native_weightbit - 1
            if hw.xbar_polarity == 2
            else native_weightbit
        )
        if precision_bits <= 0:
            raise Phase1Invalid(
                INVALID_LAYER_SHAPE,
                f"{layer.layer_id} Weightbit={native_weightbit} has no magnitude bits",
            )
    else:
        # The SimConfig fallback is already the effective magnitude precision.
        precision_bits = hw.weight_precision
    return _ceil_div(precision_bits, hw.device_bits)


def _required_xbars(layer: LayerIR, hw: HardwareProjection) -> int:
    if layer.layer_type in ("fc", "fc_like_projection"):
        rows = _positive_int(layer.shape, "Infeature", layer)
        cols = _positive_int(layer.shape, "Outfeature", layer)
        row_split_count = _ceil_div(rows, hw.xbar_row)
        output_subset_count = _ceil_div(cols, hw.xbar_col)
    elif layer.layer_type == "conv":
        input_channels = _positive_int(layer.shape, "Inputchannel", layer)
        output_channels = _positive_int(layer.shape, "Outputchannel", layer)
        kernel = _positive_int(layer.shape, "Kernelsize", layer)
        kernel_area = kernel * kernel
        if kernel_area > hw.xbar_row:
            raise Phase1Invalid(
                INVALID_LAYER_SHAPE,
                f"{layer.layer_id} Kernelsize*Kernelsize={kernel_area} exceeds xbar_row={hw.xbar_row}",
            )
        input_channels_per_xbar = hw.xbar_row // kernel_area
        if is_depthwise(layer.shape):
            try:
                block = depthwise_block_size(layer.shape, hw.xbar_row, hw.xbar_col)
            except ValueError as error:
                raise Phase1Invalid(INVALID_LAYER_SHAPE, str(error)) from error
            return _ceil_div(input_channels, block) * _weight_bit_split_part(layer, hw) * hw.xbar_polarity
        row_split_count = _ceil_div(input_channels, input_channels_per_xbar)
        output_subset_count = _ceil_div(output_channels, hw.xbar_col)
    else:
        raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"unsupported weighted layer type {layer.layer_type}")

    required = (
        row_split_count
        * output_subset_count
        * _weight_bit_split_part(layer, hw)
        * hw.xbar_polarity
    )
    if required <= 0:
        raise Phase1Invalid(INVALID_RESOURCE_FOOTPRINT, f"{layer.layer_id} has non-positive footprint {required}")
    return int(required)


def _prepass_ref(model: ModelIR, layer: LayerIR, hw: HardwareProjection) -> PrepassRef:
    shape_signature = dict(layer.shape)
    sim_fingerprint = (
        f"xbar={hw.xbar_row}x{hw.xbar_col};"
        f"device={hw.device_level};weight={hw.weight_precision};polarity={hw.xbar_polarity}"
    )
    layer_fingerprint = f"{layer.layer_type}:{sorted(shape_signature.items())}"
    return PrepassRef(
        ref_id=f"{model.model_id}:{layer.layer_id}:{RESOURCE_VERSION}",
        model_id=model.model_id,
        layer_id=layer.layer_id,
        layer_type=layer.layer_type,
        resource_version=RESOURCE_VERSION,
        sim_config_fingerprint=sim_fingerprint,
        layer_config_fingerprint=layer_fingerprint,
        weight_ref_id=layer.weight_ref_id,
        shape_signature=shape_signature,
    )


def build_layer_resource_descriptors(
    model_ir: ModelIR,
    hardware: HardwareProjection,
) -> list[LayerResourceDescriptor]:
    descriptors: list[LayerResourceDescriptor] = []
    for layer in sorted(model_ir.layers, key=lambda node: node.stable_topological_index):
        if not layer.is_weighted:
            continue
        if layer.layer_type not in WEIGHTED_LAYER_TYPES:
            raise Phase1Invalid(UNSUPPORTED_LAYER_TYPE, f"unsupported weighted layer type {layer.layer_type}")
        descriptors.append(
            LayerResourceDescriptor(
                layer_id=layer.layer_id,
                layer_type=layer.layer_type,
                input_layer_ids=list(layer.input_layer_ids),
                output_consumer_ids=list(layer.output_consumer_ids),
                required_xbar_count=_required_xbars(layer, hardware),
                prepass_ref=_prepass_ref(model_ir, layer, hardware),
                completion_contract=CompletionContract(),
            )
        )
    return descriptors


__all__ = [
    "RESOURCE_VERSION",
    "build_layer_resource_descriptors",
]
