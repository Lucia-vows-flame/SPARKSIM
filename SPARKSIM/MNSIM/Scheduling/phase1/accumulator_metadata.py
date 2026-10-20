"""Compile ADC-derived accumulator domains for real MNSIM SNN graphs."""

from __future__ import annotations

import configparser
import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from MNSIM.Scheduling.accumulator_domain import (
    build_completed_accumulator_profile,
    derive_fixed_range_adc_lsb_scale,
)
from MNSIM.Scheduling.digital_domain import (
    AccumulatorDomain,
    compile_lif_constants,
)
from MNSIM.Scheduling.round_types import INVALID_LIF_DOMAIN, Phase1Invalid


def attach_snn_accumulator_metadata(
    network: Any,
    layer_infos: list[dict[str, Any]],
    sim_config: str | Path,
) -> None:
    """Attach one completed-accumulator domain to every weighted->LIF pair."""

    parser = configparser.ConfigParser()
    if not parser.read(sim_config):
        raise Phase1Invalid(INVALID_LIF_DOMAIN, f"cannot read SimConfig {sim_config}")
    adc_bits = parser.getint("Interface level", "ADC_Precision")
    xbar_rows = int(
        parser.get("Crossbar level", "Xbar_Size").split(",", 1)[0].strip()
    )
    device_levels = parser.getint("Device level", "Device_Level")
    device_bits = int(math.log2(device_levels))

    layers = list(getattr(network, "layer_list", []))
    if len(layers) != len(layer_infos):
        raise Phase1Invalid(INVALID_LIF_DOMAIN, "network layer metadata length mismatch")

    for index, (layer, info) in enumerate(zip(layers, layer_infos)):
        if str(info.get("type")) not in {"conv", "fc"}:
            continue
        # V7 input events and every explicit LIF output are binary spikes.
        # This also covers weighted layers whose input arrives through a
        # multi-branch channel concat rather than an immediately preceding LIF.
        info["Inputbit"] = 1
        lif_index = index + 1
        # SW-BN SNN graphs keep BatchNorm between the weighted accumulator
        # and its LIF boundary. The BN is a zero-xbar domain-preserving node.
        bn_index = None
        if lif_index < len(layer_infos) and layer_infos[lif_index].get("type") == "bn":
            bn_index = lif_index
            lif_index += 1
        if lif_index >= len(layer_infos) or layer_infos[lif_index].get("type") != "lif":
            continue

        weight_scale = Fraction(str(float(layer.bit_scale_list[1, 1].item())))
        if weight_scale <= 0:
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_{info['type']} has no positive checkpoint weight scale",
            )
        weight_bits = int(layer.quantize_config["weight_bit"])
        weight_planes = math.ceil((weight_bits - 1) / device_bits)
        row_partials = len(layer.sublayer_list)
        q_bits = adc_bits + int(layer.layer_config.get("extend_ADC_bitwidth", 0))

        # A logical spike is one full-scale DAC event (0 or 1). The ADC code
        # scale therefore uses the binary spike range, not an ANN activation
        # output scale or a post-ADC quantizer.
        adc_lsb = derive_fixed_range_adc_lsb_scale(
            input_code_scale=Fraction(1, 1),
            weight_code_scale=weight_scale,
            input_digit_max=1,
            weight_digit_max=(1 << device_bits) - 1,
            xbar_rows=xbar_rows,
            adc_signed_bits=q_bits,
            adc_mode="FIX",
        )
        profile = build_completed_accumulator_profile(
            adc_signed_bits=q_bits,
            activation_cycles=1,
            input_bits_per_cycle=1,
            weight_planes=weight_planes,
            weight_bits_per_plane=device_bits,
            row_partials=row_partials,
            adc_lsb_scale=adc_lsb,
        )
        domain = AccumulatorDomain.from_scale(
            signed_bit_width=profile.signed_bit_width,
            scale=profile.accumulator_lsb_scale,
            source="fixed_adc_completed_accumulator",
        )
        lif_info = layer_infos[lif_index]
        neuron_count = _output_neuron_count(info)
        lif_info["output_neuron_region"] = [0, neuron_count]
        constants = compile_lif_constants(
            domain=domain,
            threshold=Fraction(str(lif_info.get("v_th", 0.5))),
            reset=Fraction(str(lif_info.get("v_reset", 0.0))),
        )
        info["accumulator_domain"] = domain.to_dict()
        if bn_index is not None:
            layer_infos[bn_index]["accumulator_domain"] = domain.to_dict()
        lif_info["accumulator_domain"] = domain.to_dict()
        lif_info["compiled_lif_constants"] = constants.to_dict()

    from MNSIM.Scheduling.phase1.residual_metadata import (
        attach_residual_metadata, is_residual_graph,
    )
    if is_residual_graph(network):
        attach_residual_metadata(network, layer_infos)
    attach_concat_metadata(network, layer_infos)


def attach_concat_metadata(network: Any, layer_infos: list[dict[str, Any]]) -> None:
    """Record time alignment and non-overlapping channel ownership for concat."""

    input_indices = list(getattr(network, "input_index_list", []))
    for index, info in enumerate(layer_infos):
        if str(info.get("type")) != "concat":
            continue
        parents = [index + int(offset) for offset in input_indices[index]]
        if len(parents) < 2 or any(parent < 0 or parent >= index for parent in parents):
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_concat requires at least two earlier graph inputs",
            )
        if info.get("time_policy", "same_timestep") != "same_timestep":
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_concat only supports same-timestep branch alignment",
            )
        if any(layer_infos[parent].get("type") != "lif" for parent in parents):
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_concat inputs must be explicit binary-spike LIF outputs",
            )
        input_shapes = [tuple(map(int, shape)) for shape in info.get("Inputshapes", [])]
        if len(input_shapes) != len(parents) or any(len(shape) != 4 for shape in input_shapes):
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_concat requires NCHW input shape metadata for every branch",
            )
        reference = input_shapes[0]
        if any(shape[0] != reference[0] or shape[2:] != reference[2:] for shape in input_shapes[1:]):
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_concat branches must have identical batch and spatial shapes",
            )
        concat_dim = info.get("concat_dim", "channel")
        if concat_dim not in {"channel", 1, -3}:
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_concat only supports channel concatenation for V7 SNNs",
            )
        channels = [shape[1] for shape in input_shapes]
        spatial_count = math.prod(reference[2:])
        regions = []
        offset = 0
        for channel_count in channels:
            width = channel_count * spatial_count
            regions.append([offset, offset + width])
            offset += width
        if int(info.get("Outputchannel", 0)) != sum(channels):
            raise Phase1Invalid(
                INVALID_LIF_DOMAIN,
                f"L{index:03d}_concat output channel count does not match its branches",
            )
        info["merge_mode"] = "channel_concat"
        info["merge_ownership_policy"] = "concat_channel_segments"
        info["merge_input_regions"] = regions
        info["time_policy"] = "same_timestep"
        info["value_domain"] = "binary_spike"
        info["Inputbit"] = 1
        info["Outputbit"] = 1
        info["outputbit"] = 1

    for info in layer_infos:
        if str(info.get("type")) == "lif":
            info["Outputbit"] = 1
            info["outputbit"] = 1


def _output_neuron_count(layer_info: dict[str, Any]) -> int:
    layer_type = str(layer_info.get("type"))
    if layer_type == "conv":
        output_size = list(layer_info.get("Outputsize", []))
        if len(output_size) != 2:
            raise Phase1Invalid(INVALID_LIF_DOMAIN, "conv output size must be 2-D")
        return (
            int(layer_info["Outputchannel"])
            * int(output_size[0])
            * int(output_size[1])
        )
    if layer_type == "fc":
        return int(layer_info["Outfeature"])
    raise Phase1Invalid(
        INVALID_LIF_DOMAIN, f"unsupported LIF producer type {layer_type}"
    )


__all__ = ["attach_snn_accumulator_metadata"]
