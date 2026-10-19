"""Trace numerical error and ADC saturation through an MNSIM network."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from MNSIM.Interface import quantize


_SUPPORTED_ADC_ACTIONS = frozenset({"SCALE", "FIX", "NONE"})
_SUPPORTED_CALIBRATION_MODES = frozenset({"frozen_reference"})


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(
            "reference and PIM tensors have different shapes: "
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
        )

    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    finite_count = int(finite.sum().item())
    value_count = int(reference.numel())
    reference_safe = torch.nan_to_num(reference)
    candidate_safe = torch.nan_to_num(candidate)
    difference = candidate_safe - reference_safe
    reference_l2 = float(torch.linalg.vector_norm(reference_safe).item())
    error_l2 = float(torch.linalg.vector_norm(difference).item())
    denominator = float(
        torch.linalg.vector_norm(candidate_safe).item() * reference_l2
    )
    cosine_similarity = (
        float(torch.sum(candidate_safe * reference_safe).item()) / denominator
        if denominator
        else 1.0
    )
    return {
        "shape": list(reference.shape),
        "value_count": value_count,
        "finite_pair_count": finite_count,
        "nonfinite_pair_count": value_count - finite_count,
        "reference_l2": reference_l2,
        "error_l2": error_l2,
        "relative_l2": error_l2 / reference_l2 if reference_l2 else 0.0,
        "mean_absolute_error": float(torch.mean(torch.abs(difference)).item()),
        "max_absolute_error": float(torch.max(torch.abs(difference)).item()),
        "cosine_similarity": cosine_similarity,
    }


def _layer_input(
    tensors: Sequence[torch.Tensor], input_index: Sequence[int], layer_index: int
) -> torch.Tensor | list[torch.Tensor]:
    inputs = [tensors[index + layer_index + 1] for index in input_index]
    return inputs[0] if len(inputs) == 1 else inputs


def _quantization_payload(
    state: quantize.ActivationQuantizationState
    | list[quantize.ActivationQuantizationState],
) -> dict[str, Any] | list[dict[str, Any]]:
    if isinstance(state, list):
        return [_quantization_payload(item) for item in state]
    return {"activation_bit": state.bit, "activation_scale": state.scale}


def audit_pim_numerics(
    network: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    adc_action: str = "SCALE",
    calibration_mode: str = "frozen_reference",
) -> dict[str, Any]:
    """Compare FIX_TRAIN and PIM paths for one input batch.

    The reference and PIM branches advance through the graph in lockstep. That
    makes each layer's error attributable to the preceding PIM state while
    preserving the reference quantization scales used by normal inference.
    ``NONE`` is an audit-only ADC action: it preserves bit slicing and output
    quantization while bypassing the ADC clamp. ``frozen_reference`` fixes
    BatchNorm and post-add activation scales remain frozen during the audit.
    Call ``NetworkGraph.calibrate_activation_scales`` before this function when
    a checkpoint does not yet contain a calibrated activation boundary.
    """

    if adc_action not in _SUPPORTED_ADC_ACTIONS:
        raise ValueError(
            "adc_action must be one of "
            f"{sorted(_SUPPORTED_ADC_ACTIONS)}, got {adc_action!r}"
        )
    if calibration_mode not in _SUPPORTED_CALIBRATION_MODES:
        raise ValueError(
            "calibration_mode must be one of "
            f"{sorted(_SUPPORTED_CALIBRATION_MODES)}, got {calibration_mode!r}"
        )
    if not hasattr(network, "layer_list") or not hasattr(network, "input_index_list"):
        raise TypeError("network must be an MNSIM NetworkGraph")

    original_training = network.training
    original_quantize_state = (
        quantize.last_activation_scale,
        quantize.last_weight_scale,
        quantize.last_activation_bit,
        quantize.last_weight_bit,
    )
    network.eval()
    adc_diagnostics: list[dict[str, Any]] = []
    layer_results: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            input_quantization = network.input_activation_quantization()
            quantize._publish_legacy_activation_quantization_state(
                input_quantization
            )
            reference_tensors = [input_tensor]
            pim_tensors = [input_tensor]
            reference_quantizations = [input_quantization]
            pim_quantizations = [input_quantization]

            for layer_index, layer in enumerate(network.layer_list):
                input_index = network.input_index_list[layer_index]
                reference_input = _layer_input(
                    reference_tensors, input_index, layer_index
                )
                pim_input = _layer_input(pim_tensors, input_index, layer_index)
                reference_input_quantization = _layer_input(
                    reference_quantizations, input_index, layer_index
                )
                pim_input_quantization = _layer_input(
                    pim_quantizations, input_index, layer_index
                )
                reference_output = layer.forward(
                    reference_input,
                    "FIX_TRAIN",
                    adc_action,
                    input_quantization=reference_input_quantization,
                )
                reference_output_quantization = layer.output_activation_quantization(
                    reference_input_quantization
                )

                weighted = isinstance(layer, quantize.QuantizeLayer)
                if weighted:
                    pim_output = layer.set_weights_forward(
                        pim_input,
                        layer.get_bit_weights(),
                        adc_action,
                        adc_diagnostics=adc_diagnostics,
                        layer_index=layer_index,
                        input_quantization=pim_input_quantization,
                    )
                else:
                    pim_output = layer.forward(
                        pim_input,
                        "FIX_TRAIN",
                        None,
                        input_quantization=pim_input_quantization,
                    )
                pim_output_quantization = layer.output_activation_quantization(
                    pim_input_quantization
                )
                reference_tensors.append(reference_output)
                pim_tensors.append(pim_output)
                reference_quantizations.append(reference_output_quantization)
                pim_quantizations.append(pim_output_quantization)
                layer_results.append(
                    {
                        "network_layer_index": layer_index,
                        "layer_type": layer.layer_config["type"],
                        "input_index": list(input_index),
                        "weighted": weighted,
                        "input_quantization": _quantization_payload(
                            reference_input_quantization
                        ),
                        "output_quantization": _quantization_payload(
                            reference_output_quantization
                        ),
                        "metrics": _tensor_metrics(reference_output, pim_output),
                    }
                )
    finally:
        (
            quantize.last_activation_scale,
            quantize.last_weight_scale,
            quantize.last_activation_bit,
            quantize.last_weight_bit,
        ) = original_quantize_state
        network.train(original_training)

    diagnostics_by_layer = {
        entry["network_layer_index"]: entry for entry in adc_diagnostics
    }
    for layer_result in layer_results:
        diagnostic = diagnostics_by_layer.get(layer_result["network_layer_index"])
        if diagnostic is not None:
            layer_result["adc"] = diagnostic

    conversion_count = sum(
        entry["conversion_count"] for entry in adc_diagnostics
    )
    saturation_count = sum(
        entry["saturation_count"] for entry in adc_diagnostics
    )
    input_code_count = sum(entry["input_code_count"] for entry in adc_diagnostics)
    input_saturation_count = sum(
        entry["input_saturation_count"] for entry in adc_diagnostics
    )
    return {
        "schema_version": 1,
        "status": "succeeded",
        "adc_action": adc_action,
        "calibration_mode": calibration_mode,
        "input_shape": list(input_tensor.shape),
        "weighted_layer_count": sum(
            1 for layer_result in layer_results if layer_result["weighted"]
        ),
        "layers": layer_results,
        "output": {
            "metrics": _tensor_metrics(reference_tensors[-1], pim_tensors[-1]),
            "reference_top1": reference_tensors[-1].argmax(dim=1).tolist(),
            "pim_top1": pim_tensors[-1].argmax(dim=1).tolist(),
        },
        "adc_summary": {
            "conversion_count": conversion_count,
            "saturation_count": saturation_count,
            "saturation_fraction": (
                saturation_count / conversion_count if conversion_count else 0.0
            ),
        },
        "input_quantization_summary": {
            "code_count": input_code_count,
            "saturation_count": input_saturation_count,
            "saturation_fraction": (
                input_saturation_count / input_code_count
                if input_code_count
                else 0.0
            ),
        },
    }


__all__ = ["audit_pim_numerics"]
