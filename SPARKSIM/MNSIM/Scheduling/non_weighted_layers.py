"""Registry for MNSIM non-weighted layers added by this project.

MNSIM already has built-in straight-through layers such as relu, bn, view, and
pooling. This registry is for project-specific non-weighted operators that
must be visible to both NetworkGraph/quantize and Phase 1 normalization.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron

from MNSIM.Accuracy_Model.snn_lif_perturbation import (
    LIFPerturbationArtifactError,
)


PHASE1_NODE_ROLE_HOOK = "hook"


class _AccuracySpikingJellyLIF(neuron.LIFNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tensor_tau = None
        self._tensor_v_threshold = None
        self._tensor_v_reset = None

    def install_tensor_parameters(self, tau, v_threshold, v_reset):
        self._tensor_tau = tau
        self._tensor_v_threshold = v_threshold
        self._tensor_v_reset = v_reset

    def clear_tensor_parameters(self):
        self._tensor_tau = None
        self._tensor_v_threshold = None
        self._tensor_v_reset = None

    def _has_tensor_parameters(self):
        return self._tensor_tau is not None

    def single_step_forward(self, x):
        if not self._has_tensor_parameters():
            return super().single_step_forward(x)
        return neuron.BaseNode.single_step_forward(self, x)

    def multi_step_forward(self, x_seq):
        if not self._has_tensor_parameters():
            return super().multi_step_forward(x_seq)
        return neuron.BaseNode.multi_step_forward(self, x_seq)

    def neuronal_charge(self, x):
        if not self._has_tensor_parameters():
            return super().neuronal_charge(x)
        self.v = self.v + (
            x - (self.v - self._tensor_v_reset)
        ) / self._tensor_tau

    def neuronal_fire(self):
        if not self._has_tensor_parameters():
            return super().neuronal_fire()
        return self.surrogate_function(self.v - self._tensor_v_threshold)

    def neuronal_reset(self, spike):
        if not self._has_tensor_parameters():
            return super().neuronal_reset(spike)
        spike_d = spike.detach() if self.detach_reset else spike
        self.v = (
            (1.0 - spike_d) * self.v
            + spike_d * self._tensor_v_reset
        )


class LIFNode(nn.Module):
    def __init__(self, tau: float = 2.0, v_th: float = 1.0, v_reset: float = 0.0, alpha: float = 2.0):
        super().__init__()
        self.tau = float(tau)
        self.v_th = float(v_th)
        self.v_reset = float(v_reset)
        self.alpha = float(alpha)
        self.node = _AccuracySpikingJellyLIF(
            tau=self.tau,
            decay_input=True,
            v_threshold=self.v_th,
            v_reset=self.v_reset,
            step_mode="s",
            backend="torch",
        )
        self._accuracy_runtime = None
        self._accuracy_layer_id = None
        self._sampled_accuracy_parameters = None

    def set_accuracy_perturbation(self, runtime, *, layer_id):
        if runtime is self._accuracy_runtime and layer_id == self._accuracy_layer_id:
            return
        self._accuracy_runtime = runtime
        self._accuracy_layer_id = layer_id
        self._sampled_accuracy_parameters = None
        self.node.clear_tensor_parameters()

    def sampled_accuracy_parameters(self):
        return self._sampled_accuracy_parameters

    def _prepare_accuracy_parameters(self, x_seq):
        if self._accuracy_runtime is None:
            self.node.clear_tensor_parameters()
            return None
        physical_shape = (1, *x_seq.shape[2:])
        if self._sampled_accuracy_parameters is None:
            self._sampled_accuracy_parameters = (
                self._accuracy_runtime.static_parameters(
                    self._accuracy_layer_id,
                    physical_shape,
                )
            )
        tensors = {
            name: torch.as_tensor(value, device=x_seq.device, dtype=x_seq.dtype)
            for name, value in self._sampled_accuracy_parameters.items()
        }
        tau = self.tau * (1.0 + tensors["delta_tau"])
        if not torch.isfinite(tau).all() or torch.any(tau <= 0):
            raise LIFPerturbationArtifactError(
                "perturbed LIF tau must be finite and positive"
            )
        v_threshold = self.v_th + tensors["delta_v_th"]
        v_reset = self.v_reset + tensors["delta_v_reset"]
        self.node.install_tensor_parameters(tau, v_threshold, v_reset)
        return tensors

    def reset_state(self):
        functional.reset_net(self.node)

    def forward(self, x):
        self.node.step_mode = "s"
        return self.node(x)

    def forward_sequence(self, x):
        parameters = self._prepare_accuracy_parameters(x)
        if parameters is not None:
            x = parameters["frontend_gain"] * x + parameters["frontend_offset"]
            noise = torch.stack(
                [
                    torch.as_tensor(
                        self._accuracy_runtime.dynamic_noise(
                            self._accuracy_layer_id,
                            tuple(x[timestep].shape),
                        ),
                        device=x.device,
                        dtype=x.dtype,
                    )
                    for timestep in range(x.shape[0])
                ]
            )
            x = x + noise
            self._accuracy_runtime.validate_operating_range(x)
        self.node.step_mode = "m"
        try:
            return self.node(x)
        finally:
            self.node.step_mode = "s"


@dataclass(frozen=True)
class NonWeightedLayerSpec:
    layer_type: str
    phase1_node_role: str
    straight_layer_factory: Callable[[dict[str, Any]], nn.Module]
    structure_metadata_factory: Callable[[dict[str, Any]], dict[str, Any]]

    def create_straight_layer(self, layer_config: dict[str, Any]) -> nn.Module:
        return self.straight_layer_factory(layer_config)

    def structure_metadata(self, layer_config: dict[str, Any]) -> dict[str, Any]:
        metadata = self.structure_metadata_factory(layer_config)
        metadata["type"] = self.layer_type
        return metadata


def _create_lif_layer(layer_config: dict[str, Any]) -> LIFNode:
    return LIFNode(
        tau=layer_config.get("tau", 2.0),
        v_th=layer_config.get("v_th", 0.5),
        v_reset=layer_config.get("v_reset", 0.0),
        alpha=layer_config.get("alpha", 2.0),
    )


def _lif_structure_metadata(layer_config: dict[str, Any]) -> dict[str, Any]:
    metadata = {
        "tau": layer_config.get("tau", 2.0),
        "v_th": layer_config.get("v_th", 0.5),
        "v_reset": layer_config.get("v_reset", 0.0),
        "alpha": layer_config.get("alpha", 2.0),
        "reset_mode": layer_config.get("reset_mode", "hard"),
    }
    for key in ("accumulator_domain", "compiled_lif_constants"):
        if key in layer_config:
            metadata[key] = layer_config[key]
    return metadata


NON_WEIGHTED_LAYER_SPECS = {
    "lif": NonWeightedLayerSpec(
        layer_type="lif",
        phase1_node_role=PHASE1_NODE_ROLE_HOOK,
        straight_layer_factory=_create_lif_layer,
        structure_metadata_factory=_lif_structure_metadata,
    ),
}

REGISTERED_MNSIM_NON_WEIGHTED_LAYER_TYPES = frozenset(NON_WEIGHTED_LAYER_SPECS)
REGISTERED_PHASE1_HOOK_LAYER_TYPES = frozenset(
    layer_type
    for layer_type, spec in NON_WEIGHTED_LAYER_SPECS.items()
    if spec.phase1_node_role == PHASE1_NODE_ROLE_HOOK
)


def normalize_registered_mnsim_layer_type(layer_type: str) -> str | None:
    normalized = str(layer_type).lower()
    if normalized in NON_WEIGHTED_LAYER_SPECS:
        return normalized
    return None


def get_non_weighted_layer_spec(layer_type: str) -> NonWeightedLayerSpec | None:
    normalized = normalize_registered_mnsim_layer_type(layer_type)
    if normalized is None:
        return None
    return NON_WEIGHTED_LAYER_SPECS[normalized]


__all__ = [
    "LIFNode",
    "NonWeightedLayerSpec",
    "NON_WEIGHTED_LAYER_SPECS",
    "PHASE1_NODE_ROLE_HOOK",
    "REGISTERED_MNSIM_NON_WEIGHTED_LAYER_TYPES",
    "REGISTERED_PHASE1_HOOK_LAYER_TYPES",
    "get_non_weighted_layer_spec",
    "normalize_registered_mnsim_layer_type",
]
