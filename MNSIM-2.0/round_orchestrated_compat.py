"""Compatibility workarounds owned by the round-orchestrated layer.

The original MNSIM network and checkpoint loaders remain unchanged.  This
module prepares legacy state dictionaries before handing them to those
loaders.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


def enrich_legacy_element_sum_markers(network: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``state`` with legacy residual markers materialized.

    Older checkpoints have the residual output range but no marker introduced
    by the newer graph implementation.  The adapter derives the marker only
    for actual ``element_sum`` layers exposed by the target network.
    """

    enriched = dict(state)
    for index, layer in enumerate(getattr(network, "layer_list", ())):
        config = getattr(layer, "layer_config", {})
        if config.get("type") != "element_sum":
            continue
        prefix = f"layer_list.{index}."
        marker_key = prefix + "post_add_quantizer_version"
        if marker_key in enriched:
            continue
        legacy_range = enriched.get(prefix + "last_value")
        valid = (
            isinstance(legacy_range, torch.Tensor)
            and legacy_range.numel() == 1
            and bool(torch.isfinite(legacy_range).all().item())
            and float(legacy_range.item()) > 0
        )
        enriched[marker_key] = torch.ones(1, dtype=torch.uint8) if valid else torch.zeros(1, dtype=torch.uint8)
    return enriched


def load_network_weights_compat(network: Any, weights_file: str, map_location: Any = "cpu") -> None:
    """Load a checkpoint through the original network loader after adaptation."""

    state = torch.load(weights_file, map_location=map_location)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint must contain a state mapping, got {type(state).__name__}")
    if "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    network.load_change_weights(enrich_legacy_element_sum_markers(network, state))
