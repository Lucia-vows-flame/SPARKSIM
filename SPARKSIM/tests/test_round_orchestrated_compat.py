from collections import OrderedDict

import torch

from round_orchestrated_compat import (
    enrich_legacy_element_sum_markers,
)


class _Layer:
    def __init__(self, layer_type):
        self.layer_config = {"type": layer_type}


class _Network:
    layer_list = [_Layer("conv"), _Layer("element_sum"), _Layer("element_sum")]


def test_legacy_checkpoint_markers_are_enriched_from_valid_ranges():
    state = OrderedDict(
        {
            "layer_list.1.last_value": torch.tensor([2.0]),
            "layer_list.2.last_value": torch.tensor([-1.0]),
        }
    )

    enriched = enrich_legacy_element_sum_markers(_Network(), state)

    assert enriched["layer_list.1.post_add_quantizer_version"].item() == 1
    assert enriched["layer_list.2.post_add_quantizer_version"].item() == 0
    assert "layer_list.1.post_add_quantizer_version" not in state


def test_existing_marker_is_preserved():
    state = OrderedDict(
        {
            "layer_list.1.last_value": torch.tensor([2.0]),
            "layer_list.1.post_add_quantizer_version": torch.tensor([0], dtype=torch.uint8),
        }
    )

    enriched = enrich_legacy_element_sum_markers(_Network(), state)

    assert enriched["layer_list.1.post_add_quantizer_version"].item() == 0
