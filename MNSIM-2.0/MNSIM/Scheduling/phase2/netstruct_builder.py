"""Build item-local MNSIM NetStruct objects for Phase 2."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def build_netstruct(layer_infos: list[dict[str, Any]]) -> list[list[tuple[dict[str, Any], None]]]:
    netstruct = []
    layer_count = len(layer_infos)
    for local_layer_id, layer_info in enumerate(layer_infos):
        local_info = deepcopy(dict(layer_info))
        local_info.setdefault("Layerindex", local_layer_id)
        local_info.setdefault("Inputindex", [-1] if local_layer_id == 0 else [-1])
        local_info.setdefault("Outputindex", [1] if local_layer_id + 1 < layer_count else [])
        local_info.setdefault("row_split_num", 1)
        local_info.setdefault("weight_bit_split_part", 1)
        netstruct.append([(local_info, None)])
    return netstruct


__all__ = ["build_netstruct"]
