"""MNSIM NetworkGraph-to-ModelIR frontend normalizer for Phase 1.

The Phase 1 core still consumes ``ModelIR``. This module is the official
MNSIM-facing Phase 1 input frontend: it normalizes an existing
``NetworkGraph.get_structure()`` result into that schema without running TCG,
Model_* metric evaluation, Phase 2 lowering, or real weight loading.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from MNSIM.Scheduling.non_weighted_layers import (
    get_non_weighted_layer_spec,
    normalize_registered_mnsim_layer_type,
)
from MNSIM.Scheduling.round_types import (
    INVALID_LAYER_SHAPE,
    INVALID_MODEL_DAG,
    LayerIR,
    ModelIR,
    Phase1Invalid,
    SequenceSpec,
    UNSUPPORTED_LAYER_TYPE,
    WEIGHTED_LAYER_TYPES,
)


FRONTEND_PHASE = "phase1"
FRONTEND_KIND = "mnsim_networkgraph"

__all__ = [
    "FRONTEND_KIND",
    "FRONTEND_PHASE",
    "build_model_ir_from_mnsim_network",
    "build_model_ir_from_mnsim_structure",
]


_SUPPORTED_MNSIM_LAYER_TYPES = {
    "conv": "conv",
    "fc": "fc",
    "bn": "bn",
    "relu": "relu",
    "pooling": "pooling",
    "view": "view",
    "dropout": "dropout",
    "concat": "concat",
    "element_sum": "element_sum",
    "element_multiply": "element_multiply",
}


def build_model_ir_from_mnsim_network(
    network: Any,
    model_id: str,
    backend_kind: str = "ann",
    sequence_spec: SequenceSpec | None = None,
    sim_config: str | None = None,
) -> ModelIR:
    if not hasattr(network, "get_structure"):
        raise Phase1Invalid(INVALID_MODEL_DAG, "network object does not expose get_structure()")
    layer_infos = network.get_structure()
    if sim_config is None:
        sim_config = getattr(network, "simconfig_path", None)
    if backend_kind == "snn" and sim_config is not None:
        from MNSIM.Scheduling.phase1.accumulator_metadata import (
            attach_snn_accumulator_metadata,
        )

        attach_snn_accumulator_metadata(network, layer_infos, sim_config)
    if sequence_spec is None and backend_kind == "snn":
        raw_length = getattr(network, "sequence_length", None)
        if raw_length is not None:
            recurrent_ids = [
                _layer_id(index, _normalize_layer_type(info, index))
                for index, info in enumerate(layer_infos)
                if _normalize_layer_type(info, index) == "lif"
            ]
            sequence_spec = SequenceSpec(int(raw_length), True, True, recurrent_ids)
    return build_model_ir_from_mnsim_structure(
        layer_infos,
        model_id=model_id,
        backend_kind=backend_kind,
        sequence_spec=sequence_spec,
    )


def build_model_ir_from_mnsim_structure(
    layer_infos: Sequence[dict[str, Any]],
    model_id: str,
    backend_kind: str = "ann",
    sequence_spec: SequenceSpec | None = None,
) -> ModelIR:
    normalized_types = [_normalize_layer_type(layer_info, index) for index, layer_info in enumerate(layer_infos)]
    input_layer_ids_by_index = [
        _input_layer_ids(index, layer_info, normalized_types)
        for index, layer_info in enumerate(layer_infos)
    ]
    layer_ids = [_layer_id(index, layer_type) for index, layer_type in enumerate(normalized_types)]
    if sequence_spec is None:
        recurrent_state_ids = [
            layer_id
            for layer_id, layer_type in zip(layer_ids, normalized_types)
            if layer_type == "lif"
        ]
        sequence_spec = SequenceSpec(
            sequence_length=1,
            ordered_steps=bool(recurrent_state_ids),
            cross_step_dependency=bool(recurrent_state_ids),
            recurrent_state_ids=recurrent_state_ids,
        )
    output_consumer_ids_by_index = _output_consumer_ids(input_layer_ids_by_index, layer_ids)

    layers: list[LayerIR] = []
    for index, layer_info in enumerate(layer_infos):
        layer_type = normalized_types[index]
        layer_id = _layer_id(index, layer_type)
        input_layer_ids = input_layer_ids_by_index[index]
        is_weighted = layer_type in WEIGHTED_LAYER_TYPES
        layers.append(
            LayerIR(
                layer_id=layer_id,
                layer_type=layer_type,
                is_weighted=is_weighted,
                stable_topological_index=index,
                input_tensor_ids=_input_tensor_ids(input_layer_ids),
                output_tensor_ids=[_output_tensor_id(layer_id)],
                input_layer_ids=input_layer_ids,
                output_consumer_ids=output_consumer_ids_by_index[index],
                weight_ref_id=f"{model_id}:{layer_id}:weight" if is_weighted else None,
                shape=_shape(layer_info, layer_id, layer_type),
                backend_attrs=_backend_attrs(layer_info, index),
            )
        )

    return ModelIR(
        model_id=model_id,
        backend_kind=backend_kind,
        sequence_spec=sequence_spec,
        layers=layers,
    )


def _normalize_layer_type(layer_info: dict[str, Any], index: int) -> str:
    raw_type = str(layer_info.get("type", ""))
    registered_type = normalize_registered_mnsim_layer_type(raw_type)
    if registered_type is not None:
        return registered_type
    if raw_type not in _SUPPORTED_MNSIM_LAYER_TYPES:
        raise Phase1Invalid(
            UNSUPPORTED_LAYER_TYPE,
            f"MNSIM layer {index} has unsupported type {raw_type}",
        )
    return _SUPPORTED_MNSIM_LAYER_TYPES[raw_type]


def _layer_id(index: int, layer_type: str) -> str:
    return f"L{index:03d}_{layer_type}"


def _input_layer_ids(
    layer_index: int,
    layer_info: dict[str, Any],
    normalized_types: Sequence[str],
) -> list[str]:
    input_ids: list[str] = []
    for relative_index in _relative_input_indices(layer_info):
        producer_index = layer_index + relative_index
        if producer_index < 0:
            continue
        if producer_index >= layer_index:
            raise Phase1Invalid(
                INVALID_MODEL_DAG,
                f"MNSIM layer {layer_index} references non-earlier producer offset {relative_index}",
            )
        input_ids.append(_layer_id(producer_index, normalized_types[producer_index]))
    return _dedupe_preserve_order(input_ids)


def _relative_input_indices(layer_info: dict[str, Any]) -> list[int]:
    value = layer_info.get("Inputindex", [-1])
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    raise Phase1Invalid(INVALID_MODEL_DAG, f"invalid Inputindex value {value!r}")


def _output_consumer_ids(
    input_layer_ids_by_index: Sequence[list[str]],
    layer_ids: Sequence[str],
) -> list[list[str]]:
    consumers = [[] for _ in range(len(layer_ids))]
    for consumer_index, input_layer_ids in enumerate(input_layer_ids_by_index):
        consumer_id = layer_ids[consumer_index]
        for producer_id in input_layer_ids:
            producer_index = _layer_index_from_id(producer_id)
            consumers[producer_index].append(consumer_id)
    return consumers


def _input_tensor_ids(input_layer_ids: Sequence[str]) -> list[str]:
    if not input_layer_ids:
        return ["input:0"]
    return [_output_tensor_id(layer_id) for layer_id in input_layer_ids]


def _output_tensor_id(layer_id: str) -> str:
    return f"{layer_id}:out"


def _shape(layer_info: dict[str, Any], layer_id: str, layer_type: str) -> dict[str, Any]:
    if layer_type == "conv":
        return _weighted_shape(
            layer_info,
            layer_id,
            required_keys=("Inputchannel", "Outputchannel", "Kernelsize"),
        )
    if layer_type == "fc":
        return _weighted_shape(
            layer_info,
            layer_id,
            required_keys=("Infeature", "Outfeature"),
        )
    return {
        key: _json_value(value)
        for key, value in layer_info.items()
        if key not in {"type", "Inputindex", "Outputindex"}
    }


def _weighted_shape(
    layer_info: dict[str, Any],
    layer_id: str,
    required_keys: Sequence[str],
) -> dict[str, Any]:
    shape = {
        key: _json_value(value)
        for key, value in layer_info.items()
        if key not in {"type", "Inputindex", "Outputindex"}
    }
    for key in required_keys:
        shape[key] = _required_int(layer_info, layer_id, key)
    return shape


def _required_int(layer_info: dict[str, Any], layer_id: str, key: str) -> int:
    if key not in layer_info:
        raise Phase1Invalid(INVALID_LAYER_SHAPE, f"{layer_id} missing MNSIM layer_info field {key}")
    value = int(layer_info[key])
    if value <= 0:
        raise Phase1Invalid(INVALID_LAYER_SHAPE, f"{layer_id} layer_info field {key} must be positive")
    return value


def _backend_attrs(layer_info: dict[str, Any], index: int) -> dict[str, Any]:
    attrs = {
        "mnsim_layer_index": index,
        "mnsim_layer_info": _json_value(dict(layer_info)),
    }
    registered_spec = get_non_weighted_layer_spec(str(layer_info.get("type", "")))
    if registered_spec is not None:
        attrs.update(_json_value(registered_spec.structure_metadata(layer_info)))
    for key in ("accumulator_domain", "compiled_lif_constants", "lif_owner_layer_id",
                "snn_merge_mode", "merge_ownership_policy", "merge_source_ids",
                "merge_input_regions", "concat_dim", "time_policy", "merge_mode",
                "value_domain"):
        if key in layer_info:
            attrs[key] = _json_value(layer_info[key])
    return attrs


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(value[key]) for key in sorted(value.keys(), key=str)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _layer_index_from_id(layer_id: str) -> int:
    prefix = layer_id.split("_", 1)[0]
    if not prefix.startswith("L"):
        raise Phase1Invalid(INVALID_MODEL_DAG, f"invalid layer_id {layer_id}")
    return int(prefix[1:])
