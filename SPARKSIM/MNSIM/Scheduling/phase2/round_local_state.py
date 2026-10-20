"""Round-local state for ordered Phase 2 segment execution."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from MNSIM.Scheduling.phase2.types import (
    CONTRIBUTION_PARTIAL,
    BoundaryBinding,
    BoundaryInputView,
    Phase2Invalid,
    RoundLocalStateRecord,
    StepExecutionResult,
    StepInputView,
)
from MNSIM.Scheduling.round_types import canonical_ordered_segments, step_id


class RoundLocalState:
    def __init__(self, item, boundary_input_view: BoundaryInputView) -> None:
        self.item = item
        self.boundary_input_view = boundary_input_view
        self._records_by_origin: dict[str, RoundLocalStateRecord] = {}
        self._bindings_by_origin: dict[str, list[BoundaryBinding]] = {}
        self._bindings_in_segment_order: list[BoundaryBinding] = []

    @classmethod
    def seed(cls, item, boundary_input_view: BoundaryInputView) -> "RoundLocalState":
        return cls(item=item, boundary_input_view=boundary_input_view)

    def input_view_for(self, segment) -> StepInputView:
        node_ids = _segment_node_ids(segment)
        producer_ids = _same_item_producer_ids(self.item, node_ids)
        local_state_records: list[RoundLocalStateRecord] = []
        for producer_id in producer_ids:
            record = self._records_by_origin.get(producer_id)
            if record is None:
                raise Phase2Invalid(
                    "ROUND_LOCAL_SOURCE_NOT_READY",
                    f"{self.item.item_id}: {producer_id} is not ready",
                )
            local_state_records.append(deepcopy(record))

        boundary_state_ids: list[str] = []
        boundary_state_records: list[Any] = []
        records_by_state_id = {
            str(record.state_id): record
            for record in self.boundary_input_view.state_records
        }
        readable_state_ids = {
            str(state_id)
            for state_id in self.boundary_input_view.readable_state_ids
        }
        for ref in _boundary_refs_for_nodes(self.item, node_ids):
            state_id = str(ref.state_id)
            if state_id not in readable_state_ids:
                # SNN views contain sample/timestep-qualified state IDs; graph
                # refs retain their stable producer ID across the sequence.
                producer_id = state_id.removeprefix('state:').removesuffix(':partial')
                matches = [str(record.state_id) for record in self.boundary_input_view.state_records
                           if record.sequence_length > 1
                           and record.producer_layer_id == producer_id
                           and str(record.state_id) in readable_state_ids]
                if len(matches) > 1:
                    raise Phase2Invalid('AMBIGUOUS_SEQUENCE_INPUT', f'{state_id}: multiple temporal states')
                if matches:
                    state_id = matches[0]
            if state_id not in readable_state_ids or state_id in boundary_state_ids:
                continue
            boundary_state_ids.append(state_id)
            record = records_by_state_id.get(state_id)
            if record is not None:
                boundary_state_records.append(deepcopy(record))

        return StepInputView(
            item_id=str(self.item.item_id),
            step_id=step_id(self.item.item_id, segment.segment_index),
            boundary_input_view_id=self.boundary_input_view.input_view_id,
            boundary_state_ids=boundary_state_ids,
            boundary_state_records=boundary_state_records,
            local_state_ids=[record.local_state_id for record in local_state_records],
            local_state_records=local_state_records,
        )

    def commit_step(self, result: StepExecutionResult) -> None:
        for binding in result.boundary_bindings:
            origin_layer_id = str(binding.origin_layer_id)
            readable_state_ids = {
                str(state_id)
                for state_id in self.boundary_input_view.readable_state_ids
            }
            for source_state_id in binding.source_state_ids:
                source_state_id = str(source_state_id)
                if source_state_id not in readable_state_ids:
                    raise Phase2Invalid(
                        "ROUND_LOCAL_INVALID_BOUNDARY_SOURCE",
                        f"{self.item.item_id}: {source_state_id} is not a readable boundary state",
                    )
            boundary_source_state_ids = list(binding.source_state_ids)
            for local_state_id in binding.source_local_state_ids:
                source_record = next(
                    (
                        record
                        for record in self._records_by_origin.values()
                        if record.local_state_id == str(local_state_id)
                    ),
                    None,
                )
                if source_record is None:
                    raise Phase2Invalid(
                        "ROUND_LOCAL_SOURCE_NOT_READY",
                        f"{self.item.item_id}: {local_state_id} is not ready",
                    )
                boundary_source_state_ids.extend(
                    source_record.boundary_source_state_ids
                )
            binding.source_state_ids = _unique_strings(boundary_source_state_ids)

            self._bindings_by_origin.setdefault(origin_layer_id, []).append(binding)
            self._bindings_in_segment_order.append(binding)

            previous = self._records_by_origin.get(origin_layer_id)
            boundary_source_state_ids = _unique_strings(
                [
                    *(previous.boundary_source_state_ids if previous is not None else []),
                    *binding.source_state_ids,
                ]
            )
            self._records_by_origin[origin_layer_id] = RoundLocalStateRecord(
                local_state_id=f"local:{self.item.item_id}:{origin_layer_id}",
                origin_layer_id=origin_layer_id,
                origin_layer_type=str(binding.origin_layer_type),
                shape=deepcopy(binding.shape),
                contribution_kind=str(binding.contribution_kind),
                boundary_source_state_ids=boundary_source_state_ids,
            )

    def boundary_bindings_for_finalize(self) -> list[BoundaryBinding]:
        external_state_ids = {
            str(ref.state_id)
            for ref in getattr(self.item, "external_outputs", [])
        }
        partial_state_ids = {
            str(ref.state_id)
            for ref in getattr(self.item, "partial_output_contract", [])
        }
        selected: list[BoundaryBinding] = []
        seen: set[tuple[Any, ...]] = set()
        for binding in self._bindings_in_item_segment_order():
            state_id = f"state:{binding.origin_layer_id}"
            if binding.contribution_kind == CONTRIBUTION_PARTIAL:
                include = state_id in partial_state_ids or state_id in external_state_ids
            else:
                include = state_id in external_state_ids
            if not include:
                continue
            # Partial bindings are distinct physical fragment contributions.
            # BoundaryStateManager owns their fragment-ID coverage and duplicate checks.
            if binding.contribution_kind == CONTRIBUTION_PARTIAL:
                selected.append(binding)
                continue
            dedupe_key = (
                str(binding.origin_layer_id),
                str(binding.contribution_kind),
                _hashable_value(binding.input_range),
                _hashable_value(binding.output_range),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            selected.append(binding)
        return selected

    def _bindings_in_item_segment_order(self) -> list[BoundaryBinding]:
        origin_order = _origin_segment_order(self.item)
        return [
            binding
            for _, binding in sorted(
                enumerate(self._bindings_in_segment_order),
                key=lambda indexed_binding: (
                    origin_order.get(
                        str(indexed_binding[1].origin_layer_id),
                        len(origin_order),
                    ),
                    indexed_binding[0],
                ),
            )
        ]


def _same_item_producer_ids(item, consumer_node_ids: list[str]) -> list[str]:
    consumer_node_id_set = set(consumer_node_ids)
    origin_order = _origin_segment_order(item)
    indexed_producers = [
        (edge_index, str(edge.producer_node_id))
        for edge_index, edge in enumerate(getattr(item, "same_item_edges", []))
        if str(edge.consumer_node_id) in consumer_node_id_set
    ]
    indexed_producers.sort(
        key=lambda indexed_producer: (
            origin_order.get(indexed_producer[1], len(origin_order)),
            indexed_producer[0],
        )
    )
    return _unique_strings(
        producer_id
        for _, producer_id in indexed_producers
    )


def _boundary_refs_for_nodes(item, node_ids: list[str]) -> list[Any]:
    node_id_set = set(node_ids)
    return [
        ref
        for ref in getattr(item, "external_inputs", [])
        if node_id_set.intersection(
            str(node_id)
            for node_id in getattr(ref, "consumer_node_ids", [])
        )
    ]


def _origin_segment_order(item) -> dict[str, int]:
    order: dict[str, int] = {}
    for item_segment_index, segment in enumerate(canonical_ordered_segments(item)):
        for node_id in _segment_node_ids(segment):
            order.setdefault(node_id, item_segment_index)
    return order


def _segment_node_ids(segment) -> list[str]:
    local_graph_node = getattr(segment, "local_graph_node", None)
    if local_graph_node is not None:
        return [str(local_graph_node.layer_id)]
    local_graph_nodes = list(getattr(segment, "local_graph_nodes", []) or [])
    if local_graph_nodes:
        return [str(node.layer_id) for node in local_graph_nodes]
    selected_entry = getattr(segment, "selected_entry", None)
    if selected_entry is not None:
        return [str(selected_entry.layer_id)]
    return []


def _unique_strings(values) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        string_value = str(value)
        if string_value in seen:
            continue
        seen.add(string_value)
        result.append(string_value)
    return result


def _hashable_value(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(
            (str(key), _hashable_value(nested_value))
            for key, nested_value in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_value(nested_value) for nested_value in value)
    if isinstance(value, set):
        return tuple(sorted((_hashable_value(nested_value) for nested_value in value), key=repr))
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


__all__ = ["RoundLocalState"]
