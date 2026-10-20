"""Boundary state bookkeeping for Phase 2 execution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from MNSIM.Scheduling.phase2.ddr import (
    DdrBackend,
    DdrBackendError,
    NullDdrBackend,
)
from MNSIM.Scheduling.phase2.types import (
    BOUNDARY_STATUS_COMPLETE_OUTPUT,
    BOUNDARY_STATUS_HOOK_COMPLETE,
    BOUNDARY_STATUS_MODEL_INPUT,
    BOUNDARY_STATUS_OUTPUT_SUBSET,
    BOUNDARY_STATUS_PARTIAL_SUM,
    CONTRIBUTION_COMPLETE_OUTPUT,
    CONTRIBUTION_HOOK_OUTPUT,
    CONTRIBUTION_PARTIAL,
    CONTRIBUTION_STRUCTURAL_OUTPUT,
    DUPLICATE_FRAGMENT_CONTRIBUTION,
    REPORT_ROLE_CONTEXT_ONLY,
    SOURCE_KIND_HOOK,
    SOURCE_KIND_ROUND_FINALIZE,
    SOURCE_KIND_STATIC_EVAL,
    STATIC_STATUS_OK,
    BoundaryAccessEvent,
    BoundaryBinding,
    BoundaryCommitRequest,
    BoundaryCommitResult,
    BoundaryInputView,
    BoundaryStateRecord,
    Phase2Invalid,
)


_DOWNSTREAM_VISIBLE_STATUSES = {
    BOUNDARY_STATUS_MODEL_INPUT,
    BOUNDARY_STATUS_COMPLETE_OUTPUT,
    BOUNDARY_STATUS_HOOK_COMPLETE,
}
_PARTIAL_STATUSES = {BOUNDARY_STATUS_PARTIAL_SUM, BOUNDARY_STATUS_OUTPUT_SUBSET}
_HOOK_READABLE_STATUSES = _DOWNSTREAM_VISIBLE_STATUSES


class BoundaryStateManager:
    def __init__(self, ddr_backend: DdrBackend | None = None) -> None:
        self._states: dict[str, BoundaryStateRecord] = {}
        self._next_state_index = 0
        self._partial_accumulators: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._partial_metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._complete_state_ids: dict[tuple[Any, ...], str] = {}
        self._ddr_backend = ddr_backend or NullDdrBackend()

    def seed_model_input(
        self,
        state_id,
        shape,
        tensor_role="activation",
    ) -> BoundaryStateRecord:
        state = BoundaryStateRecord(
            state_id=str(state_id),
            producer_item_id="model_input",
            producer_layer_id="model_input",
            tensor_role=str(tensor_role),
            shape=deepcopy(dict(shape)),
            region={"kind": "full"},
            status=BOUNDARY_STATUS_MODEL_INPUT,
            storage_backend="model_input",
            allowed_consumer_item_ids=[],
            lifetime_policy="inference_input",
            source_state_ids=[],
        )
        self._states[state.state_id] = state
        return state

    def build_input_view(self, item) -> BoundaryInputView:
        item_id = _item_id(item)
        state_ids = [_boundary_ref_state_id(ref) for ref in getattr(item, "external_inputs", [])]
        return self.build_input_view_for_state_ids(item_id, state_ids)

    def build_input_view_for_state_ids(
        self,
        item_id,
        state_ids,
        allow_partial=False,
        partial_origin_layer_id=None,
    ) -> BoundaryInputView:
        item_id = str(item_id)
        readable_state_ids: list[str] = []
        state_bindings: dict[str, list[str]] = {}
        state_records: list[BoundaryStateRecord] = []
        warnings: list[str] = []
        events: list[BoundaryAccessEvent] = []

        for state_id in state_ids:
            state = self._require_state(str(state_id))
            if state.status == BOUNDARY_STATUS_OUTPUT_SUBSET:
                raise Phase2Invalid(
                    "OUTPUT_SUBSET_BOUNDARY_NOT_READABLE",
                    f"state {state.state_id} has partial status {state.status}",
                )
            if state.status == BOUNDARY_STATUS_PARTIAL_SUM:
                if not allow_partial:
                    raise Phase2Invalid(
                        "PARTIAL_BOUNDARY_NOT_READABLE",
                        f"state {state.state_id} has partial status {state.status}",
                    )
                if (
                    partial_origin_layer_id is None
                    or str(partial_origin_layer_id) != state.producer_layer_id
                ):
                    raise Phase2Invalid(
                        "PARTIAL_BOUNDARY_ORIGIN_MISMATCH",
                        f"state {state.state_id} has partial_sum from layer "
                        f"{state.producer_layer_id}; requested "
                        f"partial_origin_layer_id {partial_origin_layer_id!r}",
                    )
                self._validate_allowed_consumer(item_id=item_id, state=state)
            if state.status not in _DOWNSTREAM_VISIBLE_STATUSES and state.status not in _PARTIAL_STATUSES:
                raise Phase2Invalid(
                    "BOUNDARY_STATE_NOT_READABLE",
                    f"state {state.state_id} status {state.status} is not readable",
                )
            if state.status in {
                BOUNDARY_STATUS_COMPLETE_OUTPUT,
                BOUNDARY_STATUS_HOOK_COMPLETE,
            }:
                self._validate_allowed_consumer(item_id=item_id, state=state)
            readable_state_ids.append(state.state_id)
            state_bindings.setdefault(state.producer_layer_id, []).append(state.state_id)
            state_records.append(deepcopy(state))
            if state.storage_backend in {"boundary_sram", "ddr"}:
                events.append(
                    BoundaryAccessEvent(
                        event_kind="read_boundary_state",
                        item_id=item_id,
                        state_id=state.state_id,
                        detail=_state_event_detail(state),
                    )
                )

        return BoundaryInputView(
            input_view_id=f"view:{item_id}",
            item_id=item_id,
            readable_state_ids=readable_state_ids,
            state_bindings=state_bindings,
            warnings=warnings,
            state_records=state_records,
            events=events,
        )

    def commit(self, request: BoundaryCommitRequest) -> BoundaryCommitResult:
        self._validate_request_source(request)
        bindings = self._request_bindings(request)
        created_states: list[BoundaryStateRecord] = []
        updated_states: list[BoundaryStateRecord] = []
        events: list[BoundaryAccessEvent] = [
            deepcopy(event) for event in request.input_events
        ]
        completion_updates: list[dict[str, Any]] = []
        visibility_updates: list[dict[str, Any]] = []
        output_state_bindings: list[dict[str, Any]] = []
        warnings = self._request_warnings(request)

        for binding in bindings:
            self._validate_source_state_ids(binding)
            if (
                request.source_kind in {
                    SOURCE_KIND_STATIC_EVAL,
                    SOURCE_KIND_ROUND_FINALIZE,
                }
                and binding.report_role != REPORT_ROLE_CONTEXT_ONLY
            ):
                self._validate_static_binding_sources(item_id=request.item_id, binding=binding)
            if binding.report_role == REPORT_ROLE_CONTEXT_ONLY:
                events.append(
                    BoundaryAccessEvent(
                        event_kind="ignore_context",
                        item_id=request.item_id,
                        state_id="",
                        detail={
                            "origin_layer_id": binding.origin_layer_id,
                            "contribution_kind": binding.contribution_kind,
                        },
                    )
                )
                continue

            if binding.contribution_kind == CONTRIBUTION_PARTIAL:
                self._commit_partial_contribution(
                    request=request,
                    binding=binding,
                    created_states=created_states,
                    updated_states=updated_states,
                    completion_updates=completion_updates,
                    visibility_updates=visibility_updates,
                    output_state_bindings=output_state_bindings,
                )
                continue

            if binding.contribution_kind in {
                CONTRIBUTION_COMPLETE_OUTPUT,
                CONTRIBUTION_STRUCTURAL_OUTPUT,
            }:
                state = self._create_state(
                    request=request,
                    binding=binding,
                    status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
                    input_range=binding.full_input_range,
                    output_range=binding.full_output_range,
                    full_output=True,
                )
                created_states.append(state)
                completion_update = self._completion_update(
                    binding=binding,
                    state=state,
                    status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
                    input_range_covered=binding.full_input_range,
                    output_range_covered=binding.full_output_range,
                )
                completion_updates.append(completion_update)
                self._append_output_visibility(
                    binding=binding,
                    state=state,
                    status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
                    visibility_updates=visibility_updates,
                    output_state_bindings=output_state_bindings,
                )
                continue

            if binding.contribution_kind == CONTRIBUTION_HOOK_OUTPUT:
                self._validate_hook_inputs(item_id=request.item_id, binding=binding)
                state = self._create_state(
                    request=request,
                    binding=binding,
                    status=BOUNDARY_STATUS_HOOK_COMPLETE,
                    input_range=binding.full_input_range,
                    output_range=binding.full_output_range,
                    full_output=True,
                )
                created_states.append(state)
                completion_updates.append(
                    self._completion_update(
                        binding=binding,
                        state=state,
                        status=BOUNDARY_STATUS_HOOK_COMPLETE,
                        input_range_covered=binding.full_input_range,
                        output_range_covered=binding.full_output_range,
                    )
                )
                self._append_output_visibility(
                    binding=binding,
                    state=state,
                    status=BOUNDARY_STATUS_HOOK_COMPLETE,
                    visibility_updates=visibility_updates,
                    output_state_bindings=output_state_bindings,
                )
                continue

            raise Phase2Invalid(
                "UNSUPPORTED_BOUNDARY_CONTRIBUTION",
                f"unsupported contribution_kind {binding.contribution_kind!r}",
            )

        events.extend(
            _boundary_state_event("write_boundary_state", request.item_id, state)
            for state in created_states
        )
        events.extend(
            _boundary_state_event("update_boundary_state", request.item_id, state)
            for state in updated_states
        )
        try:
            ddr_transfer = self._ddr_backend.score_events(events)
        except DdrBackendError as exc:
            raise Phase2Invalid("DDR_BACKEND_FAILURE", str(exc)) from exc

        return BoundaryCommitResult(
            item_id=request.item_id,
            item_kind=request.item_kind,
            status="ok",
            created_states=_snapshot_states(created_states),
            updated_states=_snapshot_states(updated_states),
            released_states=[],
            events=events,
            completion_updates=completion_updates,
            visibility_updates=visibility_updates,
            output_state_bindings=output_state_bindings,
            warnings=warnings,
            ddr_transfer=ddr_transfer,
        )

    def _create_state_for_test(
        self,
        producer_item_id,
        producer_layer_id,
        status,
        allowed_consumer_item_ids=None,
        state_id=None,
        shape=None,
        tensor_role="activation",
    ) -> BoundaryStateRecord:
        state = BoundaryStateRecord(
            state_id=str(state_id) if state_id is not None else self._new_state_id(producer_layer_id),
            producer_item_id=str(producer_item_id),
            producer_layer_id=str(producer_layer_id),
            tensor_role=str(tensor_role),
            shape=deepcopy(dict(shape or {"kind": "test"})),
            region={"kind": "test"},
            status=str(status),
            storage_backend=(
                "ddr"
                if bool(getattr(self._ddr_backend, "enabled", False))
                else "boundary_sram"
            ),
            allowed_consumer_item_ids=list(allowed_consumer_item_ids or []),
            lifetime_policy="test",
            source_state_ids=[],
        )
        self._states[state.state_id] = state
        return state

    def _commit_partial_contribution(
        self,
        *,
        request: BoundaryCommitRequest,
        binding: BoundaryBinding,
        created_states: list[BoundaryStateRecord],
        updated_states: list[BoundaryStateRecord],
        completion_updates: list[dict[str, Any]],
        visibility_updates: list[dict[str, Any]],
        output_state_bindings: list[dict[str, Any]],
    ) -> None:
        _validate_partial_binding_ranges(binding)
        selected_fragment_ids, required_fragment_ids = _partial_fragment_id_sets(binding)
        origin_key = self._origin_key(binding)
        self._validate_partial_metadata(
            origin_key=origin_key,
            binding=binding,
            required_fragment_ids=required_fragment_ids,
        )
        accumulator = self._partial_accumulators.get(origin_key)
        if accumulator is None:
            accumulator = {
                "state_id": None,
                "required_fragment_ids": set(required_fragment_ids),
                "covered_fragment_ids": set(),
                "source_state_ids": [],
            }
            self._partial_accumulators[origin_key] = accumulator
        elif accumulator["required_fragment_ids"] != required_fragment_ids:
            raise Phase2Invalid(
                "PARTIAL_METADATA_MISMATCH",
                f"partial contribution required fragments differ for {binding.origin_layer_id}",
            )

        duplicate_fragment_ids = accumulator["covered_fragment_ids"].intersection(
            selected_fragment_ids
        )
        if duplicate_fragment_ids:
            raise Phase2Invalid(
                DUPLICATE_FRAGMENT_CONTRIBUTION,
                (
                    f"{binding.origin_layer_id} repeats fragment ids "
                    f"{sorted(duplicate_fragment_ids)}"
                ),
            )
        accumulator["covered_fragment_ids"].update(selected_fragment_ids)
        accumulator["source_state_ids"] = _unique(
            [*accumulator["source_state_ids"], *binding.source_state_ids]
        )
        output_complete = (
            accumulator["covered_fragment_ids"]
            == accumulator["required_fragment_ids"]
        )

        if output_complete:
            complete_state, created_complete_state = self._complete_state(
                request,
                binding,
                origin_key,
                source_state_ids=accumulator["source_state_ids"],
                existing_state_id=accumulator.get("state_id"),
            )
            accumulator["state_id"] = complete_state.state_id
            if created_complete_state:
                created_states.append(complete_state)
            else:
                updated_states.append(complete_state)
            state_for_updates = complete_state
            completion_status = BOUNDARY_STATUS_COMPLETE_OUTPUT
            input_range_covered = binding.full_input_range
            output_range_covered = binding.full_output_range
        else:
            state_id = accumulator.get("state_id")
            if state_id is None:
                state = self._create_state(
                    request=request,
                    binding=binding,
                    status=BOUNDARY_STATUS_PARTIAL_SUM,
                    input_range=binding.input_range,
                    output_range=binding.output_range,
                    full_output=False,
                )
                accumulator["state_id"] = state.state_id
                created_accumulator = True
            else:
                state = self._states[state_id]
                created_accumulator = False
            completion_status = (
                BOUNDARY_STATUS_PARTIAL_SUM
            )
            state.status = completion_status
            state.source_state_ids = _unique([*state.source_state_ids, *binding.source_state_ids])
            if created_accumulator:
                created_states.append(state)
            else:
                updated_states.append(state)
            state_for_updates = state
            input_range_covered = binding.input_range
            output_range_covered = binding.output_range

        completion_updates.append(
            self._completion_update(
                binding=binding,
                state=state_for_updates,
                status=completion_status,
                input_range_covered=input_range_covered,
                output_range_covered=output_range_covered,
                selected_fragment_ids=selected_fragment_ids,
                covered_fragment_ids=accumulator["covered_fragment_ids"],
                required_fragment_ids=accumulator["required_fragment_ids"],
            )
        )
        self._append_output_visibility(
            binding=binding,
            state=state_for_updates,
            status=completion_status,
            visibility_updates=visibility_updates,
            output_state_bindings=output_state_bindings,
        )

    def _complete_state(
        self,
        request: BoundaryCommitRequest,
        binding: BoundaryBinding,
        origin_key: tuple[Any, ...],
        source_state_ids: list[str] | None = None,
        existing_state_id: str | None = None,
    ) -> tuple[BoundaryStateRecord, bool]:
        source_state_ids = list(source_state_ids or binding.source_state_ids)
        if existing_state_id is not None:
            state = self._states[existing_state_id]
            logical_state_id = self._logical_state_id(binding)
            if state.state_id != logical_state_id and logical_state_id not in self._states:
                del self._states[state.state_id]
                state.state_id = logical_state_id
                self._states[state.state_id] = state
            state.status = BOUNDARY_STATUS_COMPLETE_OUTPUT
            state.shape = _state_shape(binding, full_output=True)
            state.region = _state_region(
                binding,
                input_range=binding.full_input_range,
                output_range=binding.full_output_range,
            )
            state.allowed_consumer_item_ids = list(binding.downstream_consumer_ids)
            state.lifetime_policy = "release_after_consumers"
            state.source_state_ids = _unique([*state.source_state_ids, *source_state_ids])
            _copy_analog_payload_fields(state, binding)
            self._complete_state_ids[origin_key] = state.state_id
            return state, False

        existing_state_id = self._complete_state_ids.get(origin_key)
        if existing_state_id is not None:
            state = self._states[existing_state_id]
            state.source_state_ids = _unique([*state.source_state_ids, *source_state_ids])
            _copy_analog_payload_fields(state, binding)
            return state, False
        state = self._create_state(
            request=request,
            binding=binding,
            status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
            input_range=binding.full_input_range,
            output_range=binding.full_output_range,
            full_output=True,
        )
        state.source_state_ids = _unique(source_state_ids)
        self._complete_state_ids[origin_key] = state.state_id
        return state, True

    def _create_state(
        self,
        *,
        request: BoundaryCommitRequest,
        binding: BoundaryBinding,
        status: str,
        input_range: Any,
        output_range: Any,
        full_output: bool,
    ) -> BoundaryStateRecord:
        state = BoundaryStateRecord(
            state_id=self._new_state_id(
                binding.origin_layer_id,
                prefer_logical=full_output,
                sample_id=binding.sample_id,
                timestep=binding.timestep,
                sequence_length=binding.sequence_length,
            ),
            producer_item_id=request.item_id,
            producer_layer_id=binding.origin_layer_id,
            tensor_role=binding.tensor_role,
            shape=_state_shape(binding, full_output=full_output),
            region=_state_region(binding, input_range=input_range, output_range=output_range),
            status=status,
            storage_backend="boundary_sram",
            allowed_consumer_item_ids=list(
                binding.downstream_consumer_ids
                if full_output or status == BOUNDARY_STATUS_PARTIAL_SUM
                else []
            ),
            lifetime_policy="release_after_consumers" if full_output else "internal_until_complete",
            source_state_ids=list(binding.source_state_ids),
            sample_id=str(binding.sample_id),
            timestep=int(binding.timestep),
            sequence_length=int(binding.sequence_length),
            data_bitwidth=binding.data_bitwidth,
            accumulator_domain=deepcopy(dict(binding.accumulator_domain)),
            payload_kind=str(binding.payload_kind),
            numerical_input_payload_identity=binding.numerical_input_payload_identity,
            state_payload_identity=binding.state_payload_identity,
            spike_payload_identity=binding.spike_payload_identity,
            spike_codes=deepcopy(binding.spike_codes),
        )
        self._states[state.state_id] = state
        return state

    def _logical_state_id(self, binding: BoundaryBinding) -> str:
        return self._state_id_prefix(
            binding.origin_layer_id,
            sample_id=binding.sample_id,
            timestep=binding.timestep,
            sequence_length=binding.sequence_length,
        )

    def _state_id_prefix(
        self,
        producer_layer_id: Any,
        *,
        sample_id: Any = "sample:0",
        timestep: int = 0,
        sequence_length: int = 1,
    ) -> str:
        logical_state_id = f"state:{_safe_id(producer_layer_id)}"
        if int(sequence_length) > 1:
            logical_state_id = (
                f"{logical_state_id}:sample:{_safe_id(sample_id)}:t{int(timestep)}"
            )
        return logical_state_id

    def _new_state_id(
        self,
        producer_layer_id: Any,
        prefer_logical: bool = False,
        *,
        sample_id: Any = "sample:0",
        timestep: int = 0,
        sequence_length: int = 1,
    ) -> str:
        logical_state_id = self._state_id_prefix(
            producer_layer_id,
            sample_id=sample_id,
            timestep=timestep,
            sequence_length=sequence_length,
        )
        if prefer_logical and logical_state_id not in self._states:
            return logical_state_id
        self._next_state_index += 1
        return f"{logical_state_id}:{self._next_state_index}"

    def _validate_request_source(self, request: BoundaryCommitRequest) -> None:
        if request.source_kind == SOURCE_KIND_STATIC_EVAL:
            if request.static_eval_result is None:
                raise Phase2Invalid(
                    "INVALID_BOUNDARY_COMMIT",
                    f"static_eval commit {request.item_id} is missing static_eval_result",
                )
            if request.static_eval_result.status != STATIC_STATUS_OK:
                raise Phase2Invalid(
                    "STATIC_EVAL_NOT_OK",
                    (
                        f"static eval result for {request.item_id} has status "
                        f"{request.static_eval_result.status!r}"
                    ),
                )
            return
        if request.source_kind == SOURCE_KIND_HOOK:
            if request.hook_result is None:
                raise Phase2Invalid(
                    "INVALID_BOUNDARY_COMMIT",
                    f"hook commit {request.item_id} is missing hook_result",
                )
            return
        if request.source_kind == SOURCE_KIND_ROUND_FINALIZE:
            if not request.step_results:
                raise Phase2Invalid(
                    "INVALID_BOUNDARY_COMMIT",
                    f"round finalize {request.item_id} is missing step_results",
                )
            failed_step_ids = [
                step_result.step_id
                for step_result in request.step_results
                if step_result.status != STATIC_STATUS_OK
            ]
            if failed_step_ids:
                raise Phase2Invalid(
                    "STATIC_EVAL_NOT_OK",
                    (
                        f"round finalize {request.item_id} has non-ok steps "
                        f"{failed_step_ids!r}"
                    ),
                )
            return
        raise Phase2Invalid(
            "INVALID_BOUNDARY_SOURCE_KIND",
            f"unsupported source_kind {request.source_kind!r}",
        )

    def _request_bindings(self, request: BoundaryCommitRequest) -> list[BoundaryBinding]:
        if request.boundary_bindings:
            return list(request.boundary_bindings)
        if request.source_kind == SOURCE_KIND_STATIC_EVAL and request.static_eval_result is not None:
            return list(request.static_eval_result.boundary_bindings)
        if request.source_kind == SOURCE_KIND_HOOK and request.hook_result is not None:
            return list(request.hook_result.hook_bindings)
        return []

    def _request_warnings(self, request: BoundaryCommitRequest) -> list[str]:
        if request.source_kind == SOURCE_KIND_STATIC_EVAL and request.static_eval_result is not None:
            return list(request.static_eval_result.warnings)
        if request.source_kind == SOURCE_KIND_HOOK and request.hook_result is not None:
            return list(request.hook_result.warnings)
        if request.source_kind == SOURCE_KIND_ROUND_FINALIZE:
            return _unique(
                warning
                for step_result in request.step_results
                for warning in step_result.warnings
            )
        return []

    def _validate_source_state_ids(self, binding: BoundaryBinding) -> None:
        for state_id in binding.source_state_ids:
            if str(state_id) not in self._states:
                raise Phase2Invalid(
                    "MISSING_BOUNDARY_SOURCE",
                    f"missing source state {state_id!r} for {binding.origin_layer_id}",
                )

    def _validate_static_binding_sources(
        self,
        *,
        item_id: str,
        binding: BoundaryBinding,
    ) -> None:
        for state_id in binding.source_state_ids:
            state = self._require_state(str(state_id))
            if state.status == BOUNDARY_STATUS_MODEL_INPUT:
                continue
            if state.status in {
                BOUNDARY_STATUS_COMPLETE_OUTPUT,
                BOUNDARY_STATUS_HOOK_COMPLETE,
            }:
                self._validate_allowed_consumer(item_id=item_id, state=state)
                continue
            if state.status == BOUNDARY_STATUS_PARTIAL_SUM:
                if binding.contribution_kind != CONTRIBUTION_PARTIAL:
                    raise Phase2Invalid(
                        "BOUNDARY_STATE_NOT_READABLE",
                        (
                            f"state {state.state_id} has partial_sum status and cannot "
                            f"feed {binding.contribution_kind!r}"
                        ),
                    )
                if state.producer_layer_id != binding.origin_layer_id:
                    raise Phase2Invalid(
                        "PARTIAL_BOUNDARY_ORIGIN_MISMATCH",
                        (
                            f"state {state.state_id} has partial_sum from layer "
                            f"{state.producer_layer_id}; requested "
                            f"origin_layer_id {binding.origin_layer_id!r}"
                        ),
                    )
                self._validate_allowed_consumer(item_id=item_id, state=state)
                continue
            raise Phase2Invalid(
                "BOUNDARY_STATE_NOT_READABLE",
                f"state {state.state_id} status {state.status} is not readable",
            )

    def _validate_hook_inputs(self, *, item_id: str, binding: BoundaryBinding) -> None:
        for state_id in binding.source_state_ids:
            state = self._require_state(str(state_id))
            if state.status not in _HOOK_READABLE_STATUSES:
                raise Phase2Invalid(
                    "INVALID_HOOK_INPUT",
                    f"hook input {state.state_id} has non-complete status {state.status}",
                )
            self._validate_allowed_consumer(item_id=item_id, state=state)

    def _require_state(self, state_id: str) -> BoundaryStateRecord:
        if state_id not in self._states:
            raise Phase2Invalid("MISSING_BOUNDARY_SOURCE", f"missing source state {state_id!r}")
        return self._states[state_id]

    def _validate_allowed_consumer(
        self,
        *,
        item_id: str,
        state: BoundaryStateRecord,
    ) -> None:
        if state.status == BOUNDARY_STATUS_MODEL_INPUT:
            return
        if item_id in state.allowed_consumer_item_ids:
            return
        raise Phase2Invalid(
            "BOUNDARY_CONSUMER_NOT_ALLOWED",
            f"item {item_id!r} is not allowed to read state {state.state_id}",
        )

    def _origin_key(self, binding: BoundaryBinding) -> tuple[Any, ...]:
        return (
            str(binding.sample_id),
            int(binding.timestep),
            binding.origin_layer_id,
            binding.tensor_role,
            _range_key(binding.full_input_range),
            _range_key(binding.full_output_range),
        )

    def _validate_partial_metadata(
        self,
        *,
        origin_key: tuple[Any, ...],
        binding: BoundaryBinding,
        required_fragment_ids: set[str],
    ) -> None:
        self._validate_metadata_fields(
            metadata_key=("partial_origin", *origin_key),
            metadata={
                "shape": _jsonable(_logical_partial_shape(binding)),
                "tensor_role": binding.tensor_role,
                "downstream_consumer_ids": list(binding.downstream_consumer_ids),
                "required_fragment_ids": sorted(required_fragment_ids),
                "region": _jsonable(_partial_region_metadata(binding.region)),
                "sequence_length": int(binding.sequence_length),
                "data_bitwidth": binding.data_bitwidth,
                "accumulator_domain": _jsonable(binding.accumulator_domain),
                "payload_kind": str(binding.payload_kind),
            },
            binding=binding,
        )

    def _validate_metadata_fields(
        self,
        *,
        metadata_key: tuple[Any, ...],
        metadata: dict[str, Any],
        binding: BoundaryBinding,
    ) -> None:
        canonical = self._partial_metadata.get(metadata_key)
        if canonical is None:
            self._partial_metadata[metadata_key] = deepcopy(metadata)
            return
        for field_name, value in metadata.items():
            if canonical[field_name] != value:
                raise Phase2Invalid(
                    "PARTIAL_METADATA_MISMATCH",
                    (
                        f"partial contribution {field_name} differs for "
                        f"{binding.origin_layer_id}: expected "
                        f"{canonical[field_name]!r}, got {value!r}"
                    ),
                )

    def _completion_update(
        self,
        *,
        binding: BoundaryBinding,
        state: BoundaryStateRecord,
        status: str,
        input_range_covered: Any,
        output_range_covered: Any,
        selected_fragment_ids: set[str] | None = None,
        covered_fragment_ids: set[str] | None = None,
        required_fragment_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        update = {
            "origin_layer_id": binding.origin_layer_id,
            "origin_layer_type": binding.origin_layer_type,
            "sample_id": str(binding.sample_id),
            "timestep": int(binding.timestep),
            "sequence_length": int(binding.sequence_length),
            "data_bitwidth": binding.data_bitwidth,
            "accumulator_domain": deepcopy(dict(binding.accumulator_domain)),
            "payload_kind": str(binding.payload_kind),
            "numerical_input_payload_identity": binding.numerical_input_payload_identity,
            "state_payload_identity": binding.state_payload_identity,
            "spike_payload_identity": binding.spike_payload_identity,
            "spike_codes": deepcopy(binding.spike_codes),
            "completion_status": status,
            "state_id": state.state_id,
            "input_range_covered": deepcopy(input_range_covered),
            "output_range_covered": deepcopy(output_range_covered),
            "contribution_output_range": deepcopy(binding.output_range),
            "downstream_visible": status in _DOWNSTREAM_VISIBLE_STATUSES,
        }
        if selected_fragment_ids is not None:
            update["selected_fragment_ids"] = sorted(selected_fragment_ids)
        if covered_fragment_ids is not None:
            update["covered_fragment_count"] = len(covered_fragment_ids)
        if required_fragment_ids is not None:
            update["required_fragment_count"] = len(required_fragment_ids)
        return update

    def _append_output_visibility(
        self,
        *,
        binding: BoundaryBinding,
        state: BoundaryStateRecord,
        status: str,
        visibility_updates: list[dict[str, Any]],
        output_state_bindings: list[dict[str, Any]],
    ) -> None:
        downstream_visible = status in _DOWNSTREAM_VISIBLE_STATUSES
        output_binding = {
            "state_id": state.state_id,
            "origin_layer_id": binding.origin_layer_id,
            "origin_layer_type": binding.origin_layer_type,
            "tensor_role": binding.tensor_role,
            "completion_status": status,
            "downstream_visible": downstream_visible,
            "downstream_consumer_ids": list(binding.downstream_consumer_ids),
            "sample_id": state.sample_id,
            "timestep": state.timestep,
            "sequence_length": state.sequence_length,
            "data_bitwidth": state.data_bitwidth,
            "accumulator_domain": deepcopy(state.accumulator_domain),
            "payload_kind": state.payload_kind,
            "numerical_input_payload_identity": state.numerical_input_payload_identity,
            "state_payload_identity": state.state_payload_identity,
            "spike_payload_identity": state.spike_payload_identity,
            "spike_codes": deepcopy(state.spike_codes),
        }
        output_state_bindings.append(output_binding)
        if downstream_visible:
            visibility_updates.append(
                {
                    "state_id": state.state_id,
                    "origin_layer_id": binding.origin_layer_id,
                    "visible_to": list(binding.downstream_consumer_ids),
                    "downstream_visible": True,
                    "sample_id": state.sample_id,
                    "timestep": state.timestep,
                }
            )


def _state_shape(binding: BoundaryBinding, *, full_output: bool) -> dict[str, Any]:
    shape = deepcopy(dict(binding.shape))
    if not full_output:
        return shape
    output_interval = _as_interval(binding.full_output_range)
    if output_interval is None:
        return shape
    output_count = output_interval[1] - output_interval[0]
    if "Outputchannel" in shape:
        shape["Outputchannel"] = output_count
    elif "Outfeature" in shape:
        shape["Outfeature"] = output_count
    elif "C" in shape:
        shape["C"] = output_count
    return shape


def _logical_partial_shape(binding: BoundaryBinding) -> dict[str, Any]:
    """Normalize a fragment-local output shape to its immutable full-layer shape."""
    shape = deepcopy(dict(binding.shape))
    output_interval = _as_interval(binding.full_output_range)
    if output_interval is None:
        return shape
    output_count = output_interval[1] - output_interval[0]
    if "Outputchannel" in shape:
        shape["Outputchannel"] = output_count
    elif "Outfeature" in shape:
        shape["Outfeature"] = output_count
    elif "C" in shape:
        shape["C"] = output_count
    return shape


def _partial_region_metadata(region: Any) -> Any:
    if isinstance(region, dict):
        return {
            str(key): _partial_region_metadata(value)
            for key, value in region.items()
            if str(key) not in {"input_range", "output_range"}
        }
    if isinstance(region, (list, tuple)):
        return [_partial_region_metadata(value) for value in region]
    return region


def _state_region(binding: BoundaryBinding, *, input_range: Any, output_range: Any) -> dict[str, Any]:
    region = deepcopy(dict(binding.region))
    region["input_range"] = deepcopy(input_range)
    region["output_range"] = deepcopy(output_range)
    region["full_input_range"] = deepcopy(binding.full_input_range)
    region["full_output_range"] = deepcopy(binding.full_output_range)
    return region


def _snapshot_states(states: Iterable[BoundaryStateRecord]) -> list[BoundaryStateRecord]:
    return [deepcopy(state) for state in states]


def _copy_analog_payload_fields(
    state: BoundaryStateRecord, binding: BoundaryBinding
) -> None:
    state.numerical_input_payload_identity = binding.numerical_input_payload_identity
    state.state_payload_identity = binding.state_payload_identity
    state.spike_payload_identity = binding.spike_payload_identity
    state.spike_codes = deepcopy(binding.spike_codes)


def _boundary_state_event(
    event_kind: str,
    item_id: str,
    state: BoundaryStateRecord,
) -> BoundaryAccessEvent:
    return BoundaryAccessEvent(
        event_kind=event_kind,
        item_id=str(item_id),
        state_id=state.state_id,
        detail=_state_event_detail(state),
    )


def _state_event_detail(state: BoundaryStateRecord) -> dict[str, Any]:
    detail = {
        "producer_item_id": state.producer_item_id,
        "producer_layer_id": state.producer_layer_id,
        "status": state.status,
        "sample_id": state.sample_id,
        "timestep": state.timestep,
        "sequence_length": state.sequence_length,
        "shape": deepcopy(state.shape),
        "storage_backend": state.storage_backend,
        "data_bitwidth": state.data_bitwidth,
        "accumulator_domain": deepcopy(state.accumulator_domain),
        "payload_kind": state.payload_kind,
        "numerical_input_payload_identity": state.numerical_input_payload_identity,
        "state_payload_identity": state.state_payload_identity,
        "spike_payload_identity": state.spike_payload_identity,
        "spike_codes": deepcopy(state.spike_codes),
    }
    detail["payload_bytes"] = _state_payload_bytes(state)
    return detail


def _state_payload_bytes(state: BoundaryStateRecord) -> int:
    shape = dict(state.shape or {})
    channels = None
    for key in ("Outputchannel", "Outfeature", "C", "channels", "features"):
        if key in shape:
            try:
                channels = int(shape[key])
            except (TypeError, ValueError):
                channels = None
            break
    output_size = shape.get("Outputsize", shape.get("output_size"))
    spatial = 1
    if isinstance(output_size, (list, tuple)) and output_size:
        try:
            for value in output_size:
                spatial *= int(value)
        except (TypeError, ValueError):
            spatial = 1
    count = max(0, int(channels or 0)) * spatial
    if str(state.payload_kind) in {"spike_sequence", "activation_sequence"}:
        count *= max(1, int(state.sequence_length))
    bitwidth = max(1, int(state.data_bitwidth or 8))
    return (count * bitwidth + 7) // 8 if count else 0


def _item_id(item: Any) -> str:
    try:
        return str(item.item_id)
    except AttributeError as exc:
        raise Phase2Invalid("INVALID_BOUNDARY_INPUT_ITEM", "item is missing item_id") from exc


def _boundary_ref_state_id(ref: Any) -> str:
    if hasattr(ref, "state_id"):
        return str(ref.state_id)
    if isinstance(ref, dict) and "state_id" in ref:
        return str(ref["state_id"])
    return str(ref)


def _range_key(value: Any) -> tuple[str, str]:
    return ("range", json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_interval(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            start = int(value[0])
            end = int(value[1])
        except (TypeError, ValueError):
            return None
        if end < start:
            return None
        return start, end
    return None


def _as_strict_int_interval(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    start, end = value
    if type(start) is not int or type(end) is not int:
        return None
    if end < start:
        return None
    return start, end


def _partial_fragment_id_sets(binding: BoundaryBinding) -> tuple[set[str], set[str]]:
    selected_ids = list(binding.fragment_ids or [])
    required_ids = list(binding.required_fragment_ids or [])
    if not selected_ids or not required_ids:
        raise Phase2Invalid(
            "INVALID_FRAGMENT_CONTRIBUTION",
            f"{binding.origin_layer_id} requires selected and required fragment ids",
        )
    if not all(isinstance(fragment_id, str) and fragment_id for fragment_id in selected_ids):
        raise Phase2Invalid(
            "INVALID_FRAGMENT_CONTRIBUTION",
            f"{binding.origin_layer_id} has invalid selected fragment ids",
        )
    if not all(isinstance(fragment_id, str) and fragment_id for fragment_id in required_ids):
        raise Phase2Invalid(
            "INVALID_FRAGMENT_CONTRIBUTION",
            f"{binding.origin_layer_id} has invalid required fragment ids",
        )
    selected_set = set(selected_ids)
    required_set = set(required_ids)
    if len(selected_set) != len(selected_ids):
        raise Phase2Invalid(
            DUPLICATE_FRAGMENT_CONTRIBUTION,
            f"{binding.origin_layer_id} repeats selected fragment ids in one contribution",
        )
    if len(required_set) != len(required_ids):
        raise Phase2Invalid(
            "INVALID_FRAGMENT_CONTRIBUTION",
            f"{binding.origin_layer_id} repeats required fragment ids",
        )
    if not selected_set.issubset(required_set):
        raise Phase2Invalid(
            "INVALID_FRAGMENT_CONTRIBUTION",
            f"{binding.origin_layer_id} selected fragment ids are outside the inventory",
        )
    return selected_set, required_set


def _validate_partial_binding_ranges(binding: BoundaryBinding) -> None:
    input_interval = _require_partial_interval(binding, "input_range")
    full_input_interval = _require_partial_interval(binding, "full_input_range")
    output_interval = _require_partial_interval(binding, "output_range")
    full_output_interval = _require_partial_interval(binding, "full_output_range")

    if not _interval_contains(full_input_interval, input_interval):
        raise Phase2Invalid(
            "INVALID_PARTIAL_RANGE",
            (
                f"input_range {binding.input_range!r} is outside full_input_range "
                f"{binding.full_input_range!r} for {binding.origin_layer_id}"
            ),
        )
    if not _interval_contains(full_output_interval, output_interval):
        raise Phase2Invalid(
            "INVALID_PARTIAL_RANGE",
            (
                f"output_range {binding.output_range!r} is outside full_output_range "
                f"{binding.full_output_range!r} for {binding.origin_layer_id}"
            ),
        )


def _require_partial_interval(
    binding: BoundaryBinding,
    field_name: str,
) -> tuple[int, int]:
    value = getattr(binding, field_name)
    interval = _as_strict_int_interval(value)
    if interval is None:
        raise Phase2Invalid(
            "INVALID_PARTIAL_RANGE",
            (
                f"{field_name} must be an integer interval [start, end] with "
                f"start <= end for {binding.origin_layer_id}; got {value!r}"
            ),
        )
    return interval


def _interval_contains(full_interval: tuple[int, int], interval: tuple[int, int]) -> bool:
    return full_interval[0] <= interval[0] and interval[1] <= full_interval[1]


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _safe_id(value: Any) -> str:
    text = str(value)
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


__all__ = ["BoundaryStateManager"]
