import pytest

from MNSIM.Scheduling.phase2.boundary import BoundaryStateManager
from MNSIM.Scheduling.phase2.types import (
    BOUNDARY_STATUS_COMPLETE_OUTPUT,
    BOUNDARY_STATUS_HOOK_COMPLETE,
    BOUNDARY_STATUS_OUTPUT_SUBSET,
    BOUNDARY_STATUS_PARTIAL_SUM,
    BoundaryBinding,
    BoundaryCommitRequest,
    HookResult,
    MetricTotals,
    DUPLICATE_FRAGMENT_CONTRIBUTION,
    Phase2Invalid,
    StepExecutionResult,
    StaticEvalResult,
)
from MNSIM.Scheduling.round_types import BoundaryRef


def _binding(
    origin,
    contribution,
    input_range,
    output_range,
    full_input,
    full_output,
    source_state_ids=None,
    *,
    report_role="reportable",
    local_layer_id=0,
    hook_node_id=None,
    downstream_consumer_ids=None,
    fragment_ids=None,
    required_fragment_ids=None,
):
    kwargs = {}
    if fragment_ids is not None:
        kwargs["fragment_ids"] = list(fragment_ids)
    if required_fragment_ids is not None:
        kwargs["required_fragment_ids"] = list(required_fragment_ids)
    return BoundaryBinding(
        local_layer_id=local_layer_id,
        hook_node_id=hook_node_id,
        origin_layer_id=origin,
        origin_layer_type="conv",
        report_role=report_role,
        tensor_role="activation",
        shape={"C": full_output[1], "H": 1, "W": 1}
        if isinstance(full_output, list)
        else {"kind": "full"},
        region={"kind": "channel"},
        input_range=input_range,
        output_range=output_range,
        full_input_range=full_input,
        full_output_range=full_output,
        contribution_kind=contribution,
        source_state_ids=source_state_ids or ["model_input"],
        downstream_consumer_ids=downstream_consumer_ids
        if downstream_consumer_ids is not None
        else ["Relu1"],
        **kwargs,
    )


def _partial_binding(
    origin,
    input_range,
    output_range,
    full_input,
    full_output,
    *,
    fragment_index=0,
    fragment_count=2,
    **kwargs,
):
    required_fragment_ids = [
        f"{origin}:fragment:{index}"
        for index in range(fragment_count)
    ]
    return _binding(
        origin,
        "partial_contribution",
        input_range,
        output_range,
        full_input,
        full_output,
        fragment_ids=[required_fragment_ids[fragment_index]],
        required_fragment_ids=required_fragment_ids,
        **kwargs,
    )


def _static_result(item_id, binding, status="ok"):
    return StaticEvalResult(
        item_id=item_id,
        item_kind="weighted_partial",
        status=status,
        layer_results=[],
        item_totals=MetricTotals(1e-9, 2e-9, 2.0, 3.0),
        mapping_summary={},
        raw_mnsim_summary={},
        boundary_bindings=[binding],
        warnings=[],
    )


def _static_request(item_id, binding, status="ok"):
    return BoundaryCommitRequest(
        item_id=item_id,
        item_kind="weighted_physical_round",
        execution_index=0,
        input_view_id=f"view:{item_id}",
        source_kind="static_eval",
        static_eval_result=_static_result(item_id, binding, status=status),
        hook_result=None,
        boundary_bindings=[binding],
    )


def _step_result(item_id, step_id, origin_layer_id="Conv1", status="ok"):
    return StepExecutionResult(
        item_id=item_id,
        step_id=step_id,
        segment_index=int(step_id.rsplit("S", 1)[1]),
        step_kind="weighted_layer_segment",
        source_kind="static_eval",
        status=status,
        origin_layer_ids=[origin_layer_id],
        metrics=MetricTotals(0.0, 0.0, 0.0, 0.0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[],
        warnings=[],
    )


def _round_finalize_request(item_id, bindings, *, input_events=None, step_results=None):
    return BoundaryCommitRequest(
        item_id=item_id,
        item_kind="weighted_physical_round",
        execution_index=0,
        input_view_id=f"view:{item_id}",
        source_kind="round_finalize",
        static_eval_result=None,
        hook_result=None,
        boundary_bindings=list(bindings),
        step_results=list(step_results or [_step_result(item_id, f"{item_id}.S0")]),
        input_events=list(input_events or []),
    )


def test_partial_fragment_coverage_promotes_one_logical_complete_state():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")

    b0 = _partial_binding(
        "Conv1", [0, 2], [0, 2], [0, 4], [0, 4], fragment_index=0, fragment_count=3
    )
    r0 = mgr.commit(_static_request("R0", b0))
    assert r0.completion_updates[0]["completion_status"] == BOUNDARY_STATUS_PARTIAL_SUM

    b1 = _partial_binding(
        "Conv1", [2, 4], [0, 2], [0, 4], [0, 4], fragment_index=1, fragment_count=3
    )
    r1 = mgr.commit(_static_request("R1", b1))
    assert r1.completion_updates[0]["completion_status"] == BOUNDARY_STATUS_PARTIAL_SUM

    b2 = _partial_binding(
        "Conv1", [0, 4], [2, 4], [0, 4], [0, 4], fragment_index=2, fragment_count=3
    )
    r2 = mgr.commit(_static_request("R2", b2))
    assert r2.completion_updates[-1]["completion_status"] == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert r2.output_state_bindings[-1]["downstream_visible"] is True


def test_partial_fragment_progress_report_is_incremental_and_compact():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")

    first = _partial_binding(
        "Conv1", [0, 2], [0, 2], [0, 4], [0, 4], fragment_index=0, fragment_count=3
    )
    second = _partial_binding(
        "Conv1", [2, 4], [0, 2], [0, 4], [0, 4], fragment_index=1, fragment_count=3
    )

    mgr.commit(_static_request("R0", first))
    result = mgr.commit(_static_request("R1", second))
    update = result.completion_updates[0]

    assert update["selected_fragment_ids"] == ["Conv1:fragment:1"]
    assert update["covered_fragment_count"] == 2
    assert update["required_fragment_count"] == 3
    assert "covered_fragment_ids" not in update
    assert "required_fragment_ids" not in update


def test_partial_boundary_accepts_same_spatial_scope_for_different_weight_planes():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 128}, tensor_role="activation")
    required = {
        "L000_fc|in:0:128|out:0:128|plane:0",
        "L000_fc|in:0:128|out:0:128|plane:1",
    }
    first = _binding(
        "L000_fc",
        "partial_contribution",
        [0, 128],
        [0, 128],
        [0, 128],
        [0, 128],
        fragment_ids=["L000_fc|in:0:128|out:0:128|plane:0"],
        required_fragment_ids=required,
    )
    second = _binding(
        "L000_fc",
        "partial_contribution",
        [0, 128],
        [0, 128],
        [0, 128],
        [0, 128],
        fragment_ids=["L000_fc|in:0:128|out:0:128|plane:1"],
        required_fragment_ids=required,
    )

    first_result = mgr.commit(_static_request("R0", first))
    second_result = mgr.commit(_static_request("R1", second))

    assert first_result.created_states[-1].status == BOUNDARY_STATUS_PARTIAL_SUM
    assert second_result.updated_states[-1].status == BOUNDARY_STATUS_COMPLETE_OUTPUT


def test_partial_boundary_rejects_duplicate_fragment_even_when_ranges_match():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 128}, tensor_role="activation")
    required = {
        "L000_fc|in:0:128|out:0:128|plane:0",
        "L000_fc|in:0:128|out:0:128|plane:1",
    }
    binding = _binding(
        "L000_fc",
        "partial_contribution",
        [0, 128],
        [0, 128],
        [0, 128],
        [0, 128],
        fragment_ids=["L000_fc|in:0:128|out:0:128|plane:0"],
        required_fragment_ids=required,
    )

    mgr.commit(_static_request("R0", binding))
    with pytest.raises(Phase2Invalid) as exc_info:
        mgr.commit(_static_request("R1", binding))

    assert exc_info.value.code == DUPLICATE_FRAGMENT_CONTRIBUTION


def test_partial_boundary_remains_unreadable_until_every_fragment_is_covered():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 128}, tensor_role="activation")
    binding = _binding(
        "L000_fc",
        "partial_contribution",
        [0, 128],
        [0, 128],
        [0, 128],
        [0, 128],
        fragment_ids=["L000_fc|in:0:128|out:0:128|plane:0"],
        required_fragment_ids=[
            "L000_fc|in:0:128|out:0:128|plane:0",
            "L000_fc|in:0:128|out:0:128|plane:1",
        ],
        downstream_consumer_ids=["R1"],
    )

    result = mgr.commit(_static_request("R0", binding))
    with pytest.raises(Phase2Invalid, match="PARTIAL_BOUNDARY_NOT_READABLE"):
        mgr.build_input_view_for_state_ids("R1", [result.created_states[-1].state_id])


def test_partial_boundary_requires_selected_and_required_fragment_ids():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 128}, tensor_role="activation")
    binding = _binding(
        "L000_fc",
        "partial_contribution",
        [0, 128],
        [0, 128],
        [0, 128],
        [0, 128],
    )

    with pytest.raises(Phase2Invalid, match="INVALID_FRAGMENT_CONTRIBUTION"):
        mgr.commit(_static_request("R0", binding))


def test_partial_completion_publishes_logical_complete_state_for_downstream_contract():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")

    b0 = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 4],
        fragment_index=0,
        downstream_consumer_ids=["Z0"],
    )
    mgr.commit(_static_request("R0", b0))

    b1 = _partial_binding(
        "Conv1",
        [2, 4],
        [0, 4],
        [0, 4],
        [0, 4],
        fragment_index=1,
        downstream_consumer_ids=["Z0"],
    )
    result = mgr.commit(_static_request("R1", b1))

    assert result.completion_updates[-1]["completion_status"] == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert result.output_state_bindings[-1]["state_id"] == "state:Conv1"
    view = mgr.build_input_view_for_state_ids("Z0", ["state:Conv1"])
    assert view.readable_state_ids == ["state:Conv1"]


def test_commit_result_keeps_partial_state_snapshot_after_later_update():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("src:a", shape={"C": 4}, tensor_role="activation")
    mgr.seed_model_input("src:b", shape={"C": 4}, tensor_role="activation")

    b0 = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 8],
        fragment_index=0,
        fragment_count=3,
        source_state_ids=["src:a"],
    )
    r0 = mgr.commit(_static_request("R0", b0))
    original_state = r0.created_states[0]
    assert original_state.status == BOUNDARY_STATUS_PARTIAL_SUM

    b1 = _partial_binding(
        "Conv1",
        [2, 4],
        [0, 4],
        [0, 4],
        [0, 8],
        fragment_index=1,
        fragment_count=3,
        source_state_ids=["src:b"],
    )
    r1 = mgr.commit(_static_request("R1", b1))

    assert r1.updated_states[0].state_id == original_state.state_id
    assert original_state.status == BOUNDARY_STATUS_PARTIAL_SUM
    assert original_state.source_state_ids == ["src:a"]
    assert original_state.region["input_range"] == [0, 2]
    assert original_state.region["output_range"] == [0, 4]


@pytest.mark.parametrize(
    "status",
    [BOUNDARY_STATUS_PARTIAL_SUM, BOUNDARY_STATUS_OUTPUT_SUBSET],
)
def test_downstream_input_view_rejects_partial_state(status):
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=status,
        allowed_consumer_item_ids=["R1"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids("R1", [state.state_id], allow_partial=False)

    assert "partial" in str(excinfo.value) or "output_subset" in str(excinfo.value)


def test_allow_partial_input_view_rejects_output_subset_state():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_OUTPUT_SUBSET,
        allowed_consumer_item_ids=["R1"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids("R1", [state.state_id], allow_partial=True)

    assert "output_subset" in str(excinfo.value)


@pytest.mark.parametrize("partial_origin_layer_id", [None, "Conv2"])
def test_allow_partial_input_view_rejects_partial_sum_without_matching_origin(
    partial_origin_layer_id,
):
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["R1"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids(
            "R1",
            [state.state_id],
            allow_partial=True,
            partial_origin_layer_id=partial_origin_layer_id,
        )

    assert "partial_sum" in str(excinfo.value)


def test_allow_partial_input_view_accepts_partial_sum_for_matching_origin():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["R1"],
    )

    view = mgr.build_input_view_for_state_ids(
        "R1",
        [state.state_id],
        allow_partial=True,
        partial_origin_layer_id="Conv1",
    )

    assert view.readable_state_ids == [state.state_id]
    assert view.state_bindings == {"Conv1": [state.state_id]}


@pytest.mark.parametrize(
    "status",
    [BOUNDARY_STATUS_COMPLETE_OUTPUT, BOUNDARY_STATUS_HOOK_COMPLETE],
)
def test_downstream_visible_input_view_rejects_unlisted_consumer(status):
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=status,
        allowed_consumer_item_ids=["R_allowed"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids("R_blocked", [state.state_id])

    assert "not allowed" in str(excinfo.value)


def test_complete_output_input_view_rejects_empty_allowed_consumer_list():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
        allowed_consumer_item_ids=[],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids("R_any", [state.state_id])

    assert excinfo.value.code == "BOUNDARY_CONSUMER_NOT_ALLOWED"


def test_hook_complete_input_view_rejects_empty_allowed_consumer_list():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="Z0",
        producer_layer_id="Relu1",
        status=BOUNDARY_STATUS_HOOK_COMPLETE,
        allowed_consumer_item_ids=[],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids("R_any", [state.state_id])

    assert excinfo.value.code == "BOUNDARY_CONSUMER_NOT_ALLOWED"


@pytest.mark.parametrize(
    "status",
    [BOUNDARY_STATUS_COMPLETE_OUTPUT, BOUNDARY_STATUS_HOOK_COMPLETE],
)
def test_downstream_visible_input_view_accepts_listed_consumer(status):
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=status,
        allowed_consumer_item_ids=["R_allowed"],
    )

    view = mgr.build_input_view_for_state_ids("R_allowed", [state.state_id])

    assert view.readable_state_ids == [state.state_id]


def test_model_input_is_globally_readable_with_empty_allowed_list():
    mgr = BoundaryStateManager()
    state = mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")

    view = mgr.build_input_view_for_state_ids("R_any", [state.state_id])

    assert state.allowed_consumer_item_ids == []
    assert view.readable_state_ids == [state.state_id]


def test_partial_input_view_rejects_unlisted_consumer_after_origin_match():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["R_allowed"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids(
            "R_blocked",
            [state.state_id],
            allow_partial=True,
            partial_origin_layer_id="Conv1",
        )

    assert "not allowed" in str(excinfo.value)


def test_partial_input_view_rejects_empty_allowed_consumer_list_after_origin_match():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=[],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids(
            "R_any",
            [state.state_id],
            allow_partial=True,
            partial_origin_layer_id="Conv1",
        )

    assert excinfo.value.code == "BOUNDARY_CONSUMER_NOT_ALLOWED"


def test_partial_input_view_accepts_listed_consumer_after_origin_match():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["R_allowed"],
    )

    view = mgr.build_input_view_for_state_ids(
        "R_allowed",
        [state.state_id],
        allow_partial=True,
        partial_origin_layer_id="Conv1",
    )

    assert view.readable_state_ids == [state.state_id]


def test_committed_partial_sum_input_view_accepts_declared_continuation_consumer():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 4],
        downstream_consumer_ids=["R1"],
    )

    result = mgr.commit(_static_request("R0", binding))
    partial_state = result.created_states[0]

    assert partial_state.status == BOUNDARY_STATUS_PARTIAL_SUM
    view = mgr.build_input_view_for_state_ids(
        "R1",
        [partial_state.state_id],
        allow_partial=True,
        partial_origin_layer_id="Conv1",
    )

    assert view.readable_state_ids == [partial_state.state_id]


def test_committed_partial_sum_input_view_rejects_undeclared_continuation_consumer():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 4],
        downstream_consumer_ids=["R1"],
    )

    result = mgr.commit(_static_request("R0", binding))
    partial_state = result.created_states[0]

    assert partial_state.status == BOUNDARY_STATUS_PARTIAL_SUM
    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids(
            "R2",
            [partial_state.state_id],
            allow_partial=True,
            partial_origin_layer_id="Conv1",
        )

    assert excinfo.value.code == "BOUNDARY_CONSUMER_NOT_ALLOWED"


def test_committed_partial_sum_input_view_rejects_empty_declared_consumer_list():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 4],
        downstream_consumer_ids=[],
    )

    result = mgr.commit(_static_request("R0", binding))
    partial_state = result.created_states[0]

    assert partial_state.status == BOUNDARY_STATUS_PARTIAL_SUM
    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.build_input_view_for_state_ids(
            "R1",
            [partial_state.state_id],
            allow_partial=True,
            partial_origin_layer_id="Conv1",
        )

    assert excinfo.value.code == "BOUNDARY_CONSUMER_NOT_ALLOWED"


def test_input_view_carries_readable_state_metadata_copy():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
        allowed_consumer_item_ids=["R1"],
        shape={"C": 8},
        tensor_role="activation",
    )

    view = mgr.build_input_view_for_state_ids("R1", [state.state_id])

    assert len(view.state_records) == 1
    record = view.state_records[0]
    assert record is not state
    assert record.state_id == state.state_id
    assert record.shape == {"C": 8}
    assert record.status == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert record.region == {"kind": "test"}
    assert record.tensor_role == "activation"
    serialized = view.to_dict()["state_records"][0]
    assert serialized["state_id"] == state.state_id
    assert serialized["status"] == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert serialized["shape"] == {"C": 8}


def test_hook_commit_creates_hook_complete_state():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("state:conv1", shape={"C": 8}, tensor_role="activation")
    binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="hook_output",
        source_state_ids=["state:conv1"],
        downstream_consumer_ids=["Conv2"],
    )
    result = mgr.commit(
        BoundaryCommitRequest(
            item_id="Z0",
            item_kind="zero_xbar_round",
            execution_index=0,
            input_view_id="view:Z0",
            source_kind="hook",
            static_eval_result=None,
            hook_result=HookResult("Z0", "zero_xbar_round", [binding], []),
            boundary_bindings=[binding],
        )
    )

    assert result.created_states[0].status == BOUNDARY_STATUS_HOOK_COMPLETE


def test_hook_commit_rejects_complete_output_source_for_unlisted_consumer():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
        allowed_consumer_item_ids=["Other"],
    )
    binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="hook_output",
        source_state_ids=[state.state_id],
        downstream_consumer_ids=["Conv2"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(
            BoundaryCommitRequest(
                item_id="Z0",
                item_kind="zero_xbar_round",
                execution_index=0,
                input_view_id="view:Z0",
                source_kind="hook",
                static_eval_result=None,
                hook_result=HookResult("Z0", "zero_xbar_round", [binding], []),
                boundary_bindings=[binding],
            )
        )

    assert excinfo.value.code == "BOUNDARY_CONSUMER_NOT_ALLOWED"


def test_hook_commit_accepts_complete_output_source_for_listed_consumer():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
        allowed_consumer_item_ids=["Z0"],
    )
    binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="hook_output",
        source_state_ids=[state.state_id],
        downstream_consumer_ids=["Conv2"],
    )

    result = mgr.commit(
        BoundaryCommitRequest(
            item_id="Z0",
            item_kind="zero_xbar_round",
            execution_index=0,
            input_view_id="view:Z0",
            source_kind="hook",
            static_eval_result=None,
            hook_result=HookResult("Z0", "zero_xbar_round", [binding], []),
            boundary_bindings=[binding],
        )
    )

    assert result.created_states[0].status == BOUNDARY_STATUS_HOOK_COMPLETE


def test_hook_commit_accepts_model_input_source_with_empty_allowed_list():
    mgr = BoundaryStateManager()
    state = mgr.seed_model_input("state:conv1", shape={"C": 8}, tensor_role="activation")
    binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="hook_output",
        source_state_ids=[state.state_id],
        downstream_consumer_ids=["Conv2"],
    )

    result = mgr.commit(
        BoundaryCommitRequest(
            item_id="Z0",
            item_kind="zero_xbar_round",
            execution_index=0,
            input_view_id="view:Z0",
            source_kind="hook",
            static_eval_result=None,
            hook_result=HookResult("Z0", "zero_xbar_round", [binding], []),
            boundary_bindings=[binding],
        )
    )

    assert state.allowed_consumer_item_ids == []
    assert result.created_states[0].status == BOUNDARY_STATUS_HOOK_COMPLETE


def test_build_input_view_uses_execution_item_external_inputs():
    class Item:
        item_id = "R0"
        external_inputs = [
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["R0"],
            )
        ]

    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")

    view = mgr.build_input_view(Item())

    assert view.input_view_id == "view:R0"
    assert view.readable_state_ids == ["model_input"]
    assert view.state_bindings["model_input"] == ["model_input"]


def test_complete_output_commit_is_downstream_visible():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _binding("Conv1", "complete_output", [0, 4], [0, 4], [0, 4], [0, 4])

    result = mgr.commit(_static_request("R0", binding))

    assert result.created_states[0].status == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert result.output_state_bindings[-1]["downstream_visible"] is True


def test_partial_contribution_rejects_input_range_outside_full_bounds():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _partial_binding("Conv1", [-1, 1], [0, 4], [0, 4], [0, 4])

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R0", binding))

    assert excinfo.value.code == "INVALID_PARTIAL_RANGE"
    assert "input_range" in excinfo.value.detail


def test_partial_contribution_rejects_output_range_outside_full_bounds():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _partial_binding("Conv1", [0, 4], [0, 5], [0, 4], [0, 4])

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R0", binding))

    assert excinfo.value.code == "INVALID_PARTIAL_RANGE"
    assert "output_range" in excinfo.value.detail


def test_partial_contribution_accepts_overlapping_input_ranges_for_distinct_fragments():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding(
        "Conv1", [0, 3], [0, 4], [0, 4], [0, 4], fragment_index=0
    )
    b1 = _partial_binding(
        "Conv1", [2, 4], [0, 4], [0, 4], [0, 4], fragment_index=1
    )

    mgr.commit(_static_request("R0", b0))
    result = mgr.commit(_static_request("R1", b1))

    assert result.completion_updates[0]["completion_status"] == BOUNDARY_STATUS_COMPLETE_OUTPUT


def test_partial_contribution_accepts_overlapping_output_ranges_for_distinct_fragments():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding(
        "Conv1", [0, 2], [0, 3], [0, 4], [0, 4], fragment_index=0
    )
    b1 = _partial_binding(
        "Conv1", [0, 2], [2, 4], [0, 4], [0, 4], fragment_index=1
    )

    mgr.commit(_static_request("R0", b0))
    result = mgr.commit(_static_request("R1", b1))

    assert result.completion_updates[0]["completion_status"] == BOUNDARY_STATUS_COMPLETE_OUTPUT


def test_partial_contribution_accepts_additional_input_for_same_output_accumulator():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding(
        "Conv1", [0, 2], [0, 4], [0, 4], [0, 8], fragment_index=0, fragment_count=3
    )
    b1 = _partial_binding(
        "Conv1", [2, 4], [0, 4], [0, 4], [0, 8], fragment_index=1, fragment_count=3
    )

    mgr.commit(_static_request("R0", b0))
    result = mgr.commit(_static_request("R1", b1))

    assert result.completion_updates[0]["completion_status"] == BOUNDARY_STATUS_PARTIAL_SUM
    assert result.completion_updates[0]["selected_fragment_ids"] == ["Conv1:fragment:1"]
    assert result.completion_updates[0]["covered_fragment_count"] == 2
    assert result.completion_updates[0]["required_fragment_count"] == 3


def test_partial_contribution_accepts_distinct_output_subset_regions_for_same_origin():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding(
        "Conv1", [0, 2], [0, 4], [0, 4], [0, 8], fragment_index=0, fragment_count=3
    )
    b0.shape = {"C": 4, "H": 1, "W": 1}
    b0.region = {
        "kind": "materialized_scope",
        "scope": {"input_range": [0, 2], "output_range": [0, 4]},
    }
    b1 = _partial_binding(
        "Conv1", [2, 4], [0, 4], [0, 4], [0, 8], fragment_index=1, fragment_count=3
    )
    b1.shape = {"C": 4, "H": 1, "W": 1}
    b1.region = {
        "kind": "materialized_scope",
        "scope": {"input_range": [2, 4], "output_range": [0, 4]},
    }
    b2 = _partial_binding(
        "Conv1", [0, 4], [4, 8], [0, 4], [0, 8], fragment_index=2, fragment_count=3
    )
    b2.shape = {"C": 4, "H": 1, "W": 1}
    b2.region = {
        "kind": "materialized_scope",
        "scope": {"input_range": [0, 4], "output_range": [4, 8]},
    }

    mgr.commit(_static_request("R0", b0))
    mgr.commit(_static_request("R1", b1))
    result = mgr.commit(_static_request("R2", b2))

    complete_state = result.updated_states[-1]
    assert complete_state.status == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert complete_state.shape["C"] == 8
    assert complete_state.region["input_range"] == [0, 4]
    assert complete_state.region["output_range"] == [0, 8]


def test_partial_contribution_rejects_duplicate_fragment_after_completion():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding(
        "Conv1", [0, 4], [0, 3], [0, 4], [0, 4], fragment_index=0
    )
    b1 = _partial_binding(
        "Conv1", [0, 4], [2, 4], [0, 4], [0, 4], fragment_index=1
    )

    mgr.commit(_static_request("R0", b0))
    mgr.commit(_static_request("R1", b1))
    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R2", b0))

    assert excinfo.value.code == DUPLICATE_FRAGMENT_CONTRIBUTION


def test_partial_complete_state_source_ids_include_all_contributions():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("src:a", shape={"C": 4}, tensor_role="activation")
    mgr.seed_model_input("src:b", shape={"C": 4}, tensor_role="activation")
    mgr.seed_model_input("src:c", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 2],
        [0, 4],
        [0, 4],
        fragment_index=0,
        fragment_count=3,
        source_state_ids=["src:a"],
    )
    b1 = _partial_binding(
        "Conv1",
        [2, 4],
        [0, 2],
        [0, 4],
        [0, 4],
        fragment_index=1,
        fragment_count=3,
        source_state_ids=["src:b"],
    )
    b2 = _partial_binding(
        "Conv1",
        [0, 4],
        [2, 4],
        [0, 4],
        [0, 4],
        fragment_index=2,
        fragment_count=3,
        source_state_ids=["src:c"],
    )

    mgr.commit(_static_request("R0", b0))
    mgr.commit(_static_request("R1", b1))
    result = mgr.commit(_static_request("R2", b2))

    complete_state = result.updated_states[-1]
    assert complete_state.status == BOUNDARY_STATUS_COMPLETE_OUTPUT
    assert complete_state.source_state_ids == ["src:a", "src:b", "src:c"]


def test_context_only_binding_creates_audit_event_without_state():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _binding(
        "ConvContext",
        "complete_output",
        [0, 4],
        [0, 4],
        [0, 4],
        [0, 4],
        report_role="context_only",
    )

    result = mgr.commit(_static_request("R0", binding))

    assert result.created_states == []
    assert result.updated_states == []
    assert result.events[0].event_kind == "ignore_context"
    assert result.events[0].detail["origin_layer_id"] == "ConvContext"


def test_static_eval_status_must_be_ok():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    binding = _binding("Conv1", "complete_output", [0, 4], [0, 4], [0, 4], [0, 4])

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R0", binding, status="error"))

    assert "status" in str(excinfo.value)


def test_source_state_ids_must_exist():
    mgr = BoundaryStateManager()
    binding = _binding(
        "Conv1",
        "complete_output",
        [0, 4],
        [0, 4],
        [0, 4],
        [0, 4],
        source_state_ids=["missing"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R0", binding))

    assert "missing source state" in str(excinfo.value)


@pytest.mark.parametrize("contribution", ["complete_output", "partial_contribution"])
def test_static_reportable_commit_rejects_output_subset_source(contribution):
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_OUTPUT_SUBSET,
        allowed_consumer_item_ids=["R1"],
    )
    if contribution == "partial_contribution":
        binding = _partial_binding(
            "Conv2",
            [0, 4],
            [0, 4],
            [0, 4],
            [0, 4],
            source_state_ids=[state.state_id],
        )
    else:
        binding = _binding(
            "Conv2",
            contribution,
            [0, 4],
            [0, 4],
            [0, 4],
            [0, 4],
            source_state_ids=[state.state_id],
        )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R1", binding))

    assert excinfo.value.code == "BOUNDARY_STATE_NOT_READABLE"


def test_static_reportable_commit_rejects_complete_source_for_unlisted_consumer():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
        allowed_consumer_item_ids=["Other"],
    )
    binding = _binding(
        "Conv2",
        "complete_output",
        [0, 4],
        [0, 4],
        [0, 4],
        [0, 4],
        source_state_ids=[state.state_id],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R1", binding))

    assert excinfo.value.code == "BOUNDARY_CONSUMER_NOT_ALLOWED"


def test_static_reportable_commit_accepts_complete_source_for_listed_consumer():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
        allowed_consumer_item_ids=["R1"],
    )
    binding = _binding(
        "Conv2",
        "complete_output",
        [0, 4],
        [0, 4],
        [0, 4],
        [0, 4],
        source_state_ids=[state.state_id],
    )

    result = mgr.commit(_static_request("R1", binding))

    assert result.created_states[0].status == BOUNDARY_STATUS_COMPLETE_OUTPUT


def test_static_partial_continuation_accepts_partial_sum_source_for_same_origin():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["R1"],
    )
    binding = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 4],
        source_state_ids=[state.state_id],
    )

    result = mgr.commit(_static_request("R1", binding))

    assert result.created_states[0].status == BOUNDARY_STATUS_PARTIAL_SUM


def test_static_partial_continuation_rejects_partial_sum_source_for_wrong_origin():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["R1"],
    )
    binding = _partial_binding(
        "Conv2",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 4],
        source_state_ids=[state.state_id],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R1", binding))

    assert excinfo.value.code == "PARTIAL_BOUNDARY_ORIGIN_MISMATCH"


def test_partial_contribution_rejects_metadata_shape_mismatch_for_same_origin():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding("Conv1", [0, 2], [0, 4], [0, 4], [0, 8])
    b1 = _partial_binding("Conv1", [2, 4], [0, 4], [0, 4], [0, 8], fragment_index=1)
    b1.shape = {"C": 8, "H": 99, "W": 1}

    mgr.commit(_static_request("R0", b0))
    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R1", b1))

    assert excinfo.value.code == "PARTIAL_METADATA_MISMATCH"
    assert "shape" in excinfo.value.detail


def test_partial_contribution_rejects_metadata_consumers_mismatch_for_same_origin():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding(
        "Conv1",
        [0, 2],
        [0, 4],
        [0, 4],
        [0, 8],
        downstream_consumer_ids=["Relu1"],
    )
    b1 = _partial_binding(
        "Conv1",
        [2, 4],
        [0, 4],
        [0, 4],
        [0, 8],
        fragment_index=1,
        downstream_consumer_ids=["Relu2"],
    )

    mgr.commit(_static_request("R0", b0))
    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R1", b1))

    assert excinfo.value.code == "PARTIAL_METADATA_MISMATCH"
    assert "downstream_consumer_ids" in excinfo.value.detail


def test_partial_contribution_rejects_metadata_region_mismatch_for_same_origin():
    mgr = BoundaryStateManager()
    mgr.seed_model_input("model_input", shape={"C": 4}, tensor_role="activation")
    b0 = _partial_binding("Conv1", [0, 2], [0, 4], [0, 4], [0, 8])
    b1 = _partial_binding("Conv1", [2, 4], [0, 4], [0, 4], [0, 8], fragment_index=1)
    b1.region = {"kind": "different_channel_layout"}

    mgr.commit(_static_request("R0", b0))
    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(_static_request("R1", b1))

    assert excinfo.value.code == "PARTIAL_METADATA_MISMATCH"
    assert "region" in excinfo.value.detail


def test_hook_input_must_be_complete_or_model_input():
    mgr = BoundaryStateManager()
    state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_PARTIAL_SUM,
        allowed_consumer_item_ids=["Z0"],
    )
    binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="hook_output",
        source_state_ids=[state.state_id],
        downstream_consumer_ids=["Conv2"],
    )

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(
            BoundaryCommitRequest(
                item_id="Z0",
                item_kind="zero_xbar_round",
                execution_index=0,
                input_view_id="view:Z0",
                source_kind="hook",
                static_eval_result=None,
                hook_result=HookResult("Z0", "zero_xbar_round", [binding], []),
                boundary_bindings=[binding],
            )
        )

    assert "hook input" in str(excinfo.value)


def test_boundary_input_view_emits_reads_only_for_boundary_sram_states():
    mgr = BoundaryStateManager()
    model_input = mgr.seed_model_input("model_input", shape={"C": 4})
    boundary_state = mgr._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status=BOUNDARY_STATUS_COMPLETE_OUTPUT,
        allowed_consumer_item_ids=["R1"],
        state_id="state:Conv1",
    )

    model_view = mgr.build_input_view_for_state_ids("R0", [model_input.state_id])
    boundary_view = mgr.build_input_view_for_state_ids("R1", [boundary_state.state_id])

    assert model_view.events == []
    assert [(event.event_kind, event.item_id, event.state_id) for event in boundary_view.events] == [
        ("read_boundary_state", "R1", "state:Conv1"),
    ]


def test_same_round_w_z_w_has_no_boundary_read_or_write_events():
    mgr = BoundaryStateManager()
    model_input = mgr.seed_model_input("model_input", shape={"C": 4})
    input_view = mgr.build_input_view_for_state_ids("R0", [model_input.state_id])

    result = mgr.commit(
        _round_finalize_request(
            "R0",
            [],
            input_events=input_view.events,
            step_results=[
                _step_result("R0", "R0.S0", "Conv1"),
                _step_result("R0", "R0.S1", "Relu1"),
                _step_result("R0", "R0.S2", "Conv2"),
            ],
        )
    )

    assert [event.event_kind for event in result.events if "boundary" in event.event_kind] == []


def test_cross_round_output_has_one_boundary_write_and_later_read():
    mgr = BoundaryStateManager()
    model_input = mgr.seed_model_input("model_input", shape={"C": 4})
    first_input = mgr.build_input_view_for_state_ids("R0", [model_input.state_id])
    binding = _binding(
        "Conv1",
        "complete_output",
        [0, 4],
        [0, 4],
        [0, 4],
        [0, 4],
        downstream_consumer_ids=["R1"],
    )
    first_result = mgr.commit(
        _round_finalize_request("R0", [binding], input_events=first_input.events)
    )
    second_input = mgr.build_input_view_for_state_ids("R1", ["state:Conv1"])
    second_result = mgr.commit(
        _round_finalize_request("R1", [], input_events=second_input.events)
    )

    events = [*first_result.events, *second_result.events]
    writes = [event for event in events if event.event_kind == "write_boundary_state"]
    reads = [event for event in events if event.event_kind == "read_boundary_state"]
    assert [(event.item_id, event.state_id) for event in writes] == [("R0", "state:Conv1")]
    assert [(event.item_id, event.state_id) for event in reads] == [("R1", "state:Conv1")]


def test_round_finalize_rejects_failed_step_before_state_changes():
    mgr = BoundaryStateManager()

    with pytest.raises(Phase2Invalid) as excinfo:
        mgr.commit(
            _round_finalize_request(
                "R0",
                [],
                step_results=[_step_result("R0", "R0.S0", status="error")],
            )
        )

    assert excinfo.value.code == "STATIC_EVAL_NOT_OK"
