from typing import get_type_hints

import MNSIM.Scheduling.phase2 as phase2
from MNSIM.Scheduling.phase2.types import (
    BOUNDARY_STATUS_COMPLETE_OUTPUT,
    BOUNDARY_STATUS_HOOK_COMPLETE,
    CONTRIBUTION_COMPLETE_OUTPUT,
    CONTRIBUTION_HOOK_OUTPUT,
    CONTRIBUTION_STRUCTURAL_OUTPUT,
    REPORT_ROLE_CONTEXT_ONLY,
    REPORT_ROLE_REPORTABLE,
    SOURCE_KIND_HOOK,
    STATIC_STATUS_OK,
    BoundaryAccessEvent,
    BoundaryBinding,
    BoundaryCommitRequest,
    BoundaryCommitResult,
    BoundaryInputView,
    BoundaryStateRecord,
    HookResult,
    LoweredItem,
    MetricTotals,
    Phase2Invalid,
    Phase2Report,
    StaticEvalFrame,
    StaticEvalRequest,
    StaticEvalResult,
    StaticLayerResult,
)


def test_static_eval_result_serializes_reportable_and_context_layers():
    binding = BoundaryBinding(
        local_layer_id=1,
        hook_node_id=None,
        origin_layer_id="Pool1",
        origin_layer_type="pooling",
        report_role=REPORT_ROLE_REPORTABLE,
        tensor_role="activation",
        shape={"Outputchannel": 8, "Outputsize": [16, 16]},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind=CONTRIBUTION_STRUCTURAL_OUTPUT,
        source_state_ids=["state:conv1"],
        downstream_consumer_ids=["Conv2"],
    )
    result = StaticEvalResult(
        item_id="Z0",
        item_kind="zero_xbar_structural",
        status=STATIC_STATUS_OK,
        layer_results=[
            StaticLayerResult(
                local_layer_id=0,
                origin_layer_id="ctx:0",
                layer_type="conv",
                report_role=REPORT_ROLE_CONTEXT_ONLY,
                latency_s=0.0,
                begin_time_s=0.0,
                finish_time_s=0.0,
                area_um2=10.0,
                power_w=1.0,
                energy_j=1e-9,
                tcg_mapping={"tilenum": 1},
            ),
            StaticLayerResult(
                local_layer_id=1,
                origin_layer_id="Pool1",
                layer_type="pooling",
                report_role=REPORT_ROLE_REPORTABLE,
                latency_s=2e-9,
                begin_time_s=0.0,
                finish_time_s=2e-9,
                area_um2=3.0,
                power_w=0.5,
                energy_j=1e-9,
                tcg_mapping={"tilenum": 1},
            ),
        ],
        item_totals=MetricTotals(latency_s=2e-9, energy_j=1e-9, power_w=0.5, area_um2=3.0),
        mapping_summary={"layer_count": 2},
        raw_mnsim_summary={"raw_latency_total_s": 7e-9},
        boundary_bindings=[binding],
        warnings=["context layer filtered"],
    )

    data = result.to_dict()

    assert data["layer_results"][0]["report_role"] == "context_only"
    assert data["item_totals"]["latency_s"] == 2e-9
    assert data["boundary_bindings"][0]["origin_layer_id"] == "Pool1"


def test_boundary_commit_request_supports_hook_only_binding():
    hook_binding = BoundaryBinding(
        local_layer_id=None,
        hook_node_id="Relu1",
        origin_layer_id="Relu1",
        origin_layer_type="relu",
        report_role=REPORT_ROLE_REPORTABLE,
        tensor_role="activation",
        shape={"kind": "same_as_input"},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind=CONTRIBUTION_HOOK_OUTPUT,
        source_state_ids=["state:conv1"],
        downstream_consumer_ids=["Conv2"],
    )
    request = BoundaryCommitRequest(
        item_id="R0",
        item_kind="weighted_physical_round",
        execution_index=0,
        input_view_id="view:R0",
        source_kind=SOURCE_KIND_HOOK,
        static_eval_result=None,
        hook_result=HookResult(
            item_id="R0",
            item_kind="weighted_physical_round",
            hook_bindings=[hook_binding],
            warnings=[],
        ),
        boundary_bindings=[hook_binding],
    )

    assert request.to_dict()["boundary_bindings"][0]["local_layer_id"] is None
    assert request.boundary_bindings[0].hook_node_id == "Relu1"


def test_boundary_records_and_commit_result_schema():
    state = BoundaryStateRecord(
        state_id="state:relu1",
        producer_item_id="R0",
        producer_layer_id="Relu1",
        tensor_role="activation",
        shape={"C": 8},
        region={"kind": "full"},
        status=BOUNDARY_STATUS_HOOK_COMPLETE,
        storage_backend="boundary_sram",
        allowed_consumer_item_ids=["R1"],
        lifetime_policy="release_after_consumers",
        source_state_ids=["state:conv1"],
    )
    view = BoundaryInputView(
        input_view_id="view:R1",
        item_id="R1",
        readable_state_ids=["state:relu1"],
        state_bindings={"L002_conv": ["state:relu1"]},
        warnings=[],
    )
    commit = BoundaryCommitResult(
        item_id="R1",
        item_kind="weighted_physical_round",
        status="ok",
        created_states=[state],
        updated_states=[],
        released_states=[],
        events=[],
        completion_updates=[
            {
                "origin_layer_id": "Conv2",
                "completion_status": BOUNDARY_STATUS_COMPLETE_OUTPUT,
                "state_id": "state:conv2",
                "input_range_covered": {"kind": "full"},
                "output_range_covered": {"kind": "full"},
                "downstream_visible": True,
            }
        ],
        visibility_updates=[],
        output_state_bindings=[],
        warnings=[],
    )

    assert view.input_view_id == "view:R1"
    assert commit.created_states[0].status == BOUNDARY_STATUS_HOOK_COMPLETE
    assert CONTRIBUTION_COMPLETE_OUTPUT == "complete_output"


def test_round_finalize_schema_carries_step_results_and_input_events():
    event = BoundaryAccessEvent(
        event_kind="read_boundary_state",
        item_id="R0",
        state_id="state:input",
    )
    step = phase2.StepExecutionResult(
        item_id="R0",
        step_id="R0.S0",
        segment_index=0,
        step_kind="weighted_layer_segment",
        source_kind="static_eval",
        status=STATIC_STATUS_OK,
        origin_layer_ids=["Conv1"],
        metrics=MetricTotals(0.0, 0.0, 0.0, 0.0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[],
        warnings=[],
    )
    view = BoundaryInputView(
        input_view_id="view:R0",
        item_id="R0",
        readable_state_ids=["state:input"],
        state_bindings={"Conv1": ["state:input"]},
        warnings=[],
        events=[event],
    )
    request = BoundaryCommitRequest(
        item_id="R0",
        item_kind="weighted_physical_round",
        execution_index=0,
        input_view_id=view.input_view_id,
        source_kind="round_finalize",
        static_eval_result=None,
        hook_result=None,
        boundary_bindings=[],
        step_results=[step],
        input_events=view.events,
    )

    assert phase2.SOURCE_KIND_ROUND_FINALIZE == "round_finalize"
    assert request.to_dict()["step_results"][0]["step_id"] == "R0.S0"
    assert request.to_dict()["input_events"][0]["event_kind"] == "read_boundary_state"


def test_phase2_package_exports_public_schema_names():
    expected_exports = {
        "MetricTotals": MetricTotals,
        "BoundaryBinding": BoundaryBinding,
        "StaticEvalRequest": StaticEvalRequest,
        "BoundaryCommitResult": BoundaryCommitResult,
        "Phase2Invalid": Phase2Invalid,
    }

    for name, exported in expected_exports.items():
        assert getattr(phase2, name) is exported
        assert name in phase2.__all__


def test_step_execution_result_serializes_step_id_and_runtime_types_remain_nonreportable():
    step_execution_result = getattr(phase2, "StepExecutionResult", None)
    prepared_round = getattr(phase2, "PreparedRound", None)
    lowered_step = getattr(phase2, "LoweredStep", None)

    assert step_execution_result is not None
    assert prepared_round is not None
    assert lowered_step is not None

    result = step_execution_result(
        item_id="R0",
        step_id="R0.S0",
        segment_index=0,
        step_kind="weighted_layer_segment",
        source_kind="static_eval",
        status=STATIC_STATUS_OK,
        origin_layer_ids=["Conv1"],
        metrics=MetricTotals(latency_s=1e-9, energy_j=2e-9, power_w=2.0, area_um2=3.0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[],
        warnings=[],
    )

    assert result.to_dict()["step_id"] == "R0.S0"
    assert not hasattr(prepared_round, "to_dict")
    assert not hasattr(lowered_step, "to_dict")


def test_remaining_public_schemas_and_structured_event_serialization():
    invalid = Phase2Invalid("INVALID_BOUNDARY", "missing source state")
    event = BoundaryAccessEvent(
        event_kind="read",
        item_id="R0",
        state_id="state:input",
        detail={"consumer": "R0"},
    )
    input_view = BoundaryInputView(
        input_view_id="view:R0",
        item_id="R0",
        readable_state_ids=["state:input"],
        state_bindings={"L000_conv": ["state:input"]},
        warnings=[],
    )
    lowered = LoweredItem(
        item_id="R0",
        item_kind="weighted_physical_round",
        execution_index=0,
        input_view=input_view,
        static_eval_request=None,
        hook_result=None,
        boundary_bindings=[],
        warnings=["hook-only item"],
    )
    report = Phase2Report(
        model_name="m0",
        sim_config_path="SimConfig.ini",
        feasible=True,
        execution_item_count=1,
        inference_metrics={"latency_s": 0.0},
        item_results=[lowered.to_dict()],
        step_results=[],
        raw_mnsim_audit=[{"raw_latency_total_s": 0.0}],
        boundary_events=[event],
        warnings=[],
        assumptions=[],
    )

    data = report.to_dict()

    assert invalid.code == "INVALID_BOUNDARY"
    assert invalid.detail == "missing source state"
    assert invalid.reason == "INVALID_BOUNDARY: missing source state"
    assert data["boundary_events"][0]["event_kind"] == "read"
    assert data["boundary_events"][0]["detail"]["consumer"] == "R0"
    assert data["item_results"][0]["input_view"]["input_view_id"] == "view:R0"
    assert get_type_hints(BoundaryCommitResult)["events"] == list[BoundaryAccessEvent]
    assert get_type_hints(Phase2Report)["boundary_events"] == list[BoundaryAccessEvent]
