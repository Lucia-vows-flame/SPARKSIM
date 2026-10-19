from pathlib import Path
from types import SimpleNamespace

from MNSIM.Scheduling.phase2.boundary import BoundaryStateManager
from MNSIM.Scheduling.phase2.ddr import (
    NullDdrBackend,
    _events_to_transactions,
    _summarize_stats,
    load_ddr_backend,
)
from MNSIM.Scheduling.phase2.types import (
    BoundaryAccessEvent,
    BoundaryCommitRequest,
    BoundaryCommitResult,
    MetricTotals,
    STATIC_STATUS_OK,
    StepExecutionResult,
)
from MNSIM.Scheduling.phase2.report import ReportAggregator
from MNSIM.Scheduling.round_types import ResourceProof


def test_disabled_backend_is_stable_and_does_not_count_events():
    result = NullDdrBackend().score_events(
        [
            BoundaryAccessEvent(
                event_kind="write_boundary_state",
                item_id="R0",
                state_id="state:x",
            )
        ]
    )

    assert result == {
        "used": False,
        "backend": "disabled",
        "transaction_count": 0,
        "read_transactions": 0,
        "write_transactions": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "simulated_cycles": 0,
        "latency_s": 0.0,
        "energy_j": 0.0,
        "power_w": 0.0,
    }


def test_boundary_commit_calls_ddr_backend_once():
    class FakeBackend:
        enabled = True

        def __init__(self):
            self.events = []

        def score_events(self, events):
            self.events.append(list(events))
            return {"used": True, "backend": "fake", "latency_s": 1e-9, "energy_j": 2e-9}

    backend = FakeBackend()
    manager = BoundaryStateManager(ddr_backend=backend)
    manager._create_state_for_test(
        producer_item_id="R0",
        producer_layer_id="Conv1",
        status="complete_output",
        allowed_consumer_item_ids=["R1"],
        state_id="state:conv1",
        shape={"Outputchannel": 8, "Outputsize": [2, 2]},
    )
    view = manager.build_input_view_for_state_ids("R1", ["state:conv1"])
    result = manager.commit(
        BoundaryCommitRequest(
            item_id="R1",
            item_kind="weighted_physical_round",
            execution_index=1,
            input_view_id=view.input_view_id,
            source_kind="round_finalize",
            static_eval_result=None,
            hook_result=None,
            boundary_bindings=[],
            input_events=view.events,
            step_results=[
                StepExecutionResult(
                    item_id="R1",
                    step_id="R1.S0",
                    segment_index=0,
                    step_kind="test",
                    source_kind="hook",
                    status=STATIC_STATUS_OK,
                    origin_layer_ids=[],
                    metrics=MetricTotals(0.0, 0.0, 0.0, 0.0),
                    layer_results=[],
                    mapping_summary={},
                    raw_mnsim_audit=[],
                    boundary_bindings=[],
                    warnings=[],
                )
            ],
        )
    )

    assert result.ddr_transfer["backend"] == "fake"
    assert len(backend.events) == 1
    assert [event.event_kind for event in backend.events[0]] == ["read_boundary_state"]


def test_event_translation_uses_payload_bytes_and_stable_addresses():
    events = [
        BoundaryAccessEvent(
            event_kind="read_boundary_state",
            item_id="R1",
            state_id="state:a",
            detail={"payload_bytes": 129},
        ),
        BoundaryAccessEvent(
            event_kind="write_boundary_state",
            item_id="R1",
            state_id="state:b",
            detail={"payload_bytes": 64},
        ),
    ]

    transactions = _events_to_transactions(
        events, transaction_bytes=64, default_bitwidth=8
    )
    assert len(transactions) == 4
    assert [operation for _, operation in transactions] == [
        "READ",
        "READ",
        "READ",
        "WRITE",
    ]
    assert transactions[0][0] % 64 == 0
    assert transactions[1][0] == transactions[0][0] + 64


def test_stats_summary_converts_pj_and_cycles_to_project_units(tmp_path: Path):
    config = tmp_path / "dram.ini"
    config.write_text("[timing]\ntCK = 0.63\n", encoding="utf-8")
    stats = {
        "0": {
            "num_reads_done": 2,
            "num_writes_done": 1,
            "average_read_latency": 20.0,
            "total_energy": 1000.0,
        }
    }
    summary = _summarize_stats(
        stats,
        transactions=[(0, "READ"), (64, "WRITE")],
        transaction_bytes=64,
        simulated_cycles=128,
        config_path=config,
    )

    assert summary["read_bytes"] == 64
    assert summary["write_bytes"] == 64
    assert summary["latency_s"] == 12.6e-9
    assert summary["energy_j"] == 1e-9


def test_report_aggregates_boundary_ddr_cost_once():
    item = SimpleNamespace(
        item_id="R0",
        item_kind="weighted_physical_round",
        item_index=0,
        round_kind="full_layer_pack",
        resource_proof=ResourceProof(16, 1),
    )
    step = StepExecutionResult(
        item_id="R0",
        step_id="R0.S0",
        segment_index=0,
        step_kind="weighted_layer_segment",
        source_kind="hook",
        status=STATIC_STATUS_OK,
        origin_layer_ids=["Conv1"],
        metrics=MetricTotals(2e-9, 3e-9, 1.5, 0.0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[],
        warnings=[],
    )
    commit = BoundaryCommitResult(
        item_id="R0",
        item_kind="weighted_physical_round",
        status=STATIC_STATUS_OK,
        created_states=[],
        updated_states=[],
        released_states=[],
        events=[],
        completion_updates=[],
        visibility_updates=[],
        output_state_bindings=[],
        warnings=[],
        ddr_transfer={
            "used": True,
            "backend": "dramsim3",
            "latency_s": 5e-9,
            "energy_j": 7e-9,
        },
    )
    aggregator = ReportAggregator("m0", "SimConfig.ini", 1, "mac_as_1_op")
    record = aggregator.record_round(item, None, [step], commit)
    report = aggregator.finalize()

    assert record["latency_s"] == 7e-9
    assert record["energy_j"] == 10e-9
    assert report.inference_metrics["ddr_boundary_transfer_latency_s"] == 5e-9
    assert report.inference_metrics["ddr_boundary_transfer_energy_j"] == 7e-9
    assert report.inference_metrics["accounting_scope"]["ddr_boundary_transfer"][
        "backend"
    ] == "dramsim3"


def test_simconfig_default_keeps_ddr_disabled():
    backend = load_ddr_backend(Path("SimConfig.ini"))
    assert backend.enabled is False
