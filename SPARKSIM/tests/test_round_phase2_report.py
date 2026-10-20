from types import SimpleNamespace

import pytest

import MNSIM.Scheduling.phase2 as phase2

from MNSIM.Scheduling.phase2.report import ReportAggregator
from MNSIM.Scheduling.phase2.types import (
    BoundaryCommitResult,
    MetricTotals,
    STATIC_STATUS_OK,
    StepExecutionResult,
)
from MNSIM.Scheduling.round_types import ResourceProof


def _round(item_id, used_xbars):
    return SimpleNamespace(
        item_id=item_id,
        item_kind="weighted_physical_round",
        item_index=0,
        round_kind="full_layer_pack",
        resource_proof=ResourceProof(16, used_xbars),
    )


def _placement(area_um2, mapping_summary=None):
    return SimpleNamespace(
        item_id="R0",
        mapped_area_um2=area_um2,
        mapping_summary=mapping_summary
        if mapping_summary is not None
        else {"mapping_layer_count": 2, "weighted_step_ids": ["R0.S0", "R0.S2"]},
    )


def _step(
    step_id,
    origin_layer_id,
    latency_s,
    energy_j,
    area_um2,
    module_energy_j=None,
    raw_adc=None,
):
    raw_summary = {
        "reportable_module_energy_j": dict(module_energy_j or {})
    }
    if raw_adc:
        raw_summary.update(raw_adc)
    return StepExecutionResult(
        item_id="R0",
        step_id=step_id,
        segment_index=int(step_id.rsplit("S", 1)[1]),
        step_kind="weighted_layer_segment",
        source_kind="static_eval",
        status=STATIC_STATUS_OK,
        origin_layer_ids=[origin_layer_id],
        metrics=MetricTotals(
            latency_s=latency_s,
            energy_j=energy_j,
            power_w=energy_j / latency_s if latency_s > 0 else 0.0,
            area_um2=area_um2,
        ),
        layer_results=[],
        mapping_summary={"scope_count": 1},
        raw_mnsim_audit=[
            {
                "audit_kind": "static",
                "step_id": step_id,
                "raw_mnsim_summary": {
                    **raw_summary,
                },
            }
        ],
        boundary_bindings=[],
        warnings=[],
    )


def _analog_lif_step(step_id="R0.S1"):
    audit = {
        "lut_schema_id": "analog-lif-v1",
        "lut_profile": "synthetic",
        "lut_root": "/tmp/lut",
        "interpolation": "bounded_bilinear",
        "negative_vmem_policy": "error",
        "checksum_verified": True,
        "query_counts": {"frontend": 1, "lif": 1},
        "frontend_latency_s": 1e-9,
        "frontend_energy_j": 10e-9,
        "lif_latency_s": 2e-9,
        "lif_energy_j": 20e-9,
        "latency_s": 3e-9,
        "energy_j": 30e-9,
        "power_w": 10.0,
        "state": {},
    }
    return StepExecutionResult(
        item_id="R0",
        step_id=step_id,
        segment_index=1,
        step_kind="lif",
        source_kind="analog_lif",
        status=STATIC_STATUS_OK,
        origin_layer_ids=["L1"],
        metrics=MetricTotals(3e-9, 30e-9, 10.0, 0.0),
        layer_results=[],
        mapping_summary={"analog_lif": audit},
        raw_mnsim_audit=[],
        boundary_bindings=[],
        warnings=[],
    )


def _hook_step(step_id, origin_layer_id):
    return StepExecutionResult(
        item_id="R0",
        step_id=step_id,
        segment_index=int(step_id.rsplit("S", 1)[1]),
        step_kind="zero_xbar_segment",
        source_kind="hook",
        status=STATIC_STATUS_OK,
        origin_layer_ids=[origin_layer_id],
        metrics=MetricTotals(0.0, 0.0, 0.0, 0.0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[],
        warnings=[],
    )


def _commit(item_id, origin_layer_id):
    return BoundaryCommitResult(
        item_id=item_id,
        item_kind="weighted_physical_round",
        status=STATIC_STATUS_OK,
        created_states=[],
        updated_states=[],
        released_states=[],
        events=[],
        completion_updates=[
            {
                "origin_layer_id": origin_layer_id,
                "completion_status": "complete_output",
                "state_id": f"state:{origin_layer_id}",
                "input_range_covered": {"kind": "full"},
                "output_range_covered": {"kind": "full"},
                "downstream_visible": True,
            }
        ],
        visibility_updates=[],
        output_state_bindings=[],
        warnings=[],
    )


def test_round_report_sums_ordered_step_latency_and_energy_but_uses_placement_area():
    aggregator = ReportAggregator("m0", "SimConfig.ini", ops_total=2_000_000, ops_count_policy="mac_as_1_op")
    result = aggregator.record_round(
        _round("R0", used_xbars=4),
        _placement(area_um2=70.0),
        [
            _step("R0.S0", "Conv1", latency_s=2e-9, energy_j=4e-9, area_um2=20.0),
            _hook_step("R0.S1", "Relu1"),
            _step("R0.S2", "Conv2", latency_s=3e-9, energy_j=6e-9, area_um2=30.0),
        ],
        _commit("R0", "Conv2"),
    )
    report = aggregator.finalize()

    assert result["steps"][1]["step_id"] == "R0.S1"
    assert result["latency_s"] == 5e-9
    assert result["energy_j"] == 10e-9
    assert result["mapped_active_area_um2"] == 70.0
    assert report.inference_metrics["area_um2"] == 70.0
    assert report.inference_metrics["cumulative_scheduled_physical_xbars"] == 4
    assert report.inference_metrics["peak_scheduled_physical_xbars"] == 4
    assert report.inference_metrics["tops_per_w"] is None
    assert (
        report.inference_metrics["modeled_energy_efficiency_tops_per_w"]
        == 200.0
    )
    assert report.inference_metrics["accounting_scope"]["ddr_boundary_transfer"] == {
        "used": False,
        "energy_included": False,
        "latency_included": False,
    }
    assert report.inference_metrics["accounting_scope"][
        "cross_round_boundary_sram"
    ]["spill_to_ddr"] is False
    assert [step["step_id"] for step in report.step_results] == ["R0.S0", "R0.S1", "R0.S2"]
    assert [entry["audit_kind"] for entry in report.raw_mnsim_audit] == [
        "placement",
        "static",
        "static",
    ]


def test_structural_step_does_not_change_weighted_xbar_proof():
    record = ReportAggregator(
        "m0",
        "SimConfig.ini",
        ops_total=1,
        ops_count_policy="mac_as_1_op",
    ).record_round(
        _round("R0", used_xbars=2),
        _placement(area_um2=20.0),
        [
            _step("R0.S0", "Conv1", 1.0, 1.0, 10.0),
            _step("R0.S1", "Pool1", 2.0, 2.0, 50.0),
        ],
        _commit("R0", "Pool1"),
    )

    assert record["used_xbars"] == 2
    assert record["steps"][1]["origin_layer_ids"] == ["Pool1"]
    assert record["mapped_active_area_um2"] == 50.0


def test_round_report_keeps_fragment_proof_separate_from_area_accounting():
    record = ReportAggregator(
        "m0",
        "SimConfig.ini",
        ops_total=1,
        ops_count_policy="mac_as_1_op",
    ).record_round(
        _round("R0", used_xbars=4),
        _placement(
            area_um2=70.0,
            mapping_summary={
                "active_group_count": 2,
                "active_physical_xbar_count": 4,
            },
        ),
        [_step("R0.S0", "Conv1", latency_s=2e-9, energy_j=4e-9, area_um2=20.0)],
        _commit("R0", "Conv1"),
    )

    assert record["fragment_resource_proof"] == {
        "used_xbars": 4,
        "active_group_count": 2,
        "active_physical_xbar_count": 4,
    }
    assert record["mapped_active_area_um2"] == 70.0
    assert record["area_um2"] == 70.0


def test_phase2_package_exports_report_aggregator():
    assert phase2.ReportAggregator is ReportAggregator
    assert "ReportAggregator" in phase2.__all__


def test_report_aggregates_original_mnsim_module_energy_without_changing_total():
    aggregator = ReportAggregator(
        "m0",
        "SimConfig.ini",
        ops_total=1,
        ops_count_policy="mac_as_1_op",
    )
    aggregator.record_round(
        _round("R0", used_xbars=2),
        _placement(area_um2=20.0),
        [
            _step(
                "R0.S0",
                "Conv1",
                latency_s=2e-9,
                energy_j=4e-9,
                area_um2=20.0,
                module_energy_j={"xbar": 1e-9, "adc": 3e-9},
            )
        ],
        _commit("R0", "Conv1"),
    )

    breakdown = aggregator.finalize().inference_metrics["energy_breakdown_j"]

    assert breakdown["static_step_energy_j"] == 4e-9
    assert breakdown["mnsim_static_read_energy_j"] == 4e-9
    assert breakdown["mnsim_static_module_energy_j"] == {
        "xbar": 1e-9,
        "adc": 3e-9,
    }
    assert breakdown["mnsim_static_module_energy_total_j"] == 4e-9
    assert breakdown["mnsim_static_module_energy_residual_j"] == 0.0


def test_analog_final_accounting_subtracts_adc_but_retains_raw_audit():
    aggregator = ReportAggregator(
        "m0",
        "SimConfig.ini",
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        accounting_scope={
            "adc_conversion": {"hardware_enabled": True},
            "analog_frontend_lif": {"energy_included": True},
        },
    )
    aggregator.record_round(
        _round("R0", used_xbars=2),
        _placement(area_um2=50.0),
        [
            _step(
                "R0.S0",
                "Conv1",
                latency_s=10e-9,
                energy_j=100e-9,
                area_um2=50.0,
                module_energy_j={"xbar": 80e-9, "adc": 20e-9},
                raw_adc={
                    "raw_adc_latency_s": 2e-9,
                    "raw_adc_energy_j": 20e-9,
                    "raw_adc_power_w": 10.0,
                    "raw_adc_area_um2": 5.0,
                },
            ),
            _analog_lif_step(),
        ],
        _commit("R0", "L1"),
    )

    report = aggregator.finalize()
    metrics = report.inference_metrics
    accounting = metrics["analog_accounting"]

    assert report.raw_mnsim_audit[1]["raw_mnsim_summary"]["raw_adc_energy_j"] == 20e-9
    assert metrics["latency_s"] == pytest.approx(11e-9)
    assert metrics["energy_j"] == pytest.approx(110e-9)
    assert metrics["area_um2"] == pytest.approx(45.0)
    assert accounting["adc_subtracted_latency_s"] == pytest.approx(2e-9)
    assert accounting["adc_subtracted_energy_j"] == pytest.approx(20e-9)
    assert accounting["adc_subtracted_power_w"] == pytest.approx(10.0)
    assert accounting["adc_subtracted_area_um2"] == pytest.approx(5.0)
    assert accounting["analog_frontend_lif_latency_s"] == pytest.approx(3e-9)
    assert accounting["analog_frontend_lif_energy_j"] == pytest.approx(30e-9)
    assert accounting["reconciliation_passed"] is True
