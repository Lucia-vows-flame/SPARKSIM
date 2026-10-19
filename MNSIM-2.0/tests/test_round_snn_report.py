import MNSIM.Scheduling.phase2.executor as executor_module

from MNSIM.Scheduling.phase2.executor import execute_round_plan
from test_round_snn_executor import _Backend, _plan


def test_snn_report_separates_program_read_lif_and_excluded_ddr(monkeypatch):
    class _Ledger:
        def __init__(self, **_kwargs):
            pass

        def score_round(self, *, item_id, placement):
            return {"item_id": item_id, "latency_s": 2e-9, "energy_j": 3e-9}

    monkeypatch.setattr(executor_module, "AnnRoundPhysicalWriteLedger", _Ledger)
    report = execute_round_plan(
        _plan(),
        sim_config_path="SimConfig.ini",
        ops_total=123,
        ops_count_policy="dense_equivalent_mac_as_1_op",
        static_backend=_Backend(),
        model_inputs={"model_input": {"shape": {"C": 3}}},
        weight_codes_by_layer={"Conv1": {}},
    )

    metrics = report.inference_metrics
    record = report.item_results[0]
    lif_steps = [step for step in report.step_results if step["source_kind"] == "digital_lif"]

    assert metrics["ops_total"] == 123
    assert metrics["lif_update_count"] == 16 * 3
    assert metrics["weight_programming_latency_s"] == 2e-9
    assert metrics["weight_programming_energy_j"] == 3e-9
    assert record["weight_programming_latency_s"] == 2e-9
    assert len(lif_steps) == 3
    assert metrics["energy_breakdown_j"]["digital_lif_energy_j"] == sum(
        step["energy_j"] for step in lif_steps
    )
    assert metrics["accounting_scope"]["ddr_boundary_transfer"] == {
        "used": False,
        "energy_included": False,
        "latency_included": False,
    }
    assert all(
        step["mapping_summary"]["digital_lif"]["bit_width"] == 18
        for step in lif_steps
    )
    assert [step["timestep"] for step in lif_steps] == [0, 1, 2]
