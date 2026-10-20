import json
from pathlib import Path

from MNSIM.Interface.round_orchestrated_cli import render_terminal_result
from MNSIM.Interface.round_orchestrated_runner import (
    RunnerDependencies,
    run_round_orchestrated_simulation,
)
from MNSIM.Interface.round_orchestrated_types import RoundOrchestratedRunRequest
from MNSIM.Scheduling.phase2.types import Phase2Report
from MNSIM.Scheduling.round_types import (
    LayerIR,
    ModelIR,
    RoundPlan,
    SequenceSpec,
)


class _Network:
    def get_weights(self):
        return [{}]


class _Context:
    net = _Network()


def test_runner_persists_snn_sequence_and_terminal_summary(tmp_path):
    weights = tmp_path / "weights.pth"
    config = tmp_path / "SimConfig.ini"
    weights.write_bytes(b"test")
    config.write_text("[test]\nvalue=1\n", encoding="ascii")
    request = RoundOrchestratedRunRequest(
        network="snn-test",
        weights=weights,
        num_classes=10,
        sim_config=config,
        output_dir=tmp_path / "out",
        run_hardware=True,
        skip_accuracy=True,
    )
    sequence = SequenceSpec(3, True, True, ["L1"])
    model = ModelIR(
        model_id="snn-test",
        backend_kind="snn",
        sequence_spec=sequence,
        layers=[
            LayerIR(
                layer_id="L0",
                layer_type="fc",
                is_weighted=True,
                stable_topological_index=0,
                input_tensor_ids=["input"],
                output_tensor_ids=["weighted"],
                input_layer_ids=[],
                output_consumer_ids=["L1"],
                weight_ref_id="w0",
                shape={"Infeature": 4, "Outfeature": 8, "Inputbit": 8},
                backend_attrs={"mnsim_layer_index": 0},
            )
        ],
    )
    plan = RoundPlan(
        plan_id="plan:snn-test",
        model_id="snn-test",
        backend_kind="snn",
        hardware_budget={"xbar_size": [128, 128]},
        execution_items=[],
        layer_resource_descriptors=[],
        model_graph_ref={"model_id": "snn-test"},
        feasible=True,
        sequence_spec=sequence,
    )
    report = Phase2Report(
        model_name="snn-test",
        sim_config_path=str(config),
        feasible=True,
        execution_item_count=0,
        inference_metrics={
            "latency_s": 1e-6,
            "energy_j": 2e-6,
            "read_latency_s": 1e-6,
            "read_energy_j": 2e-6,
            "weight_programming_latency_s": 0.0,
            "weight_programming_energy_j": 0.0,
            "power_avg_w": 2.0,
            "area_um2": 10.0,
            "lif_update_count": 24,
        },
        item_results=[],
        step_results=[],
        raw_mnsim_audit=[{"audit_kind": "digital_lif", "timestep": 0}],
        boundary_events=[],
    )
    dependencies = RunnerDependencies(
        interface_factory=lambda **_kwargs: _Context(),
        model_ir_builder=lambda _net, **_kwargs: model,
        phase1_projection_builder=lambda _path: {},
        plan_builder=lambda _model, _config: plan,
        phase2_executor=lambda **_kwargs: report,
        final_rram_validator=lambda _path: None,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    round_plan_artifact = json.loads(
        (request.output_dir / "round_plan.json").read_text(encoding="utf-8")
    )
    phase2_artifact = json.loads(
        (request.output_dir / "phase2_report.json").read_text(encoding="utf-8")
    )
    assert round_plan_artifact["round_plan"]["sequence_spec"]["sequence_length"] == 3
    assert phase2_artifact["phase2_report"]["inference_metrics"]["lif_update_count"] == 24
    terminal = render_terminal_result(result)
    assert "Sequence length T: 3" in terminal
    assert "LIF update count: 24" in terminal
