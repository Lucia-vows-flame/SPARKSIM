from pathlib import Path

import pytest

import MNSIM.Interface.round_orchestrated_cli as cli
from MNSIM.Interface.round_orchestrated_types import (
    RoundOrchestratedRunRequest,
    RoundOrchestratedRunResult,
    StageStatus,
)
from MNSIM.Scheduling.phase2.types import Phase2Report
from MNSIM.Scheduling.round_types import (
    LocalGraphNode,
    PrepassRef,
    ResourceProof,
    RoundPlan,
    SelectedEntry,
    WeightedPhysicalRound,
)


def _required_args(*, network="vgg8", num_classes="10"):
    return [
        "--network",
        network,
        "--weights",
        "checkpoints/model.pth",
        "--num-classes",
        num_classes,
        "--sim-config",
        "configs/SimConfig.ini",
        "--output-dir",
        "results/run",
    ]


def _request(**changes):
    values = {
        "network": "vgg8",
        "weights": Path("checkpoints/model.pth"),
        "num_classes": 10,
        "sim_config": Path("configs/SimConfig.ini"),
        "output_dir": Path("results/run"),
        "dataset_module": "datasets.fake",
    }
    values.update(changes)
    return RoundOrchestratedRunRequest(**values)


def _weighted_item(*, item_id, item_index, layer_id, layer_type):
    prepass_ref = PrepassRef(
        ref_id="prepass:" + layer_id,
        model_id="vgg8",
        layer_id=layer_id,
        layer_type=layer_type,
        resource_version="test",
        sim_config_fingerprint="sim",
        layer_config_fingerprint="layer",
        weight_ref_id="vgg8:" + layer_id + ":weight",
        shape_signature={},
    )
    selected_entry = SelectedEntry(
        layer_id=layer_id,
        layer_type=layer_type,
        xbar_range_start=0,
        xbar_range_end=2,
        used_xbars=2,
        layer_complete_in_round=True,
        prepass_ref=prepass_ref,
    )
    return WeightedPhysicalRound(
        item_id=item_id,
        item_index=item_index,
        round_kind="single_layer_split",
        selected_entries=[selected_entry],
        local_graph_nodes=[
            LocalGraphNode(
                layer_id=layer_id,
                layer_type=layer_type,
                node_role="weighted",
            )
        ],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(xbar_nums=8, used_xbars=2),
    )


def _weighted_plan():
    item = _weighted_item(
        item_id="R000",
        item_index=0,
        layer_id="L000_conv",
        layer_type="conv",
    )
    return RoundPlan(
        plan_id="plan:vgg8",
        model_id="vgg8",
        backend_kind="ann",
        hardware_budget={"xbar_nums": 8},
        execution_items=[item],
        layer_resource_descriptors=[],
        model_graph_ref={},
        feasible=True,
    )


def _successful_hardware_result(*, request=None):
    request = request or _request(run_hardware=True)
    plan = _weighted_plan()
    report = Phase2Report(
        model_name="vgg8",
        sim_config_path="configs/SimConfig.ini",
        feasible=True,
        execution_item_count=1,
        inference_metrics={},
        item_results=[
            {
                "item_id": "R000",
                "item_kind": "weighted_physical_round",
                "used_xbars": 2,
                "latency_s": 0.0012,
                "energy_j": 0.000003,
                "power_avg_w": 0.0017,
                "mapped_active_area_um2": 2500000.0,
                "steps": [{"step_id": "R000.S0"}],
            }
        ],
        step_results=[],
        raw_mnsim_audit=[{"backend_noise": "RAW_AUDIT_MUST_NOT_APPEAR"}],
        boundary_events=[],
    )
    return RoundOrchestratedRunResult(
        status="succeeded",
        exit_code=0,
        request=request,
        stage_statuses={
            "accuracy": StageStatus(name="accuracy", status="succeeded"),
            "hardware": StageStatus(name="hardware", status="succeeded"),
        },
        accuracy_report={
            "Original accuracy": 0.912345,
            "PIM-based computing accuracy": 0.876543,
        },
        round_plan=plan,
        phase2_report=report,
        hardware_summary={
            "latency_s": 0.0012,
            "energy_j": 0.000003,
            "power_avg_w": 0.0025,
            "area_um2": 2500000.0,
            "macs_total": 123456,
            "network_workload_top_per_inference": 0.000000246912,
            "compute_tmacs_per_s": 0.000123456,
            "compute_tops": 0.000246912,
            "compute_throughput_tops": 0.000246912,
            "tmacs_per_s": 0.00010288,
            "tops": 0.00020576,
            "end_to_end_throughput_tops": 0.00020576,
            "modeled_energy_efficiency_tops_per_w": 82.304,
            "tops_per_w": 82.304,
        },
    )


def test_request_from_args_accepts_unlisted_network_and_hardware_only_paths():
    request = cli.request_from_args(
        _required_args(network="newly_registered_ann")
        + ["--run-hardware", "--skip-accuracy"]
    )

    assert request.network == "newly_registered_ann"
    assert request.weights == Path("checkpoints/model.pth")
    assert request.sim_config == Path("configs/SimConfig.ini")
    assert request.output_dir == Path("results/run")
    assert request.num_classes == 10
    assert request.requested_stages() == ("hardware",)


def test_request_from_args_accepts_snn_aware_policy_and_profile():
    request = cli.request_from_args(
        _required_args()
        + [
            "--run-hardware",
            "--skip-accuracy",
            "--schedule-policy",
            "snn_flashsef_ptr",
            "--snn-activity-profile",
            "profiles/calibration.json",
        ]
    )

    assert request.schedule_policy == "snn_flashsef_ptr"
    assert request.snn_activity_profile == Path("profiles/calibration.json")


def test_parser_defaults_stage_combinations_and_all_controls():
    default_request = cli.request_from_args(_required_args())
    both_request = cli.request_from_args(_required_args() + ["--run-hardware"])
    hardware_only_request = cli.request_from_args(
        _required_args() + ["--run-hardware", "--skip-accuracy"]
    )
    parsed = cli._parse_args(
        _required_args()
        + [
            "--dataset-module",
            "datasets.custom",
            "--run-hardware",
            "--skip-accuracy",
            "--overwrite",
            "--disable-saf",
            "--enable-variation",
            "--enable-r-ratio",
            "--accuracy-seed",
            "20260809",
            "--fixed-quantization-range",
            "--device",
            "cuda:7",
            "--disable-inner-pipeline",
            "--show-rounds",
        ]
    )
    controlled_request = cli.request_from_args(
        _required_args()
        + [
            "--dataset-module",
            "datasets.custom",
            "--run-hardware",
            "--skip-accuracy",
            "--overwrite",
            "--disable-saf",
            "--enable-variation",
            "--enable-r-ratio",
            "--accuracy-seed",
            "20260809",
            "--fixed-quantization-range",
            "--device",
            "cuda:7",
            "--disable-inner-pipeline",
            "--show-rounds",
        ]
    )

    assert default_request.requested_stages() == ("accuracy",)
    assert default_request.run_hardware is False
    assert default_request.skip_accuracy is False
    assert default_request.enable_saf is True
    assert default_request.device == "0"
    assert both_request.requested_stages() == ("accuracy", "hardware")
    assert hardware_only_request.requested_stages() == ("hardware",)
    assert parsed.show_rounds is True
    assert not hasattr(controlled_request, "show_rounds")
    assert controlled_request.dataset_module == "datasets.custom"
    assert controlled_request.overwrite is True
    assert controlled_request.enable_saf is False
    assert controlled_request.enable_variation is True
    assert controlled_request.enable_r_ratio is True
    assert controlled_request.accuracy_seed == 20260809
    assert controlled_request.fixed_quantization_range is True
    assert controlled_request.device == "cuda:7"
    assert controlled_request.disable_inner_pipeline is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--network", "vgg8"],
        _required_args(num_classes="not-an-integer"),
    ],
)
def test_request_from_args_uses_argparse_errors_without_invoking_runner(
    argv, monkeypatch
):
    invoked = False

    def fake_runner(request):
        nonlocal invoked
        invoked = True
        raise AssertionError("parser errors must not invoke the runner")

    monkeypatch.setattr(cli, "run_round_orchestrated_simulation", fake_runner)

    with pytest.raises(SystemExit) as error:
        cli.request_from_args(argv)

    assert error.value.code == 2
    assert invoked is False


def test_no_stage_selection_reaches_runner_and_main_returns_structured_exit_code(
    monkeypatch, capsys
):
    observed_requests = []
    fake_result = None

    def fake_runner(request):
        nonlocal fake_result
        observed_requests.append(request)
        print("fake backend noise")
        fake_result = RoundOrchestratedRunResult(
            status="failed",
            exit_code=2,
            request=request,
            stage_statuses={},
        )
        return fake_result

    monkeypatch.setattr(cli, "run_round_orchestrated_simulation", fake_runner)

    exit_code = cli.main(_required_args() + ["--skip-accuracy"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert len(observed_requests) == 1
    assert observed_requests[0].requested_stages() == ()
    assert captured.err == ""
    assert fake_result is not None
    assert captured.out == cli.render_terminal_result(fake_result) + "\n"
    assert "fake backend noise" not in captured.out


def test_renderer_always_renders_hardware_summary_and_optionally_round_details():
    result = _successful_hardware_result()

    rendered = cli.render_terminal_result(result)

    assert "Round-Orchestrated MNSIM" in rendered
    assert "Network: vgg8" in rendered
    assert "Checkpoint: checkpoints/model.pth" in rendered
    assert "SimConfig: configs/SimConfig.ini" in rendered
    assert "Stages: accuracy=succeeded hardware=succeeded" in rendered
    assert "Accuracy" in rendered
    assert "  Original accuracy: 0.912345" in rendered
    assert "  PIM-based computing accuracy: 0.876543" in rendered
    assert "Hardware Summary" in rendered
    assert "  Latency: 1.2 ms" in rendered
    assert "  Modeled energy: 3 uJ" in rendered
    assert "  Average power: 2.5 mW" in rendered
    assert "  Area: 2.5 mm2" in rendered
    assert "  MACs: 123456" in rendered
    assert "  Network workload: 2.46912e-07 TOP/inference" in rendered
    assert "  Compute TMAC/s (excludes weight programming): 0.000123456" in rendered
    assert "  Compute throughput (excludes weight programming): 0.000246912 TOPS" in rendered
    assert "  End-to-end TMAC/s: 0.00010288" in rendered
    assert "  End-to-end throughput: 0.00020576 TOPS" in rendered
    assert "  Modeled-energy efficiency (partial accounting): 82.304 TOPS/W" in rendered
    assert "  End-to-end TOPS/W: 82.304" in rendered
    assert "  Physical round count: 1" in rendered
    assert "Artifacts: results/run" in rendered
    assert "Status: SUCCEEDED" in rendered
    assert "Round Details" not in rendered
    assert "R000 |" not in rendered

    result.phase2_report.item_results[0].pop("used_xbars")
    with_rounds = cli.render_terminal_result(result, show_rounds=True)

    round_lines = [
        line
        for line in with_rounds.splitlines()
        if line.startswith("  Round |") or line.startswith("  R000 |")
    ]

    assert "Round Details" in with_rounds
    assert round_lines[0] == (
        "  Round | Xbars used / capacity | Utilization | Mapped area | Latency | "
        "Energy | Average power"
    )
    assert round_lines[1] == "  R000 | 2 / 8 | 25% | 2.5 mm2 | 1.2 ms | 3 uJ | 1.7 mW"
    assert "L000_conv" not in with_rounds
    assert "weighted_physical_round" not in with_rounds
    assert "R000.S0" not in with_rounds
    assert "RAW_AUDIT_MUST_NOT_APPEAR" not in with_rounds


def test_renderer_uses_full_snn_accuracy_path_names():
    result = RoundOrchestratedRunResult(
        status="succeeded",
        exit_code=0,
        request=_request(network="standard_vgg8_snn_v7_phase6"),
        stage_statuses={
            "accuracy": StageStatus(name="accuracy", status="succeeded"),
        },
        accuracy_report={
            "SNN Original Accuracy": 0.9935,
            "SNN Processing-in-Memory Accuracy": None,
        },
    )

    rendered = cli.render_terminal_result(result)

    assert "  SNN Original Accuracy: 0.9935" in rendered
    assert "  SNN Processing-in-Memory Accuracy: n/a" in rendered


def test_snn_renderer_reports_the_simconfig_selected_accuracy_profile():
    request = _request(network="standard_vgg8_snn_v7_phase6")
    result = RoundOrchestratedRunResult(
        status="succeeded",
        exit_code=0,
        request=request,
        stage_statuses={
            "accuracy": StageStatus(name="accuracy", status="succeeded")
        },
        accuracy_report={
            "SNN Original Accuracy": 0.75,
            "SNN Processing-in-Memory Accuracy": 0.5,
            "hardware_profile": "adc_digital_lif",
            "accuracy_adc_model": "READC_LLOYD_MAX",
            "accuracy_lif_perturbation_profile": None,
        },
    )

    rendered = cli.render_terminal_result(result)

    assert "Hardware profile: adc_digital_lif" in rendered
    assert "ADC numerical model: READC_LLOYD_MAX" in rendered
    assert "LIF perturbation profile: n/a" in rendered


@pytest.mark.parametrize(
    "option",
    [
        "--snn-hardware-profile",
        "--readc-artifact",
        "--lif-perturbation-artifact",
    ],
)
def test_cli_rejects_profile_and_artifact_overrides(option):
    with pytest.raises(SystemExit) as error:
        cli.request_from_args(_required_args() + [option, "forbidden"])
    assert error.value.code == 2


@pytest.mark.parametrize("hardware_summary", [None, "malformed"])
def test_renderer_successful_hardware_stage_renders_empty_summary_and_round_details(
    hardware_summary,
):
    result = _successful_hardware_result()
    result.hardware_summary = hardware_summary

    rendered = cli.render_terminal_result(result, show_rounds=True)

    assert "Hardware Summary" in rendered
    for field in (
        "Latency",
        "Modeled energy",
        "Average power",
        "Area",
        "MACs",
        "Compute TMAC/s (excludes weight programming)",
        "Network workload",
        "Compute throughput (excludes weight programming)",
        "End-to-end TMAC/s",
        "End-to-end throughput",
        "Modeled-energy efficiency (partial accounting)",
        "End-to-end TOPS/W",
    ):
        assert f"  {field}: n/a" in rendered
    assert "  Physical round count: 1" in rendered
    assert "Round Details" in rendered
    assert "  R000 |" in rendered


def test_renderer_show_rounds_rejects_non_string_item_ids_without_repr_leakage():
    result = _successful_hardware_result()
    malformed_item_id = ResourceProof(xbar_nums=8, used_xbars=2)
    result.round_plan.execution_items = [
        {
            "item_id": malformed_item_id,
            "resource_proof": {"xbar_nums": 8, "used_xbars": 2},
        }
    ]
    result.phase2_report.item_results[0]["item_id"] = malformed_item_id

    rendered = cli.render_terminal_result(result, show_rounds=True)
    detail_rows = [
        line
        for line in rendered.splitlines()
        if line.startswith("  ") and " | " in line
    ]

    assert detail_rows[1].startswith("  n/a |")
    assert "ResourceProof(" not in rendered


@pytest.mark.parametrize(
    ("quantity", "value", "expected"),
    [
        ("time", 2.0, "2 s"),
        ("time", 0.002, "2 ms"),
        ("time", 0.000002, "2 us"),
        ("time", 0.000000002, "2 ns"),
        ("energy", 2.0, "2 J"),
        ("energy", 0.002, "2 mJ"),
        ("energy", 0.000002, "2 uJ"),
        ("energy", 0.000000002, "2 nJ"),
        ("power", 2.0, "2 W"),
        ("power", 0.002, "2 mW"),
        ("power", 0.000002, "2 uW"),
        ("area", 2000000.0, "2 mm2"),
        ("area", 2.0, "2 um2"),
    ],
)
def test_format_scaled_uses_each_advertised_unit(quantity, value, expected):
    assert cli._format_scaled(value, quantity) == expected


def test_renderer_show_rounds_uses_plan_budget_for_missing_capacity_and_zero_xbars():
    result = _successful_hardware_result()
    result.round_plan.execution_items = [
        {
            "item_id": "R001",
            "resource_proof": {"used_xbars": 3},
        },
        {
            "item_id": "Z001",
            "resource_proof": {"used_xbars": 0},
        },
        {
            "item_id": "Z000",
            "resource_proof": {"used_xbars": 0, "xbar_nums": 0},
        },
    ]
    result.round_plan.hardware_budget = {"xbar_nums": 8}
    result.phase2_report.execution_item_count = 3
    result.phase2_report.item_results = [
        {
            "item_id": "R001",
            "latency_s": 0.002,
            "energy_j": 0.000004,
            "area_um2": 2.0,
            "steps": [{"step_id": "R001.S0"}],
        },
        {
            "item_id": "Z001",
            "latency_s": 0.000001,
            "energy_j": 0.000001,
            "mapped_active_area_um2": 1.0,
            "steps": [{"step_id": "Z001.S0"}],
        },
        {
            "item_id": "Z000",
            "latency_s": 0.0,
            "energy_j": 0.000001,
            "mapped_active_area_um2": float("nan"),
            "steps": [{"step_id": "Z000.S0"}],
        },
    ]

    rendered = cli.render_terminal_result(result, show_rounds=True)

    assert "R001 | 3 / 8 | 37.5% | 2 um2 | 2 ms | 4 uJ | 2 mW" in rendered
    assert "Z001 | 0 / 8 | 0% | 1 um2 | 1 us | 1 uJ | 1 W" in rendered
    assert "Z000 | 0 / n/a | n/a | n/a | 0 ns | 1 uJ | n/a" in rendered
    assert "R001.S0" not in rendered
    assert "Z001.S0" not in rendered
    assert "Z000.S0" not in rendered


@pytest.mark.parametrize(
    "item_capacity",
    [
        pytest.param("bad", id="string"),
        pytest.param(False, id="boolean"),
        pytest.param(None, id="none"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(1.5, id="non_integer"),
    ],
)
def test_renderer_show_rounds_rejects_present_malformed_item_capacity(item_capacity):
    result = _successful_hardware_result()
    result.round_plan.execution_items = [
        {
            "item_id": "R000",
            "resource_proof": {"xbar_nums": item_capacity, "used_xbars": 0},
        }
    ]
    result.round_plan.hardware_budget = {"xbar_nums": 8}
    result.phase2_report.item_results[0]["used_xbars"] = 0

    rendered = cli.render_terminal_result(result, show_rounds=True)

    assert "  R000 | 0 / n/a | n/a |" in rendered
    assert "  R000 | 0 / 8 | 0%" not in rendered


def test_renderer_show_rounds_preserves_report_order_without_layer_details():
    result = _successful_hardware_result()
    result.round_plan.execution_items.append(
        _weighted_item(
            item_id="R001",
            item_index=1,
            layer_id="FC1",
            layer_type="fc",
        )
    )
    result.phase2_report.execution_item_count = 2
    result.phase2_report.item_results = [
        {
            "item_id": "R001",
            "item_kind": "weighted_physical_round",
            "used_xbars": 2,
            "latency_s": 0.002,
            "energy_j": 0.000002,
            "power_avg_w": 0.0013,
            "mapped_active_area_um2": 2.0,
            "steps": [{"step_id": "R001.S0"}],
        },
        {
            "item_id": "R000",
            "item_kind": "weighted_physical_round",
            "used_xbars": 2,
            "latency_s": 0.001,
            "energy_j": 0.000001,
            "power_avg_w": 0.0007,
            "mapped_active_area_um2": 1.0,
            "steps": [{"step_id": "R000.S0"}],
        },
    ]

    rendered = cli.render_terminal_result(result, show_rounds=True)
    round_rows = [line for line in rendered.splitlines() if line.startswith("  R00")]

    assert round_rows == [
        "  R001 | 2 / 8 | 25% | 2 um2 | 2 ms | 2 uJ | 1.3 mW",
        "  R000 | 2 / 8 | 25% | 1 um2 | 1 ms | 1 uJ | 700 uW",
    ]
    assert "FC1" not in rendered


@pytest.mark.parametrize(
    ("result", "present_block", "absent_block", "stage_line"),
    [
        (
            RoundOrchestratedRunResult(
                status="failed",
                exit_code=1,
                request=_request(run_hardware=True),
                stage_statuses={
                    "accuracy": StageStatus(name="accuracy", status="failed"),
                    "hardware": StageStatus(name="hardware", status="not_run"),
                },
            ),
            None,
            "Accuracy\n",
            "Stages: accuracy=failed hardware=not_run",
        ),
        (
            RoundOrchestratedRunResult(
                status="succeeded",
                exit_code=0,
                request=_request(),
                stage_statuses={
                    "accuracy": StageStatus(name="accuracy", status="succeeded"),
                },
                accuracy_report={
                    "Original accuracy": 0.9,
                    "PIM-based computing accuracy": 0.8,
                },
            ),
            "Accuracy\n",
            "Hardware Summary\n",
            "Stages: accuracy=succeeded hardware=not_requested",
        ),
        (
            RoundOrchestratedRunResult(
                status="succeeded",
                exit_code=0,
                request=_request(run_hardware=True, skip_accuracy=True),
                stage_statuses={
                    "hardware": StageStatus(name="hardware", status="succeeded"),
                },
                hardware_summary={
                    "latency_s": 1.0,
                    "energy_j": 1.0,
                    "power_w": 1.0,
                    "area_um2": 1.0,
                },
            ),
            "Hardware Summary\n",
            "Accuracy\n",
            "Stages: accuracy=not_requested hardware=succeeded",
        ),
    ],
)
def test_renderer_omits_unavailable_stage_blocks(
    result, present_block, absent_block, stage_line
):
    rendered = cli.render_terminal_result(result)

    assert stage_line in rendered
    assert "Artifacts: results/run" in rendered
    assert "Status: " + result.status.upper() in rendered
    if present_block is not None:
        assert present_block in rendered
    assert absent_block not in rendered


def test_main_prints_only_rendered_result_and_returns_runner_exit_code(monkeypatch, capsys):
    request = _request(run_hardware=True)
    fake_result = _successful_hardware_result(request=request)
    fake_result.exit_code = 7
    fake_result.status = "failed"
    observed_requests = []

    def fake_runner(received_request):
        observed_requests.append(received_request)
        print("core backend noise")
        return fake_result

    monkeypatch.setattr(cli, "run_round_orchestrated_simulation", fake_runner)

    exit_code = cli.main(_required_args() + ["--dataset-module", "datasets.fake", "--run-hardware"])

    captured = capsys.readouterr()
    assert exit_code == 7
    assert observed_requests == [request]
    assert captured.err == ""
    assert captured.out == cli.render_terminal_result(fake_result) + "\n"
    assert "core backend noise" not in captured.out
