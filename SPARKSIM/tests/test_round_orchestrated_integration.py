import configparser
from pathlib import Path

import pytest
import torch

from MNSIM.Interface.network import get_net
from MNSIM.Interface.round_orchestrated_runner import (
    RunnerDependencies,
    run_round_orchestrated_simulation,
)
from MNSIM.Interface.round_orchestrated_types import RoundOrchestratedRunRequest
from MNSIM.Scheduling.phase1.config import build_hardware_projection
from test_round_phase2_real_networks import EchoStaticBackend


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_sim_config(tmp_path, *, tile_num, pe_num, group_num=2):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(_REPO_ROOT / "SimConfig.ini", encoding="UTF-8")
    parser.set("Crossbar level", "Xbar_Size", "128,128")
    parser.set("Crossbar level", "Subarray_Size", "128")
    parser.set("Architecture level", "Tile_Num", f"{tile_num},{tile_num}")
    parser.set("Tile level", "PE_Num", f"{pe_num},{pe_num}")
    parser.set("Process element level", "Group_Num", str(group_num))
    parser.set("Process element level", "Xbar_Polarity", "2")
    parser.set("Process element level", "DAC_Num", "128")
    parser.set("Process element level", "ADC_Num", "128")
    parser.set("Algorithm Configuration", "Weight_precision", "8")
    path = tmp_path / "SimConfig.round-orchestrated.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return path


def _write_model_checkpoint(tmp_path, network):
    model = get_net(
        hardware_config={
            "xbar_size": 128,
            "input_bit": 1,
            "weight_bit": 1,
            "ADC_quantize_bit": 6,
            "DAC_num": 128,
            "xbar_polarity": 2,
        },
        cate=network,
        num_classes=10,
    )
    path = tmp_path / f"{network}.pth"
    torch.save(model.state_dict(), path)
    return path


def _hardware_only_request(tmp_path, *, network, sim_config):
    return RoundOrchestratedRunRequest(
        network=network,
        weights=_write_model_checkpoint(tmp_path, network),
        num_classes=10,
        sim_config=sim_config,
        output_dir=tmp_path / "output",
        dataset_module=None,
        skip_accuracy=True,
        run_hardware=True,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("network", "expected_execution_item_count"),
    [("vgg8", 1), ("resnet18", 1)],
)
def test_g8_runner_uses_default_original_mnsim_backend(
    tmp_path,
    network,
    expected_execution_item_count,
):
    sim_config = _write_sim_config(tmp_path, tile_num=6, pe_num=4)
    assert build_hardware_projection(str(sim_config)).xbar_nums == 2304
    request = _hardware_only_request(
        tmp_path,
        network=network,
        sim_config=sim_config,
    )

    result = run_round_orchestrated_simulation(request)

    assert result.exit_code == 0
    assert result.round_plan is not None
    assert result.phase2_report is not None
    assert result.phase2_report.execution_item_count == expected_execution_item_count
    assert result.hardware_summary["ops_count_policy"] == "mac_as_2_ops"
    assert result.hardware_summary["latency_s"] > 0
    assert result.hardware_summary["energy_j"] > 0
    metrics = result.phase2_report.inference_metrics
    assert metrics["static_energy_j"] > 0.0
    assert metrics["energy_j"] == pytest.approx(
        metrics["read_energy_j"]
        + metrics["weight_programming_energy_j"]
        + metrics["static_energy_j"]
    )
    assert metrics["static_accounting_complete"] is False
    static_audit = metrics["static_power_audit"]
    inventory = static_audit["inventory"]
    assert inventory["physical_tile_count"] == 36
    assert inventory["physical_pe_count"] == 576
    assert inventory["physical_group_count"] == 1152
    assert inventory["physical_xbar_count"] == 2304
    assert inventory["blocks"]
    assert "pe_input_sram" in static_audit["modeled_blocks"]
    assert {
        "rram_array_read_ready",
        "adc_standby",
        "dac_standby",
        "crossbar_cmos_periphery",
        "pe_tile_digital_logic",
    } <= set(static_audit["unmodeled_blocks"])
    assert "tile_state_output_sram" not in static_audit["unmodeled_blocks"]
    assert "boundary_state_sram" in static_audit["excluded_blocks"]
    assert static_audit["cross_round_storage"] == {
        "backend": "shared_on_chip_sram",
        "observed": False,
        "capacity_included": False,
        "dynamic_energy_included": False,
        "latency_included": False,
        "partial_state_access_owner": "scheduled_partial_accumulator",
        "spill_policy": "no_ddr_fail_if_capacity_is_insufficient",
    }
    assert static_audit["external_memory_boundary"] == {
        "backend": "ddr",
        "observed": False,
        "energy_included": False,
        "latency_included": False,
    }
    assert "global_controller" in static_audit["excluded_blocks"]
    if network == "vgg8":
        assert "global_accumulator_sram" in static_audit["excluded_blocks"]
    else:
        assert "global_accumulator_sram" in static_audit["modeled_blocks"]
    assert (request.output_dir / "round_plan.json").is_file()
    assert (request.output_dir / "phase2_report.json").is_file()


@pytest.mark.parametrize(
    ("network", "expected_weighted_item_count"),
    [("vgg8", 597), ("resnet18", 879)],
)
def test_two_xbar_runner_preserves_continuation_boundaries(
    tmp_path,
    network,
    expected_weighted_item_count,
):
    sim_config = _write_sim_config(tmp_path, tile_num=1, pe_num=1, group_num=1)
    assert build_hardware_projection(str(sim_config)).xbar_nums == 2
    request = _hardware_only_request(
        tmp_path,
        network=network,
        sim_config=sim_config,
    )

    result = run_round_orchestrated_simulation(
        request,
        dependencies=RunnerDependencies(static_backend=EchoStaticBackend()),
    )

    assert result.exit_code == 0
    assert result.round_plan is not None
    assert result.phase2_report is not None
    weighted_items = [
        item
        for item in result.round_plan.execution_items
        if item.item_kind == "weighted_physical_round"
    ]
    assert len(weighted_items) == expected_weighted_item_count
    assert any(item.round_kind == "single_layer_split" for item in weighted_items)
    assert any(item.round_kind == "single_layer_continuation" for item in weighted_items)
    completion_statuses = {
        update["completion_status"]
        for record in result.phase2_report.item_results
        for update in record["boundary_report"]["completion_updates"]
    }
    assert {"partial_sum", "complete_output"} <= completion_statuses
    static_audit = result.phase2_report.inference_metrics["static_power_audit"]
    assert "boundary_state_sram" in static_audit["modeled_blocks"]
    assert static_audit["cross_round_storage"]["observed"] is True
    assert static_audit["external_memory_boundary"]["observed"] is False
    liveness_trace = static_audit["capacity_audit"][
        "boundary_sram_liveness_trace"
    ]
    assert liveness_trace[-1]["live_bits_after"] == 0
    assert liveness_trace[-1]["live_state_count_after"] == 0
