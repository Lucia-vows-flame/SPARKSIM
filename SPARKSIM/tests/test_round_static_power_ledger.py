import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from MNSIM.Scheduling.phase2.static_power_ledger import load_static_power_ledger
from MNSIM.Scheduling.phase2.report import ReportAggregator
from MNSIM.Scheduling.phase2.types import (
    BoundaryCommitResult,
    MetricTotals,
    Phase2Invalid,
    StepExecutionResult,
)
from MNSIM.Scheduling.round_types import ResourceProof


PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "static_power"
    / "cacti_7_0_2kib_bank_stage_a.json"
)


def _config(
    tmp_path,
    *,
    include_legacy_leakage=False,
    profile_value=None,
    profile_base=None,
):
    config_path = tmp_path / "SimConfig.ini"
    config_path.write_text(
        "\n".join(
            [
                "[Architecture level]",
                "Tile_Num = 2,1",
                "Buffer_AccessEnergyPerBit = 0.000001",
                "Buffer_Bitwidth = 128",
                "Buffer_AccessLatency = 1.84",
                "Buffer_IncludeLeakageEnergy = "
                + ("1" if include_legacy_leakage else "0"),
                "",
                "[Tile level]",
                "PE_Num = 2,2",
                "Tile_outBuf_Size = 0",
                "",
                "[Crossbar level]",
                "Xbar_Size = 128,128",
                "Subarray_Size = 128",
                "",
                "[Process element level]",
                "Group_Num = 2",
                "Xbar_Polarity = 2",
                "ADC_Num = 32",
                "DAC_Num = 128",
                "PE_inBuf_Size = 0",
                "",
                "[Static Power]",
                "Static_Power_Enable = 1",
                f"Static_Power_Profile = {profile_value or PROFILE_PATH}",
                *(
                    [f"Static_Power_Profile_Base = {profile_base}"]
                    if profile_base is not None
                    else []
                ),
                "Static_Power_Policy = no_implicit_power_gating",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _placement(*, pe_bytes=1024.0, tile_bytes=3072.0):
    return SimpleNamespace(
        mapping_summary={
            "tcg_buffer_capacity_evidence": [
                {
                    "layer_id": 0,
                    "layer_type": "conv",
                    "pe_input_working_set_bytes_per_instance": pe_bytes,
                    "tile_output_working_set_bytes_per_instance": tile_bytes,
                    "tcg_pe_input_rounded_kb": 1.0,
                    "tcg_tile_output_rounded_kb": 4.0,
                    "source_rounding_rule": "tcg_next_power_of_two_bytes",
                }
            ],
            "tcg_pe_input_rounded_capacity_bits_per_instance": 8192,
            "tcg_tile_output_rounded_capacity_bits_per_instance": 32768,
        }
    )


def _commit(*states, updated_states=(), events=()):
    return SimpleNamespace(
        created_states=list(states),
        updated_states=list(updated_states),
        events=list(events),
    )


def _boundary_state(
    state_id,
    *,
    consumers,
    elements,
    bit_width=8,
    payload_kind="activation_sequence",
    status="complete_output",
    producer_layer_id=None,
    sample_id="sample:0",
    timestep=0,
):
    return SimpleNamespace(
        state_id=state_id,
        producer_layer_id=producer_layer_id or f"producer:{state_id}",
        storage_backend="boundary_sram",
        allowed_consumer_item_ids=list(consumers),
        shape={"C": elements},
        data_bitwidth=bit_width,
        payload_kind=payload_kind,
        status=status,
        sample_id=sample_id,
        timestep=timestep,
    )


def test_stage_a_integrates_fixed_cacti_banks_over_complete_round_time(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    assert ledger is not None
    ledger.observe_round(
        item_id="R0",
        placement=_placement(),
        step_results=[],
        boundary_commit_result=_commit(),
    )

    result = ledger.finalize(
        [
            {"item_id": "R0", "latency_s": 2.0},
            {"item_id": "R1", "latency_s": 1.0},
        ]
    )

    bank_power_w = 5.13732e-8 + 4.34823e-8
    # 8 PE banks plus 2 banks for each of 2 tile SRAM instances.
    expected_power_w = 12 * bank_power_w
    assert result["static_average_power_w"] == pytest.approx(expected_power_w)
    assert result["static_energy_j"] == pytest.approx(expected_power_w * 3.0)
    assert result["static_energy_by_round_j"] == pytest.approx(
        {"R0": expected_power_w * 2.0, "R1": expected_power_w}
    )

    audit = result["static_power_audit"]
    assert audit["inventory"]["physical_tile_count"] == 2
    assert audit["inventory"]["physical_pe_count"] == 8
    assert audit["inventory"]["physical_group_count"] == 16
    assert audit["inventory"]["physical_xbar_count"] == 32
    assert audit["inventory"]["physical_adc_count"] == 512
    assert audit["inventory"]["physical_dac_count"] == 2048
    inventory_blocks = {
        entry["block_class"]: entry
        for entry in audit["inventory"]["blocks"]
    }
    assert inventory_blocks["adc_standby"] == {
        "block_class": "adc_standby",
        "instance_count": 512,
        "capacity_bits_per_instance": None,
        "inventory_source": "sim_config_physical_converter_topology",
        "characterization_status": "unmodeled",
        "power_state": "idle_or_read_ready_not_characterized",
        "operation_energy_boundary": "unknown",
    }
    assert inventory_blocks["dac_standby"]["instance_count"] == 2048
    assert inventory_blocks["dac_standby"]["characterization_status"] == "unmodeled"
    pe = audit["capacity_audit"]["blocks"]["pe_input_sram"]
    tile = audit["capacity_audit"]["blocks"]["tile_state_output_sram"]
    assert pe["peak_required_capacity_bits"] == 8192
    assert pe["materialized_capacity_bits"] == 16384
    assert pe["rounding_residual_bits"] == 8192
    assert pe["banks_per_instance"] == 1
    assert tile["peak_required_capacity_bits"] == 24576
    assert tile["materialized_capacity_bits"] == 32768
    assert tile["rounding_residual_bits"] == 8192
    assert tile["banks_per_instance"] == 2
    assert audit["static_accounting_complete"] is False
    assert "adc_standby" in audit["unmodeled_blocks"]
    assert "global_controller" in audit["excluded_blocks"]


def test_cross_round_state_uses_shared_boundary_sram(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    state = _boundary_state(
        "state:A",
        consumers=["R1"],
        elements=4096,
    )
    ledger.observe_round(
        item_id="R0",
        placement=_placement(),
        step_results=[],
        boundary_commit_result=_commit(state),
    )

    result = ledger.finalize([{"item_id": "R0", "latency_s": 1.0}])
    audit = result["static_power_audit"]
    tile = audit["capacity_audit"]["blocks"]["tile_state_output_sram"]
    boundary = audit["capacity_audit"]["blocks"]["boundary_state_sram"]

    assert tile["modeled"] is True
    assert tile["capacity_complete"] is True
    assert boundary["modeled"] is True
    assert boundary["peak_required_capacity_bits"] == 4096 * 8
    assert boundary["banks_per_instance"] == 2
    assert "tile_state_output_sram" not in audit["unmodeled_blocks"]
    assert result["static_energy_by_block_j"].keys() == {
        "pe_input_sram",
        "tile_state_output_sram",
        "boundary_state_sram",
    }
    assert result["boundary_sram_access_energy_j"] == pytest.approx(
        4096 * 8 * 1e-15
    )
    assert audit["cross_round_storage"] == {
        "backend": "shared_on_chip_sram",
        "observed": True,
        "capacity_included": True,
        "dynamic_energy_included": True,
        "latency_included": True,
        "partial_state_access_owner": "scheduled_partial_accumulator",
        "spill_policy": "no_ddr_fail_if_capacity_is_insufficient",
    }
    assert audit["external_memory_boundary"] == {
        "backend": "ddr",
        "observed": False,
        "energy_included": False,
        "latency_included": False,
    }


def test_partial_accumulator_uses_boundary_sram_without_dynamic_double_count(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    partial_step = SimpleNamespace(
        step_id="R0.S1",
        mapping_summary={
            "scheduled_partial_accumulator": {
                "state_read_bytes": 8192,
                "state_write_bytes": 16384,
            }
        },
        layer_results=[],
    )
    ledger.observe_round(
        item_id="R0",
        placement=_placement(tile_bytes=1024.0),
        step_results=[partial_step],
        boundary_commit_result=_commit(
            _boundary_state(
                "state:partial",
                consumers=["R1"],
                elements=16384,
                bit_width=8,
                payload_kind="partial_sum_sequence",
                status="partial_sum",
            )
        ),
    )

    result = ledger.finalize([{"item_id": "R0", "latency_s": 1.0}])
    audit = result["static_power_audit"]
    tile = audit["capacity_audit"]["blocks"]["tile_state_output_sram"]
    boundary = audit["capacity_audit"]["blocks"]["boundary_state_sram"]

    assert tile["modeled"] is True
    assert tile["peak_required_capacity_bits"] == 1024 * 8
    assert boundary["modeled"] is True
    assert boundary["peak_required_capacity_bits"] == 16384 * 8
    assert result["boundary_sram_access_energy_j"] == 0.0
    assert audit["external_memory_boundary"] == {
        "backend": "ddr",
        "observed": False,
        "energy_included": False,
        "latency_included": False,
    }


def test_boundary_sram_uses_safe_overlap_and_releases_last_consumer(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    state_a = _boundary_state(
        "state:A",
        consumers=["R1"],
        elements=2048,
        payload_kind="spike_sequence",
        bit_width=10,
    )
    ledger.observe_round(
        item_id="R0",
        placement=_placement(),
        step_results=[],
        boundary_commit_result=_commit(state_a),
    )
    state_b = _boundary_state(
        "state:B",
        consumers=["R2"],
        elements=4096,
        bit_width=8,
    )
    ledger.observe_round(
        item_id="R1",
        placement=_placement(),
        step_results=[],
        boundary_commit_result=_commit(
            state_b,
            events=[
                SimpleNamespace(
                    event_kind="read_boundary_state",
                    state_id="state:A",
                )
            ],
        ),
    )

    result = ledger.finalize(
        [
            {"item_id": "R0", "latency_s": 1.0},
            {"item_id": "R1", "latency_s": 1.0},
        ]
    )
    audit = result["static_power_audit"]
    boundary = audit["capacity_audit"]["blocks"]["boundary_state_sram"]

    # Spike storage is physically bit-packed despite its accumulator width.
    assert boundary["peak_required_capacity_bits"] == 2048 + 4096 * 8
    trace = audit["capacity_audit"]["boundary_sram_liveness_trace"]
    assert trace[-1]["live_bits_before"] == 2048
    assert trace[-1]["safe_overlap_live_bits"] == 2048 + 4096 * 8
    assert trace[-1]["live_bits_after"] == 4096 * 8
    assert audit["boundary_sram_access_by_round"]["R1"]["read_bits"] == 2048


def test_partial_update_renames_in_place_without_restoring_past_consumers(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    partial = _boundary_state(
        "state:L0:partial",
        consumers=["R1", "R2"],
        elements=1024,
        status="partial_sum",
        payload_kind="partial_sum_sequence",
        producer_layer_id="L0",
    )
    ledger.observe_round(
        item_id="R0",
        placement=_placement(),
        step_results=[],
        boundary_commit_result=_commit(partial),
    )

    complete = _boundary_state(
        "state:L0",
        consumers=["R1", "R2"],
        elements=1024,
        payload_kind="partial_sum_sequence",
        producer_layer_id="L0",
    )
    ledger.observe_round(
        item_id="R1",
        placement=_placement(),
        step_results=[],
        boundary_commit_result=_commit(
            updated_states=[complete],
            events=[
                SimpleNamespace(
                    event_kind="read_boundary_state",
                    state_id="state:L0:partial",
                )
            ],
        ),
    )
    ledger.observe_round(
        item_id="R2",
        placement=_placement(),
        step_results=[],
        boundary_commit_result=_commit(
            events=[
                SimpleNamespace(
                    event_kind="read_boundary_state",
                    state_id="state:L0",
                )
            ],
        ),
    )

    result = ledger.finalize(
        [
            {"item_id": "R0", "latency_s": 1.0},
            {"item_id": "R1", "latency_s": 1.0},
            {"item_id": "R2", "latency_s": 1.0},
        ]
    )
    trace = result["static_power_audit"]["capacity_audit"][
        "boundary_sram_liveness_trace"
    ]

    assert result["static_power_audit"]["capacity_audit"]["blocks"][
        "boundary_state_sram"
    ]["peak_required_capacity_bits"] == 1024 * 8
    assert trace[1]["safe_overlap_live_bits"] == 1024 * 8
    assert trace[1]["live_state_count_after"] == 1
    assert trace[2]["live_bits_after"] == 0
    assert result["boundary_sram_access_energy_j"] == 0.0


def test_central_static_ledger_rejects_legacy_access_window_leakage(tmp_path):
    with pytest.raises(Phase2Invalid, match="STATIC_POWER_DOUBLE_COUNT"):
        load_static_power_ledger(
            str(_config(tmp_path, include_legacy_leakage=True))
        )


def test_disabled_static_power_does_not_require_a_profile(tmp_path):
    config_path = tmp_path / "disabled.ini"
    config_path.write_text(
        "[Static Power]\n"
        "Static_Power_Enable = 0\n"
        "Static_Power_Profile = missing.json\n",
        encoding="utf-8",
    )

    assert load_static_power_ledger(str(config_path)) is None


def test_repository_root_profile_base_survives_relocated_sim_config(tmp_path):
    ledger = load_static_power_ledger(
        str(
            _config(
                tmp_path,
                profile_value=(
                    "configs/static_power/cacti_7_0_2kib_bank_stage_a.json"
                ),
                profile_base="repository_root",
            )
        )
    )

    assert ledger is not None
    assert ledger.profile_path == PROFILE_PATH


def test_digital_lif_state_is_added_once_per_unique_layer_and_tile(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    lif_step = SimpleNamespace(
        step_id="R0.S1",
        mapping_summary={
            "producer_layer_id": "Conv0",
            "digital_lif": {"state_bytes_by_tile": {0: 3000, 1: 1000}},
        },
    )
    ledger.observe_round(
        item_id="R0",
        placement=_placement(tile_bytes=1024.0),
        step_results=[lif_step, lif_step],
        boundary_commit_result=_commit(),
    )

    result = ledger.finalize([{"item_id": "R0", "latency_s": 1.0}])
    tile = result["static_power_audit"]["capacity_audit"]["blocks"][
        "tile_state_output_sram"
    ]

    # 1 KiB working output plus 3000 B state, rounded to two 2 KiB banks.
    assert tile["peak_required_capacity_bits"] == (1024 + 3000) * 8
    assert tile["banks_per_instance"] == 2
    assert math.isclose(
        result["static_power_audit"]["capacity_audit"]
        ["digital_lif_state_bytes_by_tile"]["0"],
        3000,
    )


def test_report_adds_static_energy_without_reclassifying_dynamic_read_energy(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    aggregator = ReportAggregator(
        "m0",
        str(_config(tmp_path)),
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_power_ledger=ledger,
    )
    item = SimpleNamespace(
        item_id="R0",
        item_kind="weighted_physical_round",
        item_index=0,
        round_kind="full_layer_pack",
        resource_proof=ResourceProof(32, 2),
    )
    placement = _placement()
    placement.item_id = "R0"
    placement.mapped_area_um2 = 1.0
    step = StepExecutionResult(
        item_id="R0",
        step_id="R0.S0",
        segment_index=0,
        step_kind="weighted_layer_segment",
        source_kind="static_eval",
        status="ok",
        origin_layer_ids=["Conv0"],
        metrics=MetricTotals(2.0, 3.0, 1.5, 1.0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[],
        warnings=[],
    )
    commit = BoundaryCommitResult(
        item_id="R0",
        item_kind="weighted_physical_round",
        status="ok",
        created_states=[],
        updated_states=[],
        released_states=[],
        events=[],
        completion_updates=[],
        visibility_updates=[],
        output_state_bindings=[],
        warnings=[],
    )

    aggregator.record_round(item, placement, [step], commit)
    report = aggregator.finalize()
    static_energy_j = report.inference_metrics["static_energy_j"]

    assert report.inference_metrics["read_energy_j"] == 3.0
    assert report.inference_metrics["weight_programming_energy_j"] == 0.0
    assert report.inference_metrics["energy_j"] == pytest.approx(
        3.0 + static_energy_j
    )
    assert report.item_results[0]["static_energy_j"] == static_energy_j
    assert report.inference_metrics["energy_breakdown_j"]["static_energy_j"] == (
        static_energy_j
    )


def test_global_accumulator_uses_peak_layer_mapping_as_one_physical_instance(tmp_path):
    ledger = load_static_power_ledger(str(_config(tmp_path)))
    global_step = SimpleNamespace(
        step_id="Z0.S0",
        mapping_summary={},
        layer_results=[
            SimpleNamespace(tcg_mapping={"global_buf_size_kb": 1.0}),
            SimpleNamespace(tcg_mapping={"global_buf_size_kb": 3.0}),
        ],
    )
    ledger.observe_round(
        item_id="Z0",
        placement=None,
        step_results=[global_step],
        boundary_commit_result=_commit(),
    )

    result = ledger.finalize([{"item_id": "Z0", "latency_s": 1.0}])
    global_buffer = result["static_power_audit"]["capacity_audit"]["blocks"][
        "global_accumulator_sram"
    ]

    assert global_buffer["instance_count"] == 1
    assert global_buffer["peak_required_capacity_bits"] == 3 * 1024 * 8
    assert global_buffer["banks_per_instance"] == 2
