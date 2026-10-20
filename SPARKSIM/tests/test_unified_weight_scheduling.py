from pathlib import Path

import numpy as np
import pytest

from MNSIM.Scheduling.weight_scheduling import (
    FlashSeFPTRPolicy,
    LogicalTile,
    RoundSchedulingContext,
    ScheduleArtifactError,
    SchedulingConstraints,
    SequentialPolicy,
    SNNFlashSeFPTRPolicy,
    SNNNativeTemporalPolicy,
    WeightScheduleProblem,
    generate_weight_schedule,
    execute_schedule_artifact,
    load_device_model,
    load_schedule_artifact,
    save_schedule_artifact,
    load_native_device_model_from_sim_config,
)
from MNSIM.Scheduling.phase2.physical_write_ledger import load_physical_write_tables


SIM_CONFIG = str(Path(__file__).resolve().parents[1] / "SimConfig.ini")


def _problem():
    device = load_device_model("rram_stanford_asu_256")
    rounds = (
        RoundSchedulingContext(
            round_id="R0",
            round_index=0,
            logical_tiles=(
                LogicalTile("a", (1, 1), {"slot": 0}),
                LogicalTile("b", (2, 2), {"slot": 1}),
            ),
        ),
        RoundSchedulingContext(
            round_id="R1",
            round_index=1,
            logical_tiles=(
                LogicalTile("a", (2, 2), {"slot": 0}),
                LogicalTile("b", (1, 1), {"slot": 1}),
            ),
        ),
    )
    return WeightScheduleProblem(
        rounds=rounds,
        constraints=SchedulingConstraints((0, 1)),
        device_model=device,
        model_id="test-model",
    )


@pytest.mark.parametrize("policy", [SequentialPolicy(), FlashSeFPTRPolicy()])
def test_policies_return_one_common_artifact_shape(policy):
    artifact = generate_weight_schedule(_problem(), policy)

    assert artifact.scheduler_id == "unified_weight_scheduler"
    assert artifact.device_model_id == "rram_stanford_asu_256"
    assert artifact.device_bits == 8
    assert artifact.device_level == 256
    assert artifact.execution_mode == "artifact_only"
    assert [record.round_id for record in artifact.rounds] == ["R0", "R1"]
    assert set(artifact.rounds[0].logical_to_physical_tile) == {"a", "b"}
    assert all(
        cost.energy_j is None and cost.latency_s is None
        for record in artifact.rounds
        for cost in record.transition_costs.values()
    )


def test_flashsef_ptr_uses_resident_state_and_preserves_contract():
    artifact = generate_weight_schedule(_problem(), FlashSeFPTRPolicy())

    first = artifact.rounds[0]
    second = artifact.rounds[1]
    assert second.resident_state_before == first.resident_state_after
    assert second.policy_metadata["assignment_method"] == "hungarian"
    assert artifact.transition_cost_objective == "changed_cell_count"


def test_artifact_round_chain_is_validated_on_load(tmp_path: Path):
    artifact = generate_weight_schedule(_problem(), SequentialPolicy())
    path = save_schedule_artifact(tmp_path / "schedule.json", artifact)
    loaded = load_schedule_artifact(path)
    assert loaded.sha256() == artifact.sha256()

    payload = path.read_text(encoding="utf-8").replace('"round_index": 1', '"round_index": 7')
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ScheduleArtifactError, match="round indexes"):
        load_schedule_artifact(path)


def test_artifact_executor_only_receives_loaded_plan(tmp_path: Path):
    artifact = generate_weight_schedule(_problem(), SequentialPolicy())
    path = save_schedule_artifact(tmp_path / "schedule.json", artifact)
    calls = []

    result = execute_schedule_artifact(
        str(path),
        lambda loaded: calls.append(loaded.policy_id) or loaded.sha256(),
        expected={"policy_id": "seq", "device_level": 128},
    )

    assert calls == ["seq"]
    assert result == artifact.sha256()


def test_schema_v2_artifact_requires_frozen_round_plan_snapshot(tmp_path: Path):
    artifact = generate_weight_schedule(_problem(), SequentialPolicy())
    artifact = artifact.__class__(
        **{
            **artifact.__dict__,
            "schema_version": 2,
            "round_plan_snapshot": {"plan_id": "frozen"},
        }
    )
    path = save_schedule_artifact(tmp_path / "schedule-v2.json", artifact)
    loaded = load_schedule_artifact(path)
    assert loaded.round_plan_snapshot == {"plan_id": "frozen"}


def test_schema_v2_artifact_without_round_plan_is_rejected(tmp_path: Path):
    artifact = generate_weight_schedule(_problem(), SequentialPolicy())
    payload = artifact.to_dict()
    payload["schema_version"] = 2
    path = tmp_path / "invalid-v2.json"
    import json

    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScheduleArtifactError, match="round_plan_snapshot"):
        load_schedule_artifact(path)


def test_asu_rejects_codes_above_eight_bit_range():
    problem = _problem()
    invalid_round = RoundSchedulingContext(
        round_id="R0",
        round_index=0,
        logical_tiles=(LogicalTile("a", (256,)),),
    )
    with pytest.raises(ValueError, match="outside"):
        WeightScheduleProblem(
            rounds=(invalid_round,),
            constraints=SchedulingConstraints((0,)),
            device_model=problem.device_model,
        )


def test_snn_aware_assignment_uses_native_write_objective_without_bit_hops():
    device = load_native_device_model_from_sim_config(SIM_CONFIG)
    rounds = (
        RoundSchedulingContext(
            "R0",
            0,
            (LogicalTile("a", (1, 1)), LogicalTile("b", (255, 255))),
        ),
        RoundSchedulingContext(
            "R1",
            1,
            (LogicalTile("a", (255, 255)), LogicalTile("b", (1, 1))),
        ),
    )
    artifact = generate_weight_schedule(
        WeightScheduleProblem(
            rounds=rounds,
            constraints=SchedulingConstraints((0, 1)),
            device_model=device,
            model_id="snn-native-test",
            metadata={"backend_kind": "snn"},
        ),
        SNNFlashSeFPTRPolicy(),
    )

    assert artifact.policy_id == "snn_flashsef_ptr"
    assert artifact.transition_cost_exact is True
    assert artifact.policy_metadata["snn_bit_hop_objective"] is False
    assert artifact.policy_metadata["snn_activity_profile_objective"] is False
    assert all(
        "estimated_spike_bit_hops_per_sample" not in record.policy_metadata
        and cost.energy_j is not None
        and cost.latency_s is not None
        for record in artifact.rounds
        for cost in record.transition_costs.values()
    )
    assert all(
        record.policy_metadata["baseline_preserved"] is True
        for record in artifact.rounds
    )


def test_snn_aware_assignment_retains_temporal_resident_state():
    device = load_native_device_model_from_sim_config(SIM_CONFIG)
    rounds = (
        RoundSchedulingContext(
            "R0", 0, (LogicalTile("source", (1,)),)
        ),
        RoundSchedulingContext(
            "R1",
            1,
            (LogicalTile("destination", (2,)),),
        ),
    )
    artifact = generate_weight_schedule(
        WeightScheduleProblem(
            rounds=rounds,
            constraints=SchedulingConstraints((0, 1)),
            device_model=device,
            metadata={"backend_kind": "snn"},
        ),
        SNNFlashSeFPTRPolicy(),
    )

    assert artifact.rounds[1].resident_state_before == artifact.rounds[0].resident_state_after
    assert artifact.rounds[1].policy_metadata["snn_bit_hop_objective"] is False


def test_native_device_model_matches_mnsim_write_table_contract():
    model = load_native_device_model_from_sim_config(SIM_CONFIG)
    tables = load_physical_write_tables(SIM_CONFIG)

    assert model.device_level == 256
    assert model.exact_write_energy is True
    assert model.exact_write_latency is True
    assert len(model.transition_energy_j or ()) == 256 * 256
    assert len(model.transition_latency_s or ()) == 256 * 256
    assert model.transition_energy_j[0] == 0.0
    assert model.transition_latency_s[0] == 0.0
    assert model.transition_energy_j[1] > 0.0
    assert model.transition_latency_s[1] > 0.0
    assert np.array_equal(
        np.asarray(model.transition_energy_j), tables.energy_nj.reshape(-1) * 1e-9
    )
    assert np.array_equal(
        np.asarray(model.transition_latency_s), tables.latency_ns.reshape(-1) * 1e-9
    )


def test_snn_native_temporal_policy_uses_native_costs_and_keeps_baseline_candidate():
    model = load_native_device_model_from_sim_config(SIM_CONFIG)
    rounds = (
        RoundSchedulingContext(
            "R0",
            0,
            (
                LogicalTile("a", (1, 1)),
                LogicalTile("b", (255, 255)),
            ),
        ),
        RoundSchedulingContext(
            "R1",
            1,
            (
                LogicalTile("a", (255, 255)),
                LogicalTile("b", (1, 1)),
            ),
        ),
    )
    problem = WeightScheduleProblem(
        rounds=rounds,
        constraints=SchedulingConstraints((0, 1)),
        device_model=model,
        model_id="snn-native-test",
        metadata={"backend_kind": "snn"},
    )

    artifact = generate_weight_schedule(problem, SNNNativeTemporalPolicy())

    assert artifact.policy_id == "snn_native_temporal"
    assert artifact.path_id == "sched-snn_native_temporal"
    assert artifact.transition_cost_exact is True
    assert artifact.policy_metadata["native_objective"] == (
        "mnsim_physical_write_energy_then_cell_pulse_latency"
    )
    assert artifact.policy_metadata["baseline_tolerance"] == 0.0
    assert all(
        cost.energy_j is not None and cost.latency_s is not None
        for record in artifact.rounds
        for cost in record.transition_costs.values()
    )
    artifact.validate()


def test_snn_native_temporal_policy_reorders_an_explicit_legal_block():
    model = load_native_device_model_from_sim_config(SIM_CONFIG)
    rounds = tuple(
        RoundSchedulingContext(
            round_id=round_id,
            round_index=index,
            logical_tiles=(LogicalTile("tile", (code,)),),
            metadata={"reorder_block": 0, "predecessor_round_ids": []},
        )
        for index, (round_id, code) in enumerate(
            (("R0", 1), ("R1", 2), ("R2", 1))
        )
    )
    artifact = generate_weight_schedule(
        WeightScheduleProblem(
            rounds=rounds,
            constraints=SchedulingConstraints((0,)),
            device_model=model,
            model_id="snn-native-order-test",
            metadata={"backend_kind": "snn"},
        ),
        SNNNativeTemporalPolicy(),
    )

    assert [record.round_id for record in artifact.rounds] == ["R0", "R2", "R1"]
    assert artifact.policy_metadata["round_order_changed"] is True
    assert artifact.rounds[1].resident_state_before == artifact.rounds[0].resident_state_after
    artifact.validate()


def test_native_physical_targets_retain_idle_slots_and_use_exact_write_energy():
    model = load_native_device_model_from_sim_config(SIM_CONFIG)
    first = LogicalTile(
        "tile",
        (1,),
        metadata={
            "_native_physical_targets": [
                {"slot": [0, 0, 0], "shape": [1, 1], "values": [1]}
            ]
        },
    )
    second = LogicalTile(
        "tile",
        (2,),
        metadata={
            "_native_physical_targets": [
                {"slot": [0, 1, 0], "shape": [1, 1], "values": [2]}
            ]
        },
    )
    artifact = generate_weight_schedule(
        WeightScheduleProblem(
            rounds=(
                RoundSchedulingContext("R0", 0, (first,)),
                RoundSchedulingContext("R1", 1, (second,)),
            ),
            constraints=SchedulingConstraints((0,)),
            device_model=model,
            model_id="snn-native-physical-state-test",
            metadata={"backend_kind": "snn"},
        ),
        SNNNativeTemporalPolicy(),
    )

    tables = load_physical_write_tables(SIM_CONFIG)
    expected_first = float(tables.energy_nj[0, 1]) * 1e-9
    expected_second = float(tables.energy_nj[0, 2]) * 1e-9
    assert artifact.rounds[0].transition_costs["tile"].energy_j == pytest.approx(
        expected_first
    )
    assert artifact.rounds[1].transition_costs["tile"].energy_j == pytest.approx(
        expected_second
    )
    assert artifact.rounds[1].transition_costs["tile"].provenance[
        "physical_fragment_slot_exact"
    ] is True
