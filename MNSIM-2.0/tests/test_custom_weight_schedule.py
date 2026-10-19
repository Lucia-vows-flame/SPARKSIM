from dataclasses import replace
import json
from pathlib import Path

import pytest

from MNSIM.Scheduling.weight_scheduling import (
    CustomSchedule,
    CustomScheduleError,
    CustomScheduleRound,
    SchedulingConstraints,
    RoundSchedulingContext,
    LogicalTile,
    WeightScheduleProblem,
    generate_weight_schedule_from_custom,
    load_custom_schedule,
    load_device_model,
    custom_schedule_from_artifact,
    save_custom_schedule,
)


def _problem() -> WeightScheduleProblem:
    device = load_device_model("rram_stanford_asu_256")
    return WeightScheduleProblem(
        rounds=(
            RoundSchedulingContext(
                "R0", 0, (LogicalTile("a", (1,)), LogicalTile("b", (2,))),
                metadata={"reorder_block": 0},
            ),
            RoundSchedulingContext(
                "R1", 1, (LogicalTile("a", (2,)), LogicalTile("b", (1,))),
                metadata={"reorder_block": 0},
            ),
        ),
        constraints=SchedulingConstraints((0, 1)),
        device_model=device,
        model_id="public-test",
    )


def _schedule() -> CustomSchedule:
    return CustomSchedule(
        policy_id="custom_greedy_v1",
        rounds=(
            # The array order is the externally selected execution order.
            CustomScheduleRound("R1", {"a": 1, "b": 0}),
            CustomScheduleRound("R0", {"a": 0, "b": 1}),
        ),
    )


def test_custom_schedule_round_trip_and_digest(tmp_path: Path):
    source = _schedule()
    path = save_custom_schedule(tmp_path / "schedule.json", source)
    loaded = load_custom_schedule(path)
    assert loaded == source
    assert loaded.to_dict()["format"] == "mnsim.weight_schedule"
    assert loaded.sha256() == source.sha256()


def test_custom_schedule_is_converted_to_simulator_artifact():
    artifact = generate_weight_schedule_from_custom(_problem(), _schedule())
    assert [record.round_id for record in artifact.rounds] == ["R1", "R0"]
    assert artifact.policy_id == "custom_greedy_v1"
    assert artifact.policy_metadata["custom_schedule_schema_version"] == 1
    assert artifact.policy_metadata["custom_schedule_sha256"] == _schedule().sha256()
    # Costs and resident state are simulator-produced, not supplied by JSON.
    assert artifact.rounds[0].transition_costs["a"].changed_cells == 1
    assert artifact.final_resident_state == {"0": [1], "1": [2]}


def test_artifact_can_supply_an_optional_decision_template(tmp_path):
    artifact = generate_weight_schedule_from_custom(_problem(), _schedule())
    # This digest may describe an already reordered plan; do not bind the
    # exported template to it when compiling against the original plan.
    artifact = replace(artifact, base_round_plan_sha256="a" * 64)
    template = custom_schedule_from_artifact(artifact, policy_id="my_algorithm")
    assert "base_round_plan_sha256" not in template.identity
    path = save_custom_schedule(tmp_path / "template.json", template)
    rebuilt = generate_weight_schedule_from_custom(_problem(), load_custom_schedule(path))
    assert rebuilt.policy_id == "my_algorithm"
    assert rebuilt.final_resident_state == artifact.final_resident_state
    assert [r.logical_to_physical_tile for r in rebuilt.rounds] == [
        r.logical_to_physical_tile for r in artifact.rounds
    ]


def test_custom_schedule_rejects_duplicate_or_unavailable_tiles():
    problem = _problem()
    with pytest.raises(CustomScheduleError, match="duplicate physical"):
        CustomScheduleRound("R0", {"a": 0, "b": 0})

    unavailable = CustomSchedule(
        "bad", (CustomScheduleRound("R0", {"a": 4, "b": 1}), CustomScheduleRound("R1", {"a": 0, "b": 1}))
    )
    with pytest.raises(ValueError, match="unavailable physical"):
        generate_weight_schedule_from_custom(problem, unavailable)


def test_custom_schedule_identity_is_checked():
    schedule = CustomSchedule(
        "custom",
        _schedule().rounds,
        identity={"model_id": "different-model"},
    )
    with pytest.raises(CustomScheduleError, match="identity mismatch"):
        generate_weight_schedule_from_custom(_problem(), schedule)


def test_custom_schedule_loader_rejects_unknown_format(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text('{"format":"other","schema_version":1}', encoding="utf-8")
    with pytest.raises(CustomScheduleError, match="unsupported custom schedule format"):
        load_custom_schedule(path)


def test_round_reordering_requires_explicit_permission_and_dependencies():
    problem = _problem()
    fixed = replace(problem, rounds=tuple(replace(c, metadata={}) for c in problem.rounds))
    with pytest.raises(CustomScheduleError, match="fixed execution boundary"):
        generate_weight_schedule_from_custom(fixed, _schedule())
    dependent = replace(problem, rounds=(problem.rounds[0], replace(
        problem.rounds[1], metadata={"reorder_block": 0, "predecessor_round_ids": ["R0"]}
    )))
    with pytest.raises(CustomScheduleError, match="predecessor dependencies"):
        generate_weight_schedule_from_custom(dependent, _schedule())


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(schema_version=True),
    lambda p: p.update(schema_version=2),
    lambda p: p.update(policy_id=None),
    lambda p: p.update(identity={"typo": "value"}),
    lambda p: p.update(identity={"weights_sha256": ""}),
    lambda p: p.update(energy_j=0),
    lambda p: p["rounds"][0].update(round_id=None),
    lambda p: p["rounds"][0].update(assignments={"a": True, "b": 0}),
    lambda p: p["rounds"][0].update(logical_tile_slots={"a": {"pe_id": 4}}),
    lambda p: p.update(metadata={"score": float("nan")}),
])
def test_malformed_decisions_are_rejected(mutation):
    payload = _schedule().to_dict()
    mutation(payload)
    with pytest.raises(CustomScheduleError):
        CustomSchedule.from_dict(payload)


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"format":"mnsim.weight_schedule","format":"mnsim.weight_schedule"}')
    with pytest.raises(CustomScheduleError, match="duplicate JSON key"):
        load_custom_schedule(path)


@pytest.mark.parametrize("rounds", [
    (CustomScheduleRound("R0", {"a": 0, "b": 1}),),
    (CustomScheduleRound("R0", {"a": 0}), CustomScheduleRound("R1", {"a": 0, "b": 1})),
])
def test_missing_rounds_or_tiles_are_rejected(rounds):
    with pytest.raises(CustomScheduleError, match="(round ids|tile ids)"):
        generate_weight_schedule_from_custom(_problem(), CustomSchedule("custom", rounds))


def test_active_tile_budget_is_checked():
    problem = replace(_problem(), constraints=SchedulingConstraints((0, 1), max_active_tiles=1))
    with pytest.raises(CustomScheduleError, match="max_active_tiles"):
        generate_weight_schedule_from_custom(problem, _schedule())


def test_custom_metadata_cannot_change_execution():
    malicious = {"slot_permutations": {"tile:0": {"0:0": "1:1"}}, "source": "forged"}
    schedule = CustomSchedule("custom", tuple(replace(r, metadata=malicious) for r in _schedule().rounds), metadata={"round_order": []})
    artifact = generate_weight_schedule_from_custom(_problem(), schedule)
    assert artifact.policy_metadata["round_order"] == ["R1", "R0"]
    assert artifact.policy_metadata["custom_metadata"] == {"round_order": []}
    assert artifact.rounds[0].policy_metadata["source"] == "custom_schedule"
    assert "slot_permutations" not in artifact.rounds[0].policy_metadata


def test_remap_updates_physical_reducer_ownership():
    from MNSIM.Scheduling.weight_scheduling.custom_schedule import remap_custom_schedule_placement

    snapshot = {
        "item_id": "R0",
        "fragment_specs_by_id": {
            "f0": {"layer_id": "L0", "output_range": [0, 2], "full_output_range": [0, 2]},
            "f1": {"layer_id": "L0", "output_range": [0, 2], "full_output_range": [0, 2]},
        },
        "fragment_slots_by_id": {
            "f0": {"tile_id": 0, "pe_id": 0, "group_id": 0},
            "f1": {"tile_id": 1, "pe_id": 0, "group_id": 0},
        },
        "output_owner_shards_by_layer": {"L0": {"1": 2}},
    }
    moved = remap_custom_schedule_placement(snapshot, {"tile:0": 1, "tile:1": 0})
    assert moved["fragment_slots_by_id"]["f0"]["tile_id"] == 1
    # The last physical contributor owns the completed interval after remap.
    assert moved["output_owner_shards_by_layer"] == {"L0": {"1": 2}}
    assert snapshot["fragment_slots_by_id"]["f0"]["tile_id"] == 0


def test_cli_accepts_custom_algorithm_without_policy_registration():
    from MNSIM.Interface.round_orchestrated_cli import request_from_args
    request = request_from_args([
        "--network", "example_snn", "--weights", "weights.pth", "--num-classes", "8",
        "--sim-config", "SimConfig.ini", "--output-dir", "out", "--run-hardware",
        "--skip-accuracy", "--custom-schedule-in", "custom.json",
    ])
    assert request.custom_schedule_in == Path("custom.json")


@pytest.mark.parametrize("policy_id", ["user_algorithm", "snn_flashsef_temporal"])
def test_custom_schedule_runs_real_snn_phase2_and_freezes_artifact(tmp_path, policy_id):
    from types import SimpleNamespace
    import numpy as np
    from MNSIM.Interface.round_orchestrated_runner import RunnerDependencies, run_round_orchestrated_simulation
    from MNSIM.Interface.round_orchestrated_types import RoundOrchestratedRunRequest
    from MNSIM.Scheduling.round_types import LayerIR, ModelIR
    from MNSIM.Scheduling.weight_scheduling import load_schedule_artifact
    from test_round_snn_executor import _plan, _Backend

    plan = _plan()
    layer = LayerIR(
        layer_id="Conv1", layer_type="conv", is_weighted=True, stable_topological_index=0,
        input_tensor_ids=["model_input"], output_tensor_ids=["Conv1:out"],
        input_layer_ids=[], output_consumer_ids=["L1"], weight_ref_id="w:Conv1",
        shape=dict(plan.execution_items[0].selected_entries[0].prepass_ref.shape_signature),
        backend_attrs={"mnsim_layer_index": 0},
    )
    model = ModelIR("m0", "snn", plan.sequence_spec, [layer])
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fixture checkpoint")
    context = SimpleNamespace(net=SimpleNamespace(get_weights=lambda: [{
        "split0_weight0_positive": np.ones((16, 3), dtype=np.uint8),
        "split0_weight0_negative": np.zeros((16, 3), dtype=np.uint8),
    }]))
    path = save_custom_schedule(tmp_path / "custom.json", CustomSchedule(
        policy_id, (CustomScheduleRound("R0", {"tile:0": 0}),),
    ))
    backend = _Backend()
    request = RoundOrchestratedRunRequest(
        network="fixture_snn", weights=weights, num_classes=16,
        sim_config=Path(__file__).resolve().parents[1] / "SimConfig.ini",
        output_dir=tmp_path / "out", run_hardware=True, skip_accuracy=True,
        custom_schedule_in=path, device="cpu",
    )
    dependencies = RunnerDependencies(
        interface_factory=lambda **kwargs: context,
        model_ir_builder=lambda *args, **kwargs: model,
        plan_builder=lambda *args, **kwargs: plan,
        static_backend=backend, final_rram_validator=lambda path: None,
    )
    result = run_round_orchestrated_simulation(request, dependencies=dependencies)
    assert result.exit_code == 0, result.errors
    artifact = load_schedule_artifact(result.artifacts["schedule_artifact"])
    assert artifact.schema_version == 2
    assert artifact.policy_id == policy_id
    assert artifact.policy_metadata["schedule_policy"] == policy_id
    assert artifact.rounds[0].placement_snapshot is not None
    assert len(backend.calls) == 3
    assert result.phase2_report.inference_metrics["weight_programming_energy_j"] > 0
    assert result.phase2_report.inference_metrics["weight_programming_latency_s"] > 0
    assert "inactive_cell_policy" not in result.phase2_report.inference_metrics[
        "accounting_scope"
    ]["weight_programming"]
    frozen = json.loads((request.output_dir / "round_plan.json").read_text())
    assert frozen["round_plan"] == artifact.round_plan_snapshot
