import pytest

from MNSIM.Scheduling.phase2.boundary import BoundaryStateManager
from MNSIM.Scheduling.phase2.types import (
    CONTRIBUTION_PARTIAL,
    BoundaryBinding,
    BoundaryCommitRequest,
    MetricTotals,
    Phase2Invalid,
    StepExecutionResult,
)


def _binding(timestep, fragment_id, *, sample_id="s0", required=("f0", "f1")):
    return BoundaryBinding(
        local_layer_id=0,
        hook_node_id=None,
        origin_layer_id="Conv1",
        origin_layer_type="conv",
        report_role="reportable",
        tensor_role="partial_sum",
        shape={"C": 4},
        region={"kind": "channel"},
        input_range=[0, 4],
        output_range=[0, 4],
        full_input_range=[0, 4],
        full_output_range=[0, 4],
        contribution_kind=CONTRIBUTION_PARTIAL,
        source_state_ids=["model_input"],
        downstream_consumer_ids=["R1"],
        fragment_ids=[fragment_id],
        required_fragment_ids=list(required),
        sample_id=sample_id,
        timestep=timestep,
        sequence_length=3,
        data_bitwidth=18,
        accumulator_domain={"domain_id": "acc:test"},
        payload_kind="partial_sum_sequence",
    )


def _request(item_id, binding):
    step = StepExecutionResult(
        item_id=item_id,
        step_id=f"{item_id}.S0",
        segment_index=0,
        step_kind="weighted_layer_segment",
        source_kind="static_eval",
        status="ok",
        origin_layer_ids=["Conv1"],
        metrics=MetricTotals(0, 0, 0, 0),
        layer_results=[],
        mapping_summary={},
        raw_mnsim_audit=[],
        boundary_bindings=[binding],
        warnings=[],
        sample_id=binding.sample_id,
        timestep=binding.timestep,
    )
    return BoundaryCommitRequest(
        item_id=item_id,
        item_kind="weighted_physical_round",
        execution_index=0,
        input_view_id=f"view:{item_id}",
        source_kind="round_finalize",
        static_eval_result=None,
        hook_result=None,
        boundary_bindings=[binding],
        step_results=[step],
        sample_id=binding.sample_id,
        timestep=binding.timestep,
    )


def test_same_fragment_id_is_legal_at_different_timesteps():
    manager = BoundaryStateManager()
    manager.seed_model_input("model_input", {"C": 4})

    first = manager.commit(_request("R0", _binding(0, "f0")))
    second = manager.commit(_request("R0", _binding(1, "f0")))

    assert first.created_states[0].timestep == 0
    assert second.created_states[0].timestep == 1
    assert first.created_states[0].state_id != second.created_states[0].state_id


def test_duplicate_fragment_at_same_sample_and_timestep_fails():
    manager = BoundaryStateManager()
    manager.seed_model_input("model_input", {"C": 4})
    binding = _binding(0, "f0")
    manager.commit(_request("R0", binding))

    with pytest.raises(Phase2Invalid, match="repeats fragment ids"):
        manager.commit(_request("R0", binding))


def test_each_timestep_completes_independently_and_preserves_audit_metadata():
    manager = BoundaryStateManager()
    manager.seed_model_input("model_input", {"C": 4})
    manager.commit(_request("R0", _binding(0, "f0")))
    manager.commit(_request("R0", _binding(1, "f0")))

    t0 = manager.commit(_request("R1", _binding(0, "f1")))
    t1 = manager.commit(_request("R1", _binding(1, "f1")))

    assert t0.completion_updates[-1]["timestep"] == 0
    assert t1.completion_updates[-1]["timestep"] == 1
    assert t0.updated_states[-1].data_bitwidth == 18
    assert t0.updated_states[-1].payload_kind == "partial_sum_sequence"
    assert t0.updated_states[-1].accumulator_domain == {"domain_id": "acc:test"}


def test_same_timestep_in_different_samples_has_separate_identity():
    manager = BoundaryStateManager()
    manager.seed_model_input("model_input", {"C": 4})

    s0 = manager.commit(_request("R0", _binding(0, "f0", sample_id="s0")))
    s1 = manager.commit(_request("R0", _binding(0, "f0", sample_id="s1")))

    assert s0.created_states[0].sample_id == "s0"
    assert s1.created_states[0].sample_id == "s1"
    assert s0.created_states[0].state_id != s1.created_states[0].state_id
