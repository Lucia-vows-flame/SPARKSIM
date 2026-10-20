from types import SimpleNamespace

import pytest

from MNSIM.Scheduling.phase2.partial_accumulator import (
    ScheduledPartialAccumulatorCostModel,
)
from MNSIM.Scheduling.phase2.types import (
    BOUNDARY_STATUS_COMPLETE_OUTPUT,
    BOUNDARY_STATUS_PARTIAL_SUM,
    BoundaryCommitResult,
)


def _plan():
    descriptor = SimpleNamespace(
        layer_id="Conv1",
        prepass_ref=SimpleNamespace(shape_signature={"Outputsize": [2, 3]}),
    )
    return SimpleNamespace(layer_resource_descriptors=[descriptor])


def _commit(covered):
    return BoundaryCommitResult(
        item_id="R1",
        item_kind="weighted_physical_round",
        status="ok",
        created_states=[],
        updated_states=[],
        released_states=[],
        events=[],
        completion_updates=[
            {
                "origin_layer_id": "Conv1",
                "timestep": 0,
                "data_bitwidth": 9,
                "selected_fragment_ids": ["f1"],
                "covered_fragment_count": covered,
                "contribution_output_range": [0, 4],
                "completion_status": (
                    BOUNDARY_STATUS_PARTIAL_SUM
                    if covered < 2
                    else BOUNDARY_STATUS_COMPLETE_OUTPUT
                ),
            }
        ],
        visibility_updates=[],
        output_state_bindings=[],
        warnings=[],
    )


def test_first_partial_contribution_writes_state_without_scheduled_addition():
    model = ScheduledPartialAccumulatorCostModel("SimConfig.ini", _plan())

    result = model.score_commit(
        SimpleNamespace(item_id="R0", ordered_segments=[]), _commit(1)
    )

    audit = result.mapping_summary["scheduled_partial_accumulator"]
    assert audit["addition_count"] == 0
    assert audit["state_read_count"] == 0
    assert audit["state_write_count"] == 1
    assert audit["state_write_bytes"] == 27
    assert result.metrics.energy_j > 0
    assert result.metrics.latency_s > 0


def test_later_partial_contribution_uses_existing_adder_parameters():
    model = ScheduledPartialAccumulatorCostModel("SimConfig.ini", _plan())
    first = model.score_commit(
        SimpleNamespace(item_id="R0", ordered_segments=[]), _commit(1)
    )
    assert first is not None

    result = model.score_commit(
        SimpleNamespace(item_id="R1", ordered_segments=[]), _commit(2)
    )

    audit = result.mapping_summary["scheduled_partial_accumulator"]
    assert result.source_kind == "scheduled_partial_accumulator"
    assert audit["addition_count"] == 4 * 2 * 3
    assert audit["bit_widths"] == [9]
    assert audit["state_read_count"] == 1
    assert audit["state_write_count"] == 0
    assert audit["state_read_bytes"] == 27
    assert result.metrics.energy_j > 0
    assert result.metrics.latency_s > 0
