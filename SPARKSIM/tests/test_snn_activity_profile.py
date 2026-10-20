import pytest

from MNSIM.Scheduling.weight_scheduling import (
    LogicalTile,
    RoundSchedulingContext,
    SNNActivityProfileError,
    bind_profile_edges_to_rounds,
    validate_snn_activity_profile,
)


def _profile(role="calibration"):
    return {
        "schema_version": 1,
        "profile_id": "cal-v7-01",
        "model_ir_sha256": "a" * 64,
        "round_plan_sha256": "b" * 64,
        "weights_sha256": "c" * 64,
        "calibration": {
            "role": role,
            "split_id": "train-calibration-v1",
            "split_sha256": "d" * 64,
            "sample_count": 20,
        },
        "timestep_count": 30,
        "spike_transfer_edges": [
            {
                "source_round_id": "R0",
                "source_tile_id": "tile:0",
                "destination_round_id": "R1",
                "destination_tile_id": "tile:1",
                "spike_events_per_sample": 120.5,
            }
        ],
    }


def test_profile_binds_cross_round_edges_to_destination_context():
    profile = validate_snn_activity_profile(_profile())
    rounds = (
        RoundSchedulingContext("R0", 0, (LogicalTile("tile:0", (1,)),)),
        RoundSchedulingContext("R1", 1, (LogicalTile("tile:1", (2,)),)),
    )

    bound = bind_profile_edges_to_rounds(profile, rounds)

    assert bound[0]["incoming_spike_transfer_edges"] == []
    assert bound[1]["incoming_spike_transfer_edges"][0]["spike_events_per_sample"] == 120.5


def test_profile_rejects_evaluation_split_provenance():
    with pytest.raises(SNNActivityProfileError, match="calibration split"):
        validate_snn_activity_profile(_profile(role="test"))


def test_profile_rejects_unknown_tile_and_zero_traffic():
    profile = validate_snn_activity_profile(_profile())
    rounds = (
        RoundSchedulingContext("R0", 0, (LogicalTile("tile:0", (1,)),)),
        RoundSchedulingContext("R1", 1, (LogicalTile("tile:2", (2,)),)),
    )
    with pytest.raises(SNNActivityProfileError, match="logical tile"):
        bind_profile_edges_to_rounds(profile, rounds)

    no_traffic = _profile()
    no_traffic["spike_transfer_edges"][0]["spike_events_per_sample"] = 0
    with pytest.raises(SNNActivityProfileError, match="invariant"):
        bind_profile_edges_to_rounds(validate_snn_activity_profile(no_traffic), (
            RoundSchedulingContext("R0", 0, (LogicalTile("tile:0", (1,)),)),
            RoundSchedulingContext("R1", 1, (LogicalTile("tile:1", (2,)),)),
        ))
