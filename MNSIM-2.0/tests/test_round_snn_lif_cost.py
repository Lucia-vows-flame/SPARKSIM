import configparser
import math

import pytest

from MNSIM.Scheduling.phase2.digital_lif import (
    BETA_Q15,
    DigitalLifCostModel,
    DigitalLifError,
)


def test_formal_config_contains_frozen_digital_lif_contract():
    config = configparser.ConfigParser()
    config.read("SimConfig.ini")

    assert config.getboolean("Digital LIF", "Digital_LIF_Enable") is True
    assert config.getint("Digital LIF", "Digital_LIF_CoefficientBit") == 16
    assert config.getint("Digital LIF", "Digital_LIF_CoefficientFractionBit") == 15
    assert config.getint("Digital LIF", "Digital_LIF_LanesPerTile") == 64
    assert config.get("Digital LIF", "Digital_LIF_DecayMode") == "csd_shift_add"
    assert config.getboolean("Digital LIF", "Digital_LIF_AllowInputRequantization") is False
    assert config.get("Digital LIF", "Digital_LIF_OverflowMode") == "error"
    assert config.getboolean("Digital LIF", "Digital_LIF_WriteTerminalState") is False


def test_cost_model_freezes_coefficient_and_operation_inventory():
    model = DigitalLifCostModel("SimConfig.ini")
    cost = model.cost_sequence(
        bit_width=8,
        ownership_by_tile={0: 64},
        sequence_length=1,
        provisioned_tile_count=1,
    )

    assert cost.beta_q15 == BETA_Q15
    assert cost.csd_terms == ((1, 15), (-1, 13), (-1, 9), (1, 6), (1, 1))
    assert cost.compute_latency_ns == pytest.approx(7.329)
    assert cost.state_read_latency_ns == 0
    assert cost.state_write_latency_ns == 0
    assert cost.operation_inventory["adder_count"] == 7
    assert cost.operation_inventory["jointmodule_proxy_count"] == 3
    assert cost.operation_inventory["register_bits_per_update"] == 25


def test_cost_scales_energy_with_actual_updates_but_area_with_provisioning():
    model = DigitalLifCostModel("SimConfig.ini")
    one = model.cost_sequence(
        bit_width=8,
        ownership_by_tile={0: 17},
        sequence_length=1,
        provisioned_tile_count=1,
    )
    six = model.cost_sequence(
        bit_width=8,
        ownership_by_tile={0: 17},
        sequence_length=6,
        provisioned_tile_count=1,
    )
    two_tiles = model.cost_sequence(
        bit_width=8,
        ownership_by_tile={0: 17},
        sequence_length=1,
        provisioned_tile_count=2,
    )

    assert six.update_count == 6 * one.update_count
    assert six.compute_energy_nj == pytest.approx(6 * one.compute_energy_nj)
    assert two_tiles.compute_energy_nj == one.compute_energy_nj
    assert two_tiles.provisioned_area_um2 == pytest.approx(2 * one.provisioned_area_um2)


def test_tile_shards_drive_batches_and_buffer_ceilings_without_redistribution():
    model = DigitalLifCostModel("SimConfig.ini")
    cost = model.cost_sequence(
        bit_width=18,
        ownership_by_tile={0: 65, 3: 1},
        sequence_length=3,
        provisioned_tile_count=4,
    )

    assert cost.neuron_count == 66
    assert cost.active_tile_count == 2
    assert cost.batches_by_tile == {0: 2, 3: 1}
    assert cost.batches_per_step == 2
    assert cost.state_bytes_by_tile == {0: math.ceil(65 * 18 / 8), 3: 3}
    assert cost.compute_latency_ns == pytest.approx(3 * 2 * 7.329)
    assert cost.state_read_energy_nj > 0
    assert cost.state_write_energy_nj > 0
    assert cost.state_read_latency_ns > 0
    assert cost.state_write_latency_ns > 0


def test_width_cost_is_monotonic_and_legacy_primitives_are_sliced():
    model = DigitalLifCostModel("SimConfig.ini")
    costs = [
        model.cost_sequence(
            bit_width=width,
            ownership_by_tile={0: 1},
            sequence_length=1,
            provisioned_tile_count=1,
        )
        for width in (8, 16, 32)
    ]

    assert costs[0].compute_energy_nj < costs[1].compute_energy_nj < costs[2].compute_energy_nj
    assert costs[0].provisioned_area_um2 < costs[1].provisioned_area_um2 < costs[2].provisioned_area_um2
    assert costs[2].slice_counts["round_decision_jointmodule"] == 4
    assert costs[2].slice_counts["state_register"] == 2


def test_invalid_ownership_and_unsupported_profile_fail_explicitly():
    model = DigitalLifCostModel("SimConfig.ini")
    with pytest.raises(DigitalLifError, match="ownership"):
        model.cost_sequence(
            bit_width=8,
            ownership_by_tile={},
            sequence_length=1,
            provisioned_tile_count=1,
        )
    with pytest.raises(DigitalLifError, match="unsupported beta_q15"):
        model.cost_sequence(
            bit_width=8,
            ownership_by_tile={0: 1},
            sequence_length=1,
            provisioned_tile_count=1,
            beta_q15=123,
        )
