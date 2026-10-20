from types import SimpleNamespace

import numpy as np
import pytest

from MNSIM.Scheduling.phase2.physical_write_ledger import (
    AnnRoundPhysicalWriteLedger,
    _fragment_target_matrix,
    load_physical_write_tables,
    score_xbar_physical_transition,
)
from MNSIM.Scheduling.phase2.types import MnsimGroupFragment


def _write_config(tmp_path):
    path = tmp_path / "write-ledger.ini"
    path.write_text(
        """[Device level]
Device_Level = 4
Device_Resistance = 100,50,25,12.5
Write_Voltage = 1,-1
pulse_latency = 1,1
SET_pulse = 0,1,2,3
RESET_pulse = 4,3,2,1

[Crossbar level]
Xbar_Size = 2,2

[Algorithm Configuration]
OU_Size = 1,2
""",
        encoding="UTF-8",
    )
    return str(path)


def _fragment(
    fragment_id,
    layer_id,
    *,
    input_range=(0, 2),
    output_range=(0, 1),
    input_block_index=0,
):
    return MnsimGroupFragment(
        fragment_id=fragment_id,
        layer_id=layer_id,
        layer_type="fc",
        input_range=input_range,
        output_range=output_range,
        full_input_range=(0, 4),
        full_output_range=(0, 2),
        weight_plane_index=0,
        polarity_count=1,
        physical_xbar_count=1,
        input_block_index=input_block_index,
    )


def _placement(fragment, *, tile_id, pe_id, group_id):
    return SimpleNamespace(
        fragment_specs_by_id={fragment.fragment_id: fragment},
        fragment_slots_by_id={
            fragment.fragment_id: {
                "tile_id": tile_id,
                "pe_id": pe_id,
                "group_id": group_id,
            }
        },
    )


def test_each_round_overwrites_active_slots_and_retains_idle_slots(tmp_path):
    config_path = _write_config(tmp_path)
    first = _fragment("F0", "L0")
    second = _fragment("F1", "L1")
    first_target = np.array([[1], [2]], dtype=np.int32)
    second_target = np.array([[3], [1]], dtype=np.int32)
    ledger = AnnRoundPhysicalWriteLedger(
        sim_config_path=config_path,
        weight_codes_by_layer={
            "L0": {"split0_weight0": first_target.T},
            "L1": {"split0_weight0": second_target.T},
        },
    )

    first_result = ledger.score_round(
        item_id="R0",
        placement=_placement(first, tile_id=7, pe_id=11, group_id=19),
    )
    second_result = ledger.score_round(
        item_id="R1",
        placement=_placement(second, tile_id=7, pe_id=11, group_id=23),
    )

    tables = load_physical_write_tables(config_path)
    zero = np.zeros((2, 2), dtype=np.int32)
    first_padded = np.zeros((2, 2), dtype=np.int32)
    first_padded[:, :1] = first_target
    second_padded = np.zeros((2, 2), dtype=np.int32)
    second_padded[:, :1] = second_target
    first_latency, first_energy = score_xbar_physical_transition(
        tables=tables,
        original_xbar=zero,
        target_xbar=first_padded,
    )
    second_latency, second_energy = score_xbar_physical_transition(
        tables=tables,
        original_xbar=zero,
        target_xbar=second_padded,
    )

    assert first_result["round_index"] == 0
    assert first_result["state_model"] == (
        "erased_initial_then_active_slot_overwrite_idle_retained"
    )
    assert first_result["latency_s"] == pytest.approx(first_latency)
    assert first_result["energy_j"] == pytest.approx(first_energy)
    assert second_result["round_index"] == 1
    assert second_result["retained_idle_slot_count"] == 1
    assert second_result["cleared_former_slot_count"] == 0
    assert second_result["transitioned_physical_xbar_count"] == 1
    assert second_result["latency_s"] == pytest.approx(second_latency)
    assert second_result["energy_j"] == pytest.approx(second_energy)

    # Reusing the formerly idle slot must transition from its retained L0
    # state, rather than from an invented erased state.
    third_result = ledger.score_round(
        item_id="R2",
        placement=_placement(second, tile_id=7, pe_id=11, group_id=19),
    )
    retained_latency, retained_energy = score_xbar_physical_transition(
        tables=tables,
        original_xbar=first_padded,
        target_xbar=second_padded,
    )
    assert third_result["latency_s"] == pytest.approx(retained_latency)
    assert third_result["energy_j"] == pytest.approx(retained_energy)


def test_checkpoint_device_codes_change_the_programming_energy(tmp_path):
    config_path = _write_config(tmp_path)
    fragment = _fragment("F0", "L0")
    placement = _placement(fragment, tile_id=37, pe_id=5, group_id=41)

    low_code = AnnRoundPhysicalWriteLedger(
        sim_config_path=config_path,
        weight_codes_by_layer={"L0": {"split0_weight0": np.array([[1, 1]])}},
    ).score_round(item_id="R0", placement=placement)
    high_code = AnnRoundPhysicalWriteLedger(
        sim_config_path=config_path,
        weight_codes_by_layer={"L0": {"split0_weight0": np.array([[3, 3]])}},
    ).score_round(item_id="R0", placement=placement)

    assert high_code["energy_j"] > low_code["energy_j"]
    assert high_code["latency_s"] > low_code["latency_s"]


def test_active_region_retention_skips_unused_padding_cells(tmp_path):
    config_path = _write_config(tmp_path)
    wide = _fragment(
        "F0",
        "L0",
        input_range=(0, 2),
        output_range=(0, 2),
    )
    narrow = _fragment(
        "F1",
        "L1",
        input_range=(0, 1),
        output_range=(0, 1),
    )
    placement_wide = _placement(wide, tile_id=0, pe_id=0, group_id=0)
    placement_narrow = _placement(narrow, tile_id=0, pe_id=0, group_id=0)
    codes = {
        "L0": {"split0_weight0": np.array([[3, 2], [1, 3]])},
        "L1": {"split0_weight0": np.array([[1]])},
    }
    padded = AnnRoundPhysicalWriteLedger(
        sim_config_path=config_path,
        weight_codes_by_layer=codes,
    )
    retained = AnnRoundPhysicalWriteLedger(
        sim_config_path=config_path,
        weight_codes_by_layer=codes,
        preserve_inactive_cells=True,
    )
    padded.score_round(item_id="R0", placement=placement_wide)
    retained.score_round(item_id="R0", placement=placement_wide)
    padded_result = padded.score_round(item_id="R1", placement=placement_narrow)
    retained_result = retained.score_round(item_id="R1", placement=placement_narrow)

    assert retained_result["energy_j"] < padded_result["energy_j"]
    assert retained_result["latency_s"] < padded_result["latency_s"]
    assert retained_result["inactive_cell_policy"] == "retain_outside_fragment_shape"


def test_conv_uses_explicit_input_block_and_transposes_checkpoint_matrix(tmp_path):
    config_path = _write_config(tmp_path)
    tables = load_physical_write_tables(config_path)
    fragment = MnsimGroupFragment(
        fragment_id="Conv0.in1.out1",
        layer_id="Conv0",
        layer_type="conv",
        input_range=(2, 4),
        output_range=(1, 3),
        full_input_range=(0, 4),
        full_output_range=(0, 3),
        weight_plane_index=0,
        polarity_count=2,
        physical_xbar_count=2,
        input_block_index=1,
    )
    codes = {
        "Conv0": {
            "split1_weight0_positive": np.array(
                [
                    [[0, 1]],
                    [[2, 3]],
                    [[1, 2]],
                ],
                dtype=np.int32,
            ),
            "split1_weight0_negative": np.zeros((3, 1, 2), dtype=np.int32),
        }
    }

    target = _fragment_target_matrix(
        fragment=fragment,
        polarity_index=0,
        weight_codes_by_layer=codes,
        tables=tables,
    )

    assert np.array_equal(target, np.array([[2, 1], [3, 2]], dtype=np.int32))
