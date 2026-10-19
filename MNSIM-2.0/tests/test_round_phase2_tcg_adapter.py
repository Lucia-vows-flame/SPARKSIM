import configparser
import math
from copy import deepcopy

import numpy as np
import pytest

from MNSIM.Mapping_Model import Tile_connection_graph
from MNSIM.Scheduling.phase2.netstruct_builder import build_netstruct
from MNSIM.Scheduling.phase2.tcg_adapter import (
    WeightedScope,
    deserialize_round_placement,
    prepare_round_placement,
    serialize_round_placement,
)
from MNSIM.Scheduling.phase2.types import (
    MnsimFragmentLayout,
    MnsimGroupFragment,
    Phase2Invalid,
)


def _sim_config_path(tmp_path, *, tile_num, pe_num="2,1", group_num="4"):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("SimConfig.ini", encoding="UTF-8")
    parser.set("Crossbar level", "Xbar_Size", "128,128")
    parser.set("Crossbar level", "Subarray_Size", "128")
    parser.set("Architecture level", "Tile_Num", tile_num)
    parser.set("Tile level", "PE_Num", pe_num)
    parser.set("Process element level", "Group_Num", group_num)
    parser.set("Process element level", "Xbar_Polarity", "2")
    parser.set("Process element level", "DAC_Num", "128")
    parser.set("Process element level", "ADC_Num", "128")
    path = tmp_path / (
        f"SimConfig.{tile_num.replace(',', '_')}."
        f"{pe_num.replace(',', '_')}.{group_num}.ini"
    )
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return str(path)


def _round_residency_config(tmp_path):
    return _sim_config_path(
        tmp_path,
        tile_num="1,1",
        pe_num="2,1",
        group_num="2",
    )


def _conv_layer_info():
    return {
        "type": "conv",
        "Inputchannel": 1,
        "Outputchannel": 1,
        "Kernelsize": 1,
        "Inputsize": [1, 1],
        "Outputsize": [1, 1],
        "Stride": 1,
        "Padding": 0,
        "Inputbit": 8,
        "Weightbit": 8,
        "outputbit": 8,
        "row_split_num": 1,
        "weight_bit_split_part": 1,
    }


def _one_pe_conv():
    layer_info = _conv_layer_info()
    layer_info["Weightbit"] = 2
    return layer_info


def _fragment_layout(
    *,
    layer_id="L0",
    layer_type="conv",
    active_group_count=1,
    active_physical_xbar_count=2,
    fragment_count=None,
):
    if fragment_count is None:
        fragment_count = (
            active_group_count
            if isinstance(active_group_count, int)
            and not isinstance(active_group_count, bool)
            and active_group_count >= 0
            else 1
        )
    fragments = tuple(
        MnsimGroupFragment(
            fragment_id=f"{layer_id}.g{index}",
            layer_id=layer_id,
            layer_type=layer_type,
            input_range=(0, 1),
            output_range=(0, 1),
            full_input_range=(0, 1),
            full_output_range=(0, 1),
            weight_plane_index=index,
            polarity_count=2,
            physical_xbar_count=2,
        )
        for index in range(fragment_count)
    )
    return MnsimFragmentLayout(
        layer_id=layer_id,
        layer_type=layer_type,
        fragments=fragments,
        native_layer_info=_conv_layer_info(),
        active_group_count=active_group_count,
        active_physical_xbar_count=active_physical_xbar_count,
    )


def _scope(
    step_id,
    scope_index,
    layer_info,
    *,
    active_group_count=1,
    active_physical_xbar_count=2,
    layer_type="conv",
):
    return WeightedScope(
        step_id,
        scope_index,
        layer_info,
        fragment_layout=_fragment_layout(
            layer_id=f"{step_id}.scope{scope_index}",
            layer_type=layer_type,
            active_group_count=active_group_count,
            active_physical_xbar_count=active_physical_xbar_count,
        ),
    )


def _two_scopes():
    return [
        _scope("R0.S0", 0, _conv_layer_info()),
        _scope("R0.S2", 0, _conv_layer_info()),
    ]


def _same_step_reordered_scopes():
    first = _conv_layer_info()
    first["Outputchannel"] = 1
    second = _conv_layer_info()
    second["Outputchannel"] = 2
    return [
        _scope("R0.S0", 1, second),
        _scope("R0.S0", 0, first),
    ]


def test_round_placement_maps_all_weighted_scopes_once(tmp_path):
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
        weighted_scopes=_two_scopes(),
        expected_used_xbars=4,
    )

    assert placement.mapping_summary["weighted_step_ids"] == ["R0.S0", "R0.S2"]
    assert placement.mapping_summary["mapping_layer_count"] == 2
    assert placement.mapping_summary["phase1_used_xbars"] == 4
    assert placement.mapping_summary["tcg_allocated_xbar_slots"] >= 4
    assert placement.mapping_summary["zero_step_ids"] == []
    capacity_evidence = placement.mapping_summary["tcg_buffer_capacity_evidence"]
    assert len(capacity_evidence) == 2
    assert all(
        entry["pe_input_working_set_bytes_per_instance"] > 0
        for entry in capacity_evidence
    )
    assert all(
        entry["tile_output_working_set_bytes_per_instance"] > 0
        for entry in capacity_evidence
    )
    assert (
        placement.mapping_summary[
            "tcg_pe_input_rounded_capacity_bits_per_instance"
        ]
        >= max(
            math.ceil(
                entry["pe_input_working_set_bytes_per_instance"] * 8
            )
            for entry in capacity_evidence
        )
    )
    assert set(placement.mapping_layer_ids_by_step) == {"R0.S0", "R0.S2"}


def test_round_placement_snapshot_round_trip_restores_phase2_slice(tmp_path):
    sim_config_path = _round_residency_config(tmp_path)
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=sim_config_path,
        weighted_scopes=[_scope("R0.S0", 0, _one_pe_conv())],
        expected_used_xbars=2,
    )

    snapshot = serialize_round_placement(placement)
    restored = deserialize_round_placement(snapshot, sim_config_path=sim_config_path)

    assert restored.item_id == placement.item_id
    assert restored.fragment_slots_by_id == placement.fragment_slots_by_id
    assert restored.mapping_layer_ids_by_step == placement.mapping_layer_ids_by_step
    assert restored.tcg.layer_tileinfo == placement.tcg.layer_tileinfo
    restored_slice = restored.slice_for(
        "R0.S0", 0, build_netstruct([_one_pe_conv()])
    )
    assert restored_slice.tileinfo == placement.slice_for(
        "R0.S0", 0, build_netstruct([_one_pe_conv()])
    ).tileinfo


def test_round_placement_shares_one_tile_between_two_weighted_scopes(tmp_path):
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=_round_residency_config(tmp_path),
        weighted_scopes=[
            _scope("R0.S0", 0, _one_pe_conv()),
            _scope("R0.S2", 0, _one_pe_conv()),
        ],
        expected_used_xbars=4,
    )

    assert placement.mapping_summary["phase1_used_xbars"] == 4
    assert placement.mapping_summary["round_residency_tile_count"] == 1
    assert placement.mapping_summary["logical_layer_tile_count"] == 2
    assert placement.mapped_area_um2 > 0.0


def test_round_placement_proves_active_physical_arrays_not_penum_padding(tmp_path):
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=_round_residency_config(tmp_path),
        weighted_scopes=[
            _scope("R0.S0", 0, _conv_layer_info()),
            _scope("R0.S2", 0, _conv_layer_info()),
        ],
        expected_used_xbars=4,
    )

    assert placement.mapping_summary["active_physical_xbar_count"] == 4
    assert placement.mapping_summary["active_group_count"] == 2
    assert placement.mapping_summary["mnsim_group_capacity"] == 4
    assert placement.mapping_summary["physical_xbar_capacity"] == 8
    assert placement.mapping_summary["tcg_allocated_xbar_slots"] == 4
    assert [info["PEnum"] for info in placement.tcg.layer_tileinfo] == [1, 1]
    assert [info["max_group"] for info in placement.tcg.layer_tileinfo] == [1, 1]
    assert placement.tcg.layer_tileinfo[0]["active_fragment_ids"] == ["R0.S0.scope0.g0"]
    assert placement.tcg.active_fragment_group_slots[0][0] == {
        "fragment_id": "R0.S0.scope0.g0",
        "tile_id": 0,
        "pe_id": 0,
        "group_id": 0,
    }


def test_active_fragment_layout_overrides_native_full_precision_mapping(tmp_path):
    full_precision_layer = _conv_layer_info()
    full_precision_layer["Weightbit"] = 9
    sim_config_path = _sim_config_path(
        tmp_path,
        tile_num="2,2",
        pe_num="2,1",
        group_num="2",
    )

    plain = Tile_connection_graph.TCG(
        build_netstruct([full_precision_layer]),
        sim_config_path,
        multiple=[1],
    )
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=sim_config_path,
        weighted_scopes=[
            _scope("R0.S0", 0, full_precision_layer)
        ],
        expected_used_xbars=2,
    )

    assert plain.layer_tileinfo[0]["PEnum"] == 4
    assert placement.tcg.layer_tileinfo[0]["PEnum"] == 1
    assert placement.tcg.layer_tileinfo[0]["max_group"] == 1
    assert placement.tcg.layer_tileinfo[0]["tilenum"] == 1
    assert placement.mapping_summary["tcg_allocated_xbar_slots"] == 2


def test_tcg_fragment_layout_is_opt_in_and_preserves_ordinary_mapping(tmp_path):
    sim_config_path = _round_residency_config(tmp_path)
    netstruct = build_netstruct([_one_pe_conv()])
    layout = _fragment_layout(layer_id="L0")

    plain = Tile_connection_graph.TCG(netstruct, sim_config_path, multiple=[1])
    active = Tile_connection_graph.TCG(
        netstruct,
        sim_config_path,
        multiple=[1],
        round_residency=True,
        active_fragment_layout={
            0: {
                "fragment_ids": [fragment.fragment_id for fragment in layout.fragments],
                "active_group_count": layout.active_group_count,
                "active_physical_xbar_count": layout.active_physical_xbar_count,
                "polarity_count": layout.fragments[0].polarity_count,
            }
        },
    )

    assert "active_group_count" not in plain.layer_tileinfo[0]
    assert active.layer_tileinfo[0]["active_group_count"] == 1
    assert active.layer_tileinfo[0]["active_physical_xbar_count"] == 2
    assert active.active_fragment_group_slots[0][0]["fragment_id"] == "L0.g0"


def test_round_placement_rejects_active_group_overflow_before_tcg(tmp_path, monkeypatch):
    tcg_calls = []
    monkeypatch.setattr(
        Tile_connection_graph,
        "TCG",
        lambda *args, **kwargs: tcg_calls.append(args),
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(
                tmp_path,
                tile_num="1,1",
                pe_num="1,1",
                group_num="1",
            ),
            weighted_scopes=[
                _scope("R0.S0", 0, _one_pe_conv()),
                _scope("R0.S1", 0, _one_pe_conv()),
            ],
            expected_used_xbars=4,
        )

    assert exc_info.value.code == "TCG_ACTIVE_XBAR_PROOF_MISMATCH"
    assert tcg_calls == []


def test_round_placement_rejects_nonweighted_fragment_layout_before_tcg(tmp_path, monkeypatch):
    tcg_calls = []
    monkeypatch.setattr(
        Tile_connection_graph,
        "TCG",
        lambda *args, **kwargs: tcg_calls.append(args),
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
            weighted_scopes=[
                _scope(
                    "R0.S0",
                    0,
                    {"type": "pooling"},
                    layer_type="pooling",
                )
            ],
            expected_used_xbars=2,
        )

    assert exc_info.value.code == "TCG_ROUND_PLACEMENT_INCONSISTENT"
    assert tcg_calls == []


def test_original_tcg_default_mode_rejects_two_one_pe_scopes_on_one_tile(tmp_path):
    with pytest.raises(AssertionError, match="Tile number is not enough"):
        Tile_connection_graph.TCG(
            build_netstruct([_one_pe_conv(), _one_pe_conv()]),
            _round_residency_config(tmp_path),
            multiple=[1, 1],
        )


def test_round_placement_rejects_numeric_string_step_id_collision_before_tcg(
    tmp_path,
    monkeypatch,
):
    tcg_calls = []
    monkeypatch.setattr(
        Tile_connection_graph,
        "TCG",
        lambda *args, **kwargs: tcg_calls.append(args),
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
            weighted_scopes=[
                _scope(1, 0, _conv_layer_info()),
                _scope("1", 0, _conv_layer_info()),
            ],
            expected_used_xbars=4,
        )

    assert exc_info.value.code == "TCG_ROUND_PLACEMENT_INCONSISTENT"
    assert tcg_calls == []


@pytest.mark.parametrize("step_id", (1, ""))
def test_round_placement_rejects_non_string_or_empty_step_ids_before_tcg(
    tmp_path,
    monkeypatch,
    step_id,
):
    tcg_calls = []

    def fail_if_constructed(*args, **kwargs):
        tcg_calls.append(args)
        raise AssertionError("invalid weighted scopes must not construct a TCG")

    monkeypatch.setattr(Tile_connection_graph, "TCG", fail_if_constructed)

    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
            weighted_scopes=[
                _scope(step_id, 0, _conv_layer_info())
            ],
            expected_used_xbars=2,
        )

    assert exc_info.value.code == "TCG_ROUND_PLACEMENT_INCONSISTENT"
    assert tcg_calls == []


@pytest.mark.parametrize("layer_info", (None, ["not", "a", "mapping"]))
def test_round_placement_rejects_non_mapping_layer_info_before_tcg(
    tmp_path,
    monkeypatch,
    layer_info,
):
    tcg_calls = []
    monkeypatch.setattr(
        Tile_connection_graph,
        "TCG",
        lambda *args, **kwargs: tcg_calls.append(args),
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
            weighted_scopes=[
                _scope("R0.S0", 0, layer_info)
            ],
            expected_used_xbars=2,
        )

    assert exc_info.value.code == "TCG_ROUND_PLACEMENT_INCONSISTENT"
    assert tcg_calls == []


def test_slice_reindexes_one_original_mapping_layer_and_never_remaps(tmp_path):
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
        weighted_scopes=_two_scopes(),
        expected_used_xbars=4,
    )
    request_netstruct = build_netstruct([_conv_layer_info()])

    placement_slice = placement.slice_for("R0.S2", 0, request_netstruct)
    original_tileinfo = deepcopy(placement.tcg.layer_tileinfo[1])
    original_mapping_ids = deepcopy(placement.mapping_layer_ids_by_step)
    original_tileinfo_table = deepcopy(placement.tcg.layer_tileinfo)
    original_in_distance = placement.tcg.inLayer_distance.copy()
    original_trans_distance = placement.tcg.transLayer_distance.copy()
    slice_mapping_ids = deepcopy(placement_slice.source_mapping_layer_ids)
    slice_tileinfo = deepcopy(placement_slice.tileinfo)
    slice_in_distance = placement_slice.inLayer_distance.copy()
    slice_trans_distance = placement_slice.transLayer_distance.copy()

    assert placement_slice.layer_num == 1
    assert placement_slice.tileinfo[0]["type"] == "conv"
    assert placement_slice.net == request_netstruct
    assert placement_slice.source_mapping_layer_ids == [1]
    assert placement_slice.tileinfo[0]["startid"] == 0
    assert placement_slice.tileinfo[0]["Inputindex"] == [-1]
    assert placement_slice.tileinfo[0]["Outputindex"] == []
    assert placement_slice.tileinfo[0]["is_branchin"] == -1
    assert placement_slice.tileinfo[0]["is_branchout"] == -1
    for field in ("type", "mx", "my", "PEnum", "tilenum", "max_PE", "max_group", "max_row", "max_column"):
        assert placement_slice.tileinfo[0][field] == original_tileinfo[field]
    placement_slice.mapping_net()
    placement_slice.calculate_transfer_distance()
    assert placement.mapping_layer_ids_by_step == original_mapping_ids
    assert placement.tcg.layer_tileinfo == original_tileinfo_table
    assert np.array_equal(placement.tcg.inLayer_distance, original_in_distance)
    assert np.array_equal(placement.tcg.transLayer_distance, original_trans_distance)
    assert placement_slice.source_mapping_layer_ids == slice_mapping_ids
    assert placement_slice.tileinfo == slice_tileinfo
    assert np.array_equal(placement_slice.inLayer_distance, slice_in_distance)
    assert np.array_equal(placement_slice.transLayer_distance, slice_trans_distance)


def test_slice_context_prefix_is_independent_and_preserves_resident_placement(tmp_path):
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
        weighted_scopes=_two_scopes(),
        expected_used_xbars=4,
    )
    placement_slice = placement.slice_for(
        "R0.S2",
        0,
        build_netstruct([_conv_layer_info()]),
    )
    prefixed_netstruct = build_netstruct([_conv_layer_info(), _conv_layer_info()])
    resident_tileinfo = deepcopy(placement.tcg.layer_tileinfo)
    resident_in_distance = placement.tcg.inLayer_distance.copy()
    resident_trans_distance = placement.tcg.transLayer_distance.copy()
    slice_tileinfo = deepcopy(placement_slice.tileinfo)
    slice_source_ids = list(placement_slice.source_mapping_layer_ids)
    slice_in_distance = placement_slice.inLayer_distance.copy()
    slice_trans_distance = placement_slice.transLayer_distance.copy()

    prefixed_slice = placement_slice.with_context_prefix(prefixed_netstruct)

    assert prefixed_slice is not placement_slice
    assert prefixed_slice.net == prefixed_netstruct
    assert prefixed_slice.layer_num == 2
    assert prefixed_slice.source_mapping_layer_ids == [-1, 1]
    assert len(prefixed_slice.tileinfo) == 2
    assert placement.tcg.layer_tileinfo == resident_tileinfo
    assert np.array_equal(placement.tcg.inLayer_distance, resident_in_distance)
    assert np.array_equal(placement.tcg.transLayer_distance, resident_trans_distance)
    assert placement_slice.tileinfo == slice_tileinfo
    assert placement_slice.source_mapping_layer_ids == slice_source_ids
    assert np.array_equal(placement_slice.inLayer_distance, slice_in_distance)
    assert np.array_equal(placement_slice.transLayer_distance, slice_trans_distance)


def test_slice_runs_original_models_without_constructing_a_new_tcg(tmp_path, monkeypatch):
    from MNSIM.Area_Model.Model_Area import Model_area
    from MNSIM.Energy_Model.Model_energy import Model_energy
    from MNSIM.Latency_Model.Model_latency import Model_latency
    from MNSIM.Power_Model.Model_inference_power import Model_inference_power

    sim_config_path = _sim_config_path(tmp_path, tile_num="2,1")
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=sim_config_path,
        weighted_scopes=_two_scopes(),
        expected_used_xbars=4,
    )
    request_netstruct = build_netstruct([_conv_layer_info()])
    placement_slice = placement.slice_for("R0.S0", 0, request_netstruct)
    tcg_calls = []

    def fail_if_constructed(*args, **kwargs):
        tcg_calls.append(args)
        raise AssertionError("a slice must reuse its resident TCG placement")

    monkeypatch.setattr(Tile_connection_graph, "TCG", fail_if_constructed)

    latency = Model_latency(
        request_netstruct,
        sim_config_path,
        TCG_mapping=placement_slice,
    )
    latency.calculate_model_latency()
    area = Model_area(
        request_netstruct,
        sim_config_path,
        TCG_mapping=placement_slice,
    )
    power = Model_inference_power(
        request_netstruct,
        sim_config_path,
        TCG_mapping=placement_slice,
    )
    energy = Model_energy(
        request_netstruct,
        sim_config_path,
        model_latency=latency,
        model_power=power,
        TCG_mapping=placement_slice,
    )

    assert max(latency.finish_time[0]) >= 0.0
    assert area.arch_total_area > 0.0
    assert power.arch_total_power > 0.0
    assert energy.arch_total_energy >= 0.0
    assert tcg_calls == []


def test_same_step_slices_follow_explicit_scope_indexes(tmp_path):
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
        weighted_scopes=_same_step_reordered_scopes(),
        expected_used_xbars=4,
    )
    request_netstruct = build_netstruct([_conv_layer_info()])

    assert placement.mapping_layer_ids_by_step == {"R0.S0": [1, 0]}
    assert placement.slice_for("R0.S0", 0, request_netstruct).source_mapping_layer_ids == [1]
    assert placement.slice_for("R0.S0", 1, request_netstruct).source_mapping_layer_ids == [0]
    assert placement.slice_for("R0.S0", 0, request_netstruct).tileinfo[0]["max_column"] == placement.tcg.layer_tileinfo[1]["max_column"]
    assert placement.slice_for("R0.S0", 1, request_netstruct).tileinfo[0]["max_column"] == placement.tcg.layer_tileinfo[0]["max_column"]


@pytest.mark.parametrize(
    ("step_id", "scope_index"),
    (("R0.UNKNOWN", 0), ("R0.S0", True), ("R0.S0", -1), ("R0.S0", 1)),
)
def test_slice_for_rejects_unknown_or_invalid_scope_requests(
    tmp_path,
    step_id,
    scope_index,
):
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
        weighted_scopes=_two_scopes(),
        expected_used_xbars=4,
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        placement.slice_for(step_id, scope_index, build_netstruct([_conv_layer_info()]))

    assert exc_info.value.code == "TCG_ROUND_PLACEMENT_INCONSISTENT"
    assert step_id in str(exc_info.value)
    assert repr(scope_index) in str(exc_info.value)


@pytest.mark.parametrize(
    ("active_physical_xbar_count", "expected_used_xbars"),
    (
        (True, 2),
        (1.5, 2),
        (-1, 2),
        (0, 2),
        (2, True),
        (2, 1.5),
        (2, -1),
        (2, 0),
    ),
)
def test_round_placement_rejects_malformed_active_xbar_counts_before_tcg(
    tmp_path,
    monkeypatch,
    active_physical_xbar_count,
    expected_used_xbars,
):
    tcg_calls = []
    monkeypatch.setattr(
        Tile_connection_graph,
        "TCG",
        lambda *args, **kwargs: tcg_calls.append(args),
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
            weighted_scopes=[
                _scope(
                    "R0.S0",
                    0,
                    _conv_layer_info(),
                    active_physical_xbar_count=active_physical_xbar_count,
                )
            ],
            expected_used_xbars=expected_used_xbars,
        )

    assert exc_info.value.code == "TCG_ACTIVE_XBAR_PROOF_MISMATCH"
    assert "R0" in str(exc_info.value)
    assert tcg_calls == []


@pytest.mark.parametrize("scope_indexes", ([0, 0], [0, 2]))
def test_round_placement_rejects_noncontiguous_scope_indexes_before_tcg(
    tmp_path,
    monkeypatch,
    scope_indexes,
):
    tcg_calls = []
    monkeypatch.setattr(
        Tile_connection_graph,
        "TCG",
        lambda *args, **kwargs: tcg_calls.append(args),
    )
    scopes = [
        _scope("R0.S0", scope_indexes[0], _conv_layer_info()),
        _scope("R0.S0", scope_indexes[1], _conv_layer_info()),
    ]

    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
            weighted_scopes=scopes,
            expected_used_xbars=4,
        )

    assert exc_info.value.code == "TCG_ROUND_PLACEMENT_INCONSISTENT"
    assert tcg_calls == []


def test_round_placement_rejects_phase1_scope_accounting_mismatch(tmp_path):
    with pytest.raises(Phase2Invalid) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_sim_config_path(tmp_path, tile_num="2,1"),
            weighted_scopes=[
                _scope("R0.S0", 0, _conv_layer_info())
            ],
            expected_used_xbars=4,
        )

    assert exc_info.value.code == "TCG_ACTIVE_XBAR_PROOF_MISMATCH"


def test_round_placement_rejects_physical_xbar_overflow_in_residency_mode(tmp_path):
    scopes = [
        _scope("R0.S0", 0, _one_pe_conv()),
        _scope("R0.S1", 0, _one_pe_conv()),
        _scope("R0.S2", 0, _one_pe_conv()),
        _scope("R0.S3", 0, _one_pe_conv()),
        _scope("R0.S4", 0, _one_pe_conv()),
    ]

    with pytest.raises(
        Phase2Invalid,
        match="active group count 5 exceeds hardware group capacity 4",
    ) as exc_info:
        prepare_round_placement(
            item_id="R0",
            sim_config_path=_round_residency_config(tmp_path),
            weighted_scopes=scopes,
            expected_used_xbars=10,
        )

    assert exc_info.value.code == "TCG_ACTIVE_XBAR_PROOF_MISMATCH"
