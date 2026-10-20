import configparser

import pytest

import MNSIM.Scheduling.phase2.mnsim_fragment_adapter as fragment_adapter_module
from MNSIM.Scheduling.phase2 import (
    INVALID_PHYSICAL_XBAR_RANGE,
    PHYSICAL_XBAR_UNIT_MISMATCH,
    PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH,
    SELECTED_FRAGMENT_COUNT_MISMATCH,
    MnsimFragmentAdapter,
    Phase2Invalid,
)
from MNSIM.Scheduling.round_types import PrepassRef, SelectedEntry


def _fingerprint(
    *,
    xbar_size=(128, 128),
    device_level=2,
    weight_precision=8,
    polarity=2,
):
    return (
        f"xbar={xbar_size[0]}x{xbar_size[1]};"
        f"device={device_level};weight={weight_precision};polarity={polarity}"
    )


def _sim_config(
    tmp_path,
    *,
    xbar_size=(128, 128),
    device_level=2,
    weight_precision=8,
    polarity=2,
):
    parser = configparser.ConfigParser()
    parser["Crossbar level"] = {"Xbar_Size": ",".join(map(str, xbar_size))}
    parser["Device level"] = {"Device_Level": str(device_level)}
    parser["Process element level"] = {"Xbar_Polarity": str(polarity)}
    parser["Algorithm Configuration"] = {"Weight_precision": str(weight_precision)}
    path = tmp_path / (
        f"SimConfig-{xbar_size[0]}x{xbar_size[1]}-d{device_level}"
        f"-w{weight_precision}-p{polarity}.ini"
    )
    with path.open("w", encoding="UTF-8") as config_file:
        parser.write(config_file)
    return str(path)


def _prepass_ref(
    layer_id,
    layer_type,
    shape,
    *,
    resource_version="mnsim20_physical_xbar_v2",
    sim_config_fingerprint=None,
):
    if sim_config_fingerprint is None:
        sim_config_fingerprint = _fingerprint()
    return PrepassRef(
        ref_id=f"m0:{layer_id}:mnsim20_physical_xbar_v2",
        model_id="m0",
        layer_id=layer_id,
        layer_type=layer_type,
        resource_version=resource_version,
        sim_config_fingerprint=sim_config_fingerprint,
        layer_config_fingerprint=f"{layer_type}:test",
        weight_ref_id=f"weight:{layer_id}",
        shape_signature=dict(shape),
    )


def _selected_entry(
    layer_id="L000_fc",
    *,
    layer_type="fc",
    start=0,
    end=2,
    used=None,
    shape=None,
    resource_version="mnsim20_physical_xbar_v2",
    sim_config_fingerprint=None,
):
    if shape is None:
        shape = {
            "Infeature": 128,
            "Outfeature": 128,
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        }
    if used is None:
        used = end - start
    return SelectedEntry(
        layer_id=layer_id,
        layer_type=layer_type,
        xbar_range_start=start,
        xbar_range_end=end,
        used_xbars=used,
        layer_complete_in_round=end == 14,
        prepass_ref=_prepass_ref(
            layer_id,
            layer_type,
            shape,
            resource_version=resource_version,
            sim_config_fingerprint=sim_config_fingerprint,
        ),
    )


def test_selects_one_complete_differential_plane_from_two_arrays(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})

    layout = adapter.select_entry(_selected_entry())

    assert layout.active_group_count == 1
    assert layout.active_physical_xbar_count == 2
    assert len(layout.fragments) == 1
    assert layout.fragments[0].weight_plane_index == 0
    assert layout.fragments[0].polarity_count == 2
    assert layout.fragments[0].physical_xbar_count == 2
    assert layout.fragments[0].fragment_id == (
        "L000_fc|in:0:128|out:0:128|plane:0"
    )


def test_reuses_one_layer_inventory_across_selected_physical_ranges(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})
    first = _selected_entry(start=0, end=2)
    second = _selected_entry(start=2, end=4)

    first_inventory = adapter.inventory_for(first)
    second_inventory = adapter.inventory_for(second)

    assert first_inventory is second_inventory


def test_accepts_large_range_not_divisible_by_full_precision_block(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 1176})
    entry = _selected_entry(
        start=0,
        end=1152,
        used=1152,
        shape={"Infeature": 128, "Outfeature": 10752, "Weightbit": 8},
    )

    layout = adapter.select_entry(entry)

    assert layout.active_physical_xbar_count == 1152
    assert layout.active_group_count == 576
    assert len(layout.fragments) == 576
    assert layout.fragments[-1].fragment_id == (
        "L000_fc|in:0:128|out:10496:10624|plane:1"
    )


def test_rejects_half_of_a_differential_pair(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(_selected_entry(start=1, end=2, used=1))

    assert exc_info.value.code == INVALID_PHYSICAL_XBAR_RANGE
    assert "[1, 2)" in exc_info.value.detail


def test_conv_inventory_keeps_full_kernel_and_final_channel_chunks(tmp_path):
    shape = {
        "Inputchannel": 15,
        "Outputchannel": 129,
        "Kernelsize": 3,
        "Inputsize": [16, 16],
        "Outputsize": [14, 14],
        "Weightbit": 2,
    }
    adapter = MnsimFragmentAdapter(
        _sim_config(tmp_path, device_level=4, weight_precision=8, polarity=1),
        {"L000_conv": 4},
    )
    sim_config_fingerprint = _fingerprint(
        device_level=4,
        weight_precision=8,
        polarity=1,
    )

    inventory = adapter.inventory_for(
        _selected_entry(
            "L000_conv",
            layer_type="conv",
            start=0,
            end=4,
            shape=shape,
            sim_config_fingerprint=sim_config_fingerprint,
        )
    )

    assert [
        (fragment.input_range, fragment.output_range, fragment.weight_plane_index)
        for fragment in inventory.fragments
    ] == [
        ((0, 14), (0, 128), 0),
        ((0, 14), (128, 129), 0),
        ((14, 15), (0, 128), 0),
        ((14, 15), (128, 129), 0),
    ]
    assert all(fragment.full_input_range == (0, 15) for fragment in inventory.fragments)
    assert all(fragment.full_output_range == (0, 129) for fragment in inventory.fragments)

    layout = adapter.select_entry(
        _selected_entry(
            "L000_conv",
            layer_type="conv",
            start=0,
            end=4,
            shape=shape,
            sim_config_fingerprint=sim_config_fingerprint,
        )
    )
    assert layout.native_layer_info["Kernelsize"] == 3
    assert layout.native_layer_info["type"] == "conv"


def test_fc_inventory_uses_contiguous_features_and_partial_final_chunks(tmp_path):
    shape = {
        "Infeature": 129,
        "Outfeature": 129,
        "Weightbit": 5,
        "_phase1_private_field": "excluded",
    }
    adapter = MnsimFragmentAdapter(
        _sim_config(tmp_path, device_level=4, weight_precision=8, polarity=1),
        {"L000_fc": 12},
    )
    entry = _selected_entry(
        start=0,
        end=12,
        shape=shape,
        sim_config_fingerprint=_fingerprint(
            device_level=4,
            weight_precision=8,
            polarity=1,
        ),
    )

    inventory = adapter.inventory_for(entry)
    layout = adapter.select_entry(entry)

    assert len(inventory.fragments) == 12
    assert [fragment.weight_plane_index for fragment in inventory.fragments[:4]] == [
        0,
        1,
        2,
        0,
    ]
    assert {
        (fragment.input_range, fragment.output_range)
        for fragment in inventory.fragments
    } == {
        ((0, 128), (0, 128)),
        ((0, 128), (128, 129)),
        ((128, 129), (0, 128)),
        ((128, 129), (128, 129)),
    }
    assert layout.native_layer_info == {
        "Infeature": 129,
        "Outfeature": 129,
        "Weightbit": 5,
        "type": "fc",
    }


def test_native_weightbit_overrides_config_and_config_precision_is_fallback(tmp_path):
    config_path = _sim_config(tmp_path, weight_precision=8)
    native_adapter = MnsimFragmentAdapter(config_path, {"native": 6})
    fallback_adapter = MnsimFragmentAdapter(config_path, {"fallback": 16})

    native_inventory = native_adapter.inventory_for(
        _selected_entry("native", shape={"Infeature": 128, "Outfeature": 128, "Weightbit": 4})
    )
    fallback_inventory = fallback_adapter.inventory_for(
        _selected_entry("fallback", shape={"Infeature": 128, "Outfeature": 128})
    )
    fallback_layout = fallback_adapter.select_entry(
        _selected_entry("fallback", shape={"Infeature": 128, "Outfeature": 128})
    )

    assert [fragment.weight_plane_index for fragment in native_inventory.fragments] == [0, 1, 2]
    assert [fragment.weight_plane_index for fragment in fallback_inventory.fragments] == list(range(8))
    assert fallback_layout.native_layer_info["Weightbit"] == 9


def test_native_layer_info_carries_config_weightbit_when_shape_omits_it(tmp_path):
    adapter = MnsimFragmentAdapter(
        _sim_config(tmp_path, device_level=4, weight_precision=8, polarity=1),
        {"fallback": 4},
    )

    layout = adapter.select_entry(
        _selected_entry(
            "fallback",
            start=0,
            end=4,
            shape={"Infeature": 128, "Outfeature": 128},
            sim_config_fingerprint=_fingerprint(
                device_level=4,
                weight_precision=8,
                polarity=1,
            ),
        )
    )

    assert layout.native_layer_info["Weightbit"] == 8


@pytest.mark.parametrize(
    ("device_level", "expected_planes", "expected_arrays"),
    [(2, 8, 16), (4, 4, 8), (8, 3, 6), (16, 2, 4), (256, 1, 2)],
)
def test_fragment_inventory_uses_generic_device_bit_width(
    tmp_path,
    device_level,
    expected_planes,
    expected_arrays,
):
    adapter = MnsimFragmentAdapter(
        _sim_config(tmp_path, device_level=device_level, weight_precision=8),
        {"generic": expected_arrays},
    )
    entry = _selected_entry(
        "generic",
        start=0,
        end=2,
        used=2,
        shape={"Infeature": 128, "Outfeature": 128, "Weightbit": 9},
        sim_config_fingerprint=_fingerprint(device_level=device_level),
    )

    inventory = adapter.inventory_for(entry)
    layout = adapter.select_entry(entry)

    assert [fragment.weight_plane_index for fragment in inventory.fragments] == list(
        range(expected_planes)
    )
    assert inventory.required_xbar_count == expected_arrays
    assert layout.active_group_count == 1
    assert layout.active_physical_xbar_count == 2


@pytest.mark.parametrize("device_level", [1, 3, 6, 12])
def test_fragment_adapter_rejects_non_bit_precise_device_levels(tmp_path, device_level):
    with pytest.raises(Phase2Invalid) as exc_info:
        MnsimFragmentAdapter(
            _sim_config(tmp_path, device_level=device_level),
            {"L000_fc": 2},
        )

    assert exc_info.value.code == "INVALID_MNSIM_FRAGMENT_CONFIG"
    assert "Device_Level" in exc_info.value.detail


def test_descriptor_count_mismatch_fails_before_selection(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 12})

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(_selected_entry())

    assert exc_info.value.code == PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH
    assert "inventory" in exc_info.value.detail
    assert "required" in exc_info.value.detail


def test_inventory_ids_are_duplicate_free(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 56})
    inventory = adapter.inventory_for(
        _selected_entry(
            start=0,
            end=56,
            shape={"Infeature": 256, "Outfeature": 256, "Weightbit": 8},
        )
    )

    fragment_ids = [fragment.fragment_id for fragment in inventory.fragments]

    assert len(fragment_ids) == len(set(fragment_ids)) == 28


def test_selected_used_xbars_mismatch_has_distinct_failure_code(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(_selected_entry(start=0, end=2, used=4))

    assert exc_info.value.code == SELECTED_FRAGMENT_COUNT_MISMATCH
    assert "used_xbars" in exc_info.value.detail


@pytest.mark.parametrize(
    ("resource_version", "sim_config_fingerprint"),
    [
        ("", _fingerprint()),
        ("mnsim20_formula_v1", _fingerprint()),
        ("mnsim20_physical_xbar_v2", ""),
        ("mnsim20_physical_xbar_v2", "shape-only-test"),
        (
            "mnsim20_physical_xbar_v2",
            _fingerprint(device_level=4),
        ),
    ],
    ids=[
        "missing-version",
        "stale-version",
        "missing-fingerprint",
        "malformed-fingerprint",
        "stale-fingerprint",
    ],
)
def test_rejects_missing_malformed_or_stale_prepass_identity_before_selection(
    tmp_path,
    resource_version,
    sim_config_fingerprint,
):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})
    entry = _selected_entry(
        start=1,
        end=2,
        used=1,
        resource_version=resource_version,
        sim_config_fingerprint=sim_config_fingerprint,
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(entry)

    assert exc_info.value.code == PHYSICAL_XBAR_UNIT_MISMATCH


def test_rejects_stale_xbar_geometry_even_when_total_inventory_count_matches(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 28})
    entry = _selected_entry(
        start=0,
        end=2,
        shape={"Infeature": 128, "Outfeature": 256, "Weightbit": 8},
        sim_config_fingerprint=_fingerprint(xbar_size=(64, 256)),
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(entry)

    assert exc_info.value.code == PHYSICAL_XBAR_UNIT_MISMATCH
    assert "xbar" in exc_info.value.detail


def test_layout_native_layer_info_is_deeply_immutable_and_serializes_to_json_values(
    tmp_path,
):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})
    layout = adapter.select_entry(
        _selected_entry(
            shape={
                "Infeature": 128,
                "Outfeature": 128,
                "Weightbit": 8,
                "Inputsize": [1, 128],
                "metadata": {"axis": [0, 1]},
            }
        )
    )

    with pytest.raises(TypeError):
        layout.native_layer_info["Weightbit"] = 4
    with pytest.raises(TypeError):
        layout.native_layer_info["metadata"]["axis"] = [1, 0]
    with pytest.raises(TypeError):
        layout.native_layer_info["metadata"]["axis"][0] = 1
    with pytest.raises(TypeError):
        dict.__setitem__(layout.native_layer_info, "Weightbit", 4)
    with pytest.raises(TypeError):
        list.append(layout.native_layer_info["metadata"]["axis"], 2)

    serialized = layout.to_dict()

    assert type(serialized["native_layer_info"]) is dict
    assert type(serialized["native_layer_info"]["Inputsize"]) is list
    assert type(serialized["native_layer_info"]["metadata"]) is dict
    serialized["native_layer_info"]["metadata"]["axis"][0] = 1
    assert layout.native_layer_info["metadata"]["axis"][0] == 0


def test_descriptor_mismatch_is_proved_before_fragment_construction(tmp_path, monkeypatch):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 12})

    def fail_fragment_construction(**_kwargs):
        pytest.fail("fragment construction occurred before the inventory-count proof")

    monkeypatch.setattr(
        fragment_adapter_module,
        "MnsimGroupFragment",
        fail_fragment_construction,
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.inventory_for(_selected_entry())

    assert exc_info.value.code == PREPASS_MNSIM_FRAGMENT_COUNT_MISMATCH


def test_rejects_boolean_used_xbars_for_single_polarity_interval(tmp_path):
    adapter = MnsimFragmentAdapter(
        _sim_config(tmp_path, device_level=4, weight_precision=2, polarity=1),
        {"L000_fc": 1},
    )
    entry = _selected_entry(
        start=0,
        end=1,
        used=True,
        shape={"Infeature": 128, "Outfeature": 128, "Weightbit": 2},
        sim_config_fingerprint=_fingerprint(
            device_level=4,
            weight_precision=2,
            polarity=1,
        ),
    )

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(entry)

    assert exc_info.value.code == SELECTED_FRAGMENT_COUNT_MISMATCH
    assert "used_xbars" in exc_info.value.detail


def test_rejects_string_used_xbars_for_selected_entry(tmp_path):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(_selected_entry(start=0, end=2, used="2"))

    assert exc_info.value.code == SELECTED_FRAGMENT_COUNT_MISMATCH
    assert "used_xbars" in exc_info.value.detail


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("0", 2),
        (0, "2"),
        (0.0, 2),
        (0, 2.0),
        (False, 2),
        (0, True),
    ],
    ids=[
        "string-start",
        "string-end",
        "float-start",
        "float-end",
        "bool-start",
        "bool-end",
    ],
)
def test_rejects_non_integral_selected_entry_range_bounds(tmp_path, start, end):
    adapter = MnsimFragmentAdapter(_sim_config(tmp_path), {"L000_fc": 14})

    with pytest.raises(Phase2Invalid) as exc_info:
        adapter.select_entry(_selected_entry(start=start, end=end, used=2))

    assert exc_info.value.code == INVALID_PHYSICAL_XBAR_RANGE
