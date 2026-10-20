import pytest

from MNSIM.Scheduling.phase1.config import build_hardware_projection
from MNSIM.Scheduling.phase1.resource_prepass import (
    RESOURCE_VERSION,
    build_layer_resource_descriptors,
)
from MNSIM.Scheduling.round_types import LayerIR, ModelIR, Phase1Invalid, SequenceSpec


def _hw_config(**overrides):
    config = {
        "Xbar_Size": [128, 128],
        "Tile_Num": [2, 2],
        "PE_Num": [2, 2],
        "Group_Num": 1,
        "Device_Level": 4,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }
    config.update(overrides)
    return config


def _model(layers):
    return ModelIR(
        model_id="m0",
        backend_kind="ann",
        sequence_spec=SequenceSpec(1, False, False, []),
        layers=layers,
    )


def _weighted(layer_id, layer_type, shape, index=0):
    return LayerIR(
        layer_id=layer_id,
        layer_type=layer_type,
        is_weighted=True,
        stable_topological_index=index,
        input_tensor_ids=["input"],
        output_tensor_ids=[f"{layer_id}:out"],
        input_layer_ids=[],
        output_consumer_ids=[],
        weight_ref_id=f"w:{layer_id}",
        shape=shape,
        backend_attrs={},
    )


def test_fc_resource_prepass_counts_spatial_precision_and_polarity():
    hw = build_hardware_projection(_hw_config())
    model = _model([_weighted("FC1", "fc", {"Infeature": 192, "Outfeature": 129})])

    [desc] = build_layer_resource_descriptors(model, hw)

    assert desc.layer_id == "FC1"
    assert desc.required_xbar_count == 2 * 2 * 4 * 2
    assert desc.prepass_ref.layer_id == "FC1"
    assert desc.prepass_ref.resource_version == "mnsim20_physical_xbar_v2"


def test_differential_weight_precision_matches_native_signed_weightbit_planes():
    hw = build_hardware_projection(
        _hw_config(Device_Level=2, Weight_precision=8, Xbar_Polarity=2)
    )
    model = _model(
        [
            _weighted(
                "FC1",
                "fc",
                {
                    "Infeature": 128,
                    "Outfeature": 128,
                    # Original MNSIM records sign plus eight magnitude bits.
                    "Weightbit": 9,
                },
            )
        ]
    )

    [desc] = build_layer_resource_descriptors(model, hw)

    assert desc.required_xbar_count == 16


@pytest.mark.parametrize(
    ("device_level", "expected_arrays"),
    [(2, 16), (4, 8), (8, 6), (16, 4), (256, 2)],
)
def test_fc_resource_prepass_uses_generic_device_bit_width(
    device_level,
    expected_arrays,
):
    hardware = build_hardware_projection(
        _hw_config(Device_Level=device_level, Weight_precision=8, Xbar_Polarity=2)
    )
    model = _model(
        [
            _weighted(
                "FC1",
                "fc",
                {"Infeature": 128, "Outfeature": 128, "Weightbit": 9},
            )
        ]
    )

    [descriptor] = build_layer_resource_descriptors(model, hardware)

    assert descriptor.required_xbar_count == expected_arrays


def test_fc_resource_prepass_uses_native_weightbit_when_present():
    hw = build_hardware_projection(
        _hw_config(Device_Level=2, Weight_precision=8, Xbar_Polarity=2)
    )
    model = _model(
        [
            _weighted(
                "FC1",
                "fc",
                {"Infeature": 128, "Outfeature": 128, "Weightbit": 4},
            )
        ]
    )

    [desc] = build_layer_resource_descriptors(model, hw)

    # Original MNSIM's differential path reserves one sign bit, leaving three
    # one-bit magnitude planes and their positive/negative physical pairs.
    assert desc.required_xbar_count == 1 * 1 * 3 * 2


def test_conv_resource_prepass_preserves_kernel_per_input_channel():
    hw = build_hardware_projection(_hw_config())
    model = _model([_weighted("Conv1", "conv", {"Inputchannel": 20, "Outputchannel": 130, "Kernelsize": 3})])

    [desc] = build_layer_resource_descriptors(model, hw)

    input_channels_per_xbar = 128 // (3 * 3)
    row_split_count = 2  # ceil(20 / 14)
    output_subset_count = 2  # ceil(130 / 128)
    assert input_channels_per_xbar == 14
    assert desc.required_xbar_count == row_split_count * output_subset_count * 4 * 2


def test_rnn_like_projection_uses_fc_formula():
    hw = build_hardware_projection(_hw_config())
    model = _model([_weighted("RNNProj", "fc_like_projection", {"Infeature": 192, "Outfeature": 128})])

    [desc] = build_layer_resource_descriptors(model, hw)

    assert desc.required_xbar_count == 2 * 1 * 4 * 2


def test_single_polarity_uses_full_weight_precision_bits():
    hw = build_hardware_projection(_hw_config(Xbar_Polarity=1))
    model = _model([_weighted("FC1", "fc", {"Infeature": 128, "Outfeature": 128})])

    [desc] = build_layer_resource_descriptors(model, hw)

    assert desc.required_xbar_count == 1 * 1 * 4 * 1


def test_prepass_ref_uses_physical_xbar_version_without_phase1_private_metadata():
    hw = build_hardware_projection(_hw_config())
    model = _model([_weighted("FC1", "fc", {"Infeature": 192, "Outfeature": 129})])

    [desc] = build_layer_resource_descriptors(model, hw)

    assert RESOURCE_VERSION == "mnsim20_physical_xbar_v2"
    assert desc.prepass_ref.resource_version == RESOURCE_VERSION
    assert desc.prepass_ref.shape_signature == {"Infeature": 192, "Outfeature": 129}
    assert not any(key.startswith("_phase1_") for key in desc.prepass_ref.shape_signature)


def test_conv_kernel_larger_than_xbar_row_is_invalid():
    hw = build_hardware_projection(_hw_config(Xbar_Size=[8, 128]))
    model = _model([_weighted("ConvBad", "conv", {"Inputchannel": 1, "Outputchannel": 1, "Kernelsize": 3})])

    try:
        build_layer_resource_descriptors(model, hw)
    except Phase1Invalid as exc:
        assert exc.code == "INVALID_LAYER_SHAPE"
        assert "Kernelsize" in exc.detail
    else:
        raise AssertionError("expected Phase1Invalid")
