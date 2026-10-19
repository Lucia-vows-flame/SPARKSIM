from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from MNSIM.Interface.round_orchestrated_metrics import (
    ModelOperationCounts,
    count_model_operations,
    derive_hardware_rates,
)
from MNSIM.Interface.round_orchestrated_types import RunnerError
from MNSIM.Scheduling.round_types import LayerIR, ModelIR, SequenceSpec


def _layer(
    layer_id: str,
    layer_type: str,
    *,
    is_weighted: bool,
    shape: dict[str, Any],
) -> LayerIR:
    return LayerIR(
        layer_id=layer_id,
        layer_type=layer_type,
        is_weighted=is_weighted,
        stable_topological_index=0,
        input_tensor_ids=[],
        output_tensor_ids=[],
        input_layer_ids=[],
        output_consumer_ids=[],
        weight_ref_id=f"weight:{layer_id}" if is_weighted else None,
        shape=shape,
        backend_attrs={},
    )


def _model(*layers: LayerIR) -> ModelIR:
    return ModelIR(
        model_id="metrics-model",
        backend_kind="ann",
        sequence_spec=SequenceSpec(
            sequence_length=1,
            ordered_steps=False,
            cross_step_dependency=False,
            recurrent_state_ids=[],
        ),
        layers=list(layers),
    )


def _conv_shape(**overrides: Any) -> dict[str, Any]:
    shape = {
        "Inputchannel": 3,
        "Outputchannel": 4,
        "Kernelsize": 3,
        "Outputsize": [2, 3],
    }
    shape.update(overrides)
    return shape


def _assert_invalid_mac_estimate(model_ir: ModelIR) -> RunnerError:
    with pytest.raises(RunnerError) as error:
        count_model_operations(model_ir)

    assert error.value.code == "INVALID_MAC_ESTIMATE"
    return error.value


def test_counts_weighted_conv_and_fc_then_derives_mnsim_hardware_rates():
    model_ir = _model(
        _layer(
            "conv1",
            "conv",
            is_weighted=True,
            shape=_conv_shape(Depthwise="normal"),
        ),
        _layer(
            "fc1",
            "fc",
            is_weighted=True,
            shape={"Infeature": 5, "Outfeature": 7},
        ),
        _layer("relu1", "relu", is_weighted=False, shape={}),
    )

    counts = count_model_operations(model_ir)
    phase2_metrics = {
        "latency_s": 1e-6,
        "read_latency_s": 0.5e-6,
        "energy_j": 1e-6,
        "area_um2": 99,
    }

    derived = derive_hardware_rates(phase2_metrics, counts)

    assert counts == ModelOperationCounts(macs_total=683, ops_total=1366)
    assert derived["macs_total"] == 683
    assert derived["ops_total"] == 1366
    assert derived["ops_count_policy"] == "mac_as_2_ops"
    assert derived["network_workload_top_per_inference"] == pytest.approx(1.366e-9)
    assert derived["compute_tmacs_per_s"] == pytest.approx(0.001366)
    assert derived["compute_tops"] == pytest.approx(0.002732)
    assert derived["compute_throughput_tops"] == pytest.approx(0.002732)
    assert derived["tmacs_per_s"] == pytest.approx(0.000683)
    assert derived["tops"] == pytest.approx(0.001366)
    assert derived["end_to_end_throughput_tops"] == pytest.approx(0.001366)
    assert derived["tops_per_w"] == pytest.approx(0.001366)
    assert derived["modeled_energy_efficiency_tops_per_w"] == pytest.approx(0.001366)
    assert derived["area_um2"] == 99
    assert phase2_metrics == {
        "latency_s": 1e-6,
        "read_latency_s": 0.5e-6,
        "energy_j": 1e-6,
        "area_um2": 99,
    }


def test_network_workload_is_independent_of_hardware_latency_and_energy():
    counts = ModelOperationCounts(macs_total=5_000_000, ops_total=10_000_000)

    small = derive_hardware_rates(
        {"latency_s": 10.0, "energy_j": 2.0, "energy_accounting_complete": True},
        counts,
    )
    large = derive_hardware_rates(
        {"latency_s": 0.01, "energy_j": 0.5, "energy_accounting_complete": True},
        counts,
    )

    assert small["network_workload_top_per_inference"] == pytest.approx(1e-5)
    assert large["network_workload_top_per_inference"] == pytest.approx(1e-5)
    assert small["end_to_end_throughput_tops"] < large["end_to_end_throughput_tops"]


@pytest.mark.parametrize(
    ("depthwise", "kernel", "effective_input_channels"),
    [
        (None, 3, 3),
        ("normal", [2, 3], 3),
        ("point", 3, 3),
        ("separable", [2, 3], 1),
    ],
)
def test_conv_depthwise_values_use_the_expected_effective_input_channels(
    depthwise, kernel, effective_input_channels
):
    shape = _conv_shape(Kernelsize=kernel)
    if depthwise is not None:
        shape["Depthwise"] = depthwise

    counts = count_model_operations(
        _model(_layer("conv1", "conv", is_weighted=True, shape=shape))
    )

    kernel_height, kernel_width = (kernel, kernel) if isinstance(kernel, int) else kernel
    expected_macs = 2 * 3 * 4 * effective_input_channels * kernel_height * kernel_width
    assert counts == ModelOperationCounts(macs_total=expected_macs, ops_total=2 * expected_macs)


def test_rejects_unsupported_depthwise_value_with_stable_runner_error():
    error = _assert_invalid_mac_estimate(
        _model(
            _layer(
                "conv1",
                "conv",
                is_weighted=True,
                shape=_conv_shape(Depthwise="grouped"),
            )
        )
    )

    assert error.detail.startswith("unsupported Depthwise value")


@pytest.mark.parametrize(
    "depthwise",
    [
        np.array(["normal"]),
        np.array(["normal", "point"]),
    ],
)
def test_rejects_numpy_array_depthwise_values_with_stable_runner_error(depthwise):
    error = _assert_invalid_mac_estimate(
        _model(
            _layer(
                "conv1",
                "conv",
                is_weighted=True,
                shape=_conv_shape(Depthwise=depthwise),
            )
        )
    )

    assert error.detail.startswith("unsupported Depthwise value")


def test_accepts_numpy_integral_conv_dimensions():
    counts = count_model_operations(
        _model(
            _layer(
                "conv1",
                "conv",
                is_weighted=True,
                shape={
                    "Inputchannel": np.int64(3),
                    "Outputchannel": np.int64(4),
                    "Kernelsize": [np.int64(2), np.int64(3)],
                    "Outputsize": [np.int64(2), np.int64(3)],
                },
            )
        )
    )

    assert counts == ModelOperationCounts(macs_total=432, ops_total=864)


@pytest.mark.parametrize(
    "shape",
    [
        _conv_shape(Kernelsize=[]),
        _conv_shape(Kernelsize=[3]),
        _conv_shape(Kernelsize=[3, 3, 3]),
        _conv_shape(Outputsize=[]),
        _conv_shape(Outputsize=[2]),
        _conv_shape(Outputsize=[2, 3, 4]),
        _conv_shape(Inputchannel=0),
        _conv_shape(Inputchannel=-1),
        _conv_shape(Inputchannel=1.5),
        _conv_shape(Inputchannel=True),
        _conv_shape(Outputchannel=0),
        _conv_shape(Outputchannel=-1),
        _conv_shape(Outputchannel=1.5),
        _conv_shape(Outputchannel=True),
        _conv_shape(Kernelsize=0),
        _conv_shape(Kernelsize=-1),
        _conv_shape(Kernelsize=1.5),
        _conv_shape(Kernelsize=True),
        _conv_shape(Kernelsize=[2, True]),
        _conv_shape(Outputsize=[0, 3]),
        _conv_shape(Outputsize=[-1, 3]),
        _conv_shape(Outputsize=[1.5, 3]),
        _conv_shape(Outputsize=[True, 3]),
    ],
)
def test_rejects_malformed_or_non_positive_conv_shapes_without_leaking_error_types(shape):
    _assert_invalid_mac_estimate(
        _model(_layer("conv1", "conv", is_weighted=True, shape=shape))
    )


@pytest.mark.parametrize(
    "shape",
    [
        {},
        {"Infeature": 0, "Outfeature": 7},
        {"Infeature": -1, "Outfeature": 7},
        {"Infeature": 1.5, "Outfeature": 7},
        {"Infeature": True, "Outfeature": 7},
        {"Infeature": 5, "Outfeature": 0},
        {"Infeature": 5, "Outfeature": -1},
        {"Infeature": 5, "Outfeature": 1.5},
        {"Infeature": 5, "Outfeature": True},
    ],
)
def test_rejects_invalid_fc_shapes_without_leaking_error_types(shape):
    _assert_invalid_mac_estimate(
        _model(_layer("fc1", "fc", is_weighted=True, shape=shape))
    )


def test_rejects_weighted_layer_types_other_than_conv_and_fc():
    error = _assert_invalid_mac_estimate(
        _model(_layer("projection", "fc_like_projection", is_weighted=True, shape={}))
    )

    assert "unsupported weighted layer type" in error.detail


def test_non_weighted_layers_contribute_no_macs_or_ops():
    counts = count_model_operations(
        _model(
            _layer("relu1", "relu", is_weighted=False, shape={}),
            _layer("pool1", "pooling", is_weighted=False, shape={}),
        )
    )

    assert counts == ModelOperationCounts(macs_total=0, ops_total=0)


@pytest.mark.parametrize(
    ("latency_s", "energy_j", "expected_tmacs", "expected_tops", "expected_tops_per_w"),
    [
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (-1.0, -2.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 2e-11),
        (1.0, 0.0, 1e-11, 2e-11, 0.0),
        (1.0, -1.0, 1e-11, 2e-11, 0.0),
    ],
)
def test_zero_or_negative_phase2_latency_and_energy_produce_zero_derived_rates(
    latency_s, energy_j, expected_tmacs, expected_tops, expected_tops_per_w
):
    counts = ModelOperationCounts(macs_total=10, ops_total=20)
    phase2_metrics = {"latency_s": latency_s, "energy_j": energy_j, "area_um2": 12.0}

    derived = derive_hardware_rates(phase2_metrics, counts)

    assert derived["tmacs_per_s"] == pytest.approx(expected_tmacs)
    assert derived["tops"] == pytest.approx(expected_tops)
    assert derived["compute_tmacs_per_s"] == pytest.approx(expected_tmacs)
    assert derived["compute_tops"] == pytest.approx(expected_tops)
    assert derived["tops_per_w"] == pytest.approx(expected_tops_per_w)
    assert derived["area_um2"] == 12.0
    assert phase2_metrics == {"latency_s": latency_s, "energy_j": energy_j, "area_um2": 12.0}
