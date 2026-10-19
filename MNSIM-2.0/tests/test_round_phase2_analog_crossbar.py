import configparser

import numpy as np
import pytest

from MNSIM.Scheduling.phase2.analog_crossbar import (
    AnalogCrossbarNumericalEngine,
    AnalogCrossbarNumericalError,
    compute_differential_currents,
)


def _sim_config(
    tmp_path, *, read_voltage="0.5", resistances="1000,500,250", polarity="2"
):
    parser = configparser.ConfigParser()
    parser["Device level"] = {
        "Device_Level": str(len(resistances.split(","))),
        "Device_Resistance_Sort": "0",
        "Device_Resistance": resistances,
        "Read_Voltage": read_voltage,
    }
    parser["Process element level"] = {"Xbar_Polarity": polarity}
    parser["Algorithm Configuration"] = {"Weight_precision": "2"}
    path = tmp_path / "SimConfig.ini"
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return path


def test_compute_differential_currents_is_exact_v_times_g_in_amperes():
    ipos, ineg = compute_differential_currents(
        np.array([2.0, 4.0]),
        0.25,
        np.array([[1.0e-3, 2.0e-3], [3.0e-3, 4.0e-3]]),
        np.array([[5.0e-3, 6.0e-3], [7.0e-3, 8.0e-3]]),
    )

    np.testing.assert_allclose(ipos, [0.0035, 0.005])
    np.testing.assert_allclose(ineg, [0.0095, 0.011])
    assert ipos.dtype.kind == "f"
    assert ineg.dtype.kind == "f"


@pytest.mark.parametrize(
    "args",
    [
        ([1.0, np.nan], 1.0, [[1.0]], [[1.0]]),
        ([1.0], np.inf, [[1.0]], [[1.0]]),
        ([1.0], 1.0, [[np.nan]], [[1.0]]),
        ([1.0], 1.0, [[1.0]], [[-1.0]]),
    ],
)
def test_compute_differential_currents_rejects_nonfinite_or_negative_values(args):
    with pytest.raises(AnalogCrossbarNumericalError):
        compute_differential_currents(*args)


def test_compute_differential_currents_rejects_shape_mismatch():
    with pytest.raises(AnalogCrossbarNumericalError, match="shape"):
        compute_differential_currents([1.0, 2.0], 1.0, [[1.0]], [[1.0]])


def test_engine_looks_up_signed_weight_codes_on_local_conductance_table(tmp_path):
    engine = AnalogCrossbarNumericalEngine.from_sim_config(
        _sim_config(tmp_path), weight_codes=np.array([[1, -2], [2, 0]])
    )

    ipos, ineg, audit = engine.compute(
        input_values=np.array([2.0, 4.0]),
        weight_codes_by_layer={"L0": np.array([[1, -2], [2, 0]])},
        output_neuron_count=2,
    )

    # G(0)=1e-3 S, G(1)=2e-3 S, G(2)=4e-3 S and V=[1, 2] V.
    np.testing.assert_allclose(ipos, [0.010, 0.003])
    np.testing.assert_allclose(ineg, [0.003, 0.006])
    assert audit["read_voltage_v"] == pytest.approx(0.5)
    assert audit["conductance_profile"] == "local_device_table"
    assert audit["rail_shapes"] == {"ipos_a": [2], "ineg_a": [2]}
    assert audit["payload_identity"].startswith("sha256:")
    assert audit["current_units"] == "A"


def test_engine_rejects_ambiguous_layers_and_matrix_shapes(tmp_path):
    engine = AnalogCrossbarNumericalEngine.from_sim_config(
        _sim_config(tmp_path), weight_codes=np.array([[1]])
    )

    with pytest.raises(AnalogCrossbarNumericalError, match="exactly one layer"):
        engine.compute(
            input_values=[1.0],
            weight_codes_by_layer={"a": [[1]], "b": [[1]]},
            output_neuron_count=1,
        )

    with pytest.raises(AnalogCrossbarNumericalError, match="2-D"):
        engine.compute(
            input_values=[1.0],
            weight_codes_by_layer={"a": [[[1]]]},
            output_neuron_count=1,
        )


def test_engine_rejects_non_differential_sim_config_polarity(tmp_path):
    with pytest.raises(AnalogCrossbarNumericalError, match="Xbar_Polarity"):
        AnalogCrossbarNumericalEngine.from_sim_config(
            _sim_config(tmp_path, polarity="1"), weight_codes=np.array([[1]])
        )


def test_engine_allows_omitted_constructor_weight_codes(tmp_path):
    engine = AnalogCrossbarNumericalEngine.from_sim_config(_sim_config(tmp_path))

    ipos, ineg, _ = engine.compute(
        input_values=[1.0],
        weight_codes_by_layer={"L0": [[1]]},
        output_neuron_count=1,
    )

    np.testing.assert_allclose(ipos, [0.001])
    np.testing.assert_allclose(ineg, [0.0005])
