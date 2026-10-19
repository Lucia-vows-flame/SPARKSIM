import numpy as np
import pytest

from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Accuracy_Model.device_nonidealities import (
    DeviceNonidealityConfig,
    apply_cell_nonidealities,
)


def _config(**overrides):
    values = dict(
        resistances_ohm=(1000.0, 500.0, 250.0, 100.0),
        variation_percent=5.0,
        saf_hrs_percent=0.0,
        saf_lrs_percent=0.0,
    )
    values.update(overrides)
    return DeviceNonidealityConfig(**values)


def test_r_ratio_maps_every_configured_resistance_level():
    result = apply_cell_nonidealities(
        np.arange(4), _config(), enable_r_ratio=True,
        enable_variation=False, enable_saf=False,
        rng=np.random.default_rng(1),
    )
    expected = (1 / np.array([1000.0, 500.0, 250.0, 100.0]) - 1 / 1000.0)
    expected *= 3 / (1 / 100.0 - 1 / 1000.0)
    np.testing.assert_allclose(result, expected)


def test_variation_is_cell_independent_and_seeded_without_enabling_r_ratio():
    codes = np.full((64, 64), 2)
    first = apply_cell_nonidealities(
        codes, _config(), enable_r_ratio=False, enable_variation=True,
        enable_saf=False, rng=np.random.default_rng(7),
    )
    second = apply_cell_nonidealities(
        codes, _config(), enable_r_ratio=False, enable_variation=True,
        enable_saf=False, rng=np.random.default_rng(7),
    )
    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size > 4000
    assert abs(first.mean() - 2.0) < 0.05


def test_saf_is_terminal_and_uses_exact_endpoints():
    result = apply_cell_nonidealities(
        np.ones(100), _config(saf_hrs_percent=50, saf_lrs_percent=50),
        enable_r_ratio=True, enable_variation=True, enable_saf=True,
        rng=np.random.default_rng(9),
    )
    assert set(np.unique(result)) == {0.0, 3.0}


def test_differential_arrays_receive_independent_draws_from_one_generator():
    rng = np.random.default_rng(11)
    positive = apply_cell_nonidealities(
        np.ones(100), _config(), enable_r_ratio=False,
        enable_variation=True, enable_saf=False, rng=rng,
    )
    negative = apply_cell_nonidealities(
        np.ones(100), _config(), enable_r_ratio=False,
        enable_variation=True, enable_saf=False, rng=rng,
    )
    assert not np.array_equal(positive, negative)


class _RecordingGenerator:
    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)
        self.normal_draws = []

    def normal(self, *args, **kwargs):
        value = self.generator.normal(*args, **kwargs)
        self.normal_draws.append(np.array(value, copy=True))
        return value

    def random(self, *args, **kwargs):
        return self.generator.random(*args, **kwargs)


def test_20ns_and_30ns_use_paired_random_sequence_with_device_specific_sigma():
    codes = np.arange(16, dtype=np.float64).reshape(4, 4)
    resistance = tuple(np.linspace(1.0e6, 1.0e4, 256))
    rng20 = _RecordingGenerator(112)
    rng30 = _RecordingGenerator(112)
    out20 = apply_cell_nonidealities(
        codes,
        DeviceNonidealityConfig(resistance, 6.69903802061613, 0.0, 0.0),
        enable_r_ratio=True,
        enable_variation=True,
        enable_saf=False,
        rng=rng20,
    )
    out30 = apply_cell_nonidealities(
        codes,
        DeviceNonidealityConfig(resistance, 7.725782066255718, 0.0, 0.0),
        enable_r_ratio=True,
        enable_variation=True,
        enable_saf=False,
        rng=rng30,
    )

    z20 = (rng20.normal_draws[0] - 1.0) / (6.69903802061613 / 100.0)
    z30 = (rng30.normal_draws[0] - 1.0) / (7.725782066255718 / 100.0)
    assert np.allclose(z20, z30)
    assert not np.allclose(out20, out30)


class _ScaleRecordingGenerator:
    def __init__(self):
        self.loc = None
        self.scale = None

    def normal(self, loc=0.0, scale=1.0, size=None):
        self.loc = np.asarray(loc, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        return np.asarray(loc, dtype=np.float64)

    def random(self, size=None):
        return np.ones(size, dtype=np.float64) * 0.5


def test_rram_milo_uses_state_dependent_additive_conductance_sigma():
    resistances = tuple(
        1.0 / np.linspace(50e-6, 1.0 / 4450.0, 128, dtype=np.float64)
    )
    config = DeviceNonidealityConfig(
        resistances_ohm=resistances,
        variation_percent=0.0,
        saf_hrs_percent=0.0,
        saf_lrs_percent=0.0,
        variation_model="crosssim_rram_milo_programming_error",
        programming_error_slope=-0.009107,
        programming_error_intercept_us=4.782321,
    )
    rng = _ScaleRecordingGenerator()

    result = apply_cell_nonidealities(
        np.array([0, 127]),
        config,
        enable_r_ratio=True,
        enable_variation=True,
        enable_saf=False,
        rng=rng,
    )

    expected_g_us = np.array([50.0, (1.0 / 4450.0) * 1e6])
    expected_sigma_us = np.maximum(-0.009107 * expected_g_us + 4.782321, 0.0)
    np.testing.assert_allclose(rng.loc, expected_g_us * 1e-6)
    np.testing.assert_allclose(rng.scale, expected_sigma_us * 1e-6)
    np.testing.assert_allclose(result, [0.0, 127.0])


def test_weight_update_rejects_explicit_rng_without_required_methods(tmp_path):
    sim_config = tmp_path / "SimConfig.ini"
    sim_config.write_text(
        """\
[Device level]
Device_SAF = 0,0
Device_Variation = 6.69903802061613
Device_Level = 2
Device_Resistance = 1000,100
""",
        encoding="UTF-8",
    )

    with pytest.raises(TypeError, match=r"rng must provide normal\(\) and random\(\)"):
        weight_update(sim_config, [], rng=object())
