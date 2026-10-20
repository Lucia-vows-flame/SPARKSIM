import numpy as np
import torch

from MNSIM.Accuracy_Model.readc import (
    ReADCCalibrationConfig, ReADCSampleCollector,
    lloyd_max,
    readc_quantize,
)


def test_collector_uses_integer_safe_indices_for_large_tensors():
    values = torch.zeros(16_777_217)
    values[-1] = 9.0
    collector = ReADCSampleCollector(17)
    collector.add(values)
    sampled = collector.values()
    assert sampled.size == 19
    assert sampled[-1] == 9.0


def test_collector_represents_every_equal_sized_calibration_batch():
    collector = ReADCSampleCollector(100)
    for batch in range(10):
        collector.add(torch.full((100_000,), float(batch)))
    sampled = collector.values()[:-2]
    counts = np.bincount(sampled.astype(np.int64), minlength=10)
    assert np.all(counts >= 8)
    assert np.all(counts <= 13)


def test_lloyd_max_is_deterministic_and_reduces_skewed_distribution_mse():
    data = np.concatenate([np.zeros(900), np.linspace(0, 10, 100)])
    config = ReADCCalibrationConfig(precision=2, max_iterations=150)
    first = lloyd_max(data, config)
    second = lloyd_max(data, config)
    assert first == second
    assert len(first["boundaries"]) == 3
    assert len(first["levels"]) == 4
    uniform_levels = np.linspace(data.min(), data.max(), 4)
    uniform_indices = np.abs(data[:, None] - uniform_levels).argmin(axis=1)
    uniform_mse = np.square(data - uniform_levels[uniform_indices]).mean()
    assert first["mse"] < uniform_mse


def test_readc_returns_reconstruction_levels_not_integer_codes():
    table = {"boundaries": [-1.0, 0.5, 3.0], "levels": [-2.0, 0.25, 1.5, 7.0]}
    result = readc_quantize(torch.tensor([-4.0, 0.0, 1.0, 8.0]), table)
    assert torch.equal(result, torch.tensor([-2.0, 0.25, 1.5, 7.0]))
