"""Offline ReADC Lloyd-Max calibration and inference tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


READC_PROVENANCE = {
    "model": "READC_LLOYD_MAX",
    "paper": "Hong et al., Nature Communications 16, 9749 (2025)",
    "doi": "10.1038/s41467-025-65233-w",
    "reference_implementation": "https://github.com/MIKEHHQ/ReADC",
}


@dataclass(frozen=True)
class ReADCCalibrationConfig:
    precision: int
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    max_samples_per_layer: int = 8000
    max_iterations: int = 500
    convergence_threshold: float = 1e-6

    def __post_init__(self):
        if self.precision < 1:
            raise ValueError("ADC precision must be positive")
        if not 0 <= self.percentile_low < self.percentile_high <= 100:
            raise ValueError("ReADC percentiles must satisfy 0 <= low < high <= 100")
        if self.max_samples_per_layer < 2:
            raise ValueError("ReADC max samples per layer must be at least two")


class ReADCSampleCollector:
    """Deterministic bounded collector for one layer's pre-ADC values."""

    def __init__(self, max_samples: int):
        self.max_samples = int(max_samples)
        self._sample = np.array([], dtype=np.float64)
        self.observed_count = 0
        self.observed_min = np.inf
        self.observed_max = -np.inf

    @staticmethod
    def _even_indices(size: int, count: int, device):
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if count == 1:
            return torch.zeros(1, dtype=torch.long, device=device)
        return (
            torch.arange(count, dtype=torch.long, device=device)
            * (size - 1)
            // (count - 1)
        )

    @staticmethod
    def _resample_numpy(values, count):
        if count <= 0:
            return np.array([], dtype=np.float64)
        if count >= values.size:
            return values
        indices = np.arange(count, dtype=np.int64) * (values.size - 1) // (count - 1)
        return values[indices]

    def add(self, values: torch.Tensor):
        flat = values.detach().reshape(-1)
        flat = flat[torch.isfinite(flat)]
        new_count = flat.numel()
        if new_count == 0:
            return
        self.observed_min = min(self.observed_min, float(flat.min().item()))
        self.observed_max = max(self.observed_max, float(flat.max().item()))
        total_count = self.observed_count + new_count
        target_size = min(self.max_samples, total_count)
        new_keep = min(
            new_count,
            max(1, int(round(target_size * new_count / total_count))),
        )
        old_keep = min(self._sample.size, target_size - new_keep)
        new_keep = target_size - old_keep
        old_sample = self._resample_numpy(self._sample, old_keep)
        indices = self._even_indices(new_count, new_keep, flat.device)
        new_sample = flat[indices].cpu().double().numpy()
        self._sample = np.concatenate([old_sample, new_sample])
        self.observed_count = total_count

    def values(self):
        if not self._sample.size:
            return np.array([], dtype=np.float64)
        return np.append(self._sample, [self.observed_min, self.observed_max])


def lloyd_max(data, config: ReADCCalibrationConfig):
    """Fit the algorithm published by ReADC, deterministically."""
    values = np.asarray(data, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("ReADC calibration requires at least two finite samples")
    original_min, original_max = float(values.min()), float(values.max())
    low, high = np.percentile(
        values, [config.percentile_low, config.percentile_high]
    )
    filtered = values[(values >= low) & (values <= high)]
    if filtered.size > config.max_samples_per_layer:
        indices = np.linspace(
            0, filtered.size - 1, config.max_samples_per_layer, dtype=np.int64
        )
        filtered = np.sort(filtered)[indices]
    filtered = np.append(filtered, [original_min, original_max])

    level_count = 2 ** config.precision
    boundaries = np.linspace(filtered.min(), filtered.max(), level_count + 1)
    levels = (boundaries[:-1] + boundaries[1:]) / 2
    previous_mse = np.inf
    iterations = 0
    for iteration in range(config.max_iterations):
        new_levels = levels.copy()
        squared_error = 0.0
        for index in range(level_count):
            upper_comparison = filtered <= boundaries[index + 1] if index == level_count - 1 else filtered < boundaries[index + 1]
            mask = (filtered >= boundaries[index]) & upper_comparison
            if np.any(mask):
                new_levels[index] = filtered[mask].mean()
                squared_error += np.square(filtered[mask] - new_levels[index]).sum()
        new_boundaries = boundaries.copy()
        new_boundaries[1:-1] = (new_levels[:-1] + new_levels[1:]) / 2
        mse = squared_error / filtered.size
        boundaries, levels = new_boundaries, new_levels
        iterations = iteration + 1
        if iteration >= 100:
            improvement = abs(previous_mse - mse) / (previous_mse + 1e-10)
            if improvement < config.convergence_threshold:
                break
        previous_mse = mse
    return {
        "boundaries": boundaries[1:-1].tolist(),
        "levels": levels.tolist(),
        "sample_count": int(filtered.size),
        "observed_min": original_min,
        "observed_max": original_max,
        "iterations": iterations,
        "mse": float(mse),
    }


def readc_quantize(values: torch.Tensor, table):
    boundaries = torch.as_tensor(
        table["boundaries"], device=values.device, dtype=values.dtype
    )
    levels = torch.as_tensor(table["levels"], device=values.device, dtype=values.dtype)
    if boundaries.numel() + 1 != levels.numel():
        raise ValueError("invalid ReADC table: levels must outnumber boundaries by one")
    if boundaries.numel() and torch.any(boundaries[1:] <= boundaries[:-1]):
        raise ValueError("invalid ReADC table: boundaries must be strictly increasing")
    return levels[torch.bucketize(values.contiguous(), boundaries)]
