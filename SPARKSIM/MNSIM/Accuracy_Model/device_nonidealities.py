"""Cell-level device non-idealities used by PIM accuracy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DeviceNonidealityConfig:
    resistances_ohm: tuple[float, ...]
    variation_percent: float
    saf_hrs_percent: float
    saf_lrs_percent: float
    variation_model: str = "multiplicative_gaussian_resistance"
    programming_error_slope: float | None = None
    programming_error_intercept_us: float | None = None

    def __post_init__(self):
        if len(self.resistances_ohm) < 2:
            raise ValueError("Device_Resistance must contain at least two levels")
        if any(not math.isfinite(r) or r <= 0 for r in self.resistances_ohm):
            raise ValueError("Device_Resistance values must be finite and positive")
        if any(value < 0 for value in (
            self.variation_percent, self.saf_hrs_percent, self.saf_lrs_percent
        )):
            raise ValueError("device non-ideality percentages must be non-negative")
        if self.saf_hrs_percent + self.saf_lrs_percent > 100:
            raise ValueError("Device_SAF percentages must sum to at most 100")
        supported = {
            "multiplicative_gaussian_resistance",
            "crosssim_rram_milo_programming_error",
        }
        if self.variation_model not in supported:
            raise ValueError(f"unsupported device variation model: {self.variation_model}")
        if self.variation_model == "crosssim_rram_milo_programming_error":
            if self.programming_error_slope is None or self.programming_error_intercept_us is None:
                raise ValueError("RRAMMilo programming error coefficients are required")

    @property
    def max_code(self) -> int:
        return 2 ** math.floor(math.log2(len(self.resistances_ohm))) - 1


def _normalized_conductance(resistance: np.ndarray, config: DeviceNonidealityConfig):
    conductance = 1.0 / resistance
    table = 1.0 / np.asarray(config.resistances_ohm, dtype=np.float64)
    # Preserve MNSIM's code-space convention: HRS is zero and the highest
    # usable conductance level represents max_code.
    return (conductance - table[0]) * (config.max_code / (table[-1] - table[0]))


def _nominal_resistance(rounded, config, *, enable_r_ratio):
    table = np.asarray(config.resistances_ohm, dtype=np.float64)
    if enable_r_ratio:
        return table[rounded]
    conductance = 1.0 / table[0] + (
        rounded / config.max_code
    ) * (1.0 / table[-1] - 1.0 / table[0])
    return 1.0 / conductance


def apply_cell_nonidealities(
    codes,
    config: DeviceNonidealityConfig,
    *,
    enable_r_ratio: bool,
    enable_variation: bool,
    enable_saf: bool,
    rng: np.random.Generator,
):
    """Map one physical cell array into numerical conductance code space."""
    code_array = np.asarray(codes)
    rounded = np.rint(code_array).astype(np.int64)
    if not np.allclose(code_array, rounded) or np.any(rounded < 0) or np.any(rounded > config.max_code):
        raise ValueError(f"device codes must be integers in [0, {config.max_code}]")

    physical_mapping = enable_r_ratio or enable_variation
    if physical_mapping:
        resistance = _nominal_resistance(
            rounded, config, enable_r_ratio=enable_r_ratio
        )
        if enable_variation and config.variation_model == "crosssim_rram_milo_programming_error":
            conductance = 1.0 / resistance
            conductance_us = conductance * 1.0e6
            sigma_us = np.maximum(
                config.programming_error_slope * conductance_us
                + config.programming_error_intercept_us,
                0.0,
            )
            conductance = rng.normal(conductance, sigma_us * 1.0e-6)
            table = 1.0 / np.asarray(config.resistances_ohm, dtype=np.float64)
            result = (conductance - table[0]) * (
                config.max_code / (table[-1] - table[0])
            )
        else:
            if enable_variation and config.variation_percent:
                sigma = config.variation_percent / 100.0
                factor = rng.normal(1.0, sigma, size=rounded.shape)
                invalid = factor <= 0
                while np.any(invalid):
                    factor[invalid] = rng.normal(1.0, sigma, size=int(invalid.sum()))
                    invalid = factor <= 0
                resistance = resistance * factor
            result = _normalized_conductance(resistance, config)
    else:
        result = rounded.astype(np.float64)

    if enable_saf:
        draw = rng.random(rounded.shape)
        hrs = draw < config.saf_hrs_percent / 100.0
        lrs = draw >= 1.0 - config.saf_lrs_percent / 100.0
        result = np.where(hrs, 0.0, result)
        lrs_resistance = np.full(rounded.shape, config.resistances_ohm[-1])
        lrs_value = _normalized_conductance(lrs_resistance, config) if physical_mapping else config.max_code
        result = np.where(lrs, lrs_value, result)
    return result.astype(np.float64, copy=False)
