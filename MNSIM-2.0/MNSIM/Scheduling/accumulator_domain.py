"""Exact completed-accumulator metadata for ADC-derived digital codes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Sequence


class AccumulatorDomainError(ValueError):
    """Raised when an ADC-code accumulator contract is invalid."""


@dataclass(frozen=True)
class CompletedAccumulatorProfile:
    adc_signed_bits: int
    activation_cycles: int
    input_bits_per_cycle: int
    weight_planes: int
    weight_bits_per_plane: int
    row_partials: int
    accumulator_lsb_scale: Fraction
    maximum_magnitude: int
    signed_bit_width: int

    @property
    def signed_min(self) -> int:
        return -(1 << (self.signed_bit_width - 1))

    @property
    def signed_max(self) -> int:
        return (1 << (self.signed_bit_width - 1)) - 1

    @property
    def adc_signed_min(self) -> int:
        return -(1 << (self.adc_signed_bits - 1))

    @property
    def adc_signed_max(self) -> int:
        return (1 << (self.adc_signed_bits - 1)) - 1

    def physical_value(self, code: int) -> Fraction:
        return int(code) * self.accumulator_lsb_scale

    def with_row_partials(self, row_partials: int) -> "CompletedAccumulatorProfile":
        return build_completed_accumulator_profile(
            adc_signed_bits=self.adc_signed_bits,
            activation_cycles=self.activation_cycles,
            input_bits_per_cycle=self.input_bits_per_cycle,
            weight_planes=self.weight_planes,
            weight_bits_per_plane=self.weight_bits_per_plane,
            row_partials=row_partials,
            adc_lsb_scale=self.accumulator_lsb_scale,
        )


def build_completed_accumulator_profile(
    *,
    adc_signed_bits: int,
    activation_cycles: int,
    input_bits_per_cycle: int,
    weight_planes: int,
    weight_bits_per_plane: int,
    row_partials: int,
    adc_lsb_scale: Fraction | int | None,
) -> CompletedAccumulatorProfile:
    """Build the sufficient-width domain for exact radix reconstruction."""

    adc_signed_bits = _positive_int(adc_signed_bits, "adc_signed_bits")
    activation_cycles = _positive_int(activation_cycles, "activation_cycles")
    input_bits_per_cycle = _positive_int(
        input_bits_per_cycle, "input_bits_per_cycle"
    )
    weight_planes = _positive_int(weight_planes, "weight_planes")
    weight_bits_per_plane = _positive_int(
        weight_bits_per_plane, "weight_bits_per_plane"
    )
    row_partials = _positive_int(row_partials, "row_partials")
    if adc_lsb_scale is None:
        raise AccumulatorDomainError("adc_lsb_scale is required")
    try:
        scale = Fraction(adc_lsb_scale)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise AccumulatorDomainError("adc_lsb_scale must be a positive rational") from exc
    if scale <= 0:
        raise AccumulatorDomainError("adc_lsb_scale must be positive")

    radix_sum = sum(
        1 << (
            activation_index * input_bits_per_cycle
            + plane_index * weight_bits_per_plane
        )
        for activation_index in range(activation_cycles)
        for plane_index in range(weight_planes)
    )
    adc_maximum_magnitude = 1 << (adc_signed_bits - 1)
    maximum_magnitude = row_partials * adc_maximum_magnitude * radix_sum
    signed_bit_width = maximum_magnitude.bit_length() + 1

    return CompletedAccumulatorProfile(
        adc_signed_bits=adc_signed_bits,
        activation_cycles=activation_cycles,
        input_bits_per_cycle=input_bits_per_cycle,
        weight_planes=weight_planes,
        weight_bits_per_plane=weight_bits_per_plane,
        row_partials=row_partials,
        accumulator_lsb_scale=scale,
        maximum_magnitude=maximum_magnitude,
        signed_bit_width=signed_bit_width,
    )


def derive_fixed_range_adc_lsb_scale(
    *,
    input_code_scale: Fraction | int,
    weight_code_scale: Fraction | int,
    input_digit_max: int,
    weight_digit_max: int,
    xbar_rows: int,
    adc_signed_bits: int,
    adc_mode: str = "FIX",
) -> Fraction:
    """Derive the physical value of one uniform fixed-range ADC code step."""

    if str(adc_mode).upper() != "FIX":
        raise AccumulatorDomainError(
            "output-scale-coupled ADC modes are not valid accumulator domains"
        )
    input_scale = _positive_fraction(input_code_scale, "input_code_scale")
    weight_scale = _positive_fraction(weight_code_scale, "weight_code_scale")
    input_digit_max = _positive_int(input_digit_max, "input_digit_max")
    weight_digit_max = _positive_int(weight_digit_max, "weight_digit_max")
    xbar_rows = _positive_int(xbar_rows, "xbar_rows")
    adc_signed_bits = _positive_int(adc_signed_bits, "adc_signed_bits")
    fixed_range = input_digit_max * weight_digit_max * xbar_rows
    return (
        input_scale
        * weight_scale
        * Fraction(fixed_range, 1 << (adc_signed_bits - 1))
    )


def reconstruct_completed_accumulator_code(
    profile: CompletedAccumulatorProfile,
    *,
    adc_codes: Sequence[Sequence[Sequence[int]]],
) -> int:
    """Reconstruct one completed integer code from row/activation/weight codes."""

    rows = list(adc_codes)
    if len(rows) != profile.row_partials:
        raise AccumulatorDomainError(
            f"expected {profile.row_partials} row partials, got {len(rows)}"
        )

    completed = 0
    for row_index, row in enumerate(rows):
        activation_entries = list(row)
        if len(activation_entries) != profile.activation_cycles:
            raise AccumulatorDomainError(
                f"row {row_index} expected {profile.activation_cycles} activation cycles, "
                f"got {len(activation_entries)}"
            )
        for activation_index, planes in enumerate(activation_entries):
            plane_entries = list(planes)
            if len(plane_entries) != profile.weight_planes:
                raise AccumulatorDomainError(
                    f"row {row_index} activation {activation_index} expected "
                    f"{profile.weight_planes} weight planes, got {len(plane_entries)}"
                )
            for plane_index, raw_code in enumerate(plane_entries):
                code = _integer(raw_code, "ADC code")
                if not profile.adc_signed_min <= code <= profile.adc_signed_max:
                    raise AccumulatorDomainError(
                        f"ADC code {code} is outside signed {profile.adc_signed_bits}-bit range"
                    )
                shift = (
                    activation_index * profile.input_bits_per_cycle
                    + plane_index * profile.weight_bits_per_plane
                )
                completed += code << shift

    if not profile.signed_min <= completed <= profile.signed_max:
        raise AccumulatorDomainError(
            f"completed accumulator code {completed} exceeds derived signed width"
        )
    return completed


def _positive_int(value: int, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise AccumulatorDomainError(f"{name} must be positive")
    return result


def _positive_fraction(value: Fraction | int, name: str) -> Fraction:
    try:
        result = Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise AccumulatorDomainError(f"{name} must be a positive rational") from exc
    if result <= 0:
        raise AccumulatorDomainError(f"{name} must be positive")
    return result


def _integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise AccumulatorDomainError(f"{name} must be an integer")
    return int(value)


__all__ = [
    "AccumulatorDomainError",
    "CompletedAccumulatorProfile",
    "build_completed_accumulator_profile",
    "derive_fixed_range_adc_lsb_scale",
    "reconstruct_completed_accumulator_code",
]
