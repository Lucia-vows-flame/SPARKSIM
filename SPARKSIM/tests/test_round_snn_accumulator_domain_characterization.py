from fractions import Fraction

import pytest

from MNSIM.Scheduling.accumulator_domain import (
    AccumulatorDomainError,
    build_completed_accumulator_profile,
    derive_fixed_range_adc_lsb_scale,
    reconstruct_completed_accumulator_code,
)


def test_bit_slice_reconstruction_is_exact_and_uses_one_common_scale():
    profile = build_completed_accumulator_profile(
        adc_signed_bits=5,
        activation_cycles=2,
        input_bits_per_cycle=2,
        weight_planes=2,
        weight_bits_per_plane=2,
        row_partials=1,
        adc_lsb_scale=Fraction(3, 128),
    )

    code = reconstruct_completed_accumulator_code(
        profile,
        adc_codes=[
            [[1, -2], [3, 4]],
        ],
    )

    assert code == 1 + (-2 << 2) + (3 << 2) + (4 << 4)
    assert profile.accumulator_lsb_scale == Fraction(3, 128)
    assert profile.physical_value(code) == Fraction(code * 3, 128)


def test_row_and_physical_fragment_completion_preserve_the_same_code_domain():
    profile = build_completed_accumulator_profile(
        adc_signed_bits=6,
        activation_cycles=2,
        input_bits_per_cycle=1,
        weight_planes=3,
        weight_bits_per_plane=2,
        row_partials=2,
        adc_lsb_scale=Fraction(5, 256),
    )
    row0 = [[[1, 2, 3], [4, 5, 6]]]
    row1 = [[[7, 8, 9], [10, 11, 12]]]

    whole = reconstruct_completed_accumulator_code(
        profile,
        adc_codes=[row0[0], row1[0]],
    )
    split_then_completed = sum(
        reconstruct_completed_accumulator_code(
            profile.with_row_partials(1),
            adc_codes=fragment,
        )
        for fragment in (row0, row1)
    )

    assert whole == split_then_completed


@pytest.mark.parametrize(
    "adc_bits,activation_cycles,input_bits,weight_planes,weight_bits,row_partials",
    [
        (2, 1, 1, 1, 1, 1),
        (10, 4, 2, 4, 2, 3),
        (12, 3, 4, 2, 4, 8),
    ],
)
def test_derived_width_contains_every_extreme_completed_code(
    adc_bits,
    activation_cycles,
    input_bits,
    weight_planes,
    weight_bits,
    row_partials,
):
    profile = build_completed_accumulator_profile(
        adc_signed_bits=adc_bits,
        activation_cycles=activation_cycles,
        input_bits_per_cycle=input_bits,
        weight_planes=weight_planes,
        weight_bits_per_plane=weight_bits,
        row_partials=row_partials,
        adc_lsb_scale=Fraction(1, 1),
    )

    assert profile.signed_min <= -profile.maximum_magnitude
    assert profile.maximum_magnitude <= profile.signed_max
    assert profile.signed_bit_width == (
        profile.maximum_magnitude.bit_length() + 1
    )


def test_profile_requires_adc_lsb_scale_not_output_activation_scale():
    with pytest.raises(AccumulatorDomainError, match="adc_lsb_scale"):
        build_completed_accumulator_profile(
            adc_signed_bits=10,
            activation_cycles=1,
            input_bits_per_cycle=8,
            weight_planes=1,
            weight_bits_per_plane=8,
            row_partials=1,
            adc_lsb_scale=None,
        )


def test_fixed_range_adc_lsb_is_independent_of_output_activation_scale():
    scale = derive_fixed_range_adc_lsb_scale(
        input_code_scale=Fraction(1, 32),
        weight_code_scale=Fraction(1, 64),
        input_digit_max=3,
        weight_digit_max=15,
        xbar_rows=128,
        adc_signed_bits=10,
    )

    assert scale == Fraction(3 * 15 * 128, 512 * 32 * 64)


def test_adc_lsb_derivation_rejects_output_scale_coupled_mode():
    with pytest.raises(AccumulatorDomainError, match="output-scale-coupled"):
        derive_fixed_range_adc_lsb_scale(
            input_code_scale=Fraction(1, 32),
            weight_code_scale=Fraction(1, 64),
            input_digit_max=3,
            weight_digit_max=15,
            xbar_rows=128,
            adc_signed_bits=10,
            adc_mode="SCALE",
        )


def test_profile_rejects_adc_codes_outside_the_declared_signed_range():
    profile = build_completed_accumulator_profile(
        adc_signed_bits=4,
        activation_cycles=1,
        input_bits_per_cycle=1,
        weight_planes=1,
        weight_bits_per_plane=1,
        row_partials=1,
        adc_lsb_scale=Fraction(1, 16),
    )

    with pytest.raises(AccumulatorDomainError, match="ADC code"):
        reconstruct_completed_accumulator_code(profile, adc_codes=[[[8]]])
