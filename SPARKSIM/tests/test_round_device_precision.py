import pytest

from MNSIM.Scheduling.device_precision import (
    DevicePrecisionError,
    device_bits_from_level,
)


@pytest.mark.parametrize(
    ("device_level", "expected_bits"),
    [(2, 1), (4, 2), (8, 3), (16, 4), (256, 8)],
)
def test_device_bits_from_level_accepts_each_bit_precise_level(
    device_level,
    expected_bits,
):
    assert device_bits_from_level(device_level) == expected_bits


@pytest.mark.parametrize("device_level", [1, 3, 6, 12, 0, -2, True, "4"])
def test_device_bits_from_level_rejects_non_bit_precise_levels(device_level):
    with pytest.raises(DevicePrecisionError, match="Device_Level"):
        device_bits_from_level(device_level)
