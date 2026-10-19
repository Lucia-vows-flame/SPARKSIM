"""Shared validation for bit-precise original-MNSIM device levels."""

from __future__ import annotations

from numbers import Integral


class DevicePrecisionError(ValueError):
    """Raised when Device_Level cannot represent an integral bit width."""


def device_bits_from_level(device_level: int) -> int:
    if isinstance(device_level, bool) or not isinstance(device_level, Integral):
        raise DevicePrecisionError(
            f"Device_Level must be an integer power of two, got {device_level!r}"
        )
    level = int(device_level)
    if level < 2 or level & (level - 1):
        raise DevicePrecisionError(
            f"Device_Level must be a power of two greater than one, got {level}"
        )
    return level.bit_length() - 1


__all__ = ["DevicePrecisionError", "device_bits_from_level"]
