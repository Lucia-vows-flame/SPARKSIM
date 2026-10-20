from __future__ import annotations

import configparser
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from MNSIM.Scheduling.device_precision import (
    DevicePrecisionError,
    device_bits_from_level,
)
from MNSIM.Scheduling.round_types import (
    INVALID_HARDWARE_PROJECTION,
    INVALID_SIM_CONFIG,
    Phase1Invalid,
)


SimConfigInput = dict[str, Any] | str | Path | configparser.ConfigParser

SECTION_KEYS = {
    "Xbar_Size": ("Crossbar level", "Xbar_Size"),
    "Tile_Num": ("Architecture level", "Tile_Num"),
    "PE_Num": ("Tile level", "PE_Num"),
    "Group_Num": ("Process element level", "Group_Num"),
    "Device_Level": ("Device level", "Device_Level"),
    "Weight_precision": ("Algorithm Configuration", "Weight_precision"),
    "Xbar_Polarity": ("Process element level", "Xbar_Polarity"),
}


@dataclass
class HardwareProjection:
    xbar_size: list[int]
    xbar_row: int
    xbar_col: int
    xbar_nums: int
    device_level: int
    device_bits: int
    weight_precision: int
    xbar_polarity: int
    raw_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "xbar_size": list(self.xbar_size),
            "xbar_row": self.xbar_row,
            "xbar_col": self.xbar_col,
            "xbar_nums": self.xbar_nums,
            "device_level": self.device_level,
            "device_bits": self.device_bits,
            "weight_precision": self.weight_precision,
            "xbar_polarity": self.xbar_polarity,
        }


def _required(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise Phase1Invalid(INVALID_SIM_CONFIG, f"missing required SimConfig field {key}")
    return config[key]


def _strict_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise Phase1Invalid(INVALID_SIM_CONFIG, f"{key} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    raise Phase1Invalid(INVALID_SIM_CONFIG, f"{key} must be an integer, got {value!r}")


def _pair(config: dict[str, Any], key: str) -> list[int]:
    value = _required(config, key)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split()]
        value = parts
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise Phase1Invalid(INVALID_SIM_CONFIG, f"{key} must be a pair")
    return [
        _strict_int(value[0], f"{key}[0]"),
        _strict_int(value[1], f"{key}[1]"),
    ]


def _from_config_parser(parser: configparser.ConfigParser) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for public_key, (section, option) in SECTION_KEYS.items():
        if not parser.has_section(section):
            raise Phase1Invalid(INVALID_SIM_CONFIG, f"missing SimConfig section {section}")
        if not parser.has_option(section, option):
            raise Phase1Invalid(INVALID_SIM_CONFIG, f"missing required SimConfig field {section}.{option}")
        flattened[public_key] = parser.get(section, option)
    return flattened


def _read_config_path(path_like: str | Path) -> dict[str, Any]:
    path = Path(path_like)
    parser = configparser.ConfigParser()
    loaded = parser.read(path)
    if not loaded:
        raise Phase1Invalid(INVALID_SIM_CONFIG, f"could not read SimConfig path {path}")
    return _from_config_parser(parser)


def _normalize_config(sim_config: SimConfigInput) -> dict[str, Any]:
    if isinstance(sim_config, configparser.ConfigParser):
        return _from_config_parser(sim_config)
    if isinstance(sim_config, (str, Path)):
        return _read_config_path(sim_config)
    if isinstance(sim_config, dict):
        return dict(sim_config)
    raise Phase1Invalid(INVALID_SIM_CONFIG, f"unsupported sim_config input type {type(sim_config).__name__}")


def build_hardware_projection(sim_config: SimConfigInput) -> HardwareProjection:
    config = _normalize_config(sim_config)
    xbar_size = _pair(config, "Xbar_Size")
    tile_num = _pair(config, "Tile_Num")
    pe_num = _pair(config, "PE_Num")
    group_num = _strict_int(_required(config, "Group_Num"), "Group_Num")
    device_level = _strict_int(_required(config, "Device_Level"), "Device_Level")
    weight_precision = _strict_int(_required(config, "Weight_precision"), "Weight_precision")
    xbar_polarity = _strict_int(_required(config, "Xbar_Polarity"), "Xbar_Polarity")

    xbar_row, xbar_col = xbar_size

    if xbar_row <= 0 or xbar_col <= 0:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, "xbar_size dimensions must be positive")
    if group_num <= 0:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, "Group_Num must be positive")
    if xbar_polarity not in (1, 2):
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, f"unsupported Xbar_Polarity {xbar_polarity}")

    xbar_nums = (
        tile_num[0]
        * tile_num[1]
        * pe_num[0]
        * pe_num[1]
        * group_num
        * xbar_polarity
    )

    if xbar_nums <= 0:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, f"xbar_nums must be positive, got {xbar_nums}")
    try:
        device_bits = device_bits_from_level(device_level)
    except DevicePrecisionError as exc:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, str(exc)) from exc
    if weight_precision <= 0:
        raise Phase1Invalid(INVALID_HARDWARE_PROJECTION, "Weight_precision must be positive")

    return HardwareProjection(
        xbar_size=xbar_size,
        xbar_row=xbar_row,
        xbar_col=xbar_col,
        xbar_nums=xbar_nums,
        device_level=device_level,
        device_bits=device_bits,
        weight_precision=weight_precision,
        xbar_polarity=xbar_polarity,
        raw_config=dict(config),
    )


__all__ = [
    "HardwareProjection",
    "SECTION_KEYS",
    "SimConfigInput",
    "build_hardware_projection",
]
