"""Strict configuration parsing for digital and local analog LIF modes."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


class LifModeConfigError(ValueError):
    """Raised when a SimConfig contains an invalid LIF mode configuration."""


@dataclass(frozen=True)
class AnalogLifConfig:
    mode: str
    lut_root: Path
    verify_checksums: bool
    profile: str
    negative_vmem_policy: str


_SECTION = "Neuron"
_DEFAULTS = {
    "lif_mode": "digital",
    "analog_lif_lut_root": "08_lut",
    "analog_lif_lut_verify_checksums": "1",
    "analog_lif_lut_profile": "fast_1h_experimental_7pct",
    "analog_lif_negative_vmem_policy": "error",
}
_OPTION_LABELS = {
    "lif_mode": "LIF_Mode",
    "analog_lif_lut_root": "Analog_LIF_LUT_Root",
    "analog_lif_lut_verify_checksums": "Analog_LIF_LUT_Verify_Checksums",
    "analog_lif_lut_profile": "Analog_LIF_LUT_Profile",
    "analog_lif_negative_vmem_policy": "Analog_LIF_Negative_Vmem_Policy",
}
_SUPPORTED_MODES = frozenset({"digital", "analog_lut"})
_SUPPORTED_POLICIES = frozenset({"error", "clamp_zero"})


def load_lif_mode_config(sim_config_path: str | Path) -> AnalogLifConfig:
    """Load and validate the ``[Neuron]`` LIF configuration.

    The LUT root is interpreted relative to the directory containing the
    supplied SimConfig, matching the artifact paths used by the simulator.
    The parser intentionally does not check that the root exists: digital
    configurations must remain loadable on machines without LUT artifacts,
    while the analog loader owns artifact existence and integrity checks.
    """

    config_path = Path(sim_config_path).expanduser()
    if not config_path.is_file():
        raise LifModeConfigError(f"cannot read SimConfig file: {config_path}")

    parser = configparser.ConfigParser(interpolation=None, strict=True)
    try:
        loaded = parser.read(config_path, encoding="UTF-8")
    except (OSError, configparser.Error) as error:
        raise LifModeConfigError(
            f"cannot parse SimConfig {config_path}: {error}"
        ) from error
    if loaded != [str(config_path)]:
        raise LifModeConfigError(f"cannot read SimConfig file: {config_path}")

    values = dict(_DEFAULTS)
    if parser.has_section(_SECTION):
        unknown = sorted(set(parser[_SECTION]) - set(_DEFAULTS))
        if unknown:
            raise LifModeConfigError(
                f"unsupported [Neuron] option(s): {', '.join(unknown)}"
            )
        values.update(
            {
                option: parser.get(_SECTION, option)
                for option in _DEFAULTS
                if parser.has_option(_SECTION, option)
            }
        )

    mode = _normalized_choice(
        values["lif_mode"], _OPTION_LABELS["lif_mode"], _SUPPORTED_MODES
    )
    policy = _normalized_choice(
        values["analog_lif_negative_vmem_policy"],
        _OPTION_LABELS["analog_lif_negative_vmem_policy"],
        _SUPPORTED_POLICIES,
    )
    root_text = values["analog_lif_lut_root"].strip()
    if not root_text:
        raise LifModeConfigError(
            f"{_OPTION_LABELS['analog_lif_lut_root']} must not be empty"
        )
    profile = values["analog_lif_lut_profile"].strip()
    if not profile:
        raise LifModeConfigError(
            f"{_OPTION_LABELS['analog_lif_lut_profile']} must not be empty"
        )

    checksum_text = values["analog_lif_lut_verify_checksums"].strip()
    if checksum_text not in {"0", "1"}:
        raise LifModeConfigError(
            f"{_OPTION_LABELS['analog_lif_lut_verify_checksums']} "
            "must be exactly 0 or 1"
        )

    root = Path(root_text).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root

    return AnalogLifConfig(
        mode=mode,
        lut_root=root.resolve(),
        verify_checksums=checksum_text == "1",
        profile=profile,
        negative_vmem_policy=policy,
    )


def _normalized_choice(
    value: str, option: str, supported: frozenset[str]
) -> str:
    normalized = value.strip().lower()
    if normalized not in supported:
        choices = ", ".join(sorted(supported))
        raise LifModeConfigError(
            f"{option} must be one of {choices}; got {value.strip()!r}"
        )
    return normalized


__all__ = [
    "AnalogLifConfig",
    "LifModeConfigError",
    "load_lif_mode_config",
]
