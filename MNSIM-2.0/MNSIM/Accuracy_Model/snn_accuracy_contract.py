"""Formal SNN accuracy profiles and immutable binary-spike semantics."""

from __future__ import annotations

import configparser
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ADC_DIGITAL_LIF = "adc_digital_lif"
FRONTEND_LUT_LIF_LUT = "frontend_lut_lif_lut"
SUPPORTED_SNN_HARDWARE_PROFILES = frozenset(
    {ADC_DIGITAL_LIF, FRONTEND_LUT_LIF_LUT}
)


class SNNAccuracyContractError(ValueError):
    """A SimConfig violates the formal SNN accuracy contract."""


@dataclass(frozen=True)
class BinarySpikeDomain:
    values: tuple[int, int] = (0, 1)
    scale: float = 1.0
    signed: bool = False
    input_digit_max: int = 1
    activation_cycle: int = 1

    def __post_init__(self) -> None:
        if not (
            type(self.values) is tuple
            and len(self.values) == 2
            and all(type(item) is int for item in self.values)
            and self.values == (0, 1)
            and type(self.scale) is float
            and self.scale == 1.0
            and type(self.signed) is bool
            and self.signed is False
            and type(self.input_digit_max) is int
            and self.input_digit_max == 1
            and type(self.activation_cycle) is int
            and self.activation_cycle == 1
        ):
            raise ValueError(
                "formal binary spike domain requires values=(0, 1), scale=1.0, "
                "signed=False, input_digit_max=1, and activation_cycle=1"
            )

    def validate(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError("binary spike value must be a torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError("binary spike tensor contains non-finite values")
        if not torch.all((value == 0) | (value == 1)):
            raise ValueError("binary spike tensor must contain only 0 and 1")
        return value


BINARY_SPIKE_DOMAIN = BinarySpikeDomain()


@dataclass(frozen=True)
class SNNAccuracyConfig:
    hardware_profile: str
    hardware_current_model: str
    adc_model: str | None
    adc_precision: int | None
    readc_percentile_low: float | None
    readc_percentile_high: float | None
    readc_max_samples_per_layer: int | None
    lif_perturbation_mode: str
    lif_perturbation_artifact: Path | None
    lif_perturbation_seed: int


def parse_snn_accuracy_config(
    parser: configparser.ConfigParser,
    *,
    source: str | Path = "SimConfig.ini",
) -> SNNAccuracyConfig:
    section = "SNN Accuracy"
    if not parser.has_section(section):
        raise SNNAccuracyContractError(f"{source} is missing [{section}]")

    def required(option: str) -> str:
        try:
            return parser.get(section, option).strip()
        except configparser.Error as exc:
            raise SNNAccuracyContractError(
                f"{source} is missing required [{section}] {option}"
            ) from exc

    profile = required("SNN_Hardware_Profile")
    current_model = required("Hardware_Current_Model")
    artifact_text = required("LIF_Perturbation_Artifact")
    perturbation_mode = parser.get(
        section, "LIF_Perturbation_Mode", fallback=""
    ).strip().lower()
    try:
        seed = int(required("LIF_Perturbation_Seed"))
    except ValueError as exc:
        raise SNNAccuracyContractError(
            f"{source} [SNN Accuracy] LIF_Perturbation_Seed must be an integer"
        ) from exc

    if profile not in SUPPORTED_SNN_HARDWARE_PROFILES:
        raise SNNAccuracyContractError(
            f"unsupported SNN_Hardware_Profile {profile!r} in {source}"
        )
    if current_model != "physical_conductance":
        raise SNNAccuracyContractError(
            "formal Hardware_Current_Model must be physical_conductance"
        )
    if seed != 112:
        raise SNNAccuracyContractError("formal LIF_Perturbation_Seed must be 112")

    if profile == ADC_DIGITAL_LIF:
        if artifact_text:
            raise SNNAccuracyContractError(
                "adc_digital_lif must not configure LIF_Perturbation_Artifact"
            )
        expected = {
            "ADC_Profile": "HONG_2025_READC_5BIT",
            "ADC_Quantization_Mode": "READC_LLOYD_MAX",
            "ADC_Precision": "5",
            "ADC_ReADC_Percentile_Low": "0.5",
            "ADC_ReADC_Percentile_High": "99.5",
            "ADC_ReADC_Max_Samples_Per_Layer": "8000",
        }
        try:
            for option, expected_value in expected.items():
                actual = parser.get("Interface level", option).strip()
                if actual != expected_value:
                    raise SNNAccuracyContractError(
                        "formal ADC profile requires "
                        f"{option}={expected_value}, got {actual}"
                    )
        except configparser.Error as exc:
            raise SNNAccuracyContractError(
                f"{source} is missing required formal ADC configuration: {exc}"
            ) from exc
        return SNNAccuracyConfig(
            hardware_profile=profile,
            hardware_current_model=current_model,
            adc_model="READC_LLOYD_MAX",
            adc_precision=5,
            readc_percentile_low=0.5,
            readc_percentile_high=99.5,
            readc_max_samples_per_layer=8000,
            lif_perturbation_mode="none",
            lif_perturbation_artifact=None,
            lif_perturbation_seed=seed,
        )

    if not artifact_text and perturbation_mode != "none":
        raise SNNAccuracyContractError(
            "frontend_lut_lif_lut requires LIF_Perturbation_Artifact or "
            "LIF_Perturbation_Mode=none"
        )
    if artifact_text and perturbation_mode not in {"", "qualified_artifact"}:
        raise SNNAccuracyContractError(
            "LIF_Perturbation_Artifact requires "
            "LIF_Perturbation_Mode=qualified_artifact"
        )
    artifact_path = None
    if artifact_text:
        artifact_path = Path(artifact_text)
        if not artifact_path.is_absolute():
            artifact_path = Path(source).resolve().parent / artifact_path
        artifact_path = artifact_path.resolve()
    return SNNAccuracyConfig(
        hardware_profile=profile,
        hardware_current_model=current_model,
        adc_model=None,
        adc_precision=None,
        readc_percentile_low=None,
        readc_percentile_high=None,
        readc_max_samples_per_layer=None,
        lif_perturbation_mode=(
            "qualified_artifact" if artifact_path is not None else "none"
        ),
        lif_perturbation_artifact=artifact_path,
        lif_perturbation_seed=seed,
    )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bit_weights_sha256(net_bit_weights: list[Any]) -> str:
    digest = hashlib.sha256()
    for layer_index, layer in enumerate(net_bit_weights):
        digest.update(f"layer:{layer_index}\n".encode("ascii"))
        if layer is None:
            digest.update(b"none\n")
            continue
        for key in sorted(layer):
            array = np.ascontiguousarray(layer[key])
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(repr(array.shape).encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "ADC_DIGITAL_LIF",
    "BINARY_SPIKE_DOMAIN",
    "FRONTEND_LUT_LIF_LUT",
    "SUPPORTED_SNN_HARDWARE_PROFILES",
    "BinarySpikeDomain",
    "SNNAccuracyConfig",
    "SNNAccuracyContractError",
    "bit_weights_sha256",
    "canonical_json_sha256",
    "file_sha256",
    "parse_snn_accuracy_config",
]
