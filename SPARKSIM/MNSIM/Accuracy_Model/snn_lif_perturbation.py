"""Qualified frontend/LIF characterization artifacts and run-scoped sampling."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from MNSIM.Accuracy_Model.snn_accuracy_contract import canonical_json_sha256


PARAMETER_ORDER = (
    "frontend_gain",
    "frontend_offset",
    "delta_tau",
    "delta_v_th",
    "delta_v_reset",
)


class LIFPerturbationArtifactError(ValueError):
    """The characterization artifact is incomplete, invalid, or tampered."""


@dataclass(frozen=True)
class LIFPerturbationArtifact:
    schema_version: int
    profile_id: str
    pdk: str
    model_files: tuple[dict[str, str], ...]
    simulator_version: str
    process_corner: str
    supply_voltage_v: float
    temperature_c: float
    monte_carlo_sample_count: int
    seed_policy: str
    parameter_order: tuple[str, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    circuit_source_sha256: str
    supported_timestep_s: float
    supported_input_min: float
    supported_input_max: float
    dynamic_noise_std: float | None
    transient_source_sha256: str | None
    artifact_checksum: str


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LIFPerturbationArtifactError(f"{field_name} must be a JSON object")
    return value


def _nonempty_text(mapping: dict[str, Any], key: str) -> str:
    try:
        value = mapping[key]
    except KeyError as error:
        raise LIFPerturbationArtifactError(f"missing {key}") from error
    if not isinstance(value, str) or not value.strip():
        raise LIFPerturbationArtifactError(f"{key} must be a nonempty string")
    return value


def _sha256(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.lower())
    ):
        raise LIFPerturbationArtifactError(f"{field_name} must be a SHA-256")
    return value


def _finite_float(mapping: dict[str, Any], key: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as error:
        raise LIFPerturbationArtifactError(f"{key} must be finite") from error
    if not math.isfinite(value):
        raise LIFPerturbationArtifactError(f"{key} must be finite")
    return value


def _positive_integer(mapping: dict[str, Any], key: str) -> int:
    try:
        raw = mapping[key]
    except KeyError as error:
        raise LIFPerturbationArtifactError(f"missing {key}") from error
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise LIFPerturbationArtifactError(f"{key} must be a positive integer")
    return raw


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LIFPerturbationArtifactError(
            f"could not read LIF perturbation artifact {path}: {error}"
        ) from error
    return _mapping(payload, "artifact")


def load_lif_perturbation_artifact(path: str | Path) -> LIFPerturbationArtifact:
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise LIFPerturbationArtifactError(
            "unsupported LIF perturbation schema_version"
        )
    if payload.get("profile_id") != "frontend_lut_lif_lut":
        raise LIFPerturbationArtifactError(
            "LIF perturbation profile_id must be frontend_lut_lif_lut"
        )

    qualification = _mapping(payload.get("qualification"), "qualification")
    pdk = _nonempty_text(qualification, "pdk")
    simulator_version = _nonempty_text(qualification, "simulator_version")
    process_corner = _nonempty_text(qualification, "process_corner")
    seed_policy = _nonempty_text(qualification, "seed_policy")
    circuit_source_sha256 = _sha256(
        qualification.get("circuit_source_sha256"),
        "circuit_source_sha256",
    )
    supply_voltage_v = _finite_float(qualification, "supply_voltage_v")
    temperature_c = _finite_float(qualification, "temperature_c")
    monte_carlo_sample_count = _positive_integer(
        qualification, "monte_carlo_sample_count"
    )

    model_file_payload = qualification.get("model_files")
    if not isinstance(model_file_payload, list) or not model_file_payload:
        raise LIFPerturbationArtifactError(
            "model_files must be a nonempty qualification list"
        )
    model_files = []
    for index, raw_model_file in enumerate(model_file_payload):
        model_file = _mapping(raw_model_file, f"model_files[{index}]")
        model_files.append(
            {
                "path": _nonempty_text(model_file, "path"),
                "sha256": _sha256(
                    model_file.get("sha256"), f"model_files[{index}].sha256"
                ),
            }
        )

    distribution = _mapping(payload.get("distribution"), "distribution")
    parameter_order_payload = distribution.get("parameter_order")
    if not isinstance(parameter_order_payload, list):
        raise LIFPerturbationArtifactError("parameter_order must be a list")
    parameter_order = tuple(parameter_order_payload)
    if parameter_order != PARAMETER_ORDER:
        raise LIFPerturbationArtifactError(
            f"parameter_order must be exactly {PARAMETER_ORDER}"
        )

    try:
        mean_array = np.asarray(distribution["mean"], dtype=np.float64)
        covariance_array = np.asarray(
            distribution["covariance"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LIFPerturbationArtifactError(
            "mean and covariance must be finite numeric arrays"
        ) from error
    if mean_array.shape != (len(PARAMETER_ORDER),) or not np.isfinite(
        mean_array
    ).all():
        raise LIFPerturbationArtifactError(
            "mean must be a finite five-element vector"
        )
    expected_shape = (len(PARAMETER_ORDER), len(PARAMETER_ORDER))
    if covariance_array.shape != expected_shape or not np.isfinite(
        covariance_array
    ).all():
        raise LIFPerturbationArtifactError(
            "covariance must be a finite 5x5 matrix"
        )
    if not np.allclose(covariance_array, covariance_array.T, rtol=0.0, atol=1e-12):
        raise LIFPerturbationArtifactError("covariance must be symmetric")
    try:
        eigenvalues = np.linalg.eigvalsh(covariance_array)
    except np.linalg.LinAlgError as error:
        raise LIFPerturbationArtifactError("covariance eigendecomposition failed") from error
    if np.any(eigenvalues < -1e-12):
        raise LIFPerturbationArtifactError(
            "covariance must be positive-semidefinite"
        )

    domain = _mapping(
        payload.get("supported_operating_domain"),
        "supported_operating_domain",
    )
    supported_timestep_s = _finite_float(domain, "timestep_s")
    supported_input_min = _finite_float(domain, "input_min")
    supported_input_max = _finite_float(domain, "input_max")
    if supported_timestep_s <= 0:
        raise LIFPerturbationArtifactError("timestep_s must be positive")
    if supported_input_min >= supported_input_max:
        raise LIFPerturbationArtifactError("input_min must be less than input_max")

    dynamic_noise = payload.get("dynamic_noise")
    dynamic_noise_std = None
    transient_source_sha256 = None
    if dynamic_noise is not None:
        dynamic_noise_mapping = _mapping(dynamic_noise, "dynamic_noise")
        dynamic_noise_std = _finite_float(dynamic_noise_mapping, "std")
        if dynamic_noise_std < 0:
            raise LIFPerturbationArtifactError(
                "dynamic noise std must be nonnegative"
            )
        try:
            transient_source = dynamic_noise_mapping["transient_source_sha256"]
        except KeyError as error:
            raise LIFPerturbationArtifactError(
                "dynamic noise requires transient characterization"
            ) from error
        try:
            transient_source_sha256 = _sha256(
                transient_source, "transient_source_sha256"
            )
        except LIFPerturbationArtifactError as error:
            raise LIFPerturbationArtifactError(
                "dynamic noise requires valid transient characterization"
            ) from error

    artifact_checksum = _sha256(
        payload.get("artifact_checksum"), "artifact_checksum"
    )
    checksum_body = {
        key: payload[key]
        for key in sorted(payload)
        if key != "artifact_checksum"
    }
    if canonical_json_sha256(checksum_body) != artifact_checksum:
        raise LIFPerturbationArtifactError(
            "LIF perturbation artifact checksum mismatch"
        )

    return LIFPerturbationArtifact(
        schema_version=1,
        profile_id="frontend_lut_lif_lut",
        pdk=pdk,
        model_files=tuple(model_files),
        simulator_version=simulator_version,
        process_corner=process_corner,
        supply_voltage_v=supply_voltage_v,
        temperature_c=temperature_c,
        monte_carlo_sample_count=monte_carlo_sample_count,
        seed_policy=seed_policy,
        parameter_order=parameter_order,
        mean=tuple(float(value) for value in mean_array),
        covariance=tuple(
            tuple(float(value) for value in row) for row in covariance_array
        ),
        circuit_source_sha256=circuit_source_sha256,
        supported_timestep_s=supported_timestep_s,
        supported_input_min=supported_input_min,
        supported_input_max=supported_input_max,
        dynamic_noise_std=dynamic_noise_std,
        transient_source_sha256=transient_source_sha256,
        artifact_checksum=artifact_checksum,
    )


def _validated_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(shape, tuple) or not shape:
        raise LIFPerturbationArtifactError("neuron shape must be a nonempty tuple")
    if any(
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension <= 0
        for dimension in shape
    ):
        raise LIFPerturbationArtifactError(
            "neuron shape dimensions must be positive integers"
        )
    return shape


class LIFPerturbationRuntime:
    def __init__(self, artifact: LIFPerturbationArtifact, *, seed: int):
        if not isinstance(artifact, LIFPerturbationArtifact):
            raise LIFPerturbationArtifactError(
                "runtime requires a validated LIF perturbation artifact"
            )
        if seed != 112:
            raise LIFPerturbationArtifactError(
                "formal LIF perturbation seed must be 112"
            )
        self.artifact = artifact
        self.seed = seed
        self._static_cache: dict[tuple[str, tuple[int, ...]], dict[str, np.ndarray]] = {}
        self._dynamic_generators: dict[str, np.random.Generator] = {}

    def _namespace_seed(self, layer_id: str, purpose: str) -> int:
        digest = canonical_json_sha256(
            {
                "artifact": self.artifact.artifact_checksum,
                "seed": self.seed,
                "layer_id": layer_id,
                "purpose": purpose,
            }
        )
        return int(digest[:16], 16)

    @staticmethod
    def _layer_id(layer_id: str) -> str:
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise LIFPerturbationArtifactError(
                "layer_id must be a nonempty string"
            )
        return layer_id

    def static_parameters(
        self, layer_id: str, neuron_shape: tuple[int, ...]
    ) -> dict[str, np.ndarray]:
        layer_id = self._layer_id(layer_id)
        neuron_shape = _validated_shape(neuron_shape)
        key = (layer_id, neuron_shape)
        if key not in self._static_cache:
            rng = np.random.default_rng(self._namespace_seed(layer_id, "static"))
            count = int(np.prod(neuron_shape))
            sample = rng.multivariate_normal(
                self.artifact.mean,
                self.artifact.covariance,
                size=count,
                check_valid="raise",
            )
            self._static_cache[key] = {
                name: sample[:, index].reshape(neuron_shape)
                for index, name in enumerate(self.artifact.parameter_order)
            }
        return self._static_cache[key]

    def dynamic_noise(
        self, layer_id: str, shape: tuple[int, ...]
    ) -> np.ndarray:
        layer_id = self._layer_id(layer_id)
        shape = _validated_shape(shape)
        if self.artifact.dynamic_noise_std is None:
            return np.zeros(shape, dtype=np.float64)
        if layer_id not in self._dynamic_generators:
            self._dynamic_generators[layer_id] = np.random.default_rng(
                self._namespace_seed(layer_id, "dynamic")
            )
        return self._dynamic_generators[layer_id].normal(
            loc=0.0,
            scale=self.artifact.dynamic_noise_std,
            size=shape,
        )

    def validate_operating_range(self, values: Any) -> None:
        if hasattr(values, "detach"):
            values = values.detach().cpu().numpy()
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise LIFPerturbationArtifactError(
                "frontend operating range requires finite numeric values"
            ) from error
        lower = self.artifact.supported_input_min
        upper = self.artifact.supported_input_max
        if (
            array.size == 0
            or not np.isfinite(array).all()
            or np.any(array < lower)
            or np.any(array > upper)
        ):
            raise LIFPerturbationArtifactError(
                "frontend input is outside the qualified operating range "
                f"[{lower}, {upper}]"
            )


__all__ = [
    "LIFPerturbationArtifact",
    "LIFPerturbationArtifactError",
    "LIFPerturbationRuntime",
    "PARAMETER_ORDER",
    "load_lif_perturbation_artifact",
]
