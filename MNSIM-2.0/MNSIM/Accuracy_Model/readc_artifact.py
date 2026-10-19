"""Identity-bound durable artifacts for formal ReADC calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    bit_weights_sha256,
    canonical_json_sha256,
    file_sha256,
)


class ReADCArtifactError(ValueError):
    """A ReADC artifact is malformed, tampered with, or identity-mismatched."""


@dataclass(frozen=True)
class ReADCArtifactIdentity:
    checkpoint_sha256: str
    dataset_split_identity: dict[str, Any]
    device_model_id: str
    device_seed: int
    weight_mapping_sha256: str
    adc_precision: int
    percentile_low: float
    percentile_high: float
    max_samples_per_layer: int


@dataclass(frozen=True)
class ReADCCalibrationArtifact:
    schema_version: int
    identity: ReADCArtifactIdentity
    layers: dict[str, dict[str, Any]]
    artifact_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": asdict(self.identity),
            "layers": self.layers,
            "artifact_checksum": self.artifact_checksum,
        }


def validate_readc_identity(identity: ReADCArtifactIdentity) -> None:
    if not isinstance(identity, ReADCArtifactIdentity):
        raise ReADCArtifactError("invalid ReADC artifact identity")
    for field_name in ("checkpoint_sha256", "weight_mapping_sha256"):
        value = getattr(identity, field_name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value.lower())
        ):
            raise ReADCArtifactError(
                f"malformed ReADC artifact identity field {field_name}"
            )
    if (
        not isinstance(identity.device_model_id, str)
        or not identity.device_model_id.strip()
    ):
        raise ReADCArtifactError(
            "malformed ReADC artifact identity field device_model_id"
        )
    expected = {
        "device_seed": 112,
        "adc_precision": 5,
        "percentile_low": 0.5,
        "percentile_high": 99.5,
        "max_samples_per_layer": 8000,
    }
    for field_name, expected_value in expected.items():
        if getattr(identity, field_name) != expected_value:
            raise ReADCArtifactError(
                "formal ReADC artifact requires "
                f"{field_name}={expected_value}"
            )
    if not isinstance(identity.dataset_split_identity, dict):
        raise ReADCArtifactError("malformed ReADC dataset split identity")
    if identity.dataset_split_identity.get("split") != "train":
        raise ReADCArtifactError(
            "formal ReADC artifact requires the training dataset split"
        )


def _validated_table(table: dict[str, Any], precision: int) -> dict[str, Any]:
    try:
        boundaries = [float(value) for value in table["boundaries"]]
        levels = [float(value) for value in table["levels"]]
        observed_count = int(table.get("observed_count", 0))
    except (KeyError, TypeError, ValueError) as error:
        raise ReADCArtifactError(f"malformed ReADC table: {error}") from error
    if len(levels) != 2**precision or len(boundaries) != len(levels) - 1:
        raise ReADCArtifactError("ReADC table size does not match ADC precision")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ReADCArtifactError("ReADC boundaries must be strictly increasing")
    if observed_count <= 0:
        raise ReADCArtifactError("ReADC calibration observed no samples")
    normalized = dict(table)
    normalized["boundaries"] = boundaries
    normalized["levels"] = levels
    normalized["observed_count"] = observed_count
    normalized["table_checksum"] = canonical_json_sha256(
        {key: normalized[key] for key in sorted(normalized) if key != "table_checksum"}
    )
    return normalized


def build_readc_artifact(
    identity: ReADCArtifactIdentity,
    layers: dict[str, dict[str, Any]],
) -> ReADCCalibrationArtifact:
    validate_readc_identity(identity)
    if not isinstance(layers, dict) or not layers:
        raise ReADCArtifactError("ReADC artifact requires layer tables")
    try:
        normalized_layers = {
            str(layer_id): _validated_table(table, identity.adc_precision)
            for layer_id, table in sorted(layers.items())
        }
        body = {
            "schema_version": 1,
            "identity": asdict(identity),
            "layers": normalized_layers,
        }
        checksum = canonical_json_sha256(body)
    except ReADCArtifactError:
        raise
    except (TypeError, ValueError) as error:
        raise ReADCArtifactError(f"invalid ReADC artifact: {error}") from error
    return ReADCCalibrationArtifact(1, identity, normalized_layers, checksum)


def validate_readc_artifact(
    payload: dict[str, Any],
    expected_identity: ReADCArtifactIdentity,
) -> ReADCCalibrationArtifact:
    try:
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ReADCArtifactError("unsupported ReADC artifact schema")
        actual_identity = ReADCArtifactIdentity(**payload["identity"])
        if actual_identity != expected_identity:
            raise ReADCArtifactError("ReADC artifact identity mismatch")
        rebuilt = build_readc_artifact(actual_identity, payload["layers"])
        if payload.get("artifact_checksum") != rebuilt.artifact_checksum:
            raise ReADCArtifactError("ReADC artifact checksum mismatch")
        for layer_id, table in rebuilt.layers.items():
            if payload["layers"][layer_id].get("table_checksum") != table["table_checksum"]:
                raise ReADCArtifactError(
                    f"ReADC table checksum mismatch for {layer_id}"
                )
        return rebuilt
    except ReADCArtifactError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise ReADCArtifactError(f"malformed ReADC artifact: {error}") from error


__all__ = [
    "ReADCArtifactError",
    "ReADCArtifactIdentity",
    "ReADCCalibrationArtifact",
    "bit_weights_sha256",
    "build_readc_artifact",
    "file_sha256",
    "validate_readc_artifact",
    "validate_readc_identity",
]
