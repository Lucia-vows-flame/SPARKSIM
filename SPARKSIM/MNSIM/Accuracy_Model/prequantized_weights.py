"""Run-local signed weight deployment checkpoints for SNN PIM inference."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from MNSIM.Interface.quantize import (
    decode_prequantized_weight_splits,
    quantize_weight_ste,
)


SCHEMA_VERSION = 1
QUANTIZER_ID = "mnsim_symmetric_signed_weight_v1"
WEIGHT_KEY = re.compile(
    r"^(layer_list\.\d+)\.sublayer_list\.(\d+)\.weight$"
)


class PrequantizedWeightError(ValueError):
    """The source checkpoint or deployment bundle violates its contract."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state_dict(path: str | Path) -> Mapping[str, object]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PrequantizedWeightError(
            f"could not load checkpoint {path}: {error}"
        ) from error
    if not isinstance(state, Mapping) or not state:
        raise PrequantizedWeightError(
            f"checkpoint {path} must contain a non-empty state dict"
        )
    if any(not isinstance(key, str) for key in state):
        raise PrequantizedWeightError(
            f"checkpoint {path} contains a non-string state key"
        )
    return state


def _group_weight_keys(state: Mapping[str, object]):
    groups: dict[str, list[tuple[int, str]]] = {}
    for key in state:
        match = WEIGHT_KEY.fullmatch(key)
        if match is None:
            continue
        groups.setdefault(match.group(1), []).append((int(match.group(2)), key))
    if not groups:
        raise PrequantizedWeightError(
            "checkpoint contains no layer_list.*.sublayer_list.*.weight tensors"
        )
    for layer_id, entries in groups.items():
        entries.sort()
        indices = [index for index, _ in entries]
        if indices != list(range(len(indices))):
            raise PrequantizedWeightError(
                f"{layer_id} has non-contiguous sublayer weight indices {indices}"
            )
    return dict(sorted(groups.items(), key=lambda item: int(item[0].split(".")[-1])))


def _clone_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    return copy.deepcopy(value)


def _values_equal(left, right) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return (
            torch.is_tensor(left)
            and torch.is_tensor(right)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left, right)
        )
    return left == right


def _atomic_torch_save(value, target: Path) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_json_save(value, target: Path) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def prepare_prequantized_deployment(
    source_checkpoint: str | Path,
    deployment_checkpoint: str | Path,
    manifest_path: str | Path,
    *,
    qbit: int = 9,
) -> dict:
    """Quantize each logical layer once and atomically persist the bundle."""
    if int(qbit) < 2:
        raise PrequantizedWeightError(
            f"SNN deployment requires at least 2-bit signed weights, got {qbit}"
        )
    source_path = Path(source_checkpoint).resolve()
    deployment_path = Path(deployment_checkpoint).resolve()
    manifest_target = Path(manifest_path).resolve()
    deployment_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_target.parent.mkdir(parents=True, exist_ok=True)

    source_state = _load_state_dict(source_path)
    groups = _group_weight_keys(source_state)
    deployment_state = OrderedDict(
        (key, _clone_value(value)) for key, value in source_state.items()
    )
    threshold = 2 ** (int(qbit) - 1) - 1
    layer_records = []

    for layer_id, entries in groups.items():
        tensors = []
        for _, key in entries:
            tensor = source_state[key]
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                raise PrequantizedWeightError(
                    f"{key} must be a floating-point tensor"
                )
            if not torch.isfinite(tensor).all():
                raise PrequantizedWeightError(f"{key} contains non-finite values")
            tensors.append(tensor.detach().cpu())
        dtype = tensors[0].dtype
        if any(tensor.dtype != dtype for tensor in tensors):
            raise PrequantizedWeightError(
                f"{layer_id} sublayer weights must share one dtype"
            )

        lengths = [tensor.numel() for tensor in tensors]
        flattened = torch.cat([tensor.reshape(-1) for tensor in tensors])
        quantized, digits, lsb = quantize_weight_ste(flattened, int(qbit))
        quantized_chunks = torch.split(quantized.detach(), lengths)
        digit_chunks = torch.split(digits.detach(), lengths)
        for (_, key), source, quantized_chunk in zip(
            entries, tensors, quantized_chunks
        ):
            deployment_state[key] = quantized_chunk.reshape(source.shape).to(
                dtype=source.dtype
            )

        decoded, decoded_lsb = decode_prequantized_weight_splits(
            [deployment_state[key] for _, key in entries], qbit=int(qbit)
        )
        expected_digits = [
            chunk.reshape(tensor.shape).to(dtype=torch.int64)
            for chunk, tensor in zip(digit_chunks, tensors)
        ]
        if any(
            not torch.equal(actual, expected)
            for actual, expected in zip(decoded, expected_digits)
        ):
            raise PrequantizedWeightError(
                f"{layer_id} failed signed-grid round-trip validation"
            )
        layer_records.append(
            {
                "layer_id": layer_id,
                "split_count": len(entries),
                "weight_value_count": int(sum(lengths)),
                "lsb": float(decoded_lsb),
                "code_min": int(min(chunk.min().item() for chunk in decoded)),
                "code_max": int(max(chunk.max().item() for chunk in decoded)),
            }
        )

    weight_keys = {key for entries in groups.values() for _, key in entries}
    non_weight_unchanged = all(
        _values_equal(source_state[key], deployment_state[key])
        for key in source_state
        if key not in weight_keys
    )
    if not non_weight_unchanged:
        raise PrequantizedWeightError("non-weight checkpoint state changed")

    try:
        _atomic_torch_save(deployment_state, deployment_path)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "quantizer_id": QUANTIZER_ID,
            "weight_bit_width": int(qbit),
            "signed_code_range": [-threshold, threshold],
            "quantization_count": 1,
            "source_checkpoint_path": str(source_path),
            "source_checkpoint_sha256": file_sha256(source_path),
            "deployment_checkpoint_path": str(deployment_path),
            "deployment_checkpoint_sha256": file_sha256(deployment_path),
            "logical_layer_count": len(layer_records),
            "weight_tensor_count": len(weight_keys),
            "all_non_weight_values_unchanged": True,
            "layers": layer_records,
        }
        _atomic_json_save(manifest, manifest_target)
        return validate_prequantized_deployment(
            deployment_path,
            manifest_target,
            qbit=int(qbit),
            expected_source_sha256=manifest["source_checkpoint_sha256"],
        )
    except BaseException:
        deployment_path.unlink(missing_ok=True)
        manifest_target.unlink(missing_ok=True)
        raise


def validate_prequantized_deployment(
    deployment_checkpoint: str | Path,
    manifest_path: str | Path,
    *,
    qbit: int = 9,
    expected_source_sha256: str | None = None,
) -> dict:
    if int(qbit) < 2:
        raise PrequantizedWeightError(
            f"SNN deployment requires at least 2-bit signed weights, got {qbit}"
        )
    deployment_path = Path(deployment_checkpoint).resolve()
    manifest_target = Path(manifest_path).resolve()
    try:
        manifest = json.loads(manifest_target.read_text(encoding="utf-8"))
    except Exception as error:
        raise PrequantizedWeightError(
            f"could not read deployment manifest {manifest_target}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise PrequantizedWeightError("deployment manifest must be a JSON object")
    expected_header = {
        "schema_version": SCHEMA_VERSION,
        "quantizer_id": QUANTIZER_ID,
        "weight_bit_width": int(qbit),
        "signed_code_range": [-(2 ** (int(qbit) - 1) - 1), 2 ** (int(qbit) - 1) - 1],
        "quantization_count": 1,
        "all_non_weight_values_unchanged": True,
    }
    for key, expected in expected_header.items():
        if manifest.get(key) != expected:
            raise PrequantizedWeightError(
                f"deployment manifest field {key} does not match {expected!r}"
            )
    if expected_source_sha256 is not None and manifest.get(
        "source_checkpoint_sha256"
    ) != expected_source_sha256:
        raise PrequantizedWeightError("deployment source checkpoint identity mismatch")
    if manifest.get("deployment_checkpoint_sha256") != file_sha256(deployment_path):
        raise PrequantizedWeightError("deployment checkpoint checksum mismatch")

    state = _load_state_dict(deployment_path)
    groups = _group_weight_keys(state)
    records = manifest.get("layers")
    if not isinstance(records, list):
        raise PrequantizedWeightError("deployment manifest layers must be a list")
    record_map = {record.get("layer_id"): record for record in records if isinstance(record, dict)}
    if len(record_map) != len(records) or set(record_map) != set(groups):
        raise PrequantizedWeightError("deployment manifest layer identities mismatch")

    for layer_id, entries in groups.items():
        try:
            digits, lsb = decode_prequantized_weight_splits(
                [state[key] for _, key in entries], qbit=int(qbit)
            )
        except Exception as error:
            raise PrequantizedWeightError(
                f"invalid deployment weights for {layer_id}: {error}"
            ) from error
        record = record_map[layer_id]
        actual = {
            "split_count": len(entries),
            "weight_value_count": int(sum(chunk.numel() for chunk in digits)),
            "lsb": float(lsb),
            "code_min": int(min(chunk.min().item() for chunk in digits)),
            "code_max": int(max(chunk.max().item() for chunk in digits)),
        }
        for key, value in actual.items():
            if record.get(key) != value:
                raise PrequantizedWeightError(
                    f"deployment manifest mismatch for {layer_id}.{key}"
                )
    return manifest


__all__ = [
    "PrequantizedWeightError",
    "prepare_prequantized_deployment",
    "validate_prequantized_deployment",
]
