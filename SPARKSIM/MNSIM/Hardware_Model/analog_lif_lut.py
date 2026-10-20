"""Local artifact loader and bounded interpolator for analog LIF LUTs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class AnalogLutArtifactError(ValueError):
    """The LUT artifact is missing, malformed, or fails integrity checks."""


class AnalogLutQueryError(ValueError):
    """A LUT query cannot be evaluated safely."""


class AnalogLutOutOfRangeError(AnalogLutQueryError):
    """A query coordinate lies outside a LUT axis."""


class AnalogLutInvalidPointError(AnalogLutQueryError):
    """A query cell contains an invalid characterization point."""


@dataclass(frozen=True)
class FrontendQuery:
    iout_to_lif_a: Any
    settle_latency_s: Any
    energy_vdd_j: Any


@dataclass(frozen=True)
class LifQuery:
    vmem_next_v: Any
    spike_margin_v: Any
    spike_flag: Any
    energy_rails_j: Any
    step_latency_s: float


@dataclass(frozen=True)
class _Table:
    name: str
    directory: Path
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.arrays[self.axes[0]].size),
            int(self.arrays[self.axes[1]].size),
        )

    @property
    def axes(self) -> tuple[str, str]:
        return tuple(self.metadata["dimension_order"])  # type: ignore[return-value]


@dataclass(frozen=True)
class AnalogLifLutBundle:
    root: Path
    frontend: _Table
    lif: _Table
    manifest: Mapping[str, Any]
    profile: str
    checksum_verified: bool

    def query_frontend(self, ipos_a: Any, ineg_a: Any) -> FrontendQuery:
        values = _query_table(
            self.frontend,
            (ipos_a, ineg_a),
            ("iout_to_lif_a", "settle_latency_s", "energy_vdd_j"),
        )
        return FrontendQuery(**values)

    def query_lif(self, iin_a: Any, vmem0_v: Any) -> LifQuery:
        values = _query_table(
            self.lif,
            (iin_a, vmem0_v),
            ("vmem_next_v", "spike_margin_v", "energy_rails_j"),
        )
        margin = values["spike_margin_v"]
        flag = np.asarray(margin >= 0, dtype=np.uint8)
        if np.ndim(margin) == 0:
            flag = int(flag)
        values["spike_flag"] = flag
        values["step_latency_s"] = float(self.lif.metadata["step_latency_s"])
        return LifQuery(**values)


_TABLES = {
    "frontend": {
        "directory": "frontend_gain112_p10008",
        "axes": ("ipos_a", "ineg_a"),
        "continuous": ("iout_to_lif_a", "settle_latency_s", "energy_vdd_j"),
        "flags": ("valid", "status"),
    },
    "lif": {
        "directory": "lif_current_step20ns",
        "axes": ("iin_a", "vmem0_v"),
        "continuous": ("vmem_next_v", "spike_margin_v", "energy_rails_j"),
        "flags": ("spike_flag", "valid", "status"),
    },
}


def load_analog_lif_lut(
    root: str | Path,
    verify_checksums: bool = True,
    profile: str | None = None,
) -> AnalogLifLutBundle:
    artifact_root = _resolve_root(root)
    manifest = _read_json(artifact_root / "lut_manifest.json")
    if manifest.get("schema_id") != "gain112_separate_frontend_lif_lut/v1":
        raise AnalogLutArtifactError(
            f"unsupported LUT schema: {manifest.get('schema_id')!r}"
        )
    records = _manifest_records(manifest)
    if verify_checksums:
        _verify_manifest_files(artifact_root, records)
    schema_id = str(manifest["schema_id"])
    frontend = _load_table(artifact_root, "frontend", records, schema_id)
    lif = _load_table(artifact_root, "lif", records, schema_id)
    frontend_profile = str(frontend.metadata.get("characterization_profile", ""))
    lif_profile = str(lif.metadata.get("characterization_profile", ""))
    if not frontend_profile or frontend_profile != lif_profile:
        raise AnalogLutArtifactError("frontend and LIF profiles must match")
    if profile is not None and profile != frontend_profile:
        raise AnalogLutArtifactError(
            f"LUT profile mismatch: {frontend_profile!r} != {profile!r}"
        )
    return AnalogLifLutBundle(
        root=artifact_root,
        frontend=frontend,
        lif=lif,
        manifest=manifest,
        profile=frontend_profile,
        checksum_verified=bool(verify_checksums),
    )


def _resolve_root(root: str | Path) -> Path:
    candidate = Path(root).expanduser()
    if candidate.is_dir() and not (candidate / "lut_manifest.json").is_file():
        nested = candidate / "08_lut"
        if nested.is_dir():
            candidate = nested
    if not candidate.is_dir():
        raise AnalogLutArtifactError(f"LUT root is not a directory: {candidate}")
    if not (candidate / "lut_manifest.json").is_file():
        raise AnalogLutArtifactError(f"missing lut_manifest.json under {candidate}")
    return candidate.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AnalogLutArtifactError(f"cannot read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise AnalogLutArtifactError(f"JSON artifact must be an object: {path}")
    return value


def _manifest_records(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise AnalogLutArtifactError("manifest files must be a non-empty list")
    records: dict[str, Mapping[str, Any]] = {}
    for record in raw_files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise AnalogLutArtifactError("manifest file records must contain paths")
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise AnalogLutArtifactError(f"manifest path escapes root: {relative}")
        key = relative.as_posix()
        if key in records:
            raise AnalogLutArtifactError(f"duplicate manifest path: {key}")
        digest = str(record.get("sha256", "")).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise AnalogLutArtifactError(f"invalid manifest checksum: {key}")
        if "bytes" in record and (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 0
        ):
            raise AnalogLutArtifactError(f"invalid manifest bytes field: {key}")
        records[key] = record
    return records


def _verify_manifest_files(
    root: Path, records: Mapping[str, Mapping[str, Any]]
) -> None:
    for relative, record in records.items():
        path = root / relative
        if not path.is_file():
            raise AnalogLutArtifactError(f"manifest references missing file: {relative}")
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and path.stat().st_size != expected_bytes:
            raise AnalogLutArtifactError(f"byte count mismatch for {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != str(record["sha256"]).lower():
            raise AnalogLutArtifactError(f"SHA-256 mismatch for {relative}")


def _load_table(
    root: Path,
    table_name: str,
    records: Mapping[str, Mapping[str, Any]],
    schema_id: str,
) -> _Table:
    contract = _TABLES[table_name]
    directory = root / contract["directory"]
    metadata_path = directory / "metadata.json"
    metadata_relative = metadata_path.relative_to(root).as_posix()
    if metadata_relative not in records:
        raise AnalogLutArtifactError(
            f"manifest missing required file: {metadata_relative}"
        )
    metadata = _read_json(metadata_path)
    if metadata.get("schema_id") != schema_id:
        raise AnalogLutArtifactError(
            f"{table_name} schema_id does not match manifest schema_id"
        )
    axes = tuple(contract["axes"])
    declared_axes = tuple(metadata.get("dimension_order", ()))
    if declared_axes != axes:
        raise AnalogLutArtifactError(
            f"{table_name} dimension order must be {axes!r}, got {declared_axes!r}"
        )
    names = axes + tuple(contract["continuous"]) + tuple(contract["flags"])
    arrays: dict[str, np.ndarray] = {}
    for name in names:
        path = directory / f"{name}.npy"
        relative = path.relative_to(root).as_posix()
        if relative not in records:
            raise AnalogLutArtifactError(f"manifest missing required file: {relative}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise AnalogLutArtifactError(f"cannot load LUT array {relative}") from error
        expected_dtype = np.uint8 if name in contract["flags"] else np.float64
        if array.dtype != np.dtype(expected_dtype):
            raise AnalogLutArtifactError(
                f"{relative} dtype is {array.dtype}, expected {np.dtype(expected_dtype)}"
            )
        record = records[relative]
        if record.get("dtype") is not None and record["dtype"] != array.dtype.str:
            raise AnalogLutArtifactError(
                f"{relative} manifest dtype is {record['dtype']!r}, expected {array.dtype.str!r}"
            )
        if record.get("shape") is not None and tuple(record["shape"]) != array.shape:
            raise AnalogLutArtifactError(
                f"{relative} manifest shape is {record['shape']!r}, expected {array.shape!r}"
            )
        arrays[name] = array
    _validate_table(table_name, metadata, arrays, axes, contract)
    return _Table(table_name, directory, metadata, arrays)


def _validate_table(
    table_name: str,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    axes: tuple[str, str],
    contract: Mapping[str, Any],
) -> None:
    for axis in axes:
        values = arrays[axis]
        if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
            raise AnalogLutArtifactError(f"{table_name}/{axis} must be finite 1-D")
        if np.any(np.diff(values) <= 0):
            raise AnalogLutArtifactError(f"{table_name}/{axis} must be strictly increasing")
    shape = (arrays[axes[0]].size, arrays[axes[1]].size)
    for name in tuple(contract["continuous"]) + tuple(contract["flags"]):
        if arrays[name].shape != shape:
            raise AnalogLutArtifactError(
                f"{table_name}/{name} shape is {arrays[name].shape}, expected {shape}"
            )
    valid = arrays["valid"]
    status = arrays["status"]
    if not np.all(np.isin(valid, (0, 1))) or not np.all(np.isin(status, (0, 1))):
        raise AnalogLutArtifactError(f"{table_name} valid/status flags are not binary")
    if not np.array_equal(valid == 1, status == 0):
        raise AnalogLutArtifactError(f"{table_name}/valid must equal (status == 0)")
    valid_mask = valid == 1
    for name in contract["continuous"]:
        values = arrays[name]
        if np.any(~np.isfinite(values[valid_mask])):
            raise AnalogLutArtifactError(f"{table_name}/{name} has invalid finite values")
        if np.any(~np.isnan(values[~valid_mask])):
            raise AnalogLutArtifactError(
                f"{table_name}/{name} must be NaN at invalid points"
            )
    metadata_arrays = metadata.get("arrays")
    if not isinstance(metadata_arrays, dict):
        raise AnalogLutArtifactError(f"{table_name} metadata arrays must be an object")
    for name, values in arrays.items():
        record = metadata_arrays.get(f"{name}.npy")
        if not isinstance(record, dict):
            raise AnalogLutArtifactError(
                f"{table_name} metadata is missing array record for {name}"
            )
        if record.get("dtype") != values.dtype.str or tuple(record.get("shape", ())) != values.shape:
            raise AnalogLutArtifactError(f"{table_name} metadata array mismatch for {name}")
    if table_name == "lif":
        step_latency = metadata.get("step_latency_s")
        if (
            isinstance(step_latency, bool)
            or not isinstance(step_latency, (int, float))
            or not np.isfinite(step_latency)
            or step_latency <= 0
        ):
            raise AnalogLutArtifactError(
                "lif metadata step_latency_s must be finite and strictly positive"
            )
        spike_flag = arrays["spike_flag"]
        if np.any(spike_flag[~valid_mask] != 255):
            raise AnalogLutArtifactError("lif/spike_flag must be 255 at invalid points")
        if not np.array_equal(
            spike_flag[valid_mask],
            (arrays["spike_margin_v"][valid_mask] >= 0).astype(np.uint8),
        ):
            raise AnalogLutArtifactError(
                "lif/spike_flag must equal (spike_margin_v >= 0)"
            )
    declared_shape = metadata.get("shape")
    if declared_shape is not None and tuple(declared_shape) != shape:
        raise AnalogLutArtifactError(f"{table_name} metadata shape mismatch")


def _query_table(
    table: _Table,
    coordinates: tuple[Any, Any],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    axes = table.axes
    values = tuple(np.asarray(value, dtype=float) for value in coordinates)
    try:
        broadcast = np.broadcast_arrays(*values)
    except ValueError as error:
        raise AnalogLutQueryError("query coordinates cannot be broadcast") from error
    for axis, coordinate in zip(axes, broadcast):
        axis_values = table.arrays[axis]
        if np.any(~np.isfinite(coordinate)):
            raise AnalogLutQueryError("query coordinates must be finite")
        if np.any((coordinate < axis_values[0]) | (coordinate > axis_values[-1])):
            raise AnalogLutOutOfRangeError(
                f"query coordinate is outside {axis}=[{axis_values[0]}, {axis_values[-1]}]"
            )
    lower_indices = []
    fractions = []
    for axis, coordinate in zip(axes, broadcast):
        axis_values = table.arrays[axis]
        upper = np.searchsorted(axis_values, coordinate, side="right")
        upper = np.clip(upper, 1, axis_values.size - 1)
        lower = upper - 1
        denominator = axis_values[upper] - axis_values[lower]
        fraction = (coordinate - axis_values[lower]) / denominator
        lower_indices.append(lower)
        fractions.append(fraction)
    i0, i1 = lower_indices
    f0, f1 = fractions
    neighbors = (
        table.arrays["valid"][i0, i1],
        table.arrays["valid"][i0, np.clip(i1 + 1, 0, table.arrays["valid"].shape[1] - 1)],
        table.arrays["valid"][np.clip(i0 + 1, 0, table.arrays["valid"].shape[0] - 1), i1],
        table.arrays["valid"][
            np.clip(i0 + 1, 0, table.arrays["valid"].shape[0] - 1),
            np.clip(i1 + 1, 0, table.arrays["valid"].shape[1] - 1),
        ],
    )
    if any(np.any(neighbor == 0) for neighbor in neighbors):
        raise AnalogLutInvalidPointError("query interpolation cell contains invalid point")
    i0_upper = np.clip(i0 + 1, 0, table.arrays[axes[0]].size - 1)
    i1_upper = np.clip(i1 + 1, 0, table.arrays[axes[1]].size - 1)
    result: dict[str, Any] = {}
    for field in fields:
        data = table.arrays[field]
        interpolated = (
            data[i0, i1] * (1 - f0) * (1 - f1)
            + data[i0, i1_upper] * (1 - f0) * f1
            + data[i0_upper, i1] * f0 * (1 - f1)
            + data[i0_upper, i1_upper] * f0 * f1
        )
        result[field] = (
            float(interpolated) if interpolated.ndim == 0 else interpolated
        )
    return result


__all__ = [
    "AnalogLutArtifactError",
    "AnalogLutInvalidPointError",
    "AnalogLutOutOfRangeError",
    "AnalogLutQueryError",
    "AnalogLifLutBundle",
    "FrontendQuery",
    "LifQuery",
    "load_analog_lif_lut",
]
