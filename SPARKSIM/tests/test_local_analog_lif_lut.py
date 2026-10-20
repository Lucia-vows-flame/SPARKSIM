import hashlib
import json

import numpy as np
import pytest

from MNSIM.Hardware_Model.analog_lif_lut import (
    AnalogLutArtifactError,
    AnalogLutInvalidPointError,
    AnalogLutOutOfRangeError,
    load_analog_lif_lut,
)


def _write_artifact(root, *, invalid_frontend=False):
    frontend = root / "frontend_gain112_p10008"
    lif = root / "lif_current_step20ns"
    frontend.mkdir(parents=True)
    lif.mkdir(parents=True)
    arrays = {
        frontend: {
            "ipos_a": np.array([0.0, 2.0]),
            "ineg_a": np.array([0.0, 4.0]),
            "iout_to_lif_a": np.array([[np.nan, 4.0], [2.0, 6.0]])
            if invalid_frontend
            else np.array([[0.0, 4.0], [2.0, 6.0]]),
            "settle_latency_s": np.array([[np.nan, 2.0], [3.0, 4.0]])
            if invalid_frontend
            else np.array([[1.0, 2.0], [3.0, 4.0]]),
            "energy_vdd_j": np.array([[np.nan, 20.0], [30.0, 40.0]])
            if invalid_frontend
            else np.array([[10.0, 20.0], [30.0, 40.0]]),
            "valid": np.array([[0, 1], [1, 1]], dtype=np.uint8)
            if invalid_frontend
            else np.ones((2, 2), dtype=np.uint8),
            "status": np.array([[1, 0], [0, 0]], dtype=np.uint8)
            if invalid_frontend
            else np.zeros((2, 2), dtype=np.uint8),
        },
        lif: {
            "iin_a": np.array([0.0, 2.0]),
            "vmem0_v": np.array([0.0, 1.0]),
            "vmem_next_v": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "spike_margin_v": np.array([[-1.0, 1.0], [1.0, -1.0]]),
            "spike_flag": np.array([[0, 1], [1, 0]], dtype=np.uint8),
            "energy_rails_j": np.array([[5.0, 6.0], [7.0, 8.0]]),
            "valid": np.ones((2, 2), dtype=np.uint8),
            "status": np.zeros((2, 2), dtype=np.uint8),
        },
    }
    manifest_files = []
    for directory, table_arrays in arrays.items():
        metadata_arrays = {}
        for name, values in table_arrays.items():
            path = directory / f"{name}.npy"
            np.save(path, values)
            metadata_arrays[f"{name}.npy"] = {
                "dtype": values.dtype.str,
                "shape": list(values.shape),
            }
            manifest_files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "dtype": values.dtype.str,
                    "shape": list(values.shape),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        metadata = {
            "schema_id": "gain112_separate_frontend_lif_lut/v1",
            "characterization_profile": "synthetic",
            "dimension_order": (
                ["ipos_a", "ineg_a"]
                if directory == frontend
                else ["iin_a", "vmem0_v"]
            ),
            "shape": [2, 2],
            "arrays": metadata_arrays,
        }
        if directory == lif:
            metadata["step_latency_s"] = 2e-8
        path = directory / "metadata.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        manifest_files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (root / "lut_manifest.json").write_text(
        json.dumps(
            {
                "schema_id": "gain112_separate_frontend_lif_lut/v1",
                "files": manifest_files,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_loads_artifact_and_queries_exact_corners(tmp_path):
    bundle = load_analog_lif_lut(_write_artifact(tmp_path / "lut"))

    assert bundle.profile == "synthetic"
    assert bundle.frontend.arrays["ipos_a"].flags.writeable is False
    frontend = bundle.query_frontend(2.0, 4.0)
    lif = bundle.query_lif(2.0, 1.0)

    assert frontend.iout_to_lif_a == pytest.approx(6.0)
    assert frontend.settle_latency_s == pytest.approx(4.0)
    assert frontend.energy_vdd_j == pytest.approx(40.0)
    assert lif.vmem_next_v == pytest.approx(0.4)
    assert lif.spike_margin_v == pytest.approx(-1.0)
    assert lif.spike_flag == 0
    assert lif.energy_rails_j == pytest.approx(8.0)


def test_bilinear_interpolation_and_spike_margin_decision(tmp_path):
    bundle = load_analog_lif_lut(_write_artifact(tmp_path / "lut"))

    frontend = bundle.query_frontend(1.0, 2.0)
    lif = bundle.query_lif(1.0, 0.5)

    assert frontend.iout_to_lif_a == pytest.approx(3.0)
    assert frontend.settle_latency_s == pytest.approx(2.5)
    assert frontend.energy_vdd_j == pytest.approx(25.0)
    assert lif.vmem_next_v == pytest.approx(0.25)
    assert lif.spike_margin_v == pytest.approx(0.0)
    assert lif.spike_flag == 1


def test_array_queries_broadcast_inputs(tmp_path):
    bundle = load_analog_lif_lut(_write_artifact(tmp_path / "lut"))

    result = bundle.query_frontend(
        np.array([0.0, 2.0]), np.array([0.0, 4.0])
    )
    assert result.iout_to_lif_a.shape == (2,)
    assert np.allclose(result.iout_to_lif_a, [0.0, 6.0])


def test_rejects_checksum_mismatch(tmp_path):
    root = _write_artifact(tmp_path / "lut")
    path = root / "frontend_gain112_p10008" / "iout_to_lif_a.npy"
    values = np.load(path)
    values[0, 0] = 99.0
    np.save(path, values)

    with pytest.raises(AnalogLutArtifactError, match="SHA-256"):
        load_analog_lif_lut(root)


def test_rejects_out_of_range_query(tmp_path):
    bundle = load_analog_lif_lut(_write_artifact(tmp_path / "lut"))

    with pytest.raises(AnalogLutOutOfRangeError):
        bundle.query_frontend(-1.0, 0.0)


def test_rejects_invalid_neighbor_query(tmp_path):
    bundle = load_analog_lif_lut(
        _write_artifact(tmp_path / "lut", invalid_frontend=True)
    )

    with pytest.raises(AnalogLutInvalidPointError):
        bundle.query_frontend(0.1, 0.1)


def test_rejects_missing_or_invalid_lif_step_latency(tmp_path):
    cases = (None, 0.0, -1.0, float("nan"), "2e-8")
    for value in cases:
        root = _write_artifact(tmp_path / f"lut-{len(list(tmp_path.iterdir()))}")
        metadata_path = root / "lif_current_step20ns" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if value is None:
            metadata.pop("step_latency_s", None)
        else:
            metadata["step_latency_s"] = value
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(AnalogLutArtifactError, match="step_latency_s"):
            load_analog_lif_lut(root, verify_checksums=False)


def test_lif_step_latency_is_returned_from_valid_metadata(tmp_path):
    lif = load_analog_lif_lut(_write_artifact(tmp_path / "lut")).query_lif(0.0, 0.0)

    assert lif.step_latency_s == pytest.approx(2e-8)


def test_rejects_table_schema_mismatch(tmp_path):
    root = _write_artifact(tmp_path / "lut")
    metadata_path = root / "frontend_gain112_p10008" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["schema_id"] = "wrong/v1"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(AnalogLutArtifactError, match="schema"):
        load_analog_lif_lut(root, verify_checksums=False)


def test_metadata_files_are_required_manifest_entries_without_checksum_verification(
    tmp_path,
):
    root = _write_artifact(tmp_path / "lut")
    manifest_path = root / "lut_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        entry
        for entry in manifest["files"]
        if not entry["path"].endswith("frontend_gain112_p10008/metadata.json")
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AnalogLutArtifactError, match="metadata.json"):
        load_analog_lif_lut(root, verify_checksums=False)


def test_rejects_malformed_manifest_byte_count_locally(tmp_path):
    root = _write_artifact(tmp_path / "lut")
    manifest_path = root / "lut_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["bytes"] = "not-an-integer"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AnalogLutArtifactError, match="bytes"):
        load_analog_lif_lut(root, verify_checksums=False)


@pytest.mark.parametrize(
    ("relative", "field", "replacement", "message"),
    (
        (
            "frontend_gain112_p10008",
            "ipos_a",
            np.array([0.0, 0.0]),
            "strictly increasing",
        ),
        (
            "frontend_gain112_p10008",
            "ineg_a",
            np.array([0.0, 4.0], dtype=np.float32),
            "dtype",
        ),
        (
            "frontend_gain112_p10008",
            "iout_to_lif_a",
            np.array([0.0, 4.0]),
            "shape",
        ),
        (
            "frontend_gain112_p10008",
            "status",
            np.array([[0, 2], [0, 0]], dtype=np.uint8),
            "binary",
        ),
        (
            "frontend_gain112_p10008",
            "energy_vdd_j",
            np.array([[np.nan, 20.0], [30.0, 40.0]]),
            "finite",
        ),
    ),
)
def test_rejects_malformed_axes_dtypes_shapes_status_and_finite_values(
    tmp_path, relative, field, replacement, message
):
    root = _write_artifact(tmp_path / f"lut-{field}")
    np.save(root / relative / f"{field}.npy", replacement)

    with pytest.raises(AnalogLutArtifactError, match=message):
        load_analog_lif_lut(root, verify_checksums=False)
