import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import MNSIM.Interface.interface as interface_module
from MNSIM.Accuracy_Model.readc_artifact import (
    ReADCArtifactError,
    ReADCArtifactIdentity,
    build_readc_artifact,
    validate_readc_artifact,
)


def _identity(**changes):
    values = {
        "checkpoint_sha256": "a" * 64,
        "dataset_split_identity": {"dataset": "v7", "split": "train"},
        "device_model_id": "rram_stanford_asu_256",
        "device_seed": 112,
        "weight_mapping_sha256": "b" * 64,
        "adc_precision": 5,
        "percentile_low": 0.5,
        "percentile_high": 99.5,
        "max_samples_per_layer": 8000,
    }
    values.update(changes)
    return ReADCArtifactIdentity(**values)


def _table():
    return {
        "boundaries": [float(index) for index in range(31)],
        "levels": [float(index) + 0.5 for index in range(32)],
        "sample_count": 32,
        "observed_count": 64,
        "observed_min": 0.0,
        "observed_max": 31.0,
        "iterations": 101,
        "mse": 0.1,
        "sampling_strategy": "stream_weighted_deterministic_fixed_capacity",
    }


def test_readc_artifact_round_trip_binds_identity_and_layer_tables():
    artifact = build_readc_artifact(_identity(), {"L000": _table()})
    validated = validate_readc_artifact(artifact.to_dict(), _identity())
    assert validated.artifact_checksum == artifact.artifact_checksum
    assert validated.layers["L000"]["table_checksum"]


@pytest.mark.parametrize(
    "changed_identity",
    [
        _identity(checkpoint_sha256="c" * 64),
        _identity(device_model_id="rram_memtorch_stanford_pku_256"),
        _identity(device_seed=113),
        _identity(weight_mapping_sha256="d" * 64),
    ],
)
def test_readc_artifact_rejects_any_runtime_identity_mismatch(changed_identity):
    artifact = build_readc_artifact(_identity(), {"L000": _table()})
    with pytest.raises(ReADCArtifactError, match="identity mismatch"):
        validate_readc_artifact(artifact.to_dict(), changed_identity)


def test_readc_artifact_rejects_tampered_or_malformed_tables():
    artifact = build_readc_artifact(_identity(), {"L000": _table()}).to_dict()
    artifact["layers"]["L000"]["boundaries"][1] = -10.0
    with pytest.raises(ReADCArtifactError):
        validate_readc_artifact(artifact, _identity())

    malformed = copy.deepcopy(artifact)
    malformed["layers"]["L000"].pop("levels")
    with pytest.raises(ReADCArtifactError):
        validate_readc_artifact(malformed, _identity())


@pytest.mark.parametrize(
    "changed_identity",
    [
        _identity(device_seed=113),
        _identity(adc_precision=4),
        _identity(percentile_low=1.0),
        _identity(percentile_high=99.0),
        _identity(max_samples_per_layer=7999),
        _identity(dataset_split_identity={"dataset": "v7", "split": "test"}),
    ],
)
def test_formal_readc_artifact_rejects_unlocked_calibration_identity(
    changed_identity,
):
    with pytest.raises(ReADCArtifactError, match="formal ReADC artifact"):
        build_readc_artifact(changed_identity, {"L000": _table()})


def test_readc_artifact_reexports_identity_hash_helpers(tmp_path):
    from MNSIM.Accuracy_Model.readc_artifact import (
        bit_weights_sha256,
        file_sha256,
    )

    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"checkpoint")
    weights = [{"weight": np.array([[1, 0], [0, 1]], dtype=np.int8)}]

    assert len(file_sha256(checkpoint)) == 64
    assert len(bit_weights_sha256(weights)) == 64


def test_calibration_clears_stale_artifact_and_tables_before_collection(monkeypatch):
    split_identity = {"dataset": "v7", "split": "train"}
    dataset = SimpleNamespace(accuracy_split_identity=lambda: split_identity)

    class CalibrationLoader:
        def __init__(self):
            self.dataset = dataset

        def __iter__(self):
            return iter([(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))])

    layer = SimpleNamespace(
        hardware_config={"ADC_readc_table": {"stale": True}},
        set_weights_forward=lambda *_args, **_kwargs: None,
    )
    network = SimpleNamespace(
        layer_list=[layer],
        to=lambda _device: None,
        eval=lambda: None,
    )
    context = interface_module.TrainTestInterface.__new__(
        interface_module.TrainTestInterface
    )
    context.dataset_module = "fake_dataset"
    context.calibration_loader = None
    context.device = torch.device("cpu")
    context.net = network
    context.readc_artifact = object()

    def forward(*_args, **_kwargs):
        assert not hasattr(context, "readc_artifact")
        assert "ADC_readc_table" not in layer.hardware_config
        layer.hardware_config["ADC_readc_collector"].add(
            torch.tensor([0.0, 1.0])
        )

    context._accuracy_forward = forward
    monkeypatch.setattr(
        interface_module,
        "import_module",
        lambda _name: SimpleNamespace(
            get_dataloader=lambda: (CalibrationLoader(), None)
        ),
    )

    artifact = context.calibrate_readc(
        [{"weight": np.array([1], dtype=np.int8)}],
        identity=_identity(dataset_split_identity=split_identity),
    )

    assert artifact.identity.dataset_split_identity == split_identity
    assert not hasattr(context, "readc_artifact")
    assert "ADC_readc_table" not in layer.hardware_config


def test_calibration_rejects_nontraining_identity_before_opening_loader(monkeypatch):
    context = interface_module.TrainTestInterface.__new__(
        interface_module.TrainTestInterface
    )
    context.dataset_module = "fake_dataset"
    context.calibration_loader = None
    context.net = SimpleNamespace(layer_list=[])
    monkeypatch.setattr(
        interface_module,
        "import_module",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("non-training identity must fail before opening a loader")
        ),
    )

    with pytest.raises(ReADCArtifactError, match="training dataset split"):
        context.calibrate_readc(
            [],
            identity=_identity(
                dataset_split_identity={"dataset": "v7", "split": "test"}
            ),
        )


def test_malformed_split_identity_fails_closed_as_artifact_error():
    malformed = _identity(dataset_split_identity=None)

    with pytest.raises(ReADCArtifactError, match="dataset split identity"):
        build_readc_artifact(malformed, {"L000": _table()})


@pytest.mark.parametrize(
    "malformed",
    [
        _identity(checkpoint_sha256="short"),
        _identity(weight_mapping_sha256="z" * 64),
        _identity(device_model_id=""),
    ],
)
def test_malformed_readc_identity_fields_fail_closed(malformed):
    with pytest.raises(ReADCArtifactError, match="identity"):
        build_readc_artifact(malformed, {"L000": _table()})


def test_readc_test_inference_rejects_missing_installed_artifact(monkeypatch):
    context = interface_module.TrainTestInterface.__new__(
        interface_module.TrainTestInterface
    )
    context.dataset_module = "fake_dataset"
    context.test_loader = None
    context.hardware_config = {"ADC_quantization_mode": "READC_LLOYD_MAX"}

    monkeypatch.setattr(
        interface_module,
        "import_module",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("test loader must not be opened before artifact validation")
        ),
    )

    with pytest.raises(ReADCArtifactError, match="installed ReADC artifact"):
        context.set_net_bits_evaluate([{"weight": np.array([1])}])
