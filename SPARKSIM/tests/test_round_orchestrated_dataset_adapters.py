import json

import numpy as np
import pytest

from round_orchestrated_dataset_adapters import (
    build_snn_accuracy_dataloaders,
)


def _write_sample(path):
    np.savez(path, x_seq=np.zeros((2, 2, 4, 4), dtype=np.uint8), y=0)


def test_snn_adapter_uses_pretrain_as_logical_train(tmp_path):
    (tmp_path / "pretrain").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "meta.json").write_text(
        json.dumps({"T_bins": 2, "C": 2, "H": 4, "W": 4, "num_classes": 1}),
        encoding="utf-8",
    )
    _write_sample(tmp_path / "pretrain" / "pre.npz")
    _write_sample(tmp_path / "test" / "test.npz")

    calibration, test = build_snn_accuracy_dataloaders(tmp_path)

    assert calibration.dataset.accuracy_split_identity()["split"] == "train"
    assert calibration.dataset.accuracy_split_identity()["resolved_split"] == "pretrain"
    assert test.dataset.accuracy_split_identity()["split"] == "test"


@pytest.mark.parametrize("physical_split", ["train", "finetune"])
def test_snn_adapter_rejects_non_pretrain_calibration(tmp_path, physical_split):
    (tmp_path / physical_split).mkdir()
    _write_sample(tmp_path / physical_split / "sample.npz")

    with pytest.raises(FileNotFoundError, match="pretrain"):
        build_snn_accuracy_dataloaders(tmp_path)
