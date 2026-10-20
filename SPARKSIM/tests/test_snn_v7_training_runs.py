import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from MNSIM.Interface.train_snn_v7 import train_net


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def sequence_accuracy_forward(self, images, weighted_mode="ORIGINAL"):
        value = images.flatten(start_dim=1).mean(dim=1) * self.weight
        return torch.stack((value, -value), dim=1)


def _loader(label):
    images = torch.ones(2, 1, 1, 1, 1)
    labels = torch.full((2,), label, dtype=torch.long)
    return DataLoader(TensorDataset(images, labels), batch_size=2)


def test_validation_selects_checkpoint_and_resume_preserves_history(tmp_path):
    train, validation, test = _loader(0), _loader(0), _loader(1)
    options = dict(
        device="cpu", prefix="tiny", epochs=1, learning_rate=0.0,
        output_dir=tmp_path, test_loader=test,
        run_config={"splits": ["pretrain", "calibration", "test"]},
    )
    first = train_net(TinyNet(), train, validation, **options)
    assert first["best_accuracy"] == 1.0
    assert first["test_accuracy"] == 0.0
    report = json.loads((tmp_path / "tiny_software_pretrain_history.json").read_text())
    assert report["completed_epochs"] == 1
    assert report["history"][0]["validation_accuracy"] == 1.0
    assert report["history"][0]["train_accuracy"] == 1.0
    assert report["history"][0]["learning_rate"] == 0.0
    assert report["test"]["accuracy"] == 0.0
    latest = torch.load(first["latest_checkpoint"], weights_only=True)
    assert latest["epoch"] == 1
    assert "optimizer" in latest and "scheduler" in latest
    assert latest["cuda_rng_state"] is None

    with pytest.raises(FileExistsError):
        train_net(TinyNet(), train, validation, **options)
    Path(first["best_checkpoint"]).unlink()
    resumed = train_net(
        TinyNet(), train, validation, **{**options, "epochs": 2, "resume": True}
    )
    assert [row["epoch"] for row in resumed["history"]] == [1, 2]
    assert Path(resumed["best_checkpoint"]).is_file()
    assert json.loads((tmp_path / "tiny_software_pretrain_history.json").read_text())[
        "completed_epochs"
    ] == 2
    latest_before = Path(resumed["latest_checkpoint"]).read_bytes()
    with pytest.raises(ValueError, match="already reached epoch 2"):
        train_net(TinyNet(), train, validation, **{**options, "epochs": 2, "resume": True})
    assert Path(resumed["latest_checkpoint"]).read_bytes() == latest_before


def test_training_requires_separate_test_loader(tmp_path):
    with pytest.raises(ValueError, match="test_loader"):
        train_net(TinyNet(), _loader(0), _loader(0), "cpu", "tiny", output_dir=tmp_path)


def test_initial_checkpoint_survives_degrading_finetune_and_resume(tmp_path):
    # Training deliberately contradicts validation. Selection must retain the
    # already-good initial model, including after restarting the fine-tune.
    options = dict(device="cpu", prefix="tiny", learning_rate=10., momentum=0.,
                   weight_decay=0., milestones=(), output_dir=tmp_path,
                   test_loader=_loader(0), evaluate_initial=True)
    result = train_net(TinyNet(), _loader(1), _loader(0), epochs=1, **options)
    state = torch.load(result["best_checkpoint"], weights_only=True)
    assert state["weight"].item() == 1.
    report = json.loads((tmp_path / "tiny_software_pretrain_history.json").read_text())
    assert report["best_epoch"] == 0 and report["initial_validation"]["accuracy"] == 1.
    assert report["history"][0]["validation_accuracy"] == 0.
    result = train_net(TinyNet(), _loader(1), _loader(0), epochs=2, resume=True, **options)
    assert torch.load(result["best_checkpoint"], weights_only=True)["weight"].item() == 1.
    assert result["test_accuracy"] == 1.


def test_freeze_batchnorm_retains_running_stats_and_trains_affine(tmp_path):
    class BatchNormNet(TinyNet):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(1)

        def sequence_accuracy_forward(self, images, weighted_mode="ORIGINAL"):
            value = self.bn(images.flatten(1)).squeeze(1) * self.weight
            return torch.stack((value, -value), dim=1)

    net = BatchNormNet()
    result = train_net(net, _loader(1), _loader(0), "cpu", "bn", epochs=1,
                       learning_rate=.1, output_dir=tmp_path, test_loader=_loader(0),
                       freeze_batchnorm=True)
    latest = torch.load(result["latest_checkpoint"], weights_only=True)["model"]
    assert torch.equal(latest["bn.running_mean"], torch.zeros(1))
    assert torch.equal(latest["bn.running_var"], torch.ones(1))
    assert latest["bn.num_batches_tracked"].item() == 0
    assert not torch.equal(latest["bn.weight"], torch.ones(1))


def test_validation_target_stops_after_min_epochs_and_still_assesses_test(tmp_path):
    result = train_net(TinyNet(), _loader(0), _loader(0), "cpu", "tiny", epochs=5,
                       learning_rate=0., output_dir=tmp_path, test_loader=_loader(1),
                       target_validation_accuracy=.95, min_epochs=2)
    report = json.loads((tmp_path / "tiny_software_pretrain_history.json").read_text())
    assert report["completed_epochs"] == 2
    assert report["stop_reason"] == "validation_target_reached"
    assert result["test_accuracy"] == 0.
    assert len(result["history"]) == 2


@pytest.mark.parametrize("optimizer_name", ["sgd", "adamw"])
def test_resume_matches_uninterrupted_run(tmp_path, optimizer_name):
    images = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1, 1, 1)
    labels = torch.tensor([0, 1, 0, 1])
    train = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=True)
    validation, test = _loader(0), _loader(1)
    options = dict(
        device="cpu", prefix="tiny", learning_rate=0.1, optimizer_name=optimizer_name, milestones=(1,),
        gamma=0.1, test_loader=test,
    )
    torch.manual_seed(7)
    complete = train_net(
        TinyNet(), train, validation, epochs=3,
        output_dir=tmp_path / "complete", **options
    )
    torch.manual_seed(7)
    train_net(
        TinyNet(), train, validation, epochs=1,
        output_dir=tmp_path / "resumed", **options
    )
    resumed = train_net(
        TinyNet(), train, validation, epochs=3, resume=True,
        output_dir=tmp_path / "resumed", **options
    )
    first_latest = torch.load(complete["latest_checkpoint"], weights_only=True)
    second_latest = torch.load(resumed["latest_checkpoint"], weights_only=True)
    assert complete["history"] == resumed["history"]
    assert complete["history"][0]["learning_rate"] == 0.1
    assert complete["history"][1]["learning_rate"] == pytest.approx(0.01)
    assert torch.equal(first_latest["model"]["weight"], second_latest["model"]["weight"])
