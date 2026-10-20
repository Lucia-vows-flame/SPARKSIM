import os
from pathlib import Path

import numpy as np
import pytest
import torch
from spikingjelly.activation_based import neuron
from torch.utils.data import SequentialSampler

import MNSIM.Interface.snn_mvm_current as current_backend
from MNSIM.Interface import quantize
from MNSIM.Interface.dvs_binned_npz import (
    DVSBinnedNPZDataset,
    _resize_seq,
    get_calibration_dataloader,
)
from MNSIM.Interface.network import NetworkGraph, get_net
from MNSIM.Scheduling.non_weighted_layers import LIFNode


def _hardware_config():
    return {
        "xbar_size": 128,
        "input_bit": 8,
        "weight_bit": 8,
        "ADC_quantize_bit": 5,
        "DAC_num": 128,
        "xbar_polarity": 2,
        "ADC_quantization_mode": "UNIFORM",
        "ADC_adaptive_thresholds": None,
    }


def _tiny_original_network():
    return NetworkGraph(
        _hardware_config(),
        [
            {
                "type": "conv",
                "in_channels": 1,
                "out_channels": 1,
                "kernel_size": 1,
            },
            {
                "type": "lif",
                "tau": 50.5016666556,
                "v_th": 0.01,
                "v_reset": 0.0,
                "alpha": 2.0,
            },
        ],
        [
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
        ],
        [[-1], [-1]],
        {
            "activation_scale": 1.0,
            "activation_bit": 9,
            "input_shape": (1, 1, 1, 1),
        },
    )


def test_original_path_uses_float_weights_and_resets_spikingjelly(monkeypatch):
    net = _tiny_original_network()
    with torch.no_grad():
        net.layer_list[0].sublayer_list[0].weight.fill_(1.0)
    weight_state_before = net.layer_list[0].bit_scale_list.clone()

    def forbidden_quantize(*_args, **_kwargs):
        raise AssertionError("SNN Original Accuracy Path must not quantize")

    monkeypatch.setattr(quantize, "Quantize", forbidden_quantize)
    sequence = torch.tensor([[[[[1.0]]], [[[1.0]]], [[[0.0]]]]])

    first = net.sequence_accuracy_forward(sequence, weighted_mode="ORIGINAL")
    second = net.sequence_accuracy_forward(sequence, weighted_mode="ORIGINAL")

    assert torch.equal(first, torch.tensor([[[[2.0]]]]))
    assert torch.equal(second, first)
    assert torch.equal(net.layer_list[0].bit_scale_list, weight_state_before)
    lif = net.layer_list[1].layer
    assert isinstance(lif.node, neuron.LIFNode)
    assert lif.node.step_mode == "s"


def test_network_assigns_stable_accuracy_ids_and_straight_layer_forwards_them(
    monkeypatch,
):
    net = _tiny_original_network()
    assert [layer.accuracy_layer_id for layer in net.layer_list] == [
        "L000",
        "L001",
    ]
    lif_wrapper = net.layer_list[1]
    calls = []

    def capture(runtime, *, layer_id):
        calls.append((runtime, layer_id))

    monkeypatch.setattr(lif_wrapper.layer, "set_accuracy_perturbation", capture)
    runtime = object()
    lif_wrapper.set_accuracy_perturbation(runtime)

    assert calls == [(runtime, "L001")]


def test_spikingjelly_lif_matches_the_canonical_software_equation():
    lif = LIFNode(tau=50.5016666556, v_th=0.01, v_reset=0.0)
    sequence = torch.tensor([[[1.0]], [[0.2]], [[0.0]], [[1.0]]])

    actual = lif.forward_sequence(sequence)
    voltage = torch.zeros_like(sequence[0])
    expected_steps = []
    for step in sequence:
        voltage = voltage + (step - voltage) / 50.5016666556
        spike = (voltage >= 0.01).to(step.dtype)
        voltage = voltage * (1.0 - spike)
        expected_steps.append(spike)

    assert torch.equal(actual, torch.stack(expected_steps))


def test_original_and_pim_use_the_same_multistep_lif_entry(monkeypatch):
    net = _tiny_original_network()
    with torch.no_grad():
        weighted = net.layer_list[0]
        weighted.sublayer_list[0].weight.fill_(1.0)
        weighted.bit_scale_list[1, 1] = 1.0
        weighted.last_value.fill_(1.0)
    net.eval()
    calls = []
    perturbations = []
    lif = net.layer_list[1].layer
    original = lif.forward_sequence

    def capture(value):
        calls.append(tuple(value.shape))
        return original(value)

    def capture_perturbation(runtime, *, layer_id):
        perturbations.append((runtime, layer_id))

    monkeypatch.setattr(lif, "forward_sequence", capture)
    monkeypatch.setattr(lif, "set_accuracy_perturbation", capture_perturbation)
    sequence = torch.tensor([[[[[1.0]]], [[[0.0]]]]])
    net.sequence_accuracy_forward(sequence, weighted_mode="ORIGINAL")
    runtime = object()
    net.sequence_accuracy_forward(
        sequence,
        weighted_mode="PIM",
        adc_action="SCALE",
        net_bit_weights=net.get_weights(),
        pim_profile="adc_digital_lif",
        lif_perturbation=runtime,
    )

    assert calls == [(2, 1, 1, 1, 1), (2, 1, 1, 1, 1)]
    assert perturbations == [(None, "L001"), (runtime, "L001")]


def test_original_and_lut_pim_accuracy_never_call_physical_current(monkeypatch):
    def forbidden_physical_current(*_args, **_kwargs):
        raise AssertionError("physical current must not enter accuracy")

    monkeypatch.setattr(
        current_backend, "mvm_current_from_bit_weights", forbidden_physical_current
    )
    monkeypatch.setattr(
        current_backend,
        "mvm_column_load_from_bit_weights",
        forbidden_physical_current,
    )
    net = _tiny_original_network()
    weighted = net.layer_list[0]
    with torch.no_grad():
        weighted.sublayer_list[0].weight.fill_(1.0)
        weighted.bit_scale_list[1, 1] = 1.0
        weighted.last_value.fill_(1.0)
    net.eval()
    sequence = torch.tensor([[[[[1.0]]], [[[0.0]]]]])

    original = net.sequence_accuracy_forward(sequence, weighted_mode="ORIGINAL")
    pim = net.sequence_accuracy_forward(
        sequence,
        weighted_mode="PIM",
        adc_action=None,
        net_bit_weights=net.get_weights(),
        pim_profile="frontend_lut_lif_lut",
        lif_perturbation=None,
    )

    assert set(original.unique().tolist()) <= {0.0, 1.0, 2.0}
    assert set(pim.unique().tolist()) <= {0.0, 1.0, 2.0}


def test_accuracy_modules_have_no_physical_current_or_hardware_lut_imports():
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "MNSIM/Interface/network.py",
        "MNSIM/Interface/quantize.py",
        "MNSIM/Scheduling/non_weighted_layers.py",
    ):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "snn_mvm_current" not in source
        assert "mvm_current_from_bit_weights" not in source
        assert "frontend_lut_lookup" not in source
        assert "lif_lut_lookup" not in source


def test_v7_spatial_reduction_is_binary_or_not_nearest_neighbor():
    sequence = torch.zeros(1, 6, 4, 4)
    sequence[0, 0, 1, 1] = 1.0
    sequence[0, 1, 3, 3] = 1.0

    reduced = _resize_seq(sequence, (2, 2))

    assert reduced.shape == (1, 6, 2, 2)
    assert reduced[0, 0, 0, 0] == 1.0
    assert reduced[0, 1, 1, 1] == 1.0
    assert set(reduced.unique().tolist()) <= {0.0, 1.0}


def test_v7_dataset_split_identity_is_deterministic_and_split_specific(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "meta.json").write_text(
        '{"T_bins": 2, "C": 6, "H": 4, "W": 4, "num_classes": 8}',
        encoding="utf-8",
    )
    sample = np.zeros((2, 6, 4, 4), dtype=np.uint8)
    np.savez(tmp_path / "train" / "sample.npz", x_seq=sample, y=0)
    np.savez(tmp_path / "test" / "sample.npz", x_seq=sample, y=0)
    train = DVSBinnedNPZDataset(str(tmp_path), "train", target_hw=(2, 2))
    test = DVSBinnedNPZDataset(str(tmp_path), "test", target_hw=(2, 2))

    assert train.accuracy_split_identity() == train.accuracy_split_identity()
    assert train.accuracy_split_identity()["split"] == "train"
    assert train.accuracy_split_identity() != test.accuracy_split_identity()


def test_v7_calibration_alias_keeps_logical_train_split_identity(tmp_path):
    (tmp_path / "pretrain").mkdir()
    (tmp_path / "meta.json").write_text(
        '{"T_bins": 2, "C": 6, "H": 4, "W": 4, "num_classes": 8}',
        encoding="utf-8",
    )
    sample = np.zeros((2, 6, 4, 4), dtype=np.uint8)
    np.savez(tmp_path / "pretrain" / "sample.npz", x_seq=sample, y=0)

    loader = get_calibration_dataloader(root=str(tmp_path))
    identity = loader.dataset.accuracy_split_identity()

    assert identity["split"] == "train"
    assert identity["resolved_split"] == "pretrain"


def test_v7_calibration_loader_does_not_require_a_test_split(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "meta.json").write_text(
        '{"T_bins": 2, "C": 2, "H": 4, "W": 4, "num_classes": 8}',
        encoding="utf-8",
    )
    sample = np.zeros((2, 2, 4, 4), dtype=np.uint8)
    np.savez(tmp_path / "train" / "sample.npz", x_seq=sample, y=0)

    loader = get_calibration_dataloader(root=str(tmp_path))

    assert loader.dataset.split == "train"
    assert loader.dataset.accuracy_split_identity()["sample_count"] == 1
    assert isinstance(loader.sampler, SequentialSampler)
    assert loader.drop_last is False

def test_v7_dataset_reads_six_channels_from_metadata(tmp_path):
    (tmp_path / "test").mkdir()
    (tmp_path / "meta.json").write_text(
        '{"T_bins": 2, "C": 6, "H": 4, "W": 4, "num_classes": 8}',
        encoding="utf-8",
    )
    sample = np.zeros((2, 6, 4, 4), dtype=np.uint8)
    sample[:, :, 1, 1] = 1
    np.savez(tmp_path / "test" / "sample.npz", x_seq=sample, y=3)

    dataset = DVSBinnedNPZDataset(str(tmp_path), "test", target_hw=(2, 2))
    sequence, label = dataset[0]

    assert sequence.shape == (2, 6, 2, 2)
    assert label == 3
    assert torch.all(sequence[:, :, 0, 0] == 1)


@pytest.mark.parametrize(
    ("network_id", "checkpoint_name", "conv_count", "lif_count"),
    [
        (
            "standard_vgg8_snn_v7_phase6",
            "standard_vgg8_snn_v7_software_pretrain_best.pth",
            7,
            8,
        ),
        (
            "vgg8_snn_v7_phase6",
            "modified_vgg8_snn_11conv_v7_software_pretrain_best.pth",
            11,
            12,
        ),
    ],
)
def test_final_v7_software_checkpoint_strictly_loads(
    network_id, checkpoint_name, conv_count, lif_count
):
    checkpoint_root = os.environ.get("MNSIM_V7_PRETRAIN_CHECKPOINT_DIR")
    if not checkpoint_root:
        pytest.skip("set MNSIM_V7_PRETRAIN_CHECKPOINT_DIR to run external checkpoint coverage")
    checkpoint = Path(checkpoint_root) / checkpoint_name
    if not checkpoint.is_file():
        pytest.skip(f"external final checkpoint is unavailable: {checkpoint}")
    net = get_net(_hardware_config(), cate=network_id, num_classes=8)

    net.load_change_weights(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )

    layer_types = [layer.layer_config["type"] for layer in net.layer_list]
    assert net.input_params["input_shape"] == (1, 6, 32, 32)
    assert net.sequence_length == 30
    assert layer_types.count("conv") == conv_count
    assert layer_types.count("fc") == 1
    assert layer_types.count("lif") == lif_count
    assert all(
        layer.layer_config["tau"] == 50.5016666556
        and layer.layer_config["v_th"] == 0.01
        and layer.layer_config["v_reset"] == 0.0
        for layer in net.layer_list
        if layer.layer_config["type"] == "lif"
    )
