#!/usr/bin/env python3
"""Train and export the fixed 16-8-10 MNIST SNN deployment artifact.

The script intentionally keeps the software experiment self-contained. It does
not invoke MNSIM or HSPICE; exported signed codes and differential arrays are
the handoff consumed by those flows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from PIL import Image
from spikingjelly.activation_based import functional, neuron, surrogate


T = 30
INPUT_DIM = 16
HIDDEN_DIM = 8
NUM_CLASSES = 10
QUANT_MAX_CODE = 255
SEED = 42


class MNIST4x4(Dataset):
    """Official MNIST with deterministic 28x28 -> 4x4 area averaging."""

    def __init__(self, root: Path, train: bool, download: bool = True):
        raw = MNIST(str(root), train=train, download=download)
        if len(raw) != (60000 if train else 10000):
            raise RuntimeError(
                f"unexpected official MNIST size: train={train} n={len(raw)}"
            )
        self.raw = raw
        pixels = raw.data.to(dtype=torch.float32).div_(255.0).unsqueeze(1)
        self.images = torch.nn.functional.interpolate(
            pixels, size=(4, 4), mode="area"
        ).squeeze(1)
        self.labels = raw.targets.to(dtype=torch.long)

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def deterministic_pulses(x: torch.Tensor, steps: int = T) -> torch.Tensor:
    """Apply s[t,i] = floor((t+1)x[i]) - floor(t*x[i])."""
    if x.ndim < 1:
        raise ValueError("input must have at least one dimension")
    if torch.any(x < 0) or torch.any(x > 1):
        raise ValueError("pulse input must be in [0, 1]")
    t = torch.arange(steps, device=x.device, dtype=x.dtype)
    shape = (steps,) + (1,) * x.ndim
    return torch.floor((t + 1).reshape(shape) * x.unsqueeze(0)) - torch.floor(
        t.reshape(shape) * x.unsqueeze(0)
    )


class TwoLayerSNN(nn.Module):
    """FC1(16,8) -> LIF1 -> FC2(8,10) -> LIF2."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIM, bias=False)
        self.lif1 = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            step_mode="s",
            decay_input=True,
            detach_reset=False,
            backend="torch",
        )
        self.fc2 = nn.Linear(HIDDEN_DIM, NUM_CLASSES, bias=False)
        self.lif2 = neuron.LIFNode(
            tau=2.0,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(alpha=2.0),
            step_mode="s",
            decay_input=True,
            detach_reset=False,
            backend="torch",
        )
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, images: torch.Tensor, return_trace: bool = False):
        if images.ndim != 3 or images.shape[-2:] != (4, 4):
            raise ValueError(f"expected [B,4,4] images, got {tuple(images.shape)}")
        pulses = deterministic_pulses(images.flatten(1), T)
        functional.reset_net(self)
        hidden_trace = []
        output_trace = []
        for pulse_t in pulses:
            hidden = self.lif1(self.fc1(pulse_t))
            output = self.lif2(self.fc2(hidden))
            hidden_trace.append(hidden)
            output_trace.append(output)
        hidden_spikes = torch.stack(hidden_trace, dim=0)
        output_spikes = torch.stack(output_trace, dim=0)
        counts = output_spikes.sum(dim=0)
        if return_trace:
            return counts, pulses, hidden_spikes, output_spikes
        return counts


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset, indices: Iterable[int]):
        self.base = base
        self.indices = list(int(i) for i in indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.base[self.indices[index]]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy_and_confusion(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    confusion = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            counts = model(images)
            loss = criterion(counts, labels)
            predictions = counts.argmax(dim=1)
            total_loss += float(loss.item()) * labels.numel()
            total += int(labels.numel())
            correct += int((predictions == labels).sum().item())
            flat = labels.cpu() * NUM_CLASSES + predictions.cpu()
            confusion += torch.bincount(
                flat, minlength=NUM_CLASSES * NUM_CLASSES
            ).reshape(NUM_CLASSES, NUM_CLASSES)
    return total_loss / total, correct / total, confusion


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        counts = model(images)
        loss = criterion(counts, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        total += int(labels.numel())
        correct += int((counts.detach().argmax(dim=1) == labels).sum().item())
    return total_loss / total, correct / total


def quantize_layer(weight: torch.Tensor):
    max_abs = float(weight.detach().abs().max().item())
    lsb = max_abs / QUANT_MAX_CODE if max_abs else 1.0
    codes = torch.clamp(torch.round(weight.detach().cpu() / lsb), -QUANT_MAX_CODE, QUANT_MAX_CODE).to(torch.int16)
    return codes, float(lsb)


def quantized_state(model: TwoLayerSNN):
    state = copy.deepcopy(model.state_dict())
    w1_codes, lsb1 = quantize_layer(model.fc1.weight)
    w2_codes, lsb2 = quantize_layer(model.fc2.weight)
    state["fc1.weight"] = (w1_codes.to(torch.float32) * lsb1).to(model.fc1.weight.dtype)
    state["fc2.weight"] = (w2_codes.to(torch.float32) * lsb2).to(model.fc2.weight.dtype)
    return state, {"fc1": {"lsb": lsb1, "codes": w1_codes}, "fc2": {"lsb": lsb2, "codes": w2_codes}}


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_samples(
    model: TwoLayerSNN,
    float_model: TwoLayerSNN,
    base_test: MNIST4x4,
    sample_indices: list[int],
    device: torch.device,
    output_dir: Path,
):
    model.eval()
    records = []
    arrays = {}
    sample_dir = output_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    with torch.no_grad():
        for sample_index in sample_indices:
            image4, label = base_test[sample_index]
            raw = base_test.raw.data[sample_index].numpy().astype(np.uint8)
            counts, pulses, hidden, output = model(
                image4.unsqueeze(0).to(device), return_trace=True
            )
            prediction = int(counts.argmax(dim=1).item())
            float_counts, _, float_hidden, float_output = float_model(
                image4.unsqueeze(0).to(device), return_trace=True
            )
            prefix = f"sample_{sample_index:05d}"
            arrays[f"{prefix}_original"] = raw
            arrays[f"{prefix}_label"] = np.asarray(int(label), dtype=np.int64)
            arrays[f"{prefix}_input_4x4"] = image4.numpy().astype(np.float32)
            arrays[f"{prefix}_input_pulses"] = pulses[:, 0].cpu().numpy().astype(np.uint8)
            arrays[f"{prefix}_lif1_spikes"] = hidden[:, 0].cpu().numpy().astype(np.uint8)
            arrays[f"{prefix}_lif2_spikes"] = output[:, 0].cpu().numpy().astype(np.uint8)
            arrays[f"{prefix}_output_counts"] = counts[0].cpu().numpy().astype(np.float32)
            arrays[f"{prefix}_float_lif1_spikes"] = float_hidden[:, 0].cpu().numpy().astype(np.uint8)
            arrays[f"{prefix}_float_lif2_spikes"] = float_output[:, 0].cpu().numpy().astype(np.uint8)
            arrays[f"{prefix}_float_output_counts"] = float_counts[0].cpu().numpy().astype(np.float32)
            Image.fromarray(raw).save(sample_dir / f"{prefix}.png")
            detail = {
                "test_index": sample_index, "label": int(label),
                "original_uint8": raw.tolist(),
                "input_4x4_float32": arrays[f"{prefix}_input_4x4"].tolist(),
                "input_row_major_16": image4.flatten().tolist(),
                "input_pulses_30x16": arrays[f"{prefix}_input_pulses"].tolist(),
                "quantized": {
                    "lif1_spikes_30x8": arrays[f"{prefix}_lif1_spikes"].tolist(),
                    "lif2_spikes_30x10": arrays[f"{prefix}_lif2_spikes"].tolist(),
                    "output_counts": counts[0].cpu().tolist(), "prediction": prediction,
                },
                "float": {
                    "lif1_spikes_30x8": arrays[f"{prefix}_float_lif1_spikes"].tolist(),
                    "lif2_spikes_30x10": arrays[f"{prefix}_float_lif2_spikes"].tolist(),
                    "output_counts": float_counts[0].cpu().tolist(),
                    "prediction": int(float_counts.argmax(dim=1).item()),
                },
            }
            (sample_dir / f"{prefix}.json").write_text(json.dumps(detail, indent=2) + "\n")
            records.append({
                "test_index": sample_index,
                "label": int(label),
                "prediction": prediction,
                "output_counts": counts[0].cpu().tolist(),
                "original_key": f"{prefix}_original",
                "input_4x4_key": f"{prefix}_input_4x4",
                "input_pulses_key": f"{prefix}_input_pulses",
                "lif1_spikes_key": f"{prefix}_lif1_spikes",
                "lif2_spikes_key": f"{prefix}_lif2_spikes",
                "detail_file": f"samples/{prefix}.json",
                "original_png": f"samples/{prefix}.png",
                "float_prediction": int(float_counts.argmax(dim=1).item()),
            })
    np.savez_compressed(output_dir / "test_samples.npz", **arrays)
    (output_dir / "test_samples.json").write_text(
        json.dumps({"sample_indices": sample_indices, "samples": records}, indent=2) + "\n",
        encoding="utf-8",
    )


def padded_differential_arrays(codes: torch.Tensor, rows: int, cols: int):
    """Return positive/negative 16x16 arrays for an [out,in] code matrix."""
    if tuple(codes.shape) != (rows, cols):
        raise ValueError(f"unexpected matrix shape {tuple(codes.shape)}")
    transposed = codes.t().numpy().astype(np.int16)
    padded = np.zeros((16, 16), dtype=np.int16)
    padded[:cols, :rows] = transposed
    return np.maximum(padded, 0), np.maximum(-padded, 0), padded


def write_exports(
    output_dir: Path,
    model: TwoLayerSNN,
    quant_state,
    metadata: dict,
    float_metrics: dict,
    quant_metrics: dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    float_checkpoint = output_dir / "checkpoint_best_float.pth"
    quant_checkpoint = output_dir / "checkpoint_deployment_quantized.pth"
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, float_checkpoint)
    torch.save({"model_state_dict": quant_state[0], "metadata": metadata,
                "quantization": quant_state[1]}, quant_checkpoint)

    w1 = quant_state[1]["fc1"]["codes"]
    w2 = quant_state[1]["fc2"]["codes"]
    p1, n1, a1 = padded_differential_arrays(w1, HIDDEN_DIM, INPUT_DIM)
    p2, n2, a2 = padded_differential_arrays(w2, NUM_CLASSES, HIDDEN_DIM)
    np.savez_compressed(
        output_dir / "quantized_weights.npz",
        W1_codes=w1.numpy().astype(np.int16),
        W2_codes=w2.numpy().astype(np.int16),
        W1_positive_16x16=p1,
        W1_negative_16x16=n1,
        W1_padded_16x16=a1,
        W2_positive_16x16=p2,
        W2_negative_16x16=n2,
        W2_padded_16x16=a2,
        W1_lsb=np.asarray(quant_state[1]["fc1"]["lsb"], dtype=np.float64),
        W2_lsb=np.asarray(quant_state[1]["fc2"]["lsb"], dtype=np.float64),
    )
    (output_dir / "quantized_weights.json").write_text(
        json.dumps({
            "matrix_orientation": "W[out,in] in software; exported arrays are transpose W[in,out]",
            "W1_shape": [HIDDEN_DIM, INPUT_DIM],
            "W2_shape": [NUM_CLASSES, HIDDEN_DIM],
            "W1_lsb": quant_state[1]["fc1"]["lsb"],
            "W2_lsb": quant_state[1]["fc2"]["lsb"],
            "W1_codes": w1.tolist(),
            "W2_codes": w2.tolist(),
            "array_mapping": {
                "W1": "transpose to 16x8, pad columns 8..15 with zero",
                "W2": "transpose to 8x10, pad rows 8..15 and columns 10..15 with zero",
                "positive": "max(code, 0)",
                "negative": "max(-code, 0)",
            },
        }, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps({"float": float_metrics, "quantized": quant_metrics}, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["files"] = {
        name: file_sha256(output_dir / name)
        for name in (
            "checkpoint_best_float.pth",
            "checkpoint_deployment_quantized.pth",
            "quantized_weights.npz",
            "quantized_weights.json",
            "metrics.json",
            "test_samples.npz",
            "test_samples.json",
        )
        if (output_dir / name).exists()
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/mnist_two_layer_snn_20261008"))
    parser.add_argument("--data-root", type=Path, default=Path("inputs/datasets/mnist"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 20 or args.batch_size < 1 or args.num_threads < 1:
        parser.error("epochs must be 1..20; batch-size and num-threads must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(args.num_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_everything(SEED)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA device but CUDA is unavailable")
    output_dir = args.output_dir.resolve()
    if (output_dir / "checkpoint_best_float.pth").exists():
        raise FileExistsError(f"use a new output directory to preserve results: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_base = MNIST4x4(args.data_root, train=True, download=args.download)
    test_base = MNIST4x4(args.data_root, train=False, download=args.download)
    split_generator = torch.Generator().manual_seed(SEED)
    permutation = torch.randperm(len(train_base), generator=split_generator).tolist()
    validation_indices = permutation[:5000]
    training_indices = permutation[5000:]
    np.savez_compressed(output_dir / "split_indices.npz",
                        training_indices=np.asarray(training_indices, dtype=np.int64),
                        validation_indices=np.asarray(validation_indices, dtype=np.int64))
    train_loader = DataLoader(
        IndexedDataset(train_base, training_indices), batch_size=args.batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(SEED),
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        IndexedDataset(train_base, validation_indices), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_base, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    model = TwoLayerSNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_accuracy = -1.0
    best_state = None
    best_epoch = None
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = train_epoch(model, train_loader, device, criterion, optimizer)
        validation_loss, validation_accuracy, _ = accuracy_and_confusion(model, validation_loader, device, criterion)
        row = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "validation_loss": validation_loss, "validation_accuracy": validation_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d}: train_loss={train_loss:.6f} train_accuracy={train_accuracy:.4f} "
            f"validation_loss={validation_loss:.6f} validation_accuracy={validation_accuracy:.4f} "
            f"seconds={row['duration_seconds']:.1f}", flush=True
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save({"model_state_dict": best_state, "epoch": epoch,
                        "validation_accuracy": best_accuracy}, output_dir / "checkpoint_best_float.pth")
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    float_validation_loss, float_validation_accuracy, float_validation_confusion = accuracy_and_confusion(model, validation_loader, device, criterion)
    float_test_loss, float_test_accuracy, float_test_confusion = accuracy_and_confusion(model, test_loader, device, criterion)
    qstate = quantized_state(model)
    quant_model = copy.deepcopy(model).to(device)
    quant_model.load_state_dict(qstate[0], strict=True)
    quant_validation_loss, quant_validation_accuracy, quant_validation_confusion = accuracy_and_confusion(quant_model, validation_loader, device, criterion)
    quant_test_loss, quant_test_accuracy, quant_test_confusion = accuracy_and_confusion(quant_model, test_loader, device, criterion)
    metadata = {
        "experiment": "mnist_two_layer_snn",
        "software_only": True,
        "hspice_executed": False,
        "dataset": {"name": "official MNIST", "train_count": 60000, "validation_count": 5000, "optimization_train_count": 55000, "test_count": 10000, "validation_seed": SEED, "validation_indices_sha256": hashlib.sha256(np.asarray(validation_indices, dtype=np.int64).tobytes()).hexdigest()},
        "preprocessing": {"source_shape": [28, 28], "normalization": "uint8 / 255", "resize": "area average to 4x4", "flatten": "row-major", "input_range": [0.0, 1.0]},
        "pulse_encoding": {"steps": T, "rule": "s[t,i] = floor((t+1)*x[i]) - floor(t*x[i])", "dtype": "binary uint8"},
        "network": {"layers": ["FC1(16,8,bias=False)", "LIF1", "FC2(8,10,bias=False)", "LIF2"], "weight_shapes": {"W1": [8, 16], "W2": [10, 8]}, "weight_orientation": "software tensors use W[out,in]", "classification": "argmax of 30-step output spike counts; ties choose smallest class index"},
        "lif": {"implementation": "spikingjelly.activation_based.neuron.LIFNode", "step_mode": "s", "backend": "torch", "tau": 2.0, "v_threshold": 1.0, "v_reset": 0.0, "reset": "hard", "surrogate": "ATan(alpha=2.0)", "decay_input": True, "detach_reset": False, "initial_voltage": 0.0, "charge_rule": "h[t] = v[t-1] + (I[t] - (v[t-1] - v_reset))/tau", "spike_rule": "z[t] = 1 if h[t] >= 1.0 else 0", "reset_rule": "v[t] = h[t]*(1-z[t]) + v_reset*z[t]", "state_boundary": "reset before each sample; retain state across 30 steps"},
        "training": {"seed": SEED, "optimizer": "AdamW", "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "batch_size": args.batch_size, "max_epochs": args.epochs, "actual_epochs": len(history), "best_epoch": best_epoch, "checkpoint_selection": "highest validation accuracy; ties keep earlier epoch", "loss": "CrossEntropyLoss on unnormalized 30-step output spike counts", "initialization": "Xavier uniform for both weights", "history": history, "device": str(device), "num_workers": args.num_workers, "num_threads": args.num_threads, "deterministic_algorithms": True, "allow_tf32": False, "test_usage": "final evaluation only; no checkpoint or quantization selection from test metrics"},
        "quantization": {"rule": "code=clip(round(weight/lsb), -255, 255); deployment_weight=code*lsb", "rounding": "torch.round: round to nearest, ties to even", "lsb_selection": "max(abs(best float layer weight))/255; fixed per layer thereafter", "signed_bits": 9, "code_min": -255, "code_max": 255, "unused_code": -256, "fc1_lsb": qstate[1]["fc1"]["lsb"], "fc2_lsb": qstate[1]["fc2"]["lsb"], "array_layout": "one positive and one negative differential array per layer; each padded to 16x16; zero is empty cell", "ste_finetune": False},
        "numeric_format": {"training_and_deployment_weights": "float32", "input_normalization_and_area_average": "float32", "pulse_floor_arithmetic": "float32", "codes": "int16 storage, signed 9-bit symmetric values", "confusion_matrix": "row=true label, column=predicted label"},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__), "torchvision": version("torchvision"), "spikingjelly": version("spikingjelly"), "numpy": np.__version__},
        "fixed_trace_samples": list(range(10)),
        "checksums": {"float_W1": tensor_sha256(model.fc1.weight), "float_W2": tensor_sha256(model.fc2.weight), "quantized_W1_codes": hashlib.sha256(qstate[1]["fc1"]["codes"].numpy().tobytes()).hexdigest(), "quantized_W2_codes": hashlib.sha256(qstate[1]["fc2"]["codes"].numpy().tobytes()).hexdigest()},
    }
    float_metrics = {"validation": {"loss": float_validation_loss, "accuracy": float_validation_accuracy, "confusion_matrix": float_validation_confusion.tolist()}, "test": {"loss": float_test_loss, "accuracy": float_test_accuracy, "confusion_matrix": float_test_confusion.tolist()}}
    quant_metrics = {"validation": {"loss": quant_validation_loss, "accuracy": quant_validation_accuracy, "confusion_matrix": quant_validation_confusion.tolist()}, "test": {"loss": quant_test_loss, "accuracy": quant_test_accuracy, "confusion_matrix": quant_test_confusion.tolist()}, "accuracy_delta_vs_float": quant_test_accuracy - float_test_accuracy}
    write_exports(output_dir, model, qstate, metadata, float_metrics, quant_metrics)
    export_samples(quant_model, model, test_base, list(range(10)), device, output_dir)
    source_dir = output_dir / "source"
    source_dir.mkdir(exist_ok=True)
    shutil.copy2(__file__, source_dir / Path(__file__).name)
    metadata["dataset"]["raw_files_sha256"] = {
        p.name: file_sha256(p) for p in sorted(Path(train_base.raw.raw_folder).glob("*-ubyte"))
    }
    metadata["files"] = {
        str(p.relative_to(output_dir)): file_sha256(p)
        for p in sorted(output_dir.rglob("*"))
        if p.is_file() and p.name not in ("config.json", "SHA256SUMS", "train.log")
    }
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    manifest = dict(metadata["files"])
    manifest["config.json"] = file_sha256(output_dir / "config.json")
    (output_dir / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in sorted(manifest.items())))
    print(json.dumps({"output_dir": str(output_dir), "float_test_accuracy": float_test_accuracy, "quantized_test_accuracy": quant_test_accuracy, "fc1_lsb": qstate[1]["fc1"]["lsb"], "fc2_lsb": qstate[1]["fc2"]["lsb"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
