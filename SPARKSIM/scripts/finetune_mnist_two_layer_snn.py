#!/usr/bin/env python3
"""Fixed-LSB software STE fine-tuning; no circuit simulator is invoked."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_mnist_two_layer_snn import (
    MNIST4x4, IndexedDataset, TwoLayerSNN, accuracy_and_confusion,
    export_samples, file_sha256, seed_everything, tensor_sha256,
    train_epoch, write_exports,
)


class STEQuantizedLinear(nn.Linear):
    """One linear layer with fixed-LSB fake quantization of its weight."""

    def __init__(self, in_features, out_features, lsb):
        super().__init__(in_features, out_features, bias=False)
        self.lsb = float(lsb)

    def forward(self, x):
        clipped = (self.weight / self.lsb).clamp(-255, 255)
        # Forward: exact round/clamp codes. Backward: identity inside the clip.
        rounded = clipped.round()
        codes = rounded.detach() + (clipped - clipped.detach())
        return torch.nn.functional.linear(x, codes * self.lsb, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("inputs/datasets/mnist"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_everything(42)
    source, output = args.source_dir.resolve(), args.output_dir.resolve()
    if (output / "checkpoint_deployment_quantized.pth").exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((source / "config.json").read_text())
    initial_metrics = json.loads((source / "metrics.json").read_text())
    baseline = torch.load(source / "checkpoint_best_float.pth", map_location="cpu", weights_only=True)
    lsb1, lsb2 = config["quantization"]["fc1_lsb"], config["quantization"]["fc2_lsb"]
    model = TwoLayerSNN()
    model.fc1 = STEQuantizedLinear(16, 8, lsb1)
    model.fc2 = STEQuantizedLinear(8, 10, lsb2)
    model.load_state_dict(baseline["model_state_dict"], strict=True)
    train_base = MNIST4x4(args.data_root, train=True, download=False)
    test_base = MNIST4x4(args.data_root, train=False, download=False)
    split = np.load(source / "split_indices.npz", allow_pickle=False)
    train_loader = DataLoader(IndexedDataset(train_base, split["training_indices"]),
                              batch_size=128, shuffle=True,
                              generator=torch.Generator().manual_seed(42), num_workers=0)
    val_loader = DataLoader(IndexedDataset(train_base, split["validation_indices"]),
                            batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_base, batch_size=128, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = initial_metrics["quantized"]["validation"]["accuracy"]
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    device = torch.device("cpu")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss, train_accuracy = train_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_accuracy, _ = accuracy_and_confusion(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
               "validation_loss": val_loss, "validation_accuracy": val_accuracy,
               "learning_rate": args.learning_rate, "duration_seconds": time.perf_counter() - start}
        history.append(row)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        (output / "qat_history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"STE epoch {epoch}: train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
              f"seconds={row['duration_seconds']:.1f}", flush=True)
    qinfo = {}
    state = {}
    for layer_name, lsb in [("fc1", lsb1), ("fc2", lsb2)]:
        latent = best_state[f"{layer_name}.weight"]
        codes = (latent / lsb).round().clamp(-255, 255).short()
        qinfo[layer_name] = {"codes": codes, "lsb": lsb}
        state[f"{layer_name}.weight"] = codes.float() * lsb
    deployed = TwoLayerSNN().eval()
    deployed.load_state_dict(state, strict=True)
    val_loss, val_acc, val_conf = accuracy_and_confusion(deployed, val_loader, device, criterion)
    if val_acc != best_accuracy:
        raise RuntimeError("baked deployment weights differ from fake-quantized validation")
    # The held-out test is evaluated after validation-only checkpoint selection.
    test_loss, test_acc, test_conf = accuracy_and_confusion(deployed, test_loader, device, criterion)
    float_model = TwoLayerSNN().eval()
    float_model.load_state_dict(baseline["model_state_dict"], strict=True)
    quant_metrics = {
        "validation": {"loss": val_loss, "accuracy": val_acc, "confusion_matrix": val_conf.tolist()},
        "test": {"loss": test_loss, "accuracy": test_acc, "confusion_matrix": test_conf.tolist()},
        "accuracy_delta_vs_float": test_acc - initial_metrics["float"]["test"]["accuracy"],
        "before_ste": initial_metrics["quantized"],
    }
    config.pop("files", None)
    config["quantization"]["ste_finetune"] = True
    config["quantization"]["code_source"] = "best fixed-LSB STE latent weights; float baseline checkpoint unchanged"
    config["qat_training"] = {
        "seed": 42, "optimizer": "AdamW", "learning_rate": args.learning_rate,
        "weight_decay": 1e-4, "batch_size": 128, "epochs": args.epochs, "device": "cpu",
        "num_threads": 1, "num_workers": 0, "deterministic_algorithms": True,
        "best_epoch": best_epoch, "initial_validation_accuracy": initial_metrics["quantized"]["validation"]["accuracy"],
        "best_validation_accuracy": best_accuracy, "history": history,
        "lsb_changed": False, "checkpoint_selection": "validation accuracy only; include initial PTQ as epoch 0",
        "surrogate_weight_gradient": "identity inside [-255*lsb,255*lsb], zero outside clip",
        "trigger": "validation drop of 0.0122 after PTQ; test was not used to decide fine-tuning",
    }
    config["checksums"]["quantized_W1_codes"] = tensor_sha256(qinfo["fc1"]["codes"])
    config["checksums"]["quantized_W2_codes"] = tensor_sha256(qinfo["fc2"]["codes"])
    write_exports(output, float_model, (state, qinfo), config, initial_metrics["float"], quant_metrics)
    quant_path = output / "checkpoint_deployment_quantized.pth"
    checkpoint = torch.load(quant_path, map_location="cpu", weights_only=True)
    checkpoint["quantization_source_state_dict"] = best_state
    torch.save(checkpoint, quant_path)
    export_samples(deployed, float_model, test_base, list(range(10)), device, output)
    shutil.copy2(source / "split_indices.npz", output / "split_indices.npz")
    shutil.copy2(source / "training_history.json", output / "training_history.json")
    source_dir = output / "source"
    source_dir.mkdir(exist_ok=True)
    for filename in ["train_mnist_two_layer_snn.py", Path(__file__).name]:
        shutil.copy2(Path(__file__).parent / filename, source_dir / filename)
    config["files"] = {str(p.relative_to(output)): file_sha256(p) for p in sorted(output.rglob("*"))
                       if p.is_file() and p.name not in ["config.json", "SHA256SUMS", "train.log"]}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = dict(config["files"])
    manifest["config.json"] = file_sha256(output / "config.json")
    (output / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in sorted(manifest.items())))
    print(json.dumps({"best_qat_epoch": best_epoch, "quantized_validation_accuracy": val_acc,
                      "quantized_test_accuracy": test_acc, "lsb1": lsb1, "lsb2": lsb2}, indent=2), flush=True)


if __name__ == "__main__":
    main()
