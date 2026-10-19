#!/usr/bin/env python3
"""Inspect spike flow and gradients in one V7 residual-SNN batch."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from MNSIM.Interface.network import get_net
from MNSIM.Interface.snn_v7_training_data import training_loaders
from scripts.train_snn_v7 import _hardware_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    loader, _, _, _ = training_loaders(args.dataset_root, "pretrain", args.batch, args.batch)
    net = get_net(_hardware_config(ROOT / "SimConfig.ini"), args.model, 8)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    net.load_state_dict(saved.get("model", saved), strict=True)
    net.to(args.device).train()

    stats = []
    hooks = []
    for index, layer in enumerate(net.layer_list):
        if layer.layer_config["type"] != "lif":
            continue
        entry = {"index": index}
        stats.append(entry)

        def prehook(module, inputs, row=entry):
            value = inputs[0].detach()
            row["input_mean"] = float(value.mean())
            row["input_abs_max"] = float(value.abs().max())

        def posthook(module, inputs, output, row=entry):
            row["spike_rate"] = float(output.detach().mean())

        hooks.append(layer.layer.node.register_forward_pre_hook(prehook))
        hooks.append(layer.layer.node.register_forward_hook(posthook))

    images, labels = next(iter(loader))
    images = images.to(args.device)
    labels = labels.to(args.device)
    output = net.sequence_accuracy_forward(images, weighted_mode="ORIGINAL")
    loss = nn.CrossEntropyLoss()(output, labels)
    loss.backward()

    print(f"model={args.model} loss={loss.item():.6f} output_std={output.std().item():.6f}")
    for row in stats:
        print(
            f"lif={row['index']:03d} input_mean={row['input_mean']:.6f} "
            f"input_abs_max={row['input_abs_max']:.6f} spike_rate={row['spike_rate']:.6f}"
        )
    weighted = [
        (i, parameter)
        for i, layer in enumerate(net.layer_list)
        if layer.layer_config["type"] in {"conv", "fc"}
        for parameter in layer.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    for i, parameter in weighted[:2] + weighted[-2:]:
        grad = parameter.grad
        print(f"weight={i:03d} grad_abs_max={float(grad.abs().max()) if grad is not None else 'none'}")
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
