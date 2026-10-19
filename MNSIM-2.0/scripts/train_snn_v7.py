#!/usr/bin/env python3
"""Train a sequential V7 SNN software checkpoint."""
from __future__ import annotations

import argparse
import configparser
from functools import partial
import math
import os
import random
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MNSIM.Interface.network import get_net
from MNSIM.Interface.train_snn_v7 import train_net, _forward_sequence
from MNSIM.Interface.snn_v7_training_data import training_loaders


def _hardware_config(path: Path) -> dict:
    config = configparser.ConfigParser()
    if not config.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"SimConfig not found: {path}")
    xbar_size = int(config.get("Crossbar level", "Xbar_Size").split(",")[0])
    device_level = int(config.get("Device level", "Device_Level"))
    return {
        "xbar_size": xbar_size,
        "input_bit": int(config.get("Interface level", "DAC_Precision")),
        "weight_bit": math.floor(math.log2(device_level)),
        "weight_quantization_bit": 9,
        "ADC_quantize_bit": int(config.get("Interface level", "ADC_Precision")),
        "DAC_num": int(config.get("Process element level", "DAC_Num")),
        "xbar_polarity": int(config.get("Process element level", "Xbar_Polarity")),
    }


MODELS = {
    "resnet18_snn_v7_phase6",
    "resnet34_snn_v7_phase6",
    "resnet50_snn_v7_phase6",
    "inception_snn_v7_phase6",
    "lenet_snn_v7_phase6",
    "alexnet_snn_v7_phase6",
    "mobilenet_snn_v7_phase6",
    "vgg8_snn_v7_phase6",
    "vgg11_snn_v7_phase6",
    "vgg13_snn_v7_phase6",
    "vgg16_snn_v7_phase6",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, default=REPO_ROOT / "SimConfig.ini")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "snn_pretrain")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[30, 60])
    parser.add_argument("--gamma", type=float, default=0.06)
    parser.add_argument(
        "--readout", choices=["spike", "membrane"],
        help="override the model default: spike count sum or terminal classifier current mean",
    )
    parser.add_argument(
        "--gradient-clip-value",
        type=float,
        help="clip each gradient element before the optimizer update",
    )
    parser.add_argument(
        "--lif-threshold",
        type=float,
        help="override MNSIM_SNN_LIF_V_TH for this run",
    )
    parser.add_argument(
        "--lif-tau",
        type=float,
        help="override MNSIM_SNN_LIF_TAU for this run",
    )
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--test-batch", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    readout = args.readout or ("membrane" if args.model.startswith("resnet") else "spike")

    if args.lif_threshold is not None:
        if not math.isfinite(args.lif_threshold) or args.lif_threshold <= 0:
            parser.error("--lif-threshold must be finite and positive")
        os.environ["MNSIM_SNN_LIF_V_TH"] = str(args.lif_threshold)
    if args.lif_tau is not None:
        if not math.isfinite(args.lif_tau) or args.lif_tau <= 1:
            parser.error("--lif-tau must be finite and greater than 1")
        os.environ["MNSIM_SNN_LIF_TAU"] = str(args.lif_tau)
    os.environ["MNSIM_DVS_BINNED_ROOT"] = str(args.dataset_root)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_loader, validation_loader, test_loader, identities = training_loaders(
        args.dataset_root, "pretrain", args.train_batch, args.test_batch
    )
    net = get_net(
        hardware_config=_hardware_config(args.sim_config),
        cate=args.model,
        num_classes=8,
    )
    result = train_net(
        net,
        train_loader,
        validation_loader,
        args.device,
        args.model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        milestones=args.milestones,
        gamma=args.gamma,
        forward_fn=(partial(_forward_with_readout, readout=readout)
                    if args.readout is not None else _forward_sequence),
        gradient_clip_value=args.gradient_clip_value,
        output_dir=args.output_dir,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        test_loader=test_loader,
        run_config={"dataset_splits": identities, "seed": args.seed,
                    "train_batch": args.train_batch, "eval_batch": args.test_batch,
                    "sim_config": str(args.sim_config.resolve()), "stage": "pretrain",
                    "lif_threshold": float(os.environ.get("MNSIM_SNN_LIF_V_TH", "0.01")),
                    "lif_tau": float(os.environ.get("MNSIM_SNN_LIF_TAU", "50.5016666556")),
                    "lif_v_reset": float(os.environ.get("MNSIM_SNN_LIF_V_RESET", "0.0")),
                    "lif_alpha": float(os.environ.get("MNSIM_SNN_LIF_ALPHA", "2.0")),
                    "software_readout": readout},
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(f"best checkpoint: {result['best_checkpoint']}")
    print(f"best validation accuracy: {result['best_accuracy']:.4f}")
    print(f"final test accuracy: {result['test_accuracy']:.4f}")
    print(f"history: {result['history_path']}")


def _forward_with_readout(net, images, *, readout):
    return net.sequence_accuracy_forward(images, weighted_mode="ORIGINAL", readout=readout)


if __name__ == "__main__":
    main()
