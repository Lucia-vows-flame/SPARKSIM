#!/usr/bin/env python3
"""FIX_TRAIN/QAT fine-tuning for a V7 sequential SNN checkpoint."""
from __future__ import annotations

import argparse
import configparser
import math
import os
import random
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MNSIM.Interface.finetune_snn_v7 import finetune_net
from MNSIM.Interface.network import get_net
from MNSIM.Interface.snn_v7_training_data import training_loaders
from MNSIM.Interface.qat_quantization import load_qat_policy, normalize_qat_policy


def _hardware_config(path: Path, qat_policy=None) -> dict:
    config = configparser.ConfigParser()
    if not config.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"SimConfig not found: {path}")
    xbar_size = int(config.get("Crossbar level", "Xbar_Size").split(",")[0])
    device_level = int(config.get("Device level", "Device_Level"))
    hardware = {
        "xbar_size": xbar_size,
        "input_bit": int(config.get("Interface level", "DAC_Precision")),
        "weight_bit": math.floor(math.log2(device_level)),
        "weight_quantization_bit": 9,
        "ADC_quantize_bit": int(config.get("Interface level", "ADC_Precision")),
        "DAC_num": int(config.get("Process element level", "DAC_Num")),
        "xbar_polarity": int(config.get("Process element level", "Xbar_Polarity")),
    }
    if qat_policy is not None and qat_policy != normalize_qat_policy():
        hardware["qat_quantization"] = qat_policy
    return hardware


def _qat_policy_from_args(args):
    policy = load_qat_policy(args.qat_policy) if args.qat_policy else normalize_qat_policy()
    for argument, key in (("precision_mode", "precision_mode"), ("weight_bits", "weight_bit"),
                          ("activation_bits", "activation_bit"),
                          ("activation_quantization", "activation_quantization"),
                          ("activation_mu", "mu")):
        value = getattr(args, argument)
        if value is not None:
            policy[key] = value
    return normalize_qat_policy(policy)


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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, default=REPO_ROOT / "SimConfig.ini")
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "runs" / "snn_qat_finetune"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--milestones", type=int, nargs="+", default=[10, 15])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--gradient-clip-value", type=float)
    parser.add_argument("--qat-policy", type=Path,
                        help="versioned JSON policy with optional per-layer widths")
    parser.add_argument("--precision-mode", choices=["fixed", "mixed"],
                        help="QAT bit-width policy (default: fixed; not AMP)")
    parser.add_argument("--weight-bits", type=int,
                        help="signed default weight width (default: 9)")
    parser.add_argument("--activation-bits", type=int,
                        help="signed non-spike activation width (default: 9)")
    parser.add_argument("--activation-quantization", choices=["uniform", "mu_law"],
                        help="activation codebook (default: uniform)")
    parser.add_argument("--activation-mu", type=float,
                        help="positive mu-law companding coefficient (default: 15)")
    parser.add_argument("--lif-threshold", type=float)
    parser.add_argument("--lif-tau", type=float)
    parser.add_argument("--readout", choices=["spike", "membrane"], default="spike",
                        help="spike count or temporal classifier current mean")
    parser.add_argument("--freeze-bn", action="store_true",
                        help="freeze BatchNorm running stats; affine parameters remain trainable")
    parser.add_argument("--calibration-batches", type=int, default=0,
                        help="initialize QAT scales from finetune samples with frozen weights/BN")
    parser.add_argument("--evaluate-initial", action="store_true",
                        help="include the calibrated initial QAT checkpoint in validation selection")
    parser.add_argument("--target-validation-accuracy", type=float,
                        help="stop after min-epochs if current validation accuracy reaches this value")
    parser.add_argument("--min-epochs", type=int, default=1)
    parser.add_argument("--train-batch", type=int, default=8)
    parser.add_argument("--test-batch", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--seed", type=int, default=42)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        qat_policy = _qat_policy_from_args(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.calibration_batches < 0:
        parser.error("--calibration-batches must be nonnegative")

    if args.lif_threshold is not None:
        if not math.isfinite(args.lif_threshold) or args.lif_threshold <= 0:
            parser.error("--lif-threshold must be finite and positive")
        os.environ["MNSIM_SNN_LIF_V_TH"] = str(args.lif_threshold)
    if args.lif_tau is not None:
        if not math.isfinite(args.lif_tau) or args.lif_tau <= 1:
            parser.error("--lif-tau must be finite and greater than 1")
        os.environ["MNSIM_SNN_LIF_TAU"] = str(args.lif_tau)

    if not args.resume and not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    os.environ["MNSIM_DVS_BINNED_ROOT"] = str(args.dataset_root)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_loader, validation_loader, test_loader, identities = training_loaders(
        args.dataset_root, "finetune", args.train_batch, args.test_batch
    )
    net = get_net(
        hardware_config=_hardware_config(args.sim_config, qat_policy),
        cate=args.model,
        num_classes=8,
    )
    print(f"QAT quantization: precision={qat_policy['precision_mode']} "
          f"weight={qat_policy['weight_bit']} activation={qat_policy['activation_bit']} "
          f"grid={qat_policy['activation_quantization']} "
          f"layer_overrides={len(qat_policy['layer_overrides'])}", flush=True)
    if not args.resume:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        if args.calibration_batches:
            net.to(args.device).eval()
            net.set_activation_calibration(True)
            with torch.no_grad():
                for batch_index, (images, _) in enumerate(train_loader):
                    if batch_index >= args.calibration_batches:
                        break
                    net.sequence_qat_forward(images.to(args.device), readout=args.readout)
            net.set_activation_calibration(False)
            print(f"Initialized QAT scales using {min(args.calibration_batches, len(train_loader))} "
                  "finetune batches; weights and BatchNorm statistics frozen", flush=True)
    result = finetune_net(
        net,
        train_loader,
        validation_loader,
        args.device,
        args.model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        milestones=args.milestones,
        gamma=args.gamma,
        gradient_clip_value=args.gradient_clip_value,
        readout=args.readout,
        freeze_batchnorm=args.freeze_bn,
        evaluate_initial=args.evaluate_initial,
        target_validation_accuracy=args.target_validation_accuracy,
        min_epochs=args.min_epochs,
        output_dir=args.output_dir,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        test_loader=test_loader,
        run_config={"dataset_splits": identities, "seed": args.seed,
                    "train_batch": args.train_batch, "eval_batch": args.test_batch,
                    "sim_config": str(args.sim_config.resolve()), "stage": "finetune",
                    "lif_threshold": float(os.environ.get("MNSIM_SNN_LIF_V_TH", "0.01")),
                    "lif_tau": float(os.environ.get("MNSIM_SNN_LIF_TAU", "50.5016666556")),
                    "lif_v_reset": float(os.environ.get("MNSIM_SNN_LIF_V_RESET", "0.0")),
                    "lif_alpha": float(os.environ.get("MNSIM_SNN_LIF_ALPHA", "2.0")),
                    "software_readout": args.readout,
                    "calibration_batches": args.calibration_batches,
                    "source_checkpoint": str(args.checkpoint.resolve())},
        resume=args.resume,
        overwrite=args.overwrite,
    )
    print(f"best checkpoint: {result['best_checkpoint']}")
    print(f"best validation accuracy: {result['best_accuracy']:.4f}")
    print(f"final test accuracy: {result['test_accuracy']:.4f}")
    print(f"history: {result['history_path']}")


if __name__ == "__main__":
    main()
