#!/usr/bin/env python3
"""Run a one-batch PIM numerical and ADC-saturation audit."""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path

import torch

from MNSIM.Accuracy_Model.pim_numerical_audit import audit_pim_numerics
from MNSIM.Interface.interface import TrainTestInterface


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--dataset-module", required=True)
    parser.add_argument("--samples", default=1, type=int)
    parser.add_argument("--adc-action", choices=("SCALE", "FIX", "NONE"), default="SCALE")
    parser.add_argument(
        "--calibration-mode",
        choices=("frozen_reference",),
        default="frozen_reference",
    )
    parser.add_argument(
        "--calibration-batches",
        default=0,
        type=int,
        help=(
            "Explicit eval-mode calibration batches before the audit. Use this "
            "for a checkpoint that predates post-add activation quantization."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.samples <= 0:
        _parser().error("samples must be a positive integer")
    if args.calibration_batches < 0:
        _parser().error("calibration-batches must be non-negative")

    try:
        context = TrainTestInterface(
            args.network,
            args.dataset_module,
            args.sim_config,
            args.weights,
            device=args.device,
            num_classes=args.num_classes,
        )
        test_loader = import_module(args.dataset_module).get_dataloader()[1]
        context.net.to(context.device)
        if args.calibration_batches:
            calibration_batches = []
            calibration_iterator = iter(test_loader)
            for _ in range(args.calibration_batches):
                calibration_images, _ = next(calibration_iterator)
                calibration_batches.append(calibration_images.to(context.device))
            context.net.calibrate_activation_scales(
                calibration_batches,
                reset=True,
            )
        images, labels = next(iter(test_loader))
        images = images[: args.samples].to(context.device)
        labels = labels[: args.samples]
        if not len(images):
            raise ValueError("dataset did not yield any audit samples")
        result = audit_pim_numerics(
            context.net,
            images,
            adc_action=args.adc_action,
            calibration_mode=args.calibration_mode,
        )
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as error:
        _parser().error(str(error))

    result.update(
        {
            "network": args.network,
            "weights": str(Path(args.weights)),
            "sim_config": str(Path(args.sim_config)),
            "dataset_module": args.dataset_module,
            "calibration_batches": args.calibration_batches,
            "sample_labels": labels.tolist(),
        }
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote PIM numerical audit: {output_path}")
    output = result["output"]
    print(
        "output relative L2: "
        f"{output['metrics']['relative_l2']:.6%}; "
        f"ADC saturation: {result['adc_summary']['saturation_fraction']:.6%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
