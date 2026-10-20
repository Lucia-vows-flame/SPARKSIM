#!/usr/bin/env python3
"""Verify the software export without invoking a circuit simulator.

Replays the exported pulse sequences using both SpikingJelly and an independent
hard-reset LIF recurrence. Matrix/padding checks catch hardware orientation
errors that ordinary classification tests would not detect.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from train_mnist_two_layer_snn import TwoLayerSNN, deterministic_pulses


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def replay(pulses, w1, w2):
    v1 = torch.zeros(w1.shape[0], dtype=torch.float32)
    v2 = torch.zeros(w2.shape[0], dtype=torch.float32)
    trace1, trace2 = [], []
    for pulse in pulses.to(torch.float32):
        current1 = torch.mv(w1, pulse)
        h1 = v1 + (current1 - v1) / 2.0
        z1 = (h1 >= 1.0).to(torch.float32)
        v1 = h1 * (1.0 - z1)
        current2 = torch.mv(w2, z1)
        h2 = v2 + (current2 - v2) / 2.0
        z2 = (h2 >= 1.0).to(torch.float32)
        v2 = h2 * (1.0 - z2)
        trace1.append(z1)
        trace2.append(z2)
    return torch.stack(trace1), torch.stack(trace2)


def verify(directory: Path):
    config = json.loads((directory / "config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    require(config["lif"]["decay_input"] is True, "decay_input mismatch")
    require(config["lif"]["detach_reset"] is False, "detach_reset mismatch")
    require((config["lif"]["tau"], config["lif"]["v_threshold"], config["lif"]["v_reset"]) == (2.0, 1.0, 0.0),
            "fixed LIF parameters")
    require(config["lif"]["step_mode"] == "s" and config["pulse_encoding"]["steps"] == 30,
            "single-step LIF and time steps")
    require(config["quantization"]["signed_bits"] == 9 and config["training"]["seed"] == 42,
            "quantization bits and seed")
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest,
                f"checksum mismatch: {name}")
    for name, digest in config["files"].items():
        require(hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest,
                f"config checksum mismatch: {name}")

    split = np.load(directory / "split_indices.npz", allow_pickle=False)
    train, val = split["training_indices"], split["validation_indices"]
    require(len(train) == 55000 and len(val) == 5000, "split sizes")
    require(np.array_equal(np.sort(np.concatenate([train, val])), np.arange(60000)),
            "split must be disjoint and cover all training examples")
    expected = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).numpy()
    require(np.array_equal(val, expected[:5000]) and np.array_equal(train, expected[5000:]),
            "seed 42 split indices mismatch")

    weights = np.load(directory / "quantized_weights.npz", allow_pickle=False)
    weight_json = json.loads((directory / "quantized_weights.json").read_text())
    quant_checkpoint = torch.load(directory / "checkpoint_deployment_quantized.pth",
                                  map_location="cpu", weights_only=True)
    float_checkpoint = torch.load(directory / "checkpoint_best_float.pth",
                                  map_location="cpu", weights_only=True)
    for name, shape, state_key in [("W1", (8, 16), "fc1.weight"),
                                   ("W2", (10, 8), "fc2.weight")]:
        codes = weights[f"{name}_codes"]
        lsb = float(weights[f"{name}_lsb"])
        require(codes.shape == shape and codes.dtype == np.int16, f"{name} shape/dtype")
        require(np.all(np.abs(codes.astype(np.int32)) <= 255), f"{name} 9-bit range")
        require(lsb > 0 and np.isfinite(lsb), f"{name} invalid LSB")
        require(np.array_equal(codes, weight_json[f"{name}_codes"]), f"{name} JSON/NPZ mismatch")
        require(lsb == weight_json[f"{name}_lsb"], f"{name} JSON/NPZ LSB mismatch")
        reconstructed = torch.from_numpy(codes).float() * lsb
        require(torch.equal(reconstructed, quant_checkpoint["model_state_dict"][state_key]),
                f"{name} checkpoint reconstruction")
        float_weight = float_checkpoint["model_state_dict"][state_key]
        require(lsb == float(float_weight.abs().max()) / 255, f"{name} LSB selection")
        code_source = quant_checkpoint.get("quantization_source_state_dict",
                                           float_checkpoint["model_state_dict"])[state_key]
        require(torch.equal(torch.from_numpy(codes), torch.round(code_source / lsb).clamp(-255, 255).short()),
                f"{name} rounding rule")
        padded = np.zeros((16, 16), dtype=np.int16)
        padded[:shape[1], :shape[0]] = codes.T
        positive, negative = weights[f"{name}_positive_16x16"], weights[f"{name}_negative_16x16"]
        require(np.array_equal(weights[f"{name}_padded_16x16"], padded), f"{name} transpose/padding")
        require(np.array_equal(positive, np.maximum(padded, 0)), f"{name} positive array")
        require(np.array_equal(negative, np.maximum(-padded, 0)), f"{name} negative array")
        require(np.array_equal(positive - negative, padded), f"{name} differential reconstruction")

    for kind in ["float", "quantized"]:
        for split_name, count in [("validation", 5000), ("test", 10000)]:
            result = metrics[kind][split_name]
            confusion = np.asarray(result["confusion_matrix"])
            require(confusion.shape == (10, 10) and confusion.sum() == count,
                    f"{kind} {split_name} full evaluation")
            require(float(confusion.trace() / count) == result["accuracy"],
                    f"{kind} {split_name} accuracy/confusion mismatch")
            if split_name == "test":
                require(np.array_equal(confusion.sum(1), [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]),
                        "official MNIST test label counts")

    samples = np.load(directory / "test_samples.npz", allow_pickle=False)
    records = json.loads((directory / "test_samples.json").read_text())["samples"]
    require(len(records) >= 10, "at least ten fixed samples required")
    for kind, checkpoint in [("quantized", quant_checkpoint), ("float", float_checkpoint)]:
        state = checkpoint["model_state_dict"]
        require(set(state) == {"fc1.weight", "fc2.weight"}, "unexpected bias/extra parameters")
        model = TwoLayerSNN().eval()
        model.load_state_dict(state, strict=True)
        images = []
        expected_counts = []
        for record in records:
            prefix = f"sample_{record['test_index']:05d}"
            image = samples[f"{prefix}_input_4x4"]
            raw = samples[f"{prefix}_original"]
            independent_area = (raw.astype(np.float32) / np.float32(255)).reshape(4, 7, 4, 7).mean(axis=(1, 3))
            require(np.allclose(image, independent_area, rtol=0, atol=2e-7), "area average")
            pulses = torch.from_numpy(samples[f"{prefix}_input_pulses"])
            require(tuple(pulses.shape) == (30, 16), "pulse shape")
            require(bool(torch.all((pulses == 0) | (pulses == 1))), "binary input")
            require(torch.equal(pulses, deterministic_pulses(torch.from_numpy(image).flatten())),
                    "row-major deterministic pulse rule")
            require(torch.equal(pulses.sum(0), torch.floor(torch.from_numpy(image).flatten() * 30)),
                    "pulse telescoping count")
            hidden, output = replay(pulses, state["fc1.weight"], state["fc2.weight"])
            suffix = "_float" if kind == "float" else ""
            require(torch.equal(hidden, torch.from_numpy(samples[f"{prefix}{suffix}_lif1_spikes"])),
                    f"{kind} sample {prefix}: independent LIF1 trace")
            require(torch.equal(output, torch.from_numpy(samples[f"{prefix}{suffix}_lif2_spikes"])),
                    f"{kind} sample {prefix}: independent LIF2 trace")
            counts = output.sum(0)
            require(torch.equal(counts, torch.from_numpy(samples[f"{prefix}{suffix}_output_counts"])),
                    f"{kind} exported counts")
            prediction = int(counts.argmax())
            require(prediction == record["float_prediction" if kind == "float" else "prediction"],
                    f"{kind} exported prediction")
            detail = json.loads((directory / record["detail_file"]).read_text())
            require(detail[kind]["prediction"] == prediction and detail["label"] == record["label"],
                    f"{kind} JSON trace metadata")
            require(np.array_equal(np.asarray(Image.open(directory / record["original_png"])), raw),
                    "original PNG pixels")
            require(np.array_equal(detail["original_uint8"], raw), "original JSON pixels")
            require(np.array_equal(np.asarray(detail["input_4x4_float32"], dtype=np.float32), image),
                    "4x4 JSON/NPZ values")
            require(np.array_equal(detail["input_pulses_30x16"], pulses.numpy()), "input pulse JSON/NPZ")
            require(np.array_equal(detail[kind]["lif1_spikes_30x8"], hidden.numpy()), "LIF1 JSON/NPZ")
            require(np.array_equal(detail[kind]["lif2_spikes_30x10"], output.numpy()), "LIF2 JSON/NPZ")
            images.append(torch.from_numpy(image))
            expected_counts.append(counts)
        with torch.no_grad():
            batch = torch.stack(images)
            expected_counts = torch.stack(expected_counts)
            require(torch.equal(model(batch), expected_counts), f"{kind} batch-independent inference")
            require(torch.equal(model(batch), expected_counts), f"{kind} state reset")

    return {"passed": True, "sample_count": len(records),
            "checks": ["SHA256 manifests", "fixed seed disjoint data split",
                       "9-bit rounding and checkpoint reconstruction", "matrix transpose and zero padding",
                       "positive/negative differential arrays", "complete validation/test confusion counts",
                       "area preprocessing and row-major input", "deterministic binary pulse encoding",
                       "independent LIF recurrence for float and quantized weights",
                       "sample independence and state reset"],
            "software_only": True, "hspice_executed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(verify(args.directory), indent=2))
