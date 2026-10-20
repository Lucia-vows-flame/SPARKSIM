"""Versioned QAT policies: explicit layer widths and activation codebooks.

Mixed precision here means a user-specified bit-width allocation, not AMP.
Default policy exactly retains signed 9-bit uniform FIX_TRAIN quantization.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


SCHEMA = "mnsim_qat_quantization/v1"
POLICY_BUFFER = "qat_quantization_policy_utf8"


def _bit(value, name):
    if type(value) is not int or not 2 <= value <= 16:
        raise ValueError(f"{name} must be an integer from 2 to 16")
    return value


def normalize_qat_policy(raw=None):
    policy = {
        "schema_id": SCHEMA, "precision_mode": "fixed",
        "weight_bit": 9, "activation_bit": 9,
        "activation_quantization": "uniform", "mu": 15.0,
        "layer_overrides": {},
    }
    if raw is not None:
        if not isinstance(raw, dict) or set(raw) - set(policy):
            raise ValueError("QAT policy must be an object with supported fields")
        policy.update(raw)
    if policy["schema_id"] != SCHEMA or policy["precision_mode"] not in ("fixed", "mixed"):
        raise ValueError("unsupported QAT schema or precision_mode")
    for name in ("weight_bit", "activation_bit"):
        _bit(policy[name], name)
    _validate_activation(policy)
    overrides = policy["layer_overrides"]
    if not isinstance(overrides, dict):
        raise ValueError("layer_overrides must map final graph layer IDs to settings")
    normalized = {}
    for layer_id, settings in overrides.items():
        if not isinstance(layer_id, str) or not layer_id.startswith("L") or not layer_id[1:].isdigit():
            raise ValueError("override IDs must be final graph IDs such as L000")
        if not isinstance(settings, dict) or not settings or set(settings) - {"weight_bit", "activation_bit", "activation_quantization", "mu"}:
            raise ValueError(f"unsupported or empty override for {layer_id}")
        local = dict(settings)
        for name in ("weight_bit", "activation_bit"):
            if name in local:
                _bit(local[name], f"{layer_id}.{name}")
                if policy["precision_mode"] == "fixed" and local[name] != policy[name]:
                    raise ValueError("fixed precision forbids differing per-layer bit widths")
        _validate_activation({**policy, **local})
        normalized[layer_id] = local
    return {**policy, "layer_overrides": normalized}


def _validate_activation(settings):
    if settings["activation_quantization"] not in ("uniform", "mu_law"):
        raise ValueError("activation_quantization must be uniform or mu_law")
    mu = settings["mu"]
    if isinstance(mu, bool) or not isinstance(mu, (int, float)) or not math.isfinite(mu) or not 0 < mu <= 1e6:
        raise ValueError("mu must be finite and in (0, 1e6]")


def load_qat_policy(path):
    return normalize_qat_policy(json.loads(Path(path).read_text(encoding="utf-8")))


def encode_qat_policy(policy):
    text = json.dumps(normalize_qat_policy(policy), sort_keys=True, separators=(",", ":"))
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.uint8)


def decode_qat_policy(value):
    if not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1:
        raise ValueError("invalid checkpoint QAT policy buffer")
    return normalize_qat_policy(json.loads(bytes(value.cpu().tolist()).decode("utf-8")))


def resolve_layer_quantization(raw, layers, configs):
    """Resolve overrides after insertion of BN/LIF, without touching spikes."""
    policy = normalize_qat_policy(raw)
    known = {f"L{i:03d}" for i in range(len(layers))}
    if set(policy["layer_overrides"]) - known:
        raise ValueError("QAT override names an unknown final graph layer")
    resolved = []
    activation_types = {"conv", "fc", "bn", "element_sum"}
    for i, (layer, config) in enumerate(zip(layers, configs)):
        layer_id = f"L{i:03d}"
        local = policy["layer_overrides"].get(layer_id, {})
        kind = layer["type"]
        if "weight_bit" in local and kind not in {"conv", "fc"}:
            raise ValueError(f"{layer_id} has no quantized weights")
        if set(local) - {"weight_bit"} and kind not in activation_types:
            raise ValueError(f"{layer_id} has no activation quantizer; LIF spikes remain binary")
        settings = {**policy, **local}
        result = {**config, "weight_bit": settings["weight_bit"], "activation_bit": settings["activation_bit"]}
        if kind in activation_types:
            result["activation_quantization"] = settings["activation_quantization"]
            result["activation_mu"] = float(settings["mu"])
        resolved.append(result)
    return policy, resolved


def activation_range(input, last_value, training, *, minimum=None):
    """Existing FIX_TRAIN EMA range rule, shared by both activation grids."""
    if last_value is None:
        raise ValueError("activation quantization requires a scale buffer")
    if training:
        std = 0.0 if minimum is not None and input.numel() == 1 else torch.std(input).item()
        observed_range = 3 * std + torch.abs(torch.mean(input)).item()
        if minimum is not None:
            observed_range = max(observed_range, minimum)
        previous_range = last_value.item()
        calibrated_range = observed_range if previous_range <= 0 else 0.707 * previous_range + (1 - 0.707) * observed_range
        last_value.data[0] = calibrated_range
    scale = last_value.item()
    if not math.isfinite(scale) or scale <= 0:
        raise RuntimeError("activation scale is not calibrated; run QAT or explicit activation calibration before eval/PIM inference")
    return scale


def mu_law_activation_ste(input, bit, last_value, training, mu=15.0):
    """Symmetric companding grid with identity STE through dequantized values."""
    bit = _bit(bit, "activation_bit")
    _validate_activation({"activation_quantization": "mu_law", "mu": mu})
    # A previously silent SNN layer is a valid calibration observation. Give
    # its new non-uniform quantizer a positive range without changing legacy
    # uniform calibration or allowing uncalibrated eval to update statistics.
    scale = activation_range(input, last_value, training, minimum=float(torch.finfo(input.dtype).eps))
    threshold = 2 ** (bit - 1) - 1
    # Compute the codebook in FP32 or FP64 even if a caller uses FP16.
    dtype = torch.float64 if input.dtype == torch.float64 else torch.float32
    normalized = (input.detach().to(dtype) / scale).clamp(-1, 1)
    log_mu = math.log1p(mu)
    compressed = normalized.sign() * torch.log1p(mu * normalized.abs()) / log_mu
    code = torch.round(compressed * threshold).clamp(-threshold, threshold)
    level = code / threshold
    dequantized = (level.sign() * torch.expm1(level.abs() * log_mu) / mu * scale).to(input.dtype)
    return (input - input.detach()) + dequantized
