#-*-coding:utf-8-*-
"""Runtime consumption of BN-folded bias as a constant-input MVM row."""
from __future__ import annotations

import collections
import copy
import math
import re
from typing import Mapping

import numpy as np
import torch


BIAS_CONSUMPTION_MODE = "runtime_constant_input_mvm_row"
SCALE_CONSUMPTION_MODE = "runtime_output_dequant"
RUNTIME_BIAS_ATTR = "_snn_runtime_constant_input_bias"
RUNTIME_WEIGHT_LSB_ATTR = "_snn_runtime_weight_lsb"
RUNTIME_OUTPUT_SCALE_ATTR = "_snn_runtime_output_dequant_scale"


def is_bn_fold_payload(obj) -> bool:
    return isinstance(obj, Mapping) and "state_dict" in obj and "bn_fold_bias" in obj


def payload_state_dict(obj):
    return obj["state_dict"] if is_bn_fold_payload(obj) else obj


def _validate_payload(payload: Mapping) -> None:
    metadata = dict(payload.get("bn_fold_metadata", {}) or {})
    if metadata.get("bias_consumption") != BIAS_CONSUMPTION_MODE:
        raise ValueError(
            "BN-fold payload must declare bias_consumption="
            f"{BIAS_CONSUMPTION_MODE!r}, got {metadata.get('bias_consumption')!r}"
        )
    if "bn_fold_weight_lsb" in payload and metadata.get("scale_consumption") != SCALE_CONSUMPTION_MODE:
        raise ValueError(
            "BN-fold payload with bn_fold_weight_lsb must declare scale_consumption="
            f"{SCALE_CONSUMPTION_MODE!r}, got {metadata.get('scale_consumption')!r}"
        )


def validate_scheduled_bn_fold_payload(payload: Mapping) -> None:
    """Validate that a folded deploy payload is usable by scheduled inference.

    Legacy payloads without per-channel LSBs remain loadable for compatibility,
    but scheduled inference cannot execute their folded bias rows faithfully.
    Catch that condition before planning physical rounds.
    """

    if not is_bn_fold_payload(payload):
        raise ValueError("scheduled inference requires a BN-fold deploy payload with state_dict and bn_fold_bias")
    _validate_payload(payload)

    bias_map = payload.get("bn_fold_bias", {}) or {}
    lsb_map = payload.get("bn_fold_weight_lsb", {}) or {}
    if not isinstance(bias_map, Mapping):
        raise ValueError("bn_fold_bias must be a mapping")
    if not isinstance(lsb_map, Mapping):
        raise ValueError("bn_fold_weight_lsb must be a mapping")

    missing_lsb = sorted(str(key) for key in set(bias_map) - set(lsb_map))
    if missing_lsb:
        raise ValueError(
            "scheduled BN-fold payload is missing bn_fold_weight_lsb for folded-bias layer(s): "
            f"{', '.join(missing_lsb)}; re-export it from the matching SWBN source checkpoint"
        )

    extra_lsb = sorted(str(key) for key in set(lsb_map) - set(bias_map))
    if extra_lsb:
        raise ValueError(
            "scheduled BN-fold payload has bn_fold_weight_lsb without matching bn_fold_bias for layer(s): "
            f"{', '.join(extra_lsb)}"
        )

    for key, bias_value in bias_map.items():
        bias = bias_value.detach().cpu() if torch.is_tensor(bias_value) else torch.as_tensor(bias_value)
        if bias.dim() != 1:
            raise ValueError(f"bn_fold_bias {key} must be 1-D, got {tuple(bias.shape)}")
        if not torch.isfinite(bias).all():
            raise ValueError(f"bn_fold_bias {key} contains non-finite values")
        _coerce_1d_layer_tensor(
            lsb_map[key],
            out_dim=int(bias.numel()),
            label=f"bn_fold_weight_lsb {key}",
            require_positive=True,
        )


def _payload_weighted_layer(net, key, *, label: str):
    match = re.match(r"^layer_list\.(\d+)$", str(key))
    if match is None:
        raise ValueError(f"unsupported {label} key: {key}")
    layer_index = int(match.group(1))
    if layer_index >= len(net.layer_list):
        raise ValueError(f"{label} key {key} exceeds layer count {len(net.layer_list)}")
    layer = net.layer_list[layer_index]
    if getattr(layer, "layer_config", {}).get("type") not in {"conv", "fc"}:
        raise ValueError(f"{label} key {key} does not target a weighted layer")
    return layer


def _collect_bn_fold_runtime_attachments(net, payload: Mapping):
    _validate_payload(payload)
    bias_attachments = []
    bias_map = payload.get("bn_fold_bias", {}) or {}
    for key, value in bias_map.items():
        layer = _payload_weighted_layer(net, key, label="bn_fold_bias")
        bias = value.detach().cpu().clone() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32)
        out_dim = int(layer.layer_config.get("out_channels", layer.layer_config.get("out_features")))
        if tuple(bias.shape) != (out_dim,):
            raise ValueError(f"bn_fold_bias {key} shape {tuple(bias.shape)} != expected {(out_dim,)}")
        bias_attachments.append((layer, bias))

    lsb_attachments = []
    lsb_map = payload.get("bn_fold_weight_lsb", {}) or {}
    for key, value in lsb_map.items():
        layer = _payload_weighted_layer(net, key, label="bn_fold_weight_lsb")
        lsb_attachments.append(
            (
                layer,
                _coerce_1d_layer_tensor(
                    value,
                    out_dim=_layer_out_dim(layer),
                    label=f"bn_fold_weight_lsb {key}",
                    require_positive=True,
                ),
            )
        )
    return bias_attachments, lsb_attachments


def validate_bn_fold_runtime_metadata_for_net(net, payload: Mapping) -> None:
    _collect_bn_fold_runtime_attachments(net, payload)


def attach_bn_fold_bias_to_net(net, payload: Mapping) -> None:
    bias_attachments, lsb_attachments = _collect_bn_fold_runtime_attachments(net, payload)

    for layer, bias in bias_attachments:
        setattr(layer, RUNTIME_BIAS_ATTR, bias)
    for layer, lsb in lsb_attachments:
        setattr(layer, RUNTIME_WEIGHT_LSB_ATTR, lsb)


def clear_bn_fold_bias_from_net(net) -> None:
    for layer in getattr(net, "layer_list", []):
        if getattr(layer, "layer_config", {}).get("type") not in {"conv", "fc"}:
            continue
        if hasattr(layer, RUNTIME_BIAS_ATTR):
            delattr(layer, RUNTIME_BIAS_ATTR)
        if hasattr(layer, RUNTIME_WEIGHT_LSB_ATTR):
            delattr(layer, RUNTIME_WEIGHT_LSB_ATTR)
        if hasattr(layer, RUNTIME_OUTPUT_SCALE_ATTR):
            delattr(layer, RUNTIME_OUTPUT_SCALE_ATTR)


def get_layer_runtime_bias(layer) -> torch.Tensor | None:
    bias = getattr(layer, RUNTIME_BIAS_ATTR, None)
    if bias is None:
        return None
    return bias.detach().cpu().clone() if torch.is_tensor(bias) else torch.as_tensor(bias, dtype=torch.float32)


def _layer_out_dim(layer) -> int:
    return int(layer.layer_config.get("out_channels", layer.layer_config.get("out_features")))


def _coerce_1d_layer_tensor(value, *, out_dim: int, label: str, require_positive: bool = False) -> torch.Tensor:
    tensor = value.detach().cpu().clone() if torch.is_tensor(value) else torch.as_tensor(value, dtype=torch.float32).clone()
    tensor = tensor.to(dtype=torch.float32)
    if tuple(tensor.shape) != (int(out_dim),):
        raise ValueError(f"{label} shape {tuple(tensor.shape)} != expected {(int(out_dim),)}")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} contains non-finite values")
    if require_positive and not torch.gt(tensor, 0).all():
        raise ValueError(f"{label} must contain positive values")
    return tensor


def get_layer_runtime_weight_lsb(layer) -> torch.Tensor | None:
    lsb = getattr(layer, RUNTIME_WEIGHT_LSB_ATTR, None)
    if lsb is None:
        return None
    return lsb.detach().cpu().clone() if torch.is_tensor(lsb) else torch.as_tensor(lsb, dtype=torch.float32)


def require_runtime_weight_lsb(layer) -> torch.Tensor:
    lsb = get_layer_runtime_weight_lsb(layer)
    if lsb is None:
        raise ValueError("missing BN-fold weight lsb for folded-bias deploy layer")
    out_dim = _layer_out_dim(layer)
    return _coerce_1d_layer_tensor(lsb, out_dim=out_dim, label="bn_fold_weight_lsb", require_positive=True)


def compute_linearized_code_voltage_gain(conductance, v_read, r_s_ohm) -> float:
    levels = int(conductance.levels)
    if levels <= 1:
        raise ValueError(f"conductance levels must be > 1, got {levels}")
    g = conductance.g_siemens
    g_t = g.detach().cpu() if torch.is_tensor(g) else torch.as_tensor(g, dtype=torch.float64)
    g_t = g_t.reshape(-1).to(dtype=torch.float64)
    if int(g_t.numel()) < levels:
        raise ValueError(f"conductance.g_siemens has {int(g_t.numel())} values but levels={levels}")
    delta_g = float(((g_t[levels - 1] - g_t[0]) / float(levels - 1)).item())
    if not math.isfinite(delta_g) or delta_g <= 0.0:
        raise ValueError("linearized conductance delta_g must be positive")
    gain = float(v_read) * float(r_s_ohm) * delta_g
    if not math.isfinite(gain) or gain <= 0.0:
        raise ValueError("linearized code voltage gain must be positive")
    return float(gain)


def attach_runtime_output_dequant_scale(layer, *, runtime_env):
    if get_layer_runtime_weight_lsb(layer) is None:
        clear_runtime_output_dequant_scale(layer)
        return None
    if not bool(runtime_env.get("linearize_device_levels", False)):
        raise ValueError("scale-aware BN-fold deploy requires Linearize_Device_Levels=1")
    gain = compute_linearized_code_voltage_gain(
        runtime_env["conductance"],
        runtime_env["v_read"],
        runtime_env["r_s_ohm"],
    )
    scale = require_runtime_weight_lsb(layer) / float(gain)
    setattr(layer, RUNTIME_OUTPUT_SCALE_ATTR, scale.detach().cpu().clone())
    return get_layer_runtime_output_dequant_scale(layer)


def clear_runtime_output_dequant_scale(layer) -> None:
    if hasattr(layer, RUNTIME_OUTPUT_SCALE_ATTR):
        delattr(layer, RUNTIME_OUTPUT_SCALE_ATTR)


def get_layer_runtime_output_dequant_scale(layer) -> torch.Tensor | None:
    scale = getattr(layer, RUNTIME_OUTPUT_SCALE_ATTR, None)
    if scale is None:
        return None
    if torch.is_tensor(scale):
        return scale.detach().cpu().clone()
    return torch.as_tensor(scale, dtype=torch.float32).detach().cpu().clone()


def apply_runtime_output_dequant_scale(layer, y: torch.Tensor) -> torch.Tensor:
    scale = get_layer_runtime_output_dequant_scale(layer)
    if scale is None:
        return y
    if not torch.is_tensor(y):
        raise TypeError(f"y must be torch.Tensor, got {type(y)}")
    if y.dim() == 4:
        view_shape = (1, -1, 1, 1)
        channel_count = int(y.shape[1])
    elif y.dim() == 2:
        view_shape = (1, -1)
        channel_count = int(y.shape[1])
    else:
        raise ValueError(f"unsupported runtime output shape for dequant scale: {tuple(y.shape)}")
    if int(scale.numel()) != channel_count:
        raise ValueError(
            "runtime output dequant scale channel mismatch: "
            f"scale={int(scale.numel())} output_channels={channel_count}"
        )
    return y * scale.to(device=y.device, dtype=y.dtype).view(*view_shape)


def _slice_count(*, weight_bit: int, device_bit: int, polarity: int) -> int:
    if polarity == 2:
        return int(math.ceil((int(weight_bit) - 1) / int(device_bit)))
    return int(math.ceil(int(weight_bit) / int(device_bit)))


def _signed_weight_threshold(layer) -> int:
    weight_bit = int(layer.quantize_config["weight_bit"])
    return int(2 ** (weight_bit - 1) - 1)


def _runtime_lsb_weight_digits(layer) -> list[torch.Tensor]:
    lsb = require_runtime_weight_lsb(layer)
    thres = _signed_weight_threshold(layer)
    digits = []
    for split_idx, sublayer in enumerate(layer.sublayer_list):
        weight = sublayer.weight.detach().cpu().to(dtype=torch.float32)
        if int(weight.shape[0]) != int(lsb.numel()):
            raise ValueError(
                "bn_fold_weight_lsb output dimension does not match sublayer weight: "
                f"split={split_idx} lsb={int(lsb.numel())} weight_out={int(weight.shape[0])}"
            )
        lsb_view = lsb.reshape((int(lsb.numel()),) + (1,) * (weight.dim() - 1))
        digit = torch.clamp(torch.round(weight / lsb_view), -thres, thres)
        digits.append(digit.to(dtype=torch.int64, device="cpu"))
    return digits


def build_bnfold_runtime_bit_weights(layer) -> collections.OrderedDict:
    weight_bit = int(layer.quantize_config["weight_bit"])
    device_bit = int(layer.hardware_config["weight_bit"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    slice_count = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)

    bit_weights = collections.OrderedDict()
    for split_idx, weight_digit in enumerate(_runtime_lsb_weight_digits(layer)):
        target_shape = tuple(weight_digit.shape)
        for slice_idx in range(slice_count):
            pos_col, neg_col = _slice_signed_digit(
                weight_digit,
                slice_idx=slice_idx,
                device_bit=device_bit,
                polarity=polarity,
            )
            if polarity == 2:
                bit_weights[f"split{split_idx}_weight{slice_idx}_positive"] = pos_col.reshape(target_shape)
                bit_weights[f"split{split_idx}_weight{slice_idx}_negative"] = neg_col.reshape(target_shape)
            else:
                bit_weights[f"split{split_idx}_weight{slice_idx}"] = pos_col.reshape(target_shape)
    return bit_weights


def _bias_digit(layer, bias: torch.Tensor) -> torch.Tensor:
    runtime_lsb = get_layer_runtime_weight_lsb(layer)
    if runtime_lsb is not None:
        lsb = require_runtime_weight_lsb(layer)
        bias_t = bias.detach().cpu().clone() if torch.is_tensor(bias) else torch.as_tensor(bias, dtype=torch.float32)
        bias_t = bias_t.to(dtype=torch.float32)
        if tuple(bias_t.shape) != (int(lsb.numel()),):
            raise ValueError(f"runtime folded bias shape {tuple(bias_t.shape)} != expected {(int(lsb.numel()),)}")
        thres = _signed_weight_threshold(layer)
        return torch.clamp(torch.round(bias_t / lsb), -thres, thres).to(dtype=torch.int64, device="cpu")

    weight_scale = float(layer.bit_scale_list[1, 1].item())
    if not math.isfinite(weight_scale) or weight_scale <= 0.0:
        raise ValueError("cannot quantize folded bias before a positive weight scale is available")
    thres = _signed_weight_threshold(layer)
    return torch.clamp(torch.round(bias.to(dtype=torch.float32) / weight_scale), -thres, thres)


def _slice_signed_digit(
    weight_digit: torch.Tensor,
    *,
    slice_idx: int,
    device_bit: int,
    polarity: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = int(2 ** (slice_idx * device_bit))
    step = int(2 ** device_bit)
    sign = torch.sign(weight_digit)
    abs_digit = torch.abs(weight_digit)
    tmp = torch.remainder(abs_digit, base * step) - torch.remainder(abs_digit, base)
    slice_digit = (tmp / base).to(dtype=torch.int64)
    if polarity == 2:
        pos = torch.where(sign > 0, slice_digit, torch.zeros_like(slice_digit))
        neg = torch.where(sign < 0, slice_digit, torch.zeros_like(slice_digit))
        return pos.cpu().numpy().reshape(-1, 1), neg.cpu().numpy().reshape(-1, 1)
    signed = slice_digit * torch.where(sign < 0, -torch.ones_like(slice_digit), torch.ones_like(slice_digit))
    return signed.cpu().numpy().reshape(-1, 1), np.zeros((int(weight_digit.numel()), 1), dtype=np.int64)


def _flatten_rows(array) -> np.ndarray:
    arr = np.asarray(array)
    return np.reshape(arr, (arr.shape[0], -1))


def _row_capacity(layer) -> int:
    return int(layer.hardware_config["xbar_size"])


def _original_mvm_rows(layer) -> int:
    if layer.layer_config["type"] == "fc":
        return int(layer.layer_config["in_features"])
    return int(layer.layer_config["in_channels"]) * int(layer.layer_config["kernel_size"]) ** 2


def row_split_has_runtime_constant_input(layer_info: Mapping, row_idx: int) -> bool:
    return bool(layer_info.get("runtime_constant_input_bias", False)) and int(
        layer_info.get("runtime_constant_input_row_split_index", -1)
    ) == int(row_idx)


def row_split_is_runtime_constant_input_only(layer_info: Mapping, row_idx: int) -> bool:
    """Return True when a runtime-only synthetic split stores only the bias row."""
    return row_split_has_runtime_constant_input(layer_info, row_idx) and bool(
        layer_info.get("runtime_constant_input_row_split_is_synthetic", False)
    )


def runtime_constant_input_source_row_split_index(layer_info: Mapping) -> int:
    return int(layer_info.get("runtime_constant_input_source_row_split_index", -1))


def split_runtime_constant_input_matrix(*, layer_type: str, sublayer, matrix_2d, layer_struct: Mapping, row_idx: int):
    if not bool(layer_struct.get("runtime_constant_input_bias", False)):
        return matrix_2d, None
    target_split = int(layer_struct.get("runtime_constant_input_row_split_index", -1))
    if int(row_idx) != target_split:
        return matrix_2d, None

    position = int(layer_struct.get("runtime_constant_input_row_position", -1))
    if position < 0:
        raise ValueError("runtime constant-input row position is missing")

    matrix_t = matrix_2d if torch.is_tensor(matrix_2d) else torch.as_tensor(matrix_2d, dtype=torch.float32)
    if matrix_t.dim() != 2:
        matrix_t = matrix_t.reshape(matrix_t.shape[0], -1)
    if int(matrix_t.shape[1]) <= position:
        raise ValueError(
            "runtime constant-input row position exceeds split width: "
            f"row_split_index={int(row_idx)} position={position} width={int(matrix_t.shape[1])}"
        )

    bias_column = matrix_t[:, position].contiguous()
    normal = torch.cat([matrix_t[:, :position], matrix_t[:, position + 1 :]], dim=1).contiguous()

    if not row_split_is_runtime_constant_input_only(layer_struct, int(row_idx)):
        expected_cols = int(sublayer.weight.reshape(sublayer.weight.shape[0], -1).shape[1])
        if int(normal.shape[1]) != expected_cols:
            raise ValueError(
                "runtime constant-input row removal produced unexpected normal width: "
                f"row_split_index={int(row_idx)} got={int(normal.shape[1])} expected={expected_cols}"
            )
    elif int(normal.shape[1]) != 0:
        raise ValueError(
            "synthetic runtime constant-input split must not retain normal columns: "
            f"row_split_index={int(row_idx)} normal_cols={int(normal.shape[1])}"
        )

    return normal, bias_column


def expand_layer_bit_weights_for_runtime_bias(layer, layer_weights: Mapping, layer_info: Mapping):
    """Append the BN-folded bias row used by runtime constant-input MVM execution.

    The extra row may either fit into the last real row split or require a synthetic
    runtime-only split appended after ``layer.sublayer_list``.
    """
    bias = get_layer_runtime_bias(layer)
    if bias is None:
        return layer_weights, layer_info

    expanded_weights = collections.OrderedDict((key, copy.deepcopy(value)) for key, value in layer_weights.items())
    expanded_info = collections.OrderedDict((key, copy.deepcopy(value)) for key, value in layer_info.items())
    row_split_num = int(expanded_info["row_split_num"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    device_bit = int(layer.hardware_config["weight_bit"])
    weight_bit = int(layer.quantize_config["weight_bit"])
    slice_count = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)
    capacity = _row_capacity(layer)

    last_split = row_split_num - 1
    sample_key = f"split{last_split}_weight0_positive" if polarity == 2 else f"split{last_split}_weight0"
    last_cols = int(_flatten_rows(expanded_weights[sample_key]).shape[1])
    if last_cols + 1 <= capacity:
        target_split = last_split
        expanded_info["row_split_num"] = row_split_num
    else:
        target_split = row_split_num
        expanded_info["row_split_num"] = row_split_num + 1

    # Runtime row splits do not always map 1:1 to layer.sublayer_list. Callers must
    # consult runtime_constant_input_row_split_is_synthetic /
    # row_split_is_runtime_constant_input_only(...) before assuming a split is backed
    # by a real sublayer. When the bias row fits in an existing split, callers must
    # drop runtime_constant_input_row_position before reshaping normal weights.
    expanded_info["runtime_constant_input_bias"] = True
    expanded_info["runtime_mvm_input_rows"] = _original_mvm_rows(layer) + 1
    expanded_info["runtime_constant_input_row_split_index"] = int(target_split)
    expanded_info["runtime_constant_input_row_position"] = int(last_cols if target_split == last_split else 0)
    expanded_info["runtime_constant_input_row_split_is_synthetic"] = bool(target_split >= len(layer.sublayer_list))
    expanded_info["runtime_constant_input_source_row_split_index"] = int(min(last_split, len(layer.sublayer_list) - 1))

    bias_digit = _bias_digit(layer, bias)
    for slice_idx in range(slice_count):
        pos_col, neg_col = _slice_signed_digit(
            bias_digit,
            slice_idx=slice_idx,
            device_bit=device_bit,
            polarity=polarity,
        )
        if polarity == 2:
            pos_key = f"split{target_split}_weight{slice_idx}_positive"
            neg_key = f"split{target_split}_weight{slice_idx}_negative"
            if target_split == row_split_num:
                expanded_weights[pos_key] = pos_col
                expanded_weights[neg_key] = neg_col
            else:
                expanded_weights[pos_key] = np.concatenate([_flatten_rows(expanded_weights[pos_key]), pos_col], axis=1)
                expanded_weights[neg_key] = np.concatenate([_flatten_rows(expanded_weights[neg_key]), neg_col], axis=1)
        else:
            key = f"split{target_split}_weight{slice_idx}"
            if target_split == row_split_num:
                expanded_weights[key] = pos_col
            else:
                expanded_weights[key] = np.concatenate([_flatten_rows(expanded_weights[key]), pos_col], axis=1)

    return expanded_weights, expanded_info


def get_runtime_expanded_weights_and_structure(net):
    raw_struct = net.get_structure()
    expanded_weights = []
    expanded_struct = []
    for layer, layer_info in zip(net.layer_list, raw_struct):
        if get_layer_runtime_weight_lsb(layer) is not None:
            layer_weights = build_bnfold_runtime_bit_weights(layer)
        else:
            layer_weights = layer.get_bit_weights()
        if layer_weights is None:
            expanded_weights.append(layer_weights)
            expanded_struct.append(layer_info)
            continue
        new_weights, new_info = expand_layer_bit_weights_for_runtime_bias(
            layer=layer,
            layer_weights=layer_weights,
            layer_info=layer_info,
        )
        expanded_weights.append(new_weights)
        expanded_struct.append(new_info)
    return expanded_weights, expanded_struct
