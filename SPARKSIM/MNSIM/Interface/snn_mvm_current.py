# -*- coding: utf-8 -*-
"""量化 conv/fc 层的 route-A 电流域 MVM。

本模块在数值上充当 route-A 中 crossbar 阵列的替身。它复用 PyTorch 的
`conv2d` / `linear` 完成累加，但把存储权重编码解释为物理电导，把输入脉冲
幅值解释为字线电压，因此输出张量代表的是物理电流代理量，而不是 ANN/PIM
路径里的 ADC 域激活。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from MNSIM.Hardware_Model.Conductance import ConductanceTable
from MNSIM.Interface.quantize import QuantizeLayer, conductance_lookup_ste, quantize_weight_ste


@dataclass(frozen=True)
class RouteAMVMConfig:
    """为后续 route-A 重构预留的占位配置容器。"""

    v_read: float
    r_s_ohm: float

    @staticmethod
    def from_env() -> "RouteAMVMConfig":
        raise NotImplementedError


def _slice_count(*, weight_bit: int, device_bit: int, polarity: int) -> int:
    """返回一个逻辑权重需要拆成多少个器件 slice。

    对差分极性，符号位由正负阵列承担，因此参与 bit-slice 的只有幅值位。
    """
    if weight_bit <= 0:
        raise ValueError(f"weight_bit must be >0, got {weight_bit}")
    if device_bit <= 0:
        raise ValueError(f"device_bit must be >0, got {device_bit}")
    if polarity == 2:
        return int(math.ceil((weight_bit - 1) / device_bit))
    return int(math.ceil(weight_bit / device_bit))


def _alpha_for_slice(slice_idx: int, *, device_bit: int) -> float:
    """返回某个 bit slice 的基数权重。

    v0 route-A 继续沿用项目现有 bit-slice 约定：第 `j` 个 slice 的权重为
    `2 ** (j * DeviceBit)`。
    """
    if slice_idx < 0:
        raise ValueError("slice_idx must be >=0")
    return float(2 ** (int(slice_idx) * int(device_bit)))


def _to_torch_code(x: np.ndarray | torch.Tensor, *, device, dtype) -> torch.Tensor:
    """把保存的 bit-weight 数组规范化为目标设备上的 torch 张量。"""
    if torch.is_tensor(x):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x))
    if t.device != device:
        t = t.to(device=device)
    if t.dtype != dtype:
        t = t.to(dtype=dtype)
    return t


def _layer_compute_device(layer: QuantizeLayer) -> torch.device:
    """返回当前量化层主权重所在设备。"""
    for sub in layer.sublayer_list:
        return sub.weight.device
    return torch.device("cpu")


def _reduce_column_load(g_tensor: torch.Tensor) -> torch.Tensor:
    """把单个 split/slice 的电导张量规约为按输出列排列的总导纳。"""
    if g_tensor.dim() < 2:
        raise ValueError(f"expected conductance tensor with dim>=2, got {tuple(g_tensor.shape)}")
    return g_tensor.reshape(g_tensor.shape[0], -1).sum(dim=1)


def mvm_current_from_bit_weights(
    *,
    layer: QuantizeLayer,
    x: torch.Tensor,
    bit_weights: Mapping[str, np.ndarray],
    conductance: ConductanceTable,
    v_read: float,
    r_s_ohm: float | None = None,
    linearize_device_levels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """计算单个 `QuantizeLayer` 的电流域 MVM。

    参数
    ----
    layer:
        描述 stride/padding 与 split 布局的 `QuantizeLayer` 实例。
    x:
        `[0, 1]` 范围内的输入幅值张量。
        - conv: `[N, C, H, W]`
        - fc: `[N, F]`
    bit_weights:
        `layer.get_bit_weights()` 的返回结果，即每个 split 的器件编码。
    conductance:
        从 `SimConfig.ini` 加载的编码到电导映射表。
    v_read:
        字线高电平。route-A 把有效输入解释为 `V_in = x * v_read`。
    r_s_ohm:
        可选的源端电阻。若给定，则额外返回 `V_src = I * R_s` 供报告/调试使用。

    返回
    ----
    i_out:
        汇总所有 split/slice 后的输出电流张量。
    v_src:
        可选的 `I * R_s` 张量。
    """
    if not isinstance(layer, QuantizeLayer):
        raise TypeError(f"layer must be QuantizeLayer, got {type(layer)}")
    if layer.layer_config["type"] not in ["conv", "fc"]:
        raise ValueError(f"unsupported layer type: {layer.layer_config.get('type')}")
    if not torch.is_tensor(x):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

    device_bit = int(layer.hardware_config["weight_bit"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    weight_bit = int(layer.quantize_config["weight_bit"])
    slice_cnt = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)

    # 项目原本就会把宽层拆成多个子层以匹配 xbar 行容量。
    # route-A 保持相同拆分方式，确保映射路径与物理代价路径一致。
    input_list = torch.split(x, int(layer.split_input), dim=1)
    assert len(input_list) == len(layer.sublayer_list)

    out = None
    for sub_idx, (x_sub, sub) in enumerate(zip(input_list, layer.sublayer_list)):
        if layer.layer_config["type"] == "conv":
            conv_fn = lambda w: F.conv2d(
                x_sub,
                w,
                bias=None,
                stride=sub.stride,
                padding=sub.padding,
                dilation=sub.dilation,
                groups=sub.groups,
            )
        else:
            conv_fn = lambda w: F.linear(x_sub, w, bias=None)

        sub_out = None
        for j in range(slice_cnt):
            # 正/负阵列共同实现有符号差分电导。
            if polarity == 2:
                code_p = bit_weights[f"split{sub_idx}_weight{j}_positive"]
                code_n = bit_weights[f"split{sub_idx}_weight{j}_negative"]
            else:
                code = bit_weights[f"split{sub_idx}_weight{j}"]
                code_p = code
                code_n = 0

            code_p_t = _to_torch_code(code_p, device=x.device, dtype=torch.float32)
            code_n_t = _to_torch_code(code_n, device=x.device, dtype=torch.float32)
            if linearize_device_levels:
                code_p_t = conductance.remap_to_linearized_levels(code_p_t).to(device=x.device, dtype=torch.float32)
                code_n_t = conductance.remap_to_linearized_levels(code_n_t).to(device=x.device, dtype=torch.float32)
            g_p = conductance.map_codes(code_p_t, dtype=torch.float32)
            g_n = conductance.map_codes(code_n_t, dtype=torch.float32)
            g_diff = g_p - g_n

            # 数值上仍复用 conv/linear，但此时权重张量已经是电导张量。
            # 因而在乘上最终 `v_read` 之前，输出量纲对应“每伏产生的电流”。
            i_slice = conv_fn(g_diff)
            if j != 0:
                i_slice = i_slice * _alpha_for_slice(j, device_bit=device_bit)
            sub_out = i_slice if sub_out is None else (sub_out + i_slice)

        out = sub_out if out is None else (out + sub_out)

    # 避免在 route-A 前向中额外分配 `out * v_read` 的大张量。
    # 对大尺寸 `[T*N,...]` 输入，原地缩放能显著降低峰值显存/内存占用。
    scale = float(v_read)
    if torch.is_grad_enabled() and out.requires_grad:
        i_out = out * scale
    else:
        out.mul_(scale)
        i_out = out
    v_src = None if r_s_ohm is None else (i_out * float(r_s_ohm))
    return i_out, v_src


def mvm_column_load_from_bit_weights(
    *,
    layer: QuantizeLayer,
    bit_weights: Mapping[str, np.ndarray],
    conductance: ConductanceTable,
    linearize_device_levels: bool = False,
) -> torch.Tensor:
    """统计每个输出列连接到前端的真实总导纳。

    说明：
    - 对差分阵列，负载语义取 `g+ + g-`，而不是读出时的差导纳 `g+ - g-`；
    - 对单极阵列，负载语义取实际接入列线的正导纳 `g_mag`。
    """
    if not isinstance(layer, QuantizeLayer):
        raise TypeError(f"layer must be QuantizeLayer, got {type(layer)}")
    if layer.layer_config["type"] not in ["conv", "fc"]:
        raise ValueError(f"unsupported layer type: {layer.layer_config.get('type')}")

    device_bit = int(layer.hardware_config["weight_bit"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    weight_bit = int(layer.quantize_config["weight_bit"])
    slice_cnt = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)
    device = _layer_compute_device(layer)

    g_col_load = None
    for sub_idx, _sub in enumerate(layer.sublayer_list):
        sub_load = None
        for j in range(slice_cnt):
            if polarity == 2:
                code_p = bit_weights[f"split{sub_idx}_weight{j}_positive"]
                code_n = bit_weights[f"split{sub_idx}_weight{j}_negative"]
                code_p_t = _to_torch_code(code_p, device=device, dtype=torch.float32)
                code_n_t = _to_torch_code(code_n, device=device, dtype=torch.float32)
                if linearize_device_levels:
                    code_p_t = conductance.remap_to_linearized_levels(code_p_t).to(device=device, dtype=torch.float32)
                    code_n_t = conductance.remap_to_linearized_levels(code_n_t).to(device=device, dtype=torch.float32)
                g_load_slice = conductance.map_codes(code_p_t, dtype=torch.float32) + conductance.map_codes(code_n_t, dtype=torch.float32)
            else:
                code = bit_weights[f"split{sub_idx}_weight{j}"]
                code_t = _to_torch_code(code, device=device, dtype=torch.float32)
                if linearize_device_levels:
                    code_t = conductance.remap_to_linearized_levels(code_t).to(device=device, dtype=torch.float32)
                g_load_slice = conductance.map_codes(code_t, dtype=torch.float32)

            reduced = _reduce_column_load(g_load_slice)
            sub_load = reduced if sub_load is None else (sub_load + reduced)
        g_col_load = sub_load if g_col_load is None else (g_col_load + sub_load)

    if g_col_load is None:
        raise RuntimeError("failed to build column load vector")
    return g_col_load


def _split_signed_digits(
    weight_digit: torch.Tensor,
    *,
    slice_idx: int,
    device_bit: int,
    polarity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把离散 signed digit 切成单个 bit-slice 的正/负编码。"""
    if slice_idx < 0:
        raise ValueError(f"slice_idx must be >=0, got {slice_idx}")
    base = int(2 ** (slice_idx * device_bit))
    step = int(2 ** device_bit)

    sign = torch.sign(weight_digit)
    abs_digit = torch.abs(weight_digit)
    tmp = torch.remainder(abs_digit, base * step) - torch.remainder(abs_digit, base)
    slice_digit = (tmp // base).to(dtype=torch.int64)

    if polarity == 2:
        code_p = torch.where(sign > 0, slice_digit, torch.zeros_like(slice_digit))
        code_n = torch.where(sign < 0, slice_digit, torch.zeros_like(slice_digit))
        return code_p, code_n

    code_mag = slice_digit
    code_sign = torch.where(sign < 0, -torch.ones_like(slice_digit), torch.ones_like(slice_digit))
    return code_mag, code_sign


def _slice_real_conductance(
    *,
    conductance: ConductanceTable,
    code_a: torch.Tensor,
    code_b: torch.Tensor,
    polarity: int,
    linearize_device_levels: bool,
) -> torch.Tensor:
    """把单个 slice 的编码映射为真实差分电导。"""
    code_a = code_a.to(dtype=torch.float32)
    code_b = code_b.to(dtype=torch.float32)
    if linearize_device_levels:
        code_a = conductance.remap_to_linearized_levels(code_a).to(device=code_a.device, dtype=torch.float32)
        code_b = conductance.remap_to_linearized_levels(code_b).to(device=code_b.device, dtype=torch.float32)

    if polarity == 2:
        g_a = conductance.map_codes(code_a, dtype=torch.float32)
        g_b = conductance.map_codes(code_b, dtype=torch.float32)
        return g_a - g_b

    g_mag = conductance.map_codes(code_a, dtype=torch.float32)
    return g_mag * code_b.to(dtype=torch.float32)


def _quantize_weight_and_optional_bias_ste(
    weight_fp: torch.Tensor,
    folded_bias: torch.Tensor | None,
    qbit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, float, float]:
    if folded_bias is None:
        weight_q, weight_digit, lsb = quantize_weight_ste(weight_fp, qbit)
        return weight_q, weight_digit, None, None, float(lsb), float(weight_fp.detach().abs().max().item())
    if folded_bias.dim() != 1:
        raise ValueError(f"folded_bias must be 1-D, got {tuple(folded_bias.shape)}")
    if not torch.isfinite(folded_bias).all():
        raise FloatingPointError("folded_bias contains non-finite values")
    thres = 2 ** (int(qbit) - 1) - 1
    if thres <= 0:
        raise ValueError(f"qbit must be >=2 for signed quantization, got {qbit}")
    max_abs_t = torch.stack([
        weight_fp.detach().abs().max(),
        folded_bias.detach().abs().max(),
    ]).max()
    eps = max(float(torch.finfo(weight_fp.dtype).eps), 1e-8)
    max_abs = max(float(max_abs_t.item()), eps)
    lsb = max_abs / float(thres)
    weight_digit = torch.clamp(torch.round(weight_fp / lsb), -thres, thres)
    bias_digit = torch.clamp(torch.round(folded_bias / lsb), -thres, thres)
    weight_q_raw = weight_digit * lsb
    bias_q_raw = bias_digit * lsb
    weight_q = weight_fp + (weight_q_raw - weight_fp).detach()
    bias_q = folded_bias + (bias_q_raw - folded_bias).detach()
    return (
        weight_q,
        weight_digit.to(dtype=torch.int64),
        bias_q,
        bias_digit.to(dtype=torch.int64),
        float(lsb),
        float(max_abs),
    )


def _quantize_weight_and_optional_bias_with_lsb_ste(
    weight_fp: torch.Tensor,
    folded_bias: torch.Tensor | None,
    qbit: int,
    per_output_lsb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Quantize folded weights with a fixed per-output deployed LSB vector."""
    if not torch.is_tensor(weight_fp):
        raise TypeError(f"weight_fp must be torch.Tensor, got {type(weight_fp)}")
    if weight_fp.dim() < 1:
        raise ValueError(f"weight_fp must have an output dimension, got {tuple(weight_fp.shape)}")
    thres = 2 ** (int(qbit) - 1) - 1
    if thres <= 0:
        raise ValueError(f"qbit must be >=2 for signed quantization, got {qbit}")
    if not torch.is_tensor(per_output_lsb):
        raise TypeError(f"per_output_lsb must be torch.Tensor, got {type(per_output_lsb)}")
    if per_output_lsb.dim() != 1:
        raise ValueError(f"per_output_lsb must be 1-D, got {tuple(per_output_lsb.shape)}")

    out_channels = int(weight_fp.shape[0])
    if int(per_output_lsb.numel()) != out_channels:
        raise ValueError(
            f"per_output_lsb length {int(per_output_lsb.numel())} != output channels {out_channels}"
        )
    lsb = per_output_lsb.detach().to(device=weight_fp.device, dtype=weight_fp.dtype)
    if not torch.isfinite(lsb).all():
        raise FloatingPointError("per_output_lsb contains non-finite values")
    if not torch.all(lsb > 0):
        raise ValueError("per_output_lsb must be positive")

    weight_lsb = lsb.view(out_channels, *([1] * (weight_fp.dim() - 1)))
    weight_digit = torch.clamp(torch.round(weight_fp / weight_lsb), -thres, thres)
    weight_q_raw = weight_digit * weight_lsb
    weight_q = weight_fp + (weight_q_raw - weight_fp).detach()

    bias_q = None
    bias_digit = None
    if folded_bias is not None:
        if folded_bias.dim() != 1:
            raise ValueError(f"folded_bias must be 1-D, got {tuple(folded_bias.shape)}")
        if int(folded_bias.numel()) != out_channels:
            raise ValueError(
                f"folded_bias length {int(folded_bias.numel())} != output channels {out_channels}"
            )
        if not torch.isfinite(folded_bias).all():
            raise FloatingPointError("folded_bias contains non-finite values")
        bias_lsb = lsb.to(device=folded_bias.device, dtype=folded_bias.dtype)
        bias_digit = torch.clamp(torch.round(folded_bias / bias_lsb), -thres, thres)
        bias_q_raw = bias_digit * bias_lsb
        bias_q = folded_bias + (bias_q_raw - folded_bias).detach()

    return (
        weight_q,
        weight_digit.to(dtype=torch.int64),
        bias_q,
        None if bias_digit is None else bias_digit.to(dtype=torch.int64),
    )


def mvm_current_trainable_from_weight(
    *,
    layer: QuantizeLayer,
    x: torch.Tensor,
    conductance: ConductanceTable,
    v_read: float,
    r_s_ohm: float | None = None,
    linearize_device_levels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float | int]]:
    """训练态 Route-A MVM：`weight_fp -> quantize -> G(code) -> current`。

    与 `mvm_current_from_bit_weights(...)` 的主要区别是：
    - 输入直接使用浮点主权重；
    - 前向仍使用真实 `G(code)` 查表；
    - 反向通过 STE 回到量化后的浮点权重。
    """
    if not isinstance(layer, QuantizeLayer):
        raise TypeError(f"layer must be QuantizeLayer, got {type(layer)}")
    if layer.layer_config["type"] not in ["conv", "fc"]:
        raise ValueError(f"unsupported layer type: {layer.layer_config.get('type')}")
    if not torch.is_tensor(x):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")

    device_bit = int(layer.hardware_config["weight_bit"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    weight_bit = int(layer.quantize_config["weight_bit"])
    slice_cnt = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)

    weight_fp = torch.cat([sub.weight for sub in layer.sublayer_list], dim=1)
    weight_q, weight_digit, _weight_lsb_scale = quantize_weight_ste(weight_fp, weight_bit)

    split_sizes = [int(sub.weight.shape[1]) for sub in layer.sublayer_list]
    q_splits = list(torch.split(weight_q, split_sizes, dim=1))
    digit_splits = list(torch.split(weight_digit, split_sizes, dim=1))
    input_list = torch.split(x, int(layer.split_input), dim=1)

    if len(input_list) != len(layer.sublayer_list):
        raise RuntimeError(
            f"split_input mismatch: got {len(input_list)} input splits but {len(layer.sublayer_list)} sublayers"
        )

    out = None
    for sub_idx, (x_sub, sub, q_sub, digit_sub) in enumerate(zip(input_list, layer.sublayer_list, q_splits, digit_splits)):
        if layer.layer_config["type"] == "conv":
            conv_fn = lambda w: F.conv2d(
                x_sub,
                w,
                bias=None,
                stride=sub.stride,
                padding=sub.padding,
                dilation=sub.dilation,
                groups=sub.groups,
            )
        else:
            conv_fn = lambda w: F.linear(x_sub, w, bias=None)

        g_real = None
        for j in range(slice_cnt):
            code_a, code_b = _split_signed_digits(
                digit_sub,
                slice_idx=j,
                device_bit=device_bit,
                polarity=polarity,
            )
            g_slice = _slice_real_conductance(
                conductance=conductance,
                code_a=code_a,
                code_b=code_b,
                polarity=polarity,
                linearize_device_levels=linearize_device_levels,
            )
            alpha = _alpha_for_slice(j, device_bit=device_bit)
            if alpha != 1.0:
                g_slice = g_slice * alpha
            g_real = g_slice if g_real is None else (g_real + g_slice)

        g_ste = conductance_lookup_ste(q_sub.to(dtype=torch.float32), g_real)
        sub_out = conv_fn(g_ste)
        out = sub_out if out is None else (out + sub_out)

    scale = float(v_read)
    if torch.is_grad_enabled() and out.requires_grad:
        i_out = out * scale
    else:
        out.mul_(scale)
        i_out = out
    v_src = None if r_s_ohm is None else (i_out * float(r_s_ohm))

    aux = {
        "weight_abs_max": float(weight_fp.detach().abs().max().item()),
        "current_mean": float(i_out.detach().mean().item()),
        "current_std": float(i_out.detach().std().item()) if i_out.numel() > 1 else 0.0,
        "slice_count": int(slice_cnt),
        "sub_layers": int(len(layer.sublayer_list)),
    }
    if v_src is not None:
        aux["voltage_mean"] = float(v_src.detach().mean().item())
        aux["voltage_std"] = float(v_src.detach().std().item()) if v_src.numel() > 1 else 0.0
    return i_out, v_src, aux


def mvm_current_trainable_from_explicit_weights(
    *,
    layer: QuantizeLayer,
    x: torch.Tensor,
    weight_splits: list[torch.Tensor] | tuple[torch.Tensor, ...],
    folded_bias: torch.Tensor | None,
    conductance: ConductanceTable,
    v_read: float,
    r_s_ohm: float | None = None,
    linearize_device_levels: bool = False,
    per_output_lsb: torch.Tensor | None = None,
    runtime_output_dequant_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, float | int]]:
    """Trainable Route-A MVM for explicit folded weights.

    `folded_bias` is treated as the constant-input column of the same logical
    extended weight matrix W''=[W', b'].
    """
    if not isinstance(layer, QuantizeLayer):
        raise TypeError(f"layer must be QuantizeLayer, got {type(layer)}")
    if layer.layer_config["type"] not in ["conv", "fc"]:
        raise ValueError(f"unsupported layer type: {layer.layer_config.get('type')}")
    if not torch.is_tensor(x):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)}")
    if len(weight_splits) != len(layer.sublayer_list):
        raise ValueError(
            f"weight_splits length {len(weight_splits)} != sublayer count {len(layer.sublayer_list)}"
        )

    device_bit = int(layer.hardware_config["weight_bit"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    weight_bit = int(layer.quantize_config["weight_bit"])
    slice_cnt = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)

    weight_fp = torch.cat([weight for weight in weight_splits], dim=1)
    max_terms = [weight_fp.detach().abs().max()]
    if folded_bias is not None:
        max_terms.append(folded_bias.detach().abs().max())
    max_abs = float(torch.stack(max_terms).max().item())
    if per_output_lsb is None:
        weight_q, weight_digit, bias_q, bias_digit, _lsb, max_abs = _quantize_weight_and_optional_bias_ste(
            weight_fp,
            folded_bias,
            weight_bit,
        )
    else:
        weight_q, weight_digit, bias_q, bias_digit = _quantize_weight_and_optional_bias_with_lsb_ste(
            weight_fp,
            folded_bias,
            weight_bit,
            per_output_lsb,
        )

    split_sizes = [int(weight.shape[1]) for weight in weight_splits]
    q_splits = list(torch.split(weight_q, split_sizes, dim=1))
    digit_splits = list(torch.split(weight_digit, split_sizes, dim=1))
    input_list = torch.split(x, int(layer.split_input), dim=1)
    if len(input_list) != len(layer.sublayer_list):
        raise RuntimeError(
            f"split_input mismatch: got {len(input_list)} input splits but {len(layer.sublayer_list)} sublayers"
        )

    out = None
    for sub_idx, (x_sub, sub, q_sub, digit_sub) in enumerate(zip(input_list, layer.sublayer_list, q_splits, digit_splits)):
        if layer.layer_config["type"] == "conv":
            conv_fn = lambda w, b=None: F.conv2d(
                x_sub,
                w,
                bias=b,
                stride=sub.stride,
                padding=sub.padding,
                dilation=sub.dilation,
                groups=sub.groups,
            )
        else:
            conv_fn = lambda w, b=None: F.linear(x_sub, w, bias=b)

        g_real = None
        bias_g_real = None
        for j in range(slice_cnt):
            code_a, code_b = _split_signed_digits(
                digit_sub,
                slice_idx=j,
                device_bit=device_bit,
                polarity=polarity,
            )
            g_slice = _slice_real_conductance(
                conductance=conductance,
                code_a=code_a,
                code_b=code_b,
                polarity=polarity,
                linearize_device_levels=linearize_device_levels,
            )
            alpha = _alpha_for_slice(j, device_bit=device_bit)
            if alpha != 1.0:
                g_slice = g_slice * alpha
            g_real = g_slice if g_real is None else (g_real + g_slice)

            if sub_idx == 0 and bias_digit is not None:
                bias_a, bias_b = _split_signed_digits(
                    bias_digit,
                    slice_idx=j,
                    device_bit=device_bit,
                    polarity=polarity,
                )
                bias_g_slice = _slice_real_conductance(
                    conductance=conductance,
                    code_a=bias_a,
                    code_b=bias_b,
                    polarity=polarity,
                    linearize_device_levels=linearize_device_levels,
                )
                if alpha != 1.0:
                    bias_g_slice = bias_g_slice * alpha
                bias_g_real = bias_g_slice if bias_g_real is None else (bias_g_real + bias_g_slice)

        g_ste = conductance_lookup_ste(q_sub.to(dtype=torch.float32), g_real)
        bias_ste = None
        if sub_idx == 0 and bias_q is not None and bias_g_real is not None:
            bias_ste = conductance_lookup_ste(bias_q.to(dtype=torch.float32), bias_g_real)
        sub_out = conv_fn(g_ste, bias_ste)
        out = sub_out if out is None else (out + sub_out)

    scale = float(v_read)
    if torch.is_grad_enabled() and out.requires_grad:
        i_out = out * scale
    else:
        out.mul_(scale)
        i_out = out
    v_src = None if r_s_ohm is None else (i_out * float(r_s_ohm))
    aux = {
        "weight_abs_max": float(max_abs),
        "current_mean": float(i_out.detach().mean().item()),
        "current_std": float(i_out.detach().std().item()) if i_out.numel() > 1 else 0.0,
        "slice_count": int(slice_cnt),
        "sub_layers": int(len(layer.sublayer_list)),
        "folded_bias": 1 if folded_bias is not None else 0,
    }
    if per_output_lsb is not None:
        aux["folded_weight_lsb_mean"] = float(per_output_lsb.detach().to(dtype=torch.float32).mean().item())
    if runtime_output_dequant_scale is not None:
        if not torch.is_tensor(runtime_output_dequant_scale):
            raise TypeError(
                "runtime_output_dequant_scale must be torch.Tensor, "
                f"got {type(runtime_output_dequant_scale)}"
            )
        aux["runtime_output_dequant_scale_mean"] = float(
            runtime_output_dequant_scale.detach().to(dtype=torch.float32).mean().item()
        )
    if v_src is not None:
        aux["voltage_mean"] = float(v_src.detach().mean().item())
        aux["voltage_std"] = float(v_src.detach().std().item()) if v_src.numel() > 1 else 0.0
    return i_out, v_src, aux


def mvm_column_load_trainable_from_weight(
    *,
    layer: QuantizeLayer,
    conductance: ConductanceTable,
    linearize_device_levels: bool = False,
) -> torch.Tensor:
    """训练态版本的列总导纳统计。"""
    if not isinstance(layer, QuantizeLayer):
        raise TypeError(f"layer must be QuantizeLayer, got {type(layer)}")
    if layer.layer_config["type"] not in ["conv", "fc"]:
        raise ValueError(f"unsupported layer type: {layer.layer_config.get('type')}")

    device_bit = int(layer.hardware_config["weight_bit"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    weight_bit = int(layer.quantize_config["weight_bit"])
    slice_cnt = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)

    weight_fp = torch.cat([sub.weight for sub in layer.sublayer_list], dim=1)
    _weight_q, weight_digit, _weight_lsb_scale = quantize_weight_ste(weight_fp, weight_bit)
    split_sizes = [int(sub.weight.shape[1]) for sub in layer.sublayer_list]
    digit_splits = list(torch.split(weight_digit, split_sizes, dim=1))

    g_col_load = None
    for digit_sub in digit_splits:
        sub_load = None
        for j in range(slice_cnt):
            code_a, code_b = _split_signed_digits(
                digit_sub,
                slice_idx=j,
                device_bit=device_bit,
                polarity=polarity,
            )
            if polarity == 2:
                code_p = code_a.to(dtype=torch.float32)
                code_n = code_b.to(dtype=torch.float32)
                if linearize_device_levels:
                    code_p = conductance.remap_to_linearized_levels(code_p).to(device=code_p.device, dtype=torch.float32)
                    code_n = conductance.remap_to_linearized_levels(code_n).to(device=code_n.device, dtype=torch.float32)
                g_load_slice = conductance.map_codes(code_p, dtype=torch.float32) + conductance.map_codes(code_n, dtype=torch.float32)
            else:
                code_mag = code_a.to(dtype=torch.float32)
                if linearize_device_levels:
                    code_mag = conductance.remap_to_linearized_levels(code_mag).to(device=code_mag.device, dtype=torch.float32)
                g_load_slice = conductance.map_codes(code_mag, dtype=torch.float32)

            reduced = _reduce_column_load(g_load_slice)
            sub_load = reduced if sub_load is None else (sub_load + reduced)
        g_col_load = sub_load if g_col_load is None else (g_col_load + sub_load)

    if g_col_load is None:
        raise RuntimeError("failed to build trainable column load vector")
    return g_col_load


def mvm_column_load_trainable_from_explicit_weights(
    *,
    layer: QuantizeLayer,
    weight_splits: list[torch.Tensor] | tuple[torch.Tensor, ...],
    folded_bias: torch.Tensor | None,
    conductance: ConductanceTable,
    linearize_device_levels: bool = False,
    per_output_lsb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Column-load statistic for explicit folded weights."""
    device_bit = int(layer.hardware_config["weight_bit"])
    polarity = int(layer.hardware_config["xbar_polarity"])
    weight_bit = int(layer.quantize_config["weight_bit"])
    slice_cnt = _slice_count(weight_bit=weight_bit, device_bit=device_bit, polarity=polarity)
    weight_fp = torch.cat([weight for weight in weight_splits], dim=1)
    if per_output_lsb is None:
        _weight_q, weight_digit, _bias_q, bias_digit, _lsb, _max_abs = _quantize_weight_and_optional_bias_ste(
            weight_fp,
            folded_bias,
            weight_bit,
        )
    else:
        _weight_q, weight_digit, _bias_q, bias_digit = _quantize_weight_and_optional_bias_with_lsb_ste(
            weight_fp,
            folded_bias,
            weight_bit,
            per_output_lsb,
        )
    split_sizes = [int(weight.shape[1]) for weight in weight_splits]
    digit_splits = list(torch.split(weight_digit, split_sizes, dim=1))

    g_col_load = None
    for sub_idx, digit_sub in enumerate(digit_splits):
        sub_load = None
        for j in range(slice_cnt):
            code_a, code_b = _split_signed_digits(
                digit_sub,
                slice_idx=j,
                device_bit=device_bit,
                polarity=polarity,
            )
            if polarity == 2:
                code_p = code_a.to(dtype=torch.float32)
                code_n = code_b.to(dtype=torch.float32)
                if linearize_device_levels:
                    code_p = conductance.remap_to_linearized_levels(code_p).to(device=code_p.device, dtype=torch.float32)
                    code_n = conductance.remap_to_linearized_levels(code_n).to(device=code_n.device, dtype=torch.float32)
                g_load_slice = conductance.map_codes(code_p, dtype=torch.float32) + conductance.map_codes(code_n, dtype=torch.float32)
            else:
                code = code_a.to(dtype=torch.float32)
                if linearize_device_levels:
                    code = conductance.remap_to_linearized_levels(code).to(device=code.device, dtype=torch.float32)
                g_load_slice = conductance.map_codes(code, dtype=torch.float32)
            reduced = _reduce_column_load(g_load_slice)
            sub_load = reduced if sub_load is None else (sub_load + reduced)

            if sub_idx == 0 and bias_digit is not None:
                bias_a, bias_b = _split_signed_digits(
                    bias_digit,
                    slice_idx=j,
                    device_bit=device_bit,
                    polarity=polarity,
                )
                if polarity == 2:
                    bp = bias_a.to(dtype=torch.float32)
                    bn = bias_b.to(dtype=torch.float32)
                    if linearize_device_levels:
                        bp = conductance.remap_to_linearized_levels(bp).to(device=bp.device, dtype=torch.float32)
                        bn = conductance.remap_to_linearized_levels(bn).to(device=bn.device, dtype=torch.float32)
                    bias_load = conductance.map_codes(bp, dtype=torch.float32) + conductance.map_codes(bn, dtype=torch.float32)
                else:
                    ba = bias_a.to(dtype=torch.float32)
                    if linearize_device_levels:
                        ba = conductance.remap_to_linearized_levels(ba).to(device=ba.device, dtype=torch.float32)
                    bias_load = conductance.map_codes(ba, dtype=torch.float32)
                sub_load = sub_load + bias_load.to(device=sub_load.device, dtype=sub_load.dtype)
        g_col_load = sub_load if g_col_load is None else (g_col_load + sub_load)
    if g_col_load is None:
        raise RuntimeError("failed to build trainable explicit column load vector")
    return g_col_load
