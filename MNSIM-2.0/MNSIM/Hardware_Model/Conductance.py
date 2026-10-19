# -*- coding: utf-8 -*-
"""量化 RRAM 编码到物理电导的映射工具。

本模块是 MNSIM 量化权重编码与 route-A 物理 `G(code)` 查表之间的最底层桥接。
这里先确立一个关键不变量：所有电流域 MVM 最终都遵循 `I = V * G(code)`，
且 `G = 1 / R`。
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass

import numpy as np
import torch


def _parse_csv_floats(raw: str) -> list[float]:
    """解析 `SimConfig.ini` 里的逗号分隔浮点列表。"""
    items = []
    for part in str(raw).split(","):
        s = part.strip()
        if not s:
            continue
        items.append(float(s))
    return items


@dataclass(frozen=True)
class ConductanceTable:
    """把离散器件编码映射为电导 `G = 1 / R`。

    说明
    ----
    - `SimConfig.ini` 中的 `Device_Level` 被视为离散编码总数。
    - 编码值会被当作 `[0, Device_Level)` 范围内的查表索引。
    - 对 `Xbar_Polarity = 2`，route-A 使用差分电流 `I = I+ - I-`，
      因此 `code == 0` 不要求映射到零电导。
    """

    r_ohm: torch.Tensor  # [L], physical resistance table in ohm
    g_siemens: torch.Tensor  # [L], conductance table in siemens

    @property
    def levels(self) -> int:
        return int(self.r_ohm.numel())

    @staticmethod
    def from_simconfig(simconfig_path: str) -> "ConductanceTable":
        """从 `SimConfig.ini` 读取器件电阻表，并转换为电导 `G`。

        这里会先严格校验表长与正值约束，因为一旦器件表异常，后续所有 route-A
        电流计算都会被静默污染。
        """
        cfg = configparser.ConfigParser()
        read_ok = cfg.read(simconfig_path, encoding="UTF-8")
        if not read_ok:
            raise FileNotFoundError(f"failed to read SimConfig: {simconfig_path}")
        if "Device level" not in cfg:
            raise KeyError("SimConfig.ini missing section [Device level]")

        level = int(cfg.get("Device level", "Device_Level"))
        if level <= 0:
            raise ValueError(f"Device_Level must be >0, got {level}")

        sort_flag = int(cfg.get("Device level", "Device_Resistance_Sort", fallback="1"))
        r_list = _parse_csv_floats(cfg.get("Device level", "Device_Resistance"))
        if len(r_list) != level:
            raise ValueError(
                f"Device_Resistance length mismatch: Device_Level={level} but got {len(r_list)} values"
            )
        r_np = np.asarray(r_list, dtype=np.float64)
        if sort_flag:
            # Route-A 把权重编码视为“有序器件级”：编码越大，电导越大（电阻越小）。
            # 一些实测表可能不是按电阻单调排列，这里可选按电阻重新排序。
            r_np = np.sort(r_np)[::-1].copy()
        if not np.all(np.isfinite(r_np)):
            raise ValueError("Device_Resistance contains non-finite values")
        if np.any(r_np <= 0):
            bad = r_np[r_np <= 0][:5]
            raise ValueError(f"Device_Resistance must be positive; got <=0 values (first few): {bad}")

        r = torch.from_numpy(r_np).to(dtype=torch.float64)
        g = (1.0 / r).to(dtype=torch.float64)
        return ConductanceTable(r_ohm=r, g_siemens=g)

    def map_codes(self, codes: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        """向量化执行编码到电导的映射。

        参数
        ----
        codes:
            器件编码张量。若输入为浮点张量，则要求每个值在四舍五入后都严格为整数。
        dtype:
            输出 dtype。若省略，则优先跟随输入浮点 dtype，否则默认 `float32`。
        """
        if not torch.is_tensor(codes):
            raise TypeError(f"codes must be a torch.Tensor, got {type(codes)}")
        if codes.numel() == 0:
            out_dtype = torch.float32 if dtype is None else dtype
            return torch.empty_like(codes, dtype=out_dtype)

        if codes.is_floating_point():
            if not torch.all(torch.isfinite(codes)):
                raise ValueError("codes contain non-finite values")
            rounded = torch.round(codes)
            if not torch.allclose(codes, rounded):
                max_abs = (codes - rounded).abs().max().item()
                raise ValueError(f"codes must be integer-valued; max |x-round(x)|={max_abs}")
            codes_i64 = rounded.to(dtype=torch.int64)
        else:
            codes_i64 = codes.to(dtype=torch.int64)

        min_code = int(codes_i64.min().item())
        max_code = int(codes_i64.max().item())
        if min_code < 0 or max_code >= self.levels:
            raise ValueError(
                f"code out of range: min={min_code} max={max_code} but levels={self.levels}"
            )

        table = self.g_siemens
        if table.device != codes_i64.device:
            table = table.to(device=codes_i64.device)

        out = table[codes_i64]
        if dtype is None:
            if codes.dtype in (torch.float16, torch.float32, torch.float64):
                dtype = codes.dtype
            else:
                dtype = torch.float32
        return out.to(dtype=dtype)

    def remap_to_linearized_levels(self, codes: torch.Tensor) -> torch.Tensor:
        """把逻辑 slice 编码重映射到最接近线性目标的物理器件级。

        背景
        ----
        软件量化器输出的是均匀间隔的逻辑 slice 数字 `m in [0, L-1]`，但实测
        `G(code)` 往往高度非线性。若直接把逻辑数字 `m` 编程到物理级 `m`，
        那么有效差分电导增量 `G(m) - G(0)` 也会是非线性的，这会在电流域缩放前
        就引入较大的软件/硬件失配。

        本函数保留真实器件表，但为每个逻辑数字 `m` 选择一个物理级 `k(m)`，
        使其差分电导增量最接近理想线性目标：

            `G(k(m)) - G(0) ~= m/(L-1) * (G(L-1) - G(0))`

        这对应的是“编程到最接近的可用物理级”，而不是纯软件重缩放。
        """
        if not torch.is_tensor(codes):
            raise TypeError(f"codes must be a torch.Tensor, got {type(codes)}")
        if codes.numel() == 0:
            return codes.to(dtype=torch.int64)

        if codes.is_floating_point():
            if not torch.all(torch.isfinite(codes)):
                raise ValueError("codes contain non-finite values")
            rounded = torch.round(codes)
            if not torch.allclose(codes, rounded):
                max_abs = (codes - rounded).abs().max().item()
                raise ValueError(f"codes must be integer-valued; max |x-round(x)|={max_abs}")
            codes_i64 = rounded.to(dtype=torch.int64)
        else:
            codes_i64 = codes.to(dtype=torch.int64)

        min_code = int(codes_i64.min().item())
        max_code = int(codes_i64.max().item())
        if min_code < 0 or max_code >= self.levels:
            raise ValueError(
                f"code out of range: min={min_code} max={max_code} but levels={self.levels}"
            )

        g = self.g_siemens.to(device=codes_i64.device, dtype=torch.float64)
        g0 = g[0]
        gmax = g[-1]
        if self.levels <= 1 or float(torch.abs(gmax - g0).item()) == 0.0:
            lut = torch.arange(self.levels, device=codes_i64.device, dtype=torch.int64)
            return lut[codes_i64]

        denom = float(self.levels - 1)
        logical = torch.arange(self.levels, device=codes_i64.device, dtype=torch.float64)
        target = g0 + (logical / denom) * (gmax - g0)
        dist = torch.abs(g.unsqueeze(0) - target.unsqueeze(1))
        lut = torch.argmin(dist, dim=1).to(dtype=torch.int64)
        lut[0] = 0
        lut[-1] = self.levels - 1
        return lut[codes_i64]


def map_codes_with_table(
    codes: torch.Tensor,
    *,
    g_table_siemens: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """供测试或轻量调用方使用的便捷封装。

    当调用方已经持有一维 `G` 表时，这里无需再从 `SimConfig.ini` 重建完整
    `ConductanceTable`。
    """
    if g_table_siemens.dim() != 1:
        raise ValueError(f"g_table_siemens must be 1-D, got shape {tuple(g_table_siemens.shape)}")
    table = ConductanceTable(r_ohm=torch.empty(0), g_siemens=g_table_siemens.to(dtype=torch.float64))
    return table.map_codes(codes, dtype=dtype)
