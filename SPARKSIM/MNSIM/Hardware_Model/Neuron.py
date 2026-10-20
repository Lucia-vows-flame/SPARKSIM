# -*- coding: utf-8 -*-
"""route-A 使用的显式 `R_s` 前端电压驱动 LIF 硬件状态机。

本模块对应我们讨论中的逐列 LIF bank：
- crossbar 输出电流 `I_xbar(t)` 先经前端 `R_s` 转成 `V_in(t)=I_xbar(t) * R_s`；
- 膜电位遵循一阶电压驱动 RC 动力学；
- 发放时采用步内 `t_fire` 求解；
- 硬件指标在同一次前向中在线累计。

实现上做了向量化，因此一个 `LIFHardwareBank` 就能表示某层某个特征张量上的全部神经元。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.autograd import Function


@dataclass(frozen=True)
class LIFHardwareParams:
    """电压驱动硬件 LIF 参数，单位均为 SI。"""

    t0_s: float = 1e-9
    c_mem_f: float = 1e-15
    r_mem_ohm: float = 1e6
    r_s_ohm: float = 1e4
    v_th_v: float = 0.5
    v_reset_v: float = 0.0
    v_dd_v: float = 1.0
    mirror_ratio_m: float = 1.0
    comp_i_bias_a: float = 1.5e-6
    comp_e_flip_j: float = 0.5e-15
    ff_c_eff_f: float = 0.5e-15
    ff_e_sw_j: float = 0.2e-15
    reset_drv_e_j: float = 0.0

    def validate(self) -> None:
        """对明显不合理的物理参数提前报错。"""
        if self.t0_s <= 0:
            raise ValueError("t0_s must be >0")
        if self.c_mem_f <= 0:
            raise ValueError("c_mem_f must be >0")
        if self.r_mem_ohm <= 0:
            raise ValueError("r_mem_ohm must be >0")
        if self.r_s_ohm <= 0:
            raise ValueError("r_s_ohm must be >0")
        if self.v_th_v <= 0:
            raise ValueError("v_th_v must be >0")
        if self.v_dd_v <= 0:
            raise ValueError("v_dd_v must be >0")
        if self.mirror_ratio_m < 0:
            raise ValueError("mirror_ratio_m must be >=0")
        if self.comp_i_bias_a < 0:
            raise ValueError("comp_i_bias_a must be >=0")
        if self.comp_e_flip_j < 0:
            raise ValueError("comp_e_flip_j must be >=0")
        if self.ff_c_eff_f < 0:
            raise ValueError("ff_c_eff_f must be >=0")
        if self.ff_e_sw_j < 0:
            raise ValueError("ff_e_sw_j must be >=0")
        if self.reset_drv_e_j < 0:
            raise ValueError("reset_drv_e_j must be >=0")


@dataclass
class LIFHardwareMetrics:
    """在 route-A 前向过程中在线累计的硬件指标。"""

    steps: int = 0
    spike_count: int = 0
    time_total_s: float = 0.0
    e_in_node_j: float = 0.0
    e_leak_j: float = 0.0
    e_rs_j: float = 0.0
    e_discard_j: float = 0.0
    e_resid_layer_j: float = 0.0
    e_mirror_j: float = 0.0
    e_comp_j: float = 0.0
    e_ff_j: float = 0.0
    e_reset_drv_j: float = 0.0

    def energy_total(self) -> float:
        """按当前能耗记账口径返回 LIF 总能耗。"""
        return (
            self.e_leak_j
            + self.e_rs_j
            + self.e_discard_j
            + self.e_resid_layer_j
            + self.e_mirror_j
            + self.e_comp_j
            + self.e_ff_j
            + self.e_reset_drv_j
        )

    def as_dict(self) -> dict:
        """把指标转换为报告层预期的字典格式。"""
        e_total = self.energy_total()
        p_avg = e_total / self.time_total_s if self.time_total_s > 0 else 0.0
        return {
            "steps": int(self.steps),
            "spike_count": int(self.spike_count),
            "time_total_s": float(self.time_total_s),
            "energy_total_j": float(e_total),
            "power_avg_w": float(p_avg),
            "e_in_node_j": float(self.e_in_node_j),
            "e_leak_j": float(self.e_leak_j),
            "e_rs_j": float(self.e_rs_j),
            "e_discard_j": float(self.e_discard_j),
            "e_resid_layer_j": float(self.e_resid_layer_j),
            "e_mirror_j": float(self.e_mirror_j),
            "e_comp_j": float(self.e_comp_j),
            "e_ff_j": float(self.e_ff_j),
            "e_reset_drv_j": float(self.e_reset_drv_j),
        }


def _rc_constants(params: LIFHardwareParams) -> tuple[float, float]:
    """返回膜网络使用的 `(R_mem, tau)`。"""
    tau = params.r_mem_ohm * params.c_mem_f
    return params.r_mem_ohm, tau


def _voltage_driven_advance_and_integrate(
    *,
    v_in: torch.Tensor,
    v0: torch.Tensor,
    dt_s: torch.Tensor,
    tau_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在单个常电压区间上，用解析形式推进 RC 状态并积分相关量。"""
    dt_s = dt_s.to(dtype=v_in.dtype, device=v_in.device)
    v_ss = v_in
    b = v0 - v_ss

    exp1 = torch.exp(-dt_s / tau_s)
    exp2 = torch.exp(-2.0 * dt_s / tau_s)
    v_end = v_ss + b * exp1

    int_v_dt = v_ss * dt_s + b * tau_s * (1.0 - exp1)
    int_v2_dt = (
        v_ss * v_ss * dt_s
        + 2.0 * v_ss * b * tau_s * (1.0 - exp1)
        + (b * b) * (tau_s / 2.0) * (1.0 - exp2)
    )
    int_vdiff_abs_dt = b.abs() * tau_s * (1.0 - exp1)
    int_vdiff2_dt = (b * b) * (tau_s / 2.0) * (1.0 - exp2)
    return v_end, int_v_dt, int_v2_dt, int_vdiff_abs_dt, int_vdiff2_dt


class LIFHardwareBank:
    """支持在线脉冲生成与指标累计的向量化 LIF bank。"""

    def __init__(self, params: LIFHardwareParams):
        params.validate()
        self.params = params
        self.v_mem: torch.Tensor | None = None
        self.metrics = LIFHardwareMetrics()

    def reset_state(self) -> None:
        """清除膜电位状态，但保留已经累计的指标。"""
        self.v_mem = None

    def reset_metrics(self) -> None:
        """清除指标，但保留当前膜电位状态。"""
        self.metrics = LIFHardwareMetrics()

    def reset(self) -> None:
        """同时清除状态与指标。"""
        self.reset_state()
        self.reset_metrics()

    def _ensure_state(self, v_in: torch.Tensor) -> None:
        """按当前输入张量的形状与设备分配膜电位存储。"""
        if self.v_mem is None or self.v_mem.shape != v_in.shape or self.v_mem.device != v_in.device or self.v_mem.dtype != v_in.dtype:
            self.v_mem = torch.zeros_like(v_in)

    def step(self, v_in: torch.Tensor) -> torch.Tensor:
        """执行一个物理步长 `T0`，并返回本步输出脉冲。"""
        if not torch.is_tensor(v_in):
            raise TypeError(f"v_in must be torch.Tensor, got {type(v_in)}")
        self._ensure_state(v_in)

        p = self.params
        t0 = float(p.t0_s)
        v_th = float(p.v_th_v)
        v_reset = float(p.v_reset_v)

        r_mem_ohm, tau_s = _rc_constants(p)
        v0 = self.v_mem
        v_ss = v_in

        # 求解步内跨越阈值的时间。
        immediate = v0 >= v_th
        possible = (v0 < v_th) & (v_ss > v_th)
        ratio = (v_th - v_ss) / (v0 - v_ss)
        ratio = torch.clamp(ratio, min=1e-12, max=1.0 - 1e-12)
        t_fire = (-tau_s) * torch.log(ratio)

        t_fire = torch.where(immediate, torch.zeros_like(t_fire), t_fire)
        fire = immediate | (possible & (t_fire <= t0))

        # 未发放神经元走整步推进路径。
        dt_full = torch.full_like(v_in, fill_value=t0)
        v_end_full, int_v_full, _int_v2_full, int_abs_vdiff_full, int_vdiff2_full = _voltage_driven_advance_and_integrate(
            v_in=v_in,
            v0=v0,
            dt_s=dt_full,
            tau_s=tau_s,
        )

        # 发放神经元走“发放前 + 复位后”两段路径。
        dt1 = torch.where(fire, t_fire, torch.zeros_like(t_fire))
        dt2 = torch.where(fire, dt_full - dt1, torch.zeros_like(dt_full))

        v_mid_1, int_v_1, _int_v2_1, int_abs_vdiff_1, int_vdiff2_1 = _voltage_driven_advance_and_integrate(
            v_in=v_in,
            v0=v0,
            dt_s=dt1,
            tau_s=tau_s,
        )
        v_reset_t = torch.full_like(v_in, fill_value=v_reset)
        v_end_2, int_v_2, _int_v2_2, int_abs_vdiff_2, int_vdiff2_2 = _voltage_driven_advance_and_integrate(
            v_in=v_in,
            v0=v_reset_t,
            dt_s=dt2,
            tau_s=tau_s,
        )

        v_end = torch.where(fire, v_end_2, v_end_full)
        int_v = torch.where(fire, int_v_1 + int_v_2, int_v_full)
        int_vdiff2 = torch.where(fire, int_vdiff2_1 + int_vdiff2_2, int_vdiff2_full)

        self.v_mem = v_end
        spike = fire.to(dtype=v_in.dtype)

        num_neurons = int(v_in.numel())
        spike_cnt = int(spike.sum().item())

        self.metrics.steps += 1
        self.metrics.spike_count += spike_cnt
        self.metrics.time_total_s += t0

        # RC 域能耗项：
        # - `e_rs`：前端电阻 `R_s` 上的耗散；
        # - `e_leak`：膜电阻 `R_mem` 上的耗散；
        # - `e_in_node`：电容充能诊断量，不计入总能耗。
        e_cap_full = 0.5 * p.c_mem_f * torch.clamp(v_end_full * v_end_full - v0 * v0, min=0.0)
        e_cap_1 = 0.5 * p.c_mem_f * torch.clamp(v_mid_1 * v_mid_1 - v0 * v0, min=0.0)
        e_cap_2 = 0.5 * p.c_mem_f * torch.clamp(v_end_2 * v_end_2 - v_reset_t * v_reset_t, min=0.0)
        e_cap_store = torch.where(fire, e_cap_1 + e_cap_2, e_cap_full)

        e_in_node = e_cap_store.sum().item()
        e_leak = (int_vdiff2 / r_mem_ohm).sum().item()
        e_rs = ((v_in * v_in) * (t0 / p.r_s_ohm)).sum().item()
        self.metrics.e_in_node_j += float(e_in_node)
        self.metrics.e_leak_j += float(e_leak)
        self.metrics.e_rs_j += float(e_rs)

        e_discard_event = 0.5 * p.c_mem_f * (p.v_th_v**2 - p.v_reset_v**2)
        self.metrics.e_discard_j += float(spike_cnt) * float(e_discard_event)

        # 外围电路能耗项复用同一份脉冲轨迹。
        int_abs_i_branch = torch.where(
            fire,
            (int_abs_vdiff_1 + int_abs_vdiff_2) / r_mem_ohm,
            int_abs_vdiff_full / r_mem_ohm,
        )
        e_mirror = (p.v_dd_v * (1.0 + p.mirror_ratio_m) * int_abs_i_branch).sum().item()
        self.metrics.e_mirror_j += float(e_mirror)

        e_comp_static = p.v_dd_v * p.comp_i_bias_a * t0 * num_neurons
        e_comp_dyn = p.comp_e_flip_j * spike_cnt
        self.metrics.e_comp_j += float(e_comp_static + e_comp_dyn)

        f_clk = 1.0 / t0
        e_ff_clk = p.ff_c_eff_f * (p.v_dd_v**2) * f_clk * t0 * num_neurons
        e_ff_dyn = p.ff_e_sw_j * spike_cnt
        self.metrics.e_ff_j += float(e_ff_clk + e_ff_dyn)

        self.metrics.e_reset_drv_j += float(p.reset_drv_e_j) * spike_cnt
        return spike

    def finish_layer(self) -> None:
        """结算层末残余电容能量，并为下一层清空膜状态。"""
        if self.v_mem is None:
            return
        e_resid = (0.5 * self.params.c_mem_f * (self.v_mem * self.v_mem)).sum().item()
        self.metrics.e_resid_layer_j += float(e_resid)
        self.v_mem = torch.zeros_like(self.v_mem)


class _ATanSpikeSurrogate(Function):
    """训练态 LIF 使用的 surrogate gradient。"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, alpha: float):
        ctx.save_for_backward(input)
        ctx.alpha = float(alpha)
        return (input >= 0).to(dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input,) = ctx.saved_tensors
        alpha = ctx.alpha
        scale = (torch.pi / 2.0) * alpha
        grad = (alpha / 2.0) / (1.0 + (scale * input).pow(2))
        return grad_output * grad, None


class TrainableRouteALIFNode(torch.nn.Module):
    """训练态 Route-A 硬件 LIF。

    - 前向使用硬件版电压驱动 RC + 步内 `t_fire`；
    - 发放判定使用硬件精确条件；
    - 反向仅对 spike 使用 surrogate gradient；
    - 不在训练图里累计能耗/延迟统计。
    """

    def __init__(self, params: LIFHardwareParams, *, alpha: float = 2.0):
        super().__init__()
        params.validate()
        self.params = params
        self.alpha = float(alpha)
        self.v_mem: torch.Tensor | None = None

    def reset_state(self) -> None:
        self.v_mem = None

    def _ensure_state(self, v_in: torch.Tensor) -> None:
        if self.v_mem is None or self.v_mem.shape != v_in.shape or self.v_mem.device != v_in.device or self.v_mem.dtype != v_in.dtype:
            self.v_mem = torch.zeros_like(v_in)

    def forward(self, v_in: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(v_in):
            raise TypeError(f"v_in must be torch.Tensor, got {type(v_in)}")
        self._ensure_state(v_in)

        p = self.params
        t0 = float(p.t0_s)
        v_th = float(p.v_th_v)
        v_reset = float(p.v_reset_v)
        _r_mem_ohm, tau_s = _rc_constants(p)

        v0 = self.v_mem
        dt_full = torch.full_like(v_in, fill_value=t0)
        v_end_full, _int_v_full, _int_v2_full, _int_abs_vdiff_full, _int_vdiff2_full = _voltage_driven_advance_and_integrate(
            v_in=v_in,
            v0=v0,
            dt_s=dt_full,
            tau_s=tau_s,
        )

        v_ss = v_in
        immediate = v0 >= v_th
        possible = (v0 < v_th) & (v_ss > v_th) & (v_end_full >= v_th)
        safe_num = torch.where(possible | immediate, torch.full_like(v_ss, fill_value=v_th) - v_ss, torch.ones_like(v_ss))
        safe_den = torch.where(possible | immediate, v0 - v_ss, torch.ones_like(v_ss))
        ratio = safe_num / safe_den
        ratio = torch.clamp(ratio, min=1e-12, max=1.0 - 1e-12)
        t_fire_candidate = (-tau_s) * torch.log(ratio)
        t_fire = torch.where(immediate, torch.zeros_like(t_fire_candidate), torch.where(possible, t_fire_candidate, dt_full))
        fire_hard = immediate | (possible & (t_fire <= t0))

        # 训练态不对步内 `t_fire` 的精确分段路径求导。
        # 原因：
        # 1) `t_fire = -tau * log(...)` 在阈值附近导数会非常尖锐；
        # 2) 我们已经对 spike 本身使用 surrogate gradient，继续对硬分段时序求导
        #    往往只会放大不稳定性，而不会显著改善收敛。
        #
        # 因此这里保留硬件前向的精确 `t_fire`，但把它从反向图中摘掉。
        t_fire_detached = t_fire.detach()
        dt1 = torch.where(fire_hard, t_fire_detached, torch.zeros_like(t_fire_detached))
        dt2 = torch.where(fire_hard, dt_full - dt1, torch.zeros_like(dt_full))
        v_reset_t = torch.full_like(v_in, fill_value=v_reset)
        v_end_2, _int_v_2, _int_v2_2, _int_abs_vdiff_2, _int_vdiff2_2 = _voltage_driven_advance_and_integrate(
            v_in=v_in,
            v0=v_reset_t,
            dt_s=dt2,
            tau_s=tau_s,
        )
        v_next_hard = torch.where(fire_hard, v_end_2, v_end_full)
        # 前向仍严格走硬件状态机；反向则退回到平滑的“未发放 RC 推进”主路径。
        self.v_mem = v_end_full + (v_next_hard - v_end_full).detach()

        spike_soft = _ATanSpikeSurrogate.apply(v_end_full - v_th, self.alpha)
        spike_hard = fire_hard.to(dtype=v_in.dtype)
        return spike_soft + (spike_hard - spike_soft).detach()
