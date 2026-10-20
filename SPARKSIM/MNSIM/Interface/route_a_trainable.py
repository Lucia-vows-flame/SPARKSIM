# -*- coding: utf-8 -*-
"""Route-A 硬件感知训练的辅助配置与诊断汇总。"""
from __future__ import annotations

from dataclasses import dataclass

from MNSIM.Hardware_Model.Neuron import LIFHardwareParams
from MNSIM.Interface.snn_hw_config import (
    RouteAConfig,
    normalize_frontend_i2v_mode,
    read_lif_params_overrides,
)


@dataclass(frozen=True)
class RouteATrainConfig:
    """训练态 Route-A 配置。

    说明：
    - 训练路径与 eval/report 路径共享物理默认值；
    - 训练路径只额外关心是否返回调试辅助信息。
    """

    simconfig_path: str
    v_read_v: float
    r_s_ohm: float
    frontend_i2v_mode: str
    align_lif_to_software: bool
    manual_v_th_override: bool
    linearize_device_levels: bool
    return_aux: bool = False

    @classmethod
    def from_simconfig(cls, simconfig_path: str, *, return_aux: bool = False) -> "RouteATrainConfig":
        base = RouteAConfig.from_simconfig(simconfig_path)
        return cls(
            simconfig_path=str(simconfig_path),
            v_read_v=float(base.v_read_v),
            r_s_ohm=float(base.r_s_ohm),
            frontend_i2v_mode=normalize_frontend_i2v_mode(base.frontend_i2v_mode),
            align_lif_to_software=bool(base.align_lif_to_software),
            manual_v_th_override=bool(base.manual_v_th_override),
            linearize_device_levels=bool(base.linearize_device_levels),
            return_aux=bool(return_aux),
        )


def merge_train_config(base: RouteATrainConfig, override: dict | None = None) -> RouteATrainConfig:
    """把运行时覆盖项叠加到默认训练配置。"""
    if not override:
        return base
    data = dict(base.__dict__)
    for key, value in override.items():
        if key not in data or value is None:
            continue
        if key in {"align_lif_to_software", "manual_v_th_override", "linearize_device_levels", "return_aux"}:
            data[key] = bool(value)
        elif key == "simconfig_path":
            data[key] = str(value)
        elif key == "frontend_i2v_mode":
            data[key] = normalize_frontend_i2v_mode(value)
        else:
            data[key] = float(value)
    return RouteATrainConfig(**data)


def build_train_lif_params(simconfig_path: str, *, r_s_ohm: float) -> LIFHardwareParams:
    """读取训练态 LIF 的基础物理参数。"""
    lif_overrides = dict(read_lif_params_overrides(simconfig_path))
    lif_overrides["r_s_ohm"] = float(r_s_ohm)
    valid_keys = set(LIFHardwareParams.__dataclass_fields__.keys())
    lif_overrides = {k: v for k, v in lif_overrides.items() if k in valid_keys}
    return LIFHardwareParams(**lif_overrides)


def build_route_a_aux_summary(layer_aux: list[dict]) -> dict:
    """把逐层调试信息整理成训练态可读摘要。"""
    current_means = []
    current_stds = []
    voltage_means = []
    voltage_stds = []
    spike_rates = []
    runtime_output_dequant_scale_means = []
    for item in layer_aux:
        if "current_mean" in item:
            current_means.append(float(item["current_mean"]))
        if "current_std" in item:
            current_stds.append(float(item["current_std"]))
        if "voltage_mean" in item:
            voltage_means.append(float(item["voltage_mean"]))
        if "voltage_std" in item:
            voltage_stds.append(float(item["voltage_std"]))
        if "spike_rate" in item:
            spike_rates.append(float(item["spike_rate"]))
        if "runtime_output_dequant_scale_mean" in item:
            runtime_output_dequant_scale_means.append(float(item["runtime_output_dequant_scale_mean"]))

    def _safe_mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "layers": layer_aux,
        "mean_current_mean": _safe_mean(current_means),
        "mean_current_std": _safe_mean(current_stds),
        "mean_voltage_mean": _safe_mean(voltage_means),
        "mean_voltage_std": _safe_mean(voltage_stds),
        "mean_spike_rate": _safe_mean(spike_rates),
        "mean_runtime_output_dequant_scale": _safe_mean(runtime_output_dequant_scale_means),
    }
