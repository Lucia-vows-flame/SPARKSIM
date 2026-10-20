# -*- coding: utf-8 -*-
"""从 `SimConfig.ini` 读取 route-A 与 LIF 硬件参数。

本文件统一管理配置解析逻辑，保证数值前向、硬件报告与 CLI 入口读取到同一套硬件默认值。
"""
from __future__ import annotations

import configparser
from dataclasses import dataclass
from typing import Any

_SUPPORTED_FRONTEND_I2V_MODES = {"r_s_voltage", "resistive_load"}
_DEFAULT_FRONTEND_I2V_MODE = "r_s_voltage"
_SUPPORTED_GAIN112_LUT_CHARACTERIZATION_PROFILES = {
    "canonical",
    "fast_1h",
    "fast_1h_experimental_7pct",
    "fast_1h_experimental_7pct_signed_vmem",
}
_SUPPORTED_GAIN112_LUT_NEGATIVE_VMEM_POLICIES = {"error", "clamp_zero"}
_DEFAULT_WEIGHT_QUANTIZATION_BIT = 9


def _read_simconfig(simconfig_path: str) -> configparser.ConfigParser:
    """读取 `SimConfig.ini`，并沿用 ConfigParser 的大小写无关查找。"""
    cfg = configparser.ConfigParser()
    cfg.read(simconfig_path, encoding="utf-8")
    return cfg


def _find_section(cfg: configparser.ConfigParser, candidates: list[str]) -> str | None:
    """在多个历史别名中返回第一个实际存在的节名。"""
    for name in candidates:
        if cfg.has_section(name):
            return name
    return None


def _get_float(cfg: configparser.ConfigParser, section: str, key: str) -> float | None:
    """读取浮点配置项；空字符串视为缺失。"""
    if not cfg.has_option(section, key):
        return None
    raw = cfg.get(section, key).strip()
    if raw == "":
        return None
    return float(raw)


def _get_int(cfg: configparser.ConfigParser, section: str, key: str) -> int | None:
    """Read an integer option; an empty value is treated as missing."""
    if not cfg.has_option(section, key):
        return None
    raw = cfg.get(section, key).strip()
    if raw == "":
        return None
    return int(raw)


def read_weight_quantization_bit(simconfig_path: str) -> int:
    """Read the signed weight quantizer width shared by accuracy and deployment."""

    cfg = _read_simconfig(simconfig_path)
    section = _find_section(
        cfg, ["Algorithm Configuration", "Algorithm configuration"]
    )
    if section is None:
        return _DEFAULT_WEIGHT_QUANTIZATION_BIT
    value = None
    key = "Weight_Quantization_Bit"
    for candidate in (
        "Weight_Quantization_Bit",
        "Weight_Quantization_Bits",
        "weight_quantization_bit",
    ):
        value = _get_int(cfg, section, candidate)
        if value is not None:
            key = candidate
            break
    if value is None:
        return _DEFAULT_WEIGHT_QUANTIZATION_BIT
    if value < 2:
        raise ValueError(
            f"{section}.{key} must be at least 2 for signed weight quantization, got {value}"
        )
    return int(value)


def normalize_frontend_i2v_mode(value: Any, *, source: str = "frontend_i2v_mode") -> str:
    """规范化前端 I->V 模式。

    当前主线支持两种模式：
    - `r_s_voltage`: `V_in = I_xbar * R_s`
    - `resistive_load`: `V_in = I_xbar / (1 / R_s + G_col_load)`
    """
    raw = str(value or _DEFAULT_FRONTEND_I2V_MODE).strip().lower()
    if raw in _SUPPORTED_FRONTEND_I2V_MODES:
        return raw
    raise ValueError(
        f"{source}={raw!r} is unsupported; only "
        f"{sorted(_SUPPORTED_FRONTEND_I2V_MODES)!r} are allowed."
    )


def normalize_gain112_lut_characterization_profile(
    value: Any,
    *,
    source: str = "gain112_lut_characterization_profile",
) -> str:
    profile = str(value or "canonical").strip().lower()
    if profile not in _SUPPORTED_GAIN112_LUT_CHARACTERIZATION_PROFILES:
        raise ValueError(
            f"{source}={profile!r} is unsupported; only "
            f"{sorted(_SUPPORTED_GAIN112_LUT_CHARACTERIZATION_PROFILES)!r} are allowed."
        )
    return profile


def normalize_gain112_lut_negative_vmem_policy(
    value: Any,
    *,
    source: str = "gain112_lut_negative_vmem_policy",
) -> str:
    policy = str(value or "error").strip().lower()
    if policy not in _SUPPORTED_GAIN112_LUT_NEGATIVE_VMEM_POLICIES:
        raise ValueError(
            f"{source}={policy!r} is unsupported; only "
            f"{sorted(_SUPPORTED_GAIN112_LUT_NEGATIVE_VMEM_POLICIES)!r} are allowed."
        )
    return policy


@dataclass(frozen=True)
class RouteAConfig:
    """route-A 顶层硬件参数，供 MVM 与 LIF 共享。

    说明：
    - 当前默认流程真正使用的是 `v_read_v`、`r_s_ohm`、`frontend_i2v_mode`、
      `frontend_i2v_latency_s`、`lif_cell_area_um2`、`align_lif_to_software`、
      `linearize_device_levels`。
    - gain-112 LUT 是 scheduled inference 的显式可选后端；默认关闭，
      以保留既有解析前端/LIF 路径。
    """

    v_read_v: float = 1.0
    r_s_ohm: float = 1e4
    frontend_i2v_mode: str = _DEFAULT_FRONTEND_I2V_MODE
    frontend_i2v_latency_s: float = 0.0
    lif_cell_area_um2: float = 127.0
    latency_step_window_s: float | None = 20e-9
    align_lif_to_software: bool = True
    manual_v_th_override: bool = False
    linearize_device_levels: bool = True
    enable_r_ratio: bool = False
    enable_variation: bool = False
    enable_saf: bool = False
    device_seed: int = 0
    gain112_lut_enabled: bool = False
    gain112_lut_artifact_root: str | None = None
    gain112_lut_verify_checksums: bool = True
    gain112_lut_characterization_profile: str = "canonical"
    gain112_lut_negative_vmem_policy: str = "error"

    @classmethod
    def from_simconfig(cls, simconfig_path: str) -> "RouteAConfig":
        """从 `SimConfig.ini` 读取 route-A 默认值。

        这里兼容多种历史节名/键名写法，保证旧配置仍可直接使用。
        """
        cfg = _read_simconfig(simconfig_path)
        sec = _find_section(cfg, ["SNN RouteA", "SNN routeA", "SNN Route A"])
        if sec is None:
            return cls()

        v_read = _get_float(cfg, sec, "v_read_v") or _get_float(cfg, sec, "v_read") or _get_float(cfg, sec, "V_read")
        r_s = _get_float(cfg, sec, "r_s_ohm") or _get_float(cfg, sec, "R_s_ohm") or _get_float(cfg, sec, "load_resistance_ohm")
        frontend_i2v_mode = cls.frontend_i2v_mode
        for key in ("frontend_i2v_mode", "Frontend_I2V_Mode"):
            if not cfg.has_option(sec, key):
                continue
            frontend_i2v_mode = normalize_frontend_i2v_mode(cfg.get(sec, key), source=f"{sec}.{key}")
            break
        frontend_i2v_latency_s = None
        for key, scale in (
            ("frontend_i2v_latency_s", 1.0),
            ("Frontend_I2V_Latency_s", 1.0),
            ("frontend_i2v_latency_ns", 1e-9),
            ("Frontend_I2V_Latency_ns", 1e-9),
        ):
            value = _get_float(cfg, sec, key)
            if value is None:
                continue
            frontend_i2v_latency_s = float(value) * float(scale)
            break
        lif_area = (
            _get_float(cfg, sec, "lif_cell_area_um2") or _get_float(cfg, sec, "LIF_cell_area_um2") or _get_float(cfg, sec, "lif_cell_area")
        )
        latency_step_window_s = None
        for key, scale in (
            ("latency_step_window_s", 1.0),
            ("Latency_Step_Window_s", 1.0),
            ("latency_step_window_ns", 1e-9),
            ("Latency_Step_Window_ns", 1e-9),
            ("schedule_step_window_ns", 1e-9),
            ("Schedule_Step_Window_ns", 1e-9),
        ):
            value = _get_float(cfg, sec, key)
            if value is None:
                continue
            latency_step_window_s = float(value) * float(scale)
            break
        align_lif = _get_bool(cfg, sec, "align_lif_to_software")
        if align_lif is None:
            align_lif = _get_bool(cfg, sec, "Align_LIF_To_Software")
        manual_v_th = _get_bool(cfg, sec, "manual_v_th_override")
        if manual_v_th is None:
            manual_v_th = _get_bool(cfg, sec, "Manual_Vth_Override")
        linearize_levels = _get_bool(cfg, sec, "linearize_device_levels")
        if linearize_levels is None:
            linearize_levels = _get_bool(cfg, sec, "Linearize_Device_Levels")
        enable_r_ratio = _get_bool(cfg, sec, "enable_r_ratio")
        if enable_r_ratio is None:
            enable_r_ratio = _get_bool(cfg, sec, "Enable_R_Ratio")
        enable_variation = _get_bool(cfg, sec, "enable_variation")
        if enable_variation is None:
            enable_variation = _get_bool(cfg, sec, "Enable_Variation")
        enable_saf = _get_bool(cfg, sec, "enable_saf")
        if enable_saf is None:
            enable_saf = _get_bool(cfg, sec, "Enable_SAF")
        device_seed = _get_int(cfg, sec, "device_seed")
        if device_seed is None:
            device_seed = _get_int(cfg, sec, "Device_Seed")
        gain112_lut_enabled = _get_bool(cfg, sec, "gain112_lut_enabled")
        if gain112_lut_enabled is None:
            gain112_lut_enabled = _get_bool(cfg, sec, "Gain112_LUT_Enable")
        gain112_lut_verify_checksums = _get_bool(cfg, sec, "gain112_lut_verify_checksums")
        if gain112_lut_verify_checksums is None:
            gain112_lut_verify_checksums = _get_bool(cfg, sec, "Gain112_LUT_Verify_Checksums")
        gain112_lut_artifact_root = None
        for key in ("gain112_lut_artifact_root", "Gain112_LUT_Artifact_Root"):
            if not cfg.has_option(sec, key):
                continue
            candidate = cfg.get(sec, key).strip()
            gain112_lut_artifact_root = candidate or None
            break
        gain112_lut_characterization_profile = cls.gain112_lut_characterization_profile
        for key in (
            "gain112_lut_characterization_profile",
            "Gain112_LUT_Characterization_Profile",
            "gain112_lut_profile",
            "Gain112_LUT_Profile",
        ):
            if not cfg.has_option(sec, key):
                continue
            gain112_lut_characterization_profile = normalize_gain112_lut_characterization_profile(
                cfg.get(sec, key),
                source=f"{sec}.{key}",
            )
            break
        gain112_lut_negative_vmem_policy = cls.gain112_lut_negative_vmem_policy
        for key in (
            "gain112_lut_negative_vmem_policy",
            "Gain112_LUT_Negative_Vmem_Policy",
        ):
            if not cfg.has_option(sec, key):
                continue
            gain112_lut_negative_vmem_policy = normalize_gain112_lut_negative_vmem_policy(
                cfg.get(sec, key),
                source=f"{sec}.{key}",
            )
            break

        return cls(
            v_read_v=float(v_read) if v_read is not None else cls.v_read_v,
            r_s_ohm=float(r_s) if r_s is not None else cls.r_s_ohm,
            frontend_i2v_mode=normalize_frontend_i2v_mode(frontend_i2v_mode),
            frontend_i2v_latency_s=float(frontend_i2v_latency_s) if frontend_i2v_latency_s is not None else cls.frontend_i2v_latency_s,
            lif_cell_area_um2=float(lif_area) if lif_area is not None else cls.lif_cell_area_um2,
            latency_step_window_s=float(latency_step_window_s) if latency_step_window_s is not None else cls.latency_step_window_s,
            align_lif_to_software=bool(align_lif) if align_lif is not None else cls.align_lif_to_software,
            manual_v_th_override=bool(manual_v_th) if manual_v_th is not None else cls.manual_v_th_override,
            linearize_device_levels=bool(linearize_levels) if linearize_levels is not None else cls.linearize_device_levels,
            enable_r_ratio=bool(enable_r_ratio) if enable_r_ratio is not None else cls.enable_r_ratio,
            enable_variation=bool(enable_variation) if enable_variation is not None else cls.enable_variation,
            enable_saf=bool(enable_saf) if enable_saf is not None else cls.enable_saf,
            device_seed=int(device_seed) if device_seed is not None else cls.device_seed,
            gain112_lut_enabled=bool(gain112_lut_enabled) if gain112_lut_enabled is not None else cls.gain112_lut_enabled,
            gain112_lut_artifact_root=gain112_lut_artifact_root,
            gain112_lut_verify_checksums=(
                bool(gain112_lut_verify_checksums)
                if gain112_lut_verify_checksums is not None
                else cls.gain112_lut_verify_checksums
            ),
            gain112_lut_characterization_profile=normalize_gain112_lut_characterization_profile(
                gain112_lut_characterization_profile
            ),
            gain112_lut_negative_vmem_policy=normalize_gain112_lut_negative_vmem_policy(
                gain112_lut_negative_vmem_policy
            ),
        )


@dataclass(frozen=True)
class SNNNoCConfig:
    """SNN NoC 配置。

    这组字段用于 Route-A 的 trace-driven NoC 估算，避免在多个入口里重复解析 `SimConfig.ini`。
    """

    enable: bool = False
    payload_mode: str = "mixed"
    spike_bitwidth: int = 1
    analog_bitwidth: int = 8
    window_mode: str = "replay_burst"
    include_same_tile: bool = False
    multicast_mode: str = "unicast"
    network_type: str = "mesh"
    arch_type: str = "serial"
    bus_width_bits: int = 32
    freq_hz: float = 1e9
    replay_overlap: float = 0.0
    energy_model: str = "bit_hop_dense_spike"
    energy_per_bit_hop_pj: float = 0.05

    @classmethod
    def from_simconfig(cls, simconfig_path: str) -> "SNNNoCConfig":
        """从 `SimConfig.ini` 读取 `[SNN NoC]`。

        解析策略：
        - 找不到配置节时直接返回默认值；
        - 无效枚举值回退到默认值；
        - 数值字段做下限保护，避免后续除零或负位宽。
        """
        cfg = _read_simconfig(simconfig_path)
        sec = _find_section(cfg, ["SNN NoC", "SNN NOC", "SNN_NoC"])
        if sec is None:
            return cls()

        def pick_str(keys: list[str], default: str, *, choices: tuple[str, ...] | None = None) -> str:
            value: str | None = None
            for key in keys:
                if cfg.has_option(sec, key):
                    raw = cfg.get(sec, key).strip()
                    if raw:
                        value = raw.lower()
                        break
            if value is None:
                return default
            if choices is not None and value not in choices:
                return default
            return value

        def pick_int(keys: list[str], default: int, *, lower_bound: int = 1) -> int:
            for key in keys:
                if not cfg.has_option(sec, key):
                    continue
                raw = cfg.get(sec, key).strip()
                if not raw:
                    continue
                try:
                    parsed = int(raw)
                except ValueError:
                    continue
                return max(parsed, lower_bound)
            return default

        enable = _get_bool(cfg, sec, "Enable")
        if enable is None:
            enable = _get_bool(cfg, sec, "enable")
        include_same_tile = _get_bool(cfg, sec, "Include_Same_Tile")
        if include_same_tile is None:
            include_same_tile = _get_bool(cfg, sec, "include_same_tile")

        payload_mode = pick_str(
            ["Payload_Mode", "payload_mode"],
            cls.payload_mode,
            choices=("mixed", "spike", "analog"),
        )
        window_mode = pick_str(
            ["Window_Mode", "window_mode"],
            cls.window_mode,
            choices=("replay_burst", "step_stream"),
        )
        multicast_mode = pick_str(
            ["Multicast_Mode", "multicast_mode"],
            cls.multicast_mode,
            choices=("unicast", "multicast"),
        )
        network_type = pick_str(
            ["Network_Type", "network_type"],
            cls.network_type,
            choices=("mesh", "htree"),
        )
        arch_type = pick_str(
            ["Arch_Type", "arch_type"],
            cls.arch_type,
            choices=("serial", "parallel"),
        )
        energy_model = pick_str(
            ["Energy_Model", "energy_model"],
            cls.energy_model,
            choices=("bit_hop_dense_spike", "booksim_full_network", "booksim_finite_burst"),
        )
        spike_bitwidth = pick_int(["Spike_Bitwidth", "spike_bitwidth"], cls.spike_bitwidth, lower_bound=1)
        analog_bitwidth = pick_int(["Analog_Bitwidth", "analog_bitwidth"], cls.analog_bitwidth, lower_bound=1)
        bus_width_bits = pick_int(["Bus_Width_Bits", "bus_width_bits"], cls.bus_width_bits, lower_bound=1)

        freq_hz = _get_float(cfg, sec, "Freq_Hz")
        if freq_hz is None:
            freq_hz = _get_float(cfg, sec, "freq_hz")
        if freq_hz is None or freq_hz <= 0:
            freq_hz = cls.freq_hz

        replay_overlap = _get_float(cfg, sec, "Replay_Overlap")
        if replay_overlap is None:
            replay_overlap = _get_float(cfg, sec, "replay_overlap")
        if replay_overlap is None:
            replay_overlap = cls.replay_overlap
        replay_overlap = min(max(float(replay_overlap), 0.0), 1.0)

        energy_per_bit_hop_pj = _get_float(cfg, sec, "Energy_Per_Bit_Hop_pJ")
        if energy_per_bit_hop_pj is None:
            energy_per_bit_hop_pj = _get_float(cfg, sec, "energy_per_bit_hop_pj")
        if energy_per_bit_hop_pj is None or energy_per_bit_hop_pj <= 0.0:
            energy_per_bit_hop_pj = cls.energy_per_bit_hop_pj

        return cls(
            enable=bool(enable) if enable is not None else cls.enable,
            payload_mode=payload_mode,
            spike_bitwidth=int(spike_bitwidth),
            analog_bitwidth=int(analog_bitwidth),
            window_mode=window_mode,
            include_same_tile=bool(include_same_tile) if include_same_tile is not None else cls.include_same_tile,
            multicast_mode=multicast_mode,
            network_type=network_type,
            arch_type=arch_type,
            bus_width_bits=int(bus_width_bits),
            freq_hz=float(freq_hz),
            replay_overlap=float(replay_overlap),
            energy_model=energy_model,
            energy_per_bit_hop_pj=float(energy_per_bit_hop_pj),
        )


@dataclass(frozen=True)
class SetweightsCorrection:
    """给旧版 `set_weights_forward` 使用的逐计算层仿射校正。

    旧的 bit-slice PIM 仿真会在稀疏 SNN 激活上同时引入幅度衰减与均值漂移。
    下游 BN+LIF 对这两种误差都敏感，因此校正必须支持如下仿射形式：

        `y_corr[l] = scale[l] * y_raw[l] + bias[l]`
    """

    scale_per_layer: tuple[float, ...] = ()
    bias_per_layer: tuple[float, ...] = ()

    def get(self, idx: int) -> tuple[float, float]:
        """返回单个计算层的 `(scale, bias)`，并提供安全默认值。"""
        scale = 1.0
        bias = 0.0
        if idx < len(self.scale_per_layer):
            scale = float(self.scale_per_layer[idx])
        if idx < len(self.bias_per_layer):
            bias = float(self.bias_per_layer[idx])
        return scale, bias


def _get_csv_floats(cfg: configparser.ConfigParser, section: str, key: str) -> list[float] | None:
    """读取逗号分隔的浮点列表；空字符串视为缺失。"""
    if not cfg.has_option(section, key):
        return None
    raw = cfg.get(section, key).strip()
    if raw == "":
        return None
    items: list[float] = []
    for part in raw.split(","):
        s = part.strip()
        if not s:
            continue
        items.append(float(s))
    return items if items else None


def _get_bool(cfg: configparser.ConfigParser, section: str, key: str) -> bool | None:
    """使用 ConfigParser 的标准真值表读取布尔项。"""
    if not cfg.has_option(section, key):
        return None
    raw = cfg.get(section, key).strip()
    if raw == "":
        return None
    return cfg.getboolean(section, key)


def read_lif_params_overrides(simconfig_path: str) -> dict[str, float]:
    """从 `SimConfig.ini` 读取 LIF 硬件参数，并换算为 SI 单位。

    返回值可直接作为 `LIFHardwareParams(**overrides)` 的关键字参数。
    """
    cfg = _read_simconfig(simconfig_path)
    sec = _find_section(cfg, ["SNN LIF Hardware", "LIF Hardware", "SNN LIF"])
    if sec is None:
        return {}

    out: dict[str, float] = {}

    def pick_scaled(*, dst_key: str, keys: list[tuple[str, float]]) -> None:
        """为某个物理量挑选第一个可用别名，并完成单位缩放。"""
        for k, scale in keys:
            v = _get_float(cfg, sec, k)
            if v is None:
                continue
            out[dst_key] = float(v) * float(scale)
            return

    pick_scaled(dst_key="t0_s", keys=[("t0_s", 1.0), ("t0_ns", 1e-9), ("T0_ns", 1e-9)])
    pick_scaled(dst_key="c_mem_f", keys=[("c_mem_f", 1.0), ("c_mem_fF", 1e-15), ("C_mem_fF", 1e-15)])
    pick_scaled(dst_key="r_mem_ohm", keys=[("r_mem_ohm", 1.0), ("R_mem_ohm", 1.0)])
    pick_scaled(dst_key="r_s_ohm", keys=[("r_s_ohm", 1.0), ("R_s_ohm", 1.0)])

    pick_scaled(dst_key="v_th_v", keys=[("v_th_v", 1.0), ("v_th", 1.0), ("V_th", 1.0)])
    pick_scaled(dst_key="v_reset_v", keys=[("v_reset_v", 1.0), ("v_reset", 1.0), ("V_reset", 1.0)])
    pick_scaled(dst_key="v_dd_v", keys=[("v_dd_v", 1.0), ("v_dd", 1.0), ("V_dd", 1.0)])

    pick_scaled(dst_key="mirror_ratio_m", keys=[("mirror_ratio_m", 1.0), ("mirror_ratio", 1.0), ("Mirror_ratio", 1.0)])
    pick_scaled(dst_key="comp_i_bias_a", keys=[("comp_i_bias_a", 1.0), ("comp_i_bias_uA", 1e-6), ("Comp_I_bias_uA", 1e-6)])
    pick_scaled(dst_key="comp_e_flip_j", keys=[("comp_e_flip_j", 1.0), ("comp_e_flip_fJ", 1e-15), ("Comp_E_flip_fJ", 1e-15)])
    pick_scaled(dst_key="ff_c_eff_f", keys=[("ff_c_eff_f", 1.0), ("ff_c_eff_fF", 1e-15), ("FF_C_eff_fF", 1e-15)])
    pick_scaled(dst_key="ff_e_sw_j", keys=[("ff_e_sw_j", 1.0), ("ff_e_sw_fJ", 1e-15), ("FF_E_sw_fJ", 1e-15)])
    pick_scaled(dst_key="reset_drv_e_j", keys=[("reset_drv_e_j", 1.0), ("reset_drv_e_fJ", 1e-15), ("ResetDrv_E_fJ", 1e-15)])

    return out


def read_setweights_correction(simconfig_path: str) -> SetweightsCorrection:
    """读取旧版 `set_weights_forward` 使用的逐层仿射校正。

    `scale_per_layer` 与 `bias_per_layer` 都是可选项；缺失时默认使用恒等校正。
    这样既兼容旧配置，也允许 SNN 路径同时补偿比例衰减与偏置漂移。
    """
    cfg = _read_simconfig(simconfig_path)
    sec = _find_section(cfg, ["SNN Setweights", "SNN setweights", "SNN SetWeights"])
    if sec is None:
        return SetweightsCorrection()
    scales = (
        _get_csv_floats(cfg, sec, "scale_correction_per_layer")
        or _get_csv_floats(cfg, sec, "Scale_Correction_Per_Layer")
        or _get_csv_floats(cfg, sec, "setweights_scale_correction_per_layer")
    )
    biases = (
        _get_csv_floats(cfg, sec, "bias_correction_per_layer")
        or _get_csv_floats(cfg, sec, "Bias_Correction_Per_Layer")
        or _get_csv_floats(cfg, sec, "setweights_bias_correction_per_layer")
    )
    return SetweightsCorrection(
        scale_per_layer=tuple(float(v) for v in (scales or ())),
        bias_per_layer=tuple(float(v) for v in (biases or ())),
    )


def read_setweights_scale_correction(simconfig_path: str) -> tuple[float, ...]:
    """给只需要 scale 项的旧调用方保留兼容访问接口。"""
    return read_setweights_correction(simconfig_path).scale_per_layer
