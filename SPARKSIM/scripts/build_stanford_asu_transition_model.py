#!/usr/bin/env python3
"""Build a source-to-target write table from the Stanford/ASU sequence.

The sequential CSV supplies calibrated SET time increments.  The compact
model supplies the nonlinear programming current; integrating it over each
gap interval gives energy, while the recorded pulse time fixes the interval
time scale.  Downward transitions use the delivered 100 ns RESET protocol
followed by SET from the reset state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _integral(nodes: np.ndarray, weights: np.ndarray, lo: float, hi: float, fn) -> float:
    midpoint = (lo + hi) * 0.5
    half_width = (hi - lo) * 0.5
    values = fn(midpoint + half_width * nodes)
    return float(half_width * np.sum(weights * values))


def _speed(gap: np.ndarray, voltage: float, *, vt: float) -> np.ndarray:
    gamma = 16.0 - 0.8 * np.power(gap, 3.0)
    return 10.0 * np.exp(-0.6 / vt) * np.sinh(
        np.abs(voltage) * gamma * 0.25 / 12.0 / vt
    )


def _current(gap: np.ndarray, voltage: float) -> np.ndarray:
    return 1e-3 * np.exp(-gap / 0.25) * np.sinh(np.abs(voltage) / 0.25) + 1e-12 * np.abs(voltage)


def build(model_path: Path, levels_path: Path, output_path: Path) -> None:
    model = json.loads(model_path.read_text(encoding="utf-8"))
    with levels_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != int(model["device"]["levels"]):
        raise ValueError("level CSV length does not match model device levels")

    level_count = len(rows)
    gaps = np.asarray([float(row["gap_after"]) for row in rows], dtype=np.float64)
    pulse_ns = np.asarray([float(row["pulse_ns"]) for row in rows], dtype=np.float64)
    if not np.all(np.diff(gaps) < 0.0):
        raise ValueError("gap_after must strictly decrease with the code")
    if np.any(pulse_ns < 0.0):
        raise ValueError("pulse_ns cannot be negative")

    summary_path = levels_path.with_name(levels_path.stem + "_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reset_gap = float(summary["reset"]["gap"])
    reset_ns = float(summary["protocol"]["reset_pulse_ns"])
    set_voltage = float(summary["protocol"]["set_voltage_V"])
    reset_voltage = abs(float(summary["protocol"]["reset_voltage_V"]))
    vt = 0.025852
    nodes, weights = np.polynomial.legendre.leggauss(24)

    set_energy_nj = np.zeros(level_count, dtype=np.float64)
    interval_raw_ns = np.zeros(level_count, dtype=np.float64)
    for code in range(1, level_count):
        lo, hi = float(gaps[code]), float(gaps[code - 1])
        raw_ns = _integral(nodes, weights, lo, hi, lambda g: 1.0 / _speed(g, set_voltage, vt=vt))
        if not math.isfinite(raw_ns) or raw_ns <= 0.0:
            raise ValueError(f"invalid SET integral for code {code}: {raw_ns}")
        interval_raw_ns[code] = raw_ns
        scale = pulse_ns[code] / raw_ns
        set_energy_nj[code] = _integral(
            nodes,
            weights,
            lo,
            hi,
            lambda g: set_voltage * _current(g, set_voltage) * scale / _speed(g, set_voltage, vt=vt),
        )

    reset_energy_nj = np.zeros(level_count, dtype=np.float64)
    for code in range(level_count):
        if code == 0:
            continue
        lo, hi = float(gaps[code]), reset_gap
        raw_ns = _integral(nodes, weights, lo, hi, lambda g: 1.0 / _speed(g, reset_voltage, vt=vt))
        scale = reset_ns / raw_ns
        reset_energy_nj[code] = _integral(
            nodes,
            weights,
            lo,
            hi,
            lambda g: reset_voltage * _current(g, reset_voltage) * scale / _speed(g, reset_voltage, vt=vt),
        )

    set_time_ns = np.cumsum(pulse_ns)
    set_energy_total_nj = np.cumsum(set_energy_nj)
    latency_ns = np.zeros((level_count, level_count), dtype=np.float64)
    energy_nj = np.zeros((level_count, level_count), dtype=np.float64)
    for source in range(level_count):
        for target in range(level_count):
            if target > source:
                latency_ns[source, target] = set_time_ns[target] - set_time_ns[source]
                energy_nj[source, target] = set_energy_total_nj[target] - set_energy_total_nj[source]
            elif target < source:
                latency_ns[source, target] = reset_ns + set_time_ns[target]
                energy_nj[source, target] = reset_energy_nj[source] + set_energy_total_nj[target]

    transition = {
        "objective_unit": "physical_write_energy_j",
        "latency_s": (latency_ns * 1e-9).reshape(-1).tolist(),
        "energy_j": (energy_nj * 1e-9).reshape(-1).tolist(),
        "semantics": "direct_set_for_increasing_code; reset_100ns_then_set_for_decreasing_code",
        "time_calibration": "sequential_csv_pulse_width_ns",
        "energy_model": "rram_hspice_manual_cml_iv_integrated_over_gap",
        "reset_gap": reset_gap,
        "reset_pulse_width_ns": reset_ns,
        "set_voltage_v": set_voltage,
        "reset_voltage_abs_v": reset_voltage,
        "thermal_voltage_v": vt,
    }
    model["transition_cost"] = transition
    model["capabilities"]["hardware_write_accounting"] = True
    model["capabilities"]["transition_model"] = "calibrated_sequential_gap_integral"
    model["capabilities"]["transition_model_exact_within_compact_model"] = True
    model["source"]["transition_table"] = str(levels_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "levels": level_count,
        "latency_ns_min": float(np.min(latency_ns[latency_ns > 0])),
        "latency_ns_max": float(np.max(latency_ns)),
        "energy_nj_min": float(np.min(energy_nj[energy_nj > 0])),
        "energy_nj_max": float(np.max(energy_nj)),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--levels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.model, args.levels, args.output)


if __name__ == "__main__":
    main()
