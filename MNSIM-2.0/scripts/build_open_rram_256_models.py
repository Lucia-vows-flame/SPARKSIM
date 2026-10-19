#!/usr/bin/env python3
"""Discretize pinned open device models at Vread=0.1 V, without SPICE.

This exports nominal frozen-state readback tables, not measured 8-bit cells
or a programming protocol. Write accounting stays unavailable until a
source-to-target programming model has been separately validated.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace


LEVEL_COUNT = 256
READ_VOLTAGE_V = 0.1
MODEL_SPECS = (
    ("rram_memtorch_stanford_pku_256", "memtorch", "Stanford_PKU.py", "Stanford_PKU"),
)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(source_dir: Path) -> dict:
    lock = json.loads((source_dir / "sources.lock.json").read_text())
    for source in lock["sources"].values():
        for name, entry in source["files"].items():
            if sha256(source_dir / name) != entry["sha256"]:
                raise ValueError(f"pinned source checksum mismatch: {name}")
    return lock


def memtorch_defaults(path: Path, class_name: str) -> dict:
    """Read constructor literals; do not import MemTorch or its dependencies."""
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = [arg.arg for arg in init.args.args][-len(init.args.defaults):]
    return dict(zip(names, map(ast.literal_eval, init.args.defaults)))


def reference_read_current(path: Path, class_name: str):
    """Execute only the pinned Stanford-PKU scalar current() method."""
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name == class_name)
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "current")
    module = ast.Module(body=[fn], type_ignores=[])
    scope = {"math": math, "np": SimpleNamespace(exp=math.exp, sinh=math.sinh)}
    exec(compile(module, str(path), "exec"), scope)
    return scope["current"]


def make_read_model(source_path: Path, class_name: str | None):
    """Return native parameters, state bounds, state inverse, and read I-V."""
    if class_name != "Stanford_PKU":
        raise ValueError(f"unsupported source model: {class_name}")
    params = memtorch_defaults(source_path, class_name)
    read = reference_read_current(source_path, class_name)
    scale = params["I_0"] * math.sinh(READ_VOLTAGE_V / params["V_0"]) / READ_VOLTAGE_V

    def inverse(g):
        return params["g_0"] * math.log(scale / g)

    def current(state, voltage):
        return read(SimpleNamespace(**{**params, "gap": state}), voltage)

    bounds = (inverse(1 / params["r_off"]), inverse(1 / params["r_on"]))
    return params, "gap_m", bounds, inverse, current


def build(repo_root: Path, output_dir: Path | None = None) -> dict:
    source_dir = repo_root / "device_models/final_rram/source/open_models"
    output_dir = output_dir or repo_root / "device_models/final_rram/models"
    lock = verify_sources(source_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "boundary": "256 nominal frozen-state readback levels; no programming/variation calibration",
        "external_hspice_energy_read": False,
        "external_spice_execution": False,
        "models": {},
    }
    for model_id, source_key, source_name, cls in MODEL_SPECS:
        source_path = source_dir / source_name
        params, state_name, bounds, inverse, current = make_read_model(source_path, cls)
        gmin, gmax = [current(state, READ_VOLTAGE_V) / READ_VOLTAGE_V for state in bounds]
        if not 0 < gmin < gmax:
            raise ValueError(f"invalid conductance range for {model_id}")
        target = [gmin + code * (gmax - gmin) / (LEVEL_COUNT - 1) for code in range(LEVEL_COUNT)]
        states = [inverse(g) for g in target]
        read_current = [current(state, READ_VOLTAGE_V) for state in states]
        conductance = [i / READ_VOLTAGE_V for i in read_current]
        resistance = [1 / g for g in conductance]
        max_relative_error = max(abs(actual / desired - 1) for actual, desired in zip(conductance, target))
        lower, upper = sorted(bounds)
        tol = (upper - lower) * 1e-12
        if not all(lower - tol <= state <= upper + tol for state in states):
            raise ValueError(f"state inverse exceeds source bounds for {model_id}")
        if (len(set(conductance)) != LEVEL_COUNT
                or not all(math.isfinite(g) and g > 0 for g in conductance)
                or not all(a < b for a, b in zip(conductance, conductance[1:]))
                or max_relative_error > 1e-12):
            raise ValueError(f"256-level readback audit failed for {model_id}")

        csv_path = output_dir / f"{model_id}_levels.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["code", state_name, "read_voltage_v", "read_current_a", "nominal_conductance_s", "nominal_conductance_uS", "nominal_resistance_ohm"])
            for code, state, i, g, r in zip(range(LEVEL_COUNT), states, read_current, conductance, resistance):
                writer.writerow([code, state, READ_VOLTAGE_V, i, g, g * 1e6, r])

        upstream = lock["sources"][source_key]
        limitations = [
            "256 bins discretize a continuous compact model; reliable physical 8-bit storage is not demonstrated",
            "readback evaluates frozen native state, with no read-disturb transient",
            "programming voltages, pulse widths, transition latency and energy are not calibrated",
            "device/cycle variation, read noise and retention are not calibrated; nominal variation is zero",
            "MNSIM uses equivalent conductance at 0.1 V; full nonlinear I-V is not executed at runtime",
        ]
        if source_key == "skywater":
            limitations.append("upstream technology specifications demonstrate 2 or 3 bits per cell, not 8 bits")
        model = {
            "schema_version": 2,
            "model_id": model_id,
            "device": {"bits": 8, "levels": LEVEL_COUNT, "initial_code": 0, "read_voltage_v": READ_VOLTAGE_V, "programming_voltage_abs_v": None, "pulse_width_ns": None},
            "levels": {"code": list(range(LEVEL_COUNT)), "nominal_conductance_s": conductance, "nominal_resistance_ohm": resistance, "read_current_a": read_current, state_name: states},
            "native_model": {"parameters": params, "state_variable": state_name, "state_at_min_max_conductance": list(bounds)},
            "nominal_fit": {"method": "uniform_read_conductance_with_native_state_inverse", "equation": "G(code)=Gmin+code*(Gmax-Gmin)/255; G=I(Vread,state)/Vread", "on_off_ratio": gmax / gmin},
            "capabilities": {"nominal_conductance": True, "continuous_state": True, "hardware_write_accounting": False, "iv_nonlinearity": "source_iv_evaluated_offline_at_fixed_read_voltage", "measured_256_level_storage": False, "programming_reachability_verified": False},
            "compatibility_sentinels": {"description": "Zero legacy write fields are unavailable-data sentinels. Full hardware write accounting must reject this nominal-only profile.", "programming_voltage_abs_v": 0, "pulse_width_ns": 0, "pulse_count_table": 0},
            "reset_model": {"status": "unavailable_nominal_levels_only"},
            "variation": {"device_variation_percent": 0.0, "enabled_by_default": False, "model": "multiplicative_gaussian_resistance", "estimator": "not_calibrated_nominal_only"},
            "saf": {"enabled_by_default": False, "device_saf_percent_hrs_lrs": [0, 0]},
            "source": {"file": "device_models/final_rram/source/open_models/sources.lock.json", "sha256": sha256(source_dir / "sources.lock.json"), "model_file": source_name, "source_sha256": sha256(source_path), "repository": upstream["repository"], "commit": upstream["commit"], "upstream_path": upstream["files"][source_name]["upstream_path"], "license": upstream["license"], "levels_file": csv_path.name, "levels_sha256": sha256(csv_path)},
            "conversion": {"version": "open_rram_256_nominal_v1", "generator": "scripts/build_open_rram_256_models.py", "protocol": "uniform_frozen_state_readback", "read_voltage_v": READ_VOLTAGE_V, "temporal_state_evolution_executed": False, "external_hspice_energy_read": False, "external_spice_execution": False},
            "limitations": limitations,
        }
        model_path = output_dir / f"{model_id}.json"
        model_path.write_bytes(json_bytes(model))
        report["models"][model_id] = {
            "levels": LEVEL_COUNT, "distinct_conductance_levels": len(set(conductance)),
            "strictly_increasing_conductance": True, "positive_finite_resistance": True,
            "native_state_within_bounds": True, "max_readback_relative_error": max_relative_error,
            "conductance_min_us": conductance[0] * 1e6, "conductance_max_us": conductance[-1] * 1e6,
            "resistance_min_ohm": resistance[-1], "resistance_max_ohm": resistance[0],
            "on_off_ratio": gmax / gmin, "source_checksum_verified": True,
            "hardware_write_accounting_supported": False,
            "model_sha256": sha256(model_path), "levels_sha256": sha256(csv_path),
        }
    (output_dir / "open_rram_256_validation.json").write_bytes(json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(build(Path(__file__).resolve().parents[1], args.output_dir), indent=2))


if __name__ == "__main__":
    main()
