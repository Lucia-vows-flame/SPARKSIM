#!/usr/bin/env python3
"""Offline nominal programming tables from pinned compact-model dynamics.

The user-selectable constant cell voltage and ideal stop-at-target protocol are
simulation assumptions, not measured 256-level programming calibration. Native
read tables remain unchanged. No SPICE results or ASU transition costs are used.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
from scipy.integrate import solve_ivp

from build_open_rram_256_models import (
    MODEL_SPECS, json_bytes, make_read_model, sha256, verify_sources,
)

VERSION = "open_rram_ideal_target_write_v1"
DEFAULT_VOLTAGE = 2.0


def native_object(path, class_name, params):
    """Bind source methods without importing upstream packages or simulate()."""
    cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
    names = ({"current", "T_current", "dg_dt"} if class_name == "Stanford_PKU"
             else {"current", "r_pn", "s_pn", "dRdt", "resistance"})
    module = ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    scope = {"math": math, "np": SimpleNamespace(exp=np.exp, sinh=np.sinh, power=np.power,
              random=SimpleNamespace(normal=lambda **kwargs: 0.0))}
    exec(compile(module, str(path), "exec"), scope)
    obj = SimpleNamespace(**params)
    for name in names:
        setattr(obj, name, MethodType(scope[name], obj))
    return obj


def native_scalar_integrals(path, cls, params, states, voltage, order=64):
    obj = native_object(path, cls, params)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    output = {}
    for sign, direction in [(1, "set"), (-1, "reset")]:
        applied = sign * voltage
        time, energy = [0.0], [0.0]
        for first, last in zip(states[:-1], states[1:]):
            midpoint, half = (first + last) / 2, abs(last - first) / 2
            times, energies = [], []
            for coordinate in midpoint + half * nodes:
                if cls == "Stanford_PKU":
                    obj.gap = float(coordinate)
                    current = obj.current(applied)
                    speed = obj.dg_dt(applied, current)
                else:
                    obj.g = 1 / float(coordinate)
                    current = obj.current(applied)
                    speed = obj.dRdt(applied)
                    boundary = obj.r_pn(applied, *(params["r_p"] if sign > 0 else params["r_n"]))
                    if (sign > 0 and coordinate <= boundary) or (sign < 0 and coordinate >= boundary):
                        raise ValueError(f"{cls}: voltage does not reach all nominal states")
                # These models use decreasing native state for increasing G.
                if not math.isfinite(speed) or speed * sign >= 0:
                    raise ValueError(f"{cls}: zero/wrong-sign state speed at {coordinate}, V={applied}")
                times.append(1 / abs(speed))
                energies.append(abs(applied * current) / abs(speed))
            time.append(time[-1] + float(half * np.dot(weights, times)))
            energy.append(energy[-1] + float(half * np.dot(weights, energies)))
        output[direction] = (np.array(time), np.array(energy))
    return output


def scalar_tables(integrals):
    st, se = integrals["set"]
    rt, re = integrals["reset"]
    size = len(st)
    time, energy = np.zeros((size, size)), np.zeros((size, size))
    for source in range(size):
        time[source, source+1:] = st[source+1:] - st[source]
        energy[source, source+1:] = se[source+1:] - se[source]
        time[source, :source] = rt[source] + st[:source]
        energy[source, :source] = re[source] + se[:source]
    return time, energy


def sky_path(params, current, initial, final, targets, voltage, initial_temperature=None, rtol=2e-8):
    """Integrate native electrothermal ODE using filament position as coordinate.

    This avoids an arbitrary time horizon and endpoint/event interpolation in a
    highly stiff time trajectory. State is monotonic under the chosen voltage;
    dt/dx=1/velocity, dE/dx=|VI|/velocity and dT/dx=thermal_derivative/velocity.
    """
    span = final - initial
    if span == 0:
        raise ValueError("empty SkyWater programming path")
    base_temperature = params["Temperature_0"]
    temperature = base_temperature if initial_temperature is None else initial_temperature

    def rhs(progress, y):
        state = initial + span * progress
        i = current(state, voltage)
        power = abs(voltage * i)
        gamma = params["gamma_k0"] + params["gamma_k1"] * ((params["Tox"] - state) / 1e-9) ** 3
        kt = 1.380649e-23 * y[2] / 1.602176634e-19
        field = gamma * 0.25e-9 / params["Tox"] * voltage
        velocity = params["velocity_k1"] * (
            math.exp((-params["Eact_generation"] + field) / kt)
            - math.exp((-params["Eact_recombination"] - field) / kt))
        if not math.isfinite(velocity) or velocity * span <= 0:
            raise ValueError("SkyWater path is not monotonic/reachable")
        dt = span / velocity
        thermal = power / params["C_thermal"] - (y[2] - base_temperature) / params["tau_thermal"]
        return [dt * 1e9, power * dt * 1e12, thermal * dt]

    solution = solve_ivp(rhs, (0, 1), (0, 0, temperature), method="Radau", rtol=rtol,
                         atol=[1e-7, 1e-7, 1e-5], dense_output=True, max_step=0.02)
    if not solution.success:
        raise ValueError(f"SkyWater electrothermal integration failed: {solution.message}")
    progress = (np.asarray(targets) - initial) / span
    if np.any(progress < -1e-12) or np.any(progress > 1 + 1e-12):
        raise ValueError("SkyWater target lies outside programming path")
    values = solution.sol(np.clip(progress, 0, 1))
    if np.any(~np.isfinite(values)) or np.any(values[:2] < -1e-20):
        raise ValueError("invalid SkyWater path integral")
    return values[0] * 1e-9, values[1] * 1e-12, float(solution.y[2, -1]), float(np.max(solution.y[2]))


def sky_tables(params, current, states, voltage, rtol=2e-8, sources=None):
    size = len(states)
    time, energy = np.zeros((size, size)), np.zeros((size, size))
    peak_temperature = params["Temperature_0"]
    for source in (range(size) if sources is None else sources):
        if source < size - 1:
            t, e, _, peak = sky_path(params, current, states[source], states[-1], states[source+1:], voltage, rtol=rtol)
            time[source, source+1:], energy[source, source+1:] = t, e
            peak_temperature = max(peak_temperature, peak)
        if source:
            rt, re, reset_temperature, peak = sky_path(params, current, states[source], states[0], [states[0]], -voltage, rtol=rtol)
            # Carry the physical temperature into SET after RESET, with no
            # invented cooldown delay or ambient-temperature reset.
            if source > 1:
                t, e, _, set_peak = sky_path(params, current, states[0], states[source-1], states[1:source], voltage,
                                            initial_temperature=reset_temperature, rtol=rtol)
                time[source, 1:source], energy[source, 1:source] = rt[0] + t, re[0] + e
                peak = max(peak, set_peak)
            time[source, 0], energy[source, 0] = rt[0], re[0]
            peak_temperature = max(peak_temperature, peak)
        if sources is None and source % 32 == 0:
            print(f"[sky130] integrated source {source}/{size-1}", flush=True)
    return time, energy, peak_temperature


def audit_tables(time, energy):
    off_diagonal = ~np.eye(len(time), dtype=bool)
    for name, table in [("latency", time), ("energy", energy)]:
        if not np.all(np.isfinite(table)) or not np.all(table[off_diagonal] > 0) or np.any(np.diag(table) != 0):
            raise ValueError(f"invalid {name} source-to-target table")
    return {"levels": len(time), "transition_count": int(time.size),
            "latency_s_min": float(time[off_diagonal].min()), "latency_s_max": float(time.max()),
            "energy_j_min": float(energy[off_diagonal].min()), "energy_j_max": float(energy.max()),
            "all_off_diagonal_transitions_positive_finite": True, "zero_cost_identity": True}


def build(root, voltage=DEFAULT_VOLTAGE, output_dir=None):
    if not math.isfinite(voltage) or voltage <= 0:
        raise ValueError("programming voltage must be positive and finite")
    source_dir = root / "device_models/final_rram/source/open_models"
    verify_sources(source_dir)
    directory = root / "device_models/final_rram/models"
    output_dir = output_dir or directory
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"version": VERSION, "external_hspice_energy_read": False, "external_spice_execution": False,
              "programming_voltage_abs_v": voltage, "models": {}}
    for parent_id, _, source_name, cls in MODEL_SPECS:
        path = source_dir / source_name
        params, state_name, _, _, current = make_read_model(path, cls)
        parent_path = directory / f"{parent_id}.json"
        model = copy.deepcopy(json.loads(parent_path.read_text()))
        states = np.asarray(model["levels"][state_name])
        output_id = parent_id.replace("_256", "_write256")
        print(f"[{output_id}] integrating native write dynamics", flush=True)
        if cls:
            time, energy = scalar_tables(native_scalar_integrals(path, cls, params, states, voltage))
            finer_time, finer_energy = scalar_tables(native_scalar_integrals(path, cls, params, states, voltage, order=96))
            mask = ~np.eye(256, dtype=bool)
            error = max(float(np.max(np.abs(finer_time[mask] / time[mask] - 1))),
                        float(np.max(np.abs(finer_energy[mask] / energy[mask] - 1))))
            thermal = "native_instantaneous_Joule_heating" if cls == "Stanford_PKU" else "native_no_thermal_state"
            peak = None
        else:
            time, energy, peak = sky_tables(params, current, states, voltage)
            selected = [0, 64, 128, 254, 255]
            finer_time, finer_energy, _ = sky_tables(params, current, states, voltage, rtol=2e-9, sources=selected)
            mask = np.zeros((256, 256), dtype=bool)
            mask[selected] = True
            np.fill_diagonal(mask, False)
            error = max(float(np.max(np.abs(finer_time[mask] / time[mask] - 1))),
                        float(np.max(np.abs(finer_energy[mask] / energy[mask] - 1))))
            thermal = "native_coupled_filament_and_temperature_ODE; reset_temperature_carried_into_set"
        if error > 5e-5:
            raise ValueError(f"write integral failed numerical convergence: {output_id}: {error}")
        audit = audit_tables(time, energy)
        model["model_id"] = output_id
        model["device"].update(programming_voltage_abs_v=voltage, reset_voltage_abs_v=voltage)
        model["capabilities"].update(hardware_write_accounting=True,
            programming_reachability_verified=True, programming_reachability_scope="ideal_continuous_compact_model_only",
            transition_model="native_dynamics_ideal_stop_at_target", transition_model_exact_within_compact_model=False)
        model["reset_model"] = {"status": "native_negative_voltage_reset_to_code_zero_then_set"}
        model["transition_cost"] = {"latency_s": time.ravel().tolist(), "energy_j": energy.ravel().tolist(),
            "objective_unit": "physical_write_energy_j",
            "semantics": "direct_set_for_increasing_code; native_reset_to_zero_then_set_for_decreasing_code",
            "time_model": "integral_native_state_derivative", "energy_model": "integral_abs_V_I_dt_native_nonlinear_IV",
            "thermal_model": thermal, "set_voltage_v": voltage, "reset_voltage_v": -voltage,
            "time_calibration": "nominal_compact_model_only_not_measured_pulse_calibration",
            "protocol": "ideal_target_terminated_constant_voltage_no_verify_or_pulse_quantization",
            "initial_thermal_state": "ambient_at_start_of_each_source_to_target_transaction",
            "electroforming": "preformed_cells; outside_weight_deployment_protocol"}
        model["write_conversion"] = {"version": VERSION, "parent_model_id": parent_id,
            "parent_sha256": sha256(parent_path), "source_sha256": sha256(path),
            "programming_voltage_selection": f"explicit_experiment_assumption_{voltage:g}V_not_an_upstream_calibrated_protocol",
            "numerical_convergence_max_relative_error": error, "max_temperature_k": peak,
            "measured_programming_calibration": False, "measured_256_level_storage": False,
            "external_hspice_energy_read": False, "external_spice_execution": False}
        model["limitations"] = [item for item in model["limitations"] if not item.startswith("programming voltages")]
        model["limitations"] += ["nominal ideal stop-at-target programming; not a calibrated discrete-pulse/verify controller",
            f"{voltage:g} V cell programming amplitude is an experiment assumption; source-default model parameters are unchanged",
            "ambient temperature is assumed at each transaction start; inter-transaction thermal memory is not retained",
            "write-cell integral excludes unmodeled verify/forming costs; MNSIM separately adds its configured peripherals"]
        output = output_dir / f"{output_id}.json"
        output.write_bytes(json_bytes(model))
        audit.update(model_sha256=sha256(output), convergence_max_relative_error=error,
                     parent_model_id=parent_id, measured_programming_calibration=False, thermal_model=thermal)
        report["models"][output_id] = audit
        print(json.dumps({output_id: audit}), flush=True)
    (output_dir / "open_rram_write_validation.json").write_bytes(json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voltage", type=float, default=DEFAULT_VOLTAGE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    build(Path(__file__).resolve().parents[1], args.voltage, args.output_dir)


if __name__ == "__main__":
    main()
