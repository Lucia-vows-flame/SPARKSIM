"""Additional SNN peripherals: explicit physical events, SI units, no SPICE I/O.

This model supplements existing array, frontend, LIF, buffer and reduction
accounting. It never bills those components again. Reads are return-to-zero
pulses; reference/supply bias is read-gated. No implicit standby is assumed.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from MNSIM.Scheduling.phase2.types import Phase2Invalid


SCHEMA = "snn_additional_peripherals/v1"
COMPONENTS = (
    "read_supply", "column_reference", "arow_row_driver", "arow_startup",
    "spike_interconnect", "control", "interface",
)
PARAMETERS = {
    "read_pulse_s", "read_supply_efficiency", "read_supply_bias_w_per_pair",
    "column_reference_w_per_column", "column_reference_capacitance_f_per_column",
    "column_reference_voltage_v", "row_capacitance_f_per_row_per_rail",
    "row_select_energy_j_per_active_row_per_rail", "row_hold_w_per_active_row_per_rail",
    "arow_startup_j_per_pair", "control_j_per_read_pair",
    "spike_link_capacitance_f_per_destination", "spike_voltage_v",
    "interface_j_per_spike_per_destination",
}
FORMULAS = {
    "read_supply": "xbar_load_power_w * read_pulse_s * (1/efficiency-1) + pair_bias_w * read_pulse_s",
    "column_reference": "physical_columns * (reference_w * read_pulse_s + reference_C * reference_V**2)",
    "arow_row_driver": "rails * (row_C * sum(row_voltage_v**2) + active_rows * (select_j + hold_w * read_pulse_s))",
    "arow_startup": "actual_startups * startup_j_per_pair",
    "control": "read_pair_count * control_j_per_read_pair",
    "spike_interconnect": "spike_count * destinations * link_C * spike_V**2",
    "interface": "spike_count * destinations * interface_j_per_event",
}


def _invalid(detail: str) -> Phase2Invalid:
    return Phase2Invalid("INVALID_SNN_PERIPHERAL_ENERGY", detail)


@dataclass(frozen=True)
class PhysicalReadEvent:
    """One executed differential pair read, not a logical layer-size proxy."""

    event_id: str
    sample_id: str
    timestep: int
    instance_id: str
    row_voltage_v: tuple[float, ...]
    columns_per_rail: int
    xbar_load_power_w: float
    rails: int = 2


class SnnPeripheralEnergyModel:
    def __init__(self, profile: dict[str, Any], *, profile_path: str, digest: str):
        required = {"schema_id", "profile_id", "status", "source", "startup_policy", "parameters", "ownership"}
        optional = {"execution_mode"}
        if set(profile) - required - optional or not required.issubset(profile):
            raise _invalid("profile has missing or unsupported fields")
        if profile["schema_id"] != SCHEMA:
            raise _invalid("unsupported peripheral profile schema")
        if profile["status"] not in {"analytical_estimate", "calibrated", "example_only"}:
            raise _invalid("unsupported profile status")
        if not all(isinstance(profile[key], str) and profile[key].strip() for key in ("profile_id", "source")):
            raise _invalid("profile_id and source must be non-empty")
        if profile["startup_policy"] not in {"once_per_sample_instance", "every_read"}:
            raise _invalid("unsupported startup policy")
        execution_mode = str(profile.get("execution_mode", "analog_activity"))
        if execution_mode not in {"analog_activity", "mnsim_static_topology"}:
            raise _invalid("unsupported execution mode")
        params = profile["parameters"]
        if not isinstance(params, dict) or set(params) != PARAMETERS:
            raise _invalid("all peripheral parameters (and only those parameters) are required")
        # Explicit ownership is necessary because aggregate LUT energy cannot
        # tell us which supply/reference/driver circuits it already contains.
        # Signal transport is allowed to remain owned by the existing
        # boundary/buffer path when the profile does not provide an activity
        # model for it.  All physical read peripherals must be explicitly
        # declared as additions.
        ownership = profile["ownership"]
        if not isinstance(ownership, dict) or set(ownership) != set(COMPONENTS):
            raise _invalid("every component requires explicit ownership")
        for component, owner in ownership.items():
            allowed = {"additional_not_in_existing_models"}
            if component in {"spike_interconnect", "interface"}:
                allowed.add("existing_boundary_and_buffer_accounting")
            if owner not in allowed:
                raise _invalid(f"unsupported ownership for {component}")
        if any(
            ownership[name] == "existing_boundary_and_buffer_accounting"
            and float(profile["parameters"].get(param, 0.0)) != 0.0
            for name, param in {
                "spike_interconnect": "spike_link_capacitance_f_per_destination",
                "interface": "interface_j_per_spike_per_destination",
            }.items()
        ):
            raise _invalid("existing boundary-owned components must have zero added parameters")
        self.parameters = {}
        for name, value in params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise _invalid(f"{name} must be finite and non-negative in SI units")
            self.parameters[name] = float(value)
        if self.parameters["read_pulse_s"] <= 0 or not 0 < self.parameters["read_supply_efficiency"] <= 1:
            raise _invalid("read_pulse_s must be positive and efficiency in (0, 1]")
        self.profile = deepcopy(profile)
        self.execution_mode = execution_mode
        self.profile_path = profile_path
        self.digest = digest
        self._seen_events: set[str] = set()
        self._started: set[tuple[str, str]] = set()

    def score(self, reads: list[PhysicalReadEvent], *, spike_codes: Any, destinations: list[str]) -> dict[str, Any]:
        """Score actual read and spike-delivery events, rejecting double submission."""
        spikes = np.asarray(spike_codes)
        if spikes.ndim != 1 or not np.all((spikes == 0) | (spikes == 1)):
            raise _invalid("spike_codes must be a one-dimensional binary vector")
        if len(destinations) != len(set(destinations)) or any(not isinstance(d, str) or not d for d in destinations):
            raise _invalid("destination IDs must be distinct non-empty strings")
        if not reads:
            raise _invalid("at least one actual read event is required")
        new_ids = set()
        for event in reads:
            if not event.event_id or event.event_id in self._seen_events or event.event_id in new_ids:
                raise _invalid("duplicate or empty physical read event ID")
            new_ids.add(event.event_id)
            if not event.sample_id or not event.instance_id or type(event.timestep) is not int or event.timestep < 0:
                raise _invalid("read event requires sample, physical instance and non-negative timestep")
            if type(event.columns_per_rail) is not int or event.columns_per_rail <= 0 or event.rails != 2:
                raise _invalid("read event requires positive columns and two physical differential rails")
            voltage = np.asarray(event.row_voltage_v, dtype=float)
            if voltage.ndim != 1 or not voltage.size or np.any(~np.isfinite(voltage)) or np.any(voltage < 0):
                raise _invalid("row voltages must be a finite non-negative vector")
            if not math.isfinite(event.xbar_load_power_w) or event.xbar_load_power_w < 0:
                raise _invalid("xbar load power must be finite and non-negative")
        p = self.parameters
        t = p["read_pulse_s"]
        energy = dict.fromkeys(COMPONENTS, 0.0)
        starts = 0
        active_rows = 0
        physical_columns = 0
        started = set(self._started)
        for event in reads:
            voltage = np.asarray(event.row_voltage_v, dtype=float)
            active = int(np.count_nonzero(voltage))
            columns = event.rails * event.columns_per_rail
            active_rows += event.rails * active
            physical_columns += columns
            energy["read_supply"] += event.xbar_load_power_w * t * (1 / p["read_supply_efficiency"] - 1) + p["read_supply_bias_w_per_pair"] * t
            energy["column_reference"] += columns * (p["column_reference_w_per_column"] * t + p["column_reference_capacitance_f_per_column"] * p["column_reference_voltage_v"] ** 2)
            energy["arow_row_driver"] += event.rails * (p["row_capacitance_f_per_row_per_rail"] * float(voltage @ voltage) + active * (p["row_select_energy_j_per_active_row_per_rail"] + p["row_hold_w_per_active_row_per_rail"] * t))
            key = (event.sample_id, event.instance_id)
            if self.profile["startup_policy"] == "every_read" or key not in started:
                starts += 1
                started.add(key)
            energy["control"] += p["control_j_per_read_pair"]
        energy["arow_startup"] = starts * p["arow_startup_j_per_pair"]
        deliveries = int(np.count_nonzero(spikes)) * len(destinations)
        energy["spike_interconnect"] = deliveries * p["spike_link_capacitance_f_per_destination"] * p["spike_voltage_v"] ** 2
        energy["interface"] = deliveries * p["interface_j_per_spike_per_destination"]
        total = math.fsum(energy.values())
        if not math.isfinite(total) or any(not math.isfinite(e) for e in energy.values()):
            raise _invalid("computed peripheral energy overflow")
        self._seen_events.update(new_ids)
        self._started = started
        return {
            "schema_id": SCHEMA,
            "included_components_j": energy,
            "total_energy_j": total,
            "component_provenance": {name: {"formula": FORMULAS[name], "parameters": dict(p), "source": self.profile["source"]} for name in COMPONENTS},
            "profile_id": self.profile["profile_id"],
            "profile_status": self.profile["status"],
            "profile_path": self.profile_path,
            "profile_sha256": self.digest,
            "ownership": dict(self.profile["ownership"]),
            "additional_model_external_energy_read": False,
            "counts": {"read_pairs": len(reads), "physical_active_rows": active_rows, "physical_columns": physical_columns, "startups": starts, "spikes": int(np.count_nonzero(spikes)), "destinations": len(destinations), "spike_deliveries": deliveries},
            "destinations": list(destinations),
            "read_events": [asdict(event) for event in reads],
            "diagnostics_j": {"electrical_xbar_load_not_added_again": math.fsum(event.xbar_load_power_w * t for event in reads)},
            "boundary": "additional peripherals only; array load is diagnostic, existing xbar/frontend/LIF/buffers/partial reductions remain billed by their original owners; read-gated references/bias; return-to-zero row pulses; no new standby model",
            "reconciliation_passed": math.isclose(total, math.fsum(energy.values()), rel_tol=1e-12, abs_tol=1e-30),
        }

    def score_static_topology(
        self,
        *,
        reads: list[PhysicalReadEvent],
        sample_id: str,
        timestep: int,
    ) -> dict[str, Any]:
        """Score a digital-LIF run from the executed MNSIM physical reads.

        The digital backend does not expose analog row values or spike vectors.
        Its static audit does expose the physical xbar read count and energy,
        so this mode charges only read-gated peripheral circuits.  Spike
        delivery remains owned by the existing boundary/buffer path and is
        explicitly recorded as unavailable rather than synthesized.
        """
        if self.execution_mode != "mnsim_static_topology":
            raise _invalid("profile is not configured for static topology execution")
        if not reads:
            raise _invalid("at least one physical read event is required")
        empty_spikes = np.zeros(0, dtype=np.int8)
        result = self.score(reads, spike_codes=empty_spikes, destinations=[])
        result["signal_source"] = "mnsim_static_read_audit"
        result["sample_id"] = str(sample_id)
        result["timestep"] = int(timestep)
        result["diagnostics_j"]["spike_activity_unavailable"] = True
        result["diagnostics_j"]["spike_delivery_owner"] = "existing_boundary_and_buffer_accounting"
        return result


def load_snn_peripheral_energy(sim_config_path: str | Path) -> SnnPeripheralEnergyModel | None:
    path = Path(sim_config_path)
    parser = configparser.ConfigParser(interpolation=None)
    try:
        if not parser.read(path, encoding="utf-8"):
            raise _invalid(f"cannot read {path}")
        section = "SNN Peripheral Energy"
        if not parser.has_section(section):
            return None
        if set(parser[section]) - {"enabled", "profile"}:
            raise _invalid("unsupported peripheral configuration option")
        if not parser.getboolean(section, "Enabled"):
            return None
        raw_path = parser.get(section, "Profile").strip()
        if not raw_path:
            raise _invalid("enabled model requires Profile")
        profile_path = Path(raw_path).expanduser()
        if not profile_path.is_absolute():
            profile_path = path.resolve().parent / profile_path
        payload = profile_path.read_bytes()
        profile = json.loads(payload)
        if not isinstance(profile, dict):
            raise _invalid("profile must be a JSON object")
        return SnnPeripheralEnergyModel(profile, profile_path=str(profile_path.resolve()), digest=hashlib.sha256(payload).hexdigest())
    except (OSError, ValueError, configparser.Error) as exc:
        raise _invalid(str(exc)) from exc


__all__ = ["PhysicalReadEvent", "SnnPeripheralEnergyModel", "load_snn_peripheral_energy"]
