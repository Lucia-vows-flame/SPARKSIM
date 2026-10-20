"""Transistor-provenance area model for the local analog LIF path.

The model deliberately estimates only the analog frontend and LIF circuit. It
does not reuse the legacy ADC area or a calibrated zero.  Device dimensions
come from the checked-in HSPICE netlists and the layout factor is explicit so
the result remains auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


_MOS_RE = re.compile(
    r"^M\S+\s+\S+\s+\S+\s+\S+\s+\S+\s+\S+"
    r"(?:.*?\bl\s*=\s*'?([^\s']+)'?)"
    r"(?:.*?\bw\s*=\s*'?([^\s']+)'?)",
    re.IGNORECASE,
)
_SCALE = {"": 1.0, "u": 1.0, "n": 1e-3, "m": 1e3, "p": 1e-6, "f": 1e-9}


@dataclass(frozen=True)
class AnalogLifAreaEstimate:
    area_um2_per_neuron: float
    transistor_count: int
    active_device_area_um2: float
    layout_overhead_factor: float
    provenance: tuple[str, ...]

    def for_neurons(self, neuron_count: int) -> float:
        count = int(neuron_count)
        if count <= 0:
            raise ValueError("neuron_count must be positive")
        return self.area_um2_per_neuron * count

    def to_dict(self) -> dict[str, object]:
        return {
            "area_um2_per_neuron": self.area_um2_per_neuron,
            "transistor_count": self.transistor_count,
            "active_device_area_um2": self.active_device_area_um2,
            "layout_overhead_factor": self.layout_overhead_factor,
            "provenance": list(self.provenance),
            "area_source": "hspice_mos_w_times_l_with_explicit_layout_overhead",
        }


def estimate_analog_lif_area(
    *,
    frontend_netlist: str | Path,
    lif_netlists: Iterable[str | Path],
    frontend_instances: int = 2,
    layout_overhead_factor: float = 2.0,
) -> AnalogLifAreaEstimate:
    """Estimate one closed-differential frontend plus one LIF neuron.

    Netlist dimensions use micrometres (the checked-in decks use ``u``).  The
    overhead factor accounts for spacing, contacts, routing and matching; it
    is intentionally supplied as a named assumption rather than hidden in a
    calibration constant.
    """
    overhead = float(layout_overhead_factor)
    if overhead < 1.0:
        raise ValueError("layout_overhead_factor must be >= 1")
    instances = int(frontend_instances)
    if instances <= 0:
        raise ValueError("frontend_instances must be positive")
    paths = [Path(frontend_netlist), *(Path(path) for path in lif_netlists)]
    total_area = 0.0
    transistor_count = 0
    for index, path in enumerate(paths):
        area, count = _parse_mos_area(path)
        if index == 0:
            area *= instances
            count *= instances
        total_area += area
        transistor_count += count
    if transistor_count == 0:
        raise ValueError("analog LIF netlists contain no MOS devices")
    return AnalogLifAreaEstimate(
        area_um2_per_neuron=total_area * overhead,
        transistor_count=transistor_count,
        active_device_area_um2=total_area,
        layout_overhead_factor=overhead,
        provenance=tuple(str(path) for path in paths),
    )


def _parse_mos_area(path: Path) -> tuple[float, int]:
    if not path.is_file():
        raise FileNotFoundError(path)
    area = 0.0
    count = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    params: dict[str, float] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not (stripped.lower().startswith(".subckt") or stripped.startswith("+")):
            continue
        for key, value in re.findall(r"([A-Za-z_]\w*)\s*=\s*([^\s]+)", stripped):
            try:
                params[key.lower()] = _parse_dimension(value, params)
            except ValueError:
                continue
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        match = _MOS_RE.match(line)
        if match is None:
            continue
        length = _parse_dimension(match.group(1), params)
        width = _parse_dimension(match.group(2), params)
        if length <= 0.0 or width <= 0.0:
            raise ValueError(f"non-positive MOS dimensions in {path}: {line}")
        area += length * width
        count += 1
    return area, count


def _parse_dimension(token: str, params: dict[str, float]) -> float:
    value = token.strip().strip("'").lower()
    if value in params:
        return params[value]
    match = re.fullmatch(r"([0-9.eE+-]+)([munpf]?)", value)
    if match is None:
        raise ValueError(f"unsupported dimension {token!r}")
    return float(match.group(1)) * _SCALE[match.group(2)]


__all__ = ["AnalogLifAreaEstimate", "estimate_analog_lif_area"]
