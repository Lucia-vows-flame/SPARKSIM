"""DDR boundary-transfer backends for the round-orchestrated Phase 2 path.

The boundary manager owns the accounting point.  Backends only translate the
normalized boundary events into a memory-system estimate and return a stable
result schema.  The default backend is intentionally disabled.
"""

from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Protocol


class DdrBackendError(RuntimeError):
    """Raised when a configured DDR backend cannot produce a result."""


class DdrBackend(Protocol):
    enabled: bool

    def score_events(self, events: list[Any]) -> dict[str, Any]:
        ...


def empty_ddr_result(*, backend: str = "disabled") -> dict[str, Any]:
    return {
        "used": False,
        "backend": str(backend),
        "transaction_count": 0,
        "read_transactions": 0,
        "write_transactions": 0,
        "read_bytes": 0,
        "write_bytes": 0,
        "simulated_cycles": 0,
        "latency_s": 0.0,
        "energy_j": 0.0,
        "power_w": 0.0,
    }


class NullDdrBackend:
    enabled = False

    def score_events(self, events: list[Any]) -> dict[str, Any]:
        del events
        return empty_ddr_result()


class Dramsim3Backend:
    """Run DRAMSim3 on the transactions represented by boundary events."""

    enabled = True

    def __init__(
        self,
        *,
        binary_path: str | Path,
        config_path: str | Path,
        transaction_bytes: int = 64,
        default_bitwidth: int = 8,
        cycles_per_transaction: int = 128,
        timeout_s: float = 120.0,
    ) -> None:
        self.binary_path = Path(binary_path).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.transaction_bytes = _positive_int(transaction_bytes, "transaction_bytes")
        self.default_bitwidth = _positive_int(default_bitwidth, "default_bitwidth")
        self.cycles_per_transaction = _positive_int(
            cycles_per_transaction, "cycles_per_transaction"
        )
        self.timeout_s = float(timeout_s)
        if self.timeout_s <= 0:
            raise DdrBackendError("timeout_s must be positive")
        if not self.binary_path.is_file():
            raise DdrBackendError(f"DRAMSim3 binary does not exist: {self.binary_path}")
        if not self.config_path.is_file():
            raise DdrBackendError(f"DRAMSim3 config does not exist: {self.config_path}")

    def score_events(self, events: list[Any]) -> dict[str, Any]:
        transactions = _events_to_transactions(
            events,
            transaction_bytes=self.transaction_bytes,
            default_bitwidth=self.default_bitwidth,
        )
        if not transactions:
            result = empty_ddr_result(backend="dramsim3")
            result["used"] = False
            return result

        cycles = max(
            100,
            len(transactions) * self.cycles_per_transaction,
        )
        with tempfile.TemporaryDirectory(prefix="mnsim_dramsim3_") as temp_dir:
            temp_root = Path(temp_dir)
            trace_path = temp_root / "boundary.trace"
            output_dir = temp_root / "output"
            output_dir.mkdir()
            trace_path.write_text(
                "".join(
                    f"{address:x} {op} {cycle}\n"
                    for cycle, (address, op) in enumerate(transactions)
                ),
                encoding="utf-8",
            )
            try:
                completed = subprocess.run(
                    [
                        str(self.binary_path),
                        str(self.config_path),
                        "--cycles",
                        str(cycles),
                        "--trace",
                        str(trace_path),
                        "--output-dir",
                        str(output_dir),
                    ],
                    cwd=str(self.binary_path.parent),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DdrBackendError(f"DRAMSim3 invocation failed: {exc}") from exc
            if completed.returncode != 0:
                raise DdrBackendError(
                    "DRAMSim3 returned "
                    f"{completed.returncode}: {completed.stderr.strip() or completed.stdout.strip()}"
                )
            stats_path = output_dir / "dramsim3.json"
            if not stats_path.is_file():
                raise DdrBackendError(
                    f"DRAMSim3 did not produce expected stats file: {stats_path}"
                )
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DdrBackendError(f"cannot parse DRAMSim3 stats: {exc}") from exc

        return _summarize_stats(
            stats,
            transactions=transactions,
            transaction_bytes=self.transaction_bytes,
            simulated_cycles=cycles,
            config_path=self.config_path,
        )


def load_ddr_backend(sim_config_path: str | Path) -> DdrBackend:
    """Load the configured backend; missing DDR configuration means disabled."""

    parser = configparser.ConfigParser(interpolation=None)
    try:
        loaded = parser.read(sim_config_path, encoding="UTF-8")
    except (OSError, configparser.Error):
        loaded = []
    section = "DDR" if parser.has_section("DDR") else "Architecture level"
    if not loaded or not parser.has_section(section):
        return NullDdrBackend()
    enabled = parser.getboolean(section, "DDR_Enable", fallback=False)
    backend_name = parser.get(section, "DDR_Backend", fallback="disabled").strip().lower()
    if not enabled or backend_name in {"", "none", "null", "disabled"}:
        return NullDdrBackend()
    if backend_name != "dramsim3":
        raise DdrBackendError(f"unsupported DDR backend: {backend_name}")

    root = Path(__file__).resolve().parents[3]
    binary_path = _resolve_config_path(
        parser.get(
            section,
            "DRAMSim3_Binary",
            fallback="third_party/DRAMSim3/dramsim3main.out",
        ),
        root,
    )
    config_path = _resolve_config_path(
        parser.get(
            section,
            "DRAMSim3_Config",
            fallback="third_party/DRAMSim3/configs/DDR4_8Gb_x8_3200.ini",
        ),
        root,
    )
    return Dramsim3Backend(
        binary_path=binary_path,
        config_path=config_path,
        transaction_bytes=parser.getint(section, "DDR_TransactionBytes", fallback=64),
        default_bitwidth=parser.getint(section, "DDR_DefaultBitwidth", fallback=8),
        cycles_per_transaction=parser.getint(
            section, "DRAMSim3_CyclesPerTransaction", fallback=128
        ),
        timeout_s=parser.getfloat(section, "DRAMSim3_TimeoutSeconds", fallback=120.0),
    )


def _resolve_config_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise DdrBackendError(f"{name} must be positive")
    return value


def _event_dict(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    detail = getattr(event, "detail", {})
    return {
        "event_kind": str(getattr(event, "event_kind", "")),
        "state_id": str(getattr(event, "state_id", "")),
        "detail": dict(detail or {}),
    }


def _events_to_transactions(
    events: list[Any], *, transaction_bytes: int, default_bitwidth: int
) -> list[tuple[int, str]]:
    transactions: list[tuple[int, str]] = []
    for event in events:
        data = _event_dict(event)
        kind = str(data.get("event_kind", ""))
        if kind == "read_boundary_state":
            operation = "READ"
        elif kind in {"write_boundary_state", "update_boundary_state"}:
            operation = "WRITE"
        else:
            continue
        detail = dict(data.get("detail", {}) or {})
        payload_bytes = int(detail.get("payload_bytes", 0) or 0)
        if payload_bytes <= 0:
            payload_bytes = _fallback_payload_bytes(detail, default_bitwidth)
        payload_bytes = max(payload_bytes, transaction_bytes)
        base_address = _stable_address(str(data.get("state_id", "")), transaction_bytes)
        for offset in range(0, payload_bytes, transaction_bytes):
            transactions.append((base_address + offset, operation))
    return transactions


def _fallback_payload_bytes(detail: dict[str, Any], default_bitwidth: int) -> int:
    shape = dict(detail.get("shape", {}) or {})
    count = _shape_element_count(shape)
    if count <= 0:
        return 0
    bitwidth = int(detail.get("data_bitwidth") or default_bitwidth)
    return max(1, (count * bitwidth + 7) // 8)


def _shape_element_count(shape: dict[str, Any]) -> int:
    channels = None
    for key in ("Outputchannel", "Outfeature", "C", "channels", "features"):
        if key in shape:
            try:
                channels = int(shape[key])
            except (TypeError, ValueError):
                channels = None
            break
    output_size = shape.get("Outputsize", shape.get("output_size"))
    if isinstance(output_size, (list, tuple)) and output_size:
        try:
            spatial = 1
            for value in output_size:
                spatial *= int(value)
        except (TypeError, ValueError):
            spatial = 1
    else:
        spatial = 1
    return max(0, int(channels or 0) * spatial)


def _stable_address(state_id: str, transaction_bytes: int) -> int:
    digest = hashlib.sha256(state_id.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "little")
    return raw - raw % transaction_bytes


def _summarize_stats(
    stats: Any,
    *,
    transactions: list[tuple[int, str]],
    transaction_bytes: int,
    simulated_cycles: int,
    config_path: Path,
) -> dict[str, Any]:
    channel_stats = [value for value in (stats or {}).values() if isinstance(value, dict)]
    read_count = sum(int(value.get("num_reads_done", 0) or 0) for value in channel_stats)
    write_count = sum(int(value.get("num_writes_done", 0) or 0) for value in channel_stats)
    total_energy_pj = sum(float(value.get("total_energy", 0.0) or 0.0) for value in channel_stats)
    average_read_latency = _weighted_average(
        ((float(value.get("average_read_latency", 0.0) or 0.0), int(value.get("num_reads_done", 0) or 0))
         for value in channel_stats),
    )
    tck_ns = _read_tck_ns(config_path)
    read_bytes = sum(1 for _, op in transactions if op == "READ") * transaction_bytes
    write_bytes = sum(1 for _, op in transactions if op == "WRITE") * transaction_bytes
    latency_s = average_read_latency * tck_ns * 1e-9
    energy_j = total_energy_pj * 1e-12
    return {
        "used": True,
        "backend": "dramsim3",
        "transaction_count": len(transactions),
        "read_transactions": read_bytes // transaction_bytes,
        "write_transactions": write_bytes // transaction_bytes,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "completed_read_transactions": read_count,
        "completed_write_transactions": write_count,
        "simulated_cycles": int(simulated_cycles),
        "tck_ns": tck_ns,
        "average_read_latency_cycles": average_read_latency,
        "energy_pj": total_energy_pj,
        "latency_s": latency_s,
        "energy_j": energy_j,
        "power_w": energy_j / latency_s if latency_s > 0 else 0.0,
    }


def _weighted_average(values: Any) -> float:
    total = 0.0
    weight = 0
    for value, count in values:
        if count <= 0:
            continue
        total += float(value) * count
        weight += count
    return total / weight if weight else 0.0


def _read_tck_ns(config_path: Path) -> float:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path, encoding="UTF-8")
    return float(parser.get("timing", "tCK", fallback="1.0"))


__all__ = [
    "DdrBackend",
    "DdrBackendError",
    "Dramsim3Backend",
    "NullDdrBackend",
    "empty_ddr_result",
    "load_ddr_backend",
]
