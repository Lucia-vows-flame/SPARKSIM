#!/usr/bin/env python3
"""Run the canonical G1-G11 resource ladder through the round runner."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from MNSIM.Interface.final_rram_contract import (
    apply_final_rram_model,
    normalize_final_rram_model_id,
    require_final_rram_hardware_write_support,
    validate_final_rram_parser,
)


RESOURCE_LADDER: tuple[tuple[str, tuple[int, int]], ...] = (
    ("G1", (2, 3)),
    ("G2", (3, 3)),
    ("G3", (3, 4)),
    ("G4", (4, 4)),
    ("G5", (4, 5)),
    ("G6", (5, 5)),
    ("G7", (5, 6)),
    ("G8", (6, 6)),
    ("G9", (7, 7)),
    ("G10", (8, 9)),
    ("G11", (9, 10)),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run canonical G1-G11 with the round-orchestrated runner."
    )
    parser.add_argument("--base-sim-config", required=True)
    parser.add_argument(
        "--device-model",
        default="rram_stanford_asu_256",
        help="Locked final RRAM model id: Stanford/ASU or Stanford-PKU.",
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--network", default="standard_vgg8_snn")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--groups",
        default=",".join(name for name, _ in RESOURCE_LADDER),
        help="Comma-separated subset in canonical order.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_config(path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    if not parser.read(path, encoding="UTF-8"):
        raise FileNotFoundError(f"cannot read base SimConfig: {path}")
    return parser


def _pair(parser: configparser.ConfigParser, section: str, option: str) -> tuple[int, int]:
    parts = [part.strip() for part in parser.get(section, option).split(",")]
    if len(parts) != 2:
        raise ValueError(f"{section}.{option} must contain two dimensions")
    values = (int(parts[0]), int(parts[1]))
    if min(values) <= 0:
        raise ValueError(f"{section}.{option} dimensions must be positive")
    return values


def _validate_base_config(parser: configparser.ConfigParser) -> None:
    if _pair(parser, "Tile level", "PE_Num") != (4, 4):
        raise ValueError("canonical G1-G11 requires PE_Num=4,4")
    if parser.getint("Process element level", "Group_Num") != 2:
        raise ValueError("canonical G1-G11 requires Group_Num=2")
    if parser.getint("Process element level", "Xbar_Polarity") != 2:
        raise ValueError("canonical G1-G11 requires Xbar_Polarity=2")
    if not parser.getboolean("Static Power", "Static_Power_Enable"):
        raise ValueError("G1-G11 static-power sweep requires Static_Power_Enable=1")
    if parser.getboolean("Architecture level", "Buffer_IncludeLeakageEnergy"):
        raise ValueError(
            "central static ledger requires Buffer_IncludeLeakageEnergy=0"
        )


def _resolve_device_model(_repo_root: Path, value: str) -> str:
    return normalize_final_rram_model_id(value)


def _write_group_config(
    *,
    base_config: Path,
    output_path: Path,
    tile_num: tuple[int, int],
    repo_root: Path,
    device_model_id: str = "rram_stanford_asu_256",
) -> tuple[int, dict[str, Any]]:
    parser = _load_config(base_config)
    _validate_base_config(parser)
    model_metadata = apply_final_rram_model(
        parser,
        device_model_id,
        repo_root=repo_root,
    )
    validate_final_rram_parser(parser, source=str(base_config), repo_root=repo_root)
    require_final_rram_hardware_write_support(
        parser, source=str(base_config), repo_root=repo_root
    )
    parser.set("Architecture level", "Tile_Num", f"{tile_num[0]},{tile_num[1]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="UTF-8") as stream:
        parser.write(stream)
    pe_num = _pair(parser, "Tile level", "PE_Num")
    physical_xbars = (
        tile_num[0]
        * tile_num[1]
        * pe_num[0]
        * pe_num[1]
        * parser.getint("Process element level", "Group_Num")
        * parser.getint("Process element level", "Xbar_Polarity")
    )
    return physical_xbars, model_metadata


def _selected_groups(raw: str) -> list[tuple[str, tuple[int, int]]]:
    requested = [value.strip() for value in raw.split(",") if value.strip()]
    known = {name for name, _ in RESOURCE_LADDER}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(f"unknown groups: {unknown}")
    requested_set = set(requested)
    return [entry for entry in RESOURCE_LADDER if entry[0] in requested_set]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="UTF-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _successful_summary(path: Path) -> bool:
    try:
        return str(_read_json(path).get("status", "")).lower() == "succeeded"
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _atomic_write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="UTF-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _result_row(
    *,
    group_name: str,
    tile_num: tuple[int, int],
    physical_xbars: int,
    group_dir: Path,
    status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "group": group_name,
        "tile_num": list(tile_num),
        "physical_xbars": physical_xbars,
        "status": status,
    }
    summary_path = group_dir / "summary.json"
    plan_path = group_dir / "round_plan.json"
    if not _successful_summary(summary_path):
        return row
    summary = _read_json(summary_path)
    hardware = dict(summary.get("hardware") or {})
    audit = dict(hardware.get("static_power_audit") or {})
    capacity_audit = dict(audit.get("capacity_audit") or {})
    blocks = dict(capacity_audit.get("blocks") or {})
    boundary = dict(blocks.get("boundary_state_sram") or {})
    accounting_scope = dict(hardware.get("accounting_scope") or {})
    ddr = dict(accounting_scope.get("ddr_boundary_transfer") or {})
    round_count = None
    if plan_path.is_file():
        plan = dict(_read_json(plan_path).get("round_plan") or {})
        round_count = len(list(plan.get("execution_items") or []))
    row.update(
        {
            "status": "succeeded",
            "round_count": round_count,
            "latency_s": hardware.get("latency_s"),
            "total_energy_j": hardware.get("energy_j"),
            "read_energy_j": hardware.get("read_energy_j"),
            "weight_programming_energy_j": hardware.get(
                "weight_programming_energy_j"
            ),
            "static_energy_j": hardware.get("static_energy_j"),
            "static_power_w": hardware.get("static_average_power_w"),
            "boundary_sram_required_bits": boundary.get(
                "peak_required_capacity_bits", 0
            ),
            "boundary_sram_materialized_bits": boundary.get(
                "materialized_capacity_bits", 0
            ),
            "boundary_sram_access_energy_j": hardware.get(
                "boundary_sram_access_energy_j", 0.0
            ),
            "boundary_sram_access_latency_s": hardware.get(
                "boundary_sram_access_latency_s", 0.0
            ),
            "ddr_used": ddr.get("used", False),
            "static_accounting_complete": hardware.get(
                "static_accounting_complete", False
            ),
        }
    )
    return row


def _attach_model_metadata(
    row: dict[str, Any], model_metadata: dict[str, Any] | None
) -> dict[str, Any]:
    if model_metadata is not None:
        row["device_model"] = model_metadata
    return row


def _runner_command(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    sim_config: Path,
    group_dir: Path,
) -> list[str]:
    return [
        "bash",
        str(repo_root / "scripts" / "run_round_orchestrated_sim.sh"),
        "--network",
        str(args.network),
        "--weights",
        str(Path(args.weights).resolve()),
        "--num-classes",
        str(args.num_classes),
        "--sim-config",
        str(sim_config),
        "--output-dir",
        str(group_dir),
        "--skip-accuracy",
        "--run-hardware",
        "--device",
        str(args.device),
        "--overwrite",
    ]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    base_config = Path(args.base_sim_config).resolve()
    device_model_id = _resolve_device_model(repo_root, args.device_model)
    weights = Path(args.weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {weights}")
    groups = _selected_groups(args.groups)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict[str, Any]] = []

    for group_name, tile_num in groups:
        group_dir = output_root / group_name
        sim_config = group_dir / "SimConfig.ini"
        physical_xbars, model_metadata = _write_group_config(
            base_config=base_config,
            output_path=sim_config,
            tile_num=tile_num,
            device_model_id=device_model_id,
            repo_root=repo_root,
        )
        summary_path = group_dir / "summary.json"
        if args.resume and _successful_summary(summary_path):
            print(f"[{group_name}] resume: already succeeded", flush=True)
            sweep_rows.append(
                _attach_model_metadata(
                    _result_row(
                        group_name=group_name,
                        tile_num=tile_num,
                        physical_xbars=physical_xbars,
                        group_dir=group_dir,
                        status="succeeded",
                    ),
                    model_metadata,
                )
            )
            _atomic_write_json(output_root / "sweep_summary.json", sweep_rows)
            continue

        if args.dry_run:
            print(
                f"[{group_name}] planned tile={tile_num[0]}x{tile_num[1]} "
                f"physical_xbars={physical_xbars}",
                flush=True,
            )
            sweep_rows.append(
                _attach_model_metadata(
                    _result_row(
                        group_name=group_name,
                        tile_num=tile_num,
                        physical_xbars=physical_xbars,
                        group_dir=group_dir,
                        status="planned",
                    ),
                    model_metadata,
                )
            )
            continue

        command = _runner_command(
            repo_root=repo_root,
            args=args,
            sim_config=sim_config,
            group_dir=group_dir,
        )
        print(
            f"[{group_name}] start tile={tile_num[0]}x{tile_num[1]} "
            f"physical_xbars={physical_xbars}",
            flush=True,
        )
        started_at = time.time()
        with (group_dir / "run.log").open("w", encoding="UTF-8") as log_stream:
            process = subprocess.Popen(
                command,
                cwd=repo_root,
                env={
                    **os.environ,
                    "MNSIM_PYTHON": sys.executable,
                },
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return_code = process.wait()
        status = "succeeded" if return_code == 0 and _successful_summary(summary_path) else "failed"
        row = _attach_model_metadata(
            _result_row(
                group_name=group_name,
                tile_num=tile_num,
                physical_xbars=physical_xbars,
                group_dir=group_dir,
                status=status,
            ),
            model_metadata,
        )
        row["elapsed_wall_s"] = time.time() - started_at
        row["return_code"] = return_code
        sweep_rows.append(row)
        _atomic_write_json(output_root / "sweep_summary.json", sweep_rows)
        print(
            f"[{group_name}] {status} elapsed={row['elapsed_wall_s']:.1f}s",
            flush=True,
        )
        if status != "succeeded" and args.stop_on_error:
            return return_code or 1

    _atomic_write_json(output_root / "sweep_summary.json", sweep_rows)
    if args.dry_run:
        return 0
    return 0 if all(row.get("status") == "succeeded" for row in sweep_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
