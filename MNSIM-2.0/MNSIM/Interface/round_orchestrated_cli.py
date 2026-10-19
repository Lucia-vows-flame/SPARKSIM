"""Command-line interface for round-orchestrated MNSIM simulation."""

from __future__ import annotations

import argparse
import io
import math
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from numbers import Real
from pathlib import Path
from typing import Any

from MNSIM.Interface.round_orchestrated_runner import run_round_orchestrated_simulation
from MNSIM.Interface.round_orchestrated_types import (
    RoundOrchestratedRunRequest,
    RoundOrchestratedRunResult,
)


_MISSING = object()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a round-orchestrated MNSIM simulation."
    )
    parser.add_argument("--network", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--sim-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-module")
    parser.add_argument("--run-hardware", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument("--skip-original", action="store_true")
    parser.add_argument("--skip-pim", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--disable-saf",
        action="store_false",
        dest="enable_saf",
        default=True,
    )
    parser.add_argument("--enable-variation", action="store_true")
    parser.add_argument("--enable-r-ratio", action="store_true")
    parser.add_argument(
        "--accuracy-seed",
        type=int,
        help="Seed NumPy while generating accuracy-stage device perturbations.",
    )
    parser.add_argument("--fixed-quantization-range", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--disable-inner-pipeline", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--show-rounds", action="store_true")
    parser.add_argument(
        "--schedule-only",
        action="store_true",
        help="Generate and save a frozen weight schedule without inference.",
    )
    parser.add_argument("--schedule-plan-out", type=Path)
    parser.add_argument(
        "--infer-with-schedule",
        action="store_true",
        help="Execute an existing frozen schedule artifact without replanning.",
    )
    parser.add_argument("--schedule-plan-in", type=Path)
    parser.add_argument(
        "--custom-schedule-in",
        type=Path,
        help="Execute an external mnsim.weight_schedule JSON decision file.",
    )
    parser.add_argument(
        "--schedule-policy",
        choices=(
            "seq",
            "flashsef_ptr",
            "snn_flashsef_ptr",
            "snn-flashsef-ptr",
            "snn_native_temporal",
            "snn_flashsef_temporal",
            "snn-flashsef-temporal",
        ),
        default="seq",
    )
    parser.add_argument(
        "--snn-activity-profile",
        type=Path,
        help=(
            "Deprecated compatibility input; snn_flashsef_ptr now uses native "
            "temporal write costs and ignores spike-transfer profiles."
        ),
    )
    return parser


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments while retaining renderer-only options."""

    return _build_parser().parse_args(argv)


def _request_from_namespace(args: argparse.Namespace) -> RoundOrchestratedRunRequest:
    if not args.skip_accuracy and args.skip_original and args.skip_pim:
        raise SystemExit("--skip-original and --skip-pim cannot be used together")
    return RoundOrchestratedRunRequest(
        network=args.network,
        weights=Path(args.weights),
        num_classes=args.num_classes,
        sim_config=Path(args.sim_config),
        output_dir=Path(args.output_dir),
        dataset_module=args.dataset_module,
        run_hardware=args.run_hardware,
        skip_accuracy=args.skip_accuracy,
        run_original=not args.skip_original,
        run_pim=not args.skip_pim,
        overwrite=args.overwrite,
        enable_saf=args.enable_saf,
        enable_variation=args.enable_variation,
        enable_r_ratio=args.enable_r_ratio,
        accuracy_seed=args.accuracy_seed,
        fixed_quantization_range=args.fixed_quantization_range,
        device=args.device,
        disable_inner_pipeline=args.disable_inner_pipeline,
        max_samples=args.max_samples,
        schedule_only=args.schedule_only,
        schedule_plan_out=args.schedule_plan_out,
        infer_with_schedule=args.infer_with_schedule,
        schedule_plan_in=args.schedule_plan_in,
        custom_schedule_in=args.custom_schedule_in,
        schedule_policy=args.schedule_policy,
        snn_activity_profile=args.snn_activity_profile,
    )


def _request_and_show_rounds(
    argv: list[str],
) -> tuple[RoundOrchestratedRunRequest, bool]:
    args = _parse_args(argv)
    return _request_from_namespace(args), bool(args.show_rounds)


def request_from_args(argv: list[str]) -> RoundOrchestratedRunRequest:
    """Create a public runner request from command-line arguments."""

    request, _ = _request_and_show_rounds(argv)
    return request


def render_terminal_result(
    result: RoundOrchestratedRunResult, *, show_rounds: bool = False
) -> str:
    """Render a deterministic, human-readable terminal summary."""

    request = result.request
    accuracy_status = _stage_status(result, "accuracy")
    hardware_status = _stage_status(result, "hardware")
    lines = [
        "Round-Orchestrated MNSIM",
        f"Network: {request.network}",
        f"Checkpoint: {request.weights}",
        f"SimConfig: {request.sim_config}",
        f"Stages: accuracy={accuracy_status} hardware={hardware_status}",
    ]

    if accuracy_status == "succeeded" and isinstance(
        result.accuracy_report, Mapping
    ):
        if "snn" in request.network.lower():
            lines.extend(
                [
                    "Accuracy",
                    "  SNN Original Accuracy: "
                    + _format_value(
                        result.accuracy_report.get("SNN Original Accuracy")
                    ),
                    "  SNN Processing-in-Memory Accuracy: "
                    + _format_value(
                        result.accuracy_report.get(
                            "SNN Processing-in-Memory Accuracy"
                        )
                    ),
                ]
            )
            lines.extend(
                [
                    "Accuracy Model",
                    "  Hardware profile: "
                    + _format_value(
                        result.accuracy_report.get("hardware_profile")
                    ),
                    "  ADC numerical model: "
                    + _format_value(
                        result.accuracy_report.get("accuracy_adc_model")
                    ),
                    "  LIF perturbation profile: "
                    + _format_value(
                        result.accuracy_report.get(
                            "accuracy_lif_perturbation_profile"
                        )
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "Accuracy",
                    "  Original accuracy: "
                    + _format_value(result.accuracy_report.get("Original accuracy")),
                    "  PIM-based computing accuracy: "
                    + _format_value(
                        result.accuracy_report.get("PIM-based computing accuracy")
                    ),
                ]
            )

    if hardware_status == "succeeded":
        lines.extend(_render_hardware_lines(result))
        if show_rounds and result.phase2_report is not None:
            lines.extend(_render_round_lines(result))

    lines.extend(
        [
            f"Artifacts: {request.output_dir}",
            "Status: " + _overall_status(result.status),
        ]
    )
    return "\n".join(lines)


def _stage_status(result: RoundOrchestratedRunResult, stage_name: str) -> str:
    stage_status = result.stage_statuses.get(stage_name)
    if stage_status is None:
        return "not_requested"
    return str(getattr(stage_status, "status", "not_run"))


def _overall_status(status: Any) -> str:
    return "SUCCEEDED" if str(status).lower() == "succeeded" else "FAILED"


def _render_hardware_lines(result: RoundOrchestratedRunResult) -> list[str]:
    summary = result.hardware_summary
    if not isinstance(summary, Mapping):
        summary = {}
    power = _metric(summary, "power_avg_w")
    if power is None:
        power = _metric(summary, "power_w")
    lines = [
        "Hardware Summary",
        "  Latency: " + _format_scaled(_metric(summary, "latency_s"), "time"),
        "  Modeled energy: " + _format_scaled(_metric(summary, "energy_j"), "energy"),
        "  Read latency: " + _format_scaled(_metric(summary, "read_latency_s"), "time"),
        "  Read energy: " + _format_scaled(_metric(summary, "read_energy_j"), "energy"),
        "  Weight-programming latency: "
        + _format_scaled(_metric(summary, "weight_programming_latency_s"), "time"),
        "  Weight-programming energy: "
        + _format_scaled(_metric(summary, "weight_programming_energy_j"), "energy"),
        "  Average power: " + _format_scaled(power, "power"),
        "  Area: " + _format_scaled(_metric(summary, "area_um2"), "area"),
        "  MACs: " + _format_integer(summary.get("macs_total")),
        "  Network workload: "
        + _format_value(summary.get("network_workload_top_per_inference"))
        + " TOP/inference",
        "  Compute TMAC/s (excludes weight programming): "
        + _format_value(summary.get("compute_tmacs_per_s")),
        "  Compute throughput (excludes weight programming): "
        + _format_value(summary.get("compute_throughput_tops", summary.get("compute_tops")))
        + " TOPS",
        "  End-to-end TMAC/s: " + _format_value(summary.get("tmacs_per_s")),
        "  End-to-end throughput: "
        + _format_value(summary.get("end_to_end_throughput_tops", summary.get("tops")))
        + " TOPS",
        "  Modeled-energy efficiency (partial accounting): "
        + _format_value(summary.get("modeled_energy_efficiency_tops_per_w"))
        + " TOPS/W",
        "  End-to-end TOPS/W: " + _format_value(summary.get("tops_per_w")),
        "  Physical round count: " + _physical_round_count(result),
        "  Cumulative scheduled physical xbars: "
        + _format_integer(summary.get("cumulative_scheduled_physical_xbars")),
        "  Peak scheduled physical xbars: "
        + _format_integer(summary.get("peak_scheduled_physical_xbars")),
    ]
    sequence_length = _integer_value(summary.get("sequence_length", _MISSING))
    lif_update_count = _integer_value(summary.get("lif_update_count", _MISSING))
    if sequence_length is not None:
        lines.append("  Sequence length T: " + str(sequence_length))
    if lif_update_count is not None:
        lines.append("  LIF update count: " + str(lif_update_count))
    return lines


def _physical_round_count(result: RoundOrchestratedRunResult) -> str:
    report = result.phase2_report
    count = _integer_value(
        getattr(report, "execution_item_count", _MISSING) if report is not None else _MISSING
    )
    if count is not None:
        return str(count)

    plan = result.round_plan
    items = getattr(plan, "execution_items", None) if plan is not None else None
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return str(len(items))
    return "n/a"


def _render_round_lines(result: RoundOrchestratedRunResult) -> list[str]:
    report = result.phase2_report
    assert report is not None
    records = getattr(report, "item_results", ())
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return []

    plan = result.round_plan
    plan_items = _plan_items_by_id(plan)
    lines = [
        "Round Details",
        "  Round | Xbars used / capacity | Utilization | Mapped area | "
        "Latency | Energy | Average power",
    ]
    for record in records:
        if not isinstance(record, Mapping):
            continue
        item_id = _round_identifier(record.get("item_id", _MISSING))
        plan_item = plan_items.get(item_id)
        used_xbars = _round_used_xbars(record, plan_item)
        capacity = _round_capacity(plan_item, plan)
        lines.append(
            "  "
            + " | ".join(
                [
                    item_id,
                    _format_xbar_usage(used_xbars, capacity),
                    _format_utilization(used_xbars, capacity),
                    _format_scaled(_mapped_area(record), "area"),
                    _format_scaled(_metric(record, "latency_s"), "time"),
                    _format_scaled(_metric(record, "energy_j"), "energy"),
                    _format_scaled(_round_average_power(record), "power"),
                ]
            )
        )
    return lines


def _plan_items_by_id(plan: Any) -> dict[str, Any]:
    items = getattr(plan, "execution_items", ()) if plan is not None else ()
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
        return {}
    return {
        item_id: item
        for item in items
        if (item_id := _round_identifier(_field(item, "item_id", _MISSING)))
        != "n/a"
    }


def _round_identifier(value: Any) -> str:
    return value if isinstance(value, str) and value else "n/a"


def _round_used_xbars(record: Mapping[str, Any], plan_item: Any) -> int | None:
    used_xbars = _integer_value(record.get("used_xbars", _MISSING))
    if used_xbars is None:
        resource_proof = _field(plan_item, "resource_proof", _MISSING)
        used_xbars = _integer_value(_field(resource_proof, "used_xbars", _MISSING))
    return used_xbars if used_xbars is not None and used_xbars >= 0 else None


def _round_capacity(plan_item: Any, plan: Any) -> int | None:
    resource_proof = _field(plan_item, "resource_proof", _MISSING)
    raw_capacity = _field(resource_proof, "xbar_nums", _MISSING)
    if raw_capacity is not _MISSING:
        item_capacity = _integer_value(raw_capacity)
        return item_capacity if item_capacity is not None and item_capacity > 0 else None
    hardware_budget = _field(plan, "hardware_budget", _MISSING)
    plan_capacity = _integer_value(_field(hardware_budget, "xbar_nums", _MISSING))
    return plan_capacity if plan_capacity is not None and plan_capacity > 0 else None


def _format_xbar_usage(used_xbars: int | None, capacity: int | None) -> str:
    return f"{_format_integer(used_xbars)} / {_format_integer(capacity)}"


def _format_utilization(used_xbars: int | None, capacity: int | None) -> str:
    if used_xbars is None or capacity is None:
        return "n/a"
    return _format_value((used_xbars / capacity) * 100.0) + "%"


def _round_average_power(record: Mapping[str, Any]) -> float | None:
    direct_power = _metric(record, "power_avg_w")
    if direct_power is not None:
        return direct_power
    energy = _metric(record, "energy_j")
    latency = _metric(record, "latency_s")
    if energy is None or latency is None or latency <= 0.0:
        return None
    return _finite_number(energy / latency)


def _mapped_area(record: Mapping[str, Any]) -> float | None:
    area = _metric(record, "mapped_active_area_um2")
    return area if area is not None else _metric(record, "area_um2")


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _metric(values: Mapping[str, Any], key: str) -> float | None:
    return _finite_number(values.get(key, _MISSING))


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _format_integer(value: Any) -> str:
    integer = _integer_value(value)
    return str(integer) if integer is not None else "n/a"


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value if value else "n/a"
    number = _finite_number(value)
    return _format_number(number) if number is not None else "n/a"


def _format_scaled(value: float | None, quantity: str) -> str:
    if value is None:
        return "n/a"
    magnitude = abs(value)
    if quantity == "time":
        if magnitude >= 1.0:
            return _with_unit(value, "s")
        if magnitude >= 1e-3:
            return _with_unit(value * 1e3, "ms")
        if magnitude >= 1e-6:
            return _with_unit(value * 1e6, "us")
        return _with_unit(value * 1e9, "ns")
    if quantity == "energy":
        if magnitude >= 1.0:
            return _with_unit(value, "J")
        if magnitude >= 1e-3:
            return _with_unit(value * 1e3, "mJ")
        if magnitude >= 1e-6:
            return _with_unit(value * 1e6, "uJ")
        return _with_unit(value * 1e9, "nJ")
    if quantity == "power":
        if magnitude >= 1.0:
            return _with_unit(value, "W")
        if magnitude >= 1e-3:
            return _with_unit(value * 1e3, "mW")
        return _with_unit(value * 1e6, "uW")
    if quantity == "area":
        if magnitude >= 1e6:
            return _with_unit(value / 1e6, "mm2")
        return _with_unit(value, "um2")
    raise ValueError(f"unsupported quantity {quantity}")


def _with_unit(value: float, unit: str) -> str:
    return _format_number(value) + " " + unit


def _format_number(value: float) -> str:
    return format(value, ".6g")


def main(argv: list[str] | None = None) -> int:
    """Run the requested simulation and print only its terminal summary."""

    arguments = sys.argv[1:] if argv is None else argv
    request, show_rounds = _request_and_show_rounds(arguments)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        result = run_round_orchestrated_simulation(request)
    print(render_terminal_result(result, show_rounds=show_rounds))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
