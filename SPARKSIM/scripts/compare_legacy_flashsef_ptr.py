#!/usr/bin/env python3
"""Replay complete PTR writes and compare with MNSIM's common read workload.

This evaluates the saved transforms; it does not project them onto the tile-only
custom schedule format. Total metrics assume identical read workload and SRAM
provisioning. The read report includes the configured MNSIM circuit ledger,
including enabled ASU SNN activity peripherals.
"""
from __future__ import annotations

import argparse
import configparser
import csv
import json
from pathlib import Path
import sys

import numpy as np

from generate_legacy_flashsef_ptr import ROOT, read_hardware, round_payloads, sha, write_json
from verify_legacy_flashsef_ptr import verify

sys.path.insert(0, str(ROOT))
from MNSIM.Scheduling.phase2.physical_write_ledger import (
    _aggregate_round_cost,
    load_physical_write_tables,
    score_xbar_physical_transition,
)


def check_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-24)


def score_tile(tables, original, target, tile_id):
    costs = {}
    for pr, pc, g, b, pol in np.ndindex(target.shape[:-2]):
        assert b == 0, "only single-plane deployments are supported"
        key = (pr, pc, g, b, pol)
        slot = (tile_id, pr * target.shape[1] + pc, g, pol)
        costs[slot] = score_xbar_physical_transition(
            tables=tables, original_xbar=original[key], target_xbar=target[key])
    return costs


def replay(schedule_path):
    """Independently score Seq active slots and PTR complete physical tiles."""
    verify(schedule_path)
    directory = schedule_path.parent
    schedule = json.loads(schedule_path.read_text())
    seq_path = ROOT / schedule["input_seq_artifact"]["path"]
    seq = json.loads(seq_path.read_text())
    config_path = directory / "SimConfig.schedule.ini"
    hw = read_hardware(config_path)
    tables = load_physical_write_tables(str(config_path))
    seq_resident, ptr_resident, output = {}, {}, []
    erased = np.zeros(tables.xbar_shape, dtype=np.int32)
    for source, saved in zip(seq["rounds"], schedule["rounds"]):
        _, payloads, _ = round_payloads(source, seq["round_plan_snapshot"], hw)
        seq_costs = {}
        for fid, spec in source["placement_snapshot"]["fragment_specs_by_id"].items():
            slot = source["placement_snapshot"]["fragment_slots_by_id"][fid]
            tid, pe, g = [int(slot[k]) for k in ["tile_id", "pe_id", "group_id"]]
            payload = payloads[f"tile:{tid}/pe:{pe}"].payload
            for pol in range(hw.xbar_polarity):
                address = (tid, pe, g, pol)
                assert address not in seq_costs
                target = payload[g, 0, pol]
                seq_costs[address] = score_xbar_physical_transition(
                    tables=tables, original_xbar=seq_resident.get(address, erased), target_xbar=target)
                seq_resident[address] = target.copy()
        seq_latency, seq_energy = _aggregate_round_cost(seq_costs)
        ptr_costs, tile_reports = {}, []
        with np.load(directory / saved["deployment_file"], allow_pickle=False) as arrays:
            for tile in saved["tiles"]:
                tid, prefix = tile["physical_tile_id"], tile["array_prefix"]
                original = arrays[f"{prefix}_original"]
                target = arrays[f"{prefix}_optimized"]
                np.testing.assert_array_equal(original, ptr_resident.get(tid, np.zeros_like(original)))
                costs = score_tile(tables, original, target, tid)
                latency, energy = _aggregate_round_cost(costs)
                check_close([latency, energy], [tile["write_latency_s"], tile["write_energy_j"]])
                assert not (set(costs) & set(ptr_costs))
                ptr_costs.update(costs)
                ptr_resident[tid] = target.copy()
                tile_reports.append({"physical_tile_id": tid, "latency_s": latency, "energy_j": energy})
        ptr_latency, ptr_energy = _aggregate_round_cost(ptr_costs)
        output.append({"round_id": source["round_id"],
                       "seq": {"latency_s": seq_latency, "energy_j": seq_energy},
                       "flashsef_ptr": {"latency_s": ptr_latency, "energy_j": ptr_energy},
                       "flashsef_ptr_tiles": tile_reports})
    return schedule, seq, output, tables


def reductions(seq, ptr):
    return {f"{k}_reduction_pct": (1 - ptr[k] / seq[k]) * 100 if seq[k] else None
            for k in ["energy_j", "latency_s"]}


def compare(schedule_path, seq_report_path):
    schedule, seq, rounds, tables = replay(schedule_path)
    payload = json.loads(seq_report_path.read_text())
    assert payload["status"] == "succeeded"
    report = payload["phase2_report"]
    metrics = report["inference_metrics"]
    manifest = json.loads((seq_report_path.parent / "run_manifest.json").read_text())
    config_path = Path(report["sim_config_path"])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    assert sha(config_path) == schedule["sim_config_sha256"]
    assert report["model_name"] == schedule["model_id"]
    assert sha(Path(manifest["request"]["weights"])) == schedule["checkpoint"]["sha256"]
    assert manifest["request"]["infer_with_schedule"] is True
    assert sha(Path(manifest["request"]["schedule_plan_in"])) == schedule["input_seq_artifact"]["sha256"]
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    device_id = cfg.get("Final RRAM Model", "Model_ID", fallback=None)
    device_metadata = schedule.get("device_model")
    if device_metadata is not None:
        assert device_metadata["model_id"] == device_id
        model_path = Path(cfg.get("Final RRAM Model", "Model_JSON"))
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        assert sha(model_path) == device_metadata["model_json_sha256"]
        # The native digest includes its config path. The PTR replay uses an
        # identical saved copy at another path, so verify native identity and
        # numerical equivalence separately.
        native_tables = load_physical_write_tables(str(config_path.resolve()))
        assert native_tables.table_digest == device_metadata["write_table_digest"]
        np.testing.assert_array_equal(tables.latency_ns, native_tables.latency_ns)
        np.testing.assert_array_equal(tables.energy_nj, native_tables.energy_nj)
    if device_id is not None:
        assert manifest["stage_statuses"]["hardware"]["summary"]["final_rram_model"]["model_id"] == device_id
    assert cfg.getint("Algorithm Configuration", "NoC_enable") == 0
    assert cfg.get("Static Power", "Static_Power_Policy") == "no_implicit_power_gating"
    runtime_rounds = {r["item_id"]: r for r in report["item_results"] if r["weight_programming"]}
    assert set(runtime_rounds) == {r["round_id"] for r in rounds}
    for r in rounds:
        actual = runtime_rounds[r["round_id"]]
        check_close([r["seq"]["latency_s"], r["seq"]["energy_j"]],
                    [actual["weight_programming_latency_s"], actual["weight_programming_energy_j"]])
    writes = {policy: {key: sum(r[policy][key] for r in rounds) for key in ["energy_j", "latency_s"]}
              for policy in ["seq", "flashsef_ptr"]}
    check_close([writes["seq"]["latency_s"], writes["seq"]["energy_j"]],
                [metrics["weight_programming_latency_s"], metrics["weight_programming_energy_j"]])
    p_static = float(metrics["static_average_power_w"])
    read = {"energy_j": float(metrics["read_energy_j"]), "latency_s": float(metrics["read_latency_s"])}
    circuit_ledger = dict(metrics.get("energy_ledger", {}) or {})
    if not circuit_ledger:
        raise AssertionError("phase2 report does not contain the circuit energy ledger")
    if circuit_ledger.get("diagnostics_j", {}).get("external_hspice_energy_read") is not False:
        raise AssertionError("phase2 report must prove no external HSPICE energy was read")
    check_close(metrics["static_energy_j"], p_static * metrics["latency_s"])
    check_close(metrics["energy_j"], read["energy_j"] + writes["seq"]["energy_j"] + metrics["static_energy_j"])
    totals = {}
    ops_total = float(metrics["ops_total"])
    for policy in writes:
        latency = read["latency_s"] + writes[policy]["latency_s"]
        static = p_static * latency
        energy = read["energy_j"] + writes[policy]["energy_j"] + static
        totals[policy] = {
            "latency_s": latency,
            "energy_j": energy,
            "static_energy_j": static,
            "tops": ops_total / latency / 1e12 if latency > 0 else 0.0,
            "tops_per_w": ops_total / energy / 1e12 if energy > 0 else 0.0,
        }
    check_close([totals["seq"]["latency_s"], totals["seq"]["energy_j"]], [metrics["latency_s"], metrics["energy_j"]])
    write_updates = {policy: {key: sum(r[policy][key] for r in rounds[1:]) for key in ["energy_j", "latency_s"]}
                     for policy in writes}
    return {"model": schedule["model_id"], "round_count": len(rounds),
            "device_model": device_metadata or {"model_id": device_id},
            "execution_profile": {"lif_mode": cfg.get("Neuron", "LIF_Mode", fallback="digital"),
                                  "weight_quantization_bit": cfg.getint("Algorithm Configuration", "Weight_Quantization_Bit"),
                                  "additional_peripheral_energy_enabled": cfg.getboolean("SNN Peripheral Energy", "Enabled", fallback=False),
                                  "accuracy_executed": not manifest["request"]["skip_accuracy"]},
            "physical_xbars": schedule["settings"]["physical_xbars"],
            "read_workload": read, "static_power_w": p_static,
            "circuit_energy_ledger": circuit_ledger,
            "write_including_initial": {**writes, **reductions(writes["seq"], writes["flashsef_ptr"])},
            "write_updates_only": {**write_updates, **reductions(write_updates["seq"], write_updates["flashsef_ptr"])},
            "modeled_total_common_read": {**totals, **reductions(totals["seq"], totals["flashsef_ptr"])},
            "validation": {"independent_saved_transform_verification": "passed",
                           "seq_replay_matches_native_phase2_all_rounds": True,
                           "ptr_replay_matches_original_kernel_all_tiles": True,
                           "energy_and_latency_closure": "passed"},
            "inputs": {"flashsef_ptr_schedule": str(schedule_path), "flashsef_ptr_sha256": sha(schedule_path),
                       "seq_phase2_report": str(seq_report_path), "seq_phase2_report_sha256": sha(seq_report_path),
                       "sim_config_sha256": schedule["sim_config_sha256"], "write_table_digest": tables.table_digest},
            "rounds": rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule-root", type=Path, default=Path("runs/legacy_flashsef_ptr_16xbar_20261010"))
    parser.add_argument("--evaluation-root", type=Path, default=Path("runs/seq_original_ptr_16xbar_eval_20261010"))
    parser.add_argument("--models", nargs="+", default=["lenet_snn", "alexnet_snn", "resnet34_snn", "vgg8_snn", "mobilenet_snn"])
    args = parser.parse_args()
    results = []
    for model in args.models:
        result = compare(args.schedule_root / model / "flashsef_ptr.schedule.json",
                         args.evaluation_root / model / "seq/phase2_report.json")
        write_json(args.evaluation_root / model / "comparison.json", result)
        results.append(result)
        print(json.dumps({"model": model, "write": result["write_including_initial"],
                          "modeled_total_common_read": result["modeled_total_common_read"]}), flush=True)
    summary = {"schema_version": 1, "algorithm": "uploaded_original_FlashSeF_PTR",
               "scope": {"write": "exact replay of saved physical states using current MNSIM transition tables",
                         "initial_state": "all_zero; first deployment included",
                         "aggregation": "cells: sum energy; OU latency: row max then sum; xbar/PE/tile: max latency; rounds: sum latency",
                         "read": "common canonical MNSIM RoundPlan workload with original SNN T; not transformed inference execution",
                         "static": "same SRAM provisioning as Seq; modeled dynamic metric does not add unrelated whole-chip standby",
                         "excluded": [],
                         "energy_boundary": "weight programming plus MNSIM inference hardware and enabled activity peripherals; no component in that boundary is dropped",
                         "external_hspice_energy_read": False,
                         "energy_accounting_complete": all(bool(r["circuit_energy_ledger"].get("accounting_complete")) for r in results)},
               "results": results}
    write_json(args.evaluation_root / "comparison.json", summary)
    with (args.evaluation_root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "rounds", "scope", "seq_energy_uJ", "ptr_energy_uJ", "energy_reduction_pct",
                         "seq_latency_ms", "ptr_latency_ms", "latency_reduction_pct"])
        for result in results:
            for scope in ["write_including_initial", "write_updates_only", "modeled_total_common_read"]:
                row = result[scope]
                writer.writerow([result["model"], result["round_count"], scope,
                                 row["seq"]["energy_j"] * 1e6, row["flashsef_ptr"]["energy_j"] * 1e6,
                                 row["energy_j_reduction_pct"], row["seq"]["latency_s"] * 1e3,
                                 row["flashsef_ptr"]["latency_s"] * 1e3, row["latency_s_reduction_pct"]])


if __name__ == "__main__":
    main()
