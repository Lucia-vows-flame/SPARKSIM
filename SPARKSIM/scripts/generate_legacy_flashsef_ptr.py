#!/usr/bin/env python3
"""Generate complete deployments by calling the uploaded FlashSeF-PTR kernels.

Run each model in a fresh process. The legacy import namespace is deliberately
isolated from the current simulator. No optimizer is reimplemented here.
"""
from __future__ import annotations

import argparse
import base64
import configparser
import contextlib
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time
import types
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "archive/FlashSeF-PTR_source_20261010"
DEFAULT_OUTPUT = ROOT / "runs/legacy_flashsef_ptr_finetune_20261010"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def repo_reference(path: Path) -> str:
    """Use a repository-relative identity when possible, otherwise preserve an absolute path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def read_hardware(config_path):
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    pe_rows, pe_cols = map(int, cfg.get("Tile level", "PE_Num").split(","))
    tile_rows, tile_cols = map(int, cfg.get("Architecture level", "Tile_Num").split(","))
    rows, cols = map(int, cfg.get("Crossbar level", "Xbar_Size").split(","))
    groups = cfg.getint("Process element level", "Group_Num")
    polarity = cfg.getint("Process element level", "Xbar_Polarity")
    # This bridge consumes the existing 9-bit, single-plane deployment artifacts.
    assert cfg.getint("Device level", "Device_Level") == 256
    assert cfg.getint("Algorithm Configuration", "Weight_Quantization_Bit") == 9
    assert groups > 0 and polarity == 2
    assert min(pe_rows, pe_cols, tile_rows, tile_cols, rows, cols) > 0
    return types.SimpleNamespace(dxb_num=groups, multi_bit=1, xbar_polarity=polarity,
                                 xbar_row=rows, xbar_column=cols,
                                 tile_row=pe_rows, tile_column=pe_cols,
                                 pe_slots_per_tile=pe_rows * pe_cols,
                                 active_tile_budget=tile_rows * tile_cols,
                                 chip_tile_row=tile_rows, chip_tile_column=tile_cols)


def install_transition_table_bridge(module, reference, config_path):
    """Supply a canonical device table to the unchanged uploaded optimizer.

    The uploaded loader only derives pulse-prefix costs from INI fields. ASU
    supplies an explicit state-transition table instead. Install the adapter
    before the optimizer imports the loader; retain its native table class and
    ns/nJ units, and restrict this process-local binding to the frozen config.
    """
    expected_path = Path(config_path).resolve()
    tables = module.PhysicalWriteTables(
        simconfig_path=str(expected_path), xbar_shape=reference.xbar_shape,
        ou_shape=reference.ou_shape, device_level=reference.device_level,
        latency_ns=reference.latency_ns.copy(), energy_nj=reference.energy_nj.copy(),
        cache_key=(str(expected_path), reference.table_digest), table_digest=reference.table_digest,
    )
    tables.latency_ns.flags.writeable = False
    tables.energy_nj.flags.writeable = False

    def load_frozen_tables(path):
        if Path(path).resolve() != expected_path:
            raise ValueError("legacy device table bridge received an unfrozen config")
        return tables

    module.load_physical_write_tables = load_frozen_tables
    return tables


def verify_source():
    checked = 0
    for line in (SOURCE / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        if sha(SOURCE / relative) != expected:
            raise ValueError(f"uploaded source checksum mismatch: {relative}")
        checked += 1
    return checked


def decode(value):
    if isinstance(value, list):
        return np.asarray(value, dtype=np.int32)
    if value["encoding"] != "zlib-base64-u8":
        raise ValueError("unsupported input code encoding")
    a = np.frombuffer(zlib.decompress(base64.b64decode(value["data"])), dtype=np.uint8)
    if a.size != value["length"]:
        raise ValueError("input code length mismatch")
    return a.astype(np.int32)


def round_payloads(round_record, plan, hw):
    """Expand exact output-major Seq code vectors into legacy physical PE payloads."""
    snapshot = round_record["placement_snapshot"]
    specs = snapshot["fragment_specs_by_id"]
    slots = snapshot["fragment_slots_by_id"]
    kernels = {node["layer_id"]: int(node["node_shape"].get("Kernelsize", 1))
               for item in plan["execution_items"] for node in item["local_graph_nodes"]}
    vectors = {int(k): decode(v) for k, v in round_record["target_states"].items()}
    cursors = dict.fromkeys(vectors, 0)
    payloads, members, used = {}, {}, set()
    for fid in sorted(specs):
        spec, slot = specs[fid], slots[fid]
        tile, pe, group = (int(slot[k]) for k in ("tile_id", "pe_id", "group_id"))
        if (tile, pe, group) in used:
            raise ValueError("duplicate input physical slot")
        used.add((tile, pe, group))
        assert 0 <= pe < hw.pe_slots_per_tile and 0 <= group < hw.dxb_num
        rows = (spec["input_range"][1] - spec["input_range"][0]) * kernels[spec["layer_id"]] ** 2
        cols = spec["output_range"][1] - spec["output_range"][0]
        assert 0 < rows <= hw.xbar_row and 0 < cols <= hw.xbar_column
        assert spec["polarity_count"] == hw.xbar_polarity and spec["weight_plane_index"] == 0
        pid = f"tile:{tile}/pe:{pe}"
        if pid not in payloads:
            payloads[pid] = np.zeros((hw.dxb_num, hw.multi_bit, hw.xbar_polarity,
                                      hw.xbar_row, hw.xbar_column), dtype=np.int32)
            members[pid] = []
        members[pid].append(fid)
        for polarity in range(hw.xbar_polarity):
            start = cursors[tile]
            end = start + rows * cols
            codes = vectors[tile][start:end]
            if codes.size != rows * cols:
                raise ValueError(f"truncated fragment codes: {fid}")
            payloads[pid][group, 0, polarity, :rows, :cols] = codes.reshape(cols, rows).T
            cursors[tile] = end
    if any(cursors[tile] != vector.size for tile, vector in vectors.items()):
        raise ValueError("unconsumed fragment codes")
    tiles = sorted({int(slots[f]["tile_id"]) for f in specs})
    assert tiles == list(range(len(tiles)))
    entries = []
    for pid in sorted(payloads):
        tile, pe = (int(part.split(":")[1]) for part in pid.split("/"))
        entries.append(types.SimpleNamespace(fragment_ids=[pid],
                                            payload_slot_start=tile * hw.pe_slots_per_tile + pe))
    return (types.SimpleNamespace(used_tile_count=len(tiles), layer_entries=entries),
            {pid: types.SimpleNamespace(payload=p) for pid, p in payloads.items()}, members)


def validate_input_bindings(model, inputs, artifact):
    manifest = json.loads((inputs / "deployment_weights_9bit_manifest.json").read_text())
    run = json.loads((inputs / "run_manifest.json").read_text())
    checkpoint = ROOT / "inputs/weights/finetune" / f"{model}_v7_phase6_software_qat_finetune_best.pth"
    assert sha(checkpoint) == manifest["source_checkpoint_sha256"]
    assert sha(inputs / "deployment_weights_9bit.pth") == manifest["deployment_checkpoint_sha256"]
    artifact_digest = hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert artifact_digest == run["artifacts"]["schedule_artifact_sha256"]
    assert run["request"]["network"] == f"{model}_v7_phase6"
    assert manifest["weight_bit_width"] == 9 and manifest["quantization_count"] == 1
    # Artifact identity hashes the checkpoint identity object, not raw file bytes.
    identity = {"checkpoint_sha256": manifest["source_checkpoint_sha256"]}
    candidate = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert candidate == artifact["weights_sha256"]


def validate_xbar(source, physical, positions, rev, cev, ou_shape, rng):
    """Check bijections, inverse weights, and routed integer MVM equivalence."""
    rr, cc = ou_shape
    nr, nc = source.shape[0] // rr, source.shape[1] // cc
    src_indices = positions[..., 0].astype(int) * nc + positions[..., 1]
    assert np.array_equal(np.sort(src_indices.ravel()), np.arange(nr * nc))
    assert np.array_equal(np.sort(rev, axis=-1), np.broadcast_to(np.arange(rr), rev.shape))
    assert np.array_equal(np.sort(cev, axis=-1), np.broadcast_to(np.arange(cc), cev.shape))
    physical_ous = physical.reshape(nr, rr, nc, cc).transpose(0, 2, 1, 3)
    inverse = np.take_along_axis(physical_ous, np.argsort(rev)[..., :, None], axis=2)
    inverse = np.take_along_axis(inverse, np.argsort(cev)[..., None, :], axis=3)
    recovered = np.empty_like(inverse)
    recovered[positions[..., 0], positions[..., 1]] = inverse
    recovered = recovered.transpose(0, 2, 1, 3).reshape(source.shape)
    assert np.array_equal(recovered, source), "RCSwap inverse changed canonical weights"
    inputs = rng.integers(-3, 4, size=source.shape[0], dtype=np.int64)
    routed = inputs[positions[..., 0, None].astype(int) * rr + rev]
    local_outputs = np.einsum("abr,abrc->abc", routed, physical_ous.astype(np.int64))
    output = np.zeros(source.shape[1], dtype=np.int64)
    np.add.at(output, (positions[..., 1, None].astype(int) * cc + cev).ravel(), local_outputs.ravel())
    assert np.array_equal(output, inputs @ source.astype(np.int64)), "routed MVM mismatch"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["lenet_snn", "alexnet_snn", "resnet34_snn", "vgg8_snn", "mobilenet_snn"])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-dir", type=Path, help="Directory containing a frozen Seq artifact and deployment weights")
    parser.add_argument("--sim-config", type=Path, default=ROOT / "runs/flashsef_ptr_finetune_20261010/SimConfig.schedule.ini")
    args = parser.parse_args()
    started = time.perf_counter()
    source_count = verify_source()
    model = args.model
    inputs = (args.input_dir or ROOT / "runs/seq_flashsef_finetune_eval_20261010" / model / "seq_schedule").resolve()
    input_manifest = json.loads((inputs / "run_manifest.json").read_text())
    input_path = Path(input_manifest["artifacts"]["schedule_artifact"])
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    artifact = json.loads(input_path.read_text())
    validate_input_bindings(model, inputs, artifact)
    plan = artifact["round_plan_snapshot"]
    output = args.output_root.resolve() / model
    output.mkdir(parents=True, exist_ok=True)
    # Keep concurrent invocations from interleaving the resident-state chain
    # or overwriting another process's incomplete per-round deployment files.
    generation_lock = (output / ".generation.lock").open("a")
    fcntl.flock(generation_lock.fileno(), fcntl.LOCK_EX)
    if (output / "flashsef_ptr.schedule.json").exists():
        raise FileExistsError("completed schedule already exists; choose a new output root")
    original_config = args.sim_config.resolve()
    if sha(original_config) != artifact["sim_config_sha256"]:
        raise ValueError("frozen simulator config identity mismatch")
    (output / "SimConfig.schedule.ini").write_bytes(original_config.read_bytes())
    cfg = configparser.ConfigParser()
    cfg.read(original_config)
    hw = read_hardware(original_config)
    cfg.set("Process element level", "Group_Num", str(hw.dxb_num * hw.xbar_polarity))
    cfg.set("Algorithm Configuration", "Weight_precision", "9")
    legacy_config = output / "SimConfig.legacy.ini"
    with legacy_config.open("w") as stream:
        cfg.write(stream)
    # Read authoritative current tables before switching the entire MNSIM namespace.
    sys.path.insert(0, str(ROOT))
    from MNSIM.Scheduling.phase2.physical_write_ledger import load_physical_write_tables as current_tables
    reference = current_tables(str(original_config))
    model_id = cfg.get("Final RRAM Model", "Model_ID", fallback=None)
    # Explicit transition costs take precedence over INI pulse-prefix fields.
    # Do not change or rewrite any of the uploaded source files.
    model_reference = cfg.get("Final RRAM Model", "Model_JSON", fallback=None)
    explicit_transition = False
    if model_reference:
        model_path = Path(model_reference)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        explicit_transition = isinstance(json.loads(model_path.read_text()).get("transition_cost"), dict)
    for key in list(sys.modules):
        if key == "MNSIM" or key.startswith("MNSIM."):
            del sys.modules[key]
    sys.path.insert(0, str(SOURCE / "legacy"))
    import_only_shim = False
    try:
        importlib.import_module("cupy")
    except ModuleNotFoundError:
        # Uploaded Tile_RCSwap imports CuPy but never calls its numerical API.
        # Any accidental numerical use remains an AttributeError, not an emulation.
        sys.modules["cupy"] = types.ModuleType("cupy")
        import_only_shim = True
    os.environ["MNSIM_FLASHSEF_TILE_SCHEDULER"] = "ptr"
    os.environ["MNSIM_XBAR_RCSWAP_ASSIGNMENT_MODE"] = "similarity"
    random.seed(20261010)
    np.random.seed(20261010)
    physical_module = importlib.import_module("MNSIM.Weight_update.PhysicalWriteCost")
    if explicit_transition:
        install_transition_table_bridge(physical_module, reference, legacy_config)
    from MNSIM.Weight_update.FlashSeF_Adapter import build_tile_transition_jobs
    from MNSIM.Weight_update.PhysicalWriteCost import load_physical_write_tables, score_tile_physical_transition
    import MNSIM.Weight_update.PE_RCSwap as pe_module
    import MNSIM.Weight_update.Tile_RCSwap as tile_module
    assert Path(tile_module.__file__).resolve().is_relative_to(SOURCE / "legacy")
    tables = load_physical_write_tables(str(legacy_config))
    np.testing.assert_allclose(tables.energy_nj, reference.energy_nj, rtol=1e-12, atol=0)
    np.testing.assert_allclose(tables.latency_ns, reference.latency_ns, rtol=1e-12, atol=0)
    assert tables.xbar_shape == (128, 128) and tables.ou_shape == (4, 4) and tables.device_level == 256
    assert tables.xbar_shape == (hw.xbar_row, hw.xbar_column)
    pe_captures, xbar_captures = [], []

    class CapturedXbar(pe_module.xbar_rcswap):
        def search_xbar_transform_exsuit_method_with_score(self):
            result = super().search_xbar_transform_exsuit_method_with_score()
            xbar_captures.append((np.asarray(self.ou_position, dtype=np.uint8).copy(),
                                  np.asarray(self.rev, dtype=np.uint8).copy(),
                                  np.asarray(self.cev, dtype=np.uint8).copy()))
            return result

    class CapturedPE(pe_module.PE_rcswap):
        def search_PE_transform_exsuit_method_multiprocess(self):
            # Existing original serial entrypoint preserves the numerical algorithm
            # while allowing observation of OU positions omitted by the old bridge.
            first = len(xbar_captures)
            result = super().search_PE_transform_exsuit_method()
            pe_captures.append((self.similarity_correspondence.copy(), xbar_captures[first:]))
            return result

    pe_module.xbar_rcswap = CapturedXbar
    tile_module.PE_rcswap = CapturedPE
    resident, rounds = {}, []
    rng = np.random.default_rng(20261010)
    xbar_checks = 0
    with (output / "generation.log").open("w", buffering=1) as kernel_log:
        for record in artifact["rounds"]:
            round_id = record["round_id"]
            round_spec, fragment_map, members = round_payloads(record, plan, hw)
            print(f"[{model}] {round_id}: assigning {round_spec.used_tile_count} tiles", flush=True)
            jobs = build_tile_transition_jobs(prev_round=None, curr_round=round_spec,
                                              fragment_map=fragment_map, hw=hw,
                                              resident_tile_state=resident,
                                              simconfig_path=str(legacy_config), criterion=5)
            assert len({j["tile_id"] for j in jobs}) == round_spec.used_tile_count
            arrays, reports, assignments = {}, [], []
            for index, job in enumerate(jobs):
                tile_id = job["tile_id"]
                assert 0 <= tile_id < hw.active_tile_budget
                expected_original = resident.get(tile_id, np.zeros_like(job["original_tile"]))
                assert np.array_equal(job["original_tile"], expected_original)
                pe_captures.clear()
                xbar_captures.clear()
                kernel = tile_module.Tile_rcswap(str(legacy_config), job["original_tile"], job["target_tile"], 5)
                with contextlib.redirect_stdout(kernel_log):
                    kernel.search_tile_transform_exsuit_method()
                optimized = kernel.reconstructed_target_tile.astype(np.uint8)
                pe_shape = (hw.tile_row, hw.tile_column, hw.dxb_num, hw.multi_bit, hw.xbar_polarity)
                ou_rows = hw.xbar_row // tables.ou_shape[0]
                ou_cols = hw.xbar_column // tables.ou_shape[1]
                correspondence = np.asarray([p[0] for p in pe_captures], dtype=np.uint8).reshape(*pe_shape, 2, 3)
                positions = np.asarray([x[0] for p in pe_captures for x in p[1]], dtype=np.uint8).reshape(*pe_shape, ou_rows, ou_cols, 2)
                rev = np.asarray([x[1] for p in pe_captures for x in p[1]], dtype=np.uint8).reshape(*pe_shape, ou_rows, ou_cols, tables.ou_shape[0])
                cev = np.asarray([x[2] for p in pe_captures for x in p[1]], dtype=np.uint8).reshape(rev.shape)
                for pr in range(hw.tile_row):
                    for pc in range(hw.tile_column):
                        corr = correspondence[pr, pc]
                        source_keys = []
                        for g, b, p in np.ndindex(hw.dxb_num, hw.multi_bit, hw.xbar_polarity):
                            dest, src = corr[g, b, p]
                            assert tuple(dest) == (g, b, p)
                            source_keys.append(tuple(src))
                            key = (pr, pc, g, b, p)
                            canonical = job["target_tile"][(pr, pc, *tuple(src))]
                            validate_xbar(canonical, optimized[key], positions[key], rev[key], cev[key], tables.ou_shape, rng)
                            xbar_checks += 1
                        assert set(source_keys) == set(np.ndindex(hw.dxb_num, hw.multi_bit, hw.xbar_polarity))
                latency, energy = score_tile_physical_transition(tables=tables, original_tile=job["original_tile"], target_tile=optimized)
                np.testing.assert_allclose([latency, energy],
                                           [kernel.totle_tile_insuit_latency * 1e-9, kernel.totle_tile_insuit_energy * 1e-9], rtol=1e-11, atol=1e-18)
                prefix = f"tile_{tile_id}"
                for name, a in [("original", job["original_tile"]), ("canonical_target", job["target_tile"]),
                                ("optimized", optimized), ("xbar_correspondence", correspondence),
                                ("ou_positions", positions), ("row_permutations", rev), ("column_permutations", cev)]:
                    arrays[f"{prefix}_{name}"] = np.asarray(a, dtype=np.uint8)
                resident[tile_id] = optimized.astype(np.int32)
                reports.append({"logical_tile_id": job["logical_tile_id"], "physical_tile_id": tile_id,
                                "array_prefix": prefix, "write_energy_j": energy, "write_latency_s": latency,
                                "assignment_metadata": job["payload_assignment_metadata"]})
                for assignment in job["payload_slot_assignments"]:
                    for fid in members[assignment["fragment_id"]]:
                        assignments.append({"fragment_id": fid,
                                            "canonical_slot": record["placement_snapshot"]["fragment_slots_by_id"][fid],
                                            "physical_tile_id": tile_id,
                                            "physical_pe_id": assignment["tile_local_slot"],
                                            "canonical_group_id": record["placement_snapshot"]["fragment_slots_by_id"][fid]["group_id"],
                                            "transform_array_prefix": prefix})
                print(f"[{model}] {round_id}: tile {index + 1}/{len(jobs)} validated", flush=True)
            expected_ids = set(record["placement_snapshot"]["fragment_specs_by_id"])
            assert len(assignments) == len(expected_ids) and {a["fragment_id"] for a in assignments} == expected_ids
            payload_path = output / f"{round_id}.deployment.npz"
            np.savez_compressed(payload_path, **arrays)
            with np.load(payload_path, allow_pickle=False) as loaded:
                assert set(loaded.files) == set(arrays)
                for key, a in arrays.items():
                    assert np.array_equal(loaded[key], a)
            rounds.append({"round_id": round_id, "round_index": record["round_index"],
                           "deployment_file": payload_path.name, "deployment_sha256": sha(payload_path),
                           "arrays": {k: {"shape": list(a.shape), "dtype": str(a.dtype)} for k, a in arrays.items()},
                           "tiles": reports, "fragment_assignments": assignments,
                           "placement_snapshot": record["placement_snapshot"]})
    final_source_count = verify_source()
    assert final_source_count == source_count
    checkpoint = ROOT / "inputs/weights/finetune" / f"{model}_v7_phase6_software_qat_finetune_best.pth"
    result = {"format": "mnsim.flashsef_ptr.legacy_deployment", "format_version": 1,
              "algorithm": "FlashSeF-PTR", "author_source_verified": True,
              "model_id": artifact["model_id"], "weights_sha256": artifact["weights_sha256"],
              "model_ir_sha256": artifact["model_ir_sha256"], "base_round_plan_sha256": artifact["base_round_plan_sha256"],
              "sim_config_sha256": artifact["sim_config_sha256"],
              "checkpoint": {"path": repo_reference(checkpoint), "sha256": sha(checkpoint)},
              "source": {"path": repo_reference(SOURCE), "sha256sums_sha256": sha(SOURCE / "SHA256SUMS"),
                         "verified_files": source_count, "provenance": json.loads((SOURCE / "PROVENANCE.json").read_text())},
              "input_seq_artifact": {"path": repo_reference(input_path), "sha256": sha(input_path)},
              "deployment_weights": {"path": repo_reference(inputs / "deployment_weights_9bit.pth"),
                                     "sha256": sha(inputs / "deployment_weights_9bit.pth")},
              "legacy_config_sha256": sha(legacy_config),
              "adaptations": {"Group_Num": f"current {hw.dxb_num} differential groups -> legacy {hw.dxb_num * hw.xbar_polarity} physical crossbars",
                              "Weight_precision": 9, "execution": "original_serial_PE_entrypoint",
                              "cupy_import_only_shim": import_only_shim,
                              "canonical_transition_table_bridge": explicit_transition},
              "device_model": {"model_id": model_id,
                               "model_json_sha256": sha(model_path) if model_reference else None,
                               "write_table_digest": reference.table_digest,
                               "cost_units": "legacy ns/nJ; outputs s/J"},
              "settings": {"tile_scheduler": "ptr", "xbar_assignment": "similarity", "criterion": 5,
                           "seed": 20261010, "initial_residency": "all_zero", "physical_tile_pool": hw.active_tile_budget,
                           "tile_array": [hw.chip_tile_row, hw.chip_tile_column],
                           "pe_array_per_tile": [hw.tile_row, hw.tile_column],
                           "physical_xbars": hw.active_tile_budget * hw.pe_slots_per_tile * hw.dxb_num * hw.xbar_polarity},
              "current_custom_schedule_directly_compatible": False,
              "compatibility_note": "Includes complete OU/xbar/row/column transforms; current tile-only custom adapter cannot execute it without a transform-aware adapter.",
              "round_plan_file": "round_plan.json", "rounds": rounds}
    (output / "round_plan.json").write_bytes((inputs / "round_plan.json").read_bytes())
    validation = {"passed": True, "physical_tables_match_current": True, "source_files_unchanged": True,
                  "fragment_coverage": "complete_each_round", "inverse_weight_checks": xbar_checks,
                  "routed_integer_mvm_checks": xbar_checks, "write_ledger_reaccount": "passed_all_tiles",
                  "resident_state_chain": "passed", "npz_roundtrip": "passed", "wall_s": time.perf_counter() - started,
                  "weighted_rounds": len(rounds), "tile_updates": sum(len(r["tiles"]) for r in rounds)}
    write_json(output / "validation.json", validation)
    write_json(output / "flashsef_ptr.schedule.json", result)
    print(json.dumps({"model": model, "schedule": str(output / "flashsef_ptr.schedule.json"), **validation}), flush=True)


if __name__ == "__main__":
    main()
