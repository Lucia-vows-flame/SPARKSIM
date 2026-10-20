#!/usr/bin/env python3
"""Validate a saved complete legacy deployment without running its optimizer."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import sys

from generate_legacy_flashsef_ptr import (
    ROOT, read_hardware, round_payloads, sha, validate_input_bindings, validate_xbar,
    verify_source, write_json,
)
sys.path.insert(0, str(ROOT))
from MNSIM.Scheduling.depthwise import is_depthwise


def verify(schedule_path):
    schedule_path = Path(schedule_path).resolve()
    directory = schedule_path.parent
    result = json.loads(schedule_path.read_text())
    assert result["format"] == "mnsim.flashsef_ptr.legacy_deployment"
    model = result["model_id"].removesuffix("_v7_phase6")
    seq_path = ROOT / result["input_seq_artifact"]["path"]
    assert sha(seq_path) == result["input_seq_artifact"]["sha256"]
    seq = json.loads(seq_path.read_text())
    validate_input_bindings(model, seq_path.parent, seq)
    for field in ["model_id", "weights_sha256", "model_ir_sha256", "base_round_plan_sha256", "sim_config_sha256"]:
        assert result[field] == seq[field]
    assert sha(directory / "SimConfig.schedule.ini") == result["sim_config_sha256"]
    assert sha(directory / "SimConfig.legacy.ini") == result["legacy_config_sha256"]
    assert sha(ROOT / result["checkpoint"]["path"]) == result["checkpoint"]["sha256"]
    assert sha(ROOT / result["deployment_weights"]["path"]) == result["deployment_weights"]["sha256"]
    assert verify_source() == result["source"]["verified_files"]
    assert len(seq["rounds"]) == len(result["rounds"])
    hw = read_hardware(directory / "SimConfig.schedule.ini")
    resident, checks = {}, 0
    weight_checks = 0
    deployment_state = torch.load(ROOT / result["deployment_weights"]["path"], map_location="cpu", weights_only=True)
    quantization_manifest = json.loads((seq_path.parent / "deployment_weights_9bit_manifest.json").read_text())
    scales = {layer["layer_id"]: layer["lsb"] for layer in quantization_manifest["layers"]}
    rng = np.random.default_rng(20261010)
    shapes = {n["layer_id"]: n["node_shape"]
              for item in seq["round_plan_snapshot"]["execution_items"] for n in item["local_graph_nodes"]}
    for source_round, saved in zip(seq["rounds"], result["rounds"]):
        assert source_round["round_id"] == saved["round_id"]
        assert source_round["placement_snapshot"] == saved["placement_snapshot"]
        _, payloads, members = round_payloads(source_round, seq["round_plan_snapshot"], hw)
        for pid, fragment_ids in members.items():
            for fid in fragment_ids:
                spec = source_round["placement_snapshot"]["fragment_specs_by_id"][fid]
                slot = source_round["placement_snapshot"]["fragment_slots_by_id"][fid]
                layer_prefix = f"layer_list.{int(spec['layer_id'][1:].split('_')[0])}"
                shape = shapes[spec["layer_id"]]
                depthwise = is_depthwise(shape)
                input_block = 0 if depthwise else spec['input_block_index']
                key = f"{layer_prefix}.sublayer_list.{input_block}.weight"
                # Recover already-quantized integer codes independently from the
                # deployment checkpoint and its saved LSB; never quantize anew.
                signed = torch.round(deployment_state[key] / scales[layer_prefix]).to(torch.int32).numpy()
                lo, hi = spec["output_range"]
                if depthwise:
                    start, end = spec["input_range"]
                    assert (start, end) == (lo, hi)
                    area = int(shape['Kernelsize']) ** 2
                    kernels = signed[lo:hi].reshape(hi - lo, area)
                    diagonal = np.zeros(((end - start) * area, hi - lo), dtype=np.int32)
                    for local_channel, kernel in enumerate(kernels):
                        diagonal[local_channel * area:(local_channel + 1) * area, local_channel] = kernel
                    signed = diagonal
                else:
                    signed = signed[lo:hi].reshape(hi - lo, -1).T
                for polarity in range(hw.xbar_polarity):
                    expected = np.zeros((hw.xbar_row, hw.xbar_column), dtype=np.int32)
                    expected[:signed.shape[0], :signed.shape[1]] = np.maximum(signed if polarity == 0 else -signed, 0)
                    assert np.array_equal(expected, payloads[pid].payload[slot["group_id"], 0, polarity]), "Seq codes differ from deployment checkpoint"
                    weight_checks += 1
        payload_path = directory / saved["deployment_file"]
        assert sha(payload_path) == saved["deployment_sha256"]
        assigned = {a["fragment_id"]: a for a in saved["fragment_assignments"]}
        assert len(assigned) == len(saved["fragment_assignments"])
        assert set(assigned) == set(source_round["placement_snapshot"]["fragment_specs_by_id"])
        assert len({t["physical_tile_id"] for t in saved["tiles"]}) == len(saved["tiles"])
        with np.load(payload_path, allow_pickle=False) as arrays:
            assert set(arrays.files) == set(saved["arrays"])
            for key, spec in saved["arrays"].items():
                assert list(arrays[key].shape) == spec["shape"] and str(arrays[key].dtype) == spec["dtype"]
            for tile in saved["tiles"]:
                tid, logical = tile["physical_tile_id"], tile["logical_tile_id"]
                assert 0 <= tid < hw.active_tile_budget
                prefix = tile["array_prefix"]
                original, canonical, optimized = [arrays[f"{prefix}_{n}"] for n in ["original", "canonical_target", "optimized"]]
                assert np.array_equal(original, resident.get(tid, np.zeros_like(original)))
                expected = original.copy()
                for pid, payload in payloads.items():
                    tile_id, pe_id = [int(part.split(":")[1]) for part in pid.split("/")]
                    if tile_id == logical:
                        expected[pe_id // hw.tile_column, pe_id % hw.tile_column] = payload.payload
                        for fid in members[pid]:
                            a = assigned[fid]
                            assert a["physical_tile_id"] == tid and a["physical_pe_id"] == pe_id
                            assert a["canonical_slot"] == source_round["placement_snapshot"]["fragment_slots_by_id"][fid]
                assert np.array_equal(expected, canonical), "saved deployment does not match input weights"
                corr = arrays[f"{prefix}_xbar_correspondence"]
                for pr, pc in np.ndindex(hw.tile_row, hw.tile_column):
                    source_keys = []
                    for g, b, p in np.ndindex(hw.dxb_num, hw.multi_bit, hw.xbar_polarity):
                        key = (pr, pc, g, b, p)
                        dest, src = corr[key]
                        assert tuple(dest) == (g, b, p)
                        source_keys.append(tuple(src))
                        validate_xbar(canonical[(pr, pc, *tuple(src))], optimized[key],
                                      arrays[f"{prefix}_ou_positions"][key],
                                      arrays[f"{prefix}_row_permutations"][key],
                                      arrays[f"{prefix}_column_permutations"][key], (4, 4), rng)
                        checks += 1
                    assert set(source_keys) == set(np.ndindex(hw.dxb_num, hw.multi_bit, hw.xbar_polarity))
                resident[tid] = optimized.copy()
    validation_path = directory / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation.update({"checkpoint_and_input_bindings": "passed", "saved_deployment_verification": "passed",
                       "deployment_checkpoint_code_checks": weight_checks,
                       "saved_routed_mvm_checks": checks, "generator_sha256": sha(ROOT / "scripts/generate_legacy_flashsef_ptr.py")})
    write_json(validation_path, validation)
    print(json.dumps({"model": model, "passed": True, "saved_xbars_verified": checks}))
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    verify(parser.parse_args().schedule)
