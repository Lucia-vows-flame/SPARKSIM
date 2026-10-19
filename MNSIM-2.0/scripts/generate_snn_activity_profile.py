#!/usr/bin/env python3
"""Build a frozen V7 SNN spike-transfer profile from a calibration split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MNSIM.Accuracy_Model.snn_accuracy_contract import canonical_json_sha256
from MNSIM.Interface.dvs_binned_npz import DVSBinnedNPZDataset
from MNSIM.Scheduling.weight_scheduling.artifact import load_schedule_artifact
from round_orchestrated_interface import RoundOrchestratedTrainTestInterface


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--sim-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    return parser


def _destination_channels(snapshot: dict) -> dict[int, set[int]]:
    channels: dict[int, set[int]] = {}
    specs = snapshot["fragment_specs_by_id"]
    slots = snapshot["fragment_slots_by_id"]
    for fragment_id, spec in specs.items():
        if spec.get("layer_id") != "L012_conv":
            continue
        tile_id = int(slots[fragment_id]["tile_id"])
        start, end = (int(value) for value in spec["input_range"])
        channels.setdefault(tile_id, set()).update(range(start, end))
    if not channels:
        raise RuntimeError("R1 placement contains no L012_conv fragments")
    return channels


def main() -> None:
    args = _parser().parse_args()
    if args.sample_count <= 0 or args.batch_size <= 0:
        raise SystemExit("sample-count and batch-size must be positive")
    if not args.weights.is_file() or not args.schedule.is_file():
        raise FileNotFoundError("weights and schedule must exist")

    schedule = load_schedule_artifact(args.schedule)
    if len(schedule.rounds) < 2:
        raise RuntimeError("the VGG8 profile requires at least R0 and R1")
    r0, r1 = schedule.rounds[0], schedule.rounds[1]
    r0_snapshot = r0.placement_snapshot or {}
    r1_snapshot = r1.placement_snapshot or {}
    owners = r0_snapshot.get("output_owner_shards_by_layer", {})
    source_owner = owners.get("L010_conv", {})
    if len(source_owner) != 1:
        raise RuntimeError("expected one R0 L010_conv output owner")
    source_tile_id = int(next(iter(source_owner)))
    destination_channels = _destination_channels(r1_snapshot)

    os.environ["MNSIM_DVS_BINNED_ROOT"] = str(args.dataset_root.resolve())
    dataset = DVSBinnedNPZDataset(str(args.dataset_root), "calibration")
    sample_count = min(args.sample_count, len(dataset))
    selected_entries = [
        {
            "relative_path": Path(path).relative_to(dataset.split_dir).as_posix(),
            "size_bytes": Path(path).stat().st_size,
        }
        for path in dataset.files[:sample_count]
    ]
    split_sha256 = canonical_json_sha256(selected_entries)
    loader = DataLoader(
        Subset(dataset, range(sample_count)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    interface = RoundOrchestratedTrainTestInterface(
        "vgg8_snn_v7_phase6",
        None,
        str(args.sim_config),
        weights_file=str(args.weights),
        device="cpu",
        num_classes=8,
    )
    net = interface.net.to(torch.device(args.device))
    net.eval()
    lif = net.layer_list[11].layer
    captures: list[torch.Tensor] = []
    original_forward_sequence = lif.forward_sequence

    def capture_forward_sequence(value):
        output = original_forward_sequence(value)
        captures.append(output.detach())
        return output

    lif.forward_sequence = capture_forward_sequence
    totals = {tile_id: 0.0 for tile_id in destination_channels}
    with torch.no_grad():
        for images, _labels in loader:
            captures.clear()
            net.sequence_accuracy_forward(
                images.to(args.device), weighted_mode="ORIGINAL"
            )
            if not captures:
                raise RuntimeError("L011_lif did not produce a sequence output")
            spikes = captures[-1].to("cpu")
            for tile_id, channel_ids in destination_channels.items():
                indices = sorted(channel_ids)
                totals[tile_id] += float(spikes[:, :, indices, :, :].sum().item())

    edges = [
        {
            "source_round_id": "R0",
            "source_tile_id": f"tile:{source_tile_id}",
            "destination_round_id": "R1",
            "destination_tile_id": f"tile:{tile_id}",
            "spike_events_per_sample": totals[tile_id] / sample_count,
        }
        for tile_id in sorted(destination_channels)
    ]
    if not any(edge["spike_events_per_sample"] > 0 for edge in edges):
        raise RuntimeError("calibration produced no positive L011_lif traffic")

    payload = {
        "schema_version": 1,
        "profile_id": f"v7-calibration-first-{sample_count}",
        "model_ir_sha256": schedule.model_ir_sha256,
        "round_plan_sha256": schedule.base_round_plan_sha256,
        "weights_sha256": _sha256(args.weights),
        "calibration": {
            "role": "calibration",
            "split_id": f"calibration:first-{sample_count}",
            "split_sha256": split_sha256,
            "sample_count": sample_count,
            "selection": "sorted calibration files, first N entries",
        },
        "timestep_count": int(schedule.round_plan_snapshot["sequence_spec"]["sequence_length"]),
        "spike_transfer_edges": edges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "edges": edges}, indent=2))


if __name__ == "__main__":
    main()
