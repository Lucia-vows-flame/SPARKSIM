#!/usr/bin/env python3
"""Run only ResNet18 QAT and archive its best weight after full-test success."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "runs/resnet18_snn_v7_phase6_qat_retrain_20261009_b128_tau2_th05"
MODEL = "resnet18_snn_v7_phase6"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / "train.log").exists():
        raise FileExistsError(f"refusing to overwrite existing training: {RUN}")
    checkpoint = REPO / f"inputs/weights/pretrain/{MODEL}_software_pretrain_best.pth"
    command = [
        sys.executable, "-u", "scripts/finetune_snn_v7.py",
        "--model", MODEL, "--checkpoint", str(checkpoint),
        "--dataset-root", "inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6",
        "--sim-config", "SimConfig.ini", "--output-dir", str(RUN), "--device", "cuda:0",
        "--epochs", "20", "--train-batch", "128", "--test-batch", "128",
        "--learning-rate", "0.0001", "--momentum", "0.9", "--weight-decay", "0.00001",
        "--milestones", "8", "14", "--gamma", "0.1", "--gradient-clip-value", "0.5",
        "--lif-tau", "2.0", "--lif-threshold", "0.5", "--readout", "membrane",
        "--freeze-bn", "--calibration-batches", "16", "--evaluate-initial",
        "--target-validation-accuracy", "0.95", "--min-epochs", "1", "--seed", "20261009",
    ]
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS="4", MNSIM_DVS_NUM_WORKERS="8",
               MNSIM_SNN_LIF_V_RESET="0.0", MNSIM_SNN_LIF_ALPHA="2.0")
    with (RUN / "train.log").open("w") as log:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        (RUN / "train.pid").write_text(f"{process.pid}\n")
        (RUN / "launch.json").write_text(json.dumps({
            "controller_pid": os.getpid(), "training_pid": process.pid,
            "command": command, "cwd": str(REPO), "physical_gpu": 0,
            "environment": {k: env[k] for k in ["CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS",
                                                  "MNSIM_DVS_NUM_WORKERS", "MNSIM_SNN_LIF_V_RESET",
                                                  "MNSIM_SNN_LIF_ALPHA"]},
            "source_checkpoint_sha256": digest(checkpoint),
            "acceptance": "full held-out test accuracy > 0.90; validation selects checkpoint",
        }, indent=2) + "\n")
        print(f"ResNet18 training PID {process.pid}; log: {RUN / 'train.log'}", flush=True)
        exit_code = process.wait()
    if exit_code:
        raise SystemExit(f"ResNet18 training failed with exit code {exit_code}")
    history_path = RUN / f"{MODEL}_software_qat_finetune_history.json"
    report = json.loads(history_path.read_text())
    if report["test"]["limited_batches"] is not None:
        raise RuntimeError("acceptance requires the full test split")
    if report["test"]["accuracy"] <= 0.90:
        (RUN / "acceptance.json").write_text(json.dumps({"passed": False,
                                                        "test": report["test"]}, indent=2) + "\n")
        raise SystemExit("test accuracy did not exceed 90%; weight was not promoted")
    best_path = RUN / f"{MODEL}_software_qat_finetune_best.pth"
    formal = REPO / "inputs/weights/finetune"
    formal.mkdir(parents=True, exist_ok=True)
    destination = formal / best_path.name
    if destination.exists():
        # Keep any older formal result within this run before replacing it.
        shutil.copy2(destination, RUN / "previous_formal_best.pth")
    with tempfile.NamedTemporaryFile(dir=formal, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    shutil.copy2(best_path, temporary_path)
    os.replace(temporary_path, destination)
    shutil.copy2(history_path, formal / history_path.name)
    weights_root = formal.parent
    sums = {str(p.relative_to(weights_root)): digest(p)
            for p in sorted(weights_root.rglob("*.pth"))}
    with tempfile.NamedTemporaryFile(mode="w", dir=weights_root, delete=False) as temporary:
        temporary.write("".join(f"{h}  {name}\n" for name, h in sorted(sums.items())))
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, weights_root / "SHA256SUMS")
    acceptance = {"passed": True, "model": MODEL, "best_epoch": report["best_epoch"],
                  "completed_epochs": report["completed_epochs"],
                  "validation_accuracy": report["best_validation_accuracy"],
                  "test_accuracy": report["test"]["accuracy"],
                  "formal_best_checkpoint": str(destination), "sha256": digest(destination),
                  "readout": "temporal classifier current mean (terminal LIF bypass)",
                  "software_qat_only": True, "hspice_executed": False}
    (RUN / "acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    print(json.dumps(acceptance, indent=2), flush=True)


if __name__ == "__main__":
    main()
