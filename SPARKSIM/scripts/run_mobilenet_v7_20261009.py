#!/usr/bin/env python3
"""Launch MobileNet V7 pretrain, then QAT, and archive evaluated best weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback


REPO = Path(__file__).resolve().parents[1]
MODEL = "mobilenet_snn_v7_phase6"
DATASET = REPO / "inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6"
RUN = REPO / "runs/mobilenet_snn_v7_phase6_retrain_20261009_b64_lr001_seed20261009"
SETTINGS = {
    "batch": 64, "seed": 20261009, "lif_tau": 50.5016666556, "lif_threshold": .01,
    "readout": "spike", "pretrain_optimizer": "sgd", "pretrain_lr": .01,
    "pretrain_epochs": 180, "pretrain_milestones": [80, 130],
    "finetune_lr": .001, "finetune_epochs": 20, "finetune_freeze_bn": False,
}


def write_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def commands() -> dict[str, list[str]]:
    shared = [
        "--model", MODEL, "--dataset-root", str(DATASET),
        "--sim-config", str(REPO / "SimConfig.ini"), "--device", "cuda:0",
        "--train-batch", str(SETTINGS["batch"]), "--test-batch", str(SETTINGS["batch"]), "--momentum", "0.9",
        "--weight-decay", "0.0001", "--gamma", "0.1", "--gradient-clip-value", "1.0",
        "--lif-tau", str(SETTINGS["lif_tau"]), "--lif-threshold", str(SETTINGS["lif_threshold"]),
        "--readout", SETTINGS["readout"], "--seed", str(SETTINGS["seed"]),
    ]
    return {
        "pretrain": [sys.executable, "-u", "scripts/train_snn_v7.py", *shared,
                     "--output-dir", str(RUN / "pretrain"), "--epochs", str(SETTINGS["pretrain_epochs"]),
                     "--optimizer", SETTINGS["pretrain_optimizer"],
                     "--learning-rate", str(SETTINGS["pretrain_lr"]), "--milestones",
                     *map(str, SETTINGS["pretrain_milestones"])],
        "finetune": [sys.executable, "-u", "scripts/finetune_snn_v7.py", *shared,
                     "--checkpoint", str(RUN / "pretrain" / f"{MODEL}_software_pretrain_best.pth"),
                     "--output-dir", str(RUN / "finetune"), "--epochs", str(SETTINGS["finetune_epochs"]),
                     "--learning-rate", str(SETTINGS["finetune_lr"]), "--milestones", "10", "15",
                     "--calibration-batches", "16", "--evaluate-initial",
                     *(["--freeze-bn"] if SETTINGS["finetune_freeze_bn"] else [])],
    }


def smoke_test() -> dict:
    """Check full-batch CUDA memory/gradients using only pretrain samples."""
    import torch
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from MNSIM.Interface.dvs_binned_npz import DVSBinnedNPZDataset
    from MNSIM.Interface.network import get_net
    from scripts.train_snn_v7 import _hardware_config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU training fallback")
    torch.manual_seed(SETTINGS["seed"])
    dataset = DVSBinnedNPZDataset(str(DATASET), "pretrain")
    # Repetition is only for a memory/gradient probe, never for training metrics.
    sample, label = dataset[0]
    images = sample.unsqueeze(0).repeat(SETTINGS["batch"], 1, 1, 1, 1).cuda()
    labels = torch.full((SETTINGS["batch"],), label, dtype=torch.long, device="cuda")
    net = get_net(_hardware_config(REPO / "SimConfig.ini"), cate=MODEL, num_classes=8).cuda()
    torch.cuda.reset_peak_memory_stats()
    output = net.sequence_accuracy_forward(images, weighted_mode="ORIGINAL", readout=SETTINGS["readout"])
    loss = torch.nn.functional.cross_entropy(output, labels)
    loss.backward()
    if not torch.isfinite(loss).item() or not all(
        torch.isfinite(p.grad).all().item() for p in net.parameters() if p.grad is not None
    ):
        raise RuntimeError("MobileNet produced nonfinite loss or gradients")
    if not any(p.grad is not None and torch.count_nonzero(p.grad).item()
               for p in net.parameters()):
        raise RuntimeError("MobileNet produced no nonzero gradients")
    torch.cuda.synchronize()
    result = {"input_shape": list(images.shape), "output_shape": list(output.shape),
              "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
              "probe_only": True, "source_split": "pretrain"}
    del loss, output, net, images, labels
    torch.cuda.empty_cache()
    return result


def archive(stage: str) -> dict:
    suffix = "software_pretrain" if stage == "pretrain" else "software_qat_finetune"
    history_path = RUN / stage / f"{MODEL}_{suffix}_history.json"
    report = json.loads(history_path.read_text())
    if report["test"] is None or report["test"]["limited_batches"] is not None:
        raise RuntimeError("Full test evaluation is required before archiving")
    if report["best_validation_accuracy"] <= 1 / 8:
        raise RuntimeError("Validation did not exceed chance; refusing to archive or proceed to QAT")
    best = RUN / stage / f"{MODEL}_{suffix}_best.pth"
    weights = REPO / "inputs/weights"
    formal = weights / stage
    formal.mkdir(parents=True, exist_ok=True)
    with (weights / ".qat-promotion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for source in (best, history_path):
            destination = formal / source.name
            if destination.exists():
                raise FileExistsError(f"Refusing to replace an existing official MobileNet file: {destination}")
        for source in (best, history_path):
            with tempfile.NamedTemporaryFile(dir=formal, delete=False) as stream:
                temporary = Path(stream.name)
            shutil.copy2(source, temporary)
            os.replace(temporary, formal / source.name)
        entries = [(str(path.relative_to(weights)), digest(path))
                   for directory in (weights / "pretrain", weights / "finetune")
                   for path in sorted(directory.glob("*_best.pth"))]
        with tempfile.NamedTemporaryFile(mode="w", dir=weights, delete=False) as stream:
            stream.write("".join(f"{checksum}  {name}\n" for name, checksum in sorted(entries)))
            temporary = Path(stream.name)
        os.replace(temporary, weights / "SHA256SUMS")
    return {"best_epoch": report["best_epoch"],
            "validation_accuracy": report["best_validation_accuracy"],
            "test_accuracy": report["test"]["accuracy"],
            "checkpoint": str(formal / best.name), "sha256": digest(formal / best.name)}


def worker() -> None:
    status = {"controller_pid": os.getpid(), "state": "smoke_test",
              "software_readout": SETTINGS["readout"], "hspice_executed": False}
    write_json(RUN / "status.json", status)
    status["smoke_test"] = smoke_test()
    print(f"CUDA smoke test passed: {status['smoke_test']}", flush=True)
    for stage, command in commands().items():
        phase = RUN / stage
        phase.mkdir()
        with (phase / "train.log").open("x") as log:
            process = subprocess.Popen(command, cwd=REPO, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            (phase / "train.pid").write_text(f"{process.pid}\n")
            status.update(state=stage, training_pid=process.pid, log=str(phase / "train.log"))
            write_json(RUN / "status.json", status)
            print(f"Started {stage}: PID {process.pid}, log {phase / 'train.log'}", flush=True)
            code = process.wait()
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {phase / 'train.log'}")
        status[stage] = archive(stage)
        print(f"Archived {stage}: {status[stage]}", flush=True)
    status.update(state="complete", training_pid=None)
    write_json(RUN / "status.json", status)


def main() -> None:
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help="JSON file containing run_dir and training parameter overrides")
    args = parser.parse_args()
    if args.config is not None:
        overrides = json.loads(args.config.read_text())
        unknown = set(overrides) - set(SETTINGS) - {"run_dir"}
        if unknown:
            raise ValueError(f"Unknown MobileNet settings: {sorted(unknown)}")
        RUN = REPO / overrides.pop("run_dir")
        SETTINGS.update(overrides)
    if args.worker:
        try:
            worker()
        except BaseException as error:
            status = json.loads((RUN / "status.json").read_text()) if (RUN / "status.json").exists() else {}
            status.update(state="failed", error=str(error))
            write_json(RUN / "status.json", status)
            traceback.print_exc()
            raise SystemExit(1)
        return
    if args.gpu < 0 or not DATASET.is_dir():
        raise ValueError("A valid GPU index and V7 dataset are required")
    RUN.mkdir(exist_ok=False)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="4", MNSIM_DVS_NUM_WORKERS="4",
               MNSIM_DVS_H="32", MNSIM_DVS_W="32", MNSIM_DVS_NUM_CLASSES="8",
               MNSIM_DVS_BINNED_ROOT=str(DATASET), MNSIM_SNN_LIF_V_RESET="0.0",
               MNSIM_SNN_LIF_ALPHA="2.0", MNSIM_SNN_LIF_TAU=str(SETTINGS["lif_tau"]),
               MNSIM_SNN_LIF_V_TH=str(SETTINGS["lif_threshold"]))
    for name in ("MNSIM_DVS_TEST_SAMPLE_LIMIT", "MNSIM_DVS_TEST_SAMPLES_PER_CLASS"):
        env.pop(name, None)
    with (RUN / "controller.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                                    *(["--config", str(args.config.resolve())] if args.config else [])],
                                   cwd=REPO, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    (RUN / "controller.pid").write_text(f"{process.pid}\n")
    write_json(RUN / "launch.json", {"utc_started": datetime.now(timezone.utc).isoformat(),
               "controller_pid": process.pid, "physical_gpu": args.gpu, "model": MODEL,
               "settings": SETTINGS,
               "commands": commands(), "environment": {k: env[k] for k in env if k.startswith("MNSIM_")
                                                          or k in {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"}}})
    print(json.dumps({"controller_pid": process.pid, "gpu": args.gpu, "run": str(RUN)}), flush=True)


if __name__ == "__main__":
    main()
