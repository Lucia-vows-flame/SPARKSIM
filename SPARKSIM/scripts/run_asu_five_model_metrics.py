#!/usr/bin/env python3
"""Generate ASU Seq/original PTR schedules and compare five fixed-9 V7 SNNs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("lenet_snn", "alexnet_snn", "resnet34_snn", "vgg8_snn", "mobilenet_snn")
CONFIG = ROOT / "configs/hardware/SimConfig_16xbar_square_asu.ini"
RUN = ROOT / "runs/asu_five_model_16xbar_20261010"


def invoke(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    with log.open("w") as stream:
        result = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def evaluate(model):
    started = time.monotonic()
    output = RUN / "seq_schedules" / model
    schedule = output / "seq.schedule.json"
    common = ["-m", "MNSIM.Interface.round_orchestrated_cli", "--network", f"{model}_v7_phase6",
              "--weights", str(ROOT / "inputs/weights/finetune" / f"{model}_v7_phase6_software_qat_finetune_best.pth"),
              "--num-classes", "8", "--sim-config", str(CONFIG), "--device", "cpu", "--skip-accuracy"]
    print(f"[{model}] generating ASU Seq schedule", flush=True)
    invoke([*common, "--run-hardware", "--schedule-only", "--schedule-policy", "seq", "--output-dir", str(output),
            "--overwrite",
            "--schedule-plan-out", str(schedule)], RUN / "logs" / f"{model}.schedule.log")
    print(f"[{model}] running native ASU Phase 2 and uploaded original PTR in parallel", flush=True)
    with ThreadPoolExecutor(max_workers=2) as stages:
        native = stages.submit(invoke,
            [*common, "--infer-with-schedule", "--schedule-plan-in", str(schedule), "--run-hardware",
             "--output-dir", str(RUN / "evaluation" / model / "seq")],
            RUN / "logs" / f"{model}.seq.log")
        ptr = stages.submit(invoke,
            ["scripts/generate_legacy_flashsef_ptr.py", model, "--input-dir", str(output),
             "--sim-config", str(CONFIG), "--output-root", str(RUN / "ptr_schedules")],
            RUN / "logs" / f"{model}.ptr.log")
        native.result()
        ptr.result()
    print(f"[{model}] complete ({time.monotonic()-started:.1f}s)", flush=True)


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RUN,
                        help="Use a fresh directory; completed PTR deployments are never overwritten")
    RUN = parser.parse_args().output_root.resolve()
    config = configparser.ConfigParser()
    config.read(CONFIG)
    assert config.get("Final RRAM Model", "Model_ID") == "rram_stanford_asu_256"
    assert config.get("Algorithm Configuration", "Weight_Quantization_Bit") == "9"
    assert config.get("Tile level", "PE_Num") == "1,1"
    assert config.get("Architecture level", "Tile_Num") == "2,2"
    RUN.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    failures = {}
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        futures = {pool.submit(evaluate, model): model for model in MODELS}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures[futures[future]] = str(error)
                print(f"[{futures[future]}] FAILED: {error}", flush=True)
    if failures:
        (RUN / "failures.json").write_text(json.dumps(failures, indent=2)+"\n")
        raise RuntimeError(f"failed models: {failures}")
    invoke(["scripts/compare_legacy_flashsef_ptr.py", "--schedule-root", str(RUN / "ptr_schedules"),
            "--evaluation-root", str(RUN / "evaluation"), "--models", *MODELS],
           RUN / "logs/comparison.log")
    print(f"All five ASU comparisons completed in {time.monotonic()-started:.1f}s: "
          f"{RUN / 'evaluation/comparison.csv'}", flush=True)


if __name__ == "__main__":
    main()
