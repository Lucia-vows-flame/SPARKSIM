"""Finalize the user-stopped MobileNet pretrain and start background QAT."""
from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import run_mobilenet_v7_20261009 as pipeline


CONFIG = pipeline.REPO / "configs/training/mobilenet_v7_retrain_20261010.json"


def worker():
    import torch
    sys.path.insert(0, str(pipeline.REPO))
    from MNSIM.Interface.network import get_net
    from MNSIM.Interface.snn_v7_training_data import training_loaders
    from MNSIM.Interface.train_snn_v7 import evaluate
    from scripts.train_snn_v7 import _hardware_config, _forward_with_readout

    run = pipeline.RUN
    status = json.loads((run / "status.json").read_text())
    status.update(controller_pid=os.getpid(), training_pid=None, state="pretrain_final_evaluation",
                  pretrain_stop_reason="user_requested_accuracy_sufficient")
    pipeline.write_json(run / "status.json", status)
    path = run / "pretrain" / f"{pipeline.MODEL}_software_pretrain_history.json"
    report = json.loads(path.read_text())
    best = run / "pretrain" / f"{pipeline.MODEL}_software_pretrain_best.pth"
    torch.manual_seed(pipeline.SETTINGS["seed"])
    _, _, test, identities = training_loaders(pipeline.DATASET, "pretrain", 64, 64)
    if identities != report["run_config"]["dataset_splits"]:
        raise RuntimeError("Dataset identities differ from the pretrain report")
    net = get_net(_hardware_config(pipeline.REPO / "SimConfig.ini"),
                  cate=pipeline.MODEL, num_classes=8).cuda()
    net.load_state_dict(torch.load(best, map_location="cpu", weights_only=True), strict=True)
    loss, accuracy = evaluate(net, test, torch.device("cuda:0"),
                              forward_fn=partial(_forward_with_readout, readout="membrane"))
    report.update(stop_reason="user_requested_accuracy_sufficient",
                  test={"loss": loss, "accuracy": accuracy, "limited_batches": None})
    pipeline.write_json(path, report)
    status["pretrain"] = pipeline.archive("pretrain")
    print(f"Pretrain best epoch {report['best_epoch']}: validation {report['best_validation_accuracy']:.6f}, "
          f"full test {accuracy:.6f}", flush=True)
    del net
    torch.cuda.empty_cache()
    phase = run / "finetune"
    phase.mkdir(exist_ok=False)
    command = pipeline.commands()["finetune"]
    pipeline.write_json(phase / "launch.json", {"command": command, "physical_gpu": 1,
                                               "source_checkpoint_sha256": pipeline.digest(best)})
    with (phase / "train.log").open("x") as log:
        process = subprocess.Popen(command, cwd=pipeline.REPO, stdout=log,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        (phase / "train.pid").write_text(f"{process.pid}\n")
        status.update(state="finetune", training_pid=process.pid, log=str(phase / "train.log"))
        pipeline.write_json(run / "status.json", status)
        print(f"Started MobileNet QAT PID {process.pid}: {phase / 'train.log'}", flush=True)
        code = process.wait()
    if code:
        raise RuntimeError(f"MobileNet QAT exited with code {code}; see {phase / 'train.log'}")
    status["finetune"] = pipeline.archive("finetune")
    status.update(state="complete", training_pid=None)
    pipeline.write_json(run / "status.json", status)
    print(json.dumps(status["finetune"]), flush=True)


def main():
    settings = json.loads(CONFIG.read_text())
    pipeline.RUN = pipeline.REPO / settings.pop("run_dir")
    pipeline.SETTINGS.update(settings)
    if "--worker" in sys.argv:
        try:
            worker()
        except BaseException as error:
            status = json.loads((pipeline.RUN / "status.json").read_text())
            status.update(state="failed", error=str(error))
            pipeline.write_json(pipeline.RUN / "status.json", status)
            traceback.print_exc()
            raise SystemExit(1)
        return
    status = json.loads((pipeline.RUN / "status.json").read_text())
    for pid in (status["controller_pid"], status.get("training_pid")):
        if pid is None:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError(f"Previous MobileNet process {pid} is still alive")
    env = dict(os.environ)
    env.update(json.loads((pipeline.RUN / "launch.json").read_text())["environment"])
    for name in ("MNSIM_DVS_TEST_SAMPLE_LIMIT", "MNSIM_DVS_TEST_SAMPLES_PER_CLASS"):
        env.pop(name, None)
    with (pipeline.RUN / "finetune_controller.log").open("x") as log:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"],
                                   cwd=pipeline.REPO, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (pipeline.RUN / "controller.pid").write_text(f"{process.pid}\n")
    print(f"MobileNet pretrain evaluation / QAT controller PID {process.pid}", flush=True)


if __name__ == "__main__":
    main()
