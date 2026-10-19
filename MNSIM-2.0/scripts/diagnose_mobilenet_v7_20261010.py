"""MobileNet pulse/gradient diagnostics and small pretrain-only parameter probes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from MNSIM.Interface.network import get_net
from MNSIM.Interface.snn_v7_training_data import training_loaders
from scripts.train_snn_v7 import _hardware_config

OUT = REPO / "runs/mobilenet_diagnostic_20261010"
MODEL = "mobilenet_snn_v7_phase6"
OLD = REPO / f"runs/mobilenet_snn_v7_phase6_retrain_20261009_b64_lr001_seed20261009/pretrain/{MODEL}_software_pretrain_latest.pth"


def main():
    adam_probe = "--adam-probe" in sys.argv
    OUT.mkdir(exist_ok=True)
    loader, _, _, identities = training_loaders(
        REPO / "inputs/datasets/snn_binned_4split_2000perclass_v7_fullres_t30_phase6",
        "pretrain", 16, 16,
    )
    torch.manual_seed(20261010)
    batches = []
    for x, y in loader:
        batches.append((x.cuda(), y.cuda()))
        if len(batches) == 8:
            break
    cases = [
        ("old_checkpoint", 50.5016666556, .01, "spike", .01, True),
        ("fresh_old", 50.5016666556, .01, "spike", .01, False),
        ("tau2_spike_lr001", 2., .5, "spike", .001, False),
        ("tau2_spike_lr01", 2., .5, "spike", .01, False),
        ("tau2_current_lr01", 2., .5, "membrane", .01, False),
        ("tau2_current_lr001", 2., .5, "membrane", .001, False),
    ]
    if adam_probe:
        cases = [("tau2_current_adam001", 2., .5, "membrane", .001, False),
                 ("tau2_spike_adam0003", 2., .5, "spike", .0003, False)]
    results = []
    for name, tau, threshold, readout, lr, old in cases:
        os.environ.update(MNSIM_SNN_LIF_TAU=str(tau), MNSIM_SNN_LIF_V_TH=str(threshold))
        torch.manual_seed(20261010)
        net = get_net(_hardware_config(REPO / "SimConfig.ini"), cate=MODEL, num_classes=8).cuda()
        if old:
            net.load_state_dict(torch.load(OLD, map_location="cpu", weights_only=True)["model"])
        pulses = []
        capture = True
        for index, layer in enumerate(net.layer_list):
            if layer.layer_config["type"] == "lif":
                original = layer.layer.forward_sequence
                def instrument(x, original=original, index=index):
                    value = original(x)
                    if not capture:
                        return value
                    pulses.append({"index": index, "rate": value.detach().float().mean().item(),
                                   "current_mean": x.detach().mean().item(),
                                   "current_min": x.detach().min().item(),
                                   "current_max": x.detach().max().item()})
                    return value
                layer.layer.forward_sequence = instrument
        net.eval()
        x, y = batches[0]
        output = net.sequence_accuracy_forward(x, weighted_mode="ORIGINAL", readout=readout)
        loss = torch.nn.functional.cross_entropy(output, y)
        def gradients():
            return [{"name": key, "norm": p.grad.norm().item(), "max": p.grad.abs().max().item()}
                    for key, p in net.named_parameters() if p.grad is not None]
        record = {"name": name, "tau": tau, "threshold": threshold, "readout": readout,
                  "lr": lr, "initial_eval_loss": loss.item(), "initial_eval_logits": output.detach().cpu().tolist(),
                  "initial_eval_pulses": pulses.copy()}
        optimizer = (torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
                     if adam_probe else torch.optim.SGD(net.parameters(), lr=lr, momentum=.9, weight_decay=1e-4))
        record["steps"] = []
        net.train()
        steps = 256 if adam_probe else (48 if not old else 1)
        for step in range(steps):
            capture = step in {0, 15, 47, 127, steps - 1}
            pulses.clear()
            optimizer.zero_grad(set_to_none=True)
            x, y = batches[step % len(batches)]
            output = net.sequence_accuracy_forward(x, weighted_mode="ORIGINAL", readout=readout)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            grad = gradients()
            torch.nn.utils.clip_grad_value_(net.parameters(), 1.)
            optimizer.step()
            row = {"step": step + 1, "loss": loss.item(),
                   "accuracy": (output.argmax(1) == y).float().mean().item(),
                   "logit_min": output.min().item(), "logit_max": output.max().item(),
                   "first_gradient_norm": grad[0]["norm"], "last_gradient_norm": grad[-1]["norm"]}
            record["steps"].append(row)
            if capture:
                print(name, json.dumps(row), flush=True)
                record[f"step{step + 1}_pulses"] = pulses.copy()
        record["final_gradients"] = grad
        results.append(record)
        filename = "learning_probe.json" if adam_probe else "diagnostic.json"
        (OUT / filename).write_text(json.dumps({"splits": identities, "results": results,
                                               "optimizer": "adamw" if adam_probe else "sgd",
                                               "probe_only": True}, indent=2) + "\n")
        del net, optimizer, output, loss
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
