"""Software-pretrain loop for the V7 sequential SNN models.

The legacy ``Interface.train`` module trains the original ANN graph and calls
``NetworkGraph.forward`` on a four-dimensional tensor.  V7 samples are
temporal tensors, so this module deliberately uses the sequence graph and its
SpikingJelly LIF nodes for BPTT.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


def _device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        index = resolved.index if resolved.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} is unavailable; visible device count is "
                f"{torch.cuda.device_count()}"
            )
    return resolved


def _forward_sequence(net, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"expected V7 batch [B,T,C,H,W], got {tuple(images.shape)}")
    if getattr(net, "network_module", "").startswith("resnet"):
        return net.sequence_accuracy_forward(
            images, weighted_mode="ORIGINAL", readout="membrane"
        )
    return net.sequence_accuracy_forward(images, weighted_mode="ORIGINAL")


def evaluate(
    net,
    loader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    forward_fn=_forward_sequence,
):
    net.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = forward_fn(net, images)
            total_loss += float(criterion(outputs, labels).item()) * labels.size(0)
            total_correct += int((outputs.argmax(dim=1) == labels).sum().item())
            total += labels.size(0)
    if total == 0:
        raise RuntimeError("evaluation loader produced no samples")
    return total_loss / total, total_correct / total


def _atomic_save(value, path: Path, *, json_format: bool = False) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if json_format:
                with open(stream.fileno(), "w", encoding="utf-8", closefd=False) as output:
                    json.dump(value, output, indent=2, ensure_ascii=False)
                    output.write("\n")
                    output.flush()
            else:
                torch.save(value, stream)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def train_net(
    net,
    train_loader,
    validation_loader,
    device: str | torch.device,
    prefix: str,
    *,
    epochs: int = 60,
    learning_rate: float = 0.01,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    milestones: Iterable[int] = (30, 60),
    gamma: float = 0.06,
    gradient_clip_value: float | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
    forward_fn=_forward_sequence,
    checkpoint_suffix: str = "_software_pretrain_best.pth",
    test_loader=None,
    run_config: dict | None = None,
    resume: bool = False,
    overwrite: bool = False,
    freeze_batchnorm: bool = False,
    evaluate_initial: bool = False,
    target_validation_accuracy: float | None = None,
    min_epochs: int = 1,
):
    """Select on validation, then assess the selected checkpoint on test once."""
    if test_loader is None:
        raise ValueError("a separate test_loader is required")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    if int(epochs) < 1:
        raise ValueError("epochs must be positive")
    if target_validation_accuracy is not None and not 0 < target_validation_accuracy <= 1:
        raise ValueError("target_validation_accuracy must be in (0, 1]")
    if not 1 <= int(min_epochs) <= int(epochs):
        raise ValueError("min_epochs must be between 1 and epochs")
    if gradient_clip_value is not None and (
        not torch.isfinite(torch.tensor(float(gradient_clip_value)))
        or float(gradient_clip_value) <= 0
    ):
        raise ValueError("gradient_clip_value must be finite and positive")
    milestones = list(milestones)
    device = _device(device)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"unsupported optimizer: {optimizer_name}")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=gamma
    )
    destination = Path(output_dir or (Path(__file__).resolve().parent / "zoo"))
    destination.mkdir(parents=True, exist_ok=True)
    best_accuracy = float("-inf")
    best_epoch = 0
    best_path = destination / f"{prefix}{checkpoint_suffix}"
    latest_path = destination / f"{prefix}{checkpoint_suffix.replace('_best.pth', '_latest.pth')}"
    history_path = destination / f"{prefix}{checkpoint_suffix.replace('_best.pth', '_history.json')}"
    if not checkpoint_suffix.endswith("_best.pth"):
        raise ValueError("checkpoint_suffix must end in _best.pth")

    config = dict(run_config or {})
    # Keep legacy SGD configurations identical for existing resumable runs.
    if optimizer_name != "sgd":
        config.update(optimizer=optimizer_name, optimizer_betas=[0.9, 0.999], optimizer_eps=1e-8)
    config.update({
        "model": prefix,
        "checkpoint_suffix": checkpoint_suffix,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "milestones": milestones,
        "gamma": gamma,
        "gradient_clip_value": gradient_clip_value,
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
    })
    if freeze_batchnorm:
        config["freeze_batchnorm_running_stats"] = True
    if evaluate_initial:
        config["evaluate_initial_checkpoint"] = True
    if target_validation_accuracy is not None:
        config["target_validation_accuracy"] = target_validation_accuracy
        config["min_epochs"] = min_epochs
    existing = [path for path in (best_path, latest_path, history_path) if path.exists()]
    if existing and not (resume or overwrite):
        raise FileExistsError(
            f"run outputs already exist: {existing}; pass --resume or --overwrite explicitly"
        )

    history = []
    initial_validation = None
    stop_reason = "maximum_epochs"
    start_epoch = 0
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"resume requires {latest_path}")
        state = torch.load(latest_path, map_location="cpu", weights_only=True)
        if state["run_config"] != config:
            raise ValueError("resume configuration differs from the saved run")
        net.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best_accuracy = float(state["best_validation_accuracy"])
        history = list(state["history"])
        initial_validation = state.get("initial_validation")
        if "best_epoch" in state:
            best_epoch = int(state["best_epoch"])
        else:
            best_epoch = next(
                row["epoch"] for row in history
                if row["validation_accuracy"] == best_accuracy
            )
        start_epoch = int(state["epoch"])
        if start_epoch >= int(epochs):
            raise ValueError(f"saved run already reached epoch {start_epoch}")
        if best_epoch == start_epoch:
            _atomic_save(state["model"], best_path)
        elif not best_path.is_file():
            raise FileNotFoundError(f"resume requires previous best checkpoint {best_path}")
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda":
            if "cuda_rng_state" in state:
                torch.cuda.set_rng_state(state["cuda_rng_state"], device)
            else:
                torch.cuda.set_rng_state_all(state["cuda_rng_states"])

    if evaluate_initial and not resume:
        initial_loss, best_accuracy = evaluate(
            net, validation_loader, device, max_batches=max_eval_batches,
            forward_fn=forward_fn,
        )
        initial_validation = {"loss": initial_loss, "accuracy": best_accuracy,
                              "limited_batches": max_eval_batches}
        _atomic_save(net.state_dict(), best_path)
        print(f"initial QAT validation: loss={initial_loss:.6f} "
              f"accuracy={best_accuracy:.4f}", flush=True)

    for epoch in range(start_epoch, int(epochs)):
        net.train()
        if freeze_batchnorm:
            for module in net.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        running_loss = 0.0
        correct = 0
        samples = 0
        for batch_index, (images, labels) in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = forward_fn(net, images)
            loss = criterion(outputs, labels)
            loss.backward()
            if gradient_clip_value is not None:
                torch.nn.utils.clip_grad_value_(
                    net.parameters(), float(gradient_clip_value)
                )
            optimizer.step()
            running_loss += float(loss.item()) * labels.size(0)
            correct += int((outputs.argmax(dim=1) == labels).sum().item())
            samples += labels.size(0)
        if samples == 0:
            raise RuntimeError("training loader produced no samples")
        eval_loss, accuracy = evaluate(
            net,
            validation_loader,
            device,
            max_batches=max_eval_batches,
            forward_fn=forward_fn,
        )
        scheduler.step()
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / samples,
            "train_accuracy": correct / samples,
            "train_samples": samples,
            "validation_loss": eval_loss,
            "validation_accuracy": accuracy,
            "learning_rate": epoch_learning_rate,
        }
        history.append(epoch_record)
        print(
            f"epoch {epoch + 1:03d}: train_loss={epoch_record['train_loss']:.6f} "
            f"train_accuracy={epoch_record['train_accuracy']:.4f} "
            f"validation_loss={eval_loss:.6f} validation_accuracy={accuracy:.4f}",
            flush=True,
        )
        improved = accuracy > best_accuracy
        if improved:
            best_accuracy = accuracy
            best_epoch = epoch + 1
        _atomic_save({
            "epoch": epoch + 1,
            "model": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_validation_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "history": history,
            "initial_validation": initial_validation,
            "run_config": config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, latest_path)
        if improved:
            _atomic_save(net.state_dict(), best_path)
        _atomic_save({
            "run_config": config,
            "completed_epochs": epoch + 1,
            "best_validation_accuracy": best_accuracy,
            "best_epoch": best_epoch,
            "history": history,
            "initial_validation": initial_validation,
            "test": None,
        }, history_path, json_format=True)
        if (target_validation_accuracy is not None
                and epoch + 1 >= min_epochs
                and accuracy >= target_validation_accuracy):
            stop_reason = "validation_target_reached"
            print(f"Validation target {target_validation_accuracy:.4f} reached "
                  f"at epoch {epoch + 1}; evaluating best checkpoint on test", flush=True)
            break

    net.load_state_dict(torch.load(best_path, map_location=device, weights_only=True), strict=True)
    test_loss, test_accuracy = evaluate(
        net, test_loader, device, max_batches=max_eval_batches, forward_fn=forward_fn
    )
    report = {
        "run_config": config,
        "completed_epochs": epoch + 1,
        "stop_reason": stop_reason,
        "best_validation_accuracy": best_accuracy,
        "best_epoch": best_epoch,
        "history": history,
        "initial_validation": initial_validation,
        "test": {"loss": test_loss, "accuracy": test_accuracy,
                 "limited_batches": max_eval_batches},
    }
    _atomic_save(report, history_path, json_format=True)
    return {"best_checkpoint": str(best_path), "latest_checkpoint": str(latest_path),
            "history_path": str(history_path), "best_accuracy": best_accuracy,
            "test_accuracy": test_accuracy, "history": history}
