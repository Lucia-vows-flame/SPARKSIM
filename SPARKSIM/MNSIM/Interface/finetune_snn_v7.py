"""Temporal FIX_TRAIN/QAT fine-tuning for V7 SNN checkpoints."""
from __future__ import annotations

from functools import partial

import torch

from MNSIM.Interface.train_snn_v7 import train_net
from MNSIM.Interface.qat_quantization import normalize_qat_policy


def _forward_qat(net, images: torch.Tensor, *, readout: str = "spike") -> torch.Tensor:
    if images.ndim != 5:
        raise ValueError(f"expected V7 batch [B,T,C,H,W], got {tuple(images.shape)}")
    return net.sequence_qat_forward(images, readout=readout)


def finetune_net(
    net,
    train_loader,
    validation_loader,
    device,
    prefix: str,
    *,
    epochs: int = 20,
    learning_rate: float = 0.001,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    gradient_clip_value: float | None = None,
    milestones=(10, 15),
    gamma: float = 0.1,
    output_dir=None,
    max_train_batches=None,
    max_eval_batches=None,
    test_loader=None,
    run_config=None,
    resume=False,
    overwrite=False,
    readout: str = "spike",
    freeze_batchnorm: bool = False,
    evaluate_initial: bool = False,
    target_validation_accuracy: float | None = None,
    min_epochs: int = 1,
):
    """Fine-tune a software checkpoint with temporal MNSIM fake quantization."""
    if readout not in {"spike", "membrane"}:
        raise ValueError(f"unsupported QAT readout: {readout}")
    config = dict(run_config or {})
    config["software_readout"] = readout
    policy = getattr(net, "qat_quantization_policy", None)
    if policy is not None and policy != normalize_qat_policy():
        config["qat_quantization"] = policy
    return train_net(
        net,
        train_loader,
        validation_loader,
        device,
        prefix,
        epochs=epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        gradient_clip_value=gradient_clip_value,
        milestones=milestones,
        gamma=gamma,
        output_dir=output_dir,
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        forward_fn=partial(_forward_qat, readout=readout),
        checkpoint_suffix="_software_qat_finetune_best.pth",
        test_loader=test_loader,
        run_config=config,
        resume=resume,
        overwrite=overwrite,
        freeze_batchnorm=freeze_batchnorm,
        evaluate_initial=evaluate_initial,
        target_validation_accuracy=target_validation_accuracy,
        min_epochs=min_epochs,
    )
