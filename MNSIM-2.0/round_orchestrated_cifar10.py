"""Offline CIFAR10 adapter for round-orchestrated ANN accuracy checks.

The original MNSIM CIFAR10 loader remains unchanged and continues to own its
legacy download behavior.  This module is an explicit offline entry point.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch.utils.data as Data
import torchvision
import torchvision.transforms as Transforms


def _root(root: str | None = None) -> str:
    value = root or os.environ.get("MNSIM_ROUND_CIFAR10_ROOT")
    if not value:
        raise FileNotFoundError(
            "MNSIM_ROUND_CIFAR10_ROOT must point to a local CIFAR10 directory"
        )
    path = Path(value)
    archive_dir = path / "cifar-10-batches-py"
    if not archive_dir.is_dir() and (path / "data_batch_1").is_file():
        return str(path)
    if not archive_dir.is_dir():
        raise FileNotFoundError(
            f"local CIFAR10 archive directory is missing: {archive_dir}"
        )
    return str(path)


NUM_CLASSES = 10


def get_dataloader(root: str | None = None):
    local_root = _root(root)
    train_dataset = torchvision.datasets.CIFAR10(
        root=local_root,
        download=False,
        train=True,
        transform=Transforms.Compose([
            Transforms.Pad(padding=4),
            Transforms.RandomCrop(32),
            Transforms.RandomHorizontalFlip(),
            Transforms.ToTensor(),
        ]),
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=local_root,
        download=False,
        train=False,
        transform=Transforms.Compose([Transforms.ToTensor()]),
    )
    return (
        Data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True),
        Data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, drop_last=False),
    )
