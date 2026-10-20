"""Explicit, disjoint V7 loaders for SNN training and assessment."""
from __future__ import annotations

import os
from pathlib import Path

from torch.utils.data import DataLoader

from MNSIM.Interface.dvs_binned_npz import DVSBinnedNPZDataset


def training_loaders(root: Path, train_split: str, train_batch: int, eval_batch: int):
    if train_batch < 1 or eval_batch < 1:
        raise ValueError("batch sizes must be positive")
    if train_split not in {"pretrain", "finetune"}:
        raise ValueError(f"unsupported training split: {train_split}")
    datasets = {
        split: DVSBinnedNPZDataset(str(root), split)
        for split in (train_split, "calibration", "test")
    }
    paths = [{Path(path).resolve() for path in dataset.files} for dataset in datasets.values()]
    if any(paths[i] & paths[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("training, validation and test splits must be disjoint")
    workers = int(os.environ.get("MNSIM_DVS_NUM_WORKERS", "0"))
    if workers < 0:
        raise ValueError("MNSIM_DVS_NUM_WORKERS must be nonnegative")

    def loader(split: str, batch: int, shuffle: bool):
        return DataLoader(
            datasets[split], batch_size=batch, shuffle=shuffle,
            num_workers=workers, pin_memory=True, drop_last=False,
        )

    identities = {split: dataset.accuracy_split_identity() for split, dataset in datasets.items()}
    return (
        loader(train_split, train_batch, True),
        loader("calibration", eval_batch, False),
        loader("test", eval_batch, False),
        identities,
    )
