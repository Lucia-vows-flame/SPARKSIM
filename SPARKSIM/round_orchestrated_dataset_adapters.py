"""Dataset adapters used by round-orchestrated accuracy runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader


def build_snn_accuracy_dataloaders(root: str | Path, *, dataset_module: str = "MNSIM.Interface.dvs_binned_npz"):
    """Build formal SNN loaders with logical ``train`` backed by ``pretrain``.

    The original dataset module is imported unchanged.  Its dataset class is
    instantiated explicitly with the required physical split, then the
    logical identity is set to ``train`` for calibration artifact compatibility.
    """

    import importlib

    base = importlib.import_module(dataset_module)
    root = str(root)
    meta = base._load_meta(root)
    if getattr(meta, "num_classes", None) is not None:
        base.NUM_CLASSES = int(meta.num_classes)
    calibration_dataset = base.DVSBinnedNPZDataset(
        root=root,
        split="pretrain",
        target_hw=(base.TARGET_H, base.TARGET_W),
    )
    calibration_dataset.split = "train"
    test_dataset = base.DVSBinnedNPZDataset(
        root=root,
        split="test",
        target_hw=(base.TARGET_H, base.TARGET_W),
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=base.TRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=base.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=base.TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=base.NUM_WORKERS,
        drop_last=False,
        pin_memory=True,
    )
    return calibration_loader, test_loader
