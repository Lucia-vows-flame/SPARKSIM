"""Strict SNN accuracy dataset entry point for round-orchestrated runs."""

from __future__ import annotations

import os

from round_orchestrated_dataset_adapters import build_snn_accuracy_dataloaders


def _root(root=None):
    if root is not None:
        return root
    value = os.environ.get("MNSIM_DVS_BINNED_ROOT")
    if not value:
        raise FileNotFoundError(
            "MNSIM_DVS_BINNED_ROOT is required for round-orchestrated SNN accuracy"
        )
    return value


def get_calibration_dataloader(root=None):
    return build_snn_accuracy_dataloaders(_root(root))[0]


def get_dataloader(root=None):
    return build_snn_accuracy_dataloaders(_root(root))
