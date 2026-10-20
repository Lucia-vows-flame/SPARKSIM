"""Resolve user-facing device selections without silently changing execution."""

from __future__ import annotations

import re
from typing import Any

import torch


_CUDA_SPEC = re.compile(r"^cuda(?::([0-9]+))?$")


def resolve_torch_device(spec: Any) -> torch.device:
    """Resolve CPU/GPU selectors and fail fast for unavailable explicit GPUs."""

    if spec is None:
        return torch.device("cpu")
    if isinstance(spec, bool):
        raise ValueError("device selector must be cpu, cuda, cuda:N, or a GPU index")
    if isinstance(spec, int):
        index = spec
    elif isinstance(spec, str):
        value = spec.strip().lower()
        if value == "cpu":
            return torch.device("cpu")
        match = _CUDA_SPEC.fullmatch(value)
        if match is not None:
            index = 0 if match.group(1) is None else int(match.group(1))
        elif value.isdigit():
            index = int(value)
        else:
            raise ValueError(
                f"invalid device selector {spec!r}; use cpu, cuda, cuda:N, or a GPU index"
            )
    else:
        raise ValueError(
            f"invalid device selector {spec!r}; use cpu, cuda, cuda:N, or a GPU index"
        )

    if index < 0:
        raise ValueError(f"GPU index must be non-negative, got {index}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable for explicit GPU request {spec!r}; refusing CPU fallback"
        )
    device_count = torch.cuda.device_count()
    if index >= device_count:
        raise ValueError(
            f"GPU index {index} is out of range for {device_count} visible CUDA device(s)"
        )
    return torch.device(f"cuda:{index}")

