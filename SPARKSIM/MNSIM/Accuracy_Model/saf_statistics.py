"""Reproducible, data-free statistics for MNSIM's stuck-at-fault model."""

from __future__ import annotations

import configparser
import copy
import hashlib
import io
import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

from MNSIM.Accuracy_Model.Weight_update import weight_update
from MNSIM.Interface.interface import TrainTestInterface


MAX_SEED = 2**32 - 1


@contextmanager
def temporary_numpy_seed(seed: int | None):
    """Temporarily seed the legacy global NumPy generator."""

    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def parse_seed_list(values: Sequence[str]) -> list[int]:
    """Parse positional and comma-separated seed values in stable order."""

    seeds: list[int] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                seed = int(token)
            except ValueError as error:
                raise ValueError(f"invalid seed: {token!r}") from error
            if seed < 0 or seed > MAX_SEED:
                raise ValueError(
                    f"seed must be in [0, {MAX_SEED}], got {seed}"
                )
            if seed not in seeds:
                seeds.append(seed)
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def iter_weight_arrays(weight_tree: Sequence[Any]) -> Iterator[tuple[int, str, np.ndarray]]:
    """Yield every stored code array in the nested MNSIM weight structure."""

    for layer_index, layer in enumerate(weight_tree):
        if layer is None:
            continue
        if not isinstance(layer, Mapping):
            raise TypeError(
                f"weight layer {layer_index} must be a mapping or None"
            )
        for name, value in layer.items():
            array = np.asarray(value)
            if array.dtype.kind not in "biuf":
                raise TypeError(
                    f"weight array {layer_index}:{name} has unsupported dtype"
                )
            yield layer_index, str(name), array


def weight_tree_sha256(weight_tree: Sequence[Any]) -> str:
    """Hash weight names, shapes, and values for audit provenance."""

    digest = hashlib.sha256()
    for layer_index, name, array in iter_weight_arrays(weight_tree):
        digest.update(str(layer_index).encode("ascii"))
        digest.update(b"\0")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def summarize_weight_changes(
    original: Sequence[Any], updated: Sequence[Any], max_code: int
) -> dict[str, Any]:
    """Summarize effective code changes after one device-error draw."""

    total = 0
    changed = 0
    forced_zero = 0
    forced_max = 0
    layers: list[dict[str, Any]] = []

    original_arrays = list(iter_weight_arrays(original))
    updated_arrays = list(iter_weight_arrays(updated))
    if [(i, n) for i, n, _ in original_arrays] != [
        (i, n) for i, n, _ in updated_arrays
    ]:
        raise ValueError("original and updated weight trees have different keys")

    for (layer_index, name, before), (_, _, after) in zip(
        original_arrays, updated_arrays
    ):
        if before.shape != after.shape:
            raise ValueError(f"weight shape changed for {layer_index}:{name}")
        changed_mask = np.not_equal(before, after)
        layer_total = int(before.size)
        layer_changed = int(changed_mask.sum())
        layer_zero = int(np.logical_and(changed_mask, after == 0).sum())
        layer_max = int(np.logical_and(changed_mask, after == max_code).sum())
        total += layer_total
        changed += layer_changed
        forced_zero += layer_zero
        forced_max += layer_max
        layers.append(
            {
                "layer_index": layer_index,
                "name": name,
                "total_codes": layer_total,
                "changed_codes": layer_changed,
                "changed_fraction": (
                    layer_changed / layer_total if layer_total else 0.0
                ),
                "changed_to_zero": layer_zero,
                "changed_to_max": layer_max,
            }
        )

    return {
        "total_codes": total,
        "changed_codes": changed,
        "changed_fraction": changed / total if total else 0.0,
        "changed_to_zero": forced_zero,
        "changed_to_max": forced_max,
        "layers": layers,
    }


def _device_model(sim_config: str | Path) -> dict[str, Any]:
    parser = configparser.ConfigParser()
    if not parser.read(sim_config, encoding="UTF-8"):
        raise ValueError(f"could not read SimConfig: {sim_config}")
    device_level = parser.getint("Device level", "Device_Level")
    if device_level <= 0:
        raise ValueError("Device_Level must be positive")
    max_code = 2 ** math.floor(math.log2(device_level)) - 1
    saf = [
        float(value.strip())
        for value in parser.get("Device level", "Device_SAF").split(",")
    ]
    if len(saf) < 2 or any(value < 0 for value in saf):
        raise ValueError("Device_SAF must contain two non-negative percentages")
    return {
        "device_level": device_level,
        "max_code": max_code,
        "saf_percentages": saf,
        "nominal_fault_fraction": (saf[0] + saf[-1]) / 100.0,
    }


def collect_saf_statistics(
    *,
    network: str,
    weights: str | Path,
    num_classes: int,
    sim_config: str | Path,
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Collect effective SAF changes without loading a dataset or using a GPU."""

    normalized_seeds = parse_seed_list([str(seed) for seed in seeds])
    device_model = _device_model(sim_config)
    weights_path = Path(weights)
    config_path = Path(sim_config)

    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        context = TrainTestInterface(
            network,
            None,
            str(config_path),
            str(weights_path),
            device="cpu",
            num_classes=num_classes,
        )
        base_weights = context.get_net_bits()

    base_hash = weight_tree_sha256(base_weights)
    per_seed: list[dict[str, Any]] = []
    for seed in normalized_seeds:
        candidate = copy.deepcopy(base_weights)
        with temporary_numpy_seed(seed):
            updated = weight_update(
                str(config_path),
                candidate,
                is_SAF=1,
                is_Variation=0,
                is_Rratio=0,
            )
        summary = summarize_weight_changes(
            base_weights, updated, device_model["max_code"]
        )
        per_seed.append({"seed": seed, **summary})

    fractions = [entry["changed_fraction"] for entry in per_seed]
    aggregate = {
        "changed_fraction_mean": float(np.mean(fractions)),
        "changed_fraction_std": float(np.std(fractions, ddof=0)),
        "changed_fraction_min": float(np.min(fractions)),
        "changed_fraction_max": float(np.max(fractions)),
    }
    return {
        "schema_version": 1,
        "status": "succeeded",
        "network": network,
        "weights": str(weights_path),
        "sim_config": str(config_path),
        "seeds": normalized_seeds,
        "base_weight_sha256": base_hash,
        "device_model": device_model,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


__all__ = [
    "MAX_SEED",
    "collect_saf_statistics",
    "iter_weight_arrays",
    "parse_seed_list",
    "summarize_weight_changes",
    "temporary_numpy_seed",
    "weight_tree_sha256",
]
