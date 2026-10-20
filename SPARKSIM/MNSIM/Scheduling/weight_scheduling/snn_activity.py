"""Frozen SNN spike-traffic profile contract for placement scheduling."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SNNActivityProfileError(ValueError):
    """An SNN activity profile is malformed or belongs to different inputs."""


def load_snn_activity_profile(path: str | Path) -> tuple[dict[str, Any], str]:
    profile_path = Path(path)
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SNNActivityProfileError(
            f"could not load SNN activity profile {profile_path}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SNNActivityProfileError("SNN activity profile must be a JSON object")
    return validate_snn_activity_profile(payload), hashlib.sha256(
        profile_path.read_bytes()
    ).hexdigest()


def validate_snn_activity_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the versioned profile without interpreting its physical costs."""
    if profile.get("schema_version") != 1:
        raise SNNActivityProfileError("SNN activity profile schema_version must be 1")
    for name in ("profile_id", "model_ir_sha256", "round_plan_sha256", "weights_sha256"):
        value = profile.get(name)
        if not isinstance(value, str) or not value:
            raise SNNActivityProfileError(f"SNN activity profile requires {name}")
    calibration = profile.get("calibration")
    if not isinstance(calibration, Mapping):
        raise SNNActivityProfileError("SNN activity profile requires calibration provenance")
    if str(calibration.get("role", "")).lower() not in {"calibration", "train_calibration"}:
        raise SNNActivityProfileError(
            "SNN activity profile calibration.role must identify a calibration split, not evaluation data"
        )
    for name in ("split_id", "split_sha256"):
        value = calibration.get(name)
        if not isinstance(value, str) or not value:
            raise SNNActivityProfileError(f"SNN activity profile calibration requires {name}")
    _positive_int(calibration.get("sample_count"), "calibration.sample_count")
    _positive_int(profile.get("timestep_count"), "timestep_count")

    raw_edges = profile.get("spike_transfer_edges")
    if not isinstance(raw_edges, list):
        raise SNNActivityProfileError("SNN activity profile requires spike_transfer_edges")
    edges: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_edges):
        if not isinstance(raw, Mapping):
            raise SNNActivityProfileError(f"spike_transfer_edges[{index}] must be an object")
        required = (
            "source_round_id",
            "source_tile_id",
            "destination_round_id",
            "destination_tile_id",
        )
        edge = {key: raw.get(key) for key in required}
        if any(not isinstance(value, str) or not value for value in edge.values()):
            raise SNNActivityProfileError(
                f"spike_transfer_edges[{index}] requires non-empty round and logical-tile ids"
            )
        try:
            events = float(raw["spike_events_per_sample"])
        except (KeyError, TypeError, ValueError) as error:
            raise SNNActivityProfileError(
                f"spike_transfer_edges[{index}].spike_events_per_sample must be numeric"
            ) from error
        if not math.isfinite(events) or events < 0:
            raise SNNActivityProfileError(
                f"spike_transfer_edges[{index}].spike_events_per_sample must be finite and non-negative"
            )
        edge["spike_events_per_sample"] = events
        edges.append(edge)
    result = dict(profile)
    result["spike_transfer_edges"] = edges
    result["calibration"] = dict(calibration)
    return result


def bind_profile_edges_to_rounds(
    profile: Mapping[str, Any],
    rounds: tuple[Any, ...] | list[Any],
) -> tuple[dict[str, Any], ...]:
    """Validate logical endpoints and attach incoming traffic to each destination round."""
    contexts = {context.round_id: context for context in rounds}
    indexes = {context.round_id: context.round_index for context in rounds}
    tiles = {
        context.round_id: {tile.tile_id for tile in context.logical_tiles}
        for context in rounds
    }
    incoming: dict[str, list[dict[str, Any]]] = {round_id: [] for round_id in contexts}
    total = 0.0
    for edge in profile["spike_transfer_edges"]:
        source_round = edge["source_round_id"]
        destination_round = edge["destination_round_id"]
        source_tile = edge["source_tile_id"]
        destination_tile = edge["destination_tile_id"]
        if source_round not in contexts or destination_round not in contexts:
            raise SNNActivityProfileError("activity profile references a round outside the RoundPlan")
        if indexes[source_round] > indexes[destination_round]:
            raise SNNActivityProfileError("spike traffic cannot flow backwards in the fixed round order")
        if source_tile not in tiles[source_round] or destination_tile not in tiles[destination_round]:
            raise SNNActivityProfileError("activity profile references a logical tile outside its round")
        events = edge["spike_events_per_sample"]
        if events > 0:
            total += events
            incoming[destination_round].append(dict(edge))
    if total <= 0:
        raise SNNActivityProfileError(
            "activity profile has no positive spike traffic; SNN-aware placement would be invariant"
        )
    return tuple(
        {
            "round_id": context.round_id,
            "incoming_spike_transfer_edges": incoming[context.round_id],
        }
        for context in rounds
    )


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise SNNActivityProfileError(f"{label} must be a positive integer")
    return value


__all__ = [
    "SNNActivityProfileError",
    "bind_profile_edges_to_rounds",
    "load_snn_activity_profile",
    "validate_snn_activity_profile",
]
