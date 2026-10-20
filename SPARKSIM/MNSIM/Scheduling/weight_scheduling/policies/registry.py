"""Policy registry for the unified scheduler."""

from __future__ import annotations

from typing import Any

_POLICIES: dict[str, Any] = {}


def register_policy(policy: Any) -> Any:
    policy_id = str(getattr(policy, "policy_id", "")).strip()
    if not policy_id:
        raise ValueError("policy must define a non-empty policy_id")
    _POLICIES[policy_id] = policy
    return policy


def get_policy(policy_id: str) -> Any:
    canonical = str(policy_id).strip().lower()
    canonical = {
        "sched-seq": "seq",
        "sequential": "seq",
        "sched-flashsef": "flashsef_ptr",
        "flashsef-ptr": "flashsef_ptr",
        "snn-native-temporal": "snn_native_temporal",
        "sched-snn_native_temporal": "snn_native_temporal",
        "snn-flashsef-temporal": "snn_flashsef_temporal",
        "sched-snn_flashsef_temporal": "snn_flashsef_temporal",
    }.get(canonical, canonical)
    if not _POLICIES:
        from MNSIM.Scheduling.weight_scheduling.policies.flashsef_ptr import FlashSeFPTRPolicy
        from MNSIM.Scheduling.weight_scheduling.policies.seq import SequentialPolicy
        from MNSIM.Scheduling.weight_scheduling.policies.snn_native_temporal import (
            SNNNativeTemporalPolicy,
        )
        from MNSIM.Scheduling.weight_scheduling.policies.snn_flashsef_temporal import (
            SNNFlashSeFTemporalPolicy,
        )

        register_policy(SequentialPolicy())
        register_policy(FlashSeFPTRPolicy())
        register_policy(SNNNativeTemporalPolicy())
        register_policy(SNNFlashSeFTemporalPolicy())
    try:
        return _POLICIES[canonical]
    except KeyError as exc:
        raise ValueError(f"unknown weight scheduling policy {policy_id!r}") from exc


__all__ = ["get_policy", "register_policy"]
