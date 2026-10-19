"""Reusable multi-branch SNN graph builders.

The builders describe branches as ordinary MNSIM graph nodes.  A merge is a
first-class structural node, so the same topology is consumed by software
sequence execution and the Phase 1/Phase 2 performance paths.
"""

from __future__ import annotations

from typing import Any


def _lif_config(lif_config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in lif_config.items() if key != "type"}


def inception_snn_layers(
    num_classes: int,
    lif_config: dict[str, Any],
    input_shape: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Build an Inception-lite/SWMR-style V7 classifier.

    Each branch is ``conv -> LIF`` and all branch outputs are aligned in
    ``[T, B, C, H, W]`` before channel concatenation.  A following pointwise
    convolution owns the merged representation.  The graph deliberately
    keeps all merge and LIF boundaries explicit for Phase 1 metadata.
    """

    if (
        len(input_shape) != 4
        or int(input_shape[1]) != 6
        or tuple(input_shape[-2:]) != (32, 32)
    ):
        raise ValueError("inception_snn_v7_phase6 expects V7 input shape [B,6,32,32]")

    layers: list[dict[str, Any]] = []
    lif = _lif_config(lif_config)

    def add(kind: str, sources: list[int] | None = None, **kwargs: Any) -> int:
        index = len(layers)
        if sources is None:
            sources = [index - 1]
        config = {"type": kind, "input_index": [source - index for source in sources]}
        config.update(kwargs)
        layers.append(config)
        return index

    def conv_lif(cin: int, cout: int, *, kernel: int = 3, stride: int = 1, source: int | None = None) -> int:
        conv = add(
            "conv",
            None if source is None else [source],
            in_channels=cin,
            out_channels=cout,
            kernel_size=kernel,
            padding=kernel // 2,
            stride=stride,
        )
        add("lif", [conv], **lif)
        return len(layers) - 1

    # A stride-2 stem produces a 16x16 event feature map.
    current = conv_lif(int(input_shape[1]), 32, kernel=3, stride=2, source=-1)

    def inception_block(source: int, cin: int, branch_channels: int, fused_channels: int, stride: int) -> int:
        branches = [
            conv_lif(cin, branch_channels, kernel=1, stride=stride, source=source),
            conv_lif(cin, branch_channels, kernel=3, stride=stride, source=source),
            conv_lif(cin, branch_channels, kernel=5, stride=stride, source=source),
        ]
        merged = add(
            "concat",
            branches,
            dim="channel",
            concat_dim="channel",
            merge_mode="channel_concat",
            time_policy="same_timestep",
        )
        return conv_lif(branch_channels * 3, fused_channels, kernel=1, source=merged)

    # 16x16 -> 8x8 -> 4x4.  All branches use the same stride, so concat is
    # shape-safe without pooling or cross-timestep buffering.
    current = inception_block(current, 32, 16, 48, stride=1)
    current = inception_block(current, 48, 24, 64, stride=2)
    current = inception_block(current, 64, 32, 96, stride=2)

    view = add("view", [current])
    fc = add("fc", [view], in_features=96 * 4 * 4, out_features=int(num_classes))
    add("lif", [fc], **lif)
    return layers


__all__ = ["inception_snn_layers"]
