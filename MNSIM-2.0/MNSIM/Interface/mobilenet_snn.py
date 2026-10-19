"""MobileNetV1-style sequential SNN graph builder for the V7 dataset."""

from __future__ import annotations

from typing import Any


def mobilenet_v1_snn_layers(
    num_classes: int,
    lif_config: dict[str, Any],
    input_shape: tuple[int, ...] = (1, 6, 32, 32),
) -> list[dict[str, Any]]:
    """Build a pooling-free MobileNetV1 graph.

    Depthwise convolutions use MNSIM's existing ``depthwise='separable'``
    representation.  The following pointwise convolution performs channel
    mixing.  LIF nodes are inserted by :func:`MNSIM.Interface.network.get_net`
    alongside every convolution, keeping the builder purely structural.
    """

    if len(input_shape) != 4 or int(input_shape[1]) != 6 or tuple(input_shape[-2:]) != (32, 32):
        raise ValueError("mobilenet_snn_v7_phase6 expects V7 input shape [B,6,32,32]")
    if int(num_classes) < 1:
        raise ValueError("num_classes must be positive")

    layers: list[dict[str, Any]] = []

    def conv(cin: int, cout: int, *, kernel: int, stride: int = 1,
             padding: int | None = None, depthwise: str = "normal") -> None:
        layers.append({
            "type": "conv",
            "in_channels": int(cin),
            "out_channels": int(cout),
            "kernel_size": int(kernel),
            "padding": int(kernel // 2 if padding is None else padding),
            "stride": int(stride),
            "depthwise": depthwise,
        })

    def depthwise_pointwise(channels_in: int, channels_out: int, *, stride: int = 1) -> None:
        conv(channels_in, channels_in, kernel=3, stride=stride, depthwise="separable")
        conv(channels_in, channels_out, kernel=1)

    # 32x32 -> 16x16 stem.
    conv(6, 32, kernel=3, stride=2)
    depthwise_pointwise(32, 64, stride=1)
    depthwise_pointwise(64, 128, stride=2)  # 16x16 -> 8x8
    depthwise_pointwise(128, 128, stride=1)
    depthwise_pointwise(128, 256, stride=2)  # 8x8 -> 4x4
    depthwise_pointwise(256, 256, stride=1)
    depthwise_pointwise(256, 512, stride=2)  # 4x4 -> 2x2

    for _ in range(5):
        depthwise_pointwise(512, 512, stride=1)

    depthwise_pointwise(512, 1024, stride=2)  # 2x2 -> 1x1
    depthwise_pointwise(1024, 1024, stride=1)
    layers.append({"type": "view"})
    layers.append({
        "type": "fc",
        "in_features": 1024,
        "out_features": int(num_classes),
    })
    layers.append({"type": "lif", **{key: value for key, value in lif_config.items() if key != "type"}})
    return layers


__all__ = ["mobilenet_v1_snn_layers"]
