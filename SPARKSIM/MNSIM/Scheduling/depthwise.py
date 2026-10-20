"""Map depthwise kernels onto independent diagonal channel blocks."""
from collections.abc import Mapping

import numpy as np


def is_depthwise(shape):
    return str(shape.get("Depthwise", "normal")) == "separable"


def depthwise_block_size(shape, xbar_rows, xbar_cols):
    channels = int(shape["Inputchannel"])
    if channels != int(shape["Outputchannel"]):
        raise ValueError("depthwise mapping requires one output per input channel")
    size = min(xbar_rows // int(shape["Kernelsize"]) ** 2, xbar_cols)
    if size <= 0:
        raise ValueError("depthwise kernel exceeds crossbar rows")
    return size


def expand_depthwise_codes(codes: Mapping, shape, xbar_rows, xbar_cols):
    """Preserve integer codes while placing each channel on its own diagonal.

    Each output-major block retains the global output axis used by the existing
    fragment selector. Only matching input/output channel blocks are deployed.
    No quantization is performed here.
    """
    block = depthwise_block_size(shape, xbar_rows, xbar_cols)
    channels = int(shape["Inputchannel"])
    area = int(shape["Kernelsize"]) ** 2
    result = {}
    for key, value in codes.items():
        if not key.startswith("split0_weight"):
            raise ValueError(f"unexpected depthwise code key {key}")
        kernels = np.asarray(value)
        if kernels.size != channels * area or kernels.shape[0] != channels:
            raise ValueError("depthwise codes must contain one kernel per channel")
        kernels = kernels.reshape(channels, area)
        for index, start in enumerate(range(0, channels, block)):
            end = min(start + block, channels)
            target = np.zeros((channels, (end - start) * area), dtype=kernels.dtype)
            for channel in range(start, end):
                offset = (channel - start) * area
                target[channel, offset:offset + area] = kernels[channel]
            result[key.replace("split0_", f"split{index}_", 1)] = target
    return result
