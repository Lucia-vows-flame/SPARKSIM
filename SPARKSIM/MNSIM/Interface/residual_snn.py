"""Explicit SpikingJelly-style residual graphs using existing MNSIM nodes."""


def resnet_layers(num_classes, lif_config, input_shape, *, blocks, bottleneck=False):
    layers = []

    def add(kind, source=None, **kwargs):
        index = len(layers)
        sources = [index - 1] if source is None else source
        layers.append(dict(type=kind, input_index=[s - index for s in sources], **kwargs))
        return index

    def conv_bn(cin, cout, stride=1, kernel=3, source=None):
        add('conv', source, in_channels=cin, out_channels=cout,
            kernel_size=kernel, padding=kernel // 2, stride=stride)
        return add('bn', features=cout)

    def lif():
        return add('lif', **{k: v for k, v in lif_config.items() if k != 'type'})

    # V7 stem downsamples with a convolution; no max/average pooling.
    conv_bn(6, 64, stride=2)
    lif()
    cin = 64
    stage_channels = (64, 128, 256, 512)
    expansion = 4 if bottleneck else 1
    for stage, base_channels in enumerate(stage_channels):
        cout = base_channels * expansion
        for block in range(blocks[stage]):
            stride = 2 if stage > 0 and block == 0 else 1
            shortcut = len(layers) - 1
            if bottleneck:
                conv_bn(cin, base_channels, kernel=1)
                lif()
                conv_bn(base_channels, base_channels, stride=stride)
                lif()
                main = conv_bn(base_channels, cout, kernel=1)
            else:
                conv_bn(cin, cout, stride)
                lif()
                main = conv_bn(cout, cout)
            if cin != cout or stride != 1:
                shortcut = conv_bn(cin, cout, stride, kernel=1, source=[shortcut])
            add('element_sum', [main, shortcut], snn_merge_mode='residual_add')
            lif()
            cin = cout
    height, width = input_shape[-2:]
    for _ in range(4):
        height, width = (height + 1) // 2, (width + 1) // 2
    add('view')
    add('fc', in_features=cin * height * width, out_features=num_classes)
    # Match the project's spike-count classifier convention.
    lif()
    return layers


def resnet18_layers(num_classes, lif_config, input_shape):
    return resnet_layers(
        num_classes, lif_config, input_shape, blocks=(2, 2, 2, 2), bottleneck=False
    )


def resnet34_layers(num_classes, lif_config, input_shape):
    return resnet_layers(
        num_classes, lif_config, input_shape, blocks=(3, 4, 6, 3), bottleneck=False
    )


def resnet50_layers(num_classes, lif_config, input_shape):
    return resnet_layers(
        num_classes, lif_config, input_shape, blocks=(3, 4, 6, 3), bottleneck=True
    )


def validate_residual_inputs(values, states, *, quantized):
    """Values are real-valued tensors, including dequantized fake-quant values.

    Distinct code scales are legal: they are NOT raw integer codes. An identity
    branch is binary, whereas the main BN branch is signed real-valued.
    """
    from MNSIM.Accuracy_Model.snn_accuracy_contract import BinarySpikeDomain
    from MNSIM.Interface.quantize import ActivationQuantizationState

    if not isinstance(values, list) or len(values) != 2:
        raise ValueError('residual element_sum requires exactly two inputs')
    if values[0].shape != values[1].shape:
        raise ValueError('residual inputs must have identical shapes; broadcasting is forbidden')
    if values[0].device != values[1].device or values[0].dtype != values[1].dtype:
        raise ValueError('residual inputs must have matching device and dtype')
    if quantized:
        if not isinstance(states, list) or len(states) != 2 or any(
            not isinstance(state, (BinarySpikeDomain, ActivationQuantizationState))
            for state in states
        ):
            raise ValueError('residual quantized inputs require explicit activation domains')
