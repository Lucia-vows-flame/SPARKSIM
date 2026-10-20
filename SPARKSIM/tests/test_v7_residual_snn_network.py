import pytest
import torch
from spikingjelly.activation_based import functional, layer, neuron
from spikingjelly.activation_based.model.spiking_resnet import BasicBlock

from MNSIM.Interface.network import NetworkGraph, get_net, _default_snn_lif_layer_config
from MNSIM.Interface.residual_snn import validate_residual_inputs
from MNSIM.Interface.quantize import ActivationQuantizationState
from MNSIM.Accuracy_Model.snn_accuracy_contract import BINARY_SPIKE_DOMAIN

HW = dict(xbar_size=128, input_shape=(1, 6, 32, 32),
          weight_quantization_bit=9, DAC_num=128, input_bit=1,
          weight_bit=8, ADC_quantize_bit=5, xbar_polarity=2)


def block_graph(projection=False):
    cin, cout, stride = (2, 4, 2) if projection else (2, 2, 1)
    lif = _default_snn_lif_layer_config()
    configs = [
        dict(type='conv', in_channels=cin, out_channels=cout, kernel_size=3, padding=1, stride=stride),
        dict(type='bn', features=cout), lif,
        dict(type='conv', in_channels=cout, out_channels=cout, kernel_size=3, padding=1),
        dict(type='bn', features=cout),
    ]
    indices = [[-1] for _ in configs]
    if projection:
        configs.extend([
            dict(type='conv', in_channels=cin, out_channels=cout, kernel_size=1, stride=2),
            dict(type='bn', features=cout),
        ])
        indices.extend([[-6], [-1]])
        merge_indices = [-3, -1]
    else:
        merge_indices = [-1, -6]
    configs.extend([dict(type='element_sum', snn_merge_mode='residual_add'), lif])
    indices.extend([merge_indices, [-1]])
    for config, inputs in zip(configs, indices):
        config['input_index'] = inputs
    net = NetworkGraph(HW, configs,
                       [dict(weight_bit=9, activation_bit=9, point_shift=-2) for _ in configs],
                       indices, dict(activation_scale=1., activation_bit=9, input_shape=(1, cin, 8, 8)))
    net.sequence_length = 30
    return net


@pytest.mark.parametrize('projection', [False, True])
def test_original_block_matches_official_spikingjelly(projection):
    torch.manual_seed(42)
    net = block_graph(projection).eval()
    cout = 4 if projection else 2
    downsample = torch.nn.Sequential(layer.Conv2d(2, 4, 1, stride=2, bias=False),
                                    layer.BatchNorm2d(4)) if projection else None
    cfg = _default_snn_lif_layer_config()
    ref = BasicBlock(2, cout, stride=2 if projection else 1, downsample=downsample,
                     spiking_neuron=neuron.LIFNode, tau=cfg['tau'],
                     v_threshold=cfg['v_th'], v_reset=cfg['v_reset']).eval()
    functional.set_step_mode(ref, 'm')
    for target, index in [(ref.conv1, 0), (ref.conv2, 3)]:
        target.weight.data.copy_(torch.cat([m.weight for m in net.layer_list[index].sublayer_list], 1))
    ref.bn1.load_state_dict(net.layer_list[1].layer.state_dict())
    ref.bn2.load_state_dict(net.layer_list[4].layer.state_dict())
    if projection:
        ref.downsample[0].weight.data.copy_(net.layer_list[5].sublayer_list[0].weight)
        ref.downsample[1].load_state_dict(net.layer_list[6].layer.state_dict())
    x = torch.randint(0, 2, (2, 5, 2, 8, 8)).float()
    with torch.no_grad():
        expected = ref(x.transpose(0, 1).contiguous()).sum(0)
        actual = net.sequence_accuracy_forward(x, weighted_mode='ORIGINAL')
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(net.sequence_accuracy_forward(x, weighted_mode='ORIGINAL'), actual)


@pytest.mark.parametrize('projection', [False, True])
def test_residual_qat_gradients_and_pim(projection):
    torch.manual_seed(7)
    net = block_graph(projection).train()
    x = torch.randint(0, 2, (2, 3, 2, 8, 8)).float()
    out = net.sequence_qat_forward(x)
    out.sum().backward()
    assert torch.isfinite(out).all()
    for index in ([0, 3, 5] if projection else [0, 3]):
        grad = net.layer_list[index].sublayer_list[0].weight.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    net.eval()
    with torch.no_grad():
        output = net.sequence_accuracy_forward(x, weighted_mode='PIM',
                   net_bit_weights=net.get_weights(), pim_profile='adc_digital_lif')
    assert output.shape == out.shape and torch.isfinite(output).all()


def test_qat_membrane_readout_preserves_signed_current_and_temporal_mean():
    configs = [
        dict(type='view'),
        dict(type='fc', in_features=1, out_features=2),
        dict(type='lif', tau=2., v_th=1., v_reset=0., alpha=2.),
    ]
    net = NetworkGraph(
        HW, configs,
        [dict(weight_bit=9, activation_bit=9, point_shift=-2) for _ in configs],
        [[-1]] * len(configs),
        dict(activation_scale=1., activation_bit=9, input_shape=(1, 1, 1, 1)),
    ).train()
    with torch.no_grad():
        net.layer_list[1].sublayer_list[0].weight.copy_(torch.tensor([[-2.], [-1.]]))
    x = torch.tensor([0., 1., 1., 0.]).reshape(1, 4, 1, 1, 1)
    current = net.sequence_qat_forward(x, readout='membrane')
    # Both negative classifier currents disappear under a spike-count readout.
    assert (current < 0).all()
    assert current.argmax(1).item() == 1
    torch.testing.assert_close(current, torch.tensor([[-1., -.5]]), atol=.004, rtol=0)
    torch.nn.functional.cross_entropy(current, torch.tensor([0])).backward()
    grad = net.layer_list[1].sublayer_list[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    with torch.no_grad():
        spikes = net.sequence_qat_forward(x)
        assert torch.equal(spikes, torch.zeros_like(spikes))
        assert spikes.argmax(1).item() == 0
        torch.testing.assert_close(net.sequence_qat_forward(x, readout='membrane'), current)


def test_residual_domains_allow_dequantized_values_but_reject_broadcasting():
    x = torch.zeros(2, 1, 4, 8, 8)
    validate_residual_inputs([x, x], [BINARY_SPIKE_DOMAIN, ActivationQuantizationState(9, .1)], quantized=True)
    with pytest.raises(ValueError, match='identical shapes'):
        validate_residual_inputs([x, x[:, :, :, :1]], [None, None], quantized=False)
    with pytest.raises(ValueError, match='explicit activation domains'):
        validate_residual_inputs([x, x], [None, None], quantized=True)


def test_resnet18_v7_graph_and_forward():
    net = get_net(HW, 'resnet18_snn_v7_phase6', 8).eval()
    types = [l.layer_config['type'] for l in net.layer_list]
    assert types.count('element_sum') == 8
    assert types.count('conv') == 20  # stem + 16 main + 3 projection
    assert types.count('lif') == 18
    assert 'pooling' not in types
    for i, kind in enumerate(types):
        assert all(-1 <= i + offset < i for offset in net.input_index_list[i])
        if kind == 'element_sum':
            assert types[i + 1] == 'lif'
            assert types[i + net.input_index_list[i][0]] == 'bn'
    with torch.no_grad():
        out = net.sequence_accuracy_forward(torch.zeros(1, 2, 6, 32, 32), weighted_mode='ORIGINAL')
    assert out.shape == (1, 8)
    assert net.sequence_length == 30


@pytest.mark.parametrize(
    ('model', 'expected_blocks', 'expected_convs', 'expected_lifs', 'expected_features'),
    [
        ('resnet34_snn_v7_phase6', 16, 36, 34, 2048),
        ('resnet50_snn_v7_phase6', 16, 53, 50, 8192),
    ],
)
def test_deeper_resnet_v7_graphs_and_forward(
    model, expected_blocks, expected_convs, expected_lifs, expected_features
):
    net = get_net(HW, model, 8).eval()
    types = [l.layer_config['type'] for l in net.layer_list]
    assert types.count('element_sum') == expected_blocks
    assert types.count('conv') == expected_convs
    assert types.count('lif') == expected_lifs
    assert 'pooling' not in types
    assert net.layer_list[-2].layer_config['in_features'] == expected_features
    for i, kind in enumerate(types):
        assert all(-1 <= i + offset < i for offset in net.input_index_list[i])
        if kind == 'element_sum':
            assert types[i + 1] == 'lif'
    with torch.no_grad():
        out = net.sequence_accuracy_forward(
            torch.zeros(1, 2, 6, 32, 32), weighted_mode='ORIGINAL'
        )
    assert out.shape == (1, 8)
    assert net.sequence_length == 30
