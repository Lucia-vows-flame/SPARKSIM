import argparse
import importlib.util
from pathlib import Path

import pytest
import torch

from MNSIM.Interface.network import NetworkGraph, get_net
from MNSIM.Interface import quantize
from MNSIM.Interface.qat_quantization import (
    POLICY_BUFFER, decode_qat_policy, encode_qat_policy, mu_law_activation_ste,
    normalize_qat_policy,
)
from MNSIM.Accuracy_Model.snn_accuracy_contract import BINARY_SPIKE_DOMAIN


def _hardware(policy=None):
    result = dict(xbar_size=128, input_bit=8, weight_bit=8,
                  weight_quantization_bit=9, ADC_quantize_bit=5, DAC_num=128, xbar_polarity=2)
    if policy is not None:
        result['qat_quantization'] = policy
    return result


def _tiny(policy=None):
    net = NetworkGraph(
        _hardware(policy),
        [dict(type='view'), dict(type='fc', in_features=2, out_features=2),
         dict(type='lif', tau=2., v_th=.2, v_reset=0., alpha=2.),
         dict(type='fc', in_features=2, out_features=2),
         dict(type='lif', tau=2., v_th=.2, v_reset=0., alpha=2.)],
        [dict(weight_bit=9, activation_bit=9, point_shift=-2) for _ in range(5)],
        [[-1] for _ in range(5)],
        dict(activation_scale=1., activation_bit=9, input_shape=(1, 1, 1, 2)),
    )
    with torch.no_grad():
        for layer in net.layer_list:
            if isinstance(layer, quantize.QuantizeLayer):
                layer.sublayer_list[0].weight.copy_(torch.tensor([[1., .2], [.2, 1.]]))
    return net


def _input():
    return torch.tensor([[[[[1., 0.]]], [[[0., 1.]]]],
                         [[[[0., 1.]]], [[[1., 1.]]]]])


def test_default_fixed_policy_retains_values_gradients_and_checkpoint_keys():
    torch.manual_seed(42)
    legacy = _tiny()
    configured = _tiny(normalize_qat_policy())
    configured.load_state_dict(legacy.state_dict(), strict=True)
    assert set(legacy.state_dict()) == set(configured.state_dict())
    assert POLICY_BUFFER not in configured.state_dict()
    x = _input()
    a = legacy.sequence_qat_forward(x, readout='membrane')
    b = configured.sequence_qat_forward(x, readout='membrane')
    assert torch.equal(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    for left, right in zip(legacy.parameters(), configured.parameters()):
        assert torch.equal(left.grad, right.grad)


def test_mixed_widths_control_weight_codes_output_grid_and_actual_graph_states():
    policy = normalize_qat_policy({
        'precision_mode': 'mixed', 'activation_bit': 6,
        'layer_overrides': {'L001': {'weight_bit': 3, 'activation_bit': 4},
                            'L003': {'weight_bit': 5, 'activation_bit': 7}},
    })
    net = _tiny(policy)
    captured = []
    original = net.layer_list[1].output_activation_quantization
    def capture(*args):
        state = original(*args)
        captured.append(state)
        return state
    net.layer_list[1].output_activation_quantization = capture
    output = net.sequence_qat_forward(_input(), readout='membrane')
    output.square().sum().backward()
    assert captured and all(s.bit == 4 for s in captured)
    for i, width in ((1, 3), (3, 5)):
        layer = net.layer_list[i]
        assert int(layer.bit_scale_list[1, 0]) == width
        codes = layer.get_bit_weights()
        assert max(v.max() for v in codes.values()) <= 2 ** (width - 1) - 1
        assert layer.sublayer_list[0].weight.grad is not None
        assert torch.isfinite(layer.sublayer_list[0].weight.grad).all()
    assert int(net.layer_list[3].bit_scale_list[0, 0]) == 1  # actual preceding LIF output
    assert net.layer_list[3].output_activation_quantization().bit == 7


def test_nonuniform_companding_levels_symmetry_saturation_and_ste():
    x = torch.linspace(-1.2, 1.2, 1001, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor([1.], dtype=torch.float64)
    y = mu_law_activation_ste(x, 4, scale, False, 15.)
    levels = torch.unique(y.detach())
    assert levels.numel() == 15  # symmetric signed convention, matching uniform
    torch.testing.assert_close(levels, -levels.flip(0), rtol=0, atol=1e-14)
    assert y[0].item() == pytest.approx(-1.)
    assert y[-1].item() == pytest.approx(1.)
    assert levels[7].item() == 0
    assert levels[8] - levels[7] < levels[-1] - levels[-2]
    expected = (torch.expm1(torch.arange(1, 8, dtype=torch.float64) / 7 * torch.log(torch.tensor(16., dtype=torch.float64))) / 15)
    torch.testing.assert_close(levels[8:], expected)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_nonuniform_handles_silent_and_singleton_training_activations():
    for x in (torch.zeros(8, requires_grad=True), torch.tensor([.1], requires_grad=True)):
        scale = torch.tensor([-1.])
        y = mu_law_activation_ste(x, 9, scale, True)
        assert scale.item() > 0 and torch.isfinite(y).all()
        y.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))
    with pytest.raises(RuntimeError, match='not calibrated'):
        mu_law_activation_ste(torch.zeros(8), 9, torch.tensor([-1.]), False)


@pytest.mark.parametrize('precision', ['fixed', 'mixed'])
def test_nonuniform_qat_combines_with_precision_modes_keeps_spikes_and_frozen_eval_scale(precision):
    raw = {'precision_mode': precision, 'activation_quantization': 'mu_law'}
    if precision == 'mixed':
        raw['layer_overrides'] = {'L001': {'weight_bit': 4, 'activation_bit': 5}}
    net = _tiny(raw)
    spike_values = []
    original = net.layer_list[2].layer.forward_sequence
    def capture(x):
        y = original(x)
        spike_values.append(y.detach())
        return y
    net.layer_list[2].layer.forward_sequence = capture
    out = net.sequence_qat_forward(_input(), readout='membrane')
    out.square().sum().backward()
    assert spike_values
    for spikes in spike_values:
        BINARY_SPIKE_DOMAIN.validate(spikes)
    state = net.layer_list[1].output_activation_quantization()
    assert isinstance(state, quantize.NonUniformActivationQuantizationState)
    net.eval()
    saved = net.layer_list[1].last_value.clone()
    net.sequence_qat_forward(_input(), readout='membrane')
    assert torch.equal(saved, net.layer_list[1].last_value)
    assert torch.isfinite(out).all()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters())


def test_nonuniform_bn_and_residual_add_own_their_quantizer():
    config = {'weight_bit': 9, 'activation_bit': 4, 'activation_quantization': 'mu_law', 'activation_mu': 15.}
    layer = quantize.StraightLayer(_hardware(), {'type': 'element_sum', 'snn_merge_mode': 'residual_add'}, config)
    x = torch.tensor([[.1, .8], [-.5, .2]], requires_grad=True)
    states = [quantize.ActivationQuantizationState(4, .1)] * 2
    y = layer.forward([x, x], 'FIX_TRAIN', input_quantization=states)
    state = layer.output_activation_quantization(states)
    assert isinstance(state, quantize.NonUniformActivationQuantizationState)
    expected = mu_law_activation_ste(2*x, 4, layer.last_value.clone(), False, 15.)
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 2.))
    bn = quantize.StraightLayer(_hardware(), {'type': 'bn', 'features': 1}, config)
    bn.forward(x.detach().reshape(2, 1, 1, 2), 'FIX_TRAIN')
    assert isinstance(bn.output_activation_quantization(), quantize.NonUniformActivationQuantizationState)


def test_custom_policy_checkpoint_is_self_describing_and_cannot_load_as_uniform(tmp_path):
    policy = normalize_qat_policy({'activation_quantization': 'mu_law'})
    net = _tiny(policy)
    net.load_state_dict(_tiny().state_dict(), strict=True)  # software pretrain initialization
    net.sequence_qat_forward(_input(), readout='membrane')
    path = tmp_path / 'custom.pth'
    torch.save(net.state_dict(), path)
    saved = torch.load(path, weights_only=True)
    assert decode_qat_policy(saved[POLICY_BUFFER]) == policy
    restored = _tiny(decode_qat_policy(saved[POLICY_BUFFER]))
    restored.load_state_dict(saved, strict=True)
    net.eval()
    restored.eval()
    assert torch.equal(net.sequence_qat_forward(_input(), readout='membrane'), restored.sequence_qat_forward(_input(), readout='membrane'))
    with pytest.raises(RuntimeError, match='policy differs'):
        _tiny().load_state_dict(saved, strict=False)
    with pytest.raises(RuntimeError, match='policy differs'):
        _tiny({'activation_quantization': 'mu_law', 'mu': 7.}).load_state_dict(saved)


def test_nonuniform_hardware_export_and_linear_reconstruction_fail_explicitly():
    net = _tiny({'activation_quantization': 'mu_law'})
    net.sequence_qat_forward(_input(), readout='membrane')
    with pytest.raises(ValueError, match='codebook decoder'):
        net.get_structure()
    net.eval()
    layer = net.layer_list[1]
    with pytest.raises(ValueError, match='codebook decoder'):
        layer.set_weights_forward(torch.ones(1, 2), layer.get_bit_weights(), 'NONE', input_quantization=BINARY_SPIKE_DOMAIN)


@pytest.mark.parametrize('raw', [
    {'weight_bit': 1}, {'weight_bit': True}, {'activation_bit': 17},
    {'activation_quantization': 'lloyd_max'}, {'mu': float('nan')},
    {'layer_overrides': {'L001': {'weight_bit': 4}}},
    {'layer_overrides': {'L999': {'activation_bit': 9}}},
    {'layer_overrides': {'L002': {'activation_bit': 9}}},
    {'precision_mode': 'mixed', 'layer_overrides': {'L002': {'weight_bit': 4}}},
])
def test_invalid_policy_or_spike_override_rejected(raw):
    with pytest.raises(ValueError):
        _tiny(raw)


def test_get_net_overrides_use_final_ids_after_bn_lif_insertion():
    base = get_net(_hardware(), cate='lenet_snn_v7_phase6', num_classes=8)
    weighted_ids = [layer.accuracy_layer_id for layer in base.layer_list if isinstance(layer, quantize.QuantizeLayer)]
    policy = {'precision_mode': 'mixed', 'layer_overrides': {weighted_ids[-1]: {'weight_bit': 4}}}
    configured = get_net(_hardware(policy), cate='lenet_snn_v7_phase6', num_classes=8)
    widths = {layer.accuracy_layer_id: layer.quantize_config['weight_bit'] for layer in configured.layer_list if isinstance(layer, quantize.QuantizeLayer)}
    assert widths[weighted_ids[-1]] == 4
    assert all(widths[k] == 9 for k in weighted_ids[:-1])


def test_training_cli_defaults_to_fixed_uniform_and_accepts_combined_policy(tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/finetune_snn_v7.py'
    spec = importlib.util.spec_from_file_location('qat_cli_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = argparse.Namespace(qat_policy=None, precision_mode=None, weight_bits=None,
                              activation_bits=None, activation_quantization=None, activation_mu=None)
    assert module._qat_policy_from_args(args) == normalize_qat_policy()
    args.precision_mode = 'mixed'
    args.activation_quantization = 'mu_law'
    policy = module._qat_policy_from_args(args)
    assert policy['precision_mode'] == 'mixed' and policy['activation_quantization'] == 'mu_law'
    assert decode_qat_policy(encode_qat_policy(policy)) == policy
