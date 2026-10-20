"""SNN-only metadata adapter for explicit BN/add/LIF residual graphs.

Uses the existing MNSIM BN/post-add output quantizer domains, not the raw
crossbar accumulator of either branch. No arithmetic cost formula lives here.
"""
from fractions import Fraction
import math

from MNSIM.Scheduling.digital_domain import AccumulatorDomain, compile_lif_constants
from MNSIM.Scheduling.round_types import INVALID_LIF_DOMAIN, Phase1Invalid


def attach_residual_metadata(network, infos):
    layers = list(network.layer_list)
    owners = {}
    shapes = {-1: tuple(network.input_params['input_shape'][1:])}

    def fail(message):
        raise Phase1Invalid(INVALID_LIF_DOMAIN, message)

    def node_id(i):
        return f'L{i:03d}_{infos[i]["type"]}'

    def quantized_domain(i):
        try:
            state = layers[i].output_activation_quantization()
            return AccumulatorDomain.from_scale(
                signed_bit_width=state.bit, scale=Fraction(str(state.scale)),
                source='mnsim_' + infos[i]['type'] + '_output_quantizer',
            )
        except (ValueError, RuntimeError, AttributeError) as exc:
            fail(f'{node_id(i)} requires trained/calibrated activation metadata: {exc}')

    for i, (layer, info) in enumerate(zip(layers, infos)):
        kind = info['type']
        parents = [i + offset for offset in network.input_index_list[i]]
        if not parents or any(p < -1 or p >= i for p in parents):
            fail(f'{node_id(i)} has invalid graph inputs')
        if kind in {'conv', 'fc'}:
            if len(parents) != 1:
                fail(f'{node_id(i)} weighted layer must have one input')
            # Binary only when the actual source is a spike/view-of-spike.
            source = parents[0]
            while source >= 0 and infos[source]['type'] == 'view':
                source += network.input_index_list[source][0]
            if source != -1 and infos[source]['type'] != 'lif':
                fail(f'{node_id(i)} residual weighted input is not binary')
            info['Inputbit'] = 1
            shapes[i] = ((int(info['Outfeature']),) if kind == 'fc' else
                         (int(info['Outputchannel']), *map(int, info['Outputsize'])))
            owners[i] = node_id(i)
        else:
            shapes[i] = shapes[parents[0]]
            owners[i] = owners.get(parents[0])
            if kind == 'view':
                shapes[i] = (math.prod(shapes[i]),)
            if kind == 'element_sum':
                if layer.layer_config.get('snn_merge_mode') != 'residual_add' or len(parents) != 2:
                    fail(f'{node_id(i)} unsupported residual merge')
                if shapes[parents[0]] != shapes[parents[1]]:
                    fail(f'{node_id(i)} residual shapes differ')
                if infos[parents[0]]['type'] != 'bn':
                    fail(f'{node_id(i)} main residual input must end in BN')
                # Output placement follows the main branch reducer, once only.
                # Shortcut data arrives through the existing structural inputs.
                info['snn_merge_mode'] = 'residual_add'
                info['merge_ownership_policy'] = 'main_branch_reducer'
                info['merge_source_ids'] = [node_id(p) if p >= 0 else 'model_input' for p in parents]
            shape = shapes[i]
            info['Outputchannel'] = shape[0]
            info['Outputsize'] = list(shape[1:]) if len(shape) > 1 else [1, 1]
            info['output_neuron_region'] = [0, math.prod(shape)]
        if kind in {'bn', 'element_sum'}:
            domain = quantized_domain(i)
            info['accumulator_domain'] = domain.to_dict()
            info['outputbit'] = domain.signed_bit_width
        if kind == 'lif':
            if len(parents) != 1:
                fail(f'{node_id(i)} LIF requires one merged input')
            parent = parents[0]
            if 'accumulator_domain' not in infos[parent]:
                fail(f'{node_id(i)} producer has no domain')
            domain = AccumulatorDomain.from_dict(infos[parent]['accumulator_domain'])
            info['accumulator_domain'] = domain.to_dict()
            info['compiled_lif_constants'] = compile_lif_constants(
                domain=domain, threshold=Fraction(str(info['v_th'])),
                reset=Fraction(str(info['v_reset'])),
            ).to_dict()
            if not owners[i]:
                fail(f'{node_id(i)} has no weighted output owner')
            info['lif_owner_layer_id'] = owners[i]
            info['outputbit'] = 1


def is_residual_graph(network):
    return any(l.layer_config.get('snn_merge_mode') == 'residual_add'
               for l in network.layer_list)
