#-*-coding:utf-8-*-
import collections
import copy
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    BINARY_SPIKE_DOMAIN,
    FRONTEND_LUT_LIF_LUT,
    SUPPORTED_SNN_HARDWARE_PROFILES,
)
from MNSIM.Interface import quantize
from MNSIM.Interface.qat_quantization import (
    POLICY_BUFFER, decode_qat_policy, encode_qat_policy,
    normalize_qat_policy, resolve_layer_quantization,
)


def _default_snn_lif_layer_config():
    # These are the software-pretrain parameters shared by both final V7 SNNs.
    tau = float(os.environ.get('MNSIM_SNN_LIF_TAU', '50.5016666556'))
    v_th = float(os.environ.get('MNSIM_SNN_LIF_V_TH', '0.01'))
    v_reset = float(os.environ.get('MNSIM_SNN_LIF_V_RESET', '0.0'))
    alpha = float(os.environ.get('MNSIM_SNN_LIF_ALPHA', '2.0'))
    return {
        'type': 'lif',
        'tau': tau,
        'v_th': v_th,
        'v_reset': v_reset,
        'alpha': alpha,
    }


class NetworkGraph(nn.Module):
    def __init__(self, hardware_config, layer_config_list, quantize_config_list, input_index_list, input_params):
        # input_params contain: 'activation_scale', 'activation_bit', and 'input_shape'
        super(NetworkGraph, self).__init__()
        # same length for layer_config_list , quantize_config_list and input_index_list
        assert len(layer_config_list) == len(quantize_config_list)
        assert len(layer_config_list) == len(input_index_list)
        raw_policy = hardware_config.get('qat_quantization')
        self.qat_quantization_policy = None
        if raw_policy is not None:
            self.qat_quantization_policy, quantize_config_list = resolve_layer_quantization(
                raw_policy, layer_config_list, quantize_config_list,
            )
            if self.qat_quantization_policy != normalize_qat_policy():
                self.register_buffer(POLICY_BUFFER, encode_qat_policy(self.qat_quantization_policy))
        # layer list
        self.layer_list = nn.ModuleList()
        # add layer to layer list by layer_config, quantize_config, and input_index
        for layer_index, (layer_config, quantize_config) in enumerate(
            zip(layer_config_list, quantize_config_list)
        ):
            assert 'type' in layer_config.keys()
            if layer_config['type'] in quantize.QuantizeLayerStr:
                # configure quantization setting of 'conv' and 'fc'
                
                layer = quantize.QuantizeLayer(hardware_config, layer_config, quantize_config)
            elif layer_config['type'] in quantize.StraightLayerStr:
                layer = quantize.StraightLayer(hardware_config, layer_config, quantize_config)
            else:
                assert 0, f'not support {layer_config["type"]}'
            layer.accuracy_layer_id = f'L{layer_index:03d}'
            self.layer_list.append(layer)
        # save input_index_list, input_index is a list
        self.input_index_list = copy.deepcopy(input_index_list)
        self.input_params = copy.deepcopy(input_params)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + POLICY_BUFFER
        saved = state_dict.get(key)
        configured = getattr(self, POLICY_BUFFER, None)
        if saved is not None:
            if configured is None or decode_qat_policy(saved) != self.qat_quantization_policy:
                raise RuntimeError('checkpoint QAT quantization policy differs from the configured network; construct with its saved qat_quantization policy')
        elif configured is not None:
            # An ordinary software-pretrain checkpoint can initialize a new
            # QAT policy. Already tagged QAT checkpoints must match above.
            state_dict[key] = configured.detach().clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def _validate_linear_hardware_quantization(self):
        if any(layer.quantize_config.get('activation_quantization', 'uniform') != 'uniform'
               for layer in self.layer_list):
            raise ValueError('non-uniform QAT activations require an explicit hardware codebook decoder; current hardware export/PIM supports uniform activation domains')

    def input_activation_quantization(self):
        return quantize.ActivationQuantizationState(
            bit=int(self.input_params['activation_bit']),
            scale=float(self.input_params['activation_scale']),
        )

    @staticmethod
    def _indexed_graph_value(values, input_index, layer_index):
        selected = [values[index + layer_index + 1] for index in input_index]
        return selected[0] if len(selected) == 1 else selected

    @staticmethod
    def _merge_layer_type(layer):
        return getattr(layer, 'layer_config', {}).get('type')

    @classmethod
    def _validate_merge_inputs(cls, layer, values, *, sequence=False):
        if not isinstance(values, (list, tuple)):
            return
        if len(values) < 2:
            raise ValueError(f'{cls._merge_layer_type(layer)} requires at least two inputs')
        if any(not torch.is_tensor(value) for value in values):
            raise TypeError('multi-input graph values must be tensors')
        layer_type = cls._merge_layer_type(layer)
        if sequence:
            first = values[0]
            if first.dim() < 3:
                raise ValueError('sequence merge inputs must include [T,B,...] dimensions')
            for value in values[1:]:
                if value.shape[:2] != first.shape[:2]:
                    raise ValueError('merge inputs must share sample and timestep axes')
        if layer_type == 'element_sum':
            if any(value.shape != values[0].shape for value in values[1:]):
                raise ValueError(f'{layer_type} inputs must have identical shapes')
        elif layer_type == 'element_multiply':
            if all(value.shape == values[0].shape for value in values[1:]):
                return
            gate, feature = values[0], values[1]
            channel_gate = (
                gate.dim() + 2 == feature.dim()
                and gate.shape == feature.shape[:-2]
            )
            expanded_channel_gate = (
                gate.dim() == feature.dim()
                and gate.shape[:-2] == feature.shape[:-2]
                and tuple(gate.shape[-2:]) == (1, 1)
            )
            if not (channel_gate or expanded_channel_gate) or any(
                value.shape != feature.shape for value in values[2:]
            ):
                raise ValueError(
                    'element_multiply inputs must have identical shapes or start '
                    'with one broadcastable channel gate'
                )
        elif layer_type == 'concat':
            rank = values[0].dim()
            configured_dim = getattr(getattr(layer, 'layer', None), 'dim', 1)
            dim = rank - 3 if configured_dim == 'channel' else int(configured_dim)
            if rank == 5 and configured_dim == 1:
                dim = 2
            if dim < 0:
                dim += rank
            if dim < 0 or dim >= rank:
                raise ValueError(f'concat dimension {dim} is out of range for rank {rank}')
            for value in values[1:]:
                if any(value.shape[axis] != values[0].shape[axis] for axis in range(rank) if axis != dim):
                    raise ValueError('concat inputs must match on every non-concat axis')

    @classmethod
    def _merged_quantization_state(cls, layer, input_state, method):
        layer_type = cls._merge_layer_type(layer)
        if not isinstance(input_state, list):
            return input_state
        if layer_type == 'element_sum' and method == 'TRADITION':
            return input_state[0] if input_state else None
        if layer_type in {'concat', 'element_multiply'}:
            if not input_state or any(state is None for state in input_state):
                return None
            first = input_state[0]
            if any(state != first for state in input_state[1:]):
                raise ValueError(f'{layer_type} inputs must use one common activation domain')
            return first
        return None

    @staticmethod
    def _run_graph_layer(
        layer, input_value, input_quantization, method, adc_action
    ):
        if isinstance(layer, quantize.QuantizeLayer):
            if isinstance(input_quantization, list):
                raise ValueError('weighted layers must have exactly one activation input')
            return layer.forward(
                input_value,
                method,
                adc_action,
                input_quantization=input_quantization,
            )
        return layer.forward(
            input_value,
            method,
            adc_action,
            input_quantization=input_quantization,
        )

    @staticmethod
    def _output_graph_quantization(layer, input_quantization, method):
        if method == 'TRADITION':
            if isinstance(input_quantization, list):
                return NetworkGraph._merged_quantization_state(layer, input_quantization, method)
            return input_quantization
        if isinstance(input_quantization, list):
            merged = NetworkGraph._merged_quantization_state(layer, input_quantization, method)
            if merged is not None:
                return merged
        return layer.output_activation_quantization(input_quantization)

    def set_activation_calibration(self, enabled):
        """Enable explicit activation-range calibration without enabling train mode."""

        for layer in self.layer_list:
            layer.set_activation_calibration(enabled)

    def reset_activation_scales(self):
        """Discard learned activation ranges before an independent calibration pass."""

        with torch.no_grad():
            for layer in self.layer_list:
                if isinstance(layer, quantize.QuantizeLayer) or layer.layer_config[
                    'type'
                ] in {'bn', 'element_sum'}:
                    layer.last_value.fill_(-1)
                if layer.layer_config['type'] == 'element_sum':
                    layer.post_add_quantizer_version.zero_()

    def calibrate_activation_scales(self, batches, max_batches=None, reset=False):
        """Calibrate activation ranges with frozen model and BatchNorm state.

        This is the supported path for a checkpoint that predates a newly
        introduced activation boundary such as ``element_sum``. It does not
        update model weights or BatchNorm running statistics.
        """

        if torch.is_tensor(batches):
            batches = (batches,)
        if reset:
            self.reset_activation_scales()
        original_training = self.training
        self.eval()
        self.set_activation_calibration(True)
        try:
            with torch.no_grad():
                for batch_index, batch in enumerate(batches):
                    if max_batches is not None and batch_index >= max_batches:
                        break
                    input_tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
                    self.forward(input_tensor, 'FIX_TRAIN')
        finally:
            self.set_activation_calibration(False)
            self.train(original_training)

    def forward(self, x, method = 'SINGLE_FIX_TEST', adc_action = 'SCALE'):
        input_quantization = self.input_activation_quantization()
        quantize._publish_legacy_activation_quantization_state(input_quantization)
        tensor_list = [x]
        quantization_list = [input_quantization]
        for i, layer in enumerate(self.layer_list):
            input_index = self.input_index_list[i]
            input_value = self._indexed_graph_value(tensor_list, input_index, i)
            input_state = self._indexed_graph_value(quantization_list, input_index, i)
            self._validate_merge_inputs(layer, input_value)
            output = self._run_graph_layer(
                layer, input_value, input_state, method, adc_action
            )
            output_state = self._output_graph_quantization(
                layer, input_state, method
            )
            tensor_list.append(output)
            quantization_list.append(output_state)
            if output_state is not None:
                quantize._publish_legacy_activation_quantization_state(output_state)
        
        return tensor_list[-1]

    def reset_state(self):
        """Reset all recurrent state at an independent sequence boundary."""
        for layer in self.layer_list:
            reset = getattr(layer, 'reset_state', None)
            if callable(reset):
                reset()

    @staticmethod
    def _merge_sequence_batch(value):
        t, batch = value.shape[:2]
        return value.reshape(t * batch, *value.shape[2:]), t, batch

    @staticmethod
    def _unmerge_sequence_batch(value, t, batch):
        return value.reshape(t, batch, *value.shape[1:])

    def sequence_accuracy_forward(
        self, x, *, weighted_mode, adc_action='SCALE', net_bit_weights=None,
        pim_profile=None, lif_perturbation=None, readout='spike',
    ):
        """Run one shared sequence graph with switchable weighted-layer compute."""
        if not torch.is_tensor(x) or x.dim() != 5:
            raise ValueError(f'expected sequence batch [B,T,C,H,W], got {getattr(x, "shape", None)}')
        weighted_mode = str(weighted_mode).upper()
        readout = str(readout).lower()
        if readout not in {'spike', 'membrane'}:
            raise ValueError(f'unsupported sequence readout: {readout}')
        if weighted_mode not in {'ORIGINAL', 'PIM'}:
            raise ValueError(f'unsupported sequence weighted mode: {weighted_mode}')
        if weighted_mode == 'PIM' and net_bit_weights is None:
            raise ValueError('PIM sequence mode requires net_bit_weights')
        if weighted_mode == 'PIM':
            self._validate_linear_hardware_quantization()
        if (
            weighted_mode == 'PIM'
            and pim_profile not in SUPPORTED_SNN_HARDWARE_PROFILES
        ):
            raise ValueError(
                f'unsupported SNN PIM hardware profile: {pim_profile}'
            )

        self.reset_state()
        BINARY_SPIKE_DOMAIN.validate(x)
        input_state = BINARY_SPIKE_DOMAIN
        tensor_list = [x.permute(1, 0, 2, 3, 4).contiguous()]
        quantization_list = [input_state]
        pim_weights = iter(
            item for item in (net_bit_weights or []) if item is not None
        )
        for i, layer in enumerate(self.layer_list):
            input_index = self.input_index_list[i]
            x_in = self._indexed_graph_value(tensor_list, input_index, i)
            q_in = self._indexed_graph_value(quantization_list, input_index, i)
            self._validate_merge_inputs(layer, x_in, sequence=True)

            if isinstance(layer, quantize.QuantizeLayer):
                if isinstance(x_in, list):
                    raise ValueError('weighted layers must have exactly one activation input')
                merged, t, batch = self._merge_sequence_batch(x_in)
                if weighted_mode == 'PIM':
                    if pim_profile == FRONTEND_LUT_LIF_LUT:
                        y_merged = layer.set_weights_forward(
                            merged,
                            next(pim_weights),
                            None,
                            input_quantization=q_in,
                            numerical_mode='NO_ADC_RECONSTRUCTION',
                        )
                    else:
                        y_merged = layer.set_weights_forward(
                            merged,
                            next(pim_weights),
                            adc_action,
                            input_quantization=q_in,
                            numerical_mode='ADC',
                        )
                else:
                    y_merged = layer.forward(merged, 'TRADITION', None)
                y = self._unmerge_sequence_batch(y_merged, t, batch)
            elif getattr(layer, 'layer_config', {}).get('type') == 'lif':
                if isinstance(x_in, list):
                    raise ValueError('LIF layers must have exactly one merged input')
                layer.set_accuracy_perturbation(
                    lif_perturbation if weighted_mode == 'PIM' else None
                )
                # A signed classifier current can stay below zero for every
                # class after an early update, making a terminal spike readout
                # exactly zero and cutting off the useful learning signal.
                # Residual SNN software pretraining may therefore request the
                # terminal membrane/current readout while retaining the LIF
                # node in the graph for hardware accounting and PIM runs.
                if readout == 'membrane' and i == len(self.layer_list) - 1:
                    y = x_in
                else:
                    y = layer.layer.forward_sequence(x_in)
            else:
                if isinstance(x_in, list):
                    if not x_in:
                        raise ValueError('structural layer received no inputs')
                    y = layer.forward(
                        x_in,
                        'TRADITION' if weighted_mode == 'ORIGINAL' else 'FIX_TRAIN',
                        None,
                        input_quantization=q_in,
                    )
                else:
                    merged, t, batch = self._merge_sequence_batch(x_in)
                    y_merged = layer.forward(
                        merged,
                        'TRADITION' if weighted_mode == 'ORIGINAL' else 'FIX_TRAIN',
                        None,
                        input_quantization=q_in,
                    )
                    y = self._unmerge_sequence_batch(y_merged, t, batch)

            if getattr(layer, 'layer_config', {}).get('type') == 'lif':
                q_out = BINARY_SPIKE_DOMAIN
            elif (
                weighted_mode == 'PIM'
                and isinstance(layer, quantize.QuantizeLayer)
                and pim_profile == FRONTEND_LUT_LIF_LUT
            ):
                q_out = None
            elif weighted_mode == 'PIM':
                q_out = self._merged_quantization_state(layer, q_in, weighted_mode)
                if q_out is None:
                    q_out = layer.output_activation_quantization(q_in)
            else:
                q_out = self._merged_quantization_state(layer, q_in, weighted_mode)
            tensor_list.append(y)
            quantization_list.append(q_out)

        if readout == 'membrane':
            # A membrane/current readout is a temporal average. Summing all
            # T steps scales classifier logits by T and destabilizes CE loss.
            return tensor_list[-1].mean(dim=0)
        return tensor_list[-1].sum(dim=0)

    def sequence_qat_forward(self, x, *, readout='spike'):
        """Run a V7 sequence with MNSIM's differentiable ``FIX_TRAIN`` path.

        Weighted layers use fake weight/activation quantization while LIF
        layers still run as shared multi-step SpikingJelly nodes.  This is the
        temporal QAT counterpart of ``NetworkGraph.forward(..., 'FIX_TRAIN')``;
        the latter only accepts ordinary four-dimensional ANN tensors.
        ``membrane`` matches the software-pretrain classifier-current readout:
        bypass only the terminal LIF and average the current over time.
        """
        if readout not in {'spike', 'membrane'}:
            raise ValueError(f'unsupported QAT readout: {readout}')
        if not torch.is_tensor(x) or x.dim() != 5:
            raise ValueError(
                f'expected sequence batch [B,T,C,H,W], got {getattr(x, "shape", None)}'
            )

        self.reset_state()
        BINARY_SPIKE_DOMAIN.validate(x)
        tensor_list = [x.permute(1, 0, 2, 3, 4).contiguous()]
        quantization_list = [BINARY_SPIKE_DOMAIN]
        for i, layer in enumerate(self.layer_list):
            input_index = self.input_index_list[i]
            x_in = self._indexed_graph_value(tensor_list, input_index, i)
            q_in = self._indexed_graph_value(quantization_list, input_index, i)
            self._validate_merge_inputs(layer, x_in, sequence=True)
            layer_type = getattr(layer, 'layer_config', {}).get('type')

            if layer_type == 'lif':
                if isinstance(x_in, list):
                    raise ValueError('LIF layers must have exactly one merged input')
                layer.set_accuracy_perturbation(None)
                if readout == 'membrane' and i == len(self.layer_list) - 1:
                    y = x_in
                    q_out = q_in
                else:
                    y = layer.layer.forward_sequence(x_in)
                    q_out = BINARY_SPIKE_DOMAIN
            else:
                if isinstance(x_in, list):
                    y = layer.forward(x_in, 'FIX_TRAIN', None, input_quantization=q_in)
                else:
                    merged, t, batch = self._merge_sequence_batch(x_in)
                    y_merged = layer.forward(
                        merged,
                        'FIX_TRAIN',
                        None,
                        input_quantization=q_in,
                    )
                    y = self._unmerge_sequence_batch(y_merged, t, batch)
                # Weighted/BN/add nodes own a new activation domain. Preserve
                # their configured bit width/codebook rather than passing the
                # incoming binary-spike metadata through a single-input node.
                q_out = self._output_graph_quantization(layer, q_in, 'FIX_TRAIN')

            tensor_list.append(y)
            quantization_list.append(q_out)

        if readout == 'membrane':
            return tensor_list[-1].mean(dim=0)
        return tensor_list[-1].sum(dim=0)
   
    # CNNParted_set_weights_forward:  the interface with CNNParted,accuracy evaluation
    def CNNParted_set_weights_forward(self,x,tensor_list_CNNParted,start_num,end_num,adc_action='SCALE'):
        self._validate_linear_hardware_quantization()
        net_bit_weights=self.get_weights()
        quantize.last_activation_scale = self.input_params['activation_scale']
        quantize.last_activation_bit = self.input_params['activation_bit']
        # filter None
        net_bit_weights = list(filter(lambda x:x!=None, net_bit_weights))
        count=0
        for i,layer in enumerate(self.layer_list[start_num:end_num],start=start_num):
            # find the input tensor
            input_index = self.input_index_list[i]
            if isinstance(layer, quantize.QuantizeLayer):
                input_value = self._indexed_graph_value(tensor_list_CNNParted, input_index, i)
                if isinstance(input_value, list):
                    raise ValueError('weighted layers must have exactly one activation input')
                tensor_list_CNNParted.append(layer.set_weights_forward(input_value, net_bit_weights[count], adc_action))
               
                count = count + 1
            else:
                input_value = self._indexed_graph_value(tensor_list_CNNParted, input_index, i)
                self._validate_merge_inputs(layer, input_value)
                tensor_list_CNNParted.append(layer.forward(input_value, 'FIX_TRAIN', None))
        return tensor_list_CNNParted,tensor_list_CNNParted[-1]
    # CNNParted Interface : calculate_equal_bit: the equaled pim supported quantization bit, using in CNNParted step2
    def calculate_equal_bit(self):
        #calculated in quantize.py   (layers)
        equal_bit_list=[]
        for index,layer in enumerate(self.layer_list):
            if layer.layer_config['type']=='conv' or layer.layer_config['type']=='fc':
                equal_bit_list.append(layer.calculate_equal_bit())
            
        return equal_bit_list
        
   
    def configure_prequantized_weights(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if enabled:
            for layer_index, layer in enumerate(self.layer_list):
                if not isinstance(layer, quantize.QuantizeLayer):
                    continue
                try:
                    quantize.decode_prequantized_weight_splits(
                        [sub.weight for sub in layer.sublayer_list],
                        qbit=int(layer.quantize_config['weight_bit']),
                    )
                except Exception as error:
                    raise ValueError(
                        f"invalid prequantized weights for layer_list.{layer_index}: {error}"
                    ) from error
        self._use_prequantized_weights = enabled

    def get_weights(self, *, use_prequantized_weights=None):
        if use_prequantized_weights is None:
            use_prequantized_weights = bool(
                getattr(self, '_use_prequantized_weights', False)
            )
        net_bit_weights = []
        for layer in self.layer_list:
            if isinstance(layer, quantize.QuantizeLayer):
                net_bit_weights.append(
                    layer.get_bit_weights(
                        use_prequantized_weights=bool(use_prequantized_weights)
                    )
                )
            else:
                net_bit_weights.append(layer.get_bit_weights())
        return net_bit_weights
    def set_weights_forward(
        self, x, net_bit_weights, adc_action = 'SCALE', adc_diagnostics=None
    ):
        self._validate_linear_hardware_quantization()
        input_quantization = self.input_activation_quantization()
        quantize._publish_legacy_activation_quantization_state(input_quantization)
        # filter None
        net_bit_weights = list(filter(lambda x:x!=None, net_bit_weights))
        tensor_list = [x]
        quantization_list = [input_quantization]
        count = 0
        for i, layer in enumerate(self.layer_list):
            input_index = self.input_index_list[i]
            input_value = self._indexed_graph_value(tensor_list, input_index, i)
            input_state = self._indexed_graph_value(quantization_list, input_index, i)
            self._validate_merge_inputs(layer, input_value)
            if isinstance(layer, quantize.QuantizeLayer):
                output = layer.set_weights_forward(
                    input_value,
                    net_bit_weights[count],
                    adc_action,
                    adc_diagnostics=adc_diagnostics,
                    layer_index=i,
                    input_quantization=input_state,
                )
                count = count + 1
            else:
                output = layer.forward(
                    input_value,
                    'FIX_TRAIN',
                    None,
                    input_quantization=input_state,
                )
            output_state = self._output_graph_quantization(
                layer, input_state, 'FIX_TRAIN'
            )
            tensor_list.append(output)
            quantization_list.append(output_state)
            if output_state is not None:
                quantize._publish_legacy_activation_quantization_state(output_state)
        return tensor_list[-1]
    def get_structure(self):
        self._validate_linear_hardware_quantization()
        # get network structure information
        x = torch.zeros(self.input_params['input_shape'])
        self.to(x.device)
        self.eval()
        self.reset_state()
        tensor_list = [x]
        for i, layer in enumerate(self.layer_list):
            # find the input tensor
            input_index = self.input_index_list[i]
            input_value = self._indexed_graph_value(tensor_list, input_index, i)
            self._validate_merge_inputs(layer, input_value)
            tensor_list.append(layer.structure_forward(input_value))
        # structure information, stored as list
        net_info = []
        for layer in self.layer_list:
            net_info.append(layer.layer_info)
        return net_info
    def load_change_weights(self, state_dict):
        # input is a state dict, weights
        # concat all layer_list weights, keys_map's format:
        # layer_list.x.layer.weight (original weight)
        # layer_list.x (split weight)
        #   |- layer_list.x.layer_list.0.weight
        #   |- layer_list.x.layer_list.1.weight
        keys_map = collections.OrderedDict()
        for key in state_dict.keys():
            tmp_key = re.sub('\.[a-z]{0,3}layer_list\.\d+\.weight$', '', key)
            if tmp_key not in keys_map.keys():
                keys_map[tmp_key] = [key]
            else:
                keys_map[tmp_key].append(key)
        # concat and split
        tmp_state_dict = collections.OrderedDict()
        for tmp_key, key_list in keys_map.items():
            if len(key_list) == 1 and tmp_key==key_list[0]:
                # print('origin weights')
                # weights on PIM is the original weights
                tmp_state_dict[tmp_key] = state_dict[key_list[0]]
            else:
                # print(f'transfer weights {tmp_key}')
                # get layer info
                layer_config = None
                hardware_config = None
                # retrieve the hardware config and 
                for i in range(len(self.layer_list)):
                    #prefix 'module' comes from  the dataparallel
                    name = f'layer_list.{i}'
                    if name == tmp_key:
                        layer_config = self.layer_list[i].layer_config
                        hardware_config = self.layer_list[i].hardware_config
                assert layer_config, 'layer must have layer config'
                assert hardware_config, 'layer must have hardware config'
                # concat weights
                total_weights = torch.cat([state_dict[key] for key in key_list], dim = 1)
                # split weights according to HW parameters (along with output channel dimension)
                if layer_config['type'] == 'conv':
                    split_len = (hardware_config['xbar_size'] // (layer_config['kernel_size'] ** 2))
                elif layer_config['type'] == 'fc':
                    split_len = hardware_config['xbar_size']
                else:
                    print(layer_config['type'])
                    assert 0, f'not support {layer_config["type"]}, only conv and fc layers have weights'
                weights_list = torch.split(total_weights, split_len, dim = 1)
                # load weights
                for i, weights in enumerate(weights_list):
                    tmp_state_dict[tmp_key + f'.sublayer_list.{i}.weight'] = weights
                
        # load weights
       
        self.load_state_dict(tmp_state_dict)
    #TODO:fix the loading of weights in different type
    # def load_change_weights_different(self, state_dict,standart_dict):
    #     #standart_dict:trained from MNSIM
    #     count=0
    #     flag=1
    #     flagg=1
    #     final_dict=collections.OrderedDict()
    #     tmp_standart_dict=collections.OrderedDict()
    #     num_key=collections.OrderedDict()
    #     tmp1_state_dict=collections.OrderedDict()
    #     #first clear the bit and last_value in standard_dict
    #     for key in standart_dict.keys():
    #         if 'last_value' in key  or re.search(r'sublayer_list\.[1-9]\d*',key) is not None :
    #             if re.search(r'sublayer_list\.[1-9]\d*',key) is not None :
    #                 count=count+1
    #                 num_key[key]=[count]
    #             continue
    #         else:
    #             count=0
    #             tmp_standart_dict[key]=standart_dict[key]
    #     for key in state_dict.keys():
    #         if 'Conv_/features/features' in key  or 'Gemm' in key :
    #             tmp1_state_dict[key]=state_dict[key]
    #         else:
    #             continue
    #     for (key1,key2) in zip(tmp1_state_dict.keys(),tmp_standart_dict.keys()):
    #         if '_amax' in key1:
    #             key_bit_scale_list=key2
    #             scale=tmp1_state_dict[key1]
    #             activation_scale=scale/2**9 #quantize_config_list[1]['activation_bit']
    #             weight_scale=scale/2**9 #quantize_config_list[1]['weight_bit']
    #             value_bit_scale_list=torch.empty(3,2)
    #             value_bit_scale_list[0,0]=9 #quantize_config_list[1]['activation_bit']
    #             value_bit_scale_list[0,1]=activation_scale
    #             value_bit_scale_list[1,0]=9 #quantize_config_list[1]['weight_bit']
    #             value_bit_scale_list[0,1]=weight_scale
    #             value_bit_scale_list[2,0]=9 #quantize_config_list[1]['activation_bit']
    #             value_bit_scale_list[0,1]=activation_scale
    #             final_dict[key_bit_scale_list]=value_bit_scale_list
    #         else:
    #             final_dict[key2]=tmp1_state_dict[key1]
          
    #     for keys in final_dict.keys():
    #         print("keys",keys)
    #     for keys in final_dict.keys():
    #         if keys in standart_dict.keys():
    #             #keys:special
    #             standart_dict[keys]=final_dict[keys]
    #     keys_map = collections.OrderedDict()
    #     for key in standart_dict.keys():
    #         tmp_key = re.sub('\.[a-z]{0,3}layer_list\.\d+\.weight$', '', key)
    #         if tmp_key not in keys_map.keys():
    #             keys_map[tmp_key] = [key]
    #         else:
    #             keys_map[tmp_key].append(key)
    #     # concat and split
    #     for key in standart_dict.keys():
    #         print(key,standart_dict[key].shape)
    #     tmp_state_dict = collections.OrderedDict()
    #     for tmp_key, key_list in keys_map.items():
    #         tmp_key = re.sub('module.', '', tmp_key)
    #         if len(key_list) == 1 and tmp_key == re.sub('module.','',key_list[0]):
    #             # weights on PIM is the original weights
    #             tmp_state_dict[tmp_key] = standart_dict[key_list[0]]
    #         else:
    #             # get layer info
    #             layer_config = None
    #             hardware_config = None
    #             # retrieve the hardware config and 
    #             for i in range(len(self.layer_list)):
    #                 #prefix 'module' comes from  the dataparallel
    #                 name = f'layer_list.{i}'
    #                 if name == tmp_key:
    #                     layer_config = self.layer_list[i].layer_config
    #                     hardware_config = self.layer_list[i].hardware_config
    #             assert layer_config, 'layer must have layer config'
    #             assert hardware_config, 'layer must have hardware config'
    #             # concat weights
    #             total_weights = standart_dict[key_list[0]]
    #             # split weights according to HW parameters (along with output channel dimension)
    #             if layer_config['type'] == 'conv':
    #                 split_len = (hardware_config['xbar_size'] // (layer_config['kernel_size'] ** 2))
    #             elif layer_config['type'] == 'fc':
    #                 split_len = hardware_config['xbar_size']
    #                 flag=0
    #             else:
    #                 print(layer_config['type'])
    #                 assert 0, f'not support {layer_config["type"]}, only conv and fc layers have weights'
    #             weights_list = torch.split(total_weights, split_len, dim = 1)
    #             # load weights
    #             for i, weights in enumerate(weights_list):
    #                 if flag==0:
    #                     tmp_state_dict[tmp_key + f'.sublayer_list.{i}.weight'] = weights.view(weights.shape[0],weights.shape[1])
    #                 else:
    #                     tmp_state_dict[tmp_key + f'.sublayer_list.{i}.weight'] = weights

    #         flag=1
    #     # load weights
    #     self.load_state_dict(tmp_state_dict,strict=True)


def get_net(hardware_config = None, cate = 'lenet', num_classes = 10):
    # define the NN structure
    # initial config
    
    if hardware_config == None:
        hardware_config = {'xbar_size': 256, 'input_bit': 2, 'weight_bit': 1, 'ADC_quantize_bit': 10, 'DAC_num': 256}
    weight_quantization_bit = int(hardware_config.get('weight_quantization_bit', 9))
    if weight_quantization_bit < 2:
        raise ValueError(
            f'weight_quantization_bit must be at least 2, got {weight_quantization_bit}'
        )
    # layer_config_list, quantize_config_list, and input_index_list
    layer_config_list = []
    quantize_config_list = []
    input_index_list = []
    # layer by layer
    # add new NN models here (conv/fc is followed by one bn layer automatically):
    #assert cate in ['lenet', 'vgg16', 'vgg8', 'alexnet', 'resnet18']
   
    # Keep canonical MNSIM vgg8 unchanged; the project-specific event/SNN
    # topology is selected explicitly through vgg8_snn or vgg8_snn_swbn.
    assert cate in [
        'lenet', 'vgg16', 'vgg8', 'vgg8_snn', 'vgg8_snn_swbn',
        'vgg8_snn_v7_phase6', 'standard_vgg8_snn',
        'standard_vgg8_snn_v7_phase6', 'alexnet', 'resnet18', 'EfficientNet',
        'lenet_snn_v7_phase6', 'alexnet_snn_v7_phase6',
        'mobilenet_snn_v7_phase6',
        'vgg11_snn_v7_phase6', 'vgg13_snn_v7_phase6',
        'vgg16_snn_v7_phase6', 'resnet18_snn_v7_phase6',
        'resnet34_snn_v7_phase6', 'resnet50_snn_v7_phase6',
        'inception_snn_v7_phase6'
    ]

    if cate.startswith('lenet') and not cate.endswith('_snn_v7_phase6'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 6, 'kernel_size': 5})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 6, 'out_channels': 16, 'kernel_size': 5})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 16, 'out_channels': 120, 'kernel_size': 5})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 120, 'out_features': 84})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 84, 'out_features': num_classes})
    elif cate.startswith('vgg16') and not cate.endswith('_snn_v7_phase6'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 2048, 'out_features': num_classes})
        # layer_config_list.append({'type': 'dropout'})
        # layer_config_list.append({'type': 'relu'})
        # layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes})
        # layer_config_list.append({'type': 'dropout'})
        # layer_config_list.append({'type': 'relu'})
        # layer_config_list.append({'type': 'fc', 'in_features': 4096, 'out_features': num_classes})
    elif cate.startswith('vgg8') and not cate.startswith('vgg8_snn'):
        # Original MNSIM VGG8: conv blocks use ANN-style ReLU/pooling layout.
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'padding': 0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': num_classes})
    elif cate in {'vgg8_snn_v7_phase6', 'standard_vgg8_snn_v7_phase6'}:
        # Formal V7 VGG8-SNN topology.  ``standard_vgg8_snn_v7_phase6`` is
        # retained as a compatibility alias for existing checkpoints and
        # reports; the canonical model id is ``vgg8_snn_v7_phase6``.
        layer_config_list.append({'type': 'conv', 'in_channels': 6, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'padding': 0, 'stride': 2})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': num_classes})
        layer_config_list.append(_default_snn_lif_layer_config())
    elif cate == 'lenet_snn_v7_phase6':
        # LeNet-style sequential SNN.  Strided convolutions replace pooling
        # so every spatial reduction remains an explicit weighted operation.
        layer_config_list.extend([
            {'type': 'conv', 'in_channels': 6, 'out_channels': 32, 'kernel_size': 5, 'padding': 2, 'stride': 2},
            {'type': 'conv', 'in_channels': 32, 'out_channels': 64, 'kernel_size': 5, 'padding': 2, 'stride': 2},
            {'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2},
            {'type': 'view'},
            {'type': 'fc', 'in_features': 128 * 4 * 4, 'out_features': 256},
            _default_snn_lif_layer_config(),
            {'type': 'fc', 'in_features': 256, 'out_features': num_classes},
            _default_snn_lif_layer_config(),
        ])
    elif cate == 'alexnet_snn_v7_phase6':
        # AlexNet-style sequential SNN with convolutional downsampling.
        layer_config_list.extend([
            {'type': 'conv', 'in_channels': 6, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 2},
            {'type': 'conv', 'in_channels': 64, 'out_channels': 192, 'kernel_size': 3, 'padding': 1, 'stride': 2},
            {'type': 'conv', 'in_channels': 192, 'out_channels': 384, 'kernel_size': 3, 'padding': 1},
            {'type': 'conv', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2},
            {'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 0, 'stride': 2},
            {'type': 'view'},
            {'type': 'fc', 'in_features': 256, 'out_features': 512},
            _default_snn_lif_layer_config(),
            {'type': 'fc', 'in_features': 512, 'out_features': num_classes},
            _default_snn_lif_layer_config(),
        ])
    elif cate == 'mobilenet_snn_v7_phase6':
        from MNSIM.Interface.mobilenet_snn import mobilenet_v1_snn_layers
        layer_config_list = mobilenet_v1_snn_layers(
            num_classes,
            _default_snn_lif_layer_config(),
            hardware_config.get('input_shape', (1, 6, 32, 32)),
        )
    elif cate in {'vgg11_snn_v7_phase6', 'vgg13_snn_v7_phase6', 'vgg16_snn_v7_phase6'}:
        # VGG-family SNNs use the same 32x32 spatial schedule: the first
        # convolution of stages 2-5 performs downsampling with stride 2.
        # The final stage starts with a valid 3x3/stride-2 convolution to
        # produce a 1x1 feature map without a pooling layer.
        vgg_snn_specs = {
            'vgg11_snn_v7_phase6': (
                ((128,), (256,), (512, 512), (512, 512), (512, 512)),
            ),
            'vgg13_snn_v7_phase6': (
                ((128, 128), (256, 256), (512, 512), (512, 512), (512, 512)),
            ),
            'vgg16_snn_v7_phase6': (
                ((128, 128), (256, 256), (512, 512, 512), (512, 512, 512), (512, 512, 512)),
            ),
        }
        stages = vgg_snn_specs[cate][0]
        in_channels = 6
        for stage_index, stage in enumerate(stages):
            for conv_index, out_channels in enumerate(stage):
                downsample = stage_index > 0 and conv_index == 0
                padding = 0 if stage_index == len(stages) - 1 and conv_index == 0 else 1
                layer_config_list.append({
                    'type': 'conv',
                    'in_channels': in_channels,
                    'out_channels': out_channels,
                    'kernel_size': 3,
                    'padding': padding,
                    'stride': 2 if downsample else 1,
                })
                in_channels = out_channels
        layer_config_list.extend([
            {'type': 'view'},
            {'type': 'fc', 'in_features': in_channels, 'out_features': num_classes},
            _default_snn_lif_layer_config(),
        ])
    elif cate.startswith('standard_vgg8_snn'):
        # Frozen topology used by the 2026-07-26/2026-08-12 DVS checkpoint.
        # Strided convolutions replace pooling; LIF hooks are inserted below.
        layer_config_list.append({'type': 'conv', 'in_channels': 2, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'padding': 0, 'stride': 2})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': num_classes})
        layer_config_list.append(_default_snn_lif_layer_config())
    elif cate.startswith('vgg8_snn') and cate != 'vgg8_snn_v7_phase6':
        # Project VGG8-SNN topology: strided conv layers replace pooling, and
        # LIF hooks are attached later so weighted layer ordering stays clear.
        input_channels = 6 if cate == 'vgg8_snn_v7_phase6' else 2
        layer_config_list.append({'type': 'conv', 'in_channels': input_channels, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 3, 'padding': 0})
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 1024, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': num_classes})
        layer_config_list.append(_default_snn_lif_layer_config())
    elif cate.startswith('alexnet') and not cate.endswith('_snn_v7_phase6'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 192, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'conv', 'in_channels': 192, 'out_channels': 384, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 384, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1024, 'out_features': 512})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes})
    elif cate in {
        'resnet18_snn_v7_phase6', 'resnet34_snn_v7_phase6',
        'resnet50_snn_v7_phase6',
    }:
        from MNSIM.Interface.residual_snn import (
            resnet18_layers, resnet34_layers, resnet50_layers,
        )
        layer_builder = {
            'resnet18_snn_v7_phase6': resnet18_layers,
            'resnet34_snn_v7_phase6': resnet34_layers,
            'resnet50_snn_v7_phase6': resnet50_layers,
        }[cate]
        layer_config_list = layer_builder(
            num_classes, _default_snn_lif_layer_config(),
            hardware_config.get('input_shape', (1, 6, 32, 32)),
        )
    elif cate == 'inception_snn_v7_phase6':
        from MNSIM.Interface.complex_snn import inception_snn_layers
        layer_config_list = inception_snn_layers(
            num_classes,
            _default_snn_lif_layer_config(),
            hardware_config.get('input_shape', (1, 6, 32, 32)),
        )
    elif cate.startswith('resnet18'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'kernel_size': 2, 'stride': 2})
        # block 1
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 2
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
            # input_index indicates the inputs of the current layer come from which layer, "-1" means the previous layer
        layer_config_list.append({'type': 'relu'})
        # block 3
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 2, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 4
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 5
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 2, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 6
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 7
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 2})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 2, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 8
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'padding': 1, 'stride': 1})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # output
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 2048, 'out_features': 512})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes})
    #TODO: train the EfficientNet
    elif cate.startswith('EfficientNet'):
        layer_config_list.append({'type': 'conv', 'in_channels':3 , 'out_channels':32 , 'kernel_size': 3, 'padding': 1,'stride':2})
        layer_config_list.append({'type': 'Swish'})
        #block1
        layer_config_list.append({'type': 'conv', 'in_channels':32 , 'out_channels':32 , 'kernel_size': 3, 'padding': 1,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 16, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 32, 'out_features': 8})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 8, 'out_features': 32})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]})                         
        layer_config_list.append({'type': 'conv', 'in_channels':32 , 'out_channels':16 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        
        
      
        
        #block2
        layer_config_list.append({'type': 'conv', 'in_channels':16 , 'out_channels':96 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':96 , 'out_channels':96 , 'kernel_size': 3, 'padding': 1,'stride':2,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 10, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 96, 'out_features': 4})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 4, 'out_features': 96})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':96 , 'out_channels':24 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        
       
        
        #block2
        layer_config_list.append({'type': 'conv', 'in_channels':24 , 'out_channels':144 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':144 , 'out_channels':144 , 'kernel_size': 3, 'padding': 1,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 7, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 144, 'out_features': 6})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 6, 'out_features': 144})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':144 , 'out_channels':24 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        
  
        
        #block3
        layer_config_list.append({'type': 'conv', 'in_channels':24 , 'out_channels':144 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':144 , 'out_channels':144 , 'kernel_size': 5, 'padding': 2,'stride':2,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 5, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 144, 'out_features': 6})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 6, 'out_features': 144})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':144 , 'out_channels':40 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
       
        
   
        
        #block3
        layer_config_list.append({'type': 'conv', 'in_channels':40 , 'out_channels':240 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':240 , 'out_channels':240 , 'kernel_size': 5, 'padding': 2,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 4, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 240, 'out_features': 10})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 10, 'out_features': 240})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':240 , 'out_channels':40 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
       
      
        #block4
        layer_config_list.append({'type': 'conv', 'in_channels':40 , 'out_channels':240 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':240 , 'out_channels':240 , 'kernel_size': 3, 'padding': 1,'stride':2,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 4, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 240, 'out_features': 10})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 10, 'out_features': 240})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':240 , 'out_channels':80 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
    
        
        #block4
        layer_config_list.append({'type': 'conv', 'in_channels':80 , 'out_channels':480 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':480 , 'out_channels':480 , 'kernel_size': 3, 'padding': 1,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 4, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 480, 'out_features': 20})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 20, 'out_features': 480})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':480 , 'out_channels':80 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
        
        #block4
        layer_config_list.append({'type': 'conv', 'in_channels':80 , 'out_channels':480 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':480 , 'out_channels':480 , 'kernel_size': 3, 'padding': 1,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 4, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 480, 'out_features': 20})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 20, 'out_features': 480})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':480 , 'out_channels':80 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
        
        
        #block5
        layer_config_list.append({'type': 'conv', 'in_channels':80 , 'out_channels':480 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':480 , 'out_channels':480 , 'kernel_size': 5, 'padding': 2,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 6, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 480, 'out_features': 20})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 20, 'out_features': 480})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':480 , 'out_channels':112 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        
     
        #block5
        layer_config_list.append({'type': 'conv', 'in_channels':112 , 'out_channels':672 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':672 , 'out_channels':672 , 'kernel_size': 5, 'padding': 2,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 8, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 672, 'out_features': 28})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 28, 'out_features': 672})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':672 , 'out_channels':112 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
  
        #block5
        layer_config_list.append({'type': 'conv', 'in_channels':112 , 'out_channels':672 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':672 , 'out_channels':672 , 'kernel_size': 5, 'padding': 2,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 10, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 672, 'out_features': 28})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 28, 'out_features': 672})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':672 , 'out_channels':112 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
  
       
        #block6
        layer_config_list.append({'type': 'conv', 'in_channels':112 , 'out_channels':672 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':672 , 'out_channels':672 , 'kernel_size': 5, 'padding': 2,'stride':2,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 6, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 672, 'out_features': 28})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 28, 'out_features': 672})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':672 , 'out_channels':192 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
   
        #block6
        layer_config_list.append({'type': 'conv', 'in_channels':192 , 'out_channels':1152 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels':1152 , 'kernel_size': 5, 'padding': 2,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 4, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1152, 'out_features': 48})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 48, 'out_features': 1152})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels':192 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
        
        #block6
        layer_config_list.append({'type': 'conv', 'in_channels':192 , 'out_channels':1152 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels':1152 , 'kernel_size': 5, 'padding': 2,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 3, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1152, 'out_features': 48})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 48, 'out_features': 1152})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels':192 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
     
        #block6
        layer_config_list.append({'type': 'conv', 'in_channels':192 , 'out_channels':1152 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels':1152 , 'kernel_size': 5, 'padding': 2,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 3, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1152, 'out_features': 48})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 48, 'out_features': 1152})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels':192 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -15]})
      
        
        #block7
        layer_config_list.append({'type': 'conv', 'in_channels':192 , 'out_channels':1152 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels':1152 , 'kernel_size': 3, 'padding': 1,'stride':1,'depthwise':'separable'})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 7, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 1152, 'out_features': 48})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'fc', 'in_features': 48, 'out_features': 1152})
        layer_config_list.append({'type': 'Sigmoid'})
        layer_config_list.append({'type': 'element_multiply', 'input_index': [-1, -7]}) 
        layer_config_list.append({'type': 'conv', 'in_channels':1152 , 'out_channels': 320, 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
      
        
        #end
        layer_config_list.append({'type': 'conv', 'in_channels':320 , 'out_channels':1280 , 'kernel_size': 1})
        layer_config_list.append({'type': 'Swish'})
        layer_config_list.append({'type': 'pooling', 'mode': 'ADA', 'kernel_size': 11, 'stride': 1})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'dropout'})
        layer_config_list.append({'type': 'fc', 'in_features': 1280, 'out_features': num_classes})
    
    else:
        assert 0, f'not support {cate}'
    
    for i in range(len(layer_config_list)):
        if layer_config_list[i]['type'] in {'conv', 'fc'}:
            # Mark weighted layers for downstream consumers; Phase 1 still uses
            # shape/resource metadata only and does not execute SNN runtime logic.
            layer_config_list[i]['snn_mode'] = ('snn' in cate)
        quantize_config_list.append({'weight_bit': weight_quantization_bit, 'activation_bit':9, 'point_shift': -2})
        if 'input_index' in layer_config_list[i]:
            input_index_list.append(layer_config_list[i]['input_index'])
        else:
            input_index_list.append([-1])
                # by default: the inputs of the current layer come from the outputs of the previous layer
    if 'snn' in cate:
        input_channels = 6 if cate.endswith('_v7_phase6') else 2
        input_params = {'activation_scale': 1.0, 'activation_bit': 9, 'input_shape': (1, input_channels, 32, 32)}
    else:
        input_params = {'activation_scale': 1. / 255., 'activation_bit': 9, 'input_shape': (1, 3, 32, 32)}
        # change the input_shape according to datasets
    if hardware_config is not None and 'input_shape' in hardware_config:
        input_params['input_shape'] = tuple(hardware_config['input_shape'])
    #add bn for every conv
    L = len(layer_config_list)
    for i in range(L-1, -1, -1):
        if cate in {
            'resnet18_snn_v7_phase6', 'resnet34_snn_v7_phase6',
            'resnet50_snn_v7_phase6', 'inception_snn_v7_phase6',
        }:
            break  # These graphs contain every LIF boundary explicitly.
        if layer_config_list[i]['type'] == 'conv':
            # ANN keeps the original BN-after-conv behavior; project SNN uses
            # registered LIF hook layers, and swbn keeps both BN and LIF.
            if cate.startswith('vgg8_snn_swbn') or cate == 'mobilenet_snn_v7_phase6':
                inserted_layers = [
                    {'type': 'bn', 'features': layer_config_list[i]['out_channels']},
                    _default_snn_lif_layer_config(),
                ]
            elif 'snn' in cate:
                inserted_layers = [_default_snn_lif_layer_config()]
            else:
                inserted_layers = [{'type': 'bn', 'features': layer_config_list[i]['out_channels']}]
            insert_at = i + 1
            for layer_offset, inserted_layer in enumerate(inserted_layers):
                layer_config_list.insert(insert_at + layer_offset, inserted_layer)
                quantize_config_list.insert(insert_at + layer_offset, {'weight_bit': weight_quantization_bit, 'activation_bit': 9, 'point_shift': -2})
                input_index_list.insert(insert_at + layer_offset, [-1])
            inserted_count = len(inserted_layers)
            for j in range(i + 1 + inserted_count, len(layer_config_list), 1):
                for relative_input_index in range(len(input_index_list[j])):
                    if j + input_index_list[j][relative_input_index] < i + 1:
                        input_index_list[j][relative_input_index] -= inserted_count
   
    # generate net
    
    net = NetworkGraph(hardware_config, layer_config_list, quantize_config_list, input_index_list, input_params)
    net.network_module = cate
    if 'snn' in cate:
        net.sequence_length = int(os.environ.get('MNSIM_SNN_SEQUENCE_LENGTH', '30'))
    return net

if __name__ == '__main__':
    assert len(sys.argv) == 3
    net = get_net(cate = sys.argv[1], num_classes = int(sys.argv[2]))
    print(net)
    for name, param in net.named_parameters():
        print(name, type(param.data), param.size())
    print(f'this is network input shape {net.input_params["input_shape"]}，output shape {net.layer_list[-1].layer_config["out_features"]}')
