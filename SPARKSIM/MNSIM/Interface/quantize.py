#-*-coding:utf-8-*-
import collections
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from MNSIM.Accuracy_Model.snn_accuracy_contract import BinarySpikeDomain
from MNSIM.Interface.qat_quantization import activation_range, mu_law_activation_ste
from MNSIM.Scheduling.non_weighted_layers import (
    LIFNode,
    REGISTERED_MNSIM_NON_WEIGHTED_LAYER_TYPES,
    get_non_weighted_layer_spec,
)
#
# last_activation_scale, last_weight_scale for activation and weight scale in last calculation
# last_activation_bit, last_weight_bit for activation and weight bit last time
last_activation_scale = None
last_weight_scale = None
last_activation_bit = None


def _adc_quantize(pre_quantized, lower_bound, upper_bound, hardware_config):
    """Quantize an ADC code using uniform or configured non-uniform bins."""
    mode = str(hardware_config.get('ADC_quantization_mode', 'UNIFORM')).upper()
    if mode == 'READC_LLOYD_MAX':
        collector = hardware_config.get('ADC_readc_collector')
        if collector is not None:
            collector.add(pre_quantized)
            return pre_quantized
        table = hardware_config.get('ADC_readc_table')
        if table is None:
            raise RuntimeError('READC_LLOYD_MAX requires offline per-layer calibration')
        from MNSIM.Accuracy_Model.readc import readc_quantize
        return readc_quantize(pre_quantized, table)
    if mode == 'UNIFORM':
        return torch.clamp(torch.round(pre_quantized), lower_bound, upper_bound)
    if mode not in {'ADAPTIVE', 'STATIC_NONUNIFORM'}:
        raise ValueError(f'unsupported ADC quantization mode: {mode}')
    thresholds = hardware_config.get('ADC_adaptive_thresholds')
    expected = int(upper_bound - lower_bound)
    if thresholds is None or len(thresholds) != expected:
        raise ValueError(
            'ADC_adaptive_thresholds must contain exactly '
            f'{expected} strictly increasing bin boundaries'
        )
    if any(float(a) >= float(b) for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError('ADC_adaptive_thresholds must be strictly increasing')
    boundaries = torch.as_tensor(thresholds, device=pre_quantized.device, dtype=pre_quantized.dtype)
    return torch.bucketize(pre_quantized, boundaries) + int(lower_bound)
last_weight_bit = None


@dataclass(frozen=True)
class ActivationQuantizationState:
    """The code-space definition of one graph activation node.

    ``scale`` is the value represented by one signed integer code step, not
    the full-scale range. Keeping this state with the graph value avoids
    inferring an input scale from whichever layer happened to execute last.
    """

    bit: int
    scale: float

    def __post_init__(self):
        if int(self.bit) < 2:
            raise ValueError(f"activation bit width must be at least 2, got {self.bit}")
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0:
            raise ValueError(
                f"activation code scale must be finite and positive, got {self.scale}"
            )


@dataclass(frozen=True)
class NonUniformActivationQuantizationState(ActivationQuantizationState):
    """Mu-law codebook; inherited scale is nominal, not a linear code step.

    Actual levels are sign(k)*expm1(abs(k)/threshold*log1p(mu))/mu*full_scale.
    Linear PIM code reconstruction must not consume this state.
    """

    mu: float = 15.0
    scheme: str = "mu_law"

    def __post_init__(self):
        super().__post_init__()
        if self.scheme != 'mu_law' or not math.isfinite(self.mu) or self.mu <= 0:
            raise ValueError('invalid non-uniform activation codebook')


def configured_activation_state(config, full_scale):
    state = activation_quantization_state(config['activation_bit'], full_scale)
    if config.get('activation_quantization', 'uniform') == 'mu_law':
        return NonUniformActivationQuantizationState(state.bit, state.scale, float(config.get('activation_mu', 15.0)))
    return state


def configured_activation_quantize(input, config, last_value, training):
    mode = config.get('activation_quantization', 'uniform')
    if mode == 'uniform':
        return Quantize(input, config['activation_bit'], 'activation', last_value, training)
    if mode == 'mu_law':
        return mu_law_activation_ste(input, config['activation_bit'], last_value, training, float(config.get('activation_mu', 15.0)))
    raise ValueError(f'unsupported activation quantization: {mode}')


def activation_quantization_state(bit, full_scale):
    """Build a node state from its calibrated full-scale range."""

    bit = int(bit)
    threshold = 2 ** (bit - 1) - 1
    if threshold <= 0:
        raise ValueError(f"activation bit width must be at least 2, got {bit}")
    full_scale = float(full_scale)
    if not math.isfinite(full_scale) or full_scale <= 0:
        raise RuntimeError(
            "activation scale is not calibrated; run QAT or explicit activation "
            "calibration before eval/PIM inference"
        )
    return ActivationQuantizationState(bit=bit, scale=full_scale / threshold)


def _legacy_activation_quantization_state():
    if last_activation_bit is None or last_activation_scale is None:
        raise RuntimeError(
            "no activation quantization state is available; execute through "
            "NetworkGraph or pass input_quantization explicitly"
        )
    return ActivationQuantizationState(
        bit=int(last_activation_bit), scale=float(last_activation_scale)
    )


def _publish_legacy_activation_quantization_state(state):
    """Keep direct legacy layer calls working without using this in graph execution."""

    global last_activation_bit
    global last_activation_scale
    if isinstance(state, NonUniformActivationQuantizationState):
        last_activation_bit = last_activation_scale = None
        return
    last_activation_bit = state.bit
    last_activation_scale = state.scale
# quantize Function
class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, input, qbit, mode, last_value = None, training = None):
        global last_weight_scale
        global last_activation_scale
        global last_weight_bit
        global last_activation_bit
        # ``training`` is also used by the explicit calibration path. Eval must
        # consume a previously calibrated range instead of adapting to the
        # current test batch.
        if mode == 'weight':
            last_weight_bit = qbit
            scale = torch.max(torch.abs(input)).item()
        elif mode == 'activation':
            last_activation_bit = qbit
            scale = activation_range(input, last_value, training)
            
        else:
            assert 0, f'not support {mode}'
        # transfer
        thres = 2 ** (qbit - 1) - 1
        output = input / scale
        output = torch.clamp(torch.round(output * thres), 0 - thres, thres - 0)
        output = output * scale / thres
        if mode == 'weight':
            last_weight_scale = scale / thres
        elif mode == 'activation':
            last_activation_scale = scale / thres
        else:
            assert 0, f'not support {mode}'
        return output
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
Quantize = QuantizeFunction.apply


def quantize_weight_ste(input: torch.Tensor, qbit: int):
    """Quantize a trainable weight while retaining a straight-through gradient."""
    threshold = 2 ** (int(qbit) - 1) - 1
    if threshold <= 0:
        raise ValueError(f'qbit must be at least 2, got {qbit}')
    scale = max(float(input.detach().abs().max().item()), float(torch.finfo(input.dtype).eps))
    lsb = scale / float(threshold)
    digit = torch.clamp(torch.round(input / lsb), -threshold, threshold)
    quantized = input + (digit * lsb - input).detach()
    return quantized, digit.to(torch.int64), float(lsb)


def decode_prequantized_weight_splits(weight_splits, *, qbit: int):
    """Validate and decode one logical layer's persisted signed-grid weights."""
    if not weight_splits:
        raise ValueError("prequantized weight_splits must not be empty")
    if int(qbit) < 2:
        raise ValueError(f"qbit must be >=2 for signed quantization, got {qbit}")

    first = weight_splits[0]
    if not torch.is_tensor(first) or not first.is_floating_point():
        raise TypeError("prequantized weights must be floating-point tensors")
    dtype = first.dtype
    max_abs = 0.0
    for split_index, weight in enumerate(weight_splits):
        if not torch.is_tensor(weight) or not weight.is_floating_point():
            raise TypeError(
                f"prequantized weight split {split_index} must be a floating-point tensor"
            )
        if weight.dtype != dtype:
            raise ValueError("prequantized weight splits must share one dtype")
        if not torch.isfinite(weight).all():
            raise FloatingPointError(
                f"prequantized weight split {split_index} contains non-finite values"
            )
        max_abs = max(max_abs, float(weight.detach().abs().max().item()))

    threshold = 2 ** (int(qbit) - 1) - 1
    lsb = max(max_abs, float(torch.finfo(dtype).eps)) / float(threshold)
    digits = []
    for split_index, weight in enumerate(weight_splits):
        detached = weight.detach()
        digit = torch.clamp(
            torch.round(detached / lsb), -threshold, threshold
        )
        reconstructed = (digit * lsb).to(dtype=dtype)
        if not torch.equal(reconstructed, detached):
            max_error = float((reconstructed - detached).abs().max().item())
            raise ValueError(
                f"weight split {split_index} is not on the signed {int(qbit)}-bit grid; "
                f"max_abs_error={max_error:.6e} lsb={lsb:.6e}"
            )
        digits.append(digit.to(dtype=torch.int64))
    return digits, float(lsb)


def conductance_lookup_ste(reference: torch.Tensor, looked_up: torch.Tensor):
    if reference.shape != looked_up.shape:
        raise ValueError('conductance lookup shape mismatch')
    # Avoid float32 cancellation: physical conductances may be ~1e-11 S while
    # the reference weight is O(1). Preserve the looked-up forward value and
    # retain an identity gradient through the reference tensor.
    return (reference - reference.detach()) + looked_up.to(reference.dtype)


def _record_adc_diagnostic(diagnostic, pre_quantized, lower_bound, upper_bound):
    """Accumulate saturation counts without changing the ADC calculation."""

    rounded = torch.round(pre_quantized)
    finite = torch.isfinite(rounded)
    finite_count = int(finite.sum().item())
    diagnostic['conversion_count'] += int(rounded.numel())
    diagnostic['nonfinite_count'] += int(rounded.numel()) - finite_count
    if finite_count == 0:
        return

    finite_values = rounded[finite]
    diagnostic['lower_saturation_count'] += int(
        (finite_values < lower_bound).sum().item()
    )
    diagnostic['upper_saturation_count'] += int(
        (finite_values > upper_bound).sum().item()
    )
    current_min = float(finite_values.min().item())
    current_max = float(finite_values.max().item())
    if diagnostic['pre_quantized_min'] is None:
        diagnostic['pre_quantized_min'] = current_min
        diagnostic['pre_quantized_max'] = current_max
    else:
        diagnostic['pre_quantized_min'] = min(
            diagnostic['pre_quantized_min'], current_min
        )
        diagnostic['pre_quantized_max'] = max(
            diagnostic['pre_quantized_max'], current_max
        )


def _record_input_diagnostic(diagnostic, pre_quantized, lower_bound, upper_bound):
    """Record activation-code clipping before the bit-slice calculation."""

    rounded = torch.round(pre_quantized)
    finite = torch.isfinite(rounded)
    finite_count = int(finite.sum().item())
    diagnostic['input_code_count'] += int(rounded.numel())
    diagnostic['input_nonfinite_count'] += int(rounded.numel()) - finite_count
    if finite_count == 0:
        return

    finite_values = rounded[finite]
    diagnostic['input_lower_saturation_count'] += int(
        (finite_values < lower_bound).sum().item()
    )
    diagnostic['input_upper_saturation_count'] += int(
        (finite_values > upper_bound).sum().item()
    )
    current_min = float(finite_values.min().item())
    current_max = float(finite_values.max().item())
    if diagnostic['input_pre_quantized_min'] is None:
        diagnostic['input_pre_quantized_min'] = current_min
        diagnostic['input_pre_quantized_max'] = current_max
    else:
        diagnostic['input_pre_quantized_min'] = min(
            diagnostic['input_pre_quantized_min'], current_min
        )
        diagnostic['input_pre_quantized_max'] = max(
            diagnostic['input_pre_quantized_max'], current_max
        )


# AB = 1
# N = 512
# Q = 10
# # METHOD = 'TRADITION'
# # METHOD = 'FIX_TRAIN'
# # METHOD = 'SINGLE_FIX_TEST'
# METHOD = ''
# quantize layer, include conv and fc without bias
class QuantizeLayer(nn.Module):
    def __init__(self, hardware_config, layer_config, quantize_config):
        super(QuantizeLayer, self).__init__()
        # load hardware layer and quantize config in setting
        self.hardware_config = copy.deepcopy(hardware_config)
        self.layer_config = copy.deepcopy(layer_config)
        self.quantize_config = copy.deepcopy(quantize_config)
        # split weights
        if self.layer_config['type'] == 'conv':
            assert 'in_channels' in self.layer_config.keys()
            channel_N = (self.hardware_config['xbar_size'] // (self.layer_config['kernel_size'] ** 2))
                # number of channels stored in one xbar
            complete_bar_num = self.layer_config['in_channels'] // channel_N
            residual_col_num = self.layer_config['in_channels'] % channel_N
            
            in_channels_list = []
                # each element stores the channel number in one xbar
            if residual_col_num > 0:
                in_channels_list = [channel_N] * complete_bar_num + [residual_col_num]
            else:
                in_channels_list = [channel_N] * complete_bar_num
            
            #add addtion quantization bit
            self.layer_config['extend_ADC_bitwidth'] = max(math.ceil(math.log2(channel_N*self.layer_config['kernel_size']**2/self.hardware_config['DAC_num'])),0)
                # extend_ADC_bitwidth: each xbar can only activate part of the rows and complete ADC quantization at a time, requiring adding these partial results of different rows, equivalent to increasing the bitwidth of the ADC.
                # temp = channel_N*self.layer_config['kernel_size']**2: number of used row in one xbar
                # temp/self.hardware_config['DAC_num']: cycles to compute all rows of one xbar

            # generate Module List
            assert 'out_channels' in self.layer_config.keys()
            assert 'kernel_size' in self.layer_config.keys()
            if 'stride' not in self.layer_config.keys():
                self.layer_config['stride'] = 1
            if 'padding' not in self.layer_config.keys():
                self.layer_config['padding'] = 0
            if 'depthwise'not in self.layer_config.keys():
                self.layer_config['depthwise']='normal'
            if self.layer_config['depthwise']=='separable':
                in_channels_list=[self.layer_config['in_channels']]
                self.sublayer_list = nn.ModuleList([nn.Conv2d(
                i, i, self.layer_config['kernel_size'],
                stride = self.layer_config['stride'], padding = self.layer_config['padding'], dilation = 1, groups = i, bias = False
                )
                for i in in_channels_list]) 
            else:
                self.sublayer_list = nn.ModuleList([nn.Conv2d(
                i, self.layer_config['out_channels'], self.layer_config['kernel_size'],
                stride = self.layer_config['stride'], padding = self.layer_config['padding'], dilation = 1, groups = 1, bias = False
                )
                for i in in_channels_list])   
        
            if self.layer_config['depthwise']=='separable':
                self.split_input=self.layer_config['in_channels']
            else:    
                self.split_input = channel_N
        
        elif self.layer_config['type'] == 'fc':
            assert 'in_features' in self.layer_config.keys()
            complete_bar_num = self.layer_config['in_features'] // self.hardware_config['xbar_size']
            residual_col_num = self.layer_config['in_features'] % self.hardware_config['xbar_size']
            if residual_col_num > 0:
                in_features_list = [self.hardware_config['xbar_size']] * complete_bar_num + [residual_col_num]
            else:
                in_features_list = [self.hardware_config['xbar_size']] * complete_bar_num

            #add addtion quantization bit
            self.layer_config['extend_ADC_bitwidth'] = max(0,math.ceil(math.log2(min(self.layer_config['in_features'], self.hardware_config['xbar_size'])/self.hardware_config['DAC_num'])))

            # generate Module List
            assert 'out_features' in self.layer_config.keys()
            self.sublayer_list = nn.ModuleList([nn.Linear(i, self.layer_config['out_features'], None) for i in in_features_list])
            self.split_input = self.hardware_config['xbar_size']
        else:
            assert 0, f'not support {self.layer_config["type"]}'
        # self.last_value = nn.Parameter(torch.ones(1))
        
        self.register_buffer('last_value', (-1) * torch.ones(1))
 
        # self.last_value[0] = 1
        self.register_buffer('bit_scale_list', torch.FloatTensor([
            [quantize_config['activation_bit'], -1],
            [quantize_config['weight_bit'], -1],
            [quantize_config['activation_bit'], -1]
        ]))
        # layer information
        self.layer_info = None
        self._activation_calibration_enabled = False

    def set_activation_calibration(self, enabled):
        self._activation_calibration_enabled = bool(enabled)

    def _activation_scale_update_enabled(self):
        return self.training or self._activation_calibration_enabled

    def output_activation_quantization(self, input_quantization=None):
        return configured_activation_state(self.quantize_config, self.last_value.item())

    def _resolve_input_quantization(self, input_quantization):
        if input_quantization is None:
            return _legacy_activation_quantization_state()
        if not isinstance(
            input_quantization, (ActivationQuantizationState, BinarySpikeDomain)
        ):
            raise TypeError(
                'input_quantization must be an ActivationQuantizationState, '
                'BinarySpikeDomain, or None'
            )
        return input_quantization

    def structure_forward(self, input):
        # TRADITION
        # get the layer structure
        input_shape = input.shape
        input_list = torch.split(input, self.split_input, dim = 1)
        output = None
        for i in range(len(self.sublayer_list)):
            if i == 0:
                output = self.sublayer_list[i](input_list[i])

            else:
                output.add_(self.sublayer_list[i](input_list[i]))
        output_shape = output.shape
        # layer_info
        # only conv and fc are QuantizeLayer
        self.layer_info = collections.OrderedDict()
        # self.layer_info['name'] = self.layer_config['name']
        if self.layer_config['type'] == 'conv':
            self.layer_info['type'] = 'conv'
            self.layer_info['Inputchannel'] = int(input_shape[1])
            self.layer_info['Inputsize'] = list(input_shape[2:])
            self.layer_info['Kernelsize'] = self.layer_config['kernel_size']
            self.layer_info['Stride'] = self.layer_config['stride']
            self.layer_info['Padding'] = self.layer_config['padding']
            self.layer_info['Depthwise']=self.layer_config['depthwise']
            self.layer_info['Outputchannel'] = int(output_shape[1])
            self.layer_info['Outputsize'] = list(output_shape[2:])
        elif self.layer_config['type'] == 'fc':
            self.layer_info['type'] = 'fc'
            self.layer_info['Infeature'] = int(input_shape[1])
            self.layer_info['Outfeature'] = int(output_shape[1])
            self.layer_info['Outputchannel'] = int(output_shape[1])
            self.layer_info['Outputsize'] = list([1,1])
        else:
            assert 0, f'not support {self.layer_config["type"]}'
        self.layer_info['Inputbit'] = int(self.bit_scale_list[0,0].item())
        self.layer_info['Weightbit'] = int(self.quantize_config['weight_bit'])
        self.layer_info['outputbit'] = int(self.quantize_config['activation_bit'])
        self.layer_info['row_split_num'] = len(self.sublayer_list)
        if self.hardware_config['xbar_polarity'] == 2:
            self.layer_info['weight_bit_split_part'] = math.ceil((self.quantize_config['weight_bit'] - 1) / (self.hardware_config['weight_bit']))
        else:
            self.layer_info['weight_bit_split_part'] = math.ceil((self.quantize_config['weight_bit']) / (self.hardware_config['weight_bit']))
        if 'input_index' in self.layer_config:
           
            self.layer_info['Inputindex'] = self.layer_config['input_index']
        else:
            self.layer_info['Inputindex'] = [-1]
        self.layer_info['Outputindex'] = [1]
        return output
    def forward(
        self,
        input,
        method = 'SINGLE_FIX_TEST',
        adc_action = 'SCALE',
        input_quantization=None,
    ):
        METHOD = method
       
        # float method
        if METHOD == 'TRADITION':
            input_list = torch.split(input, self.split_input, dim = 1)
            output = None
            for i in range(len(self.sublayer_list)):
                if i == 0:
                    output = self.sublayer_list[i](input_list[i])
                else:
                    output.add_(self.sublayer_list[i](input_list[i]))
            return output
        # fix training
        if METHOD == 'FIX_TRAIN':
           
            weight=torch.cat([l.weight for l in self.sublayer_list], dim = 1)
            
            # quantize weight
            #weight = torch.cat([l.weight for l in self.sublayer_list], dim = 1)
            global last_weight_scale
            global last_weight_bit
            # The graph owns the source state. The global fallback exists only
            # for callers that still invoke one layer directly.
            input_state = self._resolve_input_quantization(input_quantization)
            # V7 SNN inputs are an explicit binary spike domain rather than
            # an ActivationQuantizationState. Keep its metadata representable
            # in the legacy bit/scale buffer while bypassing integer fake
            # quantization of the already binary input tensor.
            if isinstance(input_state, BinarySpikeDomain):
                self.bit_scale_list.data[0, 0] = 1
                self.bit_scale_list.data[0, 1] = input_state.scale
            else:
                self.bit_scale_list.data[0, 0] = input_state.bit
                self.bit_scale_list.data[0, 1] = input_state.scale
            weight = Quantize(weight, self.quantize_config['weight_bit'], 'weight', None, None)
            
            # weight bit and scale
            self.bit_scale_list.data[1, 0] = last_weight_bit
            self.bit_scale_list.data[1, 1] = last_weight_scale
            if self.layer_config['type'] == 'conv':
                
                if self.layer_config['depthwise']=='normal':
                    output = F.conv2d(
                        input, weight, None, \
                        self.layer_config['stride'], self.layer_config['padding'], 1, 1
                    )
                elif self.layer_config['depthwise']=='point':
                    output = F.conv2d(
                        input, weight, None, \
                        self.layer_config['stride'], self.layer_config['padding'], 1,1
                    )
                elif self.layer_config['depthwise']=='separable':
                    output = F.conv2d(
                        input, weight, None, \
                        self.layer_config['stride'], self.layer_config['padding'], 1, input.shape[1]
                    )
                else :
                    assert 0, f'not support depthwise'
            elif self.layer_config['type'] == 'fc':
                output = F.linear(input, weight, None)
            else:
                assert 0, f'not support {self.layer_config["type"]}'
            
            output = configured_activation_quantize(
                output,
                self.quantize_config,
                self.last_value,
                self._activation_scale_update_enabled(),
            )

            # Output state belongs to this node, independent of the input
            # node's bit width or scale.
            output_state = self.output_activation_quantization()
            self.bit_scale_list.data[2, 0] = output_state.bit
            self.bit_scale_list.data[2, 1] = output_state.scale
            return output
        if METHOD == 'SINGLE_FIX_TEST':
            assert self.training == False
            bit_weights = self.get_bit_weights()
            output = self.set_weights_forward(
                input,
                bit_weights,
                adc_action,
                input_quantization=input_quantization,
            )
            return output
        assert 0, f'not support {METHOD}'
    #calculate_equal_bit:CNNParted Interfaces
    def calculate_equal_bit(self):
        #R:active Rows
        if layer_config['type']=='conv':
            k=self.layer_config['kernel_size']
            Cin=self.layer_config['in_channels']
            R=self.hardware_config['DAC_num']
            Padc=self.hardware_config['ADC_quantize_bit']
            Pout_equal=int(max(math.log2(k**2*Cin/R),0))+Padc
        elif layer_config['type']=='fc':
            inputchannel=self.layer_config['in_features']
            R=self.hardware_config['DAC_num']
            Pout_equal=int(max(math.log2(inputchannel/R),0))+Padc
        
        return Pout_equal
    def get_bit_weights(self, *, use_prequantized_weights=False):
        weight_bit = self.quantize_config['weight_bit']
        weight_scale = self.bit_scale_list[1, 1].item()
        assert weight_bit != 0, f'weight bit should be given by the params'
        bit_weights = collections.OrderedDict()
        prequantized_digits = None
        if use_prequantized_weights:
            prequantized_digits, _ = decode_prequantized_weight_splits(
                [layer.weight for layer in self.sublayer_list],
                qbit=int(weight_bit),
            )
        else:
            assert weight_scale != 0, f'weight scale should be given by the params'
        for layer_num, l in enumerate(self.sublayer_list):
            
            # assert (weight_bit - 1) % self.hardware_config['weight_bit'] == 0, generate weight part
            if self.hardware_config['xbar_polarity'] == 2:
                weight_bit_split_part = math.ceil((weight_bit - 1) / self.hardware_config['weight_bit'])
                    # weight_bit-1: pos and neg xbar, split weights into multiple part
            else:
                weight_bit_split_part = math.ceil(weight_bit / self.hardware_config['weight_bit'])
            # transfer part weight
            thres = 2 ** (weight_bit - 1) - 1
            if prequantized_digits is None:
                weight_digit = torch.clamp(
                    torch.round(l.weight / weight_scale), 0 - thres, thres - 0
                )
            else:
                weight_digit = prequantized_digits[layer_num]
            # split weight into bit
            sign_weight = torch.sign(weight_digit)
            weight_digit = torch.abs(weight_digit)
            base = 1
            step = 2 ** self.hardware_config['weight_bit']
            for j in range(weight_bit_split_part):
                tmp = torch.fmod(weight_digit, base * step) - torch.fmod(weight_digit, base)
                tmp = torch.mul(sign_weight, tmp)
                tmp = copy.deepcopy((tmp / base).detach().cpu().numpy())
                if self.hardware_config['xbar_polarity'] == 2:
                    # use one pos xbar and one neg xbar to store one weight value
                    bit_weights[f'split{layer_num}_weight{j}_positive'] = np.where(tmp > 0, tmp, 0)
                    bit_weights[f'split{layer_num}_weight{j}_negative'] = np.where(tmp < 0, -tmp, 0)
                else:
                    # use one xbar to store one weight value
                    bit_weights[f'split{layer_num}_weight{j}'] = tmp
                base = base * step
            
        return bit_weights
    def set_weights_forward(
        self,
        input,
        bit_weights,
        adc_action,
        adc_diagnostics=None,
        layer_index=None,
        input_quantization=None,
        numerical_mode='ADC',
    ):
        assert self.training == False
        if numerical_mode not in {'ADC', 'NO_ADC_RECONSTRUCTION', 'LEGACY_NONE'}:
            raise ValueError(
                f'unsupported weighted numerical mode: {numerical_mode}'
            )
        if numerical_mode == 'NO_ADC_RECONSTRUCTION' and adc_action is not None:
            raise ValueError(
                'NO_ADC_RECONSTRUCTION requires adc_action=None'
            )
        output = None
        output_final=None
        adc_diagnostic = None
        if adc_diagnostics is not None:
            adc_diagnostic = {
                'network_layer_index': layer_index,
                'layer_type': self.layer_config['type'],
                'adc_action': adc_action,
                'conversion_count': 0,
                'nonfinite_count': 0,
                'lower_saturation_count': 0,
                'upper_saturation_count': 0,
                'pre_quantized_min': None,
                'pre_quantized_max': None,
                'input_code_count': 0,
                'input_nonfinite_count': 0,
                'input_lower_saturation_count': 0,
                'input_upper_saturation_count': 0,
                'input_pre_quantized_min': None,
                'input_pre_quantized_max': None,
            }
        
        input_state = self._resolve_input_quantization(input_quantization)
        if (isinstance(input_state, NonUniformActivationQuantizationState)
                or self.quantize_config.get('activation_quantization', 'uniform') != 'uniform'):
            raise ValueError('non-uniform QAT activations require an explicit hardware codebook decoder; linear PIM reconstruction is unsupported')
        output_state = (
            None
            if numerical_mode == 'NO_ADC_RECONSTRUCTION'
            else self.output_activation_quantization()
        )
        input_list = torch.split(input, self.split_input, dim = 1)
            # self.split_input = xbar_size
        
        scale = self.last_value.item()
        
        # weight_bit = int(self.bit_scale_list[1, 0].item())
        weight_bit = self.quantize_config['weight_bit']
        weight_scale = self.bit_scale_list[1, 1].item()
        for layer_num, l in enumerate(self.sublayer_list):
            # Depthwise separable layers use a temporary accumulator when
            # partial xbar outputs have different channel shapes. Initialize
            # it for every sublayer so the single-sublayer case is valid too.
            tmp_buffer = None
            # assert (weight_bit - 1) % self.hardware_config['weight_bit'] == 0, generate weight cycle
            if self.hardware_config['xbar_polarity'] == 2:
                weight_bit_split_part = math.ceil((weight_bit - 1) / self.hardware_config['weight_bit'])
                    # weight_bit-1: pos and neg xbar, split weights into multiple part
            else:
                weight_bit_split_part = math.ceil(weight_bit / self.hardware_config['weight_bit'])

            weight_container = []
            base = 1
            step = 2 ** self.hardware_config['weight_bit']
            for j in range(weight_bit_split_part):
                if self.hardware_config['xbar_polarity'] == 2:
                    tmp = bit_weights[f'split{layer_num}_weight{j}_positive'] - bit_weights[f'split{layer_num}_weight{j}_negative']
                else:
                    tmp = bit_weights[f'split{layer_num}_weight{j}']
                tmp = torch.from_numpy(tmp)
                weight_container.append(tmp.to(device = input.device, dtype = input.dtype))
                base = base * step
            if isinstance(input_state, BinarySpikeDomain):
                activation_in_scale = input_state.scale
                activation_in_cycle = input_state.activation_cycle
                activation_in_container = [
                    input_state.validate(input_list[layer_num])
                ]
                if adc_diagnostic is not None:
                    _record_input_diagnostic(
                        adc_diagnostic, input_list[layer_num], 0, 1
                    )
            else:
                activation_in_bit = input_state.bit
                activation_in_scale = input_state.scale
                thres = 2 ** (activation_in_bit - 1) - 1
                input_pre_quantized = input_list[layer_num] / activation_in_scale
                if adc_diagnostic is not None:
                    _record_input_diagnostic(
                        adc_diagnostic,
                        input_pre_quantized,
                        -thres,
                        thres,
                    )
                activation_in_digit = torch.clamp(
                    torch.round(input_pre_quantized), -thres, thres
                )
                # assert (activation_in_bit - 1) % self.hardware_config['input_bit'] == 0, generate activation_in cycle
                activation_in_cycle = math.ceil((activation_in_bit - 1) / self.hardware_config['input_bit'])
                # split activation into bit
                sign_activation_in = torch.sign(activation_in_digit)
                activation_in_digit = torch.abs(activation_in_digit)
                base = 1
                step = 2 ** self.hardware_config['input_bit']
                activation_in_container = []
                for i in range(activation_in_cycle):
                    tmp = torch.fmod(activation_in_digit, base * step) -  torch.fmod(activation_in_digit, base)
                    activation_in_container.append(torch.mul(sign_activation_in, tmp) / base)
                    base = base * step
            # calculation and add
            point_shift = math.floor(self.quantize_config['point_shift'] + 0.5 * math.log2(len(self.sublayer_list)))
            Q = self.hardware_config['ADC_quantize_bit'] + self.layer_config['extend_ADC_bitwidth']
            for i in range(activation_in_cycle):
                for j in range(weight_bit_split_part):
                    tmp = None
                    if self.layer_config['type'] == 'conv':
                        if 'depthwise'not in self.layer_config.keys():
                            self.layer_config['depthwise']='normal'
                        if self.layer_config['depthwise']=='normal':
                            tmp = F.conv2d(
                                activation_in_container[i], weight_container[j], None, \
                                self.layer_config['stride'], self.layer_config['padding'], 1, 1
                            )
                        elif self.layer_config['depthwise']=='separable':
                            tmp = F.conv2d(
                                activation_in_container[i], weight_container[j], None, \
                                self.layer_config['stride'], self.layer_config['padding'], 1, activation_in_container[i].shape[1]
                            )
                   
                    elif self.layer_config['type'] == 'fc':
                        tmp = F.linear(activation_in_container[i], weight_container[j], None)
                    else:
                        assert 0, f'not support {self.layer_config["type"]}'
                    if numerical_mode == 'NO_ADC_RECONSTRUCTION':
                        tmp = tmp * weight_scale * activation_in_scale
                        tmp = tmp / scale * (
                            2
                            ** (
                                (activation_in_cycle - 1)
                                * self.hardware_config['input_bit']
                                + (weight_bit_split_part - 1)
                                * self.hardware_config['weight_bit']
                            )
                        )
                    elif adc_action in {'SCALE', 'ADAPTIVE', 'READC_CALIBRATE'}:
                        tmp = tmp * weight_scale * activation_in_scale
                        tmp = tmp / scale * (2 ** ((activation_in_cycle - 1) * self.hardware_config['input_bit'] + \
                            (weight_bit_split_part - 1) * self.hardware_config['weight_bit']))
                        transfer_point = point_shift + (Q - 1)
                        # if self.hardware_config['type'] == 0:
                        pre_quantized = tmp * (2 ** transfer_point)
                        lower_bound = 1 - 2 ** (Q - 1)
                        upper_bound = 2 ** (Q - 1) - 1
                        if adc_diagnostic is not None:
                            _record_adc_diagnostic(
                                adc_diagnostic,
                                pre_quantized,
                                lower_bound,
                                upper_bound,
                            )
                        tmp = _adc_quantize(
                            pre_quantized, lower_bound, upper_bound, self.hardware_config
                        )
                        tmp = tmp / (2 ** transfer_point)
                        
                    elif adc_action == 'FIX':
                        # fix scale range
                        fix_scale_range = (2 ** self.hardware_config['input_bit'] - 1) * \
                                          (2 ** self.hardware_config['weight_bit'] - 1) * \
                                            self.hardware_config['xbar_size']
                        pre_quantized = tmp / fix_scale_range * (2 ** (Q - 1))
                        # if self.hardware_config['type'] == 0:
                        lower_bound = 1 - 2 ** (Q - 1)
                        upper_bound = 2 ** (Q - 1) - 1
                        if adc_diagnostic is not None:
                            _record_adc_diagnostic(
                                adc_diagnostic,
                                pre_quantized,
                                lower_bound,
                                upper_bound,
                            )
                        tmp = _adc_quantize(
                            pre_quantized, lower_bound, upper_bound, self.hardware_config
                        )
                        tmp = tmp * fix_scale_range / (2 ** (Q - 1))
                        tmp = tmp * weight_scale * activation_in_scale
                        tmp = tmp / scale * (2 ** ((activation_in_cycle - 1) * self.hardware_config['input_bit'] + \
                            (weight_bit_split_part - 1) * self.hardware_config['weight_bit']))
                    elif adc_action == 'NONE':
                        tmp = tmp * weight_scale * activation_in_scale
                        tmp = tmp / scale * (2 ** ((activation_in_cycle - 1) * self.hardware_config['input_bit'] + \
                            (weight_bit_split_part - 1) * self.hardware_config['weight_bit']))
                    else:
                        assert 0, f'can not support {adc_action}'
                    # scale
                    scale_point = (activation_in_cycle - 1 - i) * self.hardware_config['input_bit'] + \
                                  (weight_bit_split_part - 1 - j) * self.hardware_config['weight_bit']
                    tmp = tmp / (2 ** scale_point)
                    # add
                    
                    #tmp_buffer to execute depthwise convolution
                    if torch.is_tensor(output):
                        if self.layer_config['type'] == 'conv':
                            if self.layer_config['depthwise']=='separable':
                                if output.shape==tmp.shape:
                                    output = output + tmp
                                else:
                                    if flag==0:
                                        tmp_buffer=tmp
                                        flag=1
                                    tmp_buffer=tmp_buffer+tmp
                            else:
                                output = output + tmp
                                
                        else:
                            output = output + tmp
                    else:
                        output = tmp
            if self.layer_config['type'] == 'conv':
                if self.layer_config['depthwise']=='separable' :
                    if torch.is_tensor(output_final):
                        if tmp_buffer!=None:
                            output_final=torch.cat([output_final,tmp_buffer],dim=1)
                        else:
                            output_final=torch.cat([output_final,output],dim=1)
                    else:
                        if tmp_buffer!=None:
                            output_final=tmp_buffer
                        else:
                            output_final=output
                else:
                    output_final=output
            else:
                output_final=output
       
        if numerical_mode == 'NO_ADC_RECONSTRUCTION':
            output_final = output_final * scale
        elif numerical_mode == 'LEGACY_NONE' or adc_action == 'NONE':
            # Preserve the legacy no-ADC numerical audit, which compares
            # against the software activation quantizer.
            activation_out_bit = output_state.bit
            threshold = 2 ** (activation_out_bit - 1) - 1
            output_final = torch.clamp(
                torch.round(output_final * threshold), -threshold, threshold
            ) * output_state.scale
        else:
            # The ADC already performed the analog-to-digital conversion.
            # Restore full-scale after digital shift/add without quantizing the
            # same weighted result a second time.
            output_final = output_final * scale
        if adc_diagnostic is not None:
            adc_diagnostic['adc_quantize_bit'] = Q
            adc_diagnostic['point_shift'] = point_shift
            adc_diagnostic['activation_cycles'] = activation_in_cycle
            adc_diagnostic['weight_bit_splits'] = weight_bit_split_part
            adc_diagnostic['row_splits'] = len(self.sublayer_list)
            saturation_count = (
                adc_diagnostic['lower_saturation_count']
                + adc_diagnostic['upper_saturation_count']
            )
            adc_diagnostic['saturation_count'] = saturation_count
            conversion_count = adc_diagnostic['conversion_count']
            adc_diagnostic['saturation_fraction'] = (
                saturation_count / conversion_count if conversion_count else 0.0
            )
            input_saturation_count = (
                adc_diagnostic['input_lower_saturation_count']
                + adc_diagnostic['input_upper_saturation_count']
            )
            adc_diagnostic['input_saturation_count'] = input_saturation_count
            input_code_count = adc_diagnostic['input_code_count']
            adc_diagnostic['input_saturation_fraction'] = (
                input_saturation_count / input_code_count if input_code_count else 0.0
            )
            adc_diagnostics.append(adc_diagnostic)
        if output_state is not None:
            _publish_legacy_activation_quantization_state(output_state)
        return output_final

    def extra_repr(self):
        return str(self.hardware_config) + ' ' + str(self.layer_config) + ' ' + str(self.quantize_config)
QuantizeLayerStr = ['conv', 'fc']

class ViewLayer(nn.Module):
    def __init__(self):
        super(ViewLayer, self).__init__()
    def forward(self, x):
        x=x.view(x.size(0), -1)
        return x

class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super(ConcatLayer, self).__init__()
        self.dim = dim if isinstance(dim, str) else int(dim)
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError('concat requires at least two tensor inputs')
        tensors = list(x)
        if any(not torch.is_tensor(item) for item in tensors):
            raise TypeError('concat inputs must be tensors')
        reference = tensors[0]
        if self.dim == 'channel':
            dim = reference.dim() - 3
        else:
            dim = int(self.dim)
            if reference.dim() == 5 and dim == 1:
                dim = 2
        if any(item.dim() != reference.dim() for item in tensors):
            raise ValueError('concat inputs must have the same rank')
        for item in tensors[1:]:
            for axis, (left, right) in enumerate(zip(reference.shape, item.shape)):
                if axis != dim and left != right:
                    raise ValueError(
                        f'concat inputs differ on non-concat axis {axis}: '
                        f'{tuple(reference.shape)} vs {tuple(item.shape)}'
                    )
        return torch.cat(tensors, dim)


class EleSumLayer(nn.Module):
    def __init__(self):
        super(EleSumLayer, self).__init__()
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError('element_sum requires at least two tensor inputs')
        output = x[0]
        for value in x[1:]:
            if output.shape != value.shape:
                raise ValueError('element_sum inputs must have identical shapes')
            output = output + value
        return output

class EleMulLayer(nn.Module):
    def __init__(self):
        super(EleMulLayer, self).__init__()
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise ValueError('element_multiply requires at least two tensor inputs')
        values = list(x)
        first = values[0]
        # Preserve MNSIM's squeeze/excitation convention for a channel gate,
        # while allowing ordinary same-shape n-ary element multiplication.
        if first.dim() + 2 == values[1].dim() and first.shape == values[1].shape[:-2]:
            first = first.reshape(*first.shape, 1, 1)
        output = first
        for value in values[1:]:
            if output.shape != value.shape and torch.broadcast_shapes(
                output.shape, value.shape
            ) != value.shape:
                raise ValueError('element_multiply inputs must have identical shapes')
            output = output * value
        return output
def drop_path(x, prob: float = 0., training: bool = False):
    if prob==0 or not training:
        return x
    keep=1-prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep) * random_tensor
    return output
    

class droppath(nn.Module):
    def __init__(self,droppro=None):
        super(droppath,self).__init__()
        self.prob=droppro
    def forward(self,x):
        x=drop_path(x,self.prob,self.training)
        return x

class StraightLayer(nn.Module):
    def __init__(self, hardware_config, layer_config, quantize_config):
        super(StraightLayer, self).__init__()
        # load hardware layer and quantize config in setting
        self.hardware_config = copy.deepcopy(hardware_config)
        self.layer_config = copy.deepcopy(layer_config)
        self.quantize_config = copy.deepcopy(quantize_config)
        # generate layer
        if self.layer_config['type'] == 'pooling':
            assert 'kernel_size' in self.layer_config.keys()
            assert 'stride' in self.layer_config.keys()
            if 'padding' not in self.layer_config.keys():
                self.layer_config['padding'] = 0
            if self.layer_config['mode'] == 'AVE':
                
                self.layer = nn.AvgPool2d(
                    kernel_size = self.layer_config['kernel_size'], \
                    stride = self.layer_config['stride'], \
                    padding = self.layer_config['padding']
                )
               
            elif self.layer_config['mode'] == 'MAX':
                self.layer = nn.MaxPool2d(
                    kernel_size = self.layer_config['kernel_size'], \
                    stride = self.layer_config['stride'], \
                    padding = self.layer_config['padding']
                )
            elif self.layer_config['mode'] == 'ADA':
                self.layer = nn.AdaptiveAvgPool2d((1, 1))
            else:
                assert 0, f'not support {self.layer_config["mode"]}'
        elif self.layer_config['type'] == 'relu':
            self.layer = nn.ReLU()
        elif self.layer_config['type']=='Swish':
            self.layer=nn.SiLU()
        elif self.layer_config['type']=='Sigmoid':
            self.layer=nn.Sigmoid()
        elif self.layer_config['type'] == 'element_multiply':
            self.layer = EleMulLayer()
        elif self.layer_config['type'] == 'view':
            self.layer = ViewLayer()
        elif self.layer_config['type'] == 'bn':
            self.layer = nn.BatchNorm2d(self.layer_config['features'])
        elif self.layer_config['type'] == 'dropout':
            self.layer=nn.Dropout(0.2)
            #self.layer = droppath(0.2)
            
        elif self.layer_config['type'] == 'element_sum':
            self.layer = EleSumLayer()
        elif self.layer_config['type'] == 'concat':
            self.layer = ConcatLayer(self.layer_config.get('dim', 1))
        else:
            registered_spec = get_non_weighted_layer_spec(self.layer_config['type'])
            if registered_spec is not None:
                self.layer = registered_spec.create_straight_layer(self.layer_config)
            else:
                assert 0, f'not support {self.layer_config["type"]}'
        # ``bn`` and ``element_sum`` create a new activation-code domain. A
        # negative initial range forces a new graph to be calibrated before
        # eval/PIM inference instead of adapting scales to test data.
        initial_range = -1.0 if self.layer_config['type'] in {'bn', 'element_sum'} else 1.0
        self.register_buffer('last_value', initial_range * torch.ones(1))
        if self.layer_config['type'] == 'element_sum':
            # Older checkpoints stored the unused ``last_value`` buffer as
            # 1.0. This marker distinguishes them from checkpoints trained or
            # explicitly calibrated with post-add quantization enabled.
            self.register_buffer(
                'post_add_quantizer_version', torch.ones(1, dtype=torch.uint8)
            )
       
        # self.last_value[0] = 1
        self.layer_info = None
        self._activation_calibration_enabled = False

    def set_activation_calibration(self, enabled):
        self._activation_calibration_enabled = bool(enabled)

    def _activation_scale_update_enabled(self):
        return self.training or self._activation_calibration_enabled

    def output_activation_quantization(self, input_quantization=None):
        if self.layer_config['type'] in {'bn', 'element_sum'}:
            if (
                self.layer_config['type'] == 'element_sum'
                and not bool(self.post_add_quantizer_version.item())
            ):
                raise RuntimeError(
                    'checkpoint predates post-add activation quantization; run '
                    'NetworkGraph.calibrate_activation_scales or QAT before '
                    'eval/PIM inference'
                )
            return configured_activation_state(self.quantize_config, self.last_value.item())
        if isinstance(input_quantization, list):
            if len(input_quantization) != 1:
                return None
            return input_quantization[0]
        return input_quantization

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        if self.layer_config['type'] == 'element_sum':
            marker_key = prefix + 'post_add_quantizer_version'
            if marker_key not in state_dict:
                # Preserve strict checkpoint loading while making the legacy
                # state explicit and non-runnable until calibration.
                state_dict[marker_key] = torch.zeros_like(
                    self.post_add_quantizer_version
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    def structure_forward(self, input):
        # get the layer structure of non conv or fc layer
        #if self.layer_config['type'] != 'element_sum' and self.layer_config['type'] != 'concat':
        
        if self.layer_config['type'] != 'element_sum' and self.layer_config['type'] != 'concat' and self.layer_config['type'] != 'element_multiply':
            # generate input shape and output shape
            self.input_shape = input.shape
            output = self.layer.forward(input)
            self.output_shape = output.shape
            # generate layer_info
            self.layer_info = collections.OrderedDict()
            # self.layer_info['name'] = self.layer_config['name']
            if self.layer_config['type'] == 'pooling':
                self.layer_info['type'] = 'pooling'
                self.layer_info['Inputchannel'] = int(self.input_shape[1])
                self.layer_info['Inputsize'] = list(self.input_shape)[2:]
                self.layer_info['Kernelsize'] = self.layer_config['kernel_size']
                self.layer_info['Stride'] = self.layer_config['stride']
                self.layer_info['Padding'] = self.layer_config['padding']
                self.layer_info['Outputchannel'] = int(self.output_shape[1])
                self.layer_info['Outputsize'] = list(self.output_shape)[2:]
            elif self.layer_config['type'] == 'relu':
                self.layer_info['type'] = 'relu'
            elif self.layer_config['type']=='Swish':
                self.layer_info['type']= 'Swish'
            elif self.layer_config['type']=='Sigmoid':
                self.layer_info['type']= 'Sigmoid'
            elif self.layer_config['type']=='Squeeze-Excitation':
                self.layer_info['type']= 'Squeeze-Excitation'
            elif self.layer_config['type'] == 'view':
                self.layer_info['type'] = 'view'
            elif self.layer_config['type'] == 'bn':
                self.layer_info['type'] = 'bn'
                self.layer_info['features'] = self.layer_config['features']
            elif self.layer_config['type'] == 'dropout':
                self.layer_info['type'] = 'dropout'
            else:
                registered_spec = get_non_weighted_layer_spec(self.layer_config['type'])
                if registered_spec is not None:
                    self.layer_info.update(registered_spec.structure_metadata(self.layer_config))
                else:
                    assert 0, f'not support {self.layer_config["type"]}'
        else:
            input_values = list(input)
            self.input_shape = tuple(i.shape for i in input_values)
            output = self.layer.forward(input)
            self.output_shape = output.shape
            self.layer_info = collections.OrderedDict()
            # self.layer_info['name'] = self.layer_config['name']
            self.layer_info['type'] = self.layer_config['type']
            self.layer_info['Inputshapes'] = [list(shape) for shape in self.input_shape]
            self.layer_info['Outputshape'] = list(output.shape)
            if output.dim() >= 2:
                self.layer_info['Outputchannel'] = int(output.shape[1])
                self.layer_info['Outputsize'] = list(output.shape[2:]) or [1, 1]
            if self.layer_config['type'] == 'concat':
                configured_dim = self.layer_config.get('dim', 1)
                resolved_dim = output.dim() - 3 if configured_dim == 'channel' else int(configured_dim)
                self.layer_info['concat_dim'] = configured_dim
                self.layer_info['time_policy'] = self.layer_config.get(
                    'time_policy', 'same_timestep'
                )
                self.layer_info['merge_ownership_policy'] = 'concat_channel_segments'
                if output.dim() >= 2 and resolved_dim == 1:
                    offset = 0
                    regions = []
                    spatial = int(torch.tensor(output.shape[2:]).prod().item()) if output.dim() > 2 else 1
                    for shape in self.input_shape:
                        channels = int(shape[1])
                        width = channels * spatial
                        regions.append([offset, offset + width])
                        offset += width
                    self.layer_info['merge_input_regions'] = regions
        self.layer_info['Inputbit'] = self.quantize_config['activation_bit']
        self.layer_info['Weightbit'] = self.quantize_config['weight_bit']
        self.layer_info['outputbit'] = self.quantize_config['activation_bit']
        if 'input_index' in self.layer_config:
            self.layer_info['Inputindex'] = self.layer_config['input_index']
        else:
            self.layer_info['Inputindex'] = [-1]
        self.layer_info['Outputindex'] = [1]
        return output
    
    def forward(
        self,
        input,
        method = 'SINGLE_FIX_TEST',
        adc_action = 'SCALE',
        input_quantization=None,
    ):
        # DOES NOT use method and adc_action, for unifying with QuantizeLayer
        if self.layer_config.get('snn_merge_mode') == 'residual_add':
            from MNSIM.Interface.residual_snn import validate_residual_inputs
            validate_residual_inputs(input, input_quantization, quantized=method != 'TRADITION')
        METHOD = method
        # float method
        if METHOD == 'TRADITION':
            output = self.layer(input)
            return output
        # fix training and single fix test
        if METHOD == 'FIX_TRAIN' or METHOD == 'SINGLE_FIX_TEST':
            output = self.layer(input)
            if self.layer_config['type'] in {'bn', 'element_sum'}:
                output = configured_activation_quantize(
                    output,
                    self.quantize_config,
                    self.last_value,
                    self._activation_scale_update_enabled(),
                )
                if (
                    self.layer_config['type'] == 'element_sum'
                    and self._activation_scale_update_enabled()
                ):
                    self.post_add_quantizer_version.fill_(1)
            return output
        assert 0, f'not support {METHOD}'

    def reset_state(self):
        if hasattr(self.layer, 'reset_state'):
            self.layer.reset_state()

    def set_accuracy_perturbation(self, runtime):
        setter = getattr(self.layer, 'set_accuracy_perturbation', None)
        if callable(setter):
            setter(runtime, layer_id=self.accuracy_layer_id)

    def get_bit_weights(self):
        return None
    def extra_repr(self):
        return str(self.hardware_config) + ' ' + str(self.layer_config) + ' ' + str(self.quantize_config)
_BASE_STRAIGHT_LAYER_TYPES = ['pooling', 'relu','Swish','Sigmoid','view', 'concat', 'bn', 'dropout', 'element_sum','element_multiply']
StraightLayerStr = list(dict.fromkeys(_BASE_STRAIGHT_LAYER_TYPES + sorted(REGISTERED_MNSIM_NON_WEIGHTED_LAYER_TYPES)))
