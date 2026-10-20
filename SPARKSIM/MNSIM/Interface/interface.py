#-*-coding:utf-8-*-
import collections
import configparser
import copy
import math
import os
import copy
from importlib import import_module

import numpy as np
import torch

from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    ADC_DIGITAL_LIF,
    FRONTEND_LUT_LIF_LUT,
    parse_snn_accuracy_config,
)
from MNSIM.Accuracy_Model.snn_lif_perturbation import (
    load_lif_perturbation_artifact,
)
from MNSIM.Interface.snn_hw_config import read_weight_quantization_bit


def _coerce_positive_num_classes(value):
    if isinstance(value, bool):
        raise ValueError('num_classes must be a positive integer')
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('num_classes must be a positive integer') from exc
    if not isinstance(value, str) and value != converted:
        raise ValueError('num_classes must be a positive integer')
    if converted <= 0:
        raise ValueError('num_classes must be a positive integer')
    return converted


class TrainTestInterface(object):
    def __init__(
        self, network_module, dataset_module, SimConfig_path, weights_file=None,
        device=None, extra_define=None, num_classes=None,
        use_prequantized_weights=False, prequantized_manifest_file=None,
    ):
        # network_module: e.g., 'lenet'
        # dataset_module: e.g., 'cifar10'
        # weights_file: e.g., './zoo/cifar10_lenet_train_params.pth'
        # load net, dataset, and weights
        # device: GPU device or CPU
        self.network_module = network_module
        self.dataset_module = dataset_module
        self.weights_file = weights_file
        self.use_prequantized_weights = bool(use_prequantized_weights)
        self.prequantized_manifest_file = prequantized_manifest_file
        self.test_loader = None
        self.calibration_loader = None
        # load simconfig
        ## xbar_size, input_bit, weight_bit, ADC_quantize_bit
        xbar_config = configparser.ConfigParser()
        xbar_config.read(SimConfig_path, encoding = 'UTF-8')
        self.snn_accuracy_config = None
        self.lif_perturbation_artifact = None
        if 'snn' in self.network_module.lower():
            self.snn_accuracy_config = parse_snn_accuracy_config(
                xbar_config, source=SimConfig_path
            )
            if (
                self.snn_accuracy_config.hardware_profile
                == FRONTEND_LUT_LIF_LUT
            ):
                artifact_path = self.snn_accuracy_config.lif_perturbation_artifact
                if artifact_path is not None:
                    if not artifact_path.is_file():
                        raise FileNotFoundError(artifact_path)
                    self.lif_perturbation_artifact = (
                        load_lif_perturbation_artifact(artifact_path)
                    )
        self.hardware_config = collections.OrderedDict()
        # xbar_size
        xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config['xbar_size'] = xbar_size[0]
        self.hardware_config['type'] = int(xbar_config.get('Process element level', 'PIM_Type'))
        self.hardware_config['xbar_polarity'] = int(xbar_config.get('Process element level', 'Xbar_Polarity'))
        self.hardware_config['DAC_num'] = int(xbar_config.get('Process element level', 'DAC_Num'))
        # device bit
        self.device_bit = int(xbar_config.get('Device level', 'Device_Level'))
        self.hardware_config['weight_bit'] = math.floor(math.log2(self.device_bit))
            # weight_bit means the weight bitwidth stored in one memory device
        self.hardware_config['weight_quantization_bit'] = read_weight_quantization_bit(
            SimConfig_path
        )
        # input bit and ADC bit
        ADC_choice = int(xbar_config.get('Interface level', 'ADC_Choice'))
        DAC_choice = int(xbar_config.get('Interface level', 'DAC_Choice'))
        temp_DAC_bit = int(xbar_config.get('Interface level', 'DAC_Precision'))
        temp_ADC_bit = int(xbar_config.get('Interface level', 'ADC_Precision'))
        ADC_precision_dict = {
            -1: temp_ADC_bit,
            1: 10,
            # reference: A 10b 1.5GS/s Pipelined-SAR ADC with Background Second-Stage Common-Mode Regulation and Offset Calibration in 14nm CMOS FinFET
            2: 8,
            # reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
            3: 8,  # reference: A >3GHz ERBW 1.1GS/s 8b Two-Step SAR ADC with Recursive-Weight DAC
            4: 6,  # reference: Area-Efficient 1GS/s 6b SAR ADC with Charge-Injection-Cell-Based DAC
            5: 8,  # ASPDAC1
            6: 6,  # ASPDAC2
            7: 4,  # ASPDAC3
            8: 1,
            9: 6
        }
        DAC_precision_dict = {
            -1: temp_DAC_bit,
            1: 1,  # 1-bit
            2: 2,  # 2-bit
            3: 3,  # 3-bit
            4: 4,  # 4-bit
            5: 6,  # 6-bit
            6: 8,  # 8-bit
            7: 1
        }
        self.input_bit = DAC_precision_dict[DAC_choice]
        self.ADC_quantize_bit = ADC_precision_dict[ADC_choice]
        
        self.hardware_config['input_bit'] = self.input_bit
        self.hardware_config['ADC_quantize_bit'] = self.ADC_quantize_bit
        adc_profile = xbar_config.get(
            'Interface level', 'ADC_Profile', fallback='LEGACY'
        ).strip().upper()
        adc_mode = xbar_config.get(
            'Interface level', 'ADC_Quantization_Mode', fallback='AUTO'
        ).strip().upper()
        profile_modes = {
            'LEGACY': 'UNIFORM',
            'HONG_2025_READC_5BIT': 'READC_LLOYD_MAX',
        }
        if adc_profile not in profile_modes:
            raise ValueError(f'unsupported ADC_Profile: {adc_profile}')
        if adc_mode == 'AUTO':
            adc_mode = profile_modes[adc_profile]
        if adc_mode not in {'UNIFORM', 'ADAPTIVE', 'STATIC_NONUNIFORM', 'READC_LLOYD_MAX'}:
            raise ValueError(f'unsupported ADC_Quantization_Mode: {adc_mode}')
        if adc_mode == 'ADAPTIVE':
            adc_mode = 'STATIC_NONUNIFORM'
        self.hardware_config['ADC_quantization_mode'] = adc_mode
        self.hardware_config['ADC_profile'] = adc_profile
        if adc_profile == 'HONG_2025_READC_5BIT' and self.ADC_quantize_bit != 5:
            raise ValueError('HONG_2025_READC_5BIT requires ADC_Precision=5')
        thresholds = xbar_config.get('Interface level', 'ADC_Adaptive_Thresholds', fallback='').strip()
        self.hardware_config['ADC_adaptive_thresholds'] = (
            [float(item.strip()) for item in thresholds.split(',') if item.strip()]
            if thresholds else None
        )
        self.hardware_config['ADC_readc_percentile_low'] = xbar_config.getfloat(
            'Interface level', 'ADC_ReADC_Percentile_Low', fallback=0.5
        )
        self.hardware_config['ADC_readc_percentile_high'] = xbar_config.getfloat(
            'Interface level', 'ADC_ReADC_Percentile_High', fallback=99.5
        )
        self.hardware_config['ADC_readc_max_samples_per_layer'] = xbar_config.getint(
            'Interface level', 'ADC_ReADC_Max_Samples_Per_Layer', fallback=8000
        )
        # group num
        self.pe_group_num = int(xbar_config.get('Process element level', 'Group_Num'))
        self.tile_size = list(map(int, xbar_config.get('Tile level', 'PE_Num').split(',')))
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]
        # net and weights
        if isinstance(device, str) and device.strip().lower() == 'cpu':
            self.device = torch.device('cpu')
        elif device != None:
            self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
        print(f'run on device {self.device}')
        if num_classes is not None:
            num_classes = _coerce_positive_num_classes(num_classes)
        elif dataset_module is None:
            raise ValueError('dataset_module or explicit num_classes is required')
        elif dataset_module.endswith('cifar10'):
            num_classes = 10
        elif dataset_module.endswith('cifar100'):
            num_classes = 100
        elif dataset_module.endswith('Imagenet'):
            num_classes = 1000
        else:
            assert 0, f'unknown dataset'
            # add num_classes manually when introducing new datasets
        if extra_define != None:
            self.hardware_config['input_bit'] = extra_define['dac_res']
            self.hardware_config['ADC_quantize_bit'] = extra_define['adc_res']
            self.hardware_config['xbar_size'] = extra_define['xbar_size']
        self.net = import_module('MNSIM.Interface.network').get_net(self.hardware_config, cate = self.network_module, num_classes = num_classes)
        self.net.simconfig_path = SimConfig_path
        if weights_file is not None:
            print(f'load weights from {weights_file}')
            if self.use_prequantized_weights:
                if prequantized_manifest_file is None:
                    raise ValueError(
                        'prequantized weights require prequantized_manifest_file'
                    )
                from MNSIM.Accuracy_Model.prequantized_weights import (
                    validate_prequantized_deployment,
                )
                self.prequantized_manifest = validate_prequantized_deployment(
                    weights_file,
                    prequantized_manifest_file,
                    qbit=self.hardware_config['weight_quantization_bit'],
                )
            # load weights and split weights according to HW parameters
            #linqiushi modified
            self.net.load_change_weights(torch.load(weights_file, map_location=self.device))
            #linqiushi above
        elif self.use_prequantized_weights:
            raise ValueError('prequantized weights require an explicit weights_file')
        if self.use_prequantized_weights:
            configure = getattr(self.net, 'configure_prequantized_weights', None)
            if not callable(configure):
                raise TypeError(
                    'network does not support direct prequantized weights'
                )
            configure(True)
    def origin_evaluate(
        self, method='TRADITION', adc_action='SCALE', max_samples=None,
        pim_profile=None, lif_perturbation=None,
    ):
        if self.dataset_module is None:
            raise ValueError('dataset_module is required for accuracy evaluation')
        if self.test_loader == None:
            self.test_loader = import_module(self.dataset_module).get_dataloader()[1]
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if max_samples is not None and test_total >= max_samples:
                    break
                images = images.to(self.device)
                if max_samples is not None:
                    remaining = max_samples - test_total
                    images, labels = images[:remaining], labels[:remaining]
                test_total += labels.size(0)
                
                outputs = self._accuracy_forward(
                    images,
                    method=method,
                    adc_action=adc_action,
                )
               
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                
                test_correct += (predicted == labels).sum().item()
                
        return test_correct / test_total
    def get_net_bits(self):
        if self.use_prequantized_weights:
            net_bit_weights = self.net.get_weights(use_prequantized_weights=True)
        else:
            net_bit_weights = self.net.get_weights()
        return net_bit_weights
    def set_net_bits_evaluate(
        self, net_bit_weights, adc_action='SCALE', max_samples=None,
        pim_profile=None, lif_perturbation=None,
    ):
        if self.dataset_module is None:
            raise ValueError('dataset_module is required for accuracy evaluation')
        if pim_profile != FRONTEND_LUT_LIF_LUT and str(self.hardware_config.get(
            'ADC_quantization_mode', 'UNIFORM'
        )).upper() == 'READC_LLOYD_MAX' and not hasattr(
            self, 'readc_artifact'
        ):
            from MNSIM.Accuracy_Model.readc_artifact import ReADCArtifactError
            raise ReADCArtifactError(
                'READC_LLOYD_MAX test inference requires an installed ReADC artifact'
            )
        if self.test_loader == None:
            self.test_loader = import_module(self.dataset_module).get_dataloader()[1]
        self.net.to(self.device)
        self.net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if max_samples is not None and test_total >= max_samples:
                    break
                images = images.to(self.device)
                if max_samples is not None:
                    remaining = max_samples - test_total
                    images, labels = images[:remaining], labels[:remaining]
                test_total += labels.size(0)
                outputs = self._accuracy_forward(
                    images, method='SET_WEIGHTS', adc_action=adc_action,
                    net_bit_weights=net_bit_weights, pim_profile=pim_profile,
                    lif_perturbation=lif_perturbation,
                )
                # predicted
                labels = labels.to(self.device)
                _, predicted = torch.max(outputs, 1)
                
                test_correct += (predicted == labels).sum().item()
                
        return test_correct / test_total

    def _get_calibration_loader(self):
        dataset = import_module(self.dataset_module)
        dedicated_loader = getattr(dataset, 'get_calibration_dataloader', None)
        if callable(dedicated_loader):
            return dedicated_loader()
        return dataset.get_dataloader()[0]

    def calibration_split_identity(self):
        if getattr(self, 'calibration_loader', None) is None:
            self.calibration_loader = self._get_calibration_loader()
        return self.calibration_loader.dataset.accuracy_split_identity()

    def calibrate_readc(self, net_bit_weights, *, identity, max_samples=None):
        """Collect training-split statistics and return a bound artifact."""
        from MNSIM.Accuracy_Model.readc import (
            ReADCCalibrationConfig, ReADCSampleCollector, lloyd_max,
        )
        from MNSIM.Accuracy_Model.readc_artifact import (
            ReADCArtifactError, build_readc_artifact, validate_readc_identity,
        )
        weighted_layers = [
            (network_index, layer)
            for network_index, layer in enumerate(self.net.layer_list)
            if hasattr(layer, 'hardware_config') and hasattr(layer, 'set_weights_forward')
        ]
        self.__dict__.pop('readc_artifact', None)
        for _network_index, layer in weighted_layers:
            layer.hardware_config.pop('ADC_readc_table', None)
            layer.hardware_config.pop('ADC_readc_collector', None)
        validate_readc_identity(identity)
        if getattr(self, 'calibration_loader', None) is None:
            self.calibration_loader = self._get_calibration_loader()
        if (
            self.calibration_loader.dataset.accuracy_split_identity()
            != identity.dataset_split_identity
        ):
            raise ReADCArtifactError("calibration split identity mismatch")
        config = ReADCCalibrationConfig(
            precision=5, percentile_low=0.5, percentile_high=99.5,
            max_samples_per_layer=8000,
        )
        for _network_index, layer in weighted_layers:
            layer.hardware_config['ADC_readc_collector'] = ReADCSampleCollector(
                config.max_samples_per_layer
            )
        try:
            self.net.to(self.device)
            self.net.eval()
            seen = 0
            with torch.no_grad():
                for images, _labels in self.calibration_loader:
                    if max_samples is not None and seen >= max_samples:
                        break
                    images = images.to(self.device)
                    if max_samples is not None:
                        images = images[:max_samples - seen]
                    seen += images.shape[0]
                    self._accuracy_forward(
                        images, method='SET_WEIGHTS', adc_action='READC_CALIBRATE',
                        net_bit_weights=net_bit_weights, pim_profile=ADC_DIGITAL_LIF,
                    )
            layer_tables = {}
            for network_index, layer in weighted_layers:
                collector = layer.hardware_config['ADC_readc_collector']
                table = lloyd_max(collector.values(), config)
                table['observed_count'] = collector.observed_count
                table['sampling_strategy'] = 'stream_weighted_deterministic_fixed_capacity'
                layer_tables[f'L{network_index:03d}'] = table
        finally:
            for _network_index, layer in weighted_layers:
                layer.hardware_config.pop('ADC_readc_collector', None)
        return build_readc_artifact(identity, layer_tables)

    def install_readc_artifact(self, artifact, *, expected_identity):
        from MNSIM.Accuracy_Model.readc_artifact import (
            ReADCArtifactError, validate_readc_artifact,
        )
        validated = validate_readc_artifact(
            artifact.to_dict(), expected_identity
        )
        weighted_layers = [
            (index, layer)
            for index, layer in enumerate(self.net.layer_list)
            if hasattr(layer, 'hardware_config')
            and hasattr(layer, 'set_weights_forward')
        ]
        expected_ids = {f'L{index:03d}' for index, _layer in weighted_layers}
        if set(validated.layers) != expected_ids:
            raise ReADCArtifactError("ReADC layer identity set mismatch")
        for index, layer in weighted_layers:
            layer.hardware_config['ADC_readc_table'] = validated.layers[
                f'L{index:03d}'
            ]
        self.readc_artifact = validated

    def _reset_sequence_state(self):
        """Reset recurrent/non-weighted state between independent samples."""
        for module in self.net.modules():
            reset = getattr(module, 'reset_state', None)
            if callable(reset):
                reset()

    def _accuracy_forward(
        self,
        images,
        *,
        method,
        adc_action,
        net_bit_weights=None,
        reference_sim_config=None,
        pim_profile=None,
        lif_perturbation=None,
    ):
        """Unified ANN/SNN accuracy forward adapter.

        ANN batches retain the original single-call path. A sequence batch
        `[B,T,C,H,W]` is evaluated one timestep at a time with recurrent state
        reset between samples; timestep outputs are accumulated as SNN logits.
        """
        if images.ndim != 5:
            if method == 'SET_WEIGHTS':
                return self.net.set_weights_forward(images, net_bit_weights, adc_action)
            return self.net(images, method, adc_action)
        return self.net.sequence_accuracy_forward(
            images,
            weighted_mode='PIM' if method == 'SET_WEIGHTS' else 'ORIGINAL',
            adc_action=adc_action,
            net_bit_weights=net_bit_weights,
            pim_profile=(pim_profile or ADC_DIGITAL_LIF)
            if method == 'SET_WEIGHTS' else None,
            lif_perturbation=lif_perturbation,
        )
    def get_structure(self):
        net_bit_weights = self.net.get_weights()
        net_structure_info = self.net.get_structure()
        # print(net_structure_info)
        assert len(net_bit_weights) == len(net_structure_info)
        # set relative index to absolute index
        #   e.g., conv-relu-bn, these conv, relu, and bn layers have the same relative index
        absolute_index = [None] * len(net_structure_info)
        absolute_count = 0
        for i in range(len(net_structure_info)):
            if not (len(net_structure_info[i]['Outputindex']) == 1 and net_structure_info[i]['Outputindex'][0] == 1):
                # TODO: current version: each layer contains only one output index
                raise Exception('duplicate output')
            #if net_structure_info[i]['type'] in ['conv', 'pooling', 'element_sum', 'concat','fc']:
            #linqiushi modified
            if net_structure_info[i]['type'] in ['conv', 'pooling', 'element_sum', 'concat','fc','element_multiply']:
            #linqiushi above
                absolute_index[i] = absolute_count
                absolute_count = absolute_count + 1
            else:
                if not len(net_structure_info[i]['Inputindex']) == 1:
                    raise Exception('duplicate input index for bn, view, and relu layers')
                absolute_index[i] = absolute_index[i + net_structure_info[i]['Inputindex'][0]]
        graph = list()
        for i in range(len(net_structure_info)):
            #if net_structure_info[i]['type'] in ['conv', 'pooling', 'element_sum', 'concat', 'fc']:
            #linqiushi modified
            if net_structure_info[i]['type'] in ['conv', 'pooling', 'element_sum', 'concat', 'fc','element_multiply']:
            #linqiushi above
                # layer num, layer type: need to be computed on PIM, e.g., conv, fc, pooling, element_sum
                layer_num = absolute_index[i]
                layer_type = net_structure_info[i]['type']
                # get the layer's input index
                layer_input = list(map(lambda x: (absolute_index[i + x] if i + x != -1 else -1), net_structure_info[i]['Inputindex']))
                # get the layer's output index
                layer_output = list()
                for tmp_i in range(len(net_structure_info)):
                    #if net_structure_info[tmp_i]['type'] in ['conv', 'pooling', 'element_sum','concat', 'fc']:
                    #linqiushi modified
                    if net_structure_info[tmp_i]['type'] in ['conv', 'pooling', 'element_sum','concat', 'fc','element_multiply']:
                    #linqiushi above
                        tmp_layer_num = absolute_index[tmp_i]
                        tmp_layer_input = list(map(lambda x: (absolute_index[tmp_i + x] if tmp_i + x != -1 else -1), net_structure_info[tmp_i]['Inputindex']))
                        if layer_num in tmp_layer_input:
                            layer_output.append(tmp_layer_num)
                graph.append((layer_num, layer_type, layer_input, layer_output))
        # add to net array
        net_array = []
        for layer_num, (layer_bit_weights, layer_structure_info) in enumerate(zip(net_bit_weights, net_structure_info)):
            # change layer structure info
            layer_structure_info = copy.deepcopy(layer_structure_info)
            layer_count = absolute_index[layer_num]
            layer_structure_info['Layerindex'] = graph[layer_count][0]
            layer_structure_info['Inputindex'] = list(map(lambda x: x - graph[layer_count][0], graph[layer_count][2]))
            layer_structure_info['Outputindex'] = list(map(lambda x: x - graph[layer_count][0], graph[layer_count][3]))
            # add for element_sum and pooling
            if layer_bit_weights == None:
                #if layer_structure_info['type'] in ['element_sum', 'pooling']:
                #linqiushi modified
                if layer_structure_info['type'] in ['element_sum', 'pooling','element_multiply']:
                #linqiushi above
                    net_array.append([(layer_structure_info, None)])
                continue
            assert len(layer_bit_weights.keys()) == layer_structure_info['row_split_num'] * layer_structure_info['weight_bit_split_part'] * self.hardware_config['xbar_polarity']
            # split
            if self.hardware_config['xbar_polarity'] == 2:
                for i in range(layer_structure_info['row_split_num']):
                    for j in range(layer_structure_info['weight_bit_split_part']):
                        layer_bit_weights[f'split{i}_weight{j}_positive'] = mysplit(layer_bit_weights[f'split{i}_weight{j}_positive'], self.xbar_column)
                        layer_bit_weights[f'split{i}_weight{j}_negative'] = mysplit(layer_bit_weights[f'split{i}_weight{j}_negative'], self.xbar_column)
            else:
                for i in range(layer_structure_info['row_split_num']):
                    for j in range(layer_structure_info['weight_bit_split_part']):
                        layer_bit_weights[f'split{i}_weight{j}'] = mysplit(layer_bit_weights[f'split{i}_weight{j}'], self.xbar_column)

            # store weights for the PE array
            xbar_array = []
            for i in range(layer_structure_info['row_split_num']):
                if self.hardware_config['xbar_polarity'] == 2:
                    L = len(layer_bit_weights[f'split{i}_weight{0}_positive'])
                    for j in range(L):
                        pe_array = []
                        for s in range(layer_structure_info['weight_bit_split_part']):
                            pe_array.append([layer_bit_weights[f'split{i}_weight{s}_positive'][j].astype(np.uint8), layer_bit_weights[f'split{i}_weight{s}_negative'][j].astype(np.uint8)])
                        xbar_array.append(pe_array)
                else:
                    L = len(layer_bit_weights[f'split{i}_weight{0}'])
                    for j in range(L):
                        pe_array = []
                        for s in range(layer_structure_info['weight_bit_split_part']):
                            pe_array.append(layer_bit_weights[f'split{i}_weight{s}'][j].astype(np.int8))
                        xbar_array.append(pe_array)
                
            # store in xbar_array
            total_array = []
            L = math.ceil(len(xbar_array) / (self.tile_row * self.tile_column))
            for i in range(L):
                tile_array = []
                for h in range(self.tile_row):
                    for w in range(self.tile_column):
                        serial_number = i * self.tile_row * self.tile_column + h * self.tile_column + w
                        if serial_number < len(xbar_array):
                            tile_array.append(xbar_array[serial_number])
                total_array.append((layer_structure_info, tile_array))
            net_array.append(total_array)
        # test index
        # graph = map(lambda x: x[0][0],net_array)
        # graph = list(map(lambda x: f'l: {x["Layerindex"]}, t: {x["type"]}, i: {x["Inputindex"]}, o: {x["Outputindex"]}', graph))
        # graph = '\n'.join(graph)
        return net_array

def mysplit(array, length):
    # reshape
    array = np.reshape(array, (array.shape[0], -1))
    # split on output
    assert array.shape[0] > 0
    tmp_index = []
    for i in range(1, array.shape[0]):
        if i % length == 0:
            tmp_index.append(i)
    return np.split(array, tmp_index, axis = 0)

if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SimConfig.ini")
    __TestInterface = TrainTestInterface('vgg8', 'MNSIM.Interface.cifar10', test_SimConfig_path, './MNSIM/Interface/zoo/cifar10_vgg8_params.pth', '7')
    print(__TestInterface.origin_evaluate(method='SINGLE_FIX_TEST'))
    print(__TestInterface.set_net_bits_evaluate(__TestInterface.get_net_bits()))
    structure_info = __TestInterface.get_structure()
