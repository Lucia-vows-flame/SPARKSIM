#!/usr/bin/python
# -*-coding:utf-8-*-
import torch
import sys
import os
import math
import configparser as cp

work_path = os.path.dirname(os.getcwd())
sys.path.append(work_path)
from MNSIM.Hardware_Model import *
from MNSIM.Hardware_Model.Crossbar import crossbar
from MNSIM.Hardware_Model.Tile import tile
from MNSIM.Interface.interface import *
import collections
from collections.abc import Mapping, Sequence
import pandas as pd


class PE_node():
    def __init__(self, PE_id=0, ltype='conv', lnum=0):
        # PE_id: the id of PE node, ltype: layer type of this PE, lnum: layer number of this PE
        self.id = PE_id
        self.type = ltype
        self.lnum = lnum
        self.inMerge_list = []
        self.outMerge = 0

    def set_inMerge(self, Merge_id):
        if Merge_id not in self.inMerge_list:
            self.inMerge_list.append(Merge_id)
            self.inMerge_list.sort()

    def set_outMerge(self, Merge_id):
        self.outMerge = Merge_id


class Merge_node():
    def __init__(self, Merge_id=0, mtype=0, lnum=0):
        # Merge_id: the id of Merge node, mtype: merge type (0: add, 1: concat, 2: pooling)
        self.id = Merge_id
        self.type = mtype
        self.lnum = lnum
        self.inPE_list = []
        self.outPE_list = []
        self.inMerge_list = []
        self.outMerge_list = []

    def set_inPE(self, PE_id):
        if PE_id not in self.inPE_list:
            self.inPE_list.append(PE_id)
            self.inPE_list.sort()

    def set_outPE(self, PE_id):
        if PE_id not in self.outPE_list:
            self.outPE_list.append(PE_id)
            self.outPE_list.sort()

    def set_inMerge(self, Merge_id):
        if Merge_id not in self.inMerge_list:
            self.inMerge_list.append(Merge_id)
            self.inMerge_list.sort()

    def set_outMerge(self, Merge_id):
        if Merge_id not in self.outMerge_list:
            self.outMerge_list.append(Merge_id)
            self.outMerge_list.sort()

# The following matrix generations aim to conduct weights mapping on tiles

def generate_normal_matrix(row, column):
    matrix = np.zeros([row, column])
    start = 0
    for i in range(row):
        for j in range(column):
            matrix[i][j] = start
            start += 1
    return matrix

def generate_snake_matrix(row, column):
    matrix = np.zeros([row, column])
    start = 0
    for i in range(row):
        for j in range(column):
            if i % 2:
                matrix[i][column - j - 1] = start
            else:
                matrix[i][j] = start
            start += 1
    return matrix

def generate_hui_matrix(row, column):
    assert row > 0 and column > 0, "Tile matrix dimensions must be positive"

    if row != column:
        # The legacy Hui state machine below expands a square one outer edge at
        # a time.  It cannot reach the unused strip of a rectangular array,
        # so use the conventional clockwise Hui traversal for that case.
        matrix = np.zeros([row, column])
        value = 0
        top, bottom = 0, row - 1
        left, right = 0, column - 1
        while top <= bottom and left <= right:
            for j in range(left, right + 1):
                matrix[top][j] = value
                value += 1
            top += 1

            for i in range(top, bottom + 1):
                matrix[i][right] = value
                value += 1
            right -= 1

            if top <= bottom:
                for j in range(right, left - 1, -1):
                    matrix[bottom][j] = value
                    value += 1
                bottom -= 1

            if left <= right:
                for i in range(bottom, top - 1, -1):
                    matrix[i][left] = value
                    value += 1
                left += 1
        return matrix

    matrix = np.zeros([row, column])
    state = 0
    stride = 1
    step = 0
    start = 0
    dl = 0
    ru = 0
    i = 0
    j = 0
    for x in range(row * column):
        if x == 0:
            matrix[i][j] = start
        else:
            if state == 0:
                j += 1
                matrix[i][j] = start
                state = 1
            elif state == 1:
                if dl == 0:
                    i += 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        dl = 1
                        step = 0
                elif dl == 1:
                    j -= 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        dl = 0
                        step = 0
                        stride += 1
                        state = 2
            elif state == 2:
                i += 1
                matrix[i][j] = start
                state = 3
            elif state == 3:
                if ru == 0:
                    j += 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        ru = 1
                        step = 0
                elif ru == 1:
                    i -= 1
                    matrix[i][j] = start
                    step += 1
                    if step == stride:
                        ru = 0
                        step = 0
                        stride += 1
                        state = 0
        start += 1
    return matrix

def generate_zigzag_matrix(row, column):
    matrix = np.zeros([row, column])
    state = 0
    stride = 1
    step = 0
    i = 0
    j = 0
    start = 0
    for x in range(row * column):
        if x == 0:
            matrix[i][j] = start
        else:
            if state == 0:
                if j < column - 1:
                    j += 1
                    matrix[i][j] = start
                else:
                    i += 1
                    matrix[i][j] = start
                state = 1
            elif state == 1:
                i += 1
                j -= 1
                matrix[i][j] = start
                step += 1
                if i == row - 1:
                    state = 2
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 2
                    stride += 1
                    step = 0
            elif state == 2:
                if i < row - 1:
                    i += 1
                    matrix[i][j] = start
                else:
                    j += 1
                    matrix[i][j] = start
                state = 3
            elif state == 3:
                j += 1
                i -= 1
                matrix[i][j] = start
                step += 1
                if j == column - 1:
                    state = 0
                    stride -= 1
                    step = 0
                elif step == stride:
                    state = 0
                    stride += 1
                    step = 0
        start += 1
    return matrix


def _active_layout_int(value, field):
    assert isinstance(value, int) and not isinstance(value, bool) and value >= 0, \
        f"active fragment layout {field} must be a non-negative integer"
    return int(value)


def _normalize_active_fragment_layout(NetStruct, active_fragment_layout, xbar_polarity):
    assert isinstance(active_fragment_layout, Mapping), \
        "active fragment layout must be keyed by mapping layer id"
    mapping_layer_ids = {
        layer_id
        for layer_id, layer_entry in enumerate(NetStruct)
        if layer_entry[0][0]['type'] in ['conv', 'fc']
    }
    normalized = {}
    for raw_layer_id, raw_payload in active_fragment_layout.items():
        assert isinstance(raw_layer_id, int) and not isinstance(raw_layer_id, bool), \
            "active fragment layout layer id must be an integer"
        assert raw_layer_id in mapping_layer_ids, \
            "active fragment layout may only reference Conv/FC mapping layers"
        assert isinstance(raw_payload, Mapping), \
            "active fragment layout payload must be a mapping"
        assert {
            'fragment_ids',
            'active_group_count',
            'active_physical_xbar_count',
            'polarity_count',
        }.issubset(raw_payload.keys()), "active fragment layout payload is incomplete"
        fragment_ids = raw_payload['fragment_ids']
        assert isinstance(fragment_ids, Sequence) and not isinstance(fragment_ids, (str, bytes)), \
            "active fragment layout fragment_ids must be a sequence"
        fragment_ids = list(fragment_ids)
        assert all(isinstance(fragment_id, str) and fragment_id for fragment_id in fragment_ids), \
            "active fragment layout fragment ids must be non-empty strings"
        assert len(fragment_ids) == len(set(fragment_ids)), \
            "active fragment layout cannot repeat a fragment id in one mapping layer"
        active_group_count = _active_layout_int(
            raw_payload['active_group_count'],
            'active_group_count',
        )
        active_physical_xbar_count = _active_layout_int(
            raw_payload['active_physical_xbar_count'],
            'active_physical_xbar_count',
        )
        polarity_count = _active_layout_int(raw_payload['polarity_count'], 'polarity_count')
        assert polarity_count == xbar_polarity, \
            "active fragment layout polarity does not match SimConfig Xbar_Polarity"
        assert len(fragment_ids) == active_group_count, \
            "active fragment layout fragment count does not match active group count"
        assert active_physical_xbar_count == active_group_count * polarity_count, \
            "active fragment layout physical count does not match complete polarity groups"
        normalized[raw_layer_id] = {
            'fragment_ids': fragment_ids,
            'active_group_count': active_group_count,
            'active_physical_xbar_count': active_physical_xbar_count,
            'polarity_count': polarity_count,
        }
    assert set(normalized) == mapping_layer_ids, \
        "active fragment layout must cover every Conv/FC mapping layer exactly once"
    all_fragment_ids = [
        fragment_id
        for payload in normalized.values()
        for fragment_id in payload['fragment_ids']
    ]
    assert len(all_fragment_ids) == len(set(all_fragment_ids)), \
        "active fragment layout cannot repeat a fragment id across mapping layers"
    return normalized


def _active_fragment_group_slots(tile_total_num, pe_total_num, group_num, active_layout):
    slots = [
        {
            'tile_id': tile_id,
            'pe_id': pe_id,
            'group_id': group_id,
        }
        for tile_id in range(tile_total_num)
        for pe_id in range(pe_total_num)
        for group_id in range(group_num)
    ]
    next_slot = 0
    slots_by_layer = {}
    for layer_id in sorted(active_layout):
        layer_slots = []
        for fragment_id in active_layout[layer_id]['fragment_ids']:
            assert next_slot < len(slots), "Tile number is not enough"
            slot = dict(slots[next_slot])
            slot['fragment_id'] = fragment_id
            layer_slots.append(slot)
            next_slot += 1
        slots_by_layer[layer_id] = layer_slots
    return slots_by_layer


def _apply_active_fragment_mapping(tmp_tileinfo, active_payload, group_num):
    """Replace full-Weightbit PE demand with the selected active groups."""
    active_group_count = active_payload['active_group_count']
    if active_group_count == 0:
        return False

    active_pe_count = math.ceil(active_group_count / group_num)
    tmp_tileinfo['mx'] = active_pe_count
    tmp_tileinfo['my'] = 1
    tmp_tileinfo['max_group'] = min(active_group_count, group_num)
    tmp_tileinfo['active_PE_count'] = active_pe_count
    return True


class TCG():
    def __init__(
        self,
        NetStruct,
        SimConfig_path,
        multiple=None,
        *,
        round_residency=False,
        active_fragment_layout=None,
    ):
        # NetStruct: layer structure, SimConfig_path: Hardware config path, multiple: allocate more resources for some layers (i.e., duplicate)
        TCG_config = cp.ConfigParser()
        TCG_config.read(SimConfig_path, encoding='UTF-8')
        if multiple is None:
            multiple = [1] * len(NetStruct)
        self.round_residency = bool(round_residency)
        self.tile = tile(SimConfig_path)
        self.net = NetStruct
        self.layer_num = len(self.net)
        self.layer_tileinfo = []
        self.xbar_polarity = int(TCG_config.get('Process element level', 'Xbar_Polarity'))
        self.tile_connection = int(TCG_config.get('Architecture level', 'Tile_Connection'))
        self.tile_num = list(map(int, TCG_config.get('Architecture level', 'Tile_Num').split(',')))
        if self.tile_num[0] == 0:
            self.tile_num[0] = 8
            self.tile_num[1] = 8
        assert self.tile_num[0] > 0, "Tile number < 0"
        assert self.tile_num[1] > 0, "Tile number < 0"
        self.tile_total_num = self.tile_num[0] * self.tile_num[1]
        self.active_fragment_layout = None
        self.active_fragment_group_slots = {}
        self.active_group_count = 0
        self.active_physical_xbar_count = 0
        self.mnsim_group_capacity = self.tile_total_num * self.tile.tile_PE_total_num * self.tile.group_num
        self.physical_xbar_capacity = self.mnsim_group_capacity * self.xbar_polarity
        if active_fragment_layout is not None:
            self.active_fragment_layout = _normalize_active_fragment_layout(
                self.net,
                active_fragment_layout,
                self.xbar_polarity,
            )
            self.active_group_count = sum(
                payload['active_group_count']
                for payload in self.active_fragment_layout.values()
            )
            self.active_physical_xbar_count = sum(
                payload['active_physical_xbar_count']
                for payload in self.active_fragment_layout.values()
            )
            assert self.active_group_count <= self.mnsim_group_capacity, "Tile number is not enough"
            assert self.active_physical_xbar_count <= self.physical_xbar_capacity, \
                "Crossbar number is not enough"
            self.active_fragment_group_slots = _active_fragment_group_slots(
                self.tile_total_num,
                self.tile.tile_PE_total_num,
                self.tile.group_num,
                self.active_fragment_layout,
            )
        self.mapping_order = -1 * np.ones(self.tile_num)
        self.mapping_result = -1 * np.ones(self.tile_num)
        start_tileid = 0
            # the start Tile id
        self.max_inbuf_size = 0
            # the maximum input buffer size of each PE, unit: KB
        self.max_outbuf_size = 0
            # the maximum output buffer size of each tile, unit: KB
        self.buffer_capacity_evidence = []
            # per-layer raw and rounded buffer working-set requirements
        self.global_buf_size = 0
            # the global buffer size for accumulator
        self.global_data_size = 0
        self.global_adder_num = 0
            # the global adder number in accumulator
        self.global_multiplier_num=0
        self.global_adder_bitwidth = 8
        self.global_multiplier_bitwidth=8
        num = []
            # track PE number of each layer 
        total_xbar_num = 0
        for layer_id in range(self.layer_num):
            layer_dict = self.net[layer_id][0][0]
            tmp_tileinfo = collections.OrderedDict()
            layer_type = layer_dict['type']
            if self.xbar_polarity == 1:
                weight_precision = int(layer_dict['Weightbit'])
            else:
                assert self.xbar_polarity == 2, "Crossbar polarity must be 1 or 2"
                weight_precision = int(layer_dict['Weightbit']) - 1
            tmp_tileinfo['startid'] = start_tileid
            input_size = 0
            inputchannel = 0
            outputchannel = 0
            data_inbuf = 0
            data_outbuf = 0

            if layer_type == 'conv':
                tmp_tileinfo['type'] = 'conv'
                tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outputchannel']) / self.tile.xbar_column)
                    # mx: PE number in x-axis
                tmp_tileinfo['my'] = math.ceil(int(layer_dict['Inputchannel']) / (self.tile.xbar_row // (int(layer_dict['Kernelsize']) ** 2)))
                    # my: PE number in y-axis
                tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                    # max_group: maximum used groups in one PE of this layer
                tmp_tileinfo['max_row'] = min((self.tile.xbar_row // (int(layer_dict['Kernelsize']) ** 2)),
                    int(layer_dict['Inputchannel'])) * (int(layer_dict['Kernelsize']) ** 2)
                    # max_row: maximum used row in one crossbar of this layer
                tmp_tileinfo['max_column'] = min(int(layer_dict['Outputchannel']), self.tile.xbar_column)
                    # max_column: maximum used column in one crossbar of this layer
                if 'Inputindex' not in layer_dict.keys():
                    tmp_tileinfo['Inputindex'] = [-1]
                else:
                    tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                    # Inputindex: the relative index of the input layers of this layer
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                    # is_branchin: if this layer is the input layer of a branch
                tmp_tileinfo['is_branchout'] = 1
                    # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                for i in tmp_tileinfo['Outputindex']:
                    tmp_layer = self.net[i+layer_id][0][0]
                    if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                        tmp_tileinfo['is_branchout'] = -1

                input_size_list = list(map(int, layer_dict['Inputsize']))
                input_size = input_size_list[0] * input_size_list[1]
                inputchannel = int(layer_dict['Inputchannel'])
                data_inbuf = input_size_list[1] * int(layer_dict['Kernelsize']) * inputchannel * int(layer_dict['Inputbit'])/8
                    # assume using the line buffer structure
                outputchannel = int(layer_dict['Outputchannel'])
                data_outbuf = outputchannel*int(layer_dict['outputbit'])/8
                # buffer_size: unit Byte
            elif layer_type == 'fc':
                tmp_tileinfo['type'] = 'fc'
                tmp_tileinfo['mx'] = math.ceil(weight_precision / self.tile.group_num) * math.ceil(int(layer_dict['Outfeature']) / self.tile.xbar_column)
                    # mx: PE number in x-axis
                tmp_tileinfo['my'] = math.ceil(int(layer_dict['Infeature']) / self.tile.xbar_row)
                    # my: PE number in y-axis
                tmp_tileinfo['max_group'] = min(weight_precision, self.tile.group_num)
                    # max_group: maximum used groups in one PE of this layer
                tmp_tileinfo['max_row'] = min(int(layer_dict['Infeature']), self.tile.xbar_row)
                    # max_row: maximum used row in one crossbar of this layer
                tmp_tileinfo['max_column'] = min(int(layer_dict['Outfeature']), self.tile.xbar_column)
                    # max_row: maximum used column in one crossbar of this layer
                if 'Inputindex' not in layer_dict.keys():
                    tmp_tileinfo['Inputindex'] = [-1]
                else:
                    tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                    # Inputindex: the relative index of the input layers of this layer
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                    # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                tmp_tileinfo['is_branchout'] = 1
                # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                for i in tmp_tileinfo['Outputindex']:
                    if (i+layer_id) < self.layer_num:
                        tmp_layer = self.net[i + layer_id][0][0]
                        if (tmp_layer['type'] != 'element_sum' and tmp_layer['type']!= 'element_multiply'):
                            tmp_tileinfo['is_branchout'] = -1
                # is_branchin: if this layer is the input layer of a branch
                input_size = int(layer_dict['Infeature'])
                inputchannel = 1
                data_inbuf = input_size * inputchannel * int(layer_dict['Inputbit'])/8
                data_outbuf = int(layer_dict['Outfeature']) * int(layer_dict['outputbit'])/8
                # buffer_size: unit Byte
            elif layer_type == 'pooling':
                tmp_tileinfo['type'] = 'pooling'
                tmp_tileinfo['mx'] = 1
                tmp_tileinfo['my'] = 1
                tmp_tileinfo['max_row'] = 0
                tmp_tileinfo['max_column'] = 0
                tmp_tileinfo['max_group'] = 0
                if 'Inputindex' not in layer_dict.keys():
                    tmp_tileinfo['Inputindex'] = [-1]
                else:
                    tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                # Inputindex: the relative index of the input layers of this layer
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                # is_branchin: if this layer is the input layer of a branch
                tmp_tileinfo['is_branchout'] = 1
                # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                for i in tmp_tileinfo['Outputindex']:
                    tmp_layer = self.net[i + layer_id][0][0]
                    if tmp_layer['type'] != 'element_sum' and tmp_layer['type'] != 'element_multiply':
                        tmp_tileinfo['is_branchout'] = -1
                input_size_list = list(map(int, layer_dict['Inputsize']))
                input_size = input_size_list[0] * input_size_list[1]
                inputchannel = int(layer_dict['Inputchannel'])
                data_inbuf = 0 # assume the pooling module shares the same buffer with xbar PEs
                data_outbuf = 0
                    # assume the buffer size depends on the conv/fc layers
            elif layer_type == 'element_sum':

                tmp_tileinfo['type'] = 'element_sum'
                tmp_tileinfo['mx'] = 0
                tmp_tileinfo['my'] = 0
                tmp_tileinfo['max_row'] = 0
                tmp_tileinfo['max_column'] = 0
                tmp_tileinfo['max_group'] = 0
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                tmp_tileinfo['is_branchout'] = -1
                # is_branchin: if this layer is the input layer of a branch
                Inputindex_list = list(map(int, layer_dict['Inputindex']))
                tmp_tileinfo['Inputindex'] = Inputindex_list
                assert len(Inputindex_list)>1, "the number of element_sum's previous layers must > 1"
                idx = 0
                previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                while previous_layer_dict['type'] == 'element_sum':
                    idx = idx+1
                    previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
               
                previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                    # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                    # the data precision of each branch output (bit)
                data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                    # unit: Byte
                self.global_data_size = self.global_data_size + data_size
                self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                    # unit: KB
                self.global_adder_num = self.global_adder_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                    self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
            elif layer_type == 'element_multiply':
                tmp_tileinfo['type'] = 'element_multiply'
                tmp_tileinfo['mx'] = 0
                tmp_tileinfo['my'] = 0
                tmp_tileinfo['max_row'] = 0
                tmp_tileinfo['max_column'] = 0
                tmp_tileinfo['max_group'] = 0
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                tmp_tileinfo['is_branchout'] = -1
                # is_branchin: if this layer is the input layer of a branch
                Inputindex_list = list(map(int, layer_dict['Inputindex']))
                tmp_tileinfo['Inputindex'] = Inputindex_list
                assert len(Inputindex_list)>1, "the number of element_multiply's previous layers must > 1"
                idx = 0
                previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                while previous_layer_dict['type'] == 'element_multiply':
                    idx = idx+1
                    previous_layer_dict = self.net[layer_id + Inputindex_list[idx]][0][0]
                #previous_output_size = list(map(int, previous_layer_dict['Outputsize']))
                tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel']
                    # the data number of each branch output, assume the previous layer generates 1*1*outputchannel each cycle
                tmp_tileinfo['bit_branchout'] = previous_layer_dict['outputbit']
                    # the data precision of each branch output (bit)
                data_size = tmp_tileinfo['datanum_branchout']*tmp_tileinfo['bit_branchout']*len(Inputindex_list)/8
                    # unit: Byte
                self.global_data_size = self.global_data_size + data_size
                self.global_buf_size = self.global_buf_size + math.pow(2,math.ceil(math.log(data_size,2)))/1024
                    # unit: KB
                self.global_multiplier_num = self.global_multiplier_num + previous_layer_dict['Outputchannel']*len(Inputindex_list)//2
                if tmp_tileinfo['bit_branchout']>self.global_adder_bitwidth:
                    self.global_adder_bitwidth = tmp_tileinfo['bit_branchout']
            if layer_type == 'conv' or layer_type == 'fc':
                active_mapping = False
                if self.active_fragment_layout is not None:
                    active_payload = self.active_fragment_layout[layer_id]
                    tmp_tileinfo['active_group_count'] = active_payload['active_group_count']
                    tmp_tileinfo['active_physical_xbar_count'] = active_payload['active_physical_xbar_count']
                    tmp_tileinfo['active_fragment_ids'] = list(active_payload['fragment_ids'])
                    active_mapping = _apply_active_fragment_mapping(
                        tmp_tileinfo,
                        active_payload,
                        self.tile.group_num,
                    )
                if active_mapping:
                    total_xbar_num += active_payload['active_group_count']
                else:
                    total_xbar_num += tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
            tmp_tileinfo['PEnum'] = tmp_tileinfo['mx'] * tmp_tileinfo['my'] * multiple[layer_id]
            num.append(tmp_tileinfo['PEnum'])
            tmp_tileinfo['tilenum'] = math.ceil(tmp_tileinfo['PEnum'] / self.tile.tile_PE_total_num)
            tmp_tileinfo['max_PE'] = min(tmp_tileinfo['PEnum'], self.tile.tile_PE_total_num)
            start_tileid += tmp_tileinfo['tilenum']
            self.layer_tileinfo.append(tmp_tileinfo)

            inputbit = int(layer_dict['Inputbit'])
            if tmp_tileinfo['type'] == 'conv' or tmp_tileinfo['type'] == 'fc':
                raw_inbuf_bytes = data_inbuf / tmp_tileinfo['PEnum']
                raw_outbuf_bytes = data_outbuf*2 / tmp_tileinfo['tilenum'] # 2: ping-pong
                tmp_inbuf_size = math.pow(2,math.ceil(math.log(raw_inbuf_bytes,2)))/1024
                tmp_outbuf_size = math.pow(2,math.ceil(math.log(raw_outbuf_bytes,2)))/1024
                self.buffer_capacity_evidence.append({
                    'layer_id': int(layer_id),
                    'layer_type': str(tmp_tileinfo['type']),
                    'pe_input_working_set_bytes_per_instance': float(raw_inbuf_bytes),
                    'tile_output_working_set_bytes_per_instance': float(raw_outbuf_bytes),
                    'tcg_pe_input_rounded_kb': float(tmp_inbuf_size),
                    'tcg_tile_output_rounded_kb': float(tmp_outbuf_size),
                    'source_rounding_rule': 'tcg_next_power_of_two_bytes',
                })
            else:
                tmp_inbuf_size = 0
                tmp_outbuf_size = 0
            # unit: KB, restricted in 2^M KB
            if tmp_inbuf_size > self.max_inbuf_size:
                self.max_inbuf_size = tmp_inbuf_size
            if tmp_outbuf_size > self.max_outbuf_size:
                self.max_outbuf_size = tmp_outbuf_size

        self.used_tile_num = start_tileid
        if self.round_residency and self.active_fragment_layout is not None:
            self.round_residency_tile_count = math.ceil(
                self.active_group_count / (self.tile.tile_PE_total_num * self.tile.group_num)
            )
        else:
            self.round_residency_tile_count = math.ceil(
                total_xbar_num / self.tile.tile_PE_total_num
            )
        if self.round_residency:
            assert self.round_residency_tile_count <= self.tile_total_num, "Tile number is not enough"
        else:
            assert self.used_tile_num <= self.tile_total_num, "Tile number is not enough"
        # TODO: update weight rewrite in xbar
        print("Total crossbar number:", total_xbar_num)
        self.inLayer_distance = np.zeros([1, self.layer_num])
        self.transLayer_distance = np.zeros([1, self.layer_num])
        self.aggregate_arg = np.zeros([self.layer_num, 2])

    def mapping_matrix_gen(self):
        if self.tile_connection == 0:
            self.mapping_order = generate_normal_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.tile_connection == 1:
            self.mapping_order = generate_snake_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.tile_connection == 2:
            self.mapping_order = generate_hui_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])
        elif self.tile_connection == 3:
            self.mapping_order = generate_zigzag_matrix(self.mapping_order.shape[0], self.mapping_order.shape[1])

    def mapping_net(self):
        self.mapping_matrix_gen()
        for i in range(self.mapping_order.shape[0]):
            for j in range(self.mapping_order.shape[1]):
                if self.mapping_order[i][j] < self.used_tile_num:
                    for layer_id in range(self.layer_num - 1):
                        if self.layer_tileinfo[layer_id]['type'] in ['conv','pooling','fc']:
                            # only allocate tile for conv layers, pooling layers, and fc layers
                            if ((self.mapping_order[i][j] >= self.layer_tileinfo[layer_id]['startid']) &
                                    (self.mapping_order[i][j] < self.layer_tileinfo[layer_id + 1]['startid'])):
                                self.mapping_result[i][j] = layer_id
                                break
                            elif self.mapping_order[i][j] >= self.layer_tileinfo[self.layer_num - 1]['startid']:
                                self.mapping_result[i][j] = self.layer_num - 1

    def calculate_transfer_distance(self):
        for layer_id in range(self.layer_num - 1):
            # Determine the aggregate node for layer 0~N-1
            if self.layer_tileinfo[layer_id]['is_branchout'] == 1:
                # for the layer which is a output layer of one branch and the next layer is element_sum
                if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc']:
                    src_pos = np.argwhere(self.mapping_result == layer_id)
                    if len(src_pos) == 1:
                        self.inLayer_distance[0][layer_id] = 0
                        self.aggregate_arg[layer_id] = src_pos[0]
                        self.transLayer_distance[0][layer_id] = abs(src_pos[0][0]-1/2*self.tile_num[0]) + src_pos[0][1]
                    else:
                        mindis_total = 1000
                        for A in range(len(src_pos)):
                            tmp_transLayer_distance = abs(src_pos[A][0]-1/2*self.tile_num[0]) + src_pos[A][1]
                            maxdis_in = 0
                            for i in range(len(src_pos)):
                                if i != A:
                                    dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                    if dis_in > maxdis_in:
                                        maxdis_in = dis_in
                            if (maxdis_in+tmp_transLayer_distance)<mindis_total:
                                self.inLayer_distance[0][layer_id] = maxdis_in
                                self.transLayer_distance[0][layer_id] = tmp_transLayer_distance
                                self.aggregate_arg[layer_id] = src_pos[A]
                                mindis_total = maxdis_in+tmp_transLayer_distance
            else:
                if self.layer_tileinfo[layer_id]['type'] in ['conv', 'pooling', 'fc']:
                    src_pos = np.argwhere(self.mapping_result == layer_id)
                    if len(src_pos) == 1:
                        self.inLayer_distance[0][layer_id] = 0
                        self.aggregate_arg[layer_id] = src_pos[0]
                        maxdis = 0
                        for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                            dst_pos = np.argwhere(self.mapping_result == (layer_id + idx))
                            for i in range(len(dst_pos)):
                                dis = abs(src_pos[0][0] - dst_pos[i][0]) + abs(src_pos[0][1] - dst_pos[i][1])
                                if dis > maxdis:
                                    maxdis = dis
                        self.transLayer_distance[0][layer_id] = maxdis
                    else:
                        mindis_total = 1000
                        for A in range(len(src_pos)):
                            maxdis_in = 0
                            maxdis_out = 0
                            for i in range(len(src_pos)):
                                if i != A:
                                    dis_in = abs(src_pos[A][0] - src_pos[i][0]) + abs(src_pos[A][1] - src_pos[i][1])
                                    if dis_in > maxdis_in:
                                        maxdis_in = dis_in
                            for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                                dst_pos = np.argwhere(self.mapping_result == (layer_id + idx))
                                for j in range(len(dst_pos)):
                                    dis_out = abs(src_pos[A][0] - dst_pos[j][0]) + abs(src_pos[A][1] - dst_pos[j][1])
                                    if dis_out > maxdis_out:
                                        maxdis_out = dis_out
                            tempdis = maxdis_in + maxdis_out
                            if tempdis < mindis_total:
                                self.inLayer_distance[0][layer_id] = maxdis_in
                                self.transLayer_distance[0][layer_id] = maxdis_out
                                self.aggregate_arg[layer_id] = src_pos[A]
                                mindis_total = tempdis
                elif self.layer_tileinfo[layer_id]['type'] == 'element_sum' or self.layer_tileinfo[layer_id]['type'] == 'element_multiply':
                    maxdis_out = 0
                    for idx in self.layer_tileinfo[layer_id]['Outputindex']:
                        dst_pos = np.argwhere(self.mapping_result == (layer_id + idx))
                        for j in range(len(dst_pos)):
                            dis_out = abs(dst_pos[0][0]-1/2*self.tile_num[0]) + dst_pos[0][1]
                            if dis_out > maxdis_out:
                                maxdis_out = dis_out
                    self.inLayer_distance[0][layer_id] = 0
                    self.transLayer_distance[0][layer_id] = maxdis_out
        final_pos = np.argwhere(self.mapping_result == self.layer_num - 1)
        # Determine the aggregate node for layer N (output layer)
        mindis = 1000
        for i in range(len(final_pos)):
            maxdis = 0
            for j in range(len(final_pos)):
                if j != i:
                    dis = abs(final_pos[i][0] - final_pos[j][0]) + abs(final_pos[i][1] - final_pos[j][1])
                    if dis > maxdis:
                        maxdis = dis
            if maxdis < mindis:
                mindis = maxdis
                self.inLayer_distance[0][self.layer_num - 1] = mindis
                self.aggregate_arg[self.layer_num - 1] = final_pos[i]
                self.transLayer_distance[0][self.layer_num - 1] = 0
        # self.total_distance = sum(sum(self.trans_time * (self.inLayer_distance + self.transLayer_distance)))


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "SimConfig.ini")
    test_weights_file_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())),
                                          "vgg8_params.pth")

    __TestInterface = TrainTestInterface('vgg8_128_9', 'MNSIM.Interface.cifar10', test_SimConfig_path,
                                         test_weights_file_path, 'cpu')
    structure_file = __TestInterface.get_structure()

    test = TCG(structure_file, test_SimConfig_path)
    test.mapping_net()
    test.calculate_transfer_distance()
    # print(test.total_distance)
