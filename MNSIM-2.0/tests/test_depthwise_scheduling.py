"""Depthwise deployment must reproduce grouped convolution without cross talk."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from MNSIM.Scheduling.depthwise import depthwise_block_size, expand_depthwise_codes
from MNSIM.Scheduling.phase1.config import build_hardware_projection
from MNSIM.Scheduling.phase1.resource_prepass import build_layer_resource_descriptors
from MNSIM.Scheduling.phase2.mnsim_fragment_adapter import MnsimFragmentAdapter
from MNSIM.Scheduling.phase2.physical_write_ledger import AnnRoundPhysicalWriteLedger
from MNSIM.Scheduling.phase2.executor import _resolve_lif_owner_through_bn
from MNSIM.Scheduling.round_types import LayerIR, ModelIR, SelectedEntry, SequenceSpec


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/hardware/SimConfig_16xbar_square.ini"


@pytest.mark.parametrize("channels,rows,cols", [(20,128,128), (17,128,8), (3,128,128)])
def test_diagonal_device_codes_reproduce_depthwise_convolution(channels, rows, cols):
    shape = {"Inputchannel": channels, "Outputchannel": channels, "Kernelsize": 3, "Depthwise": "separable"}
    rng = np.random.default_rng(12)
    signed = rng.integers(-8, 9, (channels, 1, 3, 3), dtype=np.int32)
    codes = expand_depthwise_codes({"split0_weight0_positive": np.maximum(signed,0),
                                    "split0_weight0_negative": np.maximum(-signed,0)},shape,rows,cols)
    inputs = torch.from_numpy(rng.integers(0,2,(2,channels,7,7))).float()
    expected = F.conv2d(inputs, torch.from_numpy(signed).float(), padding=1, groups=channels)
    outputs = []
    block = depthwise_block_size(shape,rows,cols)
    for index,start in enumerate(range(0,channels,block)):
        end = min(start+block,channels)
        matrix = codes[f"split{index}_weight0_positive"] - codes[f"split{index}_weight0_negative"]
        weights = torch.from_numpy(matrix[start:end].reshape(end-start,end-start,3,3)).float()
        outputs.append(F.conv2d(inputs[:,start:end], weights,padding=1))
    torch.testing.assert_close(torch.cat(outputs,dim=1), expected,rtol=0,atol=0)


def test_depthwise_prepass_fragments_and_write_ledger_agree():
    shape = {"Inputchannel": 20, "Outputchannel": 20, "Kernelsize": 3,
             "Depthwise": "separable", "Weightbit": 9}
    layer = LayerIR("DW", "conv", True, 0, ["input"], ["output"], [], [], "w:DW", shape, {})
    model = ModelIR("dw", "snn", SequenceSpec(30, True, True, []), [layer])
    hw = build_hardware_projection({"Xbar_Size":[128,128], "Tile_Num":[2,2], "PE_Num":[1,1],
                                   "Group_Num":2,"Device_Level":256,"Weight_precision":8,"Xbar_Polarity":2})
    [desc] = build_layer_resource_descriptors(model,hw)
    assert desc.required_xbar_count == 4
    entry = SelectedEntry("DW","conv",0,4,4,True,desc.prepass_ref)
    inventory = MnsimFragmentAdapter(str(CONFIG),{"DW":4}).inventory_for(entry)
    assert [(f.input_range,f.output_range) for f in inventory.fragments] == [((0,14),(0,14)),((14,20),(14,20))]
    kernels = np.arange(20*9,dtype=np.int32).reshape(20,1,3,3)%17
    codes = expand_depthwise_codes({"split0_weight0_positive":kernels,
                                    "split0_weight0_negative":np.zeros_like(kernels)},shape,128,128)
    fragments = {f.fragment_id:f for f in inventory.fragments}
    slots = {f.fragment_id:{"tile_id":i,"pe_id":0,"group_id":0} for i,f in enumerate(inventory.fragments)}
    placement=SimpleNamespace(fragment_specs_by_id=fragments,fragment_slots_by_id=slots)
    ledger=AnnRoundPhysicalWriteLedger(sim_config_path=str(CONFIG),weight_codes_by_layer={"DW":codes})
    first = ledger.score_round(item_id="R0",placement=placement)
    second = ledger.score_round(item_id="R1",placement=placement)
    assert first["target_physical_xbar_count"] == 4 and first["energy_j"] > 0
    assert second["energy_j"] == 0 and second["latency_s"] == 0


def test_depthwise_multiplier_is_rejected_instead_of_miscounted():
    with pytest.raises(ValueError,match="one output per input"):
        depthwise_block_size({"Inputchannel":3,"Outputchannel":6,"Kernelsize":3},128,128)


def test_bn_ownership_traces_weighted_reducer_across_round_boundary():
    weighted = SimpleNamespace(layer_id="conv",layer_type="conv")
    bn = SimpleNamespace(layer_id="bn",layer_type="bn")
    first = SimpleNamespace(local_graph_nodes=[weighted],same_item_edges=[],external_inputs=[])
    second = SimpleNamespace(local_graph_nodes=[bn],same_item_edges=[],external_inputs=[
        SimpleNamespace(state_id="state:conv",consumer_node_ids=["bn"])])
    assert _resolve_lif_owner_through_bn(SimpleNamespace(execution_items=[first,second]),"bn") == "conv"
