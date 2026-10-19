import collections

from MNSIM.Scheduling.phase2.mnsim_static_eval import run_mnsim_static_eval
from MNSIM.Scheduling.phase2.types import (
    BoundaryBinding,
    StaticEvalFrame,
    StaticEvalRequest,
)


def test_original_mnsim_backend_runs_single_fc_static_eval():
    layer_info = collections.OrderedDict(
        [
            ("type", "fc"),
            ("Infeature", 16),
            ("Outfeature", 8),
            ("Inputbit", 8),
            ("Weightbit", 8),
            ("outputbit", 8),
            ("row_split_num", 1),
            ("weight_bit_split_part", 1),
            ("Inputindex", [-1]),
            ("Outputindex", []),
            ("Layerindex", 0),
        ]
    )
    binding = BoundaryBinding(
        local_layer_id=0,
        hook_node_id=None,
        origin_layer_id="FC0",
        origin_layer_type="fc",
        report_role="reportable",
        tensor_role="activation",
        shape={"Outfeature": 8},
        region={"kind": "full"},
        input_range=[0, 16],
        output_range=[0, 8],
        full_input_range=[0, 16],
        full_output_range=[0, 8],
        contribution_kind="complete_output",
        source_state_ids=["model_input"],
        downstream_consumer_ids=[],
    )
    request = StaticEvalRequest(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[[(layer_info, None)]],
        sim_config_path="SimConfig.ini",
        multiple=[1],
        latency_api="nopipe",
        latency_fill_mode=0,
        frame=StaticEvalFrame([], [0], [], [binding]),
    )

    result = run_mnsim_static_eval(request)

    assert result.status == "ok"
    assert result.item_totals.latency_s >= 0.0
    assert result.item_totals.area_um2 > 0.0


def test_zero_xbar_element_sum_direct_eval_runs_without_context_stubs():
    layer_info = collections.OrderedDict(
        [
            ("type", "element_sum"),
            ("Inputindex", [-1, -2]),
            ("Outputindex", []),
            ("Inputsize", [16, 16]),
            ("Outputsize", [16, 16]),
            ("Inputchannel", 64),
            ("Outputchannel", 64),
            ("Inputbit", 8),
            ("Weightbit", 8),
            ("outputbit", 8),
            ("Layerindex", 0),
        ]
    )
    binding = BoundaryBinding(
        local_layer_id=0,
        hook_node_id=None,
        origin_layer_id="Sum0",
        origin_layer_type="element_sum",
        report_role="reportable",
        tensor_role="activation",
        shape={"Outputchannel": 64, "Outputsize": [16, 16], "outputbit": 8},
        region={"kind": "standalone_structural"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="structural_output",
        source_state_ids=["state:left", "state:right"],
        downstream_consumer_ids=[],
    )
    request = StaticEvalRequest(
        item_id="Z0",
        item_kind="zero_xbar_structural",
        netstruct=[[(layer_info, None)]],
        sim_config_path="SimConfig.ini",
        multiple=[1],
        latency_api="pipe",
        latency_fill_mode=1,
        frame=StaticEvalFrame([], [0], [], [binding]),
    )

    result = run_mnsim_static_eval(request)

    assert result.status == "ok"
    assert result.layer_results[0].origin_layer_id == "Sum0"
    assert result.item_totals.latency_s > 0.0
    assert result.item_totals.area_um2 > 0.0
    assert result.item_totals.power_w > 0.0
    assert result.item_totals.energy_j > 0.0


def test_original_mnsim_pooling_context_shim_preserves_input_resolution():
    layer_info = collections.OrderedDict(
        [
            ("type", "pooling"),
            ("Inputindex", [-1]),
            ("Outputindex", []),
            ("Inputsize", [8, 8]),
            ("Outputsize", [4, 4]),
            ("Inputchannel", 16),
            ("Outputchannel", 16),
            ("Kernelsize", 2),
            ("Stride", 2),
            ("Padding", 0),
            ("Inputbit", 8),
            ("Weightbit", 8),
            ("outputbit", 8),
            ("Layerindex", 0),
        ]
    )
    binding = BoundaryBinding(
        local_layer_id=0,
        hook_node_id=None,
        origin_layer_id="Pool0",
        origin_layer_type="pooling",
        report_role="reportable",
        tensor_role="activation",
        shape={"Outputchannel": 16, "Outputsize": [4, 4], "outputbit": 8},
        region={"kind": "standalone_structural"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="structural_output",
        source_state_ids=["state:conv"],
        downstream_consumer_ids=[],
    )
    request = StaticEvalRequest(
        item_id="R0",
        item_kind="zero_xbar_structural",
        netstruct=[[(layer_info, None)]],
        sim_config_path="SimConfig.ini",
        multiple=[1],
        latency_api="pipe",
        latency_fill_mode=1,
        frame=StaticEvalFrame([], [0], [], [binding]),
    )

    result = run_mnsim_static_eval(request)

    assert result.status == "ok"
    assert result.layer_results[-1].origin_layer_id == "Pool0"
    assert result.item_totals.latency_s > 0.0
