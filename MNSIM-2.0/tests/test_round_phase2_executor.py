import configparser
from copy import deepcopy
import math

import pytest

from MNSIM.Scheduling.device_precision import device_bits_from_level
import MNSIM.Scheduling.phase2 as phase2
import MNSIM.Scheduling.phase2.executor as executor_module

from MNSIM.Scheduling.phase2.executor import execute_round_plan
from MNSIM.Scheduling.phase2.types import (
    PHYSICAL_XBAR_UNIT_MISMATCH,
    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
    Phase2Invalid,
)
from MNSIM.Scheduling.round_types import (
    BoundaryRef,
    LayerResourceDescriptor,
    LocalGraphNode,
    PrepassRef,
    ResourceProof,
    RoundPlan,
    SameItemEdge,
    SelectedEntry,
    WeightedLayerSegment,
    WeightedPhysicalRound,
    ZeroXbarSegment,
)


class FakeStaticBackend:
    def __init__(self):
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        return {
            "layers": [
                {
                    "local_layer_id": 0,
                    "layer_type": request.netstruct[0][0][0]["type"],
                    "begin_time_s": 0.0,
                    "finish_time_s": 1e-6,
                    "area_um2": 10.0,
                    "power_w": 2.0,
                    "energy_j": 2e-6,
                    "tcg_mapping": {"tilenum": 1},
                }
            ],
            "raw_totals": {
                "raw_latency_total_s": 1e-6,
                "raw_area_total_um2": 10.0,
                "raw_power_total_w": 2.0,
                "raw_energy_total_j": 2e-6,
            },
            "mapping_summary": {"layer_count": 1},
        }


def _conv_prepass(layer_id, input_channels=3):
    return PrepassRef(
        ref_id=f"m0:{layer_id}:conv:mnsim20_physical_xbar_v2",
        model_id="m0",
        layer_id=layer_id,
        layer_type="conv",
        resource_version="mnsim20_physical_xbar_v2",
        sim_config_fingerprint="xbar=128x128;device=256;weight=8;polarity=2",
        layer_config_fingerprint=f"layer:{layer_id}",
        weight_ref_id=f"w:{layer_id}",
        shape_signature={
            "Inputchannel": input_channels,
            "Outputchannel": 16,
            "Kernelsize": 1,
            "Inputsize": [8, 8],
            "Outputsize": [8, 8],
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        },
    )


def _conv_entry(layer_id, input_channels=3):
    return SelectedEntry(
        layer_id,
        "conv",
        0,
        2,
        2,
        True,
        _conv_prepass(layer_id, input_channels),
    )


def _model_input():
    return {
        "model_input": {
            "shape": {"Inputchannel": 3, "Inputsize": [8, 8], "Inputbit": 8},
            "tensor_role": "activation",
        }
    }


def _required_xbar_count(entry):
    shape = entry.prepass_ref.shape_signature
    if entry.layer_type == "conv":
        input_count = int(shape["Inputchannel"])
        output_count = int(shape["Outputchannel"])
        input_block_size = 128 // (int(shape["Kernelsize"]) ** 2)
    else:
        input_count = int(shape["Infeature"])
        output_count = int(shape["Outfeature"])
        input_block_size = 128
    fingerprint = dict(
        part.split("=", 1)
        for part in entry.prepass_ref.sim_config_fingerprint.split(";")
    )
    device_level = int(fingerprint["device"])
    polarity = int(fingerprint["polarity"])
    cell_bits = device_bits_from_level(device_level)
    precision_bits = int(shape["Weightbit"]) - 1 if polarity == 2 else int(shape["Weightbit"])
    return (
        math.ceil(input_count / input_block_size)
        * math.ceil(output_count / 128)
        * math.ceil(precision_bits / cell_bits)
        * polarity
    )


def _round_plan(items):
    descriptors_by_layer = {}
    for item in items:
        for entry in item.selected_entries:
            descriptors_by_layer.setdefault(
                entry.layer_id,
                LayerResourceDescriptor(
                    layer_id=entry.layer_id,
                    layer_type=entry.layer_type,
                    input_layer_ids=[],
                    output_consumer_ids=[],
                    required_xbar_count=_required_xbar_count(entry),
                    prepass_ref=entry.prepass_ref,
                ),
            )
    return RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 128},
        execution_items=items,
        layer_resource_descriptors=list(descriptors_by_layer.values()),
        model_graph_ref={"model_id": "m0"},
        feasible=True,
    )


def _small_w_relu_w_plan():
    conv1 = _conv_entry("Conv1")
    conv2 = _conv_entry("Conv2", input_channels=16)
    conv1_node = LocalGraphNode("Conv1", "conv", "weighted")
    relu_node = LocalGraphNode("Relu1", "relu", "hook")
    conv2_node = LocalGraphNode("Conv2", "conv", "weighted")
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[conv1, conv2],
        local_graph_nodes=[conv1_node, relu_node, conv2_node],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["R0"],
                consumer_node_ids=["Conv1"],
            )
        ],
        same_item_edges=[
            SameItemEdge("Conv1", "Relu1"),
            SameItemEdge("Relu1", "Conv2"),
        ],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 4),
        ordered_segments=[
            WeightedLayerSegment(0, conv1, conv1_node),
            ZeroXbarSegment(1, "hook_only", [relu_node]),
            WeightedLayerSegment(2, conv2, conv2_node),
        ],
    )
    return _round_plan([item])


def _two_round_chain_plan():
    conv1 = _conv_entry("Conv1")
    conv2 = _conv_entry("Conv2", input_channels=16)
    conv1_node = LocalGraphNode("Conv1", "conv", "weighted")
    conv2_node = LocalGraphNode("Conv2", "conv", "weighted")
    first = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[conv1],
        local_graph_nodes=[conv1_node],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["R0"],
                consumer_node_ids=["Conv1"],
            )
        ],
        same_item_edges=[],
        external_outputs=[
            BoundaryRef(
                state_id="state:Conv1",
                tensor_role="activation",
                producer_item_id="R0",
                consumer_item_ids=["R1"],
            )
        ],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 2),
        ordered_segments=[WeightedLayerSegment(0, conv1, conv1_node)],
    )
    second = WeightedPhysicalRound(
        item_id="R1",
        item_index=1,
        round_kind="full_layer_pack",
        selected_entries=[conv2],
        local_graph_nodes=[conv2_node],
        external_inputs=[
            BoundaryRef(
                state_id="state:Conv1",
                tensor_role="activation",
                producer_item_id="R0",
                consumer_item_ids=["R1"],
                consumer_node_ids=["Conv2"],
            )
        ],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 2),
        ordered_segments=[WeightedLayerSegment(0, conv2, conv2_node)],
    )
    return _round_plan([first, second])


def _fragment_config_path(tmp_path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("SimConfig.ini", encoding="UTF-8")
    parser.set("Crossbar level", "Xbar_Size", "128,128")
    parser.set("Crossbar level", "Subarray_Size", "128")
    parser.set("Architecture level", "Tile_Num", "2,1")
    parser.set("Tile level", "PE_Num", "2,1")
    parser.set("Process element level", "Group_Num", "4")
    parser.set("Process element level", "Xbar_Polarity", "2")
    parser.set("Process element level", "DAC_Num", "128")
    parser.set("Process element level", "ADC_Num", "128")
    path = tmp_path / "SimConfig.fragment-executor.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return str(path)


def _two_xbar_split_fc_plan():
    prepass = PrepassRef(
        ref_id="m0:FC0:fc:mnsim20_physical_xbar_v2",
        model_id="m0",
        layer_id="FC0",
        layer_type="fc",
        resource_version="mnsim20_physical_xbar_v2",
        sim_config_fingerprint="xbar=128x128;device=256;weight=8;polarity=2",
        layer_config_fingerprint="layer:FC0",
        weight_ref_id="w:FC0",
        shape_signature={
            "Infeature": 256,
            "Outfeature": 128,
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        },
    )
    first_entry = SelectedEntry("FC0", "fc", 0, 2, 2, False, prepass)
    final_entry = SelectedEntry("FC0", "fc", 2, 4, 2, True, prepass)
    first_node = LocalGraphNode("FC0", "fc", "weighted")
    final_node = LocalGraphNode("FC0", "fc", "weighted")
    first = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="single_layer_split",
        selected_entries=[first_entry],
        local_graph_nodes=[first_node],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["R0"],
                consumer_node_ids=["FC0"],
            )
        ],
        same_item_edges=[],
        external_outputs=[
            BoundaryRef(
                state_id="state:FC0",
                tensor_role="activation",
                producer_item_id="R0",
                consumer_item_ids=[],
                partial_or_complete="partial",
            )
        ],
        partial_output_contract=[
            BoundaryRef(
                state_id="state:FC0",
                tensor_role="activation",
                producer_item_id="R0",
                consumer_item_ids=[],
                partial_or_complete="partial",
            )
        ],
        resource_proof=ResourceProof(2, 2),
        ordered_segments=[WeightedLayerSegment(0, first_entry, first_node)],
    )
    final = WeightedPhysicalRound(
        item_id="R1",
        item_index=1,
        round_kind="single_layer_continuation",
        selected_entries=[final_entry],
        local_graph_nodes=[final_node],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                producer_item_id=None,
                consumer_item_ids=["R1"],
                consumer_node_ids=["FC0"],
            )
        ],
        same_item_edges=[],
        external_outputs=[
            BoundaryRef(
                state_id="state:FC0",
                tensor_role="activation",
                producer_item_id="R1",
                consumer_item_ids=[],
            )
        ],
        partial_output_contract=[
            BoundaryRef(
                state_id="state:FC0",
                tensor_role="activation",
                producer_item_id="R1",
                consumer_item_ids=[],
                partial_or_complete="partial",
            )
        ],
        resource_proof=ResourceProof(2, 2),
        ordered_segments=[WeightedLayerSegment(0, final_entry, final_node)],
    )
    return _round_plan([first, final])


def test_executes_one_weighted_round_end_to_end_with_fake_backend(tmp_path):
    prepass = PrepassRef(
        ref_id="m0:L0:conv:mnsim20_physical_xbar_v2",
        model_id="m0",
        layer_id="L0",
        layer_type="conv",
        resource_version="mnsim20_physical_xbar_v2",
        sim_config_fingerprint="xbar=128x128;device=256;weight=8;polarity=2",
        layer_config_fingerprint="layer:test",
        weight_ref_id="w0",
        shape_signature={
            "Inputchannel": 3,
            "Outputchannel": 16,
            "Kernelsize": 3,
            "Inputsize": [32, 32],
            "Outputsize": [30, 30],
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 8,
        },
    )
    entry = SelectedEntry("L0", "conv", 0, 2, 2, True, prepass)
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[entry],
        local_graph_nodes=[],
        external_inputs=[],
        same_item_edges=[],
        external_outputs=[],
        partial_output_contract=[],
        resource_proof=ResourceProof(128, 2),
        ordered_segments=[WeightedLayerSegment(0, entry, LocalGraphNode("L0", "conv", "weighted"))],
    )
    plan = _round_plan([item])

    report = execute_round_plan(
        plan,
        sim_config_path=_fragment_config_path(tmp_path),
        ops_total=2_000_000,
        ops_count_policy="mac_as_1_op",
        static_backend=FakeStaticBackend(),
        model_inputs={"model_input": {"shape": {"C": 3}, "tensor_role": "activation"}},
    )

    assert report.feasible is True
    assert report.execution_item_count == 1
    assert report.inference_metrics["latency_s"] == 1e-6
    assert report.inference_metrics["read_energy_j"] == 2e-6
    assert report.inference_metrics["static_energy_j"] > 0.0
    assert report.inference_metrics["energy_j"] == pytest.approx(
        report.inference_metrics["read_energy_j"]
        + report.inference_metrics["static_energy_j"]
    )


def test_executor_commits_once_per_top_level_round_not_once_per_ordered_step(tmp_path):
    report = execute_round_plan(
        _small_w_relu_w_plan(),
        sim_config_path=_fragment_config_path(tmp_path),
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=FakeStaticBackend(),
        model_inputs=_model_input(),
    )

    assert report.execution_item_count == 1
    assert len(report.item_results) == 1
    assert [step["step_id"] for step in report.item_results[0]["steps"]] == [
        "R0.S0",
        "R0.S1",
        "R0.S2",
    ]
    assert [
        event.event_kind
        for event in report.boundary_events
        if "boundary" in event.event_kind
    ] == []


def test_executor_audits_physical_xbar_proof_without_rescaling_step_metrics(tmp_path):
    report = execute_round_plan(
        _small_w_relu_w_plan(),
        sim_config_path=_fragment_config_path(tmp_path),
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=FakeStaticBackend(),
        model_inputs=_model_input(),
    )

    static_audits = [
        entry
        for entry in report.raw_mnsim_audit
        if entry["audit_kind"] == "static" and entry["item_id"] == "R0"
    ]

    assert report.item_results[0]["latency_s"] == 2e-6
    assert report.inference_metrics["read_energy_j"] == 4e-6
    assert report.item_results[0]["static_energy_j"] > 0.0
    assert report.item_results[0]["energy_j"] == pytest.approx(
        report.inference_metrics["read_energy_j"]
        + report.item_results[0]["static_energy_j"]
    )
    assert sum(
        entry["raw_mnsim_summary"]["active_physical_xbar_count"]
        for entry in static_audits
    ) == 4
    assert report.item_results[0]["fragment_resource_proof"] == {
        "used_xbars": 4,
        "active_group_count": 2,
        "active_physical_xbar_count": 4,
    }


def test_executor_rejects_weighted_item_fragment_proof_mismatch():
    item = _small_w_relu_w_plan().execution_items[0]
    malformed_static_step = type(
        "MalformedStaticStep",
        (),
        {
            "source_kind": "static_eval",
            "raw_mnsim_audit": [
                {"raw_mnsim_summary": {"active_physical_xbar_count": 2}}
            ],
        },
    )()

    with pytest.raises(Phase2Invalid) as exc_info:
        executor_module._validate_weighted_item_fragment_proof(
            item,
            [malformed_static_step],
        )

    assert exc_info.value.code == TCG_ACTIVE_XBAR_PROOF_MISMATCH


def test_executor_records_one_cross_round_write_then_read(tmp_path):
    report = execute_round_plan(
        _two_round_chain_plan(),
        sim_config_path=_fragment_config_path(tmp_path),
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=FakeStaticBackend(),
        model_inputs=_model_input(),
    )

    writes = [event for event in report.boundary_events if event.event_kind == "write_boundary_state"]
    reads = [event for event in report.boundary_events if event.event_kind == "read_boundary_state"]

    assert [(event.item_id, event.state_id) for event in writes] == [("R0", "state:Conv1")]
    assert [(event.item_id, event.state_id) for event in reads] == [("R1", "state:Conv1")]


def test_executor_programs_each_weighted_round_before_its_read_phase(tmp_path, monkeypatch):
    events = []

    class RecordingWriteLedger:
        def __init__(self, *, sim_config_path, weight_codes_by_layer):
            events.append(("ledger", sim_config_path, dict(weight_codes_by_layer)))

        def score_round(self, *, item_id, placement):
            assert placement is not None
            events.append(("write", item_id))
            return {
                "item_id": item_id,
                "latency_s": 4e-9,
                "energy_j": 8e-9,
            }

    backend = FakeStaticBackend()
    original_run = backend.run

    def record_read(request):
        events.append(("read", request.item_id))
        return original_run(request)

    backend.run = record_read
    monkeypatch.setattr(executor_module, "AnnRoundPhysicalWriteLedger", RecordingWriteLedger)

    report = execute_round_plan(
        _two_round_chain_plan(),
        sim_config_path=_fragment_config_path(tmp_path),
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=backend,
        model_inputs=_model_input(),
        weight_codes_by_layer={"Conv1": {}, "Conv2": {}},
    )

    phase_events = [event[:2] for event in events if event[0] in {"write", "read"}]
    assert phase_events == [
        ("write", "R0"),
        ("read", "R0"),
        ("write", "R1"),
        ("read", "R1"),
    ]
    boundary_latency_s = report.inference_metrics[
        "boundary_sram_access_latency_s"
    ]
    boundary_energy_j = report.inference_metrics[
        "boundary_sram_access_energy_j"
    ]
    assert boundary_latency_s > 0
    assert boundary_energy_j > 0
    assert report.inference_metrics["read_latency_s"] == pytest.approx(
        2e-6 + boundary_latency_s
    )
    assert report.inference_metrics["weight_programming_latency_s"] == 8e-9
    assert report.inference_metrics["latency_s"] == pytest.approx(
        2.008e-6 + boundary_latency_s
    )
    assert report.inference_metrics["read_energy_j"] == pytest.approx(
        4e-6 + boundary_energy_j
    )
    assert report.inference_metrics["weight_programming_energy_j"] == 16e-9
    assert report.inference_metrics["static_energy_j"] > 0.0
    assert report.inference_metrics["energy_j"] == pytest.approx(
        report.inference_metrics["read_energy_j"]
        + report.inference_metrics["weight_programming_energy_j"]
        + report.inference_metrics["static_energy_j"]
    )
    assert report.inference_metrics["accounting_scope"] == {
        "weight_programming": {
            "energy_included": True,
            "latency_included": True,
            "state_policy": "erased_initial_then_active_slot_overwrite_idle_retained",
            "cell_transition_energy_included": True,
            "peripheral_energy_included": False,
        },
        "ddr_boundary_transfer": {
            "used": False,
            "energy_included": False,
            "latency_included": False,
        },
        "cross_round_boundary_sram": {
            "capacity_included": True,
            "retention_energy_included": True,
            "dynamic_access_energy_included": True,
            "latency_included": True,
            "spill_to_ddr": False,
        },
        "static_leakage": {
            "buffer_energy_included": True,
            "other_energy_included": False,
            "legacy_access_window_buffer_leakage_included": False,
            "central_sram_retention_energy_included": True,
        },
        "scheduled_partial_accumulator": {
            "energy_included": True,
            "latency_included": True,
            "partial_state_buffer_included": True,
            "primitive": "MNSIM.Hardware_Model.Adder+Buffer",
        },
    }


def test_split_continuation_commits_partial_only_at_round_boundaries(tmp_path):
    plan = _two_xbar_split_fc_plan()
    plan_before_execution = deepcopy(plan.to_dict())
    backend = FakeStaticBackend()
    report = execute_round_plan(
        plan,
        sim_config_path=_fragment_config_path(tmp_path),
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=backend,
        model_inputs={"model_input": {"shape": {"Infeature": 256}, "tensor_role": "activation"}},
    )

    assert [item["item_id"] for item in report.item_results] == ["R0", "R1"]
    assert report.item_results[0]["boundary_report"]["completion_updates"][0]["completion_status"] == "partial_sum"
    assert report.item_results[1]["boundary_report"]["completion_updates"][0]["completion_status"] == "complete_output"
    assert [step["step_id"] for step in report.item_results[0]["steps"]] == ["R0.S0"]
    assert plan.to_dict() == plan_before_execution
    assert all(request.fragment_layout is not None for request in backend.requests)


@pytest.mark.parametrize("descriptor_state", ["missing", "duplicate"])
def test_executor_rejects_missing_or_duplicate_required_xbar_descriptors(
    tmp_path,
    descriptor_state,
):
    plan = _two_xbar_split_fc_plan()
    if descriptor_state == "missing":
        plan.layer_resource_descriptors = []
    else:
        plan.layer_resource_descriptors.append(
            deepcopy(plan.layer_resource_descriptors[0])
        )

    with pytest.raises(Phase2Invalid) as exc_info:
        execute_round_plan(
            plan,
            sim_config_path=_fragment_config_path(tmp_path),
            ops_total=1,
            ops_count_policy="mac_as_1_op",
            static_backend=FakeStaticBackend(),
            model_inputs={
                "model_input": {
                    "shape": {"Infeature": 256},
                    "tensor_role": "activation",
                }
            },
        )

    assert exc_info.value.code == PHYSICAL_XBAR_UNIT_MISMATCH


def test_phase2_package_exports_executor_entrypoint():
    assert phase2.execute_round_plan is execute_round_plan
    assert "execute_round_plan" in phase2.__all__
