from fractions import Fraction

import pytest

import MNSIM.Scheduling.phase2.executor as executor_module
from MNSIM.Scheduling.digital_domain import (
    AccumulatorDomain,
    compile_lif_constants,
)
from MNSIM.Scheduling.phase2.executor import execute_round_plan
from MNSIM.Scheduling.round_types import (
    BoundaryRef,
    LayerResourceDescriptor,
    LocalGraphNode,
    PrepassRef,
    ResourceProof,
    RoundPlan,
    SameItemEdge,
    SelectedEntry,
    SequenceSpec,
    WeightedLayerSegment,
    WeightedPhysicalRound,
    ZeroXbarSegment,
)


class _Backend:
    def __init__(self):
        self.calls = []

    def run(self, request):
        self.calls.append(request)
        return {
            "layers": [
                {
                    "local_layer_id": 0,
                    "layer_type": "conv",
                    "begin_time_s": 0.0,
                    "finish_time_s": 1e-9,
                    "area_um2": 1.0,
                    "power_w": 1.0,
                    "energy_j": 1e-9,
                    "tcg_mapping": {"tilenum": 1},
                }
            ],
            "raw_totals": {
                "raw_latency_total_s": 1e-9,
                "raw_area_total_um2": 1.0,
                "raw_power_total_w": 1.0,
                "raw_energy_total_j": 1e-9,
            },
            "mapping_summary": {"layer_count": 1},
        }


def _plan():
    domain = AccumulatorDomain.from_scale(
        signed_bit_width=18, scale=Fraction(1, 256)
    )
    constants = compile_lif_constants(
        domain=domain, threshold=Fraction(1, 2), reset=0
    )
    metadata = {
        "accumulator_domain": domain.to_dict(),
        "compiled_lif_constants": constants.to_dict(),
    }
    prepass = PrepassRef(
        ref_id="m0:Conv1:conv:mnsim20_physical_xbar_v2",
        model_id="m0",
        layer_id="Conv1",
        layer_type="conv",
        resource_version="mnsim20_physical_xbar_v2",
        sim_config_fingerprint="xbar=128x128;device=256;weight=8;polarity=2",
        layer_config_fingerprint="layer:Conv1",
        weight_ref_id="w:Conv1",
        shape_signature={
            "Inputchannel": 3,
            "Outputchannel": 16,
            "Kernelsize": 1,
            "Inputsize": [1, 1],
            "Outputsize": [1, 1],
            "Inputbit": 8,
            "Weightbit": 8,
            "outputbit": 18,
        },
    )
    entry = SelectedEntry("Conv1", "conv", 0, 2, 2, True, prepass)
    conv = LocalGraphNode(
        "Conv1", "conv", "weighted", {"output_neuron_region": [0, 16]}, metadata
    )
    lif = LocalGraphNode(
        "L1", "lif", "hook", {"output_neuron_region": [0, 16]}, metadata
    )
    item = WeightedPhysicalRound(
        item_id="R0",
        item_index=0,
        round_kind="full_layer_pack",
        selected_entries=[entry],
        local_graph_nodes=[conv, lif],
        external_inputs=[
            BoundaryRef(
                state_id="model_input",
                tensor_role="activation",
                consumer_item_ids=["R0"],
                consumer_node_ids=["Conv1"],
            )
        ],
        same_item_edges=[SameItemEdge("Conv1", "L1")],
        external_outputs=[
            BoundaryRef(
                state_id="state:L1",
                tensor_role="spike",
                producer_item_id="R0",
            )
        ],
        partial_output_contract=[],
        resource_proof=ResourceProof(1152, 2),
        ordered_segments=[
            WeightedLayerSegment(0, entry, conv),
            ZeroXbarSegment(1, "hook_only", [lif]),
        ],
    )
    return RoundPlan(
        plan_id="plan:m0",
        model_id="m0",
        backend_kind="snn",
        hardware_budget={"xbar_size": [128, 128], "xbar_nums": 1152},
        execution_items=[item],
        layer_resource_descriptors=[
            LayerResourceDescriptor(
                layer_id="Conv1",
                layer_type="conv",
                input_layer_ids=[],
                output_consumer_ids=["L1"],
                required_xbar_count=2,
                prepass_ref=prepass,
            )
        ],
        model_graph_ref={"model_id": "m0"},
        feasible=True,
        sequence_spec=SequenceSpec(3, True, True, ["L1"]),
    )


def test_round_major_snn_programs_once_and_executes_all_timesteps(monkeypatch):
    writes = []

    class _Ledger:
        def __init__(self, **_kwargs):
            pass

        def score_round(self, *, item_id, placement):
            writes.append((item_id, placement.item_id))
            return {"latency_s": 2e-9, "energy_j": 3e-9}

    monkeypatch.setattr(executor_module, "AnnRoundPhysicalWriteLedger", _Ledger)
    backend = _Backend()
    report = execute_round_plan(
        _plan(),
        sim_config_path="SimConfig.ini",
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=backend,
        model_inputs={"model_input": {"shape": {"C": 3}}},
        weight_codes_by_layer={"Conv1": {}},
    )

    assert writes == [("R0", "R0")]
    assert len(backend.calls) == 3
    static_steps = [step for step in report.step_results if step["source_kind"] == "static_eval"]
    lif_steps = [step for step in report.step_results if step["source_kind"] == "digital_lif"]
    assert len(static_steps) == 3
    assert len(lif_steps) == 3
    assert [step["timestep"] for step in lif_steps] == [0, 1, 2]

    spike_writes = [
        event
        for event in report.boundary_events
        if event.event_kind == "write_boundary_state"
        and event.detail.get("payload_kind") == "spike_sequence"
    ]
    assert [event.detail["timestep"] for event in spike_writes] == [0, 1, 2]
    assert len({event.state_id for event in spike_writes}) == 3
    audits = [
        entry
        for entry in report.raw_mnsim_audit
        if entry.get("audit_kind") == "digital_lif"
    ]
    assert audits[0]["invocation_state_read_energy_nj"] == 0
    assert audits[-1]["invocation_state_write_energy_nj"] == 0
    assert all(entry["update_count"] == 16 * 3 for entry in audits)


def test_conv_channel_ownership_expands_to_spatial_lif_neurons():
    item = type(
        "Item",
        (),
        {
            "local_graph_nodes": [
                LocalGraphNode(
                    "Conv1",
                    "conv",
                    "weighted",
                    {"Outputchannel": 4, "Outputsize": [2, 3]},
                    {},
                )
            ]
        },
    )()

    expanded = executor_module._expand_output_ownership_to_neurons(
        item,
        "Conv1",
        {0: 1, 1: 3},
        expected_neuron_count=24,
    )

    assert expanded == {0: 6, 1: 18}


def test_output_ownership_merges_split_rounds_and_replaces_partial_reducer():
    def placement(item_id, fragments):
        specs = {}
        slots = {}
        for index, (fragment_id, output_range, input_block, tile_id) in enumerate(fragments):
            specs[fragment_id] = type(
                "Fragment",
                (),
                {
                    "layer_id": "Conv1",
                    "output_range": output_range,
                    "full_output_range": (0, 256),
                },
            )()
            slots[fragment_id] = {
                "tile_id": tile_id,
                "pe_id": 0,
                "group_id": index,
                "input_block_index": input_block,
            }
        return type(
            "Placement",
            (),
            {
                "item_id": item_id,
                "fragment_specs_by_id": specs,
                "fragment_slots_by_id": slots,
            },
        )()

    intervals = {}
    executor_module._update_output_ownership_intervals(
        intervals,
        placement("R0", [("f0", (0, 128), 0, 0)]),
    )
    executor_module._update_output_ownership_intervals(
        intervals,
        placement("R1", [("f1", (0, 128), 1, 1)]),
    )
    executor_module._update_output_ownership_intervals(
        intervals,
        placement("R2", [("f2", (128, 256), 0, 0)]),
    )

    ownership = executor_module._aggregate_output_ownership_intervals(intervals)

    assert ownership == {"Conv1": {0: 128, 1: 128}}
