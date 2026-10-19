import configparser
from dataclasses import replace
from pathlib import Path

import pytest

from MNSIM.Interface.network import get_net
from MNSIM.Scheduling.phase1.config import build_hardware_projection
from MNSIM.Scheduling.phase1.mnsim_frontend import build_model_ir_from_mnsim_network
from MNSIM.Scheduling.phase1.round_planner import build_round_plan
from MNSIM.Scheduling.phase2.executor import execute_round_plan
from MNSIM.Scheduling.phase2.mnsim_fragment_adapter import MnsimFragmentAdapter


class EchoStaticBackend:
    def run(self, request):
        layers = []
        for local_layer_id, layer_entry in enumerate(request.netstruct):
            layer_type = layer_entry[0][0]["type"]
            begin_time_s = local_layer_id * 1e-9
            finish_time_s = begin_time_s + 1e-9
            layers.append(
                {
                    "local_layer_id": local_layer_id,
                    "layer_type": layer_type,
                    "begin_time_s": begin_time_s,
                    "finish_time_s": finish_time_s,
                    "area_um2": 10.0,
                    "power_w": 2.0,
                    "energy_j": 2e-9,
                    "tcg_mapping": {"tilenum": 1},
                }
            )
        return {
            "layers": layers,
            "raw_totals": {
                "raw_latency_total_s": len(layers) * 1e-9,
                "raw_area_total_um2": len(layers) * 10.0,
                "raw_power_total_w": len(layers) * 2.0,
                "raw_energy_total_j": len(layers) * 2e-9,
            },
            "mapping_summary": {"layer_count": len(layers)},
        }


class _ProofOnlyPlacement:
    def __init__(self, item_id, active_group_count, active_physical_xbar_count):
        self.item_id = item_id
        self.mapped_area_um2 = 0.0
        self.mapping_summary = {
            "active_group_count": active_group_count,
            "active_physical_xbar_count": active_physical_xbar_count,
        }

    def slice_for(self, _step_id, _scope_index, _netstruct):
        return None


def _proof_only_round_placement(*, item_id, weighted_scopes, expected_used_xbars, **_kwargs):
    active_group_count = sum(
        scope.fragment_layout.active_group_count for scope in weighted_scopes
    )
    active_physical_xbar_count = sum(
        scope.fragment_layout.active_physical_xbar_count for scope in weighted_scopes
    )
    assert active_physical_xbar_count == expected_used_xbars
    return _ProofOnlyPlacement(
        item_id,
        active_group_count,
        active_physical_xbar_count,
    )


def _g8_sim_config(*, device_level=256):
    return {
        "Xbar_Size": [128, 128],
        "Tile_Num": [6, 6],
        "PE_Num": [4, 4],
        "Group_Num": 2,
        "Device_Level": device_level,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }


def _physical_two_xbar_config():
    return {
        "Xbar_Size": [128, 128],
        "Tile_Num": [1, 1],
        "PE_Num": [1, 1],
        "Group_Num": 1,
        "Device_Level": 2,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }


def _mnsim_network(category):
    return get_net(
        hardware_config={
            "xbar_size": 128,
            "input_bit": 9,
            "weight_bit": 8,
            "ADC_quantize_bit": 10,
            "DAC_num": 64,
            "xbar_polarity": 2,
        },
        cate=category,
        num_classes=10,
    )


def _set_test_device_level(parser, device_level):
    source_resistance = [
        value.strip()
        for value in parser.get("Device level", "Device_Resistance").split(",")
    ]
    assert len(source_resistance) >= device_level
    indexes = [
        round(index * (len(source_resistance) - 1) / (device_level - 1))
        for index in range(device_level)
    ]
    parser.set("Device level", "Device_Level", str(device_level))
    parser.set(
        "Device level",
        "Device_Resistance",
        ",".join(source_resistance[index] for index in indexes),
    )


def test_set_test_device_level_samples_measured_resistance_at_equal_intervals():
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(Path(__file__).resolve().parents[1] / "SimConfig.ini", encoding="UTF-8")
    source_resistance = [
        value.strip()
        for value in parser.get("Device level", "Device_Resistance").split(",")
    ]

    _set_test_device_level(parser, 4)

    assert parser.getint("Device level", "Device_Level") == 4
    assert parser.get("Device level", "Device_Resistance").split(",") == [
        source_resistance[0],
        source_resistance[85],
        source_resistance[170],
        source_resistance[255],
    ]


def _sim_config_path_for_backend(tmp_path, sim_config):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("SimConfig.ini", encoding="UTF-8")
    _set_test_device_level(parser, sim_config["Device_Level"])
    parser.set("Crossbar level", "Xbar_Size", ",".join(map(str, sim_config["Xbar_Size"])))
    parser.set("Crossbar level", "Subarray_Size", str(sim_config["Xbar_Size"][0]))
    parser.set("Architecture level", "Tile_Num", ",".join(map(str, sim_config["Tile_Num"])))
    parser.set("Tile level", "PE_Num", ",".join(map(str, sim_config["PE_Num"])))
    parser.set("Process element level", "Group_Num", str(sim_config["Group_Num"]))
    parser.set("Process element level", "Xbar_Polarity", str(sim_config["Xbar_Polarity"]))
    parser.set(
        "Algorithm Configuration",
        "Weight_precision",
        str(sim_config["Weight_precision"]),
    )
    parser.set("Process element level", "DAC_Num", "128")
    parser.set("Process element level", "ADC_Num", "128")
    path = tmp_path / "SimConfig.round-plan.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return str(path)


def _assert_backend_config_matches_phase1(sim_config_path, sim_config):
    parser = configparser.ConfigParser()
    parser.read(sim_config_path, encoding="UTF-8")
    assert parser.get("Crossbar level", "Xbar_Size") == ",".join(
        map(str, sim_config["Xbar_Size"])
    )
    assert [
        int(value)
        for value in parser.get("Architecture level", "Tile_Num").replace(",", " ").split()
    ] == sim_config["Tile_Num"]
    assert [
        int(value)
        for value in parser.get("Tile level", "PE_Num").replace(",", " ").split()
    ] == sim_config["PE_Num"]
    assert parser.getint("Process element level", "Group_Num") == sim_config["Group_Num"]
    assert parser.getint("Process element level", "Xbar_Polarity") == sim_config["Xbar_Polarity"]
    assert parser.getint("Device level", "Device_Level") == sim_config["Device_Level"]
    assert len(parser.get("Device level", "Device_Resistance").split(",")) == sim_config[
        "Device_Level"
    ]
    assert parser.getint("Algorithm Configuration", "Weight_precision") == sim_config[
        "Weight_precision"
    ]


def _model_inputs():
    return {
        "model_input": {
            "shape": {"Inputchannel": 3, "Inputsize": [32, 32], "Inputbit": 8},
            "tensor_role": "activation",
        }
    }


def _weighted_items(plan):
    return [item for item in plan.execution_items if item.item_kind == "weighted_physical_round"]


def _zero_xbar_items(plan):
    return [item for item in plan.execution_items if item.item_kind == "zero_xbar_round"]


def _segment_layer_ids(item):
    return [
        segment.selected_entry.layer_id
        if segment.segment_kind == "weighted_layer_segment"
        else [node.layer_id for node in segment.local_graph_nodes]
        for segment in item.ordered_segments
    ]


def _assert_report_consumes_plan(plan, report):
    assert report.feasible is True
    assert report.execution_item_count == len(plan.execution_items)
    assert [record["item_id"] for record in report.item_results] == [
        item.item_id for item in plan.execution_items
    ]
    assert all("steps" in record for record in report.item_results)
    for record, item in zip(report.item_results, plan.execution_items):
        assert record["used_xbars"] == item.resource_proof.used_xbars
        if item.item_kind == "weighted_physical_round":
            proof = record["fragment_resource_proof"]
            assert proof["used_xbars"] == item.resource_proof.used_xbars
            assert (
                proof["active_physical_xbar_count"]
                == item.resource_proof.used_xbars
            )


def _assert_weighted_plan_resource_proof(plan):
    weighted_items = _weighted_items(plan)
    assert weighted_items
    assert all(
        item.resource_proof.used_xbars <= plan.hardware_budget["xbar_nums"]
        for item in weighted_items
    )
    assert sum(
        entry.used_xbars
        for item in weighted_items
        for entry in item.selected_entries
    ) == sum(
        descriptor.required_xbar_count
        for descriptor in plan.layer_resource_descriptors
    )


def _assert_partial_report_fragment_coverage(plan, sim_config_path, report):
    required_xbars_by_layer = {
        descriptor.layer_id: descriptor.required_xbar_count
        for descriptor in plan.layer_resource_descriptors
    }
    adapter = MnsimFragmentAdapter(sim_config_path, required_xbars_by_layer)
    representative_entries = {}
    for item in _weighted_items(plan):
        for entry in item.selected_entries:
            representative_entries.setdefault(entry.layer_id, entry)

    expected_fragment_ids = {
        fragment.fragment_id
        for entry in representative_entries.values()
        for fragment in adapter.inventory_for(entry).fragments
    }
    partial_updates = [
        update
        for record in report.item_results
        for update in record["boundary_report"]["completion_updates"]
        if "selected_fragment_ids" in update
    ]
    reported_fragment_ids = [
        fragment_id
        for update in partial_updates
        for fragment_id in update["selected_fragment_ids"]
    ]

    assert len(reported_fragment_ids) == len(set(reported_fragment_ids))
    assert set(reported_fragment_ids) == expected_fragment_ids
    assert all(
        update["covered_fragment_count"] <= update["required_fragment_count"]
        for update in partial_updates
    )
    assert all(
        update["completion_status"] == "complete_output"
        and update["covered_fragment_count"] == update["required_fragment_count"]
        for update in partial_updates
        if update["downstream_visible"]
    )


def _first_two_xbar_continuation_window(plan):
    first, second = _weighted_items(plan)[:2]
    first_entry = first.selected_entries[0]
    second_entry = second.selected_entries[0]
    assert first_entry.layer_id == second_entry.layer_id
    assert (first_entry.xbar_range_start, first_entry.xbar_range_end) == (0, 2)
    assert (second_entry.xbar_range_start, second_entry.xbar_range_end) == (2, 4)
    return replace(plan, execution_items=[first, second])


def _assert_element_sums_are_ordered_structural_steps(plan):
    for item in _weighted_items(plan):
        segment_index_by_node_id = {
            node.layer_id: index
            for index, segment in enumerate(item.ordered_segments)
            for node in (
                [segment.local_graph_node]
                if segment.segment_kind == "weighted_layer_segment"
                else segment.local_graph_nodes
            )
        }
        for element_sum_id, element_sum_index in (
            (node.layer_id, index)
            for index, segment in enumerate(item.ordered_segments)
            if segment.segment_kind == "zero_xbar_segment"
            for node in segment.local_graph_nodes
            if node.layer_type == "element_sum"
        ):
            incoming_local = [
                edge.producer_node_id
                for edge in item.same_item_edges
                if edge.consumer_node_id == element_sum_id
            ]
            incoming_boundary = [
                ref.state_id
                for ref in item.external_inputs
                if element_sum_id in ref.consumer_node_ids
            ]
            outgoing_local = [
                edge.consumer_node_id
                for edge in item.same_item_edges
                if edge.producer_node_id == element_sum_id
            ]
            assert incoming_local or incoming_boundary
            assert outgoing_local
            assert all(segment_index_by_node_id[node_id] < element_sum_index for node_id in incoming_local)
            assert all(element_sum_index < segment_index_by_node_id[node_id] for node_id in outgoing_local)


def _has_weighted_zero_weighted_chain(record):
    step_kinds = [step["step_kind"] for step in record["steps"]]
    for first_weighted_index, step_kind in enumerate(step_kinds):
        if step_kind != "weighted_layer_segment":
            continue
        next_weighted_index = first_weighted_index + 1
        while (
            next_weighted_index < len(step_kinds)
            and step_kinds[next_weighted_index] == "zero_xbar_segment"
        ):
            next_weighted_index += 1
        if (
            next_weighted_index > first_weighted_index + 1
            and next_weighted_index < len(step_kinds)
            and step_kinds[next_weighted_index] == "weighted_layer_segment"
        ):
            return True
    return False


def test_vgg8_phase1_round_plan_feeds_phase2_executor_with_ordered_steps(tmp_path):
    model_ir = build_model_ir_from_mnsim_network(
        _mnsim_network("vgg8"),
        model_id="real_vgg8_phase2",
    )
    plan = build_round_plan(model_ir, _g8_sim_config())
    weighted_items = _weighted_items(plan)

    assert plan.feasible is True
    assert build_hardware_projection(_g8_sim_config()).xbar_nums == 2304
    _assert_weighted_plan_resource_proof(plan)
    assert _zero_xbar_items(plan) == []
    assert [
        entry.layer_id
        for item in weighted_items
        for entry in item.selected_entries
    ] == [
        layer.layer_id for layer in model_ir.layers if layer.is_weighted
    ]

    backend_config_path = _sim_config_path_for_backend(tmp_path, _g8_sim_config())
    _assert_backend_config_matches_phase1(backend_config_path, _g8_sim_config())
    report = execute_round_plan(
        plan,
        sim_config_path=backend_config_path,
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=EchoStaticBackend(),
        model_inputs=_model_inputs(),
    )

    _assert_report_consumes_plan(plan, report)
    assert any(_has_weighted_zero_weighted_chain(record) for record in report.item_results)


def test_resnet18_phase1_round_plan_feeds_phase2_executor_with_ordered_structural_steps(
    tmp_path,
):
    model_ir = build_model_ir_from_mnsim_network(
        _mnsim_network("resnet18"),
        model_id="real_resnet18_phase2",
    )
    plan = build_round_plan(model_ir, _g8_sim_config())
    weighted_items = _weighted_items(plan)

    assert plan.feasible is True
    assert build_hardware_projection(_g8_sim_config()).xbar_nums == 2304
    _assert_weighted_plan_resource_proof(plan)
    assert _zero_xbar_items(plan) == []
    assert [entry.layer_id for item in weighted_items for entry in item.selected_entries] == [
        layer.layer_id for layer in model_ir.layers if layer.is_weighted
    ]
    _assert_element_sums_are_ordered_structural_steps(plan)

    backend_config_path = _sim_config_path_for_backend(tmp_path, _g8_sim_config())
    _assert_backend_config_matches_phase1(backend_config_path, _g8_sim_config())
    report = execute_round_plan(
        plan,
        sim_config_path=backend_config_path,
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=EchoStaticBackend(),
        model_inputs=_model_inputs(),
    )

    _assert_report_consumes_plan(plan, report)
    assert any(
        any("element_sum" in step["origin_layer_ids"][0] for step in record["steps"])
        for record in report.item_results
    )


@pytest.mark.parametrize(
    ("category", "sim_config"),
    [
        ("vgg8", _g8_sim_config()),
        ("resnet18", _g8_sim_config()),
        ("vgg8", _g8_sim_config(device_level=4)),
    ],
    ids=["vgg8-8bit", "resnet18-8bit", "vgg8-2bit"],
)
def test_g8_round_plan_runs_original_mnsim_backend(
    tmp_path,
    category,
    sim_config,
):
    model_ir = build_model_ir_from_mnsim_network(
        _mnsim_network(category),
        model_id=f"real_{category}_original_backend",
    )
    plan = build_round_plan(model_ir, sim_config)

    assert plan.feasible is True
    assert build_hardware_projection(sim_config).xbar_nums == 2304
    _assert_weighted_plan_resource_proof(plan)
    backend_config_path = _sim_config_path_for_backend(tmp_path, sim_config)
    _assert_backend_config_matches_phase1(backend_config_path, sim_config)

    report = execute_round_plan(
        plan,
        sim_config_path=backend_config_path,
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        model_inputs=_model_inputs(),
    )

    _assert_report_consumes_plan(plan, report)
    assert report.warnings == [
        "Stage-A static power is an SRAM-only CACTI architecture estimate.",
        "Static accounting is incomplete; unmodeled blocks are listed explicitly.",
    ]
    assert report.inference_metrics["latency_s"] > 0
    assert report.inference_metrics["energy_j"] > 0
    assert report.inference_metrics["area_um2"] > 0


@pytest.mark.parametrize(
    "category",
    [
        "vgg8",
        "resnet18",
    ],
)
def test_two_xbar_plan_runs_fake_backend_with_continuation_boundaries(
    category,
    tmp_path,
    monkeypatch,
):
    model_ir = build_model_ir_from_mnsim_network(
        _mnsim_network(category),
        model_id=f"real_{category}_two_xbar_fake",
    )
    sim_config = _physical_two_xbar_config()
    plan = build_round_plan(model_ir, sim_config)

    assert plan.feasible is True
    assert build_hardware_projection(sim_config).xbar_nums == 2
    _assert_weighted_plan_resource_proof(plan)
    assert _zero_xbar_items(plan) == []
    backend_config_path = _sim_config_path_for_backend(tmp_path, sim_config)
    _assert_backend_config_matches_phase1(backend_config_path, sim_config)
    monkeypatch.setattr(
        "MNSIM.Scheduling.phase2.lowerer.prepare_round_placement",
        _proof_only_round_placement,
    )

    report = execute_round_plan(
        plan,
        sim_config_path=backend_config_path,
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=EchoStaticBackend(),
        model_inputs=_model_inputs(),
    )

    _assert_report_consumes_plan(plan, report)
    _assert_partial_report_fragment_coverage(plan, backend_config_path, report)
    partial_updates = [
        update
        for record in report.item_results
        for update in record["boundary_report"]["completion_updates"]
        if update["completion_status"] == "partial_sum"
    ]
    complete_updates = [
        update
        for record in report.item_results
        for update in record["boundary_report"]["completion_updates"]
        if update["completion_status"] == "complete_output"
    ]
    assert partial_updates
    assert complete_updates


@pytest.mark.parametrize("category", ["vgg8", "resnet18"])
def test_two_xbar_initial_continuation_planes_run_original_mnsim_backend(
    tmp_path,
    category,
):
    sim_config = _physical_two_xbar_config()
    model_ir = build_model_ir_from_mnsim_network(
        _mnsim_network(category),
        model_id=f"real_{category}_two_xbar_original_backend",
    )
    plan = _first_two_xbar_continuation_window(
        build_round_plan(model_ir, sim_config)
    )
    backend_config_path = _sim_config_path_for_backend(tmp_path, sim_config)

    assert build_hardware_projection(sim_config).xbar_nums == 2
    assert all(
        item.resource_proof.used_xbars <= plan.hardware_budget["xbar_nums"]
        for item in _weighted_items(plan)
    )
    _assert_backend_config_matches_phase1(backend_config_path, sim_config)

    report = execute_round_plan(
        plan,
        sim_config_path=backend_config_path,
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        model_inputs=_model_inputs(),
    )

    _assert_report_consumes_plan(plan, report)
    assert report.inference_metrics["latency_s"] > 0
    assert report.inference_metrics["energy_j"] > 0
