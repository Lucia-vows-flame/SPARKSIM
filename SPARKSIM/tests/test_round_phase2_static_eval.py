import configparser

import pytest

from MNSIM.Mapping_Model import Tile_connection_graph
import MNSIM.Scheduling.phase2 as phase2
import MNSIM.Scheduling.phase2.mnsim_static_eval as static_eval
from MNSIM.Scheduling.phase2.mnsim_static_eval import run_mnsim_static_eval
from MNSIM.Scheduling.phase2.netstruct_builder import build_netstruct
from MNSIM.Scheduling.phase2.tcg_adapter import WeightedScope, prepare_round_placement
from MNSIM.Scheduling.phase2.types import (
    BoundaryBinding,
    MetricTotals,
    MnsimFragmentLayout,
    MnsimGroupFragment,
    StaticEvalFrame,
    StaticEvalRequest,
)


class FakeBackend:
    def run(self, request):
        return {
            "layers": [
                {
                    "local_layer_id": 0,
                    "layer_type": "conv",
                    "begin_time_s": 0.0,
                    "finish_time_s": 5e-9,
                    "area_um2": 10.0,
                    "power_w": 2.0,
                    "energy_j": 10e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
                {
                    "local_layer_id": 1,
                    "layer_type": "pooling",
                    "begin_time_s": 5e-9,
                    "finish_time_s": 7e-9,
                    "area_um2": 3.0,
                    "power_w": 1.0,
                    "energy_j": 2e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
            ],
            "raw_totals": {
                "raw_latency_total_s": 7e-9,
                "raw_area_total_um2": 13.0,
                "raw_power_total_w": 3.0,
                "raw_energy_total_j": 12e-9,
            },
            "mapping_summary": {"layer_count": 2},
        }


class SingleLayerBackend:
    def run(self, request):
        return {
            "layers": [
                {
                    "local_layer_id": 0,
                    "layer_type": "fc",
                    "begin_time_s": 0.0,
                    "finish_time_s": 3e-9,
                    "area_um2": 5.0,
                    "power_w": 1.5,
                    "energy_j": 4e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
            ],
            "raw_totals": {
                "raw_latency_total_s": 3e-9,
                "raw_area_total_um2": 5.0,
                "raw_power_total_w": 1.5,
                "raw_energy_total_j": 4e-9,
            },
            "mapping_summary": {"layer_count": 1},
        }


class ExtraReportableLayerBackend:
    def run(self, request):
        return {
            "layers": [
                {
                    "local_layer_id": 0,
                    "layer_type": "conv",
                    "begin_time_s": 0.0,
                    "finish_time_s": 5e-9,
                    "area_um2": 10.0,
                    "power_w": 2.0,
                    "energy_j": 10e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
                {
                    "local_layer_id": 1,
                    "layer_type": "pooling",
                    "begin_time_s": 5e-9,
                    "finish_time_s": 7e-9,
                    "area_um2": 3.0,
                    "power_w": 1.0,
                    "energy_j": 2e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
                {
                    "local_layer_id": 2,
                    "layer_type": "pooling",
                    "begin_time_s": 7e-9,
                    "finish_time_s": 11e-9,
                    "area_um2": 4.0,
                    "power_w": 0.5,
                    "energy_j": 4e-9,
                    "tcg_mapping": {"tilenum": 1},
                },
            ],
            "raw_totals": {
                "raw_latency_total_s": 11e-9,
                "raw_area_total_um2": 17.0,
                "raw_power_total_w": 3.5,
                "raw_energy_total_j": 16e-9,
            },
            "mapping_summary": {"layer_count": 3},
        }


def _binding(local_layer_id, origin_layer_id, contribution_kind="complete_output"):
    return BoundaryBinding(
        local_layer_id=local_layer_id,
        hook_node_id=None,
        origin_layer_id=origin_layer_id,
        origin_layer_type="pooling" if origin_layer_id == "Pool1" else "conv",
        report_role="reportable",
        tensor_role="activation",
        shape={"kind": "full"},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind=contribution_kind,
        source_state_ids=[],
        downstream_consumer_ids=[],
    )


def _hook_binding(origin_layer_id):
    return BoundaryBinding(
        local_layer_id=None,
        hook_node_id=origin_layer_id,
        origin_layer_id=origin_layer_id,
        origin_layer_type="relu",
        report_role="reportable",
        tensor_role="activation",
        shape={"kind": "full"},
        region={"kind": "full"},
        input_range={"kind": "full"},
        output_range={"kind": "full"},
        full_input_range={"kind": "full"},
        full_output_range={"kind": "full"},
        contribution_kind="hook_output",
        source_state_ids=[],
        downstream_consumer_ids=[],
    )


def _request(
    *,
    item_id="Z0",
    item_kind="zero_xbar_structural",
    netstruct=None,
    multiple=None,
    latency_api="pipe",
    context_layer_ids=None,
    reportable_layer_ids=None,
    latency_offset_source_ids=None,
    boundary_bindings=None,
):
    return StaticEvalRequest(
        item_id=item_id,
        item_kind=item_kind,
        netstruct=netstruct
        if netstruct is not None
        else [[({"type": "conv"}, None)], [({"type": "pooling"}, None)]],
        sim_config_path="SimConfig.ini",
        multiple=multiple if multiple is not None else [1, 1],
        latency_api=latency_api,
        latency_fill_mode=0,
        frame=StaticEvalFrame(
            context_layer_ids=context_layer_ids if context_layer_ids is not None else [0],
            reportable_layer_ids=reportable_layer_ids if reportable_layer_ids is not None else [1],
            latency_offset_source_ids=latency_offset_source_ids
            if latency_offset_source_ids is not None
            else [0],
            boundary_bindings=boundary_bindings
            if boundary_bindings is not None
            else [_binding(1, "Pool1", "structural_output")],
        ),
    )


def _two_tile_config_path(tmp_path):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("SimConfig.ini", encoding="UTF-8")
    parser.set("Crossbar level", "Xbar_Size", "128,128")
    parser.set("Crossbar level", "Subarray_Size", "128")
    parser.set("Architecture level", "Tile_Num", "1,2")
    parser.set("Tile level", "PE_Num", "2,1")
    parser.set("Process element level", "Group_Num", "4")
    parser.set("Process element level", "Xbar_Polarity", "2")
    parser.set("Process element level", "DAC_Num", "128")
    parser.set("Process element level", "ADC_Num", "128")
    path = tmp_path / "two_tile.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return str(path)


def _conv_layer_info():
    return {
        "type": "conv",
        "Inputchannel": 1,
        "Outputchannel": 1,
        "Kernelsize": 1,
        "Inputsize": [1, 1],
        "Outputsize": [1, 1],
        "Stride": 1,
        "Padding": 0,
        "Inputbit": 8,
        "Weightbit": 2,
        "outputbit": 8,
        "row_split_num": 1,
        "weight_bit_split_part": 1,
    }


def _fragment_layout(
    *,
    layer_id="L000_conv",
    layer_type="conv",
    native_layer_info=None,
    active_group_count=1,
    active_physical_xbar_count=2,
):
    native_layer_info = native_layer_info or _conv_layer_info()
    fragments = tuple(
        MnsimGroupFragment(
            fragment_id=f"{layer_id}.g{index}",
            layer_id=layer_id,
            layer_type=layer_type,
            input_range=(0, 1),
            output_range=(0, 1),
            full_input_range=(0, 1),
            full_output_range=(0, 1),
            weight_plane_index=index,
            polarity_count=2,
            physical_xbar_count=2,
        )
        for index in range(active_group_count)
    )
    return MnsimFragmentLayout(
        layer_id=layer_id,
        layer_type=layer_type,
        fragments=fragments,
        native_layer_info=native_layer_info,
        active_group_count=active_group_count,
        active_physical_xbar_count=active_physical_xbar_count,
    )


def _weighted_conv_request_with_one_fragment(tmp_path):
    layer_info = _conv_layer_info()
    return StaticEvalRequest(
        item_id="R0",
        item_kind="weighted_layer_segment",
        netstruct=build_netstruct([layer_info]),
        sim_config_path=_two_tile_config_path(tmp_path),
        multiple=[1],
        latency_api="nopipe",
        latency_fill_mode=0,
        frame=StaticEvalFrame(
            context_layer_ids=[],
            reportable_layer_ids=[0],
            latency_offset_source_ids=[],
            boundary_bindings=[_binding(0, "L000_conv")],
        ),
        fragment_layout=_fragment_layout(native_layer_info=layer_info),
    )


def test_filters_context_layers_and_normalizes_latency_offset():
    request = _request()

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "ok"
    assert len(result.layer_results) == 2
    assert result.layer_results[0].report_role == "context_only"
    assert result.layer_results[1].begin_time_s == 0.0
    assert result.layer_results[1].finish_time_s == 2e-9
    assert result.item_totals == MetricTotals(
        latency_s=2e-9,
        energy_j=2e-9,
        power_w=1.0,
        area_um2=3.0,
    )
    assert result.raw_mnsim_summary["raw_latency_total_s"] == 7e-9


def test_rejects_multiple_length_mismatch():
    request = _request(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[[({"type": "conv"}, None)]],
        multiple=[1, 1],
        context_layer_ids=[],
        reportable_layer_ids=[0],
        latency_offset_source_ids=[],
        boundary_bindings=[_binding(0, "Conv1")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.warnings[0].startswith("netstruct_validation")


@pytest.mark.parametrize(
    ("netstruct", "context_ids", "reportable_ids", "offset_ids"),
    [
        ([[({"type": "conv"}, None)]], [1], [0], []),
        ([[({"type": "conv"}, None)]], [], [1], []),
        ([[({"type": "conv"}, None)]], [], [0], [1]),
    ],
)
def test_rejects_invalid_static_eval_frames(
    netstruct,
    context_ids,
    reportable_ids,
    offset_ids,
):
    request = _request(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=netstruct,
        multiple=[1],
        context_layer_ids=context_ids,
        reportable_layer_ids=reportable_ids,
        latency_offset_source_ids=offset_ids,
        boundary_bindings=[_binding(0, "Conv1")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.item_totals == MetricTotals(0.0, 0.0, 0.0, 0.0)
    assert result.warnings[0].startswith("netstruct_validation")


def test_rejects_zero_xbar_structural_without_reportable_layer():
    request = _request(
        item_kind="zero_xbar_structural",
        context_layer_ids=[0],
        reportable_layer_ids=[],
        latency_offset_source_ids=[0],
        boundary_bindings=[],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.warnings[0].startswith("netstruct_validation")


def test_element_sum_uses_direct_structural_eval_even_with_an_injected_backend():
    request = _request(
        item_kind="zero_xbar_structural",
        netstruct=[
            [
                (
                    {
                        "type": "element_sum",
                        "Inputindex": [-1, -2],
                        "Outputindex": [],
                        "Inputsize": [4, 4],
                        "Outputsize": [4, 4],
                        "Inputchannel": 8,
                        "Outputchannel": 8,
                        "Inputbit": 8,
                        "Weightbit": 8,
                        "outputbit": 8,
                        "Layerindex": 0,
                    },
                    None,
                )
            ]
        ],
        multiple=[1],
        context_layer_ids=[],
        reportable_layer_ids=[0],
        latency_offset_source_ids=[],
        boundary_bindings=[_binding(0, "Sum1", "structural_output")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "ok"
    assert result.mapping_summary["backend"] == "direct_zero_xbar_structural"


def test_concat_uses_buffer_routing_without_arithmetic_units():
    request = _request(
        item_kind="zero_xbar_structural",
        netstruct=[
            [
                (
                    {
                        "type": "concat",
                        "Inputindex": [-1, -2, -3],
                        "Outputindex": [],
                        "Inputsize": [4, 4],
                        "Outputsize": [4, 4],
                        "Inputchannel": 12,
                        "Outputchannel": 12,
                        "Inputbit": 1,
                        "Weightbit": 8,
                        "outputbit": 1,
                        "concat_dim": "channel",
                        "merge_ownership_policy": "concat_channel_segments",
                        "Layerindex": 0,
                    },
                    None,
                )
            ]
        ],
        multiple=[1],
        context_layer_ids=[],
        reportable_layer_ids=[0],
        latency_offset_source_ids=[],
        boundary_bindings=[_binding(0, "Concat1", "structural_output")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "ok"
    assert result.mapping_summary["backend"] == "direct_zero_xbar_structural"
    assert result.layer_results[0].tcg_mapping["concat_channel_router_num"] == 0
    assert result.layer_results[0].tcg_mapping["input_branch_count"] == 3
    assert result.item_totals.area_um2 > 0


def test_rejects_context_and_reportable_layer_overlap():
    request = _request(
        context_layer_ids=[0, 1],
        reportable_layer_ids=[1],
        latency_offset_source_ids=[0],
        boundary_bindings=[_binding(1, "Pool1", "structural_output")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.item_totals == MetricTotals(0.0, 0.0, 0.0, 0.0)
    assert result.warnings[0].startswith("netstruct_validation")


def test_rejects_latency_offset_source_that_is_not_context_layer():
    request = _request(
        context_layer_ids=[0],
        reportable_layer_ids=[1],
        latency_offset_source_ids=[1],
        boundary_bindings=[_binding(1, "Pool1", "structural_output")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.item_totals == MetricTotals(0.0, 0.0, 0.0, 0.0)
    assert result.warnings[0].startswith("netstruct_validation")


def test_rejects_boundary_binding_local_layer_id_out_of_range():
    request = _request(
        context_layer_ids=[0],
        reportable_layer_ids=[1],
        latency_offset_source_ids=[0],
        boundary_bindings=[_binding(2, "Pool1", "structural_output")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.item_totals == MetricTotals(0.0, 0.0, 0.0, 0.0)
    assert result.warnings[0].startswith("netstruct_validation")
    assert "boundary_bindings" in result.warnings[0]


def test_rejects_reportable_layer_without_non_hook_boundary_binding():
    request = _request(
        context_layer_ids=[0],
        reportable_layer_ids=[1],
        latency_offset_source_ids=[0],
        boundary_bindings=[_hook_binding("Pool1")],
    )

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.item_totals == MetricTotals(0.0, 0.0, 0.0, 0.0)
    assert result.warnings[0].startswith("netstruct_validation")
    assert "reportable_layer_ids" in result.warnings[0]


def test_rejects_unknown_latency_api():
    request = _request(latency_api="unsupported")

    result = run_mnsim_static_eval(request, backend=FakeBackend())

    assert result.status == "error"
    assert result.item_totals == MetricTotals(0.0, 0.0, 0.0, 0.0)
    assert result.warnings[0].startswith("netstruct_validation")
    assert "latency_api" in result.warnings[0]


def test_original_backend_passes_fragment_layout_to_tcg(tmp_path, monkeypatch):
    captured = {}
    original_tcg = Tile_connection_graph.TCG

    def capturing_tcg(*args, **kwargs):
        captured["active_fragment_layout"] = kwargs.get("active_fragment_layout")
        return original_tcg(*args, **kwargs)

    monkeypatch.setattr(Tile_connection_graph, "TCG", capturing_tcg)

    result = run_mnsim_static_eval(_weighted_conv_request_with_one_fragment(tmp_path))

    assert result.status == "ok"
    assert captured["active_fragment_layout"][0]["active_group_count"] == 1
    assert captured["active_fragment_layout"][0]["active_physical_xbar_count"] == 2
    assert result.layer_results[0].tcg_mapping["active_fragment_ids"] == [
        "L000_conv.g0"
    ]


def test_fragment_static_eval_preserves_original_metrics_and_adds_fragment_audit(tmp_path):
    class FixedFragmentBackend:
        def run(self, request):
            return {
                "layers": [
                    {
                        "local_layer_id": 0,
                        "layer_type": "conv",
                        "begin_time_s": 0.0,
                        "finish_time_s": 4e-9,
                        "area_um2": 7.0,
                        "power_w": 2.25e-3,
                        "energy_j": 9e-12,
                        "energy_breakdown_j": {
                            "xbar": 2e-12,
                            "adc": 3e-12,
                            "buffer_read": 4e-12,
                        },
                        "tcg_mapping": {"tilenum": 1},
                    }
                ],
                "raw_totals": {
                    "raw_latency_total_s": 4e-9,
                    "raw_area_total_um2": 7.0,
                    "raw_power_total_w": 2.25e-3,
                    "raw_energy_total_j": 9e-12,
                },
                "mapping_summary": {"layer_count": 1},
            }

    result = run_mnsim_static_eval(
        _weighted_conv_request_with_one_fragment(tmp_path),
        backend=FixedFragmentBackend(),
    )

    assert result.status == "ok"
    assert result.item_totals == MetricTotals(
        latency_s=4e-9,
        energy_j=9e-12,
        power_w=2.25e-3,
        area_um2=7.0,
    )
    assert result.raw_mnsim_summary["raw_latency_total_s"] == 4e-9
    assert result.raw_mnsim_summary["raw_energy_total_j"] == 9e-12
    assert result.raw_mnsim_summary["reportable_module_energy_j"] == {
        "xbar": 2e-12,
        "adc": 3e-12,
        "buffer_read": 4e-12,
    }
    assert result.raw_mnsim_summary["reportable_module_energy_total_j"] == 9e-12
    assert result.raw_mnsim_summary["active_group_count"] == 1
    assert result.raw_mnsim_summary["active_physical_xbar_count"] == 2
    assert result.raw_mnsim_summary["selected_fragment_ids"] == ["L000_conv.g0"]


def test_original_backend_adds_context_conv_when_first_local_layer_is_not_conv(monkeypatch):
    captured = {}

    class CapturingOriginalBackend:
        def run(self, request):
            captured["request"] = request
            return {
                "layers": [
                    {
                        "local_layer_id": 0,
                        "layer_type": "conv",
                        "begin_time_s": 0.0,
                        "finish_time_s": 2e-9,
                        "area_um2": 10.0,
                        "power_w": 2.0,
                        "energy_j": 10e-9,
                        "tcg_mapping": {"tilenum": 1},
                    },
                    {
                        "local_layer_id": 1,
                        "layer_type": "fc",
                        "begin_time_s": 2e-9,
                        "finish_time_s": 5e-9,
                        "area_um2": 5.0,
                        "power_w": 1.5,
                        "energy_j": 4e-9,
                        "tcg_mapping": {"tilenum": 1},
                    },
                ],
                "raw_totals": {
                    "raw_latency_total_s": 5e-9,
                    "raw_area_total_um2": 15.0,
                    "raw_power_total_w": 3.5,
                    "raw_energy_total_j": 14e-9,
                },
                "mapping_summary": {"layer_count": 2},
            }

    monkeypatch.setattr(static_eval, "_OriginalMnsimBackend", CapturingOriginalBackend)
    request = _request(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[[({"type": "fc"}, None)]],
        multiple=[1],
        context_layer_ids=[],
        reportable_layer_ids=[0],
        latency_offset_source_ids=[],
        boundary_bindings=[_binding(0, "FC1")],
    )

    result = static_eval.run_mnsim_static_eval(request)

    backend_request = captured["request"]
    assert backend_request.netstruct[0][0][0]["type"] == "conv"
    assert backend_request.netstruct[1][0][0]["type"] == "fc"
    assert backend_request.frame.context_layer_ids == [0]
    assert backend_request.frame.reportable_layer_ids == [1]
    assert backend_request.frame.boundary_bindings[0].local_layer_id == 1
    assert result.status == "ok"
    assert result.layer_results[0].report_role == "context_only"
    assert result.layer_results[1].origin_layer_id == "FC1"
    assert result.item_totals == MetricTotals(
        latency_s=3e-9,
        energy_j=4e-9,
        power_w=1.5,
        area_um2=5.0,
    )


def test_original_backend_context_shim_preserves_fragment_layout(monkeypatch):
    captured = {}

    class CapturingOriginalBackend:
        def run(self, request):
            captured["request"] = request
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
                        "tcg_mapping": {},
                    },
                    {
                        "local_layer_id": 1,
                        "layer_type": "fc",
                        "begin_time_s": 1e-9,
                        "finish_time_s": 3e-9,
                        "area_um2": 2.0,
                        "power_w": 1.0,
                        "energy_j": 2e-9,
                        "tcg_mapping": {},
                    },
                ],
                "raw_totals": {},
                "mapping_summary": {},
            }

    fc_layer_info = {
        "type": "fc",
        "Infeature": 1,
        "Outfeature": 1,
        "Inputbit": 8,
        "Weightbit": 2,
        "outputbit": 8,
        "row_split_num": 1,
        "weight_bit_split_part": 1,
    }
    request = StaticEvalRequest(
        item_id="R0",
        item_kind="weighted_layer_segment",
        netstruct=build_netstruct([fc_layer_info]),
        sim_config_path="SimConfig.ini",
        multiple=[1],
        latency_api="nopipe",
        latency_fill_mode=0,
        frame=StaticEvalFrame(
            context_layer_ids=[],
            reportable_layer_ids=[0],
            latency_offset_source_ids=[],
            boundary_bindings=[_binding(0, "L000_fc")],
        ),
        fragment_layout=_fragment_layout(
            layer_id="L000_fc",
            layer_type="fc",
            native_layer_info=fc_layer_info,
        ),
    )
    monkeypatch.setattr(static_eval, "_OriginalMnsimBackend", CapturingOriginalBackend)

    result = run_mnsim_static_eval(request)

    assert result.status == "ok"
    assert captured["request"].fragment_layout is request.fragment_layout


def test_default_backend_receives_supplied_placement_slice(monkeypatch):
    captured = {}
    placement_slice = object()

    class CapturingOriginalBackend:
        def run(self, request, tcg_mapping=None):
            captured["request"] = request
            captured["tcg_mapping"] = tcg_mapping
            return {
                "layers": [
                    {
                        "local_layer_id": 0,
                        "layer_type": "conv",
                        "begin_time_s": 0.0,
                        "finish_time_s": 2e-9,
                        "area_um2": 10.0,
                        "power_w": 2.0,
                        "energy_j": 10e-9,
                        "tcg_mapping": {"tilenum": 1},
                    },
                    {
                        "local_layer_id": 1,
                        "layer_type": "pooling",
                        "begin_time_s": 2e-9,
                        "finish_time_s": 5e-9,
                        "area_um2": 3.0,
                        "power_w": 1.0,
                        "energy_j": 3e-9,
                        "tcg_mapping": {"tilenum": 1},
                    },
                ],
                "raw_totals": {},
                "mapping_summary": {},
            }

    monkeypatch.setattr(static_eval, "_OriginalMnsimBackend", CapturingOriginalBackend)

    result = static_eval.run_mnsim_static_eval(
        _request(),
        placement_slice=placement_slice,
    )

    assert captured["tcg_mapping"] is placement_slice
    assert result.status == "ok"


def test_default_backend_prefixes_fc_slice_with_context_mapping(monkeypatch):
    captured = {}
    prefixed_mapping = object()

    class FakePlacementSlice:
        def __init__(self):
            self.prefix_netstructs = []

        def with_context_prefix(self, netstruct):
            self.prefix_netstructs.append(netstruct)
            return prefixed_mapping

    class CapturingOriginalBackend:
        def run(self, request, tcg_mapping=None):
            captured["request"] = request
            captured["tcg_mapping"] = tcg_mapping
            return {
                "layers": [
                    {
                        "local_layer_id": 0,
                        "layer_type": "conv",
                        "begin_time_s": 0.0,
                        "finish_time_s": 2e-9,
                        "area_um2": 10.0,
                        "power_w": 2.0,
                        "energy_j": 10e-9,
                        "tcg_mapping": {"tilenum": 1},
                    },
                    {
                        "local_layer_id": 1,
                        "layer_type": "fc",
                        "begin_time_s": 2e-9,
                        "finish_time_s": 5e-9,
                        "area_um2": 5.0,
                        "power_w": 1.5,
                        "energy_j": 4e-9,
                        "tcg_mapping": {"tilenum": 1},
                    },
                ],
                "raw_totals": {},
                "mapping_summary": {},
            }

    placement_slice = FakePlacementSlice()
    monkeypatch.setattr(static_eval, "_OriginalMnsimBackend", CapturingOriginalBackend)
    request = _request(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[[({"type": "fc"}, None)]],
        multiple=[1],
        context_layer_ids=[],
        reportable_layer_ids=[0],
        latency_offset_source_ids=[],
        boundary_bindings=[_binding(0, "FC1")],
    )

    result = static_eval.run_mnsim_static_eval(
        request,
        placement_slice=placement_slice,
    )

    assert len(placement_slice.prefix_netstructs) == 1
    assert [layer[0][0]["type"] for layer in placement_slice.prefix_netstructs[0]] == [
        "conv",
        "fc",
    ]
    assert captured["tcg_mapping"] is prefixed_mapping
    assert result.item_totals == MetricTotals(
        latency_s=3e-9,
        energy_j=4e-9,
        power_w=1.5,
        area_um2=5.0,
    )


def test_real_asymmetric_fc_slice_uses_a_nonresident_context_that_fits(tmp_path):
    sim_config_path = _two_tile_config_path(tmp_path)
    layer_info = {
        "type": "fc",
        "Infeature": 256,
        "Outfeature": 1,
        "Inputbit": 8,
        "Weightbit": 8,
        "outputbit": 8,
        "row_split_num": 1,
        "weight_bit_split_part": 1,
    }
    netstruct = build_netstruct([layer_info])
    placement = prepare_round_placement(
        item_id="R0",
        sim_config_path=sim_config_path,
        weighted_scopes=[
            WeightedScope(
                "R0.S0",
                0,
                layer_info,
                fragment_layout=_fragment_layout(
                    layer_id="FC0",
                    layer_type="fc",
                    native_layer_info=layer_info,
                    active_group_count=2,
                    active_physical_xbar_count=4,
                ),
            )
        ],
        expected_used_xbars=4,
    )
    placement_slice = placement.slice_for("R0.S0", 0, netstruct)
    request = StaticEvalRequest(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=netstruct,
        sim_config_path=sim_config_path,
        multiple=[1],
        latency_api="nopipe",
        latency_fill_mode=0,
        frame=StaticEvalFrame(
            context_layer_ids=[],
            reportable_layer_ids=[0],
            latency_offset_source_ids=[],
            boundary_bindings=[_binding(0, "FC0")],
        ),
    )

    result = run_mnsim_static_eval(request, placement_slice=placement_slice)

    assert result.status == "ok"
    assert [layer.report_role for layer in result.layer_results] == [
        "context_only",
        "reportable",
    ]
    assert result.item_totals.area_um2 > 0.0
    assert result.mapping_summary["active_group_count"] == 2
    assert result.mapping_summary["active_physical_xbar_count"] == 4
    assert result.mapping_summary["mnsim_group_capacity"] == 16
    assert result.mapping_summary["physical_xbar_capacity"] == 32
    assert result.mapping_summary["active_fragment_ids"] == ["FC0.g0", "FC0.g1"]


def test_allows_injected_backend_when_first_local_layer_is_not_conv():
    request = _request(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[[({"type": "fc"}, None)]],
        multiple=[1],
        context_layer_ids=[],
        reportable_layer_ids=[0],
        latency_offset_source_ids=[],
        boundary_bindings=[_binding(0, "FC1")],
    )

    result = run_mnsim_static_eval(request, backend=SingleLayerBackend())

    assert result.status == "ok"
    assert result.item_totals == MetricTotals(
        latency_s=3e-9,
        energy_j=4e-9,
        power_w=1.5,
        area_um2=5.0,
    )


def test_item_totals_follow_normalized_report_roles_not_frame_ids():
    request = _request(
        item_id="R0",
        item_kind="weighted_full_pack",
        netstruct=[
            [({"type": "conv"}, None)],
            [({"type": "pooling"}, None)],
            [({"type": "pooling"}, None)],
        ],
        multiple=[1, 1, 1],
        context_layer_ids=[0],
        reportable_layer_ids=[1],
        latency_offset_source_ids=[0],
        boundary_bindings=[_binding(1, "Pool1")],
    )

    result = run_mnsim_static_eval(request, backend=ExtraReportableLayerBackend())

    assert result.status == "ok"
    assert result.layer_results[2].report_role == "reportable"
    assert result.item_totals.latency_s == pytest.approx(6e-9)
    assert result.item_totals.energy_j == pytest.approx(6e-9)
    assert result.item_totals.power_w == pytest.approx(1.5)
    assert result.item_totals.area_um2 == pytest.approx(7.0)


def test_phase2_package_exports_static_eval_runner():
    assert phase2.run_mnsim_static_eval is run_mnsim_static_eval
    assert "run_mnsim_static_eval" in phase2.__all__
