"""Static Phase 2 wrapper around original MNSIM model evaluation."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from MNSIM.Scheduling.phase2.types import (
    CONTRIBUTION_STRUCTURAL_OUTPUT,
    REPORT_ROLE_CONTEXT_ONLY,
    REPORT_ROLE_REPORTABLE,
    MetricTotals,
    MnsimFragmentLayout,
    StaticEvalFrame,
    StaticEvalRequest,
    StaticEvalResult,
    StaticLayerResult,
)


_NS_TO_S = 1e-9
_NJ_TO_J = 1e-9
_DIRECT_ZERO_XBAR_STRUCTURAL_TYPES = {"concat", "element_sum", "element_multiply"}


def run_mnsim_static_eval(
    request: StaticEvalRequest,
    backend=None,
    placement_slice=None,
) -> StaticEvalResult:
    """Run static MNSIM evaluation and normalize it to the Phase 2 schema."""
    if _uses_direct_zero_xbar_structural_eval(request):
        validation_error = _validate_request(
            request,
            use_original_backend=False,
            direct_zero_xbar_structural=True,
        )
        if validation_error is not None:
            return _error_result(request, validation_error)
        return _direct_zero_xbar_structural_eval(request)

    if backend is None:
        effective_request, effective_placement_slice = _with_original_backend_context_shim(
            request,
            placement_slice,
        )
    else:
        effective_request = request
        effective_placement_slice = None
    validation_error = _validate_request(
        effective_request,
        use_original_backend=backend is None,
    )
    if validation_error is not None:
        return _error_result(request, validation_error)

    if backend is not None:
        raw_result = backend.run(effective_request)
    else:
        active_backend = _OriginalMnsimBackend()
        if effective_placement_slice is None:
            raw_result = active_backend.run(effective_request)
        else:
            raw_result = active_backend.run(
                effective_request,
                tcg_mapping=effective_placement_slice,
            )
    return _normalize_backend_result(effective_request, raw_result)


class _OriginalMnsimBackend:
    def run(
        self,
        request: StaticEvalRequest,
        tcg_mapping=None,
    ) -> dict[str, Any]:
        from MNSIM.Area_Model.Model_Area import Model_area
        from MNSIM.Energy_Model.Model_energy import Model_energy
        from MNSIM.Latency_Model.Model_latency import Model_latency
        from MNSIM.Mapping_Model.Tile_connection_graph import TCG
        from MNSIM.Power_Model.Model_inference_power import Model_inference_power

        tcg = (
            tcg_mapping
            if tcg_mapping is not None
            else TCG(
                request.netstruct,
                request.sim_config_path,
                multiple=request.multiple,
                active_fragment_layout=_tcg_layout_for_request(request),
            )
        )
        latency = Model_latency(
            request.netstruct,
            request.sim_config_path,
            multiple=request.multiple,
            TCG_mapping=tcg,
        )
        if request.latency_api == "nopipe":
            latency.calculate_model_latency_nopipe()
        else:
            latency.calculate_model_latency(mode=request.latency_fill_mode)

        area = Model_area(
            request.netstruct,
            request.sim_config_path,
            multiple=request.multiple,
            TCG_mapping=tcg,
        )
        power = Model_inference_power(
            request.netstruct,
            request.sim_config_path,
            multiple=request.multiple,
            TCG_mapping=tcg,
        )
        energy = Model_energy(
            request.netstruct,
            request.sim_config_path,
            model_latency=latency,
            model_power=power,
            multiple=request.multiple,
            TCG_mapping=tcg,
        )

        layers = []
        for local_layer_id in range(len(request.netstruct)):
            begin_time_ns = _indexed_min(getattr(latency, "begin_time", None), local_layer_id)
            finish_time_ns = _indexed_max(getattr(latency, "finish_time", None), local_layer_id)
            if finish_time_ns < begin_time_ns:
                finish_time_ns = begin_time_ns
            layers.append(
                {
                    "local_layer_id": local_layer_id,
                    "layer_type": _layer_type(request.netstruct, local_layer_id),
                    "begin_time_s": begin_time_ns * _NS_TO_S,
                    "finish_time_s": finish_time_ns * _NS_TO_S,
                    "area_um2": _indexed_number(getattr(area, "arch_area", None), local_layer_id),
                    "power_w": _indexed_number(getattr(power, "arch_power", None), local_layer_id),
                    "energy_j": _indexed_number(getattr(energy, "arch_energy", None), local_layer_id)
                    * _NJ_TO_J,
                    "adc_latency_s": _indexed_number(
                        getattr(latency, "total_ADC_latency", None), local_layer_id
                    ) * _NS_TO_S,
                    "adc_area_um2": _indexed_number(
                        getattr(area, "arch_ADC_area", None), local_layer_id
                    ),
                    "adc_power_w": _indexed_number(
                        getattr(power, "arch_ADC_power", None), local_layer_id
                    ),
                    "adc_energy_j": _indexed_number(
                        getattr(energy, "arch_ADC_energy", None), local_layer_id
                    ) * _NJ_TO_J,
                    "energy_breakdown_j": _layer_energy_breakdown_j(
                        energy, local_layer_id
                    ),
                    "tcg_mapping": _indexed_mapping(getattr(tcg, "layer_tileinfo", None), local_layer_id),
                }
            )

        return {
            "layers": layers,
            "raw_totals": {
                "raw_latency_total_s": _nested_max(getattr(latency, "finish_time", None)) * _NS_TO_S,
                "raw_area_total_um2": _number(getattr(area, "arch_total_area", 0.0)),
                "raw_power_total_w": _number(getattr(power, "arch_total_power", 0.0)),
                "raw_energy_total_j": _number(getattr(energy, "arch_total_energy", 0.0)) * _NJ_TO_J,
                # Preserve the complete original MNSIM ADC ledger.  Analog
                # accounting removes these values only after this raw result
                # has been normalized and recorded in the audit.
                "raw_adc_latency_s": _number(
                    sum(getattr(latency, "total_ADC_latency", []) or [])
                ) * _NS_TO_S,
                "raw_adc_area_um2": _number(
                    getattr(area, "arch_total_ADC_area", 0.0)
                ),
                "raw_adc_power_w": _number(
                    getattr(power, "arch_total_ADC_power", 0.0)
                ),
                "raw_adc_energy_j": _number(
                    getattr(energy, "arch_total_ADC_energy", 0.0)
                ) * _NJ_TO_J,
            },
            "mapping_summary": {
                **_tcg_mapping_summary(tcg, len(request.netstruct)),
            },
        }


def _tcg_layout_for_request(request: StaticEvalRequest) -> dict[int, dict[str, Any]] | None:
    """Translate one item-local fragment layout into TCG's opt-in payload."""
    layout = request.fragment_layout
    if layout is None:
        return None
    mapping_layer_ids = [
        local_layer_id
        for local_layer_id in range(len(request.netstruct))
        if _layer_type(request.netstruct, local_layer_id) in {"conv", "fc"}
    ]
    reportable_mapping_ids = [
        local_layer_id
        for local_layer_id in request.frame.reportable_layer_ids
        if local_layer_id in mapping_layer_ids
    ]
    if len(reportable_mapping_ids) != 1:
        raise ValueError(
            "fragment layout requires exactly one reportable Conv/FC local layer"
        )
    target_layer_id = reportable_mapping_ids[0]
    target_layer_type = _layer_type(request.netstruct, target_layer_id)
    layout_layer_type = "fc" if layout.layer_type == "fc_like_projection" else layout.layer_type
    if layout_layer_type != target_layer_type:
        raise ValueError(
            f"fragment layout type {layout.layer_type!r} does not match local "
            f"mapping type {target_layer_type!r}"
        )
    if not layout.fragments:
        raise ValueError("fragment layout must contain at least one active fragment")

    polarity_count = int(layout.fragments[0].polarity_count)
    target_payload = {
        "fragment_ids": [fragment.fragment_id for fragment in layout.fragments],
        "active_group_count": int(layout.active_group_count),
        "active_physical_xbar_count": int(layout.active_physical_xbar_count),
        "polarity_count": polarity_count,
    }
    inactive_payload = {
        "fragment_ids": [],
        "active_group_count": 0,
        "active_physical_xbar_count": 0,
        "polarity_count": polarity_count,
    }
    return {
        local_layer_id: (
            target_payload if local_layer_id == target_layer_id else dict(inactive_payload)
        )
        for local_layer_id in mapping_layer_ids
    }


def _fragment_audit(layout: MnsimFragmentLayout | None) -> dict[str, Any]:
    """Serialize the selected physical fragment proof without changing metrics."""
    if layout is None:
        return {}
    return {
        "active_group_count": int(layout.active_group_count),
        "active_physical_xbar_count": int(layout.active_physical_xbar_count),
        "selected_fragment_ids": [
            fragment.fragment_id for fragment in layout.fragments
        ],
    }


def _tcg_mapping_summary(tcg: Any, default_layer_count: int) -> dict[str, Any]:
    audit_source = _tcg_audit_source(tcg)
    summary = {
        "layer_count": int(_number(getattr(tcg, "layer_num", default_layer_count))),
        "tcg_buffer_capacity_evidence": deepcopy(
            list(getattr(audit_source, "buffer_capacity_evidence", []) or [])
        ),
        "tcg_pe_input_rounded_capacity_bits_per_instance": int(
            math.ceil(float(getattr(audit_source, "max_inbuf_size", 0.0)) * 1024 * 8)
        ),
        "tcg_tile_output_rounded_capacity_bits_per_instance": int(
            math.ceil(float(getattr(audit_source, "max_outbuf_size", 0.0)) * 1024 * 8)
        ),
    }
    tileinfo = list(getattr(tcg, "layer_tileinfo", []) or [])
    active_tileinfo = [
        info for info in tileinfo if isinstance(info, Mapping) and "active_group_count" in info
    ]
    if not active_tileinfo:
        return summary

    summary.update(
        {
            "active_group_count": sum(
                int(info["active_group_count"]) for info in active_tileinfo
            ),
            "active_physical_xbar_count": sum(
                int(info["active_physical_xbar_count"]) for info in active_tileinfo
            ),
            "active_fragment_ids": [
                fragment_id
                for info in active_tileinfo
                for fragment_id in info.get("active_fragment_ids", [])
            ],
            "mnsim_group_capacity": int(
                getattr(audit_source, "mnsim_group_capacity", 0)
            ),
            "physical_xbar_capacity": int(
                getattr(audit_source, "physical_xbar_capacity", 0)
            ),
        }
    )
    return summary


def _tcg_audit_source(tcg: Any) -> Any:
    placement = getattr(tcg, "placement", None)
    if placement is not None:
        resident_tcg = getattr(placement, "tcg", None)
        if resident_tcg is not None:
            return resident_tcg
    return tcg


def _normalize_backend_result(
    request: StaticEvalRequest,
    raw_result: Mapping[str, Any],
) -> StaticEvalResult:
    context_layer_ids = set(request.frame.context_layer_ids)
    raw_layers = list(raw_result.get("layers", []))
    layer_by_id = {_layer_id(raw_layer, index): raw_layer for index, raw_layer in enumerate(raw_layers)}
    context_ready_time = max(
        (
            _number(layer_by_id.get(local_layer_id, {}).get("finish_time_s", 0.0))
            for local_layer_id in request.frame.latency_offset_source_ids
        ),
        default=0.0,
    )

    layer_results = []
    for index, raw_layer in enumerate(raw_layers):
        local_layer_id = _layer_id(raw_layer, index)
        begin_time_s = _number(raw_layer.get("begin_time_s", 0.0))
        finish_time_s = _number(raw_layer.get("finish_time_s", begin_time_s))
        report_role = (
            REPORT_ROLE_CONTEXT_ONLY
            if local_layer_id in context_layer_ids
            else REPORT_ROLE_REPORTABLE
        )
        if report_role == REPORT_ROLE_REPORTABLE:
            begin_time_s = _seconds(begin_time_s - context_ready_time)
            finish_time_s = _seconds(finish_time_s - context_ready_time)
        latency_s = max(0.0, _seconds(finish_time_s - begin_time_s))
        layer_results.append(
            StaticLayerResult(
                local_layer_id=local_layer_id,
                origin_layer_id=_origin_layer_id(request, local_layer_id),
                layer_type=str(raw_layer.get("layer_type", _layer_type(request.netstruct, local_layer_id))),
                report_role=report_role,
                latency_s=latency_s,
                begin_time_s=begin_time_s,
                finish_time_s=finish_time_s,
                area_um2=_number(raw_layer.get("area_um2", 0.0)),
                power_w=_number(raw_layer.get("power_w", 0.0)),
                energy_j=_number(raw_layer.get("energy_j", 0.0)),
                tcg_mapping=dict(raw_layer.get("tcg_mapping", {}) or {}),
            )
        )

    reportable_layers = [
        layer for layer in layer_results if layer.report_role == REPORT_ROLE_REPORTABLE
    ]
    item_totals = MetricTotals(
        latency_s=_seconds(sum(layer.latency_s for layer in reportable_layers)),
        energy_j=sum(layer.energy_j for layer in reportable_layers),
        power_w=sum(layer.power_w for layer in reportable_layers),
        area_um2=sum(layer.area_um2 for layer in reportable_layers),
    )

    raw_mnsim_summary = dict(raw_result.get("raw_totals", {}) or {})
    reportable_raw_layers = [
        raw_layer
        for index, raw_layer in enumerate(raw_layers)
        if _layer_id(raw_layer, index) not in context_layer_ids
    ]
    for field in (
        "adc_latency_s",
        "adc_energy_j",
        "adc_power_w",
        "adc_area_um2",
    ):
        if any(field in raw_layer for raw_layer in reportable_raw_layers):
            raw_mnsim_summary[f"reportable_{field}"] = sum(
                _number(raw_layer.get(field, 0.0))
                for raw_layer in reportable_raw_layers
            )
    reportable_energy_breakdown = _sum_reportable_energy_breakdown(
        raw_layers,
        context_layer_ids,
    )
    if reportable_energy_breakdown:
        raw_mnsim_summary["reportable_module_energy_j"] = (
            reportable_energy_breakdown
        )
        raw_mnsim_summary["reportable_module_energy_total_j"] = sum(
            reportable_energy_breakdown.values()
        )
    raw_mnsim_summary.update(_fragment_audit(request.fragment_layout))
    return StaticEvalResult(
        item_id=request.item_id,
        item_kind=request.item_kind,
        status="ok",
        layer_results=layer_results,
        item_totals=item_totals,
        mapping_summary=dict(raw_result.get("mapping_summary", {}) or {}),
        raw_mnsim_summary=raw_mnsim_summary,
        boundary_bindings=request.frame.boundary_bindings,
        warnings=[],
    )


def _layer_energy_breakdown_j(energy: Any, local_layer_id: int) -> dict[str, float]:
    """Expose original MNSIM's mutually exclusive per-layer energy ledgers."""

    sources = {
        "xbar": "arch_xbar_energy",
        "adc": "arch_ADC_energy",
        "dac": "arch_DAC_energy",
        "adder": "arch_adder_energy",
        "shift_register": "arch_shiftreg_energy",
        "input_register": "arch_iReg_energy",
        "output_register": "arch_oReg_energy",
        "input_demux": "arch_input_demux_energy",
        "output_mux": "arch_output_mux_energy",
        "joint_module": "arch_jointmodule_energy",
        "buffer_read": "arch_buf_r_energy",
        "buffer_write": "arch_buf_w_energy",
        "pooling": "arch_pooling_energy",
    }
    return {
        name: _indexed_number(getattr(energy, attribute, None), local_layer_id)
        * _NJ_TO_J
        for name, attribute in sources.items()
    }


def _sum_reportable_energy_breakdown(
    raw_layers: Sequence[Mapping[str, Any]],
    context_layer_ids: set[int],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for index, raw_layer in enumerate(raw_layers):
        if _layer_id(raw_layer, index) in context_layer_ids:
            continue
        breakdown = raw_layer.get("energy_breakdown_j")
        if not isinstance(breakdown, Mapping):
            continue
        for name, value in breakdown.items():
            totals[str(name)] = totals.get(str(name), 0.0) + _number(value)
    return totals


def _validate_request(
    request: StaticEvalRequest,
    *,
    use_original_backend: bool = False,
    direct_zero_xbar_structural: bool = False,
) -> str | None:
    layer_count = len(request.netstruct)
    if len(request.multiple) != layer_count:
        return f"multiple length {len(request.multiple)} does not match netstruct length {layer_count}"

    if request.latency_api not in {"pipe", "nopipe"}:
        return f"latency_api must be one of ['nopipe', 'pipe']: {request.latency_api!r}"

    if use_original_backend and layer_count > 0:
        first_layer_type = _layer_type(request.netstruct, 0)
        if first_layer_type != "conv":
            return (
                "original MNSIM backend requires the first local layer to be conv; "
                f"first local layer is {first_layer_type!r}"
            )

    for name, ids in (
        ("context_layer_ids", request.frame.context_layer_ids),
        ("reportable_layer_ids", request.frame.reportable_layer_ids),
        ("latency_offset_source_ids", request.frame.latency_offset_source_ids),
    ):
        invalid_ids = _invalid_ids(ids, layer_count)
        if invalid_ids:
            return f"{name} out of range for {layer_count} layers: {invalid_ids}"

    overlap_ids = sorted(set(request.frame.context_layer_ids) & set(request.frame.reportable_layer_ids))
    if overlap_ids:
        return f"context_layer_ids overlap reportable_layer_ids: {overlap_ids}"

    non_context_offset_ids = sorted(
        set(request.frame.latency_offset_source_ids) - set(request.frame.context_layer_ids)
    )
    if non_context_offset_ids:
        return (
            "latency_offset_source_ids must reference context_layer_ids: "
            f"{non_context_offset_ids}"
        )

    binding_layer_ids = [
        binding.local_layer_id
        for binding in request.frame.boundary_bindings
        if binding.local_layer_id is not None
    ]
    invalid_binding_ids = _invalid_ids(binding_layer_ids, layer_count)
    if invalid_binding_ids:
        return f"boundary_bindings local_layer_id out of range for {layer_count} layers: {invalid_binding_ids}"

    bound_static_layer_ids = {
        binding.local_layer_id
        for binding in request.frame.boundary_bindings
        if binding.local_layer_id is not None and binding.hook_node_id is None
    }
    missing_reportable_bindings = sorted(
        set(request.frame.reportable_layer_ids) - bound_static_layer_ids
    )
    if missing_reportable_bindings:
        return (
            "reportable_layer_ids require matching non-hook boundary_bindings "
            f"local_layer_id: {missing_reportable_bindings}"
        )

    if request.item_kind == "zero_xbar_structural" and not request.frame.reportable_layer_ids:
        return "zero_xbar_structural requires at least one reportable layer"

    has_structural_reportable = any(
        binding.contribution_kind == CONTRIBUTION_STRUCTURAL_OUTPUT
        and binding.local_layer_id in request.frame.reportable_layer_ids
        for binding in request.frame.boundary_bindings
    )
    if (
        request.item_kind == "zero_xbar_structural"
        and has_structural_reportable
        and not request.frame.context_layer_ids
        and not direct_zero_xbar_structural
        and not _reportable_layers_can_run_standalone(request)
    ):
        return "standalone structural reportable layer requires context ids"

    if direct_zero_xbar_structural:
        direct_error = _validate_direct_zero_xbar_structural_request(request)
        if direct_error is not None:
            return direct_error

    return None


def _uses_direct_zero_xbar_structural_eval(request: StaticEvalRequest) -> bool:
    if request.item_kind != "zero_xbar_structural":
        return False
    layer_count = len(request.netstruct)
    reportable_ids = list(request.frame.reportable_layer_ids)
    if not reportable_ids or _invalid_ids(reportable_ids, layer_count):
        return False
    reportable_types = {_layer_type(request.netstruct, local_layer_id) for local_layer_id in reportable_ids}
    return reportable_types.issubset(_DIRECT_ZERO_XBAR_STRUCTURAL_TYPES)


def _validate_direct_zero_xbar_structural_request(request: StaticEvalRequest) -> str | None:
    for local_layer_id in request.frame.reportable_layer_ids:
        layer_dict = _layer_dict(request.netstruct[local_layer_id])
        layer_type = str(layer_dict.get("type", ""))
        if layer_type not in _DIRECT_ZERO_XBAR_STRUCTURAL_TYPES:
            return f"direct zero-xbar structural eval does not support {layer_type!r}"
        if not _direct_structural_required_fields(layer_dict):
            return f"direct zero-xbar structural layer {local_layer_id} is missing required shape fields"
        input_count = _direct_input_count(request, local_layer_id, layer_dict)
        if input_count <= 1:
            return f"{layer_type} direct eval requires at least two inputs"
    return None


def _direct_structural_required_fields(layer_dict: Mapping[str, Any]) -> bool:
    required_fields = {
        "Inputsize",
        "Outputsize",
        "Inputchannel",
        "Outputchannel",
        "Inputbit",
        "outputbit",
    }
    return required_fields.issubset(layer_dict)


def _direct_zero_xbar_structural_eval(request: StaticEvalRequest) -> StaticEvalResult:
    context_layer_ids = set(request.frame.context_layer_ids)
    reportable_layer_ids = set(request.frame.reportable_layer_ids)
    layer_results: list[StaticLayerResult] = []

    for local_layer_id in range(len(request.netstruct)):
        layer_dict = _layer_dict(request.netstruct[local_layer_id])
        layer_type = str(layer_dict.get("type", "unknown"))
        if local_layer_id in reportable_layer_ids:
            metrics = _direct_accumulator_metrics(request, local_layer_id, layer_dict)
            layer_results.append(
                StaticLayerResult(
                    local_layer_id=local_layer_id,
                    origin_layer_id=_origin_layer_id(request, local_layer_id),
                    layer_type=layer_type,
                    report_role=REPORT_ROLE_REPORTABLE,
                    latency_s=metrics["latency_s"],
                    begin_time_s=0.0,
                    finish_time_s=metrics["latency_s"],
                    area_um2=metrics["area_um2"],
                    power_w=metrics["power_w"],
                    energy_j=metrics["energy_j"],
                    tcg_mapping=metrics["mapping"],
                )
            )
            continue
        if local_layer_id in context_layer_ids:
            layer_results.append(
                StaticLayerResult(
                    local_layer_id=local_layer_id,
                    origin_layer_id=f"context:{local_layer_id}",
                    layer_type=layer_type,
                    report_role=REPORT_ROLE_CONTEXT_ONLY,
                    latency_s=0.0,
                    begin_time_s=0.0,
                    finish_time_s=0.0,
                    area_um2=0.0,
                    power_w=0.0,
                    energy_j=0.0,
                    tcg_mapping={"mode": "direct_zero_xbar_context"},
                )
            )

    reportable_layers = [
        layer for layer in layer_results if layer.report_role == REPORT_ROLE_REPORTABLE
    ]
    totals = MetricTotals(
        latency_s=sum(layer.latency_s for layer in reportable_layers),
        energy_j=sum(layer.energy_j for layer in reportable_layers),
        power_w=sum(layer.power_w for layer in reportable_layers),
        area_um2=sum(layer.area_um2 for layer in reportable_layers),
    )
    return StaticEvalResult(
        item_id=request.item_id,
        item_kind=request.item_kind,
        status="ok",
        layer_results=layer_results,
        item_totals=totals,
        mapping_summary={
            "layer_count": len(request.netstruct),
            "backend": "direct_zero_xbar_structural",
        },
        raw_mnsim_summary={
            "raw_latency_total_s": totals.latency_s,
            "raw_area_total_um2": totals.area_um2,
            "raw_power_total_w": totals.power_w,
            "raw_energy_total_j": totals.energy_j,
        },
        boundary_bindings=request.frame.boundary_bindings,
        warnings=[],
    )


def _direct_accumulator_metrics(
    request: StaticEvalRequest,
    local_layer_id: int,
    layer_dict: Mapping[str, Any],
) -> dict[str, Any]:
    from MNSIM.Hardware_Model.Adder import adder
    from MNSIM.Hardware_Model.Buffer import buffer
    from MNSIM.Hardware_Model.Multiplier import multiplier

    layer_type = str(layer_dict.get("type", ""))
    input_count = _direct_input_count(request, local_layer_id, layer_dict)
    outputchannel = max(1, int(layer_dict.get("Outputchannel", layer_dict.get("Inputchannel", 1))))
    outputbit = max(1, int(layer_dict.get("outputbit", layer_dict.get("Inputbit", 8))))
    if layer_type == "concat":
        output_size = [int(value) for value in layer_dict.get("Outputsize", [1, 1])]
        data_size = max(1.0, outputchannel * math.prod(output_size) * outputbit / 8)
    else:
        data_size = max(1.0, outputchannel * outputbit * input_count / 8)
    global_buf_size = _next_power_of_two_kb(data_size)
    global_buf = buffer(SimConfig_path=request.sim_config_path, buf_level=2, default_buf_size=global_buf_size)
    global_buf.calculate_buf_read_latency(rdata=data_size)
    global_buf.calculate_buf_write_latency(wdata=data_size)
    global_buf.calculate_buf_read_power()
    global_buf.calculate_buf_write_power()
    global_buf.calculate_buf_area()

    unit_count = max(1, outputchannel * input_count // 2)
    if layer_type == "concat":
        # Concatenation moves the input channel segments into one output
        # buffer; it has no arithmetic unit.  The buffer traffic remains part
        # of the structural performance cost.
        digital_area = 0.0
        digital_power = 0.0
        unit_count = 0
        unit_name = "concat_channel_router_num"
    elif layer_type == "element_multiply":
        digital = multiplier(SimConfig_path=request.sim_config_path, bitwidth=outputbit)
        digital.calculate_multiplier_area()
        digital.calculate_multiplier_power()
        digital_area = digital.multiplier_area * unit_count
        digital_power = digital.multiplier_power * unit_count
        unit_name = "global_multiplier_num"
    else:
        digital = adder(SimConfig_path=request.sim_config_path, bitwidth=outputbit)
        digital.calculate_adder_area()
        digital.calculate_adder_power()
        digital_area = digital.adder_area * unit_count
        digital_power = digital.adder_power * unit_count
        unit_name = "global_adder_num"

    latency_ns = 10.0 + global_buf.buf_rlatency + global_buf.buf_wlatency
    latency_s = latency_ns * _NS_TO_S
    power_w = digital_power + (global_buf.buf_rpower + global_buf.buf_wpower) * 1e-3
    area_um2 = digital_area + global_buf.buf_area
    energy_j = power_w * latency_s
    return {
        "latency_s": latency_s,
        "area_um2": area_um2,
        "power_w": power_w,
        "energy_j": energy_j,
        "mapping": {
            "mode": "direct_zero_xbar_structural",
            unit_name: unit_count,
            "input_branch_count": input_count,
            "global_buf_size_kb": global_buf_size,
        },
    }


def _direct_input_count(
    request: StaticEvalRequest,
    local_layer_id: int,
    layer_dict: Mapping[str, Any],
) -> int:
    input_indices = layer_dict.get("Inputindex", [])
    try:
        input_count = len(list(input_indices))
    except TypeError:
        input_count = 0
    if input_count > 0:
        return input_count
    for binding in request.frame.boundary_bindings:
        if binding.local_layer_id == local_layer_id:
            return len(binding.source_state_ids)
    return 0


def _next_power_of_two_kb(byte_count: float) -> float:
    safe_bytes = max(1.0, float(byte_count))
    return (2 ** math.ceil(math.log(safe_bytes, 2))) / 1024


def _error_result(request: StaticEvalRequest, detail: str) -> StaticEvalResult:
    return StaticEvalResult(
        item_id=request.item_id,
        item_kind=request.item_kind,
        status="error",
        layer_results=[],
        item_totals=MetricTotals(0.0, 0.0, 0.0, 0.0),
        mapping_summary={},
        raw_mnsim_summary={},
        boundary_bindings=request.frame.boundary_bindings,
        warnings=[f"netstruct_validation: {detail}"],
    )


def _origin_layer_id(request: StaticEvalRequest, local_layer_id: int) -> str:
    for binding in request.frame.boundary_bindings:
        if binding.local_layer_id == local_layer_id:
            return binding.origin_layer_id
    return f"local_layer:{local_layer_id}"


def _with_original_backend_context_shim(
    request: StaticEvalRequest,
    placement_slice,
) -> tuple[StaticEvalRequest, Any]:
    if not request.netstruct or _layer_type(request.netstruct, 0) == "conv":
        return request, placement_slice

    first_layer_dict = _layer_dict(request.netstruct[0])
    shim_layer = _context_conv_shim(first_layer_dict, len(request.netstruct))
    shimmed_netstruct = [[(shim_layer, None)]]
    for index, layer_entry in enumerate(request.netstruct, start=1):
        cloned_entry = deepcopy(layer_entry)
        layer_dict = _mutable_layer_dict(cloned_entry)
        if layer_dict is not None:
            layer_dict["Layerindex"] = index
            if index == 1:
                if "Inputindex" not in layer_dict:
                    layer_dict["Inputindex"] = [-1]
                else:
                    layer_dict["Inputindex"] = list(map(int, layer_dict["Inputindex"])) or [-1]
                if "Outputindex" not in layer_dict:
                    layer_dict["Outputindex"] = [1]
            elif "Layerindex" not in layer_dict:
                layer_dict["Layerindex"] = index
        shimmed_netstruct.append(cloned_entry)

    shimmed_request = StaticEvalRequest(
        item_id=request.item_id,
        item_kind=request.item_kind,
        netstruct=shimmed_netstruct,
        sim_config_path=request.sim_config_path,
        multiple=[1, *list(request.multiple)],
        latency_api=request.latency_api,
        latency_fill_mode=request.latency_fill_mode,
        frame=StaticEvalFrame(
            context_layer_ids=_shift_ids(request.frame.context_layer_ids, 1, include_zero=True),
            reportable_layer_ids=_shift_ids(request.frame.reportable_layer_ids, 1),
            latency_offset_source_ids=_shift_ids(
                request.frame.latency_offset_source_ids,
                1,
                include_zero=True,
            ),
            boundary_bindings=_shift_boundary_bindings(request.frame.boundary_bindings, 1),
        ),
        fragment_layout=request.fragment_layout,
    )
    shimmed_placement_slice = (
        placement_slice.with_context_prefix(shimmed_request.netstruct)
        if placement_slice is not None
        else None
    )
    return shimmed_request, shimmed_placement_slice


def _can_run_as_original_backend_first_layer(layer_dict: Mapping[str, Any]) -> bool:
    layer_type = str(layer_dict.get("type", ""))
    if layer_type == "conv":
        return True
    if layer_type == "fc":
        required_fields = {
            "Inputsize",
            "Outputsize",
            "Inputchannel",
            "Outputchannel",
            "Kernelsize",
            "Stride",
            "Padding",
            "Infeature",
            "Outfeature",
        }
        return required_fields.issubset(layer_dict)
    if layer_type == "pooling":
        required_fields = {
            "Inputsize",
            "Outputsize",
            "Inputchannel",
            "Outputchannel",
            "Kernelsize",
            "Stride",
            "Padding",
        }
        return required_fields.issubset(layer_dict)
    return False


def _reportable_layers_can_run_standalone(request: StaticEvalRequest) -> bool:
    reportable_ids = list(request.frame.reportable_layer_ids)
    if not reportable_ids:
        return False
    for local_layer_id in reportable_ids:
        if not _can_run_as_original_backend_first_layer(_layer_dict(request.netstruct[local_layer_id])):
            return False
    return True


def _context_conv_shim(first_layer_dict: Mapping[str, Any], original_layer_count: int) -> OrderedDict[str, Any]:
    layer_type = str(first_layer_dict.get("type", "fc"))
    inputbit = int(first_layer_dict.get("Inputbit", first_layer_dict.get("outputbit", 8)))
    weightbit = int(first_layer_dict.get("Weightbit", 8))
    outputbit = int(first_layer_dict.get("outputbit", first_layer_dict.get("Outputbit", inputbit)))

    if layer_type == "fc":
        # FC only reads the predecessor completion time, so its context must not reserve FC-sized weights.
        channel_count = 1
        input_size = [1, 1]
        output_size = [1, 1]
        weightbit = 2
    elif layer_type == "pooling":
        # MNSIM indexes the predecessor completion map with pooling Inputsize.
        # Keep the synthetic Conv's output map at that resolution.
        channel_count = int(first_layer_dict.get("Inputchannel", first_layer_dict.get("Outputchannel", 1)))
        input_size = list(first_layer_dict.get("Inputsize", first_layer_dict.get("Outputsize", [1, 1])))
        output_size = list(input_size)
    else:
        channel_count = int(first_layer_dict.get("Inputchannel", first_layer_dict.get("Outputchannel", 1)))
        input_size = list(first_layer_dict.get("Inputsize", first_layer_dict.get("Outputsize", [1, 1])))
        output_size = list(first_layer_dict.get("Outputsize", input_size))

    return OrderedDict(
        [
            ("type", "conv"),
            ("Inputchannel", max(1, channel_count)),
            ("Outputchannel", max(1, channel_count)),
            ("Kernelsize", 1),
            ("Inputsize", input_size),
            ("Outputsize", output_size),
            ("Stride", 1),
            ("Padding", 0),
            ("Inputbit", inputbit),
            ("Weightbit", weightbit),
            ("outputbit", outputbit),
            ("row_split_num", 1),
            ("weight_bit_split_part", 1),
            ("Inputindex", [-1]),
            ("Outputindex", [1] if original_layer_count > 0 else []),
            ("Layerindex", 0),
        ]
    )


def _mutable_layer_dict(layer_entry: Any) -> Mapping[str, Any] | None:
    if isinstance(layer_entry, Mapping):
        return layer_entry
    if isinstance(layer_entry, Sequence) and not isinstance(layer_entry, (str, bytes)):
        if not layer_entry:
            return None
        first = layer_entry[0]
        if isinstance(first, Mapping):
            return first
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)) and first:
            nested_first = first[0]
            if isinstance(nested_first, Mapping):
                return nested_first
    return None


def _shift_ids(ids: Sequence[int], offset: int, *, include_zero: bool = False) -> list[int]:
    shifted = [int(value) + offset for value in ids]
    if include_zero:
        shifted = [0, *shifted]
    return _dedupe_preserve_order(shifted)


def _shift_boundary_bindings(boundary_bindings, offset: int):
    shifted = []
    for binding in boundary_bindings:
        cloned = deepcopy(binding)
        if cloned.local_layer_id is not None:
            cloned.local_layer_id = int(cloned.local_layer_id) + offset
        shifted.append(cloned)
    return shifted


def _dedupe_preserve_order(values: Sequence[int]) -> list[int]:
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(int(value))
    return deduped


def _layer_id(raw_layer: Mapping[str, Any], index: int) -> int:
    return int(raw_layer.get("local_layer_id", index))


def _layer_type(netstruct: Sequence[Any], local_layer_id: int) -> str:
    layer_dict = _layer_dict(netstruct[local_layer_id])
    return str(layer_dict.get("type", "unknown"))


def _layer_dict(layer_entry: Any) -> Mapping[str, Any]:
    if isinstance(layer_entry, Mapping):
        return layer_entry
    if isinstance(layer_entry, Sequence) and not isinstance(layer_entry, (str, bytes)):
        if not layer_entry:
            return {}
        first = layer_entry[0]
        if isinstance(first, Mapping):
            return first
        if isinstance(first, Sequence) and not isinstance(first, (str, bytes)) and first:
            nested_first = first[0]
            if isinstance(nested_first, Mapping):
                return nested_first
    return {}


def _invalid_ids(ids: Sequence[Any], layer_count: int) -> list[Any]:
    invalid = []
    for value in ids:
        if not isinstance(value, int) or value < 0 or value >= layer_count:
            invalid.append(value)
    return invalid


def _indexed_mapping(value: Any, index: int) -> dict[str, Any]:
    selected = _indexed_value(value, index)
    if isinstance(selected, Mapping):
        return dict(selected)
    return {}


def _indexed_min(value: Any, index: int) -> float:
    numbers = list(_numbers(_indexed_value(value, index)))
    return min(numbers, default=0.0)


def _indexed_max(value: Any, index: int) -> float:
    numbers = list(_numbers(_indexed_value(value, index)))
    return max(numbers, default=0.0)


def _nested_max(value: Any) -> float:
    return max(_numbers(value), default=0.0)


def _indexed_number(value: Any, index: int) -> float:
    selected = _indexed_value(value, index)
    return _number(selected)


def _indexed_value(value: Any, index: int) -> Any:
    if isinstance(value, Mapping):
        if index in value:
            return value[index]
        return value.get(str(index), 0.0)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if 0 <= index < len(value):
            return value[index]
        return 0.0
    return value


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _seconds(value: float) -> float:
    return round(value, 18)


def _numbers(value: Any):
    if isinstance(value, Mapping):
        for nested in value.values():
            yield from _numbers(nested)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            yield from _numbers(nested)
        return
    try:
        yield float(value)
    except (TypeError, ValueError):
        return
