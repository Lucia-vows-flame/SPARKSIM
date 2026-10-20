"""End-to-end Phase 2 execution loop."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any

from MNSIM.Hardware_Model.analog_lif_lut import load_analog_lif_lut
from MNSIM.Hardware_Model.analog_lif_area import estimate_analog_lif_area
from MNSIM.Scheduling.phase2.analog_lif import AnalogLifBackend, AnalogLifError
from MNSIM.Scheduling.phase2.analog_energy import load_snn_peripheral_energy
from MNSIM.Scheduling.phase2.analog_peripherals import score_analog_peripherals, score_static_peripherals, spike_destinations
from MNSIM.Scheduling.phase2.analog_lif_inputs import (
    AnalogLifPayloadError,
    AnalogLifPayloadSession,
)
from MNSIM.Scheduling.phase2.boundary import BoundaryStateManager
from MNSIM.Scheduling.phase2.digital_lif import DigitalLifCostModel, DigitalLifError
from MNSIM.Scheduling.phase2.ddr import DdrBackendError, load_ddr_backend
from MNSIM.Scheduling.phase2.lowerer import RoundLowerer
from MNSIM.Scheduling.phase2.lif_mode import LifModeConfigError, load_lif_mode_config
from MNSIM.Scheduling.phase2.physical_write_ledger import AnnRoundPhysicalWriteLedger
from MNSIM.Scheduling.phase2.partial_accumulator import (
    ScheduledPartialAccumulatorCostModel,
)
from MNSIM.Scheduling.phase2.mnsim_static_eval import run_mnsim_static_eval
from MNSIM.Scheduling.phase2.round_local_state import RoundLocalState
from MNSIM.Scheduling.phase2.report import ReportAggregator
from MNSIM.Scheduling.phase2.static_power_ledger import load_static_power_ledger
from MNSIM.Scheduling.phase2.types import (
    SOURCE_KIND_ROUND_FINALIZE,
    SOURCE_KIND_STATIC_EVAL,
    BoundaryCommitRequest,
    BoundaryCommitResult,
    MetricTotals,
    PHYSICAL_XBAR_UNIT_MISMATCH,
    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
    Phase2Invalid,
    Phase2Report,
    StepExecutionResult,
)
from MNSIM.Scheduling.round_types import (
    WeightedLayerSegment,
    WeightedPhysicalRound,
    canonical_ordered_segments,
)


def execute_round_plan(
    plan,
    sim_config_path,
    ops_total,
    ops_count_policy,
    static_backend=None,
    model_inputs=None,
    area_um2=None,
    xbar_size=None,
    latency_api="pipe",
    latency_fill_mode=1,
    weight_codes_by_layer=None,
    placement_snapshots=None,
    schedule_artifact=None,
) -> Phase2Report:
    """Execute fixed Phase 1 execution items through the Phase 2 stack."""
    if schedule_artifact is not None:
        validate = getattr(schedule_artifact, "validate", None)
        if not callable(validate):
            raise Phase2Invalid(
                "INVALID_SCHEDULE_ARTIFACT",
                "schedule_artifact must expose validate()",
            )
        try:
            validate()
        except Exception as exc:
            raise Phase2Invalid("INVALID_SCHEDULE_ARTIFACT", str(exc)) from exc
        artifact_snapshots = {
            str(record.round_id): record.placement_snapshot
            for record in getattr(schedule_artifact, "rounds", [])
            if getattr(record, "placement_snapshot", None) is not None
        }
        if placement_snapshots is not None and artifact_snapshots:
            if dict(placement_snapshots) != artifact_snapshots:
                raise Phase2Invalid(
                    "SCHEDULE_PLACEMENT_SNAPSHOT_MISMATCH",
                    "explicit placement snapshots disagree with schedule artifact",
                )
        placement_snapshots = artifact_snapshots
        weighted_item_ids = {
            str(item.item_id)
            for item in _execution_items_in_order(plan)
            if isinstance(item, WeightedPhysicalRound)
        }
        missing_snapshots = sorted(weighted_item_ids - set(placement_snapshots))
        if missing_snapshots:
            raise Phase2Invalid(
                "MISSING_PLACEMENT_SNAPSHOT",
                f"schedule artifact has no placement snapshot for {missing_snapshots[:5]}",
            )
    if not bool(getattr(plan, "feasible", False)):
        detail = getattr(plan, "invalid_reason", None) or "RoundPlan.feasible is false"
        raise Phase2Invalid("INFEASIBLE_ROUND_PLAN", str(detail))

    resolved_xbar_size = _resolve_xbar_size(xbar_size, getattr(plan, "hardware_budget", {}))
    required_xbars_by_layer = _required_xbars_by_layer(plan)
    has_lif = _plan_has_lif(plan)
    residual_nodes = [node for item in plan.execution_items for node in item.local_graph_nodes
                      if node.digital_metadata.get('snn_merge_mode') == 'residual_add']
    lif_config = None
    analog_backend = None
    analog_payloads = None
    analog_area = None
    if has_lif:
        try:
            lif_config = load_lif_mode_config(sim_config_path)
            if lif_config.mode == "analog_lut":
                analog_backend = AnalogLifBackend(
                    load_analog_lif_lut(
                        lif_config.lut_root,
                        verify_checksums=lif_config.verify_checksums,
                        profile=lif_config.profile,
                    ),
                    negative_vmem_policy=lif_config.negative_vmem_policy,
                )
                analog_payloads = AnalogLifPayloadSession(
                    model_inputs,
                    sim_config_path=sim_config_path,
                    weight_codes_by_layer=weight_codes_by_layer,
                )
                analog_area = _load_analog_lif_area()
        except (LifModeConfigError, AnalogLifError, AnalogLifPayloadError, ValueError) as exc:
            code = getattr(exc, "code", "INVALID_ANALOG_LIF_CONFIGURATION")
            detail = getattr(exc, "detail", str(exc))
            raise Phase2Invalid(str(code), str(detail)) from exc
    try:
        ddr_backend = load_ddr_backend(sim_config_path)
    except DdrBackendError as exc:
        raise Phase2Invalid("INVALID_DDR_CONFIGURATION", str(exc)) from exc
    boundary = BoundaryStateManager(ddr_backend=ddr_backend)
    _seed_model_inputs(boundary, model_inputs)
    lowerer = RoundLowerer(
        xbar_size=resolved_xbar_size,
        sim_config_path=sim_config_path,
        required_xbars_by_layer=required_xbars_by_layer,
        latency_api=latency_api,
        latency_fill_mode=latency_fill_mode,
        placement_snapshots=placement_snapshots,
    )
    static_power_ledger = load_static_power_ledger(sim_config_path)
    aggregator = ReportAggregator(
        model_name=_model_name(plan),
        sim_config_path=sim_config_path,
        ops_total=ops_total,
        ops_count_policy=ops_count_policy,
        area_um2=area_um2,
        accounting_scope=_accounting_scope(
            sim_config_path,
            central_static_power_enabled=static_power_ledger is not None,
            lif_mode=getattr(lif_config, "mode", "digital"),
            ddr_backend=ddr_backend,
        ),
        static_power_ledger=static_power_ledger,
    )
    # A custom algorithm name is descriptive; it cannot enable execution
    # semantics absent from the custom format (such as inactive-cell retention).
    retain_inactive_cells = (
        str(getattr(schedule_artifact, "policy_id", "")) == "snn_flashsef_temporal"
        and "custom_schedule_schema_version" not in getattr(schedule_artifact, "policy_metadata", {})
    )
    if retain_inactive_cells:
        aggregator.accounting_scope["weight_programming"] = {
            **dict(aggregator.accounting_scope.get("weight_programming", {})),
            "inactive_cell_policy": "retain_outside_fragment_shape",
        }
    if residual_nodes:
        aggregator.accounting_scope['residual_snn'] = {
            'merge_backend': 'existing_mnsim_element_sum',
            'lif_backend': 'existing_digital_lif',
            'ownership_policy': 'main_branch_reducer',
            'bn_arithmetic_included': False,
            'requantization_arithmetic_included': False,
            'note': 'BN hooks and post-add quantization retain existing MNSIM cost scope; '
                    'their additional arithmetic costs are not characterized.',
        }
    if weight_codes_by_layer is None:
        write_ledger = None
    elif retain_inactive_cells:
        write_ledger = AnnRoundPhysicalWriteLedger(
            sim_config_path=sim_config_path,
            weight_codes_by_layer=weight_codes_by_layer,
            preserve_inactive_cells=True,
        )
    else:
        # Keep the narrow constructor contract used by injected test ledgers
        # and legacy callers that do not implement the optional optimization.
        write_ledger = AnnRoundPhysicalWriteLedger(
            sim_config_path=sim_config_path,
            weight_codes_by_layer=weight_codes_by_layer,
        )
    peripheral_energy = load_snn_peripheral_energy(sim_config_path)
    if peripheral_energy is not None and (not has_lif or (lif_config.mode != "analog_lut" and peripheral_energy.execution_mode != "mnsim_static_topology")):
        raise Phase2Invalid(
            "MISSING_SNN_PERIPHERAL_ACTIVITY",
            "additional peripherals require numerical analog read events; digital hardware-only execution has no actual row/spike activity",
        )
    if peripheral_energy is not None:
        aggregator.accounting_scope["snn_additional_peripherals"] = {
            "energy_included": True,
            "profile_id": peripheral_energy.profile["profile_id"],
            "existing_frontend_lif_unchanged": True,
            "additional_model_external_energy_read": False,
            "buffer_and_partial_sum_owner": "existing_models_no_duplicate_charge",
        }
    sequence_length = _sequence_length(plan)
    sample_id = "sample:0"
    lif_invocations: dict[tuple[str, str], int] = {}
    ownership_by_layer: dict[str, dict[int, int]] = {}
    ownership_intervals_by_layer: dict[str, dict[tuple[int, int], int]] = {}
    lif_cost_model = (
        DigitalLifCostModel(sim_config_path)
        if has_lif and getattr(lif_config, "mode", "digital") == "digital"
        else None
    )
    partial_accumulator_cost_model = ScheduledPartialAccumulatorCostModel(
        sim_config_path, plan
    )

    for item in _execution_items_in_order(plan):
        boundary_input = _build_sequence_input_view(
            boundary, item, sample_id=sample_id, timestep=0, sequence_length=sequence_length
        )
        prepared = lowerer.prepare_round(item, boundary_input)
        if has_lif and prepared.placement is not None:
            _update_output_ownership_intervals(
                ownership_intervals_by_layer,
                prepared.placement,
            )
            ownership_by_layer = _aggregate_output_ownership_intervals(
                ownership_intervals_by_layer
            )
        # A weighted physical round begins by materializing its target device
        # state.  The subsequent static work is the read phase for that same
        # resident state, so a failed or missing write ledger cannot be hidden
        # behind a completed read evaluation.
        write_programming = (
            write_ledger.score_round(item_id=item.item_id, placement=prepared.placement)
            if write_ledger is not None and isinstance(item, WeightedPhysicalRound)
            else None
        )
        step_results: list[StepExecutionResult] = []
        timestep_commits: list[BoundaryCommitResult] = []

        for timestep in range(sequence_length):
            boundary_input = _build_sequence_input_view(
                boundary,
                item,
                sample_id=sample_id,
                timestep=timestep,
                sequence_length=sequence_length,
            )
            local_state = RoundLocalState.seed(item, boundary_input)
            timestep_results: list[StepExecutionResult] = []
            for segment in canonical_ordered_segments(item):
                current_input = local_state.input_view_for(segment)
                lowered = lowerer.lower_step(
                    prepared,
                    segment,
                    current_input,
                    sample_id=sample_id,
                    timestep=timestep,
                    sequence_length=sequence_length,
                )
                step_result = _execute_lowered_step(
                    lowered,
                    static_backend,
                    lif_cost_model=lif_cost_model,
                    lif_invocations=lif_invocations,
                    ownership_by_layer=ownership_by_layer,
                    item=item,
                    plan=plan,
                    sim_config_path=sim_config_path,
                    lif_mode=getattr(lif_config, "mode", "digital"),
                    analog_backend=analog_backend,
                    analog_payloads=analog_payloads,
                    analog_area=analog_area,
                )
                local_state.commit_step(step_result)
                timestep_results.append(step_result)
                step_results.append(step_result)
                if peripheral_energy is not None and step_result.source_kind == "analog_lif":
                    extra = score_analog_peripherals(
                        peripheral_energy, step_result, prepared.placement,
                        destinations=spike_destinations(plan, step_result.origin_layer_ids[0]),
                    )
                    timestep_results.append(extra)
                    step_results.append(extra)
                elif (
                    peripheral_energy is not None
                    and peripheral_energy.execution_mode == "mnsim_static_topology"
                    and step_result.source_kind == SOURCE_KIND_STATIC_EVAL
                ):
                    extra = score_static_peripherals(
                        peripheral_energy, step_result, prepared.placement,
                        sim_config_path=sim_config_path, sample_id=sample_id, timestep=timestep,
                    )
                    if extra is not None:
                        timestep_results.append(extra)
                        step_results.append(extra)

            _validate_weighted_item_fragment_proof(item, timestep_results)
            timestep_commits.append(
                boundary.commit(
                    BoundaryCommitRequest(
                        item_id=item.item_id,
                        item_kind=item.item_kind,
                        execution_index=item.item_index,
                        input_view_id=boundary_input.input_view_id,
                        source_kind=SOURCE_KIND_ROUND_FINALIZE,
                        static_eval_result=None,
                        hook_result=None,
                        boundary_bindings=local_state.boundary_bindings_for_finalize(),
                        step_results=timestep_results,
                        input_events=boundary_input.events,
                        sample_id=sample_id,
                        timestep=timestep,
                    )
                )
            )

        commit_result = _merge_timestep_commits(item, timestep_commits)
        accumulator_result = partial_accumulator_cost_model.score_commit(
            item, commit_result
        )
        if accumulator_result is not None:
            step_results.append(accumulator_result)
        aggregator.record_round(
            item,
            prepared.placement,
            step_results,
            commit_result,
            write_programming=write_programming,
        )

    for lif_node_id in getattr(getattr(plan, "sequence_spec", None), "recurrent_state_ids", []) or []:
        count = lif_invocations.get((sample_id, str(lif_node_id)), 0)
        if count != sequence_length:
            raise Phase2Invalid(
                "INCOMPLETE_LIF_INVOCATION_STREAM",
                f"LIF {lif_node_id} expected {sequence_length} invocations, got {count}",
            )

    return aggregator.finalize()


def _accounting_scope(
    sim_config_path: str,
    *,
    central_static_power_enabled: bool = False,
    lif_mode: str = "digital",
    ddr_backend: Any | None = None,
) -> dict[str, Any]:
    """Expose modeled and intentionally omitted physical cost domains."""

    import configparser

    parser = configparser.ConfigParser()
    try:
        loaded = parser.read(sim_config_path, encoding="UTF-8")
        include_buffer_leakage = (
            parser.getboolean(
                "Architecture level",
                "Buffer_IncludeLeakageEnergy",
                fallback=True,
            )
            if loaded
            else True
        )
    except (OSError, ValueError, configparser.Error):
        # The original static backend owns detailed SimConfig validation.  This
        # report-only descriptor must not replace its stable error behavior.
        include_buffer_leakage = True

    static_scope = {
        "buffer_energy_included": bool(
            include_buffer_leakage or central_static_power_enabled
        ),
        "other_energy_included": False,
    }
    if central_static_power_enabled:
        static_scope.update(
            {
                "legacy_access_window_buffer_leakage_included": bool(
                    include_buffer_leakage
                ),
                "central_sram_retention_energy_included": True,
            }
        )

    scope = {
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
            "capacity_included": False,
            "retention_energy_included": False,
            "dynamic_access_energy_included": False,
            "latency_included": False,
            "spill_to_ddr": False,
        },
        "static_leakage": static_scope,
        "scheduled_partial_accumulator": {
            "energy_included": True,
            "latency_included": True,
            "partial_state_buffer_included": True,
            "primitive": "MNSIM.Hardware_Model.Adder+Buffer",
        },
    }
    if ddr_backend is not None and bool(getattr(ddr_backend, "enabled", False)):
        scope["ddr_boundary_transfer"] = {
            "used": True,
            "backend": "dramsim3",
            "energy_included": True,
            "latency_included": True,
            "accounting_owner": "BoundaryStateManager",
        }
    if lif_mode == "analog_lut":
        scope["adc_conversion"] = {
            "hardware_enabled": True,
            "raw_audit_included": True,
            "final_included": False,
            "latency_included": False,
            "energy_included": False,
            "power_included": False,
            "area_included": False,
            "virtual_output_lane_geometry_preserved": True,
            "reason": "original MNSIM ADC retained in raw audit and subtracted during final analog aggregation",
        }
        scope["analog_frontend_lif"] = {
            "latency_included": True,
            "energy_included": True,
            "power_included": True,
            "area_included": True,
            "area_status": "hspice_netlist_estimate",
            "area_source": "checked_in_transistor_level_netlists",
        }
    return scope


def _execute_lowered_step(
    lowered,
    static_backend,
    *,
    lif_cost_model=None,
    lif_invocations=None,
    ownership_by_layer=None,
    item=None,
    plan=None,
    sim_config_path=None,
    lif_mode="digital",
    analog_backend=None,
    analog_payloads=None,
    analog_area=None,
) -> StepExecutionResult:
    if lowered.digital_lif_request is not None:
        request = lowered.digital_lif_request
        key = (request.invocation.sample_id, request.invocation.node_id)
        expected = (lif_invocations or {}).get(key, 0)
        if request.invocation.timestep != expected:
            raise Phase2Invalid(
                "INVALID_LIF_INVOCATION_ORDER",
                f"{key} expected timestep {expected}, got {request.invocation.timestep}",
            )
        if lif_mode == "analog_lut":
            return _execute_analog_lif_step(
                lowered,
                request,
                key=key,
                expected_timestep=expected,
                lif_invocations=lif_invocations,
                analog_backend=analog_backend,
                analog_payloads=analog_payloads,
                analog_area=analog_area,
            )
        producer_id = _lif_producer_id(item, request.invocation.node_id)
        # Residual adapters bind BN/add outputs to the main branch reducer.
        # Keep the original direct weighted->LIF path unchanged.
        lif_node = next((node for node in item.local_graph_nodes
                         if node.layer_id == request.invocation.node_id), None)
        if lif_node is not None:
            producer_id = lif_node.digital_metadata.get('lif_owner_layer_id', producer_id)
        producer_id = _resolve_lif_owner_through_bn(plan, producer_id)
        ownership = dict((ownership_by_layer or {}).get(producer_id, {}))
        if not ownership:
            raise Phase2Invalid(
                "MISSING_LIF_TILE_OWNERSHIP",
                f"LIF {request.invocation.node_id} has no placement ownership for {producer_id}",
            )
        ownership = _expand_output_ownership_to_neurons(
            item,
            producer_id,
            ownership,
            expected_neuron_count=int(request.neuron_count or 0),
            plan=plan,
        )
        if sum(ownership.values()) != int(request.neuron_count or 0):
            raise Phase2Invalid(
                "INCOMPLETE_LIF_TILE_OWNERSHIP",
                f"LIF {request.invocation.node_id} requires {request.neuron_count} neurons, "
                f"but placement proves {sum(ownership.values())}",
            )
        if lif_cost_model is None:
            lif_cost_model = DigitalLifCostModel(lowered.static_eval_requests[0].sim_config_path)
        try:
            cost = lif_cost_model.cost_sequence(
                bit_width=request.domain.signed_bit_width,
                ownership_by_tile=ownership,
                sequence_length=request.sequence_length,
                provisioned_tile_count=_provisioned_tile_count(sim_config_path),
            )
        except DigitalLifError as exc:
            raise Phase2Invalid("INVALID_DIGITAL_LIF_COST", str(exc)) from exc
        lif_invocations[key] = expected + 1
        binding = lowered.boundary_bindings[0]
        compute_latency_ns = cost.compute_latency_ns / request.sequence_length
        compute_energy_nj = cost.compute_energy_nj / request.sequence_length
        state_read_latency_ns = (
            cost.state_read_latency_ns / max(request.sequence_length - 1, 1)
            if request.invocation.timestep > 0
            else 0.0
        )
        state_read_energy_nj = (
            cost.state_read_energy_nj / max(request.sequence_length - 1, 1)
            if request.invocation.timestep > 0
            else 0.0
        )
        state_write_latency_ns = (
            cost.state_write_latency_ns / max(request.sequence_length - 1, 1)
            if request.invocation.timestep < request.sequence_length - 1
            else 0.0
        )
        state_write_energy_nj = (
            cost.state_write_energy_nj / max(request.sequence_length - 1, 1)
            if request.invocation.timestep < request.sequence_length - 1
            else 0.0
        )
        latency_ns = compute_latency_ns + state_read_latency_ns + state_write_latency_ns
        energy_nj = compute_energy_nj + state_read_energy_nj + state_write_energy_nj
        audit = cost.to_dict()
        audit.update(
            {
                "invocation": request.invocation.to_dict(),
                "invocation_compute_latency_ns": compute_latency_ns,
                "invocation_compute_energy_nj": compute_energy_nj,
                "invocation_state_read_latency_ns": state_read_latency_ns,
                "invocation_state_read_energy_nj": state_read_energy_nj,
                "invocation_state_write_latency_ns": state_write_latency_ns,
                "invocation_state_write_energy_nj": state_write_energy_nj,
            }
        )
        return StepExecutionResult(
            item_id=lowered.item_id,
            step_id=lowered.step_id,
            segment_index=lowered.segment_index,
            step_kind=lowered.step_kind,
            source_kind="digital_lif",
            status="ok",
            origin_layer_ids=[request.invocation.node_id],
            metrics=MetricTotals(
                latency_s=latency_ns * 1e-9,
                energy_j=energy_nj * 1e-9,
                power_w=(energy_nj / latency_ns)
                if latency_ns > 0
                else 0.0,
                area_um2=cost.provisioned_area_um2,
            ),
            layer_results=[],
            mapping_summary={"digital_lif": audit, "producer_layer_id": producer_id},
            raw_mnsim_audit=[{"audit_kind": "digital_lif", **audit}],
            boundary_bindings=list(lowered.boundary_bindings),
            warnings=[],
            sample_id=request.invocation.sample_id,
            timestep=request.invocation.timestep,
        )
    if lowered.hook_result is not None:
        return StepExecutionResult(
            item_id=lowered.item_id,
            step_id=lowered.step_id,
            segment_index=lowered.segment_index,
            step_kind=lowered.step_kind,
            source_kind="hook",
            status="ok",
            origin_layer_ids=_unique_origin_layer_ids(lowered.boundary_bindings),
            metrics=MetricTotals(0.0, 0.0, 0.0, 0.0),
            layer_results=[],
            mapping_summary={},
            raw_mnsim_audit=[],
            boundary_bindings=list(lowered.boundary_bindings),
            warnings=_merge_warnings(lowered.warnings, lowered.hook_result.warnings),
            sample_id=lowered.boundary_bindings[0].sample_id,
            timestep=lowered.boundary_bindings[0].timestep,
        )

    if not lowered.static_eval_requests:
        raise Phase2Invalid(
            "EMPTY_LOWERED_STEP",
            f"{lowered.step_id}: produced no static or hook work",
        )

    if lowered.placement_slices and len(lowered.placement_slices) != len(lowered.static_eval_requests):
        raise Phase2Invalid(
            "LOWERED_STEP_SCOPE_MISMATCH",
            (
                f"{lowered.step_id}: received {len(lowered.static_eval_requests)} static "
                f"requests but {len(lowered.placement_slices)} placement slices"
            ),
        )

    scope_results = []
    for scope_index, request in enumerate(lowered.static_eval_requests):
        placement_slice = (
            lowered.placement_slices[scope_index]
            if lowered.placement_slices
            else None
        )
        result = run_mnsim_static_eval(
            request,
            backend=static_backend,
            placement_slice=placement_slice,
        )
        result.warnings = _merge_warnings(lowered.warnings, result.warnings)
        if result.status != "ok":
            raise Phase2Invalid(
                "STATIC_EVAL_NOT_OK",
                f"{lowered.step_id}: static scope evaluation failed: {result.warnings}",
            )
        scope_results.append(result)

    latency_s = max(
        (float(result.item_totals.latency_s) for result in scope_results),
        default=0.0,
    )
    energy_j = sum(float(result.item_totals.energy_j) for result in scope_results)
    area_um2 = sum(float(result.item_totals.area_um2) for result in scope_results)
    metrics = MetricTotals(
        latency_s=latency_s,
        energy_j=energy_j,
        power_w=energy_j / latency_s if latency_s > 0 else 0.0,
        area_um2=area_um2,
    )
    return StepExecutionResult(
        item_id=lowered.item_id,
        step_id=lowered.step_id,
        segment_index=lowered.segment_index,
        step_kind=lowered.step_kind,
        source_kind=SOURCE_KIND_STATIC_EVAL,
        status="ok",
        origin_layer_ids=_unique_origin_layer_ids(lowered.boundary_bindings),
        metrics=metrics,
        layer_results=[
            layer
            for result in scope_results
            for layer in result.layer_results
        ],
        mapping_summary={
            "scope_count": len(scope_results),
            "scope_mappings": [
                deepcopy(result.mapping_summary)
                for result in scope_results
            ],
        },
        raw_mnsim_audit=[
            _static_scope_audit(lowered.step_id, scope_index, result)
            for scope_index, result in enumerate(scope_results)
        ],
        boundary_bindings=list(lowered.boundary_bindings),
        warnings=_merge_warnings(
            lowered.warnings,
            [
                warning
                for result in scope_results
                for warning in result.warnings
            ],
        ),
        sample_id=lowered.boundary_bindings[0].sample_id,
        timestep=lowered.boundary_bindings[0].timestep,
    )


def _execute_analog_lif_step(
    lowered,
    request,
    *,
    key,
    expected_timestep,
    lif_invocations,
    analog_backend,
    analog_payloads,
    analog_area,
) -> StepExecutionResult:
    if analog_backend is None or analog_payloads is None:
        raise Phase2Invalid(
            "INVALID_ANALOG_LIF_CONFIGURATION",
            "analog_lut mode has no initialized local backend or payload session",
        )
    neuron_count = int(request.neuron_count or 0)
    if neuron_count <= 0:
        raise Phase2Invalid(
            "INVALID_ANALOG_LIF_NEURON_COUNT",
            f"LIF {request.invocation.node_id} has invalid neuron count {neuron_count}",
        )
    try:
        numerical = analog_payloads.resolve(
            sample_id=request.invocation.sample_id,
            node_id=request.invocation.node_id,
            timestep=request.invocation.timestep,
            sequence_length=request.sequence_length,
            neuron_count=neuron_count,
        )
        result = analog_backend.execute(
            ipos_a=numerical.ipos_a,
            ineg_a=numerical.ineg_a,
            vmem0_v=numerical.vmem0_v,
        )
        analog_payloads.commit(
            sample_id=request.invocation.sample_id,
            node_id=request.invocation.node_id,
            timestep=request.invocation.timestep,
            expected_neuron_count=neuron_count,
            vmem_next_v=result.vmem_next,
        )
    except AnalogLifPayloadError as exc:
        raise Phase2Invalid(exc.code, exc.detail) from exc
    except (AnalogLifError, ValueError) as exc:
        raise Phase2Invalid("ANALOG_LIF_EXECUTION_FAILED", str(exc)) from exc

    lif_invocations[key] = expected_timestep + 1
    binding = lowered.boundary_bindings[0]
    state = {
        "vmem0_v": numerical.vmem0_v.tolist(),
        "raw_vmem_next_v": result.raw_vmem_next_v.tolist(),
        "vmem_next_v": result.vmem_next.tolist(),
        "spike_codes": list(result.spike_codes),
    }
    state["payload_identity"] = _analog_state_identity(
        request.invocation.to_dict(), state
    )
    spike_payload_identity = _analog_state_identity(
        request.invocation.to_dict(), {"spike_codes": state["spike_codes"]}
    )
    analog_binding = deepcopy(binding)
    analog_binding.numerical_input_payload_identity = numerical.payload_identity
    analog_binding.state_payload_identity = state["payload_identity"]
    analog_binding.spike_payload_identity = spike_payload_identity
    analog_binding.spike_codes = list(result.spike_codes)
    boundary_binding = analog_binding.to_dict()
    audit = deepcopy(dict(result.audit_metadata))
    audit.update(
        {
            "backend": "local_analog_lif_lut",
            "invocation": request.invocation.to_dict(),
            "neuron_count": neuron_count,
            "numerical_input": {
                "source": numerical.source,
                "payload_identity": numerical.payload_identity,
                "ipos_a": numerical.ipos_a.tolist(),
                "ineg_a": numerical.ineg_a.tolist(),
                "crossbar": deepcopy(numerical.crossbar_audit),
            },
            "state": state,
            "boundary_binding": boundary_binding,
            "frontend_iout_to_lif_a": result.frontend_iout_to_lif_a.tolist(),
            "latency_s": result.cost.latency_s,
            "energy_j": result.cost.energy_j,
            "power_w": result.cost.power_w,
            "area_characterized": True,
            "area_um2_per_neuron": analog_area.area_um2_per_neuron,
            "area_um2": analog_area.for_neurons(neuron_count),
            "area_status": "hspice_netlist_estimate",
            "area_provenance": analog_area.to_dict(),
        }
    )
    return StepExecutionResult(
        item_id=lowered.item_id,
        step_id=lowered.step_id,
        segment_index=lowered.segment_index,
        step_kind=lowered.step_kind,
        source_kind="analog_lif",
        status="ok",
        origin_layer_ids=[request.invocation.node_id],
        metrics=MetricTotals(
            latency_s=result.cost.latency_s,
            energy_j=result.cost.energy_j,
            power_w=result.cost.power_w,
            area_um2=analog_area.for_neurons(neuron_count),
        ),
        layer_results=[],
        mapping_summary={"analog_lif": audit},
        raw_mnsim_audit=[{"audit_kind": "analog_lif", **deepcopy(audit)}],
        boundary_bindings=[analog_binding],
        warnings=[],
        sample_id=request.invocation.sample_id,
        timestep=request.invocation.timestep,
    )


def _analog_state_identity(invocation: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        {"invocation": dict(invocation), "state": dict(state)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _load_analog_lif_area():
    """Load checked-in transistor decks used by the analog-LIF accounting."""
    module_path = Path(__file__).resolve()
    candidates = list(module_path.parents)
    repository_root = module_path.parents[3]
    candidates.append(repository_root / "archive" / "legacy_master_20260925")
    root = next(
        (
            candidate
            for candidate in candidates
            if (candidate / "hspice_win_reference_ccii_direct_vcm_lif_gain4_20260725").is_dir()
        ),
        None,
    )
    if root is None:
        raise FileNotFoundError("checked-in analog LIF HSPICE netlists are unavailable")
    frontend = root / "hspice_win_reference_ccii_direct_vcm_lif_gain4_20260725" / "01_src" / "reference_ccii_current_diff_core.sp"
    lif_root = root / "hspice_win_lif_context_threshold_calibration_20260726" / "01_src"
    return estimate_analog_lif_area(
        frontend_netlist=frontend,
        lif_netlists=(
            lif_root / "comp_ct_simple.sp",
            lif_root / "spike_sr_latch.sp",
            lif_root / "lif_current_rate_cell.sp",
        ),
    )


def _static_scope_audit(step_id: str, scope_index: int, result) -> dict[str, Any]:
    return {
        "audit_kind": "static",
        "item_id": result.item_id,
        "item_kind": result.item_kind,
        "step_id": step_id,
        "scope_index": scope_index,
        "mapping_summary": deepcopy(result.mapping_summary),
        "raw_mnsim_summary": deepcopy(result.raw_mnsim_summary),
        "context_layer_ids": [
            layer.local_layer_id
            for layer in result.layer_results
            if layer.report_role == "context_only"
        ],
    }


def _validate_weighted_item_fragment_proof(item, step_results) -> None:
    """Check that static scopes reproduce the fixed Phase 1 physical count."""
    if not isinstance(item, WeightedPhysicalRound):
        return

    expected_used_xbars = getattr(getattr(item, "resource_proof", None), "used_xbars", None)
    if (
        isinstance(expected_used_xbars, bool)
        or not isinstance(expected_used_xbars, Integral)
        or expected_used_xbars <= 0
    ):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            f"{item.item_id}: invalid weighted resource proof {expected_used_xbars!r}",
        )

    active_physical_xbars = 0
    for result in step_results:
        if getattr(result, "source_kind", None) != SOURCE_KIND_STATIC_EVAL:
            continue
        for audit in list(getattr(result, "raw_mnsim_audit", []) or []):
            raw_summary = dict(audit.get("raw_mnsim_summary", {}) or {})
            active_count = raw_summary.get("active_physical_xbar_count", 0)
            if (
                isinstance(active_count, bool)
                or not isinstance(active_count, Integral)
                or active_count < 0
            ):
                raise Phase2Invalid(
                    TCG_ACTIVE_XBAR_PROOF_MISMATCH,
                    (
                        f"{item.item_id}: static audit has invalid active physical "
                        f"xbar count {active_count!r}"
                    ),
                )
            active_physical_xbars += int(active_count)

    if active_physical_xbars != int(expected_used_xbars):
        raise Phase2Invalid(
            TCG_ACTIVE_XBAR_PROOF_MISMATCH,
            (
                f"{item.item_id}: static active physical count {active_physical_xbars} "
                f"does not equal Phase 1 used_xbars {expected_used_xbars}"
            ),
        )


def _unique_origin_layer_ids(bindings) -> list[str]:
    origin_layer_ids: list[str] = []
    for binding in bindings:
        origin_layer_id = str(binding.origin_layer_id)
        if origin_layer_id not in origin_layer_ids:
            origin_layer_ids.append(origin_layer_id)
    return origin_layer_ids


def _execution_items_in_order(plan) -> list[Any]:
    return sorted(
        list(getattr(plan, "execution_items", []) or []),
        key=lambda item: int(getattr(item, "item_index", 0)),
    )


def _sequence_length(plan) -> int:
    value = getattr(getattr(plan, "sequence_spec", None), "sequence_length", 1)
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise Phase2Invalid("INVALID_SEQUENCE_SPEC", f"invalid sequence length {value!r}")
    return int(value)


def _plan_has_lif(plan: Any) -> bool:
    return any(
        str(getattr(node, "layer_type", "")) == "lif"
        for item in _execution_items_in_order(plan)
        for node in list(getattr(item, "local_graph_nodes", []) or [])
    )


def _build_sequence_input_view(
    boundary: BoundaryStateManager,
    item: Any,
    *,
    sample_id: str,
    timestep: int,
    sequence_length: int,
):
    state_ids: list[str] = []
    for ref in getattr(item, "external_inputs", []) or []:
        state_id = str(getattr(ref, "state_id", ref))
        if state_id == "model_input" or sequence_length == 1:
            resolved = state_id
        else:
            producer_layer_id = state_id.removeprefix("state:").removesuffix(":partial")
            resolved = (
                f"state:{_safe_state_id(producer_layer_id)}:sample:"
                f"{_safe_state_id(sample_id)}:t{int(timestep)}"
            )
        if resolved not in state_ids:
            state_ids.append(resolved)
    return boundary.build_input_view_for_state_ids(item.item_id, state_ids)


def _placement_output_ownership(placement: Any) -> dict[str, dict[int, int]]:
    ownership = getattr(placement, "output_owner_shards_by_layer", None)
    if not isinstance(ownership, dict):
        raise Phase2Invalid(
            "MISSING_LIF_TILE_OWNERSHIP",
            f"placement {getattr(placement, 'item_id', '<unknown>')} has no reducer ownership proof",
        )
    return {
        str(layer_id): {int(tile_id): int(count) for tile_id, count in shards.items()}
        for layer_id, shards in ownership.items()
    }


def _update_output_ownership_intervals(
    ownership_intervals_by_layer: dict[str, dict[tuple[int, int], int]],
    placement: Any,
) -> None:
    """Merge one round's final reducer proof without losing earlier output shards."""

    fragment_specs = getattr(placement, "fragment_specs_by_id", None)
    fragment_slots = getattr(placement, "fragment_slots_by_id", None)
    if not isinstance(fragment_specs, dict) or not isinstance(fragment_slots, dict):
        raise Phase2Invalid(
            "MISSING_LIF_TILE_OWNERSHIP",
            f"placement {getattr(placement, 'item_id', '<unknown>')} has no fragment ownership proof",
        )

    ordered_fragment_ids = sorted(
        fragment_specs,
        key=lambda fragment_id: (
            int(fragment_slots[fragment_id]["tile_id"]),
            int(fragment_slots[fragment_id]["pe_id"]),
            int(fragment_slots[fragment_id]["group_id"]),
            str(fragment_id),
        ),
    )
    for fragment_id in ordered_fragment_ids:
        fragment = fragment_specs[fragment_id]
        interval = tuple(int(value) for value in fragment.output_range)
        full_interval = tuple(int(value) for value in fragment.full_output_range)
        if (
            len(interval) != 2
            or len(full_interval) != 2
            or interval[0] < full_interval[0]
            or interval[1] > full_interval[1]
            or interval[1] <= interval[0]
        ):
            raise Phase2Invalid(
                "MISSING_LIF_TILE_OWNERSHIP",
                f"placement {getattr(placement, 'item_id', '<unknown>')} has invalid output interval",
            )
        layer_intervals = ownership_intervals_by_layer.setdefault(
            str(fragment.layer_id), {}
        )
        # A later input-range fragment for the same output interval is the
        # reducer holding the completed accumulator for that interval.
        layer_intervals[interval] = int(fragment_slots[fragment_id]["tile_id"])


def _aggregate_output_ownership_intervals(
    ownership_intervals_by_layer: dict[str, dict[tuple[int, int], int]],
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for layer_id, intervals in ownership_intervals_by_layer.items():
        owner_by_channel: dict[int, int] = {}
        for (start, end), tile_id in intervals.items():
            for channel_id in range(start, end):
                existing = owner_by_channel.get(channel_id)
                if existing is not None and existing != tile_id:
                    raise Phase2Invalid(
                        "MISSING_LIF_TILE_OWNERSHIP",
                        f"layer {layer_id} output channel {channel_id} has overlapping owners",
                    )
                owner_by_channel[channel_id] = tile_id
        counts: dict[int, int] = {}
        for tile_id in owner_by_channel.values():
            counts[tile_id] = counts.get(tile_id, 0) + 1
        result[layer_id] = counts
    return result


def _resolve_lif_owner_through_bn(plan: Any, producer_id: str) -> str:
    """BN preserves the weighted reducer's tile ownership, including across rounds."""
    if plan is None:
        return producer_id
    nodes = {node.layer_id: node for item in plan.execution_items for node in item.local_graph_nodes}
    visited = set()
    while producer_id in nodes and nodes[producer_id].layer_type == "bn":
        if producer_id in visited:
            raise Phase2Invalid("INVALID_LIF_PRODUCER", "cycle in BN ownership chain")
        visited.add(producer_id)
        parents = set()
        for item in plan.execution_items:
            parents.update(str(edge.producer_node_id) for edge in item.same_item_edges
                           if edge.consumer_node_id == producer_id)
            parents.update(str(ref.state_id).removeprefix("state:")
                           for ref in item.external_inputs
                           if producer_id in ref.consumer_node_ids and str(ref.state_id).startswith("state:"))
        if len(parents) != 1:
            raise Phase2Invalid("INVALID_LIF_PRODUCER", f"BN {producer_id} requires exactly one ownership source")
        producer_id = parents.pop()
    return producer_id


def _lif_producer_id(item: Any, lif_node_id: str) -> str:
    producers = [
        str(edge.producer_node_id)
        for edge in getattr(item, "same_item_edges", []) or []
        if str(edge.consumer_node_id) == str(lif_node_id)
    ]
    if len(producers) == 1:
        return producers[0]
    # A digital LIF may consume the completed output of a weighted layer whose
    # final partial-sum round is different from the LIF round. The boundary
    # contract names that producer item explicitly.
    external_producers = {
        str(ref.producer_item_id)
        for ref in getattr(item, "external_inputs", []) or []
        if str(lif_node_id) in {
            str(node_id) for node_id in getattr(ref, "consumer_node_ids", []) or []
        }
        and str(getattr(ref, "partial_or_complete", "complete")) == "complete"
        and getattr(ref, "producer_item_id", None) is not None
    }
    if len(external_producers) == 1:
        state_id = next(
            ref.state_id
            for ref in getattr(item, "external_inputs", []) or []
            if str(lif_node_id) in {
                str(node_id) for node_id in getattr(ref, "consumer_node_ids", []) or []
            }
            and str(getattr(ref, "partial_or_complete", "complete")) == "complete"
            and getattr(ref, "producer_item_id", None) is not None
        )
        return str(state_id).removeprefix("state:").removesuffix(":partial")
    if len(producers) != 1:
        raise Phase2Invalid(
            "INVALID_LIF_INPUT",
            f"LIF {lif_node_id} must have exactly one same-round producer, got {producers}",
        )
    return producers[0]


def _expand_output_ownership_to_neurons(
    item: Any,
    producer_id: str,
    ownership: dict[int, int],
    *,
    expected_neuron_count: int,
    plan: Any = None,
) -> dict[int, int]:
    """Expand placement-owned output channels across spatial positions."""

    if sum(ownership.values()) == expected_neuron_count:
        return ownership
    producer = next(
        (
            node
            for node in getattr(item, "local_graph_nodes", []) or []
            if str(getattr(node, "layer_id", "")) == str(producer_id)
        ),
        None,
    )
    if producer is None and plan is not None:
        producer = next(
            (
                node
                for plan_item in _execution_items_in_order(plan)
                for node in list(getattr(plan_item, "local_graph_nodes", []) or [])
                if str(getattr(node, "layer_id", "")) == str(producer_id)
            ),
            None,
        )
    if producer is None and plan is not None:
        producer = next(
            (
                entry.prepass_ref
                for plan_item in _execution_items_in_order(plan)
                for entry in list(getattr(plan_item, "selected_entries", []) or [])
                if str(getattr(entry, "layer_id", "")) == str(producer_id)
            ),
            None,
        )
    producer_shape = dict(getattr(producer, "node_shape", {}) or {})
    if not producer_shape:
        producer_shape = dict(getattr(producer, "shape_signature", {}) or {})
    output_size = producer_shape.get("Outputsize")
    if isinstance(output_size, (list, tuple)) and len(output_size) == 2:
        spatial = int(output_size[0]) * int(output_size[1])
        expanded = {
            int(tile_id): int(channel_count) * spatial
            for tile_id, channel_count in ownership.items()
        }
        if sum(expanded.values()) == expected_neuron_count:
            return expanded
    return ownership


def _provisioned_tile_count(sim_config_path: str) -> int:
    import configparser

    parser = configparser.ConfigParser()
    if not parser.read(sim_config_path):
        raise Phase2Invalid(
            "INVALID_HARDWARE_PROJECTION", f"cannot read {sim_config_path}"
        )
    tile_num = parser.get("Architecture level", "Tile_Num").split(",")
    return int(tile_num[0]) * int(tile_num[1])


def _merge_timestep_commits(item: Any, commits: list[BoundaryCommitResult]) -> BoundaryCommitResult:
    if not commits:
        raise Phase2Invalid("INVALID_BOUNDARY_COMMIT", f"{item.item_id} has no timestep commits")
    return BoundaryCommitResult(
        item_id=item.item_id,
        item_kind=item.item_kind,
        status="ok",
        created_states=[state for commit in commits for state in commit.created_states],
        updated_states=[state for commit in commits for state in commit.updated_states],
        released_states=[state for commit in commits for state in commit.released_states],
        events=[event for commit in commits for event in commit.events],
        completion_updates=[update for commit in commits for update in commit.completion_updates],
        visibility_updates=[update for commit in commits for update in commit.visibility_updates],
        output_state_bindings=[binding for commit in commits for binding in commit.output_state_bindings],
        warnings=_merge_warnings([], [warning for commit in commits for warning in commit.warnings]),
        ddr_transfer=_merge_ddr_transfers([commit.ddr_transfer for commit in commits]),
    )


def _merge_ddr_transfers(transfers: list[dict[str, Any]]) -> dict[str, Any]:
    if not transfers:
        return {}
    merged: dict[str, Any] = {
        "used": any(bool(item.get("used", False)) for item in transfers),
        "backend": next(
            (str(item.get("backend")) for item in transfers if item.get("backend")),
            "disabled",
        ),
    }
    sum_fields = {
        "transaction_count",
        "read_transactions",
        "write_transactions",
        "read_bytes",
        "write_bytes",
        "completed_read_transactions",
        "completed_write_transactions",
        "simulated_cycles",
        "energy_pj",
        "latency_s",
        "energy_j",
    }
    for field_name in sum_fields:
        value = sum(float(item.get(field_name, 0.0)) for item in transfers)
        if field_name.endswith(("count", "transactions", "bytes", "cycles")):
            value = int(value)
        merged[field_name] = value
    merged["power_w"] = (
        merged["energy_j"] / merged["latency_s"] if merged["latency_s"] > 0 else 0.0
    )
    return merged


def _safe_state_id(value: Any) -> str:
    return "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value)
    )


def _required_xbars_by_layer(plan) -> dict[str, int]:
    descriptors = list(getattr(plan, "layer_resource_descriptors", []) or [])
    required_by_layer: dict[str, int] = {}
    for descriptor in descriptors:
        layer_id = getattr(descriptor, "layer_id", None)
        required_xbar_count = getattr(descriptor, "required_xbar_count", None)
        if not isinstance(layer_id, str) or not layer_id:
            raise Phase2Invalid(
                PHYSICAL_XBAR_UNIT_MISMATCH,
                f"RoundPlan has descriptor with invalid layer_id {layer_id!r}",
            )
        if layer_id in required_by_layer:
            raise Phase2Invalid(
                PHYSICAL_XBAR_UNIT_MISMATCH,
                f"RoundPlan has duplicate required-xbar descriptors for {layer_id}",
            )
        if (
            isinstance(required_xbar_count, bool)
            or not isinstance(required_xbar_count, Integral)
            or required_xbar_count <= 0
        ):
            raise Phase2Invalid(
                PHYSICAL_XBAR_UNIT_MISMATCH,
                (
                    f"RoundPlan descriptor {layer_id} has invalid required physical "
                    f"xbar count {required_xbar_count!r}"
                ),
            )
        required_by_layer[layer_id] = int(required_xbar_count)

    for item in _execution_items_in_order(plan):
        for segment in canonical_ordered_segments(item):
            if not isinstance(segment, WeightedLayerSegment):
                continue
            layer_id = segment.selected_entry.layer_id
            if layer_id not in required_by_layer:
                raise Phase2Invalid(
                    PHYSICAL_XBAR_UNIT_MISMATCH,
                    f"RoundPlan has no required-xbar descriptor for weighted layer {layer_id}",
                )
    return required_by_layer


def _seed_model_inputs(boundary: BoundaryStateManager, model_inputs) -> None:
    if model_inputs is None:
        return
    if not isinstance(model_inputs, Mapping):
        raise Phase2Invalid(
            "INVALID_MODEL_INPUTS",
            "model_inputs must map state_id to input metadata",
        )
    for state_id, metadata in model_inputs.items():
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, Mapping):
            raise Phase2Invalid(
                "INVALID_MODEL_INPUTS",
                f"model input {state_id!r} metadata must be a mapping",
            )
        shape = metadata.get("shape", {})
        if not isinstance(shape, Mapping):
            raise Phase2Invalid(
                "INVALID_MODEL_INPUTS",
                f"model input {state_id!r} shape must be a mapping",
            )
        tensor_role = metadata.get("tensor_role", "activation")
        boundary.seed_model_input(state_id, shape=shape, tensor_role=tensor_role)


def _resolve_xbar_size(explicit_xbar_size, hardware_budget: Mapping[str, Any]) -> tuple[int, int]:
    value = explicit_xbar_size if explicit_xbar_size is not None else hardware_budget.get("xbar_size")
    if isinstance(value, int):
        return _positive_xbar_size(value, value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        return _positive_xbar_size(value[0], value[1])
    raise Phase2Invalid(
        "MISSING_XBAR_SIZE",
        "execute_round_plan requires xbar_size or plan.hardware_budget['xbar_size']",
    )


def _positive_xbar_size(row_count, column_count) -> tuple[int, int]:
    rows = int(row_count)
    columns = int(column_count)
    if rows <= 0 or columns <= 0:
        raise Phase2Invalid(
            "INVALID_XBAR_SIZE",
            f"xbar_size must be positive, got {(rows, columns)!r}",
        )
    return rows, columns


def _model_name(plan) -> str:
    return str(getattr(plan, "model_id", "") or getattr(plan, "plan_id", "unknown_model"))


def _merge_warnings(prefix_warnings, result_warnings) -> list[str]:
    merged: list[str] = []
    for warning in list(prefix_warnings or []) + list(result_warnings or []):
        if warning not in merged:
            merged.append(str(warning))
    return merged


__all__ = ["execute_round_plan"]
