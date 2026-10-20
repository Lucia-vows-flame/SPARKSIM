"""Aggregate ordered Phase 2 step results below physical execution rounds."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from MNSIM.Scheduling.phase2.types import (
    STATIC_STATUS_OK,
    BoundaryCommitResult,
    Phase2Invalid,
    Phase2Report,
    StepExecutionResult,
)


class ReportAggregator:
    """Build a model -> physical-round -> ordered-step report hierarchy."""

    def __init__(
        self,
        model_name: str,
        sim_config_path: str,
        ops_total: int | float | None,
        ops_count_policy: str,
        area_um2: float | None = None,
        accounting_scope: dict[str, Any] | None = None,
        static_power_ledger: Any | None = None,
    ) -> None:
        self.model_name = str(model_name)
        self.sim_config_path = str(sim_config_path)
        self.ops_total = ops_total
        self.ops_count_policy = str(ops_count_policy)
        self.explicit_area_um2 = area_um2
        self.accounting_scope = deepcopy(accounting_scope or _default_accounting_scope())
        self.static_power_ledger = static_power_ledger
        self._static_power_result: dict[str, Any] | None = None
        self._records: list[dict[str, Any]] = []
        self._step_results: list[dict[str, Any]] = []
        self._boundary_events: list[Any] = []
        self._raw_mnsim_audit: list[dict[str, Any]] = []
        self._warnings: list[str] = []
        self._assumptions: list[str] = []

    def record_round(
        self,
        item,
        placement,
        step_results: list[StepExecutionResult],
        boundary_commit_result: BoundaryCommitResult,
        write_programming: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record exactly one top-level entry for a finalized execution item."""
        item_id = str(getattr(item, "item_id", ""))
        if not item_id:
            raise Phase2Invalid("INVALID_REPORT_ITEM", "round item is missing item_id")
        self._validate_commit_alignment(item_id, boundary_commit_result)
        self._validate_status(boundary_commit_result.status, item_id)
        for result in step_results:
            self._validate_status(result.status, result.step_id)

        read_latency_s = sum(float(result.metrics.latency_s) for result in step_results)
        read_energy_j = sum(float(result.metrics.energy_j) for result in step_results)
        raw_read_latency_s = read_latency_s
        raw_read_energy_j = read_energy_j
        write_programming = deepcopy(write_programming or {})
        write_latency_s = float(write_programming.get("latency_s", 0.0))
        write_energy_j = float(write_programming.get("energy_j", 0.0))
        ddr_transfer = deepcopy(boundary_commit_result.ddr_transfer or {})
        ddr_latency_s = float(ddr_transfer.get("latency_s", 0.0))
        ddr_energy_j = float(ddr_transfer.get("energy_j", 0.0))
        latency_s = read_latency_s + write_latency_s + ddr_latency_s
        energy_j = read_energy_j + write_energy_j + ddr_energy_j
        active_areas = [float(result.metrics.area_um2) for result in step_results]
        if placement is not None:
            active_areas.append(float(getattr(placement, "mapped_area_um2", 0.0)))
        mapped_active_area_um2 = max(active_areas, default=0.0)
        raw_mapped_active_area_um2 = mapped_active_area_um2
        analog_frontend_lif = _analog_frontend_lif_terms(step_results)
        analog_mode = "adc_conversion" in self.accounting_scope
        adc_subtracted = _adc_terms(step_results) if analog_mode else _zero_adc_terms()
        if analog_mode:
            read_latency_s = _subtract_nonnegative(
                raw_read_latency_s,
                adc_subtracted["latency_s"],
                f"{item_id}: ADC latency",
            )
            read_energy_j = _subtract_nonnegative(
                raw_read_energy_j,
                adc_subtracted["energy_j"],
                f"{item_id}: ADC energy",
            )
            mapped_active_area_um2 = _subtract_nonnegative(
                raw_mapped_active_area_um2,
                adc_subtracted["area_um2"],
                f"{item_id}: ADC area",
            )
            latency_s = read_latency_s + write_latency_s + ddr_latency_s
            energy_j = read_energy_j + write_energy_j + ddr_energy_j
        step_reports = [_step_report(result, boundary_commit_result) for result in step_results]
        record = {
            "item_id": item_id,
            "item_kind": str(getattr(item, "item_kind", "")),
            "round_kind": getattr(item, "round_kind", None),
            "execution_index": int(getattr(item, "item_index", len(self._records))),
            "used_xbars": int(getattr(item.resource_proof, "used_xbars", 0)),
            "latency_s": latency_s,
            "energy_j": energy_j,
            "read_latency_s": read_latency_s + ddr_latency_s,
            "read_energy_j": read_energy_j + ddr_energy_j,
            "weight_programming_latency_s": write_latency_s,
            "weight_programming_energy_j": write_energy_j,
            "weight_programming": write_programming,
            "ddr_transfer": ddr_transfer,
            "ddr_boundary_transfer_latency_s": ddr_latency_s,
            "ddr_boundary_transfer_energy_j": ddr_energy_j,
            "power_avg_w": energy_j / latency_s if latency_s > 0 else 0.0,
            "area_um2": mapped_active_area_um2,
            "mapped_active_area_um2": mapped_active_area_um2,
            "raw_mnsim_baseline_latency_s": (
                raw_read_latency_s
                - analog_frontend_lif["latency_s"]
                + write_latency_s
                + ddr_latency_s
            ),
            "raw_mnsim_baseline_energy_j": (
                raw_read_energy_j
                - analog_frontend_lif["energy_j"]
                + write_energy_j
                + ddr_energy_j
            ),
            "raw_mnsim_mapped_active_area_um2": raw_mapped_active_area_um2,
            "adc_subtracted_latency_s": adc_subtracted["latency_s"],
            "adc_subtracted_energy_j": adc_subtracted["energy_j"],
            "adc_subtracted_power_w": adc_subtracted["power_w"],
            "adc_subtracted_area_um2": adc_subtracted["area_um2"],
            "analog_frontend_lif_latency_s": analog_frontend_lif["latency_s"],
            "analog_frontend_lif_energy_j": analog_frontend_lif["energy_j"],
            "analog_frontend_lif_power_w": analog_frontend_lif["power_w"],
            "placement_summary": _placement_summary(placement),
            "fragment_resource_proof": _fragment_resource_proof(item, placement),
            "steps": step_reports,
            "boundary_report": _boundary_report(boundary_commit_result),
        }
        if self.static_power_ledger is not None:
            self.static_power_ledger.observe_round(
                item_id=item_id,
                placement=placement,
                step_results=step_results,
                boundary_commit_result=boundary_commit_result,
            )
        self._records.append(record)
        self._step_results.extend(deepcopy(step_reports))
        self._boundary_events.extend(deepcopy(boundary_commit_result.events))
        if bool(ddr_transfer.get("used", False)):
            self.accounting_scope["ddr_boundary_transfer"] = {
                **dict(self.accounting_scope.get("ddr_boundary_transfer", {}) or {}),
                "used": True,
                "backend": str(ddr_transfer.get("backend", "dramsim3")),
                "energy_included": True,
                "latency_included": True,
                "accounting_owner": "BoundaryStateManager",
            }
        self._raw_mnsim_audit.extend(_round_audit_entries(placement, step_results))
        self._warnings.extend(_round_warnings(step_results, boundary_commit_result))
        return record

    def finalize(self) -> Phase2Report:
        if self.ops_total is None:
            raise Phase2Invalid(
                "MISSING_OPS_TOTAL",
                "ReportAggregator requires ops_total to compute TOPS/W",
            )

        if self.static_power_ledger is not None:
            self._static_power_result = self.static_power_ledger.finalize(self._records)
            self._apply_static_power_result(self._static_power_result)

        latency_s = sum(float(record["latency_s"]) for record in self._records)
        energy_j = sum(float(record["energy_j"]) for record in self._records)
        read_latency_s = sum(float(record["read_latency_s"]) for record in self._records)
        read_energy_j = sum(float(record["read_energy_j"]) for record in self._records)
        write_latency_s = sum(
            float(record["weight_programming_latency_s"]) for record in self._records
        )
        write_energy_j = sum(
            float(record["weight_programming_energy_j"]) for record in self._records
        )
        ddr_latency_s = sum(
            float(record.get("ddr_boundary_transfer_latency_s", 0.0))
            for record in self._records
        )
        ddr_energy_j = sum(
            float(record.get("ddr_boundary_transfer_energy_j", 0.0))
            for record in self._records
        )
        mapped_active_area_um2 = max(
            (float(record["mapped_active_area_um2"]) for record in self._records),
            default=0.0,
        )
        area_um2, area_assumption = self._final_area(mapped_active_area_um2)
        assumptions = list(self._assumptions)
        if area_assumption is not None:
            assumptions.append(area_assumption)

        warnings = list(self._warnings)
        if self._static_power_result is not None:
            warnings.extend(
                str(warning)
                for warning in self._static_power_result["static_power_audit"].get(
                    "warnings", []
                )
            )
        power_avg_w = energy_j / latency_s if latency_s > 0 else 0.0
        if latency_s <= 0:
            warnings.append("report latency is non-positive; TOPS set to 0.0")
        tops = float(self.ops_total) / latency_s / 1e12 if latency_s > 0 else 0.0
        if energy_j <= 0:
            warnings.append("report energy is non-positive; TOPS/W set to 0.0")
        modeled_tops_per_w = (
            float(self.ops_total) / energy_j / 1e12 if energy_j > 0 else 0.0
        )
        energy_accounting_complete = _energy_accounting_complete(self.accounting_scope)
        analog_lif = self._analog_lif_summary()
        analog_accounting = self._analog_accounting_summary() if analog_lif else {}

        inference_metrics = {
            "latency_s": latency_s,
            "energy_j": energy_j,
            "read_latency_s": read_latency_s,
            "read_energy_j": read_energy_j,
            "weight_programming_latency_s": write_latency_s,
            "weight_programming_energy_j": write_energy_j,
            "ddr_boundary_transfer_latency_s": ddr_latency_s,
            "ddr_boundary_transfer_energy_j": ddr_energy_j,
            "power_avg_w": power_avg_w,
            "area_um2": area_um2,
            "mapped_active_area_um2": mapped_active_area_um2,
            "ops_total": self.ops_total,
            "tops": tops,
            "tops_per_w": modeled_tops_per_w if energy_accounting_complete else None,
            "modeled_energy_efficiency_tops_per_w": modeled_tops_per_w,
            "energy_accounting_complete": energy_accounting_complete,
            "ops_count_policy": self.ops_count_policy,
            "cumulative_scheduled_physical_xbars": sum(
                int(record["used_xbars"]) for record in self._records
            ),
            "peak_scheduled_physical_xbars": max(
                (int(record["used_xbars"]) for record in self._records),
                default=0,
            ),
            "energy_breakdown_j": self._energy_breakdown(),
            "accounting_scope": deepcopy(self.accounting_scope),
            "assumptions": _dedupe_preserve_order(assumptions),
            "lif_update_count": sum(
                int(step.get("mapping_summary", {})
                    .get("digital_lif", {})
                    .get("neuron_count", 0))
                for step in self._step_results
                if step.get("source_kind") == "digital_lif"
            )
            + int(analog_lif.get("query_counts", {}).get("lif", 0)),
        }
        if analog_lif:
            inference_metrics["analog_lif"] = analog_lif
            inference_metrics["analog_accounting"] = analog_accounting
        # Always expose one circuit ledger.  Earlier reports only emitted the
        # ledger when the optional SNN-peripheral profile was enabled, which
        # made an ordinary run look closed even though the static audit still
        # had uncharacterized circuit blocks.  The ledger now carries those
        # gaps explicitly and keeps the total unchanged until a profile closes
        # them.
        inference_metrics["energy_ledger"] = self._circuit_energy_ledger()
        if self._static_power_result is not None:
            static_audit = deepcopy(self._static_power_result["static_power_audit"])
            inference_metrics.update(
                {
                    "static_energy_j": float(
                        self._static_power_result["static_energy_j"]
                    ),
                    "static_average_power_w": float(
                        self._static_power_result["static_average_power_w"]
                    ),
                    "static_accounting_complete": bool(
                        static_audit["static_accounting_complete"]
                    ),
                    "implementation_specific_static_complete": bool(
                        static_audit["implementation_specific_complete"]
                    ),
                    "static_power_audit": static_audit,
                    "boundary_sram_access_energy_j": float(
                        self._static_power_result.get(
                            "boundary_sram_access_energy_j", 0.0
                        )
                    ),
                    "boundary_sram_access_latency_s": float(
                        self._static_power_result.get(
                            "boundary_sram_access_latency_s", 0.0
                        )
                    ),
                }
            )
        return Phase2Report(
            model_name=self.model_name,
            sim_config_path=self.sim_config_path,
            feasible=True,
            execution_item_count=len(self._records),
            inference_metrics=inference_metrics,
            item_results=deepcopy(self._records),
            step_results=deepcopy(self._step_results),
            raw_mnsim_audit=deepcopy(self._raw_mnsim_audit),
            boundary_events=deepcopy(self._boundary_events),
            warnings=_dedupe_preserve_order(warnings),
            assumptions=_dedupe_preserve_order(assumptions),
        )

    def _apply_static_power_result(self, result: dict[str, Any]) -> None:
        by_round = dict(result.get("static_energy_by_round_j", {}) or {})
        boundary_energy_by_round = dict(
            result.get("boundary_sram_access_energy_by_round_j", {}) or {}
        )
        boundary_latency_by_round = dict(
            result.get("boundary_sram_access_latency_by_round_s", {}) or {}
        )
        assigned_static = 0.0
        assigned_boundary_energy = 0.0
        assigned_boundary_latency = 0.0
        for record in self._records:
            item_id = str(record["item_id"])
            boundary_energy_j = float(boundary_energy_by_round.get(item_id, 0.0))
            boundary_latency_s = float(
                boundary_latency_by_round.get(item_id, 0.0)
            )
            for label, value in (
                ("boundary SRAM access energy", boundary_energy_j),
                ("boundary SRAM access latency", boundary_latency_s),
            ):
                if not math.isfinite(value) or value < 0:
                    raise Phase2Invalid(
                        "INVALID_STATIC_POWER_RESULT",
                        f"{item_id} has invalid {label} {value!r}",
                    )
            static_energy_j = float(by_round.get(item_id, 0.0))
            if not math.isfinite(static_energy_j) or static_energy_j < 0:
                raise Phase2Invalid(
                    "INVALID_STATIC_POWER_RESULT",
                    f"{item_id} has invalid static energy {static_energy_j!r}",
                )
            record["boundary_sram_access_energy_j"] = boundary_energy_j
            record["boundary_sram_access_latency_s"] = boundary_latency_s
            record["read_energy_j"] = (
                float(record["read_energy_j"]) + boundary_energy_j
            )
            record["read_latency_s"] = (
                float(record["read_latency_s"]) + boundary_latency_s
            )
            record["latency_s"] = float(record["latency_s"]) + boundary_latency_s
            record["raw_mnsim_baseline_latency_s"] = (
                float(record["raw_mnsim_baseline_latency_s"])
                + boundary_latency_s
            )
            record["static_energy_j"] = static_energy_j
            record["energy_j"] = (
                float(record["energy_j"])
                + boundary_energy_j
                + static_energy_j
            )
            record["raw_mnsim_baseline_energy_j"] = (
                float(record["raw_mnsim_baseline_energy_j"])
                + boundary_energy_j
                + static_energy_j
            )
            record["power_avg_w"] = (
                float(record["energy_j"]) / float(record["latency_s"])
                if float(record["latency_s"]) > 0
                else 0.0
            )
            assigned_static += static_energy_j
            assigned_boundary_energy += boundary_energy_j
            assigned_boundary_latency += boundary_latency_s
        expected = float(result.get("static_energy_j", 0.0))
        if not math.isclose(
            assigned_static, expected, rel_tol=1e-12, abs_tol=1e-30
        ):
            raise Phase2Invalid(
                "STATIC_POWER_ENERGY_MISMATCH",
                f"per-round static energy {assigned_static} does not close to {expected}",
            )
        expected_boundary_energy = float(
            result.get("boundary_sram_access_energy_j", 0.0)
        )
        expected_boundary_latency = float(
            result.get("boundary_sram_access_latency_s", 0.0)
        )
        if not math.isclose(
            assigned_boundary_energy,
            expected_boundary_energy,
            rel_tol=1e-12,
            abs_tol=1e-30,
        ):
            raise Phase2Invalid(
                "BOUNDARY_SRAM_ENERGY_MISMATCH",
                "per-round boundary SRAM access energy does not close",
            )
        if not math.isclose(
            assigned_boundary_latency,
            expected_boundary_latency,
            rel_tol=1e-12,
            abs_tol=1e-30,
        ):
            raise Phase2Invalid(
                "BOUNDARY_SRAM_LATENCY_MISMATCH",
                "per-round boundary SRAM access latency does not close",
            )
        audit = dict(result.get("static_power_audit", {}) or {})
        unmodeled = set(audit.get("unmodeled_blocks", []) or [])
        modeled = set(audit.get("modeled_blocks", []) or [])
        sram_blocks = {
            "pe_input_sram",
            "tile_state_output_sram",
            "global_accumulator_sram",
            "boundary_state_sram",
        }
        static_scope = self.accounting_scope.setdefault("static_leakage", {})
        static_scope["central_sram_retention_energy_included"] = bool(
            modeled.intersection(sram_blocks)
        )
        static_scope["buffer_energy_included"] = not bool(
            unmodeled.intersection(sram_blocks)
        )
        boundary_scope = self.accounting_scope.setdefault(
            "cross_round_boundary_sram", {}
        )
        cross_round_audit = dict(audit.get("cross_round_storage", {}) or {})
        boundary_observed = bool(cross_round_audit.get("observed", False))
        boundary_scope.update(
            {
                "capacity_included": "boundary_state_sram" in modeled,
                "retention_energy_included": "boundary_state_sram" in modeled,
                "dynamic_access_energy_included": boundary_observed,
                "latency_included": boundary_observed,
                "spill_to_ddr": False,
            }
        )

    def _final_area(self, mapped_active_area_um2: float) -> tuple[float, str | None]:
        if self.explicit_area_um2 is not None:
            return float(self.explicit_area_um2), None
        return (
            float(mapped_active_area_um2),
            "area_um2 uses maximum mapped active area as the v1 fallback.",
        )

    def _analog_accounting_summary(self) -> dict[str, Any]:
        raw_latency_s = sum(
            float(record["raw_mnsim_baseline_latency_s"])
            for record in self._records
        )
        raw_energy_j = sum(
            float(record["raw_mnsim_baseline_energy_j"])
            for record in self._records
        )
        raw_area_um2 = max(
            (
                float(record["raw_mnsim_mapped_active_area_um2"])
                for record in self._records
            ),
            default=0.0,
        )
        adc_latency_s = sum(
            float(record["adc_subtracted_latency_s"]) for record in self._records
        )
        adc_energy_j = sum(
            float(record["adc_subtracted_energy_j"]) for record in self._records
        )
        adc_power_w = sum(
            float(record["adc_subtracted_power_w"]) for record in self._records
        )
        raw_adc_area_um2 = max(
            (float(record["adc_subtracted_area_um2"]) for record in self._records),
            default=0.0,
        )
        frontend_lif_latency_s = sum(
            float(record["analog_frontend_lif_latency_s"])
            for record in self._records
        )
        frontend_lif_energy_j = sum(
            float(record["analog_frontend_lif_energy_j"])
            for record in self._records
        )
        final_latency_s = sum(float(record["latency_s"]) for record in self._records)
        final_energy_j = sum(float(record["energy_j"]) for record in self._records)
        final_area_um2 = max(
            (float(record["mapped_active_area_um2"]) for record in self._records),
            default=0.0,
        )
        adc_area_um2 = raw_area_um2 - final_area_um2
        reconciliation = {
            "latency_s": raw_latency_s - adc_latency_s + frontend_lif_latency_s,
            "energy_j": raw_energy_j - adc_energy_j + frontend_lif_energy_j,
            "area_um2": raw_area_um2 - adc_area_um2,
        }
        passed = (
            math.isclose(final_latency_s, reconciliation["latency_s"], rel_tol=1e-9, abs_tol=1e-18)
            and math.isclose(final_energy_j, reconciliation["energy_j"], rel_tol=1e-9, abs_tol=1e-18)
            and math.isclose(final_area_um2, reconciliation["area_um2"], rel_tol=1e-9, abs_tol=1e-12)
        )
        if not passed:
            raise Phase2Invalid(
                "ANALOG_ACCOUNTING_RECONCILIATION_FAILED",
                "final analog totals do not reconcile with raw totals minus ADC",
            )
        return {
            "raw_mnsim_baseline_latency_s": raw_latency_s,
            "raw_mnsim_baseline_energy_j": raw_energy_j,
            "raw_mnsim_mapped_active_area_um2": raw_area_um2,
            "adc_subtracted_latency_s": adc_latency_s,
            "adc_subtracted_energy_j": adc_energy_j,
            "adc_subtracted_power_w": adc_power_w,
            "adc_subtracted_area_um2": adc_area_um2,
            "raw_adc_area_um2": raw_adc_area_um2,
            "analog_frontend_lif_latency_s": frontend_lif_latency_s,
            "analog_frontend_lif_energy_j": frontend_lif_energy_j,
            "analog_frontend_lif_power_w": (
                frontend_lif_energy_j / frontend_lif_latency_s
                if frontend_lif_latency_s > 0
                else 0.0
            ),
            "final_latency_s": final_latency_s,
            "final_energy_j": final_energy_j,
            "final_area_um2": final_area_um2,
            "reconciliation": reconciliation,
            "reconciliation_passed": True,
        }

    def _supplemented_energy_ledger(self) -> dict[str, Any]:
        """Reconcile original billed components and the additive supplement."""
        breakdown = self._energy_breakdown()
        included = {}
        provenance = {}
        modules = breakdown["mnsim_static_module_energy_j"]
        for name, value in modules.items():
            key = "xbar_read" if name == "xbar" else f"mnsim_{name}"
            included[key] = value - (breakdown["adc_subtracted_energy_j"] if name == "adc" else 0.0)
            provenance[key] = {"source": "original_MNSIM_module_ledger", "module": name, "policy": "existing_accounting_unchanged"}
        residual = breakdown["mnsim_static_module_energy_residual_j"]
        if "adc" not in modules:
            residual -= breakdown["adc_subtracted_energy_j"]
        included["mnsim_unclassified_read"] = residual
        provenance["mnsim_unclassified_read"] = {"source": "original_static_read_minus_reported_modules", "policy": "unclassified_cost_preserved"}
        original = {
            "hook": breakdown["hook_step_energy_j"],
            "digital_lif": breakdown["digital_lif_energy_j"],
            "partial_sum_arithmetic": breakdown["scheduled_partial_accumulator_arithmetic_energy_j"],
            "partial_state_buffer_read": breakdown["partial_state_buffer_read_energy_j"],
            "partial_state_buffer_write": breakdown["partial_state_buffer_write_energy_j"],
            "boundary_sram_access": breakdown["boundary_sram_access_energy_j"],
            "sram_retention": breakdown["static_energy_j"],
            "ddr_transfer": breakdown["ddr_boundary_transfer_energy_j"],
            "weight_programming": breakdown["weight_programming_energy_j"],
        }
        analog = self._analog_lif_summary()
        original["analog_frontend"] = float(analog.get("frontend_energy_j", 0.0))
        original["analog_lif"] = float(analog.get("lif_energy_j", 0.0))
        for key, value in original.items():
            included[key] = value
            provenance[key] = {"source": "existing_phase2_accounting", "policy": "unchanged; analog frontend and LIF retain existing LUT costs"}
        known_sources = {"static_eval", "hook", "digital_lif", "analog_lif", "scheduled_partial_accumulator", "snn_peripheral_energy"}
        included["other_existing_steps"] = sum(float(s["energy_j"]) for s in self._step_results if s["source_kind"] not in known_sources)
        provenance["other_existing_steps"] = {"source": "existing_steps_outside_named_categories"}
        supplements = []
        for step in self._step_results:
            if step["source_kind"] != "snn_peripheral_energy":
                continue
            audit = step["mapping_summary"]["snn_peripheral_energy"]
            if not math.isclose(float(step["energy_j"]), math.fsum(audit["included_components_j"].values()), rel_tol=1e-12, abs_tol=1e-30):
                raise Phase2Invalid("SNN_PERIPHERAL_RECONCILIATION_FAILED", step["step_id"])
            supplements.append(audit)
            for name, value in audit["included_components_j"].items():
                included[name] = included.get(name, 0.0) + value
                provenance[name] = audit["component_provenance"][name]
        total = sum(float(r["energy_j"]) for r in self._records)
        component_total = math.fsum(included.values())
        if not math.isclose(total, component_total, rel_tol=1e-9, abs_tol=1e-24):
            raise Phase2Invalid("SNN_PERIPHERAL_RECONCILIATION_FAILED", f"total {total} differs from component sum {component_total}")
        return {
            "schema_id": "snn_supplemented_energy_ledger/v1",
            "included_components_j": included,
            "total_energy_j": total,
            "component_provenance": provenance,
            "supplement_profile": {k: supplements[0][k] for k in ("profile_id", "profile_status", "profile_path", "profile_sha256", "ownership")} if supplements else {},
            "additional_model_external_energy_read": False,
            "existing_frontend_lif_source": "existing_lut_unchanged",
            "boundary": "existing billed energy plus additional peripherals; includes weight programming; frontend/LIF signal and cost unchanged; additional model does not imply complete whole-chip accounting",
            "audit": {"component_sum_j": component_total, "residual_j": total - component_total, "reconciliation_passed": True, "supplement_event_count": len(supplements)},
        }

    def _circuit_energy_ledger(self) -> dict[str, Any]:
        """Return a stable circuit-cost ledger for every Phase 2 run.

        Existing MNSIM owners remain authoritative for their dynamic costs.
        Optional SNN peripherals are merged when present; static blocks that
        lack a characterized profile are reported as unresolved diagnostics,
        never silently inserted as zero-energy components.
        """
        ledger = self._supplemented_energy_ledger()
        static_audit = self._static_power_result.get("static_power_audit", {}) if self._static_power_result else {}
        # This ledger intentionally covers deployment plus inference hardware
        # cost.  Whole-chip standby blocks are outside that agreed metric.
        unresolved = []
        if not self.accounting_scope.get("snn_additional_peripherals", {}).get("energy_included"):
            unresolved.extend([
                "read_supply", "column_reference", "arow_row_driver",
                "arow_startup", "spike_interconnect", "interface",
            ])
        unresolved = sorted(set(unresolved))
        ledger.update({
            "schema_id": "mnsim_circuit_energy_ledger/v1",
            "included_components_j": dict(ledger.get("included_components_j", {})),
            "excluded_components_j": {},
            "diagnostics_j": {
                "unresolved_components": unresolved,
                "static_unmodeled_blocks_outside_metric": sorted(set(static_audit.get("unmodeled_blocks", []) or [])),
                "external_hspice_energy_read": False,
                "component_sum_j": float(ledger.get("audit", {}).get("component_sum_j", ledger.get("total_energy_j", 0.0))),
                "reconciliation_passed": bool(ledger.get("audit", {}).get("reconciliation_passed", False)),
            },
            "accounting_complete": not unresolved and bool(ledger.get("audit", {}).get("reconciliation_passed", False)),
            "boundary": "physical RRAM weight programming + MNSIM inference hardware + enabled activity peripherals; standby and non-SRAM static power are outside this metric",
            "version": "mnsim_circuit_energy_ledger/v1",
        })
        return ledger

    def _energy_breakdown(self) -> dict[str, Any]:
        peripheral_energy_j = sum(
            float(step["energy_j"]) for step in self._step_results
            if step["source_kind"] == "snn_peripheral_energy"
        )
        static_step_energy_j = sum(
            float(step["energy_j"])
            for step in self._step_results
            if step["source_kind"] == "static_eval"
        )
        hook_step_energy_j = sum(
            float(step["energy_j"])
            for step in self._step_results
            if step["source_kind"] == "hook"
        )
        digital_lif_energy_j = sum(
            float(step["energy_j"])
            for step in self._step_results
            if step["source_kind"] == "digital_lif"
        )
        analog_lif_energy_j = sum(
            float(step["energy_j"])
            for step in self._step_results
            if step["source_kind"] == "analog_lif"
        )
        partial_accumulator_energy_j = sum(
            float(step["energy_j"])
            for step in self._step_results
            if step["source_kind"] == "scheduled_partial_accumulator"
        )
        partial_accumulator_latency_s = sum(
            float(step["latency_s"])
            for step in self._step_results
            if step["source_kind"] == "scheduled_partial_accumulator"
        )
        partial_accumulator_audits = [
            dict(step.get("mapping_summary", {}))
            .get("scheduled_partial_accumulator", {})
            for step in self._step_results
            if step["source_kind"] == "scheduled_partial_accumulator"
        ]
        partial_accumulator_arithmetic_energy_j = sum(
            float(audit.get("arithmetic_energy_j", 0.0))
            for audit in partial_accumulator_audits
        )
        partial_state_buffer_read_energy_j = sum(
            float(audit.get("state_read_energy_j", 0.0))
            for audit in partial_accumulator_audits
        )
        partial_state_buffer_write_energy_j = sum(
            float(audit.get("state_write_energy_j", 0.0))
            for audit in partial_accumulator_audits
        )
        partial_state_buffer_read_bytes = sum(
            int(audit.get("state_read_bytes", 0))
            for audit in partial_accumulator_audits
        )
        partial_state_buffer_write_bytes = sum(
            int(audit.get("state_write_bytes", 0))
            for audit in partial_accumulator_audits
        )
        mnsim_static_module_energy_j: dict[str, float] = {}
        for raw_entry in self._raw_mnsim_audit:
            if raw_entry.get("audit_kind") != "static":
                continue
            raw_summary = dict(raw_entry.get("raw_mnsim_summary", {}) or {})
            module_energy = raw_summary.get("reportable_module_energy_j", {})
            if not isinstance(module_energy, dict):
                continue
            for name, value in module_energy.items():
                mnsim_static_module_energy_j[str(name)] = (
                    mnsim_static_module_energy_j.get(str(name), 0.0)
                    + float(value)
                )
        classified_mnsim_static_energy_j = sum(
            mnsim_static_module_energy_j.values()
        )
        static_power_result = self._static_power_result or {}
        adc_subtracted_energy_j = sum(
            float(record.get("adc_subtracted_energy_j", 0.0))
            for record in self._records
        )
        boundary_sram_access_energy_j = float(
            static_power_result.get("boundary_sram_access_energy_j", 0.0)
        )
        boundary_sram_access_latency_s = float(
            static_power_result.get("boundary_sram_access_latency_s", 0.0)
        )
        ddr_boundary_transfer_energy_j = sum(
            float(record.get("ddr_boundary_transfer_energy_j", 0.0))
            for record in self._records
        )
        ddr_boundary_transfer_latency_s = sum(
            float(record.get("ddr_boundary_transfer_latency_s", 0.0))
            for record in self._records
        )
        return {
            # Retained for schema compatibility. This is original MNSIM read-path
            # dynamic energy, not whole-chip static leakage.
            "static_step_energy_j": static_step_energy_j,
            "mnsim_static_read_energy_j": static_step_energy_j,
            "mnsim_static_module_energy_j": mnsim_static_module_energy_j,
            "mnsim_static_module_energy_total_j": classified_mnsim_static_energy_j,
            "mnsim_static_module_energy_residual_j": (
                static_step_energy_j - classified_mnsim_static_energy_j
            ),
            "hook_step_energy_j": hook_step_energy_j,
            "digital_lif_energy_j": digital_lif_energy_j,
            "analog_lif_energy_j": analog_lif_energy_j,
            "snn_additional_peripheral_energy_j": peripheral_energy_j,
            "scheduled_partial_accumulator_energy_j": partial_accumulator_energy_j,
            "scheduled_partial_accumulator_latency_s": partial_accumulator_latency_s,
            "scheduled_partial_accumulator_arithmetic_energy_j": (
                partial_accumulator_arithmetic_energy_j
            ),
            "partial_state_buffer_read_energy_j": partial_state_buffer_read_energy_j,
            "partial_state_buffer_write_energy_j": partial_state_buffer_write_energy_j,
            "partial_state_buffer_read_bytes": partial_state_buffer_read_bytes,
            "partial_state_buffer_write_bytes": partial_state_buffer_write_bytes,
            "boundary_sram_access_energy_j": boundary_sram_access_energy_j,
            "boundary_sram_access_latency_s": boundary_sram_access_latency_s,
            "ddr_boundary_transfer_energy_j": ddr_boundary_transfer_energy_j,
            "ddr_boundary_transfer_latency_s": ddr_boundary_transfer_latency_s,
            "boundary_sram_access_energy_by_round_j": deepcopy(
                static_power_result.get(
                    "boundary_sram_access_energy_by_round_j", {}
                )
            ),
            "boundary_sram_access_latency_by_round_s": deepcopy(
                static_power_result.get(
                    "boundary_sram_access_latency_by_round_s", {}
                )
            ),
            "static_energy_j": float(static_power_result.get("static_energy_j", 0.0)),
            "static_average_power_w": float(
                static_power_result.get("static_average_power_w", 0.0)
            ),
            "static_energy_by_block_j": deepcopy(
                static_power_result.get("static_energy_by_block_j", {})
            ),
            "static_energy_by_state_j": deepcopy(
                static_power_result.get("static_energy_by_state_j", {})
            ),
            "static_energy_by_round_j": deepcopy(
                static_power_result.get("static_energy_by_round_j", {})
            ),
            "read_energy_j": (
                static_step_energy_j
                - adc_subtracted_energy_j
                + hook_step_energy_j
                + digital_lif_energy_j
                + analog_lif_energy_j
                + peripheral_energy_j
                + partial_accumulator_energy_j
                + boundary_sram_access_energy_j
                + ddr_boundary_transfer_energy_j
            ),
            "adc_subtracted_energy_j": adc_subtracted_energy_j,
            "weight_programming_energy_j": sum(
                float(record["weight_programming_energy_j"])
                for record in self._records
            ),
            "total_energy_j": sum(
                float(record["energy_j"]) for record in self._records
            ),
            "per_round": [
                {
                    "item_id": record["item_id"],
                    "energy_j": record["energy_j"],
                    "read_energy_j": record["read_energy_j"],
                    "weight_programming_energy_j": record[
                        "weight_programming_energy_j"
                    ],
                }
                for record in self._records
            ],
        }

    def _analog_lif_summary(self) -> dict[str, Any]:
        audits = [
            dict(step.get("mapping_summary", {})).get("analog_lif", {})
            for step in self._step_results
            if step.get("source_kind") == "analog_lif"
        ]
        if not audits:
            return {}
        identity_fields = (
            "lut_schema_id",
            "lut_profile",
            "lut_root",
            "interpolation",
            "negative_vmem_policy",
        )
        for field in identity_fields:
            values = {str(audit.get(field, "")) for audit in audits}
            if len(values) != 1:
                raise Phase2Invalid(
                    "INCONSISTENT_ANALOG_LIF_AUDIT",
                    f"analog LIF invocations disagree on {field}: {sorted(values)!r}",
                )
        query_counts = {
            name: sum(int(audit.get("query_counts", {}).get(name, 0)) for audit in audits)
            for name in ("frontend", "lif")
        }
        latency_s = sum(float(audit.get("latency_s", 0.0)) for audit in audits)
        energy_j = sum(float(audit.get("energy_j", 0.0)) for audit in audits)
        return {
            "backend": "local_analog_lif_lut",
            **{field: audits[0].get(field) for field in identity_fields},
            "checksum_verified": all(
                bool(audit.get("checksum_verified", False)) for audit in audits
            ),
            "invocation_count": len(audits),
            "query_counts": query_counts,
            "axis_clamp_count": sum(
                int(audit.get("axis_clamp_count", 0)) for audit in audits
            ),
            "axis_reject_count": sum(
                int(audit.get("axis_reject_count", 0)) for audit in audits
            ),
            "invalid_neighbor_reject_count": sum(
                int(audit.get("invalid_neighbor_reject_count", 0)) for audit in audits
            ),
            "negative_vmem_clamp_count": sum(
                int(audit.get("negative_vmem_clamp_count", 0)) for audit in audits
            ),
            "raw_min_clamped_vmem_next_v": min(
                (
                    float(audit["raw_min_clamped_vmem_next_v"])
                    for audit in audits
                    if audit.get("raw_min_clamped_vmem_next_v") is not None
                ),
                default=None,
            ),
            "frontend_latency_s": sum(
                float(audit.get("frontend_latency_s", 0.0)) for audit in audits
            ),
            "frontend_energy_j": sum(
                float(audit.get("frontend_energy_j", 0.0)) for audit in audits
            ),
            "lif_latency_s": sum(
                float(audit.get("lif_latency_s", 0.0)) for audit in audits
            ),
            "lif_energy_j": sum(
                float(audit.get("lif_energy_j", 0.0)) for audit in audits
            ),
            "latency_s": latency_s,
            "energy_j": energy_j,
            "power_w": energy_j / latency_s if latency_s > 0 else 0.0,
            "cost_source": "lut_only",
            "adc_hardware_cost_disabled": False,
            "adc_raw_cost_retained": True,
            "adc_final_cost_subtracted": True,
            "virtual_output_lane_geometry_preserved": True,
            "area_characterized": all(bool(audit.get("area_characterized", False)) for audit in audits),
            "area_um2": max((float(audit.get("area_um2", 0.0)) for audit in audits), default=0.0),
            "area_status": "hspice_netlist_estimate",
            "state_payload_identities": [
                audit.get("state", {}).get("payload_identity") for audit in audits
            ],
        }

    @staticmethod
    def _validate_status(status: str, subject_id: str) -> None:
        if status != STATIC_STATUS_OK:
            raise Phase2Invalid(
                "STATIC_EVAL_NOT_OK",
                f"{subject_id} has status {status!r}",
            )

    @staticmethod
    def _validate_commit_alignment(
        item_id: str,
        boundary_commit_result: BoundaryCommitResult,
    ) -> None:
        if str(boundary_commit_result.item_id) != item_id:
            raise Phase2Invalid(
                "BOUNDARY_COMMIT_MISMATCH",
                (
                    f"boundary commit item_id {boundary_commit_result.item_id!r} "
                    f"does not match {item_id!r}"
                ),
            )


def _placement_summary(placement) -> dict[str, Any] | None:
    if placement is None:
        return None
    return {
        "item_id": str(getattr(placement, "item_id", "")),
        "mapped_area_um2": float(getattr(placement, "mapped_area_um2", 0.0)),
        "mapping_summary": deepcopy(getattr(placement, "mapping_summary", {}) or {}),
    }


def _fragment_resource_proof(item, placement) -> dict[str, int] | None:
    """Expose the validated physical resource proof independently of area."""
    if placement is None:
        return None
    mapping_summary = getattr(placement, "mapping_summary", {}) or {}
    if not {
        "active_group_count",
        "active_physical_xbar_count",
    }.issubset(mapping_summary):
        return None
    return {
        "used_xbars": int(getattr(item.resource_proof, "used_xbars", 0)),
        "active_group_count": int(mapping_summary["active_group_count"]),
        "active_physical_xbar_count": int(
            mapping_summary["active_physical_xbar_count"]
        ),
    }


def _step_report(
    result: StepExecutionResult,
    boundary_commit_result: BoundaryCommitResult,
) -> dict[str, Any]:
    completion_by_origin = {
        str(update.get("origin_layer_id")): update
        for update in boundary_commit_result.completion_updates
    }
    layer_reports = []
    for layer in result.layer_results:
        report = layer.to_dict()
        completion = completion_by_origin.get(str(layer.origin_layer_id))
        if completion is not None:
            report["completion_status"] = completion["completion_status"]
            report["boundary_state_id"] = completion["state_id"]
        layer_reports.append(report)

    return {
        "item_id": result.item_id,
        "step_id": result.step_id,
        "segment_index": result.segment_index,
        "step_kind": result.step_kind,
        "source_kind": result.source_kind,
        "status": result.status,
        "origin_layer_ids": list(result.origin_layer_ids),
        "latency_s": float(result.metrics.latency_s),
        "energy_j": float(result.metrics.energy_j),
        "power_w": float(result.metrics.power_w),
        "area_um2": float(result.metrics.area_um2),
        "mapping_summary": deepcopy(result.mapping_summary),
        "layer_reports": layer_reports,
        "warnings": list(result.warnings),
        "sample_id": result.sample_id,
        "timestep": result.timestep,
    }


def _boundary_report(boundary_commit_result: BoundaryCommitResult) -> dict[str, Any]:
    return {
        "created_states": [
            state.to_dict() for state in boundary_commit_result.created_states
        ],
        "updated_states": [
            state.to_dict() for state in boundary_commit_result.updated_states
        ],
        "completion_updates": deepcopy(boundary_commit_result.completion_updates),
        "visibility_updates": deepcopy(boundary_commit_result.visibility_updates),
        "output_state_bindings": deepcopy(boundary_commit_result.output_state_bindings),
        "events": deepcopy(boundary_commit_result.events),
        "ddr_transfer": deepcopy(boundary_commit_result.ddr_transfer),
    }


def _round_audit_entries(placement, step_results: list[StepExecutionResult]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if placement is not None:
        entries.append(
            {
                "audit_kind": "placement",
                "item_id": str(getattr(placement, "item_id", "")),
                "mapped_area_um2": float(getattr(placement, "mapped_area_um2", 0.0)),
                "mapping_summary": deepcopy(getattr(placement, "mapping_summary", {}) or {}),
            }
        )
    for result in step_results:
        for raw_entry in result.raw_mnsim_audit:
            entry = deepcopy(raw_entry)
            entry.setdefault("audit_kind", result.source_kind)
            entry.setdefault("item_id", result.item_id)
            entry.setdefault("step_id", result.step_id)
            entries.append(entry)
    return entries


def _round_warnings(
    step_results: list[StepExecutionResult],
    boundary_commit_result: BoundaryCommitResult,
) -> list[str]:
    return _dedupe_preserve_order(
        [
            str(warning)
            for result in step_results
            for warning in result.warnings
        ]
        + [str(warning) for warning in boundary_commit_result.warnings]
    )


def _zero_adc_terms() -> dict[str, float]:
    return {"latency_s": 0.0, "energy_j": 0.0, "power_w": 0.0, "area_um2": 0.0}


def _analog_frontend_lif_terms(step_results: list[StepExecutionResult]) -> dict[str, float]:
    latency_s = energy_j = 0.0
    for result in step_results:
        if result.source_kind != "analog_lif":
            continue
        audit = dict(result.mapping_summary.get("analog_lif", {}) or {})
        latency_s += float(audit.get("latency_s", result.metrics.latency_s))
        energy_j += float(audit.get("energy_j", result.metrics.energy_j))
    return {
        "latency_s": latency_s,
        "energy_j": energy_j,
        "power_w": energy_j / latency_s if latency_s > 0 else 0.0,
    }


def _adc_terms(step_results: list[StepExecutionResult]) -> dict[str, float]:
    totals = _zero_adc_terms()
    for result in step_results:
        if result.source_kind != "static_eval":
            continue
        for audit in result.raw_mnsim_audit:
            summary = dict(audit.get("raw_mnsim_summary", {}) or {})
            totals["latency_s"] += float(
                summary.get("reportable_adc_latency_s", summary.get("raw_adc_latency_s", 0.0))
            )
            totals["energy_j"] += float(
                summary.get("reportable_adc_energy_j", summary.get("raw_adc_energy_j", 0.0))
            )
            totals["power_w"] += float(
                summary.get("reportable_adc_power_w", summary.get("raw_adc_power_w", 0.0))
            )
            totals["area_um2"] += float(
                summary.get("reportable_adc_area_um2", summary.get("raw_adc_area_um2", 0.0))
            )
    return totals


def _subtract_nonnegative(value: float, amount: float, label: str) -> float:
    remaining = float(value) - float(amount)
    if remaining < -1e-18:
        raise Phase2Invalid(
            "ANALOG_ACCOUNTING_RECONCILIATION_FAILED",
            f"{label} exceeds the raw total ({amount} > {value})",
        )
    return max(0.0, remaining)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _default_accounting_scope() -> dict[str, Any]:
    """Compatibility descriptor for direct ReportAggregator users."""

    return {
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
        "static_leakage": {
            "buffer_energy_included": True,
            "other_energy_included": False,
        },
        "scheduled_partial_accumulator": {
            "energy_included": True,
            "latency_included": True,
            "partial_state_buffer_included": True,
            "primitive": "MNSIM.Hardware_Model.Adder+Buffer",
        },
    }


def _energy_accounting_complete(accounting_scope: dict[str, Any]) -> bool:
    programming = dict(accounting_scope.get("weight_programming", {}) or {})
    static = dict(accounting_scope.get("static_leakage", {}) or {})
    partial = dict(accounting_scope.get("scheduled_partial_accumulator", {}) or {})
    peripherals = accounting_scope.get("snn_additional_peripherals")
    # A report without an explicitly selected peripheral profile is still
    # useful for legacy MNSIM runs, but it must not claim the SNN circuit
    # boundary is complete.
    peripherals_complete = bool(peripherals and peripherals.get("energy_included"))
    return bool(
        programming.get("energy_included")
        and programming.get("latency_included")
        and static.get("buffer_energy_included")
        and partial.get("energy_included")
        and partial.get("latency_included")
        and peripherals_complete
    )


__all__ = ["ReportAggregator"]
