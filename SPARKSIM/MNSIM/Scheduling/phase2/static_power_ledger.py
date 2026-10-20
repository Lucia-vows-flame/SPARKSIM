"""Auditable Stage-A SRAM static-power accounting."""

from __future__ import annotations

import configparser
import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from MNSIM.Scheduling.phase2.types import Phase2Invalid


BOUNDARY_ID = "round_orchestrated_accelerator_core_static_v1"
POLICY_NO_IMPLICIT_POWER_GATING = "no_implicit_power_gating"
PROFILE_SCHEMA_VERSION = 1

_BLOCK_LAYOUT = {
    "pe_input_buffer": ("pe_input_sram", "physical_pe_count"),
    "tile_state_output_buffer": (
        "tile_state_output_sram",
        "physical_tile_count",
    ),
    "global_accumulator_buffer": ("global_accumulator_sram", None),
    "boundary_state_buffer": ("boundary_state_sram", None),
}

_UNMODELED_NON_SRAM_BLOCKS = (
    "rram_array_read_ready",
    "adc_standby",
    "dac_standby",
    "crossbar_cmos_periphery",
    "pe_tile_digital_logic",
)


def load_static_power_ledger(sim_config_path: str) -> "StaticPowerLedger | None":
    """Load an opt-in ledger, failing closed when its source contract is invalid."""

    config_path = Path(sim_config_path).resolve()
    parser = configparser.ConfigParser()
    try:
        loaded = parser.read(config_path, encoding="UTF-8")
    except (OSError, configparser.Error) as error:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_CONFIG",
            f"could not read {config_path}: {error}",
        ) from error
    if not loaded or not parser.has_section("Static Power"):
        return None
    try:
        enabled = parser.getboolean("Static Power", "Static_Power_Enable")
    except (ValueError, configparser.Error) as error:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_CONFIG",
            "Static_Power_Enable must be an explicit boolean",
        ) from error
    if not enabled:
        return None

    profile_value = parser.get("Static Power", "Static_Power_Profile", fallback="").strip()
    if not profile_value:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_CONFIG",
            "enabled static power requires Static_Power_Profile",
        )
    policy = parser.get("Static Power", "Static_Power_Policy", fallback="").strip()
    if policy != POLICY_NO_IMPLICIT_POWER_GATING:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_CONFIG",
            f"unsupported Static_Power_Policy {policy!r}",
        )
    try:
        legacy_access_leakage = parser.getboolean(
            "Architecture level",
            "Buffer_IncludeLeakageEnergy",
            fallback=True,
        )
    except ValueError as error:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_CONFIG",
            "Buffer_IncludeLeakageEnergy must be boolean",
        ) from error
    if legacy_access_leakage:
        raise Phase2Invalid(
            "STATIC_POWER_DOUBLE_COUNT",
            (
                "central SRAM static power requires "
                "Buffer_IncludeLeakageEnergy = 0"
            ),
        )

    profile_path = Path(profile_value)
    if not profile_path.is_absolute():
        profile_base = parser.get(
            "Static Power",
            "Static_Power_Profile_Base",
            fallback="sim_config_directory",
        ).strip()
        if profile_base == "sim_config_directory":
            base_path = config_path.parent
        elif profile_base == "repository_root":
            base_path = Path(__file__).resolve().parents[3]
        else:
            raise Phase2Invalid(
                "INVALID_STATIC_POWER_CONFIG",
                f"unsupported Static_Power_Profile_Base {profile_base!r}",
            )
        profile_path = base_path / profile_path
    return StaticPowerLedger(
        sim_config_path=config_path,
        parser=parser,
        profile_path=profile_path.resolve(),
        policy=policy,
    )


class StaticPowerLedger:
    """Collect capacity evidence and integrate modeled SRAM power over wall time."""

    def __init__(
        self,
        *,
        sim_config_path: Path,
        parser: configparser.ConfigParser,
        profile_path: Path,
        policy: str,
    ) -> None:
        self.sim_config_path = Path(sim_config_path)
        self.profile_path = Path(profile_path)
        self.policy = str(policy)
        self.profile, self.profile_digest = _load_profile(self.profile_path)
        self.topology = _physical_topology(parser)
        self.configured_capacity_bits = {
            "pe_input_buffer": _configured_kb_bits(
                parser, "Process element level", "PE_inBuf_Size"
            ),
            "tile_state_output_buffer": _configured_kb_bits(
                parser, "Tile level", "Tile_outBuf_Size"
            ),
            "global_accumulator_buffer": None,
            "boundary_state_buffer": None,
        }
        self._buffer_access_energy_per_bit_j = _finite_positive_float(
            parser.get("Architecture level", "Buffer_AccessEnergyPerBit"),
            "Buffer_AccessEnergyPerBit",
        ) * 1e-9
        self._buffer_access_width_bits = _positive_int(
            parser.get("Architecture level", "Buffer_Bitwidth"),
            "Buffer_Bitwidth",
        )
        self._buffer_access_latency_s = _finite_positive_float(
            parser.get("Architecture level", "Buffer_AccessLatency"),
            "Buffer_AccessLatency",
        ) * 1e-9
        self._capacity_evidence: list[dict[str, Any]] = []
        self._pe_working_bits = 0
        self._tile_working_bits = 0
        self._global_working_bits = 0
        self._lif_state_bytes: dict[tuple[str, int], int] = {}
        self._pe_capacity_unknown_reasons: set[str] = set()
        self._tile_capacity_unknown_reasons: set[str] = set()
        self._boundary_live_states: dict[str, dict[str, Any]] = {}
        self._boundary_peak_live_bits = 0
        self._boundary_capacity_unknown_reasons: set[str] = set()
        self._boundary_liveness_trace: list[dict[str, Any]] = []
        self._boundary_read_bits_by_round: dict[str, int] = {}
        self._boundary_write_bits_by_round: dict[str, int] = {}
        self._boundary_completed_item_ids: set[str] = set()
        self._boundary_observed = False
        self._partial_boundary_access_observed = False
        self._global_observed = False
        self._finalized = False

    def observe_round(
        self,
        *,
        item_id: str,
        placement: Any,
        step_results: list[Any],
        boundary_commit_result: Any,
    ) -> None:
        if self._finalized:
            raise Phase2Invalid(
                "STATIC_POWER_LEDGER_FINALIZED",
                "cannot add capacity evidence after finalization",
            )

        mapping = dict(getattr(placement, "mapping_summary", {}) or {})
        raw_evidence = list(mapping.get("tcg_buffer_capacity_evidence", []) or [])
        for entry in raw_evidence:
            if not isinstance(entry, Mapping):
                raise Phase2Invalid(
                    "INVALID_STATIC_CAPACITY_EVIDENCE",
                    f"{item_id}: TCG capacity evidence must be a mapping",
                )
            pe_bits = _nonnegative_ceil_bits(
                entry.get("pe_input_working_set_bytes_per_instance", 0.0),
                f"{item_id}: PE input working set",
            )
            tile_bits = _nonnegative_ceil_bits(
                entry.get("tile_output_working_set_bytes_per_instance", 0.0),
                f"{item_id}: tile output working set",
            )
            self._pe_working_bits = max(self._pe_working_bits, pe_bits)
            self._tile_working_bits = max(self._tile_working_bits, tile_bits)
            self._capacity_evidence.append(
                {
                    "item_id": str(item_id),
                    **deepcopy(dict(entry)),
                    "pe_input_working_set_bits_per_instance": pe_bits,
                    "tile_output_working_set_bits_per_instance": tile_bits,
                }
            )

        if placement is not None and not raw_evidence:
            self._pe_capacity_unknown_reasons.add(
                "weighted placement lacks pre-rounding TCG capacity evidence"
            )
            self._tile_capacity_unknown_reasons.add(
                "weighted placement lacks pre-rounding TCG capacity evidence"
            )
            self._pe_working_bits = max(
                self._pe_working_bits,
                int(mapping.get("tcg_pe_input_rounded_capacity_bits_per_instance", 0)),
            )
            self._tile_working_bits = max(
                self._tile_working_bits,
                int(mapping.get("tcg_tile_output_rounded_capacity_bits_per_instance", 0)),
            )

        for step in step_results:
            step_mapping = dict(getattr(step, "mapping_summary", {}) or {})
            lif = dict(step_mapping.get("digital_lif", {}) or {})
            if lif:
                producer = str(step_mapping.get("producer_layer_id", step.step_id))
                for raw_tile_id, raw_byte_count in dict(
                    lif.get("state_bytes_by_tile", {}) or {}
                ).items():
                    tile_id = int(raw_tile_id)
                    byte_count = _nonnegative_int(
                        raw_byte_count,
                        f"{item_id}: Digital-LIF state bytes",
                    )
                    key = (producer, tile_id)
                    self._lif_state_bytes[key] = max(
                        self._lif_state_bytes.get(key, 0), byte_count
                    )

            partial = dict(
                step_mapping.get("scheduled_partial_accumulator", {}) or {}
            )
            if partial and (
                int(partial.get("state_read_bytes", 0)) > 0
                or int(partial.get("state_write_bytes", 0)) > 0
            ):
                # The partial-accumulator cost model already owns these SRAM
                # accesses. The central ledger owns capacity and retention,
                # and must not charge the same dynamic access a second time.
                self._partial_boundary_access_observed = True

            global_kb = step_mapping.get("global_buf_size_kb")
            global_kb_values = [global_kb] if global_kb is not None else []
            global_kb_values.extend(
                dict(getattr(layer, "tcg_mapping", {}) or {}).get(
                    "global_buf_size_kb"
                )
                for layer in list(getattr(step, "layer_results", []) or [])
                if dict(getattr(layer, "tcg_mapping", {}) or {}).get(
                    "global_buf_size_kb"
                )
                is not None
            )
            for global_kb_value in global_kb_values:
                self._global_observed = True
                self._global_working_bits = max(
                    self._global_working_bits,
                    _nonnegative_ceil_bits(
                        float(global_kb_value) * 1024,
                        f"{item_id}: global accumulator working set",
                    ),
                )

        self._observe_boundary_sram(
            item_id=str(item_id),
            boundary_commit_result=boundary_commit_result,
        )

    def _observe_boundary_sram(
        self,
        *,
        item_id: str,
        boundary_commit_result: Any,
    ) -> None:
        live_before = deepcopy(self._boundary_live_states)
        read_bits = 0
        consumed_state_ids: set[str] = set()
        for event in list(getattr(boundary_commit_result, "events", []) or []):
            if str(getattr(event, "event_kind", "")) != "read_boundary_state":
                continue
            event_detail = dict(getattr(event, "detail", {}) or {})
            if str(event_detail.get("storage_backend", "boundary_sram")) == "ddr":
                # DDR boundary traffic is accounted by the boundary manager's
                # external backend and must not enter the on-chip SRAM ledger.
                continue
            state_id = str(getattr(event, "state_id", ""))
            state_spec = live_before.get(state_id)
            if state_spec is None:
                raise Phase2Invalid(
                    "MISSING_BOUNDARY_SRAM_STATE",
                    f"{item_id}: read references unknown boundary SRAM state {state_id!r}",
                )
            consumed_state_ids.add(state_id)
            if not _is_partial_boundary_state(state_spec):
                read_bits += int(state_spec["size_bits"])

        created_states = list(
            getattr(boundary_commit_result, "created_states", []) or []
        )
        updated_states = list(
            getattr(boundary_commit_result, "updated_states", []) or []
        )
        output_specs: dict[str, dict[str, Any]] = {}
        update_source_ids: dict[str, str] = {}
        write_bits = 0
        for state, is_update in [
            *((state, False) for state in created_states),
            *((state, True) for state in updated_states),
        ]:
            if str(getattr(state, "storage_backend", "")) != "boundary_sram":
                continue
            all_consumers = [
                str(value)
                for value in list(
                    getattr(state, "allowed_consumer_item_ids", []) or []
                )
            ]
            state_id = str(getattr(state, "state_id", ""))
            update_source_id = None
            if is_update:
                update_source_id = _boundary_update_source_id(
                    state=state,
                    live_before=live_before,
                )
                if update_source_id is not None:
                    update_source_ids[state_id] = update_source_id
            consumers = [
                consumer
                for consumer in all_consumers
                if consumer != item_id
                and consumer not in self._boundary_completed_item_ids
            ]
            if not consumers:
                continue
            self._boundary_observed = True
            try:
                size_bits = _boundary_state_size_bits(state)
            except Phase2Invalid as error:
                self._boundary_capacity_unknown_reasons.add(str(error))
                continue
            status = str(getattr(state, "status", ""))
            output_specs[state_id] = {
                "size_bits": size_bits,
                "status": status,
                "remaining_consumer_item_ids": consumers,
                "producer_layer_id": str(
                    getattr(state, "producer_layer_id", "")
                ),
                "payload_kind": str(getattr(state, "payload_kind", "")),
                "logical_identity": _boundary_state_logical_identity(state),
            }
            if not _is_partial_boundary_state(output_specs[state_id]):
                write_bits += size_bits

        overlap_states = deepcopy(live_before)
        for state_id, state_spec in output_specs.items():
            source_id = update_source_ids.get(state_id)
            if source_id is not None and source_id != state_id:
                overlap_states.pop(source_id, None)
            overlap_states[state_id] = deepcopy(state_spec)
        overlap_bits = sum(
            int(spec["size_bits"]) for spec in overlap_states.values()
        )
        self._boundary_peak_live_bits = max(
            self._boundary_peak_live_bits,
            overlap_bits,
        )

        live_after = deepcopy(live_before)
        for state_id in consumed_state_ids:
            state_spec = live_after[state_id]
            remaining = [
                consumer
                for consumer in state_spec["remaining_consumer_item_ids"]
                if consumer != item_id
            ]
            if remaining:
                state_spec["remaining_consumer_item_ids"] = remaining
            else:
                del live_after[state_id]
        for state_id, source_id in update_source_ids.items():
            live_after.pop(source_id, None)
        live_after.update(deepcopy(output_specs))
        self._boundary_live_states = live_after
        self._boundary_completed_item_ids.add(item_id)

        self._boundary_read_bits_by_round[item_id] = (
            self._boundary_read_bits_by_round.get(item_id, 0) + read_bits
        )
        self._boundary_write_bits_by_round[item_id] = (
            self._boundary_write_bits_by_round.get(item_id, 0) + write_bits
        )
        self._boundary_liveness_trace.append(
            {
                "item_id": item_id,
                "live_bits_before": sum(
                    int(spec["size_bits"]) for spec in live_before.values()
                ),
                "read_bits": read_bits,
                "write_bits": write_bits,
                "safe_overlap_live_bits": overlap_bits,
                "live_bits_after": sum(
                    int(spec["size_bits"]) for spec in live_after.values()
                ),
                "live_state_count_before": len(live_before),
                "live_state_count_after": len(live_after),
            }
        )

    def finalize(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        if self._finalized:
            raise Phase2Invalid(
                "STATIC_POWER_LEDGER_FINALIZED",
                "static-power ledger may be finalized only once",
            )
        self._finalized = True
        boundary_access_by_round: dict[str, dict[str, Any]] = {}
        for record in records:
            item_id = str(record.get("item_id", ""))
            read_bits = int(self._boundary_read_bits_by_round.get(item_id, 0))
            write_bits = int(self._boundary_write_bits_by_round.get(item_id, 0))
            read_accesses = math.ceil(read_bits / self._buffer_access_width_bits)
            write_accesses = math.ceil(write_bits / self._buffer_access_width_bits)
            boundary_access_by_round[item_id] = {
                "read_bits": read_bits,
                "write_bits": write_bits,
                "read_accesses": read_accesses,
                "write_accesses": write_accesses,
                "energy_j": (
                    (read_bits + write_bits)
                    * self._buffer_access_energy_per_bit_j
                ),
                "latency_s": (
                    (read_accesses + write_accesses)
                    * self._buffer_access_latency_s
                ),
            }
        durations = []
        for record in records:
            item_id = str(record.get("item_id", ""))
            duration = _finite_nonnegative_float(
                record.get("latency_s", 0.0),
                f"{item_id or '<unknown>'}: static interval duration",
            )
            duration += float(boundary_access_by_round[item_id]["latency_s"])
            durations.append((item_id, duration))
        total_duration_s = sum(duration for _item_id, duration in durations)

        lif_bytes_by_tile: dict[int, int] = {}
        for (_producer, tile_id), byte_count in self._lif_state_bytes.items():
            lif_bytes_by_tile[tile_id] = lif_bytes_by_tile.get(tile_id, 0) + byte_count
        peak_lif_bits_per_tile = max(
            (byte_count * 8 for byte_count in lif_bytes_by_tile.values()),
            default=0,
        )

        requirements = {
            "pe_input_buffer": {
                "required_bits": self._pe_working_bits,
                "capacity_complete": not self._pe_capacity_unknown_reasons
                and self._pe_working_bits > 0,
                "reasons": sorted(self._pe_capacity_unknown_reasons)
                or ([] if self._pe_working_bits > 0 else [
                    "no PE input capacity evidence was observed"
                ]),
            },
            "tile_state_output_buffer": {
                "required_bits": self._tile_working_bits + peak_lif_bits_per_tile,
                "capacity_complete": not self._tile_capacity_unknown_reasons
                and self._tile_working_bits > 0,
                "reasons": sorted(self._tile_capacity_unknown_reasons)
                or ([] if self._tile_working_bits > 0 else [
                    "no tile output capacity evidence was observed"
                ]),
            },
            "global_accumulator_buffer": {
                "required_bits": self._global_working_bits,
                "capacity_complete": self._global_working_bits > 0,
                "reasons": [] if self._global_working_bits > 0 else [
                    "no global accumulator instance was observed"
                ],
            },
            "boundary_state_buffer": {
                "required_bits": self._boundary_peak_live_bits,
                "capacity_complete": not self._boundary_capacity_unknown_reasons
                and self._boundary_peak_live_bits > 0,
                "reasons": sorted(self._boundary_capacity_unknown_reasons)
                or ([] if self._boundary_peak_live_bits > 0 else [
                    "no cross-round boundary SRAM state was observed"
                ]),
            },
        }

        inventory: dict[str, Any] = {
            **self.topology,
            "blocks": [],
        }
        capacity_audit: dict[str, Any] = {
            "tcg_evidence": deepcopy(self._capacity_evidence),
            "digital_lif_state_bytes_by_tile": {
                str(tile_id): byte_count
                for tile_id, byte_count in sorted(lif_bytes_by_tile.items())
            },
            "boundary_sram_liveness_trace": deepcopy(
                self._boundary_liveness_trace
            ),
            "blocks": {},
        }
        modeled_blocks: list[str] = []
        unmodeled_blocks = list(_UNMODELED_NON_SRAM_BLOCKS)
        excluded_blocks = ["global_controller"]
        energy_by_block_j: dict[str, float] = {}
        power_by_block_w: dict[str, float] = {}

        for block_class, count_key in (
            ("adc_standby", "physical_adc_count"),
            ("dac_standby", "physical_dac_count"),
        ):
            inventory["blocks"].append(
                {
                    "block_class": block_class,
                    "instance_count": self.topology[count_key],
                    "capacity_bits_per_instance": None,
                    "inventory_source": "sim_config_physical_converter_topology",
                    "characterization_status": "unmodeled",
                    "power_state": "idle_or_read_ready_not_characterized",
                    "operation_energy_boundary": "unknown",
                }
            )

        for profile_key, (block_class, topology_count_key) in _BLOCK_LAYOUT.items():
            requirement = requirements[profile_key]
            if topology_count_key is not None:
                instance_count = int(self.topology[topology_count_key])
            elif profile_key == "global_accumulator_buffer":
                instance_count = 1 if self._global_observed else 0
            elif profile_key == "boundary_state_buffer":
                instance_count = 1 if self._boundary_observed else 0
            else:
                raise AssertionError(f"missing instance policy for {profile_key}")
            if instance_count == 0:
                excluded_blocks.append(block_class)
                capacity_audit["blocks"][block_class] = {
                    "instance_count": 0,
                    "capacity_complete": True,
                    "exclusion_reason": "no physical instance observed in this run",
                }
                continue

            block_spec = dict(self.profile["blocks"].get(profile_key, {}) or {})
            audit = _materialize_block_capacity(
                block_class=block_class,
                instance_count=instance_count,
                required_bits=int(requirement["required_bits"]),
                configured_bits=self.configured_capacity_bits[profile_key],
                capacity_complete=bool(requirement["capacity_complete"]),
                incomplete_reasons=list(requirement["reasons"]),
                block_spec=block_spec,
            )
            capacity_audit["blocks"][block_class] = audit
            inventory["blocks"].append(
                {
                    "block_class": block_class,
                    "instance_count": instance_count,
                    "capacity_bits_per_instance": audit.get(
                        "materialized_capacity_bits"
                    ),
                }
            )
            if not audit["modeled"]:
                unmodeled_blocks.append(block_class)
                continue
            modeled_blocks.append(block_class)
            block_power_w = float(audit["total_static_power_w"])
            power_by_block_w[block_class] = block_power_w
            energy_by_block_j[block_class] = block_power_w * total_duration_s

        total_static_power_w = sum(power_by_block_w.values())
        total_static_energy_j = total_static_power_w * total_duration_s
        per_round_energy_j = {
            item_id: total_static_power_w * duration
            for item_id, duration in durations
        }
        static_accounting_complete = not unmodeled_blocks
        warnings = [
            "Stage-A static power is an SRAM-only CACTI architecture estimate.",
        ]
        if unmodeled_blocks:
            warnings.append(
                "Static accounting is incomplete; unmodeled blocks are listed explicitly."
            )

        audit = {
            "enabled": True,
            "boundary_id": self.profile["boundary_id"],
            "profile_id": self.profile["profile_id"],
            "profile_digest": self.profile_digest,
            "profile_path": str(self.profile_path),
            "evidence_scope": self.profile["evidence_scope"],
            "pvt": deepcopy(self.profile["pvt"]),
            "policy": self.policy,
            "inventory": inventory,
            "capacity_audit": capacity_audit,
            "interval_duration_s": total_duration_s,
            "duration_by_accounting_owner_s": {
                "complete_physical_timeline": total_duration_s
            },
            "operation_energy_boundary_complete": False,
            "cross_round_storage": {
                "backend": "shared_on_chip_sram",
                "observed": self._boundary_observed,
                "capacity_included": self._boundary_observed
                and not self._boundary_capacity_unknown_reasons,
                "dynamic_energy_included": self._boundary_observed,
                "latency_included": self._boundary_observed,
                "partial_state_access_owner": (
                    "scheduled_partial_accumulator"
                ),
                "spill_policy": "no_ddr_fail_if_capacity_is_insufficient",
            },
            "external_memory_boundary": {
                "backend": "ddr",
                "observed": False,
                "energy_included": False,
                "latency_included": False,
            },
            "modeled_blocks": sorted(modeled_blocks),
            "unmodeled_blocks": sorted(set(unmodeled_blocks)),
            "excluded_blocks": sorted(set(excluded_blocks)),
            "static_accounting_complete": static_accounting_complete,
            "implementation_specific_complete": False,
            "static_power_by_block_w": power_by_block_w,
            "static_energy_by_block_j": energy_by_block_j,
            "static_energy_by_state_j": {"retention": total_static_energy_j},
            "static_energy_by_round_j": per_round_energy_j,
            "boundary_sram_access_by_round": deepcopy(
                boundary_access_by_round
            ),
            "warnings": warnings,
        }
        boundary_access_energy_by_round_j = {
            item_id: float(values["energy_j"])
            for item_id, values in boundary_access_by_round.items()
        }
        boundary_access_latency_by_round_s = {
            item_id: float(values["latency_s"])
            for item_id, values in boundary_access_by_round.items()
        }
        return {
            "static_energy_j": total_static_energy_j,
            "static_average_power_w": total_static_power_w,
            "static_energy_by_block_j": energy_by_block_j,
            "static_energy_by_state_j": {"retention": total_static_energy_j},
            "static_energy_by_round_j": per_round_energy_j,
            "boundary_sram_access_energy_j": sum(
                boundary_access_energy_by_round_j.values()
            ),
            "boundary_sram_access_latency_s": sum(
                boundary_access_latency_by_round_s.values()
            ),
            "boundary_sram_access_energy_by_round_j": (
                boundary_access_energy_by_round_j
            ),
            "boundary_sram_access_latency_by_round_s": (
                boundary_access_latency_by_round_s
            ),
            "static_power_audit": audit,
        }


def _load_profile(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        profile = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"could not load {path}: {error}",
        ) from error
    if not isinstance(profile, dict):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "profile root must be an object",
        )
    required = {
        "schema_version",
        "profile_id",
        "evidence_scope",
        "implementation_specific",
        "boundary_id",
        "pvt",
        "source_generator",
        "blocks",
    }
    missing = sorted(required.difference(profile))
    if missing:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"profile is missing fields {missing}",
        )
    if profile["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"unsupported schema_version {profile['schema_version']!r}",
        )
    if profile["boundary_id"] != BOUNDARY_ID:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"unsupported boundary_id {profile['boundary_id']!r}",
        )
    if profile["implementation_specific"] is not False:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "Stage-A CACTI profile must declare implementation_specific=false",
        )
    if profile["evidence_scope"] != "physically_grounded_architecture_estimate":
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "Stage-A profile must declare physically_grounded_architecture_estimate",
        )
    generator = profile["source_generator"]
    if not isinstance(generator, dict) or not {
        "tool",
        "revision",
        "binary_sha256",
        "power_gating",
        "ecc",
        "optimization_mode",
    }.issubset(generator):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "source_generator must pin the CACTI build and generation policy",
        )
    if generator["power_gating"] is not False or generator["ecc"] is not False:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "Stage-A source generator must disable power gating and ECC",
        )
    pvt = profile["pvt"]
    if not isinstance(pvt, dict) or not {
        "technology_nm",
        "voltage_v",
        "temperature_c",
        "process_corner",
    }.issubset(pvt):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "profile requires complete technology, voltage, temperature, and corner PVT",
        )
    _finite_positive_float(pvt["technology_nm"], "PVT technology_nm")
    _finite_positive_float(pvt["voltage_v"], "PVT voltage_v")
    _finite_float(pvt["temperature_c"], "PVT temperature_c")
    if not isinstance(pvt["process_corner"], str) or not pvt["process_corner"].strip():
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "PVT process_corner must be a nonempty string",
        )
    if not isinstance(profile["blocks"], dict):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            "blocks must be an object",
        )
    for key, spec in profile["blocks"].items():
        if key not in _BLOCK_LAYOUT:
            raise Phase2Invalid(
                "INVALID_STATIC_POWER_PROFILE",
                f"unsupported SRAM block {key!r}",
            )
        _validate_block_spec(key, spec)
    return profile, hashlib.sha256(raw).hexdigest()


def _validate_block_spec(key: str, raw_spec: Any) -> None:
    if not isinstance(raw_spec, dict):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} specification must be an object",
        )
    if raw_spec.get("model_kind") != "fixed_bank_macro_replication":
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} must use fixed_bank_macro_replication",
        )
    bank_bits = _positive_int(raw_spec.get("bank_capacity_bits"), f"{key} bank capacity")
    if raw_spec.get("capacity_unit") != "bit":
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} capacity_unit must be 'bit'",
        )
    if raw_spec.get("evidence_level") != "cacti_architecture_estimate":
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} evidence_level must identify the CACTI architecture estimate",
        )
    macro = raw_spec.get("macro_characterization")
    if not isinstance(macro, dict):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} requires macro_characterization",
        )
    if int(macro.get("capacity_bits", -1)) != bank_bits:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} macro capacity does not match bank capacity",
        )
    if macro.get("power_unit") != "W":
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} macro power_unit must be 'W'",
        )
    subthreshold = _finite_nonnegative_float(
        macro.get("subthreshold_leakage_power_w"),
        f"{key} subthreshold leakage",
    )
    gate = _finite_nonnegative_float(
        macro.get("gate_leakage_power_w"),
        f"{key} gate leakage",
    )
    if subthreshold + gate <= 0:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_PROFILE",
            f"{key} bank static power must be positive",
        )
    for digest_key in (
        "source_input_sha256",
        "source_output_sha256",
        "source_csv_sha256",
    ):
        digest = macro.get(digest_key)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Phase2Invalid(
                "INVALID_STATIC_POWER_PROFILE",
                f"{key} requires a SHA-256 {digest_key}",
            )


def _physical_topology(parser: configparser.ConfigParser) -> dict[str, int]:
    tile_count = _pair_product(parser, "Architecture level", "Tile_Num")
    pe_per_tile = _pair_product(parser, "Tile level", "PE_Num")
    group_per_pe = _positive_int(
        parser.get("Process element level", "Group_Num"), "Group_Num"
    )
    polarity = _positive_int(
        parser.get("Process element level", "Xbar_Polarity"), "Xbar_Polarity"
    )
    if polarity not in (1, 2):
        raise Phase2Invalid(
            "INVALID_STATIC_HARDWARE_INVENTORY",
            f"unsupported Xbar_Polarity {polarity}",
        )
    xbar_rows, _xbar_columns = _positive_pair(
        parser, "Crossbar level", "Xbar_Size"
    )
    subarray_rows = _positive_int(
        parser.get("Crossbar level", "Subarray_Size"), "Subarray_Size"
    )
    if xbar_rows % subarray_rows != 0:
        raise Phase2Invalid(
            "INVALID_STATIC_HARDWARE_INVENTORY",
            (
                f"Xbar_Size row count {xbar_rows} is not divisible by "
                f"Subarray_Size {subarray_rows}"
            ),
        )
    subarrays_per_xbar = xbar_rows // subarray_rows
    adc_per_subarray_per_group = _positive_int(
        parser.get("Process element level", "ADC_Num"), "ADC_Num"
    )
    dac_per_subarray_per_group = _positive_int(
        parser.get("Process element level", "DAC_Num"), "DAC_Num"
    )
    pe_count = tile_count * pe_per_tile
    group_count = pe_count * group_per_pe
    adc_per_group = adc_per_subarray_per_group * subarrays_per_xbar
    dac_per_group = dac_per_subarray_per_group * subarrays_per_xbar
    return {
        "physical_tile_count": tile_count,
        "physical_pe_count": pe_count,
        "physical_group_count": group_count,
        "physical_xbar_count": group_count * polarity,
        "subarrays_per_xbar": subarrays_per_xbar,
        "adc_instances_per_group": adc_per_group,
        "dac_instances_per_group": dac_per_group,
        "physical_adc_count": group_count * adc_per_group,
        "physical_dac_count": group_count * dac_per_group,
    }


def _boundary_state_size_bits(state: Any) -> int:
    shape = dict(getattr(state, "shape", {}) or {})
    payload_kind = str(getattr(state, "payload_kind", ""))
    if payload_kind == "spike_sequence":
        bit_width = 1
    else:
        raw_bit_width = getattr(state, "data_bitwidth", None)
        if raw_bit_width is None:
            raw_bit_width = shape.get("outputbit")
        bit_width = _positive_int(raw_bit_width, "boundary state data_bitwidth")

    region = shape.get("output_neuron_region")
    if (
        payload_kind == "spike_sequence"
        and isinstance(region, (list, tuple))
        and len(region) == 2
    ):
        element_count = int(region[1]) - int(region[0])
    elif "Outfeature" in shape:
        element_count = _positive_int(shape["Outfeature"], "boundary Outfeature")
    elif "Outputchannel" in shape:
        element_count = _positive_int(
            shape["Outputchannel"], "boundary Outputchannel"
        ) * _shape_product(shape.get("Outputsize", [1, 1]), "boundary Outputsize")
    elif "C" in shape:
        element_count = _positive_int(shape["C"], "boundary C")
        spatial_values = []
        for key in ("H", "W"):
            if key in shape:
                spatial_values.append(shape[key])
        if spatial_values:
            element_count *= _shape_product(spatial_values, "boundary spatial shape")
    else:
        raise Phase2Invalid(
            "INVALID_BOUNDARY_SRAM_SHAPE",
            f"cannot derive element count from boundary shape {shape!r}",
        )
    if element_count <= 0:
        raise Phase2Invalid(
            "INVALID_BOUNDARY_SRAM_SHAPE",
            f"boundary state element count must be positive, got {element_count}",
        )
    return element_count * bit_width


def _boundary_state_logical_identity(state: Any) -> tuple[str, str, int | None]:
    raw_timestep = getattr(state, "timestep", None)
    timestep = None if raw_timestep is None else int(raw_timestep)
    return (
        str(getattr(state, "producer_layer_id", "")),
        str(getattr(state, "sample_id", "")),
        timestep,
    )


def _is_partial_boundary_state(state: Any) -> bool:
    if isinstance(state, dict):
        status = state.get("status", "")
        payload_kind = state.get("payload_kind", "")
    else:
        status = getattr(state, "status", "")
        payload_kind = getattr(state, "payload_kind", "")
    return str(status) == "partial_sum" or str(payload_kind).startswith("partial_sum")


def _boundary_update_source_id(
    *,
    state: Any,
    live_before: dict[str, dict[str, Any]],
) -> str | None:
    state_id = str(getattr(state, "state_id", ""))
    if state_id in live_before:
        return state_id
    identity = _boundary_state_logical_identity(state)
    candidates = [
        live_state_id
        for live_state_id, state_spec in live_before.items()
        if state_spec.get("logical_identity") == identity
    ]
    if len(candidates) > 1:
        raise Phase2Invalid(
            "AMBIGUOUS_BOUNDARY_SRAM_UPDATE",
            (
                f"updated boundary state {state_id!r} matches multiple live "
                f"allocations {candidates!r}"
            ),
        )
    return candidates[0] if candidates else None


def _shape_product(values: Any, label: str) -> int:
    if not isinstance(values, (list, tuple)) or not values:
        raise Phase2Invalid(
            "INVALID_BOUNDARY_SRAM_SHAPE",
            f"{label} must be a nonempty dimension list",
        )
    product = 1
    for value in values:
        product *= _positive_int(value, label)
    return product


def _materialize_block_capacity(
    *,
    block_class: str,
    instance_count: int,
    required_bits: int,
    configured_bits: int | None,
    capacity_complete: bool,
    incomplete_reasons: list[str],
    block_spec: dict[str, Any],
) -> dict[str, Any]:
    audit = {
        "block_class": block_class,
        "instance_count": instance_count,
        "configured_capacity_bits": configured_bits,
        "peak_required_capacity_bits": required_bits,
        "capacity_complete": capacity_complete,
        "incomplete_reasons": list(incomplete_reasons),
        "modeled": False,
    }
    if not capacity_complete:
        return audit
    if not block_spec:
        audit["incomplete_reasons"].append("profile has no block characterization")
        return audit
    bank_bits = int(block_spec["bank_capacity_bits"])
    rounded_bits = int(math.ceil(required_bits / bank_bits) * bank_bits)
    materialized_bits = rounded_bits
    if configured_bits is not None:
        if configured_bits % bank_bits != 0:
            raise Phase2Invalid(
                "INVALID_STATIC_BUFFER_CAPACITY",
                (
                    f"{block_class} configured capacity {configured_bits} bits is not "
                    f"composable from {bank_bits}-bit banks"
                ),
            )
        if configured_bits < rounded_bits:
            raise Phase2Invalid(
                "INVALID_STATIC_BUFFER_CAPACITY",
                (
                    f"{block_class} configured capacity {configured_bits} bits is smaller "
                    f"than rounded requirement {rounded_bits} bits"
                ),
            )
        materialized_bits = configured_bits
    banks_per_instance = materialized_bits // bank_bits
    macro = dict(block_spec["macro_characterization"])
    bank_power_w = float(macro["subthreshold_leakage_power_w"]) + float(
        macro["gate_leakage_power_w"]
    )
    power_per_instance_w = banks_per_instance * bank_power_w
    audit.update(
        {
            "rounded_required_capacity_bits": rounded_bits,
            "materialized_capacity_bits": materialized_bits,
            "rounding_residual_bits": rounded_bits - required_bits,
            "provisioned_headroom_bits": materialized_bits - rounded_bits,
            "rounding_rule_id": "fixed_cacti_bank_integer_replication",
            "bank_capacity_bits": bank_bits,
            "banks_per_instance": banks_per_instance,
            "bank_static_power_w": bank_power_w,
            "static_power_per_instance_w": power_per_instance_w,
            "total_static_power_w": instance_count * power_per_instance_w,
            "evidence_level": block_spec["evidence_level"],
            "source_artifacts": {
                key: macro[key]
                for key in (
                    "source_input_sha256",
                    "source_output_sha256",
                    "source_csv_sha256",
                )
            },
            "modeled": True,
        }
    )
    return audit


def _configured_kb_bits(
    parser: configparser.ConfigParser, section: str, option: str
) -> int | None:
    value = _finite_nonnegative_float(parser.get(section, option), option)
    if value == 0:
        return None
    bits = value * 1024 * 8
    if not bits.is_integer():
        raise Phase2Invalid(
            "INVALID_STATIC_BUFFER_CAPACITY",
            f"{option} does not resolve to an integer bit capacity",
        )
    return int(bits)


def _pair_product(
    parser: configparser.ConfigParser, section: str, option: str
) -> int:
    raw = parser.get(section, option)
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise Phase2Invalid(
            "INVALID_STATIC_HARDWARE_INVENTORY",
            f"{option} must contain exactly two dimensions",
        )
    return _positive_int(parts[0], option) * _positive_int(parts[1], option)


def _positive_pair(
    parser: configparser.ConfigParser, section: str, option: str
) -> tuple[int, int]:
    raw = parser.get(section, option)
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 2:
        raise Phase2Invalid(
            "INVALID_STATIC_HARDWARE_INVENTORY",
            f"{option} must contain exactly two dimensions",
        )
    return _positive_int(parts[0], option), _positive_int(parts[1], option)


def _nonnegative_ceil_bits(value: Any, label: str) -> int:
    byte_count = _finite_nonnegative_float(value, label)
    return int(math.ceil(byte_count * 8))


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise Phase2Invalid("INVALID_STATIC_POWER_VALUE", f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_VALUE", f"{label} must be an integer"
        ) from error
    if str(value).strip() not in {str(parsed), f"+{parsed}"} and not isinstance(value, int):
        raise Phase2Invalid("INVALID_STATIC_POWER_VALUE", f"{label} must be an integer")
    if parsed <= 0:
        raise Phase2Invalid("INVALID_STATIC_POWER_VALUE", f"{label} must be positive")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_VALUE", f"{label} must be an integer"
        ) from error
    if parsed < 0 or isinstance(value, bool):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_VALUE", f"{label} must be nonnegative"
        )
    return parsed


def _finite_positive_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed <= 0:
        raise Phase2Invalid("INVALID_STATIC_POWER_VALUE", f"{label} must be positive")
    return parsed


def _finite_nonnegative_float(value: Any, label: str) -> float:
    parsed = _finite_float(value, label)
    if parsed < 0:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_VALUE", f"{label} must be nonnegative"
        )
    return parsed


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_VALUE", f"{label} must be numeric"
        ) from error
    if not math.isfinite(parsed):
        raise Phase2Invalid(
            "INVALID_STATIC_POWER_VALUE", f"{label} must be finite"
        )
    return parsed


__all__ = [
    "BOUNDARY_ID",
    "POLICY_NO_IMPLICIT_POWER_GATING",
    "StaticPowerLedger",
    "load_static_power_ledger",
]
