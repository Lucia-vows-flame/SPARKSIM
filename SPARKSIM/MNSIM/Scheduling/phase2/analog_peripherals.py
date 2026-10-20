"""Bind existing closed-array execution evidence to additional peripherals."""

from MNSIM.Scheduling.phase2.analog_energy import PhysicalReadEvent
from MNSIM.Scheduling.phase2.types import MetricTotals, Phase2Invalid, StepExecutionResult


def score_analog_peripherals(model, result, placement, *, destinations):
    audit = result.mapping_summary["analog_lif"]
    crossbar = audit["numerical_input"]["crossbar"]
    layer_id = crossbar["layer_id"]
    specs = getattr(placement, "fragment_specs_by_id", {})
    fragments = [f for f in specs.values() if str(f.layer_id) == layer_id]
    blocks = {(tuple(b["input_range"]), tuple(b["output_range"])): b for b in crossbar["physical_read_blocks"]}
    # Use real placement proof to bind each numerical block to a physical pair.
    # A closed payload cannot represent bit-plane scaling or a cross-round
    # partial matrix; reject those cases instead of multiplying a layer proxy.
    if len(fragments) != len(blocks):
        raise Phase2Invalid("UNSUPPORTED_PERIPHERAL_READ_TOPOLOGY", f"{layer_id}: numerical read blocks and resident physical fragments disagree")
    events = []
    matched = set()
    for fragment in fragments:
        key = (tuple(fragment.input_range), tuple(fragment.output_range))
        if (key not in blocks or key in matched or fragment.weight_plane_index != 0
                or fragment.physical_xbar_count != 2 or fragment.polarity_count != 2):
            raise Phase2Invalid("UNSUPPORTED_PERIPHERAL_READ_TOPOLOGY", "numerical block and physical fragment must describe the same differential pair")
        matched.add(key)
        block = blocks[key]
        slot = placement.fragment_slots_by_id[fragment.fragment_id]
        instance = ":".join(str(slot[k]) for k in ("tile_id", "pe_id", "group_id"))
        events.append(PhysicalReadEvent(
            event_id=f"{result.item_id}:{result.step_id}:{result.sample_id}:{result.timestep}:{fragment.fragment_id}",
            sample_id=result.sample_id, timestep=result.timestep, instance_id=instance,
            row_voltage_v=tuple(block["row_voltage_v"]),
            columns_per_rail=key[1][1] - key[1][0],
            xbar_load_power_w=block["xbar_load_power_w"],
        ))
    peripheral = model.score(events, spike_codes=audit["state"]["spike_codes"], destinations=destinations)
    peripheral["origin_layer_id"] = layer_id
    peripheral["signal_source"] = "existing_analog_lif_execution"
    peripheral["latency_policy"] = "energy_only_supplement_to_existing_action_window"
    return StepExecutionResult(
        item_id=result.item_id, step_id=f"{result.step_id}.peripherals",
        segment_index=result.segment_index, step_kind="snn_peripheral_energy",
        source_kind="snn_peripheral_energy", status="ok",
        origin_layer_ids=[layer_id],
        metrics=MetricTotals(0.0, peripheral["total_energy_j"], 0.0, 0.0),
        layer_results=[], mapping_summary={"snn_peripheral_energy": peripheral},
        raw_mnsim_audit=[{"audit_kind": "snn_peripheral_energy", **peripheral}],
        boundary_bindings=[],
        warnings=["Additional peripheral profile is example_only; replace illustrative parameters before reporting experimental performance."] if model.profile["status"] == "example_only" else [],
        sample_id=result.sample_id,
        timestep=result.timestep,
    )


def score_static_peripherals(model, result, placement, *, sim_config_path, sample_id, timestep):
    """Bind additional read peripherals to an original MNSIM static step."""
    if model.execution_mode != "mnsim_static_topology":
        raise Phase2Invalid("INVALID_SNN_PERIPHERAL_MODE", "static peripheral scoring requires mnsim_static_topology")
    specs = getattr(placement, "fragment_specs_by_id", {})
    slots = getattr(placement, "fragment_slots_by_id", {})
    parser = __import__("configparser").ConfigParser()
    if not parser.read(sim_config_path, encoding="UTF-8"):
        raise Phase2Invalid("INVALID_SNN_PERIPHERAL_CONFIG", str(sim_config_path))
    try:
        xbar_rows, xbar_cols = (int(part.strip()) for part in parser.get("Crossbar level", "Xbar_Size").split(","))
        read_voltage = parser.getfloat("Device level", "Read_Voltage")
    except (ValueError, KeyError, __import__("configparser").Error) as exc:
        raise Phase2Invalid("INVALID_SNN_PERIPHERAL_CONFIG", str(exc)) from exc
    reads = []
    for audit_index, raw in enumerate(result.raw_mnsim_audit):
        if raw.get("audit_kind") != "static":
            continue
        summary = dict(raw.get("raw_mnsim_summary", {}) or {})
        modules = dict(summary.get("reportable_module_energy_j", {}) or {})
        xbar_energy = float(modules.get("xbar", 0.0))
        xbar_count = int(summary.get("active_physical_xbar_count", 0))
        if xbar_count <= 0 or xbar_energy < 0.0:
            continue
        read_latency = float(summary.get("raw_latency_total_s", 0.0))
        if read_latency <= 0.0:
            raise Phase2Invalid("INVALID_SNN_PERIPHERAL_ACTIVITY", f"{result.step_id}: missing static read latency")
        per_xbar_power = xbar_energy / read_latency / xbar_count
        fragment_ids = list(summary.get("selected_fragment_ids", []) or [])
        if not fragment_ids:
            fragment_ids = list(getattr(placement, "mapping_summary", {}).get("active_fragment_ids", []) or [])
        for fragment_id in fragment_ids:
            spec = specs.get(str(fragment_id))
            slot = slots.get(str(fragment_id))
            if spec is None or slot is None:
                continue
            count = max(1, int(getattr(spec, "physical_xbar_count", 0)))
            for xbar_index in range(count):
                instance = ":".join(str(slot.get(key, "")) for key in ("tile_id", "pe_id", "group_id"))
                reads.append(PhysicalReadEvent(
                    event_id=f"{result.item_id}:{result.step_id}:{sample_id}:{timestep}:{audit_index}:{fragment_id}:{xbar_index}",
                    sample_id=str(sample_id), timestep=int(timestep), instance_id=f"{instance}:xbar{xbar_index}",
                    row_voltage_v=tuple([read_voltage] * xbar_rows), columns_per_rail=xbar_cols,
                    xbar_load_power_w=per_xbar_power,
                ))
    # Structural/merge/LIF steps can legitimately have no xbar read.  They
    # contribute no read-gated AROW or column-reference activity and should be
    # left to their existing MNSIM owner without manufacturing a proxy event.
    if not reads:
        return None
    peripheral = model.score_static_topology(reads=reads, sample_id=sample_id, timestep=timestep)
    peripheral["latency_policy"] = "energy_only_supplement_to_existing_action_window"
    return StepExecutionResult(
        item_id=result.item_id, step_id=f"{result.step_id}.peripherals",
        segment_index=result.segment_index, step_kind="snn_peripheral_energy",
        source_kind="snn_peripheral_energy", status="ok",
        origin_layer_ids=list(result.origin_layer_ids),
        metrics=MetricTotals(0.0, peripheral["total_energy_j"], 0.0, 0.0),
        layer_results=[], mapping_summary={"snn_peripheral_energy": peripheral},
        raw_mnsim_audit=[{"audit_kind": "snn_peripheral_energy", **peripheral}],
        boundary_bindings=[], warnings=[], sample_id=result.sample_id, timestep=result.timestep,
    )


def spike_destinations(plan, node_id):
    """Direct graph destinations, including consumers in later rounds."""
    destinations = set()
    for item in plan.execution_items:
        for edge in item.same_item_edges:
            if str(edge.producer_node_id) == node_id:
                destinations.add(str(edge.consumer_node_id))
        for ref in item.external_inputs:
            producer = str(ref.state_id).removeprefix("state:").removesuffix(":partial")
            if producer == node_id:
                destinations.update(str(n) for n in ref.consumer_node_ids)
    return sorted(destinations)
