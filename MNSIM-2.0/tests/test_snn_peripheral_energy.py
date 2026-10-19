import configparser
import builtins
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

import MNSIM.Scheduling.phase2.executor as executor
from MNSIM.Scheduling.phase2.analog_energy import (
    PhysicalReadEvent, SnnPeripheralEnergyModel, load_snn_peripheral_energy,
)
from MNSIM.Scheduling.phase2.types import Phase2Invalid
from MNSIM.Scheduling.phase2.analog_peripherals import score_analog_peripherals, spike_destinations
from MNSIM.Scheduling.phase2.analog_crossbar import AnalogCrossbarNumericalEngine
from test_round_phase2_analog_dispatch import _analog_config, _direct_payload
from test_round_snn_executor import _Backend, _plan


EXAMPLE = Path(__file__).resolve().parents[1] / "configs/peripherals/snn_additional_example.json"


def _profile():
    return json.loads(EXAMPLE.read_text())


def _model(profile=None):
    return SnnPeripheralEnergyModel(profile or _profile(), profile_path="test-only", digest="test-only")


def _read(event_id="r0", **kwargs):
    return PhysicalReadEvent(event_id, "sample:0", 0, "tile0:pe0:pair0", (0.5, 0.0), 3, 1e-3, **kwargs)


def test_electrical_energy_and_charge_are_not_supply_load_double_counted():
    p = _profile()
    p["parameters"]["read_supply_efficiency"] = 0.5
    result = _model(p).score([_read()], spike_codes=[1, 0, 1], destinations=["next"])
    c = result["included_components_j"]
    # 1 mW for 20 ns: load is 20 pJ, source loss is 20 pJ;
    # the additional ledger bills loss + 0.02 pJ bias, never another load.
    assert c["read_supply"] == pytest.approx(20.02e-12, rel=1e-12, abs=1e-30)
    # Six physical columns, 0.2 fJ bias + 0.04 fJ capacitive per column.
    assert c["column_reference"] == pytest.approx(1.44e-15, rel=1e-12, abs=1e-30)
    assert c["arow_row_driver"] == pytest.approx(0.74e-15, rel=1e-12, abs=1e-30)
    assert c["spike_interconnect"] == pytest.approx(0.98e-15, rel=1e-12, abs=1e-30)
    assert c["interface"] == pytest.approx(0.2e-15, rel=1e-12, abs=1e-30)
    assert "xbar_read" not in c
    assert "lif" not in c
    assert result["total_energy_j"] == pytest.approx(sum(c.values()), rel=1e-12, abs=1e-30)


def test_startup_follows_physical_instance_not_layer_or_timestep():
    model = _model()
    first = model.score([_read()], spike_codes=[1], destinations=[])
    second = model.score([replace(_read("r1"), timestep=1)], spike_codes=[1], destinations=[])
    moved = model.score([replace(_read("r2"), instance_id="tile1:pe0:pair0")], spike_codes=[0], destinations=[])
    assert first["counts"]["startups"] == moved["counts"]["startups"] == 1
    assert second["counts"]["startups"] == 0
    assert first["included_components_j"]["spike_interconnect"] == 0
    assert second["included_components_j"]["arow_startup"] == 0
    with pytest.raises(Phase2Invalid, match="duplicate"):
        model.score([_read()], spike_codes=[0], destinations=[])


def test_real_physical_shards_and_fanout_scale_without_network_constants():
    one = _model().score([_read()], spike_codes=[1, 0], destinations=["a"])
    two = _model().score([_read(), replace(_read("r1"), instance_id="other")], spike_codes=[1, 0], destinations=["a", "b"])
    for component, value in one["included_components_j"].items():
        assert two["included_components_j"][component] == pytest.approx(2 * value, rel=1e-12, abs=1e-30)
    silent = _model().score([replace(_read(), row_voltage_v=(0.0, 0.0), xbar_load_power_w=0)], spike_codes=[0], destinations=["a"])
    assert silent["included_components_j"]["arow_row_driver"] == 0
    assert silent["included_components_j"]["control"] > 0
    assert silent["included_components_j"]["column_reference"] > 0


@pytest.mark.parametrize("mutation", [
    lambda p: p["parameters"].pop("arow_startup_j_per_pair"),
    lambda p: p["parameters"].update(read_supply_efficiency=0),
    lambda p: p["parameters"].update(read_pulse_s=float("nan")),
    lambda p: p["parameters"].update(accepted_external_energy_j=1),
    lambda p: p["ownership"].update(column_reference="already_in_lut"),
])
def test_missing_invalid_or_overlapping_parameters_fail(mutation):
    profile = _profile()
    mutation(profile)
    with pytest.raises(Phase2Invalid):
        _model(profile)


def _enable(config):
    parser = configparser.ConfigParser()
    parser.read(config)
    parser["SNN Peripheral Energy"] = {"Enabled": "1", "Profile": str(EXAMPLE)}
    with open(config, "w") as handle:
        parser.write(handle)


def test_disabled_legacy_configs_need_no_new_profile():
    assert load_snn_peripheral_energy("SimConfig.ini") is None


@pytest.mark.parametrize("sequence_length", [3, 30])
def test_supplement_preserves_frontend_lif_spikes_membrane_latency_and_cost(tmp_path, monkeypatch, sequence_length):
    # Only area is isolated: portable checkout lacks archived transistor decks.
    # Signal, frontend/LIF costs, event extraction and aggregation are real.
    area = SimpleNamespace(area_um2_per_neuron=1.0, for_neurons=lambda n: float(n), to_dict=lambda: {"source": "test_area"})
    monkeypatch.setattr(executor, "_load_analog_lif_area", lambda: area)
    config = _analog_config(tmp_path, static_power_enable=True)
    # The shared LUT fixture uses joule-sized synthetic costs. Give its energy
    # fields physical-scale units so a missing/double supplement cannot hide
    # below floating-point resolution of the unchanged frontend/LIF cost.
    root = tmp_path / "lut"
    for energy_path in root.rglob("energy_*.npy"):
        np.save(energy_path, np.load(energy_path) * 1e-12)
    manifest_path = root / "lut_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for record in manifest["files"]:
        record["sha256"] = hashlib.sha256((root / record["path"]).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    plan = _plan()
    plan.sequence_spec.sequence_length = sequence_length
    payload = _direct_payload()
    payload["analog_lif_payloads"]["L1"]["timesteps"] *= sequence_length // 3
    kwargs = dict(plan=plan, sim_config_path=config, ops_total=1, ops_count_policy="mac_as_1_op", model_inputs={"model_input": payload})
    baseline = executor.execute_round_plan(**kwargs, static_backend=_Backend())
    _enable(config)
    supplemented = executor.execute_round_plan(**kwargs, static_backend=_Backend())
    original_steps = [s for s in supplemented.step_results if s["source_kind"] != "snn_peripheral_energy"]
    assert original_steps == baseline.step_results
    assert supplemented.boundary_events == baseline.boundary_events
    assert supplemented.inference_metrics["analog_lif"] == baseline.inference_metrics["analog_lif"]
    assert supplemented.inference_metrics["latency_s"] == baseline.inference_metrics["latency_s"]
    extra = supplemented.inference_metrics["energy_breakdown_j"]["snn_additional_peripheral_energy_j"]
    assert extra > 0
    assert supplemented.inference_metrics["energy_j"] == pytest.approx(baseline.inference_metrics["energy_j"] + extra, rel=1e-12, abs=1e-24)
    assert supplemented.inference_metrics["energy_j"] - baseline.inference_metrics["energy_j"] == pytest.approx(extra, rel=1e-6, abs=1e-20)
    ledger = supplemented.inference_metrics["energy_ledger"]
    assert ledger["audit"]["reconciliation_passed"]
    assert ledger["audit"]["supplement_event_count"] == sequence_length
    additional_steps = [s for s in supplemented.step_results if s["source_kind"] == "snn_peripheral_energy"]
    assert [s["timestep"] for s in additional_steps] == list(range(sequence_length))
    assert sum(s["mapping_summary"]["snn_peripheral_energy"]["counts"]["startups"] for s in additional_steps) == 1
    assert ledger["included_components_j"]["analog_frontend"] > 0
    assert ledger["included_components_j"]["analog_lif"] > 0
    assert ledger["total_energy_j"] == pytest.approx(sum(ledger["included_components_j"].values()), rel=1e-12, abs=1e-24)
    assert supplemented.inference_metrics["analog_accounting"]["reconciliation_passed"]
    json.dumps(supplemented.to_dict(), allow_nan=False)


def test_peripheral_model_never_opens_spice_results(tmp_path, monkeypatch):
    config = tmp_path / "SimConfig.ini"
    config.write_text(f"[SNN Peripheral Energy]\nEnabled=1\nProfile={EXAMPLE}\n")
    original = Path.open
    original_builtin = builtins.open
    opened = []

    def audited_open(path, *args, **kwargs):
        assert path.suffix.lower() not in {".lis", ".mt0", ".tr0", ".sp", ".csv", ".npy"}
        opened.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", audited_open)
    def audited_builtin(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            assert str(path) in {str(config), str(EXAMPLE)}
            opened.append(str(path))
        return original_builtin(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", audited_builtin)
    model = load_snn_peripheral_energy(config)
    result = model.score([_read()], spike_codes=[0], destinations=[])
    assert result["additional_model_external_energy_read"] is False
    assert opened == [str(config), str(EXAMPLE)]


def test_digital_metadata_only_cannot_fabricate_spike_activity(tmp_path):
    parser = configparser.ConfigParser()
    parser.read("SimConfig.ini")
    parser["SNN Peripheral Energy"] = {"Enabled": "1", "Profile": str(EXAMPLE)}
    config = tmp_path / "digital.ini"
    with config.open("w") as handle:
        parser.write(handle)
    with pytest.raises(Phase2Invalid, match="MISSING_SNN_PERIPHERAL_ACTIVITY"):
        executor.execute_round_plan(_plan(), str(config), 1, "mac_as_1_op", static_backend=_Backend())


def test_adapter_matches_rectangular_physical_fragments_and_rejects_incomplete_mapping(tmp_path):
    config = _analog_config(tmp_path)
    engine = replace(AnalogCrossbarNumericalEngine.from_sim_config(config), xbar_size=(2, 3))
    _, _, crossbar = engine.compute([1, 0, 1], {"fc": [[1] * 4] * 3}, 4)
    assert len(crossbar["physical_read_blocks"]) == 4
    assert sum(b["xbar_load_power_w"] for b in crossbar["physical_read_blocks"]) == pytest.approx(crossbar["xbar_load_power_w"], rel=1e-12, abs=1e-30)
    result = SimpleNamespace(
        mapping_summary={"analog_lif": {"numerical_input": {"crossbar": crossbar}, "state": {"spike_codes": [1, 0, 1, 0]}}},
        item_id="r0", step_id="s0", sample_id="sample:0", timestep=0, segment_index=1,
    )
    specs = {}
    slots = {}
    for i, b in enumerate(crossbar["physical_read_blocks"]):
        name = f"f{i}"
        specs[name] = SimpleNamespace(layer_id="fc", fragment_id=name, input_range=b["input_range"], output_range=b["output_range"], weight_plane_index=0, physical_xbar_count=2, polarity_count=2)
        slots[name] = {"tile_id": i, "pe_id": 0, "group_id": 0}
    placement = SimpleNamespace(fragment_specs_by_id=specs, fragment_slots_by_id=slots)
    extra = score_analog_peripherals(_model(), result, placement, destinations=["next"])
    audit = extra.mapping_summary["snn_peripheral_energy"]
    assert audit["counts"]["read_pairs"] == audit["counts"]["startups"] == 4
    assert audit["counts"]["spike_deliveries"] == 2  # layer output delivered once
    assert extra.boundary_bindings == []
    specs.pop("f3")
    with pytest.raises(Phase2Invalid, match="UNSUPPORTED_PERIPHERAL_READ_TOPOLOGY"):
        score_analog_peripherals(_model(), result, placement, destinations=["next"])


def test_destinations_include_later_rounds_and_deduplicate_graph_edges():
    edge = SimpleNamespace(producer_node_id="lif", consumer_node_id="fc")
    ref = SimpleNamespace(state_id="state:lif", consumer_node_ids=["fc", "other"])
    plan = SimpleNamespace(execution_items=[SimpleNamespace(same_item_edges=[edge], external_inputs=[]), SimpleNamespace(same_item_edges=[], external_inputs=[ref])])
    assert spike_destinations(plan, "lif") == ["fc", "other"]
