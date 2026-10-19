import configparser
import json

import pytest

from MNSIM.Hardware_Model.ADC import ADC
from MNSIM.Hardware_Model.PE import ProcessElement
from MNSIM.Latency_Model.PE_latency import PE_latency_analysis
from MNSIM.Scheduling.phase2.analog_lif_inputs import (
    AnalogLifPayloadError,
    AnalogLifPayloadSession,
)
from MNSIM.Scheduling.phase2.executor import execute_round_plan
from MNSIM.Scheduling.phase2.types import Phase2Invalid
from test_local_analog_lif_lut import _write_artifact
from test_round_snn_executor import _Backend, _plan


def _analog_config(tmp_path, *, static_power_enable=False):
    lut_root = _write_artifact(tmp_path / "lut")
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("SimConfig.ini", encoding="UTF-8")
    if not parser.has_section("Neuron"):
        parser.add_section("Neuron")
    parser.set("Neuron", "LIF_Mode", "analog_lut")
    parser.set("Neuron", "Analog_LIF_LUT_Root", str(lut_root))
    parser.set("Neuron", "Analog_LIF_LUT_Verify_Checksums", "1")
    parser.set("Neuron", "Analog_LIF_LUT_Profile", "synthetic")
    parser.set("Neuron", "Analog_LIF_Negative_Vmem_Policy", "error")
    parser.set("Static Power", "Static_Power_Enable", "1" if static_power_enable else "0")
    path = tmp_path / "SimConfig.analog.ini"
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)
    return str(path)


def _direct_payload():
    weights = [[1] * 16, [0] * 16, [0] * 16]
    return {
        "shape": {"C": 3},
        "analog_lif_payloads": {
            "L1": {
                "contract_version": 1,
                "electrical_layout": "closed_differential_outputs",
                "neuron_count": 16,
                "initial_vmem_v": [0.0] * 16,
                "timesteps": [
                    {
                        "source": "input_weight",
                        "input_values": [0.0, 0.0, 0.0],
                        "weight_codes_by_layer": {"Conv1": weights},
                    },
                    {
                        "source": "input_weight",
                        "input_values": [1.0, 0.0, 0.0],
                        "weight_codes_by_layer": {"Conv1": weights},
                    },
                    {
                        "source": "input_weight",
                        "input_values": [0.0, 0.0, 0.0],
                        "weight_codes_by_layer": {"Conv1": weights},
                    },
                ],
            }
        },
    }


def _execute(config_path, payload):
    backend = _Backend()
    report = execute_round_plan(
        _plan(),
        sim_config_path=config_path,
        ops_total=1,
        ops_count_policy="mac_as_1_op",
        static_backend=backend,
        model_inputs={"model_input": payload},
    )
    return report, backend


def test_payload_session_computes_currents_from_input_and_weight_metadata(tmp_path):
    parser = configparser.ConfigParser()
    parser["Device level"] = {
        "Device_Level": "3",
        "Device_Resistance_Sort": "0",
        "Device_Resistance": "1000,500,250",
        "Read_Voltage": "0.5",
    }
    parser["Process element level"] = {"Xbar_Polarity": "2"}
    parser["Algorithm Configuration"] = {"Weight_precision": "2"}
    config_path = tmp_path / "SimConfig.crossbar.ini"
    with config_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)

    payload = {
        "model_input": {
            "analog_lif_payloads": {
                "L1": {
                    "contract_version": 1,
                    "electrical_layout": "closed_differential_outputs",
                    "neuron_count": 2,
                    "initial_vmem_v": [0.0, 0.0],
                    "timesteps": [
                        {
                            "source": "input_weight",
                            "input_values": [2.0, 4.0],
                            "weight_codes_by_layer": {
                                "Conv1": [[1, -2], [2, 0]]
                            },
                        }
                    ],
                }
            }
        }
    }

    session = AnalogLifPayloadSession(
        payload,
        sim_config_path=str(config_path),
        weight_codes_by_layer=None,
    )
    numerical = session.resolve(
        sample_id="sample:0",
        node_id="L1",
        timestep=0,
        sequence_length=1,
        neuron_count=2,
    )

    assert numerical.source == "input_weight"
    assert numerical.ipos_a == pytest.approx([0.010, 0.003])
    assert numerical.ineg_a == pytest.approx([0.003, 0.006])
    assert numerical.crossbar_audit["engine"] == "local_v_at_g"


@pytest.mark.parametrize(
    "scope",
    ["timestep", "payload"],
)
def test_input_weight_payload_rejects_forbidden_external_numerical_fields(scope):
    payload = _direct_payload()
    forbidden = {"ipos_a": [0.0] * 16}
    if scope == "timestep":
        payload["analog_lif_payloads"]["L1"]["timesteps"][0].update(forbidden)
    else:
        payload["analog_lif_payloads"]["L1"].update(forbidden)

    session = AnalogLifPayloadSession({"model_input": payload})

    with pytest.raises(
        AnalogLifPayloadError, match="FORBIDDEN_ANALOG_NUMERICAL_FIELDS"
    ):
        session.resolve(
            sample_id="sample:0",
            node_id="L1",
            timestep=0,
            sequence_length=3,
            neuron_count=16,
        )


def test_analog_lut_dispatch_uses_real_payload_and_persists_measured_state(tmp_path):
    report, backend = _execute(_analog_config(tmp_path), _direct_payload())

    assert len(backend.calls) == 3
    assert [step["source_kind"] for step in report.step_results] == [
        "static_eval",
        "analog_lif",
        "static_eval",
        "analog_lif",
        "static_eval",
        "analog_lif",
    ]
    analog_steps = [
        step for step in report.step_results if step["source_kind"] == "analog_lif"
    ]
    assert [step["timestep"] for step in analog_steps] == [0, 1, 2]
    audits = [step["mapping_summary"]["analog_lif"] for step in analog_steps]
    assert audits[0]["backend"] == "local_analog_lif_lut"
    assert audits[0]["state"]["vmem0_v"] == pytest.approx([0.0] * 16)
    assert audits[1]["state"]["vmem0_v"] == pytest.approx(
        audits[0]["state"]["vmem_next_v"]
    )
    assert audits[2]["state"]["vmem0_v"] == pytest.approx(
        audits[1]["state"]["vmem_next_v"]
    )
    assert all(audit["query_counts"] == {"frontend": 16, "lif": 16} for audit in audits)
    assert all(audit["interpolation"] == "bounded_bilinear" for audit in audits)
    assert all(audit["checksum_verified"] is True for audit in audits)
    assert all(audit["axis_clamp_count"] == 0 for audit in audits)
    assert all(audit["state"]["payload_identity"].startswith("sha256:") for audit in audits)
    assert all(
        audit["boundary_binding"]["state_payload_identity"]
        == audit["state"]["payload_identity"]
        for audit in audits
    )
    boundary_report = report.item_results[0]["boundary_report"]
    output_bindings = boundary_report["output_state_bindings"]
    assert output_bindings
    assert all(
        binding["state_payload_identity"].startswith("sha256:")
        for binding in output_bindings
    )
    assert all(
        binding["spike_payload_identity"].startswith("sha256:")
        for binding in output_bindings
    )
    assert all(
        binding["spike_codes"] == audits[index]["state"]["spike_codes"]
        for index, binding in enumerate(output_bindings)
    )
    assert all(
        state["state_payload_identity"].startswith("sha256:")
        for state in boundary_report["created_states"]
    )
    assert all(audit["area_characterized"] is True for audit in audits)
    assert all(audit["area_status"] == "hspice_netlist_estimate" for audit in audits)
    assert all(audit["area_um2"] > 0 for audit in audits)
    assert json.loads(json.dumps(report.to_dict()))["step_results"][1][
        "mapping_summary"
    ]["analog_lif"]["state"]["vmem_next_v"] == pytest.approx(
        audits[0]["state"]["vmem_next_v"]
    )
    assert report.inference_metrics["lif_update_count"] == 48
    aggregate = report.inference_metrics["analog_lif"]
    assert aggregate["backend"] == "local_analog_lif_lut"
    assert aggregate["query_counts"] == {"frontend": 48, "lif": 48}
    assert aggregate["invocation_count"] == 3
    assert aggregate["axis_clamp_count"] == 0
    assert aggregate["area_characterized"] is True
    assert aggregate["area_um2"] > 0
    assert aggregate["energy_j"] == pytest.approx(
        sum(step["energy_j"] for step in analog_steps)
    )
    assert report.inference_metrics["energy_breakdown_j"][
        "analog_lif_energy_j"
    ] == pytest.approx(aggregate["energy_j"])
    assert report.inference_metrics["energy_breakdown_j"]["read_energy_j"] == pytest.approx(
        report.inference_metrics["read_energy_j"]
    )
    assert report.inference_metrics["accounting_scope"]["analog_frontend_lif"][
        "area_included"
    ] is True


def test_analog_lut_allows_static_power_ledger_with_adc_reconciliation(tmp_path):
    report, backend = _execute(
        _analog_config(tmp_path, static_power_enable=True), _direct_payload()
    )
    assert backend.calls
    assert report.inference_metrics["analog_accounting"]["reconciliation_passed"] is True


def test_analog_lut_dispatch_calculates_currents_from_internal_input_weight_payload(
    tmp_path,
):
    payload = _direct_payload()
    input_weight_step = {
        "source": "input_weight",
        "input_values": [1.0, 0.0, 0.0],
        "weight_codes_by_layer": {
            "Conv1": [[1] * 16, [0] * 16, [0] * 16],
        },
    }
    payload["analog_lif_payloads"]["L1"]["timesteps"] = [input_weight_step] * 3

    report, _backend = _execute(_analog_config(tmp_path), payload)

    audits = [
        step["mapping_summary"]["analog_lif"]
        for step in report.step_results
        if step["source_kind"] == "analog_lif"
    ]
    assert audits[0]["numerical_input"]["source"] == "input_weight"
    assert audits[0]["numerical_input"]["crossbar"]["engine"] == "local_v_at_g"
    assert audits[0]["numerical_input"]["ipos_a"] == pytest.approx(
        [4.0 / 567276196736.456] * 16
    )
    assert audits[0]["numerical_input"]["ineg_a"] == pytest.approx(
        [4.0 / 652273080549.48] * 16
    )


def test_metadata_only_analog_request_fails_closed_after_original_static_eval(tmp_path):
    backend = _Backend()

    with pytest.raises(Phase2Invalid, match="MISSING_ANALOG_LIF_PAYLOAD"):
        execute_round_plan(
            _plan(),
            sim_config_path=_analog_config(tmp_path),
            ops_total=1,
            ops_count_policy="mac_as_1_op",
            static_backend=backend,
            model_inputs={"model_input": {"shape": {"C": 3}}},
        )

    assert len(backend.calls) == 1


def test_analog_payload_rejects_unclosed_or_bitsliced_electrical_layout(tmp_path):
    payload = _direct_payload()
    payload["analog_lif_payloads"]["L1"]["electrical_layout"] = "bit_sliced_rails"

    with pytest.raises(Phase2Invalid, match="UNSUPPORTED_ANALOG_ELECTRICAL_LAYOUT"):
        _execute(_analog_config(tmp_path), payload)


def test_analog_payload_session_rejects_out_of_order_invocation_and_step_reset():
    payload = _direct_payload()
    session = AnalogLifPayloadSession({"model_input": payload})

    with pytest.raises(AnalogLifPayloadError, match="expected timestep 0"):
        session.resolve(
            sample_id="sample:0",
            node_id="L1",
            timestep=1,
            sequence_length=3,
            neuron_count=16,
        )

    payload["analog_lif_payloads"]["L1"]["timesteps"][0]["vmem0_v"] = [0.0] * 16
    session = AnalogLifPayloadSession({"model_input": payload})
    with pytest.raises(AnalogLifPayloadError, match="per-timestep"):
        session.resolve(
            sample_id="sample:0",
            node_id="L1",
            timestep=0,
            sequence_length=3,
            neuron_count=16,
        )


def test_analog_payload_session_rejects_successor_with_wrong_neuron_count(tmp_path):
    session = AnalogLifPayloadSession(
        {"model_input": _direct_payload()},
        sim_config_path=_analog_config(tmp_path),
    )
    session.resolve(
        sample_id="sample:0",
        node_id="L1",
        timestep=0,
        sequence_length=3,
        neuron_count=16,
    )

    with pytest.raises(AnalogLifPayloadError, match="expected 16"):
        session.commit(
            sample_id="sample:0",
            node_id="L1",
            timestep=0,
            expected_neuron_count=16,
            vmem_next_v=[0.0] * 15,
        )


def test_original_mnsim_adc_cost_is_retained_for_raw_audit(tmp_path):
    config_path = _analog_config(tmp_path)
    adc = ADC(config_path)
    assert adc.ADC_area > 0.0
    assert adc.ADC_power > 0.0
    assert adc.ADC_latency > 0.0
    assert adc.ADC_energy > 0.0
    adc.calculate_ADC_area()
    adc.calculate_ADC_power()
    adc.calculate_ADC_latency()
    adc.calculate_ADC_energy()

    assert adc.ADC_area > 0.0
    assert adc.ADC_power > 0.0
    assert adc.ADC_latency > 0.0
    assert adc.ADC_energy > 0.0
    assert adc.ADC_hardware_enabled is True

    pe = ProcessElement(config_path)
    assert pe.PE_group_ADC_num > 0
    assert pe.PE_ADC_num > 0
    assert pe.output_mux > 0
    pe.calculate_PE_area(SimConfig_path=config_path)
    assert pe.PE_ADC_area > 0.0

    latency = PE_latency_analysis(
        config_path,
        read_row=1,
        read_column=1,
        indata=1,
        rdata=1,
        inprecision=1,
    )
    assert latency.ADC_latency > 0.0


def test_analog_mode_auto_adc_count_uses_positive_virtual_lane_geometry(tmp_path):
    config_path = _analog_config(tmp_path)
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(config_path, encoding="UTF-8")
    parser.set("Process element level", "ADC_Num", "0")
    with open(config_path, "w", encoding="UTF-8") as handle:
        parser.write(handle)

    pe = ProcessElement(config_path)

    assert pe.PE_group_ADC_num > 0
    assert pe.PE_ADC_num > 0
    assert pe.output_mux > 0
    assert pe.ADC_area > 0.0
