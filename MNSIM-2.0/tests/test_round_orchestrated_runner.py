import copy
import configparser
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import MNSIM.Interface.round_orchestrated_runner as runner_module
from MNSIM.Interface.final_rram_contract import apply_final_rram_model
from MNSIM.Accuracy_Model.readc_artifact import (
    build_readc_artifact,
    validate_readc_artifact,
)
from MNSIM.Interface.round_orchestrated_artifacts import ArtifactStore
from MNSIM.Interface.round_orchestrated_runner import (
    INVALID_FINAL_RRAM_MODEL,
    RunnerDependencies,
    run_round_orchestrated_simulation,
)
from MNSIM.Interface.round_orchestrated_types import (
    RoundOrchestratedRunRequest,
    RunnerError,
)
from MNSIM.Scheduling.phase2.types import Phase2Invalid, Phase2Report
from MNSIM.Scheduling.round_types import (
    LayerIR,
    ModelIR,
    Phase1Invalid,
    RoundPlan,
    SequenceSpec,
)


class FakeNetwork:
    def __init__(self):
        self.layer_list = []

    def get_weights(self):
        # The injected Phase 2 executor does not inspect the matrices, but the
        # production runner must always hand it the real NetworkGraph contract.
        return [{}]


class FakeContext:
    def __init__(
        self,
        calls,
        *,
        fail_accuracy=False,
        print_output=False,
        original_accuracy=0.91,
        pim_accuracy=0.87,
    ):
        self.calls = calls
        self.fail_accuracy = fail_accuracy
        self.print_output = print_output
        self.original_accuracy = original_accuracy
        self.pim_accuracy = pim_accuracy
        self.net = FakeNetwork()

    def get_net_bits(self):
        self.calls.append("get_net_bits")
        if self.print_output:
            print("get-net-bits backend noise")
        return "original-bits"

    def origin_evaluate(self, *, method, adc_action):
        self.calls.append(f"origin:{method}:{adc_action}")
        if self.print_output:
            print("origin-evaluate backend noise")
        if self.fail_accuracy:
            raise ValueError("accuracy backend failed")
        return self.original_accuracy

    def set_net_bits_evaluate(self, updated_bits, *, adc_action):
        self.calls.append(f"set:{updated_bits}:{adc_action}")
        if self.print_output:
            print("set-net-bits backend noise")
        return self.pim_accuracy


class _FakeReADCArtifact:
    artifact_checksum = "c" * 64
    layers = {"L000": {"table_checksum": "d" * 64}}

    def to_dict(self):
        return {
            "schema_version": 1,
            "identity": {},
            "layers": self.layers,
            "artifact_checksum": self.artifact_checksum,
        }


class _ProfiledSNNContext(FakeContext):
    def __init__(self, calls, profile, *, lif_artifact=None):
        super().__init__(calls, original_accuracy=0.75, pim_accuracy=0.5)
        self.hardware_config = {"ADC_quantization_mode": "READC_LLOYD_MAX"}
        self.snn_accuracy_config = SimpleNamespace(
            hardware_profile=profile,
            lif_perturbation_seed=112,
        )
        self.lif_perturbation_artifact = lif_artifact
        self.net.layer_list = [
            SimpleNamespace(
                layer_config={
                    "type": "lif",
                    "tau": 50.5016666556,
                    "v_th": 0.01,
                    "v_reset": 0.0,
                }
            )
        ]

    def calibration_split_identity(self):
        self.calls.append("calibration_split_identity")
        return {"dataset": "fake-v7", "split": "train"}

    def calibrate_readc(self, value, *, identity, max_samples=None):
        self.calls.append(("calibrate_readc", value, identity, max_samples))
        return _FakeReADCArtifact()

    def install_readc_artifact(self, artifact, *, expected_identity):
        self.calls.append(("install_readc_artifact", artifact, expected_identity))

    def origin_evaluate(self, *, method, adc_action, **_kwargs):
        self.calls.append("origin")
        return self.original_accuracy

    def set_net_bits_evaluate(
        self,
        value,
        *,
        adc_action,
        pim_profile,
        lif_perturbation,
        **_kwargs,
    ):
        self.calls.append(
            ("pim_test", value, adc_action, pim_profile, lif_perturbation)
        )
        return self.pim_accuracy


def _install_formal_snn_runner_stubs(monkeypatch):
    monkeypatch.setattr(
        runner_module,
        "_load_accuracy_device_contract",
        lambda _path: SimpleNamespace(
            model_id="rram_stanford_asu_256",
            variation_percent=0.0,
            device_seed=112,
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "LIFPerturbationRuntime",
        lambda artifact, seed: ("lif-runtime", artifact, seed),
    )


class CheckpointLoadFailure(Exception):
    pass


class ContextConfigurationFailure(Exception):
    pass


class ObservedPlan:
    def __init__(self, events, *, feasible, invalid_reason=None):
        self.events = events
        self._feasible = feasible
        self.invalid_reason = invalid_reason

    @property
    def feasible(self):
        self.events.append("feasibility")
        return self._feasible

    def to_dict(self):
        self.events.append("snapshot")
        return {
            "plan_id": "observed-plan",
            "feasible": self._feasible,
            "invalid_reason": self.invalid_reason,
        }


def _model_ir(*, backend_kind="ann", layer_type="conv", shape=None):
    if shape is None:
        shape = {
            "Inputchannel": 2,
            "Inputsize": [4, 5],
            "Inputbit": 8,
            "Outputchannel": 3,
            "Kernelsize": 1,
            "Outputsize": [4, 5],
            "Depthwise": "normal",
        }
    return ModelIR(
        model_id="fake-network",
        backend_kind=backend_kind,
        sequence_spec=SequenceSpec(
            sequence_length=1,
            ordered_steps=False,
            cross_step_dependency=False,
            recurrent_state_ids=[],
        ),
        layers=[
            LayerIR(
                layer_id="L000_conv",
                layer_type=layer_type,
                is_weighted=True,
                stable_topological_index=0,
                input_tensor_ids=["input:0"],
                output_tensor_ids=["L000_conv:out"],
                input_layer_ids=[],
                output_consumer_ids=[],
                weight_ref_id="fake-network:L000_conv:weight",
                shape=shape,
                backend_attrs={"mnsim_layer_index": 0},
            )
        ],
    )


def _plan(*, feasible=True, invalid_reason=None):
    return RoundPlan(
        plan_id="plan:fake-network",
        model_id="fake-network",
        backend_kind="ann",
        hardware_budget={"xbar_size": [128, 128]},
        execution_items=[],
        layer_resource_descriptors=[],
        model_graph_ref={"backend": "fake"},
        feasible=feasible,
        invalid_reason=invalid_reason,
    )


def _phase2_report(*, inference_metrics=None):
    if inference_metrics is None:
        inference_metrics = {
            "latency_s": 2.0,
            "energy_j": 4.0,
            "power_avg_w": 2.0,
            "area_um2": 11.0,
        }
    return Phase2Report(
        model_name="fake-network",
        sim_config_path="fake-sim-config.ini",
        feasible=True,
        execution_item_count=0,
        inference_metrics=inference_metrics,
        item_results=[],
        step_results=[],
        raw_mnsim_audit=[],
        boundary_events=[],
    )


@pytest.fixture
def dependency_factory():
    def build(
        *,
        context=None,
        context_error=None,
        model=None,
        plan=None,
        report=None,
        accuracy_failure=False,
        original_accuracy=0.91,
        pim_accuracy=0.87,
        phase2_error=None,
        print_output=False,
    ):
        calls = []
        context = context or FakeContext(
            calls,
            fail_accuracy=accuracy_failure,
            print_output=print_output,
            original_accuracy=original_accuracy,
            pim_accuracy=pim_accuracy,
        )
        model = model or _model_ir()
        plan = plan or _plan()
        report = report or _phase2_report()
        executor_calls = []

        def interface_factory(**kwargs):
            calls.append(("context", kwargs))
            if print_output:
                print("context backend noise")
            if context_error is not None:
                raise context_error
            return context

        def weight_update(sim_config, net_bits, **controls):
            calls.append(("weight_update", sim_config, net_bits, controls))
            if print_output:
                print("weight-update backend noise")
            return "updated-bits"

        def model_ir_builder(net, *, model_id, backend_kind):
            calls.append(("model_ir_builder", net, model_id, backend_kind))
            if print_output:
                print("model-ir backend noise")
            return model

        def plan_builder(model_ir, sim_config):
            calls.append(("plan_builder", model_ir, sim_config))
            if print_output:
                print("plan-builder backend noise")
            return plan

        def phase1_projection_builder(sim_config_path):
            calls.append(("phase1_projection_builder", sim_config_path))
            return {"fake": "phase1-projection"}

        def phase2_executor(**kwargs):
            executor_calls.append(kwargs)
            calls.append(("phase2_executor", kwargs))
            if print_output:
                print("phase2 backend noise")
            if phase2_error is not None:
                raise phase2_error
            return report

        dependencies = RunnerDependencies(
            interface_factory=interface_factory,
            weight_update=weight_update,
            model_ir_builder=model_ir_builder,
            plan_builder=plan_builder,
            phase2_executor=phase2_executor,
            final_rram_validator=lambda _path: None,
            static_backend="fake-static-backend",
            checkpoint_error_types=(CheckpointLoadFailure,),
        )
        dependencies.phase1_projection_builder = phase1_projection_builder
        return dependencies, calls, executor_calls, context, plan, report

    return build


def _request(tmp_path, **changes):
    tmp_path.mkdir(parents=True, exist_ok=True)
    weights = tmp_path / "weights.pth"
    weights.write_bytes(b"fake weights")
    sim_config = tmp_path / "SimConfig.ini"
    sim_config.write_text("[fake]\nvalue = 1\n", encoding="utf-8")
    request = RoundOrchestratedRunRequest(
        network="fake-network",
        weights=weights,
        num_classes=10,
        sim_config=sim_config,
        output_dir=tmp_path / "output",
        dataset_module="fake-dataset",
    )
    return replace(request, **changes)


def _artifact(request, name):
    return json.loads((request.output_dir / name).read_text(encoding="utf-8"))


def _error_codes(result):
    return [error.code for error in result.errors]


def test_formal_snn_accuracy_rejects_finetune_calibration_before_inference(
    tmp_path, dependency_factory
):
    context = FakeContext([])
    context.calibration_split_identity = lambda: {
        "dataset": "fake-v7",
        "split": "train",
        "resolved_split": "finetune",
        "sample_count": 1,
    }
    dependencies, calls, _, context, _, _ = dependency_factory(context=context)
    context.calls = calls
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_pim=False,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 1
    assert _error_codes(result) == ["ACCURACY_FAILED"]
    assert not any(str(call).startswith("origin:") for call in calls)


def test_formal_snn_accuracy_rejects_empty_test_split_before_inference(
    tmp_path, dependency_factory
):
    context = FakeContext([])
    context.test_split_identity = lambda: {
        "dataset": "fake-v7",
        "split": "test",
        "resolved_split": "test",
        "sample_count": 0,
    }
    dependencies, calls, _, context, _, _ = dependency_factory(context=context)
    context.calls = calls
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_pim=False,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 1
    assert _error_codes(result) == ["ACCURACY_FAILED"]
    assert not any(str(call).startswith("origin:") for call in calls)


def test_runner_rejects_noncanonical_device_before_loading_network(tmp_path):
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)
    calls = []

    def interface_factory(**kwargs):
        calls.append(kwargs)
        raise AssertionError("network must not load before device validation")

    result = run_round_orchestrated_simulation(
        request,
        dependencies=RunnerDependencies(interface_factory=interface_factory),
    )

    assert result.exit_code == 2
    assert _error_codes(result) == [INVALID_FINAL_RRAM_MODEL]
    assert calls == []
    assert all(stage.status == "not_run" for stage in result.stage_statuses.values())


def test_runner_rejects_accuracy_only_device_before_hardware_network_load(tmp_path):
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)
    calls = []

    def interface_factory(**kwargs):
        calls.append(kwargs)
        raise AssertionError("network must not load for unsupported hardware profile")

    result = run_round_orchestrated_simulation(
        request,
        dependencies=RunnerDependencies(
            interface_factory=interface_factory,
            final_rram_validator=lambda _path: {
                "model_id": "rram_memtorch_stanford_pku_256",
                "hardware_write_accounting_supported": False,
            },
        ),
    )

    assert result.exit_code == 2
    assert _error_codes(result) == [INVALID_FINAL_RRAM_MODEL]
    assert "does not provide a complete source-to-target write transition model" in (
        result.errors[0].detail
    )
    assert calls == []
    assert all(stage.status == "not_run" for stage in result.stage_statuses.values())


def test_schedule_only_accepts_device_without_write_accounting(
    tmp_path, dependency_factory, monkeypatch
):
    dependencies, _, executor_calls, _, _, _ = dependency_factory()
    dependencies.final_rram_validator = lambda _path: {
        "model_id": "rram_stanford_asu_256",
        "hardware_write_accounting_supported": False,
    }
    schedule = SimpleNamespace(policy_id="seq", rounds=[], sha256=lambda: "a" * 64)
    monkeypatch.setattr(
        runner_module, "_build_weight_schedule_artifact", lambda **_kwargs: schedule
    )

    def save_schedule(path, _artifact):
        path.write_text("frozen schedule", encoding="utf-8")
        return path

    monkeypatch.setattr(runner_module, "save_schedule_artifact", save_schedule)
    request = _request(
        tmp_path,
        skip_accuracy=True,
        run_hardware=True,
        schedule_only=True,
        schedule_plan_out=tmp_path / "schedule.json",
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert result.stage_statuses["hardware"].status == "succeeded"
    assert executor_calls == []


def _write_projection_sim_config(
    path,
    *,
    device_level=2,
    phase1_device_level=None,
    weight_precision=8,
    xbar_size="128,128",
):
    parser = configparser.ConfigParser()
    parser["Device level"] = {"Device_Level": str(device_level)}
    parser["Crossbar level"] = {"Xbar_Size": xbar_size}
    parser["Process element level"] = {
        "Xbar_Polarity": "2",
        "Group_Num": "2",
    }
    parser["Tile level"] = {"PE_Num": "3,4"}
    parser["Architecture level"] = {"Tile_Num": "5,6"}
    parser["Algorithm Configuration"] = {}
    if weight_precision is not None:
        parser["Algorithm Configuration"]["Weight_precision"] = str(weight_precision)
    if phase1_device_level is not None:
        parser["Algorithm Configuration"]["Phase1_Device_Level"] = str(
            phase1_device_level
        )
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)


def _add_final_rram_provenance(path):
    parser = configparser.ConfigParser()
    parser.read(path, encoding="UTF-8")
    apply_final_rram_model(parser, "rram_stanford_asu_256")
    with path.open("w", encoding="UTF-8") as handle:
        parser.write(handle)


def test_final_rram_provenance_is_reported_in_hardware_summary(
    tmp_path, dependency_factory
):
    dependencies, _, _, _, _, _ = dependency_factory()
    request = _request(
        tmp_path,
        skip_accuracy=True,
        run_hardware=True,
    )
    _add_final_rram_provenance(request.sim_config)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    model = result.hardware_summary["final_rram_model"]
    assert model["model_id"] == "rram_stanford_asu_256"
    assert model["model_json"].endswith("rram_stanford_asu_256.json")


def test_code_root_sim_config_has_usable_phase1_weight_precision():
    sim_config = Path(__file__).resolve().parents[1] / "SimConfig.ini"

    projection = runner_module._build_phase1_projection(sim_config)

    assert projection["Weight_precision"] > 0


@pytest.mark.parametrize(
    ("weight_precision", "phase1_device_level", "xbar_size"),
    [
        (None, None, "128,128"),
        (0, None, "128,128"),
        (-1, None, "128,128"),
        ("not-an-integer", None, "128,128"),
        (8, None, "128"),
    ],
    ids=[
        "missing-weight-precision",
        "zero-weight-precision",
        "negative-weight-precision",
        "noninteger-weight-precision",
        "malformed-xbar-size",
    ],
)
def test_invalid_phase1_projection_fails_before_plan_or_phase2(
    tmp_path,
    dependency_factory,
    weight_precision,
    phase1_device_level,
    xbar_size,
):
    dependencies, calls, executor_calls, _, _, _ = dependency_factory()
    dependencies.phase1_projection_builder = runner_module._build_phase1_projection
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)
    _write_projection_sim_config(
        request.sim_config,
        weight_precision=weight_precision,
        phase1_device_level=phase1_device_level,
        xbar_size=xbar_size,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["PHASE1_INVALID"]
    assert not any(call[0] == "plan_builder" for call in calls)
    assert executor_calls == []


@pytest.mark.parametrize(
    "phase1_device_level",
    [256, "not-an-integer", None],
    ids=["legacy-override", "invalid-legacy-override", "absent"],
)
def test_hardware_builds_phase1_projection_from_physical_device_level(
    tmp_path,
    dependency_factory,
    phase1_device_level,
):
    dependencies, calls, executor_calls, _, _, _ = dependency_factory()
    dependencies.phase1_projection_builder = runner_module._build_phase1_projection
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)
    _write_projection_sim_config(
        request.sim_config,
        phase1_device_level=phase1_device_level,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    plan_call = next(call for call in calls if call[0] == "plan_builder")
    assert plan_call[2] == {
        "Xbar_Size": [128, 128],
        "Tile_Num": [5, 6],
        "PE_Num": [3, 4],
        "Group_Num": 2,
        "Device_Level": 2,
        "Weight_precision": 8,
        "Xbar_Polarity": 2,
    }
    context_call = next(call for call in calls if call[0] == "context")
    assert context_call[1]["SimConfig_path"] == str(request.sim_config)
    assert executor_calls[0]["sim_config_path"] == str(request.sim_config)


def test_hardware_seeds_conv_root_model_input_for_phase2(tmp_path, dependency_factory):
    dependencies, _, executor_calls, _, _, _ = dependency_factory()
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert executor_calls[0]["model_inputs"] == {
        "model_input": {
            "tensor_role": "activation",
            "shape": {
                "Inputchannel": 2,
                "Inputsize": [4, 5],
                "Inputbit": 8,
            },
        }
    }


def test_hardware_seeds_fc_root_model_input_for_phase2(tmp_path, dependency_factory):
    model = _model_ir(
        layer_type="fc",
        shape={"Infeature": 12, "Outfeature": 3, "Inputbit": 7},
    )
    dependencies, _, executor_calls, _, _, _ = dependency_factory(model=model)
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert executor_calls[0]["model_inputs"] == {
        "model_input": {
            "tensor_role": "activation",
            "shape": {"Infeature": 12, "Inputbit": 7},
        }
    }


def test_fc_like_root_model_input_metadata_is_supported():
    model = _model_ir(
        layer_type="fc_like_projection",
        shape={"Infeature": 12, "Inputbit": 7},
    )

    assert runner_module._model_inputs_from_root_layers(model) == {
        "model_input": {
            "tensor_role": "activation",
            "shape": {"Infeature": 12, "Inputbit": 7},
        }
    }


@pytest.mark.parametrize(
    "case",
    ["missing", "malformed", "unsupported", "incompatible"],
)
def test_invalid_root_model_input_metadata_fails_before_phase2(
    tmp_path,
    dependency_factory,
    case,
):
    model = _model_ir()
    root = model.layers[0]
    if case == "missing":
        model = replace(model, layers=[replace(root, shape={**root.shape, "Inputbit": None})])
    elif case == "malformed":
        model = replace(model, layers=[replace(root, shape={**root.shape, "Inputsize": [4]})])
    elif case == "unsupported":
        model = replace(
            model,
            layers=[
                replace(
                    root,
                    layer_type="pooling",
                    is_weighted=False,
                    weight_ref_id=None,
                    shape={},
                )
            ],
        )
    else:
        model = replace(
            model,
            layers=[
                root,
                replace(
                    root,
                    layer_id="L001_conv",
                    output_tensor_ids=["L001_conv:out"],
                    shape={**root.shape, "Inputchannel": 3},
                ),
            ],
        )
    dependencies, _, executor_calls, _, _, _ = dependency_factory(model=model)
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["PHASE1_INVALID"]
    assert executor_calls == []


def test_hardware_only_constructs_one_context_without_dataset_and_skips_accuracy(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    context_calls = [call for call in calls if call[0] == "context"]
    assert result.status == "succeeded"
    assert result.exit_code == 0
    assert len(context_calls) == 1
    assert context_calls[0][1]["dataset_module"] is None
    assert "accuracy" not in result.stage_statuses
    assert result.stage_statuses["hardware"].status == "succeeded"
    assert result.accuracy_report is None


def test_default_accuracy_preserves_original_evaluation_sequence_and_report_schema(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "succeeded"
    assert calls[1:] == [
        "get_net_bits",
        (
            "weight_update",
            str(request.sim_config),
            "original-bits",
            {"is_SAF": 1, "is_Variation": 0, "is_Rratio": 0},
        ),
        "origin:FIX_TRAIN:SCALE",
        "set:updated-bits:SCALE",
    ]
    report = _artifact(request, "accuracy_report.json")
    assert report == {
        "schema_version": 1,
        "status": "succeeded",
        "Original accuracy": 0.91,
        "PIM-based computing accuracy": 0.87,
        "adc_action": "SCALE",
        "controls": {
            "enable_saf": True,
            "enable_variation": False,
            "enable_r_ratio": False,
            "accuracy_seed": None,
            "fixed_quantization_range": False,
        },
    }


def test_accuracy_seed_scopes_weight_update_and_restores_global_numpy_state(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    samples = []

    def seeded_weight_update(sim_config, net_bits, **controls):
        samples.append(float(np.random.random_sample()))
        calls.append(("weight_update", sim_config, net_bits, controls))
        return "updated-bits"

    dependencies.weight_update = seeded_weight_update
    request = _request(tmp_path, accuracy_seed=20260809)
    np.random.seed(2468)
    state_before = np.random.get_state()

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    state_after = np.random.get_state()
    assert result.exit_code == 0
    assert samples == [np.random.RandomState(20260809).random_sample()]
    assert state_after[0] == state_before[0]
    assert state_after[2:] == state_before[2:]
    assert np.array_equal(state_after[1], state_before[1])
    assert _artifact(request, "accuracy_report.json")["controls"]["accuracy_seed"] == 20260809


def test_accuracy_can_run_original_without_pim(tmp_path, dependency_factory):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, run_original=True, run_pim=False)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert calls[1:] == ["origin:FIX_TRAIN:SCALE"]
    report = _artifact(request, "accuracy_report.json")
    assert report["Original accuracy"] == 0.91
    assert report["PIM-based computing accuracy"] is None
    assert report["evaluated_paths"] == {"original": True, "pim": False}


def test_snn_original_accuracy_uses_tradition_and_full_report_name(tmp_path):
    calls = []
    context = FakeContext(calls, original_accuracy=0.9935)
    context.net.layer_list = [
        type(
            "FakeLIFLayer",
            (),
            {
                "layer_config": {
                    "type": "lif",
                    "tau": 50.5016666556,
                    "v_th": 0.01,
                    "v_reset": 0.0,
                }
            },
        )()
    ]
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_original=True,
        run_pim=False,
    )

    report = runner_module._evaluate_accuracy(
        request, context, RunnerDependencies()
    )

    assert calls == ["origin:TRADITION:SCALE"]
    assert report["SNN Original Accuracy"] == 0.9935
    assert report["SNN Processing-in-Memory Accuracy"] is None
    assert report["adc_action"] is None
    assert report["snn_original_path"] == {
        "weighted_backend": "pytorch_float",
        "weight_quantization": "none",
        "activation_quantization": "none",
        "neuron_model": "spikingjelly.activation_based.neuron.LIFNode",
        "spikingjelly_step_mode": "m",
        "readout": "timestep_spike_sum",
        "lif": {
            "tau": 50.5016666556,
            "v_threshold": 0.01,
            "v_reset": 0.0,
            "decay_input": True,
            "reset_mode": "hard",
        },
    }


def test_accuracy_can_run_pim_without_original(tmp_path, dependency_factory):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, run_original=False, run_pim=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert calls[1:] == [
        "get_net_bits",
        (
            "weight_update",
            str(request.sim_config),
            "original-bits",
            {"is_SAF": 1, "is_Variation": 0, "is_Rratio": 0},
        ),
        "set:updated-bits:SCALE",
    ]
    report = _artifact(request, "accuracy_report.json")
    assert report["Original accuracy"] is None
    assert report["PIM-based computing accuracy"] == 0.87
    assert report["evaluated_paths"] == {"original": False, "pim": True}


def test_formal_snn_pim_uses_locked_device_controls_and_seed_112(
    tmp_path, dependency_factory, monkeypatch
):
    context = FakeContext([])
    context.hardware_config = {"ADC_quantization_mode": "UNIFORM"}
    dependencies, calls, _, context, _, _ = dependency_factory(context=context)
    context.calls = calls
    observed = []
    rng_samples = []

    monkeypatch.setattr(
        runner_module,
        "_load_accuracy_device_contract",
        lambda _path: SimpleNamespace(
            model_id="rram_stanford_asu_256",
            variation_percent=0.0,
            device_seed=112,
        ),
    )

    def capture_weight_update(sim_config, net_bits, **kwargs):
        observed.append(kwargs)
        rng_samples.append(float(kwargs["rng"].normal()))
        return "post-variation-bits"

    dependencies.weight_update = capture_weight_update
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_original=False,
        enable_saf=True,
        enable_variation=False,
        enable_r_ratio=False,
        accuracy_seed=None,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert len(observed) == 1
    assert observed[0]["is_SAF"] == 0
    assert observed[0]["is_Variation"] == 1
    assert observed[0]["is_Rratio"] == 1
    assert isinstance(observed[0]["rng"], np.random.Generator)
    assert rng_samples == [float(np.random.default_rng(112).normal())]
    report = _artifact(request, "accuracy_report.json")
    assert report["accuracy_device_model"] == "rram_stanford_asu_256"
    assert report["accuracy_device_variation_percent"] == pytest.approx(
        0.0
    )
    assert report["accuracy_device_seed"] == 112
    assert report["accuracy_variation_sampling_scope"] == (
        "once_per_physical_cell_per_pim_accuracy_run"
    )
    assert report["controls"] == {
        "enable_saf": False,
        "enable_variation": True,
        "enable_r_ratio": True,
        "accuracy_seed": 112,
        "source": "final_rram_contract",
    }


def test_formal_snn_reuses_one_post_variation_mapping_for_calibration_and_test(
    tmp_path, dependency_factory, monkeypatch
):
    context = FakeContext([])
    context.hardware_config = {"ADC_quantization_mode": "READC_LLOYD_MAX"}
    dependencies, calls, _, context, _, _ = dependency_factory(context=context)
    context.calls = calls
    monkeypatch.setattr(
        runner_module,
        "_load_accuracy_device_contract",
        lambda _path: SimpleNamespace(
            model_id="rram_stanford_asu_256",
            variation_percent=0.0,
            device_seed=112,
        ),
    )
    mapped = [{"weight": np.array([[1, 0], [0, 1]], dtype=np.int8)}]
    weight_update_calls = []

    def update_once(*_args, **_kwargs):
        weight_update_calls.append((_args, _kwargs))
        return mapped

    dependencies.weight_update = update_once
    observed = {"events": []}
    split_identity = {
        "dataset": "fake.v7",
        "split": "train",
        "sample_count": 1,
        "sample_manifest_sha256": "c" * 64,
    }
    context.calibration_split_identity = lambda: split_identity
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_original=False,
    )

    def calibrate(value, *, identity, **_kwargs):
        observed["events"].append("calibrate")
        observed["calibration"] = value
        observed["identity"] = identity
        return build_readc_artifact(
            identity,
            {
                "L000": {
                    "boundaries": [float(index) for index in range(31)],
                    "levels": [float(index) + 0.5 for index in range(32)],
                    "observed_count": 64,
                }
            },
        )

    def install(artifact, *, expected_identity):
        observed["events"].append("install")
        artifact_path = request.output_dir / "readc_calibration.json"
        assert artifact_path.is_file()
        assert json.loads(artifact_path.read_text(encoding="utf-8")) == (
            artifact.to_dict()
        )
        validate_readc_artifact(artifact.to_dict(), expected_identity)
        observed["installed"] = True

    def evaluate(value, **_kwargs):
        observed["events"].append("test")
        assert (request.output_dir / "readc_calibration.json").is_file()
        assert observed.get("installed") is True
        observed["test"] = value
        return 0.87

    context.calibrate_readc = calibrate
    context.install_readc_artifact = install
    context.set_net_bits_evaluate = evaluate

    result = run_round_orchestrated_simulation(
        request,
        dependencies=dependencies,
    )

    assert result.exit_code == 0
    assert len(weight_update_calls) == 1
    assert observed["calibration"] is mapped
    assert observed["test"] is mapped
    assert observed["events"] == ["calibrate", "install", "test"]
    assert observed["identity"].checkpoint_sha256 == runner_module.file_sha256(
        request.weights
    )
    assert observed["identity"].dataset_split_identity == split_identity
    assert observed["identity"].device_model_id == "rram_stanford_asu_256"
    assert observed["identity"].device_seed == 112
    assert observed["identity"].weight_mapping_sha256 == (
        runner_module.bit_weights_sha256(mapped)
    )
    assert observed["identity"].adc_precision == 5
    assert observed["identity"].percentile_low == 0.5
    assert observed["identity"].percentile_high == 99.5
    assert observed["identity"].max_samples_per_layer == 8000
    assert _artifact(request, "readc_calibration.json")["artifact_checksum"]
    report = _artifact(request, "accuracy_report.json")
    assert "_readc_artifact" not in report
    assert "layers" not in report["adc_model"]


def test_formal_snn_readc_identity_mismatch_blocks_test_inference(
    tmp_path, dependency_factory, monkeypatch
):
    context = FakeContext([])
    context.hardware_config = {"ADC_quantization_mode": "READC_LLOYD_MAX"}
    dependencies, calls, _, context, _, _ = dependency_factory(context=context)
    context.calls = calls
    monkeypatch.setattr(
        runner_module,
        "_load_accuracy_device_contract",
        lambda _path: SimpleNamespace(
            model_id="rram_stanford_asu_256",
            variation_percent=0.0,
            device_seed=112,
        ),
    )
    mapped = [{"weight": np.array([1], dtype=np.int8)}]
    dependencies.weight_update = lambda *_args, **_kwargs: mapped
    context.calibration_split_identity = lambda: {
        "dataset": "fake.v7",
        "split": "train",
    }
    test_calls = []

    def calibrate(_value, *, identity, **_kwargs):
        mismatched = replace(identity, checkpoint_sha256="d" * 64)
        return build_readc_artifact(
            mismatched,
            {
                "L000": {
                    "boundaries": [float(index) for index in range(31)],
                    "levels": [float(index) + 0.5 for index in range(32)],
                    "observed_count": 64,
                }
            },
        )

    def install(artifact, *, expected_identity):
        validate_readc_artifact(artifact.to_dict(), expected_identity)

    context.calibrate_readc = calibrate
    context.install_readc_artifact = install
    context.set_net_bits_evaluate = lambda *_args, **_kwargs: test_calls.append(True)
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_original=False,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 1
    assert _error_codes(result) == ["ACCURACY_FAILED"]
    assert test_calls == []
    assert (request.output_dir / "readc_calibration.json").is_file()


def test_formal_snn_pim_rejects_explicit_seed_other_than_112(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_original=False,
        accuracy_seed=113,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["INVALID_ACCURACY_SEED"]
    assert result.errors[0].detail == "formal SNN PIM accuracy seed is fixed to 112"
    assert calls == []


def test_adc_profile_calibrates_installs_then_tests_and_never_loads_lif_artifact(
    tmp_path, dependency_factory, monkeypatch
):
    _install_formal_snn_runner_stubs(monkeypatch)
    calls = []
    context = _ProfiledSNNContext(calls, "adc_digital_lif")
    dependencies, _factory_calls, _, _, _, _ = dependency_factory(context=context)
    mapped = object()
    dependencies.weight_update = lambda *_args, **_kwargs: mapped
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
    )
    result = run_round_orchestrated_simulation(request, dependencies=dependencies)
    report = _artifact(request, "accuracy_report.json")
    call_names = [
        item if isinstance(item, str) else item[0]
        for item in calls
    ]
    assert result.exit_code == 0
    assert call_names == [
        "get_net_bits",
        "calibration_split_identity",
        "calibrate_readc",
        "install_readc_artifact",
        "origin",
        "pim_test",
    ]
    assert report["accuracy_adc_model"] == "READC_LLOYD_MAX"
    assert report["accuracy_lif_perturbation_profile"] is None


def test_lut_profile_skips_every_readc_call_and_requires_qualified_perturbation(
    tmp_path, dependency_factory, monkeypatch
):
    _install_formal_snn_runner_stubs(monkeypatch)
    calls = []
    artifact = SimpleNamespace(
        profile_id="frontend_lut_lif_lut", artifact_checksum="e" * 64
    )
    context = _ProfiledSNNContext(
        calls, "frontend_lut_lif_lut", lif_artifact=artifact
    )
    dependencies, _factory_calls, _, _, _, _ = dependency_factory(context=context)
    dependencies.weight_update = lambda *_args, **_kwargs: object()
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
    )
    result = run_round_orchestrated_simulation(request, dependencies=dependencies)
    report = _artifact(request, "accuracy_report.json")
    call_names = [item if isinstance(item, str) else item[0] for item in calls]
    assert result.exit_code == 0
    assert "calibrate_readc" not in call_names
    assert "install_readc_artifact" not in call_names
    assert report["accuracy_adc_model"] is None
    assert report["accuracy_lif_perturbation_profile"] == artifact.profile_id


def test_original_accuracy_is_profile_independent(
    tmp_path, dependency_factory, monkeypatch
):
    reports = []
    for index, profile in enumerate(
        ("adc_digital_lif", "frontend_lut_lif_lut")
    ):
        calls = []
        context = _ProfiledSNNContext(calls, profile)
        dependencies, _, _, _, _, _ = dependency_factory(context=context)
        request = _request(
            tmp_path / str(index),
            network="standard_vgg8_snn_v7_phase6",
            run_original=True,
            run_pim=False,
        )
        result = run_round_orchestrated_simulation(
            request, dependencies=dependencies
        )
        assert result.exit_code == 0
        reports.append(_artifact(request, "accuracy_report.json"))
    adc_report, lut_report = reports
    assert adc_report["SNN Original Accuracy"] == lut_report["SNN Original Accuracy"]


def test_formal_lut_pim_without_qualified_artifact_fails_accuracy_stage(
    tmp_path, dependency_factory, monkeypatch
):
    _install_formal_snn_runner_stubs(monkeypatch)
    context = _ProfiledSNNContext([], "frontend_lut_lif_lut")
    dependencies, _, _, _, _, _ = dependency_factory(context=context)
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_original=False,
    )
    result = run_round_orchestrated_simulation(request, dependencies=dependencies)
    assert result.exit_code == 1
    assert result.errors[0].code == "ACCURACY_FAILED"


def test_lut_pim_explicit_no_perturbation_mode_runs_without_artifact(
    tmp_path, dependency_factory, monkeypatch
):
    _install_formal_snn_runner_stubs(monkeypatch)
    calls = []
    context = _ProfiledSNNContext(calls, "frontend_lut_lif_lut")
    context.snn_accuracy_config.lif_perturbation_mode = "none"
    dependencies, _, _, _, _, _ = dependency_factory(context=context)
    dependencies.weight_update = lambda *_args, **_kwargs: object()
    request = _request(
        tmp_path,
        network="standard_vgg8_snn_v7_phase6",
        run_original=False,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)
    report = _artifact(request, "accuracy_report.json")

    assert result.exit_code == 0
    assert report["accuracy_adc_model"] is None
    assert report["accuracy_lif_perturbation_profile"] == "none"
    pim_call = next(item for item in calls if isinstance(item, tuple) and item[0] == "pim_test")
    assert pim_call[-1] is None


def test_accuracy_rejects_disabling_both_paths(tmp_path, dependency_factory):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, run_original=False, run_pim=False)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["INVALID_STAGE_SELECTION"]
    assert calls == []


@pytest.mark.parametrize("accuracy_seed", [True, -1, 2**32])
def test_invalid_accuracy_seed_fails_before_context(
    tmp_path, dependency_factory, accuracy_seed
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, accuracy_seed=accuracy_seed)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["INVALID_ACCURACY_SEED"]
    assert calls == []


def test_accuracy_uses_fix_action_when_fixed_quantization_range_is_requested(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, fixed_quantization_range=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert "origin:FIX_TRAIN:FIX" in calls
    assert "set:updated-bits:FIX" in calls
    assert _artifact(request, "accuracy_report.json")["adc_action"] == "FIX"


def test_no_requested_stage_is_an_input_failure_with_manifest_and_summary(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, skip_accuracy=True, run_hardware=False)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 2
    assert _error_codes(result) == ["INVALID_STAGE_SELECTION"]
    assert calls == []
    assert _artifact(request, "run_manifest.json")["status"] == "failed"
    assert _artifact(request, "summary.json")["status"] == "failed"


@pytest.mark.parametrize("num_classes", [True, 1.5, 0, -3])
def test_invalid_num_classes_is_rejected_before_context(
    tmp_path, dependency_factory, num_classes
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, num_classes=num_classes)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["INVALID_NUM_CLASSES"]
    assert calls == []
    assert result.stage_statuses["accuracy"].status == "not_run"
    assert (request.output_dir / "summary.json").exists()


def test_accuracy_requires_a_nonempty_dataset_module_before_context(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, dataset_module="")

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["DATASET_MODULE_REQUIRED"]
    assert calls == []


def test_missing_checkpoint_is_rejected_before_context(tmp_path, dependency_factory):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path)
    request.weights.unlink()

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["CHECKPOINT_LOAD_FAILED"]
    assert calls == []


def test_missing_sim_config_is_rejected_before_context(tmp_path, dependency_factory):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path)
    request.sim_config.unlink()

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert _error_codes(result) == ["PHASE1_INVALID"]
    assert calls == []


def test_checkpoint_and_context_errors_use_stable_separate_mappings(
    tmp_path, dependency_factory
):
    checkpoint_dependencies, checkpoint_calls, _, _, _, _ = dependency_factory(
        context_error=CheckpointLoadFailure("malformed test checkpoint")
    )
    checkpoint_result = run_round_orchestrated_simulation(
        _request(tmp_path / "checkpoint"), dependencies=checkpoint_dependencies
    )

    config_dependencies, config_calls, _, _, _, _ = dependency_factory(
        context_error=ContextConfigurationFailure("invalid test configuration")
    )
    config_result = run_round_orchestrated_simulation(
        _request(tmp_path / "config"), dependencies=config_dependencies
    )

    assert checkpoint_result.exit_code == 2
    assert _error_codes(checkpoint_result) == ["CHECKPOINT_LOAD_FAILED"]
    assert config_result.exit_code == 2
    assert _error_codes(config_result) == ["PHASE1_INVALID"]
    assert len(checkpoint_calls) == 1
    assert len(config_calls) == 1


@pytest.mark.parametrize(
    ("message", "expected_code"),
    [
        (
            "PytorchStreamReader failed reading zip archive: failed finding central directory",
            "CHECKPOINT_LOAD_FAILED",
        ),
        ("Invalid magic number; corrupt file?", "CHECKPOINT_LOAD_FAILED"),
        ("INVALID PROTOCOL VERSION", "CHECKPOINT_LOAD_FAILED"),
        (
            "PytorchStreamReader failed reading file data/0",
            "CHECKPOINT_LOAD_FAILED",
        ),
        ("invalid device selection", "PHASE1_INVALID"),
    ],
)
def test_default_checkpoint_classifier_maps_only_known_torch_runtime_errors(
    tmp_path, message, expected_code
):
    calls = []

    def runtime_failure(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(message)

    dependencies = RunnerDependencies(
        interface_factory=runtime_failure,
        final_rram_validator=lambda _path: None,
    )
    result = run_round_orchestrated_simulation(
        _request(tmp_path), dependencies=dependencies
    )

    assert result.exit_code == 2
    assert _error_codes(result) == [expected_code]
    assert len(calls) == 1


def test_output_open_failure_has_no_artifacts_or_context(
    tmp_path, monkeypatch, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory()
    request = _request(tmp_path)

    def fail_open(cls, output_dir, overwrite):
        raise RunnerError("INVALID_OUTPUT_DIRECTORY", "test output failure")

    monkeypatch.setattr(runner_module.ArtifactStore, "open", classmethod(fail_open))

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 2
    assert _error_codes(result) == ["INVALID_OUTPUT_DIRECTORY"]
    assert result.artifacts == {}
    assert calls == []


def test_actual_lif_layer_is_fatal_hardware_preparation_failure(
    tmp_path, dependency_factory
):
    dependencies, _, _, _, _, _ = dependency_factory(
        model=_model_ir(backend_kind="ann", layer_type="lif")
    )
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 2
    assert result.stage_statuses["hardware"].error.code == "UNSUPPORTED_LAYER_TYPE"
    assert _error_codes(result) == ["UNSUPPORTED_LAYER_TYPE"]


def test_hardware_passes_the_exact_plan_and_mac_policy_to_phase2(
    tmp_path, dependency_factory
):
    plan = _plan()
    dependencies, _, executor_calls, _, returned_plan, _ = dependency_factory(plan=plan)
    request = _request(
        tmp_path,
        skip_accuracy=True,
        run_hardware=True,
        disable_inner_pipeline=True,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert result.round_plan is returned_plan is plan
    assert executor_calls[0]["plan"] is plan
    assert executor_calls[0]["ops_count_policy"] == "mac_as_2_ops"
    assert executor_calls[0]["latency_api"] == "nopipe"
    assert result.hardware_summary["macs_total"] == 120
    assert result.hardware_summary["ops_total"] == 240
    assert result.hardware_summary["ops_count_policy"] == "mac_as_2_ops"
    assert _artifact(request, "round_plan.json") == {
        "schema_version": 1,
        "status": "succeeded",
        "round_plan": plan.to_dict(),
    }


def test_hardware_defaults_to_inner_pipeline(tmp_path, dependency_factory):
    dependencies, _, executor_calls, _, _, _ = dependency_factory()
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert executor_calls[0]["latency_api"] == "pipe"


def test_infer_with_schedule_restores_plan_without_calling_planner(
    tmp_path, dependency_factory, monkeypatch
):
    plan = _plan()
    dependencies, calls, executor_calls, _, _, report = dependency_factory(plan=plan)
    artifact_path = tmp_path / "frozen.schedule.json"
    artifact_path.write_text("frozen", encoding="utf-8")

    class FrozenArtifact:
        round_plan_snapshot = plan.to_dict()
        execution_compatible = True
        migration_metadata = {}
        base_round_plan_sha256 = runner_module.canonical_sha256(plan)
        model_ir_sha256 = runner_module.canonical_sha256(_model_ir())
        sim_config_sha256 = runner_module.file_sha256(_request(tmp_path).sim_config)
        weights_sha256 = runner_module.canonical_sha256(
            {"checkpoint_sha256": runner_module.file_sha256(_request(tmp_path).weights)}
        )
        rounds = []

        def validate(self):
            return None

    monkeypatch.setattr(
        runner_module,
        "load_schedule_artifact",
        lambda _path: FrozenArtifact(),
    )
    dependencies.plan_builder = lambda *_args: pytest.fail(
        "Stage 2 must not invoke the Phase 1 planner"
    )
    request = _request(
        tmp_path,
        skip_accuracy=True,
        run_hardware=True,
        infer_with_schedule=True,
        schedule_plan_in=artifact_path,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert calls.count(("plan_builder",)) == 0
    assert executor_calls[0]["schedule_artifact"].__class__.__name__ == "FrozenArtifact"
    assert executor_calls[0]["plan"].to_dict() == plan.to_dict()


def test_infer_with_migrated_audit_artifact_is_rejected_before_planning(
    tmp_path, dependency_factory, monkeypatch
):
    dependencies, calls, executor_calls, _, _, _ = dependency_factory()
    artifact_path = tmp_path / "legacy-audit.schedule.json"
    artifact_path.write_text("audit", encoding="utf-8")

    class AuditArtifact:
        execution_compatible = False
        migration_metadata = {
            "incompatibility_reason": "legacy plan lacks current placement snapshots"
        }

    monkeypatch.setattr(
        runner_module,
        "load_schedule_artifact",
        lambda _path: AuditArtifact(),
    )
    dependencies.plan_builder = lambda *_args: pytest.fail(
        "audit-only artifacts must be rejected before planning"
    )
    request = _request(
        tmp_path,
        skip_accuracy=True,
        run_hardware=True,
        infer_with_schedule=True,
        schedule_plan_in=artifact_path,
    )

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert result.stage_statuses["hardware"].status == "failed"
    assert "audit-only" in result.stage_statuses["hardware"].error.detail
    assert "placement snapshots" in result.stage_statuses["hardware"].error.detail
    assert calls.count(("plan_builder",)) == 0
    assert executor_calls == []


def test_accuracy_failure_does_not_block_hardware_and_summary_keeps_only_real_partial_data(
    tmp_path, dependency_factory
):
    dependencies, calls, executor_calls, _, _, _ = dependency_factory(accuracy_failure=True)
    request = _request(tmp_path, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.stage_statuses["accuracy"].error.code == "ACCURACY_FAILED"
    assert result.stage_statuses["hardware"].status == "succeeded"
    assert len(executor_calls) == 1
    assert _artifact(request, "accuracy_report.json")["error"]["code"] == "ACCURACY_FAILED"
    summary = _artifact(request, "summary.json")
    assert summary["accuracy"] is None
    assert summary["hardware"]["macs_total"] == 120
    assert "Original accuracy" not in summary
    assert any(call[0] == "phase2_executor" for call in calls if isinstance(call, tuple))


def test_accuracy_artifact_serialization_failure_records_stage_failure_and_runs_hardware(
    tmp_path, dependency_factory
):
    dependencies, _, executor_calls, _, _, _ = dependency_factory(
        original_accuracy=float("nan")
    )
    request = _request(tmp_path, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.accuracy_report is None
    assert result.stage_statuses["accuracy"].error.code == "ACCURACY_FAILED"
    assert result.stage_statuses["hardware"].status == "succeeded"
    assert len(executor_calls) == 1
    accuracy_artifact = _artifact(request, "accuracy_report.json")
    assert accuracy_artifact["schema_version"] == 1
    assert accuracy_artifact["status"] == "failed"
    assert accuracy_artifact["error"]["code"] == "ACCURACY_FAILED"
    assert _artifact(request, "run_manifest.json")["status"] == "failed"
    summary = _artifact(request, "summary.json")
    assert summary["accuracy"] is None
    assert summary["hardware"]["macs_total"] == 120


def test_hardware_failure_does_not_block_successful_accuracy(tmp_path, dependency_factory):
    dependencies, calls, executor_calls, _, plan, _ = dependency_factory(
        phase2_error=Phase2Invalid("STATIC_EVAL_FAILED", "fake phase2 failure")
    )
    request = _request(tmp_path, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.stage_statuses["accuracy"].status == "succeeded"
    assert result.stage_statuses["hardware"].error.code == "PHASE2_FAILED"
    assert len(executor_calls) == 1
    assert "origin:FIX_TRAIN:SCALE" in calls
    assert _artifact(request, "phase2_report.json")["error"]["code"] == "PHASE2_FAILED"
    assert _artifact(request, "round_plan.json") == {
        "schema_version": 1,
        "status": "succeeded",
        "round_plan": plan.to_dict(),
    }


def test_phase2_artifact_serialization_failure_records_stage_failure_after_plan(
    tmp_path, dependency_factory
):
    report = _phase2_report(
        inference_metrics={
            "latency_s": float("nan"),
            "energy_j": 4.0,
            "power_avg_w": 2.0,
            "area_um2": 11.0,
        }
    )
    dependencies, _, _, _, plan, _ = dependency_factory(report=report)
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.phase2_report is None
    assert result.hardware_summary is None
    assert result.stage_statuses["hardware"].error.code == "PHASE2_FAILED"
    phase2_artifact = _artifact(request, "phase2_report.json")
    assert phase2_artifact["schema_version"] == 1
    assert phase2_artifact["status"] == "failed"
    assert phase2_artifact["error"]["code"] == "PHASE2_FAILED"
    assert _artifact(request, "round_plan.json") == {
        "schema_version": 1,
        "status": "succeeded",
        "round_plan": plan.to_dict(),
    }
    assert _artifact(request, "run_manifest.json")["status"] == "failed"
    assert _artifact(request, "summary.json")["hardware"] is None


def test_manifest_starts_running_and_ends_terminal_after_stage_transitions(
    tmp_path, monkeypatch, dependency_factory
):
    dependencies, _, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, run_hardware=True)
    seen_manifests = []
    original_write_json = ArtifactStore.write_json

    def record_write_json(self, artifact_name, payload):
        if artifact_name == "run_manifest.json":
            seen_manifests.append(copy.deepcopy(payload))
        return original_write_json(self, artifact_name, payload)

    monkeypatch.setattr(runner_module.ArtifactStore, "write_json", record_write_json)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 0
    assert seen_manifests[0]["status"] == "running"
    assert seen_manifests[-1]["status"] == "succeeded"
    assert seen_manifests[-1]["stage_statuses"]["accuracy"]["status"] == "succeeded"
    assert seen_manifests[-1]["stage_statuses"]["hardware"]["status"] == "succeeded"


def test_no_stage_request_writes_one_initial_running_manifest_before_terminal_update(
    tmp_path, monkeypatch, dependency_factory
):
    dependencies, _, _, _, _, _ = dependency_factory()
    request = _request(tmp_path, skip_accuracy=True, run_hardware=False)
    seen_manifests = []
    original_write_json = ArtifactStore.write_json

    def record_write_json(self, artifact_name, payload):
        if artifact_name == "run_manifest.json":
            seen_manifests.append(copy.deepcopy(payload))
        return original_write_json(self, artifact_name, payload)

    monkeypatch.setattr(runner_module.ArtifactStore, "write_json", record_write_json)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 2
    assert [manifest["status"] for manifest in seen_manifests] == ["running", "failed"]
    assert seen_manifests[0]["artifacts"] == {}
    assert "run_manifest" in seen_manifests[-1]["artifacts"]


def test_infeasible_plan_is_persisted_before_phase2_and_marks_phase1_failure(
    tmp_path, dependency_factory
):
    infeasible_plan = _plan(feasible=False, invalid_reason="fake capacity failure")
    dependencies, _, executor_calls, _, _, _ = dependency_factory(plan=infeasible_plan)
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 1
    assert result.round_plan is infeasible_plan
    assert executor_calls == []
    assert result.stage_statuses["hardware"].error.code == "PHASE1_INVALID"
    plan_artifact = _artifact(request, "round_plan.json")
    assert plan_artifact["status"] == "failed"
    assert plan_artifact["error"]["code"] == "PHASE1_INVALID"
    assert plan_artifact["round_plan"] == infeasible_plan.to_dict()


def test_round_plan_snapshot_is_written_before_feasibility_is_checked(
    tmp_path, monkeypatch, dependency_factory
):
    events = []
    observed_plan = ObservedPlan(
        events,
        feasible=False,
        invalid_reason="observed capacity failure",
    )
    dependencies, _, executor_calls, _, _, _ = dependency_factory(plan=observed_plan)
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)
    plan_statuses = []
    manifest_payloads = []
    original_write_json = ArtifactStore.write_json

    def record_write_json(self, artifact_name, payload):
        if artifact_name == "round_plan.json":
            plan_statuses.append(payload["status"])
            events.append(f"write:round_plan:{payload['status']}")
        if artifact_name == "run_manifest.json":
            manifest_payloads.append(copy.deepcopy(payload))
            events.append("write:manifest")
        return original_write_json(self, artifact_name, payload)

    monkeypatch.setattr(runner_module.ArtifactStore, "write_json", record_write_json)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.exit_code == 1
    assert executor_calls == []
    snapshot_index = events.index("snapshot")
    snapshot_write_index = events.index("write:round_plan:running")
    manifest_index = events.index("write:manifest", snapshot_write_index + 1)
    feasibility_index = events.index("feasibility")
    assert snapshot_index < snapshot_write_index < manifest_index < feasibility_index
    assert plan_statuses == ["running", "failed"]
    snapshot_manifest = next(
        payload
        for payload in manifest_payloads
        if payload["status"] == "running" and "round_plan" in payload["artifacts"]
    )
    assert snapshot_manifest["artifacts"]["round_plan"] == str(
        request.output_dir / "round_plan.json"
    )
    assert _artifact(request, "round_plan.json")["status"] == "failed"


def test_invalid_mac_estimate_is_preserved_without_flattening(tmp_path, dependency_factory):
    invalid_model = _model_ir(shape={"Inputchannel": 2})
    dependencies, _, _, _, _, _ = dependency_factory(model=invalid_model)
    request = _request(tmp_path, skip_accuracy=True, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 2
    assert result.stage_statuses["hardware"].error.code == "INVALID_MAC_ESTIMATE"
    assert _error_codes(result) == ["INVALID_MAC_ESTIMATE"]
    assert not (request.output_dir / "round_plan.json").exists()


def test_accuracy_still_runs_before_fatal_hardware_preparation_failure(
    tmp_path, dependency_factory
):
    dependencies, calls, _, _, _, _ = dependency_factory(
        model=_model_ir(backend_kind="ann", layer_type="lif")
    )
    request = _request(tmp_path, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    assert result.status == "failed"
    assert result.exit_code == 2
    assert result.stage_statuses["accuracy"].status == "succeeded"
    assert result.stage_statuses["hardware"].error.code == "UNSUPPORTED_LAYER_TYPE"
    assert "origin:FIX_TRAIN:SCALE" in calls


def test_backend_output_is_redirected_and_not_exposed_to_caller(
    tmp_path, capsys, dependency_factory
):
    dependencies, _, _, _, _, _ = dependency_factory(print_output=True)
    request = _request(tmp_path, run_hardware=True)

    result = run_round_orchestrated_simulation(request, dependencies=dependencies)

    captured = capsys.readouterr()
    assert result.exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
