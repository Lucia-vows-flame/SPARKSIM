import json

import numpy as np
import pytest
import torch
from spikingjelly.activation_based import neuron

from MNSIM.Accuracy_Model.snn_accuracy_contract import canonical_json_sha256
from MNSIM.Accuracy_Model.snn_lif_perturbation import (
    LIFPerturbationArtifactError,
    LIFPerturbationRuntime,
    load_lif_perturbation_artifact,
)
from MNSIM.Scheduling.non_weighted_layers import LIFNode


PARAMETERS = (
    "frontend_gain",
    "frontend_offset",
    "delta_tau",
    "delta_v_th",
    "delta_v_reset",
)


def _write_artifact(
    path,
    *,
    dynamic_noise=None,
    covariance=None,
    mean=None,
    input_min=-1.0,
    input_max=1.0,
):
    if covariance is None:
        covariance = [
            [1.0e-4, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0e-8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0e-4, 2.0e-7, 0.0],
            [0.0, 0.0, 2.0e-7, 1.0e-8, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0e-8],
        ]
    if mean is None:
        mean = [1.0, 0.0, 0.0, 0.0, 0.0]
    body = {
        "schema_version": 1,
        "profile_id": "frontend_lut_lif_lut",
        "qualification": {
            "pdk": "test-pdk",
            "model_files": [{"path": "models.scs", "sha256": "a" * 64}],
            "simulator_version": "test-simulator-1.0",
            "process_corner": "tt",
            "supply_voltage_v": 1.0,
            "temperature_c": 25.0,
            "monte_carlo_sample_count": 1000,
            "seed_policy": "characterization_preserved_joint_distribution",
            "circuit_source_sha256": "b" * 64,
        },
        "distribution": {
            "parameter_order": list(PARAMETERS),
            "mean": mean,
            "covariance": covariance,
        },
        "supported_operating_domain": {
            "timestep_s": 1.0e-6,
            "input_min": input_min,
            "input_max": input_max,
        },
        "dynamic_noise": dynamic_noise,
    }
    payload = dict(body)
    payload["artifact_checksum"] = canonical_json_sha256(body)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def qualified_artifact_path(tmp_path):
    return _write_artifact(tmp_path / "qualified_lif_perturbation.json")


def test_qualified_artifact_preserves_joint_distribution_and_checksum(
    qualified_artifact_path,
):
    artifact = load_lif_perturbation_artifact(qualified_artifact_path)
    assert artifact.profile_id == "frontend_lut_lif_lut"
    assert artifact.parameter_order == PARAMETERS
    assert artifact.monte_carlo_sample_count > 0
    assert artifact.artifact_checksum


def test_missing_provenance_or_checksum_fails_closed(qualified_artifact_path):
    payload = json.loads(qualified_artifact_path.read_text(encoding="utf-8"))
    del payload["qualification"]["simulator_version"]
    qualified_artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LIFPerturbationArtifactError, match="simulator_version"):
        load_lif_perturbation_artifact(qualified_artifact_path)


def test_tampered_checksum_and_non_psd_covariance_fail_closed(
    qualified_artifact_path,
):
    payload = json.loads(qualified_artifact_path.read_text(encoding="utf-8"))
    payload["qualification"]["temperature_c"] = 125.0
    qualified_artifact_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LIFPerturbationArtifactError, match="checksum"):
        load_lif_perturbation_artifact(qualified_artifact_path)

    covariance = np.eye(5)
    covariance[0, 0] = -1.0
    _write_artifact(qualified_artifact_path, covariance=covariance.tolist())
    with pytest.raises(LIFPerturbationArtifactError, match="positive-semidefinite"):
        load_lif_perturbation_artifact(qualified_artifact_path)


def test_same_artifact_seed_layer_and_shape_reproduce_static_mismatch(
    qualified_artifact_path,
):
    artifact = load_lif_perturbation_artifact(qualified_artifact_path)
    first = LIFPerturbationRuntime(artifact, seed=112)
    second = LIFPerturbationRuntime(artifact, seed=112)
    a = first.static_parameters("L001", (1, 4, 3, 3))
    b = second.static_parameters("L001", (1, 4, 3, 3))
    assert all(np.array_equal(a[key], b[key]) for key in PARAMETERS)
    assert not np.array_equal(
        a["delta_tau"], first.static_parameters("L002", (1, 4, 3, 3))["delta_tau"]
    )


def test_static_mismatch_is_cached_for_the_whole_run(qualified_artifact_path):
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(qualified_artifact_path), seed=112
    )
    first = runtime.static_parameters("L001", (1, 8))
    second = runtime.static_parameters("L001", (1, 8))
    assert all(first[key] is second[key] for key in PARAMETERS)


def test_formal_runtime_rejects_any_seed_other_than_112(qualified_artifact_path):
    artifact = load_lif_perturbation_artifact(qualified_artifact_path)
    with pytest.raises(LIFPerturbationArtifactError, match="seed must be 112"):
        LIFPerturbationRuntime(artifact, seed=113)


def test_dynamic_noise_requires_transient_characterization(
    qualified_artifact_path,
):
    _write_artifact(qualified_artifact_path, dynamic_noise={"std": 0.001})
    with pytest.raises(LIFPerturbationArtifactError, match="transient"):
        load_lif_perturbation_artifact(qualified_artifact_path)


def test_dynamic_noise_is_reproducible_and_advances_per_call(tmp_path):
    path = _write_artifact(
        tmp_path / "dynamic.json",
        dynamic_noise={"std": 0.001, "transient_source_sha256": "c" * 64},
    )
    artifact = load_lif_perturbation_artifact(path)
    first = LIFPerturbationRuntime(artifact, seed=112)
    second = LIFPerturbationRuntime(artifact, seed=112)

    first_step = first.dynamic_noise("L001", (2, 3))
    second_step = first.dynamic_noise("L001", (2, 3))

    assert np.array_equal(
        first_step, second.dynamic_noise("L001", (2, 3))
    )
    assert not np.array_equal(first_step, second_step)


def test_absent_dynamic_noise_returns_zeros(qualified_artifact_path):
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(qualified_artifact_path), seed=112
    )
    assert np.array_equal(runtime.dynamic_noise("L001", (2, 3)), np.zeros((2, 3)))
    assert runtime._dynamic_generators == {}


def test_nominal_accuracy_wrapper_is_bit_exact_with_spikingjelly():
    sequence = torch.tensor(
        [
            [[[0.2, 0.7]]],
            [[[0.5, 0.1]]],
            [[[0.9, 0.0]]],
        ]
    )
    shared = LIFNode(tau=2.5, v_th=0.4, v_reset=0.0)
    reference = neuron.LIFNode(
        tau=2.5,
        decay_input=True,
        v_threshold=0.4,
        v_reset=0.0,
        step_mode="s",
        backend="torch",
    )

    shared.set_accuracy_perturbation(None, layer_id="L001")
    actual = shared.forward_sequence(sequence)
    reference.step_mode = "m"
    expected = reference(sequence)

    assert torch.equal(actual, expected)
    assert torch.equal(shared.node.v, reference.v)


def test_static_accuracy_parameters_survive_reset_and_replay_exactly(tmp_path):
    path = _write_artifact(
        tmp_path / "static.json",
        covariance=np.zeros((5, 5)).tolist(),
        mean=[0.9, 0.01, 0.1, 0.02, 0.03],
    )
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(path), seed=112
    )
    lif = LIFNode(tau=2.0, v_th=0.5, v_reset=0.0)
    lif.set_accuracy_perturbation(runtime, layer_id="L007")
    sequence = torch.tensor(
        [
            [[[0.1, 0.4]]],
            [[[0.8, 0.2]]],
            [[[0.3, 0.7]]],
        ]
    )

    first = lif.forward_sequence(sequence)
    sampled_before = lif.sampled_accuracy_parameters()
    lif.reset_state()
    sampled_after_reset = lif.sampled_accuracy_parameters()
    second = lif.forward_sequence(sequence)

    assert sampled_before is sampled_after_reset
    assert all(
        sampled_before[name] is sampled_after_reset[name]
        for name in PARAMETERS
    )
    assert torch.equal(first, second)


def test_dynamic_noise_is_sampled_once_per_timestep_for_the_full_batch(tmp_path):
    path = _write_artifact(
        tmp_path / "dynamic-shape.json",
        covariance=np.zeros((5, 5)).tolist(),
        dynamic_noise={"std": 0.001, "transient_source_sha256": "c" * 64},
    )
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(path), seed=112
    )
    calls = []
    original_dynamic_noise = runtime.dynamic_noise

    def capture(layer_id, shape):
        calls.append((layer_id, shape))
        return original_dynamic_noise(layer_id, shape)

    runtime.dynamic_noise = capture
    lif = LIFNode(tau=2.0, v_th=0.5, v_reset=0.0)
    lif.set_accuracy_perturbation(runtime, layer_id="L005")
    lif.forward_sequence(torch.zeros(3, 2, 4))

    assert calls == [("L005", (2, 4))] * 3


def test_frontend_operating_range_violation_fails_instead_of_clamping(tmp_path):
    path = _write_artifact(
        tmp_path / "narrow.json",
        covariance=np.zeros((5, 5)).tolist(),
        mean=[2.0, 0.0, 0.0, 0.0, 0.0],
        input_min=-0.5,
        input_max=0.5,
    )
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(path), seed=112
    )
    lif = LIFNode(tau=2.0, v_th=0.5, v_reset=0.0)
    lif.set_accuracy_perturbation(runtime, layer_id="L003")

    with pytest.raises(LIFPerturbationArtifactError, match="operating range"):
        lif.forward_sequence(torch.tensor([[[[0.4]]]]))


def test_nonpositive_perturbed_tau_fails_closed(tmp_path):
    path = _write_artifact(
        tmp_path / "invalid-tau.json",
        covariance=np.zeros((5, 5)).tolist(),
        mean=[1.0, 0.0, -1.0, 0.0, 0.0],
    )
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(path), seed=112
    )
    lif = LIFNode(tau=2.0, v_th=0.5, v_reset=0.0)
    lif.set_accuracy_perturbation(runtime, layer_id="L004")

    with pytest.raises(LIFPerturbationArtifactError, match="tau"):
        lif.forward_sequence(torch.tensor([[[[0.1]]]]))


def test_operating_range_rejects_nonfinite_values(qualified_artifact_path):
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(qualified_artifact_path), seed=112
    )

    with pytest.raises(LIFPerturbationArtifactError, match="operating range"):
        runtime.validate_operating_range(torch.tensor([float("nan")]))
