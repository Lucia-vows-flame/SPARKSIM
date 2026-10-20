import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import MNSIM.Interface.round_orchestrated_runner as runner_module
import MNSIM.Accuracy_Model.prequantized_weights as deployment_module
from MNSIM.Accuracy_Model.prequantized_weights import (
    PrequantizedWeightError,
    prepare_prequantized_deployment,
    validate_prequantized_deployment,
)
from MNSIM.Interface.quantize import QuantizeLayer, quantize_weight_ste
from MNSIM.Interface.round_orchestrated_runner import (
    RunnerDependencies,
    run_round_orchestrated_simulation,
)
from MNSIM.Interface.round_orchestrated_types import RoundOrchestratedRunRequest


def _source_state():
    return OrderedDict(
        {
            "layer_list.0.last_value": torch.tensor([3.25]),
            "layer_list.0.sublayer_list.0.weight": torch.tensor(
                [[-0.9, -0.2, 0.0]], dtype=torch.float32
            ),
            "layer_list.0.sublayer_list.1.weight": torch.tensor(
                [[0.1, 0.5]], dtype=torch.float32
            ),
            "layer_list.0.bit_scale_list": torch.tensor(
                [[9.0, 1.0], [9.0, 0.123], [9.0, 1.0]]
            ),
        }
    )


def test_prepare_deployment_matches_one_shared_mnsim_quantization(tmp_path):
    source = tmp_path / "source.pth"
    deployment = tmp_path / "deployment_weights_9bit.pth"
    manifest_path = tmp_path / "deployment_weights_9bit_manifest.json"
    state = _source_state()
    torch.save(state, source)

    manifest = prepare_prequantized_deployment(
        source, deployment, manifest_path, qbit=9
    )
    deployed = torch.load(deployment, map_location="cpu", weights_only=True)
    joined = torch.cat(
        [
            state["layer_list.0.sublayer_list.0.weight"].reshape(-1),
            state["layer_list.0.sublayer_list.1.weight"].reshape(-1),
        ]
    )
    expected, expected_digits, expected_lsb = quantize_weight_ste(joined, 9)
    actual = torch.cat(
        [
            deployed["layer_list.0.sublayer_list.0.weight"].reshape(-1),
            deployed["layer_list.0.sublayer_list.1.weight"].reshape(-1),
        ]
    )

    assert torch.equal(actual, expected)
    assert torch.equal(
        torch.round(actual / expected_lsb).to(torch.int64), expected_digits
    )
    assert torch.equal(
        deployed["layer_list.0.bit_scale_list"],
        state["layer_list.0.bit_scale_list"],
    )
    assert manifest["quantization_count"] == 1
    assert manifest["layers"][0]["split_count"] == 2
    assert manifest["all_non_weight_values_unchanged"] is True


def test_prepare_deployment_accepts_configured_non_legacy_width(tmp_path):
    source = tmp_path / "source.pth"
    deployment = tmp_path / "deployment_weights_7bit.pth"
    manifest_path = tmp_path / "deployment_weights_7bit_manifest.json"
    torch.save(_source_state(), source)

    manifest = prepare_prequantized_deployment(
        source, deployment, manifest_path, qbit=7
    )

    assert manifest["weight_bit_width"] == 7
    assert manifest["signed_code_range"] == [-63, 63]
    assert validate_prequantized_deployment(
        deployment, manifest_path, qbit=7
    )["weight_bit_width"] == 7


def test_deployment_validation_rejects_checksum_tampering(tmp_path):
    source = tmp_path / "source.pth"
    deployment = tmp_path / "deployment_weights_9bit.pth"
    manifest_path = tmp_path / "deployment_weights_9bit_manifest.json"
    torch.save(_source_state(), source)
    prepare_prequantized_deployment(source, deployment, manifest_path)
    deployment.write_bytes(deployment.read_bytes() + b"tampered")

    with pytest.raises(PrequantizedWeightError, match="checksum mismatch"):
        validate_prequantized_deployment(deployment, manifest_path)


def test_failed_manifest_write_removes_incomplete_deployment_bundle(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pth"
    deployment = tmp_path / "deployment_weights_9bit.pth"
    manifest_path = tmp_path / "deployment_weights_9bit_manifest.json"
    torch.save(_source_state(), source)

    def fail_manifest(*_args, **_kwargs):
        raise OSError("manifest write failed")

    monkeypatch.setattr(deployment_module, "_atomic_json_save", fail_manifest)
    with pytest.raises(OSError, match="manifest write failed"):
        prepare_prequantized_deployment(source, deployment, manifest_path)

    assert not deployment.exists()
    assert not manifest_path.exists()


def _toy_quantize_layer():
    layer = QuantizeLayer(
        hardware_config={
            "xbar_size": 2,
            "xbar_polarity": 2,
            "DAC_num": 2,
            "weight_bit": 2,
            "input_bit": 1,
            "ADC_quantize_bit": 8,
        },
        layer_config={"type": "fc", "in_features": 5, "out_features": 3},
        quantize_config={"activation_bit": 1, "weight_bit": 9},
    )
    with torch.no_grad():
        joined = torch.linspace(-0.8, 0.7, steps=15).reshape(3, 5)
        quantized, _, _ = quantize_weight_ste(joined, 9)
        widths = [sub.weight.shape[1] for sub in layer.sublayer_list]
        for sub, chunk in zip(layer.sublayer_list, torch.split(quantized, widths, dim=1)):
            sub.weight.copy_(chunk)
        layer.bit_scale_list[1, 1] = 0.0
    return layer


def test_direct_bit_slices_do_not_use_dynamic_weight_scale(monkeypatch):
    layer = _toy_quantize_layer()
    original_round = torch.round

    def reject_dynamic_scale(value, *args, **kwargs):
        if torch.is_tensor(value) and float(value.detach().abs().max().item()) < 0.01:
            raise AssertionError("dynamic weight rounding must not run")
        return original_round(value, *args, **kwargs)

    monkeypatch.setattr(torch, "round", reject_dynamic_scale)
    bit_weights = layer.get_bit_weights(use_prequantized_weights=True)

    assert bit_weights
    assert all(isinstance(value, np.ndarray) for value in bit_weights.values())


def test_runner_reloads_generated_deployment_checkpoint_for_pim(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.pth"
    source.write_bytes(b"source-checkpoint")
    sim_config = tmp_path / "SimConfig.ini"
    sim_config.write_text(
        "[fake]\nvalue=1\n\n"
        "[Algorithm Configuration]\n"
        "Weight_Quantization_Bit=7\n",
        encoding="utf-8",
    )
    calls = []

    def prepare(source_path, deployment_path, manifest_path, *, qbit):
        Path(deployment_path).write_bytes(b"deployment-checkpoint")
        manifest = {
            "deployment_checkpoint_sha256": "d" * 64,
            "source_checkpoint_sha256": "s" * 64,
            "quantization_count": 1,
        }
        Path(manifest_path).write_text(json.dumps(manifest), encoding="utf-8")
        calls.append(("prepare", Path(source_path), qbit))
        return manifest

    class Context:
        snn_accuracy_config = None
        hardware_config = {"ADC_quantization_mode": "UNIFORM"}
        net = SimpleNamespace(layer_list=[])

        def get_net_bits(self):
            return "direct-deployment-bits"

        def set_net_bits_evaluate(self, value, **_kwargs):
            calls.append(("pim", value))
            return 0.5

    def interface_factory(**kwargs):
        calls.append(("context", kwargs))
        return Context()

    monkeypatch.setattr(
        runner_module,
        "_load_accuracy_device_contract",
        lambda _path: SimpleNamespace(
            model_id="rram_stanford_asu_256",
            variation_percent=6.69903802061613,
            device_seed=112,
        ),
    )
    request = RoundOrchestratedRunRequest(
        network="toy_snn",
        weights=source,
        num_classes=2,
        sim_config=sim_config,
        output_dir=tmp_path / "output",
        dataset_module="fake_dataset",
        run_original=False,
        run_pim=True,
    )
    deps = RunnerDependencies(
        interface_factory=interface_factory,
        deployment_preparer=prepare,
        enable_prequantized_weight_flow=True,
        final_rram_validator=lambda _path: None,
        weight_update=lambda _config, bits, **_kwargs: f"updated:{bits}",
    )

    result = run_round_orchestrated_simulation(request, dependencies=deps)

    assert result.exit_code == 0
    assert ("prepare", source, 7) in calls
    context_calls = [call for call in calls if call[0] == "context"]
    assert len(context_calls) == 1
    kwargs = context_calls[0][1]
    assert kwargs["weights_file"].endswith("deployment_weights_9bit.pth")
    assert kwargs["use_prequantized_weights"] is True
    assert kwargs["prequantized_manifest_file"].endswith(
        "deployment_weights_9bit_manifest.json"
    )
    assert ("pim", "updated:direct-deployment-bits") in calls
    assert result.accuracy_report["direct_prequantized_weight_mode"] is True
    assert result.accuracy_report["weight_quantization_count"] == 1
