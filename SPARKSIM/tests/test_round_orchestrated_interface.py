import configparser
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from MNSIM.Interface import interface


def _write_sim_config(path):
    config = configparser.ConfigParser()
    config["Crossbar level"] = {"Xbar_Size": "128,128"}
    config["Process element level"] = {
        "PIM_Type": "0",
        "Xbar_Polarity": "1",
        "DAC_Num": "1",
        "Group_Num": "1",
    }
    config["Device level"] = {"Device_Level": "16"}
    config["Interface level"] = {
        "ADC_Choice": "-1",
        "DAC_Choice": "-1",
        "ADC_Precision": "8",
        "DAC_Precision": "8",
    }
    config["Tile level"] = {"PE_Num": "128,128"}
    with path.open("w", encoding="utf-8") as stream:
        config.write(stream)


def _set_weight_quantization_bit(path, value):
    config = configparser.ConfigParser()
    config.read(path, encoding="utf-8")
    if not config.has_section("Algorithm Configuration"):
        config.add_section("Algorithm Configuration")
    config.set("Algorithm Configuration", "Weight_Quantization_Bit", str(value))
    with path.open("w", encoding="utf-8") as stream:
        config.write(stream)


def _write_snn_sim_config(path, *, profile, artifact=""):
    _write_sim_config(path)
    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")
    parser.set("Device level", "Device_Level", "256")
    parser["Interface level"].update(
        {
            "ADC_Precision": "5",
            "ADC_Profile": "HONG_2025_READC_5BIT",
            "ADC_Quantization_Mode": "READC_LLOYD_MAX",
            "ADC_ReADC_Percentile_Low": "0.5",
            "ADC_ReADC_Percentile_High": "99.5",
            "ADC_ReADC_Max_Samples_Per_Layer": "8000",
        }
    )
    parser["SNN Accuracy"] = {
        "SNN_Hardware_Profile": profile,
        "Hardware_Current_Model": "physical_conductance",
        "LIF_Perturbation_Artifact": str(artifact),
        "LIF_Perturbation_Seed": "112",
    }
    with path.open("w", encoding="utf-8") as stream:
        parser.write(stream)


class _FakeNetwork(torch.nn.Module):
    def load_change_weights(self, weights):
        self.loaded_weights = weights


_OMITTED = object()


def _construct(
    monkeypatch,
    sim_config,
    *,
    dataset_module,
    num_classes=_OMITTED,
    device="cpu",
    weights_file=None,
    positional_compat=False,
    network_module="fake_network",
):
    calls = []
    fake_net = _FakeNetwork()

    def fake_import(name):
        calls.append(name)
        if name == "MNSIM.Interface.network":
            return SimpleNamespace(
                get_net=lambda hardware_config, cate, num_classes: (
                    calls.append((hardware_config, cate, num_classes)) or fake_net
                )
            )
        raise AssertionError(f"unexpected dataset import: {name}")

    monkeypatch.setattr(interface, "import_module", fake_import)
    if positional_compat:
        instance = interface.TrainTestInterface(
            network_module, dataset_module, str(sim_config), weights_file, device
        )
    else:
        kwargs = {"device": device, "weights_file": weights_file}
        if num_classes is not _OMITTED:
            kwargs["num_classes"] = num_classes
        instance = interface.TrainTestInterface(
            network_module, dataset_module, str(sim_config), **kwargs
        )
    return instance, calls


def test_hardware_only_context_uses_explicit_classes_without_dataset_import(tmp_path, monkeypatch):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)

    instance, calls = _construct(
        monkeypatch, sim_config, dataset_module=None, num_classes="17"
    )

    assert calls[0] == "MNSIM.Interface.network"
    assert calls[1][1:] == ("fake_network", 17)
    assert instance.device == torch.device("cpu")


def test_interface_reads_shared_weight_quantization_bit_from_simconfig(tmp_path, monkeypatch):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)
    _set_weight_quantization_bit(sim_config, 7)

    instance, calls = _construct(
        monkeypatch, sim_config, dataset_module=None, num_classes="17"
    )

    assert instance.hardware_config["weight_quantization_bit"] == 7
    assert calls[1][0]["weight_quantization_bit"] == 7


def test_snn_interface_parses_the_formal_adc_profile(tmp_path, monkeypatch):
    config_path = tmp_path / "SimConfig.ini"
    _write_snn_sim_config(config_path, profile="adc_digital_lif")
    instance, _calls = _construct(
        monkeypatch,
        config_path,
        dataset_module=None,
        num_classes=8,
        network_module="standard_vgg8_snn_v7_phase6",
    )
    assert instance.snn_accuracy_config.hardware_profile == "adc_digital_lif"
    assert instance.snn_accuracy_config.adc_model == "READC_LLOYD_MAX"


def test_snn_interface_lut_profile_fails_when_artifact_is_missing(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "SimConfig.ini"
    _write_snn_sim_config(
        config_path,
        profile="frontend_lut_lif_lut",
        artifact=tmp_path / "missing.json",
    )
    with pytest.raises(FileNotFoundError, match="missing.json"):
        _construct(
            monkeypatch,
            config_path,
            dataset_module=None,
            num_classes=8,
            network_module="standard_vgg8_snn_v7_phase6",
        )


def test_explicit_num_classes_overrides_dataset_suffix(tmp_path, monkeypatch):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)

    instance, calls = _construct(
        monkeypatch,
        sim_config,
        dataset_module="MNSIM.Interface.cifar100",
        num_classes=17,
    )

    assert calls[1][2] == 17
    assert instance.device == torch.device("cpu")


@pytest.mark.parametrize(
    ("dataset_module", "expected_classes"),
    [
        ("MNSIM.Interface.cifar10", 10),
        ("MNSIM.Interface.cifar100", 100),
        ("MNSIM.Interface.Imagenet", 1000),
    ],
)
def test_dataset_suffix_infers_all_legacy_class_counts(
    tmp_path, monkeypatch, dataset_module, expected_classes
):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)
    instance, calls = _construct(
        monkeypatch,
        sim_config,
        dataset_module=dataset_module,
        positional_compat=True,
    )

    assert calls[0] == "MNSIM.Interface.network"
    assert calls[1][2] == expected_classes
    assert instance.device == torch.device("cpu")


def test_gpu_id_falls_back_to_cpu_when_cuda_unavailable(tmp_path, monkeypatch):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)
    monkeypatch.setattr(interface.torch.cuda, "is_available", lambda: False)

    instance, calls = _construct(
        monkeypatch, sim_config, dataset_module=None, num_classes=17, device=3
    )

    assert instance.device == torch.device("cpu")
    assert calls[1][2] == 17


def test_literal_cpu_stays_cpu_when_cuda_is_available(tmp_path, monkeypatch):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)
    monkeypatch.setattr(interface.torch.cuda, "is_available", lambda: True)

    instance, _ = _construct(
        monkeypatch, sim_config, dataset_module=None, num_classes=17, device="cpu"
    )

    assert instance.device == torch.device("cpu")


def test_checkpoint_uses_model_specific_loader_and_device_map_location(tmp_path, monkeypatch):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)
    loaded_weights = {"weights": "fake"}
    load = mock.Mock(return_value=loaded_weights)
    monkeypatch.setattr(interface.torch, "load", load)

    instance, _ = _construct(
        monkeypatch,
        sim_config,
        dataset_module=None,
        num_classes=17,
        weights_file="weights.pth",
    )

    load.assert_called_once_with("weights.pth", map_location=instance.device)
    assert instance.net.loaded_weights is loaded_weights


def test_missing_dataset_and_classes_fails_before_network_build(tmp_path, monkeypatch):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)
    network_import = mock.Mock(side_effect=AssertionError("network should not be built"))
    monkeypatch.setattr(interface, "import_module", network_import)

    with pytest.raises(ValueError, match="num_classes"):
        interface.TrainTestInterface(
            "fake_network", None, str(sim_config), device="cpu"
        )

    network_import.assert_not_called()


@pytest.mark.parametrize("method_name", ["origin_evaluate", "set_net_bits_evaluate"])
def test_dataset_free_accuracy_methods_raise_value_error(tmp_path, monkeypatch, method_name):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)
    instance, calls = _construct(
        monkeypatch, sim_config, dataset_module=None, num_classes=17
    )

    with pytest.raises(ValueError, match="dataset_module"):
        if method_name == "origin_evaluate":
            instance.origin_evaluate()
        else:
            instance.set_net_bits_evaluate({})

    assert calls.count("MNSIM.Interface.network") == 1


@pytest.mark.parametrize("invalid", [0, -1, 1.9, True, "not-an-int"])
def test_explicit_num_classes_must_be_positive_integer(tmp_path, monkeypatch, invalid):
    sim_config = tmp_path / "SimConfig.ini"
    _write_sim_config(sim_config)

    with pytest.raises(ValueError, match="num_classes"):
        _construct(monkeypatch, sim_config, dataset_module=None, num_classes=invalid)
