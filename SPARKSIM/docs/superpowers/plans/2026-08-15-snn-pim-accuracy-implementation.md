# SNN Processing-in-Memory Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the canonical `SNN Processing-in-Memory Accuracy` subsystem for both `adc_digital_lif` and `frontend_lut_lif_lut`, while keeping physical current, LUT lookup state, and hardware-performance metrics outside the accuracy forward path.

**Architecture:** Add a small accuracy contract layer that owns the hardware-profile selection, binary-spike numerical domain, fixed final-RRAM accuracy controls, and reproducible artifact identities. Both Original and PIM accuracy continue through one `NetworkGraph.sequence_accuracy_forward()` executor and one SpikingJelly-backed LIF wrapper; only the weighted numerical backend, ReADC action, and qualified LUT-profile perturbation differ. ReADC calibration and LUT perturbation are versioned, checksummed artifacts that fail closed before test inference when their identities do not match the checkpoint, dataset, mapping, device, or configured profile.

**Tech Stack:** Python 3.12, PyTorch, SpikingJelly activation-based neurons, NumPy, `configparser`, JSON/SHA-256 artifacts, pytest.

---

## Scope And Working Rules

Work from:

```text
/home/xzh/workspace/cim_project/MNSIM-2.0/.worktrees/
  task-20260707-round-orchestrated-mnsim/
  round_orchestrated_simulator/MNSIM-2.0
```

Use these launchers:

```bash
GIT=/home/xzh/micromamba/envs/workspace/bin/git
PYTHON=/home/xzh/micromamba/envs/workspace/bin/python
```

Do not run a repository Python file as `python path/to/file.py`. Test with
`$PYTHON -m pytest tests/test_name.py -q`; use the repository shell/runpy
wrappers for end-to-end entry points.

The worktree already contains unrelated user changes. Before every commit, run
`$GIT status --short`, stage only the exact files listed in that task, and verify
the staged diff with `$GIT diff --cached --check` and `$GIT diff --cached`.
The `git add` commands below are the task's file inventory. For every path that
was already modified or untracked before implementation, stage only the new
implementation hunks with `$GIT add -p -- path`; use `$GIT add -N path` first for
a pre-existing untracked file. Never stage the file wholesale merely because it
appears in a task command block.

This plan intentionally excludes:

- physical `G/Ipos/Ineg/Idiff` current computation;
- frontend-LUT and LIF-LUT hardware lookup execution;
- LUT or ADC energy, latency, area, static-power, and input-closure ledgers;
- round scheduling and hardware-performance variation;
- QAT and deployment-artifact generation.

Those are separate independently testable plans. This plan only consumes the
existing MNSIM bit-weight representation and produces the two public accuracy
fields already frozen by the canonical spec.

## File Map

Create:

- `MNSIM/Accuracy_Model/snn_accuracy_contract.py`: profile parsing, explicit
  `BinarySpikeDomain`, formal accuracy-device controls, and stable identity
  hashes shared by ReADC and reports.
- `MNSIM/Accuracy_Model/readc_artifact.py`: ReADC artifact schema, checksums,
  identity matching, serialization, and per-layer table validation.
- `MNSIM/Accuracy_Model/snn_lif_perturbation.py`: qualified frontend/LIF
  characterization artifact loader and deterministic joint sampler.
- `tests/test_snn_accuracy_contract.py`: profile, binary-domain, and fixed-device
  contract tests.
- `tests/test_snn_no_adc_reconstruction.py`: exact LUT-profile weighted numerical
  reconstruction and forbidden ADC/output-quantizer tests.
- `tests/test_readc_artifact.py`: ReADC artifact identity and fail-closed tests.
- `tests/test_snn_lif_perturbation.py`: artifact qualification, deterministic
  sampling, persistence, dynamic-noise, and nominal-equivalence tests.

Modify:

- `MNSIM/Interface/quantize.py`: accept `BinarySpikeDomain` at weighted inputs and
  add an internal exact no-ADC reconstruction branch without changing public
  accuracy names.
- `MNSIM/Interface/network.py`: use one sequence executor and one multi-step LIF
  call for Original and PIM, route profile-specific weighted computation, and
  reset LIF outputs to `BinarySpikeDomain`.
- `MNSIM/Scheduling/non_weighted_layers.py`: keep one SpikingJelly-backed LIF
  wrapper and optionally install a qualified per-run perturbation runtime.
- `MNSIM/Interface/interface.py`: parse the SNN accuracy profile, expose training
  split identity, calibrate/install ReADC artifacts, and pass the profile and
  perturbation runtime through evaluation.
- `MNSIM/Interface/dvs_binned_npz.py`: expose a deterministic train/test split
  identity without reading the opposite split.
- `MNSIM/Accuracy_Model/Weight_update.py`: require an explicit generator for the
  formal SNN path while preserving the legacy caller behavior.
- `MNSIM/Interface/final_rram_contract.py`: expose the locked accuracy contract
  (`R-ratio=on`, variation=on, SAF=off, seed 112) from the two final models.
- `MNSIM/Interface/round_orchestrated_artifacts.py`: own
  `readc_calibration.json` as a durable runner artifact.
- `MNSIM/Interface/round_orchestrated_runner.py`: sample device variation once,
  share it across calibration/test, fail closed by profile, and emit the full
  accuracy metadata contract.
- `MNSIM/Interface/round_orchestrated_cli.py`: render the SimConfig-selected SNN
  profile without introducing another accuracy label or a second profile source.
- `SimConfig.ini`: declare the default formal ADC profile and fixed ReADC values.
- Existing focused tests listed in the tasks below.

### Task 1: Define The SNN Accuracy Contract And Binary Spike Domain

**Files:**

- Create: `MNSIM/Accuracy_Model/snn_accuracy_contract.py`
- Create: `tests/test_snn_accuracy_contract.py`
- Modify: `MNSIM/Interface/final_rram_contract.py`
- Modify: `tests/test_final_rram_contract.py`

- [x] **Step 1: Write failing tests for the binary domain, profile parser, and fixed device controls**

Create `tests/test_snn_accuracy_contract.py` with these tests:

```python
import configparser

import pytest
import torch

from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    ADC_DIGITAL_LIF,
    FRONTEND_LUT_LIF_LUT,
    BINARY_SPIKE_DOMAIN,
    SNNAccuracyContractError,
    parse_snn_accuracy_config,
)


def _parser(profile: str) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser["SNN Accuracy"] = {
        "SNN_Hardware_Profile": profile,
        "Hardware_Current_Model": "physical_conductance",
        "LIF_Perturbation_Artifact": "",
        "LIF_Perturbation_Seed": "112",
    }
    parser["Interface level"] = {
        "ADC_Profile": "HONG_2025_READC_5BIT",
        "ADC_Quantization_Mode": "READC_LLOYD_MAX",
        "ADC_Precision": "5",
        "ADC_ReADC_Percentile_Low": "0.5",
        "ADC_ReADC_Percentile_High": "99.5",
        "ADC_ReADC_Max_Samples_Per_Layer": "8000",
    }
    return parser


def test_binary_spike_domain_is_unsigned_one_cycle_and_rejects_conversion():
    assert BINARY_SPIKE_DOMAIN.values == (0, 1)
    assert BINARY_SPIKE_DOMAIN.scale == 1.0
    assert BINARY_SPIKE_DOMAIN.signed is False
    assert BINARY_SPIKE_DOMAIN.input_digit_max == 1
    assert BINARY_SPIKE_DOMAIN.activation_cycle == 1
    assert torch.equal(
        BINARY_SPIKE_DOMAIN.validate(torch.tensor([0.0, 1.0])),
        torch.tensor([0.0, 1.0]),
    )
    with pytest.raises(ValueError, match="binary spike"):
        BINARY_SPIKE_DOMAIN.validate(torch.tensor([0.0, 0.49, 1.0]))


def test_adc_profile_requires_the_frozen_readc_configuration():
    config = parse_snn_accuracy_config(_parser(ADC_DIGITAL_LIF), source="test.ini")
    assert config.hardware_profile == ADC_DIGITAL_LIF
    assert config.hardware_current_model == "physical_conductance"
    assert config.adc_model == "READC_LLOYD_MAX"
    assert config.adc_precision == 5
    assert config.lif_perturbation_artifact is None

    parser = _parser(ADC_DIGITAL_LIF)
    parser.set("Interface level", "ADC_Precision", "6")
    with pytest.raises(SNNAccuracyContractError, match="ADC_Precision=5"):
        parse_snn_accuracy_config(parser, source="test.ini")


def test_lut_profile_requires_an_artifact_path_and_has_no_adc_model():
    parser = _parser(FRONTEND_LUT_LIF_LUT)
    with pytest.raises(SNNAccuracyContractError, match="LIF_Perturbation_Artifact"):
        parse_snn_accuracy_config(parser, source="test.ini")

    parser.set(
        "SNN Accuracy", "LIF_Perturbation_Artifact", "characterization/lif.json"
    )
    config = parse_snn_accuracy_config(parser, source="test.ini")
    assert config.hardware_profile == FRONTEND_LUT_LIF_LUT
    assert config.adc_model is None


def test_unknown_snn_profile_fails_closed():
    with pytest.raises(SNNAccuracyContractError, match="unsupported"):
        parse_snn_accuracy_config(_parser("mixed_adc_lut"), source="test.ini")


def test_adc_profile_rejects_a_lut_perturbation_artifact():
    parser = _parser(ADC_DIGITAL_LIF)
    parser.set("SNN Accuracy", "LIF_Perturbation_Artifact", "lif.json")
    with pytest.raises(SNNAccuracyContractError, match="must not configure"):
        parse_snn_accuracy_config(parser, source="test.ini")
```

Extend `tests/test_final_rram_contract.py`:

```python
from MNSIM.Interface.final_rram_contract import load_snn_accuracy_device_contract


@pytest.mark.parametrize(
    ("model_id", "variation"),
    [
        ("rram_20ns_linear_fit", 6.69903802061613),
        ("rram_30ns_linear_fit", 7.725782066255718),
    ],
)
def test_final_models_expose_the_fixed_snn_accuracy_device_contract(
    model_id, variation
):
    parser = configparser.ConfigParser()
    apply_final_rram_model(parser, model_id, repo_root=REPO_ROOT)
    contract = load_snn_accuracy_device_contract(parser, source="generated")

    assert contract.model_id == model_id
    assert contract.variation_percent == pytest.approx(variation)
    assert contract.device_seed == 112
    assert contract.enable_r_ratio is True
    assert contract.enable_variation is True
    assert contract.enable_saf is False
    assert contract.saf_percent == (0.0, 0.0)
```

- [x] **Step 2: Run the focused tests and verify they fail because the contract module/API does not exist**

Run:

```bash
$PYTHON -m pytest \
  tests/test_snn_accuracy_contract.py \
  tests/test_final_rram_contract.py -q
```

Expected: collection fails with `ModuleNotFoundError` for
`MNSIM.Accuracy_Model.snn_accuracy_contract` or an import error for
`load_snn_accuracy_device_contract`.

- [x] **Step 3: Implement the immutable profile, binary-domain, and device contracts**

Create `MNSIM/Accuracy_Model/snn_accuracy_contract.py` with these public types
and constants:

```python
from __future__ import annotations

import configparser
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ADC_DIGITAL_LIF = "adc_digital_lif"
FRONTEND_LUT_LIF_LUT = "frontend_lut_lif_lut"
SUPPORTED_SNN_HARDWARE_PROFILES = frozenset(
    {ADC_DIGITAL_LIF, FRONTEND_LUT_LIF_LUT}
)


class SNNAccuracyContractError(ValueError):
    pass


@dataclass(frozen=True)
class BinarySpikeDomain:
    values: tuple[int, int] = (0, 1)
    scale: float = 1.0
    signed: bool = False
    input_digit_max: int = 1
    activation_cycle: int = 1

    def validate(self, value: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError("binary spike value must be a torch.Tensor")
        if not torch.isfinite(value).all():
            raise ValueError("binary spike tensor contains non-finite values")
        if not torch.all((value == 0) | (value == 1)):
            raise ValueError("binary spike tensor must contain only 0 and 1")
        return value


BINARY_SPIKE_DOMAIN = BinarySpikeDomain()


@dataclass(frozen=True)
class SNNAccuracyConfig:
    hardware_profile: str
    hardware_current_model: str
    adc_model: str | None
    adc_precision: int | None
    readc_percentile_low: float | None
    readc_percentile_high: float | None
    readc_max_samples_per_layer: int | None
    lif_perturbation_artifact: Path | None
    lif_perturbation_seed: int


def parse_snn_accuracy_config(
    parser: configparser.ConfigParser, *, source: str | Path
) -> SNNAccuracyConfig:
    section = "SNN Accuracy"
    if not parser.has_section(section):
        raise SNNAccuracyContractError(f"{source} is missing [{section}]")
    profile = parser.get(section, "SNN_Hardware_Profile").strip().lower()
    if profile not in SUPPORTED_SNN_HARDWARE_PROFILES:
        raise SNNAccuracyContractError(
            f"unsupported SNN_Hardware_Profile {profile!r} in {source}"
        )
    current_model = parser.get(section, "Hardware_Current_Model").strip().lower()
    if current_model != "physical_conductance":
        raise SNNAccuracyContractError(
            "formal Hardware_Current_Model must be physical_conductance"
        )
    seed = parser.getint(section, "LIF_Perturbation_Seed", fallback=112)
    if seed != 112:
        raise SNNAccuracyContractError("formal LIF_Perturbation_Seed must be 112")
    artifact_text = parser.get(
        section, "LIF_Perturbation_Artifact", fallback=""
    ).strip()

    if profile == ADC_DIGITAL_LIF:
        if artifact_text:
            raise SNNAccuracyContractError(
                "adc_digital_lif must not configure LIF_Perturbation_Artifact"
            )
        expected = {
            "ADC_Profile": "HONG_2025_READC_5BIT",
            "ADC_Quantization_Mode": "READC_LLOYD_MAX",
            "ADC_Precision": "5",
            "ADC_ReADC_Percentile_Low": "0.5",
            "ADC_ReADC_Percentile_High": "99.5",
            "ADC_ReADC_Max_Samples_Per_Layer": "8000",
        }
        for option, expected_value in expected.items():
            actual = parser.get("Interface level", option).strip()
            if actual.upper() != expected_value.upper():
                raise SNNAccuracyContractError(
                    f"formal ADC profile requires {option}={expected_value}, got {actual}"
                )
        return SNNAccuracyConfig(
            hardware_profile=profile,
            hardware_current_model=current_model,
            adc_model="READC_LLOYD_MAX",
            adc_precision=5,
            readc_percentile_low=0.5,
            readc_percentile_high=99.5,
            readc_max_samples_per_layer=8000,
            lif_perturbation_artifact=None,
            lif_perturbation_seed=seed,
        )

    if not artifact_text:
        raise SNNAccuracyContractError(
            "frontend_lut_lif_lut requires LIF_Perturbation_Artifact"
        )
    artifact_path = Path(artifact_text)
    if not artifact_path.is_absolute():
        artifact_path = Path(source).resolve().parent / artifact_path
    return SNNAccuracyConfig(
        hardware_profile=profile,
        hardware_current_model=current_model,
        adc_model=None,
        adc_precision=None,
        readc_percentile_low=None,
        readc_percentile_high=None,
        readc_max_samples_per_layer=None,
        lif_perturbation_artifact=artifact_path.resolve(),
        lif_perturbation_seed=seed,
    )


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bit_weights_sha256(net_bit_weights: list[Any]) -> str:
    digest = hashlib.sha256()
    for layer_index, layer in enumerate(net_bit_weights):
        digest.update(f"layer:{layer_index}\n".encode("ascii"))
        if layer is None:
            digest.update(b"none\n")
            continue
        for key in sorted(layer):
            array = np.ascontiguousarray(layer[key])
            digest.update(key.encode("utf-8") + b"\0")
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(repr(array.shape).encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
```

In `MNSIM/Interface/final_rram_contract.py`, add:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SNNAccuracyDeviceContract:
    model_id: str
    variation_percent: float
    device_seed: int
    enable_r_ratio: bool
    enable_variation: bool
    enable_saf: bool
    saf_percent: tuple[float, float]


def load_snn_accuracy_device_contract(
    parser: configparser.ConfigParser, *, source: str = "SimConfig.ini"
) -> SNNAccuracyDeviceContract:
    metadata = validate_final_rram_parser(parser, source=source)
    saf = tuple(
        float(value.strip())
        for value in parser.get("Device level", "Device_SAF").split(",")
    )
    contract = SNNAccuracyDeviceContract(
        model_id=metadata["model_id"],
        variation_percent=parser.getfloat("Device level", "Device_Variation"),
        device_seed=parser.getint("SNN RouteA", "Device_Seed"),
        enable_r_ratio=parser.getboolean("SNN RouteA", "Enable_R_Ratio"),
        enable_variation=parser.getboolean("SNN RouteA", "Enable_Variation"),
        enable_saf=parser.getboolean("SNN RouteA", "Enable_SAF"),
        saf_percent=(saf[0], saf[1]),
    )
    if contract.device_seed != FINAL_RRAM_DEVICE_SEED:
        raise FinalRramContractError("formal SNN accuracy Device_Seed must be 112")
    if not contract.enable_r_ratio or not contract.enable_variation:
        raise FinalRramContractError(
            "formal SNN accuracy requires R-ratio and variation"
        )
    if contract.enable_saf or contract.saf_percent != (0.0, 0.0):
        raise FinalRramContractError("formal SNN accuracy requires SAF=0,0 and disabled")
    return contract
```

Export the two new device-contract names in `__all__`.

- [x] **Step 4: Run focused tests and verify the contracts pass**

Run:

```bash
$PYTHON -m pytest \
  tests/test_snn_accuracy_contract.py \
  tests/test_final_rram_contract.py -q
```

Expected: all tests pass.

- [x] **Step 5: Commit the contract layer**

```bash
$GIT add \
  MNSIM/Accuracy_Model/snn_accuracy_contract.py \
  MNSIM/Interface/final_rram_contract.py \
  tests/test_snn_accuracy_contract.py \
  tests/test_final_rram_contract.py
$GIT diff --cached --check
$GIT commit -m "feat: define formal SNN accuracy contracts"
```

### Task 2: Make Binary Spikes A First-Class Weighted-Layer Input

**Files:**

- Modify: `MNSIM/Interface/quantize.py`
- Modify: `MNSIM/Interface/network.py`
- Modify: `tests/test_snn_unified_accuracy.py`
- Modify: `tests/test_snn_original_accuracy.py`

- [x] **Step 1: Write failing tests for direct 0/1 consumption and LIF-domain reset**

Add to `tests/test_snn_unified_accuracy.py`:

```python
from MNSIM.Accuracy_Model.snn_accuracy_contract import BINARY_SPIKE_DOMAIN
from MNSIM.Interface.network import NetworkGraph
from MNSIM.Interface.quantize import QuantizeLayer


def _tiny_two_stage_snn():
    net = NetworkGraph(
        _hardware_config(),
        [
            {"type": "conv", "in_channels": 1, "out_channels": 1, "kernel_size": 1},
            {"type": "lif", "tau": 50.5016666556, "v_th": 0.01, "v_reset": 0.0},
            {"type": "conv", "in_channels": 1, "out_channels": 1, "kernel_size": 1},
            {"type": "lif", "tau": 50.5016666556, "v_th": 0.01, "v_reset": 0.0},
        ],
        [
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
            {"weight_bit": 9, "activation_bit": 9, "point_shift": -2},
        ],
        [[-1], [-1], [-1], [-1]],
        {
            "activation_scale": 1.0,
            "activation_bit": 9,
            "input_shape": (1, 1, 1, 1),
        },
    )
    with torch.no_grad():
        for index in (0, 2):
            layer = net.layer_list[index]
            layer.sublayer_list[0].weight.fill_(1.0)
            layer.bit_scale_list[1, 1] = 1.0
            layer.last_value.fill_(1.0)
    net.eval()
    return net


def test_binary_spike_domain_reaches_every_weighted_layer_without_conversion(
    monkeypatch,
):
    net = _tiny_two_stage_snn()
    observed = []
    original = QuantizeLayer.set_weights_forward

    def capture(self, input, bit_weights, adc_action, **kwargs):
        observed.append((input.detach().clone(), kwargs["input_quantization"]))
        return original(self, input, bit_weights, adc_action, **kwargs)

    monkeypatch.setattr(QuantizeLayer, "set_weights_forward", capture)
    sequence = torch.zeros(1, 2, 1, 1, 1)
    sequence[:, 0] = 1
    net.sequence_accuracy_forward(
        sequence,
        weighted_mode="PIM",
        adc_action="SCALE",
        net_bit_weights=net.get_weights(),
        pim_profile="adc_digital_lif",
    )

    assert len(observed) == 2
    assert all(domain is BINARY_SPIKE_DOMAIN for _value, domain in observed)
    assert all(
        set(value.unique().tolist()) <= {0.0, 1.0}
        for value, _domain in observed
    )
```

Add to `tests/test_snn_original_accuracy.py`:

```python
def test_original_and_pim_use_the_same_multistep_lif_entry(monkeypatch):
    net = _tiny_original_network()
    with torch.no_grad():
        weighted = net.layer_list[0]
        weighted.sublayer_list[0].weight.fill_(1.0)
        weighted.bit_scale_list[1, 1] = 1.0
        weighted.last_value.fill_(1.0)
    net.eval()
    calls = []
    lif = net.layer_list[1].layer
    original = lif.forward_sequence

    def capture(value):
        calls.append(tuple(value.shape))
        return original(value)

    monkeypatch.setattr(lif, "forward_sequence", capture)
    sequence = torch.tensor([[[[[1.0]]], [[[0.0]]]]])
    net.sequence_accuracy_forward(sequence, weighted_mode="ORIGINAL")
    net.sequence_accuracy_forward(
        sequence,
        weighted_mode="PIM",
        adc_action="SCALE",
        net_bit_weights=net.get_weights(),
        pim_profile="adc_digital_lif",
    )

    assert calls == [(2, 1, 1, 1, 1), (2, 1, 1, 1, 1)]
```

- [x] **Step 2: Run the tests and verify PIM still uses signed activation state and step-by-step LIF**

Run:

```bash
$PYTHON -m pytest \
  tests/test_snn_unified_accuracy.py \
  tests/test_snn_original_accuracy.py -q
```

Expected: the new tests fail because `sequence_accuracy_forward()` has no
`pim_profile`, initial/LIF domains are `ActivationQuantizationState`, and PIM LIF
does not call `forward_sequence()`.

- [x] **Step 3: Teach `QuantizeLayer` to consume `BinarySpikeDomain` exactly**

In `MNSIM/Interface/quantize.py`, import `BinarySpikeDomain` and change
`_resolve_input_quantization()` to accept either numerical-domain type:

```python
from MNSIM.Accuracy_Model.snn_accuracy_contract import BinarySpikeDomain


def _resolve_input_quantization(self, input_quantization):
    if input_quantization is None:
        return _legacy_activation_quantization_state()
    if not isinstance(
        input_quantization, (ActivationQuantizationState, BinarySpikeDomain)
    ):
        raise TypeError(
            "input_quantization must be an ActivationQuantizationState, "
            "BinarySpikeDomain, or None"
        )
    return input_quantization
```

Inside `set_weights_forward()`, replace the signed input-code construction with
an explicit branch:

```python
if isinstance(input_state, BinarySpikeDomain):
    activation_in_scale = input_state.scale
    activation_in_cycle = input_state.activation_cycle
    activation_in_container = [input_state.validate(input_list[layer_num])]
    if adc_diagnostic is not None:
        _record_input_diagnostic(adc_diagnostic, input_list[layer_num], 0, 1)
else:
    activation_in_bit = input_state.bit
    activation_in_scale = input_state.scale
    thres = 2 ** (activation_in_bit - 1) - 1
    input_pre_quantized = input_list[layer_num] / activation_in_scale
    if adc_diagnostic is not None:
        _record_input_diagnostic(adc_diagnostic, input_pre_quantized, -thres, thres)
    activation_in_digit = torch.clamp(
        torch.round(input_pre_quantized), -thres, thres
    )
    activation_in_cycle = math.ceil(
        (activation_in_bit - 1) / self.hardware_config["input_bit"]
    )
    sign_activation_in = torch.sign(activation_in_digit)
    activation_in_digit = torch.abs(activation_in_digit)
    base = 1
    step = 2 ** self.hardware_config["input_bit"]
    activation_in_container = []
    for _cycle in range(activation_in_cycle):
        digit = (
            torch.fmod(activation_in_digit, base * step)
            - torch.fmod(activation_in_digit, base)
        )
        activation_in_container.append(torch.mul(sign_activation_in, digit) / base)
        base *= step
```

Do not construct an `ActivationQuantizationState` whose `bit` argument is `1`
anywhere.

- [x] **Step 4: Unify Original and PIM around the same multi-step LIF call**

In `MNSIM/Interface/network.py`:

```python
from MNSIM.Accuracy_Model.snn_accuracy_contract import (
    ADC_DIGITAL_LIF,
    BINARY_SPIKE_DOMAIN,
    SUPPORTED_SNN_HARDWARE_PROFILES,
)
```

Extend the signature and validation:

```python
def sequence_accuracy_forward(
    self,
    x,
    *,
    weighted_mode,
    adc_action="SCALE",
    net_bit_weights=None,
    pim_profile=None,
    lif_perturbation=None,
):
    if not torch.is_tensor(x) or x.dim() != 5:
        raise ValueError(
            f"expected sequence batch [B,T,C,H,W], got {getattr(x, 'shape', None)}"
        )
    weighted_mode = str(weighted_mode).upper()
    if weighted_mode not in {"ORIGINAL", "PIM"}:
        raise ValueError(f"unsupported sequence weighted mode: {weighted_mode}")
    if weighted_mode == "PIM" and net_bit_weights is None:
        raise ValueError("PIM sequence mode requires net_bit_weights")
    if weighted_mode == "PIM" and pim_profile not in SUPPORTED_SNN_HARDWARE_PROFILES:
        raise ValueError(f"unsupported SNN PIM hardware profile: {pim_profile}")
```

Set sequence input state to the binary singleton for both modes:

```python
BINARY_SPIKE_DOMAIN.validate(x)
input_state = BINARY_SPIKE_DOMAIN
```

Replace the LIF branch with one shared call:

```python
elif getattr(layer, "layer_config", {}).get("type") == "lif":
    layer.set_accuracy_perturbation(
        lif_perturbation if weighted_mode == "PIM" else None
    )
    y = layer.layer.forward_sequence(x_in)
```

Set its output domain explicitly:

```python
if getattr(layer, "layer_config", {}).get("type") == "lif":
    q_out = BINARY_SPIKE_DOMAIN
elif weighted_mode == "PIM":
    q_out = layer.output_activation_quantization(q_in)
else:
    q_out = None
```

Add a no-op `StraightLayer.set_accuracy_perturbation(runtime)` that delegates to
the wrapped layer only when it exposes that method. Task 7 will make the LIF
implementation consume the runtime.

- [x] **Step 5: Run the focused tests**

```bash
$PYTHON -m pytest \
  tests/test_snn_unified_accuracy.py \
  tests/test_snn_original_accuracy.py -q
```

Expected: all tests pass; both paths call the same multi-step entry and every
SNN weighted layer receives exact 0/1 input after an input boundary or LIF.

- [x] **Step 6: Commit the shared sequence/domain work**

```bash
$GIT add \
  MNSIM/Interface/quantize.py \
  MNSIM/Interface/network.py \
  tests/test_snn_unified_accuracy.py \
  tests/test_snn_original_accuracy.py
$GIT diff --cached --check
$GIT commit -m "feat: preserve binary spike domains in SNN accuracy"
```

### Task 3: Add A True No-ADC Weighted Reconstruction Branch

**Files:**

- Create: `tests/test_snn_no_adc_reconstruction.py`
- Modify: `MNSIM/Interface/quantize.py`
- Modify: `MNSIM/Interface/network.py`
- Modify: `tests/test_pim_numerical_audit.py`

- [x] **Step 1: Write failing tests that forbid ADC and post-weight activation quantization**

Create `tests/test_snn_no_adc_reconstruction.py`:

```python
import numpy as np
import torch

import MNSIM.Interface.quantize as quantize_module
from MNSIM.Accuracy_Model.snn_accuracy_contract import BINARY_SPIKE_DOMAIN
from MNSIM.Interface.quantize import QuantizeLayer


def _fc_layer():
    layer = object.__new__(QuantizeLayer)
    torch.nn.Module.__init__(layer)
    layer.hardware_config = {
        "xbar_polarity": 2,
        "weight_bit": 8,
        "input_bit": 1,
        "ADC_quantize_bit": 5,
        "xbar_size": 2,
        "type": 0,
        "ADC_quantization_mode": "READC_LLOYD_MAX",
    }
    layer.quantize_config = {
        "weight_bit": 9,
        "activation_bit": 9,
        "point_shift": 0,
    }
    layer.layer_config = {"type": "fc", "extend_ADC_bitwidth": 0}
    layer.split_input = 2
    layer.sublayer_list = torch.nn.ModuleList(
        [torch.nn.Linear(2, 1, bias=False)]
    )
    layer.last_value = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
    layer.bit_scale_list = torch.nn.Parameter(
        torch.tensor([[9.0, 1.0], [9.0, 0.25], [9.0, 1.0 / 255.0]]),
        requires_grad=False,
    )
    layer.eval()
    return layer


def test_no_adc_reconstruction_matches_the_bit_sliced_weighted_sum(monkeypatch):
    layer = _fc_layer()

    def forbidden_adc(*_args, **_kwargs):
        raise AssertionError("LUT profile must not call an ADC quantizer")

    def forbidden_output_quantizer(*_args, **_kwargs):
        raise AssertionError("LUT profile must not quantize weighted output")

    monkeypatch.setattr(quantize_module, "_adc_quantize", forbidden_adc)
    monkeypatch.setattr(layer, "output_activation_quantization", forbidden_output_quantizer)
    bit_weights = {
        "split0_weight0_positive": np.array([[2.0, 0.0]]),
        "split0_weight0_negative": np.array([[0.0, 1.0]]),
    }
    spikes = torch.tensor([[1.0, 1.0]])

    actual = layer.set_weights_forward(
        spikes,
        bit_weights,
        adc_action=None,
        numerical_mode="NO_ADC_RECONSTRUCTION",
        input_quantization=BINARY_SPIKE_DOMAIN,
    )

    assert torch.equal(actual, torch.tensor([[0.25]]))


def test_no_adc_reconstruction_rejects_nonbinary_weighted_input():
    layer = _fc_layer()
    with torch.no_grad():
        try:
            layer.set_weights_forward(
                torch.tensor([[1.0, 0.25]]),
                {
                    "split0_weight0_positive": np.array([[1.0, 0.0]]),
                    "split0_weight0_negative": np.array([[0.0, 0.0]]),
                },
                adc_action=None,
                numerical_mode="NO_ADC_RECONSTRUCTION",
                input_quantization=BINARY_SPIKE_DOMAIN,
            )
        except ValueError as error:
            assert "binary spike" in str(error)
        else:
            raise AssertionError("nonbinary LUT-profile input must fail")
```

Keep the existing
`test_audit_none_is_exact_bit_slice_baseline_without_adc_clamping()` unchanged;
it is the regression that proves `adc_action="NONE"` remains audit-only. The new
LUT test above must use `numerical_mode="NO_ADC_RECONSTRUCTION"` and
`adc_action=None`, so the two branches cannot be confused.

- [x] **Step 2: Run the tests and verify the new numerical mode is unsupported**

```bash
$PYTHON -m pytest \
  tests/test_snn_no_adc_reconstruction.py \
  tests/test_pim_numerical_audit.py -q
```

Expected: the new test fails because `set_weights_forward()` has no
`numerical_mode` parameter.

- [x] **Step 3: Implement exact digital reconstruction without ADC or output quantization**

In `MNSIM/Interface/quantize.py`, extend the method signature:

```python
def set_weights_forward(
    self,
    input,
    bit_weights,
    adc_action,
    adc_diagnostics=None,
    layer_index=None,
    input_quantization=None,
    numerical_mode="ADC",
):
```

Validate the internal mode at entry:

```python
if numerical_mode not in {"ADC", "NO_ADC_RECONSTRUCTION", "LEGACY_NONE"}:
    raise ValueError(f"unsupported weighted numerical mode: {numerical_mode}")
if numerical_mode == "NO_ADC_RECONSTRUCTION" and adc_action is not None:
    raise ValueError("NO_ADC_RECONSTRUCTION requires adc_action=None")
```

For each partial sum, use the existing bit-slice significance arithmetic but
skip `_adc_quantize()` in the new branch:

```python
if numerical_mode == "NO_ADC_RECONSTRUCTION":
    tmp = tmp * weight_scale * activation_in_scale
    tmp = tmp / scale * (
        2
        ** (
            (activation_in_cycle - 1) * self.hardware_config["input_bit"]
            + (weight_bit_split_part - 1) * self.hardware_config["weight_bit"]
        )
    )
```

Insert this branch immediately before the current
`if adc_action in {'SCALE', 'ADAPTIVE', 'READC_CALIBRATE'}` block and change that
current `if` to `elif`. Leave the complete existing ADC and `FIX` bodies
unchanged; the only new behavior in the partial-sum loop is the branch shown
above.

The final output handling must be:

```python
if numerical_mode == "NO_ADC_RECONSTRUCTION":
    output_final = output_final * scale
elif numerical_mode == "LEGACY_NONE" or adc_action == "NONE":
    activation_out_bit = output_state.bit
    threshold = 2 ** (activation_out_bit - 1) - 1
    output_final = torch.clamp(
        torch.round(output_final * threshold), -threshold, threshold
    ) * output_state.scale
else:
    output_final = output_final * scale
```

Do not call `output_activation_quantization()` at all when
`numerical_mode == "NO_ADC_RECONSTRUCTION"`; set `output_state = None` for that
branch before entering the loops. Keep `adc_action="NONE"` as legacy-only so
existing numerical audit tests retain their historical meaning.

- [x] **Step 4: Route the LUT profile to the exact reconstruction branch**

In `NetworkGraph.sequence_accuracy_forward()` choose the weighted call without
creating another public accuracy mode:

```python
if pim_profile == FRONTEND_LUT_LIF_LUT:
    y_merged = layer.set_weights_forward(
        merged,
        next(pim_weights),
        None,
        input_quantization=q_in,
        numerical_mode="NO_ADC_RECONSTRUCTION",
    )
else:
    y_merged = layer.set_weights_forward(
        merged,
        next(pim_weights),
        adc_action,
        input_quantization=q_in,
        numerical_mode="ADC",
    )
```

For a LUT-profile weighted output set `q_out = None`; the following LIF consumes
software numerical units and then emits `BINARY_SPIKE_DOMAIN`.

- [x] **Step 5: Run focused and legacy numerical tests**

```bash
$PYTHON -m pytest \
  tests/test_snn_no_adc_reconstruction.py \
  tests/test_pim_numerical_audit.py \
  tests/test_no_post_adc_requantization.py -q
```

Expected: all tests pass. The new LUT branch never invokes `_adc_quantize()` or
the weighted output activation quantizer; the existing ADC and legacy audit
branches retain their previous results.

- [x] **Step 6: Commit the no-ADC branch**

```bash
$GIT add \
  MNSIM/Interface/quantize.py \
  MNSIM/Interface/network.py \
  tests/test_snn_no_adc_reconstruction.py \
  tests/test_pim_numerical_audit.py
$GIT diff --cached --check
$GIT commit -m "feat: add exact no-ADC SNN weighted reconstruction"
```

### Task 4: Sample The Formal Device Instance Once Per PIM Accuracy Run

**Files:**

- Modify: `MNSIM/Accuracy_Model/Weight_update.py`
- Modify: `MNSIM/Interface/round_orchestrated_runner.py`
- Modify: `tests/test_device_nonidealities.py`
- Modify: `tests/test_round_orchestrated_runner.py`

- [x] **Step 1: Write failing runner tests for fixed controls, explicit RNG, and one-time sampling**

Add a formal SNN fake context to `tests/test_round_orchestrated_runner.py` whose
`hardware_config` selects `adc_digital_lif`, and add:

```python
def test_formal_snn_pim_uses_locked_device_controls_and_seed_112(
    tmp_path, dependency_factory, monkeypatch
):
    context = FakeContext([])
    context.hardware_config = {"ADC_quantization_mode": "UNIFORM"}
    dependencies, calls, _, context, _, _ = dependency_factory(
        context=context
    )
    context.calls = calls
    observed = []

    monkeypatch.setattr(
        runner_module,
        "_load_accuracy_device_contract",
        lambda _path: SimpleNamespace(
            model_id="rram_20ns_linear_fit",
            variation_percent=6.69903802061613,
            device_seed=112,
        ),
    )

    def capture_weight_update(sim_config, net_bits, **kwargs):
        observed.append(kwargs)
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
    report = _artifact(request, "accuracy_report.json")
    assert report["accuracy_device_seed"] == 112
    assert report["accuracy_variation_sampling_scope"] == (
        "once_per_physical_cell_per_pim_accuracy_run"
    )


def test_formal_snn_reuses_one_post_variation_mapping_for_calibration_and_test(
    tmp_path, dependency_factory, monkeypatch
):
    context = FakeContext([])
    context.hardware_config = {"ADC_quantization_mode": "READC_LLOYD_MAX"}
    dependencies, calls, _, context, _, _ = dependency_factory(
        context=context
    )
    context.calls = calls
    monkeypatch.setattr(
        runner_module,
        "_load_accuracy_device_contract",
        lambda _path: SimpleNamespace(
            model_id="rram_20ns_linear_fit",
            variation_percent=6.69903802061613,
            device_seed=112,
        ),
    )
    mapped = object()
    dependencies.weight_update = lambda *_args, **_kwargs: mapped
    context.calibrate_readc = lambda value, **_kwargs: calls.append(
        ("calibrate", value)
    ) or {"L000": {"boundaries": [], "levels": []}}
    context.set_net_bits_evaluate = lambda value, **_kwargs: calls.append(
        ("test", value)
    ) or 0.87

    result = run_round_orchestrated_simulation(
        _request(
            tmp_path,
            network="standard_vgg8_snn_v7_phase6",
            run_original=False,
        ),
        dependencies=dependencies,
    )

    assert result.exit_code == 0
    assert ("calibrate", mapped) in calls
    assert ("test", mapped) in calls
```

Add to `tests/test_device_nonidealities.py`:

```python
class _RecordingGenerator:
    def __init__(self, seed):
        self.generator = np.random.default_rng(seed)
        self.normal_draws = []

    def normal(self, *args, **kwargs):
        value = self.generator.normal(*args, **kwargs)
        self.normal_draws.append(np.array(value, copy=True))
        return value

    def random(self, *args, **kwargs):
        return self.generator.random(*args, **kwargs)


def test_20ns_and_30ns_use_paired_random_sequence_with_device_specific_sigma():
    codes = np.arange(16, dtype=np.float64).reshape(4, 4)
    resistance = tuple(np.linspace(1.0e6, 1.0e4, 256))
    rng20 = _RecordingGenerator(112)
    rng30 = _RecordingGenerator(112)
    out20 = apply_cell_nonidealities(
        codes,
        DeviceNonidealityConfig(resistance, 6.69903802061613, 0.0, 0.0),
        enable_r_ratio=True,
        enable_variation=True,
        enable_saf=False,
        rng=rng20,
    )
    out30 = apply_cell_nonidealities(
        codes,
        DeviceNonidealityConfig(resistance, 7.725782066255718, 0.0, 0.0),
        enable_r_ratio=True,
        enable_variation=True,
        enable_saf=False,
        rng=rng30,
    )

    z20 = (rng20.normal_draws[0] - 1.0) / (6.69903802061613 / 100.0)
    z30 = (rng30.normal_draws[0] - 1.0) / (7.725782066255718 / 100.0)
    assert np.allclose(z20, z30)
    assert not np.allclose(out20, out30)
```

- [x] **Step 2: Run tests and verify the runner still uses request defaults and global NumPy scoping**

```bash
$PYTHON -m pytest \
  tests/test_device_nonidealities.py \
  tests/test_round_orchestrated_runner.py -q
```

Expected: new SNN tests fail because the runner passes request booleans, omits
`rng`, and allows `accuracy_seed=None`.

- [x] **Step 3: Use an explicit generator in `weight_update()` without breaking legacy callers**

Retain the existing optional `rng` argument in
`MNSIM/Accuracy_Model/Weight_update.py`, but make the legacy adapter local and
document that the formal SNN runner always passes a generator:

```python
if rng is None:
    class _LegacyGenerator:
        normal = staticmethod(np.random.normal)
        random = staticmethod(np.random.random_sample)

    rng = _LegacyGenerator()
elif not hasattr(rng, "normal") or not hasattr(rng, "random"):
    raise TypeError("rng must provide normal() and random()")
```

Do not create a generator per layer, split, sample, batch, or timestep.

- [x] **Step 4: Resolve formal SNN controls from the validated final-RRAM config**

In `round_orchestrated_runner.py`, add a helper that reads the already validated
SimConfig and returns `load_snn_accuracy_device_contract(parser, source=path)`. In
`_evaluate_accuracy()` use:

```python
is_snn = "snn" in request.network.lower()
if is_snn and request.run_pim:
    device_contract = _load_accuracy_device_contract(request.sim_config)
    accuracy_rng = np.random.default_rng(device_contract.device_seed)
    updated_bits = _quiet_call(
        deps.weight_update,
        str(request.sim_config),
        net_bits,
        is_SAF=0,
        is_Variation=1,
        is_Rratio=1,
        rng=accuracy_rng,
    )
else:
    with _temporary_accuracy_seed(request.accuracy_seed):
        updated_bits = _quiet_call(
            deps.weight_update,
            str(request.sim_config),
            net_bits,
            is_SAF=int(request.enable_saf),
            is_Variation=int(request.enable_variation),
            is_Rratio=int(request.enable_r_ratio),
        )
```

Call `weight_update()` exactly once. Keep the returned object and pass that same
object to both ReADC calibration and test inference. Do not run it again when
switching from calibration to test.

Report these exact fields for formal SNN PIM:

```python
report.update(
    {
        "accuracy_device_model": device_contract.model_id,
        "accuracy_device_variation_percent": device_contract.variation_percent,
        "accuracy_device_seed": device_contract.device_seed,
        "accuracy_variation_sampling_scope": (
            "once_per_physical_cell_per_pim_accuracy_run"
        ),
    }
)
```

The existing ANN request controls and `_temporary_accuracy_seed()` remain for
ANN regression only; they must not determine formal SNN PIM variation.

In `_validate_inputs()`, reject a formal SNN request whose explicitly supplied
`accuracy_seed` is not `112`:

```python
if (
    "snn" in request.network.lower()
    and request.run_pim
    and request.accuracy_seed not in {None, 112}
):
    return ErrorRecord(
        code=INVALID_ACCURACY_SEED,
        detail="formal SNN PIM accuracy seed is fixed to 112",
    )
```

For formal SNN reports, replace the request-derived `controls` values with:

```python
"controls": {
    "enable_saf": False,
    "enable_variation": True,
    "enable_r_ratio": True,
    "accuracy_seed": 112,
    "source": "final_rram_contract",
}
```

- [x] **Step 5: Run focused tests**

```bash
$PYTHON -m pytest \
  tests/test_device_nonidealities.py \
  tests/test_final_rram_contract.py \
  tests/test_round_orchestrated_runner.py -q
```

Expected: all tests pass, including existing ANN tests that still expect their
legacy request-controlled behavior.

- [x] **Step 6: Commit fixed one-time device sampling**

```bash
$GIT add \
  MNSIM/Accuracy_Model/Weight_update.py \
  MNSIM/Interface/round_orchestrated_runner.py \
  tests/test_device_nonidealities.py \
  tests/test_round_orchestrated_runner.py
$GIT diff --cached --check
$GIT commit -m "fix: lock SNN PIM accuracy device sampling"
```

### Task 5: Build And Enforce The ReADC Calibration Artifact

**Files:**

- Create: `MNSIM/Accuracy_Model/readc_artifact.py`
- Create: `tests/test_readc_artifact.py`
- Modify: `MNSIM/Accuracy_Model/readc.py`
- Modify: `MNSIM/Interface/dvs_binned_npz.py`
- Modify: `MNSIM/Interface/interface.py`
- Modify: `MNSIM/Interface/round_orchestrated_artifacts.py`
- Modify: `MNSIM/Interface/round_orchestrated_runner.py`
- Modify: `tests/test_readc.py`
- Modify: `tests/test_adc_adaptive_quantization.py`

- [x] **Step 1: Write failing artifact tests**

Create `tests/test_readc_artifact.py` with:

```python
import copy

import pytest

from MNSIM.Accuracy_Model.readc_artifact import (
    ReADCArtifactError,
    ReADCArtifactIdentity,
    build_readc_artifact,
    validate_readc_artifact,
)


def _identity(**changes):
    values = {
        "checkpoint_sha256": "a" * 64,
        "dataset_split_identity": {"dataset": "v7", "split": "train"},
        "device_model_id": "rram_20ns_linear_fit",
        "device_seed": 112,
        "weight_mapping_sha256": "b" * 64,
        "adc_precision": 5,
        "percentile_low": 0.5,
        "percentile_high": 99.5,
        "max_samples_per_layer": 8000,
    }
    values.update(changes)
    return ReADCArtifactIdentity(**values)


def _table():
    return {
        "boundaries": [float(index) for index in range(31)],
        "levels": [float(index) + 0.5 for index in range(32)],
        "sample_count": 32,
        "observed_count": 64,
        "observed_min": 0.0,
        "observed_max": 31.0,
        "iterations": 101,
        "mse": 0.1,
        "sampling_strategy": "stream_weighted_deterministic_fixed_capacity",
    }


def test_readc_artifact_round_trip_binds_identity_and_layer_tables():
    artifact = build_readc_artifact(_identity(), {"L000": _table()})
    validated = validate_readc_artifact(artifact.to_dict(), _identity())
    assert validated.artifact_checksum == artifact.artifact_checksum
    assert validated.layers["L000"]["table_checksum"]


@pytest.mark.parametrize(
    "changed_identity",
    [
        _identity(checkpoint_sha256="c" * 64),
        _identity(device_model_id="rram_30ns_linear_fit"),
        _identity(device_seed=113),
        _identity(weight_mapping_sha256="d" * 64),
    ],
)
def test_readc_artifact_rejects_any_runtime_identity_mismatch(changed_identity):
    artifact = build_readc_artifact(_identity(), {"L000": _table()})
    with pytest.raises(ReADCArtifactError, match="identity mismatch"):
        validate_readc_artifact(artifact.to_dict(), changed_identity)


def test_readc_artifact_rejects_tampered_or_malformed_tables():
    artifact = build_readc_artifact(_identity(), {"L000": _table()}).to_dict()
    artifact["layers"]["L000"]["boundaries"][1] = -10.0
    with pytest.raises(ReADCArtifactError):
        validate_readc_artifact(artifact, _identity())
```

Add a DVS dataset test to `tests/test_snn_original_accuracy.py`:

```python
def test_v7_dataset_split_identity_is_deterministic_and_split_specific(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "meta.json").write_text(
        '{"T_bins": 2, "C": 6, "H": 4, "W": 4, "num_classes": 8}',
        encoding="utf-8",
    )
    sample = np.zeros((2, 6, 4, 4), dtype=np.uint8)
    np.savez(tmp_path / "train" / "sample.npz", x_seq=sample, y=0)
    np.savez(tmp_path / "test" / "sample.npz", x_seq=sample, y=0)
    train = DVSBinnedNPZDataset(str(tmp_path), "train", target_hw=(2, 2))
    test = DVSBinnedNPZDataset(str(tmp_path), "test", target_hw=(2, 2))
    assert train.accuracy_split_identity() == train.accuracy_split_identity()
    assert train.accuracy_split_identity()["split"] == "train"
    assert train.accuracy_split_identity() != test.accuracy_split_identity()
```

- [x] **Step 2: Run tests and verify the artifact module and split identity are missing**

```bash
$PYTHON -m pytest \
  tests/test_readc_artifact.py \
  tests/test_snn_original_accuracy.py -q
```

Expected: import/method failures for the new artifact and split identity APIs.

- [x] **Step 3: Implement deterministic ReADC artifact serialization and validation**

Create `MNSIM/Accuracy_Model/readc_artifact.py` with:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from MNSIM.Accuracy_Model.snn_accuracy_contract import canonical_json_sha256


class ReADCArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class ReADCArtifactIdentity:
    checkpoint_sha256: str
    dataset_split_identity: dict[str, Any]
    device_model_id: str
    device_seed: int
    weight_mapping_sha256: str
    adc_precision: int
    percentile_low: float
    percentile_high: float
    max_samples_per_layer: int


@dataclass(frozen=True)
class ReADCCalibrationArtifact:
    schema_version: int
    identity: ReADCArtifactIdentity
    layers: dict[str, dict[str, Any]]
    artifact_checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity": asdict(self.identity),
            "layers": self.layers,
            "artifact_checksum": self.artifact_checksum,
        }


def _validated_table(table: dict[str, Any], precision: int) -> dict[str, Any]:
    boundaries = [float(value) for value in table["boundaries"]]
    levels = [float(value) for value in table["levels"]]
    if len(levels) != 2 ** precision or len(boundaries) != len(levels) - 1:
        raise ReADCArtifactError("ReADC table size does not match ADC precision")
    if any(right <= left for left, right in zip(boundaries, boundaries[1:])):
        raise ReADCArtifactError("ReADC boundaries must be strictly increasing")
    if int(table.get("observed_count", 0)) <= 0:
        raise ReADCArtifactError("ReADC calibration observed no samples")
    normalized = dict(table)
    normalized["boundaries"] = boundaries
    normalized["levels"] = levels
    normalized["table_checksum"] = canonical_json_sha256(
        {key: normalized[key] for key in sorted(normalized) if key != "table_checksum"}
    )
    return normalized


def build_readc_artifact(identity, layers):
    if identity.adc_precision != 5:
        raise ReADCArtifactError("formal ReADC artifact requires 5-bit precision")
    normalized_layers = {
        layer_id: _validated_table(table, identity.adc_precision)
        for layer_id, table in sorted(layers.items())
    }
    body = {
        "schema_version": 1,
        "identity": asdict(identity),
        "layers": normalized_layers,
    }
    return ReADCCalibrationArtifact(
        schema_version=1,
        identity=identity,
        layers=normalized_layers,
        artifact_checksum=canonical_json_sha256(body),
    )


def validate_readc_artifact(payload, expected_identity):
    if payload.get("schema_version") != 1:
        raise ReADCArtifactError("unsupported ReADC artifact schema")
    actual_identity = ReADCArtifactIdentity(**payload["identity"])
    if actual_identity != expected_identity:
        raise ReADCArtifactError("ReADC artifact identity mismatch")
    rebuilt = build_readc_artifact(actual_identity, payload["layers"])
    if payload.get("artifact_checksum") != rebuilt.artifact_checksum:
        raise ReADCArtifactError("ReADC artifact checksum mismatch")
    for layer_id, table in rebuilt.layers.items():
        if payload["layers"][layer_id].get("table_checksum") != table["table_checksum"]:
            raise ReADCArtifactError(f"ReADC table checksum mismatch for {layer_id}")
    return rebuilt
```

Use explicit exception wrapping for missing keys/type conversion so every bad
artifact raises `ReADCArtifactError`, not a raw `KeyError` or `TypeError`.

- [x] **Step 4: Add deterministic split identity to the V7 dataset**

In `DVSBinnedNPZDataset.__init__`, retain `self.split_dir`. Add:

```python
def accuracy_split_identity(self):
    entries = [
        {
            "relative_path": path.relative_to(self.split_dir).as_posix(),
            "size_bytes": path.stat().st_size,
        }
        for path in map(Path, self.files)
    ]
    return {
        "dataset": f"{type(self).__module__}.{type(self).__qualname__}",
        "root": str(Path(self.root).resolve()),
        "split": self.split,
        "sample_count": len(entries),
        "sample_manifest_sha256": canonical_json_sha256(entries),
    }
```

Import `Path` and `canonical_json_sha256`. This method reads only metadata for
the already selected split and never constructs or indexes the test dataset
during calibration.

- [x] **Step 5: Return and install a validated artifact in `TrainTestInterface`**

Change `calibrate_readc()` to accept an identity and return an artifact:

```python
def calibrate_readc(self, net_bit_weights, *, identity, max_samples=None):
    from MNSIM.Accuracy_Model.readc import (
        ReADCCalibrationConfig,
        ReADCSampleCollector,
        lloyd_max,
    )
    from MNSIM.Accuracy_Model.readc_artifact import (
        ReADCArtifactError,
        build_readc_artifact,
    )

    loaders = import_module(self.dataset_module).get_dataloader()
    self.calibration_loader = loaders[0]
    dataset = self.calibration_loader.dataset
    actual_split_identity = dataset.accuracy_split_identity()
    if actual_split_identity != identity.dataset_split_identity:
        raise ReADCArtifactError("calibration split identity mismatch")
    config = ReADCCalibrationConfig(
        precision=5,
        percentile_low=0.5,
        percentile_high=99.5,
        max_samples_per_layer=8000,
    )
    weighted_layers = [
        (network_index, layer)
        for network_index, layer in enumerate(self.net.layer_list)
        if hasattr(layer, "hardware_config")
        and hasattr(layer, "set_weights_forward")
    ]
    for _network_index, layer in weighted_layers:
        layer.hardware_config["ADC_readc_collector"] = ReADCSampleCollector(
            config.max_samples_per_layer
        )
        layer.hardware_config.pop("ADC_readc_table", None)
    self.net.to(self.device)
    self.net.eval()
    seen = 0
    with torch.no_grad():
        for images, _labels in self.calibration_loader:
            if max_samples is not None and seen >= max_samples:
                break
            images = images.to(self.device)
            if max_samples is not None:
                images = images[: max_samples - seen]
            seen += images.shape[0]
            self._accuracy_forward(
                images,
                method="SET_WEIGHTS",
                adc_action="READC_CALIBRATE",
                net_bit_weights=net_bit_weights,
                pim_profile="adc_digital_lif",
                lif_perturbation=None,
            )
    layer_tables = {}
    for network_index, layer in weighted_layers:
        collector = layer.hardware_config.pop("ADC_readc_collector")
        table = lloyd_max(collector.values(), config)
        table["observed_count"] = collector.observed_count
        table["sampling_strategy"] = (
            "stream_weighted_deterministic_fixed_capacity"
        )
        layer_tables[f"L{network_index:03d}"] = table
    return build_readc_artifact(identity, layer_tables)
```

Build `weighted_layers` as `(network_index, layer)` pairs. During collection,
tables are not considered installed: the collector is present and
`READC_CALIBRATE` returns the unquantized pre-ADC value.

Add:

```python
def install_readc_artifact(self, artifact, *, expected_identity):
    validated = validate_readc_artifact(artifact.to_dict(), expected_identity)
    weighted_layers = [
        (index, layer)
        for index, layer in enumerate(self.net.layer_list)
        if hasattr(layer, "hardware_config")
        and hasattr(layer, "set_weights_forward")
    ]
    expected_ids = {f"L{index:03d}" for index, _layer in weighted_layers}
    if set(validated.layers) != expected_ids:
        raise ReADCArtifactError("ReADC layer identity set mismatch")
    for index, layer in weighted_layers:
        layer.hardware_config["ADC_readc_table"] = validated.layers[f"L{index:03d}"]
    self.readc_artifact = validated
```

Before any ADC-profile test inference, require `self.readc_artifact` to be
installed and identity-matched. Do not fall back to `SCALE`, `FIX`, uniform ADC,
or static thresholds.

- [x] **Step 6: Make `readc_calibration.json` a runner-owned artifact**

Add `"readc_calibration.json"` to both `RUNNER_OWNED_ARTIFACTS` and the runner's
`_ARTIFACT_KEYS`. When an ADC-profile calibration succeeds, write
`artifact.to_dict()` before test inference, then install the same validated
artifact into the context.

- [x] **Step 7: Run the focused ReADC tests**

```bash
$PYTHON -m pytest \
  tests/test_readc.py \
  tests/test_readc_artifact.py \
  tests/test_adc_adaptive_quantization.py \
  tests/test_no_post_adc_requantization.py \
  tests/test_snn_original_accuracy.py -q
```

Expected: all tests pass. Malformed or mismatched artifacts fail before test
inference, and the calibration identity explicitly names the training split.

- [x] **Step 8: Commit the ReADC artifact boundary**

```bash
$GIT add \
  MNSIM/Accuracy_Model/readc.py \
  MNSIM/Accuracy_Model/readc_artifact.py \
  MNSIM/Interface/dvs_binned_npz.py \
  MNSIM/Interface/interface.py \
  MNSIM/Interface/round_orchestrated_artifacts.py \
  MNSIM/Interface/round_orchestrated_runner.py \
  tests/test_readc.py \
  tests/test_readc_artifact.py \
  tests/test_adc_adaptive_quantization.py \
  tests/test_snn_original_accuracy.py
$GIT diff --cached --check
$GIT commit -m "feat: persist identity-bound ReADC calibration"
```

### Task 6: Define The Qualified Frontend/LIF Perturbation Artifact

**Files:**

- Create: `MNSIM/Accuracy_Model/snn_lif_perturbation.py`
- Create: `tests/test_snn_lif_perturbation.py`

- [x] **Step 1: Write failing schema and sampling tests**

Create `tests/test_snn_lif_perturbation.py` with this checksummed fixture and the
tests below:

```python
import json

import numpy as np
import pytest

from MNSIM.Accuracy_Model.snn_accuracy_contract import canonical_json_sha256

from MNSIM.Accuracy_Model.snn_lif_perturbation import (
    LIFPerturbationArtifactError,
    LIFPerturbationRuntime,
    load_lif_perturbation_artifact,
)


PARAMETERS = (
    "frontend_gain",
    "frontend_offset",
    "delta_tau",
    "delta_v_th",
    "delta_v_reset",
)


@pytest.fixture
def qualified_artifact_path(tmp_path):
    body = {
        "schema_version": 1,
        "profile_id": "frontend_lut_lif_lut",
        "qualification": {
            "pdk": "test-pdk",
            "model_files": [
                {"path": "models.scs", "sha256": "a" * 64}
            ],
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
            "mean": [1.0, 0.0, 0.0, 0.0, 0.0],
            "covariance": [
                [1.0e-4, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0e-8, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0e-4, 2.0e-7, 0.0],
                [0.0, 0.0, 2.0e-7, 1.0e-8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0e-8],
            ],
        },
        "supported_operating_domain": {
            "timestep_s": 1.0e-6,
            "input_min": -1.0,
            "input_max": 1.0,
        },
        "dynamic_noise": None,
    }
    payload = dict(body)
    payload["artifact_checksum"] = canonical_json_sha256(body)
    path = tmp_path / "qualified_lif_perturbation.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_qualified_artifact_preserves_joint_distribution_and_checksum(
    qualified_artifact_path,
):
    artifact = load_lif_perturbation_artifact(qualified_artifact_path)
    assert artifact.profile_id == "frontend_lut_lif_lut"
    assert artifact.parameter_order == PARAMETERS
    assert artifact.monte_carlo_sample_count > 0
    assert artifact.artifact_checksum


def test_missing_provenance_or_checksum_fails_closed(qualified_artifact_path):
    payload = json.loads(qualified_artifact_path.read_text())
    del payload["qualification"]["simulator_version"]
    qualified_artifact_path.write_text(json.dumps(payload))
    with pytest.raises(LIFPerturbationArtifactError, match="simulator_version"):
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


def test_dynamic_noise_requires_transient_characterization(
    qualified_artifact_path,
):
    payload = json.loads(qualified_artifact_path.read_text())
    payload["dynamic_noise"] = {"std": 0.001}
    qualified_artifact_path.write_text(json.dumps(payload))
    with pytest.raises(LIFPerturbationArtifactError, match="transient"):
        load_lif_perturbation_artifact(qualified_artifact_path)
```

The fixture's `artifact_checksum` is therefore bound to every qualification,
distribution, and operating-domain field.

- [x] **Step 2: Run tests and verify the module is missing**

```bash
$PYTHON -m pytest tests/test_snn_lif_perturbation.py -q
```

Expected: collection fails with `ModuleNotFoundError`.

- [x] **Step 3: Implement strict artifact loading**

Create `MNSIM/Accuracy_Model/snn_lif_perturbation.py` with immutable records for:

```python
@dataclass(frozen=True)
class LIFPerturbationArtifact:
    schema_version: int
    profile_id: str
    pdk: str
    model_files: tuple[dict[str, str], ...]
    simulator_version: str
    process_corner: str
    supply_voltage_v: float
    temperature_c: float
    monte_carlo_sample_count: int
    seed_policy: str
    parameter_order: tuple[str, ...]
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    circuit_source_sha256: str
    supported_timestep_s: float
    supported_input_min: float
    supported_input_max: float
    dynamic_noise_std: float | None
    transient_source_sha256: str | None
    artifact_checksum: str
```

`load_lif_perturbation_artifact(path)` must:

1. parse UTF-8 JSON and require `schema_version == 1` and
   `profile_id == "frontend_lut_lif_lut"`;
2. require nonempty PDK, model-file/checksum list, simulator version, process
   corner, seed policy, and circuit source checksum;
3. require finite voltage/temperature/timestep/range values, positive timestep,
   `input_min < input_max`, and positive Monte Carlo sample count;
4. require the exact parameter order shown in the test;
5. require a finite symmetric 5x5 positive-semidefinite covariance matrix by
   checking `np.linalg.eigvalsh(covariance) >= -1e-12`;
6. accept dynamic noise only when both a nonnegative standard deviation and a
   valid transient-source SHA-256 are present;
7. recompute SHA-256 over the whole canonical JSON body excluding
   `artifact_checksum` and reject a mismatch.

Do not infer missing values, replace covariance with independent standard
deviations, or accept a different circuit's provenance.

- [x] **Step 4: Implement deterministic joint sampling and run-scoped caching**

Implement:

```python
class LIFPerturbationRuntime:
    def __init__(self, artifact: LIFPerturbationArtifact, *, seed: int):
        if seed != 112:
            raise LIFPerturbationArtifactError(
                "formal LIF perturbation seed must be 112"
            )
        self.artifact = artifact
        self.seed = seed
        self._static_cache = {}
        self._dynamic_generators = {}

    def _namespace_seed(self, layer_id: str, purpose: str) -> int:
        digest = canonical_json_sha256(
            {
                "artifact": self.artifact.artifact_checksum,
                "seed": self.seed,
                "layer_id": layer_id,
                "purpose": purpose,
            }
        )
        return int(digest[:16], 16)

    def static_parameters(self, layer_id: str, neuron_shape: tuple[int, ...]):
        key = (layer_id, tuple(neuron_shape))
        if key not in self._static_cache:
            rng = np.random.default_rng(self._namespace_seed(layer_id, "static"))
            count = int(np.prod(neuron_shape))
            sample = rng.multivariate_normal(
                self.artifact.mean,
                self.artifact.covariance,
                size=count,
                check_valid="raise",
            )
            self._static_cache[key] = {
                name: sample[:, index].reshape(neuron_shape)
                for index, name in enumerate(self.artifact.parameter_order)
            }
        return self._static_cache[key]
```

Add `dynamic_noise(layer_id, shape)` that returns zeros when
`dynamic_noise_std is None`; otherwise it uses a separate cached generator in
the `"dynamic"` namespace and advances once per timestep call. The RRAM device
generator is never passed into this object.

- [x] **Step 5: Run artifact tests**

```bash
$PYTHON -m pytest tests/test_snn_lif_perturbation.py -q
```

Expected: all schema, checksum, joint-sampling, caching, and dynamic-noise tests
pass.

- [x] **Step 6: Commit the perturbation artifact and sampler**

```bash
$GIT add \
  MNSIM/Accuracy_Model/snn_lif_perturbation.py \
  tests/test_snn_lif_perturbation.py
$GIT diff --cached --check
$GIT commit -m "feat: validate characterized SNN LIF perturbations"
```

### Task 7: Apply Qualified Perturbations Through The Shared SpikingJelly LIF

**Files:**

- Modify: `MNSIM/Scheduling/non_weighted_layers.py`
- Modify: `MNSIM/Interface/quantize.py`
- Modify: `MNSIM/Interface/network.py`
- Modify: `tests/test_snn_lif_perturbation.py`
- Modify: `tests/test_snn_original_accuracy.py`

- [x] **Step 1: Write failing execution tests**

Add these tests to `tests/test_snn_lif_perturbation.py`:

```python
import torch

from MNSIM.Scheduling.non_weighted_layers import LIFNode


def test_nominal_runtime_matches_the_existing_spikingjelly_lif_exactly():
    sequence = torch.tensor([[[1.0]], [[0.2]], [[0.0]], [[1.0]]])
    baseline = LIFNode(tau=50.5016666556, v_th=0.01, v_reset=0.0)
    candidate = LIFNode(tau=50.5016666556, v_th=0.01, v_reset=0.0)
    candidate.set_accuracy_perturbation(None, layer_id="L001")
    assert torch.equal(
        baseline.forward_sequence(sequence), candidate.forward_sequence(sequence)
    )


def test_static_parameters_persist_across_batch_resets(
    qualified_artifact_path,
):
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(qualified_artifact_path), seed=112
    )
    lif = LIFNode(tau=50.5016666556, v_th=0.01, v_reset=0.0)
    lif.set_accuracy_perturbation(runtime, layer_id="L001")
    sequence = torch.ones(3, 2, 4)
    first = lif.forward_sequence(sequence)
    parameters_before = lif.sampled_accuracy_parameters()
    lif.reset_state()
    second = lif.forward_sequence(sequence)
    parameters_after = lif.sampled_accuracy_parameters()
    assert parameters_before is parameters_after
    assert torch.equal(first, second)


def test_frontend_range_violation_fails_without_clamping(
    qualified_artifact_path,
):
    runtime = LIFPerturbationRuntime(
        load_lif_perturbation_artifact(qualified_artifact_path), seed=112
    )
    lif = LIFNode(tau=50.5016666556, v_th=0.01, v_reset=0.0)
    lif.set_accuracy_perturbation(runtime, layer_id="L001")
    with pytest.raises(LIFPerturbationArtifactError, match="operating range"):
        lif.forward_sequence(torch.full((2, 1, 4), 1.0e9))
```

The final module-import and physical-current isolation test is added in Task 9,
after all profile wiring exists. This task's execution tests are limited to the
SpikingJelly wrapper and perturbation runtime defined here.

- [x] **Step 2: Run tests and verify `LIFNode` cannot install a perturbation runtime**

```bash
$PYTHON -m pytest \
  tests/test_snn_lif_perturbation.py \
  tests/test_snn_original_accuracy.py -q
```

Expected: new tests fail because `set_accuracy_perturbation()` and tensor-valued
LIF parameters are not implemented.

- [x] **Step 3: Keep one SpikingJelly class and add a tensor-parameter execution path**

In `MNSIM/Scheduling/non_weighted_layers.py`, introduce a private subclass of
`spikingjelly.activation_based.neuron.LIFNode`:

```python
class _AccuracySpikingJellyLIF(neuron.LIFNode):
    def install_tensor_parameters(self, *, tau, v_threshold, v_reset):
        self._accuracy_tau = tau
        self._accuracy_v_threshold = v_threshold
        self._accuracy_v_reset = v_reset

    def clear_tensor_parameters(self):
        self._accuracy_tau = None
        self._accuracy_v_threshold = None
        self._accuracy_v_reset = None

    def multi_step_forward(self, x_seq):
        if self._accuracy_tau is None:
            return super().multi_step_forward(x_seq)
        return neuron.BaseNode.multi_step_forward(self, x_seq)

    def neuronal_charge(self, x):
        if self._accuracy_tau is None:
            return super().neuronal_charge(x)
        self.v = self.v + (
            x - (self.v - self._accuracy_v_reset)
        ) / self._accuracy_tau

    def neuronal_fire(self):
        if self._accuracy_v_threshold is None:
            return super().neuronal_fire()
        return (self.v >= self._accuracy_v_threshold).to(self.v)

    def neuronal_reset(self, spike):
        if self._accuracy_v_reset is None:
            return super().neuronal_reset(spike)
        self.v = self._accuracy_v_reset * spike + (1.0 - spike) * self.v
```

Instantiate `_AccuracySpikingJellyLIF` for both nominal and perturbed operation,
with `step_mode="s"`, `backend="torch"`, `decay_input=True`, and hard reset.
Thus Original and PIM still use the same wrapper/class; the nominal path
delegates to the installed SpikingJelly implementation exactly.

- [x] **Step 4: Apply static frontend/LIF parameters with physical-neuron broadcast semantics**

Add to the public `LIFNode` wrapper:

```python
# Initialize these fields in LIFNode.__init__ after creating self.node.
self._accuracy_perturbation = None
self._accuracy_layer_id = None
self._sampled_accuracy_parameters = None


def set_accuracy_perturbation(self, runtime, *, layer_id):
    runtime_changed = runtime is not self._accuracy_perturbation
    self._accuracy_perturbation = runtime
    self._accuracy_layer_id = layer_id
    if runtime_changed:
        self._sampled_accuracy_parameters = None
        self.node.clear_tensor_parameters()

def _prepare_accuracy_parameters(self, x_seq):
    runtime = self._accuracy_perturbation
    if runtime is None:
        return x_seq
    neuron_shape = (1, *x_seq.shape[2:])
    arrays = runtime.static_parameters(self._accuracy_layer_id, neuron_shape)
    if self._sampled_accuracy_parameters is None:
        self._sampled_accuracy_parameters = {
            key: torch.as_tensor(value, device=x_seq.device, dtype=x_seq.dtype)
            for key, value in arrays.items()
        }
    p = self._sampled_accuracy_parameters
    tau = self.tau * (1.0 + p["delta_tau"])
    v_threshold = self.v_th + p["delta_v_th"]
    v_reset = self.v_reset + p["delta_v_reset"]
    if torch.any(tau <= 0):
        raise LIFPerturbationArtifactError("perturbed tau must remain positive")
    self.node.install_tensor_parameters(
        tau=tau, v_threshold=v_threshold, v_reset=v_reset
    )
    transformed = p["frontend_gain"] * x_seq + p["frontend_offset"]
    for timestep in range(transformed.shape[0]):
        transformed[timestep] += torch.as_tensor(
            runtime.dynamic_noise(
                self._accuracy_layer_id, tuple(transformed[timestep].shape)
            ),
            device=x_seq.device,
            dtype=x_seq.dtype,
        )
    runtime.validate_operating_range(transformed)
    return transformed
```

`forward_sequence()` calls `_prepare_accuracy_parameters()` and then the same
`self.node` multi-step entry. `reset_state()` resets membrane state only; it does
not clear `_sampled_accuracy_parameters` or the runtime cache. Add
`sampled_accuracy_parameters()` for tests/report audit, returning the cached
mapping by identity.

Set stable layer IDs in `NetworkGraph.__init__` using the graph index:

```python
for layer_index, (layer_config, quantize_config) in enumerate(
    zip(layer_config_list, quantize_config_list)
):
    if layer_config["type"] in quantize.QuantizeLayerStr:
        layer = quantize.QuantizeLayer(
            hardware_config, layer_config, quantize_config
        )
    elif layer_config["type"] in quantize.StraightLayerStr:
        layer = quantize.StraightLayer(
            hardware_config, layer_config, quantize_config
        )
    else:
        raise ValueError(f"unsupported layer type {layer_config['type']}")
    layer.accuracy_layer_id = f"L{layer_index:03d}"
    self.layer_list.append(layer)
```

Pass that ID when `StraightLayer.set_accuracy_perturbation()` delegates to the
wrapped LIF.

- [x] **Step 5: Run nominal and characterized LIF tests**

```bash
$PYTHON -m pytest \
  tests/test_snn_lif_perturbation.py \
  tests/test_snn_original_accuracy.py \
  tests/test_snn_unified_accuracy.py -q
```

Expected: all tests pass. Nominal output remains bit-exact, static mismatch
survives batch resets, dynamic noise advances only when characterized, and
range violations fail instead of clamping.

- [x] **Step 6: Commit characterized SpikingJelly execution**

```bash
$GIT add \
  MNSIM/Scheduling/non_weighted_layers.py \
  MNSIM/Interface/quantize.py \
  MNSIM/Interface/network.py \
  tests/test_snn_lif_perturbation.py \
  tests/test_snn_original_accuracy.py \
  tests/test_snn_unified_accuracy.py
$GIT diff --cached --check
$GIT commit -m "feat: apply qualified perturbations in SNN accuracy"
```

### Task 8: Wire Profiles, Artifacts, Evaluation, CLI, And Reports

**Files:**

- Modify: `MNSIM/Interface/interface.py`
- Modify: `MNSIM/Interface/round_orchestrated_runner.py`
- Modify: `MNSIM/Interface/round_orchestrated_cli.py`
- Modify: `SimConfig.ini`
- Modify: `tests/test_round_orchestrated_interface.py`
- Modify: `tests/test_round_orchestrated_runner.py`
- Modify: `tests/test_round_orchestrated_cli.py`

- [x] **Step 1: Write failing interface and runner tests for profile-specific call graphs**

In `tests/test_round_orchestrated_interface.py`, extend `_construct()` with a
`network_module="fake_network"` keyword and pass that value to
`TrainTestInterface`. Add this helper and these tests:

```python
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
```

In `tests/test_round_orchestrated_runner.py`, import `SimpleNamespace` and add
these concrete fakes/helpers:

```python
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
            model_id="rram_20ns_linear_fit",
            variation_percent=6.69903802061613,
            device_seed=112,
        ),
    )
    monkeypatch.setattr(
        runner_module,
        "LIFPerturbationRuntime",
        lambda artifact, seed: ("lif-runtime", artifact, seed),
    )


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
```

Leave the existing `FakeContext` ANN method signatures and legacy assertions
unchanged; only `_ProfiledSNNContext` consumes the new SNN keywords.

- [x] **Step 2: Run focused tests and verify the profile is not yet threaded through the public runner**

```bash
$PYTHON -m pytest \
  tests/test_round_orchestrated_interface.py \
  tests/test_round_orchestrated_runner.py \
  tests/test_round_orchestrated_cli.py -q
```

Expected: new tests fail because the interface does not expose
`snn_accuracy_config`, the runner does not build artifact identities, and the
report lacks the canonical metadata.

- [x] **Step 3: Parse the profile once in `TrainTestInterface` and pass it through evaluation**

After reading the SimConfig in `TrainTestInterface.__init__`:

```python
self.snn_accuracy_config = None
if "snn" in self.network_module.lower():
    self.snn_accuracy_config = parse_snn_accuracy_config(
        xbar_config, source=SimConfig_path
    )
    self.lif_perturbation_artifact = None
    if self.snn_accuracy_config.hardware_profile == FRONTEND_LUT_LIF_LUT:
        self.lif_perturbation_artifact = load_lif_perturbation_artifact(
            self.snn_accuracy_config.lif_perturbation_artifact
        )
```

Change `origin_evaluate()` and `set_net_bits_evaluate()` to accept optional
`pim_profile` and `lif_perturbation` keyword arguments. Original ignores both;
PIM forwards them to `_accuracy_forward()`, which passes them to
`NetworkGraph.sequence_accuracy_forward()` only for 5-D SNN input.

Add:

```python
def calibration_split_identity(self):
    if self.calibration_loader is None:
        self.calibration_loader = import_module(
            self.dataset_module
        ).get_dataloader()[0]
    return self.calibration_loader.dataset.accuracy_split_identity()
```

- [x] **Step 4: Build one complete runtime identity before calibration/test**

In the runner's `adc_digital_lif` branch, create `ReADCArtifactIdentity` only
after `updated_bits` exists. The LUT branch must not request a calibration split
identity:

```python
checkpoint_sha256 = file_sha256(request.weights)
deployment_artifact_checksum = bit_weights_sha256(net_bits)
weight_mapping_sha256 = bit_weights_sha256(updated_bits)
identity = ReADCArtifactIdentity(
    checkpoint_sha256=checkpoint_sha256,
    dataset_split_identity=_quiet_call(context.calibration_split_identity),
    device_model_id=device_contract.model_id,
    device_seed=device_contract.device_seed,
    weight_mapping_sha256=weight_mapping_sha256,
    adc_precision=5,
    percentile_low=0.5,
    percentile_high=99.5,
    max_samples_per_layer=8000,
)
```

For `adc_digital_lif`:

```python
readc_artifact = _quiet_call(
    context.calibrate_readc,
    updated_bits,
    identity=identity,
    max_samples=request.max_samples,
)
store.write_json("readc_calibration.json", readc_artifact.to_dict())
_quiet_call(
    context.install_readc_artifact,
    readc_artifact,
    expected_identity=identity,
)
pim_accuracy = _quiet_call(
    context.set_net_bits_evaluate,
    updated_bits,
    adc_action="ADAPTIVE",
    pim_profile=ADC_DIGITAL_LIF,
    lif_perturbation=None,
    **sample_limit,
)
```

For `frontend_lut_lif_lut`:

```python
artifact = context.lif_perturbation_artifact
runtime = LIFPerturbationRuntime(
    artifact, seed=context.snn_accuracy_config.lif_perturbation_seed
)
pim_accuracy = _quiet_call(
    context.set_net_bits_evaluate,
    updated_bits,
    adc_action=None,
    pim_profile=FRONTEND_LUT_LIF_LUT,
    lif_perturbation=runtime,
    **sample_limit,
)
```

Do not call calibration, collectors, `_adc_quantize()`, ReADC installation, or
output activation quantization in the LUT branch. Do not construct a
`LIFPerturbationRuntime` in the ADC branch.

- [x] **Step 5: Emit the exact public report contract**

For SNN reports, keep only the two public accuracy labels and add metadata:

```python
report.update(
    {
        "accuracy_compute_model": "mnsim_bit_sliced_weighted_ops",
        "accuracy_hardware_coupling": "profile_selected_numerical_only",
        "accuracy_artifact_checksum": canonical_json_sha256(
            {
                "original_accuracy": original_accuracy,
                "pim_accuracy": pim_accuracy,
                "checkpoint_sha256": checkpoint_sha256,
                "weight_mapping_sha256": weight_mapping_sha256,
                "hardware_profile": profile,
                "readc_artifact_checksum": (
                    readc_artifact.artifact_checksum if readc_artifact else None
                ),
                "lif_perturbation_checksum": (
                    lif_artifact.artifact_checksum if lif_artifact else None
                ),
            }
        ),
        "accuracy_adc_model": "READC_LLOYD_MAX" if readc_artifact else None,
        "accuracy_lif_model": (
            "spikingjelly.activation_based.neuron.LIFNode"
        ),
        "accuracy_lif_perturbation_profile": (
            lif_artifact.profile_id if lif_artifact else None
        ),
        "accuracy_lif_perturbation_checksum": (
            lif_artifact.artifact_checksum if lif_artifact else None
        ),
        "accuracy_lif_perturbation_seed": (
            112 if lif_artifact else None
        ),
        "deployment_artifact_checksum": deployment_artifact_checksum,
        "deployment_device_code_dtype": "mnsim_differential_code_arrays",
        "deployment_device_code_range": [0, 255],
        "accuracy_post_variation_mapping_checksum": weight_mapping_sha256,
        "binary_spike_domain": {
            "values": [0, 1],
            "scale": 1,
            "signed": False,
            "input_digit_max": 1,
            "activation_cycle": 1,
            "rounding": "forbidden",
            "clamping": "forbidden",
        },
        "hardware_profile": profile,
    }
)
```

Nest Hong 2025 provenance, calibration source, calibration artifact checksum,
and per-layer checksums under `readc_calibration`. Do not report ADC/LUT-specific
accuracy names.

- [x] **Step 6: Render profile provenance without creating a second configuration authority**

Do not add CLI flags that select the hardware profile or override its artifact
paths. The SimConfig is the only authority. In `render_terminal_result()`, append
these lines for a successful SNN accuracy report:

```python
lines.extend(
    [
        "Accuracy Model",
        "  Hardware profile: "
        + _format_value(result.accuracy_report.get("hardware_profile")),
        "  ADC numerical model: "
        + _format_value(result.accuracy_report.get("accuracy_adc_model")),
        "  LIF perturbation profile: "
        + _format_value(
            result.accuracy_report.get("accuracy_lif_perturbation_profile")
        ),
    ]
)
```

Add to `tests/test_round_orchestrated_cli.py`:

```python
def test_snn_renderer_reports_the_simconfig_selected_accuracy_profile():
    request = _request(network="standard_vgg8_snn_v7_phase6")
    result = RoundOrchestratedRunResult(
        status="succeeded",
        exit_code=0,
        request=request,
        stage_statuses={
            "accuracy": StageStatus(name="accuracy", status="succeeded")
        },
        accuracy_report={
            "SNN Original Accuracy": 0.75,
            "SNN Processing-in-Memory Accuracy": 0.5,
            "hardware_profile": "adc_digital_lif",
            "accuracy_adc_model": "READC_LLOYD_MAX",
            "accuracy_lif_perturbation_profile": None,
        },
    )

    rendered = cli.render_terminal_result(result)

    assert "Hardware profile: adc_digital_lif" in rendered
    assert "ADC numerical model: READC_LLOYD_MAX" in rendered
    assert "LIF perturbation profile: n/a" in rendered


@pytest.mark.parametrize(
    "option",
    [
        "--snn-hardware-profile",
        "--readc-artifact",
        "--lif-perturbation-artifact",
    ],
)
def test_cli_rejects_profile_and_artifact_overrides(option):
    with pytest.raises(SystemExit) as error:
        cli.request_from_args(_required_args() + [option, "forbidden"])
    assert error.value.code == 2
```

This prevents profile/artifact drift between the request and SimConfig.

- [x] **Step 7: Update the default formal SimConfig**

Add:

```ini
[SNN Accuracy]
SNN_Hardware_Profile = adc_digital_lif
Hardware_Current_Model = physical_conductance
LIF_Perturbation_Artifact =
LIF_Perturbation_Seed = 112
```

Ensure `[Interface level]` contains exactly:

```ini
ADC_Precision = 5
ADC_Profile = HONG_2025_READC_5BIT
ADC_Quantization_Mode = READC_LLOYD_MAX
ADC_ReADC_Percentile_Low = 0.5
ADC_ReADC_Percentile_High = 99.5
ADC_ReADC_Max_Samples_Per_Layer = 8000
```

Do not add a zero-perturbation LUT artifact to the repository. Formal LUT PIM
accuracy remains intentionally unavailable until a qualified Monte Carlo/PVT
artifact exists.

- [x] **Step 8: Run interface/runner/CLI tests**

```bash
$PYTHON -m pytest \
  tests/test_round_orchestrated_interface.py \
  tests/test_round_orchestrated_runner.py \
  tests/test_round_orchestrated_cli.py -q
```

Expected: all tests pass. The ADC branch calibrates or validates ReADC before
test inference; the LUT branch never touches ReADC and fails closed without a
qualified perturbation artifact.

- [x] **Step 9: Commit the public wiring**

```bash
$GIT add \
  MNSIM/Interface/interface.py \
  MNSIM/Interface/round_orchestrated_runner.py \
  MNSIM/Interface/round_orchestrated_cli.py \
  SimConfig.ini \
  tests/test_round_orchestrated_interface.py \
  tests/test_round_orchestrated_runner.py \
  tests/test_round_orchestrated_cli.py
$GIT diff --cached --check
$GIT commit -m "feat: wire formal SNN PIM accuracy profiles"
```

### Task 9: Prove Isolation And Run The Focused Regression Suite

**Files:**

- Modify: `MNSIM/Interface/dvs_binned_npz.py`
- Modify: `tests/test_snn_original_accuracy.py`
- Modify: `tests/test_peripheral_literature_profile.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-15-master-snn-functionality-integration-design.md`
- Include: `docs/superpowers/plans/2026-08-15-snn-pim-accuracy-implementation.md`

- [x] **Step 1: Add final isolation tests**

Add to `tests/test_snn_original_accuracy.py`:

```python
import MNSIM.Interface.snn_mvm_current as current_backend


def test_original_and_lut_pim_accuracy_never_call_physical_current(
    monkeypatch,
):
    def forbidden_physical_current(*_args, **_kwargs):
        raise AssertionError("physical current must not enter accuracy")

    monkeypatch.setattr(
        current_backend, "mvm_current_from_bit_weights", forbidden_physical_current
    )
    monkeypatch.setattr(
        current_backend,
        "mvm_column_load_from_bit_weights",
        forbidden_physical_current,
    )
    net = _tiny_original_network()
    weighted = net.layer_list[0]
    with torch.no_grad():
        weighted.sublayer_list[0].weight.fill_(1.0)
        weighted.bit_scale_list[1, 1] = 1.0
        weighted.last_value.fill_(1.0)
    net.eval()
    sequence = torch.tensor([[[[[1.0]]], [[[0.0]]]]])

    original = net.sequence_accuracy_forward(sequence, weighted_mode="ORIGINAL")
    pim = net.sequence_accuracy_forward(
        sequence,
        weighted_mode="PIM",
        adc_action=None,
        net_bit_weights=net.get_weights(),
        pim_profile="frontend_lut_lif_lut",
        lif_perturbation=None,
    )

    assert set(original.unique().tolist()) <= {0.0, 1.0, 2.0}
    assert set(pim.unique().tolist()) <= {0.0, 1.0, 2.0}


def test_accuracy_modules_have_no_physical_current_or_hardware_lut_imports():
    repo_root = Path(__file__).resolve().parents[1]
    for relative_path in (
        "MNSIM/Interface/network.py",
        "MNSIM/Interface/quantize.py",
        "MNSIM/Scheduling/non_weighted_layers.py",
    ):
        source = (repo_root / relative_path).read_text(encoding="utf-8")
        assert "snn_mvm_current" not in source
        assert "mvm_current_from_bit_weights" not in source
        assert "frontend_lut_lookup" not in source
        assert "lif_lut_lookup" not in source
```

The earlier task tests remain the executable proof for profile-independent
Original accuracy, ReADC-only ADC PIM, no-ADC LUT PIM, qualified perturbation
reproducibility, and `BINARY_SPIKE_DOMAIN`. Hardware-performance RNG isolation
is intentionally left to the separate physical-current/hardware plan because
this accuracy-only subsystem neither constructs nor receives that RNG.

Keep the calibration dataset identity logical split as `train` even when the
on-disk compatibility alias resolves to `pretrain`, and record that physical
directory separately as `resolved_split`. Add the matching regression to
`tests/test_snn_original_accuracy.py`. The default formal `SimConfig.ini` also
adds an SNN-only overlay to the G8 literature profile, so include the exact
overlay regression in `tests/test_peripheral_literature_profile.py`.

Task 9 uses prerequisite commit `7f3b7ee` (`fix: lock formal configs to final
RRAM model`) to lock `SimConfig.ini`,
`configs/hardware_profiles/g8_literature_peripheral_ann.ini`, and
`configs/hardware_profiles/g8_600plus_ann.ini` to the final 20 ns RRAM model.
The prerequisite contains exactly those three configuration files. Its staged
tree passed the final-RRAM contract; the legacy exact-profile assertion failed
only because it did not yet recognize the Task 9 SNN-only overlay. Applying the
approved Task 9 overlay regression to that snapshot produced `25 passed` for
`tests/test_final_rram_contract.py` and
`tests/test_peripheral_literature_profile.py`.

- [x] **Step 2: Run the complete focused accuracy suite**

```bash
$PYTHON -m pytest \
  tests/test_snn_accuracy_contract.py \
  tests/test_snn_original_accuracy.py \
  tests/test_snn_unified_accuracy.py \
  tests/test_snn_no_adc_reconstruction.py \
  tests/test_snn_lif_perturbation.py \
  tests/test_readc.py \
  tests/test_readc_artifact.py \
  tests/test_adc_adaptive_quantization.py \
  tests/test_no_post_adc_requantization.py \
  tests/test_device_nonidealities.py \
  tests/test_final_rram_contract.py \
  tests/test_pim_numerical_audit.py \
  tests/test_round_orchestrated_interface.py \
  tests/test_round_orchestrated_runner.py \
  tests/test_round_orchestrated_cli.py \
  tests/test_round_snn_integration.py \
  tests/test_peripheral_literature_profile.py -q
```

Expected: all focused tests pass with no skipped test except tests explicitly
guarded by unavailable external V7 checkpoints/datasets.

Exact Task 9 staged-tree result on 2026-08-16: `244 passed`. Pytest emitted one
pre-existing `SyntaxWarning` for the regular-expression string in
`MNSIM/Interface/network.py`; there were no test failures or skips.

- [x] **Step 3: Run the broader non-hardware regression suite**

```bash
$PYTHON -m pytest tests -q \
  --ignore=tests/test_round_phase2_real_backend.py \
  --ignore=tests/test_round_orchestrated_g1_g11_sweep.py
```

Expected: all collected tests pass. The ignored tests are long hardware/sweep
jobs outside this accuracy-only plan and are covered by their own execution
workflow.

- [x] **Step 4: Run one wrapper-based ADC-profile smoke evaluation**

Use the supported runpy shell wrapper with the final V7 checkpoint and exported
V7 dataset:

```bash
MNSIM_PYTHON=/home/xzh/micromamba/envs/workspace/bin/python \
MNSIM_DVS_BINNED_ROOT=/home/xzh/snn_binned_4split_2000perclass_v7_fullres_t30_phase6 \
MNSIM_DVS_TRAIN_BATCH=1 \
MNSIM_DVS_TEST_BATCH=1 \
bash scripts/run_round_orchestrated_sim.sh \
  --network standard_vgg8_snn_v7_phase6 \
  --weights /home/xzh/snn_software_pretrain/standard_vgg8_snn_v7_software_pretrain_best.pth \
  --num-classes 8 \
  --sim-config SimConfig.ini \
  --output-dir results/snn_pim_adc_smoke_20260815 \
  --dataset-module MNSIM.Interface.dvs_binned_npz \
  --max-samples 1 \
  --overwrite \
  --device 0
```

The resulting manifest must include these resolved inputs:

```text
network = standard_vgg8_snn_v7_phase6
hardware_profile = adc_digital_lif
device_model = rram_20ns_linear_fit
max_samples = 1
accuracy_device_seed = 112
accuracy_adc_model = READC_LLOYD_MAX
```

Expected artifacts:

```text
run_manifest.json
summary.json
accuracy_report.json
readc_calibration.json
```

Expected: exit code 0, both public SNN accuracy labels present, ReADC artifact
checksum present, and no physical-current or LUT-runtime object recorded as an
accuracy input.

- [x] **Step 5: Document the implemented boundary and the intentional LUT block**

Update `README.md` with the supported invocation and these facts:

```text
- SNN Original Accuracy is unquantized PyTorch + shared SpikingJelly LIF.
- SNN Processing-in-Memory Accuracy has one public name.
- adc_digital_lif uses fixed 5-bit Hong 2025 ReADC calibration.
- frontend_lut_lif_lut uses exact no-ADC reconstruction plus a qualified
  frontend/LIF characterization artifact.
- Formal LUT PIM accuracy fails closed until such an artifact is supplied.
- Physical current and hardware LUT values never feed accuracy tensors/state.
```

In the canonical spec, append an implementation-status note that links this
plan and lists the exact test/smoke commands actually run. Do not change any
frozen design decision.

- [x] **Step 6: Review the final diff for forbidden coupling and public names**

Run:

```bash
rg -n "Ipos|Ineg|Idiff|snn_mvm_current|frontend_lut|lif_lut" \
  MNSIM/Accuracy_Model MNSIM/Interface/network.py MNSIM/Interface/quantize.py
rg -n "SNN .*Accuracy" MNSIM tests README.md
$GIT diff --check
$GIT status --short
```

Expected:

- no physical-current/LUT hardware call appears in an accuracy forward;
- the only public SNN result labels are `SNN Original Accuracy` and
  `SNN Processing-in-Memory Accuracy`;
- no whitespace errors;
- unrelated pre-existing worktree changes remain unstaged.

- [x] **Step 7: Commit tests and documentation**

```bash
$GIT add \
  MNSIM/Interface/dvs_binned_npz.py \
  tests/test_snn_original_accuracy.py \
  tests/test_peripheral_literature_profile.py \
  README.md \
  docs/superpowers/specs/2026-08-15-master-snn-functionality-integration-design.md \
  docs/superpowers/plans/2026-08-15-snn-pim-accuracy-implementation.md
$GIT diff --cached --check
$GIT commit -m "test: verify SNN PIM accuracy isolation"
```

## Completion Criteria

The plan is complete only when all of the following are true:

1. SNN Original and PIM use one sequence executor and the same SpikingJelly LIF
   wrapper with reset between independent batches and retained state across
   timesteps.
2. Inputs and LIF outputs use the explicit unsigned one-cycle
   `BinarySpikeDomain`; no signed one-bit activation state, rounding, or clamp is
   used.
3. Formal PIM variation uses one post-R-ratio/post-variation mapping per run,
   fixed `Device_Seed=112`, the device-specific sigma, and no SAF.
4. ADC PIM uses only 5-bit Hong 2025 ReADC, with a training-split, identity-bound,
   checksummed calibration artifact installed before test inference.
5. LUT PIM uses exact bit-sliced no-ADC reconstruction and never invokes ADC,
   ReADC, or the output activation quantizer.
6. Formal LUT PIM accuracy fails closed without a qualified frontend/LIF
   Monte Carlo/PVT artifact; test-only nominal execution is never reported as a
   formal LUT result.
7. Physical current and hardware frontend/LIF LUT values cannot enter accuracy
   tensors, spikes, membrane state, or RNG streams.
8. Reports expose only `SNN Original Accuracy` and
   `SNN Processing-in-Memory Accuracy`, plus the frozen provenance/identity
   fields.
9. Focused and broader non-hardware tests pass, and the wrapper-based one-sample
   ADC smoke produces a valid ReADC artifact and accuracy report.
