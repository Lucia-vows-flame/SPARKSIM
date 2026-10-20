# Round-Orchestrated Accuracy Compatibility Repair Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore legacy MNSIM ANN/ResNet18 accuracy compatibility and make the formal SNN accuracy path use only the pretrain checkpoint, pretrain data, and test data while preserving explicit CUDA fail-fast behavior.

**Architecture:** Keep the original MNSIM execution methods intact. Add a narrow checkpoint-schema compatibility rule that recognizes a valid residual-add range already stored in legacy checkpoints; do not silently calibrate or modify weights. Make the SNN dataset contract reject `finetune` as an accuracy source, then validate the pretrain checkpoint and test split at the runner boundary. Treat CUDA launch stabilization and accuracy validation as separate gates.

**Tech Stack:** Python 3.12, PyTorch, pytest, `runpy` launcher, existing MNSIM `NetworkGraph` and `TrainTestInterface` APIs.

**Spec:** The governing specification is the current accuracy/checkpoint contract in the source tree plus the user-confirmed pretrain-only SNN policy; no new analog-LIF or QAT specification is introduced by this repair.

## Global Constraints

- Work only in the repository root on the `task-20260707-round-orchestrated-mnsim` architecture.
- Preserve all existing user modifications in the worktree.
- Do not run `run_resnet18_post_add_qat.py`; QAT is out of scope and must not alter the source checkpoint.
- Pure calibration, if retained for an explicitly requested PTQ audit, must freeze weights and BatchNorm state; it is not part of the default accuracy runner.
- ANN ResNet18 uses the archived legacy checkpoint when available and the original MNSIM `FIX_TRAIN`/PIM flow.
- SNN accuracy uses the software-pretrain checkpoint, logical `train` mapped to physical `pretrain`, and physical `test`; `finetune` is never an accuracy source.
- Explicit GPU requests must fail fast when CUDA is unavailable; no CPU fallback is allowed.
- Run Python through `/home/xzh/micromamba/envs/workspace/bin/python` and use the existing `runpy` entry path for repository scripts.

### Task 1: Freeze the Current Failure Contract

**Files:**
- Create: `tests/test_accuracy_repair_baseline.py`
- Modify: `tests/test_round_phase1_real_networks.py`

**Interfaces:**
- Consumes: existing `NetworkGraph`, `TrainTestInterface`, dataset loader, and real checkpoint paths.
- Produces: deterministic tests that identify the legacy ResNet failure, the stale SNN checkpoint reference, and the allowed pretrain/test paths.

- [ ] **Step 1: Add a synthetic legacy residual checkpoint test.** Build the existing two-FC-plus-`element_sum` graph, remove `post_add_quantizer_version` from its state dict, retain a positive `last_value`, and assert that loading the state is the exact compatibility case to support.
- [ ] **Step 2: Add an unknown-range test.** Remove both the marker and the residual-add range, then assert that evaluation fails with an explicit “range unavailable” error requiring calibration. This prevents the compatibility rule from silently inventing a scale.
- [ ] **Step 3: Replace the hard-coded missing standard VGG8 finetune path in `test_round_phase1_real_networks.py`.** Resolve the checkpoint from `STANDARD_VGG8_SNN_CHECKPOINT`, defaulting to `/home/xzh/snn_pretrain_and_v7_dataset_20260816/snn_software_pretrain/standard_vgg8_snn_v7_software_pretrain_best.pth`; skip with a clear reason when the external artifact is absent.
- [ ] **Step 4: Run the focused baseline tests.**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest \
  tests/test_residual_activation_quantization.py \
  tests/test_round_phase1_real_networks.py \
  tests/test_snn_original_accuracy.py \
  tests/test_snn_accuracy_contract.py -q
```

Expected before implementation: the known external-checkpoint failure remains, and the synthetic legacy test documents the current marker behavior.

### Task 2: Restore Legacy ResNet18 Checkpoint Compatibility

**Files:**
- Modify: `MNSIM/Interface/quantize.py:1045-1075`
- Modify: `tests/test_residual_activation_quantization.py`
- Create: `tests/test_resnet18_legacy_checkpoint_compatibility.py`

**Interfaces:**
- Consumes: `StraightLayer._load_from_state_dict`, `last_value`, and `post_add_quantizer_version`.
- Produces: a loaded legacy checkpoint whose valid stored residual ranges are accepted by the existing `FIX_TRAIN` and PIM paths without calibration or weight updates.

- [ ] **Step 1: Define the compatibility predicate.** For an `element_sum` layer with a missing marker, inspect the checkpoint value at `prefix + "last_value"`. Accept only a scalar, finite, strictly positive value; set the in-memory marker to version `1`. If the range is absent, non-finite, or non-positive, set version `0` and retain the explicit calibration-required failure.
- [ ] **Step 2: Keep the caller state isolated.** Copy or locally augment the state entry before calling `super()._load_from_state_dict`; do not mutate a caller-owned checkpoint mapping in a way that changes later loads.
- [ ] **Step 3: Do not change the original method selection.** Keep ANN Original accuracy on the existing runner `FIX_TRAIN` path because this is the MNSIM fixed-range inference path. Keep SNN Original accuracy on `TRADITION`. The fix belongs at checkpoint compatibility, not by changing ResNet inference semantics.
- [ ] **Step 4: Update the old unit expectation.** A legacy state with a valid stored residual range must run successfully; a state without a usable range must still fail until an explicit frozen calibration pass is requested.
- [ ] **Step 5: Add a real checkpoint smoke test.** Load the archived legacy ResNet18 checkpoint when present, assert every residual-add layer has a positive range and inferred marker `1`, then execute one deterministic input through `origin_evaluate`/equivalent `FIX_TRAIN` and one PIM single-sample path. The test must not call the archived QAT script or write a checkpoint.
- [ ] **Step 6: Run only the compatibility tests.**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest \
  tests/test_residual_activation_quantization.py \
  tests/test_resnet18_legacy_checkpoint_compatibility.py -q
```

### Task 3: Enforce Pretrain-Only SNN Accuracy Data and Model Provenance

**Files:**
- Modify: `MNSIM/Interface/dvs_binned_npz.py:80-110`
- Modify: `MNSIM/Interface/round_orchestrated_runner.py:760-930`
- Modify: `scripts/run_standard_vgg8_snn_g8_seq_cpu.sh`
- Modify: `scripts/run_standard_vgg8_snn_g8_seq_gpu.sh`
- Modify: `tests/test_snn_original_accuracy.py`
- Modify: `tests/test_snn_accuracy_contract.py`
- Modify: `tests/test_round_orchestrated_runner.py`

**Interfaces:**
- Consumes: `RoundOrchestratedRunRequest`, `get_dataloader`, `get_calibration_dataloader`, and the existing formal ADC/PIM profiles.
- Produces: an accuracy report whose provenance identifies logical `train`/physical `pretrain`, physical `test`, and the software-pretrain checkpoint; a finetune-only root fails before inference.

- [ ] **Step 1: Make the formal mapping exact.** Logical `train` must resolve to physical `pretrain` for the formal V7 accuracy path; explicit logical `pretrain` also resolves to physical `pretrain`. Do not fall back to a physical `train` or `finetune` directory for formal accuracy. Preserve the artifact-facing logical identity value `"train"` and expose `resolved_split="pretrain"`.
- [ ] **Step 2: Add a regression test for a finetune-only root.** Create a temporary root containing only `finetune/`; `get_calibration_dataloader` and `get_dataloader` must raise `FileNotFoundError`, proving that a finetune directory cannot silently become the training/calibration source.
- [ ] **Step 2a: Add a regression test for a train-only root.** Create a temporary root containing only `train/`; the formal calibration loader must reject it, so a generic legacy directory cannot masquerade as the required V7 `pretrain/` source.
- [ ] **Step 3: Add a regression test for the complete V7 root contract.** Assert the calibration identity is logical `train` plus physical `pretrain`, the evaluation identity is `test`, and the two identities differ.
- [ ] **Step 4: Make the runner record split provenance.** When an SNN accuracy run performs ReADC calibration, record the calibration identity returned by `calibration_split_identity()`. Always record the test identity after the first test loader is materialized. Reject a resolved calibration split named `finetune` and reject an empty or missing `test` split before evaluating accuracy.
- [ ] **Step 5: Make the standard SNN wrappers default to the software-pretrain checkpoint.** Replace the stale `/home/xzh/wsl_standard_vgg8_snn_finetune_20260812/...` default with `/home/xzh/snn_pretrain_and_v7_dataset_20260816/snn_software_pretrain/standard_vgg8_snn_v7_software_pretrain_best.pth`; keep an environment override for a deliberate alternate artifact.
- [ ] **Step 6: Add an end-to-end one-sample SNN accuracy test.** Use `MNSIM_DVS_BINNED_ROOT=/home/xzh/workspace/cim_project/snn_binned_4split_2000perclass_v7_fullres_t30_phase6`, `MNSIM_DVS_TRAIN_SPLIT=train`, `MNSIM_DVS_TEST_SPLIT=test`, the software-pretrain checkpoint, `max_samples=1`, and `adc_digital_lif`. Assert both original and PIM paths use the expected provenance and that no finetune path appears in the report.
- [ ] **Step 7: Run SNN contract and runner tests.**

Run:

```bash
MNSIM_DVS_BINNED_ROOT=/home/xzh/workspace/cim_project/snn_binned_4split_2000perclass_v7_fullres_t30_phase6 \
MNSIM_DVS_TRAIN_SPLIT=train \
MNSIM_DVS_TEST_SPLIT=test \
/home/xzh/micromamba/envs/workspace/bin/python -m pytest \
  tests/test_snn_original_accuracy.py \
  tests/test_snn_accuracy_contract.py \
  tests/test_round_orchestrated_runner.py -q
```

### Task 4: Stabilize the Explicit CUDA Launch Gate

**Files:**
- Modify: `scripts/run_round_orchestrated_sim.sh`
- Modify: `scripts/run_standard_vgg8_snn_g8_seq_gpu.sh`
- Modify: `MNSIM/Interface/torch_device_resolver.py` only if the integration test exposes a missing probe boundary
- Modify: `tests/test_torch_device_resolver.py`
- Create: `tests/test_round_orchestrated_cuda_wrapper.py`

**Interfaces:**
- Consumes: the existing resolver and `runpy` CLI wrapper.
- Produces: a GPU wrapper that performs a lightweight `nvidia-smi` preflight before the Python import/runpy boundary and exits nonzero on explicit GPU failure.

- [ ] **Step 1: Add a wrapper test with fake executables.** Assert the wrapper preserves `--device cuda:N`, invokes the preflight once, and does not launch the CLI when the preflight fails.
- [ ] **Step 2: Add the minimal preflight.** Run `nvidia-smi --query-gpu=index --format=csv,noheader` in the shell wrapper only for explicit GPU requests; keep CPU requests free of GPU probing. Do not convert failures to CPU.
- [ ] **Step 3: Keep the resolver as the final authority.** The Python resolver must still check `torch.cuda.is_available()` and visible device count after the preflight; the shell check is stabilization, not a replacement for fail-fast validation.
- [ ] **Step 4: Run the resolver and wrapper tests, then a one-sample GPU smoke.** Use the existing runpy wrapper and `/home/xzh/micromamba/envs/workspace/bin/python`; capture the selected CUDA device in the run manifest.

### Task 5: Validate Against Original MNSIM and Close the Regression Gate

**Files:**
- Create: `tests/test_resnet18_mnsim_reference_parity.py`
- Modify: `tests/test_round_orchestrated_interface.py` only for updated provenance fields
- Modify: `README.md` or a focused `docs/specs/` note if the final contract needs user-facing documentation

**Interfaces:**
- Consumes: the repaired ANN compatibility path, original MNSIM reference checkpoint, formal SNN path, and CUDA gate.
- Produces: evidence that the repair restores behavior without introducing QAT or finetune dependencies.

- [ ] **Step 1: Build a deterministic ResNet18 reference comparison.** For one fixed CIFAR-10 tensor and the same checkpoint, run the original MNSIM master path and the repaired round-orchestrated ANN path. Compare successful execution, output shape, finite logits, and documented tolerance for the fixed-range path; record the residual ranges used by both paths.
- [ ] **Step 2: Run ANN accuracy in two modes.** Verify VGG8 Original/PIM and ResNet18 Original/PIM with one sample. The ResNet run must use the old checkpoint directly and must not create a QAT output directory or calibration checkpoint.
- [ ] **Step 3: Run the formal SNN one-sample accuracy.** Verify original software inference and `adc_digital_lif` PIM inference from the pretrain checkpoint, using only `pretrain` and `test` data identities.
- [ ] **Step 4: Run the full test suite.**

Run:

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest -q
```

Expected gate: no functional failures; tests that require unavailable external artifacts must skip with an explicit path and must not reference the obsolete finetune checkpoint by default.

- [ ] **Step 5: Record the evidence.** Save the one-sample reports, checkpoint hashes, split identities, selected device, and test summary under a new results directory. Do not overwrite existing user-generated `results/` artifacts.

## Acceptance Criteria

1. The original ResNet18 checkpoint loads without requiring post-add calibration or QAT when its residual-add `last_value` ranges are present.
2. A checkpoint with no valid residual-add range fails clearly and can be handled only by an explicit frozen calibration workflow.
3. ANN Original and PIM paths retain the original MNSIM method semantics.
4. SNN accuracy uses the software-pretrain checkpoint, logical `train`/physical `pretrain`, and physical `test`; a `finetune` directory cannot be selected implicitly.
5. Explicit CUDA requests never silently fall back to CPU, and the wrapper uses the existing runpy entry.
6. The full test suite passes apart from clearly skipped tests for absent external artifacts, with no obsolete hard-coded finetune checkpoint remaining in the default path.
