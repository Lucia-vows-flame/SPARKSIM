# MNSIM-2.0

## Active Architecture

The repository root contains the active `task-20260707-round-orchestrated-mnsim`
architecture. Use the round-orchestrated wrappers under `scripts/` for current
runs. Historical workspaces and experiment archives are not part of this
source release.
**Citation Information**: Zhenhua Zhu, Hanbo Sun, Tongxin Xie, Yu Zhu, Guohao Dai, Lixue Xia, Dimin Niu, Xiaoming Chen, Xiaobo Sharon Hu, Yu Cao, Yuan Xie, Huazhong Yang, and Yu Wang, MNSIM 2.0: A Behavior-Level Modeling Tool for Processing-In-Memory Architectures, in IEEE TRANSACTIONS ON COMPUTER-AIDED DESIGN OF INTEGRATED CIRCUITS AND SYSTEMS (TCAD), VOL. 42, NO. 11, NOVEMBER 2023

**Main Contributor**:
Zhenhua Zhu<sup>1*</sup>, Hanbo Sun<sup>1</sup>, Tongxin Xie<sup>1</sup>, Lixue Xia<sup>2</sup>, Gokul Krishnan<sup>6</sup>, Dimin Niu<sup>2</sup>, Qiuwen Lou<sup>3</sup>,

Xiaoming Chen<sup>4</sup>, Yuan Xie<sup>2, 5</sup>, Yu Cao<sup>7</sup>, X. Sharon Hu<sup>3</sup>, Yu Wang<sup>1*</sup>, and Huazhong Yang<sup>1</sup>

<sup>1</sup>Tsinghua University, <sup>2</sup>Alibaba Group, <sup>3</sup>University of Notre Dame, 
<sup>4</sup>Institute of Computing Technology, Chinese Academy of Sciences, 
<sup>5</sup>The Hong Kong University of Science and Technology,
<sup>6</sup>Arizona State University
<sup>7</sup>University of Minnesota System

<sup>*</sup>zhuzhenh18@mails.tsinghua.edu.cn, yu-wang@tsinghua.edu.cn

MNSIM-2.0 aims to model the HW performance and NN computing accuracy of Processing-In-Memory (PIM) architectures. If you have any questions and suggestions about MNSIM-2.0 please contact us via e-mail. We hope that MNSIM-2.0 can be helpful to your research work, and sincerely invite every PIM researcher to add your ideas to MNSIM-2.0 to enlarge its function.

For more information about MNSIM-2.0, please refer to the MNSIM_manual.pdf and the IEEE TCAD paper of MNSIM 2.0.

*Weights File: https://1drv.ms/f/s!AtForEDTP2-PgkzYkCJNONO9xnX6?e=y4s0XG*

## Round-Orchestrated ANN Runner

Use the supported wrapper for a combined accuracy and hardware run:

```bash
bash scripts/run_round_orchestrated_sim.sh \
  --network vgg8 \
  --weights /absolute/path/to/vgg8_checkpoint.pth \
  --num-classes 10 \
  --sim-config /absolute/path/to/SimConfig.ini \
  --output-dir /absolute/path/to/run-output \
  --dataset-module MNSIM.Interface.cifar10 \
  --run-hardware
```

Accuracy runs by default. A checkpoint is always mandatory, while
`--dataset-module` is required only when accuracy is requested. For a
hardware-only ResNet18 run, omit the dataset module and use:

```bash
bash scripts/run_round_orchestrated_sim.sh \
  --network resnet18 \
  --weights /absolute/path/to/resnet18_checkpoint.pth \
  --num-classes 10 \
  --sim-config /absolute/path/to/SimConfig.ini \
  --output-dir /absolute/path/to/run-output \
  --skip-accuracy \
  --run-hardware
```

Use `--accuracy-seed N` to make the stochastic device-error map reproducible
for an accuracy run. The runner scopes the legacy NumPy random generator to
the device-perturbation update and records the seed in the manifest and
`accuracy_report.json`; omitting the option preserves the historical random
behavior.

## Custom Weight Schedules

Custom algorithms can output a versioned JSON decision file and pass it with
`--custom-schedule-in my_schedule.json` on the hardware runner. The simulator
validates tile assignments and legal round order, builds its internal artifact,
and computes energy and latency through the existing Phase-2 path. Algorithm
inputs are unrestricted. See the [custom format v1 specification](docs/specs/2026-10-10-custom-weight-schedule-format.md),
[JSON Schema](docs/schemas/weight-schedule-v1.schema.json), and
[example](docs/examples/weight-schedule-v1.json).

## SNN Accuracy Profiles

The V7 model `resnet18_snn_v7_phase6` uses SpikingJelly-style residual blocks,
explicit BN/LIF placement, and convolutional downsampling without pooling.
Both `scripts/train_snn_v7.py` and `scripts/finetune_snn_v7.py` accept this model.
Residual hardware evaluation requires calibrated/QAT BN and post-add ranges and
the digital LIF backend. See [residual SNN runtime](docs/specs/2026-10-03-residual-snn-runtime.md)
for usage, validation, and the inherited MNSIM cost-accounting limits.

V7 training uses `pretrain` or `finetune` for optimization, `calibration` for
checkpoint selection, and `test` once after the selected checkpoint is loaded.
Each run writes `*_best.pth` (model weights), `*_latest.pth` (model, optimizer,
scheduler and random state), and `*_history.json` (per-epoch metrics and final
test result). Use `--resume` with a larger `--epochs` value to continue an
interrupted run; use `--overwrite` explicitly to replace existing run outputs.
Earlier best-only checkpoints remain loadable as finetune inputs but cannot
restore optimizer state or training history.

Select the formal SNN accuracy hardware profile only through the
`[SNN Accuracy]` section of `SimConfig.ini`. The CLI intentionally has no
profile or artifact override flags. A one-sample ADC-profile evaluation uses
the supported wrapper:

```bash
MNSIM_DVS_BINNED_ROOT=/absolute/path/to/v7_dataset \
MNSIM_DVS_TRAIN_BATCH=1 \
MNSIM_DVS_TEST_BATCH=1 \
bash scripts/run_round_orchestrated_sim.sh \
  --network vgg8_snn_v7_phase6 \
  --weights /absolute/path/to/vgg8_snn_v7_phase6_software_pretrain_best.pth \
  --num-classes 8 \
  --sim-config SimConfig.ini \
  --output-dir results/snn_pim_adc_smoke \
  --dataset-module MNSIM.Interface.dvs_binned_npz \
  --max-samples 1 \
  --overwrite \
  --device 0
```

- `SNN Original Accuracy` uses unquantized PyTorch weighted operations and the
  shared SpikingJelly LIF wrapper.
- `SNN Processing-in-Memory Accuracy` is the only public PIM accuracy name.
- `adc_digital_lif` uses the fixed 5-bit Hong 2025 ReADC model with a
  training-split, identity-bound calibration artifact.
- `frontend_lut_lif_lut` uses exact no-ADC weighted reconstruction and requires
  a qualified frontend/LIF Monte Carlo/PVT characterization artifact.
- Formal LUT-profile PIM accuracy fails closed until that qualified artifact is
  supplied.
- Physical current and hardware frontend/LIF LUT values never enter accuracy
  tensors, spikes, or membrane state.

For a one-batch numerical trace, use the safe wrapper below. It records every
graph layer's reference/PIM error, ADC saturation, and input-code saturation.

```bash
bash scripts/run_pim_numerical_audit.sh \
  --network resnet18 \
  --weights /path/to/cifar10_resnet18_params.pth \
  --num-classes 10 \
  --sim-config configs/hardware_profiles/g8_600plus_ann.ini \
  --dataset-module MNSIM.Interface.cifar10 \
  --samples 1 \
  --adc-action NONE \
  --calibration-mode frozen_reference \
  --output results/resnet18_numeric_audit.json
```

`NONE` is audit-only: it bypasses the ADC clamp while retaining bit slicing and
output quantization. `frozen_reference` keeps all calibrated activation ranges
fixed during the audit. New residual-add nodes own their configured activation
bit width and scale; the following PIM layer receives that node state
explicitly. A checkpoint predating post-add quantization must be calibrated
explicitly, for example by adding `--calibration-batches 16`; audit/eval never
adapts scales from its test sample. The JSON reports ADC and input-code
saturation separately along with every layer's input and output quantization
state.

The terminal always prints a readable aggregate hardware summary after a
successful hardware stage. `--show-rounds` additionally prints one row per
physical round with xbar use/capacity, utilization, mapped area, latency,
energy, and average power; it does not print layer slices or execution
internals. `--overwrite` replaces only runner-owned JSON artifacts and
preserves unrelated files in the output directory. The runner writes
`run_manifest.json`, `summary.json`,
`accuracy_report.json`, `round_plan.json`, and `phase2_report.json` for stages
that run. Exit code `0` means success, `1` means a requested stage failed, and
`2` means an invalid request or preparation failure.

Hardware metrics count each MAC as two operations for TOPS reporting. The
hardware flow is exclusively round-aware Phase 1/Phase 2 execution; it has no
legacy global whole-model fallback.

The bundled `scripts/run_vgg8_g8.sh` preset has 1152 MNSIM group slots
(`6x6` tiles, `4x4` PEs, and `Group_Num=2`). Its physical xbar-array count
also includes `Xbar_Polarity`, so the current differential configuration
(`Xbar_Polarity=2`) reports 2304 physical xbar arrays.

The same `SimConfig.ini` is used by the MNSIM backend and round-aware Phase 1
and Phase 2 execution. `[Algorithm Configuration] Weight_precision` must be
positive. The low-level scheduling math supports a power-of-two
`Device_Level`:

```text
Device_Level = 2^device_bits, where device_bits >= 1
```

The public round-orchestrated runner accepts only the bundled checksum-locked
Stanford/ASU and Stanford-PKU RRAM profiles described below. Both use 256
conductance states, with separate read and write-transition artifacts where
needed. Device levels must be a power of two.
Phase 1 and Phase 2 derive `device_bits` through the same helper, then use:

```text
weight_plane_count = ceil(effective_weight_bits / device_bits)
M_i = row_blocks * output_subsets * weight_plane_count * Xbar_Polarity
```

For native differential metadata, `Weightbit` loses its sign bit once when
forming `effective_weight_bits`; fallback `Weight_precision` is already an
effective magnitude width and never loses another sign bit. `Xbar_Polarity`
is a separate physical-array topology (`1` or `2`), not device precision. It
affects both physical xbar capacity and `M_i` independently of `device_bits`.

The G8 wrapper defaults to Stanford/ASU and accepts Stanford-PKU as an
alternative. `VGG8_G8_BASE_SIM_CONFIG` may provide custom architecture and
peripheral settings; the selected device profile replaces its device fields.

```bash
VGG8_G8_BASE_SIM_CONFIG=/absolute/path/to/SimConfig.custom.ini \
VGG8_G8_DEVICE_PROFILE=pku \
bash scripts/run_vgg8_g8.sh
```

## Final RRAM Models

Only two device families are bundled under `device_models/final_rram/`:

- `rram_stanford_asu_256`: Stanford/ASU 256-state model with a calibrated
  source-to-target write transition table.
- `rram_memtorch_stanford_pku_256`: Stanford-PKU nominal read model.
- `rram_memtorch_stanford_pku_write256`: Stanford-PKU compact-model write
  transitions for full hardware accounting.

The public runner validates the model JSON checksum, complete resistance table,
device level count, read settings, variation model, Route-A switches, seed,
capabilities, and provenance before loading the network. A mismatch fails with
`INVALID_FINAL_RRAM_MODEL`.

The nominal Stanford-PKU profile is suitable for read/accuracy analysis; use
its `write256` profile for hardware runs. The G1-G11 hardware sweep injects one
of the write-capable models into every generated
`SimConfig.ini` without changing the resource ladder:

```bash
bash scripts/run_round_orchestrated_g1_g11_sweep.sh \
  --base-sim-config SimConfig.ini \
  --device-model rram_stanford_asu_256 \
  --weights /absolute/path/to/checkpoint.pth \
  --output-root /absolute/path/to/sweep-output
```

Use `rram_memtorch_stanford_pku_write256` for Stanford-PKU. Omitting
`--device-model` selects Stanford/ASU; arbitrary model JSON paths and the
nominal-only Stanford-PKU profile are rejected for hardware write accounting.
Generated configs record the model id, source hash, resistance table, and
transition-model provenance in `[Final RRAM Model]`.
