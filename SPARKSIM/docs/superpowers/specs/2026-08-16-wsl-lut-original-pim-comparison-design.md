# WSL LUT Original/PIM Accuracy Comparison Design

## Scope

Extend `exports/wsl_v7_lut_pim1600_prequantized_20260816` so every accuracy
task evaluates both the unquantized software checkpoint and the LUT PIM path on
the same deterministic, class-balanced 1600-sample subset.

This change is limited to the exported WSL bundle. It does not add LUT/LIF
perturbation, ADC reconstruction, SAF, or QAT.

## Formal Test Subset

The V7 `test` split contains 4000 samples: 500 samples for each of 8 classes.
The formal bundle subset contains 1600 samples selected as follows:

- group the sorted test files by label;
- use NumPy `default_rng(112)` to deterministically permute each class;
- select 200 files from every class;
- combine the selected files in a stable deterministic order;
- reuse the exact same dataset and DataLoader for Original and PIM evaluation.

The run directory must contain `test_subset_manifest.json`. The result JSON
must record the subset seed, samples per class, class counts, and the canonical
SHA-256 of the selected relative-path manifest.

Using the same dataset contents, file names, labels, and seed must produce the
same 1600-sample manifest on every run.

## Accuracy Paths

### Original Accuracy

Load the source software-pretrain checkpoint directly with prequantized mode
disabled. Evaluate it using the existing SNN `TRADITION` forward path and
SpikingJelly LIF parameters:

- `tau = 50.5016666556`
- `v_th = 0.01`
- `v_reset = 0.0`

Original Accuracy must not use signed 9-bit quantization, R-ratio, device
variation, SAF, ADC, or LUT/LIF perturbation.

### PIM Accuracy

After Original Accuracy completes, release the original model, then retain the
existing deployment flow:

```text
software pretrain checkpoint
-> one signed 9-bit quantization and persistence
-> fresh deployment checkpoint reload
-> direct integer-code decoding
-> R-ratio
-> optional device variation
-> no ADC reconstruction
-> SpikingJelly LIF
```

R-ratio remains enabled, SAF remains disabled, and device variation remains
controlled by the existing `--variation on|off` task switch.

Only one model instance may reside on the GPU at a time so the process remains
within the 5 GiB host-memory requirement and is suitable for an RTX 3060.

## Outputs

The terminal must print:

```text
Original accuracy: <value> (<correct>/<total>)
PIM accuracy: <value> (<correct>/<total>)
Accuracy drop: <original-minus-pim>
```

`pim_accuracy.json` retains its name for compatibility and adds:

- `original_accuracy`
- `original_correct`
- `pim_accuracy`
- `pim_correct`
- `accuracy_drop`
- `test_subset_seed = 112`
- `test_samples_per_class = 200`
- `test_subset_class_counts`
- `test_subset_manifest_sha256`
- separate Original and PIM elapsed times

The existing variation state, configured variation percentage, checkpoint
hashes, quantization count, memory metrics, and PIM confusion matrix remain in
the result.

## Failure Conditions

The task must fail before inference when:

- the test split does not contain all 8 classes;
- any class has fewer than 200 samples;
- the selected subset is not exactly 1600 samples;
- Original and PIM evaluation do not use the same selected dataset;
- CUDA is unavailable during a formal WSL run;
- peak process RSS reaches 5 GiB or more.

## Verification

Verification must cover:

- deterministic manifest generation for repeated seed-112 selection;
- exactly 200 samples per class and 1600 samples total;
- legacy and explicit variation task dispatch remains unchanged;
- a CPU smoke run reports both accuracy paths and the accuracy drop;
- result JSON and terminal output contain the required comparison fields;
- bundle SHA-256 manifest verification passes after export synchronization.
