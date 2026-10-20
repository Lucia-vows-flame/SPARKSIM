# QAT precision and activation grids

Current training defaults remain fixed signed 9-bit weights and uniform
9-bit non-spike activations, with native binary LIF spikes. No training run
is started or existing checkpoint overwritten by this change. No AMP is
enabled. Existing QAT launch scripts continue to select the same defaults.

Two independent options are now supported in the software `FIX_TRAIN` QAT
path, including V7 temporal QAT:

| Option | Meaning |
| --- | --- |
| `precision_mode = fixed` | Global fixed weight/activation widths; differing layer widths are rejected |
| `precision_mode = mixed` | Explicit per-layer weight and/or activation widths |
| `activation_quantization = uniform` | Existing symmetric linear activation grid |
| `activation_quantization = mu_law` | Symmetric non-uniform companding grid |

Mixed precision means a configured allocation, not an automatic bit-width
search and not FP16/FP32 mixed-precision arithmetic. Supported signed widths
in the new policy are 2 through 16 bits. Weight quantization stays uniform
with its existing STE; only activation grids are selectable. Both signed
grids use `2**bit - 1` levels, matching the existing symmetric convention.

## Configuration and CLI

The training entry is `scripts/finetune_snn_v7.py`. Existing arguments remain
valid. Add one of these options to an ordinary training command:

```sh
# Current experiment policy (also the implicit default):
--qat-policy configs/quantization/qat_fixed9.json

# Optional fixed precision with non-uniform activation:
--qat-policy configs/quantization/qat_fixed9_mu_law.json

# Optional explicit layer widths (example only):
--qat-policy configs/quantization/qat_mixed_example.json

# Optional combination, retaining the mixed profile's allocation:
--qat-policy configs/quantization/qat_mixed_example.json --activation-quantization mu_law
```

Individual CLI options override global settings in the policy file:

```text
--precision-mode fixed|mixed
--weight-bits 9
--activation-bits 9
--activation-quantization uniform|mu_law
--activation-mu 15
```

The JSON schema ID is `mnsim_qat_quantization/v1`. See the three example
files for all fields. `layer_overrides` uses final graph IDs such as `L000`,
including the positions occupied by automatically inserted BN/LIF nodes.
Use `layer.accuracy_layer_id` to identify the layers. A weight override
requires Conv/FC; an activation override requires Conv/FC/BN/element_sum.
Unknown IDs, unsupported fields, invalid widths and attempts to quantize
LIF spike outputs are rejected. Mixed precision does not automatically
reconcile different domains at concat/multiply boundaries; existing common
domain checks still apply. Residual sums consume dequantized real values and
then use their own configured output quantizer.

Programmatic construction accepts the same policy as
`hardware_config['qat_quantization']` in `get_net`/`NetworkGraph`. When this
key is absent, legacy layer configurations and checkpoint keys are retained.

## Mu-law activation

For range `s`, coefficient `mu > 0` and `K = 2**(bit-1)-1`:

```text
x_norm = clip(x / s, -1, 1)
u = sign(x_norm) * log1p(mu * abs(x_norm)) / log1p(mu)
k = round(K * u)
x_quant = sign(k) * expm1(abs(k)/K * log1p(mu)) / mu * s
```

The grid has finer spacing near zero and coarser spacing near its endpoints.
It includes zero, is symmetric, and uses an identity straight-through
gradient. It is a deterministic companding codebook, not Lloyd-Max training,
quantile calibration or the existing READC ADC model. `mu = 15` is a
configurable example coefficient, not a claimed optimal setting.

The per-layer range uses the existing FIX_TRAIN EMA rule during training or
explicit calibration. Eval freezes it. New mu-law quantizers handle silent
and singleton calibration activations with a positive dtype-epsilon minimum;
uncalibrated eval still fails. Range and policy persist in the checkpoint.
The temporal graph now propagates each Conv/FC/BN/add node's own quantizer
metadata; LIF outputs always restore the binary spike domain. LIF equations,
thresholds, surrogate gradients and state-reset rules are unchanged.

## Persistence, resume and hardware boundary

Non-default policies add a UTF-8 JSON `qat_quantization_policy_utf8` buffer
to the model state. Decode it with `decode_qat_policy` and construct the
matching network before loading. Ordinary software-pretrain weights can
initialize a new policy. A tagged QAT checkpoint with another policy fails
to load, including under `strict=False`, rather than silently changing
activation interpretation. Non-default policy metadata is also included in
the fine-tune run configuration, so resume requires the same policy.
Default fixed-9 uniform models add no new checkpoint buffers or resume
configuration keys.

These options are integrated into software QAT. The existing simulator's
linear input-code reconstruction does not decode a mu-law codebook. Non-uniform
hardware export/PIM execution therefore fails explicitly; its nominal
`full_scale/K` metadata must not be treated as an actual constant code step.
This does not affect current fixed-9 uniform experiments. Mixed uniform
layers expose their actual widths through existing layer metadata, but the
formal signed-9 deployment contract still applies wherever it is required;
this change does not relax that contract.

## Verification

`tests/test_qat_quantization_policy.py` covers default numerical/gradient and
checkpoint compatibility, actual mixed-width codes, own-node quantization
metadata, non-uniform codebook levels/saturation/STE, silent calibration,
binary spikes, frozen eval ranges, BN/residual quantization, serialization,
mismatched checkpoint rejection, final graph IDs and CLI defaults. Additional
regressions cover existing quantization, QAT training/resume and SNN graphs.
No dataset training or full experiment is run for this implementation.
