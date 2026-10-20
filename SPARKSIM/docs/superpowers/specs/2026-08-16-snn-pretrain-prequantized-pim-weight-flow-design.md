# SNN Pretrain-to-Prequantized PIM Weight Flow Design

## Goal

The round-orchestrated SNN simulator must accept a software-pretrain checkpoint,
quantize its deployable Conv/FC weights exactly once with the current MNSIM
symmetric signed 9-bit rule, persist the resulting deployment checkpoint, then
run PIM accuracy by directly decoding that checkpoint. The PIM inference phase
must never quantize deployment weights again.

This work only closes the weight lifecycle. It does not change the formal LUT
perturbation requirement, the ADC/ReADC contract, or the existing device
non-ideality semantics.

## Supported Input

Each invocation receives an explicit software-pretrain checkpoint through the
existing weights input. The checkpoint is the source artifact and remains
unchanged. The flow currently targets the BN-free V7 SNN checkpoints and their
logical Conv/FC layers.

The source checkpoint is not accepted as a PIM deployment checkpoint. Before
PIM accuracy starts, the simulator must create a separate 9-bit deployment
artifact under the run output directory.

## Quantization Contract

For each logical Conv/FC layer, all split sublayers share one scale:

```text
threshold = 2^(9 - 1) - 1 = 255
lsb = max(max(abs(W_split))) / threshold
digit = round(W_split / lsb)
W_deploy = digit * lsb
digit in [-255, 255]
```

The implementation must reuse the repository's current MNSIM 9-bit weight
quantization rule. It must not introduce a second exporter-specific numerical
rule. Non-weight tensors, tensor keys, tensor shapes, and tensor dtypes remain
unchanged.

Quantization happens once per simulator invocation, before PIM inference. It
must not run per sample, per batch, per timestep, or inside a weighted-layer
forward call.

## Deployment Artifact

The quantization stage writes a run-local deployment bundle containing:

- `deployment_weights_9bit.pth`;
- `deployment_weights_9bit_manifest.json`.

The manifest records:

- source checkpoint path and SHA-256;
- deployment checkpoint SHA-256;
- artifact schema and quantizer identity;
- weight bit width and signed code range;
- logical layer identities;
- each layer's shared LSB, split count, code minimum, and code maximum;
- confirmation that all non-weight values are unchanged.

The deployment checkpoint must be written atomically. A failed export must not
leave a file that can be mistaken for a valid deployment artifact.

## Reload Boundary

After export, PIM accuracy must construct a fresh inference context and load
`deployment_weights_9bit.pth`. It must not continue with the in-memory model
used to produce the artifact. This reload boundary proves that PIM inference
depends only on the persisted deployment checkpoint and its declared
configuration.

The fresh context enables the prequantized-weight direct mode. Setup validates
every deployable layer before the first test sample:

- every stored weight is finite;
- every split in one logical layer shares the reconstructed LSB;
- every value reconstructs exactly from a signed 9-bit integer code in the
  checkpoint tensor dtype;
- every code lies in `[-255, 255]`;
- the checkpoint and manifest identities match.

Any validation failure terminates the run. There is no fallback to dynamic
quantization.

## PIM Consumption Order

The direct decoder produces the exact signed deployment digits. The formal PIM
path then applies the existing hardware mapping and non-idealities in their
current locations:

```text
validated signed 9-bit deployment digits
-> differential device-code / bit-slice mapping
-> R-ratio behavior
-> configured RRAM device variation with Device_Seed = 112
-> configured SAF behavior
-> profile-specific PIM numerical forward
```

The direct-weight feature does not bypass R-ratio, RRAM variation, or SAF. It
only removes the redundant conversion from already-quantized checkpoint values
back through the weight quantizer.

## Accuracy Profile Boundaries

Both formal SNN PIM hardware profiles consume the same validated deployment
weights:

- `adc_digital_lif` continues through its ADC/ReADC and digital-LIF accuracy
  contract;
- `frontend_lut_lif_lut` continues through no-ADC reconstruction and its
  characterized frontend/LIF perturbation contract.

This change does not make the LUT profile runnable without a qualified
perturbation artifact. The existing fail-closed behavior remains in force until
that independent artifact is available.

Original Accuracy, when requested, continues to use the original software
checkpoint and does not use the deployment artifact. A PIM-only invocation may
skip Original Accuracy without changing the deployment preparation stage.

## Public Interface

The existing weights argument remains the user-facing input and means
"software-pretrain checkpoint" for this flow. The run request and report expose
that deployment weights were generated and directly consumed. Public state must
distinguish:

- source checkpoint identity;
- generated deployment checkpoint identity;
- direct prequantized-weight mode;
- quantization count, which must be exactly one for a completed PIM run.

No user must manually invoke a separate exporter before starting the simulator.

## Failure Handling

The simulator fails before PIM evaluation when:

- the source checkpoint is missing or malformed;
- a deployable weight is non-finite;
- quantization cannot produce a valid signed 9-bit artifact;
- the deployment checkpoint or manifest checksum does not match;
- a deployment weight is off-grid or outside the signed code range;
- direct prequantized mode is not active in the fresh PIM context;
- any inference path attempts to invoke the dynamic weight quantizer.

Errors must identify the offending logical layer and artifact where applicable.

## Verification

Focused tests must prove:

1. a source pretrain checkpoint produces a valid run-local 9-bit deployment
   checkpoint and manifest;
2. the exporter matches the current MNSIM quantizer exactly for every Conv/FC
   weight tensor;
3. non-weight values are unchanged;
4. the PIM context reloads the persisted deployment checkpoint instead of
   reusing the exporter model;
5. the direct decoder returns the expected signed digits and shared LSB values;
6. monkeypatching the dynamic weight quantizer to raise does not break direct
   PIM inference;
7. off-grid, non-finite, out-of-range, checksum-mismatched, and wrong-bit-width
   artifacts fail before evaluation;
8. R-ratio, device variation, and SAF remain downstream of direct decoding;
9. the standard V7 checkpoint produces the same deployment codes as the
   independently verified 2026-08-16 export;
10. a small PIM accuracy smoke completes through the new two-stage flow.

Full-dataset WSL packaging is a downstream task. It starts only after this
weight flow passes focused and small real-data verification.
