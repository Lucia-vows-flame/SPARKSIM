# Local Analog LIF LUT Design

Date: 2026-09-22

## Goal

Add a local HSPICE/LUT-backed analog current-driven LIF backend to the round-orchestrated simulator without changing the Phase 1 -> Phase 2 execution architecture or importing the legacy `snn_lut_runtime` and `snn_lut_adapter` modules.

## Scope and invariants

- Phase 1 round construction remains unchanged.
- Phase 2 lowering, MNSIM static evaluation, boundary commits, and report aggregation remain the execution skeleton.
- `LIF_Mode = digital` preserves the existing digital LIF behavior.
- `LIF_Mode = analog_lut` selects the new local backend.
- The analog hardware has two independent current rails: `Ipos` charges the membrane branch and `Ineg` discharges it. No `R_s` parameter is part of this backend.
- The analytic membrane equation may be used only for optional functional/accuracy comparison. It must not produce analog frontend or LIF latency, energy, or power metrics.

## Local module boundaries

### `MNSIM/Hardware_Model/analog_lif_lut.py`

Owns artifact loading and interpolation. It defines local exceptions, immutable table metadata, scalar/array frontend queries, and scalar/array LIF queries. The loader accepts either a LUT root or an artifact directory containing `lut_manifest.json` and uses read-only NumPy arrays.

The required tables are:

```text
frontend_gain112_p10008/
  ipos_a.npy, ineg_a.npy, iout_to_lif_a.npy
  settle_latency_s.npy, energy_vdd_j.npy, valid.npy, status.npy
lif_current_step20ns/
  iin_a.npy, vmem0_v.npy, vmem_next_v.npy
  spike_margin_v.npy, spike_flag.npy, energy_rails_j.npy
  valid.npy, status.npy
```

Continuous fields use bounded bilinear interpolation. `spike_flag` is recomputed from the interpolated `spike_margin_v` and is never interpolated. A query fails closed for missing files, manifest/checksum/schema mismatch, invalid neighbors, non-finite fields, non-monotonic axes, or out-of-range coordinates.

### `MNSIM/Scheduling/phase2/analog_lif.py`

Owns the analog LIF state transition and performance accounting. It consumes positive and negative crossbar currents, queries the local frontend table, then queries the local LIF table with the current membrane state. It returns next membrane state, spike codes, and a cost record containing only LUT-derived latency and energy.

The frontend cost is aggregated over closed differential-xbar outputs. LIF energy is summed over queried neurons; LIF latency is the characterized `step_latency_s` per synchronous action window. Power is derived from LUT energy divided by LUT latency, never from the analytic equation.

### `MNSIM/Scheduling/phase2/executor.py`

Keeps the existing round loop and dispatches at the LIF execution point. The dispatcher selects the digital or analog backend from the parsed `LIF_Mode` value. Static MNSIM evaluation remains available for the same lowered round and is not replaced by a Route-A path.

## Configuration contract

Add:

```ini
[Neuron]
LIF_Mode = digital
Analog_LIF_LUT_Root = 08_lut
Analog_LIF_LUT_Verify_Checksums = 1
Analog_LIF_LUT_Profile = fast_1h_experimental_7pct
Analog_LIF_Negative_Vmem_Policy = error
```

Supported modes are exactly `digital` and `analog_lut`. The default remains `digital`. The negative successor policy defaults to `error`; an explicit `clamp_zero` policy may floor a negative measured successor before the next lookup while retaining the raw value and clamp count in the report.

## Functional and performance separation

The analog backend has two separate outputs:

1. Functional state: `vmem_next` and `spike_flag`, used by subsequent timesteps and boundary state.
2. Hardware cost: frontend settle latency/energy and LIF action-window latency/energy, copied from the LUT artifact.

No functional approximation is allowed to silently substitute for missing LUT performance data.

## Audit fields

Every analog execution report records LUT schema/profile/root, query counts, interpolation method, axis clamp/reject counts, checksum verification state, negative-Vmem policy, raw minimum clamped successor, frontend energy/latency, LIF energy/latency, and derived power.

## Verification

Tests must cover:

- valid artifact loading and manifest/checksum rejection;
- axis, dtype, shape, status, and finite-value validation;
- bounded bilinear interpolation and spike-margin sign decisions;
- out-of-range and invalid-neighbor fail-closed behavior;
- analog state progression and LUT-only cost accounting;
- `digital`/`analog_lut` configuration dispatch;
- unchanged Phase 1/Phase 2 round ordering and digital regression behavior.

