# Local Analog LIF Numerical Payload Contract

Date: 2026-09-22

## Scope

`LIF_Mode = analog_lut` keeps the existing Phase 2 round, segment, timestep,
static MNSIM, and boundary-commit order. It changes the LIF execution point to
the local LUT backend. The backend does not infer electrical values from tensor
shapes, accumulator codes, or activity estimates.

The current CLI runner supplies shape metadata only. It does not produce this
numerical payload. Therefore current metadata-only runner requests are not an
end-to-end analog numerical integration and fail with
`MISSING_ANALOG_LIF_PAYLOAD` after the preceding static MNSIM step has run.

## Location and identity

The existing `model_inputs` argument remains a mapping from input state ID to
metadata. An input metadata record may additionally carry
`analog_lif_payloads`, keyed by the formal LIF node ID:

```python
model_inputs = {
    "model_input": {
        "shape": {"Inputchannel": 3, "Inputsize": [8, 8]},
        "analog_lif_payloads": {
            "L1": {
                "contract_version": 1,
                "electrical_layout": "closed_differential_outputs",
                "neuron_count": 16,
                "initial_vmem_v": [0.0] * 16,
                "timesteps": [
                    {
                        "source": "input_weight",
                        "input_values": [0.0, 0.0, 0.0],
                        "weight_codes_by_layer": {
                            "Conv1": [[0] * 16, [0] * 16, [0] * 16]
                        }
                    }
                ],
            }
        },
    }
}
```

There must be exactly one payload for each invoked analog LIF node. Each
timestep input and each measured state/spike result receives a SHA-256 payload
identity in the Phase 2 audit report.

## Supported electrical source

`source = input_weight` supplies finite, non-negative `input_values` and signed
integer `weight_codes_by_layer`. The matrix has one row per input and one
column per output neuron. The payload session constructs the local
`AnalogCrossbarNumericalEngine` from SimConfig and computes both rails.

The optional `weight_layer_id` selects one layer when the mapping has multiple
layers. Optional scalar `read_voltage_v` metadata must match SimConfig.

Precomputed `ipos_a`/`ineg_a` and explicit `gpos_s`/`gneg_s` payloads are not
public inputs and fail closed before analog success. The contract does not
support or infer bit-slice significance, scale factors,
multi-slice merging, sample activity, accumulator-code conversion, or other
electrical layouts. Extra bit-slice dimensions and non-closed layout labels
fail closed before a successful analog report is emitted.

## Membrane state

`initial_vmem_v` is mandatory once per node/sample stream and has one value per
neuron. A timestep must not supply `vmem0_v` or another reset. After timestep
zero, the next query always consumes the previous LUT-measured `vmem_next_v`
(after the configured negative-successor policy). Invocation order and vector
size are validated for every timestep.

## Hardware and accounting

In analog mode, the original MNSIM ADC primitive remains enabled while the
original timing, power, energy, and area models run. Its latency, energy, power,
and area are retained in `raw_mnsim_audit`. Final analog aggregation reports
the corresponding `adc_subtracted_*` terms and removes them only from final
analog totals. Positive ADC lane counts and output multiplex geometry remain
available as virtual output-lane geometry, including the `ADC_Num = 0` case.

The local frontend and LIF add LUT-characterized latency and energy exactly
once. Frontend output energies are summed and frontend latency is the maximum
settle time across the synchronous output window; one characterized LIF step
latency is then added. Power is LUT energy divided by this LUT latency.

The LUT does not characterize frontend/LIF area. Reports therefore mark
`area_characterized = false`, `area_um2 = null`, and
`accounting_scope.analog_frontend_lif.area_included = false`. A zero numeric
step-area placeholder required by the existing Phase 2 schema is not evidence
of a calibrated zero-area implementation.

The central static-power ledger may remain enabled for analog runs. Its
unmodeled converter inventory is retained as audit metadata; it does not replace
the raw dynamic ADC ledger or get silently folded into the LUT frontend/LIF
terms. `analog_accounting.reconciliation_passed` asserts that final latency,
energy, and area equal the raw baseline minus ADC terms plus the LUT frontend/LIF
cost exactly once.
