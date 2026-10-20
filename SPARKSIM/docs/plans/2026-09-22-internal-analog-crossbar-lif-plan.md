# Internal Analog Crossbar LIF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Replace externally supplied analog-current payloads with an internal input/weight/read-voltage to differential-crossbar-current calculation while preserving the existing Phase 1 -> Phase 2 execution architecture.

**Architecture:** Phase 1 and the Phase 2 round/segment/timestep ordering remain unchanged. At the LIF producer boundary, the new local numerical path loads input values, quantized weight/device conductance data, and read-voltage configuration, computes `Ipos = V @ Gpos` and `Ineg = V @ Gneg`, then calls the existing local frontend/LIF LUT backend. Original MNSIM continues to produce the complete baseline hardware report; analog accounting subtracts ADC terms at final aggregation and adds LUT frontend/LIF terms.

**Tech Stack:** Python 3, NumPy, configparser, existing MNSIM crossbar/conductance tables, pytest, HSPICE/LUT metadata.

**Spec:** `docs/specs/2026-09-22-local-analog-lif-payload-contract.md` plus the approved requirements in the current session.

## Global Constraints

- Preserve Phase 1 round construction and Phase 2 execution ordering.
- `LIF_Mode = digital` remains behaviorally unchanged.
- The analog path computes currents internally; the final CLI contract must not require precomputed `Ipos`/`Ineg`.
- Use differential current rails; do not add or use `R_s`.
- Do not import legacy `snn_lut_runtime`, `snn_lut_adapter`, or Route-A runtime modules.
- Hardware performance metrics for the analog frontend/LIF come from local LUT/HSPICE data, not analytic membrane integration.
- The implementation target is single-sample inference.
- ADC terms remain in the original MNSIM result and are subtracted only in final analog accounting.

---

### Task 1: Internal Crossbar Current Engine

**Files:**
- Create: `MNSIM/Scheduling/phase2/analog_crossbar.py`
- Modify: `MNSIM/Scheduling/phase2/analog_lif_inputs.py`
- Test: `tests/test_round_phase2_analog_crossbar.py`

**Interfaces:**
- `compute_differential_currents(input_codes, read_voltage_v, gpos_s, gneg_s) -> tuple[np.ndarray, np.ndarray]`
- `AnalogCrossbarNumericalEngine.from_sim_config(sim_config_path, weight_codes, polarity=2)`
- `AnalogCrossbarNumericalEngine.compute(input_values, weight_codes_by_layer, output_neuron_count) -> (ipos_a, ineg_a, audit)`

- [ ] Write RED tests for voltage generation, conductance lookup, positive/negative arrays, matrix shape validation, finite-value validation, and exact `V @ G` units.
- [ ] Implement the pure matrix kernel and verify `Ipos[j] = sum_i V[i] * Gpos[i,j]` and the equivalent negative rail.
- [ ] Implement weight-code to conductance lookup using the local device table, with no bit-slice inference or `R_s`.
- [ ] Add current-engine audit fields for read voltage, conductance profile, rail shapes, and payload identity.
- [ ] Run `pytest tests/test_round_phase2_analog_crossbar.py -q` and commit.

### Task 2: Replace External Current Payload Contract

**Files:**
- Modify: `MNSIM/Scheduling/phase2/analog_lif_inputs.py`
- Modify: `MNSIM/Scheduling/phase2/executor.py`
- Modify: `docs/specs/2026-09-22-local-analog-lif-payload-contract.md`
- Test: `tests/test_round_phase2_analog_dispatch.py`

**Interfaces:**
- Analog payload consumes input/weight/read-voltage metadata, not precomputed rail currents.
- `AnalogLifPayloadSession.resolve(...)` obtains numerical currents from the local crossbar engine, then returns the existing `AnalogLifNumericalInput` to `AnalogLifBackend`.

- [ ] Add a RED test proving a payload with input plus conductance/weight data computes currents internally.
- [ ] Keep direct-current payload support only as an explicit low-level test fixture or remove it from the public CLI contract.
- [ ] Persist measured `Vmem_next` and spike identity exactly as currently implemented.
- [ ] Reject missing input/weight/read-voltage data before analog success; never fabricate zero currents.
- [ ] Run analog dispatch, boundary, and digital regression tests and commit.

### Task 3: ADC Final Accounting

**Files:**
- Modify: `MNSIM/Scheduling/phase2/executor.py`
- Modify: `MNSIM/Scheduling/phase2/report.py`
- Modify: `MNSIM/Scheduling/phase2/static_power_ledger.py` only if needed for explicit accounting metadata
- Test: `tests/test_round_phase2_analog_dispatch.py`
- Test: `tests/test_round_phase2_report.py`

- [ ] Add RED tests that verify original MNSIM ADC latency, energy, power, and area are present in raw audit but removed from final analog totals.
- [ ] Preserve the original MNSIM invocation and raw result unchanged.
- [ ] Subtract ADC terms only during final analog aggregation, then add frontend/LIF LUT latency and energy exactly once.
- [ ] Report `adc_subtracted_*` and `analog_frontend_lif_*` fields with a reconciliation assertion.
- [ ] Keep digital totals byte-for-byte behaviorally equivalent and commit.

### Task 4: Transistor-Based Analog Area Model

**Files:**
- Create: `MNSIM/Hardware_Model/analog_lif_area.py`
- Modify: `MNSIM/Scheduling/phase2/report.py`
- Modify: LUT metadata/spec documentation with area provenance
- Test: `tests/test_analog_lif_area.py`

- [ ] Identify the actual HSPICE/netlist transistor devices, technology node, `W/L`, device counts, matching overhead, and layout assumptions used by the analog frontend and LIF.
- [ ] Write RED tests for deterministic area calculation and provenance serialization.
- [ ] Implement an explicit transistor-density/geometry calculation with named assumptions; reject missing technology or device dimensions.
- [ ] Add calibrated area fields to analog reports; never use numeric zero as a calibrated result.
- [ ] Run area and report tests and commit.

### Task 5: End-to-End Verification and Cleanup

**Files:**
- Modify: relevant docs and tests only.

- [ ] Run focused analog, digital, boundary, report, and static-power tests.
- [ ] Run the full suite and distinguish code failures from unavailable external checkpoints.
- [ ] Run a real internal-current smoke using the supplied LUT/device conductance data.
- [ ] Verify no local analog module imports legacy LUT/Route-A runtime modules.
- [ ] Verify single-sample timestep ordering and state persistence.
- [ ] Review git status to ensure unrelated runner/export/results changes remain untouched.
- [ ] Commit only required implementation/documentation changes.
