# Local Analog LIF LUT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local HSPICE/LUT-backed analog current-driven LIF backend to the round-orchestrated simulator while preserving the existing Phase 1 -> Phase 2 -> MNSIM static-evaluation flow.

**Architecture:** Implement a self-contained LUT artifact loader/interpolator under `MNSIM/Hardware_Model`, a Phase 2 analog LIF state/cost backend under `MNSIM/Scheduling/phase2`, and a narrow executor dispatch selected by `[Neuron] LIF_Mode`. The digital backend remains the default and unchanged; the analog backend obtains all frontend/LIF performance quantities from the LUT artifact.

**Tech Stack:** Python 3, NumPy, configparser, pytest, existing round-orchestrated `MNSIM` package.

**Spec:** `docs/specs/2026-09-22-local-analog-lif-lut-design.md`

## Global Constraints

- Work only in `round_orchestrated_simulator/MNSIM-2.0`.
- Do not import or call legacy `MNSIM.Interface.snn_lut_runtime` or `snn_lut_adapter`.
- Preserve Phase 1 round construction and Phase 2 execution ordering.
- `LIF_Mode = digital` remains the default.
- Analog hardware metrics must come from LUT fields, never from the analytic membrane equation.
- No `R_s` parameter is added to the analog backend.
- LUT artifact errors and query errors fail closed.

---

### Task 1: Local LUT artifact loader and interpolation

**Files:**
- Create: `MNSIM/Hardware_Model/analog_lif_lut.py`
- Test: `tests/test_local_analog_lif_lut.py`

**Interfaces:**
- Produces `load_analog_lif_lut(root, verify_checksums=True, profile=None) -> AnalogLifLutBundle`.
- Produces `AnalogLifLutBundle.query_frontend(ipos_a, ineg_a) -> FrontendQuery`.
- Produces `AnalogLifLutBundle.query_lif(iin_a, vmem0_v) -> LifQuery`.
- Produces local exceptions `AnalogLutArtifactError`, `AnalogLutQueryError`, `AnalogLutOutOfRangeError`, and `AnalogLutInvalidPointError`.

- [ ] **Step 1: Write failing tests** for loading a small synthetic artifact, exact corner queries, bilinear interpolation, spike decision from interpolated margin, and rejected checksum/out-of-range/invalid-neighbor cases.
- [ ] **Step 2: Run tests to verify failure** with `pytest tests/test_local_analog_lif_lut.py -q`; expected failure is missing `MNSIM.Hardware_Model.analog_lif_lut` symbols.
- [ ] **Step 3: Implement the local loader** with manifest parsing, SHA256 verification, required table/array validation, read-only NumPy loading, monotonic axis checks, and profile metadata capture.
- [ ] **Step 4: Implement bounded bilinear interpolation** for frontend continuous outputs and LIF continuous outputs; derive `spike_flag` from interpolated `spike_margin_v`.
- [ ] **Step 5: Run the focused tests** with `pytest tests/test_local_analog_lif_lut.py -q`; expected PASS.
- [ ] **Step 6: Run existing artifact-related tests** with `pytest tests/test_readc_artifact.py tests/test_device_nonidealities.py -q`; expected PASS.
- [ ] **Step 7: Commit** with `git add MNSIM/Hardware_Model/analog_lif_lut.py tests/test_local_analog_lif_lut.py && git commit -m "feat: add local analog lif lut loader"`.

### Task 2: Analog LIF functional transition and LUT-only cost model

**Files:**
- Create: `MNSIM/Scheduling/phase2/analog_lif.py`
- Modify: `MNSIM/Scheduling/phase2/__init__.py`
- Test: `tests/test_round_phase2_analog_lif.py`

**Interfaces:**
- Produces `AnalogLifBackend(bundle, negative_vmem_policy="error")`.
- Produces `AnalogLifResult` with `vmem_next`, `spike_codes`, frontend/LIF latency and energy, and audit metadata.
- Produces `AnalogLifCost` with `latency_s`, `energy_j`, and `power_w` derived only from LUT values.

- [ ] **Step 1: Write failing tests** for a positive/negative rail query, state progression across two timesteps, `clamp_zero`, and proof that reported cost equals LUT energy/latency rather than analytic integration.
- [ ] **Step 2: Run tests to verify failure** with `pytest tests/test_round_phase2_analog_lif.py -q`; expected failure is missing analog backend symbols.
- [ ] **Step 3: Implement the backend** to validate matching current/state shapes, query frontend then LIF, preserve raw successor values, apply only the configured negative-state policy, and aggregate LUT-derived costs.
- [ ] **Step 4: Export the backend** from `MNSIM/Scheduling/phase2/__init__.py` without changing digital exports.
- [ ] **Step 5: Run focused tests** with `pytest tests/test_round_phase2_analog_lif.py -q`; expected PASS.
- [ ] **Step 6: Commit** with `git add MNSIM/Scheduling/phase2/analog_lif.py MNSIM/Scheduling/phase2/__init__.py tests/test_round_phase2_analog_lif.py && git commit -m "feat: add local analog lif phase2 backend"`.

### Task 3: SimConfig mode parsing and artifact configuration

**Files:**
- Create: `MNSIM/Scheduling/phase2/lif_mode.py`
- Modify: `SimConfig.ini`
- Test: `tests/test_analog_lif_config.py`

**Interfaces:**
- Produces `AnalogLifConfig` with normalized `mode`, `lut_root`, `verify_checksums`, `profile`, and `negative_vmem_policy`.
- Produces `load_lif_mode_config(sim_config_path) -> AnalogLifConfig`.

- [ ] **Step 1: Write failing tests** for default digital mode, explicit `analog_lut`, invalid mode, invalid policy, and relative LUT root resolution against the SimConfig directory.
- [ ] **Step 2: Run tests to verify failure** with `pytest tests/test_analog_lif_config.py -q`; expected failure is missing parser symbols.
- [ ] **Step 3: Implement strict configuration parsing** with `[Neuron]` defaults and fail-closed validation.
- [ ] **Step 4: Add the `[Neuron]` section** to `SimConfig.ini` with `LIF_Mode = digital` and the documented analog defaults.
- [ ] **Step 5: Run focused and digital configuration tests** with `pytest tests/test_analog_lif_config.py tests/test_peripheral_literature_profile.py -q`; expected PASS.
- [ ] **Step 6: Commit** with `git add MNSIM/Scheduling/phase2/lif_mode.py SimConfig.ini tests/test_analog_lif_config.py && git commit -m "feat: configure digital and analog lif modes"`.

### Task 4: Phase 2 executor dispatch and report integration

**Files:**
- Modify: `MNSIM/Scheduling/phase2/executor.py`
- Modify: `MNSIM/Scheduling/phase2/types.py` only if a typed result field is required by the existing report schema
- Modify: `MNSIM/Scheduling/phase2/report.py` only for additive analog audit fields
- Test: `tests/test_round_snn_executor.py`
- Test: `tests/test_round_phase2_analog_dispatch.py`

**Interfaces:**
- Existing `execute_round_plan(...)` signature remains unchanged.
- Digital requests continue through `execute_digital_lif_update` and `DigitalLifCostModel`.
- Analog requests are dispatched through the local `AnalogLifBackend` and produce additive audit fields.

- [ ] **Step 1: Write failing dispatch tests** that build the existing minimal RoundPlan, select `digital` and `analog_lut`, and assert backend identity, unchanged timestep ordering, and LUT cost fields.
- [ ] **Step 2: Run tests to verify failure** with `pytest tests/test_round_phase2_analog_dispatch.py -q`; expected failure is missing mode dispatch.
- [ ] **Step 3: Add executor initialization** that reads `load_lif_mode_config`, loads the local LUT only for `analog_lut`, and leaves the digital path untouched.
- [ ] **Step 4: Add the narrow analog branch** at the current LIF execution point; preserve invocation-order checks, boundary commits, static evaluation, and round aggregation.
- [ ] **Step 5: Add additive report metadata** for LUT profile, query counts, interpolation, checksum verification, clamp counts, and LUT energy/latency/power.
- [ ] **Step 6: Run dispatch and regression tests** with `pytest tests/test_round_phase2_analog_dispatch.py tests/test_round_snn_executor.py -q`; expected PASS.
- [ ] **Step 7: Commit** with `git add MNSIM/Scheduling/phase2/executor.py MNSIM/Scheduling/phase2/types.py MNSIM/Scheduling/phase2/report.py tests/test_round_phase2_analog_dispatch.py tests/test_round_snn_executor.py && git commit -m "feat: dispatch phase2 lif through local analog lut"`.

### Task 5: End-to-end verification and documentation

**Files:**
- Modify: `docs/specs/2026-09-22-local-analog-lif-lut-design.md` only if implementation details require clarification
- Test: existing `tests/test_round_phase2_*.py`, `tests/test_round_snn_executor.py`

- [ ] **Step 1: Run the complete local test suite** with `pytest -q` from `round_orchestrated_simulator/MNSIM-2.0`.
- [ ] **Step 2: Run a targeted real-artifact smoke test** loading `08_lut` via the local loader and querying one frontend/LIF point.
- [ ] **Step 3: Verify no legacy LUT imports** with `rg -n "snn_lut_runtime|snn_lut_adapter" MNSIM`; expected no matches in new local modules.
- [ ] **Step 4: Inspect git diff and status** to ensure unrelated pre-existing worktree changes remain untouched.
- [ ] **Step 5: Commit any required test/doc corrections** with a focused commit message.

