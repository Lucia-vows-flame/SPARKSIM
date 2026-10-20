# ANN Round Physical Write Ledger Design

Date: 2026-08-12

## Scope

This document defines physical weight-programming accounting for the new
round-orchestrated ANN simulator. It applies to ordinary ANN checkpoints and
the original MNSIM hardware-performance backend. It does not use QAT,
accuracy evaluation, BN folding, or the legacy `master + SNN` scheduler.

The implementation is:

```text
MNSIM/Scheduling/phase2/physical_write_ledger.py
```

It is invoked from `execute_round_plan()` before the static read evaluation of
each `WeightedPhysicalRound`.

## Required Round Semantics

Every weighted physical round has one write phase followed by one read phase:

```text
round r: materialize target device state -> program transition -> read resident state
```

There is no special "first load" path. At `R0`, every previously unseen
physical slot is interpreted as an erased all-zero device-code matrix and uses
the same transition primitive as all later rounds.

The ledger retains device state across rounds. Let `S[r - 1]` be the state
after the previous weighted round and `T[r]` the state mapped for the current
round. Each physical slot transitions from `S[r - 1]` to `T[r]`. A slot that
was occupied in the preceding round but is not mapped in the current round has
`T[r] = 0`, so it is explicitly cleared. This policy is independent of G8,
the number of xbars, and the number of tiles.

The serialized inference totals are:

```text
latency_total = sum_r(read_latency[r] + write_latency[r])
energy_total  = sum_r(read_energy[r]  + write_energy[r])
```

## Physical Target Construction

The Phase 2 TCG placement exposes exact fragment-to-slot metadata. A physical
slot key is:

```text
(tile_id, pe_id, group_id, polarity)
```

The real checkpoint device codes come from `context.net.get_weights()`, keyed
by the canonical Phase 1 layer id. MNSIM quantizers expose each weight plane
as `[output, flattened_input]`; the ledger transposes it to the physical xbar
orientation `[input_row, output_column]` and zero-pads incomplete arrays.

For differential mappings, each fragment addresses its positive or negative
plane independently. Weight-plane and input-block indices come from the
fragment metadata, not from a hardware-size special case.

## Cell Transition Model

All device parameters are read from the active `SimConfig.ini`:

```text
Device_Resistance
Write_Voltage
pulse_latency
SET_pulse
RESET_pulse
Xbar_Size
OU_Size
```

For an original device code `o` and target code `t`, the pulse counts follow
the existing MNSIM physical-write model:

```text
if o <= t:
    reset_pulses = 0
    set_pulses = SET_pulse[t] - SET_pulse[o]
else:
    reset_pulses = RESET_pulse[o]
    set_pulses = SET_pulse[t]
```

RESET and SET latency are their pulse counts multiplied by their corresponding
`pulse_latency`. Cell energy is voltage-squared over the geometric-mean
resistance used by the existing physical-write implementation, multiplied by
the corresponding pulse time. The ledger constructs a complete code-to-code
latency and energy table once per distinct SimConfig and records its digest in
each round result.

## Parallelism Aggregation

The aggregation order matches the existing MNSIM physical-write semantics:

```text
within one OU row: maximum cell latency
within one OU:     sum of OU-row latencies
within one xbar:   sum of OU latencies, sum of cell energies
within one PE:     maximum xbar latency, sum of xbar energies
within one tile:   maximum PE latency, sum of PE energies
within one round:  maximum tile latency, sum of tile energies
across rounds:     sum of round latency and energy
```

Thus the reported write latency is not the sum of all xbar write times: it
captures the configured parallel program capability of OU, PE, tile, and the
fabric.

## Accounting Boundary

The report exposes the current scope in `accounting_scope`:

```text
weight programming: included in latency and energy
DDR boundary transfer: excluded from latency and energy
static leakage: excluded for the G8 performance profile
```

Cross-round activation state is still created and consumed by the boundary
manager. It represents DDR for this experiment, but no DDR transfer cost is
added yet. The G8 profile uses `Buffer_IncludeLeakageEnergy = 0`; other static
or idle leakage is also not included. This is intentionally distinct from
dynamic read buffer-access energy, which remains modeled by the MNSIM backend.

## G8 Reference

The formal G8 profile has:

```text
6 x 3 tiles x 4 x 4 PEs/tile x 2 groups/PE x 2 polarity = 1152 physical xbars
```

The ordinary ResNet18 checkpoint maps to two physical rounds. The dynamic
reference artifacts are in:

```text
results/g8_literature_peripheral_1152xbar_cpu_write_dynamic_20260812/
```

```text
R0: 1030 target xbars, 1030 transitions, write 4.610700 ms / 0.630628 nJ
R1:  728 target xbars, 1030 transitions, 302 released slots cleared,
    write 28.242960 ms / 7.462345 nJ

read total:   35.108466 ms / 1.028778 uJ
write total:  32.853660 ms / 8.092973 nJ
total:        67.962126 ms / 1.036871 uJ
TOPS:         0.004526583
TOPS/W:       296.696818
```

## Verification

The focused regression suite covers:

```text
1. R0 uses the same state-transition path as later rounds.
2. Released slots are cleared before the next round read phase.
3. Different checkpoint device codes produce different write costs.
4. Conv input-block selection and output/input transposition are correct.
5. Executor order is write then read.
6. Reports preserve read, write, total, and accounting-scope ledgers.
```

The relevant tests are `test_ann_round_physical_write_ledger.py`,
`test_round_phase2_executor.py`, `test_round_phase2_report.py`, and the
round-orchestrated runner and CLI tests.
