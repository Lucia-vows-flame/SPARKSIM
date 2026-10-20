# SNN additional peripheral energy v1

This is an additive model. Existing differential frontend, LIF signal/state,
LUT energy/latency, original MNSIM read energy, weight programming, Buffer,
partial-sum reduction and SRAM retention accounting remain in place. It does
not introduce exclusions or replace the numerical network execution.

## Enable with documented hardware parameters

Add the following to a copy of the hardware SimConfig:

```ini
[SNN Peripheral Energy]
Enabled = 1
Profile = /absolute/path/to/your_peripheral_profile.json
```

Relative profile paths resolve against the SimConfig directory. With no
section, or `Enabled = 0`, the existing energy behavior is preserved.
`configs/peripherals/snn_additional_example.json` documents the complete
versioned profile. Its `example_only` values are illustrative circuit inputs,
not calibrated results and not a formal experiment configuration. No existing
hardware profile is silently changed to use them. Supply/driver/reference
characterization is not presently supplied by this repository.

Every parameter is required, non-negative and in SI units. Read pulse duration
must be positive; supply efficiency must lie in `(0, 1]`. A zero parameter
means an explicit hardware assumption, not an automatic missing-value fallback.
Profile source, status, ID, absolute path and SHA-256 appear in the report.
Before using a calibrated profile, its ownership declarations must be checked
against the actual existing frontend/LIF characterization and original MNSIM
module definitions. The code validates these declarations but cannot infer
circuit overlap from an aggregate LUT rail-energy number.

## Components and physical assumptions

| Component | Calculation |
| --- | --- |
| read_supply | Differential xbar electrical load energy times `(1/eta - 1)`, plus read-gated supply bias |
| column_reference | Enabled physical columns times reference bias energy plus `C_ref V_ref²` per read |
| arow_row_driver | Both rails' `C_row sum(V_row²)`, active-row selection and read-window holding energy |
| arow_startup | Startup events per actual tile/PE/group pair, using explicit startup policy |
| control | Controller energy per executed pair read |
| spike_interconnect | Actual output spikes times direct graph destinations times effective link `C V²` |
| interface | Actual output spike deliveries times interface event energy |

The row model assumes each pulse returns to zero. `C V²` is the energy
drawn by a full charge/discharge cycle, not the stored `1/2 C V²` energy.
Row capacitance is for one physical row of one rail at the chosen geometry;
the profile must account for its actual wiring/load. Reference and supply bias
are read-gated; full-duration standby is not newly modeled. Startup is either
once per sample and physical instance, or every read. The former assumes the
driver remains ready within the sample, including physical slot reuse.

Supply energy includes conversion loss and supply bias only, not another copy
of the xbar load. The original MNSIM xbar model remains its billing owner.
The electrical `V²G` load, including both rails and nonzero code-zero
conductance, is recorded as a diagnostic. This v1 efficiency covers the array
load only; driver/reference energy is independently modeled at its rail.

Interconnect capacitance is an effective capacitance per direct logical graph
destination. It is not a placement/distance-based NoC model and does not
infer convolution patch multicast, router hops or receiver fanout. Terminal
spikes have no fabricated off-chip destination. Explicit export/interface
events would be required to model off-chip output traffic.

No new membrane reference/state-initialization term is inferred from an
aggregate LIF LUT: that could double-charge circuitry inside the unchanged
LIF boundary. A separately characterized, demonstrably external reference
would require an additional component in a future profile version.

## Integration and current support

`analog_crossbar.py` adds read-activity evidence to its existing audit, without
changing returned currents. Numerical blocks use the configured rectangular
xbar dimensions. `analog_peripherals.py` matches every block against actual
fragment input/output ranges, differential polarity, bit plane and physical
tile/PE/group slot. All blocks must be resident in the current round.
This supports closed FC-style matrices and their resident spatial fragments.
Bit-plane scaling, cross-round numerical partial matrices and convolution
window execution cannot be inferred from this existing closed-array contract;
unsupported mappings fail instead of fabricating activity. The event cost
API itself accepts multiple physical reads.

`executor.py` scores a separate `snn_peripheral_energy` step after the existing
analog action. It has no boundary bindings, no state changes, no new latency
or area claim. The added energy belongs to the existing action window.
The original analog LIF step, membrane values, spikes and costs are unchanged.

Digital hardware-only Phase 2 has no numerical row/spike activity. Explicitly
enabling this model there raises `MISSING_SNN_PERIPHERAL_ACTIVITY`; it does not
use a fixed firing rate or activity replay. This addition therefore does not
yet supply full-network activity-based energy for the existing digital v7
hardware-only comparison runs.

Existing Buffer and scheduled partial accumulator models retain ownership of
their actual access/reduction costs; this supplement does not allocate a
second FIFO or adder. Existing internal MNSIM digital costs are also preserved;
this change does not reclassify the whole original ANN-style hardware path as
an ADC/DAC-free physical implementation.

## Report and audits

Each additional step reports included components, formulas, parameters,
profile provenance, startup/read/spike/destination counts and physical event
IDs. Repeated read event IDs are rejected. The inference `energy_ledger`
separates existing components from the seven additional components and checks:

```text
additional step energy = sum(additional step components)
final total energy = sum(all original and additional billed components)
```

Existing unclassified static-backend costs are preserved and labeled, not
silently discarded or attributed to a different circuit. The total includes
weight programming, matching the existing report boundary. Per-step and
per-round values retain timestep/sample labels.

`additional_model_external_energy_read = false` applies only to the new
model, which reads its SimConfig/profile and has no HSPICE result parser or
external-total injection argument. It is deliberately not a global claim:
the existing frontend/LIF still loads its energy LUTs, as explicitly requested.
Its archived-netlist area dependency also remains unchanged.

## Validation

`tests/test_snn_peripheral_energy.py` checks electrical energy conservation,
supply-load non-duplication, silent rows, physical startup lifetime, graph
fanout, multiple rectangular fragments, duplicate-event rejection, missing
parameters, non-overlap declarations, forbidden external-total fields, and
restricted file reads in the new model. An integration test compares enabled
and disabled execution and requires identical original steps, boundary events,
LIF costs/states/spikes and latency, plus exactly the added energy and a closed
ledger, at both 3 and 30 timesteps for sample 0 in the existing single-layer
closed-array fixture. Its synthetic area fixture isolates the portable checkout's missing
archived transistor decks; it is not a device area validation.

The requested two-layer FC/Conv 30-timestep online validation networks are
not present in this checkout. No test here substitutes precomputed per-layer
spike streams for those end-to-end validations or claims they have run.

Validation on this checkout: 77 focused tests passed, covering this model,
analog crossbar/LIF, LUT/config parsing, Phase 2 report/executor, digital SNN
executor/report and scheduled partial accumulator. `git diff --check` passed.
