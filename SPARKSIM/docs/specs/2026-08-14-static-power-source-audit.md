# Static Power Source Audit

Date: 2026-08-14

Status: source qualification for the proposed static-power ledger

## 1. Qualification Rule

A dynamic operation energy, active power, or access latency is not a static
power source. A source is accepted only when all of the following are known:

- physical block and circuit topology;
- process or technology model;
- supply voltage or tool-defined voltage;
- temperature;
- power state and bias condition;
- instance capacity, width, port count, and bank organization when applicable;
- whether the number is measured, simulated, synthesized, or analytically
  estimated;
- whether an existing operation-energy number is total supply energy during
  the operation or incremental switching energy above the idle baseline;
- a reproducible source artifact or publication identifier.

Values from different power states or incompatible circuit implementations
may be retained as cross-checks, but may not be inserted into one nominal
profile.

This audit distinguishes two different claims:

- a `physically_grounded_architecture_estimate` may use CACTI, PTM, or another
  published compact-model tool with every input and limitation preserved;
- an `implementation_specific_static_power` claim requires a foundry memory
  compiler report, transistor-level PDK simulation, post-layout power
  analysis, or measured silicon for the actual instantiated circuit.

The first is useful for architecture exploration. It is not evidence that a
fabricated 40 nm macro has exactly the reported power. No public paper number
is promoted from the first category into the second by technology scaling.

Static-power provenance and accounting-boundary provenance are both required.
If an existing `energy/operation` value already integrates total supply energy
during the active window, a static ledger may charge standby power only outside
that window. Charging standby power over the whole timeline is valid only when
the operation-energy source is explicitly incremental above the standby
baseline. An unknown boundary remains unresolved; it is not silently treated
as incremental energy.

## 2. Existing SRAM Number Is Not Valid for V1

The current configuration uses:

```text
Buffer_LeakagePowerPerBit = 1.372e-14 W/bit
                          = 13.72 nW/Mbit
```

Its cited source is:

> M. Yabuuchi et al., "A 65 nm 1.0 V 1.84 ns
> Silicon-on-Thin-Box (SOTB) embedded SRAM with 13.72 nW/Mbit standby
> power for smart IoT," Symposium on VLSI Technology, 2017.
> DOI: 10.23919/VLSIT.2017.7998145.

This value is rejected as the nominal v1 parameter for three independent
reasons:

1. It is a 65 nm SOTB SRAM, while the current buffer configuration says
   40 nm.
2. The `13.72 nW/Mbit` point is obtained with reverse back-bias control in
   sleep mode. The paper states that it is approximately `1/1000` of normal
   standby power.
3. The v1 policy is `no_implicit_power_gating` and defines no SOTB back-bias
   generator, back-bias controller, transition latency, or transition energy.

Using the sleep value while modeling an always-retained SRAM is therefore a
power-state error, not merely a technology-scaling approximation. The same
paper implies a normal-standby order of magnitude near `13.72 uW/Mbit`, but
that inferred 65 nm SOTB value is retained only as a cross-check and is not
used as the formal 40 nm parameter.

## 3. Qualified Stage-A Baseline: CACTI Macro Characterization

Stage A may use a reproducible CACTI macro characterization instead of a
single leakage-per-bit coefficient. This is the best currently reproducible
baseline source, but it remains an architecture estimate. A formal
implementation-specific profile must replace it with a memory-compiler or
post-layout result for the selected process and macro organization.

Source artifact:

```text
tool: CACTI 7.0
repository: https://github.com/HewlettPackard/cacti
revision: 1ffd8dfb10303d306ecd8d215320aea07651e878
revision date: 2017-06-19
```

The pinned source:

- accepts capacity, output width, port count, bank count, technology, and
  operating temperature;
- models scratch RAM separately from tagged cache;
- supports 40 nm by interpolation between its characterized technology
  tables;
- exposes HP, LSTP, and LOP device flavors;
- reports subthreshold leakage and gate leakage separately;
- supports explicit array/interconnect power-gating controls.

The formal CACTI input contract for every physical SRAM macro is:

```text
cache type: ram
capacity: materialized capacity of the physical instance class
block size: 16 bytes
output/input bus width: 128 bits
read-write ports: 1
exclusive read/write ports: 0
single-ended read ports: 0
technology: 0.040 um
temperature: 300 K nominal, plus declared corner runs
ECC: false unless ECC is explicitly instantiated by the architecture
array/WL/CL/interconnect power gating: false
bitline floating: false
optimization mode: NONE, never the sample-file ED or ED^2 default
```

CACTI derives voltage from the selected device flavor and interpolated
technology tables. For the reproduced 40 nm LSTP candidate, it interpolates
the 45 nm `1.1 V` and 32 nm `1.0 V` table entries to `1.061538 V`. Every
generated macro record must expose the resulting voltage; `40 nm` alone is not
a complete PVT identity.

The transistor flavor and bank organization are selected by a physical
constraint, not by the desired resource-sweep result. The generator must also
pin CACTI's internal organization-selection policy; a result produced by an
unrecorded default optimization target is not a source artifact:

1. Enumerate supported HP, LSTP, and LOP flavors and legal power-of-two bank
   counts.
2. Reject candidates whose capacity is below the required live allocation.
3. Reject candidates whose CACTI access time exceeds the configured SRAM
   access-time constraint.
4. Select the surviving candidate with minimum
   `subthreshold leakage + gate leakage`; break ties by area and then bank
   count.
5. Preserve every candidate input and output so the selection is auditable.

CACTI's built-in leakage objective optimizes its `readOp.leakage` field and
does not include the separately reported gate-leakage field. Therefore a
generator claiming minimum *total* static power must either enumerate legal
forced organizations and rank
`readOp.leakage + readOp.gate_leakage` externally, or patch and pin the CACTI
objective. It must not claim a global total-static optimum from the stock
leakage objective alone.

CACTI owns the macro static power:

```text
P_sram_static
  = P_cacti_subthreshold_leakage
  + P_cacti_gate_leakage
```

It must not be converted back into one universal per-bit coefficient. Decoder,
wordline, sense-amplifier, and other peripheral leakage do not scale linearly
with capacity.

### 3.1 Reproduction check

The pinned source was built locally and run for a 2 KiB, 128-bit, one-port,
one-bank, 40 nm, 300 K scratch RAM with power gating disabled.

For the LSTP candidate, CACTI reported:

```text
access time:                         0.764430 ns
subthreshold leakage power:         6.32347e-05 mW
gate leakage power:                 5.45753e-05 mW
total static power used by ledger:  1.17810e-07 W
```

Reproduction artifact hashes are:

```text
input configuration SHA-256:
  65107b76b6e3628f6f9a5eea4ce798688b1845bdad3fec969eb4575fdf22808b
raw output SHA-256:
  f2fb220846672e40150c2060a0a95908acc9f9ade758dd4c2ab1f0a5b2d58192
locally built binary SHA-256:
  6e79c30bd1edf5fda17a8e4f49edf04b97322a17000216bd394300bc3fd06040
```

The exact input is preserved under
`docs/source_artifacts/static_power/cacti_7_0_2kib_reproduction/`. The raw
output is omitted from the source-only release; its hash above identifies the
locally retained run and can be reproduced with the pinned CACTI source.

For the otherwise identical HP candidate, CACTI reported:

```text
access time:                         0.195381 ns
subthreshold leakage power:         0.932086 mW
gate leakage power:                 0.115212 mW
total static power used by ledger:  1.047298e-03 W
```

This four-order-of-magnitude spread demonstrates why transistor flavor cannot
be omitted or chosen by fitting the final energy trend. The 2 KiB result is a
tool-chain reproduction check, not the final accelerator parameter. Final
values are generated only after the simulator derives each materialized
capacity.

The reproduced input was derived from CACTI's distributed sample configuration
and still had `Add ECC = true` and `Optimize ED^2`. Consequently, its numeric
result proves only that the pinned source, build, input, output, and hashes are
reproducible. It is **not** a production macro record and must not be copied
into the formal static profile. Production generation must set the explicit
contract above, preserve the effective ECC flag and optimization policy in the
output record, and rerun every materialized capacity.

### 3.2 Clean 2 KiB candidate

A second input removes the sample-file contamination while preserving the same
2 KiB, 128-bit, one-port, one-bank, 40 nm, 300 K LSTP identity:

```text
Add ECC: false
Optimize ED/ED^2: NONE
UCA objective weights: 0:0:100:0:0
power gating and bitline floating: false
```

CACTI reports:

```text
access time:                         0.718940 ns
cycle time:                          0.842917 ns
subthreshold leakage power:         5.13732e-05 mW
gate leakage power:                 4.34823e-05 mW
total static power used by ledger:  9.48555e-08 W
best Ndwl/Ndbl/Nspd/Ndcm/Ndsam1/Ndsam2: 2/2/4/8/1/1
```

Artifact hashes are:

```text
input configuration SHA-256:
  1a1aa65f323f09f34b92f673bd1237ce52f18cbd9119500f81e620881642963b
raw output SHA-256:
  4e1e1867d8d9748a8eeac35cabe5c6ecc8532f1000e8ae9829d138eefb649165
CSV output SHA-256:
  08fd0eb80049a9671008b509c943de2424c3b1a5f00ab5a37b631248d53bf45b
```

The exact input is preserved under
`docs/source_artifacts/static_power/cacti_7_0_2kib_clean_candidate/`. The raw
output and CSV are omitted from the source-only release; their hashes above
identify the locally retained run and can be reproduced with the pinned CACTI
source.

This clean artifact uses `CONCISE` output. CACTI's `DETAILED` path prints
uninitialized near-zero component fields in this prerelease and is not
hash-stable. Two serial runs from the pinned CACTI source/build directory
produced byte-identical standard output and CSV output. The CSV standby field
is the sum of CACTI's subthreshold and gate-leakage fields. The build directory
is required because this version resolves supporting data relative to its
runtime working directory. Parallel reruns with distinct input/output names
also reproduced the same bytes.

The prerelease standard output also appends external-DDR IO and MemCad example
results. They are outside the scratch-SRAM macro boundary and are not ledger
inputs. Only the `Uniform Cache Access SRAM Model` result and its matching CSV
row are qualified by this artifact.

This is a valid, fully specified Stage-A candidate for an exactly matching
2 KiB instance. It is still not a memory-compiler result. It is also not proof
of the minimum-total-static CACTI organization: the stock leakage objective
optimizes subthreshold leakage and omits the separate gate-leakage field.
Production generation must enumerate legal forced organizations and rank
`subthreshold + gate leakage` externally subject to the access-time constraint.

### 3.3 Silicon cross-checks, not nominal coefficients

Two measured 40 nm SRAM publications are suitable for sanity checking the
chosen CACTI operating region:

- C.-C. Wang et al., "A 1.0 fJ energy/bit single-ended 1 kb 6T SRAM
  implemented using 40 nm CMOS process," IET Circuits, Devices & Systems,
  2023, DOI: 10.1049/cds2.12141.
- X. Fan et al., "Towards SRAM leakage power minimization by aggressive
  standby voltage scaling - Experiments on 40nm test chips," IEEE DFT 2017,
  DOI: 10.1109/DFT.2017.8244431.

Both use specialized low-power/retention techniques and fixed test macros.
They do not justify linearly scaling one measured number to every PE, tile,
and global-buffer capacity. Their role is validation of magnitude and
voltage/retention behavior.

The only source that can close an implementation-specific SRAM row is the
report for the actual compiled macro (or post-layout characterization of that
macro) at declared PVT. CACTI and the two publications remain independent
cross-checks when that report becomes available.

### 3.4 Additional measured and simulated SRAM cross-checks

Two additional publications reinforce the distinction between ordinary
standby and a deliberately engineered retention state:

- Enomoto and Kobayashi measured a 90 nm, 2-kbit 6T SRAM at `VDD = 1.0 V`
  and `25 C`. Across five chips, conventional standby was `10.12 uW`, while
  a version with self-controllable voltage-level circuits measured
  `0.957 uW`. The measurement includes clock-gated peripherals. The lower
  value depends on a different retention topology that lowers internal supply
  voltages; it is not a coefficient for the current 40 nm macro.
- Lee and Sim reported transistor-level simulation of an 180 nm, 8K-bit SRAM
  with a `0.4 V` standby cell supply and `-0.18 V` standby wordline. Its bit
  cells consume `23.85 pW`, its negative-bias generator consumes `27 pW`, and
  SRAM plus memory controller is reported as `0.11 nW` in standby. This is not
  measured silicon, and the power relies on bias and supply-switching circuits
  absent from the current policy.

Neither result is linearly scaled by bit count or technology. Their qualified
use is cross-checking the selected operating region and verifying that every
retention state includes its bias generator, controller, and state-specific
voltages. Exact document identities and hashes are preserved under
`docs/source_artifacts/static_power/paper_power_candidate_audit/README.md`.

## 4. ADC and DAC Sources Are Dynamic-Only

The current ADC source is:

> H. Hong et al., "Memristor-based adaptive analog-to-digital conversion for
> efficient and accurate compute-in-memory," Nature Communications, 2025.
> DOI: 10.1038/s41467-025-65233-w.

The reported `12.58 fJ/conversion` is an operation-energy estimate. The ADC
core is based on a prior 16 nm analog-CAM result, while its decoder is
synthesized with a 28 nm library. The paper explicitly describes this as a
mixed-technology estimate. It does not provide a compatible idle/standby
leakage value for the instantiated ADC.

This conclusion was checked against both the 11-page published article and
the 21-page Supplementary Information, not inferred from the abstract. The
supplement provides the Q-cell count, decoder truth table, robustness results,
and a UMC 28 nm layout used to validate area. It does not report an idle supply
current, standby supply power, leakage corner, or an ADC disable-state bias
vector. The earlier 16 nm analog-CAM paper states that nonvolatile memristors
need no static power *to store the range table*. That statement applies to data
retention in the passive memory elements. The same paper discusses
subthreshold leakage in pull-down transistors during search, so its storage
claim cannot be promoted to a characterized zero for the complete powered
Q-cell or ReADC.

The current DAC source is:

> S. Uenohara and K. Aihara, "Time-Domain Digital-to-Analog Converter for
> Spiking Neural Network Hardware," Circuits, Systems, and Signal Processing,
> 2021. DOI: 10.1007/s00034-020-01597-2.

The reported `27 fJ/conversion` is a Spectre operation-energy result for an
8-bit TDAC designed in TSMC 40 nm CMOS at `VDD = 0.7 V`. It does not report
the off/idle leakage needed by the static ledger.

The full 19-page paper was also checked. Its transistor-level design disables
a direct wasteful-current path during reset/inactivity, but it does not report
the residual supply current after that control action. MOS off-current,
always-biased nodes, delay logic, and the switched-current-source state remain
uncharacterized. Therefore, "the intended current path is disabled" is not
evidence of zero static power.

The exact public files and SHA-256 digests used for this full-text audit are
recorded under
`docs/source_artifacts/static_power/literature_fulltext_audit/README.md`.

Consequences:

- neither conversion-energy number may be divided by conversion latency and
  reused as standby power;
- neither block may silently receive zero static power;
- the physical inventory is nevertheless closed from SimConfig: the standard
  configuration contains 18,432 ADC instances and 73,728 DAC instances, owned
  per physical crossbar group rather than per differential polarity plane;
- ADC and DAC remain `unmodeled` in the static domain until the actual
  circuits are characterized in a declared PDK/PVT or a publication provides
  the exact compatible power state.
- the future characterization must report both idle supply power and the
  active-energy boundary. If the existing conversion energy is total active
  supply energy, idle static power is integrated only over non-conversion
  windows; no existing conversion-energy model is rewritten by this work.

This is the current blocker for `static_accounting_complete=true`.

### 4.1 Alternative ADC and complete-chip measurement evidence

Pelzers et al. measured `52 pW` standby power at room temperature for a 65 nm
temperature sensor containing a duty-cycled bridge, 9-bit asynchronous SAR
ADC, automatic power-gating logic, and retained memory. The design uses
`VDD = 0.6 V` and a secondary `VDDH = 1.0 V` supply. The value is a complete-
sensor measurement, not an ADC-core measurement, and it relies on disconnecting
the ADC's local supply after conversion. It is therefore an
`alternative_implementation` source, not a proxy for the Hong ReADC. The ReADC
row remains `unmodeled`.

Le Gallo et al.'s 14 nm PCM compute-in-memory chip provides a useful
measurement protocol: standby static and computation dynamic power are
measured separately, and total energy is formed from both contributions. The
public result does not expose a separable standby number for a compatible ADC,
DAC, SRAM, or crossbar-peripheral instance. It is retained as
`measurement_method_only`, not as a ledger parameter.

The audited files, hashes, exact circuit boundaries, and applicability
decisions are recorded under
`docs/source_artifacts/static_power/paper_power_candidate_audit/README.md`.

## 5. Digital and Crossbar-Peripheral Source

DNN+NeuroSim V2.0 is accepted only as a component-level circuit architecture
estimate for the CMOS primitives whose leakage equations are actually
implemented. It is not accepted as a complete PE, tile, ADC, or crossbar-
periphery source:

```text
repository: https://github.com/neurosim/DNN_NeuroSim_V2.0
audited revision: 05b24325f3bd885a8ed66cf7580a07c3c63db3d2
primary reference: X. Peng et al., IEEE TCAD, 2020
DOI: 10.1109/TCAD.2020.3043731
device basis: PTM HP/LP models
audited mature node nearest the design: 45 nm, VDD = 1.0 V
```

For implemented logic, `CalculateGateLeakage()` multiplies PTM off-current by
transistor width and applies average NAND/NOR input-pattern factors. The
component models then multiply the current by `VDD` and the modeled gate or
bit count. Adders, DFFs, comparators, row/wordline decoders, decoder drivers,
prechargers, digital sense encoders, shift-add logic, and repeaters contain
nonzero leakage equations. These are useful architecture estimates only when
the exact modeled topology and instance count match the simulator inventory.

The audited source also has material omissions:

- `MultilevelSenseAmp` and `CurrentSenseAmp` set leakage to zero while still
  calculating read energy;
- `Mux` and `NewMux` set leakage to zero, omitting off-state transmission-gate
  leakage;
- `SwitchMatrix` and `NewSwitchMatrix` include DFF leakage but explicitly
  assume the transmission gates have no leakage;
- the RRAM/FeFET `SubArray` leakage sum contains the selected peripheral
  objects but no leakage term for the 1T1R access transistor array;
- a digital encoder's nonzero leakage does not cover the omitted analog sense
  core;
- the model uses average gate leakage rather than a declared retained input
  vector or state distribution for the actual round-orchestrated circuit.

Consequently, a NeuroSim total is a lower-bound-like partial sum for this
purpose. It may populate separately named covered primitives after an explicit
topology mapping, but it must leave the omitted primitives `unmodeled`. It may
not close either the `crossbar_cmos_periphery` or `pe_tile_digital_logic` row.

NeuroSim supports 45 nm but not 40 nm in this version. The current hardware
configuration is itself mixed: 40 nm SRAM and TDAC, 45 nm digital settings, a
130 nm crossbar-access-transistor setting, and a mixed 16/28 nm ReADC estimate.
Those rows do not describe one PVT-compatible chip. NeuroSim output may enter
a common nominal profile only after one coherent target implementation is
declared and every omitted circuit is independently characterized at that
target PVT.

The source that can close implementation-specific digital rows is not another
published `power/op` number. It is the target standard-cell Liberty database
with state-dependent leakage at the declared PVT, the synthesized gate-level
netlist for the actual Digital-LIF/accumulator/tile datapath, the synthesis and
static-power tool reports, and the retained-state input-vector or probability
assumptions. No compatible Liberty/PDK asset is present in the audited
workspace. A generic Nangate/OpenROAD library would be useful for a separate
architecture estimate, but cannot be labeled as the current 40 nm foundry
implementation.

Similarly, implementation-specific 1T1R peripheral closure requires the
actual access-transistor, decoder, switch, bias, and sense/readout netlists
simulated with the chosen PDK at explicit `retention`, `idle`, and `read_ready`
control vectors. A paper-level NeuroSim aggregate cannot replace those decks.

## 6. Local HSPICE Evidence Does Not Match the Current Static Inventory

The project contains archived HSPICE decks for a transistor-only signed
current-differencing front end. They are stronger evidence than a normalized
paper proxy for that specific circuit because they preserve a PDK model,
schematic, corner, voltage, temperature, and raw output. They are nevertheless
rejected for the current static ledger:

- the model is SMIC `l009ll_v0p3`, i.e. a 90 nm low-leakage process, not the
  declared 40 nm SRAM/TDAC point;
- the decks use the TT corner, `VDD = 1.2 V`, and `25 C`;
- they validate signed-current gain and residual current while input currents
  are applied; they do not measure the complete block's idle supply current;
- the current run enables Digital-LIF and instantiates the Hong ReADC profile,
  so this analog current-differencer is not the ADC or Digital-LIF instance in
  the accounting boundary.

The archived branch-current values therefore cannot be relabeled as ADC,
DAC, or digital static power. A future run that physically instantiates this
front end may add it as a separate block only after a deck measures total
supply power in every ledger state.

## 7. RRAM Cell and NoC Treatment

The passive RRAM storage element may be assigned characterized zero retention
power only in a state where all terminals have zero differential bias. This is
the circuit identity `P = V * I = 0`, not an assumption that the complete
1T1R crossbar has zero leakage. Access-transistor, decoder, switch-matrix, ADC,
DAC, and bias-network leakage remain separate included blocks.

The current run has `NoC_enable = 0`. No routers or links are instantiated, so
NoC static power is outside this run's physical inventory rather than an
unmodeled zero. If NoC is enabled later, its idle power requires a separately
pinned ORION/DSENT/BookSim-compatible technology characterization.

The global controller remains excluded by user constraint.

## 8. Qualified Source Matrix

| Block | Source decision | Evidence level | Current completeness |
| --- | --- | --- | --- |
| PE input SRAM | CACTI generated baseline; compiler report required for implementation-specific claim | `cacti_architecture_estimate` | exploratory Stage-A modelable |
| Tile state/output SRAM | CACTI generated baseline; compiler report required for implementation-specific claim | `cacti_architecture_estimate` | exploratory Stage-A modelable |
| Global accumulator SRAM | CACTI generated baseline; compiler report required for implementation-specific claim | `cacti_architecture_estimate` | exploratory Stage-A modelable |
| Shared boundary-state SRAM | CACTI generated baseline; compiler report required for implementation-specific claim | `cacti_architecture_estimate` | exploratory Stage-A modelable |
| Passive RRAM cell retention | zero only at proven zero terminal bias | circuit-derived zero | conditional |
| Crossbar CMOS periphery | NeuroSim covers some gates but omits sense-core, transmission-gate, and 1T1R access-device leakage | component architecture estimate | incomplete |
| PE/tile digital logic | NeuroSim may estimate mapped gate primitives; actual retained-state netlist and uncovered logic still required | component architecture estimate | incomplete |
| ADC standby | current ReADC source rejected for static use; 65 nm power-gated SAR sensor is an incompatible alternative implementation | `unmodeled` | incomplete |
| DAC standby | current source rejected for static use | `unmodeled` | incomplete |
| NoC/router | no instance in current run | excluded | complete for current topology |
| Global controller | user-excluded | excluded | outside boundary |

## 9. Immediate Implementation Boundary

An exploratory SRAM-only Stage-A foundation is now implemented as an opt-in
audit path. It uses the clean 2 KiB CACTI candidate as one fixed physical bank
and permits only integer bank replication. It must report both
`static_accounting_complete=false` and
`evidence_scope=physically_grounded_architecture_estimate`. It must remove the
Yabuuchi sleep coefficient from the formal static path. Its result must not be
described as measured, post-layout, implementation-specific, or a complete
static-power estimate. NeuroSim cannot currently turn this partial Stage A into
a complete architecture profile because its omitted circuit leakage remains
unmodeled.

The path is enabled in the current formal configurations and requires
`Buffer_IncludeLeakageEnergy = 0`, which prevents the legacy access-window
leakage term from overlapping the new complete-timeline SRAM term. PE input
SRAM may be scored when its TCG capacity evidence closes. Cross-round
activation/spike and partial-accumulator state use a separate shared on-chip
boundary SRAM pool and therefore do not reserve tile SRAM. Capacity is the
safe-overlap peak of live state, with spikes packed at one bit and other state
stored at its declared data bit width. The pool has no DDR spill path. Its
retention and activation access costs are included; partial-state dynamic
access remains solely in the partial-accumulator cost model to prevent double
counting. DDR is unused rather than assigned zero transfer cost.

Implementation-specific static completeness is not permitted until:

1. the nominal transistor/PVT implementation is declared consistently across
   SRAM, logic, ADC, DAC, and crossbar periphery;
2. the actual SRAM compiler or post-layout reports are available for every
   materialized physical macro class;
3. the actual ReADC idle leakage is characterized or the ADC implementation is
   changed to one with compatible static data;
4. the actual TDAC idle leakage is characterized or the DAC implementation is
   changed to one with compatible static data;
5. crossbar-peripheral and digital idle leakage are generated for the same
   implementation;
6. raw input decks, tool versions, PVT corners, outputs, and digests are
   preserved for every included row.
