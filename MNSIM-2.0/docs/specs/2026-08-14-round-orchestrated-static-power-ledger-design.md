# Round-Orchestrated Static Power Ledger Design

Date: 2026-08-14

Status: Stage-A SRAM foundation enabled; cross-round state uses on-chip SRAM

## 0. Implementation Snapshot

The first Stage-A implementation is present and enabled in the current formal
profiles. It adds:

- a versioned CACTI profile parser with PVT, units, source hashes, and
  fail-closed validation;
- exact physical Tile, PE, group, and xbar inventory from SimConfig;
- pre-rounding TCG PE-input and tile-output working-set evidence;
- one fixed 2 KiB CACTI macro as the physical bank, with larger capacities
  formed only by integer bank replication;
- complete-timeline `E = P * t` integration and per-block, per-state, and
  per-round static-energy reporting;
- an explicit double-counting guard that requires
  `Buffer_IncludeLeakageEnergy = 0` when the central ledger is enabled;
- `static_accounting_complete=false`, explicit unmodeled blocks, and no formal
  TOPS/W claim for the partial Stage-A result.

The implementation deliberately fails open only in the reporting sense: a
capacity row without sufficient evidence remains `unmodeled` and is omitted
from the numeric static sum while the run reports incompleteness. It never
silently assigns zero power or infers a capacity from access traffic.

The static audit also materializes the physical ADC and DAC inventories from
`Tile_Num`, `PE_Num`, `Group_Num`, `Xbar_Size`, `Subarray_Size`, `ADC_Num`, and
`DAC_Num`. These rows remain explicitly `unmodeled`: inventory closure does
not imply that an idle/read-ready power parameter exists. In the standard
18-tile, 16-PE/tile, 2-group/PE configuration this gives 18,432 ADC instances
and 73,728 DAC instances. Differential polarity does not multiply these
counts because the configured interfaces are owned per group, matching
`Hardware_Model.PE`.

Cross-round activation/spike tensors and scheduled partial-accumulator state
use one shared on-chip `boundary_state_sram` bank pool. The pool has no DDR
spill path. Its capacity, retention energy, activation access energy, and
activation access latency are included. Partial-state dynamic accesses remain
owned by `ScheduledPartialAccumulatorCostModel`; the central SRAM ledger does
not charge them a second time. Digital-LIF state capacity is added once per
unique layer and physical tile. ADC and DAC standby power remain `unmodeled`
until a compatible circuit source provides idle supply power at a declared
PVT.

## 1. Goal

Add a physically auditable static-power ledger to the round-orchestrated
simulator. Static energy must be derived from the physical hardware inventory,
an explicit power-state policy, sourced power parameters, and exposed physical
time. It must not be inferred from dynamic read energy or tuned to force a
desired resource-sweep trend.

This design deliberately does not own weight-programming latency or
weight-programming energy. A separate implementation may supply the duration
of a programming interval and its dynamic energy. The static ledger consumes
the duration because powered circuits leak during that interval, but it does
not calculate, replace, or validate the write model.

## 2. Scope

### 2.1 In scope

- Static and retention power for hardware blocks inside the declared
  accelerator-core boundary.
- Exact physical instance counts and provisioned capacities derived from the
  run's SimConfig and finalized round plan.
- Explicit power states and interval-based energy integration.
- PVT and source provenance for every nonzero or characterized-zero parameter.
- Separation of dynamic buffer-access energy from buffer leakage energy.
- Per-block, per-state, per-round, and inference-total static-energy reports.
- Independent completeness status for the static-power domain.
- A staged path from the currently available SRAM leakage parameter to a fully
  characterized static profile.

### 2.2 Out of scope

- Weight-programming latency, cell energy, or programming-peripheral energy.
- A global controller power model.
- DDR, host, storage, or board-level power. DDR is not used by the current
  cross-round storage contract.
- Dynamic ADC, DAC, xbar-read, buffer-access, arithmetic, or NoC energy.
- Changing physical-round construction, placement, scheduling, or latency.
- Hard-coding or calibrating parameters to make lower-xbar configurations use
  more energy.
- Claiming silicon-level accuracy without silicon measurements or a
  consistently characterized circuit implementation.

## 3. Current State and Defect

The current simulator does not integrate whole-run static power.

`MNSIM.Hardware_Model.Buffer` exposes a leakage power and, when
`Buffer_IncludeLeakageEnergy=1`, adds this term only during modeled read or
write accesses:

```text
P_buffer_leakage = capacity_bits * leakage_power_per_bit

E_access_leakage = access_count * access_cycle * P_buffer_leakage
```

This is access-window leakage, not retention energy over the physical
inference timeline. No leakage is currently charged between accesses or over
the complete physical-round duration.

The following existing names must not be interpreted as static power:

```text
source_kind = static_eval
static_step_energy_j
mnsim_static_read_energy_j
```

They identify the original MNSIM static evaluation backend. Their energy is
predominantly dynamic read-path energy from the xbar, ADC, DAC, buffers, and
digital peripherals.

The repository also contains an analog LIF comparator-bias term in
`Hardware_Model.Neuron`, but the supported standard round-orchestrated SNN
path uses `DigitalLifCostModel`. The analog LIF term is therefore not part of
the current standard run and must not be reported as generic chip static
power.

## 4. Accuracy Claim

The simulator can guarantee accounting correctness and parameter
traceability. It cannot guarantee exact agreement with a physical chip unless
all modeled blocks correspond to one physical implementation and are
characterized at the same process, voltage, temperature, and power state.

The current literature-peripheral profile explicitly combines sources that
were not measured on one co-fabricated chip. Results using that profile must
therefore be described as a sourced architecture estimate, not a silicon
measurement.

Every static-power report must identify one of these evidence levels:

```text
measured_silicon
post_layout_characterized
spice_characterized
rtl_synthesized
memory_compiler_characterized
cacti_architecture_estimate
published_block_measurement
published_normalized_proxy
unmodeled
```

Only measured silicon, post-layout characterization, PDK SPICE
characterization, RTL/gate-level power analysis tied to the declared library,
and memory-compiler characterization can support an implementation-specific
claim. CACTI is a physically grounded architecture estimate, not a foundry
macro report. A CACTI result or published proxy remains useful, but its
limitation must be visible in the artifact.

## 5. Accounting Boundary

The static ledger uses this named boundary:

```text
round_orchestrated_accelerator_core_static_v1
```

Included block classes are:

- physical PE input SRAM buffers;
- physical tile output/state SRAM buffers;
- one physical global/on-chip staging buffer, if the architecture instantiates
  one;
- RRAM array retention or read-ready static power;
- ADC standby power;
- DAC standby power;
- PE digital-peripheral idle power;
- tile digital-peripheral idle power;
- NoC/router idle power, when the hardware path instantiates a NoC.

Explicit exclusions are:

- global controller;
- host and DDR;
- power delivery and regulator loss;
- cooling;
- dynamic weight-programming energy.

An excluded block is outside the boundary and does not make the static domain
incomplete. An included block without a sourced parameter is `unmodeled` and
does make the static domain incomplete.

The report must not use the unqualified phrase `complete chip energy` for this
boundary.

### 5.1 Compatibility with existing operation-energy ledgers

Static power and existing operation energy must have disjoint ownership. For
every included block that already has an `energy/operation` source, the profile
must declare one of:

```text
incremental_switching_above_idle
total_supply_energy_during_active_window
unknown
```

For `incremental_switching_above_idle`, static power is integrated over the
complete powered timeline. For `total_supply_energy_during_active_window`, the
existing operation energy remains untouched and static power is integrated
only over the complement of the block's non-overlapping active-window union.
`unknown` keeps the combined energy boundary incomplete. It must not default
to either interpretation.

This rule applies to SRAM accesses, ADC conversions, DAC conversions, and
digital operations. The externally owned weight-write energy and latency are
not modified; the static ledger consumes only its exposed duration and avoids
charging a block/state already owned by that write energy boundary.

## 6. Time Ownership

Static energy is wall-clock energy. It must be integrated over exposed
physical time exactly once.

The canonical input is a non-overlapping ordered interval stream:

```python
StaticPowerInterval(
    item_id: str,
    interval_index: int,
    phase_kind: str,
    duration_s: float,
    powered_state_counts: Mapping[str, Mapping[str, int]],
)
```

Supported `phase_kind` values are initially:

```text
weight_programming
read_compute
partial_accumulator
boundary_transfer
```

The write implementation owns the duration of `weight_programming`. The
static ledger treats that duration as opaque. Replacing the write model must
not require changing static-power equations.

Intervals must represent exposed physical time after all relevant overlap and
pipeline rules. The ledger must not sum component-local latencies that overlap
inside a physical round. For a sequential v1 executor, interval durations must
sum to the finalized round latency within floating-point tolerance:

```text
sum(round_intervals.duration_s) == item_result.latency_s
```

Across the inference:

```text
sum(all_intervals.duration_s) == inference_metrics.latency_s
```

If the executor later permits cross-round overlap, it must emit a normalized
non-overlapping interval stream before static scoring.

## 7. Physical Inventory

Static power is based on provisioned hardware, not cumulative scheduled xbar
usage and not the sum of per-round temporary MNSIM objects.

Introduce an immutable inventory:

```python
StaticHardwareInventory(
    physical_tile_count: int,
    physical_pe_count: int,
    physical_group_count: int,
    physical_xbar_count: int,
    blocks: tuple[StaticBlockInventory, ...],
)

StaticBlockInventory(
    block_class: str,
    instance_count: int,
    capacity_bits_per_instance: int | None,
    source: str,
)
```

Topology counts come directly from:

```text
Tile_Num
PE_Num
Group_Num
Xbar_Polarity
```

The identity of each storage block must be explicit:

```text
global_buffer
tile_output_buffer
pe_input_buffer
```

The v1 physical ownership decision is:

```text
one tile_state_output_sram instance per physical tile
  <- round-local tile output/activation regions
  <- Digital-LIF membrane-state regions

one global_accumulator_sram instance per accelerator
  <- direct zero-xbar element-sum/element-multiply regions
  <- global branch-merge staging regions

one boundary_state_sram bank pool per accelerator
  <- cross-round activation and spike tensors
  <- cross-round partial-accumulator state
```

Digital-LIF state does not instantiate dedicated SRAM; its static leakage
belongs to the shared physical `tile_state_output_sram` and is counted once.
Cross-round state does not occupy tile SRAM. It is provisioned in the separate
shared `boundary_state_sram` pool and is inside the current SRAM static-power
boundary.

The global accumulator SRAM remains physically distinct from tile SRAM. This
matches the original MNSIM area, power, and energy models, which add one
`global_buf` instance separately from tile buffers. Round-local `global_buf`
objects are cost-evaluation views of this one physical instance; they are not
additional hardware and must never be accumulated as static inventory.

When a configured buffer size is zero and MNSIM derives a size from mapping,
the static inventory must materialize one physical capacity. PE input SRAM is
uniformly sized in the current MNSIM architecture, so its raw capacity
requirement is the maximum mapping requirement over all prepared physical
rounds:

```text
pe_input_capacity_requirement
  = max_round(tcg PE-input working-set requirement for that round)
```

Tile and global raw requirements are the peak simultaneous live allocation for
each physical instance:

```text
tile_state_output_capacity(tile)
  = max_t(
      non-aliased tile output working bytes at t
      + live activation, partial, and LIF bytes on tile at t
    )

global_accumulator_capacity
  = max_t sum(bytes of all live global accumulator/staging regions)
```

The current SimConfig describes homogeneous PE and tile microarchitectures:
there is one `PE_inBuf_Size` and one `Tile_outBuf_Size`, not a capacity per
physical coordinate. V1 therefore materializes one uniform PE-input capacity
across all physical PEs and one uniform tile-state/output capacity across all
physical tiles:

```text
uniform_pe_input_capacity_requirement
  = max_pe,round(pe input requirement)

uniform_tile_state_output_capacity_requirement
  = max_tile,time(tile state/output requirement)
```

Per-instance requirements remain in the audit, but a lightly used tile does
not receive a smaller physical SRAM. A future heterogeneous architecture must
declare its instance classes explicitly rather than emerge from scheduling.

A producer output that becomes a retained activation in the same SRAM is one
in-place region: the working-output reservation ends when the retained
allocation begins. It must not be counted twice at the commit boundary. A
working-output reservation and unrelated retained state that coexist must be
summed.

The result is rounded up to the selected physical SRAM bank/capacity
granularity. It is not the sum of round-local temporary buffer objects, and it
is not the maximum size of one logical consumer when several consumers are
live at the same time.

For a nonzero SimConfig capacity, the value is treated as physically
provisioned capacity. It must be a legal composition under the selected SRAM
profile and must be no smaller than the rounded requirement; otherwise the run
fails. For a zero/auto SimConfig capacity, materialized capacity is the rounded
requirement. The ledger never silently enlarges an explicit nonzero capacity.

The report must record configured, required, rounded, and materialized values
separately.

### 7.1 Storage liveness and capacity derivation

The current `BoundaryStateManager` records producer, allowed consumers, shape,
region, and bit width, but its `released_states` list is not yet an
authoritative lifetime stream. Static capacity derivation must not interpret
every state left in the manager as live until inference end.

Introduce a deterministic storage-liveness plan before static scoring:

```python
StaticStorageAllocation(
    allocation_id: str,
    physical_buffer_class: str,
    physical_instance_id: str,
    logical_owner: str,
    size_bytes: int,
    birth_interval_index: int,
    release_interval_index: int,
)
```

Lifetime rules are:

1. A cross-round activation or spike tensor is written to the shared boundary
   SRAM after producer commit and released after its last allowed consumer.
2. Cross-round partial-accumulator state is allocated in the same shared pool.
   Repeated updates are in-place and do not create duplicate capacity, even
   when the boundary manager renames a temporary partial-state ID to its
   completed logical-state ID.
3. Capacity at a round boundary uses a conservative safe-overlap snapshot:
   states live before commit and newly produced states coexist before states
   whose last consumer just completed are released.
   Consumer lists are filtered against already completed rounds so an update
   cannot restore a past consumer and extend a dead allocation.
4. Spike tensors use one stored bit per element. Activation and partial state
   use the declared `data_bitwidth`.
5. DDR is unused and there is no spill path. A required capacity larger than
   the materialized boundary pool is an error, not an implicit DDR transfer.
6. Digital-LIF membrane state is one allocation per owning tile and layer
   stream, live across the required timesteps.
7. Same-round direct handoff is a transport operation and creates no
   persistent SRAM allocation.
8. Global element-sum/element-multiply operands and results use allocations in
   `global_accumulator_sram` and are released after their last consumers.

Boundary activation dynamic access uses the SimConfig buffer parameters:

```text
E_access = transferred_bits * Buffer_AccessEnergyPerBit
N_access = ceil(transferred_bits / Buffer_Bitwidth)
T_access = N_access * Buffer_AccessLatency
```

Boundary access latency extends the owning round before static power is
integrated over that round. Partial-state read/write energy and latency remain
owned by `ScheduledPartialAccumulatorCostModel`; the central boundary ledger
records their capacity/liveness but skips their dynamic traffic.

Logical size is derived from the exact tensor region and `data_bitwidth`, not
from read/write byte totals. Access-byte counters measure traffic and may
include repeated accesses; they are not storage capacity.

In particular, these existing fields are evidence about operations or logical
state, not a physical SRAM inventory:

```text
state_bytes_by_tile
buffer_read_bytes
buffer_write_bytes
global_buf_size_kb in one direct-operation cost record
```

They may feed or cross-check a logical allocation, but they may not be summed
or maximized directly to declare provisioned SRAM capacity.

The planner sweeps allocation birth and release points to calculate peak live
bytes for each physical SRAM instance. Runtime boundary events then audit the
planned lifetime. A producer/consumer or size mismatch is a hard error.

The implementation may later make `BoundaryStateManager.released_states`
authoritative, but static capacity must not depend on an empty compatibility
field in v1.

### 7.2 Capacity evidence and audit contract

Every physical SRAM instance has an explicit audit row:

```python
StaticCapacityAudit(
    physical_buffer_class: str,
    physical_instance_id: str,
    configured_capacity_bits: int | None,
    per_round_requirements: tuple[StaticCapacityRequirement, ...],
    peak_required_capacity_bits: int,
    rounded_required_capacity_bits: int,
    materialized_capacity_bits: int,
    rounding_residual_bits: int,
    provisioned_headroom_bits: int,
    rounding_rule_id: str,
    source_records: tuple[str, ...],
)

StaticCapacityRequirement(
    item_id: str,
    interval_index: int,
    working_set_bits: int,
    persistent_live_bits: int,
    aliased_bits: int,
    required_capacity_bits: int,
)
```

The row must close under these identities:

```text
required_capacity_bits
  = working_set_bits + persistent_live_bits - aliased_bits

rounding_residual_bits
  = rounded_required_capacity_bits - peak_required_capacity_bits

provisioned_headroom_bits
  = materialized_capacity_bits - rounded_required_capacity_bits
```

All quantities are nonnegative, and `aliased_bits` must be backed by an
explicit in-place ownership transition. An arbitrary utilization or reuse
factor is forbidden.

The SRAM profile owns the physical rounding rule. It must provide one of:

```text
fixed_bank_capacity_bits + integer bank replication
supported_macro_capacities_bits
reproducible memory-compiler selection rule
```

The rule identifier and source artifact digest are part of the report. The
legacy TCG next-power-of-two-KB rule may be reported as mapping evidence, but
it is not automatically accepted as a physical SRAM macro rule.

The top-level capacity audit must expose at least:

```text
configured_pe_input_capacity_bits_per_pe
configured_tile_state_output_capacity_bits_per_tile
tcg_pe_input_requirement_bits_by_round
tcg_tile_output_working_requirement_bits_by_round
tile_live_state_requirement_bits_by_round_and_tile
materialized_pe_input_capacity_bits_by_pe
materialized_tile_state_output_capacity_bits_by_tile
global_accumulator_peak_required_capacity_bits
materialized_global_accumulator_capacity_bits
rounding_residual_bits_by_instance
provisioned_headroom_bits_by_instance
```

This makes three different facts visible: what the configuration requested,
what execution requires, and what physical SRAM was actually instantiated.

### 7.3 TCG capacity export interface

The prepared round placement already retains `tcg.max_inbuf_size` and
`tcg.max_outbuf_size`, both in KB, but `_tcg_mapping_summary()` does not expose
them. Stage A must add a structured capacity export to
`RoundPlacementPlan.mapping_summary` and the static-evaluation audit:

```python
TCGBufferCapacityEvidence(
    item_id: str,
    pe_input_requirement_bits_per_instance: int,
    tile_output_requirement_bits_per_instance: int,
    source_pe_input_kb: float,
    source_tile_output_kb: float,
    source_rounding_rule: str,
)
```

The first implementation may convert the retained `max_inbuf_size` and
`max_outbuf_size` values exactly to integer bits and label their existing
rounding rule. It must not reconstruct them from access counters. To audit
rounding loss completely, TCG must also preserve the pre-rounding maximum
working-set byte counts at the point where `data_inbuf / PEnum` and
`data_outbuf * 2 / tilenum` are calculated.

`ReindexedTCGPlacementSlice` inherits these values from its resident round
placement. Slices must not each create another capacity row; one prepared
physical round produces one evidence record.

The TCG evidence is a working-set requirement, not automatically the final
tile SRAM capacity. The storage planner combines it with simultaneous
activation, partial-accumulator, and Digital-LIF allocations before bank
rounding and physical materialization.

## 8. Power States

Every included block class uses an explicit state set:

```text
off
retention
idle_on
active
```

State semantics are:

- `off`: power-gated; state need not be preserved;
- `retention`: state is preserved with no useful operation;
- `idle_on`: fully powered and ready, but not switching usefully;
- `active`: performing an operation; dynamic energy is accounted elsewhere,
  while the state's static baseline remains active.

The v1 SRAM implementation uses the characterized `retention` baseline across
the complete exposed inference timeline, including intervals in which an SRAM
contains no currently live logical allocation. Useful read/write switching is
added only by the dynamic ledger; an access does not remove or replace the
static baseline.

The v1 policy is `no_implicit_power_gating`: every provisioned PE input SRAM,
tile state/output SRAM, and the global accumulator SRAM remains powered from
the beginning of the first physical interval through the end of the last
physical interval. This avoids inventing a power-gating controller, wake-up
cost, or state-loss behavior that the current architecture does not define.

No block may be assigned `off` merely because a round does not map work to it.
Power gating requires an explicit policy and, eventually, wake-up latency and
energy. Until such a policy is modeled, instantiated SRAM that preserves
architectural state remains in `retention`.

An RRAM array may have characterized zero retention power because its state is
nonvolatile. This zero is valid only when the access periphery is electrically
disconnected in that state and the source record documents that condition.

## 9. Parameter Profile

Static parameters must live in a versioned structured profile rather than be
scattered across code. SimConfig selects the profile:

```ini
[Static Power]
Static_Power_Enable = 1
Static_Power_Profile = configs/static_power/<profile>.json
Static_Power_Profile_Base = repository_root
Static_Power_Policy = no_implicit_power_gating
```

`Static_Power_Profile_Base` must be either `sim_config_directory` or
`repository_root`. The latter keeps generated SimConfig copies valid when a
launcher places them in a result directory. Absolute profile paths ignore the
base selector. Enabling the model with a missing or invalid profile is a hard
error.

The JSON profile contains:

```json
{
  "schema_version": 1,
  "profile_id": "...",
  "evidence_scope": "physically_grounded_architecture_estimate",
  "implementation_specific": false,
  "boundary_id": "round_orchestrated_accelerator_core_static_v1",
  "pvt": {
    "technology_nm": 40,
    "voltage_v": 1.061538,
    "temperature_c": 26.85,
    "process_corner": "cacti_itrs_lstp_40nm_interpolated"
  },
  "source_generator": {
    "tool": "CACTI 7.0",
    "revision": "1ffd8dfb10303d306ecd8d215320aea07651e878",
    "power_gating": false,
    "ecc": false,
    "optimization_mode": "NONE",
    "uca_objective_weights": "0:0:100:0:0",
    "candidate_only": true
  },
  "blocks": {
    "pe_input_buffer": {
      "model_kind": "macro_static_power_table",
      "capacity_rule": {
        "kind": "cacti_candidate_selection",
        "input_output_width_bits": 128,
        "read_write_ports": 1
      },
      "macro_characterizations": [
        {
          "capacity_bits": 16384,
          "bank_count": 1,
          "device_flavor": "itrs-lstp",
          "access_time_ns": 0.71894,
          "subthreshold_leakage_power_w": 5.13732e-8,
          "gate_leakage_power_w": 4.34823e-8,
          "source_input_sha256": "1a1aa65f323f09f34b92f673bd1237ce52f18cbd9119500f81e620881642963b",
          "source_output_sha256": "4e1e1867d8d9748a8eeac35cabe5c6ecc8532f1000e8ae9829d138eefb649165",
          "source_csv_sha256": "08fd0eb80049a9671008b509c943de2424c3b1a5f00ab5a37b631248d53bf45b"
        }
      ],
      "evidence_level": "cacti_architecture_estimate"
    }
  },
  "excluded_blocks": {
    "global_controller": "outside requested boundary"
  }
}
```

`null`, missing, unknown-PVT, or unsupported parameters do not silently become
zero. They make the corresponding included block `unmodeled`.

For SRAM, an absent or incomplete `capacity_rule`, source artifact, or exact
macro characterization also keeps the block unmodeled. The row above is the
clean, explicitly configured 2 KiB candidate from the source audit; a
production profile must replace its example hashes with generated artifacts
for every materialized capacity. Leakage-per-bit provenance does not by itself
prove which physical capacity or bank organization was instantiated.

The existing `Buffer_LeakagePowerPerBit` value may seed an initial SRAM-only
profile only as a rejected compatibility comparison. It must not enter the
formal v1 total: its cited `13.72 nW/Mbit` value is a 65 nm SOTB reverse
back-bias sleep-mode result, while v1 defines neither SOTB back-bias control
nor implicit power gating.

The qualified exploratory Stage-A source is pinned CACTI macro
characterization using exact materialized capacity, width, ports, bank
organization, device flavor, temperature, and disabled power-gating controls.
CACTI subthreshold and gate leakage are added at macro level; they must not be
collapsed into a universal per-bit coefficient. Source qualification,
inspected publications, the pinned revision, and a reproduction check are
recorded in
`docs/specs/2026-08-14-static-power-source-audit.md`.

CACTI does not establish implementation-specific silicon power. Formal results
require the actual memory-compiler or post-layout record. The clean 2 KiB
candidate explicitly disables ECC and ED/ED2 optimization, but CACTI's stock
leakage objective minimizes subthreshold leakage rather than total
subthreshold-plus-gate leakage. Production macro generation must enumerate and
rank legal forced organizations externally, preserve the selected candidate
set, and regenerate every required capacity.

The current ADC and DAC publications provide operation energy but no
compatible idle leakage. Those blocks remain `unmodeled`; active energy divided
by operation latency is not an acceptable substitute for static power.

## 10. Static Energy Equation

For interval `k`, block class `b`, and state `s`:

```text
E_static(k, b, s)
  = instance_count(k, b, s)
  * static_power_per_instance(b, s)
  * duration(k)
```

For a legacy capacity-scaled SRAM proxy:

```text
static_power_per_instance
  = capacity_bits_per_instance * retention_power_per_bit
```

The accepted CACTI macro profile instead uses:

```text
static_power_per_instance
  = cacti_subthreshold_leakage_power
  + cacti_gate_leakage_power
```

at the exact materialized macro capacity and PVT. No additional SRAM
leakage-per-bit term is added.

The inference total is:

```text
E_static_total = sum_k sum_b sum_s E_static(k, b, s)
```

No utilization factor is applied to leakage unless the parameter source and
power-state policy explicitly define one.

Static energy is additive to existing dynamic energy:

```text
E_modeled_total
  = E_existing_dynamic
  + E_existing_external_or_current_write
  + E_static_total
```

This design does not change the ownership of the write term.

## 11. Buffer Leakage De-duplication

Whole-interval SRAM leakage cannot be added on top of the current access-window
leakage. `Buffer` must expose disjoint components:

```text
dynamic_read_energy
dynamic_write_energy
access_window_leakage_energy
```

When the static ledger is disabled, the legacy combined values remain
available for compatibility.

When the static ledger is enabled:

- for an `incremental_switching_above_idle` access source, dynamic consumers
  use incremental access energy and the central static ledger owns SRAM
  leakage over the complete powered timeline;
- for a `total_supply_energy_during_active_window` access source, the existing
  access energy remains unchanged and the central ledger owns leakage only
  outside the union of SRAM access windows;
- for an `unknown` access-energy boundary, standalone SRAM static energy may
  be reported, but the combined energy boundary remains incomplete;
- the legacy explicit `access_window_leakage_energy` is reported as removed
  overlap whenever the central ledger owns that same term and is not added to
  the total.

The published `1.0 fJ/bit/access` SRAM number currently in SimConfig is not
documented as incremental switching energy above idle. It must therefore not
be assumed compatible with complete-timeline leakage. It requires either an
active-window complement policy or a source that separates its baseline and
switching components.

The following invariant is mandatory:

```text
new_dynamic_buffer_energy + central_buffer_static_energy
```

contains every intended term once and only once.

## 12. Reporting Contract

Add this structure under `inference_metrics.energy_breakdown_j`:

```text
static_energy_j
static_average_power_w
static_energy_by_block_j
static_energy_by_state_j
static_energy_by_round_j
```

Add a top-level static audit object:

```json
{
  "boundary_id": "round_orchestrated_accelerator_core_static_v1",
  "profile_id": "...",
  "profile_digest": "...",
  "evidence_scope": "physically_grounded_architecture_estimate",
  "pvt": {},
  "policy": "no_implicit_power_gating",
  "inventory": {},
  "capacity_audit": {},
  "interval_duration_s": 0.0,
  "duration_by_accounting_owner_s": {},
  "operation_energy_boundary_complete": false,
  "modeled_blocks": [],
  "unmodeled_blocks": [],
  "excluded_blocks": [],
  "static_accounting_complete": false,
  "implementation_specific_complete": false,
  "warnings": []
}
```

Completeness is domain-specific:

```text
static_accounting_complete
write_accounting_complete
overall_energy_accounting_complete
```

`static_accounting_complete=true` requires every included static block to have
a valid parameter, inventory, state assignment, and complete interval
coverage. It does not imply write or total-energy completeness.

`operation_energy_boundary_complete=true` additionally requires every
pre-existing operation-energy term to declare whether it is incremental above
idle or owns its active-window supply energy.
`implementation_specific_complete` requires compiler, PDK/post-layout, or
measured evidence for every included block. These flags are independent in
principle, but the audited NeuroSim revision does not currently cover all
leakage-bearing circuitry in the selected crossbar periphery. It therefore
cannot close the present architecture-estimate profile by itself.

`capacity_audit` must include the TCG evidence records, logical storage
allocations, per-instance peak rows, uniform-class materialization decisions,
bank-rule provenance, and all rounding/headroom residuals defined in Section
7. It is part of static completeness, not optional debug output.

The existing `tops_per_w` remains unavailable until the overall requested
energy boundary and operation-energy ownership are complete. An
implementation-specific TOPS/W claim additionally requires
`implementation_specific_complete=true`. A partial modeled-energy efficiency
may continue to be reported with an explicit partial label.

## 13. Failure Policy

Fail closed for:

- enabled static power with no profile;
- unsupported profile schema;
- negative or non-finite power;
- missing units or PVT identity;
- physical inventory count mismatch;
- interval overlap or incomplete interval coverage;
- a logical buffer with no declared physical owner;
- an included block silently assigned zero power;
- static energy added while legacy access leakage remains in the dynamic sum.
- a block with an unknown operation-energy boundary included in a combined
  complete total;
- active-window and central-static duration ownership that does not close to
  the block's powered duration.

An incomplete but internally valid profile may run only in audit mode. It must
emit `static_accounting_complete=false`, identify every missing block, and keep
formal end-to-end TOPS/W unavailable.

## 14. Validation

### 14.1 Unit tests

1. Parse and validate a versioned static profile.
2. Reject missing provenance, units, PVT fields, and invalid numbers.
3. Derive exact tile, PE, group, and xbar counts from SimConfig.
4. Materialize maximum physical buffer capacity rather than sum round-local
   capacities.
5. Count a shared physical state buffer once.
6. Release an activation after its last declared cross-round consumer.
7. Preserve one partial-state allocation across repeated split-round updates.
8. Allocate no persistent SRAM for same-round direct handoff.
9. Reject traffic-byte counters as a substitute for physical capacity.
10. Export `max_inbuf_size` and `max_outbuf_size` from each prepared TCG round
    exactly once, with units and rounding-rule identity.
11. Preserve and report TCG pre-rounding working-set bytes.
12. Reject an explicit configured capacity smaller than the rounded peak
    requirement.
13. Reject an explicit configured capacity that cannot be composed under the
    selected physical bank rule.
14. Verify homogeneous PE and tile classes use the maximum per-instance
    requirement rather than workload-specific per-instance SRAM sizes.
15. Verify capacity identities, bank-rounding residuals, and provisioned
    headroom close exactly.
16. Verify interval-duration conservation per round and per inference.
17. Verify `E=P*t` for every block/state row.
18. Verify block and state subtotals close exactly to the static total.
19. Verify enabled central SRAM leakage removes legacy access-window leakage.
20. Verify incomplete profiles cannot produce formal TOPS/W.

### 14.2 Integration tests

- Run one small synthetic topology with hand-computed inventory and intervals.
- Run the standard 2-xbar and 1152-xbar configurations with the same static
  profile and compare the full audit tables.
- Run a resource sweep and verify topology, duration, and energy conservation
  for every point.

The resource sweep must not assert a predetermined energy ordering. It must
report the ordering and identify which block/state terms produce it.

### 14.3 Physical validation

For every characterized profile:

- reproduce the source or tool input deck;
- preserve raw CACTI, SPICE, synthesis, or measurement output;
- record tool version and invocation;
- bind the generated profile to those artifacts by digest;
- compare at least one block-level energy calculation against an independent
  hand calculation.
- label CACTI rows as architecture estimates and reserve
  implementation-specific labels for compiler, PDK/post-layout, or silicon
  evidence.

## 15. Delivery Stages

### Stage A: Exploratory SRAM retention ledger

- Add the profile schema and inventory builder.
- Add the pinned CACTI source-generator contract and preserve every input,
  output, revision, and digest used by the profile.
- Model one PE input SRAM per physical PE, one shared tile state/output SRAM
  per physical tile, and one separate global accumulator SRAM per accelerator.
- Model one shared boundary-state SRAM bank pool per accelerator for all
  cross-round activation/spike and partial-accumulator state.
- Export retained TCG input/output capacity evidence and preserve its raw
  pre-rounding working-set byte counts.
- Derive boundary capacity from safe-overlap peak liveness, pack spikes at one
  bit, retain activation/partial state at `data_bitwidth`, and permit no DDR
  spill.
- Derive PE, tile, and global requirements, apply homogeneous PE/tile class
  sizing, and round them using a sourced physical SRAM rule.
- Emit the complete configured/required/rounded/materialized capacity audit.
- Use CACTI macro subthreshold plus gate leakage; reject the existing 65 nm
  SOTB sleep-mode per-bit coefficient from the formal static total.
- Split buffer dynamic energy from access-window leakage.
- Integrate SRAM retention power over the complete physical interval stream.
- Report `static_accounting_complete=false` because other included blocks are
  not yet characterized.
- Report `evidence_scope=physically_grounded_architecture_estimate`; do not
  describe CACTI output as measured or implementation-specific silicon power.

### Stage B: Non-memory static blocks

- Add sourced RRAM retention/read-ready values.
- Add sourced ADC and DAC standby values.
- Add synthesized or characterized PE/tile digital idle power.
- Use NeuroSim only for individually mapped primitives with implemented
  leakage equations; keep its omitted sense-core, transmission-gate, 1T1R
  access-device, and retained-vector effects unmodeled.
- Add NoC/router idle power when NoC is present.
- Preserve each block's independent evidence level and PVT conditions.

### Stage C: Complete static-domain qualification

- Resolve all included block classes.
- Run nominal and corner profiles.
- Close inventory, interval, block, and state residuals.
- Set `static_accounting_complete=true` only after all gates pass.

### Stage D: System-level integration

- Consume the externally owned write duration and energy without changing
  that model.
- Combine independently complete energy domains.
- Enable formal TOPS/W only when the requested overall boundary is complete.

## 16. Acceptance Criteria

The design is implemented only when all of the following hold:

1. Static energy is derived from explicit physical instances and non-overlap
   physical time.
2. No write-model equation or parameter is introduced by this work.
3. No global-controller power is introduced.
4. Every nonzero and characterized-zero static parameter has provenance and
   PVT metadata.
5. Buffer leakage is not double counted.
6. Static totals close against block, state, and round subtotals.
7. Missing block models remain visible and keep static completeness false.
8. Resource-sweep results can explain their trend from auditable terms without
   enforcing that trend in code.
9. Access/state traffic fields are never used as a substitute for provisioned
   capacity.
10. Configured, required, rounded, and materialized SRAM capacities are all
    reported and close under the declared bank rule.
11. Homogeneous physical PE and tile classes remain homogeneous after capacity
    materialization.

## 17. Decisions Required Before Implementation

### 17.1 Resolved v1 decisions

1. Digital-LIF, partial-accumulator, and tile output/activation regions share
   one physical tile state/output SRAM per tile.
2. Direct zero-xbar accumulation and branch-merge staging use one separate
   global accumulator SRAM per accelerator, consistent with original MNSIM
   physical accounting.
3. Logical consumers retain separate dynamic-access attribution while shared
   SRAM leakage is counted once per physical instance.
4. SRAM capacity is peak simultaneous live allocation rounded to physical
   memory granularity.
5. V1 uses `no_implicit_power_gating`; all provisioned SRAM remains in
   retention over the complete exposed inference timeline.
6. Cross-round state uses one independent shared boundary SRAM pool; DDR is
   unused and unmodeled.
7. Activation boundary accesses are owned by the central SRAM ledger, while
   partial-state dynamic accesses are owned only by the partial-accumulator
   model.

### 17.2 Remaining evidence requirements

The following physical inputs still require explicit evidence:

1. The authoritative common PVT point for logic and crossbar periphery; Stage A
   SRAM uses the pinned CACTI candidate-generation contract.
2. Foundry memory-compiler or post-layout static-power reports for every
   materialized SRAM macro class if implementation-specific claims are needed.
3. Compatible idle leakage for the actual ReADC and TDAC implementations.
4. A same-PVT characterization for the remaining non-SRAM included blocks.
5. Any future power-gating policy, including wake-up latency, wake-up energy,
   state retention, and controlling hardware.
6. The target standard-cell Liberty leakage tables, synthesized netlists, and
   retained-state assumptions for Digital-LIF, accumulation, and tile logic.
7. PDK/SPICE decks for 1T1R access devices, row/column switches, bias networks,
   and the selected readout path in every ledger power state.

Until these decisions are resolved, Stage A may provide a transparent partial
static ledger but must not claim full static-power completeness.
