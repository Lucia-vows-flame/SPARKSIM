# SNN Digital-LIF Round Execution Design

Date: 2026-08-12

> Follow-up: `2026-08-15-master-snn-functionality-integration-design.md`
> preserves the ADC + digital-LIF design in this document and adds a second,
> configurable frontend-LUT + LIF-LUT hardware profile. It also requires that
> physical conductance/current evaluation remain isolated from PIM accuracy.
>
> Accuracy/QAT is specified separately by
> `2026-08-15-snn-qat-and-8bit-weight-deployment-design.md`, including the
> 9-bit signed logical-weight and differential 8-bit device-code contract.

## 1. Scope

This document defines the SNN extension of the new round-orchestrated MNSIM
simulator. It reuses the current CNN/ANN `ModelIR -> RoundPlan -> Phase 2`
architecture and adds ordered timestep execution plus a stateful digital LIF
non-weighted operator.

The central unification is:

```text
CNN/ANN: sequence_length T = 1
SNN:     sequence_length T > 1
```

There is no separate SNN round planner. A weighted physical round retains its
current meaning: one resident set of weighted fragments, programmed once
before the round read phase. Every execution item, including a standalone
`ZeroXbarRound`, evaluates its resident graph for all `T` timesteps; only a
`WeightedPhysicalRound` has a weight-programming phase.

This design applies to an ADC/DAC architecture in which xbar outputs are
digitized before partial-sum accumulation and LIF evaluation. It does not use
an analog LIF circuit, analog `Vmem`, HSPICE LIF LUTs, or the legacy
step-closure-aware `master + SNN` scheduler.

## 2. Confirmed Decisions

The following are fixed requirements rather than implementation options.

1. Reuse the current CNN round architecture. CNN is exactly the `T = 1`
   special case of sequence execution.
2. Program the weights of each weighted physical round once. Weight
   programming latency and energy do not scale with `T`.
3. Execute the compute work resident in every execution item for each
   timestep `t = 0 .. T - 1`. A standalone zero-xbar item also runs `T`
   times but performs no weight programming or xbar read.
4. Treat digital LIF as a stateful non-weighted layer.
5. Invoke each `(lif_node_id, timestep)` exactly once per sample and produce
   one binary spike output, including a zero output when the neuron does not
   fire.
6. A LIF invocation may run only after all weighted input contributions for
   that node and timestep are digitally complete and the preceding LIF state
   is available.
7. Weighted layers may span physical rounds. The already implemented digital
   partial-sum mechanism remains the completion authority.
8. Remove the legacy requirement that all input fragments for an SNN output
   block must reside in one physical round. Input closure becomes a runtime
   digital data-dependency barrier, not a Phase 1 packing constraint.
9. Use existing MNSIM digital hardware primitives to model digital LIF cost.
   No transistor-level LIF model is required.
10. Cross-round boundary data remains functionally materialized and consumed,
    while DDR transfer latency and energy remain excluded until a later scope.

## 3. Non-goals

This version does not define:

```text
analog LIF / Cmem / Rmem / comparator hardware
analog Vmem spill, restore, or ADC state conversion
legacy SNN bundle or single-round input-closure rules
per-timestep weight reprogramming
timestep preemption or splitting one node's T-step LIF run across rounds
continuous multi-sample streaming state
adaptive LIF, refractory counters, learnable thresholds, or recurrent models
DDR latency or energy
activity-aware clock gating or sparse xbar-read suppression
accuracy/QAT execution
```

The schema must leave room for future recurrent state, but ordinary
feed-forward LIF SNN is the first implementation target.

## 4. Existing Architecture Reused

The extension builds on these existing contracts:

```text
MNSIM/Scheduling/round_types.py
  ModelIR.sequence_spec
  LayerIR / LocalGraphNode
  WeightedPhysicalRound / ZeroXbarRound

MNSIM/Scheduling/phase1/round_planner.py
  current multi-layer physical-round packing
  current weighted split and continuation construction

MNSIM/Scheduling/phase2/executor.py
  write-before-read round order
  ordered segment execution

MNSIM/Scheduling/phase2/round_local_state.py
  same-round producer/consumer visibility

MNSIM/Scheduling/phase2/boundary.py
  cross-round partial and complete-output state

MNSIM/Scheduling/non_weighted_layers.py
  registered `lif` layer and functional LIF attributes
```

Current `lif` registration as a hook is only a frontend/planner foothold. The
existing Phase 2 hook path assigns zero hardware cost and has no timestep or
persistent-state semantics, so it is not the completed digital LIF backend.

## 5. Sequence Contract

`ModelIR.sequence_spec` remains the public source of sequence semantics.

For ANN/CNN:

```text
SequenceSpec(
    sequence_length=1,
    ordered_steps=False,
    cross_step_dependency=False,
    recurrent_state_ids=[],
)
```

For a feed-forward LIF SNN:

```text
SequenceSpec(
    sequence_length=T,
    ordered_steps=True,
    cross_step_dependency=True,
    recurrent_state_ids=[lif_node_id, ...],
)
```

In SNN v1, every recurrent state id must resolve to a `lif` node. Unknown
recurrent state ids, duplicate ids, `T <= 0`, or cross-step dependency without
ordered steps make the model invalid.

`RoundPlan` must preserve a serialized copy of the sequence contract. Phase 2
must not recover `T` from a network name, checkpoint, global default, or G8
special case.

## 6. Physical-Round Execution

The required item loop is:

```text
for execution_item r in plan order:
    if r is a WeightedPhysicalRound:
        materialize target device state for r
        program r once
        prepare resident weighted placement once

    for timestep t in 0 .. T - 1:
        execute the ordered resident segments at timestep t
        update digital partial sums for timestep t
        execute ready non-weighted nodes at timestep t
        execute each ready digital LIF at timestep t exactly once

    finalize cross-round boundary states for r
```

The loop above defines write frequency, but it does not authorize an invalid
LIF update. If a weighted layer is only partially resident, its contribution
is committed for the current timestep and its downstream LIF remains blocked.
When the final physical fragment arrives in a later round, the completed input
becomes visible and that LIF may execute.

The planner may place multiple complete layers in one round. Same-round
execution may therefore progress as:

```text
timestep t:
  weighted L1 -> lif L1 -> weighted L2 -> lif L2 -> ...
```

No forced single-layer round construction is introduced.

SNN v1 is round-major, not whole-graph timestep-major. An earlier physical
round may therefore finish all `T` contributions before a later round starts.
The strict ordering requirement applies to each recurrent LIF state chain:
`lif(l, t + 1)` may not execute before `lif(l, t)`. Within one execution item,
different timesteps and neuron batches are serialized, and the existing
ordered-segment dependencies remain authoritative.

For a weighted layer split across rounds `r0` and `r1`, the legal execution is:

```text
r0: program once; produce and store partial(l, 0..T-1)
r1: program once
    t=0: restore/complete input(l,0); run lif(l,0)
    t=1: restore/complete input(l,1); run lif(l,1)
    ...
```

For a complete same-round chain, execution remains timestep-local:

```text
r: t=0: weighted L1 -> lif L1 -> weighted L2 -> lif L2
   t=1: weighted L1 -> lif L1 -> weighted L2 -> lif L2
```

Cross-item timestep pipelining and preemption are later optimizations and must
not change data dependencies, state ordering, or exactly-once accounting.

## 7. Invocation Identity and Exactly-once Rule

Phase 2 introduces a runtime invocation identity:

```text
InvocationKey = (sample_id, node_id, timestep)
```

The static `ModelIR` and `RoundPlan` are not expanded into `T` copies. Reports
may serialize per-timestep records, but Phase 1 graph size remains independent
of `T`.

The execution state must distinguish at least:

```text
not_ready
ready
completed
```

For every sample `s`, LIF node `l`, and timestep `t`, the executor must prove:

```text
completed_count(s, l, t) == 1
```

Repeated execution is an error, not an idempotent no-op. Missing execution at
sample finalization is also an error.

## 8. Digital Input Completion

For weighted node `w`, timestep `t`, and physical fragment `f`, define a
digitized contribution:

```text
P[w, t, f]
```

The completed weighted input is:

```text
I[w, t] = sum_f P[w, t, f]
```

The existing fragment-id coverage mechanism remains authoritative. Completion
occurs only when every required fragment id has contributed exactly once.

The logical partial accumulator key must be extended from its current
layer/region identity to include sequence identity:

```text
(sample_id, producer_layer_id, timestep, output_region, ...)
```

Physical storage may batch the complete `T` dimension as one tensor:

```text
[T, output_shape...]
```

but dependency and duplicate-contribution checks remain logically indexed by
`timestep`.

The following are invalid:

```text
executing LIF before weighted input completion
publishing a partial sum as a downstream spike/activation
adding the same fragment twice at one timestep
using a contribution from timestep t for timestep t+1
```

## 9. Digital LIF Semantics

For each neuron and timestep, the functional operation is the configured
discrete LIF update. The current frontend form is:

```text
u[t]     = V[t-1] + (I[t] - V[t-1]) / tau
spike[t] = 1 if u[t] >= v_th else 0
V[t]     = v_reset if spike[t] == 1 else u[t]
```

The implementation must use the network's actual LIF attributes rather than
silently replacing them with global defaults:

```text
tau
v_th
v_reset
reset mode, when later exposed by the frontend
compiled accumulator-domain metadata
```

A zero input does not permit skipping the update. Leakage/decay still changes
state, and the node still produces a zero or one spike for the timestep.

The readiness rule is:

```text
input_complete(l, t)
and (t == 0 or lif_completed(l, t - 1))
and not lif_completed(l, t)
```

The initial state `V[-1]` is initialized once per sample from the compiled
`initial_state_code`, which v1 fixes equal to `reset_code`. It must not leak
from a preceding sample.

## 10. Digital LIF Hardware Cost

Digital LIF is modeled as a non-weighted hardware operator assembled from
existing MNSIM digital modules. It does not require HSPICE or a new transistor
model.

One update is decomposed into configured operations such as:

```text
state read
completed-accumulator input handoff
decay: fixed-coefficient CSD shift/add in v1
input/state addition
threshold compare
reset select / mux
state write
spike write or handoff
```

The first implementation should provide a `DigitalLifCostModel` composition
layer rather than copy formulas from the hardware classes. It reuses:

```text
MNSIM/Hardware_Model/Multiplier.py
MNSIM/Hardware_Model/ShiftReg.py
MNSIM/Hardware_Model/Adder.py
MNSIM/Hardware_Model/JointModule.py
MNSIM/Hardware_Model/Reg.py
MNSIM/Hardware_Model/Buffer.py
```

`Multiplier.py` remains an available library primitive, but the frozen v1
profile below deliberately does not instantiate it.

`JointModule` may provide the initial compare/mux accounting proxy if the
current MNSIM library has no dedicated comparator or mux class. The report
must label this proxy explicitly.

For `N` neurons and `T` timesteps:

```text
lif_update_count = N * T
```

Latency is based on the actual per-Tile ownership batches defined in Section
10.3 and the configured operation pipeline; dynamic energy is based on actual
update count. Area is based on provisioned lanes, not on `N * T` invocations.

### 10.1 Frozen v1 digital-domain profile

ADC is the only analog-to-digital quantization point. After ADC conversion,
bit-slice reconstruction and all xbar/PE/Tile partial-sum operations are exact
integer shifts, sign extensions, and additions at sufficient width. There is
no second activation quantizer between the completed weighted result and LIF.

Every completed digital tensor that may feed LIF must publish one immutable
domain descriptor:

```text
AccumulatorDomain(
    signed_bit_width = B_acc,
    physical_scale   = S_acc,
    domain_id        = stable compile-time identity,
)
```

SNN v1 requires each LIF node to have exactly one completed input tensor with
exactly one such descriptor. A preceding digital Add/merge or non-weighted
operator is allowed only when it has already produced one output descriptor
without requantization. All merged inputs must have identical `domain_id`,
width, and scale; otherwise Phase 1 rejects the graph. LIF itself never aligns
or combines incompatible domains.

The associated LIF inherits that input domain:

```text
B_lif[l] = B_acc[l]
S_lif[l] = S_acc[l]
I_code[t] = completed_accumulator_code[t]
```

The LIF boundary performs no rounding, clipping, saturation, rescaling, or
bit-width reduction. Sign extension and zero-cost wiring alignment are allowed
because they preserve every ADC-derived digital code exactly.

Digital state and constants are represented in that inherited domain:

```text
V_code[t]          signed B_lif-bit state
threshold_code     frontend-supplied/calibrated threshold in S_lif units
reset_code         frontend-supplied reset value in S_lif units
spike[t]           1 bit
decay coefficient  unsigned Q1.15
```

The frontend compiler, not Phase 2, converts LIF constants into the inherited
domain. For each constant `x` in `{v_th, v_reset, initial_state}`:

```text
ideal_code    = x / S_lif
compiled_code = round_nearest_even(ideal_code)
compile_error = compiled_code * S_lif - x
```

The compiler must verify that `compiled_code` fits signed `B_lif` bits and
must record the source value, code, `domain_id`, scale, rounding mode, and
compile error. `reset_code` and `initial_state_code` must use the same domain
as `threshold_code`; v1 fixes `initial_state_code = reset_code`. Phase 2 only
validates and consumes these codes. This offline constant compilation is not
an additional quantization of the xbar signal path. For the current formal
network, `v_th = 0.5` and `v_reset = 0.0`.

The current formal decay parameters are:

```text
tau  = 1.3579523543
beta = 1 / tau = 0.7364028618776407
beta_q15 = 24130 / 32768 = 0.73638916015625
```

The coefficient error is approximately `-1.3700e-5`. This coefficient
quantization is part of the finite-width digital LIF arithmetic, not a second
quantization of xbar output.

### 10.2 Frozen v1 decay datapath

The functional code-domain update is:

```text
delta_code  = I_code[t] - V_code[t-1]
scaled_code = round_even(delta_code * beta_q15 / 2^15)
u_code      = V_code[t-1] + scaled_code
spike[t]    = (u_code >= threshold_code)
next_wide   = reset_code if spike[t] else u_code
V_code[t]   = checked_narrow_B(next_wide)
```

Only `scaled_code` requires rounding because multiplication by Q1.15 creates
fractional coefficient bits. No rounding occurs on ADC output or completed
partial sums. One guard bit is retained through subtraction and state
addition. Before state writeback, `next_wide` must be exactly representable as
a signed `B`-bit value; checked narrowing then preserves the code unchanged.
Failure is an overflow invariant error, never clipping, wraparound, or an
implicit saturation quantizer.

For a layer with inherited width `B = B_lif[l]`:

```text
delta width:        W_delta = B + 1
exact product width W_prod  = W_delta + 16
state-add width:    W_state = B + 1
```

The generic MNSIM `Multiplier` is not used because the formal peripheral
profile has no explicit literature-calibrated multiplier energy. The fixed
coefficient uses the exact CSD identity:

```text
24130 = 2^15 - 2^13 - 2^9 + 2^6 + 2^1
```

The composed cost per lane is:

```text
one W_delta-bit Adder:        I - V
four W_prod-bit Adders:       balanced five-term CSD add/sub tree
one W_prod-bit ShiftReg:      signed arithmetic floor shift by 15
one W_prod-bit JointModule:   guard/sticky/tie/parity decision proxy
one W_delta-bit Adder:        conditional rounding increment
one W_state-bit Adder:        V + scaled_delta
one W_state-bit JointModule:  threshold comparator proxy
one W_state-bit JointModule:  hard-reset MUX proxy
```

The four CSD add/sub units form a three-stage balanced dependency tree. There
is no input-rescale ShiftReg or input-saturation JointModule.

Nearest-even is a signed arithmetic operation, not a property of
`ShiftReg.py`. Let `p = delta_code * beta_q15`. The functional rounder is:

```text
q = arithmetic_floor_shift(p, 15)
r = p - (q << 15)                    # 0 <= r < 2^15, including p < 0
increment = (r > 2^14) or (r == 2^14 and q is odd)
scaled_code = q + increment
```

This definition handles positive and negative half-way ties identically and
rounds them to an even integer. The hardware-cost proxy therefore includes a
discarded-bit decision stage and a conditional increment stage; the shift
alone is insufficient.

Each lane provisions:

```text
one B-bit input register
one B-bit previous-state register
one B-bit next-state register
one 1-bit spike register
```

Each neuron update charges `3 * B + 1` register-bit updates. The ingress
input/state registers update in parallel; next-state/spike registers update in
parallel.

The reference per-batch latency, excluding Tile Buffer access, is derived from
the active SimConfig:

```text
L_lif_batch
  = L_reg_ingress
    + L_delta_adder
    + 3 * L_csd_tree_level
    + L_coefficient_shift
    + L_round_decision
    + L_round_increment
    + L_state_adder
    + L_compare
    + L_reset_mux
    + L_reg_egress
```

With the current `0.481 ns` Adder/JointModule latency and `1 ns` Reg/ShiftReg
latency, `L_lif_batch = 7.329 ns`. Latency is width-independent only because
the current MNSIM module API reports one configured operation latency; the
report must retain the actual per-layer widths and identify every proxy.

Dynamic energy per neuron is width-parametric:

```text
E_lif_neuron(B)
  = E_Adder(W_delta)
    + 4 * E_Adder(W_prod)
    + E_ShiftReg(W_prod)
    + E_JointModule(W_prod)        # rounding decision proxy
    + E_Adder(W_delta)             # conditional increment
    + E_Adder(W_state)
    + 2 * E_JointModule(W_state)
    + E_RegBits(3 * B + 1)
```

Area is likewise width-parametric:

```text
A_lif_lane(B)
  = A_Adder(W_delta)
    + 4 * A_Adder(W_prod)
    + A_ShiftReg(W_prod)
    + A_JointModule(W_prod)        # rounding decision proxy
    + A_Adder(W_delta)             # conditional increment
    + A_Adder(W_state)
    + 2 * A_JointModule(W_state)
    + 3 * A_Reg(B)
    + A_Reg(1)
```

Width-parametric accounting must not pass an arbitrary width to a legacy
lookup table that saturates at its 16-bit entry. `DigitalLifCostModel` uses the
explicit per-bit primitive model when configured. For a legacy `Reg`,
`ShiftReg`, or `JointModule` table, it composes `ceil(W / 16)` physical slices,
using the matching 4/8/12/16-bit entry for the final slice. `Adder` already
scales with requested width. Area and energy must be monotonic for otherwise
identical 8-, 16-, and 32-bit profiles; reports record slice counts and proxy
types.

Because resident LIF layers may inherit different widths, provisioned lane
area on each Tile uses the maximum `B_lif` among LIF nodes that can execute on
that Tile. The incremental Buffer term is accounted by increasing the existing
per-Tile capacity requirement, not by creating a second SRAM ledger.

### 10.3 Frozen v1 parallelism and placement

Digital LIF follows the existing MNSIM non-weighted-unit organization used by
pooling:

```text
LIF lanes per provisioned Tile = 64
digital frequency              = existing Digital_Frequency (1000 MHz)
total provisioned lanes        = Tile_total * 64
```

This is a per-Tile architectural parameter, not a G8 or 1152-xbar special
case. Each LIF node inherits the deterministic Tile ownership of its completed
upstream weighted output. A LIF invocation uses only those owner Tiles; it does
not assume free redistribution to every Tile on the chip.

For one LIF invocation, Phase 1 must preserve the actual neuron ownership
shard `N_k` for every Tile `k`; it may not replace that mapping with an ideal
balanced distribution:

```text
N                  = sum_k N_k
active_tiles       = count_k(N_k > 0)
active_lanes[k]    = min(N_k, 64)
batches[k]         = ceil(N_k / 64)
batches_per_step   = max_k batches[k]
```

The ownership mapping is preserved across timesteps and in the Tile Buffer.
An empty owner Tile performs no LIF work. Any future redistribution for better
balance must create explicit data-movement events and is outside v1. Energy is
charged only for actual neuron updates. Area is charged for all provisioned
`Tile_total * 64` lanes because the units exist even when idle. No additional
always-on LIF leakage term is introduced beyond what the reused MNSIM modules
already model.

Timesteps are serialized, and neuron batches within a timestep are serialized.
The 64 lanes on each Tile execute one batch in parallel. There is no overlap
between Buffer access and LIF compute in v1.

Per-timestep compute cost is therefore:

```text
L_lif_compute[t] = batches_per_step * L_lif_batch
E_lif_compute[t] = sum_k N_k * E_lif_neuron
```

Tile Buffer state access is defined per timestep and per Tile, using one
transaction for that Tile's complete live state shard. It is not charged once
per neuron batch:

```text
state_bits[k]    = N_k * B_lif
state_bytes[k]   = ceil(state_bits[k] / 8)
accesses[k]      = ceil(state_bytes[k] * 8 / Buffer_Bitwidth)
L_read[t]        = max_k BufferReadLatency(state_bytes[k])   for t > 0
L_write[t]       = max_k BufferWriteLatency(state_bytes[k])  for t < T - 1
E_read[t]        = sum_k BufferReadEnergy(state_bytes[k])    for t > 0
E_write[t]       = sum_k BufferWriteEnergy(state_bytes[k])   for t < T - 1
L_lif_total[t]   = L_read[t] + L_lif_compute[t] + L_write[t]
```

The byte and bus-width ceiling operations occur separately for every Tile and
timestep. This preserves the current MNSIM convention that Tile accesses run
in parallel while read, compute, and write phases do not overlap. A later
per-batch SRAM organization would be a different hardware profile.

### 10.4 Frozen v1 state and spike storage

The complete digital `Vmem` tensor is backed by the existing Tile Buffer and
distributed with the neuron ownership above. The LIF lane register holds only
the currently executing neuron batch; 64 lane registers are not treated as
storage for an entire layer.

For each neuron and sample:

```text
t = 0:
  previous state comes from initial_state_code = reset_code; no Buffer read

0 <= t < T - 1:
  updated state is written to Tile Buffer for the next timestep

0 < t <= T - 1:
  previous state is read from Tile Buffer

t = T - 1:
  terminal state is not written after spike generation in single-sample v1
```

Therefore the persistent state traffic is:

```text
state Buffer reads[l]  = N[l] * max(T - 1, 0) * B_lif[l] bits
state Buffer writes[l] = N[l] * max(T - 1, 0) * B_lif[l] bits
```

Every invocation also charges the active lane registers used by the composed
datapath. Spike generation charges one 1-bit register update per neuron per
timestep. A same-round consumer receives the spike directly. A cross-round
consumer creates the existing DDR-domain `spike_sequence` boundary payload;
its transfer latency and energy remain excluded.

Tile Buffer capacity must be increased when necessary so that the distributed
state allocation fits. Buffer area, access energy, access latency, and original
MNSIM access-cycle leakage use the existing Buffer model. This state Buffer is
on-chip and must not be confused with the excluded cross-round DDR store.

All LIF states that remain live between adjacent timesteps of the same
execution item coexist. For Tile `k`, the required additional state capacity
is:

```text
lif_state_bytes[item, k]
  = sum over live lif nodes l owned by k:
      ceil(neurons_owned(l, k) * B_lif[l] / 8)

lif_state_bytes_required[k]
  = max over execution items of lif_state_bytes[item, k]
```

The Tile Buffer sizing input is the existing required capacity plus this
maximum concurrent LIF-state shard. The implementation must not sum states
from execution items that cannot be resident simultaneously, and must not use
only the largest single LIF layer when multiple LIF nodes coexist in one item.

### 10.5 Required configuration keys

The implementation must add the following profile keys, with these v1 values:

```ini
[Digital LIF]
Digital_LIF_Enable = 1
Digital_LIF_InputDomain = inherit_completed_accumulator
Digital_LIF_StateDomain = inherit_completed_accumulator
Digital_LIF_AllowInputRequantization = 0
Digital_LIF_CoefficientBit = 16
Digital_LIF_CoefficientFractionBit = 15
Digital_LIF_LanesPerTile = 64
Digital_LIF_DecayMode = csd_shift_add
Digital_LIF_StateStorage = tile_buffer
Digital_LIF_ResetMode = hard
Digital_LIF_RoundingMode = nearest_even
Digital_LIF_ConstantCompileRoundingMode = nearest_even
Digital_LIF_OverflowMode = error
Digital_LIF_InitialStateMode = reset_code
Digital_LIF_SpikeBit = 1
Digital_LIF_WriteTerminalState = 0
```

The functional `tau`, `v_th`, and `v_reset` remain node attributes supplied by
the frontend. `Digital_LIF_DecayMode = csd_shift_add` requires each supported
`tau` to be compiled into a Q1.15 coefficient and a recorded CSD decomposition.
The current formal coefficient above is the first required implementation.
Unsupported coefficients must fail explicitly rather than use the uncalibrated
generic multiplier fallback.

The v1 formal hardware path accepts exactly:

```text
tau     = 1.3579523543
v_th    = 0.5
v_reset = 0.0
reset   = hard
```

Comparison uses the compiled threshold and reset codes recorded in the report.
Those codes must be expressed in the inherited completed-accumulator domain,
and `initial_state_code` equals `reset_code`, so validation checks the compiled
domain contract rather than fragile decimal string equality. Reports include
the constant compile errors defined in Section 10.1. Other functional values
require a later compiled hardware profile and are rejected by v1.

## 11. LIF State Store

Digital `Vmem` is integer state in the inherited completed-accumulator domain.
It is not requantized at the LIF boundary, and no analog voltage state is
created or restored.

Phase 2 introduces a logically separate `LifStateStore` keyed by:

```text
(sample_id, lif_node_id)
```

Its record contains at least:

```text
state tensor shape
state bit width
accumulator domain id and scale
initial state code
last_completed_timestep
storage backend
initialization policy
```

For SNN v1, a LIF node advances in timestep order as each timestep's complete
input becomes available. Once its ordered invocation stream starts in the
round that closes those inputs, it is not preempted by an unrelated state
owner. The active neuron batch uses the provisioned lane registers, while the
complete layer state follows the mandatory Tile Buffer policy in Section 10.4.

This is deliberately different from ordinary activation boundaries:

```text
activation/spike: graph data consumed by downstream nodes
lif state:        private recurrent state consumed by the same LIF node at t+1
```

LIF state must not be published as a normal downstream activation. Future
preemption or timestep-window execution may add explicit state spill/restore
events without changing this identity.

Storage domains must remain distinct:

```text
local LIF register / on-chip LIF state buffer:
  modeled with existing MNSIM Reg/Buffer latency and energy

cross-round graph boundary store:
  represents DDR in the current experiment; payload is tracked but transfer
  latency and energy are excluded
```

Calling both objects a "buffer" does not permit combining their ledgers.

## 12. Boundary Management

Boundary management is sequence-aware and preserves three distinct payload
kinds:

```text
partial_sum_sequence
activation_sequence
spike_sequence
```

### 12.1 Same-round handoff

When producer and consumer execute in the same physical round and timestep,
data remains round-local. No cross-round boundary-store event is emitted.

### 12.2 Weighted split across rounds

An incomplete weighted layer stores digital partial sums for every executed
timestep. A later round restores and completes the matching timestep entries.
The downstream LIF is not visible until completion.

### 12.3 Complete output across rounds

When a completed non-LIF activation or LIF spike output has consumers in a
later physical round, the boundary state carries a sequence shape:

```text
[T, output_shape...]
```

A LIF spike sequence uses one logical bit per neuron per timestep unless the
frontend declares another representation.

### 12.4 Logical key and physical batching

Boundary APIs must expose timestep identity even when one physical write/read
event batches all `T` entries. Required audit fields are:

```text
sample_id
producer_node_id
consumer_node_ids
payload_kind
sequence_length
timestep_range
element_shape
data_bitwidth
storage_backend
write_count
read_count
lifetime/refcount
```

### 12.5 DDR accounting

Cross-round boundary storage continues to represent DDR in the current
experiment:

```text
functional write/read: required
payload bytes/events:  reported
DDR latency:           excluded
DDR energy:            excluded
```

This exclusion does not authorize dropping boundary data or bypassing
dependency validation.

## 13. Phase 1 Changes

Phase 1 remains resource- and dependency-oriented.

Required changes:

1. Preserve `SequenceSpec` in `RoundPlan` rather than only validating it.
2. Accept `lif` as a supported stateful non-weighted node.
3. Preserve LIF functional attributes and state metadata in
   `LocalGraphNode.node_shape` or an explicit JSON-friendly operator metadata
   field.
4. Do not add any `lif` xbar requirement.
5. Do not impose legacy SNN bundle atomicity or single-round input closure.
6. Require one uniquely described completed digital input per LIF. Reject
   incompatible multi-input domains before Phase 2.
7. Preserve the exact per-Tile neuron ownership shard `N_k` for LIF placement
   and cost accounting.
8. Continue current multi-layer packing and weighted split/continuation rules.

The planner does not schedule individual timestep invocations. It emits the
same static execution items plus the sequence contract consumed by Phase 2.

## 14. Phase 2 Changes

The first implementation should add or extend these responsibilities:

```text
types.py
  sequence-aware invocation and report records

executor.py
  one write phase per weighted round
  timestep invocation loop
  exactly-once validation

round_local_state.py
  timestep-scoped same-round visibility

boundary.py
  timestep-aware partial coverage and sequence payloads

digital_lif.py
  functional state store and composed digital hardware cost

lowerer.py
  lower `lif` to digital LIF work, not zero-cost HookResult

report.py
  read/write/LIF/boundary breakdown and invocation audit
```

Suggested public/internal records include:

```text
InvocationKey(sample_id, node_id, timestep)
AccumulatorDomain
DigitalLifRequest
DigitalLifResult
LifStateRecord
SequenceBoundaryDescriptor
```

Names may follow local conventions during implementation, but the concepts
and serialized audit fields are required.

## 15. Metric Accounting

For weighted round `r`:

```text
E_round[r] = E_program[r]
             + sum_t E_weighted_read[r, t]
             + sum_t E_non_weighted[r, t]
             + sum_t E_lif[r, t]
             + E_on_chip_state_access[r]

L_round[r] = L_program[r]
             + sequence_execution_latency[r]
             + L_on_chip_state_access[r]
```

Weight programming is outside the timestep sum.

For SNN v1, timestep latency is serialized:

```text
sequence_execution_latency[r]
  = sum_t latency_of_current_item_schedule[r, t]
```

The schedule within one timestep may retain existing same-round hardware
parallelism. No latency overlap is claimed between timestep `t` and `t+1`.

The report must separate at least:

```text
weight_programming
weighted_read
ordinary_non_weighted
digital_lif_compute
lif_state_access
partial_sum_access
spike_or_activation_boundary_access
DDR transfer, excluded
```

The current cross-round partial-sum and activation/spike boundary store is the
excluded DDR domain. Its bytes and access events are reported, but it adds zero
latency and energy. `lif_state_access` is included only when the configured
digital LIF uses an on-chip register or Buffer operation.

The first SNN implementation uses dense-equivalent operation counting for
comparability with ANN:

```text
ops_total_snn = T * 2 * sum(weighted_layer_MACs_per_timestep)
```

As in the current MNSIM TOPS convention, ordinary non-weighted and digital LIF
logic operations contribute latency and energy but are not added to the TOPS
numerator. Reports must expose the separate LIF update count so this convention
is auditable. Dynamic activity-aware energy is deferred; zero input still
performs a functionally required LIF update.

## 16. ANN Compatibility

`T = 1` is a strict regression requirement.

For a model without `lif` nodes and with the ANN `SequenceSpec`, the extension
must preserve:

```text
RoundPlan construction
physical xbar counts and placement
weight-programming count and cost
read latency and energy
boundary states
TOPS and TOPS/W
serialized result meaning
```

No ANN metric may be multiplied by an extra timestep factor, and no LIF state
store may be created for an ANN graph.

## 17. Validation and Error Conditions

The executor must fail clearly for:

```text
invalid or missing SequenceSpec
SNN T <= 1 when the frontend declares ordered recurrent LIF execution
unknown recurrent_state_id
LIF input incomplete at attempted invocation
LIF timestep executed more than once
LIF timestep skipped at sample finalization
LIF timestep t executed before t-1
partial contribution assigned to the wrong timestep
boundary sequence length mismatch
state tensor shape or bit-width mismatch
state leaking between samples
unsupported LIF functional attributes
unsupported compiled tau/threshold/reset profile
missing completed-accumulator width or scale metadata
LIF has zero inputs, multiple inputs, or no unique accumulator domain
merged digital inputs have mismatched domain id, width, or scale
threshold/reset code domain mismatch
constant code out of signed state range
initial_state_code differs from reset_code in v1
state writeback is not exactly representable in B bits
any requested LIF input requantization
```

It must not silently fall back to an analog LIF model, zero-cost hook, or CPU
execution semantics to hide these errors.

## 18. Required Tests

### 18.1 Compatibility

1. Existing ANN/CNN `T = 1` tests remain unchanged and pass.
2. The formal G8 ANN reference preserves two physical rounds and its existing
   performance totals within normal floating-point tolerance.

### 18.2 Invocation semantics

3. One LIF node with `T = 4` emits exactly four invocation records.
4. A zero-input timestep still advances LIF state and emits a spike value.
5. Duplicate `(sample, lif_node, timestep)` execution raises an error.
6. Missing invocation at sample finalization raises an error.
7. The first timestep starts from `initial_state_code = reset_code`; a second
   sample does not inherit the first sample's terminal state.

### 18.3 Boundary and partial sums

8. A weighted layer split across two rounds does not execute LIF in the first
   round.
9. The second round completes each timestep's partial sum and executes every
   LIF timestep once in order.
10. Contributions with identical fragment ids at different timesteps are
    legal; duplicate fragment ids at the same timestep are rejected.
11. Cross-round spike output preserves `[T, ...]` shape and timestep order.
12. Same-round weighted-to-LIF-to-weighted handoff creates no cross-round
    state.
13. A standalone zero-xbar LIF item runs exactly `T` times and performs no
    weight-programming event.

### 18.4 Hardware cost

14. LIF energy scales with neuron update count `N * T`.
15. LIF latency uses `T * max_k ceil(N_k / 64)` for the preserved Tile shards;
    an unbalanced shard is not replaced by ideal balancing.
16. LIF area scales with configured lane count and does not scale with `T`.
17. Weight-programming latency and energy remain unchanged when only `T`
    changes.
18. Buffer/register state accounting follows the selected storage policy.
    Non-byte-aligned, non-bus-aligned, and uneven multi-Tile shards match the
    per-Tile/per-timestep ceiling formulas exactly.
19. Cross-round DDR boundary bytes change with `T`, while their reported
    latency and energy remain zero in the current accounting scope.
20. The formal tau compiles to Q1.15 code `24130` and the recorded five-term
    CSD decomposition reconstructs that code exactly.
21. Positive and negative nearest-even cases, including both half-way tie
    parities, match an integer reference model and charge both the rounding
    decision and conditional increment hardware stages.
22. Completed accumulator codes enter LIF bit-exactly, with inherited width and
    scale and no input requantization operation or cost.
23. A LIF with one unique input domain is accepted; zero-input, multi-domain,
    and mismatched digital-merge inputs fail before Phase 2.
24. Threshold, reset, and initial-state constants compile offline with
    nearest-even, record code/scale/error, and fail when not representable.
25. Missing completed-accumulator metadata, a threshold/reset domain mismatch,
    or any request for LIF input requantization fails before Phase 2 execution.
26. Minimum/maximum signed inputs and states, positive/negative state updates,
    `u == threshold`, checked B-bit writeback, and overflow-error behavior match
    the integer reference model without clipping, wraparound, or saturation.
27. Unsupported tau, threshold, or reset fails before Phase 2 execution.
28. Reg/ShiftReg/JointModule cost is monotonic for 8-, 16-, and 32-bit
    profiles, with legacy lookup tables composed from explicit slices.
29. Concurrent LIF states size each Tile Buffer by the maximum live state sum,
    not by the maximum single layer.

### 18.5 Reports

30. Reports expose sequence length, invocation counts, LIF compute/state
    costs, boundary payload bytes, and excluded DDR scope.
31. Reports expose accumulator domain identity, constant compile metadata,
    per-Tile `N_k`, primitive slice counts, and rounding proxy costs.
32. Per-round totals reconcile exactly with the sum of their reported
    components.

## 19. Implementation Order

The recommended implementation order is:

```text
1. Preserve SequenceSpec in RoundPlan and add T=1 regression coverage.
2. Add timestep identity to Phase 2 runtime and boundary keys.
3. Extend partial-sum accumulation across T without adding LIF cost.
4. Add LifStateStore and exactly-once functional LIF execution.
5. Add DigitalLifCostModel using existing MNSIM digital modules.
6. Add sequence-aware boundary batching and audit fields.
7. Add report breakdowns and end-to-end SNN smoke tests.
8. Run ANN G8 regression, then SNN single-sample full-T regression.
```

## 20. Deferred Decisions

The v1 accumulator-domain inheritance, lane hierarchy, decay implementation,
state placement, rounding point, overflow behavior, and operation schedule are
fixed in Section 10. The remaining follow-up choices are:

```text
spike-activity-aware DAC/xbar/ADC gating
recurrent and sequence-model state beyond feed-forward LIF
timestep-window preemption and LIF state spill/restore
DDR transfer model
```

Any later choice must preserve the confirmed round, invocation, input
completion, and boundary semantics in this document.

## 21. Completion Criteria

This design is implemented when:

```text
CNN T=1 remains behaviorally and numerically compatible;
SNN T>1 uses the same physical RoundPlan abstraction;
each weighted round programs once and reads T times;
each LIF node updates exactly once per timestep;
cross-round partial sums and spike sequences preserve timestep identity;
ADC is the only analog-to-digital quantization point;
completed partial sums enter LIF without requantization;
digital LIF cost is nonzero and composed from existing MNSIM digital modules;
no analog Vmem or legacy step-closure constraint participates;
reports reconcile all included costs and explicitly mark DDR as excluded.
```
