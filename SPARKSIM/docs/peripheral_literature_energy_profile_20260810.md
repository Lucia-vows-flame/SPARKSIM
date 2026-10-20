# G8 Literature Peripheral Envelope

`configs/hardware_profiles/g8_literature_peripheral_ann.ini` is the G8
hardware-only profile for the round-orchestrated ANN simulator. It is a
literature performance envelope, not a claim that all listed blocks were
co-fabricated in one measured macro.

The profile preserves the supplied Excel-derived RRAM configuration exactly:

- the full `Device level` section, including the 256-level resistance and
  pulse tables;
- `Xbar_Size = 128,128`, `Xbar_Polarity = 2`, and `Group_Num = 2`;
- `DAC_Num = 128`, `ADC_Num = 128`, `PE_Num = 4,4`, and `Tile_Num = 6,3`.
  With two groups and differential polarity, this is `1152` physical xbars.

Only the peripheral model is replaced.

| Component | Parameter | Value | Source condition |
| --- | --- | ---: | --- |
| ADC | conversion energy | 0.19 pJ | 10 bit, 0.7 V, 200 kSamples/s |
| ADC | conversion latency | 5000 ns | reciprocal of 200 kSamples/s |
| DAC | conversion energy | 27 fJ | 8-bit signed TDAC, 0.7 V |
| DAC | conversion latency | 16 ns | Figure 9 timing envelope |
| SRAM buffer | access energy | 1.0 fJ/bit/access | one common access value |
| SRAM buffer | leakage | 13.72 nW/Mbit | standby |
| SRAM buffer | access latency | 1.84 ns | read access |
| adder / merge | energy | 3.73 aJ/bit-op | PDP proxy |
| adder / merge | latency | 0.481 ns | operation delay |
| register / shift register | energy | 0.153 fJ/bit-update | per update |

References:

1. Bindra et al., 2019, DOI `10.23919/VLSIC.2019.8778093`.
2. Uenohara and Aihara, 2021, DOI `10.1007/s00034-020-01597-2`.
3. Aiello et al., 2019, DOI `10.1109/ACCESS.2019.2938737`.
4. Wang et al., 2023, DOI `10.1049/cds2.12141`.
5. Yabuuchi et al., 2017, DOI `10.23919/VLSIT.2017.7998145`.
6. Arora and Tripathi, 2024, DOI `10.1088/1402-4896/ad5c0b`.
7. Liu et al., 2024, DOI `10.1109/ACCESS.2024.3432162`.

The backend treats ADC/DAC conversion energy and latency as explicit values.
Buffer dynamic energy scales with transferred payload bits. Following original
MNSIM, SRAM leakage scales with buffer capacity and is charged during modeled
buffer-access cycles. Digital block energy
scales with configured bit width. The GNRFET full-adder value is a PDP proxy,
not a measured CMOS switching energy.

## Current Dynamic Reference Run

The current CPU hardware-only ResNet18 run completed on 2026-08-12 with the
ordinary checkpoint, `--skip-accuracy`, and two physical rounds.  It includes
the physical weight-programming transition of every weighted round before its
read phase:

- `R0`: `1030 / 1152` physical xbars (89.41% utilization);
- `R1`: `728 / 1152` physical xbars (63.19% utilization).

- read latency: `35.108466 ms`;
- read energy: `1.028778 uJ`;
- weight-programming latency: `32.853660 ms`;
- weight-programming energy: `8.092973 nJ`;
- total latency: `67.962126 ms`;
- total energy: `1.036871 uJ`;
- average power: `15.256595 uW`;
- throughput: `0.004526583 TOPS`;
- efficiency: `296.696661 TOPS/W`.

`R0` transitions its `1030` occupied physical xbars from the erased
code-zero state.  `R1` targets `728` physical xbars; it transitions `1030`
physical xbars in total because the `302` slots released by `R0` are also
reset to code zero.  The result artifacts are in
`results/g8_literature_peripheral_1152xbar_cpu_write_mnsim_leakage_20260812/`.

Relative to the otherwise identical no-leakage run, original-MNSIM buffer
leakage adds `0.548514 pJ` and changes efficiency from `296.696818` to
`296.696661 TOPS/W`. The small delta is expected because leakage is charged
only over modeled buffer-access cycles.

The earlier 2026-08-11 read-only result (`35.108466 ms`, `1.028778 uJ`,
`299.030628 TOPS/W`) remains useful as an isolated peripheral-read reference,
but is not the formal dynamic round result.  Weight programming nearly doubles
the modeled latency here, while changing TOPS/W only modestly because its
`8.092973 nJ` energy is small compared with the read energy.

These are performance-model outputs only. No QAT, accuracy evaluation,
BN-fold, or legacy SNN path participates in this run.

The complete ResNet18 mapping requires 1758 physical xbar placements, so the
1152-xbar G8 fabric reuses its resources across the two rounds. `R0` writes
the boundary activation after `L058_relu`; `R1` reads it before continuing to
the classifier. That cross-round activation represents DDR in the current
model, but its transfer latency and energy are deliberately excluded. SRAM
buffer leakage follows original MNSIM's access-cycle accounting
(`Buffer_IncludeLeakageEnergy = 1`); no additional whole-chip always-on
static-power ledger is introduced.
