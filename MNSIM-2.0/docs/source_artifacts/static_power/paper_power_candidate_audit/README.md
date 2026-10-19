# Static-Power Literature Candidate Audit

Audit date: 2026-08-14

This record qualifies additional published static- and standby-power results
for the round-orchestrated static-power ledger. The PDFs are not copied into
the repository. DOI or public-document URLs, page counts, and SHA-256 digests
identify the exact documents inspected.

The classification used here is:

- `directly_usable`: same circuit identity, organization, process, voltage,
  temperature, and ledger power state;
- `alternative_implementation`: a real static-power source only if the
  simulator changes to the reported circuit and models its power-state policy;
- `cross_check_only`: useful for magnitude or state-behavior validation, but
  incompatible with the current implementation;
- `measurement_method_only`: supports an accounting or measurement method but
  exposes no separable parameter for the current ledger;
- `rejected`: not a static-power source.

No paper in this record is `directly_usable` for the current ReADC, TDAC, or
40 nm SRAM instances. A result is never normalized by bit count or rescaled by
technology node to promote it into the nominal profile.

## 1. Measured 90 nm 2-kbit 6T SRAM

Publication:

> T. Enomoto and N. Kobayashi, "Single-Power-Supply Six-Transistor
> CMOS SRAM Enabling Low-Voltage Writing, Low-Voltage Reading, and Low
> Standby Power Consumption," IEICE Transactions on Electronics, vol. E106-C,
> no. 9, 2023, pp. 466-476. DOI: 10.1587/transele.2022ECP5053.

Document identity:

```text
DOI URL: https://doi.org/10.1587/transele.2022ECP5053
audited PDF URL: https://www.jstage.jst.go.jp/article/transele/E106.C/9/E106.C_2022ECP5053/_pdf
pages: 11
SHA-256: 818756131506f93a3f3822ae02dc4ee8e824914949a937e32e76854f8d71a0de
```

The prototype contains a conventional 6T SRAM and a developed 6T SRAM with
self-controllable voltage-level (SVL) circuits. Figure 15 and Table 4 report
the following average measured standby powers over five chips:

```text
process:                   90 nm CMOS
capacity:                  2 kbit
external VDD:              1.0 V
temperature:               25 C
conventional SRAM:         10.12 uW
SRAM with SVL retention:   0.957 uW
```

The standby measurement includes peripherals stopped by clock gating. In the
developed SRAM, the SVL circuitry lowers the memory-cell supply and drives the
wordline-driver supply to 0 V while retaining data. The `0.957 uW` point is
therefore a specialized retention state, not ordinary powered standby. The
paper also reports close SPICE values of `10.28 uW` and `0.984 uW` for the two
implementations.

Decision: `cross_check_only`. This is strong measured evidence that topology
and retention controls can change SRAM standby power by more than an order of
magnitude even at one process, capacity, voltage, and temperature. Neither
measured value matches the current 40 nm CACTI macro identity, and the SVL
value cannot be used under `no_implicit_power_gating`.

## 2. Simulated 180 nm 8K-bit low-standby SRAM

Publication:

> K. Lee and J.-Y. Sim, "An Implantable Bio-Signal Sensor SoC with
> Low-Standby-Power 8K-Bit SRAM for Continuous Long-Term Monitoring,"
> Electronics, vol. 12, no. 10, 2023, article 2317.
> DOI: 10.3390/electronics12102317.

Document identity:

```text
DOI URL: https://doi.org/10.3390/electronics12102317
audited PDF URL: https://mdpi-res.com/d_attachment/electronics/electronics-12-02317/article_deploy/electronics-12-02317.pdf?version=1684655591
pages: 15
SHA-256: 110fca12806183fe123f098d274e9acad4f35ba9e952614fb2baa4c60744e807
```

This work is a transistor-level custom-circuit simulation in a commercial
180 nm CMOS technology. It is not a fabricated SRAM measurement. Its power
state is also highly specialized:

```text
capacity and IO:                 8 Kbit, 16-bit IO
standby cell-array supply:       0.4 V
standby wordline voltage:       -0.18 V
active cell-array supply:        1.0 V
active wordline voltage:         0 V
8K bit-cell standby power:       23.85 pW
negative-bias generator:         27 pW
SRAM plus memory controller:     0.11 nW standby
SRAM active-mode average power:  about 97.8 uW
```

The paper does not provide the PVT corner needed to transplant the numbers
into another implementation. More importantly, the low value depends on
negative wordline bias, dynamic supply-voltage switching, and their control
circuits. The `27 pW` bias-generator contribution demonstrates that support
circuitry is part of the retention-state boundary and cannot be omitted.

Decision: `cross_check_only`. The values cannot be assigned to the current
SRAM or memory controller. They validate the requirement that any future
retention state must include its bias generator, control logic, state
transition behavior, and state-specific voltage.

## 3. Measured 65 nm power-gated SAR-ADC temperature sensor

Publication:

> K. Pelzers, H. Xin, E. Cantatore, and P. Harpe, "A 2.18-pJ/conversion,
> 1656-um2 Temperature Sensor With a 0.61-pJ K2 FoM and 52-pW Stand-By
> Power," IEEE Solid-State Circuits Letters, vol. 3, 2020, pp. 82-85.
> DOI: 10.1109/LSSC.2020.3005784.

Document identity:

```text
DOI URL: https://doi.org/10.1109/LSSC.2020.3005784
audited PDF URL: https://pure.tue.nl/ws/files/174160814/publisher_version.pdf
pages: 5 (repository cover plus 4-page published letter)
SHA-256: 27bd61537ae2cfe6b544ab7eebd18747f4d7f302b28a2a7cafbe8a0116e3e78b
```

The measured block is a complete all-dynamic temperature sensor containing a
duty-cycled resistive bridge, a 9-bit asynchronous SAR ADC, power-gating
logic, and retained control memory:

```text
process:                         65 nm CMOS
core supply VDD:                 0.6 V
secondary supply VDDH:           1.0 V
temperature of standby result:   room temperature
complete-sensor standby power:   52 pW measured
conversion energy with CDS:      2.18 pJ measured
```

The `52 pW` number is not an ADC-core-only measurement. The paper attributes
the low standby result to automatic power gating that disconnects the ADC
local supply after conversion. It separately gives the following simulated
leakage results:

```text
bridge/control leakage:          1593 pW -> 31 pW
power-gating-circuit leakage:     1.6 nW -> 17 pW
retained memory leakage:          about 2 pW per cell
```

Decision: `alternative_implementation`. This is a valid measured standby
source for the complete reported sensor topology and state. It is not a
source for the Hong ReADC, which has a different conversion principle,
technology basis, supply structure, and disable behavior. Adopting this value
would require changing the modeled ADC implementation and accounting for the
bridge, power gate, retained memory, and wake/sleep policy as actually
instantiated. Until then, ReADC static power remains `unmodeled`.

## 4. IBM HERMES 14 nm PCM compute-in-memory chip

Publication:

> M. Le Gallo et al., "A 64-core mixed-signal in-memory compute chip based
> on phase-change memory for deep neural network inference," Nature
> Electronics, vol. 6, 2023, pp. 680-693.
> DOI: 10.1038/s41928-023-01010-1.

Document identity:

```text
published DOI URL: https://doi.org/10.1038/s41928-023-01010-1
audited public preprint URL: https://arxiv.org/pdf/2212.02872
preprint pages: 25
preprint SHA-256: 81c512921e635d20769b9a0752d958f6820158f8459728e9333cb320d39b1278
```

The chip uses 14 nm CMOS with back-end integrated PCM. Its Methods section
states that power is measured separately in standby mode (static) and during
computation (dynamic). Average current on the ADC and digital supplies at
`0.85 V` is measured with Fluke 87V and 185 meters. Dynamic power is measured
during repeated commands, and total energy is formed by adding static and
dynamic contributions.

The public performance table reports combined energy-efficiency results, but
does not expose a standalone standby-power number that can be separated and
mapped to the current ADC, DAC, SRAM, digital, or crossbar-peripheral rows.

Decision: `measurement_method_only`. The paper supports the ledger's rule that
standby static and computation dynamic contributions must be characterized
separately before being combined. It supplies no numeric static parameter for
the present simulator.

## 5. Ledger Consequence

The additional literature does not close a new static-power row:

| Candidate | Evidence | Qualification for current ledger |
| --- | --- | --- |
| 90 nm 2-kbit 6T SRAM | five-chip standby measurement | `cross_check_only` |
| 180 nm 8K-bit SRAM | transistor-level simulation | `cross_check_only` |
| 65 nm SAR-ADC temperature sensor | complete-system standby measurement | `alternative_implementation` |
| 14 nm PCM CIM chip | chip measurement methodology | `measurement_method_only` |

The useful result is not a universal leakage coefficient. It is evidence that
the physical power state, bias generators, retained control, supply domains,
and clock/power gating can dominate the reported standby power. Consequently:

- `0.957 uW` and `0.11 nW` are not current-SRAM parameters;
- `52 pW` is not a ReADC parameter;
- no reported number is scaled by capacity or technology node;
- CACTI remains the only reproducible Stage-A SRAM architecture estimate;
- ReADC, TDAC, 1T1R access devices, analog sense circuitry, switch matrices,
  and unmapped digital logic remain `unmodeled` in the static domain.
