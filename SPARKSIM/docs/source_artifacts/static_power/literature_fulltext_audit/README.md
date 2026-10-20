# ADC and DAC Full-Text Static-Power Audit

Audit date: 2026-08-14

This record pins the public documents inspected while qualifying the current
ReADC and TDAC sources for the static-power ledger. The PDFs are not copied into
the repository. The URLs and hashes identify the exact files that were read.

## Hong ReADC

Publication:

> H. Hong et al., "Memristor-based adaptive analog-to-digital conversion for
> efficient and accurate compute-in-memory," Nature Communications, 2025.
> DOI: 10.1038/s41467-025-65233-w.

Published article:

```text
URL: https://www.nature.com/articles/s41467-025-65233-w.pdf
pages: 11
SHA-256: 17924ce59184ae951061db673fa9865c054e3434aec2677a5992fb76dc5cfda2
```

Supplementary Information:

```text
URL: https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-025-65233-w/MediaObjects/41467_2025_65233_MOESM1_ESM.pdf
pages: 21
SHA-256: e9048bf1437abdce0e679f16c732024964911666baa8bc537ad4ab542de5e522
```

Finding: both files report conversion energy, latency, area, device variation,
and layout evidence. Neither file reports an idle/standby supply current,
leakage power, leakage PVT corner, or an ADC disable-state bias vector. The
`12.58 fJ` value is not a static-power source.

The 28 nm custom layout in Supplementary Fig. 7 validates area only. It does
not convert the main paper's mixed 16 nm analog-core plus 28 nm decoder energy
estimate into a same-PVT static characterization.

## Uenohara TDAC

Publication:

> S. Uenohara and K. Aihara, "Time-Domain Digital-to-Analog Converter for
> Spiking Neural Network Hardware," Circuits, Systems, and Signal Processing,
> 2021. DOI: 10.1007/s00034-020-01597-2.

Published article:

```text
URL: https://link.springer.com/content/pdf/10.1007/s00034-020-01597-2.pdf
pages: 19
SHA-256: fe25760279a1e19f48890800cab01a12111c355c01db947453a71edab31dc2d2
```

Finding: the TSMC 40 nm, `VDD = 0.7 V` Spectre result reports `27 fJ` for one
maximum-code conversion. The design disables a direct wasteful-current path
during reset/inactivity, but the paper does not report the remaining idle
supply current or leakage. The conversion energy is not a standby-power source,
and disabling the intended current path is not evidence of zero leakage.

## Acceptance Decision

Both publications remain valid dynamic-operation sources within their stated
boundaries. Both are rejected as static-power parameters. The current ReADC and
TDAC rows stay `unmodeled` until the actual circuits are simulated or measured
at a declared PVT with explicit idle/read-ready control vectors and total supply
power measurements.
