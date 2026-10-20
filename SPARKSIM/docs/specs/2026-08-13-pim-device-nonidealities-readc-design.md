# PIM Device Non-Idealities and ReADC Design

## Scope

This change completes the numerical accuracy model only. Scheduling, physical
rounds, latency, energy, and weight-programming accounting remain unchanged.

## Device model

Each stored weight digit is one physical cell value. For differential xbars,
positive and negative arrays are separate cells and receive independent faults.

The enabled operations are applied in this order:

1. Interpret the integer digit as a programmed device level.
2. If R-ratio is enabled, map the level through `Device_Resistance` and normalize
   conductance so one ideal code step remains one numerical weight unit.
3. If variation is enabled, independently perturb every cell resistance with
   `R' = R * (1 + Normal(0, sigma))`, resampling non-positive resistance factors.
4. If SAF is enabled, independently force cells to HRS (code zero) or LRS
   (maximum code). SAF is terminal and therefore overrides variation.

R-ratio, variation, and SAF are independent controls. Variation requires the
physical resistance mapping even when the deterministic R-ratio control is off.
All random draws use a run-owned NumPy generator when an accuracy seed is given.

## ReADC model

`READC_LLOYD_MAX` follows Hong et al. (Nature Communications 2025,
doi:10.1038/s41467-025-65233-w) and the authors' public `MIKEHHQ/ReADC` code:

- collect pre-ADC crossbar-output statistics per weighted network layer from a
  representative calibration loader;
- retain the configured percentile range and a deterministic bounded sample;
- fit `2^ADC_Precision` reconstruction levels with Lloyd-Max iterations;
- store internal decision boundaries and reconstruction levels per layer;
- use the fixed per-layer tables during inference.

This is offline calibration, not per-input threshold adaptation. Tables are
bound to the ADC hardware profile through `ADC_Quantization_Mode`. The legacy
`ADAPTIVE` static boundary mode is retained as `STATIC_NONUNIFORM` compatibility.
No activation quantizer is inserted after the ADC; the ADC reconstruction value
continues through the existing digital shift/add path.

## Reproducibility and reporting

The accuracy report records device percentages, resistance table, mapping order,
seed, ADC model/provenance, and calibration settings. Missing calibration for a
ReADC layer is an error; there is no silent uniform fallback.

The dataset training loader is the calibration source. The evaluated test
subset is never used to fit thresholds.

`ADC_Profile=HONG_2025_READC_5BIT` with `ADC_Quantization_Mode=AUTO`
selects ReADC and validates 5-bit precision. A legacy/missing profile with AUTO
selects uniform MNSIM quantization.
