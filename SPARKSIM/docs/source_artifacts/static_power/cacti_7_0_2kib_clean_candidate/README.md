# CACTI 7.0 2 KiB Clean Static-Power Candidate

Generated: 2026-08-14

This directory contains a cleaned CACTI candidate for a 2 KiB, 128-bit,
single-port, single-bank scratch SRAM at 40 nm and 300 K using ITRS-LSTP cells
and periphery. It is an architecture estimate, not a foundry memory-compiler
report or measured macro.

Source:

```text
repository: https://github.com/HewlettPackard/cacti
revision: 1ffd8dfb10303d306ecd8d215320aea07651e878
binary SHA-256: 6e79c30bd1edf5fda17a8e4f49edf04b97322a17000216bd394300bc3fd06040
```

Unlike the earlier toolchain reproduction, this input explicitly uses:

```text
Add ECC: false
Optimize ED/ED^2: NONE
UCA objective weights: delay=0, dynamic=0, leakage=100, cycle=0, area=0
array/WL/CL/interconnect power gating: false
bitline floating: false
```

The stock CACTI leakage objective covers its subthreshold-leakage field. It
does not minimize `subthreshold + gate leakage`. This artifact is therefore one
fully specified candidate, not proof of the global minimum-total-static
organization. A production generator must enumerate legal forced
organizations, banks, and device flavors and rank the resulting total static
power externally subject to the access-time constraint.

Result:

```text
access time:                         0.718940 ns
cycle time:                          0.842917 ns
subthreshold leakage power:         5.13732e-05 mW
gate leakage power:                 4.34823e-05 mW
total static power used by ledger:  9.48555e-08 W
best Ndwl/Ndbl/Nspd/Ndcm/Ndsam1/Ndsam2: 2/2/4/8/1/1
```

Input hash (input.cfg is included):

```text
SHA-256: 1a1aa65f323f09f34b92f673bd1237ce52f18cbd9119500f81e620881642963b
```

Generated output hashes (the output files are omitted from the source-only release):

```text
raw_output.txt SHA-256:
  4e1e1867d8d9748a8eeac35cabe5c6ecc8532f1000e8ae9829d138eefb649165
result.csv SHA-256:
  08fd0eb80049a9671008b509c943de2424c3b1a5f00ab5a37b631248d53bf45b
```

Invocation:

```text
cd <pinned-cacti-source-tree>
./cacti -infile input.cfg
```

The runtime working directory is part of the reproduction contract. This old
CACTI tree resolves supporting data relative to its source/build directory;
running the same binary from an arbitrary directory is not a supported
reproduction path.

`input.cfg` selects `CONCISE` output. The raw output and result CSV are not
included in this source-only release; their hashes above identify the locally
retained CACTI run. Regenerate them with the invocation below to reproduce the
recorded values. The earlier `DETAILED` mode exposed
uninitialized near-zero component fields and therefore produced unstable raw
output hashes even though the UCA result was unchanged. Two serial runs of the
clean candidate produced byte-identical standard output and byte-identical CSV
output. `raw_output.txt` is the standard output and `result.csv` is CACTI's
machine-readable `input.cfg.out`, renamed for clarity. In the CSV,
`Standby leakage per bank(mW) = 9.48555e-05` is CACTI's sum of subthreshold
and gate leakage. A separate parallel rerun with distinct input/output names
also reproduced the same bytes; the earlier failure outside the source/build
directory was a working-directory error, not evidence of nondeterminism.

This CACTI prerelease also prints an external-DDR IO estimate and a MemCad
example after the UCA result. Neither belongs to this scratch-SRAM macro or the
static ledger. Only the `Uniform Cache Access SRAM Model` block and its matching
CSV row are qualified here.
