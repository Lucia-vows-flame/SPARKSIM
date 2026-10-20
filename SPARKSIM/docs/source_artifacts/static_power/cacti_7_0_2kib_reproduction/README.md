# CACTI 2 KiB SRAM Reproduction Artifact

This directory preserves the exact input used for the CACTI toolchain
reproduction recorded in
`docs/specs/2026-08-14-static-power-source-audit.md`.

Source identity:

```text
repository: https://github.com/HewlettPackard/cacti
revision: 1ffd8dfb10303d306ecd8d215320aea07651e878
binary SHA-256: 6e79c30bd1edf5fda17a8e4f49edf04b97322a17000216bd394300bc3fd06040
```

Invocation form:

```text
./cacti -infile input.cfg
```

Input hash (input.cfg is included) and generated output hash (stdout is omitted
from the source-only release):

```text
65107b76b6e3628f6f9a5eea4ce798688b1845bdad3fec969eb4575fdf22808b  input.cfg
f2fb220846672e40150c2060a0a95908acc9f9ade758dd4c2ab1f0a5b2d58192  raw_output.txt
```

This is a reproduction-only artifact. The raw stdout is not included in this
source-only release; the hash above identifies the locally retained run.
Regenerate it with the invocation above to reproduce the record. `input.cfg`
inherited `Add ECC = true`
and `Optimize ED^2` from CACTI's sample configuration. It proves the pinned
toolchain can reproduce the cited 2 KiB result, but it is not eligible for a
production static-power profile. Production generation must use the explicit
contract in the source audit and regenerate every materialized SRAM capacity.
