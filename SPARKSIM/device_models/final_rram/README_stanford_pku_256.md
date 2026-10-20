# Stanford-PKU 256-State Model

The nominal read table is derived from the pinned MemTorch Stanford-PKU
compact model at `Vread=0.1 V`. MNSIM loads the generated 256-code table; it
does not import MemTorch at runtime.

Two artifacts are retained for this device family:

- `rram_memtorch_stanford_pku_256.json` provides nominal conductance and
  resistance values for read and accuracy analysis. It has no write transition
  costs, so hardware write accounting rejects it.
- `rram_memtorch_stanford_pku_write256.json` includes SET/RESET state
  transitions evaluated with the compact model. This is an ideal model result,
  not measured 256-level programming evidence.

`source/open_models/Stanford_PKU.py` and the pinned MemTorch GPL-3.0 license
are kept with source checksums in `source/open_models/sources.lock.json`.
The level table and transition costs are serialized in the model artifacts.
