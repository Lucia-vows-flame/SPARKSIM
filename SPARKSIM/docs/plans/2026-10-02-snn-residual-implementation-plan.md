# SNN Residual Implementation Plan

> **For agentic workers:** Execute the tasks in order. Keep the original MNSIM ANN behavior intact and preserve unrelated worktree changes.

**Goal:** Add a V7 ResNet18 SNN whose software accuracy follows the SpikingJelly residual-block placement, while reusing MNSIM's existing graph, element-sum, round planning, and performance models.

**Architecture:** Represent residual blocks as the existing ordered `NetworkGraph` nodes and `input_index_list` edges. The SNN builder explicitly places LIF nodes after the first BN and after each residual merge. The temporal accuracy path evaluates every branch at matching timesteps. The performance path reuses the existing ANN residual DAG and structural `element_sum` execution, adding only the SNN metadata and merge-output ownership needed by a LIF whose direct producer is an `element_sum` node.

**Tech Stack:** Python 3.12, PyTorch, SpikingJelly multi-step LIF, existing MNSIM `NetworkGraph`, Phase 1 round planner, Phase 2 executor, and pytest.

**Reference behavior:** SpikingJelly `BasicBlock`: `conv1 -> BN -> LIF`, `conv2 -> BN`; identity or projection shortcut; add the two paths; then apply LIF. The project keeps its V7 input contract `[B,T,6,H,W]`, 8 classes, `T=30`, and no-pooling/stride-convolution downsampling policy.

## Implementation status — 2026-10-03

- [x] Tasks 1–2: explicit ResNet18 residual graph, identity/projection structure and V7 shape checks; ANN and existing sequential SNN model definitions preserved.
- [x] Task 3: temporal ORIGINAL, QAT and PIM paths; exact output comparison against installed SpikingJelly BasicBlock for both shortcut types; backward/state reset/error tests.
- [x] Task 4: SNN-only BN/add domain and shape adapter. Existing Phase 1 domain compatibility validation already accepts structural producers; no relaxation was needed. Added verification of the declared main-branch weighted owner.
- [x] Task 5: existing element_sum and digital LIF execution reused; domain propagation and ownership bridge added. Fixed sequence-qualified cross-round input resolution, discovered by real backend integration testing.
- [x] Task 6: V7 one-batch pretrain and QAT, one-sample Original/PIM CLI evaluation, and full T=30 hardware CLI report succeeded. Targeted ANN/SNN regressions passed (113 passed, 5 skipped before adding the final split-array test parametrization).

Implementation clarifications supersede the corresponding assumptions below:

1. Branch code scales need not be equal in accuracy/QAT: tensors are dequantized real values. Binary identity plus signed BN output is a legal SpikingJelly residual. Missing domains and mismatched shapes are rejected; no raw unequal-scale integer addition is introduced.
2. LIF metadata is also required for `conv -> BN -> LIF`, not just `element_sum -> LIF`. The adapter handles both, using the actual BN/post-add quantizers.
3. Main-branch reducer ownership is explicit and validated, rather than combining ownership counts from both branches.
4. Performance support is the existing digital-LIF backend. BN/requantization arithmetic retains the original MNSIM accounting scope and is explicitly marked uncharacterized in reports. Residual merge uses the existing digital `element_sum` structural operator.
5. The end-to-end smoke uses a newly generated one-batch checkpoint, not a pre-existing trained ResNet checkpoint. It proves execution, not model accuracy; no long training job was started.

See `docs/specs/2026-10-03-residual-snn-runtime.md` for topology adaptations, usage, accounting limits, and artifact locations. The original step checklists below are retained as the design history; task completion and deviations are recorded above.

## Global Constraints

- Do not change ANN `NetworkGraph.forward()`, ANN model definitions, or ANN metric formulas to accommodate the SNN model.
- Reuse `conv`, `bn`, `element_sum`, `lif`, and `input_index_list`; do not add a residual-specific hardware layer type.
- Make SNN LIF placement explicit. Do not apply the current automatic LIF-after-every-convolution insertion rule to residual blocks.
- Keep each residual merge timestep-aligned and shape-compatible. Fail clearly on invalid branch shape or unsupported quantization-domain combinations.
- Reuse existing post-add activation quantization and structural-node accounting where their contracts match. Add SNN-specific metadata/bridging only where the current direct weighted-layer-to-LIF assumption does not apply.
- Do not alter unrelated running training jobs or their output directories during implementation.
- Complex multi-branch networks, `concat`, and SEW-ResNet ADD/AND/OR are out of scope.

## Task 1: Specify and Test the Residual Graph Contract

**Files:**
- Create: `tests/test_v7_residual_snn_network.py`
- Modify: `tests/test_v7_sequential_snn_networks.py` only for shared SNN builder assertions

**Interfaces:**
- Consumes the new `resnet18_snn_v7_phase6` model category and its `NetworkGraph` metadata.
- Produces deterministic graph assertions for identity and projection residual blocks.

- [ ] **Step 1: Add a failing graph-structure test.** Assert each basic block has `conv -> bn -> lif -> conv -> bn -> element_sum -> lif`, and that `element_sum` references the output of the second BN plus the block input or projection BN.
- [ ] **Step 2: Cover downsampling blocks.** Assert spatial downsampling uses stride convolution on the main path and a `1x1` stride projection on the shortcut; no pooling nodes appear.
- [ ] **Step 3: Assert V7 model contract.** Check six input channels, eight classes, sequence length 30, and all graph references point to earlier nodes.
- [ ] **Step 4: Confirm the tests fail for the current builder** without changing implementation.

## Task 2: Add Explicit SNN Residual Model Construction

**Files:**
- Modify: `MNSIM/Interface/network.py`
- Test: `tests/test_v7_residual_snn_network.py`

**Interfaces:**
- Adds `get_net(cate="resnet18_snn_v7_phase6", num_classes=8)` construction.
- Keeps the existing ANN `resnet18` branch and automatic ANN BN insertion unchanged.

- [ ] **Step 1: Add a narrowly scoped residual SNN builder.** Build the ResNet18 basic blocks using existing layer configuration types and relative input indices.
- [ ] **Step 2: Mark LIF placement explicitly.** Insert LIF after the first BN in each block and after each `element_sum`; omit a LIF between the second BN and merge.
- [ ] **Step 3: Reuse existing downsampling constraints.** Use stride convolution for stage changes and a `1x1` stride projection shortcut; avoid pooling.
- [ ] **Step 4: Isolate automatic layer insertion.** Ensure the generic SNN conv-insertion pass does not add duplicate LIF nodes to this model category. Preserve its behavior for LeNet, AlexNet, VGG, and ANN models.
- [ ] **Step 5: Run the graph-contract tests** and the existing sequential SNN builder tests.

## Task 3: Extend Temporal Accuracy and QAT to Structural Multi-Input Nodes

**Files:**
- Modify: `MNSIM/Interface/network.py`
- Modify: `MNSIM/Interface/quantize.py` only if the existing `element_sum` layer needs an input-count generalization
- Test: `tests/test_v7_residual_snn_network.py`
- Test: `tests/test_snn_original_accuracy.py`

**Interfaces:**
- Extends `sequence_accuracy_forward()` and `sequence_qat_forward()` to route multiple graph inputs into structural layers.
- Leaves ordinary ANN `forward()` semantics unchanged.

- [ ] **Step 1: Add failing temporal tests** using deterministic `[B,T,C,H,W]` inputs for identity and projection blocks; verify every merge combines values from the same timestep.
- [ ] **Step 2: Generalize graph input gathering.** Use `_indexed_graph_value()` for tensor and quantization-state streams rather than rejecting every multi-input node.
- [ ] **Step 3: Execute structural merges over temporal tensors.** Preserve `[T,B,...]` through the merge, use channel dimension 2 for any future temporal concat utility only if needed by shared helpers, and keep this task limited to elementwise sum.
- [ ] **Step 4: Define and validate residual merge quantization.** Reuse the existing post-add activation quantizer. Check branch tensor shape and quantization states before summation; do not silently rescale incompatible branches. Return the configured post-add state to the following LIF/weighted node.
- [ ] **Step 5: Preserve recurrent behavior.** Reset LIF state at independent sequence boundaries, retain multi-step LIF execution, sum final timestep logits as the existing V7 path does, and preserve gradients through QAT.
- [ ] **Step 6: Test ORIGINAL and FIX_TRAIN paths.** Assert finite outputs, expected logits shape, successful backward propagation on QAT, and a failure with a clear message for incompatible merge states.

## Task 4: Attach Post-Merge SNN Domain Metadata in Phase 1

**Files:**
- Modify: `MNSIM/Scheduling/phase1/accumulator_metadata.py`
- Modify: `MNSIM/Scheduling/phase1/round_planner.py`
- Modify: `MNSIM/Scheduling/phase1/mnsim_frontend.py` only if the current normalized structure drops required SNN merge metadata
- Test: `tests/test_round_snn_phase1.py`
- Test: `tests/test_round_phase1_real_networks.py`

**Interfaces:**
- Extends SNN metadata attachment to cover `weighted -> element_sum -> lif`.
- Retains the current `weighted -> lif` contract for sequential SNNs.

- [ ] **Step 1: Add failing ModelIR tests** for a residual merge followed by LIF, including both identity and projection shortcuts.
- [ ] **Step 2: Define the merge output domain from actual post-add quantization.** Use the trained/calibrated `element_sum` activation bit width and range as the authoritative merge-to-LIF numerical contract. Record source domains and merge policy as metadata. Reject missing or incompatible checkpoint/calibration state; do not invent a range.
- [ ] **Step 3: Compile LIF constants from the merge output domain.** Attach the same domain to the `element_sum` output contract and its following LIF, with neuron count derived from the merged output shape.
- [ ] **Step 4: Relax Phase 1 validation narrowly.** Permit LIF to consume a supported `element_sum` producer when its compiled domain matches the merge output domain; continue rejecting unsupported producer types and missing metadata.
- [ ] **Step 5: Verify the round plan.** Assert residual edges, weighted projection resources, merge structural nodes, LIF recurrent IDs, and sequence length 30 are retained in the plan.
- [ ] **Step 6: Re-run existing ANN ResNet18 Phase 1 tests** to confirm its plan and resource counts remain unchanged.

## Task 5: Bridge Merge-Output Ownership into the Existing Phase 2 LIF Executor

**Files:**
- Modify: `MNSIM/Scheduling/phase2/executor.py`
- Modify: `MNSIM/Scheduling/phase2/lowerer.py` only if the current boundary contract cannot describe the merge-produced LIF input
- Modify: `MNSIM/Scheduling/phase2/types.py` only if an additive typed metadata field is required
- Test: `tests/test_round_snn_executor.py`
- Test: `tests/test_round_phase2_real_networks.py`

**Interfaces:**
- Keeps `execute_round_plan()` and the digital/analog LIF cost models as the execution owners.
- Resolves the output-neuron ownership for a LIF after `element_sum` without treating the merge as a crossbar layer.

- [ ] **Step 1: Add a failing end-to-end structural test** whose graph is weighted main path plus identity/projection shortcut, then `element_sum`, then LIF.
- [ ] **Step 2: Trace the merge's weighted producers and validate geometry.** Require both branches to describe the same merged output neuron region. Determine a non-duplicating ownership policy for the merge result; reject ambiguous ownership rather than summing both branch neuron counts.
- [ ] **Step 3: Reuse the existing structural execution step** for `element_sum`, including its boundary inputs, output shape, and existing operation accounting. Do not add a residual-specific cost formula.
- [ ] **Step 4: Feed the merge output domain and resolved ownership to the existing LIF cost path.** Preserve invocation ordering, state read/write accounting, and boundary commits.
- [ ] **Step 5: Assert T invocations per LIF** and verify `T=30` produces the expected timestep stream, merge boundaries, and LIF state progression.
- [ ] **Step 6: Run existing SNN executor, digital LIF, partial accumulator, and ANN ResNet18 Phase 2 tests.** Existing ANN residual metrics and resource planning must remain unchanged.

## Task 6: End-to-End Accuracy and Hardware Verification

**Files:**
- Modify: `tests/test_round_orchestrated_runner.py` only for additive SNN residual runner coverage
- Modify: `README.md` or add a short model note if the supported model invocation needs documentation

- [ ] **Step 1: Run focused tests** for the residual model builder, temporal ORIGINAL/QAT execution, Phase 1 metadata, and Phase 2 merge-to-LIF execution.
- [ ] **Step 2: Run the existing ANN ResNet18 regression suite.** Confirm model graph, checkpoint compatibility, round plan, area/latency/energy accounting, and report schema are unchanged.
- [ ] **Step 3: Run one V7 software-accuracy smoke** using a pretrain checkpoint and test split; check finite logits and expected class count.
- [ ] **Step 4: Run one SNN performance smoke** through the round-orchestrated wrapper at sequence length 30; inspect `round_plan.json` and `phase2_report.json` for every residual sum, LIF invocation, and modeled timestep.
- [ ] **Step 5: Run the complete relevant test set**: `tests/test_v7_sequential_snn_networks.py`, `tests/test_v7_residual_snn_network.py`, `tests/test_round_snn_phase1.py`, `tests/test_round_snn_executor.py`, `tests/test_round_phase1_real_networks.py`, and `tests/test_round_phase2_real_networks.py`.
- [ ] **Step 6: Inspect the diff** and confirm no ANN behavior, existing user artifacts, or unrelated training outputs were changed.

## Acceptance Criteria

1. `resnet18_snn_v7_phase6` follows the SpikingJelly BasicBlock placement: first conv/BN/LIF, second conv/BN, add shortcut, then LIF.
2. Identity and projection residuals execute at matching timesteps and preserve the V7 input/output contract.
3. Software accuracy and `FIX_TRAIN` QAT handle the residual graph with explicit merge shape and quantization validation.
4. Phase 1 accepts a merge-produced LIF only when its post-add domain and compiled constants are valid.
5. Phase 2 executes and accounts for the existing `element_sum` and LIF at all 30 timesteps, with unambiguous merge-output ownership.
6. Existing ANN ResNet18 graph, plan, checkpoint path, and performance metrics pass unchanged.
7. No residual-specific hardware node or duplicated residual cost formula is introduced.
