# Complex-Branch SNN Design

## Scope

Complex branches are represented as an ordered DAG of existing MNSIM nodes.
Convolution and fully connected nodes keep their normal MNSIM weights and
crossbar mapping. LIF stays an explicit temporal node. Merge nodes are
structural graph operations, not new crossbar layer types.

The first model is `inception_snn_v7_phase6`, an Inception-lite/SWMR-style V7
classifier. Three convolution branches use 1x1, 3x3, and 5x5 kernels; each
branch ends in LIF. Their spikes are concatenated on the channel axis at the
same timestep and fused by a 1x1 convolution. Spatial reduction uses stride
convolutions. The graph contains no pooling.

## Merge Contract

`NetworkGraph.input_index_list` is the DAG edge list. A merge waits for every
listed producer. Sequence execution requires identical sample and timestep
axes, so a merge never combines values from different timesteps.

`concat` accepts any number of inputs and records its axis, same-timestep
policy, output shape, and disjoint flattened channel ownership intervals.
For an NCHW feature map, interval width is `C * H * W`; each branch owns one
contiguous output interval. Concat routes the existing binary spike values and
does not create an arithmetic accumulator or duplicate branch ownership.

`element_sum` accepts any number of equal-shaped inputs and rejects implicit
broadcasting. `element_multiply` accepts equal-shaped inputs and preserves
MNSIM's existing channel-gate form, where a `[B,C]` gate multiplies a
`[B,C,H,W]` feature map. Existing residual graphs retain their digital
element-sum behavior and their existing single-owner metadata.

## Accuracy and Training

The ordinary graph forward, sequence accuracy path, sequence QAT path,
CNNParted path, and shape export all consume the same arbitrary-arity inputs.
Concat joins `[T,B,C,H,W]` values along `C` and propagates the shared binary
spike domain. Its inputs must agree in shape on every other axis. The V7
pretraining and finetuning CLIs accept the model name.

## Performance Path

Phase 1 retains concat as a structural non-weighted node. Its metadata marks
binary spike values, same-timestep alignment, channel ownership intervals,
and 1-bit input/output values. Downstream weighted layers therefore expose
the correct binary input width.

Phase 2 lowers concat to a zero-crossbar structural step. The static evaluator
models full-feature-map buffer read/write traffic and routing latency; concat
has no adder, multiplier, or crossbar cost. This keeps branch joins visible
in ordered round execution without adding a hardware-specific concat layer.

## Literature Basis

- Fang et al., “Deep Residual Learning in Spiking Neural Networks,” NeurIPS
  2021, [SEW-ResNet](https://arxiv.org/abs/2102.04159): explicit spike-domain
  element-wise merge operations.
- Meng et al., “Spiking Inception Module for Multi-Layer Spiking Neural
  Networks,” IJCNN 2020, [paper](https://arxiv.org/abs/2001.10696):
  split-process-concatenate branches.
- Zhang et al., “SpikeFPN: Scale Equivariant Object Detection with
  Spiking Neural Networks,” 2024, [paper](https://arxiv.org/abs/2307.12900):
  multi-input DAG nodes and aligned branch aggregation.
- Liu et al., “SNN-SWMR: Spiking Neural Network with Spatial-wise Multi-branch
  Representation,” 2025, [paper](https://arxiv.org/abs/2409.11619):
  multi-kernel branches followed by LIF and channel concatenation.

These papers motivate topology and spike merge semantics. They do not define
the MNSIM performance costs; those remain explicit in this project's existing
buffer, crossbar, and digital-peripheral models.
