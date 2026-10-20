# 原 MNSIM SNN QAT 与 8-bit 权重部署设计

日期：2026-08-15

状态：QAT独立任务；PIM profile语义从canonical master-SNN spec引用

## 1. 文档目的

本文只定义 round-orchestrated MNSIM-2.0 的 SNN accuracy、QAT、PIM numerical
validation 和权重部署导出。它解决第一个独立任务：

```text
software SNN checkpoint
-> original MNSIM quantization-aware training
-> 9-bit signed logical weight
-> differential 8-bit device codes
-> reproducible SNN Processing-in-Memory Accuracy acceptance
```

真实电阻、电导、电流、ADC/LUT 硬件架构、输入闭合、时延、能耗和面积属于
第二个任务，由下列文档定义：

```text
2026-08-15-master-snn-functionality-integration-design.md
```

两个任务通过只读差分权重码和 activity trace 对接，不能共享数值前向或
variation实例。master + SNN融合任务不等待QAT完成：当前可以从现有checkpoint
经原MNSIM量化接口生成相同的权重码；QAT完成后只替换权重码来源。

## 2. 与已有设计的关系

本文保留原 MNSIM-2.0 的核心 accuracy 链：

```text
Quantize / bit weights
-> Weight_update
-> QuantizeLayer.set_weights_forward(...)
-> profile-selected PIM numerical peripheral action
-> shared SpikingJelly LIF
-> logits / spikes / accuracy
```

`profile-selected PIM numerical peripheral action` 的最终定义只来自
`2026-08-15-master-snn-functionality-integration-design.md`：ADC profile使用正式
ReADC，LUT profile跳过ADC并使用合格的frontend/LIF扰动模型。本文不得建立
第二套冲突规则。

本文不继续使用 Route-A current forward、模拟 frontend 或硬件 RC LIF 作为
正式 accuracy 或训练目标。Route-A 代码只能作为历史对照，不能成为本文的
默认 checkpoint、forward mode 或 loss 来源。

## 3. 固定决策

以下内容是正式合同，不是实现选项。

1. 正式增加原 MNSIM-2.0 SNN QAT 能力。
2. 网络量化配置固定使用 `weight_bit=9`。
3. `weight_bit=9` 表示 1 bit 逻辑符号和 8 bit 幅值，不表示9-bit器件。
4. `Xbar_Polarity=2` 时，正、负轨分别写入 `uint8 [0, 255]`。
5. QAT、PTQ 和 PIM validation 不得调用真实电流、frontend LUT 或 LIF LUT。
6. QAT 使用数字 LIF 的硬前向和 surrogate 反向，不使用 Route-A 硬件 LIF。
7. LIF 输出的二值 spike 必须拥有独立量化域，不能继承前一层 ADC 输出 scale。
8. 训练 checkpoint 和可烧写部署 artifact 是两个不同产物。
9. `Device_Seed=112` 只用于可复现验收，不得作为 variation-aware QAT 的唯一
   训练实例。
10. 对外只报告一个 `SNN Processing-in-Memory Accuracy`，不得按ADC/LUT创建
    新accuracy名称。

## 4. 当前代码基线与缺口

可以复用的原 MNSIM 能力包括：

```text
QuantizeFunction fake quantization
straight-through weight gradient
per-layer weight scale
get_bit_weights differential export logic
set_weights_forward PIM numerical validation
activation calibration state
```

但当前代码尚未完成本文要求的 SNN QAT，至少有三个已确认缺口。

### 4.1 `ORIGINAL` 仍是 Route-A forward

当前 `NetworkGraph.sequence_accuracy_forward(..., weighted_mode="ORIGINAL")`
调用 Route-A conductance/current/frontend 和 Route-A LIF。它不再代表本文所需的
`SNN Original Accuracy Path`，必须从正式 accuracy mode 中移除或明确标记为
legacy。

### 4.2 数字 LIF 没有 surrogate gradient

当前普通 `LIFNode` 使用硬比较产生 spike，反向无法穿过阈值。直接删除 Route-A
后运行现有 `FIX_TRAIN` 不能形成有效的跨层 SNN QAT。必须新增与数字 LIF 硬
前向一致的 surrogate 训练实现。

### 4.3 LIF 后量化域错误继承

当前 LIF 输出 `0/1` spike 后，量化元数据仍继承前一层 ADC/activation domain。
下一层 `set_weights_forward` 会按照错误的 bit/scale 重新编码 spike。正式 QAT
和 PIM validation 前必须分离二值 spike domain 与 ADC accumulator domain。

旧 `MNSIM/Interface/train.py` 只是通用图像分类训练壳，还包含固定多卡和固定
超参数，不是本文要求的可复现 SNN QAT 入口。

## 5. 正式 Accuracy 执行模式

SNN accuracy 必须使用三个名称和语义明确的模式。

| Mode | Weighted layer | LIF | 用途 |
| --- | --- | --- | --- |
| `FLOAT` | 浮点卷积/全连接 | 数字 LIF硬前向 | `SNN Original Accuracy Path` |
| `QAT` | 原 MNSIM fake quant + STE | 相同硬前向、surrogate反向 | 训练与fake-quant验证 |
| `PIM` | 部署码 + `set_weights_forward` + master-SNN spec定义的profile numerical branch | master-SNN spec定义的SpikingJelly前向 | `SNN Processing-in-Memory Accuracy Path` |

三个模式必须使用相同的网络拓扑、timestep顺序、样本边界、分类readout和
名义`tau/v_th/v_reset`。LUT-profile PIM可以按canonical master-SNN spec从
合格artifact采样frontend/LIF扰动；除此之外不得建立隐式神经元差异。

`adc_digital_lif` 和 `frontend_lut_lif_lut` 不是新的accuracy mode，二者都报告
`PIM`。但profile必须按master-SNN spec选择ReADC或无ADC+LIF扰动数值链，因此
允许改变PIM artifact；`FLOAT`和`QAT` artifact不得因硬件profile改变。

## 6. 9-bit 逻辑权重与 8-bit 器件码

原 MNSIM 的 `weight_bit` 包含符号位：

```text
logical weight_bit = 9
logical sign bits = 1
logical magnitude bits = 8
threshold = 2^(9 - 1) - 1 = 255
logical signed code q in [-255, 255]
```

逐层对称量化固定为：

```text
scale_lsb = max(abs(W_float)) / 255
q = clamp(round(W_float / scale_lsb), -255, 255)
W_fake_quant = q * scale_lsb
```

差分部署固定为：

```text
q_pos = max(q, 0)   in [0, 255]
q_neg = max(-q, 0)  in [0, 255]
q = q_pos - q_neg
```

`q_pos` 和 `q_neg` 分别写入一个 256 级 8-bit cell。正式配置必须满足：

```text
MNSIM network weight_bit = 9
device magnitude bits = 8
Device_Level = 256
Xbar_Polarity = 2
```

不得把网络 `weight_bit` 直接改成8。按原 MNSIM 定义，那会得到
`[-127, 127]`，只使用一半器件幅值范围。

PTQ、QAT、部署导出和重载验证必须使用同一种舍入规则。第一版固定沿用
PyTorch `torch.round` 的 round-to-nearest-even；报告必须记录 rounding mode。

## 7. SNN 数值域合同

正式实现至少区分三个域。

### 7.1 Weight domain

```text
float shadow weight
signed logical code q
positive/negative device code q_pos/q_neg
per-layer scale_lsb
```

### 7.2 Weighted-output domain

ADC profile使用ADC accumulator domain，描述ReADC numerical code、scale、
partial-sum和数字累加位宽。LUT profile使用无ADC reconstructed weighted-output
domain：bit-sliced结果恢复到软件数值单位后直接进入frontend/LIF扰动模型，
不得构造ADC domain或执行输出activation quantization。两者均以master-SNN
spec为最终定义。

### 7.3 Binary spike domain

该域描述数据集输入事件和 LIF 输出：

```text
allowed value = 0 or 1
logical scale = 1
signed = false
source = dataset event | LIF spike
```

不能使用当前对称量化公式中的 `activation_bit=1` 代表 binary spike，因为
`2^(1-1)-1=0`。第一版应增加显式 `BinarySpikeDomain` 或等价类型，而不是伪造
一个带符号 activation bit width。

LIF 必须输出 binary spike domain；下一 weighted layer 必须直接将 `0/1` 编为
输入码，不能继承 LIF 之前的 ADC scale。

## 8. SNN QAT 前向与梯度合同

`QAT` 模式必须处理 `[B,T,C,H,W]` sequence batch。允许将同一 weighted layer
的 `T*B` 输入合并后执行卷积/全连接，但 LIF 必须按 timestep顺序更新状态。

Weighted layer 前向使用原 MNSIM fake quantization：

```text
float shadow weight
-> qbit=9 fake quant
-> differentiable conv/linear
-> deployment-consistent profile peripheral surrogate
```

反向对weight rounding/clamp使用STE。QAT若训练ADC-profile surrogate，必须与
正式ReADC码域一致；若训练LUT-profile surrogate，必须遵守master-SNN spec的
无ADC weighted-output和合格LIF扰动合同。每个候选checkpoint仍必须分别通过
不可求导的正式profile PIM branch做最终验证。

QAT 不得把 `set_weights_forward` 的不可导 numpy/device-code路径伪装成训练前向，
也不得只验证 `FIX_TRAIN` accuracy 就宣称部署精度成立。

## 9. 数字 LIF Surrogate 合同

QAT的名义数字LIF前向必须与`FLOAT`名义前向bit-exact：

```text
v_pre = v + (x - v) / tau
spike_hard = H(v_pre - v_th)
v_next = v_pre * (1 - spike_hard) + v_reset * spike_hard
```

仅在反向中用 atan surrogate 替换阶跃导数。ADC-profile PIM使用相同名义
SpikingJelly参数；LUT-profile PIM允许按master-SNN spec施加合格的frontend/LIF
扰动，因此不要求与名义QAT/FLOAT spike bit-exact。允许复用 Route-A 已验证的 atan
数学形式，但不得复用 `TrainableRouteALIFNode` 的 RC、`t_fire`、电压或模拟
frontend 前向。

状态生命周期固定为：

1. 每个独立样本开始前清空全部 LIF state；
2. 同一序列的所有 timestep按顺序执行；
3. QAT 默认保留跨 timestep BPTT，不静默 detach state；
4. batch 内不同样本拥有独立 state row；
5. evaluation 前后不得遗留上一 batch 或上一模式的 state。

## 10. 正式 PTQ/QAT 流程

正式流程固定为：

```text
non-Route-A software SNN checkpoint
-> FLOAT full-evaluation baseline
-> deterministic 9-bit PTQ baseline
-> nominal MNSIM SNN QAT when required
-> fake-quant candidate evaluation
-> differential deployment export
-> nominal PIM validation
-> 20 ns / 30 ns variation validation
```

代码必须提供 QAT能力。具体模型是否必须执行 QAT，由 PTQ 精度门槛决定；不能
在未运行 PTQ 的情况下假定 QAT必需或不必需。

正式初始 checkpoint 必须是 Route-A fine-tune 前的软件 SNN checkpoint。若该
checkpoint 不存在，应重新训练或恢复软件 SNN 基线，不能静默把 Route-A
fine-tuned checkpoint 当成 `SNN Original Accuracy Path` 的初始 checkpoint。

训练入口必须显式记录数据集、split、sequence length、batch size、optimizer、
learning rate、epoch、训练 seed、量化配置、初始 checkpoint checksum 和代码
版本。禁止沿用旧训练壳中的固定 GPU 列表或未报告的环境默认值。

## 11. Checkpoint 与部署 Artifact

QAT checkpoint 至少保存：

```text
floating-point shadow weights
per-layer weight scale
activation/ADC calibration state
digital LIF parameters
weight_bit = 9
quantizer and rounding version
optimizer/scheduler state for resumable checkpoints
```

部署导出器必须生成：

```text
per-layer scale_lsb
logical signed integer code q in [-255, 255]
positive uint8 code q_pos in [0, 255]
negative uint8 code q_neg in [0, 255]
layer/mapping identity
shape/dtype metadata
source checkpoint checksum
artifact checksum
```

部署 artifact 必须满足 `q == q_pos-q_neg`，并能重载后 bit-exact 重建所有逻辑
整数码。只保存浮点 shadow weights 不能称为已经导出8-bit权重。

## 12. SNN Processing-in-Memory Accuracy Variation

正式器件参数固定为：

| Device model | `Device_Variation` | `Device_Seed` |
| --- | ---: | ---: |
| `rram_20ns_linear_fit` | `6.69903802061613%` | `112` |
| `rram_30ns_linear_fit` | `7.725782066255718%` | `112` |

`SNN Processing-in-Memory Accuracy Path` 使用原 MNSIM
`Weight_update -> set_weights_forward`，不调用真实
电流硬件后端：

```text
R_nom = Device_Resistance[code]
epsilon_accuracy ~ Normal(0, Device_Variation / 100)
R_accuracy = R_nom * (1 + epsilon_accuracy)
G_accuracy = 1 / R_accuracy
code_equiv = (G_accuracy - G_HRS) / (G_LRS - G_HRS) * 255
```

`code_equiv_pos-code_equiv_neg` 进入原MNSIM数字卷积/全连接，再由master-SNN
spec选择ReADC或无ADC numerical branch。这里不计算或传递
`Ipos/Ineg/Idiff`。

采样生命周期固定为：

1. 每次 `SNN Processing-in-Memory Accuracy Path` 运行开始、取得 bit weights
   后采样一次；
2. 正负阵列、不同 cell 和不同 weight slice 独立；
3. 同一次运行所有样本、batch 和 timestep复用；
4. 不按 physical round 重新采样；
5. 相同模型、映射、器件和 seed 必须可复现。

20 ns和30 ns用相同标准随机序列做配对比较，再分别乘各自 sigma。它不表示
两种器件共享同一物理实例。

Variation-aware QAT 是 nominal QAT 后的可选增强。训练必须跨多个 seed/实例
采样，并覆盖两组正式 sigma；不得固定只训练 seed 112。最终仍分别执行20 ns
和30 ns seed 112验收，并报告多 seed mean/std/worst-case。

## 13. 与硬件性能任务的隔离接口

QAT/accuracy任务只向硬件性能任务提供：

```text
immutable deployment codes and scales
checkpoint/deployment checksum
read-only input/spike activity trace
layer/timestep/sample identity
```

硬件性能任务不得返回或修改：

```text
QAT gradients
float shadow weights
logical/device codes
digital LIF state
logits/spikes/accuracy RNG
```

切换ADC/LUT profile允许按master-SNN spec改变`PIM` artifact，但不得改变
`FLOAT/QAT` artifact。真实电流运行结果、hardware energy/latency参数和
hardware-performance seed不得回灌或改变任何accuracy artifact。

## 14. 报告合同

每次正式 accuracy运行至少记录：

```text
accuracy_mode = FLOAT | QAT | PIM
accuracy_compute_model = original_mnsim
accuracy_hardware_coupling = profile_selected_numerical_only
hardware_profile
accuracy_adc_model = READC_LLOYD_MAX | none
accuracy_lif_perturbation_profile = none | qualified_profile_id
source_checkpoint_checksum
deployment_artifact_checksum
logical_weight_bit = 9
device_magnitude_bit = 8
weight_code_range = [-255, 255]
device_code_range = [0, 255]
weight_scale_scope
rounding_mode
binary_spike_domain_version
digital_lif_parameters
dataset/split/sample_count/sequence_length
accuracy_device_model
accuracy_device_variation_percent
accuracy_device_seed
accuracy_variation_sampling_scope
```

PTQ、QAT fake-quant 和 PIM validation结果必须分开命名，不能都写成一个
`quantized_accuracy`。

## 15. 禁止行为

1. 用 Route-A current/frontend/LIF fine-tune 代替原 MNSIM SNN QAT。
2. 将正式网络 `weight_bit` 改成8并误报为完整256级差分部署。
3. 只保存浮点 checkpoint，却宣称已经生成8-bit器件权重。
4. QAT 和部署导出使用不同 scale、rounding 或 clamp 规则。
5. LIF 输出 spike 继承前一 ADC domain 的 scale。
6. 用 `activation_bit=1` 套用会产生零阈值的带符号量化公式。
7. `QAT` 前向或 loss 使用真实 `R/G/Ipos/Ineg/Idiff`。
8. 只测试 fake-quant forward，不执行正式 PIM validation。
9. variation-aware QAT只训练 seed 112。
10. 按 sample、batch、timestep 或 physical round 重采样同一次 accuracy运行的
    variation weights。

## 16. 验收测试

### 16.1 执行模式

1. `FLOAT` 和 `QAT` 注入一个调用即失败的 Route-A/current backend，均必须
   完成前向；QAT还必须完成反向。
2. `PIM` 只允许调用原 MNSIM numerical path，不允许调用 hardware current。
3. 相同数字输入下，QAT surrogate LIF前向与普通数字 LIF bit-exact。
4. 每个独立样本和 batch 后验证 LIF state 生命周期。

### 16.2 权重与部署导出

1. QAT 阈值为255，逻辑码覆盖 `[-255,255]`。
2. 所有码满足 `q=q_pos-q_neg`，正、负码均为 `uint8 [0,255]`。
3. PTQ、QAT、export和PIM validation使用相同逐层 scale/rounding。
4. 导出、重载后逻辑码必须bit-exact；在相同profile、器件、ReADC/LIF
   artifact和seed下，重载前后的PIM输出必须bit-exact。
5. artifact checksum、shape、dtype和层映射缺失时 fail closed。

### 16.3 数值域与SNN梯度

1. LIF输出必须标记 binary spike domain，不能保留 ADC scale。
2. 下一 weighted layer 对 spike 0/1的输入码必须手算可验证。
3. 一个小型两层 SNN 的 loss backward 必须为第一层权重产生有限非零梯度。
4. T个 timestep的状态顺序、reset和readout必须与
   `SNN Original Accuracy Path` 一致。

### 16.4 Accuracy 与 Variation

1. 报告 `SNN Original Accuracy`、nominal PTQ、nominal QAT 和
   `SNN Processing-in-Memory Accuracy`。
2. 分别报告20 ns/30 ns seed 112 accuracy。
3. 相同 seed重复运行必须可复现。
4. 同一运行跨样本和 timestep不得重采样。
5. variation-aware QAT必须报告多 seed mean/std/worst-case。
6. 固定 seed 112 的 variation-aware训练配置必须被拒绝。

## 17. 实现顺序

1. 将正式 sequence accuracy mode改为 `FLOAT/QAT/PIM`，隔离 legacy Route-A。
2. 建立 binary spike、ADC accumulator和weight domain类型及测试。
3. 实现数字 LIF硬前向一致的 surrogate反向和SNN BPTT测试。
4. 建立非Route-A软件 SNN FLOAT全量基线。
5. 建立确定性9-bit PTQ和完整 accuracy基线。
6. 建立可恢复、可审计的 SNN QAT入口。
7. 建立 `q/q_pos/q_neg` 部署导出和重载验证。
8. 接入 nominal PIM validation。
9. 接入20 ns/30 ns accuracy variation和多 seed审计。
10. 达到正式门槛后，将deployment artifact接入已独立完成的master + SNN
    融合任务，并验证它与现有checkpoint权重码适配器使用同一数据合同。

## 18. 实现前待确认项

1. Route-A fine-tune前的软件 SNN checkpoint路径、checksum和原始 accuracy。
2. 正式训练/验证数据集、split和类别数。
3. PTQ/QAT相对 FLOAT允许下降的 accuracy门槛。
4. nominal PIM相对QAT fake-quant允许下降的门槛。
5. variation多 seed数量、seed集合和worst-case判据。
6. 第一版是否只做 nominal QAT，待 variation结果后再决定 variation-aware QAT。
7. QAT训练时使用何种profile surrogate不在本文拍板；正式PIM acceptance必须
   分别调用master-SNN spec冻结的ADC/LUT numerical branch。

这些待确认项不得改变9-bit逻辑权重、8-bit差分器件码、数字 LIF surrogate和
accuracy/hardware隔离合同。
