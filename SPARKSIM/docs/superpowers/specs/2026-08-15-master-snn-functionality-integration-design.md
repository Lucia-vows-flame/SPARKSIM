# 原 MNSIM master + SNN 功能融合设计

日期：2026-08-15

状态：Canonical，用户决策已冻结

## 1. 文档目的

本文定义第二个独立任务：把既有 `master + SNN` 的必要功能迁移并融合到
round-orchestrated MNSIM-2.0。

项目核心仍然是原 MNSIM-2.0。目标不是建立一套与 MNSIM 并列的专用
`master + SNN` 仿真器，也不是把旧 scheduler整体复制进来，而是在现有 MNSIM
网络、量化、映射、面积、时延和能耗框架中迁移 sequence、状态、真实电流、
输入闭合和可配置 SNN 硬件路径。

本设计固定以下核心要求：

1. 保留原 MNSIM 的 ADC + 数字 LIF 路径，并新增 frontend LUT + LIF LUT
   路径；两者都是正式可选硬件架构。
2. 逐物理cell的真实电流硬件后端只服务于硬件性能建模，绝不能把其
   `G/Ipos/Ineg/Idiff` tensor或运行时状态送入
   `SNN Processing-in-Memory Accuracy Path`。PIM accuracy仍允许在原MNSIM
   `Weight_update`位置读取锁定电阻表并生成等效数值码。
3. 对外只报告一个 `SNN Processing-in-Memory Accuracy`；不得为 ADC、LUT、
   无 ADC 或扰动分支创造新的 accuracy 名称。
4. 硬件 profile 同时确定 PIM accuracy 中是否存在 ADC numerical action：
   `adc_digital_lif` 固定使用 ReADC，`frontend_lut_lif_lut` 固定跳过 ADC。
5. LUT profile 的 accuracy 神经元仍由 SpikingJelly 执行，但必须施加由本项目
   frontend/LIF 电路 Monte Carlo/PVT 表征得到的扰动；不得使用任意经验噪声。

第二项是本设计最高优先级的不变量。后续实现、测试、报告和文档若与它
冲突，以本文为准。

## 2. 与已有设计的关系

本文扩展而不废除：

```text
docs/superpowers/specs/
  2026-08-12-snn-digital-lif-round-execution-design.md
```

2026-08-12 设计继续定义 `ADC + digital LIF` 架构的执行语义。本文新增第二种
`frontend LUT + LIF LUT` 架构，并定义两种架构如何共享 MNSIM 基础设施。

SNN QAT、9-bit 带符号逻辑权重、8-bit 差分器件码和部署导出由独立文档定义：

```text
docs/superpowers/specs/
  2026-08-15-snn-qat-and-8bit-weight-deployment-design.md
```

QAT任务和本文的master + SNN融合任务可以独立实现，本文不等待QAT完成。
当前正式输入可以直接由现有checkpoint和原MNSIM `get_bit_weights()`生成；未来
QAT产生的部署artifact只需实现相同的差分权重码接口，即可替换当前权重来源。
本文不定义训练流程。PIM accuracy 的 profile选择、器件 variation、ReADC、
LUT 无 ADC 数值链和 LIF 扰动以本文为最终合同；QAT spec 只能引用，不能建立
与本文冲突的第二套 PIM validation 规则。

本文同时修正一个容易混淆的历史表述：

- 原 MNSIM 会读取 `Device_Resistance`，并可使用 `1/R` 估算阵列功耗；
- 原 MNSIM 没有把逐单元 `G+/G-` 和真实 `Ipos/Ineg` 接入通用网络推理；
- 当前仓库中的 `Conductance.py` 和 `snn_mvm_current.py` 是后续 Route-A 扩展，
  不是原 MNSIM 的通用硬件执行后端。

后续可以复用 Route-A 中经过测试的公式和向量化实现，但必须重新放入本文
定义的模块边界，不能继续把 Route-A accuracy forward 当作正式集成方式。

## 3. 固定术语

### 3.1 SNN Original Accuracy Path

`SNN Original Accuracy Path` 是纯数值 SNN 基线：

```text
checkpoint floating-point weights
-> PyTorch Conv2d / Linear
-> shared SpikingJelly LIF
-> timestep spike accumulation
-> SNN Original Accuracy
```

该路径不调用 Route-A 电导、电流、frontend 或硬件 RC LIF。Route-A 旧路径只能
作为显式 legacy mode 保留，不能继续被报告为 `SNN Original Accuracy`。

### 3.2 SNN Processing-in-Memory Accuracy Path

`SNN Processing-in-Memory Accuracy Path` 对外只有一个结果名称，但内部按所选
正式硬件 profile 执行两种数值链。

`adc_digital_lif`：

```text
quantized weights / bit weights
-> locked R-ratio + device variation + SAF contract
-> QuantizeLayer.set_weights_forward(...)
-> Hong 2025 ReADC numerical action
-> digital accumulation
-> shared SpikingJelly LIF
-> timestep spike accumulation
-> SNN Processing-in-Memory Accuracy
```

`frontend_lut_lif_lut`：

```text
quantized weights / bit weights
-> same locked R-ratio + device variation + SAF contract
-> original MNSIM bit-sliced weighted computation
-> restore the weighted result to software numerical units
-> no ADC / no ReADC / no output activation quantization
-> characterized frontend perturbation
-> characterized-perturbation SpikingJelly LIF
-> timestep spike accumulation
-> SNN Processing-in-Memory Accuracy
```

两种链都不得使用物理 `Ipos/Ineg/Idiff`、frontend LUT 查询结果、LIF LUT
`vmem_next/spike_flag` 或 hardware range clamp 作为 accuracy tensor。ADC/LUT
profile 可以得到不同的 PIM accuracy，这是 ADC 是否存在以及 LUT profile LIF
扰动合同不同的预期结果，不得要求二者 bit-exact 相同。

SNN Original Accuracy 与 PIM accuracy 必须共享网络拓扑、输入、timestep顺序、
状态 reset、readout 和 checkpoint。二者的差异只允许来自本文明确列出的
weighted-layer numerical backend、ReADC和LUT-profile LIF扰动。

### 3.3 共享 SpikingJelly LIF

两条 accuracy 路径必须复用同一个 sequence executor，并使用同一个
SpikingJelly LIF实现。无扰动基线固定使用：

```python
spikingjelly.activation_based.neuron.LIFNode(
    tau=layer_config["tau"],
    decay_input=True,
    v_threshold=layer_config["v_th"],
    v_reset=layer_config["v_reset"],
    step_mode="m",
    backend="torch",
)
```

两份正式 V7 VGG8-SNN 软件模型固定使用 `tau=50.5016666556`、`v_th=0.01`、
`v_reset=0.0`。
参数必须从 layer config 读取，不能在 accuracy executor 中另设一套隐式常量。
每个独立 batch 前必须调用
`spikingjelly.activation_based.functional.reset_net(network)`；同一样本的 timestep
之间不得 reset。输入 tensor 的正式布局为 `[T, B, ...]`，由 `step_mode="m"`
一次执行完整序列。

SpikingJelly 是 accuracy 的数值神经元执行器，不是 LUT 查询器。ADC profile
使用上述无扰动参数。LUT profile 仍调用同一执行器，但在每次正式运行开始时
加载并采样合格的 frontend/LIF perturbation profile；具体合同见第10节。

### 3.4 真实电流硬件模型

真实电流硬件模型指：

```text
device code
-> locked R(code)
-> physical G(code) = 1 / R(code)
-> differential G+ / G-
-> Ipos / Ineg / column load / cell read power
```

其逐cell物理输出只能用于硬件性能、工作范围和可实现性审计。Accuracy中的
R-ratio/variation只允许通过第7.1节定义的`Weight_update`等效码映射实现，不得
调用本硬件后端或传递真实列电流。

### 3.5 硬件输出路径

`hardware output path` 指 xbar 列输出之后实例化的硬件架构：

```text
ADC path: ReADC-compatible ADC/DAC peripherals + digital accumulation + digital LIF
LUT path: current frontend LUT + LIF LUT
```

它不是物理电流或 LUT 数值回灌 accuracy 的开关，但它必须选择第3.2节定义的
PIM accuracy numerical branch：ADC profile 使用 ReADC，LUT profile 跳过 ADC
并使用合格的 LIF perturbation profile。

### 3.6 差分权重码输入

master + SNN融合任务只消费规范化后的只读差分权重码：

```text
layer identity
q_pos in [0, 255]
q_neg in [0, 255]
scale and shape metadata
source identity/checksum
```

第一版必须支持两个互换来源：

```text
current source:
  existing checkpoint -> original MNSIM get_bit_weights()

future source:
  QAT deployment artifact -> validated q_pos/q_neg loader
```

两种来源进入hardware-performance平面前必须归一化为相同的不可变数据类型。
调度、电流、能耗和时延代码不得判断权重是否经过QAT，也不得因没有QAT
artifact而拒绝执行。

## 4. 三平面架构

正式实现必须把仿真器拆成三个职责明确的平面。

### 4.1 Accuracy 平面

```text
                              +-> PyTorch weighted ops ------------------+
checkpoint + sequence graph -|                                          |
                              +-> MNSIM PIM numerical weighted ops ------+
                                                                         v
                                                         shared SpikingJelly LIF
                                                                         |
                                                                         v
                                                    separate accuracy results
-> optional read-only activity trace
```

Accuracy 平面同时拥有 `SNN Original Accuracy Path` 和
`SNN Processing-in-Memory Accuracy Path`。二者共享 sequence executor，分别
拥有 logits、spike 和最终准确率，且每次执行独立初始化 LIF 状态。

### 4.2 Scheduling 平面

```text
ModelIR
-> RoundPlan
-> physical fragment placement
-> round / timestep / boundary ordering
```

Scheduling 平面拥有资源分配和执行顺序，不拥有 accuracy 数值。

### 4.3 Hardware-performance 平面

```text
RoundPlan + placement + read-only activity trace + device model
-> physical conductance/current evaluation
-> selected ADC or LUT hardware path
-> latency / energy / area / static-power / range audit
```

Hardware-performance 平面可以消费 accuracy 产生的 trace，但不能把运行时
物理电流、LUT 查询结果或硬件状态返回给 accuracy 前向。Accuracy 平面可以
只读消费经独立表征、版本化和校验的 LIF perturbation profile；该 profile
不是一次 hardware runtime 的输出。

允许的依赖方向只有：

```text
accuracy --read-only trace--> hardware performance
scheduling -----------------> hardware performance
device model ----------------> hardware performance
hardware profile config -----> PIM accuracy numerical branch
qualified LIF perturbation --> LUT-profile PIM accuracy
```

明确禁止：

```text
hardware current ------X----> accuracy layer output
hardware LUT ----------X----> accuracy membrane state
runtime hardware variation -X-> accuracy device/LIF perturbation
hardware range clamp --X----> logits or spikes
```

## 5. Accuracy 隔离合同

以下要求不可配置、不可绕过。

1. `snn_mvm_current.py` 或后续物理电流后端的输出不得作为
   `QuantizeLayer`、卷积层、全连接层或 LIF 层的 accuracy 输出。
2. Accuracy-only 运行不得构造或调用物理电流后端。
3. 切换硬件 ADC/LUT profile不得改变 `SNN Original Accuracy`，但必须按
   第3.2节切换 PIM accuracy 的 ReADC/无ADC数值链，因此允许改变
   `SNN Processing-in-Memory Accuracy`。
4. LUT latency、LUT energy、ReADC conversion energy和hardware current backend
   参数不得改变 accuracy。只有正式器件合同、ReADC配置和合格的LIF
   perturbation profile可以按本文规则改变 PIM accuracy。
5. Hardware-performance 平面不得原地修改 accuracy tensor、bit-weight array、
   network state 或 RNG state。
6. Hardware-performance variation与accuracy variation必须拥有独立 RNG对象
   和状态；不得对已经由accuracy variation修改过的数值再次施加同一扰动。
7. 报告不得宣称 LUT profile 的 accuracy 使用了 frontend/LIF LUT 非线性；它
   使用的是由独立电路表征得到的扰动 SpikingJelly 模型。

第 7 项的正确报告口径是：

```text
accuracy_paths = snn_original | snn_processing_in_memory
accuracy_neuron_model = spikingjelly_lif
accuracy_hardware_coupling = profile_selected_numerical_only
hardware_current_model = physical_conductance
hardware_output_path = adc | lut
accuracy_adc_model = READC_LLOYD_MAX | none
accuracy_lif_perturbation_profile = none | qualified_profile_id
```

所有SNN输入事件和LIF输出必须使用显式BinarySpikeDomain：

```text
allowed values = {0, 1}
logical scale = 1
signed = false
input_digit_max = 1
activation_cycle = 1
rounding = forbidden
clamping = forbidden
```

不能用当前有符号量化公式中的`activation_bit=1`代替该域，因为
`2^(1-1)-1=0`。LIF输出后必须丢弃此前ADC accumulator或weighted-output
scale，下一weighted layer直接消费0/1输入码。

## 6. 初始支持的硬件架构

第一版只支持两个完整、物理含义明确的 profile。

| Profile | 输入/输出外围 | 累加 | 神经元 | 模拟输入闭合 |
| --- | --- | --- | --- | --- |
| `adc_digital_lif` | 原 MNSIM DAC + ReADC | ReADC 后数字累加 | 数字 LIF | 不要求 |
| `frontend_lut_lif_lut` | 二值 spike 驱动 + frontend LUT，不实例化 ADC | LUT 定义的闭合输入 | LIF LUT | 要求 |

第一版不允许任意混搭：

```text
ADC + LIF LUT
frontend LUT + digital LIF
同时实例化 ADC 和 frontend LUT
配置 LUT 但缺少 LUT 数据时退回 ADC
配置 ADC 但复用 LUT 能耗或时延
```

未来若有真实电路依据，可以通过新 profile 增加混合架构；不能仅通过组合两个
字符串开关产生未经定义的硬件。

配置必须表达一个完整硬件 profile。精确 INI section/key 名称在实现计划中
确定，但语义上至少必须包含：

```text
SNN_Hardware_Profile = adc_digital_lif | frontend_lut_lif_lut
Hardware_Current_Model = physical_conductance
```

配置解析必须 fail closed。未知 profile、缺少对应参数、LUT 文件不匹配或非法
组合都必须在网络/round 执行前失败。

## 7. 锁定器件合同

真实电流硬件模型只能使用当前两组正式器件：

```text
rram_20ns_linear_fit
rram_30ns_linear_fit
```

器件合同由 `MNSIM/Interface/final_rram_contract.py` 负责。硬件电流后端不得
重新解析任意外部电阻 JSON，也不得拥有第三套默认电阻表。

两组器件的有效相对电阻 sigma 和可复现 seed 为：

| Device model | `Device_Variation` | `Device_Seed` |
| --- | ---: | ---: |
| `rram_20ns_linear_fit` | `6.69903802061613%` | `112` |
| `rram_30ns_linear_fit` | `7.725782066255718%` | `112` |

`Device_Variation` 是从各自器件数据得到的有效相对电阻 sigma。`Device_Seed`
不是测量得到的器件参数，而是项目固定的可复现随机种子。两者必须在报告中
分别标识，不能把 seed 描述成拟合结果。

正式 PIM accuracy 对两组器件固定使用：

```text
Enable_R_Ratio = 1
Enable_Variation = 1
Enable_SAF = 0
Device_SAF = 0,0
Device_Seed = 112
```

不得通过 CLI 默认值把 `Enable_R_Ratio` 或 `Enable_Variation` 静默关闭。SAF
框架继续保留，但当前两组正式器件没有合格的非零 SAF 数据，因此正式值固定
为 `0,0`。

### 7.1 Accuracy variation 合同

PIM accuracy 必须沿用原 MNSIM `Weight_update -> set_weights_forward` 的调用位置
和数值映射框架，并使用项目修正后的逐物理 cell 独立 variation：

```text
R_nom = locked Device_Resistance[code]
epsilon_accuracy ~ Normal(0, Device_Variation / 100)
R_accuracy = R_nom * (1 + epsilon_accuracy)
G_accuracy = 1 / R_accuracy
code_equiv = normalized_conductance(G_accuracy)
```

正负阵列、不同 cell 和不同 weight slice使用独立随机变量。每次 PIM accuracy
运行在取得部署 bit weights 后只采样一次；同一运行的所有 calibration样本、
test样本、batch和timestep复用同一实例。相同 checkpoint、映射、器件和seed
必须复现。

20 ns和30 ns使用相同标准随机序列做配对比较，再分别乘各自sigma；这只用于
降低器件比较方差，不表示两种器件共享同一物理实例。

Accuracy RNG必须从非Route-A正式配置读取 `Device_Seed=112`。不得依赖默认
为 `None` 的临时 CLI seed，也不得从hardware-performance RNG继续取样。

### 7.2 Hardware-performance variation 合同

Hardware-performance variation 必须拥有独立 RNG、物理实例身份和扰动结果。
它可以使用同一个数值 seed 以保证可复现，但不得复用
`SNN Processing-in-Memory Accuracy Path` 已生成的 `epsilon_accuracy`、
`R_accuracy` 或 `code_equiv`。

每个物理 cell 的 nominal 值为：

```text
R_nom = locked Device_Resistance[code]
G_nom = 1 / R_nom
```

若启用硬件 variation：

```text
epsilon ~ Normal(0, Device_Variation / 100)
R_real = R_nom * (1 + epsilon)
G_real = 1 / R_real
```

非正 variation factor 必须重采样或明确失败，不能产生非物理电阻。物理实例
采样必须由器件 seed 和稳定的 placement/cell identity 决定，使同一物理映射
跨 timestep 和重复读取保持一致。

## 8. 真实电流和功耗语义

对差分阵列，功能列电流是：

```text
Ipos_j = sum_i(V_i * Gpos_ij)
Ineg_j = sum_i(V_i * Gneg_ij)
Idiff_j = Ipos_j - Ineg_j
```

`Idiff` 可用于硬件输出范围、ADC 输入范围或 frontend LUT 地址计算，但不能
进入 accuracy forward。

器件读取功耗不能使用 `Gpos-Gneg`，因为正负阵列都真实耗电：

```text
Pcell_total = sum_ij(V_i^2 * (Gpos_ij + Gneg_ij))
```

列负载同样使用：

```text
Gload_j = sum_i(Gpos_ij + Gneg_ij)
```

以下对象必须分别报告，不能互相替代：

```text
Ipos / Ineg
Idiff
Gload
cell read dynamic power/energy
ADC or LUT peripheral energy
static power
```

Route-A 现有向量化 `conv2d/linear` 电导计算可以作为数值实现参考，但必须增加
物理 fragment、placement、round 和 activity identity，不能继续只按逻辑层
一次性计算并宣称它等于物理 round 执行。

## 9. ADC 硬件路径

选择 `adc_digital_lif` 时：

1. 保留 2026-08-12 spec 的数字 partial-sum、跨 round buffer 和数字 LIF
   语义。
2. ADC 后的数字部分和可以跨 round 保存，因此不要求模拟 bundle 在单个窗口
   闭合。
3. 硬件电流模型可以为 ADC 提供输入范围、conversion activity、饱和风险和
   物理 xbar 读取功耗。
4. PIM accuracy固定使用原 MNSIM `set_weights_forward` 中的 Hong 2025 ReADC
   numerical action，不采用第3项计算的真实电流值。
5. ADC、DAC、数字累加和数字 LIF 的能耗与时延只在该 profile 中计入。

2026-08-12 digital-LIF spec中的统一`AccumulatorDomain`、Q15系数和整数状态用于
round硬件执行、状态存储和成本建模。该硬件functional/cost对象不得替代本节
PIM accuracy：ReADC的非均匀重建值仍按原MNSIM numerical chain完成数字合并，
再送入共享SpikingJelly LIF。两者可以相互审计，但不能互相回灌状态或输出。

正式 ADC accuracy 配置固定为：

```text
ADC_Profile = HONG_2025_READC_5BIT
ADC_Quantization_Mode = READC_LLOYD_MAX
ADC_Precision = 5
ADC_ReADC_Percentile_Low = 0.5
ADC_ReADC_Percentile_High = 99.5
ADC_ReADC_Max_Samples_Per_Layer = 8000
```

ReADC来源固定为 Hong et al., Nature Communications 16, 9749 (2025)，
DOI `10.1038/s41467-025-65233-w`，参考实现 `MIKEHHQ/ReADC`。它是 PIM accuracy
中的离线校准非均匀 ADC 数值模型，不等同于本项目 ADC 晶体管级电路模型。
选择ReADC numerical accuracy不授权复用任意普通ADC的面积、动态能耗、延迟或
静态功耗。硬件账本只有在获得ReADC对应实现和PVT来源后才能填写；否则必须
逐项标记为`unmodeled`。

每个正式器件、checkpoint和部署映射必须分别生成ReADC表。校准固定使用正式
训练 split，不得读取 test split；校准前必须先生成该器件 `Device_Seed=112`
的同一个 post-R-ratio/post-variation部署实例，校准和测试随后共享该实例。
每层收集 pre-ADC partial-sum统计，拟合 `2^5` 个Lloyd-Max重建值，并冻结表后
执行test inference。

ReADC artifact至少绑定并校验：

```text
source checkpoint/deployment checksum
dataset and calibration split identity
device model id and Device_Seed
weight mapping identity
ADC precision and calibration parameters
per-layer boundaries/levels checksum
```

缺少表、表与运行身份不匹配、校准样本为空或精度不是5 bit时必须fail closed；
不得退回uniform ADC、`SCALE`、`FIX`或静态非均匀阈值。

“硬件 ADC 输入超范围”和“accuracy ADC numerical saturation”是两个独立审计
对象。两者可以同时报告，但不能用一个对象的 clamp 结果修改另一个对象。

## 10. LUT 硬件路径

选择 `frontend_lut_lif_lut` 时：

1. 不实例化或计入 ADC。
2. 二值 spike 输入按该 profile 的直接字线驱动语义执行；不隐式计入原 MNSIM
   多位 DAC。
3. 真实 `Ipos/Ineg/Idiff/Gload` 用于 frontend LUT 的硬件工作点查询。
4. frontend LUT 输出和 LIF LUT 查询只产生硬件性能记录、范围审计和状态访问
   计数；不得替代accuracy中的weighted result、spike或膜电位。
5. LUT 的面积、静态功耗、动态能耗和时延必须有明确数据来源。缺失项必须在
   ledger 中标记为 unmodeled，不能复用 ADC 数值填充。
6. LUT 输入超出已验证域时必须 fail closed 或明确报告 infeasible；禁止静默
   clamp、外推或退回 ADC。

LUT profile 的 PIM accuracy 固定执行：

```text
post-R-ratio/post-variation bit weights
-> original MNSIM bit-sliced conv/linear
-> exact digital reconstruction in software numerical units
-> no ADC / no ReADC / no activation output quantizer
-> characterized frontend gain/offset perturbation
-> characterized-perturbation SpikingJelly LIF
-> BinarySpikeDomain
```

现有 `adc_action='NONE'` 不是该合同：它虽然跳过ADC分箱，但仍执行一次软件
activation quantization并依赖输出scale。实现必须增加独立内部 bypass分支，
但该内部实现名不得成为新的accuracy名称或报告标签。

LUT profile 的 SpikingJelly名义参数仍为 `tau=50.5016666556`、`v_th=0.01`、
`v_reset=0.0`。实际运行允许的扰动对象为：

```text
x_lif = frontend_gain * x + frontend_offset + characterized_dynamic_noise
tau_n = tau_nominal * (1 + delta_tau_n)
v_th_n = v_th_nominal + delta_v_th_n
v_reset_n = v_reset_nominal + delta_v_reset_n
```

`frontend_gain/frontend_offset/delta_tau/delta_v_th/delta_v_reset`必须来自本项目
实际 frontend + LIF 电路在声明PDK、PVT和仿真版本下的Monte Carlo/PVT表征。
当前标称25摄氏度LUT只能证明确定性工作点和性能插值，不能提供这些sigma。

静态process/mismatch参数按物理neuron/frontend lane每次运行采样一次，并在所有
sample、batch和timestep保持不变。只有具备transient/noise表征依据的动态噪声
才允许按timestep采样。LIF扰动使用独立RNG namespace；默认数值seed可以同为
`112`，但不得复用RRAM variation的随机流或状态。

合格的LIF perturbation artifact至少必须记录：

```text
schema/profile id and checksum
PDK/model files and simulator version
process corner, supply voltage and temperature
Monte Carlo sample count and seed policy
joint parameter distribution or preserved correlations
frontend/LIF circuit source checksum
supported timestep and operating range
```

不得独立采样本来相关的参数，也不得用任意百分比、RRAM variation sigma或论文
中不同电路的数据填充。缺少合格perturbation artifact时，LUT硬件性能可以按
其独立资格运行，但不得产出正式PIM accuracy；accuracy路径必须fail closed。

当前已知风险：30 ns 器件的理论最大单轨电流可能超过现有 frontend LUT
电流轴上界。正式支持 30 ns + LUT profile 前，必须扩展并验证 LUT，或将该
组合判定为不可实现。

## 11. LUT 输入闭合

输入闭合只属于 LUT 硬件路径，不应成为 ADC 路径的全局调度规则。

LUT profile 的闭合单位是一个输出块在一个 timestep 所需的完整 physical
bundle，至少包括：

```text
all input-row splits
all weight slices
positive and negative differential rails
all contributions required by the selected frontend LUT definition
```

只有 bundle 完整时，hardware-performance 平面才能生成一次 frontend LUT
事件。未闭合模拟电流不能：

```text
写入数字 SRAM
跨 round 保存
与后续 round 的数字量相加
触发一次不完整 LUT 查询
```

若完整 bundle 无法放入给定物理资源，第一版应报告 profile-specific
infeasible。不能通过引入未建模的模拟保持器、ADC 或理想模拟累加器使其通过。

ADC profile 不使用该规则。其部分结果在 ADC 后成为数字量，可以沿用现有
跨 round partial-sum 和 buffer 语义。

## 12. Activity trace 合同

真实电流依赖输入活动，不能仅由层 shape 推导。硬件性能计算必须声明活动
来源，例如：

```text
measured_sample_trace
measured_batch_trace
declared_dense_worst_case
declared_activity_factor
```

从 accuracy 采集 trace 时，trace 必须：

1. 是只读副本或不可变对象；
2. 记录 sample、layer/node、timestep、tensor quantization domain；
3. 能确定每个 physical fragment 实际读取的输入；
4. 不包含硬件电流后端修改后的值；
5. 不改变 accuracy RNG、执行顺序或状态。

正式sample-measured运行没有实际输入trace时必须fail closed。显式
`declared_dense_worst_case`或`declared_activity_factor`只允许产生声明来源的
估算hardware metrics，不得标为sample-measured，也不得用于拟合ReADC或LIF
perturbation profile。

## 13. 能耗、时延和静态功耗账本

每个硬件 profile 必须拥有互斥的模块清单。

### 13.1 ADC profile

至少包括：

```text
xbar cell read
configured input driver / DAC
ReADC conversion
digital shift/add and accumulation
digital LIF
buffers and NoC
current static-power inventory
```

ReADC面积、动态能耗、延迟和静态功耗必须分别拥有ReADC-specific来源。通用
SAR/flash ADC、Q-cell或其他转换器数据不得作为ReADC参数填入。

### 13.2 LUT profile

至少包括：

```text
xbar cell read
binary spike input driver
frontend LUT hardware
LIF LUT hardware
required state/boundary buffers
NoC
profile-specific static-power inventory
```

互斥规则：

```text
ADC profile must not charge frontend/LIF LUT cost.
LUT profile must not charge ADC or digital-LIF cost.
No profile may charge both paths for one xbar output event.
```

权重写入、DDR 和未知全局控制器仍服从现有项目边界；本文不新增其模型。

## 14. 报告合同

每次正式运行至少记录：

```text
accuracy_compute_model
accuracy_hardware_coupling
accuracy_artifact_checksum
accuracy_device_model
accuracy_device_variation_percent
accuracy_device_seed
accuracy_variation_sampling_scope
accuracy_adc_model
ReADC provenance and calibration artifact checksum
accuracy_lif_model
accuracy_lif_perturbation_profile/checksum/seed
deployment_artifact_checksum
deployment_device_code_dtype
deployment_device_code_range
hardware_profile
hardware_current_model
device_model_id
hardware_device_variation_percent
hardware_device_seed
hardware_variation_sampling_scope
activity_trace_source
input_closure_required
input_closure_status
ADC instantiated / LUT instantiated
modeled and unmodeled hardware blocks
range violations / infeasible reason
```

Accuracy report与hardware report可以出现在同一 summary 中，但指标必须分区，
不能用一个总称掩盖二者的模型边界。

Accuracy 报告字段必须使用以下全称：

```text
SNN Original Accuracy
SNN Processing-in-Memory Accuracy
```

正式报告只能使用以上两个完整标签，不允许使用别名或缩写。

## 15. 禁止行为

以下行为一律视为实现错误：

1. 将真实电流 tensor 作为 `SNN Processing-in-Memory Accuracy Path` 的层输出。
2. 选择 LUT profile 后调用任何ADC、ReADC、ADC calibration或输出activation
   quantization逻辑。
3. 选择 ADC profile 后应用 LUT 输入闭合限制。
4. 同一次执行同时计 ADC 和 LUT 能耗。
5. LUT 超范围后 clamp 到边界且不报错。
6. 缺少 LUT 文件后静默回退 ADC。
7. 使用 `Gpos-Gneg` 计算器件总耗电。
8. 把未闭合模拟部分和存入 buffer 的数字部分相加。
9. 用任意第三方器件表绕过两组 final RRAM contract。
10. 为ADC/LUT分支创建新的公开accuracy名称，而不是统一报告
    `SNN Processing-in-Memory Accuracy`。
11. 使用任意 `tau/v_th` 高斯比例、RRAM sigma或无PVT来源的论文数据模拟LIF。
12. LUT profile 缺少合格LIF perturbation artifact时静默使用零扰动。
13. ADC profile缺少ReADC表时退回uniform、`SCALE`、`FIX`或其他ADC模式。
14. LIF输出继承ADC accumulator scale，或使用带符号 `activation_bit=1` 表示
    binary spike。

## 16. 验收测试

### 16.1 Accuracy 隔离

1. 两条 accuracy 路径都注入一个调用即失败的 current backend，完整前向必须
   仍成功。
2. 同一输入分别选择ADC/LUT profile时，`SNN Original Accuracy`必须bit-exact
   相同；PIM logits、spikes和accuracy允许按正式数值链不同。
3. 人为改变硬件 current backend 输出，hardware metrics 必须变化，accuracy
   artifact 必须不变。
4. 改变hardware-performance seed不得改变accuracy artifact；改变正式accuracy
   device/LIF seed必须按相应合同改变PIM artifact并保持可复现。
5. 两条profile必须使用同一个SpikingJelly实现；ADC使用名义参数，LUT只允许
   应用合格perturbation artifact中的参数扰动。
6. 每个独立 batch 前状态必须 reset，同一样本 timestep 之间状态必须保留。
7. LIF 输出的量化域必须重置为 binary spike domain：`values={0,1}`、`scale=1`、
   `activation_cycle=1`，不得继承 LIF 之前的 ADC accumulator scale。
8. V7数据集原始事件输入也必须使用相同BinarySpikeDomain；下一weighted layer
   必须直接消费0/1输入码，不允许round或clamp。

### 16.2 ReADC accuracy

1. ADC profile必须报告Hong 2025 provenance、5-bit精度和完整校准artifact。
2. 校准只读取训练split；注入test样本读取即失败的loader后仍必须完成校准。
3. ReADC校准和测试必须共享同一个post-variation部署实例。
4. 每个器件、checkpoint、mapping或seed变化都必须使旧表身份校验失败。
5. 缺表、空表、非单调boundary、level数量错误或checksum不匹配必须fail closed。
6. LUT profile注入一个调用即失败的ReADC collector/quantizer后仍必须完成PIM
   accuracy。

### 16.3 LUT-profile LIF perturbation

1. 缺少合格perturbation artifact时，正式LUT-profile PIM accuracy必须失败。
2. 静态扰动同一运行跨sample/batch/timestep保持不变，不同物理neuron独立或按
   artifact保存的相关结构联合采样。
3. 动态噪声只有artifact声明并提供分布时才可逐timestep采样。
4. 相同artifact和seed重复运行必须复现；改变seed必须改变采样实例。
5. 禁用扰动的测试基线必须与名义SpikingJelly前向一致，但不得作为正式LUT
   PIM accuracy报告。
6. frontend/LIF LUT查询函数替换为调用即失败的stub后，accuracy仍必须完成。

### 16.4 Hardware-performance variation

1. 20 ns/30 ns分别读取 `6.69903802061613%`/`7.725782066255718%`，并默认
   使用 `Device_Seed=112`。
2. 相同器件、placement、cell identity和seed的物理实例必须可复现。
3. 同一物理 cell跨读取、timestep和round保持同一个扰动值。
4. 正负阵列和不同cell必须获得独立随机变量。
5. hardware variation对象不得复用accuracy variation生成的扰动或状态。
6. 改变hardware seed必须改变hardware metrics，但不得改变accuracy artifact。

### 16.5 电流正确性

1. 单 cell 验证 `G=1/R`、`I=VG`、`P=V^2G`。
2. 小型差分矩阵手算验证 `Ipos/Ineg/Idiff`。
3. 验证功耗使用 `Gpos+Gneg`，不是 `Gpos-Gneg`。
4. 20 ns 和 30 ns 模型分别验证完整256级表和确定性物理 variation。

### 16.6 架构选择

1. ADC profile 只实例化 ADC/DAC/数字 LIF ledger。
2. LUT profile 只实例化 frontend LUT/LIF LUT ledger。
3. 未知和混合 profile fail closed。
4. 缺少 LUT、LUT provenance 不匹配和 LUT 超范围均明确失败。

### 16.7 输入闭合

1. LUT bundle 缺少任一 row split、weight slice 或差分轨时不得执行 LUT。
2. 完整 LUT bundle 恰好产生一次 frontend LUT 事件。
3. ADC profile 的合法数字跨 round partial sum 不受 LUT closure 约束。

### 16.8 原 MNSIM 回归

1. 现有 ANN/CNN ADC accuracy 测试保持通过。
2. 现有 ADC + digital LIF SNN round 测试保持通过。
3. 未选择 LUT profile 时，新增代码不能改变原 MNSIM hardware结果，除非
   对应指标明确启用了新的 physical-current ledger，并在报告中标识。

## 17. 实现顺序

本文只定义设计，不授权在本次文档任务中修改实现。后续建议顺序是：

1. 建立共享 sequence executor，以 `ORIGINAL` 和 `PROCESSING_IN_MEMORY` 切换
   weighted backend/profile numerical branch，并使用同一SpikingJelly实现。
2. 修正 LIF 后的 binary spike domain，并完成状态 reset/readout 对照测试。
3. 建立checkpoint bit weights/QAT artifact共用的差分权重码适配器，以及
   activity trace的只读输入合同和fail-closed校验。
4. 建立 config/profile schema 和 fail-closed 校验。
5. 建立 accuracy/current 的类型与模块隔离测试。
6. 将 Route-A 电导/电流公式提取为 hardware-performance backend。
7. 将 backend 接到 physical fragment/round/activity trace。
8. 接入正式ReADC calibration artifact和ADC-profile PIM accuracy。
9. 建立真正跳过ADC和输出activation quantization的LUT-profile weighted分支。
10. 对frontend/LIF电路执行Monte Carlo/PVT表征，生成合格perturbation artifact。
11. 接入perturbed SpikingJelly LUT-profile PIM accuracy及其可复现测试。
12. 完成ADC hardware profile回归，证明原路径未被破坏。
13. 接入 frontend LUT，落实 LUT 输入闭合和范围检查。
14. 接入 LIF LUT 的性能与状态访问模型。
15. 完成 profile-specific ledger、报告和小规模端到端测试。
16. 先使用现有checkpoint经原MNSIM生成的差分码做端到端审计和资源扫描；未来
    QAT artifact接入时只做输入一致性与回归验证，不重写硬件执行路径。

## 18. 后续工程输入与非阻塞项

用户侧设计选择已经冻结。以下内容不是待讨论的语义选项，而是后续实现必须
提供或明确保持`unmodeled/infeasible`的工程输入：

1. 配置 section/key 的最终名称。
2. ADC hardware profile的逐cell真实电流范围审计实现；它不得替代ReADC
   numerical accuracy。
3. ReADC-specific面积、动态能耗、延迟和静态功耗表征；缺失项保持`unmodeled`。
4. frontend LUT 与 LIF LUT 的canonical文件、checksum、输入输出单位和插值
   规则；当前`fast_1h_experimental_7pct`只能作为experimental profile。
5. frontend/LIF Monte Carlo/PVT characterization及合格perturbation artifact；
   缺失时正式LUT-profile PIM accuracy不可运行。
6. 30 ns器件在frontend LUT范围扩展并验收前保持`infeasible`。
7. LUT profile binary spike input driver的动态/静态功耗数据；缺失项保持
   `unmodeled`。

上述工程输入不得改变第3、5、7、9和10节已经冻结的PIM accuracy合同。

## 19. 实现状态（2026-08-16）

本设计的accuracy部分已按
[`2026-08-15-snn-pim-accuracy-implementation.md`](../plans/2026-08-15-snn-pim-accuracy-implementation.md)
完成实现与隔离验证。当前正式可运行profile为`adc_digital_lif`；
`frontend_lut_lif_lut`的数值链、配置校验和fail-closed边界已经实现，但在提供
合格frontend/LIF Monte Carlo/PVT perturbation artifact前，正式LUT PIM accuracy
仍保持不可运行。

Task 9的配置前置依赖已由提交`7f3b7ee`（`fix: lock formal configs to final RRAM
model`）单独收口。该提交只包含`SimConfig.ini`、
`configs/hardware_profiles/g8_literature_peripheral_ann.ini`和
`configs/hardware_profiles/g8_600plus_ann.ini`，统一锁定最终20 ns RRAM模型；与
本Task 9批准的SNN-only overlay回归组合验证结果为`25 passed`。

实际执行的focused回归命令为：

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest \
  tests/test_snn_accuracy_contract.py \
  tests/test_snn_original_accuracy.py \
  tests/test_snn_unified_accuracy.py \
  tests/test_snn_no_adc_reconstruction.py \
  tests/test_snn_lif_perturbation.py \
  tests/test_readc.py \
  tests/test_readc_artifact.py \
  tests/test_adc_adaptive_quantization.py \
  tests/test_no_post_adc_requantization.py \
  tests/test_device_nonidealities.py \
  tests/test_final_rram_contract.py \
  tests/test_pim_numerical_audit.py \
  tests/test_round_orchestrated_interface.py \
  tests/test_round_orchestrated_runner.py \
  tests/test_round_orchestrated_cli.py \
  tests/test_round_snn_integration.py \
  tests/test_peripheral_literature_profile.py -q
```

Task 9精确暂存树的结果为`244 passed`，无失败或skip。pytest另报告一个既有
`SyntaxWarning`，来自`MNSIM/Interface/network.py`中的正则表达式字符串，不影响
本次accuracy隔离验证。

实际执行的广义非硬件回归命令为：

```bash
/home/xzh/micromamba/envs/workspace/bin/python -m pytest tests -q \
  --ignore=tests/test_round_phase2_real_backend.py \
  --ignore=tests/test_round_orchestrated_g1_g11_sweep.py
```

最终提交树的结果为`754 passed, 6 warnings`，无失败或skip。被忽略的两个文件
属于长时间硬件后端和G1-G11 sweep工作流，不属于本accuracy实现计划。warning
均为既有的正则字符串`SyntaxWarning`或`torch.load(weights_only=False)`提示。

实际执行的一样本ADC-profile wrapper smoke为：

```bash
MNSIM_PYTHON=/home/xzh/micromamba/envs/workspace/bin/python \
MNSIM_DVS_BINNED_ROOT=/home/xzh/snn_pretrain_and_v7_dataset_20260816/snn_binned_4split_2000perclass_v7_fullres_t30_phase6 \
MNSIM_DVS_TRAIN_BATCH=1 \
MNSIM_DVS_TEST_BATCH=1 \
bash scripts/run_round_orchestrated_sim.sh \
  --network standard_vgg8_snn_v7_phase6 \
  --weights /home/xzh/snn_pretrain_and_v7_dataset_20260816/snn_software_pretrain/standard_vgg8_snn_v7_software_pretrain_best.pth \
  --num-classes 8 \
  --sim-config SimConfig.ini \
  --output-dir results/snn_pim_adc_smoke_20260816_task9 \
  --dataset-module MNSIM.Interface.dvs_binned_npz \
  --max-samples 1 \
  --overwrite \
  --device 5
```

结果：exit code `0`，`SNN Original Accuracy=1.0`，
`SNN Processing-in-Memory Accuracy=1.0`，并生成`run_manifest.json`、
`summary.json`、`accuracy_report.json`和`readc_calibration.json`。报告确认
`hardware_profile=adc_digital_lif`、`accuracy_device_model=rram_20ns_linear_fit`、
`accuracy_device_seed=112`、`accuracy_adc_model=READC_LLOYD_MAX`，且ReADC artifact
包含训练逻辑split、实际`pretrain`存储目录、完整identity checksum和逐层table
checksum。该smoke未把物理电流或硬件LUT对象记录为accuracy输入。
