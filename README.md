# SPARKSIM

**A Simulator for Spiking Neural Networks on RRAM Compute-in-Memory Architectures**

SPARKSIM 由 SPARK（火花，呼应神经元脉冲触发计算）与 SIM（Simulator，仿真器）组成。

**面向脉冲神经网络的存内计算仿真器：从软件训练、量化与权重部署，到受限硬件资源下的时序执行和电路开销评估。**

SPARKSIM 将网络准确率评估与硬件性能评估连接到同一份网络定义、权重和硬件配置。硬件路径通过两阶段执行：Phase 1 根据物理资源生成分片与轮次计划；Phase 2 执行计划，维护跨轮状态，并统计权重编程和推理的延迟、能耗、面积及吞吐指标。

支持的能力包括：

- 多时间步 SNN、LIF 状态边界、残差及多分支网络；
- 软件预训练和 `FIX_TRAIN` 量化感知微调（QAT）；
- 差分 RRAM 映射、权重位平面、跨阵列 partial sum 和跨轮数据边界；
- Stanford/ASU 与 Stanford-PKU 两组 256 电导状态器件模型；
- 顺序调度、内置优化策略、冻结调度回放，以及版本化 JSON 自定义调度接口；
- 权重编程与推理分项统计、外围电路能耗、SRAM 状态开销和能耗账本审计。

发布内容为源代码、配置、器件表和测试。**数据集、训练权重、实验结果和本机 Python 环境不包含在仓库中。** 内部 Python 包名仍为 `MNSIM`，下文的导入路径和环境变量沿用这一名称。

## 目录

1. [仓库结构与命令约定](#1-仓库结构与命令约定)
2. [使用 Conda 安装环境](#2-使用-conda-安装环境)
3. [快速开始：已有权重的硬件评估](#3-快速开始已有权重的硬件评估)
4. [网络、数据集与权重准备](#4-网络数据集与权重准备)
5. [软件预训练与固定精度 QAT](#5-软件预训练与固定精度-qat)
6. [准确率与联合评估](#6-准确率与联合评估)
7. [硬件资源与器件配置](#7-硬件资源与器件配置)
8. [权重调度与两阶段运行](#8-权重调度与两阶段运行)
9. [接入自己的调度算法](#9-接入自己的调度算法)
10. [结果文件、指标与电路能耗边界](#10-结果文件指标与电路能耗边界)
11. [测试与常见问题](#11-测试与常见问题)
12. [扩展入口与详细文档](#12-扩展入口与详细文档)

## 1. 仓库结构与命令约定

```text
.
├── README.md                         # 本教程
├── PUBLICATION_CHECKLIST.md
└── SPARKSIM/                      # 运行工作目录
    ├── MNSIM/
    │   ├── Interface/                # 网络、训练、QAT、CLI 与运行器
    │   ├── Scheduling/               # Phase 1/2、调度、状态及账本
    │   ├── Hardware_Model/           # 器件、阵列、PE、tile、Buffer 等
    │   ├── Accuracy_Model/
    │   ├── Area_Model/
    │   ├── Latency_Model/
    │   ├── Power_Model/
    │   └── Energy_Model/
    ├── configs/                      # 资源、量化、外围及静态功耗配置
    ├── device_models/                # ASU/PKU 模型、转换表和来源记录
    ├── scripts/                      # 命令入口及辅助工具
    ├── tests/
    ├── docs/                         # 接口规范、设计及模型假设
    └── SimConfig.ini
```

在仓库根目录执行一次：

```bash
cd SPARKSIM
```

**除环境创建外，后续命令均在包含 `SimConfig.ini` 的这个目录执行。** 示例中的 `/absolute/path/...` 需要替换为实际路径。建议权重和数据集使用绝对路径。

项目直接从源码目录运行，目前无需 `pip install -e .`。Shell 仿真入口会切换到该目录；不要将外层仓库目录再次拼入配置路径。

## 2. 使用 Conda 安装环境

建议 Linux、Bash、Conda 与 Python 3.12。CPU 可以执行测试和硬件评估；训练及大规模准确率评估建议使用 NVIDIA GPU。

### 2.1 创建独立环境

```bash
conda create -n sparksim python=3.12 pip -y
conda activate sparksim
```

### 2.2 安装 PyTorch

根据机器选择下面一种方式。CUDA 示例使用 PyTorch 2.5.1 / torchvision 0.20.1 / CUDA 12.1 wheel；主机 NVIDIA 驱动须支持所选运行时。

```bash
# NVIDIA GPU
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

```bash
# CPU
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

安装其余常用依赖：

```bash
python -m pip install spikingjelly==0.0.0.0.14 numpy scipy pandas matplotlib tqdm pytest
```

这是主要工作流的安装起点，不是所有可选后端的锁定环境。DDR/NoC 外部工具需要单独准备；默认入门流程关闭 DDR 和 NoC，不依赖它们。

### 2.3 检查解释器和入口

```bash
python -c 'import sys, torch, spikingjelly; print(sys.executable); print(torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPUs:", torch.cuda.device_count())'
export MNSIM_PYTHON="$(command -v python)"
bash scripts/run_round_orchestrated_sim.sh --help
python scripts/train_snn_v7.py --help
python scripts/finetune_snn_v7.py --help
```

`MNSIM_PYTHON` 控制 Shell 仿真入口使用的解释器。仿真的 CPU 参数为 `--device cpu`，GPU 参数可以是 `--device 0` 或 `--device cuda:0`；训练脚本使用 `--device cuda:0` 或 `--device cpu`。

## 3. 快速开始：已有权重的硬件评估

此流程不需要数据集，也不会训练网络。需要与网络定义匹配、包含有效量化范围的 checkpoint；已有 QAT 的 `*_best.pth` 是推荐输入。

### 3.1 设置网络和权重

```bash
export NETWORK=vgg8_snn_v7_phase6
export WEIGHTS=/absolute/path/to/vgg8_snn_v7_phase6_software_qat_finetune_best.pth
export MNSIM_SNN_SEQUENCE_LENGTH=30
export MNSIM_PYTHON="$(command -v python)"
```

### 3.2 生成 ASU 配置

以下命令基于随附的 16 物理阵列配置，使用模型工具完整写入器件参数，输出一份用于本次运行的配置：

```bash
python -m scripts.apply_final_rram_model \
  --base-sim-config configs/hardware/SimConfig_16xbar_square_asu.ini \
  --device-model rram_stanford_asu_256 \
  --output runs/configs/asu_16xbar.ini
```

外围 profile 的相对路径以配置文件目录为基准。由于这里把配置写到了新目录，需将该路径改为绝对路径：

```bash
python - <<'PY'
import configparser
from pathlib import Path

path = Path("runs/configs/asu_16xbar.ini")
config = configparser.ConfigParser()
config.read(path)
config.set("SNN Peripheral Energy", "Profile",
           str(Path("configs/peripherals/asu_rram_snn_circuit_v1.json").resolve()))
with path.open("w") as stream:
    config.write(stream)
PY
```

生成器用于替换器件字段，其他相对配置引用在移动配置时也应检查。下面的命令持续使用这份配置。

### 3.3 执行权重编程加推理评估

```bash
bash scripts/run_round_orchestrated_sim.sh \
  --network "$NETWORK" \
  --weights "$WEIGHTS" \
  --num-classes 8 \
  --sim-config runs/configs/asu_16xbar.ini \
  --output-dir runs/vgg8_asu_seq \
  --device cpu \
  --skip-accuracy \
  --run-hardware \
  --schedule-policy seq \
  --show-rounds
```

完成后，终端显示总延迟、能耗、读写分项、面积、吞吐与轮次信息；结果保存于 `runs/vgg8_asu_seq/`。

`--skip-accuracy` 跳过数据集上的准确率评估，`--run-hardware` 请求硬件指标；硬件路径仍按网络的 SNN 时间步与状态依赖执行成本模型。**硬件模式并不等同于在数据集上完成数值脉冲推理**，其静态拓扑外围模型的假设见第 10 节。

不带 `--skip-accuracy` 时默认请求准确率；不带 `--run-hardware` 时不会生成硬件性能报告。

## 4. 网络、数据集与权重准备

### 4.1 已定义的 V7 SNN

训练脚本接受下列名称；所有模型使用同一套训练/评估入口，不设基线模型。

| 网络 | `--model` / `--network` |
| --- | --- |
| LeNet | `lenet_snn_v7_phase6` |
| AlexNet | `alexnet_snn_v7_phase6` |
| VGG8 | `vgg8_snn_v7_phase6` |
| VGG11 | `vgg11_snn_v7_phase6` |
| VGG13 | `vgg13_snn_v7_phase6` |
| VGG16 | `vgg16_snn_v7_phase6` |
| MobileNet | `mobilenet_snn_v7_phase6` |
| ResNet18 | `resnet18_snn_v7_phase6` |
| ResNet34 | `resnet34_snn_v7_phase6` |
| ResNet50 | `resnet50_snn_v7_phase6` |
| Inception | `inception_snn_v7_phase6` |

这些是面向当前输入和分类任务适配的 SNN 定义，不能直接加载 torchvision 同名 ANN 权重。卷积下采样取代网络中的 pooling；残差和分支通过图节点及显式 LIF 边界表示。网络定义已存在不代表仓库附有训练好的权重。

### 4.2 V7 数据契约

当前 V7 训练入口使用 **8 类、6 通道、默认 30 时间步**，模型空间输入为 `32×32`。准备如下目录，各 split 直接包含逐样本 `.npz` 文件，而不是类别子目录：

```text
your_v7_dataset/
├── meta.json
├── pretrain/
│   └── sample_000000.npz
├── finetune/
│   └── sample_000000.npz
├── calibration/
│   └── sample_000000.npz
└── test/
    └── sample_000000.npz
```

`meta.json` 示例：

```json
{"T_bins": 30, "C": 6, "H": 32, "W": 32, "num_classes": 8}
```

每个样本保存：

| 字段 | 要求 |
| --- | --- |
| `x_seq` | `[T,C,H,W]` 张量，V7 为 `T=30,C=6`；兼容字段名 `x` |
| `y` | 整数类别 `0..7`；兼容字段名 `label` |

加载器将正数转为 `1`，其余转为 `0`。原始空间尺寸可以不同，加载器用空间 OR 聚合到目标尺寸；这属于输入预处理，不是网络中的 pooling 层。输入通道数必须匹配 V7 模型，`meta.json` 不会自动把其他通道转换成六通道。

四个 split 应独立划分，避免样本泄漏。用途如下：

| 用途 | 读取的 split |
| --- | --- |
| 软件预训练更新参数 | `pretrain` |
| QAT 微调更新参数 | `finetune` |
| 训练/QAT 选择最佳 checkpoint | `calibration` |
| 最佳 checkpoint 的最终测试 | `test` |
| 正式 PIM 准确率的 ReADC 校准 | `pretrain`，在适配器中逻辑标记为 `train` |
| 正式准确率评估 | `test` |

训练脚本当前固定构建 8 类模型。修改类别数、输入通道或形状时，应同时修改模型、训练入口和数据契约；仅修改仿真 CLI 的 `--num-classes` 不会迁移原权重。

### 4.3 Checkpoint 要求

仿真和微调使用 `*_best.pth` 模型 state dict。`*_latest.pth` 包含 optimizer、scheduler 和随机状态，供训练恢复，不应直接当作仿真权重。

模型名、类别数、LIF 参数、readout 和量化策略应与训练设置一致。残差/BN/post-add 硬件域需要有效校准范围，推荐先完成 QAT；纯浮点权重不能凭空补出这些范围。默认 LIF 参数为 `tau=50.5016666556`、阈值 `0.01`、reset `0`，若训练时覆盖，评估时也需使用相同设置，例如：

```bash
export MNSIM_SNN_LIF_TAU=50.5016666556
export MNSIM_SNN_LIF_V_TH=0.01
```

## 5. 软件预训练与固定精度 QAT

已有合适 checkpoint 可以跳过本节。下列命令会实际训练，轮数和 batch size 是起步示例，应根据模型和 GPU 调整。

### 5.1 软件预训练

```bash
export DATASET_ROOT=/absolute/path/to/your_v7_dataset
python scripts/train_snn_v7.py \
  --model vgg8_snn_v7_phase6 \
  --dataset-root "$DATASET_ROOT" \
  --sim-config SimConfig.ini \
  --output-dir runs/vgg8_pretrain \
  --device cuda:0 \
  --epochs 60 \
  --train-batch 8 \
  --test-batch 8 \
  --learning-rate 0.01 \
  --seed 42
```

主要产物：

```text
vgg8_snn_v7_phase6_software_pretrain_best.pth
vgg8_snn_v7_phase6_software_pretrain_latest.pth
vgg8_snn_v7_phase6_software_pretrain_history.json
```

预训练使用软件时序 forward 和代理梯度，不统计电路能耗，也不启动器件仿真。

### 5.2 FIX_TRAIN / QAT 微调

```bash
python scripts/finetune_snn_v7.py \
  --model vgg8_snn_v7_phase6 \
  --checkpoint runs/vgg8_pretrain/vgg8_snn_v7_phase6_software_pretrain_best.pth \
  --dataset-root "$DATASET_ROOT" \
  --sim-config SimConfig.ini \
  --output-dir runs/vgg8_qat \
  --device cuda:0 \
  --epochs 20 \
  --train-batch 8 \
  --test-batch 8 \
  --learning-rate 0.001 \
  --qat-policy configs/quantization/qat_fixed9.json \
  --calibration-batches 16 \
  --seed 42
```

产物名称对应 `*_software_qat_finetune_best.pth`、`*_latest.pth`、`*_history.json`。`--calibration-batches` 使用 finetune 样本初始化量化范围，初始化时冻结权重与 BN 统计；后续训练按正常 QAT 规则进行。

当前推荐配置为 **fixed precision + uniform activation**：权重和非 spike 激活的全局有符号宽度均为 9 bit，LIF spike 保持二值。差分器件的幅值码为 `0..255`，不要将带符号 9 bit 权重和 256 状态器件混为一谈。

其他已接入的软件 QAT 选项：

| 配置 | 行为与使用边界 |
| --- | --- |
| `configs/quantization/qat_fixed9.json` | 固定 9 bit、均匀激活，推荐硬件评估流程 |
| `configs/quantization/qat_fixed9_mu_law.json` | 固定精度、非均匀激活；当前不支持非均匀硬件导出/PIM 执行 |
| `configs/quantization/qat_mixed_example.json` | 显式逐层位宽分配的软件 QAT 示例，硬件端仍需满足部署契约 |

这里的 mixed precision 是逐层量化位宽分配，不是 FP16/FP32 AMP，也不执行自动位宽搜索。非均匀激活与逐层位宽是两个独立选项。详见[量化策略规范](SPARKSIM/docs/specs/2026-10-10-qat-quantization-options.md)。

### 5.3 恢复训练与 readout

恢复训练时，重用原命令、原输出目录和原配置，添加 `--resume`，并将 `--epochs` 设为希望达到的总轮数。恢复读取 latest checkpoint；`--resume` 与 `--overwrite` 互斥。

预训练默认对 ResNet 使用 `membrane` readout，其他模型使用 `spike`；微调脚本默认 `spike`。若需要相同 readout，应显式传 `--readout spike` 或 `--readout membrane`。训练入口的 readout 选项不代表仿真 CLI 具备同名参数，比较准确率时应核对执行路径的分类读出定义。

## 6. 准确率与联合评估

### 6.1 单样本准确率冒烟

```bash
export MNSIM_DVS_BINNED_ROOT="$DATASET_ROOT"
export MNSIM_DVS_TRAIN_BATCH=1
export MNSIM_DVS_TEST_BATCH=1
export MNSIM_DVS_NUM_WORKERS=0

bash scripts/run_round_orchestrated_sim.sh \
  --network "$NETWORK" \
  --weights "$WEIGHTS" \
  --num-classes 8 \
  --sim-config runs/configs/asu_16xbar.ini \
  --output-dir runs/vgg8_accuracy_smoke \
  --dataset-module MNSIM.Interface.dvs_binned_npz \
  --device 0 \
  --max-samples 1 \
  --accuracy-seed 112
```

输出包括 `SNN Original Accuracy` 和 `SNN Processing-in-Memory Accuracy`。`--max-samples 1` 用于连通性检查，不是正式准确率结论；正式测试时移除此限制。它也不是限制 ReADC 校准数据量或硬件执行时间步的开关。

### 6.2 准确率加硬件

在上述命令中增加 `--run-hardware --show-rounds`，并使用新的 `--output-dir`，即可同时请求准确率和硬件评估。只运行软件准确率时加 `--skip-pim`；只运行 PIM 准确率时加 `--skip-original`。两者不能同时指定。

### 6.3 两类配置不要混淆

- `[SNN Accuracy] SNN_Hardware_Profile` 控制数值准确率路径。默认 `adc_digital_lif` 使用 ReADC 及绑定训练集身份的校准 artifact；`frontend_lut_lif_lut` 需要合格的前端/LIF 扰动表，缺失时明确失败。
- `[Neuron] LIF_Mode` 控制硬件 Phase 2 的 LIF 成本路径，默认 `digital`。`analog_lut` 需要额外的 LUT 和对应配置，源码发布不保证包含这些表或其归档面积依赖。

硬件前端电流和硬件 LIF 能耗表不会直接当作准确率张量。不要为了绕过缺失的准确率 artifact，随意替换为硬件能耗 LUT。

## 7. 硬件资源与器件配置

### 7.1 物理容量的计算

关键 INI 项：

| Section / 字段 | 含义 |
| --- | --- |
| `[Crossbar level] Xbar_Size` | 单阵列行列数 |
| `[Tile level] PE_Num` | 每 tile 的 PE 行列布局 |
| `[Architecture level] Tile_Num` | 芯片 tile 行列布局 |
| `[Process element level] Group_Num` | 每 PE 的 group 数 |
| `[Process element level] Xbar_Polarity` | 单端/差分物理阵列数 |
| `[Algorithm Configuration] Weight_Quantization_Bit` | 部署权重有符号量化宽度 |
| `[Neuron] LIF_Mode` | 硬件 LIF 模式 |
| `[Static Power]` | 静态功耗 profile 与计费策略 |
| `[SNN Peripheral Energy]` | 额外外围能耗 profile |

```text
物理 xbar 数 = tile_rows × tile_cols × pe_rows × pe_cols
              × Group_Num × Xbar_Polarity
```

随附配置：

| 文件（相对运行目录） | tile | 每 tile PE | groups | polarity | 物理阵列数 |
| --- | --- | --- | --- | --- | --- |
| `configs/hardware/SimConfig_16xbar_square_asu.ini` | 2×2 | 1×1 | 2 | 2 | 16 |
| `configs/hardware/SimConfig_256xbar_square.ini` | 4×4 | 2×2 | 2 | 2 | 256 |

二者的 xbar 均为 `128×128`。资源较小时通常出现更多权重更新轮次，但性能改善取决于实际权重、驻留状态和器件转换代价，不能仅凭轮次推断优化比例。

改变资源后需重新规划、生成调度；旧 artifact 不应继续用于新配置。默认 `SimConfig.ini` 与上述两个方形资源配置不同，请显式记录选用的文件。

### 7.2 选择 ASU 或 PKU

| 器件族 | 模型 ID | 用途 |
| --- | --- | --- |
| Stanford/ASU | `rram_stanford_asu_256` | 读取与完整源状态→目标状态写入代价 |
| Stanford-PKU | `rram_memtorch_stanford_pku_256` | nominal 读取/准确率，不能用于完整写入评估 |
| Stanford-PKU | `rram_memtorch_stanford_pku_write256` | 读取与 compact-model 写入转换代价 |

切换器件时使用工具完整覆盖器件字段，不能只手动修改 `Model_ID`：

```bash
python -m scripts.apply_final_rram_model \
  --base-sim-config runs/configs/asu_16xbar.ini \
  --device-model rram_memtorch_stanford_pku_write256 \
  --output runs/configs/pku_16xbar.ini
```

随后将仿真的 `--sim-config` 改为 `runs/configs/pku_16xbar.ini`。此示例保留同一资源与外围假设来比较器件；ASU 命名外围 profile 是分析估计，切换模型不会把它自动变为 PKU 器件族的独立校准结果。

原始参考 INI 位于 `SPARKSIM/device_models/final_rram/configs/`。器件 JSON、阻值表、checksum 和模型能力会被检查，不匹配时报告 `INVALID_FINAL_RRAM_MODEL`。

运行时读取预处理后的器件和转换表，不运行 HSPICE，也不把外部 HSPICE 总能耗注入最终统计。256 状态是仿真接口离散化，不等于已经实测验证 256 级存储可达性。模型来源与写入协议见[器件说明](SPARKSIM/device_models/final_rram/README.md)。

## 8. 权重调度与两阶段运行

### 8.1 先生成，再执行

对昂贵的调度，可先冻结结果，随后重复执行 Phase 2。先生成顺序调度：

```bash
bash scripts/run_round_orchestrated_sim.sh \
  --network "$NETWORK" --weights "$WEIGHTS" --num-classes 8 \
  --sim-config runs/configs/asu_16xbar.ini \
  --output-dir runs/vgg8_schedule_seq --device cpu \
  --skip-accuracy --run-hardware \
  --schedule-only --schedule-policy seq \
  --schedule-plan-out runs/vgg8_schedule_seq/seq.artifact.json
```

仅生成调度不执行完整推理，不会产生可比较的完整推理硬件指标。随后执行冻结方案：

```bash
bash scripts/run_round_orchestrated_sim.sh \
  --network "$NETWORK" --weights "$WEIGHTS" --num-classes 8 \
  --sim-config runs/configs/asu_16xbar.ini \
  --output-dir runs/vgg8_replay_seq --device cpu \
  --skip-accuracy --run-hardware --show-rounds \
  --infer-with-schedule \
  --schedule-plan-in runs/vgg8_schedule_seq/seq.artifact.json
```

回放会检查网络、权重、配置和计划身份；不重新搜索调度。只修改 `--schedule-policy` 不会替换已加载 artifact 中的决策。

### 8.2 内置策略与原始 FlashSeF-PTR 的区别

CLI 提供 `seq`、`flashsef_ptr`、`snn_flashsef_ptr`、`snn_native_temporal`、`snn_flashsef_temporal` 等策略。可以将第 8.1 节生成命令中的 `--schedule-policy` 换成相应名称，并为每个方案使用独立输出目录和 artifact 文件。

**当前发布版内置的 `flashsef_ptr` 是基于转换代价及 Hungarian assignment 的实现，不是原始 FlashSeF-PTR 源码的完整运行入口。** 原始实现的桥接脚本 `scripts/generate_legacy_flashsef_ptr.py` 依赖未包含在此发布副本中的 `archive/FlashSeF-PTR_source_20261010/`。在未单独准备该源码包前，不能把桥接脚本当作可直接运行的教程，也不能把内置策略的结果标记为原始实现的复现实验。

### 8.3 公平比较

比较不同调度时固定网络、checkpoint、器件、资源、T、外围和静态功耗配置，分别生成并执行 artifact。以完整 Phase 2 结果为准：

```text
能耗改善率 = (E_seq - E_method) / E_seq × 100%
延迟改善率 = (L_seq - L_method) / L_seq × 100%
```

负值表示变差。权重编程分项的改善不能直接作为总能耗或总延迟的改善；若复用同一推理成本，应明确说明这一比较假设。

## 9. 接入自己的调度算法

用户可以自行决定算法的输入和实现方式。仿真器仅约束输出方案，通过 `mnsim.weight_schedule` v1 JSON 接收整 tile 映射和允许范围内的轮次顺序。

### 9.1 从当前计划导出合法起始模板

先执行第 8.1 节的顺序调度生成命令，然后：

```bash
python - <<'PY'
from MNSIM.Scheduling.weight_scheduling import (
    load_schedule_artifact,
    custom_schedule_from_artifact,
    save_custom_schedule,
)

artifact = load_schedule_artifact("runs/vgg8_schedule_seq/seq.artifact.json")
schedule = custom_schedule_from_artifact(artifact, policy_id="my_scheduler")
save_custom_schedule("runs/my_schedule.json", schedule)
PY
```

自己的算法修改模板中的物理 tile 分配和合法轮次顺序。其结构如下，ID 仅用于说明，不能直接用于任意网络：

```json
{
  "format": "mnsim.weight_schedule",
  "schema_version": 1,
  "policy_id": "my_scheduler",
  "rounds": [
    {"round_id": "R0", "assignments": {"tile:0": 1, "tile:1": 0}},
    {"round_id": "R1", "assignments": {"tile:0": 0}}
  ]
}
```

### 9.2 执行自定义方案

```bash
bash scripts/run_round_orchestrated_sim.sh \
  --network "$NETWORK" --weights "$WEIGHTS" --num-classes 8 \
  --sim-config runs/configs/asu_16xbar.ini \
  --output-dir runs/vgg8_custom --device cpu \
  --skip-accuracy --run-hardware --show-rounds \
  --custom-schedule-in runs/my_schedule.json
```

成功后保存 `custom.schedule.artifact.json`，并执行同一套硬件评估。自定义方案不能提交能耗或延迟来覆盖仿真结果。

v1 的主要约束：

- 每个加权轮次恰好出现一次，每轮逻辑 tile 必须完整覆盖。
- 同轮不同逻辑 tile 不能映射到同一物理 tile；跨轮可以复用。
- 物理 ID 必须属于当前计划可用池，不能随意增加未使用的硬件 tile。
- 不能跨越 LIF、merge、固定执行项或依赖边界重排。
- 不支持改写 tile 内 PE/group/polarity、跨轮重新分组或任意预加载/清除。

自定义输出 v1 与内部冻结 artifact v2 是两个不同格式；前者用于算法接入，后者用于完整执行回放。详见[格式规范](SPARKSIM/docs/specs/2026-10-10-custom-weight-schedule-format.md)和 [JSON Schema](SPARKSIM/docs/schemas/weight-schedule-v1.schema.json)。

## 10. 结果文件、指标与电路能耗边界

### 10.1 输出文件

按所请求阶段，输出目录包含以下文件：

| 文件 | 内容 |
| --- | --- |
| `run_manifest.json` | 参数、阶段状态、错误和 artifact 路径 |
| `summary.json` | 聚合结果与阶段摘要 |
| `accuracy_report.json` | 软件/PIM 准确率及准确率路径身份 |
| `readc_calibration.json` | 正式 ReADC 校准 artifact（相应模式下） |
| `deployment_weights_9bit.pth` | 运行时生成的部署权重 |
| `deployment_weights_9bit_manifest.json` | 量化、源权重身份与映射记录 |
| `round_plan.json` | 实际执行的轮次和分片计划 |
| `phase2_report.json` | 推理指标、逐轮/逐步成本和能耗账本 |
| 自定义的 artifact 文件 | `--schedule-plan-out` 指定的冻结调度 |

`--schedule-only` 不会生成完整 `phase2_report.json`。输出目录已有运行器管理的结果时，默认拒绝覆盖；确认重跑时加 `--overwrite`。该选项用于覆盖运行器自身结果，不是清空目录的命令。

### 10.2 指标定义

JSON 使用秒、焦耳、瓦和平方微米；终端会转换为便于阅读的单位。

| 字段 | 含义 |
| --- | --- |
| `latency_s` / `energy_j` | 本次评估的权重编程 + 推理总成本 |
| `read_latency_s` / `read_energy_j` | 读取/推理路径及相应边界开销 |
| `weight_programming_latency_s` / `weight_programming_energy_j` | 实际状态转换与写入并行约束下的权重编程成本 |
| `area_um2` | 报告定义的硬件映射面积，不能跨轮相加为芯片面积 |
| `power_avg_w` | 总能耗 / 总延迟 |
| `compute_throughput_tops` | 排除权重编程的计算吞吐 |
| `end_to_end_throughput_tops` | 包含权重编程的端到端吞吐 |
| `modeled_energy_efficiency_tops_per_w` | 基于已建模能耗计算的能效 |
| `tops_per_w` | 通过能耗完整性条件时提供，否则可能为 `null` |
| `energy_accounting_complete` | 当前模型边界的计费完整性标记 |
| `accounting_scope` | 模型覆盖范围及假设 |
| `energy_ledger` | 组件能耗、来源、边界和对账结果 |

MAC 按两个操作计算，SNN 的时间步数计入工作量。对相同工作量：

```text
OPS = 2 × MACs
TOPS = OPS / latency_s / 10^12
TOPS/W = OPS / energy_j / 10^12
```

分别对比部署和推理成本，不要混用 compute 与 end-to-end 口径。硬件报告的整体成本对应本次执行计划，不能直接当成多样本准确率测试全集的累计成本。

### 10.3 电路账本与外围项

本项目的“完整”指标边界为 **权重编程 + 推理硬件开销**。现有 frontend、差分和 LIF 成本按选定后端保留，不以“排除 LIF/TIA”的方式重定义实验。不同 LIF 后端的组件粒度可能不同，聚合模型不会自动拆出未单独表征的晶体管级分项。

16 阵列 ASU 配置启用了 `asu_rram_snn_circuit_v1.json`，补充读供电、列参考、AROW 行驱动/选择/启动及控制项。其 `execution_mode=mnsim_static_topology` 根据实际 placement 与已有 MNSIM 读取成本计算，属于**分析估计**；不声称拥有硬件模式下未执行的真实逐样本行/spike 活动。spike 传递与接口的已有 Buffer/边界模型继续负责计费，避免重复收取。

器件写入转换表负责状态转移成本；写入驱动/供电等实际纳入项应以报告中的来源和账本为准，不能仅凭某个组件名称推定已完成独立电路表征。额外外围 profile 本身不增加延迟或面积，其能耗归属于已有动作窗口。

`phase2_report.json` 的外层包含 `phase2_report` 和 `hardware_summary`；账本路径为 `phase2_report.inference_metrics.energy_ledger`。其中：

- `included_components_j`：纳入总能耗的组件；
- `component_provenance`：组件计算来源；
- `diagnostics_j` 与 `audit`：未覆盖项、组件求和及对账结果；
- `boundary` / `version`：成本边界和账本版本。

`diagnostics_j.reconciliation_passed` 应为 `true`，组件之和应等于 `total_energy_j`。完整性标记表示当前约定边界通过检查，不代表所有芯片待机与非 SRAM 静态模块均已独立表征。外围 profile 的来源、参数和 ownership 也应随实验报告保留。

例如，读取第 3 节运行结果并检查组件求和与外部能耗审计：

```bash
python - <<'PY'
import json
import math
from pathlib import Path

result = json.loads(Path("runs/vgg8_asu_seq/phase2_report.json").read_text())
hardware = result["hardware_summary"]
ledger = result["phase2_report"]["inference_metrics"]["energy_ledger"]
print("Latency (s):", hardware["latency_s"])
print("Energy (J):", hardware["energy_j"])
print("TOPS/W:", hardware.get("tops_per_w"))
for name, energy in ledger["included_components_j"].items():
    print(f"{name}: {energy:.9g} J")
assert math.isclose(math.fsum(ledger["included_components_j"].values()),
                    ledger["total_energy_j"], rel_tol=1e-9, abs_tol=1e-24)
assert ledger["diagnostics_j"]["reconciliation_passed"]
assert ledger["diagnostics_j"]["external_hspice_energy_read"] is False
PY
```

`summary.json` 中的聚合硬件指标位于 `hardware`；组件账本使用上面的 Phase 2 报告，不应把两个 JSON 的字段层级混用。

## 11. 测试与常见问题

### 11.1 无数据集的入口检查

```bash
python -m pytest -q \
  tests/test_round_orchestrated_cli.py \
  tests/test_custom_weight_schedule.py \
  tests/test_qat_quantization_policy.py
```

本次发布文档校验中，上述入口、调度格式与 QAT 测试共 **86 项通过**。外围测试可单独执行：

```bash
python -m pytest -q tests/test_snn_peripheral_energy.py
```

当前外围测试结果为 **12 项通过、3 项失败**。失败发生在共享模拟配置 fixture：它保留 INI 字段大小写，在已有 `lif_mode` 之外又写入 `LIF_Mode`，严格解析器将两者视为重复字段并拒绝加载。该问题尚未修复，不能声称外围回归全部通过；它也没有验证到后续数值执行是否成功。本教程的配置生成命令使用默认大小写归一化解析器，不采用该 fixture。

完整回归：

```bash
python -m pytest -q
```

部分 legacy 或模拟 LUT 集成测试需要源码发布中未包含的归档资源；应查看具体缺失依赖，不能将跳过的测试计为验证通过。

### 11.2 常见问题

| 现象 | 检查方式 |
| --- | --- |
| Conda/PyPI 域名无法解析 | 检查 DNS、代理及源可达性；确认网络恢复后安装，不以缓存安装冒充联网成功 |
| `No module named MNSIM` | 在包含 `SimConfig.ini` 的目录运行；辅助工具用 `python -m scripts.apply_final_rram_model`，不要从其他目录直接运行脚本 |
| CUDA preflight 失败 | 核对 `nvidia-smi`、当前 Conda 环境和 `torch.cuda.is_available()`；硬件成本评估可显式使用 `--device cpu` |
| checkpoint key/shape 不匹配 | 核对模型完整名称、8 类配置、V7 六通道及是否误用了 ANN/latest checkpoint |
| 激活/BN/post-add domain 未校准 | 使用完成 QAT 的权重，或按对应规范在训练数据上显式校准；不要用测试集静默更新范围 |
| `INVALID_FINAL_RRAM_MODEL` | 用器件工具重建配置，检查 checksum；硬件写入不能选 PKU nominal-only profile |
| peripheral profile 找不到 | profile 相对配置所在目录解析；复制配置后改成绝对路径或同步修正引用 |
| `MISSING_SNN_PERIPHERAL_ACTIVITY` | activity 模式需要数值读取事件；硬件-only 应选择明确支持静态拓扑的 profile |
| artifact identity/hash 不匹配 | 使用生成时完全相同的权重和配置，或重新生成调度 |
| `tops_per_w` 为 `null` | 查看 `energy_accounting_complete`、`accounting_scope` 和能耗账本；不能直接改名为已验证完整能效 |
| 小硬件上的大网络耗时较长 | 分别计时 schedule-only 与 Phase 2；检查实际轮次数和 T，不要以 GPU 利用率判断规划器是否工作 |
| 旧实验脚本找不到 `archive` / weights | 这些脚本可能依赖研究工作区；入门流程使用本教程的通用入口并显式提供数据和权重 |

退出码：`0` 成功；`1` 请求阶段执行失败；`2` 参数、输入或准备失败。先查看输出目录的 `run_manifest.json`，其中记录阶段与错误详情。

## 12. 扩展入口与详细文档

| 任务 | 入口 |
| --- | --- |
| 定义或适配网络 | `MNSIM/Interface/network.py` 及残差/多分支构图模块 |
| 调整训练/QAT | `scripts/train_snn_v7.py`、`scripts/finetune_snn_v7.py` |
| 修改资源 | `SimConfig.ini`、`configs/hardware/` |
| 切换器件 | `scripts/apply_final_rram_model.py`、`MNSIM/Interface/final_rram_contract.py` |
| 外部调度算法接入 | `--custom-schedule-in`，优先使用第 9 节格式 |
| 新增内部策略 | `MNSIM/Scheduling/weight_scheduling/policies/` |
| 审查硬件执行与计费 | `MNSIM/Scheduling/phase2/` |

详细文档：

- [自定义权重调度格式 v1](SPARKSIM/docs/specs/2026-10-10-custom-weight-schedule-format.md)
- [调度 JSON Schema](SPARKSIM/docs/schemas/weight-schedule-v1.schema.json)与[结构示例](SPARKSIM/docs/examples/weight-schedule-v1.json)
- [QAT 量化策略与硬件支持边界](SPARKSIM/docs/specs/2026-10-10-qat-quantization-options.md)
- [残差 SNN 的图、domain 与执行约定](SPARKSIM/docs/specs/2026-10-03-residual-snn-runtime.md)
- [外围能耗模型设计](SPARKSIM/docs/specs/2026-10-10-snn-additional-peripheral-energy.md)：包含初版历史限制，静态拓扑后续支持以当前代码和 profile 为准
- [RRAM 模型及来源](SPARKSIM/device_models/final_rram/README.md)
- [发布事项](PUBLICATION_CHECKLIST.md)
