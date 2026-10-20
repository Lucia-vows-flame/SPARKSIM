# 残差 SNN：实现与运行约定

## 模型与准确率路径

新增模型名为 `resnet18_snn_v7_phase6`。四个 stage 的 BasicBlock 数量为
`[2, 2, 2, 2]`，通道为 `[64, 128, 256, 512]`，输入为 `[B,30,6,32,32]`，输出为 8 类。
stem 使用 3×3 stride-2 卷积，stage 2–4 的首个 block 使用 stride-2 卷积和
1×1 stride-2 shortcut。末端直接 flatten 后接 FC/LIF，用脉冲计数分类。
这些输入、stem 和分类头是项目适配；BasicBlock 的计算顺序对齐 SpikingJelly。
不使用 pooling，也不改变已有六个顺序 SNN 的结构。

```text
main:     conv -> BN -> LIF -> conv -> BN ─┐
shortcut: identity / (1×1 conv -> BN) ─────┼-> element_sum -> LIF
```

`residual_snn.py` 负责显式构图。仍然使用原 MNSIM 的层类型和相对输入索引；
不引入残差专用硬件层。`NetworkGraph` 的两个 sequence 方法允许二输入
`element_sum`，普通 ANN forward 保持原逻辑。

浮点预训练不引入 post-add 量化。QAT/PIM 复用 MNSIM 的 BN/post-add 量化器。
各分支张量表示反量化实数，因此不同 code scale 可以相加；identity 分支仍是
0/1 脉冲。拒绝广播形状和缺失的量化状态，不把不同尺度的原始整数码直接相加。
节点输出域由各自校准范围决定；不能用某个分支的 ADC 累加器域替代合并输出域。

## 性能路径

复用原有 Phase 1/2、TCG、`element_sum`、数字 LIF 成本模型和 T 次执行循环。
SNN 专用适配器 `phase1/residual_metadata.py` 补齐 BN、merge、LIF 的 shape/domain。
归属策略明确为 `main_branch_reducer`：合并后的 LIF 按主分支最后一个卷积的
输出归属计数，不把两条分支的神经元数量相加。Phase 1 验证该 owner 是实际主分支
祖先；跨轮 placement 信息由既有 executor 维护。

新增边界适配修复：图中稳定的 producer ID 必须解析到当前输入视图中带
sample/timestep 的边界记录。否则跨轮的 residual branch 会从输入计数中丢失。
BN 和合并节点的边界输出携带各自 domain 和 bitwidth。

成本范围沿用原 MNSIM：加法、weighted 层和数字 LIF 已建模；BN 与再量化的额外
算术成本未单独表征。报告 `accounting_scope.residual_snn` 明确记录这一点。
主分支 reducer 归属是模型中的显式放置策略，不代表新电路表征结论。
残差 merge 使用现有 MNSIM `element_sum` 的数字结构算子，不引入残差专用的
模拟电路模型。

## 训练与评估

已有 `scripts/train_snn_v7.py` 和 `scripts/finetune_snn_v7.py` 均接受新模型名，
继续使用 pretrain 做软件训练、finetune 做 FIX_TRAIN 微调、test 做评估。
硬件 domain 需要已训练/校准的 BN、merge 范围；纯浮点预训练权重不能静默生成范围。
标准流程是预训练后 QAT，再用 QAT checkpoint 做硬件评估。

```bash
MNSIM_PYTHON=../.conda-envs/mnsim-py312/bin/python \
bash scripts/run_round_orchestrated_sim.sh \
  --network resnet18_snn_v7_phase6 \
  --weights /path/to/resnet18_snn_v7_phase6_software_qat_finetune_best.pth \
  --num-classes 8 --sim-config SimConfig.ini \
  --output-dir runs/resnet18_hardware --skip-accuracy --run-hardware --device cpu
```

## 验证记录（2026-10-03）

- identity/projection 小块分别与 SpikingJelly 官方 BasicBlock 的多步输出逐元素一致；
  重复独立序列、QAT 梯度、PIM 输出、非法 shape/domain 都有测试。
- 真实 MNSIM 后端验证 T=30 的残差块；小资源配置验证跨轮分支合并。
- 完整 ResNet18 在 V7 上完成单批预训练、单批 QAT、单样本 Original/PIM 评估。
  这是连通性冒烟，准确率为 0，不是训练完成后的精度结果。
- 完整硬件报告包含 2 个物理轮次、1474 个累计调度物理 crossbar、T=30；
  240 次残差合并（8×30）、540 次 LIF 节点调用（18×30），LIF 神经元更新数 4178160。
- 冒烟产物位于 `runs/resnet18_snn_implementation_smoke/`，由 Git 忽略；
  预训练和 QAT 各自保存 checkpoint，accuracy/hardware 各自保存报告。

单批验证使用 CPU；没有停止或重启现有 GPU 训练作业。本次不自动启动长期 ResNet 训练。
