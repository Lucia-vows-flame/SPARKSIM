# 自定义权重调度方案格式 v1

自定义算法只需输出下面的 JSON。算法使用什么输入、怎样获取模型或硬件信息，均由算法作者决定；无需实现 MNSIM 的 Python 策略类、注册算法或提交统一的“调度问题文件”。

仿真器加载方案，验证它与当前模型、权重和硬件资源的关系，再转换为内部 `ScheduleArtifact`。驻留状态、器件转换代价及执行账本由仿真器生成。原有内置策略和 artifact 回放入口继续使用。

## 最小方案

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

示例中的 ID 仅用于说明。实际 `round_id` 和逻辑 tile ID 必须对应当前运行产生的计划。`assignments` 的键是该轮逻辑 tile ID，值是整数物理 tile ID。逻辑 tile ID 是不透明标识符，不应靠拆解其名称推断网络结构。当前 runner 使用 `tile:<原始 placement tile ID>`。

| 字段 | 必需 | 含义 |
| --- | --- | --- |
| `format` | 是 | 固定为 `mnsim.weight_schedule` |
| `schema_version` | 是 | 整数 `1`，未知版本拒绝加载 |
| `policy_id` | 是 | 算法名称；首字符为字母或数字，后续可含字母、数字、`.`、`_`、`-` |
| `rounds` | 是 | 非空数组，数组顺序就是请求的执行顺序 |
| `rounds[].round_id` | 是 | 当前计划中的加权轮次 ID |
| `rounds[].assignments` | 是 | 非空的逻辑 tile → 物理 tile 映射 |
| `rounds[].metadata` | 否 | 算法附加记录，不参与执行 |
| `identity` | 否 | 可选的模型、权重、配置等身份约束；提供时必须匹配 |
| `metadata` | 否 | 方案附加记录，不参与执行 |

结构 Schema 见 [weight-schedule-v1.schema.json](../schemas/weight-schedule-v1.schema.json)，示例文件见 [weight-schedule-v1.json](../examples/weight-schedule-v1.json)。Schema 检查结构；下面的资源、覆盖和依赖约束由仿真器进行语义校验。

## v1 支持的决策与边界

v1 支持**整 tile 的物理映射，以及现有计划允许的轮次排序**。它保留原 placement 的 tile 内 PE/group/polarity 分布、fragment 分组、SNN 时间步和 LIF/merge 边界。

- 当前计划的每个加权轮次必须出现一次，每轮全部逻辑 tile 必须映射一次。
- 同一轮不同逻辑 tile 不能占用同一物理 tile。不同轮次可复用同一物理 tile；其驻留状态连续维护。
- 物理 ID 必须属于可用 tile 池，并满足激活 tile 上限。当前 runner 的池是原 placement 使用的 tile ID 集合；v1 不自动扩展到硬件其他空闲 tile。
- 轮次仅能在仿真器允许的连续重排块中排序，且满足前驱依赖。固定执行项、LIF 和 merge 等边界不能跨越。自定义 JSON 不能修改这些约束。
- v1 不接受 tile 内槽位调整、跨轮重新分组、预加载或清除操作。这些执行语义如需纳入格式，应通过后续版本显式增加。

布尔值不能用作整数，物理 tile ID 必须非负。重复 JSON 键、重复 round ID、未知字段、非法版本，以及非有限 JSON 数值均会拒绝。所有附加信息嵌套保存为 `custom_metadata`，不能覆盖仿真器的执行字段。

## 可选身份约束

`identity` 允许 `model_id`、`base_round_plan_sha256`、`model_ir_sha256`、`weights_sha256`、`sim_config_sha256`、`device_model_id`，未提供的字段不做身份匹配。摘要必须为 64 位小写十六进制字符串，并使用内部 artifact 的同名身份口径。

其中 `base_round_plan_sha256` 指**未应用本次自定义调度的原始计划**；`model_ir_sha256` 为规范化 ModelIR JSON 的摘要；`sim_config_sha256` 为配置文件摘要。当前 runner 的 `weights_sha256` 是 `{"checkpoint_sha256": "<权重文件 SHA-256>"}` 这个规范化 JSON 的摘要，不能直接填权重文件摘要。建议从同配置的顺序 artifact 复制身份值。

没有身份字段的方案仍需通过完整覆盖、资源和执行依赖检查。

## 执行入口

在现有 round-orchestrated 命令上增加 `--custom-schedule-in`：

```bash
MNSIM_PYTHON=../.conda-envs/mnsim-py312/bin/python \
  bash scripts/run_round_orchestrated_sim.sh \
  --network <network> --weights <checkpoint> --num-classes <classes> \
  --sim-config <SimConfig.ini> --output-dir <output-dir> \
  --device cpu --run-hardware --skip-accuracy \
  --custom-schedule-in my_schedule.json
```

正常执行会保存 `<output-dir>/custom.schedule.artifact.json`，然后交给原 Phase-2 执行。`round_plan.json` 保存实际执行顺序；运行结果登记 artifact 路径及摘要。能耗和延迟来自现有物理写入账本和读取执行模型；自定义方案不能声明自己的代价来覆盖仿真结果。

如果只想验证、转换并保存，追加：

```bash
--schedule-only --schedule-plan-out frozen.schedule.artifact.json
```

已经冻结的 artifact 可通过原 `--infer-with-schedule --schedule-plan-in <artifact>` 入口回放；该模式不能同时传 `--custom-schedule-in`。

## 可选模板导出与 Python API

为了获取真实 ID，可先使用现有 `--schedule-only --schedule-policy seq --schedule-plan-out sequential.artifact.json` 生成顺序 artifact，再导出精简模板。这只是便利工具，不规定算法输入：

```python
from MNSIM.Scheduling.weight_scheduling import (
    load_schedule_artifact,
    custom_schedule_from_artifact,
    save_custom_schedule,
)

artifact = load_schedule_artifact("sequential.artifact.json")
template = custom_schedule_from_artifact(artifact, policy_id="my_scheduler")
save_custom_schedule("my_schedule.json", template)
```

导出只保留 tile 映射和轮次顺序，不复制内部 PE/group 重排，因此用于生成模板，而非替代完整 artifact 回放。导出会保留已有的非空身份值，但省略原始计划摘要，避免把已重排 artifact 的计划摘要误用到原始计划。

高级调用者可使用 `load_custom_schedule(path)` 和 `generate_weight_schedule_from_custom(problem, schedule)`，复用已有 `WeightScheduleProblem` 生成调度记录与成本。低层函数本身不准备完整 Phase-2 placement：runner 还需重映射 placement、重算 reducer/LIF 所有权并冻结 RoundPlan，才能得到完整可执行的 schema v2 artifact。自定义格式 v1 和内部 artifact v2 是两个独立版本。

