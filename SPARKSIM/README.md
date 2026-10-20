# SPARKSIM

SPARKSIM 的源码与运行目录。安装步骤和完整教程见[仓库首页](../README.md)。

- [安装与环境配置](../README.md#安装)
- [已有权重的硬件评估](../README.md#快速开始)
- [软件训练与 QAT](../README.md#训练与量化)
- [自定义调度算法](../README.md#自定义调度算法)
- [结果与指标](../README.md#结果与指标)

在本目录查看命令行参数：

```bash
export MNSIM_PYTHON="$(command -v python)"
bash scripts/run_round_orchestrated_sim.sh --help
```

内部 Python 包保留名称 `MNSIM`。数据集和训练权重需自行提供。
