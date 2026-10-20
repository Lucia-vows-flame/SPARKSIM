# SPARKSIM

本目录为仿真器的运行工作目录。完整安装与使用教程集中在仓库根目录的 **[README](../README.md)**，包括：

- Conda 环境创建、CPU/CUDA 安装与检查；
- V7 网络、数据格式、预训练和固定精度 QAT；
- ASU/PKU 器件与硬件资源配置；
- 准确率、权重编程加推理评估；
- 两阶段调度、冻结回放和自定义算法接入；
- 指标口径、电路能耗账本、测试与故障排查。

从本目录运行仿真入口：

```bash
export MNSIM_PYTHON="$(command -v python)"
bash scripts/run_round_orchestrated_sim.sh --help
```

内部 Python 包保留名称 `MNSIM`。数据集和训练权重需自行提供。
