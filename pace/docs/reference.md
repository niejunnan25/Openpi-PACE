# OpenPI-PACE 技术参考

运行已发布权重见[首页指南](../../README.md)。本页说明模型配置、已有资产的整理方式、训练与测试。

[模型配置](#checkpoint-configuration-details) · [本地资产](#local-assets) · [训练](#training) · [上游版本](#base-and-layout) · [测试](#tests)

<a id="checkpoint-configuration-details"></a>
## ⚙️ 模型配置说明

每个任务使用自身训练第 30,000 步的 EMA 权重和归一化统计。配置名为 `pi05_robotwin_<task>`，本地 LeRobot 数据集 ID 为 `robotwin/<task>`。所有配置的归一化资产 ID 都是 `robotwin`，但对应文件的数值各任务不同。`beat_block_hammer` 使用历史 `_50` 权重和 50 条演示数据集，不使用更早的七条演示实验。

数据适配器使用三个相机（`cam_high`、`cam_left_wrist`、`cam_right_wrist`）及 14 维关节／夹爪状态。`adapt_to_pi=False`，不进行 Aloha 关节符号翻转或夹爪转换。模型处理流程内部的 12 个机械臂关节维度使用增量表示，两个夹爪维度保持绝对值。返回给客户端的动作已恢复到环境空间，为 **50 × 14 的绝对关节／夹爪指令**。

策略默认随机数 seed 为 0，去噪步数为 10，与客户端的评估 seed 分开。`XLA_PYTHON_CLIENT_MEM_FRACTION=0.2` 沿用原调度器的 JAX 显存占比设置；可容纳的并发数取决于 GPU 可用显存。

<a id="local-assets"></a>
## 分词器缓存与已有资产

默认分词器路径为 `~/.cache/openpi/big_vision/paligemma_tokenizer.model`。若已有自定义缓存，继续使用 `OPENPI_DATA_HOME`，将分词器放在该目录的 `big_vision/paligemma_tokenizer.model`。调度时在 `eval.env` 中导出该变量，供服务进程继承。

<a id="use-existing-local-assets"></a>
### 整理原始权重

如果已有原始权重，可以使用可选辅助工具将它们整理为相同的任务目录结构。先创建一个被 Git 忽略的 JSON 清单：

```json
[
  {
    "task": "adjust_bottle",
    "checkpoint_dir": "/path/to/original/30000",
    "norm_stats": "/path/to/original/30000/assets/original-id/norm_stats.json"
  }
]
```

```bash
python pace/scripts/prepare_local.py --manifest local/sources.json \
  --checkpoint-root local/pi05-robotwin
```

目标目录必须尚不存在。工具会为参数创建符号链接，并逐字节复制较小的归一化文件，原始文件保持不变。分词器缓存需按首页的说明单独准备，也可以使用已有的 OpenPI 缓存。

若用于训练，可在每条记录中补充 `dataset_dir`，再添加 `--dataset-root local/datasets` 创建对应的数据集链接。对于按首页步骤下载得到的权重目录，无需运行此辅助工具。

<a id="training"></a>
## 🏋️ 训练

论文中的每个任务使用 50 条 `demo_clean` 演示，采样频率为 50 Hz。π₀.₅ 采用全参数微调，优化器为 AdamW，batch size 为 32，训练 seed 为 42，EMA 衰减为 0.99。学习率先用 1,000 步预热到 2.5e-5，余弦衰减调度的 `decay_steps` 为 30,000，末端学习率为 2.5e-6。总训练长度为 30,005 步，评估使用第 30,000 步的权重。

在 OpenPI-PACE 根目录激活模型环境后执行：

```bash
export HF_LEROBOT_HOME="/absolute/path/to/OpenPI-PACE/local/datasets"
python scripts/compute_norm_stats.py --config-name pi05_robotwin_adjust_bottle
python scripts/train.py pi05_robotwin_adjust_bottle --exp-name paper
```

训练资产与权重写入 `pace/runs/`。虽然内部资产 ID 相同，每个任务配置仍使用自己的资产目录。重新计算归一化统计用于训练新模型；评估已发布模型时，应使用该权重配套保存的统计。

<a id="base-and-layout"></a>
## 🗂️ 上游版本与目录结构

本仓库基于官方 OpenPI 提交 `2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea`（2025-12-05）。模型、Aloha 适配器、权重加载器和策略实现与已审计的实验实现一致。所有 OpenPI 内置配置仍可使用，另新增论文的 50 个任务配置；不包含其他私有实验及其配置。

```text
pace/
  src/openpi_pace/configs.py   # 共用设置与 50 个任务名称
  scripts/prepare_local.py    # 可选：为已有资产建立本地目录与链接
  media/                     # 论文总览图与矢量 PDF
  docs/                      # 模型配置、训练与本地资产说明
  tests/                     # 配置、数据变换与通信检查
  README.md                  # 软件包目录索引
```

`pace/` 之外的修改包括：注册新增配置与软件包、按配置的资产 ID 保存归一化统计、关闭服务端和客户端的 WebSocket 心跳以适应阻塞式仿真和模型编译，以及更新 README 和 Git 忽略项。模型推理、数据变换与权重加载沿用上游代码。

<a id="tests"></a>
## 🧪 测试

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python -m pytest pace/tests -m "not manual"
```

## 安装补充

本仓库同时安装 `openpi` 与 `openpi_pace`，保留全部官方配置并注册 50 个 PACE 任务配置。依赖版本由锁文件固定；其中 Hugging Face Hub 提供 `huggingface-cli`。更多平台细节见[上游安装说明](https://github.com/Physical-Intelligence/openpi/blob/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea/README.md#installation)。

已有原始权重可按本页工具建立链接；已按模型仓库布局下载的权重无需再次整理。
