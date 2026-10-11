<h1 align="center">🤖 PACE</h1>
<h3 align="center">面向动作分块机器人策略的<br>阶段感知动作块执行</h3>

<p align="center"><strong>OpenPI-PACE</strong><br>各任务的 π₀.₅ 策略 &middot; 模型服务 &middot; 训练</p>

<p align="center">
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=Paper&amp;message=PDF&amp;color=red&amp;logo=arxiv&amp;logoColor=white" alt="论文 PDF" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/badge/Project-Website-blue" alt="项目主页" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&amp;message=HuggingFace&amp;color=yellow" alt="Hugging Face 模型" height="24"></a>
</p>

<p align="center">
  <a href="pace/media/overview.pdf"><img src="pace/media/overview.png" alt="PACE 总览：固定执行步数的敏感性、自适应执行预测动作块、仿真与真机结果，以及单条轨迹中执行步数的变化。" width="100%"></a>
</p>
<p align="center"><sub><strong>PACE 方法总览。</strong>根据预测动作中的运动变化，决定执行动作块的前多少步，再重新查询策略。点击图片可查看矢量 PDF。</sub></p>

## ✨ 认识 OpenPI-PACE

本仓库为 RoboTwin 上的 PACE 评估提供**各任务的 π₀.₅ 配置**，以及基于 OpenPI 官方入口的**模型服务**。服务端返回完整动作块；PACE 执行步数选择器、仿真客户端和 GPU 调度器位于 **RoboTwin-PACE**。完整流程请从 [RoboTwin-PACE 评估指南](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md)开始。

> 🧭 **阅读路线**：使用已发布权重时，依次完成[安装](#install)、[下载权重](#download-a-checkpoint)和[启动模型服务](#serve-a-policy)。[训练](#training)与[整理已有本地资产](#use-existing-local-assets)是另外两种使用方式。

<a id="install"></a>
## 🛠️ 安装

JAX 模型服务使用 Linux 和 NVIDIA GPU，Python 环境与 RoboTwin 仿真器分开。本仓库要求 Python 3.11 或更高版本。克隆仓库，初始化上游子模块，然后使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 安装：

```bash
git clone https://github.com/niejunnan25/Openpi-PACE.git OpenPI-PACE
cd OpenPI-PACE
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

如果已按 RoboTwin-PACE 指南获取仓库，请从 `cd /path/to/OpenPI-PACE` 开始，跳过上面的克隆命令。显式指定的本地目录名与配套指南和调度器默认路径一致。

请从本仓库安装，以便官方服务入口能够导入 `openpi_pace` 及其 50 个任务配置。依赖版本由上游锁文件固定。其他平台细节见[上游安装说明](https://github.com/Physical-Intelligence/openpi/blob/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea/README.md#installation)。

在另一个终端，激活已有的 **RoboTwin Conda 环境**并安装轻量客户端（已按 RoboTwin-PACE 指南安装过可跳过；环境名按实际替换）：

```bash
conda activate RoboTwin
python -m pip install -e /path/to/OpenPI-PACE/packages/openpi-client
```

<a id="download-a-checkpoint"></a>
## 📦 下载模型权重

模型权重托管在 [niejunnan25/pi05-robotwin](https://huggingface.co/niejunnan25/pi05-robotwin)。可用任务和文件清单以模型卡及 [manifest.json](https://huggingface.co/niejunnan25/pi05-robotwin/blob/main/manifest.json) 为准。每个任务约 12.4 GB，推理不需要训练演示数据。

在 OpenPI-PACE 根目录激活其 Python 环境后，下载 `adjust_bottle` 权重和共用分词器：

```bash
huggingface-cli download niejunnan25/pi05-robotwin \
  --include "adjust_bottle/*" tokenizer.model manifest.json TERMS.md NOTICE \
  --local-dir local/pi05-robotwin
```

`huggingface-cli` 由锁文件指定版本的 Hugging Face Hub 提供。需要其他任务时，将 `adjust_bottle/*` 替换为对应任务的匹配模式；多个模式可同时选择多个任务。运行完整 RoboTwin 队列前，需在本地准备全部 50 个任务的权重。下载后的目录结构为：

```text
local/pi05-robotwin/
  tokenizer.model
  manifest.json
  adjust_bottle/
    params/
    assets/robotwin/norm_stats.json
```

分词器使用 OpenPI 默认缓存 `~/.cache/openpi`，无需设置额外环境变量；已有该文件时可直接复用：

```bash
mkdir -p ~/.cache/openpi/big_vision
cp local/pi05-robotwin/tokenizer.model ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

如果已有自定义缓存，继续使用原来的 `OPENPI_DATA_HOME` 设置，并将分词器放在该目录下的 `big_vision/paligemma_tokenizer.model`。

每个任务的 `params/` 和已保存的归一化统计应一起保留，无需转换参数或重新计算归一化统计。模型与分词器的许可条款见下载的 `TERMS.md` 和 `NOTICE`。

<a id="serve-a-policy"></a>
## 🚀 启动模型服务

在 OpenPI-PACE 根目录，激活其 Python 环境后运行：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_robotwin_adjust_bottle \
  --policy.dir local/pi05-robotwin/adjust_bottle
```

`--policy.config` 选择任务配置，`--policy.dir` 指向该任务的权重与归一化统计。`CUDA_VISIBLE_DEVICES` 选择模型使用的 GPU。`XLA_PYTHON_CLIENT_MEM_FRACTION=0.2` 是 OpenPI/JAX 的显存占比设置，与原调度器一致；每张 GPU 能容纳多少组服务端／客户端取决于可用显存。首次推理可能需要先编译 XLA 内核，再生成动作。

在第二个终端激活已有的 **RoboTwin Conda 环境**，进入已加入 `pace/` 的 RoboTwin 工作区运行（也可以是完整克隆的 RoboTwin-PACE）：

```bash
conda activate RoboTwin
cd /path/to/RoboTwin
python pace/scripts/evaluate.py \
  --task adjust_bottle --method pace --gpu 0 --seed 0 --port 8000
```

任务名和端口必须与服务端一致。这条命令评估 **100 个回合**。单独评估 Fixed H=25 时，使用 `--method fixed --horizon 25`。每次评估结束后，用 Ctrl-C 停止服务，并为下一次评估重新启动服务。策略的默认随机数 seed 为 0，去噪步数为 10；客户端的评估 seed 是另一项独立设置。

评估全部 50 个任务时，请使用 [RoboTwin-PACE 调度器](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-multiple-tasks)，由它负责启动和清理各组服务端／客户端。按指南在 `pace/configs/eval.env` 中指定 RoboTwin Conda 环境的 Python 路径；同级目录布局下，其余路径可沿用默认值。只有使用自定义分词器缓存时，才需要保留对应的 `OPENPI_DATA_HOME` 设置。回合结果和日志见[结果文件说明](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#read-the-results)。

<a id="checkpoint-configuration-details"></a>
## ⚙️ 模型配置说明

每个任务使用自身训练第 30,000 步的 EMA 权重和归一化统计。配置名为 `pi05_robotwin_<task>`，本地 LeRobot 数据集 ID 为 `robotwin/<task>`。所有配置的归一化资产 ID 都是 `robotwin`，但对应文件的数值各任务不同。`beat_block_hammer` 使用历史 `_50` 权重和 50 条演示数据集，不使用更早的七条演示实验。

数据适配器使用三个相机（`cam_high`、`cam_left_wrist`、`cam_right_wrist`）及 14 维关节／夹爪状态。`adapt_to_pi=False`，不进行 Aloha 关节符号翻转或夹爪转换。模型处理流程内部的 12 个机械臂关节维度使用增量表示，两个夹爪维度保持绝对值。返回给客户端的动作已恢复到环境空间，为 **50 × 14 的绝对关节／夹爪指令**。

<a id="use-existing-local-assets"></a>
## 🗃️ 使用已有本地资产

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

目标目录必须尚不存在。工具会为参数创建符号链接，并逐字节复制较小的归一化文件，原始文件保持不变。分词器缓存需按前面的说明单独准备，也可以使用已有的 OpenPI 缓存。

若用于训练，可在每条记录中补充 `dataset_dir`，再添加 `--dataset-root local/datasets` 创建对应的数据集链接。对于按前文下载得到的权重目录，无需运行此辅助工具。

<a id="training"></a>
## 🏋️ 训练

论文中的每个任务使用 50 条 `demo_clean` 演示，采样频率为 50 Hz。π₀.₅ 采用全参数微调，优化器为 AdamW，batch size 为 32，训练 seed 为 42，EMA 衰减为 0.99。学习率先用 1,000 步预热到 2.5e-5，余弦衰减调度的 `decay_steps` 为 30,000，末端学习率为 2.5e-6。总训练长度为 30,005 步，评估使用第 30,000 步的权重。

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
  tests/                     # 配置、数据变换与通信检查
  README.md                  # 软件包目录索引
```

`pace/` 之外的修改包括：注册新增配置与软件包、按配置的资产 ID 保存归一化统计、关闭服务端和客户端的 WebSocket 心跳以适应阻塞式仿真和模型编译，以及更新 README 和 Git 忽略项。模型推理、数据变换与权重加载沿用上游代码。

<a id="tests"></a>
## 🧪 测试

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python -m pytest pace/tests -m "not manual"
```

<a id="acknowledgments-and-license"></a>
## 🤝 致谢与许可证

本项目基于 Physical Intelligence 的 [OpenPI](https://github.com/Physical-Intelligence/openpi)。感谢其作者开放策略模型、训练代码与模型服务实现，也感谢 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 作者提供仿真器。PACE 之外的 OpenPI 通用示例见[上游指南](https://github.com/Physical-Intelligence/openpi/blob/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea/README.md)。

OpenPI 代码采用 Apache-2.0 许可证，详见 [LICENSE](LICENSE)。附带组件保留各自的许可证。模型权重和数据集的分发条款与源代码许可证分开。
