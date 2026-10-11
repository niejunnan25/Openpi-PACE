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

本仓库提供 RoboTwin **50 个任务的 π₀.₅ 配置、模型服务与训练入口**。模型每次返回完整的 50 步动作块，由 [RoboTwin-PACE](https://github.com/niejunnan25/RoboTwin-PACE) 负责 PACE 执行步数选择和仿真评估。

🧭 [安装](#install) · [准备权重](#download-a-checkpoint) · [启动服务](#serve-a-policy) · [训练与进阶用法](#training)

<a id="install"></a>
## 🛠️ 安装

使用 Linux、NVIDIA GPU 和 Python 3.11 或更高版本。模型服务通过 [uv](https://docs.astral.sh/uv/getting-started/installation/) 安装在本仓库的 `.venv` 中，与 RoboTwin 的 Conda 环境分开。

```bash
git clone https://github.com/niejunnan25/Openpi-PACE.git OpenPI-PACE
cd OpenPI-PACE
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

已有本仓库时直接进入目录，跳过克隆。建议与 RoboTwin 工作区放在同级目录，以使用调度器的默认路径。仿真侧的 PACE 和通信客户端安装见 [RoboTwin-PACE 环境准备](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#prepare-environments-and-models)。

<a id="download-a-checkpoint"></a>
## 📦 准备模型权重

权重托管在 [niejunnan25/pi05-robotwin](https://huggingface.co/niejunnan25/pi05-robotwin)。每任务约 **12.4 GB**，任务名和文件清单见 [manifest.json](https://huggingface.co/niejunnan25/pi05-robotwin/blob/main/manifest.json)。已有相同权重可直接复用；单任务评估只需下载对应任务，完整评估需要全部 50 个任务。

在已激活的 OpenPI 环境中，下载 `adjust_bottle` 权重和共用分词器：

```bash
huggingface-cli download niejunnan25/pi05-robotwin \
  --include "adjust_bottle/*" tokenizer.model manifest.json TERMS.md NOTICE \
  --local-dir local/pi05-robotwin
```

将 `adjust_bottle/*` 替换为其他任务即可。下载后的任务目录应包含：

```text
local/pi05-robotwin/adjust_bottle/
  params/
  assets/robotwin/norm_stats.json
```

首次使用时，将分词器放入 OpenPI 默认缓存；已有该文件可跳过：

```bash
mkdir -p ~/.cache/openpi/big_vision
cp local/pi05-robotwin/tokenizer.model ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

每个任务使用第 **30,000** 步的 EMA 权重及其配套归一化统计，保持 `params/` 和 `assets/` 完整即可。评估不需要训练数据、参数转换或重新计算归一化。自定义缓存和已有原始权重的整理方式见[进阶说明](pace/docs/reference.md#local-assets)。

<a id="serve-a-policy"></a>
## 🚀 启动模型服务

在 OpenPI-PACE 根目录，激活 `.venv` 后启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_robotwin_adjust_bottle \
  --policy.dir local/pi05-robotwin/adjust_bottle
```

这里使用 GPU 0、端口 8000。切换任务时，同时修改 `--policy.config` 和 `--policy.dir`。首次推理需要等待 XLA 编译。

保持服务运行，在另一个终端按 [RoboTwin-PACE 单任务示例](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-one-task)启动仿真客户端，使用相同的任务和端口。每次评估 100 回合；完成后用 Ctrl-C 停止服务，下一次评估重新启动。**PACE 或 Fixed 由客户端选择**，服务端始终返回完整动作块。

运行多个任务时，直接使用 [RoboTwin-PACE 调度器](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-multiple-tasks)，由它自动管理各模型服务。回合结果和日志见[查看结果](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#read-the-results)。

<a id="training"></a>
<a id="checkpoint-configuration-details"></a>
<a id="use-existing-local-assets"></a>
<a id="base-and-layout"></a>
<a id="tests"></a>
## 📚 训练与进阶用法

- 🏋️ [训练新模型](pace/docs/reference.md#training)：演示数据要求、训练命令与论文超参数。
- ⚙️ [模型配置](pace/docs/reference.md#checkpoint-configuration-details)：任务映射、观测与动作格式、推理默认值。
- 🗃️ [使用已有本地资产](pace/docs/reference.md#local-assets)：自定义缓存、原始权重和数据集链接。
- 🧪 [测试与上游版本](pace/docs/reference.md#base-and-layout)：实现范围、目录和验证命令。

<a id="acknowledgments-and-license"></a>
## 🤝 致谢与许可证

感谢 Physical Intelligence 的 [OpenPI](https://github.com/Physical-Intelligence/openpi) 和 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 作者。本仓库代码采用 [Apache-2.0 许可证](LICENSE)，附带组件保留各自许可证；模型和分词器的许可条款见模型仓库的 `TERMS.md` 与 `NOTICE`。
