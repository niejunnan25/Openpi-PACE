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

本仓库在官方 [OpenPI](https://github.com/Physical-Intelligence/openpi) 上补充了 **50 个 RoboTwin 任务的 π₀.₅ 配置**，沿用其模型、训练和推理实现。

<a id="checkpoint-configuration-details"></a>

- 🧩 **任务配置**：集中在 [configs.py](pace/src/openpi_pace/configs.py)，包括相机与动作格式、归一化资产及训练参数；训练和加载权重推理共用这些配置。
- 🤖 **模型服务**：接收三路相机图像与 14 维状态，返回 **50 × 14 的绝对关节／夹爪指令**。
- ⚡ **PACE 执行**：由 [RoboTwin-PACE](https://github.com/niejunnan25/RoboTwin-PACE#install-and-try-the-core) 根据预测动作选择执行步数，并完成仿真评估。

🧭 [接入配置](#install) · [准备权重](#download-a-checkpoint) · [启动服务](#serve-a-policy) · [训练模型](#training)

<a id="install"></a>
## 🛠️ 接入 RoboTwin 任务配置

使用 Linux、NVIDIA GPU 和 Python 3.11 或更高版本。**已有可运行的 OpenPI 时，可以继续使用原来的工作区、环境和缓存。** 应用下面的补丁即可加入任务配置及注册入口，同时包含通信适配和训练时的归一化统计路径修正。

<a id="base-and-layout"></a>
补丁基于官方 OpenPI [2b1f377](https://github.com/Physical-Intelligence/openpi/tree/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea) 生成并验证；其他版本需确认接口兼容。已使用本仓库的读者无需再应用补丁，直接准备权重即可。

<a id="reuse-openpi"></a>
在现有 OpenPI 根目录执行，下面以已有的 `.venv` 为例：

```bash
cd /path/to/openpi
source .venv/bin/activate
curl -fL https://raw.githubusercontent.com/niejunnan25/Openpi-PACE/main/pace/patches/openpi-robotwin.patch \
  -o /tmp/openpi-robotwin.patch
git apply --check /tmp/openpi-robotwin.patch && git apply /tmp/openpi-robotwin.patch
uv pip install --no-deps -e .
```

最后一条命令重新注册本地包，沿用现有依赖版本。[补丁](pace/patches/openpi-robotwin.patch)只加入必要改动，保留原有配置；若检查未通过，先核对当前版本与本地修改。

<a id="new-openpi"></a>
<details>
<summary><strong>🌱 首次使用 OpenPI</strong></summary>

先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再克隆本仓库。建议与 RoboTwin 工作区放在同级目录：

```bash
git clone https://github.com/niejunnan25/Openpi-PACE.git OpenPI-PACE
cd OpenPI-PACE
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

本仓库已包含任务配置和配套改动，无需应用上面的补丁。模型服务使用这里的 `.venv`，仿真使用 RoboTwin 的 Conda 环境。

</details>

后续命令均在你正在使用的 OpenPI 工作区根目录、已激活的模型环境中执行；示例中的 `/path/to/openpi` 按实际位置替换。

<a id="download-a-checkpoint"></a>
## 📦 准备模型权重

权重托管在 [niejunnan25/pi05-robotwin](https://huggingface.co/niejunnan25/pi05-robotwin)。每任务约 **12.4 GB**，任务名和文件清单见 [manifest.json](https://huggingface.co/niejunnan25/pi05-robotwin/blob/main/manifest.json)。已有相同权重可直接复用；单任务评估只需下载对应任务，完整评估需要全部 50 个任务。

以 `adjust_bottle` 为例，下载任务权重和共用分词器：

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

<a id="local-assets"></a>
<a id="use-existing-local-assets"></a>
首次使用时，将分词器放入 OpenPI 默认缓存；已有该文件可跳过：

```bash
mkdir -p ~/.cache/openpi/big_vision
cp local/pi05-robotwin/tokenizer.model ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

已有自定义缓存时，继续使用 `OPENPI_DATA_HOME`，分词器位于该目录的 `big_vision/paligemma_tokenizer.model`。每个任务使用第 **30,000** 步的 EMA 权重及其配套归一化统计；保留 `params/` 和 `assets/` 完整即可，评估无需训练数据、参数转换或重新计算归一化。

<a id="serve-a-policy"></a>
## 🚀 启动模型服务

使用 OpenPI 原有的服务入口，指定任务配置和权重路径：

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_robotwin_adjust_bottle \
  --policy.dir local/pi05-robotwin/adjust_bottle
```

这里使用 GPU 0、端口 8000。切换任务时，同时修改 `--policy.config` 和 `--policy.dir`。首次推理需要等待 XLA 编译。

保持服务运行，在另一个终端的 **RoboTwin Conda 环境**中，安装当前 OpenPI 工作区的通信客户端：

```bash
conda activate RoboTwin
python -m pip install -e /path/to/openpi/packages/openpi-client
```

然后按 [RoboTwin-PACE 单任务示例](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-one-task)启动仿真评估，使用相同的任务和端口。每次评估 100 回合；完成后用 Ctrl-C 停止服务，下一次评估重新启动。**PACE 或 Fixed 由客户端选择**，服务端始终返回完整动作块。

运行全部 50 个任务时，使用 [RoboTwin-PACE 批量评测脚本](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-multiple-tasks)，由它自动管理模型服务。复用已有 OpenPI 工作区时，将 `eval.env` 中的 `OPENPI_ROOT`、`SERVER_PYTHON` 和 `CHECKPOINT_ROOT` 指向实际路径。

<a id="training"></a>
## 🏋️ 训练自己的模型

准备 LeRobot 格式的 RoboTwin 演示数据，按 `robotwin/<task>` 放在数据集根目录下。例如，`adjust_bottle` 的路径为 `/path/to/datasets/robotwin/adjust_bottle`。在 OpenPI 模型环境中执行：

```bash
export HF_LEROBOT_HOME="/path/to/datasets"
python scripts/compute_norm_stats.py --config-name pi05_robotwin_adjust_bottle
python scripts/train.py pi05_robotwin_adjust_bottle --exp-name paper
```

发布配置采用每任务 50 条 `demo_clean` 演示、batch size 32、训练 seed 42，进行 π₀.₅ 全参数微调。训练资产与权重保存在 `pace/runs/`，评估使用第 30,000 步的权重。这里重新计算归一化统计用于训练新模型；评估已发布模型时直接使用其配套文件。

<a id="acknowledgments-and-license"></a>
## 🤝 致谢与许可证

感谢 Physical Intelligence 的 [OpenPI](https://github.com/Physical-Intelligence/openpi) 和 [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) 作者。本仓库代码采用 [Apache-2.0 许可证](LICENSE)，附带组件保留各自许可证；模型和分词器的许可条款见模型仓库的 `TERMS.md` 与 `NOTICE`。
