<h1 align="center">PACE: Phase-Aware Chunk Execution<br>for Robot Policies with Action Chunking</h1>

<h3 align="center">NeurIPS 2026</h3>

<p align="center">
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=Paper&amp;message=PDF&amp;color=red&amp;logo=arxiv&amp;logoColor=white" alt="Paper PDF" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project website" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&amp;message=HuggingFace&amp;color=yellow" alt="Hugging Face models" height="24"></a>
</p>

<p align="center">
  <a href="pace/media/overview.pdf"><img src="pace/media/overview.png" alt="PACE overview: sensitivity to fixed execution horizons, adaptive action-chunk execution, simulation and real-robot results, and execution horizons along a trajectory." width="100%"></a>
</p>

<a id="checkpoint-configuration-details"></a>OpenPI-PACE extends [OpenPI](https://github.com/Physical-Intelligence/openpi) with π₀.₅ configurations for all 50 RoboTwin tasks used in PACE. Training and inference use the upstream implementations and share the same [task configurations](pace/src/openpi_pace/configs.py), which specify camera inputs, action formats, normalization assets, and training settings.

The model takes three camera views and a 14-D robot state and returns 50 × 14 absolute joint and gripper commands. [RoboTwin-PACE](https://github.com/niejunnan25/RoboTwin-PACE#install-and-try-the-core) selects execution horizons from these predictions and runs the simulation evaluation.

[Installation](#install) · [Checkpoints](#download-a-checkpoint) · [Model serving](#serve-a-policy) · [Training](#training) · [Citation](#citation)

<a id="install"></a><a id="base-and-layout"></a><a id="reuse-openpi"></a>

## 🛠️ Installation

If you already use OpenPI, apply the [RoboTwin integration patch](pace/patches/openpi-robotwin.patch). It adds the 50 task configurations and model-serving adaptations for RoboTwin:

```bash
cd /path/to/openpi
source .venv/bin/activate

curl -fL \
  https://raw.githubusercontent.com/niejunnan25/Openpi-PACE/main/pace/patches/openpi-robotwin.patch \
  -o /tmp/openpi-robotwin.patch

git apply --check /tmp/openpi-robotwin.patch && \
  git apply /tmp/openpi-robotwin.patch
uv pip install --no-deps -e .
```

OpenPI-PACE already includes this patch. We validated it against [official OpenPI 2b1f377](https://github.com/Physical-Intelligence/openpi/tree/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea).

<a id="new-openpi"></a>

<details>
<summary>Installing OpenPI-PACE from scratch</summary>

OpenPI-PACE requires Python 3.11 or newer and uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependencies. On a Linux machine with an NVIDIA GPU, install uv, open a new terminal, and clone OpenPI-PACE next to your RoboTwin workspace:

```bash
git clone https://github.com/niejunnan25/Openpi-PACE.git OpenPI-PACE
cd OpenPI-PACE

git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

source .venv/bin/activate
```

The model server uses this `.venv`, while simulation runs in the RoboTwin Conda environment.

</details>

<a id="download-a-checkpoint"></a>

## 📦 Checkpoints

We release an EMA checkpoint at training step 30,000 for each task on [Hugging Face](https://huggingface.co/niejunnan25/pi05-robotwin), at about 12.4 GB per task. The [manifest](https://huggingface.co/niejunnan25/pi05-robotwin/blob/main/manifest.json) lists all tasks and files. A single-task evaluation needs one checkpoint; the full evaluation needs all 50.

From your OpenPI workspace with its model environment active, download the `adjust_bottle` checkpoint and shared tokenizer:

```bash
huggingface-cli download niejunnan25/pi05-robotwin \
  --include "adjust_bottle/*" tokenizer.model manifest.json TERMS.md NOTICE \
  --local-dir local/pi05-robotwin
```

Replace `adjust_bottle/*` to download another task, or reuse an existing copy. Keep each task's weights and matching normalization statistics together:

```text
local/pi05-robotwin/adjust_bottle/
  params/
  assets/robotwin/norm_stats.json
```

<a id="local-assets"></a><a id="use-existing-local-assets"></a>If the tokenizer is not already in OpenPI's default cache, copy it there:

```bash
mkdir -p ~/.cache/openpi/big_vision
cp local/pi05-robotwin/tokenizer.model ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

For a custom cache, set `OPENPI_DATA_HOME` and place the tokenizer at `big_vision/paligemma_tokenizer.model` within that directory.

The released checkpoints are ready for evaluation, with no training data, parameter conversion, or recomputation of normalization statistics needed.

<a id="serve-a-policy"></a>

## 🚀 Model serving

Start the model server in the OpenPI environment with the task configuration and checkpoint path:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_robotwin_adjust_bottle \
  --policy.dir local/pi05-robotwin/adjust_bottle
```

To switch tasks, update both `--policy.config` and `--policy.dir`. The first inference may take longer due to XLA compilation.

With the server running, open another terminal and install the communication client in the **RoboTwin Conda environment**:

```bash
conda activate RoboTwin
python -m pip install -e /path/to/openpi/packages/openpi-client
```

Run the [RoboTwin-PACE single-task example](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-one-task) with the same task and port. Each evaluation runs 100 episodes. The client selects PACE or Fixed; the server returns the full action chunk.

Stop the server with Ctrl-C after each run and start a fresh server for the next one.

For all 50 tasks, the [RoboTwin-PACE batch script](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-multiple-tasks) manages the model servers and simulation clients automatically.

In the RoboTwin workspace, edit [`pace/scripts/eval.yaml`](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/pace/scripts/eval.yaml): set `openpi_root` to your OpenPI directory and `checkpoint_root` to the directory containing the task checkpoints. `server_python` is relative to the OpenPI workspace, or can be an absolute interpreter path; the default is `.venv/bin/python`.

<a id="training"></a>

## 🏋️ Training

We fine-tuned π₀.₅ on 50 `demo_clean` demonstrations per task, updating all model parameters with batch size 32 and training seed 42.

Prepare demonstrations in LeRobot format under `robotwin/<task>` in your dataset root, such as `/path/to/datasets/robotwin/adjust_bottle`. In the OpenPI model environment, compute normalization statistics for the new training data and start training:

```bash
export HF_LEROBOT_HOME="/path/to/datasets"

python scripts/compute_norm_stats.py --config-name pi05_robotwin_adjust_bottle
python scripts/train.py pi05_robotwin_adjust_bottle --exp-name paper
```

Training assets and checkpoints are written to `pace/runs/`; evaluation uses the step-30,000 checkpoint.

<a id="citation"></a>

## Citation

If you use PACE in your research, please cite:

```bibtex
@inproceedings{nie2026pace,
  title     = {{PACE}: Phase-Aware Chunk Execution for Robot Policies with Action Chunking},
  author    = {Nie, Junnan and Li, Jiayi and Zhang, Jiachen and Lao, Junyi and
               Liu, Chenghao and Zhang, Tianle and Huang, Songfang},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {To appear}
}
```

<a id="acknowledgments-and-license"></a>

## Acknowledgments and license

Thanks to the [OpenPI](https://github.com/Physical-Intelligence/openpi) and [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) teams for their open-source work.

This repository uses the [Apache-2.0 License](LICENSE); bundled components retain their respective licenses. Model and tokenizer terms are provided in the model repository's `TERMS.md` and `NOTICE`.
