<h1 align="center">🤖 PACE</h1>
<h3 align="center">Phase-Aware Chunk Execution for Robot Policies<br>with Action Chunking</h3>

<p align="center">
  <strong>OpenPI-PACE</strong><br>
  Task-specific π₀.₅ policies &middot; Model serving &middot; Training
</p>

<p align="center">
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=Paper&amp;message=PDF&amp;color=red&amp;logo=arxiv&amp;logoColor=white" alt="Paper PDF" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project website" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&amp;message=HuggingFace&amp;color=yellow" alt="Hugging Face models" height="24"></a>
</p>

<p align="center">
  <a href="pace/media/overview.pdf"><img src="pace/media/overview.png" alt="PACE overview: sensitivity to fixed execution horizons, adaptive action-chunk execution, simulation and real-robot results, and execution horizons along a trajectory." width="100%"></a>
</p>
<p align="center">
  <sub><strong>PACE at a glance.</strong> Use motion changes in a predicted action chunk to decide how many steps to execute before querying the policy again.<br>Click the figure for the vector PDF.</sub>
</p>

## ✨ About OpenPI-PACE

This repository adds **π₀.₅ configurations for 50 RoboTwin tasks** to official [OpenPI](https://github.com/Physical-Intelligence/openpi), using its model, training, and inference implementations.

<a id="checkpoint-configuration-details"></a>

- 🧩 **Task configurations** — [configs.py](pace/src/openpi_pace/configs.py) defines camera and action formats, normalization assets, and training settings. Training and checkpoint inference share these configurations.
- 🤖 **Model serving** — three camera views and a 14-D state go in; **50 × 14 absolute joint/gripper commands** come out.
- ⚡ **PACE execution** — [RoboTwin-PACE](https://github.com/niejunnan25/RoboTwin-PACE#install-and-try-the-core) selects execution horizons from the predicted actions and runs simulation evaluation.

<p align="center">
  <a href="#install">Setup</a> ·
  <a href="#download-a-checkpoint">Checkpoints</a> ·
  <a href="#serve-a-policy">Model serving</a> ·
  <a href="#training">Training</a> ·
  <a href="#citation">Citation</a>
</p>

<a id="install"></a>
## 🛠️ Add the RoboTwin task configurations

Use **Linux, an NVIDIA GPU, and Python 3.11 or newer**.

**If OpenPI is already working, keep your existing workspace, environment, and cache.** The patch adds and registers the task configurations, adapts communication for simulation, and fixes the output path for normalization statistics during training.

<a id="base-and-layout"></a>

The patch was generated and validated against official OpenPI [2b1f377](https://github.com/Physical-Intelligence/openpi/tree/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea). Other versions require a compatibility check. If you already use this repository, skip the patch and proceed to the checkpoints.

<a id="reuse-openpi"></a>

From your existing OpenPI workspace, activate its environment; this example uses `.venv`:

```bash
cd /path/to/openpi
source .venv/bin/activate
curl -fL https://raw.githubusercontent.com/niejunnan25/Openpi-PACE/main/pace/patches/openpi-robotwin.patch \
  -o /tmp/openpi-robotwin.patch
git apply --check /tmp/openpi-robotwin.patch && git apply /tmp/openpi-robotwin.patch
uv pip install --no-deps -e .
```

The final command reinstalls the local package without changing runtime dependencies. The [patch](pace/patches/openpi-robotwin.patch) preserves existing configurations. If the check fails, review your OpenPI version and local changes before proceeding.

<a id="new-openpi"></a>
<details>
<summary><strong>🌱 New to OpenPI? Install it here</strong></summary>

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone this repository. We recommend placing it next to your RoboTwin workspace:

```bash
git clone https://github.com/niejunnan25/Openpi-PACE.git OpenPI-PACE
cd OpenPI-PACE
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

Task configurations and supporting changes are already included, so the patch above is not needed. The model server uses this `.venv`; simulation uses the RoboTwin Conda environment.

</details>

Run subsequent commands from your OpenPI workspace root with the model environment activated. Replace `/path/to/openpi` with its actual location.

<a id="download-a-checkpoint"></a>
## 📦 Prepare checkpoints

Checkpoints are hosted at [niejunnan25/pi05-robotwin](https://huggingface.co/niejunnan25/pi05-robotwin). Each task is approximately **12.4 GB**; task names and files are listed in [manifest.json](https://huggingface.co/niejunnan25/pi05-robotwin/blob/main/manifest.json). Reuse existing copies of the same checkpoints. A single-task evaluation needs only its checkpoint; a full evaluation needs all 50.

For `adjust_bottle`, download the task checkpoint and shared tokenizer:

```bash
huggingface-cli download niejunnan25/pi05-robotwin \
  --include "adjust_bottle/*" tokenizer.model manifest.json TERMS.md NOTICE \
  --local-dir local/pi05-robotwin
```

Replace `adjust_bottle/*` to download a different task. The task directory should contain:

```text
local/pi05-robotwin/adjust_bottle/
  params/
  assets/robotwin/norm_stats.json
```

<a id="local-assets"></a>
<a id="use-existing-local-assets"></a>

For the first run, put the tokenizer in OpenPI's default cache. Skip this if it is already there:

```bash
mkdir -p ~/.cache/openpi/big_vision
cp local/pi05-robotwin/tokenizer.model ~/.cache/openpi/big_vision/paligemma_tokenizer.model
```

For a custom cache, keep using `OPENPI_DATA_HOME`; the tokenizer belongs at `big_vision/paligemma_tokenizer.model` within that directory.

Each task uses its **EMA checkpoint at training step 30,000** and matching normalization statistics. Keep `params/` and `assets/` intact. Evaluation does not require training data, parameter conversion, or recomputing normalization statistics.

<a id="serve-a-policy"></a>
## 🚀 Start the model server

Use OpenPI's standard serving entry point with the task configuration and checkpoint path:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_robotwin_adjust_bottle \
  --policy.dir local/pi05-robotwin/adjust_bottle
```

This uses GPU 0 and port 8000. To switch tasks, update both `--policy.config` and `--policy.dir`. The first inference requires XLA compilation.

Keep the server running. In another terminal, install the communication client from your OpenPI workspace **inside the RoboTwin Conda environment**:

```bash
conda activate RoboTwin
python -m pip install -e /path/to/openpi/packages/openpi-client
```

Then follow the [RoboTwin-PACE single-task example](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-one-task), using the same task and port. Each evaluation runs 100 episodes. Stop the server with Ctrl-C after evaluation and start a fresh server for the next run. **The client selects PACE or Fixed**; the server always returns the full action chunk.

For all 50 tasks, use the [RoboTwin-PACE batch script](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-multiple-tasks), which manages model servers automatically. When reusing an existing OpenPI workspace, set `OPENPI_ROOT`, `SERVER_PYTHON`, and `CHECKPOINT_ROOT` at the top of [`pace/scripts/eval_all.sh`](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/pace/scripts/eval_all.sh) to match your workspace, model environment, and checkpoint directory.

<a id="training"></a>
## 🏋️ Train your own model

Prepare RoboTwin demonstrations in LeRobot format under `robotwin/<task>` in your dataset root. For example, place `adjust_bottle` at `/path/to/datasets/robotwin/adjust_bottle`. Run these commands in the OpenPI model environment:

```bash
export HF_LEROBOT_HOME="/path/to/datasets"
python scripts/compute_norm_stats.py --config-name pi05_robotwin_adjust_bottle
python scripts/train.py pi05_robotwin_adjust_bottle --exp-name paper
```

The released setup uses 50 `demo_clean` demonstrations per task, batch size 32, and training seed 42 for full-parameter π₀.₅ fine-tuning. Training assets and checkpoints are written to `pace/runs/`; evaluation uses the step-30,000 checkpoint.

Recompute normalization statistics when training a new model. For released checkpoints, use the statistics shipped with the weights.

<a id="citation"></a>
## 📝 Citation

If you use PACE in your research, please cite our paper, accepted at **NeurIPS 2026**:

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
## 🤝 Acknowledgments and license

We thank the authors of Physical Intelligence's [OpenPI](https://github.com/Physical-Intelligence/openpi) and [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). This repository uses the [Apache-2.0 License](LICENSE); bundled components retain their respective licenses. Model and tokenizer terms are provided in the model repository's `TERMS.md` and `NOTICE`.
