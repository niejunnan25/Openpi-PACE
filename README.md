<h1 align="center">PACE</h1>
<h3 align="center">Phase-Aware Chunk Execution for Robot Policies<br>with Action Chunking</h3>

<p align="center"><strong>OpenPI-PACE</strong><br>Task-specific π₀.₅ policies &middot; Model serving &middot; Training</p>

<p align="center">
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=Paper&amp;message=PDF&amp;color=red&amp;logo=arxiv&amp;logoColor=white" alt="Paper PDF" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Website" height="24"></a>
  <a href="https://huggingface.co/niejunnan25/pi05-robotwin"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Model&amp;message=HuggingFace&amp;color=yellow" alt="Model Hugging Face" height="24"></a>
</p>

<p align="center">
  <a href="pace/media/overview.pdf"><img src="pace/media/overview.png" alt="Overview of PACE: fixed-horizon sensitivity, adaptive execution of predicted action chunks, simulation and real-robot results, and changing horizons within one rollout." width="100%"></a>
</p>
<p align="center"><sub><strong>Overview of PACE.</strong> Adapt the executed prefix to the predicted motion, then query the policy again. Click the figure for the vector PDF.</sub></p>

This repository provides the task-specific π₀.₅ configurations and official
model server for PACE on RoboTwin. The server returns complete action chunks;
**RoboTwin-PACE** contains the PACE selector, simulator client and GPU scheduler.
Start with the [RoboTwin-PACE evaluation guide](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md)
for the complete workflow.

To run a released checkpoint, follow [installation](#install),
[download](#download-a-checkpoint) and [serve a policy](#serve-a-policy).
[Training](#training) and [existing local assets](#use-existing-local-assets)
are separate workflows.

## Install

Use Linux with an NVIDIA GPU for the JAX model server and a separate environment
from the RoboTwin simulator. This checkout requires Python 3.11 or newer.
Clone the repository, initialize the upstream submodules and install with
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/niejunnan25/Openpi-PACE.git OpenPI-PACE
cd OpenPI-PACE
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
source .venv/bin/activate
```

If you already cloned it using the RoboTwin-PACE guide, start with
`cd /path/to/OpenPI-PACE` and skip the clone command above. The explicit local
directory name matches the companion guide and scheduler defaults.

Install from this checkout so the official server can import `openpi_pace` and
its 50 task configurations. The upstream lockfile pins the dependencies.
See the [upstream installation notes](https://github.com/Physical-Intelligence/openpi/blob/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea/README.md#installation) for additional
platform details.

In your separate **RoboTwin environment**, install the client package:

```bash
python -m pip install -e /path/to/OpenPI-PACE/packages/openpi-client
```

## Download a checkpoint

The checkpoints are hosted at
[niejunnan25/pi05-robotwin](https://huggingface.co/niejunnan25/pi05-robotwin).
They are being published incrementally; consult the model card and
[manifest.json](https://huggingface.co/niejunnan25/pi05-robotwin/blob/main/manifest.json)
for available task names. Each task is approximately 12.4 GB. Training
demonstrations are not required for inference.

From the OpenPI-PACE root, with its environment active, download the
`adjust_bottle` checkpoint and shared tokenizer:

```bash
huggingface-cli download niejunnan25/pi05-robotwin \
  --include "adjust_bottle/*" tokenizer.model manifest.json TERMS.md NOTICE \
  --local-dir local/pi05-robotwin
```

`huggingface-cli` is provided by the Hugging Face Hub version in the lockfile.
Replace `adjust_bottle/*` with another available task's pattern as needed;
multiple patterns select multiple tasks. The full RoboTwin queue needs all 50
checkpoints locally. The downloaded task directory contains:

```text
local/pi05-robotwin/
  tokenizer.model
  manifest.json
  adjust_bottle/
    params/
    assets/robotwin/norm_stats.json
```

Prepare the tokenizer at the cache location expected by the official server:

```bash
mkdir -p local/cache/big_vision
cp local/pi05-robotwin/tokenizer.model local/cache/big_vision/paligemma_tokenizer.model
```

Keep each task's `params/` and saved normalization together. No parameter
conversion or normalization recomputation is needed. Model and tokenizer terms
are provided in the downloaded `TERMS.md` and `NOTICE` files.

## Serve a policy

From the OpenPI-PACE root, with its environment active:

```bash
OPENPI_DATA_HOME="$PWD/local/cache" \
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python scripts/serve_policy.py --port 8000 policy:checkpoint \
  --policy.config pi05_robotwin_adjust_bottle \
  --policy.dir local/pi05-robotwin/adjust_bottle
```

`--policy.config` selects the task configuration; `--policy.dir` selects its
checkpoint and normalization. `CUDA_VISIBLE_DEVICES` selects the model GPU.
`OPENPI_DATA_HOME` points to the tokenizer cache prepared above. The memory
fraction matches the original scheduler; available memory determines how many
server/client pairs fit on a GPU. Initial inference may compile XLA kernels
before producing actions.

In a second terminal, activate the **RoboTwin environment** and run from
**RoboTwin-PACE**:

```bash
cd /path/to/RoboTwin-PACE
python pace/scripts/evaluate.py \
  --task adjust_bottle --method pace --gpu 0 --seed 0 --port 8000
```

The task and port must match the server. This evaluates **100 episodes**.
For Fixed H=25, use `--method fixed --horizon 25` in a separate evaluation.
Stop the server with Ctrl-C afterwards, and start a fresh server for each
independent evaluation. The policy defaults remain RNG seed 0 and ten denoising
steps; the client's evaluation seed is a separate setting.

For all 50 tasks, use the [RoboTwin-PACE scheduler](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#evaluate-multiple-tasks),
which starts and cleans up each server/client pair. Put
`export OPENPI_DATA_HOME="$OPENPI_ROOT/local/cache"` in its `pace/configs/eval.env`
along with your environment paths. See [result files](https://github.com/niejunnan25/RoboTwin-PACE/blob/main/README.md#read-the-results)
for episode outcomes and logs.

## Checkpoint configuration details

Each task uses its own step-30,000 EMA weights and normalization statistics.
The configuration is `pi05_robotwin_<task>`; the local LeRobot dataset ID is
`robotwin/<task>`. Every configuration uses the normalization asset ID
`robotwin`, but the file's values are task-specific. `beat_block_hammer` uses
the historical `_50` weights and 50-demonstration dataset, not the earlier
seven-demonstration experiment.

The data adapter uses three cameras (`cam_high`, `cam_left_wrist`,
`cam_right_wrist`) and 14-dimensional joint/gripper states. `adapt_to_pi=False`:
no Aloha joint flips or gripper conversion. The 12 arm dimensions use deltas
inside the model pipeline; the two grippers remain absolute. Returned actions
are **50 × 14 absolute joint/gripper commands**, restored to environment space.

## Use existing local assets

If you already have the original checkpoints, the optional helper can arrange
them in the same task layout. Create a Git-ignored JSON manifest:

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

The destination must be new. This creates parameter symlinks and copies the
small normalization files byte-for-byte, leaving originals unchanged. Supply
the tokenizer cache separately as above, or use an existing OpenPI cache.
For training, each record may also include `dataset_dir`; add
`--dataset-root local/datasets` to create the corresponding dataset links.
This local helper is unnecessary for the downloaded checkpoint layout.

## Training

Each paper task uses 50 `demo_clean` demonstrations at 50 Hz. Full-parameter π₀.₅
fine-tuning uses AdamW, batch size 32, training seed 42 and EMA decay 0.99. Learning
rate: 1,000-step warmup to 2.5e-5, cosine decay to 2.5e-6 over 30,000 schedule steps.
The run length is 30,005 steps and the evaluated checkpoint is step 30,000.

```bash
export HF_LEROBOT_HOME="/absolute/path/to/OpenPI-PACE/local/datasets"
python scripts/compute_norm_stats.py --config-name pi05_robotwin_adjust_bottle
python scripts/train.py pi05_robotwin_adjust_bottle --exp-name paper
```

Training assets/checkpoints go under `pace/runs/`. Each configuration has its own
assets directory, even though the inner asset ID is shared. Recomputing statistics
is for training a new model; use the checkpoint's saved statistics for evaluation.

## Base and layout

Based on official OpenPI commit `2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea`
(2025-12-05). The model, Aloha adapter, checkpoint loader and policy implementation
match the audited experiment implementation. All built-in
OpenPI configurations remain available. Fifty paper configurations are added;
unrelated private experiments and their configurations are not included.

```text
pace/
  src/openpi_pace/configs.py   # shared settings and 50 task names
  scripts/prepare_local.py    # optional local views of existing assets
  tests/                     # configurations, transforms and communication
  README.md                  # package directory index
```

Outside `pace/`, changes register the new configurations/package, save computed
normalization statistics under the configured asset ID, disable server and client websocket
heartbeats during blocking simulation/model compilation, and add README/ignore entries. Model inference,
transforms and checkpoint loading remain upstream code.

## Tests

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python -m pytest pace/tests -m "not manual"
```

## Acknowledgments and license

Built on [OpenPI](https://github.com/Physical-Intelligence/openpi) by
Physical Intelligence. We thank its authors for the policy models, training
code and serving implementation, and the
[RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin) authors for the simulator.
For general OpenPI examples beyond PACE, see the
[upstream guide](https://github.com/Physical-Intelligence/openpi/blob/2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea/README.md).

OpenPI code is Apache-2.0 licensed; see [LICENSE](LICENSE). Bundled components
retain their respective licenses. Checkpoint and dataset distribution are
separate from this source-code license.
