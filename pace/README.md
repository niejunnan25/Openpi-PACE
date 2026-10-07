# OpenPI-PACE

The π₀.₅ training configurations and policy server used with
[RoboTwin-PACE](../../RoboTwin-PACE/pace/README.md). PACE's horizon selector and
calibration live only in RoboTwin-PACE; this repository returns complete action chunks.

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
  README.md
```

Outside `pace/`, changes register the new configurations/package, save computed
normalization statistics under the configured asset ID, disable server and client websocket
heartbeats during blocking simulation/model compilation, and add README/ignore entries. Model inference,
transforms and checkpoint loading remain upstream code.

## Install

Use Linux with a CUDA GPU for the JAX policy server. From this repository root,
follow the upstream [installation instructions](../README.md#installation):

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
```

The package build includes `openpi_pace`; install from this checkout so the
official server can import its registered task configurations. Keep the RoboTwin
simulator in its own environment;
install the client there with `python -m pip install -e /path/to/OpenPI-PACE/packages/openpi-client`.
The upstream lockfile pins dependencies. Reusing a working environment is possible;
review its installed versions separately from the source-code comparison.

## Checkpoints and task names

Each task uses its own step-30,000 EMA weights and its own normalization statistics:

```text
pi05-robotwin/
  adjust_bottle/
    params/
    assets/robotwin/norm_stats.json
  beat_block_hammer/
    params/
    assets/robotwin/norm_stats.json
  ...
```

Configuration: `pi05_robotwin_<task>`. Local LeRobot dataset ID: `robotwin/<task>`.
The normalization asset ID is `robotwin` for every configuration; **the file's
values are task-specific**. `beat_block_hammer` uses the historical `_50` weights
and 50-demonstration dataset, not the earlier seven-demonstration experiment.
No parameter conversion or normalization recomputation is needed for old models.

For existing local assets, create an ignored JSON manifest with a list of records:

```json
[
  {
    "task": "adjust_bottle",
    "checkpoint_dir": "/path/to/original/30000",
    "norm_stats": "/path/to/original/30000/assets/original-id/norm_stats.json",
    "dataset_dir": "/path/to/original/lerobot-task"
  }
]
```

```bash
python pace/scripts/prepare_local.py --manifest local/sources.json \
  --checkpoint-root local/pi05-robotwin --dataset-root local/datasets
```

Destinations must be new. This makes parameter/dataset symlinks and copies the
small normalization files byte-for-byte, leaving originals unchanged. These local
views are not a self-contained downloadable release; publish real files when
packaging checkpoints. Dataset links are optional for inference. The names above
are a local layout, not a claim that a Hugging Face repository is already published.

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

The data adapter uses three cameras (`cam_high`, `cam_left_wrist`,
`cam_right_wrist`) and 14-dimensional joint/gripper states. `adapt_to_pi=False`:
no Aloha joint flips or gripper conversion. The 12 arm dimensions use deltas inside
the model pipeline; the two grippers remain absolute. Returned actions are
**50 × 14 absolute joint/gripper commands**, restored to the environment's space.

## Serve a policy

Activate the OpenPI environment and run the official entry point from this
repository root:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python scripts/serve_policy.py \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi05_robotwin_adjust_bottle \
  --policy.dir /path/to/pi05-robotwin/adjust_bottle
```

`--policy.config` selects the task configuration; `--policy.dir` selects the
checkpoint, including its saved normalization. `CUDA_VISIBLE_DEVICES` selects the
model GPU. The memory fraction matches the original scheduler and allows multiple
services on a GPU with sufficient memory. The server returns complete action
chunks; Fixed H and PACE are selected by the RoboTwin client.

In a second terminal, activate the RoboTwin environment and run from
**RoboTwin-PACE**:

```bash
python pace/scripts/evaluate.py \
  --task adjust_bottle --method pace --gpu 0 --port 8000
```

For a fixed baseline use `--method fixed --horizon 25`. Each client evaluates
100 episodes. Stop the server with Ctrl-C afterwards, and start a fresh server
for each independent evaluation; this resets the model RNG to its original state.
The policy defaults remain RNG seed 0 and ten denoising steps.

See the companion [evaluation guide](../../RoboTwin-PACE/pace/README.md#evaluate-one-task)
for results and the [Bash scheduler](../../RoboTwin-PACE/pace/README.md#evaluate-multiple-tasks)
for eight GPUs with three independent server/client pairs per GPU. Scheduling
and process cleanup live in RoboTwin-PACE; no separate PACE server is needed.

## Tests

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python -m pytest pace/tests -m "not manual"
```

Upstream OpenPI code is Apache-2.0 licensed; see [LICENSE](../LICENSE).
Checkpoint and dataset distribution are separate from this source-code license.
