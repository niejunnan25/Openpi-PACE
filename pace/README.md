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
  scripts/serve.py            # one task's model server
  scripts/run_robotwin.py     # fresh server + one evaluation batch
  scripts/run_batch.py        # job CSV + one worker per specified GPU
  tests/                     # configuration, transforms and launcher checks
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

The package build includes `openpi_pace`. Checkout entry points also locate the
source directories directly. Keep the RoboTwin simulator in its own environment;
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

## Serve and evaluate

To run only a server:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 \
python pace/scripts/serve.py --task adjust_bottle \
  --checkpoint-dir /path/to/pi05-robotwin/adjust_bottle --port 8000
```

To launch one complete job, with a fresh server and automatic process cleanup:

```bash
python pace/scripts/run_robotwin.py \
  --task adjust_bottle --checkpoint-root /path/to/pi05-robotwin \
  --robotwin-root /path/to/RoboTwin-PACE \
  --client-python /path/to/robotwin-env/bin/python \
  --server-python /path/to/openpi-env/bin/python \
  --gpu 0 --port 8000 --method fixed --horizon 50 --seed 0 --batch 1 \
  --episodes 1 --save-actions --output pace/runs/smoke-fixed
```

Repeat with `--method pace --horizon 50` and a new output directory. Supply the
machine's Vulkan/NVIDIA library settings in the environment when required by
RoboTwin. For the paper protocol use 100 episodes, seeds 0/1/2 and batches 1/2/3.
The batch index labels repeats of the same scene seed sequence.

The launcher starts a fresh server per task/seed/batch, with model RNG 0 and ten
denoising steps, just as the original schedulers. The evaluation seed controls
scenes; it does not replace the model seed. Readiness/metadata checks do not query
the model. Server metadata records task, config, checkpoint path and normalization
SHA-256. Logs, episode records and optional full-precision action chunks are saved
under the job directory. The launcher checks completion and terminates only its own
server/client process groups. `--startup-timeout` and `--client-timeout` bound waits.

As in the original RoboTwin scripts, the scene seed does not seed Python's random
instruction generator. Equal evaluation seeds do not guarantee identical prompts;
use recorded prompts for comparisons that require matching language inputs.

A few episodes verify connectivity and action execution, not the paper success rate.

## Batch evaluation

Generate the paper job list with the companion repository's planner:

```bash
python /path/to/RoboTwin-PACE/pace/scripts/plan.py --output pace/runs/jobs.csv
```

From OpenPI-PACE, run a selected task and comparison on the available GPUs:

```bash
python pace/scripts/run_batch.py \
  --jobs pace/runs/jobs.csv \
  --checkpoint-root /path/to/pi05-robotwin \
  --robotwin-root /path/to/RoboTwin-PACE \
  --client-python /path/to/robotwin-env/bin/python \
  --server-python /path/to/openpi-env/bin/python \
  --gpus 0 1 --output pace/runs/evaluation \
  --tasks adjust_bottle --methods fixed_h50 pace --seeds 0 --batches 1
```

Add `--dry-run` to preview the selection without loading models or creating
outputs. Omit the four selection flags to run all 1,800 jobs: 50 tasks, four
methods, three seeds and three batches, with 100 episodes per job. For a small
integration check use a separate output directory and `--episodes 1`.

The runner uses one job per specified GPU and consecutive ports starting at
`--port-base` (default 8000). Each job calls `run_robotwin.py`, which starts a
fresh policy service and cleans it up before the next job. Ctrl-C or SIGTERM
stops new dispatches and lets active launchers clean up their own processes.
On multiple hosts, use disjoint task selections or CSV subsets and separate
output directories.

Results are saved under `<output>/<task>/<method>/seedN/batchN/`, with the
existing server/client logs and `evaluation/` records. `batch_summary.json`
lists the current selection's completed, skipped and failed jobs; `launcher_logs/`
contains each launcher's output. Any failed job makes the batch command exit
nonzero after the other queued jobs finish.

Running the command again skips results whose settings and episode records
match the requested job. Existing incomplete jobs are reported as failed. Add
`--retry-failed` to rerun them; previous attempts are moved to the adjacent
`<output>_failed/` directory, preserving their logs outside the result tree used
for aggregation. Completed results with different settings require a separate
output directory. `--save-actions` enables action recording for newly run jobs.

After the full paper grid completes, summarize from the RoboTwin environment:

```bash
/path/to/robotwin-env/bin/python /path/to/RoboTwin-PACE/pace/scripts/summarize.py \
  --runs /absolute/path/to/OpenPI-PACE/pace/runs/evaluation \
  --output /absolute/path/to/OpenPI-PACE/pace/runs/paper_summary.json
```

The paper summarizer requires the complete grid; a selected-task or one-episode
check can be inspected through `batch_summary.json` and the per-job records.

## Tests

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" python -m pytest pace/tests -m "not manual"
```

Upstream OpenPI code is Apache-2.0 licensed; see [LICENSE](../LICENSE).
Checkpoint and dataset distribution are separate from this source-code license.
