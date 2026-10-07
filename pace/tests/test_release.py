import ast
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from openpi_pace.configs import TASKS
from prepare_local import prepare
import pytest
from run_robotwin import stop
from run_robotwin import wait_ready

from openpi import transforms
from openpi.training import config

ROOT = Path(__file__).resolve().parents[2]


def test_official_configs_preserved():
    source = subprocess.check_output(
        ["git", "show", "2b1f3779ec2e3ddb56fd1425944e5b3d084c28ea:src/openpi/training/config.py"],
        cwd=ROOT,
        text=True,
    )
    tree = ast.parse(source)
    original = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_CONFIGS" for t in n.targets)
    )
    official = {
        k.value.value for call in original.elts if isinstance(call, ast.Call) for k in call.keywords if k.arg == "name"
    }
    official.update(c.name for c in config.roboarena_config.get_roboarena_configs())
    added = {f"pi05_robotwin_{task}" for task in TASKS}
    assert len(added) == 50
    assert set(config._CONFIGS_DICT) == official | added  # noqa: SLF001


@pytest.mark.parametrize("task", TASKS)
def test_paper_configs(task):
    cfg = config.get_config(f"pi05_robotwin_{task}")
    assert cfg.data.repo_id == f"robotwin/{task}"
    assert cfg.data.assets.asset_id == "robotwin"
    assert not cfg.data.adapt_to_pi
    assert cfg.data.use_delta_joint_actions
    assert cfg.model.pi05
    assert cfg.model.action_horizon == 50
    assert cfg.model.action_dim == 32
    assert (cfg.seed, cfg.batch_size, cfg.ema_decay, cfg.num_train_steps) == (42, 32, 0.99, 30005)
    assert (cfg.save_interval, cfg.keep_period, cfg.fsdp_devices) == (5000, 5000, 1)
    assert (
        cfg.lr_schedule.warmup_steps,
        cfg.lr_schedule.peak_lr,
        cfg.lr_schedule.decay_steps,
        cfg.lr_schedule.decay_lr,
    ) == (1000, 2.5e-5, 30000, 2.5e-6)
    assert cfg.optimizer.__class__.__name__ == "AdamW"
    assert not cfg.freeze_filter(("model", "kernel"), None)
    assert cfg.policy_metadata["action_dim"] == 14
    assert task in str(cfg.assets_dirs)


def test_joint_and_camera_contract(tmp_path, monkeypatch):
    # Tokenizer construction is independent of the data-space conversion tested here.
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    cfg = config.get_config("pi05_robotwin_adjust_bottle")
    data = cfg.data.create(tmp_path, cfg.model)
    state = np.linspace(-0.5, 0.5, 14, dtype=np.float32)
    actions = np.arange(50 * 14, dtype=np.float32).reshape(50, 14) / 100
    cameras = {
        name: np.full((3, 4, 5), i, dtype=np.uint8)
        for i, name in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist"))
    }
    obs = {"state": state, "actions": actions.copy(), "images": cameras, "prompt": "adjust the bottle"}
    packed = transforms.compose(data.data_transforms.inputs)(copy.deepcopy(obs))
    np.testing.assert_array_equal(packed["state"], state)
    mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
    expected = actions.copy()
    expected[:, mask] -= state[mask]
    np.testing.assert_array_equal(packed["actions"], expected)
    for i, name in enumerate(("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")):
        np.testing.assert_array_equal(packed["image"][name], np.full((4, 5, 3), i, dtype=np.uint8))
        assert packed["image_mask"][name]
    decoded = transforms.compose(data.data_transforms.outputs)(packed)
    np.testing.assert_allclose(decoded["actions"], actions, atol=1e-6)
    assert decoded["actions"].shape == (50, 14)
    assert data.use_quantile_norm


def manifest_fixture(tmp_path):
    source = tmp_path / "source"
    (source / "params").mkdir(parents=True)
    (source / "norm.json").write_text('{"norm_stats": {}}\n')
    (source / "meta").mkdir()
    (source / "meta/info.json").write_text(json.dumps({"total_episodes": 50, "fps": 50}))
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "task": "beat_block_hammer",
                    "checkpoint_dir": str(source),
                    "norm_stats": str(source / "norm.json"),
                    "dataset_dir": str(source),
                }
            ]
        )
    )
    return source, manifest


def test_asset_view_preserves_norm_bytes(tmp_path):
    source, manifest = manifest_fixture(tmp_path)
    prepare(manifest, tmp_path / "checkpoints", tmp_path / "datasets")
    task = tmp_path / "checkpoints/beat_block_hammer"
    assert (task / "params").is_symlink()
    assert (task / "params").resolve() == source / "params"
    assert (task / "assets/robotwin/norm_stats.json").read_bytes() == (source / "norm.json").read_bytes()
    assert (tmp_path / "datasets/robotwin/beat_block_hammer").resolve() == source


def test_existing_destination_does_not_create_partial_view(tmp_path):
    _, manifest = manifest_fixture(tmp_path)
    (tmp_path / "datasets/robotwin").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        prepare(manifest, tmp_path / "checkpoints", tmp_path / "datasets")
    assert not (tmp_path / "checkpoints").exists()


def test_seven_demo_data_rejected(tmp_path):
    source, manifest = manifest_fixture(tmp_path)
    (source / "meta/info.json").write_text(json.dumps({"total_episodes": 7, "fps": 50}))
    with pytest.raises(ValueError, match="50 demonstrations"):
        prepare(manifest, tmp_path / "checkpoints", tmp_path / "datasets")
    assert not (tmp_path / "checkpoints").exists()


def test_server_exit_and_cleanup():
    process = subprocess.Popen(["python3", "-c", "pass"], start_new_session=True)
    process.wait(timeout=10)
    with pytest.raises(RuntimeError, match="server exited"):
        wait_ready(process, 1, 1)
    stop(process)
    process = subprocess.Popen(["python3", "-c", "import time; time.sleep(60)"], start_new_session=True)
    stop(process)
    assert process.poll() is not None
