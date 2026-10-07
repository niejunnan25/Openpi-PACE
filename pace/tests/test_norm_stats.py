import dataclasses
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from openpi import transforms
from openpi.training import config

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("compute_norm_stats", ROOT / "scripts/compute_norm_stats.py")
compute = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compute)


@pytest.mark.parametrize("asset_id", ["robotwin", None])
def test_computed_statistics_are_loadable_by_training_config(tmp_path, monkeypatch, asset_id):
    base = config.get_config("pi05_robotwin_adjust_bottle")
    cfg = dataclasses.replace(
        base,
        assets_base_dir=str(tmp_path),
        data=dataclasses.replace(base.data, assets=config.AssetsConfig(asset_id=asset_id)),
    )
    monkeypatch.setattr(compute._config, "get_config", lambda _: cfg)  # noqa: SLF001
    monkeypatch.setattr(config.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    rng = np.random.default_rng(10)
    batch = {"state": rng.normal(size=(32, 14)), "actions": rng.normal(size=(32, 50, 14))}
    monkeypatch.setattr(compute, "create_torch_dataloader", lambda *_: ([batch], 1))
    compute.main(cfg.name)
    loaded = cfg.data.create(cfg.assets_dirs, cfg.model)
    assert loaded.norm_stats is not None
    assert loaded.asset_id == (asset_id or cfg.data.repo_id)
    np.testing.assert_allclose(loaded.norm_stats["state"].mean, batch["state"].mean(axis=0))
    np.testing.assert_allclose(loaded.norm_stats["actions"].mean, batch["actions"].mean(axis=(0, 1)))
