"""Create canonical local checkpoint/data views without modifying source assets."""

import argparse
import json
from pathlib import Path
import shutil

from _bootstrap import ROOT  # noqa: F401
from openpi_pace.configs import TASKS


def prepare(manifest, checkpoint_root, dataset_root=None):
    records = json.loads(manifest.read_text())
    tasks = [r["task"] for r in records]
    if len(tasks) != len(set(tasks)) or not set(tasks) <= set(TASKS):
        raise ValueError("manifest must contain unique canonical RoboTwin task names")
    for record in records:
        source = Path(record["checkpoint_dir"]).resolve()
        if not (source / "params").is_dir() or not Path(record["norm_stats"]).is_file():
            raise ValueError(f"checkpoint parameters or normalization missing: {record['task']}")
        if dataset_root is not None:
            info = json.loads((Path(record["dataset_dir"]) / "meta/info.json").read_text())
            if info["total_episodes"] != 50 or info["fps"] != 50:
                raise ValueError(f"expected the paper's 50 demonstrations at 50 Hz: {record['task']}")
    destinations = [checkpoint_root]
    if dataset_root is not None:
        destinations.append(dataset_root / "robotwin")
    if any(path.exists() or path.is_symlink() for path in destinations):
        raise FileExistsError("local view destinations must not already exist")
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    if dataset_root is not None:
        (dataset_root / "robotwin").mkdir(parents=True, exist_ok=False)
    for record in records:
        task = record["task"]
        target = checkpoint_root / task
        (target / "assets/robotwin").mkdir(parents=True)
        (target / "params").symlink_to(Path(record["checkpoint_dir"]).resolve() / "params", target_is_directory=True)
        shutil.copyfile(record["norm_stats"], target / "assets/robotwin/norm_stats.json")
        if dataset_root is not None:
            (dataset_root / "robotwin" / task).symlink_to(
                Path(record["dataset_dir"]).resolve(), target_is_directory=True
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    args = parser.parse_args()
    prepare(args.manifest, args.checkpoint_root, args.dataset_root)


if __name__ == "__main__":
    main()
