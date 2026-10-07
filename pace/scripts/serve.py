"""Serve one task's pi05 checkpoint using the paper inference settings."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from openpi_pace.configs import TASKS

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.config import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    checkpoint = args.checkpoint_dir.resolve()
    norm_file = checkpoint / "assets/robotwin/norm_stats.json"
    norm_hash = hashlib.sha256(norm_file.read_bytes()).hexdigest()
    config = get_config(f"pi05_robotwin_{args.task}")
    policy = policy_config.create_trained_policy(config, checkpoint)
    metadata = {
        **policy.metadata,
        "checkpoint_dir": str(checkpoint),
        "norm_stats_sha256": norm_hash,
        "server_seed": 0,
        "num_denoising_steps": 10,
    }
    logging.info("Loaded policy: %s", json.dumps(metadata))
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
