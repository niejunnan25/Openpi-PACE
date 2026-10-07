"""Run one evaluation batch with a fresh model server and the RoboTwin-PACE client."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from _bootstrap import ROOT
from openpi_client import msgpack_numpy
from openpi_pace.configs import TASKS
from websockets.sync.client import connect


def stop(process):
    if process is None:
        return
    # Workers can outlive their parent; signal the group even after parent exit.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=15)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def wait_ready(server, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError("policy server exited; see server.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("policy server did not become ready; see server.log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--gpu", required=True, help="Physical CUDA device index on this host")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--method", choices=("fixed", "pace"), required=True)
    parser.add_argument("--horizon", type=int, choices=(5, 25, 50), default=50)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--client-timeout", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-actions", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.episodes < 1 or (args.method == "pace" and args.horizon != 50):
        parser.error("episodes must be positive; paper PACE uses maximum horizon 50")
    checkpoint = (args.checkpoint_root / args.task).resolve()
    robotwin = args.robotwin_root.resolve()
    output = args.output.resolve()
    client_python = str(Path(args.client_python).absolute())
    server_python = str(Path(args.server_python).absolute())
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    output.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"
    env["PYTHONPATH"] = os.pathsep.join(
        map(str, [ROOT / "src", ROOT / "pace/src", ROOT / "packages/openpi-client/src"])
    )
    # SIGTERM from a scheduler follows the same cleanup path as Ctrl-C.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    server = client = None
    try:
        with (output / "server.log").open("w") as server_log, (output / "client.log").open("w") as client_log:
            server = subprocess.Popen(
                [
                    server_python,
                    str(ROOT / "pace/scripts/serve.py"),
                    "--task",
                    args.task,
                    "--checkpoint-dir",
                    str(checkpoint),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.port),
                ],
                cwd=ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            wait_ready(server, args.port, args.startup_timeout)
            # A metadata-only connection preserves the fresh server's inference RNG.
            with connect(f"ws://127.0.0.1:{args.port}", compression=None, max_size=None) as connection:
                metadata = msgpack_numpy.unpackb(connection.recv(timeout=10))
            expected = {
                "task": args.task,
                "config": f"pi05_robotwin_{args.task}",
                "action_horizon": 50,
                "action_dim": 14,
                "action_space": "absolute_joint",
                "checkpoint_dir": str(checkpoint),
                "server_seed": 0,
                "num_denoising_steps": 10,
                "norm_stats_sha256": hashlib.sha256(
                    (checkpoint / "assets/robotwin/norm_stats.json").read_bytes()
                ).hexdigest(),
            }
            if any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("server identity/settings differ from the requested paper policy")
            (output / "server_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            command = [
                client_python,
                str(robotwin / "pace/scripts/evaluate.py"),
                "--task",
                args.task,
                "--method",
                args.method,
                "--horizon",
                str(args.horizon),
                "--seed",
                str(args.seed),
                "--batch",
                str(args.batch),
                "--episodes",
                str(args.episodes),
                "--host",
                "localhost",
                "--port",
                str(args.port),
                "--checkpoint-id",
                str(checkpoint),
                "--output",
                str(output / "evaluation"),
            ]
            if args.save_actions:
                command.append("--save-actions")
            client = subprocess.Popen(
                command, cwd=robotwin, env=env, stdout=client_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            code = client.wait(timeout=args.client_timeout)
            if code != 0:
                raise RuntimeError(f"evaluation exited with code {code}; see client.log")
            result = json.loads((output / "evaluation/metadata.json").read_text())
            if result["status"] != "complete" or result["episodes"] != args.episodes:
                raise RuntimeError("evaluation did not complete the requested batch")
            print(
                json.dumps(
                    {
                        "task": args.task,
                        "method": args.method,
                        "episodes": result["episodes"],
                        "successes": result["successes"],
                    }
                )
            )
    finally:
        try:
            stop(client)
        finally:
            stop(server)


if __name__ == "__main__":
    main()
