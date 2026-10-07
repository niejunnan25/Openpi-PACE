"""Exercise the batch CLI with real child processes and recorded evaluation files."""

import csv
import json
import os
from pathlib import Path
import random
import runpy
import signal
import socket
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROBOTWIN = ROOT.parent / "RoboTwin-PACE"

# This stand-in has the single-job launcher's CLI/output contract, without a GPU.
FAKE_LAUNCHER = r"""
import argparse, hashlib, json, os, pathlib, signal, socket, subprocess, sys, time
p = argparse.ArgumentParser()
for name in ("task", "checkpoint-root", "robotwin-root", "client-python", "server-python",
             "gpu", "port", "method", "horizon", "seed", "batch", "episodes", "output", "startup-timeout"):
    p.add_argument("--" + name, required=True)
p.add_argument("--client-timeout")
p.add_argument("--save-actions", action="store_true")
a = p.parse_args()
out = pathlib.Path(a.output)
out.mkdir(parents=True, exist_ok=False)
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", int(a.port)))
listener.listen()
# Server-initiated close leaves TIME_WAIT, as an HTTP readiness response can.
client = socket.create_connection(("127.0.0.1", int(a.port)))
connection, _ = listener.accept()
connection.shutdown(socket.SHUT_WR)
connection.close()
assert client.recv(1) == b""
client.close()
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
key = (a.task, a.method, a.seed, a.batch)
def event(kind):
    fd = os.open(os.environ["BATCH_TEST_EVENTS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.write(fd, (json.dumps(dict(kind=kind, key=key, gpu=a.gpu, port=a.port, worker=worker.pid,
                                  client_python=a.client_python, server_python=a.server_python)) + "\n").encode())
    os.close(fd)
def cleanup(*_):
    sys.exit(143)
signal.signal(signal.SIGTERM, cleanup)
event("start")
try:
    if os.environ.get("BATCH_TEST_HOLD"):
        time.sleep(60)
    time.sleep(0.25)
    (out / "server.log").write_text("test model service\n")
    cfg = pathlib.Path(a.robotwin_root) / "pace/configs"
    paper = json.loads((cfg / "paper.json").read_text())
    source = (cfg / "calibration" / (a.task + ".json")).read_bytes()
    ckpt = (pathlib.Path(a.checkpoint_root) / a.task).resolve()
    method = "pace" if a.method == "pace" else "fixed_h" + a.horizon
    n = int(a.episodes)
    metadata = dict(task=a.task, method=method, seed=int(a.seed), batch=int(a.batch),
                    expected_episodes=n, episodes=n, successes=n // 2, status="complete",
                    max_horizon=int(a.horizon), checkpoint_id=str(ckpt), task_config="demo_clean",
                    instruction_type="unseen", prediction_horizon=50, upstream_commit=paper["robotwin_commit"],
                    threshold=None)
    if a.method == "pace":
        metadata.update(selector=paper["selector"], threshold=json.loads(source)["threshold"],
                        calibration_sha256=hashlib.sha256(source).hexdigest())
    evaluation = out / "evaluation"
    evaluation.mkdir()
    fail = os.environ.get("BATCH_TEST_FAIL_METHOD") == a.method
    if fail:
        metadata["status"] = "failed"
    (evaluation / "metadata.json").write_text(json.dumps(metadata))
    norm_hash = hashlib.sha256((ckpt / "assets/robotwin/norm_stats.json").read_bytes()).hexdigest()
    (out / "server_metadata.json").write_text(json.dumps(dict(norm_stats_sha256=norm_hash)))
    (evaluation / "episodes.jsonl").write_text("".join(json.dumps(dict(episode=i, success=i < n // 2)) + "\n" for i in range(n)))
finally:
    worker.terminate()
    worker.wait()
    listener.close()
    event("stop")
sys.exit(7 if fail else 0)
"""


@pytest.fixture
def batch_case(tmp_path):
    # The release's real task mapping and calibration are the fixture's protocol.
    if not ROBOTWIN.is_dir():
        pytest.skip("companion RoboTwin-PACE checkout required")
    root = tmp_path / "paths with spaces"
    root.mkdir()
    checkpoint = root / "checkpoints/adjust_bottle"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets/robotwin").mkdir(parents=True)
    (checkpoint / "assets/robotwin/norm_stats.json").write_text("{}")
    fake = root / "fake_launcher.py"
    fake.write_text(FAKE_LAUNCHER)
    jobs = root / "jobs.csv"
    with jobs.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "method",
                "horizon",
                "seed",
                "batch",
                "episodes",
                "checkpoint_relative_path",
                "dataset_id",
            ],
        )
        writer.writeheader()
        for seed in (0, 1, 2):
            for method in ("fixed", "pace"):
                writer.writerow(
                    {
                        "task": "adjust_bottle",
                        "method": method,
                        "horizon": 50,
                        "seed": seed,
                        "batch": 1,
                        "episodes": 100,
                        "checkpoint_relative_path": "adjust_bottle",
                        "dataset_id": "robotwin/adjust_bottle",
                    }
                )
    # Reserve consecutive ports just long enough to find a usable test range.
    for _ in range(50):
        with socket.socket() as first, socket.socket() as second:
            port = random.SystemRandom().randrange(20000, 30000)
            try:
                first.bind(("127.0.0.1", port))
                second.bind(("127.0.0.1", port + 1))
            except (OSError, OverflowError):
                continue
            break
    else:
        pytest.fail("could not allocate consecutive test ports")
    python_link = root / "python-link"
    python_link.symlink_to(sys.executable)
    harness = (
        f"import sys; sys.path.insert(0, {str(ROOT / 'pace/scripts')!r}); "
        f"import run_batch; from pathlib import Path; run_batch.LAUNCHER=Path({str(fake)!r}); "
        "sys.exit(run_batch.main())"
    )
    command = [
        sys.executable,
        "-c",
        harness,
        "--jobs",
        str(jobs),
        "--robotwin-root",
        str(ROBOTWIN),
        "--checkpoint-root",
        str(root / "checkpoints"),
        "--client-python",
        str(python_link),
        "--server-python",
        str(python_link),
        "--gpus",
        "0",
        "1",
        "--port-base",
        str(port),
        "--output",
        str(root / "results"),
    ]
    env = dict(os.environ, BATCH_TEST_EVENTS=str(root / "events.jsonl"))
    return root, command, env


def invoke(case, *extra, env=None):
    _, command, default_env = case
    return subprocess.run(
        command + list(extra), env=env or default_env, text=True, capture_output=True, timeout=20, check=False
    )


def summary(root):
    return json.loads((root / "results/batch_summary.json").read_text())


def events(root):
    path = root / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def aggregate(root):
    summarize = runpy.run_path(str(ROBOTWIN / "pace/src/pace/results.py"))["summarize_jobs"]
    return summarize(root / "results", ["adjust_bottle"], ["fixed_h50", "pace"], batches=(1,))


def test_batch_parallel_resume_and_paper_aggregation(batch_case):
    root, _, _ = batch_case
    result = invoke(batch_case)
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary(root)["counts"] == {"complete": 6}
    active, maximum = set(), 0
    for event in events(root):
        gpu = event["gpu"]
        if event["kind"] == "start":
            assert gpu not in active, "two jobs overlapped on one GPU"
            active.add(gpu)
            maximum = max(maximum, len(active))
            assert event["client_python"] == str(root / "python-link")
            assert event["server_python"] == str(root / "python-link")
        else:
            active.remove(gpu)
    assert maximum == 2
    assert not active
    old_events = events(root)
    assert invoke(batch_case).returncode == 0
    assert summary(root)["counts"] == {"skipped": 6}
    assert events(root) == old_events
    report = aggregate(root)
    assert report["jobs"] == 6
    assert [row["mean_percent"] for row in report["overall"]] == [50.0, 50.0]


def test_dry_run_filters_without_writes_or_models(batch_case):
    root, _, _ = batch_case
    result = invoke(
        batch_case,
        "--dry-run",
        "--tasks",
        "adjust_bottle",
        "--methods",
        "pace",
        "--seeds",
        "1",
        "--batches",
        "1",
        "--episodes",
        "2",
        "--checkpoint-root",
        "/missing/checkpoints",
    )
    assert result.returncode == 0, result.stderr
    assert "Selected 1 jobs, 2 episodes" in result.stdout
    assert "adjust_bottle/pace/seed1/batch1 episodes=2" in result.stdout
    assert not (root / "results").exists()
    assert not events(root)


@pytest.mark.parametrize(
    "extra", [("--gpus", "0", "0"), ("--tasks", "typo"), ("--batches", "2"), ("--port-base", "65535")]
)
def test_invalid_selection_starts_nothing(batch_case, extra):
    root, _, _ = batch_case
    assert invoke(batch_case, *extra).returncode != 0
    assert not events(root)


def test_duplicate_csv_rejected(batch_case):
    root, _, _ = batch_case
    lines = (root / "jobs.csv").read_text().splitlines()
    (root / "jobs.csv").write_text("\n".join([*lines, lines[1]]) + "\n")
    assert "duplicate job" in invoke(batch_case).stderr
    assert not events(root)


def test_busy_port_starts_nothing(batch_case):
    root, _, _ = batch_case
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        result = invoke(batch_case, "--gpus", "0", "--port-base", str(listener.getsockname()[1]))
    assert result.returncode == 1
    assert not events(root)


def test_failures_preserved_and_retry_does_not_pollute_aggregation(batch_case):
    root, _, env = batch_case
    result = invoke(batch_case, env=dict(env, BATCH_TEST_FAIL_METHOD="fixed"))
    assert result.returncode == 1
    assert summary(root)["counts"] == {"failed": 3, "complete": 3}
    prior_events = events(root)
    assert invoke(batch_case).returncode == 1
    assert events(root) == prior_events
    result = invoke(batch_case, "--retry-failed")
    assert result.returncode == 0, result.stdout + result.stderr
    assert summary(root)["counts"] == {"complete": 3, "skipped": 3}
    assert len(list((root / "results_failed").rglob("server.log"))) == 3
    assert len(list((root / "results").rglob("metadata.json"))) == 6
    assert aggregate(root)["jobs"] == 6


@pytest.mark.parametrize("change", ["checkpoint_id", "episodes", "normalization", "truncated_records"])
def test_resume_checks_actual_records(batch_case, change):
    root, _, _ = batch_case
    selection = ("--methods", "pace", "--seeds", "0")
    assert invoke(batch_case, *selection).returncode == 0
    prior_events = events(root)
    directory = root / "results/adjust_bottle/pace/seed0/batch1/evaluation"
    if change in ("checkpoint_id", "episodes"):
        path = directory / "metadata.json"
        record = json.loads(path.read_text())
        record[change] = "wrong model" if change == "checkpoint_id" else 99
        path.write_text(json.dumps(record))
    elif change == "normalization":
        (root / "checkpoints/adjust_bottle/assets/robotwin/norm_stats.json").write_text('{"different":true}')
    else:
        (directory / "episodes.jsonl").write_text("")
    assert invoke(batch_case, *selection).returncode == 1
    assert events(root) == prior_events


def test_interrupt_cleans_children_and_output_lock_prevents_duplicate_run(batch_case):
    root, command, env = batch_case
    process = subprocess.Popen(
        command, env=dict(env, BATCH_TEST_HOLD="1"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        deadline = time.monotonic() + 10
        while len(events(root)) < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(events(root)) == 2
        assert invoke(batch_case).returncode == 1
        assert len(events(root)) == 2
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 130, stdout + stderr
        assert summary(root)["counts"] == {"interrupted": 2, "pending": 4}
        assert summary(root)["interrupted"]
        assert len(events(root)) == 4
        for event in events(root):
            with pytest.raises(ProcessLookupError):
                os.kill(event["worker"], 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)
