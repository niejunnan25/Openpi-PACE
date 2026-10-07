"""Run a RoboTwin job CSV, with one fresh model service per job and one job per GPU."""

import argparse
from collections import Counter
from collections import deque
import contextlib
import csv
from dataclasses import asdict
from dataclasses import dataclass
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "pace/scripts/run_robotwin.py"
METHODS = {("fixed", 5): "fixed_h5", ("fixed", 25): "fixed_h25", ("fixed", 50): "fixed_h50", ("pace", 50): "pace"}


@dataclass(frozen=True)
class Job:
    task: str
    method: str
    horizon: int
    seed: int
    batch: int
    episodes: int

    @property
    def name(self):
        return METHODS[self.method, self.horizon]

    @property
    def key(self):
        return f"{self.task}/{self.name}/seed{self.seed}/batch{self.batch}"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True, help="CSV from RoboTwin-PACE/pace/scripts/plan.py")
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--client-python", required=True)
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument(
        "--gpus", nargs="+", type=int, required=True, help="Available physical GPU indices on this host"
    )
    parser.add_argument("--port-base", type=int, default=8000, help="One consecutive port per GPU")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS.values()))
    parser.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2))
    parser.add_argument("--batches", nargs="+", type=int, choices=(1, 2, 3))
    parser.add_argument("--episodes", type=int, help="Override CSV episode counts for a small integration check")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--client-timeout", type=float)
    parser.add_argument("--save-actions", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="Archive incomplete jobs and rerun them")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print selected jobs without starting services or writing files"
    )
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("GPU indices must be unique and nonnegative")
    if not 1 <= args.port_base <= 65536 - len(args.gpus):
        parser.error("the requested port range must fit within 1..65535")
    if args.episodes is not None and args.episodes < 1:
        parser.error("episodes must be positive")
    if args.startup_timeout <= 0 or (args.client_timeout is not None and args.client_timeout <= 0):
        parser.error("timeouts must be positive")
    args.robotwin_root = args.robotwin_root.resolve()
    args.checkpoint_root = args.checkpoint_root.resolve()
    args.output = args.output.resolve()
    return args


def load_jobs(args):
    tasks = json.loads((args.robotwin_root / "pace/configs/tasks.json").read_text())
    if args.tasks and set(args.tasks) - tasks.keys():
        raise ValueError(f"unknown tasks: {sorted(set(args.tasks) - tasks.keys())}")
    jobs, keys = [], set()
    with args.jobs.open(newline="") as handle:
        for row in csv.DictReader(handle):
            job = Job(row["task"], row["method"], *(int(row[k]) for k in ("horizon", "seed", "batch", "episodes")))
            if job.task not in tasks or (job.method, job.horizon) not in METHODS:
                raise ValueError(f"invalid task or method in job CSV: {row}")
            if job.seed not in (0, 1, 2) or job.batch not in (1, 2, 3) or job.episodes < 1:
                raise ValueError(f"invalid seed, batch or episodes: {row}")
            for field in ("checkpoint_relative_path", "dataset_id"):
                if row[field] != tasks[job.task][field]:
                    raise ValueError(f"{job.task}: CSV {field} differs from the task mapping")
            if job.key in keys:
                raise ValueError(f"duplicate job: {job.key}")
            keys.add(job.key)
            if any(
                values and value not in values
                for value, values in (
                    (job.task, args.tasks),
                    (job.name, args.methods),
                    (job.seed, args.seeds),
                    (job.batch, args.batches),
                )
            ):
                continue
            if args.episodes is not None:
                job = Job(job.task, job.method, job.horizon, job.seed, job.batch, args.episodes)
            jobs.append(job)
    if not jobs:
        raise ValueError("no jobs match the selection")
    return jobs


def expected_settings(job, args):
    config_dir = args.robotwin_root / "pace/configs"
    paper = json.loads((config_dir / "paper.json").read_text())
    checkpoint = (args.checkpoint_root / job.task).resolve()
    expected = {
        "task": job.task,
        "method": job.name,
        "max_horizon": job.horizon,
        "seed": job.seed,
        "batch": job.batch,
        "expected_episodes": job.episodes,
        "checkpoint_id": str(checkpoint),
        "task_config": paper["evaluation"]["task_config"],
        "instruction_type": paper["evaluation"]["instruction_type"],
        "prediction_horizon": paper["evaluation"]["prediction_horizon"],
        "upstream_commit": paper["robotwin_commit"],
        "threshold": None,
    }
    if job.method == "pace":
        source = (config_dir / "calibration" / f"{job.task}.json").read_bytes()
        expected.update(
            selector=paper["selector"],
            threshold=json.loads(source)["threshold"],
            calibration_sha256=hashlib.sha256(source).hexdigest(),
        )
    return expected


def check_complete(job, args):
    """Only resume a complete, internally consistent result for this exact job."""
    path = args.output / job.key
    try:
        record = json.loads((path / "evaluation/metadata.json").read_text())
        for field, value in expected_settings(job, args).items():
            if record.get(field) != value:
                raise ValueError(f"{job.key}: existing {field} differs; use a separate output directory")
        if record.get("status") != "complete":
            return False
        server = json.loads((path / "server_metadata.json").read_text())
        norm = (args.checkpoint_root / job.task / "assets/robotwin/norm_stats.json").read_bytes()
        if server.get("norm_stats_sha256") != hashlib.sha256(norm).hexdigest():
            raise ValueError(f"{job.key}: normalization changed; use a separate output directory")
        episodes = [json.loads(line) for line in (path / "evaluation/episodes.jsonl").read_text().splitlines()]
        return (
            record.get("episodes") == job.episodes == len(episodes)
            and [episode.get("episode") for episode in episodes] == list(range(job.episodes))
            and all(type(episode.get("success")) is bool for episode in episodes)
            and sum(episode["success"] for episode in episodes) == record.get("successes")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def command(job, args, gpu, port):
    result = [args.server_python, str(LAUNCHER)]
    values = {
        "task": job.task,
        "checkpoint_root": args.checkpoint_root,
        "robotwin_root": args.robotwin_root,
        "client_python": args.client_python,
        "server_python": args.server_python,
        "gpu": gpu,
        "port": port,
        "method": job.method,
        "horizon": job.horizon,
        "seed": job.seed,
        "batch": job.batch,
        "episodes": job.episodes,
        "output": args.output / job.key,
        "startup_timeout": args.startup_timeout,
    }
    if args.client_timeout is not None:
        values["client_timeout"] = args.client_timeout
    for name, value in values.items():
        result.extend(("--" + name.replace("_", "-"), str(value)))
    if args.save_actions:
        result.append("--save-actions")
    return result


def save_summary(args, records, *, interrupted=False):
    summary = {
        "jobs_file": str(args.jobs.resolve()),
        "interrupted": interrupted,
        "counts": dict(Counter(record["status"] for record in records)),
        "jobs": records,
    }
    temporary = args.output / "batch_summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output / "batch_summary.json")


def run_jobs(args, jobs):
    records = [dict(**asdict(job), key=job.key, status="pending") for job in jobs]
    pending = deque()
    # Inspect all selected output directories before launching any GPU work.
    for job, record in zip(jobs, records, strict=True):
        path = args.output / job.key
        if not path.exists():
            pending.append((job, record))
        elif check_complete(job, args):
            record["status"] = "skipped"
        elif args.retry_failed:
            pending.append((job, record))
        else:
            record.update(status="failed", error="incomplete output exists; rerun with --retry-failed")

    slots = list(zip(args.gpus, range(args.port_base, args.port_base + len(args.gpus)), strict=True))
    if pending:
        for _, port in slots:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
    active = {}
    interrupted = False

    def interrupt(_signum, _frame):
        nonlocal interrupted
        interrupted = True

    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while (pending or active) and not interrupted:
            for gpu, (process, job, record) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                record["returncode"] = code
                try:
                    complete = code == 0 and check_complete(job, args)
                    record.update(status="complete" if complete else "failed")
                    if not complete:
                        record["error"] = "launcher failed or result is incomplete; see launcher_log"
                except ValueError as error:
                    record.update(status="failed", error=str(error))
                del active[gpu]
                print(f"{record['status'].upper()} {job.key}", flush=True)
            for gpu, port in slots:
                if interrupted or gpu in active or not pending:
                    continue
                job, record = pending.popleft()
                path = args.output / job.key
                if path.exists():
                    archive = args.output.with_name(args.output.name + "_failed") / job.key / str(time.time_ns())
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(archive)
                    record["previous_attempt"] = str(archive)
                log = args.output / "launcher_logs" / f"{job.key.replace('/', '_')}_{time.time_ns()}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                record.update(status="running", gpu=gpu, port=port, launcher_log=str(log))
                try:
                    with log.open("x") as handle:
                        process = subprocess.Popen(
                            command(job, args, gpu, port),
                            cwd=ROOT,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                except OSError as error:
                    record.update(status="failed", error=str(error))
                    continue
                active[gpu] = (process, job, record)
                print(f"RUN GPU{gpu} port={port} {job.key}", flush=True)
            save_summary(args, records)
            if active:
                time.sleep(0.2)
    finally:
        # Let each launcher finish its own server/client process-group cleanup.
        for process, _, _ in active.values():
            if process.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    process.terminate()
        for process, _, record in active.values():
            record.update(status="interrupted", returncode=process.wait())
        save_summary(args, records, interrupted=interrupted or bool(active))
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(json.dumps(dict(Counter(record["status"] for record in records))), flush=True)
    return 130 if interrupted else int(any(record["status"] == "failed" for record in records))


def main():
    args = parse_args()
    try:
        jobs = load_jobs(args)
        print(f"Selected {len(jobs)} jobs, {sum(job.episodes for job in jobs)} episodes; GPUs {args.gpus}", flush=True)
        if args.dry_run:
            for job in jobs:
                print(f"{job.key} episodes={job.episodes}")
            return 0
        for name in ("server_python", "client_python"):
            executable = shutil.which(getattr(args, name))
            if executable is None:
                raise ValueError(f"{name} is not executable: {getattr(args, name)}")
            setattr(args, name, str(Path(executable).absolute()))
        for task in {job.task for job in jobs}:
            checkpoint = args.checkpoint_root / task
            if not (checkpoint / "params").is_dir() or not (checkpoint / "assets/robotwin/norm_stats.json").is_file():
                raise ValueError(f"missing checkpoint parameters or normalization: {checkpoint}")
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".batch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return run_jobs(args, jobs)
    except (OSError, ValueError, KeyError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
