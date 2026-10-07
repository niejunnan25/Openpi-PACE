import contextlib
import os
import select
import signal
import subprocess
import sys

from run_robotwin import stop


def test_cleanup_stops_workers_after_parent_has_exited():
    worker = (
        "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
    )
    parent_code = f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {worker!r}])"
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code], stdout=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        assert select.select([parent.stdout], [], [], 10)[0]
        assert parent.stdout.readline().strip() == "ready"
        parent.wait(timeout=10)
        stop(parent)
        assert select.select([parent.stdout], [], [], 2)[0], "worker survived its exited parent"
        assert parent.stdout.read() == ""
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)
        parent.stdout.close()
