import contextlib
import os
import select
import signal
import socket
import subprocess
import sys

import pytest
from run_robotwin import check_port
from run_robotwin import stop


def test_port_reuse_after_closed_connection_still_rejects_live_listener():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(OSError, match=r"[Aa]ddress already in use"):
            check_port(port)
        with socket.create_connection(("127.0.0.1", port)) as client:
            connection, _ = listener.accept()
            with connection:
                connection.shutdown(socket.SHUT_WR)
            assert client.recv(1) == b""
    check_port(port)


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
