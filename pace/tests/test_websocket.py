import functools
import inspect
import multiprocessing
import socket
import time
import urllib.error
import urllib.request

import numpy as np
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy
import pytest
import websockets.sync.client
from websockets.sync.client import connect

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("test server exited")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.1)
    raise TimeoutError("test server did not start")


class CountingPolicy:
    def __init__(self):
        self.queries = 0

    def infer(self, observation):
        self.queries += 1
        return {"actions": np.full((50, 14), self.queries, dtype=np.float32)}


def serve_counter(port):
    WebsocketPolicyServer(CountingPolicy(), host="127.0.0.1", port=port, metadata={"task": "fixture"}).serve_forever()


class SlowPolicy(CountingPolicy):
    def infer(self, observation):
        time.sleep(0.3)
        return super().infer(observation)


def serve_slow_counter(port):
    WebsocketPolicyServer(SlowPolicy(), host="127.0.0.1", port=port).serve_forever()


class ProcessAdapter:
    def __init__(self, process):
        self.process = process

    def poll(self):
        return self.process.exitcode


def test_metadata_probe_does_not_consume_first_query():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = multiprocessing.get_context("spawn").Process(target=serve_counter, args=(port,))
    process.start()
    try:
        wait_ready(ProcessAdapter(process), port, 30)
        with connect(f"ws://127.0.0.1:{port}") as connection:
            assert msgpack_numpy.unpackb(connection.recv(timeout=5)) == {"task": "fixture"}
        with connect(f"ws://127.0.0.1:{port}") as connection:
            connection.recv(timeout=5)
            connection.send(msgpack_numpy.packb({"state": np.zeros(14, dtype=np.float32)}))
            result = msgpack_numpy.unpackb(connection.recv(timeout=5))
        np.testing.assert_array_equal(result["actions"], np.ones((50, 14), dtype=np.float32))
    finally:
        process.terminate()
        process.join(timeout=10)


def test_blocking_inference_does_not_trigger_client_heartbeat(monkeypatch):
    original_connect = websockets.sync.client.connect
    if "ping_interval" not in inspect.signature(original_connect).parameters:
        pytest.skip("Older synchronous clients do not send keepalive pings")

    connections = []

    @functools.wraps(original_connect)
    def accelerated_heartbeat(*args, **kwargs):
        kwargs.setdefault("ping_interval", 0.01)
        kwargs.setdefault("ping_timeout", 0.02)
        connection = original_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(websockets.sync.client, "connect", accelerated_heartbeat)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = multiprocessing.get_context("spawn").Process(target=serve_slow_counter, args=(port,))
    process.start()
    try:
        wait_ready(ProcessAdapter(process), port, 30)
        client = WebsocketClientPolicy("127.0.0.1", port)
        for expected in (1, 2):
            result = client.infer({"state": np.zeros(14, dtype=np.float32)})
            np.testing.assert_array_equal(result["actions"], np.full((50, 14), expected, dtype=np.float32))
    finally:
        for connection in connections:
            connection.close()
        process.terminate()
        process.join(timeout=10)


def test_client_accepts_pre_keepalive_connect_api(monkeypatch):
    class MetadataConnection:
        def recv(self):
            return msgpack_numpy.packb({"task": "legacy_api"})

    def legacy_connect(uri, *, compression, max_size, additional_headers):
        assert uri == "ws://localhost:8000"
        assert compression is None
        assert max_size is None
        assert additional_headers is None
        return MetadataConnection()

    monkeypatch.setattr(websockets.sync.client, "connect", legacy_connect)
    assert WebsocketClientPolicy("localhost", 8000).get_server_metadata() == {"task": "legacy_api"}
