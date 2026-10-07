import multiprocessing
import socket

import numpy as np
from openpi_client import msgpack_numpy
from run_robotwin import wait_ready
from websockets.sync.client import connect

from openpi.serving.websocket_policy_server import WebsocketPolicyServer


class CountingPolicy:
    def __init__(self):
        self.queries = 0

    def infer(self, observation):
        self.queries += 1
        return {"actions": np.full((50, 14), self.queries, dtype=np.float32)}


def serve_counter(port):
    WebsocketPolicyServer(CountingPolicy(), host="127.0.0.1", port=port, metadata={"task": "fixture"}).serve_forever()


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
