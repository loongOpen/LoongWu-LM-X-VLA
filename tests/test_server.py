"""Real ZeroMQ sockets with a deterministic policy; no checkpoint required."""

import io
import threading
import time

import msgpack
import numpy as np
import pytest
import zmq

from lm_x.server import LMXClient, LMXServer, _MessageCodec


class Policy:
    def get_action(self, observation, options=None):
        if observation.get("fail"):
            raise ValueError("invalid observation")
        return {"arm": observation["state"]["arm"]}, {"options": options}

    def reset(self, options=None):
        return {"options": options}

    def get_io_spec(self, embodiment_tag=None):
        return {"schema_version": 1, "embodiment_tag": embodiment_tag}

    def get_modality_config(self, embodiment_tag=None):
        return {"embodiment_tag": embodiment_tag}


@pytest.fixture
def service():
    server = LMXServer(Policy(), port=0, api_token="secret")
    worker = threading.Thread(target=server.run)
    worker.start()
    try:
        yield server
    finally:
        server.stop()
        worker.join(timeout=3)
        assert not worker.is_alive(), "Idle server failed to stop"
        server.close()


def test_real_roundtrip_schema_options_and_arrays(service):
    values = np.arange(28, dtype=np.float32).reshape(2, 2, 7)
    with LMXClient(port=service.port, api_token="secret", embodiment_tag="robot") as client:
        assert client.ping()
        assert client.get_io_spec() == {"schema_version": 1, "embodiment_tag": "robot"}
        options = {"extra": 1}
        action, info = client.get_action({"state": {"arm": values}}, options)
        np.testing.assert_array_equal(action["arm"], values)
        assert info["options"] == {"extra": 1, "embodiment_tag": "robot"}
        assert options == {"extra": 1}
        assert client.reset()["options"] == {"embodiment_tag": "robot"}


def test_existing_robot_client_wire_format(service):
    def encode(value):
        if isinstance(value, np.ndarray):
            output = io.BytesIO()
            np.save(output, value, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return value

    def decode(value):
        if isinstance(value, dict) and "__ndarray_class__" in value:
            return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
        return value

    values = np.arange(14, dtype=np.float32).reshape(1, 2, 7)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{service.port}")
    try:
        request = {
            "endpoint": "get_action",
            "api_token": "secret",
            "data": {"observation": {"state": {"arm": values}}, "options": None},
        }
        socket.send(msgpack.packb(request, default=encode))
        action, _ = msgpack.unpackb(socket.recv(), object_hook=decode)
        np.testing.assert_array_equal(action["arm"], values)
    finally:
        socket.close(linger=0)
        context.term()


def test_auth_and_policy_errors_do_not_poison_socket(service):
    with LMXClient(port=service.port, api_token="wrong") as client:
        with pytest.raises(RuntimeError, match="invalid API token"):
            client.get_io_spec()
        client.api_token = "secret"
        assert client.ping()
        with pytest.raises(RuntimeError, match="invalid observation"):
            client.get_action({"fail": True})
        assert client.ping()
        with pytest.raises(RuntimeError, match="unknown endpoint"):
            client.call_endpoint("unknown")
        assert client.ping()


def test_malformed_message_then_valid_request(service):
    with LMXClient(port=service.port, api_token="secret") as client:
        client.socket.send(b"\xc1")
        response = _MessageCodec.loads(client.socket.recv())
        assert "error_type" in response
        assert client.ping()


def test_timeout_can_recover_on_same_client():
    class SlowPolicy(Policy):
        def get_action(self, observation, options=None):
            time.sleep(0.08)
            return super().get_action(observation, options)

    server = LMXServer(SlowPolicy(), port=0)
    worker = threading.Thread(target=server.run)
    worker.start()
    try:
        with LMXClient(port=server.port, timeout_ms=20) as client:
            with pytest.raises(zmq.Again):
                client.get_action({"state": {"arm": np.zeros((1, 1, 1), np.float32)}})
            client.timeout_ms = 1000
            client._init_socket()
            assert client.ping()
            client.stop_server()
        worker.join(timeout=3)
        assert not worker.is_alive()
    finally:
        server.stop()
        worker.join(timeout=3)
        server.close()


def test_codec_rejects_pickle_and_archives():
    with pytest.raises(ValueError):
        _MessageCodec.dumps(np.array([object()], dtype=object))
    archive = io.BytesIO()
    np.savez(archive, data=np.zeros(1))
    with pytest.raises(ValueError, match="not an archive"):
        _MessageCodec._decode({"__ndarray__": True, "data": archive.getvalue()})


@pytest.mark.parametrize("timeout", [0, -1, True, 1.5])
def test_invalid_timeout_rejected(timeout):
    with pytest.raises(ValueError):
        LMXClient(timeout_ms=timeout)
