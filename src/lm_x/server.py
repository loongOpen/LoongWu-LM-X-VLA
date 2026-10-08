"""Optional ZeroMQ transport for inference policies."""

from __future__ import annotations

import hmac
import io
import threading
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Callable, Protocol

import msgpack
import numpy as np
import zmq


class _Policy(Protocol):
    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]: ...


class _MessageCodec:
    @staticmethod
    def dumps(data: Any) -> bytes:
        return msgpack.packb(data, default=_MessageCodec._encode, use_bin_type=True)

    @staticmethod
    def loads(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=_MessageCodec._decode, raw=False)

    @staticmethod
    def _encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            # Keep the wire marker used by existing robot clients.
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if isinstance(obj, np.generic):
            return obj.item()
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Cannot serialize object of type {type(obj).__name__}")

    @staticmethod
    def _decode(obj: Any) -> Any:
        if not isinstance(obj, dict):
            return obj
        payload = None
        if obj.get("__ndarray_class__") is True:
            payload = obj.get("as_npy")
        elif obj.get("__ndarray__") is True:
            payload = obj.get("data")
        if payload is not None:
            value = np.load(io.BytesIO(payload), allow_pickle=False)
            if not isinstance(value, np.ndarray):
                value.close()
                raise ValueError("Expected a NumPy array, not an archive")
            return value
        return obj


@dataclass(frozen=True)
class _Endpoint:
    handler: Callable[..., Any]
    requires_input: bool = True


class LMXServer:
    """Serve a policy through a ZeroMQ request-reply socket."""

    def __init__(
        self,
        policy: _Policy,
        host: str = "127.0.0.1",
        port: int = 5555,
        api_token: str | None = None,
    ) -> None:
        self.policy = policy
        if api_token is not None and (not isinstance(api_token, str) or not api_token):
            raise ValueError("api_token must be a non-empty string or None")
        self.api_token = api_token
        self.running = True
        self._stop_event = threading.Event()
        self._serving_thread = None
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        try:
            if port == 0:
                self.port = self.socket.bind_to_random_port(f"tcp://{host}")
            else:
                self.socket.bind(f"tcp://{host}:{port}")
                self.port = port
        except Exception:
            self.socket.close(linger=0)
            self.context.term()
            raise
        self._endpoints: dict[str, _Endpoint] = {}
        self.register_endpoint("ping", self._ping, requires_input=False)
        self.register_endpoint("shutdown", self._shutdown, requires_input=False)
        self.register_endpoint("kill", self._shutdown, requires_input=False)
        self.register_endpoint("get_action", self.policy.get_action)
        self.register_endpoint("reset", self.policy.reset)
        self.register_endpoint(
            "get_modality_config",
            getattr(self.policy, "get_modality_config", lambda **_: {}),
        )
        if hasattr(self.policy, "get_io_spec"):
            self.register_endpoint("get_io_spec", self.policy.get_io_spec)

    def register_endpoint(
        self, name: str, handler: Callable[..., Any], requires_input: bool = True
    ) -> None:
        if not name:
            raise ValueError("endpoint name must not be empty")
        self._endpoints[name] = _Endpoint(handler=handler, requires_input=requires_input)

    def _ping(self) -> dict[str, str]:
        return {"status": "ok"}

    def _shutdown(self) -> dict[str, str]:
        self.running = False
        return {"status": "stopping"}

    def _authorized(self, request: dict[str, Any]) -> bool:
        token = request.get("api_token")
        return self.api_token is None or (
            isinstance(token, str)
            and hmac.compare_digest(token.encode("utf-8"), self.api_token.encode("utf-8"))
        )

    def _dispatch(self, request: Any) -> Any:
        if not isinstance(request, dict):
            raise TypeError("request must be a mapping")
        if not self._authorized(request):
            raise PermissionError("invalid API token")

        endpoint_name = request.get("endpoint", "get_action")
        endpoint = self._endpoints.get(endpoint_name)
        if endpoint is None:
            raise ValueError(f"unknown endpoint: {endpoint_name}")
        if not endpoint.requires_input:
            return endpoint.handler()

        data = request.get("data", {})
        if not isinstance(data, dict):
            raise TypeError("request data must be a mapping")
        return endpoint.handler(**data)

    def run(self) -> None:
        self._serving_thread = threading.get_ident()
        try:
            while self.running and not self._stop_event.is_set():
                if not self.socket.poll(timeout=100):
                    continue
                raw_request = self.socket.recv()
                try:
                    request = _MessageCodec.loads(raw_request)
                    response = self._dispatch(request)
                    encoded_response = _MessageCodec.dumps(response)
                except Exception as exc:
                    encoded_response = _MessageCodec.dumps(
                        {"error": str(exc), "error_type": type(exc).__name__}
                    )
                self.socket.send(encoded_response)
        finally:
            self._serving_thread = None
            self.close()

    def stop(self) -> None:
        """Request shutdown without touching a socket owned by another thread."""
        self._stop_event.set()

    def close(self) -> None:
        self.stop()
        if self._serving_thread is not None and self._serving_thread != threading.get_ident():
            return
        self.running = False
        if not self.socket.closed:
            self.socket.close(linger=0)
        if not self.context.closed:
            self.context.term()

    def __enter__(self) -> LMXServer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class LMXClient:
    """Client for a :class:`LMXServer` request-reply endpoint."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 15_000,
        api_token: str | None = None,
        embodiment_tag: str | None = None,
    ) -> None:
        self.host = host
        self.port = port
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or timeout_ms <= 0:
            raise ValueError("timeout_ms must be a positive integer")
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self.embodiment_tag = embodiment_tag
        self.context = zmq.Context()
        self.socket: zmq.Socket
        self._init_socket()

    def _init_socket(self) -> None:
        old_socket = getattr(self, "socket", None)
        if old_socket is not None and not old_socket.closed:
            old_socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(
        self, endpoint: str, data: dict[str, Any] | None = None, *, requires_input: bool = True
    ) -> Any:
        if self.context.closed or self.socket.closed:
            raise RuntimeError("LMXClient is closed")
        request: dict[str, Any] = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token is not None:
            request["api_token"] = self.api_token

        try:
            self.socket.send(_MessageCodec.dumps(request))
            response = _MessageCodec.loads(self.socket.recv())
        except zmq.error.Again:
            self._init_socket()
            raise

        if isinstance(response, dict) and "error" in response:
            error_type = response.get("error_type", "ServerError")
            raise RuntimeError(f"{error_type}: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            response = self.call_endpoint("ping", requires_input=False)
        except zmq.error.ZMQError:
            self._init_socket()
            return False
        return isinstance(response, dict) and response.get("status") == "ok"

    def stop_server(self) -> None:
        self.call_endpoint("shutdown", requires_input=False)

    def _request_options(self, options: dict[str, Any] | None) -> dict[str, Any] | None:
        if self.embodiment_tag is None:
            return options
        request_options = dict(options or {})
        request_options.setdefault("embodiment_tag", self.embodiment_tag)
        return request_options

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.call_endpoint(
            "get_action",
            {"observation": observation, "options": self._request_options(options)},
        )
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise TypeError("server returned an invalid action response")
        return response[0], response[1]

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call_endpoint("reset", {"options": self._request_options(options)})

    def get_modality_config(self, embodiment_tag: str | None = None) -> dict[str, Any]:
        tag = embodiment_tag or self.embodiment_tag
        data = {"embodiment_tag": tag} if tag is not None else {}
        return self.call_endpoint("get_modality_config", data)

    def get_io_spec(self, embodiment_tag: str | None = None) -> dict[str, Any]:
        """Return the JSON-compatible contract, identical to ``LMXPolicy.get_io_spec``."""
        tag = embodiment_tag or self.embodiment_tag
        return self.call_endpoint("get_io_spec", {"embodiment_tag": tag} if tag else {})

    def close(self) -> None:
        socket = getattr(self, "socket", None)
        if socket is not None and not socket.closed:
            socket.close(linger=0)
        context = getattr(self, "context", None)
        if context is not None and not context.closed:
            context.term()

    def __enter__(self) -> LMXClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = ["LMXClient", "LMXServer"]
