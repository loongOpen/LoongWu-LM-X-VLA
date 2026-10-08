"""龙悟LM-X public inference API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .embodiment_tags import EmbodimentTag
    from .observation import make_sample_observation
    from .policy import LMXPolicy
    from .server import LMXClient, LMXServer
    from .validation import InferenceValidationError

__all__ = [
    "EmbodimentTag",
    "LMXPolicy",
    "LMXClient",
    "LMXServer",
    "make_sample_observation",
    "InferenceValidationError",
]


def __getattr__(name: str) -> Any:
    """Load model and optional service dependencies only when requested."""
    if name == "EmbodimentTag":
        from .embodiment_tags import EmbodimentTag

        return EmbodimentTag
    if name == "LMXPolicy":
        from .policy import LMXPolicy

        return LMXPolicy
    if name == "make_sample_observation":
        from .observation import make_sample_observation

        return make_sample_observation
    if name == "InferenceValidationError":
        from .validation import InferenceValidationError

        return InferenceValidationError
    if name in {"LMXClient", "LMXServer"}:
        try:
            from .server import LMXClient, LMXServer
        except ModuleNotFoundError as exc:
            if exc.name in {"msgpack", "zmq"}:
                raise ImportError(
                    "The ZeroMQ interface requires the 'server' extra. "
                    "Install it with: uv sync --extra server"
                ) from exc
            raise
        return {"LMXClient": LMXClient, "LMXServer": LMXServer}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
