"""NumPy-only input/output contracts shared by local and served inference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class InferenceValidationError(ValueError):
    """An observation, action, or checkpoint contract is invalid."""


def modality_field(config: Any, name: str) -> Any:
    return config[name] if isinstance(config, Mapping) else getattr(config, name)


def dimensions(norm_params: Mapping, modality: str, keys: list[str]) -> dict[str, int]:
    result = {}
    for key in keys:
        try:
            value = np.asarray(norm_params[modality][key]["dim"]).item()
            dimension = int(value)
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise InferenceValidationError(
                f"Missing or invalid normalization dimension for {modality}.{key}"
            ) from exc
        if isinstance(value, bool) or dimension <= 0 or dimension != value:
            raise InferenceValidationError(f"Invalid dimension for {modality}.{key}: {value!r}")
        result[key] = dimension
    return result


def _mapping(value: Any, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise InferenceValidationError(f"{label} must be a mapping, got {type(value).__name__}")
    return value


def _keys(value: Mapping, expected: list[str], label: str) -> None:
    missing = set(expected) - set(value)
    unexpected = set(value) - set(expected)
    if missing or unexpected:
        raise InferenceValidationError(
            f"{label} keys do not match checkpoint: missing={sorted(missing)}, "
            f"unexpected={sorted(map(str, unexpected))}"
        )


def _array(value: Any, label: str, dtype: Any, rank: int, horizon: int) -> int:
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise InferenceValidationError(f"{label} must be a numpy {np.dtype(dtype)} array")
    if value.ndim != rank or any(size <= 0 for size in value.shape):
        raise InferenceValidationError(
            f"{label} must have {rank} non-empty axes, got {value.shape}"
        )
    if value.shape[1] != horizon:
        raise InferenceValidationError(f"{label} horizon must be {horizon}, got {value.shape[1]}")
    if dtype == np.float32 and not np.isfinite(value).all():
        raise InferenceValidationError(f"{label} contains NaN or infinity")
    return value.shape[0]


def _batch(actual: int, expected: int | None, label: str) -> int:
    if actual <= 0 or (expected is not None and actual != expected):
        raise InferenceValidationError(
            f"{label} batch must be {expected or 'positive'}, got {actual}"
        )
    return actual


def validate_observation(observation: Any, configs: Mapping, norm_params: Mapping) -> int:
    """Validate before preprocessing; remains active under ``python -O``."""
    observation = _mapping(observation, "observation")
    batch = None
    for modality in ("video", "state", "language"):
        streams = _mapping(observation.get(modality), modality)
        config = configs[modality]
        keys = modality_field(config, "modality_keys")
        horizon = len(modality_field(config, "delta_indices"))
        if not keys or horizon <= 0:
            raise InferenceValidationError(
                f"Checkpoint {modality} keys and horizon must be non-empty"
            )
        _keys(streams, keys, modality)
        dims = dimensions(norm_params, modality, keys) if modality == "state" else {}
        for key in keys:
            value, label = streams[key], f"{modality}.{key}"
            if modality == "language":
                if horizon != 1 or not isinstance(value, list):
                    raise InferenceValidationError(
                        f"{label} requires list[list[str]] and horizon 1"
                    )
                batch = _batch(len(value), batch, label)
                if any(
                    not isinstance(item, list) or len(item) != 1 or not isinstance(item[0], str)
                    for item in value
                ):
                    raise InferenceValidationError(
                        f"{label} requires one instruction per batch item"
                    )
            else:
                dtype, rank = (np.uint8, 5) if modality == "video" else (np.float32, 3)
                batch = _batch(_array(value, label, dtype, rank, horizon), batch, label)
                expected = 3 if modality == "video" else dims[key]
                if value.shape[-1] != expected:
                    raise InferenceValidationError(
                        f"{label} final dimension must be {expected}, got {value.shape[-1]}"
                    )
    return batch


def validate_actions(
    actions: Any, config: Any, norm_params: Mapping, *, batch_size: int | None = None
) -> dict[str, tuple[int, ...]]:
    actions = _mapping(actions, "actions")
    keys = modality_field(config, "modality_keys")
    horizon = len(modality_field(config, "delta_indices"))
    if not keys or horizon <= 0:
        raise InferenceValidationError("Checkpoint action keys and horizon must be non-empty")
    _keys(actions, keys, "actions")
    dims = dimensions(norm_params, "action", keys)
    shapes = {}
    for key in keys:
        value, label = actions[key], f"actions.{key}"
        batch_size = _batch(_array(value, label, np.float32, 3, horizon), batch_size, label)
        if value.shape[-1] != dims[key]:
            raise InferenceValidationError(
                f"{label} dimension must be {dims[key]}, got {value.shape[-1]}"
            )
        shapes[key] = value.shape
    return shapes
