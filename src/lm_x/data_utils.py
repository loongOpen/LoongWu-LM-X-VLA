"""Array transforms shared by checkpoint preprocessing and action decoding."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from lm_x.types import ModalityConfig


def encode_periodic(values: np.ndarray) -> np.ndarray:
    """Represent each angle-like feature by adjacent sine and cosine banks."""
    array = np.asarray(values)
    return np.concatenate((np.sin(array), np.cos(array)), axis=-1)


def tree_arrays(value: Any) -> Any:
    """Convert list leaves in a nested mapping to arrays without changing other leaves."""
    if isinstance(value, Mapping):
        return {key: tree_arrays(item) for key, item in value.items()}
    return np.asarray(value) if isinstance(value, list) else value


class _NumericTransform:
    """Broadcast-aware normalization kernels with explicit degenerate-feature behavior."""

    @staticmethod
    def _dtype(*values: Any) -> np.dtype:
        return np.result_type(*(np.asarray(value).dtype for value in values), np.float32)

    @classmethod
    def to_symmetric_range(cls, values: Any, params: Mapping[str, Any]) -> np.ndarray:
        source = np.asarray(values)
        lower = np.asarray(params["min"])
        upper = np.asarray(params["max"])
        span = upper - lower
        usable = ~np.isclose(span, 0)
        ratio = np.zeros(source.shape, dtype=cls._dtype(source, lower, upper))
        np.divide(source - lower, span, out=ratio, where=usable)
        transformed = 2.0 * ratio - 1.0
        return np.where(usable, transformed, 0.0).astype(ratio.dtype, copy=False)

    @classmethod
    def from_symmetric_range(cls, values: Any, params: Mapping[str, Any]) -> np.ndarray:
        source = np.asarray(values)
        lower = np.asarray(params["min"])
        upper = np.asarray(params["max"])
        clipped = np.clip(source, -1.0, 1.0)
        return ((clipped + 1.0) * 0.5 * (upper - lower) + lower).astype(
            cls._dtype(source, lower, upper),
            copy=False,
        )

    @classmethod
    def to_standard_score(cls, values: Any, params: Mapping[str, Any]) -> np.ndarray:
        source = np.asarray(values)
        mean = np.asarray(params["mean"])
        std = np.asarray(params["std"])
        result = source.astype(cls._dtype(source, mean, std), copy=True)
        np.divide(source - mean, std, out=result, where=std != 0)
        return result

    @classmethod
    def from_standard_score(cls, values: Any, params: Mapping[str, Any]) -> np.ndarray:
        source = np.asarray(values)
        mean = np.asarray(params["mean"])
        std = np.asarray(params["std"])
        restored = source * std + mean
        return np.where(std != 0, restored, source).astype(
            cls._dtype(source, mean, std),
            copy=False,
        )


def scale_to_unit(values: Any, params: Mapping[str, Any]) -> np.ndarray:
    """Map min/max statistics to ``[-1, 1]``."""
    return _NumericTransform.to_symmetric_range(values, params)


def scale_from_unit(values: Any, params: Mapping[str, Any]) -> np.ndarray:
    """Restore values from a clipped ``[-1, 1]`` representation."""
    return _NumericTransform.from_symmetric_range(values, params)


def standardize(values: Any, params: Mapping[str, Any]) -> np.ndarray:
    return _NumericTransform.to_standard_score(values, params)


def destandardize(values: Any, params: Mapping[str, Any]) -> np.ndarray:
    return _NumericTransform.from_standard_score(values, params)


def decode_modality_configs(
    raw_configs: Mapping[str, Mapping[str, ModalityConfig | dict[str, Any]]],
) -> dict[str, dict[str, ModalityConfig]]:
    """Materialize checkpoint modality dictionaries into validated schema objects."""
    decoded: dict[str, dict[str, ModalityConfig]] = {}
    for embodiment, modalities in raw_configs.items():
        decoded[str(embodiment)] = {
            str(name): config if isinstance(config, ModalityConfig) else ModalityConfig(**config)
            for name, config in modalities.items()
        }
    return decoded


# Compatibility names for integrations written against runtime 0.2.
apply_sin_cos_encoding = encode_periodic
nested_dict_to_numpy = tree_arrays
normalize_values_minmax = scale_to_unit
unnormalize_values_minmax = scale_from_unit
normalize_values_meanstd = standardize
unnormalize_values_meanstd = destandardize
parse_modality_configs = decode_modality_configs


__all__ = [
    "decode_modality_configs",
    "destandardize",
    "encode_periodic",
    "scale_from_unit",
    "scale_to_unit",
    "standardize",
    "tree_arrays",
]
