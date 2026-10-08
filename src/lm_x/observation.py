"""Serializable inference specifications and synthetic examples (no model imports)."""

from __future__ import annotations

from typing import Any

import numpy as np

from .validation import dimensions, modality_field


def build_io_spec(configs, norm_params, *, embodiment_tag: str, image_size=(256, 256)) -> dict:
    modalities = {}
    for name in ("video", "state", "language", "action"):
        config = configs[name]
        keys = list(modality_field(config, "modality_keys"))
        modalities[name] = {
            "keys": keys,
            "delta_indices": list(modality_field(config, "delta_indices")),
            "dtype": "uint8" if name == "video" else "str" if name == "language" else "float32",
        }
        if name in {"state", "action"}:
            modalities[name]["dimensions"] = dimensions(norm_params, name, keys)
    return {
        "schema_version": 1,
        "embodiment_tag": embodiment_tag,
        "modalities": modalities,
        "example_image_size": [int(size) for size in image_size],
    }


def make_sample_observation(
    spec: dict[str, Any], *, batch_size: int = 1, instruction: str = "move to the target"
) -> dict[str, dict[str, Any]]:
    """Build zero-valued synthetic inputs; never use these as live robot observations."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(instruction, str):
        raise TypeError("instruction must be a string")
    height, width = spec["example_image_size"]
    if min(height, width) <= 0:
        raise ValueError("example_image_size must be positive")
    observation = {name: {} for name in ("video", "state", "language")}
    for name, streams in observation.items():
        config = spec["modalities"][name]
        horizon = len(config["delta_indices"])
        if horizon <= 0:
            raise ValueError(f"{name} horizon must be positive")
        for key in config["keys"]:
            if name == "video":
                streams[key] = np.zeros((batch_size, horizon, height, width, 3), dtype=np.uint8)
            elif name == "state":
                dimension = config["dimensions"][key]
                if dimension <= 0:
                    raise ValueError(f"state.{key} dimension must be positive")
                streams[key] = np.zeros((batch_size, horizon, dimension), dtype=np.float32)
            else:
                if horizon != 1:
                    raise ValueError("Language horizon must be 1")
                streams[key] = [[instruction] for _ in range(batch_size)]
    return observation
