"""Opt-in real checkpoint load plus one synthetic inference.

Set both ``LMX_MODEL_PATH`` and ``LMX_BACKBONE_PATH`` to materialized local
directories. The test is skipped when either variable is absent.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any

from ._support import LFS_HEADER, SRC_ROOT

try:
    import pytest
except ModuleNotFoundError:  # unittest/compile-only environments need no pytest.
    pytest = None

if pytest is not None:
    pytestmark = [pytest.mark.gpu, pytest.mark.real_checkpoint]


def _reject_lfs_pointer(path: Path) -> None:
    if path.stat().st_size == 0:
        raise AssertionError(f"Checkpoint file is empty: {path}")
    with path.open("rb") as handle:
        header = handle.read(len(LFS_HEADER))
    if header == LFS_HEADER:
        raise AssertionError(f"Git LFS pointer is not materialized checkpoint content: {path}")


def _validate_materialized_backbone(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise AssertionError(f"LMX_BACKBONE_PATH is not a directory: {root}")
    config = root / "config.json"
    if not config.is_file():
        raise AssertionError(f"Backbone config is missing: {config}")
    _reject_lfs_pointer(config)

    weight_files = [
        candidate
        for candidate in (root / "model.safetensors", root / "pytorch_model.bin")
        if candidate.is_file()
    ]
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        _reject_lfs_pointer(index_path)
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            shard_names = set(payload["weight_map"].values())
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AssertionError(f"Invalid backbone weight index: {index_path}") from exc
        if not shard_names:
            raise AssertionError(f"Backbone weight index is empty: {index_path}")
        for shard_name in shard_names:
            shard = root / str(shard_name)
            if not shard.is_file():
                raise AssertionError(f"Backbone weight shard is missing: {shard}")
            weight_files.append(shard)

    if not weight_files:
        raise AssertionError(f"No materialized backbone weights found under {root}")
    for weight_file in weight_files:
        _reject_lfs_pointer(weight_file)
    return root


def _dimension_from_norm_params(processor: Any, tag_value: str, modality: str, key: str) -> int:
    import numpy as np

    params = processor.state_action_processor.norm_params[tag_value][modality][key]
    dimension = params.get("dim")
    if dimension is not None:
        if hasattr(dimension, "item"):
            dimension = dimension.item()
        return int(dimension)
    for statistic in ("mean", "std", "min", "max", "q01", "q99"):
        if statistic in params:
            return int(np.asarray(params[statistic]).shape[-1])
    raise AssertionError(f"Cannot infer {modality} dimension for {tag_value}/{key}")


class RealCheckpointSmokeTest(unittest.TestCase):
    def test_real_checkpoint_load_and_one_inference(self) -> None:
        model_env = os.environ.get("LMX_MODEL_PATH")
        backbone_env = os.environ.get("LMX_BACKBONE_PATH")
        if not model_env or not backbone_env:
            self.skipTest("Set LMX_MODEL_PATH and LMX_BACKBONE_PATH for the real smoke test.")

        import numpy as np
        import torch

        sys.path.insert(0, str(SRC_ROOT))
        try:
            from lm_x.checkpoint import validate_checkpoint_manifest
            from lm_x.policy import LMXPolicy
            from lm_x.types import EmbodimentTag
        finally:
            try:
                sys.path.remove(str(SRC_ROOT))
            except ValueError:
                pass

        model_path = validate_checkpoint_manifest(Path(model_env))
        backbone_path = _validate_materialized_backbone(Path(backbone_env))
        device = os.environ.get("LMX_DEVICE", "cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            self.skipTest(f"CUDA device requested but unavailable: {device}")
        if device.startswith("mps") and not torch.backends.mps.is_available():
            self.skipTest(f"MPS device requested but unavailable: {device}")

        dtype_name = os.environ.get("LMX_DTYPE", "bfloat16")
        dtype = getattr(torch, dtype_name, None)
        self.assertIsInstance(dtype, torch.dtype, f"Unknown LMX_DTYPE: {dtype_name}")

        self.assertNotEqual(
            os.environ.get("LMX_SKIP_HF_MODEL_WEIGHTS"),
            "1",
            "LMX_SKIP_HF_MODEL_WEIGHTS=1 would turn this into an architecture-only test.",
        )

        policy = LMXPolicy(
            embodiment_tag=None,
            model_path=model_path,
            backbone_path=backbone_path,
            device=device,
            local_files_only=True,
            dtype=dtype,
        )
        self.assertFalse(policy.model.training, "Model must be in eval mode for inference.")

        requested_tag = os.environ.get("LMX_EMBODIMENT_TAG")
        if requested_tag:
            tag = EmbodimentTag.resolve(requested_tag)
            self.assertIn(
                tag.value,
                policy.all_modality_configs,
                f"Checkpoint has no modality config for {requested_tag!r}",
            )
        else:
            available_values = sorted(policy.all_modality_configs)
            self.assertTrue(available_values, "Checkpoint exposes no embodiment modality configs.")
            tag = EmbodimentTag.resolve(available_values[0])

        configs = policy.all_modality_configs[tag.value]
        image_size = int(os.environ.get("LMX_SMOKE_IMAGE_SIZE", "256"))
        observation: dict[str, dict[str, Any]] = {
            "video": {},
            "state": {},
            "language": {},
        }
        for key in configs["video"].modality_keys:
            horizon = len(configs["video"].delta_indices)
            observation["video"][key] = np.zeros(
                (1, horizon, image_size, image_size, 3),
                dtype=np.uint8,
            )
        for key in configs["state"].modality_keys:
            horizon = len(configs["state"].delta_indices)
            dimension = _dimension_from_norm_params(policy.processor, tag.value, "state", key)
            observation["state"][key] = np.zeros(
                (1, horizon, dimension),
                dtype=np.float32,
            )
        for key in configs["language"].modality_keys:
            horizon = len(configs["language"].delta_indices)
            self.assertEqual(
                horizon,
                1,
                f"LMXPolicy currently requires one language step, got {horizon} for {key}",
            )
            observation["language"][key] = [["perform a safe no-op motion"]]

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        action, info = policy.get_action(
            observation,
            options={"embodiment_tag": tag.value},
        )

        expected_keys = set(configs["action"].modality_keys)
        self.assertEqual(set(action), expected_keys)
        self.assertIsInstance(info, dict)
        action_horizon = len(configs["action"].delta_indices)
        for key in sorted(expected_keys):
            value = action[key]
            expected_dimension = _dimension_from_norm_params(
                policy.processor,
                tag.value,
                "action",
                key,
            )
            self.assertIsInstance(value, np.ndarray)
            self.assertEqual(value.dtype, np.float32)
            self.assertEqual(value.shape, (1, action_horizon, expected_dimension))
            self.assertTrue(np.isfinite(value).all(), f"Non-finite action values for {key}")


if __name__ == "__main__":
    unittest.main()
