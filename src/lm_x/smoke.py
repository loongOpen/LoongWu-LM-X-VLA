"""Load real checkpoint weights and run one synthetic action-inference request."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from .validation import validate_actions as validate_action_contract


class SmokeInferenceError(RuntimeError):
    """Raised when synthetic input construction or inference validation fails."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a 龙悟LM-X checkpoint and verify one action-inference request."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Local checkpoint directory or remote model identifier.",
    )
    parser.add_argument(
        "--embodiment-tag",
        required=True,
        help="Embodiment tag present in the checkpoint processor metadata.",
    )
    parser.add_argument("--device", default="cuda:0", help="PyTorch inference device.")
    parser.add_argument(
        "--backbone-path",
        default=None,
        help="Optional local directory or remote identifier for the VLM backbone.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory used while resolving remote resources.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local checkpoint, backbone, and cache files.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional gated-resource token. Defaults to HF_TOKEN.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Floating-point dtype used by inference parameters.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="PyTorch seed for this synthetic smoke run."
    )
    parser.add_argument("--json", action="store_true", help="Print the final result as JSON.")
    return parser


def _positive_dimension(value: Any, *, key: str) -> int:
    try:
        dimension = int(np.asarray(value).item())
    except (TypeError, ValueError) as exc:
        raise SmokeInferenceError(f"Invalid state dimension for {key!r}: {value!r}") from exc
    if dimension <= 0:
        raise SmokeInferenceError(f"State dimension for {key!r} must be positive, got {dimension}")
    return dimension


def _state_sample(norm_params: dict[str, Any], dimension: int) -> np.ndarray:
    """Choose a finite state vector from processor normalization metadata."""

    for stat_name in ("mean", "min", "max"):
        value = norm_params.get(stat_name)
        if value is None:
            continue
        candidate = np.asarray(value, dtype=np.float32).reshape(-1)
        if candidate.size == dimension and np.all(np.isfinite(candidate)):
            return candidate
    return np.zeros(dimension, dtype=np.float32)


def _image_size(processor: Any) -> tuple[int, int]:
    target_size = getattr(processor, "image_target_size", None)
    if isinstance(target_size, (list, tuple)) and len(target_size) == 2:
        height, width = (int(target_size[0]), int(target_size[1]))
        if height > 0 and width > 0:
            return height, width
    shortest_edge = getattr(processor, "shortest_image_edge", None)
    if isinstance(shortest_edge, int) and shortest_edge > 0:
        return shortest_edge, shortest_edge
    return 256, 256


def build_synthetic_observation(policy: Any) -> dict[str, dict[str, Any]]:
    """Construct one strict-valid request from loaded modality and normalization metadata."""

    modality_configs = policy.get_modality_config()
    missing_modalities = {"video", "state", "language", "action"} - set(modality_configs)
    if missing_modalities:
        raise SmokeInferenceError(
            "Checkpoint modality config is missing: " + ", ".join(sorted(missing_modalities))
        )

    embodiment_tag = getattr(policy, "embodiment_tag", None)
    embodiment_value = getattr(embodiment_tag, "value", None)
    if not isinstance(embodiment_value, str) or not embodiment_value:
        raise SmokeInferenceError("The loaded policy did not resolve an embodiment tag")

    state_action_processor = getattr(policy.processor, "state_action_processor", None)
    all_norm_params = getattr(state_action_processor, "norm_params", None)
    try:
        state_norm_params = all_norm_params[embodiment_value]["state"]
    except (KeyError, TypeError) as exc:
        raise SmokeInferenceError(
            f"Processor state normalization metadata is missing for {embodiment_value!r}"
        ) from exc

    observation: dict[str, dict[str, Any]] = {"video": {}, "state": {}, "language": {}}
    image_height, image_width = _image_size(policy.processor)
    video_horizon = len(modality_configs["video"].delta_indices)
    state_horizon = len(modality_configs["state"].delta_indices)
    language_horizon = len(modality_configs["language"].delta_indices)
    if min(video_horizon, state_horizon, language_horizon) <= 0:
        raise SmokeInferenceError("Input modality horizons must all be positive")
    if language_horizon != 1:
        raise SmokeInferenceError(
            f"This inference interface requires language horizon 1, got {language_horizon}"
        )

    for key in modality_configs["video"].modality_keys:
        observation["video"][key] = np.zeros(
            (1, video_horizon, image_height, image_width, 3), dtype=np.uint8
        )

    for key in modality_configs["state"].modality_keys:
        try:
            key_norm_params = state_norm_params[key]
        except (KeyError, TypeError) as exc:
            raise SmokeInferenceError(
                f"Processor state normalization metadata is missing for key {key!r}"
            ) from exc
        dimension = _positive_dimension(key_norm_params.get("dim"), key=key)
        sample = _state_sample(key_norm_params, dimension)
        observation["state"][key] = np.broadcast_to(sample, (1, state_horizon, dimension)).copy()

    instruction = "move to the target"
    for key in modality_configs["language"].modality_keys:
        observation["language"][key] = [[instruction]]

    return observation


def validate_actions(policy: Any, actions: Any) -> dict[str, tuple[int, ...]]:
    """Validate output keys, dtype, batch/time shape, and finite values."""

    tag = policy.embodiment_tag.value
    return validate_action_contract(
        actions,
        policy.get_modality_config()["action"],
        policy.processor.state_action_processor.norm_params[tag],
        batch_size=1,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import torch

    from lm_x import LMXPolicy

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    policy = LMXPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        backbone_path=args.backbone_path,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        token=args.hf_token,
        dtype=dtype,
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    observation = build_synthetic_observation(policy)
    actions, _ = policy.get_action(observation)
    shapes = validate_actions(policy, actions)
    rendered_shapes = ", ".join(f"{key}={shape}" for key, shape in shapes.items())
    if args.json:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "device": args.device,
                    "dtype": args.dtype,
                    "seed": args.seed,
                    "action_shapes": shapes,
                },
                sort_keys=True,
            )
        )
    else:
        print(f"Smoke inference passed on {args.device}: {rendered_shapes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
