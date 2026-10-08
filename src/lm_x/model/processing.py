import json
import logging
import os
import re
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import albumentations as A
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from PIL import Image
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import cached_file

from lm_x.data_utils import decode_modality_configs
from lm_x.embodiment_tags import EmbodimentTag
from lm_x.interfaces import InferenceProcessor
from lm_x.state_action.processor import StateActionProcessor
from lm_x.types import ModalityConfig

from .image_preprocessing import (
    apply_with_replay,
    build_albumentations_image_transform,
    build_torchvision_image_transform,
)

try:
    from transformers import Qwen3VLProcessor
except ImportError:
    Qwen3VLProcessor = None

# Suppress protobuf deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google.protobuf")

logger = logging.getLogger(__name__)

_GPU_PATCHIFY = os.environ.get("LMX_GPU_PATCHIFY", "1") == "1"


def build_processor(model_name: str, transformers_loading_kwargs: dict) -> Qwen3VLProcessor:
    if Qwen3VLProcessor is None:
        raise ImportError(
            "Qwen3VLProcessor is not available. "
            "Please upgrade transformers: pip install transformers>=4.52.0"
        )
    return Qwen3VLProcessor.from_pretrained(model_name, **transformers_loading_kwargs)


def validate_action_horizons(modality_configs, max_action_horizon: int) -> None:
    """Fail at processor construction if any configured embodiment's action horizon
    (the number of action ``delta_indices``) exceeds ``max_action_horizon``.

    ``max_action_horizon`` is set from the model's ``action_horizon``; without this
    check a horizon/model mismatch only surfaces deep in the first inference call.
    """
    offenders: dict[str, int] = {}
    for tag, config in modality_configs.items():
        action = config.get("action") if isinstance(config, dict) else None
        delta_indices = getattr(action, "delta_indices", None)
        if delta_indices is None:
            continue
        horizon = len(delta_indices)
        if horizon > max_action_horizon:
            offenders[tag] = horizon
    if offenders:
        required = max(offenders.values())
        details = ", ".join(f"{tag}={horizon}" for tag, horizon in sorted(offenders.items()))
        raise ValueError(
            f"Embodiment action horizon exceeds max_action_horizon ({max_action_horizon}): "
            f"{details}. Increase model config action_horizon to >= {required} (or reduce the "
            "embodiment action delta_indices)."
        )


class LMXDataCollator:
    def __init__(
        self,
        model_name: str,
        model_type: str = "qwen",
        transformers_loading_kwargs: dict | None = None,
        processor: Any | None = None,
    ):
        self.processor = (
            processor
            if processor is not None
            else build_processor(model_name, transformers_loading_kwargs or {})
        )
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        self.model_type = model_type
        self.model_name = model_name

    def __call__(self, features: list[Dict[str, Any]]) -> BatchFeature:
        batch = {}
        keys = list(set().union(*(elem.keys() for elem in features)))

        for key in keys:
            values = [elem[key] for elem in features if key in elem]
            if key == "vlm_content":
                # # Handle vlm_content specially - extract text and images
                text_list = []
                image_inputs = []
                for v in values:
                    curr_text_list = [v["text"]]

                    text_list += curr_text_list
                    curr_image_inputs = v["images"]
                    image_inputs += curr_image_inputs

                arrays = [np.asarray(image) for image in image_inputs]
                shapes = {array.shape for array in arrays}
                image_processor = self.processor.image_processor
                patch_size = image_processor.patch_size
                merge_size = image_processor.merge_size
                use_gpu_patchify = (
                    _GPU_PATCHIFY
                    and len(shapes) == 1
                    and arrays
                    and arrays[0].ndim == 3
                    and arrays[0].shape[2] == 3
                    and arrays[0].shape[0] % (patch_size * merge_size) == 0
                    and arrays[0].shape[1] % (patch_size * merge_size) == 0
                )
                if use_gpu_patchify:
                    pixels = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
                    n_images, _, height, width = pixels.shape
                    grid_h, grid_w = height // patch_size, width // patch_size
                    token_count = grid_h * grid_w // merge_size**2
                    image_token = self.processor.image_token
                    placeholder = "<|vla_image_placeholder|>"
                    text_list = [
                        text.replace(image_token, placeholder * token_count).replace(
                            placeholder, image_token
                        )
                        for text in text_list
                    ]
                    tokens = self.processor.tokenizer(text_list, padding=True, return_tensors="pt")
                    batch.update(tokens)
                    batch["pixel_values_raw"] = pixels
                    batch["image_grid_thw"] = torch.tensor(
                        [[1, grid_h, grid_w]] * n_images, dtype=torch.long
                    )
                else:
                    vlm_inputs = self.processor(
                        text=text_list,
                        images=image_inputs,
                        return_tensors="pt",
                        padding=True,
                    )
                    for k, v in vlm_inputs.items():
                        batch[k] = v
            elif key in (
                "pixel_values",
                "image_grid_thw",
                "attention_mask",
                "input_ids",
            ):
                raise Exception("Not implemented")
            else:
                if all(isinstance(value, torch.Tensor) for value in values):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = torch.from_numpy(np.stack(values))
        return BatchFeature(data={"inputs": batch})

    def __str__(self):
        return f"LMXDataCollator(model_name={self.model_name}, model_type={self.model_type})"


class LMXProcessor(InferenceProcessor):
    data_collator_class = LMXDataCollator

    def __init__(
        self,
        modality_configs: dict[str, dict[str, ModalityConfig]],
        statistics: (dict[str, dict[str, dict[str, dict[str, list[float]]]]] | None) = None,
        use_percentiles: bool = False,
        clip_outliers: bool = True,
        image_crop_size: list[int] = None,
        image_target_size: list[int] = None,
        shortest_image_edge: int = 256,
        crop_fraction: float = 0.95,
        formalize_language: bool = True,
        model_name: str = os.getenv("LMX_BACKBONE_MODEL_NAME", "nvidia/Cosmos-Reason2-2B"),
        model_type: str = "qwen",
        max_state_dim: int = 29,
        max_action_dim: int = 29,
        max_action_horizon: int = 50,
        apply_sincos_state_encoding: bool = False,
        use_albumentations: bool = False,
        use_relative_action: bool = False,
        embodiment_id_mapping: dict[str, int] | None = None,
        transformers_loading_kwargs: dict | None = None,
        exclude_state: bool = False,
        # Normalization
        use_mean_std: bool = False,
        letter_box_transform: bool = False,
    ):
        self.modality_configs = decode_modality_configs(modality_configs)

        # Initialize StateActionProcessor for state/action normalization
        self.state_action_processor = StateActionProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            use_percentiles=use_percentiles,
            clip_outliers=clip_outliers,
            apply_sincos_state_encoding=apply_sincos_state_encoding,
            use_relative_action=use_relative_action,
        )

        # Save state action processor settings
        self.use_percentiles = use_percentiles
        self.use_mean_std = use_mean_std
        self.clip_outliers = clip_outliers
        self.apply_sincos_state_encoding = apply_sincos_state_encoding
        self.use_relative_action = use_relative_action

        self.exclude_state = exclude_state
        self.letter_box_transform = letter_box_transform

        # Save VLM settings
        self.formalize_language = formalize_language
        self.model_name = model_name
        self.model_type = model_type

        self.max_state_dim = max_state_dim
        self.max_action_dim = max_action_dim
        self.max_action_horizon = max_action_horizon
        validate_action_horizons(self.modality_configs, self.max_action_horizon)

        # Save image processing settings
        self.image_crop_size = image_crop_size
        self.image_target_size = image_target_size
        transformers_loading_kwargs = transformers_loading_kwargs or {"trust_remote_code": False}
        self.processor = build_processor(model_name, transformers_loading_kwargs)
        # Set padding side to 'left' for Flash Attention compatibility
        self.processor.tokenizer.padding_side = "left"
        if not isinstance(embodiment_id_mapping, dict) or not embodiment_id_mapping:
            raise ValueError(
                "embodiment_id.json is required and must contain the checkpoint's exact "
                "embodiment-to-projector mapping"
            )
        self.embodiment_id_mapping = {
            str(tag): int(projector_id) for tag, projector_id in embodiment_id_mapping.items()
        }
        missing_ids = sorted(set(self.modality_configs) - set(self.embodiment_id_mapping))
        if missing_ids:
            raise ValueError(
                "embodiment_id.json has no projector id for configured embodiments: "
                + ", ".join(missing_ids)
            )
        if any(projector_id < 0 for projector_id in self.embodiment_id_mapping.values()):
            raise ValueError("embodiment_id.json contains a negative projector id")
        self.shortest_image_edge = shortest_image_edge
        self.crop_fraction = crop_fraction

        self.statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = deepcopy(
            statistics or {}
        )

        self.use_albumentations = use_albumentations
        if use_albumentations:
            self.eval_image_transform = build_albumentations_image_transform(
                image_target_size,
                image_crop_size,
                shortest_image_edge,
                crop_fraction,
                letter_box_transform=self.letter_box_transform,
            )
        else:
            self.eval_image_transform = build_torchvision_image_transform(
                image_target_size,
                image_crop_size,
                letter_box_transform=self.letter_box_transform,
            )
        self._collator = self.data_collator_class(
            model_name=model_name,
            model_type=model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
            processor=self.processor,
        )
        self.state_action_processor.eval()

    @property
    def collator(self):
        return self._collator

    def eval(self):
        super().eval()
        self.state_action_processor.eval()
        return self

    def set_statistics(
        self,
        statistics: dict[str, dict[str, dict[str, dict[str, list[float]]]]],
        override: bool = False,
    ) -> None:
        """Set normalization statistics for loaded checkpoint modalities."""
        for key in statistics:
            if key not in self.statistics or override:
                if override:
                    logger.info("Overriding statistics for embodiment %r", key)
                self.statistics[key] = deepcopy(statistics[key])
            else:
                logger.warning(
                    "Statistics for embodiment %r already exist; keeping checkpoint values",
                    key,
                )

        self.state_action_processor.set_statistics(statistics, override=override)

        # Compute action dimensions for convenience
        self.action_dim = {}
        for embodiment_tag in self.state_action_processor.statistics:
            self.action_dim[embodiment_tag] = self.state_action_processor.get_action_dim(
                embodiment_tag
            )

    def decode_action(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
    ):
        """Undo action normalization and convert relative actions to absolute."""
        # Split concatenated action into joint groups
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Use StateActionProcessor to unnormalize and convert to absolute
        return self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=state
        )

    def unapply(
        self,
        action: np.ndarray,
        embodiment_tag: EmbodimentTag,
        state: dict[str, np.ndarray] | None = None,
        prev_action: dict[str, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        """Undo action normalization and convert relative→absolute.

        Args:
            action: Normalized action array of shape (..., action_horizon, action_dim)
            embodiment_tag: Embodiment tag
            state: State observations with "state." prefixed keys (for relative actions)
            prev_action: Unused (kept for API compatibility)

        Returns:
            Dict mapping "action.<key>" to unnormalized (absolute) action arrays.
        """
        out_dict = {}
        start_idx = 0
        joint_groups = self.modality_configs[embodiment_tag.value]["action"].modality_keys
        action_horizon = len(self.modality_configs[embodiment_tag.value]["action"].delta_indices)
        for key in joint_groups:
            joint_dim = self.state_action_processor.norm_params[embodiment_tag.value]["action"][
                key
            ]["dim"].item()
            out_dict[key] = action[..., :action_horizon, start_idx : start_idx + joint_dim]
            start_idx += joint_dim

        # Strip "state." prefix for StateActionProcessor
        stripped_state = None
        if state is not None:
            stripped_state = {k.replace("state.", ""): v for k, v in state.items()}

        result = self.state_action_processor.unapply_action(
            out_dict, embodiment_tag.value, state=stripped_state
        )
        return {f"action.{key}": value for key, value in result.items()}

    def _apply_vlm_processing(self, images: np.ndarray, language: str) -> BatchFeature:
        """
        Args:
            batch:
                video: [T, C, H, W]
        Returns: vlm_content format for collation
        """
        # Convert images to PIL format
        pil_images = [Image.fromarray(np.transpose(v, (1, 2, 0))) for v in images]
        conversation = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": language},
                ],
            }
        ]

        # Apply chat template but don't process yet - let collator handle it
        text = self.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )

        # Return vlm_content format for collation
        return {
            "vlm_content": {
                "text": text,
                "images": pil_images,
                "conversation": conversation,
            }
        }

    def __call__(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Convert one observation step into deterministic model inputs."""

        if len(messages) != 1:
            raise ValueError(f"Expected exactly one message, got {len(messages)}")
        content = messages[0]["content"]
        embodiment_tag = EmbodimentTag.resolve(content.embodiment)
        tag_value = embodiment_tag.value
        if tag_value not in self.modality_configs:
            raise KeyError(f"Checkpoint has no modality configuration for {tag_value!r}")

        state_keys = self.modality_configs[tag_value]["state"].modality_keys
        if self.exclude_state or getattr(
            self.modality_configs[tag_value]["state"], "exclude_state", False
        ):
            normalized_states = torch.cat(
                [torch.from_numpy(np.zeros_like(content.states[key])) for key in state_keys],
                dim=-1,
            )
        else:
            normalized = self.state_action_processor.apply_state(
                state=content.states,
                embodiment_tag=tag_value,
            )
            normalized_states = torch.cat(
                [torch.from_numpy(normalized[key]) for key in state_keys],
                dim=-1,
            )
        if normalized_states.shape[-1] > self.max_state_dim:
            raise ValueError(
                f"State dimension {normalized_states.shape[-1]} exceeds max_state_dim "
                f"{self.max_state_dim}"
            )
        normalized_states = torch.cat(
            [
                normalized_states,
                torch.zeros(
                    (
                        *normalized_states.shape[:-1],
                        self.max_state_dim - normalized_states.shape[-1],
                    ),
                    dtype=normalized_states.dtype,
                ),
            ],
            dim=-1,
        )

        language = content.text or ""
        if self.formalize_language:
            language = re.sub(r"[^\w\s]", "", language.lower())
        image_keys = self.modality_configs[tag_value]["video"].modality_keys
        vlm_inputs = self._get_vlm_inputs(
            image_keys=image_keys,
            images=content.images,
            masks=content.masks,
            image_transform=self.eval_image_transform,
            language=language,
        )
        return {
            "state": normalized_states.to(torch.get_default_dtype()),
            "embodiment_id": self.embodiment_id_mapping[tag_value],
            **vlm_inputs,
        }

    def _get_vlm_inputs(
        self,
        image_keys: list[str],
        images: list[Image.Image],
        masks: dict[str, list[np.ndarray]] | None,
        image_transform: transforms.Compose | A.Compose,
        language: str,
    ):
        temporal_stacked_images = {}

        if self.use_albumentations:
            # Use albumentations transforms
            replay = None
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                if masks is not None:
                    assert view in masks, f"{view} not in masks"
                view_masks = masks.get(view) if masks else None
                view_images = images[view]

                # Apply transforms with replay for consistency
                transformed_images, replay = apply_with_replay(
                    image_transform, view_images, view_masks, replay
                )
                temporal_stacked_images[view] = torch.stack(transformed_images)  # (T, C, H, W)
        else:
            if masks is not None:
                raise ValueError(
                    "Mask transforms require albumentations. Set use_albumentations_transforms=True."
                )
            # Use torchvision transforms
            for view in image_keys:
                assert view in images, f"{view} not in {images}"
                temporal_stacked_images[view] = torch.stack(
                    [image_transform(img) for img in images[view]]
                )  # (T, C, H, W)

        for k, v in temporal_stacked_images.items():
            assert isinstance(k, str), f"{k} is not a string"
            assert isinstance(v, torch.Tensor), f"{v} is not a torch tensor"
            assert v.ndim == 4, f"{v} is not a 4D tensor"
            assert v.dtype == torch.uint8, f"{v} is not a uint8 tensor"
            assert v.shape[1] == 3, f"{v} is not a 3 channel tensor"

        stacked_images = (
            torch.stack([temporal_stacked_images[view] for view in image_keys], dim=1)
            .flatten(0, 1)
            .numpy()
        )  # (T*V, C, H, W), processor expects numpy array

        vlm_inputs = self._apply_vlm_processing(stacked_images, language)
        return vlm_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        transformers_loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": False}
        )
        hub_keys = (
            "_commit_hash",
            "cache_dir",
            "force_download",
            "local_files_only",
            "proxies",
            "revision",
            "subfolder",
            "token",
        )
        hub_kwargs = {key: kwargs.pop(key) for key in hub_keys if key in kwargs}
        use_auth_token = kwargs.pop("use_auth_token", None)
        if "token" not in hub_kwargs and use_auth_token is not None:
            hub_kwargs["token"] = use_auth_token
        reference = str(pretrained_model_name_or_path)
        local_path = Path(pretrained_model_name_or_path).expanduser()
        is_local = local_path.is_dir()
        config_file = local_path / "processor_config.json"
        statistics_file = local_path / "statistics.json"
        embodiment_id_file = local_path / "embodiment_id.json"
        if not is_local:
            config_file = Path(cached_file(reference, "processor_config.json", **hub_kwargs))
            statistics_file = Path(cached_file(reference, "statistics.json", **hub_kwargs))
            embodiment_id_file = Path(cached_file(reference, "embodiment_id.json", **hub_kwargs))

        with open(config_file, "r") as f:
            config = json.load(f)
        with open(statistics_file, "r") as f:
            statistics = json.load(f)
        if not embodiment_id_file.exists():
            raise FileNotFoundError(f"Missing checkpoint projector mapping: {embodiment_id_file}")
        with open(embodiment_id_file, "r") as f:
            embodiment_id_mapping = json.load(f)
        processor_kwargs = dict(config["processor_kwargs"])
        for unused_key in (
            "use_full_vlm_backbone",
            "random_rotation_angle",
            "color_jitter_params",
            "extra_augmentation_config",
            "state_dropout_prob",
        ):
            processor_kwargs.pop(unused_key, None)
        processor_kwargs["statistics"] = statistics
        processor_kwargs["embodiment_id_mapping"] = embodiment_id_mapping

        processor_kwargs.setdefault(
            "model_name", os.getenv("LMX_BACKBONE_MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
        )
        processor_kwargs.setdefault("model_type", "qwen")
        processor_kwargs.setdefault("clip_outliers", True)

        # Directly override other processor kwargs
        if kwargs:
            modality_configs = kwargs.pop("modality_configs", {})
            for embodiment_tag, modality_config in modality_configs.items():
                processor_kwargs["modality_configs"][embodiment_tag] = modality_config
            override_keys = [
                "use_relative_action",
                "exclude_state",
                "use_mean_std",
                "model_name",
                "model_type",
                "max_action_horizon",
                "max_state_dim",
                "max_action_dim",
            ]
            for key in override_keys:
                if key in kwargs:
                    override = kwargs.pop(key)
                    if override is not None:
                        processor_kwargs[key] = override
        return cls(**processor_kwargs, transformers_loading_kwargs=transformers_loading_kwargs)
