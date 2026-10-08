"""End-to-end 龙悟LM-X checkpoint loading and action inference."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lm_x.checkpoint import validate_checkpoint_manifest, validate_model_loading_info
from lm_x.config import LMXConfig
from lm_x.embodiment_tags import EmbodimentTag
from lm_x.interfaces import InferenceProcessor
from lm_x.model.modeling import LMXModel
from lm_x.model.processing import LMXProcessor
from lm_x.types import LMXStepData, MessageType, ModalityConfig

from .observation import build_io_spec
from .policy_base import BasePolicy
from .validation import validate_actions, validate_observation

logger = logging.getLogger(__name__)


def _sim_language_batch_to_sequence(value: Any) -> Any:
    """Normalize sim language batches while preserving validation semantics."""
    if isinstance(value, np.ndarray):
        return value.reshape(-1).tolist()
    if isinstance(value, str):
        return [value]
    return value


class LMXPolicy(BasePolicy):
    """Load one checkpoint and expose its batched action-inference interface.

    This policy handles the end-to-end inference pipeline:
    1. Validates input observations
    2. Processes observations with the pretrained LM-X processor
    3. Runs model inference
    4. Decodes and returns actions

    The policy expects observations with specific modalities (video, state, language)
    and returns actions in the format defined by the model's modality configuration.
    """

    def __init__(
        self,
        embodiment_tag: EmbodimentTag | str | None,
        model_path: str | Path,
        *,
        device: int | str,
        strict: bool = True,
        backbone_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        token: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Load model, backbone and processor resources for inference.

        Args:
            embodiment_tag: Optional default embodiment tag. If omitted, every request
                must provide ``options["embodiment_tag"]``.
            model_path: Local checkpoint directory or Hub model identifier.
            device: Device to run the model on (e.g., 'cuda:0', 0, 'cpu')
            strict: Whether to enforce strict input validation (default: True)
            backbone_path: Optional local path or Hub identifier for the VLM backbone.
                This overrides ``config.model_name`` and is never ignored.
            cache_dir: Optional Hugging Face cache directory.
            local_files_only: Refuse Hub downloads when true.
            token: Optional Hugging Face access token for gated resources.
            dtype: Floating-point dtype used for inference parameters.
        """
        super().__init__(strict=strict)
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        validated_model_ref = validate_checkpoint_manifest(model_path)
        local_model_dir = validated_model_ref if isinstance(validated_model_ref, Path) else None
        model_ref = str(validated_model_ref)

        hub_kwargs: dict[str, Any] = {"local_files_only": local_files_only}
        if cache_dir is not None:
            hub_kwargs["cache_dir"] = str(cache_dir)
        if token is not None:
            hub_kwargs["token"] = token

        config = LMXConfig.from_pretrained(model_ref, **hub_kwargs)
        if backbone_path is not None:
            candidate = Path(backbone_path).expanduser()
            config.model_name = (
                str(candidate.resolve()) if candidate.is_dir() else str(backbone_path)
            )
        backbone_ref = str(config.model_name)
        backbone_is_local = Path(backbone_ref).expanduser().is_dir()
        backbone_loading_kwargs = dict(hub_kwargs)
        backbone_loading_kwargs["local_files_only"] = local_files_only or backbone_is_local
        backbone_loading_kwargs["trust_remote_code"] = False

        model, loading_info = LMXModel.from_pretrained(
            model_ref,
            config=config,
            transformers_loading_kwargs=backbone_loading_kwargs,
            output_loading_info=True,
            **hub_kwargs,
        )
        validate_model_loading_info(
            loading_info,
            model_reference=model_ref,
            select_layer=int(config.select_layer),
        )
        model.eval()
        model.requires_grad_(False)
        model.to(device=device, dtype=dtype)
        self.model = model
        logger.info("Loaded inference checkpoint from %s", model_ref)

        processor_ref: str | Path = model_ref
        if local_model_dir is not None:
            nested_processor = local_model_dir / "processor"
            if not (local_model_dir / "processor_config.json").is_file():
                processor_ref = nested_processor
        self.processor: InferenceProcessor = LMXProcessor.from_pretrained(
            processor_ref,
            model_name=backbone_ref,
            max_state_dim=getattr(config, "max_state_dim", None),
            max_action_dim=getattr(config, "max_action_dim", None),
            max_action_horizon=getattr(config, "action_horizon", None),
            transformers_loading_kwargs=backbone_loading_kwargs,
            **hub_kwargs,
        )
        self.processor.eval()

        # Store all configurations so the embodiment can be selected per request.
        self.all_modality_configs = self.processor.get_modality_configs()
        self.collate_fn = self.processor.collator
        self.model.collator = self.collate_fn
        self.embodiment_tag = (
            EmbodimentTag.resolve(embodiment_tag)
            if isinstance(embodiment_tag, str)
            else embodiment_tag
        )
        self.modality_configs: dict[str, ModalityConfig] = {}
        self.language_key: str | None = None
        if self.embodiment_tag is not None:
            self.embodiment_tag = self._resolve_embodiment_tag(self.embodiment_tag)
            self.modality_configs = self._get_modality_config(self.embodiment_tag)
            self.language_key = self._get_language_key(self.modality_configs)

    def _supported_embodiments_str(self) -> str:
        supported_lines = []
        for tag_value in sorted(self.all_modality_configs):
            enum_name = EmbodimentTag.reverse_lookup(tag_value)
            if enum_name != tag_value:
                supported_lines.append(f"  {enum_name:30s} (--embodiment-tag {enum_name})")
            else:
                supported_lines.append(f"  {tag_value:30s} (internal, no public enum)")
        return "\n".join(supported_lines)

    def _resolve_embodiment_tag(self, embodiment_tag: EmbodimentTag | str | None) -> EmbodimentTag:
        if embodiment_tag is None:
            embodiment_tag = self.embodiment_tag
        if embodiment_tag is None:
            raise ValueError(
                "This server has no default embodiment tag. The client must provide "
                "options['embodiment_tag'] with each request."
            )
        if isinstance(embodiment_tag, str):
            embodiment_tag = EmbodimentTag.resolve(embodiment_tag)
        if embodiment_tag.value not in self.all_modality_configs:
            raise ValueError(
                f"Embodiment tag '{embodiment_tag.name}' "
                f"(value='{embodiment_tag.value}') is not supported "
                f"by this checkpoint.\n\n"
                f"Supported tags in this checkpoint:\n{self._supported_embodiments_str()}"
            )
        return embodiment_tag

    def _resolve_request_embodiment(self, options: dict[str, Any] | None) -> EmbodimentTag:
        if options is not None and not isinstance(options, dict):
            raise TypeError("options must be a dictionary or None")
        return self._resolve_embodiment_tag((options or {}).get("embodiment_tag"))

    def _get_modality_config(self, embodiment_tag: EmbodimentTag) -> dict[str, ModalityConfig]:
        return {
            k: v
            for k, v in self.all_modality_configs[embodiment_tag.value].items()
            if k != "rl_info"
        }

    @staticmethod
    def _get_language_key(modality_configs: dict[str, ModalityConfig]) -> str:
        # Extract and validate language configuration
        # Checkpoints may expose several equivalent language keys; inference uses the first.
        language_keys = modality_configs["language"].modality_keys
        language_delta_indices = modality_configs["language"].delta_indices
        if not language_keys or len(language_delta_indices) != 1:
            raise ValueError(
                "At least one language key and exactly one language delta index are required"
            )
        return language_keys[0]

    def _unbatch_observation(self, value: dict[str, Any]) -> list[dict[str, Any]]:
        """Unbatch a batched observation into a list of single observations.

        Args:
            value: Batched observation with shape (B, ...) for each modality

        Returns:
            List of B observations, each with the batch dimension removed
        """
        unbatched_obs = []
        # Infer batch size from the first video key
        batch_size = value["video"][list(value["video"].keys())[0]].shape[0]

        # Split each modality along the batch dimension
        for i in range(batch_size):
            unbatched_value = {
                "video": {k: v[i] for k, v in value["video"].items()},
                "state": {k: v[i] for k, v in value["state"].items()},
                "language": {k: v[i] for k, v in value["language"].items()},
            }
            unbatched_obs.append(unbatched_value)
        return unbatched_obs

    def _to_vla_step_data(
        self,
        observation: dict[str, Any],
        embodiment_tag: EmbodimentTag,
        language_key: str,
    ) -> LMXStepData:
        """Convert a single observation into a LMXStepData object for processing.

        Args:
            observation: Single observation dict with video, state, and language

        Returns:
            LMXStepData object ready for processor input
        """
        return LMXStepData(
            images=observation["video"],
            states=observation["state"],
            text=observation["language"][language_key][0],
            embodiment=embodiment_tag,
        )

    def _norm_params(self, embodiment_tag: EmbodimentTag) -> dict:
        try:
            return self.processor.state_action_processor.norm_params[embodiment_tag.value]
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                f"Missing normalization metadata for {embodiment_tag.value!r}"
            ) from exc

    def check_observation(
        self,
        observation: dict[str, Any],
        embodiment_tag: EmbodimentTag | str | None = None,
    ) -> None:
        """Validate input structure, batch, dimensions and finite states before preprocessing."""
        tag = self._resolve_embodiment_tag(embodiment_tag)
        validate_observation(observation, self._get_modality_config(tag), self._norm_params(tag))

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        embodiment_tag = self._resolve_request_embodiment(options)
        return self._get_action_for_embodiment(observation, embodiment_tag, options)

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Compute an action using the embodiment selected for this request."""
        embodiment_tag = self._resolve_request_embodiment(options)
        if self.strict:
            self.check_observation(observation, embodiment_tag)
        action, info = self._get_action_for_embodiment(observation, embodiment_tag, options)
        if self.strict:
            video_key = self._get_modality_config(embodiment_tag)["video"].modality_keys[0]
            self.check_action(
                action, embodiment_tag, batch_size=observation["video"][video_key].shape[0]
            )
        return action, info

    def _get_action_for_embodiment(
        self,
        observation: dict[str, Any],
        embodiment_tag: EmbodimentTag,
        options: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Internal method to compute actions from observations.

        Pipeline:
        1. Unbatch observations into individual samples
        2. Convert each to LMXStepData and process
        3. Collate into model input batch
        4. Run model inference
        5. Decode and unnormalize actions

        Args:
            observation: Batched observation dictionary
            options: Optional model parameters. ``embodiment_tag`` is consumed by
                the policy; all other options are passed through to the model.

        Returns:
            Tuple of (actions_dict, info_dict)
        """
        modality_configs = self._get_modality_config(embodiment_tag)
        language_key = self._get_language_key(modality_configs)

        # Step 1: Split batched observation into individual observations
        unbatched_observations = self._unbatch_observation(observation)
        processed_inputs = []

        # Step 2: Process each observation through the LM-X processor
        states = []
        for obs in unbatched_observations:
            vla_step_data = self._to_vla_step_data(obs, embodiment_tag, language_key)
            states.append(vla_step_data.states)  # dict[str, np.ndarray[np.float32, (T, D)]]
            messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
            processed = self.processor(messages)
            processed_inputs.append(processed)

        # Step 3: Collate processed inputs into a single batch for model
        collated_inputs = self.collate_fn(processed_inputs)

        # Step 4: Run model inference to predict actions
        model_options = dict(options or {})
        model_options.pop("embodiment_tag", None)
        with torch.inference_mode():
            if model_options:
                model_pred = self.model.get_action(**collated_inputs, options=model_options)
            else:
                model_pred = self.model.get_action(**collated_inputs)
        normalized_action = model_pred["action_pred"].float()
        info = {}
        if "value_pred" in model_pred and model_pred["value_pred"] is not None:
            info["value_pred"] = model_pred["value_pred"].detach().cpu().numpy()
        if "action_uncertainty" in model_pred and model_pred["action_uncertainty"] is not None:
            info["action_uncertainty"] = model_pred["action_uncertainty"].detach().cpu().numpy()
        if "uncertainty_score" in model_pred and model_pred["uncertainty_score"] is not None:
            info["uncertainty_score"] = model_pred["uncertainty_score"].detach().cpu().numpy()

        # Step 5: Decode actions from normalized space back to physical units
        batched_states = {}
        for k in modality_configs["state"].modality_keys:
            batched_states[k] = np.stack([s[k] for s in states], axis=0)  # (B, T, D)
        unnormalized_action = self.processor.decode_action(
            normalized_action.cpu().numpy(), embodiment_tag, batched_states
        )

        # Cast all actions to float32 for consistency
        casted_action = {
            key: value.astype(np.float32) for key, value in unnormalized_action.items()
        }
        return casted_action, info

    def check_action(
        self,
        action: dict[str, Any],
        embodiment_tag: EmbodimentTag | str | None = None,
        *,
        batch_size: int | None = None,
    ) -> None:
        """Validate complete action keys, shape, float32 dtype and finite values."""
        tag = self._resolve_embodiment_tag(embodiment_tag)
        validate_actions(
            action,
            self._get_modality_config(tag)["action"],
            self._norm_params(tag),
            batch_size=batch_size,
        )

    def get_io_spec(self, embodiment_tag: EmbodimentTag | str | None = None) -> dict:
        """Return the same JSON-compatible input/output contract locally and over ZeroMQ."""
        tag = self._resolve_embodiment_tag(embodiment_tag)
        image_size = getattr(self.processor, "image_target_size", None)
        if not image_size:
            edge = getattr(self.processor, "shortest_image_edge", 256)
            image_size = (edge, edge)
        return build_io_spec(
            self._get_modality_config(tag),
            self._norm_params(tag),
            embodiment_tag=tag.value,
            image_size=image_size,
        )

    def get_modality_config(
        self, embodiment_tag: EmbodimentTag | str | None = None
    ) -> dict[str, ModalityConfig]:
        return self._get_modality_config(self._resolve_embodiment_tag(embodiment_tag))

    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reset the policy to its initial state.

        Args:
            options: Dictionary containing the options for the reset

        Returns:
            Dictionary containing the info after resetting the policy
        """
        self._resolve_request_embodiment(options)
        return {}
