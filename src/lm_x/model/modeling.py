import logging
import os
from typing import Any, Tuple

import torch
import tree
from torch import nn
from transformers import PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature

from lm_x.config import LMXConfig
from lm_x.model.modules.dit import FlowTransformer, RoutedVisionLanguageFlow, TokenRefiner
from lm_x.model.modules.embodiment_conditioned_mlp import EmbodimentMLP, TimedActionEncoder

logger = logging.getLogger(__name__)

_GPU_PATCHIFY = os.environ.get("LMX_GPU_PATCHIFY", "1") == "1"


class LMXActionHead(nn.Module):
    """
    龙悟LM-X action head with UA-Flow uncertainty.

    Flow Matching:
        x_t = (1 - t) * noise + t * action
        target_velocity = action - noise

    UA-Flow:
        predict velocity mean + log variance

    """

    LOG_VAR_MIN = -10.0
    LOG_VAR_MAX = 5.0

    def __init__(self, config: LMXConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim
        # ============================================================
        # Flow transformer
        # ============================================================
        if config.use_alternate_vl_dit:
            self.model = RoutedVisionLanguageFlow(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using routed vision-language flow transformer")
        else:
            self.model = FlowTransformer(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using standard flow transformer")

        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.use_uncertainty = config.use_uncertainty
        # ============================================================
        # Encoder / Decoder
        # ============================================================
        self.state_encoder = EmbodimentMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )

        self.action_encoder = TimedActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )

        # velocity mean head
        self.action_decoder = EmbodimentMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        # UA-Flow uncertainty head
        self.noise_decoder = None
        if self.use_uncertainty:
            self.noise_decoder = EmbodimentMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
        # ============================================================
        # VLM feature processing
        # ============================================================
        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(
            config,
            "vl_self_attention_cfg",
            None,
        )

        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = TokenRefiner(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(
                config.max_seq_len,
                self.input_embedding_dim,
            )
            nn.init.normal_(
                self.position_embedding.weight,
                mean=0.0,
                std=0.02,
            )

        self.num_timestep_buckets = config.num_timestep_buckets

        self.requires_grad_(False)

    # ================================================================
    # Stable log variance
    # ================================================================

    def _bound_log_var(
        self,
        raw_log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Smoothly map raw output to the configured finite interval."""

        mid = 0.5 * (self.LOG_VAR_MIN + self.LOG_VAR_MAX)

        half_range = 0.5 * (self.LOG_VAR_MAX - self.LOG_VAR_MIN)

        return mid + half_range * torch.tanh(raw_log_var.float() / half_range)

    def process_backbone_output(
        self,
        backbone_output: BatchFeature,
    ) -> BatchFeature:

        backbone_features = backbone_output["backbone_features"]

        backbone_features = self.vlln(backbone_features)

        backbone_features = self.vl_self_attention(backbone_features)

        backbone_output["backbone_features"] = backbone_features

        return backbone_output

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        vl_embeds = backbone_output.backbone_features

        embodiment_id = action_input.embodiment_id

        state = action_input.state

        assert state.shape[1] == self.config.state_history_length, (
            "current_T != state_history_length"
        )

        state = state.view(
            state.shape[0],
            1,
            -1,
        )

        state_features = self.state_encoder(
            state,
            embodiment_id,
        )

        return BatchFeature(
            data={
                "backbone_features": vl_embeds,
                "state_features": state_features,
            }
        )

    # ================================================================
    # Inference
    # ================================================================

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        value_output: BatchFeature | None,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> BatchFeature:

        vl_embeds = backbone_features

        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device

        actions = torch.randn(
            size=(
                batch_size,
                self.config.action_horizon,
                self.action_dim,
            ),
            dtype=vl_embeds.dtype,
            device=device,
        )

        action_variance = (
            torch.ones(actions.shape, dtype=torch.float32, device=device)
            if self.use_uncertainty
            else None
        )

        dt = 1.0 / self.num_inference_timesteps

        # ------------------------------------------------------------
        # Flow integration
        # ------------------------------------------------------------
        pos_embs = None
        if self.config.add_pos_embed:
            pos_ids = torch.arange(self.action_horizon, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, dtype=torch.long, device=device
            )

            def compute_mean_action(actions):
                action_features = self.action_encoder(
                    actions,
                    timesteps_tensor,
                    embodiment_id,
                )

                if pos_embs is not None:
                    action_features = action_features + pos_embs

                # Order: state, optional value_hidden, action.
                sa_parts = [state_features]
                if value_output is not None:
                    sa_parts.append(value_output["value_hidden"])
                sa_parts.append(action_features)
                sa_embs = torch.cat(sa_parts, dim=1)

                if self.config.use_alternate_vl_dit:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                        image_mask=(backbone_output.image_mask),
                        backbone_attention_mask=(backbone_output.backbone_attention_mask),
                    )

                else:
                    model_output = self.model(
                        hidden_states=sa_embs,
                        encoder_hidden_states=vl_embeds,
                        timestep=timesteps_tensor,
                    )

                pred = self.action_decoder(
                    model_output,
                    embodiment_id,
                )

                pred_velocity = pred[
                    :,
                    -self.action_horizon :,
                ]

                # Mean flow
                actions_next = actions + dt * pred_velocity
                return actions_next, model_output

            actions_next, model_output = compute_mean_action(actions)
            if self.use_uncertainty:
                assert self.noise_decoder is not None
                assert action_variance is not None
                raw_log_var = self.noise_decoder(
                    model_output,
                    embodiment_id,
                )[
                    :,
                    -self.action_horizon :,
                ]

                log_var = self._bound_log_var(raw_log_var)

                velocity_var = torch.exp(log_var)
                action_variance = action_variance + (velocity_var * dt).float().pow(2)
            actions = actions_next

        value_pred = None

        if value_output is not None:
            value_pred = value_output.get(
                "value",
                None,
            )

        output = {
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
            "value_pred": value_pred,
        }
        if action_variance is not None:
            output["action_uncertainty"] = action_variance
            output["uncertainty_score"] = action_variance.mean(dim=(1, 2))
        return BatchFeature(data=output)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        value_output: BatchFeature | None,
        action_input: BatchFeature,
    ) -> BatchFeature:

        features = self._encode_features(
            backbone_output,
            action_input,
        )

        return self.get_action_with_features(
            backbone_features=(features.backbone_features),
            state_features=(features.state_features),
            value_output=value_output,
            embodiment_id=(action_input.embodiment_id),
            backbone_output=backbone_output,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(
        self,
        batch: dict,
    ) -> BatchFeature:

        return BatchFeature(data=batch)


def get_backbone_cls(config: LMXConfig):
    if config.backbone_model_type == "qwen":
        # Lazy import: VisionLanguageBackbone needs a newer transformers than the rest of LM-X.
        from lm_x.model.modules.qwen3_backbone import VisionLanguageBackbone

        return VisionLanguageBackbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class LMXModel(PreTrainedModel):
    """龙悟LM-X model with a Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = LMXConfig

    def __init__(
        self,
        config: LMXConfig,
        transformers_loading_kwargs: dict | None = None,
    ):
        """
        Initialize LMXModel model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Keyword arguments forwarded when loading the
                separately stored VLM backbone.
        """
        super().__init__(config)
        self.config = config
        transformers_loading_kwargs = transformers_loading_kwargs or {
            "trust_remote_code": False,
            "local_files_only": True,
        }

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            select_layer=config.select_layer,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize optional value and action heads.
        self.value_model = None
        self.use_value = config.use_value
        if config.use_value:
            from .modules.value import LMXValue

            self.value_model = LMXValue(config)

        # Initialize action head
        self.action_head = LMXActionHead(config)
        self._collator = None
        self._transformers_loading_kwargs = transformers_loading_kwargs

    @property
    def collator(self):
        # Policy supplies its existing collator. Direct model users initialize it on demand.
        if self._collator is None:
            from .processing import LMXDataCollator

            self._collator = LMXDataCollator(
                model_name=self.config.model_name,
                model_type=self.config.backbone_model_type,
                transformers_loading_kwargs=self._transformers_loading_kwargs,
            )
        return self._collator

    @collator.setter
    def collator(self, value):
        self._collator = value

    def _gpu_patchify(self, pixel_values_raw: torch.Tensor) -> torch.Tensor:
        """Normalize and patchify already-resized images on the model GPU.

        The collator only selects this path when every image has one common, aligned shape;
        otherwise the standard Transformers image processor remains authoritative.
        """
        image_processor = self.collator.processor.image_processor
        patch_size = image_processor.patch_size
        temporal_patch_size = image_processor.temporal_patch_size
        merge_size = image_processor.merge_size
        reciprocal_rescale = 1.0 / image_processor.rescale_factor
        mean = torch.as_tensor(
            image_processor.image_mean, device=self.device, dtype=torch.float32
        ).view(1, -1, 1, 1)
        std = torch.as_tensor(
            image_processor.image_std, device=self.device, dtype=torch.float32
        ).view(1, -1, 1, 1)
        mean = mean * reciprocal_rescale
        std = std * reciprocal_rescale
        pixels = pixel_values_raw.to(device=self.device, dtype=torch.float32, non_blocking=True)
        pixels = (pixels - mean) / std
        pixels = pixels.unsqueeze(1)
        if pixels.shape[1] % temporal_patch_size:
            repeat = temporal_patch_size - pixels.shape[1] % temporal_patch_size
            pixels = torch.cat((pixels, pixels[:, -1:].repeat(1, repeat, 1, 1, 1)), dim=1)

        n_images, _, channels, height, width = pixels.shape
        grid_h, grid_w = height // patch_size, width // patch_size
        grid_t = pixels.shape[1] // temporal_patch_size
        pixels = pixels.view(
            n_images,
            grid_t,
            temporal_patch_size,
            channels,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        pixels = pixels.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return pixels.reshape(
            n_images * grid_t * grid_h * grid_w,
            channels * temporal_patch_size * patch_size * patch_size,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None:
            # Host-known shape metadata lets the backbone avoid .item()/.tolist() CUDA drains.
            self.backbone._vla_grid_thw_cpu = grid_thw.detach().to("cpu").clone()

        if _GPU_PATCHIFY and "pixel_values_raw" in inputs:
            inputs["pixel_values"] = self._gpu_patchify(inputs.pop("pixel_values_raw"))

        # Convert processor payloads into the tensor inputs expected by the backbone.
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(backbone_inputs)
        value_outputs = None
        if self.use_value:
            # 为 Value 分支建立一个独立的 backbone output
            # 只切断 backbone_features 到 VLM 的梯度
            value_backbone_outputs = BatchFeature(
                data={
                    **backbone_outputs,
                    "backbone_features": backbone_outputs["backbone_features"].detach(),
                }
            )
            value_outputs = self.value_model.get_action(
                value_backbone_outputs,
                action_inputs,
            )
            value_outputs["value_hidden"] = value_outputs["value_hidden"].detach()
            if "value" in value_outputs:
                value_outputs["value"] = value_outputs["value"].detach()
        del options
        action_outputs = self.action_head.get_action(
            backbone_outputs,
            value_outputs,
            action_inputs,
        )

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
