"""Flow-transformer building blocks used by the LM-X action and value heads."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.embeddings import (
    SinusoidalPositionalEmbedding,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn


def _spark_requires_math_attention() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (12, 1)


def _attention_kernel_context():
    requested = os.environ.get("LMX_DIT_SDPA_MODE")
    force_math = requested == "math" or (requested is None and _spark_requires_math_attention())
    if not force_math:
        return nullcontext()
    return torch.backends.cuda.sdp_kernel(
        enable_flash=False,
        enable_math=True,
        enable_mem_efficient=False,
        enable_cudnn=False,
    )


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        parameter_dtype = next(self.parameters()).dtype
        projected = self.time_proj(timesteps).to(parameter_dtype)
        return self.timestep_embedder(projected)


class StepConditionedNorm(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()
        self.chunk_dim = chunk_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, embedding_dim * 2)
        self.norm = nn.LayerNorm(
            embedding_dim,
            norm_eps,
            norm_elementwise_affine,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if conditioning is None:
            raise ValueError("step conditioning is required for adaptive normalization")
        scale, shift = self.linear(self.silu(conditioning)).chunk(2, dim=1)
        normalized = self.norm(hidden_states)
        return normalized * (1 + scale[:, None]) + shift[:, None]


class ResidualAttentionUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        del attention_type
        if positional_embeddings and num_positional_embeddings is None:
            raise ValueError(
                "num_positional_embeddings is required when positional_embeddings is enabled"
            )
        self.norm_type = norm_type
        self.pos_embed = (
            SinusoidalPositionalEmbedding(
                dim,
                max_seq_length=num_positional_embeddings,
            )
            if positional_embeddings == "sinusoidal"
            else None
        )
        self.norm1 = (
            StepConditionedNorm(dim)
            if norm_type == "ada_norm"
            else nn.LayerNorm(
                dim,
                elementwise_affine=norm_elementwise_affine,
                eps=norm_eps,
            )
        )
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )
        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.final_dropout = nn.Dropout(dropout) if final_dropout else None

    def _attention_input(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor | None,
    ) -> torch.Tensor:
        normalized = (
            self.norm1(hidden_states, conditioning)
            if self.norm_type == "ada_norm"
            else self.norm1(hidden_states)
        )
        return self.pos_embed(normalized) if self.pos_embed is not None else normalized

    @staticmethod
    def _flatten_extra_axis(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.squeeze(1) if hidden_states.ndim == 4 else hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        query = self._attention_input(hidden_states, temb)
        selected_mask = (
            encoder_attention_mask if encoder_hidden_states is not None else attention_mask
        )
        with _attention_kernel_context():
            attended = self.attn1(
                query,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=selected_mask,
            )
        if self.final_dropout is not None:
            attended = self.final_dropout(attended)
        hidden_states = self._flatten_extra_axis(hidden_states + attended)
        feed_forward = self.ff(self.norm3(hidden_states))
        return self._flatten_extra_axis(hidden_states + feed_forward)


LayerRoute = Callable[
    [int],
    tuple[torch.Tensor | None, torch.Tensor | None],
]


class FlowTransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        norm_type: str = "ada_norm",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        del num_embeds_ada_norm
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.timestep_encoder = DiffusionStepEmbedding(
            embedding_dim=self.inner_dim,
            compute_dtype=compute_dtype,
        )
        blocks = []
        for index in range(num_layers):
            self_attention = interleave_self_attention and index % 2 == 1
            blocks.append(
                ResidualAttentionUnit(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                    cross_attention_dim=None if self_attention else cross_attention_dim,
                )
            )
        self.transformer_blocks = nn.ModuleList(blocks)
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(self.inner_dim, output_dim)

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        route: LayerRoute,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        current = hidden_states.contiguous()
        history = [current]
        for index, block in enumerate(self.transformer_blocks):
            context, context_mask = route(index)
            current = block(
                current,
                encoder_hidden_states=context,
                encoder_attention_mask=context_mask,
                temb=conditioning,
            )
            history.append(current)
        return current, history

    def _output_projection(
        self,
        hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        modulated = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(modulated)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
    ):
        del encoder_attention_mask
        if timestep is None:
            raise ValueError("timestep is required")
        conditioning = self.timestep_encoder(timestep)
        context = encoder_hidden_states.contiguous()

        def route(index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            use_self_attention = self.config.interleave_self_attention and index % 2 == 1
            return (None, None) if use_self_attention else (context, None)

        transformed, history = self._run_layers(hidden_states, conditioning, route)
        output = self._output_projection(transformed, conditioning)
        return (output, history) if return_all_hidden_states else output


class RoutedVisionLanguageFlow(FlowTransformer):
    """Alternate cross-attention between language/state tokens and image tokens."""

    def __init__(self, *args, attend_text_every_n_blocks: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        if attend_text_every_n_blocks < 1:
            raise ValueError("attend_text_every_n_blocks must be positive")
        self.attend_text_every_n_blocks = attend_text_every_n_blocks

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden_states: bool = False,
        image_mask: Optional[torch.Tensor] = None,
        backbone_attention_mask: Optional[torch.Tensor] = None,
    ):
        del encoder_attention_mask
        if timestep is None:
            raise ValueError("timestep is required")
        if image_mask is None or backbone_attention_mask is None:
            raise ValueError("image_mask and backbone_attention_mask are required")
        if not self.config.interleave_self_attention:
            raise ValueError("RoutedVisionLanguageFlow requires interleave_self_attention")

        conditioning = self.timestep_encoder(timestep)
        context = encoder_hidden_states.contiguous()
        image_tokens = image_mask & backbone_attention_mask
        non_image_tokens = (~image_mask) & backbone_attention_mask

        def route(index: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if index % 2 == 1:
                return None, None
            period = 2 * self.attend_text_every_n_blocks
            mask = non_image_tokens if index % period == 0 else image_tokens
            return context, mask

        transformed, history = self._run_layers(hidden_states, conditioning, route)
        output = self._output_projection(transformed, conditioning)
        return (output, history) if return_all_hidden_states else output


class TokenRefiner(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        output_dim: int = 26,
        num_layers: int = 12,
        dropout: float = 0.1,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
        num_embeds_ada_norm: Optional[int] = 1000,
        upcast_attention: bool = False,
        max_num_positional_embeddings: int = 512,
        compute_dtype=torch.float32,
        final_dropout: bool = True,
        positional_embeddings: Optional[str] = "sinusoidal",
        interleave_self_attention=False,
    ):
        super().__init__()
        del output_dim, num_embeds_ada_norm, compute_dtype, interleave_self_attention
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.transformer_blocks = nn.ModuleList(
            [
                ResidualAttentionUnit(
                    self.inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=max_num_positional_embeddings,
                    final_dropout=final_dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_all_hidden_states: bool = False,
    ):
        current = hidden_states.contiguous()
        history = [current]
        for block in self.transformer_blocks:
            current = block(current)
            history.append(current)
        return (current, history) if return_all_hidden_states else current


# Previous public names remain import-compatible while new code uses the semantic names above.
TimestepEncoder = DiffusionStepEmbedding
AdaLayerNorm = StepConditionedNorm
BasicTransformerBlock = ResidualAttentionUnit
DiT = FlowTransformer
AlternateVLDiT = RoutedVisionLanguageFlow
SelfAttentionTransformer = TokenRefiner


__all__ = [
    "DiffusionStepEmbedding",
    "FlowTransformer",
    "ResidualAttentionUnit",
    "RoutedVisionLanguageFlow",
    "StepConditionedNorm",
    "TokenRefiner",
]
