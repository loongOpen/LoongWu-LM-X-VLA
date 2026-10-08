from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from lm_x.config import LMXConfig
from lm_x.model.modules.dit import TokenRefiner


class VLStableAttentionValueHead(nn.Module):
    """
    Stable VL-only value head.

    不使用 state / embodiment_id / EmbodimentMLP。
    只基于 VLM backbone 输出 token 做 masked attention scoring。

    Input:
        backbone_features: [B, T, backbone_dim]
        backbone_mask:     [B, T], True/1 表示有效 token

    Output:
        logits:       [B, value_bin_num]
        value_hidden: [B, 1, hidden_dim]
    """

    def __init__(
        self,
        backbone_dim: int,
        hidden_dim: int,
        value_bin_num: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.backbone_dim = backbone_dim
        self.hidden_dim = hidden_dim
        self.value_bin_num = value_bin_num

        self.token_proj = nn.Sequential(
            nn.LayerNorm(backbone_dim),
            nn.Linear(backbone_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, value_bin_num),
        )

    def _masked_attention_pool(
        self,
        x: torch.Tensor,
        backbone_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        x: [B, T, H]
        backbone_mask: [B, T], True 表示有效 token
        return: [B, H]
        """
        attn_logits = self.score_head(x).squeeze(-1)  # [B, T]

        attn_logits = torch.nan_to_num(
            attn_logits,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        if backbone_mask is not None:
            valid_mask = backbone_mask.bool()

            # 防止某些样本所有 token 都被 mask，softmax 后变 NaN
            all_invalid = ~valid_mask.any(dim=1)
            if all_invalid.any():
                valid_mask = valid_mask.clone()
                valid_mask[all_invalid] = True

            attn_logits = attn_logits.masked_fill(~valid_mask, -1e4)

        # 稳定 softmax
        attn_logits = attn_logits - attn_logits.max(dim=-1, keepdim=True).values
        attn_logits = attn_logits.clamp(min=-60.0, max=60.0)

        attn = torch.softmax(attn_logits.float(), dim=-1).to(dtype=x.dtype)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)

        attn_sum = attn.sum(dim=-1, keepdim=True)
        attn = attn / attn_sum.clamp_min(1e-6)

        # 极端兜底：如果某行 attention 全坏，改成均匀分布
        bad_attn = attn_sum <= 1e-6
        if bad_attn.any():
            uniform_attn = torch.full_like(attn, 1.0 / attn.shape[-1])
            attn = torch.where(bad_attn, uniform_attn, attn)

        hidden = (x * attn.unsqueeze(-1)).sum(dim=1)  # [B, H]
        return hidden

    def forward(
        self,
        backbone_features: torch.Tensor,
        backbone_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # backbone_features: [B, T, D]

        x = torch.nan_to_num(
            backbone_features,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )
        x = x.clamp(min=-1e4, max=1e4)

        x = self.token_proj(x)  # [B, T, H]
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        hidden = self._masked_attention_pool(x, backbone_mask)  # [B, H]

        hidden = self.out_norm(hidden)
        hidden = torch.nan_to_num(
            hidden,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        logits = self.head(hidden)  # [B, value_bin_num]
        logits = torch.nan_to_num(
            logits,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        value_hidden = hidden.unsqueeze(1)  # [B, 1, H]
        return logits, value_hidden


class LMXValue(nn.Module):
    """Value and latent-conditioning head used during action inference."""

    def __init__(self, config: LMXConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.value_bin_num = config.value_bin_num
        self.value_low = config.value_low
        self.value_high = config.value_high

        self.bin_width = (self.value_high - self.value_low) / self.value_bin_num

        bin_centers = torch.linspace(
            self.value_low + self.bin_width / 2,
            self.value_high - self.bin_width / 2,
            self.value_bin_num,
        )
        self.register_buffer("bin_centers", bin_centers, persistent=False)

        self.value_head = VLStableAttentionValueHead(
            backbone_dim=config.backbone_embedding_dim,
            hidden_dim=self.input_embedding_dim,
            value_bin_num=self.value_bin_num,
            dropout=0.05,
        )

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = TokenRefiner(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        self.apply(self._init_weights)

        self.requires_grad_(False)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            module._reset_parameters()

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]

        backbone_features = torch.nan_to_num(
            backbone_features,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)

        backbone_features = torch.nan_to_num(
            backbone_features,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        return BatchFeature(
            data={
                **backbone_output,
                "backbone_features": backbone_features,
            }
        )

    def _compute_value_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.nan_to_num(
            logits.float(),
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        probs = F.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        bin_centers = self.bin_centers.to(
            device=probs.device,
            dtype=probs.dtype,
        )

        value_pred = (probs * bin_centers).sum(dim=-1, keepdim=True)

        value_pred = torch.nan_to_num(
            value_pred,
            nan=self.value_low,
            posinf=self.value_high,
            neginf=self.value_low,
        )

        value_pred = torch.clamp(
            value_pred,
            min=self.value_low,
            max=self.value_high,
        )

        return value_pred

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
    ) -> BatchFeature:
        backbone_output = self.process_backbone_output(backbone_output)

        return BatchFeature(
            data={
                "backbone_features": backbone_output.backbone_features,
                "backbone_attention_mask": backbone_output.backbone_attention_mask,
            }
        )

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        logits, value_hidden = self.value_head(
            backbone_features,
            backbone_attention_mask,
        )

        value_pred = self._compute_value_from_logits(logits)

        return BatchFeature(
            data={
                "value": value_pred,
                "value_hidden": value_hidden,
                "backbone_features": backbone_features,
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        features = self._encode_features(backbone_output, action_input)

        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            backbone_attention_mask=features.backbone_attention_mask,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)
