"""Per-embodiment linear maps used by the action head encoders/decoders.

Parameter attribute names (``W``, ``b``, ``layer1``, ``layer2``, ``W1``/``W2``/``W3``,
``pos_encoding``) are part of the published checkpoint layout and must stay stable.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _silu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * torch.sigmoid(tensor)


class _TimeSinusoid(nn.Module):
    """Maps integer/float timesteps ``(B, T)`` to dim-aligned sin/cos features."""

    def __init__(self, width: int):
        super().__init__()
        self.width = width

    def forward(self, steps: torch.Tensor) -> torch.Tensor:
        steps = steps.to(dtype=torch.float32)
        batch, horizon = steps.shape
        half = self.width // 2
        scale = math.log(10000.0) / max(half, 1)
        freqs = torch.arange(half, device=steps.device, dtype=torch.float32).mul_(-scale).exp()
        angles = steps.unsqueeze(-1) * freqs
        encoded = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if encoded.shape[-1] < self.width:
            pad = torch.zeros(batch, horizon, self.width - encoded.shape[-1], device=steps.device)
            encoded = torch.cat((encoded, pad), dim=-1)
        return encoded


class EmbodimentLinear(nn.Module):
    """Batched gather of category-specific ``(in, out)`` weight matrices."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.num_categories = num_categories
        # Checkpoint keys: *.W / *.b
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, features: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        weights = self.W.index_select(0, category_ids)
        biases = self.b.index_select(0, category_ids)
        return torch.bmm(features, weights).add(biases.unsqueeze(1))


class EmbodimentMLP(nn.Module):
    """Two stacked :class:`EmbodimentLinear` layers with a ReLU in between."""

    def __init__(
        self,
        num_categories: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.num_categories = num_categories
        # Checkpoint keys: *.layer1.* / *.layer2.*
        self.layer1 = EmbodimentLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = EmbodimentLinear(num_categories, hidden_dim, output_dim)

    def forward(self, features: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.layer1(features, category_ids))
        return self.layer2(hidden, category_ids)


class TimedActionEncoder(nn.Module):
    """Inject diffusion time into action tokens with embodiment-specific maps."""

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        # Checkpoint keys: W1 / W2 / W3 / pos_encoding
        self.W1 = EmbodimentLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = EmbodimentLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = EmbodimentLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = _TimeSinusoid(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, _ = actions.shape
        if timesteps.ndim != 1 or timesteps.shape[0] != batch:
            raise ValueError(
                f"timesteps must have shape (B,), got {tuple(timesteps.shape)} for B={batch}"
            )
        time_grid = timesteps.unsqueeze(1).expand(batch, horizon)
        action_tokens = self.W1(actions, category_ids)
        time_tokens = self.pos_encoding(time_grid).to(dtype=action_tokens.dtype)
        fused = torch.cat((action_tokens, time_tokens), dim=-1)
        fused = _silu(self.W2(fused, category_ids))
        return self.W3(fused, category_ids)
