"""Guided motion adapter for locked Rex releases.

The adapter is intentionally separate from ``LeWM``. It is a small residual
module trained on controlled self-action traces while the Rex encoder,
predictor, and action embedder stay frozen.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class GuidedMotionConfig:
    latent_dim: int = 192
    action_dim: int = 192
    hidden_dim: int = 384

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class GuidedMotionAdapter(nn.Module):
    """Residual correction over Rex's one-step latent prediction.

    Inputs:
    - ``z_current``: ``[N, D]`` current Rex latent.
    - ``action_emb``: ``[N, A]`` frozen Rex action embedding.
    - ``base_next``: ``[N, D]`` frozen Rex next-latent prediction.

    Output:
    - corrected next latent, ``base_next + residual``.
    """

    def __init__(self, config: GuidedMotionConfig | None = None) -> None:
        super().__init__()
        self.config = config or GuidedMotionConfig()
        in_dim = self.config.latent_dim * 2 + self.config.action_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        z_current: torch.Tensor,
        action_emb: torch.Tensor,
        base_next: torch.Tensor,
    ) -> torch.Tensor:
        if z_current.shape != base_next.shape:
            raise ValueError(
                "z_current and base_next must have matching shape, got "
                f"{tuple(z_current.shape)} and {tuple(base_next.shape)}"
            )
        if z_current.dim() != 2 or action_emb.dim() != 2:
            raise ValueError("z_current, action_emb, and base_next must be rank-2")
        x = torch.cat([z_current, action_emb, base_next], dim=-1)
        return base_next + self.net(x)


__all__ = ["GuidedMotionAdapter", "GuidedMotionConfig"]
