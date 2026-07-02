"""Goal latent reconstruction for REX 3.0 goal-conditioned predictor.

Given MAP estimates of goal properties from the belief state, reconstruct
a latent vector ``z_goal`` in the encoder's 192-dim space.

This is now a DETERMINISTIC mapping: the reconstructor always produces
z_goal from the MAP property estimates. Confidence is handled at the
belief state level, not here.

Architecture:
    goal_type (4) ─┐
    agent_delta (2) ──────┤── concat → MLP → z_goal (latent_dim)  [always 100% signal]
    keys (16) ────────────┤
    delta_px (1) ─────────┘
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.goal_property import ARC_NUM_COLORS, NUM_GOAL_TYPES


@dataclass
class GoalLatentConfig:
    """Configuration for goal latent reconstruction."""
    latent_dim: int = 192        # Must match LeWM encoder latent_dim
    goal_type_dim: int = NUM_GOAL_TYPES  # 4
    agent_dim: int = 2
    keys_dim: int = ARC_NUM_COLORS  # 16
    delta_px_dim: int = 1
    hidden_dim: int = 256
    dropout: float = 0.1

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")


class GoalLatentReconstructor(nn.Module):
    """Deterministic mapping from MAP property estimates → z_goal.

    Takes the MAP estimate from the belief state (always 100% signal,
    no confidence gating) and projects it to the latent space.

    Input: 23-dim MAP property vector
    Output: z_goal [B, latent_dim]
    """

    def __init__(self, config: GoalLatentConfig | None = None) -> None:
        super().__init__()
        self.config = config or GoalLatentConfig()
        self.config.validate()
        cfg = self.config

        # Input feature dimensionality (MAP properties)
        input_dim = (
            cfg.goal_type_dim +
            cfg.agent_dim +
            cfg.keys_dim +
            cfg.delta_px_dim
        )

        # Two-layer MLP with normalization
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        # Output projection (deterministic — no confidence head)
        self.latent_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small random init for latent_proj — NOT zero!
        # z_goal must carry signal from the start so the LeWM predictor
        # can learn to use it.
        nn.init.normal_(self.latent_proj.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.latent_proj.bias)

    def forward(self, properties_map: torch.Tensor) -> torch.Tensor:
        """Reconstruct goal latent from MAP property estimates.

        Args:
            properties_map: [B, 23] — concatenated MAP estimates from belief state
                Order: [goal_type(4), agent_delta(2), keys(16), delta_px(1)]

        Returns:
            z_goal: [B, latent_dim] — reconstructed goal latent
        """
        h = self.net(properties_map)  # [B, hidden_dim]
        z_goal = self.latent_proj(h)  # [B, latent_dim]
        return z_goal


__all__ = [
    "GoalLatentConfig",
    "GoalLatentReconstructor",
]
