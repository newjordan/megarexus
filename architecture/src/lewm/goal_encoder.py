"""GoalFrameEncoder — encode goal-configuration frames into latent vectors.

Lightweight CNN encoder that takes a goal frame [H, W] (color-index or RGB)
and produces a fixed-dimension latent vector. Designed to be plugged into
LeWM's predictor as a goal-conditioning signal.

Unlike the main ViT encoder (which encodes the CURRENT frame), this encoder
is specialized for the TARGET configuration — the "won" state.

Architecture: 3 conv blocks → global avg pool → MLP → latent_dim.
~50K params at default config — negligible overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np


@dataclass
class GoalEncoderConfig:
    in_channels: int = 16  # ARC one-hot palette (or 3 for RGB)
    hidden_dims: tuple[int, ...] = (32, 64, 128)
    latent_dim: int = 64
    kernel_size: int = 3
    dropout: float = 0.0

    def validate(self) -> None:
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if len(self.hidden_dims) < 1:
            raise ValueError("hidden_dims must have at least one entry")


class GoalFrameEncoder(nn.Module):
    """Lightweight CNN encoder for goal-configuration frames.

    Input: [B, C, H, W] goal frame (C = in_channels, e.g. 16 for ARC one-hot)
    Output: [B, latent_dim] goal latent vector

    Usage:
        encoder = GoalFrameEncoder()
        goal_latent = encoder(goal_frames)  # [B, 64]
        # Feed into predictor as conditioning signal
    """

    def __init__(self, config: GoalEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or GoalEncoderConfig()
        self.config.validate()
        cfg = self.config

        # Convolutional blocks
        layers: list[nn.Module] = []
        in_ch = cfg.in_channels
        for hidden in cfg.hidden_dims:
            layers.extend([
                nn.Conv2d(in_ch, hidden, cfg.kernel_size, padding=cfg.kernel_size // 2),
                nn.BatchNorm2d(hidden),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
            if cfg.dropout > 0:
                layers.append(nn.Dropout2d(cfg.dropout))
            in_ch = hidden
        self.conv = nn.Sequential(*layers)

        # Compute flattened dim after conv stack
        # Each MaxPool2d(2) halves H and W
        # For 64×64 input with 3 pool layers: 64→32→16→8
        n_pools = len(cfg.hidden_dims)
        final_spatial = 64 // (2 ** n_pools)
        self.flat_dim = cfg.hidden_dims[-1] * final_spatial * final_spatial

        self.projector = nn.Sequential(
            nn.Linear(self.flat_dim, cfg.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(cfg.latent_dim * 2, cfg.latent_dim),
            nn.LayerNorm(cfg.latent_dim),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode goal frames.

        Args:
            frames: [B, C, H, W] where C = in_channels.

        Returns:
            [B, latent_dim] goal latent vector.
        """
        if frames.dim() == 3:
            # Single frame: add batch dim
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError(
                f"expected [B, C, H, W] or [C, H, W], got {tuple(frames.shape)}"
            )
        x = self.conv(frames)
        x = x.flatten(1)  # [B, flat_dim]
        return self.projector(x)


def goal_frame_to_onehot(frame: np.ndarray, num_colors: int = 16) -> torch.Tensor:
    """Convert a color-index goal frame [H, W] → one-hot [1, 16, H, W] tensor.

    Args:
        frame: numpy array [H, W] of int color indices 0..15
        num_colors: number of color channels (default 16 for ARC palette)

    Returns:
        torch.Tensor [1, num_colors, H, W]
    """
    f = np.clip(frame, 0, num_colors - 1).astype(np.int64)
    oh = np.zeros((num_colors, f.shape[0], f.shape[1]), dtype=np.float32)
    for c in range(num_colors):
        oh[c] = (f == c).astype(np.float32)
    return torch.from_numpy(oh).unsqueeze(0)
