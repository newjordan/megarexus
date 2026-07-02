"""ViT-Tiny encoder for LeWM (paper Section 3.1).

Input: RGB image batch ``[B, C, H, W]`` (paper default ``[B, 3, 224, 224]``).
Output: latent vector ``[B, latent_dim]`` extracted from the [CLS] token after
a 1-layer MLP + BatchNorm projector.

Paper-exact configuration:
- Patch size 14 → 16×16=256 patches at 224×224
- 12 transformer layers
- 3 attention heads
- hidden / latent dim 192
- learnable [CLS] token prepended; learnable positional embedding (length
  num_patches + 1)
- pre-LN transformer blocks
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


PAPER_PATCH_SIZE = 14
PAPER_NUM_LAYERS = 12
PAPER_NUM_HEADS = 3
PAPER_HIDDEN_DIM = 192
PAPER_LATENT_DIM = 192
PAPER_IMAGE_SIZE = 224
PAPER_IN_CHANNELS = 3


@dataclass(frozen=True)
class EncoderConfig:
    image_size: int = PAPER_IMAGE_SIZE
    in_channels: int = PAPER_IN_CHANNELS
    patch_size: int = PAPER_PATCH_SIZE
    num_layers: int = PAPER_NUM_LAYERS
    num_heads: int = PAPER_NUM_HEADS
    hidden_dim: int = PAPER_HIDDEN_DIM
    latent_dim: int = PAPER_LATENT_DIM
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def validate(self) -> None:
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} not divisible by "
                f"patch_size {self.patch_size}"
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} not divisible by "
                f"num_heads {self.num_heads}"
            )
        if self.in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class ViTTinyEncoder(nn.Module):
    """Patchify → Transformer → [CLS] → MLP+BN projector."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        self.config.validate()
        cfg = self.config

        self.patch_embed = nn.Conv2d(
            cfg.in_channels,
            cfg.hidden_dim,
            kernel_size=cfg.patch_size,
            stride=cfg.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches + 1, cfg.hidden_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.hidden_dim * cfg.mlp_ratio),
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_layers,
        )
        self.norm = nn.LayerNorm(cfg.hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.latent_dim),
            nn.BatchNorm1d(cfg.latent_dim),
        )
        self.latent_dim = cfg.latent_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(
                f"expected [B, C, H, W], got {tuple(images.shape)}"
            )
        b, c, h, w = images.shape
        cfg = self.config
        if c != cfg.in_channels:
            raise ValueError(
                f"expected {cfg.in_channels} channels, got {c}"
            )
        if h != cfg.image_size or w != cfg.image_size:
            raise ValueError(
                f"expected {cfg.image_size}x{cfg.image_size}, got {h}x{w}"
            )
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        cls_out = x[:, 0]
        return self.projector(cls_out)
