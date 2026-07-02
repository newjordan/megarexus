"""Mockingbird-LeWM hybrid: ViT encoder + Mockingbird predictor.

A drop-in replacement for the standard LeWM model. Uses the same encoder
(ViT-Tiny with patch embedding) but swaps the predictor for Mockingbird's
RMSNorm + QK-gain attention + leaky-ReLU² MLP + residual mixing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.mockingbird_predictor import (
    MockingbirdPredictor,
    MockingbirdPredictorConfig,
    RMSNorm,
)


@dataclass
class MockingbirdLeWMConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: MockingbirdPredictorConfig = field(default_factory=MockingbirdPredictorConfig)

    def __post_init__(self) -> None:
        if self.encoder.latent_dim != self.predictor.latent_dim:
            raise ValueError(
                f"encoder.latent_dim ({self.encoder.latent_dim}) must match "
                f"predictor.latent_dim ({self.predictor.latent_dim})"
            )

    @classmethod
    def paper_small(cls) -> MockingbirdLeWMConfig:
        """1.8M param hybrid — comparable to baseline LeWM but Mockingbird body."""
        return cls(
            encoder=EncoderConfig(
                image_size=64, in_channels=16, patch_size=16,
                num_layers=4, num_heads=4, hidden_dim=128, latent_dim=128,
            ),
            predictor=MockingbirdPredictorConfig(
                latent_dim=128, action_dim=8,
                num_layers=4, num_heads=4, ff_mult=3.75,
                qk_gain_init=5.0, dropout=0.1,
            ),
        )

    @classmethod
    def mockingbird_medium(cls) -> MockingbirdLeWMConfig:
        """~7M param — wider/deeper Mockingbird, 256-dim, 8 heads."""
        return cls(
            encoder=EncoderConfig(
                image_size=64, in_channels=16, patch_size=16,
                num_layers=6, num_heads=8, hidden_dim=256, latent_dim=256,
            ),
            predictor=MockingbirdPredictorConfig(
                latent_dim=256, action_dim=8,
                num_layers=8, num_heads=8, ff_mult=3.75,
                qk_gain_init=5.125, dropout=0.1,
                num_loops=1, loop_start=3, loop_end=5,
            ),
        )

    @classmethod
    def mockingbird_full(cls) -> MockingbirdLeWMConfig:
        """~35M param — full Mockingbird body, 512-dim, 11 layers, loops."""
        return cls(
            encoder=EncoderConfig(
                image_size=64, in_channels=16, patch_size=16,
                num_layers=8, num_heads=8, hidden_dim=512, latent_dim=512,
            ),
            predictor=MockingbirdPredictorConfig(
                latent_dim=512, action_dim=8,
                num_layers=11, num_heads=8, ff_mult=3.75,
                qk_gain_init=5.125, dropout=0.1,
                num_loops=2, loop_start=3, loop_end=5,
                skip_gates=True,
            ),
        )


class MockingbirdLeWM(nn.Module):
    """Hybrid LeWM: ViT encoder + Mockingbird predictor.

    Forward inputs:
      - obs: [B, T, C, H, W] raw pixel observations
      - action_emb: [B, T, action_dim] per-step action embeddings

    Forward outputs:
      - emb: [B, T, latent_dim] encoder outputs
      - next_emb: [B, T, latent_dim] predicted next-latents
    """

    def __init__(self, config: MockingbirdLeWMConfig | None = None):
        super().__init__()
        self.config = config or MockingbirdLeWMConfig.paper_small()
        self.encoder = ViTTinyEncoder(self.config.encoder)
        self.predictor = MockingbirdPredictor(self.config.predictor)
        self.latent_dim = self.encoder.latent_dim

    def forward(
        self,
        obs: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T = obs.shape[:2]

        # Flatten batch+time for encoder
        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        emb_flat = self.encoder(obs_flat)          # (B*T, latent_dim)
        emb = emb_flat.reshape(B, T, -1)           # (B, T, latent_dim)

        # Predict next latents with Mockingbird predictor
        next_emb = self.predictor(emb, action_emb)  # (B, T, latent_dim)

        return {"emb": emb, "next_emb": next_emb}

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a sequence without predictor (for embedding extraction)."""
        B, T = obs.shape[:2]
        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        emb_flat = self.encoder(obs_flat)
        return emb_flat.reshape(B, T, -1)
