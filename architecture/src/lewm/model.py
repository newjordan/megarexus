"""Composed LeWM model: encoder + predictor over a (B, T) sequence.

Matches paper Algorithm 1 (page 6, pseudocode)::

    emb = encoder(obs)                    # (B, T, D)
    next_emb = predictor(emb, actions)    # (B, T, D)

The training loop computes ``pred_loss = F.mse_loss(emb[:, 1:], next_emb[:, :-1])``
and ``sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))`` outside this module.

The encoder's per-frame call still operates on flat ``(N, C, H, W)`` batches —
``LeWM`` reshapes ``(B, T, C, H, W) → (B*T, C, H, W)`` for the encoder and
folds the result back to ``(B, T, D)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.multisense_encoder import MultiSenseEncoder, MultiSenseEncoderConfig
from lewm.predictor import AdaLNTransformerPredictor, PredictorConfig


@dataclass
class LeWMConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    multi_sense: MultiSenseEncoderConfig | None = None
    n_mechanics: int = 0  # 0 = no mechanic conditioning, 6 = full mechanic embedding

    def __post_init__(self) -> None:
        encoder_dim = (
            self.multi_sense.latent_dim
            if self.multi_sense is not None
            else self.encoder.latent_dim
        )
        if encoder_dim != self.predictor.latent_dim:
            raise ValueError(
                "encoder.latent_dim must match predictor.latent_dim "
                f"({encoder_dim} vs {self.predictor.latent_dim})"
            )

    @property
    def latent_dim(self) -> int:
        if self.multi_sense is not None:
            return self.multi_sense.latent_dim
        return self.encoder.latent_dim


class LeWM(nn.Module):
    """End-to-end LeWM model — paper Algorithm 1.

    Forward inputs:

    - ``obs``: ``[B, T, C, H, W]`` raw pixel observations
    - ``action_emb``: ``[B, T, action_dim]`` per-step action embeddings

    Forward outputs (dict):

    - ``emb``: ``[B, T, latent_dim]`` encoder outputs across the whole sequence
    - ``next_emb``: ``[B, T, latent_dim]`` predictor outputs (position ``t``
      predicts the latent at time ``t+1``)

    Set ``multi_sense`` in LeWMConfig to enable sprite + color-topology pathways
    alongside the ViT encoder. When ``multi_sense=None``, behavior is identical
    to the paper-faithful ViT-only encoder.
    """

    def __init__(self, config: LeWMConfig | None = None) -> None:
        super().__init__()
        self.config = config or LeWMConfig()
        if self.config.multi_sense is not None:
            self.encoder = MultiSenseEncoder(self.config.multi_sense)
        else:
            self.encoder = ViTTinyEncoder(self.config.encoder)
        self.predictor = AdaLNTransformerPredictor(self.config.predictor)
        self.latent_dim = self.encoder.latent_dim
        if self.config.n_mechanics > 0:
            self.mechanic_emb = nn.Embedding(self.config.n_mechanics, 32)
        else:
            self.mechanic_emb = None

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W] → [B, T, D]`` via flat-batched encoder calls."""

        if obs.dim() != 5:
            raise ValueError(
                f"obs must be [B, T, C, H, W], got {tuple(obs.shape)}"
            )
        b, t, c, h, w = obs.shape
        flat = obs.reshape(b * t, c, h, w)
        z_flat = self.encoder(flat)
        return z_flat.view(b, t, -1)

    def forward(
        self,
        obs: torch.Tensor,
        action_emb: torch.Tensor,
        mechanic_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.encode_sequence(obs)
        if self.mechanic_emb is not None and mechanic_idx is not None:
            m = self.mechanic_emb(mechanic_idx)  # [B, T, 32] or [B, 32]
            if m.dim() == 2:
                m = m.unsqueeze(1).expand(-1, emb.shape[1], -1)
            action_emb = torch.cat([action_emb, m], dim=-1)
        next_emb = self.predictor(emb, action_emb)
        return {"emb": emb, "next_emb": next_emb}

    def forward_with_goal(
        self,
        obs: torch.Tensor,
        action_emb: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
        goal_confidence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass (goal is NOT fed into predictor — it's a terminal cost).

        The goal is used only as a terminal attractor loss in the training loop.
        The predictor runs open-loop: ẑ_{t+1} = predictor(z_t, a_t).
        A goal-matching loss ‖ẑ_H - z_goal‖² is added separately.

        Returns:
            dict with emb, next_emb (from predictor), z_goal (reconstructed goal latent)
        """
        emb = self.encode_sequence(obs)
        if self.mechanic_emb is not None and mechanic_idx is not None:
            m = self.mechanic_emb(mechanic_idx)
            if m.dim() == 2:
                m = m.unsqueeze(1).expand(-1, emb.shape[1], -1)
            action_emb = torch.cat([action_emb, m], dim=-1)
        # Predictor does NOT receive goal — it's only a dynamics model
        next_emb = self.predictor(emb, action_emb)
        return {"emb": emb, "next_emb": next_emb, "z_goal": goal_emb}
