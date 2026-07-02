"""LeWM + Goal Inference integration with evidential (distributional) heads.

Uses LeWM's ViTTinyEncoder as the backbone encoder for ARC-AGI-3 trajectory
frames, then predicts evidential goal properties:

- goal_type: Dirichlet distribution over 4 classes
- agent_delta: Normal-Inverse-Gamma (evidential regression)
- keys_collected: Independent Beta distributions per color
- delta_px: Normal-Inverse-Gamma (evidential regression)

The evidential heads output distributional parameters, which are consumed
by the BeliefState for Kalman-style belief updates.

Key change from REX 3.0: The goal model now outputs full distributions
with uncertainty, not just point estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.evidential_heads import (
    DirichletHead,
    NIGHead,
    BetaHead,
)
from lewm.goal_property import ARC_NUM_COLORS, ARC_NUM_ACTIONS, NUM_GOAL_TYPES
from lewm.goal_property import GoalPropertyDataset, collate_property

# Re-export for backward compatibility with imports
from lewm.goal_property import (
    evaluate_property_model,
    goal_property_loss,
    PropertyTrainConfig,
    PropertyTrainer,
)


class LeWMGoalEncoder(nn.Module):
    """LeWM's ViTTinyEncoder with evidential prediction heads."""

    def __init__(
        self,
        encoder_config: EncoderConfig | None = None,
        hidden_dim: int = 192,
        pool: str = "mean",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if encoder_config is None:
            encoder_config = EncoderConfig(
                image_size=64,
                in_channels=16,
                patch_size=16,
                latent_dim=hidden_dim,
            )
        self.encoder = ViTTinyEncoder(encoder_config)
        self.latent_dim = self.encoder.latent_dim
        self.pool = pool
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Evidential prediction heads
        self.goal_type_head = DirichletHead(self.latent_dim, NUM_GOAL_TYPES)
        self.agent_head = NIGHead(self.latent_dim, out_dim=2)
        self.keys_head = BetaHead(self.latent_dim, ARC_NUM_COLORS)
        self.delta_px_head = NIGHead(self.latent_dim, out_dim=1)

    def forward(self, context_frames: torch.Tensor, context_actions: torch.Tensor) -> dict:
        """context_frames: [B, K, 16, 64, 64], context_actions: [B, K-1]"""
        B, K, C, H, W = context_frames.shape

        flat_frames = context_frames.reshape(B * K, C, H, W)
        latents = self.encoder(flat_frames)  # [B*K, D]
        latents = latents.reshape(B, K, self.latent_dim)

        if self.pool == "mean":
            h = latents.mean(dim=1)
        elif self.pool == "last":
            h = latents[:, -1, :]
        elif self.pool == "max":
            h = latents.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        goal_type_out = self.goal_type_head(h)
        agent_out = self.agent_head(h)
        keys_out = self.keys_head(h)
        delta_px_out = self.delta_px_head(h)

        return {
            # Dirichlet — goal_type
            "goal_type_alpha": goal_type_out["alpha"],
            "goal_type_logits": goal_type_out["logits"],
            # NIG — agent_delta
            "agent_gamma": agent_out["gamma"],
            "agent_nu": agent_out["nu"],
            "agent_alpha": agent_out["alpha"],
            "agent_beta": agent_out["beta"],
            # For backward compatibility (point estimate is gamma)
            "agent_delta": agent_out["gamma"],
            # Beta — keys
            "keys_alpha1": keys_out["alpha1"],
            "keys_alpha2": keys_out["alpha2"],
            # For backward compatibility (log-odds)
            "keys_logits": BetaHead.logits_from_alpha(keys_out["alpha1"], keys_out["alpha2"]),
            # NIG — delta_px
            "delta_px_gamma": delta_px_out["gamma"],
            "delta_px_nu": delta_px_out["nu"],
            "delta_px_alpha": delta_px_out["alpha"],
            "delta_px_beta": delta_px_out["beta"],
            # For backward compatibility (point estimate is gamma)
            "delta_px": delta_px_out["gamma"].squeeze(-1),
        }


class LeWMGoalInferenceModel(nn.Module):
    """Full LeWM + Goal Inference with evidential heads and belief state support.

    Architecture:
    - ViTTinyEncoder (shared) → per-frame latents [B, T, D]
    - Evidential property heads (Dirichlet, NIG, Beta)
    - The belief state uses these evidential parameters for Kalman updates
    """

    def __init__(
        self,
        encoder_config: EncoderConfig | None = None,
        hidden_dim: int = 192,
        pool: str = "mean",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if encoder_config is None:
            encoder_config = EncoderConfig(
                image_size=64,
                in_channels=16,
                patch_size=16,
                latent_dim=hidden_dim,
            )
        self.encoder = ViTTinyEncoder(encoder_config)
        self.latent_dim = self.encoder.latent_dim
        self.pool = pool
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Evidential prediction heads
        self.goal_type_head = DirichletHead(self.latent_dim, NUM_GOAL_TYPES)
        self.agent_head = NIGHead(self.latent_dim, out_dim=2)
        self.keys_head = BetaHead(self.latent_dim, ARC_NUM_COLORS)
        self.delta_px_head = NIGHead(self.latent_dim, out_dim=1)

    def forward(self, context_frames: torch.Tensor, context_actions: torch.Tensor) -> dict:
        """context_frames: [B, K, 16, 64, 64], context_actions: [B, K-1]"""
        B, K, C, H, W = context_frames.shape

        flat_frames = context_frames.reshape(B * K, C, H, W)
        latents = self.encoder(flat_frames)  # [B*K, D]
        latents = latents.reshape(B, K, self.latent_dim)

        if self.pool == "mean":
            h = latents.mean(dim=1)
        elif self.pool == "last":
            h = latents[:, -1, :]
        elif self.pool == "max":
            h = latents.max(dim=1).values
        else:
            raise ValueError(f"Unknown pool: {self.pool}")

        goal_type_out = self.goal_type_head(h)
        agent_out = self.agent_head(h)
        keys_out = self.keys_head(h)
        delta_px_out = self.delta_px_head(h)

        return {
            # Dirichlet — goal_type
            "goal_type_alpha": goal_type_out["alpha"],
            "goal_type_logits": goal_type_out["logits"],
            # NIG — agent_delta
            "agent_gamma": agent_out["gamma"],
            "agent_nu": agent_out["nu"],
            "agent_alpha": agent_out["alpha"],
            "agent_beta": agent_out["beta"],
            "agent_delta": agent_out["gamma"],
            # Beta — keys
            "keys_alpha1": keys_out["alpha1"],
            "keys_alpha2": keys_out["alpha2"],
            "keys_logits": BetaHead.logits_from_alpha(keys_out["alpha1"], keys_out["alpha2"]),
            # NIG — delta_px
            "delta_px_gamma": delta_px_out["gamma"],
            "delta_px_nu": delta_px_out["nu"],
            "delta_px_alpha": delta_px_out["alpha"],
            "delta_px_beta": delta_px_out["beta"],
            "delta_px": delta_px_out["gamma"].squeeze(-1),
        }

    def load_lewm_encoder(self, checkpoint_path: str | Path) -> None:
        """Load pretrained LeWM encoder weights from a REX checkpoint.

        REX checkpoints have 'model_state' key with encoder weights
        stored under 'encoder.X' keys.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        encoder_sd = {}
        if "model_state" in ckpt:
            for k, v in ckpt["model_state"].items():
                if k.startswith("encoder."):
                    encoder_sd[k[len("encoder."):]] = v
        elif "model_state_dict" in ckpt:
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("encoder."):
                    encoder_sd[k[len("encoder."):]] = v
        else:
            raise ValueError(f"Cannot find encoder weights in {checkpoint_path}")

        missing, unexpected = self.encoder.load_state_dict(encoder_sd, strict=False)
        if missing:
            print(f"  Encoder: missing keys: {missing}")
        if unexpected:
            print(f"  Encoder: unexpected keys: {unexpected}")
        print(f"  Loaded encoder from {checkpoint_path}")


__all__ = [
    "LeWMGoalEncoder",
    "LeWMGoalInferenceModel",
    "GoalPropertyDataset",
    "collate_property",
    "evaluate_property_model",
    "goal_property_loss",
    "PropertyTrainConfig",
    "PropertyTrainer",
]
