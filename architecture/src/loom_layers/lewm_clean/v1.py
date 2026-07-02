"""LeWM-Clean v1 — v0 + ValueHead trained on real win signal.

Iter 5 of the win-rate ladder confirmed v0 doesn't translate frame-prediction
quality into wins (0% win-rate across 30 episodes; affordance picks game-over
actions on bp35). The DECISION GATE answered Branch 6c: architecture change
needed.

v1 adds a ValueHead V(state) → expected future-max levels_completed. CEM then
plans with V as the cost (maximize future levels) instead of affordance-sum
(maximize frame change). This is RL/value learning, an explicit deviation from
the paper's pure-JEPA design per the [[feedback_no_paper_deference]] memory.

Critical data context: only 91 of 1,500 trajectories (6.1%) and 91 of 162,185
steps (0.056%) carry positive level-completion signal. v1 training must
oversample positive-trajectory windows or the value head collapses to a
constant ≈ 0.06.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from loom_layers.lewm_clean.v0 import (
    AffordanceSideHead,
    AffordanceSideHeadConfig,
    ContrastiveActionEmbedder,
    ContrastiveActionEmbedderConfig,
)
from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.data.arc_agi_3 import ARCActionEmbedder
from lewm.predictor import AdaLNTransformerPredictor, PredictorConfig
from lewm.sigreg import SIGReg, SIGRegConfig


# ---------------------------------------------------------------------------
# Value head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValueHeadConfig:
    latent_dim: int = 192
    hidden_dim: int = 256
    dropout: float = 0.1


class ValueHead(nn.Module):
    """V(state) → scalar prediction of future-max-levels-completed.

    Trained on the per-step `future_max_levels` target extracted from
    trajectories: for step t in trajectory τ, target = max(levels_completed[t..end]).
    Monotonically non-decreasing along τ; ranges over {0, 1} in our data
    (no trajectory completes level 2+).

    Uses MSE loss. Bias the output toward small positive values via init
    so the head doesn't dead-zero at the start.
    """

    def __init__(self, config: ValueHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or ValueHeadConfig()
        cfg = self.config
        self.head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )
        # Init the final layer's bias to a small positive — predicts ~0.06 (the
        # global mean future-max-levels) initially so it isn't dead-zero.
        with torch.no_grad():
            self.head[-1].bias.fill_(0.05)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """emb [B, T, D] -> value [B, T] (predicted future-max-levels)."""
        return self.head(emb).squeeze(-1)


# ---------------------------------------------------------------------------
# Composed LeWM-Clean v1
# ---------------------------------------------------------------------------


@dataclass
class LeWMCleanV1Config:
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        image_size=64, patch_size=16, in_channels=16, hidden_dim=192, latent_dim=192,
        num_layers=12, num_heads=3,
    ))
    predictor: PredictorConfig = field(default_factory=lambda: PredictorConfig(
        latent_dim=192, action_dim=192, num_layers=6, num_heads=16,
    ))
    sigreg: SIGRegConfig = field(default_factory=lambda: SIGRegConfig(latent_dim=192, num_projections=256))
    action_embedder: ContrastiveActionEmbedderConfig = field(default_factory=ContrastiveActionEmbedderConfig)
    affordance_head: AffordanceSideHeadConfig = field(default_factory=AffordanceSideHeadConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)
    coordinate_actions: bool = False

    lambda_sigreg: float = 0.1
    lambda_aff: float = 0.1
    lambda_contrast: float = 0.05
    lambda_value: float = 1.0  # value head dominates since positive signal is rare
    lambda_gain: float = 1.0

    def __post_init__(self) -> None:
        if self.encoder.latent_dim != self.predictor.latent_dim:
            raise ValueError("encoder.latent_dim must == predictor.latent_dim")


class LeWMCleanV1(nn.Module):
    """LeWM-Clean v1: v0 + ValueHead.

    Forward:
        obs [B, T, C, H, W] (16-channel one-hot ARC frames)
        action_ids [B, T] int

    Returns dict:
        emb           [B, T, latent_dim]
        next_emb      [B, T, latent_dim]
        action_emb    [B, T, action_dim]
        aff_logits    [B, T]
        value         [B, T]    <-- new
    """

    def __init__(self, config: LeWMCleanV1Config | None = None) -> None:
        super().__init__()
        self.config = config or LeWMCleanV1Config()
        cfg = self.config
        self.encoder = ViTTinyEncoder(cfg.encoder)
        self.predictor = AdaLNTransformerPredictor(cfg.predictor)
        self.sigreg = SIGReg(config=cfg.sigreg)
        self.action_embedder = (
            ARCActionEmbedder(action_dim=cfg.action_embedder.action_dim)
            if cfg.coordinate_actions
            else ContrastiveActionEmbedder(cfg.action_embedder)
        )
        self.affordance_head = AffordanceSideHead(cfg.affordance_head)
        self.value_head = ValueHead(cfg.value_head)
        self.gain_head = ValueHead(cfg.value_head)
        self.latent_dim = self.encoder.latent_dim

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 5:
            raise ValueError(f"obs must be [B, T, C, H, W], got {tuple(obs.shape)}")
        b, t, c, h, w = obs.shape
        flat = obs.reshape(b * t, c, h, w)
        z = self.encoder(flat)
        return z.view(b, t, -1)

    def embed_actions(
        self,
        action_ids: torch.Tensor,
        action_x: torch.Tensor | None = None,
        action_y: torch.Tensor | None = None,
        action_complex_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.config.coordinate_actions:
            return self.action_embedder(action_ids)
        if action_x is None:
            action_x = torch.zeros_like(action_ids)
        if action_y is None:
            action_y = torch.zeros_like(action_ids)
        if action_complex_mask is None:
            action_complex_mask = torch.zeros_like(action_ids, dtype=torch.float32)
        return self.action_embedder(
            action_ids.clamp(0, 7),
            action_x.clamp(0, 63),
            action_y.clamp(0, 63),
            action_complex_mask.float(),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_ids: torch.Tensor,
        action_x: torch.Tensor | None = None,
        action_y: torch.Tensor | None = None,
        action_complex_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        emb = self.encode_sequence(obs)
        action_emb = self.embed_actions(action_ids, action_x, action_y, action_complex_mask)
        next_emb = self.predictor(emb, action_emb)
        aff_logits = self.affordance_head(emb, action_emb)
        value = self.value_head(emb)
        gain_logits = self.gain_head(next_emb)
        return {
            "emb": emb,
            "next_emb": next_emb,
            "action_emb": action_emb,
            "aff_logits": aff_logits,
            "value": value,
            "gain_logits": gain_logits,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass
class LeWMCleanV1LossOutput:
    pred: torch.Tensor
    sigreg: torch.Tensor
    affordance: torch.Tensor
    contrastive: torch.Tensor
    value: torch.Tensor
    gain: torch.Tensor
    total: torch.Tensor


def lewm_clean_v1_loss(
    model_out: dict[str, torch.Tensor],
    *,
    did_frame_change: torch.Tensor,
    action_ids: torch.Tensor,
    future_max_levels: torch.Tensor,
    level_delta: torch.Tensor | None = None,
    model: LeWMCleanV1,
    lambda_sigreg: float | None = None,
    lambda_aff: float | None = None,
    lambda_contrast: float | None = None,
    lambda_value: float | None = None,
    lambda_gain: float | None = None,
    sample_weights: torch.Tensor | None = None,
) -> LeWMCleanV1LossOutput:
    cfg = model.config
    λs = lambda_sigreg if lambda_sigreg is not None else cfg.lambda_sigreg
    λa = lambda_aff if lambda_aff is not None else cfg.lambda_aff
    λc = lambda_contrast if lambda_contrast is not None else cfg.lambda_contrast
    λv = lambda_value if lambda_value is not None else cfg.lambda_value
    λg = lambda_gain if lambda_gain is not None else cfg.lambda_gain

    emb = model_out["emb"]
    next_emb = model_out["next_emb"]
    aff_logits = model_out["aff_logits"]
    value = model_out["value"]
    gain_logits = model_out.get("gain_logits")

    L_pred = F.mse_loss(next_emb[:, :-1].float(), emb[:, 1:].float())
    b, t, d = emb.shape
    L_sigreg = model.sigreg(emb.reshape(b * t, d))
    L_aff = F.binary_cross_entropy_with_logits(aff_logits.float(), did_frame_change.float())
    if hasattr(model.action_embedder, "contrastive_loss"):
        L_contrast = model.action_embedder.contrastive_loss(action_ids)
    else:
        L_contrast = torch.zeros((), dtype=emb.dtype, device=emb.device)

    # Weighted MSE for value head — sample_weights upweights positive-trajectory windows
    value_diff_sq = (value.float() - future_max_levels.float()) ** 2  # [B, T]
    if sample_weights is not None:
        # sample_weights [B] — broadcast over T
        L_value = (value_diff_sq * sample_weights.float().unsqueeze(-1)).sum() / (sample_weights.sum() * t)
    else:
        L_value = value_diff_sq.mean()
    if gain_logits is not None and level_delta is not None:
        gain_target = (level_delta.float() > 0).float()
        gain_loss = F.binary_cross_entropy_with_logits(gain_logits.float(), gain_target, reduction="none")
        if sample_weights is not None:
            L_gain = (gain_loss * sample_weights.float().unsqueeze(-1)).sum() / (sample_weights.sum() * t)
        else:
            L_gain = gain_loss.mean()
    else:
        L_gain = torch.zeros((), dtype=emb.dtype, device=emb.device)

    L_total = L_pred + λs * L_sigreg + λa * L_aff + λc * L_contrast + λv * L_value + λg * L_gain

    return LeWMCleanV1LossOutput(
        pred=L_pred,
        sigreg=L_sigreg,
        affordance=L_aff,
        contrastive=L_contrast,
        value=L_value,
        gain=L_gain,
        total=L_total,
    )


__all__ = [
    "ValueHead",
    "ValueHeadConfig",
    "LeWMCleanV1",
    "LeWMCleanV1Config",
    "LeWMCleanV1LossOutput",
    "lewm_clean_v1_loss",
]
