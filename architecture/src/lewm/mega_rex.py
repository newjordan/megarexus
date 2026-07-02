"""Mega-REX: the canonical LeWM world-model architecture.

Built by infusing the Mockingbird body (src/lewm/goal_property.py) into
the LeWM predictor. Mega-REX is the go-to model for all Loom training.

Key innovations over baseline LeWM:
  - RMSNorm (cleaner normalization, better SIGReg isotropy)
  - QK-gain attention (per-head learnable logit scale)
  - Leaky-ReLU² MLP (soft gate, 3.75× expansion)
  - Residual mixing with x₀ (learned blend of current + initial input)
  - Loop blocks (virtual depth — repeat middle layers)
  - Skip gates (sigmoid-gated encoder→decoder connections)

Config tiers:
  mega_rex_small    — 2.1M, 128-dim, 4L, 4 heads (baseline parity)
  mega_rex_medium   — 12.1M, 256-dim, 8L, 8 heads, 1 loop
  mega_rex_full     — 61.4M, 512-dim, 11L, 8 heads, 2 loops, skip gates

Source: Mockingbird body by Maes et al. (sota_rascal/mockingbrid),
        adapted for LeWM latent prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.mockingbird_predictor import (
    MockingbirdPredictor,
    MockingbirdPredictorConfig,
)


# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════

@dataclass
class MegaREXConfig:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    predictor: MockingbirdPredictorConfig = field(
        default_factory=MockingbirdPredictorConfig
    )
    num_goal_types: int = 0  # 0 = no goal conditioning, >0 = embed goal_id
    # LeWM-v2: goal mode determines how `forward()` consumes goal information.
    #   "none"     — no goal conditioning (default; legacy behavior)
    #   "discrete" — look up goal_id in a learned embedding table (requires num_goal_types > 0)
    #   "latent"   — accept a continuous goal_latent vector of size latent_dim (hindsight)
    goal_mode: str = "none"
    # LeWM-v2 §3.3: per-cluster output heads for mechanic-heterogeneous games.
    # 0 = no cluster heads (legacy single-head). >0 = per-cluster residual
    # linear projection on the predictor output; consumed via `cluster_id`
    # kwarg in forward(). See scripts/compute_per_game_clusters.py for the
    # game_id → cluster_idx mapping derived from per-game pixel_diff stats.
    num_clusters: int = 0

    def __post_init__(self) -> None:
        if self.encoder.latent_dim != self.predictor.latent_dim:
            raise ValueError(
                f"encoder.latent_dim ({self.encoder.latent_dim}) must match "
                f"predictor.latent_dim ({self.predictor.latent_dim})"
            )
        if self.goal_mode not in ("none", "discrete", "latent"):
            raise ValueError(
                f"goal_mode must be one of 'none', 'discrete', 'latent'; got {self.goal_mode!r}"
            )
        # Backwards-compat inference: legacy configs that set num_goal_types > 0
        # without naming a goal_mode should behave as before (discrete mode).
        if self.goal_mode == "none" and self.num_goal_types > 0:
            self.goal_mode = "discrete"
        if self.goal_mode == "discrete" and self.num_goal_types <= 0:
            raise ValueError("goal_mode='discrete' requires num_goal_types > 0")
        if self.num_clusters < 0:
            raise ValueError("num_clusters must be >= 0")

    @staticmethod
    def mega_rex_small() -> MegaREXConfig:
        """2.1M params — baseline parity, fast iteration."""
        return MegaREXConfig(
            encoder=EncoderConfig(
                image_size=64, in_channels=16, patch_size=16,
                num_layers=4, num_heads=4, hidden_dim=128, latent_dim=128,
            ),
            predictor=MockingbirdPredictorConfig(
                latent_dim=128, action_dim=8,
                num_layers=4, num_heads=4, ff_mult=3.75,
                qk_gain_init=5.0, dropout=0.1,
            ),
            num_goal_types=168,
        )

    @staticmethod
    def mega_rex_medium() -> MegaREXConfig:
        """12.1M params — 256-dim, 8L, 8 heads, 1 loop. Good GPU sweet spot."""
        return MegaREXConfig(
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

    @staticmethod
    def mega_rex_full() -> MegaREXConfig:
        """61.4M params — 512-dim, 11L, 8 heads, 2 loops, skip gates.

        Target hardware: A100 or better. Designed for 200k+ game datasets.
        """
        return MegaREXConfig(
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


# ═══════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════

class MegaREX(nn.Module):
    """Mega-REX: ViT encoder + Mockingbird predictor world model.

    Forward:
      obs:        [B, T, C, H, W]  one-hot grid frames
      action_emb: [B, T, A]        one-hot action embeddings
      → emb:      [B, T, D]        encoder latents
      → next_emb: [B, T, D]        predicted next-latents

    Training loss (paper Algorithm 1):
      L_LeWM = MSE(emb[:, 1:], next_emb[:, :-1]) + λ · SIGReg(emb)
    """

    def __init__(self, config: MegaREXConfig | None = None):
        super().__init__()
        self.config = config or MegaREXConfig.mega_rex_full()
        self.encoder = ViTTinyEncoder(self.config.encoder)
        self.predictor = MockingbirdPredictor(self.config.predictor)
        self.latent_dim = self.encoder.latent_dim

        # Goal conditioning. Mode determines how goal information is consumed.
        if self.config.goal_mode == "discrete":
            self.goal_embedding = nn.Embedding(
                self.config.num_goal_types, self.latent_dim,
            )
        else:
            self.goal_embedding = None  # 'latent' uses raw vector; 'none' uses nothing

        # Per-cluster residual heads (LeWM-v2 §3.3). Each cluster gets its own
        # nn.Linear(D, D) that adds a learned delta to the predictor's next_emb
        # output, gated by cluster_id at forward time. Cluster heads are
        # initialized to ~zero so a fresh model recovers single-head behavior.
        if self.config.num_clusters > 0:
            self.cluster_heads = nn.ModuleList([
                nn.Linear(self.latent_dim, self.latent_dim)
                for _ in range(self.config.num_clusters)
            ])
            for h in self.cluster_heads:
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)
        else:
            self.cluster_heads = None

    def forward(
        self,
        obs: torch.Tensor,
        action_emb: torch.Tensor,
        goal_id: torch.Tensor | None = None,
        goal_latent: torch.Tensor | None = None,
        cluster_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T = obs.shape[:2]

        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        emb_flat = self.encoder(obs_flat)
        emb = emb_flat.reshape(B, T, -1)

        # Resolve goal embedding for the predictor (single (B, D) tensor or None).
        goal_emb = None
        if self.config.goal_mode == "discrete":
            if self.goal_embedding is not None and goal_id is not None:
                goal_emb = self.goal_embedding(goal_id)  # (B, D)
        elif self.config.goal_mode == "latent":
            if goal_latent is not None:
                if goal_latent.dim() == 1:
                    goal_emb = goal_latent.unsqueeze(0).expand(B, -1)
                elif goal_latent.dim() == 2 and goal_latent.size(0) == 1 and B > 1:
                    goal_emb = goal_latent.expand(B, -1)
                else:
                    goal_emb = goal_latent
                if goal_emb.size(-1) != self.latent_dim:
                    raise ValueError(
                        f"goal_latent last dim {goal_emb.size(-1)} != latent_dim {self.latent_dim}"
                    )

        next_emb = self.predictor(emb, action_emb, goal_emb=goal_emb)

        # LeWM-v2 §3.3: route through per-cluster residual head if configured.
        if self.cluster_heads is not None:
            if cluster_id is None:
                raise ValueError(
                    f"num_clusters={self.config.num_clusters} > 0 but cluster_id not passed to forward()"
                )
            if cluster_id.shape[0] != B:
                raise ValueError(
                    f"cluster_id shape {tuple(cluster_id.shape)} doesn't match batch B={B}"
                )
            # Sum of per-cluster deltas, each gated by a [B,1,1] mask. Heads
            # init at zero, so this is identity at the start of training.
            delta = torch.zeros_like(next_emb)
            for c, head in enumerate(self.cluster_heads):
                mask = (cluster_id == c).view(-1, 1, 1).to(next_emb.dtype)
                delta = delta + mask * head(next_emb)
            next_emb = next_emb + delta

        return {"emb": emb, "next_emb": next_emb}

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode frames to latents without prediction (embedding extraction)."""
        B, T = obs.shape[:2]
        obs_flat = obs.reshape(B * T, *obs.shape[2:])
        emb_flat = self.encoder(obs_flat)
        return emb_flat.reshape(B, T, -1)
