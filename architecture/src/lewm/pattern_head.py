"""PatternHead — lightweight pattern-recognition model on grid frames.

Uses a lightweight CNN directly on raw one-hot ARC-AGI-3 frames.
This is necessary because the Rex encoder's latent representation is optimized
for prediction (next-state), not for game discrimination — all game frames
map to nearly identical latents under frozen Rex.

The Rex encoder's BatchNorm projector (0.046 → 0.000003 per-dim variance) plus
the [CLS] token's strong compression make it unsuitable as a feature extractor
for discrimination tasks without fine-tuning.

Instead, PatternHead uses a standalone 4-layer CNN directly on [B, 16, 64, 64]
one-hot frames. This is architecturally similar to taking the first few layers
of a vision encoder, but trained end-to-end for concept discrimination.

  Input:  [B, 16, 64, 64] one-hot color grid
  Output: concept_logits [B, C] (game concept classifier)
          goal_embedding  [B, D] (structured goal semantics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PatternHeadConfig:
    """Configuration for the PatternHead.

    Attributes:
        in_channels: Input channels (16 for one-hot ARC colors).
        cnn_dim: Base CNN channel dimension.
        hidden_dim: Hidden dimension of the MLP classifier.
        num_concepts: Number of game concepts to classify.
        goal_embed_dim: Dimension of the goal embedding vector.
        dropout: Dropout probability.
    """
    in_channels: int = 16
    cnn_dim: int = 64
    hidden_dim: int = 256
    num_concepts: int = 120
    goal_embed_dim: int = 64
    dropout: float = 0.2


class PatternHead(nn.Module):
    """Lightweight CNN for game concept recognition from ARC-AGI-3 frames.

    Two heads sharing a common CNN trunk:

    1. **Concept classifier**: frame → concept logits
    2. **Goal embedding**: frame → normalized goal embedding (contrastive)
    """

    def __init__(self, config: PatternHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or PatternHeadConfig()
        cfg = self.config

        # CNN trunk: 4 conv layers → global avg pool → MLP → heads
        c = cfg.cnn_dim
        self.conv1 = nn.Conv2d(cfg.in_channels, c, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(c * 4)
        self.conv4 = nn.Conv2d(c * 4, c * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(c * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # MLP on pooled features
        self.mlp = nn.Sequential(
            nn.Linear(c * 4, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )

        self.concept_head = nn.Linear(cfg.hidden_dim, cfg.num_concepts)
        self.goal_head = nn.Linear(cfg.hidden_dim, cfg.goal_embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        frames: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass from raw grid frames.

        Args:
            frames: [B, 16, 64, 64] one-hot color grids.

        Returns:
            dict with:
                - "concept_logits": [B, num_concepts]
                - "goal_embed": [B, goal_embed_dim] (L2-normalized)
                - "features": [B, hidden_dim]
        """
        # CNN: 64 → 32 → 16 → 8 → 8 (with poolings)
        h = F.gelu(self.bn1(self.conv1(frames)))       # [B, c, 64, 64]
        h = F.max_pool2d(h, 2)                           # [B, c, 32, 32]
        h = F.gelu(self.bn2(self.conv2(h)))              # [B, c*2, 32, 32]
        h = F.max_pool2d(h, 2)                           # [B, c*2, 16, 16]
        h = F.gelu(self.bn3(self.conv3(h)))              # [B, c*4, 16, 16]
        h = F.max_pool2d(h, 2)                           # [B, c*4, 8, 8]
        h = F.gelu(self.bn4(self.conv4(h)))              # [B, c*4, 8, 8]
        h = self.pool(h).flatten(1)                      # [B, c*4]

        h = self.mlp(h)                                   # [B, hidden_dim]

        concept_logits = self.concept_head(h)
        goal_embed = F.normalize(self.goal_head(h), p=2, dim=-1)
        return {
            "concept_logits": concept_logits,
            "goal_embed": goal_embed,
            "features": h,
        }

    def concept_loss(
        self,
        concept_logits: torch.Tensor,
        concept_idx: torch.Tensor,
    ) -> torch.Tensor:
        return F.cross_entropy(concept_logits, concept_idx)

    def contrastive_loss(
        self,
        goal_embed: torch.Tensor,
        concept_idx: torch.Tensor,
        *,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """Supervised contrastive loss (NT-Xent)."""
        b = goal_embed.shape[0]
        if b <= 1:
            return goal_embed.new_zeros(())

        unique = concept_idx.unique()
        if len(unique) <= 1:
            return goal_embed.new_zeros(())

        sim = goal_embed @ goal_embed.T / temperature
        ce = concept_idx.view(-1, 1)
        pos_mask = (ce == ce.T).float()
        eye = torch.eye(b, device=goal_embed.device)
        pos_mask = pos_mask - eye

        pos_exp = torch.exp(sim) * pos_mask
        pos_sum = pos_exp.sum(dim=1)
        neg_mask = 1.0 - eye
        denom = (torch.exp(sim) * neg_mask).sum(dim=1)

        eps = 1e-8
        loss = -torch.log(pos_sum / (denom + eps) + eps)
        valid = pos_sum > 0
        return loss[valid].mean() if valid.any() else goal_embed.new_zeros(())

    def total_loss(
        self,
        outputs: dict[str, torch.Tensor],
        concept_idx: torch.Tensor,
        *,
        contrastive_weight: float = 0.1,
        temperature: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        ce = self.concept_loss(outputs["concept_logits"], concept_idx)
        ccl = self.contrastive_loss(
            outputs["goal_embed"], concept_idx, temperature=temperature,
        )
        total = ce + contrastive_weight * ccl
        return {"total": total, "ce": ce, "contrastive": ccl}


# ---------------------------------------------------------------------------
# PatternHeadTrainer — lightweight training loop
# ---------------------------------------------------------------------------


@dataclass
class PatternTrainConfig:
    num_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    contrastive_weight: float = 0.1
    temperature: float = 0.1
    seed: int = 0
    log_every: int = 10


class PatternHeadTrainer:
    """Train a PatternHead directly on raw grid frames (no encoder)."""

    def __init__(
        self,
        *,
        pattern_head: PatternHead,
        config: PatternTrainConfig | None = None,
    ) -> None:
        self.pattern_head = pattern_head
        self.config = config or PatternTrainConfig()
        torch.manual_seed(self.config.seed)

        self.optimizer = torch.optim.AdamW(
            self.pattern_head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: list[dict[str, float]] = []

    def train_step(
        self,
        frames: torch.Tensor,
        concept_idx: torch.Tensor,
    ) -> dict[str, float]:
        self.pattern_head.train()
        outputs = self.pattern_head(frames)
        losses = self.pattern_head.total_loss(
            outputs, concept_idx,
            contrastive_weight=self.config.contrastive_weight,
            temperature=self.config.temperature,
        )

        self.optimizer.zero_grad()
        losses["total"].backward()
        self.optimizer.step()

        return {
            "total": float(losses["total"].detach()),
            "ce": float(losses["ce"].detach()),
            "contrastive": float(losses["contrastive"].detach()),
        }

    @torch.no_grad()
    def evaluate(
        self,
        frames: torch.Tensor,
        concept_idx: torch.Tensor,
    ) -> dict[str, float]:
        self.pattern_head.eval()
        outputs = self.pattern_head(frames)
        losses = self.pattern_head.total_loss(
            outputs, concept_idx,
            contrastive_weight=self.config.contrastive_weight,
            temperature=self.config.temperature,
        )

        pred = outputs["concept_logits"].argmax(dim=-1)
        acc = (pred == concept_idx).float().mean()

        return {
            "total": float(losses["total"]),
            "ce": float(losses["ce"]),
            "contrastive": float(losses["contrastive"]),
            "accuracy": float(acc),
        }


__all__ = [
    "PatternHead",
    "PatternHeadConfig",
    "PatternHeadTrainer",
    "PatternTrainConfig",
]
