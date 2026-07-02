"""AgencyMaskHead — predict which cells the agent controls, with NULL-agent fallback.

Sonnet replay-lane finding #2: Q0 can be NULL. tn36 uses cursor-as-agency
(no spatial agent), tr87 uses selector cycling. The head must handle both:
  1. SPATIAL mode: output an [H, W] mask of controlled cells
  2. NULL mode: output a binary "no spatial agent" flag

The head consumes the latent z from the encoder and outputs:
  - agency_mask: [B, 1, H, W] probability map of controlled cells
  - agency_exists: [B] probability that a spatial agent exists

At inference, if agency_exists < threshold, the mask is ignored and the
downstream planner uses cursor/selector heuristics instead.

Usage:
    from lewm.agency_mask_head import AgencyMaskHead, AgencyMaskHeadConfig

    cfg = AgencyMaskHeadConfig(latent_dim=192, image_size=64)
    head = AgencyMaskHead(cfg)
    mask, exists = head(z)  # z: [B, latent_dim]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AgencyMaskHeadConfig:
    latent_dim: int = 192
    image_size: int = 64  # output mask resolution
    hidden_dim: int = 128
    null_threshold: float = 0.5  # below this, treat as NULL-agent


class AgencyMaskHead(nn.Module):
    """Controlled-cell mask predictor with NULL-agent detection."""

    def __init__(self, config: AgencyMaskHeadConfig):
        super().__init__()
        self.config = config

        # Shared backbone
        self.shared = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Spatial mask branch: hidden_dim → [1, H, W]
        H, W = config.image_size, config.image_size
        self.spatial_proj = nn.Linear(config.hidden_dim, H * W)

        # NULL-agent detection branch: hidden_dim → 1 logit
        self.null_classifier = nn.Linear(config.hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict agency mask and existence probability.

        Args:
            z: encoder latent, shape [B, latent_dim] or [B*T, latent_dim]

        Returns:
            agency_mask: [B, 1, H, W] controlled-cell probabilities
            agency_exists: [B] probability a spatial agent exists
        """
        B = z.shape[0]
        H, W = self.config.image_size, self.config.image_size

        shared = self.shared(z)  # [B, hidden_dim]

        # Spatial mask
        mask_flat = self.spatial_proj(shared)  # [B, H*W]
        agency_mask = torch.sigmoid(mask_flat).view(B, 1, H, W)

        # NULL detection
        exists_logit = self.null_classifier(shared).squeeze(-1)  # [B]
        agency_exists = torch.sigmoid(exists_logit)

        return agency_mask, agency_exists

    def loss(
        self,
        agency_mask: torch.Tensor,
        agency_exists: torch.Tensor,
        mask_targets: torch.Tensor,
        exists_targets: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute agency mask losses.

        Args:
            agency_mask: [B, 1, H, W] predicted mask
            agency_exists: [B] predicted existence
            mask_targets: [B, 1, H, W] ground-truth controlled-cell mask
            exists_targets: [B] 1 if spatial agent exists, 0 if NULL
            valid: [B] boolean mask for valid samples (optional)

        Returns:
            dict with mask_loss, exists_loss, total_loss
        """
        if valid is None:
            valid = torch.ones(agency_mask.shape[0], device=agency_mask.device)
        valid_f = valid.float()

        # Mask loss: BCE on controlled cells, only when agent exists
        mask_loss = F.binary_cross_entropy(
            agency_mask, mask_targets, reduction="none"
        )
        mask_loss = mask_loss.mean(dim=(1, 2, 3))  # per-sample mean
        mask_loss = (mask_loss * exists_targets * valid_f).sum() / (
            exists_targets * valid_f
        ).sum().clamp(min=1)

        # NULL detection loss: BCE
        exists_loss = F.binary_cross_entropy(
            agency_exists, exists_targets, reduction="none"
        )
        exists_loss = (exists_loss * valid_f).sum() / valid_f.sum().clamp(min=1)

        total = mask_loss + exists_loss
        return {
            "mask_loss": mask_loss,
            "exists_loss": exists_loss,
            "total_loss": total,
        }

    def is_null_agent(self, agency_exists: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True where no spatial agent exists."""
        return agency_exists < self.config.null_threshold
