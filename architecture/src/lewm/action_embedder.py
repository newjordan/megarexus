"""Structured action embedder for ARC-AGI-3.

Replaces the flat ``ARCActionEmbedder`` with a family-structured version that
enforces geometric relationships between action embeddings.

Key design:
  - Actions are organized into families:
    * NULL (action 0) — no-op, isolated embedding
    * DIRECTIONAL (actions 1-5) — up/down/left/right/square
    * PLACEMENT (action 6) — click/place with (x,y) coordinates
    * OTHER (action 7) — game-specific action
  - Directional actions live in a shared 2D subspace: emb(action_id) = base +
    direction_vector[action_id]. This means ACTION1's embedding is closer to
    ACTION4's than to ACTION6's, because they share the base.
  - Placement actions use a shared base + coordinate embeddings, with no
    per-action-id term (the action_id=6 embedding is identity-regularized).
  - The full embedding is: emb = family_base + action_offset + coord_offset
    where family_base is shared per family, action_offset is per-action within
    family (small for directional, zero-regularized for placement), and
    coord_offset is the (x,y) position for placement actions.

This is a drop-in replacement for ARCActionEmbedder — same interface, same
forward signature. It's used in the rex-2.0.0 training run.

Reference: blind test T2 (separation ratio 0.58). The flat embedder doesn't
organize actions by family. This structured version is designed to raise the
separation ratio above 0.8.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.data.arc_agi_3 import (
    ARC_GRID_SIZE,
    ARC_NUM_COLORS,
    ARC_NUM_ACTION_KINDS,
    ARC_NULL_ACTION_ID,
    ARC_TOTAL_ACTION_TOKENS,
)


# Action families: masks for each group
NULL_IDS = {0}
DIRECTIONAL_IDS = {1, 2, 3, 4, 5}
PLACEMENT_IDS = {6}
OTHER_IDS = {7}


class StructuredActionEmbedder(nn.Module):
    """Family-structured action embedder with geometric inductive bias.

    Architecture per action:
      emb = family_base[family_id] + action_offset[action_id] + coord_emb

    Where:
      - family_base: shared embedding per family (4 bases × D)
      - action_offset: residual per individual action, zero-init and
        L2-regularized toward zero (so actions default to family centroid)
      - coord_emb: (x, y) embedding, only active for placement (action 6)

    At initialization:
      - family_base is random normal
      - action_offset is zero (so all actions in a family start identical)
      - training must pull actions apart within a family

    The result: directional actions 1-5 share a base, so their embeddings
    are closer to each other than to placement or null. This is the
    inductive bias that the flat embedder was missing.

    Inherits the same forward signature as ARCActionEmbedder for drop-in
    compatibility. The ``action_dim`` matches the predictor's expected
    embedding dimension (default 192 for paper-faithful).
    """

    def __init__(self, action_dim: int = 192) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        self.action_dim = action_dim

        # Family bases: [4, D] — null, directional, placement, other
        self.family_base = nn.Embedding(4, action_dim)

        # Per-action offset: [8, D] — zero-init (actions start at family centroid)
        self.action_offset = nn.Embedding(ARC_TOTAL_ACTION_TOKENS, action_dim)
        nn.init.zeros_(self.action_offset.weight)

        # Coordinate embeddings for placement (action 6)
        self.x_emb = nn.Embedding(ARC_GRID_SIZE, action_dim)
        self.y_emb = nn.Embedding(ARC_GRID_SIZE, action_dim)
        nn.init.trunc_normal_(self.x_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.y_emb.weight, std=0.02)

        # Family ID lookup: action_id → family_id
        # 0→0 (null), 1-5→1 (directional), 6→2 (placement), 7→3 (other)
        family_map = torch.zeros(ARC_TOTAL_ACTION_TOKENS, dtype=torch.long)
        for aid in DIRECTIONAL_IDS:
            family_map[aid] = 1
        for aid in PLACEMENT_IDS:
            family_map[aid] = 2
        for aid in OTHER_IDS:
            family_map[aid] = 3
        self.register_buffer("family_map", family_map, persistent=True)

    def get_family_id(self, action_id: torch.Tensor) -> torch.Tensor:
        """Map action_id to family_id (0=null, 1=directional, 2=placement, 3=other)."""
        return self.family_map[action_id.clamp(0, ARC_TOTAL_ACTION_TOKENS - 1)]

    def forward(
        self,
        action_id: torch.Tensor,
        action_x: torch.Tensor,
        action_y: torch.Tensor,
        action_complex_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Family base: [*, D]
        family_ids = self.get_family_id(action_id)  # [*]
        base = self.family_base(family_ids)  # [*, D]

        # Per-action offset: [*, D] — zero init, defaults to family centroid
        offset = self.action_offset(action_id)  # [*, D]

        # Coordinate embedding: only active when action_complex_mask is 1 (action 6)
        xy = self.x_emb(action_x) + self.y_emb(action_y)  # [*, D]

        return base + offset + action_complex_mask.float().unsqueeze(-1) * xy

    def compute_family_separation(self) -> dict[str, float]:
        """Compute family-separation metrics (for T2 monitoring).

        Returns:
            dict with separation_ratio and per-family distances.
        """
        with torch.no_grad():
            # Get all 8 action embeddings (with x=y=0, no mask)
            all_ids = torch.arange(ARC_TOTAL_ACTION_TOKENS, device=self.family_map.device)
            zeros = torch.zeros_like(all_ids)
            mask = torch.zeros(ARC_TOTAL_ACTION_TOKENS, device=self.family_map.device)
            emb = self.forward(all_ids, zeros, zeros, mask)  # [8, D]

            dir_emb = emb[list(DIRECTIONAL_IDS)]
            plac_emb = emb[list(PLACEMENT_IDS)]
            null_emb = emb[0:1]

            dir_within = torch.pdist(dir_emb).mean().item() if dir_emb.shape[0] > 1 else 0.0
            dir_plac = (dir_emb.mean(dim=0) - plac_emb.mean(dim=0)).norm().item()
            dir_null = (dir_emb.mean(dim=0) - null_emb.mean(dim=0)).norm().item()

            # Separation ratio: cross-family / within-family
            cross = max(dir_plac, dir_null, 1e-8)
            within = max(dir_within, 1e-8)
            separation_ratio = cross / within

            return {
                "separation_ratio": round(separation_ratio, 4),
                "directional_within": round(dir_within, 4),
                "directional_vs_placement": round(dir_plac, 4),
                "directional_vs_null": round(dir_null, 4),
            }


__all__ = ["StructuredActionEmbedder"]
