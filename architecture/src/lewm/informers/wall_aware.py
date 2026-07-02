"""WallAwareInformer — dampen ACTION6 when most of the grid is walls.

Uses :class:`lewm.frame_structure.FrameStructure` to compute the wall
fraction of the grid extent. When walls dominate, the informer reduces
ACTION6's prior — most ACTION6(x, y) clicks would land on a wall and
waste a planning slot.

The xy-level wall validation (rejecting individual ACTION6(x, y) clicks
that land in a wall) lives in the MCTS action_validator_fn slot, not here.
This informer only shifts the action-id-level prior.

The `FrameStructure` you pass in can come from any of three sources
(in order of preference):

1. `from_game_instance(instance)` — authoritative GameMaster ground truth.
2. `from_wall_grid_npy(frame, npy_path)` — pre-computed canonical wall
   grid from `experiments/2026-05-14_wall_grid_map_pack_001_v2/`.
3. `parse_trajectory(frames)` — heuristic fallback for real ARC frames.

The informer doesn't care which source built the structure; it just
reads `wall_mask`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from lewm.data.arc_agi_3 import ARC_COMPLEX_ACTION_ID
from lewm.frame_structure import (
    ARC_GRID_SIZE,
    FrameStructure,
    from_wall_grid_npy,
    parse_frame,
)
from lewm.imagination import Informer, InformerContext


def WallAwareInformer(
    *,
    structure: FrameStructure,
    high_wall_threshold: float = 0.5,
    dampen_scale: float = 1.5,
    boost_scale: float = 0.5,
    weight: float = 1.0,
    enabled: bool = True,
) -> Informer:
    """Build an Informer that dampens or boosts ACTION6 by wall density."""

    if structure.wall_mask is None:
        wall_frac = 0.0
    else:
        # Restrict to the grid extent so border padding doesn't dominate.
        ext = structure.grid_extent
        sub = structure.wall_mask[ext.y0:ext.y1, ext.x0:ext.x1]
        n_cells = max(1, sub.numel())
        wall_frac = float(sub.float().sum().item()) / n_cells

    def fn(ctx: InformerContext, avail: list[int]) -> torch.Tensor:
        logits = torch.zeros(len(avail), dtype=torch.float32)
        for i, aid in enumerate(avail):
            if aid == ARC_COMPLEX_ACTION_ID:
                if wall_frac >= high_wall_threshold:
                    logits[i] = -dampen_scale
                else:
                    logits[i] = boost_scale * (1.0 - wall_frac)
        return logits

    inf = Informer(name="wall_aware", fn=fn, weight=weight, enabled=enabled)
    # Stash for inspection / ablation telemetry.
    inf.fn.__dict__["wall_fraction"] = wall_frac  # type: ignore[attr-defined]
    return inf


def WallAwareInformerFromCache(
    *,
    frame: np.ndarray,
    wall_grid_npy_path: str | Path,
    **informer_kwargs,
) -> Informer:
    """Convenience factory: load a cached wall_grid npy and build the informer.

    Use this when you have a still that's covered by the
    `experiments/2026-05-14_wall_grid_map_pack_001_v2/wall_grids_npy/`
    cache — avoids re-running heuristic wall inference per call.
    """

    struct = from_wall_grid_npy(frame, wall_grid_npy_path)
    return WallAwareInformer(structure=struct, **informer_kwargs)


__all__ = ["WallAwareInformer", "WallAwareInformerFromCache"]
