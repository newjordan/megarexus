"""SpriteDensityInformer — bias ACTION6 when the frame has interesting structure.

Connection to [[sam_goal_inference]]: full SAM-based sprite detection is the
real Layer-4 goal-inference pipeline. This informer is the cheap precursor:
counts connected components of same color (poor-man's segmentation) to
detect "this frame has objects to interact with" vs "this frame is empty
or uniformly colored".

When sprite count is high, the informer boosts ACTION6's logit (the
complex click action) and dampens other actions. When the frame is mostly
one color, it does the opposite (other actions are more likely useful).
Other action logits are unchanged.

Cost: O(64×64) connected-components scan per frame. Cached per (frame_hash)
so repeated calls within one planning session are O(1) after the first.
"""

from __future__ import annotations

import hashlib
from collections import deque

import numpy as np
import torch

from lewm.data.arc_agi_3 import ARC_COMPLEX_ACTION_ID
from lewm.imagination import Informer, InformerContext


def _count_components(frame: np.ndarray) -> int:
    """Count distinct connected components (4-connectivity) of same color."""

    h, w = frame.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for i in range(h):
        for j in range(w):
            if seen[i, j]:
                continue
            color = int(frame[i, j])
            # Skip background (color 0 in ARC's default palette) to focus on objects.
            if color == 0:
                seen[i, j] = True
                continue
            # BFS-flood-fill same color from (i, j)
            queue = deque([(i, j)])
            seen[i, j] = True
            while queue:
                y, x = queue.popleft()
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and int(frame[ny, nx]) == color:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
            count += 1
    return count


def _sprite_density_logits(
    ctx: InformerContext, avail: list[int],
    *,
    boost_threshold: int = 3,
    boost_scale: float = 2.0,
    dampen_scale: float = 1.0,
) -> torch.Tensor:
    """Per-action logits: positive on ACTION6 when sprite count ≥ threshold."""

    # Cache by frame hash within this context's extras dict.
    cache = ctx.extras.setdefault("_sprite_density_cache", {})
    frame = np.asarray(ctx.current_frame, dtype=np.uint8)
    h = hashlib.sha1(frame.tobytes()).hexdigest()
    if h not in cache:
        cache[h] = _count_components(frame)
    n_components = cache[h]

    logits = torch.zeros(len(avail), dtype=torch.float32)
    for i, aid in enumerate(avail):
        if aid == ARC_COMPLEX_ACTION_ID:
            if n_components >= boost_threshold:
                logits[i] = boost_scale
            else:
                logits[i] = -dampen_scale
    return logits


def SpriteDensityInformer(
    *,
    boost_threshold: int = 3,
    boost_scale: float = 2.0,
    dampen_scale: float = 1.0,
    weight: float = 1.0,
    enabled: bool = True,
) -> Informer:
    """Factory: returns an Informer whose fn closes over the parameters."""

    def fn(ctx: InformerContext, avail: list[int]) -> torch.Tensor:
        return _sprite_density_logits(
            ctx, avail,
            boost_threshold=boost_threshold,
            boost_scale=boost_scale,
            dampen_scale=dampen_scale,
        )

    return Informer(name="sprite_density", fn=fn, weight=weight, enabled=enabled)


__all__ = ["SpriteDensityInformer"]
