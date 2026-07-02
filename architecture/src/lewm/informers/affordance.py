"""AffordanceHeadInformer — re-use LeWMCleanV1's already-trained affordance head.

Connection: ``src/loom_layers/lewm_clean/v1.py`` trained an
``AffordanceSideHead`` on the iter-0 trajectory data; it predicts a scalar
"this action is afforded by this state" score. We currently consume the
head only at training time (auxiliary loss) — this informer surfaces it as
a search prior so the work isn't wasted.

When the LeWMCleanV1 model is unavailable or untrained, the informer
returns uniform logits (no contribution). When available, it computes
per-action affordance logits over ``available_actions`` and returns them
centered to zero mean.

Cost: one forward pass per action through the affordance head per call.
``available_actions`` is small (max 7), so the cost is tiny. Caching keyed
by (frame_hash, available_actions) makes repeated calls in one session O(1).
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from lewm.imagination import Informer, InformerContext


def AffordanceHeadInformer(
    *,
    model: Optional[nn.Module] = None,        # expected to be a LeWMCleanV1 instance
    device: str = "cpu",
    weight: float = 1.0,
    enabled: bool = True,
) -> Informer:
    """Factory: returns an Informer that reads the affordance head's score per action.

    If ``model`` is None or missing ``affordance_head`` / ``encode_sequence`` /
    ``action_embedder``, the informer falls back to zero logits — strictly
    a passthrough.
    """

    has_head = (
        model is not None
        and hasattr(model, "encode_sequence")
        and hasattr(model, "action_embedder")
        and hasattr(model, "affordance_head")
    )
    cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _per_state(frame: np.ndarray, avail: list[int]) -> torch.Tensor:
        if not has_head:
            return torch.zeros(len(avail), dtype=torch.float32)
        h = hashlib.sha1(np.asarray(frame, dtype=np.uint8).tobytes()).hexdigest()
        ck = f"{h}::{','.join(map(str, avail))}"
        cached = cache.get(ck)
        if cached is not None:
            return cached

        # frame → obs [1, 1, 16, 64, 64] one-hot
        from lewm.data.arc_agi_3 import frame_to_onehot
        obs = frame_to_onehot(frame).to(device)               # [16, 64, 64]
        obs = obs.unsqueeze(0).unsqueeze(0)                    # [1, 1, 16, 64, 64]
        emb = model.encode_sequence(obs)                       # [1, 1, D]

        logits = torch.zeros(len(avail), dtype=torch.float32)
        for i, aid in enumerate(avail):
            aid_t = torch.tensor([[aid]], dtype=torch.long, device=device)
            action_emb = model.action_embedder(aid_t)          # [1, 1, A]
            aff = model.affordance_head(emb, action_emb)       # [1, 1] scalar
            logits[i] = float(aff.squeeze().item())

        # Center so the informer contributes "relative preferences" not an
        # overall offset that would distort other informers' magnitudes.
        logits = logits - logits.mean()
        cache[ck] = logits
        return logits

    def fn(ctx: InformerContext, avail: list[int]) -> torch.Tensor:
        return _per_state(ctx.current_frame, avail)

    return Informer(name="affordance_head", fn=fn, weight=weight, enabled=enabled)


__all__ = ["AffordanceHeadInformer"]
