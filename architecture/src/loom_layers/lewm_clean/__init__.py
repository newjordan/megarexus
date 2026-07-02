"""LeWM-Clean — opinionated clean iteration on paper-faithful LeWM.

v0 adds two targeted changes addressing the team's documented failures:
- ContrastiveActionEmbedder: supervised-contrastive on action_id (fixes the
  Rex 0.58 action separation ratio observed in INTERLAB BLIND_TEST_RESULTS).
- AffordanceSideHead: per-(state, action) binary "did the frame change" head
  (implements the v4 per-step affordance target queued in LEWIS_PRECISION_REGIME
  iter 23).

Keeps paper-faithful base unchanged: ViTTinyEncoder + AdaLNTransformerPredictor + SIGReg.
"""
from __future__ import annotations

__all__ = ["v0"]
