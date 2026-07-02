"""Action-dependence losses for Novel LeWM ARC experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionGuardMarginConfig:
    """Margin config for real-action preference over counterfactual actions."""

    margin: float = 1e-4
    shuffled_weight: float = 1.0
    zero_weight: float = 1.0

    def validate(self) -> None:
        if self.margin < 0:
            raise ValueError("margin must be >= 0")
        if self.shuffled_weight < 0:
            raise ValueError("shuffled_weight must be >= 0")
        if self.zero_weight < 0:
            raise ValueError("zero_weight must be >= 0")


def action_guard_margin_loss(
    real_pred_loss: torch.Tensor,
    shuffled_action_pred_loss: torch.Tensor,
    zero_action_pred_loss: torch.Tensor,
    config: ActionGuardMarginConfig | None = None,
) -> torch.Tensor:
    """Penalize cases where counterfactual actions predict as well as real actions.

    The loss is zero only when both counterfactual losses are worse than the real
    action loss by at least ``margin``:

        ``shuffled_loss >= real_loss + margin``
        ``zero_loss >= real_loss + margin``

    This is an ARC/Rex Novel objective, not part of the paper-faithful LeWM
    training loss.
    """

    cfg = config or ActionGuardMarginConfig()
    cfg.validate()
    shuffled_term = F.relu(real_pred_loss + cfg.margin - shuffled_action_pred_loss)
    zero_term = F.relu(real_pred_loss + cfg.margin - zero_action_pred_loss)
    return cfg.shuffled_weight * shuffled_term + cfg.zero_weight * zero_term
