"""Lewis-Clean v0 — paper-faithful goal-property head trained on REAL clean ARC trajectories.

Replaces the quarantined Lewis-on-v5 checkpoint (per [[engine_first_dataset_v0]])
with a fresh Lewis-class predictor trained on the clean trajectories.pt collection
(1,500 real ARC-AGI-3 games, 162k steps).

Lewis's job (per `papers/CRITICAL_FOUNDATION.md` + `src/lewm/rex_goal_packet.py`):
predict goal-relevant properties from observed frames so Rex/LeWM has GOAL CONTEXT
for planning, not just dynamics.

Quarantined Lewis predicted: goal_type, mechanic_probs, family_probs, subclass_probs,
keys_probs, delta_px, agent_delta, plus study-tags heads — those labels were synthetic
v5 study_tags and are not present in real ARC trajectories.

Lewis-Clean v0 predicts the labels that ARE in clean ARC trajectory data:
  - game_id_logits [25]    — which of the 25 public ARC games is this?
  - future_levels [1]      — future-max levels_completed (continuous, regressor)
  - terminal_state [3]     — NOT_FINISHED | WIN | GAME_OVER prediction

This is a deliberately narrower scope. We can ADD per-mechanic or per-goal-type
heads later if we manually annotate or if GameMaster's clean mechanic_cards
provides a real mapping from game_id → mechanic.

The output dict matches `RexGoalPacket`'s structure pattern so the
RexGoalPacketProjector can be extended to consume LewisClean output.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from lewm.encoder import EncoderConfig, ViTTinyEncoder


# Public ARC game IDs (alphabetical) — matches what arc_agi.Arcade lists
LEWIS_CLEAN_GAME_IDS: tuple[str, ...] = (
    "ar25-0c556536", "bp35-0a0ad940", "cd82-fb555c5d", "cn04-2fe56bfb", "dc22-fdcac232",
    "ft09-0d8bbf25", "g50t-5849a774", "ka59-38d34dbb", "lp85-305b61c3", "ls20-9607627b",
    "m0r0-492f87ba", "s5i5-18d95033", "sb26-7fbdac44", "sc25-635fd71a", "sk48-d8078629",
    "sp80-589a99af", "su15-1944f8ab", "tn36-ef4dde99", "tr87-cd924810", "tu93-0768757b",
    "vc33-5430563c", "wa30-ee6fef47",
)

TERMINAL_STATE_NAMES = ("NOT_FINISHED", "WIN", "GAME_OVER")


def game_id_to_index(game_id: str) -> int:
    """Map game_id string -> integer class. Returns -1 if unknown."""
    try:
        return LEWIS_CLEAN_GAME_IDS.index(game_id)
    except ValueError:
        return -1


def terminal_state_to_index(state_name: str) -> int:
    """Map GameState.name -> integer class. Returns 0 (NOT_FINISHED) if unknown."""
    try:
        return TERMINAL_STATE_NAMES.index(state_name)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class LewisCleanV0Config:
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        image_size=64, patch_size=16, in_channels=16,
        hidden_dim=192, latent_dim=192, num_layers=12, num_heads=3,
    ))
    num_games: int = len(LEWIS_CLEAN_GAME_IDS)
    num_terminal_states: int = len(TERMINAL_STATE_NAMES)
    head_hidden: int = 256
    head_dropout: float = 0.1

    # Loss weights
    lambda_game: float = 0.5
    lambda_value: float = 1.0
    lambda_terminal: float = 0.5


class LewisCleanV0(nn.Module):
    """ViTTinyEncoder + 3 goal-property heads.

    Forward:
        obs [B, T, C, H, W] (16-channel one-hot ARC frames)

    Returns dict:
        emb              [B, T, latent_dim]
        game_logits      [B, T, num_games]
        value            [B, T]                      (future-max levels regressor)
        terminal_logits  [B, T, num_terminal_states] (NOT_FINISHED|WIN|GAME_OVER)
    """

    def __init__(self, config: LewisCleanV0Config | None = None) -> None:
        super().__init__()
        self.config = config or LewisCleanV0Config()
        cfg = self.config
        self.encoder = ViTTinyEncoder(cfg.encoder)
        d = cfg.encoder.latent_dim
        h = cfg.head_hidden

        def mlp_head(out_dim: int) -> nn.Module:
            return nn.Sequential(
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(h, out_dim),
            )

        self.game_head = mlp_head(cfg.num_games)
        self.value_head = mlp_head(1)
        self.terminal_head = mlp_head(cfg.num_terminal_states)

        # Value head bias init so it doesn't dead-zero
        with torch.no_grad():
            self.value_head[-1].bias.fill_(0.05)

    def encode_sequence(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() != 5:
            raise ValueError(f"obs must be [B, T, C, H, W], got {tuple(obs.shape)}")
        b, t, c, h, w = obs.shape
        flat = obs.reshape(b * t, c, h, w)
        z = self.encoder(flat)
        return z.view(b, t, -1)

    def forward(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        emb = self.encode_sequence(obs)
        return {
            "emb": emb,
            "game_logits": self.game_head(emb),
            "value": self.value_head(emb).squeeze(-1),
            "terminal_logits": self.terminal_head(emb),
        }

    # Compatibility shim: returns a goal-packet-like dict for downstream Rex use.
    @torch.no_grad()
    def goal_packet(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward + softmax/sigmoid for packet shape consumable by RexGoalPacketProjector.

        Returns:
            game_id_probs    [B, T, num_games]
            game_id_argmax   [B, T] int
            value            [B, T]
            terminal_probs   [B, T, num_terminal_states]
            terminal_argmax  [B, T] int
        """
        out = self.forward(obs)
        return {
            "emb": out["emb"],
            "game_id_probs": F.softmax(out["game_logits"].float(), dim=-1),
            "game_id_argmax": out["game_logits"].argmax(dim=-1),
            "value": out["value"],
            "terminal_probs": F.softmax(out["terminal_logits"].float(), dim=-1),
            "terminal_argmax": out["terminal_logits"].argmax(dim=-1),
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass
class LewisCleanV0LossOutput:
    game: torch.Tensor
    value: torch.Tensor
    terminal: torch.Tensor
    total: torch.Tensor


def lewis_clean_v0_loss(
    model_out: dict[str, torch.Tensor],
    *,
    game_ids: torch.Tensor,        # [B, T] int in [0, num_games)
    future_max_levels: torch.Tensor,  # [B, T] float
    terminal_state_ids: torch.Tensor,  # [B, T] int in [0, 3)
    model: LewisCleanV0,
    lambda_game: float | None = None,
    lambda_value: float | None = None,
    lambda_terminal: float | None = None,
    sample_weights: torch.Tensor | None = None,
    terminal_class_weights: torch.Tensor | None = None,
) -> LewisCleanV0LossOutput:
    cfg = model.config
    λg = lambda_game if lambda_game is not None else cfg.lambda_game
    λv = lambda_value if lambda_value is not None else cfg.lambda_value
    λt = lambda_terminal if lambda_terminal is not None else cfg.lambda_terminal

    # Flatten [B, T] -> [B*T] for cross_entropy
    B, T = game_ids.shape
    game_logits = model_out["game_logits"].reshape(B * T, -1).float()
    game_targets = game_ids.reshape(-1).long()
    L_game = F.cross_entropy(game_logits, game_targets, ignore_index=-1)

    value_pred = model_out["value"].float()
    value_diff_sq = (value_pred - future_max_levels.float()) ** 2  # [B, T]
    if sample_weights is not None:
        L_value = (value_diff_sq * sample_weights.float().unsqueeze(-1)).sum() / (sample_weights.sum() * T)
    else:
        L_value = value_diff_sq.mean()

    term_logits = model_out["terminal_logits"].reshape(B * T, -1).float()
    term_targets = terminal_state_ids.reshape(-1).long()
    L_terminal = F.cross_entropy(term_logits, term_targets, weight=terminal_class_weights)

    L_total = λg * L_game + λv * L_value + λt * L_terminal

    return LewisCleanV0LossOutput(
        game=L_game, value=L_value, terminal=L_terminal, total=L_total,
    )


__all__ = [
    "LEWIS_CLEAN_GAME_IDS", "TERMINAL_STATE_NAMES",
    "game_id_to_index", "terminal_state_to_index",
    "LewisCleanV0", "LewisCleanV0Config",
    "LewisCleanV0LossOutput", "lewis_clean_v0_loss",
]
