"""Frozen learned dynamics plug for Hemisphere_E (LP specialist or wide pour).

Loads checkpoints from megarexus learned_physics / wide_world_model when present.
Fail-closed: status=degraded if missing; never blocks the brain.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[4]
_LP = _REPO / "synthesis_lab" / "experiments" / "learned_physics"
_WIDE_RUN = (
    _REPO
    / "synthesis_lab"
    / "experiments"
    / "wide_world_model"
    / "runs"
    / "sp80_from_scratch_20260713"
    / "best.pt"
)
# Prefer wide pour arch (imagination quality), then heldout D4 dynamics
_DEFAULT_CANDIDATES = [
    _WIDE_RUN,
    _LP
    / "runs"
    / "merged_v5_l1pos_20260713"
    / "dynamics_heldout_d4"
    / "best.pt",
    _LP / "runs" / "merged_v5_l1pos_20260713" / "dynamics" / "best.pt",
    _LP / "runs" / "merged_v3" / "dynamics" / "best.pt",
]


class WorldModelPlug(nn.Module):
    """Predict next board / embedding using frozen learned dynamics."""

    def __init__(
        self,
        device: str | torch.device = "cpu",
        ckpt_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.status = "degraded"
        self.backend = "none"
        self.ckpt_path: str | None = None
        self._model: nn.Module | None = None
        self._kind = "none"  # wide | unet | none
        path = self._resolve_path(ckpt_path)
        if path is not None:
            self._load(path)

    @staticmethod
    def _resolve_path(ckpt_path: str | Path | None) -> Path | None:
        if ckpt_path is not None:
            p = Path(ckpt_path)
            return p if p.is_file() else None
        env = os.environ.get("MEGAMIND_WORLD_CKPT")
        if env and Path(env).is_file():
            return Path(env)
        for c in _DEFAULT_CANDIDATES:
            if c.is_file():
                return c
        return None

    def _load(self, path: Path) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return
        # Wide path
        try:
            if path.as_posix().find("wide_world_model") >= 0 or (
                isinstance(ckpt, dict)
                and any(
                    k.startswith("enc3.")
                    for k in (ckpt.get("model_state") or {})
                )
            ):
                sys.path.insert(0, str(_LP))
                from wide_dynamics_adapter import load_wide_pour_dynamics  # type: ignore

                self._model = load_wide_pour_dynamics(str(path), self.device)
                self._kind = "wide"
                self.status = "ok"
                self.backend = f"wide:{path.name}"
                self.ckpt_path = str(path)
                return
        except Exception:
            pass
        # DynamicsUNet path
        try:
            sys.path.insert(0, str(_LP))
            from train_dynamics import DynamicsUNet  # type: ignore

            state = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
            cfg = ckpt.get("config") or {}
            base = int(cfg.get("base") or cfg.get("base_channels") or 48)
            model = DynamicsUNet(base=base)
            # tolerate nested keys
            if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
            self._model = model.to(self.device).eval()
            self._kind = "unet"
            self.status = "ok"
            self.backend = f"dynamics_unet:{path.name}"
            self.ckpt_path = str(path)
        except Exception:
            self._model = None
            self.status = "degraded"
            self.backend = "load_failed"

    @torch.no_grad()
    def predict_board(self, board_long: torch.Tensor) -> torch.Tensor | None:
        """board [B,H,W] long → predicted next [B,H,W] long, or None."""
        if self._model is None:
            return None
        board_long = board_long.long().to(self.device)
        if board_long.dim() == 2:
            board_long = board_long.unsqueeze(0)
        try:
            logits = self._model(board_long)  # [B,16,H,W]
            return logits.argmax(dim=1)
        except Exception:
            return None

    @torch.no_grad()
    def agreement(
        self, board: torch.Tensor, board_next: torch.Tensor
    ) -> dict[str, float]:
        pred = self.predict_board(board)
        if pred is None:
            return {"wm_status": 0.0, "wm_pixel_acc": 0.0, "wm_changed_acc": 0.0}
        board_next = board_next.long().to(self.device)
        if board_next.dim() == 2:
            board_next = board_next.unsqueeze(0)
        if board.dim() == 2:
            board = board.unsqueeze(0)
        board = board.long().to(self.device)
        match = (pred == board_next).float().mean().item()
        changed = board != board_next
        if changed.any():
            ch = (pred[changed] == board_next[changed]).float().mean().item()
        else:
            ch = 1.0
        return {
            "wm_status": 1.0,
            "wm_pixel_acc": float(match),
            "wm_changed_acc": float(ch),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "backend": self.backend,
            "ckpt": self.ckpt_path,
            "kind": self._kind,
        }
