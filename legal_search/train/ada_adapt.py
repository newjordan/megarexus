"""AdaJEPA-inspired online adaptation for LegalBlendLoRA.

Port of the plan-act-adapt-replan idea:
  - small online replay buffer of recent transitions
  - one (or few) gradient steps on LoRA params only
  - self-supervised loss: change-prediction on taken action
  - no reward labels, no expert demos, no 30B base

Ref: https://agenticlearning.ai/adajepa/  (arXiv:2606.32026)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

import numpy as np
import torch
import torch.nn.functional as F

from legal_search.train.features import encode_legal_features
from legal_search.train.lora import lora_parameters
from legal_search.train.model import (
    GOOSE_SIMPLE,
    GRID,
    LegalBlendConfig,
    LegalBlendLoRA,
    goose_index_tensor,
)
from legal_search.types import Action, Transition


@dataclass
class AdaConfig:
    """Defaults mirror AdaJEPA lightness: small buffer, 1 step, late params only."""

    buffer_size: int = 8  # paper default ~5; slightly larger for discrete actions
    grad_steps: int = 1
    lr: float = 1e-3
    lora_only: bool = True  # analogue of "final encoder/predictor layers only"
    device: str | None = None
    # planning: blend change vs policy for horizon-1 action choice
    plan_change_weight: float = 0.65
    plan_policy_weight: float = 0.35


@dataclass
class BufferRow:
    frame: torch.Tensor          # [H,W] long
    features: torch.Tensor       # [F]
    action_id: int
    x: int
    y: int
    did_change: float
    frame_next: torch.Tensor | None = None
    features_next: torch.Tensor | None = None


class OnlineReplayBuffer:
    """Tiny online buffer — AdaJEPA B."""

    def __init__(self, capacity: int = 8) -> None:
        self.capacity = int(capacity)
        self.rows: Deque[BufferRow] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, row: BufferRow) -> None:
        self.rows.append(row)

    def wipe(self) -> None:
        self.rows.clear()

    def as_batch(self, device: torch.device) -> dict[str, torch.Tensor]:
        if not self.rows:
            raise RuntimeError("empty online buffer")
        frame = torch.stack([r.frame for r in self.rows]).to(device)
        features = torch.stack([r.features for r in self.rows]).to(device)
        action_id = torch.tensor([r.action_id for r in self.rows], dtype=torch.long, device=device)
        x = torch.tensor([r.x for r in self.rows], dtype=torch.long, device=device)
        y = torch.tensor([r.y for r in self.rows], dtype=torch.long, device=device)
        did_change = torch.tensor([r.did_change for r in self.rows], dtype=torch.float32, device=device)
        batch = {
            "frame": frame,
            "features": features,
            "action_id": action_id,
            "x": x,
            "y": y,
            "did_change": did_change,
        }
        if all(r.frame_next is not None for r in self.rows):
            batch["frame_next"] = torch.stack([r.frame_next for r in self.rows]).to(device)  # type: ignore[arg-type]
            batch["features_next"] = torch.stack([r.features_next for r in self.rows]).to(device)  # type: ignore[arg-type]
        return batch


def ada_adapt_step(
    model: LegalBlendLoRA,
    buffer: OnlineReplayBuffer,
    *,
    opt: torch.optim.Optimizer,
    cfg: AdaConfig,
    device: torch.device,
) -> dict[str, float]:
    """One (or grad_steps) self-supervised update — AdaJEPA L_ada.

    Loss = BCE(change_logit[taken_a], did_change)
         + optional latent consistency on fused (stopgrad next)
    Only LoRA params should be in opt when cfg.lora_only.
    """
    if len(buffer) == 0:
        return {"loss": 0.0, "n": 0}
    model.train()
    total = 0.0
    for _ in range(max(1, int(cfg.grad_steps))):
        batch = buffer.as_batch(device)
        out = model(batch["frame"], batch["features"])
        losses = model.loss(
            out,
            action_id=batch["action_id"],
            x=batch["x"],
            y=batch["y"],
            did_change=batch["did_change"],
        )
        loss = losses["loss"]
        # optional JEPA-flavored latent consistency if next frame available
        if "frame_next" in batch and "features_next" in batch:
            with torch.no_grad():
                out_next = model(batch["frame_next"], batch["features_next"])
                z_next = out_next["fused"].detach()
            # predict: fused should move more when did_change=1
            z = out["fused"]
            # cosine pull when no change, mild when change (don't force identity on big deltas)
            cos = F.cosine_similarity(z, z_next, dim=-1)
            # target cos ~ 1 when no change, ~0.5 when change
            target = 1.0 - 0.5 * batch["did_change"]
            lat_loss = F.mse_loss(cos, target)
            loss = loss + 0.15 * lat_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        total += float(loss.detach())
    model.eval()
    return {"loss": total / max(1, cfg.grad_steps), "n": len(buffer)}


def build_ada_optimizer(model: LegalBlendLoRA, cfg: AdaConfig) -> torch.optim.Optimizer:
    if cfg.lora_only:
        params = list(lora_parameters(model))
        if not params:
            params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.0)


class AdaBlendPolicy:
    """Plan (horizon-1) → act → observe → adapt LoRA → replan next call.

    Drop-in richer policy for collect / live play. Offline train still builds
    the base checkpoint; this is the AdaJEPA closed loop on top.
    """

    def __init__(
        self,
        ckpt_path: Path | str | None = None,
        *,
        model: LegalBlendLoRA | None = None,
        ada: AdaConfig | None = None,
        adapt: bool = True,
        device: str | None = None,
    ) -> None:
        self.ada = ada or AdaConfig()
        self.adapt_enabled = bool(adapt)
        self.device = torch.device(
            device or self.ada.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if model is not None:
            self.model = model.to(self.device)
        else:
            if ckpt_path is None:
                raise ValueError("need ckpt_path or model")
            payload = torch.load(Path(ckpt_path), map_location=self.device, weights_only=False)
            cfg = LegalBlendConfig(**(payload.get("config") or {}))
            self.model = LegalBlendLoRA(cfg).to(self.device)
            self.model.load_state_dict(payload["model"])
        self.model.eval()
        if self.ada.lora_only:
            # freeze non-LoRA for online phase (AdaJEPA late-layer style)
            for n, p in self.model.named_parameters():
                p.requires_grad = "lora_" in n
        self.opt = build_ada_optimizer(self.model, self.ada)
        self.buffer = OnlineReplayBuffer(self.ada.buffer_size)
        self.game_id = ""
        self.adapt_log: list[dict[str, Any]] = []
        self._last_feats: np.ndarray | None = None
        self._last_frame: np.ndarray | None = None
        self._last_action: Action | None = None

    def reset_for_game(self, game_id: str) -> None:
        if self.game_id and self.game_id != game_id:
            self.buffer.wipe()
            self.adapt_log.clear()
        self.game_id = game_id

    @torch.no_grad()
    def plan(
        self,
        frame: np.ndarray,
        features: np.ndarray,
        available_actions: list[int] | None = None,
    ) -> Action:
        """Horizon-1 MPC: pick action maximizing blended change/policy score."""
        self.model.eval()
        ft = torch.from_numpy(np.asarray(frame, dtype=np.int16)).long().unsqueeze(0).to(self.device)
        ff = torch.from_numpy(np.asarray(features, dtype=np.float32)).unsqueeze(0).to(self.device)
        out = self.model(ft, ff)
        change = out["change_logits"][0].detach().cpu().numpy()
        policy = out["policy_logits"][0].detach().cpu().numpy()
        w_c = float(self.ada.plan_change_weight)
        w_p = float(self.ada.plan_policy_weight)
        score = w_c * change + w_p * policy
        avail = set(int(a) for a in (available_actions or [1, 2, 3, 4, 5, 6]))
        best_aid, best_s, best_xy = 1, -1e9, (0, 0)
        for aid in range(1, 6):
            if aid not in avail:
                continue
            s = float(score[aid - 1])
            if s > best_s:
                best_s, best_aid = s, aid
        if 6 in avail:
            click = score[GOOSE_SIMPLE:]
            idx = int(np.argmax(click))
            y, x = idx // GRID, idx % GRID
            s = float(click[idx])
            if s > best_s:
                best_s, best_aid, best_xy = s, 6, (x, y)
        if best_aid == 6:
            return Action(6, x=best_xy[0], y=best_xy[1], source="ada_plan")
        return Action(best_aid, source="ada_plan")

    def decide(
        self,
        frame: np.ndarray,
        *,
        game_id: str = "",
        transitions: list[Transition] | None = None,
        available_actions: list[int] | None = None,
        levels_completed: int = 0,
    ) -> Action:
        if game_id:
            self.reset_for_game(game_id)
        feats = encode_legal_features(
            frame,
            game_id=self.game_id or game_id,
            transitions=transitions,
            available_actions=available_actions,
            levels_completed=levels_completed,
            run_consume=True,
        )
        action = self.plan(frame, feats, available_actions)
        self._last_frame = np.asarray(frame).copy()
        self._last_feats = feats
        self._last_action = action
        return action

    def observe_and_adapt(
        self,
        frame_after: np.ndarray,
        *,
        did_change: bool | None = None,
        features_after: np.ndarray | None = None,
        transitions: list[Transition] | None = None,
        levels_completed: int = 0,
    ) -> dict[str, float]:
        """Store transition and run AdaJEPA-style LoRA adapt. Call after env.step."""
        if self._last_frame is None or self._last_action is None or self._last_feats is None:
            return {"loss": 0.0, "n": 0, "skipped": 1.0}
        before = self._last_frame
        action = self._last_action
        feats_b = self._last_feats
        after = np.asarray(frame_after)
        if did_change is None:
            did_change = not np.array_equal(before, after)
        if features_after is None:
            features_after = encode_legal_features(
                after,
                game_id=self.game_id,
                transitions=transitions,
                levels_completed=levels_completed,
                run_consume=True,
            )
        row = BufferRow(
            frame=torch.from_numpy(np.asarray(before, dtype=np.int16)).long(),
            features=torch.from_numpy(np.asarray(feats_b, dtype=np.float32)),
            action_id=int(action.action_id),
            x=int(action.x) if action.x is not None else 0,
            y=int(action.y) if action.y is not None else 0,
            did_change=1.0 if did_change else 0.0,
            frame_next=torch.from_numpy(np.asarray(after, dtype=np.int16)).long(),
            features_next=torch.from_numpy(np.asarray(features_after, dtype=np.float32)),
        )
        self.buffer.add(row)
        if not self.adapt_enabled or len(self.buffer) < 1:
            return {"loss": 0.0, "n": float(len(self.buffer))}
        stats = ada_adapt_step(self.model, self.buffer, opt=self.opt, cfg=self.ada, device=self.device)
        self.adapt_log.append(stats)
        return stats
