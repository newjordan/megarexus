"""External hemisphere full stack: world prediction + judge Internal.

Composes existing ExternalHemisphere. Learns only on real (s, a, s′) via
learn_step. Records disagreement between Internal dream and world pred.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hemispheres import ExternalHemisphere
from .hemisphere_i import _as_frame_tensor, align_actions
from .world_model_plug import WorldModelPlug


class ExternalHemisphereFull(nn.Module):
    """Full External path: encode → pred_next → judge_I / disagreement.

    Optional frozen WorldModelPlug (LP/wide dynamics) enriches world_summary and
    blends into pred_next when available.
    """

    def __init__(
        self,
        net: ExternalHemisphere | None = None,
        emb: int = 128,
        n_actions: int = 8,
        device: str | torch.device = "cpu",
        lr: float = 1e-3,
        world_ckpt: str | None = None,
        use_world_model: bool = True,
    ) -> None:
        super().__init__()
        self.emb = emb
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.net = net or ExternalHemisphere(emb=emb, n_actions=n_actions)
        self.net.to(self.device)
        self._opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._last_disagreement: float = 0.0
        self._last_judge: float = 0.0
        self._history: list[dict[str, float]] = []
        self.world: WorldModelPlug | None = None
        if use_world_model:
            try:
                self.world = WorldModelPlug(device=self.device, ckpt_path=world_ckpt)
            except Exception:
                self.world = None

    def parameters_net(self):
        return self.net.parameters()

    def run(
        self,
        frame: Any,
        action: int | torch.Tensor,
        h_I: torch.Tensor | None = None,
        dream: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """E.run(frame, action, h_I?, dream?) → h_E, pred, judge_I, disagreement."""
        frames = _as_frame_tensor(frame, self.device)
        actions = align_actions(action, int(frames.shape[0]), self.n_actions, self.device)

        with torch.set_grad_enabled(self.net.training):
            out = self.net(frames, actions, h_I=h_I, dream_next=dream)

        # Frozen learned dynamics: encode imagined next board into same emb space
        wm_meta: dict[str, Any] = {"wm_status": 0.0}
        if self.world is not None and self.world.status == "ok":
            try:
                pred_board = self.world.predict_board(frames)
                if pred_board is not None:
                    with torch.no_grad():
                        h_wm = self.net.encode(pred_board)
                    # blend world-model geometry into pred_next (stop-grad on wm)
                    out["pred_next"] = 0.5 * out["pred_next"] + 0.5 * h_wm.detach()
                    # re-judge after blend if we have I dream
                    if h_I is not None and dream is not None:
                        out["judge_I"] = self.net.judge_internal(
                            h_I.detach(), dream.detach(), out["pred_next"]
                        )
                        out["disagreement"] = F.mse_loss(
                            dream.detach(), out["pred_next"].detach(), reduction="none"
                        ).mean(dim=-1)
                    wm_meta = {
                        "wm_status": 1.0,
                        "wm_backend": self.world.backend,
                        "wm_pred_norm": float(h_wm.norm(dim=-1).mean().item()),
                    }
            except Exception as exc:
                wm_meta = {"wm_status": 0.0, "wm_error": type(exc).__name__}

        judge = out.get("judge_I")
        disagree = out.get("disagreement")
        if judge is not None:
            self._last_judge = float(judge.detach().mean().item())
        else:
            self._last_judge = 0.0
        if disagree is not None:
            self._last_disagreement = float(disagree.detach().mean().item())
        else:
            self._last_disagreement = 0.0

        world_summary: dict[str, Any] = {
            "pred_norm": float(out["pred_next"].detach().norm(dim=-1).mean().item()),
            "h_E_norm": float(out["h_E"].detach().norm(dim=-1).mean().item()),
            "judge_I": self._last_judge,
            "disagreement": self._last_disagreement,
            "action": int(actions[0].item()),
            **wm_meta,
        }
        if self.world is not None:
            world_summary["wm_plug"] = self.world.summary()
        self._history.append(
            {"judge_I": self._last_judge, "disagreement": self._last_disagreement}
        )
        if len(self._history) > 256:
            self._history = self._history[-256:]

        return {
            "h_E": out["h_E"],
            "pred_next": out["pred_next"],
            "judge_I": judge,
            "disagreement": disagree,
            "world_summary": world_summary,
            "judge_I_scalar": self._last_judge,
            "disagreement_scalar": self._last_disagreement,
        }

    def learn_step(
        self,
        s: Any,
        a: int | torch.Tensor,
        s_prime: Any,
    ) -> dict[str, float]:
        """Train world model on one real (s, a, s′) transition only."""
        self.net.train()
        frames = _as_frame_tensor(s, self.device)
        frames_next = _as_frame_tensor(s_prime, self.device)
        if frames.shape[0] != frames_next.shape[0]:
            raise ValueError(
                f"s and s' batch sizes differ ({frames.shape[0]} vs {frames_next.shape[0]})"
            )
        actions = align_actions(a, int(frames.shape[0]), self.n_actions, self.device)

        pred = self.net.predict_next(frames, actions)
        target = self.net.encode(frames_next)
        loss = F.mse_loss(pred, target)

        self._opt.zero_grad(set_to_none=True)
        loss.backward()
        self._opt.step()
        return {"loss_E": float(loss.detach().item())}

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        h_I: torch.Tensor | None = None,
        dream_next: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Net-only forward for OscillationTrainer compatibility."""
        return self.net(frames, actions, h_I=h_I, dream_next=dream_next)
