"""AdaJEPA-style online LoRA adapt for CentralCortex (owned megarexus node).

After real env transitions: buffer (s,a,s') → 1 grad step on central LoRA only.
No reward labels. Wipe buffer on new game_id.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import torch
import torch.nn.functional as F

from .central import CentralCortex
from .hemispheres import ExternalHemisphere, InternalHemisphere


@dataclass
class AdaOnlineConfig:
    buffer_size: int = 8
    grad_steps: int = 1
    lr: float = 1e-3


class CentralAdaOnline:
    def __init__(
        self,
        central: CentralCortex,
        net_I: InternalHemisphere,
        net_E: ExternalHemisphere,
        cfg: AdaOnlineConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.central = central
        self.net_I = net_I
        self.net_E = net_E
        self.cfg = cfg or AdaOnlineConfig()
        self.device = torch.device(device)
        self.buffer: Deque[dict[str, torch.Tensor]] = deque(maxlen=self.cfg.buffer_size)
        self.game_id: str | None = None
        self.opt = torch.optim.Adam(list(self.central.lora_parameters()), lr=self.cfg.lr)
        self.steps_done = 0

    def wipe(self) -> None:
        self.buffer.clear()

    def set_game(self, game_id: str | None) -> None:
        if game_id != self.game_id:
            self.wipe()
            self.game_id = game_id

    def observe_transition(
        self,
        frame: torch.Tensor,
        action: int | torch.Tensor,
        frame_next: torch.Tensor,
        extra: torch.Tensor | None = None,
        game_id: str | None = None,
    ) -> dict[str, Any]:
        """Push transition; run adapt steps if buffer non-empty."""
        if game_id is not None:
            self.set_game(game_id)
        if frame.dim() == 2:
            frame = frame.unsqueeze(0)
        if frame_next.dim() == 2:
            frame_next = frame_next.unsqueeze(0)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([int(action)], dtype=torch.long)
        action = action.long().reshape(-1)
        if action.numel() == 1 and frame.shape[0] > 1:
            action = action.expand(frame.shape[0])

        for i in range(frame.shape[0]):
            row = {
                "frame": frame[i].detach().cpu().long(),
                "action": action[i].detach().cpu().long(),
                "frame_next": frame_next[i].detach().cpu().long(),
            }
            if extra is not None:
                e = extra[i] if extra.dim() > 1 else extra
                row["extra"] = e.detach().cpu().float()
            self.buffer.append(row)

        metrics: dict[str, Any] = {"buffer": len(self.buffer), "adapted": False}
        if len(self.buffer) < 2:
            return metrics

        loss_last = 0.0
        for _ in range(self.cfg.grad_steps):
            loss_last = self._adapt_step()
            self.steps_done += 1
        metrics["adapted"] = True
        metrics["loss"] = loss_last
        metrics["ada_steps"] = self.steps_done
        return metrics

    def _adapt_step(self) -> float:
        frames = torch.stack([r["frame"] for r in self.buffer]).to(self.device)
        actions = torch.stack([r["action"] for r in self.buffer]).to(self.device)
        frames_next = torch.stack([r["frame_next"] for r in self.buffer]).to(self.device)
        extras = None
        if all("extra" in r for r in self.buffer):
            extras = torch.stack([r["extra"] for r in self.buffer]).to(self.device)

        # freeze hemispheres — features only
        self.net_I.eval()
        self.net_E.eval()
        with torch.no_grad():
            h_I = self.net_I.encode(frames)
            dream = self.net_I.dream_next(frames, actions)
            out_E = self.net_E(frames, actions, h_I=h_I, dream_next=dream)
            h_E = out_E["h_E"]
            judge = out_E.get("judge_I")
            disagree = out_E.get("disagreement")
            # did_change SSL target
            did_change = (frames != frames_next).flatten(1).any(dim=1).float()

        # only LoRA trains
        for p in self.central.parameters():
            p.requires_grad_(False)
        for p in self.central.lora_parameters():
            p.requires_grad_(True)

        self.opt.zero_grad(set_to_none=True)
        out_C = self.central(
            h_I.detach(),
            h_E.detach(),
            judge_I=judge.detach() if judge is not None else None,
            disagreement=disagree.detach() if disagree is not None else None,
            extra=extras,
        )
        # weak policy SSL on taken action + tiny change proxy via mode gas vs brake
        loss = F.cross_entropy(out_C["action_logits"], actions)
        # prefer trust_E when frames differ a lot (world more informative)
        if disagree is not None:
            prefer_E = (disagree.detach() > disagree.detach().median()).float()
            trust_t = torch.stack([1.0 - prefer_E, prefer_E], dim=-1)
            loss = loss + 0.15 * F.mse_loss(out_C["trust"], trust_t)
        # mode: gas if changed, brake if not
        mode_t = torch.zeros(frames.shape[0], 4, device=self.device)
        mode_t[did_change > 0.5, 0] = 1.0  # gas
        mode_t[did_change <= 0.5, 1] = 1.0  # brake
        loss = loss + 0.1 * F.cross_entropy(out_C["mode_logits"], mode_t.argmax(dim=-1))
        loss.backward()
        self.opt.step()
        return float(loss.detach())
