"""Three-phase oscillation trainer: E learns reality, I dreams, C arbitrates.

Phase ownership is hard: only the active phase's parameters receive gradients.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F

from .central import CentralConfig, CentralCortex
from .hemispheres import HemisphereBundle


class Phase(str, Enum):
    E = "external"   # world learns real transitions
    I = "internal"   # dream matches world teacher
    C = "central"    # cortex arbitrates (LoRA)


@dataclass
class OscillationConfig:
    emb: int = 128
    n_actions: int = 8
    lr_E: float = 1e-3
    lr_I: float = 1e-3
    lr_C: float = 2e-3
    device: str = "cpu"
    # schedule: repeat E,I,C blocks
    phases: tuple[Phase, ...] = (Phase.E, Phase.I, Phase.C)


class OscillationTrainer:
    def __init__(
        self,
        hemis: HemisphereBundle | None = None,
        central: CentralCortex | None = None,
        cfg: OscillationConfig | None = None,
    ) -> None:
        self.cfg = cfg or OscillationConfig()
        self.device = torch.device(self.cfg.device)
        self.hemis = hemis or HemisphereBundle.create(
            emb=self.cfg.emb, n_actions=self.cfg.n_actions
        )
        self.central = central or CentralCortex(
            CentralConfig(emb=self.cfg.emb, n_actions=self.cfg.n_actions)
        )
        self.hemis.internal.to(self.device)
        self.hemis.external.to(self.device)
        self.central.to(self.device)

        self.opt_E = torch.optim.Adam(self.hemis.parameters_E(), lr=self.cfg.lr_E)
        self.opt_I = torch.optim.Adam(self.hemis.parameters_I(), lr=self.cfg.lr_I)
        self.opt_C = torch.optim.Adam(list(self.central.lora_parameters()), lr=self.cfg.lr_C)
        self._phase_i = 0

    @property
    def phase(self) -> Phase:
        return self.cfg.phases[self._phase_i % len(self.cfg.phases)]

    def advance_phase(self) -> Phase:
        self._phase_i += 1
        return self.phase

    def _set_requires_grad(self, phase: Phase) -> None:
        for p in self.hemis.parameters_I():
            p.requires_grad_(phase is Phase.I)
        for p in self.hemis.parameters_E():
            p.requires_grad_(phase is Phase.E)
        for p in self.central.parameters():
            p.requires_grad_(False)
        if phase is Phase.C:
            for p in self.central.lora_parameters():
                p.requires_grad_(True)
            # trunk frozen in Phase C so only LoRA adapts (AdaJEPA spirit)
            for p in self.central.trunk.parameters():
                p.requires_grad_(False)

    def step_batch(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        frames_next: torch.Tensor,
        phase: Phase | None = None,
        extra: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """One optimization step on a real transition batch.

        frames / frames_next: [B,H,W] long color indices.
        actions: [B] long action ids.
        extra: optional [B, F] or [F] operator/legal feature pad for Phase C
               (padded/truncated to CentralConfig.extra_feats inside central).
        """
        phase = phase or self.phase
        self._set_requires_grad(phase)
        frames = frames.to(self.device)
        actions = actions.to(self.device)
        frames_next = frames_next.to(self.device)
        if extra is not None:
            extra = extra.to(self.device)
            if extra.dim() == 1:
                extra = extra.unsqueeze(0).expand(frames.shape[0], -1)
            elif extra.shape[0] == 1 and frames.shape[0] > 1:
                extra = extra.expand(frames.shape[0], -1)

        # Always forward both hemispheres; gradient flow is gated by requires_grad
        out_I = self.hemis.internal(frames, actions)
        out_E = self.hemis.external(
            frames, actions, h_I=out_I["h_I"], dream_next=out_I["dream_next"]
        )
        # target embedding of real next frame (shared geometry)
        with torch.set_grad_enabled(phase is Phase.E or phase is Phase.I):
            # encode next with the hemisphere that is allowed to train for target
            if phase is Phase.E:
                target = self.hemis.external.encode(frames_next)
            else:
                # teacher target for I: stop-grad external prediction OR real next via E encoder
                with torch.no_grad():
                    target = self.hemis.external.encode(frames_next)

        metrics: dict[str, Any] = {"phase": phase.value}

        if phase is Phase.E:
            self.opt_E.zero_grad(set_to_none=True)
            loss = F.mse_loss(out_E["pred_next"], target)
            loss.backward()
            self.opt_E.step()
            metrics["loss_E"] = float(loss.detach())
            metrics["loss"] = metrics["loss_E"]

        elif phase is Phase.I:
            self.opt_I.zero_grad(set_to_none=True)
            # Internal dreams toward real next embedding (external is frozen teacher geometry)
            loss = F.mse_loss(out_I["dream_next"], target.detach())
            # optional: also match external's pred_next (world model distillation)
            with torch.no_grad():
                teacher_pred = out_E["pred_next"]
            loss = loss + 0.5 * F.mse_loss(out_I["dream_next"], teacher_pred.detach())
            loss.backward()
            self.opt_I.step()
            metrics["loss_I"] = float(loss.detach())
            metrics["loss"] = metrics["loss_I"]
            metrics["mean_disagreement"] = float(out_E["disagreement"].detach().mean())

        else:  # Phase.C
            self.opt_C.zero_grad(set_to_none=True)
            # stop-grad hemispheres: center sees frozen features
            h_I = out_I["h_I"].detach()
            h_E = out_E["h_E"].detach()
            judge = out_E.get("judge_I")
            if judge is not None:
                judge = judge.detach()
            disagree = out_E.get("disagreement")
            if disagree is not None:
                disagree = disagree.detach()
            extra_c = extra.detach() if extra is not None else None
            out_C = self.central(
                h_I, h_E, judge_I=judge, disagreement=disagree, extra=extra_c
            )
            # weak SSL: predict taken action
            loss_pol = F.cross_entropy(out_C["action_logits"], actions.long())
            # after-the-fact: trust should prefer E when disagreement high
            # target trust_E high when disagree > median
            with torch.no_grad():
                d = disagree if disagree is not None else torch.zeros(actions.shape[0], device=self.device)
                prefer_E = (d > d.median()).float()
                trust_target = torch.stack([1.0 - prefer_E, prefer_E], dim=-1)
            loss_trust = F.mse_loss(out_C["trust"], trust_target)
            loss = loss_pol + 0.2 * loss_trust
            loss.backward()
            self.opt_C.step()
            metrics["loss_C"] = float(loss.detach())
            metrics["loss_pol"] = float(loss_pol.detach())
            metrics["loss_trust"] = float(loss_trust.detach())
            metrics["loss"] = metrics["loss_C"]
            metrics["trust_E_mean"] = float(out_C["trust"][:, 1].detach().mean())

        return metrics

    def train_cycle(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        frames_next: torch.Tensor,
        steps_per_phase: int = 1,
        extra: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        """Run the fixed 3-phase schedule once (E → I → C by default)."""
        logs = []
        for phase in self.cfg.phases:
            for _ in range(steps_per_phase):
                logs.append(
                    self.step_batch(
                        frames, actions, frames_next, phase=phase, extra=extra
                    )
                )
        return logs
