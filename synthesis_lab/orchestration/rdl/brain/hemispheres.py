"""Internal (dream/propose) and External (world/judge) hemispheres.

v0: lightweight convolutional trunks that share a frame interface so the
oscillation trainer can run without loading full LP/wide/DreamKernel stacks.
Swap `encode` / `predict_next` for real plugs later (see DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class _FrameEncoder(nn.Module):
    """64x64 int frame → embedding. Expects long/int class indices or float planes."""

    def __init__(self, num_colors: int = 16, emb: int = 128) -> None:
        super().__init__()
        self.num_colors = num_colors
        self.emb = emb
        self.net = nn.Sequential(
            nn.Conv2d(num_colors, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(96, emb)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # frames: [B,H,W] long or [B,C,H,W] float
        if frames.dim() == 3:
            x = F.one_hot(frames.clamp(0, self.num_colors - 1).long(), self.num_colors)
            x = x.permute(0, 3, 1, 2).float()
        else:
            x = frames.float()
        h = self.net(x).flatten(1)
        return self.proj(h)


class _ActionEmbed(nn.Module):
    def __init__(self, n_actions: int = 8, emb: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_actions, emb)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.emb(actions.long().clamp(0, self.emb.num_embeddings - 1))


class InternalHemisphere(nn.Module):
    """Dream / propose side: imagines next frame latent from (s, a)."""

    def __init__(self, emb: int = 128, n_actions: int = 8) -> None:
        super().__init__()
        self.encoder = _FrameEncoder(emb=emb)
        self.act = _ActionEmbed(n_actions=n_actions, emb=32)
        self.dream = nn.Sequential(
            nn.Linear(emb + 32, emb),
            nn.ReLU(inplace=True),
            nn.Linear(emb, emb),
        )
        self.plan_head = nn.Linear(emb, n_actions)  # propose logits (hypothesis)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def dream_next(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        h = self.encode(frames)
        a = self.act(actions)
        return self.dream(torch.cat([h, a], dim=-1))

    def propose_logits(self, frames: torch.Tensor) -> torch.Tensor:
        return self.plan_head(self.encode(frames))

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        h_i = self.encode(frames)
        dream = self.dream(torch.cat([h_i, self.act(actions)], dim=-1))
        return {
            "h_I": h_i,
            "dream_next": dream,
            "propose_logits": self.plan_head(h_i),
        }


class ExternalHemisphere(nn.Module):
    """World / judge side: predicts real transitions; scores Internal honesty."""

    def __init__(self, emb: int = 128, n_actions: int = 8) -> None:
        super().__init__()
        self.encoder = _FrameEncoder(emb=emb)
        self.act = _ActionEmbed(n_actions=n_actions, emb=32)
        self.predict = nn.Sequential(
            nn.Linear(emb + 32, emb),
            nn.ReLU(inplace=True),
            nn.Linear(emb, emb),
        )
        self.judge_head = nn.Sequential(
            nn.Linear(emb * 3, emb),
            nn.ReLU(inplace=True),
            nn.Linear(emb, 1),
        )

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def predict_next(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        h = self.encode(frames)
        a = self.act(actions)
        return self.predict(torch.cat([h, a], dim=-1))

    def judge_internal(
        self,
        h_I: torch.Tensor,
        dream_next: torch.Tensor,
        pred_next: torch.Tensor,
    ) -> torch.Tensor:
        """Higher score → External thinks Internal dream matches world pred."""
        # stop-grad on judge inputs from I when used as pure critic signal
        x = torch.cat([h_I, dream_next, pred_next], dim=-1)
        return self.judge_head(x).squeeze(-1)

    def forward(
        self,
        frames: torch.Tensor,
        actions: torch.Tensor,
        h_I: torch.Tensor | None = None,
        dream_next: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h_e = self.encode(frames)
        pred = self.predict(torch.cat([h_e, self.act(actions)], dim=-1))
        out: dict[str, torch.Tensor] = {"h_E": h_e, "pred_next": pred}
        if h_I is not None and dream_next is not None:
            out["judge_I"] = self.judge_internal(h_I.detach(), dream_next.detach(), pred)
            out["disagreement"] = F.mse_loss(dream_next.detach(), pred, reduction="none").mean(dim=-1)
        return out


@dataclass
class HemisphereBundle:
    internal: InternalHemisphere
    external: ExternalHemisphere

    @classmethod
    def create(cls, emb: int = 128, n_actions: int = 8) -> "HemisphereBundle":
        return cls(
            internal=InternalHemisphere(emb=emb, n_actions=n_actions),
            external=ExternalHemisphere(emb=emb, n_actions=n_actions),
        )

    def parameters_I(self):
        return self.internal.parameters()

    def parameters_E(self):
        return self.external.parameters()
