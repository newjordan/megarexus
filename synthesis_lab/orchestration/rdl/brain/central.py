"""Central cortex: fuses both hemispheres; trainable LoRA on late layers.

Owned trainable node for megarexus (LegalBlend-style / CentralCortex).
Online AdaJEPA-style updates should touch only lora_parameters().
vLLM/Laguna-style runtimes are optional *serving* for checkpoints, not part of
the brain design. Dreamweaver-derived imagination/trust is in-scope via ports.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.bias = None
        self.lora_A = nn.Parameter(torch.zeros(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        update = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.scale * update


@dataclass
class CentralConfig:
    emb: int = 128
    n_actions: int = 8
    hidden: int = 256
    lora_rank: int = 8
    lora_alpha: float = 16.0
    # input = h_I + h_E + diff + absdiff + judge + disagree + feat_pad
    # full garden pad — MegaMind sets from hemisphere_i.EXTRA_FEAT_DIM (~159×16)
    extra_feats: int = 2544


class CentralCortex(nn.Module):
    """Fuse hemisphere states → action logits + trust + mode."""

    def __init__(self, cfg: CentralConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CentralConfig()
        c = self.cfg
        # h_I, h_E, h_I-h_E, |h_I-h_E|, judge(1), disagree(1), extra
        in_dim = c.emb * 4 + 2 + c.extra_feats
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, c.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(c.hidden, c.hidden),
            nn.ReLU(inplace=True),
        )
        # LoRA on late policy / trust heads only
        self.policy = LoRALinear(nn.Linear(c.hidden, c.n_actions), c.lora_rank, c.lora_alpha)
        self.trust = LoRALinear(nn.Linear(c.hidden, 2), c.lora_rank, c.lora_alpha)  # α_I, α_E
        self.mode = LoRALinear(nn.Linear(c.hidden, 4), c.lora_rank, c.lora_alpha)  # gas/brake/probe/dream

    def fuse_features(
        self,
        h_I: torch.Tensor,
        h_E: torch.Tensor,
        judge_I: torch.Tensor | None = None,
        disagreement: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = h_I.shape[0]
        diff = h_I - h_E
        adiff = diff.abs()
        if judge_I is None:
            judge_I = torch.zeros(b, device=h_I.device, dtype=h_I.dtype)
        if disagreement is None:
            disagreement = torch.zeros(b, device=h_I.device, dtype=h_I.dtype)
        if extra is None:
            extra = torch.zeros(b, self.cfg.extra_feats, device=h_I.device, dtype=h_I.dtype)
        elif extra.shape[-1] < self.cfg.extra_feats:
            pad = torch.zeros(b, self.cfg.extra_feats - extra.shape[-1], device=h_I.device, dtype=h_I.dtype)
            extra = torch.cat([extra, pad], dim=-1)
        elif extra.shape[-1] > self.cfg.extra_feats:
            extra = extra[..., : self.cfg.extra_feats]
        j = judge_I.reshape(b, 1)
        d = disagreement.reshape(b, 1)
        return torch.cat([h_I, h_E, diff, adiff, j, d, extra], dim=-1)

    def forward(
        self,
        h_I: torch.Tensor,
        h_E: torch.Tensor,
        judge_I: torch.Tensor | None = None,
        disagreement: torch.Tensor | None = None,
        extra: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self.fuse_features(h_I, h_E, judge_I, disagreement, extra)
        z = self.trunk(x)
        trust_logits = self.trust(z)
        trust = F.softmax(trust_logits, dim=-1)
        return {
            "z": z,
            "action_logits": self.policy(z),
            "trust": trust,  # [:,0]=α_I, [:,1]=α_E
            "trust_logits": trust_logits,
            "mode_logits": self.mode(z),
        }

    def lora_parameters(self):
        for m in (self.policy, self.trust, self.mode):
            yield m.lora_A
            yield m.lora_B
