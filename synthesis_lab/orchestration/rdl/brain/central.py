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

from .hemisphere_i import EXTRA_FEAT_DIM


def _as_column(
    value: torch.Tensor | None,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a per-row scalar to ``[batch, 1]``."""
    if value is None:
        return torch.zeros(batch, 1, device=device, dtype=dtype)
    column = value.to(device=device, dtype=dtype).reshape(-1)
    if column.numel() == 1 and batch != 1:
        column = column.expand(batch)
    if int(column.numel()) != int(batch):
        raise ValueError(f"expected {batch} row values, got {int(column.numel())}")
    return column.reshape(batch, 1)


def _as_feature_rows(
    extra: torch.Tensor | None,
    batch: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast ``[F]`` or ``[1, F]`` operator features to ``[batch, width]``."""
    if extra is None:
        return torch.zeros(batch, width, device=device, dtype=dtype)
    rows = extra.to(device=device, dtype=dtype)
    if rows.dim() == 1:
        rows = rows.unsqueeze(0)
    if rows.dim() != 2:
        raise ValueError(f"extra features must be [F] or [B, F], got shape {tuple(rows.shape)}")
    if rows.shape[0] == 1 and batch != 1:
        rows = rows.expand(batch, int(rows.shape[-1]))
    if int(rows.shape[0]) != int(batch):
        raise ValueError(f"extra batch {int(rows.shape[0])} does not match frame batch {batch}")
    feat = int(rows.shape[-1])
    if feat < width:
        pad = torch.zeros(batch, width - feat, device=device, dtype=dtype)
        rows = torch.cat([rows.contiguous(), pad], dim=-1)
    elif feat > width:
        rows = rows[..., :width]
    return rows


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        if self.rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {self.rank}")
        self.scale = float(alpha) / float(self.rank)
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
    # input = h_I + h_E + diff + absdiff + judge + disagree + feat pad
    # Default matches the operator garden. A shorter vector is zero-padded;
    # a longer one is truncated, so this must stay equal to EXTRA_FEAT_DIM.
    extra_feats: int = EXTRA_FEAT_DIM


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
        j = _as_column(judge_I, b, h_I.device, h_I.dtype)
        d = _as_column(disagreement, b, h_I.device, h_I.dtype)
        feats = _as_feature_rows(extra, b, self.cfg.extra_feats, h_I.device, h_I.dtype)
        return torch.cat([h_I, h_E, diff, adiff, j, d, feats], dim=-1)

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
