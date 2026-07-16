"""Minimal LoRA for Linear layers (no peft dependency required for core train)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """y = x W_frozen^T + scale * x A^T B^T  (low-rank adapter)."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear expects nn.Linear")
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
        # x @ A^T @ B^T
        update = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.scale * update

    def trainable_parameters(self):
        yield self.lora_A
        yield self.lora_B


def swap_linear_with_lora(module: nn.Module, rank: int = 8, alpha: float = 16.0) -> int:
    """Replace child Linear modules with LoRALinear in-place. Returns count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
            count += 1
        else:
            count += swap_linear_with_lora(child, rank=rank, alpha=alpha)
    return count


def lora_parameters(module: nn.Module):
    for m in module.modules():
        if isinstance(m, LoRALinear):
            yield from m.trainable_parameters()
