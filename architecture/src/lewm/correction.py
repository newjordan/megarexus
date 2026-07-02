"""Gaussian Correction Depth (GCD) — paper-doc-faithful implementation.

Architectural blueprint: ``docs/gaussian_correction_depth.md``.

The GCD module ``G_ψ`` is a near-identity residual stack applied AFTER both the
encoder/projector output ``h_t`` and the predictor/projector output ``ĥ_t``,
with **shared weights** between the two applications (the caller's job — this
module is a single ``nn.Module`` and the caller invokes it twice).

::

    G_ψ = G_K ∘ ... ∘ G_1,    G_k(H) = H + α_k · C_{ψ_k}(LN(H), cond)

where ``C_{ψ_k}`` is a small inner module (residual MLP here; causal-transformer
variant lives in ``correction_transformer.py``) and ``α_k`` is a scalar gate
initialized to 0 so that **at init the entire stack is the identity function**.

Design notes (deviation from the doc, with rationale):

The doc's "Initialization" recommendation is "zero-initialize the last linear
in each correction block AND initialize gates α_k ≈ 0". Doing both at exactly
zero causes a gradient-vanishing pathology — every parameter inside the block
has gradient ∝ α and α has gradient ∝ inner(x), so if inner(x) is also zero
(because the last linear is zero) then *all* gradients vanish at init. This
matches the pre-existing ``predictor.py:AdaLNZeroBlock`` choice: the gate is
zero-initialized, but the inner module uses default init so that ``mlp_out``
is random-non-zero, giving ``α`` a non-zero gradient at the start. We follow
that pattern here for the same reason.

Default block type is ``MLPCorrectionBlock`` (cheaper, no temporal context).
For temporal-context calibration, swap in a ``CausalTransformerCorrectionBlock``
from ``correction_transformer.py`` via the ``block_factory`` argument.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


class MLPCorrectionBlock(nn.Module):
    """Pre-LN residual zero-gated MLP block.

    Forward:

    .. math::

        \\text{out} = x + \\alpha \\cdot W_2 \\cdot \\sigma(W_1 \\cdot (\\text{LN}(x) \\Vert d))

    where ``α`` is a single learnable scalar initialized to 0 (so the whole block
    equals the identity at init), and ``d`` is an optional conditioning token of
    width ``cond_dim`` concatenated to the post-LN features.

    Shape contract:

    - ``x``: ``(..., dim)`` — leading dims arbitrary (works for ``(B, D)`` or
      ``(B, T, D)`` etc.).
    - ``cond``: ``(..., cond_dim)`` — must match ``x``'s shape except on the last
      axis. For per-batch global conditioning when ``x`` is ``(B, T, D)``, the
      caller must broadcast: ``cond.unsqueeze(1).expand(B, T, cond_dim)``.

    Output: same shape as ``x``.
    """

    def __init__(
        self,
        *,
        dim: int,
        cond_dim: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if cond_dim < 0:
            raise ValueError("cond_dim must be >= 0")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be > 0")
        self.dim = dim
        self.cond_dim = cond_dim
        self.norm = nn.LayerNorm(dim)
        in_dim = dim + cond_dim
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        # Scalar residual gate, init = 0 → block is identity at init.
        # We do NOT zero-init fc2 (see module docstring "Design notes").
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} does not match block dim {self.dim}"
            )
        h = self.norm(x)
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError(
                    f"this block has cond_dim={self.cond_dim} but no cond was provided"
                )
            if cond.shape[-1] != self.cond_dim:
                raise ValueError(
                    f"cond last dim {cond.shape[-1]} does not match cond_dim {self.cond_dim}"
                )
            if cond.shape[:-1] != x.shape[:-1]:
                raise ValueError(
                    f"cond leading shape {tuple(cond.shape[:-1])} must match x's "
                    f"leading shape {tuple(x.shape[:-1])} (broadcast outside if needed)"
                )
            h = torch.cat([h, cond], dim=-1)
        elif cond is not None:
            raise ValueError(
                f"cond was provided but this block has cond_dim=0"
            )
        inner = self.fc2(self.drop(self.act(self.fc1(h))))
        return x + self.alpha * inner


def make_mlp_block_factory(
    *,
    dim: int,
    cond_dim: int = 0,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
) -> Callable[[], MLPCorrectionBlock]:
    """Returns a no-arg callable that builds a fresh ``MLPCorrectionBlock``."""

    def factory() -> MLPCorrectionBlock:
        return MLPCorrectionBlock(
            dim=dim, cond_dim=cond_dim, mlp_ratio=mlp_ratio, dropout=dropout,
        )
    return factory


class GPsi(nn.Module):
    """The shared post-projector Gaussian Correction Depth stack.

    Composes ``num_blocks`` residual zero-gated correction blocks. With
    ``num_blocks=0`` the module is the literal identity (returns its input
    unchanged); with ``num_blocks >= 1`` and freshly initialized blocks it is
    *exactly* the identity function up to floating-point precision.

    Block type is determined by ``block_factory``. The default factory builds
    ``MLPCorrectionBlock`` instances. For causal-transformer correction blocks,
    pass a ``make_transformer_block_factory(...)`` from
    ``correction_transformer.py``.

    The same ``GPsi`` instance is meant to be applied to BOTH the encoder's
    output ``h_t`` and the predictor's output ``ĥ_t`` (per the GCD doc's
    "shared weights" requirement). The caller does this; this class itself
    is just a forward over a stack of blocks.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_blocks: int = 1,
        cond_dim: int = 0,
        block_factory: Callable[[], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if num_blocks < 0:
            raise ValueError("num_blocks must be >= 0")
        if cond_dim < 0:
            raise ValueError("cond_dim must be >= 0")
        self.dim = dim
        self.num_blocks = num_blocks
        self.cond_dim = cond_dim
        if block_factory is None:
            block_factory = make_mlp_block_factory(dim=dim, cond_dim=cond_dim)
        self.blocks = nn.ModuleList(
            [block_factory() for _ in range(num_blocks)]
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} does not match GPsi dim {self.dim}"
            )
        for block in self.blocks:
            x = block(x, cond)
        return x

    @property
    def is_identity_at_init(self) -> bool:
        """True if every block has its scalar gate at 0 — i.e. ``GPsi(x) == x``.

        Useful as a sanity check before calibration training begins.
        """

        for block in self.blocks:
            alpha = getattr(block, "alpha", None)
            if alpha is None or float(alpha.detach().abs().max()) > 0.0:
                return False
        return True
