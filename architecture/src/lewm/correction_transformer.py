"""Causal-transformer Gaussian-correction block for LeWM.

Reference: ``docs/gaussian_correction_depth.md`` — sections "Formalizing the
Base Architecture and the Gaussian-Correction Extension" and "Practical
Design and Evaluation Protocols", plus the "Empirical Refinements" section
added 2026-05-10. The doc specifies the correction stack as a near-identity
residual Gaussianizer of the form ``G_k(H) = H + α_k * C_{ψ_k}(H)`` with
zero- or near-zero-initialized gates and zero-init final linears, applied to
BOTH the encoder/projector output ``h`` and the predictor/projector output
``ĥ``.

This module provides the **causal transformer** flavor of ``C_{ψ_k}``: a
pre-LN multi-head self-attention + position-wise MLP block. Three
conditioning modes are supported, selected via ``cond_mode``:

- ``"adaln"`` (default, backward-compatible): AdaLN-zero gated conditioning
  exactly mirroring :class:`lewm.predictor.AdaLNZeroBlock` — single shared
  ``Linear(cond_dim, 6 * dim)`` zero-initialized, producing the six tuples
  ``γ_a, β_a, α_a, γ_m, β_m, α_m``. Best inductive bias for **action-like /
  transient / per-step** cond signals (DiT classes, predictor.py's actions).
- ``"concat"``: concat-style conditioning — ``cond`` is appended to ``x``
  along the feature dim and projected through a fresh ``Linear(dim+cond_dim,
  dim)`` BEFORE the inner transformer block runs. No AdaLN. The outer scalar
  ``alpha`` gate is zero-initialized (identity-at-init via the outer gate;
  inner attn/MLP and ``cond_in_proj`` are default-init so ``delta != 0`` and
  ``alpha`` gets gradient at fresh init). Best inductive bias for
  **state-like / slow / per-frame-stable** cond signals (positional
  encodings, per-frame goal features).
- ``"both"``: concat-style AND AdaLN-zero, combined. ``cond_in_proj`` is
  default-init (provides cond-aware input perturbation), AdaLN is
  zero-init, and outer ``alpha`` is initialized to ``1.0``. At fresh init,
  AdaLN(cond) = 0 zero-gates ``α_a, α_m`` to zero, so all attn/MLP residuals
  vanish and ``delta = 0`` (identity preserved); gradient flows into AdaLN
  via ``alpha = 1`` and into ``cond_in_proj`` once ``α_a, α_m`` move off
  zero.

The cleanest formulation across all three modes is ``delta = inner_block_out
- inner_block_in``, where ``inner_block_in`` is ``cond_in_proj(cat([x,
cond]))`` for ``"concat"`` and ``"both"``, and ``x`` itself for ``"adaln"``.
This makes the residual ``out = x + alpha * delta`` strictly identity at
init for all three modes whenever the inner block is configured to produce
a zero residual at init (zero-gated AdaLN, or zero outer ``alpha`` for
``"concat"``).

Cond projection placement: ``cond_in_proj`` is applied **before** the
pre-LN normalizations of attn and MLP. Equivalently, ``cond_in_proj``
replaces ``x`` as the inner-block input, so all subsequent LN/attn/MLP
operations see a cond-injected version of x. This mirrors the
``MLPCorrectionBlock``'s "concat is the input to fc1" pattern from
``correction.py``: in both blocks, cond enters the data path BEFORE the
main transform and the final residual reads ``out = x + alpha * inner``.

Identity-at-init invariants per mode:

- ``cond_mode == "adaln"`` and ``cond_dim == 0``: outer ``alpha = 0``,
  ``adaln`` does not exist, ``cond_in_proj`` does not exist. Identity via
  outer gate. ``∂out/∂alpha = delta = attn_term + mlp_term`` is non-zero
  by random inner init.
- ``cond_mode == "adaln"`` and ``cond_dim > 0``: outer ``alpha = 1``,
  ``adaln`` is zero-init, ``cond_in_proj`` does not exist. AdaLN(cond) = 0
  zeroes the gates and ``delta = 0``. Gradient flows to ``adaln.bias``.
- ``cond_mode == "concat"``: outer ``alpha = 0``, ``adaln`` is None,
  ``cond_in_proj`` is default-init. Identity via outer gate. ``∂out/∂alpha
  = delta = attn_term + mlp_term`` (computed on cond-injected input) is
  non-zero by random inner init.
- ``cond_mode == "both"``: outer ``alpha = 1``, ``adaln`` is zero-init,
  ``cond_in_proj`` is default-init. AdaLN(cond) = 0 zeroes the gates so
  ``delta = 0``. Gradient flows to ``adaln.bias`` and to ``cond_in_proj``
  weights via the chain through attn/MLP once gates start moving.

Operational rule (from ``docs/gaussian_correction_depth.md`` Empirical
Refinements section, finding #3): use ``"adaln"`` for action-like cond
signals, ``"concat"`` for state-like cond signals, ``"both"`` when both
inductive biases are useful. The "concat" mode is the architectural fix
for the Phase C₂.5 negative result (AdaLN regressing on positional-
encoding cond) — it provides the same expressive-attention path the
Transformer block had under "adaln", but with the conditioning style that
worked for the MLP block.

The block does NOT use stop-gradients: gradients flow through ``alpha``,
``cond_in_proj`` (when present), ``adaln`` (when present), and the inner
network, in keeping with LeWM's no-stop-grad convention.

Plug-in contract: ``CausalTransformerCorrectionBlock`` is a no-arg
constructible block via :func:`make_transformer_block_factory`, which
returns a factory closure suitable for the ``GPsi(..., block_factory=...)``
container in ``correction.py``. The factory forwards ``cond_mode`` so
callers can wire concat / both modes through the standard ``GPsi``
construction path.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


_VALID_COND_MODES = ("adaln", "concat", "both")


class CausalTransformerCorrectionBlock(nn.Module):
    """Single near-identity residual correction block.

    Inputs:

    - ``x``: ``[B, T, dim]`` token sequence (encoder or predictor latents)
    - ``cond``: optional conditioning. Either ``[B, T, cond_dim]`` (per-step,
      mirroring the predictor's per-step action embedding) or ``[B, cond_dim]``
      (broadcast across time). If ``cond_dim == 0`` no conditioning is used
      and ``cond`` must be ``None``.

    Output: ``[B, T, dim]`` with the same shape as ``x``. At initialization
    the output is **exactly** ``x`` for every supported ``cond_mode`` (see
    module docstring for the per-mode identity-prior mechanism).
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        cond_dim: int = 0,
        cond_mode: str = "adaln",
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} not divisible by num_heads {num_heads}"
            )
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio}")
        if cond_dim < 0:
            raise ValueError(f"cond_dim must be >= 0, got {cond_dim}")
        if cond_mode not in _VALID_COND_MODES:
            raise ValueError(
                f"cond_mode must be one of {_VALID_COND_MODES}, got "
                f"{cond_mode!r}"
            )
        if cond_mode in ("concat", "both") and cond_dim == 0:
            raise ValueError(
                f"cond_mode={cond_mode!r} requires cond_dim > 0, got "
                f"cond_dim={cond_dim}"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.cond_dim = cond_dim
        self.cond_mode = cond_mode

        # Pre-LN affine-free norms — mirrors predictor.AdaLNZeroBlock so that
        # AdaLN can solely own the per-position scale/shift when enabled.
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)

        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout,
        )
        # No zero-init on out_proj — see module docstring for why double
        # zero-init (outer alpha=0 + inner final linear=0) traps the block.

        hidden = int(dim * mlp_ratio)
        self.mlp_in = nn.Linear(dim, hidden)
        self.mlp_act = nn.GELU()
        self.mlp_drop = nn.Dropout(dropout)
        self.mlp_out = nn.Linear(hidden, dim)
        self.mlp_out_drop = nn.Dropout(dropout)
        # No zero-init on mlp_out — same reason as out_proj above.

        # Concat-style cond input projector (only for "concat" and "both").
        # Default PyTorch init — must be NON-zero so that ``inner_block_in =
        # cond_in_proj(cat([x, cond]))`` is a non-trivial perturbation of
        # ``x`` at fresh init, giving ``alpha`` (in "concat" mode) a non-zero
        # gradient.
        self.cond_in_proj: nn.Linear | None
        if cond_mode in ("concat", "both"):
            self.cond_in_proj = nn.Linear(dim + cond_dim, dim)
        else:
            self.cond_in_proj = None

        # AdaLN conditioning. Initialization depends on cond_mode and cond_dim:
        # - ("adaln", cond_dim==0): no AdaLN; outer alpha = 0 (identity via
        #   outer gate). delta = inner attn/mlp residuals (non-zero by random
        #   inner init), so ∂out/∂alpha is non-zero and alpha trains.
        # - ("adaln", cond_dim>0): AdaLN zero-init AND outer alpha = 1.0.
        #   AdaLN(cond) = 0 → modulation = 0 → delta = 0 (identity).
        #   Gradient flows to AdaLN.bias since alpha=1 doesn't gate it to zero.
        # - ("concat", cond_dim>0): no AdaLN; outer alpha = 0; cond_in_proj is
        #   default-init. Identity via outer gate. delta = attn_term +
        #   mlp_term computed on cond-injected input (non-zero), so alpha
        #   trains. cond_in_proj gets gradient once alpha moves off zero.
        # - ("both", cond_dim>0): AdaLN zero-init, outer alpha = 1.0,
        #   cond_in_proj default-init. AdaLN(cond) = 0 zero-gates the
        #   inner residuals → delta = 0 (identity). Gradient flows to
        #   AdaLN.bias and (via alpha=1 and the modulation chain) to
        #   cond_in_proj once gates move off zero.
        # The bad combinations were active prior to 2026-05-10:
        #   * outer alpha=0 AND inner final linear=0 → trapped at identity
        #     (gradient-vanishing).
        #   * AdaLN-only on state-like cond (Phase C₂.5) → catastrophic
        #     regression because AdaLN is the wrong inductive bias for
        #     state signals. Concat / both modes are the architectural fix.
        self.adaln: nn.Linear | None
        if cond_mode == "adaln" and cond_dim > 0:
            self.adaln = nn.Linear(cond_dim, 6 * dim)
            nn.init.zeros_(self.adaln.weight)
            nn.init.zeros_(self.adaln.bias)
            self.alpha = nn.Parameter(torch.ones(1))
        elif cond_mode == "both":
            # cond_dim > 0 already validated above.
            self.adaln = nn.Linear(cond_dim, 6 * dim)
            nn.init.zeros_(self.adaln.weight)
            nn.init.zeros_(self.adaln.bias)
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            # ("adaln", cond_dim==0) or ("concat", cond_dim>0).
            self.adaln = None
            self.alpha = nn.Parameter(torch.zeros(1))

    def _resolve_cond(
        self, cond: torch.Tensor | None, b: int, t: int,
    ) -> torch.Tensor | None:
        """Validate and broadcast ``cond`` to ``[B, T, cond_dim]``.

        Returns ``None`` if ``cond_dim == 0`` (no conditioning).
        """

        if self.cond_dim == 0:
            if cond is not None:
                raise ValueError(
                    "cond must be None when cond_dim == 0, got tensor with "
                    f"shape {tuple(cond.shape)}"
                )
            return None
        if cond is None:
            raise ValueError(
                f"cond is required when cond_dim={self.cond_dim} > 0"
            )
        if cond.dim() == 2:
            if cond.shape != (b, self.cond_dim):
                raise ValueError(
                    f"cond must be [B={b}, cond_dim={self.cond_dim}] when 2-D, "
                    f"got {tuple(cond.shape)}"
                )
            cond = cond.unsqueeze(1).expand(b, t, self.cond_dim)
        elif cond.dim() == 3:
            if cond.shape != (b, t, self.cond_dim):
                raise ValueError(
                    f"cond must be [B={b}, T={t}, cond_dim={self.cond_dim}] "
                    f"when 3-D, got {tuple(cond.shape)}"
                )
        else:
            raise ValueError(
                f"cond must be 2-D [B, cond_dim] or 3-D [B, T, cond_dim], "
                f"got {cond.dim()}-D tensor with shape {tuple(cond.shape)}"
            )
        return cond

    def _inner(
        self,
        x: torch.Tensor,
        cond_btD: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """The causal-transformer interior. Returns the **delta** that the
        outer residual will scale by ``alpha`` and add to ``x``.

        For ``cond_mode in ("concat", "both")``, the inner block's input is
        ``cond_in_proj(cat([x, cond]))`` (a cond-injected version of x), and
        ``delta = inner_out - inner_in``. Otherwise the inner block's input
        is ``x`` and ``delta = inner_out - x`` (the existing AdaLN-only
        formulation). Either way, ``delta = 0`` at fresh init whenever the
        inner block is configured to produce a zero residual at init,
        guaranteeing the outer ``out = x + alpha * delta`` is exactly
        identity.
        """

        # Compute the inner-block input. For concat / both modes we inject
        # cond into x via cond_in_proj BEFORE pre-LN; the rest of the block
        # then operates on this cond-injected input.
        if self.cond_in_proj is not None and cond_btD is not None:
            inner_in = self.cond_in_proj(torch.cat([x, cond_btD], dim=-1))
        else:
            inner_in = x

        # AdaLN modulation (only for "adaln" with cond_dim>0 or "both").
        if self.adaln is not None and cond_btD is not None:
            modulation = self.adaln(cond_btD)  # [B, T, 6*dim]
            (
                gamma_a, beta_a, alpha_a,
                gamma_m, beta_m, alpha_m,
            ) = modulation.chunk(6, dim=-1)
        else:
            gamma_a = beta_a = alpha_a = None
            gamma_m = beta_m = alpha_m = None

        # Causal MHSA path (pre-LN, AdaLN-modulated if conditioned).
        x_norm = self.norm_attn(inner_in)
        if gamma_a is not None:
            x_in = x_norm * (1 + gamma_a) + beta_a
        else:
            x_in = x_norm
        attn_out, _ = self.attn(
            x_in, x_in, x_in, attn_mask=attn_mask, need_weights=False,
        )
        if alpha_a is not None:
            attn_term = alpha_a * attn_out
        else:
            attn_term = attn_out
        x_after_attn = inner_in + attn_term

        # Position-wise MLP path (pre-LN, AdaLN-modulated if conditioned).
        x_norm = self.norm_mlp(x_after_attn)
        if gamma_m is not None:
            x_in = x_norm * (1 + gamma_m) + beta_m
        else:
            x_in = x_norm
        mlp_hidden = self.mlp_drop(self.mlp_act(self.mlp_in(x_in)))
        mlp_out = self.mlp_out_drop(self.mlp_out(mlp_hidden))
        if alpha_m is not None:
            mlp_term = alpha_m * mlp_out
        else:
            mlp_term = mlp_out
        x_after_mlp = x_after_attn + mlp_term

        # The "inner" delta is the inner-block update minus the inner-block
        # input. For "adaln"-only this matches the prior formulation
        # (delta = x_after_mlp - x). For "concat" / "both" this captures
        # only the attn/mlp contributions, NOT the cond_in_proj
        # perturbation itself — that's the right choice for "both" (where
        # AdaLN gates must be able to zero the entire inner contribution at
        # init) and for "concat" (where the outer alpha gate handles
        # identity, and we want delta = attn + mlp to give alpha gradient).
        return x_after_mlp - inner_in

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"x must be [B, T, dim], got {tuple(x.shape)}"
            )
        b, t, d = x.shape
        if d != self.dim:
            raise ValueError(
                f"dim mismatch: block has dim={self.dim}, got x dim={d}"
            )

        cond_btD = self._resolve_cond(cond, b, t)

        if t > 1:
            attn_mask = torch.triu(
                torch.full((t, t), float("-inf"), device=x.device),
                diagonal=1,
            )
        else:
            attn_mask = None

        delta = self._inner(x, cond_btD, attn_mask)
        return x + self.alpha * delta


def make_transformer_block_factory(
    *,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    cond_dim: int = 0,
    cond_mode: str = "adaln",
) -> Callable[[], CausalTransformerCorrectionBlock]:
    """Return a no-arg factory that constructs a fresh correction block.

    This is the plug-in shape required by ``correction.GPsi(...,
    block_factory=...)``: a callable taking no arguments and returning a
    fresh ``nn.Module`` instance per call so each layer in the stack has
    independent parameters. ``cond_mode`` is forwarded so callers can pick
    the conditioning style (see ``CausalTransformerCorrectionBlock``).
    """

    def factory() -> CausalTransformerCorrectionBlock:
        return CausalTransformerCorrectionBlock(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            cond_dim=cond_dim,
            cond_mode=cond_mode,
        )

    return factory


def causal_rollout(
    *,
    encoder_fn: Callable[[torch.Tensor], torch.Tensor],
    predictor_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    gpsi: nn.Module,
    obs: torch.Tensor,
    action_emb: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the full causal corrected rollout from the gaussian-correction doc.

    Pipeline (matches the mermaid in ``gaussian_correction_depth.md``):

    1. Encode each observation frame to ``h_t`` using the (typically frozen)
       ``encoder_fn`` callable, which takes ``(B, C, H, W)`` and returns
       ``(B, D)``. We apply this per timestep and stack to ``(B, T, D)``.
    2. Predict next-latents ``ĥ_t`` via ``predictor_fn(h, action_emb)`` —
       interface matches :class:`AdaLNTransformerPredictor.forward`: takes
       ``(B, T, D)`` and ``(B, T, action_dim)``, returns ``(B, T, D)``.
    3. Apply ``gpsi`` (an ``nn.Module``, callable on ``(B, T, D)`` →
       ``(B, T, D)``) to BOTH ``h`` and ``ĥ`` to produce corrected
       ``g`` and ``ĝ``.

    Returns a dict with keys ``emb`` (= ``h``), ``next_emb`` (= ``ĥ``),
    ``emb_corrected`` (= ``g``), ``next_emb_corrected`` (= ``ĝ``).

    No stop-gradients anywhere — gradients flow through every step.
    """

    if obs.dim() != 5:
        raise ValueError(
            f"obs must be [B, T, C, H, W], got {tuple(obs.shape)}"
        )
    b, t, c, h_img, w_img = obs.shape

    if action_emb.dim() != 3 or action_emb.shape[:2] != (b, t):
        raise ValueError(
            f"action_emb must be [B={b}, T={t}, action_dim], "
            f"got {tuple(action_emb.shape)}"
        )

    # Step 1: per-frame encoder. Flatten (B, T) → (B*T) so a single
    # encoder_fn call handles the whole batch; reshape back to (B, T, D).
    obs_flat = obs.reshape(b * t, c, h_img, w_img)
    h_flat = encoder_fn(obs_flat)
    if h_flat.dim() != 2 or h_flat.shape[0] != b * t:
        raise ValueError(
            f"encoder_fn must return [B*T, D], got {tuple(h_flat.shape)} "
            f"for input batch of size {b * t}"
        )
    d = h_flat.shape[1]
    h = h_flat.reshape(b, t, d)

    # Step 2: causal predictor.
    h_hat = predictor_fn(h, action_emb)
    if h_hat.shape != h.shape:
        raise ValueError(
            f"predictor_fn must return shape matching h={tuple(h.shape)}, "
            f"got {tuple(h_hat.shape)}"
        )

    # Step 3: shared Gaussian-correction stack on both true and predicted
    # latent sequences.
    g = gpsi(h)
    if g.shape != h.shape:
        raise ValueError(
            f"gpsi must preserve shape={tuple(h.shape)}, got {tuple(g.shape)}"
        )
    g_hat = gpsi(h_hat)
    if g_hat.shape != h_hat.shape:
        raise ValueError(
            f"gpsi must preserve shape={tuple(h_hat.shape)}, "
            f"got {tuple(g_hat.shape)}"
        )

    return {
        "emb": h,
        "next_emb": h_hat,
        "emb_corrected": g,
        "next_emb_corrected": g_hat,
    }
