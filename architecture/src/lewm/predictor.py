"""Action-conditioned predictor for LeWM (paper Section 3.1, eq LeWM).

Paper-exact configuration:

- 6 layers, 16 attention heads, 10% dropout (~10M params at hidden=192)
- Per-block AdaLN-zero (Peebles & Xie, DiT) with PER-POSITION action conditioning
- Causal mask so position ``t`` cannot attend to ``t+1`` (paper: "auto-regressively
  with temporal causal masking")
- Final projector matches encoder's MLP+BN exactly
- Returns the full ``(B, T, D)`` output, NOT just the last token. Position ``t`` of the
  output is the predicted next-latent ``ẑ_{t+1}`` from input ``z_t`` and action ``a_t``
  (with causal context over earlier positions)

Per-block AdaLN-zero per position::

    γ_a, β_a, α_a, γ_m, β_m, α_m = MLP(action_emb_t)            # per t
    x_t ← x_t + α_a · Attn((1 + γ_a) · LN(x_t) + β_a)          # causal attention
    x_t ← x_t + α_m · MLP_block((1 + γ_m) · LN(x_t) + β_m)

The 6-tuple modulation comes from a single ``Linear(action_dim, 6 * dim)``
initialized to zeros (weight + bias). Result: at init the block is exactly
identity, so action conditioning influences the predictor "progressively"
during training (paper).

**REX 3.0 enhancement**: optional goal conditioning. When ``goal_dim > 0``,
the predictor accepts a ``goal_emb`` (``[B, D]``) and ``goal_confidence``
(``[B]`` scalar 0-1). The goal embedding is broadcast across all T positions,
scaled by confidence, concatenated to the action embedding, and fed into the
AdaLN modulation. This lets the goal direction inform every transformer block::

    c_t = concat(action_emb_t, λ · goal_emb)                    # [B, T, D + D]
    γ_a, β_a, α_a, γ_m, β_m, α_m = MLP(c_t)                     # per t
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


PAPER_NUM_LAYERS = 6
PAPER_NUM_HEADS = 16
PAPER_HIDDEN_DIM = 192
PAPER_DROPOUT = 0.1


@dataclass(frozen=True)
class PredictorConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = PAPER_NUM_LAYERS
    num_heads: int = PAPER_NUM_HEADS
    mlp_ratio: float = 4.0
    dropout: float = PAPER_DROPOUT
    goal_dim: int = 0  # 0 = no goal conditioning (REX 2.0 behavior)
    use_swiglu: bool = False  # SwiGLU gated activation (LLaMA-style)

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError(
                f"latent_dim {self.latent_dim} not divisible by "
                f"num_heads {self.num_heads}"
            )
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.goal_dim < 0:
            raise ValueError("goal_dim must be >= 0")


class AdaLNZeroBlock(nn.Module):
    """Pre-LN transformer block with per-position AdaLN-zero action modulation.

    Inputs:

    - ``x``: ``[B, T, dim]``
    - ``action_emb``: ``[B, T, action_dim]`` per-step action embeddings
    - ``attn_mask``: optional ``[T, T]`` causal mask

    REX 3.0: when ``goal_dim > 0``, the block also accepts ``goal_emb``
    (``[B, 1, goal_dim]``) and ``goal_confidence`` (``[B, 1]`` scalar 0-1).
    The goal is broadcast across T and concatenated to action_emb before
    the AdaLN modulation.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        action_dim: int,
        goal_dim: int = 0,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} not divisible by num_heads {num_heads}"
            )
        self.use_swiglu = use_swiglu
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout,
        )
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        if use_swiglu:
            # SwiGLU: gate_proj + up_proj → SiLU(gate) * up → down_proj
            self.gate_proj = nn.Linear(dim, hidden, bias=False)
            self.up_proj = nn.Linear(dim, hidden, bias=False)
            self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        self.goal_dim = goal_dim
        # AdaLN-zero: only from action_emb. Goal is NOT fed into predictor.
        self.adaln = nn.Linear(action_dim, 6 * dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

    def forward(
        self,
        x: torch.Tensor,
        action_emb: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"x must be [B, T, dim], got {tuple(x.shape)}")
        if action_emb.dim() != 3 or action_emb.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"action_emb must be [B, T, action_dim] matching x's "
                f"[B={x.shape[0]}, T={x.shape[1]}], got {tuple(action_emb.shape)}"
            )

        # Build conditioning: just action_emb (no goal in predictor)
        cond = action_emb

        modulation = self.adaln(cond)  # [B, T, 6*dim]
        gamma_a, beta_a, alpha_a, gamma_m, beta_m, alpha_m = modulation.chunk(6, dim=-1)

        x_norm = self.norm_attn(x)
        x_mod = x_norm * (1 + gamma_a) + beta_a
        attn_out, _ = self.attn(
            x_mod, x_mod, x_mod, attn_mask=attn_mask, need_weights=False,
        )
        x = x + alpha_a * attn_out

        x_norm = self.norm_mlp(x)
        x_mod = x_norm * (1 + gamma_m) + beta_m
        if self.use_swiglu:
            mlp_out = self.down_proj(
                torch.nn.functional.silu(self.gate_proj(x_mod)) * self.up_proj(x_mod)
            )
        else:
            mlp_out = self.mlp(x_mod)
        x = x + alpha_m * mlp_out
        return x


class AdaLNTransformerPredictor(nn.Module):
    """Stack of :class:`AdaLNZeroBlock` + final LayerNorm + projector.

    Inputs:

    - ``z_history``: ``[B, T, latent_dim]`` past latents
    - ``action_emb``: ``[B, T, action_dim]`` per-step action embeddings
    - ``goal_emb``: optional ``[B, goal_dim]`` goal latent (REX 3.0)
    - ``goal_confidence``: optional ``[B]`` scalar 0-1 gating goal influence

    Output: ``[B, T, latent_dim]`` per-position next-latent predictions. Position
    ``t`` of the output predicts the next latent at time ``t+1``.

    The final projector matches the encoder's MLP+BN. This keeps the predicted
    latent in the same space as the encoder output for a clean
    ``L_pred = ‖ẑ_{t+1} - z_{t+1}‖²``. BatchNorm1d is applied across the flattened
    ``(B*T, latent_dim)`` so the running statistics are computed over all positions.
    """

    def __init__(self, config: PredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or PredictorConfig()
        self.config.validate()
        cfg = self.config

        self.blocks = nn.ModuleList([
            AdaLNZeroBlock(
                cfg.latent_dim,
                num_heads=cfg.num_heads,
                action_dim=cfg.action_dim,
                goal_dim=cfg.goal_dim,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                use_swiglu=cfg.use_swiglu,
            )
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = nn.LayerNorm(cfg.latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.BatchNorm1d(cfg.latent_dim),
        )

    def forward(
        self,
        z_history: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        if z_history.dim() != 3:
            raise ValueError(
                f"z_history must be [B, T, latent_dim], got {tuple(z_history.shape)}"
            )
        b, t, d = z_history.shape
        if d != self.config.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: predictor has {self.config.latent_dim}, got {d}"
            )
        if action_emb.dim() != 3 or action_emb.shape[:2] != (b, t):
            raise ValueError(
                f"action_emb must be [B={b}, T={t}, action_dim], "
                f"got {tuple(action_emb.shape)}"
            )

        if t > 1:
            attn_mask = torch.triu(
                torch.full((t, t), float("-inf"), device=z_history.device),
                diagonal=1,
            )
        else:
            attn_mask = None

        x = z_history
        for block in self.blocks:
            x = block(x, action_emb, attn_mask=attn_mask)
        x = self.final_norm(x)
        # Reshape for BatchNorm1d which expects (N, C). We want statistics over
        # all (B*T) positions per channel — paper applies the projector
        # uniformly across all positions of the predictor's output.
        x_flat = x.reshape(b * t, d)
        out_flat = self.projector(x_flat)
        return out_flat.reshape(b, t, d)
