"""Skipless Mockingbird predictor — "Cutting the Skip" (Ji et al., ICLR 2026).

Removes residual connections from predictor blocks and applies the paper's
principled initialization to attention weights:

  W^V, W^O → scaled orthonormal via SVD (c·U, c·V^T)
  W^Q, W^K → mimetic: αZ + βI  (α=2, β=0.6)

Retains Mockingbird innovations: RMSNorm, QK-gain, AdaLN modulation,
Leaky-ReLU² MLP. Removes residual mixing with x₀ (it's a skip variant).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.mockingbird_predictor import (
    QKGainAttention,
    LeakyReLUSquaredMLP,
    RMSNorm,
    MockingbirdPredictorConfig,
    MockingbirdPredictor,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Paper initialization helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _init_v_o_svd(v_proj: nn.Linear, o_proj: nn.Linear, c: float = 3.0) -> None:
    """Initialize W^V and W^O such that W^V @ W^O is scaled orthonormal.

    Algorithm: sample random Q ∈ R^{d×d} with N(0,1) entries, compute SVD
    Q = U S V^T, assign W^V = c·U, W^O = c·V^T.
    """
    d = v_proj.weight.shape[0]
    # Use existing weight as random source (it's already random from init)
    q = v_proj.weight.data.clone()  # (d, d)
    # SVD
    try:
        u, s, vt = torch.linalg.svd(q.float(), full_matrices=True)
    except Exception:
        # Fallback: use random
        q = torch.randn(d, d)
        u, s, vt = torch.linalg.svd(q, full_matrices=True)

    v_proj.weight.data = (c * u).to(v_proj.weight.dtype)
    o_proj.weight.data = (c * vt).to(o_proj.weight.dtype)
    if v_proj.bias is not None:
        nn.init.zeros_(v_proj.bias)
    if o_proj.bias is not None:
        nn.init.zeros_(o_proj.bias)


def _init_qk_mimetic(q_proj: nn.Linear, k_proj: nn.Linear,
                     alpha: float = 2.0, beta: float = 0.6) -> None:
    """Initialize W^Q and W^K such that W^Q @ W^K^T = αZ + βI.

    Z has entries Z_ij ~ N(0, 1/d). We implement by setting W^K = I/sqrt(d)
    and W^Q = α*Z' + β*I/sqrt(d), then the product is αZ/√d + βI/d.
    Simplification: set Q and K each as random + scaled identity.
    """
    d = q_proj.weight.shape[0]
    std = 1.0 / (d ** 0.5)

    # K = identity scaled (ensures QK^T gets the βI term)
    k_proj.weight.data = torch.eye(d) * (beta ** 0.5) * std
    # Q = random + identity contribution
    q_proj.weight.data = torch.randn(d, d) * alpha * std + torch.eye(d) * (beta ** 0.5) * std

    if q_proj.bias is not None:
        nn.init.zeros_(q_proj.bias)
    if k_proj.bias is not None:
        nn.init.zeros_(k_proj.bias)


# ═══════════════════════════════════════════════════════════════════════════════
# Skipless Mockingbird Block
# ═══════════════════════════════════════════════════════════════════════════════

class SkiplessMockingbirdBlock(nn.Module):
    """Mockingbird AdaLN block WITHOUT residual connections.

    Paper: Ji et al. "Cutting the Skip", ICLR 2026.
    Retains: RMSNorm, QK-gain, AdaLN, leaky-ReLU² MLP.
    Removes: residual paths, residual mixing with x₀.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        action_dim: int,
        ff_mult: float = 3.75,
        qk_gain_init: float = 5.0,
        dropout: float = 0.0,
        layer_idx: int = 0,
        init_alpha: float = 2.0,
        init_beta: float = 0.6,
        init_c: float = 3.0,
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx

        # Norms — RMSNorm without elementwise affine
        self.attn_norm = RMSNorm(dim, elementwise_affine=False)
        self.mlp_norm = RMSNorm(dim, elementwise_affine=False)

        # Attention with QK gain
        self.attn = QKGainAttention(
            dim, num_heads, qk_gain_init=qk_gain_init, dropout=dropout,
        )

        # Leaky-ReLU² MLP
        self.mlp = LeakyReLUSquaredMLP(dim, ff_mult=ff_mult)

        # AdaLN modulation — action → 4×dim (γ_a,β_a,γ_m,β_m)
        # (no α_a,α_m — those were for residual paths)
        self.adaln = nn.Linear(action_dim, 4 * dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

        # Learned branch scales
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

        # Layer-scale decay
        self.ln_scale = 1.0 / ((layer_idx + 1) ** 0.5)

        # ── Paper initialization ──
        self._apply_paper_init(init_alpha, init_beta, init_c)

    def _apply_paper_init(self, alpha: float, beta: float, c: float) -> None:
        """Apply "Cutting the Skip" initialization to attention weights."""
        _init_v_o_svd(self.attn.v_proj, self.attn.out_proj, c=c)
        _init_qk_mimetic(self.attn.q_proj, self.attn.k_proj, alpha=alpha, beta=beta)

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,  # kept for API compat, not used
        action_emb: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Skipless forward: attn → norm → mlp → norm → output. No residuals."""
        dtype = x.dtype
        _ = x0  # unused — no residual mixing

        # AdaLN modulation
        modulation = self.adaln(action_emb)  # [B, T, 4*dim]
        gamma_a, beta_a, gamma_m, beta_m = modulation.chunk(4, dim=-1)

        # Attention branch (NO residual)
        x_norm = self.attn_norm(x) * self.ln_scale
        x_mod = x_norm * (1 + gamma_a) + beta_a
        attn_out = self.attn(x_mod, attn_mask=attn_mask)

        scale_a = self.attn_scale.to(dtype=dtype)[None, None, :]
        x_out = scale_a * attn_out  # ← NO + x, pure skipless

        # MLP branch (NO residual)
        x_norm = self.mlp_norm(x_out) * self.ln_scale
        x_mod = x_norm * (1 + gamma_m) + beta_m
        mlp_out = self.mlp(x_mod)

        scale_m = self.mlp_scale.to(dtype=dtype)[None, None, :]
        return scale_m * mlp_out  # ← NO + x_out, pure skipless


# ═══════════════════════════════════════════════════════════════════════════════
# Skipless Predictor
# ═══════════════════════════════════════════════════════════════════════════════

class SkiplessMockingbirdPredictor(nn.Module):
    """Mockingbird predictor with all residual connections removed.

    Architecture:
      input → [SkiplessMockingbirdBlock × N] → RMSNorm → projector
    """

    def __init__(self, config: MockingbirdPredictorConfig | None = None):
        super().__init__()
        cfg = config or MockingbirdPredictorConfig()
        cfg.validate()

        self.blocks = nn.ModuleList([
            SkiplessMockingbirdBlock(
                dim=cfg.latent_dim,
                num_heads=cfg.num_heads,
                action_dim=cfg.action_dim,
                ff_mult=cfg.ff_mult,
                qk_gain_init=cfg.qk_gain_init,
                dropout=cfg.dropout,
                layer_idx=i,
            )
            for i in range(cfg.num_layers)
        ])

        self.final_norm = RMSNorm(cfg.latent_dim)
        self.projector = nn.Linear(cfg.latent_dim, cfg.latent_dim)
        nn.init.trunc_normal_(self.projector.weight, std=0.02)
        nn.init.zeros_(self.projector.bias)

    def forward(
        self,
        emb: torch.Tensor,
        action_emb: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process latent sequence through skipless blocks.

        Args:
            emb: [B, T, D] latent embeddings from encoder
            action_emb: [B, T, A] action embeddings
            attn_mask: optional [T, T] causal mask
        Returns:
            [B, T, D] predicted next-timestep latents
        """
        x = emb
        x0 = emb  # kept for API compat, not used by skipless blocks

        for block in self.blocks:
            x = block(x, x0, action_emb, attn_mask=attn_mask)

        x = self.final_norm(x)
        return self.projector(x)
