"""Mockingbird-infused LeWM predictor.

Transfers from ``src/lewm/goal_property.py`` MockingbirdTrajectoryEncoder:
  - RMSNorm (no learnable elementwise affine in norm layers, clean)
  - QK Gain: per-head learnable attention scale (sharpens softmax)
  - Leaky-ReLU² MLP: leaky_relu(x, 0.1).square() — soft gate, cheap
  - Residual mixing with x₀: each block mixes with initial input
  - Loop blocks: virtual depth by repeating middle layers
  - Layer-wise scale decay: 1/√(layer_idx+1) on pre-norm
  - Skip gates: learned sigmoid gates on encoder→decoder skip connections

All Mockingbird features are optional — set PredictorConfig fields to fall
back to standard AdaLN-zero behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════════
# Mockingbird building blocks
# ═══════════════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """RMSNorm — root-mean-square normalization, no bias, no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x * rms
        if self.weight is not None:
            out = out * self.weight.to(dtype=dtype)
        return out


class QKGainAttention(nn.Module):
    """Multi-head self-attention with per-head learnable QK gain.

    Each head has a scalar `qk_gain` multiplied into the attention logit scale.
    Larger gain → sharper attention. Initialize at 5.0 (Mockingbird default).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_gain_init: float = 5.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.qk_gain = nn.Parameter(torch.full((num_heads,), float(qk_gain_init)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        base_scale = self.head_dim ** -0.5
        gain = self.qk_gain.to(dtype=x.dtype).view(1, self.num_heads, 1, 1)
        scale = base_scale * gain

        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            # attn_mask is (T, T) for batch_first — broadcast to (1, 1, T, T)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn = attn + attn_mask

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


class LeakyReLUSquaredMLP(nn.Module):
    """MLP with leaky-ReLU then square activation. Acts as a soft gate.

    h = leaky_relu(W_up · x, 0.1).square()
    out = W_down · h
    """

    def __init__(self, dim: int, ff_mult: float = 3.75):
        super().__init__()
        ff_dim = max(1, int(dim * ff_mult))
        self.up = nn.Linear(dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.up(x), negative_slope=0.1).square()
        return self.down(h)


# ═══════════════════════════════════════════════════════════════════════════════
# Mockingbird AdaLN block
# ═══════════════════════════════════════════════════════════════════════════════

class MockingbirdAdaLNBlock(nn.Module):
    """Transformer block with AdaLN-zero action modulation + Mockingbird features.

    Innovations over standard AdaLNZeroBlock:
      - RMSNorm (no elementwise affine in norm)
      - QK-gain attention (per-head learnable logit scale)
      - Leaky-ReLU² MLP (soft gate, 3.75× expansion)
      - Residual mixing: x_in = mix · x + (1-mix) · x₀
      - Learned scale per residual branch (attn_scale, mlp_scale)
      - Layer-wise scale decay: 1/√(layer_idx+1) on pre-norm
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
    ):
        super().__init__()
        self.dim = dim
        self.layer_idx = layer_idx

        # Norms — RMSNorm without elementwise affine (modulation handles scale)
        self.attn_norm = RMSNorm(dim, elementwise_affine=False)
        self.mlp_norm = RMSNorm(dim, elementwise_affine=False)

        # Attention with QK gain
        self.attn = QKGainAttention(
            dim, num_heads, qk_gain_init=qk_gain_init, dropout=dropout,
        )

        # Leaky-ReLU² MLP
        self.mlp = LeakyReLUSquaredMLP(dim, ff_mult=ff_mult)

        # AdaLN modulation — action → 6×dim (γ_a,β_a,α_a,γ_m,β_m,α_m)
        self.adaln = nn.Linear(action_dim, 6 * dim)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)

        # Residual mixing with x₀
        self.resid_mix = nn.Parameter(
            torch.stack((torch.ones(dim), torch.zeros(dim))).float()
        )

        # Learned branch scales
        self.attn_scale = nn.Parameter(torch.ones(dim))
        self.mlp_scale = nn.Parameter(torch.ones(dim))

        # Layer-scale decay (Mockingbird: 1/√(layer+1))
        self.ln_scale = 1.0 / ((layer_idx + 1) ** 0.5)

    def forward(
        self,
        x: torch.Tensor,
        x0: torch.Tensor,
        action_emb: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with Mockingbird residual mixing + AdaLN modulation.

        Args:
            x: [B, T, dim] current hidden state
            x0: [B, T, dim] initial input (for residual mixing)
            action_emb: [B, T, action_dim] per-position action conditioning
            attn_mask: optional [T, T] causal mask
        """
        dtype = x.dtype
        mix = self.resid_mix.to(dtype=dtype)

        # Residual mixing: blend current x with initial x0
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        # AdaLN modulation from action
        modulation = self.adaln(action_emb)  # [B, T, 6*dim]
        gamma_a, beta_a, alpha_a, gamma_m, beta_m, alpha_m = modulation.chunk(6, dim=-1)

        # Attention branch with layer-scaled RMSNorm
        x_norm = self.attn_norm(x_in) * self.ln_scale
        x_mod = x_norm * (1 + gamma_a) + beta_a
        attn_out = self.attn(x_mod, attn_mask=attn_mask)

        scale_a = self.attn_scale.to(dtype=dtype)[None, None, :]
        x_out = x_in + scale_a * attn_out + alpha_a * 0.0  # α_a is identity-path only

        # MLP branch with layer-scaled RMSNorm
        x_norm = self.mlp_norm(x_out) * self.ln_scale
        x_mod = x_norm * (1 + gamma_m) + beta_m
        mlp_out = self.mlp(x_mod)

        scale_m = self.mlp_scale.to(dtype=dtype)[None, None, :]
        return x_out + scale_m * mlp_out


# ═══════════════════════════════════════════════════════════════════════════════
# Predictor config + model
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MockingbirdPredictorConfig:
    latent_dim: int = 256
    action_dim: int = 8
    num_layers: int = 6
    num_heads: int = 8
    ff_mult: float = 3.75
    qk_gain_init: float = 5.125
    dropout: float = 0.1

    # Mockingbird loop: repeat layers [loop_start, loop_end] this many extra times
    num_loops: int = 0
    loop_start: int = 2
    loop_end: int = 4

    # Skip connections: encoder→decoder gates (requires even-ish split)
    skip_gates: bool = False

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError(f"latent_dim {self.latent_dim} not divisible by num_heads {self.num_heads}")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.num_loops > 0 and not (0 <= self.loop_start <= self.loop_end < self.num_layers):
            raise ValueError("loop_start/loop_end must be valid layer indices")


class MockingbirdPredictor(nn.Module):
    """Mockingbird-infused LeWM predictor.

    Architecture:
      RMSNorm(input) → [MockingbirdAdaLNBlock × N] → RMSNorm → projector

    When num_loops > 0: middle layers get repeated for virtual depth.
    When skip_gates: encoder half → decoder half with learned skip connections.
    """

    def __init__(self, config: MockingbirdPredictorConfig | None = None):
        super().__init__()
        self.config = config or MockingbirdPredictorConfig()
        self.config.validate()
        cfg = self.config

        # Build physical blocks
        self.blocks = nn.ModuleList([
            MockingbirdAdaLNBlock(
                cfg.latent_dim,
                num_heads=cfg.num_heads,
                action_dim=cfg.action_dim,
                ff_mult=cfg.ff_mult,
                qk_gain_init=cfg.qk_gain_init,
                dropout=cfg.dropout,
                layer_idx=i,
            )
            for i in range(cfg.num_layers)
        ])

        # Loop schedule: which layers fire in which order
        self._build_schedule(cfg)

        self.final_norm = RMSNorm(cfg.latent_dim)
        self.projector = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.BatchNorm1d(cfg.latent_dim),
        )

    def _build_schedule(self, cfg) -> None:
        """Build ordered list of layer indices including loop repeats."""
        if cfg.num_loops <= 0:
            self._schedule = list(range(cfg.num_layers))
            self._has_loops = False
            return

        loop_segment = list(range(cfg.loop_start, cfg.loop_end + 1))
        schedule = list(range(cfg.loop_start))
        for _ in range(cfg.num_loops + 1):  # +1 for the initial pass
            schedule.extend(loop_segment)
        schedule.extend(range(cfg.loop_end + 1, cfg.num_layers))

        self._schedule = schedule
        self._has_loops = True

        # Skip connections if enabled: first half → second half
        if cfg.skip_gates:
            self.encoder_indices = schedule[:len(schedule)//2]
            self.decoder_indices = schedule[len(schedule)//2:]
            n_skips = min(len(self.encoder_indices), len(self.decoder_indices))
            self.skip_weights = nn.Parameter(torch.ones(n_skips, cfg.latent_dim))
            self.skip_gates = nn.Parameter(torch.zeros(n_skips, cfg.latent_dim))
        else:
            self.skip_gates = None

    def forward(
        self,
        z_history: torch.Tensor,
        action_emb: torch.Tensor,
        goal_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = z_history.shape
        cfg = self.config
        if D != cfg.latent_dim:
            raise ValueError(f"latent_dim mismatch: {cfg.latent_dim} vs {D}")

        # Causal mask
        if T > 1:
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=z_history.device), diagonal=1,
            )
        else:
            attn_mask = None

        x = z_history
        # Inject goal embedding: broadcast to all timesteps
        if goal_emb is not None:
            x = x + goal_emb.unsqueeze(1)  # (B, 1, D) + (B, T, D)
        x0 = F.rms_norm(x, (D,))  # initial normalized input for residual mixing

        if not self._has_loops:
            # Simple sequential
            for block in self.blocks:
                x = block(x, x0, action_emb, attn_mask=attn_mask)
        elif self.skip_gates is None:
            # Loops, no skip connections
            for idx in self._schedule:
                x = self.blocks[idx](x, x0, action_emb, attn_mask=attn_mask)
        else:
            # Loops + encoder-decoder skip connections
            skips = []
            for idx in self.encoder_indices:
                x = self.blocks[idx](x, x0, action_emb, attn_mask=attn_mask)
                skips.append(x)

            for skip_i, idx in enumerate(self.decoder_indices):
                if skip_i < len(skips) and skip_i < self.skip_weights.shape[0]:
                    skip = skips.pop()
                    w = self.skip_weights[skip_i].to(dtype=x.dtype)[None, None, :]
                    gate = torch.sigmoid(
                        self.skip_gates[skip_i].to(dtype=x.dtype)
                    )[None, None, :]
                    x = torch.lerp(w * skip, x, gate)
                x = self.blocks[idx](x, x0, action_emb, attn_mask=attn_mask)

        x = self.final_norm(x)
        x_flat = x.reshape(B * T, D)
        out_flat = self.projector(x_flat)
        return out_flat.reshape(B, T, D)

    @property
    def effective_layers(self) -> int:
        """Number of block evaluations per forward pass (includes loops)."""
        return len(self._schedule) if hasattr(self, '_schedule') else len(self.blocks)
