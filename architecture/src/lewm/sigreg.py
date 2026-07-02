"""SIGReg: Sketched-Isotropic-Gaussian Regularizer (LeWM Appendix A).

Pulls a batch of latents ``Z`` of shape ``[N, d]`` toward isotropic
``N(0, I_d)`` via the Epps-Pulley test statistic evaluated through
trapezoid quadrature on M random unit-norm projections.

For each projection ``u^(m) ∈ S^{d-1}``, ``h^(m) = Z @ u^(m) ∈ R^N``::

    T^(m) = ∫ w(t) |φ_N(t; h^(m)) - φ_0(t)|² dt
    SIGReg(Z) = (1/M) Σ_m T^(m)

where:

- ``φ_N(t; h) = (1/N) Σ_n e^{i t h_n}`` is the empirical characteristic function
- ``φ_0(t) = e^{-t²/2}`` is the standard-normal characteristic function
- ``w(t) = e^{-t²/(2 λ²)}`` is a Gaussian weight (``λ`` here is the weight
  parameter, distinct from the L_LeWM regularization weight ``λ``)
- integration domain ``[0.2, 4]`` (paper)

Random directions are sampled once at init from ``N(0, I)``, L2-normalized,
and registered as a frozen buffer. Cramér-Wold theorem: matching all
univariate projections matches the full joint distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


PAPER_NUM_PROJECTIONS = 1024
PAPER_QUAD_NODES = 32
PAPER_QUAD_MIN = 0.2
PAPER_QUAD_MAX = 4.0
PAPER_WEIGHT_LAMBDA = 1.0


@dataclass(frozen=True)
class SIGRegConfig:
    latent_dim: int
    num_projections: int = PAPER_NUM_PROJECTIONS
    num_quad_nodes: int = PAPER_QUAD_NODES
    quad_min: float = PAPER_QUAD_MIN
    quad_max: float = PAPER_QUAD_MAX
    weight_lambda: float = PAPER_WEIGHT_LAMBDA
    seed: int | None = None

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.num_projections < 1:
            raise ValueError("num_projections must be >= 1")
        if self.num_quad_nodes < 2:
            raise ValueError("num_quad_nodes must be >= 2")
        if self.quad_max <= self.quad_min:
            raise ValueError("quad_max must be > quad_min")
        if self.weight_lambda <= 0:
            raise ValueError("weight_lambda must be > 0")


class SIGReg(nn.Module):
    def __init__(
        self,
        latent_dim: int | None = None,
        *,
        config: SIGRegConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is None:
            if latent_dim is None:
                raise ValueError("provide latent_dim or config")
            config = SIGRegConfig(latent_dim=latent_dim, **kwargs)
        config.validate()
        self.config = config

        generator = torch.Generator()
        if config.seed is not None:
            generator.manual_seed(config.seed)
        directions = torch.randn(
            config.num_projections, config.latent_dim, generator=generator,
        )
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("directions", directions)

        nodes = torch.linspace(
            config.quad_min, config.quad_max, config.num_quad_nodes,
        )
        self.register_buffer("quad_nodes", nodes)
        weight = torch.exp(-(nodes ** 2) / (2.0 * config.weight_lambda ** 2))
        self.register_buffer("quad_weight", weight)
        target_cf = torch.exp(-(nodes ** 2) / 2.0)
        self.register_buffer("target_cf", target_cf)
        self.quad_step = float(
            (config.quad_max - config.quad_min) / (config.num_quad_nodes - 1)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``[N, d]`` flat sample → scalar SIGReg statistic."""

        if z.dim() != 2:
            raise ValueError(f"SIGReg expects [N, d], got {tuple(z.shape)}")
        if z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: SIGReg has {self.config.latent_dim}, "
                f"got {z.shape[1]}"
            )
        if z.shape[0] < 2:
            return z.new_zeros(())

        h = z @ self.directions.t()
        th = h.unsqueeze(-1) * self.quad_nodes
        phi_real = th.cos().mean(dim=0)
        phi_imag = th.sin().mean(dim=0)
        diff_sq = (phi_real - self.target_cf) ** 2 + phi_imag ** 2
        weighted = diff_sq * self.quad_weight
        T_per_proj = self.quad_step * (
            weighted.sum(dim=-1) - 0.5 * (weighted[..., 0] + weighted[..., -1])
        )
        return T_per_proj.mean()

    def forward_sequence(self, z_btd: torch.Tensor) -> torch.Tensor:
        """Step-wise SIGReg over a ``(B, T, D)`` sequence (paper pseudocode).

        Paper Algorithm 1::

            sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))   # emb: (B, T, D) → (T, B, D)

        For each time-step ``t``, compute the Epps-Pulley statistic over the
        ``(B, D)`` sample at that step (so the normality test sees ``B`` i.i.d.
        draws per step), average over the ``M`` projections, then average across
        ``T`` time-steps. Returns a scalar.

        Vectorized: equivalent to a Python loop over ``t`` but does the random
        projection, characteristic-function evaluation, and quadrature in one
        batched op.
        """

        if z_btd.dim() != 3:
            raise ValueError(
                f"SIGReg.forward_sequence expects [B, T, D], got {tuple(z_btd.shape)}"
            )
        b, t_len, d = z_btd.shape
        if d != self.config.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: SIGReg has {self.config.latent_dim}, got {d}"
            )
        if b < 2:
            return z_btd.new_zeros(())

        h = z_btd @ self.directions.t()                      # (B, T, M)
        th = h.unsqueeze(-1) * self.quad_nodes                # (B, T, M, K)
        phi_real = th.cos().mean(dim=0)                       # (T, M, K) — mean over B
        phi_imag = th.sin().mean(dim=0)                       # (T, M, K)
        diff_sq = (phi_real - self.target_cf) ** 2 + phi_imag ** 2
        weighted = diff_sq * self.quad_weight
        T_per_proj = self.quad_step * (
            weighted.sum(dim=-1) - 0.5 * (weighted[..., 0] + weighted[..., -1])
        )                                                     # (T, M)
        sigreg_per_step = T_per_proj.mean(dim=-1)             # (T,) — mean over M
        return sigreg_per_step.mean()                         # scalar — mean over T
