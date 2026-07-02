"""MultiSubspaceSIGReg — intern's Sub-JEPA regularizer (arXiv 2605.09241).

Drop-in replacement for :class:`lewm.sigreg.SIGReg` with the same 2D
``forward(z)`` contract: input ``(N, D)``, output scalar.

Internally projects each sample into K frozen subspaces of dimension
``D / K`` and applies the existing per-subspace Epps-Pulley statistic.
Catches local subspace collapse that the global SIGReg statistic misses.

On TwoRoom (intern's test1), K=32 orthogonal_frozen gave 33x lower
pred_mse than vanilla SIGReg at equal wallclock.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lewm.sigreg import SIGReg, SIGRegConfig


class MultiSubspaceSIGReg(nn.Module):
    def __init__(
        self,
        latent_dim: int | None = None,
        *,
        num_subspaces: int = 32,
        subspace_dim: int | None = None,
        init_mode: str = "orthogonal_frozen",
        config: SIGRegConfig | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if config is None:
            if latent_dim is None:
                raise ValueError("provide latent_dim or config")
            config = SIGRegConfig(latent_dim=latent_dim, **kwargs)
        config.validate()
        d = config.latent_dim

        if subspace_dim is None:
            if d % num_subspaces != 0:
                raise ValueError(
                    f"latent_dim={d} must be divisible by num_subspaces={num_subspaces} "
                    f"when subspace_dim is not given"
                )
            subspace_dim = d // num_subspaces
        if subspace_dim < 1:
            raise ValueError(f"subspace_dim must be >= 1, got {subspace_dim}")
        if init_mode not in ("orthogonal_frozen", "random_frozen"):
            raise ValueError(
                f"init_mode must be 'orthogonal_frozen' or 'random_frozen', "
                f"got {init_mode!r}"
            )

        self.num_subspaces = int(num_subspaces)
        self.subspace_dim = int(subspace_dim)
        self.init_mode = init_mode
        self.config = config

        gen = torch.Generator()
        if config.seed is not None:
            gen.manual_seed(config.seed)
        mats = []
        for _ in range(self.num_subspaces):
            if init_mode == "orthogonal_frozen":
                q, _ = torch.linalg.qr(
                    torch.randn(d, self.subspace_dim, generator=gen),
                    mode="reduced",
                )
                mats.append(q.transpose(0, 1).contiguous())  # (d', D)
            else:
                mats.append(torch.randn(self.subspace_dim, d, generator=gen))
        self.register_buffer("projection_matrices", torch.stack(mats, dim=0))  # (K, d', D)

        # One internal SIGReg over the subspace dimension; reused per subspace.
        sub_cfg = SIGRegConfig(
            latent_dim=self.subspace_dim,
            num_projections=config.num_projections,
            num_quad_nodes=config.num_quad_nodes,
            quad_min=config.quad_min,
            quad_max=config.quad_max,
            weight_lambda=config.weight_lambda,
            seed=config.seed,
        )
        self.sigreg = SIGReg(config=sub_cfg)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 2:
            raise ValueError(f"expects [N, D], got {tuple(z.shape)}")
        if z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: have {self.config.latent_dim}, got {z.shape[1]}"
            )
        if z.shape[0] < 2:
            return z.new_zeros(())

        # (N, D) x (K, d', D)^T  →  (K, N, d')
        proj = torch.einsum("nd,kjd->knj", z, self.projection_matrices)

        # Per-subspace SIGReg, averaged. K=32 is small; loop is fine.
        stats = [self.sigreg(proj[k]) for k in range(self.num_subspaces)]
        return torch.stack(stats).mean()


__all__ = ["MultiSubspaceSIGReg"]
