"""Muon optimizer — from Keller Jordan / modded-nanogpt.

Orthogonalized gradient descent via Newton-Schulz iteration.
≥2D params get orthogonalized; 1D params use signSGD with momentum.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.optim.optimizer import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute G @ (G.T @ G)^{-1/2}.

    Assumes G is 2D and G.shape[0] >= G.shape[1] (or transposes if not).
    Produces an orthogonal matrix with the same span as G.
    """
    assert G.ndim == 2
    # Ensure rows >= cols
    if G.shape[1] > G.shape[0]:
        G = G.T
        transposed = True
    else:
        transposed = False

    a, b, c = (3.4445, -4.7750, 2.0315) if G.dtype in (torch.float16, torch.bfloat16) else (3.4445, -4.7750, 2.0315)

    X = G.float()
    X = X / (X.norm() + 1e-7)

    if steps > 0:
        # Apply Newton-Schulz polynomial
        for _ in range(steps):
            A = X @ X.T
            B = A @ X
            X = a * X + b * B + c * (A @ B)

    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon: Momentum-enhanced orthogonalized gradient descent.

    2D+ params: orthonormal gradient via Newton-Schulz + momentum.
    1D params: signSGD + momentum.

    Recommended: lr ≈ 0.02 for the Muon params (can use per-layer LRs).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                if p.ndim >= 2:
                    # Orthonormal gradient via Newton-Schulz
                    shape = p.shape
                    g_2d = grad.reshape(shape[0], -1)
                    update = zeropower_via_newtonschulz5(g_2d, ns_steps)
                    # Scale by 0.2 * max(1, dim/numel) as per modded-nanogpt
                    update *= 0.2 * max(1, g_2d.shape[0] / g_2d.numel()) ** 0.5
                    update = update.reshape(shape)
                else:
                    # 1D param: signSGD
                    update = grad.sign()

                # Momentum
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.data)
                buf = state["buf"]
                buf.mul_(momentum).add_(update)
                if nesterov:
                    update = update.add(buf, alpha=momentum)
                else:
                    update = buf

                p.data.add_(update, alpha=-lr)

        return loss
