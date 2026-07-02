"""SODA optimizer wrapper — "Optimistic Dual Averaging" (Pethick et al., 2026).

Simplified form (Eq. 4 from paper):
    x_{k+1} = (1 - 1/(k+2)) * x_k + 1/(k+2) * z_0 + BaseUpdate(g_k)
            = weight_decay toward z0   + base optimizer step

BaseUpdate is what the base optimizer does WITHOUT weight decay.
SODA provides the weight decay externally via the 1/(k+2) schedule.

Tail-of-run safety (Lewis precision regime, iter 13)
----------------------------------------------------
The raw 1/(k+2) schedule decays toward zero — at step 1e6, alpha ~1e-6,
so SODA's Jacobian-centering (the thing that rescues skipless
Mockingbird from the AdamW mid-training collapse) effectively vanishes.
The skipless-collapse rescue is only proven for the early/mid epochs the
probes cover; an unguarded 1M-game run would lose the regularization
exactly where it's been training longest.

Two opt-in guards, both default-off (legacy behaviour preserved):

  * ``reanchor_every`` — every N steps, re-snapshot z0 to the current
    params and reset the decay window. Centering stays perpetually
    active (each window is a fresh strong→weak decay) and the anchor
    tracks a recent stable point instead of frozen init. This is the
    pattern the SODA research report itself recommends ("periodic
    re-anchoring").
  * ``alpha_floor`` — clamp alpha to a minimum so even within a window
    the centering can't fully vanish. Belt-and-suspenders for very long
    windows.
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


class SODA(Optimizer):
    """SODA wrapper: eliminates weight decay tuning via 1/k schedule.

    Args:
        params: parameters / param groups.
        base_optimizer: any torch optimizer, run WITHOUT weight decay.
        reanchor_every: if > 0, re-snapshot the z0 anchor to the current
            params every N steps and restart the 1/(k+2) decay window.
            0 (default) = anchor frozen at init, legacy behaviour.
        alpha_floor: minimum value for the centering coefficient alpha.
            0.0 (default) = no floor, legacy behaviour.
    """

    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        *,
        reanchor_every: int = 0,
        alpha_floor: float = 0.0,
    ):
        defaults = dict()
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer
        self.step_count = 0          # global step counter
        self.window_step = 0         # steps since last (re)anchor
        self.reanchor_every = int(reanchor_every)
        self.alpha_floor = float(alpha_floor)

        # Snapshot initial params z_0 as the anchor
        self.z0 = [p.data.clone() for p in self._all_params()]

    def _all_params(self):
        for group in self.param_groups:
            yield from group["params"]

    @torch.no_grad()
    def step(self, closure=None):
        # Centering coefficient uses the WINDOW step, not the global
        # step — so periodic re-anchoring keeps alpha meaningfully large
        # instead of decaying toward zero over a long run.
        k = self.window_step
        alpha = 1.0 / (k + 2)
        if self.alpha_floor > 0.0 and alpha < self.alpha_floor:
            alpha = self.alpha_floor

        params = list(self._all_params())

        # Snapshot x_k before base optimizer modifies it
        xk = [p.data.clone() for p in params]

        # Compute loss + backward if closure provided
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Step 1: base optimizer step (no weight decay — that's our job)
        self.base_optimizer.step()

        # Step 2: compute BaseUpdate = what base optimizer just did
        # params now contain x_k + BaseUpdate
        base_updates = []
        for p, xk_i in zip(params, xk):
            base_updates.append(p.data - xk_i)
            # Restore to x_k
            p.data.copy_(xk_i)

        # Step 3: SODA update
        # x_{k+1} = (1-alpha)*x_k + alpha*z0 + BaseUpdate
        for p, z0_i, xk_i, bu in zip(params, self.z0, xk, base_updates):
            decayed = (1.0 - alpha) * xk_i + alpha * z0_i
            p.data.copy_(decayed + bu)

        self.step_count += 1
        self.window_step += 1

        # Periodic re-anchoring: snapshot a recent stable point as the
        # new z0 and restart the decay window so centering stays active.
        if self.reanchor_every > 0 and self.window_step >= self.reanchor_every:
            self.z0 = [p.data.clone() for p in params]
            self.window_step = 0

        return loss

    def zero_grad(self, set_to_none: bool = True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
