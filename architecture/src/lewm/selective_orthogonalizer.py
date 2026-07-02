"""Selective PCGrad orthogonaliser.

Wraps ``PairedGradientOrthogonalizer`` so projections only run for
pairs the gating predicate marks as conflicting. The predicate is
typically backed by ``LiveRelationalProbe.should_project``.

Cost shift:

  * Full PCGrad: N² inner-loop projections per step (each is a dot
    product + axpy on a flat trunk-grad vector). For N≈14 that's ~190
    projections; each is fast on GPU but they're sequential within
    the orthogonaliser.
  * Selective: skip pairs where running cosine is above
    ``conflict_threshold``. After warmup, most pairs are non-conflicting,
    so the effective projection count drops sharply.

The N retain-graph BACKWARD passes are still paid every step (the
orthogonaliser needs per-head trunk grads to evaluate the predicate at
all). What we save is the inner projection loop.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch

from .relational_gradients import (
    PairedGradientOrthogonalizer,
    flatten_trunk_grads,
)


class SelectivePCGradOrthogonalizer:
    """``project(head_losses)`` returns a summed-projected flat trunk grad.

    Differs from ``PairedGradientOrthogonalizer.project`` in that the
    inner projection loop consults a predicate ``should_project(a, b)``
    before subtracting the conflicting component. Pairs the predicate
    rejects are skipped — their grads contribute to the sum unchanged.

    Parameters
    ----------
    model:
        The model whose trunk grads are being projected. Must support
        ``.zero_grad(set_to_none=False)`` and named-parameter access.
    trunk_prefixes:
        Parameter-name prefixes identifying trunk params.
    head_names:
        All head names the orthogonaliser knows about. Pairs are drawn
        from this list × itself.
    should_project_fn:
        Callable ``(a: str, b: str) -> bool``. Return True to PCGrad
        the (a, b) pair, False to skip. Receives the ordered pair as
        appearing in the inner loop.
    rng_seed:
        For shuffling the per-iteration projection order (mirrors
        PairedGradientOrthogonalizer).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        trunk_prefixes: Sequence[str],
        head_names: Sequence[str],
        should_project_fn: Callable[[str, str], bool],
        *,
        rng_seed: int = 0,
    ) -> None:
        self.model = model
        self.trunk_prefixes = tuple(trunk_prefixes)
        self.head_names = list(head_names)
        self.should_project_fn = should_project_fn
        self._rng = torch.Generator()
        self._rng.manual_seed(rng_seed)
        # Same surface as PairedGradientOrthogonalizer so callers can
        # swap them transparently.
        self.last_per_head_grads: dict[str, torch.Tensor] = {}
        # Bookkeeping: how many (a, b) projections were actually run
        # vs. skipped in the most recent project() call. Useful for
        # the live snapshot + cost accounting.
        self.last_pairs_projected: int = 0
        self.last_pairs_skipped: int = 0

    def project(
        self,
        head_losses: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run N retain-graph backwards + selective PCGrad, return the
        summed projected trunk grad as a flat tensor.

        Mirrors ``PairedGradientOrthogonalizer.project``'s interface so
        callers (e.g. ``orthogonal_step``) can swap implementations.
        """
        names = [n for n in self.head_names if n in head_losses]
        if not names:
            raise ValueError("no head losses present; nothing to project")

        # Per-head trunk grads via N retain-graph backwards.
        per_head: dict[str, torch.Tensor] = {}
        for name in names:
            loss = head_losses[name]
            self.model.zero_grad(set_to_none=False)
            loss.backward(retain_graph=True)
            per_head[name] = flatten_trunk_grads(
                self.model, self.trunk_prefixes,
            ).detach().clone()

        # Selective PCGrad inner loop.
        projected: dict[str, torch.Tensor] = {
            n: g.clone() for n, g in per_head.items()
        }
        pairs_projected = 0
        pairs_skipped = 0
        for i_name in names:
            others_order = names.copy()
            perm = torch.randperm(
                len(others_order), generator=self._rng,
            ).tolist()
            others_order = [others_order[k] for k in perm]
            gi = projected[i_name]
            for j_name in others_order:
                if j_name == i_name:
                    continue
                if not self.should_project_fn(i_name, j_name):
                    pairs_skipped += 1
                    continue
                gj = per_head[j_name]
                dot = (gi * gj).sum()
                if dot.item() < 0.0:
                    denom = gj.pow(2).sum().clamp_min(1e-12)
                    gi = gi - (dot / denom) * gj
                    pairs_projected += 1
                else:
                    # Predicate said "project," but post-backward
                    # the actual dot is non-negative — count as skipped.
                    pairs_skipped += 1
            projected[i_name] = gi

        self.last_per_head_grads = per_head
        self.last_pairs_projected = pairs_projected
        self.last_pairs_skipped = pairs_skipped

        return torch.stack(list(projected.values()), dim=0).sum(dim=0)


def make_full_orthogonalizer(
    model: torch.nn.Module,
    trunk_prefixes: Sequence[str],
    head_names: Sequence[str],
    *,
    rng_seed: int = 0,
) -> PairedGradientOrthogonalizer:
    """Convenience: build the non-selective orthogonaliser with the same
    constructor signature (handy for trainer code that decides at runtime
    which implementation to instantiate).
    """
    return PairedGradientOrthogonalizer(
        model,
        trunk_prefixes=trunk_prefixes,
        head_names=head_names,
        rng_seed=rng_seed,
    )


__all__ = ["SelectivePCGradOrthogonalizer", "make_full_orthogonalizer"]
