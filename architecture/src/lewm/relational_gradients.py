"""Per-head trunk-gradient relational analysis + PCGrad-style projection.

When multiple heads share a trunk, their per-head gradients on trunk
parameters interfere. Pairs of heads whose gradients have NEGATIVE
cosine are "fighting" — pulling the trunk in opposite directions for
the same parameter. The standard fix is gradient surgery (Yu et al
2020): project out the conflicting component.

This module provides two layered utilities:

  - ``RelationalGradientProbe``: diagnostic. For each pair of heads,
    measures cosine, magnitude ratio, and conflict rate of their
    gradients on trunk parameters over N batches. Identifies which
    pairs are net-aligned vs net-fighting.

  - ``PairedGradientOrthogonalizer``: response. PCGrad-style — for
    each pair (i, j) where g_i·g_j < 0, project the conflicting
    component out of g_i. Symmetric across pairs (i,j) and (j,i)
    happen independently.

Both work on per-task TRUNK gradients only — head-local gradients
don't conflict (each head has its own params). The expense scales
O(N²) where N is the number of heads; with Lewis's ~14 heads this
is ~196 dot products per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch


# ---------------------------------------------------------------------------
# Flat-tensor helpers
# ---------------------------------------------------------------------------


def flatten_trunk_grads(
    model: torch.nn.Module,
    trunk_prefixes: Sequence[str],
) -> torch.Tensor:
    """Collect .grad of every trunk parameter into a flat float tensor.

    Returns a 1-D tensor of length = total_trunk_params. Params with
    grad=None contribute zeros. Caller is responsible for clearing
    grads + computing a backward pass BEFORE calling this.
    """
    base = model
    if hasattr(base, "module"):
        base = base.module
    if hasattr(base, "_orig_mod"):
        base = base._orig_mod
    chunks: list[torch.Tensor] = []
    for name, p in base.named_parameters():
        if not name.startswith(tuple(trunk_prefixes)):
            continue
        if p.grad is None:
            chunks.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        else:
            chunks.append(p.grad.detach().float().reshape(-1))
    if not chunks:
        raise ValueError(
            f"no trunk params matched prefixes {tuple(trunk_prefixes)}; "
            f"is the model the expected GoalPropertyModel?"
        )
    return torch.cat(chunks)


def assign_flat_trunk_grads(
    model: torch.nn.Module,
    flat_grads: torch.Tensor,
    trunk_prefixes: Sequence[str],
) -> None:
    """Reverse of ``flatten_trunk_grads`` — write a flat tensor back
    into per-parameter .grad fields. Used after PCGrad-style projection.
    """
    base = model
    if hasattr(base, "module"):
        base = base.module
    if hasattr(base, "_orig_mod"):
        base = base._orig_mod
    cursor = 0
    for name, p in base.named_parameters():
        if not name.startswith(tuple(trunk_prefixes)):
            continue
        n = p.numel()
        chunk = flat_grads[cursor: cursor + n].view_as(p).to(dtype=p.dtype)
        if p.grad is None:
            p.grad = chunk.clone()
        else:
            p.grad.copy_(chunk)
        cursor += n
    if cursor != flat_grads.numel():
        raise ValueError(
            f"flat_grads length {flat_grads.numel()} != trunk param count {cursor}"
        )


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """Cosine similarity between two 1-D tensors. Returns 0.0 if either is zero."""
    na = a.norm().item()
    nb = b.norm().item()
    if na < eps or nb < eps:
        return 0.0
    return float((a * b).sum().item() / (na * nb))


# ---------------------------------------------------------------------------
# Diagnostic — RelationalGradientProbe
# ---------------------------------------------------------------------------


@dataclass
class PairStats:
    """Running stats for one head-pair across N steps."""
    head_a: str
    head_b: str
    n_steps: int = 0
    cos_sum: float = 0.0
    cos_sq_sum: float = 0.0
    cos_min: float = float("inf")
    cos_max: float = float("-inf")
    n_conflicts: int = 0  # steps where cos < 0
    mag_ratio_sum: float = 0.0  # |g_a| / |g_b| accumulated

    def record(self, cos: float, mag_a: float, mag_b: float) -> None:
        self.n_steps += 1
        self.cos_sum += cos
        self.cos_sq_sum += cos * cos
        self.cos_min = min(self.cos_min, cos)
        self.cos_max = max(self.cos_max, cos)
        if cos < 0.0:
            self.n_conflicts += 1
        if mag_b > 1e-12:
            self.mag_ratio_sum += mag_a / mag_b

    def summary(self) -> dict[str, float]:
        n = max(1, self.n_steps)
        mean = self.cos_sum / n
        var = (self.cos_sq_sum / n) - mean * mean
        return {
            "pair": f"{self.head_a} × {self.head_b}",
            "n_steps": self.n_steps,
            "cos_mean": mean,
            "cos_std": var ** 0.5 if var > 0 else 0.0,
            "cos_min": self.cos_min if self.n_steps else 0.0,
            "cos_max": self.cos_max if self.n_steps else 0.0,
            "conflict_rate": self.n_conflicts / n,
            "mag_ratio_mean": self.mag_ratio_sum / n,
        }


class RelationalGradientProbe:
    """Walks N batches and records per-pair trunk-gradient relations.

    Usage:

        probe = RelationalGradientProbe(model, trunk_prefixes, head_losses)
        for batch in loader:
            outputs = model(batch)
            losses = compute_losses(outputs, batch)
            probe.record_step(losses)
        report = probe.report()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        trunk_prefixes: Sequence[str],
        head_names: Sequence[str],
    ) -> None:
        self.model = model
        self.trunk_prefixes = tuple(trunk_prefixes)
        self.head_names = list(head_names)
        # All unordered pairs (i, j) with i < j
        self.pairs: dict[tuple[str, str], PairStats] = {}
        for i, a in enumerate(self.head_names):
            for b in self.head_names[i + 1:]:
                self.pairs[(a, b)] = PairStats(head_a=a, head_b=b)
        self.solo_norms: dict[str, list[float]] = {n: [] for n in self.head_names}

    def record_step(self, head_losses: dict[str, torch.Tensor]) -> None:
        """For one batch's forward, isolate per-head trunk grads via
        independent backward passes (retain_graph) and record cosines.
        """
        # First pass: capture trunk gradients per head into a dict
        per_head_grads: dict[str, torch.Tensor] = {}
        for name in self.head_names:
            if name not in head_losses:
                continue
            loss = head_losses[name]
            if not torch.is_tensor(loss):
                continue
            self.model.zero_grad(set_to_none=False)
            loss.backward(retain_graph=True)
            g = flatten_trunk_grads(self.model, self.trunk_prefixes).detach().clone()
            per_head_grads[name] = g
            self.solo_norms[name].append(float(g.norm().item()))

        # Second pass: compute per-pair cosines
        for (a, b), stats in self.pairs.items():
            if a not in per_head_grads or b not in per_head_grads:
                continue
            ga, gb = per_head_grads[a], per_head_grads[b]
            c = cosine(ga, gb)
            mag_a = float(ga.norm().item())
            mag_b = float(gb.norm().item())
            stats.record(c, mag_a, mag_b)

    def report(self, top_k: int = 20) -> dict:
        """Returns a structured summary: per-pair statistics + top-N most
        conflicting pairs + per-head magnitude stats.
        """
        per_pair = [s.summary() for s in self.pairs.values() if s.n_steps > 0]
        per_pair.sort(key=lambda r: -r["conflict_rate"])  # worst conflicts first
        per_head = {
            name: {
                "n_obs": len(norms),
                "norm_mean": (sum(norms) / len(norms)) if norms else 0.0,
                "norm_max": max(norms) if norms else 0.0,
            }
            for name, norms in self.solo_norms.items()
        }
        return {
            "per_pair": per_pair,
            "per_head": per_head,
            "top_conflicts": per_pair[:top_k],
        }


# ---------------------------------------------------------------------------
# Response — PCGrad-style projection
# ---------------------------------------------------------------------------


class PairedGradientOrthogonalizer:
    """PCGrad-style projection (Yu et al 2020 'Gradient Surgery').

    For each ordered pair (i, j), if g_i · g_j < 0, project out the
    conflicting component:

        g_i ← g_i − (g_i · g_j / |g_j|²) g_j

    Crucially, projections are computed in a SHUFFLED order so the
    operation isn't biased to a specific head. The summed projected
    gradient is written back to trunk params.

    Cost: O(N²) dot products per step where N = num heads. For Lewis's
    ~14 heads that's ~190 dot products on a flat 8M-dim vector — fast
    on GPU, but each PCGrad step requires N separate backwards from
    the loss terms (which is the real expense). Best deployed
    selectively rather than every step.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        trunk_prefixes: Sequence[str],
        head_names: Sequence[str],
        *,
        rng_seed: int = 0,
    ) -> None:
        self.model = model
        self.trunk_prefixes = tuple(trunk_prefixes)
        self.head_names = list(head_names)
        self._rng = torch.Generator()
        self._rng.manual_seed(rng_seed)
        # After each project() call, this holds the per-head FLAT trunk
        # gradient tensors keyed by head name. Consumed downstream by
        # GradientAwareBalancer.update_gradients() so the balancer can
        # EMA-track per-task gradient distribution entropy.
        self.last_per_head_grads: dict[str, torch.Tensor] = {}

    def project(
        self,
        head_losses: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run N backward passes (one per head), apply PCGrad, return
        the summed-projected trunk gradient as a flat tensor.

        Caller is responsible for assigning the returned flat back via
        ``assign_flat_trunk_grads`` AND zeroing the rest of the
        gradients (head-local params remain in their last-computed state).
        """
        names = [n for n in self.head_names if n in head_losses]
        if not names:
            raise ValueError("no head losses present; nothing to project")

        # Capture per-head trunk grads
        per_head: dict[str, torch.Tensor] = {}
        for name in names:
            loss = head_losses[name]
            self.model.zero_grad(set_to_none=False)
            loss.backward(retain_graph=True)
            per_head[name] = flatten_trunk_grads(
                self.model, self.trunk_prefixes,
            ).detach().clone()

        # PCGrad: for each name, project against shuffled others
        projected: dict[str, torch.Tensor] = {n: g.clone() for n, g in per_head.items()}
        for i_name in names:
            others_order = names.copy()
            # Shuffle the order of projections (mitigates first-mover bias)
            perm = torch.randperm(len(others_order), generator=self._rng).tolist()
            others_order = [others_order[k] for k in perm]
            gi = projected[i_name]
            for j_name in others_order:
                if j_name == i_name:
                    continue
                gj = per_head[j_name]  # NOTE: use original gj, not projected
                dot = (gi * gj).sum()
                if dot.item() < 0.0:
                    denom = gj.pow(2).sum().clamp_min(1e-12)
                    gi = gi - (dot / denom) * gj
            projected[i_name] = gi

        # Expose per-head pre-projection grads for downstream consumers
        # (e.g. GradientAwareBalancer's gradient-entropy update).
        self.last_per_head_grads = per_head

        # Sum the projected gradients — this is the final trunk update direction
        return torch.stack(list(projected.values()), dim=0).sum(dim=0)
