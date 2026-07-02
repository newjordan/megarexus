"""Live (always-on) per-pair gradient probe.

The iter-4 RelationalGradientProbe is a diagnostic — it runs for N steps
then exits with a JSON report. This module promotes the probe to a
RUNTIME signal: per-pair stats updated each step, queried each step.

Two consumers:

  * ``GradientAwareBalancer.update_conflicts`` — collapse the live
    per-pair conflict-rate down to a per-task conflict-rate (mean
    across pairs containing the task) every step, so the balancer's
    conflict-penalty factor reflects *current* training state, not a
    stale offline JSON.

  * ``SelectivePCGradOrthogonalizer`` — predicate over pairs ("should
    we PCGrad this pair?"). When most pairs are NOT conflicting, the
    selective orthogonaliser skips them and pays only the backward
    cost; only conflicting pairs trigger projection. This converts
    PCGrad from "always pay N²" to "pay where it matters."

The probe assumes the orthogonaliser already runs each step (it needs
per-head trunk grads to compute pair cosines, and recomputing them
separately would double the backward cost). Specifically, after every
``PairedGradientOrthogonalizer.project`` call the per-head FLAT grads
are exposed via ``orthogonalizer.last_per_head_grads``; this probe
consumes that dict.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch

from .relational_gradients import cosine


@dataclass
class _LivePairStats:
    """EMA-tracked cosine + magnitude ratio for one ordered pair (a, b)."""

    cos_ema: float = float("nan")
    mag_a_ema: float = float("nan")
    mag_b_ema: float = float("nan")
    n_steps: int = 0
    n_conflicts: int = 0  # cumulative steps where cos < 0


class LiveRelationalProbe:
    """Per-pair running gradient stats consumed live during training.

    Parameters
    ----------
    head_names:
        Iterable of head names the probe should track. Pairs are formed
        as the ordered Cartesian product (excluding (a, a)).
    ema_alpha:
        EMA smoothing on cosine + magnitudes. Default 0.9 (recent-heavy).
    conflict_threshold:
        Pairs with EMA cosine BELOW this value are considered conflicting
        when ``should_project`` is queried. 0.0 means "any negative
        cosine"; <0 lets some negative-cosine pairs through unprojected.
    warmup_steps:
        For the first N steps, ``should_project`` returns True for all
        pairs — gives the EMA time to settle before gating PCGrad on it.
    """

    def __init__(
        self,
        head_names: Iterable[str],
        *,
        ema_alpha: float = 0.9,
        conflict_threshold: float = 0.0,
        warmup_steps: int = 50,
    ) -> None:
        names = list(head_names)
        if not names:
            raise ValueError("head_names must be non-empty")
        if not (0.0 < ema_alpha < 1.0):
            raise ValueError("ema_alpha must be in (0, 1)")
        self.head_names: tuple[str, ...] = tuple(names)
        self.ema_alpha = float(ema_alpha)
        self.conflict_threshold = float(conflict_threshold)
        self.warmup_steps = int(warmup_steps)
        self._pair_stats: dict[tuple[str, str], _LivePairStats] = {
            (a, b): _LivePairStats()
            for a in self.head_names
            for b in self.head_names
            if a != b
        }
        self.step_count: int = 0

    # ----- core update -----

    @torch.no_grad()
    def update(self, per_head_grads_flat: Mapping[str, torch.Tensor]) -> None:
        """EMA-update per-pair cosine and magnitudes from the latest
        flat per-head trunk gradients.

        Heads missing from ``per_head_grads_flat`` keep their previous
        EMA — they don't get zero'd out.
        """
        alpha = self.ema_alpha
        # Pre-compute magnitudes once per head.
        mags: dict[str, float] = {}
        for name in self.head_names:
            if name not in per_head_grads_flat:
                continue
            g = per_head_grads_flat[name]
            if not torch.is_tensor(g):
                continue
            mags[name] = float(g.norm().item())

        for (a, b), stats in self._pair_stats.items():
            if a not in per_head_grads_flat or b not in per_head_grads_flat:
                continue
            ga = per_head_grads_flat[a]
            gb = per_head_grads_flat[b]
            if not (torch.is_tensor(ga) and torch.is_tensor(gb)):
                continue
            c = cosine(ga, gb)
            if stats.cos_ema != stats.cos_ema:  # NaN: first observation
                stats.cos_ema = c
                stats.mag_a_ema = mags.get(a, 0.0)
                stats.mag_b_ema = mags.get(b, 0.0)
            else:
                stats.cos_ema = alpha * stats.cos_ema + (1.0 - alpha) * c
                stats.mag_a_ema = (
                    alpha * stats.mag_a_ema + (1.0 - alpha) * mags.get(a, 0.0)
                )
                stats.mag_b_ema = (
                    alpha * stats.mag_b_ema + (1.0 - alpha) * mags.get(b, 0.0)
                )
            stats.n_steps += 1
            if c < 0.0:
                stats.n_conflicts += 1
        self.step_count += 1

    # ----- query: should this pair be PCGrad-projected? -----

    def should_project(self, a: str, b: str) -> bool:
        """True iff the EMA cosine for the (a, b) pair is below threshold.

        During the first ``warmup_steps`` steps, always returns True so
        the orthogonaliser exercises every pair while the EMA settles.
        Returns True for pairs with no observation yet (NaN cosine).
        """
        if self.step_count < self.warmup_steps:
            return True
        # symmetric: try both orderings
        stats = self._pair_stats.get((a, b)) or self._pair_stats.get((b, a))
        if stats is None:
            return False
        if stats.cos_ema != stats.cos_ema:  # NaN
            return True
        return stats.cos_ema < self.conflict_threshold

    # ----- query: live conflict rate per task for balancer -----

    def per_task_conflict_rate(self) -> dict[str, float]:
        """Collapse per-pair conflict_rate to per-task by averaging across
        pairs containing each task.

        ``conflict_rate(task)`` = mean over pairs containing task of
        max(0, -cos_ema). i.e. negative cosines contribute, positive
        cosines contribute zero. Bounded to [0, 1].

        Tasks that have never been observed get rate 0.0.
        """
        rates: dict[str, list[float]] = defaultdict(list)
        for (a, b), stats in self._pair_stats.items():
            if stats.cos_ema != stats.cos_ema:
                continue
            penalty = max(0.0, -stats.cos_ema)
            rates[a].append(penalty)
        return {
            name: (sum(vals) / len(vals)) if vals else 0.0
            for name, vals in rates.items()
        }

    # ----- introspection -----

    def state_snapshot(self) -> dict[str, dict[str, float]]:
        """Per-pair snapshot for logging."""
        return {
            f"{a}|{b}": {
                "cos_ema": stats.cos_ema,
                "mag_a_ema": stats.mag_a_ema,
                "mag_b_ema": stats.mag_b_ema,
                "n_steps": float(stats.n_steps),
                "n_conflicts": float(stats.n_conflicts),
                "conflict_rate": (
                    stats.n_conflicts / stats.n_steps
                    if stats.n_steps > 0 else 0.0
                ),
            }
            for (a, b), stats in self._pair_stats.items()
        }


__all__ = ["LiveRelationalProbe"]
