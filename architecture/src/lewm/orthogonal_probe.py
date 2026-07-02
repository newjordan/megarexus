"""Probe-mode wrapper around the PCGrad-style orthogonal train step.

This module is a *diagnostic* sibling of
:func:`lewm.orthogonal_train_step.orthogonal_step`. It runs one
PCGrad-style step end-to-end (so head-local grads are correct and
trunk grads are projected) and snapshots three families of per-step
statistics:

  * per-head trunk-gradient L2 norm
  * per-head trunk-gradient distribution Shannon entropy (the same
    quantity :class:`lewm.gradient_aware_balancer.GradientAwareBalancer`
    EMA-tracks internally)
  * per-pair cosine similarity + a binary "conflict" indicator
    (``cos < 0``)

Plus an optional snapshot of the balancer state so the recorded run
includes the live effective weights at each step.

The class is intentionally *not* a training-loop replacement — it's a
post-hoc / smoke-time tool for understanding which head-pairs fight,
how much, and how the balancer responds. Calls are O(N²) in the number
of heads (cosine per pair) on top of the orthogonaliser cost.

Returned ``summary()`` dict has three top-level keys:

  ``per_step``  list of per-step records (one entry per ``record_step``).
  ``per_pair``  aggregated cosine stats per unordered head-pair.
  ``per_task``  aggregated per-head stats (grad-norm, grad-entropy, etc).

Everything is JSON-serialisable so callers can persist the result with
``to_json(path)`` and analyse later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch

from .gradient_aware_balancer import (
    GradientAwareBalancer,
    compute_gradient_distribution_entropy,
)
from .orthogonal_train_step import orthogonal_step
from .relational_gradients import (
    PairedGradientOrthogonalizer,
    cosine,
    flatten_trunk_grads,
)


class OrthogonalGradientProbe:
    """Diagnostic recorder for PCGrad-style orthogonal training steps.

    Parameters
    ----------
    orthogonalizer:
        Optional pre-built :class:`PairedGradientOrthogonalizer`. If the
        caller doesn't pass one here, they must pass one to every
        :meth:`record_step` call instead.
    balancer:
        Optional :class:`GradientAwareBalancer`. When supplied, each
        recorded step also includes a snapshot of the balancer's
        effective weights + per-task entropy/conflict bookkeeping.
    """

    def __init__(
        self,
        orthogonalizer: Optional[PairedGradientOrthogonalizer] = None,
        balancer: Optional[GradientAwareBalancer] = None,
    ) -> None:
        self.orthogonalizer = orthogonalizer
        self.balancer = balancer
        # Per-step records; each entry is a JSON-serialisable dict.
        self._steps: list[dict] = []

    # ----- recording -----

    def record_step(
        self,
        model: torch.nn.Module,
        head_losses: Mapping[str, torch.Tensor],
        total_loss: Optional[torch.Tensor],
        trunk_prefixes: Sequence[str],
        orthogonalizer: Optional[PairedGradientOrthogonalizer] = None,
    ) -> dict:
        """Run one PCGrad-style step and snapshot per-head/per-pair stats.

        After this returns, the model's ``.grad`` fields are populated
        as if :func:`orthogonal_step` had been called directly — the
        caller can immediately do ``optimizer.step()`` if they want a
        live training side-effect. (The probe is otherwise read-only.)

        Returns the dict that was appended to the internal record list.
        """
        orth = orthogonalizer if orthogonalizer is not None else self.orthogonalizer
        if orth is None:
            raise ValueError(
                "record_step needs an orthogonalizer (either via constructor "
                "or explicit argument)"
            )

        # Capture per-head trunk grads via the orthogonaliser's own
        # ``last_per_head_grads`` book — populated as a side effect of
        # ``project``, which ``orthogonal_step`` calls internally.
        orthogonal_step(
            model=model,
            head_losses=head_losses,
            total_loss=total_loss,
            trunk_prefixes=trunk_prefixes,
            orthogonalizer=orth,
        )
        per_head_grads = dict(orth.last_per_head_grads)  # task → flat tensor

        # ----- per-head stats -----
        per_head: dict[str, dict[str, float]] = {}
        for name, g in per_head_grads.items():
            per_head[name] = {
                "norm": float(g.norm().item()),
                "entropy": float(compute_gradient_distribution_entropy(g)),
            }

        # ----- per-pair stats -----
        names = sorted(per_head_grads.keys())
        per_pair: list[dict] = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                ga, gb = per_head_grads[a], per_head_grads[b]
                cos_val = cosine(ga, gb)
                per_pair.append({
                    "head_a": a,
                    "head_b": b,
                    "cosine": float(cos_val),
                    "conflict": bool(cos_val < 0.0),
                })

        # ----- balancer snapshot (if available) -----
        balancer_state: dict | None = None
        if self.balancer is not None:
            # state_snapshot is already JSON-serialisable (dict[str, dict[str, float]])
            balancer_state = {
                name: dict(per_task)
                for name, per_task in self.balancer.state_snapshot().items()
            }

        record = {
            "step": len(self._steps),
            "per_head": per_head,
            "per_pair": per_pair,
            "balancer": balancer_state,
        }
        self._steps.append(record)
        return record

    # ----- aggregation / export -----

    def summary(self) -> dict:
        """Aggregate the per-step records into per-pair and per-task views.

        Returns a dict with keys ``per_step``, ``per_pair``, ``per_task``.
        Pair stats: cosine mean/std/min/max + conflict rate over recorded
        steps. Task stats: gradient norm mean/max + gradient entropy
        mean over recorded steps.
        """
        # --- per_pair aggregation ---
        pair_acc: dict[tuple[str, str], dict[str, float]] = {}
        for rec in self._steps:
            for p in rec["per_pair"]:
                key = (p["head_a"], p["head_b"])
                acc = pair_acc.setdefault(key, {
                    "n": 0, "cos_sum": 0.0, "cos_sq_sum": 0.0,
                    "cos_min": float("inf"), "cos_max": float("-inf"),
                    "n_conflicts": 0,
                })
                c = float(p["cosine"])
                acc["n"] += 1
                acc["cos_sum"] += c
                acc["cos_sq_sum"] += c * c
                acc["cos_min"] = min(acc["cos_min"], c)
                acc["cos_max"] = max(acc["cos_max"], c)
                if p["conflict"]:
                    acc["n_conflicts"] += 1

        per_pair: list[dict] = []
        for (a, b), acc in pair_acc.items():
            n = max(1, int(acc["n"]))
            mean = acc["cos_sum"] / n
            var = (acc["cos_sq_sum"] / n) - mean * mean
            per_pair.append({
                "head_a": a,
                "head_b": b,
                "n_steps": int(acc["n"]),
                "cos_mean": float(mean),
                "cos_std": float(var ** 0.5) if var > 0 else 0.0,
                "cos_min": float(acc["cos_min"]) if acc["n"] else 0.0,
                "cos_max": float(acc["cos_max"]) if acc["n"] else 0.0,
                "conflict_rate": float(acc["n_conflicts"]) / n,
            })
        per_pair.sort(key=lambda r: -r["conflict_rate"])

        # --- per_task aggregation ---
        task_acc: dict[str, dict[str, float]] = {}
        for rec in self._steps:
            for name, vals in rec["per_head"].items():
                acc = task_acc.setdefault(name, {
                    "n": 0, "norm_sum": 0.0, "norm_max": 0.0,
                    "entropy_sum": 0.0,
                })
                acc["n"] += 1
                acc["norm_sum"] += float(vals["norm"])
                acc["norm_max"] = max(acc["norm_max"], float(vals["norm"]))
                acc["entropy_sum"] += float(vals["entropy"])

        per_task: dict[str, dict[str, float]] = {}
        for name, acc in task_acc.items():
            n = max(1, int(acc["n"]))
            per_task[name] = {
                "n_steps": int(acc["n"]),
                "norm_mean": float(acc["norm_sum"]) / n,
                "norm_max": float(acc["norm_max"]),
                "entropy_mean": float(acc["entropy_sum"]) / n,
            }

        return {
            "per_step": list(self._steps),
            "per_pair": per_pair,
            "per_task": per_task,
        }

    def to_json(self, path: str | Path) -> None:
        """Write the summary to ``path`` as pretty-printed JSON."""
        data = self.summary()
        Path(path).write_text(json.dumps(data, indent=2, sort_keys=False))

    # ----- ergonomics -----

    def __len__(self) -> int:
        return len(self._steps)


__all__ = ["OrthogonalGradientProbe"]
