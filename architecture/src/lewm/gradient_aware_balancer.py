"""Gradient-aware extension of the Shannon adaptive loss balancer.

The base :class:`ShannonAdaptiveBalancer` weights tasks by the Shannon
entropy of their **predictions**. That captures "how uncertain is the
head about its output". It does NOT capture:

  (a) How broadly distributed the head's TRUNK GRADIENT is. A gradient
      concentrated on a handful of parameters is exploitative (sharp
      direction); a gradient spread thinly across many parameters is
      exploratory and carries less risk of overfitting any single
      direction. Shannon entropy of the |g_i| / Σ|g_j| distribution
      quantifies that breadth.

  (b) Per-task CONFLICT RATE — how often a task's trunk gradient is
      negatively cosine-aligned with sibling tasks. The
      RelationalGradientProbe records this per pair; collapsing across
      pairs yields one number per task. Tasks that fight others should
      be down-weighted to dampen destructive interference.

We combine the three signals MULTIPLICATIVELY:

    eff_w_i = base_weight_i
              × pred_entropy_mult_i      # exploratory pressure on outputs
              × grad_entropy_mult_i      # exploratory pressure on params
              × conflict_factor_i        # peace penalty: 1/(1+α·conflict)

Multiplication is the right composition because each signal is a
*modifier* on a base weight. The user calibrated base_weight_i; the
three multipliers each shift it by a relative factor of order ~1.
Additive composition would force the operator to pick a global scale
for each signal; multiplication is scale-free and lets a strong
"conflict" signal dominate a weak "entropy" signal organically.

This file does NOT modify ``ShannonAdaptiveBalancer`` — it wraps one
by composition.
"""

from __future__ import annotations

from typing import Sequence

import torch

from .adaptive_loss_balance import ShannonAdaptiveBalancer, TaskSpec


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def compute_gradient_distribution_entropy(g: torch.Tensor, *, eps: float = 1e-12) -> float:
    """Shannon entropy of the normalized |g| distribution.

    For a flat gradient tensor ``g``, define a probability mass over
    parameter indices::

        p_i = |g_i| / Σ_j |g_j|

    and return ``H(p) = -Σ p_i log p_i``.

    Edge cases:
      - All zeros → 0.0 (no signal at all; treat as fully concentrated).
      - One-hot (only one nonzero element) → 0.0.
      - Uniform |g| over N elements → log(N).

    The log is stabilised with ``eps`` so zero-probability bins
    contribute zero (since 0 · log(eps) → 0 in the limit we mask off).
    """
    if not torch.is_tensor(g):
        raise TypeError("g must be a torch.Tensor")
    if g.numel() == 0:
        return 0.0
    flat = g.detach().reshape(-1).float().abs()
    total = float(flat.sum().item())
    if total <= 0.0:
        return 0.0
    p = flat / total
    # Only nonzero bins contribute. We use a stabilised log on the
    # nonzero subset so the eps doesn't leak into the entropy value.
    mask = p > 0.0
    if not bool(mask.any()):
        return 0.0
    p_nz = p[mask]
    h = -(p_nz * (p_nz + eps).log()).sum()
    return float(h.item())


# ---------------------------------------------------------------------------
# GradientAwareBalancer
# ---------------------------------------------------------------------------


class GradientAwareBalancer:
    """Compose Shannon adaptive balancing with gradient-entropy and
    conflict-rate awareness.

    Usage::

        bal = GradientAwareBalancer(
            tasks=[...],
            use_prediction_entropy=True,
            use_gradient_entropy=True,
            use_conflict_penalty=True,
        )
        for batch in loader:
            outputs = model(batch)
            bal.update_predictions(outputs, batch)
            # (after per-head backward passes)
            bal.update_gradients(per_head_grads_flat)
            # (after the probe has accumulated enough stats)
            bal.update_conflicts(per_task_conflict_rate)
            weights = bal.effective_weights()
    """

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        *,
        use_prediction_entropy: bool = True,
        use_gradient_entropy: bool = False,
        use_conflict_penalty: bool = False,
        gradient_entropy_temperature: float = 1.0,
        conflict_penalty_alpha: float = 1.0,
        ema_alpha: float = 0.95,
        temperature: float = 1.0,
        min_multiplier: float = 0.1,
        max_multiplier: float = 5.0,
        warmup_steps: int = 50,
    ) -> None:
        if gradient_entropy_temperature <= 0.0:
            raise ValueError("gradient_entropy_temperature must be > 0")
        if conflict_penalty_alpha < 0.0:
            raise ValueError("conflict_penalty_alpha must be >= 0")

        # Inner balancer owns the prediction-entropy logic + base_weight book.
        self._inner = ShannonAdaptiveBalancer(
            tasks,
            ema_alpha=ema_alpha,
            temperature=temperature,
            min_multiplier=min_multiplier,
            max_multiplier=max_multiplier,
            warmup_steps=warmup_steps,
        )
        self.tasks: dict[str, TaskSpec] = dict(self._inner.tasks)

        self.use_prediction_entropy = bool(use_prediction_entropy)
        self.use_gradient_entropy = bool(use_gradient_entropy)
        self.use_conflict_penalty = bool(use_conflict_penalty)
        self.gradient_entropy_temperature = float(gradient_entropy_temperature)
        self.conflict_penalty_alpha = float(conflict_penalty_alpha)

        self.ema_alpha = float(ema_alpha)
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)

        # NaN sentinel = "no observation yet"
        self._grad_entropy_ema: dict[str, float] = {n: float("nan") for n in self.tasks}
        self._grad_observed: dict[str, bool] = {n: False for n in self.tasks}
        self._conflict_rate: dict[str, float] = {n: 0.0 for n in self.tasks}
        self._conflict_observed: dict[str, bool] = {n: False for n in self.tasks}

    # ----- prediction-side -----

    def update_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Proxy to the inner ShannonAdaptiveBalancer's prediction-entropy update."""
        self._inner.update(outputs, batch=batch)

    # Alias matching ShannonAdaptiveBalancer.update so callers that hold
    # an opaque balancer reference (e.g. PropertyTrainer) work with either.
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.update_predictions(outputs, batch=batch)

    # ----- gradient-side -----

    @torch.no_grad()
    def update_gradients(self, per_head_grads_flat: dict[str, torch.Tensor]) -> None:
        """EMA-update the per-task gradient-distribution Shannon entropy.

        ``per_head_grads_flat`` maps task name → flat 1-D gradient
        tensor (the same shape the RelationalGradientProbe produces via
        ``flatten_trunk_grads``). Tasks missing from the dict keep
        their previous EMA.
        """
        for name in self.tasks:
            if name not in per_head_grads_flat:
                continue
            g = per_head_grads_flat[name]
            if not torch.is_tensor(g):
                continue
            h = compute_gradient_distribution_entropy(g)
            prev = self._grad_entropy_ema[name]
            if prev != prev:  # NaN: first observation
                self._grad_entropy_ema[name] = h
            else:
                self._grad_entropy_ema[name] = (
                    self.ema_alpha * prev + (1.0 - self.ema_alpha) * h
                )
            self._grad_observed[name] = True

    def update_conflicts(self, conflict_rate: dict[str, float]) -> None:
        """Set per-task conflict-rate.

        Each value is the average conflict_rate across pairs that
        contain this task — typically obtained by collapsing
        ``RelationalGradientProbe.report()["per_pair"]`` across pairs.
        """
        for name, rate in conflict_rate.items():
            if name not in self.tasks:
                continue
            r = float(rate)
            if r < 0.0:
                r = 0.0
            self._conflict_rate[name] = r
            self._conflict_observed[name] = True

    # ----- weight emission -----

    def _gradient_entropy_multipliers(self) -> dict[str, float]:
        """Softmax-normalize gradient entropies across observed tasks,
        mean-normalize to 1.0, and clip.

        Returns 1.0 for tasks with no observed gradient entropy yet.
        """
        active = {
            n: self._grad_entropy_ema[n]
            for n, obs in self._grad_observed.items()
            if obs and self._grad_entropy_ema[n] == self._grad_entropy_ema[n]  # not NaN
        }
        if not active:
            return {n: 1.0 for n in self.tasks}
        names = list(active)
        vals = torch.tensor([active[n] for n in names], dtype=torch.float64)
        weights = torch.softmax(vals / self.gradient_entropy_temperature, dim=0)
        weights = weights * len(names) / weights.sum().clamp_min(1e-12)
        weights = weights.clamp(self.min_multiplier, self.max_multiplier)
        result = {n: 1.0 for n in self.tasks}
        for i, n in enumerate(names):
            result[n] = float(weights[i].item())
        return result

    def _conflict_factors(self) -> dict[str, float]:
        """Convert per-task conflict_rate into a fractional penalty.

        ``factor = 1 / (1 + α · rate)`` — monotonically decreasing in
        rate, with factor=1.0 when rate=0. α=1 (default) means a
        task that conflicts on every step is down-weighted to 0.5.
        """
        return {
            n: 1.0 / (1.0 + self.conflict_penalty_alpha * self._conflict_rate[n])
            for n in self.tasks
        }

    def effective_weights(self) -> dict[str, float]:
        """Final per-task weight = base × pred_mult × grad_mult × conflict_factor.

        Each multiplier is 1.0 when its corresponding flag is False, so
        ``effective_weights()`` is identical to the inner Shannon
        balancer's when ``use_gradient_entropy`` and
        ``use_conflict_penalty`` are both False.
        """
        if self.use_prediction_entropy:
            pred_mults = self._inner.adaptive_multipliers()
        else:
            pred_mults = {n: 1.0 for n in self.tasks}

        if self.use_gradient_entropy:
            grad_mults = self._gradient_entropy_multipliers()
        else:
            grad_mults = {n: 1.0 for n in self.tasks}

        if self.use_conflict_penalty:
            conflict_factors = self._conflict_factors()
        else:
            conflict_factors = {n: 1.0 for n in self.tasks}

        out: dict[str, float] = {}
        for name, spec in self.tasks.items():
            out[name] = (
                float(spec.base_weight)
                * pred_mults[name]
                * grad_mults[name]
                * conflict_factors[name]
            )
        return out

    # ----- introspection -----

    def state_snapshot(self) -> dict[str, dict[str, float]]:
        """Per-task entropy / gradient-entropy / conflict / effective weight."""
        pred_snap = self._inner.state_snapshot()
        grad_mults = self._gradient_entropy_multipliers()
        conflict_factors = self._conflict_factors()
        eff = self.effective_weights()
        out: dict[str, dict[str, float]] = {}
        for name in self.tasks:
            g_ema = self._grad_entropy_ema[name]
            out[name] = {
                "entropy_ema": pred_snap[name]["entropy_ema"],
                # Alias matching ShannonAdaptiveBalancer's snapshot key so
                # downstream printers / dashboards keep working.
                "adaptive_multiplier": pred_snap[name]["adaptive_multiplier"],
                "pred_multiplier": pred_snap[name]["adaptive_multiplier"],
                "grad_entropy_ema": (
                    float(g_ema) if g_ema == g_ema else float("nan")
                ),
                "grad_multiplier": grad_mults[name],
                "conflict_rate": self._conflict_rate[name],
                "conflict_factor": conflict_factors[name],
                "base_weight": float(self.tasks[name].base_weight),
                "effective_weight": eff[name],
                "pred_observed": bool(pred_snap[name]["observed"]),
                "grad_observed": bool(self._grad_observed[name]),
                "conflict_observed": bool(self._conflict_observed[name]),
            }
        return out

    @property
    def step_count(self) -> int:
        return self._inner.step_count

    @property
    def inner(self) -> ShannonAdaptiveBalancer:
        """Access the wrapped Shannon balancer (for tests/introspection)."""
        return self._inner

    # ----- DDP sync -----

    def sync_across_ranks(self) -> None:
        """Sync inner balancer + gradient-entropy and conflict-rate EMAs.

        Each rank maintains local EMAs from its local backward passes;
        without averaging across ranks the per-rank weights would
        diverge step-by-step. Safe to call when not running
        distributed (no-op).
        """
        self._inner.sync_across_ranks()
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not (dist.is_available() and dist.is_initialized()):
            return
        world = dist.get_world_size()
        if world < 2:
            return

        names = sorted(self.tasks)
        # Layout: [grad_ema_i, grad_obs_i, conflict_i, conflict_obs_i] per task.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.zeros(len(names) * 4, dtype=torch.float64, device=device)
        for i, n in enumerate(names):
            g = self._grad_entropy_ema[n]
            if g == g and self._grad_observed[n]:
                payload[4 * i] = float(g)
                payload[4 * i + 1] = 1.0
            if self._conflict_observed[n]:
                payload[4 * i + 2] = float(self._conflict_rate[n])
                payload[4 * i + 3] = 1.0
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)

        for i, n in enumerate(names):
            g_obs = float(payload[4 * i + 1].item())
            if g_obs > 0:
                self._grad_entropy_ema[n] = float(payload[4 * i].item() / g_obs)
                self._grad_observed[n] = True
            c_obs = float(payload[4 * i + 3].item())
            if c_obs > 0:
                self._conflict_rate[n] = float(payload[4 * i + 2].item() / c_obs)
                self._conflict_observed[n] = True
