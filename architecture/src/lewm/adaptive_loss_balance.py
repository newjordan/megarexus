"""Shannon-entropy-driven adaptive loss balancer for multi-task training.

Replaces the static ``*_weight`` constants in PropertyTrainConfig with
runtime-adaptive weights. Core idea:

  - Each task's prediction distribution carries some Shannon entropy
    H(p̂_i). For classification heads this is −Σp̂·log p̂; for regression
    heads we proxy with the std of predictions over the batch.
  - High H → model is uncertain → task still has gradient signal worth
    pursuing → weight should be high.
  - Low H → model is confident → task either learned or stuck → weight
    decays so other tasks see relatively more pressure.
  - Weights are smoothed over time via EMA so individual batch noise
    doesn't whiplash the loss landscape.

The balancer modulates **on top of** the user's heuristic base weights:

    effective_weight_i = base_weight_i * adaptive_multiplier_i

where ``adaptive_multiplier_i`` is normalised so its mean across active
tasks is 1.0. This means turning the balancer on doesn't break the
relative magnitudes the user calibrated — it just shifts pressure
across tasks adaptively.

Compared to alternatives:
  - **GradNorm** (Chen et al 2018) — learns weights via a separate
    optimizer step on gradient-norm divergence from a target ratio.
    More principled but doubles training compute; we don't need that.
  - **Uncertainty-based** (Kendall et al 2018) — learns σ_i² per task
    as a parameter, weight = 1/(2σ²). Coupled to the loss in a way
    that can collapse if σ blows up.
  - **DWA** (Liu et al 2019) — weights by L_i(t-1) / L_i(t-2). Pure
    rate-based, no semantic signal about the task's prediction
    distribution.

This balancer is closest to a Bayesian-curiosity prior: weight the
gradient toward tasks where the model is most uncertain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class TaskSpec:
    """Description of one task the balancer manages.

    ``logits_key``: the dict-key in the model's forward output. Must
    yield a tensor whose last dim is the class/output count (for
    classification) or 1+ regression dims.

    ``kind``: ``"classification"`` (use softmax entropy) or
    ``"regression"`` (use prediction std).

    ``base_weight``: the user's heuristic weight. Used multiplicatively
    on top of the adaptive multiplier; the balancer doesn't replace
    the operator's intent.

    ``mask_key``: optional batch-dict key holding a bool mask over
    examples. When set, entropy is averaged only over masked-True
    examples. Useful for sparse-supervision heads like role_buckets.

    ``regression_sigma_cap``: regression-only. The regression entropy
    proxy ``0.5·ln(2πe·σ²)`` is unbounded in σ — a diverging head's
    prediction-std grows without limit, the balancer reads that as
    rising "uncertainty" and up-weights the head, accelerating the
    divergence (a positive feedback loop; see the dpx blow-up in the
    precision regime log). Capping σ at this value before the log
    saturates the proxy: a healthy head varies below the cap and is
    unaffected; a diverging head's entropy plateaus instead of running
    away. ``None`` = no cap (legacy behaviour).
    """

    name: str
    logits_key: str
    kind: str = "classification"
    base_weight: float = 1.0
    mask_key: str | None = None
    regression_sigma_cap: float | None = None


class ShannonAdaptiveBalancer:
    """Maintain per-task entropy EMAs; emit adaptive multipliers.

    Usage in a training loop:

        balancer = ShannonAdaptiveBalancer([...], ema_alpha=0.95)
        for batch in loader:
            outputs = model(...)
            balancer.update(outputs, batch=batch)
            weights = balancer.effective_weights()
            losses = loss_fn(outputs, batch, **weights)
            losses["total"].backward()
    """

    def __init__(
        self,
        tasks: Sequence[TaskSpec],
        *,
        ema_alpha: float = 0.95,
        temperature: float = 1.0,
        min_multiplier: float = 0.1,
        max_multiplier: float = 5.0,
        warmup_steps: int = 50,
    ) -> None:
        if not (0.0 < ema_alpha < 1.0):
            raise ValueError("ema_alpha must be in (0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if min_multiplier <= 0.0 or min_multiplier >= max_multiplier:
            raise ValueError("0 < min_multiplier < max_multiplier required")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")

        self.tasks: dict[str, TaskSpec] = {t.name: t for t in tasks}
        if len(self.tasks) != len(list(tasks)):
            raise ValueError("task names must be unique")

        self.ema_alpha = float(ema_alpha)
        self.temperature = float(temperature)
        self.min_multiplier = float(min_multiplier)
        self.max_multiplier = float(max_multiplier)
        self.warmup_steps = int(warmup_steps)

        self._entropy_ema: dict[str, float] = {n: float("nan") for n in self.tasks}
        self._step: int = 0
        self._observed_at_least_once: dict[str, bool] = {n: False for n in self.tasks}

    # ----- update step -----

    @torch.no_grad()
    def update(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Refresh entropy EMAs from the current forward outputs.

        Only updates EMAs for tasks whose ``logits_key`` is present in
        ``outputs``. Missing tasks (e.g. a head disabled this run)
        silently skip; their EMA stays NaN and they end up at base_weight.
        """
        self._step += 1
        for name, spec in self.tasks.items():
            if spec.logits_key not in outputs:
                continue
            logits = outputs[spec.logits_key]
            mask = None
            if spec.mask_key is not None and batch is not None and spec.mask_key in batch:
                mask = batch[spec.mask_key]
                if isinstance(mask, torch.Tensor):
                    mask = mask.to(device=logits.device, dtype=torch.bool)

            h = _entropy(
                logits, kind=spec.kind, mask=mask,
                sigma_cap=spec.regression_sigma_cap,
            )
            if h is None:
                continue

            prev = self._entropy_ema[name]
            if prev != prev:  # NaN check (no value yet)
                self._entropy_ema[name] = h
            else:
                self._entropy_ema[name] = (
                    self.ema_alpha * prev + (1.0 - self.ema_alpha) * h
                )
            self._observed_at_least_once[name] = True

    # ----- weight emission -----

    def adaptive_multipliers(self) -> dict[str, float]:
        """Per-task multiplicative factor; mean across active tasks = 1.0.

        During warmup, returns 1.0 for every task so training is stable
        before entropy estimates settle.
        """
        active = {
            name: self._entropy_ema[name]
            for name, observed in self._observed_at_least_once.items()
            if observed and self._entropy_ema[name] == self._entropy_ema[name]  # not NaN
        }
        if not active or self._step < self.warmup_steps:
            return {name: 1.0 for name in self.tasks}

        # Softmax-style normalization over entropies. Higher entropy → higher weight.
        names = list(active)
        vals = torch.tensor([active[n] for n in names], dtype=torch.float64)
        # Temperature-scaled softmax. tau small → sharp (winner-take-all);
        # tau large → flat (uniform).
        weights = torch.softmax(vals / self.temperature, dim=0)
        # Rescale so mean is 1.0 across active tasks
        weights = weights * len(names) / weights.sum().clamp_min(1e-12)
        # Clip
        weights = weights.clamp(self.min_multiplier, self.max_multiplier)

        result = {name: 1.0 for name in self.tasks}  # default for inactive
        for i, n in enumerate(names):
            result[n] = float(weights[i].item())
        return result

    def effective_weights(self) -> dict[str, float]:
        """Final weight per task = base_weight × adaptive_multiplier."""
        mults = self.adaptive_multipliers()
        return {
            name: float(spec.base_weight) * mults[name]
            for name, spec in self.tasks.items()
        }

    # ----- introspection -----

    def state_snapshot(self) -> dict[str, dict[str, float]]:
        """For logging: per-task EMA + multiplier + effective weight."""
        mults = self.adaptive_multipliers()
        return {
            name: {
                "entropy_ema": (
                    float(self._entropy_ema[name])
                    if self._entropy_ema[name] == self._entropy_ema[name]
                    else float("nan")
                ),
                "adaptive_multiplier": mults[name],
                "base_weight": float(self.tasks[name].base_weight),
                "effective_weight": float(self.tasks[name].base_weight) * mults[name],
                "observed": bool(self._observed_at_least_once[name]),
            }
            for name in self.tasks
        }

    @property
    def step_count(self) -> int:
        return self._step

    # ----- DDP sync -----

    def sync_across_ranks(self) -> None:
        """All-reduce per-task entropy EMAs across the DDP process group.

        Each DDP rank maintains its own EMA based on its local
        gradient updates. Without syncing, ranks would compute
        per-step weights from divergent EMAs and gradient-divergence
        compounds. This method averages the EMAs across ranks so all
        ranks see the same effective_weights() at every step.

        Safe to call when not running distributed (no-op).
        Tasks that have not been observed on ANY rank stay un-observed.
        """
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
        # Layout: [ema_0, obs_0, ema_1, obs_1, ...].
        # When NaN locally, ema=0 and obs=0 so it doesn't bias the mean.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.zeros(len(names) * 2, dtype=torch.float64, device=device)
        for i, n in enumerate(names):
            ema = self._entropy_ema[n]
            if ema == ema and self._observed_at_least_once[n]:  # not NaN, observed
                payload[2 * i] = float(ema)
                payload[2 * i + 1] = 1.0
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)

        for i, n in enumerate(names):
            obs_sum = float(payload[2 * i + 1].item())
            if obs_sum > 0:
                self._entropy_ema[n] = float(payload[2 * i].item() / obs_sum)
                self._observed_at_least_once[n] = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entropy(
    logits: torch.Tensor,
    *,
    kind: str = "classification",
    mask: torch.Tensor | None = None,
    sigma_cap: float | None = None,
) -> float | None:
    """Per-batch Shannon entropy estimate.

    Classification: H(p̂) = −Σ softmax(logits) · log_softmax(logits),
                    averaged over the batch (and over the mask if given).
    Regression:     proxy via std of predictions; treats high variance
                    as "high informational pressure".

    Returns None when there's no signal to estimate (empty mask, etc.).
    """
    if logits.numel() == 0:
        return None
    if kind == "classification":
        if logits.dim() < 2 or logits.shape[-1] < 2:
            return None
        # Flatten leading dims for masked or [B, ..., C] tensors
        flat = logits.reshape(-1, logits.shape[-1]).float()
        log_probs = F.log_softmax(flat, dim=-1)
        probs = log_probs.exp()
        # H(p̂) per example: −Σ p · log p
        per_ex = -(probs * log_probs).sum(dim=-1)
        if mask is not None:
            # Mask may be per-example ([..., ] matching per_ex's shape)
            # OR per-element ([..., C] same as logits). Reduce the
            # per-element form down by ANY across the class axis.
            mask_t = mask.bool()
            if mask_t.shape == logits.shape:
                mask_t = mask_t.any(dim=-1)
            mask_flat = mask_t.reshape(-1)
            if mask_flat.sum().item() == 0:
                return None
            per_ex = per_ex[mask_flat]
        return float(per_ex.mean().item())
    elif kind == "binary":
        # Multi-label BCE-style head: each cell is its own binary
        # classifier. Per-cell entropy: −p log p − (1−p) log(1−p).
        flat = logits.reshape(-1).float()
        probs = torch.sigmoid(flat)
        # Stabilised binary entropy
        eps = 1e-12
        per_cell = -(probs * (probs + eps).log() + (1 - probs) * (1 - probs + eps).log())
        if mask is not None:
            mask_flat = mask.reshape(-1).bool()
            if mask_flat.sum().item() == 0:
                return None
            per_cell = per_cell[mask_flat]
        return float(per_cell.mean().item())
    elif kind == "regression":
        # Use Gaussian differential entropy as the proxy so the value is
        # on the same nats-of-uncertainty scale as classification entropy.
        # H_gauss = 0.5 * ln(2πe * σ²) = 1.4189 + ln(σ)
        flat = logits.reshape(-1).float()
        if flat.numel() < 2:
            return None
        sigma = float(flat.std().item())
        if sigma <= 0.0:
            return 0.0
        # Saturate σ at the cap so a diverging head's entropy proxy
        # plateaus instead of running away (see TaskSpec docstring).
        if sigma_cap is not None and sigma > sigma_cap:
            sigma = sigma_cap
        import math
        return 0.5 * math.log(2.0 * math.pi * math.e * (sigma ** 2))
    else:
        raise ValueError(f"unknown task kind: {kind!r}")


def default_lewis_tasks(
    *,
    mechanic_weight: float = 1.0,
    family_weight: float = 1.0,
    subclass_weight: float = 0.5,
    stage_weight: float = 0.25,
    goal_type_weight: float = 0.5,
    keys_weight: float = 0.5,
    agent_weight: float = 1.0,
    delta_px_weight: float = 0.1,
    goal_family_weight: float = 0.5,
    role_buckets_weight: float = 0.3,
    capability_weight: float = 0.2,
    topology_weight: float = 0.2,
    cond_mechanic_weight: float = 0.5,
) -> list[TaskSpec]:
    """Default task list matching the Lewis loss surface.

    Pass these into ``ShannonAdaptiveBalancer(tasks=...)``. The base
    weights default to the calibrated values from
    ``scripts/launch_lewis_full_v5_b200.sh``.
    """
    return [
        TaskSpec("mechanic",      "mechanic_logits",      "classification", mechanic_weight),
        TaskSpec("family",        "family_logits",        "classification", family_weight),
        TaskSpec("subclass",      "subclass_logits",      "classification", subclass_weight),
        TaskSpec("stage",         "stage_logits",         "classification", stage_weight),
        TaskSpec("goal_type",     "goal_type_logits",     "classification", goal_type_weight),
        TaskSpec("keys",          "keys_logits",          "binary",         keys_weight),
        # Regression heads carry a sigma cap so a diverging head can't
        # feed itself ever-more weight (the dpx blow-up). Caps are loose
        # — agent_delta lives in grid units (~[-64,64]), delta_px is a
        # pixel distance on a 64x64 grid — so healthy training is
        # unaffected; only true runaway is clipped. Tune via probe.
        TaskSpec("agent",         "agent_delta",          "regression",     agent_weight,
                 regression_sigma_cap=32.0),
        TaskSpec("delta_px",      "delta_px",             "regression",     delta_px_weight,
                 regression_sigma_cap=64.0),
        TaskSpec("goal_family",   "goal_family_logits",   "classification", goal_family_weight),
        TaskSpec("role_buckets",  "role_buckets_logits",  "binary",         role_buckets_weight,
                 mask_key="role_buckets_mask"),
        TaskSpec("capability",    "capability_logits",    "binary",         capability_weight,
                 mask_key="capability_mask"),
        TaskSpec("topology",      "topology_logits",      "classification", topology_weight,
                 mask_key="topology_mask"),
        TaskSpec("cond_mechanic", "cond_mechanic_logits", "classification", cond_mechanic_weight),
    ]
