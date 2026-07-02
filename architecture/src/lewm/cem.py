"""CEM planner with continuous Gaussian distributions over action sequences.

Paper-faithful Algorithm 2:

1. Initialize μ_0 = 0 (zero action sequence) and Σ_0 = I (independent unit
   variance per step and dimension).
2. For each iteration:
   a. Sample N candidate sequences from N(μ, diag(Σ²)).
   b. Score each via a user-supplied ``cost_fn`` taking ``[N, H, action_dim]``
      and returning ``[N]`` (lower is better).
   c. Pick the K elites (lowest cost).
   d. Update μ ← mean(elites), σ ← std(elites).
3. Return the elite mean's best sequence.

Defaults from paper Table: N=300 samples, K=30 elites, 30 iterations,
horizon=5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


PAPER_CEM_DEFAULTS = dict(
    num_samples=300, num_iterations=30, elite_count=30, horizon=5,
)

# Spark-tuned default — empirically derived from the rollout_bench
# batch-scaling curve (2026-05-11, GB10 sm_121, FP16+CUDA-graph production
# rollout). At B=300 the predictor runs at ~1925 GMAC/s; at B=1024 it
# runs at ~4041 GMAC/s — 2.1× better SM utilization for 1.6× the
# wall-clock per iteration. Net: ~5× more candidate samples scored per
# second. Use ``CEMConfig.paper_faithful()`` to opt back into N=300 for
# promotion-gate reproduction.
SPARK_TUNED_CEM_DEFAULTS = dict(
    num_samples=1024, num_iterations=30, elite_count=102, horizon=5,
)


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = SPARK_TUNED_CEM_DEFAULTS["horizon"]
    action_dim: int = 1
    num_samples: int = SPARK_TUNED_CEM_DEFAULTS["num_samples"]
    num_iterations: int = SPARK_TUNED_CEM_DEFAULTS["num_iterations"]
    elite_count: int = SPARK_TUNED_CEM_DEFAULTS["elite_count"]
    init_std: float = 1.0
    min_std: float = 1e-3
    smoothing: float = 0.0  # 0 = pure replacement; >0 mixes with prior

    @classmethod
    def paper_faithful(cls, **overrides) -> "CEMConfig":
        """Construct the paper Algorithm 2 defaults (N=300, K=30, 30 iters, H=5).

        Use this for promotion-gate / reproducibility work where the bench
        comparison must match the paper baseline. Production / planning
        code should prefer the default ``CEMConfig()`` which is tuned for
        the Spark's compute-density curve.
        """

        params = {**PAPER_CEM_DEFAULTS, **overrides}
        return cls(**params)

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.elite_count < 1 or self.elite_count > self.num_samples:
            raise ValueError("elite_count must be in [1, num_samples]")
        if self.init_std <= 0:
            raise ValueError("init_std must be > 0")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")


@dataclass
class CEMResult:
    best_sequence: torch.Tensor  # [horizon, action_dim]
    best_cost: float
    final_mean: torch.Tensor     # [horizon, action_dim]
    final_std: torch.Tensor      # [horizon, action_dim]
    iterations: int


class CEMPlanner:
    """Continuous-action Cross-Entropy Method planner."""

    def __init__(
        self,
        config: CEMConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or CEMConfig()
        self.config.validate()
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def plan(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> CEMResult:
        """``cost_fn`` takes ``[N, H, action_dim]`` and returns ``[N]`` costs."""

        cfg = self.config
        mean = torch.zeros(cfg.horizon, cfg.action_dim)
        std = torch.full((cfg.horizon, cfg.action_dim), cfg.init_std)

        best_cost = float("inf")
        best_sequence: torch.Tensor | None = None

        for _ in range(cfg.num_iterations):
            noise = torch.randn(
                cfg.num_samples,
                cfg.horizon,
                cfg.action_dim,
                generator=self.generator,
            )
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            costs = cost_fn(samples)
            if costs.dim() != 1 or costs.shape[0] != cfg.num_samples:
                raise ValueError(
                    f"cost_fn must return [num_samples={cfg.num_samples}], "
                    f"got {tuple(costs.shape)}"
                )

            elite_idx = torch.topk(
                costs, k=cfg.elite_count, largest=False,
            ).indices
            elites = samples[elite_idx]
            elite_costs = costs[elite_idx]

            min_idx = int(torch.argmin(elite_costs).item())
            if float(elite_costs[min_idx]) < best_cost:
                best_cost = float(elite_costs[min_idx])
                best_sequence = elites[min_idx].clone()

            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0, unbiased=False).clamp_min(cfg.min_std)
            if cfg.smoothing > 0.0:
                mean = (1.0 - cfg.smoothing) * new_mean + cfg.smoothing * mean
                std = (1.0 - cfg.smoothing) * new_std + cfg.smoothing * std
            else:
                mean = new_mean
                std = new_std

        if best_sequence is None:
            raise RuntimeError("CEM failed to score any sequence")

        return CEMResult(
            best_sequence=best_sequence,
            best_cost=best_cost,
            final_mean=mean,
            final_std=std,
            iterations=cfg.num_iterations,
        )
