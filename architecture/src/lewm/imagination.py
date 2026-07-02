"""Imagination substrate — composable PUCT prior provider.

A "gentle informer" is a small algorithm (heuristic, learned head, judge)
that nudges the planner toward better branches without overriding it.
PUCT's selection score is ``Q(s,a) + c·P(s,a)·sqrt(N)/(1+n)``; the prior
``P`` is the single tensor where gentle advice meets hard search.

This module wraps an arbitrary collection of informers into one mixer that
matches the :data:`lewm.mcts.ClusterPriorFn` signature, so it drops into
``MCTSConfig.cluster_prior_fn`` with zero MCTS code change.

Each informer returns logits over ``available_actions``. The mixer:
  1. Filters disabled informers
  2. For each enabled informer: ``logits_i = informer.fn(ctx, avail)``
  3. Weighted sum: ``L = Σ w_i · logits_i``
  4. ``softmax(L / temperature)`` returns the prior

By default, all informers are weighted equally. Per-informer
``enabled``/``weight`` flags let the ablation harness toggle and tune
without re-instantiating. The mixer caches per-frame so calling it
repeatedly during one planning session amortizes cost.

See [[feedback_no_isolated_systems]] for the constitutional principle:
each new advisory algorithm is one new Informer file, no MCTS edits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

import numpy as np


@dataclass
class InformerContext:
    """Per-plan() context handed to every informer.

    Bundles everything an informer might need: the current observation,
    the encoded root latent (filled after the model encoder runs), the
    game identifier (for cache lookups), and a free-form dict for
    informer-specific extras (e.g. SAM masks, replay-bank snapshots).
    """

    current_frame: np.ndarray                    # (64, 64) uint8 ARC frame
    z_root: Optional[torch.Tensor] = None         # [D]; filled by MCTS at root
    game_id: Optional[str] = None
    extras: dict = field(default_factory=dict)


@dataclass
class Informer:
    """Single advisory algorithm.

    ``fn`` takes ``(InformerContext, available_action_ids)`` and returns
    a 1-D tensor of length ``|available_actions|``. Whatever scale the
    informer emits, the mixer's softmax handles normalization — but
    informers should aim for "logit-scale" outputs (mean ≈ 0,
    magnitude ≈ 1-5) so the temperature knob is meaningful.
    """

    name: str
    fn: Callable[[InformerContext, list[int]], torch.Tensor]
    weight: float = 1.0
    enabled: bool = True

    def evaluate(self, ctx: InformerContext, avail: list[int]) -> torch.Tensor:
        logits = self.fn(ctx, avail)
        if not isinstance(logits, torch.Tensor):
            logits = torch.tensor(logits, dtype=torch.float32)
        if logits.shape[-1] != len(avail):
            raise ValueError(
                f"Informer {self.name!r} returned logits of length "
                f"{logits.shape[-1]}; expected {len(avail)} (= |available_actions|)"
            )
        return logits.float()


class PriorMixer:
    """Composable advisory aggregator. Matches MCTSConfig.cluster_prior_fn signature.

    Usage::

        mixer = PriorMixer(
            context=InformerContext(current_frame=frame, game_id="lp85-..."),
            informers=[
                SpriteDensityInformer(),
                ReplayFrequencyInformer(bank=replay_bank),
                AffordanceHeadInformer(model=clean_v1),
            ],
            temperature=1.0,
        )
        cfg = MCTSConfig(cluster_prior_fn=mixer)
        result = planner.plan(model=lewm, embedder=emb, current_frame=obs,
                              goal_latent=goal, available_actions=avail, context=PlanContext(game_id="lp85-..."))

    The mixer's ``__call__(game_id, z_root)`` matches the hook signature
    expected by MCTS. ``game_id`` from the hook overrides the context's
    game_id if set; ``z_root`` is stored in the context so learned
    informers can use it.

    Telemetry: ``last_per_informer_logits`` keeps the most recent
    contribution from each informer for the ablation harness to inspect.
    """

    def __init__(
        self,
        context: InformerContext,
        informers: list[Informer],
        temperature: float = 1.0,
    ) -> None:
        if not informers:
            raise ValueError("PriorMixer requires at least one informer")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.context = context
        self.informers = informers
        self.temperature = temperature
        self.last_per_informer_logits: dict[str, torch.Tensor] = {}

    def __call__(
        self,
        game_id: Optional[str],
        z_root: torch.Tensor,
    ) -> torch.Tensor:
        """MCTS hook entry. Returns prior over available_actions.

        Note: ``available_actions`` is not passed through this hook
        (MCTS doesn't currently surface it here). The mixer reads it from
        ``self.context.extras['available_actions']`` which the caller must
        set before invoking the planner. The ablation harness sets it; in
        production, build the mixer per-call with available_actions baked in.
        """

        if game_id is not None:
            self.context.game_id = game_id
        self.context.z_root = z_root.detach()

        avail = self.context.extras.get("available_actions")
        if avail is None:
            raise ValueError(
                "PriorMixer requires context.extras['available_actions'] to be set"
            )

        self.last_per_informer_logits.clear()
        accumulated = torch.zeros(len(avail), dtype=torch.float32)
        active_weight_sum = 0.0
        for inf in self.informers:
            if not inf.enabled:
                continue
            logits = inf.evaluate(self.context, list(avail))
            self.last_per_informer_logits[inf.name] = logits.detach()
            accumulated = accumulated + inf.weight * logits
            active_weight_sum += inf.weight

        if active_weight_sum == 0.0:
            # All disabled → uniform prior.
            return torch.full((len(avail),), 1.0 / len(avail))

        return torch.softmax(accumulated / self.temperature, dim=-1)


def uniform_logits(avail: list[int]) -> torch.Tensor:
    """Helper for informer authors: a no-op logit vector (all zeros)."""

    return torch.zeros(len(avail), dtype=torch.float32)


def kl_to_uniform(prior: torch.Tensor) -> float:
    """KL(prior || uniform). Ablation harness uses this to confirm each
    informer measurably shifts the prior."""

    n = prior.shape[-1]
    uniform = torch.full_like(prior, 1.0 / n)
    eps = 1e-12
    p = prior.clamp_min(eps)
    return float((p * (p.log() - uniform.clamp_min(eps).log())).sum().item())


__all__ = [
    "Informer", "InformerContext", "PriorMixer",
    "uniform_logits", "kl_to_uniform",
]
