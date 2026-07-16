"""Discrete CEM action planner — legal mine of LeWM cem_discrete interface.

KEEP: sample sequences, elite refine, plug-in scorer.
STRIP: requires frozen LeWM weights; no game-specific pretrained policy as owner.
Uses dynamics/affordance scorers so it runs without torch model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from legal_search.algorithms.dynamics import (
    GridState,
    TranslateColorBlob,
    induce_blob_color,
    mine_dynamics,
)
from legal_search.algorithms.components import frame_delta_count
from legal_search.types import Action, Proposal, Transition


@dataclass
class DiscreteCEMConfig:
    horizon: int = 4
    num_samples: int = 48
    num_iterations: int = 6
    elite_count: int = 8
    smoothing: float = 0.15

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon")
        if self.elite_count < 1 or self.elite_count > self.num_samples:
            raise ValueError("elite_count")


def _score_sequence(
    frame: np.ndarray,
    seq: list[int],
    hyp: TranslateColorBlob,
) -> float:
    """Higher is better: total predicted pixel change under dynamics hyp."""
    st = GridState(grid=np.asarray(frame).copy())
    total = 0.0
    for aid in seq:
        before = st.grid.copy()
        st = hyp.apply(st, Action(int(aid), source="cem"))
        total += float(frame_delta_count(before, st.grid))
    # prefer shorter effective change density
    return total / float(max(1, len(seq)))


def discrete_cem_plan(
    frame: np.ndarray,
    *,
    available_actions: list[int] | None = None,
    transitions: list[Transition] | None = None,
    config: DiscreteCEMConfig | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """CEM over discrete action_ids scored by legal dynamics hypothesis."""
    cfg = config or DiscreteCEMConfig()
    cfg.validate()
    avail = list(available_actions or [1, 2, 3, 4, 5, 6])
    if not avail:
        avail = [1]
    rng = np.random.default_rng(seed)
    color = induce_blob_color(list(transitions or []))
    hyp = TranslateColorBlob(color=color)

    # categorical probs per horizon step
    n_a = len(avail)
    probs = np.full((cfg.horizon, n_a), 1.0 / n_a, dtype=np.float64)
    best_seq: list[int] = [avail[0]] * cfg.horizon
    best_score = -1e9
    history = []

    for it in range(cfg.num_iterations):
        samples = []
        scores = []
        for _ in range(cfg.num_samples):
            seq = []
            for t in range(cfg.horizon):
                idx = int(rng.choice(n_a, p=probs[t]))
                seq.append(avail[idx])
            sc = _score_sequence(frame, seq, hyp)
            # blend with transition affordance prior
            if transitions:
                rates = {}
                for tr in transitions:
                    aid = int(tr.action.action_id)
                    rates.setdefault(aid, [0.0, 0.0])
                    rates[aid][1] += 1
                    rates[aid][0] += 1.0 if tr.frame_changed else 0.0
                prior = float(
                    np.mean(
                        [
                            (rates.get(a, [0.5, 1])[0] + 0.5) / (rates.get(a, [0.5, 1])[1] + 1)
                            for a in seq
                        ]
                    )
                )
                sc = 0.7 * sc + 0.3 * prior * 64.0
            samples.append(seq)
            scores.append(sc)
        order = np.argsort(scores)[::-1]
        elites = [samples[i] for i in order[: cfg.elite_count]]
        elite_scores = [scores[i] for i in order[: cfg.elite_count]]
        if elite_scores[0] > best_score:
            best_score = float(elite_scores[0])
            best_seq = list(elites[0])
        # update probs from elites
        new_probs = np.zeros_like(probs)
        for seq in elites:
            for t, aid in enumerate(seq):
                j = avail.index(aid)
                new_probs[t, j] += 1.0
        new_probs /= max(1.0, float(cfg.elite_count))
        probs = (1.0 - cfg.smoothing) * new_probs + cfg.smoothing * (1.0 / n_a)
        # renorm
        probs = probs / probs.sum(axis=1, keepdims=True)
        history.append({"iter": it, "best_elite": float(elite_scores[0]), "mean_elite": float(np.mean(elite_scores))})

    proposals = [
        Proposal(
            actions=[Action(int(a), source="cem.discrete") for a in best_seq],
            score=float(best_score),
            kind="cem_program",
            evidence={"horizon": cfg.horizon, "iterations": cfg.num_iterations},
            algorithm_id="worldmodel.cem_discrete.v1",
        )
    ]
    # also single-step tops from final probs
    for t in range(min(1, cfg.horizon)):
        j = int(np.argmax(probs[t]))
        proposals.append(
            Proposal(
                actions=[Action(int(avail[j]), source="cem.marginal")],
                score=float(probs[t, j]),
                kind="cem_marginal_action",
                evidence={"step": t},
                algorithm_id="worldmodel.cem_discrete.v1",
            )
        )
    return {
        "best_sequence": best_seq,
        "best_score": float(best_score),
        "horizon": cfg.horizon,
        "num_samples": cfg.num_samples,
        "num_iterations": cfg.num_iterations,
        "elite_count": cfg.elite_count,
        "final_probs": probs.tolist(),
        "history": history,
        "proposals": proposals,
        "requires_lewm_weights": False,
        "scorer": "dynamics_translate_color_blob_plus_affordance_prior",
        "runtime_admitted": bool(transitions),
    }
