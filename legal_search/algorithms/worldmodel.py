"""World-model candidate rank — mined from LeWM affordance + CEM interfaces.

KEEP: S+A → change affordance rank, multi-step consistency score, runtime lock.
STRIP: unlocked pretrained policy as live action owner.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.dynamics import (
    DynamicsHypothesis,
    GridState,
    TranslateColorBlob,
    induce_blob_color,
    mine_dynamics,
)
from legal_search.types import Action, Proposal, Transition


def tabular_affordance_rates(
    transitions: list[Transition],
    available: list[int],
) -> dict[int, float]:
    rates = {a: [0.0, 0.0] for a in available}
    for tr in transitions:
        aid = int(tr.action.action_id)
        if aid not in rates:
            continue
        rates[aid][1] += 1.0
        rates[aid][0] += 1.0 if tr.frame_changed else 0.0
    return {a: (c + 0.5) / (n + 1.0) for a, (c, n) in rates.items()}


def multi_step_consistency(
    frame: np.ndarray,
    transitions: list[Transition],
    hyp: DynamicsHypothesis,
    *,
    horizon: int = 3,
) -> float:
    """Fraction of windows where hyp roll matches observed chain (overlap mean)."""
    if len(transitions) < horizon:
        return 0.0
    scores = []
    for i in range(len(transitions) - horizon + 1):
        window = transitions[i : i + horizon]
        st = GridState(grid=np.asarray(window[0].frame_before).copy())
        acc = []
        for tr in window:
            st = hyp.apply(st, tr.action)
            after = np.asarray(tr.frame_after)
            if st.grid.shape != after.shape:
                acc.append(0.0)
            else:
                acc.append(float(np.mean(st.grid == after)))
        scores.append(float(np.mean(acc)))
    return float(np.mean(scores)) if scores else 0.0


def mine_worldmodel(
    frame: np.ndarray,
    *,
    transitions: list[Transition] | None = None,
    available_actions: list[int] | None = None,
) -> dict[str, Any]:
    trs = list(transitions or [])
    avail = available_actions or [1, 2, 3, 4, 5, 6]
    rates = tabular_affordance_rates(trs, avail)
    dyn = mine_dynamics(frame, trs, available_actions=avail)
    color = induce_blob_color(trs)
    hyp = TranslateColorBlob(color=color)
    consistency = multi_step_consistency(frame, trs, hyp, horizon=3)

    proposals: list[Proposal] = []
    for aid, score in sorted(rates.items(), key=lambda kv: -kv[1]):
        proposals.append(
            Proposal(
                actions=[Action(aid, source="worldmodel.affordance")],
                score=float(score),
                kind="affordance_change_rank",
                evidence={"trials_proxy": True, "consistency": consistency},
                algorithm_id="worldmodel.candidate_predict_rank.v1",
            )
        )
    # merge top imagine proposals
    for p in dyn.get("proposals") or []:
        proposals.append(p)
    proposals.sort(key=lambda p: -p.score)

    # runtime admission: only if we have transition evidence
    runtime_admitted = bool(trs) and consistency > 0.15

    return {
        "interface": "S_t + A_k -> O_pred_rank",
        "affordance_objective": "did_frame_change",
        "affordance_rates": {str(k): float(v) for k, v in rates.items()},
        "multi_step_consistency": consistency,
        "dynamics": {
            "best_hypothesis": dyn.get("best_hypothesis"),
            "hypotheses": dyn.get("hypotheses"),
            "fixture_priors_loaded": False,
        },
        "proposals": proposals[:12],
        "runtime_admitted": runtime_admitted,
        "runtime_requires_evidence": True,
        "encoder_interface": "optional_online_or_generic_only",
        "cem_interface": "rank_candidates_by_affordance_or_imagine",
        "n_transitions": len(trs),
    }
