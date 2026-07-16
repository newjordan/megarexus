"""Match calibration — reweight role candidates only.

Mined idea from codexlab/contra_head/match_calibration (stage probes / priors)
as *legal reweighting*, never admit actor/goal truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.affordances import mine_affordance_candidates
from legal_search.algorithms.role_induce import induce_click_sensitive_cells, induce_movable_colors
from legal_search.types import Transition


def reweight_candidates(
    candidates: list[dict[str, Any]],
    *,
    movable_rates: dict[int, float],
    click_cells: list[dict[str, Any]],
    boost: float = 0.25,
) -> list[dict[str, Any]]:
    out = []
    for c in candidates:
        score = float(c.get("score") or 0.0)
        color = int(c.get("color") or -1)
        if color in movable_rates:
            score = min(1.0, score + boost * float(movable_rates[color]))
        row = dict(c)
        row["score"] = score
        row["reweighted"] = True
        row["admitted"] = False
        row["may_admit_truth"] = False
        out.append(row)
    out.sort(key=lambda r: -float(r["score"]))
    return out


def stage0_click_ranking(
    frame: np.ndarray,
    *,
    transitions: list[Transition] | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    from legal_search.algorithms.components import enrich_components

    scene = enrich_components(frame)
    trs = list(transitions or [])
    movable = induce_movable_colors(trs)
    clicks = induce_click_sensitive_cells(trs)
    click_map = {(int(c["x"]), int(c["y"])): float(c["change_rate"]) for c in clicks}
    rows = []
    for feat in scene["component_features"][:16]:
        cx = float(feat["center_x_ratio"]) * 63.0
        cy = float(feat["center_y_ratio"]) * 63.0
        x, y = int(round(cx)), int(round(cy))
        color = int(feat.get("color") or 0)
        sal = max(0.0, min(1.0, 1.0 - float(feat.get("area_ratio") or 0.0)))
        move_boost = float(movable.get(color) or 0.0)
        click_boost = float(click_map.get((x, y), 0.0))
        aspect = float(feat.get("aspect") or 1.0)
        asym = min(1.0, abs(float(np.log(max(1e-3, aspect)))))
        score = max(
            0.0,
            min(
                1.0,
                0.35 * sal
                + 0.25 * (1.0 - float(feat.get("border_contact") or 0.0))
                + 0.20 * move_boost
                + 0.10 * click_boost
                + 0.10 * asym,
            ),
        )
        rows.append(
            {
                "component_id": feat["component_id"],
                "color": color,
                "click_xy": [x, y],
                "score": score,
                "salience": sal,
                "asymmetry": asym,
                "movable_boost": move_boost,
                "click_boost": click_boost,
                "admitted": False,
                "may_admit_truth": False,
                "stage": "stage0_click_rank",
            }
        )
    rows.sort(key=lambda r: -float(r["score"]))
    return {
        "ranked_clicks": rows[:top_k],
        "n_ranked": min(top_k, len(rows)),
        "best_score": float(rows[0]["score"]) if rows else 0.0,
        "may_reweight": True,
        "may_admit_truth": False,
        "stage": "stage0",
    }


def stage1_init_delta_observation(transitions: list[Transition] | None) -> dict[str, Any]:
    trs = list(transitions or [])[:8]
    if not trs:
        return {
            "stage": "stage1",
            "n_early": 0,
            "change_rate": 0.0,
            "requires_recalibration": False,
            "may_admit_truth": False,
        }
    ch = sum(1 for tr in trs if tr.frame_changed)
    rate = ch / max(1, len(trs))
    return {
        "stage": "stage1",
        "n_early": len(trs),
        "n_changed": ch,
        "change_rate": rate,
        "requires_recalibration": rate < 0.15,
        "may_admit_truth": False,
        "admitted_actor": False,
        "admitted_goal": False,
    }


def mine_calibration(
    frame: np.ndarray,
    transitions: list[Transition] | None = None,
) -> dict[str, Any]:
    trs = list(transitions or [])
    base = mine_affordance_candidates(frame)
    movable = induce_movable_colors(trs)
    clicks = induce_click_sensitive_cells(trs)
    actors = reweight_candidates(base["actors"], movable_rates=movable, click_cells=clicks)
    targets = reweight_candidates(
        base["targets"], movable_rates=movable, click_cells=clicks, boost=0.15
    )
    supports = reweight_candidates(
        base["supports"], movable_rates={}, click_cells=clicks, boost=0.05
    )

    def top(xs):
        return float(xs[0]["score"]) if xs else 0.0

    prior_shift = abs(top(actors) - top(base["actors"])) + abs(top(targets) - top(base["targets"]))
    s0 = stage0_click_ranking(frame, transitions=trs)
    s1 = stage1_init_delta_observation(trs)
    return {
        "may_reweight": True,
        "may_admit_truth": False,
        "actors": actors,
        "targets": targets,
        "supports": supports,
        "movable_rates_used": movable,
        "n_click_cells": len(clicks),
        "prior_shift": float(prior_shift),
        "n_transitions": len(trs),
        "transfer_prior_as_solve_truth": False,
        "admitted_actor": False,
        "admitted_goal": False,
        "stage0": s0,
        "stage1": s1,
    }
