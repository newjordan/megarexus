"""Confirmed relation memory — mined from codexlab/contra_head/ghost.py.

KEEP: this-game confirmed relation rows, lineup summaries, action memory.
STRIP: action route scripts, cross-game transfer without reinduction, solve truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.ports import mine_ports_full
from legal_search.algorithms.components import connected_components, frame_delta_count
from legal_search.types import Transition


def truth_boundary() -> dict[str, bool]:
    return {
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "cross_game_memory_admitted": False,
    }


def build_confirmed_relation_rows(
    transitions: list[Transition],
    *,
    game_id: str = "",
) -> list[dict[str, Any]]:
    """Legal stand-in for ghost.build_confirmed_relation_rows from transitions only."""
    rows: list[dict[str, Any]] = []
    for tr in transitions:
        if not (tr.frame_changed or tr.levels_up):
            continue
        aid = int(tr.action.action_id)
        family = "level_up" if tr.levels_up else ("click_effect" if aid == 6 else "action_effect")
        dpx = frame_delta_count(tr.frame_before, tr.frame_after)
        # geometry: change centroid
        before = np.asarray(tr.frame_before)
        after = np.asarray(tr.frame_after)
        geom: dict[str, Any] = {"delta_pixels": dpx}
        if before.shape == after.shape:
            mask = before != after
            if np.any(mask):
                ys, xs = np.nonzero(mask)
                geom["change_center_xy"] = [float(xs.mean()), float(ys.mean())]
                geom["change_bbox_xyxy"] = [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()),
                    int(ys.max()),
                ]
        rows.append(
            {
                "relation_memory_id": f"{game_id}:{tr.step}:{family}",
                "game_id": game_id,
                "round_index": int(tr.step),
                "relation_family": family,
                "action": tr.action.to_dict(),
                "geometry": geom,
                "confirmation": {
                    "confirmed_by": [
                        name
                        for name, ok in (
                            ("frame_changed", tr.frame_changed),
                            ("levels_up", tr.levels_up),
                        )
                        if ok
                    ],
                    "levels_completed_delta": int(tr.levels_after - tr.levels_before),
                    "confidence_score": 1.0 if tr.levels_up else 0.55,
                    "confidence_is_probability": False,
                },
                "truth_boundary": truth_boundary(),
                "admitted": False,
                "not_action_authority": True,
            }
        )
    return rows


def lineup_summary_from_frame(frame: np.ndarray) -> dict[str, Any]:
    comps = connected_components(frame)[:10]
    mined = mine_ports_full(comps, frame_shape=np.asarray(frame).shape[:2])
    lineup = mined.get("lineup_matrix") or {}
    summaries = []
    for row in (lineup.get("rows") or [])[:12]:
        summaries.append(
            {
                "source_object_id": row.get("source_object_id"),
                "target_object_ids": row.get("target_object_ids"),
                "target_port_ids": row.get("target_port_ids"),
                "transform_hypothesis_xy": row.get("transform_xy"),
                "confidence_score": row.get("confidence_score"),
                "admitted": False,
            }
        )
    return {
        "summary_source": "live_frame_port_lineup",
        "lineup_matrix_row_count": len(summaries),
        "rows": summaries,
        "best_lineup_score": lineup.get("best_lineup_score", 0.0),
    }


def action_memory_summary(transitions: list[Transition]) -> dict[str, Any]:
    by_aid: dict[str, list[float]] = {}
    for tr in transitions:
        key = str(int(tr.action.action_id))
        bucket = by_aid.setdefault(key, [0.0, 0.0])
        bucket[1] += 1.0
        bucket[0] += 1.0 if tr.frame_changed else 0.0
    return {
        "n_transitions": len(transitions),
        "per_action": {
            k: {"changes": v[0], "trials": v[1], "rate": (v[0] / v[1] if v[1] else 0.0)}
            for k, v in by_aid.items()
        },
        "n_level_up": sum(1 for tr in transitions if tr.levels_up),
    }


def mine_ghost(
    frame: np.ndarray,
    transitions: list[Transition] | None = None,
    *,
    game_id: str = "",
) -> dict[str, Any]:
    trs = list(transitions or [])
    confirmed = build_confirmed_relation_rows(trs, game_id=game_id)
    lineup = lineup_summary_from_frame(frame)
    actions = action_memory_summary(trs)
    aid_rates = {
        k: float(v["rate"]) for k, v in (actions.get("per_action") or {}).items()
    }
    n_changed = sum(1 for tr in trs if tr.frame_changed)
    return {
        "confirmed_relations": confirmed[-16:],
        "n_confirmed": len(confirmed),
        "lineup_summary": lineup,
        "action_memory": actions,
        "aid_change_rates": aid_rates,
        "n_transitions": len(trs),
        "n_changed": n_changed,
        "n_level_up": actions.get("n_level_up", 0),
        "change_fraction": n_changed / max(1, len(trs)),
        "cross_game_memory_admitted": False,
        "not_action_authority": True,
        "truth_boundary": truth_boundary(),
    }
