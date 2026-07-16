"""Actor/target/support *candidates* — mined from controller_affordances.py.

KEEP: ranked unknowns from geometry/size/border.
STRIP: admitted actor/goal truth, calibration tables as control truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.components import component_features, contact_graph, connected_components


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def score_actor_candidate(feat: dict[str, Any]) -> float:
    """Heuristic only — never admitted as actor truth.

    Prefer compact mid-frame objects with moderate area (movable-ish prior).
    """
    area = _clip01(float(feat["area_ratio"]) / 0.15)
    not_border = 1.0 - float(feat["border_contact"])
    compact = float(feat["bbox_fill_ratio"])
    mid = 1.0 - abs(float(feat["center_y_ratio"]) - 0.5) * 2.0
    mid = _clip01(mid)
    return _clip01(0.35 * area + 0.25 * not_border + 0.20 * compact + 0.20 * mid)


def score_target_candidate(feat: dict[str, Any]) -> float:
    """Heuristic only — never admitted as goal truth."""
    area = _clip01(float(feat["area_ratio"]) / 0.20)
    fill = float(feat["bbox_fill_ratio"])
    # targets often larger or more bottom-biased — weak prior only
    bottomish = float(feat["center_y_ratio"])
    return _clip01(0.40 * area + 0.30 * fill + 0.30 * bottomish)


def score_support_candidate(feat: dict[str, Any]) -> float:
    """Wide / border-touching support-ish candidates — hypothesis only."""
    wide = float(feat["bbox_width_ratio"])
    bottom = float(feat["bottom_contact"])
    neighbors = _clip01(float(feat["neighbor_component_count"]) / 4.0)
    return _clip01(0.40 * wide + 0.35 * bottom + 0.25 * neighbors)


def mine_affordance_candidates(
    frame: np.ndarray,
    *,
    top_k: int = 4,
) -> dict[str, Any]:
    comps, label_map = connected_components(frame, with_label_map=True)  # type: ignore[misc]
    comps = list(comps)[:16]
    h, w = np.asarray(frame).shape[:2]
    graph = contact_graph(comps, label_map)
    feats = [component_features(c, frame_shape=(h, w), graph=graph) for c in comps]

    actors = []
    targets = []
    supports = []
    for f in feats:
        actors.append(
            {
                "component_id": f["component_id"],
                "color": f["color"],
                "score": score_actor_candidate(f),
                "role": "actor_candidate",
                "admitted": False,
            }
        )
        targets.append(
            {
                "component_id": f["component_id"],
                "color": f["color"],
                "score": score_target_candidate(f),
                "role": "target_candidate",
                "admitted": False,
            }
        )
        supports.append(
            {
                "component_id": f["component_id"],
                "color": f["color"],
                "score": score_support_candidate(f),
                "role": "support_candidate",
                "admitted": False,
            }
        )
    actors.sort(key=lambda r: -r["score"])
    targets.sort(key=lambda r: -r["score"])
    supports.sort(key=lambda r: -r["score"])
    return {
        "actors": actors[:top_k],
        "targets": targets[:top_k],
        "supports": supports[:top_k],
        "n_components_scored": len(feats),
        "any_admitted": False,
        "may_admit_truth": False,
        "evidence_status": "affordance_hypothesis_not_truth",
    }
