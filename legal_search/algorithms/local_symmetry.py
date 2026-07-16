"""Local / component symmetry — mined from codexlab/contra_head/symmetry.py.

KEEP: global axes, component self-reflection scores, mirror-pair scores.
STRIP: symmetry lock as goal/solve key, admitted truth flags.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.saliency import symmetry_score_horizontal, symmetry_score_vertical


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def bbox_mask_symmetry(frame: np.ndarray, bbox_xyxy: list[int], *, axis: str) -> float:
    """Reflect component bbox crop on H or V axis; score pixel agreement."""
    grid = np.asarray(frame)
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    h, w = grid.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    if x1 < x0 or y1 < y0:
        return 0.0
    crop = grid[y0 : y1 + 1, x0 : x1 + 1]
    if crop.size == 0:
        return 0.0
    if axis in ("vertical", "lr", "horizontal_flip"):
        # left-right mirror of crop
        flipped = np.flip(crop, axis=1)
    else:
        flipped = np.flip(crop, axis=0)
    return float(np.mean(crop == flipped))


def component_self_symmetry(frame: np.ndarray, component: dict[str, Any]) -> dict[str, Any]:
    box = component["bbox_xyxy"]
    h_score = bbox_mask_symmetry(frame, box, axis="vertical")  # LR
    v_score = bbox_mask_symmetry(frame, box, axis="horizontal")  # UD
    return {
        "component_id": component["component_id"],
        "horizontal_mirror": h_score,
        "vertical_mirror": v_score,
        "best": max(h_score, v_score),
        "admitted": False,
        "evidence_status": "static_geometry_hypothesis",
    }


def axis_pair_score(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    axis: str,
    frame_shape: tuple[int, int],
) -> float:
    """Mined from symmetry._axis_pair_score — geometric mirror pair prior."""
    height, width = frame_shape
    lx, ly = float(left["center_xy"][0]), float(left["center_xy"][1])
    rx, ry = float(right["center_xy"][0]), float(right["center_xy"][1])
    if axis == "vertical":  # left-right mirror across frame
        expected_rx = float(width - 1) - lx
        expected_ry = ly
        norm = max(1.0, float(width))
    else:
        expected_rx = lx
        expected_ry = float(height - 1) - ly
        norm = max(1.0, float(height))
    dist = float(np.hypot(expected_rx - rx, expected_ry - ry))
    distance_penalty = _clip01(dist / norm)
    lb = left["bbox_xyxy"]
    rb = right["bbox_xyxy"]
    left_w = max(1, int(lb[2]) - int(lb[0]) + 1)
    left_h = max(1, int(lb[3]) - int(lb[1]) + 1)
    right_w = max(1, int(rb[2]) - int(rb[0]) + 1)
    right_h = max(1, int(rb[3]) - int(rb[1]) + 1)
    width_similarity = min(left_w, right_w) / max(left_w, right_w)
    height_similarity = min(left_h, right_h) / max(left_h, right_h)
    area_left = max(1, int(left.get("area") or left.get("area_pixels") or 1))
    area_right = max(1, int(right.get("area") or right.get("area_pixels") or 1))
    area_similarity = min(area_left, area_right) / max(area_left, area_right)
    color_match = 1.0 if int(left.get("color") or 0) == int(right.get("color") or 0) else 0.0
    return _clip01(
        0.42 * (1.0 - distance_penalty)
        + 0.18 * width_similarity
        + 0.18 * height_similarity
        + 0.14 * area_similarity
        + 0.08 * color_match
    )


def mine_symmetry(
    frame: np.ndarray,
    components: list[dict[str, Any]],
    *,
    max_pairs: int = 12,
) -> dict[str, Any]:
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    global_h = symmetry_score_horizontal(grid)
    global_v = symmetry_score_vertical(grid)
    self_rows = [component_self_symmetry(grid, c) for c in components[:12]]
    self_rows.sort(key=lambda r: -r["best"])
    pairs: list[dict[str, Any]] = []
    comps = components[:10]
    for i, a in enumerate(comps):
        for b in comps[i + 1 :]:
            for axis in ("vertical", "horizontal"):
                score = axis_pair_score(a, b, axis=axis, frame_shape=(h, w))
                if score < 0.55:
                    continue
                pairs.append(
                    {
                        "a": a["component_id"],
                        "b": b["component_id"],
                        "axis": axis,
                        "score": score,
                        "admitted": False,
                        "evidence_status": "static_geometry_hypothesis",
                    }
                )
    pairs.sort(key=lambda r: -r["score"])
    pairs = pairs[:max_pairs]
    best_self = float(self_rows[0]["best"]) if self_rows else 0.0
    best_pair = float(pairs[0]["score"]) if pairs else 0.0
    return {
        "global_horizontal": global_h,
        "global_vertical": global_v,
        "component_self": self_rows[:8],
        "mirror_pairs": pairs,
        "n_self": len(self_rows),
        "n_pairs": len(pairs),
        "best_self": best_self,
        "best_pair": best_pair,
        "not_goal_key": True,
    }
