#!/usr/bin/env python3
"""Oracle assumption head.

Oracle is not CONTRA. It reads a CONTRA packet and emits one small assumption
for one missing element at a time. Assumptions are review objects only; they are
not control truth, goal truth, action choice, solve evidence, or predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ORACLE_ASSUMPTION_SCHEMA = "codexlab.oracle_assumption.v1"
ORACLE_ASSUMPTION_PACKET_SCHEMA = "codexlab.oracle_assumption_packet.v1"
ORACLE_SOURCE_REF_SCHEMA = "codexlab.oracle_source_ref.v1"
ORACLE_MISSING_ELEMENT_SCHEMA = "codexlab.oracle_missing_element.v1"
ORACLE_KEY_LOCK_ARROW_SCHEMA = "codexlab.oracle_key_points_to_lock_hypothesis.v1"

TRUTH_FLAGS = {
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}

COLORFUL_RAW_VALUES = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
ASYMMETRY_CONFIDENCE_THRESHOLD = 0.2
NON_CENTROID_DISTANCE_THRESHOLD = 0.18
# Cavity search used to slide every component through every other component.
# A noisy 64x64 frame has thousands of components, so that pass does not
# finish. Boards with at most this many candidates are unchanged.
_CAVITY_CANDIDATE_LIMIT = 64


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _scene_surface_component_ids(source_packet: dict[str, Any]) -> set[str]:
    scene = source_packet.get("scene_surfaces")
    if not isinstance(scene, dict):
        return set()
    out: set[str] = set()
    floor = scene.get("floor_hypothesis")
    if isinstance(floor, dict) and floor.get("component_id"):
        out.add(str(floor.get("component_id")))
    for key in ("wall_hypotheses", "background_hypotheses"):
        for row in _as_list(scene.get(key)):
            if isinstance(row, dict) and row.get("component_id"):
                out.add(str(row.get("component_id")))
    for row in _as_list(scene.get("scene_pattern_hypotheses")):
        if not isinstance(row, dict):
            continue
        for component_id in _as_list(row.get("component_ids")):
            out.add(str(component_id))
    return out


def _frame_center(source_packet: dict[str, Any]) -> tuple[float, float, float]:
    shape = source_packet.get("raw_color_component_packet", {}).get("frame_shape")
    if not isinstance(shape, list) or len(shape) < 2:
        shape = source_packet.get("scene_surfaces", {}).get("frame_shape")
    if not isinstance(shape, list) or len(shape) < 2:
        return 31.5, 31.5, 90.0
    height = max(1.0, float(shape[0]))
    width = max(1.0, float(shape[1]))
    diagonal = max(1.0, (width * width + height * height) ** 0.5)
    return (width - 1.0) / 2.0, (height - 1.0) / 2.0, diagonal


def _component_candidates_from_source_packet(source_packet: dict[str, Any]) -> list[dict[str, Any]]:
    graph = source_packet.get("scene_surfaces", {}).get("component_graph", {})
    components = graph.get("components") if isinstance(graph, dict) else []
    excluded_surface_ids = _scene_surface_component_ids(source_packet)
    cx, cy, diag = _frame_center(source_packet)
    candidates: list[dict[str, Any]] = []
    for component in components if isinstance(components, list) else []:
        if not isinstance(component, dict):
            continue
        component_id = str(component.get("component_id"))
        raw_value = int(component.get("raw_pixel_value") or -1)
        center = component.get("center_xy") if isinstance(component.get("center_xy"), list) else [cx, cy]
        dx = float(center[0]) - cx
        dy = float(center[1]) - cy
        centroid_distance_ratio = ((dx * dx + dy * dy) ** 0.5) / diag
        asymmetry = component.get("asymmetry") if isinstance(component.get("asymmetry"), dict) else {}
        asymmetry_confidence = float(asymmetry.get("confidence") or 0.0)
        facing = str(asymmetry.get("facing_direction_hypothesis") or "none")
        colorful = raw_value in COLORFUL_RAW_VALUES
        asymmetric = asymmetry_confidence >= ASYMMETRY_CONFIDENCE_THRESHOLD or facing != "none"
        non_centroid = centroid_distance_ratio >= NON_CENTROID_DISTANCE_THRESHOLD
        scene_surface = component_id in excluded_surface_ids
        area = int(component.get("area_pixels") or 0)
        candidates.append(
            {
                "component_id": component_id,
                "raw_pixel_value": raw_value,
                "area_pixels": area,
                "bbox_xyxy": component.get("bbox_xyxy"),
                "center_xy": component.get("center_xy"),
                "shape_mask_rows": _as_list(component.get("shape_mask_rows")),
                "shape_mask_encoding": component.get("shape_mask_encoding"),
                "asymmetry_confidence": asymmetry_confidence,
                "facing_direction_hypothesis": facing,
                "centroid_distance_ratio": round(float(centroid_distance_ratio), 6),
                "excluded_as_scene_surface_role": scene_surface,
                "colorful_raw_value": colorful,
                "asymmetric_component": asymmetric,
                "non_centroid_component": non_centroid,
                "not_scene_surface_role": not scene_surface,
            }
        )
    return candidates


def _mask_from_rows(rows: Any) -> list[list[int]]:
    if not isinstance(rows, list) or not rows:
        return []
    mask: list[list[int]] = []
    width = None
    for row in rows:
        if not isinstance(row, str) or not row:
            return []
        values = [1 if char == "1" else 0 for char in row]
        if width is None:
            width = len(values)
        elif len(values) != width:
            return []
        mask.append(values)
    return mask


def _mask_area(mask: list[list[int]]) -> int:
    return sum(sum(row) for row in mask)


def _mask_array(rows: Any) -> np.ndarray:
    parsed = _mask_from_rows(rows)
    if not parsed:
        return np.zeros((0, 0), dtype=np.int32)
    return np.asarray(parsed, dtype=np.int32)


def _best_open_cavity_fit(lock_component: dict[str, Any], key_component: dict[str, Any]) -> dict[str, Any]:
    lock_mask = _mask_array(lock_component.get("shape_mask_rows"))
    key_mask = _mask_array(key_component.get("shape_mask_rows"))
    if lock_mask.size == 0 or key_mask.size == 0:
        return {
            "status": "missing_shape_masks",
            "fit_score": 0.0,
            "lock_key_shape_fit": False,
        }
    lock_h, lock_w = lock_mask.shape
    key_h, key_w = key_mask.shape
    key_area = int(key_mask.sum())
    if key_area <= 0 or key_h > lock_h or key_w > lock_w:
        return {
            "status": "shape_size_mismatch",
            "fit_score": 0.0,
            "lock_key_shape_fit": False,
            "lock_shape": [int(lock_w), int(lock_h)],
            "key_shape": [int(key_w), int(key_h)],
        }
    complement_area = int(lock_w * lock_h - int(lock_mask.sum()))
    windows = np.lib.stride_tricks.sliding_window_view(lock_mask, (key_h, key_w))
    collisions = np.tensordot(windows, key_mask, axes=([2, 3], [0, 1]))
    overlap_open = key_area - collisions
    key_into_open = overlap_open / float(max(1, key_area))
    collision_ratio = collisions / float(max(1, key_area))
    open_space_coverage = overlap_open / float(max(1, complement_area))
    fit_score = np.maximum(0.0, key_into_open - 0.85 * collision_ratio) * (
        0.55 + 0.45 * np.minimum(1.0, open_space_coverage)
    )
    flat = int(np.argmax(fit_score))
    oy, ox = divmod(flat, int(fit_score.shape[1]))
    best = {
        "fit_score": round(float(fit_score[oy, ox]), 6),
        "offset_xy": [int(ox), int(oy)],
        "key_into_open_space_ratio": round(float(key_into_open[oy, ox]), 6),
        "collision_ratio": round(float(collision_ratio[oy, ox]), 6),
        "open_space_coverage_ratio": round(float(open_space_coverage[oy, ox]), 6),
    }
    same_raw_value = int(lock_component.get("raw_pixel_value")) == int(key_component.get("raw_pixel_value"))
    lock_key_shape_fit = (
        float(best["fit_score"]) >= 0.78
        and float(best["key_into_open_space_ratio"]) >= 0.85
        and float(best["collision_ratio"]) <= 0.10
    )
    return {
        "status": "hypothesis",
        "lock_component_id": lock_component.get("component_id"),
        "key_component_id": key_component.get("component_id"),
        "same_raw_value": same_raw_value,
        "color_match_required": False,
        "lock_raw_pixel_value": int(lock_component.get("raw_pixel_value")),
        "key_raw_pixel_value": int(key_component.get("raw_pixel_value")),
        "lock_shape": [int(lock_w), int(lock_h)],
        "key_shape": [int(key_w), int(key_h)],
        "lock_open_space_pixels": int(complement_area),
        "key_area_pixels": int(key_area),
        "lock_key_shape_fit": lock_key_shape_fit,
        "relation_rule": "actor key shape fits open cavity of goal lock; color match is not required",
        **best,
    }


def _best_key_fit_for_lock(lock_component: dict[str, Any], components: list[dict[str, Any]]) -> dict[str, Any]:
    fits = []
    for key_component in components:
        if str(key_component.get("component_id")) == str(lock_component.get("component_id")):
            continue
        if not bool(key_component.get("not_scene_surface_role")):
            continue
        fit = _best_open_cavity_fit(lock_component, key_component)
        fit["goal_lock_component_id"] = lock_component.get("component_id")
        fit["actor_key_component_id"] = key_component.get("component_id")
        fits.append(fit)
    fits.sort(
        key=lambda row: (
            -int(bool(row.get("lock_key_shape_fit"))),
            -float(row.get("fit_score") or 0.0),
            -float(row.get("open_space_coverage_ratio") or 0.0),
            float(row.get("collision_ratio") if row.get("collision_ratio") is not None else 1.0),
            str(row.get("key_component_id")),
        )
    )
    if not fits:
        return {
            "status": "no_key_candidates",
            "fit_score": 0.0,
            "lock_key_shape_fit": False,
            "goal_lock_component_id": lock_component.get("component_id"),
            "actor_key_component_id": None,
        }
    return fits[0]


def _actor_anchor_from_components(components: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = []
    for component in components:
        criteria = {
            "asymmetric_component": bool(component["asymmetric_component"]),
            "not_scene_surface_role": bool(component["not_scene_surface_role"]),
        }
        if not all(criteria.values()):
            continue
        distance = float(component["centroid_distance_ratio"])
        score = (
            1.0
            + (0.35 if component["colorful_raw_value"] else 0.0)
            + max(0.0, 1.0 - distance)
            + min(0.25, float(component["area_pixels"]) / 256.0)
        )
        row = dict(component)
        row["anchor_criteria"] = criteria
        row["anchor_score"] = round(float(score), 6)
        scored.append(row)
    scored.sort(
        key=lambda row: (
            float(row["centroid_distance_ratio"]),
            -float(row["anchor_score"]),
            -int(row["area_pixels"]),
            str(row["component_id"]),
        )
    )
    return scored[0] if scored else None


def _actor_goal_geometry_metrics(actor: dict[str, Any] | None, component: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(actor, dict):
        return {
            "status": "no_actor_anchor",
            "actor_component_id": None,
            "actor_facing_direction": "none",
            "actor_goal_distance_ratio": 0.0,
            "actor_facing_alignment": 0.0,
            "geometry_relation_strong": False,
        }
    actor_id = str(actor.get("component_id"))
    component_id = str(component.get("component_id"))
    actor_center = actor.get("center_xy") if isinstance(actor.get("center_xy"), list) else [0.0, 0.0]
    component_center = component.get("center_xy") if isinstance(component.get("center_xy"), list) else [0.0, 0.0]
    dx = float(component_center[0]) - float(actor_center[0])
    dy = float(component_center[1]) - float(actor_center[1])
    dist = max(1e-6, math.hypot(dx, dy))
    facing = str(actor.get("facing_direction_hypothesis") or "none")
    facing_vectors = {
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
    }
    fx, fy = facing_vectors.get(facing, (0.0, 0.0))
    alignment = (dx / dist) * fx + (dy / dist) * fy if facing in facing_vectors else 0.0
    # The component distance ratios are normalized to frame diagonal from center;
    # for pair distance we only need a comparable lower bound.
    actor_goal_distance_ratio = float(dist) / 90.0
    relation_strong = (
        actor_id != component_id
        and bool(component.get("not_scene_surface_role"))
        and bool(component.get("colorful_raw_value"))
        and alignment >= 0.70
        and actor_goal_distance_ratio >= 0.08
    )
    return {
        "status": "hypothesis",
        "actor_component_id": actor_id,
        "actor_facing_direction": facing,
        "goal_component_id": component_id,
        "vector_xy": [round(dx, 6), round(dy, 6)],
        "actor_goal_distance_ratio": round(actor_goal_distance_ratio, 6),
        "actor_facing_alignment": round(max(-1.0, min(1.0, alignment)), 6),
        "geometry_relation_strong": relation_strong,
        "relation_rule": "candidate lies in actor facing direction with sufficient separation",
    }


def _direction_from_vector(dx: float, dy: float) -> str:
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    if abs(dy) > 0.0:
        return "down" if dy > 0 else "up"
    return "none"


def _key_points_to_lock_hypothesis(
    *,
    goal_candidate: dict[str, Any] | None,
    actor_candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(goal_candidate, dict) or not isinstance(actor_candidate, dict):
        return None
    goal = goal_candidate.get("selected_component")
    actor = actor_candidate.get("selected_component")
    if not isinstance(goal, dict) or not isinstance(actor, dict):
        return None
    goal_center = goal.get("center_xy") if isinstance(goal.get("center_xy"), list) else None
    actor_center = actor.get("center_xy") if isinstance(actor.get("center_xy"), list) else None
    if not isinstance(goal_center, list) or not isinstance(actor_center, list):
        return None
    dx = float(goal_center[0]) - float(actor_center[0])
    dy = float(goal_center[1]) - float(actor_center[1])
    distance = math.hypot(dx, dy)
    direction = _direction_from_vector(dx, dy)
    lock_key_fit = goal.get("lock_key_fit") if isinstance(goal.get("lock_key_fit"), dict) else {}
    orientation_source = (
        "lock_key_fit_geometry"
        if lock_key_fit.get("lock_key_shape_fit") is True
        else "key_to_lock_centroid_vector"
    )
    facing = str(actor.get("facing_direction_hypothesis") or "none")
    agreement = "unknown"
    if facing in {"up", "down", "left", "right"}:
        agreement = "agrees" if facing == direction else "conflicts"
    return {
        "schema": ORACLE_KEY_LOCK_ARROW_SCHEMA,
        "status": "hypothesis",
        "relation_kind": "key_points_to_lock",
        "goal_lock_component_id": str(goal.get("component_id")),
        "actor_key_component_id": str(actor.get("component_id")),
        "goal_lock_center_xy": [float(goal_center[0]), float(goal_center[1])],
        "actor_key_center_xy": [float(actor_center[0]), float(actor_center[1])],
        "key_to_lock_vector_xy": [round(dx, 6), round(dy, 6)],
        "key_to_lock_direction": direction,
        "key_to_lock_distance_pixels": round(float(distance), 6),
        "orientation_source": orientation_source,
        "actor_shape_facing_direction": facing,
        "agreement_status": agreement,
        "lock_key_fit": lock_key_fit,
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def actor_goal_relation_from_contra_scene(scene_surfaces: dict[str, Any]) -> dict[str, Any] | None:
    """Return Oracle's first-pass actor/goal relation for a CONTRA scene packet.

    This is a review-only bridge. It lets CONTRA render StateCon/GeoBand
    hypotheses from Oracle's first guess without admitting actor, goal, control,
    action, or solve truth.
    """
    if not isinstance(scene_surfaces, dict):
        return None
    source_packet = {
        "schema": "codexlab.oracle_contra_scene_bridge_source.v1",
        "scene_surfaces": scene_surfaces,
        "raw_color_component_packet": {
            "frame_shape": scene_surfaces.get("frame_shape"),
        },
    }
    goal_candidate = _goal_candidate_from_source_packet(source_packet)
    actor_candidate = _actor_candidate_from_source_packet(source_packet, goal_candidate=goal_candidate)
    key_lock_arrow = _key_points_to_lock_hypothesis(
        goal_candidate=goal_candidate,
        actor_candidate=actor_candidate,
    )
    if not isinstance(goal_candidate, dict) or not isinstance(actor_candidate, dict):
        return None
    if goal_candidate.get("status") != "candidate_selected_for_review":
        return None
    if actor_candidate.get("status") != "candidate_selected_for_review":
        return None
    if not isinstance(goal_candidate.get("selected_component"), dict):
        return None
    if not isinstance(actor_candidate.get("selected_component"), dict):
        return None
    return {
        "schema": "codexlab.oracle_actor_goal_relation_bridge.v1",
        "status": "hypothesis",
        "source": "oracle_first_pass_actor_goal_relation",
        "goal_candidate": goal_candidate,
        "actor_candidate": actor_candidate,
        "key_points_to_lock_hypothesis": key_lock_arrow,
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _goal_candidate_from_source_packet(source_packet: dict[str, Any]) -> dict[str, Any]:
    component_candidates = _component_candidates_from_source_packet(source_packet)
    actor_anchor = _actor_anchor_from_components(component_candidates)
    ranked = sorted(
        component_candidates,
        key=lambda row: (-int(row.get("area_pixels") or 0), str(row.get("component_id"))),
    )
    fit_pool = ranked[:_CAVITY_CANDIDATE_LIMIT]
    fits_by_id = {
        str(component.get("component_id")): _best_key_fit_for_lock(component, fit_pool)
        for component in fit_pool
    }
    candidates = []
    for component in component_candidates:
        geometry = _actor_goal_geometry_metrics(actor_anchor, component)
        actor_goal_geometry = bool(geometry.get("geometry_relation_strong"))
        lock_key_fit = fits_by_id.get(str(component.get("component_id")))
        if lock_key_fit is None:
            lock_key_fit = {
                "status": "skipped_below_size_cap",
                "fit_score": 0.0,
                "lock_key_shape_fit": False,
                "goal_lock_component_id": component.get("component_id"),
                "actor_key_component_id": None,
            }
        lock_key_shape_fit = bool(lock_key_fit.get("lock_key_shape_fit"))
        criteria = {
            "colorful_raw_value": bool(component["colorful_raw_value"]),
            "asymmetric_component": bool(component["asymmetric_component"]),
            "non_centroid_component": bool(component["non_centroid_component"]),
            "not_scene_surface_role": bool(component["not_scene_surface_role"]),
            "actor_goal_geometric_relation": actor_goal_geometry,
            "lock_key_shape_fit": lock_key_shape_fit,
        }
        base_criteria_count = sum(
            1
            for key, value in criteria.items()
            if key not in {"actor_goal_geometric_relation", "lock_key_shape_fit"} and value
        )
        criteria_count = sum(1 for value in criteria.values() if value)
        lock_key_selected = (
            criteria["colorful_raw_value"]
            and criteria["not_scene_surface_role"]
            and criteria["lock_key_shape_fit"]
        )
        relation_selected = (
            criteria["colorful_raw_value"]
            and criteria["non_centroid_component"]
            and criteria["not_scene_surface_role"]
            and criteria["actor_goal_geometric_relation"]
        )
        area = int(component["area_pixels"])
        score = (
            (1.0 if criteria["colorful_raw_value"] else 0.0)
            + (1.0 if criteria["asymmetric_component"] else 0.0)
            + (1.0 if criteria["non_centroid_component"] else 0.0)
            + (0.75 if criteria["not_scene_surface_role"] else 0.0)
            + (1.25 if criteria["actor_goal_geometric_relation"] else 0.0)
            + (2.0 if criteria["lock_key_shape_fit"] else 0.0)
            + min(1.0, area / 256.0)
        )
        row = dict(component)
        row["actor_goal_geometry"] = geometry
        row["lock_key_fit"] = lock_key_fit
        row["criteria"] = criteria
        row["base_criteria_count"] = base_criteria_count
        row["criteria_count"] = criteria_count
        row["lock_key_selected"] = lock_key_selected
        row["relation_selected"] = relation_selected
        row["score"] = round(float(score), 6)
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -int(bool(row.get("lock_key_selected"))),
            -int(bool(row.get("relation_selected"))),
            -int(row["criteria_count"]),
            -float(row["score"]),
            -int(row["area_pixels"]),
            str(row["component_id"]),
        )
    )
    selected = candidates[0] if candidates else None
    base_goal_met = bool(selected and selected.get("base_criteria_count", 0) == 4)
    lock_key_goal_met = bool(selected and selected.get("lock_key_selected") is True)
    relation_goal_met = bool(selected and selected.get("relation_selected") is True)
    status = (
        "candidate_selected_for_review"
        if base_goal_met or lock_key_goal_met or relation_goal_met
        else "weak_or_no_candidate_for_review"
    )
    return {
        "schema": "codexlab.oracle_goal_assumption_candidate.v1",
        "status": status,
        "policy": "if_goal_unknown_assume_largest_colorful_asymmetrical_non_centroid_object",
        "policy_terms": {
            "colorful_raw_values": sorted(COLORFUL_RAW_VALUES),
            "actor_goal_geometry_override": (
                "actor-facing geometric relation can select a colorful non-centroid non-surface goal "
                "even when the goal shape itself is symmetric"
            ),
            "lock_key_shape_fit_override": (
                "a goal lock candidate whose open cavity fits an actor key shape outranks "
                "plain actor-facing geometry; lock and key colors may differ"
            ),
            "floor_pattern_components_excluded": True,
            "asymmetric_threshold": (
                "asymmetry.confidence >= "
                f"{ASYMMETRY_CONFIDENCE_THRESHOLD} or facing_direction_hypothesis != none"
            ),
            "non_centroid_threshold": (
                "distance from frame center / frame diagonal >= "
                f"{NON_CENTROID_DISTANCE_THRESHOLD}"
            ),
            "surface_roles_excluded": True,
        },
        "actor_anchor_component": actor_anchor,
        "selected_component": selected,
        "candidate_count": len(candidates),
        "candidates": candidates[:8],
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _actor_candidate_from_source_packet(
    source_packet: dict[str, Any],
    *,
    goal_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_goal = None
    selected_actor_key = None
    if isinstance(goal_candidate, dict) and isinstance(goal_candidate.get("selected_component"), dict):
        selected_goal = str(goal_candidate["selected_component"].get("component_id"))
        fit = goal_candidate["selected_component"].get("lock_key_fit")
        if isinstance(fit, dict) and fit.get("lock_key_shape_fit") is True and fit.get("key_component_id"):
            selected_actor_key = str(fit.get("key_component_id"))
    candidates = []
    for component in _component_candidates_from_source_packet(source_packet):
        excluded_as_selected_goal = bool(selected_goal and component["component_id"] == selected_goal)
        selected_by_lock_key = bool(selected_actor_key and component["component_id"] == selected_actor_key)
        criteria = {
            "asymmetric_component": bool(component["asymmetric_component"]),
            "not_scene_surface_role": bool(component["not_scene_surface_role"]),
            "not_selected_goal": not excluded_as_selected_goal,
            "fits_selected_goal_lock": selected_by_lock_key,
        }
        criteria_count = sum(1 for value in criteria.values() if value)
        distance = float(component["centroid_distance_ratio"])
        score = (
            (1.0 if criteria["asymmetric_component"] else 0.0)
            + (0.75 if criteria["not_scene_surface_role"] else 0.0)
            + (0.75 if criteria["not_selected_goal"] else 0.0)
            + (2.0 if criteria["fits_selected_goal_lock"] else 0.0)
            + max(0.0, 1.0 - distance)
        )
        row = dict(component)
        row["excluded_as_selected_goal"] = excluded_as_selected_goal
        row["criteria"] = criteria
        row["criteria_count"] = criteria_count
        row["score"] = round(float(score), 6)
        candidates.append(row)
    candidates.sort(
        key=lambda row: (
            -int(bool(row["criteria"].get("fits_selected_goal_lock"))),
            -int(row["criteria_count"]),
            float(row["centroid_distance_ratio"]),
            -float(row["score"]),
            -int(row["area_pixels"]),
            str(row["component_id"]),
        )
    )
    selected = candidates[0] if candidates else None
    lock_key_actor_met = bool(selected and selected.get("criteria", {}).get("fits_selected_goal_lock") is True)
    all_criteria_met = bool(selected and selected.get("criteria_count", 0) >= 3)
    status = (
        "candidate_selected_for_review"
        if lock_key_actor_met or all_criteria_met
        else "weak_or_no_candidate_for_review"
    )
    return {
        "schema": "codexlab.oracle_actor_assumption_candidate.v1",
        "status": status,
        "policy": "if_goal_candidate_exists_assume_actor_is_asymmetrical_non_surface_object_closest_to_centroid",
        "policy_terms": {
            "requires_goal_exclusion": selected_goal is not None,
            "selected_goal_component_id": selected_goal,
            "selected_actor_key_component_id": selected_actor_key,
            "lock_key_override": "if goal is a lock, the fitting key component is the actor candidate",
            "asymmetric_threshold": (
                "asymmetry.confidence >= "
                f"{ASYMMETRY_CONFIDENCE_THRESHOLD} or facing_direction_hypothesis != none"
            ),
            "centroid_rule": "choose the nearest qualifying component to frame center",
            "surface_roles_excluded": True,
        },
        "selected_component": selected,
        "candidate_count": len(candidates),
        "candidates": candidates[:8],
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def source_ref(*, source_system: str, source_packet_path: str | None, json_path: str, source_schema: str | None = None) -> dict[str, Any]:
    return {
        "schema": ORACLE_SOURCE_REF_SCHEMA,
        "source_system": str(source_system),
        "source_packet_path": source_packet_path,
        "source_schema": source_schema,
        "json_path": str(json_path),
    }


def missing_elements_from_contra_packet(
    *,
    contra_packet: dict[str, Any],
    source_packet_path: str | None = None,
) -> list[dict[str, Any]]:
    outputs = contra_packet.get("layer_output_packets")
    if not isinstance(outputs, dict):
        return []
    rows: list[dict[str, Any]] = []
    for idx, output in enumerate(_as_list(outputs.get("outputs"))):
        if not isinstance(output, dict) or output.get("skipped") is not True:
            continue
        output_key = str(output.get("output_key"))
        missing_inputs = [str(value) for value in _as_list(output.get("missing_inputs"))]
        reason_codes = [str(value) for value in _as_list(output.get("reason_codes"))]
        rows.append(
            {
                "schema": ORACLE_MISSING_ELEMENT_SCHEMA,
                "missing_element_id": f"missing_{output_key}",
                "output_key": output_key,
                "layer_id": str(output.get("layer_id")),
                "status": "missing_element_waiting_for_one_assumption",
                "missing_inputs": missing_inputs,
                "reason_codes": reason_codes,
                "source_ref": source_ref(
                    source_system="contra",
                    source_packet_path=source_packet_path,
                    source_schema=outputs.get("schema"),
                    json_path=f"$.layer_output_packets.outputs[{idx}]",
                ),
                "admission_status": "not_admitted",
                "not_prediction": True,
                "not_control_truth": True,
                "not_solve_truth": True,
                **TRUTH_FLAGS,
            }
        )
    return rows


def oracle_missing_element_assumption(
    *,
    missing_element: dict[str, Any],
    source_packet: dict[str, Any] | None = None,
    assumption_text: str | None = None,
    reviewer: str = "unset",
) -> dict[str, Any]:
    output_key = str(missing_element.get("output_key"))
    missing_inputs = [str(value) for value in _as_list(missing_element.get("missing_inputs"))]
    missing_text = ", ".join(missing_inputs) if missing_inputs else "unspecified missing input"
    goal_candidate = None
    actor_candidate = None
    needs_goal_or_actor = "goal_attachment" in missing_inputs or "actor_attachment" in missing_inputs
    if needs_goal_or_actor and isinstance(source_packet, dict):
        goal_candidate = _goal_candidate_from_source_packet(source_packet)
    if needs_goal_or_actor and isinstance(source_packet, dict):
        actor_candidate = _actor_candidate_from_source_packet(
            source_packet,
            goal_candidate=goal_candidate,
        )
    key_lock_arrow = _key_points_to_lock_hypothesis(
        goal_candidate=goal_candidate,
        actor_candidate=actor_candidate,
    )
    statement = assumption_text or f"Assume one review placeholder is needed for {output_key}; missing input: {missing_text}."
    if (
        goal_candidate is not None
        and goal_candidate.get("status") == "candidate_selected_for_review"
        and isinstance(goal_candidate.get("selected_component"), dict)
        and "goal_attachment" in missing_inputs
    ):
        selected = goal_candidate["selected_component"]
        if selected.get("lock_key_selected") is True:
            fit = selected.get("lock_key_fit") if isinstance(selected.get("lock_key_fit"), dict) else {}
            statement = (
                "If goal is unknown, assume the goal candidate for review is the traced lock shape "
                "whose open cavity fits an actor key: "
                f"goal lock {selected.get('component_id')} with actor key {fit.get('key_component_id')}."
            )
        elif selected.get("relation_selected") is True:
            geometry = selected.get("actor_goal_geometry") if isinstance(selected.get("actor_goal_geometry"), dict) else {}
            statement = (
                "If goal is unknown, assume the goal candidate for review is the colorful non-centroid "
                "non-surface component in the actor-facing geometric relation: "
                f"{selected.get('component_id')} from actor {geometry.get('actor_component_id')}."
            )
        else:
            statement = (
                "If goal is unknown, assume the goal candidate for review is the largest colorful, "
                f"asymmetrical, non-centroid component: {selected.get('component_id')}."
            )
        if (
            actor_candidate is not None
            and actor_candidate.get("status") == "candidate_selected_for_review"
            and isinstance(actor_candidate.get("selected_component"), dict)
        ):
            actor = actor_candidate["selected_component"]
            if actor.get("criteria", {}).get("fits_selected_goal_lock") is True:
                statement = (
                    f"{statement} With that goal lock selected, assume the actor candidate for review "
                    f"is the fitting key component: {actor.get('component_id')}."
                )
            else:
                statement = (
                    f"{statement} With that goal excluded, assume the actor candidate for review "
                    "is the asymmetrical non-surface component closest to centroid: "
                    f"{actor.get('component_id')}."
                )
    elif (
        actor_candidate is not None
        and actor_candidate.get("status") == "candidate_selected_for_review"
        and isinstance(actor_candidate.get("selected_component"), dict)
        and "actor_attachment" in missing_inputs
    ):
        actor = actor_candidate["selected_component"]
        if actor.get("criteria", {}).get("fits_selected_goal_lock") is True:
            statement = (
                "If actor/player is unknown and the goal is a lock, assume the actor candidate "
                f"for review is the fitting key component: {actor.get('component_id')}."
            )
        else:
            statement = (
                "If actor/player is unknown, assume the actor candidate for review is the "
                f"asymmetrical non-surface component closest to centroid: {actor.get('component_id')}."
            )
    assumption_payload = {}
    if goal_candidate is not None:
        assumption_payload["goal_candidate"] = goal_candidate
    if actor_candidate is not None:
        assumption_payload["actor_candidate"] = actor_candidate
    if key_lock_arrow is not None:
        assumption_payload["key_points_to_lock_hypothesis"] = key_lock_arrow
    return {
        "schema": ORACLE_ASSUMPTION_SCHEMA,
        "assumption_id": f"oracle_one_missing_element_{output_key}",
        "assumption_kind": "missing_element_placeholder",
        "subject": output_key,
        "statement": statement,
        "status": "candidate_assumption",
        "evidence_status": "assumption_placeholder_for_missing_contra_output",
        "reviewer": str(reviewer),
        "review_priority": "high",
        "missing_element": missing_element,
        "assumption_payload": assumption_payload,
        "basis_refs": [missing_element["source_ref"]] if isinstance(missing_element.get("source_ref"), dict) else [],
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "requires_review": True,
        **TRUTH_FLAGS,
    }


def build_oracle_assumption_packet(
    *,
    source_packet: dict[str, Any],
    source_packet_path: str | None = None,
    source_system: str = "contra",
    assumption_index: int = 0,
    assumption_text: str | None = None,
    reviewer: str = "unset",
) -> dict[str, Any]:
    missing_elements = missing_elements_from_contra_packet(
        contra_packet=source_packet,
        source_packet_path=source_packet_path,
    )
    assumptions: list[dict[str, Any]] = []
    selected_missing_element: dict[str, Any] | None = None
    if missing_elements:
        selected_index = max(0, min(int(assumption_index), len(missing_elements) - 1))
        selected_missing_element = missing_elements[selected_index]
        assumptions = [
            oracle_missing_element_assumption(
                missing_element=selected_missing_element,
                source_packet=source_packet,
                assumption_text=assumption_text,
                reviewer=reviewer,
            )
        ]
    return {
        "schema": ORACLE_ASSUMPTION_PACKET_SCHEMA,
        "source": "oracle_assumption_head",
        "source_system": str(source_system),
        "source_packet_path": source_packet_path,
        "source_packet_schema": source_packet.get("schema"),
        "source_game_id": source_packet.get("game_id"),
        "source_level_index": source_packet.get("level_index"),
        "status": "one_missing_element_assumption_ready_for_review" if assumptions else "no_missing_elements_found",
        "assumption_policy": "one_small_assumption_per_missing_element",
        "missing_element_count": len(missing_elements),
        "selected_missing_element_index": int(assumption_index) if missing_elements else None,
        "selected_missing_element_id": selected_missing_element.get("missing_element_id") if selected_missing_element else None,
        "missing_elements": missing_elements,
        "assumption_count": len(assumptions),
        "assumption_kinds": sorted({str(row.get("assumption_kind")) for row in assumptions}),
        "assumptions": assumptions,
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def attach_oracle_assumption(
    contra_packet: dict[str, Any],
    *,
    source_packet_path: str | None = None,
    assumption_index: int = 0,
    assumption_text: str | None = None,
    reviewer: str = "unset",
) -> dict[str, Any]:
    out = dict(contra_packet)
    out["oracle_assumption_packet"] = build_oracle_assumption_packet(
        source_packet=contra_packet,
        source_packet_path=source_packet_path,
        source_system="contra",
        assumption_index=assumption_index,
        assumption_text=assumption_text,
        reviewer=reviewer,
    )
    return out


def verify_oracle_packet(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("schema") != ORACLE_ASSUMPTION_PACKET_SCHEMA:
        failures.append("bad oracle packet schema")
    if packet.get("admission_status") != "not_admitted":
        failures.append("oracle packet admitted assumptions")
    for key, expected in TRUTH_FLAGS.items():
        if packet.get(key) is not expected:
            failures.append(f"oracle packet {key} must be false")
    rows = packet.get("assumptions")
    if not isinstance(rows, list):
        failures.append("oracle assumptions missing")
        return failures
    if len(rows) > 1:
        failures.append("oracle must emit at most one assumption per packet")
    if packet.get("assumption_count") != len(rows):
        failures.append("oracle assumption_count mismatch")
    missing_elements = packet.get("missing_elements")
    if not isinstance(missing_elements, list):
        failures.append("oracle missing_elements must be a list")
    for idx, missing in enumerate(missing_elements if isinstance(missing_elements, list) else []):
        if not isinstance(missing, dict):
            failures.append(f"missing element {idx} is not an object")
            continue
        if missing.get("schema") != ORACLE_MISSING_ELEMENT_SCHEMA:
            failures.append(f"missing element {idx} bad schema")
        if not missing.get("output_key"):
            failures.append(f"missing element {idx} missing output_key")
        for key, expected in TRUTH_FLAGS.items():
            if missing.get(key) is not expected:
                failures.append(f"missing element {idx} {key} must be false")
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"assumption {idx} is not an object")
            continue
        if row.get("schema") != ORACLE_ASSUMPTION_SCHEMA:
            failures.append(f"assumption {idx} bad schema")
        if row.get("assumption_kind") != "missing_element_placeholder":
            failures.append(f"assumption {idx} must be missing_element_placeholder")
        if row.get("admission_status") != "not_admitted":
            failures.append(f"assumption {idx} admitted")
        if row.get("requires_review") is not True:
            failures.append(f"assumption {idx} does not require review")
        if not isinstance(row.get("basis_refs"), list) or len(row.get("basis_refs")) != 1:
            failures.append(f"assumption {idx} must have exactly one basis ref")
        if not isinstance(row.get("missing_element"), dict):
            failures.append(f"assumption {idx} missing selected missing_element")
        payload = row.get("assumption_payload")
        if isinstance(payload, dict) and isinstance(payload.get("goal_candidate"), dict):
            goal_candidate = payload["goal_candidate"]
            if goal_candidate.get("admission_status") != "not_admitted":
                failures.append(f"assumption {idx} goal candidate admitted")
            for key, expected in TRUTH_FLAGS.items():
                if goal_candidate.get(key) is not expected:
                    failures.append(f"assumption {idx} goal candidate {key} must be false")
            if goal_candidate.get("status") == "candidate_selected_for_review":
                selected = goal_candidate.get("selected_component")
                if not isinstance(selected, dict):
                    failures.append(f"assumption {idx} goal candidate missing selected component")
                elif (
                    selected.get("base_criteria_count") != 4
                    and selected.get("lock_key_selected") is not True
                    and selected.get("relation_selected") is not True
                ):
                    failures.append(f"assumption {idx} goal candidate selected without base or relation criteria")
        if isinstance(payload, dict) and isinstance(payload.get("actor_candidate"), dict):
            actor_candidate = payload["actor_candidate"]
            if actor_candidate.get("admission_status") != "not_admitted":
                failures.append(f"assumption {idx} actor candidate admitted")
            for key, expected in TRUTH_FLAGS.items():
                if actor_candidate.get(key) is not expected:
                    failures.append(f"assumption {idx} actor candidate {key} must be false")
            if actor_candidate.get("status") == "candidate_selected_for_review":
                selected = actor_candidate.get("selected_component")
                if not isinstance(selected, dict):
                    failures.append(f"assumption {idx} actor candidate missing selected component")
                elif selected.get("criteria_count") != 3:
                    failures.append(f"assumption {idx} actor candidate selected without all criteria")
        if isinstance(payload, dict) and isinstance(payload.get("key_points_to_lock_hypothesis"), dict):
            arrow = payload["key_points_to_lock_hypothesis"]
            if arrow.get("schema") != ORACLE_KEY_LOCK_ARROW_SCHEMA:
                failures.append(f"assumption {idx} key-lock arrow bad schema")
            if arrow.get("admission_status") != "not_admitted":
                failures.append(f"assumption {idx} key-lock arrow admitted")
            if arrow.get("key_to_lock_direction") not in {"up", "down", "left", "right", "none"}:
                failures.append(f"assumption {idx} key-lock arrow bad direction")
            for key, expected in TRUTH_FLAGS.items():
                if arrow.get(key) is not expected:
                    failures.append(f"assumption {idx} key-lock arrow {key} must be false")
        for key, expected in TRUTH_FLAGS.items():
            if row.get(key) is not expected:
                failures.append(f"assumption {idx} {key} must be false")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contra-packet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--assumption-index", type=int, default=0)
    parser.add_argument("--assumption-text")
    parser.add_argument("--reviewer", default="unset")
    parser.add_argument("--verify", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_path = args.contra_packet
    source_packet = json.loads(source_path.read_text(encoding="utf-8"))
    packet = build_oracle_assumption_packet(
        source_packet=source_packet,
        source_packet_path=str(source_path),
        source_system="contra",
        assumption_index=int(args.assumption_index),
        assumption_text=args.assumption_text,
        reviewer=str(args.reviewer),
    )
    failures = verify_oracle_packet(packet) if args.verify else []
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} assumptions={packet['assumption_count']} "
        f"missing_elements={packet['missing_element_count']} "
        f"selected={packet['selected_missing_element_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
