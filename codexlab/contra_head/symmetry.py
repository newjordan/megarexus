#!/usr/bin/env python3
"""Static symmetry analysis for CONTRA review packets."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

CONTRA_SYMMETRY_PACKET_SCHEMA = "codexlab.contra_symmetry_hypotheses.v1"
CONTRA_SURFACE_SYMMETRY_SCHEMA = "codexlab.contra_surface_symmetry_hypothesis.v1"
CONTRA_COMPONENT_SELF_SYMMETRY_SCHEMA = "codexlab.contra_component_self_symmetry_hypothesis.v1"
CONTRA_COMPONENT_MIRROR_PAIR_SCHEMA = "codexlab.contra_component_mirror_pair_hypothesis.v1"
CONTRA_COMPONENT_REVERSE_SYMMETRY_SCHEMA = "codexlab.contra_component_reverse_symmetry_hypothesis.v1"
CONTRA_COMPONENT_POLAR_SYMMETRY_SCHEMA = "codexlab.contra_component_polar_symmetry_hypothesis.v1"
CONTRA_SYMMETRY_LOCK_KEY_SENSOR_SCHEMA = "codexlab.contra_symmetry_lock_key_sensor_hypothesis.v1"
CONTRA_CENTER_DIVIDER_SCHEMA = "codexlab.contra_center_divider_hypothesis.v1"
CONTRA_PLAYZONE_MIRROR_SCHEMA = "codexlab.contra_playzone_mirror_hypothesis.v1"
CONTRA_SYMMETRY_PASS_SCHEMA = "codexlab.contra_symmetry_pass.v1"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_token(value: Any) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_") or "none"


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _bbox_center(box: list[int]) -> list[float]:
    return [round(float(box[0] + box[2]) / 2.0, 6), round(float(box[1] + box[3]) / 2.0, 6)]


def _bbox_union(boxes: list[list[int]]) -> list[int]:
    return [
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    ]


def _mask_symmetry_score(mask: np.ndarray, *, axis: str) -> float:
    bool_mask = np.asarray(mask).astype(bool)
    if bool_mask.size == 0 or not bool_mask.any():
        return 0.0
    mirrored = bool_mask[:, ::-1] if axis == "vertical" else bool_mask[::-1, :]
    union = np.logical_or(bool_mask, mirrored).sum()
    if int(union) <= 0:
        return 0.0
    overlap = np.logical_and(bool_mask, mirrored).sum()
    return round(clamp01(float(overlap) / float(union)), 6)


def _mask_for_components(label_map: np.ndarray, component_lookup: dict[str, int], component_ids: list[str]) -> np.ndarray:
    mask = np.zeros(label_map.shape[:2], dtype=bool)
    for component_id in component_ids:
        label_index = component_lookup.get(str(component_id))
        if label_index is None:
            continue
        mask |= label_map == int(label_index)
    return mask


def _component_ids_from_rows(rows: Any) -> list[str]:
    ids: list[str] = []
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("component_id"):
            ids.append(str(row["component_id"]))
    return ids


def _floor_component_ids(scene_surfaces: dict[str, Any]) -> set[str]:
    floor = scene_surfaces.get("floor_hypothesis")
    ids: set[str] = set()
    if isinstance(floor, dict) and floor.get("component_id"):
        ids.add(str(floor["component_id"]))
    for row in scene_surfaces.get("floor_pattern_hypotheses", []) if isinstance(scene_surfaces.get("floor_pattern_hypotheses"), list) else []:
        if not isinstance(row, dict):
            continue
        ids.update(str(value) for value in row.get("component_ids", []) if value is not None)
    return ids


def _surface_component_groups(scene_surfaces: dict[str, Any]) -> list[tuple[str, list[str], list[dict[str, Any]]]]:
    groups: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    floor = scene_surfaces.get("floor_hypothesis")
    if isinstance(floor, dict) and floor.get("component_id"):
        groups.append(("floor", [str(floor["component_id"])], [floor]))
    backgrounds = scene_surfaces.get("background_hypotheses")
    background_ids = _component_ids_from_rows(backgrounds)
    if background_ids:
        groups.append(("background", background_ids, [row for row in backgrounds if isinstance(row, dict)]))
    walls = scene_surfaces.get("wall_hypotheses")
    wall_ids = _component_ids_from_rows(walls)
    if wall_ids:
        groups.append(("wall", wall_ids, [row for row in walls if isinstance(row, dict)]))
    return groups


def _axis_row(
    *,
    group_kind: str,
    axis: str,
    component_ids: list[str],
    rows: list[dict[str, Any]],
    score: float,
    frame_shape: tuple[int, int],
) -> dict[str, Any]:
    boxes = [[int(value) for value in row["bbox_xyxy"]] for row in rows if isinstance(row.get("bbox_xyxy"), list)]
    bbox = _bbox_union(boxes) if boxes else [0, 0, int(frame_shape[1]) - 1, int(frame_shape[0]) - 1]
    group_id = _stable_id("symmetry_surface", group_kind, axis, ",".join(component_ids))
    return {
        "schema": CONTRA_SURFACE_SYMMETRY_SCHEMA,
        "status": "hypothesis",
        "symmetry_group_id": group_id,
        "symmetry_pass_id": "surface_context_pass",
        "symmetry_pass_order": 1,
        "symmetry_kind": "surface_mask_reflection",
        "axis": axis,
        "surface_role": group_kind,
        "member_component_ids": component_ids,
        "member_component_count": len(component_ids),
        "bbox_xyxy": bbox,
        "center_xy": _bbox_center(bbox),
        "score": round(clamp01(score), 6),
        "mirror_scenario_clue": True,
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "derived_from_raw_surface_mask_reflection",
        "rendering_role": "symmetry_review_group",
    }


def _component_mask(component: dict[str, Any]) -> np.ndarray:
    rows = component.get("shape_mask_rows")
    if not isinstance(rows, list) or not rows:
        return np.zeros((0, 0), dtype=bool)
    return np.asarray([[ch == "1" for ch in str(row)] for row in rows], dtype=bool)


def _component_self_symmetry(component: dict[str, Any], *, axis: str) -> dict[str, Any] | None:
    mask = _component_mask(component)
    if mask.size == 0:
        return None
    score = _mask_symmetry_score(mask, axis=axis)
    if score < 0.88:
        return None
    component_id = str(component.get("component_id"))
    box = [int(value) for value in component.get("bbox_xyxy", [])]
    if len(box) != 4:
        return None
    return {
        "schema": CONTRA_COMPONENT_SELF_SYMMETRY_SCHEMA,
        "status": "hypothesis",
        "symmetry_group_id": _stable_id("symmetry_component", component_id, axis),
        "symmetry_pass_id": "component_self_symmetry_pass",
        "symmetry_pass_order": 4,
        "symmetry_kind": "component_self_reflection",
        "axis": axis,
        "member_component_ids": [component_id],
        "member_component_count": 1,
        "raw_pixel_values": [int(component.get("raw_pixel_value"))],
        "bbox_xyxy": box,
        "center_xy": _bbox_center(box),
        "score": round(clamp01(score), 6),
        "mirror_scenario_clue": True,
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "derived_from_raw_component_mask_reflection",
        "rendering_role": "symmetry_review_group",
    }


def _axis_pair_score(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    axis: str,
    frame_shape: tuple[int, int],
) -> float:
    height, width = frame_shape
    left_center = left.get("center_xy") if isinstance(left.get("center_xy"), list) else [0.0, 0.0]
    right_center = right.get("center_xy") if isinstance(right.get("center_xy"), list) else [0.0, 0.0]
    lx, ly = float(left_center[0]), float(left_center[1])
    rx, ry = float(right_center[0]), float(right_center[1])
    if axis == "vertical":
        expected_rx = float(width - 1) - lx
        expected_ry = ly
        norm = max(1.0, float(width))
    else:
        expected_rx = lx
        expected_ry = float(height - 1) - ly
        norm = max(1.0, float(height))
    distance_penalty = clamp01(math.dist((expected_rx, expected_ry), (rx, ry)) / norm)
    left_box = [int(value) for value in left.get("bbox_xyxy", [])]
    right_box = [int(value) for value in right.get("bbox_xyxy", [])]
    if len(left_box) != 4 or len(right_box) != 4:
        return 0.0
    left_w = max(1, left_box[2] - left_box[0] + 1)
    left_h = max(1, left_box[3] - left_box[1] + 1)
    right_w = max(1, right_box[2] - right_box[0] + 1)
    right_h = max(1, right_box[3] - right_box[1] + 1)
    width_similarity = min(left_w, right_w) / max(left_w, right_w)
    height_similarity = min(left_h, right_h) / max(left_h, right_h)
    area_left = max(1, int(left.get("area_pixels") or 0))
    area_right = max(1, int(right.get("area_pixels") or 0))
    area_similarity = min(area_left, area_right) / max(area_left, area_right)
    raw_match = 1.0 if int(left.get("raw_pixel_value")) == int(right.get("raw_pixel_value")) else 0.0
    return round(
        clamp01(
            0.42 * (1.0 - distance_penalty)
            + 0.18 * width_similarity
            + 0.18 * height_similarity
            + 0.14 * area_similarity
            + 0.08 * raw_match
        ),
        6,
    )


def _component_pair_symmetry(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    axis: str,
    frame_shape: tuple[int, int],
) -> dict[str, Any] | None:
    score = _axis_pair_score(left, right, axis=axis, frame_shape=frame_shape)
    if score < 0.76:
        return None
    left_id = str(left.get("component_id"))
    right_id = str(right.get("component_id"))
    boxes = [[int(value) for value in left["bbox_xyxy"]], [int(value) for value in right["bbox_xyxy"]]]
    bbox = _bbox_union(boxes)
    return {
        "schema": CONTRA_COMPONENT_MIRROR_PAIR_SCHEMA,
        "status": "hypothesis",
        "symmetry_group_id": _stable_id("symmetry_pair", axis, left_id, right_id),
        "symmetry_pass_id": "component_mirror_pair_pass",
        "symmetry_pass_order": 5,
        "symmetry_kind": "component_mirror_pair",
        "axis": axis,
        "member_component_ids": [left_id, right_id],
        "member_component_count": 2,
        "raw_pixel_values": [int(left.get("raw_pixel_value")), int(right.get("raw_pixel_value"))],
        "raw_pixel_match": int(left.get("raw_pixel_value")) == int(right.get("raw_pixel_value")),
        "bbox_xyxy": bbox,
        "center_xy": _bbox_center(bbox),
        "score": score,
        "mirror_scenario_clue": True,
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "derived_from_component_bbox_reflection_similarity",
        "rendering_role": "symmetry_review_group",
    }


def _reflected_mask_for_axis(mask: np.ndarray, *, axis: str) -> np.ndarray:
    if axis == "vertical":
        return mask[:, ::-1]
    if axis == "horizontal":
        return mask[::-1, :]
    return mask


def _reverse_mask_metrics(left: dict[str, Any], right: dict[str, Any], *, axis: str) -> dict[str, float] | None:
    left_mask = _component_mask(left)
    right_mask = _reflected_mask_for_axis(_component_mask(right), axis=axis)
    if left_mask.size == 0 or right_mask.size == 0 or left_mask.shape != right_mask.shape:
        return None
    left_area = int(left_mask.sum())
    right_area = int(right_mask.sum())
    if left_area <= 0 or right_area <= 0:
        return None
    intersection = int(np.logical_and(left_mask, right_mask).sum())
    union = int(np.logical_or(left_mask, right_mask).sum())
    symmetric_overlap = float(intersection) / float(max(1, min(left_area, right_area)))
    fill_difference = abs(left_area - right_area) / float(max(1, max(left_area, right_area)))
    xor_ratio = float(np.logical_xor(left_mask, right_mask).sum()) / float(max(1, union))
    return {
        "symmetric_overlap": round(clamp01(symmetric_overlap), 6),
        "fill_difference": round(clamp01(fill_difference), 6),
        "xor_ratio": round(clamp01(xor_ratio), 6),
    }


def _component_reverse_symmetry(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    axis: str,
    frame_shape: tuple[int, int],
) -> dict[str, Any] | None:
    box_score = _axis_pair_score(left, right, axis=axis, frame_shape=frame_shape)
    if box_score < 0.72:
        return None
    metrics = _reverse_mask_metrics(left, right, axis=axis)
    if metrics is None:
        return None
    overlap = float(metrics["symmetric_overlap"])
    fill_difference = float(metrics["fill_difference"])
    xor_ratio = float(metrics["xor_ratio"])
    raw_mismatch = int(left.get("raw_pixel_value")) != int(right.get("raw_pixel_value"))
    reverse_fill_signal = fill_difference >= 0.06 or xor_ratio >= 0.08 or raw_mismatch
    if overlap < 0.72 or not reverse_fill_signal:
        return None
    score = round(clamp01(0.48 * box_score + 0.34 * overlap + 0.10 * xor_ratio + 0.08 * (1.0 if raw_mismatch else 0.0)), 6)
    if score < 0.72:
        return None
    left_id = str(left.get("component_id"))
    right_id = str(right.get("component_id"))
    boxes = [[int(value) for value in left["bbox_xyxy"]], [int(value) for value in right["bbox_xyxy"]]]
    bbox = _bbox_union(boxes)
    return {
        "schema": CONTRA_COMPONENT_REVERSE_SYMMETRY_SCHEMA,
        "status": "hypothesis",
        "symmetry_group_id": _stable_id("reverse_symmetry_pair", axis, left_id, right_id),
        "symmetry_pass_id": "component_reverse_symmetry_pass",
        "symmetry_pass_order": 6,
        "symmetry_kind": "component_reverse_symmetry_pair",
        "axis": axis,
        "member_component_ids": [left_id, right_id],
        "member_component_count": 2,
        "raw_pixel_values": [int(left.get("raw_pixel_value")), int(right.get("raw_pixel_value"))],
        "raw_pixel_match": not raw_mismatch,
        "bbox_xyxy": bbox,
        "center_xy": _bbox_center(bbox),
        "score": score,
        "reverse_symmetry_metrics": metrics,
        "reverse_symmetry_kind": "mirrored_shape_with_reverse_or_negative_space_fill",
        "mirror_scenario_clue": True,
        "reverse_symmetry_clue": True,
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "derived_from_reflected_component_mask_overlap_with_reverse_fill_difference",
        "rendering_role": "symmetry_review_group",
    }


def _component_polar_symmetry(reverse_row: dict[str, Any]) -> dict[str, Any]:
    member_ids = [str(value) for value in reverse_row.get("member_component_ids", [])]
    axis = str(reverse_row.get("axis") or "unknown")
    row = dict(reverse_row)
    row.update(
        {
            "schema": CONTRA_COMPONENT_POLAR_SYMMETRY_SCHEMA,
            "symmetry_group_id": _stable_id("polar_symmetry_pair", axis, *member_ids),
            "symmetry_pass_id": "component_polar_symmetry_pass",
            "symmetry_pass_order": 7,
            "symmetry_kind": "component_polar_symmetry_pair",
            "polar_symmetry_kind": "opposing_handed_mirrored_shape_pair",
            "polar_symmetry_clue": True,
            "polar_symmetry_metrics": reverse_row.get("reverse_symmetry_metrics", {}),
            "based_on_reverse_symmetry_group_id": reverse_row.get("symmetry_group_id"),
            "provenance": "derived_from_reverse_symmetry_pair_as_opposing_handed_polar_relation",
        }
    )
    return row


def _component_fill_ratio(component: dict[str, Any]) -> float:
    box = [int(value) for value in component.get("bbox_xyxy", [])]
    if len(box) != 4:
        return 0.0
    box_area = max(1, (box[2] - box[0] + 1) * (box[3] - box[1] + 1))
    return round(clamp01(float(component.get("area_pixels") or 0) / float(box_area)), 6)


def _symmetry_lock_key_sensor(polar_row: dict[str, Any], component_lookup: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    member_ids = [str(value) for value in polar_row.get("member_component_ids", [])]
    if len(member_ids) != 2:
        return None
    left = component_lookup.get(member_ids[0])
    right = component_lookup.get(member_ids[1])
    if not isinstance(left, dict) or not isinstance(right, dict):
        return None
    left_fill = _component_fill_ratio(left)
    right_fill = _component_fill_ratio(right)
    if left_fill >= right_fill:
        key_id, lock_id = member_ids[0], member_ids[1]
        key_fill, lock_fill = left_fill, right_fill
    else:
        key_id, lock_id = member_ids[1], member_ids[0]
        key_fill, lock_fill = right_fill, left_fill
    confidence = round(clamp01(0.70 * float(polar_row.get("score") or 0.0) + 0.30 * abs(key_fill - lock_fill)), 6)
    sensor_id = _stable_id("symmetry_lock_key_sensor", polar_row.get("symmetry_group_id"), key_id, lock_id)
    return {
        "schema": CONTRA_SYMMETRY_LOCK_KEY_SENSOR_SCHEMA,
        "status": "hypothesis",
        "sensor_id": sensor_id,
        "sensor_layer_id": "symmetry_sensor_layer",
        "sensor_kind": "polar_lock_key_candidate",
        "source_symmetry_group_id": polar_row.get("symmetry_group_id"),
        "source_symmetry_kind": polar_row.get("symmetry_kind"),
        "axis": polar_row.get("axis"),
        "member_component_ids": member_ids,
        "key_candidate_component_id": key_id,
        "lock_candidate_component_id": lock_id,
        "key_fill_ratio": key_fill,
        "lock_fill_ratio": lock_fill,
        "key_lock_assignment_rule": "higher_fill_ratio_as_key_lower_fill_ratio_as_lock",
        "confidence": confidence,
        "bbox_xyxy": polar_row.get("bbox_xyxy"),
        "center_xy": polar_row.get("center_xy"),
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "derived_from_component_polar_symmetry_and_fill_ratio_heuristic",
        "rendering_role": "symmetry_sensor_review_marker",
    }


def _component_lookup_by_id(scene_surfaces: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("component_id")): row
        for row in scene_surfaces.get("component_graph", {}).get("components", [])
        if isinstance(row, dict) and row.get("component_id")
    }


def _center_divider_hypotheses(
    *,
    components: list[dict[str, Any]],
    frame_shape: tuple[int, int],
    excluded_component_ids: set[str],
) -> list[dict[str, Any]]:
    height, width = frame_shape
    center_x = float(width - 1) / 2.0
    rows: list[dict[str, Any]] = []
    for component in components:
        component_id = str(component.get("component_id"))
        if component_id in excluded_component_ids:
            continue
        box = [int(value) for value in component.get("bbox_xyxy", [])]
        if len(box) != 4:
            continue
        bbox_w = box[2] - box[0] + 1
        bbox_h = box[3] - box[1] + 1
        component_center_x = float(box[0] + box[2]) / 2.0
        center_offset = abs(component_center_x - center_x) / max(1.0, float(width))
        height_coverage = float(bbox_h) / max(1.0, float(height))
        thinness = 1.0 - clamp01(float(bbox_w) / max(1.0, float(width) * 0.14))
        score = clamp01(0.45 * height_coverage + 0.35 * (1.0 - center_offset * 6.0) + 0.20 * thinness)
        if score < 0.58 or height_coverage < 0.45:
            continue
        rows.append(
            {
                "schema": CONTRA_CENTER_DIVIDER_SCHEMA,
                "status": "hypothesis",
                "symmetry_group_id": _stable_id("symmetry_center_divider", component_id),
                "symmetry_pass_id": "center_divider_pass",
                "symmetry_pass_order": 2,
                "symmetry_kind": "center_divider",
                "axis": "vertical",
                "member_component_ids": [component_id],
                "member_component_count": 1,
                "raw_pixel_values": [int(component.get("raw_pixel_value"))],
                "bbox_xyxy": box,
                "center_xy": _bbox_center(box),
                "score": round(score, 6),
                "height_coverage": round(height_coverage, 6),
                "center_offset_ratio": round(center_offset, 6),
                "mirror_scenario_clue": True,
                "floor_excluded": True,
                "evidence_status": "static_geometry_hypothesis",
                "admission_status": "not_admitted",
                "not_prediction": True,
                "not_control_truth": True,
                "not_solve_truth": True,
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "provenance": "derived_from_centered_non_floor_vertical_component_geometry",
                "rendering_role": "symmetry_review_group",
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), row["symmetry_group_id"]))
    return rows[:4]


def _non_floor_mask(label_map: np.ndarray, excluded_label_indexes: set[int]) -> np.ndarray:
    mask = np.ones(label_map.shape[:2], dtype=bool)
    for label_index in excluded_label_indexes:
        mask &= label_map != int(label_index)
    return mask


def _balanced_halves_score(mask: np.ndarray, *, divider_x: int) -> float:
    height, width = mask.shape[:2]
    left_width = max(0, int(divider_x))
    right_width = max(0, int(width - divider_x - 1))
    side = min(left_width, right_width)
    if side <= 0 or height <= 0:
        return 0.0
    left = mask[:, divider_x - side : divider_x]
    right = mask[:, divider_x + 1 : divider_x + 1 + side]
    if left.shape != right.shape:
        return 0.0
    union = np.logical_or(left, right[:, ::-1]).sum()
    if int(union) <= 0:
        return 0.0
    overlap = np.logical_and(left, right[:, ::-1]).sum()
    return round(clamp01(float(overlap) / float(union)), 6)


def _playzone_mirror_hypotheses(
    *,
    divider_rows: list[dict[str, Any]],
    components: list[dict[str, Any]],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    excluded_component_ids: set[str],
    frame_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    height, width = frame_shape
    excluded_label_indexes = {
        int(component_lookup[component_id])
        for component_id in excluded_component_ids
        if component_id in component_lookup
    }
    rows: list[dict[str, Any]] = []
    component_centers = {
        str(component.get("component_id")): component.get("center_xy")
        for component in components
        if isinstance(component.get("center_xy"), list) and str(component.get("component_id")) not in excluded_component_ids
    }
    for divider in divider_rows:
        divider_box = [int(value) for value in divider["bbox_xyxy"]]
        divider_component_ids = {str(value) for value in divider.get("member_component_ids", [])}
        divider_x = int(round(float(divider_box[0] + divider_box[2]) / 2.0))
        divider_label_indexes = {
            int(component_lookup[component_id])
            for component_id in divider_component_ids
            if component_id in component_lookup
        }
        mask = _non_floor_mask(label_map, excluded_label_indexes.union(divider_label_indexes))
        score = _balanced_halves_score(mask, divider_x=divider_x)
        left_ids: list[str] = []
        right_ids: list[str] = []
        for component_id, center in component_centers.items():
            if component_id in divider_component_ids:
                continue
            cx = float(center[0])
            if cx < divider_x:
                left_ids.append(component_id)
            elif cx > divider_x:
                right_ids.append(component_id)
        group_id = _stable_id("symmetry_playzone_mirror", divider.get("symmetry_group_id"), ",".join(left_ids), ",".join(right_ids))
        rows.append(
            {
                "schema": CONTRA_PLAYZONE_MIRROR_SCHEMA,
                "status": "hypothesis",
                "symmetry_group_id": group_id,
                "symmetry_pass_id": "playzone_mirror_pass",
                "symmetry_pass_order": 3,
                "symmetry_kind": "left_right_playzone_mirror",
                "axis": "vertical",
                "divider_symmetry_group_id": divider.get("symmetry_group_id"),
                "divider_component_ids": list(divider_component_ids),
                "left_component_ids": sorted(left_ids),
                "right_component_ids": sorted(right_ids),
                "member_component_ids": [*sorted(left_ids), *sorted(right_ids)],
                "member_component_count": len(left_ids) + len(right_ids),
                "left_playzone_bbox_xyxy": [0, 0, max(0, divider_x - 1), int(height) - 1],
                "right_playzone_bbox_xyxy": [min(int(width) - 1, divider_x + 1), 0, int(width) - 1, int(height) - 1],
                "bbox_xyxy": [0, 0, int(width) - 1, int(height) - 1],
                "center_xy": [round(float(width - 1) / 2.0, 6), round(float(height - 1) / 2.0, 6)],
                "score": score,
                "mirror_scenario_clue": True,
                "floor_excluded": True,
                "evidence_status": "static_geometry_hypothesis",
                "admission_status": "not_admitted",
                "not_prediction": True,
                "not_control_truth": True,
                "not_solve_truth": True,
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "provenance": "derived_from_center_divider_non_floor_left_right_playzone_geometry",
                "rendering_role": "symmetry_review_group",
            }
        )
    rows.sort(key=lambda row: (-float(row["score"]), row["symmetry_group_id"]))
    return rows


def _pass_row(*, pass_order: int, pass_id: str, definition: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": CONTRA_SYMMETRY_PASS_SCHEMA,
        "status": "hypothesis",
        "pass_order": int(pass_order),
        "pass_id": pass_id,
        "definition": definition,
        "output_group_ids": [row.get("symmetry_group_id") or row.get("sensor_id") for row in rows],
        "output_group_count": len(rows),
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


# Pair and reverse passes only keep the highest-scoring 96 rows, but they used
# to score every unordered pair. A noisy 64x64 board has thousands of
# components, and that quadratic pass does not finish. Components beyond this
# cap still receive self-symmetry; pairwise search uses the largest ones.
_PAIRWISE_COMPONENT_LIMIT = 96


def _pairwise_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(components) <= _PAIRWISE_COMPONENT_LIMIT:
        return components
    return sorted(
        components,
        key=lambda row: (-int(row.get("area_pixels") or 0), str(row.get("component_id"))),
    )[:_PAIRWISE_COMPONENT_LIMIT]


def build_symmetry_hypotheses(
    *,
    frame: np.ndarray,
    scene_surfaces: dict[str, Any],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
) -> dict[str, Any]:
    frame_shape = tuple(int(v) for v in frame.shape[:2])
    floor_component_ids = _floor_component_ids(scene_surfaces)
    component_rows_by_id = _component_lookup_by_id(scene_surfaces)
    surface_rows: list[dict[str, Any]] = []
    for role, component_ids, rows in _surface_component_groups(scene_surfaces):
        mask = _mask_for_components(label_map, component_lookup, component_ids)
        for axis in ("vertical", "horizontal"):
            surface_rows.append(
                _axis_row(
                    group_kind=role,
                    axis=axis,
                    component_ids=component_ids,
                    rows=rows,
                    score=_mask_symmetry_score(mask, axis=axis),
                    frame_shape=frame_shape,
                )
            )

    components = [
        row
        for row in scene_surfaces.get("component_graph", {}).get("components", [])
        if isinstance(row, dict) and int(row.get("area_pixels") or 0) > 0
    ]
    non_floor_components = [
        row
        for row in components
        if str(row.get("component_id")) not in floor_component_ids
    ]
    divider_rows = _center_divider_hypotheses(
        components=non_floor_components,
        frame_shape=frame_shape,
        excluded_component_ids=floor_component_ids,
    )
    divider_component_ids = {
        str(value)
        for row in divider_rows
        for value in row.get("member_component_ids", [])
        if value is not None
    }
    playzone_rows = _playzone_mirror_hypotheses(
        divider_rows=divider_rows,
        components=components,
        label_map=label_map,
        component_lookup=component_lookup,
        excluded_component_ids=floor_component_ids,
        frame_shape=frame_shape,
    )
    self_rows: list[dict[str, Any]] = []
    for component in non_floor_components:
        if str(component.get("component_id")) in divider_component_ids:
            continue
        for axis in ("vertical", "horizontal"):
            row = _component_self_symmetry(component, axis=axis)
            if row is not None:
                self_rows.append(row)

    pair_rows: list[dict[str, Any]] = []
    pair_components = _pairwise_components(
        [
            component
            for component in non_floor_components
            if str(component.get("component_id")) not in divider_component_ids
        ]
    )
    for idx, left in enumerate(pair_components):
        for right in pair_components[idx + 1 :]:
            for axis in ("vertical", "horizontal"):
                row = _component_pair_symmetry(left, right, axis=axis, frame_shape=frame_shape)
                if row is not None:
                    pair_rows.append(row)
    pair_rows.sort(key=lambda row: (-float(row["score"]), row["symmetry_group_id"]))
    pair_rows = pair_rows[:96]
    reverse_rows: list[dict[str, Any]] = []
    for idx, left in enumerate(pair_components):
        for right in pair_components[idx + 1 :]:
            for axis in ("vertical", "horizontal"):
                row = _component_reverse_symmetry(left, right, axis=axis, frame_shape=frame_shape)
                if row is not None:
                    reverse_rows.append(row)
    reverse_rows.sort(key=lambda row: (-float(row["score"]), row["symmetry_group_id"]))
    reverse_rows = reverse_rows[:96]
    polar_rows = [_component_polar_symmetry(row) for row in reverse_rows]
    lock_key_sensor_rows = [
        row
        for row in (_symmetry_lock_key_sensor(polar_row, component_rows_by_id) for polar_row in polar_rows)
        if row is not None
    ]

    relationship_groups = sorted(
        [*surface_rows, *divider_rows, *playzone_rows, *self_rows, *pair_rows, *reverse_rows, *polar_rows],
        key=lambda row: (int(row.get("symmetry_pass_order") or 99), -float(row.get("score") or 0.0), str(row.get("symmetry_group_id"))),
    )
    passes = [
        _pass_row(
            pass_order=1,
            pass_id="surface_context_pass",
            definition="measure background/floor/wall reflection as context, not as mirror-object grouping",
            rows=surface_rows,
        ),
        _pass_row(
            pass_order=2,
            pass_id="center_divider_pass",
            definition="detect non-floor centered divider geometry that may split mirrored play zones",
            rows=divider_rows,
        ),
        _pass_row(
            pass_order=3,
            pass_id="playzone_mirror_pass",
            definition="group left and right non-floor play zones around detected center dividers",
            rows=playzone_rows,
        ),
        _pass_row(
            pass_order=4,
            pass_id="component_self_symmetry_pass",
            definition="group non-floor self-symmetric component masks",
            rows=self_rows,
        ),
        _pass_row(
            pass_order=5,
            pass_id="component_mirror_pair_pass",
            definition="group non-floor mirrored component-pair potentials, allowing color mismatch",
            rows=pair_rows,
        ),
        _pass_row(
            pass_order=6,
            pass_id="component_reverse_symmetry_pass",
            definition="group non-floor mirrored components with reversed fill or negative-space structure",
            rows=reverse_rows,
        ),
        _pass_row(
            pass_order=7,
            pass_id="component_polar_symmetry_pass",
            definition="promote opposing-handed mirrored component pairs as polar symmetry clues for gameplay review",
            rows=polar_rows,
        ),
        _pass_row(
            pass_order=8,
            pass_id="symmetry_sensor_layer_pass",
            definition="derive static lock/key-style review sensors from polar symmetry groups",
            rows=lock_key_sensor_rows,
        ),
    ]
    non_floor_group_count = sum(
        1
        for row in relationship_groups
        if row.get("surface_role") != "floor"
    )
    return {
        "schema": CONTRA_SYMMETRY_PACKET_SCHEMA,
        "status": "hypothesis",
        "definition": "sequential_static_reflection_geometry_clues_for_review_only",
        "frame_shape_hw": [int(frame_shape[0]), int(frame_shape[1])],
        "floor_component_ids_excluded_from_mirror_potential_groups": sorted(floor_component_ids),
        "symmetry_passes": passes,
        "symmetry_pass_count": len(passes),
        "surface_symmetry_hypotheses": surface_rows,
        "center_divider_hypotheses": divider_rows,
        "playzone_mirror_hypotheses": playzone_rows,
        "component_self_symmetry_hypotheses": self_rows,
        "component_mirror_pair_hypotheses": pair_rows,
        "component_reverse_symmetry_hypotheses": reverse_rows,
        "component_polar_symmetry_hypotheses": polar_rows,
        "symmetry_sensor_hypotheses": lock_key_sensor_rows,
        "symmetry_sensor_count": len(lock_key_sensor_rows),
        "lock_key_sensor_hypotheses": lock_key_sensor_rows,
        "lock_key_sensor_count": len(lock_key_sensor_rows),
        "symmetry_relationship_groups": relationship_groups,
        "symmetry_relationship_group_count": len(relationship_groups),
        "non_floor_symmetry_relationship_group_count": non_floor_group_count,
        "mirror_scenario_clue": bool(relationship_groups),
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "derived_from_raw_frame_scene_surface_and_component_geometry",
    }


def _intersects(left: list[int], right: list[int]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def symmetry_groups_for_statecon(
    symmetry_packet: dict[str, Any],
    *,
    member_component_ids: set[str],
    group_box: list[int],
    min_surface_score: float = 0.65,
) -> list[dict[str, Any]]:
    groups = symmetry_packet.get("symmetry_relationship_groups") if isinstance(symmetry_packet, dict) else []
    candidates: list[tuple[int, int, float, str, dict[str, Any]]] = []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("symmetry_group_id"))
        if not group_id:
            continue
        components = {str(value) for value in group.get("member_component_ids", []) if value is not None}
        score = float(group.get("score") or 0.0)
        box = group.get("bbox_xyxy")
        bbox_hits = isinstance(box, list) and len(box) == 4 and _intersects([int(v) for v in box], group_box)
        member_hits = bool(components.intersection(member_component_ids))
        surface_context = group.get("symmetry_kind") == "surface_mask_reflection" and score >= min_surface_score
        reverse_context = group.get("symmetry_kind") == "component_reverse_symmetry_pair"
        polar_context = group.get("symmetry_kind") == "component_polar_symmetry_pair"
        if member_hits or (bbox_hits and score >= 0.82) or surface_context:
            priority = 0 if member_hits and polar_context else 1 if member_hits and reverse_context else 2 if member_hits else 3 if bbox_hits else 4
            pass_order = int(group.get("symmetry_pass_order") or 99)
            candidates.append((priority, pass_order, -score, group_id, group))
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, _, group_id, group in sorted(candidates, key=lambda row: row[:4]):
        if group_id in seen:
            continue
        selected.append(group)
        seen.add(group_id)
        if len(selected) >= 18:
            break
    return selected
