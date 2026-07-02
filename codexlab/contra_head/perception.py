#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np

from .relationship_ladder import build_contra_relationship_ladder
from .statecon import build_state_condition_hypotheses, build_statecon_support_affordances
from .symmetry import build_symmetry_hypotheses
from .vision import analyze_scene_surface_hypotheses

CONTRA_PERCEPTION_PIPELINE_SCHEMA = "codexlab.contra_perception_pipeline.v1"
SUPPORT_ACTION_AFFORDANCE_SCHEMA = "codexlab.support_action_affordance_hypothesis.v1"
SUPPORT_ACTION_AFFORDANCE_SET_SCHEMA = "codexlab.support_action_affordance_hypotheses.v1"
COMPOUND_OBJECT_HYPOTHESIS_SCHEMA = "codexlab.compound_object_hypothesis.v1"


def support_shape_family_from_box(box: tuple[int, int, int, int]) -> str:
    x0, y0, x1, y1 = box
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    if bbox_w >= bbox_h * 2:
        return "horizontal_bar"
    if bbox_h >= bbox_w * 2:
        return "vertical_bar"
    return "compact_component"


def bbox_gap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    horizontal_gap = max(0, max(left[0], right[0]) - min(left[2], right[2]) - 1)
    vertical_gap = max(0, max(left[1], right[1]) - min(left[3], right[3]) - 1)
    return max(horizontal_gap, vertical_gap)


def horizontal_overlap_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[2], right[2]) - max(left[0], right[0]) + 1)
    denom = max(1, min(left[2] - left[0] + 1, right[2] - right[0] + 1))
    return float(overlap) / float(denom)


def vertical_overlap_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    overlap = max(0, min(left[3], right[3]) - max(left[1], right[1]) + 1)
    denom = max(1, min(left[3] - left[1] + 1, right[3] - right[1] + 1))
    return float(overlap) / float(denom)


def annotate_support_consistency(supports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for support in supports:
        if not isinstance(support, dict):
            continue
        box = bbox_from_component(support)
        shape_family = support_shape_family_from_box(box)
        support["support_shape_family"] = shape_family
        support.setdefault("base_shape_family", shape_family)
        support.setdefault("known_object_status", "known_object")
        support.setdefault("deviation_from_consistency", False)
        support.setdefault("special_object_category", None)
        support.setdefault("unknown_attachment_marker", None)
        support.setdefault("support_relation_category", "consistent_support_object")
        key = f"{support.get('support_kind')}:{shape_family}"
        support["support_consistency_group_id"] = key
        groups.setdefault(key, []).append(support)

    for key, rows in groups.items():
        raw_counts: dict[int, int] = {}
        for row in rows:
            raw_value = int(row.get("source_raw_pixel_value"))
            raw_counts[raw_value] = raw_counts.get(raw_value, 0) + 1
        dominant_raw_value = max(raw_counts, key=lambda value: (raw_counts[value], -value)) if raw_counts else None
        clear_repetition = len(rows) >= 3 and dominant_raw_value is not None and raw_counts[dominant_raw_value] >= 2
        for row in rows:
            row["support_consistency_group_size"] = len(rows)
            row["dominant_raw_pixel_value_for_consistency_group"] = dominant_raw_value
            row["consistency_group_raw_pixel_counts"] = {str(raw): count for raw, count in sorted(raw_counts.items())}
            if row.get("special_object_category") == "known_y_unknown_attached":
                continue
            if clear_repetition and int(row.get("source_raw_pixel_value")) != int(dominant_raw_value):
                row["known_object_status"] = "known_object_with_unknown_attachment"
                row["deviation_from_consistency"] = True
                row["special_object_category"] = "known_y_unknown_attached"
                row["unknown_attachment_marker"] = "Y"
                row["base_shape_family"] = row.get("support_shape_family")
                row["unknown_attachment_basis"] = "raw_color_deviates_from_repeated_support_shape_pattern"
                row["support_relation_category"] = "deviation_support_object"
                row["deviation_basis"] = {
                    "schema": "codexlab.support_consistency_deviation.v1",
                    "status": "hypothesis",
                    "consistency_group_id": key,
                    "basis": "raw_color_deviates_from_repeated_support_shape_pattern",
                    "dominant_raw_pixel_value": dominant_raw_value,
                    "observed_raw_pixel_value": int(row.get("source_raw_pixel_value")),
                    "group_size": len(rows),
                    "admission_status": "not_admitted",
                    "control_truth_admitted": False,
                }
                reasons = row.get("reasons")
                if isinstance(reasons, list):
                    reasons.append("raw_color_deviates_from_repeated_support_pattern")
            elif not clear_repetition:
                row["support_relation_category"] = "support_object_consistency_unknown"
    return supports


def bbox_from_component(component: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in component["bbox_xyxy"])  # type: ignore[return-value]


def bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def component_id_sets(*rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for group in rows:
        for row in group:
            if not isinstance(row, dict):
                continue
            for key in ("source_component_id", "actor_component_id", "orientation_key_component_id", "component_id"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    out.add(value)
            source_component_ids = row.get("source_component_ids")
            if isinstance(source_component_ids, list):
                out.update(str(value) for value in source_component_ids if isinstance(value, str) and value)
    return out


def scene_pattern_component_ids(scene_surfaces: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for row in scene_surfaces.get("scene_pattern_hypotheses", []):
        if not isinstance(row, dict):
            continue
        if row.get("role") != "floor_pattern_hypothesis":
            continue
        component_ids = row.get("component_ids")
        if isinstance(component_ids, list):
            ids.update(str(component_id) for component_id in component_ids if isinstance(component_id, str))
    return ids


def bbox_union_from_components(components: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    boxes = [bbox_from_component(component) for component in components]
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def classify_compound_shape(components: list[dict[str, Any]], box: tuple[int, int, int, int]) -> tuple[str, list[str]]:
    families = [support_shape_family_from_box(bbox_from_component(component)) for component in components]
    reasons = ["nearby_same_color_fragments_grouped"]
    has_horizontal = "horizontal_bar" in families
    has_vertical = "vertical_bar" in families
    if has_horizontal and has_vertical:
        horizontal_boxes = [bbox_from_component(component) for component, family in zip(components, families) if family == "horizontal_bar"]
        vertical_boxes = [bbox_from_component(component) for component, family in zip(components, families) if family == "vertical_bar"]
        for hbox in horizontal_boxes:
            for vbox in vertical_boxes:
                vertical_center_x = float(vbox[0] + vbox[2]) / 2.0
                horizontal_span_contains_vertical = hbox[0] <= vertical_center_x <= hbox[2]
                vertical_gap = bbox_gap(hbox, vbox)
                if horizontal_span_contains_vertical and vertical_gap <= 2:
                    reasons.append("horizontal_and_vertical_stems_form_t_candidate")
                    return "t_shape_candidate", reasons
        reasons.append("horizontal_and_vertical_stems_form_angled_or_corner_candidate")
        return "compound_bar_junction_candidate", reasons
    shape_family = support_shape_family_from_box(box)
    if len(set(families)) == 1 and shape_family in {"horizontal_bar", "vertical_bar"}:
        reasons.append("same_axis_fragments_extend_bar_candidate")
        return f"compound_{shape_family}_candidate", reasons
    return "compound_object_candidate", reasons


def build_compound_object_hypotheses(
    *,
    scene_surfaces: dict[str, Any],
    used_component_ids: set[str],
    frame_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    floor_pattern_ids = scene_pattern_component_ids(scene_surfaces)
    residuals = [
        row
        for row in scene_surfaces.get("residual_component_hypotheses", [])
        if isinstance(row, dict)
        and str(row.get("component_id")) not in used_component_ids
        and str(row.get("component_id")) not in floor_pattern_ids
    ]
    candidates = [
        row
        for row in residuals
        if int(row.get("area_pixels") or 0) <= max(12, int(frame_shape[0] * frame_shape[1] * 0.01))
        and int(row.get("raw_pixel_value")) not in {0, 1, 6, 11}
    ]
    visited: set[str] = set()
    rows: list[dict[str, Any]] = []
    by_id = {str(row.get("component_id")): row for row in candidates}
    for start in candidates:
        start_id = str(start.get("component_id"))
        if start_id in visited:
            continue
        stack = [start]
        group: list[dict[str, Any]] = []
        visited.add(start_id)
        while stack:
            current = stack.pop()
            group.append(current)
            current_box = bbox_from_component(current)
            current_raw = int(current.get("raw_pixel_value"))
            for neighbor_id, neighbor in by_id.items():
                if neighbor_id in visited:
                    continue
                if int(neighbor.get("raw_pixel_value")) != current_raw:
                    continue
                neighbor_box = bbox_from_component(neighbor)
                close_enough = bbox_gap(current_box, neighbor_box) <= 2
                overlaps_axis = (
                    horizontal_overlap_ratio(current_box, neighbor_box) >= 0.25
                    or vertical_overlap_ratio(current_box, neighbor_box) >= 0.25
                )
                if close_enough and overlaps_axis:
                    visited.add(neighbor_id)
                    stack.append(neighbor)
        if len(group) < 2:
            continue
        box = bbox_union_from_components(group)
        bbox_w = box[2] - box[0] + 1
        bbox_h = box[3] - box[1] + 1
        if bbox_w > max(12, frame_shape[1] // 3) or bbox_h > max(12, frame_shape[0] // 3):
            continue
        component_ids = sorted(str(component.get("component_id")) for component in group)
        area = sum(int(component.get("area_pixels") or 0) for component in group)
        shape_kind, reasons = classify_compound_shape(group, box)
        raw_value = int(group[0].get("raw_pixel_value"))
        compound_id = f"compound_{shape_kind}_v{raw_value:02d}_{'_'.join(component_ids)}"
        rows.append(
            {
                "schema": COMPOUND_OBJECT_HYPOTHESIS_SCHEMA,
                "status": "hypothesis",
                "compound_object_id": compound_id,
                "component_id": compound_id,
                "source_component_ids": component_ids,
                "source_component_count": len(component_ids),
                "source_raw_pixel_value": raw_value,
                "raw_pixel_value": raw_value,
                "shape_kind": shape_kind,
                "source_component_role": "compound_object_hypothesis",
                "bbox_xyxy": [int(value) for value in box],
                "center_xy": [float(value) for value in bbox_center(box)],
                "area_pixels": int(area),
                "reasons": reasons,
                "evidence_status": "hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "provenance": "derived_from_same_color_residual_component_grouping",
            }
        )
    rows.sort(key=lambda row: (0 if row.get("shape_kind") == "t_shape_candidate" else 1, -int(row["area_pixels"]), row["compound_object_id"]))
    return rows


def score_support_candidate(
    component: dict[str, Any],
    *,
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    frame_shape: tuple[int, int],
) -> tuple[float, list[str]]:
    x0, y0, x1, y1 = bbox_from_component(component)
    width = max(1, int(frame_shape[1]))
    height = max(1, int(frame_shape[0]))
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    area = int(component.get("area_pixels") or 0)
    raw_value = int(component.get("raw_pixel_value"))
    center = bbox_center((x0, y0, x1, y1))
    horizontal_bar = min(1.0, float(bbox_w) / max(1.0, float(width) * 0.30)) * (1.0 - min(1.0, float(bbox_h) / max(1.0, float(height) * 0.20)))
    compact_object = min(1.0, float(area) / 96.0)
    action_color_prior = 1.0 if raw_value in {8, 9, 15} else 0.45 if raw_value in {6, 10, 11, 14} else 0.2
    actor_centers = [
        bbox_center(tuple(int(v) for v in actor["actor_bbox_xyxy"]))
        for actor in actor_affordances
        if isinstance(actor, dict) and isinstance(actor.get("actor_bbox_xyxy"), list)
    ]
    target_centers = [
        bbox_center(tuple(int(v) for v in target["target_square_xyxy"]))
        for target in target_affordances
        if isinstance(target, dict) and isinstance(target.get("target_square_xyxy"), list)
    ]
    diag = max(1.0, float(np.hypot(width, height)))
    actor_proximity = 0.0
    if actor_centers:
        actor_proximity = 1.0 - min(1.0, min(float(np.hypot(center[0] - ax, center[1] - ay)) for ax, ay in actor_centers) / diag)
    target_proximity = 0.0
    if target_centers:
        target_proximity = 1.0 - min(1.0, min(float(np.hypot(center[0] - tx, center[1] - ty)) for tx, ty in target_centers) / diag)
    score = (
        0.24 * horizontal_bar
        + 0.22 * compact_object
        + 0.20 * action_color_prior
        + 0.18 * actor_proximity
        + 0.16 * target_proximity
    )
    reasons = []
    if horizontal_bar >= 0.45:
        reasons.append("bar_like_component")
    if action_color_prior >= 0.9:
        reasons.append("action_color_prior")
    if actor_proximity >= 0.45:
        reasons.append("near_actor_region")
    if target_proximity >= 0.45:
        reasons.append("near_goal_region")
    if compact_object >= 0.4:
        reasons.append("sufficient_component_area")
    if not reasons:
        reasons.append("low_confidence_residual_support_candidate")
    return round(max(0.0, min(1.0, score)), 6), reasons


def score_compound_support_candidate(
    component: dict[str, Any],
    *,
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    frame_shape: tuple[int, int],
) -> tuple[float, list[str]]:
    score, reasons = score_support_candidate(
        component,
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        frame_shape=frame_shape,
    )
    shape_kind = str(component.get("shape_kind") or "")
    if shape_kind == "t_shape_candidate":
        score += 0.28
        reasons.append("compound_t_shape_priority")
    elif shape_kind in {"compound_bar_junction_candidate", "compound_horizontal_bar_candidate", "compound_vertical_bar_candidate"}:
        score += 0.18
        reasons.append("compound_shape_priority")
    else:
        score += 0.10
        reasons.append("compound_object_priority")
    return round(max(0.0, min(1.0, score)), 6), reasons


def build_support_action_candidates(
    *,
    scene_surfaces: dict[str, Any],
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    existing_support_affordances: list[dict[str, Any]],
    frame_shape: tuple[int, int],
    compound_object_hypotheses: list[dict[str, Any]] | None = None,
    max_candidates: int = 8,
) -> list[dict[str, Any]]:
    used_component_ids = component_id_sets(actor_affordances, target_affordances, existing_support_affordances)
    floor_pattern_ids = scene_pattern_component_ids(scene_surfaces)
    residuals = [
        row
        for row in scene_surfaces.get("residual_component_hypotheses", [])
        if isinstance(row, dict) and str(row.get("component_id")) not in used_component_ids
    ]
    compound_objects = compound_object_hypotheses
    if compound_objects is None:
        compound_objects = build_compound_object_hypotheses(
            scene_surfaces=scene_surfaces,
            used_component_ids=used_component_ids,
            frame_shape=frame_shape,
        )
    scored: list[tuple[float, dict[str, Any], list[str]]] = []
    compound_component_ids: set[str] = set()
    for compound in compound_objects:
        score, reasons = score_compound_support_candidate(
            compound,
            actor_affordances=actor_affordances,
            target_affordances=target_affordances,
            frame_shape=frame_shape,
        )
        if score < 0.35:
            continue
        scored.append((score, compound, reasons))
        source_component_ids = compound.get("source_component_ids")
        if isinstance(source_component_ids, list):
            compound_component_ids.update(str(component_id) for component_id in source_component_ids)
    for component in residuals:
        component_id = str(component.get("component_id"))
        if component_id in floor_pattern_ids or component_id in compound_component_ids:
            continue
        raw_value = int(component.get("raw_pixel_value"))
        if raw_value in {0, 1, 4, 6, 11}:
            continue
        score, reasons = score_support_candidate(
            component,
            actor_affordances=actor_affordances,
            target_affordances=target_affordances,
            frame_shape=frame_shape,
        )
        if score < 0.35:
            continue
        scored.append((score, component, reasons))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("component_id"))))
    rows: list[dict[str, Any]] = []
    selected_components = [component for _, component, _ in scored[:max_candidates]]
    for score, component, reasons in scored[:max_candidates]:
        component_id = str(component.get("component_id"))
        source_component_ids = component.get("source_component_ids")
        source_component_role = str(component.get("source_component_role") or "residual_component_hypothesis")
        support_kind = "compound_support_action_candidate" if source_component_role == "compound_object_hypothesis" else "support_action_candidate"
        support_id_prefix = "compound_support_action_candidate" if source_component_role == "compound_object_hypothesis" else "support_action_candidate"
        support_reasons = list(reasons)
        if component_id in floor_pattern_ids:
            support_reasons.append("scene_floor_pattern_member_demoted")
        rows.append(
            {
                "schema": SUPPORT_ACTION_AFFORDANCE_SCHEMA,
                "status": "hypothesis",
                "support_id": f"{support_id_prefix}_{component_id}",
                "support_kind": support_kind,
                "source": "contra_unified_residual_support_candidate_pass",
                "source_component_role": source_component_role,
                "component_id": component_id,
                "source_component_id": component_id,
                **({"source_component_ids": [str(value) for value in source_component_ids]} if isinstance(source_component_ids, list) else {}),
                **({"compound_shape_kind": str(component.get("shape_kind"))} if source_component_role == "compound_object_hypothesis" else {}),
                "source_raw_pixel_value": int(component.get("raw_pixel_value")),
                "score": score,
                "reasons": support_reasons,
                "support_admission_policy": "heuristic_candidate_review_required",
                "evidence_status": "heuristic_hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "bbox_xyxy": [int(value) for value in component["bbox_xyxy"]],
                "center_xy": [float(value) for value in component["center_xy"]],
                "area_pixels": int(component["area_pixels"]),
                "render_color": "blue",
                "direction_render_color": "blue",
                "rendering_role": "support_action_candidate_review_marker",
                "provenance": "derived_from_same_contra_perception_pipeline",
            }
        )
    selected_component_ids = {str(row.get("source_component_id")) for row in rows}
    for row in rows:
        source_ids = row.get("source_component_ids")
        if isinstance(source_ids, list):
            selected_component_ids.update(str(component_id) for component_id in source_ids)
    selected_boxes = [
        (str(component.get("component_id")), bbox_from_component(component))
        for component in selected_components
    ]
    for component in residuals:
        component_id = str(component.get("component_id"))
        if component_id in selected_component_ids:
            continue
        if component_id in floor_pattern_ids:
            continue
        raw_value = int(component.get("raw_pixel_value"))
        if raw_value != 4:
            continue
        box = bbox_from_component(component)
        if int(component.get("area_pixels") or 0) > 16:
            continue
        left_right_neighbors = [
            neighbor_id
            for neighbor_id, neighbor_box in selected_boxes
            if bbox_gap(box, neighbor_box) <= 1 and vertical_overlap_ratio(box, neighbor_box) >= 0.67
        ]
        up_down_neighbors = [
            neighbor_id
            for neighbor_id, neighbor_box in selected_boxes
            if bbox_gap(box, neighbor_box) <= 1 and horizontal_overlap_ratio(box, neighbor_box) >= 0.67
        ]
        adjacent_support_ids = sorted(set(left_right_neighbors + up_down_neighbors))
        if len(adjacent_support_ids) < 2:
            continue
        rows.append(
            {
                "schema": SUPPORT_ACTION_AFFORDANCE_SCHEMA,
                "status": "hypothesis",
                "support_id": f"support_connector_candidate_{component_id}",
                "support_kind": "support_connector_candidate",
                "source": "contra_unified_residual_support_candidate_pass",
                "source_component_role": "residual_component_hypothesis",
                "component_id": component_id,
                "source_component_id": component_id,
                "source_raw_pixel_value": raw_value,
                "score": 0.5,
                "reasons": [
                    "small_connector_between_support_candidates",
                    "known_object_with_unknown_attachment",
                ],
                "adjacent_support_candidate_component_ids": adjacent_support_ids,
                "known_object_status": "known_object_with_unknown_attachment",
                "deviation_from_consistency": True,
                "special_object_category": "known_y_unknown_attached",
                "unknown_attachment_marker": "Y",
                "base_shape_family": "compact_component",
                "unknown_attachment_basis": "small_connector_between_adjacent_support_candidates",
                "support_relation_category": "support_connector_unknown_attachment",
                "support_admission_policy": "heuristic_candidate_review_required",
                "evidence_status": "heuristic_hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "bbox_xyxy": [int(value) for value in component["bbox_xyxy"]],
                "center_xy": [float(value) for value in component["center_xy"]],
                "area_pixels": int(component["area_pixels"]),
                "render_color": "yellow",
                "direction_render_color": "yellow",
                "rendering_role": "support_connector_candidate_review_marker",
                "provenance": "derived_from_same_contra_perception_pipeline",
            }
        )
    return rows


def build_contra_perception_pipeline(
    *,
    frame: np.ndarray,
    level_index: int,
    source_raw_frame_sha256: str,
    raw_color_inventory: dict[str, Any],
    calibration: dict[str, Any] | None,
    actor_builder: Any,
    target_builder: Any,
) -> dict[str, Any]:
    scene_surfaces, component_label_map, component_lookup = analyze_scene_surface_hypotheses(frame)
    symmetry_hypotheses = build_symmetry_hypotheses(
        frame=frame,
        scene_surfaces=scene_surfaces,
        label_map=component_label_map,
        component_lookup=component_lookup,
    )
    scene_surfaces["symmetry_hypotheses"] = symmetry_hypotheses
    scene_surfaces["symmetry_relationship_group_count"] = symmetry_hypotheses.get("symmetry_relationship_group_count", 0)
    expectations = calibration.get("scene_surface_expectations") if isinstance(calibration, dict) else None
    if not isinstance(expectations, dict):
        expectations = {}
    actor_affordances = actor_builder(
        scene_surfaces=scene_surfaces,
        actor_expectation=expectations.get("actor_region") if isinstance(expectations.get("actor_region"), dict) else None,
    )
    target_affordances = target_builder(
        frame_shape=frame.shape[:2],
        label_map=component_label_map,
        component_lookup=component_lookup,
        scene_surfaces=scene_surfaces,
        bucket_expectation=expectations.get("bucket_or_goal_region") if isinstance(expectations.get("bucket_or_goal_region"), dict) else None,
    )
    manual_support_affordances = build_statecon_support_affordances(
        component_graph=scene_surfaces.get("component_graph", {}),
        support_expectation=expectations.get("state_condition_support_regions")
        if isinstance(expectations.get("state_condition_support_regions"), dict)
        else None,
    )
    compound_object_hypotheses = build_compound_object_hypotheses(
        scene_surfaces=scene_surfaces,
        used_component_ids=component_id_sets(actor_affordances, target_affordances, manual_support_affordances),
        frame_shape=frame.shape[:2],
    )
    candidate_support_affordances = build_support_action_candidates(
        scene_surfaces=scene_surfaces,
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        existing_support_affordances=manual_support_affordances,
        frame_shape=frame.shape[:2],
        compound_object_hypotheses=compound_object_hypotheses,
    )
    scene_surfaces["compound_object_hypotheses"] = compound_object_hypotheses
    scene_surfaces["compound_object_hypothesis_count"] = len(compound_object_hypotheses)
    support_affordances = annotate_support_consistency([*manual_support_affordances, *candidate_support_affordances])
    statecons = build_state_condition_hypotheses(
        frame_shape=[int(v) for v in frame.shape[:2]],
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        support_affordances=support_affordances,
        symmetry_hypotheses=symmetry_hypotheses,
    )
    relationship_ladder = build_contra_relationship_ladder(
        level_index=int(level_index),
        source_raw_frame_sha256=source_raw_frame_sha256,
        raw_color_inventory=raw_color_inventory,
        scene_surfaces=scene_surfaces,
        symmetry_hypotheses=symmetry_hypotheses,
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        support_affordances=support_affordances,
        statecons=statecons,
    )
    return {
        "schema": CONTRA_PERCEPTION_PIPELINE_SCHEMA,
        "status": "hypothesis_pipeline",
        "level_index": int(level_index),
        "source_raw_frame_sha256": source_raw_frame_sha256,
        "source": "unified_contra_perception_pipeline",
        "scene_surface_hypotheses": scene_surfaces,
        "symmetry_hypotheses": symmetry_hypotheses,
        "component_label_map": component_label_map,
        "component_lookup": component_lookup,
        "actor_affordance_hypotheses": actor_affordances,
        "target_affordance_hypotheses": target_affordances,
        "manual_support_affordances": manual_support_affordances,
        "compound_object_hypotheses": compound_object_hypotheses,
        "candidate_support_affordances": candidate_support_affordances,
        "support_action_affordance_hypotheses": support_affordances,
        "state_condition_hypotheses": statecons,
        "contra_relationship_ladder": relationship_ladder,
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
