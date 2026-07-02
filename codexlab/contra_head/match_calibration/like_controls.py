#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
from typing import Any

from .common import admission_flags, calibration_id

MATCH_CALIBRATION_LIKE_CONTROL_PRIOR_SCHEMA = "codexlab.match_calibration.like_control_group_prior.v1"
MATCH_CALIBRATION_LIKE_CONTROL_GROUP_SCHEMA = "codexlab.match_calibration.like_control_group.v1"
LIKE_CONTROL_PRIOR_CONTEXT_STATUS = "static_similarity_control_prior_not_control_truth"

LIKE_CONTROL_TRANSFER_EXAMPLES = [
    {
        "example_id": "platform_shape_family",
        "description": "If one platform can be clicked and moved, visually similar platform shapes should be treated as the same control family until contradicted.",
        "expected_discovery_effect": "probe_one_representative_then_defer_redundant_group_members",
    },
    {
        "example_id": "dpad_direction_family",
        "description": "If right on the D-pad produces a valid move, left, up, and down inherit the same directional-control family with different vector parameters.",
        "expected_discovery_effect": "probe_one_direction_then_rank_untried_directions_as related_not_unknown",
    },
    {
        "example_id": "palette_selection_family",
        "description": "If one color swatch in a palette can be selected, neighboring like-shaped swatches are likely selectable controls in the same family.",
        "expected_discovery_effect": "probe_one_swatch_then_treat_other_swatch_clicks_as lower_novelty",
    },
]


def _component_rows(
    raw_color_component_packet: dict[str, Any] | None,
    scene_surfaces: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if isinstance(raw_color_component_packet, dict):
        for color in raw_color_component_packet.get("colors", []) or []:
            if not isinstance(color, dict):
                continue
            for component in color.get("components", []) or []:
                if isinstance(component, dict) and component.get("component_id"):
                    row = dict(component)
                    row["palette_name"] = color.get("palette_name")
                    row["color_pixel_count"] = int(color.get("pixel_count") or 0)
                    rows[str(component["component_id"])] = row
    graph = scene_surfaces.get("component_graph") if isinstance(scene_surfaces, dict) else None
    for component in graph.get("components", []) if isinstance(graph, dict) else []:
        if not isinstance(component, dict) or not component.get("component_id"):
            continue
        row = rows.setdefault(str(component["component_id"]), {})
        row.update({key: value for key, value in component.items() if value is not None})
    return rows


def _shape_hash(component: dict[str, Any]) -> str:
    rows = component.get("shape_mask_rows")
    if isinstance(rows, list) and rows:
        basis = "\n".join(str(row) for row in rows)
    else:
        box = component.get("bbox_xyxy") if isinstance(component.get("bbox_xyxy"), list) else [0, 0, 0, 0]
        width = max(0, int(box[2]) - int(box[0]) + 1) if len(box) == 4 else 0
        height = max(0, int(box[3]) - int(box[1]) + 1) if len(box) == 4 else 0
        basis = f"{width}x{height}|area={int(component.get('area_pixels') or 0)}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def _bbox_size(component: dict[str, Any]) -> tuple[int, int]:
    box = component.get("bbox_xyxy") if isinstance(component.get("bbox_xyxy"), list) else None
    if not isinstance(box, list) or len(box) != 4:
        return (0, 0)
    return (max(0, int(box[2]) - int(box[0]) + 1), max(0, int(box[3]) - int(box[1]) + 1))


def _center_sort_key(component: dict[str, Any]) -> tuple[float, float]:
    center = component.get("center_xy") if isinstance(component.get("center_xy"), list) else [0.0, 0.0]
    return (float(center[1] or 0.0), float(center[0] or 0.0))


def _center_xy(component: dict[str, Any]) -> list[float] | None:
    center = component.get("center_xy") if isinstance(component.get("center_xy"), list) else None
    if not isinstance(center, list) or len(center) != 2:
        return None
    return [float(center[0]), float(center[1])]


def _distance(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None:
        return None
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _append_unique(values: list[str], value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if text and text not in values:
        values.append(text)


def _component_id_from_row(row: dict[str, Any]) -> str | None:
    for key in ("source_component_id", "component_id", "actor_component_id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def _actor_centers(
    actor_affordances: list[dict[str, Any]] | None,
    components: dict[str, dict[str, Any]],
) -> list[list[float]]:
    centers: list[list[float]] = []
    for row in actor_affordances or []:
        if not isinstance(row, dict):
            continue
        for component_id in (row.get("actor_component_id"), row.get("orientation_key_component_id")):
            if component_id is None:
                continue
            center = _center_xy(components.get(str(component_id), {}))
            if center is not None:
                centers.append(center)
    return centers


def _target_centers(
    target_affordances: list[dict[str, Any]] | None,
    components: dict[str, dict[str, Any]],
) -> list[list[float]]:
    centers: list[list[float]] = []
    for row in target_affordances or []:
        if not isinstance(row, dict):
            continue
        box = row.get("target_bbox_xyxy") if isinstance(row.get("target_bbox_xyxy"), list) else row.get("target_square_xyxy")
        if isinstance(box, list) and len(box) == 4:
            centers.append([float(int(box[0]) + int(box[2])) / 2.0, float(int(box[1]) + int(box[3])) / 2.0])
            continue
        for component_id in (row.get("source_component_id"), row.get("component_id")):
            if component_id is None:
                continue
            center = _center_xy(components.get(str(component_id), {}))
            if center is not None:
                centers.append(center)
    return centers


def _portband_score_row(scores: dict[str, dict[str, Any]], component_id: str) -> dict[str, Any]:
    row = scores.setdefault(
        component_id,
        {
            "portband_relevance_score": 0.0,
            "portband_alignment_ids": [],
            "portband_alignment_count": 0,
            "portband_lineup_matrix_row_ids": [],
            "portband_lineup_matrix_row_count": 0,
            "portband_fleet_candidate_ids": [],
            "portband_best_lineup_confidence_score": 0.0,
            "portband_best_lineup_route_label": None,
            "portband_action_state_tag_ids": [],
            "portband_action_state_correlation_score": 0.0,
            "portband_action_state_positive_signal_count": 0,
            "portband_action_state_primitives": [],
            "portband_action_state_evidence_event_ids": [],
            "portband_action_state_action_invocation_ids": [],
            "portband_action_state_correlated_object_ids": [],
            "portband_relative_state_change_tag_ids": [],
            "portband_relative_state_change_novelty_statuses": [],
            "portband_solve_like_evidence_flags": [],
            "portband_solve_like_evidence_score": 0.0,
            "portband_action_interaction_edge_ids": [],
            "portband_action_interaction_kinds": [],
            "portband_action_interaction_participant_object_ids": [],
            "portband_route_flow_lock_ids": [],
            "portband_route_flow_confidence_score": 0.0,
            "portband_route_flow_positive_signal_count": 0,
            "portband_route_flow_lock_status": "no_locked_route_flow",
            "portband_route_flow_evidence_event_ids": [],
            "portband_route_flow_action_invocation_ids": [],
        },
    )
    for key, default in (
        ("portband_lineup_matrix_row_ids", []),
        ("portband_fleet_candidate_ids", []),
        ("portband_best_lineup_confidence_score", 0.0),
        ("portband_best_lineup_route_label", None),
        ("portband_action_state_tag_ids", []),
        ("portband_action_state_correlation_score", 0.0),
        ("portband_action_state_positive_signal_count", 0),
        ("portband_action_state_primitives", []),
        ("portband_action_state_evidence_event_ids", []),
        ("portband_action_state_action_invocation_ids", []),
        ("portband_action_state_correlated_object_ids", []),
        ("portband_relative_state_change_tag_ids", []),
        ("portband_relative_state_change_novelty_statuses", []),
        ("portband_solve_like_evidence_flags", []),
        ("portband_solve_like_evidence_score", 0.0),
        ("portband_action_interaction_edge_ids", []),
        ("portband_action_interaction_kinds", []),
        ("portband_action_interaction_participant_object_ids", []),
        ("portband_route_flow_lock_ids", []),
        ("portband_route_flow_confidence_score", 0.0),
        ("portband_route_flow_positive_signal_count", 0),
        ("portband_route_flow_lock_status", "no_locked_route_flow"),
        ("portband_route_flow_evidence_event_ids", []),
        ("portband_route_flow_action_invocation_ids", []),
    ):
        row.setdefault(key, list(default) if isinstance(default, list) else default)
    return row


def _portband_scores(statecons: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for statecon in statecons or []:
        if not isinstance(statecon, dict):
            continue
        portband = statecon.get("portband_hypotheses")
        if not isinstance(portband, dict):
            continue
        for alignment in portband.get("alignments", []) or []:
            if not isinstance(alignment, dict):
                continue
            component_ids = {
                str(value)
                for value in (alignment.get("source_component_ids") or []) + (alignment.get("target_component_ids") or [])
                if value is not None
            }
            if not component_ids:
                continue
            error = float(alignment.get("alignment_error_px") or 0.0)
            base_score = max(0.0, min(1.0, 1.0 - (error / 8.0)))
            if alignment.get("exact_alignment") is True:
                base_score = max(base_score, 1.0)
            elif alignment.get("near_alignment") is True:
                base_score = max(base_score, 0.75)
            for component_id in component_ids:
                row = _portband_score_row(scores, component_id)
                row["portband_relevance_score"] = max(float(row["portband_relevance_score"]), round(base_score, 6))
                _append_unique(row["portband_alignment_ids"], alignment.get("portband_id"))
                row["portband_alignment_count"] = len(row["portband_alignment_ids"])
        matrix = portband.get("lineup_matrix") if isinstance(portband.get("lineup_matrix"), dict) else {}
        for matrix_row in matrix.get("rows", []) or []:
            if not isinstance(matrix_row, dict):
                continue
            effective_score = (
                matrix_row.get("effective_lineup_confidence_score")
                if matrix_row.get("effective_lineup_confidence_score") is not None
                else matrix_row.get("action_state_adjusted_confidence_score")
                if matrix_row.get("action_state_adjusted_confidence_score") is not None
                else matrix_row.get("route_flow_adjusted_confidence_score")
                if matrix_row.get("route_flow_adjusted_confidence_score") is not None
                else matrix_row.get("confidence_score")
            )
            score = round(max(0.0, min(1.0, float(effective_score or 0.0))), 6)
            action_state_score = round(max(0.0, min(1.0, float(matrix_row.get("action_state_correlation_score") or 0.0))), 6)
            action_state_positive = matrix_row.get("action_state_positive_signal") == "observed_action_state_correlation"
            route_flow_score = round(max(0.0, min(1.0, float(matrix_row.get("route_flow_confidence_score") or 0.0))), 6)
            route_flow_positive = matrix_row.get("route_flow_positive_signal") == "positive_route_flow_signal"
            for component_id in [str(value) for value in matrix_row.get("source_component_ids", []) if value is not None]:
                row = _portband_score_row(scores, component_id)
                row["portband_relevance_score"] = max(float(row["portband_relevance_score"]), score)
                row["portband_best_lineup_confidence_score"] = max(float(row["portband_best_lineup_confidence_score"]), score)
                if score >= float(row.get("portband_best_lineup_confidence_score") or 0.0):
                    row["portband_best_lineup_route_label"] = matrix_row.get("lineup_route_label")
                _append_unique(row["portband_lineup_matrix_row_ids"], matrix_row.get("lineup_matrix_row_id"))
                if action_state_positive:
                    for tag_id in matrix_row.get("action_state_tag_ids", []) or []:
                        _append_unique(row["portband_action_state_tag_ids"], tag_id)
                    for primitive in matrix_row.get("action_state_primitives", []) or []:
                        _append_unique(row["portband_action_state_primitives"], primitive)
                    for event_id in matrix_row.get("action_state_evidence_event_ids", []) or []:
                        _append_unique(row["portband_action_state_evidence_event_ids"], event_id)
                    for invocation_id in matrix_row.get("action_state_action_invocation_ids", []) or []:
                        _append_unique(row["portband_action_state_action_invocation_ids"], invocation_id)
                    for object_id in matrix_row.get("action_state_correlated_object_ids", []) or []:
                        _append_unique(row["portband_action_state_correlated_object_ids"], object_id)
                    for tag_id in matrix_row.get("relative_state_change_tag_ids", []) or []:
                        _append_unique(row["portband_relative_state_change_tag_ids"], tag_id)
                    for status in matrix_row.get("relative_state_change_novelty_statuses", []) or []:
                        _append_unique(row["portband_relative_state_change_novelty_statuses"], status)
                    for flag in matrix_row.get("solve_like_evidence_flags", []) or []:
                        _append_unique(row["portband_solve_like_evidence_flags"], flag)
                    for edge_id in matrix_row.get("action_interaction_edge_ids", []) or []:
                        _append_unique(row["portband_action_interaction_edge_ids"], edge_id)
                    for kind in matrix_row.get("action_interaction_kinds", []) or []:
                        _append_unique(row["portband_action_interaction_kinds"], kind)
                    for object_id in matrix_row.get("action_interaction_participant_object_ids", []) or []:
                        _append_unique(row["portband_action_interaction_participant_object_ids"], object_id)
                    row["portband_solve_like_evidence_score"] = max(
                        float(row["portband_solve_like_evidence_score"]),
                        float(matrix_row.get("solve_like_evidence_score") or 0.0),
                    )
                    row["portband_action_state_correlation_score"] = max(float(row["portband_action_state_correlation_score"]), action_state_score)
                    row["portband_action_state_positive_signal_count"] = len(row["portband_action_state_tag_ids"])
                if route_flow_positive:
                    _append_unique(row["portband_route_flow_lock_ids"], matrix_row.get("route_flow_lock_id"))
                    for event_id in matrix_row.get("route_flow_evidence_event_ids", []) or []:
                        _append_unique(row["portband_route_flow_evidence_event_ids"], event_id)
                    for invocation_id in matrix_row.get("route_flow_action_invocation_ids", []) or []:
                        _append_unique(row["portband_route_flow_action_invocation_ids"], invocation_id)
                    row["portband_route_flow_confidence_score"] = max(float(row["portband_route_flow_confidence_score"]), route_flow_score)
                    row["portband_route_flow_lock_status"] = "has_locked_positive_route_flow"
                    row["portband_route_flow_positive_signal_count"] = len(row["portband_route_flow_lock_ids"])
                for alignment_id in matrix_row.get("linked_alignment_ids", []) or []:
                    _append_unique(row["portband_alignment_ids"], alignment_id)
                for fleet_id in matrix_row.get("linked_fleet_candidate_ids", []) or []:
                    _append_unique(row["portband_fleet_candidate_ids"], fleet_id)
                row["portband_alignment_count"] = len(row["portband_alignment_ids"])
                row["portband_lineup_matrix_row_count"] = len(row["portband_lineup_matrix_row_ids"])
    return scores


def _deltaband_scores(deltaband_observations: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(deltaband_observations, dict):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in deltaband_observations.get("component_relevance", []) or []:
        if not isinstance(row, dict) or row.get("component_id") is None:
            continue
        component_id = str(row["component_id"])
        rows[component_id] = {
            "deltaband_relevance_score": round(max(0.0, min(1.0, float(row.get("relevance_score") or 0.0))), 6),
            "deltaband_source_ids": [str(value) for value in row.get("source_ids", []) if value is not None],
        }
    return rows


def _shape_variant_score(component: dict[str, Any], representative: dict[str, Any]) -> float:
    comp_w, comp_h = _bbox_size(component)
    rep_w, rep_h = _bbox_size(representative)
    comp_area = float(component.get("area_pixels") or 0.0)
    rep_area = float(representative.get("area_pixels") or 0.0)
    size_delta = (abs(comp_w - rep_w) + abs(comp_h - rep_h)) / float(max(1, rep_w + rep_h))
    area_delta = abs(comp_area - rep_area) / float(max(1.0, rep_area))
    hash_delta = 0.0 if _shape_hash(component) == _shape_hash(representative) else 0.25
    return round(max(0.0, min(1.0, hash_delta + (0.50 * size_delta) + (0.25 * area_delta))), 6)


def _member_priority_rows(
    *,
    component_ids: list[str],
    representative: str,
    components: dict[str, dict[str, Any]],
    actor_centers: list[list[float]],
    target_centers: list[list[float]],
    portband_by_component: dict[str, dict[str, Any]],
    deltaband_by_component: dict[str, dict[str, Any]],
    inferred_control_family: str,
) -> list[dict[str, Any]]:
    representative_component = components.get(representative, {})
    max_actor_distance = 1.0
    max_target_distance = 1.0
    distances: dict[str, float | None] = {}
    target_distances: dict[str, float | None] = {}
    nearest_targets: dict[str, list[float] | None] = {}
    for component_id in component_ids:
        center = _center_xy(components.get(component_id, {}))
        component_distances = [_distance(center, actor_center) for actor_center in actor_centers]
        valid = [float(value) for value in component_distances if value is not None]
        distance = min(valid) if valid else None
        distances[component_id] = distance
        if distance is not None:
            max_actor_distance = max(max_actor_distance, float(distance))
        target_options = [(float(value), target_center) for target_center in target_centers if (value := _distance(center, target_center)) is not None]
        target_options.sort(key=lambda row: row[0])
        target_distance = target_options[0][0] if target_options else None
        target_distances[component_id] = target_distance
        nearest_targets[component_id] = target_options[0][1] if target_options else None
        if target_distance is not None:
            max_target_distance = max(max_target_distance, float(target_distance))
    rows: list[dict[str, Any]] = []
    for component_id in component_ids:
        component = components.get(component_id, {})
        actor_distance = distances.get(component_id)
        target_distance = target_distances.get(component_id)
        actor_proximity = 0.0 if actor_distance is None else max(0.0, min(1.0, 1.0 - (float(actor_distance) / max_actor_distance)))
        objective_proximity = 0.0 if target_distance is None else max(0.0, min(1.0, 1.0 - (float(target_distance) / max_target_distance)))
        shape_variant = _shape_variant_score(component, representative_component)
        portband = portband_by_component.get(component_id, {})
        deltaband = deltaband_by_component.get(component_id, {})
        portband_score = float(portband.get("portband_relevance_score") or 0.0)
        deltaband_score = float(deltaband.get("deltaband_relevance_score") or 0.0)
        representative_bonus = 0.05 if component_id == representative else 0.0
        movement_family = "movable" in inferred_control_family or "direction" in inferred_control_family
        priority_score = min(
            1.0,
            representative_bonus
            + (0.30 * portband_score)
            + (0.25 * deltaband_score)
            + (0.20 * objective_proximity if movement_family else 0.0)
            + (0.18 * actor_proximity)
            + (0.18 * shape_variant),
        )
        click_xy = _center_xy(component)
        click_policy = "component_center"
        target_center = nearest_targets.get(component_id)
        box = component.get("bbox_xyxy") if isinstance(component.get("bbox_xyxy"), list) else None
        if movement_family and click_xy is not None and target_center is not None and isinstance(box, list) and len(box) == 4:
            dx = float(target_center[0]) - float(click_xy[0])
            dy = float(target_center[1]) - float(click_xy[1])
            length = max(1.0, math.hypot(dx, dy))
            width, height = _bbox_size(component)
            step = min(3.0, max(1.0, max(width, height) * 0.35))
            x = max(float(box[0]), min(float(box[2]), float(click_xy[0]) + (dx / length) * step))
            y = max(float(box[1]), min(float(box[3]), float(click_xy[1]) + (dy / length) * step))
            click_xy = [round(x, 6), round(y, 6)]
            click_policy = "inside_component_toward_nearest_objective"
        rows.append(
            {
                "component_id": component_id,
                "member_role": "representative_probe" if component_id == representative else "deferred_like_member_probe_if_needed",
                "priority_score": round(priority_score, 6),
                "priority_basis": "representative_then_portband_deltaband_actor_proximity_shape_variant",
                "shape_variant_score": shape_variant,
                "actor_proximity_score": round(actor_proximity, 6),
                "nearest_actor_distance_px": round(float(actor_distance), 6) if actor_distance is not None else None,
                "objective_proximity_score": round(objective_proximity, 6),
                "nearest_objective_distance_px": round(float(target_distance), 6) if target_distance is not None else None,
                "nearest_objective_center_xy": [round(float(target_center[0]), 6), round(float(target_center[1]), 6)] if target_center is not None else None,
                "recommended_click_xy": click_xy,
                "recommended_click_policy": click_policy,
                "portband_relevance_score": round(portband_score, 6),
                "portband_alignment_ids": portband.get("portband_alignment_ids", []),
                "portband_lineup_matrix_row_ids": portband.get("portband_lineup_matrix_row_ids", []),
                "portband_fleet_candidate_ids": portband.get("portband_fleet_candidate_ids", []),
                "portband_best_lineup_confidence_score": round(float(portband.get("portband_best_lineup_confidence_score") or 0.0), 6),
                "portband_best_lineup_route_label": portband.get("portband_best_lineup_route_label"),
                "portband_action_state_tag_ids": portband.get("portband_action_state_tag_ids", []),
                "portband_action_state_correlation_score": round(float(portband.get("portband_action_state_correlation_score") or 0.0), 6),
                "portband_action_state_positive_signal_count": int(portband.get("portband_action_state_positive_signal_count") or 0),
                "portband_action_state_primitives": portband.get("portband_action_state_primitives", []),
                "portband_action_state_evidence_event_ids": portband.get("portband_action_state_evidence_event_ids", []),
                "portband_action_state_action_invocation_ids": portband.get("portband_action_state_action_invocation_ids", []),
                "portband_action_state_correlated_object_ids": portband.get("portband_action_state_correlated_object_ids", []),
                "portband_relative_state_change_tag_ids": portband.get("portband_relative_state_change_tag_ids", []),
                "portband_relative_state_change_novelty_statuses": portband.get("portband_relative_state_change_novelty_statuses", []),
                "portband_solve_like_evidence_flags": portband.get("portband_solve_like_evidence_flags", []),
                "portband_solve_like_evidence_score": round(float(portband.get("portband_solve_like_evidence_score") or 0.0), 6),
                "portband_action_interaction_edge_ids": portband.get("portband_action_interaction_edge_ids", []),
                "portband_action_interaction_kinds": portband.get("portband_action_interaction_kinds", []),
                "portband_action_interaction_participant_object_ids": portband.get("portband_action_interaction_participant_object_ids", []),
                "portband_route_flow_lock_ids": portband.get("portband_route_flow_lock_ids", []),
                "portband_route_flow_confidence_score": round(float(portband.get("portband_route_flow_confidence_score") or 0.0), 6),
                "portband_route_flow_positive_signal_count": int(portband.get("portband_route_flow_positive_signal_count") or 0),
                "portband_route_flow_lock_status": portband.get("portband_route_flow_lock_status"),
                "portband_route_flow_evidence_event_ids": portband.get("portband_route_flow_evidence_event_ids", []),
                "portband_route_flow_action_invocation_ids": portband.get("portband_route_flow_action_invocation_ids", []),
                "deltaband_relevance_score": round(deltaband_score, 6),
                "deltaband_source_ids": deltaband.get("deltaband_source_ids", []),
                "control_truth_admitted": False,
                "action_choice_admitted": False,
                "gameplay_move_admitted": False,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["priority_score"]),
            -float(row["portband_relevance_score"]),
            -float(row["deltaband_relevance_score"]),
            -float(row["objective_proximity_score"]),
            -float(row["actor_proximity_score"]),
            -float(row["shape_variant_score"]),
            0 if row["member_role"] == "representative_probe" else 1,
            str(row["component_id"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["priority_rank"] = index
    return rows


def _group_row(
    *,
    group_kind: str,
    group_basis: str,
    component_ids: list[str],
    components: dict[str, dict[str, Any]],
    inferred_control_family: str,
    confidence: float,
    actor_centers: list[list[float]],
    target_centers: list[list[float]],
    portband_by_component: dict[str, dict[str, Any]],
    deltaband_by_component: dict[str, dict[str, Any]],
    source_row_ids: list[str] | None = None,
) -> dict[str, Any]:
    baseline_representative = sorted(component_ids, key=lambda component_id: _center_sort_key(components.get(component_id, {})))[0]
    provisional_priority_rows = _member_priority_rows(
        component_ids=component_ids,
        representative=baseline_representative,
        components=components,
        actor_centers=actor_centers,
        target_centers=target_centers,
        portband_by_component=portband_by_component,
        deltaband_by_component=deltaband_by_component,
        inferred_control_family=inferred_control_family,
    )
    representative = str(provisional_priority_rows[0]["component_id"]) if provisional_priority_rows else baseline_representative
    priority_rows = _member_priority_rows(
        component_ids=component_ids,
        representative=representative,
        components=components,
        actor_centers=actor_centers,
        target_centers=target_centers,
        portband_by_component=portband_by_component,
        deltaband_by_component=deltaband_by_component,
        inferred_control_family=inferred_control_family,
    )
    ordered_ids = [str(row["component_id"]) for row in priority_rows]
    return {
        "schema": MATCH_CALIBRATION_LIKE_CONTROL_GROUP_SCHEMA,
        "status": "hypothesis",
        "like_control_group_id": calibration_id("like_control", group_kind, group_basis, component_ids),
        "group_kind": group_kind,
        "group_basis": group_basis,
        "member_component_ids": list(component_ids),
        "member_count": len(component_ids),
        "representative_component_id": representative,
        "deferred_member_component_ids": [component_id for component_id in component_ids if component_id != representative],
        "member_priority_policy": "representative_first_then_portband_deltaband_actor_proximity_objective_direction_shape_variant",
        "movement_click_policy": "movement_like_groups_recommend_click_inside_member_toward_nearest_objective",
        "member_priority_order": ordered_ids,
        "member_priority_rows": priority_rows,
        "one_probe_policy": "probe_representative_once_then_infer_group_until_delta_contradiction",
        "inferred_control_family": inferred_control_family,
        "group_action_inference": "members_most_likely_share_control_mechanism_pending_controlled_boundary",
        "control_similarity_confidence": round(max(0.0, min(0.99, confidence)), 6),
        "source_row_ids": source_row_ids or [],
        "evidence_status": "static_similarity_hypothesis",
        "prior_context_status": LIKE_CONTROL_PRIOR_CONTEXT_STATUS,
        "prediction_status": "not_predictive",
        **admission_flags(),
    }


def build_like_control_group_prior(
    *,
    raw_color_component_packet: dict[str, Any] | None = None,
    scene_surfaces: dict[str, Any] | None = None,
    actor_affordances: list[dict[str, Any]] | None = None,
    target_affordances: list[dict[str, Any]] | None = None,
    support_affordances: list[dict[str, Any]] | None = None,
    statecons: list[dict[str, Any]] | None = None,
    deltaband_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components = _component_rows(raw_color_component_packet, scene_surfaces)
    actor_centers = _actor_centers(actor_affordances, components)
    target_centers = _target_centers(target_affordances, components)
    portband_by_component = _portband_scores(statecons)
    deltaband_by_component = _deltaband_scores(deltaband_observations)
    target_ids = {
        str(row.get("source_component_id") or row.get("component_id"))
        for row in target_affordances or []
        if isinstance(row, dict) and (row.get("source_component_id") or row.get("component_id"))
    }
    actor_ids = {
        str(value)
        for row in actor_affordances or []
        if isinstance(row, dict)
        for value in (row.get("actor_component_id"), row.get("orientation_key_component_id"))
        if value is not None
    }
    floor_id = None
    if isinstance(scene_surfaces, dict) and isinstance(scene_surfaces.get("floor_hypothesis"), dict):
        floor_id = scene_surfaces["floor_hypothesis"].get("component_id")
    excluded_ids = set(target_ids)
    if floor_id is not None:
        excluded_ids.add(str(floor_id))

    groups: list[dict[str, Any]] = []
    support_buckets: dict[str, list[str]] = {}
    support_source_rows: dict[str, list[str]] = {}
    for row in support_affordances or []:
        if not isinstance(row, dict):
            continue
        component_id = _component_id_from_row(row)
        if not component_id or component_id in excluded_ids or component_id not in components:
            continue
        component = components.get(component_id, {})
        width, height = _bbox_size(component)
        family = str(row.get("support_shape_family") or row.get("base_shape_family") or "unknown_support_shape")
        basis = f"support_family={family}|shape={_shape_hash(component)}|size={width}x{height}|area={int(component.get('area_pixels') or 0)}"
        support_buckets.setdefault(basis, []).append(component_id)
        source_id = row.get("support_id") or row.get("support_affordance_id")
        if source_id is not None:
            support_source_rows.setdefault(basis, []).append(str(source_id))
    for basis, member_ids in support_buckets.items():
        unique_ids = sorted(set(member_ids))
        if len(unique_ids) < 2:
            continue
        groups.append(
            _group_row(
                group_kind="support_like_object_control_group",
                group_basis=basis,
                component_ids=unique_ids,
                components=components,
                inferred_control_family="movable_or_clickable_support_object_family",
                confidence=0.76,
                actor_centers=actor_centers,
                target_centers=target_centers,
                portband_by_component=portband_by_component,
                deltaband_by_component=deltaband_by_component,
                source_row_ids=sorted(set(support_source_rows.get(basis, []))),
            )
        )

    shape_buckets: dict[str, list[str]] = {}
    for component_id, component in components.items():
        if component_id in excluded_ids or component_id in actor_ids:
            continue
        area = int(component.get("area_pixels") or 0)
        if area <= 0:
            continue
        width, height = _bbox_size(component)
        if width <= 0 or height <= 0:
            continue
        if width * height > 900:
            continue
        basis = f"shape={_shape_hash(component)}|size={width}x{height}|area={area}"
        shape_buckets.setdefault(basis, []).append(component_id)
    for basis, member_ids in shape_buckets.items():
        unique_ids = sorted(set(member_ids))
        if len(unique_ids) < 2:
            continue
        raw_values = {components[component_id].get("raw_pixel_value") for component_id in unique_ids}
        group_kind = "palette_like_selection_group" if len(raw_values) >= 3 else "shape_like_control_group"
        inferred_family = "palette_or_option_selection_family" if group_kind == "palette_like_selection_group" else "same_shape_control_family"
        groups.append(
            _group_row(
                group_kind=group_kind,
                group_basis=basis,
                component_ids=unique_ids,
                components=components,
                inferred_control_family=inferred_family,
                confidence=0.70 if group_kind == "palette_like_selection_group" else 0.62,
                actor_centers=actor_centers,
                target_centers=target_centers,
                portband_by_component=portband_by_component,
                deltaband_by_component=deltaband_by_component,
            )
        )

    group_kind_priority = {
        "support_like_object_control_group": 0,
        "palette_like_selection_group": 1,
        "shape_like_control_group": 2,
    }
    groups.sort(
        key=lambda row: (
            tuple(str(value) for value in row["member_component_ids"]),
            group_kind_priority.get(str(row["group_kind"]), 9),
            str(row["like_control_group_id"]),
        )
    )
    deduped_groups: list[dict[str, Any]] = []
    seen_member_sets: set[tuple[str, ...]] = set()
    for group in groups:
        key = tuple(str(value) for value in group["member_component_ids"])
        if key in seen_member_sets:
            continue
        seen_member_sets.add(key)
        deduped_groups.append(group)
    groups = sorted(deduped_groups, key=lambda row: (-int(row["member_count"]), str(row["group_kind"]), str(row["like_control_group_id"])))
    component_context: dict[str, dict[str, Any]] = {}
    grouped_ids: set[str] = set()
    for group in groups:
        representative_id = str(group["representative_component_id"])
        group_id = str(group["like_control_group_id"])
        for component_id in group["member_component_ids"]:
            grouped_ids.add(str(component_id))
            row = component_context.setdefault(
                str(component_id),
                {
                    "component_id": str(component_id),
                    "like_control_group_ids": [],
                    "representative_for_group_ids": [],
                    "deferred_by_group_ids": [],
                    "like_control_redundancy_score": 0.0,
                    "nonconformant_curiosity_boost": 0.0,
                },
            )
            _append_unique(row["like_control_group_ids"], group_id)
            if str(component_id) == representative_id:
                _append_unique(row["representative_for_group_ids"], group_id)
            else:
                _append_unique(row["deferred_by_group_ids"], group_id)
                row["like_control_redundancy_score"] = max(float(row["like_control_redundancy_score"]), 0.55)

    action_like_ids = {
        str(component_id)
        for component_id in [
            *(_component_id_from_row(row) for row in support_affordances or [] if isinstance(row, dict)),
            *actor_ids,
        ]
        if component_id and str(component_id) in components and str(component_id) not in excluded_ids
    }
    nonconformant_ids = sorted(action_like_ids - grouped_ids)
    for component_id in nonconformant_ids:
        row = component_context.setdefault(
            component_id,
            {
                "component_id": component_id,
                "like_control_group_ids": [],
                "representative_for_group_ids": [],
                "deferred_by_group_ids": [],
                "like_control_redundancy_score": 0.0,
                "nonconformant_curiosity_boost": 0.0,
            },
        )
        row["nonconformant_curiosity_boost"] = max(float(row["nonconformant_curiosity_boost"]), 0.45)

    return {
        "schema": MATCH_CALIBRATION_LIKE_CONTROL_PRIOR_SCHEMA,
        "status": "ranked" if groups or nonconformant_ids else "no_like_control_groups_detected",
        "prior_context_status": LIKE_CONTROL_PRIOR_CONTEXT_STATUS,
        "goal": "reduce_redundant_discovery_clicks_and_promote_nonconformant_high_signal_objects",
        "one_probe_group_policy": "like_objects_should_be_clicked_once_then_group_action_inferred_until_controlled_delta_disagrees",
        "nonconformant_policy": "action_like_objects_without_a_like_group_receive_higher_curiosity",
        "calibration_examples": LIKE_CONTROL_TRANSFER_EXAMPLES,
        "like_control_groups": groups,
        "like_control_group_count": len(groups),
        "component_like_control_context": sorted(component_context.values(), key=lambda row: str(row["component_id"])),
        "grouped_component_ids": sorted(grouped_ids),
        "nonconformant_high_signal_component_ids": nonconformant_ids,
        "estimated_redundant_discovery_clicks_reduced": sum(max(0, int(group["member_count"]) - 1) for group in groups),
        "evidence_status": "static_similarity_hypothesis",
        "prediction_status": "not_predictive",
        **admission_flags(),
    }
