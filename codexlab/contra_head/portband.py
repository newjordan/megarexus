#!/usr/bin/env python3
"""PortBand static port-alignment hypotheses for GeoBand review."""

from __future__ import annotations

import hashlib
import math
from itertools import combinations
from typing import Any


PORTBAND_PACKET_SCHEMA = "codexlab.geoband_port_alignment_hypotheses.v1"
PORTBAND_ALIGNMENT_SCHEMA = "codexlab.geoband_port_alignment_hypothesis.v1"
PORTBAND_PORT_SCHEMA = "codexlab.geoband_object_port_hypothesis.v1"
PORTBAND_SURFACE_FEEDBACK_SCHEMA = "codexlab.geoband_portband_surface_feedback.v1"
PORTBAND_FLEET_SCHEMA = "codexlab.geoband_portband_fleet.v1"
PORTBAND_FLEET_CANDIDATE_SCHEMA = "codexlab.geoband_portband_fleet_candidate.v1"
PORTBAND_LINEUP_MATRIX_SCHEMA = "codexlab.geoband_portband_lineup_matrix.v1"
PORTBAND_LINEUP_MATRIX_ROW_SCHEMA = "codexlab.geoband_portband_lineup_matrix_row.v1"
PORTBAND_LINEUP_CONFIDENCE_MODEL = "static_portband_lineup_score_not_probability"
PORTBAND_EDGE_LOCK_LEDGER_SCHEMA = "codexlab.geoband_portband_edge_lock_ledger.v1"
PORTBAND_EDGE_LOCK_CONFIGURATION_SCHEMA = "codexlab.geoband_portband_edge_lock_configuration.v1"
PORTBAND_EDGE_LOCK_CONFIDENCE_MODEL = "static_shape_edge_lock_configuration_score_not_probability"
PORTBAND_EDGE_LOCK_POLICY = "copy_compare_and_rank_static_edge_locks_as_candidate_shape_configurations_not_solution_truth"
PORTBAND_EDGE_LOCK_POTENTIAL_STATUS = "candidate_shape_configuration_potential_not_solve_truth"
PORTBAND_EDGE_LOCK_BOUNDARY = (
    "ranked static shape/edge lock configurations are candidate potential signals only; "
    "not solve truth, action choice, control truth, goal truth, or prediction"
)
CONTRA_ACTION_STATE_TAG_SCHEMA = "codexlab.contra_action_state_tag.v1"
CONTRA_ACTION_STATE_TAG_SET_SCHEMA = "codexlab.contra_action_state_tag_set.v1"
CONTRA_ACTION_INTERACTION_EDGE_SCHEMA = "codexlab.contra_action_interaction_edge.v1"
CONTRA_ACTION_INTERACTION_GRAPH_SCHEMA = "codexlab.contra_action_interaction_graph.v1"
CONTRA_ACTION_INTERACTION_MATRIX_SCHEMA = "codexlab.contra_action_interaction_matrix.v1"
CONTRA_ACTION_INTERACTION_MATRIX_ROW_SCHEMA = "codexlab.contra_action_interaction_matrix_row.v1"
CONTRA_ACTION_INTERACTION_CLASSIFICATION_SCHEMA = "codexlab.contra_action_interaction_classification.v1"
CONTRA_SOLVE_LIKE_CANDIDATE_LEDGER_SCHEMA = "codexlab.contra_solve_like_candidate_ledger.v1"
CONTRA_SOLVE_LIKE_CANDIDATE_LEDGER_ROW_SCHEMA = "codexlab.contra_solve_like_candidate_ledger_row.v1"
CONTRA_OBJECT_CORRESPONDENCE_SCHEMA = "codexlab.contra_object_correspondence.v1"
CONTRA_RELATIVE_MATH_PROFILE_SCHEMA = "codexlab.contra_relative_math_profile.v1"
CONTRA_TEMPORAL_TRAJECTORY_PROFILE_SCHEMA = "codexlab.cross_game_action_temporal_trajectory_profile.v1"
CONTRA_RESOURCE_GAUGE_PROFILE_SCHEMA = "codexlab.contra_resource_gauge_profile.v1"
CONTRA_MOTION_ENERGY_CONSERVATION_SCHEMA = "codexlab.contra_motion_energy_conservation_assessment.v1"
CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL = "observed_action_result_object_overlap_score_not_probability"
CONTRA_ACTION_STATE_TAG_POLICY = "tag_controlled_action_observations_to_candidate_objects_lineups_and_relative_state_changes"
CONTRA_ACTION_INTERACTION_GRAPH_POLICY = "compile_action_state_tags_into_object_interaction_edges"
CONTRA_ACTION_INTERACTION_MATRIX_POLICY = "rank_candidate_object_interactions_from_action_state_tags_and_relative_profiles"
CONTRA_ACTION_STATE_POSITIVE_SIGNAL = "observed_action_state_correlation"
CONTRA_RELATIVE_STATE_TAG_POLICY = "tag_controlled_relative_state_changes_from_deltaband_numeric_member_and_ray_deltas"
CONTRA_SOLVE_LIKE_EVIDENCE_STATUS = "candidate_interaction_clue_not_solve_truth"
CONTRA_SOLVE_LIKE_EVIDENCE_FLAG = "candidate_action_interaction_clue"
CONTRA_ACTION_STATE_TAG_BOUNDARY = (
    "controlled action-result overlap can tag candidate object/action/state correlations; "
    "it is not control truth, goal truth, action choice, solve evidence, or prediction"
)
PORTBAND_ROUTE_FLOW_LOCK_SCHEMA = "codexlab.geoband_portband_route_flow_lock.v1"
PORTBAND_ROUTE_FLOW_LOCK_CONFIDENCE_MODEL = "single_controlled_paintflow_overlap_score_not_probability"
PORTBAND_ROUTE_FLOW_LOCK_POLICY = "single_controlled_activate_flow_surface_event_can_lock_route_flow_positive_signal"
PORTBAND_ROUTE_FLOW_LOCK_STATUS = "locked_positive_after_single_controlled_paintflow"
PORTBAND_ROUTE_FLOW_NO_LOCK_STATUS = "not_locked_no_controlled_paintflow_match"
PORTBAND_ROUTE_FLOW_POSITIVE_SIGNAL = "positive_route_flow_signal"
PORTBAND_ROUTE_FLOW_LOCK_BOUNDARY = (
    "controlled activate_flow surface-result overlap can emit a PortBand route-flow positive signal; "
    "it is not control truth, goal truth, action choice, solve evidence, or prediction"
)

TRUTH_FLAGS = {
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _center(box: list[int]) -> list[float]:
    return [round(float(box[0] + box[2]) / 2.0, 6), round(float(box[1] + box[3]) / 2.0, 6)]


def _port_row(
    *,
    statecon_id: str,
    object_id: str,
    object_role: str,
    component_ids: list[str],
    side: str,
    box: list[int],
    port_kind: str = "bbox_side_port",
) -> dict[str, Any]:
    x0, y0, x1, y1 = [int(value) for value in box]
    edge_cells: list[list[int]]
    if side == "left":
        axis = "vertical_projection"
        center = [float(x0), float(y0 + y1) / 2.0]
        coord = float(x0)
        span = [int(y0), int(y1)]
        segment = [[int(x0), int(y0)], [int(x0), int(y1)]]
        edge_cells = [[int(x0), int(y)] for y in range(y0, y1 + 1)]
    elif side == "right":
        axis = "vertical_projection"
        center = [float(x1), float(y0 + y1) / 2.0]
        coord = float(x1)
        span = [int(y0), int(y1)]
        segment = [[int(x1), int(y0)], [int(x1), int(y1)]]
        edge_cells = [[int(x1), int(y)] for y in range(y0, y1 + 1)]
    elif side == "top":
        axis = "horizontal_projection"
        center = [float(x0 + x1) / 2.0, float(y0)]
        coord = float(y0)
        span = [int(x0), int(x1)]
        segment = [[int(x0), int(y0)], [int(x1), int(y0)]]
        edge_cells = [[int(x), int(y0)] for x in range(x0, x1 + 1)]
    elif side == "bottom":
        axis = "horizontal_projection"
        center = [float(x0 + x1) / 2.0, float(y1)]
        coord = float(y1)
        span = [int(x0), int(x1)]
        segment = [[int(x0), int(y1)], [int(x1), int(y1)]]
        edge_cells = [[int(x), int(y1)] for x in range(x0, x1 + 1)]
    elif side == "open_left_wall_inner_edge":
        axis = "vertical_projection"
        coord = float(x0 - 1)
        center = [coord, float(y0 + y1) / 2.0]
        span = [int(y0), int(y1)]
        segment = [[int(coord), int(y0)], [int(coord), int(y1)]]
        edge_cells = [[int(coord), int(y)] for y in range(y0, y1 + 1)]
    elif side == "open_right_wall_inner_edge":
        axis = "vertical_projection"
        coord = float(x1 + 1)
        center = [coord, float(y0 + y1) / 2.0]
        span = [int(y0), int(y1)]
        segment = [[int(coord), int(y0)], [int(coord), int(y1)]]
        edge_cells = [[int(coord), int(y)] for y in range(y0, y1 + 1)]
    elif side == "open_top_wall_inner_edge":
        axis = "horizontal_projection"
        coord = float(y0 - 1)
        center = [float(x0 + x1) / 2.0, coord]
        span = [int(x0), int(x1)]
        segment = [[int(x0), int(coord)], [int(x1), int(coord)]]
        edge_cells = [[int(x), int(coord)] for x in range(x0, x1 + 1)]
    elif side == "open_bottom_wall_inner_edge":
        axis = "horizontal_projection"
        coord = float(y1 + 1)
        center = [float(x0 + x1) / 2.0, coord]
        span = [int(x0), int(x1)]
        segment = [[int(x0), int(coord)], [int(x1), int(coord)]]
        edge_cells = [[int(x), int(coord)] for x in range(x0, x1 + 1)]
    else:
        raise ValueError(f"unknown PortBand side: {side}")
    xs = [cell[0] for cell in edge_cells]
    ys = [cell[1] for cell in edge_cells]
    edge_bbox = [min(xs), min(ys), max(xs), max(ys)] if edge_cells else [int(x0), int(y0), int(x1), int(y1)]
    port_id = _stable_id("port", statecon_id, object_id, side, box)
    return {
        "schema": PORTBAND_PORT_SCHEMA,
        "status": "hypothesis",
        "port_id": port_id,
        "statecon_id": statecon_id,
        "object_id": object_id,
        "object_role": object_role,
        "component_ids": component_ids,
        "port_kind": port_kind,
        "side": side,
        "axis": axis,
        "center_xy": [round(float(center[0]), 6), round(float(center[1]), 6)],
        "axis_coordinate_px": round(float(coord), 6),
        "span_px": span,
        "segment_xy": segment,
        "edge_cells_xy": edge_cells,
        "edge_cell_count": len(edge_cells),
        "edge_bbox_xyxy": edge_bbox,
        "port_bbox_xyxy": edge_bbox,
        "reference_region_bbox_xyxy": [int(value) for value in box],
        "edge_reference_kind": "object_edge_cell_set",
        "bbox_xyxy": [int(value) for value in box],
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
        "provenance": "derived_from_object_bbox_side_geometry",
        "rendering_role": "portband_port_review_marker",
    }


def _object_ports(
    *,
    statecon_id: str,
    object_id: str,
    object_role: str,
    component_ids: list[str],
    box: list[int],
    open_box: list[int] | None = None,
    open_direction: str | None = None,
) -> list[dict[str, Any]]:
    if len(box) != 4:
        return []
    port_specs = [(side, box, "bbox_side_port") for side in ("left", "right", "top", "bottom")]
    if object_role == "target":
        port_specs = []
        if isinstance(open_box, list) and len(open_box) == 4:
            if open_direction in {"up", "down", None}:
                port_specs.extend(
                    [
                        ("open_left_wall_inner_edge", open_box, "target_open_wall_edge_port"),
                        ("open_right_wall_inner_edge", open_box, "target_open_wall_edge_port"),
                    ]
                )
            if open_direction in {"left", "right", None}:
                port_specs.extend(
                    [
                        ("open_top_wall_inner_edge", open_box, "target_open_wall_edge_port"),
                        ("open_bottom_wall_inner_edge", open_box, "target_open_wall_edge_port"),
                    ]
                )
    return [
        _port_row(
            statecon_id=statecon_id,
            object_id=object_id,
            object_role=object_role,
            component_ids=component_ids,
            side=side,
            box=[int(value) for value in port_box],
            port_kind=port_kind,
        )
        for side, port_box, port_kind in port_specs
    ]


def _span_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_span = left.get("span_px") if isinstance(left.get("span_px"), list) else [0, 0]
    right_span = right.get("span_px") if isinstance(right.get("span_px"), list) else [0, 0]
    lo = max(float(left_span[0]), float(right_span[0]))
    hi = min(float(left_span[1]), float(right_span[1]))
    overlap = max(0.0, hi - lo + 1.0)
    denom = max(1.0, min(float(left_span[1]) - float(left_span[0]) + 1.0, float(right_span[1]) - float(right_span[0]) + 1.0))
    return round(max(0.0, min(1.0, overlap / denom)), 6)


def _alignment_row(
    *,
    statecon_id: str,
    alignment_kind: str,
    axis: str,
    source_ports: list[dict[str, Any]],
    target_ports: list[dict[str, Any]],
    error_px: float,
    transform_xy: list[float],
    match_basis: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_ids = [str(port.get("port_id")) for port in source_ports]
    target_ids = [str(port.get("port_id")) for port in target_ports]
    exact = float(error_px) <= 0.0
    near = float(error_px) <= 2.0
    source_bboxes = []
    seen_source_bboxes: set[tuple[int, int, int, int]] = set()
    for port in source_ports:
        box = port.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        key = tuple(int(value) for value in box)
        if key in seen_source_bboxes:
            continue
        seen_source_bboxes.add(key)
        source_bboxes.append([int(value) for value in box])
    target_bboxes = [port.get("bbox_xyxy") for port in target_ports if isinstance(port.get("bbox_xyxy"), list)]
    imagined_bboxes = []
    dx = float(transform_xy[0])
    dy = float(transform_xy[1])
    for box in source_bboxes:
        if not isinstance(box, list) or len(box) != 4:
            continue
        imagined_bboxes.append(
            [
                int(round(float(box[0]) + dx)),
                int(round(float(box[1]) + dy)),
                int(round(float(box[2]) + dx)),
                int(round(float(box[3]) + dy)),
            ]
        )
    row = {
        "schema": PORTBAND_ALIGNMENT_SCHEMA,
        "status": "hypothesis",
        "portband_id": _stable_id("portband", statecon_id, alignment_kind, axis, source_ids, target_ids, error_px),
        "statecon_id": statecon_id,
        "alignment_kind": alignment_kind,
        "axis": axis,
        "source_port_ids": source_ids,
        "target_port_ids": target_ids,
        "source_object_ids": sorted({str(port.get("object_id")) for port in source_ports}),
        "target_object_ids": sorted({str(port.get("object_id")) for port in target_ports}),
        "source_component_ids": sorted({str(value) for port in source_ports for value in port.get("component_ids", [])}),
        "target_component_ids": sorted({str(value) for port in target_ports for value in port.get("component_ids", [])}),
        "source_port_centers_xy": [port.get("center_xy") for port in source_ports],
        "target_port_centers_xy": [port.get("center_xy") for port in target_ports],
        "alignment_error_px": round(float(error_px), 6),
        "exact_alignment": exact,
        "near_alignment": near,
        "match_status": "exact_alignment" if exact else ("near_alignment" if near else "residual_imagined_alignment"),
        "projection_mode": "imagination_forward_copy_transform",
        "transform_hypothesis_xy": [round(float(transform_xy[0]), 6), round(float(transform_xy[1]), 6)],
        "source_bboxes_xyxy": source_bboxes,
        "target_bboxes_xyxy": target_bboxes,
        "imagined_source_bboxes_xyxy": imagined_bboxes,
        "imagined_copy_status": "copy_transform_preview_not_environment_state",
        "match_basis": match_basis,
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
        "provenance": "derived_from_geoband_port_projection_geometry",
        "rendering_role": "portband_alignment_review_marker",
    }
    if extra:
        row.update(extra)
    return row


def _port_by_side(ports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(port.get("side")): port for port in ports if isinstance(port, dict)}


def _axis_coord(port: dict[str, Any]) -> float:
    return float(port.get("axis_coordinate_px") or 0.0)


def _union_bbox(boxes: list[list[int]]) -> list[int]:
    return [
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    ]


def _bbox_area(box: list[int]) -> int:
    if len(box) != 4:
        return 0
    return max(0, int(box[2]) - int(box[0]) + 1) * max(0, int(box[3]) - int(box[1]) + 1)


def _bbox_overlap_ratio(left: list[int], right: list[int]) -> float:
    if len(left) != 4 or len(right) != 4:
        return 0.0
    lx0, ly0, lx1, ly1 = [int(value) for value in left]
    rx0, ry0, rx1, ry1 = [int(value) for value in right]
    ix0, iy0 = max(lx0, rx0), max(ly0, ry0)
    ix1, iy1 = min(lx1, rx1), min(ly1, ry1)
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    intersection = _bbox_area([ix0, iy0, ix1, iy1])
    denom = max(1, min(_bbox_area(left), _bbox_area(right)))
    return round(max(0.0, min(1.0, float(intersection) / float(denom))), 6)


def _alignment_static_score(alignment: dict[str, Any]) -> float:
    error = float(alignment.get("alignment_error_px") or 0.0)
    score = max(0.0, min(1.0, 1.0 - (error / 8.0)))
    if alignment.get("exact_alignment") is True:
        score = max(score, 1.0)
    elif alignment.get("near_alignment") is True:
        score = max(score, 0.75)
    if alignment.get("alignment_kind") == "centered_dual_port_projection":
        score = min(1.0, score + 0.10)
    span = alignment.get("span_overlap_ratio")
    if alignment.get("alignment_kind") == "single_port_projection" and float(span or 0.0) <= 0.0:
        score = min(score, 0.22)
    return round(score, 6)


def _as_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [int(v) for v in value]
    except (TypeError, ValueError):
        return None


def _shift_bbox(box: list[int] | None, transform_xy: list[Any] | None) -> list[int] | None:
    if box is None or not isinstance(transform_xy, list) or len(transform_xy) != 2:
        return None
    dx = float(transform_xy[0])
    dy = float(transform_xy[1])
    return [
        int(round(float(box[0]) + dx)),
        int(round(float(box[1]) + dy)),
        int(round(float(box[2]) + dx)),
        int(round(float(box[3]) + dy)),
    ]


def _bbox_shape_terms(boxes: list[Any]) -> dict[str, Any]:
    clean_boxes: list[list[int]] = []
    for value in boxes:
        box = _as_bbox(value)
        if box is not None:
            clean_boxes.append(box)
    if not clean_boxes:
        return {
            "bbox_count": 0,
            "union_bbox_xyxy": None,
            "union_width_px": 0,
            "union_height_px": 0,
            "union_area_px": 0,
            "bbox_area_sum_px": 0,
            "bbox_fill_ratio": 0.0,
            "aspect_ratio": 0.0,
        }
    union_box = _union_bbox(clean_boxes)
    union_area = _bbox_area(union_box)
    area_sum = sum(_bbox_area(box) for box in clean_boxes)
    width = max(0, int(union_box[2]) - int(union_box[0]) + 1)
    height = max(0, int(union_box[3]) - int(union_box[1]) + 1)
    return {
        "bbox_count": len(clean_boxes),
        "union_bbox_xyxy": union_box,
        "union_width_px": width,
        "union_height_px": height,
        "union_area_px": union_area,
        "bbox_area_sum_px": area_sum,
        "bbox_fill_ratio": round(max(0.0, min(1.0, float(area_sum) / float(max(1, union_area)))), 6),
        "aspect_ratio": round(float(width) / float(max(1, height)), 6),
    }


def _shape_fit_from_terms(source_terms: dict[str, Any], target_terms: dict[str, Any]) -> float:
    if not source_terms or not target_terms:
        return 0.0
    sw = float(source_terms.get("union_width_px") or 0.0)
    sh = float(source_terms.get("union_height_px") or 0.0)
    sa = float(source_terms.get("union_area_px") or 0.0)
    tw = float(target_terms.get("union_width_px") or 0.0)
    th = float(target_terms.get("union_height_px") or 0.0)
    ta = float(target_terms.get("union_area_px") or 0.0)
    if min(sw, sh, sa, tw, th, ta) <= 0.0:
        return 0.0
    width_fit = min(sw, tw) / max(sw, tw)
    height_fit = min(sh, th) / max(sh, th)
    area_fit = min(sa, ta) / max(sa, ta)
    return round(max(0.0, min(1.0, (0.38 * width_fit) + (0.38 * height_fit) + (0.24 * area_fit))), 6)


def _axis_preservation_score(axis: str, transform_xy: list[Any] | None) -> float:
    if not isinstance(transform_xy, list) or len(transform_xy) != 2:
        return 0.0
    dx = abs(float(transform_xy[0]))
    dy = abs(float(transform_xy[1]))
    if axis == "vertical_projection":
        denom = max(1.0, dx + dy)
        return round(max(0.0, min(1.0, dx / denom)), 6)
    if axis == "horizontal_projection":
        denom = max(1.0, dx + dy)
        return round(max(0.0, min(1.0, dy / denom)), 6)
    return 0.5


def _edge_lock_confluence_score(
    *,
    source_object_ids: list[str],
    target_object_ids: list[str],
    source_component_ids: list[str],
    target_port_ids: list[str],
    combo_size: int | None = None,
) -> float:
    source_count = max(len(source_object_ids), int(combo_size or 0))
    target_count = len(target_object_ids)
    component_bonus = min(0.22, len(source_component_ids) * 0.035)
    port_bonus = min(0.20, len(target_port_ids) * 0.05)
    object_score = min(1.0, (0.18 * source_count) + (0.18 * target_count))
    return round(max(0.0, min(1.0, 0.22 + object_score + component_bonus + port_bonus)), 6)


def _edge_lock_quality_score(
    *,
    static_score: float,
    shape_fit_score: float,
    span_overlap_score: float,
    confluence_score: float,
    boundary_strength: float,
    axis_preservation_score: float,
    contradiction_penalty: float,
) -> float:
    score = (
        (0.30 * max(0.0, min(1.0, float(static_score))))
        + (0.21 * max(0.0, min(1.0, float(shape_fit_score))))
        + (0.16 * max(0.0, min(1.0, float(span_overlap_score))))
        + (0.14 * max(0.0, min(1.0, float(confluence_score))))
        + (0.10 * max(0.0, min(1.0, float(boundary_strength))))
        + (0.09 * max(0.0, min(1.0, float(axis_preservation_score))))
        - max(0.0, min(1.0, float(contradiction_penalty)))
    )
    return round(max(0.0, min(1.0, score)), 6)


def _edge_lock_signature_terms(
    *,
    lock_kind: str,
    axis: str,
    source_object_ids: list[str],
    target_object_ids: list[str],
    target_port_ids: list[str],
    transform_xy: list[Any] | None,
    shape_fit_score: float,
    span_overlap_score: float,
    confluence_score: float,
    contradiction_penalty: float,
) -> list[str]:
    transform = (
        [round(float(transform_xy[0]), 3), round(float(transform_xy[1]), 3)]
        if isinstance(transform_xy, list) and len(transform_xy) == 2
        else []
    )
    return _unique_strings(
        [
            f"kind:{lock_kind}",
            f"axis:{axis}",
            f"source:{','.join(source_object_ids)}",
            f"target:{','.join(target_object_ids)}",
            f"ports:{','.join(target_port_ids)}",
            f"transform:{transform}",
            f"shape_fit:{_score_band(shape_fit_score)}",
            f"span:{_score_band(span_overlap_score)}",
            f"confluence:{_score_band(confluence_score)}",
            f"contradiction:{_score_band(contradiction_penalty)}",
        ]
    )


def _edge_lock_config_from_alignment(alignment: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(alignment, dict):
        return None
    statecon_id = str(alignment.get("statecon_id") or "")
    alignment_id = str(alignment.get("portband_id") or "")
    if not statecon_id or not alignment_id:
        return None
    transform = alignment.get("transform_hypothesis_xy")
    if not isinstance(transform, list) or len(transform) != 2:
        transform = [0.0, 0.0]
    source_object_ids = _unique_strings(list(alignment.get("source_object_ids", []) or []))
    target_object_ids = _unique_strings(list(alignment.get("target_object_ids", []) or []))
    source_component_ids = _unique_strings(list(alignment.get("source_component_ids", []) or []))
    target_component_ids = _unique_strings(list(alignment.get("target_component_ids", []) or []))
    source_port_ids = _unique_strings(list(alignment.get("source_port_ids", []) or []))
    target_port_ids = _unique_strings(list(alignment.get("target_port_ids", []) or []))
    source_bboxes = [_as_bbox(box) for box in alignment.get("source_bboxes_xyxy", []) or []]
    target_bboxes = [_as_bbox(box) for box in alignment.get("target_bboxes_xyxy", []) or []]
    imagined_bboxes = [_as_bbox(box) for box in alignment.get("imagined_source_bboxes_xyxy", []) or []]
    source_bboxes = [box for box in source_bboxes if box is not None]
    target_bboxes = [box for box in target_bboxes if box is not None]
    imagined_bboxes = [box for box in imagined_bboxes if box is not None]
    source_shape_terms = _bbox_shape_terms(source_bboxes)
    target_shape_terms = _bbox_shape_terms(target_bboxes)
    shape_fit_score = _shape_fit_from_terms(source_shape_terms, target_shape_terms)
    span_value = alignment.get("span_overlap_ratio")
    if isinstance(span_value, (int, float)):
        span_overlap_score = max(0.0, min(1.0, float(span_value)))
    else:
        target_sep = float(alignment.get("target_port_separation_px") or 0.0)
        error = float(alignment.get("alignment_error_px") or 0.0)
        span_overlap_score = max(0.0, min(1.0, 1.0 - (error / max(1.0, target_sep)))) if target_sep > 0.0 else 0.0
    confluence_score = _edge_lock_confluence_score(
        source_object_ids=source_object_ids,
        target_object_ids=target_object_ids,
        source_component_ids=source_component_ids,
        target_port_ids=target_port_ids,
    )
    axis = str(alignment.get("axis") or "unknown_axis")
    axis_score = _axis_preservation_score(axis, transform)
    contradiction_penalty = 0.0
    if not source_object_ids or not target_object_ids or not target_port_ids:
        contradiction_penalty += 0.18
    if shape_fit_score <= 0.0:
        contradiction_penalty += 0.12
    if alignment.get("alignment_kind") == "single_port_projection" and span_overlap_score <= 0.0:
        contradiction_penalty += 0.18
    static_score = _alignment_static_score(alignment)
    quality_score = _edge_lock_quality_score(
        static_score=static_score,
        shape_fit_score=shape_fit_score,
        span_overlap_score=span_overlap_score,
        confluence_score=confluence_score,
        boundary_strength=1.0 if alignment.get("exact_alignment") is True else (0.72 if alignment.get("near_alignment") is True else 0.45),
        axis_preservation_score=axis_score,
        contradiction_penalty=contradiction_penalty,
    )
    signature_terms = _edge_lock_signature_terms(
        lock_kind="direct_port_alignment_lock",
        axis=axis,
        source_object_ids=source_object_ids,
        target_object_ids=target_object_ids,
        target_port_ids=target_port_ids,
        transform_xy=transform,
        shape_fit_score=shape_fit_score,
        span_overlap_score=span_overlap_score,
        confluence_score=confluence_score,
        contradiction_penalty=contradiction_penalty,
    )
    signature = "|".join(signature_terms)
    return {
        "schema": PORTBAND_EDGE_LOCK_CONFIGURATION_SCHEMA,
        "status": "ranked_static_edge_lock_configuration",
        "edge_lock_configuration_id": _stable_id("edge_lock", statecon_id, alignment_id, signature),
        "lock_kind": "direct_port_alignment_lock",
        "statecon_id": statecon_id,
        "source_object_ids": source_object_ids,
        "target_object_ids": target_object_ids,
        "source_component_ids": source_component_ids,
        "target_component_ids": target_component_ids,
        "source_port_ids": source_port_ids,
        "target_port_ids": target_port_ids,
        "linked_alignment_ids": [alignment_id],
        "linked_fleet_candidate_ids": [],
        "axis": axis,
        "flow_axis": axis,
        "transform_hypothesis_xy": [round(float(transform[0]), 6), round(float(transform[1]), 6)],
        "shared_transform_or_offset_xy": [round(float(transform[0]), 6), round(float(transform[1]), 6)],
        "source_bboxes_xyxy": source_bboxes,
        "target_bboxes_xyxy": target_bboxes,
        "imagined_source_bboxes_xyxy": imagined_bboxes,
        "source_shape_terms": source_shape_terms,
        "target_shape_terms": target_shape_terms,
        "edge_alignment_status": alignment.get("match_status"),
        "edge_alignment_error_px": alignment.get("alignment_error_px"),
        "span_overlap_score": round(float(span_overlap_score), 6),
        "shape_fit_score": shape_fit_score,
        "confluence_score": confluence_score,
        "axis_preservation_score": axis_score,
        "support_ref_count": len(source_object_ids) + len(source_component_ids) + len(source_port_ids),
        "target_ref_count": len(target_object_ids) + len(target_component_ids) + len(target_port_ids),
        "contradiction_penalty": round(max(0.0, min(1.0, contradiction_penalty)), 6),
        "edge_lock_terms": signature_terms,
        "shape_configuration_signature": signature,
        "shape_configuration_hash": _stable_id("edge_lock_sig", signature),
        "edge_lock_quality_score": quality_score,
        "confidence_score": quality_score,
        "confidence_model": PORTBAND_EDGE_LOCK_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "potential_solution_signal_status": PORTBAND_EDGE_LOCK_POTENTIAL_STATUS,
        "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
        "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
        "evidence_status": "static_geometry_edge_lock_configuration_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _edge_lock_config_from_fleet_candidate(
    *,
    candidate: dict[str, Any],
    objects_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(candidate, dict):
        return None
    statecon_id = str(candidate.get("statecon_id") or "")
    candidate_id = str(candidate.get("fleet_candidate_id") or "")
    if not statecon_id or not candidate_id:
        return None
    transform = candidate.get("transform_hypothesis_xy")
    if not isinstance(transform, list) or len(transform) != 2:
        transform = [0.0, 0.0]
    matrix = candidate.get("combo_shape_matrix") if isinstance(candidate.get("combo_shape_matrix"), dict) else {}
    source_object_ids = _unique_strings(list(candidate.get("source_object_ids", []) or []))
    target_object_ids = _unique_strings(list(candidate.get("target_object_ids", []) or []))
    source_component_ids = _unique_strings(list(candidate.get("source_component_ids", []) or []))
    target_component_ids = _unique_strings(
        [
            value
            for target_id in target_object_ids
            for value in objects_by_id.get(str(target_id), {}).get("component_ids", [])
            if value is not None
        ]
    )
    target_port_ids = _unique_strings(list(candidate.get("target_port_ids", []) or []))
    source_bboxes = [_as_bbox(member.get("bbox_xyxy")) for member in matrix.get("members", []) or [] if isinstance(member, dict)]
    source_bboxes = [box for box in source_bboxes if box is not None]
    target_bboxes = [
        _as_bbox(objects_by_id.get(str(object_id), {}).get("bbox_xyxy"))
        for object_id in target_object_ids
    ]
    target_bboxes = [box for box in target_bboxes if box is not None]
    imagined_bboxes = [_as_bbox(box) for box in candidate.get("imagined_member_bboxes_xyxy", []) or []]
    imagined_bboxes = [box for box in imagined_bboxes if box is not None]
    source_shape_terms = _bbox_shape_terms(source_bboxes)
    if matrix:
        source_shape_terms.update(
            {
                "matrix_kind": matrix.get("matrix_kind"),
                "combo_member_count": matrix.get("member_count"),
                "bbox_fill_ratio": matrix.get("bbox_fill_ratio", source_shape_terms.get("bbox_fill_ratio")),
            }
        )
    target_shape_terms = _bbox_shape_terms(target_bboxes)
    shape_fit_score = max(
        _shape_fit_from_terms(source_shape_terms, target_shape_terms),
        max(0.0, min(1.0, float(candidate.get("portband_fit_score") or 0.0))),
    )
    span_overlap_score = max(0.0, min(1.0, float(candidate.get("span_overlap_score") or 0.0)))
    combo_size = int(candidate.get("combo_size") or len(source_object_ids) or 0)
    confluence_score = _edge_lock_confluence_score(
        source_object_ids=source_object_ids,
        target_object_ids=target_object_ids,
        source_component_ids=source_component_ids,
        target_port_ids=target_port_ids,
        combo_size=combo_size,
    )
    axis = str(candidate.get("axis") or "unknown_axis")
    axis_score = _axis_preservation_score(axis, transform)
    boundary_strength = max(0.0, min(1.0, float(candidate.get("portband_boundary_strength") or 0.0)))
    contradiction_penalty = 0.0
    if combo_size < 2:
        contradiction_penalty += 0.14
    if not candidate.get("portband_boundary_source_alignment_id"):
        contradiction_penalty += 0.10
    if shape_fit_score <= 0.0:
        contradiction_penalty += 0.12
    static_score = max(0.0, min(1.0, float(candidate.get("rank_score") or 0.0)))
    quality_score = _edge_lock_quality_score(
        static_score=static_score,
        shape_fit_score=shape_fit_score,
        span_overlap_score=span_overlap_score,
        confluence_score=confluence_score,
        boundary_strength=boundary_strength,
        axis_preservation_score=axis_score,
        contradiction_penalty=contradiction_penalty,
    )
    signature_terms = _edge_lock_signature_terms(
        lock_kind="fleet_combo_shape_lock",
        axis=axis,
        source_object_ids=source_object_ids,
        target_object_ids=target_object_ids,
        target_port_ids=target_port_ids,
        transform_xy=transform,
        shape_fit_score=shape_fit_score,
        span_overlap_score=span_overlap_score,
        confluence_score=confluence_score,
        contradiction_penalty=contradiction_penalty,
    )
    signature = "|".join(signature_terms)
    linked_alignment_ids = (
        [str(candidate.get("portband_boundary_source_alignment_id"))]
        if candidate.get("portband_boundary_source_alignment_id") is not None
        else []
    )
    return {
        "schema": PORTBAND_EDGE_LOCK_CONFIGURATION_SCHEMA,
        "status": "ranked_static_edge_lock_configuration",
        "edge_lock_configuration_id": _stable_id("edge_lock", statecon_id, candidate_id, signature),
        "lock_kind": "fleet_combo_shape_lock",
        "statecon_id": statecon_id,
        "source_object_ids": source_object_ids,
        "target_object_ids": target_object_ids,
        "source_component_ids": source_component_ids,
        "target_component_ids": target_component_ids,
        "source_port_ids": [],
        "target_port_ids": target_port_ids,
        "linked_alignment_ids": linked_alignment_ids,
        "linked_fleet_candidate_ids": [candidate_id],
        "axis": axis,
        "flow_axis": axis,
        "transform_hypothesis_xy": [round(float(transform[0]), 6), round(float(transform[1]), 6)],
        "shared_transform_or_offset_xy": [round(float(transform[0]), 6), round(float(transform[1]), 6)],
        "source_bboxes_xyxy": source_bboxes,
        "target_bboxes_xyxy": target_bboxes,
        "imagined_source_bboxes_xyxy": imagined_bboxes,
        "source_shape_terms": source_shape_terms,
        "target_shape_terms": target_shape_terms,
        "combo_size": combo_size,
        "combo_shape_matrix": matrix,
        "edge_alignment_status": "fleet_combo_shape_matrix_port_fit",
        "edge_alignment_error_px": candidate.get("alignment_error_px"),
        "span_overlap_score": span_overlap_score,
        "shape_fit_score": shape_fit_score,
        "confluence_score": confluence_score,
        "axis_preservation_score": axis_score,
        "portband_boundary_strength": boundary_strength,
        "support_ref_count": len(source_object_ids) + len(source_component_ids),
        "target_ref_count": len(target_object_ids) + len(target_component_ids) + len(target_port_ids),
        "contradiction_penalty": round(max(0.0, min(1.0, contradiction_penalty)), 6),
        "edge_lock_terms": signature_terms,
        "shape_configuration_signature": signature,
        "shape_configuration_hash": _stable_id("edge_lock_sig", signature),
        "edge_lock_quality_score": quality_score,
        "confidence_score": quality_score,
        "confidence_model": PORTBAND_EDGE_LOCK_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "potential_solution_signal_status": PORTBAND_EDGE_LOCK_POTENTIAL_STATUS,
        "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
        "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
        "evidence_status": "static_geometry_edge_lock_configuration_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _best_edge_lock_rows_by_key(rows: list[dict[str, Any]], key_name: str) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        for value in row.get(key_name, []) or []:
            key = str(value)
            existing = best.get(key)
            if existing is None or float(row.get("edge_lock_quality_score") or 0.0) > float(existing.get("edge_lock_quality_score") or 0.0):
                best[key] = row
    return best


def _build_edge_lock_ledger(
    *,
    statecon_id: str,
    objects: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    portband_fleet: dict[str, Any],
) -> dict[str, Any]:
    objects_by_id = _object_by_id(objects)
    rows: list[dict[str, Any]] = []
    for alignment in alignments:
        row = _edge_lock_config_from_alignment(alignment)
        if isinstance(row, dict):
            rows.append(row)
    fleet_candidates = (
        [row for row in portband_fleet.get("ranked_candidates", []) if isinstance(row, dict)]
        if isinstance(portband_fleet, dict)
        else []
    )
    for candidate in fleet_candidates:
        row = _edge_lock_config_from_fleet_candidate(candidate=candidate, objects_by_id=objects_by_id)
        if isinstance(row, dict):
            rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row.get("edge_lock_quality_score") or 0.0),
            float(row.get("edge_alignment_error_px") or 999.0),
            0 if row.get("lock_kind") == "fleet_combo_shape_lock" else 1,
            str(row.get("edge_lock_configuration_id")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["edge_lock_rank"] = rank
    best_by_source = [
        {
            "source_object_id": source_id,
            "edge_lock_configuration_id": row.get("edge_lock_configuration_id"),
            "edge_lock_quality_score": row.get("edge_lock_quality_score"),
            "edge_lock_rank": row.get("edge_lock_rank"),
            "lock_kind": row.get("lock_kind"),
            "target_object_ids": row.get("target_object_ids", []),
            "target_port_ids": row.get("target_port_ids", []),
            "shared_transform_or_offset_xy": row.get("shared_transform_or_offset_xy"),
        }
        for source_id, row in sorted(_best_edge_lock_rows_by_key(rows, "source_object_ids").items())
    ]
    best_by_target = [
        {
            "target_object_id": target_id,
            "edge_lock_configuration_id": row.get("edge_lock_configuration_id"),
            "edge_lock_quality_score": row.get("edge_lock_quality_score"),
            "edge_lock_rank": row.get("edge_lock_rank"),
            "lock_kind": row.get("lock_kind"),
            "source_object_ids": row.get("source_object_ids", []),
            "shared_transform_or_offset_xy": row.get("shared_transform_or_offset_xy"),
        }
        for target_id, row in sorted(_best_edge_lock_rows_by_key(rows, "target_object_ids").items())
    ]
    return {
        "schema": PORTBAND_EDGE_LOCK_LEDGER_SCHEMA,
        "status": "ranked" if rows else "no_static_edge_lock_configurations",
        "statecon_id": statecon_id,
        "definition": "copied_static_portband_edge_locks_ranked_as_candidate_shape_configurations",
        "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
        "confidence_model": PORTBAND_EDGE_LOCK_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "configuration_count": len(rows),
        "top_configuration_count": len(rows[:8]),
        "direct_alignment_lock_count": sum(1 for row in rows if row.get("lock_kind") == "direct_port_alignment_lock"),
        "fleet_combo_lock_count": sum(1 for row in rows if row.get("lock_kind") == "fleet_combo_shape_lock"),
        "positive_signal_count": sum(1 for row in rows if float(row.get("edge_lock_quality_score") or 0.0) > 0.0),
        "top_edge_lock_quality_score": max((float(row.get("edge_lock_quality_score") or 0.0) for row in rows), default=0.0),
        "top_configuration_ids": [row.get("edge_lock_configuration_id") for row in rows[:8]],
        "best_by_source_object": best_by_source,
        "best_by_target_object": best_by_target,
        "rows": rows,
        "top_rows": rows[:8],
        "potential_solution_signal_status": PORTBAND_EDGE_LOCK_POTENTIAL_STATUS,
        "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
        "evidence_status": "static_geometry_edge_lock_ledger_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
        "provenance": "derived_from_portband_direct_alignments_and_fleet_combo_shape_matrices",
    }


def _edge_lock_adjusted_lineup_score(base_score: float, lock_score: float) -> float:
    base = max(0.0, min(1.0, float(base_score)))
    lock = max(0.0, min(1.0, float(lock_score)))
    return round(max(base, (0.55 * base) + (0.45 * lock)), 6)


def _edge_locks_for_lineup_row(row: dict[str, Any], edge_lock_ledger: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(edge_lock_ledger, dict):
        return []
    alignment_ids = set(_unique_strings(list(row.get("linked_alignment_ids", []) or [])))
    fleet_ids = set(_unique_strings(list(row.get("linked_fleet_candidate_ids", []) or [])))
    source_ids = set(_unique_strings([row.get("source_object_id")] + list(row.get("source_object_ids", []) or [])))
    target_ids = set(_unique_strings(list(row.get("target_object_ids", []) or [])))
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lock in edge_lock_ledger.get("rows", []) or []:
        if not isinstance(lock, dict):
            continue
        lock_id = str(lock.get("edge_lock_configuration_id") or "")
        if not lock_id or lock_id in seen:
            continue
        lock_alignment_ids = set(_unique_strings(list(lock.get("linked_alignment_ids", []) or [])))
        lock_fleet_ids = set(_unique_strings(list(lock.get("linked_fleet_candidate_ids", []) or [])))
        lock_source_ids = set(_unique_strings(list(lock.get("source_object_ids", []) or [])))
        lock_target_ids = set(_unique_strings(list(lock.get("target_object_ids", []) or [])))
        if (
            (alignment_ids and alignment_ids & lock_alignment_ids)
            or (fleet_ids and fleet_ids & lock_fleet_ids)
            or (source_ids and target_ids and source_ids & lock_source_ids and target_ids & lock_target_ids)
        ):
            seen.add(lock_id)
            matches.append(lock)
    matches.sort(
        key=lambda lock: (
            -float(lock.get("edge_lock_quality_score") or 0.0),
            int(lock.get("edge_lock_rank") or 999999),
            str(lock.get("edge_lock_configuration_id")),
        )
    )
    return matches


def _apply_edge_lock_ledger_to_lineup_rows(
    rows: list[dict[str, Any]],
    edge_lock_ledger: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for row in rows:
        matches = _edge_locks_for_lineup_row(row, edge_lock_ledger)
        best = matches[0] if matches else None
        base_score = round(
            max(
                0.0,
                min(
                    1.0,
                    max(
                        float(row.get("edge_lock_adjusted_confidence_score") or 0.0),
                        float(row.get("confidence_score") or 0.0),
                    ),
                ),
            ),
            6,
        )
        new_row = dict(row)
        if isinstance(best, dict):
            lock_score = round(max(0.0, min(1.0, float(best.get("edge_lock_quality_score") or 0.0))), 6)
            adjusted = _edge_lock_adjusted_lineup_score(base_score, lock_score)
            new_row.update(
                {
                    "edge_lock_status": "has_ranked_edge_lock_configuration",
                    "edge_lock_configuration_ids": _unique_strings([lock.get("edge_lock_configuration_id") for lock in matches]),
                    "edge_lock_top_configuration_id": best.get("edge_lock_configuration_id"),
                    "edge_lock_rank": best.get("edge_lock_rank"),
                    "edge_lock_kind": best.get("lock_kind"),
                    "edge_lock_quality_score": lock_score,
                    "edge_lock_adjusted_confidence_score": adjusted,
                    "edge_lock_confidence_model": PORTBAND_EDGE_LOCK_CONFIDENCE_MODEL,
                    "edge_lock_confidence_is_probability": False,
                    "edge_lock_shape_configuration_hash": best.get("shape_configuration_hash"),
                    "edge_lock_terms": best.get("edge_lock_terms", []),
                    "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
                    "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
                    "potential_solution_signal_status": PORTBAND_EDGE_LOCK_POTENTIAL_STATUS,
                    "effective_lineup_confidence_score": adjusted,
                    "effective_lineup_confidence_model": "static_lineup_score_with_edge_lock_configuration_not_probability",
                    "effective_lineup_confidence_is_probability": False,
                }
            )
        else:
            new_row.update(
                {
                    "edge_lock_status": "no_edge_lock_configuration",
                    "edge_lock_configuration_ids": [],
                    "edge_lock_top_configuration_id": None,
                    "edge_lock_rank": None,
                    "edge_lock_quality_score": 0.0,
                    "edge_lock_adjusted_confidence_score": base_score,
                    "edge_lock_confidence_model": PORTBAND_EDGE_LOCK_CONFIDENCE_MODEL,
                    "edge_lock_confidence_is_probability": False,
                    "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
                    "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
                    "potential_solution_signal_status": "no_candidate_shape_configuration_potential",
                    "effective_lineup_confidence_score": base_score,
                    "effective_lineup_confidence_model": PORTBAND_LINEUP_CONFIDENCE_MODEL,
                    "effective_lineup_confidence_is_probability": False,
                }
            )
        updated.append(new_row)
    return updated


def _confidence_band(score: float) -> str:
    if score >= 0.80:
        return "high_static_lineup_confidence"
    if score >= 0.45:
        return "medium_static_lineup_confidence"
    if score > 0.0:
        return "low_static_lineup_confidence"
    return "zero_static_lineup_confidence"


def _lineup_matrix_row(
    *,
    statecon_id: str,
    row_kind: str,
    source_object: dict[str, Any],
    target_object_ids: list[str],
    target_component_ids: list[str],
    target_port_ids: list[str],
    transform_xy: list[Any],
    source_bbox: list[int] | None,
    imagined_bbox: list[int] | None,
    lineup_score: float,
    confidence_score: float,
    score_terms: dict[str, Any],
    linked_alignment_ids: list[str] | None = None,
    linked_fleet_candidate_ids: list[str] | None = None,
    source_route_index: int | None = None,
) -> dict[str, Any]:
    source_object_id = str(source_object.get("object_id") or "")
    source_component_ids = [str(value) for value in source_object.get("component_ids", []) if value is not None]
    safe_score = round(max(0.0, min(1.0, float(confidence_score))), 6)
    lineup_score = round(max(0.0, min(1.0, float(lineup_score))), 6)
    alignment_ids = [str(value) for value in linked_alignment_ids or [] if value is not None]
    fleet_ids = [str(value) for value in linked_fleet_candidate_ids or [] if value is not None]
    matrix_row_id = _stable_id(
        "portband_lineup",
        statecon_id,
        row_kind,
        source_object_id,
        alignment_ids,
        fleet_ids,
        target_object_ids,
        target_port_ids,
        source_route_index,
    )
    route_label = "translate_source_object_by_transform_hypothesis_xy_to_lineup_preview"
    return {
        "schema": PORTBAND_LINEUP_MATRIX_ROW_SCHEMA,
        "status": "hypothesis",
        "lineup_matrix_row_id": matrix_row_id,
        "statecon_id": statecon_id,
        "row_kind": row_kind,
        "source_object_id": source_object_id,
        "source_object_role": str(source_object.get("object_role") or "unknown"),
        "source_component_ids": source_component_ids,
        "source_bbox_xyxy": source_bbox,
        "target_object_ids": [str(value) for value in target_object_ids if value is not None],
        "target_component_ids": [str(value) for value in target_component_ids if value is not None],
        "target_port_ids": [str(value) for value in target_port_ids if value is not None],
        "transform_hypothesis_xy": [round(float(transform_xy[0]), 6), round(float(transform_xy[1]), 6)],
        "imagined_lineup_bbox_xyxy": imagined_bbox,
        "lineup_route_label": route_label,
        "lineup_route_instruction": (
            "Review this source object translated by transform_hypothesis_xy as a static lineup preview; "
            "executor must not treat it as an action route."
        ),
        "lineup_route_status": "static_lineup_route_hypothesis_not_executable_route",
        "route_authority": "none_static_review_only",
        "lineup_score": lineup_score,
        "confidence_score": safe_score,
        "confidence_band": _confidence_band(safe_score),
        "confidence_model": PORTBAND_LINEUP_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "score_terms": score_terms,
        "linked_alignment_ids": alignment_ids,
        "linked_fleet_candidate_ids": fleet_ids,
        "lineup_link_status": "linked_to_portband_alignment_or_fleet_candidate",
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
        "provenance": "derived_from_portband_alignment_and_fleet_static_geometry",
    }


def _object_by_id(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(obj.get("object_id")): obj for obj in objects if isinstance(obj, dict) and obj.get("object_id") is not None}


def _fallback_source_bbox(alignment: dict[str, Any], source_index: int = 0) -> list[int] | None:
    boxes = alignment.get("source_bboxes_xyxy")
    if isinstance(boxes, list) and 0 <= source_index < len(boxes):
        return _as_bbox(boxes[source_index])
    return None


def _build_lineup_matrix(
    *,
    statecon_id: str,
    objects: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    portband_fleet: dict[str, Any],
    edge_lock_ledger: dict[str, Any] | None = None,
) -> dict[str, Any]:
    objects_by_id = _object_by_id(objects)
    rows: list[dict[str, Any]] = []
    for alignment in alignments:
        if not isinstance(alignment, dict):
            continue
        alignment_id = str(alignment.get("portband_id") or "")
        transform = alignment.get("transform_hypothesis_xy")
        if not isinstance(transform, list) or len(transform) != 2:
            continue
        confidence = _alignment_static_score(alignment)
        source_ids = [str(value) for value in alignment.get("source_object_ids", []) if value is not None]
        for source_index, source_object_id in enumerate(source_ids):
            source_object = objects_by_id.get(source_object_id, {"object_id": source_object_id, "object_role": "unknown", "component_ids": []})
            source_bbox = _as_bbox(source_object.get("bbox_xyxy")) or _fallback_source_bbox(alignment, source_index)
            rows.append(
                _lineup_matrix_row(
                    statecon_id=statecon_id,
                    row_kind="direct_port_alignment_lineup",
                    source_object=source_object,
                    target_object_ids=[str(value) for value in alignment.get("target_object_ids", []) if value is not None],
                    target_component_ids=[str(value) for value in alignment.get("target_component_ids", []) if value is not None],
                    target_port_ids=[str(value) for value in alignment.get("target_port_ids", []) if value is not None],
                    transform_xy=transform,
                    source_bbox=source_bbox,
                    imagined_bbox=_shift_bbox(source_bbox, transform),
                    lineup_score=confidence,
                    confidence_score=confidence,
                    score_terms={
                        "alignment_error_px": alignment.get("alignment_error_px"),
                        "exact_alignment": alignment.get("exact_alignment"),
                        "near_alignment": alignment.get("near_alignment"),
                        "match_status": alignment.get("match_status"),
                        "span_overlap_ratio": alignment.get("span_overlap_ratio"),
                        "alignment_kind": alignment.get("alignment_kind"),
                    },
                    linked_alignment_ids=[alignment_id] if alignment_id else [],
                    source_route_index=source_index,
                )
            )

    fleet_candidates = (
        [row for row in portband_fleet.get("ranked_candidates", []) if isinstance(row, dict)]
        if isinstance(portband_fleet, dict)
        else []
    )
    for candidate in fleet_candidates:
        candidate_id = str(candidate.get("fleet_candidate_id") or "")
        transform = candidate.get("transform_hypothesis_xy")
        if not isinstance(transform, list) or len(transform) != 2:
            continue
        matrix = candidate.get("combo_shape_matrix") if isinstance(candidate.get("combo_shape_matrix"), dict) else {}
        members = matrix.get("members") if isinstance(matrix.get("members"), list) else []
        imagined_members = candidate.get("imagined_member_bboxes_xyxy") if isinstance(candidate.get("imagined_member_bboxes_xyxy"), list) else []
        confidence = round(max(0.0, min(1.0, float(candidate.get("rank_score") or 0.0))), 6)
        target_component_ids = [
            str(value)
            for target_id in candidate.get("target_object_ids", []) or []
            for value in objects_by_id.get(str(target_id), {}).get("component_ids", [])
            if value is not None
        ]
        for source_index, member in enumerate(members):
            if not isinstance(member, dict):
                continue
            source_object_id = str(member.get("object_id") or "")
            source_object = objects_by_id.get(
                source_object_id,
                {
                    "object_id": source_object_id,
                    "object_role": "support_group",
                    "component_ids": [str(value) for value in member.get("component_ids", []) if value is not None],
                    "bbox_xyxy": member.get("bbox_xyxy"),
                },
            )
            source_bbox = _as_bbox(member.get("bbox_xyxy")) or _as_bbox(source_object.get("bbox_xyxy"))
            imagined_bbox = _as_bbox(imagined_members[source_index]) if source_index < len(imagined_members) else _shift_bbox(source_bbox, transform)
            rows.append(
                _lineup_matrix_row(
                    statecon_id=statecon_id,
                    row_kind="fleet_combo_member_lineup",
                    source_object=source_object,
                    target_object_ids=[str(value) for value in candidate.get("target_object_ids", []) if value is not None],
                    target_component_ids=target_component_ids,
                    target_port_ids=[str(value) for value in candidate.get("target_port_ids", []) if value is not None],
                    transform_xy=transform,
                    source_bbox=source_bbox,
                    imagined_bbox=imagined_bbox,
                    lineup_score=confidence,
                    confidence_score=confidence,
                    score_terms={
                        "fleet_rank": candidate.get("fleet_rank"),
                        "rank_score": candidate.get("rank_score"),
                        "portband_fit_score": candidate.get("portband_fit_score"),
                        "span_overlap_score": candidate.get("span_overlap_score"),
                        "portband_boundary_strength": candidate.get("portband_boundary_strength"),
                        "alignment_error_px": candidate.get("alignment_error_px"),
                        "combo_size": candidate.get("combo_size"),
                    },
                    linked_alignment_ids=[str(candidate.get("portband_boundary_source_alignment_id"))]
                    if candidate.get("portband_boundary_source_alignment_id") is not None
                    else [],
                    linked_fleet_candidate_ids=[candidate_id] if candidate_id else [],
                    source_route_index=source_index,
                )
            )

    rows = _apply_edge_lock_ledger_to_lineup_rows(rows, edge_lock_ledger)
    rows.sort(
        key=lambda row: (
            -float(row.get("effective_lineup_confidence_score") or row.get("confidence_score") or 0.0),
            -float(row.get("edge_lock_quality_score") or 0.0),
            -float(row.get("confidence_score") or 0.0),
            -float(row.get("lineup_score") or 0.0),
            0 if row.get("row_kind") == "direct_port_alignment_lineup" else 1,
            str(row.get("source_object_id")),
            str(row.get("lineup_matrix_row_id")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["lineup_rank"] = rank

    best_by_support: list[dict[str, Any]] = []
    seen_supports: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_object_id") or "")
        if not source_id or source_id in seen_supports or row.get("source_object_role") != "support_group":
            continue
        seen_supports.add(source_id)
        best_by_support.append(
            {
                "source_object_id": source_id,
                "best_lineup_matrix_row_id": row.get("lineup_matrix_row_id"),
                "confidence_score": row.get("confidence_score"),
                "confidence_band": row.get("confidence_band"),
                "edge_lock_status": row.get("edge_lock_status"),
                "edge_lock_top_configuration_id": row.get("edge_lock_top_configuration_id"),
                "edge_lock_configuration_ids": row.get("edge_lock_configuration_ids", []),
                "edge_lock_rank": row.get("edge_lock_rank"),
                "edge_lock_quality_score": row.get("edge_lock_quality_score"),
                "edge_lock_adjusted_confidence_score": row.get("edge_lock_adjusted_confidence_score"),
                "potential_solution_signal_status": row.get("potential_solution_signal_status"),
                "effective_lineup_confidence_score": row.get("effective_lineup_confidence_score"),
                "lineup_rank": row.get("lineup_rank"),
                "lineup_route_label": row.get("lineup_route_label"),
                "transform_hypothesis_xy": row.get("transform_hypothesis_xy"),
                "imagined_lineup_bbox_xyxy": row.get("imagined_lineup_bbox_xyxy"),
                "linked_alignment_ids": row.get("linked_alignment_ids", []),
                "linked_fleet_candidate_ids": row.get("linked_fleet_candidate_ids", []),
            }
        )

    return {
        "schema": PORTBAND_LINEUP_MATRIX_SCHEMA,
        "status": "ranked" if rows else "no_lineup_matrix_rows",
        "statecon_id": statecon_id,
        "definition": "per_support_object_lineup_route_confidence_matrix",
        "lineup_route_policy": "static_transform_preview_only_not_gameplay_route",
        "confidence_model": PORTBAND_LINEUP_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "edge_lock_ledger_schema": PORTBAND_EDGE_LOCK_LEDGER_SCHEMA,
        "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
        "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
        "edge_lock_configuration_count": int(edge_lock_ledger.get("configuration_count") or 0) if isinstance(edge_lock_ledger, dict) else 0,
        "edge_lock_positive_signal_count": sum(1 for row in rows if row.get("edge_lock_status") == "has_ranked_edge_lock_configuration"),
        "top_edge_lock_quality_score": max((float(row.get("edge_lock_quality_score") or 0.0) for row in rows), default=0.0),
        "row_count": len(rows),
        "support_object_count": len(seen_supports),
        "support_object_ids": sorted(seen_supports),
        "best_by_support_object": best_by_support,
        "rows": rows,
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
        "provenance": "linked_from_portband_direct_alignments_and_portband_fleet_candidates",
    }


def _event_surface_bbox(event: dict[str, Any]) -> list[int] | None:
    surface = event.get("surface") if isinstance(event.get("surface"), dict) else {}
    box = surface.get("bbox_xyxy")
    if isinstance(box, list) and len(box) == 4:
        return [int(value) for value in box]
    return None


def _event_dynamic_score(event: dict[str, Any]) -> float:
    contact = event.get("contact") if isinstance(event.get("contact"), dict) else {}
    confidence = float(event.get("confidence") or 0.0)
    frame_bonus = min(0.30, float(contact.get("contact_frame_count") or 0.0) / 80.0)
    contact_bonus = min(0.20, float(contact.get("max_surface_contact_cell_count") or 0.0) / 320.0)
    return round(max(0.0, min(1.0, (0.50 * confidence) + frame_bonus + contact_bonus)), 6)


def _unique_strings(values: list[Any]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _is_controlled_route_flow_event(event: dict[str, Any]) -> bool:
    return (
        event.get("action_primitive") == "activate_flow"
        and event.get("evidence_status") == "observed_action_surface_result_delta"
        and bool(event.get("action_invocation_id"))
    )


def _is_observed_action_state_event(event: dict[str, Any]) -> bool:
    return (
        event.get("evidence_status") == "observed_action_surface_result_delta"
        and bool(event.get("event_id"))
        and bool(event.get("action_invocation_id"))
    )


def _correlated_object_refs(alignment: dict[str, Any], matched_event: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    source_overlap = float(matched_event.get("source_overlap_ratio") or 0.0)
    target_overlap = float(matched_event.get("target_overlap_ratio") or 0.0)
    if source_overlap > 0.0:
        refs.append(
            {
                "candidate_role": "source_candidate",
                "object_ids": _unique_strings(list(alignment.get("source_object_ids", []) or [])),
                "component_ids": _unique_strings(list(alignment.get("source_component_ids", []) or [])),
                "overlap_ratio": round(source_overlap, 6),
            }
        )
    if target_overlap > 0.0:
        refs.append(
            {
                "candidate_role": "target_candidate",
                "object_ids": _unique_strings(list(alignment.get("target_object_ids", []) or [])),
                "component_ids": _unique_strings(list(alignment.get("target_component_ids", []) or [])),
                "overlap_ratio": round(target_overlap, 6),
            }
        )
    return refs


def _action_state_correlation_score(*, static_score: float, matched_event: dict[str, Any]) -> float:
    dynamic_score = float(matched_event.get("dynamic_event_score") or 0.0)
    overlap = max(float(matched_event.get("source_overlap_ratio") or 0.0), float(matched_event.get("target_overlap_ratio") or 0.0))
    event_confidence = float(matched_event.get("event_confidence") or 0.0)
    score = (0.42 * dynamic_score) + (0.30 * overlap) + (0.18 * float(static_score)) + (0.10 * event_confidence)
    return round(max(0.0, min(1.0, score)), 6)


def _action_state_tag_record(
    *,
    alignment: dict[str, Any],
    matched_event: dict[str, Any],
    static_score: float,
) -> dict[str, Any] | None:
    if not _is_observed_action_state_event(matched_event):
        return None
    portband_id = str(alignment.get("portband_id") or "")
    action_primitive = str(matched_event.get("action_primitive") or "unknown_action")
    event_id = str(matched_event.get("event_id") or "")
    action_invocation_id = str(matched_event.get("action_invocation_id") or "")
    score = _action_state_correlation_score(static_score=static_score, matched_event=matched_event)
    object_refs = _correlated_object_refs(alignment, matched_event)
    return {
        "schema": CONTRA_ACTION_STATE_TAG_SCHEMA,
        "status": "tagged_observed_action_state_correlation",
        "action_state_tag_id": _stable_id("action_state_tag", alignment.get("statecon_id"), portband_id, event_id, action_invocation_id, action_primitive),
        "tag_kind": "controlled_action_surface_result_overlap_to_candidate_object",
        "statecon_id": alignment.get("statecon_id"),
        "portband_id": portband_id,
        "linked_alignment_ids": [portband_id] if portband_id else [],
        "action_invocation_id": action_invocation_id,
        "action_primitive": action_primitive,
        "event_id": event_id,
        "event_rank": matched_event.get("event_rank"),
        "event_evidence_status": matched_event.get("evidence_status"),
        "event_surface_object_id": matched_event.get("surface_object_id"),
        "event_surface_source_object_id": matched_event.get("surface_source_object_id"),
        "event_surface_role": matched_event.get("surface_role"),
        "event_surface_bbox_xyxy": matched_event.get("surface_bbox_xyxy"),
        "dominant_side": matched_event.get("dominant_side"),
        "contact_frame_count": matched_event.get("contact_frame_count"),
        "max_surface_contact_cell_count": matched_event.get("max_surface_contact_cell_count"),
        "final_changed_pixel_count": matched_event.get("final_changed_pixel_count"),
        "source_overlap_ratio": matched_event.get("source_overlap_ratio"),
        "target_overlap_ratio": matched_event.get("target_overlap_ratio"),
        "dynamic_event_score": matched_event.get("dynamic_event_score"),
        "correlation_score": score,
        "confidence_score": score,
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "solve_like_evidence_status": CONTRA_SOLVE_LIKE_EVIDENCE_STATUS,
        "solve_like_evidence_flag": CONTRA_SOLVE_LIKE_EVIDENCE_FLAG,
        "solve_like_evidence_score": score,
        "solve_like_evidence_boundary": "candidate interaction clue only; not solve truth",
        "action_state_positive_signal": CONTRA_ACTION_STATE_POSITIVE_SIGNAL,
        "correlated_object_refs": object_refs,
        "correlated_object_ids": _unique_strings(
            [
                value
                for ref in object_refs
                for value in ref.get("object_ids", [])
            ]
        ),
        "correlated_component_ids": _unique_strings(
            [
                value
                for ref in object_refs
                for value in ref.get("component_ids", [])
            ]
            + list(matched_event.get("surface_component_ids", []) or [])
        ),
        "tagger_policy": CONTRA_ACTION_STATE_TAG_POLICY,
        "action_state_tagger_boundary": CONTRA_ACTION_STATE_TAG_BOUNDARY,
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _delta_bucket(delta: float) -> str:
    abs_delta = abs(float(delta))
    if abs_delta >= 45.0:
        return "very_large"
    if abs_delta >= 10.0:
        return "large"
    if abs_delta >= 2.0:
        return "medium"
    if abs_delta > 0.0:
        return "small"
    return "zero"


def _is_resource_gauge_field(field_path: str) -> bool:
    text = str(field_path or "").lower()
    if text.startswith(("resource.", "resources.", "fuel.", "energy.", "gauge.", "meter.", "budget.")):
        return True
    tokens = (
        "fuel_gauge",
        "fuel",
        "energy_gauge",
        "energy_meter",
        "resource_gauge",
        "budget_remaining",
        "actions_remaining",
        "steps_remaining",
        "turns_remaining",
        "remaining_fuel",
        "remaining_energy",
        "charge_meter",
        "battery_meter",
    )
    return any(token in text for token in tokens)


def _delta_family(field_path: str) -> str:
    if _is_resource_gauge_field(field_path):
        return "resource_gauge_change"
    if "temporal" in field_path or "trajectory" in field_path or "path_distance" in field_path:
        return "temporal_trajectory_change"
    if field_path.startswith("shape.") or "bbox_diagonal" in field_path or "aspect_ratio" in field_path:
        return "shape_geometry_change"
    if field_path.startswith("topology.") or "bbox_iou" in field_path or "overlap" in field_path:
        return "topological_relation_change"
    if field_path.startswith("color.") or "raw_value" in field_path:
        return "color_or_value_relation_change"
    if "angle" in field_path or "bearing" in field_path or "angular" in field_path:
        return "angle_relation_change"
    if "bandwidth" in field_path or "proximity" in field_path or "axis_alignment" in field_path:
        return "spatial_band_relation_change"
    if "symmetry" in field_path:
        return "symmetry_relation_change"
    return "numeric_relation_change"


def _numeric_delta_features(observation: dict[str, Any]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for row in observation.get("numeric_deltas", []) or []:
        if not isinstance(row, dict):
            continue
        delta = row.get("delta")
        if not isinstance(delta, (int, float)):
            continue
        field_path = str(row.get("field_path") or "")
        is_resource_gauge = _is_resource_gauge_field(field_path)
        if float(delta) == 0.0 and not is_resource_gauge:
            continue
        features.append(
            {
                "field_path": field_path,
                "before_value": row.get("before_value"),
                "after_value": row.get("after_value"),
                "delta": round(float(delta), 6),
                "abs_delta": round(abs(float(delta)), 6),
                "direction": "stable" if float(delta) == 0.0 else "increased" if float(delta) > 0.0 else "decreased",
                "delta_bucket": _delta_bucket(float(delta)),
                "relationship_family": _delta_family(field_path),
            }
        )
    features.sort(key=lambda row: (-float(row.get("abs_delta") or 0.0), str(row.get("field_path"))))
    return features


def _feature_by_path(features: list[dict[str, Any]], field_path: str) -> dict[str, Any] | None:
    for feature in features:
        if feature.get("field_path") == field_path:
            return feature
    return None


def _feature_containing(features: list[dict[str, Any]], text: str) -> dict[str, Any] | None:
    for feature in features:
        if text in str(feature.get("field_path") or ""):
            return feature
    return None


def _feature_delta(features: list[dict[str, Any]], field_path: str, default: float = 0.0) -> float:
    feature = _feature_by_path(features, field_path)
    if not isinstance(feature, dict):
        return default
    value = feature.get("delta")
    return float(value) if isinstance(value, (int, float)) else default


def _signed_bucket(delta: float) -> str:
    if float(delta) == 0.0:
        return "stable_zero"
    return f"{'positive' if float(delta) > 0.0 else 'negative'}_{_delta_bucket(float(delta))}"


def _direction_8way(dx: float, dy: float) -> str:
    if float(dx) == 0.0 and float(dy) == 0.0:
        return "stable"
    horizontal = "east" if float(dx) > 0.0 else "west" if float(dx) < 0.0 else ""
    vertical = "south" if float(dy) > 0.0 else "north" if float(dy) < 0.0 else ""
    if horizontal and vertical:
        return f"{vertical}_{horizontal}"
    return horizontal or vertical


def _axis_family(dx: float, dy: float) -> str:
    abs_dx = abs(float(dx))
    abs_dy = abs(float(dy))
    if abs_dx == 0.0 and abs_dy == 0.0:
        return "stable"
    if abs_dx >= abs_dy * 2.0:
        return "horizontal_dominant"
    if abs_dy >= abs_dx * 2.0:
        return "vertical_dominant"
    return "diagonal_or_balanced"


def _score_band(score: float) -> str:
    value = max(0.0, min(1.0, float(score)))
    if value >= 0.80:
        return "high"
    if value >= 0.55:
        return "medium"
    if value > 0.0:
        return "low"
    return "none"


def _temporal_trajectory_summary(observation: dict[str, Any]) -> dict[str, Any]:
    trajectory = observation.get("temporal_trajectory") if isinstance(observation.get("temporal_trajectory"), dict) else {}
    if not trajectory:
        return {
            "schema": CONTRA_TEMPORAL_TRAJECTORY_PROFILE_SCHEMA,
            "status": "unavailable",
            "confidence_is_probability": False,
            "evidence_status": "no_temporal_trajectory_profile_record",
            "admission_status": "not_admitted",
            "not_prediction": True,
            "not_control_truth": True,
            "not_solve_truth": True,
            **TRUTH_FLAGS,
        }
    vector = trajectory.get("trajectory_vector") if isinstance(trajectory.get("trajectory_vector"), dict) else {}
    return {
        "schema": trajectory.get("schema") or CONTRA_TEMPORAL_TRAJECTORY_PROFILE_SCHEMA,
        "status": trajectory.get("status") or "profiled",
        "trajectory_status": trajectory.get("trajectory_status"),
        "frame_count": trajectory.get("frame_count"),
        "multi_frame_available": trajectory.get("multi_frame_available") is True,
        "nonzero_frame_count": trajectory.get("nonzero_frame_count"),
        "path_distance_px": trajectory.get("path_distance_px", trajectory.get("path_distance")),
        "path_distance_bucket": trajectory.get("path_distance_bucket"),
        "peak_changed_frame_index": trajectory.get("peak_changed_frame_index"),
        "peak_changed_pixel_count": trajectory.get("peak_changed_pixel_count"),
        "peak_step_frame_index": trajectory.get("peak_step_frame_index"),
        "peak_step_changed_pixel_count": trajectory.get("peak_step_changed_pixel_count"),
        "final_changed_pixel_count": trajectory.get("final_changed_pixel_count"),
        "first_changed_center_xy": trajectory.get("first_changed_center_xy"),
        "last_changed_center_xy": trajectory.get("last_changed_center_xy"),
        "trajectory_vector": {
            "dx": vector.get("dx"),
            "dy": vector.get("dy"),
            "euclidean_magnitude": vector.get("euclidean_magnitude"),
            "magnitude_bucket": vector.get("magnitude_bucket"),
            "direction_8way": vector.get("direction_8way"),
            "axis_family": vector.get("axis_family"),
        },
        "trajectory_terms": trajectory.get("trajectory_terms", []),
        "confidence_is_probability": False,
        "evidence_status": trajectory.get("evidence_status") or "temporal_profile_from_controlled_action_return_frames_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _observation_section(observation: dict[str, Any], key: str) -> dict[str, Any]:
    section = observation.get(key)
    return section if isinstance(section, dict) else {}


def _shape_geometry_summary(observation: dict[str, Any], features: list[dict[str, Any]]) -> dict[str, Any]:
    section = _observation_section(observation, "shape_geometry")
    width = _feature_by_path(features, "shape.width_pixels")
    height = _feature_by_path(features, "shape.height_pixels")
    aspect = _feature_by_path(features, "shape.aspect_ratio")
    area_ratio = _feature_by_path(features, "shape.area_ratio_before_after")
    diagonal = _feature_by_path(features, "shape.bbox_diagonal_pixels")
    return {
        "status": "profiled" if section or any(isinstance(row, dict) for row in [width, height, aspect, area_ratio, diagonal]) else "unavailable",
        "before_bbox_xyxy": section.get("before_bbox_xyxy"),
        "after_bbox_xyxy": section.get("after_bbox_xyxy"),
        "width_delta_pixels": section.get("width_delta_pixels", width.get("delta") if isinstance(width, dict) else None),
        "height_delta_pixels": section.get("height_delta_pixels", height.get("delta") if isinstance(height, dict) else None),
        "aspect_ratio_delta": section.get("aspect_ratio_delta", aspect.get("delta") if isinstance(aspect, dict) else None),
        "area_delta_pixels": section.get("area_delta_pixels"),
        "area_ratio_before_after": section.get("area_ratio_before_after", area_ratio.get("after_value") if isinstance(area_ratio, dict) else None),
        "bbox_diagonal_delta_pixels": section.get("bbox_diagonal_delta_pixels", diagonal.get("delta") if isinstance(diagonal, dict) else None),
        "shape_change_status": (_observation_section(observation, "relationship_descriptors").get("shape_change") or "unknown"),
        "evidence_status": section.get("evidence_status") or "candidate_shape_geometry_delta_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _topology_relationship_summary(observation: dict[str, Any], features: list[dict[str, Any]]) -> dict[str, Any]:
    section = _observation_section(observation, "topology_relationship")
    iou = _feature_by_path(features, "topology.bbox_iou_before_after")
    overlap = _feature_by_path(features, "topology.bbox_overlap_area_pixels")
    overlap_ratio = _feature_by_path(features, "topology.bbox_overlap_ratio_of_smaller")
    distance = _feature_by_path(features, "topology.center_distance_pixels")
    manhattan = _feature_by_path(features, "topology.manhattan_distance_pixels")
    return {
        "status": "profiled" if section or any(isinstance(row, dict) for row in [iou, overlap, overlap_ratio, distance, manhattan]) else "unavailable",
        "bbox_iou_before_after": section.get("bbox_iou_before_after", iou.get("after_value") if isinstance(iou, dict) else None),
        "bbox_overlap_area_pixels": section.get("bbox_overlap_area_pixels", overlap.get("after_value") if isinstance(overlap, dict) else None),
        "bbox_overlap_ratio_of_smaller": section.get("bbox_overlap_ratio_of_smaller", overlap_ratio.get("after_value") if isinstance(overlap_ratio, dict) else None),
        "bbox_overlap_status": section.get("bbox_overlap_status") or _observation_section(observation, "relationship_descriptors").get("overlap_status"),
        "center_distance_pixels": section.get("center_distance_pixels", distance.get("after_value") if isinstance(distance, dict) else None),
        "manhattan_distance_pixels": section.get("manhattan_distance_pixels", manhattan.get("after_value") if isinstance(manhattan, dict) else None),
        "direction_8way": section.get("direction_8way"),
        "axis_family": section.get("axis_family"),
        "evidence_status": section.get("evidence_status") or "candidate_topology_relationship_delta_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _color_value_relationship_summary(observation: dict[str, Any], features: list[dict[str, Any]]) -> dict[str, Any]:
    section = _observation_section(observation, "color_value_relationship")
    raw_delta = _feature_by_path(features, "color.raw_value_delta")
    raw_changed = _feature_by_path(features, "color.raw_value_changed")
    return {
        "status": "profiled" if section or isinstance(raw_delta, dict) or isinstance(raw_changed, dict) else "unavailable",
        "before_raw_pixel_value": section.get("before_raw_pixel_value"),
        "after_raw_pixel_value": section.get("after_raw_pixel_value"),
        "raw_value_delta": section.get("raw_value_delta", raw_delta.get("delta") if isinstance(raw_delta, dict) else None),
        "raw_value_changed": section.get("raw_value_changed", bool(raw_changed.get("delta")) if isinstance(raw_changed, dict) else None),
        "raw_value_transition": section.get("raw_value_transition"),
        "raw_value_status": _observation_section(observation, "relationship_descriptors").get("raw_value_status"),
        "evidence_status": section.get("evidence_status") or "candidate_color_or_value_delta_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _numeric_value(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _resource_kind(field_paths: list[str], section: dict[str, Any]) -> str:
    declared = section.get("resource_kind")
    if isinstance(declared, str) and declared:
        return declared
    joined = " ".join(field_paths).lower()
    if "fuel" in joined:
        return "fuel_gauge"
    if "energy" in joined:
        return "energy_gauge"
    if "budget" in joined or "remaining" in joined:
        return "budget_gauge"
    if "meter" in joined or "charge" in joined or "battery" in joined:
        return "meter_gauge"
    return "resource_gauge"


def _resource_gauge_summary(observation: dict[str, Any], features: list[dict[str, Any]]) -> dict[str, Any]:
    section = _observation_section(observation, "resource_gauge")
    gauge_features = [feature for feature in features if _is_resource_gauge_field(str(feature.get("field_path") or ""))]
    if not section and not gauge_features:
        return {
            "schema": CONTRA_RESOURCE_GAUGE_PROFILE_SCHEMA,
            "status": "unavailable",
            "confidence_is_probability": False,
            "evidence_status": "no_resource_gauge_observation",
            "admission_status": "not_admitted",
            "not_prediction": True,
            "not_control_truth": True,
            "not_solve_truth": True,
            **TRUTH_FLAGS,
        }
    field_paths = _unique_strings([feature.get("field_path") for feature in gauge_features])
    before_values = [_numeric_value(feature.get("before_value")) for feature in gauge_features]
    after_values = [_numeric_value(feature.get("after_value")) for feature in gauge_features]
    deltas = [_numeric_value(feature.get("delta")) for feature in gauge_features]
    before_total = section.get("before_total")
    after_total = section.get("after_total")
    delta_total = section.get("delta_total")
    if not isinstance(before_total, (int, float)) and any(value is not None for value in before_values):
        before_total = sum(float(value) for value in before_values if value is not None)
    if not isinstance(after_total, (int, float)) and any(value is not None for value in after_values):
        after_total = sum(float(value) for value in after_values if value is not None)
    if not isinstance(delta_total, (int, float)) and any(value is not None for value in deltas):
        delta_total = sum(float(value) for value in deltas if value is not None)
    delta_value = float(delta_total) if isinstance(delta_total, (int, float)) else 0.0
    if delta_value < 0.0:
        status = "resource_consumed"
    elif delta_value > 0.0:
        status = "resource_replenished"
    else:
        status = "stable_resource_gauge"
    return {
        "schema": CONTRA_RESOURCE_GAUGE_PROFILE_SCHEMA,
        "status": section.get("status") or "profiled",
        "resource_kind": _resource_kind(field_paths, section),
        "gauge_status": section.get("gauge_status") or status,
        "gauge_field_paths": field_paths,
        "gauge_feature_count": len(gauge_features),
        "before_total": round(float(before_total), 6) if isinstance(before_total, (int, float)) else None,
        "after_total": round(float(after_total), 6) if isinstance(after_total, (int, float)) else None,
        "delta_total": round(delta_value, 6),
        "abs_delta_total": round(abs(delta_value), 6),
        "delta_bucket": _delta_bucket(delta_value),
        "resource_changed": delta_value != 0.0,
        "resource_consumed": delta_value < 0.0,
        "resource_replenished": delta_value > 0.0,
        "resource_stable": delta_value == 0.0,
        "feature_rows": gauge_features[:8],
        "confidence_is_probability": False,
        "evidence_status": section.get("evidence_status") or "candidate_resource_gauge_delta_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _motion_energy_conservation_assessment(
    *,
    vector_delta: dict[str, Any],
    shape_delta: dict[str, Any],
    color_value_delta: dict[str, Any],
    temporal_trajectory: dict[str, Any],
    resource_gauge: dict[str, Any],
) -> dict[str, Any]:
    temporal_vector = (
        temporal_trajectory.get("trajectory_vector")
        if isinstance(temporal_trajectory.get("trajectory_vector"), dict)
        else {}
    )
    vector_y = abs(float(vector_delta.get("dy") or 0.0))
    temporal_y = abs(float(temporal_vector.get("dy") or 0.0))
    y_motion_pixels = round(max(vector_y, temporal_y), 6)
    paint_or_value_motion = (
        color_value_delta.get("raw_value_changed") is True
        or shape_delta.get("shape_change_status") == "changed"
        or float(shape_delta.get("area_delta_pixels") or 0.0) != 0.0
    )
    gauge_available = resource_gauge.get("status") != "unavailable"
    resource_delta = float(resource_gauge.get("delta_total") or 0.0) if gauge_available else None
    resource_stable = gauge_available and resource_delta == 0.0
    if y_motion_pixels <= 0.0:
        status = "no_y_motion_observed"
        y_motion_resource_coupling_status = "not_applicable_no_y_motion"
        y_motion_paint_coupling_status = "not_applicable_no_y_motion"
    elif not gauge_available:
        status = "resource_gauge_unobserved_for_y_motion"
        y_motion_resource_coupling_status = "unmeasured_resource_channel"
        y_motion_paint_coupling_status = (
            "candidate_y_motion_with_paint_channel_no_resource_gauge"
            if paint_or_value_motion
            else "candidate_y_motion_without_paint_delta_no_resource_gauge"
        )
    elif resource_stable:
        status = "candidate_y_motion_resource_conserved"
        y_motion_resource_coupling_status = "candidate_y_motion_free_from_resource_debit"
        y_motion_paint_coupling_status = (
            "candidate_y_motion_free_from_paint_resource_debit"
            if paint_or_value_motion
            else "candidate_y_motion_resource_conserved_without_paint_delta"
        )
    else:
        status = "candidate_y_motion_resource_coupled"
        y_motion_resource_coupling_status = "candidate_y_motion_coupled_to_resource_delta"
        y_motion_paint_coupling_status = (
            "candidate_y_motion_paint_resource_coupled"
            if paint_or_value_motion
            else "candidate_y_motion_resource_coupled_without_paint_delta"
        )
    return {
        "schema": CONTRA_MOTION_ENERGY_CONSERVATION_SCHEMA,
        "status": status,
        "y_motion_pixels": y_motion_pixels,
        "vector_y_delta_pixels": round(vector_y, 6),
        "temporal_y_delta_pixels": round(temporal_y, 6),
        "paint_or_value_motion_present": paint_or_value_motion,
        "resource_gauge_available": gauge_available,
        "resource_kind": resource_gauge.get("resource_kind"),
        "resource_gauge_status": resource_gauge.get("gauge_status"),
        "resource_delta_total": round(resource_delta, 6) if isinstance(resource_delta, (int, float)) else None,
        "y_motion_resource_coupling_status": y_motion_resource_coupling_status,
        "y_motion_paint_coupling_status": y_motion_paint_coupling_status,
        "assessment_basis": _unique_strings(
            [
                f"vector_y:{_delta_bucket(vector_y)}",
                f"temporal_y:{_delta_bucket(temporal_y)}",
                f"paint_or_value:{paint_or_value_motion}",
                f"resource:{resource_gauge.get('gauge_status')}",
                f"resource_delta:{_delta_bucket(resource_delta or 0.0) if isinstance(resource_delta, (int, float)) else 'unavailable'}",
            ]
        ),
        "assessment_policy": "compare motion, paint/value, and resource-gauge channels as candidate conservation signals only",
        "confidence_is_probability": False,
        "evidence_status": "candidate_motion_energy_conservation_assessment_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _relationship_profile_terms(features: list[dict[str, Any]], observation: dict[str, Any]) -> list[str]:
    dx = _feature_delta(features, "spatial.centroid_dx_pixels")
    dy = _feature_delta(features, "spatial.centroid_dy_pixels")
    angle_feature = _feature_containing(features, "angle")
    area_feature = _feature_by_path(features, "component.area_pixels")
    changed_pixels = _feature_by_path(features, "frame.changed_pixel_count")
    member_delta = observation.get("member_delta") if isinstance(observation.get("member_delta"), dict) else {}
    correspondence = observation.get("object_correspondence") if isinstance(observation.get("object_correspondence"), dict) else {}
    temporal = _temporal_trajectory_summary(observation)
    temporal_vector = temporal.get("trajectory_vector") if isinstance(temporal.get("trajectory_vector"), dict) else {}
    shape = _shape_geometry_summary(observation, features)
    topology = _topology_relationship_summary(observation, features)
    color_value = _color_value_relationship_summary(observation, features)
    resource_gauge = _resource_gauge_summary(observation, features)
    terms = [
        f"vector:{_direction_8way(dx, dy)}:{_axis_family(dx, dy)}:{_delta_bucket(math.hypot(dx, dy))}",
        f"member:{'changed' if member_delta.get('changed') is True else 'stable'}",
        f"ray:{_signed_bucket(float(observation.get('ray_count_delta') or 0.0))}",
    ]
    if correspondence:
        terms.append(f"correspondence:{_score_band(float(correspondence.get('confidence_score') or 0.0))}")
    if isinstance(angle_feature, dict):
        terms.append(f"angle:{angle_feature.get('direction')}:{angle_feature.get('delta_bucket')}")
    if isinstance(area_feature, dict):
        terms.append(f"area:{area_feature.get('direction')}:{area_feature.get('delta_bucket')}")
    if isinstance(changed_pixels, dict):
        terms.append(f"pixels:{changed_pixels.get('direction')}:{changed_pixels.get('delta_bucket')}")
    if shape.get("status") != "unavailable":
        terms.extend(
            [
                f"shape:{shape.get('shape_change_status')}:{_delta_bucket(float(shape.get('area_delta_pixels') or 0.0))}",
                f"shape_aspect:{_delta_bucket(float(shape.get('aspect_ratio_delta') or 0.0))}",
                f"shape_area_ratio:{_delta_bucket(float(shape.get('area_ratio_before_after') or 0.0) - 1.0)}",
            ]
        )
    if topology.get("status") != "unavailable":
        terms.extend(
            [
                f"topology:{topology.get('bbox_overlap_status')}:{_delta_bucket(float(topology.get('center_distance_pixels') or 0.0))}",
                f"topology_iou:{_delta_bucket(float(topology.get('bbox_iou_before_after') or 0.0))}",
            ]
        )
    if color_value.get("status") != "unavailable":
        terms.append(f"raw_value:{color_value.get('raw_value_status')}:{color_value.get('raw_value_transition')}")
    if resource_gauge.get("status") != "unavailable":
        terms.extend(
            [
                f"resource_gauge:{resource_gauge.get('resource_kind')}:{resource_gauge.get('gauge_status')}:{resource_gauge.get('delta_bucket')}",
                f"resource_stable:{resource_gauge.get('resource_stable') is True}",
            ]
        )
    if temporal.get("status") != "unavailable":
        terms.extend(
            [
                f"temporal_frames:{temporal.get('frame_count')}",
                f"temporal_trajectory:{temporal_vector.get('direction_8way')}:{temporal_vector.get('axis_family')}:{temporal_vector.get('magnitude_bucket')}",
                f"temporal_path:{temporal.get('path_distance_bucket')}",
                f"temporal_peak:{_delta_bucket(float(temporal.get('peak_changed_pixel_count') or 0.0))}",
            ]
        )
    return _unique_strings(terms)


def _mathematical_invariant_profile(
    *,
    profile_terms: list[str],
    dominant_families: list[str],
    vector_delta: dict[str, Any],
    angle_delta: dict[str, Any],
    area_delta: dict[str, Any],
    shape_delta: dict[str, Any],
    topology_delta: dict[str, Any],
    color_value_delta: dict[str, Any],
    temporal_trajectory: dict[str, Any],
    resource_gauge: dict[str, Any],
    motion_energy_conservation: dict[str, Any],
    member_change: dict[str, Any],
) -> dict[str, Any]:
    temporal_vector = (
        temporal_trajectory.get("trajectory_vector")
        if isinstance(temporal_trajectory.get("trajectory_vector"), dict)
        else {}
    )
    terms = _unique_strings(
        [
            f"vector_direction:{vector_delta.get('direction_8way')}",
            f"vector_axis:{vector_delta.get('axis_family')}",
            f"vector_magnitude:{vector_delta.get('magnitude_bucket')}",
            f"angle:{angle_delta.get('direction')}:{angle_delta.get('delta_bucket')}",
            f"area:{area_delta.get('direction')}:{area_delta.get('delta_bucket')}",
            f"shape:{shape_delta.get('shape_change_status')}:{_delta_bucket(float(shape_delta.get('area_delta_pixels') or 0.0))}",
            f"shape_aspect:{_delta_bucket(float(shape_delta.get('aspect_ratio_delta') or 0.0))}",
            f"topology:{topology_delta.get('bbox_overlap_status')}:{_delta_bucket(float(topology_delta.get('center_distance_pixels') or 0.0))}",
            f"topology_iou:{_delta_bucket(float(topology_delta.get('bbox_iou_before_after') or 0.0))}",
            f"value:{color_value_delta.get('raw_value_status')}:{color_value_delta.get('raw_value_changed')}",
            f"resource:{resource_gauge.get('resource_kind')}:{resource_gauge.get('gauge_status')}:{resource_gauge.get('delta_bucket')}",
            f"conservation:{motion_energy_conservation.get('status')}:{motion_energy_conservation.get('y_motion_resource_coupling_status')}",
            f"temporal_direction:{temporal_vector.get('direction_8way')}",
            f"temporal_axis:{temporal_vector.get('axis_family')}",
            f"temporal_magnitude:{temporal_vector.get('magnitude_bucket')}",
            f"temporal_path:{temporal_trajectory.get('path_distance_bucket')}",
            f"member_changed:{member_change.get('changed') is True}",
        ]
        + [f"family:{family}" for family in dominant_families]
    )
    signature = "|".join(terms)
    return {
        "status": "profiled",
        "invariant_signature": signature,
        "invariant_hash": _stable_id("math_invariant", signature),
        "invariant_terms": terms,
        "supporting_profile_terms": _unique_strings(profile_terms)[:18],
        "dominant_relationship_families": dominant_families,
        "invariant_policy": "bucket_observed_relative_math_terms_for_cross_game_reuse_not_truth",
        "confidence_is_probability": False,
        "evidence_status": "candidate_mathematical_invariant_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _mathematical_relationship_profile(
    *,
    action_primitive: str | None,
    observation: dict[str, Any],
    features: list[dict[str, Any]],
    signature: str,
    signature_count: int,
) -> dict[str, Any]:
    dx = _feature_delta(features, "spatial.centroid_dx_pixels")
    dy = _feature_delta(features, "spatial.centroid_dy_pixels")
    magnitude = round(math.hypot(dx, dy), 6)
    manhattan = round(abs(dx) + abs(dy), 6)
    angle_feature = _feature_containing(features, "angle")
    area_feature = _feature_by_path(features, "component.area_pixels")
    bbox_area_feature = _feature_by_path(features, "component.changed_bbox_area_pixels")
    changed_pixels = _feature_by_path(features, "frame.changed_pixel_count")
    member_delta = observation.get("member_delta") if isinstance(observation.get("member_delta"), dict) else {}
    correspondence = observation.get("object_correspondence") if isinstance(observation.get("object_correspondence"), dict) else {}
    temporal = _temporal_trajectory_summary(observation)
    temporal_vector = temporal.get("trajectory_vector") if isinstance(temporal.get("trajectory_vector"), dict) else {}
    shape = _shape_geometry_summary(observation, features)
    topology = _topology_relationship_summary(observation, features)
    color_value = _color_value_relationship_summary(observation, features)
    resource_gauge = _resource_gauge_summary(observation, features)
    profile_terms = _relationship_profile_terms(features, observation)
    dominant_families = _unique_strings([feature.get("relationship_family") for feature in features])
    vector_delta = {
        "dx": round(dx, 6),
        "dy": round(dy, 6),
        "euclidean_magnitude": magnitude,
        "manhattan_magnitude": manhattan,
        "magnitude_bucket": _delta_bucket(magnitude),
        "direction_8way": _direction_8way(dx, dy),
        "axis_family": _axis_family(dx, dy),
    }
    angle_delta = {
        "field_path": angle_feature.get("field_path") if isinstance(angle_feature, dict) else None,
        "delta_degrees": angle_feature.get("delta") if isinstance(angle_feature, dict) else None,
        "direction": angle_feature.get("direction") if isinstance(angle_feature, dict) else None,
        "delta_bucket": angle_feature.get("delta_bucket") if isinstance(angle_feature, dict) else None,
    }
    area_delta = {
        "field_path": area_feature.get("field_path") if isinstance(area_feature, dict) else None,
        "delta_pixels": area_feature.get("delta") if isinstance(area_feature, dict) else None,
        "direction": area_feature.get("direction") if isinstance(area_feature, dict) else None,
        "delta_bucket": area_feature.get("delta_bucket") if isinstance(area_feature, dict) else None,
        "changed_bbox_area_delta": bbox_area_feature.get("delta") if isinstance(bbox_area_feature, dict) else None,
    }
    member_change = {
        "changed": member_delta.get("changed") is True,
        "before_member_count": len(_member_ref_values(member_delta.get("before_members"))),
        "after_member_count": len(_member_ref_values(member_delta.get("after_members"))),
    }
    motion_energy_conservation = _motion_energy_conservation_assessment(
        vector_delta=vector_delta,
        shape_delta=shape,
        color_value_delta=color_value,
        temporal_trajectory=temporal,
        resource_gauge=resource_gauge,
    )
    invariant_profile = _mathematical_invariant_profile(
        profile_terms=profile_terms,
        dominant_families=dominant_families,
        vector_delta=vector_delta,
        angle_delta=angle_delta,
        area_delta=area_delta,
        shape_delta=shape,
        topology_delta=topology,
        color_value_delta=color_value,
        temporal_trajectory=temporal,
        resource_gauge=resource_gauge,
        motion_energy_conservation=motion_energy_conservation,
        member_change=member_change,
    )
    relation_type_parts = [
        _direction_8way(dx, dy),
        _axis_family(dx, dy),
    ]
    if isinstance(angle_feature, dict):
        relation_type_parts.append(f"angle_{angle_feature.get('direction')}_{angle_feature.get('delta_bucket')}")
    if isinstance(area_feature, dict):
        relation_type_parts.append(f"area_{area_feature.get('direction')}_{area_feature.get('delta_bucket')}")
    if shape.get("status") != "unavailable":
        relation_type_parts.append(f"shape_{shape.get('shape_change_status')}_{_delta_bucket(float(shape.get('area_delta_pixels') or 0.0))}")
    if topology.get("status") != "unavailable":
        relation_type_parts.append(f"topology_{topology.get('bbox_overlap_status')}_{_delta_bucket(float(topology.get('center_distance_pixels') or 0.0))}")
    if color_value.get("status") != "unavailable":
        relation_type_parts.append(f"raw_value_{color_value.get('raw_value_status')}")
    if resource_gauge.get("status") != "unavailable":
        relation_type_parts.append(
            f"resource_{resource_gauge.get('resource_kind')}_{resource_gauge.get('gauge_status')}"
        )
    if motion_energy_conservation.get("status") not in {"no_y_motion_observed", None, ""}:
        relation_type_parts.append(f"conservation_{motion_energy_conservation.get('status')}")
    if temporal.get("status") != "unavailable":
        relation_type_parts.append(
            f"temporal_{temporal_vector.get('direction_8way')}_{temporal_vector.get('magnitude_bucket')}"
        )
    if member_delta.get("changed") is True:
        relation_type_parts.append("member_changed")
    relation_type_label = "relative_" + "_".join(str(part) for part in relation_type_parts if part)
    return {
        "schema": CONTRA_RELATIVE_MATH_PROFILE_SCHEMA,
        "status": "derived_from_controlled_relative_state_change",
        "action_primitive": str(action_primitive or "unknown_action"),
        "relation_type_label": relation_type_label,
        "relationship_signature": signature,
        "signature_count_in_action_boundary": int(signature_count),
        "signature_terms": profile_terms,
        "dominant_relationship_families": dominant_families,
        "mathematical_invariant_profile": invariant_profile,
        "mathematical_invariant_signature": invariant_profile.get("invariant_signature"),
        "mathematical_invariant_hash": invariant_profile.get("invariant_hash"),
        "mathematical_invariant_terms": invariant_profile.get("invariant_terms", []),
        "vector_delta": vector_delta,
        "angle_delta": angle_delta,
        "area_delta": area_delta,
        "pixel_delta": {
            "changed_pixel_delta": changed_pixels.get("delta") if isinstance(changed_pixels, dict) else None,
            "delta_bucket": changed_pixels.get("delta_bucket") if isinstance(changed_pixels, dict) else None,
        },
        "shape_delta": shape,
        "topology_delta": topology,
        "color_value_delta": color_value,
        "resource_gauge": resource_gauge,
        "motion_energy_conservation": motion_energy_conservation,
        "member_change": member_change,
        "object_correspondence": {
            "schema": correspondence.get("schema"),
            "status": correspondence.get("status"),
            "confidence_score": correspondence.get("confidence_score"),
            "confidence_band": _score_band(float(correspondence.get("confidence_score") or 0.0)),
            "confidence_model": correspondence.get("confidence_model"),
            "confidence_is_probability": False,
            "candidate_pair_index": correspondence.get("candidate_pair_index"),
            "selected_pair_rank": correspondence.get("selected_pair_rank"),
            "selected_pair": correspondence.get("selected_pair"),
            "candidate_pair_count": correspondence.get("candidate_pair_count"),
            "correspondence_policy": correspondence.get("correspondence_policy"),
            "evidence_status": correspondence.get("evidence_status"),
            "admission_status": "not_admitted",
            "not_prediction": True,
            "not_control_truth": True,
            "not_solve_truth": True,
            **TRUTH_FLAGS,
        }
        if correspondence
        else {
            "schema": CONTRA_OBJECT_CORRESPONDENCE_SCHEMA,
            "status": "unavailable",
            "confidence_score": 0.0,
            "confidence_band": "none",
            "confidence_model": "none",
            "confidence_is_probability": False,
            "evidence_status": "no_object_correspondence_record",
            "admission_status": "not_admitted",
            "not_prediction": True,
            "not_control_truth": True,
            "not_solve_truth": True,
            **TRUTH_FLAGS,
        },
        "temporal_trajectory": temporal,
        "mathematical_basis": [
            "centroid vector from before candidate component to after candidate component",
            "angle delta from frame center to candidate component",
            "component area and changed-pixel deltas",
            "shape geometry deltas for width, height, aspect, area ratio, and bbox diagonal",
            "topological relationship deltas for bbox overlap, iou, and center distance",
            "color or raw-value transition deltas when visible",
            "resource-gauge or fuel-meter deltas, including stable zero deltas when observed",
            "motion/paint/resource conservation assessment for candidate free-motion channels",
            "temporal trajectory from returned action frame sequence when available",
            "bucketed mathematical invariant signature for cross-game relationship matching",
            "scored before/after object correspondence when available",
            "member-set and ray-count deltas",
        ],
        "evidence_status": "candidate_math_profile_from_controlled_relative_delta_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _relationship_signature(
    *,
    action_primitive: str | None,
    observation: dict[str, Any],
    features: list[dict[str, Any]],
) -> str:
    changed_fields = _unique_strings(list(observation.get("changed_fields", []) or []))
    member_delta = observation.get("member_delta") if isinstance(observation.get("member_delta"), dict) else {}
    member_changed = "member_changed" if member_delta.get("changed") is True else "member_stable"
    ray_delta = int(observation.get("ray_count_delta") or 0)
    top_features = [
        f"{row.get('relationship_family')}:{row.get('field_path')}:{row.get('direction')}:{row.get('delta_bucket')}"
        for row in features[:4]
    ]
    profile_terms = _relationship_profile_terms(features, observation)
    return "|".join(
        [
            str(action_primitive or "unknown_action"),
            member_changed,
            f"ray_delta:{ray_delta}",
            ",".join(changed_fields[:8]),
            ",".join(_unique_strings(profile_terms + top_features)[:10]),
        ]
    )


def _relationship_novelty_map(
    *,
    observations: list[dict[str, Any]],
    action_primitive: str | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for observation in observations:
        if not isinstance(observation, dict) or observation.get("relationship_changed") is not True:
            continue
        features = _numeric_delta_features(observation)
        signature = _relationship_signature(action_primitive=action_primitive, observation=observation, features=features)
        counts[signature] = counts.get(signature, 0) + 1
    return counts


def _relationship_action_state_tag_record(
    *,
    portband_hypotheses: dict[str, Any],
    observation: dict[str, Any],
    action_primitive: str | None,
    signature_counts: dict[str, int],
) -> dict[str, Any] | None:
    if observation.get("relationship_changed") is not True:
        return None
    statecon_id = str(portband_hypotheses.get("statecon_id") or "")
    before_statecon_id = str(observation.get("before_statecon_id") or "")
    if statecon_id and before_statecon_id and statecon_id != before_statecon_id:
        return None
    features = _numeric_delta_features(observation)
    signature = _relationship_signature(action_primitive=action_primitive, observation=observation, features=features)
    signature_count = int(signature_counts.get(signature, 0) or 0)
    math_profile = _mathematical_relationship_profile(
        action_primitive=action_primitive,
        observation=observation,
        features=features,
        signature=signature,
        signature_count=signature_count,
    )
    member_delta = observation.get("member_delta") if isinstance(observation.get("member_delta"), dict) else {}
    ray_delta = int(observation.get("ray_count_delta") or 0)
    changed_fields = _unique_strings(list(observation.get("changed_fields", []) or []))
    max_abs_delta = max((float(row.get("abs_delta") or 0.0) for row in features), default=0.0)
    normalized_delta = min(1.0, max_abs_delta / 90.0)
    structural_bonus = 0.18 if member_delta.get("changed") is True else 0.0
    ray_bonus = min(0.16, abs(ray_delta) * 0.04)
    uniqueness_bonus = 0.18 if signature_count == 1 else 0.0
    field_bonus = min(0.20, len(changed_fields) * 0.025)
    temporal_profile = math_profile.get("temporal_trajectory") if isinstance(math_profile.get("temporal_trajectory"), dict) else {}
    temporal_path_bonus = min(0.10, float(temporal_profile.get("path_distance_px") or 0.0) / 180.0)
    temporal_sequence_bonus = 0.03 if temporal_profile.get("multi_frame_available") is True else 0.0
    temporal_bonus = temporal_path_bonus + temporal_sequence_bonus
    score = round(max(0.0, min(1.0, 0.28 + (0.30 * normalized_delta) + structural_bonus + ray_bonus + uniqueness_bonus + field_bonus + temporal_bonus)), 6)
    dominant_feature = features[0] if features else None
    return {
        "schema": CONTRA_ACTION_STATE_TAG_SCHEMA,
        "status": "tagged_controlled_relative_state_change",
        "action_state_tag_id": _stable_id("action_state_tag", statecon_id, observation.get("deltaband_id"), action_primitive, signature),
        "tag_kind": "controlled_action_relative_state_change",
        "tag_scope": "statecon_relation",
        "statecon_id": statecon_id or before_statecon_id,
        "before_statecon_id": before_statecon_id,
        "after_statecon_id": observation.get("after_statecon_id"),
        "deltaband_id": observation.get("deltaband_id"),
        "action_invocation_id": observation.get("action_invocation_id"),
        "action_primitive": str(action_primitive or "unknown_action"),
        "event_id": observation.get("deltaband_id"),
        "event_evidence_status": observation.get("evidence_status"),
        "relationship_changed": True,
        "changed_fields": changed_fields,
        "numeric_delta_features": features[:12],
        "dominant_numeric_delta": dominant_feature,
        "member_delta": member_delta,
        "ray_count_delta": ray_delta,
        "mathematical_relationships": {
            "relationship_signature": signature,
            "mathematical_invariant_signature": math_profile.get("mathematical_invariant_signature"),
            "mathematical_invariant_hash": math_profile.get("mathematical_invariant_hash"),
            "mathematical_invariant_terms": math_profile.get("mathematical_invariant_terms", []),
            "relationship_profile": math_profile,
            "relation_type_label": math_profile.get("relation_type_label"),
            "signature_terms": math_profile.get("signature_terms", []),
            "dominant_relationship_families": math_profile.get("dominant_relationship_families", []),
            "signature_count_in_action_boundary": signature_count,
            "uniqueness_status": "unique_relative_state_signature" if signature_count == 1 else "repeated_relative_state_signature",
            "dominant_delta_family": dominant_feature.get("relationship_family") if isinstance(dominant_feature, dict) else None,
            "dominant_delta_field_path": dominant_feature.get("field_path") if isinstance(dominant_feature, dict) else None,
            "dominant_delta_direction": dominant_feature.get("direction") if isinstance(dominant_feature, dict) else None,
            "dominant_abs_delta": dominant_feature.get("abs_delta") if isinstance(dominant_feature, dict) else None,
            "changed_field_count": len(changed_fields),
            "structural_member_change": member_delta.get("changed") is True,
            "ray_count_delta": ray_delta,
            "temporal_trajectory_status": temporal_profile.get("trajectory_status"),
            "temporal_trajectory_frame_count": temporal_profile.get("frame_count"),
            "temporal_trajectory_multi_frame_available": temporal_profile.get("multi_frame_available"),
            "temporal_trajectory_path_distance_px": temporal_profile.get("path_distance_px"),
            "temporal_trajectory_bonus": round(temporal_bonus, 6),
        },
        "novelty_status": "novel_unique_relationship_candidate" if signature_count == 1 else "known_repeated_relationship_candidate",
        "novelty_score": round(1.0 / float(max(1, signature_count)), 6),
        "correlation_score": score,
        "confidence_score": score,
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "solve_like_evidence_status": CONTRA_SOLVE_LIKE_EVIDENCE_STATUS,
        "solve_like_evidence_flag": CONTRA_SOLVE_LIKE_EVIDENCE_FLAG,
        "solve_like_evidence_score": score,
        "solve_like_evidence_boundary": "candidate relative-state clue only; not solve truth",
        "action_state_positive_signal": CONTRA_ACTION_STATE_POSITIVE_SIGNAL,
        "tagger_policy": CONTRA_RELATIVE_STATE_TAG_POLICY,
        "action_state_tagger_boundary": CONTRA_ACTION_STATE_TAG_BOUNDARY,
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _member_ref_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int, float)):
        return [str(value)]
    if isinstance(value, list):
        return _unique_strings([item for row in value for item in _member_ref_values(row)])
    if isinstance(value, dict):
        return _unique_strings([item for row in value.values() for item in _member_ref_values(row)])
    return []


def _classify_interaction_candidate(
    *,
    interaction_kind: str,
    profile: dict[str, Any],
    mathematical_relationships: dict[str, Any],
    changed_fields: list[Any],
    member_delta: dict[str, Any],
) -> dict[str, Any]:
    families = set(str(row) for row in profile.get("dominant_relationship_families", []) or [])
    changed = set(str(row) for row in changed_fields or [])
    vector = profile.get("vector_delta") if isinstance(profile.get("vector_delta"), dict) else {}
    temporal = profile.get("temporal_trajectory") if isinstance(profile.get("temporal_trajectory"), dict) else {}
    shape = profile.get("shape_delta") if isinstance(profile.get("shape_delta"), dict) else {}
    topology = profile.get("topology_delta") if isinstance(profile.get("topology_delta"), dict) else {}
    color_value = profile.get("color_value_delta") if isinstance(profile.get("color_value_delta"), dict) else {}
    resource_gauge = profile.get("resource_gauge") if isinstance(profile.get("resource_gauge"), dict) else {}
    conservation = (
        profile.get("motion_energy_conservation")
        if isinstance(profile.get("motion_energy_conservation"), dict)
        else {}
    )
    reasons: list[str] = []
    focus: list[str] = []
    interaction_class = "candidate_relative_state_interaction"
    family = "generic_relative_change"

    if interaction_kind == "action_surface_contact_candidate_edge":
        interaction_class = "surface_contact_interaction"
        family = "contact_or_overlap"
        reasons.append("action surface result overlapped candidate object support")
        focus.append("inspect action-result surface contact and correlated object support")

    if "angle_relation_change" in families:
        interaction_class = "angular_alignment_interaction"
        family = "geometry_alignment"
        reasons.append("angle or bearing delta changed under the action boundary")
        focus.append("inspect angular error, target bearing, and orientation alignment")
    if "symmetry_relation_change" in families:
        interaction_class = "symmetry_alignment_interaction"
        family = "geometry_alignment"
        reasons.append("symmetry relationship delta changed under the action boundary")
        focus.append("inspect symmetry or rotational alignment before/after the action")
    if "shape_geometry_change" in families or shape.get("shape_change_status") == "changed":
        interaction_class = "shape_transform_interaction"
        family = "shape_geometry"
        reasons.append("shape geometry changed: width, height, area, aspect, or bbox diagonal")
        focus.append("inspect shape transform and component geometry changes")
    if "topological_relation_change" in families or topology.get("bbox_overlap_status") in {"overlap", "disjoint"}:
        if interaction_class == "candidate_relative_state_interaction":
            interaction_class = "topological_contact_interaction"
            family = "contact_or_overlap"
        reasons.append("topological relationship changed or measured through overlap/distance")
        focus.append("inspect bbox overlap, center distance, and contact/disjoint status")
    if "color_or_value_relation_change" in families or color_value.get("raw_value_changed") is True:
        interaction_class = "value_transition_interaction"
        family = "state_value_transition"
        reasons.append("raw value or color state transitioned across the action boundary")
        focus.append("inspect raw-value transition and value-specific component changes")
    if "resource_gauge_change" in families or resource_gauge.get("status") != "unavailable":
        if interaction_class == "candidate_relative_state_interaction":
            interaction_class = "resource_conservation_interaction"
            family = "resource_or_energy_conservation"
        reasons.append("resource gauge or fuel meter was observed across the action boundary")
        focus.append("inspect fuel/resource gauge delta alongside movement and paint/value deltas")
    if conservation.get("status") not in {"", None, "no_y_motion_observed"}:
        reasons.append("motion/resource conservation assessment produced a candidate Y-motion coupling status")
        focus.append("inspect whether Y motion is free from paint/resource debit before routing")
    if "temporal_trajectory_change" in families and temporal.get("status") != "unavailable":
        if interaction_class == "candidate_relative_state_interaction":
            interaction_class = "temporal_motion_interaction"
            family = "motion_flow"
        reasons.append("returned action frames contain a temporal trajectory profile")
        focus.append("inspect action-frame sequence path, direction, and peak changed pixels")
    if member_delta.get("changed") is True or "members" in changed:
        interaction_class = "membership_change_interaction"
        family = "membership_or_object_set"
        reasons.append("relationship member set changed across before/after state")
        focus.append("inspect object membership additions/removals and component identity changes")
    if any(field.startswith("game_state.") for field in changed):
        interaction_class = "game_state_transition_interaction"
        family = "game_state_transition"
        reasons.append("game-state numeric field changed under the action boundary")
        focus.append("inspect game-state counters, score-like deltas, and state flags")

    if not reasons:
        dominant_family = mathematical_relationships.get("dominant_delta_family")
        if dominant_family:
            reasons.append(f"dominant relationship family: {dominant_family}")
        else:
            reasons.append("candidate interaction was emitted without a more specific family")
        focus.append("inspect controlled before/after relation deltas")

    evidence_terms = _unique_strings(
        [
            f"interaction_kind:{interaction_kind}",
            f"class:{interaction_class}",
            f"family:{family}",
            f"vector:{vector.get('direction_8way')}:{vector.get('axis_family')}:{vector.get('magnitude_bucket')}",
            f"temporal:{temporal.get('trajectory_status')}:{temporal.get('path_distance_bucket')}",
            f"shape:{shape.get('shape_change_status')}",
            f"topology:{topology.get('bbox_overlap_status')}",
            f"value:{color_value.get('raw_value_status')}:{color_value.get('raw_value_transition')}",
            f"resource:{resource_gauge.get('resource_kind')}:{resource_gauge.get('gauge_status')}",
            f"conservation:{conservation.get('status')}:{conservation.get('y_motion_resource_coupling_status')}",
        ]
        + [f"field:{field}" for field in sorted(changed)[:8]]
    )
    return {
        "schema": CONTRA_ACTION_INTERACTION_CLASSIFICATION_SCHEMA,
        "status": "classified_candidate_interaction",
        "interaction_class": interaction_class,
        "interaction_class_family": family,
        "classification_reasons": _unique_strings(reasons),
        "recommended_review_focus": _unique_strings(focus),
        "evidence_terms": evidence_terms,
        "classification_policy": "derive_candidate_interaction_class_from_observed_relative_state_families_not_game_specific_rules",
        "confidence_is_probability": False,
        "evidence_status": "candidate_interaction_classification_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _interaction_edge_from_tag(tag: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tag, dict):
        return None
    tag_id = str(tag.get("action_state_tag_id") or "")
    if not tag_id:
        return None
    tag_kind = str(tag.get("tag_kind") or "unknown_action_state_tag")
    action_primitive = str(tag.get("action_primitive") or "unknown_action")
    math_rel = tag.get("mathematical_relationships") if isinstance(tag.get("mathematical_relationships"), dict) else {}
    source_ids: list[str] = []
    target_ids: list[str] = []
    participant_components = _unique_strings(list(tag.get("correlated_component_ids", []) or []))
    if tag_kind == "controlled_action_relative_state_change":
        member_delta = tag.get("member_delta") if isinstance(tag.get("member_delta"), dict) else {}
        before_ids = _member_ref_values(member_delta.get("before_members"))
        after_ids = _member_ref_values(member_delta.get("after_members"))
        source_ids = before_ids or _unique_strings([tag.get("before_statecon_id"), tag.get("statecon_id")])
        target_ids = _unique_strings([value for value in after_ids if value not in source_ids]) or after_ids or _unique_strings([tag.get("after_statecon_id"), tag.get("statecon_id")])
        interaction_kind = "relative_state_change_between_relationship_members"
    else:
        source_ids = _unique_strings(list(tag.get("correlated_object_ids", []) or []))
        target_ids = _unique_strings([tag.get("event_surface_source_object_id"), tag.get("event_surface_object_id")])
        if not source_ids:
            source_ids = _unique_strings([tag.get("portband_id"), tag.get("statecon_id")])
        if not target_ids:
            target_ids = _unique_strings([tag.get("event_surface_role"), tag.get("statecon_id")])
        interaction_kind = "action_surface_contact_candidate_edge"
    participant_ids = _unique_strings(source_ids + target_ids)
    score = round(max(0.0, min(1.0, float(tag.get("solve_like_evidence_score") or tag.get("confidence_score") or 0.0))), 6)
    classification = _classify_interaction_candidate(
        interaction_kind=interaction_kind,
        profile=math_rel.get("relationship_profile") if isinstance(math_rel.get("relationship_profile"), dict) else {},
        mathematical_relationships=math_rel,
        changed_fields=list(tag.get("changed_fields", []) or []),
        member_delta=tag.get("member_delta") if isinstance(tag.get("member_delta"), dict) else {},
    )
    return {
        "schema": CONTRA_ACTION_INTERACTION_EDGE_SCHEMA,
        "status": "candidate_interaction_edge",
        "interaction_edge_id": _stable_id("interaction_edge", tag_id, interaction_kind, participant_ids, math_rel.get("relationship_signature")),
        "interaction_kind": interaction_kind,
        "source_action_state_tag_id": tag_id,
        "statecon_id": tag.get("statecon_id"),
        "action_primitive": action_primitive,
        "action_invocation_id": tag.get("action_invocation_id"),
        "from_object_ids": source_ids,
        "to_object_ids": target_ids,
        "participant_object_ids": participant_ids,
        "participant_component_ids": participant_components,
        "relationship_signature": math_rel.get("relationship_signature"),
        "mathematical_invariant_signature": math_rel.get("mathematical_invariant_signature"),
        "mathematical_invariant_hash": math_rel.get("mathematical_invariant_hash"),
        "mathematical_invariant_terms": math_rel.get("mathematical_invariant_terms", []),
        "relation_type_label": math_rel.get("relation_type_label"),
        "relationship_profile": math_rel.get("relationship_profile"),
        "mathematical_relationships": math_rel,
        "interaction_classification": classification,
        "interaction_class": classification.get("interaction_class"),
        "interaction_class_family": classification.get("interaction_class_family"),
        "interaction_class_reasons": classification.get("classification_reasons", []),
        "recommended_review_focus": classification.get("recommended_review_focus", []),
        "changed_fields": tag.get("changed_fields", []),
        "novelty_status": tag.get("novelty_status"),
        "novelty_score": tag.get("novelty_score"),
        "solve_like_evidence_status": tag.get("solve_like_evidence_status"),
        "solve_like_evidence_flag": tag.get("solve_like_evidence_flag"),
        "solve_like_evidence_score": score,
        "confidence_score": score,
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "edge_policy": CONTRA_ACTION_INTERACTION_GRAPH_POLICY,
        "edge_boundary": "candidate interaction edge only; not solve truth or action authority",
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _build_action_interaction_graph(action_state_tags: list[dict[str, Any]]) -> dict[str, Any]:
    edges = [
        edge
        for tag in action_state_tags
        if isinstance(tag, dict)
        if (edge := _interaction_edge_from_tag(tag)) is not None
    ]
    edges.sort(
        key=lambda row: (
            -float(row.get("solve_like_evidence_score") or 0.0),
            str(row.get("interaction_kind")),
            str(row.get("interaction_edge_id")),
        )
    )
    for rank, row in enumerate(edges, start=1):
        row["interaction_rank"] = rank
    participant_ids = _unique_strings([value for edge in edges for value in edge.get("participant_object_ids", [])])
    novel_edges = [edge for edge in edges if edge.get("novelty_status") == "novel_unique_relationship_candidate"]
    clue_edges = [edge for edge in edges if edge.get("solve_like_evidence_flag")]
    return {
        "schema": CONTRA_ACTION_INTERACTION_GRAPH_SCHEMA,
        "status": "ranked" if edges else "no_candidate_interaction_edges",
        "graph_policy": CONTRA_ACTION_INTERACTION_GRAPH_POLICY,
        "edge_count": len(edges),
        "participant_object_count": len(participant_ids),
        "participant_object_ids": participant_ids,
        "interaction_kinds": _unique_strings([edge.get("interaction_kind") for edge in edges]),
        "novel_interaction_edge_count": len(novel_edges),
        "solve_like_interaction_edge_count": len(clue_edges),
        "top_solve_like_evidence_score": max((float(edge.get("solve_like_evidence_score") or 0.0) for edge in edges), default=0.0),
        "edges": edges,
        "edge_ids": _unique_strings([edge.get("interaction_edge_id") for edge in edges]),
        "evidence_status": "candidate_action_interaction_graph_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _candidate_route_hint(edge: dict[str, Any]) -> dict[str, Any]:
    kind = str(edge.get("interaction_kind") or "unknown_interaction")
    classification = edge.get("interaction_classification") if isinstance(edge.get("interaction_classification"), dict) else {}
    interaction_class = str(classification.get("interaction_class") or edge.get("interaction_class") or "candidate_interaction")
    review_focus = _unique_strings(list(classification.get("recommended_review_focus", []) or []))
    profile = edge.get("relationship_profile") if isinstance(edge.get("relationship_profile"), dict) else {}
    temporal = profile.get("temporal_trajectory") if isinstance(profile.get("temporal_trajectory"), dict) else {}
    has_temporal = temporal.get("status") not in {None, "", "unavailable"}
    if kind == "relative_state_change_between_relationship_members":
        if interaction_class == "angular_alignment_interaction":
            label = "inspect_angular_alignment_interaction"
            instruction = "Review the angle/bearing delta, correlated objects, and controlled before/after frames before considering a follow-up route."
        elif interaction_class == "shape_transform_interaction":
            label = "inspect_shape_transform_interaction"
            instruction = "Review the shape geometry delta, changed component support, and controlled frame sequence before considering a follow-up route."
        elif interaction_class == "value_transition_interaction":
            label = "inspect_value_transition_interaction"
            instruction = "Review raw-value/color transition and object correspondence before considering a follow-up route."
        elif interaction_class == "membership_change_interaction":
            label = "inspect_membership_change_interaction"
            instruction = "Review object membership additions/removals and whether the action boundary explains the changed set."
        elif interaction_class == "game_state_transition_interaction":
            label = "inspect_game_state_transition_interaction"
            instruction = "Review score/state numeric deltas alongside frame evidence before considering a follow-up route."
        elif interaction_class == "resource_conservation_interaction":
            label = "inspect_resource_conservation_interaction"
            instruction = "Review fuel/resource gauge stability or debit alongside motion, paint/value, and controlled before/after frames before considering a follow-up route."
        elif has_temporal:
            label = "inspect_temporal_relative_state_change"
            instruction = "Review the correlated before/after objects plus returned action-frame trajectory, then retest candidate actions under a controlled runner."
        else:
            label = "inspect_relative_state_change"
            instruction = "Review the correlated before/after objects and retest candidate actions under a controlled runner."
    elif kind == "action_surface_contact_candidate_edge":
        label = "inspect_action_surface_contact"
        instruction = "Review the action-result surface overlap and retest nearby object interactions under a controlled runner."
    else:
        label = "inspect_candidate_interaction"
        instruction = "Review the candidate interaction edge under a controlled runner before treating it as useful."
    return {
        "status": "candidate_review_route_not_action_authority",
        "label": label,
        "instruction": instruction,
        "interaction_class": interaction_class,
        "interaction_class_family": classification.get("interaction_class_family"),
        "classification_reasons": classification.get("classification_reasons", []),
        "recommended_review_focus": review_focus,
        "route_evidence_terms": classification.get("evidence_terms", []),
        "confidence_score": round(max(0.0, min(1.0, float(edge.get("confidence_score") or 0.0))), 6),
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "boundary": "review routing hint only; not action choice, solve truth, control truth, or prediction",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _action_interaction_matrix_row(edge: dict[str, Any], rank: int) -> dict[str, Any]:
    profile = edge.get("relationship_profile") if isinstance(edge.get("relationship_profile"), dict) else {}
    classification = edge.get("interaction_classification") if isinstance(edge.get("interaction_classification"), dict) else {}
    vector_delta = profile.get("vector_delta") if isinstance(profile.get("vector_delta"), dict) else {}
    correspondence = profile.get("object_correspondence") if isinstance(profile.get("object_correspondence"), dict) else {}
    temporal = profile.get("temporal_trajectory") if isinstance(profile.get("temporal_trajectory"), dict) else {}
    temporal_vector = temporal.get("trajectory_vector") if isinstance(temporal.get("trajectory_vector"), dict) else {}
    shape = profile.get("shape_delta") if isinstance(profile.get("shape_delta"), dict) else {}
    topology = profile.get("topology_delta") if isinstance(profile.get("topology_delta"), dict) else {}
    color_value = profile.get("color_value_delta") if isinstance(profile.get("color_value_delta"), dict) else {}
    resource_gauge = profile.get("resource_gauge") if isinstance(profile.get("resource_gauge"), dict) else {}
    conservation = (
        profile.get("motion_energy_conservation")
        if isinstance(profile.get("motion_energy_conservation"), dict)
        else {}
    )
    score = round(max(0.0, min(1.0, float(edge.get("confidence_score") or edge.get("solve_like_evidence_score") or 0.0))), 6)
    participant_ids = _unique_strings(list(edge.get("participant_object_ids", []) or []))
    return {
        "schema": CONTRA_ACTION_INTERACTION_MATRIX_ROW_SCHEMA,
        "status": "ranked_candidate_interaction",
        "interaction_matrix_row_id": _stable_id("interaction_matrix_row", edge.get("interaction_edge_id"), rank),
        "interaction_rank": int(rank),
        "interaction_edge_id": edge.get("interaction_edge_id"),
        "source_action_state_tag_id": edge.get("source_action_state_tag_id"),
        "action_primitive": edge.get("action_primitive"),
        "interaction_kind": edge.get("interaction_kind"),
        "from_object_ids": edge.get("from_object_ids", []),
        "to_object_ids": edge.get("to_object_ids", []),
        "participant_object_ids": participant_ids,
        "participant_component_ids": edge.get("participant_component_ids", []),
        "relationship_signature": edge.get("relationship_signature"),
        "mathematical_invariant_signature": edge.get("mathematical_invariant_signature") or profile.get("mathematical_invariant_signature"),
        "mathematical_invariant_hash": edge.get("mathematical_invariant_hash") or profile.get("mathematical_invariant_hash"),
        "mathematical_invariant_terms": edge.get("mathematical_invariant_terms") or profile.get("mathematical_invariant_terms", []),
        "relation_type_label": edge.get("relation_type_label"),
        "interaction_classification": classification,
        "interaction_class": classification.get("interaction_class") or edge.get("interaction_class"),
        "interaction_class_family": classification.get("interaction_class_family") or edge.get("interaction_class_family"),
        "interaction_class_reasons": classification.get("classification_reasons", edge.get("interaction_class_reasons", [])),
        "recommended_review_focus": classification.get("recommended_review_focus", edge.get("recommended_review_focus", [])),
        "signature_terms": profile.get("signature_terms", []),
        "dominant_relationship_families": profile.get("dominant_relationship_families", []),
        "vector_direction_8way": vector_delta.get("direction_8way"),
        "vector_axis_family": vector_delta.get("axis_family"),
        "vector_magnitude": vector_delta.get("euclidean_magnitude"),
        "temporal_trajectory_status": temporal.get("trajectory_status"),
        "temporal_trajectory_frame_count": temporal.get("frame_count"),
        "temporal_trajectory_multi_frame_available": temporal.get("multi_frame_available"),
        "temporal_trajectory_direction_8way": temporal_vector.get("direction_8way"),
        "temporal_trajectory_axis_family": temporal_vector.get("axis_family"),
        "temporal_trajectory_magnitude": temporal_vector.get("euclidean_magnitude"),
        "temporal_trajectory_magnitude_bucket": temporal_vector.get("magnitude_bucket"),
        "temporal_trajectory_path_distance_px": temporal.get("path_distance_px"),
        "temporal_trajectory_peak_changed_pixel_count": temporal.get("peak_changed_pixel_count"),
        "temporal_trajectory_evidence_status": temporal.get("evidence_status"),
        "shape_change_status": shape.get("shape_change_status"),
        "shape_area_delta_pixels": shape.get("area_delta_pixels"),
        "shape_area_ratio_before_after": shape.get("area_ratio_before_after"),
        "shape_aspect_ratio_delta": shape.get("aspect_ratio_delta"),
        "topology_bbox_overlap_status": topology.get("bbox_overlap_status"),
        "topology_bbox_iou_before_after": topology.get("bbox_iou_before_after"),
        "topology_center_distance_pixels": topology.get("center_distance_pixels"),
        "color_raw_value_transition": color_value.get("raw_value_transition"),
        "color_raw_value_changed": color_value.get("raw_value_changed"),
        "resource_gauge_status": resource_gauge.get("gauge_status"),
        "resource_kind": resource_gauge.get("resource_kind"),
        "resource_gauge_delta_total": resource_gauge.get("delta_total"),
        "resource_gauge_delta_bucket": resource_gauge.get("delta_bucket"),
        "resource_gauge_changed": resource_gauge.get("resource_changed"),
        "resource_gauge_field_paths": resource_gauge.get("gauge_field_paths", []),
        "motion_energy_conservation_status": conservation.get("status"),
        "y_motion_pixels": conservation.get("y_motion_pixels"),
        "y_motion_resource_coupling_status": conservation.get("y_motion_resource_coupling_status"),
        "y_motion_paint_coupling_status": conservation.get("y_motion_paint_coupling_status"),
        "paint_or_value_motion_present": conservation.get("paint_or_value_motion_present"),
        "motion_energy_assessment_basis": conservation.get("assessment_basis", []),
        "candidate_pair_index": correspondence.get("candidate_pair_index"),
        "selected_pair_rank": correspondence.get("selected_pair_rank"),
        "object_correspondence_confidence_score": correspondence.get("confidence_score"),
        "object_correspondence_confidence_band": correspondence.get("confidence_band"),
        "novelty_status": edge.get("novelty_status"),
        "novelty_score": edge.get("novelty_score"),
        "solve_like_evidence_status": edge.get("solve_like_evidence_status"),
        "solve_like_evidence_flag": edge.get("solve_like_evidence_flag"),
        "solve_like_evidence_score": edge.get("solve_like_evidence_score"),
        "confidence_score": score,
        "confidence_band": _score_band(score),
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "routing_recommendation": _candidate_route_hint(edge),
        "matrix_policy": CONTRA_ACTION_INTERACTION_MATRIX_POLICY,
        "evidence_status": "candidate_action_interaction_matrix_row_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _build_action_interaction_matrix(action_interaction_graph: dict[str, Any]) -> dict[str, Any]:
    edges = [edge for edge in action_interaction_graph.get("edges", []) or [] if isinstance(edge, dict)]
    rows = [_action_interaction_matrix_row(edge, rank) for rank, edge in enumerate(edges, start=1)]
    rows.sort(
        key=lambda row: (
            -float(row.get("confidence_score") or 0.0),
            str(row.get("interaction_kind")),
            str(row.get("interaction_matrix_row_id")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["interaction_rank"] = rank
    participant_ids = _unique_strings([value for row in rows for value in row.get("participant_object_ids", [])])
    best_by_support: list[dict[str, Any]] = []
    seen: set[str] = set()
    for support_id in participant_ids:
        candidates = [row for row in rows if support_id in set(row.get("participant_object_ids", []) or [])]
        if not candidates or support_id in seen:
            continue
        seen.add(support_id)
        best = candidates[0]
        best_by_support.append(
            {
                "support_object_id": support_id,
                "best_interaction_matrix_row_id": best.get("interaction_matrix_row_id"),
                "interaction_edge_id": best.get("interaction_edge_id"),
                "interaction_kind": best.get("interaction_kind"),
                "interaction_class": best.get("interaction_class"),
                "interaction_class_family": best.get("interaction_class_family"),
                "recommended_review_focus": best.get("recommended_review_focus", []),
                "action_primitive": best.get("action_primitive"),
                "confidence_score": best.get("confidence_score"),
                "confidence_band": best.get("confidence_band"),
                "relation_type_label": best.get("relation_type_label"),
                "relationship_signature": best.get("relationship_signature"),
                "temporal_trajectory_direction_8way": best.get("temporal_trajectory_direction_8way"),
                "temporal_trajectory_path_distance_px": best.get("temporal_trajectory_path_distance_px"),
                "temporal_trajectory_multi_frame_available": best.get("temporal_trajectory_multi_frame_available"),
                "resource_gauge_status": best.get("resource_gauge_status"),
                "resource_gauge_delta_total": best.get("resource_gauge_delta_total"),
                "motion_energy_conservation_status": best.get("motion_energy_conservation_status"),
                "y_motion_resource_coupling_status": best.get("y_motion_resource_coupling_status"),
                "solve_like_evidence_flag": best.get("solve_like_evidence_flag"),
                "routing_recommendation": best.get("routing_recommendation"),
            }
        )
    return {
        "schema": CONTRA_ACTION_INTERACTION_MATRIX_SCHEMA,
        "status": "ranked" if rows else "no_candidate_interaction_rows",
        "matrix_policy": CONTRA_ACTION_INTERACTION_MATRIX_POLICY,
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "row_count": len(rows),
        "support_object_count": len(participant_ids),
        "support_object_ids": participant_ids,
        "best_by_support_object": best_by_support,
        "rows": rows,
        "evidence_status": "candidate_action_interaction_matrix_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _solve_like_candidate_ledger_row(matrix_row: dict[str, Any], rank: int) -> dict[str, Any]:
    route_hint = matrix_row.get("routing_recommendation") if isinstance(matrix_row.get("routing_recommendation"), dict) else {}
    score = round(max(0.0, min(1.0, float(matrix_row.get("solve_like_evidence_score") or matrix_row.get("confidence_score") or 0.0))), 6)
    novelty = str(matrix_row.get("novelty_status") or "")
    invariant_hash = str(matrix_row.get("mathematical_invariant_hash") or "")
    evidence_terms = _unique_strings(
        list(matrix_row.get("mathematical_invariant_terms", []) or [])[:10]
        + list(route_hint.get("route_evidence_terms", []) or [])[:10]
        + list(matrix_row.get("signature_terms", []) or [])[:10]
    )
    return {
        "schema": CONTRA_SOLVE_LIKE_CANDIDATE_LEDGER_ROW_SCHEMA,
        "status": "candidate_solve_like_clue",
        "solve_like_candidate_id": _stable_id(
            "solve_like_candidate",
            matrix_row.get("interaction_matrix_row_id"),
            matrix_row.get("interaction_edge_id"),
            matrix_row.get("relationship_signature"),
            invariant_hash,
        ),
        "candidate_rank": int(rank),
        "interaction_matrix_row_id": matrix_row.get("interaction_matrix_row_id"),
        "interaction_edge_id": matrix_row.get("interaction_edge_id"),
        "action_primitive": matrix_row.get("action_primitive"),
        "interaction_kind": matrix_row.get("interaction_kind"),
        "interaction_class": matrix_row.get("interaction_class"),
        "interaction_class_family": matrix_row.get("interaction_class_family"),
        "participant_object_ids": matrix_row.get("participant_object_ids", []),
        "participant_component_ids": matrix_row.get("participant_component_ids", []),
        "candidate_pair_index": matrix_row.get("candidate_pair_index"),
        "selected_pair_rank": matrix_row.get("selected_pair_rank"),
        "relationship_signature": matrix_row.get("relationship_signature"),
        "relation_type_label": matrix_row.get("relation_type_label"),
        "mathematical_invariant_hash": invariant_hash or None,
        "mathematical_invariant_signature": matrix_row.get("mathematical_invariant_signature"),
        "mathematical_invariant_terms": matrix_row.get("mathematical_invariant_terms", []),
        "novelty_status": novelty,
        "novelty_score": matrix_row.get("novelty_score"),
        "is_unique_relationship_candidate": novelty == "novel_unique_relationship_candidate",
        "solve_like_evidence_status": matrix_row.get("solve_like_evidence_status"),
        "solve_like_evidence_flag": matrix_row.get("solve_like_evidence_flag"),
        "solve_like_evidence_score": score,
        "confidence_score": score,
        "confidence_band": _score_band(score),
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "routing_recommendation": route_hint,
        "recommended_review_focus": matrix_row.get("recommended_review_focus", route_hint.get("recommended_review_focus", [])),
        "evidence_terms": evidence_terms,
        "ledger_policy": "rank_candidate_interaction_clues_from_action_interaction_matrix_without_admitting_solve_truth",
        "evidence_status": "candidate_solve_like_ledger_row_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _build_solve_like_candidate_ledger(action_interaction_matrix: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in action_interaction_matrix.get("rows", []) or []
        if isinstance(row, dict) and row.get("solve_like_evidence_flag") == CONTRA_SOLVE_LIKE_EVIDENCE_FLAG
    ]
    rows.sort(
        key=lambda row: (
            0 if row.get("novelty_status") == "novel_unique_relationship_candidate" else 1,
            -float(row.get("solve_like_evidence_score") or row.get("confidence_score") or 0.0),
            str(row.get("interaction_matrix_row_id")),
        )
    )
    ledger_rows = [_solve_like_candidate_ledger_row(row, rank) for rank, row in enumerate(rows, start=1)]
    unique_rows = [row for row in ledger_rows if row.get("is_unique_relationship_candidate") is True]
    return {
        "schema": CONTRA_SOLVE_LIKE_CANDIDATE_LEDGER_SCHEMA,
        "status": "ranked" if ledger_rows else "no_candidate_solve_like_clues",
        "ledger_policy": "rank_candidate_interaction_clues_for_review_not_solve_truth_or_action_authority",
        "row_count": len(ledger_rows),
        "unique_candidate_count": len(unique_rows),
        "interaction_class_counts": dict(sorted({klass: sum(1 for row in ledger_rows if row.get("interaction_class") == klass) for klass in _unique_strings([row.get("interaction_class") for row in ledger_rows])}.items())),
        "top_solve_like_evidence_score": max((float(row.get("solve_like_evidence_score") or 0.0) for row in ledger_rows), default=0.0),
        "confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "rows": ledger_rows,
        "top_rows": ledger_rows[:8],
        "evidence_status": "candidate_solve_like_ledger_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _route_flow_confidence_score(
    *,
    static_score: float,
    dynamic_score: float,
    adjusted_score: float,
    controlled_events: list[dict[str, Any]],
) -> float:
    event_support = max(
        (
            float(row.get("dynamic_event_score") or 0.0)
            * max(float(row.get("source_overlap_ratio") or 0.0), float(row.get("target_overlap_ratio") or 0.0))
            for row in controlled_events
        ),
        default=0.0,
    )
    score = max(
        float(adjusted_score),
        (0.52 * event_support) + (0.28 * float(dynamic_score)) + (0.20 * float(static_score)),
    )
    return round(max(0.0, min(1.0, score)), 6)


def _route_flow_lock_record(
    *,
    alignment: dict[str, Any],
    controlled_events: list[dict[str, Any]],
    static_score: float,
    dynamic_score: float,
    adjusted_score: float,
) -> dict[str, Any] | None:
    if not controlled_events:
        return None
    event_ids = _unique_strings([row.get("event_id") for row in controlled_events])
    action_invocation_ids = _unique_strings([row.get("action_invocation_id") for row in controlled_events])
    confidence_score = _route_flow_confidence_score(
        static_score=static_score,
        dynamic_score=dynamic_score,
        adjusted_score=adjusted_score,
        controlled_events=controlled_events,
    )
    portband_id = str(alignment.get("portband_id") or "")
    return {
        "schema": PORTBAND_ROUTE_FLOW_LOCK_SCHEMA,
        "status": PORTBAND_ROUTE_FLOW_LOCK_STATUS,
        "route_flow_positive_signal": PORTBAND_ROUTE_FLOW_POSITIVE_SIGNAL,
        "route_flow_lock_id": _stable_id("route_flow_lock", alignment.get("statecon_id"), portband_id, event_ids, action_invocation_ids),
        "statecon_id": alignment.get("statecon_id"),
        "portband_id": portband_id,
        "linked_alignment_ids": [portband_id] if portband_id else [],
        "confidence_score": confidence_score,
        "confidence_model": PORTBAND_ROUTE_FLOW_LOCK_CONFIDENCE_MODEL,
        "confidence_is_probability": False,
        "evidence_event_ids": event_ids,
        "action_invocation_ids": action_invocation_ids,
        "action_primitives": _unique_strings([row.get("action_primitive") for row in controlled_events]),
        "evidence_statuses": _unique_strings([row.get("evidence_status") for row in controlled_events]),
        "matched_event_count": len(controlled_events),
        "best_dynamic_event_score": max((float(row.get("dynamic_event_score") or 0.0) for row in controlled_events), default=0.0),
        "best_source_overlap_ratio": max((float(row.get("source_overlap_ratio") or 0.0) for row in controlled_events), default=0.0),
        "best_target_overlap_ratio": max((float(row.get("target_overlap_ratio") or 0.0) for row in controlled_events), default=0.0),
        "lock_policy": PORTBAND_ROUTE_FLOW_LOCK_POLICY,
        "route_flow_lock_boundary": PORTBAND_ROUTE_FLOW_LOCK_BOUNDARY,
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def build_portband_surface_feedback(
    *,
    portband_hypotheses: dict[str, Any],
    action_surface_result_events: list[dict[str, Any]],
    action_state_relationship_observations: list[dict[str, Any]] | None = None,
    action_primitive: str | None = None,
) -> dict[str, Any]:
    """Rank PortBand alignments using observed DeltaBand surface-result events.

    This is a fail-safe calibration layer: it annotates and ranks static
    geometry hypotheses but never admits prediction, control, goal, or solve
    truth.
    """

    alignments = [row for row in portband_hypotheses.get("alignments", []) if isinstance(row, dict)] if isinstance(portband_hypotheses, dict) else []
    events = [row for row in action_surface_result_events if isinstance(row, dict)]
    relationship_observations = [row for row in (action_state_relationship_observations or []) if isinstance(row, dict)]
    signature_counts = _relationship_novelty_map(observations=relationship_observations, action_primitive=action_primitive)
    relationship_tags = [
        tag
        for row in relationship_observations
        if (
            tag := _relationship_action_state_tag_record(
                portband_hypotheses=portband_hypotheses,
                observation=row,
                action_primitive=action_primitive,
                signature_counts=signature_counts,
            )
        )
        is not None
    ]
    ranked: list[dict[str, Any]] = []
    for alignment in alignments:
        source_boxes = [box for box in alignment.get("source_bboxes_xyxy", []) if isinstance(box, list) and len(box) == 4]
        target_boxes = [box for box in alignment.get("target_bboxes_xyxy", []) if isinstance(box, list) and len(box) == 4]
        matched_events: list[dict[str, Any]] = []
        for event in events:
            event_box = _event_surface_bbox(event)
            if event_box is None:
                continue
            source_overlap = max((_bbox_overlap_ratio([int(v) for v in box], event_box) for box in source_boxes), default=0.0)
            target_overlap = max((_bbox_overlap_ratio([int(v) for v in box], event_box) for box in target_boxes), default=0.0)
            if max(source_overlap, target_overlap) <= 0.0:
                continue
            matched_events.append(
                {
                    "event_id": event.get("event_id"),
                    "event_rank": event.get("event_rank"),
                    "action_invocation_id": event.get("action_invocation_id"),
                    "action_primitive": event.get("action_primitive"),
                    "evidence_status": event.get("evidence_status"),
                    "surface_object_id": (event.get("surface") or {}).get("surface_object_id") if isinstance(event.get("surface"), dict) else None,
                    "surface_source_object_id": (event.get("surface") or {}).get("source_object_id") if isinstance(event.get("surface"), dict) else None,
                    "surface_component_ids": (event.get("surface") or {}).get("component_ids") if isinstance(event.get("surface"), dict) else [],
                    "surface_role": (event.get("surface") or {}).get("surface_role") if isinstance(event.get("surface"), dict) else None,
                    "surface_bbox_xyxy": event_box,
                    "dominant_side": (event.get("surface_edge") or {}).get("dominant_side") if isinstance(event.get("surface_edge"), dict) else None,
                    "contact_frame_count": (event.get("contact") or {}).get("contact_frame_count") if isinstance(event.get("contact"), dict) else None,
                    "max_surface_contact_cell_count": (event.get("contact") or {}).get("max_surface_contact_cell_count") if isinstance(event.get("contact"), dict) else None,
                    "final_changed_pixel_count": (event.get("result_delta") or {}).get("final_changed_pixel_count") if isinstance(event.get("result_delta"), dict) else None,
                    "event_confidence": event.get("confidence"),
                    "source_overlap_ratio": source_overlap,
                    "target_overlap_ratio": target_overlap,
                    "dynamic_event_score": _event_dynamic_score(event),
                }
            )
        static_score = _alignment_static_score(alignment)
        dynamic_score = max((float(row.get("dynamic_event_score") or 0.0) * max(float(row.get("source_overlap_ratio") or 0.0), 0.65 * float(row.get("target_overlap_ratio") or 0.0)) for row in matched_events), default=0.0)
        adjusted_score = round(max(0.0, min(1.0, (0.62 * dynamic_score) + (0.38 * static_score))), 6)
        action_state_tags = [
            tag
            for row in matched_events
            if (tag := _action_state_tag_record(alignment=alignment, matched_event=row, static_score=static_score)) is not None
        ]
        action_state_score = max((float(tag.get("correlation_score") or 0.0) for tag in action_state_tags), default=0.0)
        controlled_route_flow_events = [row for row in matched_events if _is_controlled_route_flow_event(row)]
        route_flow_lock = _route_flow_lock_record(
            alignment=alignment,
            controlled_events=controlled_route_flow_events,
            static_score=static_score,
            dynamic_score=round(dynamic_score, 6),
            adjusted_score=adjusted_score,
        )
        route_flow_lock_status = PORTBAND_ROUTE_FLOW_LOCK_STATUS if isinstance(route_flow_lock, dict) else PORTBAND_ROUTE_FLOW_NO_LOCK_STATUS
        annotated = {
            **alignment,
            "surface_feedback_status": "matched_delta_surface_result_event" if matched_events else "no_delta_surface_result_event_match",
            "surface_feedback_policy": "promote_static_portband_rows_using_deltaband_surface_result_without_admitting_truth",
            "surface_feedback_static_score": static_score,
            "surface_feedback_dynamic_score": round(dynamic_score, 6),
            "surface_feedback_adjusted_score": adjusted_score,
            "surface_feedback_event_count": len(matched_events),
            "surface_feedback_events": matched_events[:8],
            "action_state_tag_status": "has_observed_action_state_tags" if action_state_tags else "no_observed_action_state_tags",
            "action_state_tag_count": len(action_state_tags),
            "action_state_tags": action_state_tags[:8],
            "action_state_tag_ids": _unique_strings([tag.get("action_state_tag_id") for tag in action_state_tags]),
            "action_state_correlation_score": round(action_state_score, 6),
            "action_state_confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
            "action_state_confidence_is_probability": False,
            "action_state_positive_signal": CONTRA_ACTION_STATE_POSITIVE_SIGNAL if action_state_tags else "none",
            "action_state_primitives": _unique_strings([tag.get("action_primitive") for tag in action_state_tags]),
            "action_state_event_ids": _unique_strings([tag.get("event_id") for tag in action_state_tags]),
            "action_state_action_invocation_ids": _unique_strings([tag.get("action_invocation_id") for tag in action_state_tags]),
            "action_state_tagger_policy": CONTRA_ACTION_STATE_TAG_POLICY,
            "action_state_tagger_boundary": CONTRA_ACTION_STATE_TAG_BOUNDARY,
            "route_flow_lock": route_flow_lock,
            "route_flow_lock_status": route_flow_lock_status,
            "route_flow_positive_signal": PORTBAND_ROUTE_FLOW_POSITIVE_SIGNAL if isinstance(route_flow_lock, dict) else "none",
            "route_flow_confidence_score": route_flow_lock.get("confidence_score") if isinstance(route_flow_lock, dict) else 0.0,
            "route_flow_confidence_model": PORTBAND_ROUTE_FLOW_LOCK_CONFIDENCE_MODEL,
            "route_flow_confidence_is_probability": False,
            "route_flow_evidence_event_ids": route_flow_lock.get("evidence_event_ids", []) if isinstance(route_flow_lock, dict) else [],
            "route_flow_action_invocation_ids": route_flow_lock.get("action_invocation_ids", []) if isinstance(route_flow_lock, dict) else [],
            "route_flow_lock_policy": PORTBAND_ROUTE_FLOW_LOCK_POLICY,
            "route_flow_lock_boundary": PORTBAND_ROUTE_FLOW_LOCK_BOUNDARY,
            "prediction_status": "not_predictive",
            "admission_status": "not_admitted",
            "not_prediction": True,
            "not_control_truth": True,
            "not_solve_truth": True,
            "control_truth_admitted": False,
            "goal_condition_admitted": False,
            "action_choice_admitted": False,
            "solve_condition_admitted": False,
        }
        ranked.append(annotated)
    ranked.sort(
        key=lambda row: (
            -float(row.get("surface_feedback_adjusted_score") or 0.0),
            -float(row.get("surface_feedback_dynamic_score") or 0.0),
            float(row.get("alignment_error_px") or 999.0),
            str(row.get("portband_id")),
        )
    )
    for rank, row in enumerate(ranked, start=1):
        row["surface_feedback_rank"] = rank
    route_flow_locks = [row["route_flow_lock"] for row in ranked if isinstance(row.get("route_flow_lock"), dict)]
    action_state_tags = [
        tag
        for row in ranked
        for tag in row.get("action_state_tags", [])
        if isinstance(tag, dict)
    ] + relationship_tags
    action_interaction_graph = _build_action_interaction_graph(action_state_tags)
    action_interaction_matrix = _build_action_interaction_matrix(action_interaction_graph)
    solve_like_candidate_ledger = _build_solve_like_candidate_ledger(action_interaction_matrix)
    return {
        "schema": PORTBAND_SURFACE_FEEDBACK_SCHEMA,
        "status": "ranked" if ranked else "no_portband_alignments",
        "statecon_id": portband_hypotheses.get("statecon_id") if isinstance(portband_hypotheses, dict) else None,
        "definition": "deltaband_surface_result_feedback_for_static_portband_alignment_ranking",
        "alignment_count": len(alignments),
        "ranked_alignment_count": len(ranked),
        "matched_alignment_count": sum(1 for row in ranked if int(row.get("surface_feedback_event_count") or 0) > 0),
        "action_surface_result_event_count": len(events),
        "ranked_alignments": ranked,
        "action_state_tagger_schema": CONTRA_ACTION_STATE_TAG_SET_SCHEMA,
        "action_state_tagger_policy": CONTRA_ACTION_STATE_TAG_POLICY,
        "action_state_tag_count": len(action_state_tags),
        "action_state_positive_signal_count": len(action_state_tags),
        "relative_state_change_tag_count": len(relationship_tags),
        "relative_state_change_tag_ids": _unique_strings([tag.get("action_state_tag_id") for tag in relationship_tags]),
        "relative_state_change_signature_count": len(signature_counts),
        "relative_state_change_unique_signature_count": sum(1 for count in signature_counts.values() if int(count) == 1),
        "novel_relationship_candidate_count": sum(1 for tag in relationship_tags if tag.get("novelty_status") == "novel_unique_relationship_candidate"),
        "solve_like_evidence_flag_count": sum(1 for tag in action_state_tags if tag.get("solve_like_evidence_flag")),
        "top_solve_like_evidence_score": max((float(tag.get("solve_like_evidence_score") or 0.0) for tag in action_state_tags), default=0.0),
        "action_state_tag_ids": _unique_strings([tag.get("action_state_tag_id") for tag in action_state_tags]),
        "action_state_primitives": _unique_strings([tag.get("action_primitive") for tag in action_state_tags]),
        "action_state_event_ids": _unique_strings([tag.get("event_id") for tag in action_state_tags]),
        "action_state_action_invocation_ids": _unique_strings([tag.get("action_invocation_id") for tag in action_state_tags]),
        "action_state_tags": action_state_tags,
        "action_interaction_graph": action_interaction_graph,
        "action_interaction_edge_count": action_interaction_graph.get("edge_count"),
        "action_interaction_edge_ids": action_interaction_graph.get("edge_ids", []),
        "action_interaction_kinds": action_interaction_graph.get("interaction_kinds", []),
        "novel_interaction_edge_count": action_interaction_graph.get("novel_interaction_edge_count"),
        "solve_like_interaction_edge_count": action_interaction_graph.get("solve_like_interaction_edge_count"),
        "action_interaction_matrix": action_interaction_matrix,
        "action_interaction_matrix_schema": action_interaction_matrix.get("schema"),
        "action_interaction_matrix_policy": action_interaction_matrix.get("matrix_policy"),
        "action_interaction_matrix_row_count": action_interaction_matrix.get("row_count"),
        "action_interaction_support_object_count": action_interaction_matrix.get("support_object_count"),
        "action_interaction_support_object_ids": action_interaction_matrix.get("support_object_ids", []),
        "action_interaction_best_by_support_object": action_interaction_matrix.get("best_by_support_object", []),
        "solve_like_candidate_ledger": solve_like_candidate_ledger,
        "solve_like_candidate_ledger_schema": solve_like_candidate_ledger.get("schema"),
        "solve_like_candidate_ledger_row_count": solve_like_candidate_ledger.get("row_count"),
        "solve_like_candidate_ledger_unique_count": solve_like_candidate_ledger.get("unique_candidate_count"),
        "solve_like_candidate_ledger_top_rows": solve_like_candidate_ledger.get("top_rows", []),
        "action_state_confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "action_state_confidence_is_probability": False,
        "action_state_tagger_boundary": CONTRA_ACTION_STATE_TAG_BOUNDARY,
        "route_flow_lock_policy": PORTBAND_ROUTE_FLOW_LOCK_POLICY,
        "route_flow_lock_count": len(route_flow_locks),
        "route_flow_positive_signal_count": len(route_flow_locks),
        "route_flow_locked_alignment_ids": _unique_strings([lock.get("portband_id") for lock in route_flow_locks]),
        "route_flow_locks": route_flow_locks,
        "route_flow_confidence_model": PORTBAND_ROUTE_FLOW_LOCK_CONFIDENCE_MODEL,
        "route_flow_confidence_is_probability": False,
        "route_flow_lock_boundary": PORTBAND_ROUTE_FLOW_LOCK_BOUNDARY,
        "feedback_policy": "fail_safe_ranking_overlay_original_static_alignments_preserved",
        "evidence_status": "deltaband_surface_result_feedback_not_truth",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _route_flow_adjusted_lineup_score(base_score: float, lock_score: float) -> float:
    base = max(0.0, min(1.0, float(base_score)))
    lock = max(0.0, min(1.0, float(lock_score)))
    return round(max(base, (0.42 * base) + (0.58 * lock)), 6)


def _action_state_adjusted_lineup_score(base_score: float, tag_score: float) -> float:
    base = max(0.0, min(1.0, float(base_score)))
    tag = max(0.0, min(1.0, float(tag_score)))
    return round(max(base, (0.48 * base) + (0.52 * tag)), 6)


def _action_state_tag_map(feedback: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    tags_by_alignment: dict[str, list[dict[str, Any]]] = {}
    for tag in feedback.get("action_state_tags", []) or []:
        if not isinstance(tag, dict):
            continue
        alignment_ids = _unique_strings([tag.get("portband_id")] + list(tag.get("linked_alignment_ids", []) or []))
        for alignment_id in alignment_ids:
            tags_by_alignment.setdefault(alignment_id, []).append(tag)
    for rows in tags_by_alignment.values():
        rows.sort(key=lambda tag: -float(tag.get("confidence_score") or tag.get("correlation_score") or 0.0))
    return tags_by_alignment


def _interaction_edge_map(feedback: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    graph = feedback.get("action_interaction_graph") if isinstance(feedback.get("action_interaction_graph"), dict) else {}
    edges_by_tag: dict[str, list[dict[str, Any]]] = {}
    for edge in graph.get("edges", []) or []:
        if not isinstance(edge, dict):
            continue
        tag_id = str(edge.get("source_action_state_tag_id") or "")
        if tag_id:
            edges_by_tag.setdefault(tag_id, []).append(edge)
    for rows in edges_by_tag.values():
        rows.sort(key=lambda edge: -float(edge.get("solve_like_evidence_score") or 0.0))
    return edges_by_tag


def _interaction_matrix_row_map(feedback: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    matrix = feedback.get("action_interaction_matrix") if isinstance(feedback.get("action_interaction_matrix"), dict) else {}
    rows_by_edge: dict[str, list[dict[str, Any]]] = {}
    for row in matrix.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        edge_id = str(row.get("interaction_edge_id") or "")
        if edge_id:
            rows_by_edge.setdefault(edge_id, []).append(row)
    for rows in rows_by_edge.values():
        rows.sort(key=lambda row: -float(row.get("confidence_score") or 0.0))
    return rows_by_edge


def _route_flow_lock_map(feedback: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lock_by_alignment: dict[str, dict[str, Any]] = {}
    for lock in feedback.get("route_flow_locks", []) or []:
        if not isinstance(lock, dict):
            continue
        alignment_ids = _unique_strings([lock.get("portband_id")] + list(lock.get("linked_alignment_ids", []) or []))
        for alignment_id in alignment_ids:
            existing = lock_by_alignment.get(alignment_id)
            if existing is None or float(lock.get("confidence_score") or 0.0) > float(existing.get("confidence_score") or 0.0):
                lock_by_alignment[alignment_id] = lock
    return lock_by_alignment


def _apply_action_state_feedback_to_lineup_matrix(
    *,
    lineup_matrix: dict[str, Any] | None,
    feedback: dict[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(lineup_matrix, dict):
        return lineup_matrix
    locks_by_alignment = _route_flow_lock_map(feedback)
    tags_by_alignment = _action_state_tag_map(feedback)
    edges_by_tag = _interaction_edge_map(feedback)
    matrix_rows_by_edge = _interaction_matrix_row_map(feedback)
    statecon_level_tags = [
        tag
        for tag in feedback.get("action_state_tags", []) or []
        if isinstance(tag, dict) and tag.get("tag_scope") == "statecon_relation"
    ]
    rows: list[dict[str, Any]] = []
    for row in lineup_matrix.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        linked_alignment_ids = _unique_strings(list(row.get("linked_alignment_ids", []) or []))
        matching_locks = [locks_by_alignment[alignment_id] for alignment_id in linked_alignment_ids if alignment_id in locks_by_alignment]
        matching_locks.sort(key=lambda lock: -float(lock.get("confidence_score") or 0.0))
        lock = matching_locks[0] if matching_locks else None
        matching_tags = [
            tag
            for alignment_id in linked_alignment_ids
            for tag in tags_by_alignment.get(alignment_id, [])
            if isinstance(tag, dict)
        ] + statecon_level_tags
        matching_tags.sort(key=lambda tag: -float(tag.get("confidence_score") or tag.get("correlation_score") or 0.0))
        best_tag = matching_tags[0] if matching_tags else None
        matching_edges = [
            edge
            for tag in matching_tags
            for edge in edges_by_tag.get(str(tag.get("action_state_tag_id") or ""), [])
            if isinstance(edge, dict)
        ]
        matching_edges.sort(key=lambda edge: -float(edge.get("solve_like_evidence_score") or 0.0))
        matching_matrix_rows = [
            matrix_row
            for edge in matching_edges
            for matrix_row in matrix_rows_by_edge.get(str(edge.get("interaction_edge_id") or ""), [])
            if isinstance(matrix_row, dict)
        ]
        matching_matrix_rows.sort(key=lambda matrix_row: -float(matrix_row.get("confidence_score") or 0.0))
        base_score = round(
            max(
                0.0,
                min(
                    1.0,
                    max(
                        float(row.get("edge_lock_adjusted_confidence_score") or 0.0),
                        float(row.get("confidence_score") or 0.0),
                    ),
                ),
            ),
            6,
        )
        new_row = dict(row)
        new_row.setdefault("edge_lock_status", "no_edge_lock_configuration")
        new_row.setdefault("edge_lock_configuration_ids", [])
        new_row.setdefault("edge_lock_top_configuration_id", None)
        new_row.setdefault("edge_lock_rank", None)
        new_row.setdefault("edge_lock_quality_score", 0.0)
        new_row.setdefault("edge_lock_adjusted_confidence_score", base_score)
        new_row.setdefault("edge_lock_confidence_model", PORTBAND_EDGE_LOCK_CONFIDENCE_MODEL)
        new_row.setdefault("edge_lock_confidence_is_probability", False)
        new_row.setdefault("edge_lock_policy", PORTBAND_EDGE_LOCK_POLICY)
        new_row.setdefault("edge_lock_boundary", PORTBAND_EDGE_LOCK_BOUNDARY)
        new_row.setdefault("potential_solution_signal_status", "no_candidate_shape_configuration_potential")
        action_state_score = round(max(0.0, min(1.0, float(best_tag.get("confidence_score") or 0.0))), 6) if isinstance(best_tag, dict) else 0.0
        action_state_adjusted_score = _action_state_adjusted_lineup_score(base_score, action_state_score) if isinstance(best_tag, dict) else base_score
        solve_like_score = round(max((float(tag.get("solve_like_evidence_score") or 0.0) for tag in matching_tags), default=0.0), 6)
        new_row.update(
            {
                "action_state_tag_status": "has_observed_action_state_tags" if matching_tags else "no_observed_action_state_tags",
                "action_state_positive_signal": CONTRA_ACTION_STATE_POSITIVE_SIGNAL if matching_tags else "none",
                "action_state_tag_ids": _unique_strings([tag.get("action_state_tag_id") for tag in matching_tags]),
                "action_state_tag_count": len(matching_tags),
                "action_state_correlation_score": action_state_score,
                "action_state_confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
                "action_state_confidence_is_probability": False,
                "action_state_primitives": _unique_strings([tag.get("action_primitive") for tag in matching_tags]),
                "action_state_evidence_event_ids": _unique_strings([tag.get("event_id") for tag in matching_tags]),
                "action_state_action_invocation_ids": _unique_strings([tag.get("action_invocation_id") for tag in matching_tags]),
                "action_state_correlated_object_ids": _unique_strings(
                    [value for tag in matching_tags for value in tag.get("correlated_object_ids", [])]
                ),
                "action_state_correlated_component_ids": _unique_strings(
                    [value for tag in matching_tags for value in tag.get("correlated_component_ids", [])]
                ),
                "relative_state_change_tag_ids": _unique_strings(
                    [tag.get("action_state_tag_id") for tag in matching_tags if tag.get("tag_kind") == "controlled_action_relative_state_change"]
                ),
                "relative_state_change_novelty_statuses": _unique_strings([tag.get("novelty_status") for tag in matching_tags]),
                "relative_state_change_signatures": _unique_strings(
                    [
                        ((tag.get("mathematical_relationships") or {}).get("relationship_signature"))
                        for tag in matching_tags
                        if isinstance(tag.get("mathematical_relationships"), dict)
                    ]
                ),
                "action_interaction_edge_ids": _unique_strings([edge.get("interaction_edge_id") for edge in matching_edges]),
                "action_interaction_kinds": _unique_strings([edge.get("interaction_kind") for edge in matching_edges]),
                "action_interaction_participant_object_ids": _unique_strings(
                    [value for edge in matching_edges for value in edge.get("participant_object_ids", [])]
                ),
                "action_interaction_graph_policy": CONTRA_ACTION_INTERACTION_GRAPH_POLICY,
                "action_interaction_matrix_row_ids": _unique_strings([matrix_row.get("interaction_matrix_row_id") for matrix_row in matching_matrix_rows]),
                "action_interaction_matrix_row_count": len(matching_matrix_rows),
                "action_interaction_matrix_policy": CONTRA_ACTION_INTERACTION_MATRIX_POLICY,
                "action_interaction_routing_recommendations": [
                    matrix_row.get("routing_recommendation")
                    for matrix_row in matching_matrix_rows[:4]
                    if isinstance(matrix_row.get("routing_recommendation"), dict)
                ],
                "solve_like_evidence_status": CONTRA_SOLVE_LIKE_EVIDENCE_STATUS if matching_tags else "no_candidate_interaction_clue",
                "solve_like_evidence_flags": _unique_strings([tag.get("solve_like_evidence_flag") for tag in matching_tags]),
                "solve_like_evidence_score": solve_like_score,
                "solve_like_evidence_boundary": "candidate interaction clue only; not solve truth",
                "action_state_tagger_policy": CONTRA_ACTION_STATE_TAG_POLICY,
                "action_state_tagger_boundary": CONTRA_ACTION_STATE_TAG_BOUNDARY,
                "action_state_adjusted_confidence_score": action_state_adjusted_score,
            }
        )
        if isinstance(lock, dict):
            lock_score = round(max(0.0, min(1.0, float(lock.get("confidence_score") or 0.0))), 6)
            route_flow_score = _route_flow_adjusted_lineup_score(base_score, lock_score)
            effective_score = max(action_state_adjusted_score, route_flow_score)
            new_row.update(
                {
                    "route_flow_lock_id": lock.get("route_flow_lock_id"),
                    "route_flow_lock_status": PORTBAND_ROUTE_FLOW_LOCK_STATUS,
                    "route_flow_positive_signal": PORTBAND_ROUTE_FLOW_POSITIVE_SIGNAL,
                    "route_flow_confidence_score": lock_score,
                    "route_flow_confidence_model": PORTBAND_ROUTE_FLOW_LOCK_CONFIDENCE_MODEL,
                    "route_flow_confidence_is_probability": False,
                    "route_flow_evidence_event_ids": lock.get("evidence_event_ids", []),
                    "route_flow_action_invocation_ids": lock.get("action_invocation_ids", []),
                    "route_flow_lock_policy": PORTBAND_ROUTE_FLOW_LOCK_POLICY,
                    "route_flow_lock_boundary": PORTBAND_ROUTE_FLOW_LOCK_BOUNDARY,
                    "route_flow_adjusted_confidence_score": effective_score,
                    "effective_lineup_confidence_score": effective_score,
                    "effective_lineup_confidence_model": "static_lineup_score_with_single_controlled_paintflow_route_lock_not_probability",
                    "effective_lineup_confidence_is_probability": False,
                    "evidence_status": "static_geometry_plus_controlled_paintflow_route_signal_not_truth",
                    "prediction_status": "not_predictive",
                    "admission_status": "not_admitted",
                    "not_prediction": True,
                    "not_control_truth": True,
                    "not_solve_truth": True,
                    **TRUTH_FLAGS,
                }
            )
        else:
            new_row.update(
                {
                    "route_flow_lock_status": PORTBAND_ROUTE_FLOW_NO_LOCK_STATUS,
                    "route_flow_positive_signal": "none",
                    "route_flow_confidence_score": 0.0,
                    "route_flow_confidence_model": PORTBAND_ROUTE_FLOW_LOCK_CONFIDENCE_MODEL,
                    "route_flow_confidence_is_probability": False,
                    "route_flow_evidence_event_ids": [],
                    "route_flow_action_invocation_ids": [],
                    "route_flow_lock_policy": PORTBAND_ROUTE_FLOW_LOCK_POLICY,
                    "route_flow_lock_boundary": PORTBAND_ROUTE_FLOW_LOCK_BOUNDARY,
                    "route_flow_adjusted_confidence_score": base_score,
                    "effective_lineup_confidence_score": action_state_adjusted_score,
                    "effective_lineup_confidence_model": "static_lineup_score_with_generic_action_state_tag_not_probability"
                    if matching_tags
                    else "static_lineup_score_with_edge_lock_configuration_not_probability"
                    if row.get("edge_lock_status") == "has_ranked_edge_lock_configuration"
                    else PORTBAND_LINEUP_CONFIDENCE_MODEL,
                    "effective_lineup_confidence_is_probability": False,
                }
            )
        if matching_tags:
            new_row["evidence_status"] = "static_geometry_plus_controlled_action_state_tag_not_truth"
            new_row["prediction_status"] = "not_predictive"
            new_row["admission_status"] = "not_admitted"
            new_row["not_prediction"] = True
            new_row["not_control_truth"] = True
            new_row["not_solve_truth"] = True
            new_row.update(TRUTH_FLAGS)
        rows.append(new_row)

    rows.sort(
        key=lambda row: (
            -float(row.get("effective_lineup_confidence_score") or row.get("confidence_score") or 0.0),
            -float(row.get("confidence_score") or 0.0),
            -float(row.get("lineup_score") or 0.0),
            0 if row.get("row_kind") == "direct_port_alignment_lineup" else 1,
            str(row.get("source_object_id")),
            str(row.get("lineup_matrix_row_id")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["lineup_rank"] = rank

    best_by_support: list[dict[str, Any]] = []
    seen_supports: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_object_id") or "")
        if not source_id or source_id in seen_supports or row.get("source_object_role") != "support_group":
            continue
        seen_supports.add(source_id)
        best_by_support.append(
            {
                "source_object_id": source_id,
                "best_lineup_matrix_row_id": row.get("lineup_matrix_row_id"),
                "confidence_score": row.get("confidence_score"),
                "confidence_band": row.get("confidence_band"),
                "edge_lock_status": row.get("edge_lock_status"),
                "edge_lock_top_configuration_id": row.get("edge_lock_top_configuration_id"),
                "edge_lock_configuration_ids": row.get("edge_lock_configuration_ids", []),
                "edge_lock_rank": row.get("edge_lock_rank"),
                "edge_lock_quality_score": row.get("edge_lock_quality_score"),
                "edge_lock_adjusted_confidence_score": row.get("edge_lock_adjusted_confidence_score"),
                "potential_solution_signal_status": row.get("potential_solution_signal_status"),
                "action_state_adjusted_confidence_score": row.get("action_state_adjusted_confidence_score"),
                "action_state_tag_status": row.get("action_state_tag_status"),
                "action_state_positive_signal": row.get("action_state_positive_signal"),
                "action_state_correlation_score": row.get("action_state_correlation_score"),
                "action_state_tag_ids": row.get("action_state_tag_ids", []),
                "action_state_primitives": row.get("action_state_primitives", []),
                "action_state_evidence_event_ids": row.get("action_state_evidence_event_ids", []),
                "action_state_action_invocation_ids": row.get("action_state_action_invocation_ids", []),
                "action_state_correlated_object_ids": row.get("action_state_correlated_object_ids", []),
                "relative_state_change_tag_ids": row.get("relative_state_change_tag_ids", []),
                "relative_state_change_novelty_statuses": row.get("relative_state_change_novelty_statuses", []),
                "relative_state_change_signatures": row.get("relative_state_change_signatures", []),
                "action_interaction_edge_ids": row.get("action_interaction_edge_ids", []),
                "action_interaction_kinds": row.get("action_interaction_kinds", []),
                "action_interaction_participant_object_ids": row.get("action_interaction_participant_object_ids", []),
                "action_interaction_matrix_row_ids": row.get("action_interaction_matrix_row_ids", []),
                "action_interaction_matrix_row_count": row.get("action_interaction_matrix_row_count"),
                "action_interaction_routing_recommendations": row.get("action_interaction_routing_recommendations", []),
                "solve_like_evidence_status": row.get("solve_like_evidence_status"),
                "solve_like_evidence_flags": row.get("solve_like_evidence_flags", []),
                "solve_like_evidence_score": row.get("solve_like_evidence_score"),
                "route_flow_adjusted_confidence_score": row.get("route_flow_adjusted_confidence_score"),
                "effective_lineup_confidence_score": row.get("effective_lineup_confidence_score"),
                "route_flow_lock_status": row.get("route_flow_lock_status"),
                "route_flow_positive_signal": row.get("route_flow_positive_signal"),
                "route_flow_confidence_score": row.get("route_flow_confidence_score"),
                "route_flow_evidence_event_ids": row.get("route_flow_evidence_event_ids", []),
                "route_flow_action_invocation_ids": row.get("route_flow_action_invocation_ids", []),
                "lineup_rank": row.get("lineup_rank"),
                "lineup_route_label": row.get("lineup_route_label"),
                "transform_hypothesis_xy": row.get("transform_hypothesis_xy"),
                "imagined_lineup_bbox_xyxy": row.get("imagined_lineup_bbox_xyxy"),
                "linked_alignment_ids": row.get("linked_alignment_ids", []),
                "linked_fleet_candidate_ids": row.get("linked_fleet_candidate_ids", []),
            }
        )

    route_flow_lock_count = len(feedback.get("route_flow_locks", []) or [])
    positive_row_count = sum(1 for row in rows if row.get("route_flow_positive_signal") == PORTBAND_ROUTE_FLOW_POSITIVE_SIGNAL)
    action_state_tag_count = len(feedback.get("action_state_tags", []) or [])
    action_state_positive_row_count = sum(1 for row in rows if row.get("action_state_positive_signal") == CONTRA_ACTION_STATE_POSITIVE_SIGNAL)
    edge_lock_positive_row_count = sum(1 for row in rows if row.get("edge_lock_status") == "has_ranked_edge_lock_configuration")
    return {
        **lineup_matrix,
        "lineup_route_policy": "static_transform_preview_with_controlled_action_state_tags_not_gameplay_route",
        "edge_lock_ledger_schema": PORTBAND_EDGE_LOCK_LEDGER_SCHEMA,
        "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
        "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
        "edge_lock_positive_signal_count": edge_lock_positive_row_count,
        "top_edge_lock_quality_score": max((float(row.get("edge_lock_quality_score") or 0.0) for row in rows), default=0.0),
        "action_state_tagger_schema": CONTRA_ACTION_STATE_TAG_SET_SCHEMA,
        "action_state_tagger_policy": CONTRA_ACTION_STATE_TAG_POLICY,
        "action_state_tag_count": action_state_tag_count,
        "action_state_positive_signal_count": action_state_positive_row_count,
        "relative_state_change_tag_count": len(feedback.get("relative_state_change_tag_ids", []) or []),
        "action_interaction_edge_count": int(feedback.get("action_interaction_edge_count") or 0),
        "action_interaction_edge_ids": feedback.get("action_interaction_edge_ids", []),
        "action_interaction_kinds": feedback.get("action_interaction_kinds", []),
        "action_interaction_matrix_schema": feedback.get("action_interaction_matrix_schema"),
        "action_interaction_matrix_policy": feedback.get("action_interaction_matrix_policy"),
        "action_interaction_matrix_row_count": int(feedback.get("action_interaction_matrix_row_count") or 0),
        "action_interaction_support_object_count": int(feedback.get("action_interaction_support_object_count") or 0),
        "action_interaction_support_object_ids": feedback.get("action_interaction_support_object_ids", []),
        "action_interaction_best_by_support_object": feedback.get("action_interaction_best_by_support_object", []),
        "novel_interaction_edge_count": int(feedback.get("novel_interaction_edge_count") or 0),
        "solve_like_interaction_edge_count": int(feedback.get("solve_like_interaction_edge_count") or 0),
        "novel_relationship_candidate_count": int(feedback.get("novel_relationship_candidate_count") or 0),
        "solve_like_evidence_flag_count": int(feedback.get("solve_like_evidence_flag_count") or 0),
        "top_solve_like_evidence_score": feedback.get("top_solve_like_evidence_score"),
        "action_state_tag_ids": feedback.get("action_state_tag_ids", []),
        "action_state_primitives": feedback.get("action_state_primitives", []),
        "action_state_event_ids": feedback.get("action_state_event_ids", []),
        "action_state_action_invocation_ids": feedback.get("action_state_action_invocation_ids", []),
        "action_state_confidence_model": CONTRA_ACTION_STATE_TAG_CONFIDENCE_MODEL,
        "action_state_confidence_is_probability": False,
        "action_state_tagger_boundary": CONTRA_ACTION_STATE_TAG_BOUNDARY,
        "route_flow_lock_status": "has_locked_positive_route_flow" if route_flow_lock_count else "no_locked_route_flow",
        "route_flow_lock_policy": PORTBAND_ROUTE_FLOW_LOCK_POLICY,
        "route_flow_lock_count": route_flow_lock_count,
        "route_flow_positive_signal_count": positive_row_count,
        "route_flow_locked_alignment_ids": feedback.get("route_flow_locked_alignment_ids", []),
        "route_flow_confidence_model": PORTBAND_ROUTE_FLOW_LOCK_CONFIDENCE_MODEL,
        "route_flow_confidence_is_probability": False,
        "route_flow_lock_boundary": PORTBAND_ROUTE_FLOW_LOCK_BOUNDARY,
        "row_count": len(rows),
        "support_object_count": len(seen_supports),
        "support_object_ids": sorted(seen_supports),
        "best_by_support_object": best_by_support,
        "rows": rows,
        "evidence_status": "static_geometry_plus_controlled_action_state_feedback_not_truth"
        if action_state_tag_count
        else lineup_matrix.get("evidence_status"),
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def apply_portband_surface_feedback_to_statecons(
    *,
    statecons: list[dict[str, Any]],
    action_surface_result_events: list[dict[str, Any]],
    action_state_relationship_observations: list[dict[str, Any]] | None = None,
    action_primitive: str | None = None,
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for statecon in statecons:
        if not isinstance(statecon, dict):
            updated.append(statecon)
            continue
        portband = statecon.get("portband_hypotheses")
        if not isinstance(portband, dict):
            updated.append(dict(statecon))
            continue
        feedback = build_portband_surface_feedback(
            portband_hypotheses=portband,
            action_surface_result_events=action_surface_result_events,
            action_state_relationship_observations=action_state_relationship_observations,
            action_primitive=action_primitive,
        )
        action_state_lineup_matrix = _apply_action_state_feedback_to_lineup_matrix(
            lineup_matrix=portband.get("lineup_matrix") if isinstance(portband.get("lineup_matrix"), dict) else None,
            feedback=feedback,
        )
        new_portband = {
            **portband,
            "surface_feedback": feedback,
            "surface_feedback_status": feedback.get("status"),
            "surface_feedback_matched_alignment_count": feedback.get("matched_alignment_count"),
            "surface_feedback_adjusted_alignments": feedback.get("ranked_alignments", []),
            "action_state_tagger_schema": feedback.get("action_state_tagger_schema"),
            "action_state_tagger_policy": feedback.get("action_state_tagger_policy"),
            "action_state_tag_status": "has_observed_action_state_tags" if int(feedback.get("action_state_tag_count") or 0) else "no_observed_action_state_tags",
            "action_state_tag_count": feedback.get("action_state_tag_count"),
            "action_state_positive_signal_count": feedback.get("action_state_positive_signal_count"),
            "relative_state_change_tag_count": feedback.get("relative_state_change_tag_count"),
            "relative_state_change_tag_ids": feedback.get("relative_state_change_tag_ids", []),
            "relative_state_change_signature_count": feedback.get("relative_state_change_signature_count"),
            "relative_state_change_unique_signature_count": feedback.get("relative_state_change_unique_signature_count"),
            "action_interaction_graph": feedback.get("action_interaction_graph"),
            "action_interaction_edge_count": feedback.get("action_interaction_edge_count"),
            "action_interaction_edge_ids": feedback.get("action_interaction_edge_ids", []),
            "action_interaction_kinds": feedback.get("action_interaction_kinds", []),
            "action_interaction_matrix": feedback.get("action_interaction_matrix"),
            "action_interaction_matrix_schema": feedback.get("action_interaction_matrix_schema"),
            "action_interaction_matrix_policy": feedback.get("action_interaction_matrix_policy"),
            "action_interaction_matrix_row_count": feedback.get("action_interaction_matrix_row_count"),
            "action_interaction_support_object_count": feedback.get("action_interaction_support_object_count"),
            "action_interaction_support_object_ids": feedback.get("action_interaction_support_object_ids", []),
            "action_interaction_best_by_support_object": feedback.get("action_interaction_best_by_support_object", []),
            "novel_interaction_edge_count": feedback.get("novel_interaction_edge_count"),
            "solve_like_interaction_edge_count": feedback.get("solve_like_interaction_edge_count"),
            "novel_relationship_candidate_count": feedback.get("novel_relationship_candidate_count"),
            "solve_like_evidence_flag_count": feedback.get("solve_like_evidence_flag_count"),
            "top_solve_like_evidence_score": feedback.get("top_solve_like_evidence_score"),
            "action_state_tag_ids": feedback.get("action_state_tag_ids", []),
            "action_state_primitives": feedback.get("action_state_primitives", []),
            "action_state_event_ids": feedback.get("action_state_event_ids", []),
            "action_state_action_invocation_ids": feedback.get("action_state_action_invocation_ids", []),
            "action_state_tags": feedback.get("action_state_tags", []),
            "action_state_confidence_model": feedback.get("action_state_confidence_model"),
            "action_state_confidence_is_probability": False,
            "action_state_tagger_boundary": feedback.get("action_state_tagger_boundary"),
            "route_flow_lock_status": "has_locked_positive_route_flow" if int(feedback.get("route_flow_lock_count") or 0) else "no_locked_route_flow",
            "route_flow_lock_policy": feedback.get("route_flow_lock_policy"),
            "route_flow_lock_count": feedback.get("route_flow_lock_count"),
            "route_flow_positive_signal_count": feedback.get("route_flow_positive_signal_count"),
            "route_flow_locked_alignment_ids": feedback.get("route_flow_locked_alignment_ids", []),
            "route_flow_locks": feedback.get("route_flow_locks", []),
            "route_flow_confidence_model": feedback.get("route_flow_confidence_model"),
            "route_flow_confidence_is_probability": False,
            "route_flow_lock_boundary": feedback.get("route_flow_lock_boundary"),
        }
        if isinstance(action_state_lineup_matrix, dict):
            new_portband["lineup_matrix"] = action_state_lineup_matrix
        updated.append(
            {
                **statecon,
                "portband_hypotheses": new_portband,
                "portband_surface_feedback": feedback,
            }
        )
    return updated


def _virtual_combo_edge(axis: str, side: str, union_box: list[int]) -> dict[str, Any]:
    x0, y0, x1, y1 = [int(value) for value in union_box]
    if axis == "vertical_projection":
        coord = float(x0 if side == "left" else x1)
        span = [int(y0), int(y1)]
        segment = [[int(coord), int(y0)], [int(coord), int(y1)]]
        center = [coord, float(y0 + y1) / 2.0]
    else:
        coord = float(y0 if side == "top" else y1)
        span = [int(x0), int(x1)]
        segment = [[int(x0), int(coord)], [int(x1), int(coord)]]
        center = [float(x0 + x1) / 2.0, coord]
    return {
        "side": side,
        "axis": axis,
        "axis_coordinate_px": round(coord, 6),
        "span_px": span,
        "segment_xy": segment,
        "center_xy": [round(float(center[0]), 6), round(float(center[1]), 6)],
        "edge_reference_kind": "virtual_union_bbox_edge_not_raw_component",
    }


def _combo_shape_matrix(combo: tuple[dict[str, Any], ...], union_box: list[int]) -> dict[str, Any]:
    ux0, uy0, ux1, uy1 = [int(value) for value in union_box]
    member_rows: list[dict[str, Any]] = []
    occupied_area = 0
    for obj in combo:
        box = [int(value) for value in obj.get("bbox_xyxy", [])]
        if len(box) != 4:
            continue
        occupied_area += _bbox_area(box)
        member_rows.append(
            {
                "object_id": str(obj.get("object_id")),
                "component_ids": [str(value) for value in obj.get("component_ids", [])],
                "bbox_xyxy": box,
                "relative_bbox_xyxy": [
                    int(box[0]) - ux0,
                    int(box[1]) - uy0,
                    int(box[2]) - ux0,
                    int(box[3]) - uy0,
                ],
            }
        )
    union_area = _bbox_area(union_box)
    return {
        "matrix_kind": "support_combo_union_bbox_matrix",
        "member_count": len(member_rows),
        "union_bbox_xyxy": [ux0, uy0, ux1, uy1],
        "union_width_px": int(ux1 - ux0 + 1),
        "union_height_px": int(uy1 - uy0 + 1),
        "union_bbox_area_px": union_area,
        "member_bbox_area_sum_px": occupied_area,
        "bbox_fill_ratio": round(max(0.0, min(1.0, float(occupied_area) / float(max(1, union_area)))), 6),
        "members": member_rows,
    }


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, int, str]:
    return (
        -float(row.get("rank_score") or 0.0),
        float(row.get("alignment_error_px") or 999.0),
        int(row.get("combo_size") or 99),
        str(row.get("fleet_candidate_id")),
    )


def _alignment_strength(alignment: dict[str, Any]) -> float:
    error = float(alignment.get("alignment_error_px") or 0.0)
    span = float(alignment.get("span_overlap_ratio") or 0.0)
    kind_bonus = 0.12 if alignment.get("alignment_kind") == "centered_dual_port_projection" else 0.0
    exact_bonus = 0.08 if error <= 0.0 else 0.0
    return round(max(0.0, min(1.0, (1.0 / (1.0 + error)) + span + kind_bonus + exact_bonus)), 6)


def _boundary_rows_from_alignments(
    alignments: list[dict[str, Any]],
    *,
    max_rows: int = 12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], str]] = set()
    for alignment in sorted(alignments, key=_candidate_sort_key):
        if not isinstance(alignment, dict):
            continue
        axis = str(alignment.get("axis") or "")
        target_port_ids = tuple(str(value) for value in alignment.get("target_port_ids", []) if value)
        source_object_ids = [str(value) for value in alignment.get("source_object_ids", []) if value]
        if axis not in {"vertical_projection", "horizontal_projection"} or not target_port_ids or not source_object_ids:
            continue
        boundary_kind = str(alignment.get("alignment_kind") or "portband_alignment")
        if len(target_port_ids) == 1:
            key = (axis, target_port_ids, boundary_kind)
        else:
            key = (axis, target_port_ids[:2], boundary_kind)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "boundary_id": _stable_id("portband_boundary", axis, target_port_ids, source_object_ids, alignment.get("portband_id")),
                "source_alignment_id": alignment.get("portband_id"),
                "boundary_kind": boundary_kind,
                "axis": axis,
                "target_port_ids": list(target_port_ids[:2]),
                "source_object_ids": source_object_ids,
                "alignment_error_px": alignment.get("alignment_error_px"),
                "boundary_strength": _alignment_strength(alignment),
            }
        )
        if len(rows) >= max_rows:
            break
    return rows


def _passes_boundary_sample(
    *,
    combo_object_ids: list[str],
    boundary_source_ids: set[str],
    combo_size: int,
    sample_index: int,
    sample_budget: int,
    statecon_id: str,
    axis: str,
    target_port_ids: list[str],
) -> bool:
    if set(combo_object_ids) & boundary_source_ids:
        return True
    if sample_budget <= 0:
        return False
    digest = hashlib.sha1(
        "|".join([statecon_id, axis, ",".join(target_port_ids), ",".join(combo_object_ids), str(combo_size)]).encode("utf-8")
    ).hexdigest()
    return (int(digest[:8], 16) % max(1, sample_budget * 3)) < sample_budget and sample_index < sample_budget


def _build_portband_fleet(
    *,
    statecon_id: str,
    support_objects: list[dict[str, Any]],
    target_objects: list[dict[str, Any]],
    ports_by_object: dict[str, list[dict[str, Any]]],
    portband_alignments: list[dict[str, Any]] | None = None,
    max_combo_size: int = 4,
    retained_candidate_limit: int = 64,
    boundary_row_limit: int = 12,
    monte_carlo_sample_budget: int = 48,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    total_combo_expansions = 0
    total_candidate_expansions = 0
    boundary_rejected_count = 0
    max_size = min(int(max_combo_size), len(support_objects))
    portband_boundary_rows = _boundary_rows_from_alignments(portband_alignments or [], max_rows=boundary_row_limit)
    ports_by_id = {str(port.get("port_id")): port for ports in ports_by_object.values() for port in ports if isinstance(port, dict)}
    sample_index = 0
    for combo_size in range(2, max_size + 1):
        for combo in combinations(support_objects, combo_size):
            boxes = [[int(value) for value in obj.get("bbox_xyxy", [])] for obj in combo if isinstance(obj.get("bbox_xyxy"), list)]
            if len(boxes) != combo_size or any(len(box) != 4 for box in boxes):
                continue
            total_combo_expansions += 1
            union_box = _union_bbox(boxes)
            shape_matrix = _combo_shape_matrix(combo, union_box)
            combo_component_ids = sorted({str(value) for obj in combo for value in obj.get("component_ids", []) if value is not None})
            combo_object_ids = [str(obj.get("object_id")) for obj in combo]
            for boundary in portband_boundary_rows:
                axis = str(boundary.get("axis") or "")
                target_port_ids = [str(value) for value in boundary.get("target_port_ids", []) if value]
                if len(target_port_ids) != 2:
                    boundary_rejected_count += 1
                    continue
                anchor_a = ports_by_id.get(target_port_ids[0])
                anchor_b = ports_by_id.get(target_port_ids[1])
                if not isinstance(anchor_a, dict) or not isinstance(anchor_b, dict):
                    boundary_rejected_count += 1
                    continue
                if axis == "vertical_projection":
                    sides = ("left", "right")
                elif axis == "horizontal_projection":
                    sides = ("top", "bottom")
                else:
                    boundary_rejected_count += 1
                    continue
                boundary_source_ids = {str(value) for value in boundary.get("source_object_ids", []) if value}
                if not _passes_boundary_sample(
                    combo_object_ids=combo_object_ids,
                    boundary_source_ids=boundary_source_ids,
                    combo_size=combo_size,
                    sample_index=sample_index,
                    sample_budget=monte_carlo_sample_budget,
                    statecon_id=statecon_id,
                    axis=axis,
                    target_port_ids=target_port_ids,
                ):
                    boundary_rejected_count += 1
                    continue
                sample_index += 1
                source_a = _virtual_combo_edge(axis, sides[0], union_box)
                source_b = _virtual_combo_edge(axis, sides[1], union_box)
                source_sep = abs(_axis_coord(source_b) - _axis_coord(source_a))
                if source_sep <= 0.0:
                    boundary_rejected_count += 1
                    continue
                target_sep = abs(_axis_coord(anchor_b) - _axis_coord(anchor_a))
                if target_sep <= 0.0:
                    boundary_rejected_count += 1
                    continue
                total_candidate_expansions += 1
                error = abs(source_sep - target_sep)
                fit_score = max(0.0, min(1.0, 1.0 - (error / max(1.0, target_sep))))
                span_score = min(_span_overlap_ratio(source_a, anchor_a), _span_overlap_ratio(source_b, anchor_b))
                fill_ratio = float(shape_matrix.get("bbox_fill_ratio") or 0.0)
                boundary_strength = float(boundary.get("boundary_strength") or 0.0)
                rank_score = max(0.0, min(1.0, (0.58 * fit_score) + (0.16 * span_score) + (0.12 * fill_ratio) + (0.14 * boundary_strength)))
                source_center = source_a.get("center_xy") if isinstance(source_a.get("center_xy"), list) else []
                target_center = anchor_a.get("center_xy") if isinstance(anchor_a.get("center_xy"), list) else []
                if len(source_center) >= 2 and len(target_center) >= 2:
                    transform = [
                        float(target_center[0]) - float(source_center[0]),
                        float(target_center[1]) - float(source_center[1]),
                    ]
                elif axis == "vertical_projection":
                    transform = [_axis_coord(anchor_a) - _axis_coord(source_a), 0.0]
                else:
                    transform = [0.0, _axis_coord(anchor_a) - _axis_coord(source_a)]
                dx, dy = float(transform[0]), float(transform[1])
                imagined_member_bboxes = [
                    [
                        int(round(float(box[0]) + dx)),
                        int(round(float(box[1]) + dy)),
                        int(round(float(box[2]) + dx)),
                        int(round(float(box[3]) + dy)),
                    ]
                    for box in boxes
                ]
                candidates.append(
                    {
                        "schema": PORTBAND_FLEET_CANDIDATE_SCHEMA,
                        "status": "hypothesis",
                        "fleet_candidate_id": _stable_id("portband_fleet", statecon_id, combo_object_ids, axis, anchor_a.get("port_id"), anchor_b.get("port_id")),
                        "statecon_id": statecon_id,
                        "candidate_kind": "portband_bounded_support_combo_shape_matrix_port_fit",
                        "combo_size": combo_size,
                        "source_object_ids": combo_object_ids,
                        "source_component_ids": combo_component_ids,
                        "target_object_ids": [str(anchor_a.get("object_id")), str(anchor_b.get("object_id"))],
                        "target_port_ids": [str(anchor_a.get("port_id")), str(anchor_b.get("port_id"))],
                        "axis": axis,
                        "source_virtual_edges": [source_a, source_b],
                        "source_port_separation_px": round(float(source_sep), 6),
                        "target_port_separation_px": round(float(target_sep), 6),
                        "alignment_error_px": round(float(error), 6),
                        "portband_fit_score": round(float(fit_score), 6),
                        "span_overlap_score": round(float(span_score), 6),
                        "portband_boundary_strength": round(float(boundary_strength), 6),
                        "rank_score": round(float(rank_score), 6),
                        "combo_shape_matrix": shape_matrix,
                        "transform_hypothesis_xy": [round(dx, 6), round(dy, 6)],
                        "imagined_member_bboxes_xyxy": imagined_member_bboxes,
                        "imagined_combo_bbox_xyxy": [
                            int(round(float(union_box[0]) + dx)),
                            int(round(float(union_box[1]) + dy)),
                            int(round(float(union_box[2]) + dx)),
                            int(round(float(union_box[3]) + dy)),
                        ],
                        "imagined_copy_status": "copy_transform_preview_not_environment_state",
                        "portband_boundary_id": boundary.get("boundary_id"),
                        "portband_boundary_source_alignment_id": boundary.get("source_alignment_id"),
                        "portband_boundary_kind": boundary.get("boundary_kind"),
                        "exploration_boundary": "portband_alignment_frontier",
                        "search_policy": "portband_bounded_monte_carlo_frontier_not_rollout",
                        "mcts_semantics": "mcts_shaped_frontier_expansion_no_simulated_playouts",
                        "evidence_status": "static_geometry_hypothesis",
                        "admission_status": "not_admitted",
                        "not_prediction": True,
                        "not_control_truth": True,
                        "not_solve_truth": True,
                        **TRUTH_FLAGS,
                        "provenance": "derived_from_portband_boundary_support_combo_union_bbox_and_target_port_edges",
                    }
                )
    candidates.sort(key=_candidate_sort_key)
    retained_candidates = candidates[: max(0, int(retained_candidate_limit))]
    for index, row in enumerate(candidates, start=1):
        row["fleet_rank"] = index
    return {
        "schema": PORTBAND_FLEET_SCHEMA,
        "status": "ranked" if retained_candidates else "no_portband_bounded_combo_candidates",
        "statecon_id": statecon_id,
        "definition": "portband_bounded_support_combo_shape_matrix_ranked_by_portband_fit",
        "search_policy": "portband_bounded_monte_carlo_frontier_not_rollout",
        "mcts_semantics": "frontier_expansion_only_no_rollout_no_value_backup",
        "support_object_count": len(support_objects),
        "target_object_count": len(target_objects),
        "max_combo_size": max_size,
        "retained_candidate_limit": int(retained_candidate_limit),
        "monte_carlo_sample_budget": int(monte_carlo_sample_budget),
        "portband_boundary_row_limit": int(boundary_row_limit),
        "portband_boundary_count": len(portband_boundary_rows),
        "portband_boundaries": portband_boundary_rows,
        "combo_expansion_count": total_combo_expansions,
        "candidate_expansion_count": total_candidate_expansions,
        "candidate_count": len(retained_candidates),
        "candidate_total_before_beam": len(candidates),
        "boundary_rejected_count": boundary_rejected_count,
        "ranked_candidates": retained_candidates,
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
        "provenance": "derived_from_statecon_support_combo_shape_matrix",
    }


def _port_with_side(target_ports: list[dict[str, Any]], side: str) -> dict[str, Any] | None:
    for port in target_ports:
        if isinstance(port, dict) and port.get("side") == side:
            return port
    return None


def _target_pair_facing_wall_ports(
    *,
    left_target: dict[str, Any],
    right_target: dict[str, Any],
    ports_by_object: dict[str, list[dict[str, Any]]],
    axis: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    left_ports = ports_by_object.get(str(left_target.get("object_id")), [])
    right_ports = ports_by_object.get(str(right_target.get("object_id")), [])
    if axis == "vertical_projection":
        return (
            _port_with_side(left_ports, "open_right_wall_inner_edge"),
            _port_with_side(right_ports, "open_left_wall_inner_edge"),
        )
    return (
        _port_with_side(left_ports, "open_bottom_wall_inner_edge"),
        _port_with_side(right_ports, "open_top_wall_inner_edge"),
    )


def _lineup_anchor_port(target_ports: list[dict[str, Any]], axis: str, source_coord: float) -> dict[str, Any] | None:
    candidates = [port for port in target_ports if isinstance(port, dict) and port.get("axis") == axis]
    if not candidates:
        return None
    return min(candidates, key=lambda port: abs(_axis_coord(port) - source_coord))


def build_portband_hypotheses(
    *,
    statecon_id: str,
    actor: dict[str, Any],
    target: dict[str, Any],
    all_targets: list[dict[str, Any]],
    support_groups: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build static port-alignment clue rows for one StateCon branch."""

    objects: list[dict[str, Any]] = []
    actor_box = actor.get("actor_bbox_xyxy")
    if isinstance(actor_box, list) and len(actor_box) == 4:
        objects.append(
            {
                "object_id": str(actor.get("actor_id")),
                "object_role": "actor",
                "component_ids": [str(value) for value in (actor.get("actor_component_id"), actor.get("orientation_key_component_id")) if value],
                "bbox_xyxy": [int(value) for value in actor_box],
            }
        )
    for target_row in all_targets:
        if not isinstance(target_row, dict):
            continue
        target_box = target_row.get("target_bbox_xyxy")
        if not isinstance(target_box, list) or len(target_box) != 4:
            target_box = target_row.get("target_square_xyxy")
        if not isinstance(target_box, list) or len(target_box) != 4:
            continue
        objects.append(
            {
                "object_id": str(target_row.get("target_id")),
                "object_role": "target",
                "component_ids": [str(target_row.get("source_component_id") or target_row.get("component_id"))],
                "bbox_xyxy": [int(value) for value in target_box],
                "open_box_xyxy": [int(value) for value in target_box],
                "open_direction": str(target_row.get("open_direction") or ""),
                "is_current_target": str(target_row.get("target_id")) == str(target.get("target_id")),
            }
        )
    for group in support_groups:
        if not isinstance(group, dict):
            continue
        group_box = group.get("group_bbox_xyxy")
        if not isinstance(group_box, list) or len(group_box) != 4:
            continue
        objects.append(
            {
                "object_id": str(group.get("support_group_id")),
                "object_role": "support_group",
                "component_ids": [str(value) for value in group.get("component_ids", []) if value is not None],
                "bbox_xyxy": [int(value) for value in group_box],
            }
        )

    ports: list[dict[str, Any]] = []
    ports_by_object: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        obj_ports = _object_ports(
            statecon_id=statecon_id,
            object_id=str(obj.get("object_id")),
            object_role=str(obj.get("object_role")),
            component_ids=[str(value) for value in obj.get("component_ids", [])],
            box=[int(value) for value in obj.get("bbox_xyxy", [])],
            open_box=[int(value) for value in obj.get("open_box_xyxy", [])] if isinstance(obj.get("open_box_xyxy"), list) else None,
            open_direction=str(obj.get("open_direction") or "") or None,
        )
        ports.extend(obj_ports)
        ports_by_object[str(obj.get("object_id"))] = obj_ports

    alignments: list[dict[str, Any]] = []
    target_objects = [obj for obj in objects if obj.get("object_role") == "target"]
    source_objects = [obj for obj in objects if obj.get("object_role") in {"support_group", "actor"}]

    for source_obj in [obj for obj in source_objects if obj.get("object_role") == "support_group"]:
        source_ports = ports_by_object.get(str(source_obj.get("object_id")), [])
        for source_port in source_ports:
            if source_port.get("side") not in {"left", "right", "top", "bottom"}:
                continue
            axis = str(source_port.get("axis"))
            source_coord = _axis_coord(source_port)
            for target_obj in target_objects:
                target_ports = ports_by_object.get(str(target_obj.get("object_id")), [])
                target_port = _lineup_anchor_port(target_ports, axis, source_coord)
                if target_port is None:
                    continue
                error = abs(source_coord - _axis_coord(target_port))
                if error > 2.0:
                    continue
                alignments.append(
                    _alignment_row(
                        statecon_id=statecon_id,
                        alignment_kind="single_port_projection",
                        axis=axis,
                        source_ports=[source_port],
                        target_ports=[target_port],
                        error_px=error,
                        transform_xy=[0.0, 0.0],
                        match_basis="current_position_side_port_axis_coordinate_match",
                        extra={
                            "span_overlap_ratio": _span_overlap_ratio(source_port, target_port),
                        },
                    )
                )

    for source_obj in source_objects:
        source_ports = _port_by_side(ports_by_object.get(str(source_obj.get("object_id")), []))
        for axis, sides in (
            ("vertical_projection", ("left", "right")),
            ("horizontal_projection", ("top", "bottom")),
        ):
            left = source_ports.get(sides[0])
            right = source_ports.get(sides[1])
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            source_sep = abs(_axis_coord(right) - _axis_coord(left))
            if source_sep <= 0.0:
                continue
            for target_a, target_b in combinations(target_objects, 2):
                target_a_center = _center([int(value) for value in target_a.get("bbox_xyxy", [])])
                target_b_center = _center([int(value) for value in target_b.get("bbox_xyxy", [])])
                if axis == "vertical_projection":
                    left_target, right_target = (target_a, target_b) if target_a_center[0] <= target_b_center[0] else (target_b, target_a)
                else:
                    left_target, right_target = (target_a, target_b) if target_a_center[1] <= target_b_center[1] else (target_b, target_a)
                anchor_a, anchor_b = _target_pair_facing_wall_ports(
                    left_target=left_target,
                    right_target=right_target,
                    ports_by_object=ports_by_object,
                    axis=axis,
                )
                if not isinstance(anchor_a, dict) or not isinstance(anchor_b, dict):
                    continue
                target_sep = abs(_axis_coord(anchor_b) - _axis_coord(anchor_a))
                if target_sep <= 0.0:
                    continue
                error = abs(source_sep - target_sep)
                if error > 8.0:
                    continue
                source_center = left.get("center_xy") if isinstance(left.get("center_xy"), list) else []
                target_center = anchor_a.get("center_xy") if isinstance(anchor_a.get("center_xy"), list) else []
                if len(source_center) >= 2 and len(target_center) >= 2:
                    transform = [
                        float(target_center[0]) - float(source_center[0]),
                        float(target_center[1]) - float(source_center[1]),
                    ]
                elif axis == "vertical_projection":
                    transform = [_axis_coord(anchor_a) - _axis_coord(left), 0.0]
                else:
                    transform = [0.0, _axis_coord(anchor_a) - _axis_coord(left)]
                alignments.append(
                    _alignment_row(
                        statecon_id=statecon_id,
                        alignment_kind="centered_dual_port_projection",
                        axis=axis,
                        source_ports=[left, right],
                        target_ports=[anchor_a, anchor_b],
                        error_px=error,
                        transform_xy=transform,
                        match_basis="source_opposite_edges_match_facing_target_wall_edges_under_copy_transform",
                        extra={
                            "source_port_separation_px": round(float(source_sep), 6),
                            "target_port_separation_px": round(float(target_sep), 6),
                            "target_pair_count": 2,
                            "target_edge_match_rule": "left_target_open_right_wall_inner_edge_to_right_target_open_left_wall_inner_edge",
                        },
                    )
                )

    alignments.sort(
        key=lambda row: (
            float(row.get("alignment_error_px") or 999.0),
            0 if row.get("alignment_kind") == "centered_dual_port_projection" else 1,
            str(row.get("portband_id")),
        )
    )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
    for row in alignments:
        key = (
            str(row.get("alignment_kind")),
            tuple(str(value) for value in row.get("source_port_ids", [])),
            tuple(str(value) for value in row.get("target_port_ids", [])),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    portband_fleet = _build_portband_fleet(
        statecon_id=statecon_id,
        support_objects=[obj for obj in source_objects if obj.get("object_role") == "support_group"],
        target_objects=target_objects,
        ports_by_object=ports_by_object,
        portband_alignments=deduped,
    )
    visible_alignments = deduped[:32]
    edge_lock_ledger = _build_edge_lock_ledger(
        statecon_id=statecon_id,
        objects=objects,
        alignments=visible_alignments,
        portband_fleet=portband_fleet,
    )
    lineup_matrix = _build_lineup_matrix(
        statecon_id=statecon_id,
        objects=objects,
        alignments=visible_alignments,
        portband_fleet=portband_fleet,
        edge_lock_ledger=edge_lock_ledger,
    )
    return {
        "schema": PORTBAND_PACKET_SCHEMA,
        "status": "hypothesis",
        "statecon_id": statecon_id,
        "definition": "static_object_port_alignment_clue",
        "port_count": len(ports),
        "alignment_count": len(deduped),
        "portband_fleet_status": portband_fleet.get("status"),
        "portband_fleet_candidate_count": portband_fleet.get("candidate_count"),
        "edge_lock_ledger_status": edge_lock_ledger.get("status"),
        "edge_lock_configuration_count": edge_lock_ledger.get("configuration_count"),
        "edge_lock_positive_signal_count": edge_lock_ledger.get("positive_signal_count"),
        "edge_lock_top_quality_score": edge_lock_ledger.get("top_edge_lock_quality_score"),
        "edge_lock_policy": PORTBAND_EDGE_LOCK_POLICY,
        "edge_lock_boundary": PORTBAND_EDGE_LOCK_BOUNDARY,
        "lineup_matrix_status": lineup_matrix.get("status"),
        "lineup_matrix_row_count": lineup_matrix.get("row_count"),
        "ports": ports,
        "alignments": visible_alignments,
        "portband_fleet": portband_fleet,
        "edge_lock_ledger": edge_lock_ledger,
        "lineup_matrix": lineup_matrix,
        "pass_ids": [
            "bbox_side_port_extraction",
            "single_port_projection",
            "centered_dual_port_projection",
            "portband_fleet_combo_shape_matrix",
            "portband_edge_lock_ledger",
            "portband_lineup_matrix",
        ],
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
        "provenance": "derived_from_statecon_actor_target_support_geometry",
    }
