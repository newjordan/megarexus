#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import math
from typing import Any

from .boundary_pass import build_boundary_queue, build_geoband_boundary_passes
from .geoband import (
    angular_error_degrees,
    bearing_degrees,
    bbox_center,
    bbox_union,
    build_geoband_hypothesis,
    clamp01,
    direction_axis,
    directional_gap,
    lateral_offset,
    node_center_from_bbox,
    ray_intersects_box,
)
from .portband import build_portband_hypotheses
from .symmetry import symmetry_groups_for_statecon

STATECON_HYPOTHESIS_SCHEMA = "codexlab.state_condition_hypothesis.v1"
STATECON_SET_SCHEMA = "codexlab.state_condition_hypotheses.v1"
STATECON_IDENTITY_MODEL_SCHEMA = "codexlab.statecon_identity_mlp.v1"
STATECON_SUPPORT_AFFORDANCE_SCHEMA = "codexlab.statecon_support_affordance_hypothesis.v1"
STATECON_BRANCH_GRAPH_SCHEMA = "codexlab.statecon_branch_graph.v1"

def stable_statecon_id(actor_id: str, target_id: str, condition_kind: str) -> str:
    digest = hashlib.sha1(f"{condition_kind}|{actor_id}|{target_id}".encode("utf-8")).hexdigest()[:10]
    return f"statecon_{condition_kind}_{digest}"


def _safe_token(value: object) -> str:
    text = str(value)
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    return cleaned.strip("_") or "unknown"


def distance_point_to_segment(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return math.dist(point, start), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    clamped_t = max(0.0, min(1.0, t))
    closest = (ax + clamped_t * dx, ay + clamped_t * dy)
    return math.dist(point, closest), clamped_t


def branch_support_affordances(
    *,
    actor: dict[str, Any],
    target: dict[str, Any],
    supports: list[dict[str, Any]],
    frame_shape: list[int],
) -> list[dict[str, Any]]:
    if not supports:
        return []
    height = max(1.0, float(frame_shape[0]))
    width = max(1.0, float(frame_shape[1]))
    actor_center = bbox_center([int(value) for value in actor["actor_bbox_xyxy"]])
    target_center = bbox_center([int(value) for value in target["target_square_xyxy"]])
    target_bearing = bearing_degrees(actor_center, target_center)
    corridor_width = max(5.0, min(width, height) * 0.14)
    branch_rows: list[dict[str, Any]] = []
    for support in supports:
        box = [int(value) for value in support["bbox_xyxy"]]
        support_center = bbox_center(box)
        lateral_distance, projection = distance_point_to_segment(support_center, actor_center, target_center)
        support_bearing = bearing_degrees(actor_center, support_center)
        angular_error = angular_error_degrees(target_bearing, support_bearing)
        branch_member = (
            projection >= 0.0
            and projection <= 1.05
            and (lateral_distance <= corridor_width or angular_error <= 24.0)
        )
        if not branch_member:
            continue
        row = dict(support)
        row["branch_relation"] = {
            "schema": "codexlab.statecon_branch_support_relation.v1",
            "status": "hypothesis",
            "lateral_distance_pixels": round(float(lateral_distance), 6),
            "projection_ratio_actor_to_target": round(float(projection), 6),
            "actor_to_target_bearing_degrees": target_bearing,
            "actor_to_support_bearing_degrees": support_bearing,
            "angular_error_degrees": angular_error,
            "branch_membership_basis": "near_actor_target_corridor_or_bearing",
            "admission_status": "not_admitted",
            "control_truth_admitted": False,
        }
        branch_rows.append(row)
    branch_rows.sort(
        key=lambda row: (
            float((row.get("branch_relation") or {}).get("lateral_distance_pixels") or 0.0),
            str(row.get("support_id")),
        )
    )
    return branch_rows


def support_groups_by_kind(supports: list[dict[str, Any]], *, statecon_id: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for support in supports:
        support_kind = str(support.get("support_kind"))
        relation_category = str(support.get("special_object_category") or support.get("support_relation_category") or "none")
        grouped.setdefault(f"{support_kind}:{relation_category}", []).append(support)
    rows: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        support_kind = str(group_rows[0].get("support_kind"))
        relation_category = str(group_rows[0].get("special_object_category") or group_rows[0].get("support_relation_category") or "none")
        boxes = [[int(value) for value in row["bbox_xyxy"]] for row in group_rows]
        group_bbox = bbox_union(boxes)
        rows.append(
            {
                "support_group_id": f"{statecon_id}:support_group_{_safe_token(group_key)}",
                "support_kind": support_kind,
                "support_relation_category": relation_category,
                "known_with_unknown_attached_count": sum(
                    1 for row in group_rows if row.get("special_object_category") == "known_y_unknown_attached"
                ),
                "support_ids": [row.get("support_id") for row in group_rows],
                "component_ids": [row.get("source_component_id") for row in group_rows],
                "group_bbox_xyxy": group_bbox,
                "group_center_xy": node_center_from_bbox(group_bbox),
                "support_count": len(group_rows),
                "group_status": "typed_support_cell",
                "group_scope": "statecon_branch_local",
            }
        )
    return rows


def primary_geoband_support_groups(supports: list[dict[str, Any]], *, statecon_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for support in supports:
        support_id = str(support.get("support_id"))
        box = [int(value) for value in support["bbox_xyxy"]]
        relation_category = str(support.get("special_object_category") or support.get("support_relation_category") or "none")
        rows.append(
            {
                "support_group_id": f"{statecon_id}:geoband_object_{_safe_token(support_id)}",
                "support_kind": str(support.get("support_kind")),
                "support_relation_category": relation_category,
                "known_with_unknown_attached_count": 1 if support.get("special_object_category") == "known_y_unknown_attached" else 0,
                "support_ids": [support_id],
                "component_ids": [support.get("source_component_id")],
                "group_bbox_xyxy": box,
                "group_center_xy": node_center_from_bbox(box),
                "support_count": 1,
                "group_status": "geoband_primary_relationship_object",
                "group_scope": "geoband_primary_all_objects",
            }
        )
    return rows


def build_statecon_branch_graph(
    *,
    statecon_id: str,
    actor: dict[str, Any],
    target: dict[str, Any],
    support_groups: list[dict[str, Any]],
    supports: list[dict[str, Any]],
    symmetry_groups: list[dict[str, Any]],
    symmetry_sensors: list[dict[str, Any]],
    group_box: list[int],
) -> dict[str, Any]:
    actor_id = str(actor.get("actor_id"))
    target_id = str(target.get("target_id"))
    nodes = [
        {
            "node_id": f"{statecon_id}:macro",
            "node_kind": "macro_statecon",
            "anchor_kind": "relation_bbox_center_not_object_anchor",
            "label": "StateCon",
            "bbox_xyxy": group_box,
            "center_xy": node_center_from_bbox(group_box),
            "render_color": "amber",
        },
        {
            "node_id": actor_id,
            "node_kind": "actor_affordance",
            "anchor_kind": "component_bbox_center",
            "label": "actor",
            "component_id": actor.get("actor_component_id"),
            "bbox_xyxy": [int(value) for value in actor["group_bbox_xyxy"]],
            "center_xy": node_center_from_bbox([int(value) for value in actor["group_bbox_xyxy"]]),
            "render_color": "bright_purple",
        },
        {
            "node_id": target_id,
            "node_kind": "target_affordance",
            "anchor_kind": "target_square_center",
            "label": "target",
            "component_id": target.get("source_component_id"),
            "bbox_xyxy": [int(value) for value in target["target_square_xyxy"]],
            "center_xy": node_center_from_bbox([int(value) for value in target["target_square_xyxy"]]),
            "render_color": "bright_green",
        },
    ]
    edges = [
        {
            "edge_id": f"{statecon_id}:macro_to_actor",
            "edge_kind": "macro_to_actor",
            "from_node_id": f"{statecon_id}:macro",
            "to_node_id": actor_id,
        },
        {
            "edge_id": f"{statecon_id}:macro_to_target",
            "edge_kind": "macro_to_target",
            "from_node_id": f"{statecon_id}:macro",
            "to_node_id": target_id,
        },
    ]
    for group in support_groups:
        support_group_id = str(group["support_group_id"])
        relation_category = str(group.get("support_relation_category") or "none")
        render_color = "yellow" if group.get("known_with_unknown_attached_count") else "blue"
        nodes.append(
            {
                "node_id": support_group_id,
                "node_kind": "statecon_support_group",
                "anchor_kind": "support_group_bbox_center",
                "label": str(group["support_kind"]),
                "support_kind": group["support_kind"],
                "support_relation_category": relation_category,
                "known_with_unknown_attached_count": group.get("known_with_unknown_attached_count"),
                "bbox_xyxy": group["group_bbox_xyxy"],
                "center_xy": group["group_center_xy"],
                "render_color": render_color,
            }
        )
        edges.append(
            {
                "edge_id": f"{statecon_id}:macro_to_{support_group_id}",
                "edge_kind": "macro_to_support_group",
                "from_node_id": f"{statecon_id}:macro",
                "to_node_id": support_group_id,
            }
        )
    support_group_id_by_support_id = {
        str(support_id): str(group.get("support_group_id"))
        for group in support_groups
        for support_id in group.get("support_ids", [])
    }
    for support in supports:
        support_id = str(support.get("support_id"))
        support_kind = str(support.get("support_kind"))
        box = [int(value) for value in support["bbox_xyxy"]]
        relation_category = str(support.get("special_object_category") or support.get("support_relation_category") or "none")
        render_color = "yellow" if support.get("special_object_category") == "known_y_unknown_attached" else "blue"
        nodes.append(
            {
                "node_id": support_id,
                "node_kind": "statecon_support_affordance",
                "anchor_kind": "support_component_bbox_center",
                "label": support_kind,
                "support_kind": support_kind,
                "support_relation_category": relation_category,
                "known_object_status": support.get("known_object_status"),
                "special_object_category": support.get("special_object_category"),
                "unknown_attachment_marker": support.get("unknown_attachment_marker"),
                "component_id": support.get("source_component_id"),
                "bbox_xyxy": box,
                "center_xy": node_center_from_bbox(box),
                "render_color": render_color,
            }
        )
        edges.append(
            {
                "edge_id": f"{statecon_id}:{support_kind}_to_{support_id}",
                "edge_kind": "support_group_to_support",
                "from_node_id": support_group_id_by_support_id.get(support_id, ""),
                "to_node_id": support_id,
            }
        )
    for group in symmetry_groups:
        group_id = str(group.get("symmetry_group_id"))
        box = group.get("bbox_xyxy")
        if not group_id or not isinstance(box, list) or len(box) != 4:
            continue
        node_id = f"{statecon_id}:symmetry_{_safe_token(group_id)}"
        nodes.append(
            {
                "node_id": node_id,
                "node_kind": "symmetry_group",
                "anchor_kind": "symmetry_group_bbox_center",
                "label": "symmetry",
                "symmetry_group_id": group_id,
                "symmetry_kind": group.get("symmetry_kind"),
                "axis": group.get("axis"),
                "score": group.get("score"),
                "component_ids": group.get("member_component_ids", []),
                "bbox_xyxy": [int(value) for value in box],
                "center_xy": node_center_from_bbox([int(value) for value in box]),
                "render_color": "cyan",
            }
        )
        edges.append(
            {
                "edge_id": f"{statecon_id}:macro_to_symmetry_{_safe_token(group_id)}",
                "edge_kind": "macro_to_symmetry_group",
                "from_node_id": f"{statecon_id}:macro",
                "to_node_id": node_id,
            }
        )
    for sensor in symmetry_sensors:
        sensor_id = str(sensor.get("sensor_id") or "")
        box = sensor.get("bbox_xyxy")
        if not sensor_id or not isinstance(box, list) or len(box) != 4:
            continue
        node_id = f"{statecon_id}:symmetry_sensor_{_safe_token(sensor_id)}"
        source_group_id = str(sensor.get("source_symmetry_group_id") or "")
        nodes.append(
            {
                "node_id": node_id,
                "node_kind": "symmetry_sensor",
                "anchor_kind": "symmetry_sensor_bbox_center",
                "label": "lock/key",
                "sensor_layer_id": sensor.get("sensor_layer_id"),
                "sensor_id": sensor_id,
                "sensor_kind": sensor.get("sensor_kind"),
                "source_symmetry_group_id": source_group_id,
                "key_candidate_component_id": sensor.get("key_candidate_component_id"),
                "lock_candidate_component_id": sensor.get("lock_candidate_component_id"),
                "confidence": sensor.get("confidence"),
                "bbox_xyxy": [int(value) for value in box],
                "center_xy": node_center_from_bbox([int(value) for value in box]),
                "render_color": "yellow_green",
            }
        )
        source_node_id = f"{statecon_id}:symmetry_{_safe_token(source_group_id)}" if source_group_id else f"{statecon_id}:macro"
        edges.append(
            {
                "edge_id": f"{statecon_id}:symmetry_sensor_{_safe_token(sensor_id)}",
                "edge_kind": "symmetry_group_to_symmetry_sensor" if source_group_id else "macro_to_symmetry_sensor",
                "from_node_id": source_node_id,
                "to_node_id": node_id,
            }
        )
    return {
        "schema": STATECON_BRANCH_GRAPH_SCHEMA,
        "graph_kind": "statecon_branch_node_visual_hypothesis",
        "status": "hypothesis",
        "macro_node_id": f"{statecon_id}:macro",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }

def symmetry_sensors_for_statecon(
    symmetry_packet: dict[str, Any] | None,
    *,
    symmetry_groups: list[dict[str, Any]],
    member_component_ids: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(symmetry_packet, dict):
        return []
    sensors = symmetry_packet.get("symmetry_sensor_hypotheses")
    if not isinstance(sensors, list):
        return []
    selected_group_ids = {str(group.get("symmetry_group_id")) for group in symmetry_groups if isinstance(group, dict)}
    rows: list[dict[str, Any]] = []
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        source_group_id = str(sensor.get("source_symmetry_group_id") or "")
        sensor_component_ids = {str(value) for value in sensor.get("member_component_ids", []) if value is not None}
        if source_group_id in selected_group_ids or bool(sensor_component_ids.intersection(member_component_ids)):
            rows.append(sensor)
    rows.sort(key=lambda row: (-float(row.get("confidence") or 0.0), str(row.get("sensor_id") or "")))
    return rows[:24]

def build_statecon_support_affordances(
    *,
    component_graph: dict[str, Any],
    support_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(support_expectation, dict) or support_expectation.get("status") != "user_supplied_answer":
        return []
    components = {
        str(component.get("component_id")): component
        for component in component_graph.get("components", [])
        if isinstance(component, dict)
    }
    support_kind = str(support_expectation.get("support_kind"))
    raw_pixel_value = support_expectation.get("raw_pixel_value")
    rows: list[dict[str, Any]] = []
    for component_id in support_expectation.get("component_ids", []):
        component = components.get(str(component_id))
        if component is None:
            continue
        rows.append(
            {
                "schema": STATECON_SUPPORT_AFFORDANCE_SCHEMA,
                "status": "user_supplied_answer",
                "support_id": f"statecon_support_{support_kind}_{component_id}",
                "support_kind": support_kind,
                "source": "manual_sp80_surface_calibration",
                "source_component_role": "residual_component_hypothesis",
                "component_id": str(component_id),
                "source_component_id": str(component_id),
                "source_raw_pixel_value": int(component.get("raw_pixel_value")),
                "expected_raw_pixel_value": raw_pixel_value,
                "support_admission_policy": str(support_expectation.get("admission_policy")),
                "evidence_status": "manual_hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "bbox_xyxy": [int(value) for value in component["bbox_xyxy"]],
                "center_xy": [float(value) for value in component["center_xy"]],
                "area_pixels": int(component["area_pixels"]),
                "render_color": "blue",
                "rendering_role": "statecon_support_review_marker",
                "provenance": support_expectation.get("provenance"),
            }
        )
    return rows


class StateConIdentityMLP:
    """Small deterministic MLP-shaped scorer for relational StateCon features.

    This is a dormant scaffold, not a trained model. The row metadata records
    that distinction so verifier/users do not treat the scores as truth.
    """

    input_feature_names = [
        "ray_intersects_target",
        "axis_alignment",
        "directional_proximity",
        "target_open_axis_match",
        "actor_key_contact",
        "actor_direction_geometry_agreement",
    ]

    hidden_weights = [
        [1.25, 0.95, 0.80, 0.25, 0.35, 0.45],
        [-0.55, 0.30, 0.65, 0.20, 0.80, 0.30],
        [0.40, 1.10, 0.20, 0.55, -0.10, 0.30],
    ]
    hidden_bias = [-1.20, -0.45, -0.70]
    output_weights = [1.05, 0.55, 0.70]
    output_bias = -0.65

    @classmethod
    def score(cls, features: dict[str, float]) -> dict[str, Any]:
        values = [float(features[name]) for name in cls.input_feature_names]
        hidden = []
        for weights, bias in zip(cls.hidden_weights, cls.hidden_bias):
            activation = sum(weight * value for weight, value in zip(weights, values)) + bias
            hidden.append(max(0.0, activation))
        raw_score = sum(weight * value for weight, value in zip(cls.output_weights, hidden)) + cls.output_bias
        relation_descriptor_score = 1.0 / (1.0 + math.exp(-raw_score))
        same_statecon_score = clamp01(0.50 + 0.30 * features["actor_direction_geometry_agreement"] + 0.20 * features["actor_key_contact"])
        if relation_descriptor_score >= 0.66:
            condition_status = "partial"
        elif relation_descriptor_score >= 0.34:
            condition_status = "unknown"
        else:
            condition_status = "false_hypothesis"
        return {
            "same_statecon_score": round(same_statecon_score, 6),
            "relation_descriptor_score": round(clamp01(relation_descriptor_score), 6),
            "event_enable_score": round(clamp01(relation_descriptor_score), 6),
            "condition_status_hypothesis": condition_status,
            "hidden_activations": [round(clamp01(value), 6) for value in hidden],
        }


def relation_features(actor: dict[str, Any], target: dict[str, Any], *, frame_shape: list[int]) -> dict[str, float]:
    height = max(1.0, float(frame_shape[0]))
    width = max(1.0, float(frame_shape[1]))
    actor_box = [int(value) for value in actor["actor_bbox_xyxy"]]
    target_box = [int(value) for value in target["target_square_xyxy"]]
    actor_center = bbox_center(actor_box)
    target_center = bbox_center(target_box)
    flow_direction = str(actor.get("flow_direction_hypothesis"))
    open_direction = str(target.get("open_direction"))
    axis_alignment = 1.0 - clamp01(lateral_offset(actor_center, flow_direction, target_box) / max(width, height))
    directional_proximity = 1.0 - clamp01(directional_gap(actor_center, flow_direction, target_box) / max(width, height))
    distance = math.dist(actor_center, target_center)
    return {
        "ray_intersects_target": 1.0 if ray_intersects_box(actor_center, flow_direction, target_box) else 0.0,
        "axis_alignment": round(clamp01(axis_alignment), 6),
        "directional_proximity": round(clamp01(directional_proximity), 6),
        "target_open_axis_match": 1.0 if direction_axis(flow_direction) == direction_axis(open_direction) else 0.0,
        "actor_key_contact": 1.0 if int(actor.get("contact_edge_count") or 0) > 0 else 0.0,
        "actor_direction_geometry_agreement": 1.0 if actor.get("direction_agrees_with_raw_geometry") is True else 0.0,
        "actor_target_distance_ratio": round(clamp01(distance / math.dist((0, 0), (width, height))), 6),
    }


def build_state_condition_hypotheses(
    *,
    frame_shape: list[int],
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    support_affordances: list[dict[str, Any]] | None = None,
    symmetry_hypotheses: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    supports = [row for row in (support_affordances or []) if isinstance(row, dict)]
    for actor in actor_affordances:
        if not isinstance(actor, dict):
            continue
        actor_id = str(actor.get("actor_id"))
        for target in target_affordances:
            if not isinstance(target, dict):
                continue
            target_id = str(target.get("target_id"))
            condition_kind = "actor_target_flow_alignment"
            features = relation_features(actor, target, frame_shape=frame_shape)
            mlp = StateConIdentityMLP.score(features)
            branch_supports = branch_support_affordances(
                actor=actor,
                target=target,
                supports=supports,
                frame_shape=frame_shape,
            )
            support_boxes = [[int(value) for value in support["bbox_xyxy"]] for support in supports]
            group_box = bbox_union(
                [
                    [int(value) for value in actor["group_bbox_xyxy"]],
                    [int(value) for value in target["target_square_xyxy"]],
                    *support_boxes,
                ]
            )
            statecon_id = stable_statecon_id(actor_id, target_id, condition_kind)
            support_groups = primary_geoband_support_groups(supports, statecon_id=statecon_id)
            member_component_ids = {
                str(value)
                for value in [
                    actor.get("actor_component_id"),
                    actor.get("orientation_key_component_id"),
                    target.get("source_component_id"),
                    *(support.get("source_component_id") for support in supports),
                ]
                if value
            }
            symmetry_groups = symmetry_groups_for_statecon(
                symmetry_hypotheses or {},
                member_component_ids=member_component_ids,
                group_box=group_box,
            )
            symmetry_sensors = symmetry_sensors_for_statecon(
                symmetry_hypotheses,
                symmetry_groups=symmetry_groups,
                member_component_ids=member_component_ids,
            )
            geoband_subgroups = support_groups_by_kind(branch_supports, statecon_id=statecon_id)
            boundary_queue = build_boundary_queue(
                statecon_id=statecon_id,
                all_supports=supports,
                subgroup_supports=branch_supports,
            )
            geoband_passes = build_geoband_boundary_passes(
                primary_supports=supports,
                subgroup_supports=branch_supports,
                primary_groups=support_groups,
                subgroups=geoband_subgroups,
                boundary_queue=boundary_queue,
            )
            portband_hypotheses = build_portband_hypotheses(
                statecon_id=statecon_id,
                actor=actor,
                target=target,
                all_targets=target_affordances,
                support_groups=support_groups,
            )
            geoband = build_geoband_hypothesis(
                statecon_id=statecon_id,
                actor=actor,
                peer_actors=actor_affordances,
                target=target,
                support_groups=support_groups,
                symmetry_groups=symmetry_groups,
                symmetry_sensors=symmetry_sensors,
                portband_hypotheses=portband_hypotheses,
                features=features,
            )
            branch_graph = build_statecon_branch_graph(
                statecon_id=statecon_id,
                actor=actor,
                target=target,
                support_groups=support_groups,
                supports=supports,
                symmetry_groups=symmetry_groups,
                symmetry_sensors=symmetry_sensors,
                group_box=group_box,
            )
            hypotheses.append(
                {
                    "schema": STATECON_HYPOTHESIS_SCHEMA,
                    "status": "hypothesis",
                    "statecon_id": statecon_id,
                    "condition_kind": condition_kind,
                    "sticky_identity": True,
                    "sticky_identity_source": "statecon_identity_mlp",
                    "identity_model": {
                        "schema": STATECON_IDENTITY_MODEL_SCHEMA,
                        "model_kind": "deterministic_untrained_mlp_scaffold",
                        "input_feature_names": StateConIdentityMLP.input_feature_names,
                        "training_status": "untrained",
                        "weights_status": "fixed_scaffold",
                        "truth_status": "not_truth",
                    },
                    "sticky_group": {
                        "actor_affordance_id": actor_id,
                        "actor_set_id": actor.get("actor_set_id"),
                        "actor_option_rank": actor.get("actor_option_rank"),
                        "actor_option_count": actor.get("actor_option_count"),
                        "actor_option_component_ids": actor.get("actor_option_component_ids", []),
                        "actor_option_relation": actor.get("actor_option_relation"),
                        "actor_set_symmetry_group_ids": actor.get("actor_set_symmetry_group_ids", []),
                        "target_affordance_id": target_id,
                        "target_set_id": target.get("target_set_id"),
                        "goal_option_rank": target.get("goal_option_rank"),
                        "goal_option_count": target.get("goal_option_count"),
                        "goal_option_component_ids": target.get("goal_option_component_ids", []),
                        "goal_option_relation": target.get("goal_option_relation"),
                        "target_set_symmetry_group_id": target.get("target_set_symmetry_group_id"),
                        "actor_component_id": actor.get("actor_component_id"),
                        "orientation_key_component_id": actor.get("orientation_key_component_id"),
                        "target_component_id": target.get("source_component_id"),
                        "support_affordance_ids": [support.get("support_id") for support in supports],
                        "support_component_ids": [support.get("source_component_id") for support in supports],
                        "symmetry_group_ids": [group.get("symmetry_group_id") for group in symmetry_groups],
                        "symmetry_sensor_ids": [sensor.get("sensor_id") for sensor in symmetry_sensors],
                        "available_support_affordance_ids": [support.get("support_id") for support in supports],
                        "geoband_primary_policy": "include_all_known_relationship_objects_before_subgroup_refinement",
                        "geoband_subgroup_policy": "actor_target_corridor_or_bearing_local_second_pass",
                        "geoband_subgroup_support_affordance_ids": [support.get("support_id") for support in branch_supports],
                        "relation_members": [
                            "actor_flow_direction",
                            "orientation_key_grouping",
                            "open_target_region",
                            "calibrated_support_regions",
                            "actor_target_ray",
                            "relative_distance",
                            "symmetry_relationships",
                            "symmetry_sensor_layer",
                            "port_alignment_clues",
                        ],
                        "group_status": "small_relational_group_in_flux",
                    },
                    "predicate": "actor_flow_direction_may_intersect_open_target_region",
                    "actor_affordance_id": actor_id,
                    "actor_set_id": actor.get("actor_set_id"),
                    "actor_option_rank": actor.get("actor_option_rank"),
                    "actor_option_count": actor.get("actor_option_count"),
                    "actor_option_component_ids": actor.get("actor_option_component_ids", []),
                    "actor_option_relation": actor.get("actor_option_relation"),
                    "actor_set_symmetry_group_ids": actor.get("actor_set_symmetry_group_ids", []),
                    "target_affordance_id": target_id,
                    "target_set_id": target.get("target_set_id"),
                    "goal_option_rank": target.get("goal_option_rank"),
                    "goal_option_count": target.get("goal_option_count"),
                    "goal_option_component_ids": target.get("goal_option_component_ids", []),
                    "goal_option_relation": target.get("goal_option_relation"),
                    "target_set_symmetry_group_id": target.get("target_set_symmetry_group_id"),
                    "support_affordance_hypotheses": supports,
                    "support_groups": support_groups,
                    "symmetry_groups": symmetry_groups,
                    "symmetry_sensors": symmetry_sensors,
                    "portband_hypotheses": portband_hypotheses,
                    "portband_alignment_hypotheses": portband_hypotheses.get("alignments", []),
                    "portband_lineup_matrix": portband_hypotheses.get("lineup_matrix"),
                    "geoband_subgroups": geoband_subgroups,
                    "geoband_subgroup_support_affordance_hypotheses": branch_supports,
                    "geoband_boundary_queue": boundary_queue,
                    "geoband_passes": geoband_passes,
                    "support_member_policy": "manual_or_gated_only",
                    "statecon_branch_graph": branch_graph,
                    "geoband_hypothesis": geoband,
                    "features": features,
                    "same_statecon_score": mlp["same_statecon_score"],
                    "condition_status_hypothesis": mlp["condition_status_hypothesis"],
                    "relation_descriptor_score": mlp["relation_descriptor_score"],
                    "score_semantics": "descriptive_relation_score_not_prediction",
                    "event_enable_score": mlp["event_enable_score"],
                    "event_hypothesis": "action_button_may_emit_liquid_toward_open_target",
                    "event_hypothesis_status": "future_temporal_placeholder_not_admitted",
                    "evidence_status": "mlp_scored_hypothesis",
                    "admission_status": "not_admitted",
                    "control_truth_admitted": False,
                    "goal_condition_admitted": False,
                    "action_choice_admitted": False,
                    "solve_condition_admitted": False,
                    "group_bbox_xyxy": group_box,
                    "render_color": "amber",
                    "rendering_role": "review_overlay_marker",
                    "provenance": "derived_from_fail_closed_actor_target_affordances",
                    "mlp_debug": {
                        "hidden_activations": mlp["hidden_activations"],
                    },
                }
            )
    hypotheses.sort(key=lambda row: (-float(row["relation_descriptor_score"]), row["statecon_id"]))
    return hypotheses
