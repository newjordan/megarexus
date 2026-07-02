#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Any

from .relationship_ladder import geoband_completeness_from_inputs

GEOBAND_HYPOTHESIS_SCHEMA = "codexlab.geoband_hypothesis.v1"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_center(box: list[int]) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def bbox_union(boxes: list[list[int]]) -> list[int]:
    return [
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    ]


def node_center_from_bbox(box: list[int]) -> list[float]:
    center = bbox_center(box)
    return [round(center[0], 6), round(center[1], 6)]


def direction_axis(direction: str) -> str:
    if direction in {"left", "right"}:
        return "horizontal"
    if direction in {"up", "down"}:
        return "vertical"
    return "none"


def direction_bearing_degrees(direction: str) -> float:
    return {
        "right": 0.0,
        "down": 90.0,
        "left": 180.0,
        "up": 270.0,
    }.get(direction, 0.0)


def bearing_degrees(origin: tuple[float, float], point: tuple[float, float]) -> float:
    dx = float(point[0]) - float(origin[0])
    dy = float(point[1]) - float(origin[1])
    return round((math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0, 6)


def angular_error_degrees(left: float, right: float) -> float:
    delta = abs((float(left) - float(right) + 180.0) % 360.0 - 180.0)
    return round(delta, 6)


def euclidean_distance(left: tuple[float, float], right: tuple[float, float]) -> float:
    return round(math.dist(left, right), 6)


def relationship_edge(
    *,
    edge_kind: str,
    from_node_id: str,
    to_node_id: str,
    from_center: tuple[float, float],
    to_center: tuple[float, float],
    relation_role: str,
) -> dict[str, Any]:
    return {
        "edge_kind": edge_kind,
        "from_node_id": from_node_id,
        "to_node_id": to_node_id,
        "relation_role": relation_role,
        "bearing_degrees": bearing_degrees(from_center, to_center),
        "distance_pixels": euclidean_distance(from_center, to_center),
    }


def ray_intersects_box(origin: tuple[float, float], direction: str, box: list[int]) -> bool:
    x, y = origin
    x0, y0, x1, y1 = [float(value) for value in box]
    if direction == "up":
        return x0 <= x <= x1 and y >= y1
    if direction == "down":
        return x0 <= x <= x1 and y <= y0
    if direction == "left":
        return y0 <= y <= y1 and x >= x1
    if direction == "right":
        return y0 <= y <= y1 and x <= x0
    return False


def directional_gap(origin: tuple[float, float], direction: str, box: list[int]) -> float:
    x, y = origin
    x0, y0, x1, y1 = [float(value) for value in box]
    if direction == "up":
        return max(0.0, y - y1)
    if direction == "down":
        return max(0.0, y0 - y)
    if direction == "left":
        return max(0.0, x - x1)
    if direction == "right":
        return max(0.0, x0 - x)
    return 64.0


def lateral_offset(origin: tuple[float, float], direction: str, box: list[int]) -> float:
    x, y = origin
    x0, y0, x1, y1 = [float(value) for value in box]
    if direction in {"up", "down"}:
        if x0 <= x <= x1:
            return 0.0
        return min(abs(x - x0), abs(x - x1))
    if direction in {"left", "right"}:
        if y0 <= y <= y1:
            return 0.0
        return min(abs(y - y0), abs(y - y1))
    return 64.0

def build_geoband_hypothesis(
    *,
    statecon_id: str,
    actor: dict[str, Any],
    peer_actors: list[dict[str, Any]],
    target: dict[str, Any],
    support_groups: list[dict[str, Any]],
    symmetry_groups: list[dict[str, Any]] | None = None,
    symmetry_sensors: list[dict[str, Any]] | None = None,
    portband_hypotheses: dict[str, Any] | None = None,
    features: dict[str, float],
) -> dict[str, Any]:
    actor_id = str(actor.get("actor_id"))
    target_id = str(target.get("target_id"))
    actor_center = bbox_center([int(value) for value in actor["actor_bbox_xyxy"]])
    target_center = bbox_center([int(value) for value in target["target_square_xyxy"]])
    relationship_nodes = [
        {
            "node_id": actor_id,
            "node_kind": "actor_affordance",
            "relation_role": "actor",
            "center_xy": [round(actor_center[0], 6), round(actor_center[1], 6)],
            "anchor_kind": "component_bbox_center",
        },
        {
            "node_id": target_id,
            "node_kind": "target_affordance",
            "relation_role": "goal_attachment",
            "center_xy": [round(target_center[0], 6), round(target_center[1], 6)],
            "anchor_kind": "target_square_center",
        },
    ]
    relationship_edges = [
        relationship_edge(
            edge_kind="actor_to_goal_attachment",
            from_node_id=actor_id,
            to_node_id=target_id,
            from_center=actor_center,
            to_center=target_center,
            relation_role="goal_attachment",
        )
    ]
    completeness_status = geoband_completeness_from_inputs(
        has_actor=bool(actor_id),
        has_goal=bool(target_id),
        has_support=bool(support_groups),
    )
    flow_direction = str(actor.get("flow_direction_hypothesis"))
    flow_bearing = direction_bearing_degrees(flow_direction)
    target_bearing = bearing_degrees(actor_center, target_center)
    target_angular_error = angular_error_degrees(flow_bearing, target_bearing)
    support_bearings = []
    support_rays = []
    support_node_ids: list[str] = []
    support_center_by_node_id: dict[str, tuple[float, float]] = {}
    for group in support_groups:
        center = group.get("group_center_xy")
        if not isinstance(center, list) or len(center) != 2:
            continue
        center_xy = [round(float(center[0]), 6), round(float(center[1]), 6)]
        support_node_id = str(group.get("support_group_id"))
        support_node_ids.append(support_node_id)
        support_center = (float(center[0]), float(center[1]))
        support_center_by_node_id[support_node_id] = support_center
        relationship_nodes.append(
            {
                "node_id": support_node_id,
                "node_kind": "statecon_support_group",
                "relation_role": str(group.get("support_kind")),
                "support_relation_category": group.get("support_relation_category"),
                "known_with_unknown_attached_count": group.get("known_with_unknown_attached_count"),
                "group_scope": group.get("group_scope"),
                "center_xy": center_xy,
                "anchor_kind": "support_group_bbox_center",
            }
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="actor_to_support_group",
                from_node_id=actor_id,
                to_node_id=support_node_id,
                from_center=actor_center,
                to_center=support_center,
                relation_role=str(group.get("support_kind")),
            )
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="support_group_to_goal_attachment",
                from_node_id=support_node_id,
                to_node_id=target_id,
                from_center=support_center,
                to_center=target_center,
                relation_role="support_goal_context",
            )
        )
        support_bearings.append(
            {
                "support_group_id": group.get("support_group_id"),
                "support_kind": group.get("support_kind"),
                "support_relation_category": group.get("support_relation_category"),
                "known_with_unknown_attached_count": group.get("known_with_unknown_attached_count"),
                "center_xy": center_xy,
                "bearing_degrees": bearing_degrees(actor_center, (float(center[0]), float(center[1]))),
            }
        )
        support_rays.append(
            {
                "ray_kind": "actor_to_support_group",
                "support_group_id": group.get("support_group_id"),
                "support_kind": group.get("support_kind"),
                "support_relation_category": group.get("support_relation_category"),
                "origin_xy": [round(actor_center[0], 6), round(actor_center[1], 6)],
                "target_xy": center_xy,
                "bearing_degrees": bearing_degrees(actor_center, (float(center[0]), float(center[1]))),
            }
        )
    symmetry_group_rows = [row for row in (symmetry_groups or []) if isinstance(row, dict)]
    symmetry_node_id_by_group_id: dict[str, str] = {}
    symmetry_center_by_group_id: dict[str, tuple[float, float]] = {}
    for group in symmetry_group_rows:
        group_id = str(group.get("symmetry_group_id"))
        box = group.get("bbox_xyxy")
        center = group.get("center_xy")
        if not group_id or not isinstance(box, list) or len(box) != 4:
            continue
        if not isinstance(center, list) or len(center) != 2:
            center = node_center_from_bbox([int(value) for value in box])
        symmetry_node_id = f"{statecon_id}:symmetry_{group_id}"
        symmetry_center = (float(center[0]), float(center[1]))
        symmetry_node_id_by_group_id[group_id] = symmetry_node_id
        symmetry_center_by_group_id[group_id] = symmetry_center
        relationship_nodes.append(
            {
                "node_id": symmetry_node_id,
                "node_kind": "symmetry_group",
                "relation_role": "symmetry_context",
                "symmetry_group_id": group_id,
                "symmetry_kind": group.get("symmetry_kind"),
                "axis": group.get("axis"),
                "score": group.get("score"),
                "member_component_ids": group.get("member_component_ids", []),
                "center_xy": [round(symmetry_center[0], 6), round(symmetry_center[1], 6)],
                "anchor_kind": "symmetry_group_bbox_center",
            }
        )
        member_components = {str(value) for value in group.get("member_component_ids", []) if value is not None}
        actor_components = {
            str(value)
            for value in (actor.get("actor_component_id"), actor.get("orientation_key_component_id"))
            if value
        }
        target_components = {str(target.get("source_component_id"))} if target.get("source_component_id") else set()
        linked_to_network = False
        if member_components.intersection(actor_components) or group.get("symmetry_kind") == "surface_mask_reflection":
            relationship_edges.append(
                relationship_edge(
                    edge_kind="actor_to_symmetry_group",
                    from_node_id=actor_id,
                    to_node_id=symmetry_node_id,
                    from_center=actor_center,
                    to_center=symmetry_center,
                    relation_role="symmetry_context",
                )
            )
            linked_to_network = True
        if member_components.intersection(target_components) or group.get("symmetry_kind") == "surface_mask_reflection":
            relationship_edges.append(
                relationship_edge(
                    edge_kind="symmetry_group_to_goal_attachment",
                    from_node_id=symmetry_node_id,
                    to_node_id=target_id,
                    from_center=symmetry_center,
                    to_center=target_center,
                    relation_role="symmetry_goal_context",
                )
            )
            linked_to_network = True
        for support_node_id, support_group in zip(support_node_ids, support_groups):
            support_components = {str(value) for value in support_group.get("component_ids", []) if value is not None}
            if not member_components.intersection(support_components):
                continue
            support_center = support_center_by_node_id.get(support_node_id)
            if support_center is None:
                continue
            relationship_edges.append(
                relationship_edge(
                    edge_kind="support_group_to_symmetry_group",
                    from_node_id=support_node_id,
                    to_node_id=symmetry_node_id,
                    from_center=support_center,
                    to_center=symmetry_center,
                    relation_role="support_symmetry_context",
                )
            )
            linked_to_network = True
        if not linked_to_network:
            relationship_edges.append(
                relationship_edge(
                    edge_kind="actor_to_symmetry_group",
                    from_node_id=actor_id,
                    to_node_id=symmetry_node_id,
                    from_center=actor_center,
                    to_center=symmetry_center,
                    relation_role="local_symmetry_context",
                )
            )
            relationship_edges.append(
                relationship_edge(
                    edge_kind="symmetry_group_to_goal_attachment",
                    from_node_id=symmetry_node_id,
                    to_node_id=target_id,
                    from_center=symmetry_center,
                    to_center=target_center,
                    relation_role="local_symmetry_goal_context",
                )
            )
    symmetry_sensor_rows = [row for row in (symmetry_sensors or []) if isinstance(row, dict)]
    for sensor in symmetry_sensor_rows:
        sensor_id = str(sensor.get("sensor_id") or "")
        box = sensor.get("bbox_xyxy")
        center = sensor.get("center_xy")
        if not sensor_id or not isinstance(box, list) or len(box) != 4:
            continue
        if not isinstance(center, list) or len(center) != 2:
            center = node_center_from_bbox([int(value) for value in box])
        sensor_node_id = f"{statecon_id}:symmetry_sensor_{sensor_id}"
        sensor_center = (float(center[0]), float(center[1]))
        source_group_id = str(sensor.get("source_symmetry_group_id") or "")
        relationship_nodes.append(
            {
                "node_id": sensor_node_id,
                "node_kind": "symmetry_sensor",
                "relation_role": "symmetry_sensor_context",
                "sensor_layer_id": sensor.get("sensor_layer_id"),
                "sensor_id": sensor_id,
                "sensor_kind": sensor.get("sensor_kind"),
                "source_symmetry_group_id": source_group_id,
                "key_candidate_component_id": sensor.get("key_candidate_component_id"),
                "lock_candidate_component_id": sensor.get("lock_candidate_component_id"),
                "confidence": sensor.get("confidence"),
                "center_xy": [round(sensor_center[0], 6), round(sensor_center[1], 6)],
                "anchor_kind": "symmetry_sensor_bbox_center",
            }
        )
        source_node_id = symmetry_node_id_by_group_id.get(source_group_id)
        source_center = symmetry_center_by_group_id.get(source_group_id)
        if source_node_id is not None and source_center is not None:
            relationship_edges.append(
                relationship_edge(
                    edge_kind="symmetry_group_to_symmetry_sensor",
                    from_node_id=source_node_id,
                    to_node_id=sensor_node_id,
                    from_center=source_center,
                    to_center=sensor_center,
                    relation_role="symmetry_sensor_context",
                )
            )
        else:
            relationship_edges.append(
                relationship_edge(
                    edge_kind="actor_to_symmetry_sensor",
                    from_node_id=actor_id,
                    to_node_id=sensor_node_id,
                    from_center=actor_center,
                    to_center=sensor_center,
                    relation_role="local_symmetry_sensor_context",
                )
            )
    portband = portband_hypotheses if isinstance(portband_hypotheses, dict) else {}
    portband_alignment_rows = [row for row in portband.get("alignments", []) if isinstance(row, dict)]
    portband_fleet = portband.get("portband_fleet") if isinstance(portband.get("portband_fleet"), dict) else {}
    portband_fleet_rows = [row for row in portband_fleet.get("ranked_candidates", []) if isinstance(row, dict)]
    portband_lineup_matrix = portband.get("lineup_matrix") if isinstance(portband.get("lineup_matrix"), dict) else {}
    portband_lineup_rows = [row for row in portband_lineup_matrix.get("rows", []) if isinstance(row, dict)]
    for alignment in portband_alignment_rows:
        alignment_id = str(alignment.get("portband_id") or "")
        centers = alignment.get("source_port_centers_xy")
        if not alignment_id or not isinstance(centers, list) or not centers:
            continue
        valid_centers = [center for center in centers if isinstance(center, list) and len(center) == 2]
        if not valid_centers:
            continue
        alignment_center = (
            sum(float(center[0]) for center in valid_centers) / float(len(valid_centers)),
            sum(float(center[1]) for center in valid_centers) / float(len(valid_centers)),
        )
        portband_node_id = f"{statecon_id}:portband_{alignment_id}"
        relationship_nodes.append(
            {
                "node_id": portband_node_id,
                "node_kind": "portband_alignment",
                "relation_role": "port_alignment_context",
                "portband_id": alignment_id,
                "alignment_kind": alignment.get("alignment_kind"),
                "axis": alignment.get("axis"),
                "alignment_error_px": alignment.get("alignment_error_px"),
                "match_status": alignment.get("match_status"),
                "source_port_ids": alignment.get("source_port_ids", []),
                "target_port_ids": alignment.get("target_port_ids", []),
                "source_object_ids": alignment.get("source_object_ids", []),
                "target_object_ids": alignment.get("target_object_ids", []),
                "center_xy": [round(alignment_center[0], 6), round(alignment_center[1], 6)],
                "anchor_kind": "portband_source_port_center",
            }
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="actor_to_portband_alignment",
                from_node_id=actor_id,
                to_node_id=portband_node_id,
                from_center=actor_center,
                to_center=alignment_center,
                relation_role="port_alignment_context",
            )
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="portband_alignment_to_goal_attachment",
                from_node_id=portband_node_id,
                to_node_id=target_id,
                from_center=alignment_center,
                to_center=target_center,
                relation_role="port_alignment_goal_context",
            )
        )
    for candidate in portband_fleet_rows[:16]:
        candidate_id = str(candidate.get("fleet_candidate_id") or "")
        candidate_box = candidate.get("imagined_combo_bbox_xyxy")
        if not candidate_id or not isinstance(candidate_box, list) or len(candidate_box) != 4:
            continue
        candidate_center = bbox_center([int(value) for value in candidate_box])
        candidate_node_id = f"{statecon_id}:portband_fleet_{candidate_id}"
        relationship_nodes.append(
            {
                "node_id": candidate_node_id,
                "node_kind": "portband_fleet_candidate",
                "relation_role": "combo_shape_matrix_port_fit_context",
                "fleet_candidate_id": candidate_id,
                "fleet_rank": candidate.get("fleet_rank"),
                "rank_score": candidate.get("rank_score"),
                "portband_fit_score": candidate.get("portband_fit_score"),
                "alignment_error_px": candidate.get("alignment_error_px"),
                "combo_size": candidate.get("combo_size"),
                "axis": candidate.get("axis"),
                "source_object_ids": candidate.get("source_object_ids", []),
                "target_object_ids": candidate.get("target_object_ids", []),
                "imagined_combo_bbox_xyxy": [int(value) for value in candidate_box],
                "center_xy": [round(candidate_center[0], 6), round(candidate_center[1], 6)],
                "anchor_kind": "portband_fleet_imagined_combo_bbox_center",
                "evidence_status": candidate.get("evidence_status"),
                "admission_status": candidate.get("admission_status"),
            }
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="actor_to_portband_fleet_candidate",
                from_node_id=actor_id,
                to_node_id=candidate_node_id,
                from_center=actor_center,
                to_center=candidate_center,
                relation_role="combo_shape_matrix_context",
            )
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="portband_fleet_candidate_to_goal_attachment",
                from_node_id=candidate_node_id,
                to_node_id=target_id,
                from_center=candidate_center,
                to_center=target_center,
                relation_role="combo_shape_matrix_goal_context",
            )
        )
    for lineup in portband_lineup_rows[:16]:
        lineup_id = str(lineup.get("lineup_matrix_row_id") or "")
        lineup_box = lineup.get("imagined_lineup_bbox_xyxy")
        if not lineup_id or not isinstance(lineup_box, list) or len(lineup_box) != 4:
            continue
        lineup_center = bbox_center([int(value) for value in lineup_box])
        lineup_node_id = f"{statecon_id}:portband_lineup_{lineup_id}"
        relationship_nodes.append(
            {
                "node_id": lineup_node_id,
                "node_kind": "portband_lineup_matrix_row",
                "relation_role": "lineup_route_confidence_context",
                "lineup_matrix_row_id": lineup_id,
                "source_object_id": lineup.get("source_object_id"),
                "target_object_ids": lineup.get("target_object_ids", []),
                "confidence_score": lineup.get("confidence_score"),
                "confidence_model": lineup.get("confidence_model"),
                "lineup_route_label": lineup.get("lineup_route_label"),
                "transform_hypothesis_xy": lineup.get("transform_hypothesis_xy"),
                "imagined_lineup_bbox_xyxy": [int(value) for value in lineup_box],
                "center_xy": [round(lineup_center[0], 6), round(lineup_center[1], 6)],
                "anchor_kind": "portband_lineup_imagined_bbox_center",
                "evidence_status": lineup.get("evidence_status"),
                "admission_status": lineup.get("admission_status"),
            }
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="actor_to_portband_lineup_matrix_row",
                from_node_id=actor_id,
                to_node_id=lineup_node_id,
                from_center=actor_center,
                to_center=lineup_center,
                relation_role="lineup_route_confidence_context",
            )
        )
    for peer in peer_actors:
        peer_id = str(peer.get("actor_id"))
        if not peer_id or peer_id == actor_id:
            continue
        peer_box = peer.get("actor_bbox_xyxy")
        if not isinstance(peer_box, list) or len(peer_box) != 4:
            continue
        peer_center = bbox_center([int(value) for value in peer_box])
        relationship_nodes.append(
            {
                "node_id": peer_id,
                "node_kind": "actor_affordance",
                "relation_role": "peer_actor",
                "center_xy": [round(peer_center[0], 6), round(peer_center[1], 6)],
                "anchor_kind": "component_bbox_center",
            }
        )
        relationship_edges.append(
            relationship_edge(
                edge_kind="actor_to_peer_actor",
                from_node_id=actor_id,
                to_node_id=peer_id,
                from_center=actor_center,
                to_center=peer_center,
                relation_role="actor_actor_context",
            )
        )
    bearing_values = [target_bearing, *[float(row["bearing_degrees"]) for row in support_bearings]]
    angular_spread = 0.0
    if len(bearing_values) > 1:
        angular_spread = max(angular_error_degrees(bearing_values[0], value) for value in bearing_values[1:])
    target_distance = math.dist(actor_center, target_center)
    angular_tolerance = round(max(5.0, min(45.0, 8.0 + 18.0 * (1.0 - float(features["actor_target_distance_ratio"])))), 6)
    lateral_bandwidth = round(max(1.0, target_distance * math.sin(math.radians(angular_tolerance))), 6)
    symmetry_score = clamp01(
        0.45 * (1.0 - clamp01(target_angular_error / 180.0))
        + 0.35 * float(features["axis_alignment"])
        + 0.20 * float(features["target_open_axis_match"])
    )
    return {
        "schema": GEOBAND_HYPOTHESIS_SCHEMA,
        "status": "hypothesis",
        "geoband_id": f"geoband_{statecon_id}",
        "statecon_id": statecon_id,
        "definition": "geometry_affordance_bandwidth",
        "geometry_kind": "relational_affordance_network",
        "network_kind": "actor_goal_support_peer_relationship_network",
        "required_attachments": ["actor_attachment", "goal_attachment", "support_action_attachment"],
        "present_attachments": [
            "actor_attachment",
            "goal_attachment",
            *("support_action_attachment" for _ in [None] if support_groups),
        ],
        "missing_attachments": [] if support_groups else ["support_action_attachment"],
        "completeness_status": completeness_status,
        "rendering_status": "full_geoband_network" if completeness_status == "complete" else "incomplete_geoband_network_dimmed",
        "members": {
            "actor_affordance_id": actor.get("actor_id"),
            "actor_set_id": actor.get("actor_set_id"),
            "actor_option_rank": actor.get("actor_option_rank"),
            "actor_option_count": actor.get("actor_option_count"),
            "actor_option_component_ids": actor.get("actor_option_component_ids", []),
            "actor_option_relation": actor.get("actor_option_relation"),
            "actor_set_symmetry_group_ids": actor.get("actor_set_symmetry_group_ids", []),
            "target_affordance_id": target.get("target_id"),
            "target_set_id": target.get("target_set_id"),
            "goal_option_rank": target.get("goal_option_rank"),
            "goal_option_count": target.get("goal_option_count"),
            "goal_option_component_ids": target.get("goal_option_component_ids", []),
            "goal_option_relation": target.get("goal_option_relation"),
            "target_set_symmetry_group_id": target.get("target_set_symmetry_group_id"),
            "support_group_ids": [group.get("support_group_id") for group in support_groups],
            "support_relation_categories": [group.get("support_relation_category") for group in support_groups],
            "symmetry_group_ids": [group.get("symmetry_group_id") for group in symmetry_group_rows],
            "symmetry_group_count": len(symmetry_group_rows),
            "symmetry_sensor_ids": [sensor.get("sensor_id") for sensor in symmetry_sensor_rows],
            "symmetry_sensor_count": len(symmetry_sensor_rows),
            "portband_alignment_ids": [row.get("portband_id") for row in portband_alignment_rows],
            "portband_alignment_count": len(portband_alignment_rows),
            "portband_fleet_candidate_ids": [row.get("fleet_candidate_id") for row in portband_fleet_rows],
            "portband_fleet_candidate_count": len(portband_fleet_rows),
            "portband_fleet_preview_node_count": min(16, len(portband_fleet_rows)),
            "portband_lineup_matrix_row_ids": [row.get("lineup_matrix_row_id") for row in portband_lineup_rows],
            "portband_lineup_matrix_row_count": len(portband_lineup_rows),
            "portband_lineup_preview_node_count": min(16, len(portband_lineup_rows)),
            "peer_actor_affordance_ids": [
                peer.get("actor_id")
                for peer in peer_actors
                if isinstance(peer, dict) and peer.get("actor_id") != actor.get("actor_id")
            ],
        },
        "relationship_nodes": relationship_nodes,
        "relationship_edges": relationship_edges,
        "rays": [
            {
                "ray_kind": "actor_flow_axis",
                "origin_xy": [round(actor_center[0], 6), round(actor_center[1], 6)],
                "bearing_degrees": flow_bearing,
            },
            {
                "ray_kind": "actor_to_target",
                "origin_xy": [round(actor_center[0], 6), round(actor_center[1], 6)],
                "target_xy": [round(target_center[0], 6), round(target_center[1], 6)],
                "bearing_degrees": target_bearing,
            },
            *support_rays,
        ],
        "support_group_bearings": support_bearings,
        "angles": {
            "actor_flow_bearing_degrees": flow_bearing,
            "actor_to_target_bearing_degrees": target_bearing,
            "flow_to_target_angular_error_degrees": target_angular_error,
            "support_angular_spread_degrees": round(float(angular_spread), 6),
            "angular_tolerance_degrees": angular_tolerance,
        },
        "bandwidth": {
            "lateral_bandwidth_pixels": lateral_bandwidth,
            "axis_alignment": features["axis_alignment"],
            "directional_proximity": features["directional_proximity"],
            "symmetry_score": round(symmetry_score, 6),
        },
        "evidence_status": "static_geometry_hypothesis",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "derived_from_statecon_branch_geometry",
    }
