"""Affordance builders used by the CONTRA controller head."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from .controller_geometry import (
    bbox_from_component as _bbox_from_component,
    connected_open_region_bbox as _connected_open_region_bbox,
    contact_edge_count as _contact_edge_count,
    direction_between_components as _direction_between_components,
    direction_from_key_to_actor as _direction_from_key_to_actor,
    opposite_direction as _opposite_direction,
    short_digest as _short_digest,
    square_from_bbox as _square_from_bbox,
    union_bbox as _union_bbox,
)

ACTOR_AFFORDANCE_HYPOTHESIS_SCHEMA = "codexlab.actor_affordance_hypothesis.v1"
TARGET_AFFORDANCE_HYPOTHESIS_SCHEMA = "codexlab.target_affordance_hypothesis.v1"


def _oracle_head_module() -> Any | None:
    for module_name in ("oracle", "codexlab.oracle"):
        try:
            return importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 - Oracle is a pluggable sidecar.
            continue
    return None


def _oracle_actor_goal_relation(scene_surfaces: dict[str, Any]) -> dict[str, Any] | None:
    oracle_head = _oracle_head_module()
    if oracle_head is None or not hasattr(oracle_head, "actor_goal_relation_from_contra_scene"):
        return None
    relation = oracle_head.actor_goal_relation_from_contra_scene(scene_surfaces)
    return relation if isinstance(relation, dict) else None


def _parallel_goal_options(
    *,
    goal_candidate: dict[str, Any],
    selected_goal: dict[str, Any],
    actor_component_id: str,
    scene_surfaces: dict[str, Any],
) -> dict[str, Any]:
    selected_component_id = str(selected_goal.get("component_id"))
    selected_fit = selected_goal.get("lock_key_fit") if isinstance(selected_goal.get("lock_key_fit"), dict) else {}
    selected_key_component_id = str(selected_fit.get("key_component_id") or actor_component_id)
    selected_raw = selected_goal.get("raw_pixel_value")
    candidates = goal_candidate.get("candidates") if isinstance(goal_candidate.get("candidates"), list) else []
    option_by_id: dict[str, dict[str, Any]] = {selected_component_id: selected_goal}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("component_id"))
        fit = candidate.get("lock_key_fit") if isinstance(candidate.get("lock_key_fit"), dict) else {}
        if candidate.get("lock_key_selected") is not True:
            continue
        if fit.get("lock_key_shape_fit") is not True:
            continue
        if str(fit.get("key_component_id") or "") != selected_key_component_id:
            continue
        if selected_raw is not None and candidate.get("raw_pixel_value") != selected_raw:
            continue
        option_by_id[candidate_id] = candidate

    options = sorted(
        option_by_id.values(),
        key=lambda row: (
            0 if str(row.get("component_id")) == selected_component_id else 1,
            -float(row.get("score") or 0.0),
            str(row.get("component_id")),
        ),
    )
    option_ids = [str(row.get("component_id")) for row in options]
    symmetry = scene_surfaces.get("symmetry_hypotheses") if isinstance(scene_surfaces.get("symmetry_hypotheses"), dict) else {}
    mirror_group_id = None
    for group in symmetry.get("component_mirror_pair_hypotheses", []) or []:
        if not isinstance(group, dict):
            continue
        members = {str(value) for value in group.get("member_component_ids", []) if value is not None}
        if set(option_ids).issubset(members):
            mirror_group_id = str(group.get("symmetry_group_id"))
            break
    relation = "mirror_pair_same_lock_key_fit" if mirror_group_id else "parallel_same_lock_key_fit"
    return {
        "target_set_id": f"oracle_goal_set_{_short_digest(sorted(option_ids))}",
        "goal_option_component_ids": option_ids,
        "goal_option_count": len(option_ids),
        "goal_option_relation": relation,
        "target_set_symmetry_group_id": mirror_group_id,
        "options": options,
    }


def _component_graph_rows(scene_surfaces: dict[str, Any]) -> list[dict[str, Any]]:
    graph = scene_surfaces.get("component_graph") if isinstance(scene_surfaces.get("component_graph"), dict) else {}
    return [row for row in graph.get("components", []) or [] if isinstance(row, dict)]


def _parallel_actor_options(
    *,
    actor_candidate: dict[str, Any],
    selected_actor: dict[str, Any],
    goal_options: dict[str, Any],
    scene_surfaces: dict[str, Any],
) -> dict[str, Any]:
    selected_component_id = str(selected_actor.get("component_id"))
    selected_raw = selected_actor.get("raw_pixel_value")
    selected_area = selected_actor.get("area_pixels")
    selected_shape_rows = selected_actor.get("shape_mask_rows")
    goal_ids = {str(value) for value in goal_options.get("goal_option_component_ids", []) if value is not None}
    option_by_id: dict[str, dict[str, Any]] = {selected_component_id: selected_actor}

    for candidate in _component_graph_rows(scene_surfaces):
        candidate_id = str(candidate.get("component_id"))
        if candidate_id in goal_ids:
            continue
        if selected_raw is not None and candidate.get("raw_pixel_value") != selected_raw:
            continue
        if selected_area is not None and candidate.get("area_pixels") != selected_area:
            continue
        if selected_shape_rows is not None and candidate.get("shape_mask_rows") != selected_shape_rows:
            continue
        option_by_id[candidate_id] = candidate

    for candidate in actor_candidate.get("candidates", []) or []:
        if not isinstance(candidate, dict):
            continue
        candidate_id = str(candidate.get("component_id"))
        if candidate_id in goal_ids:
            continue
        criteria = candidate.get("criteria") if isinstance(candidate.get("criteria"), dict) else {}
        if criteria.get("fits_selected_goal_lock") is True:
            option_by_id[candidate_id] = candidate

    options = sorted(
        option_by_id.values(),
        key=lambda row: (
            0 if str(row.get("component_id")) == selected_component_id else 1,
            int((row.get("bbox_xyxy") or [0, 0, 0, 0])[1]),
            int((row.get("bbox_xyxy") or [0, 0, 0, 0])[0]),
            str(row.get("component_id")),
        ),
    )
    option_ids = [str(row.get("component_id")) for row in options]
    option_id_set = set(option_ids)
    symmetry = scene_surfaces.get("symmetry_hypotheses") if isinstance(scene_surfaces.get("symmetry_hypotheses"), dict) else {}
    mirror_group_ids: list[str] = []
    for group in symmetry.get("component_mirror_pair_hypotheses", []) or []:
        if not isinstance(group, dict):
            continue
        members = {str(value) for value in group.get("member_component_ids", []) if value is not None}
        if len(option_id_set.intersection(members)) >= 2:
            mirror_group_ids.append(str(group.get("symmetry_group_id")))
    relation = "mirror_pair_same_actor_key_shape" if mirror_group_ids else "parallel_same_actor_key_shape"
    return {
        "actor_set_id": f"oracle_actor_set_{_short_digest(sorted(option_ids))}",
        "actor_option_component_ids": option_ids,
        "actor_option_count": len(option_ids),
        "actor_option_relation": relation,
        "actor_set_symmetry_group_ids": sorted(set(mirror_group_ids)),
        "options": options,
    }


def oracle_actor_affordance_hypotheses(
    *,
    scene_surfaces: dict[str, Any],
    actor_expectation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    relation = _oracle_actor_goal_relation(scene_surfaces)
    if not isinstance(relation, dict):
        return []
    actor_candidate = relation.get("actor_candidate") if isinstance(relation.get("actor_candidate"), dict) else {}
    goal_candidate = relation.get("goal_candidate") if isinstance(relation.get("goal_candidate"), dict) else {}
    actor = actor_candidate.get("selected_component") if isinstance(actor_candidate.get("selected_component"), dict) else None
    goal = goal_candidate.get("selected_component") if isinstance(goal_candidate.get("selected_component"), dict) else None
    if not isinstance(actor, dict) or not isinstance(goal, dict):
        return []
    actor_component_id = str(actor.get("component_id"))
    goal_component_id = str(goal.get("component_id"))
    goal_options = _parallel_goal_options(
        goal_candidate=goal_candidate,
        selected_goal=goal,
        actor_component_id=actor_component_id,
        scene_surfaces=scene_surfaces,
    )
    actor_options = _parallel_actor_options(
        actor_candidate=actor_candidate,
        selected_actor=actor,
        goal_options=goal_options,
        scene_surfaces=scene_surfaces,
    )
    arrow = relation.get("key_points_to_lock_hypothesis") if isinstance(relation.get("key_points_to_lock_hypothesis"), dict) else {}
    rows: list[dict[str, Any]] = []
    for option_rank, option in enumerate(actor_options["options"], 1):
        option_component_id = str(option.get("component_id"))
        flow_direction = str(
            arrow.get("key_to_lock_direction") if option_component_id == actor_component_id else ""
        ) or _direction_between_components(option, goal)
        if flow_direction not in {"up", "down", "left", "right"}:
            continue
        actor_bbox = _bbox_from_component(option)
        rows.append(
            {
                "schema": ACTOR_AFFORDANCE_HYPOTHESIS_SCHEMA,
                "status": "hypothesis",
                "actor_id": f"oracle_actor_key_{option_component_id}",
                "actor_kind": "oracle_first_pass_key_actor_candidate",
                "object_role": "payload_source",
                "control_effector_confirmed": False,
                "object_role_basis": "oracle_lock_key_key_is_routing_payload_source_unless_deltaband_motion_lock_confirms_effector",
                "source": "oracle_first_pass_actor_goal_relation",
                "source_component_role": "residual_component_hypothesis",
                "actor_component_id": option_component_id,
                "source_component_id": option_component_id,
                "actor_raw_pixel_value": int(option.get("raw_pixel_value")),
                "orientation_key_component_id": None,
                "orientation_key_raw_pixel_value": None,
                "orientation_key_role": "none_oracle_key_is_actor",
                "group_component_ids": [option_component_id],
                "selected_actor_component_id": actor_component_id,
                "actor_set_id": actor_options["actor_set_id"],
                "actor_option_rank": option_rank,
                "actor_option_count": actor_options["actor_option_count"],
                "actor_option_component_ids": actor_options["actor_option_component_ids"],
                "parallel_actor_component_ids": actor_options["actor_option_component_ids"],
                "parallel_actor_option_count": actor_options["actor_option_count"],
                "actor_option_relation": actor_options["actor_option_relation"],
                "actor_set_symmetry_group_ids": actor_options["actor_set_symmetry_group_ids"],
                "paired_goal_component_id": goal_component_id,
                "paired_goal_component_ids": goal_options["goal_option_component_ids"],
                "target_set_id": goal_options["target_set_id"],
                "goal_option_count": goal_options["goal_option_count"],
                "goal_option_relation": goal_options["goal_option_relation"],
                "target_set_symmetry_group_id": goal_options["target_set_symmetry_group_id"],
                "direction_rule": "oracle_key_points_to_goal_lock",
                "flow_direction_hypothesis": flow_direction,
                "direction_from_raw_geometry": _direction_between_components(actor, goal),
                "direction_agrees_with_raw_geometry": _direction_between_components(actor, goal) == flow_direction,
                "activation_condition_hypothesis": "oracle_first_pass_actor_goal_relation",
                "evidence_status": "oracle_first_pass_hypothesis",
                "admission_status": "not_admitted",
                "not_prediction": True,
                "not_control_truth": True,
                "not_solve_truth": True,
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "actor_bbox_xyxy": [int(v) for v in actor_bbox],
                "orientation_key_bbox_xyxy": None,
                "group_bbox_xyxy": [int(v) for v in actor_bbox],
                "contact_edge_count": 0,
                "oracle_relation_schema": relation.get("schema"),
                "oracle_relation_kind": arrow.get("relation_kind"),
                "oracle_key_to_lock_direction": flow_direction,
                "render_color": "bright_purple",
                "direction_render_color": "cyan",
                "provenance": "oracle_first_pass_review_hypothesis_without_human_config_gate",
                "rendering_role": "review_overlay_marker",
            }
        )
    return rows


def oracle_target_affordance_hypotheses(
    *,
    frame_shape: tuple[int, int],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    scene_surfaces: dict[str, Any],
    bucket_expectation: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    relation = _oracle_actor_goal_relation(scene_surfaces)
    if not isinstance(relation, dict):
        return []
    goal_candidate = relation.get("goal_candidate") if isinstance(relation.get("goal_candidate"), dict) else {}
    actor_candidate = relation.get("actor_candidate") if isinstance(relation.get("actor_candidate"), dict) else {}
    goal = goal_candidate.get("selected_component") if isinstance(goal_candidate.get("selected_component"), dict) else None
    actor = actor_candidate.get("selected_component") if isinstance(actor_candidate.get("selected_component"), dict) else None
    if not isinstance(goal, dict) or not isinstance(actor, dict):
        return []
    goal_component_id = str(goal.get("component_id"))
    actor_component_id = str(actor.get("component_id"))
    arrow = relation.get("key_points_to_lock_hypothesis") if isinstance(relation.get("key_points_to_lock_hypothesis"), dict) else {}
    goal_options = _parallel_goal_options(
        goal_candidate=goal_candidate,
        selected_goal=goal,
        actor_component_id=actor_component_id,
        scene_surfaces=scene_surfaces,
    )
    actor_options = _parallel_actor_options(
        actor_candidate=actor_candidate,
        selected_actor=actor,
        goal_options=goal_options,
        scene_surfaces=scene_surfaces,
    )
    rows: list[dict[str, Any]] = []
    for option_rank, option in enumerate(goal_options["options"], 1):
        option_component_id = str(option.get("component_id"))
        key_to_lock_direction = str(
            arrow.get("key_to_lock_direction") if option_component_id == goal_component_id else ""
        ) or _direction_between_components(actor, option)
        open_direction = _opposite_direction(key_to_lock_direction)
        if open_direction not in {"up", "down", "left", "right"}:
            continue
        label_index = component_lookup.get(option_component_id)
        x0, y0, x1, y1 = _bbox_from_component(option)
        target_bbox = (x0, y0, x1, y1)
        if label_index is not None:
            component_mask = label_map[y0 : y1 + 1, x0 : x1 + 1] == int(label_index)
            region_bbox = _connected_open_region_bbox(component_mask, open_direction)
            if region_bbox is not None:
                rx0, ry0, rx1, ry1 = region_bbox
                target_bbox = (x0 + rx0, y0 + ry0, x0 + rx1, y0 + ry1)
        rows.append(
            {
                "schema": TARGET_AFFORDANCE_HYPOTHESIS_SCHEMA,
                "status": "hypothesis",
                "target_id": f"oracle_goal_lock_{option_component_id}",
                "target_kind": "oracle_first_pass_lock_goal_candidate",
                "object_role": "goal_sink",
                "object_role_basis": "oracle_lock_goal_candidate_is_routing_sink",
                "source": "oracle_first_pass_actor_goal_relation",
                "source_component_role": "residual_component_hypothesis",
                "component_id": option_component_id,
                "source_component_id": option_component_id,
                "source_raw_pixel_value": int(option.get("raw_pixel_value")),
                "source_shape_family": "oracle_lock_key_shape_fit",
                "paired_actor_component_id": actor_component_id,
                "paired_actor_component_ids": actor_options["actor_option_component_ids"],
                "actor_set_id": actor_options["actor_set_id"],
                "actor_option_count": actor_options["actor_option_count"],
                "actor_option_relation": actor_options["actor_option_relation"],
                "actor_set_symmetry_group_ids": actor_options["actor_set_symmetry_group_ids"],
                "selected_goal_component_id": goal_component_id,
                "target_set_id": goal_options["target_set_id"],
                "goal_option_rank": option_rank,
                "goal_option_count": goal_options["goal_option_count"],
                "goal_option_component_ids": goal_options["goal_option_component_ids"],
                "parallel_goal_component_ids": goal_options["goal_option_component_ids"],
                "parallel_goal_option_count": goal_options["goal_option_count"],
                "goal_option_relation": goal_options["goal_option_relation"],
                "target_set_symmetry_group_id": goal_options["target_set_symmetry_group_id"],
                "open_direction": open_direction,
                "key_to_lock_direction": key_to_lock_direction,
                "evidence_status": "oracle_first_pass_hypothesis",
                "admission_status": "not_admitted",
                "not_prediction": True,
                "not_control_truth": True,
                "not_solve_truth": True,
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
                "target_bbox_xyxy": [int(v) for v in target_bbox],
                "target_square_xyxy": _square_from_bbox(target_bbox, frame_shape=frame_shape),
                "oracle_relation_schema": relation.get("schema"),
                "oracle_relation_kind": arrow.get("relation_kind"),
                "render_color": "bright_green",
                "provenance": "oracle_first_pass_review_hypothesis_without_human_config_gate",
                "rendering_role": "review_overlay_marker",
            }
        )
    return rows


def calibration_actor_affordance_hypotheses(
    *,
    scene_surfaces: dict[str, Any],
    actor_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(actor_expectation, dict) or actor_expectation.get("status") != "user_supplied_answer":
        return oracle_actor_affordance_hypotheses(scene_surfaces=scene_surfaces)
    components = {
        str(component.get("component_id")): component
        for component in scene_surfaces.get("component_graph", {}).get("components", [])
        if isinstance(component, dict)
    }
    rows: list[dict[str, Any]] = []
    direction_rule = str(actor_expectation.get("direction_rule"))
    for group in actor_expectation.get("component_groups", []):
        if not isinstance(group, dict):
            continue
        actor_component_id = str(group.get("actor_component_id"))
        key_component_id = str(group.get("orientation_key_component_id"))
        actor = components.get(actor_component_id)
        key = components.get(key_component_id)
        expected_direction = str(group.get("flow_direction"))
        if actor is None or key is None or expected_direction not in {"up", "down", "left", "right"}:
            continue
        inferred_direction = _direction_from_key_to_actor(actor, key)
        actor_bbox = _bbox_from_component(actor)
        key_bbox = _bbox_from_component(key)
        rows.append(
            {
                "schema": ACTOR_AFFORDANCE_HYPOTHESIS_SCHEMA,
                "status": "user_supplied_answer",
                "actor_id": f"actor_emitter_{actor_component_id}",
                "actor_kind": str(actor_expectation.get("actor_kind")),
                "object_role": "effector",
                "control_effector_confirmed": True,
                "object_role_basis": "user_calibration_declared_control_actor",
                "source": "contra_controller_calibration_adapter",
                "source_component_role": "residual_component_hypothesis",
                "actor_component_id": actor_component_id,
                "source_component_id": actor_component_id,
                "actor_raw_pixel_value": int(actor.get("raw_pixel_value")),
                "orientation_key_component_id": key_component_id,
                "orientation_key_raw_pixel_value": int(key.get("raw_pixel_value")),
                "orientation_key_role": "flow_direction_marker",
                "group_component_ids": [actor_component_id, key_component_id],
                "direction_rule": direction_rule,
                "flow_direction_hypothesis": expected_direction,
                "direction_from_raw_geometry": inferred_direction,
                "direction_agrees_with_raw_geometry": inferred_direction == expected_direction,
                "activation_condition_hypothesis": "action_button",
                "evidence_status": "manual_hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "actor_bbox_xyxy": [int(v) for v in actor_bbox],
                "orientation_key_bbox_xyxy": [int(v) for v in key_bbox],
                "group_bbox_xyxy": _union_bbox(actor_bbox, key_bbox),
                "contact_edge_count": _contact_edge_count(scene_surfaces, actor_component_id, key_component_id),
                "render_color": "bright_purple",
                "direction_render_color": "cyan",
                "provenance": actor_expectation.get("provenance"),
                "rendering_role": "review_overlay_marker",
            }
        )
    return rows


def calibration_target_affordance_hypotheses(
    *,
    frame_shape: tuple[int, int],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    scene_surfaces: dict[str, Any],
    bucket_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(bucket_expectation, dict) or bucket_expectation.get("status") != "user_supplied_answer":
        return oracle_target_affordance_hypotheses(
            frame_shape=frame_shape,
            label_map=label_map,
            component_lookup=component_lookup,
            scene_surfaces=scene_surfaces,
        )
    components = {
        str(component.get("component_id")): component
        for component in scene_surfaces.get("component_graph", {}).get("components", [])
        if isinstance(component, dict)
    }
    rows: list[dict[str, Any]] = []
    for hypothesis in bucket_expectation.get("manual_asymmetry_hypotheses", []):
        if not isinstance(hypothesis, dict) or hypothesis.get("asymmetry_makes_open_target") is not True:
            continue
        component_id = str(hypothesis.get("component_id"))
        open_direction = str(hypothesis.get("open_direction"))
        component = components.get(component_id)
        label_index = component_lookup.get(component_id)
        if component is None or label_index is None or open_direction not in {"up", "down", "left", "right"}:
            continue
        x0, y0, x1, y1 = _bbox_from_component(component)
        component_mask = label_map[y0 : y1 + 1, x0 : x1 + 1] == int(label_index)
        region_bbox = _connected_open_region_bbox(component_mask, open_direction)
        if region_bbox is None:
            target_bbox = (x0, y0, x1, y1)
        else:
            rx0, ry0, rx1, ry1 = region_bbox
            target_bbox = (x0 + rx0, y0 + ry0, x0 + rx1, y0 + ry1)
        rows.append(
            {
                "schema": TARGET_AFFORDANCE_HYPOTHESIS_SCHEMA,
                "status": "user_supplied_answer",
                "target_id": f"target_open_{component_id}",
                "target_kind": "open_target",
                "object_role": "goal_sink",
                "object_role_basis": "user_calibration_declared_open_target",
                "source": "contra_controller_calibration_adapter",
                "source_component_role": "residual_component_hypothesis",
                "component_id": component_id,
                "source_component_id": component_id,
                "source_raw_pixel_value": int(component.get("raw_pixel_value")),
                "source_shape_family": str(bucket_expectation.get("shape_family")),
                "open_direction": open_direction,
                "evidence_status": "manual_hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "target_bbox_xyxy": [int(v) for v in target_bbox],
                "target_square_xyxy": _square_from_bbox(target_bbox, frame_shape=frame_shape),
                "render_color": "bright_green",
                "provenance": bucket_expectation.get("provenance"),
                "rendering_role": "review_overlay_marker",
            }
        )
    return rows


def empty_actor_affordance_hypotheses(
    *,
    scene_surfaces: dict[str, Any],
    actor_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return []


def empty_target_affordance_hypotheses(
    *,
    frame_shape: tuple[int, int],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    scene_surfaces: dict[str, Any],
    bucket_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return []
