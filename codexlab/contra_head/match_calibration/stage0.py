#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from .common import admission_flags, calibration_id, complete_artifact_fields, ladder_id, missing_artifact_fields
from .like_controls import LIKE_CONTROL_PRIOR_CONTEXT_STATUS
from .transfer_prior import MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS

MATCH_CALIBRATION_STAGE0_REQUIREMENT_SCHEMA = "codexlab.match_calibration.stage0_init_canvas_click_requirement.v1"
MATCH_CALIBRATION_STAGE0_OBSERVATION_SCHEMA = "codexlab.match_calibration.stage0_init_canvas_click_observation.v1"
MATCH_CALIBRATION_STAGE0_ID = "deltaband_stage_0_init_calibration"
MATCH_CALIBRATION_STAGE0_EVENT_KIND = "init_canvas_click"
MATCH_CALIBRATION_STAGE0_REQUIRED_ARTIFACTS = [
    "pre_init_frame",
    "init_click_invocation_log",
    "post_init_frame_sequence",
    "post_init_contra_packet",
    "init_delta_classification",
    "initialized_baseline_frame",
]
MATCH_CALIBRATION_STAGE0_CLICK_POLICY = "pre_target_relational_novelty_click_policy_v1"
GEOBAND_CONTEXT_STATUS = "static_relational_context_not_target_truth"
MACROBAND_CONTEXT_STATUS = "static_multisensor_object_ranking_not_control_truth"


def _component_rows(raw_color_component_packet: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_color_component_packet, dict):
        return rows
    total_pixels = sum(
        int(color.get("pixel_count") or 0)
        for color in raw_color_component_packet.get("colors", [])
        if isinstance(color, dict)
    )
    for color in raw_color_component_packet.get("colors", []):
        if not isinstance(color, dict):
            continue
        color_pixels = int(color.get("pixel_count") or 0)
        for component in color.get("components", []):
            if not isinstance(component, dict) or not component.get("component_id"):
                continue
            row = dict(component)
            row["palette_name"] = color.get("palette_name")
            row["color_pixel_count"] = color_pixels
            row["frame_color_pixel_total"] = total_pixels
            rows[str(component["component_id"])] = row
    return rows


def _ids_from_rows(rows: list[dict[str, Any]] | None, *keys: str) -> set[str]:
    ids: set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                ids.update(str(item) for item in value if item is not None)
            elif value is not None:
                ids.add(str(value))
    return ids


def _surface_component_ids(scene_surfaces: dict[str, Any] | None) -> set[str]:
    ids: set[str] = set()
    if not isinstance(scene_surfaces, dict):
        return ids
    for key in ("floor_hypothesis",):
        row = scene_surfaces.get(key)
        if isinstance(row, dict) and row.get("component_id") is not None:
            ids.add(str(row["component_id"]))
    for key in ("background_hypotheses", "wall_hypotheses", "floor_pattern_hypotheses"):
        rows = scene_surfaces.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            ids.update(str(value) for value in row.get("component_ids", []) if value is not None)
            if row.get("component_id") is not None:
                ids.add(str(row["component_id"]))
    return ids


def _geobands_from_statecons(statecons: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    geobands: list[dict[str, Any]] = []
    for statecon in statecons or []:
        if not isinstance(statecon, dict):
            continue
        geoband = statecon.get("geoband_hypothesis")
        if isinstance(geoband, dict):
            geobands.append(geoband)
    return geobands


def _append_unique(values: list[str], value: Any) -> None:
    if value is None:
        return
    text = str(value)
    if text and text not in values:
        values.append(text)


def _symmetry_context_by_component(
    symmetry_hypotheses: dict[str, Any] | None,
    *,
    actor_like_ids: set[str],
    geobands: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    context: dict[str, dict[str, Any]] = {}
    if not isinstance(symmetry_hypotheses, dict):
        return context

    group_to_members: dict[str, set[str]] = {}
    group_to_kind: dict[str, str] = {}
    sensor_to_members: dict[str, set[str]] = {}

    def row_for(component_id: str) -> dict[str, Any]:
        return context.setdefault(
            component_id,
            {
                "component_id": component_id,
                "symmetry_group_ids": [],
                "symmetry_sensor_ids": [],
                "static_geoband_context_ids": [],
                "static_geoband_symmetry_group_ids": [],
                "static_geoband_symmetry_sensor_ids": [],
                "linked_actor_like_component_ids": [],
                "symmetry_kinds": [],
                "mirror_group_ids": [],
                "reverse_group_ids": [],
                "polar_group_ids": [],
                "lock_key_sensor_ids": [],
                "max_symmetry_score": 0.0,
                "polar_reverse_score": 0.0,
                "mirror_score": 0.0,
                "sensor_confidence": 0.0,
            },
        )

    def add_group(group: dict[str, Any], *, kind: str) -> None:
        group_id = str(group.get("symmetry_group_id") or "")
        members = {str(value) for value in group.get("member_component_ids", []) if value is not None}
        if not group_id or not members:
            return
        group_to_members[group_id] = set(members)
        group_to_kind[group_id] = kind
        score = float(group.get("score") or 0.0)
        for component_id in members:
            row = row_for(component_id)
            _append_unique(row["symmetry_group_ids"], group_id)
            _append_unique(row["symmetry_kinds"], kind)
            row["max_symmetry_score"] = max(float(row["max_symmetry_score"]), score)
            if kind == "mirror":
                _append_unique(row["mirror_group_ids"], group_id)
                row["mirror_score"] = max(float(row["mirror_score"]), score)
            elif kind == "reverse":
                _append_unique(row["reverse_group_ids"], group_id)
                row["polar_reverse_score"] = max(float(row["polar_reverse_score"]), score)
            elif kind == "polar":
                _append_unique(row["polar_group_ids"], group_id)
                row["polar_reverse_score"] = max(float(row["polar_reverse_score"]), score)
            for actor_id in sorted(actor_like_ids.intersection(members - {component_id})):
                _append_unique(row["linked_actor_like_component_ids"], actor_id)

    for key, kind in (
        ("component_self_symmetry_hypotheses", "self"),
        ("component_mirror_pair_hypotheses", "mirror"),
        ("component_reverse_symmetry_hypotheses", "reverse"),
        ("component_polar_symmetry_hypotheses", "polar"),
    ):
        for group in symmetry_hypotheses.get(key, []) or []:
            if isinstance(group, dict):
                add_group(group, kind=kind)

    for sensor in symmetry_hypotheses.get("lock_key_sensor_hypotheses", []) or []:
        if not isinstance(sensor, dict):
            continue
        sensor_id = str(sensor.get("sensor_id") or "")
        members = {str(value) for value in sensor.get("member_component_ids", []) if value is not None}
        for key in ("key_candidate_component_id", "lock_candidate_component_id"):
            if sensor.get(key) is not None:
                members.add(str(sensor[key]))
        if not sensor_id or not members:
            continue
        sensor_to_members[sensor_id] = set(members)
        confidence = float(sensor.get("confidence") or 0.0)
        source_group_id = sensor.get("source_symmetry_group_id")
        for component_id in members:
            row = row_for(component_id)
            _append_unique(row["symmetry_sensor_ids"], sensor_id)
            _append_unique(row["lock_key_sensor_ids"], sensor_id)
            _append_unique(row["symmetry_kinds"], "lock_key_sensor")
            if source_group_id:
                _append_unique(row["symmetry_group_ids"], source_group_id)
            row["sensor_confidence"] = max(float(row["sensor_confidence"]), confidence)
            for actor_id in sorted(actor_like_ids.intersection(members - {component_id})):
                _append_unique(row["linked_actor_like_component_ids"], actor_id)

    for geoband in geobands:
        geoband_id = str(geoband.get("geoband_id") or geoband.get("statecon_id") or "")
        members = geoband.get("members") if isinstance(geoband.get("members"), dict) else {}
        if not geoband_id:
            continue
        context_components: set[str] = set()
        for group_id in members.get("symmetry_group_ids", []) or []:
            group_id = str(group_id)
            context_components.update(group_to_members.get(group_id, set()))
            for component_id in group_to_members.get(group_id, set()):
                row = row_for(component_id)
                _append_unique(row["static_geoband_context_ids"], geoband_id)
                _append_unique(row["static_geoband_symmetry_group_ids"], group_id)
                _append_unique(row["symmetry_kinds"], f"geoband_{group_to_kind.get(group_id, 'symmetry')}")
        for sensor_id in members.get("symmetry_sensor_ids", []) or []:
            sensor_id = str(sensor_id)
            context_components.update(sensor_to_members.get(sensor_id, set()))
            for component_id in sensor_to_members.get(sensor_id, set()):
                row = row_for(component_id)
                _append_unique(row["static_geoband_context_ids"], geoband_id)
                _append_unique(row["static_geoband_symmetry_sensor_ids"], sensor_id)
                _append_unique(row["symmetry_kinds"], "geoband_lock_key_sensor")
        for component_id in context_components:
            _append_unique(row_for(component_id)["static_geoband_context_ids"], geoband_id)
    return context


def _component_asymmetry(component: dict[str, Any]) -> dict[str, Any]:
    rows = component.get("shape_mask_rows")
    if not isinstance(rows, list) or not rows:
        return {"score": 0.0, "axis": "none", "mirror_x_difference": 0.0, "mirror_y_difference": 0.0}
    mask = [[ch == "1" for ch in str(row)] for row in rows]
    height = len(mask)
    width = max((len(row) for row in mask), default=0)
    if height <= 0 or width <= 0:
        return {"score": 0.0, "axis": "none", "mirror_x_difference": 0.0, "mirror_y_difference": 0.0}
    padded = [row + [False] * (width - len(row)) for row in mask]
    area = sum(1 for row in padded for value in row if value)
    if area <= 0:
        return {"score": 0.0, "axis": "none", "mirror_x_difference": 0.0, "mirror_y_difference": 0.0}
    x_diff = sum(
        1
        for y in range(height)
        for x in range(width)
        if padded[y][x] != padded[y][width - 1 - x]
    ) / float(max(1, 2 * area))
    y_diff = sum(
        1
        for y in range(height)
        for x in range(width)
        if padded[y][x] != padded[height - 1 - y][x]
    ) / float(max(1, 2 * area))
    score = max(0.0, min(1.0, max(x_diff, y_diff)))
    axis = "horizontal" if x_diff >= y_diff and score > 0 else "vertical" if score > 0 else "none"
    return {
        "score": round(score, 6),
        "axis": axis,
        "mirror_x_difference": round(max(0.0, min(1.0, x_diff)), 6),
        "mirror_y_difference": round(max(0.0, min(1.0, y_diff)), 6),
    }


def _color_salience(component: dict[str, Any]) -> float:
    total = float(component.get("frame_color_pixel_total") or 0)
    color_pixels = float(component.get("color_pixel_count") or 0)
    if total <= 0:
        return 0.0
    return round(max(0.0, min(1.0, 1.0 - (color_pixels / total))), 6)


def _support_component_ids_from_geobands(geobands: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_component: dict[str, list[str]] = {}
    for geoband in geobands:
        geoband_id = str(geoband.get("geoband_id") or geoband.get("statecon_id") or "")
        members = geoband.get("members") if isinstance(geoband.get("members"), dict) else {}
        for support_id in members.get("support_group_ids", []):
            if not isinstance(support_id, str) or "_candidate_" not in support_id:
                continue
            component_id = support_id.rsplit("_candidate_", 1)[-1]
            if component_id:
                by_component.setdefault(component_id, []).append(geoband_id)
    return by_component


def _transfer_prior_by_component(transfer_prior: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(transfer_prior, dict):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in transfer_prior.get("component_role_reweights", []) or []:
        if isinstance(row, dict) and row.get("component_id"):
            rows[str(row["component_id"])] = row
    return rows


def build_stage0_state_change_click_ranking(
    *,
    raw_color_component_packet: dict[str, Any] | None = None,
    scene_surfaces: dict[str, Any] | None = None,
    actor_affordances: list[dict[str, Any]] | None = None,
    target_affordances: list[dict[str, Any]] | None = None,
    support_affordances: list[dict[str, Any]] | None = None,
    statecons: list[dict[str, Any]] | None = None,
    symmetry_hypotheses: dict[str, Any] | None = None,
    action_result_transfer_prior: dict[str, Any] | None = None,
    like_control_group_prior: dict[str, Any] | None = None,
    macroband_object_ranking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components = _component_rows(raw_color_component_packet)
    geobands = _geobands_from_statecons(statecons)
    support_geobands = _support_component_ids_from_geobands(geobands)
    actor_ids = _ids_from_rows(actor_affordances, "actor_component_id", "source_component_id", "group_component_ids")
    target_ids = _ids_from_rows(target_affordances, "component_id", "source_component_id")
    surface_ids = _surface_component_ids(scene_surfaces)
    excluded_ids = target_ids | surface_ids
    symmetry_context = _symmetry_context_by_component(symmetry_hypotheses, actor_like_ids=actor_ids, geobands=geobands)
    transfer_by_component = _transfer_prior_by_component(action_result_transfer_prior)
    like_control_context = {
        str(row.get("component_id")): row
        for row in (like_control_group_prior or {}).get("component_like_control_context", [])
        if isinstance(row, dict) and row.get("component_id") is not None
    } if isinstance(like_control_group_prior, dict) else {}
    macroband_by_component = {
        str(row.get("component_id")): row
        for row in (macroband_object_ranking or {}).get("ranked_objects", [])
        if isinstance(row, dict) and row.get("component_id") is not None
    } if isinstance(macroband_object_ranking, dict) else {}
    primary_actor_prior_ids = {
        str(value)
        for value in (action_result_transfer_prior or {}).get("primary_actor_prior_component_ids", [])
        if value is not None
    } if isinstance(action_result_transfer_prior, dict) else set()
    support_by_component = {
        str(row.get("source_component_id") or row.get("component_id")): row
        for row in support_affordances or []
        if isinstance(row, dict) and (row.get("source_component_id") or row.get("component_id"))
    }
    rankings: list[dict[str, Any]] = []
    asymmetry_by_component = {component_id: _component_asymmetry(component) for component_id, component in components.items()}
    asymmetry_candidate_ids = {
        component_id
        for component_id, asymmetry in asymmetry_by_component.items()
        if float(asymmetry.get("score") or 0.0) > 0.0
    }
    relational_candidate_ids = {
        component_id
        for component_id, context in symmetry_context.items()
        if context.get("static_geoband_context_ids")
        or context.get("linked_actor_like_component_ids")
        or context.get("lock_key_sensor_ids")
        or context.get("polar_group_ids")
        or context.get("reverse_group_ids")
    }
    candidate_ids = sorted(
        set(support_by_component)
        | set(support_geobands)
        | asymmetry_candidate_ids
        | relational_candidate_ids
        | set(transfer_by_component)
        | set(like_control_context)
        | set(macroband_by_component)
    )
    for component_id in candidate_ids:
        component = components.get(component_id)
        if not isinstance(component, dict) or component_id in excluded_ids:
            continue
        area = int(component.get("area_pixels") or 0)
        total = int(component.get("frame_color_pixel_total") or 0)
        if total > 0 and float(area) / float(total) > 0.35:
            continue
        asymmetry = asymmetry_by_component.get(component_id, _component_asymmetry(component))
        color_salience = _color_salience(component)
        support = support_by_component.get(component_id, {})
        geoband_ids = sorted(set(support_geobands.get(component_id, [])))
        context = symmetry_context.get(component_id, {})
        static_geoband_context_ids = sorted(set(context.get("static_geoband_context_ids", [])))
        linked_actor_like_ids = sorted(set(context.get("linked_actor_like_component_ids", [])))
        symmetry_sensor_ids = sorted(set(context.get("symmetry_sensor_ids", [])))
        symmetry_group_ids = sorted(set(context.get("symmetry_group_ids", [])))
        support_score = float(support.get("score") or 0.0) if isinstance(support, dict) else 0.0
        transfer_row = transfer_by_component.get(component_id, {})
        like_control_row = like_control_context.get(component_id, {})
        macroband_row = macroband_by_component.get(component_id, {})
        macroband_signals = macroband_row.get("signals") if isinstance(macroband_row.get("signals"), dict) else {}
        actor_origin_prior_confidence = float(transfer_row.get("max_actor_origin_confidence") or 0.0)
        primary_actor_transfer_prior = bool(transfer_row.get("primary_actor_prior"))
        nonprimary_actor_like_due_to_transfer_prior = bool(primary_actor_prior_ids and component_id in actor_ids and component_id not in primary_actor_prior_ids)
        like_control_redundancy_score = float(like_control_row.get("like_control_redundancy_score") or 0.0) if isinstance(like_control_row, dict) else 0.0
        nonconformant_curiosity_boost = float(like_control_row.get("nonconformant_curiosity_boost") or 0.0) if isinstance(like_control_row, dict) else 0.0
        macro_curiosity_score = float(macroband_row.get("curiosity_score") or 0.0) if isinstance(macroband_row, dict) else 0.0
        macro_danger_score = float(macroband_row.get("danger_score") or 0.0) if isinstance(macroband_row, dict) else 0.0
        macro_probe_value_score = float(macroband_row.get("probe_value_score") or 0.0) if isinstance(macroband_row, dict) else 0.0
        macroband_decision = str(macroband_row.get("click_decision") or "not_ranked") if isinstance(macroband_row, dict) else "not_ranked"
        representative_group_ids = sorted(set(like_control_row.get("representative_for_group_ids", []))) if isinstance(like_control_row, dict) else []
        deferred_group_ids = sorted(set(like_control_row.get("deferred_by_group_ids", []))) if isinstance(like_control_row, dict) else []
        asymmetry_score = float(asymmetry["score"])
        polar_reverse_score = float(context.get("polar_reverse_score") or 0.0) if isinstance(context, dict) else 0.0
        sensor_confidence = float(context.get("sensor_confidence") or 0.0) if isinstance(context, dict) else 0.0
        relational_link_score = min(
            1.0,
            (0.35 if static_geoband_context_ids else 0.0)
            + (0.35 if symmetry_sensor_ids else 0.0)
            + (0.35 if linked_actor_like_ids else 0.0)
            + (0.25 * polar_reverse_score)
            + (0.15 if geoband_ids else 0.0),
        )
        shape_novelty_score = min(
            1.0,
            (0.30 if context.get("lock_key_sensor_ids") else 0.0)
            + (0.25 if context.get("polar_group_ids") or context.get("reverse_group_ids") else 0.0)
            + (0.15 if context.get("mirror_group_ids") else 0.0)
            + (0.20 * asymmetry_score)
            + (0.10 if component_id not in actor_ids else 0.0),
        )
        rank_tuple = [
            0 if macroband_decision == "blocked_do_not_probe" else 1,
            round(macro_probe_value_score, 6),
            -round(macro_danger_score, 6),
            round(macro_curiosity_score, 6),
            0 if primary_actor_transfer_prior else 1,
            round(actor_origin_prior_confidence, 6),
            min(3, len(linked_actor_like_ids)),
            round(relational_link_score, 6),
            round(shape_novelty_score, 6),
            round(nonconformant_curiosity_boost, 6),
            1 if representative_group_ids else 0,
            -round(like_control_redundancy_score, 6),
            round(polar_reverse_score, 6),
            round(asymmetry_score, 6),
            round(color_salience, 6),
            1 if static_geoband_context_ids or geoband_ids else 0,
            round(support_score, 6),
        ]
        center = component.get("center_xy") if isinstance(component.get("center_xy"), list) else [None, None]
        if linked_actor_like_ids:
            candidate_kind = "actor_link_relational_novelty_candidate"
        elif symmetry_sensor_ids:
            candidate_kind = "symmetry_sensor_relational_candidate"
        elif static_geoband_context_ids:
            candidate_kind = "geoband_static_relational_candidate"
        elif geoband_ids:
            candidate_kind = "geoband_support_state_change_candidate"
        elif component_id in support_by_component:
            candidate_kind = "support_state_change_candidate"
        else:
            candidate_kind = "asymmetry_state_change_candidate"
        if primary_actor_transfer_prior:
            candidate_kind = "primary_actor_transfer_prior_candidate"
        elif nonprimary_actor_like_due_to_transfer_prior:
            candidate_kind = f"secondary_{candidate_kind}"
        if nonconformant_curiosity_boost > 0.0 and not representative_group_ids:
            candidate_kind = f"nonconformant_like_control_{candidate_kind}"
        elif deferred_group_ids and not representative_group_ids:
            candidate_kind = f"deferred_like_control_group_member_{candidate_kind}"
        rankings.append(
            {
                "candidate_component_id": component_id,
                "candidate_kind": candidate_kind,
                "state_change_rank_basis": "macroband_probe_value_then_pre_target_relational_novelty_then_shape_then_color",
                "rank_tuple": rank_tuple,
                "macroband_context_status": MACROBAND_CONTEXT_STATUS,
                "macroband_rank": macroband_row.get("macro_rank") if isinstance(macroband_row, dict) else None,
                "macroband_click_decision": macroband_decision,
                "macroband_curiosity_score": round(macro_curiosity_score, 6),
                "macroband_danger_score": round(macro_danger_score, 6),
                "macroband_probe_value_score": round(macro_probe_value_score, 6),
                "macroband_curiosity_reasons": macroband_row.get("curiosity_reasons", []) if isinstance(macroband_row, dict) else [],
                "macroband_danger_reasons": macroband_row.get("danger_reasons", []) if isinstance(macroband_row, dict) else [],
                "macroband_signals": macroband_signals,
                "actor_origin_prior_confidence": round(actor_origin_prior_confidence, 6),
                "primary_actor_transfer_prior": primary_actor_transfer_prior,
                "nonprimary_actor_like_due_to_transfer_prior": nonprimary_actor_like_due_to_transfer_prior,
                "transfer_prior_context_status": MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS,
                "transfer_prior_role_reweights": transfer_row.get("role_reweights", []),
                "like_control_prior_context_status": LIKE_CONTROL_PRIOR_CONTEXT_STATUS,
                "like_control_group_ids": sorted(set(like_control_row.get("like_control_group_ids", []))) if isinstance(like_control_row, dict) else [],
                "like_control_representative_for_group_ids": representative_group_ids,
                "like_control_deferred_by_group_ids": deferred_group_ids,
                "like_control_discovery_role": "group_representative_probe"
                if representative_group_ids
                else "defer_redundant_group_member"
                if deferred_group_ids
                else "nonconformant_high_signal_probe"
                if nonconformant_curiosity_boost > 0.0
                else "ungrouped",
                "like_control_redundancy_score": round(like_control_redundancy_score, 6),
                "nonconformant_curiosity_boost": round(nonconformant_curiosity_boost, 6),
                "relational_link_score": round(relational_link_score, 6),
                "shape_novelty_score": round(shape_novelty_score, 6),
                "polar_reverse_score": round(polar_reverse_score, 6),
                "sensor_confidence_score": round(sensor_confidence, 6),
                "asymmetry_score": round(asymmetry_score, 6),
                "asymmetry_axis": asymmetry["axis"],
                "mirror_x_difference": asymmetry["mirror_x_difference"],
                "mirror_y_difference": asymmetry["mirror_y_difference"],
                "color_salience_score": color_salience,
                "raw_pixel_value": component.get("raw_pixel_value"),
                "palette_name": component.get("palette_name"),
                "support_score": round(support_score, 6),
                "linked_geoband_ids": geoband_ids,
                "static_geoband_context_ids": static_geoband_context_ids,
                "static_geoband_symmetry_group_ids": sorted(set(context.get("static_geoband_symmetry_group_ids", []))),
                "static_geoband_symmetry_sensor_ids": sorted(set(context.get("static_geoband_symmetry_sensor_ids", []))),
                "symmetry_group_ids": symmetry_group_ids,
                "symmetry_sensor_ids": symmetry_sensor_ids,
                "symmetry_kinds": sorted(set(context.get("symmetry_kinds", []))),
                "linked_actor_like_component_ids": linked_actor_like_ids,
                "actor_like_candidate": component_id in actor_ids,
                "geoband_context_status": GEOBAND_CONTEXT_STATUS,
                "click_xy": [round(float(center[0]), 6), round(float(center[1]), 6)]
                if len(center) == 2 and center[0] is not None and center[1] is not None
                else None,
                "rejected_actor_or_goal_or_surface": False,
                "target_truth_admitted": False,
                "actor_truth_admitted": False,
                "admission_status": "not_admitted",
                "gameplay_move_admitted": False,
                "action_choice_admitted": False,
                "control_truth_admitted": False,
            }
        )
    rankings.sort(
        key=lambda row: (
            -int(row["rank_tuple"][0]),
            -float(row["rank_tuple"][1]),
            -float(row["rank_tuple"][2]),
            -float(row["rank_tuple"][3]),
            -int(row["rank_tuple"][4]),
            -float(row["rank_tuple"][5]),
            -int(row["rank_tuple"][6]),
            -float(row["rank_tuple"][7]),
            -float(row["rank_tuple"][8]),
            -float(row["rank_tuple"][9]),
            -int(row["rank_tuple"][10]),
            -float(row["rank_tuple"][11]),
            -float(row["rank_tuple"][12]),
            -float(row["rank_tuple"][13]),
            -float(row["rank_tuple"][14]),
            -int(row["rank_tuple"][15]),
            -float(row["rank_tuple"][16]),
            str(row["candidate_component_id"]),
        )
    )
    for index, row in enumerate(rankings, start=1):
        row["rank"] = index
    selected = rankings[0] if rankings else None
    return {
        "ranking_schema": "codexlab.match_calibration.stage0_state_change_click_ranking.v1",
        "ranking_status": "ranked" if rankings else "no_ranked_relational_candidates",
        "ranking_policy": "macroband_probe_value_then_pre_target_relational_novelty_then_shape_then_color",
        "geoband_context_status": GEOBAND_CONTEXT_STATUS,
        "transfer_prior_context_status": MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS,
        "like_control_prior_context_status": LIKE_CONTROL_PRIOR_CONTEXT_STATUS,
        "macroband_context_status": MACROBAND_CONTEXT_STATUS,
        "macroband_object_ranking_status": (macroband_object_ranking or {}).get("status") if isinstance(macroband_object_ranking, dict) else "not_supplied",
        "like_control_group_prior_status": (like_control_group_prior or {}).get("status") if isinstance(like_control_group_prior, dict) else "not_supplied",
        "like_control_group_count": (like_control_group_prior or {}).get("like_control_group_count") if isinstance(like_control_group_prior, dict) else 0,
        "estimated_redundant_discovery_clicks_reduced": (like_control_group_prior or {}).get("estimated_redundant_discovery_clicks_reduced")
        if isinstance(like_control_group_prior, dict)
        else 0,
        "primary_actor_prior_component_ids": sorted(primary_actor_prior_ids),
        "target_truth_admitted": False,
        "actor_truth_admitted": False,
        "actor_like_candidate_component_ids": sorted(actor_ids),
        "excluded_target_component_ids": sorted(target_ids),
        "excluded_surface_component_ids": sorted(surface_ids),
        "candidate_count": len(rankings),
        "ranked_candidates": rankings[:12],
        "selected_candidate": selected,
        "selected_click_xy": selected.get("click_xy") if isinstance(selected, dict) else None,
    }


def classify_stage0_init_delta(
    *,
    before_raw_frame_sha256: str | None,
    after_raw_frame_sha256: str | None,
    raw_pixel_delta_count: int | None = None,
    visible_message: str | None = None,
    map_loaded: bool | None = None,
    canvas_rejected: bool | None = None,
    focus_changed: bool | None = None,
) -> dict[str, Any]:
    if canvas_rejected is True:
        status = "click_rejected"
    elif map_loaded is True:
        status = "initialized_map_loaded"
    elif focus_changed is True:
        status = "focus_only"
    elif raw_pixel_delta_count == 0 or (
        before_raw_frame_sha256
        and after_raw_frame_sha256
        and str(before_raw_frame_sha256) == str(after_raw_frame_sha256)
    ):
        status = "no_visible_delta"
    else:
        status = "unknown_delta"
    return {
        "classification_status": status,
        "before_raw_frame_sha256": before_raw_frame_sha256,
        "after_raw_frame_sha256": after_raw_frame_sha256,
        "raw_pixel_delta_count": raw_pixel_delta_count,
        "visible_message": visible_message,
        "map_loaded": map_loaded,
        "canvas_rejected": canvas_rejected,
        "focus_changed": focus_changed,
        "gameplay_move_admitted": False,
    }


def build_stage0_init_calibration_requirement(
    *,
    level_index: int,
    source_raw_frame_sha256: str,
    relationship_ladder: dict[str, Any] | None,
    click_policy: str = MATCH_CALIBRATION_STAGE0_CLICK_POLICY,
    raw_color_component_packet: dict[str, Any] | None = None,
    scene_surfaces: dict[str, Any] | None = None,
    actor_affordances: list[dict[str, Any]] | None = None,
    target_affordances: list[dict[str, Any]] | None = None,
    support_affordances: list[dict[str, Any]] | None = None,
    statecons: list[dict[str, Any]] | None = None,
    symmetry_hypotheses: dict[str, Any] | None = None,
    action_result_transfer_prior: dict[str, Any] | None = None,
    like_control_group_prior: dict[str, Any] | None = None,
    macroband_object_ranking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pre_ladder_id = ladder_id(relationship_ladder)
    click_ranking = build_stage0_state_change_click_ranking(
        raw_color_component_packet=raw_color_component_packet,
        scene_surfaces=scene_surfaces,
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        support_affordances=support_affordances,
        statecons=statecons,
        symmetry_hypotheses=symmetry_hypotheses,
        action_result_transfer_prior=action_result_transfer_prior,
        like_control_group_prior=like_control_group_prior,
        macroband_object_ranking=macroband_object_ranking,
    )
    selected_click_xy = click_ranking.get("selected_click_xy")
    return {
        "schema": MATCH_CALIBRATION_STAGE0_REQUIREMENT_SCHEMA,
        "stage_id": MATCH_CALIBRATION_STAGE0_ID,
        "status": "not_run_requires_init_canvas_click_boundary",
        "calibration_run_status": "not_run",
        "event_kind": MATCH_CALIBRATION_STAGE0_EVENT_KIND,
        "intent": "initialize_or_focus_game_canvas",
        "click_policy": click_policy,
        "click_policy_order": [
            "static_relational_link_to_actor_like_candidate",
            "action_result_transfer_prior_if_observed",
            "shape_novelty",
            "symmetry_polar_reverse_sensor_context",
            "nonconformant_like_control_curiosity",
            "one_probe_like_control_group_representative",
            "defer_redundant_like_control_group_members",
            "macroband_curiosity_danger_probe_value",
            "color_salience",
            "canvas_centroid_fallback",
        ],
        "geoband_context_status": GEOBAND_CONTEXT_STATUS,
        "transfer_prior_context_status": MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS,
        "like_control_prior_context_status": LIKE_CONTROL_PRIOR_CONTEXT_STATUS,
        "macroband_context_status": MACROBAND_CONTEXT_STATUS,
        "like_control_group_prior": like_control_group_prior or {},
        "macroband_object_ranking": macroband_object_ranking or {},
        "target_truth_admitted": False,
        "actor_truth_admitted": False,
        "click_target": {
            "coordinate_space": "runtime_canvas",
            "x_policy": "ranked_candidate_center_x_if_available_else_canvas_width_div_2",
            "y_policy": "ranked_candidate_center_y_if_available_else_canvas_height_div_2",
            "selected_xy": selected_click_xy,
            "fallback_policy": "canvas_center_if_no_ranked_candidate",
        },
        "state_change_click_ranking": click_ranking,
        "level_index": int(level_index),
        "source_raw_frame_sha256": str(source_raw_frame_sha256),
        "pre_init_ladder_id": pre_ladder_id,
        "required_before_deltaband": True,
        **missing_artifact_fields(MATCH_CALIBRATION_STAGE0_REQUIRED_ARTIFACTS),
        "init_click_invocation_id": None,
        "pre_init_frame_path": None,
        "post_init_frame_sequence_paths": [],
        "initialized_baseline_frame_path": None,
        "delta_classification": None,
        "prediction_status": "not_predictive",
        "evidence_status": "not_observed",
        **admission_flags(),
    }


def build_stage0_init_calibration_observation(
    *,
    level_index: int,
    source_raw_frame_sha256: str,
    relationship_ladder: dict[str, Any] | None,
    init_click_invocation_id: str,
    pre_init_frame_path: str,
    post_init_frame_sequence_paths: list[str],
    initialized_baseline_frame_path: str,
    before_raw_frame_sha256: str,
    after_raw_frame_sha256: str,
    raw_pixel_delta_count: int | None = None,
    visible_message: str | None = None,
    map_loaded: bool | None = None,
    canvas_rejected: bool | None = None,
    focus_changed: bool | None = None,
    click_policy: str = "canvas_center",
) -> dict[str, Any]:
    classification = classify_stage0_init_delta(
        before_raw_frame_sha256=before_raw_frame_sha256,
        after_raw_frame_sha256=after_raw_frame_sha256,
        raw_pixel_delta_count=raw_pixel_delta_count,
        visible_message=visible_message,
        map_loaded=map_loaded,
        canvas_rejected=canvas_rejected,
        focus_changed=focus_changed,
    )
    return {
        "schema": MATCH_CALIBRATION_STAGE0_OBSERVATION_SCHEMA,
        "stage_id": MATCH_CALIBRATION_STAGE0_ID,
        "calibration_id": calibration_id(
            "matchcal_stage0",
            init_click_invocation_id,
            before_raw_frame_sha256,
            after_raw_frame_sha256,
        ),
        "status": "observed_init_canvas_click_boundary",
        "calibration_run_status": "complete",
        "event_kind": MATCH_CALIBRATION_STAGE0_EVENT_KIND,
        "intent": "initialize_or_focus_game_canvas",
        "click_policy": click_policy,
        "level_index": int(level_index),
        "source_raw_frame_sha256": str(source_raw_frame_sha256),
        "pre_init_ladder_id": ladder_id(relationship_ladder),
        "required_before_deltaband": True,
        **complete_artifact_fields(MATCH_CALIBRATION_STAGE0_REQUIRED_ARTIFACTS),
        "init_click_invocation_id": str(init_click_invocation_id),
        "pre_init_frame_path": str(pre_init_frame_path),
        "post_init_frame_sequence_paths": [str(path) for path in post_init_frame_sequence_paths],
        "initialized_baseline_frame_path": str(initialized_baseline_frame_path),
        "delta_classification": classification,
        "prediction_status": "not_predictive",
        "evidence_status": "observed_calibration_boundary",
        **admission_flags(),
    }
