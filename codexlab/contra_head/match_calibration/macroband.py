#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from .common import admission_flags
from .like_controls import LIKE_CONTROL_PRIOR_CONTEXT_STATUS
from .like_controls import _deltaband_scores, _portband_scores
from .stage0 import (
    GEOBAND_CONTEXT_STATUS,
    _color_salience,
    _component_asymmetry,
    _component_rows,
    _geobands_from_statecons,
    _ids_from_rows,
    _support_component_ids_from_geobands,
    _surface_component_ids,
    _symmetry_context_by_component,
    _transfer_prior_by_component,
)
from .transfer_prior import MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS

MATCH_CALIBRATION_MACROBAND_OBJECT_RANKING_SCHEMA = "codexlab.match_calibration.macroband_object_ranking.v1"
MATCH_CALIBRATION_MACROBAND_OBJECT_ROW_SCHEMA = "codexlab.match_calibration.macroband_object_row.v1"
MACROBAND_CONTEXT_STATUS = "static_multisensor_object_ranking_not_control_truth"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _component_id_from_row(row: dict[str, Any]) -> str | None:
    for key in ("source_component_id", "component_id", "actor_component_id", "orientation_key_component_id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def _support_by_component(support_affordances: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in support_affordances or []:
        if not isinstance(row, dict):
            continue
        component_id = _component_id_from_row(row)
        if component_id:
            rows[component_id] = row
    return rows


def _like_control_context_by_component(like_control_group_prior: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(like_control_group_prior, dict):
        return {}
    return {
        str(row.get("component_id")): row
        for row in like_control_group_prior.get("component_like_control_context", []) or []
        if isinstance(row, dict) and row.get("component_id") is not None
    }


def _group_member_priority_by_component(like_control_group_prior: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not isinstance(like_control_group_prior, dict):
        return rows
    for group in like_control_group_prior.get("like_control_groups", []) or []:
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("like_control_group_id") or "")
        for priority_row in group.get("member_priority_rows", []) or []:
            if not isinstance(priority_row, dict) or priority_row.get("component_id") is None:
                continue
            component_id = str(priority_row["component_id"])
            existing = rows.get(component_id)
            if existing is None or float(priority_row.get("priority_score") or 0.0) > float(existing.get("priority_score") or 0.0):
                merged = dict(priority_row)
                merged["like_control_group_id"] = group_id
                rows[component_id] = merged
    return rows


def _portband_signal_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_alignment_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_lineup_matrix_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_lineup_matrix_row_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_fleet_candidate_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_fleet_candidate_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_state_tag_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_state_tag_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_state_primitives(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_state_primitives")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_state_event_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_state_evidence_event_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_state_action_invocation_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_state_action_invocation_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_state_correlated_object_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_state_correlated_object_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_relative_state_change_tag_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_relative_state_change_tag_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_relative_state_change_novelty_statuses(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_relative_state_change_novelty_statuses")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_solve_like_evidence_flags(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_solve_like_evidence_flags")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_interaction_edge_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_interaction_edge_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_interaction_kinds(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_interaction_kinds")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_action_interaction_participant_object_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_action_interaction_participant_object_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_route_flow_lock_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_route_flow_lock_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_route_flow_event_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_route_flow_evidence_event_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _portband_route_flow_action_invocation_ids(portband: dict[str, Any]) -> list[str]:
    values = portband.get("portband_route_flow_action_invocation_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _deltaband_signal_ids(deltaband: dict[str, Any]) -> list[str]:
    values = deltaband.get("deltaband_source_ids")
    return [str(value) for value in values or [] if value is not None] if isinstance(values, list) else []


def _danger_level(danger_score: float) -> str:
    if danger_score >= 0.85:
        return "blocked"
    if danger_score >= 0.55:
        return "high"
    if danger_score >= 0.28:
        return "moderate"
    return "low"


def _decision(*, danger_score: float, probe_value_score: float, component_id: str, excluded_ids: set[str]) -> str:
    if component_id in excluded_ids or danger_score >= 0.85:
        return "blocked_do_not_probe"
    if danger_score >= 0.55:
        return "review_before_probe"
    if probe_value_score >= 0.45:
        return "probe_candidate"
    if probe_value_score >= 0.20:
        return "low_priority_probe_candidate"
    return "observe_only"


def _reason_append(reasons: list[str], condition: bool, reason: str) -> None:
    if condition and reason not in reasons:
        reasons.append(reason)


def build_macroband_object_ranking(
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
    deltaband_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    components = _component_rows(raw_color_component_packet)
    geobands = _geobands_from_statecons(statecons)
    support_geobands = _support_component_ids_from_geobands(geobands)
    support_by_component = _support_by_component(support_affordances)
    actor_ids = _ids_from_rows(actor_affordances, "actor_component_id", "source_component_id", "group_component_ids")
    target_ids = _ids_from_rows(target_affordances, "component_id", "source_component_id")
    surface_ids = _surface_component_ids(scene_surfaces)
    excluded_ids = target_ids | surface_ids
    primary_actor_prior_ids = {
        str(value)
        for value in (action_result_transfer_prior or {}).get("primary_actor_prior_component_ids", [])
        if value is not None
    } if isinstance(action_result_transfer_prior, dict) else set()

    symmetry_context = _symmetry_context_by_component(symmetry_hypotheses, actor_like_ids=actor_ids, geobands=geobands)
    transfer_by_component = _transfer_prior_by_component(action_result_transfer_prior)
    like_context = _like_control_context_by_component(like_control_group_prior)
    like_priority = _group_member_priority_by_component(like_control_group_prior)
    portband_by_component = _portband_scores(statecons)
    deltaband_by_component = _deltaband_scores(deltaband_observations)
    asymmetry_by_component = {component_id: _component_asymmetry(component) for component_id, component in components.items()}

    candidate_ids = sorted(
        set(components)
        | set(support_by_component)
        | set(support_geobands)
        | set(symmetry_context)
        | set(transfer_by_component)
        | set(like_context)
        | set(portband_by_component)
        | set(deltaband_by_component)
    )
    rows: list[dict[str, Any]] = []
    for component_id in candidate_ids:
        component = components.get(component_id)
        if not isinstance(component, dict):
            continue
        area = int(component.get("area_pixels") or 0)
        total = int(component.get("frame_color_pixel_total") or 0)
        area_ratio = float(area) / float(total) if total > 0 else 0.0
        context = symmetry_context.get(component_id, {})
        support = support_by_component.get(component_id, {})
        transfer = transfer_by_component.get(component_id, {})
        like_row = like_context.get(component_id, {})
        like_priority_row = like_priority.get(component_id, {})
        portband = portband_by_component.get(component_id, {})
        deltaband = deltaband_by_component.get(component_id, {})
        asymmetry = asymmetry_by_component.get(component_id, {"score": 0.0})

        actor_like = component_id in actor_ids
        target_like = component_id in target_ids
        surface_like = component_id in surface_ids
        support_like = component_id in support_by_component or component_id in support_geobands
        primary_actor_prior = bool(transfer.get("primary_actor_prior")) or component_id in primary_actor_prior_ids
        deferred = bool(like_row.get("deferred_by_group_ids"))
        representative = bool(like_row.get("representative_for_group_ids"))
        nonconformant = float(like_row.get("nonconformant_curiosity_boost") or 0.0) > 0.0

        geoband_score = 1.0 if support_geobands.get(component_id) else 0.0
        support_score = _clamp01(float(support.get("score") or 0.0)) if isinstance(support, dict) else 0.0
        portband_score = _clamp01(float(portband.get("portband_relevance_score") or 0.0))
        portband_lineup_confidence_score = _clamp01(float(portband.get("portband_best_lineup_confidence_score") or 0.0))
        portband_action_state_correlation_score = _clamp01(float(portband.get("portband_action_state_correlation_score") or 0.0))
        portband_action_state_positive_signal_count = int(portband.get("portband_action_state_positive_signal_count") or 0)
        portband_solve_like_evidence_score = _clamp01(float(portband.get("portband_solve_like_evidence_score") or 0.0))
        portband_route_flow_confidence_score = _clamp01(float(portband.get("portband_route_flow_confidence_score") or 0.0))
        portband_route_flow_positive_signal_count = int(portband.get("portband_route_flow_positive_signal_count") or 0)
        portband_route_flow_lock_status = str(portband.get("portband_route_flow_lock_status") or "no_locked_route_flow")
        deltaband_score = _clamp01(float(deltaband.get("deltaband_relevance_score") or 0.0))
        symmetry_score = _clamp01(float(context.get("max_symmetry_score") or 0.0)) if isinstance(context, dict) else 0.0
        polar_reverse_score = _clamp01(float(context.get("polar_reverse_score") or 0.0)) if isinstance(context, dict) else 0.0
        lock_key_score = _clamp01(float(context.get("sensor_confidence") or 0.0)) if isinstance(context, dict) else 0.0
        actor_origin_prior_score = _clamp01(float(transfer.get("max_actor_origin_confidence") or 0.0))
        like_control_priority_score = _clamp01(float(like_priority_row.get("priority_score") or 0.0))
        shape_novelty_score = _clamp01(
            (0.35 * float(asymmetry.get("score") or 0.0))
            + (0.25 if polar_reverse_score > 0.0 else 0.0)
            + (0.20 if lock_key_score > 0.0 else 0.0)
            + (0.10 if not actor_like else 0.0)
        )
        color_salience_score = _color_salience(component)
        redundancy_score = _clamp01(float(like_row.get("like_control_redundancy_score") or 0.0))
        nonconformant_score = _clamp01(float(like_row.get("nonconformant_curiosity_boost") or 0.0))

        curiosity_score = _clamp01(
            (0.24 * portband_score)
            + (0.10 * portband_solve_like_evidence_score)
            + (0.20 * deltaband_score)
            + (0.16 * geoband_score)
            + (0.14 * shape_novelty_score)
            + (0.10 * like_control_priority_score)
            + (0.08 * support_score)
            + (0.05 * color_salience_score)
            + (0.08 * nonconformant_score)
            + (0.04 * polar_reverse_score)
            + (0.04 * lock_key_score)
        )
        danger_score = _clamp01(
            (0.90 if target_like else 0.0)
            + (0.85 if surface_like else 0.0)
            + (0.66 if primary_actor_prior else 0.0)
            + (0.42 if actor_like and not primary_actor_prior else 0.0)
            + (0.16 * redundancy_score)
            + (0.18 if area_ratio > 0.35 else 0.0)
        )
        probe_value_score = _clamp01(curiosity_score - (0.62 * danger_score))

        curiosity_reasons: list[str] = []
        danger_reasons: list[str] = []
        _reason_append(curiosity_reasons, portband_score > 0.0, "portband_alignment_signal")
        _reason_append(curiosity_reasons, portband_action_state_correlation_score > 0.0, "portband_action_state_correlation_signal")
        _reason_append(curiosity_reasons, portband_solve_like_evidence_score > 0.0, "portband_solve_like_interaction_clue")
        _reason_append(curiosity_reasons, portband_route_flow_confidence_score > 0.0, "portband_route_flow_positive_signal")
        _reason_append(curiosity_reasons, deltaband_score > 0.0, "deltaband_observed_relevance")
        _reason_append(curiosity_reasons, geoband_score > 0.0, "geoband_support_context")
        _reason_append(curiosity_reasons, shape_novelty_score > 0.0, "shape_or_symmetry_novelty")
        _reason_append(curiosity_reasons, representative, "like_control_group_representative")
        _reason_append(curiosity_reasons, nonconformant, "nonconformant_like_control_signal")
        _reason_append(curiosity_reasons, support_score > 0.0, "support_affordance_signal")
        _reason_append(curiosity_reasons, color_salience_score > 0.0, "rare_color_salience")
        _reason_append(danger_reasons, target_like, "target_component_do_not_probe")
        _reason_append(danger_reasons, surface_like, "surface_component_do_not_probe")
        _reason_append(danger_reasons, primary_actor_prior, "primary_actor_prior_high_gameplay_risk")
        _reason_append(danger_reasons, actor_like and not primary_actor_prior, "actor_like_component_gameplay_risk")
        _reason_append(danger_reasons, deferred, "redundant_like_control_member")
        _reason_append(danger_reasons, area_ratio > 0.35, "large_frame_component")

        center = component.get("center_xy") if isinstance(component.get("center_xy"), list) else None
        decision = _decision(
            danger_score=danger_score,
            probe_value_score=probe_value_score,
            component_id=component_id,
            excluded_ids=excluded_ids,
        )
        row = {
            "schema": MATCH_CALIBRATION_MACROBAND_OBJECT_ROW_SCHEMA,
            "component_id": component_id,
            "component_center_xy": [round(float(center[0]), 6), round(float(center[1]), 6)] if isinstance(center, list) and len(center) == 2 else None,
            "macro_score": round(probe_value_score, 6),
            "curiosity_score": round(curiosity_score, 6),
            "danger_score": round(danger_score, 6),
            "danger_level": _danger_level(danger_score),
            "probe_value_score": round(probe_value_score, 6),
            "click_decision": decision,
            "object_roles": {
                "actor_like": actor_like,
                "target_like": target_like,
                "surface_like": surface_like,
                "support_like": support_like,
                "primary_actor_transfer_prior": primary_actor_prior,
                "like_control_group_member": bool(like_row.get("like_control_group_ids")),
                "like_control_group_representative": representative,
                "like_control_deferred_member": deferred,
                "nonconformant_like_control": nonconformant,
            },
            "signals": {
                "geoband_score": round(geoband_score, 6),
                "support_score": round(support_score, 6),
                "portband_score": round(portband_score, 6),
                "deltaband_score": round(deltaband_score, 6),
                "symmetry_score": round(symmetry_score, 6),
                "polar_reverse_score": round(polar_reverse_score, 6),
                "lock_key_score": round(lock_key_score, 6),
                "shape_novelty_score": round(shape_novelty_score, 6),
                "asymmetry_score": round(float(asymmetry.get("score") or 0.0), 6),
                "color_salience_score": round(color_salience_score, 6),
                "actor_origin_prior_score": round(actor_origin_prior_score, 6),
                "like_control_priority_score": round(like_control_priority_score, 6),
                "like_control_redundancy_score": round(redundancy_score, 6),
                "nonconformant_curiosity_boost": round(nonconformant_score, 6),
                "portband_lineup_confidence_score": round(portband_lineup_confidence_score, 6),
                "portband_action_state_correlation_score": round(portband_action_state_correlation_score, 6),
                "portband_action_state_positive_signal_count": portband_action_state_positive_signal_count,
                "portband_solve_like_evidence_score": round(portband_solve_like_evidence_score, 6),
                "portband_route_flow_confidence_score": round(portband_route_flow_confidence_score, 6),
                "portband_route_flow_positive_signal_count": portband_route_flow_positive_signal_count,
            },
            "signal_refs": {
                "linked_geoband_ids": sorted(set(support_geobands.get(component_id, []))),
                "static_geoband_context_ids": sorted(set(context.get("static_geoband_context_ids", []))) if isinstance(context, dict) else [],
                "portband_alignment_ids": _portband_signal_ids(portband),
                "portband_lineup_matrix_row_ids": _portband_lineup_matrix_ids(portband),
                "portband_fleet_candidate_ids": _portband_fleet_candidate_ids(portband),
                "portband_best_lineup_route_label": portband.get("portband_best_lineup_route_label"),
                "portband_action_state_tag_ids": _portband_action_state_tag_ids(portband),
                "portband_action_state_primitives": _portband_action_state_primitives(portband),
                "portband_action_state_evidence_event_ids": _portband_action_state_event_ids(portband),
                "portband_action_state_action_invocation_ids": _portband_action_state_action_invocation_ids(portband),
                "portband_action_state_correlated_object_ids": _portband_action_state_correlated_object_ids(portband),
                "portband_relative_state_change_tag_ids": _portband_relative_state_change_tag_ids(portband),
                "portband_relative_state_change_novelty_statuses": _portband_relative_state_change_novelty_statuses(portband),
                "portband_solve_like_evidence_flags": _portband_solve_like_evidence_flags(portband),
                "portband_action_interaction_edge_ids": _portband_action_interaction_edge_ids(portband),
                "portband_action_interaction_kinds": _portband_action_interaction_kinds(portband),
                "portband_action_interaction_participant_object_ids": _portband_action_interaction_participant_object_ids(portband),
                "portband_route_flow_lock_ids": _portband_route_flow_lock_ids(portband),
                "portband_route_flow_evidence_event_ids": _portband_route_flow_event_ids(portband),
                "portband_route_flow_action_invocation_ids": _portband_route_flow_action_invocation_ids(portband),
                "portband_route_flow_lock_status": portband_route_flow_lock_status,
                "deltaband_source_ids": _deltaband_signal_ids(deltaband),
                "symmetry_group_ids": sorted(set(context.get("symmetry_group_ids", []))) if isinstance(context, dict) else [],
                "symmetry_sensor_ids": sorted(set(context.get("symmetry_sensor_ids", []))) if isinstance(context, dict) else [],
                "like_control_group_ids": sorted(set(like_row.get("like_control_group_ids", []))) if isinstance(like_row, dict) else [],
            },
            "curiosity_reasons": curiosity_reasons,
            "danger_reasons": danger_reasons,
            "ranking_basis": "curiosity_minus_danger_from_static_contra_sensor_signals",
            "context_status": MACROBAND_CONTEXT_STATUS,
            "geoband_context_status": GEOBAND_CONTEXT_STATUS,
            "transfer_prior_context_status": MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS,
            "like_control_prior_context_status": LIKE_CONTROL_PRIOR_CONTEXT_STATUS,
            "evidence_status": "static_multisensor_hypothesis",
            "prediction_status": "not_predictive",
            "target_truth_admitted": False,
            "actor_truth_admitted": False,
            **admission_flags(),
        }
        rows.append(row)

    rows.sort(
        key=lambda row: (
            -float(row["probe_value_score"]),
            float(row["danger_score"]),
            -float(row["curiosity_score"]),
            str(row["component_id"]),
        )
    )
    for index, row in enumerate(rows, start=1):
        row["macro_rank"] = index

    return {
        "schema": MATCH_CALIBRATION_MACROBAND_OBJECT_RANKING_SCHEMA,
        "status": "ranked" if rows else "no_ranked_objects",
        "context_status": MACROBAND_CONTEXT_STATUS,
        "ranking_policy": "rank_objects_by_probe_value_curiosity_minus_danger",
        "curiosity_policy": "reward_portband_route_flow_deltaband_geoband_symmetry_like_control_shape_color_support_signals",
        "danger_policy": "penalize_or_block_targets_surfaces_primary_actor_actor_like_large_redundant_group_members",
        "object_count": len(rows),
        "ranked_object_count": len(rows),
        "top_probe_candidate_component_id": rows[0]["component_id"] if rows else None,
        "top_probe_candidate_click_decision": rows[0]["click_decision"] if rows else None,
        "blocked_component_ids": [str(row["component_id"]) for row in rows if row.get("click_decision") == "blocked_do_not_probe"],
        "ranked_objects": rows,
        "ranked_objects_preview": rows[:16],
        "target_truth_admitted": False,
        "actor_truth_admitted": False,
        "evidence_status": "static_multisensor_hypothesis",
        "prediction_status": "not_predictive",
        **admission_flags(),
    }
