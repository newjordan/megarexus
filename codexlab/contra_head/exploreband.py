#!/usr/bin/env python3
"""ExploreBand: bounded probe budget for Oracle/CONTRA deployment loops."""

from __future__ import annotations

import hashlib
from typing import Any

from .match_calibration.common import admission_flags


EXPLOREBAND_SCHEMA = "codexlab.exploreband.budget.v1"
EXPLOREBAND_PROBE_SCHEMA = "codexlab.exploreband.probe_candidate.v1"
EXPLOREBAND_LEDGER_ROW_SCHEMA = "codexlab.exploreband.spent_probe_ledger_row.v1"
EXPLOREBAND_CONTEXT_STATUS = "budgeted_exploration_candidates_not_action_authority"

DEFAULT_ALLOWED_ACTION_PRIMITIVES = [
    "init_canvas_click",
    "focus_or_select_probe",
    "single_step_direction_probe",
    "single_action_evaluator_probe",
]
DEFAULT_BLOCKED_ACTION_PRIMITIVES = [
    "unbudgeted_portband_action",
    "direct_solve_attempt",
    "multi_step_unrecorded_rollout",
    "irreversible_action_without_llm_budget_approval",
]
DEFAULT_STOP_CONDITIONS = [
    "remaining_probe_count_is_zero",
    "candidate_queue_empty",
    "risk_score_exceeds_budget_limit",
    "executor_cannot_record_controlled_before_after_boundary",
    "oracle_declines_all_candidates",
]


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _clamp01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))


def _config_int(config: dict[str, Any], key: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    return _clamp01(config.get(key, default))


def _as_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    rows = [str(row) for row in value if row is not None]
    return rows if rows else list(default)


def _spent_rows(previous_ledger: Any) -> list[dict[str, Any]]:
    if isinstance(previous_ledger, dict):
        rows = previous_ledger.get("spent_probe_ledger") or previous_ledger.get("ledger_rows") or []
    else:
        rows = previous_ledger
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        normalized.append(
            {
                "schema": EXPLOREBAND_LEDGER_ROW_SCHEMA,
                "ledger_index": int(row.get("ledger_index") or index),
                "probe_id": row.get("probe_id"),
                "component_id": row.get("component_id") or row.get("candidate_component_id"),
                "action_invocation_id": row.get("action_invocation_id"),
                "oracle_selection_ref": row.get("oracle_selection_ref"),
                "execution_status": row.get("execution_status") or row.get("status") or "observed_prior_spend",
                "result_ref": row.get("result_ref"),
                "budget_debit": _config_int(row, "budget_debit", 1, minimum=0),
            }
        )
    return normalized


def _stage0_rows(stage0_requirement: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not isinstance(stage0_requirement, dict):
        return {}
    ranking = stage0_requirement.get("state_change_click_ranking")
    if not isinstance(ranking, dict):
        ranking = stage0_requirement.get("click_ranking")
    if not isinstance(ranking, dict):
        return {}
    rows = ranking.get("ranked_candidates")
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("candidate_component_id")): row
        for row in rows
        if isinstance(row, dict) and row.get("candidate_component_id") is not None
    }


def _source_evidence_refs(macro_row: dict[str, Any], stage0_row: dict[str, Any] | None) -> dict[str, Any]:
    refs = macro_row.get("signal_refs") if isinstance(macro_row.get("signal_refs"), dict) else {}
    signals = macro_row.get("signals") if isinstance(macro_row.get("signals"), dict) else {}
    return {
        "macroband_rank": macro_row.get("macro_rank"),
        "macroband_decision": macro_row.get("click_decision"),
        "stage0_rank": stage0_row.get("rank") if isinstance(stage0_row, dict) else None,
        "stage0_candidate_kind": stage0_row.get("candidate_kind") if isinstance(stage0_row, dict) else None,
        "portband_alignment_ids": refs.get("portband_alignment_ids", []),
        "portband_lineup_matrix_row_ids": refs.get("portband_lineup_matrix_row_ids", []),
        "portband_fleet_candidate_ids": refs.get("portband_fleet_candidate_ids", []),
        "portband_best_lineup_route_label": refs.get("portband_best_lineup_route_label"),
        "portband_action_state_tag_ids": refs.get("portband_action_state_tag_ids", []),
        "portband_action_state_primitives": refs.get("portband_action_state_primitives", []),
        "portband_action_state_evidence_event_ids": refs.get("portband_action_state_evidence_event_ids", []),
        "portband_action_state_action_invocation_ids": refs.get("portband_action_state_action_invocation_ids", []),
        "portband_action_state_correlated_object_ids": refs.get("portband_action_state_correlated_object_ids", []),
        "portband_relative_state_change_tag_ids": refs.get("portband_relative_state_change_tag_ids", []),
        "portband_relative_state_change_novelty_statuses": refs.get("portband_relative_state_change_novelty_statuses", []),
        "portband_solve_like_evidence_flags": refs.get("portband_solve_like_evidence_flags", []),
        "portband_action_interaction_edge_ids": refs.get("portband_action_interaction_edge_ids", []),
        "portband_action_interaction_kinds": refs.get("portband_action_interaction_kinds", []),
        "portband_action_interaction_participant_object_ids": refs.get("portband_action_interaction_participant_object_ids", []),
        "portband_route_flow_lock_ids": refs.get("portband_route_flow_lock_ids", []),
        "portband_route_flow_evidence_event_ids": refs.get("portband_route_flow_evidence_event_ids", []),
        "portband_route_flow_action_invocation_ids": refs.get("portband_route_flow_action_invocation_ids", []),
        "portband_route_flow_lock_status": refs.get("portband_route_flow_lock_status"),
        "deltaband_source_ids": refs.get("deltaband_source_ids", []),
        "linked_geoband_ids": refs.get("linked_geoband_ids", []),
        "symmetry_group_ids": refs.get("symmetry_group_ids", []),
        "symmetry_sensor_ids": refs.get("symmetry_sensor_ids", []),
        "like_control_group_ids": refs.get("like_control_group_ids", []),
        "signal_scores": {
            "portband_score": signals.get("portband_score"),
            "deltaband_score": signals.get("deltaband_score"),
            "geoband_score": signals.get("geoband_score"),
            "symmetry_score": signals.get("symmetry_score"),
            "shape_novelty_score": signals.get("shape_novelty_score"),
            "nonconformant_curiosity_boost": signals.get("nonconformant_curiosity_boost"),
            "portband_lineup_confidence_score": signals.get("portband_lineup_confidence_score"),
            "portband_action_state_correlation_score": signals.get("portband_action_state_correlation_score"),
            "portband_action_state_positive_signal_count": signals.get("portband_action_state_positive_signal_count"),
            "portband_solve_like_evidence_score": signals.get("portband_solve_like_evidence_score"),
            "portband_route_flow_confidence_score": signals.get("portband_route_flow_confidence_score"),
            "portband_route_flow_positive_signal_count": signals.get("portband_route_flow_positive_signal_count"),
        },
    }


def _probe_kind(macro_row: dict[str, Any], stage0_row: dict[str, Any] | None) -> str:
    if isinstance(stage0_row, dict) and stage0_row.get("candidate_kind"):
        return str(stage0_row["candidate_kind"])
    reasons = set(str(reason) for reason in macro_row.get("curiosity_reasons", []) if reason is not None)
    if "portband_solve_like_interaction_clue" in reasons:
        return "portband_solve_like_clue_probe_candidate"
    if "portband_action_state_correlation_signal" in reasons:
        return "portband_action_state_informed_object_probe_candidate"
    if "portband_route_flow_positive_signal" in reasons:
        return "portband_route_flow_informed_object_probe_candidate"
    if "portband_alignment_signal" in reasons:
        return "portband_informed_object_probe_candidate"
    if "deltaband_observed_relevance" in reasons:
        return "deltaband_informed_object_probe_candidate"
    if "like_control_group_representative" in reasons:
        return "like_control_group_representative_probe_candidate"
    return "macroband_ranked_object_probe_candidate"


def _candidate_probe(
    *,
    macro_row: dict[str, Any],
    stage0_row: dict[str, Any] | None,
    level_index: int,
    source_raw_frame_sha256: str,
    risk_limit: float,
) -> dict[str, Any]:
    component_id = str(macro_row.get("component_id"))
    curiosity_score = _clamp01(macro_row.get("curiosity_score"))
    danger_score = _clamp01(macro_row.get("danger_score"))
    probe_value_score = _clamp01(macro_row.get("probe_value_score"))
    stage0_click = stage0_row.get("click_xy") if isinstance(stage0_row, dict) else None
    macro_click = macro_row.get("component_center_xy")
    click_xy = stage0_click if isinstance(stage0_click, list) else macro_click if isinstance(macro_click, list) else None
    expected_information_gain = _clamp01((0.62 * probe_value_score) + (0.24 * curiosity_score) + (0.14 * (1.0 - danger_score)))
    llm_review_required = danger_score > risk_limit or str(macro_row.get("click_decision")) == "review_before_probe"
    probe_id = _stable_id("explore_probe", level_index, source_raw_frame_sha256, component_id, macro_row.get("macro_rank"), click_xy)
    return {
        "schema": EXPLOREBAND_PROBE_SCHEMA,
        "probe_id": probe_id,
        "level_index": int(level_index),
        "component_id": component_id,
        "probe_kind": _probe_kind(macro_row, stage0_row),
        "proposed_observation_target": {
            "target_kind": "component",
            "component_id": component_id,
            "click_xy": click_xy,
            "selection_basis": "macroband_stage0_ranked_probe_candidate",
        },
        "allowed_probe_primitives": ["focus_or_select_probe", "single_step_direction_probe"],
        "blocked_probe_primitives": list(DEFAULT_BLOCKED_ACTION_PRIMITIVES),
        "risk_score": round(danger_score, 6),
        "curiosity_score": round(curiosity_score, 6),
        "probe_value_score": round(probe_value_score, 6),
        "expected_information_gain": round(expected_information_gain, 6),
        "llm_review_required": bool(llm_review_required),
        "oracle_selection_status": "not_selected",
        "executor_authorization_status": "not_authorized_requires_oracle_budget_selection",
        "execution_status": "not_run",
        "portband_action_authority": "observe_only_never_direct_action_authority",
        "probe_authority_policy": "candidate_may_be_selected_only_by_budget_controller_then_validated_by_executor",
        "required_execution_boundary": "controlled_before_after_action_boundary",
        "source_evidence_refs": _source_evidence_refs(macro_row, stage0_row),
        "curiosity_reasons": macro_row.get("curiosity_reasons", []),
        "danger_reasons": macro_row.get("danger_reasons", []),
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_exploreband_budget(
    *,
    level_index: int,
    source_raw_frame_sha256: str,
    macroband_object_ranking: dict[str, Any] | None = None,
    stage0_init_calibration_requirement: dict[str, Any] | None = None,
    previous_ledger: Any = None,
    budget_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = budget_config if isinstance(budget_config, dict) else {}
    max_probe_count = _config_int(config, "max_probe_count", 3, minimum=0)
    max_irreversible_actions = _config_int(config, "max_irreversible_actions", 0, minimum=0)
    queue_limit = _config_int(config, "candidate_queue_limit", 12, minimum=1)
    risk_limit = _config_float(config, "risk_limit", 0.55)
    allowed_action_primitives = _as_list(config.get("allowed_action_primitives"), DEFAULT_ALLOWED_ACTION_PRIMITIVES)
    blocked_action_primitives = _as_list(config.get("blocked_action_primitives"), DEFAULT_BLOCKED_ACTION_PRIMITIVES)
    stop_conditions = _as_list(config.get("stop_conditions"), DEFAULT_STOP_CONDITIONS)
    ledger_rows = _spent_rows(previous_ledger)
    spent_probe_count = sum(int(row.get("budget_debit") or 0) for row in ledger_rows)
    remaining_probe_count = max(0, max_probe_count - spent_probe_count)
    stage0_by_component = _stage0_rows(stage0_init_calibration_requirement)

    macro_rows = []
    if isinstance(macroband_object_ranking, dict) and isinstance(macroband_object_ranking.get("ranked_objects"), list):
        macro_rows = [row for row in macroband_object_ranking["ranked_objects"] if isinstance(row, dict)]
    candidate_rows = []
    for row in macro_rows:
        decision = str(row.get("click_decision") or "")
        if decision == "blocked_do_not_probe" or _clamp01(row.get("danger_score")) >= 0.85:
            continue
        if decision not in {"probe_candidate", "low_priority_probe_candidate", "review_before_probe", "observe_only"}:
            continue
        if decision == "observe_only" and _clamp01(row.get("probe_value_score")) < 0.20:
            continue
        component_id = str(row.get("component_id"))
        candidate_rows.append(
            _candidate_probe(
                macro_row=row,
                stage0_row=stage0_by_component.get(component_id),
                level_index=level_index,
                source_raw_frame_sha256=source_raw_frame_sha256,
                risk_limit=risk_limit,
            )
        )
    candidate_rows.sort(
        key=lambda row: (
            -float(row.get("expected_information_gain") or 0.0),
            float(row.get("risk_score") or 0.0),
            -float(row.get("probe_value_score") or 0.0),
            str(row.get("probe_id")),
        )
    )
    retained = candidate_rows[:queue_limit]
    for rank, row in enumerate(retained, start=1):
        row["budget_rank"] = rank
        row["within_current_budget"] = rank <= remaining_probe_count

    status = "ready_for_oracle_selection" if retained and remaining_probe_count > 0 else "budget_exhausted" if retained else "no_probe_candidates"
    return {
        "schema": EXPLOREBAND_SCHEMA,
        "status": status,
        "context_status": EXPLOREBAND_CONTEXT_STATUS,
        "level_index": int(level_index),
        "source_raw_frame_sha256": str(source_raw_frame_sha256),
        "exploreband_budget_id": _stable_id("exploreband", level_index, source_raw_frame_sha256, max_probe_count, spent_probe_count),
        "deployment_loop": "contra_observation_to_oracle_budget_selection_to_controlled_action_to_deltaband_feedback",
        "budget_controller": {
            "controller_kind": "llm_oracle_or_policy",
            "selection_status": "not_selected",
            "controller_required": True,
            "controller_may_rank": True,
            "controller_may_authorize_without_budget": False,
            "oracle_selection_ref": None,
        },
        "budget_policy": {
            "max_probe_count": max_probe_count,
            "spent_probe_count": spent_probe_count,
            "remaining_probe_count": remaining_probe_count,
            "max_irreversible_actions": max_irreversible_actions,
            "risk_limit": round(risk_limit, 6),
            "candidate_queue_limit": queue_limit,
            "allowed_action_primitives": allowed_action_primitives,
            "blocked_action_primitives": blocked_action_primitives,
            "stop_conditions": stop_conditions,
        },
        "candidate_probe_count": len(retained),
        "candidate_probe_queue": retained,
        "selected_probe": None,
        "spent_probe_ledger": ledger_rows,
        "oracle_feed_packet": {
            "schema": "codexlab.exploreband.oracle_feed.v1",
            "status": "ready" if retained and remaining_probe_count > 0 else "not_ready",
            "candidate_probe_ids": [str(row.get("probe_id")) for row in retained if row.get("within_current_budget")],
            "selection_instruction": "select_at_most_one_probe_id_within_budget_then_executor_records_controlled_boundary",
            "portband_boundary": "PortBand is observation-only; it can explain curiosity but cannot authorize action.",
        },
        "portband_action_authority": "observe_only_never_direct_action_authority",
        "execution_authority": "none_until_oracle_budget_selection_and_executor_validation",
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        **admission_flags(),
    }
