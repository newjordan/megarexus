#!/usr/bin/env python3
"""Export controlled paint-control skill feedback rows from SP80 runs."""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sp80_base_controller import ROOT, sha256_file


SCHEMA = "codexlab.paint_control_skill_feedback_export.v1"
ROW_SCHEMA = "codexlab.paint_control_skill_feedback_row.v1"
SOURCE_EDGE_SUMMARY_SCHEMA = "codexlab.paint_control_skill_source_edge_summary.v1"
SELECTION_OUTCOME_SUMMARY_SCHEMA = "codexlab.paint_control_skill_selection_outcome_summary.v1"
FALLBACK_EXHAUSTION_SUMMARY_SCHEMA = "codexlab.paint_control_skill_fallback_exhaustion_summary.v1"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def image_path(frame: dict[str, Any] | None) -> str | None:
    if not isinstance(frame, dict):
        return None
    png = frame.get("png")
    if isinstance(png, dict) and isinstance(png.get("path"), str):
        return png["path"]
    return None


def html_asset_src(out_dir: Path, repo_relative_path: Any) -> str:
    if not isinstance(repo_relative_path, str) or not repo_relative_path:
        return ""
    source = ROOT / repo_relative_path
    try:
        return os.path.relpath(source.resolve(), out_dir.resolve())
    except OSError:
        return repo_relative_path


def distance_delta(packet: dict[str, Any]) -> float | None:
    progress = packet.get("goal_progress") if isinstance(packet.get("goal_progress"), dict) else {}
    value = progress.get("distance_delta")
    return float(value) if isinstance(value, (int, float)) else None


def classify(packet: dict[str, Any], brain_raw: dict[str, Any]) -> str:
    if packet.get("status") == "brain_error":
        reason = packet.get("error") or brain_raw.get("fail_closed_reason")
        if reason == "repeated_stationary_move_feedback":
            return "fail_closed_repeated_stationary_move"
        if reason == "fallback_stationary_move_feedback":
            return "fail_closed_fallback_stationary_move"
        if reason == "fallback_negative_distance_feedback":
            return "fail_closed_fallback_negative_distance"
        if reason == "fallback_repeated_movement_oscillation":
            return "fail_closed_fallback_movement_oscillation"
        if reason == "activation_readiness_unknown":
            return "fail_closed_activation_readiness_unknown"
        if reason == "activation_readiness_target_misaligned":
            return "fail_closed_activation_readiness_target_misaligned"
        if reason == "activation_prior_no_progress_feedback":
            return "fail_closed_activation_prior_no_progress"
        if reason == "forward_model_predicted_blocked_or_away":
            return "fail_closed_forward_model_predicted_blocked_or_away"
        if reason == "selection_budget_infeasible":
            return "fail_closed_selection_budget_infeasible"
        if reason == "ranked_fallback_exhausted":
            return "fail_closed_ranked_fallback_exhausted"
        return "fail_closed_other"
    if int(packet.get("levels_completed_delta") or 0) > 0:
        return "solve_positive"
    delta = distance_delta(packet)
    if isinstance(packet.get("decision_primitive"), str) and str(packet["decision_primitive"]).startswith("move_"):
        if packet.get("selected_display_coord_delta") == [0, 0] and (delta is None or delta <= 0.0):
            return "hard_negative_stationary_move"
    if delta is not None and delta > 0.0:
        return "positive_goal_distance_reduced"
    if delta is not None and delta < 0.0:
        return "negative_goal_distance_increased"
    return "neutral_no_goal_progress"


def nested_dict(value: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def activation_feedback_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    live = raw.get("live_feedback") if isinstance(raw.get("live_feedback"), dict) else {}
    activation = live.get("activation_feedback")
    if isinstance(activation, dict):
        return activation
    fallback_activation = nested_dict(raw, "fallback", "selected_activation_feedback")
    if fallback_activation:
        return fallback_activation
    return {}


def outcome_feedback_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    outcome = raw.get("outcome_history_feedback")
    if isinstance(outcome, dict):
        return outcome
    fallback_outcome = nested_dict(raw, "fallback", "selected_outcome_history_feedback")
    if fallback_outcome:
        return fallback_outcome
    return {}


def selection_identity_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    recommendation = raw.get("recommendation")
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    evidence_refs = recommendation.get("evidence_refs")
    evidence_refs = evidence_refs if isinstance(evidence_refs, dict) else {}
    target_override = recommendation.get("target_override")
    target_override = target_override if isinstance(target_override, dict) else {}
    signals = recommendation.get("signals")
    signals = signals if isinstance(signals, dict) else {}
    return {
        "schema": "codexlab.paint_control_skill_selection_identity.v1",
        "primitive": recommendation.get("primitive"),
        "native_action_id": recommendation.get("native_action_id") or raw.get("native_action_id"),
        "object_slot": recommendation.get("object_slot") or nested_dict(raw, "native_action").get("object_slot"),
        "candidate_kind": recommendation.get("candidate_kind"),
        "source_candidate_kind": recommendation.get("source_candidate_kind"),
        "portband_task_key": evidence_refs.get("portband_task_key"),
        "active_portband_task_key": evidence_refs.get("active_portband_task_key"),
        "portband_route_key": evidence_refs.get("portband_route_key"),
        "lineage_key": evidence_refs.get("lineage_key"),
        "lineup_matrix_row_id": evidence_refs.get("lineup_matrix_row_id"),
        "source_component_ids": target_override.get("source_component_ids") or evidence_refs.get("source_component_ids"),
        "source_object_id": target_override.get("source_object_id") or evidence_refs.get("source_object_id"),
        "source_bbox_xyxy": target_override.get("source_bbox_xyxy") or evidence_refs.get("source_bbox_xyxy"),
        "click_xy": target_override.get("click_xy"),
        "component_id": target_override.get("component_id"),
        "target_kind": target_override.get("target_kind"),
        "selection_basis": target_override.get("selection_basis"),
        "target_component_ids": evidence_refs.get("target_component_ids"),
        "target_port_ids": evidence_refs.get("target_port_ids"),
        "edge_lock_rank": evidence_refs.get("edge_lock_rank"),
        "edge_lock_top_configuration_id": evidence_refs.get("edge_lock_top_configuration_id"),
        "match_status": signals.get("match_status"),
        "base_transform_primitive": signals.get("base_transform_primitive"),
        "lineup_matrix_score": signals.get("lineup_matrix_score"),
        "lineup_step_cost": signals.get("lineup_step_cost"),
        "source_selection_status": signals.get("source_selection_status"),
        "truth_boundary": (
            "selection_identity_is_controller_feedback_for_hypothesis_mining_not_selection_truth"
        ),
    }


def activation_candidate_feedbacks_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    activation = activation_feedback_from_raw(raw)
    if activation and activation.get("primitive") == "activate_flow":
        candidates.append(
            {
                "candidate_role": "selected_or_direct",
                "action_id": nested_dict(raw, "native_action").get("action_id"),
                "primitive": "activate_flow",
                "activation_feedback": activation,
            }
        )
    rejected = nested_dict(raw, "fallback").get("rejected")
    if isinstance(rejected, list):
        for row in rejected:
            if not isinstance(row, dict):
                continue
            activation_feedback = row.get("activation_feedback")
            if not isinstance(activation_feedback, dict) or activation_feedback.get("primitive") != "activate_flow":
                continue
            candidates.append(
                {
                    "candidate_role": "rejected",
                    "action_id": row.get("action_id"),
                    "primitive": row.get("primitive"),
                    "reason": row.get("reason"),
                    "activation_feedback": activation_feedback,
                }
            )
    return candidates


def _first_string(values: Any) -> str | None:
    if isinstance(values, list):
        for value in values:
            if isinstance(value, str) and value:
                return value
    return None


def _candidate_source_key(candidate: dict[str, Any]) -> str | None:
    selection_feedback = candidate.get("selection_feedback")
    selection_feedback = selection_feedback if isinstance(selection_feedback, dict) else {}
    source_key = selection_feedback.get("candidate_source_key")
    if isinstance(source_key, str) and source_key:
        return source_key
    outcome = candidate.get("outcome_history_feedback")
    outcome = outcome if isinstance(outcome, dict) else {}
    candidate_keys = outcome.get("candidate_keys")
    if isinstance(candidate_keys, list):
        for key in candidate_keys:
            if isinstance(key, str) and key.startswith("source_component:"):
                return key
        return _first_string(candidate_keys)
    return None


def _feedback_status(feedback: Any) -> dict[str, Any]:
    if not isinstance(feedback, dict):
        return {"status": None, "fail_closed_reason": None}
    return {
        "status": feedback.get("status"),
        "fail_closed_reason": feedback.get("fail_closed_reason"),
    }


def _sanitize_candidate_score(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    keys = [
        "score",
        "rank_index",
        "primitive",
        "current_display_coord",
        "goal_direction_primitives",
        "recent_positive_count",
        "recent_stationary_count",
        "recent_harmful_count",
        "forward_model_demoted",
        "forward_model_demotion_penalty",
    ]
    return {key: value.get(key) for key in keys if key in value}


def fallback_rejected_candidates_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    rejected = nested_dict(raw, "fallback").get("rejected")
    if not isinstance(rejected, list):
        return []
    candidates: list[dict[str, Any]] = []
    for row in rejected:
        if not isinstance(row, dict):
            continue
        candidate_score = _sanitize_candidate_score(row.get("candidate_score"))
        outcome = row.get("outcome_history_feedback") if isinstance(row.get("outcome_history_feedback"), dict) else {}
        activation = row.get("activation_feedback") if isinstance(row.get("activation_feedback"), dict) else {}
        selection_budget = (
            row.get("selection_budget_feedback") if isinstance(row.get("selection_budget_feedback"), dict) else {}
        )
        selection_feedback = row.get("selection_feedback") if isinstance(row.get("selection_feedback"), dict) else {}
        candidate = {
            "schema": "codexlab.paint_control_skill_fallback_rejected_candidate.v1",
            "action_id": row.get("action_id"),
            "native_action_id": row.get("action_id"),
            "primitive": row.get("primitive"),
            "reason": row.get("reason"),
            "source_key": _candidate_source_key(row),
            "candidate_keys": outcome.get("candidate_keys") if isinstance(outcome.get("candidate_keys"), list) else [],
            "candidate_score": candidate_score,
            "forward_model_status": "demoted" if candidate_score.get("forward_model_demoted") is True else "not_recorded",
            "matched_outcome_rule_id": outcome.get("matched_rule_id"),
            "outcome_effect_kind": outcome.get("effect_kind"),
            "outcome_confidence": outcome.get("confidence"),
            "outcome_status": outcome.get("status"),
            "outcome_fail_closed_reason": outcome.get("fail_closed_reason"),
            "activation_readiness": activation.get("activation_readiness"),
            "activation_target_distance": activation.get("target_distance"),
            "activation_status": activation.get("status"),
            "activation_fail_closed_reason": activation.get("fail_closed_reason"),
            "selection_status": selection_feedback.get("status"),
            "selection_fail_closed_reason": selection_feedback.get("fail_closed_reason"),
            "selection_budget_status": selection_budget.get("status"),
            "selection_budget_fail_closed_reason": selection_budget.get("fail_closed_reason"),
            "selection_budget_remaining": selection_budget.get("remaining_action_budget"),
            "selection_budget_estimated_cost": selection_budget.get("estimated_action_cost"),
            "selection_budget_post_selection": selection_budget.get("post_selection_action_budget"),
            "feedback_statuses": {
                "candidate": _feedback_status(row.get("candidate_feedback")),
                "selection": _feedback_status(row.get("selection_feedback")),
                "selection_budget": _feedback_status(row.get("selection_budget_feedback")),
                "activation": _feedback_status(row.get("activation_feedback")),
                "outcome": _feedback_status(row.get("outcome_history_feedback")),
            },
            "truth_boundary": "fallback_rejected_candidate_is_controller_audit_not_action_truth_or_solve_truth",
        }
        candidates.append(candidate)
    return candidates


def fallback_viable_candidates_from_raw(raw: dict[str, Any]) -> list[dict[str, Any]]:
    viable = nested_dict(raw, "fallback").get("viable_candidates")
    if not isinstance(viable, list):
        return []
    candidates: list[dict[str, Any]] = []
    for row in viable:
        if not isinstance(row, dict):
            continue
        candidate_score = _sanitize_candidate_score(row.get("candidate_score"))
        candidates.append(
            {
                "schema": "codexlab.paint_control_skill_fallback_viable_candidate.v1",
                "action_id": row.get("action_id"),
                "native_action_id": row.get("action_id"),
                "primitive": row.get("primitive"),
                "candidate_score": candidate_score,
                "forward_model_status": "demoted" if candidate_score.get("forward_model_demoted") is True else "not_demoted",
                "truth_boundary": "fallback_viable_candidate_is_controller_audit_not_action_truth_or_solve_truth",
            }
        )
    return candidates


def source_edge_context(row: dict[str, Any]) -> dict[str, Any]:
    activation = row.get("activation_feedback") if isinstance(row.get("activation_feedback"), dict) else {}
    progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
    context = {
        "schema": "codexlab.paint_control_skill_source_edge_context.v1",
        "primitive": row.get("primitive"),
        "active_portband_source_key": row.get("active_portband_source_key"),
        "focus_edge": row.get("focus_edge"),
        "target_distance": activation.get("target_distance"),
        "activation_readiness": activation.get("activation_readiness"),
        "activation_fail_closed_reason": activation.get("fail_closed_reason"),
        "outcome_fail_closed_reason": (row.get("outcome_history_feedback") or {}).get("fail_closed_reason")
        if isinstance(row.get("outcome_history_feedback"), dict)
        else None,
        "selected_display_coord_before": row.get("selected_display_coord_before"),
        "selected_display_coord_after": row.get("selected_display_coord_after"),
        "selected_display_coord_delta": row.get("selected_display_coord_delta"),
        "selected_name_before": row.get("selected_name_before"),
        "selected_name_after": row.get("selected_name_after"),
        "selected_shape_before": row.get("selected_shape_before"),
        "selected_shape_after": row.get("selected_shape_after"),
        "distance_delta": progress.get("distance_delta"),
        "levels_completed_delta": row.get("levels_completed_delta"),
        "label": row.get("label"),
        "truth_boundary": "source_edge_context_is_controlled_feedback_for_hypothesis_mining_not_action_or_solve_truth",
    }
    return context


def selection_outcome_context(row: dict[str, Any]) -> dict[str, Any]:
    identity = row.get("selection_identity") if isinstance(row.get("selection_identity"), dict) else {}
    progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
    return {
        "schema": "codexlab.paint_control_skill_selection_outcome_context.v1",
        "primitive": row.get("primitive"),
        "selection_identity": identity,
        "portband_task_key": identity.get("portband_task_key"),
        "source_component_ids": identity.get("source_component_ids"),
        "source_object_id": identity.get("source_object_id"),
        "click_xy": identity.get("click_xy"),
        "selected_name_before": row.get("selected_name_before"),
        "selected_name_after": row.get("selected_name_after"),
        "selected_shape_before": row.get("selected_shape_before"),
        "selected_shape_after": row.get("selected_shape_after"),
        "selected_display_coord_before": row.get("selected_display_coord_before"),
        "selected_display_coord_after": row.get("selected_display_coord_after"),
        "selected_display_coord_delta": row.get("selected_display_coord_delta"),
        "distance_delta": progress.get("distance_delta"),
        "levels_completed_delta": row.get("levels_completed_delta"),
        "label": row.get("label"),
        "truth_boundary": "selection_outcome_context_is_controlled_feedback_for_hypothesis_mining_not_action_or_solve_truth",
    }


def source_edge_outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    activation_rows = 0
    activation_candidates = 0
    for row in rows:
        candidates = row.get("activation_candidate_feedbacks")
        if not isinstance(candidates, list):
            candidates = []
        if row.get("primitive") == "activate_flow":
            activation_rows += 1
            rejected_candidates = [
                candidate for candidate in candidates
                if isinstance(candidate, dict) and candidate.get("candidate_role") == "rejected"
            ]
            candidates = [
                {
                    "candidate_role": "executed",
                    "activation_feedback": row.get("activation_feedback") if isinstance(row.get("activation_feedback"), dict) else {},
                    "reason": None,
                },
                *rejected_candidates,
            ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            activation = candidate.get("activation_feedback")
            if not isinstance(activation, dict):
                continue
            activation_candidates += 1
            context = source_edge_context({**row, "activation_feedback": activation})
            before = context.get("selected_display_coord_before")
            before_key = ",".join(str(v) for v in before) if isinstance(before, list) else "unknown"
            selected_transition_key = "unknown_transition"
            if row.get("primitive") == "activate_flow" and candidate.get("candidate_role") == "executed":
                before_name = str(context.get("selected_name_before") or "unknown_before")
                after_name = str(context.get("selected_name_after") or "unknown_after")
                before_shape = context.get("selected_shape_before")
                after_shape = context.get("selected_shape_after")
                before_shape_key = ",".join(str(v) for v in before_shape) if isinstance(before_shape, list) else "unknown_shape"
                after_shape_key = ",".join(str(v) for v in after_shape) if isinstance(after_shape, list) else "unknown_shape"
                selected_transition_key = "|".join(
                    [
                        "selected_" + before_name + "_to_" + after_name,
                        "shape_" + before_shape_key + "_to_" + after_shape_key,
                    ]
                )
            key = "|".join(
                [
                    str(context.get("active_portband_source_key") or "unknown_source"),
                    "edge_" + str(context.get("focus_edge")),
                    "dist_" + str(context.get("target_distance")),
                    "coord_" + before_key,
                    selected_transition_key,
                ]
            )
            candidate_role = str(candidate.get("candidate_role") or "unknown")
            fail_reason = candidate.get("reason") or activation.get("fail_closed_reason")
            label = str(context.get("label") or "unknown")
            if candidate_role == "rejected" and fail_reason:
                label = "fail_closed_" + str(fail_reason)
            bucket = buckets.setdefault(
                key,
                {
                    "key": key,
                    "active_portband_source_key": context.get("active_portband_source_key"),
                    "focus_edge": context.get("focus_edge"),
                    "target_distance": context.get("target_distance"),
                    "selected_display_coord_before": before,
                    "selected_display_coord_after": context.get("selected_display_coord_after"),
                    "selected_display_coord_delta": context.get("selected_display_coord_delta"),
                    "selected_name_before": context.get("selected_name_before")
                    if selected_transition_key != "unknown_transition"
                    else None,
                    "selected_name_after": context.get("selected_name_after")
                    if selected_transition_key != "unknown_transition"
                    else None,
                    "selected_shape_before": context.get("selected_shape_before")
                    if selected_transition_key != "unknown_transition"
                    else None,
                    "selected_shape_after": context.get("selected_shape_after")
                    if selected_transition_key != "unknown_transition"
                    else None,
                    "selected_transition_key": selected_transition_key,
                    "row_count": 0,
                    "candidate_count": 0,
                    "rejected_candidate_count": 0,
                    "solve_positive_count": 0,
                    "positive_goal_distance_reduced_count": 0,
                    "negative_goal_distance_increased_count": 0,
                    "neutral_no_goal_progress_count": 0,
                    "hard_negative_stationary_move_count": 0,
                    "fail_closed_count": 0,
                    "labels": {},
                    "example_action_indices": [],
                    "truth_boundary": "source_edge_outcome_summary_is_feedback_mining_not_control_truth_not_solve_truth",
                },
            )
            if row.get("primitive") == "activate_flow":
                bucket["row_count"] += 1
            bucket["candidate_count"] += 1
            if candidate_role == "rejected":
                bucket["rejected_candidate_count"] += 1
            bucket["labels"][label] = int(bucket["labels"].get(label, 0)) + 1
            if label == "solve_positive":
                bucket["solve_positive_count"] += 1
            elif label == "positive_goal_distance_reduced":
                bucket["positive_goal_distance_reduced_count"] += 1
            elif label == "negative_goal_distance_increased":
                bucket["negative_goal_distance_increased_count"] += 1
            elif label == "neutral_no_goal_progress":
                bucket["neutral_no_goal_progress_count"] += 1
            elif label == "hard_negative_stationary_move":
                bucket["hard_negative_stationary_move_count"] += 1
            elif label.startswith("fail_closed_"):
                bucket["fail_closed_count"] += 1
            action_index = row.get("action_index")
            if action_index not in bucket["example_action_indices"] and len(bucket["example_action_indices"]) < 8:
                bucket["example_action_indices"].append(action_index)
    activation_buckets = sorted(
        buckets.values(),
        key=lambda item: (
            -int(item.get("solve_positive_count") or 0),
            -int(item.get("positive_goal_distance_reduced_count") or 0),
            int(item.get("negative_goal_distance_increased_count") or 0),
            str(item.get("key") or ""),
        ),
    )
    return {
        "schema": SOURCE_EDGE_SUMMARY_SCHEMA,
        "activation_row_count": activation_rows,
        "activation_candidate_count": activation_candidates,
        "activation_bucket_count": len(activation_buckets),
        "activation_positive_level_count": sum(int(row.get("solve_positive_count") or 0) for row in activation_buckets),
        "activation_positive_distance_count": sum(int(row.get("positive_goal_distance_reduced_count") or 0) for row in activation_buckets),
        "activation_negative_distance_count": sum(int(row.get("negative_goal_distance_increased_count") or 0) for row in activation_buckets),
        "activation_neutral_count": sum(int(row.get("neutral_no_goal_progress_count") or 0) for row in activation_buckets),
        "activation_fail_closed_count": sum(int(row.get("fail_closed_count") or 0) for row in activation_buckets),
        "activation_buckets": activation_buckets,
        "truth_boundary": "source_edge_summary_mines_controlled_feedback_for_hypotheses_not_action_truth_goal_truth_or_solve_truth",
    }


def selection_outcome_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    selection_rows = 0
    for row in rows:
        if row.get("primitive") != "select_object":
            continue
        selection_rows += 1
        context = selection_outcome_context(row)
        identity = context.get("selection_identity") if isinstance(context.get("selection_identity"), dict) else {}
        source_components = context.get("source_component_ids")
        if isinstance(source_components, list) and source_components:
            source_key = ",".join(str(item) for item in source_components)
        else:
            source_key = str(context.get("portband_task_key") or context.get("source_object_id") or "unknown_source")
        before = context.get("selected_display_coord_before")
        before_key = ",".join(str(v) for v in before) if isinstance(before, list) else "unknown"
        after = context.get("selected_display_coord_after")
        after_key = ",".join(str(v) for v in after) if isinstance(after, list) else "unknown"
        key = "|".join(
            [
                source_key,
                "native_" + str(identity.get("native_action_id")),
                "before_" + before_key,
                "after_" + after_key,
            ]
        )
        label = str(context.get("label") or "unknown")
        bucket = buckets.setdefault(
            key,
            {
                "key": key,
                "source_key": source_key,
                "portband_task_key": context.get("portband_task_key"),
                "source_component_ids": source_components,
                "source_object_id": context.get("source_object_id"),
                "click_xy": context.get("click_xy"),
                "native_action_id": identity.get("native_action_id"),
                "object_slot": identity.get("object_slot"),
                "selected_name_before": context.get("selected_name_before"),
                "selected_name_after": context.get("selected_name_after"),
                "selected_display_coord_before": before,
                "selected_display_coord_after": after,
                "row_count": 0,
                "solve_positive_count": 0,
                "positive_goal_distance_reduced_count": 0,
                "negative_goal_distance_increased_count": 0,
                "neutral_no_goal_progress_count": 0,
                "hard_negative_stationary_move_count": 0,
                "labels": {},
                "example_action_indices": [],
                "example_runs": [],
                "truth_boundary": "selection_outcome_summary_is_feedback_mining_not_selection_truth_not_solve_truth",
            },
        )
        bucket["row_count"] += 1
        bucket["labels"][label] = int(bucket["labels"].get(label, 0)) + 1
        if label == "solve_positive":
            bucket["solve_positive_count"] += 1
        elif label == "positive_goal_distance_reduced":
            bucket["positive_goal_distance_reduced_count"] += 1
        elif label == "negative_goal_distance_increased":
            bucket["negative_goal_distance_increased_count"] += 1
        elif label == "neutral_no_goal_progress":
            bucket["neutral_no_goal_progress_count"] += 1
        elif label == "hard_negative_stationary_move":
            bucket["hard_negative_stationary_move_count"] += 1
        if len(bucket["example_action_indices"]) < 8:
            bucket["example_action_indices"].append(row.get("action_index"))
        level_dir = row.get("level_dir")
        if isinstance(level_dir, str) and len(bucket["example_runs"]) < 5 and level_dir not in bucket["example_runs"]:
            bucket["example_runs"].append(level_dir)
    selection_buckets = sorted(
        buckets.values(),
        key=lambda item: (
            -int(item.get("solve_positive_count") or 0),
            -int(item.get("positive_goal_distance_reduced_count") or 0),
            int(item.get("negative_goal_distance_increased_count") or 0),
            str(item.get("key") or ""),
        ),
    )
    return {
        "schema": SELECTION_OUTCOME_SUMMARY_SCHEMA,
        "selection_row_count": selection_rows,
        "selection_bucket_count": len(selection_buckets),
        "selection_positive_level_count": sum(int(row.get("solve_positive_count") or 0) for row in selection_buckets),
        "selection_positive_distance_count": sum(int(row.get("positive_goal_distance_reduced_count") or 0) for row in selection_buckets),
        "selection_negative_distance_count": sum(int(row.get("negative_goal_distance_increased_count") or 0) for row in selection_buckets),
        "selection_neutral_count": sum(int(row.get("neutral_no_goal_progress_count") or 0) for row in selection_buckets),
        "selection_buckets": selection_buckets,
        "truth_boundary": "selection_summary_mines_controlled_feedback_for_hypotheses_not_action_truth_goal_truth_or_solve_truth",
    }


def _bump(counter: dict[str, int], key: Any, *, default: str = "unknown") -> None:
    text = str(key) if key not in {None, ""} else default
    counter[text] = int(counter.get(text, 0)) + 1


def _fallback_filter_family(candidate: dict[str, Any]) -> str:
    reason = str(candidate.get("reason") or "")
    primitive = str(candidate.get("primitive") or "")
    if "budget" in reason:
        return "selection_budget"
    if reason.startswith("selection_"):
        return "source_selection"
    if reason.startswith("activation_") or primitive == "activate_flow":
        return "activation"
    if "outcome" in reason:
        return "outcome_history"
    if reason.startswith("fallback_") or primitive.startswith("move_"):
        return "motion_feedback"
    if reason.startswith("native_") or reason == "invalid_edge_authority":
        return "native_or_edge_authority"
    return "other"


def fallback_exhaustion_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    by_primitive: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    by_filter_family: dict[str, int] = {}
    exhausted_decisions = 0
    total_rejected = 0
    final_viable_total = 0
    zero_viable_terminal_states = 0
    missing_rejected_candidate_lists = 0

    for row in rows:
        exhausted = (
            row.get("label") == "fail_closed_ranked_fallback_exhausted"
            or row.get("fallback_status") == "ranked_fallback_exhausted"
        )
        if not exhausted:
            continue
        exhausted_decisions += 1
        viable = row.get("fallback_viable_candidates")
        viable = viable if isinstance(viable, list) else []
        final_viable_total += len(viable)
        if not viable:
            zero_viable_terminal_states += 1
        rejected = row.get("fallback_rejected_candidates")
        if not isinstance(rejected, list) or not rejected:
            missing_rejected_candidate_lists += 1
            rejected = [
                {
                    "action_id": None,
                    "native_action_id": None,
                    "primitive": "unknown_primitive",
                    "reason": "unavailable_rejected_candidates",
                    "source_key": "unknown_source",
                    "matched_outcome_rule_id": None,
                    "forward_model_status": "not_recorded",
                    "candidate_score": {},
                    "truth_boundary": "fallback_exhaustion_missing_rejected_candidates_is_fail_closed_audit_gap",
                }
            ]
        before = row.get("selected_display_coord_before")
        before_key = ",".join(str(v) for v in before) if isinstance(before, list) else "unknown_coord"
        selected_name = row.get("selected_name_before") or "unknown_selected"
        for candidate in rejected:
            if not isinstance(candidate, dict):
                continue
            primitive = str(candidate.get("primitive") or "unknown_primitive")
            reason = str(candidate.get("reason") or "unknown_reason")
            source_key = str(candidate.get("source_key") or "unknown_source")
            matched_rule = str(candidate.get("matched_outcome_rule_id") or "no_rule")
            forward_model_status = str(candidate.get("forward_model_status") or "not_recorded")
            filter_family = _fallback_filter_family(candidate)
            total_rejected += 1
            _bump(by_primitive, primitive)
            _bump(by_reason, reason)
            _bump(by_filter_family, filter_family)
            key = "|".join(
                [
                    "coord_" + before_key,
                    "selected_" + str(selected_name),
                    "primitive_" + primitive,
                    "reason_" + reason,
                    "source_" + source_key,
                    "rule_" + matched_rule,
                    "forward_" + forward_model_status,
                ]
            )
            bucket = buckets.setdefault(
                key,
                {
                    "key": key,
                    "selected_display_coord_before": before,
                    "selected_name_before": row.get("selected_name_before"),
                    "selected_shape_before": row.get("selected_shape_before"),
                    "target_coord": None,
                    "primitive": primitive,
                    "reason": reason,
                    "filter_family": filter_family,
                    "source_key": source_key,
                    "matched_outcome_rule_id": matched_rule,
                    "forward_model_status": forward_model_status,
                    "row_count": 0,
                    "rejected_candidate_count": 0,
                    "final_viable_candidate_count": 0,
                    "candidate_action_ids": [],
                    "candidate_keys": [],
                    "example_action_indices": [],
                    "original_fail_reasons": {},
                    "truth_boundary": "fallback_exhaustion_bucket_is_feedback_mining_not_control_truth_not_solve_truth",
                },
            )
            bucket["row_count"] += 1
            bucket["rejected_candidate_count"] += 1
            bucket["final_viable_candidate_count"] += len(viable)
            action_id = candidate.get("action_id")
            if action_id is not None and len(bucket["candidate_action_ids"]) < 12:
                bucket["candidate_action_ids"].append(action_id)
            candidate_keys = candidate.get("candidate_keys")
            if isinstance(candidate_keys, list):
                for candidate_key in candidate_keys:
                    if candidate_key not in bucket["candidate_keys"] and len(bucket["candidate_keys"]) < 12:
                        bucket["candidate_keys"].append(candidate_key)
            action_index = row.get("action_index")
            if action_index not in bucket["example_action_indices"] and len(bucket["example_action_indices"]) < 8:
                bucket["example_action_indices"].append(action_index)
            original_reason = str(row.get("fallback_original_fail_reason") or "unknown_original_fail_reason")
            bucket["original_fail_reasons"][original_reason] = int(bucket["original_fail_reasons"].get(original_reason, 0)) + 1

    sorted_buckets = sorted(
        buckets.values(),
        key=lambda item: (
            -int(item.get("rejected_candidate_count") or 0),
            str(item.get("primitive") or ""),
            str(item.get("reason") or ""),
            str(item.get("source_key") or ""),
        ),
    )
    return {
        "schema": FALLBACK_EXHAUSTION_SUMMARY_SCHEMA,
        "exhausted_decision_count": exhausted_decisions,
        "total_rejected_candidate_count": total_rejected,
        "missing_rejected_candidate_list_count": missing_rejected_candidate_lists,
        "zero_viable_terminal_state_count": zero_viable_terminal_states,
        "final_viable_candidate_count": final_viable_total,
        "viable_candidates_after_filter_family": {
            "recorded_final": final_viable_total,
            "after_all_filters": final_viable_total,
        },
        "rejected_candidate_count_by_primitive": by_primitive,
        "rejected_candidate_count_by_reason": by_reason,
        "rejected_candidate_count_by_filter_family": by_filter_family,
        "fallback_exhaustion_bucket_count": len(sorted_buckets),
        "fallback_exhaustion_buckets": sorted_buckets,
        "truth_boundary": "fallback_exhaustion_summary_mines_controller_audits_not_action_truth_goal_truth_or_solve_truth",
    }


def build_row(level_dir: Path, index: int, packet: dict[str, Any], brain_call: dict[str, Any] | None) -> dict[str, Any]:
    decision = (brain_call or {}).get("contra_decision") or packet.get("contra_decision") or packet.get("brain_decision") or {}
    raw = decision.get("raw") if isinstance(decision.get("raw"), dict) else {}
    native = raw.get("native_action") if isinstance(raw.get("native_action"), dict) else {}
    label = classify(packet, raw)
    before_state = packet.get("before_state") if isinstance(packet.get("before_state"), dict) else {}
    after_state = packet.get("after_state") if isinstance(packet.get("after_state"), dict) else {}
    activation_feedback = activation_feedback_from_raw(raw)
    outcome_feedback = outcome_feedback_from_raw(raw)
    activation_candidate_feedbacks = activation_candidate_feedbacks_from_raw(raw)
    selection_identity = selection_identity_from_raw(raw)
    fallback_rejected_candidates = fallback_rejected_candidates_from_raw(raw)
    fallback_viable_candidates = fallback_viable_candidates_from_raw(raw)
    row = {
        "schema": ROW_SCHEMA,
        "level_dir": rel(level_dir),
        "row_index": int(index),
        "case_id": packet.get("case_id"),
        "action_index": packet.get("action_index"),
        "label": label,
        "packet_status": packet.get("status"),
        "packet_error": packet.get("error"),
        "primitive": packet.get("decision_primitive"),
        "native_action_id": raw.get("native_action_id"),
        "native_action_kind": native.get("action_kind"),
        "native_object_slot": native.get("object_slot"),
        "native_primitive": native.get("primitive"),
        "confidence": decision.get("confidence"),
        "observation_hash": raw.get("observation_hash"),
        "observation_shape": raw.get("observation_shape"),
        "checkpoint_path": raw.get("checkpoint_path"),
        "checkpoint_sha256": raw.get("checkpoint_sha256"),
        "focus_edge": raw.get("focus_edge"),
        "active_portband_source_key": raw.get("active_portband_source_key"),
        "stack_depth": raw.get("stack_depth"),
        "required_count": raw.get("required_count"),
        "known_object_slots": raw.get("known_object_slots"),
        "object_slot_mapping_status": raw.get("object_slot_mapping_status"),
        "fail_closed_reason": raw.get("fail_closed_reason") or packet.get("error"),
        "live_feedback": raw.get("live_feedback") if isinstance(raw.get("live_feedback"), dict) else {},
        "activation_feedback": activation_feedback,
        "activation_candidate_feedbacks": activation_candidate_feedbacks,
        "outcome_history_feedback": outcome_feedback,
        "selection_identity": selection_identity,
        "fallback_original_fail_reason": nested_dict(raw, "fallback").get("original_fail_reason"),
        "fallback_status": nested_dict(raw, "fallback").get("status"),
        "fallback_rejected_candidates": fallback_rejected_candidates,
        "fallback_viable_candidates": fallback_viable_candidates,
        "selected_coord_before": before_state.get("selected_coord"),
        "selected_coord_after": after_state.get("selected_coord"),
        "selected_display_coord_before": before_state.get("selected_display_coord"),
        "selected_display_coord_after": after_state.get("selected_display_coord"),
        "selected_name_before": before_state.get("selected_name"),
        "selected_name_after": after_state.get("selected_name"),
        "selected_shape_before": before_state.get("selected_shape"),
        "selected_shape_after": after_state.get("selected_shape"),
        "selected_coord_delta": packet.get("selected_coord_delta"),
        "selected_display_coord_delta": packet.get("selected_display_coord_delta"),
        "goal_progress": packet.get("goal_progress"),
        "levels_completed_delta": int(packet.get("levels_completed_delta") or 0),
        "total_levels_completed_delta": int(packet.get("total_levels_completed_delta") or 0),
        "before_png": image_path(packet.get("before_frame")),
        "after_png": image_path(packet.get("after_frame")),
        "truth_boundary": (
            "controlled SP80 harness feedback row; useful as bridge-training signal, "
            "not a solved-level claim unless levels_completed_delta is positive"
        ),
    }
    row["source_edge_context"] = source_edge_context(row)
    row["selection_outcome_context"] = selection_outcome_context(row)
    return row


def level_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "packets.jsonl").is_file():
        return [run_dir]
    return sorted(path for path in run_dir.glob("level_*") if path.is_dir() and (path / "packets.jsonl").is_file())


def write_index(out_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    visual_rows = [row for row in rows if row.get("before_png")][:16]
    cards = "\n".join(
        f"""
        <article>
          <h3>{html.escape(str(row.get("case_id")))}: {html.escape(str(row.get("label")))}</h3>
          <div class="frames">
            <figure><img src="{html.escape(html_asset_src(out_dir, row.get("before_png")))}" alt="before"><figcaption>Before</figcaption></figure>
            {f'<figure><img src="{html.escape(html_asset_src(out_dir, row.get("after_png")))}" alt="after"><figcaption>After</figcaption></figure>' if row.get("after_png") else ''}
          </div>
          <table>
            <tr><th>Primitive</th><td>{html.escape(str(row.get("primitive")))}</td></tr>
            <tr><th>Native action</th><td>{html.escape(str(row.get("native_action_id")))}</td></tr>
            <tr><th>Display delta</th><td>{html.escape(json.dumps(row.get("selected_display_coord_delta")))}</td></tr>
            <tr><th>Distance delta</th><td>{html.escape(str((row.get("goal_progress") or {}).get("distance_delta")))}</td></tr>
            <tr><th>Fail reason</th><td>{html.escape(str(row.get("fail_closed_reason") or ""))}</td></tr>
          </table>
        </article>
        """
        for row in visual_rows
    )
    label_rows = "\n".join(
        f"<tr><td>{html.escape(str(label))}</td><td>{count}</td></tr>"
        for label, count in sorted(summary["label_counts"].items())
    )
    source_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(summary["source_hashes"].items())
    )
    source_edge = summary.get("source_edge_outcome_summary")
    source_edge = source_edge if isinstance(source_edge, dict) else {}
    selection_summary = summary.get("selection_outcome_summary")
    selection_summary = selection_summary if isinstance(selection_summary, dict) else {}
    fallback_summary = summary.get("fallback_exhaustion_summary")
    fallback_summary = fallback_summary if isinstance(fallback_summary, dict) else {}
    activation_bucket_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(str(row.get("active_portband_source_key") or "unknown"))}</td>
          <td>{html.escape(str(row.get("focus_edge")))}</td>
          <td>{html.escape(str(row.get("target_distance")))}</td>
          <td>{html.escape(json.dumps(row.get("selected_display_coord_before")))}</td>
          <td>{html.escape(str(row.get("selected_name_before") or ""))}</td>
          <td>{html.escape(str(row.get("selected_name_after") or ""))}</td>
          <td>{html.escape(str(row.get("selected_transition_key") or ""))}</td>
          <td>{int(row.get("row_count") or 0)}</td>
          <td>{int(row.get("solve_positive_count") or 0)}</td>
          <td>{int(row.get("positive_goal_distance_reduced_count") or 0)}</td>
          <td>{int(row.get("negative_goal_distance_increased_count") or 0)}</td>
          <td>{int(row.get("neutral_no_goal_progress_count") or 0)}</td>
          <td>{html.escape(json.dumps(row.get("example_action_indices")))}</td>
        </tr>
        """
        for row in source_edge.get("activation_buckets", [])[:24]
        if isinstance(row, dict)
    )
    selection_bucket_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(str(row.get("source_key") or "unknown"))}</td>
          <td>{html.escape(str(row.get("native_action_id")))}</td>
          <td>{html.escape(json.dumps(row.get("click_xy")))}</td>
          <td>{html.escape(json.dumps(row.get("selected_display_coord_before")))}</td>
          <td>{html.escape(json.dumps(row.get("selected_display_coord_after")))}</td>
          <td>{int(row.get("row_count") or 0)}</td>
          <td>{int(row.get("solve_positive_count") or 0)}</td>
          <td>{int(row.get("positive_goal_distance_reduced_count") or 0)}</td>
          <td>{int(row.get("negative_goal_distance_increased_count") or 0)}</td>
          <td>{int(row.get("neutral_no_goal_progress_count") or 0)}</td>
          <td>{html.escape(json.dumps(row.get("example_action_indices")))}</td>
        </tr>
        """
        for row in selection_summary.get("selection_buckets", [])[:24]
        if isinstance(row, dict)
    )
    fallback_bucket_rows = "\n".join(
        f"""
        <tr>
          <td>{html.escape(json.dumps(row.get("selected_display_coord_before")))}</td>
          <td>{html.escape(str(row.get("selected_name_before") or "unknown"))}</td>
          <td>{html.escape(str(row.get("primitive") or "unknown"))}</td>
          <td>{html.escape(str(row.get("reason") or "unknown"))}</td>
          <td>{html.escape(str(row.get("filter_family") or "unknown"))}</td>
          <td>{html.escape(str(row.get("source_key") or "unknown"))}</td>
          <td>{html.escape(str(row.get("matched_outcome_rule_id") or "no_rule"))}</td>
          <td>{html.escape(str(row.get("forward_model_status") or "not_recorded"))}</td>
          <td>{int(row.get("rejected_candidate_count") or 0)}</td>
          <td>{html.escape(json.dumps(row.get("candidate_action_ids")))}</td>
          <td>{html.escape(json.dumps(row.get("example_action_indices")))}</td>
        </tr>
        """
        for row in fallback_summary.get("fallback_exhaustion_buckets", [])[:32]
        if isinstance(row, dict)
    )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Paint Control Skill Feedback Export</title>
  <style>
    body {{ margin: 24px; font-family: system-ui, sans-serif; background: #f6f7f9; color: #18202a; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    .boundary {{ color: #53606d; max-width: 920px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    article {{ background: white; border: 1px solid #d9dee6; border-radius: 6px; padding: 12px; }}
    h1 {{ font-size: 26px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-top: 24px; }}
    h3 {{ font-size: 15px; margin: 0 0 10px; }}
    .frames {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; image-rendering: pixelated; border: 1px solid #ccd3dc; background: white; }}
    figcaption {{ font-size: 12px; color: #53606d; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 10px 0 18px; }}
    th, td {{ border: 1px solid #d9dee6; padding: 6px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #eef2f6; }}
    code {{ background: #e9edf2; padding: 2px 4px; }}
  </style>
</head>
<body>
<main>
  <h1>Paint Control Skill Feedback Export</h1>
  <p class="boundary">Controlled SP80 feedback rows for Puff bridge training. This is not solve evidence unless a row records positive level completion.</p>
  <table>
    <tr><th>Run dir</th><td><code>{html.escape(summary["run_dir"])}</code></td></tr>
    <tr><th>Rows</th><td>{summary["row_count"]}</td></tr>
    <tr><th>Output JSONL</th><td><a href="{html.escape(Path(summary["feedback_jsonl"]).name)}">{html.escape(summary["feedback_jsonl"])}</a></td></tr>
    <tr><th>Summary JSON</th><td><a href="{html.escape(Path(summary["summary_json"]).name)}">{html.escape(summary["summary_json"])}</a></td></tr>
  </table>
  <h2>Label Counts</h2>
  <table><tr><th>Label</th><th>Count</th></tr>{label_rows}</table>
  <h2>Source/Edge Activation Outcomes</h2>
  <p class="boundary">Activation rows grouped for mechanism induction. These rows are feedback hypotheses only; they are not action authority, goal truth, or solve truth.</p>
  <table>
    <tr>
      <th>Rows</th><td>{int(source_edge.get("activation_row_count") or 0)}</td>
      <th>Positive level</th><td>{int(source_edge.get("activation_positive_level_count") or 0)}</td>
      <th>Positive distance</th><td>{int(source_edge.get("activation_positive_distance_count") or 0)}</td>
      <th>Negative distance</th><td>{int(source_edge.get("activation_negative_distance_count") or 0)}</td>
    </tr>
  </table>
  <table>
    <tr>
      <th>Source key</th><th>Focus edge</th><th>Target distance</th><th>Before coord</th>
      <th>Selected before</th><th>Selected after</th><th>Transition key</th>
      <th>Rows</th><th>Solve +</th><th>Distance +</th><th>Distance -</th><th>Neutral</th><th>Examples</th>
    </tr>
    {activation_bucket_rows}
  </table>
  <h2>Selection Outcomes</h2>
  <p class="boundary">Selection rows grouped by selected source identity and before/after selected object state. These rows expose candidate source outcomes; they are not selection authority or solve truth.</p>
  <table>
    <tr>
      <th>Rows</th><td>{int(selection_summary.get("selection_row_count") or 0)}</td>
      <th>Positive level</th><td>{int(selection_summary.get("selection_positive_level_count") or 0)}</td>
      <th>Positive distance</th><td>{int(selection_summary.get("selection_positive_distance_count") or 0)}</td>
      <th>Negative distance</th><td>{int(selection_summary.get("selection_negative_distance_count") or 0)}</td>
    </tr>
  </table>
  <table>
    <tr>
      <th>Source key</th><th>Native action</th><th>Click</th><th>Before coord</th><th>After coord</th>
      <th>Rows</th><th>Solve +</th><th>Distance +</th><th>Distance -</th><th>Neutral</th><th>Examples</th>
    </tr>
    {selection_bucket_rows}
  </table>
  <h2>Fallback Exhaustion</h2>
  <p class="boundary">Rejected ranked-fallback candidates grouped for missing-representation diagnosis. These rows are controller audit feedback only; they are not action authority, goal truth, prediction truth, or solve truth.</p>
  <table>
    <tr>
      <th>Exhausted decisions</th><td>{int(fallback_summary.get("exhausted_decision_count") or 0)}</td>
      <th>Rejected candidates</th><td>{int(fallback_summary.get("total_rejected_candidate_count") or 0)}</td>
      <th>Zero viable terminals</th><td>{int(fallback_summary.get("zero_viable_terminal_state_count") or 0)}</td>
      <th>Missing candidate lists</th><td>{int(fallback_summary.get("missing_rejected_candidate_list_count") or 0)}</td>
    </tr>
  </table>
  <table>
    <tr>
      <th>Before coord</th><th>Selected</th><th>Primitive</th><th>Reason</th><th>Family</th>
      <th>Source key</th><th>Rule</th><th>Forward model</th><th>Rejected</th><th>Action IDs</th><th>Examples</th>
    </tr>
    {fallback_bucket_rows}
  </table>
  <h2>Visual Rows</h2>
  <section class="grid">{cards}</section>
  <h2>Source Hashes</h2>
  <table><tr><th>Path</th><th>SHA-256</th></tr>{source_rows}</table>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(text, encoding="utf-8")


def export_feedback(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for level_dir in level_dirs(run_dir):
        packets = read_jsonl(level_dir / "packets.jsonl")
        brain_calls = read_jsonl(level_dir / "brain_calls.jsonl")
        for idx, packet in enumerate(packets):
            brain_call = brain_calls[idx] if idx < len(brain_calls) else None
            rows.append(build_row(level_dir, len(rows), packet, brain_call))

    feedback_jsonl = out_dir / "paint_control_skill_feedback.jsonl"
    with feedback_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    label_counts: dict[str, int] = {}
    for row in rows:
        label = str(row.get("label"))
        label_counts[label] = label_counts.get(label, 0) + 1
    source_edge_summary = source_edge_outcome_summary(rows)
    selection_summary = selection_outcome_summary(rows)
    fallback_summary = fallback_exhaustion_summary(rows)
    summary = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": rel(run_dir),
        "out_dir": rel(out_dir),
        "row_count": len(rows),
        "label_counts": label_counts,
        "source_edge_outcome_summary": source_edge_summary,
        "selection_outcome_summary": selection_summary,
        "fallback_exhaustion_summary": fallback_summary,
        "feedback_jsonl": rel(feedback_jsonl),
        "summary_json": rel(out_dir / "paint_control_skill_feedback_summary.json"),
        "index_html": rel(out_dir / "index.html"),
        "source_hashes": {
            "codexlab/paint_control_skill_feedback_export.py": sha256_file(ROOT / "codexlab" / "paint_control_skill_feedback_export.py"),
            "codexlab/paint_control_skill.py": sha256_file(ROOT / "codexlab" / "paint_control_skill.py"),
            "codexlab/sp80_player_harness.py": sha256_file(ROOT / "codexlab" / "sp80_player_harness.py"),
        },
        "truth_boundary": (
            "Exported from controlled SP80 harness artifacts. Rows can train/evaluate bridge feedback; "
            "they are not native PufferLib training results and not solve evidence without level completion."
        ),
    }
    (out_dir / "paint_control_skill_feedback_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_index(out_dir, summary, rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = run_dir / "paint_control_skill_feedback"
    summary = export_feedback(run_dir, out_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
