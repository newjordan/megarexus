#!/usr/bin/env python3
"""CONTRA episode memory for perception-action-learning loops."""

from __future__ import annotations

import hashlib
import json
from typing import Any


CONTRA_EPISODE_MEMORY_SCHEMA = "codexlab.contra_episode_memory.v1"
CONTRA_ROUND_STATE_SCHEMA = "codexlab.contra_round_state.v1"
CONTRA_FRONTIER_ITEM_SCHEMA = "codexlab.contra_episode_frontier_item.v1"

TRUTH_FLAGS = {
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def packet_hash(packet: dict[str, Any] | None) -> str:
    if not isinstance(packet, dict):
        return "missing_packet"
    basis = packet.get("packet_sha256_basis")
    if isinstance(basis, str) and basis:
        return basis
    payload = json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _guards(evidence_status: str = "episode_memory_working_state_not_truth") -> dict[str, Any]:
    return {
        "prediction_status": "not_predictive",
        "evidence_status": evidence_status,
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _frontier_from_packet(packet: dict[str, Any] | None, *, limit: int = 16) -> list[dict[str, Any]]:
    if not isinstance(packet, dict):
        return []
    rows: list[dict[str, Any]] = []
    for statecon in packet.get("state_condition_hypotheses", []) or []:
        if not isinstance(statecon, dict):
            continue
        statecon_id = str(statecon.get("statecon_id") or "")
        score = _clamp01(float(statecon.get("event_enable_score") or statecon.get("same_statecon_score") or 0.0))
        if statecon_id:
            rows.append(
                {
                    "schema": CONTRA_FRONTIER_ITEM_SCHEMA,
                    "frontier_item_id": statecon_id,
                    "item_kind": "statecon_relation",
                    "score": round(score, 6),
                    "basis": "statecon_event_enable_score",
                    "source_statecon_id": statecon_id,
                    **_guards(),
                }
            )
        portband = statecon.get("portband_hypotheses") if isinstance(statecon.get("portband_hypotheses"), dict) else {}
        matrix = portband.get("lineup_matrix") if isinstance(portband.get("lineup_matrix"), dict) else {}
        for lineup in matrix.get("best_by_support_object", []) or []:
            if not isinstance(lineup, dict):
                continue
            row_id = str(lineup.get("best_lineup_matrix_row_id") or "")
            if not row_id:
                continue
            score_value = (
                lineup.get("effective_lineup_confidence_score")
                if lineup.get("effective_lineup_confidence_score") is not None
                else lineup.get("route_flow_adjusted_confidence_score")
                if lineup.get("route_flow_adjusted_confidence_score") is not None
                else lineup.get("confidence_score")
            )
            rows.append(
                {
                    "schema": CONTRA_FRONTIER_ITEM_SCHEMA,
                    "frontier_item_id": row_id,
                    "item_kind": "portband_lineup_matrix_row",
                    "score": round(_clamp01(float(score_value or 0.0)), 6),
                    "basis": "portband_lineup_matrix_effective_confidence_score_not_probability",
                    "source_statecon_id": statecon_id,
                    "source_ids": [lineup.get("source_object_id")],
                    "lineup_route_label": lineup.get("lineup_route_label"),
                    "transform_hypothesis_xy": lineup.get("transform_hypothesis_xy"),
                    "action_state_tag_status": lineup.get("action_state_tag_status"),
                    "action_state_positive_signal": lineup.get("action_state_positive_signal"),
                    "action_state_correlation_score": lineup.get("action_state_correlation_score"),
                    "action_state_tag_ids": lineup.get("action_state_tag_ids", []),
                    "action_state_primitives": lineup.get("action_state_primitives", []),
                    "action_state_evidence_event_ids": lineup.get("action_state_evidence_event_ids", []),
                    "action_state_action_invocation_ids": lineup.get("action_state_action_invocation_ids", []),
                    "action_state_correlated_object_ids": lineup.get("action_state_correlated_object_ids", []),
                    "relative_state_change_tag_ids": lineup.get("relative_state_change_tag_ids", []),
                    "relative_state_change_novelty_statuses": lineup.get("relative_state_change_novelty_statuses", []),
                    "action_interaction_edge_ids": lineup.get("action_interaction_edge_ids", []),
                    "action_interaction_kinds": lineup.get("action_interaction_kinds", []),
                    "action_interaction_participant_object_ids": lineup.get("action_interaction_participant_object_ids", []),
                    "solve_like_evidence_status": lineup.get("solve_like_evidence_status"),
                    "solve_like_evidence_flags": lineup.get("solve_like_evidence_flags", []),
                    "solve_like_evidence_score": lineup.get("solve_like_evidence_score"),
                    "route_flow_lock_status": lineup.get("route_flow_lock_status"),
                    "route_flow_positive_signal": lineup.get("route_flow_positive_signal"),
                    "route_flow_confidence_score": lineup.get("route_flow_confidence_score"),
                    "route_flow_evidence_event_ids": lineup.get("route_flow_evidence_event_ids", []),
                    "route_flow_action_invocation_ids": lineup.get("route_flow_action_invocation_ids", []),
                    **_guards(),
                }
            )
        fleet = portband.get("portband_fleet") if isinstance(portband.get("portband_fleet"), dict) else {}
        for candidate in fleet.get("ranked_candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            candidate_id = str(candidate.get("fleet_candidate_id") or "")
            if not candidate_id:
                continue
            rows.append(
                {
                    "schema": CONTRA_FRONTIER_ITEM_SCHEMA,
                    "frontier_item_id": candidate_id,
                    "item_kind": "portband_fleet_candidate",
                    "score": round(_clamp01(float(candidate.get("rank_score") or 0.0)), 6),
                    "basis": "portband_fleet_rank_score",
                    "source_statecon_id": statecon_id,
                    "target_ids": candidate.get("target_object_ids", []),
                    "source_ids": candidate.get("source_object_ids", []),
                    **_guards(),
                }
            )
    tabula = packet.get("tabula_rasa_notebook") if isinstance(packet.get("tabula_rasa_notebook"), dict) else {}
    for note in tabula.get("notes", []) or []:
        if not isinstance(note, dict):
            continue
        note_id = str(note.get("note_id") or "")
        if not note_id:
            continue
        rows.append(
            {
                "schema": CONTRA_FRONTIER_ITEM_SCHEMA,
                "frontier_item_id": note_id,
                "item_kind": "tabula_rasa_note",
                "score": 0.25,
                "basis": "open_assumption_requires_observation",
                "source_note_id": note_id,
                "item_ref": note.get("item_id"),
                **_guards(),
            }
        )
    exploreband = packet.get("exploreband_budget") if isinstance(packet.get("exploreband_budget"), dict) else {}
    for probe in exploreband.get("candidate_probe_queue", []) or []:
        if not isinstance(probe, dict):
            continue
        probe_id = str(probe.get("probe_id") or "")
        if not probe_id:
            continue
        rows.append(
            {
                "schema": CONTRA_FRONTIER_ITEM_SCHEMA,
                "frontier_item_id": probe_id,
                "item_kind": "exploreband_budgeted_probe_candidate",
                "score": round(_clamp01(float(probe.get("expected_information_gain") or 0.0)), 6),
                "basis": "exploreband_expected_information_gain",
                "source_component_id": probe.get("component_id"),
                "budget_rank": probe.get("budget_rank"),
                "within_current_budget": probe.get("within_current_budget"),
                "oracle_selection_status": probe.get("oracle_selection_status"),
                "executor_authorization_status": probe.get("executor_authorization_status"),
                "portband_action_authority": probe.get("portband_action_authority"),
                **_guards(),
            }
        )
    rows.sort(key=lambda row: (-float(row.get("score") or 0.0), str(row.get("frontier_item_id"))))
    return rows[:limit]


def _confidence_summary(packet: dict[str, Any] | None) -> dict[str, Any]:
    frontier = _frontier_from_packet(packet, limit=8)
    top_score = float(frontier[0].get("score") or 0.0) if frontier else 0.0
    mean_score = sum(float(row.get("score") or 0.0) for row in frontier) / float(len(frontier)) if frontier else 0.0
    return {
        "confidence_model": "frontier_score_summary_not_probability",
        "confidence_is_probability": False,
        "top_frontier_score": round(top_score, 6),
        "mean_frontier_score": round(mean_score, 6),
        "frontier_item_count": len(frontier),
        "top_frontier_item_id": frontier[0].get("frontier_item_id") if frontier else None,
    }


def _deltaband_summary(deltaband: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(deltaband, dict):
        return {
            "status": "not_observed",
            "relationship_change_count": 0,
            "raw_pixel_delta_count": None,
            "null_action_signal": None,
        }
    portband_feedback = (
        deltaband.get("portband_surface_feedback_summary")
        if isinstance(deltaband.get("portband_surface_feedback_summary"), dict)
        else {}
    )
    return {
        "status": deltaband.get("status"),
        "action_invocation_id": deltaband.get("action_invocation_id"),
        "action_primitive": deltaband.get("action_primitive"),
        "relationship_change_count": int(deltaband.get("relationship_change_count") or 0),
        "action_surface_result_event_count": int(deltaband.get("action_surface_result_event_count") or 0),
        "top_action_surface_result_event": (deltaband.get("action_surface_result_events") or [None])[0],
        "raw_pixel_delta_count": deltaband.get("raw_pixel_delta_count"),
        "component_delta_count": deltaband.get("component_delta_count"),
        "null_action_signal": deltaband.get("null_action_signal"),
        "changed_geoband_boundary_ids": deltaband.get("changed_geoband_boundary_ids", []),
        "fixed_geoband_boundary_ids": deltaband.get("fixed_geoband_boundary_ids", []),
        "portband_action_state_tag_count": int(portband_feedback.get("action_state_tag_count") or 0),
        "portband_action_state_positive_signal_count": int(portband_feedback.get("action_state_positive_signal_count") or 0),
        "portband_action_state_tagger_policy": portband_feedback.get("action_state_tagger_policy"),
        "portband_action_state_primitives": portband_feedback.get("action_state_primitives", []),
        "portband_relative_state_change_tag_count": int(portband_feedback.get("relative_state_change_tag_count") or 0),
        "portband_novel_relationship_candidate_count": int(portband_feedback.get("novel_relationship_candidate_count") or 0),
        "portband_solve_like_evidence_flag_count": int(portband_feedback.get("solve_like_evidence_flag_count") or 0),
        "portband_top_solve_like_evidence_score": portband_feedback.get("top_solve_like_evidence_score"),
        "portband_route_flow_lock_count": int(portband_feedback.get("route_flow_lock_count") or 0),
        "portband_route_flow_positive_signal_count": int(portband_feedback.get("route_flow_positive_signal_count") or 0),
        "portband_route_flow_lock_policy": portband_feedback.get("route_flow_lock_policy"),
    }


def _actionmodband_summary(actionmodband: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(actionmodband, dict):
        return {
            "status": "not_observed",
            "response_count": 0,
            "direct_motion_response_count": 0,
            "downstream_response_count": 0,
            "levels_completed_delta": None,
        }
    return {
        "schema": actionmodband.get("schema"),
        "status": actionmodband.get("status"),
        "actionmodband_id": actionmodband.get("actionmodband_id"),
        "action_invocation_id": actionmodband.get("action_invocation_id"),
        "response_count": int(actionmodband.get("response_count") or 0),
        "direct_motion_response_count": int(actionmodband.get("direct_motion_response_count") or 0),
        "blocked_motion_response_count": int(actionmodband.get("blocked_motion_response_count") or 0),
        "downstream_response_count": int(actionmodband.get("downstream_response_count") or 0),
        "levels_completed_delta": actionmodband.get("levels_completed_delta"),
        "top_downstream_response": (actionmodband.get("downstream_action_responses") or [None])[0],
    }


def _assumption_updates(before_packet: dict[str, Any] | None, after_packet: dict[str, Any] | None, deltaband: dict[str, Any] | None) -> list[dict[str, Any]]:
    before_notes = []
    tabula = before_packet.get("tabula_rasa_notebook") if isinstance(before_packet, dict) and isinstance(before_packet.get("tabula_rasa_notebook"), dict) else {}
    for note in tabula.get("notes", []) or []:
        if isinstance(note, dict):
            before_notes.append(note)
    changed = bool(isinstance(deltaband, dict) and (deltaband.get("null_action_signal") is False or int(deltaband.get("relationship_change_count") or 0) > 0))
    after_conf = _confidence_summary(after_packet)
    rows: list[dict[str, Any]] = []
    for note in before_notes[:12]:
        note_id = str(note.get("note_id") or "")
        if not note_id:
            continue
        before_conf = 0.25
        confidence_after = 0.35 if changed else 0.15
        rows.append(
            {
                "note_id": note_id,
                "item_kind": note.get("item_kind"),
                "item_id": note.get("item_id"),
                "pre_round_assumption": note.get("current_assumption"),
                "post_round_assumption": "State changed after action; keep this assumption active for the next round."
                if changed
                else "No meaningful state change observed; demote this assumption until new evidence appears.",
                "confidence_before": round(before_conf, 6),
                "confidence_after": round(confidence_after, 6),
                "confidence_basis": "deltaband_change_signal_plus_next_frontier_context_not_probability",
                "next_top_frontier_item_id": after_conf.get("top_frontier_item_id"),
                **_guards(),
            }
        )
    return rows


def build_initial_episode_memory(
    *,
    game_id: str,
    level_index: int,
    contra_packet: dict[str, Any],
) -> dict[str, Any]:
    current_hash = packet_hash(contra_packet)
    frontier = _frontier_from_packet(contra_packet)
    confidence = _confidence_summary(contra_packet)
    return {
        "schema": CONTRA_EPISODE_MEMORY_SCHEMA,
        "status": "initialized_no_action_yet",
        "episode_memory_id": _stable_id("episode_memory", game_id, level_index, current_hash),
        "game_id": str(game_id),
        "level_index": int(level_index),
        "round_index": 0,
        "last_contra_packet_hash": current_hash,
        "previous_contra_packet_hash": None,
        "action_taken": None,
        "deltaband_result": _deltaband_summary(None),
        "actionmodband_result": _actionmodband_summary(None),
        "updated_assumptions": [],
        "confidence_before": confidence,
        "confidence_after": confidence,
        "promoted_candidate_ids": [],
        "demoted_candidate_ids": [],
        "current_curiosity_action_frontier": frontier,
        "frontier_count": len(frontier),
        "carry_forward_policy": "persist_memory_between_rounds_and_update_after_controlled_action",
        **_guards(),
    }


def build_episode_memory_after_round(
    *,
    game_id: str,
    level_index: int,
    round_index: int,
    previous_memory: dict[str, Any] | None,
    before_contra_packet: dict[str, Any],
    after_contra_packet: dict[str, Any],
    action_taken: dict[str, Any],
    deltaband_result: dict[str, Any],
    actionmodband_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before_hash = packet_hash(before_contra_packet)
    after_hash = packet_hash(after_contra_packet)
    before_conf = _confidence_summary(before_contra_packet)
    after_conf = _confidence_summary(after_contra_packet)
    before_frontier = _frontier_from_packet(before_contra_packet)
    after_frontier = _frontier_from_packet(after_contra_packet)
    before_ids = {str(row.get("frontier_item_id")) for row in before_frontier}
    after_ids = {str(row.get("frontier_item_id")) for row in after_frontier}
    changed_ids = [str(value) for value in (deltaband_result.get("changed_geoband_boundary_ids", []) or [])]
    fixed_ids = [str(value) for value in (deltaband_result.get("fixed_geoband_boundary_ids", []) or [])]
    promoted = sorted((after_ids - before_ids) | set(changed_ids))
    demoted = sorted((before_ids - after_ids) | ({str(action_taken.get("primitive") or action_taken.get("action_name") or "action")} if deltaband_result.get("null_action_signal") is True else set()))
    return {
        "schema": CONTRA_EPISODE_MEMORY_SCHEMA,
        "status": "updated_after_controlled_round",
        "episode_memory_id": _stable_id("episode_memory", game_id, level_index, round_index, before_hash, after_hash),
        "game_id": str(game_id),
        "level_index": int(level_index),
        "round_index": int(round_index),
        "previous_episode_memory_id": previous_memory.get("episode_memory_id") if isinstance(previous_memory, dict) else None,
        "last_contra_packet_hash": after_hash,
        "previous_contra_packet_hash": before_hash,
        "action_taken": action_taken,
        "deltaband_result": _deltaband_summary(deltaband_result),
        "actionmodband_result": _actionmodband_summary(actionmodband_result),
        "updated_assumptions": _assumption_updates(before_contra_packet, after_contra_packet, deltaband_result),
        "confidence_before": before_conf,
        "confidence_after": after_conf,
        "confidence_delta": {
            "top_frontier_score_delta": round(float(after_conf.get("top_frontier_score") or 0.0) - float(before_conf.get("top_frontier_score") or 0.0), 6),
            "mean_frontier_score_delta": round(float(after_conf.get("mean_frontier_score") or 0.0) - float(before_conf.get("mean_frontier_score") or 0.0), 6),
        },
        "promoted_candidate_ids": promoted,
        "demoted_candidate_ids": demoted,
        "fixed_candidate_ids": fixed_ids,
        "current_curiosity_action_frontier": after_frontier,
        "frontier_count": len(after_frontier),
        "carry_forward_policy": "persist_memory_between_rounds_and_update_after_controlled_action",
        **_guards("controlled_action_episode_memory_not_truth"),
    }
