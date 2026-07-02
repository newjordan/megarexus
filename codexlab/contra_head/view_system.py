"""Unified CONTRA view-system pass reports."""

from __future__ import annotations

from typing import Any

from .controller_affordances import oracle_actor_affordance_hypotheses, oracle_target_affordance_hypotheses

CONTRA_CONFIG_PASS_SCHEMA = "codexlab.contra_config_pass.v1"
CONTRA_ORACLE_PASS_SCHEMA = "codexlab.contra_oracle_pass.v1"
CONTRA_PASS_SCHEMA = "codexlab.contra_pass.v1"
CONTRA_VIEW_SYSTEM_PASS_SET_SCHEMA = "codexlab.contra_view_system_pass_set.v1"


def _truth_boundary() -> dict[str, Any]:
    return {
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def _first_click_record(calibration: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(calibration, dict):
        return {}
    record = calibration.get("first_click_calibration")
    if isinstance(record, dict):
        return record
    return calibration


def _selected_click_xy(record: dict[str, Any], frame_shape: list[int] | None) -> list[int] | None:
    value = record.get("selected_click_xy")
    if isinstance(value, list) and len(value) >= 2:
        try:
            return [int(value[0]), int(value[1])]
        except (TypeError, ValueError):
            return None
    if frame_shape and len(frame_shape) >= 2:
        return [int(frame_shape[1]) // 2, int(frame_shape[0]) // 2]
    return None


def build_config_pass(
    *,
    game_id: str,
    level_index: int,
    mode: str,
    raw_frame_sha256: str | None,
    frame_shape: list[int] | None,
    calibration: dict[str, Any] | None,
    invalid_input: bool = False,
    first_loading_screen_detected: bool = False,
) -> dict[str, Any]:
    record = _first_click_record(calibration)
    accepted = record.get("accepted_screen_click")
    post_click_frame_ref = record.get("post_click_frame_ref")
    error_message_ref = record.get("error_message_ref")
    click_invocation_id = record.get("click_invocation_id") or record.get("init_click_invocation_id")
    first_level = int(level_index) == 0 and not invalid_input
    click_allowed_context = first_level and bool(first_loading_screen_detected)
    selected_click_xy = _selected_click_xy(record, frame_shape) if click_allowed_context else None
    if invalid_input:
        allowance_status = "not_allowed_invalid_raw_frame"
        evidence_status = "invalid_input_no_config_evidence"
        status = "not_available_invalid_input"
    elif accepted is True and click_allowed_context:
        allowance_status = "completed_accepted"
        evidence_status = "accepted_screen_click_post_frame_recorded" if post_click_frame_ref else "accepted_screen_click_missing_post_click_frame_ref"
        status = "config_evidence_recorded"
    elif accepted is False and click_allowed_context:
        allowance_status = "completed_rejected"
        evidence_status = "rejected_screen_click_error_recorded" if error_message_ref else "rejected_screen_click_missing_error_message_ref"
        status = "config_evidence_recorded"
    elif accepted in {True, False} and not click_allowed_context:
        allowance_status = "not_allowed_not_white_loading_screen" if first_level else "not_allowed_non_start_level"
        evidence_status = "click_result_rejected_wrong_config_context"
        status = "config_status_only"
    elif click_allowed_context:
        allowance_status = "allowance_issued"
        evidence_status = "first_white_loading_screen_click_allowance"
        status = "config_calibration_allowance"
    elif first_level:
        allowance_status = "not_allowed_not_white_loading_screen"
        evidence_status = "level_zero_frame_not_white_loading_screen_no_first_click_allowance"
        status = "config_status_only"
    else:
        allowance_status = "not_allowed_non_start_level"
        evidence_status = "non_start_level_no_first_click_allowance"
        status = "config_status_only"
    return {
        "schema": CONTRA_CONFIG_PASS_SCHEMA,
        "status": status,
        "source": "contra_view_system_config_pass",
        "game_id": str(game_id),
        "level_index": int(level_index),
        "mode": str(mode),
        "source_raw_frame_sha256": raw_frame_sha256,
        "first_click_allowance_status": allowance_status,
        "first_click_scope": "first_white_loading_screen_click_calibration" if click_allowed_context else "none",
        "first_loading_screen_detected": bool(first_loading_screen_detected),
        "first_loading_screen_detector": "raw_frame_all_pixels_equal_white_value_0",
        "selected_click_xy": selected_click_xy,
        "click_invocation_id": str(click_invocation_id) if click_invocation_id else None,
        "post_click_frame_ref": post_click_frame_ref if isinstance(post_click_frame_ref, (str, dict)) else None,
        "accepted_screen_click": accepted if accepted in {True, False} else None,
        "error_message_ref": error_message_ref if isinstance(error_message_ref, (str, dict)) else None,
        "config_evidence_status": evidence_status,
        "evidence_role": "approved_calibration_evidence_not_gameplay_truth",
        "allowance_count": 1 if allowance_status == "allowance_issued" else 0,
        **_truth_boundary(),
    }


def build_oracle_pass(
    *,
    level_index: int,
    raw_frame_sha256: str | None,
    scene_surfaces: dict[str, Any] | None,
    frame_shape: tuple[int, int] | None,
    label_map: Any | None,
    component_lookup: dict[str, int] | None,
    invalid_input: bool = False,
) -> dict[str, Any]:
    actor_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    reason_codes: list[str] = []
    if invalid_input or not isinstance(scene_surfaces, dict):
        reason_codes.append("oracle_inputs_unavailable")
    else:
        try:
            actor_rows = oracle_actor_affordance_hypotheses(scene_surfaces=scene_surfaces)
        except Exception as exc:  # noqa: BLE001 - report, do not hide pass.
            reason_codes.append(f"oracle_actor_exception:{type(exc).__name__}")
        if frame_shape is None or label_map is None or component_lookup is None:
            reason_codes.append("oracle_target_inputs_unavailable")
        else:
            try:
                target_rows = oracle_target_affordance_hypotheses(
                    frame_shape=frame_shape,
                    label_map=label_map,
                    component_lookup=component_lookup,
                    scene_surfaces=scene_surfaces,
                )
            except Exception as exc:  # noqa: BLE001
                reason_codes.append(f"oracle_target_exception:{type(exc).__name__}")
    hypothesis_count = len(actor_rows) + len(target_rows)
    if hypothesis_count:
        status = "review_hypotheses_emitted"
        evidence_status = "oracle_review_hypotheses_not_admitted"
    elif reason_codes:
        status = "no_hypothesis_with_reasons"
        evidence_status = "explicit_no_hypothesis_status"
    else:
        status = "no_hypothesis"
        evidence_status = "explicit_no_hypothesis_status"
    return {
        "schema": CONTRA_ORACLE_PASS_SCHEMA,
        "status": status,
        "source": "contra_view_system_oracle_pass",
        "level_index": int(level_index),
        "source_raw_frame_sha256": raw_frame_sha256,
        "input_refs": ["config_pass", "raw_color_component_packet", "scene_surfaces", "symmetry_hypotheses"],
        "hypothesis_count": hypothesis_count,
        "actor_hypothesis_count": len(actor_rows),
        "target_hypothesis_count": len(target_rows),
        "actor_review_hypotheses": actor_rows,
        "target_review_hypotheses": target_rows,
        "reason_codes": sorted(set(reason_codes)),
        "oracle_evidence_status": evidence_status,
        "evidence_role": "review_hypothesis_not_truth_source",
        **_truth_boundary(),
    }


def build_contra_pass(
    *,
    level_index: int,
    raw_frame_sha256: str | None,
    layer_output_packets: dict[str, Any],
    relationship_ladder: dict[str, Any],
    deltaband_gate: dict[str, Any],
    admission: dict[str, Any],
    config_pass: dict[str, Any],
    oracle_pass: dict[str, Any],
    ghost_pass: dict[str, Any] | None = None,
) -> dict[str, Any]:
    layer_outputs = layer_output_packets.get("outputs") if isinstance(layer_output_packets, dict) else []
    output_count = len(layer_outputs) if isinstance(layer_outputs, list) else 0
    return {
        "schema": CONTRA_PASS_SCHEMA,
        "status": "unified_analysis_report",
        "source": "contra_view_system_contra_pass",
        "level_index": int(level_index),
        "source_raw_frame_sha256": raw_frame_sha256,
        "input_report_refs": ["config_pass", "oracle_pass", "ghost_pass", "raw_color_component_packet", "scene_surfaces", "symmetry_hypotheses"],
        "layer_outputs_ref": "layer_output_packets",
        "layer_output_count": output_count,
        "computed_output_count": layer_output_packets.get("computed_output_count") if isinstance(layer_output_packets, dict) else None,
        "skipped_output_count": layer_output_packets.get("skipped_output_count") if isinstance(layer_output_packets, dict) else None,
        "relationship_ladder_ref": "contra_relationship_ladder",
        "relationship_ladder_status": relationship_ladder.get("status") if isinstance(relationship_ladder, dict) else None,
        "deltaband_gate_ref": "deltaband_gate",
        "deltaband_requirement_status": deltaband_gate.get("controlled_run_status") if isinstance(deltaband_gate, dict) else None,
        "admission_ref": "admission",
        "admission_status": admission.get("admission_status") if isinstance(admission, dict) else "not_admitted",
        "config_pass_status": config_pass.get("status"),
        "oracle_pass_status": oracle_pass.get("status"),
        "ghost_pass_status": ghost_pass.get("status") if isinstance(ghost_pass, dict) else "not_loaded",
        "pass_relationships": [
            {"from_pass": "config_pass", "to_pass": "oracle_pass", "relationship": "calibration_context_available_for_review"},
            {"from_pass": "config_pass", "to_pass": "contra_pass", "relationship": "config_report_preserved"},
            {"from_pass": "oracle_pass", "to_pass": "contra_pass", "relationship": "oracle_review_report_preserved"},
            {"from_pass": "ghost_pass", "to_pass": "contra_pass", "relationship": "grounded_relationship_memory_preserved"},
            {"from_pass": "contra_pass", "to_pass": "view_system_passes", "relationship": "unified_reporting_ledger"},
        ],
        "feedback_signals": {
            "config_evidence_status": config_pass.get("config_evidence_status"),
            "oracle_evidence_status": oracle_pass.get("oracle_evidence_status"),
            "ghost_status": ghost_pass.get("status") if isinstance(ghost_pass, dict) else "not_loaded",
            "ghost_relation_memory_row_count": ghost_pass.get("relation_memory_row_count") if isinstance(ghost_pass, dict) else 0,
            "fail_closed_reasons": admission.get("fail_closed_reasons", []) if isinstance(admission, dict) else [],
        },
        "review_page_ledger": {
            "required_pass_cards": ["config_pass", "oracle_pass", "ghost_pass", "contra_pass"],
            "internal_pass_cards_ref": "layer_output_packets.outputs",
        },
        "evidence_role": "unified_report_network_not_truth_source",
        **_truth_boundary(),
    }


def build_view_system_passes(
    *,
    config_pass: dict[str, Any],
    oracle_pass: dict[str, Any],
    ghost_pass: dict[str, Any],
    contra_pass: dict[str, Any],
) -> dict[str, Any]:
    passes = [
        {"pass_id": "config_pass", "schema": config_pass.get("schema"), "status": config_pass.get("status"), "report_ref": "config_pass"},
        {"pass_id": "oracle_pass", "schema": oracle_pass.get("schema"), "status": oracle_pass.get("status"), "report_ref": "oracle_pass"},
        {"pass_id": "ghost_pass", "schema": ghost_pass.get("schema"), "status": ghost_pass.get("status"), "report_ref": "ghost_pass"},
        {"pass_id": "contra_pass", "schema": contra_pass.get("schema"), "status": contra_pass.get("status"), "report_ref": "contra_pass"},
    ]
    return {
        "schema": CONTRA_VIEW_SYSTEM_PASS_SET_SCHEMA,
        "status": "complete",
        "source": "contra_view_system_pass_set",
        "pass_count": len(passes),
        "required_pass_ids": ["config_pass", "oracle_pass", "ghost_pass", "contra_pass"],
        "passes": passes,
        **_truth_boundary(),
    }
