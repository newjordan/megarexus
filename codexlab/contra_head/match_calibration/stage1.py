#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from .common import admission_flags, ladder_id, missing_artifact_fields
from .stage0 import MATCH_CALIBRATION_STAGE0_ID

MATCH_CALIBRATION_STAGE1_REQUIREMENT_SCHEMA = "codexlab.match_calibration.stage1_init_action_choice_requirement.v1"
MATCH_CALIBRATION_STAGE1_ID = "deltaband_stage_1_init_action_choice"
MATCH_CALIBRATION_STAGE1_EVENT_KIND = "init_select_or_dpad_probe"
MATCH_CALIBRATION_STAGE1_REQUIRED_ARTIFACTS = [
    "post_stage0_baseline_frame",
    "initialization_choice_tree_log",
    "select_option_probe_invocation_log",
    "select_option_probe_delta_classification",
    "dpad_move_probe_invocation_log_if_select_unavailable",
    "dpad_move_probe_delta_classification_if_select_unavailable",
    "post_choice_baseline_frame",
]


def build_stage1_init_action_choice_requirement(
    *,
    level_index: int,
    source_raw_frame_sha256: str,
    relationship_ladder: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema": MATCH_CALIBRATION_STAGE1_REQUIREMENT_SCHEMA,
        "stage_id": MATCH_CALIBRATION_STAGE1_ID,
        "status": "not_run_requires_select_or_dpad_init_choice_boundary",
        "calibration_run_status": "not_run",
        "event_kind": MATCH_CALIBRATION_STAGE1_EVENT_KIND,
        "intent": "discover_post_canvas_initialization_input",
        "depends_on_stage_id": MATCH_CALIBRATION_STAGE0_ID,
        "choice_policy": "select_option_tree_first_else_dpad_move",
        "choice_status": "undefined_until_controlled_probe",
        "ordered_choice_tree": [
            {
                "priority": 1,
                "branch_id": "init_select_option_tree",
                "input_family": "select_or_options",
                "action_definition_status": "undefined",
                "candidate_action_policy": "undefined_select_option_tree",
                "fallback_condition": "no_select_path_detected",
            },
            {
                "priority": 2,
                "branch_id": "init_dpad_move_tree",
                "input_family": "dpad_move",
                "action_definition_status": "undefined",
                "candidate_action_policy": "undefined_dpad_move",
                "fallback_condition": "select_path_not_available",
            },
        ],
        "level_index": int(level_index),
        "source_raw_frame_sha256": str(source_raw_frame_sha256),
        "pre_choice_ladder_id": ladder_id(relationship_ladder),
        "required_before_deltaband": True,
        **missing_artifact_fields(MATCH_CALIBRATION_STAGE1_REQUIRED_ARTIFACTS),
        "selected_branch_id": None,
        "select_probe_invocation_id": None,
        "dpad_probe_invocation_id": None,
        "post_choice_baseline_frame_path": None,
        "delta_classification": None,
        "prediction_status": "not_predictive",
        "evidence_status": "not_observed",
        **admission_flags(),
    }
