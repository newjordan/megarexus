#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from .deltaband import DELTABAND_REQUIRED_BOUNDARY
from .exploreband import build_exploreband_budget
from .match_calibration import (
    build_action_result_transfer_prior,
    build_like_control_group_prior,
    build_macroband_object_ranking,
    build_stage0_init_calibration_requirement,
    build_stage1_init_action_choice_requirement,
)

SP80_MANUAL_CALIBRATION_REVIEW_SCHEMA = "codexlab.sp80_manual_calibration_review.v1"
DELTABAND_CONTROLLED_EVENT_REQUIREMENT_SCHEMA = "codexlab.deltaband_controlled_event_requirement.v1"

MANUAL_CALIBRATION_FIELDS = [
    "actor_object_ids",
    "target_object_ids",
    "known_action_support_object_ids",
    "connector_object_ids",
    "known_y_object_ids",
    "unresolved_object_ids",
    "image_hash",
]

DELTABAND_REQUIRED_ARTIFACTS = [
    "before_frame",
    "action_invocation_log",
    "after_frame_sequence",
    "pre_action_contra_packet",
    "post_action_contra_packet",
    "changed_geoband_boundary_ids",
    "fixed_geoband_boundary_ids",
    "null_action_events",
]


def _empty_reviewed_expectations() -> dict[str, Any]:
    return {
        "actor_object_ids": [],
        "target_object_ids": [],
        "known_action_support_object_ids": [],
        "connector_object_ids": [],
        "known_y_object_ids": [],
        "unresolved_object_ids": [],
        "image_hash": None,
    }


def _scene_expectations(calibration: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(calibration, dict):
        return {}
    expectations = calibration.get("scene_surface_expectations")
    return expectations if isinstance(expectations, dict) else {}


def _component_ids(row: dict[str, Any] | None) -> list[str]:
    if not isinstance(row, dict):
        return []
    values = row.get("component_ids")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value is not None]


def _actor_component_ids(actor_region: dict[str, Any] | None) -> list[str]:
    if not isinstance(actor_region, dict):
        return []
    ids: list[str] = []
    for row in actor_region.get("component_groups", []):
        if not isinstance(row, dict):
            continue
        actor_id = row.get("actor_component_id")
        key_id = row.get("orientation_key_component_id")
        if actor_id is not None:
            ids.append(str(actor_id))
        if key_id is not None:
            ids.append(str(key_id))
    return ids


def _support_component_ids(support_region: dict[str, Any] | None) -> list[str]:
    if not isinstance(support_region, dict):
        return []
    return _component_ids(support_region)


def _support_rows_by_category(support_affordances: list[dict[str, Any]], category: str) -> list[dict[str, Any]]:
    return [
        row
        for row in support_affordances
        if isinstance(row, dict) and row.get("special_object_category") == category
    ]


def _support_ids(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("support_id")) for row in rows if row.get("support_id") is not None]


def _support_component_ids_from_rows(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("source_component_id")) for row in rows if row.get("source_component_id") is not None]


def build_sp80_manual_calibration_review(
    *,
    level_index: int,
    source_raw_frame_sha256: str,
    source_raw_grid_sha256: str,
    calibration: dict[str, Any] | None,
    support_affordances: list[dict[str, Any]],
) -> dict[str, Any]:
    expectations = _scene_expectations(calibration)
    actor_region = expectations.get("actor_region") if isinstance(expectations.get("actor_region"), dict) else {}
    target_region = expectations.get("bucket_or_goal_region") if isinstance(expectations.get("bucket_or_goal_region"), dict) else {}
    support_region = expectations.get("state_condition_support_regions") if isinstance(expectations.get("state_condition_support_regions"), dict) else {}
    unresolved_region = expectations.get("ignore_or_residual") if isinstance(expectations.get("ignore_or_residual"), dict) else {}
    connector_rows = _support_rows_by_category(support_affordances, "connector_candidate")
    known_y_rows = _support_rows_by_category(support_affordances, "known_y_unknown_attached")
    return {
        "schema": SP80_MANUAL_CALIBRATION_REVIEW_SCHEMA,
        "status": "requires_human_review",
        "review_status": "not_reviewed",
        "review_policy": "human_review_required_before_truth_admission",
        "level_index": int(level_index),
        "source_raw_frame_sha256": str(source_raw_frame_sha256),
        "source_raw_grid_sha256": str(source_raw_grid_sha256),
        "required_human_review_fields": list(MANUAL_CALIBRATION_FIELDS),
        "fields_pending_human_review": list(MANUAL_CALIBRATION_FIELDS),
        "seed_candidate_status": "agent_seed_or_user_supplied_hypothesis_for_review_only",
        "seed_candidate_ids": {
            "actor_object_ids": _actor_component_ids(actor_region),
            "target_object_ids": _component_ids(target_region),
            "known_action_support_object_ids": _support_component_ids(support_region),
            "connector_object_ids": _support_ids(connector_rows),
            "connector_component_ids": _support_component_ids_from_rows(connector_rows),
            "known_y_object_ids": _support_ids(known_y_rows),
            "known_y_component_ids": _support_component_ids_from_rows(known_y_rows),
            "unresolved_object_ids": _component_ids(unresolved_region),
            "image_hash": str(source_raw_frame_sha256),
        },
        "human_reviewed_expectations": _empty_reviewed_expectations(),
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_deltaband_controlled_event_requirement(
    *,
    level_index: int,
    source_raw_frame_sha256: str,
    relationship_ladder: dict[str, Any],
    raw_color_component_packet: dict[str, Any] | None = None,
    scene_surfaces: dict[str, Any] | None = None,
    actor_affordances: list[dict[str, Any]] | None = None,
    target_affordances: list[dict[str, Any]] | None = None,
    support_affordances: list[dict[str, Any]] | None = None,
    statecons: list[dict[str, Any]] | None = None,
    symmetry_hypotheses: dict[str, Any] | None = None,
    prior_observations: list[dict[str, Any]] | None = None,
    exploration_budget_config: dict[str, Any] | None = None,
    exploration_budget_ledger: Any = None,
) -> dict[str, Any]:
    ladder_id = relationship_ladder.get("ladder_id") if isinstance(relationship_ladder, dict) else None
    action_result_transfer_prior = build_action_result_transfer_prior(
        level_index=level_index,
        raw_color_component_packet=raw_color_component_packet,
        prior_observations=prior_observations,
    )
    like_control_group_prior = build_like_control_group_prior(
        raw_color_component_packet=raw_color_component_packet,
        scene_surfaces=scene_surfaces,
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        support_affordances=support_affordances,
        statecons=statecons,
    )
    macroband_object_ranking = build_macroband_object_ranking(
        raw_color_component_packet=raw_color_component_packet,
        scene_surfaces=scene_surfaces,
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        support_affordances=support_affordances,
        statecons=statecons,
        symmetry_hypotheses=symmetry_hypotheses,
        action_result_transfer_prior=action_result_transfer_prior,
        like_control_group_prior=like_control_group_prior,
    )
    stage0_requirement = build_stage0_init_calibration_requirement(
        level_index=level_index,
        source_raw_frame_sha256=source_raw_frame_sha256,
        relationship_ladder=relationship_ladder,
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
    stage1_requirement = build_stage1_init_action_choice_requirement(
        level_index=level_index,
        source_raw_frame_sha256=source_raw_frame_sha256,
        relationship_ladder=relationship_ladder,
    )
    exploreband_budget = build_exploreband_budget(
        level_index=level_index,
        source_raw_frame_sha256=source_raw_frame_sha256,
        macroband_object_ranking=macroband_object_ranking,
        stage0_init_calibration_requirement=stage0_requirement,
        previous_ledger=exploration_budget_ledger,
        budget_config=exploration_budget_config,
    )
    match_calibration_requirements = [stage0_requirement, stage1_requirement]
    return {
        "schema": DELTABAND_CONTROLLED_EVENT_REQUIREMENT_SCHEMA,
        "status": "not_run_requires_controlled_before_after_action_boundary",
        "controlled_run_status": "not_run",
        "match_calibration_sequence_schema": "codexlab.match_calibration.initialization_sequence_requirement.v1",
        "action_result_transfer_prior": action_result_transfer_prior,
        "action_result_transfer_prior_status": action_result_transfer_prior.get("status"),
        "like_control_group_prior": like_control_group_prior,
        "like_control_group_prior_status": like_control_group_prior.get("status"),
        "macroband_object_ranking": macroband_object_ranking,
        "macroband_object_ranking_status": macroband_object_ranking.get("status"),
        "exploreband_budget": exploreband_budget,
        "exploreband_budget_status": exploreband_budget.get("status"),
        "exploreband_remaining_probe_count": (exploreband_budget.get("budget_policy") or {}).get("remaining_probe_count"),
        "exploreband_candidate_probe_count": exploreband_budget.get("candidate_probe_count"),
        "match_calibration_requirements": match_calibration_requirements,
        "match_calibration_required_stage_ids": [
            str(row.get("stage_id"))
            for row in match_calibration_requirements
            if isinstance(row, dict) and row.get("stage_id")
        ],
        "stage0_calibration_required": True,
        "stage0_calibration_status": stage0_requirement.get("status"),
        "stage0_calibration_run_status": stage0_requirement.get("calibration_run_status"),
        "stage0_init_calibration_requirement": stage0_requirement,
        "stage1_calibration_required": True,
        "stage1_calibration_status": stage1_requirement.get("status"),
        "stage1_calibration_run_status": stage1_requirement.get("calibration_run_status"),
        "stage1_init_action_choice_requirement": stage1_requirement,
        "required_boundary": DELTABAND_REQUIRED_BOUNDARY,
        "level_index": int(level_index),
        "source_raw_frame_sha256": str(source_raw_frame_sha256),
        "pre_action_ladder_id": ladder_id,
        "required_artifacts": list(DELTABAND_REQUIRED_ARTIFACTS),
        "present_artifacts": [],
        "missing_artifacts": list(DELTABAND_REQUIRED_ARTIFACTS),
        "action_invocation_id": None,
        "before_frame_path": None,
        "after_frame_sequence_paths": [],
        "changed_geoband_boundary_ids": [],
        "fixed_geoband_boundary_ids": [],
        "null_action_events": [],
        "prediction_status": "not_predictive",
        "evidence_status": "not_observed",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
