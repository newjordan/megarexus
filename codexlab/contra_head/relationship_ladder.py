#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from typing import Any

CONTRA_RELATIONSHIP_LADDER_SCHEMA = "codexlab.contra_relationship_ladder.v1"
CONTRA_RELATIONSHIP_LADDER_STAGE_SCHEMA = "codexlab.contra_relationship_ladder_stage.v1"


def stable_ladder_id(*, level_index: int, source_raw_frame_sha256: str) -> str:
    digest = hashlib.sha1(f"{level_index}|{source_raw_frame_sha256}".encode("utf-8")).hexdigest()[:10]
    return f"contra_relationship_ladder_l{level_index:02d}_{digest}"


def ladder_stage(
    *,
    stage_name: str,
    input_stage_ids: list[str],
    required_inputs: list[str],
    present_inputs: list[str],
    completeness_status: str | None = None,
    stage_status: str = "hypothesis",
) -> dict[str, Any]:
    missing_inputs = [value for value in required_inputs if value not in set(present_inputs)]
    status = completeness_status
    if status is None:
        status = "complete" if not missing_inputs else "incomplete_missing_" + "_and_".join(missing_inputs)
    return {
        "schema": CONTRA_RELATIONSHIP_LADDER_STAGE_SCHEMA,
        "stage_id": stage_name,
        "stage_name": stage_name,
        "stage_status": stage_status,
        "input_stage_ids": input_stage_ids,
        "required_inputs": required_inputs,
        "present_inputs": present_inputs,
        "missing_inputs": missing_inputs,
        "completeness_status": status,
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def geoband_completeness_from_inputs(*, has_actor: bool, has_goal: bool, has_support: bool) -> str:
    missing = []
    if not has_actor:
        missing.append("actor_attachment")
    if not has_goal:
        missing.append("goal_attachment")
    if not has_support:
        missing.append("support_action_attachment")
    if missing:
        return "incomplete_missing_" + "_and_".join(missing)
    return "complete"


def build_contra_relationship_ladder(
    *,
    level_index: int,
    source_raw_frame_sha256: str,
    raw_color_inventory: dict[str, Any],
    scene_surfaces: dict[str, Any],
    symmetry_hypotheses: dict[str, Any] | None,
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    support_affordances: list[dict[str, Any]],
    statecons: list[dict[str, Any]],
    deltaband_observations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    color_count = len(raw_color_inventory.get("colors", [])) if isinstance(raw_color_inventory, dict) else 0
    has_components = bool(scene_surfaces.get("component_graph", {}).get("components")) if isinstance(scene_surfaces, dict) else False
    has_symmetry_packet = isinstance(symmetry_hypotheses, dict) and symmetry_hypotheses.get("schema") == "codexlab.contra_symmetry_hypotheses.v1"
    has_scene_surfaces = bool(
        isinstance(scene_surfaces, dict)
        and (
            scene_surfaces.get("floor_hypothesis") is not None
            or scene_surfaces.get("wall_hypotheses")
            or scene_surfaces.get("background_hypotheses")
            or scene_surfaces.get("residual_component_hypotheses")
        )
    )
    has_actor = bool(actor_affordances)
    has_goal = bool(target_affordances)
    has_support = bool(support_affordances)
    has_statecon = bool(statecons)
    has_geoband = any(isinstance(row.get("geoband_hypothesis"), dict) for row in statecons if isinstance(row, dict))
    geoband_status = geoband_completeness_from_inputs(has_actor=has_actor, has_goal=has_goal, has_support=has_support)
    deltaband_present = isinstance(deltaband_observations, dict)
    deltaband_complete = bool(
        deltaband_present
        and deltaband_observations.get("pre_action_ladder_id")
        and deltaband_observations.get("post_action_ladder_id")
        and deltaband_observations.get("action_invocation_id")
        and deltaband_observations.get("before_raw_frame_sha256")
        and deltaband_observations.get("after_raw_frame_sha256")
    )
    stages = [
        ladder_stage(
            stage_name="raw_color_components",
            input_stage_ids=[],
            required_inputs=["raw_color_inventory", "component_graph"],
            present_inputs=[
                *("raw_color_inventory" for _ in [None] if color_count > 0),
                *("component_graph" for _ in [None] if has_components),
            ],
        ),
        ladder_stage(
            stage_name="scene_surface_hypotheses",
            input_stage_ids=["raw_color_components"],
            required_inputs=["raw_color_components"],
            present_inputs=["raw_color_components"] if has_scene_surfaces else [],
        ),
        ladder_stage(
            stage_name="symmetry_hypotheses",
            input_stage_ids=["raw_color_components", "scene_surface_hypotheses"],
            required_inputs=["raw_color_components", "scene_surface_hypotheses", "symmetry_analysis"],
            present_inputs=[
                "raw_color_components",
                *("scene_surface_hypotheses" for _ in [None] if has_scene_surfaces),
                *("symmetry_analysis" for _ in [None] if has_symmetry_packet),
            ],
        ),
        ladder_stage(
            stage_name="actor_affordances",
            input_stage_ids=["raw_color_components", "scene_surface_hypotheses", "symmetry_hypotheses"],
            required_inputs=["raw_color_components", "scene_surface_hypotheses", "actor_attachment"],
            present_inputs=[
                "raw_color_components",
                "scene_surface_hypotheses",
                *("actor_attachment" for _ in [None] if has_actor),
            ],
        ),
        ladder_stage(
            stage_name="target_affordances",
            input_stage_ids=["raw_color_components", "scene_surface_hypotheses"],
            required_inputs=["raw_color_components", "scene_surface_hypotheses", "goal_attachment"],
            present_inputs=[
                "raw_color_components",
                "scene_surface_hypotheses",
                *("goal_attachment" for _ in [None] if has_goal),
            ],
        ),
        ladder_stage(
            stage_name="support_action_affordances",
            input_stage_ids=["raw_color_components", "scene_surface_hypotheses"],
            required_inputs=["raw_color_components", "scene_surface_hypotheses", "support_action_attachment"],
            present_inputs=[
                "raw_color_components",
                "scene_surface_hypotheses",
                *("support_action_attachment" for _ in [None] if has_support),
            ],
        ),
        ladder_stage(
            stage_name="state_condition_hypotheses",
            input_stage_ids=["actor_affordances", "target_affordances", "support_action_affordances"],
            required_inputs=["actor_attachment", "goal_attachment", "support_action_attachment"],
            present_inputs=[
                *("actor_attachment" for _ in [None] if has_actor),
                *("goal_attachment" for _ in [None] if has_goal),
                *("support_action_attachment" for _ in [None] if has_support),
            ],
            completeness_status="complete" if has_statecon and geoband_status == "complete" else geoband_status,
        ),
        ladder_stage(
            stage_name="geoband_hypotheses",
            input_stage_ids=["state_condition_hypotheses"],
            required_inputs=["actor_attachment", "goal_attachment", "support_action_attachment"],
            present_inputs=[
                *("actor_attachment" for _ in [None] if has_actor),
                *("goal_attachment" for _ in [None] if has_goal),
                *("support_action_attachment" for _ in [None] if has_support),
            ],
            completeness_status="complete" if has_geoband and geoband_status == "complete" else geoband_status,
        ),
        ladder_stage(
            stage_name="deltaband_observations",
            input_stage_ids=["pre_action_ladder", "action_invocation", "post_action_ladder"],
            required_inputs=[
                "pre_action_ladder",
                "action_invocation",
                "post_action_ladder",
                "before_raw_frame_sha256",
                "after_raw_frame_sha256",
            ],
            present_inputs=[
                *("pre_action_ladder" for _ in [None] if deltaband_present and deltaband_observations.get("pre_action_ladder_id")),
                *("action_invocation" for _ in [None] if deltaband_present and deltaband_observations.get("action_invocation_id")),
                *("post_action_ladder" for _ in [None] if deltaband_present and deltaband_observations.get("post_action_ladder_id")),
                *("before_raw_frame_sha256" for _ in [None] if deltaband_present and deltaband_observations.get("before_raw_frame_sha256")),
                *("after_raw_frame_sha256" for _ in [None] if deltaband_present and deltaband_observations.get("after_raw_frame_sha256")),
            ],
            completeness_status="complete" if deltaband_complete else "not_run_requires_controlled_before_after_action_boundary",
            stage_status="not_run" if not deltaband_present else "observation",
        ),
    ]
    incomplete = [stage["stage_id"] for stage in stages if stage["completeness_status"] != "complete"]
    return {
        "schema": CONTRA_RELATIONSHIP_LADDER_SCHEMA,
        "ladder_id": stable_ladder_id(level_index=level_index, source_raw_frame_sha256=source_raw_frame_sha256),
        "level_index": level_index,
        "source_raw_frame_sha256": source_raw_frame_sha256,
        "definition": "fail_closed_contra_relationship_admission_ladder",
        "stage_count": len(stages),
        "stages": stages,
        "incomplete_stage_ids": incomplete,
        "completeness_status": "complete" if not incomplete else "incomplete",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
