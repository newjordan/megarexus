#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from typing import Any

GEOBAND_BOUNDARY_PASS_SCHEMA = "codexlab.geoband_boundary_pass.v1"
GEOBAND_BOUNDARY_QUEUE_SCHEMA = "codexlab.geoband_boundary_queue.v1"
GEOBAND_UNRESOLVED_BOUNDARY_SCHEMA = "codexlab.geoband_unresolved_boundary.v1"

BOUNDARY_TEST_TYPES = {
    "shape_equivalence_unknown",
    "shape_plus_y_attachment",
    "group_asymmetry_unresolved",
    "independent_vs_compound_unknown",
    "actor_target_support_role_unknown",
    "geometry_relationship_unresolved",
}


def stable_boundary_id(statecon_id: str, boundary_type: str, object_ids: list[str]) -> str:
    digest = hashlib.sha1(f"{statecon_id}|{boundary_type}|{'|'.join(sorted(object_ids))}".encode("utf-8")).hexdigest()[:10]
    return f"boundary_{boundary_type}_{digest}"


def unresolved_boundary(
    *,
    statecon_id: str,
    boundary_type: str,
    input_object_ids: list[str],
    reason_codes: list[str],
    suggested_next_pass: str,
    related_object_ids: list[str] | None = None,
) -> dict[str, Any]:
    object_ids = [str(value) for value in input_object_ids if value is not None]
    return {
        "schema": GEOBAND_UNRESOLVED_BOUNDARY_SCHEMA,
        "boundary_id": stable_boundary_id(statecon_id, boundary_type, object_ids),
        "boundary_type": boundary_type,
        "status": "unresolved",
        "input_object_ids": object_ids,
        "related_object_ids": [str(value) for value in (related_object_ids or []) if value is not None],
        "reason_codes": [str(value) for value in reason_codes],
        "suggested_next_pass": suggested_next_pass,
        "boundary_test_not_question_prompt": True,
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_boundary_queue(
    *,
    statecon_id: str,
    all_supports: list[dict[str, Any]],
    subgroup_supports: list[dict[str, Any]],
) -> dict[str, Any]:
    subgroup_ids = {str(row.get("support_id")) for row in subgroup_supports if isinstance(row, dict)}
    boundaries: list[dict[str, Any]] = []
    for support in all_supports:
        if not isinstance(support, dict):
            continue
        support_id = str(support.get("support_id"))
        if support.get("special_object_category") == "known_y_unknown_attached":
            boundaries.append(
                unresolved_boundary(
                    statecon_id=statecon_id,
                    boundary_type="shape_plus_y_attachment",
                    input_object_ids=[support_id],
                    related_object_ids=support.get("adjacent_support_candidate_component_ids") or [],
                    reason_codes=[
                        "known_shape_with_unresolved_extra_attachment",
                        str(support.get("unknown_attachment_basis") or "unknown_attachment_basis_unset"),
                    ],
                    suggested_next_pass="resolve_shape_plus_y_attachment",
                )
            )
        if support_id not in subgroup_ids:
            boundaries.append(
                unresolved_boundary(
                    statecon_id=statecon_id,
                    boundary_type="geometry_relationship_unresolved",
                    input_object_ids=[support_id],
                    reason_codes=["present_in_primary_geoband_but_not_in_refined_subgroup"],
                    suggested_next_pass="recheck_geometry_or_mark_independent",
                )
            )
        if support.get("evidence_status") == "heuristic_hypothesis":
            boundaries.append(
                unresolved_boundary(
                    statecon_id=statecon_id,
                    boundary_type="actor_target_support_role_unknown",
                    input_object_ids=[support_id],
                    reason_codes=["support_role_is_heuristic_not_reviewed"],
                    suggested_next_pass="manual_role_review_or_controlled_delta",
                )
            )
    if len(all_supports) >= 2:
        boundaries.append(
            unresolved_boundary(
                statecon_id=statecon_id,
                boundary_type="group_asymmetry_unresolved",
                input_object_ids=[str(row.get("support_id")) for row in all_supports if isinstance(row, dict)],
                reason_codes=["multi_object_group_asymmetry_not_resolved"],
                suggested_next_pass="resolve_internal_vs_group_asymmetry",
            )
        )
        boundaries.append(
            unresolved_boundary(
                statecon_id=statecon_id,
                boundary_type="independent_vs_compound_unknown",
                input_object_ids=[str(row.get("support_id")) for row in all_supports if isinstance(row, dict)],
                reason_codes=["compound_vs_independent_support_objects_not_resolved"],
                suggested_next_pass="resolve_independent_vs_compound_objects",
            )
        )
    return {
        "schema": GEOBAND_BOUNDARY_QUEUE_SCHEMA,
        "status": "hypothesis",
        "statecon_id": statecon_id,
        "boundary_count": len(boundaries),
        "unresolved_boundary_count": len(boundaries),
        "boundaries": boundaries,
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def geoband_pass(
    *,
    pass_id: str,
    pass_order: int,
    policy: str,
    input_object_ids: list[str],
    output_group_ids: list[str],
    unresolved_boundary_ids: list[str],
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "schema": GEOBAND_BOUNDARY_PASS_SCHEMA,
        "pass_id": pass_id,
        "pass_order": int(pass_order),
        "status": "hypothesis",
        "policy": policy,
        "input_object_ids": [str(value) for value in input_object_ids],
        "output_group_ids": [str(value) for value in output_group_ids],
        "unresolved_boundary_ids": [str(value) for value in unresolved_boundary_ids],
        "reason_codes": [str(value) for value in reason_codes],
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_geoband_boundary_passes(
    *,
    primary_supports: list[dict[str, Any]],
    subgroup_supports: list[dict[str, Any]],
    primary_groups: list[dict[str, Any]],
    subgroups: list[dict[str, Any]],
    boundary_queue: dict[str, Any],
) -> list[dict[str, Any]]:
    primary_ids = [str(row.get("support_id")) for row in primary_supports if isinstance(row, dict)]
    subgroup_ids = [str(row.get("support_id")) for row in subgroup_supports if isinstance(row, dict)]
    boundary_ids = [
        str(row.get("boundary_id"))
        for row in boundary_queue.get("boundaries", [])
        if isinstance(row, dict)
    ]
    return [
        geoband_pass(
            pass_id="geoband_primary_relationship_group",
            pass_order=1,
            policy="include_all_visible_relationship_objects",
            input_object_ids=primary_ids,
            output_group_ids=[str(row.get("support_group_id")) for row in primary_groups if isinstance(row, dict)],
            unresolved_boundary_ids=boundary_ids,
            reason_codes=["primary_pass_must_not_drop_visible_relationship_objects"],
        ),
        geoband_pass(
            pass_id="geoband_subgroup_refinement",
            pass_order=2,
            policy="resolve_boundary_tests_into_local_subgroups",
            input_object_ids=primary_ids,
            output_group_ids=[str(row.get("support_group_id")) for row in subgroups if isinstance(row, dict)],
            unresolved_boundary_ids=boundary_ids,
            reason_codes=[
                "subgroup_pass_may_refine_but_not_erase_primary_objects",
                f"subgroup_member_count_{len(subgroup_ids)}",
            ],
        ),
    ]
