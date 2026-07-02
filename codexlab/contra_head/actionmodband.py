#!/usr/bin/env python3
"""ActionModBand: per-object action response signatures."""

from __future__ import annotations

import hashlib
from typing import Any


ACTIONMODBAND_SCHEMA = "codexlab.actionmodband.v1"
ACTIONMODBAND_RESPONSE_SCHEMA = "codexlab.actionmodband.object_action_response.v1"
ACTIONMODBAND_DOWNSTREAM_SCHEMA = "codexlab.actionmodband.downstream_action_response.v1"

TRUTH_FLAGS = {
    "prediction_status": "not_predictive",
    "admission_status": "not_admitted",
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}

MOVE_ACTION_NAMES = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}
SELECT_ACTION_NAMES = {"ACTION6"}
FLOW_ACTION_NAMES = {"ACTION5"}


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _action_name(action_taken: dict[str, Any] | None) -> str | None:
    if not isinstance(action_taken, dict):
        return None
    name = action_taken.get("name")
    if name is not None:
        return str(name)
    value = action_taken.get("action")
    if isinstance(value, dict) and value.get("name") is not None:
        return str(value.get("name"))
    return None


def _selected_ref(state: dict[str, Any] | None, *, phase: str) -> dict[str, Any] | None:
    if not isinstance(state, dict) or not state.get("selected"):
        return None
    coord = state.get("selected_coord")
    shape = state.get("selected_shape")
    if not (isinstance(coord, list) and len(coord) >= 2 and isinstance(shape, list) and len(shape) >= 2):
        return None
    name = str(state.get("selected_name") or "unknown")
    x = int(coord[0])
    y = int(coord[1])
    w = int(shape[0])
    h = int(shape[1])
    return {
        "object_ref_id": _stable_id("object_ref", name, x, y, w, h, phase),
        "object_signature_id": _stable_id("object_sig", name, w, h),
        "name": name,
        "coord_xy": [x, y],
        "shape_wh": [w, h],
        "bbox_xyxy": [x, y, x + max(0, w - 1), y + max(0, h - 1)],
        "display_coord_xy": state.get("selected_display_coord"),
    }


def _coord_delta(before_ref: dict[str, Any] | None, after_ref: dict[str, Any] | None) -> dict[str, Any]:
    if not before_ref or not after_ref:
        return {"dx": None, "dy": None, "manhattan": None, "axis": "unknown", "sign": "unknown"}
    bx, by = before_ref["coord_xy"]
    ax, ay = after_ref["coord_xy"]
    dx = int(ax - bx)
    dy = int(ay - by)
    if abs(dx) > abs(dy):
        axis = "x"
        sign = "positive" if dx > 0 else "negative" if dx < 0 else "zero"
    elif abs(dy) > 0:
        axis = "y"
        sign = "positive" if dy > 0 else "negative"
    else:
        axis = "none"
        sign = "zero"
    return {"dx": dx, "dy": dy, "manhattan": abs(dx) + abs(dy), "axis": axis, "sign": sign}


def _effect_kind(action_name: str | None, before_ref: dict[str, Any] | None, after_ref: dict[str, Any] | None, delta: dict[str, Any]) -> str:
    if action_name in FLOW_ACTION_NAMES:
        return "downstream_flow_evaluation"
    if action_name in SELECT_ACTION_NAMES:
        if before_ref and after_ref and before_ref.get("object_signature_id") != after_ref.get("object_signature_id"):
            return "selection_changed"
        return "selection_probe"
    if action_name in MOVE_ACTION_NAMES:
        return "direct_object_motion" if (delta.get("dx") or delta.get("dy")) else "blocked_or_null_motion"
    return "unknown_action_effect"


def build_object_action_response(
    *,
    before_state: dict[str, Any] | None,
    after_state: dict[str, Any] | None,
    action_taken: dict[str, Any],
    frame_delta: dict[str, Any] | None = None,
    returned_frame_count: int | None = None,
    action_invocation_id: str | None = None,
    action_index: int | None = None,
    downstream_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one controlled action response row.

    This is a measurement packet, not a prediction. It binds an action to the
    object selected at the action boundary and reports the direct response.
    """

    before_ref = _selected_ref(before_state, phase="before")
    after_ref = _selected_ref(after_state, phase="after")
    action_name = _action_name(action_taken)
    delta = _coord_delta(before_ref, after_ref)
    changed_pixels = None
    if isinstance(frame_delta, dict) and isinstance(frame_delta.get("changed_pixels"), int):
        changed_pixels = int(frame_delta["changed_pixels"])
    effect_kind = _effect_kind(action_name, before_ref, after_ref, delta)
    is_move_action = action_name in MOVE_ACTION_NAMES
    is_direct_motion = effect_kind == "direct_object_motion"
    object_id = (before_ref or after_ref or {}).get("object_signature_id") or "no_selected_object"
    response_id = _stable_id("action_response", action_invocation_id, action_index, action_name, object_id, delta.get("dx"), delta.get("dy"), changed_pixels)
    levels_before = before_state.get("levels_completed") if isinstance(before_state, dict) else None
    levels_after = after_state.get("levels_completed") if isinstance(after_state, dict) else None
    levels_delta = None
    if isinstance(levels_before, int) and isinstance(levels_after, int):
        levels_delta = int(levels_after - levels_before)

    return {
        "schema": ACTIONMODBAND_RESPONSE_SCHEMA,
        "response_id": response_id,
        "action_invocation_id": action_invocation_id,
        "action_index": action_index,
        "action": action_taken,
        "action_name": action_name,
        "selected_object_before": before_ref,
        "selected_object_after": after_ref,
        "selected_object_signature_id": object_id,
        "action_modifier_parameter": {
            "effect_kind": effect_kind,
            "scope": "direct_selected_object" if action_name not in FLOW_ACTION_NAMES else "downstream_evaluator",
            "delta_scope": "direct_selected_object_motion" if is_move_action else "selected_object_state_boundary",
            "dx": delta.get("dx"),
            "dy": delta.get("dy"),
            "axis": delta.get("axis"),
            "sign": delta.get("sign"),
            "manhattan_delta": delta.get("manhattan"),
            "legal_motion_observed": bool(is_direct_motion and delta.get("manhattan")),
            "selection_changed": bool(before_ref and after_ref and before_ref.get("object_signature_id") != after_ref.get("object_signature_id")),
            "rotation_quarters_before": before_state.get("rotation_quarters") if isinstance(before_state, dict) else None,
            "rotation_quarters_after": after_state.get("rotation_quarters") if isinstance(after_state, dict) else None,
            "returned_frame_count": returned_frame_count,
            "changed_pixels": changed_pixels,
        },
        "levels_completed_delta": levels_delta,
        "downstream_result": downstream_result or {},
        "correlation_status": "direct_controlled_boundary_observation"
        if action_name not in FLOW_ACTION_NAMES
        else "downstream_evaluator_observation",
        "evidence_status": "observed_object_action_response",
        **TRUTH_FLAGS,
    }


def _latest_direct_responses(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        effect_kind = (row.get("action_modifier_parameter") or {}).get("effect_kind")
        if effect_kind not in {"direct_object_motion", "blocked_or_null_motion", "selection_changed", "selection_probe"}:
            continue
        object_id = str(row.get("selected_object_signature_id") or "no_selected_object")
        latest[object_id] = row
    return latest


def build_downstream_action_response(
    *,
    flow_action_response: dict[str, Any],
    prior_action_responses: list[dict[str, Any]],
    levels_completed_delta: int | None = None,
) -> dict[str, Any]:
    latest = _latest_direct_responses(prior_action_responses)
    contributors = []
    for object_id, row in sorted(latest.items()):
        modifier = row.get("action_modifier_parameter") if isinstance(row.get("action_modifier_parameter"), dict) else {}
        contributors.append(
            {
                "object_signature_id": object_id,
                "last_response_id": row.get("response_id"),
                "last_action_name": row.get("action_name"),
                "last_effect_kind": modifier.get("effect_kind"),
                "last_direct_delta": {
                    "dx": modifier.get("dx"),
                    "dy": modifier.get("dy"),
                    "axis": modifier.get("axis"),
                    "sign": modifier.get("sign"),
                    "manhattan_delta": modifier.get("manhattan_delta"),
                },
                "temporal_relation_to_flow": "prior_selected_object_action_before_flow",
                "causal_status": "not_proven_by_temporal_correlation",
            }
        )
    outcome = "win_observed" if isinstance(levels_completed_delta, int) and levels_completed_delta > 0 else "no_win_observed"
    return {
        "schema": ACTIONMODBAND_DOWNSTREAM_SCHEMA,
        "downstream_response_id": _stable_id("downstream_response", flow_action_response.get("response_id"), levels_completed_delta, len(contributors)),
        "flow_action_response_id": flow_action_response.get("response_id"),
        "flow_action_name": flow_action_response.get("action_name"),
        "outcome": outcome,
        "levels_completed_delta": levels_completed_delta,
        "object_contributors": contributors,
        "object_contributor_count": len(contributors),
        "correlation_policy": "correlate_last_direct_object_responses_before_downstream_evaluator_without_promoting_causality",
        "evidence_status": "downstream_action_correlation_not_causal_truth",
        **TRUTH_FLAGS,
    }


def build_actionmodband(
    *,
    action_responses: list[dict[str, Any]],
    action_invocation_id: str | None = None,
    levels_completed_delta: int | None = None,
) -> dict[str, Any]:
    rows = [row for row in action_responses if isinstance(row, dict)]
    flow_rows = [row for row in rows if row.get("action_name") in FLOW_ACTION_NAMES]
    downstream_rows = [
        build_downstream_action_response(
            flow_action_response=flow,
            prior_action_responses=[row for row in rows if int(row.get("action_index") or 0) < int(flow.get("action_index") or 0)],
            levels_completed_delta=levels_completed_delta if levels_completed_delta is not None else flow.get("levels_completed_delta"),
        )
        for flow in flow_rows
    ]
    direct_motion_count = sum(1 for row in rows if (row.get("action_modifier_parameter") or {}).get("effect_kind") == "direct_object_motion")
    blocked_motion_count = sum(1 for row in rows if (row.get("action_modifier_parameter") or {}).get("effect_kind") == "blocked_or_null_motion")
    return {
        "schema": ACTIONMODBAND_SCHEMA,
        "status": "observation_set" if rows else "empty_no_action_responses",
        "actionmodband_id": _stable_id("actionmodband", action_invocation_id, len(rows), levels_completed_delta),
        "definition": "per_object_action_response_signature_and_downstream_action_correlation",
        "action_invocation_id": action_invocation_id,
        "response_count": len(rows),
        "direct_motion_response_count": direct_motion_count,
        "blocked_motion_response_count": blocked_motion_count,
        "downstream_response_count": len(downstream_rows),
        "levels_completed_delta": levels_completed_delta,
        "object_action_responses": rows,
        "downstream_action_responses": downstream_rows,
        "planning_use": "rank_future_actions_by_observed_object_specific_action_response_without_promoting_to_truth",
        "evidence_status": "controlled_action_response_observation",
        **TRUTH_FLAGS,
    }
