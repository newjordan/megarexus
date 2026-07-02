"""Unified paint-control backend skill for CONTRA/SP80.

The skill is deliberately one BrainBackend-sized unit: it builds the L1
paint-control observation, runs one policy step, translates the native action
id to an existing SP80 primitive, and returns an auditable BrainDecision.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from contra_head.crucible.model import DEFAULT_DISCREPANCY_LEDGER_PATH, CrucibleModel
from players.base import BrainDecision
from sp80_base_controller import ROOT, sha256_file
from sp80_puffer_policy import inspect_pufferlib


DECISION_SCHEMA = "codexlab.paint_control_skill_decision.v1"
JSON_POLICY_SCHEMA = "codexlab.paint_control_skill_policy.v1"
DEFAULT_POLICY_ENV = "rexus_sp80_paintflow_l1_multi"
OBSERVATION_SIZE = 32
GRID_W = 14
GRID_H = 15
SP80_DISPLAY_W = 64
SP80_DISPLAY_H = 64
TARGET_ROWS = [4, 8, 12]
DEFAULT_TARGET_X = 6
REPEATED_STATIONARY_MOVE_FAIL_THRESHOLD = 2
RANKED_FALLBACK_REASONS = {
    "repeated_stationary_move_feedback",
    "fallback_repeated_movement_oscillation",
    "fallback_negative_distance_feedback",
    "fallback_stationary_move_feedback",
    "selection_lock_preserve_positive_source",
    "activation_readiness_unknown",
    "activation_readiness_target_misaligned",
    "activation_prior_no_progress_feedback",
    "forward_model_predicted_blocked_or_away",
    "selection_outcome_harmful_history",
    "activation_outcome_harmful_history",
    "selection_budget_infeasible",
}
RANKED_FALLBACK_EXHAUSTED_REASON = "ranked_fallback_exhausted"
INVERSE_MOVES = {
    "move_left": "move_right",
    "move_right": "move_left",
    "move_up": "move_down",
    "move_down": "move_up",
}
DEFAULT_PUFFERLIB_PATHS = [
    ROOT / ".external" / "pufferlib-4.0-newjordan",
    Path("/tmp/pufferlib-4.0-newjordan"),
]
DEFAULT_OUTCOME_LEDGER_PATH = ROOT / "codexlab" / "datasets" / "sp80_paint_skill_outcomes_v0" / "ledger.json"
OUTCOME_LEDGER_SCHEMA = "codexlab.paint_control_skill_outcome_ledger.v1"


@dataclass(frozen=True)
class NativeAction:
    action_id: int
    object_slot: int | None
    primitive: str | None
    action_kind: str


@dataclass(frozen=True)
class SkillObservation:
    values: list[float]
    object_slots: dict[int, dict[str, Any]]
    focus_edge: int | None
    required_count: int
    stack_depth: int
    target_locations: list[dict[str, int]]
    mapping_status: str


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def safe_sha256_file(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return sha256_file(path)


def latest_default_checkpoint(policy_env: str = DEFAULT_POLICY_ENV) -> Path | None:
    run_root = ROOT / "codexlab" / "runs" / "_system_run"
    pattern = f"rexus_sp80_paintflow_l1_multi_native_train_v0*/checkpoints/{policy_env}/**/*.bin"
    candidates = [path for path in run_root.glob(pattern) if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def native_action_from_id(action_id: int) -> NativeAction:
    action = int(action_id)
    if action == 0:
        return NativeAction(action, None, None, "noop")
    if 1 <= action <= 12:
        slot = (action - 1) // 4
        move = (action - 1) % 4
        primitive = ["move_up", "move_down", "move_left", "move_right"][move]
        return NativeAction(action, slot, primitive, "move_object")
    if 13 <= action <= 15:
        return NativeAction(action, action - 13, "activate_flow", "activate_edge")
    return NativeAction(action, None, None, "invalid")


def _int_pair(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        return [int(round(float(value[0]))), int(round(float(value[1])))]
    except (TypeError, ValueError):
        return None


def _manhattan_pair(left: list[int] | None, right: list[int] | None) -> int | None:
    if left is None or right is None:
        return None
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def _request_selected_display_coord(request: dict[str, Any]) -> list[int] | None:
    observation = request.get("observation") if isinstance(request.get("observation"), dict) else {}
    state = observation.get("state") if isinstance(observation.get("state"), dict) else {}
    return _int_pair(state.get("selected_display_coord"))


def _int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    try:
        return [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None


def _request_selected_state(request: dict[str, Any]) -> dict[str, Any]:
    observation = request.get("observation") if isinstance(request.get("observation"), dict) else {}
    state = observation.get("state") if isinstance(observation.get("state"), dict) else {}
    selected: dict[str, Any] = {}
    coord = _int_pair(state.get("selected_display_coord"))
    if coord is not None:
        selected["selected_display_coord"] = coord
    selected_coord = _int_pair(state.get("selected_coord"))
    if selected_coord is not None:
        # Coordinate authority: contra_head.crucible.canon uses SP80 [x,y] and grid[x,y].
        selected["selected_coord"] = selected_coord
    name = state.get("selected_name")
    if name is not None:
        selected["selected_name"] = str(name)
    shape = _int_list(state.get("selected_shape"))
    if shape is not None:
        selected["selected_shape"] = shape
    return selected


def _scale_pair(value: list[int], *, source_w: int = SP80_DISPLAY_W, source_h: int = SP80_DISPLAY_H) -> list[int]:
    x = int(round(float(value[0]) / float(max(1, source_w - 1)) * float(GRID_W - 1)))
    y = int(round(float(value[1]) / float(max(1, source_h - 1)) * float(GRID_H - 1)))
    return [max(0, min(GRID_W - 1, x)), max(0, min(GRID_H - 1, y))]


def _bucket_index(value: Any) -> int | None:
    text = str(value or "")
    if "_" not in text:
        return None
    try:
        return int(text.rsplit("_", 1)[1])
    except ValueError:
        return None


def _target_override_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    target = row.get("target_override")
    if isinstance(target, dict):
        click = _int_pair(target.get("click_xy"))
        if click is not None:
            return {**target, "click_xy": click}
    bbox = row.get("source_bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        return {
            "target_kind": "paint_control_skill_source_object",
            "source_bbox_xyxy": bbox,
            "click_xy": [int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0))],
            "selection_basis": "paint_control_skill_object_slot",
            "source_object_id": str(row.get("source_object_id") or ""),
            "source_component_ids": row.get("source_component_ids") if isinstance(row.get("source_component_ids"), list) else [],
        }
    return None


def _selection_fallback_authority(recommendation: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(recommendation, dict):
        return {
            "status": "fail_closed",
            "fail_closed_reason": "fallback_select_object_slot_authority_unclear",
        }
    target_override = recommendation.get("target_override")
    evidence_refs = recommendation.get("evidence_refs") if isinstance(recommendation.get("evidence_refs"), dict) else {}
    evidence_sources = recommendation.get("evidence_sources") if isinstance(recommendation.get("evidence_sources"), list) else []
    candidate_kind = str(recommendation.get("candidate_kind") or "")
    target_kind = str((target_override or {}).get("target_kind") or "") if isinstance(target_override, dict) else ""
    target_components = evidence_refs.get("target_component_ids")
    target_ports = evidence_refs.get("target_port_ids")
    portband_locked = (
        candidate_kind in {"multi_object_goal_adjacency_plan", "portband_lineup_matrix_row"}
        or "portband.lineup_matrix.rows" in evidence_sources
    )
    portband_locked = (
        portband_locked
        and target_kind == "portband_source_object"
        and isinstance(target_components, list)
        and bool(target_components)
        and isinstance(target_ports, list)
        and bool(target_ports)
    )
    if not portband_locked:
        return {
            "status": "fail_closed",
            "fail_closed_reason": "fallback_select_object_slot_authority_unclear",
            "candidate_kind": candidate_kind,
            "target_kind": target_kind,
            "evidence_sources": evidence_sources,
            "target_component_ids": target_components if isinstance(target_components, list) else [],
            "target_port_ids": target_ports if isinstance(target_ports, list) else [],
            "boundary": "ranked fallback may only convert native move-object intent into selection when PortBand supplies lock/key target context",
        }
    return {
        "status": "observed",
        "fail_closed_reason": None,
        "candidate_kind": candidate_kind,
        "target_kind": target_kind,
        "evidence_sources": evidence_sources,
        "target_component_ids": target_components,
        "target_port_ids": target_ports,
        "boundary": "paint_control_skill_uses_portband_locked_candidate_context_but_retains_action_authority",
    }


def _portband_source_key(recommendation: dict[str, Any] | None) -> str | None:
    if not isinstance(recommendation, dict):
        return None
    evidence_refs = recommendation.get("evidence_refs") if isinstance(recommendation.get("evidence_refs"), dict) else {}
    task_key = evidence_refs.get("portband_task_key")
    if isinstance(task_key, str) and task_key:
        return task_key
    source_components = evidence_refs.get("source_component_ids")
    if isinstance(source_components, list) and source_components:
        return "source_component:" + str(source_components[0])
    target_override = recommendation.get("target_override")
    target_override = target_override if isinstance(target_override, dict) else {}
    source_components = target_override.get("source_component_ids")
    if isinstance(source_components, list) and source_components:
        return "source_component:" + str(source_components[0])
    component_id = target_override.get("component_id")
    target_kind = str(target_override.get("target_kind") or "")
    if component_id is not None and target_kind == "portband_source_object":
        return "source_component:" + str(component_id)
    return None


def _non_positive_distance_delta(row: dict[str, Any]) -> bool:
    progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
    distance_delta = progress.get("distance_delta")
    if isinstance(distance_delta, (int, float)):
        return float(distance_delta) <= 0.0
    return progress.get("distance_reduced") is not True


def stationary_move_feedback(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str | None,
    object_slot: int | None,
    threshold: int = REPEATED_STATIONARY_MOVE_FAIL_THRESHOLD,
) -> dict[str, Any]:
    if primitive is None or not str(primitive).startswith("move_"):
        return {
            "status": "not_applicable",
            "stationary_repeat_count": 0,
            "threshold": int(threshold),
            "fail_closed_reason": None,
        }
    count = 0
    considered: list[dict[str, Any]] = []
    for row in reversed(action_memory[-8:]):
        if not isinstance(row, dict):
            break
        if row.get("status") != "executed" or row.get("primitive") != primitive:
            break
        selected_delta = row.get("selected_display_coord_delta")
        stationary = selected_delta == [0, 0] and _non_positive_distance_delta(row)
        considered.append(
            {
                "action_index": row.get("action_index"),
                "primitive": row.get("primitive"),
                "selected_display_coord_delta": selected_delta,
                "distance_delta": (row.get("goal_progress") or {}).get("distance_delta")
                if isinstance(row.get("goal_progress"), dict)
                else None,
                "stationary": stationary,
            }
        )
        if not stationary:
            break
        count += 1
    fail_reason = "repeated_stationary_move_feedback" if count >= int(threshold) else None
    return {
        "status": "fail_closed" if fail_reason else "observed",
        "stationary_repeat_count": int(count),
        "threshold": int(threshold),
        "primitive": primitive,
        "object_slot": object_slot,
        "considered_recent_actions": considered,
        "fail_closed_reason": fail_reason,
    }


def _movement_sequence(action_memory: list[dict[str, Any]]) -> list[str]:
    sequence: list[str] = []
    for row in reversed(action_memory):
        primitive = str(row.get("primitive") or "")
        if primitive not in INVERSE_MOVES:
            break
        sequence.append(primitive)
    return list(reversed(sequence))


def _memory_row_source_key(row: dict[str, Any]) -> str | None:
    recommendation = row.get("selected_recommendation") if isinstance(row, dict) else None
    return _portband_source_key(recommendation if isinstance(recommendation, dict) else None)


def _oscillation_pair_key(left: str, right: str) -> str:
    return "|".join(sorted([left, right]))


def _goal_direction_primitives(request: dict[str, Any]) -> set[str]:
    goal = request.get("goal_condition") if isinstance(request.get("goal_condition"), dict) else {}
    target = goal.get("target_goal") if isinstance(goal.get("target_goal"), dict) else None
    if target is None:
        target = goal.get("nearest_goal") if isinstance(goal.get("nearest_goal"), dict) else None
    if not isinstance(target, dict):
        return set()
    signs = target.get("display_delta_sign")
    if not isinstance(signs, list) or len(signs) != 2:
        return set()
    primitives: set[str] = set()
    if signs[0] == "positive":
        primitives.add("move_right")
    elif signs[0] == "negative":
        primitives.add("move_left")
    if signs[1] == "positive":
        primitives.add("move_down")
    elif signs[1] == "negative":
        primitives.add("move_up")
    return primitives


def _predicted_delta_is_away_from_goal(request: dict[str, Any], predicted_delta: tuple[int, int] | list[int] | None) -> bool:
    if not isinstance(predicted_delta, (list, tuple)) or len(predicted_delta) != 2:
        return False
    try:
        dx = int(predicted_delta[0])
        dy = int(predicted_delta[1])
    except (TypeError, ValueError):
        return False
    if dx == 0 and dy == 0:
        return False
    goal = request.get("goal_condition") if isinstance(request.get("goal_condition"), dict) else {}
    target = goal.get("target_goal") if isinstance(goal.get("target_goal"), dict) else None
    if target is None:
        target = goal.get("nearest_goal") if isinstance(goal.get("nearest_goal"), dict) else None
    if not isinstance(target, dict):
        return False
    signs = target.get("display_delta_sign")
    if not isinstance(signs, list) or len(signs) != 2:
        return False
    desired_x = 1 if signs[0] == "positive" else -1 if signs[0] == "negative" else 0
    desired_y = 1 if signs[1] == "positive" else -1 if signs[1] == "negative" else 0
    if dx != 0 and signs[0] == "same":
        return True
    if dy != 0 and signs[1] == "same":
        return True
    if dx != 0 and desired_x != 0:
        return dx * desired_x < 0
    if dy != 0 and desired_y != 0:
        return dy * desired_y < 0
    return False


def fallback_candidate_score(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str | None,
    request: dict[str, Any],
    rank_index: int,
    recommendation: dict[str, Any] | None = None,
    current_display_coord: list[int] | None = None,
    window: int = 12,
) -> dict[str, Any]:
    goal_primitives = _goal_direction_primitives(request)
    positive_count = 0
    harmful_count = 0
    stationary_count = 0
    for row in reversed(action_memory[-int(window) :]):
        if not isinstance(row, dict) or row.get("status") != "executed" or row.get("primitive") != primitive:
            continue
        progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
        try:
            distance_delta = float(progress.get("distance_delta"))
        except (TypeError, ValueError):
            distance_delta = 1.0 if progress.get("distance_reduced") is True else 0.0
        if distance_delta > 0.0 and row.get("selected_display_coord_delta") != [0, 0]:
            positive_count += 1
        elif row.get("selected_display_coord_delta") == [0, 0]:
            coord_distance = _manhattan_pair(_int_pair(row.get("selected_display_coord_before")), current_display_coord)
            if coord_distance is not None and coord_distance != 0:
                continue
            stationary_count += 1
        elif distance_delta < 0.0:
            coord_distance = _manhattan_pair(_int_pair(row.get("selected_display_coord_before")), current_display_coord)
            if coord_distance is not None and coord_distance != 0:
                continue
            harmful_count += 1
    score = 0.0
    if primitive in goal_primitives:
        score += 50.0
    score += 8.0 * positive_count
    score -= 20.0 * harmful_count
    score -= 14.0 * stationary_count
    if primitive == "select_object" and goal_primitives:
        score -= 30.0
    selection_budget = selection_budget_feedback(
        primitive=primitive,
        recommendation=recommendation,
        request=request,
    )
    if primitive == "select_object" and selection_budget.get("status") == "observed":
        cost = selection_budget.get("estimated_action_cost")
        try:
            score -= min(40.0, 0.25 * float(cost))
        except (TypeError, ValueError):
            pass
    if primitive == "activate_flow":
        score += 5.0
    score -= 0.01 * int(rank_index)
    return {
        "score": round(score, 6),
        "primitive": primitive,
        "rank_index": int(rank_index),
        "current_display_coord": current_display_coord,
        "goal_direction_primitives": sorted(goal_primitives),
        "recent_positive_count": positive_count,
        "recent_harmful_count": harmful_count,
        "recent_stationary_count": stationary_count,
        "selection_budget_feedback": selection_budget,
        "boundary": "fallback_score_prioritizes_live_goal_direction_and_recent_control_feedback_not_solve_truth",
    }


def fallback_candidate_feedback(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str | None,
    source_key: str | None = None,
    current_display_coord: list[int] | None = None,
    window: int = 12,
) -> dict[str, Any]:
    if primitive is None or primitive not in INVERSE_MOVES:
        return {
            "status": "not_applicable",
            "primitive": primitive,
            "fail_closed_reason": None,
        }

    scoped_memory = [
        row
        for row in action_memory
        if isinstance(row, dict) and (source_key is None or _memory_row_source_key(row) == source_key)
    ]
    sequence = _movement_sequence(scoped_memory)
    pair_key = None
    prior_pair_count = 0
    if sequence and INVERSE_MOVES[primitive] == sequence[-1]:
        pair_key = _oscillation_pair_key(sequence[-1], primitive)
        for idx in range(1, len(sequence)):
            left = sequence[idx - 1]
            right = sequence[idx]
            if INVERSE_MOVES[left] == right and _oscillation_pair_key(left, right) == pair_key:
                prior_pair_count += 1
    last_move = scoped_memory[-1] if scoped_memory else {}
    last_stationary_at_current = (
        isinstance(last_move, dict)
        and last_move.get("primitive") == INVERSE_MOVES.get(primitive)
        and last_move.get("selected_display_coord_delta") == [0, 0]
        and _manhattan_pair(_int_pair(last_move.get("selected_display_coord_before")), current_display_coord) in {0, None}
    )
    if pair_key and prior_pair_count >= 1 and not last_stationary_at_current:
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "current_display_coord": current_display_coord,
            "movement_sequence_tail": sequence[-6:],
            "movement_oscillation_pair_key": pair_key,
            "movement_oscillation_prior_pair_count": prior_pair_count,
            "fail_closed_reason": "fallback_repeated_movement_oscillation",
        }

    harmful_rows: list[dict[str, Any]] = []
    stationary_rows: list[dict[str, Any]] = []
    for row in reversed(action_memory[-int(window) :]):
        if not isinstance(row, dict) or row.get("primitive") != primitive or row.get("status") != "executed":
            continue
        if source_key is not None and _memory_row_source_key(row) != source_key:
            continue
        coord_distance = _manhattan_pair(_int_pair(row.get("selected_display_coord_before")), current_display_coord)
        if coord_distance is not None and coord_distance != 0:
            continue
        progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
        distance_delta = progress.get("distance_delta")
        try:
            delta = float(distance_delta)
        except (TypeError, ValueError):
            delta = 0.0 if progress.get("distance_reduced") is not True else 1.0
        row_summary = {
            "action_index": row.get("action_index"),
            "primitive": primitive,
            "selected_display_coord_delta": row.get("selected_display_coord_delta"),
            "distance_delta": delta,
        }
        if delta < 0.0:
            harmful_rows.append(row_summary)
        if row.get("selected_display_coord_delta") == [0, 0] and delta <= 0.0:
            stationary_rows.append(row_summary)
    if harmful_rows:
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "source_key": source_key,
            "current_display_coord": current_display_coord,
            "harmful_prior_count": len(harmful_rows),
            "harmful_prior_rows": harmful_rows[:4],
            "fail_closed_reason": "fallback_negative_distance_feedback",
        }
    if stationary_rows:
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "source_key": source_key,
            "current_display_coord": current_display_coord,
            "stationary_prior_count": len(stationary_rows),
            "stationary_prior_rows": stationary_rows[:4],
            "fail_closed_reason": "fallback_stationary_move_feedback",
        }

    return {
        "status": "observed",
        "primitive": primitive,
        "source_key": source_key,
        "current_display_coord": current_display_coord,
        "movement_sequence_tail": sequence[-6:],
        "movement_oscillation_pair_key": pair_key,
        "movement_oscillation_prior_pair_count": prior_pair_count,
        "last_stationary_at_current": bool(last_stationary_at_current),
        "fail_closed_reason": None,
    }


def selection_lock_feedback(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str | None,
    recommendation: dict[str, Any] | None,
    window: int = 8,
    positive_threshold: int = 2,
    saturation_threshold: int = 2,
) -> dict[str, Any]:
    if primitive != "select_object":
        return {
            "status": "not_applicable",
            "primitive": primitive,
            "fail_closed_reason": None,
        }

    positive_rows: list[dict[str, Any]] = []
    saturation_rows: list[dict[str, Any]] = []
    for row in reversed(action_memory[-int(window) :]):
        if not isinstance(row, dict) or row.get("status") != "executed":
            continue
        row_primitive = str(row.get("primitive") or "")
        if not row_primitive.startswith("move_"):
            continue
        progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
        try:
            distance_delta = float(progress.get("distance_delta"))
        except (TypeError, ValueError):
            distance_delta = 1.0 if progress.get("distance_reduced") is True else 0.0
        if distance_delta <= 0.0 or row.get("selected_display_coord_delta") == [0, 0]:
            saturation_rows.append(
                {
                    "action_index": row.get("action_index"),
                    "primitive": row.get("primitive"),
                    "selected_display_coord_before": row.get("selected_display_coord_before"),
                    "selected_display_coord_delta": row.get("selected_display_coord_delta"),
                    "distance_delta": distance_delta,
                    "source_key": _memory_row_source_key(row),
                }
            )
            continue
        positive_rows.append(
            {
                "action_index": row.get("action_index"),
                "primitive": row.get("primitive"),
                "selected_display_coord_delta": row.get("selected_display_coord_delta"),
                "distance_delta": distance_delta,
                "source_key": _memory_row_source_key(row),
            }
        )

    fail_reason = None
    if len(positive_rows) >= int(positive_threshold):
        fail_reason = "selection_lock_preserve_positive_source"
    if fail_reason is not None and len(saturation_rows) >= int(saturation_threshold):
        fail_reason = None
    return {
        "status": "fail_closed" if fail_reason else "observed",
        "primitive": primitive,
        "candidate_source_key": _portband_source_key(recommendation),
        "recent_positive_move_count": len(positive_rows),
        "recent_positive_move_rows": positive_rows[:6],
        "recent_saturation_move_count": len(saturation_rows),
        "recent_saturation_move_rows": saturation_rows[:6],
        "positive_threshold": int(positive_threshold),
        "saturation_threshold": int(saturation_threshold),
        "selection_lock_released_after_saturation": bool(len(positive_rows) >= int(positive_threshold) and len(saturation_rows) >= int(saturation_threshold)),
        "fail_closed_reason": fail_reason,
        "boundary": "recent controlled movement progress preserves the selected object; PortBand remains review context, not selection authority",
    }


def _recommendation_step_cost(recommendation: dict[str, Any] | None) -> float | None:
    if not isinstance(recommendation, dict):
        return None
    signals = recommendation.get("signals") if isinstance(recommendation.get("signals"), dict) else {}
    for key in ("lineup_step_cost", "total_step_cost", "average_step_cost"):
        value = signals.get(key)
        try:
            cost = float(value)
        except (TypeError, ValueError):
            continue
        if cost >= 0.0:
            return cost
    return None


def _remaining_action_budget(request: dict[str, Any]) -> int | None:
    budget = request.get("episode_budget") if isinstance(request.get("episode_budget"), dict) else {}
    value = budget.get("remaining_action_budget")
    try:
        remaining = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, remaining)


def selection_budget_feedback(
    *,
    primitive: str | None,
    recommendation: dict[str, Any] | None,
    request: dict[str, Any],
) -> dict[str, Any]:
    if primitive != "select_object":
        return {
            "status": "not_applicable",
            "primitive": primitive,
            "fail_closed_reason": None,
        }
    estimated_cost = _recommendation_step_cost(recommendation)
    remaining = _remaining_action_budget(request)
    if estimated_cost is None or remaining is None:
        return {
            "status": "unavailable",
            "primitive": primitive,
            "estimated_action_cost": estimated_cost,
            "remaining_action_budget": remaining,
            "fail_closed_reason": None,
            "boundary": "selection_budget_is_controlled_harness_context_not_goal_truth_or_solve_truth",
        }
    post_selection_budget = max(0, int(remaining) - 1)
    infeasible = float(estimated_cost) > float(post_selection_budget)
    return {
        "status": "fail_closed" if infeasible else "observed",
        "primitive": primitive,
        "estimated_action_cost": estimated_cost,
        "remaining_action_budget": int(remaining),
        "post_selection_action_budget": int(post_selection_budget),
        "fail_closed_reason": "selection_budget_infeasible" if infeasible else None,
        "boundary": "selection_budget_is_controlled_harness_context_not_goal_truth_or_solve_truth",
    }


def activation_flow_feedback(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str | None,
    observation: SkillObservation,
    native_action: NativeAction,
    window: int = 8,
    adjacent_saturation_threshold: int = 2,
) -> dict[str, Any]:
    if primitive != "activate_flow":
        return {
            "status": "not_applicable",
            "primitive": primitive,
            "fail_closed_reason": None,
        }

    prior_bad: list[dict[str, Any]] = []
    for row in reversed(action_memory[-int(window) :]):
        if not isinstance(row, dict) or row.get("primitive") != "activate_flow" or row.get("status") != "executed":
            continue
        progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
        try:
            distance_delta = float(progress.get("distance_delta"))
        except (TypeError, ValueError):
            distance_delta = 0.0 if progress.get("distance_reduced") is not True else 1.0
        levels_delta = int(row.get("levels_completed_delta") or 0)
        if levels_delta <= 0 and distance_delta <= 0.0:
            prior_bad.append(
                {
                    "action_index": row.get("action_index"),
                    "distance_delta": distance_delta,
                    "levels_completed_delta": levels_delta,
                    "selected_display_coord_delta": row.get("selected_display_coord_delta"),
                }
            )
    if prior_bad:
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "prior_bad_activation_count": len(prior_bad),
            "prior_bad_activation_rows": prior_bad[:4],
            "fail_closed_reason": "activation_prior_no_progress_feedback",
        }

    if native_action.object_slot is None or observation.focus_edge is None or native_action.object_slot != observation.focus_edge:
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "native_object_slot": native_action.object_slot,
            "focus_edge": observation.focus_edge,
            "fail_closed_reason": "activation_readiness_unknown",
        }
    selected_slot = observation.object_slots.get(0)
    if not isinstance(selected_slot, dict) or selected_slot.get("known") is not True:
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "focus_edge": observation.focus_edge,
            "fail_closed_reason": "activation_readiness_unknown",
        }
    target = observation.target_locations[observation.focus_edge]
    try:
        selected_x = int(selected_slot["x"])
        selected_y = int(selected_slot["y"])
        target_x = int(target["x"])
        target_y = int(target["y"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "focus_edge": observation.focus_edge,
            "fail_closed_reason": "activation_readiness_unknown",
        }
    target_distance = abs(target_x - selected_x) + abs(target_y - selected_y)
    if target_distance != 0:
        saturated_rows: list[dict[str, Any]] = []
        selected_display_coord = _int_pair(selected_slot.get("display_coord"))
        selected_source_key = selected_slot.get("portband_source_key")
        effective_adjacent_saturation_threshold = 1 if selected_source_key is not None else int(adjacent_saturation_threshold)
        if target_distance == 1:
            for row in reversed(action_memory[-int(window) :]):
                if not isinstance(row, dict) or row.get("status") != "executed":
                    continue
                row_primitive = str(row.get("primitive") or "")
                if not row_primitive.startswith("move_"):
                    continue
                coord_distance = _manhattan_pair(_int_pair(row.get("selected_display_coord_before")), selected_display_coord)
                if selected_display_coord is not None and coord_distance not in {0, None}:
                    continue
                progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
                try:
                    distance_delta = float(progress.get("distance_delta"))
                except (TypeError, ValueError):
                    distance_delta = 1.0 if progress.get("distance_reduced") is True else 0.0
                if row.get("selected_display_coord_delta") != [0, 0] or distance_delta > 0.0:
                    continue
                saturated_rows.append(
                    {
                        "action_index": row.get("action_index"),
                        "primitive": row.get("primitive"),
                        "selected_display_coord_before": row.get("selected_display_coord_before"),
                        "selected_display_coord_delta": row.get("selected_display_coord_delta"),
                        "distance_delta": distance_delta,
                    }
                )
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "focus_edge": observation.focus_edge,
            "selected_slot": {"slot": 0, "x": selected_x, "y": selected_y},
            "selected_display_coord": selected_display_coord,
            "selected_source_key": selected_source_key,
            "activation_target": {"x": target_x, "y": target_y},
            "target_distance": target_distance,
            "adjacent_saturation_move_count": len(saturated_rows),
            "adjacent_saturation_threshold": effective_adjacent_saturation_threshold,
            "adjacent_saturation_rows": saturated_rows[:4],
            "fail_closed_reason": "activation_readiness_target_misaligned",
            "boundary": "distance_one_activation_has_only_neutral_or_negative_controlled_history_until_a_source_edge_model_promotes_it",
        }
    return {
        "status": "observed",
        "primitive": primitive,
        "focus_edge": observation.focus_edge,
        "selected_slot": {"slot": 0, "x": selected_x, "y": selected_y},
        "activation_target": {"x": target_x, "y": target_y},
        "target_distance": 0,
        "fail_closed_reason": None,
    }


def source_edge_activation_probe_feedback(
    action_memory: list[dict[str, Any]],
    *,
    activation_feedback: dict[str, Any],
    outcome_feedback: dict[str, Any],
    window: int = 24,
) -> dict[str, Any]:
    if activation_feedback.get("fail_closed_reason") != "activation_readiness_target_misaligned":
        return activation_feedback
    if int(activation_feedback.get("target_distance") or -1) != 1:
        return activation_feedback
    if outcome_feedback.get("fail_closed_reason") is not None:
        return activation_feedback
    selected_source_key = activation_feedback.get("selected_source_key")
    if not isinstance(selected_source_key, str) or not selected_source_key:
        return activation_feedback
    selected_display_coord = _int_pair(activation_feedback.get("selected_display_coord"))
    focus_edge = activation_feedback.get("focus_edge")
    try:
        adjacent_saturation_count = int(activation_feedback.get("adjacent_saturation_move_count") or 0)
        adjacent_saturation_threshold = int(activation_feedback.get("adjacent_saturation_threshold") or 1)
    except (TypeError, ValueError):
        return activation_feedback
    if adjacent_saturation_count < max(1, adjacent_saturation_threshold):
        return activation_feedback

    source_probe_rows: list[dict[str, Any]] = []
    prior_same_coord_activation_rows: list[dict[str, Any]] = []
    for row in reversed(action_memory[-int(window) :]):
        if not isinstance(row, dict) or row.get("status") != "executed":
            continue
        primitive = row.get("primitive")
        if primitive == "activate_flow":
            coord_before = _int_pair(row.get("selected_display_coord_before"))
            if selected_display_coord is not None and coord_before == selected_display_coord:
                progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
                try:
                    distance_delta = float(progress.get("distance_delta"))
                except (TypeError, ValueError):
                    distance_delta = 1.0 if progress.get("distance_reduced") is True else 0.0
                prior_same_coord_activation_rows.append(
                    {
                        "action_index": row.get("action_index"),
                        "selected_display_coord_before": coord_before,
                        "distance_delta": distance_delta,
                        "levels_completed_delta": int(row.get("levels_completed_delta") or 0),
                    }
                )
            continue
        if primitive != "select_object":
            continue
        recommendation = row.get("selected_recommendation")
        recommendation = recommendation if isinstance(recommendation, dict) else {}
        source_key = _portband_source_key(recommendation)
        if source_key != selected_source_key:
            continue
        if recommendation.get("candidate_kind") != "paint_control_skill_source_diversity_probe":
            continue
        source_probe_rows.append(
            {
                "action_index": row.get("action_index"),
                "source_key": source_key,
                "candidate_kind": recommendation.get("candidate_kind"),
            }
        )
        break
    if prior_same_coord_activation_rows:
        return {
            **activation_feedback,
            "prior_same_coord_activation_rows": prior_same_coord_activation_rows[:4],
            "fail_closed_reason": "activation_prior_no_progress_feedback",
            "boundary": "source_edge_activation_probe_is_one_shot_per_source_coordinate_until_positive_history_exists",
        }
    if not source_probe_rows:
        return activation_feedback
    return {
        **activation_feedback,
        "status": "observed",
        "activation_readiness": "source_edge_discrimination_probe",
        "source_edge_discrimination_probe": True,
        "source_edge_probe_basis": {
            "source_diversity_selection_rows": source_probe_rows[:2],
            "adjacent_saturation_move_count": adjacent_saturation_count,
            "adjacent_saturation_threshold": adjacent_saturation_threshold,
            "outcome_history_status": outcome_feedback.get("status"),
            "outcome_history_matched_rule_id": outcome_feedback.get("matched_rule_id"),
        },
        "fail_closed_reason": None,
        "boundary": "source_edge_discrimination_probe_is_one_shot_hypothesis_action_not_activation_truth_not_solve_truth",
    }


def _selection_outcome_candidate_keys(recommendation: dict[str, Any] | None) -> list[str]:
    if not isinstance(recommendation, dict):
        return []
    keys: list[str] = []
    source_key = _portband_source_key(recommendation)
    if source_key:
        keys.append(str(source_key))
    evidence_refs = recommendation.get("evidence_refs") if isinstance(recommendation.get("evidence_refs"), dict) else {}
    source_components = evidence_refs.get("source_component_ids")
    if isinstance(source_components, list):
        for value in source_components:
            if value is not None:
                keys.append(str(value))
                short = str(value).split("_v", 1)[0]
                if short:
                    keys.append("source_component:" + short)
    target_override = recommendation.get("target_override")
    target_override = target_override if isinstance(target_override, dict) else {}
    component_id = target_override.get("component_id")
    if component_id is not None:
        keys.append(str(component_id))
        short = str(component_id).split("_v", 1)[0]
        if short:
            keys.append("source_component:" + short)
    seen: set[str] = set()
    unique: list[str] = []
    for key in keys:
        if key and key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def _outcome_rule_matches(
    rule: dict[str, Any],
    *,
    primitive: str | None,
    recommendation: dict[str, Any] | None,
    current_display_coord: list[int] | None,
    observation: SkillObservation | None = None,
    selected_state: dict[str, Any] | None = None,
) -> bool:
    match = rule.get("match") if isinstance(rule.get("match"), dict) else {}
    rule_primitive = match.get("primitive")
    if rule_primitive is not None and str(rule_primitive) != str(primitive):
        return False
    coord = _int_pair(match.get("current_display_coord"))
    if coord is not None and _manhattan_pair(coord, current_display_coord) != 0:
        return False
    if primitive == "select_object":
        candidate_keys = set(_selection_outcome_candidate_keys(recommendation))
        match_keys = match.get("candidate_keys")
        if isinstance(match_keys, list) and match_keys:
            return any(str(key) in candidate_keys for key in match_keys)
        source_key = match.get("source_key")
        if source_key is not None:
            return str(source_key) in candidate_keys
        component_id = match.get("component_id")
        if component_id is not None:
            return str(component_id) in candidate_keys
        return False
    if primitive == "activate_flow":
        if observation is None or observation.focus_edge is None:
            return False
        selected_state = selected_state if isinstance(selected_state, dict) else {}
        match_selected_name = match.get("selected_name_before")
        if match_selected_name is None:
            match_selected_name = match.get("selected_name")
        if match_selected_name is not None and str(match_selected_name) != str(selected_state.get("selected_name")):
            return False
        match_selected_shape = _int_list(match.get("selected_shape_before"))
        if match_selected_shape is None:
            match_selected_shape = _int_list(match.get("selected_shape"))
        if match_selected_shape is not None and match_selected_shape != _int_list(selected_state.get("selected_shape")):
            return False
        selected_slot = observation.object_slots.get(0)
        current_source_key = None
        if isinstance(selected_slot, dict):
            current_source_key = selected_slot.get("portband_source_key")
        match_source_key = match.get("active_portband_source_key")
        if match_source_key is None:
            match_source_key = match.get("source_key")
        if match_source_key is not None and str(match_source_key) != str(current_source_key):
            return False
        focus_edge = match.get("focus_edge")
        if focus_edge is not None:
            try:
                if int(focus_edge) != int(observation.focus_edge):
                    return False
            except (TypeError, ValueError):
                return False
        target_distance = match.get("target_distance")
        if target_distance is None:
            return True
        if not isinstance(selected_slot, dict):
            return False
        target = observation.target_locations[observation.focus_edge]
        try:
            distance = abs(int(target["x"]) - int(selected_slot["x"])) + abs(int(target["y"]) - int(selected_slot["y"]))
            return int(target_distance) == distance
        except (KeyError, TypeError, ValueError):
            return False
    return False


def outcome_history_feedback(
    ledger: dict[str, Any] | None,
    *,
    primitive: str | None,
    recommendation: dict[str, Any] | None,
    current_display_coord: list[int] | None,
    observation: SkillObservation | None = None,
    selected_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if primitive not in {"select_object", "activate_flow"}:
        return {
            "status": "not_applicable",
            "primitive": primitive,
            "fail_closed_reason": None,
        }
    if not isinstance(ledger, dict):
        return {
            "status": "unavailable",
            "primitive": primitive,
            "fail_closed_reason": None,
        }
    rules = ledger.get("rules")
    if not isinstance(rules, list):
        return {
            "status": "unavailable",
            "primitive": primitive,
            "fail_closed_reason": None,
        }
    for rule in rules:
        if not isinstance(rule, dict) or not _outcome_rule_matches(
            rule,
            primitive=primitive,
            recommendation=recommendation,
            current_display_coord=current_display_coord,
            observation=observation,
            selected_state=selected_state,
        ):
            continue
        effect_kind = str(rule.get("effect_kind") or "")
        if effect_kind not in {"harmful_selection", "harmful_activation", "no_progress_selection", "no_progress_activation"}:
            continue
        fail_reason = "activation_outcome_harmful_history" if primitive == "activate_flow" else "selection_outcome_harmful_history"
        return {
            "status": "fail_closed",
            "primitive": primitive,
            "candidate_keys": _selection_outcome_candidate_keys(recommendation) if primitive == "select_object" else [],
            "current_display_coord": current_display_coord,
            "selected_state": selected_state or {},
            "matched_rule_id": rule.get("rule_id"),
            "effect_kind": effect_kind,
            "support": int(rule.get("support") or 0),
            "contradictions": int(rule.get("contradictions") or 0),
            "confidence": float(rule.get("confidence") or 0.0),
            "ledger_sha256": ledger.get("ledger_sha256"),
            "fail_closed_reason": fail_reason,
            "boundary": "paint_skill_outcome_history_is_controlled_run_feedback_not_selection_or_activation_truth",
        }
    return {
        "status": "observed",
        "primitive": primitive,
        "candidate_keys": _selection_outcome_candidate_keys(recommendation) if primitive == "select_object" else [],
        "current_display_coord": current_display_coord,
        "fail_closed_reason": None,
        "boundary": "paint_skill_outcome_history_is_controlled_run_feedback_not_selection_or_activation_truth",
    }


def _slots_from_request(
    request: dict[str, Any],
    required_count: int,
    *,
    active_portband_source_key: str | None = None,
) -> dict[int, dict[str, Any]]:
    slots: dict[int, dict[str, Any]] = {}
    observation = request.get("observation") if isinstance(request.get("observation"), dict) else {}
    state = observation.get("state") if isinstance(observation.get("state"), dict) else {}
    selected_coord = _int_pair(state.get("selected_coord"))
    if state.get("selected") is True and selected_coord is not None:
        # Coordinate authority: contra_head.crucible.canon uses SP80 [x,y] and grid[x,y].
        slots[0] = {
            "slot": 0,
            "known": True,
            "selected": True,
            "x": selected_coord[0],
            "y": selected_coord[1],
            "target_override": None,
            "source": "live_selected_coord",
        }

    packet = request.get("contra_recommendation_packet")
    rows = packet.get("ranked_recommendations") if isinstance(packet, dict) else []
    candidates: list[tuple[dict[str, Any], dict[str, Any] | None, str | None, bool]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        target = _target_override_from_row(row)
        source_key = _portband_source_key(row)
        active_source_selected = active_portband_source_key is not None and source_key == active_portband_source_key
        if target is None and not active_source_selected:
            continue
        candidates.append((row, target, source_key, active_source_selected))

    next_slot = 0
    seen_source_keys = {
        str(slot.get("portband_source_key"))
        for slot in slots.values()
        if isinstance(slot, dict) and slot.get("portband_source_key")
    }
    deferred_duplicate_sources: list[tuple[dict[str, Any], dict[str, Any] | None, str | None, bool]] = []

    def add_candidate(
        row: dict[str, Any],
        target: dict[str, Any] | None,
        source_key: str | None,
        active_source_selected: bool,
    ) -> bool:
        nonlocal next_slot
        while next_slot in slots:
            next_slot += 1
        if next_slot >= required_count:
            return False
        click = _int_pair(target.get("click_xy")) if isinstance(target, dict) else None
        selected = active_source_selected
        source = "contra_recommendation_target_override"
        x = max(0, min(GRID_W - 1, click[0] if click else DEFAULT_TARGET_X))
        y = max(0, min(GRID_H - 1, click[1] if click else TARGET_ROWS[next_slot]))
        if selected and selected_coord is not None:
            # Coordinate authority: contra_head.crucible.canon uses SP80 [x,y] and grid[x,y].
            x = selected_coord[0]
            y = selected_coord[1]
            source = "active_portband_source_selected_coord"
        if selected and target is None:
            source = "active_portband_source_without_click_target"
        slots[next_slot] = {
            "slot": next_slot,
            "known": True,
            "selected": selected,
            "x": x,
            "y": y,
            "target_override": target,
            "source_recommendation": row,
            "portband_source_key": source_key,
            "source": source,
        }
        if source_key:
            seen_source_keys.add(str(source_key))
        next_slot += 1
        return True

    for row, target, source_key, active_source_selected in candidates:
        if source_key and str(source_key) in seen_source_keys and not active_source_selected:
            deferred_duplicate_sources.append((row, target, source_key, active_source_selected))
            continue
        if not add_candidate(row, target, source_key, active_source_selected):
            break
    if next_slot < required_count:
        for row, target, source_key, active_source_selected in deferred_duplicate_sources:
            if not add_candidate(row, target, source_key, active_source_selected):
                break
    return slots


def _target_locations_from_request(request: dict[str, Any]) -> list[dict[str, int]]:
    targets = [{"x": DEFAULT_TARGET_X, "y": TARGET_ROWS[idx]} for idx in range(3)]
    labels = request.get("goal_labels") if isinstance(request.get("goal_labels"), dict) else {}
    if not labels:
        view_state = request.get("view_state") if isinstance(request.get("view_state"), dict) else {}
        labels = view_state.get("goal_labels") if isinstance(view_state.get("goal_labels"), dict) else {}
    objects = labels.get("objects") if isinstance(labels.get("objects"), list) else []
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("semantic_label") != "bucket":
            continue
        idx = _bucket_index(obj.get("object_id"))
        if idx is None or idx < 0 or idx >= len(targets):
            continue
        center = _int_pair(obj.get("display_center"))
        if center is None:
            bbox = obj.get("display_bbox_xyxy")
            if isinstance(bbox, list) and len(bbox) == 4:
                try:
                    x0, y0, x1, y1 = [float(v) for v in bbox]
                    center = [int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0))]
                except (TypeError, ValueError):
                    center = None
        if center is not None:
            x, y = _scale_pair(center)
            targets[idx] = {"x": x, "y": y}
    return targets


def build_l1_observation(
    request: dict[str, Any],
    *,
    focus_edge_override: int | None = None,
    active_portband_source_key: str | None = None,
) -> SkillObservation:
    direct = request.get("paint_skill_observation")
    if isinstance(direct, dict):
        values = direct.get("values")
        if not isinstance(values, list) or len(values) != OBSERVATION_SIZE:
            raise ValueError("bad_observation_shape")
        try:
            obs = [float(v) for v in values]
        except (TypeError, ValueError) as exc:
            raise ValueError("bad_observation_value") from exc
        slots = {
            int(k): dict(v)
            for k, v in (direct.get("object_slots") or {}).items()
            if isinstance(v, dict)
        }
        return SkillObservation(
            values=obs,
            object_slots=slots,
            focus_edge=int(direct["focus_edge"]) if direct.get("focus_edge") is not None else None,
            required_count=int(direct.get("required_count") or 2),
            stack_depth=int(direct.get("stack_depth") or 0),
            target_locations=list(direct.get("target_locations") or []),
            mapping_status=str(direct.get("mapping_status") or "direct_test_observation"),
        )

    goal = request.get("goal_condition") if isinstance(request.get("goal_condition"), dict) else {}
    required_count = int(goal.get("goal_object_count") or 2)
    required_count = max(2, min(3, required_count))
    focus_edge = focus_edge_override
    if focus_edge is None:
        focus_edge = _bucket_index((goal.get("nearest_goal") or {}).get("object_id") if isinstance(goal.get("nearest_goal"), dict) else None)
    if focus_edge is None:
        focus_edge = 0
    focus_edge = max(0, min(required_count - 1, int(focus_edge)))

    slots = _slots_from_request(request, required_count, active_portband_source_key=active_portband_source_key)
    observation = request.get("observation") if isinstance(request.get("observation"), dict) else {}
    state = observation.get("state") if isinstance(observation.get("state"), dict) else {}
    selected_display_coord = _int_pair(state.get("selected_display_coord"))
    if state.get("selected") is True and selected_display_coord is not None and 0 in slots:
        display_x, display_y = _scale_pair(selected_display_coord)
        slots[0] = {
            **slots[0],
            "x": display_x,
            "y": display_y,
            "raw_selected_coord": slots[0].get("raw_selected_coord") or _int_pair(state.get("selected_coord")),
            "display_coord": selected_display_coord,
            "source": "live_selected_display_coord_scaled",
        }
    if selected_display_coord is not None and active_portband_source_key is not None:
        active_source_slot = None
        for slot in slots.values():
            if slot.get("portband_source_key") != active_portband_source_key:
                continue
            active_source_slot = slot
            display_x, display_y = _scale_pair(selected_display_coord)
            slot["x"] = display_x
            slot["y"] = display_y
            slot["display_coord"] = selected_display_coord
            slot["selected"] = True
            slot["source"] = "active_portband_source_selected_display_coord_scaled"
        if active_source_slot is not None and 0 in slots and slots[0].get("selected") is True:
            slots[0]["source_recommendation"] = active_source_slot.get("source_recommendation")
            slots[0]["portband_source_key"] = active_portband_source_key
            slots[0]["source"] = "live_selected_active_portband_source_display_coord_scaled"
        elif 0 in slots and slots[0].get("selected") is True:
            slots[0]["portband_source_key"] = active_portband_source_key
            slots[0]["source"] = "live_selected_active_portband_source_key_memory_display_coord_scaled"
    for slot in slots.values():
        target_override = slot.get("target_override")
        if slot.get("selected") is True or not isinstance(target_override, dict):
            continue
        click = _int_pair(target_override.get("click_xy"))
        if click is None:
            continue
        display_x, display_y = _scale_pair(click)
        slot["x"] = display_x
        slot["y"] = display_y
        slot["display_coord"] = click
    target_locations = _target_locations_from_request(request)
    stack_depth = 0
    values: list[float] = [
        float(required_count - 2),
        float(focus_edge) / 2.0,
        0.0,
        0.0,
    ]
    for idx in range(3):
        active = idx < required_count
        slot = slots.get(idx)
        target = target_locations[idx]
        if active and slot is not None:
            x = max(0, min(GRID_W - 1, int(slot["x"])))
            y = max(0, min(GRID_H - 1, int(slot["y"])))
        else:
            x = target["x"]
            y = target["y"]
        placed = active and x == target["x"] and y == target["y"]
        if placed:
            stack_depth += 1
        dx = target["x"] - x
        dy = target["y"] - y
        values.extend(
            [
                1.0 if active else 0.0,
                1.0 if idx == focus_edge else 0.0,
                float(x) / float(GRID_W - 1),
                float(y) / float(GRID_H - 1),
                float(target["x"]) / float(GRID_W - 1),
                float(target["y"]) / float(GRID_H - 1),
                float(dx) / float(GRID_W),
                float(dy) / float(GRID_H),
                1.0 if placed else 0.0,
            ]
        )
    values[3] = float(stack_depth) / float(required_count)
    total_distance = 0.0
    for idx in range(required_count):
        slot = slots.get(idx)
        target = target_locations[idx]
        x = int(slot["x"]) if slot is not None else target["x"]
        y = int(slot["y"]) if slot is not None else target["y"]
        total_distance += abs(target["x"] - x) + abs(target["y"] - y)
    values.append(float(total_distance) / float(3 * (GRID_W + GRID_H)))
    if len(values) != OBSERVATION_SIZE:
        raise ValueError("bad_observation_shape")
    return SkillObservation(
        values=values,
        object_slots=slots,
        focus_edge=focus_edge,
        required_count=required_count,
        stack_depth=stack_depth,
        target_locations=target_locations,
        mapping_status="live_selected_and_recommendation_slots" if slots else "no_live_object_slots",
    )


class PaintControlSkillBrain:
    name = "paint_control_skill"

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        policy_env: str = DEFAULT_POLICY_ENV,
        require_native: bool = False,
        focus_edge: int | None = None,
        pufferlib_path: Path | None = None,
        forward_model_ledger_path: Path | None = DEFAULT_DISCREPANCY_LEDGER_PATH,
        outcome_ledger_path: Path | None = DEFAULT_OUTCOME_LEDGER_PATH,
    ) -> None:
        self.policy_env = policy_env
        self.checkpoint_path = (checkpoint_path or latest_default_checkpoint(policy_env))
        self.require_native = bool(require_native)
        self.focus_edge = focus_edge
        self.pufferlib_path = pufferlib_path
        self.pufferlib = inspect_pufferlib()
        self.active_portband_source_key: str | None = None
        self.forward_model_ledger_path = forward_model_ledger_path
        self._forward_model_loaded = False
        self._forward_model: CrucibleModel | None = None
        self.outcome_ledger_path = outcome_ledger_path
        self._outcome_ledger_loaded = False
        self._outcome_ledger: dict[str, Any] | None = None

    def decide(self, request: dict[str, Any]) -> BrainDecision:
        started = time.perf_counter()
        try:
            observation = build_l1_observation(
                request,
                focus_edge_override=self.focus_edge,
                active_portband_source_key=self.active_portband_source_key,
            )
            action_id, inference_raw = self._infer_action(observation.values)
            inference_raw, forward_model_fail_reason, forward_model_rerank = self._forward_model_rerank(
                request=request,
                observation=observation,
                action_id=action_id,
                inference_raw=inference_raw,
            )
            native_action = native_action_from_id(action_id)
            primitive, recommendation, fail_reason = self._translate(native_action, observation)
            if fail_reason is None:
                fail_reason = forward_model_fail_reason
            feedback = stationary_move_feedback(
                request.get("action_memory") if isinstance(request.get("action_memory"), list) else [],
                primitive=primitive,
                object_slot=native_action.object_slot,
            )
            action_memory = request.get("action_memory") if isinstance(request.get("action_memory"), list) else []
            current_display_coord = _request_selected_display_coord(request)
            recommendation_source_key = _portband_source_key(recommendation)
            candidate_feedback = fallback_candidate_feedback(
                action_memory,
                primitive=primitive,
                source_key=recommendation_source_key,
                current_display_coord=current_display_coord,
            )
            selection_feedback = selection_lock_feedback(
                action_memory,
                primitive=primitive,
                recommendation=recommendation,
            )
            selection_budget = selection_budget_feedback(
                primitive=primitive,
                recommendation=recommendation,
                request=request,
            )
            activation_feedback = activation_flow_feedback(
                action_memory,
                primitive=primitive,
                observation=observation,
                native_action=native_action,
            )
            outcome_ledger = self._load_outcome_ledger()
            selected_state = _request_selected_state(request)
            outcome_feedback = outcome_history_feedback(
                outcome_ledger,
                primitive=primitive,
                recommendation=recommendation,
                current_display_coord=current_display_coord,
                observation=observation,
                selected_state=selected_state,
            )
            activation_feedback = source_edge_activation_probe_feedback(
                action_memory,
                activation_feedback=activation_feedback,
                outcome_feedback=outcome_feedback,
            )
            if fail_reason is None:
                fail_reason = feedback.get("fail_closed_reason")
            if fail_reason is None:
                fail_reason = candidate_feedback.get("fail_closed_reason")
                if fail_reason is not None:
                    feedback = {
                        **feedback,
                        "candidate_feedback": candidate_feedback,
                        "fail_closed_reason": fail_reason,
                    }
            if fail_reason is None:
                fail_reason = selection_feedback.get("fail_closed_reason")
                if fail_reason is not None:
                    feedback = {
                        **feedback,
                        "selection_feedback": selection_feedback,
                        "fail_closed_reason": fail_reason,
                    }
            if fail_reason is None:
                fail_reason = activation_feedback.get("fail_closed_reason")
                if fail_reason is not None:
                    feedback = {
                        **feedback,
                        "activation_feedback": activation_feedback,
                        "fail_closed_reason": fail_reason,
                    }
            if fail_reason is None:
                fail_reason = selection_budget.get("fail_closed_reason")
                if fail_reason is not None:
                    feedback = {
                        **feedback,
                        "selection_budget_feedback": selection_budget,
                        "fail_closed_reason": fail_reason,
                    }
            if fail_reason is None:
                fail_reason = outcome_feedback.get("fail_closed_reason")
                if fail_reason is not None:
                    feedback = {
                        **feedback,
                        "outcome_history_feedback": outcome_feedback,
                        "fail_closed_reason": fail_reason,
                    }
            if fail_reason is None and primitive == "activate_flow":
                feedback = {
                    **feedback,
                    "activation_feedback": activation_feedback,
                }
            fallback_raw = None
            if fail_reason in RANKED_FALLBACK_REASONS:
                fallback = self._fallback_from_ranked_actions(
                    inference_raw=inference_raw,
                    observation=observation,
                    request=request,
                    original_action_id=action_id,
                    original_fail_reason=fail_reason,
                    original_feedback=feedback,
                )
                if fallback is not None:
                    action_id = fallback["action_id"]
                    native_action = fallback["native_action"]
                    primitive = fallback["primitive"]
                    recommendation = fallback["recommendation"]
                    fail_reason = fallback["fail_reason"]
                    feedback = fallback["feedback"]
                    fallback_raw = fallback["fallback_raw"]
                    outcome_feedback = fallback["outcome_history_feedback"]
            raw = self._raw_decision(
                observation=observation,
                action_id=action_id,
                inference_raw=inference_raw,
                native_action=native_action,
                primitive=primitive,
                recommendation=recommendation,
                fail_closed_reason=fail_reason,
                live_feedback=feedback,
                fallback_raw=fallback_raw,
                forward_model_rerank=forward_model_rerank,
                outcome_history_feedback=outcome_feedback,
            )
            if fail_reason:
                return BrainDecision(
                    backend=self.name,
                    primitive=None,
                    confidence=None,
                    raw=raw,
                    latency_ms=(time.perf_counter() - started) * 1000.0,
                    error=fail_reason,
                )
            confidence = inference_raw.get("confidence")
            self._remember_selection_state(primitive=primitive, recommendation=recommendation)
            return BrainDecision(
                backend=self.name,
                primitive=primitive,
                confidence=float(confidence) if confidence is not None else None,
                raw=raw,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 - fail-closed backend records the exact reason.
            reason = str(exc) or type(exc).__name__
            raw = self._base_raw(fail_closed_reason=reason)
            return BrainDecision(
                backend=self.name,
                primitive=None,
                confidence=None,
                raw=raw,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=reason,
            )

    def _infer_action(self, values: list[float]) -> tuple[int, dict[str, Any]]:
        if self.checkpoint_path is None:
            raise FileNotFoundError("missing_paint_skill_checkpoint")
        checkpoint = self.checkpoint_path if self.checkpoint_path.is_absolute() else ROOT / self.checkpoint_path
        if not checkpoint.is_file():
            raise FileNotFoundError("missing_paint_skill_checkpoint")
        if checkpoint.suffix == ".json":
            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            if payload.get("schema") != JSON_POLICY_SCHEMA:
                raise ValueError("bad_paint_skill_policy_schema")
            if "action_id" in payload:
                action_id = int(payload["action_id"])
                confidence = float(payload.get("confidence", 1.0))
            else:
                logits = payload.get("logits")
                if not isinstance(logits, list) or len(logits) != 16:
                    raise ValueError("bad_json_policy_logits")
                action_id = max(range(16), key=lambda idx: float(logits[idx]))
                confidence = 1.0
                ranked_action_ids = sorted(range(16), key=lambda idx: float(logits[idx]), reverse=True)
            if "action_id" in payload:
                ranked_action_ids = [action_id]
            return action_id, {
                "inference_backend": "json_policy_smoke",
                "confidence": confidence,
                "ranked_action_ids": ranked_action_ids,
            }
        return self._infer_checkpoint(checkpoint, values)

    def _load_forward_model(self) -> CrucibleModel | None:
        if self._forward_model_loaded:
            return self._forward_model
        self._forward_model_loaded = True
        if self.forward_model_ledger_path is None:
            return None
        ledger_path = self.forward_model_ledger_path
        if not ledger_path.is_absolute():
            ledger_path = ROOT / ledger_path
        self._forward_model = CrucibleModel.load(ledger_path)
        return self._forward_model

    def _load_outcome_ledger(self) -> dict[str, Any] | None:
        if self._outcome_ledger_loaded:
            return self._outcome_ledger
        self._outcome_ledger_loaded = True
        if self.outcome_ledger_path is None:
            return None
        ledger_path = self.outcome_ledger_path
        if not ledger_path.is_absolute():
            ledger_path = ROOT / ledger_path
        if not ledger_path.is_file():
            return None
        try:
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("schema") != OUTCOME_LEDGER_SCHEMA:
            return None
        self._outcome_ledger = {**payload, "ledger_sha256": safe_sha256_file(ledger_path), "ledger_path": display_path(ledger_path)}
        return self._outcome_ledger

    def _forward_model_inputs(self, request: dict[str, Any]) -> tuple[np.ndarray, Any, int] | None:
        observation = request.get("observation") if isinstance(request.get("observation"), dict) else {}
        before_frame = observation.get("before_frame") if isinstance(observation.get("before_frame"), dict) else {}
        raw_grid = before_frame.get("raw_grid") if isinstance(before_frame.get("raw_grid"), dict) else {}
        path_text = raw_grid.get("path")
        state = observation.get("state") if isinstance(observation.get("state"), dict) else {}
        actor_coord = state.get("selected_coord")
        rotation = state.get("rotation_quarters")
        if not isinstance(path_text, str) or actor_coord is None or not isinstance(rotation, int):
            return None
        path = Path(path_text)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            return None
        try:
            return np.load(path), actor_coord, int(rotation)
        except Exception:
            return None

    def _forward_model_rerank(
        self,
        *,
        request: dict[str, Any],
        observation: SkillObservation,
        action_id: int,
        inference_raw: dict[str, Any],
    ) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
        ledger_path = self.forward_model_ledger_path
        ledger_display = display_path(ledger_path if ledger_path is None or ledger_path.is_absolute() else ROOT / ledger_path)
        audit: dict[str, Any] = {
            "status": "unavailable",
            "ledger_path": ledger_display,
            "ledger_sha256": safe_sha256_file(ledger_path if ledger_path is None or ledger_path.is_absolute() else ROOT / ledger_path),
            "demoted_action_ids": [],
            "predictions": [],
            "boundary": "crucible_forward_model_predictions_are_demotion_only_hypotheses_not_control_truth",
        }
        model = self._load_forward_model()
        if model is None:
            return inference_raw, None, audit
        audit["status"] = "no_opinion"
        audit["ledger_sha256"] = getattr(model, "ledger_sha256", None) or audit.get("ledger_sha256")
        inputs = self._forward_model_inputs(request)
        if inputs is None:
            audit["status"] = "no_grid_or_actor_coord"
            return inference_raw, None, audit
        ranked = inference_raw.get("ranked_action_ids")
        if not isinstance(ranked, list):
            audit["status"] = "no_ranked_actions"
            return inference_raw, None, audit
        before_grid, actor_coord, rotation = inputs
        demoted: list[int] = []
        kept: list[int] = []
        predictions: list[dict[str, Any]] = []
        for raw_action_id in ranked:
            try:
                candidate_id = int(raw_action_id)
            except (TypeError, ValueError):
                continue
            native_action = native_action_from_id(candidate_id)
            should_query = (
                native_action.action_kind == "move_object"
                and native_action.primitive is not None
                and native_action.object_slot is not None
                and (observation.object_slots.get(native_action.object_slot) or {}).get("selected") is True
            )
            if not should_query:
                kept.append(candidate_id)
                continue
            prediction = model.predict(
                before_grid,
                native_action.primitive or "",
                actor_coord,
                rotation_quarters=rotation,
            )
            predicted_away = _predicted_delta_is_away_from_goal(request, prediction.predicted_canon_delta)
            prediction_row = {
                "action_id": candidate_id,
                "primitive": native_action.primitive,
                "prediction": prediction.to_dict(),
                "predicted_away_from_goal": predicted_away,
            }
            predictions.append(prediction_row)
            if (prediction.predicted_blocked or predicted_away) and prediction.confidence >= 0.70:
                demoted.append(candidate_id)
            else:
                kept.append(candidate_id)
        audit["predictions"] = predictions
        audit["demoted_action_ids"] = demoted
        if not demoted:
            return inference_raw, None, audit
        reranked = [*kept, *demoted]
        updated_inference = {
            **inference_raw,
            "ranked_action_ids": reranked,
            "forward_model_demoted_action_ids": demoted,
        }
        audit["status"] = "demoted"
        fail_reason = "forward_model_predicted_blocked_or_away" if int(action_id) in demoted else None
        return updated_inference, fail_reason, audit

    def _fallback_from_ranked_actions(
        self,
        *,
        inference_raw: dict[str, Any],
        observation: SkillObservation,
        request: dict[str, Any],
        original_action_id: int,
        original_fail_reason: str | None = None,
        original_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        ranked = inference_raw.get("ranked_action_ids")
        if not isinstance(ranked, list):
            return None
        action_memory = request.get("action_memory") if isinstance(request.get("action_memory"), list) else []
        current_display_coord = _request_selected_display_coord(request)
        outcome_ledger = self._load_outcome_ledger()
        forward_model_demoted = {
            int(value)
            for value in inference_raw.get("forward_model_demoted_action_ids", [])
            if isinstance(value, (int, float))
        }
        rejected: list[dict[str, Any]] = []
        viable: list[dict[str, Any]] = []
        for rank_index, raw_action_id in enumerate(ranked):
            try:
                candidate_id = int(raw_action_id)
            except (TypeError, ValueError):
                continue
            if candidate_id == int(original_action_id):
                continue
            native_action = native_action_from_id(candidate_id)
            primitive, recommendation, fail_reason = self._translate(native_action, observation)
            recommendation_source_key = _portband_source_key(recommendation)
            feedback = stationary_move_feedback(
                action_memory,
                primitive=primitive,
                object_slot=native_action.object_slot,
            )
            candidate_feedback = fallback_candidate_feedback(
                action_memory,
                primitive=primitive,
                source_key=recommendation_source_key,
                current_display_coord=current_display_coord,
            )
            selection_feedback = selection_lock_feedback(
                action_memory,
                primitive=primitive,
                recommendation=recommendation,
            )
            selection_budget = selection_budget_feedback(
                primitive=primitive,
                recommendation=recommendation,
                request=request,
            )
            activation_feedback = activation_flow_feedback(
                action_memory,
                primitive=primitive,
                observation=observation,
                native_action=native_action,
            )
            outcome_feedback = outcome_history_feedback(
                outcome_ledger,
                primitive=primitive,
                recommendation=recommendation,
                current_display_coord=current_display_coord,
                observation=observation,
                selected_state=_request_selected_state(request),
            )
            activation_feedback = source_edge_activation_probe_feedback(
                action_memory,
                activation_feedback=activation_feedback,
                outcome_feedback=outcome_feedback,
            )
            if fail_reason is None:
                fail_reason = feedback.get("fail_closed_reason")
            if fail_reason is None:
                fail_reason = candidate_feedback.get("fail_closed_reason")
            if fail_reason is None:
                fail_reason = selection_feedback.get("fail_closed_reason")
            if fail_reason is None:
                fail_reason = activation_feedback.get("fail_closed_reason")
            if fail_reason is None:
                fail_reason = selection_budget.get("fail_closed_reason")
            if fail_reason is None and primitive == "select_object" and native_action.action_kind == "move_object":
                selection_feedback = _selection_fallback_authority(recommendation)
                fail_reason = selection_feedback.get("fail_closed_reason")
                if fail_reason is not None:
                    candidate_feedback = {
                        **candidate_feedback,
                        **selection_feedback,
                        "status": "fail_closed",
                        "native_action_kind": native_action.action_kind,
                        "native_object_slot": native_action.object_slot,
                    }
            if fail_reason is None:
                fail_reason = outcome_feedback.get("fail_closed_reason")
            if fail_reason is None:
                candidate_score = fallback_candidate_score(
                    action_memory,
                    primitive=primitive,
                    request=request,
                    rank_index=rank_index,
                    recommendation=recommendation,
                    current_display_coord=current_display_coord,
                )
                if (
                    primitive == "activate_flow"
                    and activation_feedback.get("activation_readiness") == "source_edge_discrimination_probe"
                ):
                    candidate_score = {
                        **candidate_score,
                        "score": float(candidate_score.get("score") or 0.0) + 130.0,
                        "source_edge_discrimination_probe_bonus": 130.0,
                        "source_edge_discrimination_probe": True,
                    }
                if candidate_id in forward_model_demoted:
                    candidate_score = {
                        **candidate_score,
                        "score": float(candidate_score.get("score") or 0.0) - 1000.0,
                        "forward_model_demoted": True,
                        "forward_model_demotion_penalty": 1000.0,
                    }
                viable.append(
                    {
                        "action_id": candidate_id,
                        "native_action": native_action,
                        "primitive": primitive,
                        "recommendation": recommendation,
                        "fail_reason": None,
                        "feedback": feedback,
                        "candidate_feedback": candidate_feedback,
                        "selection_feedback": selection_feedback,
                        "selection_budget_feedback": selection_budget,
                        "activation_feedback": activation_feedback,
                        "outcome_history_feedback": outcome_feedback,
                        "candidate_score": candidate_score,
                    }
                )
                continue
            rejected.append(
                {
                    "action_id": candidate_id,
                    "primitive": primitive,
                    "reason": fail_reason,
                    "candidate_feedback": candidate_feedback,
                    "selection_feedback": selection_feedback,
                    "selection_budget_feedback": selection_budget,
                    "activation_feedback": activation_feedback,
                    "outcome_history_feedback": outcome_feedback,
                }
            )
        if viable:
            selected = max(
                viable,
                key=lambda row: (
                    float((row.get("candidate_score") or {}).get("score") or 0.0),
                    -int((row.get("candidate_score") or {}).get("rank_index") or 0),
                ),
            )
            return {
                "action_id": selected["action_id"],
                "native_action": selected["native_action"],
                "primitive": selected["primitive"],
                "recommendation": selected["recommendation"],
                "fail_reason": None,
                "feedback": selected["feedback"],
                "outcome_history_feedback": selected["outcome_history_feedback"],
                "fallback_raw": {
                    "status": "used_ranked_fallback_after_stationary_feedback",
                    "original_action_id": int(original_action_id),
                    "original_fail_reason": original_fail_reason,
                    "original_feedback": original_feedback if isinstance(original_feedback, dict) else {},
                    "selected_action_id": selected["action_id"],
                    "rejected": rejected,
                    "viable_candidates": [
                        {
                            "action_id": row["action_id"],
                            "primitive": row["primitive"],
                            "candidate_score": row["candidate_score"],
                        }
                        for row in viable
                    ],
                    "selected_candidate_score": selected["candidate_score"],
                    "selected_candidate_feedback": selected["candidate_feedback"],
                    "selected_selection_feedback": selected["selection_feedback"],
                    "selected_selection_budget_feedback": selected["selection_budget_feedback"],
                    "selected_activation_feedback": selected["activation_feedback"],
                    "selected_outcome_history_feedback": selected["outcome_history_feedback"],
                },
            }
        if not rejected:
            return None
        source_probe = self._source_diversity_probe_after_exhaustion(
            request=request,
            observation=observation,
            action_memory=action_memory,
            current_display_coord=current_display_coord,
            outcome_ledger=outcome_ledger,
            original_action_id=original_action_id,
            original_fail_reason=original_fail_reason,
            original_feedback=original_feedback,
            rejected=rejected,
        )
        if source_probe is not None:
            return source_probe
        contra_fallback = self._contra_packet_fallback_after_exhaustion(
            request=request,
            original_action_id=original_action_id,
            original_fail_reason=original_fail_reason,
            original_feedback=original_feedback,
            rejected=rejected,
        )
        if contra_fallback is not None:
            return contra_fallback
        return {
            "action_id": int(original_action_id),
            "native_action": native_action_from_id(int(original_action_id)),
            "primitive": None,
            "recommendation": None,
            "fail_reason": RANKED_FALLBACK_EXHAUSTED_REASON,
            "feedback": {
                "status": "fail_closed",
                "primitive": None,
                "original_feedback": original_feedback if isinstance(original_feedback, dict) else {},
                "fail_closed_reason": RANKED_FALLBACK_EXHAUSTED_REASON,
                "boundary": "ranked fallback exhaustion records missing representation and is not solve or control truth",
            },
            "outcome_history_feedback": {
                "status": "not_applicable",
                "primitive": None,
                "fail_closed_reason": None,
            },
            "fallback_raw": {
                "status": "ranked_fallback_exhausted",
                "original_action_id": int(original_action_id),
                "original_fail_reason": original_fail_reason,
                "original_feedback": original_feedback if isinstance(original_feedback, dict) else {},
                "selected_action_id": None,
                "rejected": rejected,
                "viable_candidates": [],
                "boundary": "ranked fallback exhaustion records missing representation and is not solve or control truth",
            },
        }

    def _contra_packet_fallback_after_exhaustion(
        self,
        *,
        request: dict[str, Any],
        original_action_id: int,
        original_fail_reason: str | None,
        original_feedback: dict[str, Any] | None,
        rejected: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        packet = request.get("contra_recommendation_packet")
        rows = packet.get("ranked_recommendations") if isinstance(packet, dict) else []
        if not isinstance(rows, list):
            return None
        allowed = {
            "select_object",
            "move_up",
            "move_down",
            "move_left",
            "move_right",
            "activate_flow",
            "no_selected_motion",
        }
        for rank_index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            primitive = row.get("primitive")
            if primitive not in allowed:
                continue
            recommendation = dict(row)
            if primitive == "select_object":
                target_override = _target_override_from_row(recommendation)
                if target_override is None:
                    continue
                recommendation["target_override"] = target_override
            recommendation["candidate_kind"] = "paint_control_skill_contra_packet_fallback"
            recommendation["source_candidate_kind"] = row.get("candidate_kind")
            recommendation["native_action_id"] = -2
            native_action = NativeAction(-2, None, str(primitive), "contra_packet_fallback")
            return {
                "action_id": -2,
                "native_action": native_action,
                "primitive": str(primitive),
                "recommendation": recommendation,
                "fail_reason": None,
                "feedback": {
                    "status": "observed",
                    "primitive": str(primitive),
                    "fail_closed_reason": None,
                    "fallback_source": "contra_recommendation_packet",
                    "boundary": "contra_packet_fallback_after_ranked_paint_skill_exhaustion_not_control_truth_or_solve_truth",
                },
                "outcome_history_feedback": {
                    "status": "not_applicable",
                    "primitive": str(primitive),
                    "fail_closed_reason": None,
                },
                "fallback_raw": {
                    "status": "used_contra_packet_fallback_after_ranked_fallback_exhausted",
                    "original_action_id": int(original_action_id),
                    "original_fail_reason": original_fail_reason,
                    "original_feedback": original_feedback if isinstance(original_feedback, dict) else {},
                    "selected_action_id": -2,
                    "selected_rank_index": int(rank_index),
                    "selected_primitive": str(primitive),
                    "source_candidate_kind": row.get("candidate_kind"),
                    "rejected": rejected,
                    "boundary": "uses_live_CONTRA_recommendation_packet_when_paint_skill_native_ranker_has_no_viable_action",
                },
            }
        return None

    def _source_diversity_probe_after_exhaustion(
        self,
        *,
        request: dict[str, Any],
        observation: SkillObservation,
        action_memory: list[dict[str, Any]],
        current_display_coord: list[int] | None,
        outcome_ledger: dict[str, Any] | None,
        original_action_id: int,
        original_fail_reason: str | None,
        original_feedback: dict[str, Any] | None,
        rejected: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        rejected_source_keys: set[str] = set()
        for row in rejected:
            if not isinstance(row, dict) or row.get("primitive") != "select_object":
                continue
            selection_feedback = row.get("selection_feedback") if isinstance(row.get("selection_feedback"), dict) else {}
            source_key = selection_feedback.get("candidate_source_key")
            if isinstance(source_key, str) and source_key:
                rejected_source_keys.add(source_key)
                continue
            outcome = row.get("outcome_history_feedback") if isinstance(row.get("outcome_history_feedback"), dict) else {}
            candidate_keys = outcome.get("candidate_keys")
            if isinstance(candidate_keys, list):
                for key in candidate_keys:
                    if isinstance(key, str) and key.startswith("source_component:"):
                        rejected_source_keys.add(key)
        if not rejected_source_keys:
            return None

        packet = request.get("contra_recommendation_packet")
        rows = packet.get("ranked_recommendations") if isinstance(packet, dict) else []
        if not isinstance(rows, list):
            return None
        viable: list[dict[str, Any]] = []
        seen_source_keys: set[str] = set()
        for rank_index, recommendation in enumerate(rows):
            if not isinstance(recommendation, dict) or recommendation.get("primitive") != "select_object":
                continue
            source_key = _portband_source_key(recommendation)
            if not source_key or source_key in rejected_source_keys or source_key in seen_source_keys:
                continue
            seen_source_keys.add(source_key)
            target_override = _target_override_from_row(recommendation)
            if target_override is None:
                continue
            recommendation = {
                **recommendation,
                "primitive": "select_object",
                "target_override": target_override,
                "candidate_kind": "paint_control_skill_source_diversity_probe",
                "source_candidate_kind": recommendation.get("candidate_kind"),
                "native_action_id": -1,
                "object_slot": None,
            }
            selection_feedback = selection_lock_feedback(
                action_memory,
                primitive="select_object",
                recommendation=recommendation,
            )
            selection_budget = selection_budget_feedback(
                primitive="select_object",
                recommendation=recommendation,
                request=request,
            )
            authority_feedback = _selection_fallback_authority(recommendation)
            outcome_feedback = outcome_history_feedback(
                outcome_ledger,
                primitive="select_object",
                recommendation=recommendation,
                current_display_coord=current_display_coord,
                observation=observation,
                selected_state=_request_selected_state(request),
            )
            fail_reason = (
                selection_feedback.get("fail_closed_reason")
                or selection_budget.get("fail_closed_reason")
                or authority_feedback.get("fail_closed_reason")
                or outcome_feedback.get("fail_closed_reason")
            )
            if fail_reason is not None:
                continue
            candidate_score = fallback_candidate_score(
                action_memory,
                primitive="select_object",
                request=request,
                rank_index=rank_index,
                recommendation=recommendation,
                current_display_coord=current_display_coord,
            )
            candidate_score = {
                **candidate_score,
                "source_diversity_probe": True,
                "source_key": source_key,
                "ranked_fallback_rejected_source_keys": sorted(rejected_source_keys),
            }
            viable.append(
                {
                    "source_key": source_key,
                    "recommendation": recommendation,
                    "selection_feedback": selection_feedback,
                    "selection_budget_feedback": selection_budget,
                    "selection_authority_feedback": authority_feedback,
                    "outcome_history_feedback": outcome_feedback,
                    "candidate_score": candidate_score,
                }
            )
        if not viable:
            return None
        selected = max(
            viable,
            key=lambda row: (
                float((row.get("candidate_score") or {}).get("score") or 0.0),
                float((row.get("recommendation") or {}).get("value_score") or 0.0),
                -int((row.get("candidate_score") or {}).get("rank_index") or 0),
            ),
        )
        native_action = NativeAction(-1, None, "select_object", "source_diversity_probe")
        return {
            "action_id": -1,
            "native_action": native_action,
            "primitive": "select_object",
            "recommendation": selected["recommendation"],
            "fail_reason": None,
            "feedback": {
                "status": "observed",
                "primitive": "select_object",
                "source_diversity_probe": True,
                "selected_source_key": selected["source_key"],
                "fail_closed_reason": None,
                "boundary": "source_diversity_probe_uses_lower_ranked_portband_source_after_ranked_fallback_exhaustion_not_selection_truth",
            },
            "outcome_history_feedback": selected["outcome_history_feedback"],
            "fallback_raw": {
                "status": "used_source_diversity_probe_after_ranked_fallback_exhausted",
                "original_action_id": int(original_action_id),
                "original_fail_reason": original_fail_reason,
                "original_feedback": original_feedback if isinstance(original_feedback, dict) else {},
                "selected_action_id": -1,
                "selected_source_key": selected["source_key"],
                "rejected": rejected,
                "viable_candidates": [],
                "source_diversity_candidates": [
                    {
                        "source_key": row["source_key"],
                        "candidate_score": row["candidate_score"],
                        "value_score": row["recommendation"].get("value_score"),
                    }
                    for row in viable
                ],
                "selected_candidate_score": selected["candidate_score"],
                "selected_selection_feedback": selected["selection_feedback"],
                "selected_selection_budget_feedback": selected["selection_budget_feedback"],
                "selected_selection_authority_feedback": selected["selection_authority_feedback"],
                "selected_outcome_history_feedback": selected["outcome_history_feedback"],
                "boundary": "source diversity probe is a bounded fallback after exhausted native slots, not control truth or solve truth",
            },
        }

    def _infer_checkpoint(self, checkpoint: Path, values: list[float]) -> tuple[int, dict[str, Any]]:
        if self.require_native:
            self._require_native_runtime()
        pufferlib_path = self._resolve_pufferlib_path()
        if pufferlib_path is not None and str(pufferlib_path) not in sys.path:
            sys.path.insert(0, str(pufferlib_path))
        import torch  # noqa: PLC0415

        try:
            return self._infer_torch_state_dict(checkpoint, values, torch)
        except Exception as state_exc:  # noqa: BLE001 - raw native checkpoints are not torch pickles.
            try:
                return self._infer_raw_native_checkpoint(checkpoint, values, torch, state_error=state_exc)
            except Exception:
                raise state_exc

    def _infer_torch_state_dict(self, checkpoint: Path, values: list[float], torch: Any) -> tuple[int, dict[str, Any]]:
        import pufferlib.models as models  # noqa: PLC0415

        config = self._load_puffer_config(checkpoint)
        policy_cfg = config.get("policy") if isinstance(config.get("policy"), dict) else {}
        torch_cfg = config.get("torch") if isinstance(config.get("torch"), dict) else {}
        hidden_size = int(policy_cfg.get("hidden_size") or 96)
        network_name = str(torch_cfg.get("network") or "MinGRU")
        encoder_name = str(torch_cfg.get("encoder") or "DefaultEncoder")
        decoder_name = str(torch_cfg.get("decoder") or "DefaultDecoder")
        network_cls = getattr(models, network_name)
        encoder_cls = getattr(models, encoder_name)
        decoder_cls = getattr(models, decoder_name)
        policy = models.Policy(
            encoder_cls(OBSERVATION_SIZE, hidden_size),
            decoder_cls([16], hidden_size),
            network_cls(**policy_cfg),
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state_dict = {str(key).replace("module.", ""): value for key, value in state_dict.items()}
        policy.load_state_dict(state_dict)
        policy.eval()
        obs = torch.tensor([values], dtype=torch.float32)
        state = policy.initial_state(1, "cpu")
        with torch.no_grad():
            logits, value, _state = policy.forward_eval(obs, state)
            probs = torch.softmax(logits, dim=-1)
            action_id = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0, action_id].item())
            value_estimate = float(value.reshape(-1)[0].item())
            ranked_action_ids = [int(idx) for idx in torch.argsort(probs[0], descending=True).tolist()]
        return action_id, {
            "inference_backend": "torch_puffer_policy_state_dict",
            "confidence": confidence,
            "value_estimate": value_estimate,
            "ranked_action_ids": ranked_action_ids,
            "pufferlib_path": display_path(pufferlib_path) if pufferlib_path else None,
        }

    def _infer_raw_native_checkpoint(
        self,
        checkpoint: Path,
        values: list[float],
        torch: Any,
        *,
        state_error: Exception,
    ) -> tuple[int, dict[str, Any]]:
        config = self._load_puffer_config(checkpoint)
        policy_cfg = config.get("policy") if isinstance(config.get("policy"), dict) else {}
        hidden_size = int(policy_cfg.get("hidden_size") or 96)
        num_layers = int(policy_cfg.get("num_layers") or 3)
        action_count = 16
        raw = torch.from_file(str(checkpoint), dtype=torch.float32, size=checkpoint.stat().st_size // 4)

        def aligned_take(offset: int, count: int) -> tuple[Any, int]:
            tensor = raw[offset : offset + count]
            offset += count
            offset = (offset + 7) & ~7
            return tensor, offset

        offset = 0
        encoder, offset = aligned_take(offset, hidden_size * OBSERVATION_SIZE)
        decoder, offset = aligned_take(offset, (action_count + 1) * hidden_size)
        mingru_weights = []
        for _idx in range(num_layers):
            weight, offset = aligned_take(offset, 3 * hidden_size * hidden_size)
            mingru_weights.append(weight.reshape(3 * hidden_size, hidden_size))
        if offset > raw.numel():
            raise ValueError("raw_native_checkpoint_too_short")

        h = torch.tensor(values, dtype=torch.float32).reshape(1, OBSERVATION_SIZE)
        h = h @ encoder.reshape(hidden_size, OBSERVATION_SIZE).T
        for weight in mingru_weights:
            hidden, gate, proj = (h @ weight.T).chunk(3, dim=-1)
            out = torch.lerp(torch.zeros_like(hidden), torch.where(hidden >= 0, hidden + 0.5, hidden.sigmoid()), gate.sigmoid())
            h = proj.sigmoid() * out + (1.0 - proj.sigmoid()) * h
        decoded = h @ decoder.reshape(action_count + 1, hidden_size).T
        logits = decoded[:, :action_count]
        value = decoded[:, action_count:]
        probs = torch.softmax(logits, dim=-1)
        action_id = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0, action_id].item())
        value_estimate = float(value.reshape(-1)[0].item())
        ranked_action_ids = [int(idx) for idx in torch.argsort(probs[0], descending=True).tolist()]
        return action_id, {
            "inference_backend": "raw_native_puffer_weight_forward",
            "confidence": confidence,
            "value_estimate": value_estimate,
            "ranked_action_ids": ranked_action_ids,
            "raw_weight_float_count": int(raw.numel()),
            "raw_weight_consumed_float_count": int(offset),
            "torch_state_dict_load_error": str(state_error).splitlines()[0],
        }

    def _translate(
        self,
        native_action: NativeAction,
        observation: SkillObservation,
    ) -> tuple[str | None, dict[str, Any] | None, str | None]:
        if native_action.action_kind == "invalid":
            return None, None, "invalid_native_action_id"
        if native_action.action_kind == "noop":
            return None, None, "native_noop_has_no_sp80_primitive"
        if native_action.object_slot is None:
            return None, None, "unknown_object_slot"
        slot = observation.object_slots.get(native_action.object_slot)
        if native_action.action_kind == "activate_edge":
            if observation.focus_edge is None or native_action.object_slot != observation.focus_edge:
                return None, None, "invalid_edge_authority"
            return "activate_flow", None, None
        if slot is None or slot.get("known") is not True:
            return None, None, "unknown_object_slot"
        if slot.get("selected") is not True:
            target_override = slot.get("target_override")
            if not isinstance(target_override, dict):
                return None, None, "unknown_object_slot"
            source_recommendation = slot.get("source_recommendation")
            source_recommendation = source_recommendation if isinstance(source_recommendation, dict) else {}
            evidence_sources = source_recommendation.get("evidence_sources")
            evidence_refs = source_recommendation.get("evidence_refs")
            signals = source_recommendation.get("signals")
            recommendation = {
                "primitive": "select_object",
                "candidate_kind": "paint_control_skill_object_selection",
                "target_override": target_override,
                "native_action_id": native_action.action_id,
                "object_slot": native_action.object_slot,
                "source_candidate_kind": source_recommendation.get("candidate_kind"),
                "evidence_sources": evidence_sources if isinstance(evidence_sources, list) else [],
                "evidence_refs": evidence_refs if isinstance(evidence_refs, dict) else {},
                "signals": signals if isinstance(signals, dict) else {},
            }
            return "select_object", recommendation, None
        source_recommendation = slot.get("source_recommendation")
        source_recommendation = source_recommendation if isinstance(source_recommendation, dict) else {}
        source_key = slot.get("portband_source_key")
        if source_key is not None and source_recommendation:
            evidence_sources = source_recommendation.get("evidence_sources")
            evidence_refs = source_recommendation.get("evidence_refs")
            signals = source_recommendation.get("signals")
            recommendation = {
                "primitive": native_action.primitive,
                "candidate_kind": "paint_control_skill_active_portband_source_movement",
                "native_action_id": native_action.action_id,
                "object_slot": native_action.object_slot,
                "active_portband_source_key": source_key,
                "source_candidate_kind": source_recommendation.get("candidate_kind"),
                "evidence_sources": evidence_sources if isinstance(evidence_sources, list) else [],
                "evidence_refs": evidence_refs if isinstance(evidence_refs, dict) else {},
                "signals": signals if isinstance(signals, dict) else {},
            }
            return native_action.primitive, recommendation, None
        return native_action.primitive, None, None

    def _remember_selection_state(self, *, primitive: str | None, recommendation: dict[str, Any] | None) -> None:
        if primitive != "select_object":
            return
        source_key = _portband_source_key(recommendation)
        self.active_portband_source_key = source_key

    def _raw_decision(
        self,
        *,
        observation: SkillObservation,
        action_id: int,
        inference_raw: dict[str, Any],
        native_action: NativeAction,
        primitive: str | None,
        recommendation: dict[str, Any] | None,
        fail_closed_reason: str | None,
        live_feedback: dict[str, Any] | None = None,
        fallback_raw: dict[str, Any] | None = None,
        forward_model_rerank: dict[str, Any] | None = None,
        outcome_history_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = self._base_raw(fail_closed_reason=fail_closed_reason)
        raw.update(
            {
                "observation_shape": [len(observation.values)],
                "observation_hash": sha256_json(observation.values),
                "native_action_id": int(action_id),
                "native_action": {
                    "action_id": native_action.action_id,
                    "object_slot": native_action.object_slot,
                    "action_kind": native_action.action_kind,
                    "primitive": native_action.primitive,
                },
                "translated_primitive": primitive,
                "selected_object_slot": native_action.object_slot,
                "focus_edge": observation.focus_edge,
                "stack_depth": observation.stack_depth,
                "required_count": observation.required_count,
                "object_slot_mapping_status": observation.mapping_status,
                "known_object_slots": sorted(int(key) for key in observation.object_slots),
                "active_portband_source_key": self.active_portband_source_key,
                "target_locations": observation.target_locations,
                "inference": inference_raw,
                "live_feedback": live_feedback or {},
                "forward_model_rerank": forward_model_rerank or {},
                "outcome_history_feedback": outcome_history_feedback or {},
            }
        )
        if recommendation is not None:
            raw["recommendation"] = recommendation
        if fallback_raw is not None:
            raw["fallback"] = fallback_raw
        return raw

    def _base_raw(self, *, fail_closed_reason: str | None) -> dict[str, Any]:
        checkpoint = self.checkpoint_path if self.checkpoint_path is None else (
            self.checkpoint_path if self.checkpoint_path.is_absolute() else ROOT / self.checkpoint_path
        )
        return {
            "schema": DECISION_SCHEMA,
            "skill_id": self.name,
            "policy_env": self.policy_env,
            "checkpoint_path": display_path(checkpoint) if checkpoint is not None else None,
            "checkpoint_sha256": safe_sha256_file(checkpoint),
            "pufferlib": self.pufferlib,
            "fail_closed_reason": fail_closed_reason,
            "truth_boundary": "paint_control_skill_is_action_authority_for_this_mechanic_only_inside_controlled_sp80_harness",
        }

    def _resolve_pufferlib_path(self) -> Path | None:
        if self.pufferlib_path is not None:
            return self.pufferlib_path
        for path in DEFAULT_PUFFERLIB_PATHS:
            if path.is_dir():
                return path
        return None

    def _require_native_runtime(self) -> None:
        pufferlib_path = self._resolve_pufferlib_path()
        if pufferlib_path is not None and str(pufferlib_path) not in sys.path:
            sys.path.insert(0, str(pufferlib_path))
        pufferl = importlib.import_module("pufferlib.pufferl")
        native = importlib.import_module("pufferlib._C")
        compiled_env = getattr(native, "env_name", None)
        if compiled_env not in (None, self.policy_env):
            raise RuntimeError(f"native_puffer_env_mismatch:{compiled_env}")
        if not hasattr(pufferl, "load_config"):
            raise RuntimeError("native_puffer_missing_load_config")

    @staticmethod
    def _load_puffer_config(checkpoint: Path) -> dict[str, Any]:
        for parent in checkpoint.parents:
            candidate = parent / "resolved_config.json"
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding="utf-8"))
        default = ROOT / "codexlab" / "runs" / "_system_run" / "rexus_sp80_paintflow_l1_multi_native_train_v0_20260611T195551Z" / "resolved_config.json"
        if default.is_file():
            return json.loads(default.read_text(encoding="utf-8"))
        return {
            "policy": {"hidden_size": 96, "num_layers": 3, "expansion_factor": 1},
            "torch": {"network": "MinGRU", "encoder": "DefaultEncoder", "decoder": "DefaultDecoder"},
        }
