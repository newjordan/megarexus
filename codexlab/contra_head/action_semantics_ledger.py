"""Static ActionSemanticsLedger packet builder."""

from __future__ import annotations

import hashlib
from typing import Any


ACTION_SEMANTICS_LEDGER_SCHEMA = "codexlab.contra_action_semantics_ledger.v1"
ACTION_SEMANTICS_LEDGER_ROW_SCHEMA = "codexlab.contra_action_semantics_ledger_row.v1"

TRUTH_FLAGS = {
    "prediction_status": "not_predictive",
    "admission_status": "not_admitted",
    "not_prediction": True,
    "not_control_truth": True,
    "not_solve_truth": True,
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}

TEACHER_PRIOR_SOURCE = "arc_opus48_scorecard_reasoning_corpus_v0"
TEACHER_PRIOR_BOUNDARY = "public_teacher_reasoning_prior_not_local_contra_evidence"


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _first_bbox(row: dict[str, Any], keys: tuple[str, ...]) -> list[int] | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, list) and len(value) >= 4:
            try:
                return [int(value[0]), int(value[1]), int(value[2]), int(value[3])]
            except (TypeError, ValueError):
                continue
    return None


def _append_unique(values: list[str], value: str | None) -> None:
    if value and value not in values:
        values.append(value)


def _action_family_for_row(row: dict[str, Any], role: str) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "actor_kind",
            "target_kind",
            "support_kind",
            "activation_condition_hypothesis",
            "flow_direction_hypothesis",
            "direction_rule",
        )
    ).lower()
    if "click" in text or "select" in text:
        return "click_or_select_probe"
    if role == "support":
        return "support_surface_probe"
    direction = row.get("flow_direction_hypothesis")
    if direction in {"up", "down", "left", "right"}:
        return "directional_motion_probe"
    return "unknown_action_family_probe"


def _probe_action_names(action_family: str) -> list[str]:
    if action_family == "directional_motion_probe":
        return ["ACTION1", "ACTION2", "ACTION3", "ACTION4"]
    if action_family == "click_or_select_probe":
        return ["ACTION6"]
    if action_family == "support_surface_probe":
        return ["ACTION5", "ACTION6"]
    return ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"]


def _candidate_rows(
    *,
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    support_affordances: list[dict[str, Any]],
    statecons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for role, rows in (
        ("actor", actor_affordances),
        ("target", target_affordances),
        ("support", support_affordances),
    ):
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            component_id = (
                row.get("actor_component_id")
                or row.get("component_id")
                or row.get("source_component_id")
                or row.get("support_component_id")
                or row.get("support_id")
                or f"{role}_{index}"
            )
            candidates.append(
                {
                    "role": role,
                    "source_row_index": index,
                    "component_id": str(component_id),
                    "bbox_xyxy": _first_bbox(
                        row,
                        (
                            "actor_bbox_xyxy",
                            "target_bbox_xyxy",
                            "support_bbox_xyxy",
                            "group_bbox_xyxy",
                            "bbox_xyxy",
                        ),
                    ),
                    "source_status": str(row.get("status") or "hypothesis"),
                    "flow_direction_hypothesis": row.get("flow_direction_hypothesis"),
                    "action_family": _action_family_for_row(row, role),
                }
            )
    if not candidates:
        candidates.append(
            {
                "role": "scene",
                "source_row_index": 0,
                "component_id": "scene_static_probe",
                "bbox_xyxy": None,
                "source_status": "hypothesis",
                "flow_direction_hypothesis": None,
                "action_family": "unknown_action_family_probe",
                "state_condition_count": len([row for row in statecons if isinstance(row, dict)]),
            }
        )
    return candidates


def build_action_semantics_ledger(
    *,
    game_id: str,
    level_index: int,
    source_raw_frame_sha256: str,
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    support_affordances: list[dict[str, Any]],
    statecons: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build static action-family probe hypotheses for a CONTRA packet."""

    candidates = _candidate_rows(
        actor_affordances=actor_affordances,
        target_affordances=target_affordances,
        support_affordances=support_affordances,
        statecons=statecons,
    )
    rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    for rank, candidate in enumerate(candidates, start=1):
        family = str(candidate["action_family"])
        family_counts[family] = family_counts.get(family, 0) + 1
        basis = ["static_contra_affordance_candidate", TEACHER_PRIOR_BOUNDARY]
        if candidate.get("flow_direction_hypothesis"):
            basis.append("directional_affordance_hint")
        if candidate["role"] == "scene":
            basis.append("no_affordance_rows_fallback")
        probe_actions = _probe_action_names(family)
        rows.append(
            {
                "schema": ACTION_SEMANTICS_LEDGER_ROW_SCHEMA,
                "status": "hypothesis_static_probe_recommendation",
                "ledger_row_id": _stable_id(
                    "action_semantics_row",
                    game_id,
                    level_index,
                    source_raw_frame_sha256,
                    candidate["role"],
                    candidate["component_id"],
                    family,
                ),
                "rank": rank,
                "source_role": candidate["role"],
                "source_component_id": candidate["component_id"],
                "source_row_index": candidate["source_row_index"],
                "source_status": candidate["source_status"],
                "bbox_xyxy": candidate.get("bbox_xyxy"),
                "action_family_hypothesis": family,
                "probe_action_names": probe_actions,
                "probe_action_count": len(probe_actions),
                "flow_direction_hypothesis": candidate.get("flow_direction_hypothesis"),
                "teacher_prior_source": TEACHER_PRIOR_SOURCE,
                "teacher_prior_boundary": TEACHER_PRIOR_BOUNDARY,
                "confidence_basis": basis,
                "required_next_local_verifier": "controlled_before_action_after_packet",
                "planning_use": "rank_action_semantics_probes_only",
                "forbidden_use": "action_choice_or_control_truth_or_solve_truth",
                **TRUTH_FLAGS,
            }
        )

    family_order: list[str] = []
    for row in rows:
        _append_unique(family_order, row.get("action_family_hypothesis"))
    return {
        "schema": ACTION_SEMANTICS_LEDGER_SCHEMA,
        "status": "hypothesis_static_probe_ledger",
        "ledger_id": _stable_id("action_semantics_ledger", game_id, level_index, source_raw_frame_sha256, len(rows)),
        "game_id": str(game_id),
        "level_index": int(level_index),
        "source_raw_frame_sha256": source_raw_frame_sha256,
        "truth_status": "hypothesis_static_contra_signal",
        "boundary": "static CONTRA packet plus public teacher prior only; not local action/control evidence",
        "teacher_prior_source": TEACHER_PRIOR_SOURCE,
        "teacher_prior_boundary": TEACHER_PRIOR_BOUNDARY,
        "row_count": len(rows),
        "action_family_hypothesis_order": family_order,
        "action_family_counts": family_counts,
        "rows": rows,
        "next_valid_step": "controlled ActionModBand before/action/after probe",
        "planning_use": "offline_signal_scoring_and_probe_ranking_only",
        "forbidden_use": "gameplay_claims_action_choice_control_truth_goal_truth_prediction_or_solve_truth",
        **TRUTH_FLAGS,
    }


def verify_action_semantics_ledger(packet: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if packet.get("schema") != ACTION_SEMANTICS_LEDGER_SCHEMA:
        failures.append("bad action semantics ledger schema")
    rows = packet.get("rows")
    if not isinstance(rows, list):
        failures.append("ledger rows missing")
        return failures
    if packet.get("row_count") != len(rows):
        failures.append("ledger row_count mismatch")
    for key, expected in TRUTH_FLAGS.items():
        if packet.get(key) != expected:
            failures.append(f"ledger {key} changed")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            failures.append(f"row {index} is not object")
            continue
        if row.get("schema") != ACTION_SEMANTICS_LEDGER_ROW_SCHEMA:
            failures.append(f"row {index} bad schema")
        if not row.get("probe_action_names"):
            failures.append(f"row {index} missing probe actions")
        for key, expected in TRUTH_FLAGS.items():
            if row.get(key) != expected:
                failures.append(f"row {index} {key} changed")
    return failures
