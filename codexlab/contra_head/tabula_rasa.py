#!/usr/bin/env python3
"""Tabula-rasa LLM notebook scaffold for CONTRA.

This packet is working memory for a future LLM/teacher loop. It stores
assumptions and revision slots, but it never admits those notes as evidence.
"""

from __future__ import annotations

import hashlib
from typing import Any


TABULA_RASA_NOTEBOOK_SCHEMA = "codexlab.contra_tabula_rasa_notebook.v1"
TABULA_RASA_NOTE_SCHEMA = "codexlab.contra_tabula_rasa_item_note.v1"
TABULA_RASA_ROUND_UPDATE_SCHEMA = "codexlab.contra_tabula_rasa_round_update.v1"

TRUTH_FLAGS = {
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def _truth_guards() -> dict[str, Any]:
    return {
        "evidence_status": "llm_working_memory_not_evidence",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        **TRUTH_FLAGS,
    }


def _round_update_slot(*, round_index: int, basis_ref: str) -> dict[str, Any]:
    return {
        "schema": TABULA_RASA_ROUND_UPDATE_SCHEMA,
        "status": "pending_observation",
        "round_index": int(round_index),
        "basis_ref": str(basis_ref),
        "pre_round_assumption": None,
        "observation_summary": None,
        "post_round_assumption": None,
        "confidence_before": None,
        "confidence_after": None,
        "delta_summary": None,
        "requires_llm_update": True,
        "update_policy": "append_only_after_round_observation",
        **_truth_guards(),
    }


def _note_row(
    *,
    game_id: str,
    level_index: int,
    round_index: int,
    item_kind: str,
    item_id: str,
    assumption: str,
    basis_refs: list[str],
    score_refs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    note_id = _stable_id("tabula_note", game_id, level_index, round_index, item_kind, item_id, basis_refs)
    return {
        "schema": TABULA_RASA_NOTE_SCHEMA,
        "status": "assumption_open",
        "note_id": note_id,
        "game_id": str(game_id),
        "level_index": int(level_index),
        "round_index": int(round_index),
        "item_kind": str(item_kind),
        "item_id": str(item_id),
        "current_assumption": str(assumption),
        "assumption_status": "unconfirmed",
        "basis_refs": [str(value) for value in basis_refs if value],
        "score_refs": score_refs or {},
        "round_updates": [_round_update_slot(round_index=round_index, basis_ref=note_id)],
        "autoresearch_policy": "assume_then_revise_after_each_round",
        "write_policy": "llm_may_append_round_update_only",
        "promotion_policy": "never_promote_notes_to_truth_without_controlled_evidence",
        **_truth_guards(),
    }


def _top_rows(rows: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)][:limit]


def build_tabula_rasa_notebook(
    *,
    game_id: str,
    level_index: int,
    source_raw_frame_sha256: str,
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    support_affordances: list[dict[str, Any]],
    statecons: list[dict[str, Any]],
    round_index: int = 0,
) -> dict[str, Any]:
    """Build a blank per-item assumption notebook for a future LLM loop."""

    notes: list[dict[str, Any]] = []
    for actor in _top_rows(actor_affordances, limit=4):
        actor_id = str(actor.get("actor_id") or actor.get("actor_component_id") or "")
        if actor_id:
            notes.append(
                _note_row(
                    game_id=game_id,
                    level_index=level_index,
                    round_index=round_index,
                    item_kind="actor_candidate",
                    item_id=actor_id,
                    assumption="This item may initiate state change; revise after observed action/result.",
                    basis_refs=[actor_id, str(actor.get("actor_component_id") or "")],
                    score_refs={
                        "actor_option_rank": actor.get("actor_option_rank"),
                        "actor_option_count": actor.get("actor_option_count"),
                        "status": actor.get("status"),
                    },
                )
            )
    for target in _top_rows(target_affordances, limit=4):
        target_id = str(target.get("target_id") or target.get("component_id") or "")
        if target_id:
            notes.append(
                _note_row(
                    game_id=game_id,
                    level_index=level_index,
                    round_index=round_index,
                    item_kind="target_candidate",
                    item_id=target_id,
                    assumption="This item may define success or required destination; revise after observed round outcome.",
                    basis_refs=[target_id, str(target.get("component_id") or "")],
                    score_refs={
                        "goal_option_rank": target.get("goal_option_rank"),
                        "goal_option_count": target.get("goal_option_count"),
                        "open_direction": target.get("open_direction"),
                        "status": target.get("status"),
                    },
                )
            )
    for support in _top_rows(support_affordances, limit=6):
        support_id = str(support.get("support_id") or support.get("component_id") or "")
        if support_id:
            notes.append(
                _note_row(
                    game_id=game_id,
                    level_index=level_index,
                    round_index=round_index,
                    item_kind="support_candidate",
                    item_id=support_id,
                    assumption="This item may mediate control, routing, collision, or transfer; revise after interaction evidence.",
                    basis_refs=[support_id, str(support.get("component_id") or "")],
                    score_refs={
                        "support_kind": support.get("support_kind"),
                        "support_score": support.get("score"),
                        "status": support.get("status"),
                    },
                )
            )
    for statecon in _top_rows(statecons, limit=6):
        statecon_id = str(statecon.get("statecon_id") or "")
        if not statecon_id:
            continue
        geoband = statecon.get("geoband_hypothesis") if isinstance(statecon.get("geoband_hypothesis"), dict) else {}
        members = geoband.get("members") if isinstance(geoband.get("members"), dict) else {}
        notes.append(
            _note_row(
                game_id=game_id,
                level_index=level_index,
                round_index=round_index,
                item_kind="statecon_relation",
                item_id=statecon_id,
                assumption="This relation bundle may explain actor-target-support mechanics; revise after each round delta.",
                basis_refs=[statecon_id],
                score_refs={
                    "same_statecon_score": statecon.get("same_statecon_score"),
                    "event_enable_score": statecon.get("event_enable_score"),
                    "portband_fleet_candidate_count": members.get("portband_fleet_candidate_count"),
                    "portband_alignment_count": members.get("portband_alignment_count"),
                    "portband_lineup_matrix_row_count": (statecon.get("portband_lineup_matrix") or {}).get("row_count")
                    if isinstance(statecon.get("portband_lineup_matrix"), dict)
                    else None,
                },
            )
        )

    return {
        "schema": TABULA_RASA_NOTEBOOK_SCHEMA,
        "status": "ready_for_llm_notes",
        "game_id": str(game_id),
        "level_index": int(level_index),
        "round_index": int(round_index),
        "source_raw_frame_sha256": str(source_raw_frame_sha256),
        "definition": "tabula_rasa_per_item_assumption_notebook_for_autoresearch_loop",
        "note_count": len(notes),
        "notes": notes,
        "autoresearch_loop": {
            "step_1": "write_or_accept_initial_assumption_per_item",
            "step_2": "run_or_observe_next_round",
            "step_3": "append_update_with_observation_and_revised_assumption",
            "step_4": "keep_notes_separate_from_truth_admission",
        },
        "write_policy": "append_only_round_updates_by_note_id",
        "promotion_policy": "notes_can_inform_curiosity_but_cannot_admit_truth",
        **_truth_guards(),
        "provenance": "derived_from_contra_hypothesis_packet_for_future_llm_working_memory",
    }
