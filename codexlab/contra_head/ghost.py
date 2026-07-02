#!/usr/bin/env python3
"""Ghost grounded relation memory artifacts.

Ghost records relationship evidence from a controlled completed episode. It
does not record an action route and it is not action authority.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CONTRA_GHOST_SCHEMA = "codexlab.contra_ghost.v1"
CONTRA_GHOST_PASS_SCHEMA = "codexlab.contra_ghost_pass.v1"
CONFIRMED_RELATION_MEMORY_SCHEMA = "codexlab.contra_confirmed_relation_memory.v1"
GHOST_RELATION_SUMMARY_SCHEMA = "codexlab.contra_ghost_relation_summary.v1"

__all__ = [
    "CONTRA_GHOST_SCHEMA",
    "CONTRA_GHOST_PASS_SCHEMA",
    "CONFIRMED_RELATION_MEMORY_SCHEMA",
    "GHOST_RELATION_SUMMARY_SCHEMA",
    "build_ghost",
    "build_confirmed_relation_rows",
    "build_ghost_pass",
    "load_ghost_context",
    "write_ghost",
]

TRUTH_BOUNDARY = {
    "control_truth_admitted": False,
    "goal_truth_admitted": False,
    "goal_condition_admitted": False,
    "solve_truth_admitted": False,
    "solve_condition_admitted": False,
    "action_choice_admitted": False,
}


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def packet_digest(packet: dict[str, Any] | None) -> str:
    if not isinstance(packet, dict):
        return "missing_packet"
    basis = packet.get("packet_sha256_basis")
    if isinstance(basis, str) and basis:
        return basis
    return stable_digest(packet)


def relpath(path: Path, root: Path) -> str:
    resolved = path.resolve()
    root_resolved = root.resolve()
    return resolved.relative_to(root_resolved).as_posix() if resolved.is_relative_to(root_resolved) else str(resolved)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_lineup_matrix_rows(value: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stack = [value]
    while stack and len(rows) < limit:
        current = stack.pop()
        if isinstance(current, dict):
            if current.get("schema") == "codexlab.geoband_portband_lineup_matrix.v1":
                for row in current.get("rows") or []:
                    if isinstance(row, dict):
                        rows.append(row)
                    if len(rows) >= limit:
                        break
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return rows[:limit]


def relation_memory_id(*parts: Any) -> str:
    return "relation_memory_" + hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]


def truth_boundary() -> dict[str, Any]:
    return {
        **TRUTH_BOUNDARY,
        "admission_status": "not_admitted",
        "evidence_boundary": "ghost_memory_not_future_truth",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "route_or_replay_used": False,
        "coordinate_copy_used": False,
        "action_authority": "none",
    }


def selected_relation_from_packet(packet: dict[str, Any]) -> dict[str, Any] | None:
    selected = packet.get("selected_recommendation")
    if not isinstance(selected, dict):
        return None
    sources = selected.get("evidence_sources") or []
    refs = selected.get("evidence_refs") if isinstance(selected.get("evidence_refs"), dict) else {}
    signals = selected.get("signals") if isinstance(selected.get("signals"), dict) else {}
    if "portband.lineup_matrix.rows" not in sources:
        return None
    return {
        "relation_family": "edge_alignment",
        "source_refs": {
            "case_id": packet.get("case_id"),
            "action_index": packet.get("action_index"),
            "lineup_matrix_row_id": refs.get("lineup_matrix_row_id"),
            "lineage_key": refs.get("lineage_key"),
            "portband_route_key": refs.get("portband_route_key"),
            "source_component_ids": refs.get("source_component_ids", []),
            "target_component_ids": refs.get("target_component_ids", []),
            "target_object_ids": refs.get("target_object_ids", []),
            "target_port_ids": refs.get("target_port_ids", []),
        },
        "geometry": {
            "shared_transform_xy": signals.get("transform_hypothesis_xy"),
            "edge_alignment_kind": signals.get("alignment_kind"),
            "flow_axis": "unknown",
            "flow_direction": "unknown",
        },
        "confirmation_basis": "selected_portband_lineup_row_in_controlled_episode",
    }


def build_confirmed_relation_rows(
    *,
    source_run_id: str,
    game_id: str,
    result: dict[str, Any],
    packets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    level_index = int(result.get("level_index") or 0)
    for packet in packets:
        if not isinstance(packet, dict) or packet.get("status") != "executed":
            continue
        relation = selected_relation_from_packet(packet)
        actionmodband = packet.get("actionmodband_observations") if isinstance(packet.get("actionmodband_observations"), dict) else {}
        deltaband = packet.get("deltaband_observations") if isinstance(packet.get("deltaband_observations"), dict) else {}
        if relation is None and int(packet.get("levels_completed_delta") or 0) <= 0:
            continue
        relation_family = relation.get("relation_family") if isinstance(relation, dict) else "activation_survived"
        geometry = relation.get("geometry") if isinstance(relation, dict) else {}
        source_refs = relation.get("source_refs") if isinstance(relation, dict) else {"case_id": packet.get("case_id")}
        round_index = int(packet.get("action_index") or 0)
        completed_delta = int(packet.get("levels_completed_delta") or 0)
        rows.append(
            {
                "schema": CONFIRMED_RELATION_MEMORY_SCHEMA,
                "relation_memory_id": relation_memory_id(source_run_id, level_index, round_index, relation_family),
                "source_run_id": source_run_id,
                "source_packet_ids": [str(packet.get("case_id") or f"episode_action_{round_index:03d}")],
                "game_id": game_id,
                "level_index": level_index,
                "round_index": round_index,
                "relation_family": relation_family,
                "object_role_refs": {
                    "source": source_refs.get("source_component_ids", []),
                    "targets": source_refs.get("target_component_ids", []),
                    "supports": source_refs.get("target_object_ids", []),
                },
                "source_evidence_refs": source_refs,
                "geometry": {
                    "flow_axis": geometry.get("flow_axis", "unknown"),
                    "flow_direction": geometry.get("flow_direction", "unknown"),
                    "edge_alignment_kind": geometry.get("edge_alignment_kind"),
                    "shared_transform_xy": geometry.get("shared_transform_xy"),
                    "angle_error": None,
                    "offset_error": None,
                },
                "confirmation": {
                    "confirmed_by": [
                        name
                        for name, present in (
                            ("deltaband", bool(deltaband)),
                            ("actionmodband", bool(actionmodband)),
                            ("levels_completed_delta", completed_delta > 0),
                        )
                        if present
                    ],
                    "levels_completed_delta": completed_delta,
                    "survived_activation": completed_delta > 0 or packet.get("decision_primitive") == "activate_flow",
                    "confidence_score": 1.0 if completed_delta > 0 else 0.5,
                    "confidence_is_probability": False,
                    "relationship_change_count": int(deltaband.get("relationship_change_count") or 0),
                    "actionmodband_response_count": int(actionmodband.get("response_count") or 0),
                },
                "truth_boundary": truth_boundary(),
            }
        )
    return rows


def build_final_relation_summary(final_packet: dict[str, Any] | None) -> dict[str, Any]:
    rows = collect_lineup_matrix_rows((final_packet or {}).get("after_contra_packet") or {})
    summaries: list[dict[str, Any]] = []
    for row in rows:
        summaries.append(
            {
                "lineup_matrix_row_id": row.get("lineup_matrix_row_id"),
                "source_component_ids": row.get("source_component_ids", []),
                "target_component_ids": row.get("target_component_ids", []),
                "target_object_ids": row.get("target_object_ids", []),
                "target_port_ids": row.get("target_port_ids", []),
                "alignment_kind": row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind"),
                "transform_hypothesis_xy": row.get("transform_hypothesis_xy"),
                "confidence_score": row.get("confidence_score"),
                "effective_lineup_confidence_score": row.get("effective_lineup_confidence_score"),
            }
        )
    return {
        "schema": GHOST_RELATION_SUMMARY_SCHEMA,
        "summary_source": "terminal_after_contra_packet",
        "lineup_matrix_row_count": len(summaries),
        "lineup_matrix_rows": summaries,
        "truth_boundary": truth_boundary(),
    }


def build_action_memory_summary(packets: list[dict[str, Any]]) -> dict[str, Any]:
    primitives = [str(packet.get("decision_primitive") or "") for packet in packets if isinstance(packet, dict)]
    histogram = Counter(value for value in primitives if value)
    terminal = packets[-1] if packets else {}
    return {
        "schema": "codexlab.contra_ghost_action_memory_summary.v1",
        "action_count": len(packets),
        "primitive_histogram": dict(sorted(histogram.items())),
        "terminal_case_id": terminal.get("case_id") if isinstance(terminal, dict) else None,
        "terminal_primitive": terminal.get("decision_primitive") if isinstance(terminal, dict) else None,
        "ordered_route_replay_stored": False,
        "coordinate_script_stored": False,
        "truth_boundary": truth_boundary(),
    }


def build_ghost(
    *,
    result: dict[str, Any],
    packets: list[dict[str, Any]],
    run_dir: Path,
    root: Path,
    game_id: str,
) -> dict[str, Any] | None:
    if int(result.get("levels_completed_delta") or 0) <= 0:
        return None
    final_packet = packets[-1] if packets else {}
    final_frame = final_packet.get("after_frame") if isinstance(final_packet, dict) else {}
    final_contra_packet = final_packet.get("after_contra_packet") if isinstance(final_packet, dict) else {}
    source_run_id = run_dir.name
    ghost = {
        "schema": CONTRA_GHOST_SCHEMA,
        "ghost_id": "ghost_" + stable_digest([source_run_id, str(game_id), result.get("level_index"), result.get("created_at_utc")])[:12],
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "captured_from_controlled_completion",
        "evidence_category": "controlled_run",
        "source_run_id": source_run_id,
        "source_run_dir": relpath(run_dir, root),
        "source_result_json": relpath(run_dir / "result.json", root),
        "source_packets_jsonl": result.get("packets_jsonl"),
        "source_brain_calls_jsonl": result.get("brain_calls_jsonl"),
        "game_id": str(game_id),
        "level_index": int(result.get("level_index") or 0),
        "completion_proof": {
            "source": "controlled_episode_result.levels_completed_delta",
            "levels_completed_delta": int(result.get("levels_completed_delta") or 0),
            "result_status": result.get("status"),
            "terminal_packet_case_id": final_packet.get("case_id") if isinstance(final_packet, dict) else None,
            "terminal_packet_stop_reason": final_packet.get("stop_reason") if isinstance(final_packet, dict) else None,
        },
        "verifier_status": {
            "source_episode_verifier": "required_before_reporting_as_verified",
            "ghost_verifier": "pending",
        },
        "final_raw_frame": final_frame,
        "final_contra_packet_ref": {
            "packet_location": "terminal_packet.after_contra_packet_in_packets_jsonl",
            "terminal_packet_case_id": final_packet.get("case_id") if isinstance(final_packet, dict) else None,
            "packet_hash": packet_digest(final_contra_packet),
        },
        "action_memory_summary": build_action_memory_summary(packets),
        "confirmed_relationship_rows": build_confirmed_relation_rows(
            source_run_id=source_run_id,
            game_id=str(game_id),
            result=result,
            packets=packets,
        ),
        "final_relation_summary": build_final_relation_summary(final_packet if isinstance(final_packet, dict) else None),
        "truth_boundary": truth_boundary(),
    }
    ghost["confirmed_relationship_row_count"] = len(ghost["confirmed_relationship_rows"])
    ghost["ghost_sha256_basis"] = stable_digest(ghost)
    return ghost


def write_ghost(
    *,
    result: dict[str, Any],
    packets: list[dict[str, Any]],
    run_dir: Path,
    root: Path,
    game_id: str,
) -> dict[str, Any] | None:
    ghost = build_ghost(result=result, packets=packets, run_dir=run_dir, root=root, game_id=str(game_id))
    if ghost is None:
        return None
    path = run_dir / "ghost.json"
    write_json(path, ghost)
    return {
        "path": relpath(path, root),
        "schema": ghost["schema"],
        "ghost_id": ghost["ghost_id"],
        "ghost_sha256": stable_digest(ghost),
        "confirmed_relationship_row_count": ghost["confirmed_relationship_row_count"],
        "status": ghost["status"],
    }


def load_ghost_context(*, run_dir: Path, root: Path, game_id: str) -> dict[str, Any]:
    """Load compatible ghost artifacts visible to a harness run directory."""

    candidates: list[dict[str, Any]] = []
    runs_dir = run_dir.parent
    if not runs_dir.is_dir():
        return build_ghost_context(rows=[], loaded_reason="runs_dir_missing")
    for path in sorted(runs_dir.glob("*/ghost.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - loader reports omission, it does not trust bad ghosts.
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("schema") != CONTRA_GHOST_SCHEMA:
            continue
        if str(payload.get("game_id") or "") != str(game_id):
            continue
        if int((payload.get("completion_proof") or {}).get("levels_completed_delta") or 0) <= 0:
            continue
        candidates.append(
            {
                "ghost_id": payload.get("ghost_id"),
                "path": relpath(path, root),
                "source_run_id": payload.get("source_run_id"),
                "game_id": payload.get("game_id"),
                "level_index": payload.get("level_index"),
                "confirmed_relationship_rows": payload.get("confirmed_relationship_rows", []),
                "confirmed_relationship_row_count": int(payload.get("confirmed_relationship_row_count") or 0),
                "truth_boundary": truth_boundary(),
            }
        )
    reason = "compatible_ghosts_loaded" if candidates else "no_compatible_ghosts_found"
    return build_ghost_context(rows=candidates[:4], loaded_reason=reason)


def build_ghost_context(*, rows: list[dict[str, Any]], loaded_reason: str) -> dict[str, Any]:
    return {
        "schema": "codexlab.contra_ghost_context.v1",
        "status": "loaded" if rows else "not_loaded",
        "ghost_count": len(rows),
        "loaded_reason": loaded_reason,
        "ghosts": rows,
        "truth_boundary": truth_boundary(),
    }


def build_ghost_pass(
    *,
    game_id: str,
    level_index: int,
    raw_frame_sha256: str | None,
    ghost_context: dict[str, Any] | None,
    invalid_input: bool = False,
) -> dict[str, Any]:
    ghosts = ghost_context.get("ghosts") if isinstance(ghost_context, dict) else []
    if invalid_input:
        status = "not_available_invalid_raw_frame"
        reason_codes = ["invalid_raw_frame"]
        ghosts = []
    elif not isinstance(ghost_context, dict):
        status = "not_loaded"
        reason_codes = ["ghost_context_missing"]
        ghosts = []
    elif not ghosts:
        status = "not_loaded"
        reason_codes = [str(ghost_context.get("loaded_reason") or "no_compatible_ghosts_found")]
    else:
        status = "relationship_memory_loaded"
        reason_codes = [str(ghost_context.get("loaded_reason") or "compatible_ghosts_loaded")]
    relation_rows: list[dict[str, Any]] = []
    for ghost in ghosts or []:
        if not isinstance(ghost, dict):
            continue
        for row in ghost.get("confirmed_relationship_rows") or []:
            if not isinstance(row, dict):
                continue
            relation_rows.append(
                {
                    "ghost_id": ghost.get("ghost_id"),
                    "source_run_id": ghost.get("source_run_id"),
                    "source_ghost_path": ghost.get("path"),
                    "relation_memory_id": row.get("relation_memory_id"),
                    "relation_family": row.get("relation_family"),
                    "geometry": row.get("geometry", {}),
                    "confirmation": row.get("confirmation", {}),
                    "boundary": "relationship_memory_not_action_authority",
                    "truth_boundary": truth_boundary(),
                }
            )
            if len(relation_rows) >= 24:
                break
        if len(relation_rows) >= 24:
            break
    return {
        "schema": CONTRA_GHOST_PASS_SCHEMA,
        "status": status,
        "source": "contra_view_system_ghost_pass",
        "game_id": str(game_id),
        "level_index": int(level_index),
        "source_raw_frame_sha256": raw_frame_sha256,
        "input_refs": ["ghost_context", "current_contra_packet_relationship_surfaces"],
        "ghost_count": len(ghosts or []),
        "relation_memory_row_count": len(relation_rows),
        "relation_memory_rows": relation_rows,
        "reason_codes": sorted(set(reason_codes)),
        "evidence_role": "ghost_relationship_memory_not_truth_source",
        **truth_boundary(),
    }
