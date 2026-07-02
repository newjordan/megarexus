#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from .common import admission_flags, calibration_id, missing_artifact_fields

MATCH_CALIBRATION_PROBE_SIDECAR_SCHEMA = "codexlab.match_calibration.probe_sidecar_requirement.v1"
MATCH_CALIBRATION_PROBE_SIDECAR_ID = "match_calibration_controlled_probe_sidecar"
MATCH_CALIBRATION_PROBE_SIDECAR_REQUIRED_ARTIFACTS = [
    "contra_api_packet",
    "selected_budget_probe",
    "executor_validation_log",
    "controlled_before_frame",
    "controlled_action_invocation_log",
    "controlled_after_frame",
    "post_action_contra_api_packet",
    "deltaband_observation_packet",
    "calibration_ledger_update",
]


def _probe_queue(contra_packet: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(contra_packet, dict):
        return []
    budget = contra_packet.get("exploreband_budget")
    if not isinstance(budget, dict):
        return []
    rows = budget.get("candidate_probe_queue")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _portband_summary(contra_packet: dict[str, Any] | None) -> dict[str, Any]:
    direct_alignments: list[dict[str, Any]] = []
    fleet_candidates: list[dict[str, Any]] = []
    lineup_rows: list[dict[str, Any]] = []
    if not isinstance(contra_packet, dict):
        return {
            "direct_alignment_count": 0,
            "fleet_candidate_count": 0,
            "lineup_matrix_row_count": 0,
            "direct_alignment_ids": [],
            "fleet_candidate_ids": [],
            "lineup_matrix_row_ids": [],
        }
    for statecon in contra_packet.get("state_condition_hypotheses", []) or []:
        if not isinstance(statecon, dict):
            continue
        portband = statecon.get("portband_hypotheses") if isinstance(statecon.get("portband_hypotheses"), dict) else {}
        direct_alignments.extend([row for row in portband.get("alignments", []) or [] if isinstance(row, dict)])
        fleet = portband.get("portband_fleet") if isinstance(portband.get("portband_fleet"), dict) else {}
        fleet_candidates.extend([row for row in fleet.get("ranked_candidates", []) or [] if isinstance(row, dict)])
        matrix = portband.get("lineup_matrix") if isinstance(portband.get("lineup_matrix"), dict) else {}
        lineup_rows.extend([row for row in matrix.get("rows", []) or [] if isinstance(row, dict)])
    return {
        "direct_alignment_count": len(direct_alignments),
        "fleet_candidate_count": len(fleet_candidates),
        "lineup_matrix_row_count": len(lineup_rows),
        "direct_alignment_ids": [str(row.get("portband_id")) for row in direct_alignments if row.get("portband_id") is not None],
        "fleet_candidate_ids": [str(row.get("fleet_candidate_id")) for row in fleet_candidates if row.get("fleet_candidate_id") is not None],
        "lineup_matrix_row_ids": [str(row.get("lineup_matrix_row_id")) for row in lineup_rows if row.get("lineup_matrix_row_id") is not None],
        "lineup_confidence_model": "static_portband_lineup_score_not_probability",
        "portband_contract": "read_only_contra_api_packet_evidence_no_sidecar_recompute_no_action_authority",
    }


def build_probe_sidecar_requirement(
    *,
    contra_packet: dict[str, Any],
    source_packet_path: str | None = None,
    level_index: int | None = None,
) -> dict[str, Any]:
    """Build a calibration-sidecar requirement from an existing CONTRA API packet.

    This sidecar is deployment plumbing. It does not own PortBand, PortFleet,
    ExploreBand, GeoBand, Oracle, or DeltaBand semantics.
    """

    packet_level = contra_packet.get("level_index") if isinstance(contra_packet, dict) else None
    resolved_level = int(level_index if level_index is not None else packet_level or 0)
    source_sha = str((contra_packet or {}).get("source_raw_frame_sha256") or "unknown")
    queue = _probe_queue(contra_packet)
    selected = queue[0] if queue else None
    selected_probe_id = selected.get("probe_id") if isinstance(selected, dict) else None
    budget = contra_packet.get("exploreband_budget") if isinstance(contra_packet.get("exploreband_budget"), dict) else {}
    return {
        "schema": MATCH_CALIBRATION_PROBE_SIDECAR_SCHEMA,
        "stage_id": MATCH_CALIBRATION_PROBE_SIDECAR_ID,
        "calibration_id": calibration_id("matchcal_probe_sidecar", resolved_level, source_sha, selected_probe_id),
        "status": "not_run_requires_controlled_probe_boundary",
        "calibration_run_status": "not_run",
        "event_kind": "controlled_probe_sidecar_from_contra_api",
        "intent": "spend_budgeted_probe_only_after_executor_validation",
        "level_index": resolved_level,
        "source_raw_frame_sha256": source_sha,
        "contra_api_packet_ref": source_packet_path,
        "contra_api_source": contra_packet.get("source") if isinstance(contra_packet, dict) else None,
        "input_contract": {
            "required_input": "contra.run_contra_view packet",
            "sidecar_may_read": [
                "exploreband_budget",
                "match_calibration_requirements",
                "state_condition_hypotheses.portband_hypotheses",
                "macroband_object_ranking",
                "debug_summary",
            ],
            "sidecar_must_not_call_or_mutate": [
                "portband builders",
                "portfleet builders",
                "geoband builders",
                "oracle builders",
                "exploreband scoring internals",
            ],
        },
        "ownership_boundary": {
            "contra_api_owns_perception_and_static_hypotheses": True,
            "match_calibration_owns_controlled_probe_execution": True,
            "portband_action_authority": "none_observation_only",
            "sidecar_recompute_allowed": False,
            "game_specific_executor_allowed_only_outside_contra_core": True,
        },
        "exploreband_budget_ref": {
            "status": budget.get("status"),
            "candidate_probe_count": budget.get("candidate_probe_count"),
            "selected_probe_id": selected_probe_id,
            "selected_component_id": selected.get("component_id") if isinstance(selected, dict) else None,
            "remaining_probe_count": (budget.get("budget_policy") or {}).get("remaining_probe_count")
            if isinstance(budget.get("budget_policy"), dict)
            else None,
        },
        "portband_read_only_summary": _portband_summary(contra_packet),
        "selected_probe": selected,
        "executor_authorization_status": "not_authorized_requires_sidecar_executor_validation",
        "execution_status": "not_run",
        "ledger_policy": "debit_only_after_controlled_before_action_after_artifacts_exist",
        **missing_artifact_fields(MATCH_CALIBRATION_PROBE_SIDECAR_REQUIRED_ARTIFACTS),
        **admission_flags(),
    }


def build_probe_sidecar_from_contra_api(
    *,
    raw_frame: Any,
    game_id: str,
    level_index: int,
    calibration: dict[str, Any] | None = None,
    mode: str = "match_calibration_probe_sidecar",
) -> dict[str, Any]:
    from ..controller_head import run_contra_view

    contra_packet = run_contra_view(
        raw_frame=raw_frame,
        game_id=str(game_id),
        level_index=int(level_index),
        calibration=calibration,
        mode=str(mode),
    )
    return build_probe_sidecar_requirement(contra_packet=contra_packet, level_index=int(level_index))
