#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
from typing import Any

CONTRA_DEBUG_SUMMARY_SCHEMA = "codexlab.contra_debug_summary.v1"
CONTRA_DEBUG_VIEW_SCHEMA = "codexlab.contra_debug_view_summary.v1"


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _stage_map(view: dict[str, Any]) -> dict[str, dict[str, Any]]:
    ladder = view.get("contra_relationship_ladder")
    stages = ladder.get("stages") if isinstance(ladder, dict) else []
    return {
        str(stage.get("stage_name")): stage
        for stage in _as_list(stages)
        if isinstance(stage, dict)
    }


def _stage_status(stage: dict[str, Any] | None) -> str:
    if not isinstance(stage, dict):
        return "missing_stage"
    return str(stage.get("completeness_status") or "unknown")


def _stage_missing(stage: dict[str, Any] | None) -> list[str]:
    if not isinstance(stage, dict):
        return []
    return [str(value) for value in _as_list(stage.get("missing_inputs"))]


def _count_geoband_pass_sets(statecons: list[Any]) -> int:
    return sum(
        1
        for row in statecons
        if isinstance(row, dict) and len(_as_list(row.get("geoband_passes"))) > 0
    )


def _count_boundary_queues(statecons: list[Any]) -> tuple[int, int]:
    queue_count = 0
    boundary_count = 0
    for row in statecons:
        if not isinstance(row, dict):
            continue
        queue = row.get("geoband_boundary_queue")
        if not isinstance(queue, dict):
            continue
        queue_count += 1
        boundary_count += len(_as_list(queue.get("boundaries")))
    return queue_count, boundary_count


def _reason_codes(*, stage_by_name: dict[str, dict[str, Any]], counts: dict[str, int]) -> list[str]:
    reasons: list[str] = []
    if counts["statecons"] == 0:
        reasons.append("no_statecon_rows")
    if counts["geobands"] == 0:
        reasons.append("no_active_geoband_rows")
    for stage_name in (
        "actor_affordances",
        "target_affordances",
        "symmetry_hypotheses",
        "support_action_affordances",
        "state_condition_hypotheses",
        "geoband_hypotheses",
    ):
        stage = stage_by_name.get(stage_name)
        missing = _stage_missing(stage)
        if missing:
            reasons.append(f"{stage_name}_missing_" + "_and_".join(missing))
    return sorted(set(reasons))


def summarize_view(view: dict[str, Any]) -> dict[str, Any]:
    statecons = _as_list(view.get("state_condition_hypotheses"))
    actor_rows = _as_list(view.get("actor_affordance_hypotheses"))
    target_rows = _as_list(view.get("target_affordance_hypotheses"))
    support_rows = _as_list(view.get("support_action_affordance_hypotheses"))
    symmetry_packet = view.get("symmetry_hypotheses")
    colors = _as_list(view.get("colors"))
    scene_surfaces = view.get("scene_surface_hypotheses")
    components = []
    compound_objects = []
    if isinstance(scene_surfaces, dict):
        graph = scene_surfaces.get("component_graph")
        if isinstance(graph, dict):
            components = _as_list(graph.get("components"))
        compound_objects = _as_list(scene_surfaces.get("compound_object_hypotheses"))
    geobands = [
        row.get("geoband_hypothesis")
        for row in statecons
        if isinstance(row, dict) and isinstance(row.get("geoband_hypothesis"), dict)
    ]
    queue_count, boundary_count = _count_boundary_queues(statecons)
    stage_by_name = _stage_map(view)
    counts = {
        "raw_colors": len(colors),
        "raw_components": len(components),
        "compound_objects": len(compound_objects),
        "actors": len(actor_rows),
        "targets": len(target_rows),
        "supports": len(support_rows),
        "symmetry_groups": int(symmetry_packet.get("symmetry_relationship_group_count") or 0)
        if isinstance(symmetry_packet, dict)
        else 0,
        "statecons": len(statecons),
        "geobands": len(geobands),
        "geoband_pass_sets": _count_geoband_pass_sets(statecons),
        "boundary_queues": queue_count,
        "boundaries": boundary_count,
    }
    geoband_stage = stage_by_name.get("geoband_hypotheses")
    statecon_stage = stage_by_name.get("state_condition_hypotheses")
    return {
        "schema": CONTRA_DEBUG_VIEW_SCHEMA,
        "game_id": view.get("game_id"),
        "game_short": view.get("game_short"),
        "title": view.get("title"),
        "level_index": view.get("level_index"),
        "source_raw_frame_sha256": view.get("source_raw_frame_sha256"),
        "counts": counts,
        "stage_status": {
            name: _stage_status(stage_by_name.get(name))
            for name in (
                "raw_color_components",
                "scene_surface_hypotheses",
                "symmetry_hypotheses",
                "actor_affordances",
                "target_affordances",
                "support_action_affordances",
                "state_condition_hypotheses",
                "geoband_hypotheses",
                "deltaband_observations",
            )
        },
        "missing_inputs": {
            name: _stage_missing(stage_by_name.get(name))
            for name in (
                "actor_affordances",
                "symmetry_hypotheses",
                "target_affordances",
                "support_action_affordances",
                "state_condition_hypotheses",
                "geoband_hypotheses",
                "deltaband_observations",
            )
        },
        "contra_ran": _stage_status(stage_by_name.get("raw_color_components")) == "complete",
        "statecon_active": counts["statecons"] > 0 and _stage_status(statecon_stage) == "complete",
        "geoband_reached": isinstance(geoband_stage, dict),
        "geoband_active": counts["geobands"] > 0 and _stage_status(geoband_stage) == "complete",
        "geoband_activation_summary": "active"
        if counts["geobands"] > 0
        else "reached_but_inactive_missing_inputs",
        "reason_codes": _reason_codes(stage_by_name=stage_by_name, counts=counts),
        "artifacts": {
            "raw_board": (view.get("raw_board") or {}).get("path") if isinstance(view.get("raw_board"), dict) else None,
            "scene_surface_overlay": (view.get("scene_surface_overlay") or {}).get("path")
            if isinstance(view.get("scene_surface_overlay"), dict)
            else None,
            "scene_surface_sheet": (view.get("scene_surface_sheet") or {}).get("path")
            if isinstance(view.get("scene_surface_sheet"), dict)
            else None,
            "review_sheet": (view.get("review_sheet") or {}).get("path") if isinstance(view.get("review_sheet"), dict) else None,
        },
    }


def summarize_manifest(manifest: dict[str, Any], *, manifest_path: str | None = None) -> dict[str, Any]:
    views = [view for view in _as_list(manifest.get("views")) if isinstance(view, dict)]
    view_summaries = [summarize_view(view) for view in views]
    reason_counts = Counter(
        reason
        for row in view_summaries
        for reason in _as_list(row.get("reason_codes"))
    )
    active_geoband_count = sum(1 for row in view_summaries if row.get("geoband_active") is True)
    active_statecon_count = sum(1 for row in view_summaries if row.get("statecon_active") is True)
    return {
        "schema": CONTRA_DEBUG_SUMMARY_SCHEMA,
        "source_manifest": manifest_path,
        "source_manifest_schema": manifest.get("schema"),
        "source_status": manifest.get("status"),
        "source_command": manifest.get("command"),
        "view_count": len(view_summaries),
        "statecon_active_view_count": active_statecon_count,
        "geoband_active_view_count": active_geoband_count,
        "geoband_inactive_view_count": len(view_summaries) - active_geoband_count,
        "all_geoband_active": bool(view_summaries) and active_geoband_count == len(view_summaries),
        "reason_counts": dict(sorted(reason_counts.items())),
        "views": view_summaries,
    }
