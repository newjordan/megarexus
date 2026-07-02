#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

DELTABAND_OBSERVATION_SCHEMA = "codexlab.deltaband_observation.v1"
DELTABAND_SET_SCHEMA = "codexlab.deltaband_observations.v1"
DELTABAND_SURFACE_RESULT_EVENT_SCHEMA = "codexlab.deltaband.action_surface_result_event.v1"
DELTABAND_CHANGE_MODEL = "structured_geoband_delta_not_scalar"
DELTABAND_REQUIRED_BOUNDARY = "controlled_before_after_action_boundary"

NUMERIC_GEOBAND_PATHS = (
    ("angles", "actor_flow_bearing_degrees"),
    ("angles", "actor_to_target_bearing_degrees"),
    ("angles", "flow_to_target_angular_error_degrees"),
    ("angles", "support_angular_spread_degrees"),
    ("angles", "angular_tolerance_degrees"),
    ("bandwidth", "lateral_bandwidth_pixels"),
    ("bandwidth", "axis_alignment"),
    ("bandwidth", "directional_proximity"),
    ("bandwidth", "symmetry_score"),
)


def stable_deltaband_id(before_id: str, after_id: str, action_invocation_id: str) -> str:
    digest = hashlib.sha1(f"{action_invocation_id}|{before_id}|{after_id}".encode("utf-8")).hexdigest()[:10]
    return f"deltaband_{digest}"


def nested_number(row: dict[str, Any], path: tuple[str, str]) -> float | None:
    parent = row.get(path[0])
    if not isinstance(parent, dict):
        return None
    value = parent.get(path[1])
    if not isinstance(value, (int, float)):
        return None
    return float(value)


def geoband_member_signature(geoband: dict[str, Any]) -> dict[str, Any]:
    members = geoband.get("members")
    if not isinstance(members, dict):
        members = {}
    return {
        "actor_affordance_id": members.get("actor_affordance_id"),
        "target_affordance_id": members.get("target_affordance_id"),
        "support_group_ids": sorted(str(value) for value in members.get("support_group_ids", []) if value is not None),
    }


def extract_geobands(statecons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    geobands: dict[str, dict[str, Any]] = {}
    for statecon in statecons:
        if not isinstance(statecon, dict):
            continue
        geoband = statecon.get("geoband_hypothesis")
        if not isinstance(geoband, dict):
            continue
        statecon_id = geoband.get("statecon_id") or statecon.get("statecon_id")
        if isinstance(statecon_id, str) and statecon_id:
            geobands[statecon_id] = geoband
    return geobands


def _stable_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _as_frame(frame: Any) -> np.ndarray | None:
    if frame is None:
        return None
    arr = np.asarray(frame)
    if arr.ndim < 2:
        return None
    return arr


def _changed_mask(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if before.shape != after.shape:
        return np.zeros(before.shape[:2], dtype=bool)
    changed = before != after
    if changed.ndim == 3:
        return np.any(changed, axis=2)
    return np.asarray(changed, dtype=bool)


def _bbox_from(value: Any) -> list[int] | None:
    if not (isinstance(value, list) and len(value) == 4):
        return None
    return [int(v) for v in value]


def _bbox_edge_cells(box: list[int], shape_hw: tuple[int, int]) -> dict[str, list[list[int]]]:
    h, w = shape_hw
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, x1 = max(0, min(x0, w - 1)), max(0, min(x1, w - 1))
    y0, y1 = max(0, min(y0, h - 1)), max(0, min(y1, h - 1))
    if x1 < x0 or y1 < y0:
        return {"left": [], "right": [], "top": [], "bottom": []}
    return {
        "left": [[x0, y] for y in range(y0, y1 + 1)],
        "right": [[x1, y] for y in range(y0, y1 + 1)],
        "top": [[x, y0] for x in range(x0, x1 + 1)],
        "bottom": [[x, y1] for x in range(x0, x1 + 1)],
    }


def _dilated_edge_mask(edge_cells: dict[str, list[list[int]]], shape_hw: tuple[int, int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    h, w = shape_hw
    full = np.zeros((h, w), dtype=bool)
    by_side: dict[str, np.ndarray] = {}
    for side, cells in edge_cells.items():
        side_mask = np.zeros((h, w), dtype=bool)
        for x, y in cells:
            for nx, ny in ((x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < w and 0 <= ny < h:
                    side_mask[ny, nx] = True
                    full[ny, nx] = True
        by_side[side] = side_mask
    return full, by_side


def _surface_objects_from_statecons(statecons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[int, int, int, int], tuple[str, ...]]] = set()

    def add_surface(*, role: str, source_id: Any, component_ids: Any, bbox: Any, statecon_id: Any) -> None:
        box = _bbox_from(bbox)
        if box is None:
            return
        comps = tuple(sorted(str(value) for value in (component_ids or []) if value is not None))
        key = (role, tuple(box), comps)
        if key in seen:
            return
        seen.add(key)
        rows.append(
            {
                "surface_object_id": _stable_id("surface", role, source_id, box, comps),
                "source_object_id": str(source_id),
                "surface_role": role,
                "source_statecon_id": str(statecon_id),
                "component_ids": list(comps),
                "bbox_xyxy": box,
            }
        )

    for statecon in statecons:
        if not isinstance(statecon, dict):
            continue
        statecon_id = statecon.get("statecon_id")
        for group in statecon.get("support_groups") or []:
            if isinstance(group, dict):
                add_surface(
                    role=str(group.get("support_kind") or "support_surface"),
                    source_id=group.get("support_group_id"),
                    component_ids=group.get("component_ids"),
                    bbox=group.get("group_bbox_xyxy"),
                    statecon_id=statecon_id,
                )
        geoband = statecon.get("geoband_hypothesis") if isinstance(statecon.get("geoband_hypothesis"), dict) else {}
        actor_box = geoband.get("actor_bbox_xyxy") or statecon.get("actor_bbox_xyxy")
        if actor_box is not None:
            add_surface(
                role="actor_surface",
                source_id=geoband.get("actor_id") or statecon.get("actor_affordance_id"),
                component_ids=[geoband.get("actor_component_id")],
                bbox=actor_box,
                statecon_id=statecon_id,
            )
        target_box = geoband.get("target_bbox_xyxy") or statecon.get("target_bbox_xyxy")
        if target_box is not None:
            add_surface(
                role="target_surface",
                source_id=geoband.get("target_id") or statecon.get("target_affordance_id"),
                component_ids=[geoband.get("target_component_id")],
                bbox=target_box,
                statecon_id=statecon_id,
            )
    return rows


def _contact_kind(changed: np.ndarray, edge_cells: dict[str, list[list[int]]], contact_mask: np.ndarray) -> str:
    for cells in edge_cells.values():
        for x, y in cells:
            if 0 <= y < changed.shape[0] and 0 <= x < changed.shape[1] and bool(changed[y, x]):
                return "changed_entity_overlaps_surface_edge"
    if np.count_nonzero(contact_mask) > 0:
        return "changed_entity_adjacent_to_surface_edge"
    return "no_surface_contact"


def build_action_surface_result_events(
    *,
    before_statecons: list[dict[str, Any]],
    before_raw_frame: Any | None,
    action_frame_sequence: list[Any] | None,
    action_invocation_id: str,
    action_primitive: str | None,
    max_events: int = 128,
) -> list[dict[str, Any]]:
    """Detect generic action -> changed entity -> surface contact -> result rows."""

    before = _as_frame(before_raw_frame)
    frames = [_as_frame(frame) for frame in (action_frame_sequence or [])]
    frames = [frame for frame in frames if frame is not None]
    if before is None or not frames:
        return []
    if any(frame.shape != before.shape for frame in frames):
        return []

    shape_hw = before.shape[:2]
    surfaces = _surface_objects_from_statecons(before_statecons)
    final_mask = _changed_mask(before, frames[-1])
    final_changed_count = int(np.count_nonzero(final_mask))
    events: list[dict[str, Any]] = []

    for surface in surfaces:
        box = surface.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        edge_cells = _bbox_edge_cells([int(v) for v in box], shape_hw)
        edge_mask, side_masks = _dilated_edge_mask(edge_cells, shape_hw)
        contact_frames: list[dict[str, Any]] = []
        side_counts: dict[str, int] = {side: 0 for side in edge_cells}
        max_contact_cell_count = 0
        max_changed_count = 0
        for frame_index, frame in enumerate(frames):
            changed = _changed_mask(before, frame)
            changed_count = int(np.count_nonzero(changed))
            contact_mask = changed & edge_mask
            contact_count = int(np.count_nonzero(contact_mask))
            if contact_count <= 0:
                continue
            max_contact_cell_count = max(max_contact_cell_count, contact_count)
            max_changed_count = max(max_changed_count, changed_count)
            frame_side_counts = {side: int(np.count_nonzero(changed & side_mask)) for side, side_mask in side_masks.items()}
            for side, count in frame_side_counts.items():
                side_counts[side] += count
            contact_frames.append(
                {
                    "frame_index": frame_index,
                    "changed_pixel_count": changed_count,
                    "surface_contact_cell_count": contact_count,
                    "contact_kind": _contact_kind(changed, edge_cells, contact_mask),
                    "side_contact_counts": frame_side_counts,
                }
            )
        if not contact_frames:
            continue
        dominant_side = max(side_counts.items(), key=lambda item: (item[1], item[0]))[0]
        confidence = min(1.0, 0.25 + 0.06 * len(contact_frames) + 0.02 * min(max_contact_cell_count, 20))
        first_frame = int(contact_frames[0]["frame_index"])
        last_frame = int(contact_frames[-1]["frame_index"])
        event_id = _stable_id("surface_result", action_invocation_id, surface.get("surface_object_id"), first_frame, last_frame, dominant_side)
        events.append(
            {
                "schema": DELTABAND_SURFACE_RESULT_EVENT_SCHEMA,
                "status": "observation",
                "event_id": event_id,
                "action_invocation_id": action_invocation_id,
                "action_primitive": action_primitive,
                "event_model": "action_to_changed_entity_to_surface_contact_to_result_delta",
                "moving_delta_entity": {
                    "entity_kind": "changed_frame_region",
                    "definition": "pixels_changed_relative_to_pre_action_frame",
                    "frame_sequence_basis": "controlled_action_return_frames",
                    "final_changed_pixel_count": final_changed_count,
                    "max_changed_pixel_count_during_contact": max_changed_count,
                },
                "surface": surface,
                "surface_edge": {
                    "edge_reference_kind": "surface_bbox_edge_cell_set",
                    "dominant_side": dominant_side,
                    "side_contact_counts": side_counts,
                    "edge_cells_xy": edge_cells.get(dominant_side, []),
                },
                "contact": {
                    "first_frame_index": first_frame,
                    "last_frame_index": last_frame,
                    "contact_frame_count": len(contact_frames),
                    "max_surface_contact_cell_count": max_contact_cell_count,
                    "contact_frames": contact_frames[:24],
                },
                "result_delta": {
                    "result_delta_kind": "surface_contact_with_downstream_frame_delta",
                    "final_changed_pixel_count": final_changed_count,
                    "contact_to_result_status": "temporal_contact_result_observed_not_causal_truth",
                },
                "confidence": round(float(confidence), 6),
                "prediction_status": "not_predictive",
                "evidence_status": "observed_action_surface_result_delta",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "solve_condition_admitted": False,
            }
        )
    events.sort(
        key=lambda row: (
            -float(row.get("confidence") or 0.0),
            -int(((row.get("contact") or {}).get("max_surface_contact_cell_count") or 0)),
            str(row.get("event_id")),
        )
    )
    for rank, row in enumerate(events[: max(0, int(max_events))], start=1):
        row["event_rank"] = rank
    return events[: max(0, int(max_events))]


def build_deltaband_observation(
    *,
    before_geoband: dict[str, Any],
    after_geoband: dict[str, Any],
    action_invocation_id: str,
    resource_gauge_numeric_deltas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    before_id = str(before_geoband.get("statecon_id") or before_geoband.get("geoband_id"))
    after_id = str(after_geoband.get("statecon_id") or after_geoband.get("geoband_id"))
    numeric_deltas: list[dict[str, Any]] = []
    changed_fields: list[str] = []
    for path in NUMERIC_GEOBAND_PATHS:
        before_value = nested_number(before_geoband, path)
        after_value = nested_number(after_geoband, path)
        field_path = ".".join(path)
        if before_value is None or after_value is None:
            numeric_deltas.append(
                {
                    "field_path": field_path,
                    "before_value": before_value,
                    "after_value": after_value,
                    "delta": None,
                    "status": "missing_numeric_value",
                }
            )
            changed_fields.append(field_path)
            continue
        delta = round(after_value - before_value, 6)
        if delta != 0.0:
            changed_fields.append(field_path)
        numeric_deltas.append(
            {
                "field_path": field_path,
                "before_value": round(before_value, 6),
                "after_value": round(after_value, 6),
                "delta": delta,
                "status": "observed_numeric_delta",
            }
        )
    for row in resource_gauge_numeric_deltas or []:
        if not isinstance(row, dict):
            continue
        field_path = str(row.get("field_path") or "")
        if not field_path:
            continue
        delta = row.get("delta")
        if isinstance(delta, (int, float)) and float(delta) != 0.0:
            changed_fields.append(field_path)
        numeric_deltas.append(
            {
                "field_path": field_path,
                "before_value": row.get("before_value"),
                "after_value": row.get("after_value"),
                "delta": round(float(delta), 6) if isinstance(delta, (int, float)) else None,
                "status": row.get("status") or "observed_numeric_delta",
                "resource_gauge_kind": row.get("resource_gauge_kind") or "resource_gauge",
            }
        )

    before_members = geoband_member_signature(before_geoband)
    after_members = geoband_member_signature(after_geoband)
    member_changed = before_members != after_members
    if member_changed:
        changed_fields.append("members")
    ray_count_delta = len(after_geoband.get("rays", []) or []) - len(before_geoband.get("rays", []) or [])
    if ray_count_delta != 0:
        changed_fields.append("rays")

    return {
        "schema": DELTABAND_OBSERVATION_SCHEMA,
        "status": "observation",
        "deltaband_id": stable_deltaband_id(before_id, after_id, action_invocation_id),
        "definition": "observed_change_in_geoband_relationship_descriptor",
        "change_model": DELTABAND_CHANGE_MODEL,
        "action_invocation_id": action_invocation_id,
        "before_statecon_id": before_id,
        "after_statecon_id": after_id,
        "before_geoband_id": before_geoband.get("geoband_id"),
        "after_geoband_id": after_geoband.get("geoband_id"),
        "numeric_deltas": numeric_deltas,
        "member_delta": {
            "before_members": before_members,
            "after_members": after_members,
            "changed": member_changed,
        },
        "ray_count_delta": ray_count_delta,
        "relationship_changed": bool(changed_fields),
        "changed_fields": sorted(set(changed_fields)),
        "prediction_status": "not_predictive",
        "evidence_status": "observed_relationship_delta",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_deltaband_observation_set(
    *,
    before_statecons: list[dict[str, Any]],
    after_statecons: list[dict[str, Any]],
    action_invocation_id: str,
    action_primitive: str | None,
    raw_pixel_delta_count: int | None,
    component_delta_count: int | None,
    pre_action_ladder: dict[str, Any] | None = None,
    post_action_ladder: dict[str, Any] | None = None,
    before_raw_frame_sha256: str | None = None,
    after_raw_frame_sha256: str | None = None,
    before_raw_frame: Any | None = None,
    action_frame_sequence: list[Any] | None = None,
    resource_gauge_numeric_deltas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    before_geobands = extract_geobands(before_statecons)
    after_geobands = extract_geobands(after_statecons)
    matched_ids = sorted(set(before_geobands).intersection(after_geobands))
    observations = [
        build_deltaband_observation(
            before_geoband=before_geobands[statecon_id],
            after_geoband=after_geobands[statecon_id],
            action_invocation_id=action_invocation_id,
            resource_gauge_numeric_deltas=resource_gauge_numeric_deltas,
        )
        for statecon_id in matched_ids
    ]
    unmatched_before_ids = sorted(set(before_geobands) - set(after_geobands))
    unmatched_after_ids = sorted(set(after_geobands) - set(before_geobands))
    relationship_change_count = sum(1 for row in observations if row.get("relationship_changed") is True)
    pixel_count = raw_pixel_delta_count if isinstance(raw_pixel_delta_count, int) else None
    component_count = component_delta_count if isinstance(component_delta_count, int) else None
    no_pixel_change = pixel_count == 0 if pixel_count is not None else False
    no_component_change = component_count == 0 if component_count is not None else False
    no_geoband_change = relationship_change_count == 0 and not unmatched_before_ids and not unmatched_after_ids
    null_action_signal = bool(no_pixel_change and no_component_change and no_geoband_change)
    action_surface_result_events = build_action_surface_result_events(
        before_statecons=before_statecons,
        before_raw_frame=before_raw_frame,
        action_frame_sequence=action_frame_sequence,
        action_invocation_id=action_invocation_id,
        action_primitive=action_primitive,
    )
    pre_ladder_id = pre_action_ladder.get("ladder_id") if isinstance(pre_action_ladder, dict) else None
    post_ladder_id = post_action_ladder.get("ladder_id") if isinstance(post_action_ladder, dict) else None
    missing_boundary_inputs = [
        name
        for name, value in (
            ("pre_action_ladder", pre_ladder_id),
            ("action_invocation", action_invocation_id),
            ("post_action_ladder", post_ladder_id),
            ("before_raw_frame_sha256", before_raw_frame_sha256),
            ("after_raw_frame_sha256", after_raw_frame_sha256),
        )
        if not value
    ]

    return {
        "schema": DELTABAND_SET_SCHEMA,
        "status": "observation_set" if not missing_boundary_inputs else "invalid_missing_controlled_boundary_inputs",
        "definition": "observed_change_in_geoband_relationships_across_action_boundary",
        "required_boundary": DELTABAND_REQUIRED_BOUNDARY,
        "change_model": DELTABAND_CHANGE_MODEL,
        "scalar_summary_status": "derived_summary_not_source_truth",
        "prediction_status": "not_predictive",
        "pre_action_ladder_id": pre_ladder_id,
        "post_action_ladder_id": post_ladder_id,
        "action_invocation_id": action_invocation_id,
        "action_primitive": action_primitive,
        "before_raw_frame_sha256": before_raw_frame_sha256,
        "after_raw_frame_sha256": after_raw_frame_sha256,
        "required_inputs": [
            "pre_action_ladder",
            "action_invocation",
            "post_action_ladder",
            "before_raw_frame_sha256",
            "after_raw_frame_sha256",
        ],
        "missing_inputs": missing_boundary_inputs,
        "completeness_status": "complete" if not missing_boundary_inputs else "incomplete_missing_controlled_boundary_inputs",
        "raw_pixel_delta_count": pixel_count,
        "component_delta_count": component_count,
        "matched_geoband_count": len(matched_ids),
        "relationship_change_count": relationship_change_count,
        "resource_gauge_numeric_delta_count": len([row for row in (resource_gauge_numeric_deltas or []) if isinstance(row, dict)]),
        "resource_gauge_field_paths": sorted(
            {
                str(row.get("field_path"))
                for row in (resource_gauge_numeric_deltas or [])
                if isinstance(row, dict) and row.get("field_path")
            }
        ),
        "action_surface_result_event_count": len(action_surface_result_events),
        "action_surface_result_events": action_surface_result_events,
        "surface_contact_result_status": "observed" if action_surface_result_events else "not_observed",
        "unmatched_before_statecon_ids": unmatched_before_ids,
        "unmatched_after_statecon_ids": unmatched_after_ids,
        "null_action_signal": null_action_signal,
        "repeat_null_action_guard": "record_and_block_same_action_in_same_state"
        if null_action_signal
        else "not_applicable",
        "observations": observations,
        "evidence_status": "observed_relationship_delta",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
