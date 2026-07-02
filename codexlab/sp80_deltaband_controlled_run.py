#!/usr/bin/env python3
"""Run controlled SP80 action boundaries and record DeltaBand observations."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import oracle
from contra_head import (
    DELTABAND_REQUIRED_BOUNDARY,
    DELTABAND_SET_SCHEMA,
    build_actionmodband,
    build_episode_memory_after_round,
    build_initial_episode_memory,
    build_contra_perception_pipeline,
    build_deltaband_observation_set,
    build_object_action_response,
    apply_portband_surface_feedback_to_statecons,
    build_tabula_rasa_notebook,
    contra_module_source_paths,
)
from players.sp80.registry import SP80_ACTION_SET
from sp80_base_controller import ROOT, SP80_PATH, SP80BaseController, frame_delta, rgb_frame, save_frame, sha256_file, sha256_frame
from sp80_color_board_viewer import (
    actor_affordance_hypotheses,
    draw_statecon_hypotheses,
    load_calibration_by_level,
    render_boundary_queue_mask,
    render_geoband_primary_mask,
    render_geoband_subgroup_mask,
    render_portband_mask,
    target_affordance_hypotheses,
)
from sp80_object_label_viewer import relpath, resolve_repo_path, write_json
from sp80_sam_hypothesis_viewer import raw_color_inventory


MANIFEST_SCHEMA = "codexlab.sp80_deltaband_controlled_run_manifest.v1"
DEFAULT_LEVELS = [0, 1, 2, 3, 4, 5]


def save_raw_grid(frame: np.ndarray, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame, dtype=np.int16)
    np.save(path, arr)
    return {
        "path": relpath(path),
        "npy_sha256": sha256_file(path),
        "raw_frame_sha256": sha256_frame(arr),
        "shape": [int(value) for value in arr.shape],
        "dtype": str(arr.dtype),
    }


def raw_grid_frame_sha256(frame: np.ndarray) -> str:
    return sha256_frame(np.asarray(frame, dtype=np.int16))


def resource_gauge_numeric_deltas(before_state: dict[str, Any], after_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    before_value = before_state.get("steps_remaining")
    after_value = after_state.get("steps_remaining")
    if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
        rows.append(
            {
                "field_path": "resource.fuel_gauge_steps_remaining",
                "before_value": float(before_value),
                "after_value": float(after_value),
                "delta": round(float(after_value) - float(before_value), 6),
                "status": "observed_numeric_delta",
                "resource_gauge_kind": "fuel_gauge",
                "source_state_key": "steps_remaining",
            }
        )
    return rows


def save_frame_pair(frame: np.ndarray, *, out_dir: Path, name: str, scale: int) -> dict[str, Any]:
    return {
        "png": save_frame(np.asarray(frame), out_dir / "png" / f"{name}.png", scale),
        "raw_grid": save_raw_grid(np.asarray(frame), out_dir / "raw_grid" / f"{name}.npy"),
    }


def save_frame_sequence(frames: list[np.ndarray], *, out_dir: Path, scale: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, frame in enumerate(frames):
        rows.append(save_frame_pair(frame, out_dir=out_dir, name=f"frame_{idx:03d}", scale=scale))
    return rows


def build_perception(frame: np.ndarray, *, level_index: int, calibration: dict[str, Any] | None) -> dict[str, Any]:
    source_raw_frame_sha256 = sha256_frame(frame)
    inventory = raw_color_inventory(frame, min_component_area=1)
    return build_contra_perception_pipeline(
        frame=frame,
        level_index=level_index,
        source_raw_frame_sha256=source_raw_frame_sha256,
        raw_color_inventory=inventory,
        calibration=calibration or {},
        actor_builder=actor_affordance_hypotheses,
        target_builder=target_affordance_hypotheses,
    )


def json_artifact(path: Path) -> dict[str, Any]:
    return {"path": relpath(path), "sha256": sha256_file(path)}


def write_json_artifact(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    write_json(path, payload)
    return json_artifact(path)


def selected_component_id(candidate: dict[str, Any]) -> str | None:
    selected = candidate.get("selected_component") if isinstance(candidate.get("selected_component"), dict) else None
    if not isinstance(selected, dict) or selected.get("component_id") is None:
        return None
    return str(selected.get("component_id"))


def candidate_confidence(candidate: dict[str, Any]) -> dict[str, Any]:
    selected = candidate.get("selected_component") if isinstance(candidate.get("selected_component"), dict) else {}
    selected_id = str(selected.get("component_id")) if selected.get("component_id") is not None else None
    selected_score = float(selected.get("score") or 0.0)
    candidates = [row for row in candidate.get("candidates", []) if isinstance(row, dict)]
    runner_up_score = None
    runner_up_component_id = None
    for row in candidates:
        row_id = str(row.get("component_id")) if row.get("component_id") is not None else None
        if row_id == selected_id:
            continue
        runner_up_score = float(row.get("score") or 0.0)
        runner_up_component_id = row_id
        break
    if runner_up_score is None:
        confidence = 1.0 if selected_id is not None and selected_score > 0.0 else 0.0
    elif selected_score <= 0.0 and runner_up_score <= 0.0:
        confidence = 0.0
    else:
        confidence = selected_score / max(1e-6, selected_score + max(0.0, runner_up_score))
    return {
        "confidence_score": round(float(confidence), 6),
        "confidence_model": "selected_score_share_vs_runner_up_not_probability",
        "confidence_is_probability": False,
        "selected_score": round(float(selected_score), 6),
        "runner_up_score": None if runner_up_score is None else round(float(runner_up_score), 6),
        "score_margin": None if runner_up_score is None else round(float(selected_score - runner_up_score), 6),
        "runner_up_component_id": runner_up_component_id,
    }


def compact_candidate(candidate: dict[str, Any], *, kind: str) -> dict[str, Any]:
    selected = candidate.get("selected_component") if isinstance(candidate.get("selected_component"), dict) else {}
    lock_key_fit = selected.get("lock_key_fit") if isinstance(selected.get("lock_key_fit"), dict) else {}
    criteria = selected.get("criteria") if isinstance(selected.get("criteria"), dict) else {}
    return {
        "kind": kind,
        "status": candidate.get("status"),
        "selected_component_id": selected_component_id(candidate),
        "selected_raw_pixel_value": selected.get("raw_pixel_value"),
        "selected_bbox_xyxy": selected.get("bbox_xyxy"),
        "selected_center_xy": selected.get("center_xy"),
        "criteria": criteria,
        "criteria_count": selected.get("criteria_count"),
        "base_criteria_count": selected.get("base_criteria_count"),
        "lock_key_selected": selected.get("lock_key_selected"),
        "relation_selected": selected.get("relation_selected"),
        "lock_key_shape_fit": lock_key_fit.get("lock_key_shape_fit"),
        "lock_key_fit_score": lock_key_fit.get("fit_score"),
        "lock_key_component_id": lock_key_fit.get("key_component_id"),
        "candidate_count": candidate.get("candidate_count"),
        "top_candidate_component_ids": [
            str(row.get("component_id"))
            for row in candidate.get("candidates", [])
            if isinstance(row, dict) and row.get("component_id") is not None
        ],
        **candidate_confidence(candidate),
    }


def oracle_prediction_from_perception(perception: dict[str, Any], *, phase: str) -> dict[str, Any]:
    scene_surfaces = perception.get("scene_surface_hypotheses")
    relation = oracle.actor_goal_relation_from_contra_scene(scene_surfaces if isinstance(scene_surfaces, dict) else {})
    if not isinstance(relation, dict):
        return {
            "schema": "codexlab.sp80_oracle_goal_prediction_snapshot.v1",
            "phase": phase,
            "status": "no_oracle_actor_goal_relation",
            "confidence_model": "not_available",
            "admission_status": "not_admitted",
            "not_control_truth": True,
            "not_solve_truth": True,
            "confidence_is_probability": False,
        }
    goal_candidate = relation.get("goal_candidate") if isinstance(relation.get("goal_candidate"), dict) else {}
    actor_candidate = relation.get("actor_candidate") if isinstance(relation.get("actor_candidate"), dict) else {}
    arrow = relation.get("key_points_to_lock_hypothesis") if isinstance(relation.get("key_points_to_lock_hypothesis"), dict) else {}
    goal = compact_candidate(goal_candidate, kind="goal")
    actor = compact_candidate(actor_candidate, kind="actor")
    return {
        "schema": "codexlab.sp80_oracle_goal_prediction_snapshot.v1",
        "phase": phase,
        "status": "oracle_actor_goal_relation_observed",
        "relation_schema": relation.get("schema"),
        "relation_source": relation.get("source"),
        "goal_prediction": goal,
        "actor_prediction": actor,
        "key_points_to_lock_hypothesis": {
            "status": arrow.get("status"),
            "goal_lock_component_id": arrow.get("goal_lock_component_id"),
            "actor_key_component_id": arrow.get("actor_key_component_id"),
            "key_to_lock_direction": arrow.get("key_to_lock_direction"),
            "agreement_status": arrow.get("agreement_status"),
            "key_to_lock_distance_pixels": arrow.get("key_to_lock_distance_pixels"),
        },
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "confidence_is_probability": False,
        "confidence_boundary": "Oracle confidence is a rank-score review metric, not a calibrated probability and not goal truth.",
    }


def oracle_prediction_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_goal = before.get("goal_prediction") if isinstance(before.get("goal_prediction"), dict) else {}
    after_goal = after.get("goal_prediction") if isinstance(after.get("goal_prediction"), dict) else {}
    before_actor = before.get("actor_prediction") if isinstance(before.get("actor_prediction"), dict) else {}
    after_actor = after.get("actor_prediction") if isinstance(after.get("actor_prediction"), dict) else {}
    return {
        "schema": "codexlab.sp80_oracle_goal_prediction_delta.v1",
        "status": "computed" if before_goal and after_goal else "missing_oracle_prediction",
        "goal_component_changed": before_goal.get("selected_component_id") != after_goal.get("selected_component_id"),
        "actor_component_changed": before_actor.get("selected_component_id") != after_actor.get("selected_component_id"),
        "before_goal_component_id": before_goal.get("selected_component_id"),
        "after_goal_component_id": after_goal.get("selected_component_id"),
        "before_actor_component_id": before_actor.get("selected_component_id"),
        "after_actor_component_id": after_actor.get("selected_component_id"),
        "goal_confidence_delta": round(
            float(after_goal.get("confidence_score") or 0.0) - float(before_goal.get("confidence_score") or 0.0),
            6,
        ),
        "actor_confidence_delta": round(
            float(after_actor.get("confidence_score") or 0.0) - float(before_actor.get("confidence_score") or 0.0),
            6,
        ),
        "goal_score_delta": round(
            float(after_goal.get("selected_score") or 0.0) - float(before_goal.get("selected_score") or 0.0),
            6,
        ),
        "actor_score_delta": round(
            float(after_actor.get("selected_score") or 0.0) - float(before_actor.get("selected_score") or 0.0),
            6,
        ),
        "confidence_model": before_goal.get("confidence_model") or after_goal.get("confidence_model"),
        "confidence_is_probability": False,
        "admission_status": "not_admitted",
        "not_control_truth": True,
        "not_solve_truth": True,
    }


def overlay_font(scale: int) -> ImageFont.ImageFont:
    size = max(18, int(scale * 1.6))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_labeled_box(draw: ImageDraw.ImageDraw, bbox: Any, *, scale: int, color: tuple[int, int, int], label: str) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return
    x0, y0, x1, y1 = [int(value) * scale for value in bbox]
    x1 += scale - 1
    y1 += scale - 1
    draw.rectangle([x0, y0, x1, y1], outline=color, width=max(3, scale // 2))


def draw_overlay_legend(draw: ImageDraw.ImageDraw, labels: list[tuple[str, tuple[int, int, int]]], *, scale: int) -> None:
    font = overlay_font(scale)
    padding = max(8, scale // 2)
    line_gap = max(6, scale // 3)
    boxes = [draw.textbbox((0, 0), label, font=font) for label, _color in labels]
    width = max((box[2] - box[0] for box in boxes), default=120) + padding * 2
    line_heights = [box[3] - box[1] for box in boxes]
    height = sum(line_heights) + line_gap * max(0, len(labels) - 1) + padding * 2
    draw.rectangle([0, 0, width, height], fill=(8, 10, 14), outline=(228, 238, 247), width=2)
    y = padding
    for idx, (label, color) in enumerate(labels):
        draw.text((padding, y), label, fill=color, font=font)
        y += line_heights[idx] + line_gap


def save_oracle_overlay(frame: np.ndarray, prediction: dict[str, Any], *, out_dir: Path, name: str, scale: int) -> dict[str, Any]:
    path = out_dir / "oracle_overlays" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb_frame(frame, scale), mode="RGB")
    draw = ImageDraw.Draw(image)
    goal = prediction.get("goal_prediction") if isinstance(prediction.get("goal_prediction"), dict) else {}
    actor = prediction.get("actor_prediction") if isinstance(prediction.get("actor_prediction"), dict) else {}
    goal_label = f"goal {goal.get('selected_component_id')} c={goal.get('confidence_score')}"
    actor_label = f"actor {actor.get('selected_component_id')} c={actor.get('confidence_score')}"
    draw_overlay_legend(
        draw,
        [
            (goal_label, (83, 255, 122)),
            (actor_label, (214, 120, 255)),
        ],
        scale=scale,
    )
    draw_labeled_box(draw, goal.get("selected_bbox_xyxy"), scale=scale, color=(83, 255, 122), label=goal_label)
    draw_labeled_box(draw, actor.get("selected_bbox_xyxy"), scale=scale, color=(214, 120, 255), label=actor_label)
    image.save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def save_deltaband_pixel_delta(before_frame: np.ndarray, after_frame: np.ndarray, *, out_dir: Path, scale: int) -> dict[str, Any]:
    path = out_dir / "deltaband_overlays" / "before_after_pixel_delta.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    before = np.asarray(before_frame)
    after = np.asarray(after_frame)
    image = Image.fromarray(rgb_frame(after, scale), mode="RGB")
    draw = ImageDraw.Draw(image)
    changed = before != after
    changed_count = int(np.count_nonzero(changed))
    draw_overlay_legend(
        draw,
        [(f"DeltaBand pixel delta: {changed_count} changed pixels", (255, 96, 96))],
        scale=scale,
    )
    ys, xs = np.nonzero(changed)
    for y, x in zip(ys, xs, strict=False):
        x0 = int(x) * scale
        y0 = int(y) * scale
        draw.rectangle(
            [x0, y0, x0 + scale - 1, y0 + scale - 1],
            outline=(255, 32, 32),
            width=max(3, scale // 2),
        )
    image.save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path), "changed_pixel_count": changed_count}


def save_deltaband_surface_contact_overlay(frame: np.ndarray, events: list[dict[str, Any]], *, out_dir: Path, scale: int) -> dict[str, Any]:
    path = out_dir / "deltaband_overlays" / "action_surface_result_events.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(rgb_frame(frame, scale), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    draw_overlay_legend(
        draw,
        [(f"DeltaBand action-surface-result events: {len(events)}", (255, 211, 88))],
        scale=scale,
    )
    colors = [
        (255, 211, 88, 230),
        (80, 220, 255, 230),
        (255, 120, 180, 230),
        (130, 255, 140, 230),
        (210, 150, 255, 230),
    ]
    font = overlay_font(scale)
    for idx, event in enumerate(events[:24]):
        surface = event.get("surface") if isinstance(event.get("surface"), dict) else {}
        edge = event.get("surface_edge") if isinstance(event.get("surface_edge"), dict) else {}
        box = surface.get("bbox_xyxy")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        color = colors[idx % len(colors)]
        x0, y0, x1, y1 = [int(value) * scale for value in box]
        draw.rectangle([x0, y0, x1 + scale - 1, y1 + scale - 1], outline=color, width=max(2, scale // 2))
        for x, y in (edge.get("edge_cells_xy") or [])[:96]:
            px0, py0 = int(x) * scale, int(y) * scale
            draw.rectangle([px0, py0, px0 + scale - 1, py0 + scale - 1], fill=color)
        label = f"#{event.get('event_rank', idx + 1)} {edge.get('dominant_side')} f{(event.get('contact') or {}).get('first_frame_index')}"
        draw.text((x0 + 2, max(0, y0 - max(10, scale * 2))), label, fill=(255, 255, 255, 255), font=font)
    image.save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path), "event_count": len(events)}


def save_action_sequence_sheet(frames: list[np.ndarray], *, out_dir: Path, scale: int) -> dict[str, Any]:
    path = out_dir / "action_sequence" / "action_sequence_sheet.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_scale = max(4, min(8, scale // 2))
    cols = 6
    label_height = max(24, int(sheet_scale * 4))
    tile_h = int(frames[0].shape[0]) * sheet_scale + label_height if frames else 1
    tile_w = int(frames[0].shape[1]) * sheet_scale if frames else 1
    rows = int(np.ceil(len(frames) / cols)) if frames else 1
    image = Image.new("RGB", (tile_w * cols, tile_h * rows), (8, 10, 14))
    draw = ImageDraw.Draw(image)
    font = overlay_font(sheet_scale)
    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = col * tile_w
        y = row * tile_h
        draw.rectangle([x, y, x + tile_w - 1, y + label_height - 1], fill=(17, 25, 35))
        draw.text((x + 6, y + 3), f"action frame {idx:03d}", fill=(232, 238, 247), font=font)
        tile = Image.fromarray(rgb_frame(frame, sheet_scale), mode="RGB")
        image.paste(tile, (x, y + label_height))
    image.save(path)
    return {
        "path": relpath(path),
        "png_sha256": sha256_file(path),
        "frame_count": len(frames),
        "sheet_scale": sheet_scale,
        "columns": cols,
    }


def save_image_artifact(image: Image.Image, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def save_geoband_artifacts(
    frame: np.ndarray,
    perception: dict[str, Any],
    *,
    out_dir: Path,
    phase: str,
    scale: int,
) -> dict[str, Any]:
    statecons = perception.get("state_condition_hypotheses")
    statecons = statecons if isinstance(statecons, list) else []
    frame_shape = tuple(int(value) for value in np.asarray(frame).shape[:2])
    overlay = Image.fromarray(rgb_frame(frame, scale), mode="RGB")
    draw = ImageDraw.Draw(overlay, "RGBA")
    draw_statecon_hypotheses(draw, statecons, scale=scale)
    primary = render_geoband_primary_mask(frame_shape=frame_shape, statecons=statecons, scale=scale)
    subgroups = render_geoband_subgroup_mask(frame_shape=frame_shape, statecons=statecons, scale=scale)
    boundary_queue = render_boundary_queue_mask(frame_shape=frame_shape, statecons=statecons, scale=scale)
    portband = render_portband_mask(frame_shape=frame_shape, statecons=statecons, scale=scale, frame=frame)
    return {
        "statecon_geoband_overlay": save_image_artifact(
            overlay,
            out_dir / "geoband_overlays" / f"{phase}_statecon_geoband_overlay.png",
        ),
        "primary_geoband_field": save_image_artifact(
            primary,
            out_dir / "geoband_overlays" / f"{phase}_primary_geoband_field.png",
        ),
        "geoband_subgroups": save_image_artifact(
            subgroups,
            out_dir / "geoband_overlays" / f"{phase}_geoband_subgroups.png",
        ),
        "boundary_queue": save_image_artifact(
            boundary_queue,
            out_dir / "geoband_overlays" / f"{phase}_boundary_queue.png",
        ),
        "portband_alignment": save_image_artifact(
            portband,
            out_dir / "geoband_overlays" / f"{phase}_portband_alignment.png",
        ),
        "statecon_count": len(statecons),
        "geoband_count": sum(1 for row in statecons if isinstance(row, dict) and isinstance(row.get("geoband_hypothesis"), dict)),
        "portband_alignment_count": sum(
            len(((row.get("portband_hypotheses") or {}).get("alignments") or []))
            for row in statecons
            if isinstance(row, dict) and isinstance(row.get("portband_hypotheses"), dict)
        ),
    }


def wrap_lines(text: str, *, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
    if current:
        lines.append(current)
    return lines


def save_deltaband_relationship_delta_sheet(
    *,
    before_geoband: dict[str, Any],
    after_geoband: dict[str, Any],
    boundary_delta: dict[str, Any],
    deltaband: dict[str, Any],
    out_dir: Path,
    scale: int,
) -> dict[str, Any]:
    before_path = ROOT / str((before_geoband.get("statecon_geoband_overlay") or {}).get("path"))
    after_path = ROOT / str((after_geoband.get("statecon_geoband_overlay") or {}).get("path"))
    before_image = Image.open(before_path).convert("RGB")
    after_image = Image.open(after_path).convert("RGB")
    width = max(before_image.width, after_image.width)
    before_image = before_image.resize((width, before_image.height), resample=Image.Resampling.NEAREST)
    after_image = after_image.resize((width, after_image.height), resample=Image.Resampling.NEAREST)
    font = overlay_font(scale)
    title_h = max(96, scale * 6)
    body_h = max(before_image.height, after_image.height)
    footer_h = max(120, scale * 8)
    sheet = Image.new("RGB", (width * 2, title_h + body_h + footer_h), (8, 10, 14))
    draw = ImageDraw.Draw(sheet)
    changed_ids = [str(value) for value in boundary_delta.get("changed_geoband_boundary_ids", [])]
    fixed_ids = [str(value) for value in boundary_delta.get("fixed_geoband_boundary_ids", [])]
    title = (
        "DeltaBand relationship delta: "
        f"{len(changed_ids)} changed boundaries, {len(fixed_ids)} fixed, "
        f"{deltaband.get('relationship_change_count')} changed relationship descriptors"
    )
    draw.rectangle([0, 0, sheet.width, title_h - 1], fill=(17, 25, 35))
    y = 10
    for line in wrap_lines(title, max_chars=110):
        draw.text((14, y), line, fill=(255, 232, 163), font=font)
        y += max(24, scale * 2)
    draw.text((14, title_h - 32), "Before GeoBand", fill=(232, 238, 247), font=font)
    draw.text((width + 14, title_h - 32), "After GeoBand", fill=(232, 238, 247), font=font)
    sheet.paste(before_image, (0, title_h))
    sheet.paste(after_image, (width, title_h))
    footer_y = title_h + body_h + 10
    footer_lines = [
        f"changed_geoband_boundary_ids: {', '.join(changed_ids[:12]) if changed_ids else 'none'}",
        f"fixed_geoband_boundary_ids: {', '.join(fixed_ids[:12]) if fixed_ids else 'none'}",
        f"unmatched_before: {len(deltaband.get('unmatched_before_statecon_ids', []) or [])}; unmatched_after: {len(deltaband.get('unmatched_after_statecon_ids', []) or [])}; null_action: {deltaband.get('null_action_signal')}",
    ]
    for line in footer_lines:
        for wrapped in wrap_lines(line, max_chars=150):
            draw.text((14, footer_y), wrapped, fill=(216, 242, 255), font=font)
            footer_y += max(24, scale * 2)
    path = out_dir / "deltaband_overlays" / "relationship_delta_sheet.png"
    return save_image_artifact(sheet, path)


def boundary_rows(statecons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for statecon in statecons:
        if not isinstance(statecon, dict):
            continue
        queue = statecon.get("geoband_boundary_queue")
        boundaries = queue.get("boundaries") if isinstance(queue, dict) else []
        for boundary in boundaries if isinstance(boundaries, list) else []:
            if not isinstance(boundary, dict):
                continue
            boundary_id = boundary.get("boundary_id")
            if isinstance(boundary_id, str) and boundary_id:
                rows[boundary_id] = boundary
    return rows


def stable_boundary_payload(boundary: dict[str, Any]) -> dict[str, Any]:
    return {
        "boundary_type": boundary.get("boundary_type"),
        "status": boundary.get("status"),
        "input_object_ids": sorted(str(value) for value in boundary.get("input_object_ids", [])),
        "related_object_ids": sorted(str(value) for value in boundary.get("related_object_ids", [])),
        "reason_codes": sorted(str(value) for value in boundary.get("reason_codes", [])),
        "suggested_next_pass": boundary.get("suggested_next_pass"),
    }


def compare_boundary_rows(before_statecons: list[dict[str, Any]], after_statecons: list[dict[str, Any]]) -> dict[str, Any]:
    before = boundary_rows(before_statecons)
    after = boundary_rows(after_statecons)
    before_ids = set(before)
    after_ids = set(after)
    fixed_ids = sorted(
        boundary_id
        for boundary_id in before_ids & after_ids
        if stable_boundary_payload(before[boundary_id]) == stable_boundary_payload(after[boundary_id])
    )
    changed_existing_ids = sorted(
        boundary_id
        for boundary_id in before_ids & after_ids
        if stable_boundary_payload(before[boundary_id]) != stable_boundary_payload(after[boundary_id])
    )
    removed_ids = sorted(before_ids - after_ids)
    added_ids = sorted(after_ids - before_ids)
    return {
        "changed_geoband_boundary_ids": sorted(changed_existing_ids + removed_ids + added_ids),
        "fixed_geoband_boundary_ids": fixed_ids,
        "added_geoband_boundary_ids": added_ids,
        "removed_geoband_boundary_ids": removed_ids,
        "changed_existing_geoband_boundary_ids": changed_existing_ids,
        "before_boundary_count": len(before_ids),
        "after_boundary_count": len(after_ids),
    }


def component_count(perception: dict[str, Any]) -> int:
    components = (
        perception.get("scene_surface_hypotheses", {})
        .get("component_graph", {})
        .get("components", [])
    )
    return len(components) if isinstance(components, list) else 0


def run_level(*, level_index: int, out_dir: Path, scale: int, calibration: dict[str, Any] | None) -> dict[str, Any]:
    level_dir = out_dir / f"level_{level_index:02d}"
    controller = SP80BaseController(level_index=level_index)
    game, obs = controller.new_game()
    before_frame = np.asarray(obs.frame[-1])
    before_state = controller.state(game, obs)
    before_perception = build_perception(before_frame, level_index=level_index, calibration=calibration)
    before_oracle_prediction = oracle_prediction_from_perception(before_perception, phase="before_action")
    grounded = controller.ground_primitive(game, "activate_flow")
    after_obs = controller.execute_grounded_action(game, grounded)
    action_frames = [np.asarray(frame) for frame in after_obs.frame]
    after_frame = action_frames[-1]
    after_state = controller.state(game, after_obs)
    after_perception = build_perception(after_frame, level_index=level_index, calibration=calibration)
    after_oracle_prediction = oracle_prediction_from_perception(after_perception, phase="after_action")
    oracle_delta = oracle_prediction_delta(before_oracle_prediction, after_oracle_prediction)

    before_artifact = save_frame_pair(before_frame, out_dir=level_dir, name="before", scale=scale)
    after_artifact = save_frame_pair(after_frame, out_dir=level_dir, name="after", scale=scale)
    action_sequence = save_frame_sequence(action_frames, out_dir=level_dir / "action_sequence", scale=scale)
    action_sequence_sheet = save_action_sequence_sheet(action_frames, out_dir=level_dir, scale=scale)
    deltaband_delta_overlay = save_deltaband_pixel_delta(before_frame, after_frame, out_dir=level_dir, scale=scale)
    before_geoband_artifacts = save_geoband_artifacts(
        before_frame,
        before_perception,
        out_dir=level_dir,
        phase="before",
        scale=scale,
    )
    after_geoband_artifacts = save_geoband_artifacts(
        after_frame,
        after_perception,
        out_dir=level_dir,
        phase="after",
        scale=scale,
    )
    before_oracle_overlay = save_oracle_overlay(
        before_frame,
        before_oracle_prediction,
        out_dir=level_dir,
        name="before_oracle_goal_prediction",
        scale=scale,
    )
    after_oracle_overlay = save_oracle_overlay(
        after_frame,
        after_oracle_prediction,
        out_dir=level_dir,
        name="after_oracle_goal_prediction",
        scale=scale,
    )
    before_packet_artifact = write_json_artifact(level_dir / "before_contra_perception_pipeline.json", {
        key: value for key, value in before_perception.items() if key not in {"component_label_map", "component_lookup"}
    })
    after_packet_artifact = write_json_artifact(level_dir / "after_contra_perception_pipeline.json", {
        key: value for key, value in after_perception.items() if key not in {"component_label_map", "component_lookup"}
    })
    action_log_artifact = write_json_artifact(
        level_dir / "action_invocation_log.json",
        {
            "schema": "codexlab.sp80_controlled_action_invocation_log.v1",
            "action_invocation_id": f"sp80_l{level_index:02d}_activate_flow_action5",
            "action": grounded.packet(),
            "before_state": before_state,
            "after_state": after_state,
            "action_frame_count": len(action_frames),
            "route_or_replay_used": False,
            "human_trace_used": False,
            "coordinate_script_used": False,
        },
    )
    pixel_delta = frame_delta(before_frame, after_frame)
    component_delta_count = abs(component_count(after_perception) - component_count(before_perception))
    action_invocation_id = f"sp80_l{level_index:02d}_activate_flow_action5"
    deltaband = build_deltaband_observation_set(
        before_statecons=before_perception["state_condition_hypotheses"],
        after_statecons=after_perception["state_condition_hypotheses"],
        action_invocation_id=action_invocation_id,
        action_primitive="activate_flow",
        raw_pixel_delta_count=int(pixel_delta.get("changed_pixels") or 0),
        component_delta_count=component_delta_count,
        pre_action_ladder=before_perception["contra_relationship_ladder"],
        post_action_ladder=after_perception["contra_relationship_ladder"],
        before_raw_frame_sha256=raw_grid_frame_sha256(before_frame),
        after_raw_frame_sha256=raw_grid_frame_sha256(after_frame),
        before_raw_frame=before_frame,
        action_frame_sequence=action_frames,
        resource_gauge_numeric_deltas=resource_gauge_numeric_deltas(before_state, after_state),
    )
    boundary_delta = compare_boundary_rows(
        before_perception["state_condition_hypotheses"],
        after_perception["state_condition_hypotheses"],
    )
    deltaband = {
        **deltaband,
        **boundary_delta,
        "action_sequence_frame_count": len(action_frames),
        "before_frame_artifact": before_artifact,
        "after_frame_artifact": after_artifact,
        "after_frame_sequence_artifacts": action_sequence,
    }
    portband_feedback_statecons = apply_portband_surface_feedback_to_statecons(
        statecons=before_perception["state_condition_hypotheses"],
        action_surface_result_events=deltaband.get("action_surface_result_events", [])
        if isinstance(deltaband.get("action_surface_result_events"), list)
        else [],
        action_state_relationship_observations=deltaband.get("observations", [])
        if isinstance(deltaband.get("observations"), list)
        else [],
        action_primitive=deltaband.get("action_primitive"),
    )
    portband_feedback_summary = {
        "schema": "codexlab.sp80_portband_surface_feedback_summary.v1",
        "status": "ranked",
        "statecon_count": len(portband_feedback_statecons),
        "action_surface_result_event_count": int(deltaband.get("action_surface_result_event_count") or 0),
        "matched_alignment_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("surface_feedback_matched_alignment_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "action_state_tag_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("action_state_tag_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "action_state_positive_signal_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("action_state_positive_signal_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "relative_state_change_tag_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("relative_state_change_tag_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "action_interaction_edge_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("action_interaction_edge_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "novel_interaction_edge_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("novel_interaction_edge_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "solve_like_interaction_edge_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("solve_like_interaction_edge_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "action_interaction_kinds": sorted(
            {
                str(kind)
                for row in portband_feedback_statecons
                if isinstance(row, dict)
                for kind in (((row.get("portband_hypotheses") or {}).get("action_interaction_kinds") or []))
                if kind is not None
            }
        ),
        "action_interaction_edge_ids": sorted(
            {
                str(edge_id)
                for row in portband_feedback_statecons
                if isinstance(row, dict)
                for edge_id in (((row.get("portband_hypotheses") or {}).get("action_interaction_edge_ids") or []))
                if edge_id is not None
            }
        ),
        "novel_relationship_candidate_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("novel_relationship_candidate_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "solve_like_evidence_flag_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("solve_like_evidence_flag_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "top_solve_like_evidence_score": max(
            (
                float(((row.get("portband_hypotheses") or {}).get("top_solve_like_evidence_score") or 0.0))
                for row in portband_feedback_statecons
                if isinstance(row, dict)
            ),
            default=0.0,
        ),
        "route_flow_lock_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("route_flow_lock_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "route_flow_positive_signal_count": sum(
            int(((row.get("portband_hypotheses") or {}).get("route_flow_positive_signal_count") or 0))
            for row in portband_feedback_statecons
            if isinstance(row, dict)
        ),
        "top_feedback_alignments": [
            {
                "statecon_id": row.get("statecon_id"),
                "portband_id": alignment.get("portband_id"),
                "feedback_rank": alignment.get("surface_feedback_rank"),
                "feedback_adjusted_score": alignment.get("surface_feedback_adjusted_score"),
                "feedback_dynamic_score": alignment.get("surface_feedback_dynamic_score"),
                "alignment_kind": alignment.get("alignment_kind"),
                "alignment_error_px": alignment.get("alignment_error_px"),
                "source_bboxes_xyxy": alignment.get("source_bboxes_xyxy"),
                "target_bboxes_xyxy": alignment.get("target_bboxes_xyxy"),
                "imagined_source_bboxes_xyxy": alignment.get("imagined_source_bboxes_xyxy"),
                "surface_feedback_event_count": alignment.get("surface_feedback_event_count"),
                "surface_feedback_events": alignment.get("surface_feedback_events"),
                "action_state_tag_status": alignment.get("action_state_tag_status"),
                "action_state_tag_count": alignment.get("action_state_tag_count"),
                "action_state_positive_signal": alignment.get("action_state_positive_signal"),
                "action_state_correlation_score": alignment.get("action_state_correlation_score"),
                "action_state_tag_ids": alignment.get("action_state_tag_ids"),
                "action_state_primitives": alignment.get("action_state_primitives"),
                "action_state_event_ids": alignment.get("action_state_event_ids"),
                "action_state_action_invocation_ids": alignment.get("action_state_action_invocation_ids"),
                "relative_state_change_tag_ids": alignment.get("relative_state_change_tag_ids"),
                "relative_state_change_novelty_statuses": alignment.get("relative_state_change_novelty_statuses"),
                "relative_state_change_signatures": alignment.get("relative_state_change_signatures"),
                "solve_like_evidence_flags": alignment.get("solve_like_evidence_flags"),
                "solve_like_evidence_score": alignment.get("solve_like_evidence_score"),
                "route_flow_lock_status": alignment.get("route_flow_lock_status"),
                "route_flow_positive_signal": alignment.get("route_flow_positive_signal"),
                "route_flow_confidence_score": alignment.get("route_flow_confidence_score"),
                "route_flow_evidence_event_ids": alignment.get("route_flow_evidence_event_ids"),
                "route_flow_action_invocation_ids": alignment.get("route_flow_action_invocation_ids"),
            }
            for row in portband_feedback_statecons
            if isinstance(row, dict)
            for alignment in (((row.get("portband_hypotheses") or {}).get("surface_feedback_adjusted_alignments") or [])[:3])
            if isinstance(alignment, dict)
        ][:24],
        "top_action_state_tags": [
            {
                "statecon_id": row.get("statecon_id"),
                "tag_id": tag.get("action_state_tag_id"),
                "tag_kind": tag.get("tag_kind"),
                "tag_scope": tag.get("tag_scope"),
                "action_primitive": tag.get("action_primitive"),
                "correlation_score": tag.get("correlation_score"),
                "novelty_status": tag.get("novelty_status"),
                "solve_like_evidence_flag": tag.get("solve_like_evidence_flag"),
                "solve_like_evidence_score": tag.get("solve_like_evidence_score"),
                "mathematical_relationships": tag.get("mathematical_relationships"),
                "changed_fields": tag.get("changed_fields"),
            }
            for row in portband_feedback_statecons
            if isinstance(row, dict)
            for tag in ((row.get("portband_hypotheses") or {}).get("action_state_tags") or [])[:6]
            if isinstance(tag, dict)
        ][:24],
        "top_action_interaction_edges": [
            {
                "statecon_id": row.get("statecon_id"),
                "edge_id": edge.get("interaction_edge_id"),
                "interaction_rank": edge.get("interaction_rank"),
                "interaction_kind": edge.get("interaction_kind"),
                "action_primitive": edge.get("action_primitive"),
                "from_object_ids": edge.get("from_object_ids"),
                "to_object_ids": edge.get("to_object_ids"),
                "participant_object_ids": edge.get("participant_object_ids"),
                "relationship_signature": edge.get("relationship_signature"),
                "relation_type_label": edge.get("relation_type_label"),
                "relationship_profile": edge.get("relationship_profile"),
                "novelty_status": edge.get("novelty_status"),
                "solve_like_evidence_flag": edge.get("solve_like_evidence_flag"),
                "solve_like_evidence_score": edge.get("solve_like_evidence_score"),
            }
            for row in portband_feedback_statecons
            if isinstance(row, dict)
            for edge in ((((row.get("portband_hypotheses") or {}).get("action_interaction_graph") or {}).get("edges") or [])[:6])
            if isinstance(edge, dict)
        ][:24],
        "feedback_policy": "fail_safe_delta_surface_result_promotes_portband_ranking_and_route_flow_signal_original_static_alignments_preserved",
        "action_state_tagger_policy": "tag_controlled_action_observations_to_candidate_objects_lineups_and_relative_state_changes",
        "action_state_tagger_boundary": "action-state tags are candidate object/action/state correlations only, not control truth, goal truth, action choice, solve evidence, or prediction",
        "route_flow_lock_policy": "single_controlled_activate_flow_surface_event_can_lock_route_flow_positive_signal",
        "route_flow_lock_boundary": "route-flow signal only, not control truth, goal truth, action choice, solve evidence, or prediction",
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
    deltaband["portband_surface_feedback_summary"] = portband_feedback_summary
    deltaband["portband_action_state_tag_count"] = portband_feedback_summary["action_state_tag_count"]
    deltaband["portband_action_state_positive_signal_count"] = portband_feedback_summary["action_state_positive_signal_count"]
    deltaband["portband_relative_state_change_tag_count"] = portband_feedback_summary["relative_state_change_tag_count"]
    deltaband["portband_action_interaction_edge_count"] = portband_feedback_summary["action_interaction_edge_count"]
    deltaband["portband_novel_relationship_candidate_count"] = portband_feedback_summary["novel_relationship_candidate_count"]
    deltaband["portband_solve_like_evidence_flag_count"] = portband_feedback_summary["solve_like_evidence_flag_count"]
    deltaband["portband_route_flow_lock_count"] = portband_feedback_summary["route_flow_lock_count"]
    deltaband["portband_route_flow_positive_signal_count"] = portband_feedback_summary["route_flow_positive_signal_count"]
    deltaband_surface_contact_overlay = save_deltaband_surface_contact_overlay(
        before_frame,
        deltaband.get("action_surface_result_events", []) if isinstance(deltaband.get("action_surface_result_events"), list) else [],
        out_dir=level_dir,
        scale=scale,
    )
    portband_surface_feedback_overlay = save_image_artifact(
        render_portband_mask(
            frame_shape=before_frame.shape[:2],
            statecons=portband_feedback_statecons,
            scale=scale,
            frame=before_frame,
        ),
        level_dir / "geoband_overlays" / "before_portband_surface_feedback.png",
    )
    portband_surface_feedback_artifact = write_json_artifact(
        level_dir / "portband_surface_feedback.json",
        portband_feedback_summary,
    )
    levels_completed_delta = int(after_state.get("levels_completed") or 0) - int(before_state.get("levels_completed") or 0)
    action_response = build_object_action_response(
        before_state=before_state,
        after_state=after_state,
        action_taken=grounded.packet(),
        frame_delta=pixel_delta,
        returned_frame_count=len(action_frames),
        action_invocation_id=action_invocation_id,
        action_index=0,
        downstream_result={
            "levels_completed_delta": levels_completed_delta,
            "action_frame_count": len(action_frames),
            "raw_pixel_delta_count": int(pixel_delta.get("changed_pixels") or 0),
            "deltaband_relationship_change_count": int(deltaband.get("relationship_change_count") or 0),
        },
    )
    actionmodband = build_actionmodband(
        action_responses=[action_response],
        action_invocation_id=action_invocation_id,
        levels_completed_delta=levels_completed_delta,
    )
    before_tabula = build_tabula_rasa_notebook(
        game_id="sp80-589a99af",
        level_index=int(level_index),
        source_raw_frame_sha256=raw_grid_frame_sha256(before_frame),
        actor_affordances=before_perception.get("actor_affordance_hypotheses", []),
        target_affordances=before_perception.get("target_affordance_hypotheses", []),
        support_affordances=before_perception.get("support_action_affordance_hypotheses", []),
        statecons=before_perception.get("state_condition_hypotheses", []),
        round_index=0,
    )
    after_tabula = build_tabula_rasa_notebook(
        game_id="sp80-589a99af",
        level_index=int(level_index),
        source_raw_frame_sha256=raw_grid_frame_sha256(after_frame),
        actor_affordances=after_perception.get("actor_affordance_hypotheses", []),
        target_affordances=after_perception.get("target_affordance_hypotheses", []),
        support_affordances=after_perception.get("support_action_affordance_hypotheses", []),
        statecons=after_perception.get("state_condition_hypotheses", []),
        round_index=1,
    )
    before_contra_packet = {
        "packet_sha256_basis": hashlib.sha256(f"sp80-589a99af|{level_index}|before|{raw_grid_frame_sha256(before_frame)}".encode("utf-8")).hexdigest(),
        "game_id": "sp80-589a99af",
        "level_index": int(level_index),
        "source_raw_frame_sha256": raw_grid_frame_sha256(before_frame),
        "state_condition_hypotheses": before_perception.get("state_condition_hypotheses", []),
        "tabula_rasa_notebook": before_tabula,
    }
    after_contra_packet = {
        "packet_sha256_basis": hashlib.sha256(f"sp80-589a99af|{level_index}|after|{raw_grid_frame_sha256(after_frame)}".encode("utf-8")).hexdigest(),
        "game_id": "sp80-589a99af",
        "level_index": int(level_index),
        "source_raw_frame_sha256": raw_grid_frame_sha256(after_frame),
        "state_condition_hypotheses": after_perception.get("state_condition_hypotheses", []),
        "tabula_rasa_notebook": after_tabula,
    }
    before_episode_memory = build_initial_episode_memory(
        game_id="sp80-589a99af",
        level_index=int(level_index),
        contra_packet=before_contra_packet,
    )
    episode_memory_after_round = build_episode_memory_after_round(
        game_id="sp80-589a99af",
        level_index=int(level_index),
        round_index=1,
        previous_memory=before_episode_memory,
        before_contra_packet=before_contra_packet,
        after_contra_packet=after_contra_packet,
        action_taken=grounded.packet(),
        deltaband_result=deltaband,
        actionmodband_result=actionmodband,
    )
    deltaband_relationship_delta_sheet = save_deltaband_relationship_delta_sheet(
        before_geoband=before_geoband_artifacts,
        after_geoband=after_geoband_artifacts,
        boundary_delta=boundary_delta,
        deltaband=deltaband,
        out_dir=level_dir,
        scale=scale,
    )
    row = {
        "schema": "codexlab.sp80_deltaband_controlled_level_run.v1",
        "status": "controlled",
        "level_index": int(level_index),
        "condition_name": "sp80_activate_flow_deltaband_controlled_boundary",
        "condition_natural_language": "Run one real SP80 activate_flow action and compare before/after CONTRA GeoBand boundary rows.",
        "controlled_artifact_status": "controlled",
        "action_invocation_id": action_invocation_id,
        "action": grounded.packet(),
        "before_state": before_state,
        "after_state": after_state,
        "before_frame": before_artifact,
        "after_frame": after_artifact,
        "after_frame_sequence": action_sequence,
        "action_sequence_sheet": action_sequence_sheet,
        "action_frame_count": len(action_frames),
        "frame_delta": pixel_delta,
        "deltaband_pixel_delta_overlay": deltaband_delta_overlay,
        "deltaband_surface_contact_overlay": deltaband_surface_contact_overlay,
        "portband_surface_feedback_overlay": portband_surface_feedback_overlay,
        "portband_surface_feedback_artifact": portband_surface_feedback_artifact,
        "portband_surface_feedback_summary": portband_feedback_summary,
        "before_geoband_artifacts": before_geoband_artifacts,
        "after_geoband_artifacts": after_geoband_artifacts,
        "deltaband_relationship_delta_sheet": deltaband_relationship_delta_sheet,
        "action_invocation_log_artifact": action_log_artifact,
        "pre_action_contra_packet_artifact": before_packet_artifact,
        "post_action_contra_packet_artifact": after_packet_artifact,
        "oracle_goal_prediction_before": before_oracle_prediction,
        "oracle_goal_prediction_after": after_oracle_prediction,
        "oracle_goal_prediction_delta": oracle_delta,
        "oracle_goal_prediction_before_overlay": before_oracle_overlay,
        "oracle_goal_prediction_after_overlay": after_oracle_overlay,
        "before_contra_perception_pipeline": {
            key: value
            for key, value in before_perception.items()
            if key not in {"component_label_map", "component_lookup"}
        },
        "after_contra_perception_pipeline": {
            key: value
            for key, value in after_perception.items()
            if key not in {"component_label_map", "component_lookup"}
        },
        "deltaband_observations": deltaband,
        "actionmodband_observations": actionmodband,
        "before_episode_memory": before_episode_memory,
        "episode_memory_after_round": episode_memory_after_round,
        "changed_geoband_boundary_ids": boundary_delta["changed_geoband_boundary_ids"],
        "fixed_geoband_boundary_ids": boundary_delta["fixed_geoband_boundary_ids"],
        "null_action_events": [
            {
                "action_invocation_id": action_invocation_id,
                "reason": "raw_pixels_components_and_geobands_unchanged",
            }
        ]
        if deltaband.get("null_action_signal") is True
        else [],
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
    write_json(level_dir / "controlled_level_run.json", row)
    return row


def parse_levels(value: str) -> list[int]:
    if value == "all":
        return list(DEFAULT_LEVELS)
    levels: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        levels.append(int(part))
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="all", help="Comma-separated SP80 level indices, or all.")
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/contra_v001_sp80_deltaband_controlled_run"))
    parser.add_argument("--scale", type=int, default=12)
    return parser


def rel_for_html(repo_relpath: str, index_dir: Path) -> str:
    target = ROOT / repo_relpath
    return html.escape(os.path.relpath(target.resolve(), index_dir.resolve()))


def build_index(manifest: dict[str, Any], *, index_dir: Path) -> str:
    levels = [row for row in manifest.get("levels", []) if isinstance(row, dict)]
    total_changed = sum(len(row.get("changed_geoband_boundary_ids", [])) for row in levels)
    total_fixed = sum(len(row.get("fixed_geoband_boundary_ids", [])) for row in levels)
    total_surface_events = sum(
        int(((row.get("deltaband_observations") or {}).get("action_surface_result_event_count") or 0))
        for row in levels
        if isinstance(row.get("deltaband_observations"), dict)
    )
    total_portband_feedback_matches = sum(
        int(((row.get("portband_surface_feedback_summary") or {}).get("matched_alignment_count") or 0))
        for row in levels
        if isinstance(row.get("portband_surface_feedback_summary"), dict)
    )
    total_action_state_tags = sum(
        int(((row.get("portband_surface_feedback_summary") or {}).get("action_state_tag_count") or 0))
        for row in levels
        if isinstance(row.get("portband_surface_feedback_summary"), dict)
    )
    total_relative_state_tags = sum(
        int(((row.get("portband_surface_feedback_summary") or {}).get("relative_state_change_tag_count") or 0))
        for row in levels
        if isinstance(row.get("portband_surface_feedback_summary"), dict)
    )
    total_interaction_edges = sum(
        int(((row.get("portband_surface_feedback_summary") or {}).get("action_interaction_edge_count") or 0))
        for row in levels
        if isinstance(row.get("portband_surface_feedback_summary"), dict)
    )
    total_novel_relationship_candidates = sum(
        int(((row.get("portband_surface_feedback_summary") or {}).get("novel_relationship_candidate_count") or 0))
        for row in levels
        if isinstance(row.get("portband_surface_feedback_summary"), dict)
    )
    total_route_flow_locks = sum(
        int(((row.get("portband_surface_feedback_summary") or {}).get("route_flow_lock_count") or 0))
        for row in levels
        if isinstance(row.get("portband_surface_feedback_summary"), dict)
    )
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 DeltaBand Controlled Run</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:0;background:#090d12;color:#e8eef7}",
        "main{max-width:1760px;margin:0 auto;padding:24px}",
        "section{margin:34px 0 56px;padding-top:12px;border-top:1px solid #283649}",
        "a{color:#8fd4ff}",
        "img{image-rendering:pixelated;border:1px solid #40546d;background:#111;max-width:100%;height:auto}",
        "table{border-collapse:collapse;font-size:14px;margin-top:12px;width:100%}",
        "td,th{border:1px solid #33465d;padding:7px 9px;text-align:left;vertical-align:top}",
        "th{background:#111a25;color:#f4f8ff}",
        ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:18px 0}",
        ".card{background:#101923;border:1px solid #36506d;border-radius:8px;padding:12px}",
        ".card b{display:block;font-size:24px;color:#fff}",
        ".frame-deck{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(760px,100%),1fr));gap:22px;align-items:start}",
        ".wide-deck{display:grid;grid-template-columns:1fr;gap:18px;margin-top:18px}",
        ".tile{min-width:0}",
        ".tile h3{margin:0 0 8px;font-size:20px}",
        ".tile img{width:100%}",
        ".mono,code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#111923;color:#d8f2ff;padding:1px 4px;border-radius:3px}",
        ".boundary{background:#151c24;border-left:4px solid #f0b429;padding:10px 12px;color:#ffe8a3}",
        ".system-label{font-size:13px;text-transform:uppercase;letter-spacing:.04em;color:#9fb2ca;margin:22px 0 8px}",
        "</style></head><body><main>",
        "<h1>SP80 DeltaBand Controlled Run</h1>",
        "<p class='boundary'>Frames-first debug surface. Controlled before/action/after ACTION5 / activate_flow boundary. DeltaBand observations are real for this action boundary; Oracle goal confidence is a review-rank score, not calibrated probability, goal truth, action truth, or solve truth.</p>",
        f"<p><b>Manifest:</b> <a href='{rel_for_html(str(manifest.get('manifest_json')), index_dir)}'>{html.escape(str(manifest.get('manifest_json')))}</a></p>",
        "<div class='cards'>",
        f"<div class='card'>Levels<b>{html.escape(str(manifest.get('level_count')))}</b></div>",
        f"<div class='card'>Action<b>{html.escape(str(manifest.get('action_name')))}</b></div>",
        f"<div class='card'>Changed GeoBands<b>{total_changed}</b></div>",
        f"<div class='card'>Fixed GeoBands<b>{total_fixed}</b></div>",
        f"<div class='card'>Surface Result Events<b>{total_surface_events}</b></div>",
        f"<div class='card'>PortBand Feedback Matches<b>{total_portband_feedback_matches}</b></div>",
        f"<div class='card'>Action-State Tags<b>{total_action_state_tags}</b></div>",
        f"<div class='card'>Relative-State Tags<b>{total_relative_state_tags}</b></div>",
        f"<div class='card'>Interaction Edges<b>{total_interaction_edges}</b></div>",
        f"<div class='card'>Novel Relationships<b>{total_novel_relationship_candidates}</b></div>",
        f"<div class='card'>PortBand Route-Flow Locks<b>{total_route_flow_locks}</b></div>",
        "</div>",
    ]
    body.append("<h2>Frame Debug Deck</h2>")
    body.append("<p class='boundary'>Visuals come first: raw state, action rollout frames, GeoBand before/after, DeltaBand relationship deltas, Oracle overlays, and pixel deltas. Metrics follow below.</p>")
    for row in levels:
        level_index = int(row.get("level_index"))
        before_png = ((row.get("before_frame") or {}).get("png") or {}).get("path")
        after_png = ((row.get("after_frame") or {}).get("png") or {}).get("path")
        before_oracle_png = (row.get("oracle_goal_prediction_before_overlay") or {}).get("path")
        after_oracle_png = (row.get("oracle_goal_prediction_after_overlay") or {}).get("path")
        before_geoband = row.get("before_geoband_artifacts") if isinstance(row.get("before_geoband_artifacts"), dict) else {}
        after_geoband = row.get("after_geoband_artifacts") if isinstance(row.get("after_geoband_artifacts"), dict) else {}
        before_geoband_overlay = (before_geoband.get("statecon_geoband_overlay") or {}).get("path")
        after_geoband_overlay = (after_geoband.get("statecon_geoband_overlay") or {}).get("path")
        before_primary_geoband = (before_geoband.get("primary_geoband_field") or {}).get("path")
        after_primary_geoband = (after_geoband.get("primary_geoband_field") or {}).get("path")
        before_boundary_queue = (before_geoband.get("boundary_queue") or {}).get("path")
        after_boundary_queue = (after_geoband.get("boundary_queue") or {}).get("path")
        before_portband = (before_geoband.get("portband_alignment") or {}).get("path")
        after_portband = (after_geoband.get("portband_alignment") or {}).get("path")
        delta_png = (row.get("deltaband_pixel_delta_overlay") or {}).get("path")
        surface_contact_png = (row.get("deltaband_surface_contact_overlay") or {}).get("path")
        portband_feedback_png = (row.get("portband_surface_feedback_overlay") or {}).get("path")
        relationship_delta_png = (row.get("deltaband_relationship_delta_sheet") or {}).get("path")
        action_sheet_png = (row.get("action_sequence_sheet") or {}).get("path")
        sequence = row.get("after_frame_sequence") if isinstance(row.get("after_frame_sequence"), list) else []
        final_png = ((sequence[-1] or {}).get("png") or {}).get("path") if sequence and isinstance(sequence[-1], dict) else None
        body.append(f"<section><h2>Level {level_index} Frames</h2>")
        body.append("<div class='system-label'>Raw control boundary</div><div class='frame-deck'>")
        for title, path_value in (("Raw Before", before_png), ("Raw After", after_png), ("Final Action Frame", final_png)):
            if isinstance(path_value, str):
                body.append(f"<div class='tile'><h3>{title}</h3><img src='{rel_for_html(path_value, index_dir)}'></div>")
        body.append("</div>")
        body.append("<div class='system-label'>Action rollout frames</div><div class='wide-deck'>")
        if isinstance(action_sheet_png, str):
            body.append(f"<div class='tile'><h3>Every Captured Action Frame</h3><img src='{rel_for_html(action_sheet_png, index_dir)}'></div>")
        body.append("</div>")
        body.append("<div class='system-label'>GeoBand frames</div><div class='frame-deck'>")
        for title, path_value in (
            ("GeoBand Before: StateCon + Relationships", before_geoband_overlay),
            ("GeoBand After: StateCon + Relationships", after_geoband_overlay),
            ("GeoBand Before: Primary Field", before_primary_geoband),
            ("GeoBand After: Primary Field", after_primary_geoband),
            ("GeoBand Before: Boundary Queue", before_boundary_queue),
            ("GeoBand After: Boundary Queue", after_boundary_queue),
            ("PortBand Before: Port Alignment", before_portband),
            ("PortBand After: Port Alignment", after_portband),
        ):
            if isinstance(path_value, str):
                body.append(f"<div class='tile'><h3>{title}</h3><img src='{rel_for_html(path_value, index_dir)}'></div>")
        body.append("</div>")
        body.append("<div class='system-label'>DeltaBand relationship frame</div><div class='wide-deck'>")
        if isinstance(relationship_delta_png, str):
            body.append(f"<div class='tile'><h3>DeltaBand Before/After Relationship Delta</h3><img src='{rel_for_html(relationship_delta_png, index_dir)}'></div>")
        body.append("</div>")
        body.append("<div class='system-label'>DeltaBand action-surface-result events</div><div class='frame-deck'>")
        if isinstance(surface_contact_png, str):
            body.append(f"<div class='tile'><h3>Action -> Surface Contact -> Result</h3><img src='{rel_for_html(surface_contact_png, index_dir)}'></div>")
        if isinstance(portband_feedback_png, str):
            body.append(f"<div class='tile'><h3>PortBand Feedback Ranking From Surface Events</h3><img src='{rel_for_html(portband_feedback_png, index_dir)}'></div>")
        body.append("</div>")
        body.append("<div class='system-label'>Oracle goal/actor overlays</div><div class='frame-deck'>")
        for title, path_value in (("Oracle Before", before_oracle_png), ("Oracle After", after_oracle_png)):
            if isinstance(path_value, str):
                body.append(f"<div class='tile'><h3>{title}</h3><img src='{rel_for_html(path_value, index_dir)}'></div>")
        body.append("</div>")
        body.append("<div class='system-label'>DeltaBand pixel delta</div><div class='frame-deck'>")
        if isinstance(delta_png, str):
            body.append(f"<div class='tile'><h3>Before -> After Pixel Delta</h3><img src='{rel_for_html(delta_png, index_dir)}'></div>")
        body.append("</div></section>")

    body.append("<h2>Metrics</h2>")
    body.append("<table><thead><tr><th>level</th><th>frames</th><th>pixels changed</th><th>surface result events</th><th>PortBand feedback matches</th><th>action tags</th><th>relative tags</th><th>interaction edges</th><th>novel relationships</th><th>solve-like flags</th><th>boundaries changed</th><th>boundaries fixed</th><th>null action</th></tr></thead><tbody>")
    for row in levels:
        deltaband = row.get("deltaband_observations") if isinstance(row.get("deltaband_observations"), dict) else {}
        delta = row.get("frame_delta") if isinstance(row.get("frame_delta"), dict) else {}
        feedback = row.get("portband_surface_feedback_summary") if isinstance(row.get("portband_surface_feedback_summary"), dict) else {}
        body.append(
            "<tr>"
            f"<td>{int(row.get('level_index'))}</td>"
            f"<td>{int(row.get('action_frame_count') or 0)}</td>"
            f"<td>{html.escape(str(delta.get('changed_pixels')))}</td>"
            f"<td>{html.escape(str(deltaband.get('action_surface_result_event_count')))}</td>"
            f"<td>{html.escape(str(feedback.get('matched_alignment_count')))}</td>"
            f"<td>{html.escape(str(feedback.get('action_state_tag_count')))}</td>"
            f"<td>{html.escape(str(feedback.get('relative_state_change_tag_count')))}</td>"
            f"<td>{html.escape(str(feedback.get('action_interaction_edge_count')))}</td>"
            f"<td>{html.escape(str(feedback.get('novel_relationship_candidate_count')))}</td>"
            f"<td>{html.escape(str(feedback.get('solve_like_evidence_flag_count')))}</td>"
            f"<td>{len(row.get('changed_geoband_boundary_ids', []))}</td>"
            f"<td>{len(row.get('fixed_geoband_boundary_ids', []))}</td>"
            f"<td>{html.escape(str(deltaband.get('null_action_signal')))}</td>"
            "</tr>"
        )
    body.append("</tbody></table>")
    body.append("<h2>DeltaBand Action-Surface-Result Events</h2>")
    body.append("<p class='boundary'>These rows bind an action-produced changed region to contacted surface edges and the resulting frame delta. They are observed temporal contacts, not causal truth or solve truth.</p>")
    body.append("<table><thead><tr><th>level</th><th>rank</th><th>surface role</th><th>bbox</th><th>edge</th><th>first frame</th><th>last frame</th><th>contact frames</th><th>max contact cells</th><th>final changed px</th><th>confidence</th></tr></thead><tbody>")
    for row in levels:
        level_index = int(row.get("level_index"))
        deltaband = row.get("deltaband_observations") if isinstance(row.get("deltaband_observations"), dict) else {}
        for event in (deltaband.get("action_surface_result_events") or [])[:12]:
            if not isinstance(event, dict):
                continue
            surface = event.get("surface") if isinstance(event.get("surface"), dict) else {}
            edge = event.get("surface_edge") if isinstance(event.get("surface_edge"), dict) else {}
            contact = event.get("contact") if isinstance(event.get("contact"), dict) else {}
            result = event.get("result_delta") if isinstance(event.get("result_delta"), dict) else {}
            body.append(
                "<tr>"
                f"<td>{level_index}</td>"
                f"<td>{html.escape(str(event.get('event_rank')))}</td>"
                f"<td>{html.escape(str(surface.get('surface_role')))}</td>"
                f"<td>{html.escape(str(surface.get('bbox_xyxy')))}</td>"
                f"<td>{html.escape(str(edge.get('dominant_side')))}</td>"
                f"<td>{html.escape(str(contact.get('first_frame_index')))}</td>"
                f"<td>{html.escape(str(contact.get('last_frame_index')))}</td>"
                f"<td>{html.escape(str(contact.get('contact_frame_count')))}</td>"
                f"<td>{html.escape(str(contact.get('max_surface_contact_cell_count')))}</td>"
                f"<td>{html.escape(str(result.get('final_changed_pixel_count')))}</td>"
                f"<td>{html.escape(str(event.get('confidence')))}</td>"
                "</tr>"
            )
    body.append("</tbody></table>")
    body.append("<h2>PortBand Feedback Calibration</h2>")
    body.append("<p class='boundary'>Fail-safe calibration: DeltaBand surface-result events can reorder a copied PortBand ranking overlay and tag generic action-state correlations for any controlled action primitive. Route-flow is only one specialization. Original static PortBand alignments are preserved and no truth is admitted.</p>")
    body.append("<table><thead><tr><th>level</th><th>statecon</th><th>rank</th><th>kind</th><th>error</th><th>adjusted</th><th>dynamic</th><th>events</th><th>action tags</th><th>action primitives</th><th>action score</th><th>route lock</th><th>route score</th><th>route event ids</th><th>source boxes</th><th>imagined boxes</th><th>target boxes</th></tr></thead><tbody>")
    for row in levels:
        level_index = int(row.get("level_index"))
        feedback = row.get("portband_surface_feedback_summary") if isinstance(row.get("portband_surface_feedback_summary"), dict) else {}
        for alignment in (feedback.get("top_feedback_alignments") or [])[:16]:
            if not isinstance(alignment, dict):
                continue
            body.append(
                "<tr>"
                f"<td>{level_index}</td>"
                f"<td>{html.escape(str(alignment.get('statecon_id')))}</td>"
                f"<td>{html.escape(str(alignment.get('feedback_rank')))}</td>"
                f"<td>{html.escape(str(alignment.get('alignment_kind')))}</td>"
                f"<td>{html.escape(str(alignment.get('alignment_error_px')))}</td>"
                f"<td>{html.escape(str(alignment.get('feedback_adjusted_score')))}</td>"
                f"<td>{html.escape(str(alignment.get('feedback_dynamic_score')))}</td>"
                f"<td>{html.escape(str(alignment.get('surface_feedback_event_count')))}</td>"
                f"<td>{html.escape(str(alignment.get('action_state_tag_count')))}</td>"
                f"<td>{html.escape(str(alignment.get('action_state_primitives')))}</td>"
                f"<td>{html.escape(str(alignment.get('action_state_correlation_score')))}</td>"
                f"<td>{html.escape(str(alignment.get('route_flow_lock_status')))}</td>"
                f"<td>{html.escape(str(alignment.get('route_flow_confidence_score')))}</td>"
                f"<td>{html.escape(str(alignment.get('route_flow_evidence_event_ids')))}</td>"
                f"<td>{html.escape(str(alignment.get('source_bboxes_xyxy')))}</td>"
                f"<td>{html.escape(str(alignment.get('imagined_source_bboxes_xyxy')))}</td>"
                f"<td>{html.escape(str(alignment.get('target_bboxes_xyxy')))}</td>"
                "</tr>"
            )
    body.append("</tbody></table>")
    body.append("<h2>Action-State / Relative-State Tags</h2>")
    body.append("<p class='boundary'>These rows are generic interaction clues from controlled action boundaries. They can flag novel relative state changes and mathematical relationship signatures, but they are not solve truth or action authority.</p>")
    body.append("<table><thead><tr><th>level</th><th>statecon</th><th>tag kind</th><th>scope</th><th>primitive</th><th>score</th><th>novelty</th><th>solve-like flag</th><th>solve-like score</th><th>changed fields</th><th>math signature</th></tr></thead><tbody>")
    for row in levels:
        level_index = int(row.get("level_index"))
        feedback = row.get("portband_surface_feedback_summary") if isinstance(row.get("portband_surface_feedback_summary"), dict) else {}
        for tag in (feedback.get("top_action_state_tags") or [])[:24]:
            if not isinstance(tag, dict):
                continue
            math_rel = tag.get("mathematical_relationships") if isinstance(tag.get("mathematical_relationships"), dict) else {}
            body.append(
                "<tr>"
                f"<td>{level_index}</td>"
                f"<td>{html.escape(str(tag.get('statecon_id')))}</td>"
                f"<td>{html.escape(str(tag.get('tag_kind')))}</td>"
                f"<td>{html.escape(str(tag.get('tag_scope')))}</td>"
                f"<td>{html.escape(str(tag.get('action_primitive')))}</td>"
                f"<td>{html.escape(str(tag.get('correlation_score')))}</td>"
                f"<td>{html.escape(str(tag.get('novelty_status')))}</td>"
                f"<td>{html.escape(str(tag.get('solve_like_evidence_flag')))}</td>"
                f"<td>{html.escape(str(tag.get('solve_like_evidence_score')))}</td>"
                f"<td>{html.escape(str(tag.get('changed_fields')))}</td>"
                f"<td>{html.escape(str(math_rel.get('relationship_signature')))}</td>"
                "</tr>"
            )
    body.append("</tbody></table>")
    body.append("<h2>Action Interaction Edges</h2>")
    body.append("<p class='boundary'>Edges compile tags into object-to-object or relation-member interaction clues. They are ranked candidate evidence only and never authorize a move or claim a solve.</p>")
    body.append("<table><thead><tr><th>level</th><th>statecon</th><th>rank</th><th>kind</th><th>primitive</th><th>from</th><th>to</th><th>participants</th><th>score</th><th>novelty</th><th>solve-like flag</th><th>relation label</th><th>math signature</th></tr></thead><tbody>")
    for row in levels:
        level_index = int(row.get("level_index"))
        feedback = row.get("portband_surface_feedback_summary") if isinstance(row.get("portband_surface_feedback_summary"), dict) else {}
        for edge in (feedback.get("top_action_interaction_edges") or [])[:24]:
            if not isinstance(edge, dict):
                continue
            body.append(
                "<tr>"
                f"<td>{level_index}</td>"
                f"<td>{html.escape(str(edge.get('statecon_id')))}</td>"
                f"<td>{html.escape(str(edge.get('interaction_rank')))}</td>"
                f"<td>{html.escape(str(edge.get('interaction_kind')))}</td>"
                f"<td>{html.escape(str(edge.get('action_primitive')))}</td>"
                f"<td>{html.escape(str(edge.get('from_object_ids')))}</td>"
                f"<td>{html.escape(str(edge.get('to_object_ids')))}</td>"
                f"<td>{html.escape(str(edge.get('participant_object_ids')))}</td>"
                f"<td>{html.escape(str(edge.get('solve_like_evidence_score')))}</td>"
                f"<td>{html.escape(str(edge.get('novelty_status')))}</td>"
                f"<td>{html.escape(str(edge.get('solve_like_evidence_flag')))}</td>"
                f"<td>{html.escape(str(edge.get('relation_type_label')))}</td>"
                f"<td>{html.escape(str(edge.get('relationship_signature')))}</td>"
                "</tr>"
            )
    body.append("</tbody></table>")
    body.append("<h2>Oracle Goal Rollout</h2>")
    body.append("<table><thead><tr><th>level</th><th>before goal</th><th>before confidence</th><th>after goal</th><th>after confidence</th><th>goal delta</th><th>actor before - after</th></tr></thead><tbody>")
    for row in levels:
        before = row.get("oracle_goal_prediction_before") if isinstance(row.get("oracle_goal_prediction_before"), dict) else {}
        after = row.get("oracle_goal_prediction_after") if isinstance(row.get("oracle_goal_prediction_after"), dict) else {}
        odelta = row.get("oracle_goal_prediction_delta") if isinstance(row.get("oracle_goal_prediction_delta"), dict) else {}
        before_goal = before.get("goal_prediction") if isinstance(before.get("goal_prediction"), dict) else {}
        after_goal = after.get("goal_prediction") if isinstance(after.get("goal_prediction"), dict) else {}
        before_actor = before.get("actor_prediction") if isinstance(before.get("actor_prediction"), dict) else {}
        after_actor = after.get("actor_prediction") if isinstance(after.get("actor_prediction"), dict) else {}
        goal_delta = {
            "changed": odelta.get("goal_component_changed"),
            "confidence_delta": odelta.get("goal_confidence_delta"),
            "score_delta": odelta.get("goal_score_delta"),
        }
        body.append(
            "<tr>"
            f"<td>{int(row.get('level_index'))}</td>"
            f"<td>{html.escape(str(before_goal.get('selected_component_id')))}</td>"
            f"<td>{html.escape(str(before_goal.get('confidence_score')))}</td>"
            f"<td>{html.escape(str(after_goal.get('selected_component_id')))}</td>"
            f"<td>{html.escape(str(after_goal.get('confidence_score')))}</td>"
            f"<td>{html.escape(json.dumps(goal_delta, sort_keys=True))}</td>"
            f"<td>{html.escape(str(before_actor.get('selected_component_id')))} - {html.escape(str(after_actor.get('selected_component_id')))}</td>"
            "</tr>"
        )
    body.append("</tbody></table>")
    body.append("<h2>Per-Level Calculation Details</h2>")
    for row in levels:
        level_index = int(row.get("level_index"))
        body.extend([f"<section><h2>Level {level_index} Calculations</h2>"])
        before = row.get("oracle_goal_prediction_before") if isinstance(row.get("oracle_goal_prediction_before"), dict) else {}
        after = row.get("oracle_goal_prediction_after") if isinstance(row.get("oracle_goal_prediction_after"), dict) else {}
        odelta = row.get("oracle_goal_prediction_delta") if isinstance(row.get("oracle_goal_prediction_delta"), dict) else {}
        for phase, prediction in (("before", before), ("after", after)):
            goal = prediction.get("goal_prediction") if isinstance(prediction.get("goal_prediction"), dict) else {}
            actor = prediction.get("actor_prediction") if isinstance(prediction.get("actor_prediction"), dict) else {}
            body.append(f"<h3>Oracle {html.escape(phase.title())}</h3>")
            body.append("<table><thead><tr><th>kind</th><th>component</th><th>confidence</th><th>score</th><th>runner up</th><th>criteria</th><th>lock/key</th></tr></thead><tbody>")
            for kind, pred in (("goal", goal), ("actor", actor)):
                lock_key = {
                    "fit": pred.get("lock_key_shape_fit"),
                    "fit_score": pred.get("lock_key_fit_score"),
                    "key": pred.get("lock_key_component_id"),
                }
                body.append(
                    "<tr>"
                    f"<td>{html.escape(kind)}</td>"
                    f"<td>{html.escape(str(pred.get('selected_component_id')))}</td>"
                    f"<td>{html.escape(str(pred.get('confidence_score')))}</td>"
                    f"<td>{html.escape(str(pred.get('selected_score')))}</td>"
                    f"<td>{html.escape(str(pred.get('runner_up_component_id')))} / {html.escape(str(pred.get('runner_up_score')))}</td>"
                    f"<td>{html.escape(json.dumps(pred.get('criteria'), sort_keys=True))}</td>"
                    f"<td>{html.escape(json.dumps(lock_key, sort_keys=True))}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
        body.append(f"<p><b>Oracle delta:</b> <span class='mono'>{html.escape(json.dumps(odelta, sort_keys=True))}</span></p>")
        body.append("<table><thead><tr><th>field</th><th>value</th></tr></thead><tbody>")
        for key in ("action_invocation_id", "changed_geoband_boundary_ids", "fixed_geoband_boundary_ids", "null_action_events"):
            body.append(
                "<tr>"
                f"<td>{html.escape(key)}</td>"
                f"<td><code>{html.escape(json.dumps(row.get(key), sort_keys=True))}</code></td>"
                "</tr>"
            )
        for label, artifact_key in (
            ("action_invocation_log", "action_invocation_log_artifact"),
            ("pre_action_contra_packet", "pre_action_contra_packet_artifact"),
            ("post_action_contra_packet", "post_action_contra_packet_artifact"),
        ):
            artifact = row.get(artifact_key) if isinstance(row.get(artifact_key), dict) else {}
            path_value = artifact.get("path")
            if isinstance(path_value, str):
                body.append(
                    "<tr>"
                    f"<td>{html.escape(label)}</td>"
                    f"<td><a href='{rel_for_html(path_value, index_dir)}'>{html.escape(path_value)}</a></td>"
                    "</tr>"
                )
        body.append("</tbody></table></section>")
    body.extend(["</main></body></html>"])
    return "\n".join(body)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = parse_levels(str(args.levels))
    calibration_by_level = load_calibration_by_level()
    rows = [
        run_level(
            level_index=level_index,
            out_dir=out_dir,
            scale=max(1, int(args.scale)),
            calibration=calibration_by_level.get(level_index, {}),
        )
        for level_index in levels
    ]
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "controlled",
        "condition_name": "sp80_activate_flow_deltaband_controlled_boundary",
        "condition_natural_language": "Run real SP80 activate_flow actions and compare before/after CONTRA GeoBand boundary rows.",
        "controlled_artifact_status": "controlled",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "required_boundary": DELTABAND_REQUIRED_BOUNDARY,
        "deltaband_observations_schema": DELTABAND_SET_SCHEMA,
        "action_primitive": "activate_flow",
        "action_id": 5,
        "action_name": "ACTION5",
        "sp80_action_set": SP80_ACTION_SET,
        "level_count": len(rows),
        "levels": rows,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "prediction_status": "not_predictive",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "source_game": relpath(SP80_PATH),
        "source_game_sha256": sha256_file(SP80_PATH),
        "source_script": "codexlab/sp80_deltaband_controlled_run.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_deltaband_controlled_run.py"),
        "source_contra": "codexlab/contra_head/contra.py",
        "source_contra_sha256": sha256_file(ROOT / "codexlab" / "contra_head" / "contra.py"),
        "source_contra_modules": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in contra_module_source_paths()
        ],
    }
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)
    index_path = out_dir / "index.html"
    index_path.write_text(build_index(manifest, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "index_html": relpath(index_path),
                "status": "controlled",
                "manifest": relpath(manifest_path),
                "level_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
