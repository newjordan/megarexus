#!/usr/bin/env python3
"""Exact raw-color board viewer for SP80 calibration frames.

This viewer is a pre-SAM board visibility pass. It reads recorded raw frame
grids, enumerates exact palette values, and renders per-color/component masks.
It does not run SAM, use semantic labels, controlled centers, routes, or
replays.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from contra_head import (
    CONTRA_PERCEPTION_PIPELINE_SCHEMA,
    CONTRA_RELATIONSHIP_LADDER_SCHEMA,
    DELTABAND_REQUIRED_BOUNDARY,
    DELTABAND_CONTROLLED_EVENT_REQUIREMENT_SCHEMA,
    DELTABAND_SET_SCHEMA,
    GEOBAND_HYPOTHESIS_SCHEMA,
    SP80_MANUAL_CALIBRATION_REVIEW_SCHEMA,
    STATECON_HYPOTHESIS_SCHEMA,
    STATECON_SET_SCHEMA,
    SUPPORT_ACTION_AFFORDANCE_SET_SCHEMA,
    build_deltaband_controlled_event_requirement,
    build_contra_perception_pipeline,
    build_sp80_manual_calibration_review,
    contra_module_source_paths,
)
from sp80_base_controller import PALETTE, ROOT, rgb_frame, sha256_file, sha256_frame
from sp80_object_label_viewer import relpath, resolve_repo_path, write_json
from review_helpers import html_value_list, rel_for_html
from sp80_sam_hypothesis_viewer import PALETTE_NAMES, raw_color_inventory


MANIFEST_SCHEMA = "codexlab.sp80_color_board_viewer_manifest.v1"
DEFAULT_SP80_CALIBRATION = ROOT / "codexlab" / "sp80_surface_calibration.json"
PRIMARY_COLOR_VALUES = {6, 8, 9, 10, 11, 14, 15}
SURFACE_ROLE_COLORS = {
    "background_hypothesis": (60, 120, 255, 110),
    "floor_hypothesis": (0, 210, 130, 145),
    "residual_component_hypothesis": (250, 235, 40, 130),
    "wall_hypothesis": (255, 70, 60, 145),
}
OPEN_TARGET_COLOR = (0, 255, 0, 185)
TARGET_AFFORDANCE_HYPOTHESIS_SCHEMA = "codexlab.target_affordance_hypothesis.v1"
TARGET_AFFORDANCE_SET_SCHEMA = "codexlab.target_affordance_hypotheses.v1"
ACTOR_AFFORDANCE_HYPOTHESIS_SCHEMA = "codexlab.actor_affordance_hypothesis.v1"
ACTOR_AFFORDANCE_SET_SCHEMA = "codexlab.actor_affordance_hypotheses.v1"


def image_meta(path: Path) -> dict[str, Any]:
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def bbox_from_component(component: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(v) for v in component["bbox_xyxy"])  # type: ignore[return-value]


def scale_bbox(bbox: tuple[int, int, int, int], scale: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (x0 * scale, y0 * scale, (x1 + 1) * scale - 1, (y1 + 1) * scale - 1)


def overlay_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def image_size_for_draw(draw: ImageDraw.ImageDraw) -> tuple[int, int] | None:
    image = getattr(draw, "_image", None)
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])
    core_size = getattr(getattr(draw, "im", None), "size", None)
    if isinstance(core_size, tuple) and len(core_size) == 2:
        return int(core_size[0]), int(core_size[1])
    return None


def text_pixel_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    text_box = draw.textbbox((0, 0), text, font=font)
    return text_box[2] - text_box[0], text_box[3] - text_box[1]


def fit_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if max_width <= 0:
        return ""
    if text_pixel_size(draw, text, font)[0] <= max_width:
        return text
    suffix = "..."
    if text_pixel_size(draw, suffix, font)[0] > max_width:
        return ""
    lo = 0
    hi = len(text)
    best = suffix
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if text_pixel_size(draw, candidate, font)[0] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def label_boxes_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int], *, gap: int = 3) -> bool:
    return not (
        left[2] + gap < right[0]
        or right[2] + gap < left[0]
        or left[3] + gap < right[1]
        or right[3] + gap < left[1]
    )


def clamped_label_box(
    x: int,
    y: int,
    box_w: int,
    box_h: int,
    image_size: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    if image_size is not None:
        image_w, image_h = image_size
        x = min(max(0, int(x)), max(0, image_w - box_w))
        y = min(max(0, int(y)), max(0, image_h - box_h))
    return (int(x), int(y), int(x + box_w - 1), int(y + box_h - 1))


def placed_label_box(
    xy: tuple[int, int],
    box_w: int,
    box_h: int,
    image_size: tuple[int, int] | None,
    occupied_boxes: list[tuple[int, int, int, int]] | None,
) -> tuple[int, int, int, int]:
    if not occupied_boxes:
        return clamped_label_box(int(xy[0]), int(xy[1]), box_w, box_h, image_size)
    step_x = box_w + 6
    step_y = box_h + 6
    seen: set[tuple[int, int]] = set()
    candidates: list[tuple[int, int, int, int]] = []
    for ring in range(0, 12):
        offsets = [(0, 0)] if ring == 0 else []
        if ring > 0:
            for ix in range(-ring, ring + 1):
                offsets.append((ix * step_x, -ring * step_y))
                offsets.append((ix * step_x, ring * step_y))
            for iy in range(-ring + 1, ring):
                offsets.append((-ring * step_x, iy * step_y))
                offsets.append((ring * step_x, iy * step_y))
        for dx, dy in offsets:
            box = clamped_label_box(int(xy[0]) + dx, int(xy[1]) + dy, box_w, box_h, image_size)
            origin = (box[0], box[1])
            if origin in seen:
                continue
            seen.add(origin)
            candidates.append(box)
            if not any(label_boxes_overlap(box, taken) for taken in occupied_boxes):
                return box
    return min(candidates, key=lambda box: sum(1 for taken in occupied_boxes if label_boxes_overlap(box, taken)))


def draw_text_tag(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    color: tuple[int, int, int],
    *,
    occupied_boxes: list[tuple[int, int, int, int]] | None = None,
    font_size: int = 14,
) -> None:
    text = str(text)
    font = overlay_font(font_size, bold=True)
    image_size = image_size_for_draw(draw)
    if image_size is not None:
        max_text_w = max(1, image_size[0] - 12)
        text = fit_text_to_width(draw, text, font, max_text_w)
    text_w, text_h = text_pixel_size(draw, text, font)
    box_w = max(46, text_w + 12)
    box_h = max(22, text_h + 10)
    if image_size is not None:
        box_w = min(box_w, image_size[0])
        box_h = min(box_h, image_size[1])
    box = placed_label_box((int(xy[0]), int(xy[1])), box_w, box_h, image_size, occupied_boxes)
    if occupied_boxes is not None:
        occupied_boxes.append(box)
    draw.rectangle(box, fill=(5, 8, 12, 220), outline=(*color, 255), width=1)
    draw.text((box[0] + 6, box[1] + 4), text, fill=(*color, 255), font=font)


def value_rgb(value: int) -> tuple[int, int, int]:
    rgb = PALETTE[int(value)].tolist()
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def save_raw_board(frame: np.ndarray, path: Path, scale: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_frame(frame, scale), mode="RGB").save(path)
    return image_meta(path)


def save_color_mask(
    frame: np.ndarray,
    *,
    value: int,
    components: list[dict[str, Any]],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    arr = np.asarray(frame)
    h, w = arr.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[arr == int(value)] = np.asarray(value_rgb(value), dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        image = image.resize((w * scale, h * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    outline = (255, 255, 255) if value not in {0, 1, 10, 11} else (0, 0, 0)
    for component in components:
        draw.rectangle(scale_bbox(bbox_from_component(component), scale), outline=outline, width=max(1, scale // 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image_meta(path)


def save_color_overlay(
    frame: np.ndarray,
    *,
    value: int,
    components: list[dict[str, Any]],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    image = Image.fromarray(rgb_frame(frame, scale), mode="RGB").convert("RGBA")
    arr = np.asarray(frame)
    mask = (arr == int(value)).astype(np.uint8)
    if scale > 1:
        mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    overlay = Image.new("RGBA", image.size, (*value_rgb(value), 0))
    overlay.putalpha(Image.fromarray(mask * 120, mode="L"))
    image.alpha_composite(overlay)
    draw = ImageDraw.Draw(image)
    label_boxes: list[tuple[int, int, int, int]] = []
    for idx, component in enumerate(components):
        bbox = scale_bbox(bbox_from_component(component), scale)
        draw.rectangle(bbox, outline=(*value_rgb(value), 255), width=max(1, scale // 2))
        draw_text_tag(draw, (bbox[0] + 1, bbox[1] + 1), f"{value}:{idx}", (255, 255, 255), occupied_boxes=label_boxes, font_size=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)
    return image_meta(path)


def mask_for_component_ids(label_map: np.ndarray, component_lookup: dict[str, int], component_ids: list[str]) -> np.ndarray:
    mask = np.zeros(label_map.shape[:2], dtype=bool)
    for component_id in component_ids:
        label_index = component_lookup.get(str(component_id))
        if label_index is None:
            continue
        mask |= label_map == int(label_index)
    return mask


def square_from_bbox(
    bbox: tuple[int, int, int, int],
    *,
    frame_shape: tuple[int, int],
) -> list[int]:
    height, width = frame_shape
    x0, y0, x1, y1 = bbox
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    side = max(3, bbox_w, bbox_h)
    sx0 = x0 - (side - bbox_w) // 2
    sy0 = y0 - (side - bbox_h) // 2
    sx0 = max(0, min(width - side, sx0))
    sy0 = max(0, min(height - side, sy0))
    return [int(sx0), int(sy0), int(sx0 + side - 1), int(sy0 + side - 1)]


def union_bbox(*boxes: tuple[int, int, int, int]) -> list[int]:
    return [
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    ]


def direction_from_key_to_actor(actor: dict[str, Any], key: dict[str, Any]) -> str:
    actor_center = actor.get("center_xy") or [0.0, 0.0]
    key_center = key.get("center_xy") or [0.0, 0.0]
    dx = float(actor_center[0]) - float(key_center[0])
    dy = float(actor_center[1]) - float(key_center[1])
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    if abs(dy) > 0.0:
        return "down" if dy > 0 else "up"
    return "none"


def contact_edge_count(scene_surfaces: dict[str, Any], component_a: str, component_b: str) -> int:
    expected = {str(component_a), str(component_b)}
    for contact in scene_surfaces.get("component_graph", {}).get("contacts", []):
        if not isinstance(contact, dict):
            continue
        if {str(contact.get("component_a")), str(contact.get("component_b"))} == expected:
            return int(contact.get("contact_edge_count") or 0)
    return 0


def connected_open_region_bbox(component_mask: np.ndarray, open_direction: str) -> tuple[int, int, int, int] | None:
    empty = ~component_mask
    height, width = empty.shape[:2]
    seeds: list[tuple[int, int]] = []
    if open_direction == "up":
        seeds = [(0, x) for x in range(width) if empty[0, x]]
    elif open_direction == "down":
        seeds = [(height - 1, x) for x in range(width) if empty[height - 1, x]]
    elif open_direction == "left":
        seeds = [(y, 0) for y in range(height) if empty[y, 0]]
    elif open_direction == "right":
        seeds = [(y, width - 1) for y in range(height) if empty[y, width - 1]]
    if not seeds:
        return None

    seen = np.zeros(empty.shape, dtype=bool)
    stack = list(seeds)
    for y, x in seeds:
        seen[y, x] = True
    points: list[tuple[int, int]] = []
    while stack:
        y, x = stack.pop()
        points.append((y, x))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width:
                continue
            if seen[ny, nx] or not empty[ny, nx]:
                continue
            seen[ny, nx] = True
            stack.append((ny, nx))
    if not points:
        return None
    ys = [point[0] for point in points]
    xs = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)


def actor_affordance_hypotheses(
    *,
    scene_surfaces: dict[str, Any],
    actor_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(actor_expectation, dict) or actor_expectation.get("status") != "user_supplied_answer":
        return []
    components = {
        str(component.get("component_id")): component
        for component in scene_surfaces.get("component_graph", {}).get("components", [])
        if isinstance(component, dict)
    }
    hypotheses: list[dict[str, Any]] = []
    direction_rule = str(actor_expectation.get("direction_rule"))
    for row in actor_expectation.get("component_groups", []):
        if not isinstance(row, dict):
            continue
        actor_component_id = str(row.get("actor_component_id"))
        key_component_id = str(row.get("orientation_key_component_id"))
        actor = components.get(actor_component_id)
        key = components.get(key_component_id)
        expected_direction = str(row.get("flow_direction"))
        if actor is None or key is None or expected_direction not in {"up", "down", "left", "right"}:
            continue
        inferred_direction = direction_from_key_to_actor(actor, key)
        actor_bbox = bbox_from_component(actor)
        key_bbox = bbox_from_component(key)
        group_box = union_bbox(actor_bbox, key_bbox)
        actor_id = f"actor_emitter_{actor_component_id}"
        hypotheses.append(
            {
                "schema": ACTOR_AFFORDANCE_HYPOTHESIS_SCHEMA,
                "status": "user_supplied_answer",
                "actor_id": actor_id,
                "actor_kind": str(actor_expectation.get("actor_kind")),
                "source": "manual_sp80_surface_calibration",
                "source_component_role": "residual_component_hypothesis",
                "actor_component_id": actor_component_id,
                "source_component_id": actor_component_id,
                "actor_raw_pixel_value": int(actor.get("raw_pixel_value")),
                "orientation_key_component_id": key_component_id,
                "orientation_key_raw_pixel_value": int(key.get("raw_pixel_value")),
                "orientation_key_role": "flow_direction_marker",
                "group_component_ids": [actor_component_id, key_component_id],
                "direction_rule": direction_rule,
                "flow_direction_hypothesis": expected_direction,
                "direction_from_raw_geometry": inferred_direction,
                "direction_agrees_with_raw_geometry": inferred_direction == expected_direction,
                "activation_condition_hypothesis": "action_button",
                "evidence_status": "manual_hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "actor_bbox_xyxy": [int(v) for v in actor_bbox],
                "orientation_key_bbox_xyxy": [int(v) for v in key_bbox],
                "group_bbox_xyxy": group_box,
                "contact_edge_count": contact_edge_count(scene_surfaces, actor_component_id, key_component_id),
                "render_color": "bright_purple",
                "direction_render_color": "cyan",
                "provenance": actor_expectation.get("provenance"),
                "rendering_role": "review_overlay_marker",
            }
        )
    return hypotheses


def target_affordance_hypotheses(
    *,
    frame_shape: tuple[int, int],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    scene_surfaces: dict[str, Any],
    bucket_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if not isinstance(bucket_expectation, dict) or bucket_expectation.get("status") != "user_supplied_answer":
        return []
    components = {
        str(component.get("component_id")): component
        for component in scene_surfaces.get("component_graph", {}).get("components", [])
        if isinstance(component, dict)
    }
    hypotheses: list[dict[str, Any]] = []
    for row in bucket_expectation.get("manual_asymmetry_hypotheses", []):
        if not isinstance(row, dict) or row.get("asymmetry_makes_open_target") is not True:
            continue
        component_id = str(row.get("component_id"))
        open_direction = str(row.get("open_direction"))
        component = components.get(component_id)
        label_index = component_lookup.get(component_id)
        if component is None or label_index is None or open_direction not in {"up", "down", "left", "right"}:
            continue
        x0, y0, x1, y1 = bbox_from_component(component)
        component_mask = label_map[y0 : y1 + 1, x0 : x1 + 1] == int(label_index)
        region_bbox = connected_open_region_bbox(component_mask, open_direction)
        if region_bbox is None:
            target_bbox = (x0, y0, x1, y1)
        else:
            rx0, ry0, rx1, ry1 = region_bbox
            target_bbox = (x0 + rx0, y0 + ry0, x0 + rx1, y0 + ry1)
        target_id = f"target_open_{component_id}"
        hypotheses.append(
            {
                "schema": TARGET_AFFORDANCE_HYPOTHESIS_SCHEMA,
                "status": "user_supplied_answer",
                "target_id": target_id,
                "target_kind": "open_target",
                "source": "manual_sp80_surface_calibration",
                "source_component_role": "residual_component_hypothesis",
                "component_id": component_id,
                "source_component_id": component_id,
                "source_raw_pixel_value": int(component.get("raw_pixel_value")),
                "source_shape_family": str(bucket_expectation.get("shape_family")),
                "open_direction": open_direction,
                "evidence_status": "manual_hypothesis",
                "admission_status": "not_admitted",
                "control_truth_admitted": False,
                "goal_condition_admitted": False,
                "action_choice_admitted": False,
                "target_bbox_xyxy": [int(v) for v in target_bbox],
                "target_square_xyxy": square_from_bbox(target_bbox, frame_shape=frame_shape),
                "render_color": "bright_green",
                "provenance": bucket_expectation.get("provenance"),
                "rendering_role": "review_overlay_marker",
            }
        )
    return hypotheses


def draw_target_affordances(
    draw: ImageDraw.ImageDraw,
    target_affordances: list[dict[str, Any]],
    *,
    scale: int,
    label_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    if label_boxes is None:
        label_boxes = []
    for affordance in target_affordances:
        square = affordance.get("target_square_xyxy")
        if not isinstance(square, list) or len(square) != 4:
            continue
        bbox = scale_bbox(tuple(int(v) for v in square), scale)
        draw.rectangle(bbox, fill=(0, 255, 0, 70), outline=(0, 255, 0, 255), width=max(2, scale // 2))
        label = str(affordance.get("target_kind") or "target").replace("_", " ")
        draw_text_tag(draw, (bbox[0] + 1, bbox[1] + 1), label, (0, 255, 0), occupied_boxes=label_boxes)


def draw_actor_affordances(
    draw: ImageDraw.ImageDraw,
    actor_affordances: list[dict[str, Any]],
    *,
    scale: int,
    label_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    if label_boxes is None:
        label_boxes = []
    for affordance in actor_affordances:
        group = affordance.get("group_bbox_xyxy")
        actor_box = affordance.get("actor_bbox_xyxy")
        key_box = affordance.get("orientation_key_bbox_xyxy")
        if not (
            isinstance(group, list)
            and len(group) == 4
            and isinstance(actor_box, list)
            and len(actor_box) == 4
            and isinstance(key_box, list)
            and len(key_box) == 4
        ):
            continue
        group_bbox = scale_bbox(tuple(int(v) for v in group), scale)
        actor_bbox = scale_bbox(tuple(int(v) for v in actor_box), scale)
        key_bbox = scale_bbox(tuple(int(v) for v in key_box), scale)
        draw.rectangle(group_bbox, fill=(190, 0, 255, 45), outline=(190, 0, 255, 255), width=max(2, scale // 2))
        draw.rectangle(actor_bbox, outline=(255, 80, 255, 255), width=max(2, scale // 2))
        draw.rectangle(key_bbox, outline=(255, 255, 255, 255), width=max(2, scale // 2))
        actor_cx = (actor_bbox[0] + actor_bbox[2]) // 2
        actor_cy = (actor_bbox[1] + actor_bbox[3]) // 2
        key_cx = (key_bbox[0] + key_bbox[2]) // 2
        key_cy = (key_bbox[1] + key_bbox[3]) // 2
        draw.line((key_cx, key_cy, actor_cx, actor_cy), fill=(0, 255, 255, 255), width=max(2, scale // 2))
        arrow = max(3, scale)
        direction = str(affordance.get("flow_direction_hypothesis"))
        if direction == "up":
            points = [(actor_cx, actor_cy - arrow), (actor_cx - arrow, actor_cy + arrow), (actor_cx + arrow, actor_cy + arrow)]
        elif direction == "down":
            points = [(actor_cx, actor_cy + arrow), (actor_cx - arrow, actor_cy - arrow), (actor_cx + arrow, actor_cy - arrow)]
        elif direction == "left":
            points = [(actor_cx - arrow, actor_cy), (actor_cx + arrow, actor_cy - arrow), (actor_cx + arrow, actor_cy + arrow)]
        else:
            points = [(actor_cx + arrow, actor_cy), (actor_cx - arrow, actor_cy - arrow), (actor_cx - arrow, actor_cy + arrow)]
        draw.polygon(points, fill=(0, 255, 255, 200))
        label = f"actor {direction}"
        draw_text_tag(draw, (group_bbox[0] + 1, group_bbox[1] + 1), label, (255, 80, 255), occupied_boxes=label_boxes)


def draw_statecon_hypotheses(
    draw: ImageDraw.ImageDraw,
    statecons: list[dict[str, Any]],
    *,
    scale: int,
    label_boxes: list[tuple[int, int, int, int]] | None = None,
) -> None:
    if label_boxes is None:
        label_boxes = []
    for statecon in statecons[:6]:
        group = statecon.get("group_bbox_xyxy")
        if not isinstance(group, list) or len(group) != 4:
            continue
        group_bbox = scale_bbox(tuple(int(v) for v in group), scale)
        score = float(statecon.get("relation_descriptor_score") or statecon.get("event_enable_score") or 0.0)
        draw.rectangle(group_bbox, outline=(255, 180, 0, 255), width=max(2, scale // 2))
        geoband = statecon.get("geoband_hypothesis")
        if isinstance(geoband, dict):
            geoband_complete = geoband.get("completeness_status") == "complete"
            for ray in geoband.get("rays", []):
                if not isinstance(ray, dict):
                    continue
                origin = ray.get("origin_xy")
                if not isinstance(origin, list) or len(origin) != 2:
                    continue
                ox = int(float(origin[0]) * scale)
                oy = int(float(origin[1]) * scale)
                target_xy = ray.get("target_xy")
                if isinstance(target_xy, list) and len(target_xy) == 2:
                    tx = int(float(target_xy[0]) * scale)
                    ty = int(float(target_xy[1]) * scale)
                else:
                    bearing = math.radians(float(ray.get("bearing_degrees") or 0.0))
                    length = max(group_bbox[2] - group_bbox[0], group_bbox[3] - group_bbox[1], scale * 8)
                    tx = int(ox + math.cos(bearing) * length)
                    ty = int(oy + math.sin(bearing) * length)
                ray_kind = str(ray.get("ray_kind"))
                ray_color = {
                    "actor_flow_axis": (255, 120, 0, 170),
                    "actor_to_target": (0, 255, 0, 165),
                    "actor_to_support_group": (40, 160, 255, 170),
                }.get(ray_kind, (255, 120, 0, 170))
                if not geoband_complete:
                    ray_color = (120, 120, 120, 90)
                draw.line((ox, oy, tx, ty), fill=ray_color, width=max(1, scale // 4 if geoband_complete else scale // 6))
            relationship_nodes = geoband.get("relationship_nodes")
            relationship_edges = geoband.get("relationship_edges")
            if isinstance(relationship_nodes, list) and isinstance(relationship_edges, list):
                network_node_by_id = {
                    str(node.get("node_id")): node
                    for node in relationship_nodes
                    if isinstance(node, dict)
                }
                network_color_by_kind = {
                    "actor_to_goal_attachment": (0, 255, 0, 145),
                    "actor_to_support_group": (40, 160, 255, 145),
                    "support_group_to_goal_attachment": (120, 255, 180, 145),
                    "actor_to_peer_actor": (255, 80, 255, 145),
                    "actor_to_portband_alignment": (255, 110, 40, 150),
                    "portband_alignment_to_goal_attachment": (255, 190, 40, 150),
                }
                for edge in relationship_edges:
                    if not isinstance(edge, dict):
                        continue
                    left = network_node_by_id.get(str(edge.get("from_node_id")))
                    right = network_node_by_id.get(str(edge.get("to_node_id")))
                    if not isinstance(left, dict) or not isinstance(right, dict):
                        continue
                    left_center = left.get("center_xy")
                    right_center = right.get("center_xy")
                    if not isinstance(left_center, list) or not isinstance(right_center, list) or len(left_center) != 2 or len(right_center) != 2:
                        continue
                    draw.line(
                        (
                            int(float(left_center[0]) * scale),
                            int(float(left_center[1]) * scale),
                            int(float(right_center[0]) * scale),
                            int(float(right_center[1]) * scale),
                        ),
                        fill=network_color_by_kind.get(str(edge.get("edge_kind")), (255, 255, 255, 120))
                        if geoband_complete
                        else (120, 120, 120, 95),
                        width=max(1, scale // 5 if geoband_complete else scale // 8),
                    )
        graph = statecon.get("statecon_branch_graph")
        nodes = graph.get("nodes") if isinstance(graph, dict) else []
        edges = graph.get("edges") if isinstance(graph, dict) else []
        node_by_id = {str(node.get("node_id")): node for node in nodes if isinstance(node, dict)}
        for edge in edges if isinstance(edges, list) else []:
            if not isinstance(edge, dict):
                continue
            left = node_by_id.get(str(edge.get("from_node_id")))
            right = node_by_id.get(str(edge.get("to_node_id")))
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            if "macro_statecon" in {str(left.get("node_kind")), str(right.get("node_kind"))}:
                continue
            left_center = left.get("center_xy")
            right_center = right.get("center_xy")
            if not isinstance(left_center, list) or not isinstance(right_center, list) or len(left_center) != 2 or len(right_center) != 2:
                continue
            draw.line(
                (
                    int(float(left_center[0]) * scale),
                    int(float(left_center[1]) * scale),
                    int(float(right_center[0]) * scale),
                    int(float(right_center[1]) * scale),
                ),
                fill=(255, 210, 80, 210),
                width=max(1, scale // 3),
            )
        color_by_kind = {
            "macro_statecon": (255, 180, 0, 255),
            "actor_affordance": (255, 80, 255, 255),
            "target_affordance": (0, 255, 0, 255),
            "statecon_support_group": (40, 160, 255, 255),
            "statecon_support_affordance": (40, 160, 255, 255),
            "portband_alignment": (255, 150, 40, 255),
        }
        for node in nodes if isinstance(nodes, list) else []:
            if not isinstance(node, dict):
                continue
            center = node.get("center_xy")
            if not isinstance(center, list) or len(center) != 2:
                continue
            if node.get("node_kind") == "macro_statecon":
                continue
            cx = int(float(center[0]) * scale)
            cy = int(float(center[1]) * scale)
            radius = max(3, scale // 2)
            color = color_by_kind.get(str(node.get("node_kind")), (255, 255, 255, 255))
            if node.get("special_object_category") == "known_y_unknown_attached" or node.get("unknown_attachment_marker") == "Y":
                color = (255, 245, 0, 255)
            elif int(node.get("known_with_unknown_attached_count") or 0) > 0:
                color = (255, 245, 0, 255)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(0, 0, 0, 255), width=max(1, scale // 6))
            if node.get("unknown_attachment_marker") == "Y":
                draw.text((cx + radius + 2, cy - radius), "Y", fill=(0, 0, 0, 255))
                draw.text((cx + radius + 1, cy - radius - 1), "Y", fill=(255, 245, 0, 255))
            if node.get("node_kind") in {"statecon_support_group", "statecon_support_affordance"}:
                node_box = node.get("bbox_xyxy")
                if isinstance(node_box, list) and len(node_box) == 4:
                    draw.rectangle(scale_bbox(tuple(int(v) for v in node_box), scale), outline=color, width=max(2, scale // 2))
        geoband = statecon.get("geoband_hypothesis")
        label = f"statecon {score:.2f}"
        if isinstance(geoband, dict) and geoband.get("completeness_status") != "complete":
            label += " geoband incomplete"
        draw_text_tag(draw, (group_bbox[0] + 1, group_bbox[3] - 22), label, (255, 220, 80), occupied_boxes=label_boxes)


def selected_surface_component_ids(scene_surfaces: dict[str, Any], role: str) -> list[str]:
    if role == "floor_hypothesis":
        floor = scene_surfaces.get("floor_hypothesis")
        return [] if not isinstance(floor, dict) else [str(floor["component_id"])]
    key_by_role = {
        "wall_hypothesis": "wall_hypotheses",
        "background_hypothesis": "background_hypotheses",
        "residual_component_hypothesis": "residual_component_hypotheses",
    }
    key = key_by_role[role]
    return [str(row["component_id"]) for row in scene_surfaces.get(key, []) if isinstance(row, dict)]


def render_scene_surface_overlay(
    frame: np.ndarray,
    *,
    scene_surfaces: dict[str, Any],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    scale: int,
    target_affordances: list[dict[str, Any]] | None = None,
    actor_affordances: list[dict[str, Any]] | None = None,
    statecons: list[dict[str, Any]] | None = None,
) -> Image.Image:
    image = Image.fromarray(rgb_frame(frame, scale), mode="RGB").convert("RGBA")
    for role in ("background_hypothesis", "floor_hypothesis", "wall_hypothesis", "residual_component_hypothesis"):
        component_ids = selected_surface_component_ids(scene_surfaces, role)
        if not component_ids:
            continue
        mask = mask_for_component_ids(label_map, component_lookup, component_ids).astype(np.uint8)
        if scale > 1:
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        overlay = Image.new("RGBA", image.size, SURFACE_ROLE_COLORS[role])
        overlay.putalpha(Image.fromarray(mask * SURFACE_ROLE_COLORS[role][3], mode="L"))
        image.alpha_composite(overlay)

    draw = ImageDraw.Draw(image)
    label_boxes: list[tuple[int, int, int, int]] = []
    role_rows = []
    floor = scene_surfaces.get("floor_hypothesis")
    if isinstance(floor, dict):
        role_rows.append(("floor", floor, SURFACE_ROLE_COLORS["floor_hypothesis"]))
    role_rows.extend(("wall", row, SURFACE_ROLE_COLORS["wall_hypothesis"]) for row in scene_surfaces.get("wall_hypotheses", []))
    role_rows.extend(
        ("background", row, SURFACE_ROLE_COLORS["background_hypothesis"])
        for row in scene_surfaces.get("background_hypotheses", [])
    )
    role_rows.extend(
        ("residual", row, SURFACE_ROLE_COLORS["residual_component_hypothesis"])
        for row in scene_surfaces.get("residual_component_hypotheses", [])[:16]
    )
    for role_name, row, color in role_rows:
        bbox = scale_bbox(bbox_from_component(row), scale)
        draw.rectangle(bbox, outline=(color[0], color[1], color[2], 255), width=max(1, scale // 2))
        label = f"{role_name} {row['score']:.2f}"
        draw_text_tag(draw, (bbox[0] + 1, bbox[1] + 1), label, (255, 255, 255), occupied_boxes=label_boxes)
    draw_statecon_hypotheses(draw, statecons or [], scale=scale, label_boxes=label_boxes)
    draw_actor_affordances(draw, actor_affordances or [], scale=scale, label_boxes=label_boxes)
    draw_target_affordances(draw, target_affordances or [], scale=scale, label_boxes=label_boxes)
    return image.convert("RGB")


def save_scene_surface_overlay(
    frame: np.ndarray,
    *,
    scene_surfaces: dict[str, Any],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    path: Path,
    scale: int,
    target_affordances: list[dict[str, Any]] | None = None,
    actor_affordances: list[dict[str, Any]] | None = None,
    statecons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    render_scene_surface_overlay(
        frame,
        scene_surfaces=scene_surfaces,
        label_map=label_map,
        component_lookup=component_lookup,
        target_affordances=target_affordances,
        actor_affordances=actor_affordances,
        statecons=statecons,
        scale=scale,
    ).save(path)
    return image_meta(path)


def render_role_mask(
    *,
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    component_ids: list[str],
    role: str,
    scale: int,
) -> Image.Image:
    mask = mask_for_component_ids(label_map, component_lookup, component_ids)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color = SURFACE_ROLE_COLORS[role]
    rgb[mask] = np.asarray(color[:3], dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    if scale > 1:
        image = image.resize((mask.shape[1] * scale, mask.shape[0] * scale), Image.Resampling.NEAREST)
    return image


def render_target_affordance_mask(
    *,
    frame_shape: tuple[int, int],
    target_affordances: list[dict[str, Any]],
    scale: int,
) -> Image.Image:
    height, width = frame_shape
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_target_affordances(draw, target_affordances, scale=scale)
    return image


def render_actor_affordance_mask(
    *,
    frame_shape: tuple[int, int],
    actor_affordances: list[dict[str, Any]],
    scale: int,
) -> Image.Image:
    height, width = frame_shape
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_actor_affordances(draw, actor_affordances, scale=scale)
    return image


def render_statecon_mask(
    *,
    frame_shape: tuple[int, int],
    statecons: list[dict[str, Any]],
    scale: int,
) -> Image.Image:
    height, width = frame_shape
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw_statecon_hypotheses(draw, statecons, scale=scale)
    return image


def render_geoband_primary_mask(
    *,
    frame_shape: tuple[int, int],
    statecons: list[dict[str, Any]],
    scale: int,
) -> Image.Image:
    height, width = frame_shape
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    for statecon in statecons[:6]:
        geoband = statecon.get("geoband_hypothesis") if isinstance(statecon, dict) else None
        if not isinstance(geoband, dict):
            continue
        for node in geoband.get("relationship_nodes", []):
            if not isinstance(node, dict) or node.get("node_kind") not in {"statecon_support_group", "symmetry_group"}:
                continue
            center = node.get("center_xy")
            if not isinstance(center, list) or len(center) != 2:
                continue
            cx = int(float(center[0]) * scale)
            cy = int(float(center[1]) * scale)
            radius = max(3, scale // 2)
            if node.get("node_kind") == "symmetry_group":
                color = (34, 211, 238, 255)
            else:
                color = (255, 245, 0, 255) if int(node.get("known_with_unknown_attached_count") or 0) > 0 else (40, 160, 255, 255)
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=(0, 0, 0, 255), width=max(1, scale // 6))
        for edge in geoband.get("relationship_edges", []):
            if not isinstance(edge, dict):
                continue
            if edge.get("edge_kind") not in {
                "actor_to_support_group",
                "support_group_to_goal_attachment",
                "actor_to_symmetry_group",
                "symmetry_group_to_goal_attachment",
                "support_group_to_symmetry_group",
            }:
                continue
            nodes = {
                str(node.get("node_id")): node
                for node in geoband.get("relationship_nodes", [])
                if isinstance(node, dict)
            }
            left = nodes.get(str(edge.get("from_node_id")))
            right = nodes.get(str(edge.get("to_node_id")))
            if not isinstance(left, dict) or not isinstance(right, dict):
                continue
            left_center = left.get("center_xy")
            right_center = right.get("center_xy")
            if not isinstance(left_center, list) or not isinstance(right_center, list):
                continue
            draw.line(
                (
                    int(float(left_center[0]) * scale),
                    int(float(left_center[1]) * scale),
                    int(float(right_center[0]) * scale),
                    int(float(right_center[1]) * scale),
                ),
                fill=(34, 211, 238, 150) if "symmetry" in str(edge.get("edge_kind")) else (120, 255, 180, 130),
                width=max(1, scale // 6),
            )
    return image


def render_portband_mask(
    *,
    frame_shape: tuple[int, int],
    statecons: list[dict[str, Any]],
    scale: int,
    frame: np.ndarray | None = None,
) -> Image.Image:
    height, width = frame_shape
    if frame is not None:
        image = Image.fromarray(rgb_frame(frame, scale), mode="RGB").convert("RGBA")
    else:
        image = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 255))
    draw = ImageDraw.Draw(image, "RGBA")
    rendered_alignment_keys: set[tuple[str, str, str]] = set()
    for statecon in statecons[:6]:
        if not isinstance(statecon, dict):
            continue
        portband = statecon.get("portband_hypotheses")
        if not isinstance(portband, dict):
            continue
        ports = {
            str(port.get("port_id")): port
            for port in portband.get("ports", [])
            if isinstance(port, dict)
        }
        alignment_rows = portband.get("surface_feedback_adjusted_alignments")
        if not isinstance(alignment_rows, list) or not alignment_rows:
            alignment_rows = portband.get("alignments", []) if isinstance(portband.get("alignments"), list) else []
        for alignment in alignment_rows:
            if not isinstance(alignment, dict):
                continue
            render_key = (
                str(alignment.get("alignment_kind")),
                str(alignment.get("source_component_ids")),
                str(alignment.get("imagined_source_bboxes_xyxy")),
            )
            if render_key in rendered_alignment_keys:
                continue
            rendered_alignment_keys.add(render_key)
            source_ids = [str(value) for value in alignment.get("source_port_ids", []) if value is not None]
            target_ids = [str(value) for value in alignment.get("target_port_ids", []) if value is not None]
            exact = alignment.get("exact_alignment") is True
            feedback_score = float(alignment.get("surface_feedback_dynamic_score") or 0.0)
            color = (80, 235, 255, 255) if feedback_score > 0.0 else ((255, 190, 40, 255) if exact else (255, 110, 40, 220))
            for source_box, imagined_box in zip(
                alignment.get("source_bboxes_xyxy", []) if isinstance(alignment.get("source_bboxes_xyxy"), list) else [],
                alignment.get("imagined_source_bboxes_xyxy", []) if isinstance(alignment.get("imagined_source_bboxes_xyxy"), list) else [],
            ):
                if not isinstance(source_box, list) or len(source_box) != 4:
                    continue
                if not isinstance(imagined_box, list) or len(imagined_box) != 4:
                    continue
                ix0, iy0, ix1, iy1 = [int(value) for value in imagined_box]
                if frame is not None:
                    sx0, sy0, sx1, sy1 = [int(value) for value in source_box]
                    sx0 = max(0, min(width - 1, sx0))
                    sx1 = max(0, min(width - 1, sx1))
                    sy0 = max(0, min(height - 1, sy0))
                    sy1 = max(0, min(height - 1, sy1))
                    if sx1 >= sx0 and sy1 >= sy0:
                        crop = np.asarray(frame)[sy0 : sy1 + 1, sx0 : sx1 + 1]
                        ghost = Image.fromarray(rgb_frame(crop, scale), mode="RGB").convert("RGBA")
                        alpha = Image.new("L", ghost.size, 165 if exact else 125)
                        image.paste(ghost, (ix0 * scale, iy0 * scale), alpha)
                draw.rectangle(
                    scale_bbox((ix0, iy0, ix1, iy1), scale),
                    outline=(255, 255, 255, 240) if exact else (255, 170, 40, 230),
                    width=max(2, scale // 3),
                )
            dx, dy = (0.0, 0.0)
            transform = alignment.get("transform_hypothesis_xy")
            if isinstance(transform, list) and len(transform) == 2:
                dx, dy = float(transform[0]), float(transform[1])
            for port_id in source_ids:
                port = ports.get(port_id)
                if not isinstance(port, dict):
                    continue
                edge_cells = port.get("edge_cells_xy")
                for cell in edge_cells if isinstance(edge_cells, list) else []:
                    if not isinstance(cell, list) or len(cell) != 2:
                        continue
                    x, y = int(round(float(cell[0]) + dx)), int(round(float(cell[1]) + dy))
                    draw.rectangle(
                        scale_bbox((x, y, x, y), scale),
                        fill=(color[0], color[1], color[2], 115),
                        outline=color,
                        width=max(1, scale // 8),
                    )
            for port_id in target_ids:
                port = ports.get(port_id)
                if not isinstance(port, dict):
                    continue
                edge_cells = port.get("edge_cells_xy")
                for cell in edge_cells if isinstance(edge_cells, list) else []:
                    if not isinstance(cell, list) or len(cell) != 2:
                        continue
                    x, y = int(cell[0]), int(cell[1])
                    draw.rectangle(
                        scale_bbox((x, y, x, y), scale),
                        fill=(0, 255, 160, 115),
                        outline=(0, 255, 160, 255),
                        width=max(1, scale // 8),
                    )
    return image.convert("RGB")


def render_geoband_subgroup_mask(
    *,
    frame_shape: tuple[int, int],
    statecons: list[dict[str, Any]],
    scale: int,
) -> Image.Image:
    height, width = frame_shape
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    label_boxes: list[tuple[int, int, int, int]] = []
    for statecon in statecons[:6]:
        for group in statecon.get("geoband_subgroups", []) if isinstance(statecon, dict) else []:
            if not isinstance(group, dict):
                continue
            box = group.get("group_bbox_xyxy")
            if not isinstance(box, list) or len(box) != 4:
                continue
            color = (255, 245, 0, 255) if int(group.get("known_with_unknown_attached_count") or 0) > 0 else (40, 160, 255, 255)
            draw.rectangle(scale_bbox(tuple(int(v) for v in box), scale), outline=color, width=max(2, scale // 2))
    return image


def render_boundary_queue_mask(
    *,
    frame_shape: tuple[int, int],
    statecons: list[dict[str, Any]],
    scale: int,
) -> Image.Image:
    height, width = frame_shape
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    label_boxes: list[tuple[int, int, int, int]] = []
    for statecon in statecons[:6]:
        if not isinstance(statecon, dict):
            continue
        support_by_id = {
            str(row.get("support_id")): row
            for row in statecon.get("support_affordance_hypotheses", [])
            if isinstance(row, dict)
        }
        queue = statecon.get("geoband_boundary_queue")
        boundaries = queue.get("boundaries") if isinstance(queue, dict) else []
        for boundary in boundaries if isinstance(boundaries, list) else []:
            if not isinstance(boundary, dict):
                continue
            for support_id in boundary.get("input_object_ids", []):
                support = support_by_id.get(str(support_id))
                if not isinstance(support, dict):
                    continue
                box = support.get("bbox_xyxy")
                if not isinstance(box, list) or len(box) != 4:
                    continue
                bbox = scale_bbox(tuple(int(v) for v in box), scale)
                draw.rectangle(bbox, outline=(255, 245, 0, 255), width=max(2, scale // 2))
                label = str(boundary.get("boundary_type") or "boundary").replace("_", " ")
                draw_text_tag(draw, (bbox[0] + 1, bbox[1] + 1), label[:18], (255, 245, 0), occupied_boxes=label_boxes)
    return image


def save_scene_surface_sheet(
    *,
    frame: np.ndarray,
    scene_surfaces: dict[str, Any],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    path: Path,
    scale: int,
    target_affordances: list[dict[str, Any]] | None = None,
    actor_affordances: list[dict[str, Any]] | None = None,
    statecons: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tile_w = frame.shape[1] * scale
    tile_h = frame.shape[0] * scale
    title_h = 34
    gap = 12
    tiles = [
        {"title": "raw board", "image": Image.fromarray(rgb_frame(frame, scale), mode="RGB")},
        {
            "title": "scene surface hypotheses",
            "image": render_scene_surface_overlay(
                frame,
                scene_surfaces=scene_surfaces,
                label_map=label_map,
                component_lookup=component_lookup,
                target_affordances=target_affordances,
                actor_affordances=actor_affordances,
                statecons=statecons,
                scale=scale,
            ),
        },
    ]
    for role, title in (
        ("floor_hypothesis", "floor hypothesis"),
        ("wall_hypothesis", "wall hypotheses"),
        ("background_hypothesis", "background hypotheses"),
        ("residual_component_hypothesis", "residual component hypotheses"),
    ):
        tiles.append(
            {
                "title": title,
                "image": render_role_mask(
                    label_map=label_map,
                    component_lookup=component_lookup,
                    component_ids=selected_surface_component_ids(scene_surfaces, role),
                    role=role,
                    scale=scale,
                ),
            }
        )
    if target_affordances:
        tiles.append(
            {
                "title": "target affordance hypotheses",
                "image": render_target_affordance_mask(
                    frame_shape=frame.shape[:2],
                    target_affordances=target_affordances,
                    scale=scale,
                ),
            }
        )
    if actor_affordances:
        tiles.append(
            {
                "title": "actor affordance hypotheses",
                "image": render_actor_affordance_mask(
                    frame_shape=frame.shape[:2],
                    actor_affordances=actor_affordances,
                    scale=scale,
                ),
            }
        )
    if statecons:
        tiles.append(
            {
                "title": "state condition hypotheses",
                "image": render_statecon_mask(
                    frame_shape=frame.shape[:2],
                    statecons=statecons,
                    scale=scale,
                ),
            }
        )
        tiles.extend(
            [
                {
                    "title": "primary geoband field",
                    "image": render_geoband_primary_mask(
                        frame_shape=frame.shape[:2],
                        statecons=statecons,
                        scale=scale,
                    ),
                },
                {
                    "title": "geoband subgroups",
                    "image": render_geoband_subgroup_mask(
                        frame_shape=frame.shape[:2],
                        statecons=statecons,
                        scale=scale,
                    ),
                },
                {
                    "title": "boundary queue",
                    "image": render_boundary_queue_mask(
                        frame_shape=frame.shape[:2],
                        statecons=statecons,
                        scale=scale,
                    ),
                },
            ]
        )

    cols = 3
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, rows * (tile_h + title_h) + gap), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + title_h)
        draw.text((x, y), str(tile["title"]), fill=(20, 20, 20), font=font)
        sheet.paste(tile["image"], (x, y + title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return image_meta(path)


def save_level_sheet(
    *,
    frame: np.ndarray,
    colors: list[dict[str, Any]],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    tile_w = frame.shape[1] * scale
    tile_h = frame.shape[0] * scale
    title_h = 34
    gap = 12
    focused = [color for color in colors if color["raw_pixel_value"] in PRIMARY_COLOR_VALUES]
    if not focused:
        focused = [color for color in colors if not bool(color.get("dominant_background_candidate"))]
    tiles = [{"kind": "raw", "title": "raw board", "value": None, "image": Image.fromarray(rgb_frame(frame, scale), mode="RGB")}]
    for color in focused:
        value = int(color["raw_pixel_value"])
        rgb = np.zeros((*frame.shape[:2], 3), dtype=np.uint8)
        rgb[np.asarray(frame) == value] = np.asarray(value_rgb(value), dtype=np.uint8)
        image = Image.fromarray(rgb, mode="RGB").resize((tile_w, tile_h), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        outline = (255, 255, 255) if value not in {0, 1, 10, 11} else (0, 0, 0)
        for component in color.get("components") or []:
            draw.rectangle(scale_bbox(bbox_from_component(component), scale), outline=outline, width=max(1, scale // 3))
        title = f"v{value:02d} {color['palette_name']} c={color['component_count']}"
        tiles.append({"kind": "color", "title": title, "value": value, "image": image})
    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * tile_w + (cols + 1) * gap, rows * (tile_h + title_h) + gap), (245, 245, 242))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        x = gap + col * (tile_w + gap)
        y = gap + row * (tile_h + title_h)
        draw.text((x, y), str(tile["title"]), fill=(20, 20, 20), font=font)
        sheet.paste(tile["image"], (x, y + title_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return image_meta(path)


def load_frames(frames_dir: Path) -> list[tuple[int, Path, np.ndarray]]:
    rows: list[tuple[int, Path, np.ndarray]] = []
    for path in sorted(frames_dir.glob("level_*_raw.npy")):
        stem = path.stem
        try:
            level = int(stem.split("_")[1])
        except Exception as exc:
            raise RuntimeError(f"cannot parse level index from {path}") from exc
        rows.append((level, path, np.load(path, allow_pickle=False)))
    if not rows:
        raise RuntimeError(f"no level_*_raw.npy frames found in {frames_dir}")
    return rows


def load_calibration_by_level() -> dict[int, dict[str, Any]]:
    if not DEFAULT_SP80_CALIBRATION.is_file():
        return {}
    calibration = json.loads(DEFAULT_SP80_CALIBRATION.read_text(encoding="utf-8"))
    levels = calibration.get("levels")
    if not isinstance(levels, list):
        return {}
    return {int(row["level_index"]): row for row in levels if isinstance(row, dict) and isinstance(row.get("level_index"), int)}


def build_index(manifest: dict[str, Any], *, index_dir: Path) -> str:
    calibration_by_level = load_calibration_by_level()
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 Color Board Viewer</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f7f7f4;color:#191919}",
        "section{margin-bottom:36px}",
        "img{image-rendering:pixelated;border:1px solid #aaa;background:#111;max-width:100%;height:auto}",
        "table{border-collapse:collapse;font-size:12px;margin-top:10px}",
        "td,th{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}",
        ".swatch{display:inline-block;width:18px;height:18px;border:1px solid #333;vertical-align:middle;margin-right:5px}",
        ".row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}",
        ".tile{max-width:540px}",
        ".surface-detail{margin:12px 0 18px 0}",
        ".surface-detail img{width:min(100%,1152px);max-width:none}",
        ".surface-sheet img{width:min(100%,1600px);max-width:none}",
        ".surface-links{font-size:12px;margin:4px 0 8px 0}",
        "code{background:#eee;padding:1px 3px}",
        "</style></head><body>",
        "<h1>SP80 Color Board Viewer</h1>",
        "<p>Exact raw palette isolation from recorded raw grids. No SAM, no semantic names, no routes, no replay.</p>",
        f"<p><b>Manifest:</b> <code>{html.escape(manifest['manifest_json'])}</code></p>",
    ]
    for view in manifest["views"]:
        body.extend(
            [
                f"<section><h2>Calibration Slice {view['level_index']}</h2>",
                "<div class='surface-detail'>",
                "<h3>Large Scene Surface Overlay</h3>",
                f"<div class='surface-links'><a href='{rel_for_html(view['scene_surface_overlay']['path'], index_dir)}'>open full-size overlay</a></div>",
                f"<img src='{rel_for_html(view['scene_surface_overlay']['path'], index_dir)}'>",
                "</div>",
                "<div class='surface-detail surface-sheet'>",
                "<h3>Large Hypothesis Sheet</h3>",
                f"<div class='surface-links'><a href='{rel_for_html(view['scene_surface_sheet']['path'], index_dir)}'>open full-size sheet</a></div>",
                f"<img src='{rel_for_html(view['scene_surface_sheet']['path'], index_dir)}'>",
                "</div>",
                "<div class='row'>",
                f"<div class='tile'><h3>Board And Primary Separations</h3><img src='{rel_for_html(view['review_sheet']['path'], index_dir)}'></div>",
                f"<div class='tile'><h3>Raw Board</h3><img src='{rel_for_html(view['raw_board']['path'], index_dir)}'></div>",
                "</div>",
            ]
        )
        ladder = view.get("contra_relationship_ladder")
        if isinstance(ladder, dict):
            body.append("<h3>Relationship Ladder</h3>")
            body.append("<table><thead><tr><th>stage</th><th>completeness</th><th>missing</th><th>admission</th></tr></thead><tbody>")
            for stage in ladder.get("stages", []):
                if not isinstance(stage, dict):
                    continue
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(stage.get('stage_name')))}</code></td>"
                    f"<td>{html.escape(str(stage.get('completeness_status')))}</td>"
                    f"<td>{html.escape(', '.join(str(value) for value in stage.get('missing_inputs', [])))}</td>"
                    f"<td>{html.escape(str(stage.get('admission_status')))}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
        manual_review = view.get("manual_calibration_review")
        if isinstance(manual_review, dict):
            seed = manual_review.get("seed_candidate_ids") if isinstance(manual_review.get("seed_candidate_ids"), dict) else {}
            body.extend(
                [
                    "<h3>Manual Calibration Review Gate</h3>",
                    "<table><thead><tr><th>status</th><th>pending fields</th><th>seed candidates</th><th>admission</th></tr></thead><tbody>",
                    "<tr>",
                    f"<td>{html.escape(str(manual_review.get('status')))} / {html.escape(str(manual_review.get('review_status')))}</td>",
                    f"<td>{html_value_list(manual_review.get('fields_pending_human_review'))}</td>",
                    "<td>"
                    f"actors: {html_value_list(seed.get('actor_object_ids'))}<br>"
                    f"targets: {html_value_list(seed.get('target_object_ids'))}<br>"
                    f"support/action: {html_value_list(seed.get('known_action_support_object_ids'))}<br>"
                    f"connectors: {html_value_list(seed.get('connector_component_ids'))}<br>"
                    f"known-Y: {html_value_list(seed.get('known_y_component_ids'))}<br>"
                    f"unresolved: {html_value_list(seed.get('unresolved_object_ids'))}"
                    "</td>",
                    f"<td>{html.escape(str(manual_review.get('admission_status')))}</td>",
                    "</tr></tbody></table>",
                ]
            )
        deltaband_requirement = view.get("deltaband_controlled_event_requirement")
        if isinstance(deltaband_requirement, dict):
            body.extend(
                [
                    "<h3>DeltaBand Controlled Event Gate</h3>",
                    "<table><thead><tr><th>status</th><th>required boundary</th><th>missing artifacts</th><th>admission</th></tr></thead><tbody>",
                    "<tr>",
                    f"<td>{html.escape(str(deltaband_requirement.get('status')))} / {html.escape(str(deltaband_requirement.get('controlled_run_status')))}</td>",
                    f"<td>{html.escape(str(deltaband_requirement.get('required_boundary')))}</td>",
                    f"<td>{html_value_list(deltaband_requirement.get('missing_artifacts'))}</td>",
                    f"<td>{html.escape(str(deltaband_requirement.get('admission_status')))}</td>",
                    "</tr></tbody></table>",
                ]
            )
        calibration = calibration_by_level.get(int(view["level_index"]))
        bucket = None
        if isinstance(calibration, dict):
            expectations = calibration.get("scene_surface_expectations")
            if isinstance(expectations, dict):
                bucket = expectations.get("bucket_or_goal_region")
                actor = expectations.get("actor_region")
            else:
                actor = None
        else:
            actor = None
        if isinstance(actor, dict) and actor.get("status") == "user_supplied_answer":
            body.extend(
                [
                    "<table><thead><tr><th>manual actor calibration</th><th>actor/key groups</th><th>direction rule</th><th>provenance</th></tr></thead><tbody>",
                    "<tr>",
                    f"<td>{html.escape(str(actor.get('semantic_role')))} / {html.escape(str(actor.get('actor_kind')))} / actor raw {actor.get('actor_raw_pixel_value')} key raw {actor.get('orientation_key_raw_pixel_value')}</td>",
                    f"<td>{', '.join(html.escape(str(row.get('actor_component_id'))) + '+' + html.escape(str(row.get('orientation_key_component_id'))) + ':' + html.escape(str(row.get('flow_direction'))) for row in actor.get('component_groups', []) if isinstance(row, dict))}</td>",
                    f"<td>{html.escape(str(actor.get('direction_rule')))}</td>",
                    f"<td>{html.escape(str(actor.get('provenance')))}</td>",
                    "</tr>",
                    "</tbody></table>",
                ]
            )
        if isinstance(bucket, dict) and bucket.get("status") == "user_supplied_answer":
            body.extend(
                [
                    "<table><thead><tr><th>manual calibration</th><th>components</th><th>open directions</th><th>provenance</th></tr></thead><tbody>",
                    "<tr>",
                    f"<td>{html.escape(str(bucket.get('semantic_role')))} / {html.escape(str(bucket.get('shape_family')))} / raw {bucket.get('raw_pixel_value')}</td>",
                    f"<td>{', '.join(f'<code>{html.escape(str(component_id))}</code>' for component_id in bucket.get('component_ids', []))}</td>",
                    f"<td>{', '.join(html.escape(str(row.get('component_id'))) + ':' + html.escape(str(row.get('open_direction'))) for row in bucket.get('manual_asymmetry_hypotheses', []) if isinstance(row, dict))}</td>",
                    f"<td>{html.escape(str(bucket.get('provenance')))}</td>",
                    "</tr>",
                    "</tbody></table>",
                ]
            )
        actor_affordances = view.get("actor_affordance_hypotheses", [])
        if isinstance(actor_affordances, list) and actor_affordances:
            body.extend(
                [
                    "<table><thead><tr><th>actor affordance</th><th>actor component</th><th>orientation key</th><th>flow direction</th><th>admission</th></tr></thead><tbody>",
                ]
            )
            for affordance in actor_affordances:
                if not isinstance(affordance, dict):
                    continue
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(affordance.get('actor_id')))}</code> {html.escape(str(affordance.get('actor_kind')))}</td>"
                    f"<td><code>{html.escape(str(affordance.get('actor_component_id')))}</code></td>"
                    f"<td><code>{html.escape(str(affordance.get('orientation_key_component_id')))}</code></td>"
                    f"<td>{html.escape(str(affordance.get('flow_direction_hypothesis')))} / geom {html.escape(str(affordance.get('direction_from_raw_geometry')))}</td>"
                    f"<td>{html.escape(str(affordance.get('admission_status')))}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
        target_affordances = view.get("target_affordance_hypotheses", [])
        if isinstance(target_affordances, list) and target_affordances:
            body.extend(
                [
                    "<table><thead><tr><th>target affordance</th><th>source component</th><th>open direction</th><th>target square</th><th>admission</th></tr></thead><tbody>",
                ]
            )
            for affordance in target_affordances:
                if not isinstance(affordance, dict):
                    continue
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(affordance.get('target_id')))}</code> {html.escape(str(affordance.get('target_kind')))}</td>"
                    f"<td><code>{html.escape(str(affordance.get('source_component_id')))}</code></td>"
                    f"<td>{html.escape(str(affordance.get('open_direction')))}</td>"
                    f"<td><code>{html.escape(str(affordance.get('target_square_xyxy')))}</code></td>"
                    f"<td>{html.escape(str(affordance.get('admission_status')))}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
        support_affordances = view.get("statecon_support_affordances", [])
        if isinstance(support_affordances, list) and support_affordances:
            body.extend(
                [
                    "<table><thead><tr><th>statecon support</th><th>component</th><th>kind</th><th>category</th><th>bbox</th><th>admission</th></tr></thead><tbody>",
                ]
            )
            for support in support_affordances:
                if not isinstance(support, dict):
                    continue
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(support.get('support_id')))}</code></td>"
                    f"<td><code>{html.escape(str(support.get('source_component_id')))}</code> raw {support.get('source_raw_pixel_value')}</td>"
                    f"<td>{html.escape(str(support.get('support_kind')))}</td>"
                    f"<td>{html.escape(str(support.get('special_object_category') or support.get('support_relation_category')))}"
                    f" {html.escape(str(support.get('unknown_attachment_marker') or ''))}</td>"
                    f"<td><code>{html.escape(str(support.get('bbox_xyxy')))}</code></td>"
                    f"<td>{html.escape(str(support.get('admission_status')))}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
        statecons = view.get("state_condition_hypotheses", [])
        if isinstance(statecons, list) and statecons:
            body.extend(
                [
                    "<table><thead><tr><th>state condition</th><th>group</th><th>status</th><th>relation score</th><th>GeoBand</th><th>features</th><th>admission</th></tr></thead><tbody>",
                ]
            )
            for statecon in statecons[:12]:
                if not isinstance(statecon, dict):
                    continue
                group = statecon.get("sticky_group") if isinstance(statecon.get("sticky_group"), dict) else {}
                features = statecon.get("features") if isinstance(statecon.get("features"), dict) else {}
                geoband = statecon.get("geoband_hypothesis") if isinstance(statecon.get("geoband_hypothesis"), dict) else {}
                angles = geoband.get("angles") if isinstance(geoband.get("angles"), dict) else {}
                bandwidth = geoband.get("bandwidth") if isinstance(geoband.get("bandwidth"), dict) else {}
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(statecon.get('statecon_id')))}</code><br>{html.escape(str(statecon.get('condition_kind')))}</td>"
                    f"<td>actor <code>{html.escape(str(group.get('actor_affordance_id')))}</code><br>target <code>{html.escape(str(group.get('target_affordance_id')))}</code></td>"
                    f"<td>{html.escape(str(statecon.get('condition_status_hypothesis')))}</td>"
                    f"<td>{float(statecon.get('relation_descriptor_score') or statecon.get('event_enable_score') or 0.0):.3f}</td>"
                    f"<td>err={angles.get('flow_to_target_angular_error_degrees')} tol={angles.get('angular_tolerance_degrees')} sym={bandwidth.get('symmetry_score')}</td>"
                    f"<td>ray={features.get('ray_intersects_target')} axis={features.get('axis_alignment')} prox={features.get('directional_proximity')}</td>"
                    f"<td>{html.escape(str(statecon.get('admission_status')))}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
        body.extend(
            [
                "<table><thead><tr><th>role</th><th>component</th><th>raw</th><th>score</th><th>facing</th><th>asym</th><th>reasons</th></tr></thead><tbody>",
            ]
        )
        floor = view["scene_surface_hypotheses"].get("floor_hypothesis")
        surface_rows = []
        if isinstance(floor, dict):
            surface_rows.append(("floor", floor))
        surface_rows.extend(("wall", row) for row in view["scene_surface_hypotheses"].get("wall_hypotheses", []))
        surface_rows.extend(("background", row) for row in view["scene_surface_hypotheses"].get("background_hypotheses", []))
        surface_rows.extend(
            ("residual", row) for row in view["scene_surface_hypotheses"].get("residual_component_hypotheses", [])[:20]
        )
        for role, row in surface_rows:
            asymmetry = row.get("asymmetry") if isinstance(row.get("asymmetry"), dict) else {}
            facing = row.get("facing_direction_hypothesis") or asymmetry.get("facing_direction_hypothesis") or "none"
            confidence = row.get("facing_direction_confidence")
            if confidence is None:
                confidence = asymmetry.get("confidence")
            body.append(
                "<tr>"
                f"<td>{html.escape(role)}</td>"
                f"<td><code>{html.escape(str(row['component_id']))}</code></td>"
                f"<td>{row['raw_pixel_value']}</td>"
                f"<td>{float(row['score']):.3f}</td>"
                f"<td>{html.escape(str(facing))} {float(confidence or 0.0):.2f}</td>"
                f"<td>x={float(asymmetry.get('mirror_x_difference') or 0.0):.2f} y={float(asymmetry.get('mirror_y_difference') or 0.0):.2f}</td>"
                f"<td>{html.escape(', '.join(row.get('reasons') or []))}</td>"
                "</tr>"
            )
        relationships = view["scene_surface_hypotheses"].get("antisymmetric_relationship_hypotheses", [])
        if relationships:
            body.extend(
                [
                    "</tbody></table>",
                    "<table><thead><tr><th>antisymmetric pair</th><th>axis</th><th>facing</th><th>score</th><th>reasons</th></tr></thead><tbody>",
                ]
            )
            for relationship in relationships[:16]:
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(relationship['component_a']))}</code> / <code>{html.escape(str(relationship['component_b']))}</code></td>"
                    f"<td>{html.escape(str(relationship['axis']))}</td>"
                    f"<td>{html.escape(str(relationship['facing_direction_a']))} / {html.escape(str(relationship['facing_direction_b']))}</td>"
                    f"<td>{float(relationship['score']):.3f}</td>"
                    f"<td>{html.escape(', '.join(relationship.get('reasons') or []))}</td>"
                    "</tr>"
                )
        body.extend(
            [
                "</tbody></table>",
                "<table><thead><tr><th>raw</th><th>color</th><th>pixels</th><th>components</th><th>background</th><th>mask</th><th>overlay</th></tr></thead><tbody>",
            ]
        )
        for color in view["colors"]:
            rgb = value_rgb(int(color["raw_pixel_value"]))
            swatch = f"<span class='swatch' style='background:rgb({rgb[0]},{rgb[1]},{rgb[2]})'></span>"
            body.append(
                "<tr>"
                f"<td>{color['raw_pixel_value']}</td>"
                f"<td>{swatch}{html.escape(color['palette_name'])}</td>"
                f"<td>{color['pixel_count']}</td>"
                f"<td>{color['component_count']}</td>"
                f"<td>{str(color['dominant_background_candidate']).lower()}</td>"
                f"<td><a href='{rel_for_html(color['mask_png']['path'], index_dir)}'>mask</a></td>"
                f"<td><a href='{rel_for_html(color['overlay_png']['path'], index_dir)}'>overlay</a></td>"
                "</tr>"
            )
        body.extend(["</tbody></table></section>"])
    body.extend(["</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("codexlab/runs/sp80_sam_color_components/frames"),
        help="Directory containing recorded level_*_raw.npy frames.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_color_board_viewer"))
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--min-component-area", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    frames_dir = resolve_repo_path(args.frames_dir)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = max(1, int(args.scale))
    min_component_area = max(1, int(args.min_component_area))
    views: list[dict[str, Any]] = []
    calibration_by_level = load_calibration_by_level()
    for level_index, frame_path, frame in load_frames(frames_dir):
        inventory = raw_color_inventory(frame, min_component_area=min_component_area)
        calibration = calibration_by_level.get(int(level_index), {})
        source_raw_frame_sha256 = sha256_frame(frame)
        perception = build_contra_perception_pipeline(
            frame=frame,
            level_index=int(level_index),
            source_raw_frame_sha256=source_raw_frame_sha256,
            raw_color_inventory=inventory,
            calibration=calibration,
            actor_builder=actor_affordance_hypotheses,
            target_builder=target_affordance_hypotheses,
        )
        scene_surfaces = perception["scene_surface_hypotheses"]
        component_label_map = perception["component_label_map"]
        component_lookup = perception["component_lookup"]
        actor_affordances = perception["actor_affordance_hypotheses"]
        target_affordances = perception["target_affordance_hypotheses"]
        support_affordances = perception["support_action_affordance_hypotheses"]
        statecons = perception["state_condition_hypotheses"]
        relationship_ladder = perception["contra_relationship_ladder"]
        manual_calibration_review = build_sp80_manual_calibration_review(
            level_index=int(level_index),
            source_raw_frame_sha256=source_raw_frame_sha256,
            source_raw_grid_sha256=sha256_file(frame_path),
            calibration=calibration,
            support_affordances=support_affordances,
        )
        deltaband_controlled_event_requirement = build_deltaband_controlled_event_requirement(
            level_index=int(level_index),
            source_raw_frame_sha256=source_raw_frame_sha256,
            relationship_ladder=relationship_ladder,
        )
        raw_board = save_raw_board(frame, out_dir / "boards" / f"level_{level_index:02d}_raw_board.png", scale)
        scene_surface_overlay = save_scene_surface_overlay(
            frame,
            scene_surfaces=scene_surfaces,
            label_map=component_label_map,
            component_lookup=component_lookup,
            target_affordances=target_affordances,
            actor_affordances=actor_affordances,
            statecons=statecons,
            path=out_dir / "scene_surface_overlays" / f"level_{level_index:02d}_surface_hypotheses.png",
            scale=scale,
        )
        scene_surface_sheet = save_scene_surface_sheet(
            frame=frame,
            scene_surfaces=scene_surfaces,
            label_map=component_label_map,
            component_lookup=component_lookup,
            target_affordances=target_affordances,
            actor_affordances=actor_affordances,
            statecons=statecons,
            path=out_dir / "scene_surface_sheets" / f"level_{level_index:02d}_surface_hypotheses.png",
            scale=scale,
        )
        colors: list[dict[str, Any]] = []
        for color in inventory["colors"]:
            value = int(color["raw_pixel_value"])
            components = list(color.get("components") or [])
            mask_png = save_color_mask(
                frame,
                value=value,
                components=components,
                path=out_dir / "color_masks" / f"level_{level_index:02d}_v{value:02d}_{color['palette_name']}.png",
                scale=scale,
            )
            overlay_png = save_color_overlay(
                frame,
                value=value,
                components=components,
                path=out_dir / "color_overlays" / f"level_{level_index:02d}_v{value:02d}_{color['palette_name']}.png",
                scale=scale,
            )
            colors.append({**color, "mask_png": mask_png, "overlay_png": overlay_png})
        review_sheet = save_level_sheet(
            frame=frame,
            colors=colors,
            path=out_dir / "review_sheets" / f"level_{level_index:02d}_color_board.png",
            scale=scale,
        )
        views.append(
            {
                "level_index": int(level_index),
                "source_raw_grid": relpath(frame_path),
                "source_raw_grid_sha256": sha256_file(frame_path),
                "source_raw_frame_sha256": source_raw_frame_sha256,
                "frame_shape": [int(v) for v in frame.shape],
                "raw_board": raw_board,
                "scene_surface_overlay": scene_surface_overlay,
                "scene_surface_sheet": scene_surface_sheet,
                "scene_surface_hypotheses": scene_surfaces,
                "actor_affordance_hypotheses": actor_affordances,
                "target_affordance_hypotheses": target_affordances,
                "statecon_support_affordances": support_affordances,
                "manual_support_affordances": perception["manual_support_affordances"],
                "candidate_support_affordances": perception["candidate_support_affordances"],
                "support_action_affordance_hypotheses": support_affordances,
                "state_condition_hypotheses": statecons,
                "contra_relationship_ladder": relationship_ladder,
                "manual_calibration_review": manual_calibration_review,
                "deltaband_controlled_event_requirement": deltaband_controlled_event_requirement,
                "contra_perception_pipeline": {
                    key: value
                    for key, value in perception.items()
                    if key not in {"component_label_map", "component_lookup"}
                },
                "review_sheet": review_sheet,
                "raw_color_inventory": inventory,
                "colors": colors,
            }
        )
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "sp80_exact_color_board_viewer",
        "condition_natural_language": "Render exact raw color boards and per-color separations from recorded raw frame grids.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "frames_dir": relpath(frames_dir),
        "scale": scale,
        "min_component_area_pixels": min_component_area,
        "primary_color_values": sorted(PRIMARY_COLOR_VALUES),
        "scene_surface_hypotheses_schema": "codexlab.scene_surface_hypotheses.v3",
        "scene_surface_hypotheses_status": "hypothesis_not_admitted",
        "contra_perception_pipeline_schema": CONTRA_PERCEPTION_PIPELINE_SCHEMA,
        "contra_perception_pipeline_status": "unified_hypothesis_pipeline_not_admitted",
        "target_affordance_hypotheses_schema": TARGET_AFFORDANCE_SET_SCHEMA,
        "target_affordance_hypotheses_status": "manual_hypothesis_not_admitted",
        "target_affordance_control_truth_admitted": False,
        "target_affordance_goal_condition_admitted": False,
        "target_affordance_action_choice_admitted": False,
        "actor_affordance_hypotheses_schema": ACTOR_AFFORDANCE_SET_SCHEMA,
        "actor_affordance_hypotheses_status": "manual_hypothesis_not_admitted",
        "actor_affordance_control_truth_admitted": False,
        "actor_affordance_goal_condition_admitted": False,
        "actor_affordance_action_choice_admitted": False,
        "support_action_affordance_hypotheses_schema": SUPPORT_ACTION_AFFORDANCE_SET_SCHEMA,
        "support_action_affordance_hypotheses_status": "manual_or_heuristic_hypothesis_not_admitted",
        "support_action_affordance_control_truth_admitted": False,
        "support_action_affordance_goal_condition_admitted": False,
        "support_action_affordance_action_choice_admitted": False,
        "state_condition_hypotheses_schema": STATECON_SET_SCHEMA,
        "state_condition_hypotheses_status": "mlp_scored_hypothesis_not_admitted",
        "state_condition_identity_model_schema": "codexlab.statecon_identity_mlp.v1",
        "state_condition_identity_model_status": "deterministic_untrained_mlp_scaffold",
        "geoband_hypotheses_schema": GEOBAND_HYPOTHESIS_SCHEMA,
        "geoband_hypotheses_status": "static_geometry_hypothesis_not_admitted",
        "contra_relationship_ladder_schema": CONTRA_RELATIONSHIP_LADDER_SCHEMA,
        "contra_relationship_ladder_status": "hypothesis_admission_ladder_not_admitted",
        "manual_calibration_review_schema": SP80_MANUAL_CALIBRATION_REVIEW_SCHEMA,
        "manual_calibration_review_status": "requires_human_review_not_admitted",
        "deltaband_observations_schema": DELTABAND_SET_SCHEMA,
        "deltaband_observations_status": "not_run_requires_controlled_before_after_action_boundary",
        "deltaband_required_boundary": DELTABAND_REQUIRED_BOUNDARY,
        "deltaband_prediction_status": "not_predictive",
        "deltaband_controlled_event_requirement_schema": DELTABAND_CONTROLLED_EVENT_REQUIREMENT_SCHEMA,
        "deltaband_controlled_event_requirement_status": "not_run_requires_controlled_before_after_action_boundary",
        "state_condition_control_truth_admitted": False,
        "state_condition_goal_condition_admitted": False,
        "state_condition_action_choice_admitted": False,
        "state_condition_solve_condition_admitted": False,
        "view_count": len(views),
        "views": views,
        "sam_status": "not_run",
        "semantic_labels_used": False,
        "controlled_labels_used": False,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "source_script": "codexlab/sp80_color_board_viewer.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_color_board_viewer.py"),
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
                "status": "exported",
                "manifest": relpath(manifest_path),
                "index_html": relpath(index_path),
                "view_count": len(views),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
