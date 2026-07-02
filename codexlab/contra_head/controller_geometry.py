"""Pure geometry helpers for the CONTRA controller head."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def as_raw_grid(raw_frame: Any) -> np.ndarray:
    arr = np.asarray(raw_frame)
    if arr.ndim != 2:
        raise ValueError(f"CONTRA expects a 2D raw frame grid, got shape {arr.shape}")
    return arr.astype(np.int16, copy=False)


def bbox_from_component(component: dict[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in component["bbox_xyxy"])  # type: ignore[return-value]


def union_bbox(*boxes: tuple[int, int, int, int]) -> list[int]:
    return [
        int(min(box[0] for box in boxes)),
        int(min(box[1] for box in boxes)),
        int(max(box[2] for box in boxes)),
        int(max(box[3] for box in boxes)),
    ]


def square_from_bbox(bbox: tuple[int, int, int, int], *, frame_shape: tuple[int, int]) -> list[int]:
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


def opposite_direction(direction: str) -> str:
    return {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }.get(str(direction), "none")


def direction_between_components(source: dict[str, Any], target: dict[str, Any]) -> str:
    source_center = source.get("center_xy") if isinstance(source.get("center_xy"), list) else [0.0, 0.0]
    target_center = target.get("center_xy") if isinstance(target.get("center_xy"), list) else [0.0, 0.0]
    dx = float(target_center[0]) - float(source_center[0])
    dy = float(target_center[1]) - float(source_center[1])
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


def short_digest(parts: list[str]) -> str:
    basis = "|".join(str(part) for part in parts)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
