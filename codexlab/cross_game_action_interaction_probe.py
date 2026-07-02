#!/usr/bin/env python3
"""Controlled non-SP80 action probe for PortBand interaction tagging.

The runner executes real local ARC game actions, records before/action/after
raw frames, derives candidate action-surface and relative-state observations
from the observed frame delta, and sends them through PortBand surface feedback.
The resulting report is controlled action evidence for the executed action
boundary only; it is not solve truth or action-choice authority.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import inspect
import json
import math
import os
import socket
import sys
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from arcengine import ARCBaseGame, ActionInput, GameAction
from contra_head import (
    CONTRA_ACTION_INTERACTION_EDGE_SCHEMA,
    CONTRA_ACTION_INTERACTION_MATRIX_SCHEMA,
    CONTRA_OBJECT_CORRESPONDENCE_SCHEMA,
    PORTBAND_ALIGNMENT_SCHEMA,
    PORTBAND_LINEUP_CONFIDENCE_MODEL,
    PORTBAND_LINEUP_MATRIX_ROW_SCHEMA,
    PORTBAND_LINEUP_MATRIX_SCHEMA,
    apply_portband_surface_feedback_to_statecons,
)
from sp80_base_controller import PALETTE, ROOT, sha256_file, sha256_frame


SCHEMA = "codexlab.cross_game_action_interaction_probe.v1"
CASE_SCHEMA = "codexlab.cross_game_action_interaction_case.v1"
TEMPORAL_TRAJECTORY_SCHEMA = "codexlab.cross_game_action_temporal_trajectory_profile.v1"
RELATIONSHIP_ATLAS_SCHEMA = "codexlab.cross_game_action_relationship_atlas.v1"
RELATIONSHIP_ATLAS_ROW_SCHEMA = "codexlab.cross_game_action_relationship_atlas_row.v1"
ENV_ROOT = ROOT / "environment" / "environment_files"
TRUTH_FLAGS = {
    "admission_status": "not_admitted",
    "not_prediction": True,
    "not_control_truth": True,
    "not_solve_truth": True,
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}
ACTION_ORDER = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
    GameAction.ACTION7,
]


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def safe_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def stable_digest(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_games() -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for meta_path in sorted(ENV_ROOT.glob("*/*/metadata.json")):
        game_short = meta_path.parents[1].name
        if game_short == "sp80":
            continue
        game_hash = meta_path.parent.name
        py_path = meta_path.parent / f"{game_short}.py"
        if not py_path.is_file():
            continue
        metadata = load_json(meta_path)
        games.append(
            {
                "game_id": str(metadata.get("game_id") or f"{game_short}-{game_hash}"),
                "title": str(metadata.get("title") or game_short),
                "game_short": game_short,
                "game_hash": game_hash,
                "metadata_path": meta_path,
                "source_path": py_path,
                "source_sha256": sha256_file(py_path),
            }
        )
    return games


def load_game_class(path: Path) -> type[ARCBaseGame]:
    module_name = f"codexlab_cross_game_action_probe_{path.parent.parent.name}_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for obj in vars(module).values():
        if inspect.isclass(obj) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
            return obj
    raise RuntimeError(f"no ARCBaseGame subclass found in {path}")


def normalize_raw_frame(frame_value: Any) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(frame_value, dtype=np.int16)
    raw_shape = [int(v) for v in arr.shape]
    returned_frame_count = 1
    if arr.ndim == 3:
        returned_frame_count = int(arr.shape[0])
        arr = arr[-1]
    if arr.ndim != 2:
        raise RuntimeError(f"expected 2D or frame-sequence raw frame, got shape={raw_shape}")
    return np.asarray(arr, dtype=np.int16), {
        "raw_return_shape": raw_shape,
        "normalized_shape": [int(v) for v in arr.shape],
        "returned_frame_count": returned_frame_count,
    }


def normalize_raw_frame_sequence(frame_value: Any) -> tuple[list[np.ndarray], dict[str, Any]]:
    arr = np.asarray(frame_value, dtype=np.int16)
    raw_shape = [int(v) for v in arr.shape]
    if arr.ndim == 2:
        frames = [np.asarray(arr, dtype=np.int16)]
    elif arr.ndim == 3:
        frames = [np.asarray(arr[index], dtype=np.int16) for index in range(int(arr.shape[0]))]
    else:
        raise RuntimeError(f"expected 2D or frame-sequence raw frame, got shape={raw_shape}")
    final_frame = frames[-1]
    return frames, {
        "raw_return_shape": raw_shape,
        "normalized_shape": [int(v) for v in final_frame.shape],
        "returned_frame_count": len(frames),
        "sequence_frame_hashes": [sha256_frame(frame) for frame in frames],
    }


def raw_image(frame: np.ndarray, *, scale: int) -> Image.Image:
    arr = np.asarray(frame, dtype=np.int64).clip(0, len(PALETTE) - 1)
    rgb = PALETTE[arr]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def save_frame_artifacts(frame: np.ndarray, *, png_path: Path, npy_path: Path, scale: int) -> dict[str, Any]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame, dtype=np.int16)
    np.save(npy_path, arr)
    raw_image(arr, scale=scale).save(png_path)
    return {
        "png": {"path": relpath(png_path), "sha256": sha256_file(png_path)},
        "raw_grid": {
            "path": relpath(npy_path),
            "sha256": sha256_file(npy_path),
            "raw_frame_sha256": sha256_frame(arr),
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
        },
    }


def save_frame_sequence_artifacts(frames: list[np.ndarray], *, directory: Path, scale: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        rows.append(
            {
                "frame_index": index,
                **save_frame_artifacts(
                    frame,
                    png_path=directory / f"frame_{index:03d}.png",
                    npy_path=directory / f"frame_{index:03d}.npy",
                    scale=scale,
                ),
            }
        )
    return rows


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_union(*boxes: list[int] | None) -> list[int]:
    present = [box for box in boxes if isinstance(box, list) and len(box) == 4]
    if not present:
        return [0, 0, 0, 0]
    return [
        min(int(box[0]) for box in present),
        min(int(box[1]) for box in present),
        max(int(box[2]) for box in present),
        max(int(box[3]) for box in present),
    ]


def bbox_area(box: list[int] | None) -> int:
    if not isinstance(box, list) or len(box) != 4:
        return 0
    return max(0, int(box[2]) - int(box[0]) + 1) * max(0, int(box[3]) - int(box[1]) + 1)


def bbox_width(box: list[int] | None) -> int:
    if not isinstance(box, list) or len(box) != 4:
        return 0
    return max(0, int(box[2]) - int(box[0]) + 1)


def bbox_height(box: list[int] | None) -> int:
    if not isinstance(box, list) or len(box) != 4:
        return 0
    return max(0, int(box[3]) - int(box[1]) + 1)


def bbox_diagonal(box: list[int] | None) -> float:
    return round(math.hypot(float(bbox_width(box)), float(bbox_height(box))), 6)


def bbox_aspect_ratio(box: list[int] | None) -> float:
    height = bbox_height(box)
    if height <= 0:
        return 0.0
    return round(float(bbox_width(box)) / float(height), 6)


def bbox_overlap_area(left: list[int], right: list[int]) -> int:
    x0 = max(int(left[0]), int(right[0]))
    y0 = max(int(left[1]), int(right[1]))
    x1 = min(int(left[2]), int(right[2]))
    y1 = min(int(left[3]), int(right[3]))
    if x1 < x0 or y1 < y0:
        return 0
    return (x1 - x0 + 1) * (y1 - y0 + 1)


def bbox_iou(left: list[int], right: list[int]) -> float:
    overlap = bbox_overlap_area(left, right)
    union = bbox_area(left) + bbox_area(right) - overlap
    if union <= 0:
        return 0.0
    return round(float(overlap) / float(union), 6)


def center_of_bbox(box: list[int]) -> list[float]:
    return [round((float(box[0]) + float(box[2])) / 2.0, 6), round((float(box[1]) + float(box[3])) / 2.0, 6)]


def center_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def direction_8way(dx: float, dy: float) -> str:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "stable"
    if abs(dx) >= 2.0 * abs(dy):
        return "east" if dx > 0 else "west"
    if abs(dy) >= 2.0 * abs(dx):
        return "south" if dy > 0 else "north"
    if dx > 0 and dy > 0:
        return "south_east"
    if dx > 0 and dy < 0:
        return "north_east"
    if dx < 0 and dy > 0:
        return "south_west"
    return "north_west"


def axis_family(dx: float, dy: float) -> str:
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return "stable"
    if abs(dx) >= 1.5 * abs(dy):
        return "horizontal_dominant"
    if abs(dy) >= 1.5 * abs(dx):
        return "vertical_dominant"
    return "diagonal_or_balanced"


def delta_bucket(value: float) -> str:
    abs_value = abs(float(value))
    if abs_value >= 45.0:
        return "very_large"
    if abs_value >= 10.0:
        return "large"
    if abs_value >= 2.0:
        return "medium"
    if abs_value > 0.0:
        return "small"
    return "zero"


def frame_delta_summary(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    before_arr = np.asarray(before, dtype=np.int16)
    after_arr = np.asarray(after, dtype=np.int16)
    if before_arr.shape != after_arr.shape:
        return {
            "shape_match": False,
            "before_shape": [int(v) for v in before_arr.shape],
            "after_shape": [int(v) for v in after_arr.shape],
            "changed_pixel_count": 0,
            "changed_bbox_xyxy": None,
            "changed_center_xy": None,
        }
    mask = before_arr != after_arr
    changed = int(np.count_nonzero(mask))
    bbox = bbox_from_mask(mask) if changed > 0 else None
    return {
        "shape_match": True,
        "changed_pixel_count": changed,
        "changed_ratio": round(float(changed) / float(max(1, mask.size)), 6),
        "changed_bbox_xyxy": bbox,
        "changed_center_xy": center_of_bbox(bbox) if isinstance(bbox, list) else None,
    }


def action_temporal_trajectory_profile(before_frame: np.ndarray, action_frames: list[np.ndarray]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    previous = before_frame
    for index, frame in enumerate(action_frames):
        vs_before = frame_delta_summary(before_frame, frame)
        vs_previous = frame_delta_summary(previous, frame)
        rows.append(
            {
                "frame_index": index,
                "raw_frame_sha256": sha256_frame(frame),
                "changed_pixel_count_vs_before": vs_before.get("changed_pixel_count"),
                "changed_bbox_xyxy_vs_before": vs_before.get("changed_bbox_xyxy"),
                "changed_center_xy_vs_before": vs_before.get("changed_center_xy"),
                "changed_pixel_count_from_previous": vs_previous.get("changed_pixel_count"),
                "changed_bbox_xyxy_from_previous": vs_previous.get("changed_bbox_xyxy"),
                "changed_center_xy_from_previous": vs_previous.get("changed_center_xy"),
                "shape_match": vs_before.get("shape_match") is True and vs_previous.get("shape_match") is True,
            }
        )
        previous = frame
    path_points = [
        row.get("changed_center_xy_vs_before")
        for row in rows
        if isinstance(row.get("changed_center_xy_vs_before"), list)
    ]
    if len(path_points) >= 2:
        first_point = path_points[0]
        last_point = path_points[-1]
        dx = round(float(last_point[0]) - float(first_point[0]), 6)
        dy = round(float(last_point[1]) - float(first_point[1]), 6)
        path_distance = round(sum(center_distance(path_points[i - 1], path_points[i]) for i in range(1, len(path_points))), 6)
    else:
        first_point = path_points[0] if path_points else None
        last_point = path_points[-1] if path_points else None
        dx = 0.0
        dy = 0.0
        path_distance = 0.0
    peak = max(rows, key=lambda row: int(row.get("changed_pixel_count_vs_before") or 0)) if rows else {}
    peak_step = max(rows, key=lambda row: int(row.get("changed_pixel_count_from_previous") or 0)) if rows else {}
    vector_magnitude = round(math.hypot(dx, dy), 6)
    return {
        "schema": TEMPORAL_TRAJECTORY_SCHEMA,
        "status": "profiled",
        "trajectory_status": "multi_frame_action_sequence_profiled" if len(action_frames) > 1 else "single_frame_action_result_profiled",
        "frame_count": len(action_frames),
        "multi_frame_available": len(action_frames) > 1,
        "nonzero_frame_count": sum(1 for row in rows if int(row.get("changed_pixel_count_vs_before") or 0) > 0),
        "path_points": path_points,
        "first_changed_center_xy": first_point,
        "last_changed_center_xy": last_point,
        "trajectory_vector": {
            "dx": dx,
            "dy": dy,
            "euclidean_magnitude": vector_magnitude,
            "magnitude_bucket": delta_bucket(vector_magnitude),
            "direction_8way": direction_8way(dx, dy),
            "axis_family": axis_family(dx, dy),
        },
        "path_distance": path_distance,
        "path_distance_px": path_distance,
        "path_distance_bucket": delta_bucket(path_distance),
        "peak_changed_frame_index": peak.get("frame_index"),
        "peak_changed_pixel_count": peak.get("changed_pixel_count_vs_before", 0),
        "peak_step_frame_index": peak_step.get("frame_index"),
        "peak_step_changed_pixel_count": peak_step.get("changed_pixel_count_from_previous", 0),
        "final_changed_pixel_count": rows[-1].get("changed_pixel_count_vs_before", 0) if rows else 0,
        "frame_deltas": rows,
        "trajectory_terms": [
            f"frames:{len(action_frames)}",
            f"trajectory:{direction_8way(dx, dy)}:{axis_family(dx, dy)}:{delta_bucket(vector_magnitude)}",
            f"path:{delta_bucket(path_distance)}",
            f"peak_frame:{peak.get('frame_index')}",
        ],
        "evidence_status": "temporal_profile_from_controlled_action_return_frames_not_truth",
        **TRUTH_FLAGS,
    }


def connected_components(frame: np.ndarray) -> list[dict[str, Any]]:
    values, counts = np.unique(frame, return_counts=True)
    background_value = int(values[int(np.argmax(counts))]) if len(values) else 0
    h, w = frame.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            if visited[y, x] or int(frame[y, x]) == background_value:
                continue
            raw_value = int(frame[y, x])
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        continue
                    if visited[ny, nx] or int(frame[ny, nx]) != raw_value:
                        continue
                    visited[ny, nx] = True
                    queue.append((nx, ny))
            xs = [cell[0] for cell in cells]
            ys = [cell[1] for cell in cells]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            components.append(
                {
                    "component_id": f"component_value_{raw_value}_{min(xs)}_{min(ys)}_{max(xs)}_{max(ys)}_{len(cells)}",
                    "raw_pixel_value": raw_value,
                    "area_pixels": len(cells),
                    "bbox_xyxy": bbox,
                    "center_xy": center_of_bbox(bbox),
                }
            )
    components.sort(key=lambda row: (-int(row["area_pixels"]), str(row["component_id"])))
    return components


def nearest_component(components: list[dict[str, Any]], center_xy: list[float]) -> dict[str, Any] | None:
    if not components:
        return None
    return min(
        components,
        key=lambda row: (
            (float(row["center_xy"][0]) - float(center_xy[0])) ** 2
            + (float(row["center_xy"][1]) - float(center_xy[1])) ** 2,
            -int(row.get("area_pixels") or 0),
            str(row.get("component_id")),
        ),
    )


def component_for_change(
    components: list[dict[str, Any]],
    *,
    change_bbox: list[int],
    change_center: list[float],
) -> dict[str, Any] | None:
    if not components:
        return None
    scored = [
        (bbox_overlap_area(row["bbox_xyxy"], change_bbox), int(row.get("area_pixels") or 0), row)
        for row in components
        if isinstance(row.get("bbox_xyxy"), list)
    ]
    scored.sort(key=lambda item: (-item[0], -item[1], str(item[2].get("component_id"))))
    if scored and scored[0][0] > 0:
        return scored[0][2]
    return nearest_component(components, change_center)


def component_change_score(component: dict[str, Any], *, change_bbox: list[int], change_center: list[float], frame_shape: tuple[int, int]) -> float:
    box = component.get("bbox_xyxy") if isinstance(component.get("bbox_xyxy"), list) else [0, 0, 0, 0]
    overlap = bbox_overlap_area(box, change_bbox)
    overlap_ratio = float(overlap) / float(max(1, min(bbox_area(box), bbox_area(change_bbox))))
    diagonal = math.hypot(float(frame_shape[1]), float(frame_shape[0]))
    proximity = 1.0 - min(1.0, center_distance(component.get("center_xy") or [0.0, 0.0], change_center) / max(1.0, diagonal))
    area_bonus = min(1.0, float(component.get("area_pixels") or 0.0) / float(max(1, bbox_area(change_bbox))))
    return round(max(0.0, min(1.0, (0.58 * overlap_ratio) + (0.30 * proximity) + (0.12 * area_bonus))), 6)


def component_candidates_for_change(
    components: list[dict[str, Any]],
    *,
    change_bbox: list[int],
    change_center: list[float],
    frame_shape: tuple[int, int],
    limit: int = 12,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, dict) or not isinstance(component.get("bbox_xyxy"), list):
            continue
        score = component_change_score(component, change_bbox=change_bbox, change_center=change_center, frame_shape=frame_shape)
        rows.append({**component, "change_region_score": score})
    rows.sort(key=lambda row: (-float(row.get("change_region_score") or 0.0), -int(row.get("area_pixels") or 0), str(row.get("component_id"))))
    return rows[:limit]


def object_correspondence_profile(
    *,
    before_components: list[dict[str, Any]],
    after_components: list[dict[str, Any]],
    change_bbox: list[int],
    change_center: list[float],
    frame_shape: tuple[int, int],
) -> dict[str, Any] | None:
    before_candidates = component_candidates_for_change(
        before_components,
        change_bbox=change_bbox,
        change_center=change_center,
        frame_shape=frame_shape,
    )
    after_candidates = component_candidates_for_change(
        after_components,
        change_bbox=change_bbox,
        change_center=change_center,
        frame_shape=frame_shape,
    )
    if not before_candidates or not after_candidates:
        return None
    diagonal = math.hypot(float(frame_shape[1]), float(frame_shape[0]))
    pair_rows: list[dict[str, Any]] = []
    for before in before_candidates:
        for after in after_candidates:
            before_box = [int(v) for v in before["bbox_xyxy"]]
            after_box = [int(v) for v in after["bbox_xyxy"]]
            before_area = int(before.get("area_pixels") or 0)
            after_area = int(after.get("area_pixels") or 0)
            area_similarity = 1.0 - min(1.0, abs(float(after_area - before_area)) / float(max(1, before_area, after_area)))
            raw_match_score = 1.0 if int(before.get("raw_pixel_value") or -1) == int(after.get("raw_pixel_value") or -2) else 0.35
            motion_distance = center_distance(before.get("center_xy") or [0.0, 0.0], after.get("center_xy") or [0.0, 0.0])
            motion_proximity = 1.0 - min(1.0, motion_distance / max(1.0, diagonal))
            spatial_iou = bbox_iou(before_box, after_box)
            score = round(
                max(
                    0.0,
                    min(
                        1.0,
                        (0.24 * float(before.get("change_region_score") or 0.0))
                        + (0.24 * float(after.get("change_region_score") or 0.0))
                        + (0.16 * area_similarity)
                        + (0.14 * raw_match_score)
                        + (0.12 * motion_proximity)
                        + (0.10 * spatial_iou),
                    ),
                ),
                6,
            )
            pair_rows.append(
                {
                    "before_component_id": before.get("component_id"),
                    "after_component_id": after.get("component_id"),
                    "before_raw_pixel_value": before.get("raw_pixel_value"),
                    "after_raw_pixel_value": after.get("raw_pixel_value"),
                    "before_bbox_xyxy": before_box,
                    "after_bbox_xyxy": after_box,
                    "before_center_xy": before.get("center_xy"),
                    "after_center_xy": after.get("center_xy"),
                    "before_change_region_score": before.get("change_region_score"),
                    "after_change_region_score": after.get("change_region_score"),
                    "bbox_iou": spatial_iou,
                    "area_similarity": round(area_similarity, 6),
                    "raw_value_match": int(before.get("raw_pixel_value") or -1) == int(after.get("raw_pixel_value") or -2),
                    "motion_distance_px": round(motion_distance, 6),
                    "motion_proximity_score": round(motion_proximity, 6),
                    "correspondence_score": score,
                }
            )
    pair_rows.sort(
        key=lambda row: (
            -float(row.get("correspondence_score") or 0.0),
            -float(row.get("bbox_iou") or 0.0),
            -float(row.get("area_similarity") or 0.0),
            str(row.get("before_component_id")),
            str(row.get("after_component_id")),
        )
    )
    for rank, row in enumerate(pair_rows, start=1):
        row["candidate_pair_rank"] = rank
    selected = pair_rows[0]
    return {
        "schema": CONTRA_OBJECT_CORRESPONDENCE_SCHEMA,
        "status": "ranked_candidate_pair",
        "correspondence_policy": "rank_before_after_components_by_change_overlap_area_raw_value_and_motion_consistency",
        "confidence_score": selected["correspondence_score"],
        "confidence_model": "change_region_component_pair_score_not_probability",
        "confidence_is_probability": False,
        "selected_pair": selected,
        "candidate_pair_count": len(pair_rows),
        "top_candidate_pairs": pair_rows[:8],
        "before_candidate_count": len(before_candidates),
        "after_candidate_count": len(after_candidates),
        "change_bbox_xyxy": change_bbox,
        "change_center_xy": change_center,
        "evidence_status": "candidate_object_correspondence_from_controlled_frame_delta_not_truth",
        **TRUTH_FLAGS,
    }


def correspondence_profile_for_pair(
    *,
    base_profile: dict[str, Any],
    selected_pair: dict[str, Any],
    pair_index: int,
) -> dict[str, Any]:
    return {
        **base_profile,
        "status": "ranked_candidate_pair",
        "selected_pair": selected_pair,
        "selected_pair_rank": int(selected_pair.get("candidate_pair_rank") or pair_index),
        "candidate_pair_index": int(pair_index),
        "confidence_score": selected_pair.get("correspondence_score"),
        "evidence_status": "candidate_object_correspondence_from_controlled_frame_delta_not_truth",
        **TRUTH_FLAGS,
    }


def select_candidate_interaction_pairs(
    *,
    object_correspondence: dict[str, Any],
    before_components: list[dict[str, Any]],
    after_components: list[dict[str, Any]],
    max_pairs: int,
) -> list[dict[str, Any]]:
    before_by_id = {str(row.get("component_id")): row for row in before_components if isinstance(row, dict)}
    after_by_id = {str(row.get("component_id")): row for row in after_components if isinstance(row, dict)}
    pair_rows = [row for row in object_correspondence.get("top_candidate_pairs", []) or [] if isinstance(row, dict)]
    selected_pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in pair_rows:
        before_id = str(row.get("before_component_id") or "")
        after_id = str(row.get("after_component_id") or "")
        key = (before_id, after_id)
        if not before_id or not after_id or key in seen:
            continue
        source_component = before_by_id.get(before_id)
        target_component = after_by_id.get(after_id)
        if source_component is None or target_component is None:
            continue
        seen.add(key)
        pair_index = len(selected_pairs) + 1
        selected_pairs.append(
            {
                "pair_index": pair_index,
                "selected_pair": row,
                "source_component": source_component,
                "target_component": target_component,
                "object_correspondence": correspondence_profile_for_pair(
                    base_profile=object_correspondence,
                    selected_pair=row,
                    pair_index=pair_index,
                ),
            }
        )
        if len(selected_pairs) >= max(1, int(max_pairs)):
            break
    return selected_pairs


def action_label(action: GameAction) -> str:
    return action.name.lower()


def statecon_from_observation(
    *,
    case_id: str,
    source_component: dict[str, Any],
    target_component: dict[str, Any],
) -> dict[str, Any]:
    return statecon_from_component_pairs(
        case_id=case_id,
        component_pairs=[
            {
                "pair_index": 1,
                "source_component": source_component,
                "target_component": target_component,
            }
        ],
    )


def statecon_from_component_pairs(
    *,
    case_id: str,
    component_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    statecon_id = f"statecon_{case_id}"
    alignments: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    support_object_ids: list[str] = []
    for pair in component_pairs:
        pair_index = int(pair.get("pair_index") or (len(alignments) + 1))
        source_component = pair["source_component"]
        target_component = pair["target_component"]
        pair_id = f"pair_{pair_index:02d}"
        portband_id = f"portband_{case_id}_{pair_id}"
        source_object_id = f"{case_id}:before:{source_component['component_id']}"
        target_object_id = f"{case_id}:after:{target_component['component_id']}"
        source_bbox = [int(v) for v in source_component["bbox_xyxy"]]
        target_bbox = [int(v) for v in target_component["bbox_xyxy"]]
        support_object_ids.append(source_object_id)
        alignments.append(
            {
                "schema": PORTBAND_ALIGNMENT_SCHEMA,
                "status": "hypothesis",
                "statecon_id": statecon_id,
                "portband_id": portband_id,
                "source_object_ids": [source_object_id],
                "source_component_ids": [str(source_component["component_id"])],
                "source_bboxes_xyxy": [source_bbox],
                "target_object_ids": [target_object_id],
                "target_component_ids": [str(target_component["component_id"])],
                "target_bboxes_xyxy": [target_bbox],
                "target_port_ids": [f"target_port_{case_id}_{pair_id}"],
                "transform_hypothesis_xy": [int(target_bbox[0]) - int(source_bbox[0]), int(target_bbox[1]) - int(source_bbox[1])],
                "imagined_source_bboxes_xyxy": [target_bbox],
                "alignment_kind": "controlled_cross_game_before_after_component_alignment",
                "match_status": "derived_from_observed_before_after_delta",
                "candidate_pair_index": pair_index,
                "alignment_error_px": 0.0,
                "span_overlap_ratio": 1.0,
                "exact_alignment": True,
                "near_alignment": True,
                "evidence_status": "controlled_cross_game_frame_delta_component_alignment_not_truth",
                **TRUTH_FLAGS,
            }
        )
        rows.append(
            {
                "schema": PORTBAND_LINEUP_MATRIX_ROW_SCHEMA,
                "status": "hypothesis",
                "lineup_matrix_row_id": f"lineup_{case_id}_{pair_id}",
                "statecon_id": statecon_id,
                "row_kind": "direct_port_alignment_lineup",
                "source_object_id": source_object_id,
                "source_object_role": "support_group",
                "source_component_ids": [str(source_component["component_id"])],
                "target_object_ids": [target_object_id],
                "target_component_ids": [str(target_component["component_id"])],
                "target_port_ids": [f"target_port_{case_id}_{pair_id}"],
                "lineup_route_label": "translate_source_object_by_observed_component_delta_to_lineup_preview",
                "lineup_route_status": "controlled_before_after_delta_lineup_hypothesis_not_executable_route",
                "route_authority": "none_static_review_only",
                "candidate_pair_index": pair_index,
                "lineup_score": 0.5,
                "confidence_score": 0.5,
                "confidence_is_probability": False,
                "linked_alignment_ids": [portband_id],
                "linked_fleet_candidate_ids": [],
                "evidence_status": "controlled_cross_game_frame_delta_component_alignment_not_truth",
                **TRUTH_FLAGS,
            }
        )
    return {
        "statecon_id": statecon_id,
        "portband_hypotheses": {
            "statecon_id": statecon_id,
            "alignments": alignments,
            "lineup_matrix": {
                "schema": PORTBAND_LINEUP_MATRIX_SCHEMA,
                "status": "ranked",
                "statecon_id": statecon_id,
                "confidence_model": PORTBAND_LINEUP_CONFIDENCE_MODEL,
                "confidence_is_probability": False,
                "row_count": len(rows),
                "support_object_count": len(set(support_object_ids)),
                "support_object_ids": sorted(set(support_object_ids)),
                "best_by_support_object": [],
                "rows": rows,
                "evidence_status": "controlled_cross_game_frame_delta_component_alignment_not_truth",
                **TRUTH_FLAGS,
            },
        },
    }


def build_surface_event(
    *,
    case_id: str,
    action: GameAction,
    pair_index: int = 1,
    source_component: dict[str, Any],
    target_component: dict[str, Any],
    surface_bbox: list[int],
    changed_pixel_count: int,
    confidence: float = 0.72,
) -> dict[str, Any]:
    pair_id = f"pair_{int(pair_index):02d}"
    return {
        "schema": "codexlab.deltaband.action_surface_result_event.v1",
        "status": "observed",
        "event_id": f"surface_result_{case_id}_{action.name.lower()}_{pair_id}",
        "event_rank": int(pair_index),
        "action_invocation_id": f"action_{case_id}_{action.name.lower()}",
        "action_primitive": action_label(action),
        "candidate_pair_index": int(pair_index),
        "surface": {
            "surface_object_id": f"{case_id}:after:{target_component['component_id']}",
            "source_object_id": f"{case_id}:before:{source_component['component_id']}",
            "surface_role": "observed_cross_game_action_delta_surface",
            "bbox_xyxy": surface_bbox,
            "component_ids": [str(source_component["component_id"]), str(target_component["component_id"])],
        },
        "surface_edge": {"dominant_side": "delta_union"},
        "contact": {
            "contact_frame_count": 1,
            "max_surface_contact_cell_count": int(changed_pixel_count),
        },
        "result_delta": {"final_changed_pixel_count": int(changed_pixel_count)},
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
        "evidence_status": "observed_action_surface_result_delta",
        **TRUTH_FLAGS,
    }


def angle_degrees(dx: float, dy: float) -> float:
    return round(math.degrees(math.atan2(float(dy), float(dx))), 6)


def build_relationship_observation(
    *,
    case_id: str,
    action: GameAction,
    pair_index: int = 1,
    statecon_id: str,
    before_frame: np.ndarray,
    after_frame: np.ndarray,
    source_component: dict[str, Any],
    target_component: dict[str, Any],
    changed_pixel_count: int,
    change_bbox: list[int],
    object_correspondence: dict[str, Any],
    temporal_trajectory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pair_id = f"pair_{int(pair_index):02d}"
    source_center = [float(v) for v in source_component["center_xy"]]
    target_center = [float(v) for v in target_component["center_xy"]]
    dx = float(target_center[0]) - float(source_center[0])
    dy = float(target_center[1]) - float(source_center[1])
    source_bbox = [int(v) for v in source_component.get("bbox_xyxy", [0, 0, 0, 0])]
    target_bbox = [int(v) for v in target_component.get("bbox_xyxy", [0, 0, 0, 0])]
    source_area = float(source_component.get("area_pixels") or 0.0)
    target_area = float(target_component.get("area_pixels") or 0.0)
    source_raw_value = int(source_component.get("raw_pixel_value") or 0)
    target_raw_value = int(target_component.get("raw_pixel_value") or 0)
    source_width = float(bbox_width(source_bbox))
    target_width = float(bbox_width(target_bbox))
    source_height = float(bbox_height(source_bbox))
    target_height = float(bbox_height(target_bbox))
    source_aspect = bbox_aspect_ratio(source_bbox)
    target_aspect = bbox_aspect_ratio(target_bbox)
    source_diagonal = bbox_diagonal(source_bbox)
    target_diagonal = bbox_diagonal(target_bbox)
    area_ratio_before_after = round(target_area / source_area, 6) if source_area > 0.0 else 0.0
    overlap_area = bbox_overlap_area(source_bbox, target_bbox)
    overlap_ratio_of_smaller = round(float(overlap_area) / float(max(1, min(bbox_area(source_bbox), bbox_area(target_bbox)))), 6)
    center_distance_px = round(center_distance(source_center, target_center), 6)
    manhattan_distance_px = round(abs(dx) + abs(dy), 6)
    topology_iou = bbox_iou(source_bbox, target_bbox)
    center_before = [float(before_frame.shape[1] - 1) / 2.0, float(before_frame.shape[0] - 1) / 2.0]
    before_angle = angle_degrees(source_center[0] - center_before[0], source_center[1] - center_before[1])
    after_angle = angle_degrees(target_center[0] - center_before[0], target_center[1] - center_before[1])
    numeric_deltas = [
        {
            "field_path": "frame.changed_pixel_count",
            "before_value": 0.0,
            "after_value": float(changed_pixel_count),
            "delta": float(changed_pixel_count),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "spatial.centroid_dx_pixels",
            "before_value": 0.0,
            "after_value": dx,
            "delta": dx,
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "spatial.centroid_dy_pixels",
            "before_value": 0.0,
            "after_value": dy,
            "delta": dy,
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "angles.center_to_component_angle_degrees",
            "before_value": before_angle,
            "after_value": after_angle,
            "delta": round(after_angle - before_angle, 6),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "component.area_pixels",
            "before_value": source_area,
            "after_value": target_area,
            "delta": round(target_area - source_area, 6),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "component.changed_bbox_area_pixels",
            "before_value": 0.0,
            "after_value": float(bbox_area(change_bbox)),
            "delta": float(bbox_area(change_bbox)),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "shape.width_pixels",
            "before_value": source_width,
            "after_value": target_width,
            "delta": round(target_width - source_width, 6),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "shape.height_pixels",
            "before_value": source_height,
            "after_value": target_height,
            "delta": round(target_height - source_height, 6),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "shape.aspect_ratio",
            "before_value": source_aspect,
            "after_value": target_aspect,
            "delta": round(target_aspect - source_aspect, 6),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "shape.area_ratio_before_after",
            "before_value": 1.0,
            "after_value": area_ratio_before_after,
            "delta": round(area_ratio_before_after - 1.0, 6),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "shape.bbox_diagonal_pixels",
            "before_value": source_diagonal,
            "after_value": target_diagonal,
            "delta": round(target_diagonal - source_diagonal, 6),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "topology.bbox_iou_before_after",
            "before_value": 0.0,
            "after_value": topology_iou,
            "delta": topology_iou,
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "topology.bbox_overlap_area_pixels",
            "before_value": 0.0,
            "after_value": float(overlap_area),
            "delta": float(overlap_area),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "topology.bbox_overlap_ratio_of_smaller",
            "before_value": 0.0,
            "after_value": overlap_ratio_of_smaller,
            "delta": overlap_ratio_of_smaller,
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "topology.center_distance_pixels",
            "before_value": 0.0,
            "after_value": center_distance_px,
            "delta": center_distance_px,
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "topology.manhattan_distance_pixels",
            "before_value": 0.0,
            "after_value": manhattan_distance_px,
            "delta": manhattan_distance_px,
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "color.raw_value_delta",
            "before_value": float(source_raw_value),
            "after_value": float(target_raw_value),
            "delta": float(target_raw_value - source_raw_value),
            "status": "observed_numeric_delta",
        },
        {
            "field_path": "color.raw_value_changed",
            "before_value": 0.0,
            "after_value": 1.0 if source_raw_value != target_raw_value else 0.0,
            "delta": 1.0 if source_raw_value != target_raw_value else 0.0,
            "status": "observed_numeric_delta",
        },
    ]
    temporal_profile = temporal_trajectory if isinstance(temporal_trajectory, dict) else {}
    temporal_vector = temporal_profile.get("trajectory_vector") if isinstance(temporal_profile.get("trajectory_vector"), dict) else {}
    if temporal_profile:
        temporal_delta_specs = [
            ("temporal.frame_count", float(temporal_profile.get("frame_count") or 0.0)),
            ("temporal.nonzero_frame_count", float(temporal_profile.get("nonzero_frame_count") or 0.0)),
            ("temporal.path_distance_px", float(temporal_profile.get("path_distance_px") or temporal_profile.get("path_distance") or 0.0)),
            ("temporal.trajectory_dx_pixels", float(temporal_vector.get("dx") or 0.0)),
            ("temporal.trajectory_dy_pixels", float(temporal_vector.get("dy") or 0.0)),
            ("temporal.trajectory_magnitude_px", float(temporal_vector.get("euclidean_magnitude") or 0.0)),
            ("temporal.peak_changed_pixel_count", float(temporal_profile.get("peak_changed_pixel_count") or 0.0)),
            ("temporal.peak_step_changed_pixel_count", float(temporal_profile.get("peak_step_changed_pixel_count") or 0.0)),
        ]
        for field_path, value in temporal_delta_specs:
            numeric_deltas.append(
                {
                    "field_path": field_path,
                    "before_value": 0.0,
                    "after_value": value,
                    "delta": value,
                    "status": "observed_numeric_delta",
                }
            )
    changed_fields = [
        row["field_path"]
        for row in numeric_deltas
        if isinstance(row.get("delta"), (int, float)) and abs(float(row.get("delta") or 0.0)) > 0.0
    ]
    if source_component.get("component_id") != target_component.get("component_id"):
        changed_fields.append("members")
    return {
        "schema": "codexlab.deltaband_observation.v1",
        "status": "observed",
        "deltaband_id": f"deltaband_{case_id}_{action.name.lower()}_{pair_id}",
        "action_invocation_id": f"action_{case_id}_{action.name.lower()}",
        "candidate_pair_index": int(pair_index),
        "before_statecon_id": statecon_id,
        "after_statecon_id": statecon_id,
        "numeric_deltas": numeric_deltas,
        "shape_geometry": {
            "before_bbox_xyxy": source_bbox,
            "after_bbox_xyxy": target_bbox,
            "before_width_pixels": source_width,
            "after_width_pixels": target_width,
            "width_delta_pixels": round(target_width - source_width, 6),
            "before_height_pixels": source_height,
            "after_height_pixels": target_height,
            "height_delta_pixels": round(target_height - source_height, 6),
            "before_aspect_ratio": source_aspect,
            "after_aspect_ratio": target_aspect,
            "aspect_ratio_delta": round(target_aspect - source_aspect, 6),
            "before_area_pixels": source_area,
            "after_area_pixels": target_area,
            "area_delta_pixels": round(target_area - source_area, 6),
            "area_ratio_before_after": area_ratio_before_after,
            "before_bbox_diagonal_pixels": source_diagonal,
            "after_bbox_diagonal_pixels": target_diagonal,
            "bbox_diagonal_delta_pixels": round(target_diagonal - source_diagonal, 6),
            "evidence_status": "candidate_shape_geometry_delta_not_truth",
            **TRUTH_FLAGS,
        },
        "topology_relationship": {
            "bbox_iou_before_after": topology_iou,
            "bbox_overlap_area_pixels": overlap_area,
            "bbox_overlap_ratio_of_smaller": overlap_ratio_of_smaller,
            "bbox_overlap_status": "overlap" if overlap_area > 0 else "disjoint",
            "center_distance_pixels": center_distance_px,
            "manhattan_distance_pixels": manhattan_distance_px,
            "direction_8way": direction_8way(dx, dy),
            "axis_family": axis_family(dx, dy),
            "evidence_status": "candidate_topology_relationship_delta_not_truth",
            **TRUTH_FLAGS,
        },
        "color_value_relationship": {
            "before_raw_pixel_value": source_raw_value,
            "after_raw_pixel_value": target_raw_value,
            "raw_value_delta": target_raw_value - source_raw_value,
            "raw_value_changed": source_raw_value != target_raw_value,
            "raw_value_transition": f"{source_raw_value}->{target_raw_value}",
            "evidence_status": "candidate_color_or_value_delta_not_truth",
            **TRUTH_FLAGS,
        },
        "relationship_descriptors": {
            "direction_8way": direction_8way(dx, dy),
            "axis_family": axis_family(dx, dy),
            "shape_change": "changed" if any(abs(v) > 0.0 for v in [target_width - source_width, target_height - source_height, target_area - source_area, target_aspect - source_aspect]) else "stable",
            "overlap_status": "overlap" if overlap_area > 0 else "disjoint",
            "raw_value_status": "changed" if source_raw_value != target_raw_value else "stable",
            "evidence_status": "candidate_relationship_descriptor_not_truth",
            **TRUTH_FLAGS,
        },
        "member_delta": {
            "before_members": {
                "object_ids": [f"{case_id}:before:{source_component['component_id']}"],
                "component_ids": [str(source_component["component_id"])],
            },
            "after_members": {
                "object_ids": [f"{case_id}:after:{target_component['component_id']}"],
                "component_ids": [str(target_component["component_id"])],
            },
            "changed": source_component.get("component_id") != target_component.get("component_id"),
        },
        "object_correspondence": object_correspondence,
        "temporal_trajectory": temporal_profile,
        "ray_count_delta": 1 if changed_pixel_count > 0 else 0,
        "relationship_changed": bool(changed_fields),
        "changed_fields": changed_fields,
        "evidence_status": "observed_relationship_delta",
        **TRUTH_FLAGS,
    }


def delta_image(before: np.ndarray, after: np.ndarray, path: Path, *, scale: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    before_rgb = np.asarray(raw_image(before, scale=scale).convert("RGB"))
    after_rgb = np.asarray(raw_image(after, scale=scale).convert("RGB"))
    mask = np.repeat(np.repeat((before != after).astype(np.uint8), scale, axis=0), scale, axis=1)
    out = before_rgb.copy()
    out[mask > 0] = np.asarray([235, 40, 65], dtype=np.uint8)
    blend = ((0.55 * out) + (0.45 * after_rgb)).clip(0, 255).astype(np.uint8)
    Image.fromarray(blend, mode="RGB").save(path)
    return {"path": relpath(path), "sha256": sha256_file(path)}


def _scaled_box(box: list[int], *, scale: int, x_offset: int = 0, y_offset: int = 0) -> list[int]:
    return [
        x_offset + int(box[0]) * scale,
        y_offset + int(box[1]) * scale,
        x_offset + (int(box[2]) + 1) * scale - 1,
        y_offset + (int(box[3]) + 1) * scale - 1,
    ]


def _scaled_point(point: list[float], *, scale: int, x_offset: int = 0, y_offset: int = 0) -> tuple[int, int]:
    return (x_offset + int(round(float(point[0]) * scale)), y_offset + int(round(float(point[1]) * scale)))


def _draw_rect(draw: ImageDraw.ImageDraw, box: list[int], color: tuple[int, int, int], *, width: int = 3) -> None:
    for offset in range(width):
        draw.rectangle([box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset], outline=color)


def _draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
    draw.line([start, end], fill=color, width=3)
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = max(1.0, math.hypot(dx, dy))
    ux = dx / length
    uy = dy / length
    left = (int(end[0] - (8 * ux) - (5 * uy)), int(end[1] - (8 * uy) + (5 * ux)))
    right = (int(end[0] - (8 * ux) + (5 * uy)), int(end[1] - (8 * uy) - (5 * ux)))
    draw.polygon([end, left, right], fill=color)


def interaction_overlay(
    *,
    after_frame: np.ndarray,
    candidate_pairs: list[dict[str, Any]],
    relative_edges: list[dict[str, Any]],
    solve_like_rows: list[dict[str, Any]] | None = None,
    change_bbox: list[int],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    colors = [
        (240, 90, 55),
        (0, 150, 210),
        (40, 170, 90),
        (170, 95, 220),
        (230, 170, 35),
        (210, 80, 150),
    ]
    frame = raw_image(after_frame, scale=scale).convert("RGB")
    legend_w = 440
    margin = 16
    header_h = 44
    legend_row_h = 68
    canvas_h = max(frame.height + header_h + 32, 190 + (legend_row_h * min(6, len(candidate_pairs))))
    canvas = Image.new("RGB", (frame.width + legend_w + (margin * 3), canvas_h), (247, 247, 244))
    draw = ImageDraw.Draw(canvas)
    frame_x = margin
    frame_y = header_h
    canvas.paste(frame, (frame_x, frame_y))
    draw.text((margin, 14), "candidate interaction overlay (review aid, not raw frame)", fill=(20, 20, 20))
    _draw_rect(draw, _scaled_box(change_bbox, scale=scale, x_offset=frame_x, y_offset=frame_y), (235, 40, 65), width=2)
    relation_by_pair: dict[int, str] = {}
    for edge in relative_edges:
        profile = edge.get("relationship_profile") if isinstance(edge.get("relationship_profile"), dict) else {}
        correspondence = profile.get("object_correspondence") if isinstance(profile.get("object_correspondence"), dict) else {}
        pair_index = int(correspondence.get("candidate_pair_index") or 0)
        if pair_index and edge.get("relation_type_label"):
            relation_by_pair[pair_index] = str(edge.get("relation_type_label"))
    ledger_by_pair: dict[int, dict[str, Any]] = {}
    for row in solve_like_rows or []:
        if not isinstance(row, dict):
            continue
        pair_index = int(row.get("candidate_pair_index") or 0)
        if pair_index <= 0:
            continue
        existing = ledger_by_pair.get(pair_index)
        if existing is None or float(row.get("solve_like_evidence_score") or 0.0) > float(existing.get("solve_like_evidence_score") or 0.0):
            ledger_by_pair[pair_index] = row
    legend_x = frame_x + frame.width + margin
    draw.text((legend_x, 14), "ranked candidate object interactions + solve-like ledger", fill=(20, 20, 20))
    for idx, pair in enumerate(candidate_pairs[:6]):
        color = colors[idx % len(colors)]
        selected_pair = pair.get("selected_pair") if isinstance(pair.get("selected_pair"), dict) else {}
        before_box = selected_pair.get("before_bbox_xyxy") if isinstance(selected_pair.get("before_bbox_xyxy"), list) else None
        after_box = selected_pair.get("after_bbox_xyxy") if isinstance(selected_pair.get("after_bbox_xyxy"), list) else None
        before_center = selected_pair.get("before_center_xy") if isinstance(selected_pair.get("before_center_xy"), list) else None
        after_center = selected_pair.get("after_center_xy") if isinstance(selected_pair.get("after_center_xy"), list) else None
        pair_index = int(pair.get("pair_index") or (idx + 1))
        if isinstance(before_box, list):
            _draw_rect(draw, _scaled_box(before_box, scale=scale, x_offset=frame_x, y_offset=frame_y), color, width=2)
        if isinstance(after_box, list):
            _draw_rect(draw, _scaled_box(after_box, scale=scale, x_offset=frame_x, y_offset=frame_y), color, width=4)
        if isinstance(before_center, list) and isinstance(after_center, list):
            start = _scaled_point(before_center, scale=scale, x_offset=frame_x, y_offset=frame_y)
            end = _scaled_point(after_center, scale=scale, x_offset=frame_x, y_offset=frame_y)
            if abs(start[0] - end[0]) + abs(start[1] - end[1]) <= 2:
                draw.ellipse([end[0] - 6, end[1] - 6, end[0] + 6, end[1] + 6], outline=color, width=3)
            else:
                _draw_arrow(draw, start, end, color)
        legend_y = 52 + (idx * legend_row_h)
        draw.rectangle([legend_x, legend_y + 2, legend_x + 16, legend_y + 18], fill=color)
        score = selected_pair.get("correspondence_score")
        relation = relation_by_pair.get(pair_index, "relative_profile_available_in_feedback")
        ledger = ledger_by_pair.get(pair_index, {})
        hint = ledger.get("routing_recommendation") if isinstance(ledger.get("routing_recommendation"), dict) else {}
        ledger_line = (
            f"{ledger.get('interaction_class')} score={ledger.get('solve_like_evidence_score')} "
            f"unique={ledger.get('is_unique_relationship_candidate')} route={hint.get('label')}"
            if ledger
            else "solve-like ledger row in feedback"
        )
        invariant_line = f"invariant={ledger.get('mathematical_invariant_hash')}" if ledger.get("mathematical_invariant_hash") else "invariant=n/a"
        draw.text((legend_x + 24, legend_y), f"pair {pair_index} score={score}", fill=(20, 20, 20))
        draw.text((legend_x + 24, legend_y + 17), str(relation)[:62], fill=(20, 20, 20))
        draw.text((legend_x + 24, legend_y + 34), ledger_line[:70], fill=(55, 55, 55))
        draw.text((legend_x + 24, legend_y + 47), invariant_line[:70], fill=(55, 55, 55))
    draw.text((legend_x, canvas_h - 58), "Boundary: annotation from controlled frame delta.", fill=(20, 20, 20))
    draw.text((legend_x, canvas_h - 40), "Not solve truth, action choice, control truth, or prediction.", fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {
        "path": relpath(path),
        "sha256": sha256_file(path),
        "kind": "candidate_interaction_overlay_review_aid_not_raw_frame",
        "candidate_pair_count": len(candidate_pairs),
        "solve_like_candidate_count": len([row for row in solve_like_rows or [] if isinstance(row, dict)]),
        "solve_like_candidate_pair_count": len(ledger_by_pair),
        "truth_boundary": "annotation derived from controlled before/action/after frame delta; not solve truth, action choice, control truth, or prediction",
    }


def action_sequence_contact_sheet(
    *,
    action_frames: list[np.ndarray],
    temporal_trajectory: dict[str, Any],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    rendered = [raw_image(frame, scale=scale).convert("RGB") for frame in action_frames]
    if not rendered:
        raise RuntimeError("cannot build contact sheet without action frames")
    frame_w = max(image.width for image in rendered)
    frame_h = max(image.height for image in rendered)
    cols = min(5, max(1, len(rendered)))
    rows = int(math.ceil(float(len(rendered)) / float(cols)))
    margin = 16
    gap = 14
    label_h = 58
    header_h = 92
    footer_h = 58
    cell_w = frame_w + 18
    cell_h = frame_h + label_h
    canvas_w = (margin * 2) + (cols * cell_w) + ((cols - 1) * gap)
    canvas_h = header_h + (rows * cell_h) + ((rows - 1) * gap) + footer_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (247, 247, 244))
    draw = ImageDraw.Draw(canvas)
    vector = temporal_trajectory.get("trajectory_vector") if isinstance(temporal_trajectory.get("trajectory_vector"), dict) else {}
    draw.text((margin, 14), "action frame sequence contact sheet (review aid, raw frames saved separately)", fill=(20, 20, 20))
    draw.text(
        (margin, 38),
        (
            f"frames={len(rendered)} multi_frame={temporal_trajectory.get('multi_frame_available')} "
            f"direction={vector.get('direction_8way')} axis={vector.get('axis_family')} "
            f"path_px={temporal_trajectory.get('path_distance_px')}"
        ),
        fill=(20, 20, 20),
    )
    draw.text(
        (margin, 62),
        f"peak_frame={temporal_trajectory.get('peak_changed_frame_index')} peak_changed={temporal_trajectory.get('peak_changed_pixel_count')}",
        fill=(20, 20, 20),
    )
    frame_deltas = [
        row
        for row in temporal_trajectory.get("frame_deltas", []) or []
        if isinstance(row, dict)
    ]
    for index, image in enumerate(rendered):
        row = index // cols
        col = index % cols
        x = margin + (col * (cell_w + gap))
        y = header_h + (row * (cell_h + gap))
        canvas.paste(image, (x, y))
        delta = frame_deltas[index] if index < len(frame_deltas) else {}
        bbox = delta.get("changed_bbox_xyxy_vs_before") if isinstance(delta.get("changed_bbox_xyxy_vs_before"), list) else None
        if isinstance(bbox, list) and len(bbox) == 4:
            _draw_rect(draw, _scaled_box(bbox, scale=scale, x_offset=x, y_offset=y), (235, 40, 65), width=2)
        label_y = y + frame_h + 6
        draw.text((x, label_y), f"frame {index:03d}", fill=(20, 20, 20))
        draw.text(
            (x, label_y + 18),
            f"vs_before={delta.get('changed_pixel_count_vs_before')} prev={delta.get('changed_pixel_count_from_previous')}",
            fill=(55, 55, 55),
        )
        draw.text((x, label_y + 36), f"hash={sha256_frame(action_frames[index])[:12]}", fill=(55, 55, 55))
    draw.text((margin, canvas_h - 42), "Red boxes mark changed bbox vs the pre-action frame.", fill=(20, 20, 20))
    draw.text((margin, canvas_h - 22), "Boundary: annotation from controlled returned action frames; not solve truth or action authority.", fill=(20, 20, 20))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {
        "path": relpath(path),
        "sha256": sha256_file(path),
        "kind": "action_sequence_contact_sheet_review_aid_not_raw_frame",
        "frame_count": len(action_frames),
        "multi_frame_available": temporal_trajectory.get("multi_frame_available") is True,
        "truth_boundary": "annotation derived from controlled returned action frames; not solve truth, action choice, control truth, or prediction",
    }


def case_sheet(
    *,
    before_png: Path,
    after_png: Path,
    delta_png: Path,
    case: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    before = Image.open(before_png).convert("RGB")
    after = Image.open(after_png).convert("RGB")
    delta = Image.open(delta_png).convert("RGB")
    panel_w = max(before.width, after.width, delta.width)
    text_h = 220
    sheet = Image.new("RGB", ((panel_w * 3) + 64, max(before.height, after.height, delta.height) + text_h), (247, 247, 244))
    draw = ImageDraw.Draw(sheet)
    sheet.paste(before, (16, 44))
    sheet.paste(after, (32 + panel_w, 44))
    sheet.paste(delta, (48 + panel_w * 2, 44))
    draw.text((16, 16), "before reset frame", fill=(20, 20, 20))
    draw.text((32 + panel_w, 16), "after action frame", fill=(20, 20, 20))
    draw.text((48 + panel_w * 2, 16), "delta overlay", fill=(20, 20, 20))
    y = max(before.height, after.height, delta.height) + 64
    feedback = case.get("portband_surface_feedback_summary") or {}
    correspondence = case.get("object_correspondence") if isinstance(case.get("object_correspondence"), dict) else {}
    trajectory = case.get("action_temporal_trajectory") if isinstance(case.get("action_temporal_trajectory"), dict) else {}
    trajectory_vector = trajectory.get("trajectory_vector") if isinstance(trajectory.get("trajectory_vector"), dict) else {}
    lines = [
        f"{case['game_id']} level={case['level_index']} action={case['action']['name']} changed_pixels={case['changed_pixel_count']}",
        f"edge_count={feedback.get('action_interaction_edge_count')} kinds={feedback.get('action_interaction_kinds')} relative_tags={feedback.get('relative_state_change_tag_count')}",
        f"candidate_pairs={feedback.get('candidate_interaction_pair_count')} matrix_rows={feedback.get('action_interaction_matrix_row_count')} surface_events={feedback.get('action_surface_result_event_count')}",
        f"object_correspondence_score={correspondence.get('confidence_score')} model={correspondence.get('confidence_model')}",
        f"temporal_frames={trajectory.get('frame_count')} multi_frame={trajectory.get('multi_frame_available')} direction={trajectory_vector.get('direction_8way')} path_px={trajectory.get('path_distance_px')}",
        f"route_flow_locks={feedback.get('route_flow_lock_count')} solve_like_edges={feedback.get('solve_like_interaction_edge_count')}",
        "Boundary: controlled before/action/after observation; not solve truth, prediction, action choice, or control authority.",
    ]
    for line in lines:
        draw.text((16, y), line[:220], fill=(20, 20, 20))
        y += 22
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return {"path": relpath(path), "sha256": sha256_file(path)}


def run_action_case(
    *,
    game_info: dict[str, Any],
    action: GameAction,
    level_index: int,
    out_dir: Path,
    scale: int,
    max_correspondence_pairs: int,
) -> dict[str, Any] | None:
    cls = load_game_class(game_info["source_path"])
    game = cls()
    game.set_level(level_index)
    reset_obs = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    before_frame, before_frame_meta = normalize_raw_frame(reset_obs.frame)
    action_obs = game.perform_action(ActionInput(id=action, data={}), raw=True)
    action_frames, after_frame_meta = normalize_raw_frame_sequence(action_obs.frame)
    after_frame = action_frames[-1]
    temporal_trajectory = action_temporal_trajectory_profile(before_frame, action_frames)
    if before_frame.shape != after_frame.shape:
        return None
    diff_mask = before_frame != after_frame
    changed_pixel_count = int(np.count_nonzero(diff_mask))
    if changed_pixel_count <= 0:
        return None
    change_bbox = bbox_from_mask(diff_mask)
    if change_bbox is None:
        return None
    change_center = center_of_bbox(change_bbox)
    before_components = connected_components(before_frame)
    after_components = connected_components(after_frame)
    object_correspondence = object_correspondence_profile(
        before_components=before_components,
        after_components=after_components,
        change_bbox=change_bbox,
        change_center=change_center,
        frame_shape=before_frame.shape,
    )
    if not isinstance(object_correspondence, dict):
        return None
    candidate_pairs = select_candidate_interaction_pairs(
        object_correspondence=object_correspondence,
        before_components=before_components,
        after_components=after_components,
        max_pairs=max(1, int(max_correspondence_pairs)),
    )
    if not candidate_pairs:
        return None
    case_id = safe_part(f"{game_info['game_short']}_{game_info['game_hash']}_level_{level_index:02d}_{action.name.lower()}")
    case_dir = out_dir / "cases" / case_id
    first_pair = candidate_pairs[0]
    source_component = first_pair["source_component"]
    target_component = first_pair["target_component"]
    surface_bbox = bbox_union(source_component["bbox_xyxy"], target_component["bbox_xyxy"])
    statecon = statecon_from_component_pairs(
        case_id=case_id,
        component_pairs=candidate_pairs,
    )
    surface_events: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    for pair in candidate_pairs:
        pair_index = int(pair["pair_index"])
        pair_source = pair["source_component"]
        pair_target = pair["target_component"]
        pair_surface_bbox = bbox_union(pair_source["bbox_xyxy"], pair_target["bbox_xyxy"])
        pair_correspondence = pair["object_correspondence"]
        surface_events.append(
            build_surface_event(
                case_id=case_id,
                action=action,
                pair_index=pair_index,
                source_component=pair_source,
                target_component=pair_target,
                surface_bbox=pair_surface_bbox,
                changed_pixel_count=changed_pixel_count,
                confidence=float(pair_correspondence.get("confidence_score") or 0.72),
            )
        )
        relationships.append(
            build_relationship_observation(
                case_id=case_id,
                action=action,
                pair_index=pair_index,
                statecon_id=str(statecon["statecon_id"]),
                before_frame=before_frame,
                after_frame=after_frame,
                source_component=pair_source,
                target_component=pair_target,
                changed_pixel_count=changed_pixel_count,
                change_bbox=change_bbox,
                object_correspondence=pair_correspondence,
                temporal_trajectory=temporal_trajectory,
            )
        )
    updated_statecons = apply_portband_surface_feedback_to_statecons(
        statecons=[statecon],
        action_surface_result_events=surface_events,
        action_state_relationship_observations=relationships,
        action_primitive=action_label(action),
    )
    portband = ((updated_statecons[0] or {}).get("portband_hypotheses") or {}) if updated_statecons else {}
    feedback = portband.get("surface_feedback") if isinstance(portband.get("surface_feedback"), dict) else {}
    graph = feedback.get("action_interaction_graph") if isinstance(feedback.get("action_interaction_graph"), dict) else {}
    edges = [edge for edge in graph.get("edges", []) or [] if isinstance(edge, dict)]
    relative_edges = [edge for edge in edges if edge.get("interaction_kind") == "relative_state_change_between_relationship_members"]
    relative_profile_edges = [
        edge
        for edge in relative_edges
        if isinstance(edge.get("relationship_profile"), dict) and edge.get("relationship_profile", {}).get("relation_type_label")
    ]
    matrix = feedback.get("action_interaction_matrix") if isinstance(feedback.get("action_interaction_matrix"), dict) else {}
    matrix_rows = [row for row in matrix.get("rows", []) or [] if isinstance(row, dict)]
    solve_like_ledger = feedback.get("solve_like_candidate_ledger") if isinstance(feedback.get("solve_like_candidate_ledger"), dict) else {}
    summary = {
        "action_state_tag_count": int(feedback.get("action_state_tag_count") or 0),
        "relative_state_change_tag_count": int(feedback.get("relative_state_change_tag_count") or 0),
        "action_interaction_edge_count": int(feedback.get("action_interaction_edge_count") or 0),
        "action_interaction_kinds": feedback.get("action_interaction_kinds", []),
        "action_interaction_matrix_schema": matrix.get("schema"),
        "action_interaction_matrix_policy": matrix.get("matrix_policy"),
        "action_interaction_matrix_row_count": int(matrix.get("row_count") or 0),
        "action_interaction_support_object_count": int(matrix.get("support_object_count") or 0),
        "action_interaction_support_object_ids": matrix.get("support_object_ids", []),
        "action_interaction_best_by_support_object": matrix.get("best_by_support_object", []),
        "solve_like_candidate_ledger_schema": solve_like_ledger.get("schema"),
        "solve_like_candidate_ledger_row_count": int(solve_like_ledger.get("row_count") or 0),
        "solve_like_candidate_ledger_unique_count": int(solve_like_ledger.get("unique_candidate_count") or 0),
        "top_solve_like_candidate_ledger_rows": solve_like_ledger.get("top_rows", []),
        "relative_profile_edge_count": len(relative_profile_edges),
        "relative_profile_relation_type_labels": sorted(
            {
                str((edge.get("relationship_profile") or {}).get("relation_type_label"))
                for edge in relative_profile_edges
                if isinstance(edge.get("relationship_profile"), dict) and (edge.get("relationship_profile") or {}).get("relation_type_label")
            }
        ),
        "candidate_interaction_pair_count": len(candidate_pairs),
        "action_surface_result_event_count": len(surface_events),
        "relationship_observation_count": len(relationships),
        "novel_relationship_candidate_count": int(feedback.get("novel_relationship_candidate_count") or 0),
        "solve_like_evidence_flag_count": int(feedback.get("solve_like_evidence_flag_count") or 0),
        "solve_like_interaction_edge_count": int(feedback.get("solve_like_interaction_edge_count") or 0),
        "route_flow_lock_count": int(feedback.get("route_flow_lock_count") or 0),
        "temporal_trajectory_status": temporal_trajectory.get("trajectory_status"),
        "temporal_trajectory_frame_count": temporal_trajectory.get("frame_count"),
        "temporal_trajectory_multi_frame_available": temporal_trajectory.get("multi_frame_available"),
        "temporal_trajectory_nonzero_frame_count": temporal_trajectory.get("nonzero_frame_count"),
        "temporal_trajectory_path_distance_px": temporal_trajectory.get("path_distance_px"),
        "temporal_trajectory_path_distance_bucket": temporal_trajectory.get("path_distance_bucket"),
        "temporal_trajectory_direction_8way": (temporal_trajectory.get("trajectory_vector") or {}).get("direction_8way")
        if isinstance(temporal_trajectory.get("trajectory_vector"), dict)
        else None,
        "temporal_trajectory_axis_family": (temporal_trajectory.get("trajectory_vector") or {}).get("axis_family")
        if isinstance(temporal_trajectory.get("trajectory_vector"), dict)
        else None,
        "temporal_trajectory_peak_changed_frame_index": temporal_trajectory.get("peak_changed_frame_index"),
        "temporal_trajectory_peak_changed_pixel_count": temporal_trajectory.get("peak_changed_pixel_count"),
        "top_solve_like_evidence_score": feedback.get("top_solve_like_evidence_score"),
        "top_action_interaction_edges": edges[:6],
        "top_relative_action_interaction_edges": relative_edges[:6],
        "top_action_interaction_matrix_rows": matrix_rows[:6],
    }
    before_meta = save_frame_artifacts(before_frame, png_path=case_dir / "before.png", npy_path=case_dir / "before.npy", scale=scale)
    after_meta = save_frame_artifacts(after_frame, png_path=case_dir / "after.png", npy_path=case_dir / "after.npy", scale=scale)
    action_frame_sequence_meta = save_frame_sequence_artifacts(action_frames, directory=case_dir / "action_frames", scale=scale)
    action_sequence_contact_sheet_meta = action_sequence_contact_sheet(
        action_frames=action_frames,
        temporal_trajectory=temporal_trajectory,
        path=case_dir / "action_sequence_contact_sheet.png",
        scale=scale,
    )
    delta_meta = delta_image(before_frame, after_frame, case_dir / "delta.png", scale=scale)
    interaction_overlay_meta = interaction_overlay(
        after_frame=after_frame,
        candidate_pairs=candidate_pairs,
        relative_edges=relative_edges,
        solve_like_rows=solve_like_ledger.get("rows", []),
        change_bbox=change_bbox,
        path=case_dir / "interaction_overlay.png",
        scale=scale,
    )
    write_json(case_dir / "portband_surface_feedback.json", feedback)
    case = {
        "schema": CASE_SCHEMA,
        "status": "controlled",
        "case_id": case_id,
        "game_id": game_info["game_id"],
        "title": game_info["title"],
        "game_short": game_info["game_short"],
        "game_hash": game_info["game_hash"],
        "source_path": relpath(game_info["source_path"]),
        "source_sha256": game_info["source_sha256"],
        "level_index": int(level_index),
        "action": {"id": int(action.value), "name": action.name, "primitive": action_label(action), "data": {}},
        "before_frame": before_meta,
        "after_frame": after_meta,
        "delta_overlay": delta_meta,
        "interaction_overlay": interaction_overlay_meta,
        "before_frame_return": before_frame_meta,
        "after_frame_return": after_frame_meta,
        "action_frame_sequence": action_frame_sequence_meta,
        "action_sequence_contact_sheet": action_sequence_contact_sheet_meta,
        "action_temporal_trajectory": temporal_trajectory,
        "changed_pixel_count": changed_pixel_count,
        "change_bbox_xyxy": change_bbox,
        "surface_bbox_xyxy": surface_bbox,
        "source_component": source_component,
        "target_component": target_component,
        "object_correspondence": object_correspondence,
        "candidate_interaction_pair_count": len(candidate_pairs),
        "candidate_interaction_pairs": [
            {
                "pair_index": int(pair["pair_index"]),
                "selected_pair": pair["selected_pair"],
                "object_correspondence": pair["object_correspondence"],
                "source_component_id": str(pair["source_component"].get("component_id")),
                "target_component_id": str(pair["target_component"].get("component_id")),
            }
            for pair in candidate_pairs
        ],
        "object_correspondence_profiles": [pair["object_correspondence"] for pair in candidate_pairs],
        "action_surface_result_event": surface_events[0],
        "action_surface_result_events": surface_events,
        "relationship_observation": relationships[0],
        "relationship_observations": relationships,
        "portband_surface_feedback_path": relpath(case_dir / "portband_surface_feedback.json"),
        "portband_surface_feedback_summary": summary,
        "evidence_category": "controlled_run",
        "truth_boundary": "observed before/action/after frame delta only; not solve truth, prediction, action choice, or control authority",
        **TRUTH_FLAGS,
    }
    sheet_meta = case_sheet(
        before_png=case_dir / "before.png",
        after_png=case_dir / "after.png",
        delta_png=case_dir / "delta.png",
        case=case,
        path=case_dir / "case_sheet.png",
    )
    case["case_sheet"] = sheet_meta
    write_json(case_dir / "case.json", case)
    return case


def coverage_ratio(numerator: int, denominator: int) -> float:
    return round(float(numerator) / float(max(1, denominator)), 6)


def skip_reason_code(skip_reason: str) -> str:
    text = str(skip_reason or "unknown_skip")
    if ":" in text:
        return text.split(":", 1)[0].strip() or "unknown_skip"
    return text


def case_feedback_summary(case: dict[str, Any]) -> dict[str, Any]:
    return case.get("portband_surface_feedback_summary") if isinstance(case.get("portband_surface_feedback_summary"), dict) else {}


def case_has_relative_profile(case: dict[str, Any]) -> bool:
    summary = case_feedback_summary(case)
    if int(summary.get("relative_profile_edge_count") or 0) > 0:
        return True
    for edge in summary.get("top_action_interaction_edges", []) or []:
        if not isinstance(edge, dict) or edge.get("interaction_kind") != "relative_state_change_between_relationship_members":
            continue
        profile = edge.get("relationship_profile") if isinstance(edge.get("relationship_profile"), dict) else {}
        if profile.get("schema") == "codexlab.contra_relative_math_profile.v1" and profile.get("relation_type_label"):
            return True
    return False


def case_has_object_correspondence(case: dict[str, Any]) -> bool:
    correspondence = case.get("object_correspondence") if isinstance(case.get("object_correspondence"), dict) else {}
    return (
        correspondence.get("schema") == CONTRA_OBJECT_CORRESPONDENCE_SCHEMA
        and correspondence.get("status") == "ranked_candidate_pair"
        and isinstance(correspondence.get("confidence_score"), (int, float))
        and float(correspondence.get("confidence_score") or 0.0) > 0.0
    )


def build_coverage_summary(cases: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    attempted_game_ids = sorted(
        {
            str(row.get("game_id"))
            for row in list(cases) + list(skipped)
            if isinstance(row, dict) and row.get("game_id")
        }
    )
    controlled_game_ids = sorted({str(case.get("game_id")) for case in cases if isinstance(case, dict) and case.get("game_id")})
    attempted_action_names: set[str] = set()
    controlled_action_names: set[str] = set()
    for case in cases:
        action = case.get("action") if isinstance(case.get("action"), dict) else {}
        if action.get("name"):
            name = str(action.get("name"))
            attempted_action_names.add(name)
            controlled_action_names.add(name)
    for skip in skipped:
        if isinstance(skip, dict) and skip.get("action_name"):
            attempted_action_names.add(str(skip.get("action_name")))

    kind_counts: Counter[str] = Counter()
    relation_labels: set[str] = set()
    correspondence_scores: list[float] = []
    edge_coverage_case_count = 0
    surface_edge_case_count = 0
    relative_edge_case_count = 0
    matrix_case_count = 0
    multi_candidate_case_count = 0
    total_candidate_pair_count = 0
    novel_relationship_case_count = 0
    total_novel_relationship_count = 0
    action_surface_event_case_count = 0
    relationship_observation_case_count = 0
    interaction_overlay_case_count = 0
    temporal_trajectory_case_count = 0
    action_frame_sequence_case_count = 0
    action_sequence_contact_sheet_case_count = 0
    multi_frame_action_case_count = 0
    temporal_direction_counts: Counter[str] = Counter()
    route_flow_lock_count = 0
    for case in cases:
        summary = case_feedback_summary(case)
        candidate_pair_count = int(summary.get("candidate_interaction_pair_count") or case.get("candidate_interaction_pair_count") or 0)
        edge_count = int(summary.get("action_interaction_edge_count") or 0)
        matrix_row_count = int(summary.get("action_interaction_matrix_row_count") or 0)
        kinds = [str(kind) for kind in summary.get("action_interaction_kinds", []) or []]
        kind_counts.update(kinds)
        route_flow_lock_count += int(summary.get("route_flow_lock_count") or 0)
        total_candidate_pair_count += candidate_pair_count
        if candidate_pair_count > 1:
            multi_candidate_case_count += 1
        novel_relationship_count = int(summary.get("novel_relationship_candidate_count") or 0)
        total_novel_relationship_count += novel_relationship_count
        if novel_relationship_count > 0:
            novel_relationship_case_count += 1
        if int(summary.get("action_surface_result_event_count") or 0) >= candidate_pair_count > 0:
            action_surface_event_case_count += 1
        if int(summary.get("relationship_observation_count") or 0) >= candidate_pair_count > 0:
            relationship_observation_case_count += 1
        trajectory = case.get("action_temporal_trajectory") if isinstance(case.get("action_temporal_trajectory"), dict) else {}
        trajectory_vector = trajectory.get("trajectory_vector") if isinstance(trajectory.get("trajectory_vector"), dict) else {}
        if (
            trajectory.get("schema") == TEMPORAL_TRAJECTORY_SCHEMA
            and trajectory.get("status") == "profiled"
            and int(trajectory.get("frame_count") or 0) > 0
        ):
            temporal_trajectory_case_count += 1
            if trajectory_vector.get("direction_8way"):
                temporal_direction_counts.update([str(trajectory_vector.get("direction_8way"))])
        action_frame_sequence = [row for row in case.get("action_frame_sequence", []) or [] if isinstance(row, dict)]
        if action_frame_sequence and len(action_frame_sequence) == int(trajectory.get("frame_count") or len(action_frame_sequence)):
            action_frame_sequence_case_count += 1
        sequence_sheet = case.get("action_sequence_contact_sheet") if isinstance(case.get("action_sequence_contact_sheet"), dict) else {}
        if (
            sequence_sheet.get("path")
            and sequence_sheet.get("kind") == "action_sequence_contact_sheet_review_aid_not_raw_frame"
            and int(sequence_sheet.get("frame_count") or 0) == len(action_frame_sequence)
        ):
            action_sequence_contact_sheet_case_count += 1
        if trajectory.get("multi_frame_available") is True:
            multi_frame_action_case_count += 1
        overlay = case.get("interaction_overlay") if isinstance(case.get("interaction_overlay"), dict) else {}
        if overlay.get("path") and int(overlay.get("candidate_pair_count") or 0) >= candidate_pair_count > 0:
            interaction_overlay_case_count += 1
        if edge_count >= max(2, 2 * candidate_pair_count):
            edge_coverage_case_count += 1
        if "action_surface_contact_candidate_edge" in kinds:
            surface_edge_case_count += 1
        if "relative_state_change_between_relationship_members" in kinds:
            relative_edge_case_count += 1
        if summary.get("action_interaction_matrix_schema") == CONTRA_ACTION_INTERACTION_MATRIX_SCHEMA and matrix_row_count >= edge_count >= max(2, 2 * candidate_pair_count):
            matrix_case_count += 1
        for edge in summary.get("top_action_interaction_edges", []) or []:
            if isinstance(edge, dict) and edge.get("relation_type_label"):
                relation_labels.add(str(edge.get("relation_type_label")))
        correspondence = case.get("object_correspondence") if isinstance(case.get("object_correspondence"), dict) else {}
        if isinstance(correspondence.get("confidence_score"), (int, float)):
            correspondence_scores.append(float(correspondence.get("confidence_score") or 0.0))

    attempted_action_count = len(cases) + len(skipped)
    relative_profile_case_count = sum(1 for case in cases if case_has_relative_profile(case))
    object_correspondence_case_count = sum(1 for case in cases if case_has_object_correspondence(case))
    score_summary = {
        "count": len(correspondence_scores),
        "min": round(min(correspondence_scores), 6) if correspondence_scores else 0.0,
        "max": round(max(correspondence_scores), 6) if correspondence_scores else 0.0,
        "mean": round(sum(correspondence_scores) / float(len(correspondence_scores)), 6) if correspondence_scores else 0.0,
    }
    return {
        "schema": "codexlab.cross_game_action_interaction_coverage.v1",
        "status": "covered" if cases else "no_controlled_cases",
        "attempted_action_count": attempted_action_count,
        "controlled_case_count": len(cases),
        "skipped_count": len(skipped),
        "attempted_game_count": len(attempted_game_ids),
        "controlled_game_count": len(controlled_game_ids),
        "attempted_action_name_count": len(attempted_action_names),
        "controlled_action_name_count": len(controlled_action_names),
        "controlled_game_ids": controlled_game_ids,
        "controlled_action_names": sorted(controlled_action_names),
        "edge_coverage_case_count": edge_coverage_case_count,
        "surface_edge_case_count": surface_edge_case_count,
        "relative_edge_case_count": relative_edge_case_count,
        "relative_profile_case_count": relative_profile_case_count,
        "candidate_interaction_pair_count_total": total_candidate_pair_count,
        "candidate_interaction_pair_count_mean": round(float(total_candidate_pair_count) / float(max(1, len(cases))), 6),
        "multi_candidate_interaction_case_count": multi_candidate_case_count,
        "novel_relationship_case_count": novel_relationship_case_count,
        "novel_relationship_candidate_count_total": total_novel_relationship_count,
        "action_surface_event_case_count": action_surface_event_case_count,
        "relationship_observation_case_count": relationship_observation_case_count,
        "temporal_trajectory_case_count": temporal_trajectory_case_count,
        "action_frame_sequence_case_count": action_frame_sequence_case_count,
        "action_sequence_contact_sheet_case_count": action_sequence_contact_sheet_case_count,
        "multi_frame_action_case_count": multi_frame_action_case_count,
        "interaction_overlay_case_count": interaction_overlay_case_count,
        "object_correspondence_case_count": object_correspondence_case_count,
        "action_interaction_matrix_case_count": matrix_case_count,
        "route_flow_lock_count": route_flow_lock_count,
        "controlled_case_ratio": coverage_ratio(len(cases), attempted_action_count),
        "edge_coverage_ratio": coverage_ratio(edge_coverage_case_count, len(cases)),
        "surface_edge_ratio": coverage_ratio(surface_edge_case_count, len(cases)),
        "relative_edge_ratio": coverage_ratio(relative_edge_case_count, len(cases)),
        "relative_profile_ratio": coverage_ratio(relative_profile_case_count, len(cases)),
        "multi_candidate_interaction_ratio": coverage_ratio(multi_candidate_case_count, len(cases)),
        "action_surface_event_ratio": coverage_ratio(action_surface_event_case_count, len(cases)),
        "relationship_observation_ratio": coverage_ratio(relationship_observation_case_count, len(cases)),
        "temporal_trajectory_ratio": coverage_ratio(temporal_trajectory_case_count, len(cases)),
        "action_frame_sequence_ratio": coverage_ratio(action_frame_sequence_case_count, len(cases)),
        "action_sequence_contact_sheet_ratio": coverage_ratio(action_sequence_contact_sheet_case_count, len(cases)),
        "multi_frame_action_ratio": coverage_ratio(multi_frame_action_case_count, len(cases)),
        "interaction_overlay_ratio": coverage_ratio(interaction_overlay_case_count, len(cases)),
        "object_correspondence_ratio": coverage_ratio(object_correspondence_case_count, len(cases)),
        "action_interaction_matrix_ratio": coverage_ratio(matrix_case_count, len(cases)),
        "action_interaction_kind_counts": dict(sorted(kind_counts.items())),
        "temporal_trajectory_direction_counts": dict(sorted(temporal_direction_counts.items())),
        "relation_type_labels": sorted(relation_labels),
        "object_correspondence_confidence": score_summary,
        "skip_reason_counts": dict(sorted(Counter(skip_reason_code(str(row.get("skip_reason") or "")) for row in skipped).items())),
        "evidence_status": "controlled_action_probe_aggregate_not_truth",
        **TRUTH_FLAGS,
    }


def score_band(score: float) -> str:
    value = max(0.0, min(1.0, float(score)))
    if value >= 0.80:
        return "high"
    if value >= 0.55:
        return "medium"
    if value > 0.0:
        return "low"
    return "none"


def build_relationship_atlas(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    occurrence_count = 0
    for case in cases:
        summary = case_feedback_summary(case)
        matrix_rows = {
            str(row.get("interaction_edge_id")): row
            for row in summary.get("top_action_interaction_matrix_rows", []) or []
            if isinstance(row, dict) and row.get("interaction_edge_id")
        }
        edges_by_id: dict[str, dict[str, Any]] = {}
        for source in ("top_action_interaction_edges", "top_relative_action_interaction_edges"):
            for edge in summary.get(source, []) or []:
                if not isinstance(edge, dict) or edge.get("interaction_kind") != "relative_state_change_between_relationship_members":
                    continue
                edge_id = str(edge.get("interaction_edge_id") or "")
                if edge_id and edge_id not in edges_by_id:
                    edges_by_id[edge_id] = edge
        for edge_id, edge in edges_by_id.items():
            profile = edge.get("relationship_profile") if isinstance(edge.get("relationship_profile"), dict) else {}
            signature = str(edge.get("relationship_signature") or profile.get("relationship_signature") or "")
            relation_label = str(edge.get("relation_type_label") or profile.get("relation_type_label") or "")
            if not signature or not relation_label:
                continue
            occurrence_count += 1
            row = grouped.setdefault(
                signature,
                {
                    "relationship_signature": signature,
                    "relation_type_labels": Counter(),
                    "case_ids": set(),
                    "game_ids": set(),
                    "action_primitives": set(),
                    "candidate_pair_indices": set(),
                    "dominant_relationship_families": set(),
                    "signature_terms": set(),
                    "mathematical_invariant_hashes": Counter(),
                    "mathematical_invariant_terms": set(),
                    "vector_directions": Counter(),
                    "temporal_directions": Counter(),
                    "routing_labels": Counter(),
                    "interaction_classes": Counter(),
                    "interaction_class_families": Counter(),
                    "confidence_scores": [],
                    "solve_like_scores": [],
                    "solve_like_flag_count": 0,
                    "occurrences": [],
                    "example_case_refs": [],
                },
            )
            action = case.get("action") if isinstance(case.get("action"), dict) else {}
            matrix_row = matrix_rows.get(edge_id, {})
            vector = profile.get("vector_delta") if isinstance(profile.get("vector_delta"), dict) else {}
            temporal = profile.get("temporal_trajectory") if isinstance(profile.get("temporal_trajectory"), dict) else {}
            temporal_vector = temporal.get("trajectory_vector") if isinstance(temporal.get("trajectory_vector"), dict) else {}
            correspondence = profile.get("object_correspondence") if isinstance(profile.get("object_correspondence"), dict) else {}
            route_hint = matrix_row.get("routing_recommendation") if isinstance(matrix_row.get("routing_recommendation"), dict) else {}
            interaction_class = str(matrix_row.get("interaction_class") or route_hint.get("interaction_class") or "candidate_interaction")
            interaction_class_family = str(matrix_row.get("interaction_class_family") or route_hint.get("interaction_class_family") or "generic_relative_change")
            invariant_hash = str(matrix_row.get("mathematical_invariant_hash") or profile.get("mathematical_invariant_hash") or "")
            row["relation_type_labels"].update([relation_label])
            row["case_ids"].add(str(case.get("case_id")))
            row["game_ids"].add(str(case.get("game_id")))
            row["action_primitives"].add(str(edge.get("action_primitive") or action.get("primitive") or action.get("name") or "unknown_action"))
            if correspondence.get("candidate_pair_index") is not None:
                row["candidate_pair_indices"].add(int(correspondence.get("candidate_pair_index") or 0))
            for family in profile.get("dominant_relationship_families", []) or []:
                row["dominant_relationship_families"].add(str(family))
            for term in profile.get("signature_terms", []) or []:
                row["signature_terms"].add(str(term))
            if invariant_hash:
                row["mathematical_invariant_hashes"].update([invariant_hash])
            for term in profile.get("mathematical_invariant_terms", []) or []:
                row["mathematical_invariant_terms"].add(str(term))
            if vector.get("direction_8way"):
                row["vector_directions"].update([str(vector.get("direction_8way"))])
            if temporal_vector.get("direction_8way"):
                row["temporal_directions"].update([str(temporal_vector.get("direction_8way"))])
            if route_hint.get("label"):
                row["routing_labels"].update([str(route_hint.get("label"))])
            row["interaction_classes"].update([interaction_class])
            row["interaction_class_families"].update([interaction_class_family])
            confidence = edge.get("confidence_score", edge.get("solve_like_evidence_score"))
            if isinstance(confidence, (int, float)):
                row["confidence_scores"].append(float(confidence))
            solve_score = edge.get("solve_like_evidence_score")
            if isinstance(solve_score, (int, float)):
                row["solve_like_scores"].append(float(solve_score))
            if edge.get("solve_like_evidence_flag"):
                row["solve_like_flag_count"] += 1
            occurrence = {
                "case_id": case.get("case_id"),
                "game_id": case.get("game_id"),
                "level_index": case.get("level_index"),
                "action": action,
                "interaction_edge_id": edge_id,
                "interaction_rank": edge.get("interaction_rank"),
                "confidence_score": confidence,
                "solve_like_evidence_score": solve_score,
                "candidate_pair_index": correspondence.get("candidate_pair_index"),
                "vector_direction_8way": vector.get("direction_8way"),
                "temporal_direction_8way": temporal_vector.get("direction_8way"),
                "temporal_path_distance_px": temporal.get("path_distance_px"),
                "routing_label": route_hint.get("label"),
                "interaction_class": interaction_class,
                "interaction_class_family": interaction_class_family,
                "mathematical_invariant_hash": invariant_hash or None,
                "mathematical_invariant_signature": matrix_row.get("mathematical_invariant_signature") or profile.get("mathematical_invariant_signature"),
                "mathematical_invariant_terms": matrix_row.get("mathematical_invariant_terms") or profile.get("mathematical_invariant_terms", []),
                "route_evidence_terms": route_hint.get("route_evidence_terms", []),
                "interaction_overlay_path": (case.get("interaction_overlay") or {}).get("path")
                if isinstance(case.get("interaction_overlay"), dict)
                else None,
                "action_sequence_contact_sheet_path": (case.get("action_sequence_contact_sheet") or {}).get("path")
                if isinstance(case.get("action_sequence_contact_sheet"), dict)
                else None,
                "portband_surface_feedback_path": case.get("portband_surface_feedback_path"),
            }
            row["occurrences"].append(occurrence)
            if len(row["example_case_refs"]) < 4:
                row["example_case_refs"].append(occurrence)

    rows: list[dict[str, Any]] = []
    for signature, aggregate in grouped.items():
        confidence_scores = [float(value) for value in aggregate["confidence_scores"]]
        solve_scores = [float(value) for value in aggregate["solve_like_scores"]]
        case_ids = sorted(aggregate["case_ids"])
        game_ids = sorted(aggregate["game_ids"])
        action_primitives = sorted(aggregate["action_primitives"])
        occurrence_total = len(aggregate["occurrences"])
        max_confidence = round(max(confidence_scores), 6) if confidence_scores else 0.0
        mean_confidence = round(sum(confidence_scores) / float(len(confidence_scores)), 6) if confidence_scores else 0.0
        rows.append(
            {
                "schema": RELATIONSHIP_ATLAS_ROW_SCHEMA,
                "status": "ranked_relationship_signature",
                "atlas_row_id": f"relationship_atlas_{stable_digest(signature)}",
                "relationship_signature": signature,
                "relation_type_label": aggregate["relation_type_labels"].most_common(1)[0][0]
                if aggregate["relation_type_labels"]
                else "",
                "relation_type_label_counts": dict(sorted(aggregate["relation_type_labels"].items())),
                "occurrence_count": occurrence_total,
                "case_count": len(case_ids),
                "game_count": len(game_ids),
                "action_count": len(action_primitives),
                "game_ids": game_ids,
                "case_ids": case_ids[:12],
                "action_primitives": action_primitives,
                "candidate_pair_indices": sorted(aggregate["candidate_pair_indices"]),
                "dominant_relationship_families": sorted(aggregate["dominant_relationship_families"]),
                "signature_terms": sorted(aggregate["signature_terms"])[:18],
                "mathematical_invariant_hash_counts": dict(sorted(aggregate["mathematical_invariant_hashes"].items())),
                "mathematical_invariant_hashes": sorted(aggregate["mathematical_invariant_hashes"]),
                "mathematical_invariant_terms": sorted(aggregate["mathematical_invariant_terms"])[:24],
                "vector_direction_counts": dict(sorted(aggregate["vector_directions"].items())),
                "temporal_direction_counts": dict(sorted(aggregate["temporal_directions"].items())),
                "routing_label_counts": dict(sorted(aggregate["routing_labels"].items())),
                "routing_recommendation_labels": sorted(aggregate["routing_labels"]),
                "interaction_class_counts": dict(sorted(aggregate["interaction_classes"].items())),
                "interaction_classes": sorted(aggregate["interaction_classes"]),
                "interaction_class_family_counts": dict(sorted(aggregate["interaction_class_families"].items())),
                "interaction_class_families": sorted(aggregate["interaction_class_families"]),
                "confidence_score_max": max_confidence,
                "confidence_score_mean": mean_confidence,
                "confidence_band": score_band(max_confidence),
                "confidence_model": "cross_game_observed_relationship_signature_aggregate_not_probability",
                "confidence_is_probability": False,
                "solve_like_evidence_flag_count": int(aggregate["solve_like_flag_count"]),
                "solve_like_evidence_score_max": round(max(solve_scores), 6) if solve_scores else 0.0,
                "novelty_scope": "unique_in_controlled_probe" if occurrence_total == 1 else "repeated_in_controlled_probe",
                "cross_game_status": "repeated_across_games" if len(game_ids) > 1 else "single_game_observed",
                "review_priority_score": round(
                    max_confidence
                    + (0.10 if occurrence_total == 1 else 0.0)
                    + min(0.12, max(0, len(game_ids) - 1) * 0.03)
                    + (0.05 if "temporal_trajectory_change" in aggregate["dominant_relationship_families"] else 0.0),
                    6,
                ),
                "example_case_refs": aggregate["example_case_refs"],
                "occurrences": aggregate["occurrences"][:12],
                "evidence_status": "cross_game_relationship_atlas_candidate_not_truth",
                **TRUTH_FLAGS,
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row.get("review_priority_score") or 0.0),
            0 if row.get("novelty_scope") == "unique_in_controlled_probe" else 1,
            -int(row.get("game_count") or 0),
            str(row.get("relation_type_label")),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["atlas_rank"] = rank
    return {
        "schema": RELATIONSHIP_ATLAS_SCHEMA,
        "status": "ranked" if rows else "no_relationship_signatures",
        "definition": "cross_game_relative_state_relationship_signature_atlas",
        "row_count": len(rows),
        "total_occurrence_count": occurrence_count,
        "unique_signature_count": sum(1 for row in rows if row.get("novelty_scope") == "unique_in_controlled_probe"),
        "repeated_signature_count": sum(1 for row in rows if row.get("novelty_scope") == "repeated_in_controlled_probe"),
        "cross_game_repeated_signature_count": sum(1 for row in rows if row.get("cross_game_status") == "repeated_across_games"),
        "solve_like_evidence_flag_count": sum(int(row.get("solve_like_evidence_flag_count") or 0) for row in rows),
        "confidence_model": "cross_game_observed_relationship_signature_aggregate_not_probability",
        "confidence_is_probability": False,
        "atlas_policy": "index_controlled_relative_state_signatures_for_review_not_action_authority",
        "rows": rows,
        "top_rows": rows[:24],
        "evidence_status": "cross_game_relationship_atlas_candidate_not_truth",
        **TRUTH_FLAGS,
    }


def build_index(manifest: dict[str, Any], *, index_dir: Path) -> str:
    coverage = manifest.get("coverage_summary") if isinstance(manifest.get("coverage_summary"), dict) else {}
    correspondence_confidence = coverage.get("object_correspondence_confidence") if isinstance(coverage.get("object_correspondence_confidence"), dict) else {}
    relationship_atlas = manifest.get("relationship_atlas") if isinstance(manifest.get("relationship_atlas"), dict) else {}
    atlas_rel = os.path.relpath(resolve_repo_path(manifest.get("relationship_atlas_json")), start=index_dir) if manifest.get("relationship_atlas_json") else ""
    cases = [case for case in manifest.get("cases", []) or [] if isinstance(case, dict)]
    first_case = cases[0] if cases else {}
    first_sheet = ""
    first_overlay = ""
    first_feedback = ""
    first_sequence_sheet = ""
    if first_case:
        first_sheet = os.path.relpath(resolve_repo_path((first_case.get("case_sheet") or {}).get("path")), start=index_dir)
        first_overlay = os.path.relpath(resolve_repo_path((first_case.get("interaction_overlay") or {}).get("path")), start=index_dir)
        first_feedback = os.path.relpath(resolve_repo_path(first_case.get("portband_surface_feedback_path")), start=index_dir)
        first_sequence_sheet = os.path.relpath(resolve_repo_path((first_case.get("action_sequence_contact_sheet") or {}).get("path")), start=index_dir)
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Cross-Game Action Interaction Probe</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f7f7f4;color:#191919}",
        "section{margin:0 0 36px 0;padding-bottom:24px;border-bottom:1px solid #ddd}",
        "img{image-rendering:pixelated;border:1px solid #aaa;background:#111;max-width:100%;height:auto}",
        "table{border-collapse:collapse;font-size:12px;margin-top:12px;max-width:100%}",
        "td,th{border:1px solid #ccc;padding:5px 7px;text-align:left;vertical-align:top}",
        "code{background:#eee;padding:1px 3px}",
        ".cards{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}",
        ".card{background:#fff;border:1px solid #ddd;padding:10px;min-width:140px}",
        ".card b{display:block;font-size:24px;margin-top:4px}",
        "</style></head><body>",
        "<h1>Cross-Game Action Interaction Probe</h1>",
        f"<p><b>Status:</b> <code>{html.escape(str(manifest.get('status')))}</code> "
        f"<b>Run ID:</b> <code>{html.escape(str(manifest.get('run_id')))}</code></p>",
        f"<p><b>Command:</b> <code>{html.escape(str(manifest.get('command')))}</code></p>",
        "<p>Each case executes RESET then one real non-SP80 ARC action through the environment API. "
        "The derived PortBand rows are candidate interaction evidence only; they are not solve truth or action authority.</p>",
    ]
    if first_case:
        body.extend(
            [
                "<h2>First Controlled Frame Delta</h2>",
                f"<p><b>{html.escape(str(first_case.get('game_id')))}</b> level {html.escape(str(first_case.get('level_index')))} / "
                f"{html.escape(str((first_case.get('action') or {}).get('name')))} "
                f"<a href='{first_overlay}'>open interaction overlay</a> | "
                f"<a href='{first_sequence_sheet}'>open action sequence</a> | "
                f"<a href='{first_sheet}'>open case sheet</a> | "
                f"<a href='{first_feedback}'>open feedback JSON</a></p>",
                f"<img src='{first_overlay}'>",
                f"<img src='{first_sequence_sheet}'>",
                f"<img src='{first_sheet}'>",
            ]
        )
    body.extend(
        [
        "<div class='cards'>",
        f"<div class='card'>Cases<b>{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Games<b>{coverage.get('controlled_game_count', len(set(manifest.get('game_ids') or [])))}</b></div>",
        f"<div class='card'>Actions<b>{coverage.get('controlled_action_name_count')}</b></div>",
        f"<div class='card'>Candidate Pairs<b>{coverage.get('candidate_interaction_pair_count_total')}</b></div>",
        f"<div class='card'>Multi-Pair Cases<b>{coverage.get('multi_candidate_interaction_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Temporal Profiles<b>{coverage.get('temporal_trajectory_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Action Frame Seqs<b>{coverage.get('action_frame_sequence_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Sequence Sheets<b>{coverage.get('action_sequence_contact_sheet_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Multi-Frame Actions<b>{coverage.get('multi_frame_action_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Interaction Edges<b>{manifest.get('total_action_interaction_edge_count')}</b></div>",
        f"<div class='card'>Novel Cases<b>{coverage.get('novel_relationship_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Overlays<b>{coverage.get('interaction_overlay_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Matrix Coverage<b>{coverage.get('action_interaction_matrix_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Solve-Like Ledger Rows<b>{manifest.get('total_solve_like_candidate_ledger_row_count')}</b></div>",
        f"<div class='card'>Unique Ledger Rows<b>{manifest.get('total_solve_like_candidate_ledger_unique_count')}</b></div>",
        f"<div class='card'>Profiles<b>{coverage.get('relative_profile_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Correspondence<b>{coverage.get('object_correspondence_case_count')}/{manifest.get('case_count')}</b></div>",
        f"<div class='card'>Relative Tags<b>{manifest.get('total_relative_state_change_tag_count')}</b></div>",
        f"<div class='card'>Atlas Rows<b>{relationship_atlas.get('row_count')}</b></div>",
        f"<div class='card'>Unique Signatures<b>{relationship_atlas.get('unique_signature_count')}</b></div>",
        f"<div class='card'>Cross-Game Repeats<b>{relationship_atlas.get('cross_game_repeated_signature_count')}</b></div>",
        f"<div class='card'>Route-Flow Locks<b>{manifest.get('total_route_flow_lock_count')}</b></div>",
        "</div>",
        "<h2>Coverage Matrix</h2>",
        "<table><thead><tr><th>field</th><th>value</th></tr></thead><tbody>",
        f"<tr><td>attempted actions</td><td>{coverage.get('attempted_action_count')}</td></tr>",
        f"<tr><td>controlled cases</td><td>{coverage.get('controlled_case_count')}</td></tr>",
        f"<tr><td>distinct games</td><td>{coverage.get('controlled_game_count')}</td></tr>",
        f"<tr><td>distinct actions</td><td>{coverage.get('controlled_action_name_count')}</td></tr>",
        f"<tr><td>candidate interaction pairs</td><td>{coverage.get('candidate_interaction_pair_count_total')}</td></tr>",
        f"<tr><td>candidate pairs per case mean</td><td>{coverage.get('candidate_interaction_pair_count_mean')}</td></tr>",
        f"<tr><td>multi-pair cases</td><td>{coverage.get('multi_candidate_interaction_case_count')}</td></tr>",
        f"<tr><td>temporal trajectory cases</td><td>{coverage.get('temporal_trajectory_case_count')}</td></tr>",
        f"<tr><td>saved action frame sequences</td><td>{coverage.get('action_frame_sequence_case_count')}</td></tr>",
        f"<tr><td>action sequence contact sheets</td><td>{coverage.get('action_sequence_contact_sheet_case_count')}</td></tr>",
        f"<tr><td>multi-frame action cases</td><td>{coverage.get('multi_frame_action_case_count')}</td></tr>",
        f"<tr><td>temporal direction counts</td><td>{html.escape(str(coverage.get('temporal_trajectory_direction_counts')))}</td></tr>",
        f"<tr><td>novel relationship cases</td><td>{coverage.get('novel_relationship_case_count')}</td></tr>",
        f"<tr><td>novel relationship candidates</td><td>{coverage.get('novel_relationship_candidate_count_total')}</td></tr>",
        f"<tr><td>edge coverage ratio</td><td>{coverage.get('edge_coverage_ratio')}</td></tr>",
        f"<tr><td>surface event ratio</td><td>{coverage.get('action_surface_event_ratio')}</td></tr>",
        f"<tr><td>relationship observation ratio</td><td>{coverage.get('relationship_observation_ratio')}</td></tr>",
        f"<tr><td>interaction overlay ratio</td><td>{coverage.get('interaction_overlay_ratio')}</td></tr>",
        f"<tr><td>relative profile ratio</td><td>{coverage.get('relative_profile_ratio')}</td></tr>",
        f"<tr><td>object correspondence ratio</td><td>{coverage.get('object_correspondence_ratio')}</td></tr>",
        f"<tr><td>interaction matrix ratio</td><td>{coverage.get('action_interaction_matrix_ratio')}</td></tr>",
        f"<tr><td>solve-like ledger rows</td><td>{manifest.get('total_solve_like_candidate_ledger_row_count')}</td></tr>",
        f"<tr><td>unique solve-like ledger rows</td><td>{manifest.get('total_solve_like_candidate_ledger_unique_count')}</td></tr>",
        f"<tr><td>correspondence confidence mean</td><td>{correspondence_confidence.get('mean')}</td></tr>",
        f"<tr><td>interaction kinds</td><td>{html.escape(str(coverage.get('action_interaction_kind_counts')))}</td></tr>",
        f"<tr><td>relation labels</td><td>{html.escape(str(coverage.get('relation_type_labels')))}</td></tr>",
        f"<tr><td>relationship atlas rows</td><td>{relationship_atlas.get('row_count')}</td></tr>",
        f"<tr><td>unique relationship signatures</td><td>{relationship_atlas.get('unique_signature_count')}</td></tr>",
        f"<tr><td>cross-game repeated signatures</td><td>{relationship_atlas.get('cross_game_repeated_signature_count')}</td></tr>",
        f"<tr><td>relationship atlas json</td><td><a href='{atlas_rel}'>{html.escape(str(manifest.get('relationship_atlas_json')))}</a></td></tr>",
        f"<tr><td>skip reasons</td><td>{html.escape(str(coverage.get('skip_reason_counts')))}</td></tr>",
        "</tbody></table>",
        "<h2>Relationship Atlas</h2>",
        "<p>Cross-game index of relative-state signatures from controlled candidate object interactions. Review guidance only; not action authority or solve truth.</p>",
        "<table><thead><tr><th>rank</th><th>scope</th><th>cross-game</th><th>interaction classes</th><th>math invariant</th><th>relation</th><th>occurrences</th><th>games</th><th>actions</th><th>confidence</th><th>families</th><th>temporal dirs</th><th>route labels</th><th>example</th></tr></thead><tbody>",
        ]
    )
    for row in relationship_atlas.get("top_rows", []) or []:
        if not isinstance(row, dict):
            continue
        example = (row.get("example_case_refs") or [{}])[0] if isinstance(row.get("example_case_refs"), list) and row.get("example_case_refs") else {}
        example_link = ""
        if isinstance(example, dict) and example.get("interaction_overlay_path"):
            example_link = os.path.relpath(resolve_repo_path(example.get("interaction_overlay_path")), start=index_dir)
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('atlas_rank')))}</td>"
            f"<td>{html.escape(str(row.get('novelty_scope')))}</td>"
            f"<td>{html.escape(str(row.get('cross_game_status')))}</td>"
            f"<td>{html.escape(str(row.get('interaction_classes')))}</td>"
            f"<td>{html.escape(str(row.get('mathematical_invariant_hashes')))}</td>"
            f"<td>{html.escape(str(row.get('relation_type_label')))}</td>"
            f"<td>{html.escape(str(row.get('occurrence_count')))}</td>"
            f"<td>{html.escape(str(row.get('game_count')))}</td>"
            f"<td>{html.escape(str(row.get('action_primitives')))}</td>"
            f"<td>{html.escape(str(row.get('confidence_score_max')))} / {html.escape(str(row.get('confidence_band')))}</td>"
            f"<td>{html.escape(str(row.get('dominant_relationship_families')))}</td>"
            f"<td>{html.escape(str(row.get('temporal_direction_counts')))}</td>"
            f"<td>{html.escape(str(row.get('routing_recommendation_labels')))}</td>"
            f"<td><a href='{example_link}'>{html.escape(str(example.get('case_id') if isinstance(example, dict) else ''))}</a></td>"
            "</tr>"
        )
    body.extend(
        [
        "</tbody></table>",
        ]
    )
    for case in cases:
        feedback = case.get("portband_surface_feedback_summary") or {}
        trajectory = case.get("action_temporal_trajectory") if isinstance(case.get("action_temporal_trajectory"), dict) else {}
        trajectory_vector = trajectory.get("trajectory_vector") if isinstance(trajectory.get("trajectory_vector"), dict) else {}
        sequence_rows = [row for row in case.get("action_frame_sequence", []) or [] if isinstance(row, dict)]
        first_action_frame = ""
        last_action_frame = ""
        if sequence_rows:
            first_action_frame = os.path.relpath(resolve_repo_path((sequence_rows[0].get("png") or {}).get("path")), start=index_dir)
            last_action_frame = os.path.relpath(resolve_repo_path((sequence_rows[-1].get("png") or {}).get("path")), start=index_dir)
        sequence_sheet = os.path.relpath(
            resolve_repo_path((case.get("action_sequence_contact_sheet") or {}).get("path")),
            start=index_dir,
        )
        body.extend(
            [
                "<section>",
                f"<h2>{html.escape(str(case['game_id']))} level {case['level_index']} / {html.escape(str(case['action']['name']))}</h2>",
                f"<p><b>Changed pixels:</b> {case['changed_pixel_count']} "
                f"<b>Change bbox:</b> <code>{html.escape(str(case['change_bbox_xyxy']))}</code> "
                f"<b>Returned frames:</b> {case['after_frame_return'].get('returned_frame_count')} "
                f"<b>Trajectory:</b> {html.escape(str(trajectory_vector.get('direction_8way')))} / path {html.escape(str(trajectory.get('path_distance_px')))}</p>",
                f"<p><a href='{os.path.relpath(resolve_repo_path((case.get('interaction_overlay') or {}).get('path')), start=index_dir)}'>open interaction overlay</a> | "
                f"<a href='{sequence_sheet}'>open action sequence</a> | "
                f"<a href='{os.path.relpath(resolve_repo_path(case['case_sheet']['path']), start=index_dir)}'>open case sheet</a> | "
                f"<a href='{os.path.relpath(resolve_repo_path(case['portband_surface_feedback_path']), start=index_dir)}'>open feedback JSON</a>"
                + (f" | <a href='{first_action_frame}'>first action frame</a> | <a href='{last_action_frame}'>last action frame</a>" if first_action_frame and last_action_frame else "")
                + "</p>",
                f"<img src='{os.path.relpath(resolve_repo_path((case.get('interaction_overlay') or {}).get('path')), start=index_dir)}'>",
                f"<img src='{sequence_sheet}'>",
                f"<img src='{os.path.relpath(resolve_repo_path(case['case_sheet']['path']), start=index_dir)}'>",
                "<table><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>",
                f"<tr><td>action tags</td><td>{feedback.get('action_state_tag_count')}</td></tr>",
                f"<tr><td>relative tags</td><td>{feedback.get('relative_state_change_tag_count')}</td></tr>",
                f"<tr><td>interaction edges</td><td>{feedback.get('action_interaction_edge_count')}</td></tr>",
                f"<tr><td>interaction kinds</td><td>{html.escape(str(feedback.get('action_interaction_kinds')))}</td></tr>",
                f"<tr><td>interaction matrix rows</td><td>{feedback.get('action_interaction_matrix_row_count')}</td></tr>",
                f"<tr><td>solve-like ledger rows</td><td>{feedback.get('solve_like_candidate_ledger_row_count')}</td></tr>",
                f"<tr><td>unique solve-like ledger rows</td><td>{feedback.get('solve_like_candidate_ledger_unique_count')}</td></tr>",
                f"<tr><td>interaction support objects</td><td>{feedback.get('action_interaction_support_object_count')}</td></tr>",
                f"<tr><td>relative profile edges</td><td>{feedback.get('relative_profile_edge_count')}</td></tr>",
                f"<tr><td>relative profile labels</td><td>{html.escape(str(feedback.get('relative_profile_relation_type_labels')))}</td></tr>",
                f"<tr><td>candidate interaction pairs</td><td>{feedback.get('candidate_interaction_pair_count')}</td></tr>",
                f"<tr><td>surface result events</td><td>{feedback.get('action_surface_result_event_count')}</td></tr>",
                f"<tr><td>relationship observations</td><td>{feedback.get('relationship_observation_count')}</td></tr>",
                f"<tr><td>temporal trajectory status</td><td>{feedback.get('temporal_trajectory_status')}</td></tr>",
                f"<tr><td>temporal frame count</td><td>{feedback.get('temporal_trajectory_frame_count')}</td></tr>",
                f"<tr><td>temporal multi-frame</td><td>{feedback.get('temporal_trajectory_multi_frame_available')}</td></tr>",
                f"<tr><td>temporal direction</td><td>{feedback.get('temporal_trajectory_direction_8way')}</td></tr>",
                f"<tr><td>temporal path distance px</td><td>{feedback.get('temporal_trajectory_path_distance_px')}</td></tr>",
                f"<tr><td>temporal peak changed pixels</td><td>{feedback.get('temporal_trajectory_peak_changed_pixel_count')}</td></tr>",
                f"<tr><td>novel relationships</td><td>{feedback.get('novel_relationship_candidate_count')}</td></tr>",
                f"<tr><td>solve-like flags</td><td>{feedback.get('solve_like_evidence_flag_count')}</td></tr>",
                f"<tr><td>route-flow locks</td><td>{feedback.get('route_flow_lock_count')}</td></tr>",
                f"<tr><td>object correspondence score</td><td>{html.escape(str((case.get('object_correspondence') or {}).get('confidence_score')))}</td></tr>",
                "</tbody></table>",
                "<table><thead><tr><th>rank</th><th>class</th><th>score</th><th>unique</th><th>math invariant</th><th>participants</th><th>route</th></tr></thead><tbody>",
            ]
        )
        for row in feedback.get("top_solve_like_candidate_ledger_rows", []) or []:
            hint = row.get("routing_recommendation") if isinstance(row.get("routing_recommendation"), dict) else {}
            body.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('candidate_rank')))}</td>"
                f"<td>{html.escape(str(row.get('interaction_class')))}</td>"
                f"<td>{html.escape(str(row.get('solve_like_evidence_score')))}</td>"
                f"<td>{html.escape(str(row.get('is_unique_relationship_candidate')))}</td>"
                f"<td>{html.escape(str(row.get('mathematical_invariant_hash')))}</td>"
                f"<td>{html.escape(str(row.get('participant_object_ids')))}</td>"
                f"<td>{html.escape(str(hint.get('label')))}</td>"
                "</tr>"
            )
        body.extend(
            [
                "</tbody></table>",
                "<table><thead><tr><th>rank</th><th>kind</th><th>action</th><th>participants</th><th>score</th><th>novelty</th><th>flag</th><th>relation label</th><th>vector</th><th>signature</th></tr></thead><tbody>",
            ]
        )
        for edge in feedback.get("top_action_interaction_edges", []) or []:
            profile = edge.get("relationship_profile") if isinstance(edge.get("relationship_profile"), dict) else {}
            body.append(
                "<tr>"
                f"<td>{html.escape(str(edge.get('interaction_rank')))}</td>"
                f"<td>{html.escape(str(edge.get('interaction_kind')))}</td>"
                f"<td>{html.escape(str(edge.get('action_primitive')))}</td>"
                f"<td>{html.escape(str(edge.get('participant_object_ids')))}</td>"
                f"<td>{html.escape(str(edge.get('solve_like_evidence_score')))}</td>"
                f"<td>{html.escape(str(edge.get('novelty_status')))}</td>"
                f"<td>{html.escape(str(edge.get('solve_like_evidence_flag')))}</td>"
                f"<td>{html.escape(str(edge.get('relation_type_label')))}</td>"
                f"<td>{html.escape(str(profile.get('vector_delta')))}</td>"
                f"<td>{html.escape(str(edge.get('relationship_signature')))}</td>"
                "</tr>"
            )
        body.extend(
            [
                "</tbody></table>",
                "<table><thead><tr><th>rank</th><th>relation label</th><th>score</th><th>novelty</th><th>signature</th></tr></thead><tbody>",
            ]
        )
        for edge in feedback.get("top_relative_action_interaction_edges", []) or []:
            body.append(
                "<tr>"
                f"<td>{html.escape(str(edge.get('interaction_rank')))}</td>"
                f"<td>{html.escape(str(edge.get('relation_type_label')))}</td>"
                f"<td>{html.escape(str(edge.get('solve_like_evidence_score')))}</td>"
                f"<td>{html.escape(str(edge.get('novelty_status')))}</td>"
                f"<td>{html.escape(str(edge.get('relationship_signature')))}</td>"
                "</tr>"
            )
        body.extend(
            [
                "</tbody></table>",
                "<table><thead><tr><th>rank</th><th>kind</th><th>class</th><th>math invariant</th><th>participants</th><th>confidence</th><th>relation</th><th>route hint</th></tr></thead><tbody>",
            ]
        )
        for row in feedback.get("top_action_interaction_matrix_rows", []) or []:
            hint = row.get("routing_recommendation") if isinstance(row.get("routing_recommendation"), dict) else {}
            body.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('interaction_rank')))}</td>"
                f"<td>{html.escape(str(row.get('interaction_kind')))}</td>"
                f"<td>{html.escape(str(row.get('interaction_class')))}</td>"
                f"<td>{html.escape(str(row.get('mathematical_invariant_hash')))}</td>"
                f"<td>{html.escape(str(row.get('participant_object_ids')))}</td>"
                f"<td>{html.escape(str(row.get('confidence_score')))}</td>"
                f"<td>{html.escape(str(row.get('relation_type_label')))}</td>"
                f"<td>{html.escape(str(hint.get('label')))}</td>"
                "</tr>"
            )
        body.extend(["</tbody></table>", "</section>"])
    body.extend(["</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/cross_game_action_interaction_probe"))
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--max-cases-per-game", type=int, default=1)
    parser.add_argument("--max-correspondence-pairs", type=int, default=3)
    parser.add_argument("--level-index", type=int, default=0)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--game-short", action="append", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    run_id = (
        out_dir.name
        if out_dir.name.startswith("cross_game_action_interaction_probe_")
        else f"cross_game_action_interaction_probe_{started.strftime('%Y%m%dT%H%M%SZ')}"
    )
    selected_games = discover_games()
    if args.game_short:
        allowed = set(args.game_short)
        selected_games = [game for game in selected_games if game["game_short"] in allowed]
    cases: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    used_action_names: set[str] = set()
    for game_info in selected_games:
        if len(cases) >= int(args.max_cases):
            break
        game_case_count = 0
        action_candidates = sorted(
            ACTION_ORDER,
            key=lambda action: (action.name in used_action_names, ACTION_ORDER.index(action)),
        )
        for action in action_candidates:
            if len(cases) >= int(args.max_cases):
                break
            if game_case_count >= int(args.max_cases_per_game):
                break
            try:
                case = run_action_case(
                    game_info=game_info,
                    action=action,
                    level_index=int(args.level_index),
                    out_dir=out_dir,
                    scale=max(1, int(args.scale)),
                    max_correspondence_pairs=max(1, int(args.max_correspondence_pairs)),
                )
            except Exception as exc:
                skipped.append(
                    {
                        "game_id": game_info["game_id"],
                        "game_short": game_info["game_short"],
                        "level_index": int(args.level_index),
                        "action_name": action.name,
                        "skip_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            if case is None:
                skipped.append(
                    {
                        "game_id": game_info["game_id"],
                        "game_short": game_info["game_short"],
                        "level_index": int(args.level_index),
                        "action_name": action.name,
                        "skip_reason": "no_2d_final_frame_delta",
                    }
                )
                continue
            cases.append(case)
            game_case_count += 1
            used_action_names.add(action.name)
            trace_rows.append(
                {
                    "schema": "codexlab.cross_game_action_interaction_trace_event.v1",
                    "run_id": run_id,
                    "event_index": len(trace_rows) + 1,
                    "event_type": "controlled_action_case",
                    "case_id": case["case_id"],
                    "game_id": case["game_id"],
                    "level_index": case["level_index"],
                    "action": case["action"],
                    "before_frame_sha256": case["before_frame"]["raw_grid"]["raw_frame_sha256"],
                    "after_frame_sha256": case["after_frame"]["raw_grid"]["raw_frame_sha256"],
                    "changed_pixel_count": case["changed_pixel_count"],
                    "action_returned_frame_count": case["after_frame_return"].get("returned_frame_count"),
                    "temporal_trajectory_status": (case.get("action_temporal_trajectory") or {}).get("trajectory_status")
                    if isinstance(case.get("action_temporal_trajectory"), dict)
                    else None,
                    "temporal_trajectory_path_distance_px": (case.get("action_temporal_trajectory") or {}).get("path_distance_px")
                    if isinstance(case.get("action_temporal_trajectory"), dict)
                    else None,
                    "action_sequence_contact_sheet": case.get("action_sequence_contact_sheet"),
                    "candidate_interaction_pair_count": case["portband_surface_feedback_summary"]["candidate_interaction_pair_count"],
                    "action_interaction_edge_count": case["portband_surface_feedback_summary"]["action_interaction_edge_count"],
                    "action_interaction_matrix_row_count": case["portband_surface_feedback_summary"]["action_interaction_matrix_row_count"],
                    "solve_like_candidate_ledger_row_count": case["portband_surface_feedback_summary"].get("solve_like_candidate_ledger_row_count"),
                    "truth_boundary": case["truth_boundary"],
                    **TRUTH_FLAGS,
                }
            )
    manifest_path = out_dir / "manifest.json"
    trace_path = out_dir / "report_trace.jsonl"
    atlas_path = out_dir / "relationship_atlas.json"
    coverage_summary = build_coverage_summary(cases, skipped)
    relationship_atlas = build_relationship_atlas(cases)
    manifest = {
        "schema": SCHEMA,
        "status": "controlled" if cases else "no_controlled_cases",
        "run_id": run_id,
        "created_at_utc": started.isoformat().replace("+00:00", "Z"),
        "finished_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "working_directory": str(ROOT),
        "host": socket.gethostname(),
        "command": " ".join(sys.argv),
        "evidence_category": "controlled_run" if cases else "invalid",
        "case_count": len(cases),
        "skipped_count": len(skipped),
        "manifest_json": relpath(manifest_path),
        "report_trace_jsonl": relpath(trace_path),
        "relationship_atlas_json": relpath(atlas_path),
        "index_html": relpath(out_dir / "index.html"),
        "source_paths": [
            {"path": "codexlab/cross_game_action_interaction_probe.py", "sha256": sha256_file(ROOT / "codexlab/cross_game_action_interaction_probe.py")},
            {"path": "codexlab/contra_head/portband.py", "sha256": sha256_file(ROOT / "codexlab/contra_head/portband.py")},
        ],
        "case_ids": [case["case_id"] for case in cases],
        "game_ids": [case["game_id"] for case in cases],
        "total_changed_pixel_count": sum(int(case["changed_pixel_count"]) for case in cases),
        "total_action_interaction_edge_count": sum(
            int((case.get("portband_surface_feedback_summary") or {}).get("action_interaction_edge_count") or 0) for case in cases
        ),
        "total_action_interaction_matrix_row_count": sum(
            int((case.get("portband_surface_feedback_summary") or {}).get("action_interaction_matrix_row_count") or 0) for case in cases
        ),
        "total_solve_like_candidate_ledger_row_count": sum(
            int((case.get("portband_surface_feedback_summary") or {}).get("solve_like_candidate_ledger_row_count") or 0) for case in cases
        ),
        "total_solve_like_candidate_ledger_unique_count": sum(
            int((case.get("portband_surface_feedback_summary") or {}).get("solve_like_candidate_ledger_unique_count") or 0) for case in cases
        ),
        "total_candidate_interaction_pair_count": sum(
            int((case.get("portband_surface_feedback_summary") or {}).get("candidate_interaction_pair_count") or 0) for case in cases
        ),
        "total_relative_state_change_tag_count": sum(
            int((case.get("portband_surface_feedback_summary") or {}).get("relative_state_change_tag_count") or 0) for case in cases
        ),
        "total_temporal_trajectory_case_count": sum(
            1
            for case in cases
            if isinstance(case.get("action_temporal_trajectory"), dict)
            and (case.get("action_temporal_trajectory") or {}).get("schema") == TEMPORAL_TRAJECTORY_SCHEMA
        ),
        "total_action_sequence_contact_sheet_count": sum(
            1
            for case in cases
            if isinstance(case.get("action_sequence_contact_sheet"), dict)
            and (case.get("action_sequence_contact_sheet") or {}).get("kind") == "action_sequence_contact_sheet_review_aid_not_raw_frame"
        ),
        "total_multi_frame_action_case_count": sum(
            1
            for case in cases
            if isinstance(case.get("action_temporal_trajectory"), dict)
            and (case.get("action_temporal_trajectory") or {}).get("multi_frame_available") is True
        ),
        "total_route_flow_lock_count": sum(
            int((case.get("portband_surface_feedback_summary") or {}).get("route_flow_lock_count") or 0) for case in cases
        ),
        "relationship_atlas_row_count": relationship_atlas.get("row_count"),
        "relationship_atlas_unique_signature_count": relationship_atlas.get("unique_signature_count"),
        "relationship_atlas_cross_game_repeated_signature_count": relationship_atlas.get("cross_game_repeated_signature_count"),
        "relationship_atlas_total_occurrence_count": relationship_atlas.get("total_occurrence_count"),
        "coverage_summary": coverage_summary,
        "relationship_atlas": relationship_atlas,
        "cases": cases,
        "skipped": skipped,
        "boundaries": [
            "Each case executes RESET then one environment action through arcengine.",
            "PortBand interaction edges are candidate evidence only.",
            "Temporal trajectory profiles are derived from returned action frames when the environment returns a sequence.",
            "No edge is solve truth, prediction, action choice, or control authority.",
            "No SP80 route-flow lock is expected or admitted for these non-SP80 games.",
        ],
        **TRUTH_FLAGS,
    }
    write_json(atlas_path, relationship_atlas)
    write_json(manifest_path, manifest)
    write_jsonl(trace_path, trace_rows)
    (out_dir / "index.html").write_text(build_index(manifest, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "case_count": manifest["case_count"],
                "coverage_summary": manifest["coverage_summary"],
                "manifest": manifest["manifest_json"],
                "index_html": manifest["index_html"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
