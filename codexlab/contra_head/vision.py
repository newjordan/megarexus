#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

# Scene surface hypotheses are part of Contra vision.
SCENE_SURFACE_SCHEMA = "codexlab.scene_surface_hypotheses.v3"
COMPONENT_GRAPH_SCHEMA = "codexlab.raw_frame_color_component_graph.v1"
ROLE_HYPOTHESIS_SCHEMA = "codexlab.scene_surface_role_hypothesis.v1"
ASYMMETRY_SCHEMA = "codexlab.raw_component_asymmetry_hypothesis.v1"
ANTISYMMETRIC_RELATIONSHIP_SCHEMA = "codexlab.antisymmetric_component_relationship_hypothesis.v1"
SCENE_PATTERN_HYPOTHESIS_SCHEMA = "codexlab.scene_pattern_hypothesis.v1"


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def raw_pixel_components_with_labels(raw_frame: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    arr = np.asarray(raw_frame).astype(int)
    height, width = arr.shape[:2]
    seen = np.zeros((height, width), dtype=bool)
    label_map = np.full((height, width), -1, dtype=np.int32)
    components: list[dict[str, Any]] = []
    value_component_counts: dict[int, int] = defaultdict(int)

    for start_y in range(height):
        for start_x in range(width):
            if seen[start_y, start_x]:
                continue
            raw_value = int(arr[start_y, start_x])
            component_index_for_color = value_component_counts[raw_value]
            value_component_counts[raw_value] += 1
            label_index = len(components)
            stack = [(start_y, start_x)]
            seen[start_y, start_x] = True
            label_map[start_y, start_x] = label_index
            points: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                points.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if seen[ny, nx] or int(arr[ny, nx]) != raw_value:
                        continue
                    seen[ny, nx] = True
                    label_map[ny, nx] = label_index
                    stack.append((ny, nx))
            ys = np.asarray([point[0] for point in points], dtype=np.float32)
            xs = np.asarray([point[1] for point in points], dtype=np.float32)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            component_id = f"c{label_index:04d}_v{raw_value:02d}_{component_index_for_color:02d}"
            local_mask = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=np.uint8)
            for py, px in points:
                local_mask[int(py) - bbox[1], int(px) - bbox[0]] = 1
            components.append(
                {
                    "component_id": component_id,
                    "label_index": label_index,
                    "raw_pixel_value": raw_value,
                    "component_index_for_color": component_index_for_color,
                    "area_pixels": int(len(points)),
                    "bbox_xyxy": bbox,
                    "center_xy": [float(xs.mean()), float(ys.mean())],
                    "shape_mask_rows": ["".join("1" if value else "0" for value in row) for row in local_mask],
                    "shape_mask_encoding": "bbox_local_binary_rows",
                }
            )
    return components, label_map


def component_asymmetry(component: dict[str, Any], label_map: np.ndarray) -> dict[str, Any]:
    label_index = int(component["label_index"])
    x0, y0, x1, y1 = [int(v) for v in component["bbox_xyxy"]]
    crop = label_map[y0 : y1 + 1, x0 : x1 + 1] == label_index
    area = int(crop.sum())
    if area <= 0:
        return {
            "schema": ASYMMETRY_SCHEMA,
            "status": "invalid_empty_component",
            "mirror_x_difference": 0.0,
            "mirror_y_difference": 0.0,
            "left_right_imbalance": 0.0,
            "top_bottom_imbalance": 0.0,
            "primary_asymmetry_axis": "none",
            "facing_direction_hypothesis": "none",
            "confidence": 0.0,
            "reasons": ["empty_component"],
        }

    mirror_x_difference = float(np.logical_xor(crop, crop[:, ::-1]).sum()) / float(max(1, 2 * area))
    mirror_y_difference = float(np.logical_xor(crop, crop[::-1, :]).sum()) / float(max(1, 2 * area))
    height, width = crop.shape[:2]
    xs = np.nonzero(crop)[1].astype(np.float32)
    ys = np.nonzero(crop)[0].astype(np.float32)
    center_x = float(width - 1) / 2.0
    center_y = float(height - 1) / 2.0
    norm_x = max(1.0, center_x)
    norm_y = max(1.0, center_y)
    left_right_imbalance = float((xs.mean() - center_x) / norm_x) if len(xs) else 0.0
    top_bottom_imbalance = float((ys.mean() - center_y) / norm_y) if len(ys) else 0.0

    axis = "none"
    facing = "none"
    confidence = 0.0
    reasons: list[str] = []
    horizontal_strength = mirror_x_difference * min(1.0, abs(left_right_imbalance) / 0.35)
    vertical_strength = mirror_y_difference * min(1.0, abs(top_bottom_imbalance) / 0.35)
    if horizontal_strength >= 0.08 or vertical_strength >= 0.08:
        if horizontal_strength >= vertical_strength:
            axis = "horizontal"
            facing = "right" if left_right_imbalance > 0 else "left"
            confidence = clamp01(horizontal_strength / 0.45)
            reasons.append("left_right_mass_imbalance")
            if mirror_x_difference >= 0.18:
                reasons.append("high_x_mirror_difference")
        else:
            axis = "vertical"
            facing = "down" if top_bottom_imbalance > 0 else "up"
            confidence = clamp01(vertical_strength / 0.45)
            reasons.append("top_bottom_mass_imbalance")
            if mirror_y_difference >= 0.18:
                reasons.append("high_y_mirror_difference")
    if not reasons:
        reasons.append("near_symmetric_or_low_confidence")

    return {
        "schema": ASYMMETRY_SCHEMA,
        "status": "hypothesis",
        "mirror_x_difference": round(clamp01(mirror_x_difference), 6),
        "mirror_y_difference": round(clamp01(mirror_y_difference), 6),
        "left_right_imbalance": round(float(max(-1.0, min(1.0, left_right_imbalance))), 6),
        "top_bottom_imbalance": round(float(max(-1.0, min(1.0, top_bottom_imbalance))), 6),
        "primary_asymmetry_axis": axis,
        "facing_direction_hypothesis": facing,
        "confidence": round(clamp01(confidence), 6),
        "reasons": reasons,
    }


def contact_graph(components: list[dict[str, Any]], label_map: np.ndarray) -> dict[str, Any]:
    height, width = label_map.shape[:2]
    id_by_label = {int(component["label_index"]): str(component["component_id"]) for component in components}
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    neighbor_ids: dict[int, set[int]] = defaultdict(set)
    neighbor_contact_pixels: dict[int, int] = defaultdict(int)

    for y in range(height):
        for x in range(width):
            here = int(label_map[y, x])
            for dy, dx in ((1, 0), (0, 1)):
                ny = y + dy
                nx = x + dx
                if ny >= height or nx >= width:
                    continue
                there = int(label_map[ny, nx])
                if there == here:
                    continue
                pair = (min(here, there), max(here, there))
                pair_counts[pair] += 1
                neighbor_ids[here].add(there)
                neighbor_ids[there].add(here)
                neighbor_contact_pixels[here] += 1
                neighbor_contact_pixels[there] += 1

    contacts = [
        {
            "component_a": id_by_label[a],
            "component_b": id_by_label[b],
            "contact_edge_count": int(count),
        }
        for (a, b), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    return {
        "contact_pair_count": len(contacts),
        "contacts": contacts,
        "neighbor_ids_by_label": neighbor_ids,
        "neighbor_contact_pixels_by_label": neighbor_contact_pixels,
        "contact_count_by_pair": pair_counts,
    }


def component_features(component: dict[str, Any], *, frame_shape: tuple[int, int], graph: dict[str, Any]) -> dict[str, Any]:
    height, width = frame_shape
    x0, y0, x1, y1 = [int(v) for v in component["bbox_xyxy"]]
    area = int(component["area_pixels"])
    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    label_index = int(component["label_index"])
    neighbor_ids = graph["neighbor_ids_by_label"].get(label_index, set())
    neighbor_contact_pixels = int(graph["neighbor_contact_pixels_by_label"].get(label_index, 0))
    left_edge = 1.0 if x0 == 0 else 0.0
    right_edge = 1.0 if x1 == width - 1 else 0.0
    top_edge = 1.0 if y0 == 0 else 0.0
    bottom_edge = 1.0 if y1 == height - 1 else 0.0
    border_contact = max(left_edge, right_edge, top_edge, bottom_edge)
    edge_contact_count = int(left_edge + right_edge + top_edge + bottom_edge)
    fill_ratio = float(area) / float(max(1, bbox_w * bbox_h))
    thin_horizontal = float(bbox_h) / float(max(1, height))
    thin_vertical = float(bbox_w) / float(max(1, width))
    return {
        "area_ratio": float(area) / float(max(1, height * width)),
        "bbox_width_ratio": float(bbox_w) / float(max(1, width)),
        "bbox_height_ratio": float(bbox_h) / float(max(1, height)),
        "bbox_fill_ratio": fill_ratio,
        "thinness_ratio": min(thin_horizontal, thin_vertical),
        "center_x_ratio": float(component["center_xy"][0]) / float(max(1, width - 1)),
        "center_y_ratio": float(component["center_xy"][1]) / float(max(1, height - 1)),
        "bottom_contact": bottom_edge,
        "top_contact": top_edge,
        "left_contact": left_edge,
        "right_contact": right_edge,
        "border_contact": border_contact,
        "edge_contact_count": edge_contact_count,
        "full_frame_contact": 1.0 if edge_contact_count >= 3 else 0.0,
        "neighbor_component_count": len(neighbor_ids),
        "neighbor_contact_pixels": neighbor_contact_pixels,
        "neighbor_contact_ratio": float(neighbor_contact_pixels) / float(max(1, area)),
    }


def score_floor(features: dict[str, Any]) -> tuple[float, list[str]]:
    area = clamp01(float(features["area_ratio"]) / 0.22)
    horizontal_span = clamp01(float(features["bbox_width_ratio"]))
    bottom_contact = clamp01(float(features["bottom_contact"]))
    low_center = clamp01(float(features["center_y_ratio"]))
    neighbor_diversity = clamp01(float(features["neighbor_component_count"]) / 5.0)
    contact_richness = clamp01(float(features["neighbor_contact_ratio"]) / 0.45)
    full_frame_penalty = 0.28 * float(features["full_frame_contact"])
    score = (
        0.26 * area
        + 0.22 * horizontal_span
        + 0.14 * bottom_contact
        + 0.12 * low_center
        + 0.16 * neighbor_diversity
        + 0.10 * contact_richness
        - full_frame_penalty
    )
    reasons = []
    if bottom_contact >= 0.5:
        reasons.append("touches_bottom_edge")
    if horizontal_span >= 0.65:
        reasons.append("wide_horizontal_span")
    if low_center >= 0.65:
        reasons.append("low_centroid")
    if neighbor_diversity >= 0.4:
        reasons.append("contacts_multiple_components")
    if area >= 0.5:
        reasons.append("large_area")
    if full_frame_penalty > 0:
        reasons.append("penalized_full_frame_substrate")
    return round(clamp01(score), 6), reasons


def score_wall(
    features: dict[str, Any],
    *,
    label_index: int,
    floor_label_index: int | None,
    graph: dict[str, Any],
) -> tuple[float, list[str]]:
    long_run = max(float(features["bbox_width_ratio"]), float(features["bbox_height_ratio"]))
    side_or_top_edge = max(
        float(features["left_contact"]),
        float(features["right_contact"]),
        float(features["top_contact"]),
    )
    thinness = 1.0 - clamp01(float(features["thinness_ratio"]) / 0.18)
    floor_contact_edges = 0
    if floor_label_index is not None:
        pair = (min(label_index, floor_label_index), max(label_index, floor_label_index))
        floor_contact_edges = int(graph["contact_count_by_pair"].get(pair, 0))
    floor_contact = clamp01(float(floor_contact_edges) / 8.0)
    boundary_like = clamp01(0.65 * thinness + 0.35 * long_run)
    area = clamp01(float(features["area_ratio"]) / 0.16)
    filled_substrate_penalty = 0.38 * clamp01(float(features["area_ratio"]) / 0.45) * clamp01(
        (float(features["bbox_fill_ratio"]) - 0.55) / 0.35
    )
    score = (
        0.28 * floor_contact
        + 0.24 * side_or_top_edge
        + 0.20 * thinness
        + 0.16 * boundary_like
        + 0.12 * min(area, 0.7)
        - filled_substrate_penalty
    )
    reasons = []
    if floor_contact >= 0.25:
        reasons.append("contacts_floor_candidate")
    if side_or_top_edge >= 0.5:
        reasons.append("touches_outer_edge")
    if long_run >= 0.55:
        reasons.append("long_run")
    if thinness >= 0.5:
        reasons.append("thin_edge_or_bar")
    if boundary_like >= 0.45:
        reasons.append("boundary_like")
    if filled_substrate_penalty > 0:
        reasons.append("penalized_filled_substrate")
    return round(clamp01(score), 6), reasons


def score_background(features: dict[str, Any], *, dominant_color_match: bool) -> tuple[float, list[str]]:
    area = clamp01(float(features["area_ratio"]) / 0.25)
    border = clamp01(float(features["border_contact"]))
    top_or_side = max(float(features["top_contact"]), float(features["left_contact"]), float(features["right_contact"]))
    low_contact = 1.0 - clamp01(float(features["neighbor_contact_ratio"]) / 0.35)
    filled_region = clamp01((float(features["bbox_fill_ratio"]) - 0.45) / 0.45)
    full_frame = float(features["full_frame_contact"])
    dominant = 1.0 if dominant_color_match else 0.0
    score = (
        0.30 * area
        + 0.18 * border
        + 0.12 * top_or_side
        + 0.12 * low_contact
        + 0.12 * dominant
        + 0.10 * filled_region
        + 0.06 * full_frame
    )
    reasons = []
    if dominant_color_match:
        reasons.append("dominant_raw_color")
    if border >= 0.5:
        reasons.append("touches_outer_edge")
    if top_or_side >= 0.5:
        reasons.append("top_or_side_backdrop")
    if low_contact >= 0.55:
        reasons.append("low_neighbor_contact")
    if area >= 0.5:
        reasons.append("large_area")
    if filled_region >= 0.5:
        reasons.append("filled_region")
    if full_frame >= 0.5:
        reasons.append("full_frame_substrate")
    return round(score, 6), reasons


def ranked_candidate(
    *,
    component: dict[str, Any],
    role: str,
    score: float,
    reasons: list[str],
    features: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": ROLE_HYPOTHESIS_SCHEMA,
        "role": role,
        "component_id": component["component_id"],
        "raw_pixel_value": int(component["raw_pixel_value"]),
        "component_index_for_color": int(component["component_index_for_color"]),
        "score": float(score),
        "reasons": reasons,
        "features": features,
        "asymmetry": component.get("asymmetry"),
        "bbox_xyxy": component["bbox_xyxy"],
        "area_pixels": int(component["area_pixels"]),
        "center_xy": component["center_xy"],
    }


def _cluster_axis_positions(values: list[float], *, tolerance: float) -> list[float]:
    if not values:
        return []
    clusters: list[list[float]] = []
    for value in sorted(float(v) for v in values):
        if clusters and abs(value - (sum(clusters[-1]) / len(clusters[-1]))) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [round(sum(cluster) / len(cluster), 6) for cluster in clusters]


def _nearest_cluster_index(value: float, clusters: list[float]) -> int:
    return min(range(len(clusters)), key=lambda idx: abs(float(value) - float(clusters[idx])))


def checkerboard_floor_pattern_hypotheses(
    components: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    height, width = frame_shape
    tile_candidates = []
    for component in components:
        x0, y0, x1, y1 = [int(value) for value in component["bbox_xyxy"]]
        bbox_w = x1 - x0 + 1
        bbox_h = y1 - y0 + 1
        area = int(component["area_pixels"])
        if area < 4:
            continue
        if area > max(24, int(0.05 * height * width)):
            continue
        fill_ratio = float(area) / float(max(1, bbox_w * bbox_h))
        aspect = float(bbox_w) / float(max(1, bbox_h))
        if fill_ratio < 0.65 or aspect < 0.45 or aspect > 2.20:
            continue
        tile_candidates.append(
            {
                "component_id": str(component["component_id"]),
                "raw_pixel_value": int(component["raw_pixel_value"]),
                "area_pixels": area,
                "bbox_xyxy": [x0, y0, x1, y1],
                "center_xy": [float(component["center_xy"][0]), float(component["center_xy"][1])],
                "bbox_width": bbox_w,
                "bbox_height": bbox_h,
                "bbox_fill_ratio": round(fill_ratio, 6),
            }
        )

    raw_values = sorted({int(row["raw_pixel_value"]) for row in tile_candidates})
    hypotheses: list[dict[str, Any]] = []
    for left_idx, left_raw in enumerate(raw_values):
        for right_raw in raw_values[left_idx + 1 :]:
            rows = [row for row in tile_candidates if int(row["raw_pixel_value"]) in {left_raw, right_raw}]
            left_count = sum(1 for row in rows if int(row["raw_pixel_value"]) == left_raw)
            right_count = sum(1 for row in rows if int(row["raw_pixel_value"]) == right_raw)
            if len(rows) < 8 or left_count < 3 or right_count < 3:
                continue
            widths = [float(row["bbox_width"]) for row in rows]
            heights = [float(row["bbox_height"]) for row in rows]
            median_w = float(np.median(widths))
            median_h = float(np.median(heights))
            tolerance = max(1.0, min(median_w, median_h) * 0.70)
            x_clusters = _cluster_axis_positions([float(row["center_xy"][0]) for row in rows], tolerance=tolerance)
            y_clusters = _cluster_axis_positions([float(row["center_xy"][1]) for row in rows], tolerance=tolerance)
            if len(x_clusters) < 3 or len(y_clusters) < 2:
                continue
            grid: dict[tuple[int, int], int] = {}
            component_ids_by_cell: dict[tuple[int, int], str] = {}
            for row in rows:
                col = _nearest_cluster_index(float(row["center_xy"][0]), x_clusters)
                rix = _nearest_cluster_index(float(row["center_xy"][1]), y_clusters)
                cell = (rix, col)
                existing = grid.get(cell)
                if existing is not None:
                    continue
                grid[cell] = int(row["raw_pixel_value"])
                component_ids_by_cell[cell] = str(row["component_id"])
            possible_cells = max(1, len(x_clusters) * len(y_clusters))
            occupancy_ratio = float(len(grid)) / float(possible_cells)
            if occupancy_ratio < 0.45:
                continue
            adjacent_pairs = 0
            alternating_pairs = 0
            for rix in range(len(y_clusters)):
                for col in range(len(x_clusters)):
                    here = grid.get((rix, col))
                    if here is None:
                        continue
                    for neighbor in ((rix + 1, col), (rix, col + 1)):
                        there = grid.get(neighbor)
                        if there is None:
                            continue
                        adjacent_pairs += 1
                        if int(here) != int(there):
                            alternating_pairs += 1
            if adjacent_pairs < 6:
                continue
            alternating_ratio = float(alternating_pairs) / float(max(1, adjacent_pairs))
            if alternating_ratio < 0.72:
                continue
            x0 = min(int(row["bbox_xyxy"][0]) for row in rows)
            y0 = min(int(row["bbox_xyxy"][1]) for row in rows)
            x1 = max(int(row["bbox_xyxy"][2]) for row in rows)
            y1 = max(int(row["bbox_xyxy"][3]) for row in rows)
            bbox_area_ratio = float((x1 - x0 + 1) * (y1 - y0 + 1)) / float(max(1, height * width))
            lower_half_bias = float((y0 + y1) / 2.0) / float(max(1, height - 1))
            count_balance = min(left_count, right_count) / float(max(1, max(left_count, right_count)))
            score = clamp01(
                0.34 * alternating_ratio
                + 0.22 * occupancy_ratio
                + 0.18 * clamp01(float(len(grid)) / 24.0)
                + 0.14 * count_balance
                + 0.12 * clamp01(lower_half_bias)
            )
            covered_ids = [component_ids_by_cell[cell] for cell in sorted(component_ids_by_cell)]
            hypotheses.append(
                {
                    "schema": SCENE_PATTERN_HYPOTHESIS_SCHEMA,
                    "status": "hypothesis",
                    "pattern_id": f"checkerboard_floor_v{left_raw:02d}_v{right_raw:02d}",
                    "pattern_kind": "checkerboard_floor_candidate",
                    "role": "floor_pattern_hypothesis",
                    "source": "raw_frame_exact_color_component_graph",
                    "raw_pixel_values": [int(left_raw), int(right_raw)],
                    "component_ids": covered_ids,
                    "component_count": len(covered_ids),
                    "bbox_xyxy": [int(x0), int(y0), int(x1), int(y1)],
                    "score": round(score, 6),
                    "metrics": {
                        "grid_rows": len(y_clusters),
                        "grid_columns": len(x_clusters),
                        "occupied_cells": len(grid),
                        "occupancy_ratio": round(clamp01(occupancy_ratio), 6),
                        "adjacent_pair_count": adjacent_pairs,
                        "alternating_pair_count": alternating_pairs,
                        "alternating_ratio": round(clamp01(alternating_ratio), 6),
                        "count_balance": round(clamp01(count_balance), 6),
                        "bbox_area_ratio": round(clamp01(bbox_area_ratio), 6),
                        "lower_half_bias": round(clamp01(lower_half_bias), 6),
                        "median_tile_width": round(median_w, 6),
                        "median_tile_height": round(median_h, 6),
                    },
                    "reasons": [
                        "many_small_repeated_tile_components",
                        "two_color_alternation",
                        "grid_like_component_centers",
                    ],
                    "evidence_status": "hypothesis",
                    "admission_status": "not_admitted",
                    "control_truth_admitted": False,
                    "goal_condition_admitted": False,
                    "action_choice_admitted": False,
                    "solve_condition_admitted": False,
                }
            )
    hypotheses.sort(key=lambda row: (-float(row["score"]), -int(row["component_count"]), row["pattern_id"]))
    return hypotheses


def antisymmetric_relationship_hypotheses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    for idx, left in enumerate(rows):
        left_asym = left.get("asymmetry")
        if not isinstance(left_asym, dict):
            continue
        left_facing = str(left_asym.get("facing_direction_hypothesis"))
        left_confidence = float(left_asym.get("confidence") or 0.0)
        if left_facing not in {"left", "right", "up", "down"} or left_confidence < 0.20:
            continue
        for right in rows[idx + 1 :]:
            right_asym = right.get("asymmetry")
            if not isinstance(right_asym, dict):
                continue
            right_facing = str(right_asym.get("facing_direction_hypothesis"))
            right_confidence = float(right_asym.get("confidence") or 0.0)
            if right_confidence < 0.20:
                continue
            if {left_facing, right_facing} not in ({"left", "right"}, {"up", "down"}):
                continue
            if int(left["raw_pixel_value"]) != int(right["raw_pixel_value"]):
                continue
            area_a = int(left["area_pixels"])
            area_b = int(right["area_pixels"])
            area_similarity = 1.0 - abs(area_a - area_b) / float(max(area_a, area_b, 1))
            if area_similarity < 0.65:
                continue
            axis = "horizontal" if {left_facing, right_facing} == {"left", "right"} else "vertical"
            center_a = left["center_xy"]
            center_b = right["center_xy"]
            if axis == "horizontal":
                alignment = 1.0 - min(1.0, abs(float(center_a[1]) - float(center_b[1])) / 16.0)
            else:
                alignment = 1.0 - min(1.0, abs(float(center_a[0]) - float(center_b[0])) / 16.0)
            if alignment < 0.60:
                continue
            score = clamp01(0.42 * min(left_confidence, right_confidence) + 0.33 * area_similarity + 0.25 * alignment)
            if score < 0.45:
                continue
            relationships.append(
                {
                    "schema": ANTISYMMETRIC_RELATIONSHIP_SCHEMA,
                    "status": "hypothesis",
                    "component_a": left["component_id"],
                    "component_b": right["component_id"],
                    "raw_pixel_value": int(left["raw_pixel_value"]),
                    "axis": axis,
                    "facing_direction_a": left_facing,
                    "facing_direction_b": right_facing,
                    "score": round(score, 6),
                    "metrics": {
                        "area_similarity": round(clamp01(area_similarity), 6),
                        "axis_alignment": round(clamp01(alignment), 6),
                        "component_a_confidence": round(clamp01(left_confidence), 6),
                        "component_b_confidence": round(clamp01(right_confidence), 6),
                    },
                    "reasons": ["opposed_facing_hypotheses", "same_raw_color", "similar_area"],
                }
            )
    relationships.sort(key=lambda row: (-float(row["score"]), row["component_a"], row["component_b"]))
    return relationships


def analyze_scene_surface_hypotheses(raw_frame: np.ndarray) -> tuple[dict[str, Any], np.ndarray, dict[str, int]]:
    arr = np.asarray(raw_frame).astype(int)
    height, width = arr.shape[:2]
    components, label_map = raw_pixel_components_with_labels(arr)
    for component in components:
        component["asymmetry"] = component_asymmetry(component, label_map)
    graph = contact_graph(components, label_map)
    values, counts = np.unique(arr, return_counts=True)
    dominant_raw_pixel_value = None
    if len(values) > 0:
        dominant_raw_pixel_value = int(values[int(np.argmax(counts))])

    features_by_label = {
        int(component["label_index"]): component_features(component, frame_shape=(height, width), graph=graph)
        for component in components
    }

    background_candidates_all = []
    for component in components:
        label_index = int(component["label_index"])
        score, reasons = score_background(
            features_by_label[label_index],
            dominant_color_match=int(component["raw_pixel_value"]) == dominant_raw_pixel_value,
        )
        background_candidates_all.append(
            ranked_candidate(
                component=component,
                role="background_hypothesis",
                score=score,
                reasons=reasons,
                features=features_by_label[label_index],
            )
        )
    background_candidates_all.sort(key=lambda row: (-float(row["score"]), -int(row["area_pixels"]), row["component_id"]))
    selected_background_ids = {
        str(row["component_id"])
        for row in background_candidates_all
        if ("dominant_raw_color" in row["reasons"] and int(row["area_pixels"]) >= max(16, int(0.04 * height * width)))
        or (
            float(row["score"]) >= 0.60
            and int(row["area_pixels"]) >= max(16, int(0.16 * height * width))
            and ("large_area" in row["reasons"] or "filled_region" in row["reasons"] or "full_frame_substrate" in row["reasons"])
        )
    }

    floor_candidates = []
    for component in components:
        if str(component["component_id"]) in selected_background_ids:
            continue
        label_index = int(component["label_index"])
        score, reasons = score_floor(features_by_label[label_index])
        floor_candidates.append(
            ranked_candidate(
                component=component,
                role="floor_hypothesis",
                score=score,
                reasons=reasons,
                features=features_by_label[label_index],
            )
        )
    floor_candidates.sort(key=lambda row: (-float(row["score"]), -int(row["area_pixels"]), row["component_id"]))
    bottom_floor_candidates = [
        row
        for row in floor_candidates
        if float(row["features"]["bottom_contact"]) >= 0.5
        and float(row["features"]["bbox_width_ratio"]) >= 0.50
        and float(row["features"]["area_ratio"]) <= 0.18
    ]
    if bottom_floor_candidates and float(bottom_floor_candidates[0]["score"]) >= 0.30:
        selected_floor = bottom_floor_candidates[0]
    else:
        selected_floor = None
    floor_label_index = None
    if selected_floor is not None:
        floor_label_index = next(
            int(component["label_index"])
            for component in components
            if component["component_id"] == selected_floor["component_id"]
        )

    wall_candidates = []
    for component in components:
        label_index = int(component["label_index"])
        if selected_floor is not None and component["component_id"] == selected_floor["component_id"]:
            continue
        if str(component["component_id"]) in selected_background_ids:
            continue
        score, reasons = score_wall(
            features_by_label[label_index],
            label_index=label_index,
            floor_label_index=floor_label_index,
            graph=graph,
        )
        wall_candidates.append(
            ranked_candidate(
                component=component,
                role="wall_hypothesis",
                score=score,
                reasons=reasons,
                features=features_by_label[label_index],
            )
        )
    wall_candidates.sort(key=lambda row: (-float(row["score"]), -int(row["area_pixels"]), row["component_id"]))

    selected_wall_ids = {
        str(row["component_id"])
        for row in wall_candidates
        if (
            float(row["score"]) >= 0.50
            and (
                "thin_edge_or_bar" in row["reasons"]
                or "boundary_like" in row["reasons"]
                or "contacts_floor_candidate" in row["reasons"]
            )
        )
        or ("thin_edge_or_bar" in row["reasons"] and "touches_outer_edge" in row["reasons"] and float(row["score"]) >= 0.38)
    }

    background_candidates = []
    for row in background_candidates_all:
        if selected_floor is not None and row["component_id"] == selected_floor["component_id"]:
            continue
        if row["component_id"] in selected_wall_ids:
            continue
        background_candidates.append(row)
    background_candidates.sort(key=lambda row: (-float(row["score"]), -int(row["area_pixels"]), row["component_id"]))
    selected_background_ids = {str(row["component_id"]) for row in background_candidates if row["component_id"] in selected_background_ids}
    if not selected_background_ids and background_candidates:
        top_background = background_candidates[0]
        if float(top_background["score"]) >= 0.44 and int(top_background["area_pixels"]) >= max(16, int(0.04 * height * width)):
            selected_background_ids.add(str(top_background["component_id"]))

    selected_ids = set(selected_background_ids) | set(selected_wall_ids)
    if selected_floor is not None:
        selected_ids.add(str(selected_floor["component_id"]))
    residual_component_hypotheses = []
    for component in components:
        if str(component["component_id"]) in selected_ids:
            continue
        label_index = int(component["label_index"])
        features = features_by_label[label_index]
        if int(component["area_pixels"]) < 1:
            continue
        residual_component_hypotheses.append(
            {
                **ranked_candidate(
                component=component,
                role="residual_component_hypothesis",
                score=round(
                    clamp01(
                        0.50 * clamp01(float(features["neighbor_contact_ratio"]) / 0.75)
                        + 0.30 * (1.0 - clamp01(float(features["area_ratio"]) / 0.20))
                        + 0.20 * (1.0 - float(features["border_contact"]))
                    ),
                    6,
                ),
                reasons=["residual_after_scene_static_surfaces"],
                features=features,
                ),
                "facing_direction_hypothesis": component["asymmetry"]["facing_direction_hypothesis"],
                "facing_direction_confidence": component["asymmetry"]["confidence"],
            }
        )
    residual_component_hypotheses.sort(key=lambda row: (-int(row["area_pixels"]), row["component_id"]))
    relationship_hypotheses = antisymmetric_relationship_hypotheses(residual_component_hypotheses)
    floor_pattern_hypotheses = checkerboard_floor_pattern_hypotheses(
        components,
        frame_shape=(height, width),
    )

    component_lookup = {str(component["component_id"]): int(component["label_index"]) for component in components}
    component_graph = {
        "schema": COMPONENT_GRAPH_SCHEMA,
        "source": "raw_frame_exact_color_components",
        "frame_shape": [int(height), int(width)],
        "component_count": len(components),
        "components": components,
        "contact_pair_count": graph["contact_pair_count"],
        "contacts": graph["contacts"],
    }
    packet = {
        "schema": SCENE_SURFACE_SCHEMA,
        "source": "raw_frame_exact_color_component_graph",
        "evidence_status": "hypothesis",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "role_group": "scene_static_surfaces",
        "dominant_raw_pixel_value": dominant_raw_pixel_value,
        "selection_policy": {
            "background": "reserve dominant/full-frame filled substrate candidates before wall scoring",
            "floor": "bottom-contact wide raw component after background reservation; reject otherwise",
            "walls": "thin/boundary component score after background and floor reservation; reject weak edge-only matches",
            "floor_patterns": "multi-component checkerboard floor candidates are grouped separately from single-component floor hypotheses",
            "residual_components": "unassigned raw-color components after scene static surface hypotheses",
        },
        "component_graph": component_graph,
        "floor_hypothesis": selected_floor,
        "floor_candidates": floor_candidates[:8],
        "wall_hypotheses": [row for row in wall_candidates if row["component_id"] in selected_wall_ids],
        "wall_candidates": wall_candidates[:10],
        "background_hypotheses": [row for row in background_candidates if row["component_id"] in selected_background_ids],
        "background_candidates": background_candidates[:10],
        "residual_component_hypotheses": residual_component_hypotheses,
        "residual_component_hypothesis_count": len(residual_component_hypotheses),
        "floor_pattern_hypotheses": floor_pattern_hypotheses,
        "scene_pattern_hypotheses": floor_pattern_hypotheses,
        "scene_pattern_hypothesis_count": len(floor_pattern_hypotheses),
        "antisymmetric_relationship_hypotheses": relationship_hypotheses,
        "antisymmetric_relationship_hypothesis_count": len(relationship_hypotheses),
    }
    return packet, label_map, component_lookup

# Shape/color control isolation.
CONTROL_ISOLATION_SCHEMA = "codexlab.contra_control_isolation.v2"
CONTROL_ISOLATION_SOURCE = "raw_frame_shape_trace_with_color_group_support"
BACKGROUND_SELECTION_POLICY = "dominant_raw_pixel_value"
PRIMARY_VIEW = "shape_traces"
SHAPE_TRACE_SOURCE = "raw_frame_orthogonal_shape_trace_graph"
COLOR_GROUP_ROLE = "supporting_boundary_evidence"
SAM_ADMITTED_PRIMARY_VIEW = "sam_admitted_object_masks"
SAM_ADMITTED_VIEW_SOURCE = "sam2_mask_with_fail_closed_review_admission"
SAM_ADMISSION_SCHEMA = "codexlab.sp80_sam_contra_object_view_admission.v1"

PALETTE_NAMES = {
    0: "white",
    1: "light_gray",
    2: "gray",
    3: "dark_gray",
    4: "near_black",
    5: "black",
    6: "magenta",
    7: "pink",
    8: "red",
    9: "blue",
    10: "cyan",
    11: "yellow",
    12: "orange",
    13: "maroon",
    14: "green",
    15: "purple",
}


def sha256_raw_grid(frame: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(frame, dtype=np.int16).tobytes()).hexdigest()


def component_neighbors(connectivity: int) -> tuple[tuple[int, int], ...]:
    if connectivity == 8:
        return (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    return ((-1, 0), (0, -1), (0, 1), (1, 0))


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def bbox_rect_mask(frame_shape: tuple[int, int], bbox: list[int]) -> np.ndarray:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    mask = np.zeros(frame_shape, dtype=np.uint8)
    mask[y0 : y1 + 1, x0 : x1 + 1] = 1
    return mask


def bbox_edge_touches(mask: np.ndarray, bbox: list[int]) -> dict[str, bool]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {"left": False, "right": False, "top": False, "bottom": False}
    x0, y0, x1, y1 = [int(v) for v in bbox]
    return {
        "left": bool(np.any(xs == x0)),
        "right": bool(np.any(xs == x1)),
        "top": bool(np.any(ys == y0)),
        "bottom": bool(np.any(ys == y1)),
    }


def component_geometry(mask: np.ndarray) -> dict[str, Any]:
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return {
            "bbox_xyxy": None,
            "centroid_xy": None,
            "touches_edge": False,
            "edge_touch_count": 0,
            "edge_touches": {"left": False, "right": False, "top": False, "bottom": False},
            "width": 0,
            "height": 0,
        }
    ys, xs = np.nonzero(mask)
    height, width = mask.shape[:2]
    touches = {
        "left": bbox[0] == 0,
        "right": bbox[2] == width - 1,
        "top": bbox[1] == 0,
        "bottom": bbox[3] == height - 1,
    }
    return {
        "bbox_xyxy": bbox,
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "touches_edge": any(touches.values()),
        "edge_touch_count": int(sum(1 for value in touches.values() if value)),
        "edge_touches": touches,
        "width": int(bbox[2] - bbox[0] + 1),
        "height": int(bbox[3] - bbox[1] + 1),
    }


def geometry_hint(area_fraction: float, geometry: dict[str, Any]) -> str:
    if area_fraction >= 0.30:
        return "large_field"
    if int(geometry["height"]) == 1 or int(geometry["width"]) == 1:
        return "thin_line"
    if area_fraction <= 0.0015:
        return "single_or_tiny_pixel_group"
    if int(geometry.get("edge_touch_count", 0)) >= 2:
        return "edge_connected_group"
    return "local_pixel_group"


def connected_components(frame: np.ndarray, *, connectivity: int) -> list[dict[str, Any]]:
    arr = np.asarray(frame)
    height, width = arr.shape[:2]
    seen = np.zeros((height, width), dtype=bool)
    rows: list[dict[str, Any]] = []
    neighbors = component_neighbors(connectivity)
    component_counter: Counter[int] = Counter()
    for start_y in range(height):
        for start_x in range(width):
            if seen[start_y, start_x]:
                continue
            value = int(arr[start_y, start_x])
            component_counter[value] += 1
            stack = [(start_y, start_x)]
            seen[start_y, start_x] = True
            mask = np.zeros((height, width), dtype=np.uint8)
            while stack:
                y, x = stack.pop()
                mask[y, x] = 1
                for dy, dx in neighbors:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if seen[ny, nx] or int(arr[ny, nx]) != value:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            rows.append(
                {
                    "raw_pixel_value": value,
                    "component_index_for_value": int(component_counter[value] - 1),
                    "mask": mask,
                }
            )
    return rows


def connected_components_mask(mask: np.ndarray, *, connectivity: int) -> list[np.ndarray]:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape[:2]
    seen = np.zeros((height, width), dtype=bool)
    neighbors = component_neighbors(connectivity)
    components: list[np.ndarray] = []
    for start_y in range(height):
        for start_x in range(width):
            if seen[start_y, start_x] or not source[start_y, start_x]:
                continue
            stack = [(start_y, start_x)]
            seen[start_y, start_x] = True
            out = np.zeros((height, width), dtype=np.uint8)
            while stack:
                y, x = stack.pop()
                out[y, x] = 1
                for dy, dx in neighbors:
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if seen[ny, nx] or not source[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((ny, nx))
            components.append(out)
    return components


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape[:2]
    boundary = np.zeros((height, width), dtype=np.uint8)
    ys, xs = np.nonzero(source)
    for y, x in zip(ys, xs):
        for dy, dx in component_neighbors(4):
            ny = int(y) + dy
            nx = int(x) + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width or not source[ny, nx]:
                boundary[y, x] = 1
                break
    return boundary


def orthogonal_perimeter_edges(mask: np.ndarray) -> int:
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape[:2]
    total = 0
    ys, xs = np.nonzero(source)
    for y, x in zip(ys, xs):
        for dy, dx in component_neighbors(4):
            ny = int(y) + dy
            nx = int(x) + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width or not source[ny, nx]:
                total += 1
    return int(total)


def orthogonal_touch_count(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    count = 0
    count += int(np.count_nonzero(a[:-1, :] & b[1:, :]))
    count += int(np.count_nonzero(a[1:, :] & b[:-1, :]))
    count += int(np.count_nonzero(a[:, :-1] & b[:, 1:]))
    count += int(np.count_nonzero(a[:, 1:] & b[:, :-1]))
    return count


def value_histogram(frame: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    selected = np.asarray(frame)[np.asarray(mask, dtype=bool)]
    values, counts = np.unique(selected.astype(int), return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def palette_histogram(frame: np.ndarray, mask: np.ndarray) -> list[dict[str, Any]]:
    hist = value_histogram(frame, mask)
    return [
        {
            "raw_pixel_value": int(value),
            "palette_name": PALETTE_NAMES.get(int(value), f"value_{value}"),
            "pixel_count": int(count),
        }
        for value, count in sorted((int(k), int(v)) for k, v in hist.items())
    ]


def component_sort_key(component: dict[str, Any]) -> tuple[int, int, int, str]:
    bbox = component.get("bbox_xyxy") or [0, 0, 0, 0]
    return (
        int(component.get("raw_pixel_value", 0)),
        int(bbox[1]),
        int(bbox[0]),
        str(component.get("component_id", "")),
    )


def shape_sort_key(shape: dict[str, Any]) -> tuple[int, int, str]:
    bbox = shape.get("bbox_xyxy") or [0, 0, 0, 0]
    return (int(bbox[1]), int(bbox[0]), str(shape.get("shape_id", "")))


def sam_object_sort_key(obj: dict[str, Any]) -> tuple[int, int, str]:
    bbox = obj.get("bbox_xyxy") or [0, 0, 0, 0]
    return (int(bbox[1]), int(bbox[0]), str(obj.get("sam_object_id", obj.get("hypothesis_id", ""))))


def build_admitted_sam_object_rows(admitted_sam_objects: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if admitted_sam_objects is None:
        return []
    rows: list[dict[str, Any]] = []
    for idx, source in enumerate(admitted_sam_objects):
        if not isinstance(source, dict):
            raise ValueError(f"admitted SAM object {idx} is not an object")
        if source.get("schema") != SAM_ADMISSION_SCHEMA:
            raise ValueError(f"admitted SAM object {idx} has bad schema {source.get('schema')!r}")
        if source.get("admitted_for_contra_object_view") is not True:
            raise ValueError(f"admitted SAM object {idx} is not admitted for Contra object view")
        for flag in ["admitted_for_control_truth", "admitted_for_goal_condition", "admitted_for_action_choice"]:
            if source.get(flag) is not False:
                raise ValueError(f"admitted SAM object {idx} has unsafe {flag}")
        row = {
            "sam_object_id": source.get("sam_object_id"),
            "hypothesis_id": source.get("hypothesis_id"),
            "source": SAM_ADMITTED_VIEW_SOURCE,
            "mask_source": source.get("mask_source"),
            "admission_status": source.get("admission_status"),
            "admission_scope": source.get("admission_scope"),
            "primary_view_unit": True,
            "shape_trace_role": "raw_shape_trace_audit_support",
            "raw_shape_trace_support_required": True,
            "area_pixels": source.get("area_pixels"),
            "bbox_xyxy": source.get("bbox_xyxy"),
            "centroid_xy": source.get("centroid_xy"),
            "width": source.get("width"),
            "height": source.get("height"),
            "source_pixel_mask": source.get("source_pixel_mask"),
            "source_hypothesis_jsonl": source.get("source_hypothesis_jsonl"),
            "source_hypothesis_line_no": source.get("source_hypothesis_line_no"),
            "source_hypothesis_json_sha256": source.get("source_hypothesis_json_sha256"),
            "source_acceptance_jsonl": source.get("source_acceptance_jsonl"),
            "source_acceptance_line_no": source.get("source_acceptance_line_no"),
            "source_acceptance_json_sha256": source.get("source_acceptance_json_sha256"),
            "accepted_controlled_match": source.get("accepted_controlled_match"),
            "accepted_semantic_label": source.get("accepted_semantic_label"),
            "accepted_source_tag": source.get("accepted_source_tag"),
            "review_decision": source.get("review_decision"),
            "review_decision_recorded": source.get("review_decision_recorded"),
            "admitted_for_contra_object_view": True,
            "admitted_for_control_truth": False,
            "admitted_for_goal_condition": False,
            "admitted_for_action_choice": False,
            "route_or_replay_used": False,
            "human_trace_used": False,
            "coordinate_script_used": False,
        }
        rows.append(row)
    rows.sort(key=sam_object_sort_key)
    return rows


def build_color_group_rows(
    arr: np.ndarray,
    *,
    phase: str,
    case_id: str,
    connectivity: int,
    min_area_pixels: int,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    total_pixels = int(arr.size)
    kept_by_value: Counter[int] = Counter()
    rows: list[dict[str, Any]] = []
    masks_by_id: dict[str, np.ndarray] = {}
    case = safe_id(str(case_id))
    phase_id = safe_id(str(phase))
    for raw_component in connected_components(arr, connectivity=connectivity):
        value = int(raw_component["raw_pixel_value"])
        mask = raw_component["mask"]
        area = int(np.count_nonzero(mask))
        if area < int(min_area_pixels):
            continue
        component_idx = kept_by_value[value]
        kept_by_value[value] += 1
        component_id = f"cg_{phase_id}_{case}_v{value:02d}_g{component_idx:03d}"
        area_fraction = float(area / max(1, total_pixels))
        geometry = component_geometry(mask)
        row = {
            "component_id": component_id,
            "component_source": "raw_frame_exact_value_connected_component",
            "control_view_role": COLOR_GROUP_ROLE,
            "primary_view_unit": False,
            "raw_pixel_value": value,
            "palette_name": PALETTE_NAMES.get(value, f"value_{value}"),
            "area_pixels": area,
            "area_fraction": area_fraction,
            "bbox_xyxy": geometry["bbox_xyxy"],
            "centroid_xy": geometry["centroid_xy"],
            "touches_edge": geometry["touches_edge"],
            "edge_touch_count": geometry["edge_touch_count"],
            "edge_touches": geometry["edge_touches"],
            "width": geometry["width"],
            "height": geometry["height"],
            "geometry_hint": geometry_hint(area_fraction, geometry),
            "semantic_label": "unknown",
            "semantic_label_status": "not_semantic_label",
            "route_or_replay_used": False,
            "human_trace_used": False,
            "coordinate_script_used": False,
        }
        rows.append(row)
        masks_by_id[component_id] = mask
    rows.sort(key=component_sort_key)
    return rows, masks_by_id


def select_background_value(
    color_groups: list[dict[str, Any]],
    *,
    background_raw_pixel_value: int | None,
) -> tuple[int, dict[str, Any]]:
    if background_raw_pixel_value is not None:
        value = int(background_raw_pixel_value)
        matching = [row for row in color_groups if int(row["raw_pixel_value"]) == value]
        return value, {
            "policy": "provided_by_caller",
            "raw_pixel_value": value,
            "palette_name": PALETTE_NAMES.get(value, f"value_{value}"),
            "matched_component_count": len(matching),
            "matched_pixel_count": int(sum(int(row["area_pixels"]) for row in matching)),
        }
    if not color_groups:
        return -1, {
            "policy": BACKGROUND_SELECTION_POLICY,
            "raw_pixel_value": None,
            "palette_name": None,
            "matched_component_count": 0,
            "matched_pixel_count": 0,
        }
    pixels_by_value: Counter[int] = Counter()
    for row in color_groups:
        pixels_by_value[int(row["raw_pixel_value"])] += int(row["area_pixels"])
    value = max(pixels_by_value, key=lambda key: (int(pixels_by_value[key]), -int(key)))
    matching = [row for row in color_groups if int(row["raw_pixel_value"]) == value]
    best = max(matching, key=lambda row: int(row["area_pixels"]))
    return value, {
        "policy": BACKGROUND_SELECTION_POLICY,
        "raw_pixel_value": value,
        "palette_name": PALETTE_NAMES.get(value, f"value_{value}"),
        "selected_component_id": best["component_id"],
        "selected_component_area_pixels": int(best["area_pixels"]),
        "matched_component_count": len(matching),
        "matched_pixel_count": int(sum(int(row["area_pixels"]) for row in matching)),
    }


def shape_trace_exclusion_reason(row: dict[str, Any], frame_shape: tuple[int, int]) -> str | None:
    bbox = row.get("bbox_xyxy")
    if bbox is None:
        return "empty_component"
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    full_width_or_height = int(row["width"]) >= frame_w or int(row["height"]) >= frame_h
    if (
        bool(row.get("touches_edge"))
        and float(row.get("area_fraction", 0.0)) >= 0.01
        and (int(row.get("edge_touch_count", 0)) >= 2 or full_width_or_height)
    ):
        return "edge_frame_or_wall_context"
    return None


def mark_background_membership(
    color_groups: list[dict[str, Any]],
    *,
    background_value: int,
    frame_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    foreground: list[dict[str, Any]] = []
    background: list[dict[str, Any]] = []
    shape_trace_groups: list[dict[str, Any]] = []
    for row in color_groups:
        is_background = int(row["raw_pixel_value"]) == int(background_value)
        exclusion_reason = None if is_background else shape_trace_exclusion_reason(row, frame_shape)
        row["background_context"] = is_background
        row["shape_trace_exclusion_reason"] = exclusion_reason
        row["foreground_for_shape_trace"] = not is_background and exclusion_reason is None
        row["admitted_for_contra_view"] = not is_background
        row["admitted_for_goal_condition"] = False
        row["control_view_role"] = COLOR_GROUP_ROLE
        row["primary_view_unit"] = False
        if is_background:
            background.append(row)
        else:
            foreground.append(row)
            if exclusion_reason is None:
                shape_trace_groups.append(row)
    return foreground, background, shape_trace_groups


def build_shape_traces(
    arr: np.ndarray,
    *,
    phase: str,
    case_id: str,
    shape_trace_groups: list[dict[str, Any]],
    background_value: int,
    masks_by_id: dict[str, np.ndarray],
    shape_connectivity: int,
) -> list[dict[str, Any]]:
    frame_shape = tuple(int(v) for v in arr.shape[:2])
    foreground_union = np.zeros(frame_shape, dtype=np.uint8)
    for row in shape_trace_groups:
        foreground_union = np.maximum(foreground_union, masks_by_id[row["component_id"]].astype(np.uint8))
    shape_masks = connected_components_mask(foreground_union, connectivity=shape_connectivity)
    shapes: list[dict[str, Any]] = []
    case = safe_id(str(case_id))
    phase_id = safe_id(str(phase))
    total_pixels = int(arr.size)
    for shape_idx, shape_mask in enumerate(shape_masks):
        shape_id = f"shape_{phase_id}_{case}_{shape_idx:03d}"
        member_rows: list[dict[str, Any]] = []
        for row in shape_trace_groups:
            mask = masks_by_id[row["component_id"]]
            if bool(np.any((shape_mask > 0) & (mask > 0))):
                member_rows.append(row)
        member_rows.sort(key=component_sort_key)
        geometry = component_geometry(shape_mask)
        bbox = geometry["bbox_xyxy"]
        shape_boundary = boundary_mask(shape_mask)
        negative_space_regions: list[dict[str, Any]] = []
        if bbox is not None and background_value >= 0:
            bbox_mask = bbox_rect_mask(frame_shape, bbox)
            candidate = ((arr == int(background_value)) & (bbox_mask > 0)).astype(np.uint8)
            for inset_idx, negative_mask in enumerate(connected_components_mask(candidate, connectivity=4)):
                touch_count = orthogonal_touch_count(shape_mask, negative_mask)
                if touch_count <= 0:
                    continue
                edge_touches = bbox_edge_touches(negative_mask, bbox)
                neg_geometry = component_geometry(negative_mask)
                relation = "edge_open_negative_space" if any(edge_touches.values()) else "enclosed_hole"
                negative_space_regions.append(
                    {
                        "negative_space_id": f"neg_{shape_id}_{inset_idx:03d}",
                        "source": "background_component_inside_shape_bbox_touching_shape_orthogonally",
                        "raw_pixel_value": int(background_value),
                        "palette_name": PALETTE_NAMES.get(int(background_value), f"value_{background_value}"),
                        "area_pixels": int(np.count_nonzero(negative_mask)),
                        "bbox_xyxy": neg_geometry["bbox_xyxy"],
                        "centroid_xy": neg_geometry["centroid_xy"],
                        "width": neg_geometry["width"],
                        "height": neg_geometry["height"],
                        "touches_shape_orthogonally": True,
                        "shape_orthogonal_touch_edges": int(touch_count),
                        "shape_bbox_edge_touches": edge_touches,
                        "negative_space_relation": relation,
                    }
                )
        hist = value_histogram(arr, shape_mask)
        shape_area = int(np.count_nonzero(shape_mask))
        supporting_color_group_ids = [row["component_id"] for row in member_rows]
        supporting_raw_pixel_values = sorted({int(row["raw_pixel_value"]) for row in member_rows})
        shapes.append(
            {
                "shape_id": shape_id,
                "shape_trace_role": "primary_contra_view_unit",
                "primary_view_unit": True,
                "shape_source": SHAPE_TRACE_SOURCE,
                "shape_trace_definition": "4-connected foreground color-group union with background negative space preserved, not filled",
                "shape_trace_input_role": COLOR_GROUP_ROLE,
                "shape_connectivity": int(shape_connectivity),
                "area_pixels": shape_area,
                "area_fraction": float(shape_area / max(1, total_pixels)),
                "bbox_xyxy": bbox,
                "centroid_xy": geometry["centroid_xy"],
                "touches_edge": geometry["touches_edge"],
                "edge_touch_count": geometry["edge_touch_count"],
                "edge_touches": geometry["edge_touches"],
                "width": geometry["width"],
                "height": geometry["height"],
                "orthogonal_perimeter_edges": orthogonal_perimeter_edges(shape_mask),
                "boundary_pixel_count": int(np.count_nonzero(shape_boundary)),
                "raw_pixel_value_histogram": hist,
                "palette_histogram": palette_histogram(arr, shape_mask),
                "supporting_color_group_ids": supporting_color_group_ids,
                "supporting_raw_pixel_values": supporting_raw_pixel_values,
                "member_color_group_ids": supporting_color_group_ids,
                "member_raw_pixel_values": supporting_raw_pixel_values,
                "member_count": len(member_rows),
                "supporting_color_group_count": len(member_rows),
                "color_groups_are_supporting_evidence": True,
                "negative_space_preserved": True,
                "background_fill_used": False,
                "negative_space_count": len(negative_space_regions),
                "negative_space_regions": negative_space_regions,
                "semantic_label": "unknown",
                "semantic_label_status": "not_semantic_label",
                "admitted_for_contra_view": True,
                "admitted_for_goal_condition": False,
            }
        )
    shapes.sort(key=shape_sort_key)
    return shapes


def build_control_isolation_packet(
    frame: np.ndarray,
    *,
    game: str,
    phase: str,
    case_id: str,
    background_raw_pixel_value: int | None = None,
    connectivity: int = 4,
    min_area_pixels: int = 1,
    shape_connectivity: int = 4,
    admitted_sam_objects: list[dict[str, Any]] | None = None,
    sam_admission_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arr = np.asarray(frame, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError(f"control isolation expects a 2D raw frame, got shape {arr.shape}")
    frame_shape = [int(v) for v in arr.shape]
    pixel_values, pixel_counts = np.unique(arr.astype(int), return_counts=True)
    pixel_value_counts = {
        str(int(value)): int(count)
        for value, count in sorted(zip(pixel_values, pixel_counts), key=lambda item: int(item[0]))
    }
    color_groups, masks_by_id = build_color_group_rows(
        arr,
        phase=phase,
        case_id=case_id,
        connectivity=connectivity,
        min_area_pixels=min_area_pixels,
    )
    background_value, background_selection = select_background_value(
        color_groups,
        background_raw_pixel_value=background_raw_pixel_value,
    )
    frame_shape_tuple = tuple(int(v) for v in arr.shape[:2])
    foreground_groups, background_groups, shape_trace_groups = mark_background_membership(
        color_groups,
        background_value=background_value,
        frame_shape=frame_shape_tuple,
    )
    shape_traces = build_shape_traces(
        arr,
        phase=phase,
        case_id=case_id,
        shape_trace_groups=shape_trace_groups,
        background_value=background_value,
        masks_by_id=masks_by_id,
        shape_connectivity=shape_connectivity,
    )
    sam_objects = build_admitted_sam_object_rows(admitted_sam_objects)
    sam_is_primary = len(sam_objects) > 0
    packet = {
        "schema": CONTROL_ISOLATION_SCHEMA,
        "game": game,
        "phase": phase,
        "case_id": case_id,
        "source": CONTROL_ISOLATION_SOURCE,
        "evidence_status": "controlled_runtime_observation",
        "raw_pixels_are_observation_ground_truth": True,
        "frame_raw_grid_sha256": sha256_raw_grid(arr),
        "frame_shape": frame_shape,
        "connectivity": int(connectivity),
        "min_area_pixels": int(min_area_pixels),
        "primary_view": SAM_ADMITTED_PRIMARY_VIEW if sam_is_primary else PRIMARY_VIEW,
        "primary_view_source": SAM_ADMITTED_VIEW_SOURCE if sam_is_primary else SHAPE_TRACE_SOURCE,
        "primary_view_unit": "sam_admitted_object_mask" if sam_is_primary else "shape_trace",
        "shape_trace_source": SHAPE_TRACE_SOURCE,
        "shape_trace_count": len(shape_traces),
        "shape_traces": shape_traces,
        "shape_trace_contract": {
            "primary_control_view": not sam_is_primary,
            "color_groups_are_supporting_evidence": True,
            "background_negative_space_preserved": True,
            "background_fill_used": False,
            "semantic_labels_used": False,
        },
        "background_selection": background_selection,
        "background_raw_pixel_value": None if background_value < 0 else int(background_value),
        "background_palette_name": None
        if background_value < 0
        else PALETTE_NAMES.get(int(background_value), f"value_{background_value}"),
        "pixel_value_counts": pixel_value_counts,
        "color_groups_role": COLOR_GROUP_ROLE,
        "color_group_count": len(color_groups),
        "color_groups": color_groups,
        "background_component_count": len(background_groups),
        "background_component_ids": [row["component_id"] for row in background_groups],
        "foreground_component_count": len(foreground_groups),
        "foreground_component_ids": [row["component_id"] for row in foreground_groups],
        "shape_trace_component_count": len(shape_trace_groups),
        "shape_trace_component_ids": [row["component_id"] for row in shape_trace_groups],
        "component_count": len(foreground_groups),
        "components": foreground_groups,
        "closed_shape_trace_count": len(shape_traces),
        "closed_shape_traces": shape_traces,
        "control_truth_scope": "raw_shape_trace_view_structure_only_not_semantic_goal",
        "sam_role": "primary_object_view_only_after_fail_closed_admission" if sam_is_primary else "hypothesis_or_refinement_only_must_match_raw_shape_trace_packet",
        "sam_status": "admitted_for_contra_object_view" if sam_is_primary else "not_run",
        "controlled_labels_used": False,
        "engine_tags_used": False,
        "semantic_labels_used": sam_is_primary,
        "admitted_for_contra_view": True,
        "admitted_for_goal_condition": False,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
    }
    if sam_is_primary:
        packet.update(
            {
                "sam_admission_policy": "fail_closed_reviewed_acceptance_then_contra_object_view_admission",
                "sam_admission_manifest": sam_admission_manifest,
                "admitted_sam_object_count": len(sam_objects),
                "admitted_sam_objects": sam_objects,
                "admitted_for_control_truth": False,
                "admitted_for_action_choice": False,
                "control_truth_scope": "sam_masks_are_contra_object_view_only_not_control_truth_or_goal_truth",
            }
        )
        packet["shape_trace_contract"]["raw_shape_traces_remain_audit_evidence"] = True
    return packet
