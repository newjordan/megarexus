#!/usr/bin/env python3
"""Modular scoring for SP80 SAM mask hypotheses.

SAM rows are hypotheses. This scorer can confirm overlap against controlled
engine masks, but it does not promote SAM output into controller truth.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCORE_SCHEMA = "codexlab.sam_hypothesis_score.v1"
SCORING_VERSION = "sam_scoring_v0.1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def centroid_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = np.asarray(a, dtype=bool)
    b_bool = np.asarray(b, dtype=bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    return float(inter / union) if union else 0.0


def geometry_hint(mask: np.ndarray, bbox: list[int] | None) -> str:
    if bbox is None:
        return "empty"
    x0, y0, x1, y1 = bbox
    h, w = mask.shape
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    area = int(np.count_nonzero(mask))
    touches_edge = x0 == 0 or y0 == 0 or x1 == w - 1 or y1 == h - 1
    if touches_edge and area > 0.18 * h * w:
        return "boundary_or_large_field"
    if bw > 0.55 * w and bh < 0.16 * h:
        return "horizontal_boundary_candidate"
    if bh > 0.55 * h and bw < 0.16 * w:
        return "vertical_boundary_candidate"
    if area < 0.035 * h * w:
        return "small_object_candidate"
    if area < 0.14 * h * w:
        return "object_actor_or_goal_candidate"
    return "region_or_mass_candidate"


def raw_pixel_counts(frame: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    values = np.asarray(frame)[np.asarray(mask, dtype=bool)]
    counts = Counter(int(v) for v in values.tolist())
    return {str(k): int(v) for k, v in sorted(counts.items())}


def color_coherence(frame: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    counts = raw_pixel_counts(frame, mask)
    total = sum(counts.values())
    if total == 0:
        return {
            "dominant_raw_pixel_value": None,
            "dominant_raw_pixel_fraction": 0.0,
            "masked_raw_pixel_value_counts": counts,
        }
    dominant_value, dominant_count = max(counts.items(), key=lambda item: item[1])
    return {
        "dominant_raw_pixel_value": int(dominant_value),
        "dominant_raw_pixel_fraction": float(dominant_count / total),
        "masked_raw_pixel_value_counts": counts,
    }


def load_controlled_mask(label: dict[str, Any]) -> np.ndarray:
    npy_path = ((label.get("pixel_mask") or {}).get("npy") or {}).get("path")
    if not npy_path:
        raise ValueError(f"controlled label {label.get('object_id')} missing mask path")
    return np.load(resolve_repo_path(npy_path), allow_pickle=False).astype(bool)


def best_overlap(mask: np.ndarray, controlled_labels: list[dict[str, Any]], *, semantic_label: str | None = None) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for label in controlled_labels:
        if semantic_label is not None and label.get("semantic_label") != semantic_label:
            continue
        controlled_mask = load_controlled_mask(label)
        iou = mask_iou(mask, controlled_mask)
        area = int(np.logical_and(mask.astype(bool), controlled_mask).sum())
        candidate = {
            "object_id": label.get("object_id"),
            "semantic_label": label.get("semantic_label"),
            "source_tag": label.get("source_tag"),
            "iou": iou,
            "intersection_pixels": area,
        }
        if best is None or iou > float(best.get("iou") or 0.0):
            best = candidate
    return best


def score_sam_hypothesis(
    *,
    mask: np.ndarray,
    raw_frame: np.ndarray,
    controlled_labels: list[dict[str, Any]],
    min_area_fraction: float,
    max_area_fraction: float,
    confirmed_iou_threshold: float,
) -> dict[str, Any]:
    mask_bool = np.asarray(mask, dtype=bool)
    h, w = mask_bool.shape
    total_pixels = max(1, h * w)
    area = int(mask_bool.sum())
    area_fraction = float(area / total_pixels)
    bbox = bbox_from_mask(mask_bool)
    centroid = centroid_from_mask(mask_bool)
    touches_edge = False if bbox is None else bbox[0] == 0 or bbox[1] == 0 or bbox[2] == w - 1 or bbox[3] == h - 1
    geom = geometry_hint(mask_bool, bbox)
    color = color_coherence(raw_frame, mask_bool)
    best_controlled = best_overlap(mask_bool, controlled_labels)
    best_bucket = best_overlap(mask_bool, controlled_labels, semantic_label="bucket")
    best_iou = float((best_controlled or {}).get("iou") or 0.0)
    best_bucket_iou = float((best_bucket or {}).get("iou") or 0.0)

    components = [
        {
            "name": "mask_in_frame_bounds",
            "score": 1.0,
            "passed": mask_bool.shape == np.asarray(raw_frame).shape[:2],
            "weight": 0.15,
        },
        {
            "name": "nonzero_area",
            "score": 1.0 if area > 0 else 0.0,
            "passed": area > 0,
            "weight": 0.15,
        },
        {
            "name": "area_fraction_range",
            "score": 1.0 if min_area_fraction <= area_fraction <= max_area_fraction else 0.0,
            "passed": min_area_fraction <= area_fraction <= max_area_fraction,
            "weight": 0.15,
            "min_area_fraction": min_area_fraction,
            "max_area_fraction": max_area_fraction,
            "observed_area_fraction": area_fraction,
        },
        {
            "name": "controlled_mask_overlap",
            "score": best_iou,
            "passed": best_iou >= confirmed_iou_threshold,
            "weight": 0.45,
            "confirmed_iou_threshold": confirmed_iou_threshold,
            "best_match": best_controlled,
        },
        {
            "name": "raw_pixel_color_coherence",
            "score": float(color["dominant_raw_pixel_fraction"]),
            "passed": float(color["dominant_raw_pixel_fraction"]) >= 0.5,
            "weight": 0.10,
        },
    ]
    weighted = sum(float(c["score"]) * float(c["weight"]) for c in components)
    weight_sum = sum(float(c["weight"]) for c in components)
    confirmed = all(bool(c["passed"]) for c in components[:3]) and best_iou >= confirmed_iou_threshold
    if not all(bool(c["passed"]) for c in components[:3]):
        confirmation_status = "invalid_mask_geometry"
    elif confirmed:
        confirmation_status = "confirmed_against_controlled_mask"
    else:
        confirmation_status = "hypothesis_unverified"

    return {
        "schema": SCORE_SCHEMA,
        "scoring_version": SCORING_VERSION,
        "confirmation_status": confirmation_status,
        "score_status": "confirmed" if confirmed else "unverified_or_invalid",
        "overall_score": float(weighted / weight_sum) if weight_sum else 0.0,
        "area_pixels": area,
        "area_fraction": area_fraction,
        "bbox_xyxy": bbox,
        "centroid_xy": centroid,
        "touches_edge": touches_edge,
        "geometry_hint": geom,
        "best_controlled_match": best_controlled,
        "best_bucket_match": best_bucket,
        "best_bucket_iou": best_bucket_iou,
        "bucket_like_by_controlled_overlap": best_bucket_iou >= confirmed_iou_threshold,
        **color,
        "components": components,
    }
