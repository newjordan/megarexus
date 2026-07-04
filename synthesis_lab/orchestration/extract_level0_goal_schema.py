#!/usr/bin/env python3
"""Extract static goal-schema hypotheses from recorded level-up animations.

This is a review tool, not a controller. It uses captured policy-frame archives
to identify components that are visually filled during a level completion
animation, then writes a frames-first packet for human review.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from codexlab.contra_head.vision import (  # noqa: E402
    analyze_scene_surface_hypotheses,
    raw_pixel_components_with_labels,
)


SCHEMA = "synthesis_lab.megarexus.level0_goal_schema_hypotheses.v1"
WINDOW_SCHEMA = "synthesis_lab.megarexus.level_completion_window_hypothesis.v1"
BOUNDARY = (
    "Static hindsight review from recorded completion windows. Not control truth, "
    "not solve evidence, not CNN mastery, and not an admitted live objective."
)
TRUTH_FLAGS = {
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
}

PALETTE = np.array(
    [
        [8, 10, 14],
        [230, 236, 244],
        [79, 124, 255],
        [255, 205, 84],
        [83, 215, 160],
        [246, 103, 103],
        [173, 123, 255],
        [255, 143, 65],
        [82, 190, 229],
        [216, 88, 167],
        [155, 170, 185],
        [40, 49, 66],
        [147, 222, 91],
        [252, 252, 164],
        [178, 108, 74],
        [119, 232, 224],
    ],
    dtype=np.uint8,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(grid, dtype=np.int16).tobytes()).hexdigest()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def bbox_for_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def component_mask(label_map: np.ndarray, component: dict[str, Any]) -> np.ndarray:
    return label_map == int(component["label_index"])


def value_counts(arr: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def dominant_value(values: np.ndarray) -> tuple[int | None, float]:
    if values.size == 0:
        return None, 0.0
    counts = Counter(int(v) for v in values.tolist())
    value, count = counts.most_common(1)[0]
    return int(value), float(count) / float(max(1, values.size))


def render_grid(grid: np.ndarray, path: Path, *, scale: int = 8) -> None:
    arr = np.asarray(grid, dtype=np.int16)
    arr = np.clip(arr, 0, len(PALETTE) - 1)
    image = Image.fromarray(PALETTE[arr], mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_component_boxes(
    grid: np.ndarray,
    components: list[dict[str, Any]],
    path: Path,
    *,
    target_ids: set[str],
    key_ids: set[str],
    source_ids: set[str],
    scale: int = 8,
) -> None:
    arr = np.clip(np.asarray(grid, dtype=np.int16), 0, len(PALETTE) - 1)
    image = Image.fromarray(PALETTE[arr], mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for component in components:
        cid = str(component["component_id"])
        if cid in target_ids:
            color = "#ff2f4f"
            prefix = "T"
        elif cid in key_ids:
            color = "#ffd84a"
            prefix = "K"
        elif cid in source_ids:
            color = "#38d9ff"
            prefix = "S"
        else:
            continue
        x0, y0, x1, y1 = [int(v) * scale for v in component["bbox_xyxy"]]
        draw.rectangle([x0, y0, x1 + scale - 1, y1 + scale - 1], outline=color, width=3)
        draw.text((x0 + 3, y0 + 3), f"{prefix}:{cid.split('_')[0]}", fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def draw_delta_overlay(
    base: np.ndarray,
    scoring_frames: np.ndarray,
    path: Path,
    *,
    target_masks: list[np.ndarray],
    key_masks: list[np.ndarray],
    source_value_masks: list[np.ndarray],
    scale: int = 8,
) -> None:
    arr = np.clip(np.asarray(base, dtype=np.int16), 0, len(PALETTE) - 1)
    rgb = PALETTE[arr].astype(np.float32)
    scoring_any = np.any(scoring_frames != base[None, :, :], axis=0)
    rgb[scoring_any] = 0.45 * rgb[scoring_any] + 0.55 * np.array([255, 255, 255], dtype=np.float32)
    for mask in source_value_masks:
        rgb[mask] = 0.30 * rgb[mask] + 0.70 * np.array([56, 217, 255], dtype=np.float32)
    for mask in key_masks:
        rgb[mask] = 0.30 * rgb[mask] + 0.70 * np.array([255, 216, 74], dtype=np.float32)
    for mask in target_masks:
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * np.array([255, 47, 79], dtype=np.float32)
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_contact_sheet(frames: np.ndarray, path: Path, *, scale: int = 4, columns: int = 5) -> None:
    frames = np.asarray(frames, dtype=np.int16)
    if frames.ndim != 3:
        raise ValueError(f"expected frame stack, got shape {frames.shape}")
    height, width = frames.shape[1:]
    tile_w = width * scale
    label_h = 18
    tile_h = height * scale + label_h
    rows = int(np.ceil(len(frames) / float(columns)))
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), "#111318")
    draw = ImageDraw.Draw(sheet)
    prev = None
    for idx, frame in enumerate(frames):
        r = idx // columns
        c = idx % columns
        x = c * tile_w
        y = r * tile_h
        arr = np.clip(frame, 0, len(PALETTE) - 1)
        image = Image.fromarray(PALETTE[arr], mode="RGB")
        image = image.resize((tile_w, height * scale), Image.Resampling.NEAREST)
        sheet.paste(image, (x, y + label_h))
        changed = 0 if prev is None else int(np.sum(frame != prev))
        draw.text((x + 4, y + 3), f"f{idx:02d} chg={changed}", fill="#e8edf4")
        prev = frame
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def parse_episode(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        path_obj = Path(value)
        label = path_obj.parent.name or path_obj.name
        path = value
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("episode label cannot be empty")
    return label, Path(path)


def load_episode(episode_dir: Path) -> list[dict[str, Any]]:
    meta_path = episode_dir / "policy_frames_meta.json"
    npz_path = episode_dir / "policy_frames.npz"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames_npz = np.load(npz_path, allow_pickle=False)
    rows: list[dict[str, Any]] = []
    for row in meta:
        call = int(row["call"])
        key = f"call{call:04d}"
        frames = np.asarray(frames_npz[key], dtype=np.int16)
        rows.append({**row, "call_key": key, "frames": frames, "grid": frames[-1]})
    return rows


def find_level_windows(rows: list[dict[str, Any]], *, from_level: int | None) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    for idx in range(1, len(rows)):
        prev_level = int(rows[idx - 1].get("levels") or 0)
        cur_level = int(rows[idx].get("levels") or 0)
        n_frames = int(rows[idx].get("n_frames") or len(rows[idx]["frames"]))
        if cur_level <= prev_level or n_frames <= 1:
            continue
        if from_level is not None and prev_level != from_level:
            continue
        windows.append((idx - 1, idx))
    return windows


def find_scoring_end(frames: np.ndarray) -> tuple[int, dict[str, Any]]:
    if len(frames) <= 1:
        return len(frames), {"reset_frame_index": None, "frame_to_frame_changed_pixels": []}
    diffs = [int(np.sum(frames[idx] != frames[idx - 1])) for idx in range(1, len(frames))]
    prior_nonzero = [d for d in diffs[:-1] if d > 0]
    baseline = float(np.median(prior_nonzero)) if prior_nonzero else float(np.median([d for d in diffs if d > 0] or [0]))
    threshold = max(512.0, baseline * 8.0, float(frames.shape[1] * frames.shape[2]) * 0.20)
    reset_idx = None
    for offset, changed in enumerate(diffs, start=1):
        if offset < 2:
            continue
        if float(changed) >= threshold:
            reset_idx = offset
            break
    scoring_end = int(reset_idx) if reset_idx is not None else len(frames)
    return scoring_end, {
        "reset_frame_index": int(reset_idx) if reset_idx is not None else None,
        "reset_detection_threshold_pixels": round(float(threshold), 3),
        "baseline_changed_pixels": round(float(baseline), 3),
        "frame_to_frame_changed_pixels": diffs,
    }


def component_change_rows(
    pre_frame: np.ndarray,
    scoring_frames: np.ndarray,
    components: list[dict[str, Any]],
    label_map: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    final_scoring = scoring_frames[-1]
    frame_count = len(scoring_frames)
    for component in components:
        mask = component_mask(label_map, component)
        area = int(mask.sum())
        if area <= 0:
            continue
        per_frame_changed = [float(np.mean(frame[mask] != pre_frame[mask])) for frame in scoring_frames]
        first_change = next((idx for idx, frac in enumerate(per_frame_changed) if frac >= 0.05), None)
        final_values = final_scoring[mask]
        dom_final, dom_final_frac = dominant_value(final_values)
        changed_final = final_values != pre_frame[mask]
        changed_final_pixels = int(np.sum(changed_final))
        changed_final_frac = float(changed_final_pixels) / float(max(1, area))
        first_change_norm = (
            float(first_change) / float(max(1, frame_count - 1)) if first_change is not None else None
        )
        rows.append(
            {
                "component_id": str(component["component_id"]),
                "label_index": int(component["label_index"]),
                "raw_pixel_value": int(component["raw_pixel_value"]),
                "area_pixels": area,
                "bbox_xyxy": [int(v) for v in component["bbox_xyxy"]],
                "center_xy": [round(float(v), 3) for v in component["center_xy"]],
                "first_change_frame": int(first_change) if first_change is not None else None,
                "first_change_norm": round(float(first_change_norm), 6)
                if first_change_norm is not None
                else None,
                "final_changed_pixels": changed_final_pixels,
                "final_changed_fraction": round(float(changed_final_frac), 6),
                "final_dominant_value": int(dom_final) if dom_final is not None else None,
                "final_dominant_fraction": round(float(dom_final_frac), 6),
                "per_frame_changed_fraction": [round(float(v), 6) for v in per_frame_changed],
            }
        )
    return rows


def select_targets(change_rows: list[dict[str, Any]], frame_count: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in change_rows:
        area = int(row["area_pixels"])
        first = row.get("first_change_frame")
        if first is None:
            continue
        changed = float(row["final_changed_fraction"])
        dom_frac = float(row["final_dominant_fraction"])
        first_norm = float(row["first_change_norm"] or 0.0)
        if area < 8 or area > 1000:
            continue
        if changed < 0.65 or dom_frac < 0.65:
            continue
        if first_norm < 0.45:
            continue
        score = clamp01(0.45 * changed + 0.35 * first_norm + 0.20 * dom_frac)
        if score < 0.65:
            continue
        selected.append(
            {
                **row,
                "candidate_role": "hindsight_target_fill_candidate",
                "score": round(score, 6),
                "evidence_status": "static_hindsight_hypothesis",
                "admission_status": "not_admitted",
                "reasons": [
                    "component_recolored_late_in_recorded_scoring_animation",
                    "component_has_single_dominant_final_fill_value",
                    "reset_frame_excluded_from_scoring_delta",
                ],
                **TRUTH_FLAGS,
            }
        )
    selected.sort(key=lambda r: (-float(r["score"]), r["component_id"]))
    return selected


def axis_relation(a_center: list[float], b_center: list[float]) -> dict[str, Any]:
    dx = float(b_center[0]) - float(a_center[0])
    dy = float(b_center[1]) - float(a_center[1])
    axis = "horizontal" if abs(dx) >= abs(dy) else "vertical"
    if axis == "horizontal":
        direction = "right" if dx > 0 else "left" if dx < 0 else "same"
    else:
        direction = "down" if dy > 0 else "up" if dy < 0 else "same"
    return {
        "vector_xy": [round(dx, 3), round(dy, 3)],
        "primary_axis": axis,
        "direction_from_candidate_to_target_centroid": direction,
        "distance_pixels": round(float(abs(dx) + abs(dy)), 3),
    }


def select_keys(
    change_rows: list[dict[str, Any]],
    target_candidates: list[dict[str, Any]],
    component_count_by_value: dict[int, int],
) -> list[dict[str, Any]]:
    if not target_candidates:
        return []
    target_center = [
        float(np.mean([float(row["center_xy"][0]) for row in target_candidates])),
        float(np.mean([float(row["center_xy"][1]) for row in target_candidates])),
    ]
    target_ids = {str(row["component_id"]) for row in target_candidates}
    candidates: list[dict[str, Any]] = []
    for row in change_rows:
        cid = str(row["component_id"])
        if cid in target_ids:
            continue
        area = int(row["area_pixels"])
        if area < 4 or area > 1200:
            continue
        raw_value = int(row["raw_pixel_value"])
        unique_value = component_count_by_value.get(raw_value, 0) == 1
        first = row.get("first_change_frame")
        early_change = first is not None and int(first) <= 1 and float(row["final_changed_fraction"]) >= 0.65
        if not early_change:
            continue
        relation = axis_relation([float(v) for v in row["center_xy"]], target_center)
        useful_relation = relation["direction_from_candidate_to_target_centroid"] != "same"
        score = (
            0.35 * float(unique_value)
            + 0.35 * float(early_change)
            + 0.20 * float(useful_relation)
            + 0.10 * clamp01(float(area) / 80.0)
        )
        if score < 0.45:
            continue
        reasons = []
        if unique_value:
            reasons.append("raw_value_has_single_component_in_pre_frame")
        if early_change:
            reasons.append("component_changes_at_start_of_scoring_animation")
        if useful_relation:
            reasons.append("component_is_spatially_opposite_target_centroid")
        candidates.append(
            {
                **row,
                "candidate_role": "key_block_candidate",
                "target_centroid_relation": relation,
                "score": round(clamp01(score), 6),
                "evidence_status": "static_hindsight_hypothesis",
                "admission_status": "not_admitted",
                "reasons": reasons,
                **TRUTH_FLAGS,
            }
        )
    candidates.sort(key=lambda r: (-float(r["score"]), r["component_id"]))
    return candidates[:5]


def select_sources(
    pre_frame: np.ndarray,
    scoring_frames: np.ndarray,
    components: list[dict[str, Any]],
    component_count_by_value: dict[int, int],
) -> list[dict[str, Any]]:
    pre_counts = value_counts(pre_frame)
    max_counts: dict[int, int] = defaultdict(int)
    for frame in scoring_frames:
        for value, count in value_counts(frame).items():
            max_counts[int(value)] = max(int(max_counts[int(value)]), int(count))
    growing_values = {
        value: int(max_counts.get(value, 0) - pre_counts.get(value, 0))
        for value in set(pre_counts) | set(max_counts)
    }
    candidates: list[dict[str, Any]] = []
    for component in components:
        raw_value = int(component["raw_pixel_value"])
        growth = int(growing_values.get(raw_value, 0))
        area = int(component["area_pixels"])
        if growth < max(16, area):
            continue
        unique_value = component_count_by_value.get(raw_value, 0) == 1
        score = clamp01(0.55 * clamp01(growth / 256.0) + 0.25 * float(unique_value) + 0.20 * clamp01(area / 64.0))
        candidates.append(
            {
                "component_id": str(component["component_id"]),
                "raw_pixel_value": raw_value,
                "area_pixels": area,
                "bbox_xyxy": [int(v) for v in component["bbox_xyxy"]],
                "center_xy": [round(float(v), 3) for v in component["center_xy"]],
                "value_growth_pixels_during_scoring": growth,
                "candidate_role": "paint_source_value_expansion_candidate",
                "score": round(score, 6),
                "evidence_status": "static_hindsight_hypothesis",
                "admission_status": "not_admitted",
                "reasons": [
                    "component_raw_value_expands_during_scoring_animation",
                    "source_is_pre_frame_component_with_expanding_value",
                ],
                **TRUTH_FLAGS,
            }
        )
    candidates.sort(key=lambda r: (-float(r["score"]), r["component_id"]))
    return candidates[:5]


def analyze_window(
    *,
    label: str,
    episode_dir: Path,
    rows: list[dict[str, Any]],
    pre_idx: int,
    window_idx: int,
    out_dir: Path,
) -> dict[str, Any]:
    pre_row = rows[pre_idx]
    window_row = rows[window_idx]
    pre_frame = np.asarray(pre_row["frames"][-1], dtype=np.int16)
    completion_frames = np.asarray(window_row["frames"], dtype=np.int16)
    scoring_end, reset_info = find_scoring_end(completion_frames)
    scoring_frames = completion_frames[:scoring_end]
    if len(scoring_frames) == 0:
        scoring_frames = completion_frames[:1]
    final_scoring = scoring_frames[-1]

    components, label_map = raw_pixel_components_with_labels(pre_frame)
    component_count_by_value = Counter(int(c["raw_pixel_value"]) for c in components)
    change_rows = component_change_rows(pre_frame, scoring_frames, components, label_map)
    target_candidates = select_targets(change_rows, len(scoring_frames))
    key_candidates = select_keys(change_rows, target_candidates, dict(component_count_by_value))
    source_candidates = select_sources(pre_frame, scoring_frames, components, dict(component_count_by_value))

    scene_packet, _, _ = analyze_scene_surface_hypotheses(pre_frame)
    antisym = scene_packet.get("antisymmetric_relationship_hypotheses") or []

    image_prefix = f"{label}_call{int(window_row['call']):04d}_l{int(pre_row.get('levels') or 0)}to{int(window_row.get('levels') or 0)}"
    pre_rel = Path("frames") / f"{image_prefix}_pre.png"
    scoring_rel = Path("frames") / f"{image_prefix}_scoring_sheet.png"
    component_rel = Path("frames") / f"{image_prefix}_component_roles.png"
    overlay_rel = Path("frames") / f"{image_prefix}_scoring_delta_overlay.png"
    final_rel = Path("frames") / f"{image_prefix}_scoring_final.png"
    render_grid(pre_frame, out_dir / pre_rel)
    render_grid(final_scoring, out_dir / final_rel)
    render_contact_sheet(scoring_frames, out_dir / scoring_rel)

    component_by_id = {str(c["component_id"]): c for c in components}
    target_ids = {str(row["component_id"]) for row in target_candidates}
    key_ids = {str(row["component_id"]) for row in key_candidates[:1]}
    source_ids = {str(row["component_id"]) for row in source_candidates[:1]}
    draw_component_boxes(
        pre_frame,
        components,
        out_dir / component_rel,
        target_ids=target_ids,
        key_ids=key_ids,
        source_ids=source_ids,
    )
    target_masks = [component_mask(label_map, component_by_id[cid]) for cid in target_ids if cid in component_by_id]
    key_masks = [component_mask(label_map, component_by_id[cid]) for cid in key_ids if cid in component_by_id]
    source_value_masks = []
    for row in source_candidates[:1]:
        source_value_masks.append(np.any(scoring_frames == int(row["raw_pixel_value"]), axis=0) & (pre_frame != int(row["raw_pixel_value"])))
    draw_delta_overlay(
        pre_frame,
        scoring_frames,
        out_dir / overlay_rel,
        target_masks=target_masks,
        key_masks=key_masks,
        source_value_masks=source_value_masks,
    )

    target_inventory = {
        "candidate_target_count": len(target_candidates),
        "candidate_target_components": [
            {
                "component_id": row["component_id"],
                "raw_pixel_value": row["raw_pixel_value"],
                "bbox_xyxy": row["bbox_xyxy"],
                "center_xy": row["center_xy"],
                "final_dominant_value": row["final_dominant_value"],
                "first_change_frame": row["first_change_frame"],
                "score": row["score"],
            }
            for row in target_candidates
        ],
        "derivation_rule": "components recolored late inside the recorded scoring frames after reset-frame exclusion",
        "derived_pours_or_streams": "not_extracted_here",
        "evidence_status": "static_hindsight_hypothesis",
        "admission_status": "not_admitted",
        **TRUTH_FLAGS,
    }
    source_inventory = {
        "candidate_source_count": len(source_candidates),
        "candidate_source_components": [
            {
                "component_id": row["component_id"],
                "raw_pixel_value": row["raw_pixel_value"],
                "bbox_xyxy": row["bbox_xyxy"],
                "center_xy": row["center_xy"],
                "value_growth_pixels_during_scoring": row["value_growth_pixels_during_scoring"],
                "score": row["score"],
            }
            for row in source_candidates
        ],
        "derivation_rule": "pre-frame component raw values whose pixel mass expands during scoring frames",
        "evidence_status": "static_hindsight_hypothesis",
        "admission_status": "not_admitted",
        **TRUTH_FLAGS,
    }
    key_inventory = {
        "candidate_key_count": len(key_candidates),
        "candidate_key_components": [
            {
                "component_id": row["component_id"],
                "raw_pixel_value": row["raw_pixel_value"],
                "bbox_xyxy": row["bbox_xyxy"],
                "center_xy": row["center_xy"],
                "first_change_frame": row["first_change_frame"],
                "target_centroid_relation": row["target_centroid_relation"],
                "score": row["score"],
            }
            for row in key_candidates
        ],
        "derivation_rule": "unique/salient pre-frame components with early scoring change and relation to target centroid",
        "evidence_status": "static_hindsight_hypothesis",
        "admission_status": "not_admitted",
        **TRUTH_FLAGS,
    }

    scoring_delta = np.any(scoring_frames != pre_frame[None, :, :], axis=0)
    return {
        "schema": WINDOW_SCHEMA,
        "episode_label": label,
        "episode_dir": str(episode_dir),
        "pre_call": int(pre_row["call"]),
        "completion_call": int(window_row["call"]),
        "from_level": int(pre_row.get("levels") or 0),
        "to_level": int(window_row.get("levels") or 0),
        "completion_frame_count": int(len(completion_frames)),
        "scoring_frame_count": int(len(scoring_frames)),
        "reset_detection": reset_info,
        "pre_frame_sha256": frame_hash(pre_frame),
        "scoring_final_frame_sha256": frame_hash(final_scoring),
        "pre_value_counts": value_counts(pre_frame),
        "scoring_final_value_counts": value_counts(final_scoring),
        "scoring_delta_pixels": int(scoring_delta.sum()),
        "scoring_delta_bbox_xyxy": bbox_for_mask(scoring_delta),
        "target_inventory_hypothesis": target_inventory,
        "key_block_hypothesis": key_inventory,
        "paint_source_hypothesis": source_inventory,
        "legacy_antisymmetric_relationship_count": len(antisym),
        "legacy_antisymmetric_relationship_top": antisym[:5],
        "component_change_rows": change_rows,
        "images": {
            "pre_frame": str(pre_rel),
            "scoring_contact_sheet": str(scoring_rel),
            "scoring_final_frame": str(final_rel),
            "scoring_delta_overlay": str(overlay_rel),
            "component_roles_overlay": str(component_rel),
        },
        "boundary": BOUNDARY,
        "evidence_status": "static_hindsight_hypothesis",
        "admission_status": "not_admitted",
        **TRUTH_FLAGS,
    }


def html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p class=\"empty\">None selected by this static heuristic.</p>"
    head = "".join(f"<th>{html.escape(col)}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, (dict, list)):
                text = json.dumps(value, sort_keys=True)
            else:
                text = "" if value is None else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_review_page(packet: dict[str, Any], out_dir: Path) -> None:
    windows = packet["windows"]
    sections = []
    for window in windows:
        images = window["images"]
        targets = window["target_inventory_hypothesis"]["candidate_target_components"]
        keys = window["key_block_hypothesis"]["candidate_key_components"]
        sources = window["paint_source_hypothesis"]["candidate_source_components"]
        sections.append(
            f"""
            <section>
              <h2>{html.escape(window['episode_label'])}: call {window['completion_call']} level {window['from_level']} -> {window['to_level']}</h2>
              <p class="boundary">{html.escape(window['boundary'])}</p>
              <div class="metrics">
                <div><strong>Scoring frames</strong><span>{window['scoring_frame_count']} / {window['completion_frame_count']}</span></div>
                <div><strong>Reset frame</strong><span>{window['reset_detection'].get('reset_frame_index')}</span></div>
                <div><strong>Target candidates</strong><span>{len(targets)}</span></div>
                <div><strong>Legacy antisym rels</strong><span>{window['legacy_antisymmetric_relationship_count']}</span></div>
              </div>
              <div class="deck">
                <figure><img src="{html.escape(images['pre_frame'])}" alt="raw pre frame"><figcaption>Raw pre-frame</figcaption></figure>
                <figure><img src="{html.escape(images['scoring_contact_sheet'])}" alt="scoring contact sheet"><figcaption>Scoring animation frames, reset excluded</figcaption></figure>
                <figure><img src="{html.escape(images['scoring_delta_overlay'])}" alt="delta overlay"><figcaption>Delta overlay: targets red, key yellow, source flow cyan</figcaption></figure>
                <figure><img src="{html.escape(images['component_roles_overlay'])}" alt="component role overlay"><figcaption>Component role candidates on pre-frame</figcaption></figure>
              </div>
              <h3>Target Inventory Candidates</h3>
              {html_table(targets, ['component_id', 'raw_pixel_value', 'bbox_xyxy', 'center_xy', 'final_dominant_value', 'first_change_frame', 'score'])}
              <h3>Key Block Candidates</h3>
              {html_table(keys, ['component_id', 'raw_pixel_value', 'bbox_xyxy', 'center_xy', 'first_change_frame', 'target_centroid_relation', 'score'])}
              <h3>Paint Source Candidates</h3>
              {html_table(sources, ['component_id', 'raw_pixel_value', 'bbox_xyxy', 'center_xy', 'value_growth_pixels_during_scoring', 'score'])}
            </section>
            """
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Level-0 Goal Schema Static Review</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #e8edf4; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ letter-spacing: 0; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin: 34px 0 8px; font-size: 22px; }}
    h3 {{ margin: 24px 0 8px; font-size: 17px; }}
    a {{ color: #8fc7ff; }}
    .boundary {{ color: #f4d278; max-width: 1000px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 10px; margin: 16px 0; }}
    .metrics div {{ border: 1px solid #2d3440; border-radius: 6px; padding: 10px; background: #171b22; }}
    .metrics strong {{ display: block; color: #9aa8b8; font-size: 12px; font-weight: 600; }}
    .metrics span {{ display: block; margin-top: 4px; font-size: 20px; }}
    .deck {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; align-items: start; }}
    figure {{ margin: 0; border: 1px solid #2d3440; border-radius: 6px; background: #171b22; padding: 10px; }}
    figure img {{ width: 100%; height: auto; image-rendering: pixelated; display: block; background: #080a0e; }}
    figcaption {{ margin-top: 8px; color: #b8c3cf; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; table-layout: fixed; font-size: 13px; margin-bottom: 16px; }}
    th, td {{ border: 1px solid #2d3440; padding: 7px 8px; vertical-align: top; word-break: break-word; }}
    th {{ background: #1d2430; color: #cbd6e2; text-align: left; }}
    td {{ background: #151920; }}
    .empty {{ color: #9aa8b8; }}
    .provenance {{ border-top: 1px solid #2d3440; margin-top: 38px; padding-top: 18px; color: #b8c3cf; }}
    code {{ color: #d7e6ff; }}
  </style>
</head>
<body>
<main>
  <h1>Level-0 Goal Schema Static Review</h1>
  <p class="boundary">{html.escape(packet['boundary'])}</p>
  <p>Raw packet: <a href="goal_schema_hypotheses.json">goal_schema_hypotheses.json</a>. Manifest: <a href="manifest.json">manifest.json</a>.</p>
  {''.join(sections)}
  <section class="provenance">
    <h2>Provenance</h2>
    <p>Command: <code>{html.escape(packet['command'])}</code></p>
    <p>Working directory: <code>{html.escape(packet['working_directory'])}</code></p>
    <p>Generated at: <code>{html.escape(packet['generated_at_utc'])}</code></p>
  </section>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", action="append", type=parse_episode, required=True, help="LABEL=policy_frames_dir")
    parser.add_argument("--from-level", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--condition", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    windows: list[dict[str, Any]] = []
    inputs = []
    for label, episode_dir in args.episode:
        rows = load_episode(episode_dir)
        npz_path = episode_dir / "policy_frames.npz"
        meta_path = episode_dir / "policy_frames_meta.json"
        inputs.append(
            {
                "label": label,
                "episode_dir": str(episode_dir),
                "policy_frames_npz_sha256": sha256_file(npz_path),
                "policy_frames_meta_sha256": sha256_file(meta_path),
            }
        )
        for pre_idx, window_idx in find_level_windows(rows, from_level=args.from_level):
            windows.append(
                analyze_window(
                    label=label,
                    episode_dir=episode_dir,
                    rows=rows,
                    pre_idx=pre_idx,
                    window_idx=window_idx,
                    out_dir=out_dir,
                )
            )

    packet = {
        "schema": SCHEMA,
        "run_label": out_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
        "command": " ".join(sys.argv),
        "source_script": str(Path(__file__)),
        "source_script_sha256": sha256_file(Path(__file__)),
        "condition_artifact": str(args.condition) if args.condition else None,
        "from_level": int(args.from_level),
        "input_policy_frame_archives": inputs,
        "window_count": len(windows),
        "windows": windows,
        "boundary": BOUNDARY,
        "evidence_status": "static_hindsight_hypothesis",
        "admission_status": "not_admitted",
        **TRUTH_FLAGS,
    }
    manifest = {
        "schema": "synthesis_lab.megarexus.goal_schema_extraction_manifest.v1",
        "run_label": out_dir.name,
        "generated_at_utc": packet["generated_at_utc"],
        "command": packet["command"],
        "working_directory": packet["working_directory"],
        "source_script": packet["source_script"],
        "source_script_sha256": packet["source_script_sha256"],
        "condition_artifact": packet["condition_artifact"],
        "input_policy_frame_archives": inputs,
        "outputs": {
            "packet": "goal_schema_hypotheses.json",
            "review_page": "index.html",
            "frames_dir": "frames",
        },
        "status": "completed_static_review",
        "boundary": BOUNDARY,
        **TRUTH_FLAGS,
    }
    write_json(out_dir / "goal_schema_hypotheses.json", packet)
    write_json(out_dir / "manifest.json", manifest)
    write_review_page(packet, out_dir)
    print(json.dumps({"out_dir": str(out_dir), "window_count": len(windows), "status": "completed_static_review"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
