#!/usr/bin/env python3
"""SAM2 hypothesis viewer for SP80/Liquid Pour object and boundary masks.

This runner keeps SAM output in a hypothesis lane. Raw frame pixels and the
controlled engine-sprite masks remain the confirmation surfaces.
"""

from __future__ import annotations

import argparse
import html
import json
import socket
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from sp80_base_controller import ROOT, SP80_PATH, rgb_frame, sha256_file, sha256_frame
from sp80_object_label_viewer import (
    detect_level_count,
    parse_levels,
    rel_for_html,
    relpath,
    resolve_repo_path,
    run_level as run_controlled_object_level,
    save_mask,
    write_json,
    write_jsonl,
)
from sp80_sam_scoring import SCORING_VERSION, mask_iou, score_sam_hypothesis


SAM_HYPOTHESIS_SCHEMA = "codexlab.sp80_sam_hypothesis.v1"
SAM_MANIFEST_SCHEMA = "codexlab.sp80_sam_hypothesis_viewer_manifest.v1"
SAM_SCORECARD_SCHEMA = "codexlab.sp80_sam_scorecard.v1"
DEFAULT_SAM2_MODEL = Path(
    "/srv/models-hdd/huggingface/hub/models--facebook--sam2.1-hiera-large/"
    "snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610"
)
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


@dataclass(frozen=True)
class SamCandidate:
    prompt_source: str
    prompt_source_object_id: str | None
    prompt_metadata: dict[str, Any]
    prompt_point_scaled_xy: tuple[int, int]
    prompt_point_display_xy: tuple[float, float]
    sam_iou_score: float
    scaled_bbox_xyxy: tuple[int, int, int, int]
    raw_mask: np.ndarray


def prompt_points(width: int, height: int, grid: int) -> list[tuple[int, int]]:
    xs = np.linspace(width / (grid + 1), width - width / (grid + 1), grid)
    ys = np.linspace(height / (grid + 1), height - height / (grid + 1), grid)
    return [(int(round(x)), int(round(y))) for y in ys for x in xs]


def raw_pixel_components(mask: np.ndarray) -> list[dict[str, Any]]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, Any]] = []
    height, width = mask.shape[:2]
    for start_y, start_x in zip(*np.nonzero(mask)):
        start_y = int(start_y)
        start_x = int(start_x)
        if seen[start_y, start_x]:
            continue
        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = y + dy
                nx = x + dx
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    continue
                if seen[ny, nx] or not bool(mask[ny, nx]):
                    continue
                seen[ny, nx] = True
                stack.append((ny, nx))
        ys = np.asarray([point[0] for point in points], dtype=np.float32)
        xs = np.asarray([point[1] for point in points], dtype=np.float32)
        center_x = float(xs.mean())
        center_y = float(ys.mean())
        nearest_idx = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
        prompt_y, prompt_x = points[nearest_idx]
        components.append(
            {
                "area_pixels": int(len(points)),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "center_xy": [center_x, center_y],
                "prompt_xy": [int(prompt_x), int(prompt_y)],
            }
        )
    components.sort(key=lambda row: (row["bbox_xyxy"][1], row["bbox_xyxy"][0], -row["area_pixels"]))
    return components


def raw_color_inventory(raw_frame: np.ndarray, *, min_component_area: int) -> dict[str, Any]:
    arr = np.asarray(raw_frame)
    values, counts = np.unique(arr.astype(int), return_counts=True)
    if len(values) == 0:
        background_value = None
    else:
        background_value = int(values[int(np.argmax(counts))])
    colors: list[dict[str, Any]] = []
    for value, count in sorted(zip(values, counts), key=lambda item: int(item[0])):
        raw_value = int(value)
        components = [
            {
                **component,
                "component_index_for_color": idx,
            }
            for idx, component in enumerate(raw_pixel_components(arr == raw_value))
            if int(component["area_pixels"]) >= int(min_component_area)
        ]
        colors.append(
            {
                "raw_pixel_value": raw_value,
                "palette_name": PALETTE_NAMES.get(raw_value, f"value_{raw_value}"),
                "pixel_count": int(count),
                "component_count": len(components),
                "dominant_background_candidate": background_value == raw_value,
                "components": components,
            }
        )
    return {
        "schema": "codexlab.raw_frame_color_inventory.v1",
        "source": "raw_frame_exact_pixel_values",
        "background_selection_policy": "dominant_raw_pixel_value",
        "background_raw_pixel_value": background_value,
        "min_component_area_pixels": int(min_component_area),
        "color_count": len(colors),
        "colors": colors,
    }


def sam_prompt_points(
    *,
    raw_frame: np.ndarray,
    width: int,
    height: int,
    grid: int,
    scale: int,
    prompt_sources: set[str],
    controlled_labels: list[dict[str, Any]],
    visual_hints: dict[str, Any] | None,
    raw_color_inventory_packet: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "grid" in prompt_sources:
        rows.extend(
            {
                "prompt_source": "grid",
                "prompt_source_object_id": None,
                "prompt_metadata": {},
                "x": x,
                "y": y,
            }
            for x, y in prompt_points(width, height, grid)
        )
    if "controlled_centers" in prompt_sources:
        for label in controlled_labels:
            center = label.get("display_pixel_center")
            if not isinstance(center, list) or len(center) != 2:
                continue
            x = int(round((float(center[0]) + 0.5) * scale))
            y = int(round((float(center[1]) + 0.5) * scale))
            rows.append(
                {
                    "prompt_source": "controlled_center",
                    "prompt_source_object_id": label.get("object_id"),
                    "prompt_metadata": {},
                    "x": max(0, min(width - 1, x)),
                    "y": max(0, min(height - 1, y)),
                }
            )
    if "raw_yellow_components" in prompt_sources:
        if visual_hints is None:
            raise RuntimeError("raw_yellow_components prompt source requires --visual-hints")
        hint = visual_hints.get("hint") or {}
        target_raw_pixel_value = 11
        component_mask = np.asarray(raw_frame) == target_raw_pixel_value
        for component_idx, component in enumerate(raw_pixel_components(component_mask)):
            prompt_x, prompt_y = component["prompt_xy"]
            x = int(round((float(prompt_x) + 0.5) * scale))
            y = int(round((float(prompt_y) + 0.5) * scale))
            rows.append(
                {
                    "prompt_source": "raw_yellow_components",
                    "prompt_source_object_id": f"raw_yellow_component_{component_idx:02d}",
                    "prompt_metadata": {
                        "source": "raw_frame_palette_value_component",
                        "human_hint_schema": visual_hints.get("schema"),
                        "target_semantic_label": hint.get("target_semantic_label"),
                        "color_name": hint.get("color_name") or "yellow",
                        "shape_description": hint.get("shape_description"),
                        "target_raw_pixel_value": target_raw_pixel_value,
                        "controlled_labels_used_for_prompt": False,
                        "text_prompt_used_by_sam2": False,
                        "sam2_prompt_type": "positive_point_from_raw_pixel_component",
                        "component_area_pixels": component["area_pixels"],
                        "component_bbox_xyxy": component["bbox_xyxy"],
                    },
                    "x": max(0, min(width - 1, x)),
                    "y": max(0, min(height - 1, y)),
                }
            )
    if "raw_color_components" in prompt_sources:
        if raw_color_inventory_packet is None:
            raise RuntimeError("raw_color_components prompt source requires raw color inventory")
        for color in raw_color_inventory_packet.get("colors") or []:
            raw_value = int(color["raw_pixel_value"])
            for component in color.get("components") or []:
                prompt_x, prompt_y = component["prompt_xy"]
                x = int(round((float(prompt_x) + 0.5) * scale))
                y = int(round((float(prompt_y) + 0.5) * scale))
                component_idx = int(component["component_index_for_color"])
                rows.append(
                    {
                        "prompt_source": "raw_color_components",
                        "prompt_source_object_id": f"raw_v{raw_value:02d}_component_{component_idx:03d}",
                        "prompt_metadata": {
                            "source": "raw_frame_color_inventory_component",
                            "raw_color_inventory_schema": raw_color_inventory_packet.get("schema"),
                            "raw_pixel_value": raw_value,
                            "palette_name": color.get("palette_name"),
                            "color_pixel_count": int(color.get("pixel_count") or 0),
                            "color_component_count": int(color.get("component_count") or 0),
                            "dominant_background_candidate": bool(color.get("dominant_background_candidate")),
                            "controlled_labels_used_for_prompt": False,
                            "text_prompt_used_by_sam2": False,
                            "sam2_prompt_type": "positive_point_from_raw_color_component",
                            "component_index_for_color": component_idx,
                            "component_area_pixels": int(component["area_pixels"]),
                            "component_bbox_xyxy": component["bbox_xyxy"],
                            "component_center_xy": component["center_xy"],
                        },
                        "x": max(0, min(width - 1, x)),
                        "y": max(0, min(height - 1, y)),
                    }
                )
    if "visual_hint_yellow_u" in prompt_sources:
        if visual_hints is None:
            raise RuntimeError("visual_hint_yellow_u prompt source requires --visual-hints")
        hint = visual_hints.get("hint") or {}
        target_label = str(hint.get("target_semantic_label") or "bucket")
        for label in controlled_labels:
            if label.get("semantic_label") != target_label:
                continue
            center = label.get("display_pixel_center")
            if not isinstance(center, list) or len(center) != 2:
                continue
            x = int(round((float(center[0]) + 0.5) * scale))
            y = int(round((float(center[1]) + 0.5) * scale))
            rows.append(
                {
                    "prompt_source": "visual_hint_yellow_u",
                    "prompt_source_object_id": label.get("object_id"),
                    "prompt_metadata": {
                        "source": "human_visual_hint_plus_controlled_bucket_center",
                        "human_hint_schema": visual_hints.get("schema"),
                        "target_semantic_label": target_label,
                        "color_name": hint.get("color_name"),
                        "shape_description": hint.get("shape_description"),
                        "controlled_labels_used_for_prompt": True,
                        "text_prompt_used_by_sam2": False,
                        "sam2_prompt_type": "positive_point",
                    },
                    "x": max(0, min(width - 1, x)),
                    "y": max(0, min(height - 1, y)),
                }
            )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None, int, int]] = set()
    for row in rows:
        key = (row["prompt_source"], row["prompt_source_object_id"], int(row["x"]), int(row["y"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def scaled_to_raw_mask(mask: np.ndarray, *, raw_shape: tuple[int, int]) -> np.ndarray:
    raw_h, raw_w = raw_shape
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    small = image.resize((raw_w, raw_h), Image.Resampling.NEAREST)
    return (np.asarray(small) > 0).astype(np.uint8)


def model_file_fingerprints(model_dir: Path, *, hash_model_weights: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"path": str(model_dir)}
    for name in ["config.json", "preprocessor_config.json", "processor_config.json"]:
        path = model_dir / name
        if path.is_file():
            out[name] = {"path": str(path), "sha256": sha256_file(path)}
    weight_path = model_dir / "model.safetensors"
    if weight_path.is_file():
        out["model.safetensors"] = {
            "path": str(weight_path),
            "sha256": sha256_file(weight_path) if hash_model_weights else "not_hashed",
            "hash_reason": "hash_model_weights flag disabled" if not hash_model_weights else "hashed",
        }
    return out


def load_sam2(model_dir: Path, device: str) -> tuple[Any, Any]:
    try:
        from transformers import Sam2Model, Sam2Processor
    except Exception as exc:  # pragma: no cover - environment-dependent guard
        raise RuntimeError(f"transformers SAM2 imports failed: {exc!r}") from exc
    if not model_dir.is_dir():
        raise RuntimeError(f"SAM2 model directory does not exist: {model_dir}")
    processor = Sam2Processor.from_pretrained(str(model_dir), local_files_only=True)
    model = Sam2Model.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
    return processor, model


def extract_sam2_candidates(
    *,
    raw_frame: np.ndarray,
    scale: int,
    processor: Any,
    model: Any,
    device: str,
    grid: int,
    prompt_sources: set[str],
    controlled_labels: list[dict[str, Any]],
    visual_hints: dict[str, Any] | None,
    raw_color_inventory_packet: dict[str, Any] | None,
    max_masks: int,
    min_area_fraction: float,
    max_area_fraction: float,
    dedupe_iou: float,
) -> list[SamCandidate]:
    scaled_rgb = rgb_frame(raw_frame, scale)
    image = Image.fromarray(scaled_rgb, mode="RGB")
    raw_shape = tuple(int(v) for v in raw_frame.shape[:2])
    total_raw = raw_shape[0] * raw_shape[1]
    candidates: list[SamCandidate] = []
    prompts = sam_prompt_points(
        raw_frame=raw_frame,
        width=image.width,
        height=image.height,
        grid=grid,
        scale=scale,
        prompt_sources=prompt_sources,
        controlled_labels=controlled_labels,
        visual_hints=visual_hints,
        raw_color_inventory_packet=raw_color_inventory_packet,
    )
    for prompt in prompts:
        px = int(prompt["x"])
        py = int(prompt["y"])
        inputs = processor(
            images=image,
            input_points=[[[[float(px), float(py)]]]],
            input_labels=[[[1]]],
            return_tensors="pt",
        )
        original_sizes = inputs["original_sizes"]
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            original_sizes,
            mask_threshold=0.0,
            binarize=True,
            apply_non_overlapping_constraints=False,
        )[0][0]
        scores = outputs.iou_scores.detach().cpu().numpy()[0, 0]
        for variant_idx in range(masks.shape[0]):
            scaled_mask = np.asarray(masks[variant_idx], dtype=bool)
            raw_mask = scaled_to_raw_mask(scaled_mask, raw_shape=raw_shape)
            area_fraction = float(np.count_nonzero(raw_mask) / max(1, total_raw))
            if area_fraction < min_area_fraction or area_fraction > max_area_fraction:
                continue
            bbox = bbox_from_mask(scaled_mask)
            if bbox is None:
                continue
            candidates.append(
                SamCandidate(
                    prompt_source=str(prompt["prompt_source"]),
                    prompt_source_object_id=prompt.get("prompt_source_object_id"),
                    prompt_metadata=dict(prompt.get("prompt_metadata") or {}),
                    prompt_point_scaled_xy=(int(px), int(py)),
                    prompt_point_display_xy=(float(px / scale), float(py / scale)),
                    sam_iou_score=float(scores[variant_idx]),
                    scaled_bbox_xyxy=bbox,
                    raw_mask=raw_mask,
                )
            )

    candidates.sort(key=lambda rec: (-rec.sam_iou_score, -int(np.count_nonzero(rec.raw_mask))))
    chosen: list[SamCandidate] = []
    for candidate in candidates:
        if any(mask_iou(candidate.raw_mask, old.raw_mask) >= dedupe_iou for old in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= max_masks:
            break
    return chosen


def draw_sam_overlay(raw_frame: np.ndarray, rows: list[dict[str, Any]], path: Path, scale: int) -> dict[str, Any]:
    base = Image.fromarray(rgb_frame(raw_frame, scale), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(base)
    palette = [
        (255, 64, 64, 96),
        (64, 200, 255, 96),
        (255, 210, 64, 96),
        (64, 230, 120, 96),
        (220, 80, 255, 96),
        (255, 140, 50, 96),
        (30, 30, 30, 96),
    ]
    for idx, row in enumerate(rows):
        color = palette[idx % len(palette)]
        mask = np.load(ROOT / row["pixel_mask"]["npy"]["path"], allow_pickle=False).astype(np.uint8)
        if scale > 1:
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        overlay = Image.new("RGBA", base.size, (*color[:3], 0))
        alpha = Image.fromarray(mask * color[3], mode="L")
        overlay.putalpha(alpha)
        base.alpha_composite(overlay)
        bbox = row["score"]["bbox_xyxy"]
        if bbox is not None:
            x0, y0, x1, y1 = [int(v) * scale for v in bbox]
            x1 += scale - 1
            y1 += scale - 1
            draw.rectangle((x0, y0, x1, y1), outline=(*color[:3], 255), width=max(1, scale // 2))
            draw.text((x0 + 2, y0 + 2), row["hypothesis_id"], fill=(0, 0, 0, 255))
            draw.text((x0 + 1, y0 + 1), row["hypothesis_id"], fill=(255, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def build_hypothesis_rows_for_level(
    *,
    level_index: int,
    raw_frame: np.ndarray,
    controlled_labels: list[dict[str, Any]],
    raw_color_inventory_packet: dict[str, Any],
    out_dir: Path,
    scale: int,
    processor: Any,
    model: Any,
    device: str,
    grid: int,
    prompt_sources: set[str],
    visual_hints: dict[str, Any] | None,
    max_masks: int,
    min_area_fraction: float,
    max_area_fraction: float,
    dedupe_iou: float,
    confirmed_iou_threshold: float,
) -> list[dict[str, Any]]:
    candidates = extract_sam2_candidates(
        raw_frame=raw_frame,
        scale=scale,
        processor=processor,
        model=model,
        device=device,
        grid=grid,
        prompt_sources=prompt_sources,
        controlled_labels=controlled_labels,
        visual_hints=visual_hints,
        raw_color_inventory_packet=raw_color_inventory_packet,
        max_masks=max_masks,
        min_area_fraction=min_area_fraction,
        max_area_fraction=max_area_fraction,
        dedupe_iou=dedupe_iou,
    )
    rows: list[dict[str, Any]] = []
    for idx, candidate in enumerate(candidates):
        hypothesis_id = f"sam_l{level_index:02d}_m{idx:03d}"
        score = score_sam_hypothesis(
            mask=candidate.raw_mask,
            raw_frame=raw_frame,
            controlled_labels=controlled_labels,
            min_area_fraction=min_area_fraction,
            max_area_fraction=max_area_fraction,
            confirmed_iou_threshold=confirmed_iou_threshold,
        )
        mask_meta = save_mask(
            candidate.raw_mask,
            npy_path=out_dir / "sam_masks" / f"level_{level_index:02d}" / f"{hypothesis_id}.npy",
            png_path=out_dir / "sam_masks" / f"level_{level_index:02d}" / f"{hypothesis_id}.png",
            scale=scale,
        )
        rows.append(
            {
                "schema": SAM_HYPOTHESIS_SCHEMA,
                "game": "sp80_liquid_pour",
                "calibration_slice_index": int(level_index),
                "level_index": int(level_index),
                "hypothesis_id": hypothesis_id,
                "hypothesis_source": f"sam2_transformers_{candidate.prompt_source}",
                "hypothesis_status": score["confirmation_status"],
                "evidence_status": "hypothesis",
                "raw_pixels_are_observation_ground_truth": True,
                "sam_status": "ran_sam2_transformers",
                "sam_backend": "sam2_transformers",
                "sam_iou_score": candidate.sam_iou_score,
                "prompt_source": candidate.prompt_source,
                "prompt_source_object_id": candidate.prompt_source_object_id,
                "prompt_metadata": candidate.prompt_metadata,
                "prompt_point_scaled_xy": list(candidate.prompt_point_scaled_xy),
                "prompt_point_display_xy": list(candidate.prompt_point_display_xy),
                "scaled_bbox_xyxy": list(candidate.scaled_bbox_xyxy),
                "pixel_mask": mask_meta,
                "score": score,
                "admitted_for_control_truth": False,
                "admitted_for_goal_condition": False,
                "admitted_for_contra_object_view": False,
                "admission_blocked_reason": "sam_hypotheses_require_explicit_promotion_after_controlled_confirmation",
                "route_or_replay_used": False,
                "human_trace_used": False,
                "coordinate_script_used": False,
            }
        )
    return rows


def build_scorecard(rows: list[dict[str, Any]], *, score_config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    status_counts = Counter(row["score"]["confirmation_status"] for row in rows)
    geometry_counts = Counter(row["score"]["geometry_hint"] for row in rows)
    prompt_source_counts = Counter(row["prompt_source"] for row in rows)
    confirmed_bucket = [
        row
        for row in rows
        if row["score"].get("bucket_like_by_controlled_overlap") is True
    ]
    scorecard = {
        "schema": SAM_SCORECARD_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "scoring_version": SCORING_VERSION,
        "score_config": score_config,
        "hypothesis_count": len(rows),
        "confirmation_status_counts": dict(sorted(status_counts.items())),
        "geometry_hint_counts": dict(sorted(geometry_counts.items())),
        "prompt_source_counts": dict(sorted(prompt_source_counts.items())),
        "bucket_overlap_candidate_count": len(confirmed_bucket),
        "confirmed_against_controlled_mask_count": int(status_counts.get("confirmed_against_controlled_mask", 0)),
        "admitted_for_control_truth_count": 0,
        "admitted_for_contra_object_view_count": 0,
    }
    path = out_dir / "sam_scorecard.json"
    scorecard["scorecard_json"] = relpath(path)
    write_json(path, scorecard)
    return scorecard


def build_index(manifest: dict[str, Any], rows: list[dict[str, Any]], *, index_dir: Path) -> str:
    rows_by_level: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_level.setdefault(int(row["level_index"]), []).append(row)
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 SAM Hypothesis Viewer</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f4;color:#191919}",
        "section{margin-bottom:36px}",
        "img{image-rendering:pixelated;border:1px solid #aaa;background:white}",
        "table{border-collapse:collapse;font-size:13px;margin-top:12px}",
        "td,th{border:1px solid #ccc;padding:5px 7px;text-align:left;vertical-align:top}",
        ".pair{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}",
        ".note{max-width:980px;line-height:1.35}",
        "code{background:#eee;padding:1px 3px}",
        "</style></head><body>",
        "<h1>SP80 SAM Hypothesis Viewer</h1>",
        "<p class='note'>SAM2 masks are hypotheses. Raw frame pixels are the observation ground truth. "
        "The scorer can confirm overlap against controlled engine masks, but all SAM rows remain blocked from controller truth and goal-condition admission.</p>",
        f"<p><b>Manifest:</b> <code>{html.escape(manifest['manifest_json'])}</code></p>",
        f"<p><b>Scorecard:</b> <code>{html.escape(manifest['sam_scorecard_json'])}</code></p>",
    ]
    for view in manifest["views"]:
        level_index = int(view["level_index"])
        rows_for_level = rows_by_level.get(level_index, [])
        body.extend(
            [
                f"<section><h2>Calibration Slice {level_index}</h2>",
                "<div class='pair'>",
                f"<div><h3>Raw Frame</h3><img src='{rel_for_html(view['raw_frame']['png']['path'], index_dir)}'></div>",
                f"<div><h3>Controlled Overlay</h3><img src='{rel_for_html(view['controlled_overlay']['path'], index_dir)}'></div>",
                f"<div><h3>SAM Hypotheses</h3><img src='{rel_for_html(view['sam_overlay']['path'], index_dir)}'></div>",
                "</div>",
                "<table><thead><tr>"
                "<th>hypothesis</th><th>status</th><th>score</th><th>geometry</th><th>best controlled</th>"
                "<th>bucket iou</th><th>area</th><th>mask</th>"
                "</tr></thead><tbody>",
            ]
        )
        for row in rows_for_level:
            best = row["score"].get("best_controlled_match") or {}
            body.append(
                "<tr>"
                f"<td>{html.escape(row['hypothesis_id'])}</td>"
                f"<td>{html.escape(row['score']['confirmation_status'])}</td>"
                f"<td>{row['score']['overall_score']:.3f}</td>"
                f"<td>{html.escape(row['score']['geometry_hint'])}</td>"
                f"<td><code>{html.escape(json.dumps(best, sort_keys=True))}</code></td>"
                f"<td>{row['score']['best_bucket_iou']:.3f}</td>"
                f"<td>{row['score']['area_pixels']}</td>"
                f"<td><a href='{rel_for_html(row['pixel_mask']['png']['path'], index_dir)}'>png</a></td>"
                "</tr>"
            )
        body.extend(["</tbody></table></section>"])
    body.extend(["</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="0", help="'all' or comma-separated calibration slice indexes")
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_sam_hypothesis_viewer"))
    parser.add_argument("--sam2-model", type=Path, default=DEFAULT_SAM2_MODEL)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument(
        "--prompt-sources",
        default="grid,controlled_centers",
        help="comma-separated: grid, controlled_centers, raw_color_components, raw_yellow_components, visual_hint_yellow_u",
    )
    parser.add_argument("--visual-hints", type=Path, default=None)
    parser.add_argument("--raw-color-min-component-area", type=int, default=1)
    parser.add_argument("--max-masks", type=int, default=8)
    parser.add_argument("--min-area-fraction", type=float, default=0.0008)
    parser.add_argument("--max-area-fraction", type=float, default=0.80)
    parser.add_argument("--dedupe-iou", type=float, default=0.74)
    parser.add_argument("--confirmed-iou-threshold", type=float, default=0.50)
    parser.add_argument("--hash-model-weights", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    level_count = detect_level_count()
    levels = parse_levels(args.levels, level_count)
    scale = max(1, int(args.scale))
    grid = max(1, int(args.grid))
    max_masks = max(1, int(args.max_masks))
    prompt_sources = {part.strip() for part in str(args.prompt_sources).split(",") if part.strip()}
    unknown_prompt_sources = sorted(
        prompt_sources - {"grid", "controlled_centers", "raw_color_components", "raw_yellow_components", "visual_hint_yellow_u"}
    )
    if unknown_prompt_sources:
        raise SystemExit(f"unknown prompt source(s): {unknown_prompt_sources}")
    if not prompt_sources:
        raise SystemExit("at least one prompt source is required")
    visual_hints = None
    visual_hints_path = None
    if args.visual_hints is not None:
        visual_hints_path = resolve_repo_path(args.visual_hints)
        visual_hints = json.loads(visual_hints_path.read_text(encoding="utf-8"))
    if {"raw_yellow_components", "visual_hint_yellow_u"} & prompt_sources and visual_hints is None:
        raise SystemExit("--visual-hints is required when a visual-hint prompt source is selected")
    sam2_model = args.sam2_model.resolve()

    processor, model = load_sam2(sam2_model, args.device)
    all_rows: list[dict[str, Any]] = []
    all_controlled_labels: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for level_index in levels:
        controlled = run_controlled_object_level(level_index, out_dir, scale)
        view = controlled["view"]
        raw_grid_path = ROOT / view["raw_frame"]["raw_grid"]["path"]
        raw_frame = np.load(raw_grid_path, allow_pickle=False)
        color_inventory = raw_color_inventory(
            raw_frame,
            min_component_area=max(1, int(args.raw_color_min_component_area)),
        )
        controlled_labels = controlled["labels"]
        all_controlled_labels.extend(controlled_labels)
        rows = build_hypothesis_rows_for_level(
            level_index=level_index,
            raw_frame=raw_frame,
            controlled_labels=controlled_labels,
            raw_color_inventory_packet=color_inventory,
            out_dir=out_dir,
            scale=scale,
            processor=processor,
            model=model,
            device=args.device,
            grid=grid,
            prompt_sources=prompt_sources,
            visual_hints=visual_hints,
            max_masks=max_masks,
            min_area_fraction=float(args.min_area_fraction),
            max_area_fraction=float(args.max_area_fraction),
            dedupe_iou=float(args.dedupe_iou),
            confirmed_iou_threshold=float(args.confirmed_iou_threshold),
        )
        overlay = draw_sam_overlay(
            raw_frame,
            rows,
            out_dir / "sam_overlays" / f"level_{level_index:02d}_sam_hypotheses.png",
            scale,
        )
        view = {
            **view,
            "controlled_overlay": view["overlay"],
            "sam_overlay": overlay,
            "sam_hypothesis_count": len(rows),
            "sam_hypothesis_ids": [row["hypothesis_id"] for row in rows],
            "raw_color_inventory": color_inventory,
        }
        write_json(out_dir / "sam_views" / f"level_{level_index:02d}.json", view)
        views.append(view)
        all_rows.extend(rows)

    controlled_labels_path = out_dir / "controlled_object_labels.jsonl"
    write_jsonl(controlled_labels_path, all_controlled_labels)
    rows_path = out_dir / "sam_hypotheses.jsonl"
    write_jsonl(rows_path, all_rows)
    score_config = {
        "scoring_version": SCORING_VERSION,
        "min_area_fraction": float(args.min_area_fraction),
        "max_area_fraction": float(args.max_area_fraction),
        "dedupe_iou": float(args.dedupe_iou),
        "confirmed_iou_threshold": float(args.confirmed_iou_threshold),
    }
    scorecard = build_scorecard(all_rows, score_config=score_config, out_dir=out_dir)
    status_counts = Counter(row["score"]["confirmation_status"] for row in all_rows)
    manifest = {
        "schema": SAM_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "sp80_sam2_hypothesis_viewer",
        "condition_natural_language": "Run SAM2 point-prompt segmentation over SP80 raw-frame renders and score the masks against controlled engine labels without admitting SAM as truth.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "game": "sp80_liquid_pour",
        "detected_calibration_slice_count": level_count,
        "requested_calibration_slices": levels,
        "raw_pixels_are_observation_ground_truth": True,
        "sam_status": "ran_sam2_transformers",
        "sam_backend": "sam2_transformers",
        "sam_model": model_file_fingerprints(sam2_model, hash_model_weights=bool(args.hash_model_weights)),
        "device": args.device,
        "scale": scale,
        "grid": grid,
        "max_masks": max_masks,
        "prompt_sources": sorted(prompt_sources),
        "visual_hints_prompt_used_by_sam2_point_prompts": bool(
            {"raw_yellow_components", "visual_hint_yellow_u"} & prompt_sources
        ),
        "controlled_labels_used_for_prompt_points": bool({"controlled_centers", "visual_hint_yellow_u"} & prompt_sources),
        "raw_pixels_used_for_prompt_points": bool({"raw_color_components", "raw_yellow_components"} & prompt_sources),
        "raw_color_inventory_used_for_prompt_points": "raw_color_components" in prompt_sources,
        "raw_color_min_component_area_pixels": max(1, int(args.raw_color_min_component_area)),
        "source_visual_hints_json": None if visual_hints_path is None else relpath(visual_hints_path),
        "source_visual_hints_json_sha256": None if visual_hints_path is None else sha256_file(visual_hints_path),
        "visual_hints": visual_hints,
        "score_config": score_config,
        "confirmation_status_counts": dict(sorted(status_counts.items())),
        "sam_hypothesis_count": len(all_rows),
        "controlled_label_source": "engine_sprite_pixels",
        "sam_truth_boundary": "SAM rows are hypotheses; controller truth and goal conditions remain blocked unless explicitly promoted later.",
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "source_game": relpath(SP80_PATH),
        "source_game_sha256": sha256_file(SP80_PATH),
        "source_controller": "codexlab/sp80_base_controller.py",
        "source_controller_sha256": sha256_file(ROOT / "codexlab" / "sp80_base_controller.py"),
        "source_object_label_viewer": "codexlab/sp80_object_label_viewer.py",
        "source_object_label_viewer_sha256": sha256_file(ROOT / "codexlab" / "sp80_object_label_viewer.py"),
        "source_scoring": "codexlab/sp80_sam_scoring.py",
        "source_scoring_sha256": sha256_file(ROOT / "codexlab" / "sp80_sam_scoring.py"),
        "source_script": "codexlab/sp80_sam_hypothesis_viewer.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_sam_hypothesis_viewer.py"),
        "sam_hypothesis_schema": SAM_HYPOTHESIS_SCHEMA,
        "controlled_object_labels_jsonl": relpath(controlled_labels_path),
        "controlled_object_labels_jsonl_sha256": sha256_file(controlled_labels_path),
        "sam_hypotheses_jsonl": relpath(rows_path),
        "sam_hypotheses_jsonl_sha256": sha256_file(rows_path),
        "sam_scorecard_json": scorecard["scorecard_json"],
        "sam_scorecard_json_sha256": sha256_file(out_dir / "sam_scorecard.json"),
        "views": views,
    }
    manifest_path = out_dir / "sam_hypothesis_manifest.json"
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)

    index_path = out_dir / "index.html"
    index_path.write_text(build_index(manifest, all_rows, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "exported",
                "manifest": relpath(manifest_path),
                "index_html": relpath(index_path),
                "sam_hypotheses_jsonl": relpath(rows_path),
                "sam_hypothesis_count": len(all_rows),
                "confirmation_status_counts": dict(sorted(status_counts.items())),
                "requested_calibration_slices": levels,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
