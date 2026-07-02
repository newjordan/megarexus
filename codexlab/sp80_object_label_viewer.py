#!/usr/bin/env python3
"""Pixel-grounded object-label viewer for SP80/Liquid Pour.

This is a viewer and label provenance tool, not a solver. It records raw SP80
frames as the observation ground truth, then overlays controlled engine-sprite
pixel masks for review. SAM or another segmenter can later add hypothesis rows
beside these controlled rows, but those rows must not replace the raw-frame
and mask provenance recorded here.
"""

from __future__ import annotations

import argparse
import html
import json
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from sp80_base_controller import (
    ROOT,
    SP80_PATH,
    SP80BaseController,
    load_sp80_class,
    rgb_frame,
    save_frame,
    sha256_file,
    sha256_frame,
)


OBJECT_LABEL_SCHEMA = "codexlab.sp80_object_label.v1"
OBJECT_VIEW_SCHEMA = "codexlab.sp80_object_view.v1"
MANIFEST_SCHEMA = "codexlab.sp80_object_label_viewer_manifest.v1"

TAG_PRIORITY = [
    "repwkzbkhxl",
    "plzwjbfyfli",
    "tuvkdkhdokr",
    "liolfvkveqg",
    "sowlljgtjvn",
    "waoewejnqzc",
]

TAG_SEMANTICS = {
    "repwkzbkhxl": {
        "semantic_label": "bucket",
        "semantic_label_status": "controlled_sp80_engine_tag_mapping",
        "admitted_for_goal_condition": True,
    },
    "plzwjbfyfli": {
        "semantic_label": "selectable_control_object",
        "semantic_label_status": "controlled_sp80_selectable_api_mapping",
        "admitted_for_goal_condition": False,
    },
    "tuvkdkhdokr": {
        "semantic_label": "selectable_control_object",
        "semantic_label_status": "controlled_sp80_selectable_api_mapping",
        "admitted_for_goal_condition": False,
    },
    "liolfvkveqg": {
        "semantic_label": "engine_object",
        "semantic_label_status": "uninterpreted_engine_tag",
        "admitted_for_goal_condition": False,
    },
    "sowlljgtjvn": {
        "semantic_label": "engine_object",
        "semantic_label_status": "uninterpreted_engine_tag",
        "admitted_for_goal_condition": False,
    },
    "waoewejnqzc": {
        "semantic_label": "engine_object",
        "semantic_label_status": "uninterpreted_engine_tag",
        "admitted_for_goal_condition": False,
    },
    "bodekplurlf": {
        "semantic_label": "engine_object",
        "semantic_label_status": "uninterpreted_engine_name_prefix",
        "admitted_for_goal_condition": False,
    },
}

OVERLAY_COLORS = [
    (230, 65, 65),
    (46, 160, 67),
    (30, 120, 230),
    (225, 160, 30),
    (160, 90, 220),
    (0, 170, 170),
    (230, 90, 180),
    (120, 120, 30),
]


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def save_raw_grid(frame: np.ndarray, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame, dtype=np.int16)
    np.save(path, arr)
    return {
        "path": relpath(path),
        "npy_sha256": sha256_file(path),
        "raw_frame_sha256": sha256_frame(arr),
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
    }


def save_mask(mask: np.ndarray, *, npy_path: Path, png_path: Path, scale: int) -> dict[str, Any]:
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(mask, dtype=np.uint8)
    np.save(npy_path, arr)
    png = arr * 255
    if scale > 1:
        png = np.repeat(np.repeat(png, scale, axis=0), scale, axis=1)
    Image.fromarray(png, mode="L").save(png_path)
    return {
        "npy": {
            "path": relpath(npy_path),
            "npy_sha256": sha256_file(npy_path),
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
        },
        "png": {
            "path": relpath(png_path),
            "png_sha256": sha256_file(png_path),
        },
        "area_pixels": int(np.count_nonzero(arr)),
    }


def detect_level_count() -> int:
    game = load_sp80_class()()
    levels = getattr(game, "_levels", None)
    if levels is None:
        raise RuntimeError("could not find SP80 _levels")
    return len(levels)


def parse_levels(value: str, level_count: int) -> list[int]:
    if value == "all":
        return list(range(level_count))
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    for level_index in levels:
        if level_index < 0 or level_index >= level_count:
            raise ValueError(f"calibration slice {level_index} outside 0..{level_count - 1}")
    return levels


def choose_source_tag(sprite: Any) -> str:
    tags = [str(tag) for tag in (getattr(sprite, "tags", []) or [])]
    for tag in TAG_PRIORITY:
        if tag in tags:
            return tag
    name = str(getattr(sprite, "name", ""))
    if name.startswith("bodekplurlf"):
        return "bodekplurlf"
    return tags[0] if tags else name or "unknown"


def semantic_for(sprite: Any, selectable_ids: set[int]) -> dict[str, Any]:
    source_tag = choose_source_tag(sprite)
    semantic = dict(TAG_SEMANTICS.get(source_tag, {}))
    if not semantic:
        semantic = {
            "semantic_label": "engine_object",
            "semantic_label_status": "uninterpreted_engine_tag",
            "admitted_for_goal_condition": False,
        }
    if id(sprite) in selectable_ids:
        semantic["semantic_label"] = "selectable_control_object"
        semantic["semantic_label_status"] = "controlled_sp80_selectable_api_mapping"
    semantic["source_tag"] = source_tag
    return semantic


def display_xy(game: Any, raw_x: int, raw_y: int) -> tuple[int, int]:
    if hasattr(game, "rqlmciubuz"):
        x, y = game.rqlmciubuz(raw_x, raw_y)
        return int(x), int(y)
    return int(raw_x), int(raw_y)


def sprite_display_mask(game: Any, sprite: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    camera = getattr(game, "camera", None) or getattr(game, "_camera", None)
    if camera is None:
        return np.zeros((height, width), dtype=np.uint8)

    camera_height = int(getattr(camera, "height", height))
    camera_width = int(getattr(camera, "width", width))
    raw_mask = np.zeros((camera_height, camera_width), dtype=np.uint8)
    pixels = np.asarray(sprite.render() if hasattr(sprite, "render") else getattr(sprite, "pixels", []))
    if pixels.ndim != 2:
        return np.zeros((height, width), dtype=np.uint8)

    rel_x = int(sprite.x) - int(getattr(camera, "x", 0))
    rel_y = int(sprite.y) - int(getattr(camera, "y", 0))
    sprite_height, sprite_width = pixels.shape
    dest_x_start = max(0, rel_x)
    dest_x_end = min(camera_width, rel_x + sprite_width)
    dest_y_start = max(0, rel_y)
    dest_y_end = min(camera_height, rel_y + sprite_height)
    if dest_x_end <= dest_x_start or dest_y_end <= dest_y_start:
        return np.zeros((height, width), dtype=np.uint8)

    sprite_x_start = max(0, -rel_x)
    sprite_x_end = sprite_width - max(0, (rel_x + sprite_width) - camera_width)
    sprite_y_start = max(0, -rel_y)
    sprite_y_end = sprite_height - max(0, (rel_y + sprite_height) - camera_height)
    visible_mask = pixels[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end] >= 0
    raw_target = raw_mask[dest_y_start:dest_y_end, dest_x_start:dest_x_end]
    raw_target[visible_mask] = 1

    if hasattr(camera, "_calculate_scale_and_offset"):
        camera_scale, x_offset, y_offset = camera._calculate_scale_and_offset()
    else:
        camera_scale = max(1, min(width // max(1, camera_width), height // max(1, camera_height)))
        x_offset = (width - camera_width * camera_scale) // 2
        y_offset = (height - camera_height * camera_scale) // 2

    if camera_scale > 1:
        raw_mask = np.repeat(np.repeat(raw_mask, camera_scale, axis=0), camera_scale, axis=1)
    mask = np.zeros((height, width), dtype=np.uint8)
    y_end = min(height, y_offset + raw_mask.shape[0])
    x_end = min(width, x_offset + raw_mask.shape[1])
    src_h = max(0, y_end - y_offset)
    src_w = max(0, x_end - x_offset)
    if src_h > 0 and src_w > 0:
        mask[y_offset:y_end, x_offset:x_end] = raw_mask[:src_h, :src_w]

    rotation_k = int(getattr(game, "fahhoimkk", 0)) % 4
    if rotation_k:
        mask = np.rot90(mask, k=rotation_k).copy()
    return mask


def bbox_from_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def masked_value_counts(frame: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    values = np.asarray(frame)[np.asarray(mask, dtype=bool)]
    counts = Counter(int(value) for value in values.tolist())
    return {str(key): int(value) for key, value in sorted(counts.items())}


def object_id_for(source_tag: str, semantic_label: str, index: int) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in source_tag).strip("_") or "object"
    if semantic_label == "bucket":
        return f"bucket_{index:02d}"
    return f"{clean}_{index:02d}"


def draw_overlay(frame: np.ndarray, labels: list[dict[str, Any]], path: Path, scale: int) -> dict[str, Any]:
    base = Image.fromarray(rgb_frame(frame, scale), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(base)
    for idx, label in enumerate(labels):
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        mask_meta = label["pixel_mask"]
        mask = np.load(ROOT / mask_meta["npy"]["path"])
        if scale > 1:
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        overlay = Image.new("RGBA", base.size, (*color, 0))
        alpha = Image.fromarray((mask.astype(np.uint8) * 95), mode="L")
        overlay.putalpha(alpha)
        base.alpha_composite(overlay)
        bbox = label.get("display_pixel_bbox_xyxy")
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) * scale for v in bbox]
            x2 += scale - 1
            y2 += scale - 1
            draw.rectangle([x1, y1, x2, y2], outline=(*color, 255), width=max(1, scale // 2))
            draw.text((x1 + 2, y1 + 2), label["object_id"], fill=(0, 0, 0, 255))
            draw.text((x1 + 1, y1 + 1), label["object_id"], fill=(255, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def sprite_rows_for_level(
    *,
    controller: SP80BaseController,
    game: Any,
    frame: np.ndarray,
    level_index: int,
    out_dir: Path,
    scale: int,
) -> list[dict[str, Any]]:
    selectable_ids = {id(sprite) for sprite in game.fbrwmvzsym()} if hasattr(game, "fbrwmvzsym") else set()
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for sprite_index, sprite in enumerate(game.current_level.get_sprites()):
        if not bool(getattr(sprite, "visible", True)):
            continue
        semantic = semantic_for(sprite, selectable_ids)
        source_tag = semantic["source_tag"]
        semantic_label = semantic["semantic_label"]
        object_index = counters[semantic_label if semantic_label == "bucket" else source_tag]
        counters[semantic_label if semantic_label == "bucket" else source_tag] += 1
        object_id = object_id_for(source_tag, semantic_label, object_index)
        mask = sprite_display_mask(game, sprite, frame.shape[:2])
        if int(np.count_nonzero(mask)) == 0:
            continue
        mask_meta = save_mask(
            mask,
            npy_path=out_dir / "masks" / f"level_{level_index:02d}" / f"{object_id}.npy",
            png_path=out_dir / "masks" / f"level_{level_index:02d}" / f"{object_id}.png",
            scale=scale,
        )
        display_bbox = bbox_from_mask(mask)
        row = {
            "schema": OBJECT_LABEL_SCHEMA,
            "game": "sp80_liquid_pour",
            "calibration_slice_index": int(level_index),
            "level_index": int(level_index),
            "object_id": object_id,
            "sprite_index": int(sprite_index),
            "name": str(getattr(sprite, "name", "")),
            "source_tag": source_tag,
            "tags": [str(tag) for tag in (getattr(sprite, "tags", []) or [])],
            "semantic_label": semantic_label,
            "semantic_label_status": semantic["semantic_label_status"],
            "label_source": "engine_sprite_pixels",
            "mask_source": "engine_sprite_nontransparent_pixels_projected_to_display_frame",
            "evidence_status": "controlled",
            "raw_pixels_are_observation_ground_truth": True,
            "admitted_for_contra_object_view": True,
            "admitted_for_goal_condition": bool(semantic["admitted_for_goal_condition"]),
            "sam_status": "not_run",
            "bbox_from": "pixel_mask",
            "display_pixel_bbox_xyxy": display_bbox,
            "display_pixel_center": None
            if display_bbox is None
            else [
                (display_bbox[0] + display_bbox[2]) / 2.0,
                (display_bbox[1] + display_bbox[3]) / 2.0,
            ],
            "raw_sprite_bbox_xyxy": [
                int(sprite.x),
                int(sprite.y),
                int(sprite.x + sprite.width - 1),
                int(sprite.y + sprite.height - 1),
            ],
            "pixel_mask": mask_meta,
            "masked_raw_pixel_value_counts": masked_value_counts(frame, mask),
            "route_or_replay_used": False,
            "human_trace_used": False,
            "coordinate_script_used": False,
        }
        rows.append(row)
    return rows


def run_level(level_index: int, out_dir: Path, scale: int) -> dict[str, Any]:
    controller = SP80BaseController(level_index=level_index)
    game, obs = controller.new_game()
    frame = np.asarray(obs.frame[-1])
    raw_png = save_frame(frame, out_dir / "frames" / f"level_{level_index:02d}_raw.png", scale)
    raw_grid = save_raw_grid(frame, out_dir / "frames" / f"level_{level_index:02d}_raw.npy")
    labels = sprite_rows_for_level(
        controller=controller,
        game=game,
        frame=frame,
        level_index=level_index,
        out_dir=out_dir,
        scale=scale,
    )
    overlay = draw_overlay(
        frame,
        labels,
        out_dir / "overlays" / f"level_{level_index:02d}_object_overlay.png",
        scale,
    )
    view = {
        "schema": OBJECT_VIEW_SCHEMA,
        "game": "sp80_liquid_pour",
        "calibration_slice_index": int(level_index),
        "level_index": int(level_index),
        "raw_frame": {"png": raw_png, "raw_grid": raw_grid},
        "overlay": overlay,
        "object_count": len(labels),
        "controlled_object_count": len([row for row in labels if row["evidence_status"] == "controlled"]),
        "bucket_count": len([row for row in labels if row["semantic_label"] == "bucket"]),
        "sam_hypothesis_count": 0,
        "object_ids": [row["object_id"] for row in labels],
    }
    write_json(out_dir / "object_views" / f"level_{level_index:02d}.json", view)
    return {"view": view, "labels": labels}


def build_index(
    manifest: dict[str, Any],
    views: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    *,
    index_dir: Path,
) -> str:
    labels_by_level: dict[int, list[dict[str, Any]]] = {}
    for label in labels:
        labels_by_level.setdefault(int(label["level_index"]), []).append(label)
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 Object Label Viewer</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f4;color:#191919}",
        "section{margin-bottom:36px}",
        "img{image-rendering:pixelated;border:1px solid #bbb;background:white}",
        "table{border-collapse:collapse;font-size:13px;margin-top:12px}",
        "td,th{border:1px solid #ccc;padding:5px 7px;text-align:left;vertical-align:top}",
        ".pair{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}",
        ".note{max-width:900px;line-height:1.35}",
        "code{background:#eee;padding:1px 3px}",
        "</style></head><body>",
        "<h1>SP80 Object Label Viewer</h1>",
        "<p class='note'>Raw frame pixels are the observation ground truth. Colored overlays are review aids. "
        "Rows marked <code>controlled</code> use engine sprite pixels projected onto the raw display frame. "
        "SAM rows are not present in this run.</p>",
        f"<p><b>Manifest:</b> <code>{html.escape(manifest['manifest_json'])}</code></p>",
    ]
    for view in views:
        level_index = int(view["level_index"])
        body.extend(
            [
                f"<section><h2>Calibration Slice {level_index}</h2>",
                "<div class='pair'>",
                f"<div><h3>Raw Frame</h3><img src='{rel_for_html(view['raw_frame']['png']['path'], index_dir)}'></div>",
                f"<div><h3>Controlled Object Overlay</h3><img src='{rel_for_html(view['overlay']['path'], index_dir)}'></div>",
                "</div>",
                "<table><thead><tr>"
                "<th>object_id</th><th>semantic_label</th><th>source_tag</th><th>status</th>"
                "<th>mask pixels</th><th>bbox</th><th>pixel values</th>"
                "</tr></thead><tbody>",
            ]
        )
        for label in labels_by_level.get(level_index, []):
            body.append(
                "<tr>"
                f"<td>{html.escape(label['object_id'])}</td>"
                f"<td>{html.escape(label['semantic_label'])}</td>"
                f"<td>{html.escape(label['source_tag'])}</td>"
                f"<td>{html.escape(label['evidence_status'])}</td>"
                f"<td>{label['pixel_mask']['area_pixels']}</td>"
                f"<td><code>{html.escape(json.dumps(label['display_pixel_bbox_xyxy']))}</code></td>"
                f"<td><code>{html.escape(json.dumps(label['masked_raw_pixel_value_counts'], sort_keys=True))}</code></td>"
                "</tr>"
            )
        body.extend(["</tbody></table></section>"])
    body.extend(["</body></html>"])
    return "\n".join(body)


def rel_for_html(repo_relpath: str, index_dir: Path) -> str:
    target = ROOT / repo_relpath
    return html.escape(target.resolve().relative_to(index_dir.resolve()).as_posix())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="0", help="'all' or comma-separated calibration slice indexes")
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_object_label_viewer"))
    parser.add_argument("--scale", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    level_count = detect_level_count()
    levels = parse_levels(args.levels, level_count)
    scale = max(1, int(args.scale))

    all_labels: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for level_index in levels:
        result = run_level(level_index, out_dir, scale)
        views.append(result["view"])
        all_labels.extend(result["labels"])

    labels_path = out_dir / "object_labels.jsonl"
    write_jsonl(labels_path, all_labels)
    labels_by_semantic = Counter(row["semantic_label"] for row in all_labels)
    labels_by_source_tag = Counter(row["source_tag"] for row in all_labels)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "sp80_pixel_grounded_object_label_viewer",
        "condition_natural_language": "Record SP80 raw frames and controlled engine-sprite pixel masks for object-label review.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "game": "sp80_liquid_pour",
        "detected_calibration_slice_count": level_count,
        "requested_calibration_slices": levels,
        "raw_pixels_are_observation_ground_truth": True,
        "label_boundary": "controlled engine pixel masks are review/training candidates; SAM output, when added, must be hypothesis until verified.",
        "sam_status": "not_configured",
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "source_game": relpath(SP80_PATH),
        "source_game_sha256": sha256_file(SP80_PATH),
        "source_controller": "codexlab/sp80_base_controller.py",
        "source_controller_sha256": sha256_file(ROOT / "codexlab" / "sp80_base_controller.py"),
        "source_script": "codexlab/sp80_object_label_viewer.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_object_label_viewer.py"),
        "object_label_schema": OBJECT_LABEL_SCHEMA,
        "object_view_schema": OBJECT_VIEW_SCHEMA,
        "object_labels_jsonl": relpath(labels_path),
        "object_labels_jsonl_sha256": sha256_file(labels_path),
        "object_view_count": len(views),
        "object_label_count": len(all_labels),
        "labels_by_semantic_label": dict(sorted(labels_by_semantic.items())),
        "labels_by_source_tag": dict(sorted(labels_by_source_tag.items())),
        "views": views,
    }
    manifest_path = out_dir / "label_review_manifest.json"
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)

    index_path = out_dir / "index.html"
    index_path.write_text(build_index(manifest, views, all_labels, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "exported",
                "manifest": relpath(manifest_path),
                "index_html": relpath(index_path),
                "object_labels_jsonl": relpath(labels_path),
                "object_label_count": len(all_labels),
                "requested_calibration_slices": levels,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
