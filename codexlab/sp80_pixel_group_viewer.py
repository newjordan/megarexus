#!/usr/bin/env python3
"""Raw-pixel connected-component viewer for SP80/Liquid Pour.

This is a viewer-first visual pass. It groups exact raw frame pixel values into
connected components. It does not use SAM, controlled engine labels, human
replay, routes, or semantic object names.
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
from PIL import Image, ImageDraw, ImageFont

from contra_head import (
    BACKGROUND_SELECTION_POLICY,
    CONTROL_ISOLATION_SCHEMA,
    CONTROL_ISOLATION_SOURCE,
    build_control_isolation_packet,
    contra_module_source_paths,
)
from sp80_base_controller import (
    PALETTE,
    ROOT,
    SP80_PATH,
    SP80BaseController,
    rgb_frame,
    save_frame,
    sha256_file,
    sha256_frame,
)
from sp80_object_label_viewer import (
    detect_level_count,
    parse_levels,
    relpath,
    resolve_repo_path,
    save_mask,
    save_raw_grid,
    write_json,
    write_jsonl,
)
from review_helpers import rel_for_html


PIXEL_GROUP_SCHEMA = "codexlab.sp80_raw_pixel_group.v1"
ORTHOGONAL_SHAPE_SCHEMA = "codexlab.sp80_raw_orthogonal_shape.v1"
BACKGROUND_INSET_SCHEMA = "codexlab.sp80_raw_background_inset.v1"
SHAPE_WITH_BACKGROUND_INSETS_SCHEMA = "codexlab.sp80_shape_with_background_insets.v1"
PIXEL_GROUP_MANIFEST_SCHEMA = "codexlab.sp80_pixel_group_viewer_manifest.v1"

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


def component_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    bbox = row.get("bbox_xyxy") or [0, 0, 0, 0]
    return (int(row.get("raw_pixel_value", 0)), int(bbox[1]), int(bbox[0]), str(row.get("group_id", "")))


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


def resize_nearest(image: Image.Image, scale: int) -> Image.Image:
    if scale <= 1:
        return image
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def image_meta(path: Path) -> dict[str, Any]:
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def component_geometry(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "bbox_xyxy": None,
            "centroid_xy": None,
            "touches_edge": False,
            "edge_touch_count": 0,
            "width": 0,
            "height": 0,
        }
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
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
    if geometry["height"] == 1 or geometry["width"] == 1:
        return "thin_line"
    if area_fraction <= 0.0015:
        return "single_or_tiny_pixel_group"
    if geometry.get("edge_touch_count", 0) >= 2:
        return "edge_connected_group"
    return "local_pixel_group"


def connected_components(frame: np.ndarray, *, connectivity: int) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    seen = np.zeros((height, width), dtype=bool)
    rows: list[dict[str, Any]] = []
    neighbors = component_neighbors(connectivity)
    component_counter: Counter[int] = Counter()
    for start_y in range(height):
        for start_x in range(width):
            if seen[start_y, start_x]:
                continue
            value = int(frame[start_y, start_x])
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
                    if seen[ny, nx] or int(frame[ny, nx]) != value:
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


def load_row_mask(row: dict[str, Any]) -> np.ndarray:
    return np.load(ROOT / row["pixel_mask"]["npy"]["path"], allow_pickle=False)


def edge_touch_count(row: dict[str, Any]) -> int:
    if "edge_touch_count" in row:
        return int(row["edge_touch_count"])
    touches = row.get("edge_touches") or {}
    return int(sum(1 for value in touches.values() if value))


def shape_exclusion_reason(
    row: dict[str, Any],
    *,
    frame_shape: tuple[int, int],
    large_component_area_fraction_min: float,
    edge_component_area_fraction_min: float,
) -> str | None:
    if float(row["area_fraction"]) >= large_component_area_fraction_min:
        return "large_field_component"
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    full_width_or_height = int(row["width"]) >= frame_w or int(row["height"]) >= frame_h
    if (
        bool(row.get("touches_edge"))
        and float(row["area_fraction"]) >= edge_component_area_fraction_min
        and (edge_touch_count(row) >= 2 or full_width_or_height)
    ):
        return "edge_frame_or_wall_component"
    return None


def color_for_value(value: int) -> tuple[int, int, int]:
    rgb = PALETTE[int(value)].tolist()
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


SHAPE_OVERLAY_COLORS = [
    (20, 92, 214),
    (225, 83, 45),
    (28, 150, 82),
    (154, 80, 220),
    (210, 145, 24),
    (0, 154, 170),
    (204, 62, 135),
    (88, 88, 88),
]


def draw_label_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = ImageFont.load_default()
    lines = text.split("\n")
    widths = [draw.textbbox((0, 0), line, font=font)[2] for line in lines]
    line_h = draw.textbbox((0, 0), "Ag", font=font)[3] + 3
    width = max(widths) + 10
    height = len(lines) * line_h + 7
    x, y = xy
    draw.rectangle([x, y, x + width, y + height], fill=fill)
    for idx, line in enumerate(lines):
        draw.text((x + 5, y + 4 + idx * line_h), line, fill="white", font=font)


def draw_component_overlay(frame: np.ndarray, rows: list[dict[str, Any]], path: Path, scale: int) -> dict[str, Any]:
    base = Image.fromarray(rgb_frame(frame, scale), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(base)
    for row in rows:
        value = int(row["raw_pixel_value"])
        color = color_for_value(value)
        mask = np.load(ROOT / row["pixel_mask"]["npy"]["path"], allow_pickle=False)
        if scale > 1:
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        overlay = Image.new("RGBA", base.size, (*color, 0))
        alpha = Image.fromarray((mask.astype(np.uint8) * 95), mode="L")
        overlay.putalpha(alpha)
        base.alpha_composite(overlay)
        bbox = row["bbox_xyxy"]
        if bbox is not None:
            x0, y0, x1, y1 = [int(v) * scale for v in bbox]
            x1 += scale - 1
            y1 += scale - 1
            outline = (255 - color[0], 255 - color[1], 255 - color[2])
            draw.rectangle([x0, y0, x1, y1], outline=(*outline, 255), width=max(1, scale // 2))
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def isolated_component_image(frame: np.ndarray, row: dict[str, Any], scale: int) -> Image.Image:
    raw_rgb = rgb_frame(frame, 1)
    mask = np.load(ROOT / row["pixel_mask"]["npy"]["path"], allow_pickle=False).astype(bool)
    muted = (raw_rgb.astype(np.float32) * 0.35 + np.array([180, 180, 180], dtype=np.float32) * 0.65).astype(np.uint8)
    out = muted
    out[mask] = raw_rgb[mask]
    image = Image.fromarray(out, mode="RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    bbox = row["bbox_xyxy"]
    if bbox is not None:
        x0, y0, x1, y1 = [int(v) * scale for v in bbox]
        x1 += scale - 1
        y1 += scale - 1
        draw.rectangle([x0, y0, x1, y1], outline=(20, 65, 170), width=max(1, scale // 2))
    text = "\n".join(
        [
            row["group_id"],
            f"value={row['raw_pixel_value']} {row['palette_name']}",
            f"area={row['area_pixels']} {row['geometry_hint']}",
        ]
    )
    draw_label_box(draw, (8, 8), text, (20, 65, 170))
    return image


def make_slice_contact(
    *,
    frame: np.ndarray,
    view: dict[str, Any],
    rows: list[dict[str, Any]],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    thumb_w = int(frame.shape[1] * scale)
    thumb_h = int(frame.shape[0] * scale)
    gap = 16
    label_h = 44
    top_h = label_h + thumb_h
    selected = sorted(rows, key=lambda row: (row["raw_pixel_value"], row["bbox_xyxy"][1], row["bbox_xyxy"][0]))
    cols = 3
    component_rows = (len(selected) + cols - 1) // cols
    width = cols * thumb_w + (cols + 1) * gap
    height = top_h + gap + component_rows * (thumb_h + gap) + 24
    canvas = Image.new("RGB", (width, max(height, top_h + 80)), (247, 247, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f"SP80 Raw Pixel Group Review - Calibration Slice {view['level_index']}", fill=(20, 20, 20))
    raw = Image.open(ROOT / view["raw_frame"]["png"]["path"]).convert("RGB")
    overlay = Image.open(ROOT / view["pixel_group_overlay"]["path"]).convert("RGB")
    for idx, (title, image) in enumerate([("raw frame", raw), ("pixel group overlay", overlay)]):
        x = gap + idx * (thumb_w + gap)
        y = label_h
        canvas.paste(image, (x, y))
        draw_label_box(draw, (x + 6, y + 6), title, (45, 45, 45))
    draw.text((gap, label_h + thumb_h + 10), "Isolated raw pixel groups", fill=(20, 20, 20))
    start_y = label_h + thumb_h + 32
    for idx, row in enumerate(selected):
        col = idx % cols
        row_idx = idx // cols
        x = gap + col * (thumb_w + gap)
        y = start_y + row_idx * (thumb_h + gap)
        canvas.paste(isolated_component_image(frame, row, scale), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def draw_shape_overlay(frame: np.ndarray, shapes: list[dict[str, Any]], path: Path, scale: int) -> dict[str, Any]:
    base = Image.fromarray(rgb_frame(frame, scale), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(base)
    for idx, shape in enumerate(shapes):
        color = SHAPE_OVERLAY_COLORS[idx % len(SHAPE_OVERLAY_COLORS)]
        mask = np.load(ROOT / shape["shape_mask"]["npy"]["path"], allow_pickle=False)
        if scale > 1:
            mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
        overlay = Image.new("RGBA", base.size, (*color, 0))
        alpha = Image.fromarray((mask.astype(np.uint8) * 105), mode="L")
        overlay.putalpha(alpha)
        base.alpha_composite(overlay)
        bbox = shape["bbox_xyxy"]
        if bbox is not None:
            x0, y0, x1, y1 = [int(v) * scale for v in bbox]
            x1 += scale - 1
            y1 += scale - 1
            draw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=max(1, scale // 2))
            draw.text((x0 + 3, y0 + 3), shape["shape_id"].split("_")[-1], fill=(255, 255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def draw_control_inplace_isolation_overlay(
    frame: np.ndarray,
    rows: list[dict[str, Any]],
    path: Path,
    scale: int,
    background_value: int,
) -> dict[str, Any]:
    raw_rgb = rgb_frame(frame, 1)
    out = raw_rgb.copy()
    background = np.asarray(frame) == int(background_value)
    if np.any(background):
        bg_tint = np.array([221, 187, 148], dtype=np.float32)
        out[background] = (raw_rgb[background].astype(np.float32) * 0.76 + bg_tint * 0.24).astype(np.uint8)
    for row in sorted(rows, key=component_sort_key):
        if int(row["raw_pixel_value"]) == int(background_value):
            continue
        mask = load_row_mask(row).astype(bool)
        boundary = boundary_mask(mask).astype(bool)
        rgb = np.array(color_for_value(int(row["raw_pixel_value"])), dtype=np.uint8)
        brightness = float(np.mean(rgb))
        outline = np.array([255, 255, 255] if brightness < 70 else [0, 0, 0], dtype=np.uint8)
        out[mask] = raw_rgb[mask]
        out[boundary] = outline
    image = Image.fromarray(out, mode="RGB")
    image = resize_nearest(image, scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return image_meta(path)


def make_shape_contact(
    *,
    frame: np.ndarray,
    view: dict[str, Any],
    shapes: list[dict[str, Any]],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    thumb_w = int(frame.shape[1] * scale)
    thumb_h = int(frame.shape[0] * scale)
    gap = 16
    label_h = 44
    cols = 2
    rows = 1
    width = cols * thumb_w + (cols + 1) * gap
    height = label_h + rows * (thumb_h + gap) + 56
    canvas = Image.new("RGB", (width, height), (247, 247, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, 10),
        f"SP80 In-Place Orthogonal Isolation Review - Calibration Slice {view['level_index']}",
        fill=(20, 20, 20),
    )
    raw = Image.open(ROOT / view["raw_frame"]["png"]["path"]).convert("RGB")
    control_overlay = Image.open(ROOT / view["control_inplace_isolation_overlay"]["path"]).convert("RGB")
    for idx, (title, image) in enumerate(
        [
            ("raw frame", raw),
            ("control map isolation", control_overlay),
        ]
    ):
        col = idx % cols
        row = idx // cols
        x = gap + col * (thumb_w + gap)
        y = label_h + row * (thumb_h + gap)
        canvas.paste(image, (x, y))
        draw_label_box(draw, (x + 6, y + 6), title, (45, 45, 45))
    draw.text(
        (gap, height - 28),
        f"{len(shapes)} orthogonal foreground components plus map structures reviewed in place; orange insets stay visible as background.",
        fill=(20, 20, 20),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def shape_with_insets_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    bbox = row.get("host_bbox_xyxy") or row.get("bbox_xyxy") or [0, 0, 0, 0]
    return (
        int(row.get("host_raw_pixel_value", 0)),
        int(bbox[1]),
        int(bbox[0]),
        str(row.get("host_group_id", "")),
    )


def draw_background_inset_overlay(
    frame: np.ndarray,
    shape_with_insets: list[dict[str, Any]],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    base = Image.fromarray(rgb_frame(frame, scale), mode="RGB").convert("RGBA")
    draw = ImageDraw.Draw(base)
    for idx, shape in enumerate(sorted(shape_with_insets, key=shape_with_insets_sort_key)):
        host_color = color_for_value(int(shape["host_raw_pixel_value"]))
        inset_mask = np.load(ROOT / shape["background_insets_union_mask"]["npy"]["path"], allow_pickle=False)
        full_mask = np.load(ROOT / shape["shape_with_background_insets_mask"]["npy"]["path"], allow_pickle=False)
        host_mask = np.asarray(full_mask, dtype=bool) & ~np.asarray(inset_mask, dtype=bool)
        if scale > 1:
            inset_mask = np.repeat(np.repeat(inset_mask, scale, axis=0), scale, axis=1)
            full_mask = np.repeat(np.repeat(full_mask, scale, axis=0), scale, axis=1)
            host_mask = np.repeat(np.repeat(host_mask.astype(np.uint8), scale, axis=0), scale, axis=1)
        host_overlay = Image.new("RGBA", base.size, (*host_color, 0))
        host_overlay.putalpha(Image.fromarray((host_mask.astype(np.uint8) * 80), mode="L"))
        base.alpha_composite(host_overlay)
        inset_overlay = Image.new("RGBA", base.size, (0, 185, 255, 0))
        inset_overlay.putalpha(Image.fromarray((inset_mask.astype(np.uint8) * 185), mode="L"))
        base.alpha_composite(inset_overlay)
        bbox = shape["bbox_xyxy"]
        if bbox is not None:
            x0, y0, x1, y1 = [int(v) * scale for v in bbox]
            x1 += scale - 1
            y1 += scale - 1
            draw.rectangle([x0, y0, x1, y1], outline=(0, 120, 210, 255), width=max(1, scale // 2))
            draw.text((x0 + 3, y0 + 3), str(idx), fill=(0, 0, 0, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def isolated_shape_with_insets_image(
    frame: np.ndarray,
    shape: dict[str, Any],
    scale: int,
) -> Image.Image:
    raw_rgb = rgb_frame(frame, 1)
    inset_mask = np.load(ROOT / shape["background_insets_union_mask"]["npy"]["path"], allow_pickle=False).astype(bool)
    full_mask = np.load(ROOT / shape["shape_with_background_insets_mask"]["npy"]["path"], allow_pickle=False).astype(bool)
    muted = (raw_rgb.astype(np.float32) * 0.28 + np.array([196, 196, 188], dtype=np.float32) * 0.72).astype(np.uint8)
    out = muted
    out[full_mask] = raw_rgb[full_mask]
    out[inset_mask] = np.array([0, 190, 255], dtype=np.uint8)
    image = Image.fromarray(out, mode="RGB")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    bbox = shape["bbox_xyxy"]
    if bbox is not None:
        x0, y0, x1, y1 = [int(v) * scale for v in bbox]
        x1 += scale - 1
        y1 += scale - 1
        draw.rectangle([x0, y0, x1, y1], outline=(0, 120, 210), width=max(1, scale // 2))
    host_bbox = shape["host_bbox_xyxy"]
    if host_bbox is not None and host_bbox != bbox:
        x0, y0, x1, y1 = [int(v) * scale for v in host_bbox]
        x1 += scale - 1
        y1 += scale - 1
        draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=max(1, scale // 2))
    text = "\n".join(
        [
            str(shape["host_group_id"]),
            f"host={shape['host_raw_pixel_value']} {shape['host_palette_name']} area={shape['host_area_pixels']}",
            f"orange insets={shape['background_inset_count']} area={shape['background_inset_area_pixels']}",
            f"complete area={shape['area_pixels']}",
        ]
    )
    draw_label_box(draw, (8, 8), text, (0, 105, 180))
    return image


def make_background_inset_contact(
    *,
    frame: np.ndarray,
    view: dict[str, Any],
    shape_with_insets: list[dict[str, Any]],
    path: Path,
    scale: int,
) -> dict[str, Any]:
    thumb_w = int(frame.shape[1] * scale)
    thumb_h = int(frame.shape[0] * scale)
    gap = 16
    label_h = 44
    top_h = label_h + thumb_h
    selected = sorted(shape_with_insets, key=shape_with_insets_sort_key)
    cols = 3
    host_rows = (len(selected) + cols - 1) // cols
    width = cols * thumb_w + (cols + 1) * gap
    height = top_h + gap + host_rows * (thumb_h + gap) + 36
    canvas = Image.new("RGB", (width, max(height, top_h + 80)), (247, 247, 244))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f"SP80 Background Inset Review - Calibration Slice {view['level_index']}", fill=(20, 20, 20))
    raw = Image.open(ROOT / view["raw_frame"]["png"]["path"]).convert("RGB")
    group_overlay = Image.open(ROOT / view["pixel_group_overlay"]["path"]).convert("RGB")
    inset_overlay = Image.open(ROOT / view["background_inset_overlay"]["path"]).convert("RGB")
    for idx, (title, image) in enumerate(
        [("raw frame", raw), ("pixel groups", group_overlay), ("orange inset overlay", inset_overlay)]
    ):
        x = gap + idx * (thumb_w + gap)
        y = label_h
        canvas.paste(image, (x, y))
        draw_label_box(draw, (x + 6, y + 6), title, (45, 45, 45))
    draw.text(
        (gap, label_h + thumb_h + 10),
        "Foreground hosts plus attached raw orange background insets",
        fill=(20, 20, 20),
    )
    start_y = label_h + thumb_h + 32
    for idx, shape in enumerate(selected):
        col = idx % cols
        row_idx = idx // cols
        x = gap + col * (thumb_w + gap)
        y = start_y + row_idx * (thumb_h + gap)
        canvas.paste(isolated_shape_with_insets_image(frame, shape, scale), (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def build_background_insets_for_level(
    *,
    level_index: int,
    frame: np.ndarray,
    rows: list[dict[str, Any]],
    out_dir: Path,
    scale: int,
    background_value: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame_shape = tuple(int(v) for v in frame.shape[:2])
    total_pixels = int(frame_shape[0] * frame_shape[1])
    masks_by_group = {row["group_id"]: load_row_mask(row) for row in rows}
    insets: list[dict[str, Any]] = []
    shapes_with_insets: list[dict[str, Any]] = []
    background_palette = PALETTE_NAMES.get(background_value, f"value_{background_value}")
    host_rows = [
        row
        for row in rows
        if row.get("foreground_for_orthogonal_shape") is True and int(row["raw_pixel_value"]) != int(background_value)
    ]
    for host_row in sorted(host_rows, key=component_sort_key):
        host_id = str(host_row["group_id"])
        host_mask = np.asarray(masks_by_group[host_id], dtype=np.uint8)
        host_bbox = host_row.get("bbox_xyxy")
        if host_bbox is None:
            continue
        bbox_mask = bbox_rect_mask(frame_shape, host_bbox)
        candidate = ((np.asarray(frame) == int(background_value)) & (bbox_mask > 0)).astype(np.uint8)
        inset_components = connected_components_mask(candidate, connectivity=4)
        host_insets: list[dict[str, Any]] = []
        inset_union = np.zeros(frame_shape, dtype=np.uint8)
        for component in inset_components:
            touch_count = orthogonal_touch_count(host_mask, component)
            if touch_count <= 0:
                continue
            inset_index = len(host_insets)
            inset_id = f"bgin_{host_id}_i{inset_index:03d}"
            geometry = component_geometry(component)
            bbox_touches = bbox_edge_touches(component, host_bbox)
            relation = "edge_open_inset" if any(bbox_touches.values()) else "enclosed_hole"
            inset_mask_meta = save_mask(
                component,
                npy_path=out_dir / "background_inset_masks" / f"level_{level_index:02d}" / f"{inset_id}.npy",
                png_path=out_dir / "background_inset_masks" / f"level_{level_index:02d}" / f"{inset_id}.png",
                scale=scale,
            )
            row = {
                "schema": BACKGROUND_INSET_SCHEMA,
                "game": "sp80_liquid_pour",
                "calibration_slice_index": int(level_index),
                "level_index": int(level_index),
                "background_inset_id": inset_id,
                "inset_source": "raw_frame_background_component_inside_host_bbox",
                "evidence_status": "raw_background_inset_hypothesis",
                "raw_pixels_are_observation_ground_truth": True,
                "raw_pixel_value": int(background_value),
                "palette_name": background_palette,
                "palette_rgb": [int(v) for v in PALETTE[int(background_value)].tolist()],
                "connectivity": 4,
                "area_pixels": int(np.count_nonzero(component)),
                "area_fraction": float(np.count_nonzero(component) / max(1, total_pixels)),
                "bbox_xyxy": geometry["bbox_xyxy"],
                "centroid_xy": geometry["centroid_xy"],
                "width": geometry["width"],
                "height": geometry["height"],
                "host_group_id": host_id,
                "host_raw_pixel_value": int(host_row["raw_pixel_value"]),
                "host_palette_name": str(host_row["palette_name"]),
                "host_bbox_xyxy": host_bbox,
                "host_area_pixels": int(host_row["area_pixels"]),
                "host_orthogonal_touch_edges": int(touch_count),
                "touches_host_orthogonally": True,
                "host_bbox_edge_touches": bbox_touches,
                "touches_host_bbox_edge": any(bbox_touches.values()),
                "inset_relation": relation,
                "inset_mask": inset_mask_meta,
                "sam_status": "not_run",
                "controlled_labels_used": False,
                "engine_tags_used": False,
                "semantic_label": "unknown",
                "semantic_label_status": "not_semantic_label",
                "admitted_for_control_truth": False,
                "admitted_for_goal_condition": False,
                "admitted_for_contra_object_view": False,
                "route_or_replay_used": False,
                "human_trace_used": False,
                "coordinate_script_used": False,
            }
            host_insets.append(row)
            inset_union = np.maximum(inset_union, component.astype(np.uint8))
        if not host_insets:
            continue
        insets.extend(host_insets)
        full_mask = np.maximum(host_mask.astype(np.uint8), inset_union.astype(np.uint8))
        shape_id = f"shapein_{host_id}"
        geometry = component_geometry(full_mask)
        inset_union_meta = save_mask(
            inset_union,
            npy_path=out_dir
            / "background_inset_unions"
            / f"level_{level_index:02d}"
            / f"{host_id}_background_insets.npy",
            png_path=out_dir
            / "background_inset_unions"
            / f"level_{level_index:02d}"
            / f"{host_id}_background_insets.png",
            scale=scale,
        )
        full_mask_meta = save_mask(
            full_mask,
            npy_path=out_dir
            / "shape_with_background_insets_masks"
            / f"level_{level_index:02d}"
            / f"{shape_id}.npy",
            png_path=out_dir
            / "shape_with_background_insets_masks"
            / f"level_{level_index:02d}"
            / f"{shape_id}.png",
            scale=scale,
        )
        relation_counts = Counter(str(row["inset_relation"]) for row in host_insets)
        shapes_with_insets.append(
            {
                "schema": SHAPE_WITH_BACKGROUND_INSETS_SCHEMA,
                "game": "sp80_liquid_pour",
                "calibration_slice_index": int(level_index),
                "level_index": int(level_index),
                "shape_with_background_insets_id": shape_id,
                "shape_source": "raw_frame_foreground_host_plus_background_insets",
                "complete_shape_definition": "foreground host mask plus 4-connected raw background components inside the host bbox that touch the host orthogonally",
                "evidence_status": "raw_shape_with_background_insets",
                "raw_pixels_are_observation_ground_truth": True,
                "host_group_id": host_id,
                "host_raw_pixel_value": int(host_row["raw_pixel_value"]),
                "host_palette_name": str(host_row["palette_name"]),
                "host_bbox_xyxy": host_bbox,
                "host_area_pixels": int(host_row["area_pixels"]),
                "background_raw_pixel_value": int(background_value),
                "background_palette_name": background_palette,
                "background_inset_count": len(host_insets),
                "background_inset_ids": [row["background_inset_id"] for row in host_insets],
                "background_inset_area_pixels": int(np.count_nonzero(inset_union)),
                "background_inset_relation_counts": dict(sorted(relation_counts.items())),
                "area_pixels": int(np.count_nonzero(full_mask)),
                "area_fraction": float(np.count_nonzero(full_mask) / max(1, total_pixels)),
                "bbox_xyxy": geometry["bbox_xyxy"],
                "centroid_xy": geometry["centroid_xy"],
                "width": geometry["width"],
                "height": geometry["height"],
                "raw_pixel_value_histogram": value_histogram(frame, full_mask),
                "palette_histogram": palette_histogram(frame, full_mask),
                "background_insets_union_mask": inset_union_meta,
                "shape_with_background_insets_mask": full_mask_meta,
                "sam_status": "not_run",
                "controlled_labels_used": False,
                "engine_tags_used": False,
                "semantic_label": "unknown",
                "semantic_label_status": "not_semantic_label",
                "admitted_for_control_truth": False,
                "admitted_for_goal_condition": False,
                "admitted_for_contra_object_view": False,
                "route_or_replay_used": False,
                "human_trace_used": False,
                "coordinate_script_used": False,
            }
        )
    return insets, shapes_with_insets


def build_orthogonal_shapes_for_level(
    *,
    level_index: int,
    frame: np.ndarray,
    rows: list[dict[str, Any]],
    out_dir: Path,
    scale: int,
    shape_connectivity: int,
    large_component_area_fraction_min: float,
    edge_component_area_fraction_min: float,
) -> list[dict[str, Any]]:
    frame_shape = tuple(int(v) for v in frame.shape[:2])
    masks_by_group = {row["group_id"]: load_row_mask(row) for row in rows}
    foreground = np.zeros(frame_shape, dtype=np.uint8)
    included_group_ids: set[str] = set()
    for row in rows:
        reason = shape_exclusion_reason(
            row,
            frame_shape=frame_shape,
            large_component_area_fraction_min=large_component_area_fraction_min,
            edge_component_area_fraction_min=edge_component_area_fraction_min,
        )
        row["orthogonal_shape_capture_status"] = "candidate"
        row["foreground_for_orthogonal_shape"] = reason is None
        row["orthogonal_shape_exclusion_reason"] = reason
        row["orthogonal_shape_id"] = None
        if reason is None:
            group_id = str(row["group_id"])
            included_group_ids.add(group_id)
            foreground = np.maximum(foreground, masks_by_group[group_id].astype(np.uint8))
    shape_masks = connected_components_mask(foreground, connectivity=shape_connectivity)
    shapes: list[dict[str, Any]] = []
    total_pixels = int(frame_shape[0] * frame_shape[1])
    for shape_idx, shape_mask in enumerate(shape_masks):
        shape_id = f"shape_l{level_index:02d}_o{shape_idx:03d}"
        member_rows: list[dict[str, Any]] = []
        for row in rows:
            group_id = str(row["group_id"])
            if group_id not in included_group_ids:
                continue
            if bool(np.any((shape_mask > 0) & (masks_by_group[group_id] > 0))):
                member_rows.append(row)
                row["orthogonal_shape_id"] = shape_id
        member_rows.sort(key=component_sort_key)
        geometry = component_geometry(shape_mask)
        shape_boundary = boundary_mask(shape_mask)
        area = int(np.count_nonzero(shape_mask))
        shape_mask_meta = save_mask(
            shape_mask,
            npy_path=out_dir / "orthogonal_shape_masks" / f"level_{level_index:02d}" / f"{shape_id}.npy",
            png_path=out_dir / "orthogonal_shape_masks" / f"level_{level_index:02d}" / f"{shape_id}.png",
            scale=scale,
        )
        boundary_mask_meta = save_mask(
            shape_boundary,
            npy_path=out_dir / "orthogonal_shape_boundaries" / f"level_{level_index:02d}" / f"{shape_id}_boundary.npy",
            png_path=out_dir / "orthogonal_shape_boundaries" / f"level_{level_index:02d}" / f"{shape_id}_boundary.png",
            scale=scale,
        )
        adjacency_edges = []
        for idx_a, row_a in enumerate(member_rows):
            for row_b in member_rows[idx_a + 1 :]:
                touch_count = orthogonal_touch_count(masks_by_group[row_a["group_id"]], masks_by_group[row_b["group_id"]])
                if touch_count > 0:
                    adjacency_edges.append(
                        {
                            "a": row_a["group_id"],
                            "b": row_b["group_id"],
                            "orthogonal_touch_edges": int(touch_count),
                        }
                    )
        hist = value_histogram(frame, shape_mask)
        shapes.append(
            {
                "schema": ORTHOGONAL_SHAPE_SCHEMA,
                "game": "sp80_liquid_pour",
                "calibration_slice_index": int(level_index),
                "level_index": int(level_index),
                "shape_id": shape_id,
                "shape_source": "raw_frame_orthogonal_foreground_component",
                "complete_shape_definition": "single 4-neighbor connected foreground component after recorded background exclusions",
                "evidence_status": "raw_orthogonal_shape",
                "raw_pixels_are_observation_ground_truth": True,
                "shape_connectivity": int(shape_connectivity),
                "area_pixels": area,
                "area_fraction": float(area / max(1, total_pixels)),
                "bbox_xyxy": geometry["bbox_xyxy"],
                "centroid_xy": geometry["centroid_xy"],
                "touches_edge": geometry["touches_edge"],
                "edge_touch_count": geometry["edge_touch_count"],
                "edge_touches": geometry["edge_touches"],
                "width": geometry["width"],
                "height": geometry["height"],
                "orthogonal_perimeter_edges": orthogonal_perimeter_edges(shape_mask),
                "boundary_pixel_count": int(np.count_nonzero(shape_boundary)),
                "raw_pixel_value_histogram": hist,
                "palette_histogram": palette_histogram(frame, shape_mask),
                "shape_kind": "single_value_complete_shape" if len(hist) == 1 else "multi_value_complete_shape",
                "member_pixel_group_ids": [row["group_id"] for row in member_rows],
                "member_raw_pixel_values": sorted({int(row["raw_pixel_value"]) for row in member_rows}),
                "orthogonal_adjacency_edges": adjacency_edges,
                "shape_mask": shape_mask_meta,
                "shape_boundary_mask": boundary_mask_meta,
                "object_definition": "shape_mask is the in-place foreground object component; background pixels are context only and are not unioned into this orthogonal isolation",
                "background_context_is_object": False,
                "background_fill_used": False,
                "sam_status": "not_run",
                "controlled_labels_used": False,
                "engine_tags_used": False,
                "semantic_label": "unknown",
                "semantic_label_status": "not_semantic_label",
                "admitted_for_control_truth": False,
                "admitted_for_goal_condition": False,
                "admitted_for_contra_object_view": False,
                "route_or_replay_used": False,
                "human_trace_used": False,
                "coordinate_script_used": False,
            }
        )
    return shapes


def build_rows_for_level(
    *,
    level_index: int,
    frame: np.ndarray,
    out_dir: Path,
    scale: int,
    connectivity: int,
    min_area: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_pixels = int(frame.shape[0] * frame.shape[1])
    raw_components = connected_components(frame, connectivity=connectivity)
    kept_by_value: Counter[int] = Counter()
    for component in raw_components:
        mask = component["mask"]
        area = int(np.count_nonzero(mask))
        if area < min_area:
            continue
        value = int(component["raw_pixel_value"])
        component_idx = kept_by_value[value]
        kept_by_value[value] += 1
        group_id = f"px_l{level_index:02d}_v{value:02d}_g{component_idx:03d}"
        geometry = component_geometry(mask)
        area_fraction = float(area / max(1, total_pixels))
        mask_meta = save_mask(
            mask,
            npy_path=out_dir / "pixel_group_masks" / f"level_{level_index:02d}" / f"{group_id}.npy",
            png_path=out_dir / "pixel_group_masks" / f"level_{level_index:02d}" / f"{group_id}.png",
            scale=scale,
        )
        rows.append(
            {
                "schema": PIXEL_GROUP_SCHEMA,
                "game": "sp80_liquid_pour",
                "calibration_slice_index": int(level_index),
                "level_index": int(level_index),
                "group_id": group_id,
                "group_source": "raw_frame_exact_value_connected_component",
                "evidence_status": "raw_pixel_group",
                "raw_pixels_are_observation_ground_truth": True,
                "raw_pixel_value": value,
                "palette_name": PALETTE_NAMES.get(value, f"value_{value}"),
                "palette_rgb": [int(v) for v in PALETTE[value].tolist()],
                "connectivity": int(connectivity),
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
                "pixel_mask": mask_meta,
                "sam_status": "not_run",
                "controlled_labels_used": False,
                "engine_tags_used": False,
                "admitted_for_control_truth": False,
                "admitted_for_goal_condition": False,
                "admitted_for_contra_object_view": False,
                "route_or_replay_used": False,
                "human_trace_used": False,
                "coordinate_script_used": False,
            }
        )
    return rows


def run_level(
    level_index: int,
    out_dir: Path,
    scale: int,
    connectivity: int,
    min_area: int,
    shape_capture: str,
    shape_connectivity: int,
    shape_background_area_fraction: float,
    shape_edge_background_area_fraction: float,
    background_inset_capture: str,
    background_value: int,
) -> dict[str, Any]:
    controller = SP80BaseController(level_index=level_index)
    _game, obs = controller.new_game()
    frame = np.asarray(obs.frame[-1])
    raw_png = save_frame(frame, out_dir / "frames" / f"level_{level_index:02d}_raw.png", scale)
    raw_grid = save_raw_grid(frame, out_dir / "frames" / f"level_{level_index:02d}_raw.npy")
    rows = build_rows_for_level(
        level_index=level_index,
        frame=frame,
        out_dir=out_dir,
        scale=scale,
        connectivity=connectivity,
        min_area=min_area,
    )
    overlay = draw_component_overlay(frame, rows, out_dir / "pixel_group_overlays" / f"level_{level_index:02d}.png", scale)
    control_isolation_packet = build_control_isolation_packet(
        frame,
        game="sp80_liquid_pour",
        phase="calibration_view",
        case_id=f"calibration_slice_{level_index:02d}",
        connectivity=connectivity,
        min_area_pixels=min_area,
    )
    inferred_control_background = control_isolation_packet.get("background_raw_pixel_value")
    control_background_value = int(background_value if inferred_control_background is None else inferred_control_background)
    control_overlay = draw_control_inplace_isolation_overlay(
        frame,
        rows,
        out_dir / "control_inplace_isolation_overlays" / f"level_{level_index:02d}.png",
        scale,
        control_background_value,
    )
    shapes: list[dict[str, Any]] = []
    background_insets: list[dict[str, Any]] = []
    shapes_with_insets: list[dict[str, Any]] = []
    view = {
        "schema": "codexlab.sp80_pixel_group_view.v1",
        "game": "sp80_liquid_pour",
        "calibration_slice_index": int(level_index),
        "level_index": int(level_index),
        "raw_frame": {"png": raw_png, "raw_grid": raw_grid},
        "pixel_group_overlay": overlay,
        "control_inplace_isolation_overlay": control_overlay,
        "control_isolation_packet": control_isolation_packet,
        "control_inplace_isolation_definition": (
            "full game frame with every non-background raw component outlined in place; "
            "background context is inferred from the raw frame, so insets remain visible as negative space"
        ),
        "pixel_group_count": len(rows),
        "pixel_group_ids": [row["group_id"] for row in rows],
        "pixel_value_counts": {str(k): int(v) for k, v in sorted(Counter(int(v) for v in frame.flatten()).items())},
    }
    if shape_capture == "orthogonal":
        shapes = build_orthogonal_shapes_for_level(
            level_index=level_index,
            frame=frame,
            rows=rows,
            out_dir=out_dir,
            scale=scale,
            shape_connectivity=shape_connectivity,
            large_component_area_fraction_min=shape_background_area_fraction,
            edge_component_area_fraction_min=shape_edge_background_area_fraction,
        )
        shape_overlay = draw_shape_overlay(
            frame,
            shapes,
            out_dir / "orthogonal_shape_overlays" / f"level_{level_index:02d}.png",
            scale,
        )
        view["orthogonal_shape_overlay"] = shape_overlay
        view["orthogonal_shape_count"] = len(shapes)
        view["orthogonal_shape_ids"] = [shape["shape_id"] for shape in shapes]
        view["orthogonal_shape_foreground_pixel_group_ids"] = [
            row["group_id"] for row in rows if row.get("foreground_for_orthogonal_shape") is True
        ]
        excluded: dict[str, list[str]] = {}
        for row in rows:
            reason = row.get("orthogonal_shape_exclusion_reason")
            if reason:
                excluded.setdefault(str(reason), []).append(row["group_id"])
        view["orthogonal_shape_excluded_pixel_group_ids_by_reason"] = excluded
        shape_contact = make_shape_contact(
            frame=frame,
            view=view,
            shapes=shapes,
            path=out_dir / "orthogonal_shape_review" / "slices" / f"slice_{level_index:02d}_orthogonal_shapes.png",
            scale=scale,
        )
        view["orthogonal_shape_review"] = shape_contact
    else:
        for row in rows:
            row["orthogonal_shape_capture_status"] = "not_run"
            row["foreground_for_orthogonal_shape"] = None
            row["orthogonal_shape_exclusion_reason"] = None
            row["orthogonal_shape_id"] = None
    if background_inset_capture == "orange" and shape_capture == "orthogonal":
        background_insets, shapes_with_insets = build_background_insets_for_level(
            level_index=level_index,
            frame=frame,
            rows=rows,
            out_dir=out_dir,
            scale=scale,
            background_value=background_value,
        )
        inset_overlay = draw_background_inset_overlay(
            frame,
            shapes_with_insets,
            out_dir / "background_inset_overlays" / f"level_{level_index:02d}.png",
            scale,
        )
        view["background_inset_overlay"] = inset_overlay
        view["background_inset_count"] = len(background_insets)
        view["shape_with_background_insets_count"] = len(shapes_with_insets)
        view["background_inset_ids"] = [row["background_inset_id"] for row in background_insets]
        view["shape_with_background_insets_ids"] = [
            row["shape_with_background_insets_id"] for row in shapes_with_insets
        ]
        view["background_inset_host_group_ids"] = [row["host_group_id"] for row in shapes_with_insets]
        inset_contact = make_background_inset_contact(
            frame=frame,
            view=view,
            shape_with_insets=shapes_with_insets,
            path=out_dir / "background_inset_review" / "slices" / f"slice_{level_index:02d}_background_insets.png",
            scale=scale,
        )
        view["background_inset_review"] = inset_contact
    else:
        view["background_inset_count"] = 0
        view["shape_with_background_insets_count"] = 0
    contact = make_slice_contact(
        frame=frame,
        view=view,
        rows=rows,
        path=out_dir / "pixel_group_review" / "slices" / f"slice_{level_index:02d}_pixel_groups.png",
        scale=scale,
    )
    view["pixel_group_review"] = contact
    write_json(out_dir / "pixel_group_views" / f"level_{level_index:02d}.json", view)
    return {
        "view": view,
        "rows": rows,
        "shapes": shapes,
        "background_insets": background_insets,
        "shapes_with_insets": shapes_with_insets,
    }


def build_index(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    shapes: list[dict[str, Any]],
    shapes_with_insets: list[dict[str, Any]],
    *,
    index_dir: Path,
) -> str:
    rows_by_level: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_level.setdefault(int(row["level_index"]), []).append(row)
    shapes_by_level: dict[int, list[dict[str, Any]]] = {}
    for shape in shapes:
        shapes_by_level.setdefault(int(shape["level_index"]), []).append(shape)
    shapes_with_insets_by_level: dict[int, list[dict[str, Any]]] = {}
    for shape in shapes_with_insets:
        shapes_with_insets_by_level.setdefault(int(shape["level_index"]), []).append(shape)
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 Pixel Group Viewer</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f4;color:#191919}",
        "section{margin-bottom:36px}",
        "img{image-rendering:pixelated;border:1px solid #aaa;background:white;max-width:100%}",
        "table{border-collapse:collapse;font-size:13px;margin-top:12px}",
        "td,th{border:1px solid #ccc;padding:5px 7px;text-align:left;vertical-align:top}",
        ".pair{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}",
        ".note{max-width:980px;line-height:1.35}",
        "code{background:#eee;padding:1px 3px}",
        "</style></head><body>",
        "<h1>SP80 Pixel Group Viewer</h1>",
        "<p class='note'>This is a viewer-first raw-pixel pass. Rows are exact-value connected components from the raw frame. "
        "No SAM masks, controlled engine labels, human replay, or route scripts are used to create these groups.</p>",
        f"<p><b>Manifest:</b> <code>{html.escape(manifest['manifest_json'])}</code></p>",
    ]
    for view in manifest["views"]:
        level_index = int(view["level_index"])
        level_rows = rows_by_level.get(level_index, [])
        body.extend(
            [
                f"<section><h2>Calibration Slice {level_index}</h2>",
                "<div class='pair'>",
                f"<div><h3>Raw Frame</h3><img src='{rel_for_html(view['raw_frame']['png']['path'], index_dir)}'></div>",
                f"<div><h3>Control Map Isolation</h3><img src='{rel_for_html(view['control_inplace_isolation_overlay']['path'], index_dir)}'></div>",
                f"<div><h3>Review Sheet</h3><a href='{rel_for_html(view['pixel_group_review']['path'], index_dir)}'>open sheet</a></div>",
                "</div>",
            ]
        )
        if "orthogonal_shape_overlay" in view:
            body.extend(
                [
                    "<div class='pair'>",
                    f"<div><h3>Debug: Orthogonal Sheet</h3><a href='{rel_for_html(view['orthogonal_shape_review']['path'], index_dir)}'>open sheet</a></div>",
                    f"<div><h3>Debug: Pixel Group Overlay</h3><a href='{rel_for_html(view['pixel_group_overlay']['path'], index_dir)}'>open image</a></div>",
                    f"<div><h3>Debug: Orthogonal Shape Overlay</h3><a href='{rel_for_html(view['orthogonal_shape_overlay']['path'], index_dir)}'>open image</a></div>",
                    "</div>",
                ]
            )
        if "background_inset_overlay" in view:
            body.extend(
                [
                    "<div class='pair'>",
                    f"<div><h3>Background Inset Overlay</h3><img src='{rel_for_html(view['background_inset_overlay']['path'], index_dir)}'></div>",
                    f"<div><h3>Background Inset Sheet</h3><a href='{rel_for_html(view['background_inset_review']['path'], index_dir)}'>open sheet</a></div>",
                    "</div>",
                ]
            )
        body.extend(
            [
                "<table><thead><tr>"
                "<th>group</th><th>value</th><th>area</th><th>bbox</th><th>centroid</th><th>geometry</th><th>shape</th><th>mask</th>"
                "</tr></thead><tbody>",
            ]
        )
        for row in sorted(level_rows, key=component_sort_key):
            shape_id = row.get("orthogonal_shape_id") or row.get("orthogonal_shape_exclusion_reason") or ""
            body.append(
                "<tr>"
                f"<td>{html.escape(row['group_id'])}</td>"
                f"<td>{row['raw_pixel_value']} {html.escape(row['palette_name'])}</td>"
                f"<td>{row['area_pixels']}</td>"
                f"<td><code>{html.escape(json.dumps(row['bbox_xyxy']))}</code></td>"
                f"<td><code>{html.escape(json.dumps(row['centroid_xy']))}</code></td>"
                f"<td>{html.escape(row['geometry_hint'])}</td>"
                f"<td>{html.escape(str(shape_id))}</td>"
                f"<td><a href='{rel_for_html(row['pixel_mask']['png']['path'], index_dir)}'>png</a></td>"
                "</tr>"
            )
        body.extend(["</tbody></table>"])
        if shapes:
            level_shapes = shapes_by_level.get(level_index, [])
            body.extend(
                [
                    "<table><thead><tr>"
                    "<th>shape</th><th>area</th><th>bbox</th><th>values</th><th>members</th>"
                    "<th>adjacency edges</th><th>background fill</th><th>mask</th>"
                    "</tr></thead><tbody>",
                ]
            )
            for shape in level_shapes:
                values = ", ".join(
                    f"{item['raw_pixel_value']} {item['palette_name']}:{item['pixel_count']}"
                    for item in shape["palette_histogram"]
                )
                body.append(
                    "<tr>"
                    f"<td>{html.escape(shape['shape_id'])}</td>"
                    f"<td>{shape['area_pixels']}</td>"
                    f"<td><code>{html.escape(json.dumps(shape['bbox_xyxy']))}</code></td>"
                    f"<td>{html.escape(values)}</td>"
                    f"<td>{len(shape['member_pixel_group_ids'])}</td>"
                    f"<td>{len(shape['orthogonal_adjacency_edges'])}</td>"
                    f"<td>{html.escape(str(shape.get('background_fill_used')))}</td>"
                    f"<td><a href='{rel_for_html(shape['shape_mask']['png']['path'], index_dir)}'>png</a></td>"
                    "</tr>"
                )
            body.extend(["</tbody></table>"])
        if shapes_with_insets:
            level_shapes_with_insets = shapes_with_insets_by_level.get(level_index, [])
            body.extend(
                [
                    "<table><thead><tr>"
                    "<th>host</th><th>host value</th><th>host area</th><th>orange inset area</th>"
                    "<th>complete area</th><th>bbox</th><th>relations</th><th>mask</th>"
                    "</tr></thead><tbody>",
                ]
            )
            for shape in sorted(level_shapes_with_insets, key=shape_with_insets_sort_key):
                relations = ", ".join(
                    f"{key}:{value}" for key, value in sorted(shape["background_inset_relation_counts"].items())
                )
                body.append(
                    "<tr>"
                    f"<td>{html.escape(shape['host_group_id'])}</td>"
                    f"<td>{shape['host_raw_pixel_value']} {html.escape(shape['host_palette_name'])}</td>"
                    f"<td>{shape['host_area_pixels']}</td>"
                    f"<td>{shape['background_inset_area_pixels']}</td>"
                    f"<td>{shape['area_pixels']}</td>"
                    f"<td><code>{html.escape(json.dumps(shape['bbox_xyxy']))}</code></td>"
                    f"<td>{html.escape(relations)}</td>"
                    f"<td><a href='{rel_for_html(shape['shape_with_background_insets_mask']['png']['path'], index_dir)}'>png</a></td>"
                    "</tr>"
                )
            body.extend(["</tbody></table>"])
        body.extend(["</section>"])
    body.extend(["</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="0", help="'all' or comma-separated calibration slice indexes")
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_pixel_group_viewer"))
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=4)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--shape-capture", choices=["off", "orthogonal"], default="orthogonal")
    parser.add_argument("--shape-connectivity", type=int, choices=[4], default=4)
    parser.add_argument("--shape-background-area-fraction", type=float, default=0.30)
    parser.add_argument("--shape-edge-background-area-fraction", type=float, default=0.01)
    parser.add_argument("--background-inset-capture", choices=["off", "orange"], default="off")
    parser.add_argument("--background-value", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = max(1, int(args.scale))
    connectivity = int(args.connectivity)
    min_area = max(1, int(args.min_area))
    shape_capture = str(args.shape_capture)
    shape_connectivity = int(args.shape_connectivity)
    shape_background_area_fraction = max(0.0, min(1.0, float(args.shape_background_area_fraction)))
    shape_edge_background_area_fraction = max(0.0, min(1.0, float(args.shape_edge_background_area_fraction)))
    background_inset_capture = str(args.background_inset_capture)
    background_value = int(args.background_value)
    level_count = detect_level_count()
    levels = parse_levels(args.levels, level_count)
    all_rows: list[dict[str, Any]] = []
    all_shapes: list[dict[str, Any]] = []
    all_background_insets: list[dict[str, Any]] = []
    all_shapes_with_insets: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for level_index in levels:
        result = run_level(
            level_index,
            out_dir,
            scale,
            connectivity,
            min_area,
            shape_capture,
            shape_connectivity,
            shape_background_area_fraction,
            shape_edge_background_area_fraction,
            background_inset_capture,
            background_value,
        )
        views.append(result["view"])
        all_rows.extend(result["rows"])
        all_shapes.extend(result["shapes"])
        all_background_insets.extend(result["background_insets"])
        all_shapes_with_insets.extend(result["shapes_with_insets"])
    rows_path = out_dir / "pixel_groups.jsonl"
    write_jsonl(rows_path, all_rows)
    shapes_path = out_dir / "orthogonal_shapes.jsonl"
    if shape_capture == "orthogonal":
        write_jsonl(shapes_path, all_shapes)
    background_insets_path = out_dir / "background_insets.jsonl"
    shapes_with_insets_path = out_dir / "shape_with_background_insets.jsonl"
    if background_inset_capture == "orange" and shape_capture == "orthogonal":
        write_jsonl(background_insets_path, all_background_insets)
        write_jsonl(shapes_with_insets_path, all_shapes_with_insets)
    value_counts = Counter(int(row["raw_pixel_value"]) for row in all_rows)
    geometry_counts = Counter(str(row["geometry_hint"]) for row in all_rows)
    shape_kind_counts = Counter(str(shape["shape_kind"]) for shape in all_shapes)
    inset_host_counts = Counter(str(row["host_palette_name"]) for row in all_shapes_with_insets)
    inset_relation_counts = Counter(str(row["inset_relation"]) for row in all_background_insets)
    yellow_host_count = sum(1 for row in all_rows if int(row["raw_pixel_value"]) == 11 and row.get("foreground_for_orthogonal_shape") is True)
    yellow_host_with_inset_count = sum(
        1 for row in all_shapes_with_insets if int(row["host_raw_pixel_value"]) == 11 and int(row["background_inset_count"]) > 0
    )
    inferred_background_values = [
        int(((view.get("control_isolation_packet") or {}).get("background_raw_pixel_value")))
        for view in views
        if ((view.get("control_isolation_packet") or {}).get("background_raw_pixel_value")) is not None
    ]
    inferred_background_counts = Counter(inferred_background_values)
    shared_background_value = (
        inferred_background_values[0]
        if inferred_background_values and all(value == inferred_background_values[0] for value in inferred_background_values)
        else None
    )
    manifest = {
        "schema": PIXEL_GROUP_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "sp80_raw_pixel_group_viewer",
        "condition_natural_language": "Group exact raw SP80 frame pixel values into connected components before SAM or controlled labels are used.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "game": "sp80_liquid_pour",
        "detected_calibration_slice_count": level_count,
        "requested_calibration_slices": levels,
        "raw_pixels_are_observation_ground_truth": True,
        "group_source": "raw_frame_exact_value_connected_component",
        "connectivity": connectivity,
        "min_area": min_area,
        "scale": scale,
        "pixel_group_count": len(all_rows),
        "pixel_group_counts_by_value": {str(k): int(v) for k, v in sorted(value_counts.items())},
        "geometry_hint_counts": dict(sorted(geometry_counts.items())),
        "orthogonal_shape_capture_status": "exported" if shape_capture == "orthogonal" else "not_run",
        "orthogonal_shape_schema": ORTHOGONAL_SHAPE_SCHEMA,
        "orthogonal_shape_count": len(all_shapes),
        "orthogonal_shape_kind_counts": dict(sorted(shape_kind_counts.items())),
        "orthogonal_shape_source": "raw_frame_orthogonal_foreground_component",
        "control_inplace_isolation_status": "exported",
        "contra_control_isolation_schema": CONTROL_ISOLATION_SCHEMA,
        "control_inplace_isolation_view_count": len(views),
        "control_inplace_isolation_source": CONTROL_ISOLATION_SOURCE,
        "control_inplace_isolation_definition": "full game frame with every non-background raw component outlined in place; background context is inferred from the raw frame and insets remain visible as negative space",
        "control_inplace_background_policy": BACKGROUND_SELECTION_POLICY,
        "control_inplace_background_raw_pixel_value": shared_background_value,
        "control_inplace_background_raw_pixel_value_counts": {
            str(k): int(v) for k, v in sorted(inferred_background_counts.items())
        },
        "orthogonal_shape_foreground_policy": {
            "name": "exclude_large_and_edge_background_components",
            "complete_shape_definition": "single 4-neighbor connected foreground component after recorded background exclusions",
            "connectivity": shape_connectivity,
            "large_component_area_fraction_min": shape_background_area_fraction,
            "edge_component_area_fraction_min": shape_edge_background_area_fraction,
            "excluded_component_reasons": ["large_field_component", "edge_frame_or_wall_component"],
        },
        "background_inset_capture_status": "exported"
        if background_inset_capture == "orange" and shape_capture == "orthogonal"
        else "not_run",
        "background_inset_schema": BACKGROUND_INSET_SCHEMA,
        "background_inset_count": len(all_background_insets),
        "background_inset_relation_counts": dict(sorted(inset_relation_counts.items())),
        "background_inset_source": "raw_frame_background_component_inside_host_bbox",
        "background_inset_policy": {
            "name": "orange_components_inside_foreground_host_bbox_touching_host",
            "background_raw_pixel_value": int(background_value),
            "background_palette_name": PALETTE_NAMES.get(background_value, f"value_{background_value}"),
            "host_source": "foreground_for_orthogonal_shape_pixel_groups",
            "candidate_window": "host_bbox_xyxy",
            "component_connectivity": 4,
            "required_host_orthogonal_touch_edges_min": 1,
            "relations": ["edge_open_inset", "enclosed_hole"],
        },
        "shape_with_background_insets_schema": SHAPE_WITH_BACKGROUND_INSETS_SCHEMA,
        "shape_with_background_insets_count": len(all_shapes_with_insets),
        "shape_with_background_insets_host_palette_counts": dict(sorted(inset_host_counts.items())),
        "yellow_raw_value_foreground_host_count": yellow_host_count,
        "yellow_raw_value_hosts_with_background_insets_count": yellow_host_with_inset_count,
        "sam_status": "not_run",
        "controlled_labels_used": False,
        "engine_tags_used": False,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "admitted_for_control_truth_count": 0,
        "admitted_for_goal_condition_count": 0,
        "admitted_for_contra_object_view_count": 0,
        "source_game": relpath(SP80_PATH),
        "source_game_sha256": sha256_file(SP80_PATH),
        "source_controller": "codexlab/sp80_base_controller.py",
        "source_controller_sha256": sha256_file(ROOT / "codexlab" / "sp80_base_controller.py"),
        "source_script": "codexlab/sp80_pixel_group_viewer.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_pixel_group_viewer.py"),
        "source_contra": "codexlab/contra_head/contra.py",
        "source_contra_sha256": sha256_file(ROOT / "codexlab" / "contra_head" / "contra.py"),
        "source_contra_modules": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in contra_module_source_paths()
        ],
        "pixel_group_schema": PIXEL_GROUP_SCHEMA,
        "pixel_groups_jsonl": relpath(rows_path),
        "pixel_groups_jsonl_sha256": sha256_file(rows_path),
        "orthogonal_shapes_jsonl": relpath(shapes_path) if shape_capture == "orthogonal" else None,
        "orthogonal_shapes_jsonl_sha256": sha256_file(shapes_path) if shape_capture == "orthogonal" else None,
        "background_insets_jsonl": relpath(background_insets_path)
        if background_inset_capture == "orange" and shape_capture == "orthogonal"
        else None,
        "background_insets_jsonl_sha256": sha256_file(background_insets_path)
        if background_inset_capture == "orange" and shape_capture == "orthogonal"
        else None,
        "shape_with_background_insets_jsonl": relpath(shapes_with_insets_path)
        if background_inset_capture == "orange" and shape_capture == "orthogonal"
        else None,
        "shape_with_background_insets_jsonl_sha256": sha256_file(shapes_with_insets_path)
        if background_inset_capture == "orange" and shape_capture == "orthogonal"
        else None,
        "views": views,
    }
    manifest_path = out_dir / "pixel_group_manifest.json"
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)
    index_path = out_dir / "index.html"
    index_path.write_text(
        build_index(manifest, all_rows, all_shapes, all_shapes_with_insets, index_dir=out_dir),
        encoding="utf-8",
    )
    review_index_path = out_dir / "pixel_group_review" / "index.html"
    review_index_path.parent.mkdir(parents=True, exist_ok=True)
    review_index_path.write_text(
        build_index(manifest, all_rows, all_shapes, all_shapes_with_insets, index_dir=review_index_path.parent),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "exported",
                "manifest": relpath(manifest_path),
                "index_html": relpath(index_path),
                "review_index_html": relpath(review_index_path),
                "pixel_groups_jsonl": relpath(rows_path),
                "pixel_group_count": len(all_rows),
                "background_insets_jsonl": relpath(background_insets_path)
                if background_inset_capture == "orange" and shape_capture == "orthogonal"
                else None,
                "background_inset_count": len(all_background_insets),
                "shape_with_background_insets_count": len(all_shapes_with_insets),
                "requested_calibration_slices": levels,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
