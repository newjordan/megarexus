#!/usr/bin/env python3
"""Render Ghost relationship mechanics overlays from recorded run artifacts."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PALETTE = np.array(
    [
        [255, 255, 255],
        [204, 204, 204],
        [153, 153, 153],
        [102, 102, 102],
        [51, 51, 51],
        [0, 0, 0],
        [229, 58, 163],
        [255, 123, 204],
        [249, 60, 49],
        [30, 147, 255],
        [136, 216, 241],
        [255, 220, 0],
        [255, 133, 27],
        [146, 18, 49],
        [79, 204, 48],
        [163, 86, 214],
    ],
    dtype=np.uint8,
)


def repo_rel(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def raw_image(raw_grid: np.ndarray, scale: int) -> Image.Image:
    arr = np.asarray(raw_grid, dtype=np.int64).clip(0, 15)
    rgb = PALETTE[arr]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return Image.fromarray(rgb, mode="RGB")


def bbox_center(bbox: Any, scale: int) -> tuple[int, int] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
    return (int(round(((x0 + x1 + 1.0) / 2.0) * scale)), int(round(((y0 + y1 + 1.0) / 2.0) * scale)))


def scaled_rect(bbox: Any, scale: int) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, list) or len(bbox) < 4:
        return None
    x0, y0, x1, y1 = [int(round(float(value))) for value in bbox[:4]]
    return (x0 * scale, y0 * scale, (x1 + 1) * scale - 1, (y1 + 1) * scale - 1)


def component_boundary_pixels(raw_grid: np.ndarray, *, background_values: set[int]) -> list[tuple[int, int]]:
    arr = np.asarray(raw_grid, dtype=np.int64)
    h, w = arr.shape
    visited = np.zeros((h, w), dtype=bool)
    boundary: list[tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if visited[y, x] or int(arr[y, x]) in background_values:
                continue
            color = int(arr[y, x])
            queue: deque[tuple[int, int]] = deque([(x, y)])
            visited[y, x] = True
            while queue:
                cx, cy = queue.popleft()
                is_boundary = False
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = cx + dx, cy + dy
                    if nx < 0 or ny < 0 or nx >= w or ny >= h:
                        is_boundary = True
                        continue
                    if int(arr[ny, nx]) != color:
                        is_boundary = True
                    elif not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                if is_boundary:
                    boundary.append((cx, cy))
    return boundary


def terminal_packet(ghost: dict[str, Any], packets: list[dict[str, Any]]) -> dict[str, Any]:
    terminal_case_id = ((ghost.get("completion_proof") or {}).get("terminal_packet_case_id"))
    for packet in packets:
        if packet.get("case_id") == terminal_case_id:
            return packet
    return packets[-1]


def exact_terminal_lineup_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    contra_packet = packet.get("after_contra_packet") if isinstance(packet.get("after_contra_packet"), dict) else packet
    for statecon in (contra_packet.get("state_condition_hypotheses") or []):
        portband = statecon.get("portband_hypotheses") if isinstance(statecon, dict) else {}
        matrix = portband.get("lineup_matrix") if isinstance(portband, dict) else {}
        for row in matrix.get("rows") or []:
            if not isinstance(row, dict):
                continue
            row_id = str(row.get("lineup_matrix_row_id") or "")
            if not row_id or row_id in seen:
                continue
            score_terms = row.get("score_terms") if isinstance(row.get("score_terms"), dict) else {}
            if row.get("confidence_score") == 1.0 and score_terms.get("exact_alignment") is True:
                seen.add(row_id)
                rows.append(row)
    return rows


def draw_overlay(
    *,
    raw_grid: np.ndarray,
    lineup_rows: list[dict[str, Any]],
    ghost: dict[str, Any],
    out_path: Path,
    base_png_path: Path | None = None,
) -> dict[str, Any]:
    if base_png_path is not None:
        board = Image.open(base_png_path).convert("RGBA")
        scale = max(1, board.width // int(raw_grid.shape[1]))
    else:
        scale = 12
        board = raw_image(raw_grid, scale=scale).convert("RGBA")
    overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    green = (20, 255, 90, 235)
    dark_green = (0, 148, 68, 245)
    cyan_green = (65, 255, 190, 220)
    yellow = (255, 240, 80, 230)

    for x, y in component_boundary_pixels(raw_grid, background_values={12}):
        draw.rectangle((x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1), outline=green, width=2)

    for idx, row in enumerate(lineup_rows):
        source = row.get("source_bbox_xyxy")
        imagined = row.get("imagined_lineup_bbox_xyxy")
        source_rect = scaled_rect(source, scale)
        imagined_rect = scaled_rect(imagined, scale)
        source_center = bbox_center(source, scale)
        imagined_center = bbox_center(imagined, scale)
        if source_rect:
            draw.rectangle(source_rect, outline=yellow, width=4)
        if imagined_rect:
            draw.rectangle(imagined_rect, outline=dark_green, width=4)
        if source_center and imagined_center:
            draw.line((*source_center, *imagined_center), fill=cyan_green, width=4)
            r = 5
            draw.ellipse((imagined_center[0] - r, imagined_center[1] - r, imagined_center[0] + r, imagined_center[1] + r), fill=green)
        if idx < 10 and imagined_rect:
            draw.text((imagined_rect[0] + 4, imagined_rect[1] + 4), str(idx + 1), fill=(0, 45, 20, 255))

    board.alpha_composite(overlay)

    canvas = Image.new("RGB", (max(1500, board.width + 500), max(980, board.height + 180)), (9, 13, 18))
    canvas_draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
        body_font = ImageFont.truetype("DejaVuSans.ttf", 19)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        title_font = body_font = small_font = ImageFont.load_default()
    canvas_draw.text((28, 24), "Ghost Completion Mechanics Overlay", fill=(245, 250, 255), font=title_font)
    canvas_draw.text(
        (28, 64),
        "Green edges: recorded foreground boundaries. Green vectors: terminal exact PortBand lineup signals over recorded solve frame.",
        fill=(200, 214, 230),
        font=body_font,
    )
    canvas.paste(board.convert("RGB"), (28, 110))
    canvas_draw.rectangle((28, 110, 28 + board.width, 110 + board.height), outline=(92, 110, 130), width=2)

    x = 830
    y = 110
    rows = [
        ("ghost_id", ghost.get("ghost_id")),
        ("source_run", ghost.get("source_run_id")),
        ("terminal_case", (ghost.get("completion_proof") or {}).get("terminal_packet_case_id")),
        ("base_image", repo_rel(base_png_path) if base_png_path is not None else "raw_grid_render"),
        ("levels_completed_delta", (ghost.get("completion_proof") or {}).get("levels_completed_delta")),
        ("confirmed_relation_rows", ghost.get("confirmed_relationship_row_count")),
        ("terminal_exact_portband_rows", len(lineup_rows)),
        ("boundary", "review only; not route/action/solve truth"),
    ]
    for key, value in rows:
        canvas_draw.text((x, y), f"{key}: {value}", fill=(220, 235, 250), font=body_font)
        y += 34
    y += 14
    canvas_draw.text((x, y), "Legend", fill=(245, 250, 255), font=title_font)
    y += 44
    legend = [
        ((20, 255, 90), "component edge boundary"),
        ((0, 148, 68), "imagined aligned PortBand target"),
        ((65, 255, 190), "source -> aligned target vector"),
        ((255, 240, 80), "source object bbox"),
    ]
    for color, label in legend:
        canvas_draw.rectangle((x, y + 4, x + 28, y + 24), fill=color)
        canvas_draw.text((x + 40, y), label, fill=(220, 235, 250), font=body_font)
        y += 34
    y += 12
    canvas_draw.text((x, y), "Top PortBand row ids", fill=(245, 250, 255), font=body_font)
    y += 30
    for row in lineup_rows[:8]:
        canvas_draw.text((x, y), str(row.get("lineup_matrix_row_id")), fill=(190, 230, 210), font=small_font)
        y += 24
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "path": repo_rel(out_path),
        "ghost_id": ghost.get("ghost_id"),
        "terminal_exact_portband_row_count": len(lineup_rows),
        "confirmed_relationship_row_count": int(ghost.get("confirmed_relationship_row_count") or 0),
        "boundary": "review_overlay_not_action_authority",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghost-json", type=Path, required=True)
    parser.add_argument("--packets-jsonl", type=Path, required=True)
    parser.add_argument("--contra-packet-json", type=Path)
    parser.add_argument("--raw-grid", type=Path)
    parser.add_argument("--base-png", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ghost_path = args.ghost_json if args.ghost_json.is_absolute() else ROOT / args.ghost_json
    packets_path = args.packets_jsonl if args.packets_jsonl.is_absolute() else ROOT / args.packets_jsonl
    contra_packet_path = (
        args.contra_packet_json
        if args.contra_packet_json is None or args.contra_packet_json.is_absolute()
        else ROOT / args.contra_packet_json
    )
    raw_grid_arg_path = args.raw_grid if args.raw_grid is None or args.raw_grid.is_absolute() else ROOT / args.raw_grid
    base_png_path = args.base_png if args.base_png is None or args.base_png.is_absolute() else ROOT / args.base_png
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    ghost = read_json(ghost_path)
    packets = read_jsonl(packets_path)
    packet = read_json(contra_packet_path) if contra_packet_path is not None else terminal_packet(ghost, packets)
    raw_grid_path = raw_grid_arg_path or ROOT / (((ghost.get("final_raw_frame") or {}).get("raw_grid") or {}).get("path"))
    raw_grid = np.load(raw_grid_path)
    lineup_rows = exact_terminal_lineup_rows(packet)
    summary = draw_overlay(
        raw_grid=raw_grid,
        lineup_rows=lineup_rows,
        ghost=ghost,
        out_path=out_path,
        base_png_path=base_png_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
