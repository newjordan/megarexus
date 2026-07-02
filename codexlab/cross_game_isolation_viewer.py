#!/usr/bin/env python3
"""Cross-game raw-frame CONTRA isolation spot-check viewer.

This is a review aid. It loads local ARC game files, renders selected reset
frames, runs the same deterministic shape-trace packet builder used by CONTRA,
and writes static HTML/PNG artifacts for human inspection.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import inspect
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from arcengine import ARCBaseGame, ActionInput, GameAction
from contra_head import build_control_isolation_packet
from sp80_base_controller import PALETTE, ROOT, sha256_file, sha256_frame


ENV_ROOT = ROOT / "environment" / "environment_files"
SCHEMA = "codexlab.cross_game_contra_isolation_viewer.v1"
OVERLAY_COLORS = [
    (230, 60, 60),
    (35, 145, 255),
    (45, 170, 70),
    (210, 130, 25),
    (155, 90, 220),
    (0, 165, 165),
    (230, 80, 180),
    (80, 80, 80),
]


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def rel_to_dir(path_value: str, base_dir: Path) -> str:
    return (ROOT / path_value).resolve().relative_to(base_dir.resolve()).as_posix()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rgb_frame(frame: np.ndarray, scale: int) -> Image.Image:
    arr = np.asarray(frame, dtype=np.int64).clip(0, 15)
    rgb = PALETTE[arr]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def save_raw_frame(frame: np.ndarray, png_path: Path, npy_path: Path, scale: int) -> dict[str, Any]:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame, dtype=np.int16)
    np.save(npy_path, arr)
    rgb_frame(arr, scale).save(png_path)
    return {
        "png": {"path": relpath(png_path), "png_sha256": sha256_file(png_path)},
        "raw_grid": {
            "path": relpath(npy_path),
            "npy_sha256": sha256_file(npy_path),
            "raw_frame_sha256": sha256_frame(arr),
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
        },
    }


def draw_shape_overlay(frame: np.ndarray, packet: dict[str, Any], path: Path, scale: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = rgb_frame(frame, scale).convert("RGBA")
    draw = ImageDraw.Draw(image)
    for idx, shape in enumerate(packet.get("shape_traces") or []):
        bbox = shape.get("bbox_xyxy")
        if not bbox:
            continue
        color = OVERLAY_COLORS[idx % len(OVERLAY_COLORS)]
        x0, y0, x1, y1 = [int(v) * scale for v in bbox]
        x1 += max(0, scale - 1)
        y1 += max(0, scale - 1)
        draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=max(1, scale // 2))
        draw.text((x0 + 2, y0 + 2), str(idx), fill=color + (255,))
    image.convert("RGB").save(path)
    return {"path": relpath(path), "png_sha256": sha256_file(path)}


def draw_review_sheet(
    frame: np.ndarray,
    packet: dict[str, Any],
    *,
    game_id: str,
    title: str,
    level_index: int,
    raw_png: Path,
    overlay_png: Path,
    out_path: Path,
    scale: int,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = Image.open(raw_png).convert("RGB")
    overlay = Image.open(overlay_png).convert("RGB")
    width = raw.width + overlay.width + 96
    table_height = max(240, min(760, 24 * (len(packet.get("shape_traces") or []) + 4)))
    height = max(raw.height, overlay.height) + table_height
    sheet = Image.new("RGB", (width, height), (247, 247, 244))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), f"{game_id} / {title} / level {level_index}", fill=(20, 20, 20))
    draw.text((16, 34), "raw frame", fill=(20, 20, 20))
    draw.text((raw.width + 48, 34), "CONTRA shape-trace isolation overlay", fill=(20, 20, 20))
    sheet.paste(raw, (16, 56))
    sheet.paste(overlay, (raw.width + 48, 56))
    y = max(raw.height, overlay.height) + 76
    draw.text(
        (16, y),
        (
            f"schema={packet.get('schema')}  source={packet.get('source')}  "
            f"primary={packet.get('primary_view')}  shapes={packet.get('shape_trace_count')}  "
            f"color_groups={packet.get('color_group_count')}  background={packet.get('background_raw_pixel_value')}"
        ),
        fill=(20, 20, 20),
    )
    y += 28
    draw.text((16, y), "shape  area  bbox  raw-values  supporting-groups", fill=(20, 20, 20))
    y += 22
    for idx, shape in enumerate(packet.get("shape_traces") or []):
        values = ",".join(str(v) for v in shape.get("supporting_raw_pixel_values") or [])
        line = (
            f"{idx:02d}  {int(shape.get('area_pixels', 0)):4d}  "
            f"{shape.get('bbox_xyxy')}  {values}  "
            f"{shape.get('supporting_color_group_count')}"
        )
        draw.text((16, y), line, fill=OVERLAY_COLORS[idx % len(OVERLAY_COLORS)])
        y += 20
        if y > height - 24:
            draw.text((16, y), "... truncated in sheet; see JSON view for full rows", fill=(90, 90, 90))
            break
    sheet.save(out_path)
    return {"path": relpath(out_path), "png_sha256": sha256_file(out_path)}


def discover_games() -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for meta_path in sorted(ENV_ROOT.glob("*/*/metadata.json")):
        metadata = load_json(meta_path)
        game_short = meta_path.parents[1].name
        game_hash = meta_path.parent.name
        py_path = meta_path.parent / f"{game_short}.py"
        if not py_path.is_file():
            continue
        games.append(
            {
                "game_id": str(metadata.get("game_id") or f"{game_short}-{game_hash}"),
                "title": str(metadata.get("title") or game_short),
                "game_short": game_short,
                "game_hash": game_hash,
                "metadata_path": meta_path,
                "source_path": py_path,
            }
        )
    return games


def load_game_class(path: Path) -> type[ARCBaseGame]:
    module_name = f"codexlab_cross_game_{path.parent.parent.name}_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for obj in vars(module).values():
        if inspect.isclass(obj) and issubclass(obj, ARCBaseGame) and obj is not ARCBaseGame:
            return obj
    raise RuntimeError(f"no ARCBaseGame subclass found in {path}")


def raw_reset_frame(game: ARCBaseGame, level_index: int) -> np.ndarray:
    game.set_level(level_index)
    frame_data = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    frame = np.asarray(frame_data.frame, dtype=np.int16)
    if frame.ndim == 3 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 2:
        raise RuntimeError(f"expected 2D raw frame for {game.game_id} level {level_index}, got {frame.shape}")
    return frame


def choose_samples(games: list[dict[str, Any]], *, seed: int, game_count: int, levels_per_game: int) -> list[tuple[dict[str, Any], list[int]]]:
    rng = random.Random(seed)
    eligible = [game for game in games if not str(game["game_id"]).startswith("sp80-")]
    selected = rng.sample(eligible, min(game_count, len(eligible)))
    samples: list[tuple[dict[str, Any], list[int]]] = []
    for game_info in selected:
        cls = load_game_class(game_info["source_path"])
        game = cls()
        level_count = len(getattr(game, "_levels", []))
        if level_count <= 0:
            continue
        level_indices = rng.sample(range(level_count), min(levels_per_game, level_count))
        samples.append((game_info, sorted(level_indices)))
    return samples


def build(args: argparse.Namespace) -> int:
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    games = discover_games()
    samples = choose_samples(games, seed=args.seed, game_count=args.game_count, levels_per_game=args.levels_per_game)
    rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    for game_info, levels in samples:
        cls = load_game_class(game_info["source_path"])
        game = cls()
        for level_index in levels:
            try:
                frame = raw_reset_frame(game, level_index)
            except Exception as exc:
                skipped_rows.append(
                    {
                        "game_id": game_info["game_id"],
                        "title": game_info["title"],
                        "game_short": game_info["game_short"],
                        "source_path": relpath(game_info["source_path"]),
                        "source_sha256": sha256_file(game_info["source_path"]),
                        "level_index": level_index,
                        "skip_reason": str(exc),
                    }
                )
                continue
            case_id = f"{game_info['game_short']}_level_{level_index:02d}"
            packet = build_control_isolation_packet(
                frame,
                game=str(game_info["game_id"]),
                phase="cross_game_reset_view",
                case_id=case_id,
                connectivity=4,
                min_area_pixels=1,
            )
            base = out_dir / str(game_info["game_short"]) / f"level_{level_index:02d}"
            raw_meta = save_raw_frame(
                frame,
                base / "raw.png",
                base / "raw.npy",
                args.scale,
            )
            overlay_meta = draw_shape_overlay(frame, packet, base / "contra_shape_isolation_overlay.png", args.scale)
            write_json(base / "control_isolation_packet.json", packet)
            sheet_meta = draw_review_sheet(
                frame,
                packet,
                game_id=str(game_info["game_id"]),
                title=str(game_info["title"]),
                level_index=level_index,
                raw_png=ROOT / raw_meta["png"]["path"],
                overlay_png=ROOT / overlay_meta["path"],
                out_path=base / "review_sheet.png",
                scale=args.scale,
            )
            rows.append(
                {
                    "game_id": game_info["game_id"],
                    "title": game_info["title"],
                    "game_short": game_info["game_short"],
                    "source_path": relpath(game_info["source_path"]),
                    "source_sha256": sha256_file(game_info["source_path"]),
                    "level_index": level_index,
                    "raw_frame": raw_meta,
                    "control_isolation_packet": relpath(base / "control_isolation_packet.json"),
                    "control_isolation_packet_sha256": sha256_file(base / "control_isolation_packet.json"),
                    "overlay": overlay_meta,
                    "review_sheet": sheet_meta,
                    "shape_trace_count": int(packet.get("shape_trace_count", 0)),
                    "color_group_count": int(packet.get("color_group_count", 0)),
                    "background_raw_pixel_value": packet.get("background_raw_pixel_value"),
                }
            )
    manifest = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "run_label": "cross_game_random_spot_check",
        "random_seed": args.seed,
        "game_count_requested": args.game_count,
        "levels_per_game_requested": args.levels_per_game,
        "sample_count": len(rows),
        "skipped_sample_count": len(skipped_rows),
        "source": "raw_reset_frame_shape_trace_with_color_group_support",
        "semantic_labels_used": False,
        "sam_status": "not_run",
        "route_or_replay_used": False,
        "human_trace_used": False,
        "skip_policy": "unsupported or failing reset frames are recorded in skipped_rows and not converted into proxy 2D images",
        "skipped_rows": skipped_rows,
        "rows": rows,
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    index_lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Cross-Game CONTRA Isolation Spot Check</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f4;color:#191919} img{image-rendering:pixelated;max-width:100%;border:1px solid #aaa;background:white} section{margin-bottom:36px}.pair{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start} code{background:#eee;padding:1px 3px}</style>",
        "</head><body>",
        "<h1>Cross-Game CONTRA Isolation Spot Check</h1>",
        f"<p>Seed <code>{args.seed}</code>. Same deterministic shape-trace packet builder as CONTRA. No SAM, semantic labels, human trace, route, or replay used.</p>",
        f"<p>Manifest: <code>{html.escape(relpath(manifest_path))}</code></p>",
    ]
    for row in rows:
        raw_href = rel_to_dir(row["raw_frame"]["png"]["path"], out_dir)
        sheet_href = rel_to_dir(row["review_sheet"]["path"], out_dir)
        overlay_href = rel_to_dir(row["overlay"]["path"], out_dir)
        packet_href = rel_to_dir(row["control_isolation_packet"], out_dir)
        index_lines.extend(
            [
                "<section>",
                f"<h2>{html.escape(str(row['game_id']))} level {row['level_index']}</h2>",
                f"<p>shapes={row['shape_trace_count']} color_groups={row['color_group_count']} background={row['background_raw_pixel_value']}</p>",
                "<div class='pair'>",
                f"<div><h3>Raw</h3><img src='{html.escape(raw_href)}'></div>",
                f"<div><h3>CONTRA Isolation</h3><img src='{html.escape(overlay_href)}'></div>",
                "</div>",
                f"<p><a href='{html.escape(sheet_href)}'>review sheet</a> | <a href='{html.escape(overlay_href)}'>overlay</a> | <a href='{html.escape(packet_href)}'>packet JSON</a></p>",
                "</section>",
            ]
        )
    index_lines.extend(["</body></html>"])
    (out_dir / "index.html").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": relpath(manifest_path),
                "index": relpath(out_dir / "index.html"),
                "sample_count": len(rows),
                "skipped_sample_count": len(skipped_rows),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="codexlab/runs/cross_game_isolation_spotcheck")
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--game-count", type=int, default=2)
    parser.add_argument("--levels-per-game", type=int, default=2)
    parser.add_argument("--scale", type=int, default=8)
    args = parser.parse_args()
    return build(args)


if __name__ == "__main__":
    raise SystemExit(main())
