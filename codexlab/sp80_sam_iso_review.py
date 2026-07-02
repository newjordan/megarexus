#!/usr/bin/env python3
"""Build per-slice isolation review images for SP80 SAM hypotheses.

The generated images are review aids. They are derived from recorded raw frame
PNGs, SAM masks, controlled overlays, and acceptance-ledger rows.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SCHEMA = "codexlab.sp80_sam_iso_review_manifest.v1"


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


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def rel_for_html(path_value: str, index_dir: Path) -> str:
    return Path(resolve_repo_path(path_value)).resolve().relative_to(index_dir.resolve()).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_visual_hint(run_dir: Path, source_manifest: dict[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    path = run_dir / "sam_visual_hints.json"
    if not path.is_file():
        source_path = source_manifest.get("source_visual_hints_json")
        if not source_path:
            return None, None
        resolved = resolve_repo_path(source_path)
        if source_manifest.get("visual_hints") is not None:
            return source_manifest.get("visual_hints"), resolved
        return load_json(resolved), resolved
    return load_json(path), path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def thumb(image: Image.Image, size: int) -> Image.Image:
    out = image.copy()
    out.thumbnail((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(out.convert("RGB"), ((size - out.width) // 2, (size - out.height) // 2))
    return canvas


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


def overlay_mask(raw_png: Path, mask_png: Path, row: dict[str, Any], acceptance: dict[str, Any] | None, scale: int) -> Image.Image:
    base = Image.open(raw_png).convert("RGBA")
    mask = Image.open(mask_png).convert("L")
    if mask.size != base.size:
        mask = mask.resize(base.size, Image.Resampling.NEAREST)
    status = (acceptance or {}).get("acceptance_status") or "unreviewed"
    if status == "accepted":
        color = (28, 160, 78, 118)
        outline = (14, 110, 48)
    elif row.get("score", {}).get("confirmation_status") == "confirmed_against_controlled_mask":
        color = (230, 150, 30, 104)
        outline = (170, 95, 12)
    else:
        color = (70, 130, 220, 92)
        outline = (35, 75, 150)
    color_layer = Image.new("RGBA", base.size, color)
    alpha = mask.point(lambda px: color[3] if px > 0 else 0)
    color_layer.putalpha(alpha)
    base.alpha_composite(color_layer)
    draw = ImageDraw.Draw(base)
    bbox = row.get("score", {}).get("bbox_xyxy")
    if bbox:
        x0, y0, x1, y1 = [int(v) * scale for v in bbox]
        x1 += scale - 1
        y1 += scale - 1
        draw.rectangle([x0, y0, x1, y1], outline=outline, width=max(2, scale // 2))
    best = row.get("score", {}).get("best_controlled_match") or {}
    text = "\n".join(
        [
            str(row.get("hypothesis_id")),
            f"{status} | {row.get('score', {}).get('confirmation_status')}",
            f"iou={float(best.get('iou') or 0.0):.3f} match={best.get('object_id') or 'none'}",
        ]
    )
    draw_label_box(draw, (8, 8), text, outline)
    return base.convert("RGB")


def make_slice_contact(
    *,
    level_index: int,
    view: dict[str, Any],
    rows: list[dict[str, Any]],
    acceptance_by_id: dict[str, dict[str, Any]],
    out_path: Path,
    scale: int,
    visual_hint: dict[str, Any] | None,
) -> dict[str, Any]:
    tile = 256
    gap = 16
    header_h = 44
    row_title_h = 28
    top_sources = [
        ("raw frame", resolve_repo_path(view["raw_frame"]["png"]["path"])),
        ("controlled overlay", resolve_repo_path(view["controlled_overlay"]["path"])),
        ("SAM overlay", resolve_repo_path(view["sam_overlay"]["path"])),
    ]
    top_tiles: list[Image.Image] = []
    for label, path in top_sources:
        img = thumb(Image.open(path), tile)
        draw = ImageDraw.Draw(img)
        draw_label_box(draw, (6, 6), label, (45, 45, 45))
        top_tiles.append(img)

    iso_tiles: list[Image.Image] = []
    raw_png = resolve_repo_path(view["raw_frame"]["png"]["path"])
    for row in rows:
        mask_png = resolve_repo_path(row["pixel_mask"]["png"]["path"])
        acceptance = acceptance_by_id.get(str(row.get("hypothesis_id")))
        iso_tiles.append(thumb(overlay_mask(raw_png, mask_png, row, acceptance, scale), tile))

    columns = 3
    iso_rows = max(1, (len(iso_tiles) + columns - 1) // columns)
    width = columns * tile + (columns + 1) * gap
    hint_h = 24 if visual_hint else 0
    height = header_h + hint_h + row_title_h + tile + gap + row_title_h + iso_rows * tile + (iso_rows + 1) * gap
    canvas = Image.new("RGB", (width, height), (246, 246, 242))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 14), f"SP80 SAM Isolation Review - Calibration Slice {level_index}", fill=(20, 20, 20), font=ImageFont.load_default())
    y = header_h
    if visual_hint:
        hint = visual_hint.get("hint") or {}
        draw.text(
            (gap, y),
            f"Human visual hint: {hint.get('target_semantic_label', 'target')} = {hint.get('color_name', 'unknown')} {hint.get('shape_description', 'shape')}",
            fill=(95, 69, 0),
            font=ImageFont.load_default(),
        )
        y += hint_h
    draw.text((gap, y), "Context frames", fill=(20, 20, 20), font=ImageFont.load_default())
    y += row_title_h
    for idx, img in enumerate(top_tiles):
        x = gap + idx * (tile + gap)
        canvas.paste(img, (x, y))
    y += tile + gap
    draw.text((gap, y), "Isolated SAM mask hypotheses", fill=(20, 20, 20), font=ImageFont.load_default())
    y += row_title_h
    for idx, img in enumerate(iso_tiles):
        col = idx % columns
        row = idx // columns
        x = gap + col * (tile + gap)
        yy = y + row * (tile + gap)
        canvas.paste(img, (x, yy))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {
        "level_index": level_index,
        "calibration_slice_index": level_index,
        "path": relpath(out_path),
        "png_sha256": sha256_file(out_path),
        "hypothesis_count": len(rows),
        "accepted_hypothesis_ids": [
            str(row.get("hypothesis_id"))
            for row in rows
            if (acceptance_by_id.get(str(row.get("hypothesis_id"))) or {}).get("acceptance_status") == "accepted"
        ],
    }


def build_index(
    *,
    manifest: dict[str, Any],
    source_manifest: dict[str, Any],
    rows_by_level: dict[int, list[dict[str, Any]]],
    acceptance_by_id: dict[str, dict[str, Any]],
    index_dir: Path,
    visual_hint: dict[str, Any] | None,
) -> str:
    if visual_hint:
        hint = visual_hint.get("hint") or {}
        hint_text = (
            f"Human visual hint: {hint.get('target_semantic_label', 'target')} = "
            f"{hint.get('color_name', 'unknown')} {hint.get('shape_description', 'shape')}."
        )
        hint_note = visual_hint.get("current_sam_run_used_this_hint_note") or ""
    else:
        hint_text = "Human visual hint: none recorded."
        hint_note = ""
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 SAM Isolation Review</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:24px;background:#f7f7f4;color:#191919}",
        "section{margin:0 0 36px 0}",
        "img{image-rendering:pixelated;border:1px solid #999;background:white;max-width:100%;height:auto}",
        "table{border-collapse:collapse;font-size:13px;margin:12px 0 18px 0}",
        "th,td{border:1px solid #ccc;padding:5px 7px;text-align:left;vertical-align:top}",
        "code{background:#eee;padding:1px 3px}",
        ".note{max-width:980px;line-height:1.35}",
        ".accepted{color:#14743a;font-weight:700}",
        ".blocked{color:#666}",
        "</style></head><body>",
        "<h1>SP80 SAM Isolation Review</h1>",
        "<p class='note'>These are review aids generated from recorded raw frames, SAM masks, controlled overlays, and the acceptance ledger. They are not new environment observations and they are not admitted into Contra control truth, goal conditions, or action choice.</p>",
        f"<p class='note'><b>{html.escape(hint_text)}</b> {html.escape(hint_note)}</p>",
        f"<p><b>Source SAM manifest:</b> <code>{html.escape(manifest['source_sam_manifest_json'])}</code></p>",
        f"<p><b>Acceptance manifest:</b> <code>{html.escape(manifest['source_acceptance_manifest_json'])}</code></p>",
        f"<p><b>Review manifest:</b> <code>{html.escape(manifest['manifest_json'])}</code></p>",
    ]
    for view in source_manifest["views"]:
        level_index = int(view["level_index"])
        rows = rows_by_level.get(level_index, [])
        slice_info = next(item for item in manifest["slice_outputs"] if item["level_index"] == level_index)
        body.extend(
            [
                f"<section><h2>Calibration Slice {level_index}</h2>",
                f"<p><a href='{rel_for_html(slice_info['path'], index_dir)}'>Open isolation PNG</a></p>",
                f"<img src='{rel_for_html(slice_info['path'], index_dir)}'>",
                "<table><thead><tr><th>hypothesis</th><th>acceptance</th><th>SAM status</th><th>best controlled match</th></tr></thead><tbody>",
            ]
        )
        for row in rows:
            hypothesis_id = str(row.get("hypothesis_id"))
            acceptance = acceptance_by_id.get(hypothesis_id) or {}
            status = acceptance.get("acceptance_status") or "missing"
            best = row.get("score", {}).get("best_controlled_match") or {}
            cls = "accepted" if status == "accepted" else "blocked"
            body.append(
                "<tr>"
                f"<td><code>{html.escape(hypothesis_id)}</code></td>"
                f"<td class='{cls}'>{html.escape(status)}</td>"
                f"<td>{html.escape(str(row.get('score', {}).get('confirmation_status')))}</td>"
                f"<td><code>{html.escape(json.dumps(best, sort_keys=True))}</code></td>"
                "</tr>"
            )
        body.extend(["</tbody></table></section>"])
    body.extend(["</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("codexlab/runs/sp80_sam_hypothesis_viewer"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = resolve_repo_path(args.run_dir)
    out_dir = resolve_repo_path(args.out_dir) if args.out_dir is not None else run_dir / "iso_review"
    out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest_path = run_dir / "sam_hypothesis_manifest.json"
    acceptance_manifest_path = run_dir / "sam_acceptance_manifest.json"
    source_manifest = load_json(source_manifest_path)
    acceptance_manifest = load_json(acceptance_manifest_path)
    source_rows_path = resolve_repo_path(source_manifest["sam_hypotheses_jsonl"])
    acceptance_rows_path = resolve_repo_path(acceptance_manifest["accepted_object_hypotheses_jsonl"])
    visual_hint, visual_hint_path = load_visual_hint(run_dir, source_manifest)
    rows = load_jsonl(source_rows_path)
    acceptance_rows = load_jsonl(acceptance_rows_path)
    rows_by_level: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_level.setdefault(int(row["level_index"]), []).append(row)
    acceptance_by_id = {str(row["hypothesis_id"]): row for row in acceptance_rows}
    scale = int(source_manifest.get("scale") or 8)
    slice_outputs = [
        make_slice_contact(
            level_index=int(view["level_index"]),
            view=view,
            rows=rows_by_level.get(int(view["level_index"]), []),
            acceptance_by_id=acceptance_by_id,
            out_path=out_dir / "slices" / f"slice_{int(view['level_index']):02d}_iso.png",
            scale=scale,
            visual_hint=visual_hint,
        )
        for view in source_manifest["views"]
    ]
    status_counts = Counter(row.get("acceptance_status") for row in acceptance_rows)
    manifest_path = out_dir / "iso_review_manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "sp80_sam_iso_review",
        "condition_natural_language": "Create per-calibration-slice review images from recorded SP80 raw frames, SAM masks, controlled overlays, and the SAM acceptance ledger.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "game": "sp80_liquid_pour",
        "review_artifact_only": True,
        "raw_pixels_are_observation_ground_truth": True,
        "source_sam_manifest_json": relpath(source_manifest_path),
        "source_sam_manifest_json_sha256": sha256_file(source_manifest_path),
        "source_sam_hypotheses_jsonl": relpath(source_rows_path),
        "source_sam_hypotheses_jsonl_sha256": sha256_file(source_rows_path),
        "source_acceptance_manifest_json": relpath(acceptance_manifest_path),
        "source_acceptance_manifest_json_sha256": sha256_file(acceptance_manifest_path),
        "source_acceptance_jsonl": relpath(acceptance_rows_path),
        "source_acceptance_jsonl_sha256": sha256_file(acceptance_rows_path),
        "source_visual_hints_json": None if visual_hint_path is None else relpath(visual_hint_path),
        "source_visual_hints_json_sha256": None if visual_hint_path is None else sha256_file(visual_hint_path),
        "visual_hint": visual_hint,
        "source_script": "codexlab/sp80_sam_iso_review.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_sam_iso_review.py"),
        "acceptance_status_counts": dict(sorted(status_counts.items())),
        "slice_outputs": slice_outputs,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "admitted_for_control_truth_count": 0,
        "admitted_for_goal_condition_count": 0,
        "admitted_for_contra_object_view_count": 0,
        "admitted_for_action_choice_count": 0,
    }
    manifest["manifest_json"] = relpath(manifest_path)
    index_path = out_dir / "index.html"
    index_path.write_text(
        build_index(
            manifest=manifest,
            source_manifest=source_manifest,
            rows_by_level=rows_by_level,
            acceptance_by_id=acceptance_by_id,
            index_dir=out_dir,
            visual_hint=visual_hint,
        ),
        encoding="utf-8",
    )
    manifest["index_html"] = relpath(index_path)
    manifest["index_html_sha256"] = sha256_file(index_path)
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "exported",
                "index_html": relpath(index_path),
                "manifest": relpath(manifest_path),
                "slice_outputs": slice_outputs,
                "acceptance_status_counts": dict(sorted(status_counts.items())),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
