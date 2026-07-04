"""Build a fast component-graph review from saved Megarexus arrange frames.

This is a static review tool: it does not replay the policy, score candidates,
or step the environment. It turns captured 64x64 policy frames into a readable
frames-first packet with component boxes, pairwise relations, and motion deltas.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

_LAB_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_LAB_ROOT, _LAB_ROOT / "orchestration"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402


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


def load_episode(episode_dir: Path) -> list[dict[str, Any]]:
    meta = json.loads((episode_dir / "policy_frames_meta.json").read_text(encoding="utf-8"))
    frames_npz = np.load(episode_dir / "policy_frames.npz")
    rows = []
    for row in meta:
        call = int(row["call"])
        frames = np.asarray(frames_npz[f"call{call:04d}"], dtype=np.int8)
        rows.append({**row, "frames": frames, "grid": frames[-1]})
    return rows


def component_row(comp: dict[str, Any], index: int) -> dict[str, Any]:
    x0, y0, x1, y1 = [int(v) for v in comp["bbox_xyxy"]]
    cx, cy = [int(v) for v in comp["center_xy"]]
    return {
        "index": int(index),
        "bbox_xyxy": [x0, y0, x1, y1],
        "center_xy": [cx, cy],
        "width": x1 - x0 + 1,
        "height": y1 - y0 + 1,
        "area": int(comp.get("area", (x1 - x0 + 1) * (y1 - y0 + 1))),
        "aspect": round((x1 - x0 + 1) / max(y1 - y0 + 1, 1), 3),
    }


def graph_for_grid(grid: np.ndarray) -> dict[str, Any]:
    comps = [component_row(comp, idx) for idx, comp in enumerate(frame_components(grid))]
    relations = []
    for a in comps:
        ax0, ay0, ax1, ay1 = a["bbox_xyxy"]
        acx, acy = a["center_xy"]
        for b in comps:
            if a["index"] == b["index"]:
                continue
            bx0, by0, bx1, by1 = b["bbox_xyxy"]
            bcx, bcy = b["center_xy"]
            x_gap = max(0, max(bx0 - ax1 - 1, ax0 - bx1 - 1))
            y_gap = max(0, max(by0 - ay1 - 1, ay0 - by1 - 1))
            x_overlap = max(0, min(ax1, bx1) - max(ax0, bx0) + 1)
            y_overlap = max(0, min(ay1, by1) - max(ay0, by0) + 1)
            relations.append(
                {
                    "a": a["index"],
                    "b": b["index"],
                    "dx_center": int(bcx - acx),
                    "dy_center": int(bcy - acy),
                    "x_gap": int(x_gap),
                    "y_gap": int(y_gap),
                    "x_overlap": int(x_overlap),
                    "y_overlap": int(y_overlap),
                    "left_right": "right" if bcx > acx else "left" if bcx < acx else "same_x",
                    "above_below": "below" if bcy > acy else "above" if bcy < acy else "same_y",
                }
            )
    return {"components": comps, "relations": relations}


def match_components(prev: list[dict[str, Any]], cur: list[dict[str, Any]]) -> list[dict[str, Any]]:
    matches = []
    used: set[int] = set()
    for p in prev:
        pcx, pcy = p["center_xy"]
        best = None
        best_dist = 10**9
        for c in cur:
            if c["index"] in used:
                continue
            if c["area"] != p["area"]:
                continue
            ccx, ccy = c["center_xy"]
            dist = abs(ccx - pcx) + abs(ccy - pcy)
            if dist < best_dist:
                best = c
                best_dist = dist
        if best is None:
            continue
        used.add(best["index"])
        ccx, ccy = best["center_xy"]
        matches.append(
            {
                "prev_index": p["index"],
                "cur_index": best["index"],
                "area": p["area"],
                "dx": int(ccx - pcx),
                "dy": int(ccy - pcy),
                "prev_bbox": p["bbox_xyxy"],
                "cur_bbox": best["bbox_xyxy"],
            }
        )
    return matches


def render_grid(grid: np.ndarray, comps: list[dict[str, Any]], path: Path, scale: int = 6) -> None:
    arr = np.asarray(grid, dtype=np.int16)
    arr = np.clip(arr, 0, len(PALETTE) - 1)
    image = Image.fromarray(PALETTE[arr], mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    colors = ["#ff3355", "#33aa55", "#3377ff", "#eeaa00", "#aa44ee", "#00aaaa"]
    for comp in comps:
        x0, y0, x1, y1 = [int(v) * scale for v in comp["bbox_xyxy"]]
        color = colors[comp["index"] % len(colors)]
        draw.rectangle([x0, y0, x1 + scale - 1, y1 + scale - 1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), str(comp["index"]), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def episode_report(label: str, episode_dir: Path, out_dir: Path, image_prefix: str) -> dict[str, Any]:
    rows = load_episode(episode_dir)
    frame_reports = []
    previous_components: list[dict[str, Any]] | None = None
    for row in rows:
        call = int(row["call"])
        grid = np.asarray(row["grid"], dtype=np.int8)
        graph = graph_for_grid(grid)
        comps = graph["components"]
        delta = None
        if previous_components is not None:
            matches = match_components(previous_components, comps)
            moved = [m for m in matches if m["dx"] or m["dy"]]
            delta = {
                "matched": len(matches),
                "moved": moved,
                "horizontal_moves": sum(1 for m in moved if m["dx"] and not m["dy"]),
                "vertical_moves": sum(1 for m in moved if m["dy"]),
                "nonzero_moves": len(moved),
            }
        rel_path = Path("frames") / f"{image_prefix}_call{call:04d}.png"
        render_grid(grid, comps, out_dir / rel_path)
        frame_reports.append(
            {
                "call": call,
                "levels": int(row.get("levels") or 0),
                "state": str(row.get("state", "")),
                "resets": int(row.get("resets") or 0),
                "frame_sha256": frame_hash(grid),
                "component_count": len(comps),
                "components": comps,
                "relations": graph["relations"],
                "delta_from_previous": delta,
                "image": str(rel_path),
            }
        )
        previous_components = comps
    move_counter = Counter()
    for frame in frame_reports:
        delta = frame.get("delta_from_previous") or {}
        move_counter["horizontal_moves"] += int(delta.get("horizontal_moves") or 0)
        move_counter["vertical_moves"] += int(delta.get("vertical_moves") or 0)
        move_counter["nonzero_moves"] += int(delta.get("nonzero_moves") or 0)
    return {
        "label": label,
        "episode_dir": str(episode_dir),
        "policy_frames_npz_sha256": sha256_file(episode_dir / "policy_frames.npz"),
        "policy_frames_meta_sha256": sha256_file(episode_dir / "policy_frames_meta.json"),
        "calls": len(frame_reports),
        "unique_frame_hashes": len({f["frame_sha256"] for f in frame_reports}),
        "move_summary": dict(move_counter),
        "frames": frame_reports,
    }


def compare_episodes(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    pairs = zip(a["frames"], b["frames"], strict=False)
    compared = []
    for fa, fb in pairs:
        compared.append(
            {
                "call_a": fa["call"],
                "call_b": fb["call"],
                "same_frame_hash": fa["frame_sha256"] == fb["frame_sha256"],
                "same_component_count": fa["component_count"] == fb["component_count"],
            }
        )
    return {
        "labels": [a["label"], b["label"]],
        "compared_calls": len(compared),
        "identical_frame_hashes": sum(1 for row in compared if row["same_frame_hash"]),
        "all_frame_hashes_identical": all(row["same_frame_hash"] for row in compared),
        "detail": compared,
    }


def write_html(report: dict[str, Any], out_dir: Path) -> None:
    def esc(value: Any) -> str:
        if isinstance(value, str):
            return html.escape(value)
        return html.escape(json.dumps(value, sort_keys=True))

    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Megarexus Arrange Graph Review</title>",
        "<style>",
        "body{font-family:Inter,system-ui,sans-serif;margin:24px;background:#f8fafc;color:#18202a}",
        "h1,h2,h3{margin:0 0 10px} section{margin:24px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border:1px solid #d7dee8;padding:6px;vertical-align:top} th{background:#edf2f7;text-align:left}",
        ".deck{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}",
        ".panel{background:white;border:1px solid #d7dee8;border-radius:6px;padding:10px}",
        "img{image-rendering:pixelated;width:100%;max-width:384px;border:1px solid #ccd5e1}",
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}",
        "</style>",
        "<h1>Megarexus Arrange Graph Review</h1>",
        "<p><b>Boundary:</b> static component-graph review from saved policy frames. Not a gate, not a live scorecard, not solve evidence.</p>",
        f"<p><b>Command:</b> <code>{esc(report['command'])}</code></p>",
        "<section><h2>Summary</h2><table><tr><th>field</th><th>value</th></tr>",
    ]
    for key in ("created_local", "script_sha256", "episodes", "comparison"):
        lines.append(f"<tr><td>{key}</td><td><code>{esc(report.get(key))}</code></td></tr>")
    lines.append("</table></section>")
    for episode in report["episode_reports"]:
        lines.append(f"<section><h2>{esc(episode['label'])}</h2>")
        lines.append("<table><tr><th>calls</th><th>unique frames</th><th>move summary</th><th>npz sha</th></tr>")
        lines.append(
            f"<tr><td>{episode['calls']}</td><td>{episode['unique_frame_hashes']}</td>"
            f"<td><code>{esc(episode['move_summary'])}</code></td>"
            f"<td><code>{esc(episode['policy_frames_npz_sha256'])}</code></td></tr></table>"
        )
        lines.append("<h3>Frame Deck</h3><div class='deck'>")
        for frame in episode["frames"]:
            moved = frame.get("delta_from_previous") or {}
            moved_items = moved.get("moved") or []
            lines.append("<div class='panel'>")
            lines.append(f"<h3>call {frame['call']}</h3>")
            lines.append(f"<a href='{html.escape(frame['image'])}'><img src='{html.escape(frame['image'])}'></a>")
            lines.append(
                f"<p>components={frame['component_count']} "
                f"moved={len(moved_items)} hash=<code>{esc(frame['frame_sha256'][:16])}</code></p>"
            )
            if moved_items:
                lines.append(f"<p><code>{esc(moved_items[:4])}</code></p>")
            lines.append("</div>")
        lines.append("</div>")
        lines.append("<h3>Per-Call Graph Rows</h3><table><tr><th>call</th><th>components</th><th>moved</th><th>sample relations</th></tr>")
        for frame in episode["frames"]:
            delta = frame.get("delta_from_previous") or {}
            lines.append(
                f"<tr><td>{frame['call']}</td><td><code>{esc(frame['components'])}</code></td>"
                f"<td><code>{esc(delta.get('moved', []))}</code></td>"
                f"<td><code>{esc(frame['relations'][:8])}</code></td></tr>"
            )
        lines.append("</table></section>")
    lines.append("<section><h2>Raw Artifacts</h2><ul><li><a href='arrange_graph_review.json'>arrange_graph_review.json</a></li></ul></section>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(args.label)
    while len(labels) < len(args.episode_dir):
        labels.append(args.episode_dir[len(labels)].name)

    report: dict[str, Any] = {
        "schema": "synthesis_lab.arrange_graph_review.v0",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boundary": "static component-graph review from saved policy frames; not a gate and not live scorecard evidence",
        "command": " ".join(sys.argv),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "episodes": len(args.episode_dir),
        "episode_reports": [],
    }
    for idx, episode_dir in enumerate(args.episode_dir):
        report["episode_reports"].append(
            episode_report(labels[idx], episode_dir.resolve(), out_dir, f"episode{idx}")
        )
    if len(report["episode_reports"]) >= 2:
        report["comparison"] = compare_episodes(
            report["episode_reports"][0], report["episode_reports"][1]
        )
    else:
        report["comparison"] = None

    report_path = out_dir / "arrange_graph_review.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_html(report, out_dir)
    print(f"[arrange_graph_review] report={report_path}")
    print(f"[arrange_graph_review] review={out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
