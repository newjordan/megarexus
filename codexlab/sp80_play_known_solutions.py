#!/usr/bin/env python3
"""Controlled SP80 playback for recorded level 1/2 solution attempts."""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from arcengine import ActionInput, GameAction
from PIL import Image, ImageDraw

from contra_head import build_actionmodband, build_object_action_response
from sp80_base_controller import (
    ROOT,
    SP80BaseController,
    frame_delta,
    rgb_frame,
    save_frame,
    save_frame_sequence,
    sha256_file,
    sha256_frame,
)


SOLUTIONS: dict[int, list[dict[str, Any]]] = {
    1: [
        {"label": "select idx0 plzwjbfyfli-3", "action": "ACTION6", "data": {"x": 28, "y": 26}},
        *[{"label": f"move idx0 left {i + 1}/6", "action": "ACTION4", "data": {}} for i in range(6)],
        {"label": "select idx1 plzwjbfyfli-3", "action": "ACTION6", "data": {"x": 8, "y": 18}},
        *[{"label": f"move idx1 left {i + 1}/3", "action": "ACTION4", "data": {}} for i in range(3)],
        {"label": "select idx2 plzwjbfyfli-5", "action": "ACTION6", "data": {"x": 20, "y": 38}},
        *[{"label": f"move idx2 left {i + 1}/3", "action": "ACTION4", "data": {}} for i in range(3)],
        {"label": "activate paint flow", "action": "ACTION5", "data": {}},
    ],
    2: [
        {"label": "select idx0 plzwjbfyfli-4", "action": "ACTION6", "data": {"x": 8, "y": 22}},
        {"label": "move idx0 left 1/1", "action": "ACTION4", "data": {}},
        {"label": "select idx1 plzwjbfyfli-5", "action": "ACTION6", "data": {"x": 40, "y": 30}},
        *[{"label": f"move idx1 right {i + 1}/2", "action": "ACTION3", "data": {}} for i in range(2)],
        {"label": "select idx2 plzwjbfyfli-6", "action": "ACTION6", "data": {"x": 8, "y": 34}},
        *[{"label": f"move idx2 right {i + 1}/2", "action": "ACTION3", "data": {}} for i in range(2)],
        {"label": "select idx3 plzwjbfyfli-6", "action": "ACTION6", "data": {"x": 36, "y": 42}},
        {"label": "move idx3 left 1/1", "action": "ACTION4", "data": {}},
        {"label": "activate paint flow", "action": "ACTION5", "data": {}},
    ],
}


def action_id(name: str) -> GameAction:
    try:
        return getattr(GameAction, name)
    except AttributeError as exc:
        raise ValueError(f"unknown action {name!r}") from exc


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_raw(frame: np.ndarray, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(frame))
    return {"path": rel(path), "sha256": sha256_file(path), "raw_frame_sha256": sha256_frame(frame)}


def make_sheet(frames: list[np.ndarray], labels: list[str], path: Path, *, scale: int, cols: int = 4) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = [Image.fromarray(rgb_frame(frame, scale), mode="RGB") for frame in frames]
    if not rendered:
        raise ValueError("no frames to sheet")
    label_h = 42
    tile_w, tile_h = rendered[0].size[0], rendered[0].size[1] + label_h
    rows = (len(rendered) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * tile_h), (12, 16, 22))
    draw = ImageDraw.Draw(sheet)
    for idx, img in enumerate(rendered):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        sheet.paste(img, (x, y + label_h))
        draw.text((x + 8, y + 6), labels[idx][:58], fill=(232, 236, 242))
        draw.rectangle([x, y + label_h, x + img.size[0] - 1, y + label_h + img.size[1] - 1], outline=(76, 89, 110), width=2)
    sheet.save(path)
    return {"path": rel(path), "sha256": sha256_file(path), "frame_count": len(frames)}


def run_level(level_index: int, out_dir: Path, *, scale: int, max_settle_ticks: int) -> dict[str, Any]:
    if level_index not in SOLUTIONS:
        raise ValueError(f"no recorded solution for level {level_index}")
    level_dir = out_dir / f"level_{level_index:02d}"
    level_dir.mkdir(parents=True, exist_ok=True)
    controller = SP80BaseController(level_index=level_index)
    game, obs = controller.new_game()

    frames: list[np.ndarray] = [np.asarray(obs.frame[-1]).copy()]
    labels: list[str] = ["00 initial"]
    paint_frames: list[np.ndarray] = []
    paint_labels: list[str] = []
    action_rows: list[dict[str, Any]] = []
    starting_state = controller.state(game, obs)
    starting_completed = int(starting_state.get("levels_completed") or 0)

    for planned_index, planned in enumerate(SOLUTIONS[level_index], start=1):
        before_frame = np.asarray(obs.frame[-1]).copy()
        before_state = controller.state(game, obs)
        obs = game.perform_action(ActionInput(id=action_id(str(planned["action"])), data=dict(planned.get("data") or {})), raw=True)
        if obs is None or not obs.frame:
            raise RuntimeError(f"level {level_index} action {planned_index} produced no frame")
        returned_frames = [np.asarray(frame).copy() for frame in obs.frame]
        after_frame = np.asarray(obs.frame[-1]).copy()
        after_state = controller.state(game, obs)
        for returned_index, returned_frame in enumerate(returned_frames, start=1):
            frames.append(returned_frame)
            labels.append(
                f"{len(frames) - 1:02d} {planned['action']} {planned['label']} "
                f"frame {returned_index}/{len(returned_frames)}"
            )
            if planned["action"] == "ACTION5":
                paint_frames.append(returned_frame)
                paint_labels.append(f"paint {returned_index}/{len(returned_frames)}")
        action_rows.append(
            {
                "index": planned_index,
                "phase": "solution",
                "label": planned["label"],
                "action": planned["action"],
                "data": planned.get("data") or {},
                "returned_frame_count": len(returned_frames),
                "before_state": before_state,
                "after_state": after_state,
                "delta": frame_delta(before_frame, after_frame),
            }
        )

    settle_tick = 0
    while int(controller.state(game, obs).get("levels_completed") or 0) <= starting_completed and settle_tick < max_settle_ticks:
        settle_tick += 1
        before_frame = np.asarray(obs.frame[-1]).copy()
        before_state = controller.state(game, obs)
        obs = game.perform_action(ActionInput(id=GameAction.ACTION5, data={}), raw=True)
        if obs is None or not obs.frame:
            raise RuntimeError(f"level {level_index} settle tick {settle_tick} produced no frame")
        returned_frames = [np.asarray(frame).copy() for frame in obs.frame]
        after_frame = np.asarray(obs.frame[-1]).copy()
        after_state = controller.state(game, obs)
        for returned_index, returned_frame in enumerate(returned_frames, start=1):
            frames.append(returned_frame)
            labels.append(f"{len(frames) - 1:02d} settle tick {settle_tick} frame {returned_index}/{len(returned_frames)}")
            paint_frames.append(returned_frame)
            paint_labels.append(f"settle {settle_tick} frame {returned_index}/{len(returned_frames)}")
        action_rows.append(
            {
                "index": len(action_rows) + 1,
                "phase": "settle",
                "label": f"settle tick {settle_tick}",
                "action": "ACTION5",
                "data": {},
                "returned_frame_count": len(returned_frames),
                "before_state": before_state,
                "after_state": after_state,
                "delta": frame_delta(before_frame, after_frame),
            }
        )

    ending_state = controller.state(game, obs)
    completed_delta = int(ending_state.get("levels_completed") or 0) - starting_completed
    status = "won" if completed_delta > 0 else "not_completed"
    action_invocation_id = f"sp80_l{level_index:02d}_known_solution_controlled_playback"
    action_responses = [
        build_object_action_response(
            before_state=row.get("before_state"),
            after_state=row.get("after_state"),
            action_taken={"name": row.get("action"), "data": row.get("data", {})},
            frame_delta=row.get("delta"),
            returned_frame_count=int(row.get("returned_frame_count") or 0),
            action_invocation_id=action_invocation_id,
            action_index=int(row.get("index") or 0),
            downstream_result={
                "levels_completed_delta": int(row.get("after_state", {}).get("levels_completed") or 0)
                - int(row.get("before_state", {}).get("levels_completed") or 0)
            },
        )
        for row in action_rows
    ]
    actionmodband = build_actionmodband(
        action_responses=action_responses,
        action_invocation_id=action_invocation_id,
        levels_completed_delta=completed_delta,
    )

    png_entries = save_frame_sequence(frames, level_dir / "frames" / "png", scale)
    raw_entries = [save_raw(frame, level_dir / "frames" / "raw_grid" / f"frame_{idx:03d}.npy") for idx, frame in enumerate(frames)]
    before_meta = save_frame(frames[0], level_dir / "before.png", scale)
    after_meta = save_frame(frames[-1], level_dir / "after.png", scale)
    sheet_meta = make_sheet(frames, labels, level_dir / "frame_sheet.png", scale=scale)
    paint_sheet_meta = None
    if paint_frames:
        paint_sheet_meta = make_sheet(paint_frames, paint_labels, level_dir / "paint_flow_sheet.png", scale=scale, cols=4)

    result = {
        "schema": "codexlab.sp80_known_solution_playback.v1",
        "condition": f"controlled SP80 level {level_index} recorded solution playback",
        "status": status,
        "controlled": True,
        "proxy": False,
        "level_index": level_index,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "starting_state": starting_state,
        "ending_state": ending_state,
        "levels_completed_delta": completed_delta,
        "solution_action_count": len(SOLUTIONS[level_index]),
        "settle_tick_count": settle_tick,
        "recorded_frame_count": len(frames),
        "paint_flow_frame_count": len(paint_frames),
        "actions": action_rows,
        "actionmodband_observations": actionmodband,
        "frames": {
            "png_sequence": png_entries,
            "raw_grid_sequence": raw_entries,
            "before": before_meta,
            "after": after_meta,
            "sheet": sheet_meta,
            "paint_flow_sheet": paint_sheet_meta,
        },
    }
    write_json(level_dir / "controlled_play_result.json", result)
    write_level_page(level_dir, result)
    return result


def write_level_page(level_dir: Path, result: dict[str, Any]) -> None:
    actions = result["actions"]
    rows = []
    for action in actions:
        before = action["before_state"]
        after = action["after_state"]
        rows.append(
            "<tr>"
            f"<td>{action['index']}</td>"
            f"<td>{html.escape(action['phase'])}</td>"
            f"<td>{html.escape(action['label'])}</td>"
            f"<td>{html.escape(action['action'])}</td>"
            f"<td><code>{html.escape(json.dumps(action['data'], sort_keys=True))}</code></td>"
            f"<td>{html.escape(str(action.get('returned_frame_count')))}</td>"
            f"<td>{html.escape(str(before.get('selected_coord')))} -> {html.escape(str(after.get('selected_coord')))}</td>"
            f"<td>{html.escape(str(action['delta'].get('changed_pixels')))}</td>"
            f"<td>{html.escape(str(before.get('levels_completed')))} -> {html.escape(str(after.get('levels_completed')))}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SP80 level {result['level_index']:02d} controlled playback</title>
<style>
body {{ margin:0; background:#0b0f15; color:#e8edf4; font-family:Inter,system-ui,Segoe UI,sans-serif; }}
main {{ max-width:1600px; margin:0 auto; padding:28px; }}
a {{ color:#8cc8ff; }}
.boundary {{ color:#b9c2d2; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin:16px 0 24px; }}
.card {{ background:#111923; border:1px solid #263345; border-radius:6px; padding:12px; }}
.card b {{ display:block; font-size:12px; color:#94a3b8; text-transform:uppercase; }}
.card span {{ font-size:22px; }}
.deck {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(520px,1fr)); gap:16px; }}
figure {{ margin:0 0 18px; background:#101722; border:1px solid #263345; border-radius:6px; padding:12px; }}
figcaption {{ color:#cbd5e1; margin-bottom:8px; }}
img {{ max-width:100%; height:auto; image-rendering:pixelated; }}
.frame {{ width:640px; }}
.sheet {{ width:100%; }}
table {{ width:100%; border-collapse:collapse; margin-top:20px; font-size:14px; }}
th,td {{ border-bottom:1px solid #263345; padding:8px; text-align:left; vertical-align:top; }}
th {{ color:#cbd5e1; background:#111923; }}
code {{ color:#d7e3f4; }}
</style>
</head>
<body>
<main>
<h1>SP80 level {result['level_index']:02d} controlled playback</h1>
<p class="boundary">Controlled recorded action playback. Frames are raw environment renders saved after each click/move/settle action; the page does not claim this was discovered by an autonomous policy.</p>
<p><a href="controlled_play_result.json">Raw result JSON</a></p>
<section class="cards">
<div class="card"><b>Status</b><span>{html.escape(result['status'])}</span></div>
<div class="card"><b>Completed delta</b><span>{result['levels_completed_delta']}</span></div>
<div class="card"><b>Solution actions</b><span>{result['solution_action_count']}</span></div>
<div class="card"><b>Settle ticks</b><span>{result['settle_tick_count']}</span></div>
<div class="card"><b>Frames</b><span>{result['recorded_frame_count']}</span></div>
<div class="card"><b>Paint frames</b><span>{result.get('paint_flow_frame_count', 0)}</span></div>
<div class="card"><b>ActionMod responses</b><span>{(result.get('actionmodband_observations') or {}).get('response_count')}</span></div>
</section>
<section class="deck">
<figure><figcaption>Before</figcaption><img class="frame" src="before.png"></figure>
<figure><figcaption>After</figcaption><img class="frame" src="after.png"></figure>
</section>
{"<figure><figcaption>ACTION5 paint-flow returned frames</figcaption><img class='sheet' src='paint_flow_sheet.png'></figure>" if result.get("paint_flow_frame_count") else ""}
<figure><figcaption>Full action frame sheet</figcaption><img class="sheet" src="frame_sheet.png"></figure>
<h2>Action Log</h2>
<table>
<thead><tr><th>#</th><th>Phase</th><th>Label</th><th>Action</th><th>Data</th><th>Returned frames</th><th>Selected coord</th><th>Changed pixels</th><th>Levels completed</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</main>
</body>
</html>
"""
    (level_dir / "index.html").write_text(page, encoding="utf-8")


def write_index(out_dir: Path, results: list[dict[str, Any]], command: str) -> None:
    cards = []
    for result in results:
        cards.append(
            "<article class='card'>"
            f"<h2>Level {result['level_index']:02d}</h2>"
            f"<p>Status: <b>{html.escape(result['status'])}</b></p>"
            f"<p>Completed delta: {result['levels_completed_delta']} | Frames: {result['recorded_frame_count']} | "
            f"Paint frames: {result.get('paint_flow_frame_count', 0)} | "
            f"ActionMod responses: {(result.get('actionmodband_observations') or {}).get('response_count')} | "
            f"Settle ticks: {result['settle_tick_count']}</p>"
            f"<p><a href='level_{result['level_index']:02d}/index.html'>Open large level {result['level_index']:02d} frame page</a> | "
            f"<a href='level_{result['level_index']:02d}/controlled_play_result.json'>raw JSON</a></p>"
            + (
                f"<a href='level_{result['level_index']:02d}/index.html'><img src='level_{result['level_index']:02d}/paint_flow_sheet.png'></a>"
                if result.get("paint_flow_frame_count")
                else ""
            )
            + f"<a href='level_{result['level_index']:02d}/index.html'><img src='level_{result['level_index']:02d}/frame_sheet.png'></a>"
            "</article>"
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SP80 controlled playback levels 01-02</title>
<style>
body {{ margin:0; background:#0b0f15; color:#e8edf4; font-family:Inter,system-ui,Segoe UI,sans-serif; }}
main {{ max-width:1700px; margin:0 auto; padding:28px; }}
a {{ color:#8cc8ff; }}
.boundary {{ color:#b9c2d2; }}
.grid {{ display:grid; grid-template-columns:1fr; gap:28px; }}
.card {{ background:#101722; border:1px solid #263345; border-radius:6px; padding:14px; }}
img {{ width:100%; height:auto; image-rendering:pixelated; border:1px solid #263345; }}
code {{ color:#d7e3f4; }}
</style>
</head>
<body>
<main>
<h1>SP80 controlled playback levels 01-02</h1>
<p class="boundary">Controlled recorded action playback for levels requested as rounds 1 and 2. This proves the listed click/move/settle sequences completed the levels under the current SP80 runtime; it does not prove autonomous discovery.</p>
<p>Command: <code>{html.escape(command)}</code></p>
<p><a href="manifest.json">Manifest</a></p>
<section class="grid">
{''.join(cards)}
</section>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "codexlab" / "runs" / "sp80_known_solution_play_01_02")
    parser.add_argument("--scale", type=int, default=10)
    parser.add_argument("--max-settle-ticks", type=int, default=20)
    args = parser.parse_args()
    if not args.out_dir.is_absolute():
        args.out_dir = (ROOT / args.out_dir).resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    results = [run_level(level, args.out_dir, scale=args.scale, max_settle_ticks=args.max_settle_ticks) for level in args.levels]
    manifest = {
        "schema": "codexlab.sp80_known_solution_playback_manifest.v1",
        "condition": "controlled SP80 level 1 and 2 recorded solution playback",
        "controlled": True,
        "proxy": False,
        "command": command,
        "working_directory": str(ROOT),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "levels": [
            {
                "level_index": result["level_index"],
                "status": result["status"],
                "levels_completed_delta": result["levels_completed_delta"],
                "result_path": rel(args.out_dir / f"level_{result['level_index']:02d}" / "controlled_play_result.json"),
                "page_path": rel(args.out_dir / f"level_{result['level_index']:02d}" / "index.html"),
            }
            for result in results
        ],
    }
    write_json(args.out_dir / "manifest.json", manifest)
    write_index(args.out_dir, results, command)
    print(json.dumps({"status": "passed" if all(r["status"] == "won" for r in results) else "failed", "out_dir": rel(args.out_dir), "levels": manifest["levels"]}, indent=2))
    return 0 if all(r["status"] == "won" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
