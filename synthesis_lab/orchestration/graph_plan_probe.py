"""Bounded graph-plan scout for failed Megarexus perturb seeds.

This is a scout, not model capability evidence. It recreates the Stage 1b
perturbed state, then tries a small set of component-graph-derived plans that
include vertical moves. The purpose is to test whether the current horizontal
search space is missing useful degrees of freedom before changing the adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
for _entry in (
    _REPO_ROOT / "codexlab",
    _LAB_ROOT,
    _LAB_ROOT / "orchestration",
    _LAB_ROOT / "experiments" / "megarexus_cnn",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arrange_probe import execute_plan, plan_actions  # noqa: E402
from collect_catch_play import EpisodeRecorder, game_live, step  # noqa: E402
from collect_plumbing_play import GAME_ID, reach_next_level  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from perturb_gate import (  # noqa: E402
    action_rows,
    frame_hash,
    last_grid,
    perturb_state,
    replay_actions,
)


ENV_DIR = _REPO_ROOT / "environment_files"
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


def render_grid(grid: np.ndarray, comps: list[dict[str, Any]], path: Path, scale: int = 6) -> None:
    arr = np.asarray(grid, dtype=np.int16)
    arr = np.clip(arr, 0, len(PALETTE) - 1)
    image = Image.fromarray(PALETTE[arr], mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    colors = ["#ff3355", "#33aa55", "#3377ff", "#eeaa00", "#aa44ee", "#00aaaa"]
    for idx, comp in enumerate(comps):
        x0, y0, x1, y1 = [int(v) * scale for v in comp["bbox_xyxy"]]
        color = colors[idx % len(colors)]
        draw.rectangle([x0, y0, x1 + scale - 1, y1 + scale - 1], outline=color, width=2)
        draw.text((x0 + 2, y0 + 2), str(idx), fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_model(ckpt_path: Path, device: str) -> MegarexusCNN:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    stack = int(payload["config"].get("stack_frames", 2))
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    model.load_state_dict(payload["model"])
    return model


def component_summary(comp: dict[str, Any], index: int) -> dict[str, Any]:
    x0, y0, x1, y1 = [int(v) for v in comp["bbox_xyxy"]]
    cx, cy = [int(v) for v in comp["center_xy"]]
    return {
        "index": int(index),
        "bbox_xyxy": [x0, y0, x1, y1],
        "center_xy": [cx, cy],
        "area": int(comp.get("area", (x1 - x0 + 1) * (y1 - y0 + 1))),
        "width": x1 - x0 + 1,
        "height": y1 - y0 + 1,
    }


def midband_movables(grid: np.ndarray) -> list[dict[str, Any]]:
    band = [
        comp
        for comp in frame_components(grid)
        if 12 <= int(comp["bbox_xyxy"][1]) <= 48 and int(comp.get("area", 0)) >= 32
    ]
    return sorted(band, key=lambda c: int(c["center_xy"][1]))


def target_x_from_graph(grid: np.ndarray) -> int:
    bottom = [
        comp
        for comp in frame_components(grid)
        if int(comp["bbox_xyxy"][1]) >= 52 and int(comp.get("area", 0)) <= 24
    ]
    if bottom:
        return int(round(float(np.median([int(c["center_xy"][0]) for c in bottom]))))
    return 41


def steps_to_target_x(comp: dict[str, Any], target_x: int) -> int:
    cx = int(comp["center_xy"][0])
    return int(np.clip(int(round((target_x - cx) / 4.0)), -3, 3))


def make_graph_candidates(grid: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    movables = midband_movables(grid)
    target_x = target_x_from_graph(grid)
    if len(movables) != 3:
        return [], [component_summary(c, i) for i, c in enumerate(movables)], target_x
    target_dx = [steps_to_target_x(comp, target_x) for comp in movables]
    x_profiles = [
        ("target_x", target_dx),
        # Human-demo basin from collect_plumbing_play: upper +3, middle +2,
        # lower +1. Perturb seed 20290103 differs exactly on the lower bar
        # if we blindly align every center to the bottom target x.
        ("demo_x", [3, 2, 1]),
        ("hybrid_target_upper_mid_demo_lower", [target_dx[0], target_dx[1], 1]),
    ]
    recipes = [
        ("x_align_only", (0, 0, 0)),
        ("lower_down_1", (0, 0, 1)),
        ("lower_down_2", (0, 0, 2)),
        ("lower_down_3", (0, 0, 3)),
        ("middle_down_1_lower_down_2", (0, 1, 2)),
        ("middle_down_1_lower_down_3", (0, 1, 3)),
        ("middle_down_2_lower_down_3", (0, 2, 3)),
        ("stair_down_1_2_3", (1, 2, 3)),
        ("stair_down_0_1_3", (0, 1, 3)),
        ("all_down_1", (1, 1, 1)),
        ("upper_up_1_middle_down_1_lower_down_3", (-1, 1, 3)),
    ]
    candidates = []
    order = (1, 0, 2)  # demo order by y-rank: middle, upper, lower.
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    for profile_name, dx in x_profiles:
        for recipe_name, dy in recipes:
            plan = [(rank, int(dx[rank]), int(dy[rank])) for rank in order]
            key = tuple(plan)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "name": f"{profile_name}_{recipe_name}",
                    "plan": [[int(v) for v in item] for item in plan],
                    "action_items": [
                        {"movable": int(m), "moves": [int(a) for a in moves]}
                        for m, moves in plan_actions(plan, movables)
                    ],
                    "dx_by_y_rank": [int(v) for v in dx],
                    "dy_by_y_rank": [int(v) for v in dy],
                    "boundary": "graph-authored candidate; not model-selected",
                }
            )
    return candidates, [component_summary(c, i) for i, c in enumerate(movables)], target_x


def recreate_perturbed_state(
    *,
    arcade: Arcade,
    model: MegarexusCNN,
    device: str,
    seed: int,
    perturb_min_rounds: int,
    perturb_max_rounds: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = arcade.make(GAME_ID, seed=seed)
    fd = env.reset()
    cur = last_grid(fd)
    prev = cur
    rec = EpisodeRecorder()
    fd, cur, prev, reached, reach_pours = reach_next_level(model, device, env, rec, fd, cur, prev, rng)
    reach_actions = action_rows(rec.steps)
    perturb_log: list[dict[str, Any]] = []
    perturb_actions: list[dict[str, Any]] = []
    if reached and game_live(fd):
        fd, cur, prev, perturb_log, perturb_actions = perturb_state(
            env=env,
            rec=rec,
            fd=fd,
            cur=cur,
            prev=prev,
            rng=rng,
            min_rounds=perturb_min_rounds,
            max_rounds=perturb_max_rounds,
        )
    return {
        "reached": bool(reached),
        "reach_pours": int(reach_pours),
        "reach_actions": reach_actions,
        "perturb_log": perturb_log,
        "perturb_actions": perturb_actions,
        "perturbed_levels": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
        "perturbed_frame_sha256": frame_hash(cur),
        "cur": cur,
        "prev": prev,
    }


def execute_candidate(
    *,
    arcade: Arcade,
    seed: int,
    replay_actions_prefix: list[dict[str, Any]],
    candidate: dict[str, Any],
    image_dir: Path,
) -> dict[str, Any]:
    env = arcade.make(GAME_ID, seed=seed)
    fd, cur, prev, replay_trace = replay_actions(env, replay_actions_prefix)
    replay_match_hash = frame_hash(cur)
    movables = midband_movables(cur)
    rec = EpisodeRecorder()
    before_grid = cur.copy()
    before_components = frame_components(before_grid)
    status = "not_started"
    moved_counts: list[int] = []
    pre_pour_hash = None
    after_pour_hash = None
    if len(movables) == 3 and game_live(fd):
        plan = [tuple(int(v) for v in item) for item in candidate["plan"]]
        fd, cur, _roster, moved_counts = execute_plan(env, rec, fd, cur, movables, plan)
        pre_pour_hash = frame_hash(cur)
        if game_live(fd):
            fd, cur, _frames = step(env, rec, cur, 5)
        after_pour_hash = frame_hash(cur)
        status = "completed"
    else:
        status = f"insufficient_midband_movables:{len(movables)}"
    levels = int(getattr(fd, "levels_completed", 0) or 0) if fd else 0
    final_state = str(getattr(fd, "state", None)) if fd else "None"

    before_rel = image_dir / f"{candidate['name']}_before.png"
    after_rel = image_dir / f"{candidate['name']}_after.png"
    render_grid(before_grid, before_components, before_rel)
    render_grid(cur, frame_components(cur), after_rel)
    return {
        "name": candidate["name"],
        "status": status,
        "levels_completed": levels,
        "passed_levels_ge_2": bool(levels >= 2),
        "final_state": final_state,
        "actions": len(action_rows(rec.steps)),
        "moved_counts": [int(v) for v in moved_counts],
        "replay_frame_sha256": replay_match_hash,
        "pre_pour_frame_sha256": pre_pour_hash,
        "after_pour_frame_sha256": after_pour_hash,
        "plan": candidate["plan"],
        "action_items": candidate["action_items"],
        "before_image": str(before_rel),
        "after_image": str(after_rel),
        "replay_action_count": len(replay_trace),
    }


def write_html(report: dict[str, Any], out_dir: Path) -> None:
    def esc(value: Any) -> str:
        if isinstance(value, str):
            return html.escape(value)
        return html.escape(json.dumps(value, sort_keys=True))

    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Megarexus Graph Plan Probe</title>",
        "<style>",
        "body{font-family:Inter,system-ui,sans-serif;margin:24px;background:#f8fafc;color:#18202a}",
        "h1,h2,h3{margin:0 0 10px} section{margin:24px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border:1px solid #d7dee8;padding:6px;vertical-align:top} th{background:#edf2f7;text-align:left}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}",
        ".panel{background:white;border:1px solid #d7dee8;border-radius:6px;padding:10px}",
        "img{image-rendering:pixelated;width:100%;max-width:384px;border:1px solid #ccd5e1}",
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}",
        "</style>",
        "<h1>Megarexus Graph Plan Probe</h1>",
        "<p><b>Boundary:</b> graph-authored offline scout from replay-verified perturbed state. Not model-selected, not live scorecard, not solve evidence.</p>",
        f"<p><b>Command:</b> <code>{esc(report['command'])}</code></p>",
        "<section><h2>Summary</h2><table><tr><th>field</th><th>value</th></tr>",
    ]
    for key in (
        "checkpoint",
        "checkpoint_sha256",
        "seed",
        "perturbed_frame_sha256",
        "target_x",
        "midband_movables",
        "pass_count",
        "candidate_count",
    ):
        lines.append(f"<tr><td>{key}</td><td><code>{esc(report.get(key))}</code></td></tr>")
    lines.append("</table></section>")
    lines.append("<section><h2>Candidate Results</h2><table><tr><th>name</th><th>levels</th><th>actions</th><th>moved</th><th>plan</th><th>images</th></tr>")
    for row in report["candidate_results"]:
        lines.append(
            f"<tr><td>{esc(row['name'])}</td><td>{row['levels_completed']}</td>"
            f"<td>{row['actions']}</td><td><code>{esc(row['moved_counts'])}</code></td>"
            f"<td><code>{esc(row['plan'])}</code></td>"
            f"<td><a href='{html.escape(row['before_image'])}'>before</a> "
            f"<a href='{html.escape(row['after_image'])}'>after</a></td></tr>"
        )
    lines.append("</table></section>")
    lines.append("<section><h2>Frame Deck</h2><div class='grid'>")
    for row in report["candidate_results"]:
        lines.append("<div class='panel'>")
        lines.append(f"<h3>{esc(row['name'])} after</h3>")
        lines.append(f"<a href='{html.escape(row['after_image'])}'><img src='{html.escape(row['after_image'])}'></a>")
        lines.append(f"<p>levels={row['levels_completed']} state={esc(row['final_state'])}</p>")
        lines.append("</div>")
    lines.append("</div></section>")
    lines.append("<section><h2>Raw Artifacts</h2><ul><li><a href='graph_plan_probe.json'>graph_plan_probe.json</a></li></ul></section>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20290103)
    parser.add_argument("--perturb-min-rounds", type=int, default=1)
    parser.add_argument("--perturb-max-rounds", type=int, default=3)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    image_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt.resolve(), device)
    arcade = Arcade(
        arc_api_key="anonymous",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    t0 = time.time()
    perturbed = recreate_perturbed_state(
        arcade=arcade,
        model=model,
        device=device,
        seed=args.seed,
        perturb_min_rounds=args.perturb_min_rounds,
        perturb_max_rounds=args.perturb_max_rounds,
    )
    candidates, movables, target_x = make_graph_candidates(perturbed["cur"])
    prefix = perturbed["reach_actions"] + perturbed["perturb_actions"]
    results = [
        execute_candidate(
            arcade=arcade,
            seed=args.seed,
            replay_actions_prefix=prefix,
            candidate=candidate,
            image_dir=image_dir,
        )
        for candidate in candidates
    ]
    report = {
        "schema": "synthesis_lab.graph_plan_probe.v0",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boundary": "graph-authored offline scout; not model-selected, not live scorecard evidence",
        "command": " ".join(sys.argv),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_hashes": {
            "graph_plan_probe.py": sha256_file(Path(__file__).resolve()),
            "perturb_gate.py": sha256_file(_LAB_ROOT / "orchestration" / "perturb_gate.py"),
            "collect_plumbing_play.py": sha256_file(_LAB_ROOT / "orchestration" / "collect_plumbing_play.py"),
        },
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "device": device,
        "seed": int(args.seed),
        "perturb_rounds": [int(args.perturb_min_rounds), int(args.perturb_max_rounds)],
        "reached": perturbed["reached"],
        "reach_pours": perturbed["reach_pours"],
        "perturbed_levels": perturbed["perturbed_levels"],
        "perturbed_frame_sha256": perturbed["perturbed_frame_sha256"],
        "perturb_log": perturbed["perturb_log"],
        "target_x": int(target_x),
        "midband_movables": movables,
        "candidate_count": len(candidates),
        "pass_count": sum(1 for row in results if row["passed_levels_ge_2"]),
        "candidate_results": results,
        "wall_seconds": round(time.time() - t0, 1),
    }
    report_path = out_dir / "graph_plan_probe.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_html(report, out_dir)
    print(f"[graph_plan_probe] pass_count={report['pass_count']}/{len(results)}")
    print(f"[graph_plan_probe] report={report_path}")
    print(f"[graph_plan_probe] review={out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
