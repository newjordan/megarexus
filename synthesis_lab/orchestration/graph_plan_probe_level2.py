"""Bounded graph-plan scout for Megarexus live level-2 SP80 state.

This is a scout, not model capability evidence. It reaches level 2 through
the same learned catch path plus the controlled-scout level-1 graph plan,
then tests anchor-derived four-bar level-2 plans in the real offline env.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
from datetime import datetime
from itertools import product
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
from collect_plumbing_play import GAME_ID, click_and_move, reach_next_level, relocate  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from perturb_gate import action_rows, frame_hash, last_grid, replay_actions  # noqa: E402


ENV_DIR = _REPO_ROOT / "environment_files"
LEVEL1_GRAPH_PLAN = [(1, 2, 0), (0, 3, 0), (2, 1, 0)]
AID_LABEL = {1: "up", 2: "down", 3: "action3", 4: "action4", 5: "pour", 6: "select"}
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


def load_model(ckpt_path: Path, device: str) -> MegarexusCNN:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    stack = int(payload["config"].get("stack_frames", 2))
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    model.load_state_dict(payload["model"])
    return model


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


def anchor_xs(grid: np.ndarray) -> list[int]:
    comps = frame_components(grid)
    xs = {
        int(c["center_xy"][0])
        for c in comps
        if (
            int(c["bbox_xyxy"][1]) <= 12 and int(c.get("area", 0)) >= 32
        )
        or (
            int(c["bbox_xyxy"][1]) >= 52 and int(c.get("area", 0)) <= 24
        )
    }
    return sorted(xs)


def dx_options_for(comp: dict[str, Any], anchors: list[int]) -> list[int]:
    cx = int(comp["center_xy"][0])
    opts = {
        int(np.clip(int(round((target_x - cx) / 4.0)), -3, 3))
        for target_x in anchors
    }
    opts.add(0)
    return sorted(opts)


def build_prefix_to_level2(
    *,
    arcade: Arcade,
    model: MegarexusCNN,
    device: str,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = arcade.make(GAME_ID, seed=seed)
    fd = env.reset()
    cur = last_grid(fd)
    prev = cur
    rec = EpisodeRecorder()
    fd, cur, prev, reached_l1, reach_pours = reach_next_level(model, device, env, rec, fd, cur, prev, rng)
    if not reached_l1 or not game_live(fd):
        return {
            "status": "failed_reach_level1",
            "reached_level1": bool(reached_l1),
            "levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
        }
    level1_movables = midband_movables(cur)
    if len(level1_movables) != 3:
        return {
            "status": f"failed_level1_roster:{len(level1_movables)}",
            "reached_level1": True,
            "levels_completed": int(getattr(fd, "levels_completed", 0) or 0),
            "level1_movables": [component_summary(c, i) for i, c in enumerate(level1_movables)],
        }
    fd, cur, _roster, moved_counts = execute_plan(env, rec, fd, cur, level1_movables, LEVEL1_GRAPH_PLAN)
    pre_pour_hash = frame_hash(cur)
    if game_live(fd):
        prev_grid = cur
        fd, cur, frames = step(env, rec, cur, 5)
        prev = np.asarray(frames[-2], dtype=np.int8) if len(frames) >= 2 else prev_grid
    level2 = int(getattr(fd, "levels_completed", 0) or 0) if fd else 0
    prefix = action_rows(rec.steps)
    return {
        "status": "ok" if level2 >= 2 and game_live(fd) else "failed_reach_level2",
        "levels_completed": level2,
        "reach_pours": int(reach_pours),
        "level1_moved_counts": [int(v) for v in moved_counts],
        "level1_pre_pour_frame_sha256": pre_pour_hash,
        "level2_frame_sha256": frame_hash(cur),
        "prefix_actions": prefix,
        "prefix_action_count": len(prefix),
        "level2_movables": [component_summary(c, i) for i, c in enumerate(midband_movables(cur))],
        "level2_anchor_xs": anchor_xs(cur),
    }


def make_horizontal_candidates(movables: list[dict[str, Any]], anchors: list[int]) -> list[dict[str, Any]]:
    if len(movables) != 4:
        return []
    options = [dx_options_for(comp, anchors) for comp in movables]
    orders = {
        "middle_first": [1, 0, 2, 3],
        "top_down": [0, 1, 2, 3],
        "bottom_up": [3, 2, 1, 0],
    }
    seen: set[tuple[tuple[int, int, int], ...]] = set()
    candidates: list[dict[str, Any]] = []
    for order_name, order in orders.items():
        for dx_tuple in product(*options):
            plan = [(rank, int(dx_tuple[rank]), 0) for rank in order if int(dx_tuple[rank]) != 0]
            key = tuple(plan)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "name": f"{order_name}_dx_{'_'.join(str(int(v)) for v in dx_tuple)}",
                    "order": order_name,
                    "representation": "anchor_horizontal",
                    "dx_by_y_rank": [int(v) for v in dx_tuple],
                    "dy_by_y_rank": [0 for _ in dx_tuple],
                    "plan": [[int(v) for v in item] for item in plan],
                    "action_items": [
                        {"movable": int(m), "moves": [int(a) for a in moves]}
                        for m, moves in plan_actions(plan, movables)
                    ],
                    "boundary": "anchor-derived four-bar horizontal plan; not model-selected",
                }
            )
    return candidates


def _with_optional_vertical(
    *,
    item: dict[str, Any],
    vertical_rank: int | None,
    vertical_aid: int | None,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(item))
    copied["vertical_probe"] = None
    if vertical_rank is None or vertical_aid is None:
        return copied
    found = False
    for action_item in copied["action_items"]:
        if int(action_item["movable"]) == int(vertical_rank):
            action_item["moves"].append(int(vertical_aid))
            found = True
            break
    if not found:
        copied["action_items"].append({"movable": int(vertical_rank), "moves": [int(vertical_aid)]})
    dy = [0, 0, 0, 0]
    dy[int(vertical_rank)] = -1 if int(vertical_aid) == 1 else 1
    copied["dy_by_y_rank"] = dy
    copied["vertical_probe"] = {
        "movable": int(vertical_rank),
        "action_id": int(vertical_aid),
        "action_label": AID_LABEL.get(int(vertical_aid), str(vertical_aid)),
        "timing": "after_horizontal_moves",
    }
    copied["name"] = f"{copied['name']}_v{vertical_rank}_{AID_LABEL.get(int(vertical_aid), vertical_aid)}"
    return copied


def make_action_template_candidates(
    *,
    count_delta: int,
    include_vertical_single: bool,
) -> list[dict[str, Any]]:
    """Generate raw-action-ID four-bar candidates around the controlled level-2 template.

    This does not translate through dx semantics. Existing runners disagree in
    comments about ACTION3/ACTION4 names, so the candidate records preserve raw
    action IDs and let measured environment movement/outcome decide.
    """
    templates = [
        {
            "template": "codexlab_solution_level2_action_ids",
            "directions": [4, 3, 3, 4],
            "base_counts": [1, 2, 2, 1],
        },
        {
            "template": "codexlab_solution_level2_inverse_action_ids",
            "directions": [3, 4, 4, 3],
            "base_counts": [1, 2, 2, 1],
        },
    ]
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
    delta = max(0, int(count_delta))
    for template in templates:
        count_options = [
            range(max(0, int(base) - delta), int(base) + delta + 1)
            for base in template["base_counts"]
        ]
        for counts in product(*count_options):
            if not any(int(c) for c in counts):
                continue
            action_items = []
            for rank, (aid, count) in enumerate(zip(template["directions"], counts, strict=True)):
                moves = [int(aid)] * int(count)
                if moves:
                    action_items.append({"movable": int(rank), "moves": moves})
            key = tuple((int(row["movable"]), tuple(int(v) for v in row["moves"])) for row in action_items)
            if key in seen:
                continue
            seen.add(key)
            base = {
                "name": f"{template['template']}_counts_{'_'.join(str(int(c)) for c in counts)}",
                "order": "top_down_y_rank",
                "template": template["template"],
                "representation": "raw_action_id_template",
                "dx_by_y_rank": None,
                "dy_by_y_rank": [0, 0, 0, 0],
                "count_by_y_rank": [int(c) for c in counts],
                "direction_action_id_by_y_rank": [int(a) for a in template["directions"]],
                "plan": [
                    [int(rank), int(template["directions"][rank]), int(count)]
                    for rank, count in enumerate(counts)
                    if int(count)
                ],
                "action_items": action_items,
                "boundary": "controlled-solution-derived raw action-ID four-bar template; not model-selected",
            }
            candidates.append(base)
            if include_vertical_single:
                for rank in range(4):
                    for aid in (1, 2):
                        candidates.append(
                            _with_optional_vertical(item=base, vertical_rank=rank, vertical_aid=aid)
                        )
    return candidates


def make_candidates(
    movables: list[dict[str, Any]],
    anchors: list[int],
    *,
    family: str,
    count_delta: int,
    include_vertical_single: bool,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if family in ("horizontal", "hybrid"):
        candidates.extend(make_horizontal_candidates(movables, anchors))
    if family in ("action-template", "hybrid"):
        candidates.extend(
            make_action_template_candidates(
                count_delta=count_delta,
                include_vertical_single=include_vertical_single,
            )
        )
    if candidate_limit > 0:
        candidates = candidates[: int(candidate_limit)]
    return candidates


def execute_action_items(env, rec, fd, cur, movables, action_items: list[dict[str, Any]]):
    roster = list(movables)
    moved_counts = []
    for item in action_items:
        m = int(item["movable"])
        moves = [int(a) for a in item.get("moves", [])]
        if m < 0 or m >= len(roster):
            moved_counts.append(0)
            continue
        comp = relocate(cur, roster[m]) or roster[m]
        i = 0
        while i < len(moves) and game_live(fd):
            aid = int(moves[i])
            n = 1
            while i + n < len(moves) and int(moves[i + n]) == aid:
                n += 1
            fd, cur, _prev, comp, moved, _spent = click_and_move(env, rec, fd, cur, comp, aid, n)
            moved_counts.append(int(moved))
            i += n
        roster[m] = comp
        if not game_live(fd):
            break
    return fd, cur, roster, moved_counts


def execute_candidate(
    *,
    arcade: Arcade,
    seed: int,
    prefix_actions: list[dict[str, Any]],
    candidate: dict[str, Any],
    image_dir: Path,
    render_images: bool,
) -> dict[str, Any]:
    env = arcade.make(GAME_ID, seed=seed)
    fd, cur, _prev, replay_trace = replay_actions(env, prefix_actions)
    before_grid = cur.copy()
    before_hash = frame_hash(before_grid)
    movables = midband_movables(cur)
    rec = EpisodeRecorder()
    status = "not_started"
    moved_counts: list[int] = []
    if len(movables) == 4 and game_live(fd):
        fd, cur, _roster, moved_counts = execute_action_items(
            env, rec, fd, cur, movables, candidate["action_items"]
        )
        if game_live(fd):
            fd, cur, _frames = step(env, rec, cur, 5)
        status = "completed"
    else:
        status = f"insufficient_level2_midband_movables:{len(movables)}"
    levels = int(getattr(fd, "levels_completed", 0) or 0) if fd else 0
    passed = levels >= 3
    before_image = after_image = None
    if render_images or passed:
        before_rel = image_dir / f"{candidate['name']}_before.png"
        after_rel = image_dir / f"{candidate['name']}_after.png"
        render_grid(before_grid, frame_components(before_grid), before_rel)
        render_grid(cur, frame_components(cur), after_rel)
        before_image = str(before_rel)
        after_image = str(after_rel)
    return {
        "name": candidate["name"],
        "status": status,
        "levels_completed": levels,
        "passed_levels_ge_3": bool(passed),
        "final_state": str(getattr(fd, "state", None)) if fd else "None",
        "candidate_actions": len(action_rows(rec.steps)),
        "moved_counts": [int(v) for v in moved_counts],
        "replay_action_count": len(replay_trace),
        "before_frame_sha256": before_hash,
        "after_frame_sha256": frame_hash(cur),
        "dx_by_y_rank": candidate["dx_by_y_rank"],
        "dy_by_y_rank": candidate.get("dy_by_y_rank"),
        "count_by_y_rank": candidate.get("count_by_y_rank"),
        "direction_action_id_by_y_rank": candidate.get("direction_action_id_by_y_rank"),
        "representation": candidate.get("representation"),
        "template": candidate.get("template"),
        "vertical_probe": candidate.get("vertical_probe"),
        "order": candidate["order"],
        "plan": candidate["plan"],
        "action_items": candidate["action_items"],
        "before_image": before_image,
        "after_image": after_image,
    }


def write_html(report: dict[str, Any], out_dir: Path) -> None:
    def esc(value: Any) -> str:
        if isinstance(value, str):
            return html.escape(value)
        return html.escape(json.dumps(value, sort_keys=True))

    winners = [row for row in report["candidate_results"] if row["passed_levels_ge_3"]]
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Megarexus Level-2 Graph Plan Probe</title>",
        "<style>",
        "body{font-family:Inter,system-ui,sans-serif;margin:24px;background:#f8fafc;color:#18202a}",
        "h1,h2,h3{margin:0 0 10px} section{margin:24px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border:1px solid #d7dee8;padding:6px;vertical-align:top} th{background:#edf2f7;text-align:left}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}",
        ".panel{background:white;border:1px solid #d7dee8;border-radius:6px;padding:10px}",
        "img{image-rendering:pixelated;width:100%;max-width:384px;border:1px solid #ccd5e1}",
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}",
        ".pass{color:#11613a;font-weight:700}.fail{color:#9a241d;font-weight:700}",
        "</style>",
        "<h1>Megarexus Level-2 Graph Plan Probe</h1>",
        "<p><b>Boundary:</b> offline four-bar graph scout from replayed prefix to level 2. Not model-selected, not a live scorecard, not solve evidence.</p>",
        f"<p><b>Command:</b> <code>{esc(report['command'])}</code></p>",
        "<section><h2>Summary</h2><table><tr><th>field</th><th>value</th></tr>",
    ]
    for key in (
        "checkpoint",
        "seed",
        "prefix_status",
        "level2_frame_sha256",
        "level2_anchor_xs",
        "level2_movables",
        "candidate_count",
        "candidate_family",
        "candidate_limit",
        "pass_count",
        "wall_seconds",
    ):
        lines.append(f"<tr><td>{key}</td><td><code>{esc(report.get(key))}</code></td></tr>")
    lines.append("</table></section>")
    lines.append("<section><h2>Winning Candidates</h2>")
    if winners:
        lines.append("<div class='grid'>")
        for row in winners[:24]:
            lines.append("<div class='panel'>")
            lines.append(f"<h3>{esc(row['name'])}</h3>")
            if row.get("after_image"):
                lines.append(f"<a href='{html.escape(row['after_image'])}'><img src='{html.escape(row['after_image'])}'></a>")
            lines.append(f"<p>levels={row['levels_completed']} plan=<code>{esc(row['plan'])}</code></p>")
            lines.append("</div>")
        lines.append("</div>")
    else:
        lines.append("<p>No candidate reached level 3.</p>")
    lines.append("</section>")
    lines.append("<section><h2>Candidate Table</h2><table><tr><th>name</th><th>pass</th><th>levels</th><th>actions</th><th>moved</th><th>repr</th><th>dx</th><th>dy</th><th>action items</th><th>images</th></tr>")
    for row in report["candidate_results"]:
        cls = "pass" if row["passed_levels_ge_3"] else "fail"
        image_cell = ""
        if row.get("after_image"):
            image_cell = f"<a href='{html.escape(row['before_image'])}'>before</a> <a href='{html.escape(row['after_image'])}'>after</a>"
        lines.append(
            f"<tr><td>{esc(row['name'])}</td><td class='{cls}'>{esc(row['passed_levels_ge_3'])}</td>"
            f"<td>{row['levels_completed']}</td><td>{row['candidate_actions']}</td>"
            f"<td><code>{esc(row['moved_counts'])}</code></td><td><code>{esc(row.get('representation'))}</code></td>"
            f"<td><code>{esc(row.get('dx_by_y_rank'))}</code></td><td><code>{esc(row.get('dy_by_y_rank'))}</code></td>"
            f"<td><code>{esc(row['action_items'])}</code></td><td>{image_cell}</td></tr>"
        )
    lines.append("</table></section>")
    lines.append("<section><h2>Raw Artifacts</h2><ul><li><a href='graph_plan_probe_level2.json'>graph_plan_probe_level2.json</a></li></ul></section>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20280001)
    parser.add_argument("--render-all-images", action="store_true")
    parser.add_argument("--candidate-family", choices=["horizontal", "action-template", "hybrid"], default="horizontal")
    parser.add_argument("--template-count-delta", type=int, default=1)
    parser.add_argument("--include-vertical-single", action="store_true")
    parser.add_argument("--candidate-limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    image_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt.resolve(), device)
    source_hashes = {
        "graph_plan_probe_level2.py": sha256_file(Path(__file__).resolve()),
        "arrange_probe.py": sha256_file(_LAB_ROOT / "orchestration" / "arrange_probe.py"),
        "collect_plumbing_play.py": sha256_file(_LAB_ROOT / "orchestration" / "collect_plumbing_play.py"),
        "perturb_gate.py": sha256_file(_LAB_ROOT / "orchestration" / "perturb_gate.py"),
    }
    condition = {
        "schema": "synthesis_lab.condition.graph_plan_probe_level2.v1",
        "condition_name": (
            "stage17_level2_fourbar_raw_action_template_probe"
            if args.candidate_family == "action-template"
            else "level2_fourbar_graph_plan_probe"
        ),
        "run_label": "scout",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "natural_language_condition": (
            "Replay the learned catch plus controlled level-1 graph prefix to level 2, then test "
            f"{args.candidate_family} four-bar level-2 candidate plans in the offline SP80 environment."
        ),
        "command": " ".join(sys.argv),
        "working_directory": str(_REPO_ROOT),
        "script": str(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "seed": int(args.seed),
        "candidate_family": args.candidate_family,
        "template_count_delta": int(args.template_count_delta),
        "include_vertical_single": bool(args.include_vertical_single),
        "candidate_limit": int(args.candidate_limit),
        "evidence_boundary": "offline controlled scout; not model-selected, not live scorecard evidence",
    }
    condition_path = out_dir / "CONDITION.json"
    condition_path.write_text(json.dumps(condition, indent=1, sort_keys=True), encoding="utf-8")
    arcade = Arcade(
        arc_api_key="anonymous",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )
    t0 = time.time()
    prefix = build_prefix_to_level2(arcade=arcade, model=model, device=device, seed=args.seed)
    candidates: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    if prefix.get("status") == "ok":
        movables = [
            {"bbox_xyxy": row["bbox_xyxy"], "center_xy": row["center_xy"], "area": row["area"]}
            for row in prefix["level2_movables"]
        ]
        candidates = make_candidates(
            movables,
            prefix["level2_anchor_xs"],
            family=args.candidate_family,
            count_delta=args.template_count_delta,
            include_vertical_single=args.include_vertical_single,
            candidate_limit=args.candidate_limit,
        )
        for idx, candidate in enumerate(candidates):
            if idx and idx % 50 == 0:
                print(f"[graph_plan_probe_level2] executed {idx}/{len(candidates)}", flush=True)
            results.append(
                execute_candidate(
                    arcade=arcade,
                    seed=args.seed,
                    prefix_actions=prefix["prefix_actions"],
                    candidate=candidate,
                    image_dir=image_dir,
                    render_images=args.render_all_images,
                )
            )
    report = {
        "schema": "synthesis_lab.graph_plan_probe_level2.v1",
        "created_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "boundary": f"{args.candidate_family} four-bar offline scout; not model-selected and not live scorecard evidence",
        "command": " ".join(sys.argv),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_hashes": source_hashes,
        "condition_path": str(condition_path),
        "condition_sha256": sha256_file(condition_path),
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "device": device,
        "seed": int(args.seed),
        "prefix_status": prefix.get("status"),
        "prefix": {k: v for k, v in prefix.items() if k != "prefix_actions"},
        "prefix_action_count": int(prefix.get("prefix_action_count") or 0),
        "level2_frame_sha256": prefix.get("level2_frame_sha256"),
        "level2_movables": prefix.get("level2_movables", []),
        "level2_anchor_xs": prefix.get("level2_anchor_xs", []),
        "candidate_family": args.candidate_family,
        "template_count_delta": int(args.template_count_delta),
        "include_vertical_single": bool(args.include_vertical_single),
        "candidate_limit": int(args.candidate_limit),
        "candidate_count": len(candidates),
        "pass_count": sum(1 for row in results if row["passed_levels_ge_3"]),
        "candidate_results": results,
        "wall_seconds": round(time.time() - t0, 1),
    }
    report_path = out_dir / "graph_plan_probe_level2.json"
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    write_html(report, out_dir)
    print(f"[graph_plan_probe_level2] pass_count={report['pass_count']}/{len(results)}")
    print(f"[graph_plan_probe_level2] report={report_path}")
    print(f"[graph_plan_probe_level2] review={out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
