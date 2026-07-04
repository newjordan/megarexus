"""Static arrange-search diagnostic for saved Megarexus policy-frame traces.

This is not a gate and not solve evidence. It replays already-recorded
`policy_frames.npz` calls through the adapter to recover action/plan decisions,
then scores the current arrange search landscape from the exact saved frames.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
for _entry in (
    _LAB_ROOT,
    _LAB_ROOT / "orchestration",
    _LAB_ROOT / "experiments" / "megarexus_cnn",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from arrange_probe import (  # noqa: E402
    exhaustive_x_plans,
    imagine_plan,
    plan_actions,
    rank_movables_if_clicked,
)
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from megarexus_live_policy import MegarexusArcPolicy, plan_key  # noqa: E402


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


def serialise_plan(plan: Any) -> list[list[int]]:
    return [[int(v) for v in item] for item in plan]


def serialise_bbox(comp: dict[str, Any]) -> dict[str, Any]:
    return {
        "bbox_xyxy": [int(v) for v in comp["bbox_xyxy"]],
        "center_xy": [int(v) for v in comp["center_xy"]],
        "area": int(comp.get("area", 0)),
    }


def action_items(plan: Any, movables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"movable": int(m), "moves": [int(a) for a in moves]}
        for m, moves in plan_actions(plan, movables)
    ]


def render_grid(grid: np.ndarray, path: Path, scale: int = 8) -> None:
    arr = np.asarray(grid, dtype=np.int16)
    arr = np.clip(arr, 0, len(PALETTE) - 1)
    image = Image.fromarray(PALETTE[arr], mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def load_episode_frames(episode_dir: Path) -> list[dict[str, Any]]:
    meta_path = episode_dir / "policy_frames_meta.json"
    frames_path = episode_dir / "policy_frames.npz"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frames_npz = np.load(frames_path)
    rows = []
    for row in meta:
        call = int(row["call"])
        key = f"call{call:04d}"
        frames = np.asarray(frames_npz[key], dtype=np.int8)
        rows.append({**row, "frames": frames})
    return rows


def replay_policy(
    *,
    ckpt: Path,
    episode_rows: list[dict[str, Any]],
    arrange_memory: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    old_flag = os.environ.get("MEGAREXUS_ARRANGE_MEMORY")
    if arrange_memory:
        os.environ["MEGAREXUS_ARRANGE_MEMORY"] = "1"
    else:
        os.environ.pop("MEGAREXUS_ARRANGE_MEMORY", None)
    try:
        policy = MegarexusArcPolicy(checkpoint_path=ckpt)
    finally:
        if old_flag is None:
            os.environ.pop("MEGAREXUS_ARRANGE_MEMORY", None)
        else:
            os.environ["MEGAREXUS_ARRANGE_MEMORY"] = old_flag

    history: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    prev_seen: np.ndarray | None = None
    plan_builds: list[dict[str, Any]] = []
    for row in episode_rows:
        frames = [np.asarray(f, dtype=np.int8) for f in row["frames"]]
        grid = frames[-1]
        prev = frames[-2] if len(frames) >= 2 else (prev_seen if prev_seen is not None else grid)
        before_plan_count = len(policy.plan_summaries)
        obs = {
            "grid": grid,
            "frames": frames,
            "resets": int(row.get("resets") or 0),
            "levels_completed": int(row.get("levels") or 0),
            "state": str(row.get("state", "")),
            "available_actions": [],
            "perception": {},
        }
        aid, x, y = policy(obs, history)
        record = dict(policy.last_action_record or {})
        record["call"] = int(row["call"])
        record["frame_sha256"] = frame_hash(grid)
        record["prev_frame_sha256"] = frame_hash(prev)
        records.append(record)
        history.append(
            {
                "step": len(history),
                "action": int(aid),
                "x": x,
                "y": y,
                "levels": int(row.get("levels") or 0),
                "source": record.get("source"),
                "before_frame_sha256": frame_hash(grid),
            }
        )
        for idx in range(before_plan_count, len(policy.plan_summaries)):
            summary = dict(policy.plan_summaries[idx])
            summary["call"] = int(row["call"])
            summary["frame_sha256"] = frame_hash(grid)
            summary["prev_frame_sha256"] = frame_hash(prev)
            plan_builds.append(summary)
        prev_seen = grid.copy()
    return records, policy.export_trace(), plan_builds


def local_y_variants(seed_plans: list[Any]) -> list[Any]:
    """Small vertical probe around top x-plans; not exhaustive."""
    variants: list[Any] = []
    seen: set[tuple] = set()
    for plan in seed_plans:
        base = [(int(m), int(dx), int(dy)) for m, dx, dy in plan]
        for m_idx in range(3):
            for dy in (-1, 1):
                candidate = []
                found = False
                for m, dx, old_dy in base:
                    if m == m_idx:
                        candidate.append((m, dx, int(np.clip(old_dy + dy, -1, 1))))
                        found = True
                    else:
                        candidate.append((m, dx, old_dy))
                if not found:
                    candidate.append((m_idx, 0, dy))
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate)
    return variants


@torch.no_grad()
def score_landscape(
    *,
    ckpt: Path,
    episode_rows: list[dict[str, Any]],
    plan_builds: list[dict[str, Any]],
    top_n: int,
    y_probe_seeds: int,
) -> list[dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = MegarexusArcPolicy(checkpoint_path=ckpt)
    model = policy.model
    rows_by_call = {int(row["call"]): row for row in episode_rows}
    landscapes: list[dict[str, Any]] = []
    prev_seen: dict[int, np.ndarray] = {}
    last_grid: np.ndarray | None = None
    for row in episode_rows:
        frames = [np.asarray(f, dtype=np.int8) for f in row["frames"]]
        grid = frames[-1]
        prev = frames[-2] if len(frames) >= 2 else (last_grid if last_grid is not None else grid)
        prev_seen[int(row["call"])] = prev
        last_grid = grid

    for build in plan_builds:
        call = int(build["call"])
        row = rows_by_call[call]
        grid = np.asarray(row["frames"][-1], dtype=np.int8)
        prev = prev_seen[call]
        all_components = [serialise_bbox(comp) for comp in frame_components(grid)]
        ranked = rank_movables_if_clicked(model, device, grid, prev, top_k=3)
        movables = [comp for _score, comp in ranked]
        if len(movables) < 3:
            landscapes.append(
                {
                    "call": call,
                    "status": "insufficient_roster",
                    "component_count": len(all_components),
                    "components": all_components,
                }
            )
            continue
        scored = []
        for plan in exhaustive_x_plans():
            score, imagined_end = imagine_plan(model, device, grid, prev, movables, plan)
            scored.append(
                {
                    "family": "current_exhaustive_x",
                    "score": float(score),
                    "plan": serialise_plan(plan),
                    "plan_key": serialise_plan(plan_key(plan)),
                    "items": action_items(plan, movables),
                    "imagined_end_sha256": frame_hash(imagined_end),
                    "is_null": all(int(dx) == 0 and int(dy) == 0 for _m, dx, dy in plan),
                }
            )
        scored.sort(key=lambda item: -float(item["score"]))
        top_x = scored[: max(top_n, y_probe_seeds)]
        y_scored = []
        if y_probe_seeds > 0:
            for plan in local_y_variants([item["plan"] for item in top_x[:y_probe_seeds]]):
                score, imagined_end = imagine_plan(model, device, grid, prev, movables, plan)
                y_scored.append(
                    {
                        "family": "local_y_probe_not_exhaustive",
                        "score": float(score),
                        "plan": serialise_plan(plan),
                        "plan_key": serialise_plan(plan_key(plan)),
                        "items": action_items(plan, movables),
                        "imagined_end_sha256": frame_hash(imagined_end),
                        "is_null": False,
                    }
                )
            y_scored.sort(key=lambda item: -float(item["score"]))
        landscapes.append(
            {
                "call": call,
                "status": "scored",
                "frame_sha256": frame_hash(grid),
                "component_count": len(all_components),
                "components": all_components,
                "roster": [serialise_bbox(comp) for comp in movables],
                "current_x_candidates_scored": len(scored),
                "current_x_top": scored[:top_n],
                "local_y_probe_boundary": "small vertical perturbation around top current-x plans; not exhaustive and not environment execution",
                "local_y_candidates_scored": len(y_scored),
                "local_y_top": y_scored[:top_n],
            }
        )
    return landscapes


def write_html(report: dict[str, Any], out_dir: Path) -> None:
    def esc(value: Any) -> str:
        return html.escape(json.dumps(value, sort_keys=True) if not isinstance(value, str) else value)

    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Megarexus Arrange Trace Diagnostic</title>",
        "<style>",
        "body{font-family:Inter,system-ui,sans-serif;margin:24px;background:#f8fafc;color:#18202a}",
        "h1,h2,h3{margin:0 0 10px} section{margin:24px 0}",
        "table{border-collapse:collapse;width:100%;font-size:13px;background:white}",
        "th,td{border:1px solid #d7dee8;padding:6px;vertical-align:top} th{background:#edf2f7;text-align:left}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}",
        ".panel{background:white;border:1px solid #d7dee8;border-radius:6px;padding:12px}",
        "img{image-rendering:pixelated;width:100%;max-width:512px;border:1px solid #ccd5e1}",
        "code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}",
        "</style>",
        "<h1>Megarexus Arrange Trace Diagnostic</h1>",
        "<p><b>Boundary:</b> static replay and model-score review from saved policy frames. Not a gate, not a live scorecard, not solve evidence.</p>",
        f"<p><b>Command:</b> <code>{esc(report['command'])}</code></p>",
        "<section><h2>Summary</h2><table>",
        "<tr><th>field</th><th>value</th></tr>",
    ]
    for key in (
        "checkpoint",
        "checkpoint_sha256",
        "arrange_memory_replay",
        "episodes",
        "source_policy_sha256",
        "source_arrange_probe_sha256",
    ):
        lines.append(f"<tr><td>{key}</td><td><code>{esc(report.get(key))}</code></td></tr>")
    lines.extend(["</table></section>"])
    for episode in report["episode_reports"]:
        lines.append(f"<section><h2>{esc(episode['label'])}</h2>")
        lines.append("<div class='grid'>")
        for image in episode["images"]:
            lines.append("<div class='panel'>")
            lines.append(f"<h3>call {image['call']}: {esc(image['source'])}</h3>")
            lines.append(f"<a href='{html.escape(image['path'])}'><img src='{html.escape(image['path'])}'></a>")
            lines.append(f"<p><code>{esc(image['frame_sha256'][:16])}</code></p>")
            lines.append("</div>")
        lines.append("</div>")
        lines.append("<h3>Reconstructed Actions</h3><table><tr><th>call</th><th>source</th><th>action</th><th>x</th><th>y</th><th>info</th></tr>")
        for row in episode["reconstructed_actions"]:
            lines.append(
                f"<tr><td>{row['call']}</td><td>{esc(row.get('source'))}</td>"
                f"<td>{row.get('action')}</td><td>{row.get('x')}</td><td>{row.get('y')}</td>"
                f"<td><code>{esc(row.get('info', {}))}</code></td></tr>"
            )
        lines.append("</table>")
        lines.append("<h3>Plan Builds</h3><table><tr><th>call</th><th>score</th><th>margins</th><th>top5 scores</th><th>plan</th><th>roster</th></tr>")
        for build in episode["plan_builds"]:
            lines.append(
                f"<tr><td>{build.get('call')}</td><td>{build.get('plan_score')}</td>"
                f"<td>5th={build.get('margin_vs_5th')} null={build.get('margin_vs_null')}</td>"
                f"<td><code>{esc(build.get('top5_scores'))}</code></td>"
                f"<td><code>{esc(build.get('plan'))}</code></td>"
                f"<td><code>{esc(build.get('roster_bboxes'))}</code></td></tr>"
            )
        lines.append("</table>")
        lines.append("<h3>Candidate Landscapes</h3>")
        for landscape in episode["candidate_landscapes"]:
            lines.append(f"<h4>call {landscape['call']} - {esc(landscape['status'])}</h4>")
            lines.append("<table><tr><th>family</th><th>score</th><th>plan</th><th>items</th><th>end hash</th></tr>")
            for item in landscape.get("current_x_top", []):
                lines.append(
                    f"<tr><td>{esc(item['family'])}</td><td>{item['score']:.4f}</td>"
                    f"<td><code>{esc(item['plan'])}</code></td><td><code>{esc(item['items'])}</code></td>"
                    f"<td><code>{esc(item['imagined_end_sha256'][:16])}</code></td></tr>"
                )
            for item in landscape.get("local_y_top", []):
                lines.append(
                    f"<tr><td>{esc(item['family'])}</td><td>{item['score']:.4f}</td>"
                    f"<td><code>{esc(item['plan'])}</code></td><td><code>{esc(item['items'])}</code></td>"
                    f"<td><code>{esc(item['imagined_end_sha256'][:16])}</code></td></tr>"
                )
            lines.append("</table>")
            if landscape.get("local_y_top"):
                lines.append(f"<p><b>Boundary:</b> {esc(landscape['local_y_probe_boundary'])}</p>")
        lines.append("</section>")
    lines.append("<section><h2>Raw Artifacts</h2><ul><li><a href='arrange_trace_diagnostic.json'>arrange_trace_diagnostic.json</a></li></ul></section>")
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--arrange-memory-replay", action="store_true")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--y-probe-seeds", type=int, default=3)
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="Only replay the first N saved policy calls. Useful for first-plan diagnostics.",
    )
    parser.add_argument(
        "--skip-landscape",
        action="store_true",
        help="Do not rescore candidates after policy replay; keep policy plan_summaries only.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(args.label)
    while len(labels) < len(args.episode_dir):
        labels.append(args.episode_dir[len(labels)].name)

    report: dict[str, Any] = {
        "schema": "synthesis_lab.arrange_trace_diagnostic.v0",
        "boundary": "static replay and model-score review from saved policy frames; not a gate, not live scorecard evidence",
        "command": " ".join(sys.argv),
        "checkpoint": str(args.ckpt.resolve()),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "source_policy_sha256": sha256_file(_LAB_ROOT / "orchestration" / "megarexus_live_policy.py"),
        "source_arrange_probe_sha256": sha256_file(_LAB_ROOT / "orchestration" / "arrange_probe.py"),
        "arrange_memory_replay": bool(args.arrange_memory_replay),
        "episodes": len(args.episode_dir),
        "episode_reports": [],
    }

    image_root = out_dir / "frames"
    for idx, episode_dir in enumerate(args.episode_dir):
        episode_dir = episode_dir.resolve()
        episode_rows = load_episode_frames(episode_dir)
        if args.max_calls is not None:
            episode_rows = episode_rows[: max(0, int(args.max_calls))]
        actions, trace, plan_builds = replay_policy(
            ckpt=args.ckpt.resolve(),
            episode_rows=episode_rows,
            arrange_memory=args.arrange_memory_replay,
        )
        landscapes = []
        if not args.skip_landscape:
            landscapes = score_landscape(
                ckpt=args.ckpt.resolve(),
                episode_rows=episode_rows,
                plan_builds=plan_builds,
                top_n=args.top_n,
                y_probe_seeds=args.y_probe_seeds,
            )
        interesting_calls = sorted(
            {
                0,
                len(episode_rows) - 1,
                *(int(row["call"]) for row in actions if row.get("source") in {
                    "megarexus_arrange_select",
                    "megarexus_arrange_pour",
                    "megarexus_plan_no_confidence",
                    "megarexus_pour_budget_exhausted",
                }),
            }
        )
        images = []
        rows_by_call = {int(row["call"]): row for row in episode_rows}
        actions_by_call = {int(row["call"]): row for row in actions}
        for call in interesting_calls:
            if call not in rows_by_call:
                continue
            grid = np.asarray(rows_by_call[call]["frames"][-1], dtype=np.int8)
            rel = Path("frames") / f"episode{idx}_call{call:04d}.png"
            render_grid(grid, out_dir / rel)
            images.append(
                {
                    "call": call,
                    "path": str(rel),
                    "source": actions_by_call.get(call, {}).get("source", "n/a"),
                    "frame_sha256": frame_hash(grid),
                }
            )
        source_counts = Counter(str(row.get("source")) for row in actions)
        report["episode_reports"].append(
            {
                "label": labels[idx],
                "episode_dir": str(episode_dir),
                "frames_npz_sha256": sha256_file(episode_dir / "policy_frames.npz"),
                "frames_meta_sha256": sha256_file(episode_dir / "policy_frames_meta.json"),
                "calls": len(episode_rows),
                "source_histogram": dict(source_counts),
                "reconstructed_actions": actions,
                "trace": {k: v for k, v in trace.items() if k != "ledger"},
                "plan_builds": plan_builds,
                "candidate_landscapes": landscapes,
                "images": images,
            }
        )

    report_path = out_dir / "arrange_trace_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    write_html(report, out_dir)
    print(f"[diagnose_arrange_trace] report={report_path}")
    print(f"[diagnose_arrange_trace] review={out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
