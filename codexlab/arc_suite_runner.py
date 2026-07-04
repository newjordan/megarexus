#!/usr/bin/env python3
"""Offline ARC-AGI-3 local-suite runner.

This is a local scout/proxy runner over environment/environment_files. It does
not open a live ARC scorecard unless the underlying harness is changed; this
file intentionally constructs an offline Arcade directly.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

from arc_play_harness import (  # noqa: E402
    LOCAL_ENV_DIR,
    POLICIES,
    _ACT,
    _json_safe,
    grid_of,
    perception,
    tufa_action_policy,
)
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


RESULT_SCHEMA = "codexlab.arc_suite_result.v1"
MANIFEST_SCHEMA = "codexlab.arc_suite_manifest.v1"
TRACE_SCHEMA = "codexlab.arc_suite_trace_event.v1"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(args: list[str], default: str = "unknown") -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def git_dirty() -> bool | str:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
        return bool(out.strip())
    except Exception:
        return "unknown"


def discover_game_ids(env_dir: Path = LOCAL_ENV_DIR) -> list[str]:
    """Return every local environment prefix/id pair under environment_files."""
    game_ids: list[str] = []
    if not env_dir.exists():
        return game_ids
    for prefix_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
        for ident_dir in sorted(p for p in prefix_dir.iterdir() if p.is_dir()):
            if any(ident_dir.glob("*.py")):
                game_ids.append(f"{prefix_dir.name}-{ident_dir.name}")
    return game_ids


def frame_sha256(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=np.int16)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def safe_perception(grid: np.ndarray) -> dict[str, Any]:
    try:
        return perception(grid)
    except Exception as exc:  # noqa: BLE001 - runner records policy input degradation.
        return {"status": "perception_error", "error": repr(exc)}


def build_policy(policy_name: str, game_id: str, solution_root: str | None) -> Callable[[dict[str, Any], list[dict[str, Any]]], tuple[int, int | None, int | None]]:
    if policy_name == "tufa_action":
        return tufa_action_policy(solution_root=solution_root, game_id=game_id)
    if policy_name == "tufa_contra":
        return POLICIES[policy_name](solution_root=solution_root, game_id=game_id)
    factory = POLICIES[policy_name]
    return factory()


def action_id(value: Any) -> int:
    return int(getattr(value, "value", value))


def run_game(
    *,
    arc: Arcade,
    scorecard_id: Any,
    game_id: str,
    policy_name: str,
    max_actions: int,
    solution_root: str | None,
    auto_reset: bool,
    include_perception: bool,
    seed: int | None = None,
    policy: Callable[..., tuple[int, int | None, int | None]] | None = None,
) -> dict[str, Any]:
    started_at = iso_utc_now()
    is_live = os.environ.get("ARC_ALLOW_LIVE", "").strip().lower() in TRUE_VALUES
    arc_mode = "live_scorecard" if is_live else "offline_local_environment_files"
    arc_operation_mode = str(getattr(arc, "operation_mode", OperationMode.OFFLINE))
    result: dict[str, Any] = {
        "game_id": game_id,
        "policy": policy_name,
        "started_at_utc": started_at,
        "max_actions": int(max_actions),
        "arc_mode": arc_mode,
        "arc_operation_mode": arc_operation_mode,
        "status": "not_started",
        "levels_completed": 0,
        "actions": 0,
        "resets": 0,
        "final_state": None,
        "policy_error_count": 0,
        "policy_errors": [],
        "first_frame_sha256": None,
        "last_frame_sha256": None,
    }
    history: list[dict[str, Any]] = []
    policy_errors: list[dict[str, Any]] = []
    resets = 0
    best_levels = 0
    final_state = ""
    try:
        make_kwargs: dict[str, Any] = {"scorecard_id": scorecard_id}
        if seed is not None:
            make_kwargs["seed"] = int(seed)
        env = arc.make(game_id, **make_kwargs)
        if policy is None:
            policy = build_policy(policy_name, game_id, solution_root)
        frame = env.reset()
        first_hash: str | None = None
        for step_index in range(max(0, int(max_actions))):
            grid = grid_of(frame)
            grid_hash = frame_sha256(grid)
            if first_hash is None:
                first_hash = grid_hash
            state = str(getattr(frame, "state", ""))
            if "GAME_OVER" in state and auto_reset:
                frame = env.step(GameAction.RESET)
                resets += 1
                continue
            raw_frames = frame.frame if isinstance(frame.frame, list) else [frame.frame]
            obs = {
                "grid": grid,
                # Full animation frame list for the current FrameData, in
                # engine order. Policies trained on the recorder convention
                # (cur = frames[-1], prev = frames[-2]) need these; obs["grid"]
                # stays frames[0] so existing policies are untouched.
                "frames": [np.asarray(f) for f in raw_frames],
                "resets": int(resets),
                "levels_completed": int(getattr(frame, "levels_completed", 0) or 0),
                "state": state,
                "available_actions": list(getattr(frame, "available_actions", []) or []),
                "perception": safe_perception(grid) if include_perception else {},
            }
            try:
                aid, x, y = policy(obs, history)
            except Exception as exc:  # noqa: BLE001 - record and keep the suite moving.
                policy_errors.append({"step": int(step_index), "error": repr(exc)})
                aid, x, y = 0, None, None
            aid = int(aid)
            act = _ACT.get(aid, GameAction.RESET)
            data = {"x": int(x), "y": int(y)} if aid == 6 and x is not None and y is not None else None
            frame = env.step(act, data=data)
            levels = int(getattr(frame, "levels_completed", 0) or 0)
            best_levels = max(best_levels, levels)
            final_state = str(getattr(frame, "state", ""))
            history.append(
                {
                    "step": int(step_index),
                    "action": aid,
                    "x": x,
                    "y": y,
                    "levels": levels,
                    "before_frame_sha256": grid_hash,
                    "source": getattr(policy, "last_action_source", policy_name),
                }
            )
            if "WIN" in final_state:
                break
        last_grid = grid_of(frame)
        result.update(
            {
                "status": "completed",
                "levels_completed": int(best_levels),
                "actions": len(history),
                "resets": int(resets),
                "final_state": final_state or str(getattr(frame, "state", "")),
                "policy_error_count": len(policy_errors),
                "policy_errors": policy_errors[:8],
                "first_frame_sha256": first_hash,
                "last_frame_sha256": frame_sha256(last_grid),
                "action_trace_sample": history[:12],
                "action_trace_tail": history[-12:] if len(history) > 12 else [],
            }
        )
        export_trace = getattr(policy, "export_trace", None)
        if callable(export_trace):
            result["policy_trace"] = export_trace()
    except Exception as exc:  # noqa: BLE001 - suite rows should fail closed per game.
        result.update(
            {
                "status": "environment_error",
                "error": repr(exc),
                "policy_error_count": len(policy_errors),
                "policy_errors": policy_errors[:8],
                "actions": len(history),
                "resets": int(resets),
            }
        )
    return result


def scorecard_env_rows(scorecard_close: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for env_row in scorecard_close.get("environments") or []:
        if not isinstance(env_row, dict):
            continue
        game_id = str(env_row.get("id") or "")
        if not game_id:
            continue
        runs = [r for r in env_row.get("runs") or [] if isinstance(r, dict)]
        last_run = runs[-1] if runs else {}
        rows[game_id] = {
            "score": float(env_row.get("score") or 0.0),
            "scorecard_actions": int(env_row.get("actions") or 0),
            "scorecard_resets": int(env_row.get("resets") or 0),
            "scorecard_levels_completed": int(env_row.get("levels_completed") or 0),
            "level_count": int(env_row.get("level_count") or 0),
            "level_actions": last_run.get("level_actions") or [],
            "level_baseline_actions": last_run.get("level_baseline_actions") or [],
            "level_scores": last_run.get("level_scores") or [],
            "scorecard_state": last_run.get("state"),
        }
    return rows


def merge_scorecard_rows(games: list[dict[str, Any]], scorecard_close: dict[str, Any]) -> None:
    rows = scorecard_env_rows(scorecard_close)
    for game in games:
        game.update(rows.get(str(game.get("game_id")), {}))


def render_index(result: dict[str, Any], out_dir: Path) -> str:
    games = result.get("games") or []
    rows = []
    for game in games:
        status_class = "ok" if game.get("status") == "completed" and int(game.get("policy_error_count") or 0) == 0 else "warn"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(game.get('game_id')))}</td>"
            f"<td class='{status_class}'>{html.escape(str(game.get('status')))}</td>"
            f"<td>{int(game.get('levels_completed') or 0)}</td>"
            f"<td>{float(game.get('score') or 0.0):.6f}</td>"
            f"<td>{int(game.get('actions') or 0)}</td>"
            f"<td>{int(game.get('resets') or 0)}</td>"
            f"<td>{int(game.get('policy_error_count') or 0)}</td>"
            f"<td>{html.escape(str(game.get('final_state') or game.get('scorecard_state') or ''))}</td>"
            "</tr>"
        )
    top_rows = "".join(rows) or "<tr><td colspan='8'>No games recorded.</td></tr>"
    command = html.escape(str(result.get("command") or ""))
    boundary = html.escape(str(result.get("boundary") or ""))
    title = html.escape(str(result.get("run_id") or "arc_suite"))
    facts = {
        "Policy": result.get("policy"),
        "Games": result.get("game_count"),
        "Max actions/game": result.get("max_actions_per_game"),
        "Aggregate score": f"{float(result.get('aggregate_score') or 0.0):.6f}",
        "Total levels": result.get("total_levels_completed"),
        "Policy errors": result.get("policy_error_count"),
        "ARC mode": result.get("arc_mode"),
    }
    cards = "".join(
        f"<section class='card'><div>{html.escape(str(k))}</div><strong>{html.escape(str(v))}</strong></section>"
        for k, v in facts.items()
    )
    ranking = result.get("ranked_games") or []
    ranking_rows = "".join(
        "<tr>"
        f"<td>{idx + 1}</td>"
        f"<td>{html.escape(str(row.get('game_id')))}</td>"
        f"<td>{int(row.get('levels_completed') or 0)}</td>"
        f"<td>{float(row.get('score') or 0.0):.6f}</td>"
        f"<td>{int(row.get('actions') or 0)}</td>"
        "</tr>"
        for idx, row in enumerate(ranking[:12])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; color: #202124; background: #f6f7f8; }}
    header {{ background: #18202a; color: white; padding: 24px 32px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 10px; font-size: 18px; letter-spacing: 0; }}
    a {{ color: #1155cc; }}
    pre {{ white-space: pre-wrap; background: #fff; border: 1px solid #d6d9de; border-radius: 6px; padding: 12px; overflow-x: auto; }}
    .boundary {{ max-width: 980px; color: #d8e0ea; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ background: #fff; border: 1px solid #d6d9de; border-radius: 6px; padding: 12px; }}
    .card div {{ color: #5f6368; font-size: 12px; }}
    .card strong {{ display: block; font-size: 20px; margin-top: 3px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d6d9de; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e3e6ea; text-align: left; vertical-align: top; }}
    th {{ background: #eceff3; font-size: 12px; color: #3c4043; }}
    .ok {{ color: #137333; font-weight: 650; }}
    .warn {{ color: #a14200; font-weight: 650; }}
    .links {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div class="boundary">{boundary}</div>
    <div class="links">
      <a href="result.json">result.json</a>
      <a href="report_manifest.json">report_manifest.json</a>
      <a href="report_trace.jsonl">report_trace.jsonl</a>
    </div>
  </header>
  <main>
    <section class="cards">{cards}</section>
    <h2>Per-Game Results</h2>
    <table>
      <thead><tr><th>Game</th><th>Status</th><th>Levels</th><th>Score</th><th>Actions</th><th>Resets</th><th>Policy Errors</th><th>Final State</th></tr></thead>
      <tbody>{top_rows}</tbody>
    </table>
    <h2>Ranked Progress</h2>
    <table>
      <thead><tr><th>Rank</th><th>Game</th><th>Levels</th><th>Score</th><th>Actions</th></tr></thead>
      <tbody>{ranking_rows or "<tr><td colspan='5'>No ranked rows.</td></tr>"}</tbody>
    </table>
    <h2>Command</h2>
    <pre>{command}</pre>
    <h2>Boundary</h2>
    <p>This is a local offline scout over copied environment files. It is not a live ARC Prize submission, public leaderboard result, hidden answer route, replay route, or full CONTRA gameplay report.</p>
  </main>
</body>
</html>
"""


def write_trace(result: dict[str, Any], out_dir: Path) -> None:
    trace_path = out_dir / "report_trace.jsonl"
    is_live = result.get("evidence_category") == "live_scorecard" or result.get("arc_mode") == "live_scorecard"
    truth_boundary = (
        "live_arc_scorecard_not_local_environment_files_proxy"
        if is_live
        else "local_offline_environment_files_scout_not_live_leaderboard_claim"
    )
    with trace_path.open("w", encoding="utf-8") as handle:
        for idx, game in enumerate(result.get("games") or []):
            row = {
                "schema": TRACE_SCHEMA,
                "run_id": result.get("run_id"),
                "event_index": idx,
                "event_type": "arc_suite_game_result",
                "game_id": game.get("game_id"),
                "policy": game.get("policy"),
                "status": game.get("status"),
                "levels_completed": game.get("levels_completed"),
                "score": game.get("score"),
                "actions": game.get("actions"),
                "resets": game.get("resets"),
                "policy_error_count": game.get("policy_error_count"),
                "truth_boundary": truth_boundary,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_partial_result(
    *,
    out_dir: Path,
    run_id: str,
    args: argparse.Namespace,
    games: list[dict[str, Any]],
    all_games: list[str],
    scorecard_id: Any,
    command: str,
) -> None:
    """Write a crash-resumable snapshot before scorecard close is available."""
    ranked_games = sorted(
        games,
        key=lambda row: (int(row.get("levels_completed") or 0), float(row.get("score") or 0.0)),
        reverse=True,
    )
    partial = {
        "schema": RESULT_SCHEMA,
        "partial": True,
        "run_id": run_id,
        "updated_at_utc": iso_utc_now(),
        "condition": f"offline local suite scout policy={args.policy} max_actions_per_game={int(args.max_actions)}",
        "run_label": "scout_partial",
        "evidence_category": "proxy",
        "boundary": "Partial local offline environment_files scout. Not a live ARC Prize submission or leaderboard claim.",
        "policy": str(args.policy),
        "max_actions_per_game": int(args.max_actions),
        "game_count": len(games),
        "requested_game_ids": all_games,
        "arc_mode": "offline_local_environment_files",
        "arc_operation_mode": str(OperationMode.OFFLINE),
        "scorecard_id": str(scorecard_id),
        "aggregate_score": sum(float(game.get("score") or 0.0) for game in games),
        "total_levels_completed": sum(int(game.get("levels_completed") or 0) for game in games),
        "policy_error_count": sum(int(game.get("policy_error_count") or 0) for game in games),
        "games": games,
        "ranked_games": [
            {
                "game_id": row.get("game_id"),
                "levels_completed": int(row.get("levels_completed") or 0),
                "score": float(row.get("score") or 0.0),
                "actions": int(row.get("actions") or 0),
                "resets": int(row.get("resets") or 0),
                "policy_error_count": int(row.get("policy_error_count") or 0),
            }
            for row in ranked_games
        ],
        "working_directory": str(ROOT),
        "command": command,
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
    }
    (out_dir / "partial_result.json").write_text(json.dumps(partial, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "partial_index.html").write_text(render_index(partial, out_dir), encoding="utf-8")


def build_manifest(result: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    result_path = out_dir / "result.json"
    trace_path = out_dir / "report_trace.jsonl"
    index_path = out_dir / "index.html"
    is_live = result.get("evidence_category") == "live_scorecard" or result.get("arc_mode") == "live_scorecard"
    source_paths = [
        CODEXLAB / "arc_suite_runner.py",
        CODEXLAB / "arc_play_harness.py",
        CODEXLAB / "arc_live_suite_runner.py",
        ROOT / "synthesis_lab" / "orchestration" / "megarexus_live_policy.py",
    ]
    raw_inputs = (
        [{"path": f"ARC live scorecard {result.get('scorecard_id')}", "sha256": "remote"}]
        if is_live
        else [{"path": repo_relative(LOCAL_ENV_DIR), "sha256": "directory"}]
    )
    boundaries = (
        [
            "Live ARC scorecard opened through the ARC SDK.",
            "Not a local environment_files proxy.",
            "Policy actions are recorded in per-game action traces and any exported policy_trace.",
        ]
        if is_live
        else [
            "Local offline scout over environment/environment_files.",
            "Not a live ARC Prize scorecard or leaderboard claim.",
            "Not a hidden answer, source-derived route, or replay route.",
            "Not a full CONTRA gameplay report because per-action frames and packets are not recorded.",
        ]
    )
    return {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": result.get("created_at_utc"),
        "run_id": result.get("run_id"),
        "report_status": "live_scorecard" if is_live else "proxy",
        "condition": result.get("condition"),
        "working_directory": str(ROOT),
        "command": result.get("command"),
        "git_commit": result.get("git_commit"),
        "git_dirty": result.get("git_dirty"),
        "source_paths": [
            {"path": repo_relative(path), "sha256": sha256_file(path)}
            for path in source_paths
            if path.exists()
        ],
        "raw_inputs": raw_inputs,
        "raw_outputs": [
            {"path": repo_relative(result_path), "sha256": sha256_file(result_path)},
            {"path": repo_relative(trace_path), "sha256": sha256_file(trace_path)},
            {"path": repo_relative(index_path), "sha256": sha256_file(index_path)},
        ],
        "game_count": result.get("game_count"),
        "policy_error_count": result.get("policy_error_count"),
        "verification": {
            "index_exists": index_path.exists(),
            "result_json_exists": result_path.exists(),
            "trace_exists": trace_path.exists(),
            "offline_mode": result.get("arc_mode") == "offline_local_environment_files",
            "live_scorecard": is_live,
        },
        "boundaries": boundaries,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="tufa_action")
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--games", nargs="*", default=None, help="Game IDs such as sp80-589a99af. Defaults to all local IDs.")
    parser.add_argument("--limit", type=int, default=None, help="Limit discovered game list for smoke runs.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--solution-root", default=None)
    parser.add_argument("--no-auto-reset", action="store_true")
    parser.add_argument("--no-perception", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if os.environ.get("ARC_ALLOW_LIVE", "").strip().lower() in TRUE_VALUES:
        print("refusing suite run with ARC_ALLOW_LIVE set; this runner is offline-only", file=sys.stderr)
        return 2
    all_games = list(args.games) if args.games else discover_game_ids()
    if args.limit is not None:
        all_games = all_games[: max(0, int(args.limit))]
    if not all_games:
        print(f"no local games found under {LOCAL_ENV_DIR}", file=sys.stderr)
        return 2

    stamp = utc_stamp()
    policy_slug = "tufa" if args.policy == "tufa_action" else str(args.policy)
    run_id = args.run_id or f"arc_suite_{policy_slug}_{stamp}"
    out_dir = (args.out_dir or (CODEXLAB / "runs" / run_id)).resolve()
    if out_dir.exists():
        print(f"refusing to overwrite existing run dir: {repo_relative(out_dir)}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=False)

    command = " ".join(shlex.quote(part) for part in sys.argv)
    arc = Arcade(arc_api_key="anonymous", environments_dir=str(LOCAL_ENV_DIR), operation_mode=OperationMode.OFFLINE)
    scorecard_id = arc.open_scorecard()
    games: list[dict[str, Any]] = []
    scorecard_close: dict[str, Any] = {}
    try:
        for game_id in all_games:
            game_result = run_game(
                arc=arc,
                scorecard_id=scorecard_id,
                game_id=game_id,
                policy_name=str(args.policy),
                max_actions=int(args.max_actions),
                solution_root=args.solution_root,
                auto_reset=not bool(args.no_auto_reset),
                include_perception=not bool(args.no_perception),
            )
            games.append(game_result)
            write_partial_result(
                out_dir=out_dir,
                run_id=run_id,
                args=args,
                games=games,
                all_games=all_games,
                scorecard_id=scorecard_id,
                command=command,
            )
    finally:
        try:
            scorecard_close = _json_safe(arc.close_scorecard(scorecard_id)) or {}
        except Exception as exc:  # noqa: BLE001 - preserve partial results.
            scorecard_close = {"close_error": repr(exc)}

    merge_scorecard_rows(games, scorecard_close)
    policy_error_count = sum(int(game.get("policy_error_count") or 0) for game in games)
    aggregate_score = float(scorecard_close.get("score") or sum(float(game.get("score") or 0.0) for game in games))
    total_levels_completed = int(
        scorecard_close.get("total_levels_completed")
        or sum(int(game.get("scorecard_levels_completed", game.get("levels_completed") or 0) or 0) for game in games)
    )
    ranked_games = sorted(
        games,
        key=lambda row: (int(row.get("levels_completed") or 0), float(row.get("score") or 0.0)),
        reverse=True,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "run_id": run_id,
        "created_at_utc": iso_utc_now(),
        "condition": f"offline local suite scout policy={args.policy} max_actions_per_game={int(args.max_actions)}",
        "run_label": "scout",
        "evidence_category": "proxy",
        "boundary": "Local offline environment_files scout. Not a live ARC Prize submission or leaderboard claim.",
        "target_reference": {"score_percent": 12.58, "levels_completed": 18, "source": "user-provided plan"},
        "policy": str(args.policy),
        "max_actions_per_game": int(args.max_actions),
        "game_count": len(games),
        "requested_game_ids": all_games,
        "arc_mode": "offline_local_environment_files",
        "arc_operation_mode": str(OperationMode.OFFLINE),
        "scorecard_id": str(scorecard_id),
        "aggregate_score": aggregate_score,
        "total_levels_completed": total_levels_completed,
        "policy_error_count": policy_error_count,
        "games": games,
        "ranked_games": [
            {
                "game_id": row.get("game_id"),
                "levels_completed": int(row.get("levels_completed") or 0),
                "score": float(row.get("score") or 0.0),
                "actions": int(row.get("actions") or 0),
                "resets": int(row.get("resets") or 0),
                "policy_error_count": int(row.get("policy_error_count") or 0),
            }
            for row in ranked_games
        ],
        "scorecard_close": scorecard_close,
        "working_directory": str(ROOT),
        "command": command,
        "git_commit": git_value(["rev-parse", "HEAD"]),
        "git_dirty": git_dirty(),
        "source_paths": [
            {"path": repo_relative(CODEXLAB / "arc_suite_runner.py"), "sha256": sha256_file(CODEXLAB / "arc_suite_runner.py")},
            {"path": repo_relative(CODEXLAB / "arc_play_harness.py"), "sha256": sha256_file(CODEXLAB / "arc_play_harness.py")},
        ],
        "random_seed": "tufa_action_internal_time_hash_seed_not_exposed" if args.policy == "tufa_action" else "policy_default",
    }

    result_path = out_dir / "result.json"
    index_path = out_dir / "index.html"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_trace(result, out_dir)
    index_path.write_text(render_index(result, out_dir), encoding="utf-8")
    manifest = build_manifest(result, out_dir)
    manifest_path = out_dir / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        f"wrote {repo_relative(out_dir)} "
        f"games={len(games)} policy={args.policy} mode=offline_local_environment_files "
        f"score={aggregate_score:.6f} levels={total_levels_completed} policy_errors={policy_error_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
