#!/usr/bin/env python3
"""Live ARC scorecard suite runner.

This opens one online scorecard, runs requested games sequentially under that
same scorecard, closes it, and writes a compact result/review. It is explicitly
live-only and refuses to run without --allow-live.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arc_suite_runner import (  # noqa: E402
    build_manifest,
    discover_game_ids,
    merge_scorecard_rows,
    render_index,
    run_game,
    write_trace,
)

DEFAULT_ENV = Path.home() / ".env"
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_arc_api_key(env_path: Path) -> bool:
    if not env_path.exists():
        return False
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != "ARC_API_KEY":
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ["ARC_API_KEY"] = value
            return True
    return False


def render_live_index(result: dict[str, Any], out_dir: Path) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))

    cards = {
        "Live score": result.get("aggregate_score"),
        "Levels": result.get("total_levels_completed"),
        "Games": result.get("game_count"),
        "Policy errors": result.get("policy_error_count"),
        "Scorecard": result.get("scorecard_id"),
        "Operation mode": result.get("arc_operation_mode"),
        "Competition": result.get("competition_mode_requested"),
    }
    card_html = "".join(f"<div><b>{esc(k)}</b><br>{esc(v)}</div>" for k, v in cards.items())
    source_url = result.get("source_url")
    source_html = (
        f'<p>Source URL: <a href="{esc(source_url)}">{esc(source_url)}</a></p>'
        if source_url
        else "<p>Source URL: none recorded.</p>"
    )
    rows = []
    for row in result.get("ranked_games") or []:
        rows.append(
            "<tr>"
            f"<td>{esc(row.get('game_id'))}</td>"
            f"<td>{esc(row.get('levels_completed'))}</td>"
            f"<td>{esc(row.get('score'))}</td>"
            f"<td>{esc(row.get('actions'))}</td>"
            f"<td>{esc(row.get('resets'))}</td>"
            f"<td>{esc(row.get('policy_error_count'))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>{esc(result.get('run_id'))}</title>
<style>
body{{font-family:system-ui,Arial,sans-serif;margin:32px;max-width:1200px}}
.boundary{{padding:12px;border:1px solid #d8c36a;background:#fff8dc;margin:16px 0}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}}
.cards div{{border:1px solid #ccc;padding:10px;background:#fafafa}}
table{{border-collapse:collapse;width:100%;margin-top:16px}}th,td{{border:1px solid #ccc;padding:7px;text-align:left}}th{{background:#f4f4f4}}
</style>
<h1>{esc(result.get('run_id'))}</h1>
<div class="boundary">Live ARC scorecard suite attempt. This is not an offline proxy. Raw result: <a href="result.json">result.json</a></div>
{source_html}
<div class="cards">{card_html}</div>
<table><thead><tr><th>Game</th><th>Levels</th><th>Score</th><th>Actions</th><th>Resets</th><th>Policy errors</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="tufa_action")
    parser.add_argument("--max-actions", type=int, default=1500)
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--solution-root", default=None)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--competition", action="store_true", help="Open the ARC scorecard with OperationMode.COMPETITION.")
    parser.add_argument("--source-url", default=None, help="Source URL to attach to the opened ARC scorecard.")
    parser.add_argument("--tag", dest="tags", action="append", default=[], help="Tag to attach to the opened ARC scorecard; may be repeated.")
    parser.add_argument("--opaque-json", default=None, help="Optional JSON object to attach as scorecard opaque metadata.")
    parser.add_argument("--no-auto-reset", action="store_true")
    parser.add_argument("--no-perception", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.allow_live and os.environ.get("ARC_ALLOW_LIVE", "").strip().lower() not in TRUE_VALUES:
        print("refusing live suite run without --allow-live or ARC_ALLOW_LIVE=1", file=sys.stderr)
        return 2
    os.environ["ARC_ALLOW_LIVE"] = "1"
    key_loaded = load_arc_api_key(args.env)
    if not key_loaded and not os.environ.get("ARC_API_KEY"):
        print(f"missing ARC_API_KEY in {args.env}", file=sys.stderr)
        return 2
    if key_loaded:
        os.environ["ARC_API_KEY_PATH"] = str(Path.home() / ".arc_live_suite_no_key_file")

    games = list(args.games) if args.games else discover_game_ids()
    if args.limit is not None:
        games = games[: max(0, int(args.limit))]
    if not games:
        print("no games requested", file=sys.stderr)
        return 2
    opaque = None
    if args.opaque_json:
        try:
            opaque = json.loads(args.opaque_json)
        except json.JSONDecodeError as exc:
            print(f"invalid --opaque-json: {exc}", file=sys.stderr)
            return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"arc_live_suite_{args.policy}_{stamp}"
    out_dir = (args.out_dir or (CODEXLAB / "runs" / run_id)).resolve()
    if out_dir.exists():
        print(f"refusing to overwrite existing run dir: {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=False)

    operation_mode = OperationMode.COMPETITION if args.competition else OperationMode.ONLINE
    arc = Arcade(arc_api_key=os.environ["ARC_API_KEY"].strip(), operation_mode=operation_mode)
    scorecard_id = arc.open_scorecard(
        source_url=args.source_url,
        tags=list(args.tags) if args.tags else None,
        opaque=opaque,
    )
    command = " ".join(shlex.quote(part) for part in sys.argv)
    rows: list[dict[str, Any]] = []
    scorecard_close: dict[str, Any] = {}
    try:
        for game_id in games:
            rows.append(
                run_game(
                    arc=arc,
                    scorecard_id=scorecard_id,
                    game_id=game_id,
                    policy_name=str(args.policy),
                    max_actions=int(args.max_actions),
                    solution_root=args.solution_root,
                    auto_reset=not bool(args.no_auto_reset),
                    include_perception=not bool(args.no_perception),
                )
            )
    finally:
        try:
            close_payload = arc.close_scorecard(scorecard_id)
            scorecard_close = close_payload.model_dump() if hasattr(close_payload, "model_dump") else close_payload.dict()
        except Exception as exc:  # noqa: BLE001
            scorecard_close = {"close_error": repr(exc)}

    merge_scorecard_rows(rows, scorecard_close)
    ranked = sorted(rows, key=lambda row: (int(row.get("levels_completed") or 0), float(row.get("score") or 0.0)), reverse=True)
    result = {
        "schema": "codexlab.arc_live_suite_result.v1",
        "run_id": run_id,
        "created_at_utc": iso_utc_now(),
        "condition": (
            f"live ARC scorecard policy={args.policy} max_actions_per_game={int(args.max_actions)} "
            f"operation_mode={str(getattr(arc, 'operation_mode', operation_mode))}"
        ),
        "evidence_category": "live_scorecard",
        "boundary": "Live ARC scorecard. Not an offline proxy.",
        "policy": str(args.policy),
        "max_actions_per_game": int(args.max_actions),
        "game_count": len(rows),
        "requested_game_ids": games,
        "scorecard_id": str(scorecard_id),
        "competition_mode_requested": bool(args.competition),
        "arc_operation_mode": str(getattr(arc, "operation_mode", operation_mode)),
        "source_url": args.source_url,
        "tags": list(args.tags) if args.tags else None,
        "opaque": opaque,
        "aggregate_score": float(scorecard_close.get("score") or 0.0),
        "total_levels_completed": int(scorecard_close.get("total_levels_completed") or 0),
        "policy_error_count": sum(int(row.get("policy_error_count") or 0) for row in rows),
        "games": rows,
        "ranked_games": [
            {
                "game_id": row.get("game_id"),
                "levels_completed": int(row.get("levels_completed") or 0),
                "score": float(row.get("score") or 0.0),
                "actions": int(row.get("actions") or 0),
                "resets": int(row.get("resets") or 0),
                "policy_error_count": int(row.get("policy_error_count") or 0),
            }
            for row in ranked
        ],
        "scorecard_close": scorecard_close,
        "working_directory": str(ROOT),
        "command": command,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_trace(result, out_dir)
    (out_dir / "index.html").write_text(render_live_index(result, out_dir), encoding="utf-8")
    (out_dir / "report_manifest.json").write_text(json.dumps(build_manifest(result, out_dir), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {out_dir / 'result.json'} scorecard={scorecard_id} "
        f"score={result['aggregate_score']} levels={result['total_levels_completed']} games={len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
