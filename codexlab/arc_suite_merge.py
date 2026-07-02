#!/usr/bin/env python3
"""Merge local ARC suite result JSON files without contacting ARC services."""

from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "codexlab.arc_suite_merge.v1"
BOUNDARY = (
    "Offline merge of existing result.json artifacts only. This tool does not "
    "open scorecards, call ARC APIs, run games, verify live leaderboard state, "
    "or convert proxy/local rows into solve evidence."
)


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_result_path(path: Path) -> Path:
    if path.is_dir():
        path = path / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"missing result JSON: {path}")
    if not path.is_file():
        raise ValueError(f"not a file or run directory with result.json: {path}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected object JSON in {path}")
    return data


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(float(text))
        except ValueError:
            return default
    return default


def as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def max_int_field(row: dict[str, Any], names: tuple[str, ...]) -> int:
    values = [as_int(row[name]) for name in names if name in row and row.get(name) is not None]
    return max(values) if values else 0


def max_float_field(row: dict[str, Any], names: tuple[str, ...]) -> float:
    values = [as_float(row[name]) for name in names if name in row and row.get(name) is not None]
    return max(values) if values else 0.0


def json_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    games = result.get("games")
    if isinstance(games, list):
        return [row for row in games if isinstance(row, dict)]
    if result.get("game_id"):
        return [result]
    ranked = result.get("ranked_games")
    if isinstance(ranked, list):
        return [row for row in ranked if isinstance(row, dict)]
    return []


def normalize_row(
    *,
    row: dict[str, Any],
    result: dict[str, Any],
    result_path: Path,
    source_index: int,
    row_index: int,
) -> dict[str, Any] | None:
    game_id = str(row.get("game_id") or "").strip()
    if not game_id:
        return None
    score = max_float_field(row, ("score", "aggregate_score"))
    if score == 0.0 and "score" not in row and "aggregate_score" in result and len(json_rows(result)) == 1:
        score = as_float(result.get("aggregate_score"))
    levels = max_int_field(row, ("levels_completed", "scorecard_levels_completed", "total_levels_completed"))
    if levels == 0 and "levels_completed" not in row and "total_levels_completed" in result and len(json_rows(result)) == 1:
        levels = as_int(result.get("total_levels_completed"))

    return {
        "game_id": game_id,
        "levels_completed": int(levels),
        "score": float(score),
        "policy_error_count": as_int(row.get("policy_error_count")),
        "policy_error_count_present": "policy_error_count" in row,
        "status": row.get("status"),
        "policy": row.get("policy", result.get("policy")),
        "actions": max_int_field(row, ("actions", "scorecard_actions")),
        "resets": max_int_field(row, ("resets", "scorecard_resets")),
        "final_state": row.get("final_state", row.get("scorecard_state")),
        "source_run_id": result.get("run_id"),
        "source_schema": result.get("schema"),
        "source_arc_mode": row.get("arc_mode", result.get("arc_mode")),
        "source_evidence_category": result.get("evidence_category"),
        "source_path": str(result_path),
        "source_index": int(source_index),
        "source_row_index": int(row_index),
    }


def select_unique_rows(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        game_id = str(row["game_id"])
        grouped.setdefault(game_id, []).append(row)
        current = selected.get(game_id)
        if current is None or better_key(row) > better_key(current):
            selected[game_id] = row

    duplicate_games = []
    for game_id, rows in sorted(grouped.items()):
        if len(rows) <= 1:
            continue
        chosen = selected[game_id]
        duplicate_games.append(
            {
                "game_id": game_id,
                "candidate_count": len(rows),
                "selected_source_path": chosen["source_path"],
                "source_paths": [row["source_path"] for row in rows],
            }
        )

    ranked = sorted(
        selected.values(),
        key=lambda row: (-int(row["levels_completed"]), -float(row["score"]), str(row["game_id"])),
    )
    return ranked, duplicate_games


def better_key(row: dict[str, Any]) -> tuple[int, float, int, int]:
    return (
        int(row.get("levels_completed") or 0),
        float(row.get("score") or 0.0),
        -int(row.get("policy_error_count") or 0),
        -int(row.get("source_index") or 0),
    )


def build_merged(inputs: list[Path]) -> dict[str, Any]:
    source_results = []
    candidates = []
    errors = []
    for source_index, raw_path in enumerate(inputs):
        try:
            result_path = resolve_result_path(raw_path)
            result = load_json(result_path)
            rows = json_rows(result)
            source_results.append(
                {
                    "input_path": str(raw_path),
                    "result_path": str(result_path),
                    "schema": result.get("schema"),
                    "run_id": result.get("run_id"),
                    "arc_mode": result.get("arc_mode"),
                    "evidence_category": result.get("evidence_category"),
                    "row_count": len(rows),
                }
            )
            for row_index, row in enumerate(rows):
                normalized = normalize_row(
                    row=row,
                    result=result,
                    result_path=result_path,
                    source_index=source_index,
                    row_index=row_index,
                )
                if normalized is not None:
                    candidates.append(normalized)
        except Exception as exc:  # noqa: BLE001 - merge should report bad inputs, not hide them.
            errors.append({"input_path": str(raw_path), "error": repr(exc)})

    ranked_games, duplicate_games = select_unique_rows(candidates)
    policy_error_total = sum(int(row.get("policy_error_count") or 0) for row in ranked_games)
    candidate_policy_error_total = sum(int(row.get("policy_error_count") or 0) for row in candidates)
    missing_policy_error_count = sum(1 for row in candidates if not row.get("policy_error_count_present"))
    source_modes = sorted({str(row.get("source_arc_mode")) for row in candidates if row.get("source_arc_mode")})

    return {
        "schema": SCHEMA,
        "created_at_utc": iso_utc_now(),
        "boundary": BOUNDARY,
        "selection_policy": (
            "Unique game_id rows select the candidate with highest levels_completed, "
            "then highest score, then lowest policy_error_count, then earliest input."
        ),
        "input_count": len(inputs),
        "source_results": source_results,
        "source_modes": source_modes,
        "input_errors": errors,
        "candidate_row_count": len(candidates),
        "unique_game_count": len(ranked_games),
        "duplicate_game_count": len(duplicate_games),
        "duplicate_games": duplicate_games,
        "total_score": sum(float(row.get("score") or 0.0) for row in ranked_games),
        "aggregate_score": sum(float(row.get("score") or 0.0) for row in ranked_games),
        "total_levels_completed": sum(int(row.get("levels_completed") or 0) for row in ranked_games),
        "policy_error_count": policy_error_total,
        "policy_error_summary": {
            "selected_unique_total": policy_error_total,
            "input_candidate_total": candidate_policy_error_total,
            "missing_policy_error_count": missing_policy_error_count,
        },
        "ranked_games": ranked_games,
    }


def render_html(merged: dict[str, Any]) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value if value is not None else ""))

    cards = {
        "Unique games": merged.get("unique_game_count"),
        "Total levels": merged.get("total_levels_completed"),
        "Aggregate score": f"{float(merged.get('aggregate_score') or 0.0):.6f}",
        "Policy errors": (merged.get("policy_error_summary") or {}).get("selected_unique_total"),
        "Candidate rows": merged.get("candidate_row_count"),
        "Duplicate games": merged.get("duplicate_game_count"),
    }
    card_html = "".join(
        "<section class='card'>"
        f"<div>{esc(label)}</div>"
        f"<strong>{esc(value)}</strong>"
        "</section>"
        for label, value in cards.items()
    )
    rows = []
    for rank, row in enumerate(merged.get("ranked_games") or [], start=1):
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{esc(row.get('game_id'))}</td>"
            f"<td>{int(row.get('levels_completed') or 0)}</td>"
            f"<td>{float(row.get('score') or 0.0):.6f}</td>"
            f"<td>{int(row.get('policy_error_count') or 0)}</td>"
            f"<td>{esc(row.get('policy'))}</td>"
            f"<td>{esc(row.get('source_run_id'))}</td>"
            f"<td>{esc(row.get('source_path'))}</td>"
            "</tr>"
        )
    table_rows = "".join(rows) or "<tr><td colspan='8'>No game rows loaded.</td></tr>"
    errors = merged.get("input_errors") or []
    error_rows = "".join(
        "<tr>"
        f"<td>{esc(row.get('input_path'))}</td>"
        f"<td>{esc(row.get('error'))}</td>"
        "</tr>"
        for row in errors
    )
    errors_html = ""
    if error_rows:
        errors_html = (
            "<h2>Input Errors</h2>"
            "<table><thead><tr><th>Input</th><th>Error</th></tr></thead>"
            f"<tbody>{error_rows}</tbody></table>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ARC Suite Merge</title>
  <style>
    body {{ margin: 0; font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; color: #202124; background: #f8f9fa; }}
    header {{ background: #203040; color: #fff; padding: 22px 28px; }}
    main {{ max-width: 1160px; margin: 0 auto; padding: 22px; }}
    h1 {{ margin: 0 0 8px; font-size: 24px; letter-spacing: 0; }}
    h2 {{ margin: 24px 0 10px; font-size: 17px; letter-spacing: 0; }}
    a {{ color: #0b57d0; }}
    .boundary {{ max-width: 980px; color: #dce6f0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(145px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ background: #fff; border: 1px solid #d6d9de; border-radius: 6px; padding: 12px; }}
    .card div {{ color: #5f6368; font-size: 12px; }}
    .card strong {{ display: block; font-size: 19px; margin-top: 2px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d6d9de; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e5e9; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
    th {{ background: #edf0f3; color: #3c4043; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>ARC Suite Merge</h1>
    <div class="boundary">{esc(merged.get("boundary"))}</div>
    <p><a href="merged_result.json">merged_result.json</a></p>
  </header>
  <main>
    <section class="cards">{card_html}</section>
    <h2>Ranked Games</h2>
    <table>
      <thead><tr><th>Rank</th><th>Game</th><th>Levels</th><th>Score</th><th>Policy Errors</th><th>Policy</th><th>Run</th><th>Source</th></tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
    {errors_html}
  </main>
</body>
</html>
"""


def write_outputs(merged: dict[str, Any], out_dir: Path, *, force: bool, html_enabled: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "merged_result.json"
    html_path = out_dir / "index.html"
    targets = [json_path]
    if html_enabled:
        targets.append(html_path)
    existing = [path for path in targets if path.exists()]
    if existing and not force:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite existing output(s): {paths}")
    json_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if html_enabled:
        html_path.write_text(render_html(merged), encoding="utf-8")


def print_summary(merged: dict[str, Any]) -> None:
    errors = merged.get("input_errors") or []
    policy_summary = merged.get("policy_error_summary") or {}
    print(
        "arc-suite-merge "
        f"unique_games={merged.get('unique_game_count')} "
        f"candidate_rows={merged.get('candidate_row_count')} "
        f"levels={merged.get('total_levels_completed')} "
        f"score={float(merged.get('aggregate_score') or 0.0):.6f} "
        f"policy_errors={policy_summary.get('selected_unique_total')} "
        f"input_errors={len(errors)}"
    )
    for rank, row in enumerate((merged.get("ranked_games") or [])[:10], start=1):
        print(
            f"{rank:02d} {row.get('game_id')} "
            f"levels={int(row.get('levels_completed') or 0)} "
            f"score={float(row.get('score') or 0.0):.6f} "
            f"policy_errors={int(row.get('policy_error_count') or 0)} "
            f"source={row.get('source_path')}"
        )
    if errors:
        print("input errors:", file=sys.stderr)
        for row in errors:
            print(f"  {row.get('input_path')}: {row.get('error')}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="result.json files or run directories containing result.json")
    parser.add_argument("--out-dir", type=Path, default=None, help="Write merged_result.json and compact index.html here")
    parser.add_argument("--force", action="store_true", help="Overwrite files in --out-dir")
    parser.add_argument("--no-html", action="store_true", help="With --out-dir, write only merged_result.json")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    merged = build_merged(args.inputs)
    if args.out_dir is not None:
        try:
            write_outputs(merged, args.out_dir, force=bool(args.force), html_enabled=not bool(args.no_html))
        except Exception as exc:  # noqa: BLE001 - CLI should report output failures plainly.
            print(f"arc_suite_merge: {exc}", file=sys.stderr)
            return 2
    print_summary(merged)
    return 1 if merged.get("input_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
