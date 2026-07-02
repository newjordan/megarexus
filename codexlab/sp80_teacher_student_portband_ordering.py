#!/usr/bin/env python3
"""Score teacher/student PortBand ordering from controlled SP80 episode packets.

Teacher here means the deterministic ordering contract encoded in the current
CONTRA recommendation signals: high-definedness PortBand tasks should be acted
on before lower-definedness or later exploratory tasks. Student means the
recorded primitive stream from the live episode harness.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "codexlab.sp80_teacher_student_portband_ordering.v1"
MOVE_PRIMITIVES = {"move_up", "move_down", "move_left", "move_right"}
KNOWN_THRESHOLD = 0.9


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    resolved.relative_to(ROOT)
    return resolved


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_dirty() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(proc.stdout.strip()) if proc.returncode == 0 else True


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def selected_recommendation(packet: dict[str, Any]) -> dict[str, Any]:
    selected = packet.get("selected_recommendation")
    return selected if isinstance(selected, dict) else {}


def row_from_packet(packet: dict[str, Any]) -> dict[str, Any]:
    selected = selected_recommendation(packet)
    refs = selected.get("evidence_refs") if isinstance(selected.get("evidence_refs"), dict) else {}
    signals = selected.get("signals") if isinstance(selected.get("signals"), dict) else {}
    primitive = str(packet.get("decision_primitive") or "")
    task_key = refs.get("portband_task_key")
    definedness = as_float(signals.get("lineup_definedness_score"))
    distance_delta = as_float((packet.get("goal_progress") or {}).get("distance_delta"))
    action_index = packet.get("action_index")
    if action_index is None:
        case_id = str(packet.get("case_id") or "")
        try:
            action_index = int(case_id.rsplit("_", 1)[-1])
        except ValueError:
            action_index = None
    return {
        "action_index": action_index,
        "primitive": primitive,
        "task_key": str(task_key) if task_key else None,
        "candidate_kind": selected.get("candidate_kind"),
        "definedness": definedness,
        "is_known_task": bool(task_key) and definedness >= KNOWN_THRESHOLD,
        "is_move": primitive in MOVE_PRIMITIVES,
        "positive_motion_signal": primitive in MOVE_PRIMITIVES and distance_delta > 0.0,
        "distance_delta": distance_delta,
        "value_score": as_float(selected.get("value_score")),
        "confidence": as_float(selected.get("confidence")),
        "source_selection_status": signals.get("source_selection_status"),
        "round_completed_portband_task_keys": signals.get("round_completed_portband_task_keys") or [],
        "source_selection_priority_sort": signals.get("source_selection_priority_sort"),
        "lineup_known_task_priority_score": as_float(signals.get("lineup_known_task_priority_score")),
    }


def first_index(rows: list[dict[str, Any]], predicate: Any) -> int | None:
    for idx, row in enumerate(rows):
        if predicate(row):
            action_index = row.get("action_index")
            return int(action_index) if isinstance(action_index, int) else idx
    return None


def distinct_in_order(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def analyze_run(run_dir: Path) -> dict[str, Any]:
    result = read_json(run_dir / "result.json")
    packets = read_jsonl(run_dir / "packets.jsonl")
    rows = [row_from_packet(packet) for packet in packets]

    known_task_order = distinct_in_order([row["task_key"] for row in rows if row["is_known_task"]])
    teacher_known_window = known_task_order[:2]
    expected_known_count = max(1, len(teacher_known_window))
    positive_known_tasks = distinct_in_order(
        [row["task_key"] for row in rows if row["is_known_task"] and row["positive_motion_signal"]]
    )
    first_positive_by_task = {
        task: first_index(rows, lambda row, task=task: row["task_key"] == task and row["positive_motion_signal"])
        for task in known_task_order
    }
    first_seen_by_task = {
        task: first_index(rows, lambda row, task=task: row["task_key"] == task)
        for task in known_task_order
    }
    first_later_task_index = first_index(
        rows,
        lambda row: bool(row["task_key"])
        and row["task_key"] not in teacher_known_window
        and len(teacher_known_window) >= 2,
    )
    positives_before_later = [
        first_positive_by_task.get(task)
        for task in teacher_known_window
        if first_positive_by_task.get(task) is not None
    ]
    first_two_positive_before_later = (
        len(teacher_known_window) < 2
        or first_later_task_index is None
        or (
            len(positives_before_later) == len(teacher_known_window)
            and all(index < first_later_task_index for index in positives_before_later)
        )
    )
    undefined_before_known_rows = [
        row
        for row in rows
        if row["task_key"]
        and not row["is_known_task"]
        and (
            not positives_before_later
            or row["action_index"] is None
            or row["action_index"] < max(positives_before_later)
        )
    ]
    move_rows = [row for row in rows if row["is_move"]]
    known_move_rows = [row for row in move_rows if row["is_known_task"]]
    positive_known_move_rows = [row for row in known_move_rows if row["positive_motion_signal"]]
    negative_known_move_rows = [
        row for row in known_move_rows if row["distance_delta"] < 0.0
    ]
    zero_known_move_rows = [
        row for row in known_move_rows if row["distance_delta"] == 0.0
    ]
    select_rows = [row for row in rows if row["primitive"] == "select_object"]
    blocked_selection_rows = [
        row
        for row in rows
        if str(row.get("source_selection_status") or "").startswith("source_selection_blocked")
    ]

    ordering_score = 1.0
    if undefined_before_known_rows:
        ordering_score -= 0.35
    if not first_two_positive_before_later:
        ordering_score -= 0.35
    if len(set(positive_known_tasks) & set(teacher_known_window)) < len(teacher_known_window):
        ordering_score -= 0.20
    ordering_score = max(0.0, round(ordering_score, 3))
    efficiency_score = round(len(positive_known_move_rows) / max(1, len(rows)), 3)
    motion_signal_score = round(
        len(set(positive_known_tasks) & set(teacher_known_window)) / expected_known_count,
        3,
    )

    verdict = "teacher_order_matched"
    if undefined_before_known_rows:
        verdict = "teacher_order_violation_undefined_before_known"
    elif not first_two_positive_before_later:
        verdict = "teacher_order_violation_later_task_too_early"
    elif motion_signal_score < 1.0:
        verdict = "teacher_order_partial_motion_signal"

    return {
        "schema": SCHEMA,
        "run_dir": rel(run_dir),
        "teacher_contract": {
            "teacher_role": "deterministic_packet_contract",
            "contract": "known high-definedness PortBand tasks before later or undefined exploration; count win signal only when a move primitive gives positive goal-distance delta",
            "known_threshold": KNOWN_THRESHOLD,
            "teacher_known_window": teacher_known_window,
        },
        "student": {
            "role": "recorded_contra_episode_actions",
            "backend": result.get("backend"),
            "mode": result.get("mode"),
            "level_index": result.get("level_index"),
            "status": result.get("status"),
            "action_count": int(result.get("action_count") or len(rows)),
            "levels_completed_delta": int(result.get("levels_completed_delta") or 0),
        },
        "metrics": {
            "verdict": verdict,
            "ordering_score": ordering_score,
            "motion_signal_score": motion_signal_score,
            "efficiency_score_positive_known_moves_per_action": efficiency_score,
            "known_task_order_seen": known_task_order,
            "positive_known_task_order": positive_known_tasks,
            "first_seen_by_task": first_seen_by_task,
            "first_positive_motion_by_task": first_positive_by_task,
            "first_later_task_index": first_later_task_index,
            "first_two_positive_before_later": first_two_positive_before_later,
            "undefined_before_known_count": len(undefined_before_known_rows),
            "move_count": len(move_rows),
            "known_move_count": len(known_move_rows),
            "positive_known_move_count": len(positive_known_move_rows),
            "negative_known_move_count": len(negative_known_move_rows),
            "zero_known_move_count": len(zero_known_move_rows),
            "select_count": len(select_rows),
            "blocked_selection_count": len(blocked_selection_rows),
        },
        "action_rows": rows,
        "boundaries": [
            "This is controlled-run packet analysis, not solve truth.",
            "The teacher is an explicit ordering contract derived from current recommendation signals, not a public ARC teacher trace.",
            "Positive PortBand motion signal requires a move primitive and positive goal-distance delta.",
            "Static PortBand rows remain evidence and ranking input; they are not action authority by themselves.",
        ],
    }


def html_table(rows: list[list[Any]], headers: list[str]) -> str:
    cells = "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
            + "</tr>"
        )
    return f"<table><thead><tr>{cells}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_html(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    run_tables: list[str] = []
    for analysis in payload["runs"]:
        metrics = analysis["metrics"]
        student = analysis["student"]
        contract = analysis["teacher_contract"]
        trace_rows = []
        for row in analysis["action_rows"]:
            trace_rows.append(
                [
                    row["action_index"],
                    row["primitive"],
                    row["task_key"],
                    row["is_known_task"],
                    row["positive_motion_signal"],
                    row["distance_delta"],
                    row["source_selection_status"],
                    row["round_completed_portband_task_keys"],
                ]
            )
        run_tables.append(
            f"""
            <section>
              <h2>{html.escape(analysis['run_dir'])}</h2>
              <p><b>student:</b> level {html.escape(str(student['level_index']))},
              status {html.escape(str(student['status']))}, actions {html.escape(str(student['action_count']))},
              completion delta {html.escape(str(student['levels_completed_delta']))}</p>
              <p><b>teacher known window:</b> <code>{html.escape(json.dumps(contract['teacher_known_window']))}</code></p>
              {html_table([
                  ['verdict', metrics['verdict']],
                  ['ordering_score', metrics['ordering_score']],
                  ['motion_signal_score', metrics['motion_signal_score']],
                  ['efficiency_score_positive_known_moves_per_action', metrics['efficiency_score_positive_known_moves_per_action']],
                  ['known_task_order_seen', metrics['known_task_order_seen']],
                  ['positive_known_task_order', metrics['positive_known_task_order']],
                  ['first_positive_motion_by_task', metrics['first_positive_motion_by_task']],
                  ['first_later_task_index', metrics['first_later_task_index']],
                  ['first_two_positive_before_later', metrics['first_two_positive_before_later']],
                  ['undefined_before_known_count', metrics['undefined_before_known_count']],
                  ['positive_known_move_count', metrics['positive_known_move_count']],
                  ['negative_known_move_count', metrics['negative_known_move_count']],
                  ['zero_known_move_count', metrics['zero_known_move_count']],
                  ['blocked_selection_count', metrics['blocked_selection_count']],
              ], ['metric', 'value'])}
              <h3>Action Trace</h3>
              {html_table(trace_rows, ['i', 'primitive', 'task', 'known', 'positive_motion', 'delta', 'selection_status', 'completed_keys'])}
            </section>
            """
        )
    boundaries = "".join(
        f"<li>{html.escape(str(boundary))}</li>" for boundary in payload["boundaries"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>CONTRA PortBand Teacher/Student Ordering</title>
  <style>
    body {{ background: #101418; color: #e8edf2; font-family: Arial, sans-serif; margin: 32px; }}
    h1, h2, h3 {{ color: #f7fafc; }}
    code {{ background: #1d2530; padding: 2px 5px; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #334155; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #1f2937; }}
    tr:nth-child(even) {{ background: #151c25; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 18px 0; }}
    .box {{ background: #18212b; border: 1px solid #334155; padding: 12px; border-radius: 6px; }}
    .label {{ color: #aab6c5; font-size: 12px; }}
    .value {{ font-size: 18px; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>CONTRA PortBand Teacher/Student Ordering</h1>
  <p>Generated {html.escape(payload['generated_at'])}; git {html.escape(str(payload.get('git_commit')))}, dirty={html.escape(str(payload.get('git_dirty')))}.</p>
  <div class="summary">
    <div class="box"><div class="label">runs</div><div class="value">{html.escape(str(summary['run_count']))}</div></div>
    <div class="box"><div class="label">matched</div><div class="value">{html.escape(str(summary['teacher_order_matched_count']))}</div></div>
    <div class="box"><div class="label">mean ordering</div><div class="value">{html.escape(str(summary['mean_ordering_score']))}</div></div>
    <div class="box"><div class="label">mean efficiency</div><div class="value">{html.escape(str(summary['mean_efficiency_score']))}</div></div>
  </div>
  <h2>Boundaries</h2>
  <ul>{boundaries}</ul>
  {''.join(run_tables)}
</body>
</html>
"""


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {
            "run_count": 0,
            "teacher_order_matched_count": 0,
            "mean_ordering_score": 0.0,
            "mean_motion_signal_score": 0.0,
            "mean_efficiency_score": 0.0,
        }
    return {
        "run_count": len(runs),
        "teacher_order_matched_count": sum(
            1 for run in runs if run["metrics"]["verdict"] == "teacher_order_matched"
        ),
        "mean_ordering_score": round(
            sum(run["metrics"]["ordering_score"] for run in runs) / len(runs),
            3,
        ),
        "mean_motion_signal_score": round(
            sum(run["metrics"]["motion_signal_score"] for run in runs) / len(runs),
            3,
        ),
        "mean_efficiency_score": round(
            sum(
                run["metrics"]["efficiency_score_positive_known_moves_per_action"]
                for run in runs
            )
            / len(runs),
            3,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dirs = [resolve_repo_path(path) for path in args.run_dir]
    out_dir = resolve_repo_path(args.out_dir) if args.out_dir else run_dirs[0] / "teacher_student_report"
    analyses = [analyze_run(run_dir) for run_dir in run_dirs]
    payload = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "runs": analyses,
        "summary": aggregate(analyses),
        "boundaries": [
            "This is controlled-run packet analysis, not solve truth.",
            "The teacher contract is known-first PortBand ordering from current CONTRA recommendation signals.",
            "The student path is the recorded SP80 episode action stream.",
            "No public ARC teacher trace, human trace, route replay, or coordinate script is used by this analyzer.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "teacher_student_portband_ordering.json"
    html_path = out_dir / "teacher_student_portband_ordering.html"
    write_json(json_path, payload)
    html_path.write_text(build_html(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "reported",
                "json": rel(json_path),
                "html": rel(html_path),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
