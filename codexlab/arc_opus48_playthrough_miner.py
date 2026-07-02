#!/usr/bin/env python3
"""Mine ARC Prize model recordings for CONTRA/SP80 design hypotheses.

This script treats public model playthroughs as hypothesis material only. It
does not create local CONTRA evidence or solve claims.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


SCORECARD_URL = "https://arcprize.org/api/models/scorecard/{config_id}"
RECORDING_URL = "https://arcprize.org/api/recordings/{game_id}/{guid}"

PATTERNS = {
    "control_surface_probe": [
        "what action",
        "what each action",
        "available_actions",
        "control",
        "interface",
        "click",
        "move",
        "action1",
        "action2",
        "action3",
        "action4",
        "action5",
        "action6",
    ],
    "predict_observe_update": [
        "observe",
        "effect",
        "changed",
        "frame change",
        "test",
        "probe",
        "learn",
        "deviation",
    ],
    "goal_inference": [
        "goal",
        "complete",
        "target",
        "win",
        "need to",
        "objective",
    ],
    "resource_budget": [
        "actions remaining",
        "steps remaining",
        "fuel",
        "budget",
        "cost",
        "resource",
    ],
    "mental_replay_planning": [
        "plan",
        "sequence",
        "replay",
        "simulate",
        "before committing",
        "count",
    ],
    "contradiction_revision": [
        "wrong",
        "contradiction",
        "revise",
        "not what",
        "unexpected",
        "mistake",
    ],
    "repeat_risk": [
        "try again",
        "repeat",
        "same action",
        "reset",
        "stuck",
    ],
}


def utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def iter_recording(url: str):
    with urllib.request.urlopen(url, timeout=120) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            yield json.loads(line)


def parse_reasoning(action_input: dict[str, Any]) -> dict[str, Any]:
    raw = action_input.get("reasoning")
    if not raw:
        return {"text": "", "usage": {}, "cost": {}}
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"text": str(raw), "usage": {}, "cost": {}}
    return {
        "text": str(payload.get("output") or ""),
        "usage": payload.get("usage") or {},
        "cost": payload.get("cost") or {},
    }


def keyword_hits(text: str) -> dict[str, int]:
    lower = text.lower()
    hits = {}
    for name, needles in PATTERNS.items():
        count = 0
        for needle in needles:
            count += lower.count(needle)
        if count:
            hits[name] = count
    return hits


def summarize_recording(game_id: str, run: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    guid = run["guid"]
    url = RECORDING_URL.format(game_id=game_id.strip(), guid=guid)
    events_path = out_dir / f"{game_id.strip()}__{guid}__events.jsonl"
    reasoning_path = out_dir / f"{game_id.strip()}__{guid}__reasoning.jsonl"

    action_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    level_progressions = []
    reasoning_samples = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    event_count = 0
    previous_levels_completed = None
    max_levels_completed = 0

    with events_path.open("w", encoding="utf-8") as event_file, reasoning_path.open(
        "w", encoding="utf-8"
    ) as reasoning_file:
        for event in iter_recording(url):
            event_count += 1
            data = event.get("data") or {}
            action_input = data.get("action_input") or {}
            action_id = str(action_input.get("id") or "")
            action_counts[action_id] += 1
            levels_completed = int(data.get("levels_completed") or 0)
            max_levels_completed = max(max_levels_completed, levels_completed)
            if previous_levels_completed is None:
                previous_levels_completed = levels_completed
            elif levels_completed > previous_levels_completed:
                level_progressions.append(
                    {
                        "event_index": event_count,
                        "from": previous_levels_completed,
                        "to": levels_completed,
                        "action_id": action_id,
                        "timestamp": event.get("timestamp"),
                    }
                )
                previous_levels_completed = levels_completed

            reasoning = parse_reasoning(action_input)
            text = reasoning["text"]
            hits = keyword_hits(text)
            pattern_counts.update(hits)
            usage = reasoning["usage"]
            cost = reasoning["cost"]
            total_input_tokens += int(usage.get("input_tokens") or 0)
            total_output_tokens += int(usage.get("output_tokens") or 0)
            total_cost += float(cost.get("total_cost") or 0.0)

            stripped = {
                "timestamp": event.get("timestamp"),
                "game_id": data.get("game_id"),
                "guid": data.get("guid"),
                "state": data.get("state"),
                "levels_completed": data.get("levels_completed"),
                "available_actions": data.get("available_actions"),
                "action_input": {
                    "id": action_id,
                    "data": action_input.get("data") or {},
                    "reasoning_present": bool(text),
                },
                "frame_stripped": "public recording frame omitted by miner",
            }
            event_file.write(json.dumps(stripped, sort_keys=True) + "\n")

            if text:
                row = {
                    "event_index": event_count,
                    "timestamp": event.get("timestamp"),
                    "action_id": action_id,
                    "levels_completed": levels_completed,
                    "pattern_hits": hits,
                    "text_excerpt": compact_text(text, 700),
                    "usage": usage,
                    "cost": cost,
                }
                reasoning_file.write(json.dumps(row, sort_keys=True) + "\n")
                if len(reasoning_samples) < 5 or hits:
                    reasoning_samples.append(row)
                    reasoning_samples = reasoning_samples[:12]

    return {
        "game_id": game_id.strip(),
        "guid": guid,
        "recording_url": url,
        "event_count": event_count,
        "events_path": str(events_path),
        "reasoning_path": str(reasoning_path),
        "scorecard_score": run.get("score"),
        "scorecard_levels_completed": run.get("levels_completed"),
        "max_recording_levels_completed": max_levels_completed,
        "scorecard_actions": run.get("actions"),
        "action_counts": dict(action_counts),
        "pattern_counts": dict(pattern_counts),
        "level_progressions": level_progressions,
        "token_usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_cost": round(total_cost, 6),
        },
        "reasoning_samples": reasoning_samples,
    }


def compact_text(text: str, limit: int) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def candidate_system_additions(run_summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    aggregate = Counter()
    zero_score_repeat = []
    for summary in run_summaries:
        aggregate.update(summary["pattern_counts"])
        if (
            int(summary.get("scorecard_levels_completed") or 0) == 0
            and int(summary.get("scorecard_actions") or 0) >= 100
        ):
            zero_score_repeat.append(summary["game_id"])

    additions = [
        {
            "name": "Action Semantics Probe Ledger",
            "source_signal": "Repeated control_surface_probe and predict_observe_update language across public Claude recordings.",
            "implementation_target": "Record per-action before/after deltas for ACTION1-6 by game family before planning.",
            "boundary": "Hypothesis only until validated by a controlled local harness run.",
        },
        {
            "name": "No-Repeat Exploration Memory",
            "source_signal": f"Zero-completion runs with large action budgets: {', '.join(zero_score_repeat[:12])}",
            "implementation_target": "Add dead-probe keys over (game_id, level, action_id, selected object/click cell, delta signature).",
            "boundary": "Should reweight exploration only; not action authority or solve truth.",
        },
        {
            "name": "Goal-Spec Candidate Extractor",
            "source_signal": "Goal-inference terms appear while recordings still spend many actions without level completion.",
            "implementation_target": "Promote bordered/invariant/target-region candidates into a review-only goal-spec packet.",
            "boundary": "Goal candidates are hypotheses until engine/verifier confirms completion semantics.",
        },
        {
            "name": "Resource-Aware Probe Budget",
            "source_signal": "Recordings expose long late-level stalls and reset-heavy runs.",
            "implementation_target": "Track actions/resets/steps remaining in CONTRA exploration packets and decay stale branches.",
            "boundary": "Budgeting controls search allocation only, not truth of a candidate.",
        },
        {
            "name": "Family-First Planner Routing",
            "source_signal": "Scorecard winners are clustered in games where early controls and mechanic family are discoverable.",
            "implementation_target": "Use source decode families plus observed action response to choose click-only, D-pad, sprite-move, or placement planner.",
            "boundary": "Family routing is a prior; controlled deltas must override it.",
        },
    ]
    if aggregate.get("contradiction_revision", 0):
        additions.append(
            {
                "name": "Contradiction-Triggered Replan",
                "source_signal": "Reasoning text includes explicit wrong/unexpected/revise signals.",
                "implementation_target": "When expected delta and observed delta disagree twice, clear local hypothesis rank and force a new representation branch.",
                "boundary": "This is a stale-loop prevention rule, not evidence of correctness.",
            }
        )
    return additions


def write_markdown(
    out_dir: Path,
    config_id: str,
    scorecard: dict[str, Any],
    run_summaries: list[dict[str, Any]],
    additions: list[dict[str, str]],
) -> Path:
    path = out_dir / "index.md"
    ranked = sorted(
        run_summaries,
        key=lambda row: (int(row.get("scorecard_levels_completed") or 0), float(row.get("scorecard_score") or 0)),
        reverse=True,
    )
    lines = [
        "# ARC Opus 4.8 Playthrough Mining",
        "",
        "Category: historical public scorecard and recording review.",
        "Boundary: hypothesis material only; not local CONTRA control truth, goal truth, action choice, prediction, or solve evidence.",
        "",
        "## Source",
        "",
        f"- Config: `{config_id}`",
        f"- Scorecard API: `{SCORECARD_URL.format(config_id=config_id)}`",
        f"- Published: `{scorecard.get('published_at')}`",
        f"- Last update: `{scorecard.get('last_update')}`",
        f"- Model: `{scorecard.get('model')}`",
        f"- Runner: `{scorecard.get('runner')}`",
        "",
        "## Scorecard Summary",
        "",
        f"- Score: `{scorecard.get('score')}`",
        f"- Levels completed: `{scorecard.get('total_levels_completed')} / {scorecard.get('total_levels')}`",
        f"- Environments completed: `{scorecard.get('total_environments_completed')} / {scorecard.get('total_environments')}`",
        f"- Total actions: `{scorecard.get('total_actions')}`",
        "",
        "## Candidate System Additions",
        "",
    ]
    for addition in additions:
        lines.extend(
            [
                f"### {addition['name']}",
                "",
                f"- Source signal: {addition['source_signal']}",
                f"- Implementation target: {addition['implementation_target']}",
                f"- Boundary: {addition['boundary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Ranked Recording Targets",
            "",
            "| Game | Levels | Score | Actions | Events | Top pattern counts | Recording |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in ranked:
        patterns = ", ".join(f"{k}={v}" for k, v in sorted(row["pattern_counts"].items())[:4]) or "none"
        lines.append(
            "| {game} | {levels} | {score} | {actions} | {events} | {patterns} | `{url}` |".format(
                game=row["game_id"],
                levels=row.get("scorecard_levels_completed"),
                score=row.get("scorecard_score"),
                actions=row.get("scorecard_actions"),
                events=row.get("event_count"),
                patterns=patterns,
                url=row["recording_url"],
            )
        )
    lines.extend(["", "## Raw Local Files", ""])
    lines.append("- `scorecard.json`")
    lines.append("- `summary.json`")
    lines.append("- `*__events.jsonl` strips frames and keeps action/state provenance.")
    lines.append("- `*__reasoning.jsonl` keeps bounded reasoning excerpts, pattern hits, usage, and cost.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_html(md_path: Path) -> Path:
    html_path = md_path.with_suffix(".html")
    text = md_path.read_text(encoding="utf-8")
    body_lines = []
    in_table = False
    for line in text.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            if in_table:
                body_lines.append("</tbody></table>")
                in_table = False
            body_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            if in_table:
                body_lines.append("</tbody></table>")
                in_table = False
            body_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("| ") and " --- " not in line:
            cells = [html.escape(cell.strip()) for cell in line.strip("|").split("|")]
            tag = "th" if not in_table else "td"
            if not in_table:
                body_lines.append("<table><tbody>")
                in_table = True
            body_lines.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
        elif line.startswith("| "):
            continue
        elif line.startswith("- "):
            body_lines.append(f"<p class='bullet'>{html.escape(line)}</p>")
        elif line.strip():
            body_lines.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body_lines.append("</tbody></table>")
    html_doc = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ARC Opus 4.8 Playthrough Mining</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 32px; background: #101214; color: #e8e8e8; }
    h1, h2, h3 { color: #ffffff; }
    table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }
    th, td { border: 1px solid #3a3f44; padding: 8px; vertical-align: top; }
    th { background: #24282d; text-align: left; }
    tr:nth-child(even) { background: #171a1d; }
    .bullet { margin-left: 16px; }
  </style>
</head>
<body>
""" + "\n".join(body_lines) + "\n</body>\n</html>\n"
    html_path.write_text(html_doc, encoding="utf-8")
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", default="anthropic-opus-4-8-high")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=0, help="Limit recordings for smoke tests.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"codexlab/runs/arc_opus48_playthrough_mining_{utc_now_compact()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    scorecard = fetch_json(SCORECARD_URL.format(config_id=args.config_id))
    (out_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")

    run_summaries = []
    run_items = []
    for env in scorecard.get("environments") or []:
        for run in env.get("runs") or []:
            if run.get("guid"):
                run_items.append((env["id"], run))
    if args.limit:
        run_items = run_items[: args.limit]

    for index, (game_id, run) in enumerate(run_items, start=1):
        print(f"[{index}/{len(run_items)}] mining {game_id.strip()} {run['guid']}", file=sys.stderr)
        try:
            run_summaries.append(summarize_recording(game_id, run, out_dir))
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            run_summaries.append(
                {
                    "game_id": game_id.strip(),
                    "guid": run["guid"],
                    "recording_url": RECORDING_URL.format(game_id=game_id.strip(), guid=run["guid"]),
                    "fetch_error": str(exc),
                    "scorecard_score": run.get("score"),
                    "scorecard_levels_completed": run.get("levels_completed"),
                    "scorecard_actions": run.get("actions"),
                    "pattern_counts": {},
                }
            )

    additions = candidate_system_additions(run_summaries)
    summary = {
        "schema": "codexlab.arc_opus48_playthrough_mining.v1",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "category": "historical_public_recording_review",
        "boundary": "Hypothesis material only; not local CONTRA evidence.",
        "scorecard": {
            key: scorecard.get(key)
            for key in [
                "card_id",
                "score",
                "published_at",
                "last_update",
                "model",
                "runner",
                "config",
                "total_levels_completed",
                "total_levels",
                "total_environments_completed",
                "total_environments",
                "total_actions",
            ]
        },
        "run_count": len(run_summaries),
        "candidate_system_additions": additions,
        "runs": run_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    md_path = write_markdown(out_dir, args.config_id, scorecard, run_summaries, additions)
    html_path = write_html(md_path)
    print(str(html_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
