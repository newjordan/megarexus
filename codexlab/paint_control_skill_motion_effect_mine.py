#!/usr/bin/env python3
"""Mine issued-action versus observed-effect rows from paint-control SP80 runs."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from paint_control_skill_feedback_export import classify
from sp80_base_controller import ROOT, sha256_file


SCHEMA = "codexlab.paint_control_skill_motion_effect_mine.v1"
ROW_SCHEMA = "codexlab.paint_control_skill_motion_effect_row.v1"
EXPECTED_DISPLAY_DELTAS = {
    "move_up": [0, -1],
    "move_down": [0, 1],
    "move_left": [-1, 0],
    "move_right": [1, 0],
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def level_dirs(run_dir: Path) -> list[Path]:
    if (run_dir / "packets.jsonl").is_file():
        return [run_dir]
    return sorted(path for path in run_dir.glob("level_*") if path.is_dir() and (path / "packets.jsonl").is_file())


def image_path(frame: dict[str, Any] | None) -> str | None:
    if not isinstance(frame, dict):
        return None
    png = frame.get("png")
    if isinstance(png, dict) and isinstance(png.get("path"), str):
        return png["path"]
    return None


def html_asset_src(out_dir: Path, repo_relative_path: Any) -> str:
    if not isinstance(repo_relative_path, str) or not repo_relative_path:
        return ""
    source = ROOT / repo_relative_path
    try:
        return os.path.relpath(source.resolve(), out_dir.resolve())
    except OSError:
        return repo_relative_path


def distance_delta(packet: dict[str, Any]) -> float | None:
    progress = packet.get("goal_progress") if isinstance(packet.get("goal_progress"), dict) else {}
    value = progress.get("distance_delta")
    return float(value) if isinstance(value, (int, float)) else None


def top_recommendation(packet: dict[str, Any]) -> dict[str, Any]:
    recommendation_packet = packet.get("contra_recommendation_packet")
    if not isinstance(recommendation_packet, dict):
        return {}
    rows = recommendation_packet.get("ranked_recommendations")
    if isinstance(rows, list) and rows and isinstance(rows[0], dict):
        return rows[0]
    return {}


def refs_for(row: dict[str, Any]) -> dict[str, Any]:
    refs = row.get("evidence_refs")
    return refs if isinstance(refs, dict) else {}


def native_action(raw: dict[str, Any]) -> dict[str, Any]:
    native = raw.get("native_action")
    return native if isinstance(native, dict) else {}


def build_row(level_dir: Path, row_index: int, packet: dict[str, Any], brain_call: dict[str, Any] | None) -> dict[str, Any]:
    decision = (brain_call or {}).get("contra_decision") or packet.get("contra_decision") or packet.get("brain_decision") or {}
    raw = decision.get("raw") if isinstance(decision.get("raw"), dict) else {}
    native = native_action(raw)
    before_state = packet.get("before_state") if isinstance(packet.get("before_state"), dict) else {}
    after_state = packet.get("after_state") if isinstance(packet.get("after_state"), dict) else {}
    top_row = top_recommendation(packet)
    top_refs = refs_for(top_row)
    primitive = packet.get("decision_primitive")
    display_delta = packet.get("selected_display_coord_delta")
    expected_delta = EXPECTED_DISPLAY_DELTAS.get(str(primitive))
    return {
        "schema": ROW_SCHEMA,
        "level_dir": rel(level_dir),
        "row_index": int(row_index),
        "case_id": packet.get("case_id"),
        "action_index": packet.get("action_index"),
        "status": packet.get("status"),
        "error": packet.get("error"),
        "label": classify(packet, raw),
        "primitive": primitive,
        "native_action_id": raw.get("native_action_id"),
        "native_primitive": native.get("primitive"),
        "native_action_kind": native.get("action_kind"),
        "native_object_slot": native.get("object_slot"),
        "selected_object_slot": raw.get("selected_object_slot"),
        "selected_name_before": before_state.get("selected_name"),
        "selected_name_after": after_state.get("selected_name"),
        "selected_display_coord_before": before_state.get("selected_display_coord"),
        "selected_display_coord_after": after_state.get("selected_display_coord"),
        "selected_display_coord_delta": display_delta,
        "selected_coord_delta": packet.get("selected_coord_delta"),
        "expected_display_coord_delta": expected_delta,
        "display_delta_matches_primitive": None if expected_delta is None else display_delta == expected_delta,
        "goal_distance_delta": distance_delta(packet),
        "goal_progress": packet.get("goal_progress"),
        "active_portband_source_key": raw.get("active_portband_source_key"),
        "top_recommendation_primitive": top_row.get("primitive"),
        "top_recommendation_kind": top_row.get("candidate_kind"),
        "top_recommendation_value_score": top_row.get("value_score"),
        "top_source_component_ids": top_refs.get("source_component_ids", []),
        "top_target_component_ids": top_refs.get("target_component_ids", []),
        "before_png": image_path(packet.get("before_frame")),
        "after_png": image_path(packet.get("after_frame")),
    }


def delta_key(value: Any) -> str:
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_primitive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_object_primitive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_position_primitive: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        primitive = str(row.get("primitive"))
        by_primitive[primitive].append(row)
        by_object_primitive[f"{row.get('selected_name_before')}|{primitive}"].append(row)
        by_position_primitive[
            f"{row.get('selected_name_before')}|{primitive}|{delta_key(row.get('selected_display_coord_before'))}"
        ].append(row)

    return {
        "by_primitive": [summarize_group(key, value) for key, value in sorted(by_primitive.items())],
        "by_object_primitive": [
            summarize_group(key, value)
            for key, value in sorted(by_object_primitive.items())
            if len(value) >= 2
        ],
        "by_position_primitive": [
            summarize_group(key, value)
            for key, value in sorted(by_position_primitive.items())
            if len(value) >= 2
        ],
        "conflicts": conflict_rows(rows),
    }


def summarize_group(key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(str(row.get("label")) for row in rows)
    observed_deltas = Counter(delta_key(row.get("selected_display_coord_delta")) for row in rows)
    distance_values = [float(row["goal_distance_delta"]) for row in rows if isinstance(row.get("goal_distance_delta"), (int, float))]
    return {
        "key": key,
        "count": len(rows),
        "label_counts": dict(sorted(label_counts.items())),
        "observed_display_delta_counts": dict(sorted(observed_deltas.items())),
        "stationary_count": sum(1 for row in rows if row.get("selected_display_coord_delta") == [0, 0]),
        "display_delta_mismatch_count": sum(1 for row in rows if row.get("display_delta_matches_primitive") is False),
        "positive_goal_distance_count": sum(1 for row in rows if isinstance(row.get("goal_distance_delta"), (int, float)) and row["goal_distance_delta"] > 0),
        "negative_goal_distance_count": sum(1 for row in rows if isinstance(row.get("goal_distance_delta"), (int, float)) and row["goal_distance_delta"] < 0),
        "mean_goal_distance_delta": None if not distance_values else round(mean(distance_values), 6),
        "examples": [example_ref(row) for row in rows[:6]],
    }


def example_ref(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "level_dir": row.get("level_dir"),
        "action_index": row.get("action_index"),
        "primitive": row.get("primitive"),
        "selected_name_before": row.get("selected_name_before"),
        "selected_display_coord_before": row.get("selected_display_coord_before"),
        "selected_display_coord_delta": row.get("selected_display_coord_delta"),
        "goal_distance_delta": row.get("goal_distance_delta"),
        "label": row.get("label"),
    }


def conflict_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []
    for row in rows:
        primitive = str(row.get("primitive"))
        if primitive.startswith("move_") and row.get("display_delta_matches_primitive") is False:
            conflicts.append({**example_ref(row), "reason": "observed_display_delta_does_not_match_primitive"})
        elif primitive.startswith("move_") and row.get("selected_display_coord_delta") == [0, 0]:
            conflicts.append({**example_ref(row), "reason": "stationary_move"})
        elif primitive == "select_object" and isinstance(row.get("goal_distance_delta"), (int, float)) and row["goal_distance_delta"] < 0:
            conflicts.append({**example_ref(row), "reason": "selection_switched_away_from_goal"})
    return conflicts


def write_index(out_dir: Path, report: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    primitive_rows = "\n".join(group_table_row(group) for group in report["summary"]["by_primitive"])
    conflict_table_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.get('action_index')))}</td><td>{html.escape(str(row.get('primitive')))}</td>"
        f"<td>{html.escape(str(row.get('selected_name_before')))}</td><td>{html.escape(json.dumps(row.get('selected_display_coord_before')))}</td>"
        f"<td>{html.escape(json.dumps(row.get('selected_display_coord_delta')))}</td><td>{html.escape(str(row.get('goal_distance_delta')))}</td>"
        f"<td>{html.escape(str(row.get('reason')))}</td></tr>"
        for row in report["summary"]["conflicts"][:80]
    )
    visual_cards = "\n".join(visual_card(out_dir, row) for row in [row for row in rows if row.get("before_png")][:24])
    source_rows = "\n".join(
        f"<tr><td>{html.escape(str(path))}</td><td>{html.escape(str(value))}</td></tr>"
        for path, value in sorted(report["source_hashes"].items())
    )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Paint Control Motion Effect Mine</title>
  <style>
    body {{ margin: 24px; font-family: system-ui, sans-serif; background: #f7f8fa; color: #16202a; }}
    main {{ max-width: 1260px; margin: 0 auto; }}
    h1 {{ font-size: 26px; margin-bottom: 4px; }}
    h2 {{ font-size: 18px; margin-top: 26px; }}
    .boundary {{ max-width: 960px; color: #53606d; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 10px 0 18px; }}
    th, td {{ border: 1px solid #d8dee8; padding: 6px 8px; text-align: left; vertical-align: top; font-size: 13px; }}
    th {{ background: #edf1f5; }}
    code {{ background: #e9edf2; padding: 2px 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 14px; }}
    article {{ background: white; border: 1px solid #d8dee8; border-radius: 6px; padding: 12px; }}
    .frames {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    figure {{ margin: 0; }}
    img {{ width: 100%; image-rendering: pixelated; border: 1px solid #cbd4df; background: white; }}
    figcaption {{ font-size: 12px; color: #53606d; }}
  </style>
</head>
<body>
<main>
  <h1>Paint Control Motion Effect Mine</h1>
  <p class="boundary">Controlled SP80 harness artifact. This mines issued primitives against observed selected-object deltas and target-distance deltas; it is diagnostic evidence, not solve evidence.</p>
  <table>
    <tr><th>Run dirs</th><td><code>{html.escape(', '.join(report["run_dirs"]))}</code></td></tr>
    <tr><th>Rows</th><td>{report["row_count"]}</td></tr>
    <tr><th>Motion JSON</th><td><a href="motion_effects.json">motion_effects.json</a></td></tr>
    <tr><th>Rows JSONL</th><td><a href="motion_effect_rows.jsonl">motion_effect_rows.jsonl</a></td></tr>
  </table>
  <h2>Primitive Effects</h2>
  <table><tr><th>Group</th><th>Count</th><th>Labels</th><th>Observed deltas</th><th>Stationary</th><th>Mismatch</th><th>Mean goal delta</th></tr>{primitive_rows}</table>
  <h2>Conflicts</h2>
  <table><tr><th>Action</th><th>Primitive</th><th>Object</th><th>Before coord</th><th>Observed delta</th><th>Goal delta</th><th>Reason</th></tr>{conflict_table_rows}</table>
  <h2>Visual Rows</h2>
  <section class="grid">{visual_cards}</section>
  <h2>Source Hashes</h2>
  <table><tr><th>Path</th><th>SHA-256</th></tr>{source_rows}</table>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(text, encoding="utf-8")


def group_table_row(group: dict[str, Any]) -> str:
    return (
        f"<tr><td>{html.escape(str(group.get('key')))}</td><td>{group.get('count')}</td>"
        f"<td>{html.escape(json.dumps(group.get('label_counts'), sort_keys=True))}</td>"
        f"<td>{html.escape(json.dumps(group.get('observed_display_delta_counts'), sort_keys=True))}</td>"
        f"<td>{group.get('stationary_count')}</td><td>{group.get('display_delta_mismatch_count')}</td>"
        f"<td>{html.escape(str(group.get('mean_goal_distance_delta')))}</td></tr>"
    )


def visual_card(out_dir: Path, row: dict[str, Any]) -> str:
    return f"""
    <article>
      <h3>Action {html.escape(str(row.get('action_index')))}: {html.escape(str(row.get('primitive')))} / {html.escape(str(row.get('label')))}</h3>
      <div class="frames">
        <figure><img src="{html.escape(html_asset_src(out_dir, row.get('before_png')))}" alt="before"><figcaption>Before</figcaption></figure>
        <figure><img src="{html.escape(html_asset_src(out_dir, row.get('after_png')))}" alt="after"><figcaption>After</figcaption></figure>
      </div>
      <table>
        <tr><th>Object</th><td>{html.escape(str(row.get('selected_name_before')))}</td></tr>
        <tr><th>Before coord</th><td>{html.escape(json.dumps(row.get('selected_display_coord_before')))}</td></tr>
        <tr><th>Observed delta</th><td>{html.escape(json.dumps(row.get('selected_display_coord_delta')))}</td></tr>
        <tr><th>Expected delta</th><td>{html.escape(json.dumps(row.get('expected_display_coord_delta')))}</td></tr>
        <tr><th>Goal delta</th><td>{html.escape(str(row.get('goal_distance_delta')))}</td></tr>
        <tr><th>Top PortBand target</th><td>{html.escape(json.dumps(row.get('top_target_component_ids')))}</td></tr>
      </table>
    </article>
    """


def mine_motion_effects(run_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    resolved_run_dirs = [(path if path.is_absolute() else ROOT / path).resolve() for path in run_dirs]
    for run_dir in resolved_run_dirs:
        for level_dir in level_dirs(run_dir):
            packets = read_jsonl(level_dir / "packets.jsonl")
            brain_calls = read_jsonl(level_dir / "brain_calls.jsonl")
            for packet_index, packet in enumerate(packets):
                brain_call = brain_calls[packet_index] if packet_index < len(brain_calls) else None
                rows.append(build_row(level_dir, len(rows), packet, brain_call))

    rows_path = out_dir / "motion_effect_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dirs": [rel(path) for path in resolved_run_dirs],
        "out_dir": rel(out_dir),
        "row_count": len(rows),
        "motion_effect_rows_jsonl": rel(rows_path),
        "index_html": rel(out_dir / "index.html"),
        "summary": summarize_rows(rows),
        "source_hashes": {
            "codexlab/paint_control_skill_motion_effect_mine.py": sha256_file(ROOT / "codexlab" / "paint_control_skill_motion_effect_mine.py"),
            "codexlab/paint_control_skill_feedback_export.py": sha256_file(ROOT / "codexlab" / "paint_control_skill_feedback_export.py"),
            "codexlab/sp80_player_harness.py": sha256_file(ROOT / "codexlab" / "sp80_player_harness.py"),
        },
        "truth_boundary": "Derived from controlled SP80 harness packets; diagnostic for action-effect modeling, not solve evidence.",
    }
    report_path = out_dir / "motion_effects.json"
    report["motion_effects_json"] = rel(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_index(out_dir, report, rows)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = mine_motion_effects(args.run_dir, args.out_dir)
    print(json.dumps({k: report[k] for k in ["schema", "run_dirs", "out_dir", "row_count", "motion_effects_json", "index_html"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
