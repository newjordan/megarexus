#!/usr/bin/env python3
"""Export a full public ARC scorecard reasoning corpus.

This corpus is for algorithm extraction and process supervision only. It keeps
public action-level reasoning text while omitting frames from the derived rows.
It is not local CONTRA evidence and is not gameplay truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROW_SCHEMA = "codexlab.arc_scorecard_reasoning_row.v1"
MANIFEST_SCHEMA = "codexlab.arc_scorecard_reasoning_manifest.v1"

HELDOUT_GAME_IDS = {"sp80-589a99af", "g50t-5849a774", "tn36-ef4dde99", "wa30-ee6fef47"}

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
        "action7",
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
    "goal_inference": ["goal", "complete", "target", "win", "need to", "objective"],
    "resource_budget": ["actions remaining", "steps remaining", "fuel", "budget", "cost", "resource"],
    "mental_replay_planning": ["plan", "sequence", "replay", "simulate", "before committing", "count"],
    "contradiction_revision": ["wrong", "contradiction", "revise", "not what", "unexpected", "mistake"],
    "repeat_risk": ["try again", "repeat", "same action", "reset", "stuck"],
}


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def parse_reasoning(action_input: dict[str, Any]) -> dict[str, Any]:
    raw = action_input.get("reasoning")
    if not raw:
        return {"text": "", "usage": {}, "cost": {}, "parse_status": "missing"}
    if isinstance(raw, dict):
        payload = raw
    else:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {"text": str(raw), "usage": {}, "cost": {}, "parse_status": "raw_text"}
    return {
        "text": str(payload.get("output") or ""),
        "usage": payload.get("usage") or {},
        "cost": payload.get("cost") or {},
        "parse_status": "parsed_json",
    }


def keyword_hits(text: str) -> dict[str, int]:
    lower = text.lower()
    hits: dict[str, int] = {}
    for family, needles in PATTERNS.items():
        count = sum(lower.count(needle) for needle in needles)
        if count:
            hits[family] = count
    return hits


def infer_algorithm_hooks(pattern_hits: dict[str, int]) -> list[str]:
    hooks = []
    if pattern_hits.get("control_surface_probe") or pattern_hits.get("predict_observe_update"):
        hooks.append("action_semantics_ledger")
    if pattern_hits.get("goal_inference"):
        hooks.append("goal_spec_candidate_packet")
    if pattern_hits.get("resource_budget"):
        hooks.append("budget_governor")
    if pattern_hits.get("contradiction_revision"):
        hooks.append("contradiction_replan_gate")
    if pattern_hits.get("repeat_risk"):
        hooks.append("dead_probe_memory")
    if pattern_hits.get("mental_replay_planning"):
        hooks.append("family_route_prior")
    return hooks


def load_recording_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            event["_line_number"] = line_number
            rows.append(event)
    return rows


def make_row(
    *,
    event: dict[str, Any],
    event_index: int,
    recording: dict[str, Any],
    download_manifest_path: Path,
    scorecard_path: Path,
) -> dict[str, Any] | None:
    data = event.get("data") or {}
    action_input = data.get("action_input") or {}
    reasoning = parse_reasoning(action_input)
    text = reasoning["text"]
    if not text:
        return None

    game_id = str(data.get("game_id") or recording["game_id"]).strip()
    guid = str(data.get("guid") or recording["guid"]).strip()
    action_id = str(action_input.get("id") or "")
    pattern_hits = keyword_hits(text)
    split = "heldout" if game_id in HELDOUT_GAME_IDS else "train"
    row_id = f"{safe_id(game_id)}__{safe_id(guid)}__event_{event_index:05d}"
    return {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "split": split,
        "truth_status": "hypothesis_teacher_reasoning",
        "boundary": "public ARC scorecard reasoning only; not local CONTRA evidence",
        "game_id": game_id,
        "recording_guid": guid,
        "event_index": event_index,
        "source_line_number": event.get("_line_number"),
        "timestamp": event.get("timestamp"),
        "state": data.get("state"),
        "levels_completed": data.get("levels_completed"),
        "win_levels": data.get("win_levels"),
        "available_actions": data.get("available_actions") or [],
        "full_reset": data.get("full_reset"),
        "action": {
            "id": action_id,
            "data": action_input.get("data") or {},
        },
        "scorecard_run": {
            "score": recording.get("score"),
            "levels_completed": recording.get("levels_completed"),
            "actions": recording.get("actions"),
            "resets": recording.get("resets"),
            "state": recording.get("state"),
            "level_scores": recording.get("level_scores"),
            "level_actions": recording.get("level_actions"),
            "level_baseline_actions": recording.get("level_baseline_actions"),
        },
        "reasoning": {
            "text": text,
            "text_sha256": sha256_text(text),
            "char_count": len(text),
            "parse_status": reasoning["parse_status"],
            "usage": reasoning["usage"],
            "cost": reasoning["cost"],
        },
        "algorithm_extraction": {
            "pattern_hits": pattern_hits,
            "candidate_contra_hooks": infer_algorithm_hooks(pattern_hits),
            "target_use": "distill public reasoning process into CONTRA packet algorithms",
            "forbidden_use": "do not treat as action truth, solve truth, or local CONTRA evidence",
        },
        "source": {
            "download_manifest": relpath(download_manifest_path),
            "scorecard_json": relpath(scorecard_path),
            "recording_jsonl": recording.get("path"),
            "recording_sha256": recording.get("sha256"),
            "recording_url": recording.get("recording_url"),
        },
    }


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    rid = row.get("row_id", "<missing-row-id>")
    if row.get("schema") != ROW_SCHEMA:
        errors.append(f"{rid}: bad schema")
    if row.get("truth_status") != "hypothesis_teacher_reasoning":
        errors.append(f"{rid}: bad truth_status")
    if not row.get("reasoning", {}).get("text"):
        errors.append(f"{rid}: missing reasoning text")
    if row.get("source", {}).get("recording_jsonl") is None:
        errors.append(f"{rid}: missing recording path")
    if "local CONTRA evidence" not in row.get("boundary", ""):
        errors.append(f"{rid}: missing boundary")
    return errors


def export_corpus(*, download_manifest_path: Path, out_dir: Path) -> dict[str, Any]:
    manifest_in = load_json(download_manifest_path)
    scorecard_path = resolve_repo_path(manifest_in["scorecard_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    events_by_game: dict[str, int] = {}
    reasoning_rows_by_game: dict[str, int] = {}
    pattern_totals: Counter[str] = Counter()
    total_reasoning_chars = 0

    for recording in manifest_in.get("recordings") or []:
        recording_path = resolve_repo_path(recording["path"])
        game_id = str(recording["game_id"]).strip()
        recording_events = load_recording_rows(recording_path)
        events_by_game[game_id] = len(recording_events)
        reasoning_count = 0
        for event_index, event in enumerate(recording_events, start=1):
            row = make_row(
                event=event,
                event_index=event_index,
                recording=recording,
                download_manifest_path=download_manifest_path,
                scorecard_path=scorecard_path,
            )
            if row is None:
                continue
            rows.append(row)
            reasoning_count += 1
            total_reasoning_chars += int(row["reasoning"]["char_count"])
            pattern_totals.update(row["algorithm_extraction"]["pattern_hits"])
        reasoning_rows_by_game[game_id] = reasoning_count

    errors = []
    for row in rows:
        errors.extend(validate_row(row))
    if errors:
        raise SystemExit("row validation failed:\n" + "\n".join(errors[:50]))

    train_path = out_dir / "train.jsonl"
    heldout_path = out_dir / "heldout.jsonl"
    all_path = out_dir / "all_reasoning.jsonl"
    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)
    write_jsonl(all_path, rows)

    scorecard = load_json(scorecard_path)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "dataset": out_dir.name,
        "status": "exported",
        "truth_status": "hypothesis_teacher_reasoning",
        "boundary": "public ARC scorecard reasoning only; not local CONTRA evidence",
        "row_schema": ROW_SCHEMA,
        "source_scorecard_id": manifest_in.get("config_id"),
        "source_scorecard_summary": {
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
        "source_download_manifest": relpath(download_manifest_path),
        "source_download_manifest_sha256": sha256_file(download_manifest_path),
        "source_scorecard_json": relpath(scorecard_path),
        "source_scorecard_json_sha256": sha256_file(scorecard_path),
        "exporter_source": "codexlab/arc_export_scorecard_reasoning_corpus.py",
        "exporter_source_sha256": sha256_file(ROOT / "codexlab" / "arc_export_scorecard_reasoning_corpus.py"),
        "all_reasoning_jsonl": relpath(all_path),
        "all_reasoning_jsonl_sha256": sha256_file(all_path),
        "train_jsonl": relpath(train_path),
        "train_jsonl_sha256": sha256_file(train_path),
        "heldout_jsonl": relpath(heldout_path),
        "heldout_jsonl_sha256": sha256_file(heldout_path),
        "row_count": len(rows),
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "recording_count": len(manifest_in.get("recordings") or []),
        "game_count": len(events_by_game),
        "events_by_game": events_by_game,
        "reasoning_rows_by_game": reasoning_rows_by_game,
        "total_reasoning_chars": total_reasoning_chars,
        "pattern_totals": dict(pattern_totals),
        "heldout_game_ids": sorted(HELDOUT_GAME_IDS),
        "next_valid_step": "extract candidate algorithms, then implement CONTRA packet producers and offline signal evaluators",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=Path("codexlab/runs/arc_public_recordings_anthropic-opus-4-8-high_20260601T222454Z/download_manifest.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("codexlab/datasets/arc_opus48_scorecard_reasoning_corpus_v0"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = export_corpus(
        download_manifest_path=resolve_repo_path(str(args.download_manifest)),
        out_dir=resolve_repo_path(str(args.out_dir)),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
