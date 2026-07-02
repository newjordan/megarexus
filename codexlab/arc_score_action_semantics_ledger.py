#!/usr/bin/env python3
"""Offline score ActionSemanticsLedger packets against teacher-signal rows.

This is a static historical/proxy evaluation over public ARC recordings. It is
not local gameplay evidence and does not score solves.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

import contra_head  # noqa: E402


MANIFEST_SCHEMA = "codexlab.arc_action_semantics_ledger_offline_score_manifest.v1"
SCORE_ROW_SCHEMA = "codexlab.arc_action_semantics_ledger_offline_score_row.v1"


def utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def first_frame(recording_path: Path) -> list[list[int]]:
    with recording_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            frame = ((event.get("data") or {}).get("frame") or [])
            if frame and isinstance(frame, list):
                if len(frame) == 1 and isinstance(frame[0], list):
                    return frame[0]
                return frame
    raise ValueError(f"no frame found in {recording_path}")


def expected_family_hint(teacher_row: dict[str, Any]) -> str:
    input_obj = teacher_row.get("input") if isinstance(teacher_row.get("input"), dict) else {}
    summary = input_obj.get("observed_public_trace_summary") if isinstance(input_obj.get("observed_public_trace_summary"), dict) else {}
    return str(summary.get("action_family_hint") or "unknown")


def ledger_families(ledger: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for row in ledger.get("rows", []) or []:
        if isinstance(row, dict) and row.get("action_family_hypothesis"):
            values.add(str(row["action_family_hypothesis"]))
    return values


def ledger_probe_actions(ledger: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for row in ledger.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        for action in row.get("probe_action_names", []) or []:
            values.add(str(action))
    return values


def family_hint_match(family_hint: str, families: set[str]) -> bool:
    if family_hint == "click_dominant":
        return "click_or_select_probe" in families
    if family_hint == "directional_dominant":
        return "directional_motion_probe" in families or "unknown_action_family_probe" in families
    if family_hint == "mixed_action_family":
        return bool(families.intersection({"directional_motion_probe", "click_or_select_probe", "support_surface_probe", "unknown_action_family_probe"}))
    return bool(families)


def score_teacher_row(*, teacher_row: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    target = teacher_row.get("target") if isinstance(teacher_row.get("target"), dict) else {}
    evidence = teacher_row.get("evidence") if isinstance(teacher_row.get("evidence"), dict) else {}
    recording_path = resolve_repo_path(str(evidence["source_recording_jsonl"]))
    frame = first_frame(recording_path)
    game_id = str(teacher_row.get("game_id"))
    packet = contra_head.run_contra_view(
        raw_frame=frame,
        game_id=game_id,
        level_index=0,
        mode="teacher_student_static_offline",
    )
    ledger = packet.get("action_semantics_ledger") if isinstance(packet.get("action_semantics_ledger"), dict) else {}
    packet_path = out_dir / "packets" / f"{game_id}__action_semantics_ledger_packet.json"
    write_json(packet_path, packet)

    families = ledger_families(ledger)
    probes = ledger_probe_actions(ledger)
    progression_actions = {str(action) for action in target.get("progression_actions", []) or []}
    required_probe_overlap = sorted(probes.intersection(progression_actions))
    checks = {
        "recommended_packet_match": target.get("recommended_packet") == "action_semantics_ledger",
        "ledger_schema_match": ledger.get("schema") == contra_head.ACTION_SEMANTICS_LEDGER_SCHEMA,
        "ledger_boundary_preserved": ledger.get("admission_status") == "not_admitted"
        and ledger.get("action_choice_admitted") is False
        and ledger.get("control_truth_admitted") is False
        and ledger.get("solve_condition_admitted") is False,
        "required_verifier_match": target.get("required_next_local_verifier") == "controlled_before_action_after_packet"
        and ledger.get("next_valid_step") == "controlled ActionModBand before/action/after probe",
        "family_hint_match": family_hint_match(expected_family_hint(teacher_row), families),
        "progression_action_probe_overlap": bool(required_probe_overlap) if progression_actions else True,
    }
    score = sum(1 for value in checks.values() if value) / max(1, len(checks))
    return {
        "schema": SCORE_ROW_SCHEMA,
        "row_id": teacher_row.get("row_id"),
        "split": teacher_row.get("split"),
        "game_id": game_id,
        "recording_guid": teacher_row.get("recording_guid"),
        "category": "historical_static_proxy",
        "truth_status": "offline_hypothesis_score",
        "boundary": "public recording static frame and teacher signal comparison only; not local CONTRA gameplay evidence",
        "teacher_signal_family": teacher_row.get("signal_family"),
        "teacher_action_family_hint": expected_family_hint(teacher_row),
        "target_progression_actions": sorted(progression_actions),
        "student_ledger_schema": ledger.get("schema"),
        "student_family_hypotheses": sorted(families),
        "student_probe_actions": sorted(probes),
        "progression_action_probe_overlap": required_probe_overlap,
        "checks": checks,
        "score": round(score, 6),
        "student_packet": relpath(packet_path),
        "student_packet_sha256": sha256_file(packet_path),
        "source_recording_jsonl": relpath(recording_path),
        "source_recording_sha256": evidence.get("source_recording_sha256"),
    }


def build_index(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>ActionSemanticsLedger Offline Score</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;background:#111;color:#eee}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #444;padding:7px;vertical-align:top}th{background:#252525}.bad{color:#ffb3b3}.ok{color:#b7f2c2}</style>",
        "</head><body>",
        "<h1>ActionSemanticsLedger Offline Score</h1>",
        f"<p><b>Category:</b> {html.escape(manifest['category'])}</p>",
        f"<p><b>Boundary:</b> {html.escape(manifest['boundary'])}</p>",
        f"<p><b>Rows:</b> {manifest['row_count']} | <b>Mean score:</b> {manifest['mean_score']}</p>",
        "<table><tr><th>Game</th><th>Split</th><th>Score</th><th>Teacher hint</th><th>Student families</th><th>Probe overlap</th><th>Packet</th></tr>",
    ]
    for row in rows:
        klass = "ok" if float(row.get("score") or 0.0) >= 0.8 else "bad"
        href = html.escape(Path(row["student_packet"]).name)
        packet_href = html.escape("../" + row["student_packet"])
        lines.append(
            "<tr>"
            f"<td>{html.escape(str(row['game_id']))}</td>"
            f"<td>{html.escape(str(row['split']))}</td>"
            f"<td class='{klass}'>{row['score']}</td>"
            f"<td>{html.escape(str(row['teacher_action_family_hint']))}</td>"
            f"<td>{html.escape(', '.join(row['student_family_hypotheses']))}</td>"
            f"<td>{html.escape(', '.join(row['progression_action_probe_overlap']))}</td>"
            f"<td><a href='{packet_href}'>{href}</a></td>"
            "</tr>"
        )
    lines.extend(["</table>", "</body></html>"])
    return "\n".join(lines) + "\n"


def run_score(*, teacher_dataset_dir: Path, out_dir: Path, limit: int = 0) -> dict[str, Any]:
    rows = []
    for name in ("train.jsonl", "heldout.jsonl"):
        rows.extend(load_jsonl(teacher_dataset_dir / name))
    teacher_rows = [row for row in rows if row.get("signal_family") == "action_semantics_signal"]
    if limit:
        teacher_rows = teacher_rows[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)
    score_rows = [score_teacher_row(teacher_row=row, out_dir=out_dir) for row in teacher_rows]
    score_path = out_dir / "score_rows.jsonl"
    write_jsonl(score_path, score_rows)
    mean_score = round(sum(float(row["score"]) for row in score_rows) / max(1, len(score_rows)), 6)
    check_pass_counts = {}
    for row in score_rows:
        for key, value in row["checks"].items():
            check_pass_counts[key] = check_pass_counts.get(key, 0) + int(bool(value))
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "category": "historical_static_proxy",
        "truth_status": "offline_hypothesis_score",
        "boundary": "public recording static frame and teacher signal comparison only; not local CONTRA gameplay evidence",
        "condition": "score static ActionSemanticsLedger packet contract against action_semantics_signal teacher rows",
        "teacher_dataset_manifest": relpath(teacher_dataset_dir / "manifest.json"),
        "teacher_dataset_manifest_sha256": sha256_file(teacher_dataset_dir / "manifest.json"),
        "score_rows_jsonl": relpath(score_path),
        "score_rows_jsonl_sha256": sha256_file(score_path),
        "row_count": len(score_rows),
        "mean_score": mean_score,
        "check_pass_counts": check_pass_counts,
        "next_valid_step": "use failures to refine static ledger or build controlled ActionModBand probes; do not claim gameplay solve",
    }
    manifest_path = out_dir / "manifest.json"
    write_json(manifest_path, manifest)
    index_path = out_dir / "index.html"
    index_path.write_text(build_index(manifest, score_rows), encoding="utf-8")
    manifest["index_html"] = relpath(index_path)
    manifest["index_html_sha256"] = sha256_file(index_path)
    write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-dataset-dir", type=Path, default=Path("codexlab/datasets/arc_opus48_teacher_signals_v0"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(f"codexlab/runs/action_semantics_ledger_offline_score_{utc_now_compact()}")
    manifest = run_score(
        teacher_dataset_dir=resolve_repo_path(args.teacher_dataset_dir),
        out_dir=resolve_repo_path(out_dir),
        limit=int(args.limit),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
