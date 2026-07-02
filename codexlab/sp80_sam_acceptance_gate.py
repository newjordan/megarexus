#!/usr/bin/env python3
"""Fail-closed acceptance gate for SP80 SAM mask hypotheses.

SAM output starts as a hypothesis. This gate records which hypotheses have a
review decision and which ones remain blocked. It does not feed SAM into the
Contra controller or the SP80 goal condition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_SCHEMA = "codexlab.sp80_sam_acceptance.v1"
ACCEPTANCE_MANIFEST_SCHEMA = "codexlab.sp80_sam_acceptance_manifest.v1"
ACCEPTANCE_SCORECARD_SCHEMA = "codexlab.sp80_sam_acceptance_scorecard.v1"
REVIEW_DECISION_SCHEMA = "codexlab.sp80_sam_review_decision.v1"
HYPOTHESIS_SCHEMA = "codexlab.sp80_sam_hypothesis.v1"
CONTROLLED_LABEL_SCHEMA = "codexlab.sp80_object_label.v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no}: row is not an object")
        row["_source_line_no"] = line_no
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def confirmed_iou_threshold(score: dict[str, Any]) -> float:
    for component in score.get("components") or []:
        if component.get("name") == "controlled_mask_overlap":
            return float(component.get("confirmed_iou_threshold") or 0.0)
    return 0.0


def load_review_decisions(path: Path | None) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if path is None:
        return {}, []
    decisions: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for row in load_jsonl(path):
        hypothesis_id = row.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            errors.append(f"{path}:{row.get('_source_line_no')}: missing hypothesis_id")
            continue
        if hypothesis_id in decisions:
            errors.append(f"{path}:{row.get('_source_line_no')}: duplicate hypothesis_id {hypothesis_id}")
            continue
        decision = row.get("review_decision")
        if decision not in {"accept", "reject", "block"}:
            errors.append(f"{path}:{row.get('_source_line_no')}: invalid review_decision for {hypothesis_id}")
            continue
        if row.get("schema") not in {None, REVIEW_DECISION_SCHEMA}:
            errors.append(f"{path}:{row.get('_source_line_no')}: bad schema for {hypothesis_id}")
            continue
        decisions[hypothesis_id] = public_row(row)
    return decisions, errors


def review_provenance_errors(decision: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not decision.get("reviewer"):
        errors.append("accept_decision_missing_reviewer")
    if not decision.get("reviewed_at_utc"):
        errors.append("accept_decision_missing_reviewed_at_utc")
    if not decision.get("review_note") and not decision.get("review_artifact"):
        errors.append("accept_decision_missing_note_or_artifact")
    return errors


def acceptance_for(
    *,
    row: dict[str, Any],
    row_path: Path,
    decision: dict[str, Any] | None,
    controlled_by_key: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    source = public_row(row)
    hypothesis_id = str(source.get("hypothesis_id"))
    level_index = int(source.get("level_index"))
    score = source.get("score") or {}
    best_match = score.get("best_controlled_match") or {}
    threshold = confirmed_iou_threshold(score)
    source_status = score.get("confirmation_status")
    blocked_reasons: list[str] = []
    review_decision = None if decision is None else str(decision.get("review_decision"))

    if source.get("schema") != HYPOTHESIS_SCHEMA:
        blocked_reasons.append("source_hypothesis_bad_schema")
    if source.get("evidence_status") != "hypothesis":
        blocked_reasons.append("source_not_marked_hypothesis")
    for flag in ["admitted_for_control_truth", "admitted_for_goal_condition", "admitted_for_contra_object_view"]:
        if source.get(flag) is not False:
            blocked_reasons.append(f"source_{flag}_not_false")

    controlled_label = None
    if best_match.get("object_id") is not None:
        controlled_label = controlled_by_key.get((level_index, str(best_match.get("object_id"))))

    if decision is None:
        status = "blocked"
        blocked_reasons.append("no_review_decision_recorded")
    elif review_decision == "reject":
        status = "rejected"
    elif review_decision == "block":
        status = "blocked"
        blocked_reasons.append(str(decision.get("review_note") or "review_decision_blocked"))
    elif review_decision == "accept":
        blocked_reasons.extend(review_provenance_errors(decision))
        if source_status != "confirmed_against_controlled_mask":
            blocked_reasons.append("source_not_confirmed_against_controlled_mask")
        if float(best_match.get("iou") or 0.0) < threshold:
            blocked_reasons.append("controlled_match_below_threshold")
        if controlled_label is None:
            blocked_reasons.append("controlled_match_label_missing")
        status = "accepted" if not blocked_reasons else "blocked"
    else:
        status = "blocked"
        blocked_reasons.append("invalid_or_unknown_review_decision")

    accepted_match = best_match if status == "accepted" else None
    controlled_label_hash = stable_json_hash(controlled_label) if controlled_label is not None else None
    return {
        "schema": ACCEPTANCE_SCHEMA,
        "game": "sp80_liquid_pour",
        "calibration_slice_index": int(source.get("calibration_slice_index", level_index)),
        "level_index": level_index,
        "hypothesis_id": hypothesis_id,
        "source_hypothesis_jsonl": relpath(row_path),
        "source_hypothesis_line_no": int(row.get("_source_line_no") or 0),
        "source_hypothesis_json_sha256": stable_json_hash(source),
        "source_hypothesis_status": source.get("hypothesis_status"),
        "source_score_confirmation_status": source_status,
        "source_prompt": {
            "prompt_source": source.get("prompt_source"),
            "prompt_source_object_id": source.get("prompt_source_object_id"),
            "prompt_point_display_xy": source.get("prompt_point_display_xy"),
        },
        "source_pixel_mask": source.get("pixel_mask"),
        "source_best_controlled_match": best_match or None,
        "controlled_match_threshold": threshold,
        "controlled_label_json_sha256": controlled_label_hash,
        "review_decision_recorded": decision is not None,
        "review_decision": decision,
        "acceptance_status": status,
        "blocked_reasons": sorted(set(blocked_reasons)),
        "accepted_controlled_match": accepted_match,
        "accepted_semantic_label": None if accepted_match is None else accepted_match.get("semantic_label"),
        "accepted_source_tag": None if accepted_match is None else accepted_match.get("source_tag"),
        "contra_object_view_eligible": status == "accepted",
        "admission_scope": "object_hypothesis_review_only",
        "admitted_for_control_truth": False,
        "admitted_for_goal_condition": False,
        "admitted_for_contra_object_view": False,
        "admitted_for_action_choice": False,
        "raw_pixels_are_observation_ground_truth": True,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
    }


def build_scorecard(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    acceptance_counts = Counter(row["acceptance_status"] for row in rows)
    source_status_counts = Counter((public_row(row).get("score") or {}).get("confirmation_status") for row in source_rows)
    review_counts = Counter(
        (row.get("review_decision") or {}).get("review_decision") if row.get("review_decision_recorded") else "none"
        for row in rows
    )
    scorecard = {
        "schema": ACCEPTANCE_SCORECARD_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "hypothesis_count": len(source_rows),
        "acceptance_status_counts": dict(sorted(acceptance_counts.items())),
        "source_confirmation_status_counts": dict(sorted(source_status_counts.items())),
        "review_decision_counts": dict(sorted(review_counts.items())),
        "accepted_count": int(acceptance_counts.get("accepted", 0)),
        "rejected_count": int(acceptance_counts.get("rejected", 0)),
        "blocked_count": int(acceptance_counts.get("blocked", 0)),
        "contra_object_view_eligible_count": sum(1 for row in rows if row.get("contra_object_view_eligible") is True),
        "admitted_for_control_truth_count": sum(1 for row in rows if row.get("admitted_for_control_truth") is True),
        "admitted_for_goal_condition_count": sum(1 for row in rows if row.get("admitted_for_goal_condition") is True),
        "admitted_for_contra_object_view_count": sum(1 for row in rows if row.get("admitted_for_contra_object_view") is True),
        "admitted_for_action_choice_count": sum(1 for row in rows if row.get("admitted_for_action_choice") is True),
        "default_policy": "fail_closed_block_without_review_decision",
        "scorecard_json": relpath(out_path),
    }
    write_json(out_path, scorecard)
    return scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("codexlab/runs/sp80_sam_hypothesis_viewer"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--review-decisions", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = resolve_repo_path(args.run_dir)
    out_dir = resolve_repo_path(args.out_dir) if args.out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest_path = run_dir / "sam_hypothesis_manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_rows_path = resolve_repo_path(source_manifest["sam_hypotheses_jsonl"])
    controlled_path = resolve_repo_path(source_manifest["controlled_object_labels_jsonl"])
    source_scorecard_path = resolve_repo_path(source_manifest["sam_scorecard_json"])
    source_rows = load_jsonl(source_rows_path)
    controlled_rows = load_jsonl(controlled_path)
    controlled_by_key = {
        (int(row["level_index"]), str(row["object_id"])): public_row(row)
        for row in controlled_rows
        if row.get("schema") == CONTROLLED_LABEL_SCHEMA
    }

    decisions_path = resolve_repo_path(args.review_decisions) if args.review_decisions is not None else None
    decisions, decision_errors = load_review_decisions(decisions_path)
    source_ids = {str(public_row(row).get("hypothesis_id")) for row in source_rows}
    unknown_decisions = sorted(set(decisions) - source_ids)
    if unknown_decisions:
        decision_errors.extend(f"review decision references missing hypothesis {hypothesis_id}" for hypothesis_id in unknown_decisions)

    rows = [
        acceptance_for(
            row=row,
            row_path=source_rows_path,
            decision=decisions.get(str(public_row(row).get("hypothesis_id"))),
            controlled_by_key=controlled_by_key,
        )
        for row in source_rows
    ]
    rows_path = out_dir / "accepted_object_hypotheses.jsonl"
    write_jsonl(rows_path, rows)
    scorecard_path = out_dir / "sam_acceptance_scorecard.json"
    scorecard = build_scorecard(rows, source_rows, scorecard_path)
    manifest_path = out_dir / "sam_acceptance_manifest.json"
    manifest = {
        "schema": ACCEPTANCE_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "sp80_sam_acceptance_gate",
        "condition_natural_language": "Record fail-closed review acceptance status for SP80 SAM mask hypotheses without admitting SAM rows into Contra, goal conditions, or action choice.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "game": "sp80_liquid_pour",
        "default_policy": "fail_closed_block_without_review_decision",
        "raw_pixels_are_observation_ground_truth": True,
        "sam_rows_remain_hypotheses": True,
        "review_decisions_jsonl": None if decisions_path is None else relpath(decisions_path),
        "review_decisions_jsonl_sha256": None if decisions_path is None else sha256_file(decisions_path),
        "review_decision_error_count": len(decision_errors),
        "review_decision_errors": decision_errors,
        "source_sam_manifest_json": relpath(source_manifest_path),
        "source_sam_manifest_json_sha256": sha256_file(source_manifest_path),
        "source_sam_hypotheses_jsonl": relpath(source_rows_path),
        "source_sam_hypotheses_jsonl_sha256": sha256_file(source_rows_path),
        "source_controlled_object_labels_jsonl": relpath(controlled_path),
        "source_controlled_object_labels_jsonl_sha256": sha256_file(controlled_path),
        "source_sam_scorecard_json": relpath(source_scorecard_path),
        "source_sam_scorecard_json_sha256": sha256_file(source_scorecard_path),
        "accepted_object_hypotheses_jsonl": relpath(rows_path),
        "accepted_object_hypotheses_jsonl_sha256": sha256_file(rows_path),
        "sam_acceptance_scorecard_json": scorecard["scorecard_json"],
        "sam_acceptance_scorecard_json_sha256": sha256_file(scorecard_path),
        "hypothesis_count": len(source_rows),
        "acceptance_status_counts": scorecard["acceptance_status_counts"],
        "source_script": "codexlab/sp80_sam_acceptance_gate.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_sam_acceptance_gate.py"),
        "source_verifier": "codexlab/sp80_verify_sam_acceptance.py",
        "source_verifier_sha256": sha256_file(ROOT / "codexlab" / "sp80_verify_sam_acceptance.py")
        if (ROOT / "codexlab" / "sp80_verify_sam_acceptance.py").is_file()
        else None,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "admitted_for_control_truth_count": scorecard["admitted_for_control_truth_count"],
        "admitted_for_goal_condition_count": scorecard["admitted_for_goal_condition_count"],
        "admitted_for_contra_object_view_count": scorecard["admitted_for_contra_object_view_count"],
        "admitted_for_action_choice_count": scorecard["admitted_for_action_choice_count"],
    }
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "exported",
                "manifest": relpath(manifest_path),
                "accepted_object_hypotheses_jsonl": relpath(rows_path),
                "acceptance_status_counts": scorecard["acceptance_status_counts"],
                "review_decision_error_count": len(decision_errors),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
