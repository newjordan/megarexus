#!/usr/bin/env python3
"""Promote reviewed SAM rows into Contra object-view admissions only.

This is the second gate after SAM acceptance. It admits only rows that are
already accepted and Contra-object-view eligible. It still forbids admission to
control truth, goal conditions, or action choice.
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

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_SCHEMA = "codexlab.sp80_sam_acceptance.v1"
ADMISSION_SCHEMA = "codexlab.sp80_sam_contra_object_view_admission.v1"
ADMISSION_MANIFEST_SCHEMA = "codexlab.sp80_sam_contra_object_view_admission_manifest.v1"
ADMISSION_SCORECARD_SCHEMA = "codexlab.sp80_sam_contra_object_view_admission_scorecard.v1"


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


def mask_geometry(mask_path: Path) -> dict[str, Any]:
    mask = np.load(mask_path, allow_pickle=False).astype(bool)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "area_pixels": 0,
            "bbox_xyxy": None,
            "centroid_xy": None,
            "width": 0,
            "height": 0,
        }
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "area_pixels": int(len(xs)),
        "bbox_xyxy": bbox,
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "width": int(bbox[2] - bbox[0] + 1),
        "height": int(bbox[3] - bbox[1] + 1),
    }


def row_block_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if row.get("schema") != ACCEPTANCE_SCHEMA:
        reasons.append("acceptance_row_bad_schema")
    if row.get("acceptance_status") != "accepted":
        reasons.append("acceptance_status_not_accepted")
    if row.get("contra_object_view_eligible") is not True:
        reasons.append("not_contra_object_view_eligible")
    if row.get("review_decision_recorded") is not True:
        reasons.append("review_decision_not_recorded")
    if row.get("accepted_controlled_match") is None:
        reasons.append("accepted_controlled_match_missing")
    for flag in ["admitted_for_control_truth", "admitted_for_goal_condition", "admitted_for_contra_object_view", "admitted_for_action_choice"]:
        if row.get(flag) is not False:
            reasons.append(f"source_{flag}_not_false")
    return reasons


def build_admission_row(row: dict[str, Any], rows_path: Path) -> dict[str, Any] | None:
    source = public_row(row)
    reasons = row_block_reasons(source)
    if reasons:
        return None
    mask_meta = source.get("source_pixel_mask") or {}
    npy_meta = mask_meta.get("npy") or {}
    mask_path_value = npy_meta.get("path")
    if not mask_path_value:
        return None
    mask_path = resolve_repo_path(mask_path_value)
    geometry = mask_geometry(mask_path)
    level_index = int(source["level_index"])
    hypothesis_id = str(source["hypothesis_id"])
    return {
        "schema": ADMISSION_SCHEMA,
        "game": "sp80_liquid_pour",
        "level_index": level_index,
        "calibration_slice_index": int(source.get("calibration_slice_index", level_index)),
        "sam_object_id": f"sam_obj_l{level_index:02d}_{hypothesis_id}",
        "hypothesis_id": hypothesis_id,
        "mask_source": "sam2_hypothesis_after_acceptance_gate",
        "admission_status": "admitted_for_contra_object_view_only",
        "admission_scope": "contra_object_view_only_not_control_truth_not_goal_truth_not_action_choice",
        "source_acceptance_jsonl": relpath(rows_path),
        "source_acceptance_line_no": int(row.get("_source_line_no") or 0),
        "source_acceptance_json_sha256": stable_json_hash(source),
        "source_hypothesis_jsonl": source.get("source_hypothesis_jsonl"),
        "source_hypothesis_line_no": source.get("source_hypothesis_line_no"),
        "source_hypothesis_json_sha256": source.get("source_hypothesis_json_sha256"),
        "source_pixel_mask": source.get("source_pixel_mask"),
        "source_score_confirmation_status": source.get("source_score_confirmation_status"),
        "accepted_controlled_match": source.get("accepted_controlled_match"),
        "accepted_semantic_label": source.get("accepted_semantic_label"),
        "accepted_source_tag": source.get("accepted_source_tag"),
        "review_decision_recorded": source.get("review_decision_recorded"),
        "review_decision": source.get("review_decision"),
        "raw_pixels_are_observation_ground_truth": True,
        "area_pixels": geometry["area_pixels"],
        "bbox_xyxy": geometry["bbox_xyxy"],
        "centroid_xy": geometry["centroid_xy"],
        "width": geometry["width"],
        "height": geometry["height"],
        "admitted_for_contra_object_view": True,
        "admitted_for_control_truth": False,
        "admitted_for_goal_condition": False,
        "admitted_for_action_choice": False,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
    }


def build_scorecard(admissions: list[dict[str, Any]], acceptance_rows: list[dict[str, Any]], out_path: Path) -> dict[str, Any]:
    acceptance_counts = Counter(public_row(row).get("acceptance_status") for row in acceptance_rows)
    eligible_count = sum(1 for row in acceptance_rows if public_row(row).get("contra_object_view_eligible") is True)
    scorecard = {
        "schema": ADMISSION_SCORECARD_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "acceptance_row_count": len(acceptance_rows),
        "acceptance_status_counts": dict(sorted(acceptance_counts.items())),
        "contra_object_view_eligible_count": int(eligible_count),
        "admitted_for_contra_object_view_count": len(admissions),
        "admitted_for_control_truth_count": 0,
        "admitted_for_goal_condition_count": 0,
        "admitted_for_action_choice_count": 0,
        "default_policy": "fail_closed_require_accepted_reviewed_contra_eligible_sam_row",
        "scorecard_json": relpath(out_path),
    }
    write_json(out_path, scorecard)
    return scorecard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("codexlab/runs/sp80_sam_hypothesis_viewer"))
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = resolve_repo_path(args.run_dir)
    out_dir = resolve_repo_path(args.out_dir) if args.out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    acceptance_manifest_path = run_dir / "sam_acceptance_manifest.json"
    acceptance_manifest = load_json(acceptance_manifest_path)
    acceptance_rows_path = resolve_repo_path(acceptance_manifest["accepted_object_hypotheses_jsonl"])
    acceptance_rows = load_jsonl(acceptance_rows_path)
    admissions = [
        admission
        for row in acceptance_rows
        if (admission := build_admission_row(row, acceptance_rows_path)) is not None
    ]
    admissions_path = out_dir / "contra_sam_admitted_masks.jsonl"
    write_jsonl(admissions_path, admissions)
    scorecard_path = out_dir / "contra_sam_admission_scorecard.json"
    scorecard = build_scorecard(admissions, acceptance_rows, scorecard_path)
    manifest_path = out_dir / "contra_sam_admission_manifest.json"
    manifest = {
        "schema": ADMISSION_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "sp80_sam_contra_object_view_admission_gate",
        "condition_natural_language": "Admit only accepted reviewed SAM masks into the Contra object view while keeping control truth, goal truth, and action choice fail-closed.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "game": "sp80_liquid_pour",
        "default_policy": "fail_closed_require_accepted_reviewed_contra_eligible_sam_row",
        "raw_pixels_are_observation_ground_truth": True,
        "source_acceptance_manifest_json": relpath(acceptance_manifest_path),
        "source_acceptance_manifest_json_sha256": sha256_file(acceptance_manifest_path),
        "source_acceptance_jsonl": relpath(acceptance_rows_path),
        "source_acceptance_jsonl_sha256": sha256_file(acceptance_rows_path),
        "contra_sam_admitted_masks_jsonl": relpath(admissions_path),
        "contra_sam_admitted_masks_jsonl_sha256": sha256_file(admissions_path),
        "contra_sam_admission_scorecard_json": scorecard["scorecard_json"],
        "contra_sam_admission_scorecard_json_sha256": sha256_file(scorecard_path),
        "acceptance_row_count": len(acceptance_rows),
        "contra_object_view_eligible_count": scorecard["contra_object_view_eligible_count"],
        "admitted_for_contra_object_view_count": len(admissions),
        "admitted_for_control_truth_count": 0,
        "admitted_for_goal_condition_count": 0,
        "admitted_for_action_choice_count": 0,
        "source_script": "codexlab/sp80_sam_contra_admission_gate.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_sam_contra_admission_gate.py"),
        "source_verifier": "codexlab/sp80_verify_sam_contra_admission.py",
        "source_verifier_sha256": sha256_file(ROOT / "codexlab" / "sp80_verify_sam_contra_admission.py")
        if (ROOT / "codexlab" / "sp80_verify_sam_contra_admission.py").is_file()
        else None,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
    }
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": "exported",
                "manifest": relpath(manifest_path),
                "contra_sam_admitted_masks_jsonl": relpath(admissions_path),
                "admitted_for_contra_object_view_count": len(admissions),
                "admitted_for_control_truth_count": 0,
                "admitted_for_goal_condition_count": 0,
                "admitted_for_action_choice_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
