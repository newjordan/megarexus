#!/usr/bin/env python3
"""Build a blank human signoff scaffold for SP80 manual calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sp80_base_controller import ROOT, sha256_file
from sp80_object_label_viewer import relpath, resolve_repo_path, write_json


SIGNOFF_SCHEMA = "codexlab.sp80_manual_calibration_human_signoff_template.v1"


def blank_reviewed_expectations(source_raw_frame_sha256: str) -> dict[str, Any]:
    return {
        "actor_object_ids": [],
        "target_object_ids": [],
        "known_action_support_object_ids": [],
        "connector_object_ids": [],
        "known_y_object_ids": [],
        "unresolved_object_ids": [],
        "image_hash": source_raw_frame_sha256,
    }


def build_signoff_row(row: dict[str, Any]) -> dict[str, Any]:
    frame_hash = str(row.get("source_raw_frame_sha256") or "")
    return {
        "schema": "codexlab.sp80_manual_calibration_human_signoff_row_template.v1",
        "status": "requires_human_review",
        "level_index": int(row["level_index"]),
        "source_raw_frame_sha256": frame_hash,
        "source_raw_grid_sha256": row.get("source_raw_grid_sha256"),
        "raw_board": row.get("raw_board"),
        "scene_surface_sheet": row.get("scene_surface_sheet"),
        "seed_candidate_ids": row.get("seed_candidate_ids"),
        "human_reviewed_expectations": blank_reviewed_expectations(frame_hash),
        "fields_pending_human_review": row.get("fields_pending_human_review"),
        "review_instructions": {
            "actor_object_ids": "Fill with reviewed actor component ids.",
            "target_object_ids": "Fill with reviewed target/open-region component ids.",
            "known_action_support_object_ids": "Fill with reviewed support/action object ids.",
            "connector_object_ids": "Fill with reviewed connector ids; empty list is allowed if none exist.",
            "known_y_object_ids": "Fill with reviewed Y-object ids; empty list is allowed if none exist.",
            "unresolved_object_ids": "Fill with reviewed unresolved/ignore ids; empty list is allowed if none remain.",
            "image_hash": "Must equal source_raw_frame_sha256 for this level.",
        },
        "reviewed_by": "",
        "reviewed_at_utc": "",
        "review_note": "",
        "review_status": "not_reviewed",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-table",
        type=Path,
        default=Path("codexlab/runs/contra_v001_sp80_manual_calibration_table/manual_calibration_table.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("codexlab/runs/contra_v001_sp80_manual_calibration_table/human_signoff_template.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_path = resolve_repo_path(args.source_table)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = [build_signoff_row(row) for row in source.get("rows", []) if isinstance(row, dict)]
    payload = {
        "schema": SIGNOFF_SCHEMA,
        "status": "requires_human_review",
        "review_status": "not_reviewed",
        "source_table_json": relpath(source_path),
        "source_table_json_sha256": sha256_file(source_path),
        "row_count": len(rows),
        "rows": rows,
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
    out_path = resolve_repo_path(args.out)
    write_json(out_path, payload)
    print(
        json.dumps(
            {
                "status": "requires_human_review",
                "signoff_template": str(out_path.relative_to(ROOT)),
                "row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
