#!/usr/bin/env python3
"""Apply a completed human SP80 calibration signoff into a reviewed table."""

from __future__ import annotations

import argparse
import html
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sp80_base_controller import ROOT, sha256_file
from sp80_object_label_viewer import relpath, resolve_repo_path, write_json


MANIFEST_SCHEMA = "codexlab.sp80_manual_calibration_table_manifest.v1"
TABLE_SCHEMA = "codexlab.sp80_manual_calibration_table.v1"
ROW_SCHEMA = "codexlab.sp80_manual_calibration_table_row.v1"
REQUIRED_FIELDS = {
    "actor_object_ids",
    "target_object_ids",
    "known_action_support_object_ids",
    "connector_object_ids",
    "known_y_object_ids",
    "unresolved_object_ids",
    "image_hash",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def validate_reviewed_row(row: dict[str, Any], *, label: str) -> None:
    require(row.get("status") == "human_reviewed", f"{label} status must be human_reviewed")
    require(row.get("review_status") == "human_reviewed", f"{label} review_status must be human_reviewed")
    require(row.get("fields_pending_human_review") == [], f"{label} still has pending fields")
    for key in ("reviewed_by", "reviewed_at_utc", "review_note"):
        require(isinstance(row.get(key), str) and bool(row.get(key)), f"{label} missing {key}")
    reviewed = row.get("human_reviewed_expectations")
    require(isinstance(reviewed, dict), f"{label} missing human_reviewed_expectations")
    if not isinstance(reviewed, dict):
        return
    require(set(reviewed) == REQUIRED_FIELDS, f"{label} reviewed fields mismatch")
    require(reviewed.get("image_hash") == row.get("source_raw_frame_sha256"), f"{label} image hash mismatch")
    for key in REQUIRED_FIELDS - {"image_hash"}:
        values = reviewed.get(key)
        require(isinstance(values, list), f"{label} {key} must be a list")
        require(all(isinstance(value, str) and value for value in values), f"{label} {key} has non-string ids")


def reviewed_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": ROW_SCHEMA,
        "status": "human_reviewed",
        "level_index": int(row["level_index"]),
        "source_raw_frame_sha256": row.get("source_raw_frame_sha256"),
        "source_raw_grid_sha256": row.get("source_raw_grid_sha256"),
        "raw_board": row.get("raw_board"),
        "scene_surface_sheet": row.get("scene_surface_sheet"),
        "seed_candidate_ids": row.get("seed_candidate_ids"),
        "human_reviewed_expectations": row.get("human_reviewed_expectations"),
        "fields_pending_human_review": [],
        "reviewed_by": row.get("reviewed_by"),
        "reviewed_at_utc": row.get("reviewed_at_utc"),
        "review_note": row.get("review_note"),
        "review_status": "human_reviewed",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def rel_for_html(repo_relpath: str, index_dir: Path) -> str:
    target = ROOT / repo_relpath
    return html.escape(os.path.relpath(target.resolve(), index_dir.resolve()))


def html_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(f"<code>{html.escape(str(value))}</code>" for value in values)


def build_index(manifest: dict[str, Any], *, index_dir: Path) -> str:
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 Reviewed Manual Calibration</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f7f7f4;color:#191919}",
        "table{border-collapse:collapse;font-size:12px;margin-top:10px}",
        "td,th{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}",
        "img{image-rendering:pixelated;border:1px solid #aaa;background:#111;width:256px;height:auto}",
        "code{background:#eee;padding:1px 3px}",
        "</style></head><body>",
        "<h1>SP80 Reviewed Manual Calibration</h1>",
        "<p>Human-reviewed calibration expectations. These rows still are not admitted as control truth, goal truth, action choice, or solve truth.</p>",
        f"<p><b>Manifest:</b> <code>{html.escape(str(manifest.get('manifest_json')))}</code></p>",
        "<table><thead><tr><th>level</th><th>raw</th><th>actor ids</th><th>target ids</th><th>support/action ids</th><th>connectors</th><th>known-Y</th><th>unresolved</th><th>reviewer</th></tr></thead><tbody>",
    ]
    for row in manifest.get("rows", []):
        if not isinstance(row, dict):
            continue
        reviewed = row.get("human_reviewed_expectations") if isinstance(row.get("human_reviewed_expectations"), dict) else {}
        raw_path = ((row.get("raw_board") or {}).get("path")) if isinstance(row.get("raw_board"), dict) else None
        raw_img = f"<img src='{rel_for_html(raw_path, index_dir)}'>" if isinstance(raw_path, str) else ""
        body.append(
            "<tr>"
            f"<td>{int(row.get('level_index'))}<br><code>{html.escape(str(row.get('source_raw_frame_sha256'))[:12])}</code></td>"
            f"<td>{raw_img}</td>"
            f"<td>{html_list(reviewed.get('actor_object_ids'))}</td>"
            f"<td>{html_list(reviewed.get('target_object_ids'))}</td>"
            f"<td>{html_list(reviewed.get('known_action_support_object_ids'))}</td>"
            f"<td>{html_list(reviewed.get('connector_object_ids'))}</td>"
            f"<td>{html_list(reviewed.get('known_y_object_ids'))}</td>"
            f"<td>{html_list(reviewed.get('unresolved_object_ids'))}</td>"
            f"<td>{html.escape(str(row.get('reviewed_by')))}<br>{html.escape(str(row.get('reviewed_at_utc')))}</td>"
            "</tr>"
        )
    body.extend(["</tbody></table>", "</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signoff",
        type=Path,
        default=Path("codexlab/runs/contra_v001_sp80_manual_calibration_table/human_signoff_template.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("codexlab/runs/contra_v001_sp80_manual_calibration_reviewed"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    signoff_path = resolve_repo_path(args.signoff)
    signoff = json.loads(signoff_path.read_text(encoding="utf-8"))
    require(signoff.get("status") == "human_reviewed", "signoff status must be human_reviewed")
    require(signoff.get("review_status") == "human_reviewed", "signoff review_status must be human_reviewed")
    rows = signoff.get("rows")
    require(isinstance(rows, list) and bool(rows), "signoff rows must be a non-empty list")
    reviewed_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows if isinstance(rows, list) else []):
        require(isinstance(row, dict), f"rows[{idx}] must be an object")
        validate_reviewed_row(row, label=f"rows[{idx}]")
        reviewed_rows.append(reviewed_row(row))

    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {
        "schema": TABLE_SCHEMA,
        "status": "human_reviewed",
        "review_status": "human_reviewed",
        "source_signoff_json": relpath(signoff_path),
        "source_signoff_json_sha256": sha256_file(signoff_path),
        "rows": reviewed_rows,
        "row_count": len(reviewed_rows),
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
    table_path = out_dir / "manual_calibration_table.json"
    write_json(table_path, table)
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "human_reviewed",
        "condition_name": "sp80_manual_calibration_table_human_reviewed",
        "condition_natural_language": "Apply a completed human signoff into a reviewed SP80 calibration table without admitting it as control truth.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "source_signoff_json": relpath(signoff_path),
        "source_signoff_json_sha256": sha256_file(signoff_path),
        "table_json": relpath(table_path),
        "table_json_sha256": sha256_file(table_path),
        "row_count": len(reviewed_rows),
        "rows": reviewed_rows,
        "review_status": "human_reviewed",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "source_script": "codexlab/sp80_apply_manual_calibration_signoff.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_apply_manual_calibration_signoff.py"),
    }
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)
    index_path = out_dir / "index.html"
    index_path.write_text(build_index(manifest, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "human_reviewed",
                "manifest": relpath(manifest_path),
                "index_html": relpath(index_path),
                "row_count": len(reviewed_rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
