#!/usr/bin/env python3
"""Export a human-review SP80 calibration table from static CONTRA views."""

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


def rel_for_html(repo_relpath: str, index_dir: Path) -> str:
    target = ROOT / repo_relpath
    return html.escape(os.path.relpath(target.resolve(), index_dir.resolve()))


def html_list(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(f"<code>{html.escape(str(value))}</code>" for value in values)


def review_row_from_view(view: dict[str, Any]) -> dict[str, Any]:
    review = view.get("manual_calibration_review")
    if not isinstance(review, dict):
        raise RuntimeError(f"view level {view.get('level_index')} missing manual_calibration_review")
    seed = review.get("seed_candidate_ids")
    if not isinstance(seed, dict):
        seed = {}
    return {
        "schema": "codexlab.sp80_manual_calibration_table_row.v1",
        "status": "requires_human_review",
        "level_index": int(view["level_index"]),
        "source_raw_frame_sha256": view.get("source_raw_frame_sha256"),
        "source_raw_grid_sha256": view.get("source_raw_grid_sha256"),
        "raw_board": view.get("raw_board"),
        "scene_surface_sheet": view.get("scene_surface_sheet"),
        "seed_candidate_ids": seed,
        "human_reviewed_expectations": {
            "actor_object_ids": [],
            "target_object_ids": [],
            "known_action_support_object_ids": [],
            "connector_object_ids": [],
            "known_y_object_ids": [],
            "unresolved_object_ids": [],
            "image_hash": None,
        },
        "fields_pending_human_review": review.get("fields_pending_human_review"),
        "review_status": "not_reviewed",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_index(manifest: dict[str, Any], *, index_dir: Path) -> str:
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>SP80 Manual Calibration Table</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f7f7f4;color:#191919}",
        "table{border-collapse:collapse;font-size:12px;margin-top:10px}",
        "td,th{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}",
        "img{image-rendering:pixelated;border:1px solid #aaa;background:#111;width:256px;height:auto}",
        "code{background:#eee;padding:1px 3px}",
        "</style></head><body>",
        "<h1>SP80 Manual Calibration Table</h1>",
        "<p>This is a human-review table. Seed candidates are review aids only and are not admitted as semantic truth, control truth, goal truth, action choice, or solve truth.</p>",
        f"<p><b>Manifest:</b> <code>{html.escape(str(manifest.get('manifest_json')))}</code></p>",
        "<table><thead><tr><th>level</th><th>raw</th><th>seed actor ids</th><th>seed target ids</th><th>seed support/action ids</th><th>connectors</th><th>known-Y</th><th>unresolved</th><th>review status</th></tr></thead><tbody>",
    ]
    for row in manifest.get("rows", []):
        if not isinstance(row, dict):
            continue
        seed = row.get("seed_candidate_ids") if isinstance(row.get("seed_candidate_ids"), dict) else {}
        raw_path = ((row.get("raw_board") or {}).get("path")) if isinstance(row.get("raw_board"), dict) else None
        raw_img = f"<img src='{rel_for_html(raw_path, index_dir)}'>" if isinstance(raw_path, str) else ""
        body.append(
            "<tr>"
            f"<td>{int(row.get('level_index'))}<br><code>{html.escape(str(row.get('source_raw_frame_sha256'))[:12])}</code></td>"
            f"<td>{raw_img}</td>"
            f"<td>{html_list(seed.get('actor_object_ids'))}</td>"
            f"<td>{html_list(seed.get('target_object_ids'))}</td>"
            f"<td>{html_list(seed.get('known_action_support_object_ids'))}</td>"
            f"<td>{html_list(seed.get('connector_component_ids'))}</td>"
            f"<td>{html_list(seed.get('known_y_component_ids'))}</td>"
            f"<td>{html_list(seed.get('unresolved_object_ids'))}</td>"
            f"<td>{html.escape(str(row.get('review_status')))} / {html.escape(str(row.get('admission_status')))}</td>"
            "</tr>"
        )
    body.extend(["</tbody></table>", "</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("codexlab/runs/contra_v001_sp80_boundary_pass_review/manifest.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("codexlab/runs/contra_v001_sp80_manual_calibration_table"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_manifest_path = resolve_repo_path(args.source_manifest)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [review_row_from_view(view) for view in source_manifest.get("views", []) if isinstance(view, dict)]
    table = {
        "schema": TABLE_SCHEMA,
        "status": "requires_human_review",
        "review_status": "not_reviewed",
        "source_manifest_json": relpath(source_manifest_path),
        "source_manifest_json_sha256": sha256_file(source_manifest_path),
        "rows": rows,
        "row_count": len(rows),
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
        "status": "requires_human_review",
        "condition_name": "sp80_manual_calibration_table_review",
        "condition_natural_language": "Export SP80 calibration seed candidates into a human-review table without admitting them as truth.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "source_manifest_json": relpath(source_manifest_path),
        "source_manifest_json_sha256": sha256_file(source_manifest_path),
        "table_json": relpath(table_path),
        "table_json_sha256": sha256_file(table_path),
        "row_count": len(rows),
        "rows": rows,
        "review_status": "not_reviewed",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "source_script": "codexlab/sp80_manual_calibration_table.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "sp80_manual_calibration_table.py"),
    }
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)
    index_path = out_dir / "index.html"
    index_path.write_text(build_index(manifest, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "requires_human_review",
                "manifest": relpath(manifest_path),
                "index_html": relpath(index_path),
                "row_count": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
