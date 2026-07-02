#!/usr/bin/env python3
"""Build concise CONTRA debug summaries from run manifests."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

from contra_head import summarize_manifest
from sp80_base_controller import ROOT, sha256_file
from sp80_object_label_viewer import relpath, resolve_repo_path, write_json


REPORT_SCHEMA = "codexlab.contra_debug_report.v1"


def rel_for_html(repo_relpath: str, index_dir: Path) -> str:
    target = ROOT / repo_relpath
    return html.escape(os.path.relpath(target.resolve(), index_dir.resolve()))


def manifest_label(path: Path) -> str:
    try:
        return path.parent.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.parent)


def load_summary(path: Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    summary = summarize_manifest(manifest, manifest_path=relpath(resolved))
    summary["source_manifest_sha256"] = sha256_file(resolved)
    summary["source_label"] = manifest_label(resolved)
    return summary


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CONTRA Debug Summary",
        "",
        "| source | views | StateCon active | GeoBand active | GeoBand inactive | top reasons |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in report["summaries"]:
        reasons = ", ".join(
            f"{key}:{value}" for key, value in list(summary.get("reason_counts", {}).items())[:6]
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary.get('source_label')}`",
                    str(summary.get("view_count")),
                    str(summary.get("statecon_active_view_count")),
                    str(summary.get("geoband_active_view_count")),
                    str(summary.get("geoband_inactive_view_count")),
                    reasons or "none",
                ]
            )
            + " |"
        )
    lines.append("")
    for summary in report["summaries"]:
        lines.extend(
            [
                f"## {summary.get('source_label')}",
                "",
                "| map | level | components | actors | targets | supports | StateCon | GeoBand | missing | sheet |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for view in summary.get("views", []):
            counts = view.get("counts", {})
            missing = view.get("missing_inputs", {}).get("geoband_hypotheses", [])
            artifacts = view.get("artifacts", {})
            sheet = artifacts.get("scene_surface_sheet")
            sheet_text = f"`{sheet}`" if sheet else ""
            game = view.get("game_id") or view.get("game_short") or "sp80"
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(game),
                        str(view.get("level_index")),
                        str(counts.get("raw_components")),
                        str(counts.get("actors")),
                        str(counts.get("targets")),
                        str(counts.get("supports")),
                        str(counts.get("statecons")),
                        str(counts.get("geobands")),
                        ", ".join(missing) or "none",
                        sheet_text,
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def build_html(report: dict[str, Any], *, index_dir: Path) -> str:
    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>CONTRA Debug Summary</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f7f7f4;color:#191919}",
        "table{border-collapse:collapse;font-size:12px;margin:12px 0 28px}",
        "td,th{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}",
        "code{background:#eee;padding:1px 3px}",
        ".active{color:#087f3f;font-weight:700}.inactive{color:#9b2c2c;font-weight:700}",
        "</style></head><body>",
        "<h1>CONTRA Debug Summary</h1>",
        "<table><thead><tr><th>source</th><th>views</th><th>StateCon active</th><th>GeoBand active</th><th>GeoBand inactive</th><th>top reasons</th></tr></thead><tbody>",
    ]
    for summary in report["summaries"]:
        reasons = ", ".join(
            f"{html.escape(str(key))}:{value}" for key, value in list(summary.get("reason_counts", {}).items())[:6]
        )
        body.append(
            "<tr>"
            f"<td><code>{html.escape(str(summary.get('source_label')))}</code></td>"
            f"<td>{summary.get('view_count')}</td>"
            f"<td>{summary.get('statecon_active_view_count')}</td>"
            f"<td>{summary.get('geoband_active_view_count')}</td>"
            f"<td>{summary.get('geoband_inactive_view_count')}</td>"
            f"<td>{reasons or 'none'}</td>"
            "</tr>"
        )
    body.append("</tbody></table>")
    for summary in report["summaries"]:
        body.append(f"<h2>{html.escape(str(summary.get('source_label')))}</h2>")
        body.append("<table><thead><tr><th>map</th><th>level</th><th>components</th><th>actors</th><th>targets</th><th>supports</th><th>StateCon</th><th>GeoBand</th><th>missing</th><th>sheet</th></tr></thead><tbody>")
        for view in summary.get("views", []):
            counts = view.get("counts", {})
            missing = view.get("missing_inputs", {}).get("geoband_hypotheses", [])
            artifacts = view.get("artifacts", {})
            sheet = artifacts.get("scene_surface_sheet")
            sheet_link = f"<a href='{rel_for_html(sheet, index_dir)}'>sheet</a>" if isinstance(sheet, str) else ""
            game = view.get("game_id") or view.get("game_short") or "sp80"
            geoband_class = "active" if view.get("geoband_active") else "inactive"
            statecon_class = "active" if view.get("statecon_active") else "inactive"
            body.append(
                "<tr>"
                f"<td>{html.escape(str(game))}</td>"
                f"<td>{html.escape(str(view.get('level_index')))}</td>"
                f"<td>{counts.get('raw_components')}</td>"
                f"<td>{counts.get('actors')}</td>"
                f"<td>{counts.get('targets')}</td>"
                f"<td>{counts.get('supports')}</td>"
                f"<td class='{statecon_class}'>{counts.get('statecons')}</td>"
                f"<td class='{geoband_class}'>{counts.get('geobands')}</td>"
                f"<td>{html.escape(', '.join(missing) or 'none')}</td>"
                f"<td>{sheet_link}</td>"
                "</tr>"
            )
        body.append("</tbody></table>")
    body.extend(["</body></html>"])
    return "\n".join(body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/contra_debug_summary"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [load_summary(path) for path in args.manifests]
    report = {
        "schema": REPORT_SCHEMA,
        "summary_count": len(summaries),
        "summaries": summaries,
    }
    report_path = out_dir / "contra_debug_summary.json"
    write_json(report_path, report)
    md_path = out_dir / "contra_debug_summary.md"
    md_path.write_text(build_markdown(report) + "\n", encoding="utf-8")
    html_path = out_dir / "index.html"
    html_path.write_text(build_html(report, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "exported",
                "summary_json": relpath(report_path),
                "summary_markdown": relpath(md_path),
                "index_html": relpath(html_path),
                "summary_count": len(summaries),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
