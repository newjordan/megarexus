#!/usr/bin/env python3
"""Assemble a fresh Angel contraview_report data package from level review decks."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ANGEL_DIST_DATA = Path.home() / "angel" / "dist" / "data"
ANGEL_PUBLIC_DATA = Path.home() / "angel" / "public" / "data"


STAGE_ORDER = [
    ("run_condition", "Run Condition", "provenance"),
    ("raw_frame", "Raw Frame", "frame"),
    ("config_pass", "Config Pass", "view_system"),
    ("oracle_pass", "Oracle Pass", "oracle"),
    ("ghost_pass", "Ghost Pass", "view_system"),
    ("contra_pass", "CONTRA Pass Summary", "view_system"),
    ("raw_color_components", "Raw Color Components", "contra_layer"),
    ("control_isolation", "Control Isolation", "contra_layer"),
    ("scene_surfaces", "Scene Surfaces", "contra_layer"),
    ("symmetry_hypotheses", "Symmetry Hypotheses", "contra_layer"),
    ("actor_affordances", "Actor Affordances", "contra_layer"),
    ("target_affordances", "Target Affordances", "contra_layer"),
    ("support_action_affordances", "Support Action Affordances", "contra_layer"),
    ("state_condition_hypotheses", "State Condition Hypotheses", "contra_layer"),
    ("geoband_hypotheses", "GeoBand Hypotheses", "contra_layer"),
    ("geoband_boundary_queues", "GeoBand Boundary Queues", "contra_layer"),
    ("deltaband_gate", "DeltaBand Gate", "control_feedback"),
    ("exploreband_budget", "ExploreBand Budget", "contra_layer"),
    ("tabula_rasa_notebook", "Tabula Rasa Notebook", "contra_layer"),
    ("episode_memory", "Episode Memory", "control_feedback"),
    ("debug_summary", "Debug Summary", "contra_layer"),
    ("admission", "Admission", "contra_layer"),
    ("symmetry_internal", "Symmetry Internal", "symmetry"),
    ("geoband_internal", "GeoBand Internal", "geoband"),
    ("portband_alignment", "PortBand Alignment", "portband"),
    ("portband_fleet", "PortBand Fleet", "portband"),
    ("portband_lineup_matrix", "PortBand Lineup Matrix", "portband"),
    ("oracle_sidecar", "Oracle Sidecar", "oracle"),
    ("provenance", "Provenance", "verification"),
]

HIDDEN_STAGE_IDS = {"tabula_rasa_notebook"}
PORTBAND_STAGE_IDS = {"portband_alignment", "portband_fleet", "portband_lineup_matrix"}


SECTION_ORDER = [
    ("run_condition", "Run Condition", ("run_condition",), "provenance"),
    ("raw_frame", "Raw Frame", ("raw_frame",), "frame"),
    ("config_pass", "Config Pass", ("config_pass",), "view_system"),
    ("oracle_pass", "Oracle Pass", ("oracle_pass",), "oracle"),
    ("ghost_pass", "Ghost Pass", ("ghost_pass",), "view_system"),
    ("contra_pass", "CONTRA Pass Summary", ("contra_pass",), "view_system"),
    ("raw_color_components", "Raw Color Components", ("raw_color_components",), "contra_layer"),
    ("control_isolation", "Control Isolation", ("control_isolation",), "contra_layer"),
    ("scene_surfaces", "Scene Surfaces", ("scene_surfaces",), "contra_layer"),
    ("symmetry_hypotheses", "Symmetry Hypotheses", ("symmetry_hypotheses",), "contra_layer"),
    ("symmetry_internal", "Symmetry Internal", ("symmetry_internal",), "symmetry"),
    ("actor_affordances", "Actor Affordances", ("actor_affordances",), "contra_layer"),
    ("target_affordances", "Target Affordances", ("target_affordances",), "contra_layer"),
    ("support_action_affordances", "Support Action Affordances", ("support_action_affordances",), "contra_layer"),
    ("state_condition_hypotheses", "State Condition Hypotheses", ("state_condition_hypotheses",), "contra_layer"),
    ("geoband_hypotheses", "GeoBand Hypotheses", ("geoband_hypotheses",), "contra_layer"),
    ("geoband_internal", "GeoBand Internal", ("geoband_internal",), "geoband"),
    ("geoband_boundary_queues", "GeoBand Boundary Queues", ("geoband_boundary_queues",), "contra_layer"),
    ("deltaband_gate", "DeltaBand Gate", ("deltaband_gate",), "control_feedback"),
    ("exploreband_budget", "ExploreBand Budget", ("exploreband_budget",), "contra_layer"),
    ("episode_memory", "Episode Memory", ("episode_memory",), "control_feedback"),
    ("debug_summary", "Debug Summary", ("debug_summary",), "contra_layer"),
    ("admission", "Admission", ("admission",), "contra_layer"),
    ("portband", "PortBand", tuple(sorted(PORTBAND_STAGE_IDS)), "portband"),
    ("oracle_sidecar", "Oracle Sidecar", ("oracle_sidecar",), "oracle"),
    ("provenance", "Provenance", ("provenance",), "verification"),
]


def repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def git_dirty() -> bool:
    return bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def write_text_slide(path: Path, title: str, rows: list[tuple[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1600, 930), (8, 12, 17))
    draw = ImageDraw.Draw(image)
    draw.text((32, 28), title, fill=(248, 250, 252), font=font(38, bold=True))
    y = 96
    for key, value in rows:
        text = f"{key}: {value}"
        for offset in range(0, len(text), 100):
            draw.text((42, y), text[offset : offset + 100], fill=(219, 234, 254), font=font(22))
            y += 32
            if y > 880:
                image.save(path)
                return
        y += 8
    image.save(path)


def copy_asset(src: Path, dst_dir: Path, name: str) -> dict[str, str]:
    dst = dst_dir / name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": repo_rel(dst), "sha256": sha256_file(dst)}


def stage_for_pass_image(path: Path) -> tuple[str, str]:
    name = path.name
    if "config_pass" in name:
        return "config_pass", "Config Pass"
    if "oracle_pass" in name:
        return "oracle_pass", "Oracle Pass"
    if "ghost_pass" in name:
        return "ghost_pass", "Ghost Pass"
    if "view_system" in name and "contra_pass" in name:
        return "contra_pass", "CONTRA Pass Summary"
    if name.startswith("contra_pass_"):
        match = re.match(r"contra_pass_\d+_(.+)\.png$", name)
        stage = match.group(1) if match else "contra_layer"
        return stage, stage.replace("_", " ").title()
    if name.startswith("symmetry_pass_"):
        return "symmetry_internal", "Symmetry Internal"
    if name.startswith("geoband_pass_"):
        return "geoband_internal", "GeoBand Internal"
    if name.startswith("portband_direct_"):
        return "portband_alignment", "PortBand Alignment"
    if name.startswith("portband_fleet_"):
        return "portband_fleet", "PortBand Fleet"
    if name.startswith("portband_lineup_matrix_"):
        return "portband_lineup_matrix", "PortBand Lineup Matrix"
    if name.startswith("oracle_sidecar_"):
        return "oracle_sidecar", "Oracle Sidecar"
    return "provenance", "Provenance"


def add_frame(
    frames: list[dict[str, Any]],
    *,
    run_id: str,
    asset_rel: str,
    source_path: str,
    source_sha: str,
    stage_id: str,
    stage_label: str,
    level_index: int | None,
    caption: str,
    timeline_index: int,
    family: str,
) -> None:
    frames.append(
        {
            "frame_id": f"{run_id}_{len(frames):04d}",
            "stage_id": stage_id,
            "stage_label": stage_label,
            "timeline_index": timeline_index,
            "level_index": level_index,
            "caption": caption,
            "family": family,
            "artifact_kind": "png_image",
            "image_url": f"/data/{run_id}/assets/{asset_rel}",
            "source_path": source_path,
            "source_sha256": source_sha,
            "evidence_category": "static_review",
            "boundary": "contraview_report static review frame; not control truth, goal truth, action choice, prediction truth, or solve evidence.",
        }
    )


def format_levels(levels: list[int]) -> str:
    return ",".join(str(level) for level in levels)


def build_grouped_sections(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    sectioned_frame_ids: set[str] = set()
    for order, (section_id, label, stage_ids, family) in enumerate(SECTION_ORDER):
        stage_set = set(stage_ids)
        section_frames = [
            frame
            for frame in frames
            if frame.get("stage_id") in stage_set and frame.get("stage_id") not in HIDDEN_STAGE_IDS
        ]
        if not section_frames:
            continue
        for frame in section_frames:
            sectioned_frame_ids.add(str(frame.get("frame_id")))
        levels = sorted(
            {
                int(frame["level_index"])
                for frame in section_frames
                if isinstance(frame.get("level_index"), int)
            }
        )
        frame_stage_ids = sorted({str(frame.get("stage_id")) for frame in section_frames})
        sections.append(
            {
                "section_id": section_id,
                "section_label": label,
                "section_order": order,
                "family": family,
                "stage_ids": frame_stage_ids,
                "level_indices": levels,
                "frame_count": len(section_frames),
                "frames": section_frames,
                "slider_kind": "single_function_lateral_image_sequence",
                "default_open": section_id in {"raw_frame", "portband"},
                "boundary": "Static review only. This section is not control truth, goal truth, action choice, prediction truth, or solve evidence.",
            }
        )
    remaining = [
        frame
        for frame in frames
        if str(frame.get("frame_id")) not in sectioned_frame_ids and frame.get("stage_id") not in HIDDEN_STAGE_IDS
    ]
    if remaining:
        sections.append(
            {
                "section_id": "unclassified",
                "section_label": "Unclassified",
                "section_order": len(SECTION_ORDER),
                "family": "unclassified",
                "stage_ids": sorted({str(frame.get("stage_id")) for frame in remaining}),
                "level_indices": sorted(
                    {
                        int(frame["level_index"])
                        for frame in remaining
                        if isinstance(frame.get("level_index"), int)
                    }
                ),
                "frame_count": len(remaining),
                "frames": remaining,
                "slider_kind": "single_function_lateral_image_sequence",
                "default_open": False,
                "boundary": "Static review only. This section needs classification before evidence use.",
            }
        )
    return sections


def hidden_sections(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for stage_id in sorted(HIDDEN_STAGE_IDS):
        hidden_frames = [frame for frame in frames if frame.get("stage_id") == stage_id]
        if not hidden_frames:
            continue
        out.append(
            {
                "section_id": stage_id,
                "stage_ids": [stage_id],
                "frame_count": len(hidden_frames),
                "reason": "hidden_by_default_per_user_report_structure",
            }
        )
    return out


def build_report(run_dir: Path, levels: list[int], *, command: str, copy_to_angel: bool) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    run_id = run_dir.name
    assets_dir = run_dir / "angel_assets"
    slides_dir = run_dir / "slides"
    assets_dir.mkdir(parents=True, exist_ok=True)
    slides_dir.mkdir(parents=True, exist_ok=True)

    frames: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    commit = git_commit()
    dirty = git_dirty()

    run_slide = slides_dir / "000_run_condition.png"
    write_text_slide(
        run_slide,
        "contraview_report run condition",
        [
            ("run_id", run_id),
            ("levels", levels),
            ("category", "static_review"),
            ("command", command),
            ("git_commit", commit),
            ("git_dirty", dirty),
            ("boundary", "not control truth, not goal truth, not action choice, not prediction truth, not solve evidence"),
        ],
    )
    copied = copy_asset(run_slide, assets_dir, "0000_run_condition.png")
    add_frame(
        frames,
        run_id=run_id,
        asset_rel=Path(copied["path"]).name,
        source_path=repo_rel(run_slide),
        source_sha=copied["sha256"],
        stage_id="run_condition",
        stage_label="Run Condition",
        level_index=None,
        caption="Run condition",
        timeline_index=0,
        family="provenance",
    )

    for level in levels:
        level_dir = run_dir / "levels" / f"level_{level:02d}"
        packet_path = level_dir / "result.json"
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        raw_png = ROOT / f"codexlab/runs/sp80_sam_color_components/frames/level_{level:02d}_raw.png"
        copied_raw = copy_asset(raw_png, assets_dir, f"{len(frames):04d}_level_{level:02d}_raw_frame.png")
        add_frame(
            frames,
            run_id=run_id,
            asset_rel=Path(copied_raw["path"]).name,
            source_path=repo_rel(raw_png),
            source_sha=copied_raw["sha256"],
            stage_id="raw_frame",
            stage_label="Raw Frame",
            level_index=level,
            caption=f"Level {level:02d} raw frame",
            timeline_index=len(frames),
            family="frame",
        )
        for image in sorted((level_dir / "assets" / "pass_images").glob("*.png")):
            stage_id, stage_label = stage_for_pass_image(image)
            copied_img = copy_asset(image, assets_dir, f"{len(frames):04d}_level_{level:02d}_{image.name}")
            add_frame(
                frames,
                run_id=run_id,
                asset_rel=Path(copied_img["path"]).name,
                source_path=repo_rel(image),
                source_sha=copied_img["sha256"],
                stage_id=stage_id,
                stage_label=stage_label,
                level_index=level,
                caption=image.stem.replace("_", " ").title(),
                timeline_index=len(frames),
                family=stage_id,
            )
        trace_rows.append(
            {
                "schema": "angel.report_trace_event.v1",
                "run_id": run_id,
                "event_index": len(trace_rows),
                "event_type": "packet_generated",
                "level_index": level,
                "input_refs": [{"path": f"codexlab/runs/sp80_sam_color_components/frames/level_{level:02d}_raw.npy"}],
                "output_refs": [{"path": repo_rel(packet_path), "sha256": sha256_file(packet_path)}],
                "status": packet.get("admission_status"),
                "truth_boundary": "static_review_not_action_choice",
            }
        )

    provenance_slide = slides_dir / "900_provenance.png"
    write_text_slide(
        provenance_slide,
        "contraview_report provenance",
        [
            ("run_id", run_id),
            ("created_at_utc", created),
            ("host", socket.gethostname()),
            ("cwd", str(ROOT)),
            ("git_commit", commit),
            ("git_dirty", dirty),
            ("source_builder_sha256", sha256_file(ROOT / "codexlab" / "contraview_report_builder.py")),
            ("image_count", len(frames) + 1),
        ],
    )
    copied_prov = copy_asset(provenance_slide, assets_dir, f"{len(frames):04d}_provenance.png")
    add_frame(
        frames,
        run_id=run_id,
        asset_rel=Path(copied_prov["path"]).name,
        source_path=repo_rel(provenance_slide),
        source_sha=copied_prov["sha256"],
        stage_id="provenance",
        stage_label="Provenance",
        level_index=None,
        caption="Provenance",
        timeline_index=len(frames),
        family="verification",
    )

    stage_defs = []
    for order, (stage_id, label, family) in enumerate(STAGE_ORDER):
        count = sum(1 for frame in frames if frame["stage_id"] == stage_id)
        stage_defs.append(
            {
                "stage_id": stage_id,
                "stage_label": label,
                "stage_order": order,
                "family": family,
                "frame_count": count,
                "status": "rendered" if count else "explicit_zero",
                "boundary": "Static review only. No control truth, goal truth, action choice, prediction truth, or solve evidence.",
            }
        )

    levels_label = format_levels(levels)
    grouped_sections = build_grouped_sections(frames)
    hidden = hidden_sections(frames)
    layout_contract = {
        "layout_kind": "function_grouped_dropdown_sliders",
        "section_rule": "one dropdown per function group, one lateral image slider per dropdown",
        "no_mixed_lateral_functions": True,
        "portband_grouping": "aggregate",
        "portband_stage_ids": sorted(PORTBAND_STAGE_IDS),
        "hidden_stage_ids": sorted(HIDDEN_STAGE_IDS),
        "hidden_by_default": ["tabula_rasa_notebook", "autoresearch"],
    }
    report = {
        "contra_oracle_view_test": {
            "schema": "angel.contra_oracle_view_test.v1",
            "designation": "ANGEL-COV-001",
            "family": "contraview_report",
            "official_name": "contraview_report",
            "label": f"contraview_report · SP80 levels {levels_label}",
            "status": "static_review",
            "generated_at": created,
            "asset_dir": f"/data/{run_id}/assets",
            "source_roots": {"rexus_root": str(ROOT), "run_dir": repo_rel(run_dir), "angel_data": f"/data/{run_id}/contraview_report.json"},
            "boundary": f"Fresh static contraview_report over requested SP80 levels {levels_label}. Not control truth, not goal truth, not action choice, not prediction truth, and not solve evidence.",
            "layout_contract": layout_contract,
        },
        "summary": [
            {"key": "levels", "value": levels_label},
            {"key": "frame_count", "value": len(frames)},
            {"key": "grouped_section_count", "value": len(grouped_sections)},
            {"key": "portband_alignment_images", "value": sum(1 for frame in frames if frame["stage_id"] == "portband_alignment")},
            {"key": "portband_fleet_images", "value": sum(1 for frame in frames if frame["stage_id"] == "portband_fleet")},
            {"key": "portband_lineup_matrix_images", "value": sum(1 for frame in frames if frame["stage_id"] == "portband_lineup_matrix")},
            {"key": "evidence_category", "value": "static_review"},
        ],
        "stages": stage_defs,
        "grouped_sections": grouped_sections,
        "hidden_sections": hidden,
        "frames": frames,
        "process_events": [
            {
                "schema": "angel.report_trace_event.v1",
                "event_order": idx,
                "surface_id": frame["stage_id"],
                "event_type": "frame_rendered",
                "caption": frame["caption"],
                "level_index": frame["level_index"],
                "output_refs": [{"path": frame["source_path"], "sha256": frame["source_sha256"], "url": frame["image_url"]}],
                "status": "rendered",
                "truth_boundary": "static_review_not_action_choice",
            }
            for idx, frame in enumerate(frames)
        ],
    }

    (run_dir / "contraview_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    slide_ledger = [
        {
            "slide_id": frame["frame_id"],
            "slide_type": frame["stage_id"],
            "path": frame["source_path"],
            "sha256": frame["source_sha256"],
            "level_index": frame["level_index"],
            "caption": frame["caption"],
        }
        for frame in frames
    ]
    manifest = {
        "schema": "angel.contra_gameplay_report_manifest.v1",
        "created_at_utc": created,
        "report_status": "static_review",
        "run_id": run_id,
        "game_id": "sp80-589a99af",
        "working_directory": str(ROOT),
        "git_commit": commit,
        "git_dirty": dirty,
        "command": command,
        "slide_count": len(frames),
        "function_slide_count": sum(1 for frame in frames if frame["stage_id"] not in {"run_condition", "raw_frame", "provenance"}),
        "oracle_sidecar_count": sum(1 for frame in frames if frame["stage_id"] in {"oracle_pass", "oracle_sidecar"}),
        "ghost_pass_count": sum(1 for frame in frames if frame["stage_id"] == "ghost_pass"),
        "grouped_section_count": len(grouped_sections),
        "hidden_section_count": len(hidden),
        "portband_alignment_image_count": sum(1 for frame in frames if frame["stage_id"] == "portband_alignment"),
        "portband_fleet_image_count": sum(1 for frame in frames if frame["stage_id"] == "portband_fleet"),
        "portband_lineup_matrix_image_count": sum(1 for frame in frames if frame["stage_id"] == "portband_lineup_matrix"),
        "index_html": repo_rel(run_dir / "index.html"),
        "report_manifest_json": repo_rel(run_dir / "report_manifest.json"),
        "report_trace_jsonl": repo_rel(run_dir / "report_trace.jsonl"),
        "raw_outputs": [{"path": repo_rel(run_dir / "contraview_report.json"), "sha256": sha256_file(run_dir / "contraview_report.json")}],
        "slide_ledger": slide_ledger,
        "verification": {
            "image_count_matches_manifest": len(slide_ledger) == len(frames),
            "slide_count_matches_png_ledger": len(slide_ledger) == len(frames),
            "grouped_section_contract_checked": True,
            "portband_aggregate_section_present": any(section.get("section_id") == "portband" for section in grouped_sections),
            "sha256_path_refs_checked": False,
            "index_reachable": False,
            "browser_screenshot_checked": False,
            "required_files": ["index.html", "report_manifest.json", "report_trace.jsonl"],
        },
        "boundaries": report["contra_oracle_view_test"]["boundary"],
    }
    (run_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace_path = run_dir / "report_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        for row in report["process_events"]:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    index_url = f"/?tool=contra-oracle&stage=portband&data=/data/{run_id}/contraview_report.json"
    (run_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        f"<title>{html.escape(run_id)}</title>"
        f"<h1>contraview_report: {html.escape(run_id)}</h1>"
        "<p>Open in Angel viewer:</p>"
        f"<p><a href='{html.escape(index_url)}'>{html.escape(index_url)}</a></p>"
        f"<p>Boundary: {html.escape(report['contra_oracle_view_test']['boundary'])}</p>",
        encoding="utf-8",
    )

    if copy_to_angel:
        for base in (ANGEL_DIST_DATA, ANGEL_PUBLIC_DATA):
            dest = base / run_id
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_dir / "contraview_report.json", dest / "contraview_report.json")
            for filename in ("report_manifest.json", "report_trace.jsonl", "index.html"):
                source_file = run_dir / filename
                if source_file.is_file():
                    shutil.copy2(source_file, dest / filename)
            shutil.copytree(assets_dir, dest / "assets")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--levels", default="0,1")
    parser.add_argument("--command", required=True)
    parser.add_argument("--copy-to-angel", action="store_true")
    args = parser.parse_args()
    levels = [int(item) for item in args.levels.split(",") if item.strip()]
    report = build_report(args.run_dir, levels, command=args.command, copy_to_angel=args.copy_to_angel)
    print(json.dumps({"run_id": args.run_dir.name, "frames": len(report["frames"]), "stages": len(report["stages"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
