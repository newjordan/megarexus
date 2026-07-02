#!/usr/bin/env python3
"""Exact raw-color board viewer for cross-game spot-check frames."""

from __future__ import annotations

import argparse
import html
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from contra_head import (
    CONTRA_PERCEPTION_PIPELINE_SCHEMA,
    CONTRA_RELATIONSHIP_LADDER_SCHEMA,
    DELTABAND_CONTROLLED_EVENT_REQUIREMENT_SCHEMA,
    DELTABAND_REQUIRED_BOUNDARY,
    DELTABAND_SET_SCHEMA,
    SUPPORT_ACTION_AFFORDANCE_SET_SCHEMA,
    build_deltaband_controlled_event_requirement,
    build_contra_perception_pipeline,
    contra_module_source_paths,
)
from sp80_base_controller import ROOT, sha256_file, sha256_frame
from sp80_color_board_viewer import (
    PRIMARY_COLOR_VALUES,
    rel_for_html,
    save_color_mask,
    save_color_overlay,
    save_level_sheet,
    save_raw_board,
    save_scene_surface_overlay,
    save_scene_surface_sheet,
    value_rgb,
)
from sp80_object_label_viewer import relpath, resolve_repo_path, write_json
from sp80_sam_hypothesis_viewer import raw_color_inventory


MANIFEST_SCHEMA = "codexlab.cross_game_color_board_viewer_manifest.v1"


def empty_actor_affordances(*, scene_surfaces: dict[str, Any], actor_expectation: dict[str, Any] | None) -> list[dict[str, Any]]:
    return []


def empty_target_affordances(
    *,
    frame_shape: tuple[int, int],
    label_map: np.ndarray,
    component_lookup: dict[str, int],
    scene_surfaces: dict[str, Any],
    bucket_expectation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return []


def build_index(manifest: dict[str, Any], *, index_dir: Path) -> str:
    def html_value_list(values: Any) -> str:
        if not isinstance(values, list):
            return ""
        return ", ".join(html.escape(str(value)) for value in values)

    body = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Cross-Game Color Board Viewer</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:20px;background:#f7f7f4;color:#191919}",
        "section{margin-bottom:36px}",
        "img{image-rendering:pixelated;border:1px solid #aaa;background:#111;max-width:100%;height:auto}",
        "table{border-collapse:collapse;font-size:12px;margin-top:10px}",
        "td,th{border:1px solid #ccc;padding:4px 6px;text-align:left;vertical-align:top}",
        ".swatch{display:inline-block;width:18px;height:18px;border:1px solid #333;vertical-align:middle;margin-right:5px}",
        ".row{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}",
        ".tile{max-width:540px}",
        ".surface-detail{margin:12px 0 18px 0}",
        ".surface-detail img{width:min(100%,1152px);max-width:none}",
        ".surface-sheet img{width:min(100%,1600px);max-width:none}",
        ".surface-links{font-size:12px;margin:4px 0 8px 0}",
        "code{background:#eee;padding:1px 3px}",
        "</style></head><body>",
        "<h1>Cross-Game Color Board Viewer</h1>",
        "<p>Exact raw palette isolation from recorded random spot-check raw grids. No SAM, no semantic labels, no routes, no replay.</p>",
        f"<p><b>Source random seed:</b> <code>{manifest.get('source_random_seed')}</code></p>",
        f"<p><b>Manifest:</b> <code>{html.escape(manifest['manifest_json'])}</code></p>",
    ]
    for view in manifest["views"]:
        body.extend(
            [
                f"<section><h2>{html.escape(view['title'])} / {html.escape(view['game_id'])} / level {view['level_index']}</h2>",
                "<div class='surface-detail'>",
                "<h3>Large Scene Surface Overlay</h3>",
                f"<div class='surface-links'><a href='{rel_for_html(view['scene_surface_overlay']['path'], index_dir)}'>open full-size overlay</a></div>",
                f"<img src='{rel_for_html(view['scene_surface_overlay']['path'], index_dir)}'>",
                "</div>",
                "<div class='surface-detail surface-sheet'>",
                "<h3>Large Hypothesis Sheet</h3>",
                f"<div class='surface-links'><a href='{rel_for_html(view['scene_surface_sheet']['path'], index_dir)}'>open full-size sheet</a></div>",
                f"<img src='{rel_for_html(view['scene_surface_sheet']['path'], index_dir)}'>",
                "</div>",
                "<div class='row'>",
                f"<div class='tile'><h3>Board And Primary Separations</h3><img src='{rel_for_html(view['review_sheet']['path'], index_dir)}'></div>",
                f"<div class='tile'><h3>Raw Board</h3><img src='{rel_for_html(view['raw_board']['path'], index_dir)}'></div>",
                "</div>",
            ]
        )
        ladder = view.get("contra_relationship_ladder")
        if isinstance(ladder, dict):
            body.append("<h3>Relationship Ladder</h3>")
            body.append("<table><thead><tr><th>stage</th><th>completeness</th><th>missing</th><th>admission</th></tr></thead><tbody>")
            for stage in ladder.get("stages", []):
                if not isinstance(stage, dict):
                    continue
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(stage.get('stage_name')))}</code></td>"
                    f"<td>{html.escape(str(stage.get('completeness_status')))}</td>"
                    f"<td>{html.escape(', '.join(str(value) for value in stage.get('missing_inputs', [])))}</td>"
                    f"<td>{html.escape(str(stage.get('admission_status')))}</td>"
                    "</tr>"
                )
            body.append("</tbody></table>")
        deltaband_requirement = view.get("deltaband_controlled_event_requirement")
        if isinstance(deltaband_requirement, dict):
            body.extend(
                [
                    "<h3>DeltaBand Controlled Event Gate</h3>",
                    "<table><thead><tr><th>status</th><th>required boundary</th><th>missing artifacts</th><th>admission</th></tr></thead><tbody>",
                    "<tr>",
                    f"<td>{html.escape(str(deltaband_requirement.get('status')))} / {html.escape(str(deltaband_requirement.get('controlled_run_status')))}</td>",
                    f"<td>{html.escape(str(deltaband_requirement.get('required_boundary')))}</td>",
                    f"<td>{html_value_list(deltaband_requirement.get('missing_artifacts'))}</td>",
                    f"<td>{html.escape(str(deltaband_requirement.get('admission_status')))}</td>",
                    "</tr></tbody></table>",
                ]
            )
        body.append("<table><thead><tr><th>role</th><th>component</th><th>raw</th><th>score</th><th>facing</th><th>asym</th><th>reasons</th></tr></thead><tbody>")
        floor = view["scene_surface_hypotheses"].get("floor_hypothesis")
        surface_rows = []
        if isinstance(floor, dict):
            surface_rows.append(("floor", floor))
        surface_rows.extend(("wall", row) for row in view["scene_surface_hypotheses"].get("wall_hypotheses", []))
        surface_rows.extend(("background", row) for row in view["scene_surface_hypotheses"].get("background_hypotheses", []))
        surface_rows.extend(
            ("residual", row) for row in view["scene_surface_hypotheses"].get("residual_component_hypotheses", [])[:20]
        )
        for role, row in surface_rows:
            asymmetry = row.get("asymmetry") if isinstance(row.get("asymmetry"), dict) else {}
            facing = row.get("facing_direction_hypothesis") or asymmetry.get("facing_direction_hypothesis") or "none"
            confidence = row.get("facing_direction_confidence")
            if confidence is None:
                confidence = asymmetry.get("confidence")
            body.append(
                "<tr>"
                f"<td>{html.escape(role)}</td>"
                f"<td><code>{html.escape(str(row['component_id']))}</code></td>"
                f"<td>{row['raw_pixel_value']}</td>"
                f"<td>{float(row['score']):.3f}</td>"
                f"<td>{html.escape(str(facing))} {float(confidence or 0.0):.2f}</td>"
                f"<td>x={float(asymmetry.get('mirror_x_difference') or 0.0):.2f} y={float(asymmetry.get('mirror_y_difference') or 0.0):.2f}</td>"
                f"<td>{html.escape(', '.join(row.get('reasons') or []))}</td>"
                "</tr>"
            )
        relationships = view["scene_surface_hypotheses"].get("antisymmetric_relationship_hypotheses", [])
        if relationships:
            body.extend(
                [
                    "</tbody></table>",
                    "<table><thead><tr><th>antisymmetric pair</th><th>axis</th><th>facing</th><th>score</th><th>reasons</th></tr></thead><tbody>",
                ]
            )
            for relationship in relationships[:16]:
                body.append(
                    "<tr>"
                    f"<td><code>{html.escape(str(relationship['component_a']))}</code> / <code>{html.escape(str(relationship['component_b']))}</code></td>"
                    f"<td>{html.escape(str(relationship['axis']))}</td>"
                    f"<td>{html.escape(str(relationship['facing_direction_a']))} / {html.escape(str(relationship['facing_direction_b']))}</td>"
                    f"<td>{float(relationship['score']):.3f}</td>"
                    f"<td>{html.escape(', '.join(relationship.get('reasons') or []))}</td>"
                    "</tr>"
                )
        body.extend(
            [
                "</tbody></table>",
                "<table><thead><tr><th>raw</th><th>color</th><th>pixels</th><th>components</th><th>background</th><th>mask</th><th>overlay</th></tr></thead><tbody>",
            ]
        )
        for color in view["colors"]:
            rgb = value_rgb(int(color["raw_pixel_value"]))
            swatch = f"<span class='swatch' style='background:rgb({rgb[0]},{rgb[1]},{rgb[2]})'></span>"
            body.append(
                "<tr>"
                f"<td>{color['raw_pixel_value']}</td>"
                f"<td>{swatch}{html.escape(color['palette_name'])}</td>"
                f"<td>{color['pixel_count']}</td>"
                f"<td>{color['component_count']}</td>"
                f"<td>{str(color['dominant_background_candidate']).lower()}</td>"
                f"<td><a href='{rel_for_html(color['mask_png']['path'], index_dir)}'>mask</a></td>"
                f"<td><a href='{rel_for_html(color['overlay_png']['path'], index_dir)}'>overlay</a></td>"
                "</tr>"
            )
        body.extend(["</tbody></table></section>"])
    body.extend(["</body></html>"])
    return "\n".join(body)


def safe_part(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("codexlab/runs/cross_game_isolation_spotcheck/manifest.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/cross_game_color_board_viewer"))
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--min-component-area", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_manifest_path = resolve_repo_path(args.source_manifest)
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scale = max(1, int(args.scale))
    min_component_area = max(1, int(args.min_component_area))
    views: list[dict[str, Any]] = []
    for row in source_manifest.get("rows") or []:
        raw_grid = ((row.get("raw_frame") or {}).get("raw_grid") or {})
        raw_path = resolve_repo_path(raw_grid["path"])
        frame = np.load(raw_path, allow_pickle=False)
        game_short = safe_part(str(row["game_short"]))
        level_index = int(row["level_index"])
        sample_id = f"{game_short}_level_{level_index:02d}"
        inventory = raw_color_inventory(frame, min_component_area=min_component_area)
        source_raw_frame_sha256 = sha256_frame(frame)
        perception = build_contra_perception_pipeline(
            frame=frame,
            level_index=level_index,
            source_raw_frame_sha256=source_raw_frame_sha256,
            raw_color_inventory=inventory,
            calibration=None,
            actor_builder=empty_actor_affordances,
            target_builder=empty_target_affordances,
        )
        scene_surfaces = perception["scene_surface_hypotheses"]
        component_label_map = perception["component_label_map"]
        component_lookup = perception["component_lookup"]
        actor_affordances = perception["actor_affordance_hypotheses"]
        target_affordances = perception["target_affordance_hypotheses"]
        support_affordances = perception["support_action_affordance_hypotheses"]
        statecons = perception["state_condition_hypotheses"]
        relationship_ladder = perception["contra_relationship_ladder"]
        deltaband_controlled_event_requirement = build_deltaband_controlled_event_requirement(
            level_index=level_index,
            source_raw_frame_sha256=source_raw_frame_sha256,
            relationship_ladder=relationship_ladder,
        )
        raw_board = save_raw_board(frame, out_dir / "boards" / f"{sample_id}_raw_board.png", scale)
        scene_surface_overlay = save_scene_surface_overlay(
            frame,
            scene_surfaces=scene_surfaces,
            label_map=component_label_map,
            component_lookup=component_lookup,
            target_affordances=target_affordances,
            actor_affordances=actor_affordances,
            statecons=statecons,
            path=out_dir / "scene_surface_overlays" / f"{sample_id}_surface_hypotheses.png",
            scale=scale,
        )
        scene_surface_sheet = save_scene_surface_sheet(
            frame=frame,
            scene_surfaces=scene_surfaces,
            label_map=component_label_map,
            component_lookup=component_lookup,
            target_affordances=target_affordances,
            actor_affordances=actor_affordances,
            statecons=statecons,
            path=out_dir / "scene_surface_sheets" / f"{sample_id}_surface_hypotheses.png",
            scale=scale,
        )
        colors: list[dict[str, Any]] = []
        for color in inventory["colors"]:
            value = int(color["raw_pixel_value"])
            components = list(color.get("components") or [])
            color_name = safe_part(str(color["palette_name"]))
            mask_png = save_color_mask(
                frame,
                value=value,
                components=components,
                path=out_dir / "color_masks" / f"{sample_id}_v{value:02d}_{color_name}.png",
                scale=scale,
            )
            overlay_png = save_color_overlay(
                frame,
                value=value,
                components=components,
                path=out_dir / "color_overlays" / f"{sample_id}_v{value:02d}_{color_name}.png",
                scale=scale,
            )
            colors.append({**color, "mask_png": mask_png, "overlay_png": overlay_png})
        review_sheet = save_level_sheet(
            frame=frame,
            colors=colors,
            path=out_dir / "review_sheets" / f"{sample_id}_color_board.png",
            scale=scale,
        )
        views.append(
            {
                "sample_id": sample_id,
                "game_id": row["game_id"],
                "game_short": row["game_short"],
                "title": row["title"],
                "level_index": level_index,
                "source_raw_grid": relpath(raw_path),
                "source_raw_grid_sha256": sha256_file(raw_path),
                "source_raw_frame_sha256": source_raw_frame_sha256,
                "frame_shape": [int(v) for v in frame.shape],
                "raw_board": raw_board,
                "scene_surface_overlay": scene_surface_overlay,
                "scene_surface_sheet": scene_surface_sheet,
                "scene_surface_hypotheses": scene_surfaces,
                "actor_affordance_hypotheses": actor_affordances,
                "target_affordance_hypotheses": target_affordances,
                "statecon_support_affordances": support_affordances,
                "manual_support_affordances": perception["manual_support_affordances"],
                "candidate_support_affordances": perception["candidate_support_affordances"],
                "support_action_affordance_hypotheses": support_affordances,
                "state_condition_hypotheses": statecons,
                "contra_relationship_ladder": relationship_ladder,
                "deltaband_controlled_event_requirement": deltaband_controlled_event_requirement,
                "contra_perception_pipeline": {
                    key: value
                    for key, value in perception.items()
                    if key not in {"component_label_map", "component_lookup"}
                },
                "review_sheet": review_sheet,
                "raw_color_inventory": inventory,
                "colors": colors,
            }
        )
    manifest_path = out_dir / "manifest.json"
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "condition_name": "cross_game_exact_color_board_viewer",
        "condition_natural_language": "Render exact raw color boards and per-color separations from recorded random cross-game raw frame grids.",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "source_manifest_json": relpath(source_manifest_path),
        "source_manifest_json_sha256": sha256_file(source_manifest_path),
        "source_random_seed": source_manifest.get("random_seed"),
        "scale": scale,
        "min_component_area_pixels": min_component_area,
        "primary_color_values": sorted(PRIMARY_COLOR_VALUES),
        "scene_surface_hypotheses_schema": "codexlab.scene_surface_hypotheses.v3",
        "scene_surface_hypotheses_status": "hypothesis_not_admitted",
        "contra_perception_pipeline_schema": CONTRA_PERCEPTION_PIPELINE_SCHEMA,
        "contra_perception_pipeline_status": "unified_hypothesis_pipeline_not_admitted",
        "support_action_affordance_hypotheses_schema": SUPPORT_ACTION_AFFORDANCE_SET_SCHEMA,
        "support_action_affordance_hypotheses_status": "heuristic_hypothesis_not_admitted",
        "contra_relationship_ladder_schema": CONTRA_RELATIONSHIP_LADDER_SCHEMA,
        "contra_relationship_ladder_status": "hypothesis_admission_ladder_not_admitted",
        "deltaband_observations_schema": DELTABAND_SET_SCHEMA,
        "deltaband_observations_status": "not_run_requires_controlled_before_after_action_boundary",
        "deltaband_required_boundary": DELTABAND_REQUIRED_BOUNDARY,
        "deltaband_prediction_status": "not_predictive",
        "deltaband_controlled_event_requirement_schema": DELTABAND_CONTROLLED_EVENT_REQUIREMENT_SCHEMA,
        "deltaband_controlled_event_requirement_status": "not_run_requires_controlled_before_after_action_boundary",
        "view_count": len(views),
        "views": views,
        "sam_status": "not_run",
        "semantic_labels_used": False,
        "controlled_labels_used": False,
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
        "source_script": "codexlab/cross_game_color_board_viewer.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab" / "cross_game_color_board_viewer.py"),
        "source_contra": "codexlab/contra_head/contra.py",
        "source_contra_sha256": sha256_file(ROOT / "codexlab" / "contra_head" / "contra.py"),
        "source_contra_modules": [
            {"path": path, "sha256": sha256_file(ROOT / path)}
            for path in contra_module_source_paths()
        ],
    }
    manifest["manifest_json"] = relpath(manifest_path)
    write_json(manifest_path, manifest)
    index_path = out_dir / "index.html"
    index_path.write_text(build_index(manifest, index_dir=out_dir), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "exported",
                "manifest": relpath(manifest_path),
                "index_html": relpath(index_path),
                "view_count": len(views),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
