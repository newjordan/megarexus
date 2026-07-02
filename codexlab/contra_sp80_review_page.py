#!/usr/bin/env python3
"""Build a human review page for a single SP80 CONTRA paint packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from review_helpers import code, compact_json, h, rel_for_html, repo_rel


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PACKET = ROOT / "codexlab" / "runs" / "contra_sp80_level_00_latest" / "result.json"
DEFAULT_OUT_DIR = ROOT / "codexlab" / "runs" / "contra_sp80_level_00_latest"
DEFAULT_RAW_GRID = ROOT / "codexlab" / "runs" / "sp80_sam_color_components" / "frames" / "level_00_raw.npy"
DEFAULT_SOURCE_COMMAND = (
    "python3 codexlab/contra_api_smoke.py "
    "--raw-grid codexlab/runs/sp80_sam_color_components/frames/level_00_raw.npy "
    "--game-id sp80-589a99af --level-index 0 --mode review "
    "--calibration codexlab/sp80_surface_calibration.json "
    "--out codexlab/runs/contra_sp80_level_00_latest/result.json"
)


def raw_grid_path_for_packet(packet: dict[str, Any]) -> Path:
    level_index = int(packet.get("level_index") or 0)
    return ROOT / "codexlab" / "runs" / "sp80_sam_color_components" / "frames" / f"level_{level_index:02d}_raw.npy"


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def value_list(value: Any, *, limit: int | None = None) -> str:
    if value is None:
        return ""
    if not isinstance(value, list):
        return code(value)
    items = value[:limit] if limit is not None else value
    rendered = " ".join(code(item) for item in items)
    if limit is not None and len(value) > limit:
        rendered += f" <span class='muted'>+{len(value) - limit} more</span>"
    return rendered


def packet_counts(packet: dict[str, Any]) -> dict[str, Any]:
    return ((packet.get("debug_summary") or {}).get("counts") or {}) if isinstance(packet.get("debug_summary"), dict) else {}


RAW_COLORS = {
    0: (255, 255, 255),
    1: (190, 190, 190),
    2: (128, 128, 128),
    3: (80, 80, 80),
    4: (51, 51, 51),
    5: (0, 0, 0),
    6: (210, 55, 165),
    8: (255, 70, 70),
    9: (30, 147, 255),
    10: (136, 216, 241),
    11: (255, 220, 0),
    12: (255, 132, 25),
    13: (128, 0, 0),
    14: (40, 205, 35),
    15: (145, 70, 255),
}


PASS_COLORS = [
    (74, 222, 128),
    (250, 204, 21),
    (125, 211, 252),
    (251, 113, 133),
    (210, 90, 255),
    (45, 160, 255),
]


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def raw_image(raw_grid: np.ndarray, *, scale: int = 12) -> Image.Image:
    rgb = np.zeros((*raw_grid.shape, 3), dtype=np.uint8)
    for value in np.unique(raw_grid):
        rgb[raw_grid == int(value)] = RAW_COLORS.get(int(value), (180, 80, 180))
    image = Image.fromarray(rgb, "RGB")
    return image.resize((raw_grid.shape[1] * scale, raw_grid.shape[0] * scale), Image.Resampling.NEAREST)


def bbox_tuple(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return tuple(int(round(float(v))) for v in value[:4])  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def center_tuple(value: Any, *, scale: int) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return int((float(value[0]) + 0.5) * scale), int((float(value[1]) + 0.5) * scale)
    except (TypeError, ValueError):
        return None


def coordinate_tuple(value: Any, *, scale: int) -> tuple[int, int] | None:
    if not isinstance(value, list) or len(value) < 2:
        return None
    try:
        return int(round(float(value[0]) * scale)), int(round(float(value[1]) * scale))
    except (TypeError, ValueError):
        return None


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: Any, *, scale: int, color: tuple[int, int, int], label: str | None = None) -> None:
    box = bbox_tuple(bbox)
    if box is None:
        return
    x0, y0, x1, y1 = box
    scaled = (x0 * scale, y0 * scale, (x1 + 1) * scale - 1, (y1 + 1) * scale - 1)
    draw.rectangle(scaled, outline=color, width=4)
    if label:
        f = font(16, bold=True)
        tx, ty = scaled[0] + 5, scaled[1] + 5
        tw = int(draw.textlength(label, font=f))
        draw.rectangle((tx - 2, ty - 2, tx + tw + 7, ty + 21), fill=(5, 7, 10), outline=color)
        draw.text((tx + 2, ty), label, fill=color, font=f)


def draw_port_marker(draw: ImageDraw.ImageDraw, point: Any, *, scale: int, color: tuple[int, int, int], label: str) -> None:
    center = coordinate_tuple(point, scale=scale)
    if center is None:
        return
    x, y = center
    r = 7
    draw.ellipse((x - r, y - r, x + r, y + r), outline=(*color, 255), width=4)
    draw.line((x - r - 3, y, x + r + 3, y), fill=(*color, 235), width=3)
    draw.line((x, y - r - 3, x, y + r + 3), fill=(*color, 235), width=3)
    f = font(15, bold=True)
    tx, ty = x + 9, max(0, y - 20)
    tw = int(draw.textlength(label, font=f))
    draw.rectangle((tx - 2, ty - 2, tx + tw + 6, ty + 19), fill=(5, 7, 10, 230), outline=(*color, 255))
    draw.text((tx + 2, ty), label, fill=(*color, 255), font=f)


def shifted_points(points: Any, shift: Any) -> list[list[float]]:
    if not isinstance(points, list) or not isinstance(shift, list) or len(shift) < 2:
        return []
    try:
        dx = float(shift[0])
        dy = float(shift[1])
    except (TypeError, ValueError):
        return []
    out: list[list[float]] = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            out.append([float(point[0]) + dx, float(point[1]) + dy])
        except (TypeError, ValueError):
            continue
    return out


def wrap_text(text: str, *, width: int = 62) -> list[str]:
    words = str(text).replace("\n", " ").split()
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        extra = 1 if current else 0
        if current and current_len + extra + len(word) > width:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra + len(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def pass_card(
    *,
    raw_grid: np.ndarray,
    title: str,
    subtitle: str,
    rows: list[tuple[str, Any]],
    out_path: Path,
    bboxes: list[tuple[Any, str | None, tuple[int, int, int]]] | None = None,
    edges: list[tuple[Any, Any, tuple[int, int, int]]] | None = None,
    coordinate_edges: list[tuple[Any, Any, tuple[int, int, int]]] | None = None,
    port_markers: list[tuple[Any, str, tuple[int, int, int]]] | None = None,
) -> dict[str, Any]:
    scale = 12
    board = raw_image(raw_grid, scale=scale).convert("RGBA")
    overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for start, end, color in edges or []:
        a = center_tuple(start, scale=scale)
        b = center_tuple(end, scale=scale)
        if a is not None and b is not None:
            odraw.line((a[0], a[1], b[0], b[1]), fill=(*color, 220), width=5)
    for start, end, color in coordinate_edges or []:
        a = coordinate_tuple(start, scale=scale)
        b = coordinate_tuple(end, scale=scale)
        if a is not None and b is not None:
            odraw.line((a[0], a[1], b[0], b[1]), fill=(*color, 235), width=5)
    for idx, (bbox, label, color) in enumerate(bboxes or []):
        draw_bbox(odraw, bbox, scale=scale, color=color, label=label)
    for point, label, color in port_markers or []:
        draw_port_marker(odraw, point, scale=scale, color=color, label=label)
    board.alpha_composite(overlay)

    canvas = Image.new("RGB", (1600, 930), (8, 12, 17))
    draw = ImageDraw.Draw(canvas)
    title_font = font(34, bold=True)
    sub_font = font(20)
    body_font = font(20)
    small_font = font(16)
    draw.text((30, 24), title, fill=(248, 250, 252), font=title_font)
    draw.text((30, 68), subtitle, fill=(181, 192, 206), font=sub_font)
    canvas.paste(board.convert("RGB"), (30, 120))
    draw.rectangle((30, 120, 30 + board.width, 120 + board.height), outline=(67, 82, 102), width=2)
    x = 840
    y = 120
    draw.text((x, y), "Calculation Result", fill=(248, 250, 252), font=font(26, bold=True))
    y += 42
    for key, value in rows:
        line = f"{key}: {value}"
        for wrapped in wrap_text(line, width=58):
            draw.text((x, y), wrapped, fill=(219, 234, 254), font=body_font)
            y += 27
            if y > 870:
                draw.text((x, y), "... truncated on image; full packet table remains below", fill=(251, 191, 36), font=small_font)
                break
        y += 6
        if y > 870:
            break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return {"path": repo_rel(out_path), "sha256": sha256_file(out_path), "title": title, "subtitle": subtitle, "row_count": len(rows)}


def components_by_id(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for color in (packet.get("raw_color_component_packet") or {}).get("colors", []):
        if not isinstance(color, dict):
            continue
        for component in color.get("components", []):
            if isinstance(component, dict) and component.get("component_id"):
                out[str(component["component_id"])] = component
    return out


def object_bbox_lookup(packet: dict[str, Any]) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for row in packet.get("actor_affordance_hypotheses", []) or []:
        if isinstance(row, dict):
            lookup[str(row.get("actor_id"))] = row.get("group_bbox_xyxy") or row.get("actor_bbox_xyxy")
            lookup[str(row.get("actor_component_id"))] = row.get("actor_bbox_xyxy")
            lookup[str(row.get("orientation_key_component_id"))] = row.get("orientation_key_bbox_xyxy")
    for row in packet.get("target_affordance_hypotheses", []) or []:
        if isinstance(row, dict):
            lookup[str(row.get("target_id"))] = row.get("target_bbox_xyxy")
            lookup[str(row.get("component_id"))] = row.get("target_bbox_xyxy")
    for row in packet.get("support_action_affordance_hypotheses", []) or []:
        if isinstance(row, dict):
            lookup[str(row.get("support_id"))] = row.get("bbox_xyxy")
            lookup[str(row.get("component_id"))] = row.get("bbox_xyxy")
    for statecon in packet.get("state_condition_hypotheses", []) or []:
        if not isinstance(statecon, dict):
            continue
        lookup[str(statecon.get("statecon_id"))] = statecon.get("group_bbox_xyxy")
        for group in statecon.get("support_groups", []) or []:
            if isinstance(group, dict):
                lookup[str(group.get("support_group_id"))] = group.get("group_bbox_xyxy")
    for component_id, component in components_by_id(packet).items():
        lookup.setdefault(component_id, component.get("bbox_xyxy"))
    return lookup


def symmetry_rows(symmetry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, group_name in (
        ("center_divider_hypotheses", "center_divider"),
        ("playzone_mirror_hypotheses", "playzone_mirror"),
        ("component_self_symmetry_hypotheses", "self_symmetry"),
        ("component_mirror_pair_hypotheses", "mirror_pair"),
        ("component_reverse_symmetry_hypotheses", "reverse_symmetry"),
        ("component_polar_symmetry_hypotheses", "polar_symmetry"),
        ("lock_key_sensor_hypotheses", "lock_key_sensor"),
        ("surface_symmetry_hypotheses", "surface_symmetry"),
    ):
        values = symmetry.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            item["_group_name"] = group_name
            rows.append(item)
    return rows


def rows_for_output(row: dict[str, Any]) -> list[tuple[str, Any]]:
    return [
        ("layer", row.get("layer_id")),
        ("status", row.get("status")),
        ("data_ref", row.get("data_ref")),
        ("computed", row.get("computed")),
        ("skipped", row.get("skipped")),
        ("counts", compact_json(row.get("counts", {}))),
        ("missing_inputs", ", ".join(str(v) for v in row.get("missing_inputs", []) or [])),
        ("reason_codes", ", ".join(str(v) for v in row.get("reason_codes", []) or [])),
    ]


def rows_for_view_system_pass(packet: dict[str, Any], pass_id: str) -> list[tuple[str, Any]]:
    row = packet.get(pass_id) if isinstance(packet.get(pass_id), dict) else {}
    if pass_id == "config_pass":
        return [
            ("schema", row.get("schema")),
            ("status", row.get("status")),
            ("first_click_allowance_status", row.get("first_click_allowance_status")),
            ("first_click_scope", row.get("first_click_scope")),
            ("first_loading_screen_detected", row.get("first_loading_screen_detected")),
            ("first_loading_screen_detector", row.get("first_loading_screen_detector")),
            ("selected_click_xy", row.get("selected_click_xy")),
            ("click_invocation_id", row.get("click_invocation_id")),
            ("accepted_screen_click", row.get("accepted_screen_click")),
            ("post_click_frame_ref", row.get("post_click_frame_ref")),
            ("error_message_ref", row.get("error_message_ref")),
            ("config_evidence_status", row.get("config_evidence_status")),
            ("evidence_role", row.get("evidence_role")),
        ]
    if pass_id == "oracle_pass":
        return [
            ("schema", row.get("schema")),
            ("status", row.get("status")),
            ("hypothesis_count", row.get("hypothesis_count")),
            ("actor_hypothesis_count", row.get("actor_hypothesis_count")),
            ("target_hypothesis_count", row.get("target_hypothesis_count")),
            ("reason_codes", ", ".join(str(v) for v in row.get("reason_codes", []) or [])),
            ("oracle_evidence_status", row.get("oracle_evidence_status")),
            ("evidence_role", row.get("evidence_role")),
        ]
    if pass_id == "ghost_pass":
        return [
            ("schema", row.get("schema")),
            ("status", row.get("status")),
            ("ghost_count", row.get("ghost_count")),
            ("relation_memory_row_count", row.get("relation_memory_row_count")),
            ("reason_codes", ", ".join(str(v) for v in row.get("reason_codes", []) or [])),
            ("evidence_boundary", row.get("evidence_boundary")),
            ("evidence_role", row.get("evidence_role")),
            ("route_or_replay_used", row.get("route_or_replay_used")),
            ("coordinate_copy_used", row.get("coordinate_copy_used")),
        ]
    return [
        ("schema", row.get("schema")),
        ("status", row.get("status")),
        ("layer_output_count", row.get("layer_output_count")),
        ("computed_output_count", row.get("computed_output_count")),
        ("skipped_output_count", row.get("skipped_output_count")),
        ("relationship_ladder_status", row.get("relationship_ladder_status")),
        ("deltaband_requirement_status", row.get("deltaband_requirement_status")),
        ("admission_status", row.get("admission_status")),
        ("config_pass_status", row.get("config_pass_status")),
        ("oracle_pass_status", row.get("oracle_pass_status")),
        ("ghost_pass_status", row.get("ghost_pass_status")),
        ("feedback_signals", compact_json(row.get("feedback_signals", {}))),
        ("evidence_role", row.get("evidence_role")),
    ]


def bboxes_for_layer(packet: dict[str, Any], layer_id: str) -> list[tuple[Any, str, tuple[int, int, int]]]:
    out: list[tuple[Any, str, tuple[int, int, int]]] = []
    color_idx = 0
    if layer_id == "raw_color_components":
        for component_id, component in components_by_id(packet).items():
            out.append((component.get("bbox_xyxy"), component_id, PASS_COLORS[color_idx % len(PASS_COLORS)]))
            color_idx += 1
    elif layer_id == "control_isolation":
        traces = (packet.get("control_isolation_packet") or {}).get("closed_shape_traces") or []
        for idx, trace in enumerate(traces):
            if isinstance(trace, dict):
                out.append((trace.get("bbox_xyxy"), str(trace.get("shape_id", f"shape_{idx}")), PASS_COLORS[idx % len(PASS_COLORS)]))
    elif layer_id == "scene_surfaces":
        surfaces = packet.get("scene_surfaces") if isinstance(packet.get("scene_surfaces"), dict) else {}
        for role, rows, color in (
            ("BG", surfaces.get("background_hypotheses") or [], (45, 125, 255)),
            ("FLOOR", [surfaces.get("floor_hypothesis")] if isinstance(surfaces.get("floor_hypothesis"), dict) else [], (30, 210, 105)),
            ("WALL", surfaces.get("wall_hypotheses") or [], (245, 145, 35)),
            ("RES", surfaces.get("residual_component_hypotheses") or [], (200, 200, 200)),
        ):
            for item in rows:
                if isinstance(item, dict):
                    out.append((item.get("bbox_xyxy"), f"{role}:{item.get('component_id')}", color))
    elif layer_id == "symmetry_hypotheses":
        for idx, item in enumerate(symmetry_rows(packet.get("symmetry_hypotheses") or {})):
            out.append((item.get("bbox_xyxy"), str(item.get("symmetry_kind") or item.get("_group_name")), PASS_COLORS[idx % len(PASS_COLORS)]))
    elif layer_id == "actor_affordances":
        for row in packet.get("actor_affordance_hypotheses", []) or []:
            if isinstance(row, dict):
                label = "ACTOR"
                if row.get("actor_option_count"):
                    label = f"A{row.get('actor_option_rank')}/{row.get('actor_option_count')}"
                out.append((row.get("actor_bbox_xyxy"), label, (210, 90, 255)))
                out.append((row.get("orientation_key_bbox_xyxy"), "KEY", (250, 204, 21)))
    elif layer_id == "target_affordances":
        for row in packet.get("target_affordance_hypotheses", []) or []:
            if isinstance(row, dict):
                label = "TARGET"
                if row.get("goal_option_count"):
                    label = f"T{row.get('goal_option_rank')}/{row.get('goal_option_count')}"
                out.append((row.get("target_bbox_xyxy"), label, (74, 222, 128)))
    elif layer_id == "support_action_affordances":
        for row in packet.get("support_action_affordance_hypotheses", []) or []:
            if isinstance(row, dict):
                out.append((row.get("bbox_xyxy"), "SUPPORT", (45, 160, 255)))
    elif layer_id in {"state_condition_hypotheses", "geoband_hypotheses", "geoband_boundary_queues"}:
        lookup = object_bbox_lookup(packet)
        for statecon in packet.get("state_condition_hypotheses", []) or []:
            if not isinstance(statecon, dict):
                continue
            nested = statecon.get("geoband_hypothesis") if isinstance(statecon.get("geoband_hypothesis"), dict) else {}
            members = nested.get("members") if isinstance(nested.get("members"), dict) else {}
            for object_id, label, color in (
                (members.get("actor_affordance_id"), "ACTOR", (210, 90, 255)),
                (members.get("target_affordance_id"), "TARGET", (74, 222, 128)),
            ):
                if object_id:
                    out.append((lookup.get(str(object_id)), label, color))
            for group_id in members.get("support_group_ids", []) or []:
                out.append((lookup.get(str(group_id)), "SUPPORT", (45, 160, 255)))
    return [(bbox, label, color) for bbox, label, color in out if bbox is not None]


def edges_for_geoband(statecon: dict[str, Any]) -> list[tuple[Any, Any, tuple[int, int, int]]]:
    geoband = statecon.get("geoband_hypothesis") if isinstance(statecon.get("geoband_hypothesis"), dict) else {}
    nodes = geoband.get("relationship_nodes") if isinstance(geoband.get("relationship_nodes"), list) else []
    centers = {str(node.get("node_id")): node.get("center_xy") for node in nodes if isinstance(node, dict)}
    out: list[tuple[Any, Any, tuple[int, int, int]]] = []
    for edge in geoband.get("relationship_edges", []) if isinstance(geoband.get("relationship_edges"), list) else []:
        if not isinstance(edge, dict):
            continue
        start = centers.get(str(edge.get("from_node_id")))
        end = centers.get(str(edge.get("to_node_id")))
        if start is not None and end is not None:
            out.append((start, end, (74, 222, 128) if edge.get("edge_kind") == "actor_to_goal_attachment" else (125, 211, 252)))
    return out


def find_symmetry_group(packet: dict[str, Any], group_id: str) -> dict[str, Any] | None:
    for row in symmetry_rows(packet.get("symmetry_hypotheses") or {}):
        if str(row.get("symmetry_group_id")) == str(group_id):
            return row
    return None


def build_pass_review_images(packet: dict[str, Any], *, out_dir: Path, raw_grid_path: Path | None = None) -> list[dict[str, Any]]:
    raw_grid_path = raw_grid_path or raw_grid_path_for_packet(packet)
    raw_grid = np.load(raw_grid_path, allow_pickle=False).astype(int)
    pass_dir = out_dir / "assets" / "pass_images"
    pass_dir.mkdir(parents=True, exist_ok=True)
    for old_png in pass_dir.glob("*.png"):
        old_png.unlink()
    cards: list[dict[str, Any]] = []
    for idx, (pass_id, title, subtitle) in enumerate(
        (
            ("config_pass", "Config pass", "first-click calibration allowance/result; not gameplay truth"),
            ("oracle_pass", "Oracle pass", "review hypotheses or explicit no-hypothesis status"),
            ("ghost_pass", "Ghost pass", "controlled relationship memory status; not future truth or action authority"),
            ("contra_pass", "CONTRA pass", "unified analysis report over Config, Oracle, and CONTRA layer outputs"),
        ),
        1,
    ):
        pass_packet = packet.get(pass_id) if isinstance(packet.get(pass_id), dict) else {}
        markers = []
        if pass_id == "config_pass" and isinstance(pass_packet.get("selected_click_xy"), list):
            markers.append((pass_packet.get("selected_click_xy"), "CLICK?", (250, 204, 21)))
        card = pass_card(
            raw_grid=raw_grid,
            title=f"{title}: {pass_packet.get('status')}",
            subtitle=subtitle,
            rows=rows_for_view_system_pass(packet, pass_id),
            out_path=pass_dir / f"view_system_{idx:02d}_{pass_id}.png",
            port_markers=markers,
        )
        card.update({"pass_family": "view_system", "pass_order": idx, "pass_id": pass_id, "status": pass_packet.get("status")})
        cards.append(card)

    layer_outputs = (packet.get("layer_output_packets") or {}).get("outputs") or []
    for idx, output in enumerate(layer_outputs, 1):
        if not isinstance(output, dict):
            continue
        layer_id = str(output.get("layer_id"))
        edges: list[tuple[Any, Any, tuple[int, int, int]]] = []
        if layer_id == "geoband_hypotheses":
            for statecon in packet.get("state_condition_hypotheses", []) or []:
                if isinstance(statecon, dict):
                    edges.extend(edges_for_geoband(statecon))
        card = pass_card(
            raw_grid=raw_grid,
            title=f"CONTRA pass {idx:02d}: {layer_id}",
            subtitle="top-level layer output from layer_output_packets",
            rows=rows_for_output(output),
            out_path=pass_dir / f"contra_pass_{idx:02d}_{layer_id}.png",
            bboxes=bboxes_for_layer(packet, layer_id),
            edges=edges,
        )
        card.update({"pass_family": "contra_layer", "pass_order": idx, "pass_id": layer_id, "status": output.get("status")})
        cards.append(card)

    oracle_rows: list[tuple[str, dict[str, Any]]] = []
    for row in packet.get("actor_affordance_hypotheses", []) or []:
        if isinstance(row, dict) and str(row.get("source", "")).startswith("oracle_first_pass"):
            oracle_rows.append(("actor", row))
    for row in packet.get("target_affordance_hypotheses", []) or []:
        if isinstance(row, dict) and str(row.get("source", "")).startswith("oracle_first_pass"):
            oracle_rows.append(("target", row))
    for idx, (kind, row) in enumerate(oracle_rows, 1):
        if kind == "actor":
            label = "ORACLE ACTOR"
            if row.get("actor_option_count"):
                label = f"ORACLE A{row.get('actor_option_rank')}/{row.get('actor_option_count')}"
            bboxes = [(row.get("actor_bbox_xyxy"), label, (250, 204, 21))]
            paired = row.get("paired_goal_component_id")
        else:
            label = "ORACLE GOAL"
            if row.get("goal_option_count"):
                label = f"ORACLE T{row.get('goal_option_rank')}/{row.get('goal_option_count')}"
            bboxes = [(row.get("target_bbox_xyxy"), label, (74, 222, 128))]
            paired = row.get("paired_actor_component_id")
        card = pass_card(
            raw_grid=raw_grid,
            title=f"Oracle pass {idx:02d}: {kind}",
            subtitle="Oracle first-pass actor/goal relation; review-only, not admitted",
            rows=[
                ("source", row.get("source")),
                ("id", row.get("actor_id") if kind == "actor" else row.get("target_id")),
                ("component", row.get("actor_component_id") if kind == "actor" else row.get("component_id")),
                ("paired_component", paired),
                ("actor_set_id", row.get("actor_set_id")),
                ("actor_option", f"{row.get('actor_option_rank')}/{row.get('actor_option_count')}" if row.get("actor_option_count") else None),
                ("target_set_id", row.get("target_set_id")),
                ("goal_option", f"{row.get('goal_option_rank')}/{row.get('goal_option_count')}" if row.get("goal_option_count") else None),
                ("direction", row.get("flow_direction_hypothesis") if kind == "actor" else row.get("open_direction")),
                ("oracle_relation_kind", row.get("oracle_relation_kind")),
                ("status", row.get("status")),
                ("admission", row.get("admission_status")),
            ],
            out_path=pass_dir / f"oracle_pass_{idx:02d}_{kind}.png",
            bboxes=bboxes,
        )
        card.update({"pass_family": "oracle_sidecar", "pass_order": idx, "pass_id": kind, "status": row.get("status")})
        cards.append(card)
    for sidecar_index, sidecar_path in enumerate(sorted((out_dir / "oracle").glob("*.json")), len(oracle_rows) + 1):
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        assumptions = sidecar.get("assumptions") if isinstance(sidecar.get("assumptions"), list) else []
        assumption = assumptions[0] if assumptions and isinstance(assumptions[0], dict) else {}
        payload = assumption.get("assumption_payload") if isinstance(assumption.get("assumption_payload"), dict) else {}
        goal = payload.get("goal_candidate") if isinstance(payload.get("goal_candidate"), dict) else {}
        actor = payload.get("actor_candidate") if isinstance(payload.get("actor_candidate"), dict) else {}
        goal_component = goal.get("selected_component") if isinstance(goal.get("selected_component"), dict) else {}
        actor_component = actor.get("selected_component") if isinstance(actor.get("selected_component"), dict) else {}
        bboxes = []
        if goal_component.get("bbox_xyxy"):
            bboxes.append((goal_component.get("bbox_xyxy"), "ORACLE GOAL?", (74, 222, 128)))
        if actor_component.get("bbox_xyxy"):
            bboxes.append((actor_component.get("bbox_xyxy"), "ORACLE ACTOR?", (250, 204, 21)))
        card = pass_card(
            raw_grid=raw_grid,
            title=f"Oracle sidecar {sidecar_index:02d}: {sidecar.get('selected_missing_element_id')}",
            subtitle=f"direct Oracle assumption packet: {repo_rel(sidecar_path)}",
            rows=[
                ("selected_missing_element_id", sidecar.get("selected_missing_element_id")),
                ("assumption_count", sidecar.get("assumption_count")),
                ("missing_element_count", sidecar.get("missing_element_count")),
                ("statement", assumption.get("assumption_statement")),
                ("goal_status", goal.get("status")),
                ("goal_component", goal_component.get("component_id")),
                ("actor_status", actor.get("status")),
                ("actor_component", actor_component.get("component_id")),
                ("admission", assumption.get("admission_status")),
            ],
            out_path=pass_dir / f"oracle_sidecar_{sidecar_index:02d}_{sidecar.get('selected_missing_element_id')}.png",
            bboxes=bboxes,
        )
        card.update(
            {
                "pass_family": "oracle_sidecar",
                "pass_order": sidecar_index,
                "pass_id": sidecar.get("selected_missing_element_id"),
                "status": assumption.get("status", sidecar.get("status")),
            }
        )
        cards.append(card)

    symmetry = packet.get("symmetry_hypotheses") if isinstance(packet.get("symmetry_hypotheses"), dict) else {}
    for sym_pass in symmetry.get("symmetry_passes", []) or []:
        if not isinstance(sym_pass, dict):
            continue
        pass_id = str(sym_pass.get("pass_id"))
        order = int(sym_pass.get("pass_order") or 0)
        bboxes = []
        for idx, group_id in enumerate(sym_pass.get("output_group_ids", []) or []):
            group = find_symmetry_group(packet, str(group_id))
            if group is not None:
                bboxes.append((group.get("bbox_xyxy"), str(group.get("symmetry_kind") or group_id), PASS_COLORS[idx % len(PASS_COLORS)]))
        card = pass_card(
            raw_grid=raw_grid,
            title=f"Symmetry pass {order:02d}: {pass_id}",
            subtitle=str(sym_pass.get("definition") or "static symmetry calculation"),
            rows=[
                ("pass_id", pass_id),
                ("output_group_count", sym_pass.get("output_group_count")),
                ("output_group_ids", ", ".join(str(v) for v in sym_pass.get("output_group_ids", []) or [])),
                ("status", sym_pass.get("status")),
                ("admission", sym_pass.get("admission_status")),
            ],
            out_path=pass_dir / f"symmetry_pass_{order:02d}_{pass_id}.png",
            bboxes=bboxes,
        )
        card.update({"pass_family": "symmetry_internal", "pass_order": order, "pass_id": pass_id, "status": sym_pass.get("status")})
        cards.append(card)

    lookup = object_bbox_lookup(packet)
    for statecon_index, statecon in enumerate(packet.get("state_condition_hypotheses", []) or [], 1):
        if not isinstance(statecon, dict):
            continue
        statecon_id = str(statecon.get("statecon_id"))
        for geoband_pass in statecon.get("geoband_passes", []) or []:
            if not isinstance(geoband_pass, dict):
                continue
            pass_id = str(geoband_pass.get("pass_id"))
            order = int(geoband_pass.get("pass_order") or 0)
            bboxes = []
            for idx, object_id in enumerate(geoband_pass.get("input_object_ids", []) or []):
                bboxes.append((lookup.get(str(object_id)), f"IN:{object_id}", PASS_COLORS[idx % len(PASS_COLORS)]))
            for idx, group_id in enumerate(geoband_pass.get("output_group_ids", []) or []):
                bboxes.append((lookup.get(str(group_id)), f"OUT:{group_id}", (74, 222, 128)))
            card = pass_card(
                raw_grid=raw_grid,
                title=f"GeoBand pass S{statecon_index}.{order}: {pass_id}",
                subtitle=statecon_id,
                rows=[
                    ("pass_id", pass_id),
                    ("input_object_ids", ", ".join(str(v) for v in geoband_pass.get("input_object_ids", []) or [])),
                    ("output_group_ids", ", ".join(str(v) for v in geoband_pass.get("output_group_ids", []) or [])),
                    ("unresolved_boundary_ids", ", ".join(str(v) for v in geoband_pass.get("unresolved_boundary_ids", []) or [])),
                    ("reason_codes", ", ".join(str(v) for v in geoband_pass.get("reason_codes", []) or [])),
                    ("status", geoband_pass.get("status")),
                ],
                out_path=pass_dir / f"geoband_pass_s{statecon_index}_{order}_{pass_id}.png",
                bboxes=[row for row in bboxes if row[0] is not None],
            )
            card.update({"pass_family": "geoband_internal", "pass_order": order, "pass_id": pass_id, "status": geoband_pass.get("status"), "statecon_id": statecon_id})
            cards.append(card)
        portband = statecon.get("portband_hypotheses") if isinstance(statecon.get("portband_hypotheses"), dict) else {}
        alignments = [row for row in portband.get("alignments", []) or [] if isinstance(row, dict)]
        for alignment_index, alignment in enumerate(alignments[:6], 1):
            bboxes = []
            for box in alignment.get("source_bboxes_xyxy", []) or []:
                bboxes.append((box, None, (251, 113, 133)))
            for box in alignment.get("imagined_source_bboxes_xyxy", []) or []:
                bboxes.append((box, None, (125, 211, 252)))
            for box in alignment.get("target_bboxes_xyxy", []) or []:
                bboxes.append((box, None, (74, 222, 128)))
            source_ports = alignment.get("source_port_centers_xy") if isinstance(alignment.get("source_port_centers_xy"), list) else []
            target_ports = alignment.get("target_port_centers_xy") if isinstance(alignment.get("target_port_centers_xy"), list) else []
            projected_ports = shifted_points(source_ports, alignment.get("transform_hypothesis_xy"))
            port_edges: list[tuple[Any, Any, tuple[int, int, int]]] = []
            port_markers: list[tuple[Any, str, tuple[int, int, int]]] = []
            if len(source_ports) >= 2:
                port_edges.append((source_ports[0], source_ports[1], (251, 113, 133)))
                for port_index, point in enumerate(source_ports[:2], 1):
                    port_markers.append((point, f"S{port_index}", (251, 113, 133)))
            if len(target_ports) >= 2:
                port_edges.append((target_ports[0], target_ports[1], (74, 222, 128)))
                for port_index, point in enumerate(target_ports[:2], 1):
                    port_markers.append((point, f"T{port_index}", (74, 222, 128)))
            if len(projected_ports) >= 2:
                port_edges.append((projected_ports[0], projected_ports[1], (125, 211, 252)))
                for port_index, point in enumerate(projected_ports[:2], 1):
                    port_markers.append((point, f"M{port_index}", (125, 211, 252)))
            for projected, target in zip(projected_ports[:2], target_ports[:2]):
                port_edges.append((projected, target, (250, 204, 21)))
            card = pass_card(
                raw_grid=raw_grid,
                title=f"PortBand direct S{statecon_index}.{alignment_index}",
                subtitle="actual PortBand edge match: S=source ports, M=projected source match, T=target ports; review-only",
                rows=[
                    ("statecon_id", statecon_id),
                    ("portband_id", alignment.get("portband_id")),
                    ("match_basis", alignment.get("match_basis")),
                    ("target_edge_match_rule", alignment.get("target_edge_match_rule")),
                    ("match_status", alignment.get("match_status")),
                    ("exact_alignment", alignment.get("exact_alignment")),
                    ("near_alignment", alignment.get("near_alignment")),
                    ("alignment_error_px", alignment.get("alignment_error_px")),
                    ("axis", alignment.get("axis")),
                    ("transform_hypothesis_xy", alignment.get("transform_hypothesis_xy")),
                    ("projected_source_port_centers_xy", projected_ports),
                    ("projection_match_edges", f"{min(len(projected_ports), len(target_ports), 2)} projected-source-to-target connectors"),
                    ("source_object_ids", ", ".join(str(v) for v in alignment.get("source_object_ids", []) or [])),
                    ("target_object_ids", ", ".join(str(v) for v in alignment.get("target_object_ids", []) or [])),
                    ("source_port_centers_xy", alignment.get("source_port_centers_xy")),
                    ("target_port_centers_xy", alignment.get("target_port_centers_xy")),
                ],
                out_path=pass_dir / f"portband_direct_s{statecon_index}_rank_{alignment_index:03d}.png",
                bboxes=[row for row in bboxes if row[0] is not None],
                coordinate_edges=port_edges,
                port_markers=port_markers,
            )
            card.update(
                {
                    "pass_family": "portband_direct",
                    "pass_order": alignment_index,
                    "pass_id": alignment.get("portband_id"),
                    "status": alignment.get("match_status"),
                    "statecon_id": statecon_id,
                }
            )
            cards.append(card)
        fleet = portband.get("portband_fleet") if isinstance(portband.get("portband_fleet"), dict) else {}
        candidates = [row for row in fleet.get("ranked_candidates", []) or [] if isinstance(row, dict)]
        for candidate in candidates[:3]:
            rank = int(candidate.get("fleet_rank") or 0)
            imagined_boxes = candidate.get("imagined_member_bboxes_xyxy") if isinstance(candidate.get("imagined_member_bboxes_xyxy"), list) else []
            bboxes = [
                (box, f"F{rank}.{idx + 1}", PASS_COLORS[idx % len(PASS_COLORS)])
                for idx, box in enumerate(imagined_boxes)
                if isinstance(box, list) and len(box) == 4
            ]
            for idx, target_id in enumerate(candidate.get("target_object_ids", []) or []):
                bboxes.append((lookup.get(str(target_id)), f"T{idx + 1}", (74, 222, 128)))
            card = pass_card(
                raw_grid=raw_grid,
                title=f"PortBandFleet S{statecon_index} rank {rank}",
                subtitle="static combo-shape matrix candidate; imagined copy preview only",
                rows=[
                    ("statecon_id", statecon_id),
                    ("fleet_candidate_id", candidate.get("fleet_candidate_id")),
                    ("candidate_count", fleet.get("candidate_count")),
                    ("rank_score", candidate.get("rank_score")),
                    ("portband_fit_score", candidate.get("portband_fit_score")),
                    ("alignment_error_px", candidate.get("alignment_error_px")),
                    ("combo_size", candidate.get("combo_size")),
                    ("axis", candidate.get("axis")),
                    ("transform_hypothesis_xy", candidate.get("transform_hypothesis_xy")),
                    ("source_object_ids", ", ".join(str(v) for v in candidate.get("source_object_ids", []) or [])),
                    ("target_object_ids", ", ".join(str(v) for v in candidate.get("target_object_ids", []) or [])),
                    ("boundary", candidate.get("mcts_semantics")),
                ],
                out_path=pass_dir / f"portband_fleet_s{statecon_index}_rank_{rank:03d}.png",
                bboxes=[row for row in bboxes if row[0] is not None],
            )
            card.update(
                {
                    "pass_family": "portband_fleet",
                    "pass_order": rank,
                    "pass_id": candidate.get("fleet_candidate_id"),
                    "status": candidate.get("status"),
                    "statecon_id": statecon_id,
                }
            )
            cards.append(card)
        matrix = portband.get("lineup_matrix") if isinstance(portband.get("lineup_matrix"), dict) else {}
        lineup_rows = [row for row in matrix.get("rows", []) or [] if isinstance(row, dict)]
        for lineup in lineup_rows[:4]:
            rank = int(lineup.get("lineup_rank") or 0)
            bboxes = []
            source_box = lineup.get("source_bbox_xyxy")
            imagined_box = lineup.get("imagined_lineup_bbox_xyxy")
            if isinstance(source_box, list) and len(source_box) == 4:
                bboxes.append((source_box, "SRC", (251, 113, 133)))
            if isinstance(imagined_box, list) and len(imagined_box) == 4:
                bboxes.append((imagined_box, "LINEUP", (125, 211, 252)))
            for idx, target_id in enumerate(lineup.get("target_object_ids", []) or []):
                bboxes.append((lookup.get(str(target_id)), f"T{idx + 1}", (74, 222, 128)))
            card = pass_card(
                raw_grid=raw_grid,
                title=f"PortBandLineup S{statecon_index} rank {rank}",
                subtitle="per-support lineup route confidence matrix; static route preview only",
                rows=[
                    ("statecon_id", statecon_id),
                    ("lineup_matrix_row_id", lineup.get("lineup_matrix_row_id")),
                    ("row_kind", lineup.get("row_kind")),
                    ("source_object_id", lineup.get("source_object_id")),
                    ("source_component_ids", ", ".join(str(v) for v in lineup.get("source_component_ids", []) or [])),
                    ("target_object_ids", ", ".join(str(v) for v in lineup.get("target_object_ids", []) or [])),
                    ("confidence_score", lineup.get("confidence_score")),
                    ("confidence_band", lineup.get("confidence_band")),
                    ("confidence_model", lineup.get("confidence_model")),
                    ("lineup_route_label", lineup.get("lineup_route_label")),
                    ("transform_hypothesis_xy", lineup.get("transform_hypothesis_xy")),
                    ("linked_alignment_ids", ", ".join(str(v) for v in lineup.get("linked_alignment_ids", []) or [])),
                    ("linked_fleet_candidate_ids", ", ".join(str(v) for v in lineup.get("linked_fleet_candidate_ids", []) or [])),
                    ("boundary", lineup.get("lineup_route_status")),
                ],
                out_path=pass_dir / f"portband_lineup_matrix_s{statecon_index}_rank_{rank:03d}.png",
                bboxes=[row for row in bboxes if row[0] is not None],
            )
            card.update(
                {
                    "pass_family": "portband_lineup_matrix",
                    "pass_order": rank,
                    "pass_id": lineup.get("lineup_matrix_row_id"),
                    "status": lineup.get("status"),
                    "statecon_id": statecon_id,
                }
            )
            cards.append(card)
    return cards


def oracle_pass_count(packet: dict[str, Any]) -> int:
    actors = packet.get("actor_affordance_hypotheses") if isinstance(packet.get("actor_affordance_hypotheses"), list) else []
    targets = packet.get("target_affordance_hypotheses") if isinstance(packet.get("target_affordance_hypotheses"), list) else []
    oracle_rows = [
        row
        for row in [*actors, *targets]
        if isinstance(row, dict) and str(row.get("source", "")).startswith("oracle_first_pass")
    ]
    return len(oracle_rows)


def render_cards(packet: dict[str, Any], pass_assets: list[dict[str, Any]] | None = None) -> str:
    counts = packet_counts(packet)
    admission = packet.get("admission") if isinstance(packet.get("admission"), dict) else {}
    layer = packet.get("layer_output_packets") if isinstance(packet.get("layer_output_packets"), dict) else {}
    pass_assets = pass_assets or []
    pass_family_counts = {
        family: sum(1 for item in pass_assets if item.get("pass_family") == family)
        for family in (
            "view_system",
            "contra_layer",
            "oracle_sidecar",
            "symmetry_internal",
            "geoband_internal",
            "portband_direct",
            "portband_fleet",
            "portband_lineup_matrix",
        )
    }
    direct_alignment_count = 0
    fleet_candidate_count = 0
    lineup_matrix_row_count = 0
    for statecon in packet.get("state_condition_hypotheses", []) or []:
        if not isinstance(statecon, dict):
            continue
        portband = statecon.get("portband_hypotheses") if isinstance(statecon.get("portband_hypotheses"), dict) else {}
        direct_alignment_count += len([row for row in portband.get("alignments", []) or [] if isinstance(row, dict)])
        fleet = portband.get("portband_fleet") if isinstance(portband.get("portband_fleet"), dict) else {}
        fleet_candidate_count += int(fleet.get("candidate_count") or 0)
        matrix = portband.get("lineup_matrix") if isinstance(portband.get("lineup_matrix"), dict) else {}
        lineup_matrix_row_count += int(matrix.get("row_count") or 0)
    cards = [
        ("CONTRA ran", bool((packet.get("debug_summary") or {}).get("contra_ran")), "good"),
        ("Admission", admission.get("admission_status", packet.get("admission_status")), "bad"),
        ("Unified pass cards", pass_family_counts["view_system"], "good"),
        ("CONTRA pass images", pass_family_counts["contra_layer"], "good"),
        ("Oracle pass images", pass_family_counts["oracle_sidecar"], "warn"),
        ("Symmetry pass images", pass_family_counts["symmetry_internal"], "warn"),
        ("GeoBand pass images", pass_family_counts["geoband_internal"], "warn"),
        ("PortBand direct images", pass_family_counts["portband_direct"], "warn"),
        ("PortBand direct alignments", direct_alignment_count, "warn"),
        ("PortBandFleet images", pass_family_counts["portband_fleet"], "warn"),
        ("PortBandFleet candidates", fleet_candidate_count, "warn"),
        ("PortBand lineup images", pass_family_counts["portband_lineup_matrix"], "warn"),
        ("PortBand lineup rows", lineup_matrix_row_count, "warn"),
        ("Raw colors", counts.get("raw_colors"), ""),
        ("Components", counts.get("raw_components"), ""),
        ("Actors", counts.get("actors"), ""),
        ("Targets", counts.get("targets"), ""),
        ("Supports", counts.get("supports"), ""),
        ("StateCons", counts.get("statecons"), ""),
        ("GeoBands", counts.get("geobands"), ""),
        ("Symmetry groups", counts.get("symmetry_groups"), "warn"),
        ("Computed outputs", layer.get("computed_output_count"), "good"),
        ("Skipped outputs", layer.get("skipped_output_count"), "bad"),
    ]
    out = ["<section class='cards'>"]
    for label, value, cls in cards:
        if isinstance(value, bool):
            value = "yes" if value else "no"
        out.append(
            "<div class='card'>"
            f"<div class='label'>{h(label)}</div>"
            f"<div class='value {cls}'>{h(value)}</div>"
            "</div>"
        )
    out.append("</section>")
    return "\n".join(out)


def render_pass_image_review(pass_assets: list[dict[str, Any]], *, index_dir: Path) -> str:
    if not pass_assets:
        return "<h2>Pass Image Ledger</h2><p class='boundary'>No pass images were generated.</p>"
    by_family = {
        "view_system": "Config, Oracle, Ghost, And CONTRA Pass Cards",
        "contra_layer": "CONTRA Layer Pass Images",
        "oracle_sidecar": "Oracle Sidecar Pass Images",
        "symmetry_internal": "Symmetry Internal Pass Images",
        "geoband_internal": "GeoBand Internal Pass Images",
        "portband_direct": "PortBand Direct Alignment Images",
        "portband_fleet": "PortBandFleet Combo Shape Images",
        "portband_lineup_matrix": "PortBand Lineup Matrix Images",
    }
    out = ["<h2>Pass Image Ledger</h2>"]
    out.append(
        "<p class='boundary'>Every generated pass image is a review aid over the same static SP80 raw frame for this packet level. "
        "The text panel on each image is the calculation result pulled from the packet row for that pass.</p>"
    )
    out.append("<table><thead><tr><th>family</th><th>count</th></tr></thead><tbody>")
    for family, title in by_family.items():
        out.append(f"<tr><td>{h(title)}</td><td>{sum(1 for row in pass_assets if row.get('pass_family') == family)}</td></tr>")
    out.append("</tbody></table>")
    for family, title in by_family.items():
        rows = [row for row in pass_assets if row.get("pass_family") == family]
        if not rows:
            continue
        out.append(f"<h2>{h(title)}</h2>")
        out.append("<section class='pass-images'>")
        for row in rows:
            href = rel_for_html(row["path"], index_dir)
            out.append(
                "<div class='pass-card'>"
                f"<h3>{h(row.get('title'))}</h3>"
                f"<a href='{href}'><img src='{href}' alt='{h(row.get('title'))}'></a>"
                f"<p>{h(row.get('subtitle'))}</p>"
                f"<p class='muted'>status={h(row.get('status'))} pass_id={h(row.get('pass_id'))}</p>"
                "</div>"
            )
        out.append("</section>")
    return "\n".join(out)


def render_visuals(level_index: int, index_dir: Path) -> str:
    level = f"{level_index:02d}"
    final_frame = ROOT / f"codexlab/runs/contra_v001_sp80_deltaband_controlled_run/level_{level}/action_sequence/png/frame_021.png"
    visuals = [
        (
            "Raw frame",
            ROOT / f"codexlab/runs/sp80_sam_color_components/frames/level_{level}_raw.png",
            "Raw SP80 frame used by the static CONTRA review packet.",
            "frame",
        ),
        (
            "Large scene surface overlay",
            ROOT / f"codexlab/runs/sp80_color_board_viewer/scene_surface_overlays/level_{level}_surface_hypotheses.png",
            "Existing SP80 viewer overlay.",
            "sheet",
        ),
        (
            "Large hypothesis sheet",
            ROOT / f"codexlab/runs/sp80_color_board_viewer/scene_surface_sheets/level_{level}_surface_hypotheses.png",
            "Existing SP80 viewer hypothesis sheet.",
            "sheet",
        ),
        (
            "Color board review sheet",
            ROOT / f"codexlab/runs/sp80_color_board_viewer/review_sheets/level_{level}_color_board.png",
            "Existing color-board evidence page asset.",
            "sheet",
        ),
        (
            "Controlled before",
            ROOT / f"codexlab/runs/contra_v001_sp80_deltaband_controlled_run/level_{level}/png/before.png",
            "Reference controlled-action page frame; not part of the static packet admission.",
            "frame",
        ),
        (
            "Controlled after",
            ROOT / f"codexlab/runs/contra_v001_sp80_deltaband_controlled_run/level_{level}/png/after.png",
            "Reference controlled-action page frame; linked for paint context.",
            "frame",
        ),
        (
            "Final action frame",
            final_frame,
            "Reference final action frame from the controlled DeltaBand page.",
            "frame",
        ),
    ]
    out = ["<section class='visual-grid'>"]
    for title, path, note, cls in visuals:
        if not path.exists():
            continue
        href = rel_for_html(path, index_dir)
        out.append(
            f"<div class='visual {cls}'>"
            f"<h3>{h(title)}</h3>"
            f"<a href='{href}'><img src='{href}' alt='{h(title)}'></a>"
            f"<p>{h(note)}</p>"
            "</div>"
        )
    out.append("</section>")
    return "\n".join(out)


def render_object_tables(packet: dict[str, Any]) -> str:
    actors = packet.get("actor_affordance_hypotheses") if isinstance(packet.get("actor_affordance_hypotheses"), list) else []
    targets = packet.get("target_affordance_hypotheses") if isinstance(packet.get("target_affordance_hypotheses"), list) else []
    supports = (
        packet.get("support_action_affordance_hypotheses")
        if isinstance(packet.get("support_action_affordance_hypotheses"), list)
        else []
    )
    statecons = (
        packet.get("state_condition_hypotheses")
        if isinstance(packet.get("state_condition_hypotheses"), list)
        else []
    )
    out = ["<h2>Paint Objects</h2>"]
    out.append("<table><thead><tr><th>kind</th><th>id</th><th>component</th><th>raw</th><th>bbox / group</th><th>relationship</th><th>sets</th><th>status</th></tr></thead><tbody>")
    for row in actors:
        out.append(
            "<tr>"
            "<td>actor</td>"
            f"<td>{code(row.get('actor_id'))}</td>"
            f"<td>{code(row.get('actor_component_id'))}<br>key {code(row.get('orientation_key_component_id'))}</td>"
            f"<td>{h(row.get('actor_raw_pixel_value'))}<br>key {h(row.get('orientation_key_raw_pixel_value'))}</td>"
            f"<td>{code(row.get('actor_bbox_xyxy'))}<br>group {code(row.get('group_bbox_xyxy'))}</td>"
            f"<td>flow {h(row.get('flow_direction_hypothesis'))}<br>{h(row.get('activation_condition_hypothesis'))}</td>"
            f"<td>{code(row.get('actor_set_id'))}<br>A {h(row.get('actor_option_rank'))}/{h(row.get('actor_option_count'))}<br>{code(row.get('target_set_id'))}</td>"
            f"<td>{h(row.get('status'))}<br>{h(row.get('admission_status'))}</td>"
            "</tr>"
        )
    for row in targets:
        out.append(
            "<tr>"
            "<td>target</td>"
            f"<td>{code(row.get('target_id'))}</td>"
            f"<td>{code(row.get('component_id'))}</td>"
            f"<td>{h(row.get('source_raw_pixel_value'))}</td>"
            f"<td>{code(row.get('target_bbox_xyxy'))}</td>"
            f"<td>{h(row.get('target_kind'))}<br>open {h(row.get('open_direction'))}</td>"
            f"<td>{code(row.get('target_set_id'))}<br>T {h(row.get('goal_option_rank'))}/{h(row.get('goal_option_count'))}<br>{code(row.get('actor_set_id'))}</td>"
            f"<td>{h(row.get('status'))}<br>{h(row.get('admission_status'))}</td>"
            "</tr>"
        )
    for row in supports:
        out.append(
            "<tr>"
            "<td>support</td>"
            f"<td>{code(row.get('support_id'))}</td>"
            f"<td>{code(row.get('component_id'))}</td>"
            f"<td>{h(row.get('source_raw_pixel_value'))}</td>"
            f"<td>{code(row.get('bbox_xyxy'))}</td>"
            f"<td>{h(row.get('support_kind'))}<br>{h(row.get('support_shape_family'))}</td>"
            "<td></td>"
            f"<td>{h(row.get('status'))}<br>{h(row.get('admission_status'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    out.append("<h2>State Conditions</h2>")
    out.append("<table><thead><tr><th>statecon</th><th>actor</th><th>target</th><th>sets</th><th>support</th><th>score</th><th>status</th></tr></thead><tbody>")
    for row in statecons:
        nested_geoband = row.get("geoband_hypothesis") if isinstance(row.get("geoband_hypothesis"), dict) else {}
        members = nested_geoband.get("members") if isinstance(nested_geoband.get("members"), dict) else {}
        out.append(
            "<tr>"
            f"<td>{code(row.get('statecon_id'))}<br>{h(row.get('condition_kind'))}</td>"
            f"<td>{code(row.get('actor_affordance_id') or members.get('actor_affordance_id'))}<br>A {h(row.get('actor_option_rank') or members.get('actor_option_rank'))}/{h(row.get('actor_option_count') or members.get('actor_option_count'))}</td>"
            f"<td>{code(row.get('target_affordance_id') or members.get('target_affordance_id'))}<br>T {h(row.get('goal_option_rank') or members.get('goal_option_rank'))}/{h(row.get('goal_option_count') or members.get('goal_option_count'))}</td>"
            f"<td>{code(row.get('actor_set_id') or members.get('actor_set_id'))}<br>{code(row.get('target_set_id') or members.get('target_set_id'))}</td>"
            f"<td>{value_list(row.get('support_affordance_ids') or members.get('support_group_ids'), limit=3)}</td>"
            f"<td>{h(row.get('event_enable_score'))}</td>"
            f"<td>{h(row.get('condition_status_hypothesis'))}<br>{h(row.get('admission_status'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def render_symmetry(packet: dict[str, Any]) -> str:
    symmetry = packet.get("symmetry_hypotheses") if isinstance(packet.get("symmetry_hypotheses"), dict) else {}
    rows = symmetry_rows(symmetry)
    counts = {
        "center": len(symmetry.get("center_divider_hypotheses") or []),
        "playzone": len(symmetry.get("playzone_mirror_hypotheses") or []),
        "self": len(symmetry.get("component_self_symmetry_hypotheses") or []),
        "mirror": len(symmetry.get("component_mirror_pair_hypotheses") or []),
        "reverse": len(symmetry.get("component_reverse_symmetry_hypotheses") or []),
        "polar": len(symmetry.get("component_polar_symmetry_hypotheses") or []),
        "lock/key": len(symmetry.get("lock_key_sensor_hypotheses") or []),
    }
    out = ["<h2>Symmetry Sensor Layer</h2>"]
    out.append(
        "<p class='boundary'>"
        f"Counts: {', '.join(f'{k}={v}' for k, v in counts.items())}. "
        "These are static geometry labels and mirror clues; they do not predict an action."
        "</p>"
    )
    out.append("<table><thead><tr><th>group</th><th>kind</th><th>components</th><th>raw</th><th>axis</th><th>score</th><th>bbox</th><th>notes</th></tr></thead><tbody>")
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{code(row.get('symmetry_group_id'))}</td>"
            f"<td>{h(row.get('_group_name'))}<br>{h(row.get('symmetry_kind'))}</td>"
            f"<td>{value_list(row.get('member_component_ids'), limit=8)}</td>"
            f"<td>{value_list(row.get('raw_pixel_values'))}</td>"
            f"<td>{h(row.get('axis'))}</td>"
            f"<td>{h(row.get('score'))}</td>"
            f"<td>{code(row.get('bbox_xyxy'))}</td>"
            f"<td>floor_excluded={h(row.get('floor_excluded'))}<br>mirror_clue={h(row.get('mirror_scenario_clue'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def render_geobands(packet: dict[str, Any]) -> str:
    geobands = packet.get("geoband_hypotheses") if isinstance(packet.get("geoband_hypotheses"), list) else []
    boundary_by_statecon = {
        row.get("statecon_id"): row
        for row in packet.get("boundary_queues", [])
        if isinstance(row, dict) and row.get("statecon_id")
    }
    out = ["<h2>GeoBand Output With Symmetry Links</h2>"]
    out.append("<table><thead><tr><th>geoband</th><th>actor -> target</th><th>sets</th><th>support</th><th>scores</th><th>symmetry groups</th><th>boundaries</th><th>status</th></tr></thead><tbody>")
    for row in geobands:
        members = row.get("members") if isinstance(row.get("members"), dict) else {}
        bandwidth = row.get("bandwidth") if isinstance(row.get("bandwidth"), dict) else {}
        angles = row.get("angles") if isinstance(row.get("angles"), dict) else {}
        boundaries = boundary_by_statecon.get(row.get("statecon_id"), {})
        boundary_ids = [
            b.get("boundary_id")
            for b in boundaries.get("boundaries", [])
            if isinstance(b, dict) and b.get("boundary_id")
        ]
        out.append(
            "<tr>"
            f"<td>{code(row.get('geoband_id'))}<br>{h(row.get('definition'))}</td>"
            f"<td>{code(members.get('actor_affordance_id'))}<br>to {code(members.get('target_affordance_id'))}</td>"
            f"<td>{code(members.get('actor_set_id'))}<br>A {h(members.get('actor_option_rank'))}/{h(members.get('actor_option_count'))}<br>{code(members.get('target_set_id'))}<br>T {h(members.get('goal_option_rank'))}/{h(members.get('goal_option_count'))}</td>"
            f"<td>{value_list(members.get('support_group_ids'), limit=3)}</td>"
            f"<td>axis={h(bandwidth.get('axis_alignment'))}<br>sym={h(bandwidth.get('symmetry_score'))}<br>angle_err={h(angles.get('flow_to_target_angular_error_degrees'))}</td>"
            f"<td>count {h(members.get('symmetry_group_count'))}<br>{value_list(members.get('symmetry_group_ids'), limit=6)}</td>"
            f"<td>unresolved {h(boundaries.get('unresolved_boundary_count'))}<br>{value_list(boundary_ids, limit=4)}</td>"
            f"<td>{h(row.get('completeness_status'))}<br>{h(row.get('admission_status'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def render_ladder(packet: dict[str, Any]) -> str:
    ladder = packet.get("contra_relationship_ladder") if isinstance(packet.get("contra_relationship_ladder"), dict) else {}
    gate = packet.get("deltaband_gate") if isinstance(packet.get("deltaband_gate"), dict) else {}
    match_calibrations = gate.get("match_calibration_requirements")
    if not isinstance(match_calibrations, list):
        stage0 = gate.get("stage0_init_calibration_requirement") if isinstance(gate.get("stage0_init_calibration_requirement"), dict) else {}
        match_calibrations = [stage0] if stage0 else []
    out = ["<h2>Relationship Ladder And DeltaBand Gate</h2>"]
    out.append("<table><thead><tr><th>stage</th><th>completeness</th><th>present</th><th>missing</th><th>admission</th></tr></thead><tbody>")
    for stage in ladder.get("stages", []):
        if not isinstance(stage, dict):
            continue
        out.append(
            "<tr>"
            f"<td>{code(stage.get('stage_name'))}</td>"
            f"<td>{h(stage.get('completeness_status'))}</td>"
            f"<td>{value_list(stage.get('present_inputs'), limit=5)}</td>"
            f"<td>{value_list(stage.get('missing_inputs'), limit=5)}</td>"
            f"<td>{h(stage.get('admission_status'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    transfer_prior = gate.get("action_result_transfer_prior") if isinstance(gate.get("action_result_transfer_prior"), dict) else {}
    out.append("<table><thead><tr><th>DeltaBand status</th><th>transfer prior</th><th>required boundary</th><th>present artifacts</th><th>missing artifacts</th><th>pre-action ladder</th></tr></thead><tbody>")
    out.append(
        "<tr>"
        f"<td>{h(gate.get('status'))}<br>{h(gate.get('controlled_run_status'))}</td>"
        f"<td>{h(transfer_prior.get('status'))}<br>obs={h(transfer_prior.get('observation_count'))}<br>{value_list(transfer_prior.get('primary_actor_prior_component_ids'), limit=3)}</td>"
        f"<td>{h(gate.get('required_boundary'))}</td>"
        f"<td>{value_list(gate.get('present_artifacts'))}</td>"
        f"<td>{value_list(gate.get('missing_artifacts'), limit=8)}</td>"
        f"<td>{code(gate.get('pre_action_ladder_id'))}</td>"
        "</tr>"
    )
    out.append("</tbody></table>")
    out.append("<h3>Match Calibration Initialization Requirements</h3>")
    out.append(
        "<table><thead><tr>"
        "<th>stage</th><th>event</th><th>policy</th><th>status</th><th>missing artifacts</th><th>admission</th>"
        "</tr></thead><tbody>"
    )
    for row in match_calibrations:
        if not isinstance(row, dict):
            continue
        click_target = row.get("click_target") if isinstance(row.get("click_target"), dict) else {}
        choice_tree = row.get("ordered_choice_tree") if isinstance(row.get("ordered_choice_tree"), list) else []
        if choice_tree:
            policy = (
                f"{h(row.get('choice_policy'))}<br>"
                + "<br>".join(
                    f"{h(branch.get('priority'))}. {h(branch.get('branch_id'))}: {h(branch.get('candidate_action_policy'))}"
                    for branch in choice_tree
                    if isinstance(branch, dict)
                )
            )
        else:
            policy = f"{h(row.get('click_policy'))}<br>{h(click_target.get('coordinate_space'))}<br>{h(click_target.get('x_policy'))}, {h(click_target.get('y_policy'))}"
        out.append(
            "<tr>"
            f"<td>{code(row.get('stage_id'))}</td>"
            f"<td>{h(row.get('event_kind'))}<br>{h(row.get('intent'))}</td>"
            f"<td>{policy}</td>"
            f"<td>{h(row.get('status'))}<br>{h(row.get('calibration_run_status'))}</td>"
            f"<td>{value_list(row.get('missing_artifacts'), limit=8)}</td>"
            f"<td>gameplay={h(row.get('gameplay_move_admitted'))}<br>action={h(row.get('action_choice_admitted'))}<br>control={h(row.get('control_truth_admitted'))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    rankings = []
    for row in match_calibrations:
        if not isinstance(row, dict):
            continue
        ranking_packet = row.get("state_change_click_ranking")
        if isinstance(ranking_packet, dict) and isinstance(ranking_packet.get("ranked_candidates"), list):
            rankings = [candidate for candidate in ranking_packet["ranked_candidates"] if isinstance(candidate, dict)]
            break
    if rankings:
        out.append("<h3>Stage 0 Ranked Click Candidates</h3>")
        out.append(
            "<table><thead><tr>"
            "<th>rank</th><th>component</th><th>click</th><th>prior</th><th>relational / shape</th><th>symmetry context</th><th>color</th><th>basis</th>"
            "</tr></thead><tbody>"
        )
        for candidate in rankings[:8]:
            out.append(
                "<tr>"
                f"<td>{h(candidate.get('rank'))}</td>"
                f"<td>{code(candidate.get('candidate_component_id'))}<br>{h(candidate.get('candidate_kind'))}</td>"
                f"<td>{h(candidate.get('click_xy'))}</td>"
                f"<td>actor={h(candidate.get('actor_origin_prior_confidence'))}<br>primary={h(candidate.get('primary_actor_transfer_prior'))}<br>nonprimary={h(candidate.get('nonprimary_actor_like_due_to_transfer_prior'))}</td>"
                f"<td>rel={h(candidate.get('relational_link_score'))}<br>shape={h(candidate.get('shape_novelty_score'))}<br>asym={h(candidate.get('asymmetry_score'))}</td>"
                f"<td>actors {value_list(candidate.get('linked_actor_like_component_ids'), limit=2)}"
                f"<br>sensors {value_list(candidate.get('symmetry_sensor_ids'), limit=2)}"
                f"<br>static ctx {value_list(candidate.get('static_geoband_context_ids'), limit=2)}</td>"
                f"<td>{h(candidate.get('palette_name'))}<br>raw={h(candidate.get('raw_pixel_value'))}<br>salience={h(candidate.get('color_salience_score'))}</td>"
                f"<td>{h(candidate.get('state_change_rank_basis'))}<br>{h(candidate.get('geoband_context_status'))}<br>support={h(candidate.get('support_score'))}</td>"
                "</tr>"
            )
        out.append("</tbody></table>")
    return "\n".join(out)


def render_color_inventory(packet: dict[str, Any]) -> str:
    color_packet = packet.get("raw_color_component_packet") if isinstance(packet.get("raw_color_component_packet"), dict) else {}
    out = ["<h2>Raw Color / Component Inventory</h2>"]
    out.append("<table><thead><tr><th>raw</th><th>palette</th><th>pixels</th><th>components</th><th>bbox</th><th>center</th></tr></thead><tbody>")
    for color in color_packet.get("colors", []):
        if not isinstance(color, dict):
            continue
        comps = color.get("components") if isinstance(color.get("components"), list) else []
        out.append(
            "<tr>"
            f"<td>{h(color.get('raw_pixel_value'))}</td>"
            f"<td>{h(color.get('palette_name'))}</td>"
            f"<td>{h(color.get('pixel_count'))}</td>"
            f"<td>{' '.join(code(c.get('component_id')) for c in comps if isinstance(c, dict))}</td>"
            f"<td>{'<br>'.join(code(c.get('bbox_xyxy')) for c in comps if isinstance(c, dict))}</td>"
            f"<td>{'<br>'.join(code(c.get('center_xy')) for c in comps if isinstance(c, dict))}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def render_layer_outputs(packet: dict[str, Any]) -> str:
    layer = packet.get("layer_output_packets") if isinstance(packet.get("layer_output_packets"), dict) else {}
    out = ["<h2>Layer Outputs</h2>"]
    out.append("<table><thead><tr><th>layer</th><th>status</th><th>data</th><th>counts</th><th>missing</th><th>reason</th></tr></thead><tbody>")
    for row in layer.get("outputs", []):
        if not isinstance(row, dict):
            continue
        status_cls = "bad" if row.get("skipped") else "good"
        out.append(
            "<tr>"
            f"<td>{code(row.get('layer_id'))}</td>"
            f"<td class='{status_cls}'>{h(row.get('status'))}</td>"
            f"<td>{code(row.get('data_ref'))}</td>"
            f"<td>{code(compact_json(row.get('counts', {})))}</td>"
            f"<td>{value_list(row.get('missing_inputs'), limit=6)}</td>"
            f"<td>{value_list(row.get('reason_codes'), limit=6)}</td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "\n".join(out)


def render_links(index_dir: Path, *, packet_path: Path) -> str:
    links = [
        ("Raw packet JSON", packet_path),
        ("SP80 color board viewer", ROOT / "codexlab/runs/sp80_color_board_viewer/index.html"),
        ("SP80 controlled DeltaBand page", ROOT / "codexlab/runs/contra_v001_sp80_deltaband_controlled_run/index.html"),
        ("CONTRA symmetry smoke page", ROOT / "codexlab/runs/contra_api_symmetry_smoke/index.html"),
        ("Surface calibration JSON", ROOT / "codexlab/sp80_surface_calibration.json"),
    ]
    out = ["<h2>Source Review Pages</h2>", "<ul class='links'>"]
    for label, path in links:
        if path.exists():
            out.append(f"<li><a href='{rel_for_html(path, index_dir)}'>{h(label)}</a> <span>{h(repo_rel(path))}</span></li>")
    for path in sorted((index_dir / "oracle").glob("*.json")):
        out.append(f"<li><a href='{rel_for_html(path, index_dir)}'>Oracle sidecar JSON</a> <span>{h(repo_rel(path))}</span></li>")
    out.append("</ul>")
    return "\n".join(out)


def render_provenance(packet_path: Path, out_dir: Path, packet: dict[str, Any], source_command: str) -> str:
    source_script = ROOT / "codexlab" / "contra_sp80_review_page.py"
    source_api = ROOT / "codexlab" / "contra_api_smoke.py"
    level_index = int(packet.get("level_index") or 0)
    values = [
        ("condition", f"static SP80 level {level_index:02d} CONTRA paint review packet"),
        ("working_directory", str(ROOT)),
        ("packet", repo_rel(packet_path)),
        ("packet_sha256", sha256_file(packet_path)),
        ("source_raw_frame_sha256", packet.get("source_raw_frame_sha256")),
        ("generator", repo_rel(source_script)),
        ("generator_sha256", sha256_file(source_script)),
        ("contra_api_smoke_sha256", sha256_file(source_api) if source_api.exists() else "missing"),
        ("generated_at_utc", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
        ("host", socket.gethostname()),
        ("command_that_built_packet", source_command),
        ("command_that_built_page", " ".join(sys.argv)),
        ("output_index", repo_rel(out_dir / "index.html")),
    ]
    out = ["<h2>Provenance</h2>", "<table><tbody>"]
    for key, value in values:
        out.append(f"<tr><th>{h(key)}</th><td>{code(value)}</td></tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def render_html(
    packet: dict[str, Any],
    *,
    packet_path: Path,
    out_dir: Path,
    source_command: str,
    pass_assets: list[dict[str, Any]],
) -> str:
    debug = packet.get("debug_summary") if isinstance(packet.get("debug_summary"), dict) else {}
    admission = packet.get("admission") if isinstance(packet.get("admission"), dict) else {}
    level_index = int(packet.get("level_index", debug.get("level_index", 0)) or 0)
    fail_reasons = packet.get("fail_closed_reasons") or admission.get("fail_closed_reasons") or []
    title = f"SP80 Paint CONTRA Review: level {level_index:02d}"
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{h(title)}</title>
  <style>
    :root {{ color-scheme: dark; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #080c11; color: #e5edf5; font-size: 16px; line-height: 1.45; }}
    header {{ padding: 24px 30px; border-bottom: 1px solid #253244; background: #05070a; }}
    main {{ padding: 22px 30px 48px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; line-height: 1.15; letter-spacing: 0; }}
    h2 {{ margin: 34px 0 12px; font-size: 24px; line-height: 1.2; letter-spacing: 0; }}
    h3 {{ margin: 0 0 8px; font-size: 18px; line-height: 1.25; letter-spacing: 0; }}
    p {{ margin: 8px 0; }}
    a {{ color: #7dd3fc; }}
    code {{ display: inline-block; max-width: 100%; overflow-wrap: anywhere; background: #0d1520; color: #dbeafe; border: 1px solid #243247; border-radius: 4px; padding: 1px 5px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 18px; background: #101821; table-layout: fixed; }}
    th, td {{ border: 1px solid #2b3a4d; padding: 8px 10px; text-align: left; vertical-align: top; overflow-wrap: anywhere; }}
    th {{ background: #192536; color: #f8fafc; font-weight: 700; }}
    ul.links {{ margin: 0; padding-left: 20px; }}
    ul.links span {{ color: #a7b4c5; display: block; font-size: 13px; margin-top: 2px; }}
    .boundary {{ color: #c7d2df; background: #101821; border: 1px solid #2b3a4d; border-left: 4px solid #f59e0b; padding: 10px 12px; border-radius: 6px; }}
    .muted {{ color: #a7b4c5; }}
    .good {{ color: #4ade80; }}
    .warn {{ color: #fbbf24; }}
    .bad {{ color: #fb7185; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 18px 0; }}
    .card {{ min-height: 82px; background: #101821; border: 1px solid #2b3a4d; border-radius: 8px; padding: 12px; overflow: hidden; }}
    .label {{ color: #a7b4c5; font-size: 13px; }}
    .value {{ margin-top: 4px; font-size: 24px; font-weight: 760; line-height: 1.1; overflow-wrap: anywhere; }}
    .visual-grid {{ display: flex; flex-direction: column; gap: 18px; align-items: stretch; }}
    .visual {{ background: #101821; border: 1px solid #2b3a4d; border-radius: 8px; padding: 12px; }}
    .visual img {{ display: block; width: 100%; height: auto; image-rendering: pixelated; border: 1px solid #40516a; background: #05070a; }}
    .visual.frame img {{ width: min(100%, 768px); }}
    .visual.sheet img {{ width: min(100%, 1600px); }}
    .visual p {{ color: #a7b4c5; font-size: 14px; margin: 8px 0 0; }}
    .pass-images {{ display: flex; flex-direction: column; gap: 18px; }}
    .pass-card {{ background: #101821; border: 1px solid #2b3a4d; border-radius: 8px; padding: 12px; }}
    .pass-card img {{ display: block; width: min(100%, 1600px); height: auto; image-rendering: pixelated; border: 1px solid #40516a; background: #05070a; }}
    .pass-card p {{ color: #a7b4c5; font-size: 16px; margin: 8px 0 0; }}
    .summary {{ display: grid; grid-template-columns: minmax(280px, 1fr) minmax(280px, 1fr); gap: 14px; }}
    .panel {{ background: #101821; border: 1px solid #2b3a4d; border-radius: 8px; padding: 12px; }}
    @media (max-width: 980px) {{
      body {{ font-size: 15px; }}
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      h1 {{ font-size: 28px; }}
      table {{ table-layout: auto; font-size: 14px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{h(title)}</h1>
    <p class="muted">Static review packet for paint/CONTRA surface inspection. Not control truth, not goal truth, not action choice, not solve evidence.</p>
  </header>
  <main>
    <section class="summary">
      <div class="panel">
        <h3>Condition</h3>
        <p>{h(debug.get("game_id", packet.get("game_id")))} level {h(level_index)} / mode {h(packet.get("mode"))}</p>
        <p>Raw frame SHA256: {code(packet.get("source_raw_frame_sha256"))}</p>
      </div>
      <div class="panel">
        <h3>Fail-Closed Boundary</h3>
        <p>Status: <span class="bad">{h(admission.get("status", packet.get("status")))}</span></p>
        <p>{value_list(fail_reasons)}</p>
      </div>
    </section>
    {render_cards(packet, pass_assets)}
    {render_links(out_dir, packet_path=packet_path)}
    {render_pass_image_review(pass_assets, index_dir=out_dir)}
    {render_visuals(level_index, out_dir)}
    {render_object_tables(packet)}
    {render_symmetry(packet)}
    {render_geobands(packet)}
    {render_ladder(packet)}
    {render_color_inventory(packet)}
    {render_layer_outputs(packet)}
    {render_provenance(packet_path, out_dir, packet, source_command)}
  </main>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-grid", type=Path, default=None)
    parser.add_argument("--source-command", default=DEFAULT_SOURCE_COMMAND)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    packet_path = resolve_repo_path(args.packet)
    out_dir = resolve_repo_path(args.out_dir)
    packet = json.loads(packet_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_grid_path = resolve_repo_path(args.raw_grid) if args.raw_grid else raw_grid_path_for_packet(packet)
    pass_assets = build_pass_review_images(packet, out_dir=out_dir, raw_grid_path=raw_grid_path)
    html_text = render_html(
        packet,
        packet_path=packet_path,
        out_dir=out_dir,
        source_command=args.source_command,
        pass_assets=pass_assets,
    )
    index_path = out_dir / "index.html"
    index_path.write_text(html_text, encoding="utf-8")
    print(repo_rel(index_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
