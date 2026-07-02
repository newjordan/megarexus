#!/usr/bin/env python3
"""Build an Angel-style visual report from an existing SP80 harness run."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

from contra_service import projections  # noqa: E402  (reporting reads CONTRA internals via the internal-API projection contract)

ANGEL_DIST_DATA = Path("/home/frosty40/angel/dist/data")
ANGEL_PUBLIC_DATA = Path("/home/frosty40/angel/public/data")
REPORT_SCHEMA = "angel.contra_gameplay_report_manifest.v1"
TRACE_SCHEMA = "angel.report_trace_event.v1"
CONTRAVIEW_SCHEMA = "angel.contra_oracle_view_test.v1"
HIDDEN_STAGE_IDS = {"tabula_rasa_notebook"}

ENGINE_MECHANIC_SPECS = [
    ("config_pass", "Config Pass", "config", "config_pass", "config_pass"),
    ("contra_pass", "CONTRA Pass", "contra", "contra_pass", "contra_pass"),
    ("view_system_passes", "View System Passes", "contra", "view_system_passes", "view_system_passes"),
    ("raw_color_components", "Raw Color Components", "vision", "raw_color_component_packet", "raw_color_components"),
    ("control_isolation", "Control Isolation", "vision", "control_isolation_packet", "control_isolation"),
    ("scene_surfaces", "Scene Surfaces", "vision", "scene_surfaces", "scene_surfaces"),
    ("symmetry_hypotheses", "Symmetry Hypotheses", "relationships", "symmetry_hypotheses", "symmetry_hypotheses"),
    ("actor_affordances", "Actor Affordances", "affordance", "actor_affordance_hypotheses", "actor_affordances"),
    ("target_affordances", "Target Affordances", "affordance", "target_affordance_hypotheses", "target_affordances"),
    ("support_action_affordances", "Support Action Affordances", "affordance", "support_action_affordance_hypotheses", "support_action_affordances"),
    ("state_condition_hypotheses", "State Condition Hypotheses", "relationships", "state_condition_hypotheses", "state_condition_hypotheses"),
    ("geoband_boundary_queues", "GeoBand Boundary Queues", "relationships", "boundary_queues", "geoband_boundary_queues"),
    ("exploreband_budget", "ExploreBand Budget", "control_feedback", "exploreband_budget", "exploreband_budget"),
    ("action_semantics_ledger", "Action Semantics Ledger", "control_feedback", "action_semantics_ledger", "action_semantics_ledger"),
    ("layer_handoffs", "Layer Handoffs", "verification", "layer_handoffs", None),
    ("debug_summary", "Debug Summary", "verification", "debug_summary", "debug_summary"),
    ("admission", "Admission", "verification", "admission", "admission"),
]
LAYER_OUTPUT_VISUAL_STAGE_OVERRIDES = {
    "geoband_hypotheses": "relationship_graph",
    "tabula_rasa_notebook": "tabula_rasa_notebook",
}
HIDDEN_LAYER_OUTPUT_IDS = {"tabula_rasa_notebook"}

STAGE_LABELS = {
    "run_condition": ("Run Condition", "provenance"),
    "raw_frame": ("Raw Frame", "frame"),
    **{stage_id: (label, family) for stage_id, label, family, _packet_key, _layer_id in ENGINE_MECHANIC_SPECS},
    "oracle_pass": ("Oracle Pass", "oracle"),
    "ghost_pass": ("Ghost Pass", "ghost"),
    "ghost_memory": ("Ghost Memory", "ghost"),
    "portband_alignment": ("PortBand", "portband"),
    "relationship_graph": ("Relationship Graph", "relationships"),
    "relationship_ladder": ("Relationship Ladder", "relationships"),
    "brain_decision": ("Brain Decision", "brain"),
    "action_overlay": ("Action Overlay", "game_action"),
    "gameplay_frame": ("Gameplay Frame", "gameplay"),
    "action_before": ("Action Before", "game_action"),
    "action_after": ("Action After", "game_action"),
    "action_delta": ("Action Delta", "game_action"),
    "action_sequence": ("Action Sequence", "game_action"),
    "deltaband_gate": ("DeltaBand Gate", "control_feedback"),
    "actionmodband": ("ActionModBand", "control_feedback"),
    "episode_memory": ("Episode Memory", "control_feedback"),
    "provenance": ("Provenance", "verification"),
}

SECTION_ORDER = [
    ("run_condition", "Run Condition", ("run_condition",), "provenance"),
    ("raw_frame", "Raw Frame", ("raw_frame",), "frame"),
    *[(stage_id, label, (stage_id,), family) for stage_id, label, family, _packet_key, _layer_id in ENGINE_MECHANIC_SPECS],
    ("oracle_pass", "Oracle Pass", ("oracle_pass",), "oracle"),
    ("ghost_pass", "Ghost Pass", ("ghost_pass",), "ghost"),
    ("ghost_memory", "Ghost Memory", ("ghost_memory",), "ghost"),
    ("portband", "PortBand", ("portband_alignment",), "portband"),
    ("relationship_graph", "Relationship Graph", ("relationship_graph",), "relationships"),
    ("relationship_ladder", "Relationship Ladder", ("relationship_ladder",), "relationships"),
    ("brain_decision", "Brain Decision", ("brain_decision",), "brain"),
    ("action_overlay", "Action Overlay", ("action_overlay",), "game_action"),
    ("gameplay_frame", "Gameplay Frames", ("gameplay_frame",), "gameplay"),
    ("action_before", "Action Before", ("action_before",), "game_action"),
    ("action_after", "Action After", ("action_after",), "game_action"),
    ("action_delta", "Action Delta", ("action_delta",), "game_action"),
    ("action_sequence", "Action Sequence", ("action_sequence",), "game_action"),
    ("deltaband_gate", "DeltaBand Gate", ("deltaband_gate",), "control_feedback"),
    ("actionmodband", "ActionModBand", ("actionmodband",), "control_feedback"),
    ("episode_memory", "Episode Memory", ("episode_memory",), "control_feedback"),
    ("provenance", "Provenance", ("provenance",), "verification"),
]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    resolved = path if path.is_absolute() else ROOT / path
    resolved = resolved.resolve()
    resolved.relative_to(ROOT)
    return resolved


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def git_dirty() -> bool:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return bool(proc.stdout.strip()) if proc.returncode == 0 else True


def font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT_18 = font(18)
FONT_22 = font(22)
FONT_30 = font(30)
FONT_40 = font(40)
MONO_18 = font(18)


def wrap_text(text: str, width: int) -> list[str]:
    words = str(text).replace("\n", " \n ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if word == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_lines(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[str], *, fill: tuple[int, int, int], fnt: ImageFont.ImageFont, step: int) -> int:
    for line in lines:
        draw.text((x, y), line, fill=fill, font=fnt)
        y += step
    return y


def copy_frame(src_value: Any, out_path: Path) -> Path | None:
    if not isinstance(src_value, dict):
        return None
    path_value = src_value.get("path")
    if not path_value:
        return None
    src = resolve_repo_path(str(path_value))
    if not src.is_file():
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    return out_path


def frame_path(meta: dict[str, Any], phase: str) -> Path | None:
    frame = meta.get(f"{phase}_frame")
    if not isinstance(frame, dict):
        return None
    png = frame.get("png")
    if not isinstance(png, dict) or not png.get("path"):
        return None
    path = resolve_repo_path(str(png["path"]))
    return path if path.is_file() else None


def base_slide(title: str, subtitle: str = "", *, size: tuple[int, int] = (1600, 1000)) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", size, (10, 14, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size[0], 96], fill=(17, 25, 35))
    draw.text((32, 24), title, fill=(245, 248, 252), font=FONT_40)
    if subtitle:
        draw.text((34, 68), subtitle[:160], fill=(185, 194, 208), font=FONT_18)
    return img, draw


def paste_frame(img: Image.Image, src: Path | None, box: tuple[int, int, int, int]) -> None:
    if src is None or not src.is_file():
        return
    frame = Image.open(src).convert("RGB")
    max_w = box[2] - box[0]
    max_h = box[3] - box[1]
    scale = min(max_w / frame.width, max_h / frame.height)
    resized = frame.resize((max(1, int(frame.width * scale)), max(1, int(frame.height * scale))), Image.Resampling.NEAREST)
    x = box[0] + (max_w - resized.width) // 2
    y = box[1] + (max_h - resized.height) // 2
    img.paste(resized, (x, y))


def paste_frame_with_boxes(
    img: Image.Image,
    src: Path | None,
    box: tuple[int, int, int, int],
    bboxes: list[tuple[list[int], tuple[int, int, int], str]],
) -> None:
    if src is None or not src.is_file():
        return
    frame = Image.open(src).convert("RGB")
    max_w = box[2] - box[0]
    max_h = box[3] - box[1]
    scale = min(max_w / frame.width, max_h / frame.height)
    resized = frame.resize((max(1, int(frame.width * scale)), max(1, int(frame.height * scale))), Image.Resampling.NEAREST)
    x = box[0] + (max_w - resized.width) // 2
    y = box[1] + (max_h - resized.height) // 2
    overlay = resized.copy()
    draw = ImageDraw.Draw(overlay)
    raw_scale_x = resized.width / 64.0
    raw_scale_y = resized.height / 64.0
    for bbox, color, label in bboxes:
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        rect = [
            int(round(x1 * raw_scale_x)),
            int(round(y1 * raw_scale_y)),
            int(round((x2 + 1) * raw_scale_x)),
            int(round((y2 + 1) * raw_scale_y)),
        ]
        draw.rectangle(rect, outline=color, width=4)
        draw.text((rect[0] + 4, max(0, rect[1] - 24)), label[:28], fill=color, font=FONT_18)
    img.paste(overlay, (x, y))


def save_slide(path: Path, img: Image.Image) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return {"path": rel(path), "sha256": sha256_file(path)}


def _project_contra_pass(packet: dict[str, Any], projector, key: str) -> dict[str, Any]:
    """Read a CONTRA sub-pass through the internal-API projection contract.

    When the recorded ``contra_packet`` is a full CONTRA response the projection is
    contract-checked (schema asserted) and returns the same sub-pass the report
    rendered before, so slide bytes/hashes are unchanged.  Partial/legacy packets
    fall back to a defensive dict read so the report still renders an empty pass
    rather than crashing (SOP: show 0, do not omit).
    """
    contra_packet = packet.get("contra_packet")
    if isinstance(contra_packet, dict):
        try:
            return projector(contra_packet) or {}
        except ValueError:
            sub = contra_packet.get(key)
            if isinstance(sub, dict):
                return sub
    return {}


def oracle_pass_slide(path: Path, *, packet: dict[str, Any], level_index: int) -> dict[str, Any]:
    oracle_pass = _project_contra_pass(packet, projections.project_oracle_pass, "oracle_pass")
    actor_rows = [row for row in oracle_pass.get("actor_review_hypotheses", []) or [] if isinstance(row, dict)]
    target_rows = [row for row in oracle_pass.get("target_review_hypotheses", []) or [] if isinstance(row, dict)]
    boxes: list[tuple[list[int], tuple[int, int, int], str]] = []
    for row in actor_rows[:6]:
        bbox = row.get("actor_bbox_xyxy") or row.get("group_bbox_xyxy")
        if isinstance(bbox, list):
            boxes.append((bbox, (232, 72, 255), str(row.get("actor_id") or row.get("source_component_id") or "actor")))
    for row in target_rows[:8]:
        bbox = row.get("target_bbox_xyxy") or row.get("target_square_xyxy")
        if isinstance(bbox, list):
            boxes.append((bbox, (80, 220, 120), str(row.get("target_id") or row.get("component_id") or "target")))
    img, draw = base_slide("Oracle Pass", f"level {level_index:02d} / first episode packet")
    paste_frame_with_boxes(img, frame_path(packet, "before"), (44, 140, 844, 940), boxes)
    rows = [
        f"status: {oracle_pass.get('status')}",
        f"admission_status: {oracle_pass.get('admission_status')}",
        f"actor_hypothesis_count: {oracle_pass.get('actor_hypothesis_count')}",
        f"target_hypothesis_count: {oracle_pass.get('target_hypothesis_count')}",
        f"hypothesis_count: {oracle_pass.get('hypothesis_count')}",
        f"control_truth_admitted: {oracle_pass.get('control_truth_admitted')}",
        f"goal_condition_admitted: {oracle_pass.get('goal_condition_admitted')}",
        f"solve_condition_admitted: {oracle_pass.get('solve_condition_admitted')}",
    ]
    y = 150
    draw.text((900, y), "Recorded Oracle Hypotheses", fill=(140, 200, 255), font=FONT_30)
    y += 48
    y = draw_lines(draw, 920, y, rows, fill=(226, 232, 240), fnt=FONT_22, step=32)
    draw.text((900, y + 22), "Boundary", fill=(240, 180, 70), font=FONT_30)
    draw_lines(
        draw,
        920,
        y + 68,
        wrap_text("Oracle is review hypothesis only. It is not control truth, goal truth, action authority, prediction, or solve evidence.", 44),
        fill=(255, 232, 163),
        fnt=FONT_22,
        step=31,
    )
    return save_slide(path, img)


def ghost_pass_slide(path: Path, *, packet: dict[str, Any], level_index: int, ghost: dict[str, Any] | None) -> dict[str, Any]:
    ghost_pass = _project_contra_pass(packet, projections.project_ghost_pass, "ghost_pass")
    img, draw = base_slide("Ghost Pass", f"level {level_index:02d}")
    ghost_frame = None
    if isinstance(ghost, dict):
        final_raw = ghost.get("final_raw_frame") or {}
        png = final_raw.get("png") if isinstance(final_raw, dict) else None
        if isinstance(png, dict) and png.get("path"):
            candidate = resolve_repo_path(str(png["path"]))
            ghost_frame = candidate if candidate.is_file() else None
    paste_frame(img, ghost_frame or frame_path(packet, "before"), (44, 140, 844, 940))
    completion = (ghost or {}).get("completion_proof") if isinstance(ghost, dict) else {}
    rows = [
        f"ghost_pass_status: {ghost_pass.get('status')}",
        f"ghost_pass_count: {ghost_pass.get('ghost_count')}",
        f"ghost_memory_status: {(ghost or {}).get('status') if isinstance(ghost, dict) else 'absent'}",
        f"levels_completed_delta: {(completion or {}).get('levels_completed_delta')}",
        f"confirmed_relationship_row_count: {(ghost or {}).get('confirmed_relationship_row_count') if isinstance(ghost, dict) else 0}",
        f"route_or_replay_used: {ghost_pass.get('route_or_replay_used')}",
        f"coordinate_copy_used: {ghost_pass.get('coordinate_copy_used')}",
        f"admission_status: {ghost_pass.get('admission_status')}",
    ]
    y = 150
    draw.text((900, y), "Recorded Ghost Memory", fill=(140, 200, 255), font=FONT_30)
    y += 48
    y = draw_lines(draw, 920, y, rows, fill=(226, 232, 240), fnt=FONT_22, step=32)
    relation_rows = (ghost or {}).get("confirmed_relationship_rows", []) if isinstance(ghost, dict) else []
    if relation_rows:
        draw.text((900, y + 24), "First Confirmed Relations", fill=(140, 200, 255), font=FONT_30)
        y += 70
        for row in relation_rows[:4]:
            refs = row.get("source_evidence_refs") or {}
            line = f"{row.get('relation_memory_id')} {row.get('relation_family')} row={refs.get('lineup_matrix_row_id')}"
            y = draw_lines(draw, 920, y, wrap_text(line, 46), fill=(226, 232, 240), fnt=FONT_18, step=24)
    draw.text((900, 900), "Boundary: ghost memory is not future truth and has no action authority.", fill=(255, 232, 163), font=FONT_18)
    return save_slide(path, img)


def candidate_label(row: dict[str, Any]) -> str:
    for key in (
        "component_id",
        "actor_id",
        "target_id",
        "support_id",
        "state_condition_id",
        "geoband_id",
        "symmetry_group_id",
        "queue_id",
        "layer_id",
        "pass_id",
        "schema",
        "status",
    ):
        value = row.get(key)
        if value is not None:
            return str(value)
    return "packet row"


def collect_bboxes(value: Any, *, limit: int = 48) -> list[tuple[list[int], tuple[int, int, int], str]]:
    boxes: list[tuple[list[int], tuple[int, int, int], str]] = []
    colors = [
        (125, 211, 252),
        (74, 222, 128),
        (250, 204, 21),
        (251, 113, 133),
        (196, 181, 253),
        (251, 146, 60),
    ]

    def walk(current: Any, parent: dict[str, Any] | None = None) -> None:
        if len(boxes) >= limit:
            return
        if isinstance(current, dict):
            for key in (
                "bbox_xyxy",
                "actor_bbox_xyxy",
                "target_bbox_xyxy",
                "target_square_xyxy",
                "group_bbox_xyxy",
                "source_bbox_xyxy",
            ):
                bbox = current.get(key)
                if (
                    isinstance(bbox, list)
                    and len(bbox) == 4
                    and all(isinstance(v, (int, float)) for v in bbox)
                ):
                    label = candidate_label(current)
                    color = colors[len(boxes) % len(colors)]
                    boxes.append(([int(round(float(v))) for v in bbox], color, label))
                    break
            for child in current.values():
                walk(child, current)
        elif isinstance(current, list):
            for child in current:
                walk(child, parent)

    walk(value)
    return boxes


def count_rows(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        count = 1
        for child in value.values():
            if isinstance(child, list):
                count += len(child)
            elif isinstance(child, dict):
                count += 1
        return count
    return 0


def layer_output_row(packet: dict[str, Any], layer_id: str | None) -> dict[str, Any]:
    if not layer_id:
        return {}
    outputs = (((packet.get("contra_packet") or {}).get("layer_output_packets") or {}).get("outputs") or [])
    for row in outputs:
        if isinstance(row, dict) and row.get("layer_id") == layer_id:
            return row
    return {}


def layer_output_visual_stage_id(layer_id: str) -> str:
    if layer_id in LAYER_OUTPUT_VISUAL_STAGE_OVERRIDES:
        return LAYER_OUTPUT_VISUAL_STAGE_OVERRIDES[layer_id]
    for stage_id, _label, _family, _packet_key, candidate_layer_id in ENGINE_MECHANIC_SPECS:
        if candidate_layer_id == layer_id:
            return stage_id
    return layer_id


def layer_output_visual_map(packet: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = (((packet.get("contra_packet") or {}).get("layer_output_packets") or {}).get("outputs") or [])
    rows: list[dict[str, Any]] = []
    for row in outputs:
        if not isinstance(row, dict) or not row.get("layer_id"):
            continue
        layer_id = str(row["layer_id"])
        visual_stage_id = layer_output_visual_stage_id(layer_id)
        hidden = layer_id in HIDDEN_LAYER_OUTPUT_IDS or visual_stage_id in HIDDEN_STAGE_IDS
        rows.append(
            {
                "layer_id": layer_id,
                "visual_stage_id": visual_stage_id,
                "status": row.get("status"),
                "admission_status": row.get("admission_status"),
                "hidden": hidden,
                "hidden_reason": "hidden_by_default_per_user_report_structure" if hidden else None,
            }
        )
    return rows


def compact_value(value: Any) -> str:
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text[:110] + ("..." if len(text) > 110 else "")


def mechanic_summary_lines(value: Any, layer_row: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    if isinstance(layer_row, dict) and layer_row:
        for key in ("layer_id", "status", "admission_status", "missing_input_count", "recovered_error"):
            if key in layer_row:
                lines.append(f"layer.{key}: {layer_row.get(key)}")
    if isinstance(value, dict):
        for key in (
            "schema",
            "status",
            "admission_status",
            "source",
            "pass_count",
            "output_count",
            "computed_output_count",
            "skipped_output_count",
            "recovered_output_count",
            "color_count",
            "component_count",
            "hypothesis_count",
            "actor_hypothesis_count",
            "target_hypothesis_count",
            "relationship_change_count",
            "control_truth_admitted",
            "goal_condition_admitted",
            "action_choice_admitted",
            "solve_condition_admitted",
        ):
            if key in value:
                lines.append(f"{key}: {value.get(key)}")
        for key, child in value.items():
            if isinstance(child, list):
                lines.append(f"{key}: {len(child)} rows")
            elif isinstance(child, dict) and key not in {"features", "signals"}:
                lines.append(f"{key}: object")
    elif isinstance(value, list):
        lines.append(f"row_count: {len(value)}")
        for row in value[:8]:
            if isinstance(row, dict):
                parts = []
                for key in ("schema", "status", "admission_status", "component_id", "actor_id", "target_id", "support_id", "score"):
                    if key in row:
                        parts.append(f"{key}={row.get(key)}")
                if parts:
                    lines.append("row: " + " ".join(parts))
    else:
        lines.append(f"value: {value}")
    return lines[:18]


def engine_mechanic_slide(
    path: Path,
    *,
    packet: dict[str, Any],
    level_index: int,
    stage_id: str,
    stage_label: str,
    packet_key: str,
    layer_id: str | None,
) -> dict[str, Any]:
    contra_packet = packet.get("contra_packet") or {}
    value = contra_packet.get(packet_key)
    layer_row = layer_output_row(packet, layer_id)
    img, draw = base_slide(stage_label, f"level {level_index:02d} / recorded first episode packet / {stage_id}")
    boxes = collect_bboxes(value)
    paste_frame_with_boxes(img, frame_path(packet, "before"), (44, 140, 844, 940), boxes)
    y = 140
    draw.text((900, y), "Recorded Packet Mechanic", fill=(140, 200, 255), font=FONT_30)
    y += 46
    lines = [
        f"packet_key: {packet_key}",
        f"recorded_value_type: {type(value).__name__}",
        f"recorded_row_count: {count_rows(value)}",
        f"overlay_bbox_count: {len(boxes)}",
    ]
    lines.extend(mechanic_summary_lines(value, layer_row))
    for line in lines[:20]:
        y = draw_lines(draw, 920, y, wrap_text(line, 48), fill=(226, 232, 240), fnt=FONT_18, step=24)
        if y > 820:
            break
    if isinstance(value, list) and value:
        draw.text((900, 840), "First row", fill=(140, 200, 255), font=FONT_22)
        draw_lines(draw, 920, 872, wrap_text(compact_value(value[0]), 58)[:3], fill=(203, 213, 225), fnt=FONT_18, step=23)
    elif isinstance(value, dict):
        draw.text((900, 840), "Packet keys", fill=(140, 200, 255), font=FONT_22)
        draw_lines(draw, 920, 872, wrap_text(", ".join(sorted(str(k) for k in value.keys())[:24]), 58)[:3], fill=(203, 213, 225), fnt=FONT_18, step=23)
    draw.text((900, 954), "Boundary: recorded CONTRA mechanic image; review hypothesis only, not action authority or solve truth.", fill=(255, 232, 163), font=FONT_18)
    return save_slide(path, img)


def raw_xy_to_slide(point: Any, *, x0: int = 44, y0: int = 140, size: int = 800) -> tuple[int, int] | None:
    if not isinstance(point, list) or len(point) < 2:
        return None
    try:
        return (
            x0 + int(round(float(point[0]) / 64.0 * size)),
            y0 + int(round(float(point[1]) / 64.0 * size)),
        )
    except (TypeError, ValueError):
        return None


def relationship_graph_slide(path: Path, *, packet: dict[str, Any], level_index: int) -> dict[str, Any]:
    contra_packet = packet.get("contra_packet") or {}
    geobands = [row for row in contra_packet.get("geoband_hypotheses", []) or [] if isinstance(row, dict)]
    statecons = [row for row in contra_packet.get("state_condition_hypotheses", []) or [] if isinstance(row, dict)]
    geoband = geobands[0] if geobands else {}
    nodes = [row for row in geoband.get("relationship_nodes", []) or [] if isinstance(row, dict)]
    edges = [row for row in geoband.get("relationship_edges", []) or [] if isinstance(row, dict)]
    img, draw = base_slide("Relationship Graph", f"level {level_index:02d} / first episode packet")
    before = frame_path(packet, "before")
    if before and before.is_file():
        frame = Image.open(before).convert("RGB").resize((800, 800), Image.Resampling.NEAREST)
        img.paste(frame, (44, 140))
        draw.rectangle([44, 140, 844, 940], outline=(67, 82, 102), width=2)
    node_centers: dict[str, tuple[int, int]] = {}
    for node in nodes:
        center = raw_xy_to_slide(node.get("center_xy"))
        if center is not None:
            node_centers[str(node.get("node_id"))] = center
    color_by_role = {
        "actor": (232, 72, 255),
        "goal_attachment": (80, 220, 120),
        "support_action_candidate": (255, 220, 60),
        "symmetry_context": (125, 211, 252),
        "port_alignment_context": (255, 132, 25),
        "lineup_route_confidence_context": (255, 132, 25),
    }
    for edge in edges[:64]:
        a = node_centers.get(str(edge.get("from_node_id")))
        b = node_centers.get(str(edge.get("to_node_id")))
        if a is None or b is None:
            continue
        role = str(edge.get("relation_role") or "")
        color = color_by_role.get(role, (148, 163, 184))
        draw.line([a, b], fill=color, width=3)
    for node in nodes[:48]:
        center = node_centers.get(str(node.get("node_id")))
        if center is None:
            continue
        role = str(node.get("relation_role") or "")
        color = color_by_role.get(role, (226, 232, 240))
        x, y = center
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill=color, outline=(5, 7, 10), width=2)
        label = str(node.get("node_kind") or role or "node").replace("_", " ")
        draw.text((x + 10, max(140, y - 12)), label[:32], fill=color, font=FONT_18)
    rows = [
        f"statecon_count: {len(statecons)}",
        f"geoband_count: {len(geobands)}",
        f"relationship_nodes: {len(nodes)}",
        f"relationship_edges: {len(edges)}",
        f"geoband_status: {geoband.get('status')}",
        f"completeness_status: {geoband.get('completeness_status')}",
        f"admission_status: {geoband.get('admission_status')}",
    ]
    y = 150
    draw.text((900, y), "Recorded Relationship Network", fill=(140, 200, 255), font=FONT_30)
    y += 48
    y = draw_lines(draw, 920, y, rows, fill=(226, 232, 240), fnt=FONT_22, step=32)
    first_statecon = statecons[0] if statecons else {}
    feature_rows = (first_statecon.get("features") or {}) if isinstance(first_statecon, dict) else {}
    if isinstance(feature_rows, dict) and feature_rows:
        draw.text((900, y + 28), "StateCon Features", fill=(140, 200, 255), font=FONT_30)
        y += 74
        for key, value in list(feature_rows.items())[:8]:
            draw.text((920, y), f"{key}: {value}", fill=(226, 232, 240), font=FONT_18)
            y += 25
    draw.text((900, 900), "Boundary: relationship graph is static recorded hypothesis evidence, not action authority.", fill=(255, 232, 163), font=FONT_18)
    return save_slide(path, img)


def relationship_ladder_slide(path: Path, *, packet: dict[str, Any], level_index: int) -> dict[str, Any]:
    ladder = ((packet.get("contra_packet") or {}).get("contra_relationship_ladder") or {})
    stages = [row for row in ladder.get("stages", []) or [] if isinstance(row, dict)]
    img, draw = base_slide("Relationship Ladder", f"level {level_index:02d} / first episode packet")
    rows = [
        f"ladder_id: {ladder.get('ladder_id')}",
        f"stage_count: {ladder.get('stage_count')}",
        f"completeness_status: {ladder.get('completeness_status')}",
        f"admission_status: {ladder.get('admission_status')}",
        f"incomplete_stage_ids: {ladder.get('incomplete_stage_ids')}",
    ]
    y = 130
    draw.text((48, y), "Summary", fill=(140, 200, 255), font=FONT_30)
    y += 44
    y = draw_lines(draw, 64, y, rows, fill=(226, 232, 240), fnt=FONT_22, step=32)
    x0, y0 = 64, 350
    col_w = 360
    row_h = 92
    for idx, stage in enumerate(stages[:12]):
        col = idx % 4
        row = idx // 4
        x = x0 + col * col_w
        y = y0 + row * row_h
        complete = stage.get("completeness_status") == "complete"
        fill = (14, 32, 24) if complete else (38, 25, 18)
        outline = (74, 222, 128) if complete else (250, 204, 21)
        draw.rectangle([x, y, x + col_w - 18, y + row_h - 14], fill=fill, outline=outline, width=2)
        draw.text((x + 10, y + 8), str(stage.get("stage_name") or stage.get("stage_id"))[:34], fill=(245, 248, 252), font=FONT_18)
        draw.text((x + 10, y + 34), f"status: {stage.get('stage_status')}", fill=(203, 213, 225), font=FONT_18)
        missing = stage.get("missing_inputs") or []
        draw.text((x + 10, y + 58), f"missing: {len(missing)}", fill=(203, 213, 225), font=FONT_18)
    draw.text((64, 880), "Boundary: ladder is admission/completeness evidence only; no control, goal, prediction, action, or solve truth.", fill=(255, 232, 163), font=FONT_18)
    return save_slide(path, img)


def stage_id_for_slide(slide_type: str) -> str:
    if slide_type == "portband_selected_recommendations":
        return "portband_alignment"
    if slide_type == "deltaband":
        return "deltaband_gate"
    return slide_type


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
        sectioned_frame_ids.update(str(frame.get("frame_id")) for frame in section_frames)
        sections.append(
            {
                "section_id": section_id,
                "section_label": label,
                "section_order": order,
                "family": family,
                "stage_ids": sorted({str(frame.get("stage_id")) for frame in section_frames}),
                "level_indices": sorted(
                    {
                        int(frame["level_index"])
                        for frame in section_frames
                        if isinstance(frame.get("level_index"), int)
                    }
                ),
                "frame_count": len(section_frames),
                "frames": section_frames,
                "slider_kind": "single_function_lateral_image_sequence",
                "default_open": section_id in {"raw_frame", "portband"},
                "boundary": "Controlled harness review only. This section is not control truth, goal truth, action choice, prediction truth, or solve evidence.",
            }
        )
    remaining = [
        frame
        for frame in frames
        if str(frame.get("frame_id")) not in sectioned_frame_ids and frame.get("stage_id") not in HIDDEN_STAGE_IDS
    ]
    for frame in remaining:
        section_id = str(frame.get("stage_id") or "unclassified")
        stage_label, family = STAGE_LABELS.get(section_id, (section_id.replace("_", " ").title(), "unclassified"))
        sections.append(
            {
                "section_id": section_id,
                "section_label": stage_label,
                "section_order": len(SECTION_ORDER) + len(sections),
                "family": family,
                "stage_ids": [section_id],
                "level_indices": [int(frame["level_index"])] if isinstance(frame.get("level_index"), int) else [],
                "frame_count": 1,
                "frames": [frame],
                "slider_kind": "single_function_lateral_image_sequence",
                "default_open": False,
                "boundary": "Controlled harness review only. This section needs classification before evidence use.",
            }
        )
    return sections


def write_viewer_index(path: Path, *, run_id: str) -> str:
    viewer_url = f"/?tool=contra-oracle&stage=portband&data=/data/{run_id}/contraview_report.json"
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Angel contraview_report</title>
<meta http-equiv="refresh" content="0; url={html.escape(viewer_url)}">
<style>
body{{margin:0;background:#0b0f15;color:#e8edf4;font-family:Inter,system-ui,Segoe UI,sans-serif}}
main{{max-width:900px;margin:10vh auto;padding:28px}}
a{{color:#8cc8ff}}
.boundary{{background:#151c24;border-left:4px solid #f0b429;padding:10px 12px;color:#ffe8a3}}
code{{color:#d7e3f4}}
</style>
</head>
<body>
<main>
<h1>Angel contraview_report</h1>
<p class="boundary">Primary report surface is the Angel contra-oracle viewer. Controlled harness review only; not solve evidence unless levels_completed increased in the recorded run.</p>
<p><a href="{html.escape(viewer_url)}">Open Angel viewer</a></p>
<p>Data: <code>/data/{html.escape(run_id)}/contraview_report.json</code></p>
</main>
</body>
</html>
""",
        encoding="utf-8",
    )
    return viewer_url


def text_slide(path: Path, *, title: str, subtitle: str, sections: list[tuple[str, list[str]]]) -> dict[str, Any]:
    img, draw = base_slide(title, subtitle)
    y = 130
    for header, lines in sections:
        draw.text((48, y), header, fill=(140, 200, 255), font=FONT_30)
        y += 42
        wrapped_lines: list[str] = []
        for line in lines:
            wrapped_lines.extend(wrap_text(str(line), 82))
        y = draw_lines(draw, 64, y, wrapped_lines, fill=(226, 232, 240), fnt=FONT_22, step=31)
        y += 24
    return save_slide(path, img)


def action_slide(path: Path, *, title: str, packet: dict[str, Any], frame_phase: str, level_index: int, action_index: int) -> dict[str, Any]:
    img, draw = base_slide(title, f"level {level_index} action {action_index:03d} / {packet.get('case_id')}")
    src = frame_path(packet, "before" if frame_phase == "before" else "after")
    paste_frame(img, src, (36, 136, 836, 936))
    info = [
        f"case_id: {packet.get('case_id')}",
        f"status: {packet.get('status')}",
        f"brain primitive: {(packet.get('brain_decision') or {}).get('primitive')}",
        f"expected primitive: {packet.get('expected_primitive')}",
        f"action: {(packet.get('action') or {}).get('name')}",
        f"level completed: {(packet.get('before_state') or {}).get('levels_completed')} -> {(packet.get('after_state') or {}).get('levels_completed')}",
        f"changed pixels: {((packet.get('action_visual_correlation') or {}).get('frame_delta') or {}).get('changed_pixels')}",
        f"route_or_replay_used: {packet.get('route_or_replay_used')}",
        f"human_trace_used: {packet.get('human_trace_used')}",
        f"coordinate_script_used: {packet.get('coordinate_script_used')}",
    ]
    y = 150
    for line in info:
        draw.text((890, y), line, fill=(226, 232, 240), font=FONT_22)
        y += 34
    draw.text((890, 760), "Boundary", fill=(240, 180, 70), font=FONT_30)
    boundary = "Controlled action evidence only. This slide does not admit goal truth, action-choice truth, prediction truth, or solve truth."
    draw_lines(draw, 890, 808, wrap_text(boundary, 44), fill=(255, 232, 163), fnt=FONT_22, step=31)
    return save_slide(path, img)


def packet_click_xy(packet: dict[str, Any]) -> list[int | float] | None:
    selected = packet.get("selected_recommendation") if isinstance(packet.get("selected_recommendation"), dict) else {}
    target_override = selected.get("target_override") if isinstance(selected.get("target_override"), dict) else {}
    action = packet.get("action") if isinstance(packet.get("action"), dict) else {}
    action_data = action.get("data") if isinstance(action.get("data"), dict) else {}
    click_xy = action_data.get("x"), action_data.get("y")
    if not all(isinstance(v, (int, float)) for v in click_xy):
        override_click = target_override.get("click_xy")
        if isinstance(override_click, list) and len(override_click) == 2:
            click_xy = override_click[0], override_click[1]
    if all(isinstance(v, (int, float)) for v in click_xy):
        return [click_xy[0], click_xy[1]]
    return None


def packet_returned_frame_count(packet: dict[str, Any]) -> int:
    seq = packet.get("action_png_sequence")
    if not isinstance(seq, list):
        return 0
    count = 0
    for entry in seq:
        if isinstance(entry, dict) and entry.get("path"):
            try:
                if resolve_repo_path(str(entry["path"])).is_file():
                    count += 1
            except ValueError:
                continue
    return count


def action_overlay_slide(path: Path, *, packet: dict[str, Any], level_index: int, action_index: int) -> dict[str, Any]:
    img, draw = base_slide("Action Overlay", f"level {level_index} action {action_index:03d} / {packet.get('case_id')}")
    before = frame_path(packet, "before")
    selected = packet.get("selected_recommendation") if isinstance(packet.get("selected_recommendation"), dict) else {}
    refs = selected.get("evidence_refs") if isinstance(selected.get("evidence_refs"), dict) else {}
    target_override = selected.get("target_override") if isinstance(selected.get("target_override"), dict) else {}
    action = packet.get("action") if isinstance(packet.get("action"), dict) else {}
    click_xy_value = packet_click_xy(packet)
    bbox = refs.get("source_bbox_xyxy") or target_override.get("source_bbox_xyxy")
    boxes: list[tuple[list[int], tuple[int, int, int], str]] = []
    if isinstance(bbox, list):
        boxes.append((bbox, (255, 220, 60), "selected/source"))
    paste_frame_with_boxes(img, before, (36, 136, 836, 936), boxes)
    if click_xy_value is not None:
        # The pasted raw frame is centered in an 800x800 box and source frames are 64x64.
        px = 36 + int(round(float(click_xy_value[0]) / 64.0 * 800.0))
        py = 136 + int(round(float(click_xy_value[1]) / 64.0 * 800.0))
        draw.ellipse([px - 16, py - 16, px + 16, py + 16], outline=(255, 64, 64), width=5)
        draw.line([px - 24, py, px + 24, py], fill=(255, 64, 64), width=3)
        draw.line([px, py - 24, px, py + 24], fill=(255, 64, 64), width=3)
    rows = [
        f"case_id: {packet.get('case_id')}",
        f"primitive: {packet.get('decision_primitive')}",
        f"action: {action.get('name')}",
        f"click_xy: {click_xy_value}",
        f"source_bbox_xyxy: {bbox}",
        f"lineup_matrix_row_id: {refs.get('lineup_matrix_row_id')}",
        f"portband_route_key: {refs.get('portband_route_key')}",
        f"target_component_ids: {refs.get('target_component_ids')}",
        f"target_port_ids: {refs.get('target_port_ids')}",
        f"levels_completed_delta: {packet.get('levels_completed_delta')}",
    ]
    y = 150
    draw.text((890, y), "Action / Click Evidence", fill=(140, 200, 255), font=FONT_30)
    y += 46
    for row in rows:
        y = draw_lines(draw, 910, y, wrap_text(row, 46), fill=(226, 232, 240), fnt=FONT_22, step=30)
    draw.text((890, 880), "Boundary: overlay is recorded action evidence, not action authority or solve truth.", fill=(255, 232, 163), font=FONT_18)
    slide = save_slide(path, img)
    slide["click_xy"] = click_xy_value
    slide["has_click_marker"] = click_xy_value is not None
    slide["action_name"] = action.get("name")
    return slide


def delta_slide(path: Path, *, packet: dict[str, Any], level_index: int, action_index: int) -> dict[str, Any]:
    img, draw = base_slide("Action Delta", f"level {level_index} action {action_index:03d} / {packet.get('case_id')}")
    before = frame_path(packet, "before")
    after = frame_path(packet, "after")
    paste_frame(img, before, (36, 150, 500, 614))
    paste_frame(img, after, (548, 150, 1012, 614))
    if before and after:
        a = Image.open(before).convert("RGB")
        b = Image.open(after).convert("RGB")
        diff = ImageChops.difference(a, b)
        diff = diff.point(lambda p: 255 if p else 0)
        paste = diff.resize((464, 464), Image.Resampling.NEAREST)
        img.paste(paste, (1060, 150))
    labels = [
        "before",
        "after",
        "delta mask",
        f"changed_pixels: {((packet.get('action_visual_correlation') or {}).get('frame_delta') or {}).get('changed_pixels')}",
        f"selected_coord_delta: {packet.get('selected_coord_delta')}",
        f"selected_display_coord_delta: {packet.get('selected_display_coord_delta')}",
        f"actionmodband levels_completed_delta: {(packet.get('actionmodband_observations') or {}).get('levels_completed_delta')}",
    ]
    y = 650
    for line in labels:
        draw.text((48, y), str(line), fill=(226, 232, 240), font=FONT_22)
        y += 32
    return save_slide(path, img)


def brain_slide(path: Path, *, call: dict[str, Any], level_index: int, action_index: int) -> dict[str, Any]:
    decision = call.get("decision") or {}
    request = call.get("request") or {}
    sections = [
        (
            "Decision",
            [
                f"case_id: {call.get('case_id')}",
                f"primitive: {decision.get('primitive')}",
                f"confidence: {decision.get('confidence')}",
                f"latency_ms: {decision.get('latency_ms')}",
                f"error: {decision.get('error')}",
            ],
        ),
        (
            "Contract",
            [
                f"allowed_primitives: {', '.join(request.get('allowed_primitives') or [])}",
                "output: one JSON primitive only",
                "may_emit_route: false",
                "may_choose_coordinates: false",
            ],
        ),
    ]
    return text_slide(
        path,
        title="Brain Decision",
        subtitle=f"level {level_index} action {action_index:03d} / {call.get('case_id')}",
        sections=sections,
    )


def memory_slide(path: Path, *, packet: dict[str, Any], level_index: int, action_index: int, kind: str) -> dict[str, Any]:
    actionmod = packet.get("actionmodband_observations") or {}
    episode = packet.get("episode_memory_after_round") or {}
    deltaband = (episode.get("rounds") or [{}])[-1].get("deltaband_result") if isinstance(episode.get("rounds"), list) else None
    if not isinstance(deltaband, dict):
        deltaband = {}
    if kind == "deltaband":
        rows = [
            f"schema: {deltaband.get('schema')}",
            f"action_primitive: {deltaband.get('action_primitive')}",
            f"relationship_change_count: {deltaband.get('relationship_change_count')}",
            f"null_action_signal: {deltaband.get('null_action_signal')}",
            f"raw_pixel_delta_count: {deltaband.get('raw_pixel_delta_count')}",
        ]
    elif kind == "actionmodband":
        rows = [
            f"schema: {actionmod.get('schema')}",
            f"response_count: {actionmod.get('response_count')}",
            f"levels_completed_delta: {actionmod.get('levels_completed_delta')}",
            f"action_invocation_id: {actionmod.get('action_invocation_id')}",
        ]
    else:
        rows = [
            f"schema: {episode.get('schema')}",
            f"status: {episode.get('status')}",
            f"round_count: {len(episode.get('rounds') or [])}",
            f"memory_id: {episode.get('memory_id')}",
        ]
    return text_slide(
        path,
        title=kind.replace("_", " ").title(),
        subtitle=f"level {level_index} action {action_index:03d} / {packet.get('case_id')}",
        sections=[("Recorded Packet Summary", rows), ("Boundary", ["Packet is review/control feedback only, not solve evidence."])],
    )


def collect_lineup_matrix_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if current.get("schema") == "codexlab.geoband_portband_lineup_matrix.v1":
                rows.extend([row for row in current.get("rows", []) or [] if isinstance(row, dict)])
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return rows


def portband_action_summary(packet: dict[str, Any], *, level_index: int, action_index: int) -> dict[str, Any]:
    selected = packet.get("selected_recommendation") if isinstance(packet.get("selected_recommendation"), dict) else {}
    refs = selected.get("evidence_refs") if isinstance(selected.get("evidence_refs"), dict) else {}
    signals = selected.get("signals") if isinstance(selected.get("signals"), dict) else {}
    sources = [str(value) for value in (selected.get("evidence_sources") or [])]
    return {
        "level_index": level_index,
        "action_index": action_index,
        "case_id": packet.get("case_id"),
        "primitive": packet.get("decision_primitive"),
        "candidate_kind": selected.get("candidate_kind"),
        "value_score": selected.get("value_score"),
        "evidence_sources": sources,
        "lineup_matrix_row_id": refs.get("lineup_matrix_row_id"),
        "portband_route_key": refs.get("portband_route_key"),
        "active_portband_route_key": refs.get("active_portband_route_key"),
        "blocked_portband_route_keys": refs.get("blocked_portband_route_keys") or [],
        "source_component_ids": refs.get("source_component_ids") or [],
        "target_component_ids": refs.get("target_component_ids") or [],
        "target_port_ids": refs.get("target_port_ids") or [],
        "linked_alignment_ids": refs.get("linked_alignment_ids") or [],
        "edge_lock_top_configuration_id": refs.get("edge_lock_top_configuration_id"),
        "edge_lock_kind": refs.get("edge_lock_kind"),
        "edge_lock_rank": refs.get("edge_lock_rank"),
        "edge_lock_status": signals.get("edge_lock_status"),
        "edge_lock_quality_score": signals.get("edge_lock_quality_score"),
        "transform_hypothesis_xy": signals.get("transform_hypothesis_xy"),
        "alignment_kind": signals.get("alignment_kind"),
        "movement_oscillation_hard_blocked": signals.get("movement_oscillation_hard_blocked"),
        "movement_oscillation_prior_pair_count": signals.get("movement_oscillation_prior_pair_count"),
        "repeat_no_progress_penalty": signals.get("repeat_no_progress_penalty"),
        "source_lineup_matrix_row_count": len(collect_lineup_matrix_rows(packet.get("contra_packet") or {})),
        "selected_from_portband": (
            "portband.lineup_matrix.rows" in sources
            or "portband_signal_scores" in sources
            or any("portband" in source for source in sources)
        ),
        "levels_completed_delta": packet.get("levels_completed_delta"),
        "goal_distance_delta": (packet.get("goal_progress") or {}).get("distance_delta"),
    }


def portband_summary_slide(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    level_index: int,
    packets: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    packets = packets or []
    selected_count = sum(1 for row in rows if row.get("selected_from_portband"))
    lineup_count = sum(1 for row in rows if "portband.lineup_matrix.rows" in (row.get("evidence_sources") or []))
    completion_delta = sum(int(row.get("levels_completed_delta") or 0) for row in rows)
    img, draw = base_slide("PortBand Selected Recommendations", f"level {level_index:02d}")
    draw.text((44, 122), "Gameplay Frames", fill=(140, 200, 255), font=FONT_30)
    first_before = frame_path(packets[0], "before") if packets else None
    paste_frame(img, first_before, (44, 168, 604, 728))
    draw.rectangle([626, 168, 1554, 728], outline=(38, 51, 69), width=2)
    tile_w = 156
    tile_h = 156
    x0 = 642
    y0 = 194
    for idx, packet in enumerate(packets[:6]):
        col = idx % 3
        row = idx // 3
        x = x0 + col * 292
        y = y0 + row * 248
        before = frame_path(packet, "before")
        after = frame_path(packet, "after")
        paste_frame(img, before, (x, y, x + tile_w, y + tile_h))
        paste_frame(img, after, (x + 132, y, x + 132 + tile_w, y + tile_h))
        draw.text((x, y - 22), f"a{idx:03d}", fill=(226, 232, 240), font=FONT_18)
        draw.text((x, y + tile_h + 3), "before", fill=(148, 163, 184), font=FONT_18)
        draw.text((x + 132, y + tile_h + 3), "after", fill=(148, 163, 184), font=FONT_18)

    summary_lines = [
        f"actions: {len(rows)}",
        f"selected_from_portband: {selected_count}",
        f"selected_from_lineup_matrix: {lineup_count}",
        f"levels_completed_delta: {completion_delta}",
    ]
    y = 760
    draw.text((48, y), "Summary", fill=(140, 200, 255), font=FONT_30)
    y += 40
    y = draw_lines(draw, 64, y, summary_lines, fill=(226, 232, 240), fnt=FONT_22, step=31)
    draw.text((520, 760), "Selected Rows", fill=(140, 200, 255), font=FONT_30)
    selected_lines: list[str] = []
    for row in rows[:14]:
        sources = "+".join(row.get("evidence_sources") or [])
        selected_lines.append(
            "a{action_index:03d} {primitive} score={value_score} row={lineup_matrix_row_id} "
            "blocked={movement_oscillation_hard_blocked} delta={levels_completed_delta} sources={sources}".format(
                sources=sources[:58],
                **row,
            )
        )
    if len(rows) > 14:
        selected_lines.append(f"... {len(rows) - 14} more actions in report_manifest.portband_table")
    y = 802
    for line in selected_lines[:2]:
        y = draw_lines(draw, 536, y, wrap_text(line, 72)[:2], fill=(226, 232, 240), fnt=FONT_18, step=24)
    if len(selected_lines) > 2:
        draw.text((536, min(y + 4, 908)), f"... {len(selected_lines) - 2} more selected rows in report_manifest.portband_table", fill=(148, 163, 184), font=FONT_18)
    return save_slide(path, img)


def sequence_slide(path: Path, *, packet: dict[str, Any], level_index: int, action_index: int) -> dict[str, Any]:
    img, draw = base_slide("Action Sequence", f"level {level_index} action {action_index:03d} / {packet.get('case_id')}")
    seq = packet.get("action_png_sequence")
    if not isinstance(seq, list) or not seq:
        return text_slide(
            path,
            title="Action Sequence",
            subtitle=f"level {level_index} action {action_index:03d} / {packet.get('case_id')}",
            sections=[("Sequence", ["No multi-frame action sequence recorded for this action."])],
        )
    x, y = 42, 130
    tile = 148
    for idx, entry in enumerate(seq[:40]):
        png = resolve_repo_path(str(entry["path"]))
        frame = Image.open(png).convert("RGB").resize((tile, tile), Image.Resampling.NEAREST)
        cx = x + (idx % 8) * 190
        cy = y + (idx // 8) * 174
        img.paste(frame, (cx, cy + 24))
        draw.text((cx, cy), f"frame {idx:03d}", fill=(226, 232, 240), font=FONT_18)
    return save_slide(path, img)


def add_event(events: list[dict[str, Any]], **kwargs: Any) -> None:
    events.append({"schema": TRACE_SCHEMA, "event_index": len(events), **kwargs})


def build_report(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    slides_dir = out_dir / "slides"
    assets_dir = out_dir / "assets" / "pass_images"
    assets_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "summary.json"
    single_result_path = run_dir / "result.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        summary_rows = list(summary.get("rows", []))
        if summary.get("schema") == "codexlab.sp80_contra_episode_levels.v1":
            levels = summary.get("requested_levels") or [
                row.get("level_index") for row in summary_rows if isinstance(row, dict)
            ]
            level_label = ",".join(str(level) for level in levels)
            run_label = f"SP80 CONTRA persistent episode levels {level_label}"
            run_id = f"contra_sp80_episode_levels_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            persistent_boundary = "Persistent episode controlled_run: CONTRA recommendation packets drive the player harness on live SP80 state."
        else:
            run_label = "SP80 connected player harness levels 0 and 1"
            run_id = f"contra_sp80_connected_harness_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            persistent_boundary = "Connected control harness evidence; not a persistent solve-sequence policy."
    elif single_result_path.is_file():
        single_result = read_json(single_result_path)
        summary = {
            "status": single_result.get("status"),
            "rows": [
                {
                    "run_dir": rel(run_dir),
                    "verifier_returncode": 0,
                    "harness_stdout_tail": single_result.get("condition", {}).get("command", ""),
                }
            ],
        }
        summary_rows = list(summary["rows"])
        run_label = f"SP80 persistent CONTRA episode level {int(single_result.get('level_index') or 0):02d}"
        run_id = f"contra_sp80_contra_episode_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        persistent_boundary = "Persistent episode: no reset between actions; CONTRA recommendation packet drives the harness brain."
    else:
        raise FileNotFoundError(f"{run_dir} has neither summary.json nor result.json")
    events: list[dict[str, Any]] = []
    slide_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    brain_rows: list[dict[str, Any]] = []
    portband_rows: list[dict[str, Any]] = []
    raw_inputs: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []

    source_paths = [
        "codexlab/gameplay_report_harness.py",
        "codexlab/sp80_run_contra_levels.py",
        "codexlab/sp80_sweep_levels.py",
        "codexlab/sp80_player_harness.py",
        "codexlab/sp80_verify_run.py",
        "codexlab/sp80_verify_episode_run.py",
        "codexlab/players/base.py",
        "codexlab/players/sp80/controller.py",
        "codexlab/players/sp80/registry.py",
    ]
    source_refs = [{"path": p, "sha256": sha256_file(ROOT / p)} for p in source_paths if (ROOT / p).is_file()]
    command_lines = [
        str(row.get("harness_command") or row.get("harness_stdout_tail") or "").strip()
        for row in summary_rows
        if str(row.get("harness_command") or row.get("harness_stdout_tail") or "").strip()
    ]
    verifier_commands = [
        str(row.get("verifier_command") or "").strip()
        for row in summary_rows
        if str(row.get("verifier_command") or "").strip()
    ]

    run_condition = text_slide(
        slides_dir / "000_run_condition.png",
        title="Run Condition",
        subtitle=run_label,
        sections=[
            ("Run", [f"run_id: {run_id}", "category: controlled_run", f"source run: {rel(run_dir)}"]),
            ("Command", command_lines or ["See summary.json rows for exact harness and verifier commands."]),
            ("Boundary", [persistent_boundary, "Oracle sidecar count is explicitly 0."]),
        ],
    )
    slide_rows.append({"slide_type": "run_condition", **run_condition})
    add_event(events, run_id=run_id, event_type="function_slide", function_name="run_condition", function_family="provenance", output_refs=[run_condition], status="rendered")

    verifier_status = "passed"
    level_dirs: list[Path] = []
    for row in summary_rows:
        level_dirs.append(resolve_repo_path(row["run_dir"]))
        if row.get("verifier_returncode") != 0:
            verifier_status = "failed"

    for level_dir in level_dirs:
        result = read_json(level_dir / "result.json")
        packets = read_jsonl(level_dir / "packets.jsonl")
        calls = read_jsonl(level_dir / "brain_calls.jsonl")
        level_index = int(result.get("level_index") if result.get("level_index") is not None else (result.get("condition") or {}).get("level_index"))
        ghost_path = level_dir / "ghost.json"
        ghost_packet = read_json(ghost_path) if ghost_path.is_file() else None
        raw_outputs.extend(
            [
                {"path": rel(level_dir / "result.json"), "sha256": sha256_file(level_dir / "result.json")},
                {"path": rel(level_dir / "packets.jsonl"), "sha256": sha256_file(level_dir / "packets.jsonl")},
                {"path": rel(level_dir / "brain_calls.jsonl"), "sha256": sha256_file(level_dir / "brain_calls.jsonl")},
            ]
        )
        if ghost_path.is_file():
            raw_outputs.append({"path": rel(ghost_path), "sha256": sha256_file(ghost_path)})

        if packets:
            first_before = frame_path(packets[0], "before")
            raw = text_slide(
                slides_dir / f"010_level_{level_index:02d}_raw_frame.png",
                title="Raw Frame",
                subtitle=f"level {level_index:02d} first before-action frame",
                sections=[
                    ("Frame", [f"source: {rel(first_before) if first_before else 'missing'}"]),
                    ("Boundary", ["Raw environment render from connected harness packet."]),
                ],
            )
            if first_before:
                img = Image.open(raw["path"] if Path(raw["path"]).is_absolute() else ROOT / raw["path"]).convert("RGB")
                draw = ImageDraw.Draw(img)
                paste_frame(img, first_before, (860, 150, 1500, 790))
                img.save(ROOT / raw["path"])
                raw["sha256"] = sha256_file(ROOT / raw["path"])
            slide_rows.append({"slide_type": "raw_frame", "level_index": level_index, **raw})
            add_event(events, run_id=run_id, event_type="function_slide", level_index=level_index, function_name="raw_frame", function_family="game_action", output_refs=[raw], status="rendered")

        if packets:
            for spec_index, (stage_id, stage_label, family, packet_key, layer_id) in enumerate(ENGINE_MECHANIC_SPECS):
                mechanic = engine_mechanic_slide(
                    slides_dir / f"{20 + spec_index:03d}_level_{level_index:02d}_{stage_id}.png",
                    packet=packets[0],
                    level_index=level_index,
                    stage_id=stage_id,
                    stage_label=stage_label,
                    packet_key=packet_key,
                    layer_id=layer_id,
                )
                slide_rows.append({"slide_type": stage_id, "level_index": level_index, **mechanic})
                add_event(
                    events,
                    run_id=run_id,
                    event_type="function_slide",
                    level_index=level_index,
                    function_name=stage_id,
                    function_family=family,
                    output_refs=[mechanic],
                    admission_status="not_admitted",
                    truth_boundary="contra_mechanic_review_image_not_action_authority",
                    status="rendered",
                )
            oracle = oracle_pass_slide(
                slides_dir / f"030_level_{level_index:02d}_oracle_pass.png",
                packet=packets[0],
                level_index=level_index,
            )
            slide_rows.append({"slide_type": "oracle_pass", "level_index": level_index, **oracle})
            add_event(
                events,
                run_id=run_id,
                event_type="function_slide",
                level_index=level_index,
                function_name="oracle_pass",
                function_family="oracle",
                output_refs=[oracle],
                admission_status="not_admitted",
                truth_boundary="oracle_review_hypothesis_not_control_truth",
                status="rendered",
            )
            ghost_slide = ghost_pass_slide(
                slides_dir / f"040_level_{level_index:02d}_ghost_pass.png",
                packet=packets[0],
                level_index=level_index,
                ghost=ghost_packet,
            )
            slide_rows.append({"slide_type": "ghost_memory" if ghost_packet else "ghost_pass", "level_index": level_index, **ghost_slide})
            add_event(
                events,
                run_id=run_id,
                event_type="function_slide",
                level_index=level_index,
                function_name="ghost_memory" if ghost_packet else "ghost_pass",
                function_family="ghost",
                output_refs=[ghost_slide],
                admission_status="not_admitted",
                truth_boundary="ghost_memory_not_future_truth",
                status="rendered",
            )
            relationship_graph = relationship_graph_slide(
                slides_dir / f"060_level_{level_index:02d}_relationship_graph.png",
                packet=packets[0],
                level_index=level_index,
            )
            slide_rows.append({"slide_type": "relationship_graph", "level_index": level_index, **relationship_graph})
            add_event(
                events,
                run_id=run_id,
                event_type="function_slide",
                level_index=level_index,
                function_name="relationship_graph",
                function_family="relationships",
                output_refs=[relationship_graph],
                admission_status="not_admitted",
                truth_boundary="relationship_hypothesis_not_action_authority",
                status="rendered",
            )
            relationship_ladder = relationship_ladder_slide(
                slides_dir / f"070_level_{level_index:02d}_relationship_ladder.png",
                packet=packets[0],
                level_index=level_index,
            )
            slide_rows.append({"slide_type": "relationship_ladder", "level_index": level_index, **relationship_ladder})
            add_event(
                events,
                run_id=run_id,
                event_type="function_slide",
                level_index=level_index,
                function_name="relationship_ladder",
                function_family="relationships",
                output_refs=[relationship_ladder],
                admission_status="not_admitted",
                truth_boundary="relationship_ladder_not_control_goal_prediction_or_solve_truth",
                status="rendered",
            )

        level_portband_rows = [
            portband_action_summary(packet, level_index=level_index, action_index=action_index)
            for action_index, packet in enumerate(packets)
        ]
        portband_rows.extend(level_portband_rows)
        pb = portband_summary_slide(
            slides_dir / f"120_level_{level_index:02d}_portband_selected_recommendations.png",
            rows=level_portband_rows,
            level_index=level_index,
            packets=packets,
        )
        slide_rows.append({"slide_type": "portband_selected_recommendations", "level_index": level_index, **pb})
        add_event(
            events,
            run_id=run_id,
            event_type="function_slide",
            level_index=level_index,
            function_name="portband_selected_recommendations",
            function_family="portband",
            output_refs=[pb],
            admission_status="not_admitted",
            truth_boundary="portband_review_route_hints_not_action_authority",
            status="rendered",
        )

        for action_index, (packet, call) in enumerate(zip(packets, calls)):
            action_id = f"{run_id}_level_{level_index:02d}_action_{action_index:03d}"
            action_oracle = oracle_pass_slide(
                slides_dir / f"160_level_{level_index:02d}_action_{action_index:03d}_oracle_pass.png",
                packet=packet,
                level_index=level_index,
            )
            action_ghost = ghost_pass_slide(
                slides_dir / f"161_level_{level_index:02d}_action_{action_index:03d}_ghost_pass.png",
                packet=packet,
                level_index=level_index,
                ghost=None,
            )
            action_relationship_graph = relationship_graph_slide(
                slides_dir / f"162_level_{level_index:02d}_action_{action_index:03d}_relationship_graph.png",
                packet=packet,
                level_index=level_index,
            )
            action_relationship_ladder = relationship_ladder_slide(
                slides_dir / f"163_level_{level_index:02d}_action_{action_index:03d}_relationship_ladder.png",
                packet=packet,
                level_index=level_index,
            )
            action_portband = portband_summary_slide(
                slides_dir / f"164_level_{level_index:02d}_action_{action_index:03d}_portband_selected_recommendation.png",
                rows=[portband_action_summary(packet, level_index=level_index, action_index=action_index)],
                level_index=level_index,
                packets=[packet],
            )
            action_passes: list[tuple[str, str, dict[str, Any], str]] = [
                ("oracle_pass", "oracle", action_oracle, "oracle_review_hypothesis_not_control_truth"),
                ("ghost_pass", "ghost", action_ghost, "ghost_pass_not_future_truth"),
                ("relationship_graph", "relationships", action_relationship_graph, "relationship_hypothesis_not_action_authority"),
                ("relationship_ladder", "relationships", action_relationship_ladder, "relationship_ladder_not_control_goal_prediction_or_solve_truth"),
                ("portband_selected_recommendations", "portband", action_portband, "portband_review_route_hints_not_action_authority"),
            ]
            action_mechanics: list[tuple[str, str, dict[str, Any]]] = []
            for spec_index, (stage_id, stage_label, family, packet_key, layer_id) in enumerate(ENGINE_MECHANIC_SPECS):
                mechanic = engine_mechanic_slide(
                    slides_dir / f"180_level_{level_index:02d}_action_{action_index:03d}_{spec_index:02d}_{stage_id}.png",
                    packet=packet,
                    level_index=level_index,
                    stage_id=stage_id,
                    stage_label=stage_label,
                    packet_key=packet_key,
                    layer_id=layer_id,
                )
                action_mechanics.append((stage_id, family, mechanic))
            brain = brain_slide(slides_dir / f"200_level_{level_index:02d}_brain_{action_index:03d}.png", call=call, level_index=level_index, action_index=action_index)
            overlay = action_overlay_slide(slides_dir / f"250_level_{level_index:02d}_action_{action_index:03d}_overlay.png", packet=packet, level_index=level_index, action_index=action_index)
            before = action_slide(slides_dir / f"300_level_{level_index:02d}_action_{action_index:03d}_before.png", title="Action Before", packet=packet, frame_phase="before", level_index=level_index, action_index=action_index)
            after = action_slide(slides_dir / f"301_level_{level_index:02d}_action_{action_index:03d}_after.png", title="Action After", packet=packet, frame_phase="after", level_index=level_index, action_index=action_index)
            delta = delta_slide(slides_dir / f"302_level_{level_index:02d}_action_{action_index:03d}_delta.png", packet=packet, level_index=level_index, action_index=action_index)
            seq = sequence_slide(slides_dir / f"303_level_{level_index:02d}_action_{action_index:03d}_sequence.png", packet=packet, level_index=level_index, action_index=action_index)
            db = memory_slide(slides_dir / f"400_level_{level_index:02d}_action_{action_index:03d}_deltaband.png", packet=packet, level_index=level_index, action_index=action_index, kind="deltaband")
            am = memory_slide(slides_dir / f"410_level_{level_index:02d}_action_{action_index:03d}_actionmodband.png", packet=packet, level_index=level_index, action_index=action_index, kind="actionmodband")
            em = memory_slide(slides_dir / f"420_level_{level_index:02d}_action_{action_index:03d}_episode_memory.png", packet=packet, level_index=level_index, action_index=action_index, kind="episode_memory")
            for slide_type, family, slide, boundary in action_passes:
                slide_rows.append({"slide_type": slide_type, "level_index": level_index, "action_index": action_index, **slide})
                add_event(
                    events,
                    run_id=run_id,
                    event_type="function_slide",
                    level_index=level_index,
                    action_invocation_id=action_id,
                    function_name=stage_id_for_slide(slide_type),
                    function_family=family,
                    output_refs=[slide],
                    admission_status="not_admitted",
                    truth_boundary=boundary,
                    status="rendered",
                )
            for slide_type, family, slide in action_mechanics:
                slide_rows.append({"slide_type": slide_type, "level_index": level_index, "action_index": action_index, **slide})
                add_event(
                    events,
                    run_id=run_id,
                    event_type="function_slide",
                    level_index=level_index,
                    action_invocation_id=action_id,
                    function_name=slide_type,
                    function_family=family,
                    output_refs=[slide],
                    admission_status="not_admitted",
                    truth_boundary="per_action_contra_mechanic_review_image_not_action_authority",
                    status="rendered",
                )
            for slide_type, slide in [
                ("brain_decision", brain),
                ("action_overlay", overlay),
                ("action_before", before),
                ("action_after", after),
                ("action_delta", delta),
                ("action_sequence", seq),
                ("deltaband", db),
                ("actionmodband", am),
                ("episode_memory", em),
            ]:
                slide_rows.append({"slide_type": slide_type, "level_index": level_index, "action_index": action_index, **slide})
                add_event(
                    events,
                    run_id=run_id,
                    event_type="function_slide",
                    level_index=level_index,
                    action_invocation_id=action_id,
                    function_name=slide_type,
                    function_family="game_action" if slide_type.startswith("action") else "brain" if slide_type == "brain_decision" else "memory_control_feedback",
                    output_refs=[slide],
                    admission_status="not_admitted",
                    truth_boundary="controlled_action_observation_not_solve_truth",
                    status="rendered",
                )
            action_rows.append(
                {
                    "level_index": level_index,
                    "action_index": action_index,
                    "case_id": packet.get("case_id"),
                    "status": packet.get("status"),
                    "executed": packet.get("status") == "executed",
                    "brain_primitive": (packet.get("brain_decision") or {}).get("primitive"),
                    "expected_primitive": packet.get("expected_primitive"),
                    "action_name": (packet.get("action") or {}).get("name"),
                    "click_xy": packet_click_xy(packet),
                    "has_recorded_click": packet_click_xy(packet) is not None,
                    "returned_frame_count": packet_returned_frame_count(packet) if packet.get("status") == "executed" else 0,
                    "expected_gameplay_frame_count": 2 + packet_returned_frame_count(packet)
                    if packet.get("status") == "executed"
                    else 1
                    if frame_path(packet, "before") is not None
                    else 0,
                    "contra_layer_visual_map": layer_output_visual_map(packet),
                    "hidden_layer_output_ids": [
                        row["layer_id"]
                        for row in layer_output_visual_map(packet)
                        if row.get("hidden") is True
                    ],
                    "changed_pixels": (((packet.get("action_visual_correlation") or {}).get("frame_delta") or {}).get("changed_pixels") or (packet.get("delta") or {}).get("changed_pixels")),
                    "levels_completed_delta": int((packet.get("after_state") or {}).get("levels_completed") or 0) - int((packet.get("before_state") or {}).get("levels_completed") or 0),
                }
            )
            brain_rows.append(
                {
                    "level_index": level_index,
                    "action_index": action_index,
                    "case_id": call.get("case_id"),
                    "primitive": (call.get("decision") or {}).get("primitive"),
                    "confidence": (call.get("decision") or {}).get("confidence"),
                    "latency_ms": (call.get("decision") or {}).get("latency_ms"),
                    "error": (call.get("decision") or {}).get("error"),
                }
            )
            for phase in ("before", "after"):
                path = frame_path(packet, phase)
                if path:
                    raw_inputs.append({"path": rel(path), "sha256": sha256_file(path)})

    provenance = text_slide(
        slides_dir / "900_provenance.png",
        title="Provenance",
        subtitle="Commands, hashes, verification",
        sections=[
            ("Source", [f"git_commit: {git_commit()}", f"git_dirty: {git_dirty()}", f"host: {socket.gethostname()}", f"cwd: {ROOT}"]),
            ("Verification", [f"sweep status: {summary.get('status')}", f"verifier status: {verifier_status}", f"slide count: {len(slide_rows) + 1}", "oracle sidecar count: 0"]),
            ("Boundary", ["Controlled harness evidence only; solve evidence requires levels_completed increase in the recorded run."]),
        ],
    )
    slide_rows.append({"slide_type": "provenance", **provenance})
    add_event(events, run_id=run_id, event_type="verifier", function_name="provenance", function_family="verifier", output_refs=[provenance], status="rendered")

    trace_path = out_dir / "report_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")

    manifest_path = out_dir / "report_manifest.json"
    index_path = out_dir / "index.html"
    legacy_index_path = out_dir / "legacy_deck.html"
    screenshot_path = out_dir / "screenshots" / "index_desktop.png"
    manifest = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report_status": "controlled_run",
        "run_id": run_id,
        "game_id": "sp80-589a99af",
        "working_directory": str(ROOT),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "command": str(summary_rows[0].get("harness_stdout_tail") or "See source result.json condition.command"),
        "commands": command_lines,
        "verifier_commands": verifier_commands,
        "source_run_dir": rel(run_dir),
        "source_paths": source_refs,
        "raw_inputs": dedupe_refs(raw_inputs),
        "raw_outputs": raw_outputs + [{"path": rel(trace_path), "sha256": sha256_file(trace_path)}],
        "index_html": rel(index_path),
        "legacy_deck_html": rel(legacy_index_path),
        "report_manifest_json": rel(manifest_path),
        "report_trace_jsonl": rel(trace_path),
        "slide_count": len(slide_rows),
        "function_slide_count": len(slide_rows),
        "action_count": len(action_rows),
        "brain_call_count": len(brain_rows),
        "portband_selected_count": sum(1 for row in portband_rows if row.get("selected_from_portband")),
        "portband_table": portband_rows,
        "oracle_sidecar_count": 0,
        "level_count": len(level_dirs),
        "levels_completed_delta_total": sum(int(row["levels_completed_delta"]) for row in action_rows),
        "verification": {
            "image_count_matches_manifest": True,
            "slide_count_matches_png_ledger": True,
            "gameplay_visual_contract_required": True,
            "image_first_review": True,
            "sha256_path_refs_checked": False,
            "index_reachable": False,
            "browser_screenshot_checked": False,
            "screenshot_path": rel(screenshot_path),
            "verifier_status": verifier_status,
        },
        "slide_ledger": slide_rows,
        "action_timeline": action_rows,
        "brain_table": brain_rows,
        "boundaries": [
            "Controlled harness action evidence only.",
            "This run is not known-solution playback.",
            persistent_boundary,
            "Solve evidence requires recorded levels_completed increase.",
            "Oracle sidecar count is 0.",
        ],
    }
    write_json(manifest_path, manifest)
    legacy_index_path.write_text(build_index(manifest, action_rows, brain_rows, portband_rows, out_dir=out_dir), encoding="utf-8")
    report_path, viewer_url = write_contraview_report(
        run_dir=run_dir,
        out_dir=out_dir,
        manifest=manifest,
        events=events,
        copy_to_angel=True,
    )
    manifest["contraview_report_json"] = rel(report_path)
    manifest["angel_viewer_url"] = viewer_url
    manifest["raw_outputs"].extend(
        [
            {"path": rel(report_path), "sha256": sha256_file(report_path)},
            {"path": rel(legacy_index_path), "sha256": sha256_file(legacy_index_path)},
        ]
    )
    manifest["verification"]["contraview_report_present"] = True
    manifest["verification"]["primary_index_is_angel_viewer_pointer"] = True
    write_json(manifest_path, manifest)
    for base in (ANGEL_DIST_DATA, ANGEL_PUBLIC_DATA):
        dest_manifest = base / str(manifest["run_id"]) / "report_manifest.json"
        if dest_manifest.parent.is_dir():
            shutil.copy2(manifest_path, dest_manifest)
    return manifest


def write_contraview_report(
    *,
    run_dir: Path,
    out_dir: Path,
    manifest: dict[str, Any],
    events: list[dict[str, Any]],
    copy_to_angel: bool,
) -> tuple[Path, str]:
    run_id = str(manifest["run_id"])
    asset_dir = out_dir / "angel_assets"
    if asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, Any]] = []
    for idx, slide in enumerate(manifest.get("slide_ledger") or []):
        if not isinstance(slide, dict):
            continue
        source_path = resolve_repo_path(str(slide["path"]))
        stage_id = stage_id_for_slide(str(slide.get("slide_type") or "provenance"))
        stage_label, family = STAGE_LABELS.get(stage_id, (stage_id.replace("_", " ").title(), "unclassified"))
        asset_name = f"{idx:04d}_{source_path.name}"
        copied = asset_dir / asset_name
        shutil.copy2(source_path, copied)
        source_sha = sha256_file(source_path)
        frame = {
            "frame_id": f"{run_id}_{idx:04d}",
            "stage_id": stage_id,
            "stage_label": stage_label,
            "timeline_index": idx,
            "level_index": slide.get("level_index") if isinstance(slide.get("level_index"), int) else None,
            "action_index": slide.get("action_index") if isinstance(slide.get("action_index"), int) else None,
            "caption": str(slide.get("slide_type") or stage_label).replace("_", " ").title(),
            "family": family,
            "artifact_kind": "png_image",
            "image_url": f"/data/{run_id}/assets/{asset_name}",
            "source_path": rel(source_path),
            "source_sha256": source_sha,
            "evidence_category": "controlled_run",
            "boundary": "Controlled harness action evidence only. This frame is not solve evidence unless levels_completed increased in the recorded run.",
        }
        for key in ("click_xy", "has_click_marker", "action_name"):
            if key in slide:
                frame[key] = slide[key]
        frames.append(frame)

    gameplay_frame_count = append_gameplay_frames(
        frames=frames,
        run_id=run_id,
        run_dir=run_dir,
        asset_dir=asset_dir,
    )

    grouped_sections = build_grouped_sections(frames)
    hidden_sections = [
        {
            "section_id": "tabula_rasa_notebook",
            "section_label": "Tabula Rasa Notebook",
            "stage_ids": ["tabula_rasa_notebook"],
            "frame_count": 0,
            "frames": [],
            "reason": "hidden_by_default_per_user_report_structure",
        }
    ]
    stage_defs = []
    for order, (stage_id, (stage_label, family)) in enumerate(STAGE_LABELS.items()):
        count = sum(1 for frame in frames if frame.get("stage_id") == stage_id)
        stage_defs.append(
            {
                "stage_id": stage_id,
                "stage_label": stage_label,
                "stage_order": order,
                "family": family,
                "frame_count": count,
                "status": "rendered" if count else "explicit_zero",
                "boundary": "Controlled harness review only. No control truth, goal truth, action choice, prediction truth, or solve evidence.",
            }
        )

    created = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    report = {
        "contra_oracle_view_test": {
            "schema": CONTRAVIEW_SCHEMA,
            "designation": "ANGEL-COV-001",
            "family": "contraview_report",
            "official_name": "contraview_report",
            "label": f"contraview_report · {run_id}",
            "status": "controlled_run",
            "generated_at": created,
            "asset_dir": f"/data/{run_id}/assets",
            "source_roots": {
                "rexus_root": str(ROOT),
                "source_run_dir": rel(run_dir),
                "report_dir": rel(out_dir),
                "angel_data": f"/data/{run_id}/contraview_report.json",
            },
            "boundary": "Controlled harness action evidence only. Not control truth, goal truth, action choice truth, prediction truth, or solve evidence unless levels_completed increased in the recorded run.",
            "layout_contract": {
                "layout_kind": "function_grouped_dropdown_sliders",
                "section_rule": "one dropdown per function group, one lateral image slider per dropdown",
                "no_mixed_lateral_functions": True,
                "portband_grouping": "aggregate",
                "portband_stage_ids": ["portband_alignment", "portband_fleet", "portband_lineup_matrix"],
                "hidden_stage_ids": sorted(HIDDEN_STAGE_IDS),
                "hidden_by_default": ["tabula_rasa_notebook", "autoresearch"],
                "image_first_review": True,
                "initial_review_section": "portband",
                "required_action_visual_stages": [
                    "brain_decision",
                    "action_overlay",
                    "gameplay_frame",
                    "action_before",
                    "action_after",
                    "action_delta",
                    "action_sequence",
                    "deltaband_gate",
                    "actionmodband",
                    "episode_memory",
                ],
                "required_level_visual_stages": [
                    "oracle_pass",
                    "ghost_pass_or_ghost_memory",
                    "relationship_graph",
                    "relationship_ladder",
                    "portband_alignment",
                ],
            },
        },
        "summary": [
            {"key": "source_run_dir", "value": rel(run_dir)},
            {"key": "frame_count", "value": len(frames)},
            {"key": "grouped_section_count", "value": len(grouped_sections)},
            {"key": "portband_alignment_images", "value": sum(1 for frame in frames if frame["stage_id"] == "portband_alignment")},
            {"key": "evidence_category", "value": "controlled_run"},
            {"key": "levels_completed_delta_total", "value": manifest.get("levels_completed_delta_total")},
            {"key": "gameplay_frame_count", "value": gameplay_frame_count},
        ],
        "stages": stage_defs,
        "grouped_sections": grouped_sections,
        "hidden_sections": hidden_sections,
        "frames": frames,
        "process_events": [
            {
                "schema": TRACE_SCHEMA,
                "event_order": idx,
                "surface_id": frame["stage_id"],
                "event_type": "frame_rendered",
                "caption": frame["caption"],
                "level_index": frame["level_index"],
                "output_refs": [{"path": frame["source_path"], "sha256": frame["source_sha256"], "url": frame["image_url"]}],
                "status": "rendered",
                "truth_boundary": "controlled_harness_review_not_solve_truth",
            }
            for idx, frame in enumerate(frames)
        ]
        + events,
    }

    report_path = out_dir / "contraview_report.json"
    write_json(report_path, report)
    viewer_url = write_viewer_index(out_dir / "index.html", run_id=run_id)

    if copy_to_angel:
        for base in (ANGEL_DIST_DATA, ANGEL_PUBLIC_DATA):
            dest = base / run_id
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(report_path, dest / "contraview_report.json")
            for filename in ("index.html", "report_manifest.json", "report_trace.jsonl", "legacy_deck.html"):
                source = out_dir / filename
                if source.is_file():
                    shutil.copy2(source, dest / filename)
            shutil.copytree(asset_dir, dest / "assets")
    return report_path, viewer_url


def append_gameplay_frames(
    *,
    frames: list[dict[str, Any]],
    run_id: str,
    run_dir: Path,
    asset_dir: Path,
) -> int:
    count = 0
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
        level_dirs = [
            resolve_repo_path(row["run_dir"])
            for row in summary.get("rows", []) or []
            if isinstance(row, dict) and row.get("run_dir")
        ]
    else:
        level_dirs = [run_dir]
    for level_dir in level_dirs:
        packets_path = level_dir / "packets.jsonl"
        if not packets_path.is_file():
            continue
        for packet in read_jsonl(packets_path):
            level_index = int(packet.get("level_index") if packet.get("level_index") is not None else (read_json(level_dir / "result.json").get("level_index") or 0))
            action_index = int(packet.get("action_index") or 0)
            candidates: list[tuple[str, int, Path]] = []
            for phase in ("before", "after"):
                path = frame_path(packet, phase)
                if path is not None:
                    candidates.append((phase, 0, path))
            seq = packet.get("action_png_sequence")
            if isinstance(seq, list):
                for idx, entry in enumerate(seq):
                    if isinstance(entry, dict) and entry.get("path"):
                        path = resolve_repo_path(str(entry["path"]))
                        if path.is_file():
                            candidates.append(("returned_frame", idx, path))
            for phase, frame_index, source_path in candidates:
                asset_name = f"{len(frames):04d}_level_{level_index:02d}_action_{action_index:03d}_{phase}_{frame_index:03d}_{source_path.name}"
                dest = asset_dir / asset_name
                shutil.copy2(source_path, dest)
                frames.append(
                    {
                        "frame_id": f"{run_id}_{len(frames):04d}",
                        "stage_id": "gameplay_frame",
                        "stage_label": "Gameplay Frame",
                        "timeline_index": len(frames),
                        "level_index": level_index,
                        "action_index": action_index,
                        "frame_index": frame_index,
                        "phase": phase,
                        "caption": f"Level {level_index:02d} action {action_index:03d} {phase} frame {frame_index:03d}",
                        "family": "gameplay",
                        "artifact_kind": "png_image",
                        "image_url": f"/data/{run_id}/assets/{asset_name}",
                        "source_path": rel(source_path),
                        "source_sha256": sha256_file(source_path),
                        "evidence_category": "controlled_run",
                        "boundary": "Raw gameplay frame from recorded harness artifact; not prediction or solve evidence by itself.",
                    }
                )
                count += 1
    return count


def dedupe_refs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("path"))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def link(path: str, out_dir: Path) -> str:
    target = ROOT / path
    return html.escape(os.path.relpath(target.resolve(), out_dir.resolve()))


def build_index(
    manifest: dict[str, Any],
    action_rows: list[dict[str, Any]],
    brain_rows: list[dict[str, Any]],
    portband_rows: list[dict[str, Any]],
    *,
    out_dir: Path,
) -> str:
    cards = [
        ("Status", manifest["report_status"]),
        ("Slides", manifest["slide_count"]),
        ("Actions", manifest["action_count"]),
        ("Brain calls", manifest["brain_call_count"]),
        ("PortBand selected", manifest["portband_selected_count"]),
        ("Oracle sidecars", manifest["oracle_sidecar_count"]),
        ("Completion delta", manifest["levels_completed_delta_total"]),
        ("Verifier", manifest["verification"]["verifier_status"]),
    ]
    card_html = "".join(f"<div class='card'><b>{html.escape(str(k))}</b><span>{html.escape(str(v))}</span></div>" for k, v in cards)
    slide_cards = []
    for slide in manifest["slide_ledger"]:
        slide_cards.append(
            "<figure>"
            f"<figcaption>{html.escape(str(slide.get('slide_type')))}"
            + (f" / level {slide.get('level_index')}" if slide.get("level_index") is not None else "")
            + (f" / action {slide.get('action_index')}" if slide.get("action_index") is not None else "")
            + "</figcaption>"
            f"<a href='{link(slide['path'], out_dir)}'><img src='{link(slide['path'], out_dir)}'></a>"
            "</figure>"
        )
    action_html = "".join(
        "<tr>"
        f"<td>{row['level_index']}</td><td>{row['action_index']}</td><td>{html.escape(str(row['case_id']))}</td>"
        f"<td>{html.escape(str(row['brain_primitive']))}</td><td>{html.escape(str(row['action_name']))}</td>"
        f"<td>{html.escape(str(row['changed_pixels']))}</td><td>{html.escape(str(row['levels_completed_delta']))}</td>"
        "</tr>"
        for row in action_rows
    )
    brain_html = "".join(
        "<tr>"
        f"<td>{row['level_index']}</td><td>{row['action_index']}</td><td>{html.escape(str(row['case_id']))}</td>"
        f"<td>{html.escape(str(row['primitive']))}</td><td>{html.escape(str(row['confidence']))}</td>"
        f"<td>{html.escape(str(row['latency_ms']))}</td><td>{html.escape(str(row['error']))}</td>"
        "</tr>"
        for row in brain_rows
    )
    portband_html = "".join(
        "<tr>"
        f"<td>{row['level_index']}</td><td>{row['action_index']}</td>"
        f"<td>{html.escape(str(row.get('primitive')))}</td>"
        f"<td>{html.escape(str(row.get('value_score')))}</td>"
        f"<td>{html.escape(str(row.get('candidate_kind')))}</td>"
        f"<td>{html.escape(str(row.get('lineup_matrix_row_id')))}</td>"
        f"<td>{html.escape(str(row.get('transform_hypothesis_xy')))}</td>"
        f"<td>{html.escape(str(row.get('blocked_portband_route_keys')))}</td>"
        f"<td>{html.escape(str(row.get('movement_oscillation_hard_blocked')))}</td>"
        f"<td>{html.escape(str(row.get('levels_completed_delta')))}</td>"
        f"<td>{html.escape('+'.join(row.get('evidence_sources') or []))}</td>"
        "</tr>"
        for row in portband_rows
    )
    source_links = [
        ("manifest", manifest["report_manifest_json"]),
        ("trace", manifest["report_trace_jsonl"]),
        ("source run", manifest["source_run_dir"]),
    ]
    sources = " | ".join(f"<a href='{link(path, out_dir)}'>{html.escape(label)}</a>" for label, path in source_links)
    boundary_text = " ".join(str(row) for row in manifest.get("boundaries", []))
    command_html = "<br>".join(
        f"<code>{html.escape(str(command))}</code>"
        for command in (manifest.get("commands") or [manifest.get("command")])
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CONTRA SP80 Connected Harness Angel Report</title>
<style>
body{{margin:0;background:#0b0f15;color:#e8edf4;font-family:Inter,system-ui,Segoe UI,sans-serif}}
main{{max-width:1700px;margin:0 auto;padding:28px}}
a{{color:#8cc8ff}}
.boundary{{background:#151c24;border-left:4px solid #f0b429;padding:10px 12px;color:#ffe8a3}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:18px 0}}
.card{{background:#111923;border:1px solid #263345;border-radius:6px;padding:12px}}
.card b{{display:block;font-size:12px;color:#94a3b8;text-transform:uppercase}}
.card span{{font-size:22px}}
.deck{{display:grid;grid-template-columns:1fr;gap:22px;margin-top:18px}}
figure{{margin:0;background:#101722;border:1px solid #263345;border-radius:6px;padding:12px}}
figcaption{{color:#cbd5e1;margin-bottom:8px}}
img{{width:100%;height:auto;image-rendering:pixelated}}
table{{width:100%;border-collapse:collapse;margin-top:18px;font-size:14px}}
th,td{{border-bottom:1px solid #263345;padding:8px;text-align:left;vertical-align:top}}
th{{background:#111923;color:#cbd5e1}}
code{{color:#d7e3f4}}
</style>
</head>
<body>
<main>
<h1>CONTRA SP80 Connected Harness Angel Report</h1>
<p class="boundary">Category: controlled_run. {html.escape(boundary_text)}</p>
<p>{sources}</p>
<section class="cards">{card_html}</section>
<h2>Visual Deck</h2>
<section class="deck">{''.join(slide_cards)}</section>
<h2>PortBand Selected Recommendations</h2>
<table><thead><tr><th>Level</th><th>#</th><th>Primitive</th><th>Score</th><th>Kind</th><th>Lineup row</th><th>Transform</th><th>Blocked routes</th><th>Oscillation blocked</th><th>Completion delta</th><th>Sources</th></tr></thead><tbody>{portband_html}</tbody></table>
<h2>Gameplay Action Timeline</h2>
<table><thead><tr><th>Level</th><th>#</th><th>Case</th><th>Brain primitive</th><th>Action</th><th>Changed pixels</th><th>Completion delta</th></tr></thead><tbody>{action_html}</tbody></table>
<h2>Brain Calls</h2>
<table><thead><tr><th>Level</th><th>#</th><th>Case</th><th>Primitive</th><th>Confidence</th><th>Latency ms</th><th>Error</th></tr></thead><tbody>{brain_html}</tbody></table>
<h2>Provenance</h2>
<p>{command_html}</p>
<p>git_commit: <code>{html.escape(str(manifest.get('git_commit')))}</code>; git_dirty: <code>{html.escape(str(manifest.get('git_dirty')))}</code></p>
</main>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = resolve_repo_path(args.run_dir)
    out_dir = resolve_repo_path(args.out_dir) if args.out_dir else run_dir / "angel_report"
    manifest = build_report(run_dir, out_dir)
    print(json.dumps({"status": "reported", "index_html": manifest["index_html"], "manifest": manifest["report_manifest_json"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
