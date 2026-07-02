"""Mine activate_flow transition rows for offline Crucible activation modeling."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .canon import SUPPORTED_ROTATION_QUARTERS, canonicalize_grid, display_coord
from .corpus import (
    DEFAULT_ADMITTED_MANIFEST,
    ROOT,
    action_signature,
    grid_ref,
    load_admitted_runs,
    packet_paths,
    primitive_from_packet,
    read_jsonl,
    rel,
    resolve_repo_path,
    sha256_file,
    source_hashes,
    stable_id,
)


DEFAULT_ACTIVATION_DIR = ROOT / "codexlab" / "datasets" / "sp80_crucible_activation_transitions_v0"
ACTIVATION_SCHEMA = "codexlab.crucible_activation_transition.v1"
ACTIVATION_CORPUS_SCHEMA = "codexlab.crucible_activation_transition_corpus.v1"
ACTIVATION_BOUNDARY_LITERAL = "crucible_activation_transition_is_observation_evidence_not_control_truth"


def truth_boundary() -> dict[str, Any]:
    return {
        "control_truth_admitted": False,
        "goal_truth_admitted": False,
        "goal_condition_admitted": False,
        "solve_truth_admitted": False,
        "solve_condition_admitted": False,
        "action_choice_admitted": False,
        "route_or_replay_used": False,
        "coordinate_copy_used": False,
        "boundary": ACTIVATION_BOUNDARY_LITERAL,
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _pair(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a0, a1 = value
    if not isinstance(a0, (int, float)) or not isinstance(a1, (int, float)):
        return None
    return [int(a0), int(a1)]


def _shape(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)):
        return None
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return None


def _state_selection(state: dict[str, Any], grid_shape: tuple[int, int], rotation: int) -> dict[str, Any]:
    raw_coord = _pair(state.get("selected_coord"))
    canon = display_coord(raw_coord, grid_shape, rotation) if raw_coord is not None else None
    return {
        "selected": state.get("selected") if isinstance(state.get("selected"), bool) else None,
        "selected_coord": raw_coord,
        "selected_display_coord": _pair(state.get("selected_display_coord")),
        "selected_canon_coord": None if canon is None else [canon[0], canon[1]],
        "selected_name": str(state.get("selected_name")) if state.get("selected_name") is not None else None,
        "selected_shape": _shape(state.get("selected_shape")),
    }


def _raw_click(packet: dict[str, Any], before_state: dict[str, Any]) -> list[int] | None:
    action = packet.get("action") if isinstance(packet.get("action"), dict) else {}
    data = action.get("data") if isinstance(action.get("data"), dict) else {}
    for value in (
        data.get("click_xy"),
        data.get("click"),
        action.get("selected_target"),
        before_state.get("selected_coord"),
    ):
        pair = _pair(value)
        if pair is not None:
            return pair
    return None


def _candidate_keys(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _source_context(packet: dict[str, Any]) -> dict[str, Any]:
    probes: list[tuple[str, dict[str, Any]]] = []
    brain = packet.get("brain_decision") if isinstance(packet.get("brain_decision"), dict) else {}
    brain_raw = brain.get("raw") if isinstance(brain.get("raw"), dict) else {}
    if brain_raw:
        probes.append(("brain_decision.raw", brain_raw))
    contra = packet.get("contra_decision") if isinstance(packet.get("contra_decision"), dict) else {}
    contra_raw = contra.get("raw") if isinstance(contra.get("raw"), dict) else {}
    if contra_raw:
        probes.append(("contra_decision.raw", contra_raw))
    fallback = brain_raw.get("fallback") if isinstance(brain_raw.get("fallback"), dict) else {}
    selected_activation = (
        fallback.get("selected_activation_feedback") if isinstance(fallback.get("selected_activation_feedback"), dict) else {}
    )
    if selected_activation:
        probes.append(("brain_decision.raw.fallback.selected_activation_feedback", selected_activation))

    merged: dict[str, Any] = {
        "active_portband_source_key": None,
        "focus_edge": None,
        "candidate_keys": None,
        "target_distance": None,
        "extraction_source": None,
    }
    for source, payload in probes:
        source_key = payload.get("active_portband_source_key") or payload.get("selected_source_key")
        focus_edge = payload.get("focus_edge")
        target_distance = payload.get("target_distance")
        candidate_keys = payload.get("candidate_keys")
        if candidate_keys is None:
            selected_outcome = payload.get("selected_outcome_history_feedback")
            if isinstance(selected_outcome, dict):
                candidate_keys = selected_outcome.get("candidate_keys")
        if merged["active_portband_source_key"] is None and isinstance(source_key, str):
            merged["active_portband_source_key"] = source_key
            merged["extraction_source"] = source
        if merged["focus_edge"] is None and isinstance(focus_edge, (int, float)):
            merged["focus_edge"] = int(focus_edge)
            merged["extraction_source"] = merged["extraction_source"] or source
        if merged["target_distance"] is None and isinstance(target_distance, (int, float)):
            merged["target_distance"] = float(target_distance)
            merged["extraction_source"] = merged["extraction_source"] or source
        keys = _candidate_keys(candidate_keys)
        if merged["candidate_keys"] is None and keys is not None:
            merged["candidate_keys"] = keys
            merged["extraction_source"] = merged["extraction_source"] or source
    return merged


def _adjacent_saturation(canon_grid: np.ndarray, coord: tuple[int, int] | None) -> dict[str, int | None]:
    if coord is None:
        return {"up": None, "down": None, "left": None, "right": None}
    x, y = coord
    neighbors = {
        "up": (x, y - 1),
        "down": (x, y + 1),
        "left": (x - 1, y),
        "right": (x + 1, y),
    }
    out: dict[str, int | None] = {}
    for name, (nx, ny) in neighbors.items():
        if 0 <= nx < canon_grid.shape[0] and 0 <= ny < canon_grid.shape[1]:
            out[name] = int(canon_grid[nx, ny])
        else:
            out[name] = None
    return out


def _run_dir_from_packet_path(path: Path) -> Path:
    parent = path.parent
    if parent.name.startswith("level_"):
        return parent.parent
    return parent


def _level_from_packet_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part.startswith("level_"):
            return part
    return None


def _goal_progress(packet: dict[str, Any]) -> dict[str, Any]:
    progress = packet.get("goal_progress") if isinstance(packet.get("goal_progress"), dict) else {}
    return {
        "distance_delta": _number(progress.get("distance_delta")),
        "distance_reduced": progress.get("distance_reduced"),
        "before_display_manhattan_distance": _number(progress.get("before_display_manhattan_distance")),
        "after_display_manhattan_distance": _number(progress.get("after_display_manhattan_distance")),
        "target_bucket_index": progress.get("target_bucket_index"),
        "target_object_id": progress.get("target_object_id"),
    }


def outcome_class(levels_completed_delta: int, distance_delta: float | None) -> str:
    if levels_completed_delta > 0:
        return "level_completed"
    if distance_delta is None or distance_delta == 0:
        return "neutral"
    return "distance_positive" if distance_delta > 0 else "distance_negative"


def build_activation_row(packet_path: Path, line_number: int, packet: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if packet.get("status") != "executed":
        return None, f"status_{packet.get('status') or 'missing'}"
    primitive = primitive_from_packet(packet)
    if primitive != "activate_flow":
        return None, "not_activate_flow"
    before_ref = grid_ref(packet.get("before_frame"))
    after_ref = grid_ref(packet.get("after_frame"))
    if before_ref is None or after_ref is None:
        return None, "missing_raw_grid_ref"
    before_path = resolve_repo_path(before_ref["path"])
    after_path = resolve_repo_path(after_ref["path"])
    if not before_path.is_file() or not after_path.is_file():
        return None, "missing_raw_grid_file"
    if sha256_file(before_path) != before_ref["npy_sha256"] or sha256_file(after_path) != after_ref["npy_sha256"]:
        return None, "raw_grid_sha_mismatch"
    before_grid = np.load(before_path)
    after_grid = np.load(after_path)
    if before_grid.shape != after_grid.shape or len(before_grid.shape) != 2:
        return None, "bad_grid_shape"
    before_state = packet.get("before_state") if isinstance(packet.get("before_state"), dict) else {}
    after_state = packet.get("after_state") if isinstance(packet.get("after_state"), dict) else {}
    rotation = before_state.get("rotation_quarters")
    if not isinstance(rotation, int) or rotation not in SUPPORTED_ROTATION_QUARTERS:
        return None, f"unsupported_rotation_{rotation!r}"
    raw_click = _raw_click(packet, before_state)
    canon_click = display_coord(raw_click, before_grid.shape, rotation) if raw_click is not None else None
    before_selection = _state_selection(before_state, before_grid.shape, rotation)
    after_selection = _state_selection(after_state, after_grid.shape, rotation)
    progress = _goal_progress(packet)
    levels_delta = int(packet.get("levels_completed_delta") or 0)
    action_sig = action_signature(packet, primitive)
    row = {
        "schema": ACTIVATION_SCHEMA,
        "activation_id": stable_id("crucible_activation_transition", before_ref["npy_sha256"], action_sig, after_ref["npy_sha256"]),
        "action": {
            "primitive": primitive,
            "action_signature": action_sig,
            "action_packet": packet.get("action") if isinstance(packet.get("action"), dict) else None,
            "raw_click": {"x": raw_click[0], "y": raw_click[1]} if raw_click is not None else None,
        },
        "click": {
            "raw_xy": raw_click,
            "display_coord": None if canon_click is None else [canon_click[0], canon_click[1]],
            "canon_coord": None if canon_click is None else [canon_click[0], canon_click[1]],
            "rotation_quarters": rotation,
        },
        "selection_before": before_selection,
        "selection_after": after_selection,
        "derived": {
            "selected_object_changed": before_selection.get("selected_name") != after_selection.get("selected_name"),
            "selected_shape_changed": before_selection.get("selected_shape") != after_selection.get("selected_shape"),
        },
        "source_context": _source_context(packet),
        "adjacent_saturation": _adjacent_saturation(canonicalize_grid(before_grid, rotation), canon_click),
        "before_grid": before_ref,
        "after_grid": after_ref,
        "goal_progress": progress,
        "levels_completed_delta": levels_delta,
        "outcome_class": outcome_class(levels_delta, progress.get("distance_delta")),
        "provenance": {
            "packets_jsonl": rel(packet_path),
            "line_number": int(line_number),
            "run_dir": rel(_run_dir_from_packet_path(packet_path)),
            "level_dir": rel(packet_path.parent),
            "level_id": _level_from_packet_path(packet_path),
            "case_id": packet.get("case_id"),
            "action_index": packet.get("action_index"),
        },
        "truth_boundary": truth_boundary(),
    }
    return row, None


def _presence_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    fields = {
        "focus_edge": lambda row: (row.get("source_context") or {}).get("focus_edge"),
        "active_portband_source_key": lambda row: (row.get("source_context") or {}).get("active_portband_source_key"),
        "target_distance": lambda row: (row.get("source_context") or {}).get("target_distance"),
        "candidate_keys": lambda row: (row.get("source_context") or {}).get("candidate_keys"),
        "before_selected_name": lambda row: (row.get("selection_before") or {}).get("selected_name"),
        "after_selected_name": lambda row: (row.get("selection_after") or {}).get("selected_name"),
    }
    return {name: sum(1 for row in rows if getter(row) is not None) for name, getter in fields.items()}


def mine_activation_corpus(
    runs_root: Path,
    out_dir: Path = DEFAULT_ACTIVATION_DIR,
    admitted_manifest: Path = DEFAULT_ADMITTED_MANIFEST,
) -> dict[str, Any]:
    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)
    admitted = load_admitted_runs(admitted_manifest)
    rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    scanned, path_skip_reasons, not_admitted_run_dirs = packet_paths(runs_root, admitted)
    skip_reasons.update(path_skip_reasons)
    seen: set[str] = set()
    duplicate_count = 0
    for packets_jsonl in scanned:
        for line_number, packet in enumerate(read_jsonl(packets_jsonl), start=1):
            row, skip = build_activation_row(packets_jsonl, line_number, packet)
            if row is None:
                skip_reasons[str(skip or "unknown_skip")] += 1
                continue
            key = str(row["activation_id"])
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            rows.append(row)

    corpus_path = resolved_out / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_path = resolved_out / "manifest.json"
    outcome_histogram = Counter(str(row.get("outcome_class")) for row in rows)
    per_run_counts = Counter(str((row.get("provenance") or {}).get("run_dir")) for row in rows)
    manifest = {
        "schema": ACTIVATION_CORPUS_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "out_dir": rel(resolved_out),
        "manifest_json": rel(manifest_path),
        "corpus_jsonl": rel(corpus_path),
        "row_count": len(rows),
        "dedup_duplicate_count": duplicate_count,
        "scanned_packets_jsonl_count": len(scanned),
        "field_presence_histogram": _presence_histogram(rows),
        "outcome_class_histogram": dict(sorted(outcome_histogram.items())),
        "per_run_counts": dict(sorted(per_run_counts.items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "admitted_manifest": rel(admitted["manifest_path"]),
        "admitted_manifest_sha256": admitted["manifest_sha256"],
        "admitted_run_dirs": sorted(admitted["admitted_run_dirs"]),
        "not_admitted_skipped_count": int(skip_reasons.get("not_in_admitted_manifest", 0)),
        "not_admitted_run_dirs": not_admitted_run_dirs,
        "source_hashes": source_hashes(
            [
                "codexlab/contra_head/crucible/activation.py",
                "codexlab/crucible_mine_activation_transitions.py",
            ]
        ),
        "truth_boundary": truth_boundary(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
