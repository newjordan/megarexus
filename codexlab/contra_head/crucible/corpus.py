"""Mine controlled SP80 transition rows for Crucible."""

from __future__ import annotations

import hashlib
import json
import fnmatch
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .canon import SUPPORTED_ROTATION_QUARTERS, canonicalize_coord, canonicalize_grid, coord_delta


ROOT = Path(__file__).resolve().parents[3]
CODEXLAB = ROOT / "codexlab"
DEFAULT_CORPUS_DIR = CODEXLAB / "datasets" / "sp80_crucible_transitions_v0"
DEFAULT_ADMITTED_MANIFEST = CODEXLAB / "datasets" / "sp80_crucible_admitted_runs_v0" / "admitted_runs.json"
TRANSITION_SCHEMA = "codexlab.crucible_transition.v1"
CORPUS_SCHEMA = "codexlab.crucible_transition_corpus.v1"
ADMITTED_RUNS_SCHEMA = "codexlab.crucible_admitted_runs.v1"
TRUTH_BOUNDARY_LITERAL = "crucible_transition_corpus_is_observation_evidence_not_control_truth"
# SP80 state coordinates are emitted as [x, y] and are indexable into the
# saved grid in that order. This mirrors display deltas used by the harness.
EXPECTED_CANON_DELTAS = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
    "select_object": (0, 0),
    "activate_flow": (0, 0),
}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_id(namespace: str, *parts: Any) -> str:
    payload = stable_json([namespace, *parts]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_admitted_runs(manifest_path: Path = DEFAULT_ADMITTED_MANIFEST) -> dict[str, Any]:
    path = manifest_path if manifest_path.is_absolute() else ROOT / manifest_path
    if not path.is_file():
        raise FileNotFoundError(f"missing admitted-runs manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != ADMITTED_RUNS_SCHEMA:
        raise ValueError(f"bad admitted-runs manifest schema: {path}")
    if payload.get("default_policy") != "exclude":
        raise ValueError("admitted-runs manifest must use default_policy=exclude")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("admitted-runs manifest entries must be a list")
    admitted: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("admitted") is not True:
            continue
        run_dir = entry.get("run_dir")
        if isinstance(run_dir, str) and run_dir:
            admitted.add(rel(resolve_repo_path(run_dir)))
    exclusions = []
    for entry in payload.get("explicit_exclusions") or []:
        if isinstance(entry, dict) and isinstance(entry.get("run_dir_or_glob"), str):
            exclusions.append(entry["run_dir_or_glob"])
    return {
        "manifest_path": path,
        "manifest_sha256": sha256_file(path),
        "admitted_run_dirs": admitted,
        "explicit_exclusions": exclusions,
        "raw": payload,
    }


def _is_explicitly_excluded(run_dir: str, exclusions: list[str]) -> bool:
    return any(fnmatch.fnmatch(run_dir, pattern) or pattern in run_dir for pattern in exclusions)


def packet_paths(runs_root: Path, admitted: dict[str, Any] | None = None) -> tuple[list[Path], Counter[str], list[str]]:
    root = runs_root if runs_root.is_absolute() else ROOT / runs_root
    paths: list[Path] = []
    skip_reasons: Counter[str] = Counter()
    not_admitted_run_dirs: set[str] = set()
    admitted_dirs = admitted.get("admitted_run_dirs", set()) if isinstance(admitted, dict) else None
    exclusions = admitted.get("explicit_exclusions", []) if isinstance(admitted, dict) else []
    for path in sorted(root.glob("**/packets.jsonl")):
        if not path.is_file():
            continue
        run_dir = rel(_run_dir_from_packet_path(path))
        if _is_explicitly_excluded(run_dir, exclusions):
            skip_reasons["explicitly_excluded_manifest_path"] += 1
            not_admitted_run_dirs.add(run_dir)
            continue
        if admitted_dirs is not None and run_dir not in admitted_dirs:
            skip_reasons["not_in_admitted_manifest"] += 1
            not_admitted_run_dirs.add(run_dir)
            continue
        paths.append(path)
    return paths, skip_reasons, sorted(not_admitted_run_dirs)


def source_hashes(extra_paths: list[str] | None = None) -> dict[str, str]:
    paths = [
        "codexlab/contra_head/crucible/canon.py",
        "codexlab/contra_head/crucible/corpus.py",
        "codexlab/contra_head/crucible/model.py",
        "codexlab/crucible_mine_transitions.py",
        "codexlab/contra_verify_crucible.py",
        "codexlab/sp80_player_harness.py",
    ]
    paths.extend(extra_paths or [])
    hashes: dict[str, str] = {}
    for item in paths:
        path = ROOT / item
        if path.is_file():
            hashes[item] = sha256_file(path)
    return hashes


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
        "boundary": TRUTH_BOUNDARY_LITERAL,
    }


def grid_ref(frame_entry: Any) -> dict[str, Any] | None:
    if not isinstance(frame_entry, dict):
        return None
    raw_grid = frame_entry.get("raw_grid")
    if not isinstance(raw_grid, dict):
        return None
    path = raw_grid.get("path")
    digest = raw_grid.get("npy_sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        return None
    return {
        "path": path,
        "npy_sha256": digest,
        "raw_frame_sha256": raw_grid.get("raw_frame_sha256"),
        "shape": raw_grid.get("shape"),
        "dtype": raw_grid.get("dtype"),
    }


def primitive_from_packet(packet: dict[str, Any]) -> str | None:
    primitive = packet.get("decision_primitive")
    if isinstance(primitive, str) and primitive:
        return primitive
    action = packet.get("action") if isinstance(packet.get("action"), dict) else {}
    primitive = action.get("primitive")
    return primitive if isinstance(primitive, str) and primitive else None


def action_signature(packet: dict[str, Any], primitive: str) -> dict[str, Any]:
    action = packet.get("action") if isinstance(packet.get("action"), dict) else {}
    selected = packet.get("selected_recommendation") if isinstance(packet.get("selected_recommendation"), dict) else {}
    return {
        "primitive": primitive,
        "action_id": action.get("id"),
        "action_name": action.get("name"),
        "action_data": action.get("data"),
        "target_override": selected.get("target_override"),
    }


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _goal_progress(packet: dict[str, Any]) -> dict[str, Any]:
    progress = packet.get("goal_progress")
    return progress if isinstance(progress, dict) else {}


def _level_from_packet_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        if part.startswith("level_"):
            return part
    return None


def _run_dir_from_packet_path(path: Path) -> Path:
    parent = path.parent
    if parent.name.startswith("level_"):
        return parent.parent
    return parent


def build_transition_row(packet_path: Path, line_number: int, packet: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if packet.get("status") != "executed":
        return None, f"status_{packet.get('status') or 'missing'}"
    primitive = primitive_from_packet(packet)
    if not primitive:
        return None, "missing_primitive"
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
    before_state = packet.get("before_state") if isinstance(packet.get("before_state"), dict) else {}
    after_state = packet.get("after_state") if isinstance(packet.get("after_state"), dict) else {}
    rotation = before_state.get("rotation_quarters")
    if not isinstance(rotation, int) or rotation not in SUPPORTED_ROTATION_QUARTERS:
        return None, f"unsupported_rotation_{rotation!r}"

    before_grid = np.load(before_path)
    after_grid = np.load(after_path)
    if before_grid.shape != after_grid.shape or len(before_grid.shape) != 2:
        return None, "bad_grid_shape"
    before_raw_coord = before_state.get("selected_coord")
    after_raw_coord = after_state.get("selected_coord")
    before_canon = canonicalize_coord(before_raw_coord, before_grid.shape, rotation)
    after_canon = canonicalize_coord(after_raw_coord, after_grid.shape, rotation)
    if before_canon is None or after_canon is None:
        return None, "missing_actor_coord"
    canon_grid = canonicalize_grid(before_grid, rotation)
    actor_color = int(canon_grid[before_canon[0], before_canon[1]])
    if actor_color == 0:
        return None, "actor_color_invariant_failed"

    observed_delta = coord_delta(before_canon, after_canon)
    expected_delta = EXPECTED_CANON_DELTAS.get(primitive)
    action_sig = action_signature(packet, primitive)
    transition_id = stable_id("crucible_transition", before_ref["npy_sha256"], action_sig, after_ref["npy_sha256"])
    before_name = before_state.get("selected_name")
    after_name = after_state.get("selected_name")
    selection_changed = bool(before_name != after_name or primitive == "select_object")
    progress = _goal_progress(packet)
    row = {
        "schema": TRANSITION_SCHEMA,
        "transition_id": transition_id,
        "action": {
            "primitive": primitive,
            "action_signature": action_sig,
            "action_packet": packet.get("action") if isinstance(packet.get("action"), dict) else None,
        },
        "before_grid": before_ref,
        "after_grid": after_ref,
        "actor": {
            "color": actor_color,
            "before_raw_coord": before_raw_coord,
            "after_raw_coord": after_raw_coord,
            "before_display_coord": before_state.get("selected_display_coord"),
            "after_display_coord": after_state.get("selected_display_coord"),
            "before_canon_coord": [before_canon[0], before_canon[1]],
            "after_canon_coord": [after_canon[0], after_canon[1]],
            "rotation_quarters": rotation,
            "selected_name_before": before_name,
            "selected_name_after": after_name,
            "selection_changed": selection_changed,
        },
        "observed": {
            "canon_delta": None if observed_delta is None else [observed_delta[0], observed_delta[1]],
            "moved": observed_delta not in (None, (0, 0)),
            "blocked": primitive.startswith("move_") and observed_delta == (0, 0),
        },
        "expected_canon_delta": None if expected_delta is None else [expected_delta[0], expected_delta[1]],
        "visual_delta": {
            "selected_coord_delta": packet.get("selected_coord_delta"),
            "selected_display_coord_delta": packet.get("selected_display_coord_delta"),
            "changed_pixels": (packet.get("delta") or {}).get("changed_pixels") if isinstance(packet.get("delta"), dict) else None,
        },
        "goal_progress": {
            "distance_delta": _number(progress.get("distance_delta")),
            "distance_reduced": progress.get("distance_reduced"),
            "before_display_manhattan_distance": _number(progress.get("before_display_manhattan_distance")),
            "after_display_manhattan_distance": _number(progress.get("after_display_manhattan_distance")),
            "target_bucket_index": progress.get("target_bucket_index"),
            "target_object_id": progress.get("target_object_id"),
        },
        "levels_completed_delta": int(packet.get("levels_completed_delta") or 0),
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


def baseline_mismatch_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    move_rows = [
        row for row in rows
        if str(((row.get("action") or {}).get("primitive") or "")).startswith("move_")
        and (row.get("actor") or {}).get("selection_changed") is False
    ]
    stationary = 0
    wrong_direction = 0
    matched = 0
    for row in move_rows:
        observed = (row.get("observed") or {}).get("canon_delta")
        expected = row.get("expected_canon_delta")
        if observed == [0, 0]:
            stationary += 1
        elif observed == expected:
            matched += 1
        else:
            wrong_direction += 1
    total = len(move_rows)
    mismatch = wrong_direction + stationary
    return {
        "move_transition_count": total,
        "matched_expected_delta_count": matched,
        "wrong_direction_count": wrong_direction,
        "stationary_count": stationary,
        "mismatch_count": mismatch,
        "wrong_direction_rate": 0.0 if total == 0 else round(wrong_direction / total, 6),
        "stationary_rate": 0.0 if total == 0 else round(stationary / total, 6),
        "mismatch_rate": 0.0 if total == 0 else round(mismatch / total, 6),
        "expected_model_version": "pinned_canon_delta_v0",
    }


def build_manifest(
    *,
    out_dir: Path,
    corpus_path: Path,
    rows: list[dict[str, Any]],
    skip_reasons: Counter[str],
    duplicate_count: int,
    scanned_packet_paths: list[Path],
    runs_root: Path,
    admitted: dict[str, Any],
    not_admitted_run_dirs: list[str],
) -> dict[str, Any]:
    primitive_histogram = Counter(str((row.get("action") or {}).get("primitive")) for row in rows)
    rotation_histogram = Counter(str((row.get("actor") or {}).get("rotation_quarters")) for row in rows)
    per_level_counts = Counter(str((row.get("provenance") or {}).get("level_id")) for row in rows)
    per_run_counts = Counter(str((row.get("provenance") or {}).get("run_dir")) for row in rows)
    return {
        "schema": CORPUS_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs_root": rel(runs_root if runs_root.is_absolute() else ROOT / runs_root),
        "admitted_manifest": rel(admitted["manifest_path"]),
        "admitted_manifest_sha256": admitted["manifest_sha256"],
        "admitted_run_dirs": sorted(admitted["admitted_run_dirs"]),
        "not_admitted_skipped_count": int(skip_reasons.get("not_in_admitted_manifest", 0)),
        "not_admitted_run_dirs": not_admitted_run_dirs,
        "out_dir": rel(out_dir),
        "corpus_jsonl": rel(corpus_path),
        "row_count": len(rows),
        "dedup_duplicate_count": int(duplicate_count),
        "scanned_packets_jsonl_count": len(scanned_packet_paths),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "primitive_histogram": dict(sorted(primitive_histogram.items())),
        "rotation_histogram": dict(sorted(rotation_histogram.items())),
        "per_level_counts": dict(sorted(per_level_counts.items())),
        "per_run_counts": dict(sorted(per_run_counts.items())),
        "baseline_mismatch_summary": baseline_mismatch_summary(rows),
        "source_hashes": source_hashes(),
        "truth_boundary": truth_boundary(),
    }


def mine_transition_corpus(
    runs_root: Path,
    out_dir: Path = DEFAULT_CORPUS_DIR,
    admitted_manifest: Path = DEFAULT_ADMITTED_MANIFEST,
) -> dict[str, Any]:
    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)
    admitted = load_admitted_runs(admitted_manifest)
    rows: list[dict[str, Any]] = []
    skip_reasons: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    duplicate_count = 0
    scanned, path_skip_reasons, not_admitted_run_dirs = packet_paths(runs_root, admitted)
    skip_reasons.update(path_skip_reasons)
    for packets_jsonl in scanned:
        for line_number, packet in enumerate(read_jsonl(packets_jsonl), start=1):
            row, skip = build_transition_row(packets_jsonl, line_number, packet)
            if row is None:
                skip_reasons[str(skip or "unknown_skip")] += 1
                continue
            key = (
                str((row.get("before_grid") or {}).get("npy_sha256")),
                stable_json((row.get("action") or {}).get("action_signature")),
                str((row.get("after_grid") or {}).get("npy_sha256")),
            )
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            rows.append(row)

    corpus_path = resolved_out / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = build_manifest(
        out_dir=resolved_out,
        corpus_path=corpus_path,
        rows=rows,
        skip_reasons=skip_reasons,
        duplicate_count=duplicate_count,
        scanned_packet_paths=scanned,
        runs_root=runs_root,
        admitted=admitted,
        not_admitted_run_dirs=not_admitted_run_dirs,
    )
    manifest_path = resolved_out / "manifest.json"
    manifest["manifest_json"] = rel(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
