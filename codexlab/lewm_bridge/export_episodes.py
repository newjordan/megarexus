#!/usr/bin/env python3
"""Export full SP80 episodes from admitted runs as a stable LeWM training-feed surface.

Unlike the crucible transition corpus (deduped per-transition rows), this export
keeps every executed step of every admitted episode in order, with grid refs,
action signatures, and goal-progress fields. It is the future LeWM training feed;
it admits no truth and is not consumed by any CONTRA solver path.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_CODEXLAB = Path(__file__).resolve().parents[1]
if str(_CODEXLAB) not in sys.path:
    sys.path.insert(0, str(_CODEXLAB))

from contra_head.crucible.corpus import (  # noqa: E402
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

DEFAULT_EPISODES_DIR = _CODEXLAB / "datasets" / "sp80_lewm_episodes_v0"
EPISODE_SCHEMA = "codexlab.lewm_episode.v1"
EPISODE_CORPUS_SCHEMA = "codexlab.lewm_episode_corpus.v1"
TRUTH_BOUNDARY_LITERAL = "lewm_episode_export_is_observation_evidence_not_control_truth"


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


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _state(packet: dict[str, Any], key: str) -> dict[str, Any]:
    value = packet.get(key)
    return value if isinstance(value, dict) else {}


def build_step(
    line_number: int,
    packet: dict[str, Any],
    *,
    verify_grid_sha: bool,
) -> tuple[dict[str, Any] | None, str | None]:
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
    if verify_grid_sha:
        if sha256_file(before_path) != before_ref["npy_sha256"] or sha256_file(after_path) != after_ref["npy_sha256"]:
            return None, "raw_grid_sha_mismatch"
    before_state = _state(packet, "before_state")
    after_state = _state(packet, "after_state")
    progress = packet.get("goal_progress") if isinstance(packet.get("goal_progress"), dict) else {}
    delta = packet.get("delta") if isinstance(packet.get("delta"), dict) else {}
    step = {
        "line_number": int(line_number),
        "action_index": packet.get("action_index"),
        "case_id": packet.get("case_id"),
        "action": {
            "primitive": primitive,
            "action_signature": action_signature(packet, primitive),
        },
        "before_grid": before_ref,
        "after_grid": after_ref,
        "rotation_quarters": before_state.get("rotation_quarters"),
        "actor": {
            "before_raw_coord": before_state.get("selected_coord"),
            "after_raw_coord": after_state.get("selected_coord"),
            "before_display_coord": before_state.get("selected_display_coord"),
            "after_display_coord": after_state.get("selected_display_coord"),
            "selected_name_before": before_state.get("selected_name"),
            "selected_name_after": after_state.get("selected_name"),
        },
        "goal_progress": {
            "distance_delta": _number(progress.get("distance_delta")),
            "distance_reduced": progress.get("distance_reduced"),
            "before_display_manhattan_distance": _number(progress.get("before_display_manhattan_distance")),
            "after_display_manhattan_distance": _number(progress.get("after_display_manhattan_distance")),
            "target_bucket_index": progress.get("target_bucket_index"),
            "target_object_id": progress.get("target_object_id"),
        },
        "levels_completed_before": before_state.get("levels_completed"),
        "levels_completed_after": after_state.get("levels_completed"),
        "levels_completed_delta": int(packet.get("levels_completed_delta") or 0),
        "changed_pixels": delta.get("changed_pixels"),
    }
    return step, None


def build_episode(packets_jsonl: Path, *, verify_grid_sha: bool) -> tuple[dict[str, Any] | None, Counter[str]]:
    skip_reasons: Counter[str] = Counter()
    steps: list[dict[str, Any]] = []
    for line_number, packet in enumerate(read_jsonl(packets_jsonl), start=1):
        step, skip = build_step(line_number, packet, verify_grid_sha=verify_grid_sha)
        if step is None:
            skip_reasons[str(skip or "unknown_skip")] += 1
            continue
        steps.append(step)
    if not steps:
        skip_reasons["episode_without_steps"] += 1
        return None, skip_reasons

    # Discontinuities (a step whose before-grid is not the previous step's
    # after-grid) matter for sequence training; surface them, don't hide them.
    discontinuities = sum(
        1
        for prev, cur in zip(steps, steps[1:])
        if prev["after_grid"]["npy_sha256"] != cur["before_grid"]["npy_sha256"]
    )
    level_dir = packets_jsonl.parent
    run_dir = level_dir.parent if level_dir.name.startswith("level_") else level_dir
    level_id = level_dir.name if level_dir.name.startswith("level_") else None
    levels_completed_delta_total = sum(step["levels_completed_delta"] for step in steps)
    episode = {
        "schema": EPISODE_SCHEMA,
        "episode_id": stable_id("lewm_episode", rel(run_dir), level_id, len(steps)),
        "run_dir": rel(run_dir),
        "level_id": level_id,
        "packets_jsonl": rel(packets_jsonl),
        "solved": levels_completed_delta_total > 0,
        "levels_completed_delta_total": levels_completed_delta_total,
        "step_count": len(steps),
        "discontinuity_count": discontinuities,
        "step_skip_reasons": dict(sorted(skip_reasons.items())),
        "steps": steps,
        "truth_boundary": truth_boundary(),
    }
    return episode, skip_reasons


def export_episodes(
    runs_root: Path,
    out_dir: Path = DEFAULT_EPISODES_DIR,
    admitted_manifest: Path = DEFAULT_ADMITTED_MANIFEST,
    *,
    verify_grid_sha: bool = True,
) -> dict[str, Any]:
    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)
    admitted = load_admitted_runs(admitted_manifest)
    scanned, path_skip_reasons, not_admitted_run_dirs = packet_paths(runs_root, admitted)
    skip_reasons: Counter[str] = Counter()
    skip_reasons.update(path_skip_reasons)
    episodes: list[dict[str, Any]] = []
    for packets_jsonl in scanned:
        episode, episode_skips = build_episode(packets_jsonl, verify_grid_sha=verify_grid_sha)
        skip_reasons.update(episode_skips)
        if episode is not None:
            episodes.append(episode)

    episodes_path = resolved_out / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, sort_keys=True) + "\n")

    solved_episode_ids = [ep["episode_id"] for ep in episodes if ep["solved"]]
    manifest = {
        "schema": EPISODE_CORPUS_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runs_root": rel(runs_root if runs_root.is_absolute() else ROOT / runs_root),
        "admitted_manifest": rel(admitted["manifest_path"]),
        "admitted_manifest_sha256": admitted["manifest_sha256"],
        "admitted_run_dirs": sorted(admitted["admitted_run_dirs"]),
        "not_admitted_run_dirs": not_admitted_run_dirs,
        "out_dir": rel(resolved_out),
        "episodes_jsonl": rel(episodes_path),
        "episode_count": len(episodes),
        "solved_episode_count": len(solved_episode_ids),
        "solved_episode_ids": solved_episode_ids,
        "total_step_count": sum(ep["step_count"] for ep in episodes),
        "total_discontinuity_count": sum(ep["discontinuity_count"] for ep in episodes),
        "scanned_packets_jsonl_count": len(scanned),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "verify_grid_sha": verify_grid_sha,
        "per_episode_step_counts": {ep["episode_id"]: ep["step_count"] for ep in episodes},
        "source_hashes": source_hashes(["codexlab/lewm_bridge/export_episodes.py"]),
        "truth_boundary": truth_boundary(),
    }
    manifest_path = resolved_out / "manifest.json"
    manifest["manifest_json"] = rel(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("codexlab/runs"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--admitted-manifest", type=Path, default=DEFAULT_ADMITTED_MANIFEST)
    parser.add_argument("--no-verify-grid-sha", action="store_true", help="skip per-step npy sha verification")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = export_episodes(
        args.runs_root,
        args.out_dir,
        args.admitted_manifest,
        verify_grid_sha=not args.no_verify_grid_sha,
    )
    keys = [
        "schema",
        "manifest_json",
        "episodes_jsonl",
        "episode_count",
        "solved_episode_count",
        "total_step_count",
        "total_discontinuity_count",
        "scanned_packets_jsonl_count",
        "skip_reasons",
    ]
    print(json.dumps({key: manifest.get(key) for key in keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
