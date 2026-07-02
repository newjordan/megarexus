#!/usr/bin/env python3
"""Export public ARC Opus 4.8 traces into CONTRA teacher-signal rows.

The output is hypothesis/process supervision only. Rows are not local CONTRA
evidence and are not action labels.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROW_SCHEMA = "codexlab.arc_teacher_signal_row.v1"
MANIFEST_SCHEMA = "codexlab.arc_teacher_signal_manifest.v1"

DEFAULT_MIN_REPEAT_RISK = 80
DEFAULT_MIN_CONTRADICTION = 25
DEFAULT_MIN_BUDGET_ACTIONS = 100

SIGNAL_FAMILIES = [
    "control_surface_signal",
    "family_route_prior_signal",
    "action_semantics_signal",
    "coordinate_vetting_signal",
    "dead_probe_memory_signal",
    "budget_governor_signal",
    "goal_spec_candidate_signal",
    "contradiction_replan_signal",
]

CLICK_ACTION = "ACTION6"
HELDOUT_GAME_IDS = {"sp80-589a99af", "g50t-5849a774", "tn36-ef4dde99", "wa30-ee6fef47"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def compact_text(text: str, limit: int = 480) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 3] + "..."


def top_action(action_counts: dict[str, Any]) -> str | None:
    best_action = None
    best_count = -1
    for action, count in action_counts.items():
        if action == "RESET":
            continue
        try:
            numeric_count = int(count)
        except (TypeError, ValueError):
            continue
        if numeric_count > best_count:
            best_action = action
            best_count = numeric_count
    return best_action


def action_family(summary: dict[str, Any]) -> str:
    counts = summary.get("action_counts") or {}
    non_reset = {k: int(v) for k, v in counts.items() if k != "RESET"}
    if not non_reset:
        return "unknown"
    click_count = non_reset.get(CLICK_ACTION, 0)
    total = sum(non_reset.values())
    if total and click_count / total >= 0.7:
        return "click_dominant"
    directional = sum(non_reset.get(name, 0) for name in ["ACTION1", "ACTION2", "ACTION3", "ACTION4"])
    if total and directional / total >= 0.7:
        return "directional_dominant"
    return "mixed_action_family"


def source_files_for_game(mining_dir: Path, game_id: str) -> dict[str, Path | None]:
    matches = sorted(mining_dir.glob(f"{game_id}__*__reasoning.jsonl"))
    return {"reasoning": matches[0] if matches else None}


def select_reasoning_samples(rows: list[dict[str, Any]], family: str, limit: int = 3) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        hits = row.get("pattern_hits") or {}
        if family == "contradiction_replan_signal" and hits.get("contradiction_revision"):
            selected.append(row)
        elif family == "budget_governor_signal" and hits.get("resource_budget"):
            selected.append(row)
        elif family == "dead_probe_memory_signal" and hits.get("repeat_risk"):
            selected.append(row)
        elif family == "goal_spec_candidate_signal" and hits.get("goal_inference"):
            selected.append(row)
        elif family in {"control_surface_signal", "action_semantics_signal", "coordinate_vetting_signal"} and (
            hits.get("control_surface_probe") or hits.get("predict_observe_update")
        ):
            selected.append(row)
        if len(selected) >= limit:
            break
    if not selected:
        selected = rows[:limit]
    return [
        {
            "event_index": row.get("event_index"),
            "timestamp": row.get("timestamp"),
            "action_id": row.get("action_id"),
            "levels_completed": row.get("levels_completed"),
            "pattern_hits": row.get("pattern_hits") or {},
            "text_excerpt": compact_text(str(row.get("text_excerpt") or "")),
        }
        for row in selected
    ]


def make_base_row(
    *,
    row_id: str,
    split: str,
    signal_family: str,
    game_summary: dict[str, Any],
    recording_manifest: dict[str, Any] | None,
    reasoning_path: Path | None,
    source_summary_path: Path,
    download_manifest_path: Path,
    reasoning_samples: list[dict[str, Any]],
    target: dict[str, Any],
) -> dict[str, Any]:
    game_id = game_summary["game_id"]
    guid = game_summary.get("guid")
    evidence = {
        "truth_status": "hypothesis_teacher_signal",
        "boundary": "public ARC recording mining only; not local CONTRA evidence",
        "source_summary": relpath(source_summary_path),
        "source_download_manifest": relpath(download_manifest_path),
        "source_reasoning_jsonl": relpath(reasoning_path) if reasoning_path else None,
        "source_recording_jsonl": recording_manifest.get("path") if recording_manifest else None,
        "source_recording_sha256": recording_manifest.get("sha256") if recording_manifest else None,
        "scorecard_score": game_summary.get("scorecard_score"),
        "scorecard_levels_completed": game_summary.get("scorecard_levels_completed"),
        "scorecard_actions": game_summary.get("scorecard_actions"),
        "event_count": game_summary.get("event_count"),
        "level_progressions": game_summary.get("level_progressions") or [],
        "pattern_counts": game_summary.get("pattern_counts") or {},
        "action_counts": game_summary.get("action_counts") or {},
        "reasoning_samples": reasoning_samples,
    }
    input_obj = {
        "task": "emit_contra_signal_packet",
        "game_id": game_id,
        "recording_guid": guid,
        "signal_family": signal_family,
        "allowed_signal_families": SIGNAL_FAMILIES,
        "observed_public_trace_summary": {
            "levels_completed": game_summary.get("scorecard_levels_completed"),
            "actions": game_summary.get("scorecard_actions"),
            "event_count": game_summary.get("event_count"),
            "action_family_hint": action_family(game_summary),
            "top_action": top_action(game_summary.get("action_counts") or {}),
            "pattern_counts": game_summary.get("pattern_counts") or {},
        },
        "contract": {
            "student_output": "signal packet target",
            "may_claim_solve": False,
            "may_claim_local_contra_evidence": False,
            "must_preserve_boundary": True,
        },
    }
    return {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "split": split,
        "game_id": game_id,
        "recording_guid": guid,
        "signal_family": signal_family,
        "input": input_obj,
        "target": target,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Emit one CONTRA teacher signal packet as JSON. "
                    "Do not claim solve truth, action truth, or local CONTRA evidence."
                ),
            },
            {"role": "user", "content": json.dumps(input_obj, sort_keys=True)},
            {"role": "assistant", "content": json.dumps(target, sort_keys=True)},
        ],
        "evidence": evidence,
    }


def build_targets(game_summary: dict[str, Any], reasoning_rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    game_id = game_summary["game_id"]
    pattern_counts = game_summary.get("pattern_counts") or {}
    counts = game_summary.get("action_counts") or {}
    progressions = game_summary.get("level_progressions") or []
    family = action_family(game_summary)
    target_rows: list[tuple[str, dict[str, Any]]] = []

    target_rows.append(
        (
            "control_surface_signal",
            {
                "signal_family": "control_surface_signal",
                "recommended_packet": "control_surface_packet",
                "action_family_hint": family,
                "top_action": top_action(counts),
                "confidence_basis": "public_trace_action_counts_and_reasoning_patterns",
                "required_next_local_verifier": "controlled_action_semantics_probe",
                "truth_status": "hypothesis_teacher_signal",
            },
        )
    )
    target_rows.append(
        (
            "family_route_prior_signal",
            {
                "signal_family": "family_route_prior_signal",
                "recommended_packet": "family_route_prior",
                "route_prior": family,
                "source_priority": ["source_decode_family", "observed_public_action_deltas", "teacher_reasoning"],
                "missing_local_inputs": ["controlled_before_after_deltas"],
                "truth_status": "hypothesis_teacher_signal",
            },
        )
    )

    if progressions:
        transition_actions = [str(item.get("action_id")) for item in progressions if item.get("action_id")]
        target_rows.append(
            (
                "action_semantics_signal",
                {
                    "signal_family": "action_semantics_signal",
                    "recommended_packet": "action_semantics_ledger",
                    "progression_actions": transition_actions,
                    "first_progression": progressions[0],
                    "teacher_hint": "record per-action before/after deltas before committing plans",
                    "required_next_local_verifier": "controlled_before_action_after_packet",
                    "truth_status": "hypothesis_teacher_signal",
                },
            )
        )
    if counts.get(CLICK_ACTION) or family == "click_dominant":
        target_rows.append(
            (
                "coordinate_vetting_signal",
                {
                    "signal_family": "coordinate_vetting_signal",
                    "recommended_packet": "coordinate_vetting",
                    "click_action": CLICK_ACTION,
                    "coordinate_map_status": "needs_local_verification",
                    "bulk_click_plan_allowed": False,
                    "required_next_local_verifier": "single_low_cost_click_probe",
                    "truth_status": "hypothesis_teacher_signal",
                },
            )
        )
    if int(game_summary.get("scorecard_levels_completed") or 0) == 0 and int(game_summary.get("scorecard_actions") or 0) >= DEFAULT_MIN_BUDGET_ACTIONS:
        target_rows.append(
            (
                "dead_probe_memory_signal",
                {
                    "signal_family": "dead_probe_memory_signal",
                    "recommended_packet": "dead_probe_memory",
                    "repeat_risk_count": pattern_counts.get("repeat_risk", 0),
                    "dead_probe_key_fields": [
                        "game_id",
                        "level_index",
                        "action_id",
                        "object_id_or_click_cell",
                        "delta_signature",
                        "local_hypothesis_id",
                    ],
                    "branch_action": "decay_or_block_repeated_low_novelty_probe",
                    "truth_status": "hypothesis_teacher_signal",
                },
            )
        )
    if int(game_summary.get("scorecard_actions") or 0) >= DEFAULT_MIN_BUDGET_ACTIONS or pattern_counts.get("resource_budget", 0):
        target_rows.append(
            (
                "budget_governor_signal",
                {
                    "signal_family": "budget_governor_signal",
                    "recommended_packet": "budget_governor",
                    "actions_spent": game_summary.get("scorecard_actions"),
                    "levels_completed": game_summary.get("scorecard_levels_completed"),
                    "resource_budget_pattern_count": pattern_counts.get("resource_budget", 0),
                    "trigger": "diagnostic_mode_when_budget_burns_without_objective_gain",
                    "truth_status": "hypothesis_teacher_signal",
                },
            )
        )
    if pattern_counts.get("goal_inference", 0):
        target_rows.append(
            (
                "goal_spec_candidate_signal",
                {
                    "signal_family": "goal_spec_candidate_signal",
                    "recommended_packet": "goal_spec_candidate_packet",
                    "candidate_types": ["goal", "timer", "resource", "progress_counter", "decoration", "unknown"],
                    "admission_status": "review_only",
                    "required_next_local_verifier": "engine_state_or_controlled_completion_delta",
                    "truth_status": "hypothesis_teacher_signal",
                },
            )
        )
    if pattern_counts.get("contradiction_revision", 0) >= DEFAULT_MIN_CONTRADICTION:
        target_rows.append(
            (
                "contradiction_replan_signal",
                {
                    "signal_family": "contradiction_replan_signal",
                    "recommended_packet": "contradiction_replan_gate",
                    "contradiction_count": pattern_counts.get("contradiction_revision", 0),
                    "stale_loop_rule": "two_same_basin_mismatches_force_new_representation_branch",
                    "truth_status": "hypothesis_teacher_signal",
                },
            )
        )

    return target_rows


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    rid = row.get("row_id", "<missing-row-id>")
    if row.get("schema") != ROW_SCHEMA:
        errors.append(f"{rid}: bad schema")
    if row.get("signal_family") not in SIGNAL_FAMILIES:
        errors.append(f"{rid}: unknown signal family {row.get('signal_family')!r}")
    if row.get("target", {}).get("truth_status") != "hypothesis_teacher_signal":
        errors.append(f"{rid}: target truth_status is not hypothesis_teacher_signal")
    evidence = row.get("evidence") or {}
    if evidence.get("truth_status") != "hypothesis_teacher_signal":
        errors.append(f"{rid}: evidence truth_status is not hypothesis_teacher_signal")
    if not evidence.get("source_summary") or not evidence.get("source_download_manifest"):
        errors.append(f"{rid}: missing source provenance")
    if "solve" in json.dumps(row.get("target", {})).lower() and row.get("target", {}).get("may_claim_solve"):
        errors.append(f"{rid}: target appears to permit solve claims")
    messages = row.get("messages") or []
    if len(messages) != 3:
        errors.append(f"{rid}: messages length is not 3")
    return errors


def export_dataset(
    *,
    mining_dir: Path,
    download_manifest_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    summary_path = mining_dir / "summary.json"
    summary = load_json(summary_path)
    download_manifest = load_json(download_manifest_path)
    manifest_by_game = {row["game_id"]: row for row in download_manifest.get("recordings") or []}

    rows: list[dict[str, Any]] = []
    for game_summary in summary.get("runs") or []:
        game_id = game_summary["game_id"]
        source_files = source_files_for_game(mining_dir, game_id)
        reasoning_path = source_files["reasoning"]
        reasoning_rows = load_jsonl(reasoning_path) if reasoning_path and reasoning_path.is_file() else []
        for signal_family, target in build_targets(game_summary, reasoning_rows):
            split = "heldout" if game_id in HELDOUT_GAME_IDS else "train"
            row_id = f"{safe_id(game_id)}__{signal_family}"
            samples = select_reasoning_samples(reasoning_rows, signal_family)
            rows.append(
                make_base_row(
                    row_id=row_id,
                    split=split,
                    signal_family=signal_family,
                    game_summary=game_summary,
                    recording_manifest=manifest_by_game.get(game_id),
                    reasoning_path=reasoning_path,
                    source_summary_path=summary_path,
                    download_manifest_path=download_manifest_path,
                    reasoning_samples=samples,
                    target=target,
                )
            )

    errors = []
    for row in rows:
        errors.extend(validate_row(row))
    if errors:
        raise SystemExit("row validation failed:\n" + "\n".join(errors[:50]))

    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    heldout_path = out_dir / "heldout.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at_utc": utc_now(),
        "dataset": out_dir.name,
        "status": "exported",
        "truth_status": "hypothesis_teacher_signal",
        "boundary": "public ARC recording teacher data only; not local CONTRA evidence",
        "row_schema": ROW_SCHEMA,
        "signal_families": SIGNAL_FAMILIES,
        "source_scorecard_id": summary.get("scorecard", {}).get("card_id"),
        "source_scorecard_model": summary.get("scorecard", {}).get("model"),
        "source_scorecard_summary": summary.get("scorecard"),
        "source_mining_summary": relpath(summary_path),
        "source_mining_summary_sha256": sha256_file(summary_path),
        "source_download_manifest": relpath(download_manifest_path),
        "source_download_manifest_sha256": sha256_file(download_manifest_path),
        "exporter_source": "codexlab/arc_export_teacher_signal_dataset.py",
        "exporter_source_sha256": sha256_file(ROOT / "codexlab" / "arc_export_teacher_signal_dataset.py"),
        "train_jsonl": relpath(train_path),
        "train_jsonl_sha256": sha256_file(train_path),
        "heldout_jsonl": relpath(heldout_path),
        "heldout_jsonl_sha256": sha256_file(heldout_path),
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "row_count": len(rows),
        "heldout_game_ids": sorted(HELDOUT_GAME_IDS),
        "counts_by_signal_family": {
            family: sum(1 for row in rows if row["signal_family"] == family) for family in SIGNAL_FAMILIES
        },
        "next_valid_step": "build CONTRA packet producers and offline signal evaluators; do not claim gameplay solve",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mining-dir",
        type=Path,
        default=Path("codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z"),
    )
    parser.add_argument(
        "--download-manifest",
        type=Path,
        default=Path("codexlab/runs/arc_public_recordings_anthropic-opus-4-8-high_20260601T222454Z/download_manifest.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("codexlab/datasets/arc_opus48_teacher_signals_v0"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = export_dataset(
        mining_dir=resolve_repo_path(str(args.mining_dir)),
        download_manifest_path=resolve_repo_path(str(args.download_manifest)),
        out_dir=resolve_repo_path(str(args.out_dir)),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
