#!/usr/bin/env python3
"""Export verified SP80 control packets into primitive-only SFT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from players.sp80.registry import CONTROL_PRIMITIVES

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_PRIMITIVES = CONTROL_PRIMITIVES

STATE_KEYS = [
    "state",
    "mode",
    "rotation_quarters",
    "steps_remaining",
    "selected",
    "selected_name",
    "selected_coord",
    "selected_display_coord",
    "selected_shape",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def run_verifier(run_dir: Path) -> dict[str, Any]:
    cmd = [sys.executable, "codexlab/sp80_verify_run.py", relpath(run_dir)]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def parse_heldout_levels(value: str, passed_levels: list[int]) -> list[int]:
    if not passed_levels:
        raise ValueError("no passed calibration slices available")
    if value == "last":
        return [max(passed_levels)]
    if value == "none":
        return []
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    missing = sorted(set(levels) - set(passed_levels))
    if missing:
        raise ValueError(f"heldout calibration slices are not verifier-passed slices: {missing}")
    return levels


def state_subset(state: dict[str, Any]) -> dict[str, Any]:
    return {key: state.get(key) for key in STATE_KEYS}


def frame_ref(frame_artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "png": frame_artifact.get("png"),
        "raw_grid": frame_artifact.get("raw_grid"),
    }


def qwen_messages(input_obj: dict[str, Any], primitive: str) -> list[dict[str, str]]:
    system = (
        "Choose exactly one SP80 control primitive. Return JSON only. "
        "Do not emit coordinates, routes, or multiple actions."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(input_obj, sort_keys=True)},
        {"role": "assistant", "content": json.dumps({"primitive": primitive}, sort_keys=True)},
    ]


def build_row(
    *,
    packet: dict[str, Any],
    level_index: int,
    split: str,
    packet_path: Path,
    packet_line: int,
    brain_calls_path: Path,
    result_path: Path,
    drift_report_path: Path,
    recommended_training_space: str,
) -> dict[str, Any]:
    primitive = str(packet["expected_primitive"])
    before_state = packet.get("before_state") or {}
    target_sprite = packet.get("target_sprite") or {}
    action = packet.get("action") or {}
    goal_condition = packet.get("goal_condition") or {}
    input_obj = {
        "game": "sp80_liquid_pour",
        "task": "choose_one_control_primitive",
        "intent": packet.get("intent"),
        "allowed_primitives": ALLOWED_PRIMITIVES,
        "view_state": packet.get("view_before"),
        "goal_condition": goal_condition,
        "observation": {
            "state": state_subset(before_state),
            "target_sprite": target_sprite,
            "setup": packet.get("setup"),
            "setup_action_count": len(packet.get("setup_actions") or []),
        },
        "controller_contract": {
            "output": "one primitive",
            "may_choose_coordinates": False,
            "may_emit_route": False,
            "motor_executes_actions": True,
        },
    }
    row = {
        "schema": "codexlab.sp80_control_sft_row.v1",
        "row_id": f"sp80_l{level_index:02d}_{packet['case_id']}",
        "split": split,
        "game": "sp80_liquid_pour",
        "calibration_slice_index": level_index,
        "level_index": level_index,
        "case_id": packet.get("case_id"),
        "input": input_obj,
        "target": {"primitive": primitive},
        "messages": qwen_messages(input_obj, primitive),
        "evidence": {
            "source_packet": relpath(packet_path),
            "source_packet_line": packet_line,
            "source_result": relpath(result_path),
            "source_brain_calls": relpath(brain_calls_path),
            "source_control_drift_report": relpath(drift_report_path),
            "status": packet.get("status"),
            "controlled_runtime": packet.get("controlled_runtime"),
            "route_or_replay_used": packet.get("route_or_replay_used"),
            "human_trace_used": packet.get("human_trace_used"),
            "coordinate_script_used": packet.get("coordinate_script_used"),
            "action": {
                "primitive": action.get("primitive"),
                "id": action.get("id"),
                "name": action.get("name"),
            },
            "selected_coord_delta": packet.get("selected_coord_delta"),
            "selected_display_coord_delta": packet.get("selected_display_coord_delta"),
            "recommended_training_space": recommended_training_space,
            "view_before": packet.get("view_before"),
            "view_after": packet.get("view_after"),
            "action_visual_correlation": packet.get("action_visual_correlation"),
            "frame_delta": packet.get("delta"),
            "goal_condition": goal_condition,
            "goal_progress": packet.get("goal_progress"),
            "action_frame_count": packet.get("action_frame_count"),
            "before_frame": frame_ref(packet.get("before_frame") or {}),
            "after_frame": frame_ref(packet.get("after_frame") or {}),
        },
    }
    return row


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []
    target = row.get("target") or {}
    primitive = target.get("primitive")
    if primitive not in ALLOWED_PRIMITIVES:
        errors.append(f"{row.get('row_id')}: target primitive not allowed: {primitive}")
    messages = row.get("messages") or []
    if len(messages) != 3:
        errors.append(f"{row.get('row_id')}: messages length is not 3")
    if "coordinates" in json.dumps(target).lower():
        errors.append(f"{row.get('row_id')}: target appears to contain coordinates")
    evidence = row.get("evidence") or {}
    for flag in ["route_or_replay_used", "human_trace_used", "coordinate_script_used"]:
        if evidence.get(flag) is not False:
            errors.append(f"{row.get('row_id')}: evidence {flag} is not false")
    if evidence.get("controlled_runtime") is not True:
        errors.append(f"{row.get('row_id')}: controlled_runtime is not true")
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/datasets/sp80_control_sft_v0"))
    parser.add_argument("--heldout-levels", default="last", help="'last', 'none', or comma-separated calibration slice indexes")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sweep_dir = resolve_repo_path(str(args.sweep_dir))
    out_dir = resolve_repo_path(str(args.out_dir))

    summary_path = sweep_dir / "summary.json"
    drift_report_path = sweep_dir / "control_drift_report.json"
    if not summary_path.is_file():
        raise SystemExit(f"missing sweep summary: {summary_path}")
    if not drift_report_path.is_file():
        raise SystemExit(f"missing control drift report: {drift_report_path}")

    summary = load_json(summary_path)
    drift_report = load_json(drift_report_path)
    if summary.get("schema") != "codexlab.sp80_level_sweep.v1":
        raise SystemExit("bad sweep summary schema")
    if summary.get("status") != "passed":
        raise SystemExit("sweep summary is not passed")
    if drift_report.get("schema") != "codexlab.sp80_control_drift_report.v1":
        raise SystemExit("bad control drift report schema")
    if drift_report.get("status") == "failed":
        raise SystemExit("control drift report failed")

    passed_levels = [int(level) for level in summary.get("passed_levels", [])]
    heldout_levels = parse_heldout_levels(args.heldout_levels, passed_levels)
    train_levels = [level for level in passed_levels if level not in set(heldout_levels)]
    recommended_training_space = str(drift_report.get("recommended_training_space") or "selected_display_coord_delta")

    rows: list[dict[str, Any]] = []
    verifier_records: dict[str, Any] = {}
    for level_index in passed_levels:
        level_dir = sweep_dir / f"level_{level_index:02d}"
        verifier = run_verifier(level_dir)
        verifier_records[str(level_index)] = verifier
        if verifier["returncode"] != 0:
            raise SystemExit(f"verifier failed for level {level_index}: {verifier['stderr']}")

        packet_path = level_dir / "packets.jsonl"
        brain_calls_path = level_dir / "brain_calls.jsonl"
        result_path = level_dir / "result.json"
        packets = load_jsonl(packet_path)
        split = "heldout" if level_index in heldout_levels else "train"
        for line_no, packet in enumerate(packets, start=1):
            if packet.get("status") != "executed":
                raise SystemExit(f"packet not executed at {packet_path}:{line_no}")
            rows.append(
                build_row(
                    packet=packet,
                    level_index=level_index,
                    split=split,
                    packet_path=packet_path,
                    packet_line=line_no,
                    brain_calls_path=brain_calls_path,
                    result_path=result_path,
                    drift_report_path=drift_report_path,
                    recommended_training_space=recommended_training_space,
                )
            )

    validation_errors = []
    for row in rows:
        validation_errors.extend(validate_row(row))
    if validation_errors:
        raise SystemExit("\n".join(validation_errors))

    train_rows = [row for row in rows if row["split"] == "train"]
    heldout_rows = [row for row in rows if row["split"] == "heldout"]
    train_path = out_dir / "train.jsonl"
    heldout_path = out_dir / "heldout.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(heldout_path, heldout_rows)

    manifest = {
        "schema": "codexlab.sp80_control_sft_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "exported",
        "dataset": "sp80_control_sft_v0",
        "reporting_term": "calibration_slice",
        "reporting_note": "level_index is retained only as SP80 environment provenance; slice rows are not gameplay-completion claims.",
        "source_sweep_dir": relpath(sweep_dir),
        "source_summary": relpath(summary_path),
        "source_summary_sha256": sha256_file(summary_path),
        "source_control_drift_report": relpath(drift_report_path),
        "source_control_drift_report_sha256": sha256_file(drift_report_path),
        "exporter_source": "codexlab/sp80_export_control_dataset.py",
        "exporter_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_export_control_dataset.py"),
        "controller_boundary": "primitive-only target; no routes, no coordinates",
        "recommended_training_space": recommended_training_space,
        "allowed_primitives": ALLOWED_PRIMITIVES,
        "passed_calibration_slices": passed_levels,
        "train_calibration_slices": train_levels,
        "heldout_calibration_slices": heldout_levels,
        "passed_levels": passed_levels,
        "train_levels": train_levels,
        "heldout_levels": heldout_levels,
        "row_count": len(rows),
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "train_jsonl": relpath(train_path),
        "train_jsonl_sha256": sha256_file(train_path),
        "heldout_jsonl": relpath(heldout_path),
        "heldout_jsonl_sha256": sha256_file(heldout_path),
        "verifier_records": verifier_records,
        "drift_ping_ids": [ping.get("id") for ping in drift_report.get("pings", [])],
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "exported",
                "manifest": relpath(manifest_path),
                "train_rows": len(train_rows),
                "heldout_rows": len(heldout_rows),
                "train_calibration_slices": train_levels,
                "heldout_calibration_slices": heldout_levels,
                "train_levels": train_levels,
                "heldout_levels": heldout_levels,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
