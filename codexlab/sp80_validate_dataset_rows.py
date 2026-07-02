#!/usr/bin/env python3
"""Validate every row of an exported SP80 control SFT dataset against its schema.

This is the standalone, on-disk counterpart to the exporter's inline
validate_row(): it re-checks the structural schema `codexlab.sp80_control_sft_row.v1`
for each row of train.jsonl and heldout.jsonl, so a dataset can be re-validated
after the fact (e.g. before training) without re-running the export.

Per row it checks: schema tag, required top-level keys and their types, the
row_id shape (sp80_l<NN>_<case>), level_index == calibration_slice_index, the
input block, a primitive-only target with no coordinates, exactly 3 chat
messages with system/user/assistant roles and an assistant primitive that
matches `target`, and the evidence provenance flags (controlled_runtime true;
route/human/coordinate flags false; status executed).

Usage:
  python3 codexlab/sp80_validate_dataset_rows.py codexlab/datasets/sp80_control_sft_v0
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from players.sp80.registry import CONTROL_PRIMITIVES

ROOT = Path(__file__).resolve().parents[1]
ROW_SCHEMA = "codexlab.sp80_control_sft_row.v1"

ALLOWED_PRIMITIVES = CONTROL_PRIMITIVES

ROW_ID_RE = re.compile(r"^sp80_l\d{2}_[a-z0-9_]+$")

TOP_LEVEL_TYPES: dict[str, type | tuple[type, ...]] = {
    "schema": str,
    "row_id": str,
    "split": str,
    "game": str,
    "calibration_slice_index": int,
    "level_index": int,
    "case_id": str,
    "input": dict,
    "target": dict,
    "messages": list,
    "evidence": dict,
}

INPUT_KEYS = ["game", "task", "intent", "allowed_primitives", "view_state", "observation", "controller_contract"]
MESSAGE_ROLES = ["system", "user", "assistant"]
EVIDENCE_FALSE_FLAGS = ["route_or_replay_used", "human_trace_used", "coordinate_script_used"]


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)  # raises if outside the repo
    return resolved


def validate_row(row: dict[str, Any], *, expected_split: str, line_no: int) -> list[str]:
    rid = row.get("row_id", f"<{expected_split}.jsonl:{line_no}>")
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"{rid}: {msg}")

    if row.get("schema") != ROW_SCHEMA:
        err(f"schema is {row.get('schema')!r}, expected {ROW_SCHEMA!r}")

    for key, typ in TOP_LEVEL_TYPES.items():
        if key not in row:
            err(f"missing key {key!r}")
        elif not isinstance(row[key], typ) or isinstance(row[key], bool):
            err(f"key {key!r} has type {type(row[key]).__name__}, expected {typ}")

    if isinstance(row.get("row_id"), str) and not ROW_ID_RE.match(row["row_id"]):
        err(f"row_id {row['row_id']!r} does not match sp80_l<NN>_<case>")

    if row.get("split") != expected_split:
        err(f"split is {row.get('split')!r}, expected {expected_split!r} (file it lives in)")

    if isinstance(row.get("level_index"), int) and isinstance(row.get("calibration_slice_index"), int):
        if row["level_index"] != row["calibration_slice_index"]:
            err(f"level_index {row['level_index']} != calibration_slice_index {row['calibration_slice_index']}")

    # input block
    input_obj = row.get("input")
    if isinstance(input_obj, dict):
        for k in INPUT_KEYS:
            if k not in input_obj:
                err(f"input missing key {k!r}")
        if input_obj.get("allowed_primitives") != ALLOWED_PRIMITIVES:
            err("input.allowed_primitives does not match the allowed primitive set")
        view_state = input_obj.get("view_state")
        if not isinstance(view_state, dict):
            err("input.view_state is not an object")
        else:
            if view_state.get("schema") != "codexlab.sp80_view_state.v1":
                err("input.view_state schema is not codexlab.sp80_view_state.v1")
            if view_state.get("phase") != "before":
                err("input.view_state phase is not before")
            if view_state.get("look_order") != "before_action":
                err("input.view_state look_order is not before_action")
            if view_state.get("raw_pixels_are_observation_ground_truth") is not True:
                err("input.view_state raw pixel ground-truth flag is not True")
        if not isinstance(input_obj.get("observation"), dict):
            err("input.observation is not an object")
        goal_condition = input_obj.get("goal_condition")
        if goal_condition is not None:
            if not isinstance(goal_condition, dict):
                err("input.goal_condition is not an object")
            else:
                if goal_condition.get("condition") != "pour_into_buckets":
                    err("input.goal_condition.condition is not pour_into_buckets")
                if goal_condition.get("target_label") != "bucket":
                    err("input.goal_condition.target_label is not bucket")
                if goal_condition.get("route_or_replay_used") is not False:
                    err("input.goal_condition.route_or_replay_used is not False")
                if goal_condition.get("coordinate_target_used") is not False:
                    err("input.goal_condition.coordinate_target_used is not False")

    # target: primitive-only, allowed, no coordinates
    target = row.get("target")
    primitive = None
    if isinstance(target, dict):
        primitive = target.get("primitive")
        if primitive not in ALLOWED_PRIMITIVES:
            err(f"target primitive not allowed: {primitive!r}")
        if set(target.keys()) != {"primitive"}:
            err(f"target has extra keys: {sorted(set(target.keys()) - {'primitive'})}")
        if "coordinate" in json.dumps(target).lower():
            err("target appears to contain coordinates")
    else:
        err("target is not an object")

    # messages: 3 turns, roles in order, assistant primitive matches target
    messages = row.get("messages")
    if isinstance(messages, list):
        if len(messages) != 3:
            err(f"messages length is {len(messages)}, expected 3")
        else:
            for i, (msg, role) in enumerate(zip(messages, MESSAGE_ROLES)):
                if not isinstance(msg, dict) or msg.get("role") != role:
                    err(f"messages[{i}] role is {msg.get('role') if isinstance(msg, dict) else msg!r}, expected {role!r}")
            assistant = messages[2] if isinstance(messages[2], dict) else {}
            try:
                payload = json.loads(assistant.get("content", ""))
                if payload != {"primitive": primitive}:
                    err(f"assistant content {payload!r} does not match target primitive {primitive!r}")
            except (json.JSONDecodeError, TypeError):
                err("assistant message content is not the expected primitive JSON")

    # evidence provenance flags
    evidence = row.get("evidence")
    if isinstance(evidence, dict):
        for flag in EVIDENCE_FALSE_FLAGS:
            if evidence.get(flag) is not False:
                err(f"evidence.{flag} is {evidence.get(flag)!r}, expected False")
        if evidence.get("controlled_runtime") is not True:
            err(f"evidence.controlled_runtime is {evidence.get('controlled_runtime')!r}, expected True")
        if evidence.get("status") != "executed":
            err(f"evidence.status is {evidence.get('status')!r}, expected 'executed'")
    else:
        err("evidence is not an object")

    return errors


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows


def validate_dataset(dataset_dir: Path) -> dict[str, Any]:
    errors: list[str] = []
    counts: dict[str, int] = {}
    for split in ("train", "heldout"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.is_file():
            raise SystemExit(f"missing dataset split: {path}")
        rows = load_jsonl(path)
        counts[split] = len(rows)
        for line_no, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                errors.append(f"{split}.jsonl:{line_no}: row is not an object")
                continue
            errors.extend(validate_row(row, expected_split=split, line_no=line_no))

    return {
        "schema": "codexlab.sp80_dataset_row_validation_report.v1",
        "dataset_dir": dataset_dir.relative_to(ROOT).as_posix(),
        "status": "passed" if not errors else "failed",
        "row_schema": ROW_SCHEMA,
        "train_count": counts.get("train", 0),
        "heldout_count": counts.get("heldout", 0),
        "error_count": len(errors),
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path("codexlab/datasets/sp80_control_sft_v0"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = resolve_repo_path(str(args.dataset_dir))
    report = validate_dataset(dataset_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
