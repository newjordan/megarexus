#!/usr/bin/env python3
"""Run the SP80 control harness across calibration slices.

The SP80 environment API calls these "levels"; this script reports them as
calibration slices because these are internal control checks, not gameplay
completion claims.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sp80_base_controller import ROOT, load_sp80_class, sha256_file


def detect_level_count() -> int:
    game = load_sp80_class()()
    levels = getattr(game, "_levels", None)
    if levels is None:
        raise RuntimeError("could not find SP80 _levels")
    return len(levels)


def run_command(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["tiny_mlp", "lora_adapter", "qwen_http"], default="tiny_mlp")
    parser.add_argument("--levels", default="all", help="'all' or comma-separated calibration slice indexes")
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_player_harness_tiny_mlp_levels"))
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--qwen-url", default=None)
    parser.add_argument("--qwen-mode", choices=["simple_json", "openai_chat"], default="simple_json")
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--qwen-timeout-s", type=float, default=20.0)
    parser.add_argument("--lora-adapter", type=Path, default=None)
    return parser


def parse_levels(value: str, level_count: int) -> list[int]:
    if value == "all":
        return list(range(level_count))
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    for level in levels:
        if level < 0 or level >= level_count:
            raise ValueError(f"calibration slice {level} outside available range 0..{level_count - 1}")
    return levels


def main() -> int:
    args = build_parser().parse_args()
    level_count = detect_level_count()
    levels = parse_levels(args.levels, level_count)
    sweep_dir = ROOT / args.out_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for level_index in levels:
        level_dir = sweep_dir / f"level_{level_index:02d}"
        harness_cmd = [
            sys.executable,
            "codexlab/sp80_player_harness.py",
            "--backend",
            args.backend,
            "--level-index",
            str(level_index),
            "--out-dir",
            level_dir.relative_to(ROOT).as_posix(),
            "--scale",
            str(max(1, int(args.scale))),
        ]
        if args.strict:
            harness_cmd.append("--strict")
        if args.backend == "lora_adapter":
            if args.lora_adapter is None:
                raise SystemExit("--lora-adapter is required for --backend lora_adapter")
            harness_cmd.extend(["--lora-adapter", str(args.lora_adapter)])
        if args.backend == "qwen_http":
            if args.qwen_url:
                harness_cmd.extend(["--qwen-url", args.qwen_url])
            harness_cmd.extend(["--qwen-mode", args.qwen_mode])
            if args.qwen_model:
                harness_cmd.extend(["--qwen-model", args.qwen_model])
            harness_cmd.extend(["--qwen-timeout-s", str(args.qwen_timeout_s)])

        harness = run_command(harness_cmd, cwd=ROOT)
        verifier_cmd = [
            sys.executable,
            "codexlab/sp80_verify_run.py",
            level_dir.relative_to(ROOT).as_posix(),
        ]
        verifier = run_command(verifier_cmd, cwd=ROOT) if harness["returncode"] == 0 else None

        row = {
            "calibration_slice_index": level_index,
            "level_index": level_index,
            "run_dir": level_dir.relative_to(ROOT).as_posix(),
            "harness_returncode": harness["returncode"],
            "verifier_returncode": None if verifier is None else verifier["returncode"],
            "status": "passed"
            if harness["returncode"] == 0 and verifier is not None and verifier["returncode"] == 0
            else "failed",
            "harness_stdout_tail": harness["stdout"][-4000:],
            "harness_stderr_tail": harness["stderr"][-4000:],
            "verifier_stdout": None if verifier is None else verifier["stdout"],
            "verifier_stderr": None if verifier is None else verifier["stderr"],
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "calibration_slice_index",
                        "status",
                        "harness_returncode",
                        "verifier_returncode",
                    ]
                }
            )
        )
        if args.strict and row["status"] != "passed":
            break

    summary = {
        "schema": "codexlab.sp80_level_sweep.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
        "backend": args.backend,
        "reporting_term": "calibration_slice",
        "reporting_note": "level_index is retained only as SP80 environment provenance; slice status is not gameplay completion.",
        "detected_calibration_slice_count": level_count,
        "requested_calibration_slices": levels,
        "passed_calibration_slices": [row["level_index"] for row in rows if row["status"] == "passed"],
        "failed_calibration_slices": [row["level_index"] for row in rows if row["status"] != "passed"],
        "detected_level_count": level_count,
        "requested_levels": levels,
        "passed_levels": [row["level_index"] for row in rows if row["status"] == "passed"],
        "failed_levels": [row["level_index"] for row in rows if row["status"] != "passed"],
        "sweep_source": "codexlab/sp80_sweep_levels.py",
        "sweep_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_sweep_levels.py"),
        "rows": rows,
    }
    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jsonl_path = sweep_dir / "summary.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
