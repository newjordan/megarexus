#!/usr/bin/env python3
"""Run CONTRA through the SP80 player harness for real persistent episodes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sp80_base_controller import ROOT, sha256_file


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def read_json_if_present(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def parse_levels(value: str) -> list[int]:
    levels = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not levels:
        raise ValueError("--levels must contain at least one level index")
    for level in levels:
        if level < 0:
            raise ValueError(f"level {level} is invalid")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="0,1", help="comma-separated SP80 ARCPrize round level indexes")
    parser.add_argument(
        "--backend",
        choices=[
            "tiny_mlp",
            "lora_adapter",
            "qwen_http",
            "puffer_policy",
            "paint_control_skill",
            "spill_planner",
        ],
        default="tiny_mlp",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--skip-report", action="store_true", help="run episodes only; do not build Angel report")
    parser.add_argument("--qwen-url", default=None)
    parser.add_argument("--qwen-mode", choices=["simple_json", "openai_chat"], default="simple_json")
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--qwen-timeout-s", type=float, default=20.0)
    parser.add_argument("--lora-adapter", type=Path, default=None)
    parser.add_argument("--puffer-policy", type=Path, default=None)
    parser.add_argument("--require-pufferlib", action="store_true")
    parser.add_argument("--target-bucket-index", type=int, default=0)
    parser.add_argument("--paint-skill-checkpoint", type=Path, default=None)
    parser.add_argument("--paint-skill-env", default="rexus_sp80_paintflow_l1_multi")
    parser.add_argument("--paint-skill-require-native", action="store_true")
    parser.add_argument("--paint-skill-forward-model-ledger", type=Path, default=None)
    parser.add_argument("--spill-search-time-budget", type=float, default=12.0)
    parser.add_argument("--spill-search-restarts", type=int, default=12)
    parser.add_argument(
        "--compact-episode-artifacts",
        action="store_true",
        help="pass compact episode JSONL logging through to the SP80 harness",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    levels = parse_levels(args.levels)
    level_label = "_".join(f"{level:02d}" for level in levels)
    run_dir = args.out_dir or Path(f"codexlab/runs/contra_sp80_episode_levels_{level_label}_{utc_stamp()}")
    run_dir = run_dir if run_dir.is_absolute() else ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for level in levels:
        level_dir = run_dir / f"level_{level:02d}"
        harness_cmd = [
            sys.executable,
            "-B",
            "codexlab/sp80_player_harness.py",
            "--backend",
            args.backend,
            "--mode",
            "episode",
            "--level-index",
            str(level),
            "--out-dir",
            level_dir.relative_to(ROOT).as_posix(),
            "--scale",
            str(max(1, int(args.scale))),
            "--max-actions",
            str(max(1, int(args.max_actions))),
        ]
        if args.compact_episode_artifacts:
            harness_cmd.append("--compact-episode-artifacts")
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
        if args.backend == "puffer_policy":
            if args.puffer_policy is None:
                raise SystemExit("--puffer-policy is required for --backend puffer_policy")
            harness_cmd.extend(["--puffer-policy", str(args.puffer_policy)])
            if args.require_pufferlib:
                harness_cmd.append("--require-pufferlib")
        if args.backend == "paint_control_skill":
            harness_cmd.extend(["--paint-skill-env", args.paint_skill_env])
            harness_cmd.extend(["--target-bucket-index", str(int(args.target_bucket_index))])
            if args.paint_skill_checkpoint is not None:
                harness_cmd.extend(["--paint-skill-checkpoint", str(args.paint_skill_checkpoint)])
            if args.paint_skill_require_native:
                harness_cmd.append("--paint-skill-require-native")
            if args.paint_skill_forward_model_ledger is not None:
                harness_cmd.extend(["--paint-skill-forward-model-ledger", str(args.paint_skill_forward_model_ledger)])
        if args.backend == "spill_planner":
            harness_cmd.extend(["--spill-search-time-budget", str(float(args.spill_search_time_budget))])
            harness_cmd.extend(["--spill-search-restarts", str(int(args.spill_search_restarts))])

        harness = run_command(harness_cmd, cwd=ROOT)
        verifier_cmd = [
            sys.executable,
            "-B",
            "codexlab/sp80_verify_episode_run.py",
            "--run-dir",
            level_dir.relative_to(ROOT).as_posix(),
        ]
        verifier = run_command(verifier_cmd, cwd=ROOT) if harness["returncode"] == 0 else None
        result = read_json_if_present(level_dir / "result.json")
        row = {
            "level_index": level,
            "run_dir": level_dir.relative_to(ROOT).as_posix(),
            "harness_command": " ".join(harness_cmd),
            "harness_returncode": harness["returncode"],
            "harness_stdout_tail": harness["stdout"][-4000:],
            "harness_stderr_tail": harness["stderr"][-4000:],
            "verifier_command": " ".join(verifier_cmd),
            "verifier_returncode": None if verifier is None else verifier["returncode"],
            "verifier_stdout": None if verifier is None else verifier["stdout"],
            "verifier_stderr": None if verifier is None else verifier["stderr"],
            "status": "nominal"
            if harness["returncode"] == 0 and verifier is not None and verifier["returncode"] == 0
            else "failed",
            "episode_status": result.get("status"),
            "action_count": int(result.get("action_count") or 0),
            "levels_completed_delta": int(result.get("levels_completed_delta") or 0),
            "mode": result.get("mode"),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "level_index": row["level_index"],
                    "status": row["status"],
                    "episode_status": row["episode_status"],
                    "action_count": row["action_count"],
                    "levels_completed_delta": row["levels_completed_delta"],
                },
                sort_keys=True,
            )
        )

    summary = {
        "schema": "codexlab.sp80_contra_episode_levels.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "nominal" if all(row["status"] == "nominal" for row in rows) else "failed",
        "backend": args.backend,
        "game_id": "sp80-589a99af",
        "mode": "episode",
        "condition": "CONTRA-driven persistent SP80 ARCPrize episode levels",
        "requested_levels": levels,
        "completed_levels": [row["level_index"] for row in rows if int(row["levels_completed_delta"]) > 0],
        "failed_or_incomplete_levels": [row["level_index"] for row in rows if int(row["levels_completed_delta"]) <= 0],
        "levels_completed_delta_total": sum(int(row["levels_completed_delta"]) for row in rows),
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "runner_source": "codexlab/sp80_run_contra_levels.py",
        "runner_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_run_contra_levels.py"),
        "harness_source": "codexlab/sp80_player_harness.py",
        "harness_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_player_harness.py"),
        "report_source": "codexlab/gameplay_report_harness.py",
        "report_source_sha256": sha256_file(ROOT / "codexlab" / "gameplay_report_harness.py"),
        "boundary": "Persistent episode controlled_run. Not solve evidence for a level unless that row records levels_completed_delta > 0.",
        "rows": rows,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (run_dir / "summary.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    report_returncode = None
    report_verify_returncode = None
    if not args.skip_report:
        report_dir = run_dir / "angel_report"
        report_cmd = [
            sys.executable,
            "-B",
            "codexlab/gameplay_report_harness.py",
            "--run-dir",
            run_dir.relative_to(ROOT).as_posix(),
            "--out-dir",
            report_dir.relative_to(ROOT).as_posix(),
        ]
        report = run_command(report_cmd, cwd=ROOT)
        report_returncode = report["returncode"]
        verify_cmd = [
            sys.executable,
            "-B",
            "codexlab/angel_report_verify.py",
            report_dir.relative_to(ROOT).as_posix(),
            "--strict-png-ledger",
            "--require-gameplay-visual-contract",
        ]
        report_verify = run_command(verify_cmd, cwd=ROOT) if report_returncode == 0 else None
        report_verify_returncode = None if report_verify is None else report_verify["returncode"]
        manifest = read_json_if_present(report_dir / "report_manifest.json")
        summary["report"] = {
            "report_dir": report_dir.relative_to(ROOT).as_posix(),
            "report_command": " ".join(report_cmd),
            "report_returncode": report_returncode,
            "report_stdout": report["stdout"],
            "report_stderr": report["stderr"],
            "verify_command": " ".join(verify_cmd),
            "verify_returncode": report_verify_returncode,
            "verify_stdout": None if report_verify is None else report_verify["stdout"],
            "verify_stderr": None if report_verify is None else report_verify["stderr"],
            "angel_viewer_url": manifest.get("angel_viewer_url"),
            "contraview_report_json": manifest.get("contraview_report_json"),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ok = summary["status"] == "nominal"
    if not args.skip_report:
        ok = ok and report_returncode == 0 and report_verify_returncode == 0
    print(
        json.dumps(
            {
                "status": "reported" if ok and not args.skip_report else summary["status"],
                "run_dir": summary["run_dir"],
                "completed_levels": summary["completed_levels"],
                "levels_completed_delta_total": summary["levels_completed_delta_total"],
                "angel_viewer_url": (summary.get("report") or {}).get("angel_viewer_url"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
