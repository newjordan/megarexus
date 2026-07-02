#!/usr/bin/env python3
"""Probe PufferLib compatibility with the SP80 primitive-policy boundary."""

from __future__ import annotations

import argparse
import html
import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from players.sp80.registry import CONTROL_PRIMITIVES
from sp80_base_controller import ROOT, sha256_file
from sp80_puffer_policy import POLICY_SCHEMA, PufferPolicyBrain, inspect_pufferlib


PROBE_SCHEMA = "codexlab.sp80_pufferlib_probe.v1"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_html(path: Path, result: dict[str, Any]) -> None:
    status = html.escape(str(result["status"]))
    rows = [
        ("Run label", result["condition"]["run_label"]),
        ("PufferLib available", result["pufferlib"]["available"]),
        ("PufferLib version", result["pufferlib"].get("version")),
        ("Policy primitive", (result.get("policy_decision") or {}).get("primitive")),
        ("Harness status", (result.get("harness_smoke") or {}).get("status")),
        ("Boundary", result["condition"]["truth_boundary"]),
    ]
    body = "\n".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in rows
    )
    path.write_text(
        "\n".join(
            [
                "<!doctype html>",
                "<html><head><meta charset='utf-8'><title>SP80 PufferLib Probe</title>",
                "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:960px}"
                "table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:.4rem .6rem;text-align:left}"
                "pre{white-space:pre-wrap;background:#f5f5f5;padding:1rem}</style></head><body>",
                f"<h1>SP80 PufferLib Probe: {status}</h1>",
                "<table>",
                body,
                "</table>",
                "<h2>Raw Summary</h2>",
                f"<pre>{html.escape(json.dumps(result, indent=2, sort_keys=True))}</pre>",
                "</body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def default_policy() -> dict[str, Any]:
    probabilities = {primitive: 0.01 for primitive in CONTROL_PRIMITIVES}
    probabilities["move_right"] = 0.90
    return {
        "schema": POLICY_SCHEMA,
        "source": "codexlab scout default; not a trained PufferLib export",
        "policy_kind": "manual_compatibility_scout",
        "primitive_probabilities": probabilities,
        "case_primitive_overrides": {
            "select_object": "select_object",
            "move_up": "move_up",
            "move_down": "move_down",
            "move_left": "move_left",
            "move_right": "move_right",
            "activate_flow": "activate_flow",
            "no_selected_motion": "no_selected_motion",
            "approach_nearest_bucket": "move_down",
        },
    }


def run_harness(policy_path: Path, out_dir: Path, require_pufferlib: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-B",
        "codexlab/sp80_player_harness.py",
        "--backend",
        "puffer_policy",
        "--puffer-policy",
        policy_path.relative_to(ROOT).as_posix() if policy_path.is_relative_to(ROOT) else str(policy_path),
        "--out-dir",
        out_dir.relative_to(ROOT).as_posix() if out_dir.is_relative_to(ROOT) else str(out_dir),
        "--mode",
        "action_set",
    ]
    if require_pufferlib:
        cmd.append("--require-pufferlib")
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": cmd,
        "returncode": int(completed.returncode),
        "status": "passed" if completed.returncode == 0 else "failed",
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def build_sample_request() -> dict[str, Any]:
    return {
        "mode": "probe",
        "intent": "Choose one primitive for compatibility probing",
        "allowed_primitives": CONTROL_PRIMITIVES,
        "observation": {"state": {"selected": False, "probe": True}},
        "setup_actions": [],
        "action_memory": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--write-default-policy", action="store_true")
    parser.add_argument("--run-harness", action="store_true")
    parser.add_argument("--require-pufferlib", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or ROOT / "codexlab" / "runs" / f"pufferlib_sp80_probe_{stamp}"
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_path = args.policy or out_dir / "puffer_policy.json"
    if not policy_path.is_absolute():
        policy_path = ROOT / policy_path
    if args.write_default_policy or args.policy is None:
        write_json(policy_path, default_policy())

    pufferlib = inspect_pufferlib()
    policy_decision = None
    policy_error = None
    try:
        brain = PufferPolicyBrain(policy_path, require_pufferlib=args.require_pufferlib)
        policy_decision = brain.decide(build_sample_request()).to_packet()
    except Exception as exc:  # noqa: BLE001 - probe records failure.
        policy_error = {"error_type": type(exc).__name__, "error_message": str(exc)}

    harness_smoke = None
    if args.run_harness:
        harness_smoke = run_harness(policy_path, out_dir / "harness_action_set", args.require_pufferlib)

    status = "passed"
    if args.require_pufferlib and not pufferlib.get("available"):
        status = "failed"
    if policy_error is not None:
        status = "failed"
    if harness_smoke is not None and harness_smoke["returncode"] != 0:
        status = "failed"

    result = {
        "schema": PROBE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": status,
        "pufferlib": pufferlib,
        "policy_decision": policy_decision,
        "policy_error": policy_error,
        "harness_smoke": harness_smoke,
        "artifacts": {
            "result_json": (out_dir / "result.json").relative_to(ROOT).as_posix(),
            "index_html": (out_dir / "index.html").relative_to(ROOT).as_posix(),
            "policy_json": policy_path.relative_to(ROOT).as_posix() if policy_path.is_relative_to(ROOT) else str(policy_path),
        },
        "condition": {
            "run_label": "scout",
            "cwd": str(ROOT),
            "command": " ".join(sys.argv),
            "host": socket.gethostname(),
            "source": "codexlab/sp80_pufferlib_probe.py",
            "source_sha256": sha256_file(ROOT / "codexlab" / "sp80_pufferlib_probe.py"),
            "adapter_source": "codexlab/sp80_puffer_policy.py",
            "adapter_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_puffer_policy.py"),
            "truth_boundary": "compatibility probe only; not control truth, solve evidence, or trained policy evidence",
        },
    }
    write_json(out_dir / "result.json", result)
    write_html(out_dir / "index.html", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
