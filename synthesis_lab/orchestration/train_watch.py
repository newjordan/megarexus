#!/usr/bin/env python3
"""Deterministic training babysitter. Runs ONE pre-registered condition.

"Train on this for 8 hours" means: this script launches the exact command
recorded in a CONDITION.json, watches it, enforces a wall-clock limit, and
writes a factual report. It never trains anything else, never edits code,
never mines data, never invents follow-up stages. Agent sessions launch it
detached and END; the unsupervised hours belong to this script, not to an
agent with a metric to chase.

Preflight refuses to launch if:
  - any code file hash in the condition's code_hashes differs on disk
  - the init checkpoint hash differs
  - any recorded data shard is missing or hash-mismatched
  - an undeclared assist env flag (MEGAREXUS_GRAPH_DEMO_PLAN / MEGAREXUS_ROUTE_*)
    is set in the environment
  - another train.py process is already running

Usage:
  python3 synthesis_lab/orchestration/train_watch.py --condition RUNS/stageX/CONDITION.json
  python3 synthesis_lab/orchestration/train_watch.py --condition ... --dry-run   # preflight only
Options: --max-hours 8.0  --heartbeat-secs 300

Output (in the condition's out_dir):
  WATCH_LOG.jsonl    heartbeat records
  WATCH_REPORT.json  exit code, duration, checkpoint hashes, log tail
Exit codes: 0 completed, 1 preflight refused, 2 training failed, 3 timeout hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from synthesis_lab.orchestration.no_imitation_guard import (  # noqa: E402
    find_banned_lineage,
    format_violations,
)

ASSIST_ENV_PREFIXES = ("MEGAREXUS_ROUTE_",)
ASSIST_ENV_EXACT = ("MEGAREXUS_GRAPH_DEMO_PLAN",)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def preflight(cond: dict) -> list[str]:
    problems: list[str] = []

    violations = find_banned_lineage(cond, context="condition")
    if violations:
        problems.append(
            "no-imitation guard refused condition:\n"
            + format_violations(violations)
        )

    for rel, want in (cond.get("code_hashes") or {}).items():
        p = REPO / rel
        if not p.exists():
            problems.append(f"code file missing: {rel}")
        elif sha256_file(p) != want:
            problems.append(f"code hash mismatch (disk differs from recorded condition): {rel}")

    init = cond.get("init_checkpoint") or {}
    if init.get("path"):
        p = REPO / init["path"]
        if not p.exists():
            problems.append(f"init checkpoint missing: {init['path']}")
        elif init.get("sha256") and sha256_file(p) != init["sha256"]:
            problems.append(f"init checkpoint hash mismatch: {init['path']}")

    for key in ("rollout_shards", "val_rollout_shards"):
        for shard in cond.get(key) or []:
            p = REPO / shard["path"]
            if not p.exists():
                problems.append(f"data shard missing: {shard['path']}")
            elif shard.get("sha256") and sha256_file(p) != shard["sha256"]:
                problems.append(f"data shard hash mismatch: {shard['path']}")

    allowed = set(cond.get("assist_env_flags_allowed") or [])
    for name in os.environ:
        hit = name in ASSIST_ENV_EXACT or any(name.startswith(pfx) for pfx in ASSIST_ENV_PREFIXES)
        if hit and name not in allowed:
            problems.append(f"undeclared assist env flag set: {name}")

    probe = subprocess.run(["pgrep", "-af", "train.py"], capture_output=True, text=True)
    lines = [ln for ln in probe.stdout.splitlines() if "train_watch.py" not in ln]
    if lines:
        problems.append(f"another train.py process is already running: {lines[0]}")

    if not isinstance(cond.get("command"), list) or not cond["command"]:
        problems.append("condition has no recorded command list")
    out = cond.get("output") or {}
    if not out.get("out_dir") or not out.get("log"):
        problems.append("condition output.out_dir / output.log not recorded")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", type=Path, required=True)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--heartbeat-secs", type=float, default=300.0)
    ap.add_argument("--dry-run", action="store_true", help="preflight only, do not launch")
    args = ap.parse_args()

    cond = json.loads(args.condition.read_text())
    problems = preflight(cond)
    if problems:
        print(f"[train_watch] PREFLIGHT REFUSED ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        print("[train_watch] Fix the condition or the tree; do not work around this.")
        return 1
    print("[train_watch] preflight clean")
    if args.dry_run:
        return 0

    out_dir = REPO / cond["output"]["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = REPO / cond["output"]["log"]
    watch_log = out_dir / "WATCH_LOG.jsonl"
    report_path = out_dir / "WATCH_REPORT.json"

    started = now()
    deadline = time.monotonic() + args.max_hours * 3600
    with open(log_path, "ab") as logf:
        proc = subprocess.Popen(
            cond["command"], cwd=REPO, stdout=logf, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"[train_watch] launched pid={proc.pid} condition={cond.get('condition_id')} "
          f"max_hours={args.max_hours}")

    timeout_hit = False
    while True:
        try:
            proc.wait(timeout=args.heartbeat_secs)
            break
        except subprocess.TimeoutExpired:
            pass
        ckpts = sorted(out_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        beat = {
            "ts": now(),
            "alive": True,
            "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
            "checkpoints": len(ckpts),
            "newest_checkpoint": ckpts[-1].name if ckpts else None,
        }
        with open(watch_log, "a") as f:
            f.write(json.dumps(beat) + "\n")
        if time.monotonic() >= deadline:
            timeout_hit = True
            print("[train_watch] max-hours reached; stopping training")
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            break

    tail = ""
    if log_path.exists():
        with open(log_path, "rb") as f:
            f.seek(max(0, log_path.stat().st_size - 4096))
            tail = f.read().decode(errors="replace")
    checkpoints = {
        p.name: sha256_file(p)
        for p in sorted(out_dir.glob("*.pt"))
    }
    report = {
        "schema": "synthesis_lab.train_watch_report.v1",
        "condition": str(args.condition),
        "condition_id": cond.get("condition_id"),
        "started": started,
        "ended": now(),
        "exit_code": proc.returncode,
        "timeout_hit": timeout_hit,
        "checkpoints_sha256": checkpoints,
        "log_tail": tail.splitlines()[-30:],
        "next": "Run the pre-registered gates in this condition's gate_after. "
                "Do not invent new conditions, data, or code from this result.",
    }
    report_path.write_text(json.dumps(report, indent=1))
    print(f"[train_watch] done exit={proc.returncode} timeout={timeout_hit} "
          f"checkpoints={len(checkpoints)} report={report_path}")
    if timeout_hit:
        return 3
    return 0 if proc.returncode == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
