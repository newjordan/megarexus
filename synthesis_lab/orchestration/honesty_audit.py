#!/usr/bin/env python3
"""Honesty audit: classify every run's evidence by assistance markers.

This is a heuristic sweep over run artifacts (CONDITION.json, gate JSONs,
result.json). It flags known assist markers so that assisted runs can never
be reported as autonomous model progress. Passing this audit is NOT proof of
autonomy; failing it is proof the run may only be cited as assisted/diagnostic.

Classes:
  AUTONOMOUS_CANDIDATE  no known assist markers found
  IMITATION_ASSISTED    route/demo-plan runtime assistance or human-mined /
                        replay / route data in the checkpoint or dataset path
  INVALID_HEADLINE      gate_verdict PASS where the gate's own pass checks
                        require assistance to be attached (e.g.
                        route_provider_attached) - the gate was shaped around
                        the assist and its PASS must never be a headline claim

Usage:
  python3 synthesis_lab/orchestration/honesty_audit.py            # sweep all runs
  python3 synthesis_lab/orchestration/honesty_audit.py --run DIR  # one run dir
  python3 synthesis_lab/orchestration/honesty_audit.py --json OUT # machine report
Exit codes: 0 clean active sweep, 2 if any active INVALID_HEADLINE exists, 1 on errors.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "synthesis_lab" / "experiments" / "megarexus_cnn" / "runs"

# Runtime assistance markers: any of these appearing in a run's recorded
# condition/command/env or gate JSON means the runtime was not the CNN
# deciding from the current frame alone.
ASSIST_ENV_MARKERS = (
    "MEGAREXUS_GRAPH_DEMO_PLAN",
    "MEGAREXUS_ROUTE_PROGRESS_POLICY",
    "MEGAREXUS_ROUTE_LEVEL_POLICY",
    "MEGAREXUS_ROUTE_LEVEL5_POLICY",
    "MEGAREXUS_ROUTE_",
)

# Data provenance taint: shards or checkpoints whose lineage includes human
# answers or mined/replayed routes. Matching path segments only, to avoid
# false hits on prose.
TAINT_PATH_PAT = re.compile(r"human[_-]?mined|human[_-]?episode|runtime[_-]?replay|route", re.I)

# Gate pass-checks that structurally require assistance to be attached.
ASSIST_REQUIRED_CHECKS = ("route_provider_attached",)

MAX_JSON_BYTES = 8 * 1024 * 1024


def iter_json_files(run_dir: Path):
    for p in sorted(run_dir.glob("*.json")):
        try:
            if p.stat().st_size > MAX_JSON_BYTES:
                continue
            yield p, p.read_text(errors="replace")
        except OSError:
            continue


def is_quarantined(run_dir: Path) -> bool:
    try:
        return "_invalid" in run_dir.relative_to(RUNS).parts
    except ValueError:
        return False


def iter_run_dirs() -> list[Path]:
    dirs: list[Path] = []
    for p in sorted(RUNS.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "_invalid":
            dirs.extend(sorted(q for q in p.iterdir() if q.is_dir()))
        else:
            dirs.append(p)
    return dirs


def audit_run(run_dir: Path) -> dict:
    markers: list[str] = []
    taints: list[str] = []
    invalid_headline = False
    disclosed = any(run_dir.glob("*INVALIDATION*")) or any(run_dir.glob("*BOUNDARY*"))
    gate_verdicts: list[str] = []

    for path, text in iter_json_files(run_dir):
        for m in ASSIST_ENV_MARKERS:
            if m in text and f"env:{m}" not in markers:
                markers.append(f"env:{m}")
        try:
            data = json.loads(text)
        except (ValueError, RecursionError):
            data = None
        if isinstance(data, dict):
            verdict = data.get("gate_verdict")
            if verdict:
                gate_verdicts.append(str(verdict))
            # structural runtime-assist fields inside gate detail rows
            det = data.get("detail")
            rows = det.get("model", []) if isinstance(det, dict) else []
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                for key in ("route_provider", "route_action_policy"):
                    sub = row.get(key)
                    if isinstance(sub, dict) and sub.get("enabled") and f"gate:{key}.enabled" not in markers:
                        markers.append(f"gate:{key}.enabled")
                checks = row.get("checks")
                if isinstance(checks, dict):
                    for c in ASSIST_REQUIRED_CHECKS:
                        if c in checks:
                            invalid_headline = invalid_headline or str(verdict) == "PASS"
                            if f"gate-check-requires-assist:{c}" not in markers:
                                markers.append(f"gate-check-requires-assist:{c}")
            # taint in recorded checkpoint / data / command paths
            for key in ("checkpoint", "command", "init_checkpoint"):
                val = data.get(key)
                if isinstance(val, str):
                    hit = TAINT_PATH_PAT.search(val)
                    if hit and f"path:{hit.group(0).lower()}" not in taints:
                        taints.append(f"path:{hit.group(0).lower()}")
        # taint in any recorded dataset path strings
        for hit in set(TAINT_PATH_PAT.findall(text)) if "data/" in text else ():
            tag = f"data:{hit.lower()}"
            if tag not in taints:
                taints.append(tag)

    if invalid_headline:
        cls = "INVALID_HEADLINE"
    elif markers or taints:
        cls = "IMITATION_ASSISTED"
    else:
        cls = "AUTONOMOUS_CANDIDATE"
    return {
        "run": str(run_dir.relative_to(RUNS)),
        "class": cls,
        "quarantined": is_quarantined(run_dir),
        "gate_verdicts": gate_verdicts,
        "assist_markers": markers,
        "data_taint": taints,
        "disclosure_file_present": disclosed,
    }


def integrity_check() -> list[str]:
    """Tracked artifacts under runs/ modified in the working tree."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", str(RUNS.relative_to(REPO))],
            cwd=REPO, capture_output=True, text=True, check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError) as exc:
        return [f"git status failed: {exc}"]
    return [line for line in out.splitlines() if line.startswith(" M") or line.startswith("M ")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, help="audit a single run directory")
    ap.add_argument("--json", type=Path, help="write machine-readable report")
    args = ap.parse_args()

    run_dirs = [args.run] if args.run else iter_run_dirs()
    reports = [audit_run(d) for d in run_dirs]
    modified = integrity_check()

    counts: dict[str, int] = {}
    active_counts: dict[str, int] = {}
    quarantine_counts: dict[str, int] = {}
    for r in reports:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
        bucket = quarantine_counts if r["quarantined"] else active_counts
        bucket[r["class"]] = bucket.get(r["class"], 0) + 1

    for r in reports:
        if r["class"] == "AUTONOMOUS_CANDIDATE":
            continue
        flags = []
        if r["quarantined"]:
            flags.append("quarantined")
        if r["disclosure_file_present"]:
            flags.append("disclosed")
        flag = f" [{' '.join(flags)}]" if flags else ""
        print(f"{r['class']:<20} {r['run']}{flag}")
        for m in r["assist_markers"]:
            print(f"    {m}")
        for t in r["data_taint"]:
            print(f"    taint {t}")
    print()
    print("summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("active_summary:", ", ".join(f"{k}={v}" for k, v in sorted(active_counts.items())))
    if quarantine_counts:
        print("quarantined_summary:", ", ".join(f"{k}={v}" for k, v in sorted(quarantine_counts.items())))
    if modified:
        print("integrity: committed run artifacts modified in working tree:")
        for line in modified:
            print(f"    {line}")

    if args.json:
        args.json.write_text(json.dumps(
            {
                "reports": reports,
                "summary": counts,
                "active_summary": active_counts,
                "quarantined_summary": quarantine_counts,
                "modified_tracked_artifacts": modified,
            },
            indent=1,
        ))

    return 2 if any(
        r["class"] == "INVALID_HEADLINE" and not r["quarantined"]
        for r in reports
    ) else 0


if __name__ == "__main__":
    sys.exit(main())
