#!/usr/bin/env python3
"""CONTRA Loop Harness -- GroundProbe (GROUND step).

Shells ``contra run --levels <level_set> --skip-report`` (equivalent to
``python codexlab/sp80_run_contra_levels.py --levels ... --skip-report``),
parses the resulting summary.jsonl (schema
``codexlab.sp80_contra_episode_levels.v1``), reads the top-ranked
recommendation from every packet per level, and emits ``baseline.json``
(schema ``codexlab.contra_loop.baseline.v1``) into the run_dir.

Flags regressions against the solver registry's admitted-level baseline.
A regression is a registry-baseline level whose ``levels_completed_delta``
drops to zero in the new run.

CARDINAL RULE ENFORCED HERE: solve evidence is ``levels_completed_delta > 0``
read from result.json / summary.jsonl ONLY.  The strings ``status: nominal``
(summary row) and ``status: passed`` (verifier stdout) are NOT solve evidence;
L1 in the live artifact has both while remaining unsolved (delta=0).

Usage::

    # Plan only -- no episode run, no network, no solve needed:
    python codexlab/contra_loop/baseline.py --dry-run

    # Run fresh baseline on levels 0 and 1:
    python codexlab/contra_loop/baseline.py --levels 0,1

    # Re-parse an existing run_dir (no new episode):
    python codexlab/contra_loop/baseline.py \\
        --from-run-dir codexlab/runs/contra_sp80_episode_levels_00_01_20260608T175142Z

Output::

    <run_dir>/baseline.json   (schema codexlab.contra_loop.baseline.v1)
    stdout: a single-line JSON summary (run_dir, completed_levels,
            regression_detected, baseline_json path)

Exit code::

    0  -- clean run; no regressions detected
    1  -- regression detected OR run failed to produce an artifact
    2  -- argument error
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup -- mirror contra.py so codexlab/ internals resolve correctly.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

SCHEMA = "codexlab.contra_loop.baseline.v1"
SUMMARY_SCHEMA = "codexlab.sp80_contra_episode_levels.v1"

# Maximum packets per level included in top-rec summaries; keeps baseline.json tractable.
_MAX_PACKETS_PER_LEVEL = 30


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Registry baseline reader
# ---------------------------------------------------------------------------

def load_registry_baseline() -> list[int]:
    """Return level indices already admitted in the solver registry.

    An admitted level is one where at least one registered solver carries a
    ``levels_completed_delta > 0`` provenance on that game:level target.

    Used only to detect regressions: a registry-baseline level dropping to
    delta=0 in a new run is a regression; the caller must HALT.

    Returns an empty list rather than raising if the registry is unavailable
    (e.g. first-ever run before ``solver_registry.json`` exists).
    """
    try:
        from contra_loop.solver_registry import load_registry  # type: ignore[import]
        reg = load_registry()
        admitted: set[int] = set()
        for solver in reg.get("solvers", []):
            target = solver.get("target", "")
            if ":" in target:
                _, _, lvl_str = target.partition(":")
                try:
                    admitted.add(int(lvl_str))
                except ValueError:
                    pass
        return sorted(admitted)
    except Exception:  # noqa: BLE001 -- registry unavailable is non-fatal for GROUND
        return []


# ---------------------------------------------------------------------------
# Episode runner (subprocess)
# ---------------------------------------------------------------------------

def _run_episode(
    levels: list[int],
    *,
    backend: str,
    max_actions: int,
) -> dict[str, Any]:
    """Shell ``sp80_run_contra_levels.py --levels ... --skip-report``.

    This is equivalent to ``contra run --levels ... --skip-report`` (the
    cockpit just delegates to the same script with no extra logic).

    Never trusts return-code prose.  Returns the raw stdout/stderr alongside
    the extracted run_dir so the caller can verify the artifact directly.
    """
    levels_arg = ",".join(str(lvl) for lvl in sorted(set(levels)))
    cmd = [
        sys.executable,
        "-B",
        "codexlab/sp80_run_contra_levels.py",
        "--levels",
        levels_arg,
        "--backend",
        backend,
        "--max-actions",
        str(max_actions),
        "--skip-report",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)

    # Extract run_dir from stdout using the same regex pattern as loop_driver.py.
    # The runner prints a final JSON summary containing "run_dir"; prefer the
    # run-root path (no "/level_" suffix) over individual level dirs.
    run_dir: str | None = None
    matches = re.findall(r'"run_dir":\s*"([^"]+)"', proc.stdout)
    for m in matches:
        if "/level_" not in m:
            run_dir = m
    if run_dir is None and matches:
        run_dir = matches[-1]

    return {
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "run_dir": run_dir,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-800:],
    }


# ---------------------------------------------------------------------------
# Summary artifact parsers
# ---------------------------------------------------------------------------

def _parse_summary_jsonl(run_dir_path: Path) -> list[dict[str, Any]]:
    """Read ``summary.jsonl`` and return per-level row dicts.

    Raises ``FileNotFoundError`` when the file is absent (episode runner
    crashed or the run_dir is wrong).
    """
    path = run_dir_path / "summary.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"GroundProbe: summary.jsonl not found in {run_dir_path}. "
            "The episode runner may have crashed."
        )
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _parse_summary_json(run_dir_path: Path) -> dict[str, Any]:
    """Read ``summary.json`` (provenance artifact) if present."""
    path = run_dir_path / "summary.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# packets.jsonl top-recommendation parser
# ---------------------------------------------------------------------------

def _top_rec_from_packet(packet: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the top-ranked recommendation summary from one packet dict.

    Returns ``None`` for packets that produced no recommendations (e.g.
    early packets before the pipeline warms up).

    Field locations verified against the live artifact
    ``codexlab/runs/contra_sp80_episode_levels_00_01_20260608T175142Z/``:
      - ``contra_recommendation_packet.top_primitive``      -- top action
      - ``contra_recommendation_packet.top_value_score``    -- float 0-1
      - ``contra_recommendation_packet.ranking_policy``     -- policy name
      - ``contra_recommendation_packet.ranked_recommendations[0]``
            .candidate_kind, .evidence_tier
    """
    crp = packet.get("contra_recommendation_packet") or {}
    top_prim = crp.get("top_primitive")
    if not top_prim:
        return None  # packet produced no recommendation
    top_val = crp.get("top_value_score")
    ranking_policy = crp.get("ranking_policy")
    ranked: list[dict[str, Any]] = crp.get("ranked_recommendations") or []
    first = ranked[0] if ranked else {}
    return {
        "action_index": packet.get("action_index"),
        "primitive": top_prim,
        "value_score": top_val,
        "candidate_kind": first.get("candidate_kind"),
        "evidence_tier": first.get("evidence_tier"),
        "ranking_policy": ranking_policy,
        "recommendation_count": len(ranked),
    }


def _parse_top_recs_for_level(level_dir: Path) -> list[dict[str, Any]]:
    """Read ``packets.jsonl`` for one level and return top-rec per packet.

    Capped at ``_MAX_PACKETS_PER_LEVEL`` to keep baseline.json readable.
    Missing files (episode did not reach this level) return an empty list.
    """
    packets_path = level_dir / "packets.jsonl"
    if not packets_path.is_file():
        return []
    recs: list[dict[str, Any]] = []
    for i, line in enumerate(packets_path.read_text(encoding="utf-8").splitlines()):
        if i >= _MAX_PACKETS_PER_LEVEL:
            break
        line = line.strip()
        if not line:
            continue
        try:
            pkt = json.loads(line)
            rec = _top_rec_from_packet(pkt)
            if rec is not None:
                recs.append(rec)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return recs


# ---------------------------------------------------------------------------
# Regression check
# ---------------------------------------------------------------------------

def check_regression(
    rows: list[dict[str, Any]],
    registry_baseline: list[int],
) -> tuple[bool, list[int]]:
    """Return ``(regression_detected, regressed_level_indices)``.

    A regression occurs when a registry-baseline level has
    ``levels_completed_delta <= 0`` in the current run.

    IMPORTANT: reads ONLY ``levels_completed_delta``.  The strings
    ``status: nominal`` (summary row) and ``status: passed`` (verifier output)
    are intentionally ignored -- both can appear on levels that are NOT solved
    (e.g. L1 in the 20260608 run: delta=0, episode_status=budget_exhausted,
    verifier_returncode=0, status=nominal).  Only delta>0 is solve evidence.
    """
    delta_by_level: dict[int, int] = {
        int(row["level_index"]): int(row.get("levels_completed_delta") or 0)
        for row in rows
    }
    regressed = sorted(
        lvl for lvl in registry_baseline
        if delta_by_level.get(lvl, 0) <= 0
    )
    return bool(regressed), regressed


# ---------------------------------------------------------------------------
# Core snapshot function
# ---------------------------------------------------------------------------

def snapshot(
    levels: list[int],
    *,
    backend: str = "tiny_mlp",
    max_actions: int = 20,
    out_dir: Path | None = None,
    from_run_dir: str | None = None,
) -> dict[str, Any]:
    """GROUND step: run (or reuse) an episode and build baseline.json.

    Parameters
    ----------
    levels:
        Level indices to run.  Callers should pass
        ``registry_baseline_levels ∪ {target_level}`` so the single run
        covers both regression and new-target evidence.
    backend:
        Backend flag forwarded to ``sp80_run_contra_levels.py``.
    max_actions:
        Per-level action budget.
    out_dir:
        Write ``baseline.json`` here instead of inside the run_dir.
        Defaults to the run_dir.
    from_run_dir:
        Re-parse this existing run_dir instead of launching a new episode.
        Useful for re-baselining or testing without a network connection.

    Returns
    -------
    The full baseline dict (also written to disk as ``baseline.json``).

    Callers MUST check ``result["regression_detected"]`` before proceeding.
    A ``True`` value means a registry-baseline level no longer reaches
    delta>0; the loop must HALT.

    Raises
    ------
    RuntimeError
        If a new episode run fails to produce a run_dir.
    FileNotFoundError
        If ``summary.jsonl`` is absent from the run_dir.
    """
    run_invocation: dict[str, Any] = {}
    run_dir_str: str | None = from_run_dir

    if from_run_dir is None:
        run_invocation = _run_episode(levels, backend=backend, max_actions=max_actions)
        run_dir_str = run_invocation.get("run_dir")
        if not run_dir_str:
            raise RuntimeError(
                "GroundProbe: episode runner did not produce a run_dir "
                f"(returncode={run_invocation.get('returncode')}).\n"
                f"stderr: {run_invocation.get('stderr_tail', '')}"
            )

    assert run_dir_str is not None  # guaranteed by the branch above
    run_dir_path = (
        ROOT / run_dir_str
        if not Path(run_dir_str).is_absolute()
        else Path(run_dir_str)
    )

    # Parse summary artifacts from the run_dir.
    summary_rows = _parse_summary_jsonl(run_dir_path)
    summary_full = _parse_summary_json(run_dir_path)

    # Registry baseline -- the levels every new run MUST keep passing.
    registry_baseline = load_registry_baseline()

    # Regression check: reads levels_completed_delta only; ignores status strings.
    regression_detected, regressed_levels = check_regression(summary_rows, registry_baseline)

    # Build per-level detail: summary row fields + top-rec list from packets.jsonl.
    per_level: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        lvl = int(row.get("level_index", -1))
        level_dir = run_dir_path / f"level_{lvl:02d}"
        top_recs = _parse_top_recs_for_level(level_dir)
        delta = int(row.get("levels_completed_delta") or 0)

        # Aggregate the top-rec list into a fast-scan summary.
        primitives_seen = list(dict.fromkeys(r["primitive"] for r in top_recs if r.get("primitive")))
        kinds_seen = list(dict.fromkeys(r["candidate_kind"] for r in top_recs if r.get("candidate_kind")))
        tier_vals = [r["evidence_tier"] for r in top_recs if r.get("evidence_tier") is not None]

        per_level[str(lvl)] = {
            "level_index": lvl,
            # THE solve signal; do not substitute status strings for this.
            "levels_completed_delta": delta,
            "episode_status": row.get("episode_status"),
            "verifier_returncode": row.get("verifier_returncode"),
            # "nominal" = harness ran without error; does NOT mean solved.
            # Solved IFF levels_completed_delta > 0.
            "row_status": row.get("status"),
            "action_count": int(row.get("action_count") or 0),
            "is_baseline_level": lvl in registry_baseline,
            "regression": lvl in regressed_levels,
            # Top-rec per packet (up to _MAX_PACKETS_PER_LEVEL entries).
            "top_recs_per_packet": top_recs,
            # One-line scan of what the pipeline recommended this episode.
            "top_recs_summary": {
                "packet_count_with_recommendation": len(top_recs),
                "primitives_seen": primitives_seen,
                "candidate_kinds_seen": kinds_seen,
                "min_evidence_tier_seen": min(tier_vals, default=None),
            },
        }

    # Levels that actually completed in this run (delta>0 rows).
    completed_levels = sorted(
        int(row["level_index"])
        for row in summary_rows
        if int(row.get("levels_completed_delta") or 0) > 0
    )

    # Source SHA provenance forwarded from summary.json
    # (written by sp80_run_contra_levels.py lines 167-171).
    provenance: dict[str, Any] = {
        "summary_schema": summary_full.get("schema", SUMMARY_SCHEMA),
        "runner_source_sha256": summary_full.get("runner_source_sha256"),
        "harness_source_sha256": summary_full.get("harness_source_sha256"),
        "report_source_sha256": summary_full.get("report_source_sha256"),
    }

    baseline: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at_utc": _utc_now(),
        "run_dir": run_dir_str,
        "game_id": summary_full.get("game_id", "sp80-589a99af"),
        "backend": summary_full.get("backend", backend),
        "levels_requested": sorted(set(levels)),
        # Completed = delta>0.  NOT derived from any status string.
        "completed_levels": completed_levels,
        "levels_completed_delta_total": sum(
            int(r.get("levels_completed_delta") or 0) for r in summary_rows
        ),
        "registry_baseline_levels": registry_baseline,
        # True means a regression was detected; callers must HALT.
        "regression_detected": regression_detected,
        "regressed_levels": regressed_levels,
        # Verbatim boundary -- must not be parsed as solve evidence.
        "boundary": (
            "GroundProbe output only.  "
            "Not solve evidence for any level unless that level's per_level entry "
            "records levels_completed_delta > 0.  "
            "status:nominal (summary row) and verifier status:passed are NOT "
            "solve evidence; both appear on unsolved levels."
        ),
        "provenance": provenance,
        "per_level": per_level,
        # Raw invocation record for audit; empty when from_run_dir was used.
        "run_invocation": run_invocation if run_invocation else {
            "from_existing_run_dir": from_run_dir
        },
    }

    # Write baseline.json to disk.
    write_dir = out_dir if out_dir else run_dir_path
    write_dir.mkdir(parents=True, exist_ok=True)
    out_path = write_dir / "baseline.json"
    out_path.write_text(
        json.dumps(baseline, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        baseline_json_display = str(out_path.relative_to(ROOT))
    except ValueError:
        baseline_json_display = str(out_path)

    # Single-line stdout summary for the loop orchestrator to parse.
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "run_dir": run_dir_str,
                "completed_levels": completed_levels,
                "registry_baseline_levels": registry_baseline,
                "regression_detected": regression_detected,
                "regressed_levels": regressed_levels,
                "baseline_json": baseline_json_display,
            },
            indent=2,
            sort_keys=True,
        )
    )

    return baseline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--levels",
        default="0,1",
        help=(
            "comma-separated level indices to run "
            "(should be registry_baseline_levels ∪ target); "
            "default: 0,1"
        ),
    )
    p.add_argument(
        "--backend",
        default="tiny_mlp",
        choices=["tiny_mlp", "lora_adapter", "qwen_http"],
    )
    p.add_argument(
        "--max-actions",
        type=int,
        default=20,
        help="per-level action budget passed to sp80_run_contra_levels.py",
    )
    p.add_argument(
        "--from-run-dir",
        default=None,
        metavar="RUN_DIR",
        help=(
            "parse this existing run_dir instead of running a new episode; "
            "useful for re-baselining or --dry-run inspection of a past run"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="write baseline.json here; defaults to inside the run_dir",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the snapshot plan; execute nothing; exit 0",
    )
    return p


def main() -> int:  # noqa: C901
    args = _build_parser().parse_args()

    levels_str = [l.strip() for l in args.levels.split(",") if l.strip()]
    if not levels_str:
        print("error: --levels must contain at least one level index", file=sys.stderr)
        return 2
    try:
        levels = [int(l) for l in levels_str]
    except ValueError:
        print("error: --levels must be comma-separated integers", file=sys.stderr)
        return 2
    if any(l < 0 for l in levels):
        print("error: level indices must be >= 0", file=sys.stderr)
        return 2

    registry_baseline = load_registry_baseline()

    if args.dry_run:
        levels_arg = ",".join(str(l) for l in sorted(set(levels)))
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "levels_to_run": sorted(set(levels)),
                    "registry_baseline_levels": registry_baseline,
                    "planned_cmd": (
                        f"python codexlab/sp80_run_contra_levels.py "
                        f"--levels {levels_arg} "
                        f"--backend {args.backend} "
                        f"--max-actions {args.max_actions} "
                        f"--skip-report"
                    ),
                    "output": "baseline.json written into run_dir",
                    "solve_signal": "levels_completed_delta > 0 ONLY",
                    "not_solve_signal": [
                        "status:nominal (summary row)",
                        "verifier status:passed",
                        "verifier_returncode:0 alone",
                    ],
                    "regression_trigger": (
                        "any registry_baseline level with delta<=0 in the new run"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    try:
        result = snapshot(
            levels,
            backend=args.backend,
            max_actions=args.max_actions,
            out_dir=args.out_dir,
            from_run_dir=args.from_run_dir,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Exit 1 on regression (loop must HALT); exit 0 otherwise.
    return 1 if result.get("regression_detected") else 0


if __name__ == "__main__":
    raise SystemExit(main())
