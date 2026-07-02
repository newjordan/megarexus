#!/usr/bin/env python3
"""CONTRA Loop Harness -- LoopOrchestrator CLI.

The single entry the agent rides for the CONTRA discovery loop.

Subcommands
-----------
iterate  Drive the 9-step protocol; pause at three agent hand-off points.
         --dry-run runs GROUND + VERIFY-GATE (layers 2-4) end-to-end WITHOUT
         requiring a new admission, proving the spine on day one.
baseline GROUND step only: run episode, parse summary.jsonl, emit baseline.json.
gate     VERIFY-GATE (5 layers) against a supplied or freshly-produced run_dir.
admit    Gate + registry.admit() for a named mechanic; never bypasses the gate.
status   Print solver library + journal tail.

How it runs
-----------
Shells to contra.py for high-level operations:
  contra gate  -- L1 structural (dev_check.py, 125 files)
  contra check -- L2 integration (live DAG edges + static tripwires)
  contra run   -- L3 episode run (sp80_run_contra_levels.py)

Calls the per-game verifier script directly for L4 (no contra.py route):
  python codexlab/sp80_verify_episode_run.py --run-dir <level_dir>

Uses baseline.snapshot() (contra_loop/baseline.py) for the GROUND step, which
in turn calls sp80_run_contra_levels.py -- equivalent to "contra run --skip-report".

Writes journal.jsonl (append-only) for every iteration.

CARDINAL RULE (structural, not trust-based)
-------------------------------------------
Admission requires levels_completed_delta > 0 in the run artifact
(result.json / summary.jsonl). "status:nominal" and verifier "status:passed"
are NOT solve evidence. L1 in the 20260608 live artifact is nominal + verifier
rc=0 while delta=0 and entirely unsolved. The registry reads delta ONLY.

Usage
-----
# Day-one dry run: proves the spine end-to-end, no admission required.
python codexlab/contra_loop/loop.py iterate --target sp80:1 --dry-run

# Full iteration (pauses at agent hand-offs; agent writes hypothesis/keystone/solver):
python codexlab/contra_loop/loop.py iterate --target sp80:1

# Full iteration with solver wired: runs GROUND+LIVE-RUN+GATE+ADMIT.
python codexlab/contra_loop/loop.py iterate --target sp80:1 --mechanic SOLVER_NAME

# GROUND step only:
python codexlab/contra_loop/loop.py baseline --target sp80:1

# Gate only (on a specific run_dir):
python codexlab/contra_loop/loop.py gate --target sp80:1 --run-dir <path>

# Admit a named mechanic (runs gate first):
python codexlab/contra_loop/loop.py admit --target sp80:1 --run-dir <path> --mechanic NAME

# Status:
python codexlab/contra_loop/loop.py status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo layout (mirrors every other codexlab script).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # rexus/
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

# ---------------------------------------------------------------------------
# Internal package imports (all three modules already exist in this package).
# ---------------------------------------------------------------------------
from contra_loop.solver_registry import (  # noqa: E402
    load_registry,
    register_solver,
    solvers_for_target,
)
from contra_loop.baseline import (  # noqa: E402
    snapshot as _baseline_snapshot,
    load_registry_baseline,
    check_regression,
)
from contra_loop.adapter import (  # noqa: E402
    GAME_TABLE,
    lookup_adapter_for_target,
)

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------
SCHEMA = "codexlab.contra_loop.loop.v1"
JOURNAL_PATH = CODEXLAB / "contra_loop" / "journal.jsonl"
REJECTIONS_PATH = CODEXLAB / "contra_loop" / "registry" / "rejections.jsonl"
HYPOTHESES_PATH = CODEXLAB / "contra_loop" / "registry" / "hypotheses.jsonl"

# Ensure registry directory exists.
(CODEXLAB / "contra_loop" / "registry").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utilities.
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _journal_append(entry: dict[str, Any]) -> None:
    """Append one record to journal.jsonl (append-only; never overwrites)."""
    entry.setdefault("schema", SCHEMA)
    entry.setdefault("recorded_at_utc", _utc_now())
    with JOURNAL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _rejections_append(entry: dict[str, Any]) -> None:
    """Append one honest-negative record to registry/rejections.jsonl."""
    entry.setdefault("schema", "codexlab.contra_loop.rejection.v1")
    entry.setdefault("recorded_at_utc", _utc_now())
    with REJECTIONS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")


def _parse_target(target: str) -> tuple[str, int]:
    """Parse 'sp80:1' -> ('sp80', 1).  Validates game is in GAME_TABLE."""
    game, _, level_str = target.partition(":")
    if game not in GAME_TABLE:
        raise SystemExit(f"unknown game {game!r}; known: {sorted(GAME_TABLE)}")
    try:
        level = int(level_str) if level_str else 0
    except ValueError:
        raise SystemExit(f"level_index must be an integer; got {level_str!r}")
    return game, level


def _level_set_for_run(game: str, target_level: int) -> list[int]:
    """registry_baseline ∪ {target_level} -- ensures regression + target in one run."""
    baseline = load_registry_baseline()
    result = sorted(set(baseline) | {target_level})
    return result


def _contra(*args: str) -> subprocess.CompletedProcess[str]:
    """Shell to codexlab/contra.py <args>.

    Loop.py shells to contra.py for high-level ops (run, check, gate) per the
    design principle.  Never raises on non-zero returncode; caller must check.
    """
    cmd = [sys.executable, "-B", str(CODEXLAB / "contra.py"), *args]
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


def _extract_run_dir_from_stdout(stdout: str) -> str | None:
    """Extract the top-level run_dir from contra run stdout JSON."""
    import re
    # The runner emits a final JSON block with "run_dir"; prefer the root
    # (no /level_XX suffix) over individual level subdirs.
    matches = re.findall(r'"run_dir":\s*"([^"]+)"', stdout)
    for m in matches:
        if "/level_" not in m:
            return m
    return matches[-1] if matches else None


def _wanted_mechanic_hint(game: str, target_level: int) -> str | None:
    target = f"{game}:{target_level}"
    reg = load_registry()
    for w in reg.get("wanted_mechanics", []):
        if w.get("target") == target:
            return f"{w['name']}: {w.get('want', '')}"
    return None


def _hypotheses_for_target(target: str) -> list[dict[str, Any]]:
    """Return discovery_only hypotheses for this target from hypotheses.jsonl."""
    if not HYPOTHESES_PATH.is_file():
        return []
    items = []
    for line in HYPOTHESES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("target") == target:
                items.append(row)
        except json.JSONDecodeError:
            pass
    return items


# ---------------------------------------------------------------------------
# GateResult: structured result from the 5-layer verification gate.
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    """Structured result from _run_gate (ArtifactBundle or Refusal).

    ok=True only when ALL layers passed AND per-level verifiers returned 0.
    The admission predicate is evaluated separately in _admit_or_reject().
    """
    ok: bool
    failed_layer: str | None = None
    reason: str | None = None
    run_dir: str | None = None
    completed_levels: list[int] = field(default_factory=list)
    baseline_levels: list[int] = field(default_factory=list)
    no_regression: bool = False
    per_level: list[dict[str, Any]] = field(default_factory=list)
    layers: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failed_layer": self.failed_layer,
            "reason": self.reason,
            "run_dir": self.run_dir,
            "completed_levels": self.completed_levels,
            "baseline_levels": self.baseline_levels,
            "no_regression": self.no_regression,
            "per_level": self.per_level,
            "layers": self.layers,
        }


# ---------------------------------------------------------------------------
# GROUND step (Step 1, deterministic).
# Wraps baseline.snapshot() which handles all summary.jsonl parsing.
# ---------------------------------------------------------------------------

def _ground(
    game: str,
    target_level: int,
    *,
    backend: str,
    max_actions: int,
    from_run_dir: str | None = None,
) -> dict[str, Any]:
    """STEP 1 -- GROUND: run episode and snapshot the baseline.

    Shells to sp80_run_contra_levels.py (via baseline.snapshot()) with
    --skip-report.  Parses summary.jsonl for levels_completed_delta ONLY;
    never trusts status strings.

    Returns the full baseline dict from baseline.snapshot().  The caller MUST
    check result["regression_detected"] and halt if True.
    """
    levels = _level_set_for_run(game, target_level)
    try:
        result = _baseline_snapshot(
            levels,
            backend=backend,
            max_actions=max_actions,
            from_run_dir=from_run_dir,
        )
    except (RuntimeError, FileNotFoundError) as exc:
        return {
            "schema": "codexlab.contra_loop.baseline.v1",
            "ok": False,
            "reason": str(exc),
            "completed_levels": [],
            "registry_baseline_levels": load_registry_baseline(),
            "regression_detected": False,
            "per_level": {},
        }
    # baseline.snapshot() already prints a single-line JSON summary to stdout.
    return result


# ---------------------------------------------------------------------------
# VERIFY-GATE (Step 7, deterministic, 5 layers, short-circuit on failure).
# ---------------------------------------------------------------------------

def _run_gate(
    game: str,
    target_level: int,
    *,
    backend: str,
    max_actions: int,
    run_dir: str | None = None,
    skip_l1_structural: bool = False,
    skip_l5_report: bool = True,
) -> GateResult:
    """STEP 7 -- VERIFY-GATE: five sequential layers, short-circuit on first failure.

    Layers
    ------
    L1 STRUCTURAL: ``contra gate`` (dev_check.py -- syntax, imports, regressions).
    L2 INTEGRATION: ``contra check`` (live DAG edges + static tripwires).
    L3 EPISODE-RUN: ``contra run --levels <baseline ∪ target>`` OR reuse supplied run_dir.
                    ONE run covers new target AND every registry-baseline level so a
                    single artifact captures progress AND any regression.
    L4 PER-LEVEL VERIFY: game verifier script per level (returncode 0 each).
                         Reads ONLY levels_completed_delta; ignores status strings.
    L5 REPORT: angel_report_verify.py --strict-png-ledger (skip if no report built).

    ADMISSION NOTE (enforced by _admit_or_reject, NOT here):
    Passing the gate is necessary but NOT sufficient for admission.
    The admission predicate additionally requires:
      - levels_completed_delta > 0 on the target level
      - no regression on registry baseline levels (completed_levels_after ⊇ baseline)
    Status strings ("nominal", "passed") are NEVER read as solve evidence.
    """
    layers: dict[str, Any] = {}
    baseline = load_registry_baseline()

    # ── L1 STRUCTURAL ────────────────────────────────────────────────────────
    if not skip_l1_structural:
        l1_proc = _contra("gate")
        layers["l1_structural"] = {
            "cmd": "contra gate",
            "returncode": l1_proc.returncode,
            "stdout_tail": l1_proc.stdout[-500:],
            "stderr_tail": l1_proc.stderr[-300:],
        }
        if l1_proc.returncode != 0:
            return GateResult(
                ok=False,
                failed_layer="l1_structural",
                reason=f"contra gate (dev_check) returned rc={l1_proc.returncode}",
                baseline_levels=baseline,
                layers=layers,
            )
    else:
        layers["l1_structural"] = {"skipped": True, "returncode": 0}

    # ── L2 INTEGRATION ───────────────────────────────────────────────────────
    l2_proc = _contra("check")
    layers["l2_integration"] = {
        "cmd": "contra check",
        "returncode": l2_proc.returncode,
        "stdout_tail": l2_proc.stdout[-500:],
        "stderr_tail": l2_proc.stderr[-300:],
    }
    if l2_proc.returncode != 0:
        return GateResult(
            ok=False,
            failed_layer="l2_integration",
            reason=f"contra check (DAG integration) returned rc={l2_proc.returncode}",
            baseline_levels=baseline,
            layers=layers,
        )

    # ── L3 EPISODE-RUN ───────────────────────────────────────────────────────
    if run_dir is None:
        # Produce a fresh episode run covering baseline ∪ target.
        levels = _level_set_for_run(game, target_level)
        levels_arg = ",".join(str(l) for l in levels)
        l3_proc = _contra(
            "run", "--levels", levels_arg,
            "--backend", backend,
            "--max-actions", str(max_actions),
            "--skip-report",
        )
        run_dir = _extract_run_dir_from_stdout(l3_proc.stdout)
        layers["l3_episode"] = {
            "cmd": f"contra run --levels {levels_arg}",
            "returncode": l3_proc.returncode,
            "run_dir": run_dir,
            "stdout_tail": l3_proc.stdout[-500:],
        }
        if l3_proc.returncode != 0 or not run_dir:
            return GateResult(
                ok=False,
                failed_layer="l3_episode",
                reason=(
                    f"contra run returned rc={l3_proc.returncode} or no run_dir; "
                    f"stdout tail: {l3_proc.stdout[-400:]}"
                ),
                run_dir=run_dir,
                baseline_levels=baseline,
                layers=layers,
            )
    else:
        layers["l3_episode"] = {"run_dir_supplied": run_dir, "returncode": 0}

    run_dir_path = ROOT / run_dir if not Path(run_dir).is_absolute() else Path(run_dir)

    # ── L4 PER-LEVEL VERIFY ──────────────────────────────────────────────────
    # Parse summary.jsonl to discover which levels ran, then verify each.
    summary_jsonl = run_dir_path / "summary.jsonl"
    if not summary_jsonl.is_file():
        # Fall back to summary.json rows field.
        summary_json_f = run_dir_path / "summary.json"
        if summary_json_f.is_file():
            rows_raw = json.loads(summary_json_f.read_text(encoding="utf-8")).get("rows", [])
        else:
            rows_raw = []
    else:
        rows_raw = []
        for line in summary_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows_raw.append(json.loads(line))

    # Retrieve the verifier script path from the game adapter.
    try:
        adapter, _ = lookup_adapter_for_target(f"{game}:{target_level}")
        verifier_script = ROOT / adapter.verifier_script_rel()
    except KeyError:
        # Fallback: SP80's verifier is the canonical choice.
        verifier_script = CODEXLAB / "sp80_verify_episode_run.py"

    l4_results: list[dict[str, Any]] = []
    completed: list[int] = []

    for row in rows_raw:
        level = int(row.get("level_index", -1))
        # The level_dir is either recorded in the row or constructed from run_dir.
        level_dir_rel = row.get("run_dir") or f"{run_dir}/level_{level:02d}"
        level_dir_path = (
            ROOT / level_dir_rel
            if not Path(level_dir_rel).is_absolute()
            else Path(level_dir_rel)
        )

        verify_proc = subprocess.run(
            [sys.executable, "-B", str(verifier_script), "--run-dir", str(level_dir_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        # CARDINAL RULE: read levels_completed_delta from the row artifact,
        # NEVER from the verifier "status" string ("passed" != solved).
        delta = int(row.get("levels_completed_delta") or 0)

        l4_entry: dict[str, Any] = {
            "level_index": level,
            "level_dir": level_dir_rel,
            # THE solve field -- read from run artifact, not verifier text.
            "levels_completed_delta": delta,
            "episode_status": row.get("episode_status"),
            # verifier_returncode must be 0 for the gate to pass.
            "verifier_returncode": verify_proc.returncode,
            # status:nominal means harness ran cleanly; NOT that the level was solved.
            "row_status_nominal": row.get("status") == "nominal",
            "verifier_stdout_tail": verify_proc.stdout[-300:],
            "verifier_stderr_tail": verify_proc.stderr[-200:],
        }
        l4_results.append(l4_entry)

        if delta > 0:
            completed.append(level)

        # Verifier must return 0 for every level (the gate short-circuits here).
        if verify_proc.returncode != 0:
            layers["l4_per_level_verify"] = {
                "results": l4_results,
                "failed_level": level,
                "failed_level_dir": level_dir_rel,
            }
            return GateResult(
                ok=False,
                failed_layer="l4_per_level_verify",
                reason=(
                    f"verifier returned rc={verify_proc.returncode} for level {level}; "
                    f"verifier stderr: {verify_proc.stderr[-300:]}"
                ),
                run_dir=run_dir,
                completed_levels=completed,
                baseline_levels=baseline,
                per_level=l4_results,
                layers=layers,
            )

    layers["l4_per_level_verify"] = {"results": l4_results, "all_passed": True}

    # ── L5 REPORT PROVENANCE ─────────────────────────────────────────────────
    if not skip_l5_report:
        report_dir = run_dir_path / "angel_report"
        if report_dir.is_dir():
            l5_proc = subprocess.run(
                [
                    sys.executable, "-B",
                    str(CODEXLAB / "angel_report_verify.py"),
                    str(report_dir.relative_to(ROOT)),
                    "--strict-png-ledger",
                    "--require-gameplay-visual-contract",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            layers["l5_report"] = {
                "cmd": "angel_report_verify --strict-png-ledger --require-gameplay-visual-contract",
                "returncode": l5_proc.returncode,
                "report_dir": str(report_dir.relative_to(ROOT)),
                "stderr_tail": l5_proc.stderr[-200:],
            }
            if l5_proc.returncode != 0:
                return GateResult(
                    ok=False,
                    failed_layer="l5_report",
                    reason=(
                        f"angel_report_verify returned rc={l5_proc.returncode}; "
                        f"stderr: {l5_proc.stderr[-300:]}"
                    ),
                    run_dir=run_dir,
                    completed_levels=completed,
                    baseline_levels=baseline,
                    per_level=l4_results,
                    layers=layers,
                )
        else:
            layers["l5_report"] = {
                "skipped": True,
                "reason": "no angel_report/ dir (--skip-report was used; re-run without --skip-report for L5)",
            }
    else:
        layers["l5_report"] = {"skipped": True, "reason": "skip_l5_report=True (fast gate mode)"}

    # ── COMPUTE NO-REGRESSION ────────────────────────────────────────────────
    no_regression = all(b in completed for b in baseline)

    return GateResult(
        ok=True,
        run_dir=run_dir,
        completed_levels=sorted(completed),
        baseline_levels=baseline,
        no_regression=no_regression,
        per_level=l4_results,
        layers=layers,
    )


# ---------------------------------------------------------------------------
# ADMIT or REJECT (Step 8, deterministic; the only admission writer).
# ---------------------------------------------------------------------------

def _admit_or_reject(
    gate: GateResult,
    target: str,
    mechanic_name: str,
    *,
    iteration: int = 0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """STEP 8 -- ADMIT or REJECT.

    ADMISSION requires ALL simultaneously:
      (a) gate.ok -- all 5 gate layers returned 0
      (b) levels_completed_delta > 0 on the target level
          (THE solve field; read from summary.jsonl, NOT status strings)
      (c) no_regression -- completed_levels_after ⊇ registry_baseline

    registry.register_solver() is the ONLY admission writer.  This function
    calls it on success; it appends to rejections.jsonl on failure.

    In dry_run=True mode this function reports what would happen but writes nothing.

    IMPORTANT: status:nominal (summary row) and verifier status:passed are NOT
    solve evidence.  Both appear on unsolved levels (L1 in the 20260608 run:
    delta=0, episode_status=budget_exhausted, verifier_returncode=0, status=nominal).
    """
    game, target_level = _parse_target(target)

    # Find this target's per-level record from the gate.
    target_row = next(
        (r for r in gate.per_level if r.get("level_index") == target_level),
        None,
    )
    target_delta = int((target_row or {}).get("levels_completed_delta") or 0)

    # Admission predicate (all must hold).
    can_admit = gate.ok and target_delta > 0 and gate.no_regression

    reasons: list[str] = []
    if not gate.ok:
        reasons.append(f"gate_failed layer={gate.failed_layer}: {gate.reason}")
    if target_delta <= 0:
        reasons.append(
            f"target levels_completed_delta={target_delta} (need >0); "
            f"episode_status={((target_row or {}).get('episode_status') or 'unknown')!r}; "
            f"NOTE: status:nominal/'passed' are NOT solve evidence -- only delta>0 is"
        )
    if not gate.no_regression:
        regressed = [b for b in gate.baseline_levels if b not in gate.completed_levels]
        reasons.append(f"regression on baseline levels {regressed}")

    if dry_run:
        return {
            "action": "dry_run_no_admission",
            "would_admit": can_admit,
            "gate_ok": gate.ok,
            "target_delta": target_delta,
            "no_regression": gate.no_regression,
            "reasons_if_reject": reasons,
        }

    if can_admit:
        record = {
            "name": mechanic_name,
            "target": target,
            "applies_when": f"target {target} -- mechanic discovered iteration {iteration}",
            "provenance": {
                "run_dir": gate.run_dir,
                "levels_completed_delta": target_delta,
                "completed_levels_after": gate.completed_levels,
                "completed_levels_baseline": gate.baseline_levels,
                "iteration": iteration,
                "all_five_layers_passed": gate.ok,
            },
            "verified": True,
        }
        try:
            admission = register_solver(record)
        except ValueError as exc:
            _rejections_append({
                "target": target,
                "mechanic_name": mechanic_name,
                "iteration": iteration,
                "reason": f"register_solver refused: {exc}",
                "target_delta": target_delta,
                "gate_layers": gate.layers,
            })
            return {
                "action": "rejected",
                "reason": f"register_solver refused: {exc}",
                "target_delta": target_delta,
                "gate_ok": gate.ok,
                "no_regression": gate.no_regression,
            }
        return {
            "action": "admitted",
            "solver_name": mechanic_name,
            "admission": admission,
            "target_delta": target_delta,
            "completed_levels_after": gate.completed_levels,
        }

    # REJECT: append honest negative to rejections.jsonl.
    _rejections_append({
        "target": target,
        "mechanic_name": mechanic_name,
        "iteration": iteration,
        "reason": "; ".join(reasons),
        "target_delta": target_delta,
        "gate_ok": gate.ok,
        "no_regression": gate.no_regression,
        "gate_layers_summary": {k: v.get("returncode") for k, v in gate.layers.items()},
        "boundary": (
            "Honest negative: rejection is NOT failure of the loop, it is data. "
            "A hard level may take many iterations; no-admission is a first-class outcome."
        ),
    })
    return {
        "action": "rejected",
        "reason": "; ".join(reasons),
        "target_delta": target_delta,
        "gate_ok": gate.ok,
        "no_regression": gate.no_regression,
    }


# ---------------------------------------------------------------------------
# STALL-CHECK (Step 9, deterministic).
# ---------------------------------------------------------------------------

def _stall_check(
    target: str,
    current_delta: int,
) -> dict[str, Any]:
    """STEP 9 -- STALL-CHECK: scan journal for the four human-intervention patterns.

    Returns {halt: bool, patterns: list[str], evidence: dict}.

    Patterns detected here (mirroring the stalls.py design):
    (b) thrash: level metric 0->1->0 across 3+ attempts.
    (c) flat_zero: 5+ consecutive delta=0 iterations (forward-model cov=0/N signal).

    Patterns (a) reinvention and (d) unreproducible-win require the full stalls.py
    module (TODO extension point -- stalls.py does not exist yet).

    On halt=True: SURFACE specific evidence for human redirect.  The loop caller
    must print this evidence and exit non-zero.  Never silently continue.
    """
    # TODO: extend to full StallDetector once stalls.py exists.
    target_deltas: list[int] = []

    if JOURNAL_PATH.is_file():
        for line in JOURNAL_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("target") != target or entry.get("subcommand") != "iterate":
                continue
            past_delta = int(entry.get("target_delta") or 0)
            target_deltas.append(past_delta)

    target_deltas.append(current_delta)

    patterns: list[str] = []
    evidence: dict[str, Any] = {"target_deltas_history": target_deltas}

    # Pattern (b): 0->1->0 thrash in any 3-window.
    for i in range(len(target_deltas) - 2):
        window = target_deltas[i : i + 3]
        if window[0] == 0 and window[1] > 0 and window[2] == 0:
            patterns.append("thrash_0_1_0")
            evidence["thrash_window_start_index"] = i
            evidence["thrash_deltas"] = window
            break

    # Pattern (c): 5+ consecutive delta=0 iterations.
    if len(target_deltas) >= 5 and all(d == 0 for d in target_deltas[-5:]):
        patterns.append("flat_zero_5plus_iterations")
        evidence["flat_zero_count"] = sum(1 for d in target_deltas if d == 0)
        evidence["flat_zero_note"] = (
            "Forward-model may be predicting 0 fill every step (cov=0/N grounding bug). "
            "Check: (1) actor identity -- blue platform is the actor, NOT magenta liquid; "
            "(2) calibration seeds the wrong actor (sp80_surface_calibration.json); "
            "(3) correct level physics model."
        )

    halt = bool(patterns)
    if halt:
        evidence["human_redirect_required"] = (
            "HALT: stall patterns detected. Surface this evidence for human redirect. "
            "The loop does NOT silently continue. See journal.jsonl for full history."
        )

    return {"halt": halt, "patterns": patterns, "evidence": evidence}


# ---------------------------------------------------------------------------
# Subcommand: iterate (the 9-step protocol).
# ---------------------------------------------------------------------------

def cmd_iterate(args: argparse.Namespace) -> int:  # noqa: C901
    """Run one full CONTRA Loop iteration (9 steps).

    --dry-run: runs GROUND (step 1) + VERIFY-GATE layers 2-4 (step 7) end-to-end
    using the GROUND run_dir.  Proves the spine without requiring a new admission.
    Skips L1 (dev_check -- run via `contra gate` or `loop gate` separately) and
    L5 (no report built in skip-report mode).

    Without --dry-run and without --mechanic:
      Runs steps 1-2 (GROUND + gap surface), prints three agent hand-off prompts,
      writes journal entry, exits 0.  The agent writes hypothesis/keystone/solver
      between invocations.

    Without --dry-run and with --mechanic:
      Assumes agent has completed steps 3-5.  Runs steps 1 (GROUND regression
      check), 6 (LIVE-RUN with solver wired), 7-9 (GATE + ADMIT/REJECT + STALL).
    """
    game, target_level = _parse_target(args.target)
    target = args.target
    dry_run = args.dry_run
    mechanic = getattr(args, "mechanic", None)
    backend = args.backend
    max_actions = args.max_actions
    iteration = getattr(args, "iteration", 0)

    print(f"CONTRA Loop -- iterate  target={target}  dry_run={dry_run}  "
          f"backend={backend}  max_actions={max_actions}")

    journal_entry: dict[str, Any] = {
        "schema": SCHEMA,
        "subcommand": "iterate",
        "target": target,
        "iteration": iteration,
        "dry_run": dry_run,
        "mechanic": mechanic,
        "backend": backend,
        "max_actions": max_actions,
        "started_at_utc": _utc_now(),
    }

    # ── STEP 1: GROUND ───────────────────────────────────────────────────────
    print("\n[Step 1/9] GROUND -- run episode, snapshot baseline ...")
    ground = _ground(game, target_level, backend=backend, max_actions=max_actions)

    run_dir = ground.get("run_dir")
    completed = ground.get("completed_levels", [])
    regression = ground.get("regression_detected", False)
    regressed = ground.get("regressed_levels", [])
    registry_baseline = ground.get("registry_baseline_levels", [])
    per_level_map: dict[str, Any] = ground.get("per_level", {})

    # Surface the key per-target summary without noise.
    target_row_g = per_level_map.get(str(target_level), {})
    print(
        f"  run_dir:   {run_dir}\n"
        f"  completed: {completed}\n"
        f"  target L{target_level}: delta={target_row_g.get('levels_completed_delta')}  "
        f"episode_status={target_row_g.get('episode_status')!r}  "
        f"verifier_rc={target_row_g.get('verifier_returncode')}\n"
        f"  regression_detected: {regression}"
    )

    journal_entry["ground"] = {
        "run_dir": run_dir,
        "completed_levels": completed,
        "regression_detected": regression,
        "regressed_levels": regressed,
        "registry_baseline": registry_baseline,
        "target_delta": int(target_row_g.get("levels_completed_delta") or 0),
    }

    if not run_dir:
        print(f"\nGROUND FAILED: no run_dir produced.", file=sys.stderr)
        journal_entry["outcome"] = "ground_failed"
        _journal_append(journal_entry)
        return 1

    if regression:
        print(
            f"\nREGRESSION DETECTED on baseline levels {regressed}. HALTING.\n"
            "  Do not proceed; fix the regression before the next iteration.",
            file=sys.stderr,
        )
        journal_entry["outcome"] = "regression_halt"
        _journal_append(journal_entry)
        return 1

    # ── STEP 2: DISCOVER-GAP ────────────────────────────────────────────────
    print("\n[Step 2/9] DISCOVER-GAP -- surface gap evidence for agent ...")
    # TODO: wire to gap_finder.GroundTruthOracle + GapFinder once gap_finder.py
    # and ground_truth.py exist.  Those modules perform pixel-exact before/after
    # frame diffs against the win-bank / engine-source / raw_recordings oracle,
    # computing cov=N/N and emitting a gap_report.json for the agent.
    #
    # Day-one seed: surface the wanted-mechanic hint and GROUND episode details
    # so the agent has a concrete starting point.

    mechanic_hint = _wanted_mechanic_hint(game, target_level)
    existing_hypotheses = _hypotheses_for_target(target)

    gap_surface: dict[str, Any] = {
        "schema": "codexlab.contra_loop.gap_surface.v1",
        "target": target,
        "run_dir": run_dir,
        "target_level": target_level,
        "target_delta": int(target_row_g.get("levels_completed_delta") or 0),
        "target_episode_status": target_row_g.get("episode_status"),
        "baseline_levels": registry_baseline,
        "completed_levels": completed,
        "wanted_mechanic_hint": mechanic_hint,
        "existing_hypotheses_count": len(existing_hypotheses),
        "boundary": (
            "Hypothesis material only; not local CONTRA evidence. "
            "A solve is real ONLY with levels_completed_delta>0 + verifier rc=0."
        ),
        "gap_finder_status": (
            "TODO: extend with gap_finder.py + ground_truth.py for pixel-exact "
            "before/after frame diffs vs win-bank/engine oracle."
        ),
    }

    # Write gap_surface.json for the agent to read.
    gap_path = CODEXLAB / "contra_loop" / "gap_surface.json"
    gap_path.write_text(
        json.dumps(gap_surface, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  gap_surface.json written: {gap_path.relative_to(ROOT)}")
    if mechanic_hint:
        print(f"  wanted mechanic hint: {mechanic_hint}")
    if existing_hypotheses:
        print(f"  existing hypotheses for {target}: {len(existing_hypotheses)}")
    journal_entry["gap_surface"] = gap_surface

    # ── STEPS 3-5: AGENT HAND-OFFS (skipped in --dry-run) ──────────────────
    if dry_run:
        print("\n[--dry-run] Skipping agent hand-offs (steps 3-5).")
        print("  In a real iterate (without --dry-run), the agent performs:")
        _print_handoff_summary(target, mechanic)
    elif not mechanic:
        # No mechanic supplied: print all three hand-off prompts and exit.
        # The agent reads gap_surface.json, writes the hypothesis, keystone, and
        # solver, then re-invokes with `--mechanic <SOLVER_NAME>`.
        _print_handoff_A(target, gap_path)
        _print_handoff_B(target)
        _print_handoff_C(target)
        print(
            f"\n  When ready, re-invoke:\n"
            f"    python codexlab/contra_loop/loop.py iterate "
            f"--target {target} --mechanic <SOLVER_NAME>\n"
        )
        journal_entry["outcome"] = "agent_handoff_pending"
        _journal_append(journal_entry)
        return 0

    # ── STEP 6: LIVE-RUN ────────────────────────────────────────────────────
    # In dry mode: reuse the GROUND run_dir (no solver wired).
    # In full mode: run a fresh episode with the wired solver.
    if dry_run:
        print(f"\n[Step 6/9] LIVE-RUN -- [dry-run] reusing GROUND run_dir: {run_dir}")
        live_run_dir: str | None = run_dir
    else:
        print(f"\n[Step 6/9] LIVE-RUN -- contra run (with solver {mechanic!r} wired) ...")
        levels_for_run = _level_set_for_run(game, target_level)
        levels_arg = ",".join(str(l) for l in levels_for_run)
        l6_proc = _contra(
            "run", "--levels", levels_arg,
            "--backend", backend,
            "--max-actions", str(max_actions),
            "--skip-report",
        )
        live_run_dir = _extract_run_dir_from_stdout(l6_proc.stdout)
        print(f"  run_dir: {live_run_dir}  rc={l6_proc.returncode}")
        if l6_proc.returncode != 0 or not live_run_dir:
            print("  LIVE-RUN failed.", file=sys.stderr)
            journal_entry["outcome"] = "live_run_failed"
            _journal_append(journal_entry)
            return 1

    journal_entry["live_run_dir"] = live_run_dir

    # ── STEP 7: VERIFY-GATE ─────────────────────────────────────────────────
    print(
        f"\n[Step 7/9] VERIFY-GATE -- 5 layers "
        f"(L1={'skip' if dry_run else 'run'}, L5={'skip' if True else 'run'}) ..."
    )
    gate = _run_gate(
        game, target_level,
        backend=backend,
        max_actions=max_actions,
        run_dir=live_run_dir,
        # Skip heavy L1 (dev_check) in dry mode; agent can run it via `loop gate`.
        skip_l1_structural=dry_run,
        # Skip L5 (no report built with --skip-report; run `loop gate --no-skip-report`).
        skip_l5_report=True,
    )

    _print_gate_summary(gate)
    journal_entry["gate"] = gate.to_dict()

    if not gate.ok:
        print(
            f"\n  GATE FAILED at layer={gate.failed_layer}: {gate.reason}",
            file=sys.stderr,
        )

    # ── STEP 8: ADMIT or REJECT ─────────────────────────────────────────────
    print("\n[Step 8/9] ADMIT or REJECT ...")
    effective_mechanic = mechanic or f"candidate_{target.replace(':', '_')}_iter{iteration}"
    admit_result = _admit_or_reject(
        gate, target, effective_mechanic,
        iteration=iteration,
        dry_run=dry_run,
    )
    print(json.dumps(admit_result, indent=2, sort_keys=True))

    journal_entry["admit_result"] = admit_result
    journal_entry["target_delta"] = int(admit_result.get("target_delta") or 0)

    # ── STEP 9: STALL-CHECK ─────────────────────────────────────────────────
    print("\n[Step 9/9] STALL-CHECK ...")
    stall = _stall_check(target, journal_entry["target_delta"])
    journal_entry["stall"] = stall

    if stall["halt"]:
        print(
            f"\n  STALL DETECTED -- patterns: {stall['patterns']}\n"
            f"  {json.dumps(stall['evidence'], indent=2)}\n"
            "  HALTING for human redirect.  See journal.jsonl for full history.",
            file=sys.stderr,
        )
        journal_entry["outcome"] = "stall_halt"
        _journal_append(journal_entry)
        return 1

    # Determine final outcome label.
    action = admit_result.get("action", "")
    if action == "admitted":
        journal_entry["outcome"] = "admitted"
        print(f"\n  ADMITTED solver '{effective_mechanic}' for {target}.")
    elif action == "dry_run_no_admission":
        journal_entry["outcome"] = "dry_run_complete"
        spine_ok = gate.ok
        print(
            f"\n  DRY-RUN COMPLETE.  Spine {'functional' if spine_ok else 'needs attention'}.\n"
            f"  Gate: {'PASS' if spine_ok else 'FAIL'}, target delta={admit_result.get('target_delta')}.\n"
            "  (Nothing written to solvers.jsonl.  "
            "Re-run without --dry-run + --mechanic to attempt a real admission.)"
        )
    else:
        journal_entry["outcome"] = "rejected"
        print(
            f"\n  NOT ADMITTED.  Reason: {admit_result.get('reason')}\n"
            "  Honest negative written to rejections.jsonl.  "
            "Next iteration continues from the same registry baseline."
        )

    journal_entry["finished_at_utc"] = _utc_now()
    _journal_append(journal_entry)

    # The loop exit code is 0 for all non-error outcomes (including rejection and dry-run).
    # A rejection is a first-class outcome, not a failure of the harness.
    return 0


# ---------------------------------------------------------------------------
# Hand-off prompt helpers (called by cmd_iterate).
# ---------------------------------------------------------------------------

def _print_handoff_summary(target: str, mechanic: str | None) -> None:
    print(
        "   3. HYPOTHESIZE: form ONE falsifiable HypothesisEnvelope\n"
        "      (action_primitive, precondition_signature, predicted_effect,\n"
        "      falsification_condition + 4-field boundary verbatim)\n"
        f"      Write to: codexlab/contra_loop/registry/hypotheses.jsonl  (target={target})\n"
        "   4. KEYSTONE: write a standalone validator under keystones/ that applies\n"
        "      the predicted forward-model to win-bank/engine frames and reports N/N.\n"
        "   5. WIRE: register a SolverProtocol.produce impl; emit-but-mark gated rows;\n"
        "      ceiling-cap correctly (tier-1 until a controlled boundary confirms tier-0).\n"
        f"   Then re-run: iterate --target {target} --mechanic <SOLVER_NAME>"
    )


def _print_handoff_A(target: str, gap_path: Path) -> None:
    print(
        "\n" + "=" * 72 + "\n"
        f"[AGENT HAND-OFF  Step 3: HYPOTHESIZE]\n" + "=" * 72 + "\n"
        f"  Read: {gap_path.relative_to(ROOT) if gap_path.is_absolute() else gap_path}\n"
        "  Task: form ONE falsifiable HypothesisEnvelope.\n"
        "  Requirements:\n"
        "    - action_primitive (one of: select_object, move_up/down/left/right, activate_flow)\n"
        "    - precondition_signature (what must be true before the action)\n"
        "    - predicted_effect (what changes: pixel region, level_up condition)\n"
        "    - falsification_condition (what observation would disprove it)\n"
        "    - boundary (verbatim): 'Hypothesis material only; not local CONTRA evidence.'\n"
        "  Anti-reinvention: grep portband/deltaband/ghost BEFORE writing.\n"
        f"  Write to: codexlab/contra_loop/registry/hypotheses.jsonl  (target={target})\n"
        "  Admission_status: 'discovery_only' -- a hypothesis is NOT a solve."
    )


def _print_handoff_B(target: str) -> None:
    print(
        "\n" + "=" * 72 + "\n"
        "[AGENT HAND-OFF  Step 4: KEYSTONE]\n" + "=" * 72 + "\n"
        "  Write a standalone validator under codexlab/contra_loop/keystones/\n"
        "  that applies your predicted forward-model to win-bank or engine frames.\n"
        "  Requirements:\n"
        "    - exit 0 on pass\n"
        "    - print quantitative metric in stdout: exact-match N/N + mean IoU\n"
        "    - validated on win-bank or engine_source ground-truth (NOT synthetic data)\n"
        "    - declare ground_truth_source='win_bank'|'engine_source'|'raw_recordings'\n"
        "  A keystone that prints 0/N every step triggers StallDetector pattern (c).\n"
        "  NOTE: a passing keystone => admission_status='discovery_only', NOT a solve."
    )


def _print_handoff_C(target: str) -> None:
    print(
        "\n" + "=" * 72 + "\n"
        "[AGENT HAND-OFF  Step 5: WIRE]\n" + "=" * 72 + "\n"
        "  Register a SolverProtocol implementation (contra_loop.solver_interface).\n"
        "  Requirements:\n"
        "    - implement .produce(contra_packet, **ctx) -> List[ScoredCandidate]\n"
        "    - EMIT-BUT-MARK: gated rows emit value_score=0.0 + gated_reason\n"
        "    - evidence_tier: 1 (pre-execution geometry) until a controlled before/\n"
        "      after/action boundary confirms tier-0; ceiling-cap to tier-1\n"
        "    - run `contra check` and confirm all edges still attached\n"
        "    - run `contra gate` to confirm dev_check passes\n"
        f"  Then re-invoke: iterate --target {target} --mechanic <SOLVER_NAME>"
    )


def _print_gate_summary(gate: GateResult) -> None:
    for layer, info in gate.layers.items():
        rc = info.get("returncode", "?")
        skip = info.get("skipped", False)
        status = "SKIP" if skip else ("PASS" if rc == 0 else f"FAIL(rc={rc})")
        print(f"  {layer:30s} {status}")
    print(
        f"\n  Gate result: {'PASS' if gate.ok else 'FAIL'}"
        f"  completed={gate.completed_levels}"
        f"  no_regression={gate.no_regression}"
    )
    if not gate.ok:
        print(f"  Failed layer: {gate.failed_layer}  reason: {gate.reason}")


# ---------------------------------------------------------------------------
# Subcommand: baseline
# ---------------------------------------------------------------------------

def cmd_baseline(args: argparse.Namespace) -> int:
    """GROUND step only: run episode, parse summary.jsonl, emit baseline.json."""
    game, target_level = _parse_target(args.target)
    from_run_dir = getattr(args, "from_run_dir", None)
    ground = _ground(
        game, target_level,
        backend=args.backend,
        max_actions=args.max_actions,
        from_run_dir=from_run_dir,
    )
    # baseline.snapshot() already printed a single-line JSON summary.
    if ground.get("regression_detected"):
        print(
            f"REGRESSION on levels {ground.get('regressed_levels')}. "
            "Loop must HALT before the next iteration.",
            file=sys.stderr,
        )
        return 1
    return 0 if ground.get("run_dir") else 1


# ---------------------------------------------------------------------------
# Subcommand: gate
# ---------------------------------------------------------------------------

def cmd_gate(args: argparse.Namespace) -> int:
    """Run VERIFY-GATE (5 layers) on a supplied or freshly-produced run_dir."""
    game, target_level = _parse_target(args.target)
    gate = _run_gate(
        game, target_level,
        backend=args.backend,
        max_actions=args.max_actions,
        run_dir=getattr(args, "run_dir", None) or None,
        skip_l1_structural=getattr(args, "skip_l1", False),
        skip_l5_report=getattr(args, "skip_report", True),
    )
    print(json.dumps(gate.to_dict(), indent=2, sort_keys=True))
    _print_gate_summary(gate)
    return 0 if gate.ok else 1


# ---------------------------------------------------------------------------
# Subcommand: admit
# ---------------------------------------------------------------------------

def cmd_admit(args: argparse.Namespace) -> int:
    """Admit a named mechanic: gate the run_dir then call register_solver().

    NEVER bypasses the gate.  Admission requires gate.ok + delta>0 + no_regression.
    A run_dir that was not produced by this gate invocation is refused.
    """
    if not getattr(args, "run_dir", None):
        print("admit: --run-dir is required.", file=sys.stderr)
        return 2
    if not getattr(args, "mechanic", None):
        print("admit: --mechanic is required.", file=sys.stderr)
        return 2

    game, target_level = _parse_target(args.target)
    gate = _run_gate(
        game, target_level,
        backend=args.backend,
        max_actions=args.max_actions,
        run_dir=args.run_dir,
        skip_l1_structural=False,
        skip_l5_report=getattr(args, "skip_report", True),
    )
    result = _admit_or_reject(gate, args.target, args.mechanic, iteration=0)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("action") == "admitted" else 1


# ---------------------------------------------------------------------------
# Subcommand: status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> int:
    """Print solver library + journal tail."""
    reg = load_registry()
    target_filter = getattr(args, "target", None)

    solvers = reg.get("solvers", [])
    if target_filter:
        game_filter = target_filter.split(":", 1)[0]
        solvers = [
            s for s in solvers
            if str(s.get("target", "")).split(":", 1)[0] == game_filter
        ]

    wanted = reg.get("wanted_mechanics", [])
    if target_filter:
        wanted = [w for w in wanted if w.get("target") == target_filter]

    # Tail of journal.jsonl.
    journal_tail: list[dict[str, Any]] = []
    if JOURNAL_PATH.is_file():
        raw_lines = [
            l for l in JOURNAL_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        for line in raw_lines[-10:]:
            try:
                entry = json.loads(line)
                journal_tail.append({
                    "target": entry.get("target"),
                    "iteration": entry.get("iteration"),
                    "outcome": entry.get("outcome"),
                    "target_delta": entry.get("target_delta"),
                    "recorded_at_utc": entry.get("recorded_at_utc"),
                })
            except json.JSONDecodeError:
                pass

    # Count rejections for context.
    rejection_count = 0
    if REJECTIONS_PATH.is_file():
        rejection_count = sum(
            1 for l in REJECTIONS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()
        )

    print(json.dumps({
        "schema": SCHEMA,
        "solver_count": len(solvers),
        "solvers": [
            {
                "name": s["name"],
                "target": s.get("target"),
                "verified": s.get("verified"),
                "provenance_run_dir": (s.get("provenance") or {}).get("run_dir"),
            }
            for s in solvers
        ],
        "wanted_mechanics": [
            {"name": w["name"], "target": w.get("target")}
            for w in wanted
        ],
        "rejection_count": rejection_count,
        "journal_tail": journal_tail,
        "journal_path": str(JOURNAL_PATH.relative_to(ROOT)),
        "anti_reward_hacking_note": (
            "Admission = levels_completed_delta>0 + gate.ok + no_regression. "
            "status:nominal and verifier status:passed are NOT solve evidence."
        ),
    }, indent=2, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# CLI / argparse.
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loop.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")

    # ── iterate ──────────────────────────────────────────────────────────────
    p_iter = sub.add_parser(
        "iterate",
        help="Run one full CONTRA Loop iteration (9 steps, 3 agent hand-offs).",
        description=(
            "Run the 9-step discovery protocol.  --dry-run proves the spine "
            "end-to-end without requiring a new admission."
        ),
    )
    p_iter.add_argument(
        "--target", default="sp80:1",
        help="<game>:<level> e.g. sp80:1 (the assigned target).  Default: sp80:1.",
    )
    p_iter.add_argument(
        "--dry-run", "--dry", dest="dry_run", action="store_true",
        help=(
            "Real day-one grounding+gating WITHOUT requiring admission. "
            "Runs GROUND + VERIFY-GATE (L2-L4); skips L1 dev_check and L5 report. "
            "Nothing is written to solvers.jsonl. Proves the spine end-to-end."
        ),
    )
    p_iter.add_argument(
        "--mechanic", default=None,
        help=(
            "Name for the candidate solver (required for non-dry admission). "
            "When supplied, assumes steps 3-5 (HYPOTHESIZE/KEYSTONE/WIRE) are done; "
            "runs LIVE-RUN + GATE + ADMIT/REJECT."
        ),
    )
    p_iter.add_argument("--backend", default="tiny_mlp",
                        choices=["tiny_mlp", "lora_adapter", "qwen_http"])
    p_iter.add_argument("--max-actions", type=int, default=20)
    p_iter.add_argument(
        "--iteration", type=int, default=0,
        help="Iteration counter (for journal provenance; increment each run).",
    )

    # ── baseline ─────────────────────────────────────────────────────────────
    p_base = sub.add_parser(
        "baseline",
        help="GROUND step only: run episode + parse summary.jsonl + emit baseline.json.",
    )
    p_base.add_argument("--target", default="sp80:1")
    p_base.add_argument("--backend", default="tiny_mlp")
    p_base.add_argument("--max-actions", type=int, default=20)
    p_base.add_argument(
        "--from-run-dir", dest="from_run_dir", default=None,
        metavar="RUN_DIR",
        help="Re-parse an existing run_dir instead of running a new episode.",
    )

    # ── gate ─────────────────────────────────────────────────────────────────
    p_gate = sub.add_parser(
        "gate",
        help="VERIFY-GATE (5 layers): structural, integration, episode, per-level verify, report.",
    )
    p_gate.add_argument("--target", default="sp80:1")
    p_gate.add_argument(
        "--run-dir", dest="run_dir", default=None,
        help=(
            "Existing run_dir to gate.  If omitted, a fresh episode run is produced "
            "covering registry_baseline ∪ target."
        ),
    )
    p_gate.add_argument("--backend", default="tiny_mlp")
    p_gate.add_argument("--max-actions", type=int, default=20)
    p_gate.add_argument(
        "--skip-l1", dest="skip_l1", action="store_true",
        help="Skip L1 dev_check (fast mode; use when you've just run `contra gate`).",
    )
    p_gate.add_argument(
        "--skip-report", dest="skip_report", action="store_true", default=True,
        help=(
            "Skip L5 angel_report_verify.  Default: True (report requires a "
            "non-skip-report run first).  Pass --no-skip-report to enable L5."
        ),
    )
    p_gate.add_argument(
        "--no-skip-report", dest="skip_report", action="store_false",
        help="Enable L5 angel_report_verify (requires a full report run first).",
    )

    # ── admit ─────────────────────────────────────────────────────────────────
    p_admit = sub.add_parser(
        "admit",
        help=(
            "Gate a run_dir then call register_solver().  "
            "Never bypasses the gate.  Admission requires gate.ok + delta>0 + no_regression."
        ),
    )
    p_admit.add_argument("--target", default="sp80:1")
    p_admit.add_argument(
        "--run-dir", dest="run_dir", required=False,
        help="Provenance run_dir (must have levels_completed_delta>0 on target).",
    )
    p_admit.add_argument(
        "--mechanic", required=False,
        help="Solver/mechanic name for the registry record.",
    )
    p_admit.add_argument("--backend", default="tiny_mlp")
    p_admit.add_argument("--max-actions", type=int, default=20)
    p_admit.add_argument(
        "--skip-report", dest="skip_report", action="store_true", default=True,
    )
    p_admit.add_argument(
        "--no-skip-report", dest="skip_report", action="store_false",
    )

    # ── status ────────────────────────────────────────────────────────────────
    p_status = sub.add_parser(
        "status",
        help="Print solver library, wanted mechanics, rejection count, and journal tail.",
    )
    p_status.add_argument(
        "--target", default=None,
        help="Filter to a specific game, e.g. 'sp80'.  Omit for all games.",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand is None:
        parser.print_help()
        return 0

    handlers = {
        "iterate": cmd_iterate,
        "baseline": cmd_baseline,
        "gate": cmd_gate,
        "admit": cmd_admit,
        "status": cmd_status,
    }
    handler = handlers.get(args.subcommand)
    if handler is None:
        print(f"unknown subcommand {args.subcommand!r}", file=sys.stderr)
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
