#!/usr/bin/env python3
"""CONTRA Loop Harness -- five-layer VerificationGate.

VerificationGate.run(target, baseline_levels) executes all five gate layers in
fixed order, short-circuits on the first failure, and returns an ArtifactBundle
(all layers passed + admission predicates hold) or a Refusal naming the failed
layer.

Five layers
-----------
  L1  STRUCTURAL   python codexlab/dev_check.py  (via ``contra gate``)
  L2  INTEGRATION  python codexlab/contra.py check  (live + static DAG edges)
  L3  EPISODE      sp80_run_contra_levels --levels <baseline_levels ∪ target>
                   ONE run covering both new target AND every previously-admitted
                   level so regressions cannot hide in a target-only artifact.
  L4  PER-LEVEL    sp80_verify_episode_run.verify_run(level_dir) in-process for
                   EVERY level in the run (returncode 0 each).
  L5  REPORT       angel_report_verify.verify_report(strict_png_ledger=True,
                   require_gameplay_visual_contract=True).

Admission predicate (all must hold simultaneously after all five layers pass)
-----------------------------------------------------------------------------
  METRIC          target row levels_completed_delta > 0 in summary.jsonl
                  (NOT status='nominal', NOT verifier_returncode alone --
                  those are nominal/passed while a level is still unsolved.)
  NO-REGRESSION   completed_levels_after ⊇ baseline_levels
  SOURCE-SHA PIN  summary.json runner/harness/report_source_sha256 match the
                  current worktree files (blocks "edit harness, then run").
  GATE-INTEGRITY  gate script SHAs match gate_pin.json when it exists;
                  "unpinned" on bootstrap but never blocking on first run.

ArtifactBundle | Refusal
------------------------
ArtifactBundle is the ONLY thing registry.admit() may accept.  A Refusal is
appended to rejections.jsonl by registry.reject() and NEVER enters
solvers.jsonl.  Both are dataclasses that serialize to dict via .as_dict().

Note: schema.py (codexlab/contra_loop/schema.py) will become the canonical
home for ArtifactBundle and Refusal once that module ships.  Until then these
dataclasses live here so gate.py is self-contained and import-clean.

Usage
-----
    # plan only -- run nothing, print the five layers:
    python codexlab/contra_loop/gate.py --target sp80:1 --baseline-levels 0 --dry-run

    # full gate run (runs all five layers, may take several minutes):
    python codexlab/contra_loop/gate.py --target sp80:1 --baseline-levels 0

    # bootstrap: write gate_pin.json from the current worktree (human review required):
    python codexlab/contra_loop/gate.py --write-gate-pin

    # show pinned / current gate script SHAs:
    python codexlab/contra_loop/gate.py --show-gate-pin
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Union

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]           # rexus/
CODEXLAB = ROOT / "codexlab"
GATE_PIN_PATH = Path(__file__).resolve().parent / "registry" / "gate_pin.json"
SUMMARY_SCHEMA = "codexlab.sp80_contra_episode_levels.v1"

# Gate scripts whose SHAs are pinned in gate_pin.json.
# Changing any of these requires a human-reviewed gate-change record.
GATE_SCRIPT_FILES: dict[str, Path] = {
    "sp80_verify_episode_run.py": CODEXLAB / "sp80_verify_episode_run.py",
    "dev_check.py": CODEXLAB / "dev_check.py",
    "contra.py": CODEXLAB / "contra.py",
}

# Source files whose SHAs are embedded in summary.json by sp80_run_contra_levels.py
# (sp80_run_contra_levels.py:167-171).  Keys match summary.json field names exactly.
SOURCE_SHA_FILES: dict[str, Path] = {
    "runner_source_sha256": CODEXLAB / "sp80_run_contra_levels.py",
    "harness_source_sha256": CODEXLAB / "sp80_player_harness.py",
    "report_source_sha256": CODEXLAB / "gameplay_report_harness.py",
}

# Known game adapters.  Add new games here without touching the gate layers.
# Each entry needs 'runner': the rel-path under ROOT to the level-run script.
GAME_ADAPTERS: dict[str, dict[str, str]] = {
    "sp80": {
        "runner": "codexlab/sp80_run_contra_levels.py",
    },
}


# ---------------------------------------------------------------------------
# Data transfer objects
# (Promoted to codexlab/contra_loop/schema.py when that module lands.)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ArtifactBundle:
    """A complete, gate-verified artifact.  All fields required for admission.

    Schema: "codexlab.contra_loop.artifact_bundle.v1"

    The solve field is ``levels_completed_delta > 0``.  "nominal" status and
    ``verifier_returncode=0`` alone are NOT solve evidence (an unsolved level
    can carry both while delta=0).
    """

    schema: str = "codexlab.contra_loop.artifact_bundle.v1"
    target: str = ""                          # e.g. "sp80:1"
    run_dir: str = ""                         # relative to rexus/ ROOT
    summary_json_path: str = ""
    summary_json_sha256: str = ""
    summary_schema: str = SUMMARY_SCHEMA

    # The solve field -- > 0 required for admission.
    target_level_index: int = -1
    levels_completed_delta: int = 0

    # Regression baseline.
    completed_levels_after: list[int] = dataclasses.field(default_factory=list)
    completed_levels_baseline: list[int] = dataclasses.field(default_factory=list)
    no_regression: bool = False

    # Per-layer return codes.
    dev_check_returncode: int = -1          # L1
    contra_check_returncode: int = -1       # L2
    verifier_returncode: int = -1           # L4 target-level result
    report_verify_returncode: int = -1      # L5

    all_five_layers_passed: bool = False

    # Content-hash for reverify-on-read (registry.reverify hashes these).
    level_dir_file_hashes: dict[str, str] = dataclasses.field(default_factory=dict)

    # Source provenance (copied from summary.json, sp80_run_contra_levels.py:167-171).
    runner_source_sha256: str = ""
    harness_source_sha256: str = ""
    report_source_sha256: str = ""
    source_sha_pin_ok: bool = False

    # Gate-integrity pin.
    gate_script_hashes: dict[str, str] = dataclasses.field(default_factory=dict)
    gate_integrity_pin_status: str = "unpinned"   # "ok" | "unpinned" | "mismatch"

    git_commit_sha: str = ""
    created_at_utc: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Refusal:
    """Gate refusal.  Names the exact layer that failed.

    Schema: "codexlab.contra_loop.refusal.v1"

    Refusals are appended to rejections.jsonl; they NEVER appear in
    solvers.jsonl.  An honest no-admission is the correct outcome when the
    level is not yet solved.
    """

    schema: str = "codexlab.contra_loop.refusal.v1"
    target: str = ""
    failed_layer: str = ""   # L1_structural | L2_integration | L3_episode |
                              # L4_verify | L5_report | metric | no_regression |
                              # source_sha_pin | gate_integrity_pin
    detail: str = ""
    run_dir: str | None = None
    levels_completed_delta: int = 0         # 0 = not solved (the normal honest case)
    created_at_utc: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# SHA / hashing helpers (inline to avoid importing sp80_base_controller)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _run_stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

def _delegate_env() -> dict[str, str]:
    """Subprocess env with CODEXLAB on PYTHONPATH (mirrors contra.py:_delegate_env)."""
    import os  # noqa: PLC0415
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(CODEXLAB) if not existing else f"{CODEXLAB}{os.pathsep}{existing}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _run_subprocess(args: list[str]) -> tuple[int, str, str]:
    """Run a subprocess under ROOT cwd.  Returns (returncode, stdout, stderr)."""
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env=_delegate_env(),
    )
    return proc.returncode, proc.stdout, proc.stderr


def _git_head() -> str:
    """Current git HEAD SHA (best-effort; returns 'unknown' on failure)."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        return proc.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Gate-pin helpers
# ---------------------------------------------------------------------------

def _current_gate_script_hashes() -> dict[str, str]:
    """Compute SHA-256 for every gate-pinned script."""
    return {
        name: (_sha256_file(path) if path.is_file() else "missing")
        for name, path in GATE_SCRIPT_FILES.items()
    }


def _check_gate_integrity(current_hashes: dict[str, str]) -> tuple[str, str]:
    """Compare current gate script SHAs against gate_pin.json.

    Returns ``(status, detail)`` where status is one of:
      "ok"        -- gate_pin.json exists and all SHAs match.
      "unpinned"  -- gate_pin.json does not exist yet (bootstrap mode).
      "mismatch"  -- gate_pin.json exists but at least one SHA differs.
    """
    if not GATE_PIN_PATH.is_file():
        return (
            "unpinned",
            (
                f"gate_pin.json not found at {GATE_PIN_PATH.relative_to(ROOT)}. "
                "Gate integrity is unverified (bootstrap mode). "
                "Run --write-gate-pin once, then have a human review it before "
                "treating any admission as fully pinned."
            ),
        )

    try:
        pinned_doc = json.loads(GATE_PIN_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return "mismatch", f"gate_pin.json is unreadable: {exc}"

    pinned_hashes: dict[str, str] = pinned_doc.get("gate_script_hashes", {})
    mismatches: list[str] = []
    for name, current_sha in current_hashes.items():
        pinned_sha = pinned_hashes.get(name)
        if pinned_sha is None:
            mismatches.append(f"{name}: not in gate_pin.json")
        elif pinned_sha != current_sha:
            mismatches.append(
                f"{name}: pin={pinned_sha[:12]}...  current={current_sha[:12]}..."
            )

    if mismatches:
        return "mismatch", "Gate script SHA mismatch(es): " + "; ".join(mismatches)

    return "ok", "All gate script SHAs match gate_pin.json"


def _check_source_sha_pin(summary: dict[str, Any]) -> tuple[bool, str]:
    """Verify summary.json's runner/harness/report_source_sha256 against real files.

    sp80_run_contra_levels.py embeds these at run time (lines 167-171); we check
    them here to ensure nobody edited the source between the run and the gate.
    """
    mismatches: list[str] = []
    for field, path in SOURCE_SHA_FILES.items():
        recorded = summary.get(field)
        if not recorded:
            mismatches.append(f"{field}: field absent in summary.json")
            continue
        if not path.is_file():
            mismatches.append(f"{field}: source file missing at {path.relative_to(ROOT)}")
            continue
        actual = _sha256_file(path)
        if actual != recorded:
            mismatches.append(
                f"{field}: recorded {str(recorded)[:12]}...  "
                f"actual {actual[:12]}...  "
                f"(source was edited after the run)"
            )

    if mismatches:
        return False, "Source SHA pin failed: " + "; ".join(mismatches)
    return True, "Source SHA pin OK"


def _level_dir_file_hashes(level_dir: Path) -> dict[str, str]:
    """SHA-256 the three canonical level-dir files (for registry.reverify-on-read)."""
    return {
        fname: (_sha256_file(level_dir / fname) if (level_dir / fname).is_file() else "missing")
        for fname in ("result.json", "packets.jsonl", "brain_calls.jsonl")
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


# ---------------------------------------------------------------------------
# VerificationGate
# ---------------------------------------------------------------------------

class VerificationGate:
    """Five-layer verification gate for the CONTRA Loop Harness.

    The gate is stateless between calls.  Instantiate once; call
    ``.run(target, baseline_levels)`` per gate attempt.

    Every ``run()`` call:
      1. Short-circuits on the first failing layer and returns a Refusal.
      2. On full pass, evaluates the admission predicate and returns either
         an ArtifactBundle (admit) or a Refusal (metric / no-regression /
         source-sha-pin / gate-integrity-pin failure).

    The run_dir is produced inside this call; ``registry.admit()`` MUST
    refuse any run_dir not produced by the gate invocation it is processing
    (blocks externally-supplied forged run dirs).
    """

    def __init__(
        self,
        *,
        backend: str = "tiny_mlp",
        max_actions: int = 20,
        verbose: bool = True,
    ) -> None:
        self.backend = backend
        self.max_actions = max_actions
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    # ------------------------------------------------------------------
    # Individual layer runners (private -- called only by run())
    # ------------------------------------------------------------------

    def _l1_structural(self) -> tuple[int, str]:
        """L1: ``contra gate`` -> dev_check.py.

        Covers: syntax check on ~125 .py files, 4 regression suites,
        contra-service-contract drift, in-process↔HTTP byte parity, import smoke.
        """
        self._log("  [L1] contra gate (dev_check) ...")
        rc, out, err = _run_subprocess(
            [sys.executable, "-B", "codexlab/contra.py", "gate"]
        )
        return rc, (out + err)[-2000:].strip()

    def _l2_integration(self) -> tuple[int, str]:
        """L2: ``contra check`` -> live + static DAG edges.

        Exercises the real CONTRA path on the smoke frame:
        Oracle→CONTRA, Ghost→CONTRA, Oracle→scorer, Ghost→scorer.
        Also checks five source tripwires (harness still calls
        run_contra_view, build_contra_recommendation_packet, etc.).
        """
        self._log("  [L2] contra check (live+static DAG edges) ...")
        rc, out, err = _run_subprocess(
            [sys.executable, "-B", "codexlab/contra.py", "check"]
        )
        return rc, (out + err)[-2000:].strip()

    def _l3_episode(
        self,
        game: str,
        levels: list[int],
        run_dir: Path,
    ) -> tuple[int, str]:
        """L3: sp80_run_contra_levels --levels <all> --out-dir <run_dir>.

        Runs ALL baseline + target levels in ONE command so the single
        summary.json captures both new progress AND regressions.
        Does NOT pass --skip-report so the Angel report is built here;
        L5 verifies it independently.
        """
        runner = GAME_ADAPTERS[game]["runner"]
        level_str = ",".join(str(lvl) for lvl in levels)
        self._log(
            f"  [L3] sp80_run_contra_levels --levels {level_str} -> {run_dir.name} ..."
        )
        cmd = [
            sys.executable, "-B", f"codexlab/{runner}",
            "--levels", level_str,
            "--backend", self.backend,
            "--max-actions", str(self.max_actions),
            "--out-dir", run_dir.relative_to(ROOT).as_posix(),
            # No --skip-report: the report is produced here so L5 can re-verify it.
        ]
        rc, out, err = _run_subprocess(cmd)
        return rc, (out[-1200:] + err[-1200:]).strip()

    def _l4_verify_all(
        self,
        run_dir: Path,
        levels: list[int],
    ) -> tuple[int, str, dict[int, dict[str, Any]]]:
        """L4: in-process verify_run for every level.

        Returns ``(rc, detail, per_level_results)``.
        rc=0 only if ALL levels pass.

        sp80_verify_episode_run uses only stdlib so the import is safe.
        """
        # Ensure CODEXLAB is importable in this process.
        if str(CODEXLAB) not in sys.path:
            sys.path.insert(0, str(CODEXLAB))

        try:
            from sp80_verify_episode_run import verify_run  # noqa: PLC0415
        except ImportError as exc:
            return 1, f"cannot import sp80_verify_episode_run: {exc}", {}

        per_level: dict[int, dict[str, Any]] = {}
        for lvl in levels:
            level_dir = run_dir / f"level_{lvl:02d}"
            self._log(f"  [L4] verify_run level_{lvl:02d} ...")
            try:
                result = verify_run(level_dir)
                per_level[lvl] = result
            except AssertionError as exc:
                per_level[lvl] = {"status": "failed", "detail": str(exc)}
                return (
                    1,
                    f"level_{lvl:02d}: verify_run assertion failed: {exc}",
                    per_level,
                )
            except Exception as exc:  # noqa: BLE001
                per_level[lvl] = {"status": "error", "detail": str(exc)}
                return (
                    1,
                    f"level_{lvl:02d}: verify_run error: {exc}",
                    per_level,
                )

        failed = [
            (lvl, r)
            for lvl, r in per_level.items()
            if r.get("status") != "passed"
        ]
        if failed:
            lvl, r = failed[0]
            return (
                1,
                f"level_{lvl:02d}: verify_run returned status={r.get('status')!r}",
                per_level,
            )

        return 0, "all verify_run checks passed", per_level

    def _l5_report(self, run_dir: Path) -> tuple[int, str]:
        """L5: angel_report_verify on the angel_report dir produced in L3.

        Calls verify_report in-process (angel_report_verify uses only stdlib).
        strict_png_ledger=True and require_gameplay_visual_contract=True are
        required to match the design spec.
        """
        report_dir = run_dir / "angel_report"
        self._log(f"  [L5] angel_report_verify {report_dir.relative_to(ROOT)} ...")

        if not report_dir.is_dir():
            return (
                1,
                (
                    f"angel_report directory not found at "
                    f"{report_dir.relative_to(ROOT)}. "
                    "The report harness was not run or failed to produce output "
                    "(check L3 output above)."
                ),
            )

        if str(CODEXLAB) not in sys.path:
            sys.path.insert(0, str(CODEXLAB))

        try:
            from angel_report_verify import verify_report  # noqa: PLC0415
        except ImportError as exc:
            return 1, f"cannot import angel_report_verify: {exc}"

        failures = verify_report(
            report_dir,
            require_reachable=False,
            require_screenshot=False,
            strict_png_ledger=True,
            require_gameplay_visual_contract=True,
        )
        if failures:
            return (
                1,
                "angel_report_verify failures: " + "; ".join(failures[:8]),
            )
        return 0, "angel_report_verify passed"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        target: str,
        baseline_levels: list[int],
    ) -> Union[ArtifactBundle, Refusal]:
        """Run all five gate layers and return an ArtifactBundle or Refusal.

        Parameters
        ----------
        target : str
            "<game>:<level>" -- e.g. "sp80:1".  This is the level the gate
            is trying to admit.
        baseline_levels : list[int]
            Level indices already admitted to the registry.  The run covers
            ``sorted(set(baseline_levels) | {target_level})``.  Every
            baseline level must still produce delta > 0 (no-regression).

        Returns
        -------
        ArtifactBundle
            All five layers passed AND all admission predicates hold.
            ``levels_completed_delta > 0`` is the solve field.
        Refusal
            Something failed.  ``.failed_layer`` names the first layer
            (or predicate) that rejected the run.

        Design invariant
        ----------------
        This method is the sole producer of the run_dir inside each call.
        ``registry.admit()`` MUST refuse any ArtifactBundle whose run_dir
        it did not see produced in the current gate invocation -- this blocks
        externally-supplied forged run dirs.
        """
        now = _utc_now()

        # --- Parse target ---
        game, _, level_str = target.partition(":")
        if game not in GAME_ADAPTERS:
            return Refusal(
                target=target,
                failed_layer="L1_structural",
                detail=(
                    f"unknown game adapter {game!r}; "
                    f"known adapters: {sorted(GAME_ADAPTERS)}"
                ),
                created_at_utc=now,
            )
        try:
            target_level = int(level_str)
        except (ValueError, TypeError):
            return Refusal(
                target=target,
                failed_layer="L1_structural",
                detail=f"target level index is not an integer: {level_str!r}",
                created_at_utc=now,
            )

        levels_to_run = sorted(set(baseline_levels) | {target_level})

        # Produce a deterministic run_dir path for this gate invocation.
        # The gate owns this path; nobody else may supply it.
        stamp = _run_stamp()
        level_label = "_".join(f"{lvl:02d}" for lvl in levels_to_run)
        run_dir = CODEXLAB / "runs" / f"gate_run_{level_label}_{stamp}"

        self._log(
            f"\n=== VerificationGate.run  target={target}  "
            f"baseline={baseline_levels}  levels={levels_to_run} ==="
        )
        self._log(f"    run_dir -> {run_dir.relative_to(ROOT)}")

        # Compute gate-script SHAs early (included in both ArtifactBundle and Refusal).
        gate_hashes = _current_gate_script_hashes()
        gate_pin_status, gate_pin_detail = _check_gate_integrity(gate_hashes)

        # Gate-integrity is BLOCKING when gate_pin.json exists but SHAs differ.
        # "unpinned" (no gate_pin.json) is a warning, not a block, to allow bootstrap.
        if gate_pin_status == "mismatch":
            return Refusal(
                target=target,
                failed_layer="gate_integrity_pin",
                detail=(
                    f"{gate_pin_detail}  "
                    "Changing a gate script requires a separate human-reviewed "
                    "gate-change record (update gate_pin.json and note the reason)."
                ),
                created_at_utc=now,
            )
        if gate_pin_status == "unpinned":
            self._log(f"  [gate-pin] WARNING: {gate_pin_detail}")
        else:
            self._log("  [gate-pin] OK (all gate script SHAs match gate_pin.json)")

        # ----------------------------------------------------------------
        # L1 -- STRUCTURAL
        # ----------------------------------------------------------------
        l1_rc, l1_detail = self._l1_structural()
        if l1_rc != 0:
            return Refusal(
                target=target,
                failed_layer="L1_structural",
                detail=f"contra gate (dev_check) rc={l1_rc}: {l1_detail[-800:]}",
                created_at_utc=now,
            )
        self._log("  [L1] OK")

        # ----------------------------------------------------------------
        # L2 -- INTEGRATION
        # ----------------------------------------------------------------
        l2_rc, l2_detail = self._l2_integration()
        if l2_rc != 0:
            return Refusal(
                target=target,
                failed_layer="L2_integration",
                detail=f"contra check rc={l2_rc}: {l2_detail[-800:]}",
                created_at_utc=now,
            )
        self._log("  [L2] OK")

        # ----------------------------------------------------------------
        # L3 -- EPISODE PRODUCTION
        # ----------------------------------------------------------------
        l3_rc, l3_detail = self._l3_episode(game, levels_to_run, run_dir)
        # run_dir may or may not exist yet depending on where L3 failed.
        run_dir_rel = (
            run_dir.relative_to(ROOT).as_posix()
            if run_dir.exists()
            else str(run_dir)
        )
        if l3_rc != 0:
            return Refusal(
                target=target,
                failed_layer="L3_episode",
                detail=f"sp80_run_contra_levels rc={l3_rc}: {l3_detail[-800:]}",
                run_dir=run_dir_rel,
                created_at_utc=now,
            )
        self._log("  [L3] OK")

        # ----------------------------------------------------------------
        # L4 -- PER-LEVEL VERIFY
        # ----------------------------------------------------------------
        l4_rc, l4_detail, l4_per_level = self._l4_verify_all(run_dir, levels_to_run)
        if l4_rc != 0:
            return Refusal(
                target=target,
                failed_layer="L4_verify",
                detail=f"verify_run failed: {l4_detail[-800:]}",
                run_dir=run_dir_rel,
                created_at_utc=now,
            )
        self._log("  [L4] OK")

        # ----------------------------------------------------------------
        # L5 -- REPORT PROVENANCE
        # ----------------------------------------------------------------
        l5_rc, l5_detail = self._l5_report(run_dir)
        if l5_rc != 0:
            return Refusal(
                target=target,
                failed_layer="L5_report",
                detail=f"angel_report_verify rc={l5_rc}: {l5_detail[-800:]}",
                run_dir=run_dir_rel,
                created_at_utc=now,
            )
        self._log("  [L5] OK")

        # ================================================================
        # All five layers passed.  Evaluate the admission predicate.
        # ================================================================

        # Load summary.json (written by L3 / sp80_run_contra_levels.py:175).
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            return Refusal(
                target=target,
                failed_layer="metric",
                detail=f"summary.json not found at {summary_path.relative_to(ROOT)}",
                run_dir=run_dir_rel,
                created_at_utc=now,
            )
        try:
            summary: dict[str, Any] = json.loads(
                summary_path.read_text(encoding="utf-8")
            )
        except Exception as exc:  # noqa: BLE001
            return Refusal(
                target=target,
                failed_layer="metric",
                detail=f"cannot parse summary.json: {exc}",
                run_dir=run_dir_rel,
                created_at_utc=now,
            )

        # --- METRIC: levels_completed_delta > 0 for the TARGET level ---
        # CRITICAL: do NOT read status ('nominal') or verifier_returncode alone.
        # An unsolved level can be "nominal" with verifier_returncode=0 while
        # delta=0 -- that is NOT a solve (see the live L1 row in summary.jsonl).
        rows: list[dict[str, Any]] = summary.get("rows", [])
        target_row = next(
            (r for r in rows if int(r.get("level_index", -1)) == target_level),
            None,
        )
        if target_row is None:
            return Refusal(
                target=target,
                failed_layer="metric",
                detail=(
                    f"summary.json has no row for level_index={target_level}. "
                    f"Rows present: {[r.get('level_index') for r in rows]}"
                ),
                run_dir=run_dir_rel,
                created_at_utc=now,
            )

        target_delta = int(target_row.get("levels_completed_delta") or 0)
        if target_delta <= 0:
            # Honest non-admission.  This is the EXPECTED outcome for an unsolved
            # level -- a Refusal here is NOT a bug; it is the loop working correctly.
            return Refusal(
                target=target,
                failed_layer="metric",
                detail=(
                    f"levels_completed_delta={target_delta} for level {target_level}. "
                    f"episode_status={target_row.get('episode_status')!r}  "
                    f"row_status={target_row.get('status')!r}  "
                    f"verifier_returncode={target_row.get('verifier_returncode')!r}. "
                    "NOTE: status='nominal' and verifier_returncode=0 do NOT mean solved. "
                    "Only levels_completed_delta>0 is solve evidence."
                ),
                run_dir=run_dir_rel,
                levels_completed_delta=target_delta,
                created_at_utc=now,
            )

        # --- NO-REGRESSION: completed_levels_after ⊇ baseline_levels ---
        completed_after: list[int] = sorted(summary.get("completed_levels") or [])
        completed_set = set(completed_after)
        baseline_set = set(baseline_levels)
        regressed = sorted(baseline_set - completed_set)
        if regressed:
            return Refusal(
                target=target,
                failed_layer="no_regression",
                detail=(
                    f"regression: level(s) {regressed} were in the baseline "
                    f"but are absent from completed_levels={completed_after}. "
                    "A previously-admitted solver stopped working -- investigate "
                    "before admitting new solvers."
                ),
                run_dir=run_dir_rel,
                levels_completed_delta=target_delta,
                created_at_utc=now,
            )

        # --- SOURCE-SHA PIN ---
        source_pin_ok, source_pin_detail = _check_source_sha_pin(summary)
        if not source_pin_ok:
            return Refusal(
                target=target,
                failed_layer="source_sha_pin",
                detail=source_pin_detail,
                run_dir=run_dir_rel,
                levels_completed_delta=target_delta,
                created_at_utc=now,
            )

        # ================================================================
        # All admission predicates hold.  Build and return ArtifactBundle.
        # ================================================================

        # Collect per-file hashes for the TARGET level dir (registry.reverify uses these).
        target_level_dir = run_dir / f"level_{target_level:02d}"
        file_hashes = _level_dir_file_hashes(target_level_dir)

        # Derive target-level verifier returncode from the in-process L4 results.
        target_l4_result = l4_per_level.get(target_level, {})
        verifier_rc = 0 if target_l4_result.get("status") == "passed" else 1

        git_sha = _git_head()

        bundle = ArtifactBundle(
            target=target,
            run_dir=run_dir_rel,
            summary_json_path=(
                summary_path.relative_to(ROOT).as_posix()
                if summary_path.is_relative_to(ROOT)
                else str(summary_path)
            ),
            summary_json_sha256=_sha256_file(summary_path),
            target_level_index=target_level,
            levels_completed_delta=target_delta,
            completed_levels_after=completed_after,
            completed_levels_baseline=sorted(baseline_levels),
            no_regression=True,
            dev_check_returncode=l1_rc,
            contra_check_returncode=l2_rc,
            verifier_returncode=verifier_rc,
            report_verify_returncode=l5_rc,
            all_five_layers_passed=True,
            level_dir_file_hashes=file_hashes,
            runner_source_sha256=summary.get("runner_source_sha256", ""),
            harness_source_sha256=summary.get("harness_source_sha256", ""),
            report_source_sha256=summary.get("report_source_sha256", ""),
            source_sha_pin_ok=True,
            gate_script_hashes=gate_hashes,
            gate_integrity_pin_status=gate_pin_status,
            git_commit_sha=git_sha,
            created_at_utc=now,
        )

        self._log(
            f"\n  GATE PASSED  target={target}  delta={target_delta}  "
            f"completed_after={completed_after}  "
            f"gate_pin={gate_pin_status}  git={git_sha[:12]}"
        )
        return bundle


# ---------------------------------------------------------------------------
# Gate-pin bootstrap helpers (callable from CLI)
# ---------------------------------------------------------------------------

def write_gate_pin(*, overwrite: bool = False) -> dict[str, Any]:
    """Write gate_pin.json from current worktree SHAs.

    HUMAN REVIEW REQUIRED after writing.  The agent must NOT call this
    automatically; a human must inspect the resulting file and confirm the
    gate scripts are the correct, unweakened versions.

    Returns the pinned document.
    """
    GATE_PIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if GATE_PIN_PATH.is_file() and not overwrite:
        raise FileExistsError(
            f"gate_pin.json already exists at {GATE_PIN_PATH}.  "
            "Pass overwrite=True (or --overwrite on CLI) to replace it.  "
            "Remember: changing gate scripts requires a human-reviewed gate-change record."
        )
    current_hashes = _current_gate_script_hashes()
    doc = {
        "schema": "codexlab.contra_loop.gate_pin.v1",
        "created_at_utc": _utc_now(),
        "description": (
            "SHA-256 allowlist for the CONTRA Loop gate scripts. "
            "Changing any of these files requires updating this document "
            "with a human-reviewed gate-change record. "
            "An agent that rewrites a gate script without updating this pin "
            "will be caught by VerificationGate._check_gate_integrity()."
        ),
        "gate_script_hashes": current_hashes,
        "provenance": {
            "historical_fixes": [
                {
                    "date": "pre-20260604",
                    "file": "sp80_verify_episode_run.py",
                    "fix": "verifier-contract bug: forced rank-1 PortBand priority vs "
                           "no-static-family policy; fixed to require a scored PortBand row "
                           "OR saturated-escape shape (not a winning row).",
                },
                {
                    "date": "20260604-20260608",
                    "file": "sp80_verify_episode_run.py",
                    "fix": "pre-existing episode-verifier failure on L0 packet-2 "
                           "(ExploreBand/PortBand high-value scoring escape check); "
                           "fixed; L0 now verifier_returncode=0 delta=1 in "
                           "contra_sp80_episode_levels_00_01_20260608T175142Z.",
                },
            ],
        },
    }
    GATE_PIN_PATH.write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return doc


def show_gate_pin() -> None:
    """Print current worktree gate script SHAs and compare against gate_pin.json."""
    current = _current_gate_script_hashes()
    print("Current worktree gate script SHAs:")
    for name, sha in current.items():
        print(f"  {name}: {sha}")

    if not GATE_PIN_PATH.is_file():
        print(f"\ngate_pin.json: NOT FOUND at {GATE_PIN_PATH.relative_to(ROOT)}")
        return

    try:
        pinned_doc = json.loads(GATE_PIN_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"\ngate_pin.json: UNREADABLE -- {exc}")
        return

    pinned = pinned_doc.get("gate_script_hashes", {})
    print(f"\ngate_pin.json ({GATE_PIN_PATH.relative_to(ROOT)}):")
    for name in sorted(set(current) | set(pinned)):
        c = current.get(name, "(missing)")
        p = pinned.get(name, "(not pinned)")
        status = "OK" if c == p else "MISMATCH"
        print(f"  [{status}] {name}")
        if c != p:
            print(f"         current: {c}")
            print(f"         pinned:  {p}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--target",
        default="sp80:1",
        help="<game>:<level> to gate-verify (default: sp80:1)",
    )
    p.add_argument(
        "--baseline-levels",
        type=lambda s: [int(x) for x in s.split(",") if x.strip()],
        default=[],
        metavar="0,1,...",
        help="comma-separated previously-admitted level indices (default: none)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print the five-layer plan; execute nothing",
    )
    p.add_argument(
        "--backend",
        default="tiny_mlp",
        help="episode runner backend (default: tiny_mlp)",
    )
    p.add_argument(
        "--max-actions",
        type=int,
        default=20,
        help="max actions per episode level (default: 20)",
    )
    p.add_argument(
        "--write-gate-pin",
        action="store_true",
        help="(bootstrap) write gate_pin.json from current worktree SHAs then exit. "
             "HUMAN REVIEW REQUIRED afterwards.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="allow --write-gate-pin to overwrite an existing gate_pin.json",
    )
    p.add_argument(
        "--show-gate-pin",
        action="store_true",
        help="show current vs pinned gate script SHAs then exit",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.show_gate_pin:
        show_gate_pin()
        return 0

    if args.write_gate_pin:
        try:
            doc = write_gate_pin(overwrite=args.overwrite)
        except FileExistsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(
            f"gate_pin.json written to {GATE_PIN_PATH.relative_to(ROOT)}\n"
            "HUMAN REVIEW REQUIRED: inspect this file and confirm the gate scripts\n"
            "are the correct, unweakened versions before treating any admission\n"
            "as fully gate-integrity-pinned."
        )
        print(json.dumps(doc, indent=2, sort_keys=True))
        return 0

    if args.dry_run:
        game, _, level_str = args.target.partition(":")
        baseline = args.baseline_levels
        try:
            target_level = int(level_str)
        except ValueError:
            print(f"ERROR: invalid target {args.target!r}", file=sys.stderr)
            return 1
        levels = sorted(set(baseline) | {target_level})

        print(f"=== VerificationGate DRY-RUN  target={args.target}  baseline={baseline} ===")
        print(f"  levels to run: {levels}")
        print(f"  backend: {args.backend}  max-actions: {args.max_actions}")
        print()
        print("Layer plan (nothing executed):")
        print(
            f"  L1 STRUCTURAL:   python codexlab/contra.py gate  "
            f"(dev_check.py -- syntax + regression + import smoke)"
        )
        print(
            f"  L2 INTEGRATION:  python codexlab/contra.py check  "
            f"(Oracle/Ghost -> CONTRA -> scorer -> harness live + static edges)"
        )
        print(
            f"  L3 EPISODE:      python codexlab/sp80_run_contra_levels.py "
            f"--levels {','.join(str(l) for l in levels)} --backend {args.backend} "
            f"--max-actions {args.max_actions}"
        )
        print(
            f"  L4 PER-LEVEL:    sp80_verify_episode_run.verify_run(level_dir) "
            f"in-process for each of {levels}"
        )
        print(
            f"  L5 REPORT:       angel_report_verify.verify_report("
            f"strict_png_ledger=True, require_gameplay_visual_contract=True)"
        )
        print()
        print("Admission predicate (after all five layers pass):")
        print(f"  METRIC:           target level {target_level} levels_completed_delta > 0")
        print(f"  NO-REGRESSION:    completed_levels_after ⊇ {baseline}")
        print(
            "  SOURCE-SHA PIN:   summary.json runner/harness/report sha256 == worktree"
        )
        pin_status, _ = _check_gate_integrity(_current_gate_script_hashes())
        print(
            f"  GATE-INTEGRITY:   gate_pin.json status = {pin_status!r}  "
            f"(path: {GATE_PIN_PATH.relative_to(ROOT)})"
        )
        print()
        print(
            "Run without --dry-run to execute.  Expected day-one outcome for an "
            "unsolved level: Refusal(failed_layer='metric', delta=0) -- that is the "
            "correct, honest result and is appended to rejections.jsonl."
        )
        return 0

    # Full gate run.
    gate = VerificationGate(
        backend=args.backend,
        max_actions=args.max_actions,
        verbose=True,
    )
    result = gate.run(args.target, args.baseline_levels)

    print()
    if isinstance(result, ArtifactBundle):
        print("GATE RESULT: ArtifactBundle (PASS)")
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
        return 0
    else:
        print(
            f"GATE RESULT: Refusal (failed_layer={result.failed_layer!r})",
            file=sys.stderr,
        )
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
