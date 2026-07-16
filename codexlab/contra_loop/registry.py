#!/usr/bin/env python3
"""SolverRegistry -- the SOLE admission writer for the CONTRA Loop Harness.

Append-only, content-addressed ledgers:
  registry/solvers.jsonl       admitted solvers  (schema: codexlab.contra_loop.solver_record.v1)
  registry/rejections.jsonl    honest negatives  (dead hypotheses never re-attempted)
  registry/hypotheses.jsonl    discovery-stage   (passing keystone, no episode delta yet)

CARDINAL RULE, enforced structurally:
  admit() requires ``levels_completed_delta > 0`` from the target level row in
  summary.jsonl.  NEVER from ``status: nominal``, NEVER from
  ``status: passed`` in the verifier dict.  These strings exist deliberately to
  prevent confusion and must not be read as solve evidence.

The solver_id IS the proof: it is the SHA-256 fingerprint of the verification
artifact's content-addressed fields.  reverify() re-hashes level files and
re-runs sp80_verify_episode_run on the stored run_dir so a later edit that
breaks reproduction flags artifact_drift immediately.

Usage::

    python codexlab/contra_loop/registry.py --list
    python codexlab/contra_loop/registry.py --reverify-all
    python codexlab/contra_loop/registry.py --baseline-levels
    python codexlab/contra_loop/registry.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]          # repo root
CODEXLAB = ROOT / "codexlab"
REGISTRY_DIR = Path(__file__).resolve().parent / "registry"

SOLVERS_PATH = REGISTRY_DIR / "solvers.jsonl"
REJECTIONS_PATH = REGISTRY_DIR / "rejections.jsonl"
HYPOTHESES_PATH = REGISTRY_DIR / "hypotheses.jsonl"
GATE_PIN_PATH = REGISTRY_DIR / "gate_pin.json"

SOLVER_RECORD_SCHEMA = "codexlab.contra_loop.solver_record.v1"
REJECTION_SCHEMA = "codexlab.contra_loop.rejection.v1"
HYPOTHESIS_SCHEMA = "codexlab.contra_loop.hypothesis.v1"
SUMMARY_SCHEMA = "codexlab.sp80_contra_episode_levels.v1"

# Gate scripts whose SHAs are pinned in gate_pin.json.
# Changing any of these requires a human-reviewed gate-change record that
# updates gate_pin.json.  The registry refuses admission otherwise.
GATE_SCRIPT_KEYS = {
    "sp80_verify_episode_run.py": CODEXLAB / "sp80_verify_episode_run.py",
    "dev_check.py": CODEXLAB / "dev_check.py",
    "contra.py": CODEXLAB / "contra.py",
}

# Level-artifact files that reverify() re-hashes on every read.
# A mismatch means the stored artifact drifted (edit or fabrication attempt).
LEVEL_ARTIFACT_FILES = ("result.json", "packets.jsonl", "brain_calls.jsonl")

# In-memory set of run_dirs that were produced by gate calls in this Python
# process.  admit() refuses any run_dir NOT registered here.
# This prevents an agent from supplying an externally-fabricated run_dir.
_TRUSTED_RUN_DIRS: set[str] = set()


# ---------------------------------------------------------------------------
# ArtifactBundle -- canonical definition will live in schema.py (TODO).
# This dataclass is the transfer object gate.py fills and registry.admit() reads.
# ---------------------------------------------------------------------------

@dataclass
class ArtifactBundle:
    """Verification artifact produced by gate.py.

    gate.py calls ``register_trusted_run_dir(run_dir)`` and populates this
    bundle from the real gate run, then passes it to ``SolverRegistry.admit()``.

    schema.py will be the canonical home for this class once written.
    """

    # Run provenance
    run_dir: str = ""                     # relative from ROOT (episode root, not level_NN)
    summary_json_path: str = ""           # relative from ROOT
    summary_json_sha256: str = ""
    summary_schema: str = SUMMARY_SCHEMA

    # THE solve-evidence field -- must be > 0 for admission
    # (read from summary.jsonl target row, NEVER from status strings)
    levels_completed_delta: int = 0
    target_level_index: int = 0

    # Five-layer gate results (gate.py fills all five)
    verifier_returncode: int = -1
    dev_check_returncode: int = -1
    contra_check_returncode: int = -1
    report_verify_returncode: int = -1
    all_five_layers_passed: bool = False

    # Regression
    completed_levels_after: list[int] = field(default_factory=list)
    completed_levels_baseline: list[int] = field(default_factory=list)
    no_regression: bool = False

    # Content-hash anchors for reverify-on-read
    # {"result.json": sha256, "packets.jsonl": sha256, "brain_calls.jsonl": sha256}
    level_dir_file_hashes: dict[str, str] = field(default_factory=dict)

    # Source provenance pins (copied from summary.json sp80_run_contra_levels.py:167-171)
    runner_source_sha256: str = ""
    harness_source_sha256: str = ""
    report_source_sha256: str = ""

    # Gate-integrity pins (gate.py fills these from current worktree, then
    # registry checks them against gate_pin.json)
    gate_script_hashes: dict[str, str] = field(default_factory=dict)

    # Git anchor at run time
    git_commit_sha: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> tuple[bool, str]:
        """Return (ok, reason).  ok=True means the bundle is structurally complete."""
        if not self.run_dir:
            return False, "run_dir is empty"
        if self.levels_completed_delta <= 0:
            return (
                False,
                f"levels_completed_delta={self.levels_completed_delta} (must be >0 for admission);"
                " note: status:nominal and status:passed are not solve evidence",
            )
        if not self.all_five_layers_passed:
            return False, "all_five_layers_passed is False -- gate did not clear all five layers"
        if not self.no_regression:
            return False, "no_regression is False -- a previously admitted level lost delta>0"
        if not self.level_dir_file_hashes:
            return False, "level_dir_file_hashes is empty -- cannot reverify without artifact hashes"
        return True, "bundle complete"


# ---------------------------------------------------------------------------
# Low-level helpers (no external imports required)
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    """Hash a file's bytes; returns a sentinel string if the file is missing."""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return "file_missing:" + path.name
    return h.hexdigest()


def _canonical_json(value: Any) -> str:
    """Deterministic JSON for hashing (same as ghost.stable_digest basis)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # corrupt line: skip without poisoning the ledger
    return rows


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object to an append-only JSONL ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(_canonical_json(record) + "\n")


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        return result.stdout.strip()
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# Solver ID  (content-addressed -- the ID IS the proof)
# ---------------------------------------------------------------------------

def _solver_id(bundle: ArtifactBundle, mechanic_class: str, source_sha256: str) -> str:
    """Derive a content-addressed solver_id.

    The ID encodes the mechanic spec, solver implementation SHA, and artifact
    file hashes -- so it changes whenever any of those change.
    Per the solver_schema: ``solver_id = "solver_" + sha256 of
    canonicalized {mechanic_spec, solver_impl.source_sha256, level_dir_file_hashes}``.
    """
    basis = {
        "mechanic_class": mechanic_class,
        "source_sha256": source_sha256,
        "level_dir_file_hashes": bundle.level_dir_file_hashes,
    }
    return "solver_" + _sha256_digest(basis)[:24]


# ---------------------------------------------------------------------------
# Gate-pin bootstrap and check
# ---------------------------------------------------------------------------

def _current_gate_script_hashes() -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in GATE_SCRIPT_KEYS.items()}


def _load_gate_pin() -> dict[str, Any]:
    if GATE_PIN_PATH.is_file():
        try:
            return json.loads(GATE_PIN_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _bootstrap_gate_pin() -> dict[str, Any]:
    """Write gate_pin.json from the current worktree state (first-time bootstrap only).

    After bootstrap any gate-script change requires a human-reviewed
    gate-change record; the registry will refuse admission otherwise.
    """
    pin = {
        "schema": "codexlab.contra_loop.gate_pin.v1",
        "created_at_utc": _utc_now(),
        "note": (
            "SHA-256 allowlist for gate scripts. "
            "Changing any of these files requires a human-reviewed gate-change record "
            "that updates this file before the registry will accept a new admission."
        ),
        "provenance_notes": [
            "sp80_verify_episode_run.py was fixed twice (verifier-contract bug, "
            "pre-existing episode-verifier failure); both fixes are in the current "
            "worktree as of HEAD 2c155aab.",
        ],
        "hashes": _current_gate_script_hashes(),
    }
    GATE_PIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GATE_PIN_PATH.write_text(json.dumps(pin, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return pin


def _check_gate_pin(bundle_gate_hashes: dict[str, str]) -> tuple[bool, str]:
    """Verify bundle_gate_hashes match gate_pin.json.

    Returns (ok, reason).
    """
    pin = _load_gate_pin()
    if not pin:
        # Not bootstrapped yet -- this is a first-admit situation; caller bootstraps.
        return True, "gate_pin not yet bootstrapped (will be created on first admit)"
    pinned = pin.get("hashes", {})
    mismatches = []
    for key in GATE_SCRIPT_KEYS:
        pinned_sha = pinned.get(key, "")
        bundle_sha = bundle_gate_hashes.get(key, "")
        if not pinned_sha:
            mismatches.append(f"{key}: not in gate_pin.json")
        elif bundle_sha != pinned_sha:
            mismatches.append(
                f"{key}: bundle_sha={bundle_sha[:12]}... != pinned_sha={pinned_sha[:12]}..."
            )
    if mismatches:
        return False, "gate-script SHA mismatch (gate-change requires human review): " + "; ".join(mismatches)
    return True, "gate_pin match"


# ---------------------------------------------------------------------------
# Source-SHA pin check
# ---------------------------------------------------------------------------

def _check_source_pin(bundle: ArtifactBundle) -> tuple[bool, str]:
    """Verify runner/harness/report SHAs in the bundle match current worktree files.

    sp80_run_contra_levels.py records these at lines 167-171; the registry re-checks
    them here to block 'edit the harness to fake a result, then run'.
    """
    checks = [
        ("runner", CODEXLAB / "sp80_run_contra_levels.py", bundle.runner_source_sha256),
        ("harness", CODEXLAB / "sp80_player_harness.py", bundle.harness_source_sha256),
        ("report", CODEXLAB / "gameplay_report_harness.py", bundle.report_source_sha256),
    ]
    mismatches = []
    for label, path, expected in checks:
        if not expected:
            # Missing SHA in bundle -- skip (gate.py may not have filled it yet on genesis)
            continue
        actual = _sha256_file(path)
        if actual != expected:
            mismatches.append(
                f"{label}: bundle={expected[:12]}... != current={actual[:12]}..."
            )
    if mismatches:
        return False, "source-SHA pin mismatch (harness was edited after the run?): " + "; ".join(mismatches)
    return True, "source pin match"


# ---------------------------------------------------------------------------
# Summary.jsonl reader (reads THE solve-evidence field, never status strings)
# ---------------------------------------------------------------------------

def _read_episode_root(run_dir_rel: str) -> Path:
    """Resolve run_dir (relative to ROOT) to an absolute Path."""
    return (ROOT / run_dir_rel).resolve()


def _read_summary_rows(run_dir_rel: str) -> list[dict[str, Any]]:
    """Read per-level rows from summary.jsonl in the episode root."""
    ep_root = _read_episode_root(run_dir_rel)
    for candidate in (ep_root / "summary.jsonl", ep_root.parent / "summary.jsonl"):
        if candidate.is_file():
            return _read_jsonl(candidate)
    return []


def _read_summary_json(run_dir_rel: str) -> dict[str, Any]:
    """Read summary.json from the episode root."""
    ep_root = _read_episode_root(run_dir_rel)
    for candidate in (ep_root / "summary.json", ep_root.parent / "summary.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _target_row_delta(run_dir_rel: str, level_index: int) -> int:
    """Return levels_completed_delta for a specific level from summary.jsonl.

    Returns 0 if the level row is not found or the field is absent/zero.
    This is the SOLE admission-evidence reader.  status:nominal and
    status:passed are deliberately IGNORED here.
    """
    rows = _read_summary_rows(run_dir_rel)
    for row in rows:
        if int(row.get("level_index", -1)) == level_index:
            return int(row.get("levels_completed_delta") or 0)
    return 0


# ---------------------------------------------------------------------------
# reverify helper -- subprocess verify_run on stored run_dir
# ---------------------------------------------------------------------------

def _reverify_level(run_dir_rel: str, level_index: int) -> dict[str, Any]:
    """Run sp80_verify_episode_run.py on the stored level dir via subprocess.

    Returns {"returncode": int, "stdout": str, "stderr": str}.
    """
    level_dir = _read_episode_root(run_dir_rel) / f"level_{level_index:02d}"
    # Prefer the level subdir that was stored in verification_artifact.run_dir
    # (which points to the level subdir, e.g. run/level_00).
    if not level_dir.exists():
        # Maybe run_dir_rel already points at the level dir (legacy records)
        level_dir = _read_episode_root(run_dir_rel)
    cmd = [
        sys.executable, "-B",
        str(CODEXLAB / "sp80_verify_episode_run.py"),
        "--run-dir", str(level_dir),
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, text=True, capture_output=True, check=False,
        )
        return {"returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr[:800]}
    except OSError as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}


# ---------------------------------------------------------------------------
# Public registration surface
# ---------------------------------------------------------------------------

def register_trusted_run_dir(run_dir: str) -> None:
    """Called by gate.py after it has produced a run_dir in this process.

    admit() will refuse any run_dir NOT in this set, blocking externally-supplied
    forged run_dirs from ever being admitted.
    """
    _TRUSTED_RUN_DIRS.add(run_dir)


# ---------------------------------------------------------------------------
# SolverRegistry
# ---------------------------------------------------------------------------

class SolverRegistry:
    """Append-only, content-addressed, re-verified solver ledger.

    This is the ONLY object that writes to solvers.jsonl, rejections.jsonl, and
    hypotheses.jsonl.  The agent writes CODE; this class writes admission truth.
    """

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def all_solvers(self) -> list[dict[str, Any]]:
        """Return all admitted solver records (immutable ledger view)."""
        return _read_jsonl(SOLVERS_PATH)

    def all_rejections(self) -> list[dict[str, Any]]:
        """Return all rejection records."""
        return _read_jsonl(REJECTIONS_PATH)

    def all_hypotheses(self) -> list[dict[str, Any]]:
        """Return all discovery-stage hypothesis records."""
        return _read_jsonl(HYPOTHESES_PATH)

    def get_solver(self, solver_id: str) -> dict[str, Any] | None:
        """Return the solver record with this solver_id, or None."""
        for record in self.all_solvers():
            if record.get("solver_id") == solver_id:
                return record
        return None

    # ------------------------------------------------------------------
    # baseline_levels  (the regression target for every new run)
    # ------------------------------------------------------------------

    def baseline_levels(self, game_id: str | None = None) -> list[int]:
        """Return the sorted list of level_indices that every run must keep passing.

        These are the levels covered by admitted solvers (deduped by level_index).
        A cold registry returns [] -- no regression baseline to defend yet.
        gate.py passes ``registry.baseline_levels() | {target}`` to ``contra run``.
        """
        seen: set[int] = set()
        for record in self.all_solvers():
            # Optional game_id filter (future: multi-game registries)
            if game_id and str(record.get("game_id", "")) != str(game_id):
                continue
            va = record.get("verification_artifact") or {}
            level = va.get("target_level_index")
            if level is not None:
                seen.add(int(level))
        return sorted(seen)

    # ------------------------------------------------------------------
    # no_regression_check
    # ------------------------------------------------------------------

    def no_regression_check(
        self,
        run_dir_rel: str,
        game_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify that the run_dir covers all baseline levels with delta>0.

        Returns::

            {
              "no_regression": bool,
              "reason": str,
              "baseline": [int, ...],
              "completed_after": [int, ...],
              "missing_levels": [int, ...],
            }

        A cold registry always returns no_regression=True (no levels to defend).
        """
        baseline = self.baseline_levels(game_id)
        if not baseline:
            return {
                "no_regression": True,
                "reason": "registry baseline is empty (genesis admission)",
                "baseline": [],
                "completed_after": [],
                "missing_levels": [],
            }

        # Read the actual per-level delta from the run's summary.jsonl
        rows = _read_summary_rows(run_dir_rel)
        completed_after: list[int] = []
        for row in rows:
            if int(row.get("levels_completed_delta") or 0) > 0:
                completed_after.append(int(row.get("level_index", -1)))

        missing = [lvl for lvl in baseline if lvl not in completed_after]
        ok = len(missing) == 0
        return {
            "no_regression": ok,
            "reason": ("all baseline levels passed" if ok
                       else f"regression: levels {missing} lost delta>0"),
            "baseline": baseline,
            "completed_after": sorted(completed_after),
            "missing_levels": missing,
        }

    # ------------------------------------------------------------------
    # admit  (THE sole admission writer)
    # ------------------------------------------------------------------

    def admit(
        self,
        record: dict[str, Any],
        bundle: ArtifactBundle,
    ) -> dict[str, Any]:
        """Admit a solver IFF the gate cleared all five layers AND delta>0 AND no regression.

        The ``record`` is a SolverRecord dict (schema codexlab.contra_loop.solver_record.v1)
        that the agent or crystallize.py provides.  ``bundle`` carries the
        machine-produced verification artifact from gate.py.

        Raises ValueError with a specific Refusal reason if any admission predicate fails.
        On success, appends ONE canonical JSON line to solvers.jsonl and returns the
        admitted record dict.

        ADMISSION PREDICATES (all must hold simultaneously):
          1. bundle.levels_completed_delta > 0  (THE only solve evidence)
          2. bundle.all_five_layers_passed = True
          3. bundle.no_regression = True  (completed_levels_after ⊇ baseline)
          4. run_dir in _TRUSTED_RUN_DIRS  (self-produced guard)
          5. source-SHA pin: runner/harness/report SHAs match worktree files
          6. gate-script pin: bundle.gate_script_hashes match gate_pin.json
          7. No duplicate lineage_key already in solvers.jsonl
        """
        # --- 1. Bundle structural validity ---
        ok, reason = bundle.validate()
        if not ok:
            return self._refusal(f"bundle invalid: {reason}", record, bundle)

        # --- 2. Self-produced guard ---
        if bundle.run_dir not in _TRUSTED_RUN_DIRS:
            return self._refusal(
                f"run_dir {bundle.run_dir!r} was not produced by this gate invocation; "
                "externally-supplied run_dirs cannot be admitted",
                record, bundle,
            )

        # --- 3. Source-SHA pin ---
        ok, reason = _check_source_pin(bundle)
        if not ok:
            return self._refusal(reason, record, bundle)

        # --- 4. Gate-script pin (bootstrap if first admit) ---
        pin = _load_gate_pin()
        if not pin:
            pin = _bootstrap_gate_pin()
        if bundle.gate_script_hashes:
            ok, reason = _check_gate_pin(bundle.gate_script_hashes)
            if not ok:
                return self._refusal(reason, record, bundle)

        # --- 5. Re-verify delta from summary.jsonl (must be >0; ignores status strings) ---
        confirmed_delta = _target_row_delta(bundle.run_dir, bundle.target_level_index)
        if confirmed_delta <= 0:
            return self._refusal(
                f"summary.jsonl row for level {bundle.target_level_index} shows "
                f"levels_completed_delta={confirmed_delta}; "
                "note: status:nominal and status:passed are NOT solve evidence",
                record, bundle,
            )

        # --- 6. No-regression double-check against live ledger ---
        reg_check = self.no_regression_check(
            bundle.run_dir, game_id=record.get("game_id")
        )
        if not reg_check["no_regression"]:
            return self._refusal(reg_check["reason"], record, bundle)

        # --- 7. Dedup by lineage_key ---
        dedup_key = str(record.get("dedup_key") or record.get("solver_id") or "")
        if dedup_key:
            for existing in self.all_solvers():
                if str(existing.get("dedup_key") or "") == dedup_key:
                    return self._refusal(
                        f"duplicate lineage_key {dedup_key!r} already in solvers.jsonl; "
                        "supersedes requires explicit supersedes field",
                        record, bundle,
                    )

        # --- Build final SolverRecord ---
        mechanic_class = str(record.get("mechanic_class") or "unknown")
        source_sha = str(
            (record.get("solver_impl") or {}).get("source_sha256", "")
            or bundle.harness_source_sha256
        )
        sid = _solver_id(bundle, mechanic_class, source_sha)

        admitted_record: dict[str, Any] = {
            "schema": SOLVER_RECORD_SCHEMA,
            "solver_id": sid,
            "created_at_utc": _utc_now(),
            "status": "admitted",
            "game_id": record.get("game_id", ""),
            "mechanic_class": mechanic_class,
            "level_index": bundle.target_level_index,
            "dedup_key": dedup_key or sid,
            "supersedes": record.get("supersedes"),
            # Carry agent-provided fields through unchanged
            "hypothesis": record.get("hypothesis"),
            "solver_impl": record.get("solver_impl"),
            "forward_model_ref": record.get("forward_model_ref"),
            "evidence_tier": int(record.get("evidence_tier") or 0),
            "wiring": record.get("wiring"),
            "keystone_artifact": record.get("keystone_artifact"),
            # The verification artifact -- machine-produced, not agent-writable
            "verification_artifact": {
                **bundle.as_dict(),
                # Re-stamp the confirmed delta from the ledger (not bundle's cached value)
                "levels_completed_delta": confirmed_delta,
            },
            "provenance_chain": record.get("provenance_chain"),
            "confirmed_relation_memory": record.get("confirmed_relation_memory"),
            # confidence_score inherits ghost's exactly-two-values rule:
            # 1.0 = completion proof; 0.5 = keystone/deltaband only.
            # The registry MUST NOT elevate 0.5->1.0 without a new completion proof.
            "confidence_score": 1.0 if confirmed_delta > 0 else 0.5,
            "truth_boundary": {
                "not_prediction": True,
                "not_control_truth": True,
                "not_solve_truth": False,        # False ONLY because delta>0
                "solve_condition_admitted": True,
                "admission_status": "admitted",
                "admission_evidence": "levels_completed_delta>0 from summary.jsonl",
            },
        }

        _append_jsonl(SOLVERS_PATH, admitted_record)
        return {
            "admitted": True,
            "solver_id": sid,
            "mechanic_class": mechanic_class,
            "target_level_index": bundle.target_level_index,
            "levels_completed_delta": confirmed_delta,
        }

    # ------------------------------------------------------------------
    # reject  (honest negative -- so dead hypotheses are never re-attempted)
    # ------------------------------------------------------------------

    def reject(
        self,
        hypothesis_or_solver: dict[str, Any],
        *,
        refusal_reason: str,
        keystone_metric: str | None = None,
        failed_gate_layer: str | None = None,
        bundle: ArtifactBundle | None = None,
    ) -> dict[str, Any]:
        """Record an honest negative in rejections.jsonl.

        A gate-failed attempt is never written to solvers.jsonl.  Recording the
        negative means iteration N+1 does not re-derive the same dead end.
        """
        rejection: dict[str, Any] = {
            "schema": REJECTION_SCHEMA,
            "created_at_utc": _utc_now(),
            "rejection_id": "rejection_" + _sha256_digest({
                "refusal_reason": refusal_reason,
                "ts": _utc_now(),
            })[:16],
            "status": "rejected",
            "refusal_reason": refusal_reason,
            "failed_gate_layer": failed_gate_layer,
            "keystone_metric": keystone_metric,
            "hypothesis_name": hypothesis_or_solver.get("name") or hypothesis_or_solver.get("solver_id"),
            "mechanic_class": hypothesis_or_solver.get("mechanic_class"),
            "game_id": hypothesis_or_solver.get("game_id"),
            "level_index": hypothesis_or_solver.get("level_index"),
            "boundary": "Rejection record. Not solve evidence. dead_hypothesis=True.",
            "bundle_summary": (
                {
                    "run_dir": bundle.run_dir,
                    "levels_completed_delta": bundle.levels_completed_delta,
                    "all_five_layers_passed": bundle.all_five_layers_passed,
                }
                if bundle else None
            ),
        }
        _append_jsonl(REJECTIONS_PATH, rejection)
        return {"rejected": True, "rejection_id": rejection["rejection_id"], "reason": refusal_reason}

    # ------------------------------------------------------------------
    # add_hypothesis  (discovery-stage; admission_status=discovery_only)
    # ------------------------------------------------------------------

    def add_hypothesis(self, hypothesis: dict[str, Any]) -> dict[str, Any]:
        """Store a discovery-stage hypothesis with admission_status=discovery_only.

        A passing keystone => discovery_only, NEVER a solve.
        The hypothesis stays here until a completed episode promotes it via admit().
        """
        # Minimal validation: the four required fields of the boundary envelope
        required = ("action_primitive", "precondition_signature", "predicted_effect", "falsification_condition")
        missing = [f for f in required if not hypothesis.get(f)]
        if missing:
            raise ValueError(
                f"hypothesis is under-specified: missing {missing}; "
                "schema.validate() rejects vague specs (cardinal rule: "
                "a hypothesis without falsification_condition cannot be proven wrong)"
            )

        record: dict[str, Any] = {
            "schema": HYPOTHESIS_SCHEMA,
            "created_at_utc": _utc_now(),
            "hypothesis_id": "hyp_" + _sha256_digest(hypothesis)[:16],
            "admission_status": "discovery_only",
            "solve_condition_admitted": False,
            "boundary": hypothesis.get("boundary") or "Hypothesis material only; not local CONTRA evidence.",
            **hypothesis,
        }
        # Carry boundary verbatim so it can never be parsed as solve evidence
        _append_jsonl(HYPOTHESES_PATH, record)
        return {"hypothesis_id": record["hypothesis_id"], "admission_status": "discovery_only"}

    # ------------------------------------------------------------------
    # reverify  (re-hash artifacts + re-run verifier; drift detection)
    # ------------------------------------------------------------------

    def reverify(self, solver_id: str) -> dict[str, Any]:
        """Re-verify a stored solver record against its artifact on disk.

        Steps:
          1. Load the SolverRecord from solvers.jsonl.
          2. Re-hash ``result.json``, ``packets.jsonl``, ``brain_calls.jsonl``
             in the stored level_dir and compare to verification_artifact.level_dir_file_hashes.
          3. Re-run ``sp80_verify_episode_run.py --run-dir <level_dir>`` via subprocess.
          4. Re-check levels_completed_delta > 0 from result.json.

        Returns a dict with ``reverify_passed``, ``artifact_drift``, and ``reason``.
        A drift or verifier failure means the stored solver is no longer trustworthy.
        """
        record = self.get_solver(solver_id)
        if record is None:
            return {
                "solver_id": solver_id,
                "reverify_passed": False,
                "artifact_drift": False,
                "reason": f"solver_id {solver_id!r} not found in solvers.jsonl",
            }

        va = record.get("verification_artifact") or {}
        run_dir_rel = va.get("run_dir") or ""
        level_index = int(va.get("target_level_index") or 0)
        stored_hashes = va.get("level_dir_file_hashes") or {}

        if not run_dir_rel:
            return {
                "solver_id": solver_id,
                "reverify_passed": False,
                "artifact_drift": True,
                "reason": "verification_artifact.run_dir is missing",
            }

        # Step 2: re-hash level artifact files
        ep_root = _read_episode_root(run_dir_rel)
        level_dir = ep_root / f"level_{level_index:02d}"
        if not level_dir.exists():
            level_dir = ep_root  # legacy: run_dir pointed at level dir directly

        drift_files: list[str] = []
        for fname in LEVEL_ARTIFACT_FILES:
            fpath = level_dir / fname
            current_sha = _sha256_file(fpath)
            stored_sha = stored_hashes.get(fname, "")
            if stored_sha and current_sha != stored_sha:
                drift_files.append(
                    f"{fname}: stored={stored_sha[:12]}... current={current_sha[:12]}..."
                )
            elif not stored_sha:
                drift_files.append(f"{fname}: not in stored hashes (cannot verify)")

        artifact_drift = len(drift_files) > 0

        # Step 3: re-run verifier via subprocess
        verify_result = _reverify_level(run_dir_rel, level_index)
        verifier_passed = verify_result["returncode"] == 0

        # Step 4: re-check delta from result.json
        result_path = level_dir / "result.json"
        live_delta = 0
        try:
            result_json = json.loads(result_path.read_text(encoding="utf-8"))
            live_delta = int(result_json.get("levels_completed_delta") or 0)
        except (OSError, json.JSONDecodeError, ValueError):
            live_delta = 0

        delta_ok = live_delta > 0
        reverify_passed = (not artifact_drift) and verifier_passed and delta_ok

        reasons: list[str] = []
        if artifact_drift:
            reasons.append(f"artifact_drift in {drift_files}")
        if not verifier_passed:
            reasons.append(f"verifier rc={verify_result['returncode']}: {verify_result['stderr'][:200]}")
        if not delta_ok:
            reasons.append(f"live result.json levels_completed_delta={live_delta} (must be >0)")

        return {
            "solver_id": solver_id,
            "reverify_passed": reverify_passed,
            "artifact_drift": artifact_drift,
            "drift_files": drift_files,
            "verifier_returncode": verify_result["returncode"],
            "live_delta": live_delta,
            "reason": "; ".join(reasons) if reasons else "all checks passed",
        }

    # ------------------------------------------------------------------
    # Internal refusal helper
    # ------------------------------------------------------------------

    def _refusal(
        self,
        reason: str,
        record: dict[str, Any],
        bundle: ArtifactBundle,
    ) -> dict[str, Any]:
        """Record a failed admission attempt and raise ValueError."""
        self.reject(
            record,
            refusal_reason=reason,
            bundle=bundle,
        )
        raise ValueError(f"ADMISSION REFUSED: {reason}")


# ---------------------------------------------------------------------------
# Module-level convenience API (mirrors the class, operates on a singleton)
# ---------------------------------------------------------------------------

_registry = SolverRegistry()


def admit(record: dict[str, Any], bundle: ArtifactBundle) -> dict[str, Any]:
    """Convenience wrapper -- see SolverRegistry.admit()."""
    return _registry.admit(record, bundle)


def reject(
    record: dict[str, Any],
    *,
    refusal_reason: str,
    keystone_metric: str | None = None,
    failed_gate_layer: str | None = None,
    bundle: ArtifactBundle | None = None,
) -> dict[str, Any]:
    """Convenience wrapper -- see SolverRegistry.reject()."""
    return _registry.reject(
        record,
        refusal_reason=refusal_reason,
        keystone_metric=keystone_metric,
        failed_gate_layer=failed_gate_layer,
        bundle=bundle,
    )


def add_hypothesis(hypothesis: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper -- see SolverRegistry.add_hypothesis()."""
    return _registry.add_hypothesis(hypothesis)


def reverify(solver_id: str) -> dict[str, Any]:
    """Convenience wrapper -- see SolverRegistry.reverify()."""
    return _registry.reverify(solver_id)


def baseline_levels(game_id: str | None = None) -> list[int]:
    """Return the sorted level indices that every regression run must still pass."""
    return _registry.baseline_levels(game_id)


def no_regression_check(run_dir_rel: str, game_id: str | None = None) -> dict[str, Any]:
    """Check that run_dir covers all admitted baseline levels with delta>0."""
    return _registry.no_regression_check(run_dir_rel, game_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_list() -> int:
    solvers = _registry.all_solvers()
    rejections = _registry.all_rejections()
    hypotheses = _registry.all_hypotheses()
    print(json.dumps({
        "schema": SOLVER_RECORD_SCHEMA,
        "solver_count": len(solvers),
        "rejection_count": len(rejections),
        "hypothesis_count": len(hypotheses),
        "baseline_levels": baseline_levels(),
        "solvers": [
            {
                "solver_id": s.get("solver_id"),
                "mechanic_class": s.get("mechanic_class"),
                "game_id": s.get("game_id"),
                "level_index": s.get("level_index"),
                "confidence_score": s.get("confidence_score"),
                "run_dir": (s.get("verification_artifact") or {}).get("run_dir"),
                "delta": (s.get("verification_artifact") or {}).get("levels_completed_delta"),
            }
            for s in solvers
        ],
        "rejections": [
            {
                "rejection_id": r.get("rejection_id"),
                "mechanic_class": r.get("mechanic_class"),
                "reason": r.get("refusal_reason"),
            }
            for r in rejections
        ],
    }, indent=2))
    return 0


def _cmd_reverify_all() -> int:
    solvers = _registry.all_solvers()
    if not solvers:
        print(json.dumps({"status": "no_solvers", "message": "registry is empty"}))
        return 0
    results = []
    any_failed = False
    for s in solvers:
        sid = s.get("solver_id", "")
        result = reverify(sid)
        results.append(result)
        if not result.get("reverify_passed"):
            any_failed = True
    print(json.dumps({"reverify_results": results, "all_passed": not any_failed}, indent=2))
    return 1 if any_failed else 0


def _cmd_baseline_levels() -> int:
    levels = baseline_levels()
    print(json.dumps({"baseline_levels": levels, "count": len(levels)}, indent=2))
    return 0


def _cmd_dry_run() -> int:
    print(json.dumps({
        "mode": "dry_run",
        "registry_dir": str(REGISTRY_DIR),
        "solvers_path": str(SOLVERS_PATH),
        "rejections_path": str(REJECTIONS_PATH),
        "hypotheses_path": str(HYPOTHESES_PATH),
        "gate_pin_path": str(GATE_PIN_PATH),
        "gate_pin_exists": GATE_PIN_PATH.is_file(),
        "solver_count": len(_registry.all_solvers()),
        "rejection_count": len(_registry.all_rejections()),
        "hypothesis_count": len(_registry.all_hypotheses()),
        "baseline_levels": baseline_levels(),
        "trusted_run_dirs_in_session": list(_TRUSTED_RUN_DIRS),
        "admission_predicates": [
            "levels_completed_delta > 0 from summary.jsonl (NEVER from status strings)",
            "all_five_layers_passed = True",
            "no_regression: completed_levels_after ⊇ baseline",
            "run_dir in _TRUSTED_RUN_DIRS (self-produced guard)",
            "source-SHA pin: runner/harness/report SHAs match worktree files",
            "gate-script pin: bundle.gate_script_hashes match gate_pin.json",
            "no duplicate lineage_key in solvers.jsonl",
        ],
        "boundary": (
            "SolverRegistry dry-run. No files written. "
            "The only solve evidence is levels_completed_delta>0 from summary.jsonl rows."
        ),
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--list", action="store_true", help="list admitted solvers + rejections + hypotheses")
    p.add_argument("--reverify-all", action="store_true", help="re-hash + re-verify every admitted solver")
    p.add_argument("--baseline-levels", action="store_true", help="print the levels every run must still pass")
    p.add_argument("--dry-run", action="store_true", help="print registry state without writing anything")
    args = p.parse_args()

    if args.reverify_all:
        return _cmd_reverify_all()
    if args.baseline_levels:
        return _cmd_baseline_levels()
    if args.dry_run:
        return _cmd_dry_run()
    # default and --list
    return _cmd_list()


if __name__ == "__main__":
    raise SystemExit(main())
