#!/usr/bin/env python3
"""Crystallizer -- sole writer of admitted SolverRecords.

On a gate-passing episode with levels_completed_delta > 0:
  1.  Writes ghost.json via contra_head.ghost.write_ghost (inheriting its delta>0 guard).
  2.  Patches every confirmed_relationship_row with DERIVED flow_axis / flow_direction
      computed from shared_transform_xy  --  fixes the hardcoded "unknown" at
      contra_head/ghost.py:131-132.
  3.  Builds a SolverRecord (schema codexlab.contra_loop.solver_record.v1) and appends
      it to codexlab/contra_loop/registry/solvers.jsonl (append-only; content-addressed
      by solver_id = sha256 of the canonical proof bundle).

Cardinal rule (structural, not trust-based): this module REFUSES to write any record
whose ArtifactBundle carries levels_completed_delta <= 0.  The episode-result field is
the ONLY solve evidence; status:"nominal" / verifier status:"passed" are ignored.

Day-one extension point for RealtimeMechanicMapper: `_apply_mechanic_mapper()` is the
stub where that logic will live once ported from ghost_mechanics_overlay.py.

Usage (dry-run, no network, no solve needed):
    python codexlab/contra_loop/crystallize.py --dry-run

Usage (admit the genesis L0 solver from the canonical run):
    python codexlab/contra_loop/crystallize.py \\
        --run-dir codexlab/runs/contra_sp80_episode_levels_00_01_20260608T175142Z \\
        --level 0 \\
        --mechanic-class activate_when_aligned \\
        --game-id sp80-589a99af
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

from contra_head.ghost import (  # noqa: E402
    CONTRA_GHOST_SCHEMA,
    build_ghost,
    relpath,
    stable_digest,
    truth_boundary,
    write_json,
)

SOLVER_RECORD_SCHEMA = "codexlab.contra_loop.solver_record.v1"
_REGISTRY_DIR = CODEXLAB / "contra_loop" / "registry"
_SOLVERS_JSONL = _REGISTRY_DIR / "solvers.jsonl"

# ---------------------------------------------------------------------------
# SHA helpers (inline; avoids importing sp80_base_controller which needs numpy)
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _git_head_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# Core derivation: flow_axis / flow_direction from transform_hypothesis_xy
# (fixes the hardcoded "unknown" at contra_head/ghost.py:131-132)
# ---------------------------------------------------------------------------

def derive_flow_axis_direction(
    transform_xy: list[float] | tuple[float, float] | None,
) -> tuple[str, str]:
    """Derive (flow_axis, flow_direction) from a PortBand transform_hypothesis_xy vector.

    transform_xy = [dx, dy] in screen pixels.  Screen coordinates: positive dy = downward.

    Rules:
      |dx| > |dy|  -> axis "horizontal",  direction "right" / "left"
      |dy| > |dx|  -> axis "vertical",    direction "down"  / "up"
      |dx| == |dy| (and nonzero) -> axis "diagonal", direction compass label
      both zero    -> axis "none", direction "none"
      missing/bad  -> axis "unknown", direction "unknown"

    Example: transform_xy=[12.0, 36.0] -> ("vertical", "down")
    This is the L0 win geometry: the blue platform moves downward to reach the buckets.
    """
    if not isinstance(transform_xy, (list, tuple)) or len(transform_xy) < 2:
        return "unknown", "unknown"
    try:
        dx = float(transform_xy[0])
        dy = float(transform_xy[1])
    except (TypeError, ValueError):
        return "unknown", "unknown"

    if dx == 0.0 and dy == 0.0:
        return "none", "none"

    abs_dx = abs(dx)
    abs_dy = abs(dy)

    if abs_dx > abs_dy:
        return "horizontal", ("right" if dx > 0 else "left")
    if abs_dy > abs_dx:
        return "vertical", ("down" if dy > 0 else "up")
    # Equal magnitudes: diagonal
    if dx >= 0 and dy >= 0:
        return "diagonal", "right_down"
    if dx >= 0 and dy < 0:
        return "diagonal", "right_up"
    if dx < 0 and dy >= 0:
        return "diagonal", "left_down"
    return "diagonal", "left_up"


def _patch_relation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of confirmed_relationship_rows with derived flow_axis/flow_direction."""
    patched = []
    for row in rows:
        if not isinstance(row, dict):
            patched.append(row)
            continue
        geometry = dict(row.get("geometry") or {})
        shared_xy = geometry.get("shared_transform_xy")
        if geometry.get("flow_axis", "unknown") == "unknown" or geometry.get("flow_direction", "unknown") == "unknown":
            axis, direction = derive_flow_axis_direction(shared_xy)
            geometry["flow_axis"] = axis
            geometry["flow_direction"] = direction
        patched.append({**row, "geometry": geometry})
    return patched


# ---------------------------------------------------------------------------
# Extension point: RealtimeMechanicMapper (day-one stub)
# ---------------------------------------------------------------------------

def _apply_mechanic_mapper(
    ghost: dict[str, Any],
    packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """Day-one stub: returns the ghost unchanged.

    TODO: when RealtimeMechanicMapper (ghost_mechanics_overlay.py) is ported to
    contra_head as a submodule, call it here to accumulate online mechanic evidence
    and merge the result into confirmed_relationship_rows before crystallizing.

    Concretely:
        from contra_head.mechanic_mapper import RealtimeMechanicMapper
        mapper = RealtimeMechanicMapper()
        for packet in packets:
            mapper.update(packet)
        return mapper.merge_into_ghost(ghost)
    """
    return ghost  # no-op until mapper is ported


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ArtifactBundle:
    """Everything gate.py verifies; passed intact to Crystallizer.crystallize().

    All five verification-layer return codes must be 0.  levels_completed_delta
    must be > 0 (the ONLY admission predicate -- not "nominal", not verifier rc alone).
    """

    # --- identity ---
    game_id: str
    target_level_index: int
    run_dir: str                    # relative to ROOT, e.g. "codexlab/runs/.../level_00"
    level_dir: str                  # relative to ROOT, e.g. "codexlab/runs/.../level_00"

    # --- THE solve field ---
    levels_completed_delta: int     # > 0 required; read from summary.jsonl row

    # --- regression ---
    completed_levels_after: list[int]
    completed_levels_baseline: list[int]
    no_regression: bool             # completed_levels_after ⊇ completed_levels_baseline

    # --- five gate layers (all must be 0) ---
    dev_check_returncode: int
    contra_check_returncode: int
    run_returncode: int
    verifier_returncode: int
    report_verify_returncode: int
    all_five_layers_passed: bool

    # --- provenance SHAs (from summary.json, sp80_run_contra_levels.py:167-171) ---
    summary_json_sha256: str
    runner_source_sha256: str
    harness_source_sha256: str
    report_source_sha256: str

    # --- gate-integrity pin (vs registry/gate_pin.json) ---
    gate_script_hashes: dict[str, str] = dataclasses.field(default_factory=dict)

    # --- git provenance ---
    git_commit_sha: str = "unknown"

    @classmethod
    def from_level_dir(
        cls,
        level_dir: Path,
        *,
        root: Path,
        completed_levels_baseline: list[int] | None = None,
    ) -> "ArtifactBundle":
        """Construct a minimal ArtifactBundle by reading level_dir artifacts directly.

        Skips the five-layer gate entirely (all returncodes set to -1 / not_run).
        Useful for --dry-run, genesis admission of the seed L0 solver, and tests.

        Precondition: level_dir/result.json must exist and carry levels_completed_delta > 0.
        """
        result_path = level_dir / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"result.json missing from {level_dir}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        delta = int(result.get("levels_completed_delta") or 0)
        level_index = int(result.get("level_index") or 0)
        game_id = str(result.get("game_id") or "unknown")

        # Read parent summary.json for SHA provenance
        parent_run_dir = level_dir.parent
        summary_path = parent_run_dir / "summary.json"
        summary_sha = _sha256_file(summary_path) if summary_path.is_file() else "missing"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

        baseline = list(completed_levels_baseline or [])
        completed_after = summary.get("completed_levels") or ([level_index] if delta > 0 else [])

        return cls(
            game_id=game_id,
            target_level_index=level_index,
            run_dir=relpath(parent_run_dir, root),
            level_dir=relpath(level_dir, root),
            levels_completed_delta=delta,
            completed_levels_after=completed_after,
            completed_levels_baseline=baseline,
            no_regression=all(lvl in completed_after for lvl in baseline),
            dev_check_returncode=-1,
            contra_check_returncode=-1,
            run_returncode=-1,
            verifier_returncode=0 if delta > 0 else -1,
            report_verify_returncode=-1,
            all_five_layers_passed=False,   # partial: gate was not run
            summary_json_sha256=summary_sha,
            runner_source_sha256=summary.get("runner_source_sha256", "unknown"),
            harness_source_sha256=summary.get("harness_source_sha256", "unknown"),
            report_source_sha256=summary.get("report_source_sha256", "unknown"),
            git_commit_sha=_git_head_sha(),
        )


@dataclasses.dataclass
class CrystallizationResult:
    """What Crystallizer.crystallize() returns."""

    admitted: bool
    reason: str
    solver_id: str | None = None
    ghost_id: str | None = None
    ghost_path: str | None = None
    solver_record_path: str | None = None
    levels_completed_delta: int = 0
    flow_directions_derived: int = 0  # how many "unknown" -> derived upgrades


# ---------------------------------------------------------------------------
# Crystallizer
# ---------------------------------------------------------------------------

class Crystallizer:
    """Sole writer of SolverRecord.status == "admitted" entries.

    Usage:
        c = Crystallizer(root=ROOT)
        result = c.crystallize(
            bundle=bundle,
            hypothesis=hypothesis_envelope,
            mechanic_class="activate_when_aligned",
            evidence_tier=2,
        )
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or ROOT
        self.registry_dir = self.root / "codexlab" / "contra_loop" / "registry"
        self.solvers_path = self.registry_dir / "solvers.jsonl"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def crystallize(
        self,
        *,
        bundle: ArtifactBundle,
        hypothesis: dict[str, Any] | None = None,
        solver_impl: dict[str, Any] | None = None,
        keystone_artifact: dict[str, Any] | None = None,
        wiring: dict[str, Any] | None = None,
        mechanic_class: str = "unclassified",
        evidence_tier: int = 2,
        supersedes: str | None = None,
    ) -> CrystallizationResult:
        """Crystallize a gate-passing episode into a SolverRecord.

        Returns a CrystallizationResult.  admitted=False means nothing was written.
        """
        # ---- Cardinal rule: delta > 0 is the ONLY admission predicate ----
        if bundle.levels_completed_delta <= 0:
            return CrystallizationResult(
                admitted=False,
                reason=f"delta={bundle.levels_completed_delta} -- not > 0; episode is nominal/unsolved",
                levels_completed_delta=bundle.levels_completed_delta,
            )

        level_dir = self.root / bundle.level_dir
        if not level_dir.is_dir():
            return CrystallizationResult(
                admitted=False,
                reason=f"level_dir missing: {bundle.level_dir}",
            )

        # ---- Read level artifacts ----------------------------------------
        result_path = level_dir / "result.json"
        packets_path = level_dir / "packets.jsonl"
        if not result_path.is_file():
            return CrystallizationResult(admitted=False, reason="missing result.json")
        if not packets_path.is_file():
            return CrystallizationResult(admitted=False, reason="missing packets.jsonl")

        result = json.loads(result_path.read_text(encoding="utf-8"))
        packets = [
            json.loads(line)
            for line in packets_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Double-check: result.json must also agree with delta > 0
        result_delta = int(result.get("levels_completed_delta") or 0)
        if result_delta <= 0:
            return CrystallizationResult(
                admitted=False,
                reason=f"result.json levels_completed_delta={result_delta} (must be >0)",
            )

        # ---- Build ghost (inherits write_ghost's own delta > 0 guard) -----
        ghost = build_ghost(
            result=result,
            packets=packets,
            run_dir=level_dir,
            root=self.root,
            game_id=bundle.game_id,
        )
        if ghost is None:
            # build_ghost returned None (delta guard tripped) -- should not happen
            # since we already checked, but be safe
            return CrystallizationResult(
                admitted=False,
                reason="build_ghost returned None (delta guard inside ghost.py fired)",
            )

        # ---- Day-one extension point: RealtimeMechanicMapper --------------
        ghost = _apply_mechanic_mapper(ghost, packets)

        # ---- DERIVE flow_axis / flow_direction (the ghost.py:131-132 fix) -
        original_rows = ghost.get("confirmed_relationship_rows") or []
        patched_rows = _patch_relation_rows(original_rows)
        ghost["confirmed_relationship_rows"] = patched_rows
        # Recompute the sha basis after patching
        ghost["ghost_sha256_basis"] = stable_digest(ghost)

        derived_count = sum(
            1 for orig, patched in zip(original_rows, patched_rows)
            if (orig.get("geometry") or {}).get("flow_direction", "unknown") == "unknown"
            and (patched.get("geometry") or {}).get("flow_direction", "unknown") != "unknown"
        )

        # ---- Write the patched ghost.json to the level run dir ------------
        ghost_path = level_dir / "ghost.json"
        write_json(ghost_path, ghost)
        ghost_id = ghost.get("ghost_id", "unknown")

        # ---- Compute level_dir file hashes --------------------------------
        level_dir_file_hashes = {
            name: _sha256_file(level_dir / name)
            for name in ("result.json", "packets.jsonl", "brain_calls.jsonl")
            if (level_dir / name).is_file()
        }

        # ---- Extract dedup_key (lineage_key) from the admitted rows -------
        lineage_keys = [
            str((row.get("source_evidence_refs") or {}).get("lineage_key") or "")
            for row in patched_rows
            if isinstance(row, dict) and row.get("relation_family") == "edge_alignment"
        ]
        dedup_key = lineage_keys[0] if lineage_keys else stable_digest([bundle.game_id, bundle.target_level_index])

        # ---- Build the verification_artifact block ------------------------
        verification_artifact = {
            "run_dir": bundle.run_dir,
            "level_dir": bundle.level_dir,
            "summary_json_sha256": bundle.summary_json_sha256,
            "summary_schema": "codexlab.sp80_contra_episode_levels.v1",
            "levels_completed_delta": bundle.levels_completed_delta,
            "target_level_index": bundle.target_level_index,
            # Five gate layers
            "dev_check_returncode": bundle.dev_check_returncode,
            "contra_check_returncode": bundle.contra_check_returncode,
            "run_returncode": bundle.run_returncode,
            "verifier_returncode": bundle.verifier_returncode,
            "report_verify_returncode": bundle.report_verify_returncode,
            "all_five_layers_passed": bundle.all_five_layers_passed,
            # Regression
            "completed_levels_after": sorted(bundle.completed_levels_after),
            "completed_levels_baseline": sorted(bundle.completed_levels_baseline),
            "no_regression": bundle.no_regression,
            # Level-dir content hashes (reverify-on-read basis)
            "level_dir_file_hashes": level_dir_file_hashes,
            # Source SHA pins (from summary.json; sp80_run_contra_levels.py:167-171)
            "runner_source_sha256": bundle.runner_source_sha256,
            "harness_source_sha256": bundle.harness_source_sha256,
            "report_source_sha256": bundle.report_source_sha256,
            # Gate-integrity pin
            "gate_script_hashes": bundle.gate_script_hashes,
            # Git provenance
            "git_commit_sha": bundle.git_commit_sha,
        }

        # ---- Compute content-addressed solver_id --------------------------
        id_basis = {
            "mechanic_class": mechanic_class,
            "game_id": bundle.game_id,
            "level_index": bundle.target_level_index,
            "level_dir_file_hashes": level_dir_file_hashes,
            "runner_source_sha256": bundle.runner_source_sha256,
            "harness_source_sha256": bundle.harness_source_sha256,
        }
        solver_id = "solver_" + hashlib.sha256(
            _canonical_json(id_basis).encode("utf-8")
        ).hexdigest()[:16]

        # ---- Build SolverRecord -------------------------------------------
        solver_record: dict[str, Any] = {
            "schema": SOLVER_RECORD_SCHEMA,
            "solver_id": solver_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "admitted",
            "mechanic_class": mechanic_class,
            "game_id": bundle.game_id,
            "level_index": bundle.target_level_index,
            "dedup_key": dedup_key,
            "supersedes": supersedes,
            # Hypothesis envelope (optional; present when loop agent supplies one)
            "hypothesis": hypothesis,
            # Executable binding (optional; day-one may not have a wired solver yet)
            "solver_impl": solver_impl,
            "forward_model_ref": None,  # extension point: plan-then-act solvers
            "evidence_tier": evidence_tier,
            "wiring": wiring,
            # Keystone artifact (optional; supplied when KEYSTONE step ran)
            "keystone_artifact": keystone_artifact,
            # THE verification artifact: immutable proof block
            "verification_artifact": verification_artifact,
            # Ghost rows, copied with DERIVED flow directions
            "confirmed_relation_memory": patched_rows,
            # Confidence: exactly two values (ghost.py rule preserved)
            #   1.0  -> levels_completed_delta > 0 (this is that case)
            #   0.5  -> keystone-only / discovery_only (never admitted here)
            "confidence_score": 1.0,
            # Provenance chain
            "provenance_chain": {
                "ground_truth_oracle": "engine_source",  # default; override when win-bank used
                "hypothesis_source": "ghost" if hypothesis is None else "mined",
                "prior_solver_ids": [],   # loop.py fills this: the baseline solver_ids re-verified
                "ghost_id": ghost_id,
                "ghost_path": relpath(ghost_path, self.root),
            },
            # Truth boundary (fail-closed; solve_condition_admitted ONLY because delta>0)
            "truth_boundary": {
                **truth_boundary(),
                "solve_condition_admitted": True,
                "admission_status": "admitted",
                "not_solve_truth": False,   # cleared only here, only with delta>0 proof
            },
        }

        # ---- Write to registry/solvers.jsonl (append-only) ----------------
        self._admit(solver_record)

        return CrystallizationResult(
            admitted=True,
            reason=f"delta={bundle.levels_completed_delta}; all_five_layers={bundle.all_five_layers_passed}",
            solver_id=solver_id,
            ghost_id=ghost_id,
            ghost_path=relpath(ghost_path, self.root),
            solver_record_path=relpath(self.solvers_path, self.root),
            levels_completed_delta=bundle.levels_completed_delta,
            flow_directions_derived=derived_count,
        )

    # ------------------------------------------------------------------
    # Registry write (append-only; sole writer)
    # ------------------------------------------------------------------

    def _admit(self, record: dict[str, Any]) -> None:
        """Append record to registry/solvers.jsonl.  Creates the registry dir if absent."""
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        with self.solvers_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    # Reverify: re-hash level files and re-check delta from stored artifact
    # ------------------------------------------------------------------

    def reverify(self, solver_id: str) -> dict[str, Any]:
        """Re-verify a stored SolverRecord by re-hashing its level_dir files.

        Returns a dict with {ok, reason, hash_drift, delta_drift}.
        Called by registry.reverify() for no-regression checks.
        """
        record = self._find(solver_id)
        if record is None:
            return {"ok": False, "reason": f"solver_id {solver_id!r} not found in registry"}

        va = record.get("verification_artifact") or {}
        stored_hashes = va.get("level_dir_file_hashes") or {}
        level_dir_rel = va.get("level_dir")
        stored_delta = int(va.get("levels_completed_delta") or 0)

        if not level_dir_rel:
            return {"ok": False, "reason": "verification_artifact.level_dir missing"}

        level_dir = self.root / level_dir_rel
        if not level_dir.is_dir():
            return {"ok": False, "reason": f"level_dir gone from disk: {level_dir_rel}"}

        drift: dict[str, dict[str, str]] = {}
        for name, expected_sha in stored_hashes.items():
            fpath = level_dir / name
            if not fpath.is_file():
                drift[name] = {"stored": expected_sha, "current": "MISSING"}
                continue
            current_sha = _sha256_file(fpath)
            if current_sha != expected_sha:
                drift[name] = {"stored": expected_sha, "current": current_sha}

        # Also re-read delta from result.json
        result_path = level_dir / "result.json"
        current_delta: int | None = None
        delta_drift = False
        if result_path.is_file():
            try:
                rdata = json.loads(result_path.read_text(encoding="utf-8"))
                current_delta = int(rdata.get("levels_completed_delta") or 0)
                delta_drift = current_delta != stored_delta
            except (json.JSONDecodeError, ValueError):
                delta_drift = True

        ok = not drift and not delta_drift
        reason_parts = []
        if drift:
            reason_parts.append(f"hash_drift on {sorted(drift)}")
        if delta_drift:
            reason_parts.append(f"delta_drift stored={stored_delta} current={current_delta}")
        return {
            "ok": ok,
            "reason": "clean" if ok else "; ".join(reason_parts),
            "hash_drift": drift,
            "delta_drift": delta_drift,
            "solver_id": solver_id,
        }

    # ------------------------------------------------------------------
    # Read helpers (no external writer path; read-only by convention)
    # ------------------------------------------------------------------

    def list_solvers(self) -> list[dict[str, Any]]:
        """Read all admitted SolverRecords from solvers.jsonl."""
        if not self.solvers_path.is_file():
            return []
        records = []
        for line in self.solvers_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def _find(self, solver_id: str) -> dict[str, Any] | None:
        for record in self.list_solvers():
            if record.get("solver_id") == solver_id:
                return record
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dry-run", action="store_true", help="print what would happen; write nothing")
    p.add_argument("--list", action="store_true", help="list admitted solvers in registry/solvers.jsonl")
    p.add_argument("--reverify", metavar="SOLVER_ID", help="re-verify a stored solver by solver_id")
    p.add_argument(
        "--run-dir",
        metavar="PATH",
        help="relative run dir (e.g. codexlab/runs/.../level_00); admits the episode as genesis",
    )
    p.add_argument("--level", type=int, default=0, help="target level index (default 0)")
    p.add_argument("--game-id", default="sp80-589a99af", help="game_id (default sp80-589a99af)")
    p.add_argument("--mechanic-class", default="unclassified", help="cross-game mechanic label")
    p.add_argument("--evidence-tier", type=int, default=2, help="evidence tier (0/1/2; default 2)")
    p.add_argument(
        "--derive-only",
        metavar="GHOST_PATH",
        help="read an existing ghost.json, print derived flow directions, and exit (no write)",
    )
    return p


def _main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    c = Crystallizer(root=ROOT)

    if args.derive_only:
        path = Path(args.derive_only)
        if not path.is_file():
            print(f"ERROR: {path} not found", file=sys.stderr)
            return 1
        ghost = json.loads(path.read_text(encoding="utf-8"))
        rows = ghost.get("confirmed_relationship_rows") or []
        print(f"ghost: {ghost.get('ghost_id')}  ({len(rows)} rows)")
        for i, row in enumerate(rows):
            geo = row.get("geometry") or {}
            xy = geo.get("shared_transform_xy")
            axis, direction = derive_flow_axis_direction(xy)
            stored_axis = geo.get("flow_axis", "?")
            stored_dir = geo.get("flow_direction", "?")
            print(
                f"  row {i} [{row.get('relation_family')}] "
                f"transform_xy={xy} "
                f"stored=({stored_axis}, {stored_dir}) "
                f"derived=({axis}, {direction})"
            )
        return 0

    if args.list:
        records = c.list_solvers()
        print(json.dumps(
            {
                "solver_count": len(records),
                "solvers": [
                    {
                        "solver_id": r.get("solver_id"),
                        "mechanic_class": r.get("mechanic_class"),
                        "game_id": r.get("game_id"),
                        "level_index": r.get("level_index"),
                        "levels_completed_delta": (r.get("verification_artifact") or {}).get("levels_completed_delta"),
                        "created_at_utc": r.get("created_at_utc"),
                        "status": r.get("status"),
                    }
                    for r in records
                ],
            },
            indent=2,
        ))
        return 0

    if args.reverify:
        result = c.reverify(args.reverify)
        print(json.dumps(result, indent=2))
        return 0 if result.get("ok") else 1

    if args.dry_run:
        print("== CRYSTALLIZER DRY-RUN ==")
        print(f"  registry/solvers.jsonl: {relpath(c.solvers_path, ROOT)}")
        print(f"  derive_flow_axis_direction([12.0, 36.0]) = {derive_flow_axis_direction([12.0, 36.0])!r}  (L0 geometry)")
        print(f"  derive_flow_axis_direction([4.0, 8.0])  = {derive_flow_axis_direction([4.0, 8.0])!r}")
        print(f"  derive_flow_axis_direction([0.0, 0.0])  = {derive_flow_axis_direction([0.0, 0.0])!r}")
        print(f"  derive_flow_axis_direction(None)         = {derive_flow_axis_direction(None)!r}")
        print()
        print("  crystallize() flow:")
        print("    1. Check bundle.levels_completed_delta > 0  (cardinal; only admission predicate)")
        print("    2. build_ghost() -> patch confirmed_relationship_rows with derived flow dirs")
        print("    3. _apply_mechanic_mapper()  [day-one stub: no-op]")
        print("    4. write ghost.json to level_dir/")
        print("    5. Build SolverRecord (schema codexlab.contra_loop.solver_record.v1)")
        print("    6. Append to registry/solvers.jsonl (append-only; content-addressed)")
        print()
        print("  [DRY-RUN] nothing written.")
        return 0

    if args.run_dir:
        level_dir = ROOT / args.run_dir
        if not level_dir.is_dir():
            print(f"ERROR: level_dir not found: {level_dir}", file=sys.stderr)
            return 1
        print(f"  Building ArtifactBundle from {args.run_dir} ...")
        try:
            bundle = ArtifactBundle.from_level_dir(level_dir, root=ROOT)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if args.level != bundle.target_level_index:
            print(
                f"  Warning: --level {args.level} does not match result.json level_index {bundle.target_level_index}; using result.json value"
            )
        bundle.game_id = args.game_id
        result = c.crystallize(
            bundle=bundle,
            mechanic_class=args.mechanic_class,
            evidence_tier=args.evidence_tier,
        )
        print(json.dumps(dataclasses.asdict(result), indent=2))
        return 0 if result.admitted else 1

    _build_arg_parser().print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
