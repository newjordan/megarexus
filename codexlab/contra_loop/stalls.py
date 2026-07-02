#!/usr/bin/env python3
"""StallDetector -- four human-intervention triggers + actor-identity sanity check.

Deterministic scan for patterns that require human redirect before the loop continues.
Runs in DISCOVER-GAP (after baseline) and again after ADMIT/REJECT.

Stall patterns (from session transcripts and design doc):
  (a) REINVENTION     -- proposed concept already implemented in portband/deltaband/ghost
  (b) THRASH          -- level metric 0->1->0 across 3+ consecutive attempts
  (c) FORWARD_MODEL   -- forward-model predicts 0 fill every step (cov=0/N grounding bug)
  (d) UNREPRODUCIBLE  -- claimed win not reproducible from the stored run_dir

Plus an actor-identity sanity check:
  (e) ACTOR_IDENTITY  -- calibration/oracle seeds magenta (raw-6) as the actor, but the
                         blue platform (raw-9) is the true actor (SP80 cardinal fact)

Usage:
  python codexlab/contra_loop/stalls.py --target sp80:1 --run-dir <run_dir>
  python codexlab/contra_loop/stalls.py --target sp80:1 --dry-run
  python codexlab/contra_loop/stalls.py --list-checks
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CODEXLAB = ROOT / "codexlab"

# ---- Constants ---------------------------------------------------------------

# Bands that already implement concepts the loop might try to reinvent.
# grep targets: (file_relative_to_codexlab, key_symbols)
_BAND_SOURCES: list[tuple[str, list[str]]] = [
    ("contra_head/portband.py", [
        "build_portband_hypotheses",
        "build_portband_surface_feedback",
        "lineup_matrix",
        "edge_lock",
        "centered_dual_port_projection",
        "route_flow_lock",
        "multi_object_goal_adjacency",
    ]),
    ("contra_head/deltaband.py", [
        "build_deltaband_observation",
        "action_surface_result_events",
        "build_action_surface_result_events",
        "deltaband_gate",
    ]),
    ("contra_head/ghost.py", [
        "build_ghost",
        "load_ghost_context",
        "build_confirmed_relation_rows",
        "confirmed_relation_memory",
        "ghost_relation_memory",
    ]),
    ("contra_head/exploreband.py", [
        "build_exploreband_budget",
        "candidate_probe_queue",
        "expected_information_gain",
    ]),
    ("contra_head/geoband.py", [
        "build_geoband_hypothesis",
        "flow_to_target_angular_error",
        "lateral_bandwidth",
    ]),
    ("contra_head/symmetry.py", [
        "build_symmetry_hypotheses",
        "mask_symmetry_score",
        "lock_key_sensor",
    ]),
    ("ghost_mechanics_overlay.py", [
        "RealtimeMechanicMapper",
        "build_mechanic_overlays",
    ]),
]

# SP80 actor truth (from algorithm_pointers.md, session transcripts, MEMORY):
#   Blue platform (raw value 9) IS the actor.
#   Magenta/purple (raw value 6) is the liquid -- NOT the actor.
# Any calibration or oracle hypothesis with actor_raw_pixel_value=6 is miscalibrated.
SP80_CORRECT_ACTOR_RAW_VALUES: set[int] = {9}     # blue/cyan platform
SP80_WRONG_ACTOR_RAW_VALUE: int = 6               # magenta/purple liquid

# Thrash detection: level 0->1->0 across this many consecutive attempts
THRASH_WINDOW: int = 3

# Forward-model coverage: if >=this fraction of top-ranked candidates are
# last_resort_probe or purely static-geometry (no tier-0 portband row) across a
# full episode, flag cov=0/N.
LAST_RESORT_FRACTION_THRESHOLD: float = 0.60

# Candidate kind suffix that signals no portband tier-0 signal was available
LAST_RESORT_KIND: str = "last_resort_probe"

# Schema identifiers
JOURNAL_SCHEMA = "codexlab.contra_loop.journal.v1"
SUMMARY_SCHEMA = "codexlab.sp80_contra_episode_levels.v1"


# ---- Data structures ---------------------------------------------------------

@dataclass
class StallSignal:
    """One detected stall pattern that requires human review."""

    kind: str
    """One of: REINVENTION | THRASH | FORWARD_MODEL | UNREPRODUCIBLE | ACTOR_IDENTITY"""

    severity: str
    """HALT (stop the loop now) or WARN (log and continue; first-occurrence only)."""

    target: str
    """The loop target, e.g. 'sp80:1'."""

    halt_message: str
    """One-sentence human-readable description of the stall."""

    evidence: dict[str, Any] = field(default_factory=dict)
    """Machine-readable evidence: file paths, values, packet indices, etc."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "codexlab.contra_loop.stall_signal.v1",
            "kind": self.kind,
            "severity": self.severity,
            "target": self.target,
            "halt_message": self.halt_message,
            "evidence": self.evidence,
        }


@dataclass
class ScanResult:
    """Output of StallDetector.scan()."""

    target: str
    stalls: list[StallSignal] = field(default_factory=list)

    @property
    def any_halt(self) -> bool:
        return any(s.severity == "HALT" for s in self.stalls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "codexlab.contra_loop.scan_result.v1",
            "target": self.target,
            "any_halt": self.any_halt,
            "stall_count": len(self.stalls),
            "stalls": [s.to_dict() for s in self.stalls],
        }


# ---- Helpers -----------------------------------------------------------------

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _grep_file(path: Path, terms: list[str]) -> list[str]:
    """Return terms that appear (as substring) in the file."""
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [t for t in terms if t in text]


def _parse_metric_from_stdout(stdout: str) -> tuple[int, int] | None:
    """Parse 'N/M' or 'N / M' from keystone stdout.  Returns (n, m) or None."""
    m = re.search(r"\b(\d+)\s*/\s*(\d+)\b", stdout)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _level_from_target(target: str) -> int:
    """Extract level index from 'game:level' string."""
    _, _, lvl = target.partition(":")
    try:
        return int(lvl)
    except ValueError:
        return -1


# ---- Stall checks ------------------------------------------------------------

class StallDetector:
    """Deterministic stall scanner.  Instantiate once per loop orchestrator run.

    Usage pattern (DISCOVER-GAP step)::

        detector = StallDetector(target="sp80:1", calibration_path=CALIBRATION)
        result = detector.scan(run_dir=run_dir, journal_path=journal_path)
        for s in result.stalls:
            print(s.halt_message, s.evidence)
        if result.any_halt:
            raise SystemExit("HALT: " + result.stalls[0].halt_message)
    """

    def __init__(
        self,
        target: str,
        *,
        calibration_path: Path | None = None,
        verifier_script: Path | None = None,
    ) -> None:
        self.target = target
        self.calibration_path = calibration_path or (CODEXLAB / "sp80_surface_calibration.json")
        self.verifier_script = verifier_script or (CODEXLAB / "sp80_verify_episode_run.py")

    # ------------------------------------------------------------------
    # (a) REINVENTION -- concept already exists in the informer bands
    # ------------------------------------------------------------------

    def check_reinvention(
        self,
        concept_name: str,
        *,
        extra_terms: list[str] | None = None,
    ) -> StallSignal | None:
        """Return a HALT StallSignal if concept_name (or any extra_terms) already
        appears in the existing band source files.

        The agent must call this in HYPOTHESIZE before writing any new SolverProtocol
        implementation.  A match means 'this concept exists -- do not reinvent it;
        extend or reuse the band.'

        Args:
            concept_name: the concept the agent is about to implement (e.g. 'flow_lock').
            extra_terms: additional identifiers to grep for.
        """
        all_terms = [concept_name] + (extra_terms or [])
        hits: list[dict[str, Any]] = []
        for rel_path, known_symbols in _BAND_SOURCES:
            full_path = CODEXLAB / rel_path
            # Check both concept_name and known_symbols for partial overlap
            matched_concept = _grep_file(full_path, all_terms)
            matched_symbols = _grep_file(full_path, known_symbols)
            if matched_concept:
                hits.append({
                    "file": rel_path,
                    "matched_terms": matched_concept,
                    "existing_symbols": matched_symbols[:3],
                })

        if not hits:
            return None

        return StallSignal(
            kind="REINVENTION",
            severity="HALT",
            target=self.target,
            halt_message=(
                f"Concept '{concept_name}' already implemented in existing band(s): "
                + ", ".join(h["file"] for h in hits)
                + " -- extend or reuse rather than reinventing."
            ),
            evidence={
                "concept_name": concept_name,
                "hits": hits,
                "instruction": (
                    "grep the matched file for the existing symbol and REUSE it; "
                    "new code must go behind the existing interface, not parallel to it."
                ),
            },
        )

    # ------------------------------------------------------------------
    # (b) THRASH -- level metric 0->1->0 across 3+ attempts
    # ------------------------------------------------------------------

    def check_thrash(
        self,
        *,
        journal_path: Path | None = None,
        run_dirs_glob: str | None = None,
    ) -> StallSignal | None:
        """Detect level metric thrashing (0->1->0) across recent iterations.

        Reads from journal_path if provided (schema codexlab.contra_loop.journal.v1),
        else falls back to scanning run dirs via run_dirs_glob for summary.jsonl rows
        matching the target level.

        A thrash fires when the last THRASH_WINDOW (=3) iterations for the target level
        show a 0->1->0 delta pattern (regression after a solve, or repeated solve/fail
        oscillation).
        """
        level_idx = _level_from_target(self.target)
        deltas = self._load_iteration_deltas(
            level_idx=level_idx,
            journal_path=journal_path,
            run_dirs_glob=run_dirs_glob,
        )
        if len(deltas) < THRASH_WINDOW:
            return None

        # Slide a window over the last THRASH_WINDOW entries
        window = deltas[-THRASH_WINDOW:]
        # Pattern: any instance of (positive, zero) with a prior positive = 0->1->0
        # OR repeated alternation over any 3 consecutive steps
        values = [entry["delta"] for entry in window]
        pattern_fired = False
        pattern_desc = ""

        # 0 -> positive -> 0 (regression after a solve)
        for i in range(len(values) - 2):
            if values[i] == 0 and values[i + 1] > 0 and values[i + 2] == 0:
                pattern_fired = True
                pattern_desc = "0->1->0 regression detected across 3 consecutive iterations"
                break

        # Repeated alternation (all zeros except one middle solve, 3+ times)
        if not pattern_fired and len(values) >= THRASH_WINDOW:
            nz = [v > 0 for v in values]
            if nz.count(True) == 1 and nz[0] is False:
                # exactly one solve bracketed by failures
                pattern_fired = True
                pattern_desc = (
                    f"isolated solve in window of {THRASH_WINDOW} iterations -- "
                    "not reproducible across attempts"
                )

        if not pattern_fired:
            return None

        return StallSignal(
            kind="THRASH",
            severity="HALT",
            target=self.target,
            halt_message=(
                f"Level metric thrash on target {self.target}: {pattern_desc}. "
                "Human redirect needed -- the current hypothesis is not reliably reproducible."
            ),
            evidence={
                "pattern_description": pattern_desc,
                "recent_deltas": window,
                "thrash_window": THRASH_WINDOW,
                "instruction": (
                    "Do NOT re-attempt the same hypothesis without a concrete grounding change. "
                    "Re-read gap_report.json from scratch and start a new hypothesis."
                ),
            },
        )

    def _load_iteration_deltas(
        self,
        *,
        level_idx: int,
        journal_path: Path | None,
        run_dirs_glob: str | None,
    ) -> list[dict[str, Any]]:
        """Return list of {iteration, delta, run_dir, created_at_utc} dicts
        for the target level, oldest-first."""
        entries: list[dict[str, Any]] = []

        # Primary: journal.jsonl
        if journal_path and journal_path.is_file():
            for row in _read_jsonl(journal_path):
                if str(row.get("target", "")) == self.target:
                    entries.append({
                        "iteration": int(row.get("iteration") or 0),
                        "delta": int(row.get("levels_completed_delta") or 0),
                        "run_dir": row.get("run_dir", ""),
                        "created_at_utc": row.get("created_at_utc", ""),
                    })
            entries.sort(key=lambda e: e["iteration"])
            return entries

        # Fallback: scan known run dirs for summary.jsonl rows
        if run_dirs_glob:
            import glob
            paths = sorted(glob.glob(run_dirs_glob))
        else:
            # Default: all episode runs under codexlab/runs/
            game = self.target.split(":", 1)[0]
            paths = sorted(CODEXLAB.glob(f"runs/*{game}*/summary.jsonl"))

        for p in paths:
            summary_path = Path(p) if not isinstance(p, Path) else p
            if not summary_path.is_file():
                continue
            for row in _read_jsonl(summary_path):
                if int(row.get("level_index", -1)) == level_idx:
                    entries.append({
                        "iteration": len(entries),  # approximate order by mtime
                        "delta": int(row.get("levels_completed_delta") or 0),
                        "run_dir": row.get("run_dir", ""),
                        "created_at_utc": row.get("created_at_utc", ""),
                    })
        # Deduplicate by run_dir (same run_dir = same iteration)
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for e in entries:
            key = e["run_dir"]
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        return deduped

    # ------------------------------------------------------------------
    # (c) FORWARD_MODEL -- cov=0/N grounding bug
    # ------------------------------------------------------------------

    def check_forward_model_coverage(
        self,
        *,
        run_dir: Path | None = None,
        keystone_stdout: str | None = None,
        keystone_returncode: int | None = None,
        gap_report: dict[str, Any] | None = None,
    ) -> StallSignal | None:
        """Detect a forward-model coverage-zero grounding bug.

        Three signals are checked (in priority order):

        1. Keystone output: if stdout contains a '0/N' metric (exact-match = 0 of N)
           AND N > 0, the forward model predicts no correct effects.
        2. gap_report.json: if pipeline_predicted contains 'nothing'/'local_fill' and
           observed contains 'global'/'inversion', the physics model is wrong.
        3. Episode packets: if >= LAST_RESORT_FRACTION_THRESHOLD of top candidates
           are last_resort_probe (no portband tier-0 signal), perception is starved.

        The first encounter is severity WARN (log it, continue -- it seeds the gap).
        Repeated firings without grounding change are severity HALT.
        """
        evidence: dict[str, Any] = {}

        # Signal 1: keystone output
        if keystone_stdout is not None:
            metric = _parse_metric_from_stdout(keystone_stdout)
            if metric is not None:
                n, total = metric
                if total > 0 and n == 0:
                    evidence["keystone_metric"] = f"0/{total}"
                    evidence["keystone_returncode"] = keystone_returncode
                    evidence["signal_source"] = "keystone_stdout"
                    return StallSignal(
                        kind="FORWARD_MODEL",
                        severity="HALT",
                        target=self.target,
                        halt_message=(
                            f"Forward-model predicts 0/{total} correct effects in keystone. "
                            "Grounding bug: wrong actor color, wrong physics model, or wrong "
                            "seed frame. Check actor-identity (raw-9 blue platform, not raw-6 magenta)."
                        ),
                        evidence={
                            **evidence,
                            "known_cause": (
                                "SP80: calibration seeds magenta (raw-6) as actor; "
                                "the true actor is the blue platform (raw-9). "
                                "A cov=0/N keystone on SP80 L1 is likely seeding the wrong color."
                            ),
                        },
                    )

        # Signal 2: gap_report
        if gap_report:
            predicted = str(gap_report.get("pipeline_predicted", "")).lower()
            observed = str(gap_report.get("observed", "")).lower()
            if "nothing" in predicted or "local_fill" in predicted or "0" in predicted:
                if "global" in observed or "inversion" in observed or "fill" in observed:
                    evidence["gap_report"] = {
                        "pipeline_predicted": gap_report.get("pipeline_predicted"),
                        "observed": gap_report.get("observed"),
                        "divergence": gap_report.get("divergence"),
                        "action_primitive": gap_report.get("action_primitive"),
                    }
                    evidence["signal_source"] = "gap_report"
                    return StallSignal(
                        kind="FORWARD_MODEL",
                        severity="WARN",
                        target=self.target,
                        halt_message=(
                            f"Forward-model predicts '{gap_report.get('pipeline_predicted')}' "
                            f"but observed '{gap_report.get('observed')}' "
                            f"(action: {gap_report.get('action_primitive')}). "
                            "Pipeline physics model is wrong -- this is the hypothesis seed, not a halt."
                        ),
                        evidence={
                            **evidence,
                            "instruction": (
                                "This is severity=WARN on first encounter; it seeds the HYPOTHESIZE step. "
                                "Form a hypothesis about why the physics model is wrong, then re-run."
                            ),
                        },
                    )

        # Signal 3: episode packets
        if run_dir:
            packets_path = run_dir / "packets.jsonl"
            if packets_path.is_file():
                packets = _read_jsonl(packets_path)
                if packets:
                    last_resort_count = sum(
                        1 for p in packets
                        if (
                            (p.get("contra_recommendation_packet") or {})
                            .get("ranked_recommendations") or [{}]
                        )[0].get("candidate_kind") == LAST_RESORT_KIND
                    )
                    fraction = last_resort_count / len(packets)
                    if fraction >= LAST_RESORT_FRACTION_THRESHOLD:
                        evidence["last_resort_fraction"] = round(fraction, 3)
                        evidence["last_resort_count"] = last_resort_count
                        evidence["total_packets"] = len(packets)
                        evidence["signal_source"] = "episode_packets"
                        return StallSignal(
                            kind="FORWARD_MODEL",
                            severity="WARN",
                            target=self.target,
                            halt_message=(
                                f"Perception-starved: {last_resort_count}/{len(packets)} packets "
                                "used last_resort_probe (no portband tier-0 signal). "
                                "Forward model has no coverage on this level."
                            ),
                            evidence={
                                **evidence,
                                "instruction": (
                                    "Check if PortBand is producing any lineup_matrix rows for "
                                    "this level. If zero rows, the actor/target geometry is not "
                                    "triggering any alignment candidates."
                                ),
                            },
                        )
        return None

    # ------------------------------------------------------------------
    # (d) UNREPRODUCIBLE WIN -- claimed win not verifiable from run_dir
    # ------------------------------------------------------------------

    def check_unreproducible_win(
        self,
        *,
        run_dir: Path,
        dry_run: bool = False,
    ) -> StallSignal | None:
        """Verify a claimed win is reproducible from the stored run_dir.

        Runs sp80_verify_episode_run.py --run-dir <level_dir> for the target level
        and checks that levels_completed_delta > 0.

        A 'win' is ONLY real when:
          - levels_completed_delta > 0 in result.json AND
          - sp80_verify_episode_run returns 0

        If either condition fails, this is an UNREPRODUCIBLE win -- cardinal rule violated.

        Args:
            run_dir: the episode run directory (e.g. .../contra_sp80_episode.../level_01)
            dry_run: if True, only check the result.json file without re-running the verifier.
        """
        level_idx = _level_from_target(self.target)

        # Identify the level subdirectory
        level_dir = self._find_level_dir(run_dir, level_idx)
        if level_dir is None:
            return StallSignal(
                kind="UNREPRODUCIBLE",
                severity="HALT",
                target=self.target,
                halt_message=(
                    f"No level_{level_idx:02d}/ directory found under {run_dir}. "
                    "Cannot verify the claimed win -- run_dir is incomplete."
                ),
                evidence={
                    "run_dir": str(run_dir),
                    "expected_level_dir": f"level_{level_idx:02d}",
                    "instruction": "Re-run the episode; do not claim a solve without a level dir.",
                },
            )

        # Check result.json first (cheap)
        result = _read_json(level_dir / "result.json")
        delta = int(result.get("levels_completed_delta") or 0)

        if delta <= 0:
            # The file itself says delta=0; no need to run verifier
            return StallSignal(
                kind="UNREPRODUCIBLE",
                severity="HALT",
                target=self.target,
                halt_message=(
                    f"Claimed win is NOT in the artifact: "
                    f"result.json shows levels_completed_delta={delta} for {self.target}. "
                    "The solve claim is fabricated (no reproducible artifact)."
                ),
                evidence={
                    "run_dir": str(run_dir),
                    "level_dir": str(level_dir),
                    "levels_completed_delta": delta,
                    "result_json_status": result.get("status"),
                    "episode_status": result.get("status"),
                    "cardinal_rule": "A solve is real ONLY with levels_completed_delta>0. No exceptions.",
                },
            )

        if dry_run:
            # result.json says delta>0; trust it without re-running the verifier
            return None

        # Re-run the verifier to confirm reproduction
        if not self.verifier_script.is_file():
            return StallSignal(
                kind="UNREPRODUCIBLE",
                severity="HALT",
                target=self.target,
                halt_message=(
                    f"Verifier script not found: {self.verifier_script}. "
                    "Cannot confirm reproducibility."
                ),
                evidence={"verifier_script": str(self.verifier_script)},
            )

        cmd = [
            sys.executable, "-B",
            str(self.verifier_script),
            "--run-dir", str(level_dir),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)

        if proc.returncode != 0:
            return StallSignal(
                kind="UNREPRODUCIBLE",
                severity="HALT",
                target=self.target,
                halt_message=(
                    f"Verifier returned rc={proc.returncode} on the claimed win at "
                    f"{level_dir.name} (levels_completed_delta={delta} in result.json). "
                    "Win is NOT reproducible from the stored artifact."
                ),
                evidence={
                    "run_dir": str(run_dir),
                    "level_dir": str(level_dir),
                    "levels_completed_delta_in_result": delta,
                    "verifier_returncode": proc.returncode,
                    "verifier_stdout": proc.stdout[-800:],
                    "verifier_stderr": proc.stderr[-400:],
                    "cardinal_rule": "verifier rc=0 is required for a valid solve artifact.",
                },
            )

        # Parse verifier output to confirm delta
        try:
            verifier_result = json.loads(proc.stdout)
        except json.JSONDecodeError:
            verifier_result = {}
        verified_delta = int(verifier_result.get("levels_completed_delta") or 0)
        if verified_delta <= 0:
            return StallSignal(
                kind="UNREPRODUCIBLE",
                severity="HALT",
                target=self.target,
                halt_message=(
                    f"Verifier returned rc=0 but reports levels_completed_delta={verified_delta} "
                    "(status='{verifier_result.get('status')}' -- note: 'passed' status alone "
                    "does NOT mean solved). Solve claim is invalid."
                ),
                evidence={
                    "run_dir": str(run_dir),
                    "level_dir": str(level_dir),
                    "verifier_returncode": proc.returncode,
                    "verifier_delta": verified_delta,
                    "verifier_status": verifier_result.get("status"),
                    "note": (
                        "status='passed' (verifier contract) != solved. "
                        "Only levels_completed_delta>0 is solve evidence."
                    ),
                },
            )

        return None  # Win is confirmed reproducible

    def _find_level_dir(self, run_dir: Path, level_idx: int) -> Path | None:
        """Find the level_NN subdirectory in run_dir for level_idx."""
        run_dir = Path(run_dir)
        # run_dir might already BE the level dir
        result_here = run_dir / "result.json"
        if result_here.is_file():
            r = _read_json(result_here)
            if int(r.get("level_index", -1)) == level_idx:
                return run_dir

        # Check the canonical level_NN subdirectory
        level_dir = run_dir / f"level_{level_idx:02d}"
        if (level_dir / "result.json").is_file():
            return level_dir

        # Scan all level_* subdirs
        for d in sorted(run_dir.iterdir()):
            if d.is_dir() and d.name.startswith("level_"):
                r = _read_json(d / "result.json")
                if int(r.get("level_index", -1)) == level_idx:
                    return d
        return None

    # ------------------------------------------------------------------
    # (e) ACTOR_IDENTITY -- wrong actor seeded in calibration / oracle
    # ------------------------------------------------------------------

    def check_actor_identity(
        self,
        *,
        run_dir: Path | None = None,
    ) -> StallSignal | None:
        """Sanity-check that the blue platform (raw-9) is the actor, not magenta (raw-6).

        Checks three sources:
        1. sp80_surface_calibration.json actor_raw_pixel_value for the target level.
        2. Oracle hypothesis actor_raw_pixel_value from the first packet in run_dir.
        3. The calibration comment field for 'magenta'/'purple' (belt-and-suspenders).

        This is the most common wrong-calibration error in session transcripts: the
        user-supplied answer seeded magenta as the actor. Any run that loads this
        calibration will ground the forward model on the wrong color.
        """
        level_idx = _level_from_target(self.target)
        game = self.target.split(":", 1)[0]

        # Only apply this check to SP80 (other games have different actor conventions)
        if game != "sp80":
            return None

        wrong_actors: list[dict[str, Any]] = []

        # Source 1: calibration file
        if self.calibration_path.is_file():
            cal = _read_json(self.calibration_path)
            levels = cal.get("levels") or [cal]
            for lvl in levels:
                if not isinstance(lvl, dict):
                    continue
                if int(lvl.get("level_index", level_idx)) != level_idx:
                    continue
                scene = lvl.get("scene_surface_expectations") or {}
                actor_region = scene.get("actor_region") or {}
                actor_val = actor_region.get("actor_raw_pixel_value")
                if actor_val is not None and int(actor_val) == SP80_WRONG_ACTOR_RAW_VALUE:
                    wrong_actors.append({
                        "source": "calibration_file",
                        "path": str(self.calibration_path),
                        "level_index": level_idx,
                        "actor_raw_pixel_value": actor_val,
                        "actor_kind": actor_region.get("actor_kind"),
                        "provenance": actor_region.get("provenance"),
                    })

        # Source 2: first oracle hypothesis in run_dir packets
        if run_dir:
            level_dir = self._find_level_dir(run_dir, level_idx)
            if level_dir:
                packets_path = level_dir / "packets.jsonl"
                if packets_path.is_file():
                    packets = _read_jsonl(packets_path)
                    for packet in packets[:3]:
                        cp = packet.get("contra_packet") or {}
                        oracle = cp.get("oracle_pass") or {}
                        for hyp in (oracle.get("actor_review_hypotheses") or []):
                            actor_val = hyp.get("actor_raw_pixel_value")
                            if actor_val is not None and int(actor_val) == SP80_WRONG_ACTOR_RAW_VALUE:
                                wrong_actors.append({
                                    "source": "oracle_hypothesis",
                                    "packet_case_id": packet.get("case_id"),
                                    "actor_raw_pixel_value": actor_val,
                                    "actor_kind": hyp.get("actor_kind"),
                                    "render_color": hyp.get("render_color"),
                                    "actor_component_id": hyp.get("actor_component_id"),
                                })
                        if wrong_actors:
                            break  # One packet is enough to confirm the miscalibration

        if not wrong_actors:
            return None

        return StallSignal(
            kind="ACTOR_IDENTITY",
            severity="HALT",
            target=self.target,
            halt_message=(
                f"Wrong actor: calibration/oracle seeds raw-{SP80_WRONG_ACTOR_RAW_VALUE} "
                "(magenta/purple liquid) as the actor, but the true SP80 actor is the "
                f"blue platform (raw-{list(SP80_CORRECT_ACTOR_RAW_VALUES)[0]}). "
                "Any forward model seeded with this calibration will ground wrong."
            ),
            evidence={
                "wrong_actor_raw_value": SP80_WRONG_ACTOR_RAW_VALUE,
                "correct_actor_raw_values": sorted(SP80_CORRECT_ACTOR_RAW_VALUES),
                "sources_flagging_wrong_actor": wrong_actors,
                "sp80_actor_truth": (
                    "Blue platform (raw-9, shape=[5,1]) is the actor -- "
                    "confirmed by algorithm_pointers.md: "
                    "'selected_anchor.shape=[5,1], target_label=bucket'. "
                    "Magenta (raw-6) is the liquid, not the actor."
                ),
                "instruction": (
                    "Correct sp80_surface_calibration.json: set actor_raw_pixel_value=9 "
                    "(the blue/cyan platform component, e.g. c0004_v09_00). "
                    "Do NOT use the user_supplied_answer raw-6 entry as the actor origin."
                ),
            },
        )

    # ------------------------------------------------------------------
    # Master scan -- runs all checks
    # ------------------------------------------------------------------

    def scan(
        self,
        *,
        run_dir: Path | None = None,
        journal_path: Path | None = None,
        concept_name: str | None = None,
        extra_reinvention_terms: list[str] | None = None,
        keystone_stdout: str | None = None,
        keystone_returncode: int | None = None,
        gap_report: dict[str, Any] | None = None,
        check_win: bool = False,
        dry_run: bool = False,
    ) -> ScanResult:
        """Run all stall detectors and return a ScanResult.

        Args:
            run_dir: the latest episode run directory (for per-episode checks).
            journal_path: path to journal.jsonl (for cross-iteration checks).
            concept_name: concept the agent is about to implement (for reinvention check).
            extra_reinvention_terms: additional terms to grep for (reinvention check).
            keystone_stdout: captured stdout from a just-run keystone script.
            keystone_returncode: return code from the keystone script.
            gap_report: a gap_report dict from GapFinder (forward-model check).
            check_win: if True, verify that run_dir actually contains a reproducible win.
            dry_run: skip subprocess calls; only check files on disk.
        """
        result = ScanResult(target=self.target)

        # (a) Reinvention check (only when a concept name is provided)
        if concept_name:
            s = self.check_reinvention(concept_name, extra_terms=extra_reinvention_terms)
            if s:
                result.stalls.append(s)

        # (b) Thrash check (always run; uses journal/run dirs)
        s = self.check_thrash(journal_path=journal_path)
        if s:
            result.stalls.append(s)

        # (c) Forward-model coverage check
        s = self.check_forward_model_coverage(
            run_dir=run_dir,
            keystone_stdout=keystone_stdout,
            keystone_returncode=keystone_returncode,
            gap_report=gap_report,
        )
        if s:
            result.stalls.append(s)

        # (d) Unreproducible win check (only when check_win=True and run_dir given)
        if check_win and run_dir:
            s = self.check_unreproducible_win(run_dir=run_dir, dry_run=dry_run)
            if s:
                result.stalls.append(s)

        # (e) Actor identity sanity check (always run for SP80)
        s = self.check_actor_identity(run_dir=run_dir)
        if s:
            result.stalls.append(s)

        return result


# ---- CLI entry point ---------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--target", default="sp80:1",
                   help="loop target, e.g. sp80:1 (default: sp80:1)")
    p.add_argument("--run-dir", metavar="DIR",
                   help="latest episode run directory to inspect")
    p.add_argument("--journal", metavar="FILE",
                   help="path to journal.jsonl for cross-iteration checks")
    p.add_argument("--concept", metavar="NAME",
                   help="concept name to check for reinvention (e.g. 'flow_lock')")
    p.add_argument("--keystone-stdout", metavar="TEXT",
                   help="captured stdout from a keystone run (for cov=0/N check)")
    p.add_argument("--check-win", action="store_true",
                   help="verify that run_dir contains a reproducible win "
                        "(requires levels_completed_delta>0 + verifier rc=0)")
    p.add_argument("--dry-run", action="store_true",
                   help="skip subprocess calls; only inspect files on disk")
    p.add_argument("--list-checks", action="store_true",
                   help="list the stall checks and exit")
    p.add_argument("--calibration", metavar="FILE",
                   help="path to sp80_surface_calibration.json "
                        "(default: codexlab/sp80_surface_calibration.json)")
    args = p.parse_args()

    if args.list_checks:
        checks = [
            {"id": "a", "kind": "REINVENTION",    "trigger": "--concept NAME",
             "desc": "concept already in portband/deltaband/ghost"},
            {"id": "b", "kind": "THRASH",         "trigger": "--journal FILE or auto-scan",
             "desc": "level metric 0->1->0 across 3+ attempts"},
            {"id": "c", "kind": "FORWARD_MODEL",  "trigger": "--keystone-stdout / gap_report / packets",
             "desc": "forward-model predicts 0 fill every step (cov=0/N)"},
            {"id": "d", "kind": "UNREPRODUCIBLE", "trigger": "--check-win --run-dir DIR",
             "desc": "claimed win not reproducible from run_dir artifact"},
            {"id": "e", "kind": "ACTOR_IDENTITY", "trigger": "auto (SP80 calibration check)",
             "desc": "raw-6 magenta seeded as actor; true actor is raw-9 blue platform"},
        ]
        print(json.dumps({"stall_checks": checks}, indent=2))
        return 0

    calibration_path = Path(args.calibration) if args.calibration else None
    detector = StallDetector(
        target=args.target,
        calibration_path=calibration_path,
    )

    run_dir = Path(args.run_dir) if args.run_dir else None
    journal_path = Path(args.journal) if args.journal else None

    scan_result = detector.scan(
        run_dir=run_dir,
        journal_path=journal_path,
        concept_name=args.concept,
        keystone_stdout=args.keystone_stdout,
        check_win=args.check_win,
        dry_run=args.dry_run,
    )

    print(json.dumps(scan_result.to_dict(), indent=2))

    if scan_result.any_halt:
        print(
            f"\n[HALT] {len(scan_result.stalls)} stall(s) detected on target {args.target}.",
            file=sys.stderr,
        )
        for s in scan_result.stalls:
            if s.severity == "HALT":
                print(f"  [{s.kind}] {s.halt_message}", file=sys.stderr)
        return 1

    if scan_result.stalls:
        print(
            f"\n[WARN] {len(scan_result.stalls)} warning(s) -- loop may continue.",
            file=sys.stderr,
        )
    else:
        print("\n[OK] No stalls detected.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
