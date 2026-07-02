#!/usr/bin/env python3
"""Uniform solver interface for the CONTRA Loop Harness.

Defines the SolverProtocol (contra_packet)->List[ScoredCandidate] contract, the
ScoredCandidate dataclass with mandatory truth_boundary flags, and the
MergeRegistry that applies the informer ceiling so rows[0].value_score==max.

Wraps the three existing inline band consumers (PortBand, Oracle, Ghost) as
registered SolverProtocol implementations so new agent-written solvers and the
existing harness logic share one merge path.

EMIT-BUT-MARK: any candidate whose gate fires emits value_score=0.0 plus a
reason string instead of being silently dropped.  It sinks to the bottom of the
value-score sort (never selected) but stays auditable.

Evidence-tier ordering (mirrors sp80_player_harness.py:2132-2147):
  tier-0  execution-derived (PortBand lineup_matrix + feedback)
  tier-1  Oracle review hypotheses (informer, ceiling-capped)
  tier-2  Ghost relation memory  (informer, ceiling-capped)
  higher  new agent solvers that have not yet earned tier-0 status are capped

Informer ceiling: min(tier-0 value_scores) - 1e-3.
If no tier-0 candidates are present the ceiling is not applied (informers
surface as normal candidates so the agent has something to act on).

Usage:
    python codexlab/contra_loop/solver_interface.py --dry-run
    python codexlab/contra_loop/solver_interface.py --list-solvers
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Ensure codexlab/ is on sys.path so contra_head.ghost is importable.
# File lives at codexlab/contra_loop/solver_interface.py -> parents[1] is codexlab/.
_CODEXLAB = Path(__file__).resolve().parents[1]
if str(_CODEXLAB) not in sys.path:
    sys.path.insert(0, str(_CODEXLAB))

from contra_head.ghost import TRUTH_BOUNDARY  # noqa: E402 (import after path fix)

# ---------------------------------------------------------------------------
# Public schema identifier
# ---------------------------------------------------------------------------
SCHEMA = "codexlab.contra_loop.solver_interface.v1"
SCORED_CANDIDATE_SCHEMA = "codexlab.contra_loop.scored_candidate.v1"

# Canonical informer-tier ceiling formula (mirrors sp80_player_harness.py:2133).
# Any solver at tier >= INFORMER_TIER_FLOOR is ceiling-capped below the lowest
# tier-0 verified value.
INFORMER_TIER_FLOOR = 1  # tier-0 is verified execution-derived; tier-1+ are informers

# ---------------------------------------------------------------------------
# ScoredCandidate dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScoredCandidate:
    """One value-scored action candidate emitted by a SolverProtocol.produce() call.

    Mandatory truth-boundary flags (all False by default — fail-closed):
      not_prediction       True: this is a candidate, not a ground-truth claim
      not_control_truth    True: the solver does not control the game engine
      not_solve_truth      True: a high score is not solve evidence
      solve_condition_admitted: True ONLY when a registry-admitted artifact
                               (levels_completed_delta>0) backs this candidate

    EMIT-BUT-MARK contract: when gated=True the candidate MUST have
    value_score==0.0 and a non-empty gated_reason.  The MergeRegistry
    preserves gated rows in the returned list (never drops them).
    """

    # Core action spec
    primitive: str
    value_score: float  # clamped to [0, 1]
    candidate_kind: str
    evidence_tier: int  # 0=execution-derived; 1=oracle; 2=ghost; higher=unverified

    # Provenance
    evidence_sources: list[str] = dataclasses.field(default_factory=list)
    evidence_refs: dict[str, Any] = dataclasses.field(default_factory=dict)
    signals: dict[str, Any] = dataclasses.field(default_factory=dict)

    # Human-readable label (required; used for audit and stall detection)
    reason: str = ""

    # Emit-but-mark gate fields
    gated: bool = False          # True when blocked by oscillation/stall gate
    gated_reason: str = ""       # e.g. "stale_repetition_hard_blocked"

    # Truth-boundary flags (fail-closed: all False unless proven)
    not_prediction: bool = True
    not_control_truth: bool = True
    not_solve_truth: bool = True
    solve_condition_admitted: bool = False  # True only when registry has delta>0 proof

    # Assigned by MergeRegistry.merge() after sorting
    rank: int = 0

    # Derived schema tag for JSON serialization
    schema: str = SCORED_CANDIDATE_SCHEMA

    def clamp(self) -> "ScoredCandidate":
        """Return a copy with value_score clamped to [0, 1]."""
        return dataclasses.replace(self, value_score=max(0.0, min(1.0, self.value_score)))

    def truth_boundary_dict(self) -> dict[str, Any]:
        """Serializable truth-boundary block matching ghost.truth_boundary() shape."""
        return {
            **TRUTH_BOUNDARY,                      # all False base from ghost.py:35-42
            "admission_status": "not_admitted",
            "not_prediction": self.not_prediction,
            "not_control_truth": self.not_control_truth,
            "not_solve_truth": self.not_solve_truth,
            "solve_condition_admitted": self.solve_condition_admitted,
            "evidence_boundary": "solver_interface_candidate_not_solve_truth",
            "route_or_replay_used": False,
            "coordinate_copy_used": False,
            "action_authority": "recommendation_only",
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to a dict compatible with the harness recommendation_row format."""
        d: dict[str, Any] = {
            "schema": self.schema,
            "rank": self.rank,
            "primitive": self.primitive,
            "value_score": round(self.value_score, 6),
            "confidence": round(self.value_score, 6),
            "evidence_tier": self.evidence_tier,
            "candidate_kind": self.candidate_kind,
            "evidence_sources": list(self.evidence_sources),
            "evidence_refs": dict(self.evidence_refs),
            "signals": dict(self.signals),
            "reason": self.reason,
            "truth_boundary": self.truth_boundary_dict(),
        }
        if self.gated:
            d["gated"] = True
            d["gated_reason"] = self.gated_reason
        return d

    @classmethod
    def from_recommendation_row(cls, row: dict[str, Any]) -> "ScoredCandidate":
        """Convert an existing harness recommendation_row dict to ScoredCandidate.

        Used by the band wrappers to translate sp80_player_harness.py outputs
        into the uniform ScoredCandidate interface without re-implementing the
        band scoring logic.
        """
        value = max(0.0, min(1.0, float(row.get("value_score") or 0.0)))
        gated = bool(row.get("gated", False))
        gated_reason = str(row.get("gated_reason") or row.get("reason") or "")
        if gated and value > 0.0:
            # Harness uses value=0.0 for gated rows; enforce the contract here.
            value = 0.0
        tb = row.get("truth_boundary") or {}
        return cls(
            primitive=str(row.get("primitive") or "activate_flow"),
            value_score=value,
            candidate_kind=str(row.get("candidate_kind") or "unknown"),
            evidence_tier=int(row.get("evidence_tier") or 0),
            evidence_sources=list(row.get("evidence_sources") or []),
            evidence_refs=dict(row.get("evidence_refs") or {}),
            signals=dict(row.get("signals") or {}),
            reason=str(row.get("reason") or ""),
            gated=gated,
            gated_reason=gated_reason if gated else "",
            not_prediction=bool(tb.get("not_prediction", True)),
            not_control_truth=bool(tb.get("not_control_truth", True)),
            not_solve_truth=bool(tb.get("not_solve_truth", True)),
            solve_condition_admitted=bool(tb.get("solve_condition_admitted", False)),
        )


# Convenience constructor for gated (emit-but-mark) candidates
def gated_candidate(
    primitive: str,
    candidate_kind: str,
    gated_reason: str,
    evidence_tier: int,
    evidence_sources: list[str] | None = None,
    evidence_refs: dict[str, Any] | None = None,
    signals: dict[str, Any] | None = None,
) -> ScoredCandidate:
    """Build a gated ScoredCandidate (value_score=0.0, gated=True).

    EMIT-BUT-MARK: the candidate is included in the merged list at the bottom
    of the sort, never selected, but remains auditable in the packet JSON.
    """
    return ScoredCandidate(
        primitive=primitive,
        value_score=0.0,
        candidate_kind=candidate_kind,
        evidence_tier=evidence_tier,
        evidence_sources=list(evidence_sources or []),
        evidence_refs=dict(evidence_refs or {}),
        signals=dict(signals or {}),
        reason=gated_reason,
        gated=True,
        gated_reason=gated_reason,
    )


# ---------------------------------------------------------------------------
# SolverProtocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SolverProtocol(Protocol):
    """Uniform interface every band consumer and agent solver must implement.

    The MergeRegistry calls .produce(contra_packet, **ctx) on each registered
    solver and merges the results under the informer ceiling policy.

    ctx keyword arguments supplied by the MergeRegistry:
        before_state  (dict)  : harness before-state (selection, goal proximity)
        action_memory (list)  : harness action-memory list (oscillation history)

    Additional game-specific context can be passed via ctx; implementations
    must silently ignore keys they do not recognize.
    """

    # Identifying metadata (set as class/instance attributes)
    solver_id: str        # unique stable identifier, e.g. "portband_lineup"
    evidence_tier: int    # the MAXIMUM tier this solver may produce

    def produce(
        self,
        contra_packet: dict[str, Any],
        **ctx: Any,
    ) -> list[ScoredCandidate]:
        """Produce a list of value-scored candidates from the current CONTRA packet.

        Contract:
          - Every returned ScoredCandidate.value_score is in [0, 1].
          - Gated rows MUST have value_score==0.0 AND gated==True.
          - Never raise; return [] on any internal error (fail-closed).
          - The MergeRegistry applies the informer ceiling after this call
            (do not cap your own rows here; that is the registry's job).
        """
        ...


# ---------------------------------------------------------------------------
# Band consumer wrappers (lazy-import so solver_interface.py is import-clean
# even when arcengine / numpy are not installed in the test environment)
# ---------------------------------------------------------------------------


class PortBandSolver:
    """Wraps portband_lineup_matrix_candidates + portband_signal_candidates.

    Evidence tier: 0 (execution-derived, static edge-alignment).
    These are the primary tier-0 candidates: PortBand lineup_matrix rows
    (transform_hypothesis_xy -> directional move primitive) + ExploreBAND
    portband signal probes.  They are NEVER ceiling-capped.
    """

    solver_id = "portband_lineup"
    evidence_tier = 0

    def produce(
        self,
        contra_packet: dict[str, Any],
        **ctx: Any,
    ) -> list[ScoredCandidate]:
        before_state: dict[str, Any] = ctx.get("before_state") or {}
        action_memory: list[dict[str, Any]] = ctx.get("action_memory") or []
        try:
            # Lazy import avoids triggering sp80_player_harness module-level
            # SP80BaseController / arcengine init when solver_interface is imported.
            import sp80_player_harness as _h  # type: ignore[import]
            rows: list[dict[str, Any]] = []
            rows.extend(_h.portband_lineup_matrix_candidates(
                contra_packet, before_state=before_state, action_memory=action_memory
            ))
            rows.extend(_h.portband_signal_candidates(contra_packet, action_memory))
            return [ScoredCandidate.from_recommendation_row(r) for r in rows]
        except Exception as exc:  # pylint: disable=broad-except
            # Fail-closed: log and return empty so the merge continues.
            return [gated_candidate(
                primitive="activate_flow",
                candidate_kind="portband_lineup_import_error",
                gated_reason=f"portband_import_or_call_error: {exc}",
                evidence_tier=self.evidence_tier,
                evidence_sources=["portband_lineup_matrix_candidates"],
            )]


class OracleSolver:
    """Wraps oracle_review_candidates (evidence_tier 1, informer, ceiling-capped).

    Oracle hypotheses are review-only (review_hypothesis_not_truth_source).
    They surface only when tier-0 sources are empty (L1 perception starvation).
    """

    solver_id = "oracle_review"
    evidence_tier = 1

    def produce(
        self,
        contra_packet: dict[str, Any],
        **ctx: Any,
    ) -> list[ScoredCandidate]:
        before_state: dict[str, Any] = ctx.get("before_state") or {}
        action_memory: list[dict[str, Any]] = ctx.get("action_memory") or []
        try:
            import sp80_player_harness as _h  # type: ignore[import]
            rows = _h.oracle_review_candidates(
                contra_packet, before_state=before_state, action_memory=action_memory
            )
            return [ScoredCandidate.from_recommendation_row(r) for r in rows]
        except Exception as exc:  # pylint: disable=broad-except
            return [gated_candidate(
                primitive="activate_flow",
                candidate_kind="oracle_import_error",
                gated_reason=f"oracle_import_or_call_error: {exc}",
                evidence_tier=self.evidence_tier,
                evidence_sources=["oracle_review_candidates"],
            )]


class GhostSolver:
    """Wraps ghost_relation_memory_candidates (evidence_tier 2, informer, ceiling-capped).

    Ghost priors surface when both tier-0 and Oracle are empty, replacing the
    blind last-resort activate_flow with a geometry-consistent probe from a
    confirmed relation.  Ghost ONLY carries confirmed relations from episodes
    where levels_completed_delta>0 (ghost.py:257 guard).
    """

    solver_id = "ghost_relation_memory"
    evidence_tier = 2

    def produce(
        self,
        contra_packet: dict[str, Any],
        **ctx: Any,
    ) -> list[ScoredCandidate]:
        before_state: dict[str, Any] = ctx.get("before_state") or {}
        action_memory: list[dict[str, Any]] = ctx.get("action_memory") or []
        try:
            import sp80_player_harness as _h  # type: ignore[import]
            rows = _h.ghost_relation_memory_candidates(
                contra_packet, before_state=before_state, action_memory=action_memory
            )
            return [ScoredCandidate.from_recommendation_row(r) for r in rows]
        except Exception as exc:  # pylint: disable=broad-except
            return [gated_candidate(
                primitive="activate_flow",
                candidate_kind="ghost_import_error",
                gated_reason=f"ghost_import_or_call_error: {exc}",
                evidence_tier=self.evidence_tier,
                evidence_sources=["ghost_relation_memory_candidates"],
            )]


# ---------------------------------------------------------------------------
# MergeRegistry
# ---------------------------------------------------------------------------


class MergeRegistry:
    """Registry of SolverProtocol implementations + the merge policy.

    The merge() method:
      1. Collects candidates from all registered solvers.
      2. Computes the informer ceiling from tier-0 (verified) candidates.
      3. Caps tier-1+ candidates to the ceiling (preserving rows[0].value==max).
      4. Preserves all gated (emit-but-mark) rows with value_score=0.0.
      5. Sorts by value_score descending and assigns ranks.

    Invariants enforced:
      - rows[0].value_score == max(all value_scores)
        (sp80_verify_episode_run requires top recommendation == max value)
      - No tier-1/2 candidate ever outranks any tier-0 candidate when tier-0
        is non-empty.
      - Gated rows never outrank live rows (value_score == 0.0).
    """

    def __init__(self) -> None:
        self._solvers: dict[str, SolverProtocol] = {}

    def register(self, solver: SolverProtocol) -> None:
        """Register a solver.  Later registrations with the same solver_id win."""
        sid = str(solver.solver_id)
        self._solvers[sid] = solver

    def deregister(self, solver_id: str) -> None:
        self._solvers.pop(solver_id, None)

    def solver_ids(self) -> list[str]:
        return list(self._solvers)

    def merge(
        self,
        contra_packet: dict[str, Any],
        **ctx: Any,
    ) -> list[ScoredCandidate]:
        """Run all registered solvers and merge under the informer ceiling.

        Returns a list of ScoredCandidate sorted by value_score descending,
        with ranks assigned (1-based).  Gated rows are included at the tail.

        ctx forwarded to every solver.produce() call:
            before_state  (dict)  : harness before-state
            action_memory (list)  : harness action-memory list

        The merge preserves the EMIT-BUT-MARK contract: rows with gated=True
        appear in the output with value_score=0.0, never silently dropped.
        """
        all_candidates: list[ScoredCandidate] = []

        for solver in self._solvers.values():
            try:
                produced = solver.produce(contra_packet, **ctx)
            except Exception as exc:  # pylint: disable=broad-except
                # Fail-closed: a crashing solver becomes one gated emit row.
                produced = [gated_candidate(
                    primitive="activate_flow",
                    candidate_kind=f"{getattr(solver, 'solver_id', 'unknown')}_crash",
                    gated_reason=f"solver_produce_raised: {exc}",
                    evidence_tier=int(getattr(solver, "evidence_tier", 99)),
                    evidence_sources=[str(getattr(solver, "solver_id", "unknown"))],
                )]
            for cand in (produced or []):
                all_candidates.append(cand.clamp())

        return _apply_informer_ceiling_and_sort(all_candidates)

    def __repr__(self) -> str:
        return f"MergeRegistry(solvers={list(self._solvers)})"


# ---------------------------------------------------------------------------
# Merge logic (pure function, testable without a full MergeRegistry)
# ---------------------------------------------------------------------------


def _apply_informer_ceiling_and_sort(
    candidates: list[ScoredCandidate],
) -> list[ScoredCandidate]:
    """Apply the informer ceiling and return a value-sorted list with ranks.

    Mirrors sp80_player_harness.py:2132-2167:
      informer_ceiling = min(tier-0 value_scores) - 1e-3  [if any tier-0 > 0]
      tier-1+ candidates are capped to this ceiling
      final sort: value_score descending; gated rows (value=0) trail

    Gated rows (value_score==0.0, gated=True) are KEPT in the list.
    They are placed at the tail of the sort because value_score==0.0 ensures
    no gated row ever leads when a live row exists.
    """
    # Separate live tier-0 candidates (positive, not gated) for ceiling calc.
    tier0_positive = [
        c.value_score
        for c in candidates
        if c.evidence_tier < INFORMER_TIER_FLOOR
        and not c.gated
        and c.value_score > 0.0
    ]
    informer_ceiling: float | None = (
        (min(tier0_positive) - 1e-3) if tier0_positive else None
    )

    result: list[ScoredCandidate] = []
    for cand in candidates:
        if cand.evidence_tier >= INFORMER_TIER_FLOOR and informer_ceiling is not None:
            if not cand.gated:
                # Cap: informer cannot outrank lowest verified tier-0 candidate.
                capped_value = min(cand.value_score, informer_ceiling)
                if capped_value <= 0.0:
                    # Would vanish -- emit-but-mark instead of dropping.
                    result.append(dataclasses.replace(
                        cand,
                        value_score=0.0,
                        gated=True,
                        gated_reason=(
                            cand.gated_reason
                            or f"informer_ceiling_zeroed_{cand.solver_id if hasattr(cand, 'solver_id') else cand.candidate_kind}"
                        ),
                        signals={**cand.signals, "informer_value_ceiling": informer_ceiling},
                    ))
                else:
                    result.append(dataclasses.replace(
                        cand,
                        value_score=round(capped_value, 6),
                        signals={**cand.signals, "informer_value_ceiling": informer_ceiling},
                    ))
            else:
                # Already gated (value_score==0.0): keep as-is.
                result.append(cand)
        else:
            result.append(cand)

    # Sort: descending by value_score (gated rows have 0.0, trail naturally).
    result.sort(key=lambda c: c.value_score, reverse=True)

    # Assign 1-based ranks.
    for idx, cand in enumerate(result, start=1):
        cand.rank = idx

    # Invariant check: rows[0].value_score == max(all value_scores).
    # (This is always true after the sort; assert for auditability in dev.)
    if result:
        max_value = max(c.value_score for c in result)
        assert abs(result[0].value_score - max_value) < 1e-9, (
            f"MergeRegistry: rows[0].value_score={result[0].value_score} "
            f"!= max={max_value} -- informer ceiling logic is broken"
        )

    return result


# ---------------------------------------------------------------------------
# Default registry (the three built-in band consumers pre-registered)
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: MergeRegistry | None = None


def default_registry() -> MergeRegistry:
    """Return the module-level default MergeRegistry (created once).

    Pre-registered solvers (in evidence-tier order):
      tier-0  portband_lineup      PortBandSolver
      tier-1  oracle_review        OracleSolver
      tier-2  ghost_relation_memory GhostSolver

    Agent-written solvers are registered via registry.register(solver) after
    obtaining the registry from this function.  They are assigned an
    evidence_tier >= 1 (claiming tier-0 requires a verified artifact -- the
    registry.py admission gate enforces this).
    """
    global _DEFAULT_REGISTRY  # noqa: PLW0603
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = MergeRegistry()
        _DEFAULT_REGISTRY.register(PortBandSolver())
        _DEFAULT_REGISTRY.register(OracleSolver())
        _DEFAULT_REGISTRY.register(GhostSolver())
    return _DEFAULT_REGISTRY


def reset_registry() -> None:
    """Reset the module-level default registry (used in tests)."""
    global _DEFAULT_REGISTRY  # noqa: PLW0603
    _DEFAULT_REGISTRY = None


# ---------------------------------------------------------------------------
# Top-level merge() convenience function
# ---------------------------------------------------------------------------


def merge(
    contra_packet: dict[str, Any],
    *,
    before_state: dict[str, Any] | None = None,
    action_memory: list[dict[str, Any]] | None = None,
    registry: MergeRegistry | None = None,
) -> list[ScoredCandidate]:
    """Merge all registered solvers into a ranked ScoredCandidate list.

    This is the primary entry point for callers that want to consume the full
    merged list without managing a MergeRegistry explicitly.  It uses the
    module-level default_registry() unless a custom registry is supplied.

    Args:
        contra_packet:  the full CONTRA output packet (run_contra_view() result)
        before_state:   harness before-state dict (optional)
        action_memory:  harness action-memory list (optional)
        registry:       custom MergeRegistry; uses default_registry() if None

    Returns:
        List[ScoredCandidate] sorted by value_score descending, ranks assigned.
        Gated rows are included at the tail.  rows[0].value_score == max.
    """
    reg = registry if registry is not None else default_registry()
    return reg.merge(
        contra_packet,
        before_state=before_state or {},
        action_memory=action_memory or [],
    )


# ---------------------------------------------------------------------------
# CLI (--dry-run / --list-solvers)
# ---------------------------------------------------------------------------


def _dry_run_demo() -> None:
    """Exercise the registry with a minimal synthetic contra_packet stub."""
    print(f"schema: {SCHEMA}")
    reg = default_registry()
    print(f"registered solvers: {reg.solver_ids()}")

    # Synthetic stub: empty packet (simulates L1 perception-starved case where
    # no PortBand rows are present and informers are the only source).
    stub_packet: dict[str, Any] = {
        "schema": "codexlab.contra_view.v1",
        "game_id": "sp80-589a99af",
        "level_index": 1,
        "oracle_pass": None,
        "ghost_pass": None,
    }
    before_state: dict[str, Any] = {"selected": False}
    action_memory: list[dict[str, Any]] = []

    candidates = merge(
        stub_packet,
        before_state=before_state,
        action_memory=action_memory,
        registry=reg,
    )

    print(f"merged candidates: {len(candidates)}")
    for cand in candidates[:8]:
        tag = " [GATED]" if cand.gated else ""
        print(
            f"  rank={cand.rank}  tier={cand.evidence_tier}"
            f"  value={cand.value_score:.4f}"
            f"  primitive={cand.primitive!r}"
            f"  kind={cand.candidate_kind!r}{tag}"
        )

    if not candidates:
        print("  (no candidates -- expected for an empty stub packet)")
    elif candidates[0].value_score > 0.0:
        print(f"top recommendation: {candidates[0].primitive!r} (value={candidates[0].value_score:.4f})")
    else:
        print("all candidates gated (value=0.0) -- no positive recommendation")

    print("\ntruth_boundary (top candidate):")
    if candidates:
        print(json.dumps(candidates[0].truth_boundary_dict(), indent=2))

    print(
        "\nMerge invariant check: rows[0].value_score == max:"
        f" {'PASS' if not candidates or abs(candidates[0].value_score - max(c.value_score for c in candidates)) < 1e-9 else 'FAIL'}"
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Exercise the registry with a stub packet and print results.",
    )
    p.add_argument(
        "--list-solvers",
        action="store_true",
        help="Print the registered solver IDs and evidence tiers.",
    )
    args = p.parse_args(argv)

    if args.list_solvers:
        reg = default_registry()
        print(json.dumps({
            "schema": SCHEMA,
            "registered_solvers": [
                {"solver_id": sid, "evidence_tier": getattr(reg._solvers[sid], "evidence_tier", "?")}
                for sid in reg.solver_ids()
            ],
        }, indent=2))
        return 0

    # Default (and --dry-run): run the demo.
    _dry_run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
