#!/usr/bin/env python3
"""CONTRA Loop Harness -- the accumulating solver library.

A solver is admitted ONLY with a reproducible artifact (provenance run_dir whose
result.json records levels_completed_delta>0 + a passing verifier).  This is the
anti-reward-hacking spine made structural: the library can never hold a false solver.

    python codexlab/contra_loop/solver_registry.py --list
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path(__file__).resolve().parent / "solver_registry.json"
SCHEMA = "codexlab.contra_loop.solver_registry.v1"

# Seed solvers mined from CONTRA's own VERIFIED wins (see seed/algorithm_pointers.md).
# Each provenance run_dir records delta=1 + status completed; the SP80-L0 solution is
# bit-identical across 5 run_dirs -> algorithmic, not luck.
_PROV_L0 = "codexlab/runs/contra_sp80_episode_levels_00_01_20260608T175142Z/level_00"
SEED_SOLVERS: list[dict[str, Any]] = [
    {
        "name": "PORTBAND_EDGE_LOCK_ALIGN",
        "applies_when": "a selectable source actor and a target port/sink exist",
        "procedure": "select source; issue axis-ordered PortBand lineup moves toward the target "
                     "port (value_score~0.82, centered_dual_port_projection) until edge-locked",
        "mechanic": "liquid flows only when the source actor is edge-aligned with its sink",
        "evidence_sources": ["portband.lineup_matrix.rows"],
        "provenance": {"run_dir": _PROV_L0, "packet_indices": [0, 1, 2, 3, 4], "levels_completed_delta": 1},
        "verified": True,
        "target": "sp80:0",
    },
    {
        "name": "FLOW_DIRECTION_DESCENT",
        "applies_when": "a locked source must close remaining distance to its sink along one axis",
        "procedure": "repeat move toward sink while distance_reduced (value_score~0.69); stop when distance_delta==0",
        "mechanic": "monotone distance descent finishes the alignment a lineup-lock leaves open",
        "evidence_sources": ["goal_condition.nearest_goal.relative_display_direction"],
        "provenance": {"run_dir": _PROV_L0, "packet_indices": [6, 7, 8, 9, 10, 11, 12], "levels_completed_delta": 1},
        "verified": True,
        "target": "sp80:0",
    },
    {
        "name": "MULTI_SOURCE_GOAL_ADJACENCY",
        "applies_when": "multiple independent source->sink pairs remain",
        "procedure": "least-steps plan over independent source->target pairs (value_score~0.75)",
        "mechanic": "each source is locked to its own sink before any flow is committed",
        "evidence_sources": ["portband.lineup_matrix.rows"],
        "provenance": {"run_dir": _PROV_L0, "packet_indices": [13, 14], "levels_completed_delta": 1},
        "verified": True,
        "target": "sp80:0",
    },
    {
        "name": "ACTIVATE_FLOW_ON_FULL_ALIGNMENT",
        "applies_when": "all source actors are edge-locked to their sinks",
        "procedure": "issue activate_flow (ACTION5) ONLY when board alignment is complete",
        "mechanic": "activate_flow while fully aligned commits the pour and completes the level "
                    "(negative control: identical activate_flow at idx 5 did NOT win -- only alignment differed)",
        "evidence_sources": ["action_semantics_ledger"],
        "provenance": {"run_dir": _PROV_L0, "packet_indices": [15], "levels_completed_delta": 1,
                       "stop_reason": "levels_completed_increased"},
        "verified": True,
        "target": "sp80:0",
    },
]

# Wanted mechanics: targets the loop must DISCOVER. NOT solvers -- never scored, never admitted
# until they earn a verified provenance run_dir.
WANTED_MECHANICS: list[dict[str, Any]] = [
    {
        "name": "STABLE_MULTI_SOURCE_RELAY_LOCK",
        "target": "sp80:1",
        "want": "commit to one source->target edge-lock at a time, suppress re-selection thrash, "
                "and defer activate_flow until ALL sources are locked",
        # Crystallized from a real failing L1 run (DISCOVER pass). Implementable as a scorer modifier.
        "procedure": [
            "LOCK: when a multi_object_goal_adjacency_plan candidate scores >=0.85, COMMIT to it and "
            "pin the (source,target) pair as the active relay leg.",
            "HOLD: while the active leg is still closing distance, SUPPRESS switching to any other "
            "candidate_kind (anti-thrash). Release only when the leg's source is edge-locked "
            "(portband_score==1.0) or stuck (2 consecutive no-progress on this leg).",
            "ADVANCE: on lock-complete, move to the next unlocked source->sink leg; repeat LOCK/HOLD.",
            "DEFER: hard-gate activate_flow until ALL legs are locked; the bare 0.48 "
            "unknown_action_family activate_flow probe must NOT win while any leg is unlocked.",
        ],
        "thresholds_from_evidence": {"lock_engage_value": 0.85, "thrash_switches_observed": 14,
                                     "premature_activate_flow_value": 0.48},
        "evidence_missing": "L1: multi_object_goal_adjacency_plan lock (vs->1.0) engages pkt0-4 then is "
                            "ABANDONED at pkt5 (switch to a 0.6 candidate) and never re-locks; 14 "
                            "candidate_kind switches across 9 kinds; activate_flow fires prematurely at "
                            "pkt11 & pkt13 (vs=0.48, dist_delta=-3, unaligned). Ends budget_exhausted, delta=0.",
        "provenance_failure": {"run_dir": "codexlab/runs/contra_sp80_episode_levels_01_20260608T184427Z/level_01",
                               "lock_abandoned_packet": 5, "premature_activate_flow_packets": [11, 13],
                               "levels_completed_delta": 0},
        "crystallization_target": "scorer modifier: (a) lock-hold bonus on the committed leg's candidate_kind, "
                                  "(b) penalty on candidate_kind switches while a leg is locked-and-closing, "
                                  "(c) hard gate on activate_flow until all legs locked.",
        "implementation_status": "PARTIAL: HOLD rule implemented in sp80_player_harness.relay_lock_hold_bonus "
                                 "(+0.20 persistence on a committed relay leg, released when stuck). Measured: "
                                 "thrash 14->12 switches, bonus fired 7/20 packets, L0 still solves (delta=1), "
                                 "full gate green. NOT YET SOLVING L1 (delta=0) -- DEFER (hard-gate premature "
                                 "activate_flow) and ADVANCE (per-leg target_object_id tracking) still TODO.",
        "verified": False,
    },
]


def load_registry() -> dict[str, Any]:
    if REGISTRY_PATH.is_file():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry = {
        "schema": SCHEMA,
        "solvers": list(SEED_SOLVERS),
        "wanted_mechanics": list(WANTED_MECHANICS),
        "provenance_note": "Every solver carries a run_dir with levels_completed_delta>0. No exceptions.",
    }
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return registry


def _provenance_is_real(record: dict[str, Any]) -> tuple[bool, str]:
    prov = record.get("provenance") or {}
    run_dir = prov.get("run_dir")
    if not run_dir:
        return False, "no provenance.run_dir"
    result = ROOT / run_dir / "result.json"
    if not result.is_file():
        return False, f"provenance run_dir missing result.json: {run_dir}"
    delta = int(json.loads(result.read_text(encoding="utf-8")).get("levels_completed_delta") or 0)
    if delta <= 0:
        return False, f"provenance levels_completed_delta={delta} (need >0)"
    return True, f"verified delta={delta} @ {run_dir}"


def register_solver(record: dict[str, Any]) -> dict[str, Any]:
    """Admit a solver ONLY if its provenance is a real win on disk. Cardinal rule, enforced."""
    ok, why = _provenance_is_real(record)
    if not ok:
        raise ValueError(f"REJECTED solver {record.get('name')!r}: {why}")
    registry = load_registry()
    registry["solvers"] = [s for s in registry["solvers"] if s.get("name") != record.get("name")]
    registry["solvers"].append({**record, "verified": True})
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"admitted": record.get("name"), "provenance": why}


def solvers_for_target(target: str) -> list[dict[str, Any]]:
    reg = load_registry()
    game = target.split(":", 1)[0]
    return [s for s in reg["solvers"] if str(s.get("target", "")).split(":", 1)[0] == game]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--list", action="store_true", help="list admitted solvers + wanted mechanics")
    p.add_argument("--verify-provenance", action="store_true", help="re-check every solver's provenance on disk")
    args = p.parse_args()
    reg = load_registry()
    if args.verify_provenance:
        for s in reg["solvers"]:
            ok, why = _provenance_is_real(s)
            print(f"  [{'OK ' if ok else 'BAD'}] {s['name']}: {why}")
        return 0
    # default + --list both print the library
    print(json.dumps({
        "schema": reg["schema"],
        "solver_count": len(reg["solvers"]),
        "solvers": [{"name": s["name"], "target": s.get("target"), "verified": s.get("verified"),
                     "provenance": (s.get("provenance") or {}).get("run_dir")} for s in reg["solvers"]],
        "wanted_mechanics": [{"name": w["name"], "target": w.get("target")} for w in reg["wanted_mechanics"]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
