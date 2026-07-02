#!/usr/bin/env python3
"""CONTRA -- the primary cockpit.  The one tool you invoke: "use contra".

    python codexlab/contra.py <command> [args...]      (also: python -m contra <command>)

Commands:
  check    Verify the WHOLE system is attached: Oracle/Ghost -> CONTRA -> scorer
           -> harness -> reporting.  The flagship -- run this and if any edge has
           detached it fails loud and NAMES the broken edge (no more hand re-attaching).
  serve    Launch the informer-DAG HTTP API (contra_service).      [--port ...]
  run      Run the SP80 episode pipeline (sp80_run_contra_levels). [--levels 0,1 ...]
  report   Build a gameplay report (gameplay_report_harness).      [--run-dir ... --out-dir ...]
  gate     Run the full local green gate (dev_check).
  loop     CONTRA Loop Harness engine (9-step GROUND->..->ADMIT protocol, 5-layer gate).
           Subcommands: iterate|baseline|gate|admit|status.
           e.g. loop iterate --target sp80:1 --dry-run   (proves the spine; no admission)
           Accumulates ONLY artifact-proven solvers (levels_completed_delta>0); the
           registry RE-VERIFIES every record on read. Point an agent at the canonical
           training doc: codexlab/contra_loop/CONTRA_LOOP.md.
  view     Run CONTRA's controller view on one raw frame and print a summary.

Subcomponents (NOT the headline -- reach them through this cockpit):
  contra_head/      controller-head internals (perception, scorer building blocks)
  contra_service/   the informer-DAG HTTP API
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

GAME_ID = "sp80-589a99af"
SMOKE_FRAME = CODEXLAB / "runs" / "sp80_sam_color_components" / "frames" / "level_00_raw.npy"
CALIBRATION = CODEXLAB / "sp80_surface_calibration.json"
HARNESS_SRC = CODEXLAB / "sp80_player_harness.py"
REPORT_SRC = CODEXLAB / "gameplay_report_harness.py"


def _delegate_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(CODEXLAB) if not existing else f"{CODEXLAB}{os.pathsep}{existing}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _delegate(rel_script: str, rest: list[str]) -> int:
    return subprocess.run(
        [sys.executable, "-B", f"codexlab/{rel_script}", *rest], cwd=ROOT, env=_delegate_env()
    ).returncode


# ===========================================================================
# check -- the integration contract IS the cockpit's core (the thing that holds
# the system together).  Exercises the whole informer DAG on a deterministic
# smoke frame and fails loud naming any detached edge.
#
#   Oracle ─┐
#           ├─► CONTRA (run_contra_view) ─► scorer (recommendation) ─► harness ─► Ghost memory
#   Ghost  ─┘                                                                       │
#           └──────────────────────── load_ghost_context ◄──────────────────────────┘
# ===========================================================================
def _load_calibration(level_index: int) -> dict[str, Any] | None:
    if not CALIBRATION.exists():
        return None
    payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    for row in payload.get("levels", []) if isinstance(payload, dict) else []:
        if isinstance(row, dict) and int(row.get("level_index", -1)) == int(level_index):
            return row
    return None


def _check_live_edges() -> list[str]:
    """Run the real CONTRA path on the smoke frame; assert each DAG edge is live."""
    import numpy as np  # noqa: PLC0415

    import contra_head  # noqa: PLC0415
    from sp80_player_harness import build_contra_recommendation_packet  # noqa: PLC0415

    failures: list[str] = []

    def edge(ok: bool, name: str, detail: str) -> None:
        if not ok:
            failures.append(f"DETACHED [{name}]: {detail}")

    frame = np.load(SMOKE_FRAME, allow_pickle=False)
    response = contra_head.run_contra_view(
        raw_frame=frame, game_id=GAME_ID, level_index=0, calibration=_load_calibration(0), mode="review"
    )

    # Oracle -> CONTRA.  Importlib silent-detach shows up here as 0 hypotheses.
    oracle_pass = response.get("oracle_pass") or {}
    edge(
        oracle_pass.get("evidence_role") == "review_hypothesis_not_truth_source"
        and int(oracle_pass.get("actor_hypothesis_count") or 0) >= 1,
        "oracle->contra",
        f"run_contra_view produced no Oracle hypotheses on the calibrated smoke frame "
        f"(actor_hypothesis_count={oracle_pass.get('actor_hypothesis_count')!r}); "
        f"check oracle importlib loading in contra_head/controller_affordances.py",
    )

    # Ghost -> CONTRA.  The ghost pass must be assembled (rows may be 0 without context).
    ghost_pass = response.get("ghost_pass") or {}
    edge(
        ghost_pass.get("evidence_role") == "ghost_relationship_memory_not_truth_source",
        "ghost->contra",
        f"run_contra_view did not assemble a ghost_pass (got {ghost_pass.get('evidence_role')!r})",
    )

    # Oracle -> scorer.  The recommendation must actually consume the oracle pass.
    rec = build_contra_recommendation_packet(
        level_index=0, action_index=0, before_state={"selected": True}, goal_condition={},
        contra_packet=response, action_memory=[],
    )
    counts = rec.get("informer_candidate_counts") or {}
    edge(
        int(counts.get("oracle_review") or 0) >= 1,
        "oracle->scorer",
        "build_contra_recommendation_packet ignored oracle_pass "
        "(informer_candidate_counts.oracle_review=0); the scorer no longer reads the Oracle informer",
    )

    # Ghost -> scorer.  Feed a confirmed relation row; the scorer must surface it.
    ghost_probe = build_contra_recommendation_packet(
        level_index=1, action_index=0, before_state={"selected": True}, goal_condition={},
        contra_packet={
            "oracle_pass": oracle_pass,
            "ghost_pass": {
                "evidence_role": "ghost_relationship_memory_not_truth_source",
                "relation_memory_rows": [
                    {"relation_memory_id": "contract_probe", "relation_family": "flow_alignment",
                     "geometry": {"flow_direction": "right"}, "confirmation": {"confidence_score": 1.0}}
                ],
            },
        },
        action_memory=[],
    )
    probe_counts = ghost_probe.get("informer_candidate_counts") or {}
    edge(
        int(probe_counts.get("ghost_relation_memory") or 0) >= 1,
        "ghost->scorer",
        "build_contra_recommendation_packet ignored ghost_pass.relation_memory_rows; "
        "the scorer no longer reads the Ghost informer",
    )

    return failures


def _check_static_edges() -> list[str]:
    """Source tripwires: catch structural detachment (someone ripped out the wiring)."""
    failures: list[str] = []

    def edge(ok: bool, name: str, detail: str) -> None:
        if not ok:
            failures.append(f"DETACHED [{name}]: {detail}")

    harness = HARNESS_SRC.read_text(encoding="utf-8")
    report = REPORT_SRC.read_text(encoding="utf-8")

    edge("run_contra_view(" in harness, "harness->contra_view",
         "sp80_player_harness no longer calls run_contra_view")
    edge("build_contra_recommendation_packet(" in harness, "harness->scorer",
         "sp80_player_harness no longer calls build_contra_recommendation_packet (CONTRA guidance unhooked)")
    edge("ghost_context=ghost_context" in harness, "ghost->live_view",
         "sp80_player_harness no longer threads ghost_context into run_contra_view")
    edge("load_ghost_context(" in harness and "write_ghost(" in harness, "ghost_memory_feedback_loop",
         "sp80_player_harness no longer wires the Ghost feedback loop (load_ghost_context/write_ghost)")
    edge("projections" in report and "_project_contra_pass" in report, "reporting->internal_api",
         "gameplay_report_harness no longer reads CONTRA through the projection contract")

    return failures


def cmd_check(rest: list[str]) -> int:
    failures = _check_live_edges() + _check_static_edges()
    if failures:
        print("FAIL contra check -- a CONTRA attachment has DETACHED:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("contra integration contract passed: Oracle/Ghost -> CONTRA -> scorer -> harness -> reporting all attached")
    return 0


def cmd_view(rest: list[str]) -> int:
    import numpy as np  # noqa: PLC0415

    import contra_head  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="contra view")
    parser.add_argument("--raw-grid", type=Path, default=SMOKE_FRAME)
    parser.add_argument("--game-id", default=GAME_ID)
    parser.add_argument("--level-index", type=int, default=0)
    args = parser.parse_args(rest)
    frame = np.load(args.raw_grid, allow_pickle=False)
    response = contra_head.run_contra_view(
        raw_frame=frame, game_id=str(args.game_id), level_index=int(args.level_index),
        calibration=_load_calibration(int(args.level_index)), mode="review",
    )
    counts = response.get("debug_summary", {}).get("counts", {})
    print(json.dumps({
        "schema": response.get("schema"),
        "admission_status": response.get("admission_status"),
        "oracle_actor_hypotheses": (response.get("oracle_pass") or {}).get("actor_hypothesis_count"),
        "ghost_rows": (response.get("ghost_pass") or {}).get("relation_memory_row_count"),
        "counts": counts,
    }, indent=2, sort_keys=True))
    return 0


COMMANDS = {
    "check": cmd_check,
    "view": cmd_view,
    "serve": lambda rest: _delegate("contra_service/serve.py", rest),
    "run": lambda rest: _delegate("sp80_run_contra_levels.py", rest),
    "report": lambda rest: _delegate("gameplay_report_harness.py", rest),
    "gate": lambda rest: _delegate("dev_check.py", rest),
    "loop": lambda rest: _delegate("contra_loop/loop.py", rest),
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    command, rest = argv[0], argv[1:]
    handler = COMMANDS.get(command)
    if handler is None:
        print(f"contra: unknown command {command!r}\n", file=sys.stderr)
        print(__doc__)
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
