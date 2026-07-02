#!/usr/bin/env python3
"""CONTRA Loop Harness driver -- one iteration of attempt -> verify, with DISCOVER/CRYSTALLIZE
extension points for the looped agent. See codexlab/contra_loop/CONTRA_LOOP.md.

    python codexlab/contra.py loop --target sp80:1 --dry-run    # plan only, run nothing
    python codexlab/contra.py loop --target sp80:1              # one real ATTEMPT + VERIFY
    python codexlab/contra.py loop --list-solvers
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "codexlab"))

from contra_loop.solver_registry import load_registry, solvers_for_target  # noqa: E402

ADAPTERS = {"sp80": {"runner": "codexlab/sp80_run_contra_levels.py", "verifier": "codexlab/sp80_verify_episode_run.py"}}


def parse_target(target: str) -> tuple[str, int]:
    game, _, level = target.partition(":")
    if game not in ADAPTERS:
        raise SystemExit(f"unknown game adapter {game!r}; known: {sorted(ADAPTERS)}")
    return game, int(level or 0)


def attempt(game: str, level: int, *, backend: str, max_actions: int) -> dict[str, Any]:
    """ATTEMPT stage: run CONTRA on the target level via the real episode runner (subprocess)."""
    cmd = [sys.executable, "-B", ADAPTERS[game]["runner"], "--backend", backend,
           "--levels", str(level), "--max-actions", str(max_actions), "--skip-report"]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    run_dir = None
    # The runner prints run_dir inside a pretty-printed (multi-line) JSON block, so scan the
    # whole stdout rather than per-line. Prefer the run root (episode_levels), not a level_NN dir.
    matches = re.findall(r'"run_dir":\s*"([^"]+)"', proc.stdout)
    for m in matches:
        if "/level_" not in m:
            run_dir = m
    run_dir = run_dir or (matches[-1] if matches else None)
    return {"cmd": " ".join(cmd), "returncode": proc.returncode, "run_dir": run_dir,
            "stdout_tail": proc.stdout[-1500:], "stderr_tail": proc.stderr[-800:]}


def verify(run_dir: str, level: int) -> dict[str, Any]:
    """VERIFY stage: solve is real ONLY if levels_completed_delta>0 on the target level."""
    summary = ROOT / run_dir / "summary.json"
    if not summary.is_file():
        return {"solved": False, "reason": "no summary.json", "delta": 0}
    rows = json.loads(summary.read_text(encoding="utf-8")).get("rows", [])
    row = next((r for r in rows if int(r.get("level_index", -1)) == level), None)
    if row is None:
        return {"solved": False, "reason": f"no row for level {level}", "delta": 0}
    delta = int(row.get("levels_completed_delta") or 0)
    return {"solved": delta > 0, "delta": delta, "episode_status": row.get("episode_status"),
            "verifier_returncode": row.get("verifier_returncode"), "row_status": row.get("status")}


def run_iteration(target: str, *, backend: str, max_actions: int, dry_run: bool) -> int:
    game, level = parse_target(target)
    active = solvers_for_target(target)
    wanted = [w for w in load_registry()["wanted_mechanics"] if w.get("target") == target]

    print(f"== CONTRA LOOP :: target {target} ==")
    print(f"  active solvers ({len(active)}): " + ", ".join(s["name"] for s in active) or "  (none)")
    if wanted:
        print(f"  wanted mechanic for this target: {wanted[0]['name']} -- {wanted[0]['want']}")

    if dry_run:
        print("\n  [DRY-RUN] plan (nothing executed):")
        print("   1. ATTEMPT  -> run episode on the target with active solvers")
        print("   2. VERIFY   -> solved iff levels_completed_delta>0 (verifier rc 0)")
        print("   3. DISCOVER -> [agent] if unsolved, form ONE mechanic hypothesis from frame deltas")
        print("   4. CRYSTALLIZE -> [agent] encode hypothesis as an algorithm pointer / candidate solver")
        print("   5. register_solver() admits it ONLY with a verified provenance run_dir")
        return 0

    print("\n  ATTEMPT ...")
    a = attempt(game, level, backend=backend, max_actions=max_actions)
    if not a["run_dir"]:
        print(f"  ATTEMPT failed to produce a run_dir (rc={a['returncode']})\n{a['stderr_tail']}")
        return 1
    print(f"  run_dir: {a['run_dir']}")

    v = verify(a["run_dir"], level)
    print("\n  VERIFY:")
    print(json.dumps(v, indent=2, sort_keys=True))

    if v["solved"]:
        print(f"\n  >>> SOLVED target {target} (delta={v['delta']}). "
              f"CRYSTALLIZE the winning trace and call register_solver() with this run_dir as provenance.")
    else:
        nxt = wanted[0]["name"] if wanted else "form a new mechanic hypothesis"
        print(f"\n  >>> NOT SOLVED (delta={v['delta']}, episode_status={v.get('episode_status')}). "
              f"row status is '{v.get('row_status')}' (nominal != solved). "
              f"\n      DISCOVER stage [agent]: target the wanted mechanic -> {nxt}")
        print("      (DISCOVER + CRYSTALLIZE are the agent's job this iteration -- see CONTRA_LOOP.md)")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="sp80:1", help="<game>:<level>, e.g. sp80:1 (the assigned target)")
    p.add_argument("--dry-run", action="store_true", help="print the loop plan; execute nothing")
    p.add_argument("--list-solvers", action="store_true", help="print the solver library and exit")
    p.add_argument("--backend", default="tiny_mlp")
    p.add_argument("--max-actions", type=int, default=20)
    args = p.parse_args()

    if args.list_solvers:
        reg = load_registry()
        print(json.dumps({"solver_count": len(reg["solvers"]),
                          "solvers": [s["name"] for s in reg["solvers"]],
                          "wanted_mechanics": [w["name"] for w in reg["wanted_mechanics"]]}, indent=2))
        return 0
    return run_iteration(args.target, backend=args.backend, max_actions=args.max_actions, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
