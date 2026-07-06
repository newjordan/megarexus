"""P2b — L2 solve with the EXACT droplet CA (dropletsim, 100% env agreement). A droplet-win
IS an env-win; a full-sweep negative is definitive. Search block-move genomes; score by
droplet_pour; env-verify any win. Frames built via the real env (reachable arrangements only).

    python3 p2_droplet_search.py --level 2 --out fixtures/p2_droplet_l2_result.json
"""
from __future__ import annotations

import argparse
import json
import itertools

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID  # noqa: E402
from imag_search_solve import frame_of, real_win  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from dropletsim import droplet_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

DIRS = (1, 2, 3, 4)
MAGS = (1, 2, 3, 4, 5, 6, 7, 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--hill", type=int, default=40, help="top partial genomes to extend")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, a.level)
    comps = frame_components(c0)[:8]
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in comps]
    blocks = [i for i, c in enumerate(comps) if int(c["color"]) in (8, 9)]

    scored = {}
    builds = 0

    def ev(genome):
        nonlocal builds
        g = tuple(genome)
        if g in scored:
            return scored[g]
        cf = frame_of(arc, a.seed, a.level, list(genome), centers)
        builds += 1
        if cf is None:
            scored[g] = None
            return None
        d = droplet_pour(cf)
        scored[g] = (d["win"], d["n_filled"], not d["fail"], d)
        return scored[g]

    def rank(v):   # win, then #filled, then no-fail
        return (v[0], v[1], v[2]) if v else (False, -1, False)

    ev(())                                              # default
    for ci in blocks:                                   # all single moves
        for di in DIRS:
            for mi in MAGS:
                ev(((ci, di, mi),))
    for ci, cj in itertools.combinations(blocks, 2):    # all two-block moves
        for di in DIRS:
            for mi in MAGS:
                for dj in DIRS:
                    for mj in MAGS:
                        ev(((ci, di, mi), (cj, dj, mj)))
    print(f"[p2b] {builds} frames after single+two-block; "
          f"best n_filled={max((v[1] for v in scored.values() if v), default=-1)}", flush=True)

    # hill-climb: extend the best partial genomes with a 3rd (and 4th) block move
    for _round in range(2):
        best = sorted((kv for kv in scored.items() if kv[1]), key=lambda kv: rank(kv[1]), reverse=True)
        if best and best[0][1][0]:
            break
        for base_g, _v in best[:a.hill]:
            used = {gene[0] for gene in base_g}
            for ci in blocks:
                if ci in used:
                    continue
                for di in DIRS:
                    for mi in MAGS:
                        ev(tuple(base_g) + ((ci, di, mi),))
        print(f"[p2b] {builds} frames after hill round {_round}; "
              f"best n_filled={max((v[1] for v in scored.values() if v), default=-1)}", flush=True)

    ranked = sorted((kv for kv in scored.items() if kv[1]), key=lambda kv: rank(kv[1]), reverse=True)
    wins = [(g, v) for g, v in ranked if v[0]]
    solved_at, verified = None, []
    for g, v in (wins[:8] if wins else ranked[:4]):
        won = real_win(arc, a.seed, a.level, list(g), centers)
        verified.append({"genome": [list(x) for x in g], "droplet": v[3]["filled"],
                         "droplet_win": v[0], "env_win": bool(won)})
        if won and solved_at is None:
            solved_at = [list(x) for x in g]
    best = ranked[0][1][3] if ranked else None
    report = {
        "level": a.level, "frames_built": builds,
        "n_droplet_wins": len(wins),
        "best_droplet": best,
        "SOLVED_genome": solved_at,
        "verified": verified,
        "verdict": ("L2 SOLVED (env-confirmed)" if solved_at else
                    ("droplet found win but env rejected (fidelity gap!)" if wins else
                     "no win in reachable single/two/three-block space (EXACT sim negative)")),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "verified"}, indent=1))
    for r in verified:
        print(f"  verify: droplet_filled={r['droplet']} droplet_win={r['droplet_win']} -> env_win={r['env_win']}")
    if a.out:
        from pathlib import Path
        out_p = Path(a.out)
        if not out_p.is_absolute():
            out_p = FIXTURES / out_p.name
        out_p.write_text(json.dumps(report, indent=1))
        print(f"[p2b] wrote {out_p}")


if __name__ == "__main__":
    main()
