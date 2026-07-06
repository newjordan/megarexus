"""SIM3D stage S3 — in-budget solve attempt, simulator-guided (no learned scorer).

EXACTLY the imag_search_solve protocol (pre-registered, honest): multi-restart
hill-climb over arrangement genomes; candidate frames are realized with clean env
steps (NO pours); every pour during search is IMAGINED by the sim3d stream lane;
the env grades only the final <=4 real confirmation pours (the game budget).
No answer/route injection, no learned weights anywhere in the loop.

    python3 s3_solve.py --levels 1 --restarts 24 --steps 60

Fitness = decisive margin  min_t(fill_t - lavan_t)  from pertarget_labels on the
SIMULATED rollout, plus a small native-fill shaping term to give the hill-climb a
gradient inside equal-margin plateaus. A candidate is poured for real only if it
is among the top-K distinct arrangements at the end.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import LAB, PKG

from imagine_search_gate import _present, GAME_ID  # noqa: E402
from imag_search_solve import frame_of, real_win, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

from pour3d import imagine_pour


def sim_fitness(frame, lane, emit_volume):
    out = imagine_pour(frame, lane=lane, emit_volume=emit_volume)
    shaping = 0.1 * float(np.mean(out["native"]["mouth_fill"])) if out["native"]["mouth_fill"] else 0.0
    shaping -= 0.05 * float(out["native"]["lava"])
    return out["margin"] + shaping, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--topk", type=int, default=4)        # the game budget
    ap.add_argument("--max-genes", type=int, default=8)
    ap.add_argument("--lane", default="stream")
    ap.add_argument("--emit-volume", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    report = {}
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        env0 = arc.make(GAME_ID, seed=a.seed)
        _, c0 = _present(env0, level)
        centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
                   for c in frame_components(c0)[:8]]
        rng = np.random.default_rng(a.seed + level)

        def rand_g():
            k = int(rng.integers(1, a.max_genes + 1))
            return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                     int(rng.integers(1, 4))) for _ in range(k)]

        def mutate(g):
            g = [list(x) for x in g]
            op = rng.integers(4)
            if op == 0 and g:
                i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
            elif op == 1 and g:
                i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
            elif op == 2 and len(g) < a.max_genes:
                g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                          int(rng.integers(1, 4))])
            elif g and len(g) > 1:
                del g[int(rng.integers(len(g)))]
            return [tuple(x) for x in g]

        found = []          # (fitness, margin, genome, frame_hash, restart)
        evals = 0
        for r in range(a.restarts):
            cur = rand_g()
            cf = frame_of(arc, a.seed, level, cur, centers)
            if cf is None:
                continue
            fit, out = sim_fitness(cf, a.lane, a.emit_volume)
            evals += 1
            best = (fit, out["margin"], cur, hash(cf.tobytes()), r)
            for _ in range(a.steps):
                cand = mutate(best[2])
                cff = frame_of(arc, a.seed, level, cand, centers)
                if cff is None:
                    continue
                f2, o2 = sim_fitness(cff, a.lane, a.emit_volume)
                evals += 1
                if f2 > best[0]:
                    best = (f2, o2["margin"], cand, hash(cff.tobytes()), r)
            found.append(best)
            print(f"[s3] L{level} restart {r}: fitness={best[0]:+.3f} "
                  f"margin={best[1]:+.3f}", flush=True)

        found.sort(key=lambda x: -x[0])
        seen, topk = set(), []
        for fit, margin, g, fh, r in found:
            if fh in seen:
                continue
            seen.add(fh)
            topk.append((fit, margin, g))
            if len(topk) >= a.topk:
                break

        solved_at, poured = None, []
        for rank, (fit, margin, g) in enumerate(topk):
            won = real_win(arc, a.seed, level, g, centers)
            poured.append({"rank": rank + 1, "fitness": fit, "sim_margin": margin,
                           "genome": [list(x) for x in g], "env_win": bool(won)})
            if won:
                solved_at = rank + 1
                break
        line = (f"[s3] L{level}: {'SOLVED@' + str(solved_at) if solved_at else 'NOT solved'} "
                f"in-budget (<= {a.topk} real pours) | sim margins "
                f"{[round(m, 3) for _, m, _ in topk]} | {evals} imagined evals, 0 search pours")
        print(line, flush=True)
        report[f"L{level}"] = {"solved_at": solved_at, "topk": poured, "evals": evals,
                               "protocol": "imag_search_solve; env pours only for final topk"}

    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(report, indent=1))
        print(f"[s3] wrote {a.out}")


if __name__ == "__main__":
    main()
