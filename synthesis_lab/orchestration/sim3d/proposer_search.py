"""Stage-C proposer v1 — sim-guided search with a SUM-of-clean-fills fitness (stronger
gradient than s3_solve's min-over-targets margin), on the now-correct multi-source sim.

Rationale: L2's winner needs ALL sources routed to ALL targets with no overpour. min-margin
is dominated by the worst target (flat until near a full win); summing per-target clean fill
gives a gradient that rewards feeding each additional target, so the hill-climb can build up
to a full clean win. Protocol = s3_solve's (all search IMAGINED; env grades only final top-K
≤ budget). Honest: no answer/route injection, no learned weights.

    python3 proposer_search.py --levels 2 --restarts 64 --steps 160
"""
from __future__ import annotations
import argparse, json
import numpy as np
import _paths  # noqa: F401
from _paths import LAB
from imagine_search_gate import _present, GAME_ID  # noqa: E402
from imag_search_solve import frame_of, real_win, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from pour3d import imagine_pour  # noqa: E402


def sum_fitness(frame, emit_volume):
    """Sum of per-target clean fill (fill_t - lavan_t, floored at 0) + full-win bonus."""
    out = imagine_pour(frame, lane="stream", emit_volume=emit_volume)
    ts = out["labels"]["targets"]
    clean = sum(max(0.0, t["fill"] - t["lavan"]) for t in ts)
    nfull = sum(1 for t in ts if t["fill"] >= 0.85 and t["lavan"] < 0.5)
    win_bonus = 5.0 if (ts and all(t["fill"] >= 0.85 and t["lavan"] < 0.5 for t in ts)) else 0.0
    # tiny native-fill shaping to break plateaus
    shape = 0.05 * float(np.mean(out["native"]["mouth_fill"])) if out["native"]["mouth_fill"] else 0.0
    return clean + 0.5 * nfull + win_bonus + shape, out["margin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="2")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=64)
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max-genes", type=int, default=10)
    ap.add_argument("--emit-volume", type=float, default=37.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    report = {}
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, level)
        centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
                   for c in frame_components(c0)[:8]]
        rng = np.random.default_rng(a.seed + level)

        def rand_g():
            k = int(rng.integers(1, a.max_genes + 1))
            return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                     int(rng.integers(1, 4))) for _ in range(k)]

        def mutate(g):
            g = [list(x) for x in g]; op = rng.integers(4)
            if op == 0 and g:
                i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
            elif op == 1 and g:
                i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
            elif op == 2 and len(g) < a.max_genes:
                g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))])
            elif g and len(g) > 1:
                del g[int(rng.integers(len(g)))]
            return [tuple(x) for x in g]

        found, evals, best_clean = [], 0, -1.0
        for r in range(a.restarts):
            cur = rand_g(); cf = frame_of(arc, a.seed, level, cur, centers)
            if cf is None:
                continue
            fit, margin = sum_fitness(cf, a.emit_volume); evals += 1
            best = (fit, margin, cur, hash(cf.tobytes()))
            for _ in range(a.steps):
                cand = mutate(best[2]); cff = frame_of(arc, a.seed, level, cand, centers)
                if cff is None:
                    continue
                f2, m2 = sum_fitness(cff, a.emit_volume); evals += 1
                if f2 > best[0]:
                    best = (f2, m2, cand, hash(cff.tobytes()))
            found.append(best); best_clean = max(best_clean, best[0])
            print(f"[prop] L{level} restart {r}: best_fit={best[0]:.2f} margin={best[1]:+.2f}", flush=True)

        found.sort(key=lambda x: -x[0])
        seen, topk = set(), []
        for fit, margin, g, fh in found:
            if fh in seen:
                continue
            seen.add(fh); topk.append((fit, margin, g))
            if len(topk) >= a.topk:
                break
        solved_at, poured = None, []
        for rank, (fit, margin, g) in enumerate(topk):
            won = real_win(arc, a.seed, level, g, centers)
            poured.append({"rank": rank + 1, "fitness": fit, "sim_margin": margin,
                           "genome": [list(x) for x in g], "env_win": bool(won)})
            if won:
                solved_at = rank + 1; break
        print(f"[prop] L{level}: {'SOLVED@'+str(solved_at) if solved_at else 'NOT solved'} "
              f"| top fitnesses {[round(f,2) for f,_,_ in topk]} | {evals} imagined evals", flush=True)
        report[f"L{level}"] = {"solved_at": solved_at, "topk": poured, "evals": evals,
                               "fitness": "sum-clean-fills+win-bonus (multi-source sim)"}
    if a.out:
        from pathlib import Path
        Path(a.out).write_text(json.dumps(report, indent=1)); print(f"[prop] wrote {a.out}")


if __name__ == "__main__":
    main()
