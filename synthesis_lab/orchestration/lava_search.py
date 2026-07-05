"""Hill-climb an SP80 arrangement on the "floor is lava" objective (NIGHT FINDING 26-27).

The win = fill ALL targets (positive) AND avoid walls/lava (negative). The negative
signal (wall-color-14 growth above the pre-pour baseline = actual wall HITS) perfectly
discriminates L0 wins (AUC 1.0) and gives a CLIMBABLE gradient on L1 where positive-only
was flat. So search arrangement-space by mutating block placements, pouring for REAL, and
climbing fitness = sustained_fill - lambda * lava_contact. Goal: the FIRST honest L1 win
(env levels_completed), which also bootstraps L1 positives for the shape-based imagination.

Legal: arrangements are DISCOVERED via env feedback (fill + lava from real pours), not
injected; no route lineage, no answer. The env grades every pour.

    python3 synthesis_lab/orchestration/lava_search.py --level 1 --budget 3000 --lam 0.004
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

_LAB = Path(__file__).resolve().parents[1]; _REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))
from imagine_search_gate import (_present, _apply_real, _grid,
                                 GAME_ID, ACTION_OF_ID, game_live)  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

MOVES = (1, 2, 3, 4)


def sustained_fill(frames, ntarg, K=3):
    n13 = [int((x == 13).sum()) for x in frames]
    b = 0
    for i in range(len(n13) - K + 1):
        b = max(b, min(n13[i:i + K]))
    return b / max(ntarg, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=0.004)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--max-genes", type=int, default=6)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(GAME_ID, seed=a.seed)
    _, c0 = _present(env, a.level)
    ntarg = int((c0 == 11).sum())
    base_wall = int((c0 == 14).sum())
    comps = frame_components(c0)[:8]
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in comps]
    print(f"[lava] L{a.level}: {ntarg} targets, {len(centers)} movable comps, base_wall={base_wall}", flush=True)
    rng = np.random.default_rng(a.seed)

    def genome_to_actions(genome):
        acts = []
        for (ci, direction, mag) in genome:
            if ci < len(centers):
                cx, cy = centers[ci]
                acts.append((6, cx, cy))
                acts += [(int(direction), 0, 0)] * int(mag)
        return acts

    evals = {"n": 0}

    def evaluate(genome):
        evals["n"] += 1
        er = arc.make(GAME_ID, seed=a.seed)
        fdr, cur = _present(er, a.level)
        base = int(getattr(fdr, "levels_completed", 0) or 0)
        fdr, cur, _ = _apply_real(er, fdr, cur, genome_to_actions(genome))
        if not game_live(fdr):
            return -1e9, 0.0, 999, False
        fdr = er.step(ACTION_OF_ID[5])
        won = int(getattr(fdr, "levels_completed", 0) or 0) > base
        f = fdr.frame
        frames = [np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list) \
            else [np.asarray(f, dtype=np.int8)]
        fill = sustained_fill(frames, ntarg)
        lava = max(int((x == 14).sum()) for x in frames) - base_wall
        fit = fill - a.lam * lava
        return fit, fill, lava, won

    def rand_genome():
        k = int(rng.integers(1, a.max_genes + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4)))
                for _ in range(k)]

    def mutate(genome):
        g = [list(x) for x in genome]
        op = rng.integers(4)
        if op == 0 and g:            # change a gene's direction
            i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g:          # change a gene's magnitude
            i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
        elif op == 2 and len(g) < a.max_genes:  # add a gene
            g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))])
        elif g and len(g) > 1:       # remove a gene
            del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    global_best = (-1e9, None, 0.0, 999)
    win_found = None
    for r in range(a.restarts):
        if evals["n"] >= a.budget or win_found:
            break
        cur = rand_genome()
        cfit, cfill, clava, cwon = evaluate(cur)
        if cwon:
            win_found = cur
        steps_no_improve = 0
        while evals["n"] < a.budget and not win_found and steps_no_improve < 60:
            cand = mutate(cur)
            fit, fill, lava, won = evaluate(cand)
            if won:
                win_found = cand
                global_best = (fit, cand, fill, lava)
                break
            if fit > cfit:
                cur, cfit, cfill, clava = cand, fit, fill, lava
                steps_no_improve = 0
            else:
                steps_no_improve += 1
            if cfit > global_best[0]:
                global_best = (cfit, cur, cfill, clava)
        print(f"[lava] restart {r}: evals={evals['n']} best_fit={global_best[0]:.3f} "
              f"best_fill={global_best[2]:.2f} best_lava={global_best[3]}", flush=True)

    if win_found is not None:
        fit, fill, lava, won = evaluate(win_found)
        print(f"[lava] *** L{a.level} WIN FOUND *** genes={win_found} fill={fill:.2f} lava={lava} "
              f"env_win={won} evals={evals['n']}", flush=True)
    else:
        print(f"[lava] no win in {evals['n']} evals. best_fit={global_best[0]:.3f} "
              f"best_fill={global_best[2]:.2f} best_lava={global_best[3]} genes={global_best[1]}", flush=True)


if __name__ == "__main__":
    main()
