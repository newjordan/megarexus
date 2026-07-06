"""Solve L3 with the EXACT native droplet CA (native_scene + pour_core, 100/100 env agreement).
Hill-climb move-genomes (real env moves = reachable/legit), score by the native CA on the
flood-severity gradient (n_filled up, n_wall down). A native-win IS an env win; env-verify it.

    python3 l3_solve.py --restarts 300 --steps 500 --out fixtures/l3_solve.json
"""
from __future__ import annotations

import argparse
import json
import numpy as np

import _paths  # noqa: F401
from _paths import LAB, FIXTURES
from imagine_search_gate import _present, _apply_real, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from imag_search_solve import actions, real_win, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from native_scene import native_grid  # noqa: E402
from dropletsim import pour_core_v  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=300)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--max-genes", type=int, default=14)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, a.level)
    comps = frame_components(c0)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in comps[:10]]
    blocks = [i for i, c in enumerate(comps[:10]) if int(c["color"]) in (8, 9)]
    rng = np.random.default_rng(a.seed)
    cache = {}

    def ev(genome):
        g = tuple(genome)
        if g in cache:
            return cache[g]
        e = arc.make(GAME_ID, seed=a.seed); fd, cur = _present(e, a.level)
        fd, cur, _ = _apply_real(e, fd, cur, actions(list(genome), centers))
        r = None
        if game_live(fd):
            ng = native_grid(e._game)
            r = pour_core_v(ng["block"], ng["target"], ng["wall"], ng["redir"], ng["tid"],
                            ng["spouts"], ng["n_targets"])
        cache[g] = r
        return r

    def fit(r):
        return (int(r["win"]), r["n_filled"], -r["n_wall"]) if r else (-1, -1, -99)

    def rand_g():
        k = int(rng.integers(2, a.max_genes + 1))
        return [(int(rng.choice(blocks)), int(rng.choice(MOVES)), int(rng.integers(1, 9)))
                for _ in range(k)]

    def mutate(g):
        g = [list(x) for x in g]; op = int(rng.integers(5))
        if op == 0 and g:
            i = int(rng.integers(len(g))); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g:
            i = int(rng.integers(len(g))); g[i][2] = int(rng.integers(1, 9))
        elif op == 2 and g:
            i = int(rng.integers(len(g))); g[i][0] = int(rng.choice(blocks))
        elif op == 3 and len(g) < a.max_genes:
            g.append([int(rng.choice(blocks)), int(rng.choice(MOVES)), int(rng.integers(1, 9))])
        elif len(g) > 2:
            del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    best, bf, evals, win_g = None, (-1, -1, -99), 0, None
    for r in range(a.restarts):
        cur = rand_g(); cr = ev(cur); evals += 1; cf = fit(cr)
        for _s in range(a.steps):
            cand = mutate(cur); rr = ev(cand); evals += 1
            if fit(rr) > cf:
                cur, cf = cand, fit(rr)
                if rr and rr["win"]:
                    win_g = cur; break
        if cf > bf:
            bf, best = cf, cur
        if win_g is not None:
            break
        if (r + 1) % 20 == 0:
            print(f"[l3] restart {r+1}: best fit={bf} evals={evals}", flush=True)

    chosen = win_g or best
    rc = ev(chosen)
    env_ok = real_win(arc, a.seed, a.level, list(chosen), centers) if chosen else False
    report = {"level": a.level, "evals": evals, "native": rc,
              "genome": [list(x) for x in chosen] if chosen else None,
              "native_win": bool(rc and rc["win"]), "env_win": bool(env_ok),
              "verdict": ("L3 SOLVED — env-confirmed" if env_ok else
                          ("native-win but env-REJECTED (native fidelity gap)" if rc and rc["win"] else
                           "no win this run (best n_filled=%s/%s n_wall=%s)" % (
                               rc["n_filled"] if rc else "?", rc["n_targets"] if rc else "?",
                               rc["n_wall"] if rc else "?")))}
    print(json.dumps(report, indent=1))
    out = a.out or str(FIXTURES / "l3_solve.json")
    from pathlib import Path
    Path(out).write_text(json.dumps(report, indent=1)); print(f"[l3] wrote {out}")


if __name__ == "__main__":
    main()
