"""P2b — reach an L2 win via LEGAL moves, guided by the exact droplet CA. The grid search
proved a clean win exists in bar-CONFIG space; here we hill-climb over MOVE-GENOMES (real
env block moves, so reachable/legit — no answer injection), scoring each with the exact
droplet CA (dropletsim, 100% env agreement) on the flood-severity gradient (n_filled up,
n_wall down). A droplet-win genome IS an env win; we env-verify it.

    python3 p2_droplet_solve.py --restarts 200 --steps 400 --out fixtures/p2_droplet_solve.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB
from imagine_search_gate import _present, GAME_ID  # noqa: E402
from imag_search_solve import frame_of, real_win, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from dropletsim import droplet_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--max-genes", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, 2)
    comps = frame_components(c0)[:8]
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in comps]
    blocks = [i for i, c in enumerate(comps) if int(c["color"]) in (8, 9)]
    rng = np.random.default_rng(a.seed)
    cache = {}

    def ev(genome):
        g = tuple(genome)
        if g in cache:
            return cache[g]
        cf = frame_of(arc, a.seed, 2, list(genome), centers)
        r = None if cf is None else droplet_pour(cf)
        cache[g] = r
        return r

    def fit(r):
        if r is None:
            return (-1, -1, -99)
        return (int(r["win"]), r["n_filled"], -r["n_wall"])

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
            print(f"[solve] restart {r+1}: best fit={bf} evals={evals}", flush=True)

    chosen = win_g or best
    rc = ev(chosen)
    env_ok = real_win(arc, a.seed, 2, list(chosen), centers) if chosen else False
    report = {"evals": evals, "droplet": rc, "genome": [list(x) for x in chosen] if chosen else None,
              "droplet_win": bool(rc and rc["win"]), "env_win": bool(env_ok),
              "verdict": ("L2 SOLVED — env-confirmed" if env_ok else
                          ("droplet-win but env-REJECTED (fidelity gap)" if rc and rc["win"] else
                           "no win genome found this run"))}
    print(json.dumps(report, indent=1))
    if a.out:
        from pathlib import Path
        out_p = Path(a.out)
        if not out_p.is_absolute():
            out_p = FIXTURES / out_p.name
        out_p.write_text(json.dumps(report, indent=1))
        print(f"[solve] wrote {out_p}")


if __name__ == "__main__":
    main()
