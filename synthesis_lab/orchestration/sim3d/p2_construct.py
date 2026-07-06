"""P2 — mechanic-aware CONSTRUCTIVE route sweep for L2 (roadmap Stage C, upgraded).

Not blind mutation (known-insufficient, doc §2): a SYSTEMATIC enumeration over movable-bar
placements, seeded implicitly by the default (which already fills T0+T1), scored by the
POST-P0 fill-faithful sim. Goal: find a CLEAN (no-flood) all-3-fill arrangement (sim margin>0),
or produce the exhaustive-ish evidence for the pre-registered decision-branch (§2 P2): if the
faithful sim finds NO margin>0 arrangement, that is evidence L2 is not arrange-and-pour solvable
-> STOP + escalate the intended-mechanic question.

Enumerates: all single-block moves + all two-block move combinations over a mechanic-guided
move set (per source->mouth routing), then a short 3-block hill-climb from the best. Pours the
top clean-multi-fill candidates for real (<=4 budget). EVAL-ONLY real level; env grades.

    python3 p2_construct.py --level 2 --out fixtures/p2_l2_construct_result.json
"""
from __future__ import annotations

import argparse
import json
import itertools

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID  # noqa: E402
from imag_search_solve import frame_of, real_win, actions  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from pour3d import imagine_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

DIRS = (1, 2, 3, 4)   # up/down/left/right
MAGS = (1, 2, 3, 4, 5, 6)


def score(arc, seed, level, genome, centers):
    cf = frame_of(arc, seed, level, genome, centers)
    if cf is None:
        return None
    out = imagine_pour(cf, lane="stream")
    ts = out["labels"]["targets"]
    fills = [t["fill"] for t in ts]; lavs = [t["lavan"] for t in ts]
    nclean = sum(1 for f, l in zip(fills, lavs) if f >= 0.85 and l < 0.5)
    return {"margin": float(out["margin"]), "winner_ok": bool(out["winner_ok"]),
            "n_clean": nclean, "fills": [round(f, 2) for f in fills],
            "lavs": [round(l, 2) for l in lavs],
            "sum_clean": float(sum(max(0.0, f - l) for f, l in zip(fills, lavs))),
            "flood": float(sum(lavs))}


def key(s):   # rank: clean fills, then margin, then least flood, then sum-clean
    return (s["n_clean"], s["margin"], -s["flood"], s["sum_clean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--topk-real", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, a.level)
    comps = frame_components(c0)[:8]
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in comps]
    blocks = [i for i, c in enumerate(comps) if int(c["color"]) in (8, 9)]

    evals = 0
    results = {}   # genome-tuple -> score

    def ev(genome):
        nonlocal evals
        g = tuple(genome)
        if g in results:
            return results[g]
        s = score(arc, a.seed, a.level, list(genome), centers)
        evals += 1
        if s is not None:
            results[g] = s
        return s

    # default (empty) + all single moves
    ev(())
    single = []
    for ci in blocks:
        for d in DIRS:
            for m in MAGS:
                if ev(((ci, d, m),)) is not None:
                    single.append((ci, d, m))
    # all two-block combinations (distinct blocks)
    for ci, cj in itertools.combinations(blocks, 2):
        for di in DIRS:
            for mi in MAGS:
                for dj in DIRS:
                    for mj in MAGS:
                        ev(((ci, di, mi), (cj, dj, mj)))
    print(f"[p2] {evals} imagined evals over single+two-block sweep", flush=True)

    # short 3-block hill-climb from the best two-block genomes
    ranked = sorted(results.items(), key=lambda kv: key(kv[1]), reverse=True)
    for base_g, _s in ranked[:12]:
        for ci in blocks:
            if any(gene[0] == ci for gene in base_g):
                continue
            for d in DIRS:
                for m in MAGS:
                    ev(tuple(base_g) + ((ci, d, m),))
    print(f"[p2] {evals} imagined evals total (with 3-block hill-climb)", flush=True)

    ranked = sorted(results.items(), key=lambda kv: key(kv[1]), reverse=True)
    best = ranked[0][1]
    # any clean multi-fill in the SIM?
    clean_multi = [(g, s) for g, s in ranked if s["n_clean"] >= 2 and s["margin"] > 0]
    margin_pos = [(g, s) for g, s in ranked if s["margin"] > 0]

    # pour the top-K by rank for real (env grades) — prefer margin>0, else best-ranked
    pour_from = margin_pos if margin_pos else ranked
    real_results, solved_at = [], None
    for rank, (g, s) in enumerate(pour_from[:a.topk_real]):
        won = real_win(arc, a.seed, a.level, list(g), centers)
        real_results.append({"rank": rank + 1, "genome": [list(x) for x in g],
                             "sim": s, "env_win": bool(won)})
        if won:
            solved_at = rank + 1; break

    report = {
        "level": a.level, "imagined_evals": evals,
        "best_sim": best,
        "n_margin_positive": len(margin_pos),
        "n_clean_multifill_sim": len(clean_multi),
        "solved_at": solved_at,
        "real_pours": real_results,
        "decision_branch": (
            "SOLVED" if solved_at else
            "NEGATIVE: no sim margin>0 arrangement over the constructive sweep -> evidence L2 is "
            "not arrange-and-pour solvable (every reachable pour floods); escalate intended-mechanic"
            if len(margin_pos) == 0 else
            "SIM-POSITIVE-BUT-ENV-LOSS: sim found margin>0 but env rejected top-K (sim FP / precision)"),
    }
    print(json.dumps({k: v for k, v in report.items() if k != "real_pours"}, indent=1))
    for r in real_results:
        print(f"  real pour rank{r['rank']}: sim={r['sim']['fills']}/lav{r['sim']['lavs']} "
              f"margin={r['sim']['margin']} -> env_win={r['env_win']}")
    if a.out:
        from pathlib import Path
        out_p = Path(a.out)
        if not out_p.is_absolute():
            out_p = FIXTURES / out_p.name
        out_p.write_text(json.dumps(report, indent=1))
        print(f"[p2] wrote {out_p}")


if __name__ == "__main__":
    main()
